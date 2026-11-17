"""The opt-in compilation scope and pinned full-graph CPU trace contract."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from posteriorforge.study import training_compilation
from posteriorforge.study.model import (
    ModelQualificationError,
    PreparedMember,
    TabPFNBackbone,
    attach_lora,
    canonical_full_support_mixture,
    weights_fingerprint,
)
from posteriorforge.study.training import TrainingConfig, make_optimizer
from posteriorforge.study.training_compilation import compiled_training_mlps
from posteriorforge.study.training_execution import conditional_forward_cache
from posteriorforge.study.training_numerics import compare_named_tensors, trainable_gradient_snapshot


@pytest.fixture
def scope_fixture(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    monkeypatch.setattr(training_compilation, "_qualified_mlps", lambda model: tuple(model.named_children()))
    return model


def test_disabled_has_no_compilation_side_effect(monkeypatch):
    def unexpected(*args, **options):
        pytest.fail("disabled scope attempted to compile a module")

    monkeypatch.setattr(nn.Module, "compile", unexpected)
    model = nn.Linear(2, 1)
    with compiled_training_mlps(model) as metadata:
        assert metadata == {"enabled": False, "policy": "eager", "module_names": []}
        assert "_compiled_call_impl" not in model.__dict__
    with pytest.raises(TypeError, match="explicit boolean"):
        with compiled_training_mlps(model, enabled=1):
            pass


def test_unknown_model_is_rejected():
    with pytest.raises(ModelQualificationError, match="pinned TabPFN backbone"):
        with compiled_training_mlps(nn.Linear(2, 1), enabled=True, backend="eager"):
            pass


def test_cpu_inductor_is_rejected_before_lazy_installation(scope_fixture):
    with pytest.raises(ModelQualificationError, match="on one cuda device"):
        with compiled_training_mlps(scope_fixture, enabled=True):
            pass
    assert all("_compiled_call_impl" not in module.__dict__ for module in scope_fixture)


def test_suppressed_graph_errors_are_rejected(scope_fixture, monkeypatch):
    from torch._dynamo import config

    monkeypatch.setattr(config, "suppress_errors", True)
    with pytest.raises(ModelQualificationError, match="error propagation"):
        with compiled_training_mlps(scope_fixture, enabled=True, backend="eager"):
            pass


def test_installation_restores_preexisting_attributes_on_partial_failure(scope_fixture, monkeypatch):
    first, second = scope_fixture
    first._compiled_call_impl = None

    def compile_call(module, **options):
        assert options == {"backend": "eager", "mode": "default", "fullgraph": True, "dynamic": False}
        module._compiled_call_impl = module._call_impl
        if module is second:
            raise RuntimeError("compiler initialization failed")

    monkeypatch.setattr(nn.Module, "compile", compile_call)
    with pytest.raises(RuntimeError, match="compiler initialization failed"):
        with compiled_training_mlps(scope_fixture, enabled=True, backend="eager"):
            pass
    assert first.__dict__["_compiled_call_impl"] is None
    assert "_compiled_call_impl" not in second.__dict__


def test_nested_scope_preserves_active_compiled_calls(scope_fixture):
    with compiled_training_mlps(scope_fixture, enabled=True, backend="eager") as metadata:
        assert metadata["module_names"] == ["0", "1"]
        assert metadata["fullgraph"] and not metadata["dynamic"]
        calls = [module._compiled_call_impl for module in scope_fixture]
        with pytest.raises(ModelQualificationError, match="cannot nest"):
            with compiled_training_mlps(scope_fixture, enabled=True, backend="eager"):
                pass
        assert [module._compiled_call_impl for module in scope_fixture] == calls
    assert all("_compiled_call_impl" not in module.__dict__ for module in scope_fixture)


class _GraphBreak(nn.Module):
    def forward(self, value):
        torch._dynamo.graph_break()
        return value + 1


def test_unsupported_full_graph_raises_and_restores_scope(scope_fixture):
    from torch._dynamo.exc import Unsupported

    scope_fixture[1] = _GraphBreak()
    with pytest.raises(Unsupported):
        with compiled_training_mlps(scope_fixture, enabled=True, backend="eager"):
            scope_fixture(torch.zeros((2, 3)))
    assert all("_compiled_call_impl" not in module.__dict__ for module in scope_fixture)


@pytest.fixture(scope="module")
def frozen_checkpoint():
    checkpoint = os.environ.get("FTF_CPU_CHECKPOINT")
    if not checkpoint:
        pytest.skip("real cached checkpoint opt-in, with no download")
    return TabPFNBackbone.from_checkpoint(Path(checkpoint))


def _one_conditional(model, members):
    model.zero_grad(set_to_none=True)
    with conditional_forward_cache(model, enabled=True):
        outputs = model(members)
        density = canonical_full_support_mixture(outputs, [model.borders] * 8, model.borders)
        target = torch.tensor([.25], dtype=torch.float64)
        loss = -density.log_prob(target, log_abs_det_decode=.17).sum()
        loss.backward()
    return (
        {str(i): out.detach() for i, out in enumerate(outputs)},
        {"density_weights": density.weights.detach(), "loss": loss.detach()},
        trainable_gradient_snapshot(model),
    )


@pytest.mark.parametrize("rank", [4, 16, None], ids=["lora_r4", "lora_r16", "full_finetune"])
def test_real_checkpoint_compiled_calls_preserve_identity_and_full_tail_training(rank, frozen_checkpoint):
    model = deepcopy(frozen_checkpoint)
    generator = torch.Generator().manual_seed(2026111705)
    if rank is None:
        model.requires_grad_(True)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026111700 + rank)
            model.backbone = attach_lora(model.backbone, rank=rank)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(mean=0., std=.001, generator=generator)
    members = [
        PreparedMember(
            torch.randn(16, width, generator=generator),
            torch.randn(16, generator=generator),
            torch.randn(1, width, generator=generator),
        )
        for width in (3, 3, 3, 3, 2, 2, 2, 2)
    ]
    config = TrainingConfig(method="full_finetune" if rank is None else "lora", rank=rank or 4)
    optimizer = make_optimizer(model, config)
    expected = _one_conditional(model, members)
    original_fingerprint = weights_fingerprint(model)
    keys = tuple(model.state_dict())
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}

    with compiled_training_mlps(model, enabled=True, backend="eager") as metadata:
        assert len(metadata["module_names"]) == 12
        assert all(name.endswith(".mlp") for name in metadata["module_names"])
        assert len(metadata["source_sha256"]) == 64
        json.dumps(metadata, allow_nan=False)
        assert tuple(model.state_dict()) == keys
        assert weights_fingerprint(model) == original_fingerprint
        assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
        assert ({id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
                == {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        actual = _one_conditional(model, members)

    for baseline, compiled in zip(expected[:2], actual[:2]):
        report = compare_named_tensors(baseline, compiled, atol=0., rtol=0.)
        assert report["passed"], json.dumps(report, sort_keys=True)
    report = compare_named_tensors(expected[2], actual[2], atol=2e-7, rtol=2e-5)
    assert report["passed"], json.dumps(report, sort_keys=True)
    assert all(model.get_submodule(name)._compiled_call_impl is None for name in metadata["module_names"])
    assert weights_fingerprint(model) == original_fingerprint
