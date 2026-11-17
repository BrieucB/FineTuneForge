"""Independent packed-attention/PEFT cache parity; no GPU or training campaign."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from posteriorforge.study.model import CANONICAL_FULL_SUPPORT, attach_lora, canonical_full_support_mixture
from posteriorforge.study.training import (TrainingConfig, TrainingState,
    load_training_checkpoint, make_optimizer, save_training_checkpoint)
from posteriorforge.study.training_numerics import compare_named_tensors, trainable_gradient_snapshot

# Declared before comparisons. Cached reuse changes FP32 gradient summation order;
# forward logits/density retain identical operations and should be bitwise equal.
GRADIENT_ATOL = 2e-7
GRADIENT_RTOL = 2e-5
STATE_ATOL = 2e-7
STATE_RTOL = 2e-5


def _assert_report(report):
    assert report["passed"], json.dumps(report, sort_keys=True)


def test_comparison_requires_all_names_finite_gradients_shapes_and_dtypes():
    value = torch.tensor([1., 0.], dtype=torch.float32)
    equal = compare_named_tensors({"x": value}, {"x": value.clone()}, atol=0, rtol=0)
    assert equal["passed"] and equal["tensors"]["x"]["exact"]
    assert not compare_named_tensors({}, {}, atol=0, rtol=0)["passed"]
    assert not compare_named_tensors({"x": value}, {}, atol=0, rtol=0)["passed"]
    assert not compare_named_tensors({"x": value}, {"x": value.double()}, atol=0, rtol=0)["passed"]
    assert not compare_named_tensors({"x": None}, {"x": None}, atol=0, rtol=0)["passed"]
    assert not compare_named_tensors({"x": value}, {"x": torch.full_like(value, float("nan"))}, atol=0, rtol=0)["passed"]
    with pytest.raises(ValueError):
        compare_named_tensors({"x": value}, {"x": value}, atol=-1, rtol=0)


class _PackedFixture(nn.Module):
    """Actual TabPFN encoder/checkpoint and packed attention; execution fixture."""
    def __init__(self):
        super().__init__()
        from tabpfn.model.layer import PerFeatureEncoderLayer
        from tabpfn.model.transformer import LayerStack
        self.transformer_encoder = LayerStack(layer_creator=lambda: PerFeatureEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=8, recompute_attn=False,
            device=torch.device("cpu"), dtype=torch.float32), num_layers=2,
            recompute_each_layer=True)
        self.decoder_dict = nn.ModuleDict({"standard": nn.Linear(4, 5, bias=False)})

    def forward(self, value):
        value = self.transformer_encoder(value.unsqueeze(2), single_eval_pos=16,
            cache_trainset_representation=False)
        return self.decoder_dict["standard"](value)[:, -1, 0]


def _model(rank):
    pytest.importorskip("tabpfn")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202611174)
        model = _PackedFixture().eval()
        if rank is not None:
            pytest.importorskip("peft")
            model.requires_grad_(False)
            model = attach_lora(model, rank=rank)
            # Nonzero adapters exercise A and B derivatives, beyond zero-adapter checks.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=.03)
    return model


def _density(model, stage, *, nonce=0):
    generator = torch.Generator().manual_seed(2026111700 + 17 * nonce + stage)
    # Sixteen context tokens plus one query, all eight members, four conditionals.
    inputs = [torch.randn(1, 17, 4, generator=generator) for _ in range(8)]
    canonical = torch.tensor([-4., -2., -.5, .4, 1.8, 4.], dtype=torch.float64)
    sampling = canonical.float() * 1.7
    outputs = [model(value) for value in inputs]
    borders = [torch.tensor([-3.1, -1.5, -.2, .5, 1.7, 3.2], dtype=torch.float64) + .03 * member for member in range(8)]
    density = canonical_full_support_mixture(outputs, borders, canonical, sampling_borders=sampling)
    return outputs, density


def _loss_and_backward(model, *, cached, nonce=0):
    model.zero_grad(set_to_none=True)
    scalar = []
    logits = {}
    weights = {}
    targets = (-5., -.7, 1.1, 5.)  # Both half-normal tails and interior bins.
    from posteriorforge.study.training_execution import conditional_forward_cache
    for stage in range(4):
        # The wrapper scope survives checkpoint recomputation; individual tensor
        # caches still end at each layer return, before the next layer/member.
        with conditional_forward_cache(model, enabled=cached):
            outputs, density = _density(model, stage, nonce=nonce)
            assert density.weights.dtype == torch.float64
            assert density.weights.shape == (1, 5)
            for member, output in enumerate(outputs):
                logits[f"stage{stage}/member{member}"] = output.detach().clone()
            weights[f"stage{stage}"] = density.weights.detach().clone()
            loss = -density.log_prob(torch.tensor([targets[stage]], dtype=torch.float64),
                log_abs_det_decode=torch.tensor([.07 * stage], dtype=torch.float64))
            divisor = 6 if stage < 2 else 3
            contribution = loss.sum() / divisor
            scalar.append(float(contribution.detach()))
            contribution.backward()
    return {"loss": torch.tensor(sum(scalar), dtype=torch.float64)}, logits, weights


def _optimizer_tensors(optimizer):
    return {f"{key}/{name}": value.detach().cpu().clone()
        for key, state in optimizer.state_dict()["state"].items()
        for name, value in state.items() if isinstance(value, torch.Tensor)}


def _one_update(model, optimizer, state, *, cached, nonce):
    loss, logits, weights = _loss_and_backward(model, cached=cached, nonce=nonce)
    gradients = trainable_gradient_snapshot(model)
    assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients.values())
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    optimizer.step()
    state.update += 1
    state.data_position += 1
    return loss, logits, weights, gradients, {"norm": norm.detach().clone()}


@pytest.mark.parametrize("rank", [None, 4, 16], ids=["full_finetune", "lora_r4", "lora_r16"])
def test_all_eight_members_full_grid_loss_gradients_adamw_and_updated_weights_match(rank):
    reference, candidate = _model(rank), _model(rank)
    config = TrainingConfig(method="full_finetune" if rank is None else "lora", rank=rank or 4)
    expected_optimizer, actual_optimizer = make_optimizer(reference, config), make_optimizer(candidate, config)
    expected_state, actual_state = TrainingState(stream_id="parity"), TrainingState(stream_id="parity")
    for update in range(2):
        expected = _one_update(reference, expected_optimizer, expected_state, cached=False, nonce=update)
        actual = _one_update(candidate, actual_optimizer, actual_state, cached=True, nonce=update)
        # On update zero, equal tensors enter the same forward operations.
        forward_atol, forward_rtol = (0., 0.) if update == 0 else (STATE_ATOL, STATE_RTOL)
        for part in range(3):
            _assert_report(compare_named_tensors(expected[part], actual[part], atol=forward_atol, rtol=forward_rtol))
        for part in (3, 4):
            _assert_report(compare_named_tensors(expected[part], actual[part], atol=GRADIENT_ATOL, rtol=GRADIENT_RTOL))
        _assert_report(compare_named_tensors(reference.state_dict(), candidate.state_dict(), atol=STATE_ATOL, rtol=STATE_RTOL))
        _assert_report(compare_named_tensors(_optimizer_tensors(expected_optimizer), _optimizer_tensors(actual_optimizer), atol=STATE_ATOL, rtol=STATE_RTOL))
        assert expected_optimizer.state_dict()["param_groups"] == actual_optimizer.state_dict()["param_groups"]
        assert expected_state == actual_state


@pytest.mark.parametrize("rank", [4, 16])
def test_cached_adamw_checkpoint_resume_is_exact_and_does_not_reuse_old_graph(tmp_path, rank):
    model = _model(rank)
    config = TrainingConfig(rank=rank)
    optimizer = make_optimizer(model, config)
    state = TrainingState(stream_id="cached-restart")
    episode_rng = np.random.default_rng(20261117)
    _one_update(model, optimizer, state, cached=True, nonce=int(episode_rng.integers(1000)))
    checkpoint = tmp_path / "checkpoint"
    manifest = save_training_checkpoint(checkpoint, model, optimizer, state, config=config, episode_rng=episode_rng)
    assert manifest["scientifically_qualified"] is False
    nonce = int(episode_rng.integers(1000))
    expected = _one_update(model, optimizer, state, cached=True, nonce=nonce)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = _optimizer_tensors(optimizer)
    expected_torch_rng = torch.get_rng_state().clone()
    restored = _model(rank)
    restored_optimizer = make_optimizer(restored, config)
    restored_rng = np.random.default_rng(0)
    restored_state = load_training_checkpoint(checkpoint, restored, restored_optimizer,
        config=config, expected_stream_id="cached-restart", episode_rng=restored_rng)
    assert int(restored_rng.integers(1000)) == nonce
    actual = _one_update(restored, restored_optimizer, restored_state, cached=True, nonce=nonce)
    for left, right in zip(expected, actual):
        _assert_report(compare_named_tensors(left, right, atol=0, rtol=0))
    _assert_report(compare_named_tensors(expected_model, restored.state_dict(), atol=0, rtol=0))
    _assert_report(compare_named_tensors(expected_optimizer, _optimizer_tensors(restored_optimizer), atol=0, rtol=0))
    assert restored_state == state
    assert torch.equal(expected_torch_rng, torch.get_rng_state())


@pytest.mark.parametrize("rank", [4, 16])
def test_cached_real_checkpoint_all_member_full_tail_loss_and_gradients_match(rank):
    checkpoint = os.environ.get("FTF_CPU_CHECKPOINT")
    if not checkpoint:
        pytest.skip("real cached checkpoint opt-in; no downloads")
    from posteriorforge.study.model import PreparedMember, TabPFNBackbone
    from posteriorforge.study.training_execution import conditional_forward_cache
    torch.set_num_threads(2)
    network = TabPFNBackbone.from_checkpoint(Path(checkpoint))
    generator = torch.Generator().manual_seed(2026111705)
    context = torch.randn(16, 3, generator=generator)
    targets = torch.randn(16, generator=generator)
    query = torch.randn(1, 3, generator=generator)
    members = [PreparedMember(context[:, torch.roll(torch.arange(3), i % 3)], targets,
        query[:, torch.roll(torch.arange(3), i % 3)]) for i in range(8)]
    network.backbone = attach_lora(network.backbone, rank=rank)
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0., std=.001, generator=generator)
    records = []
    for cached in (False, True):
        network.zero_grad(set_to_none=True)
        with conditional_forward_cache(network, enabled=cached):
            outputs = network(members)
            density = canonical_full_support_mixture(outputs, [network.borders] * 8, network.borders)
            loss = -density.log_prob(torch.tensor([.25], dtype=torch.float64), log_abs_det_decode=.17).sum()
            loss.backward()
        records.append(({"loss": loss.detach().clone()},
            {str(i): value.detach().clone() for i, value in enumerate(outputs)},
            {"weights": density.weights.detach().clone()}, trainable_gradient_snapshot(network)))
    for part in range(3):
        _assert_report(compare_named_tensors(records[0][part], records[1][part], atol=0, rtol=0))
    _assert_report(compare_named_tensors(records[0][3], records[1][3], atol=GRADIENT_ATOL, rtol=GRADIENT_RTOL))
