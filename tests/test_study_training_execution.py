"""Cache-lifetime and packed-autograd fixtures; no production qualification."""
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn.utils import parametrize

from posteriorforge.study.model import ModelQualificationError, _PackedProjectionDelta, attach_lora
from posteriorforge.study.training import (
    TrainingConfig, TrainingState, load_training_checkpoint, make_optimizer,
    save_training_checkpoint,
)
from posteriorforge.study.training_execution import conditional_forward_cache


class CountingLowRankDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([[.2, -.4, .3], [-.1, .5, .2]]))
        self.b = nn.Parameter(torch.tensor([[.1, .2], [-.3, .4], [.2, -.1]]))
        self.calls = 0

    def forward(self, original):
        self.calls += 1
        return original + self.b @ self.a


class RepeatedProjectionFixture(nn.Module):
    """Repeated packed-weight access without changing the foundation model."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[.3, -.1, .5], [.2, .7, -.2], [-.3, .4, .1]]))
        parametrize.register_parametrization(self.projection, "weight", CountingLowRankDelta())
        self.delta.calls = 0

    @property
    def delta(self):
        return self.projection.parametrizations.weight[0]

    def forward(self, members):
        result = []
        for inputs in members:
            # TabPFN also probes a parametrized tensor before applying it.
            assert self.projection.weight is not None
            result.append(self.projection(inputs) + self.projection(inputs * .3))
        return torch.stack(result)


def member_inputs(offset=0.):
    return tuple(torch.tensor([[.1 + .03*i + offset, -.5, .2], [.4, .7 - .02*i, -.1 + offset]]) for i in range(8))


def run_update(model, optimizer, *, enabled):
    optimizer.zero_grad(set_to_none=True)
    for index in range(3):
        with conditional_forward_cache(model, enabled=enabled):
            output = model(member_inputs(.04 * index))
            # Wrapper lifetime includes scoring/backward; cache lifetime ends at forward return.
            loss = output.double().square().mean() / 3
            loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
    optimizer.step()
    return norm


def assert_optimizer_equal(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    for index, entry in left_state["state"].items():
        for name, value in entry.items():
            other = right_state["state"][index][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (index, name)
            else:
                assert value == other


def test_opt_in_reuses_one_graph_for_all_eight_members_and_preserves_forward_bits():
    reference, actual = RepeatedProjectionFixture(), RepeatedProjectionFixture()
    expected = reference(member_inputs())
    with conditional_forward_cache(actual, enabled=True):
        observed = actual(member_inputs())
    assert reference.delta.calls == 24 and actual.delta.calls == 1
    assert torch.equal(expected, observed)
    expected.double().square().mean().backward()
    observed.double().square().mean().backward()
    for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters()):
        if left.requires_grad:
            assert left.grad is not None and right.grad is not None, name
            assert bool((left.grad != 0).any()), name
            assert torch.allclose(left.grad, right.grad, atol=1e-7, rtol=1e-5), name


def test_disabled_default_is_uncached_and_preserves_no_grad_behavior():
    model = RepeatedProjectionFixture()
    with torch.no_grad(), conditional_forward_cache(model):
        output = model(member_inputs())
    assert model.delta.calls == 24 and not output.requires_grad
    with pytest.raises(TypeError, match="explicit boolean"):
        with conditional_forward_cache(model, enabled="yes"):
            pass


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_enabled_scope_requires_autograd(mode):
    model = RepeatedProjectionFixture()
    with mode(), pytest.raises(ModelQualificationError, match="gradient-enabled"):
        with conditional_forward_cache(model, enabled=True):
            model(member_inputs())
    assert model.delta.calls == 0


def test_enabled_scope_requires_trainable_parameters_and_supported_cache_counter(monkeypatch):
    with pytest.raises(ModelQualificationError, match="trainable parameters"):
        with conditional_forward_cache(nn.Linear(2, 2).requires_grad_(False), enabled=True):
            pass
    model = RepeatedProjectionFixture()
    monkeypatch.setattr(parametrize, "_cache_enabled", None)
    with pytest.raises(ModelQualificationError, match="isolated"):
        with conditional_forward_cache(model, enabled=True):
            pass
    assert model.delta.calls == 0


@pytest.mark.parametrize("outer_helper", [False, True])
def test_nested_cache_is_refused_before_an_inner_forward(outer_helper):
    model = RepeatedProjectionFixture()
    outer = conditional_forward_cache(model, enabled=True) if outer_helper else parametrize.cached()
    with outer:
        with pytest.raises(ModelQualificationError, match="already-active"):
            with conditional_forward_cache(model, enabled=True):
                model(member_inputs())
    assert model.delta.calls == 0
    with conditional_forward_cache(model, enabled=True):
        output = model(member_inputs())
    output.sum().backward()
    assert model.delta.calls == 1 and model.delta.a.grad is not None


def test_exception_discards_cache_before_changed_weights_are_reused():
    model = RepeatedProjectionFixture()
    before = deepcopy(model.state_dict())
    hooks = {name: len(module._forward_hooks) for name, module in model.named_modules()}
    with pytest.raises(RuntimeError, match="injected"):
        with conditional_forward_cache(model, enabled=True):
            initial = model(member_inputs()).detach()
            raise RuntimeError("injected forward failure")
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    with torch.no_grad():
        model.delta.b.add_(.2)
    with conditional_forward_cache(model, enabled=True):
        changed = model(member_inputs())
    assert model.delta.calls == 2 and not torch.equal(initial, changed)
    changed.sum().backward()
    assert model.delta.a.grad is not None and torch.isfinite(model.delta.a.grad).all()
    assert hooks == {name: len(module._forward_hooks) for name, module in model.named_modules()}
    assert set(before) == set(model.state_dict())


def test_three_conditionals_preserve_gradients_and_one_adamw_update():
    reference, actual = RepeatedProjectionFixture(), RepeatedProjectionFixture()
    config = TrainingConfig(learning_rate=1e-3)
    reference_optimizer, actual_optimizer = make_optimizer(reference, config), make_optimizer(actual, config)
    expected_norm = run_update(reference, reference_optimizer, enabled=False)
    actual_norm = run_update(actual, actual_optimizer, enabled=True)
    assert reference.delta.calls == 72 and actual.delta.calls == 3
    assert torch.allclose(expected_norm, actual_norm, atol=1e-7, rtol=1e-5)
    for (name, left), (_, right) in zip(reference.named_parameters(), actual.named_parameters()):
        assert torch.allclose(left, right, atol=1e-7, rtol=1e-5), name
        if left.requires_grad:
            assert torch.allclose(left.grad, right.grad, atol=1e-7, rtol=1e-5), name
    for index, entry in reference_optimizer.state_dict()["state"].items():
        for name, value in entry.items():
            other = actual_optimizer.state_dict()["state"][index][name]
            assert torch.allclose(value, other, atol=1e-7, rtol=1e-5), (index, name)


def test_checkpoint_continuation_has_no_cache_state_and_is_bitwise(tmp_path):
    uninterrupted = RepeatedProjectionFixture()
    config = TrainingConfig(learning_rate=1e-3)
    optimizer = make_optimizer(uninterrupted, config)
    run_update(uninterrupted, optimizer, enabled=True)
    state = TrainingState(update=1, data_position=16, stream_id="cache-lifetime-fixture")
    manifest = save_training_checkpoint(tmp_path / "checkpoint", uninterrupted, optimizer, state, config=config)
    resumed = RepeatedProjectionFixture()
    resumed_optimizer = make_optimizer(resumed, config)
    resumed_state = load_training_checkpoint(tmp_path / "checkpoint", resumed, resumed_optimizer, config=config, expected_stream_id=state.stream_id)
    assert resumed_state == state
    assert manifest["trainable_parameter_names"] == [name for name, value in resumed.named_parameters() if value.requires_grad]
    assert not any("cache" in name for name in resumed.state_dict())
    run_update(uninterrupted, optimizer, enabled=True)
    run_update(resumed, resumed_optimizer, enabled=True)
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in uninterrupted.state_dict().items())
    assert_optimizer_equal(optimizer, resumed_optimizer)


class PackedAttentionFixture(nn.Module):
    """Actual TabPFN packed attention and PEFT branches, without a replacement model."""
    def __init__(self):
        super().__init__()
        from tabpfn.model.multi_head_attention import MultiHeadAttention
        self.attention = MultiHeadAttention(input_size=4, output_size=4, d_k=2, d_v=2, nhead=2,
                                           device=torch.device("cpu"), dtype=torch.float32)
        self.encoder = nn.Module()
        self.encoder.mlp = nn.Sequential(nn.Linear(4, 8, bias=False), nn.GELU(), nn.Linear(8, 4, bias=False))
        self.decoder_dict = nn.ModuleDict({"standard": nn.Linear(4, 3, bias=False)})

    def forward(self, inputs):
        return self.decoder_dict["standard"](self.encoder.mlp(self.attention(inputs)))


@pytest.mark.parametrize("rank", [4, 16])
def test_real_peft_packed_attention_recomputes_once_per_scope_and_retains_all_gradients(rank):
    pytest.importorskip("peft")
    pytest.importorskip("tabpfn")
    torch.manual_seed(951)
    model = attach_lora(PackedAttentionFixture().eval().requires_grad_(False), rank=rank)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(0., .02)
    packed = {name: module for name, module in model.named_modules() if isinstance(module, _PackedProjectionDelta)}
    assert packed
    counts = {name: 0 for name in packed}
    handles = []
    for name, module in packed.items():
        def record(_module, _inputs, _output, name=name):
            counts[name] += 1
        handles.append(module.register_forward_hook(record))
    inputs = tuple(torch.randn(2, 3, 4) + .01 * index for index in range(8))
    try:
        expected = torch.stack([model(value) for value in inputs])
        uncached_counts = counts.copy()
        expected.double().square().mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        model.zero_grad(set_to_none=True)
        counts.update({name: 0 for name in packed})
        with conditional_forward_cache(model, enabled=True):
            actual = torch.stack([model(value) for value in inputs])
        assert torch.equal(expected, actual)
        assert all(count == 8 for count in counts.values())
        assert all(uncached_counts[name] >= counts[name] for name in packed)
        assert any(uncached_counts[name] > counts[name] for name in packed)
        actual.double().square().mean().backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert torch.allclose(gradients[name], parameter.grad, atol=1e-7, rtol=1e-5), name
    finally:
        for handle in handles:
            handle.remove()


class CheckpointedTransformerFixture(nn.Module):
    """Actual TabPFN checkpointed encoder layers, without foundation substitution."""
    def __init__(self, *, nested=False, unknown=False):
        super().__init__()
        from tabpfn.model.layer import PerFeatureEncoderLayer
        from tabpfn.model.transformer import LayerStack
        creator = (lambda: nn.Linear(4, 4, bias=False)) if unknown else (lambda: PerFeatureEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=8, recompute_attn=nested,
            device=torch.device("cpu"), dtype=torch.float32))
        self.encoder = LayerStack(layer_creator=creator, num_layers=2, recompute_each_layer=True)
        self.decoder_dict = nn.ModuleDict({"standard": nn.Linear(4, 3, bias=False)})

    def forward(self, value):
        encoded = self.encoder(value, single_eval_pos=4)
        return self.decoder_dict["standard"](encoded[:, -1, -1])


def checkpointed_model(rank):
    pytest.importorskip("peft")
    pytest.importorskip("tabpfn")
    torch.manual_seed(15831)
    model = attach_lora(CheckpointedTransformerFixture().eval().requires_grad_(False), rank=rank)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0., .02)
    return model


@pytest.mark.parametrize("rank", [4, 16])
def test_actual_checkpointed_layers_recompute_with_same_scope_and_match_update(rank):
    from posteriorforge.study.model import CANONICAL_FULL_SUPPORT, canonical_full_support_mixture
    reference, actual = checkpointed_model(rank), checkpointed_model(rank)
    actual_base = actual.get_base_model()
    layers = tuple(actual_base.encoder.layers)
    original_forwards = tuple(layer.forward.__func__ for layer in layers)
    state_names = set(actual.state_dict())
    generator = torch.Generator().manual_seed(15832)
    inputs = tuple(torch.randn(1, 5, 2, 4, generator=generator).requires_grad_(True) for _ in range(8))
    calls = [0] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        def record(_module, _args, index=index):
            calls[index] += 1
        handles.append(layer.register_forward_pre_hook(record))
    reference_optimizer = make_optimizer(reference, TrainingConfig(learning_rate=1e-3))
    actual_optimizer = make_optimizer(actual, TrainingConfig(learning_rate=1e-3))
    borders = torch.tensor([-2., -1., 1., 3.])
    records = []
    try:
        for model, optimizer, enabled in ((reference, reference_optimizer, False), (actual, actual_optimizer, True)):
            with conditional_forward_cache(model, enabled=enabled):
                outputs = tuple(model(value) for value in inputs)
                density = canonical_full_support_mixture(outputs, [borders] * 8, borders)
                loss = -density.log_prob(torch.tensor([.25], dtype=torch.float64)).sum()
                if enabled:
                    assert calls == [8, 8]
                    assert all("forward" in layer.__dict__ for layer in layers)
                    assert parametrize._cache_enabled == 0
                loss.backward()
                if enabled:
                    assert calls == [16, 16]
                    assert parametrize._cache_enabled == 0
            gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
            optimizer.step()
            records.append((torch.stack(outputs).detach(), density.weights.detach(), loss.detach(), gradients, norm.detach()))
        assert torch.equal(records[0][0], records[1][0])
        assert torch.equal(records[0][1], records[1][1])
        assert torch.equal(records[0][2], records[1][2])
        for name, gradient in records[0][3].items():
            assert torch.allclose(gradient, records[1][3][name], atol=2e-7, rtol=2e-5), name
        assert torch.allclose(records[0][4], records[1][4], atol=2e-7, rtol=2e-5)
        for name, value in reference.state_dict().items():
            assert torch.allclose(value, actual.state_dict()[name], atol=2e-7, rtol=2e-5), name
        assert set(actual.state_dict()) == state_names
        assert actual_base.encoder.recompute_each_layer is True
        assert all(layer.recompute_attn is False for layer in layers)
        assert all("forward" not in layer.__dict__ for layer in layers)
        assert all(layer.forward.__func__ is original for layer, original in zip(layers, original_forwards))
    finally:
        for handle in handles:
            handle.remove()


def test_checkpointed_layer_wrappers_are_restored_after_backward_failure():
    model = checkpointed_model(4)
    layers = tuple(model.get_base_model().encoder.layers)
    generator = torch.Generator().manual_seed(15833)
    value = torch.randn(1, 5, 2, 4, generator=generator).requires_grad_(True)
    parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    def fail(_gradient):
        raise RuntimeError("injected backward failure")
    handle = parameter.register_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected backward"):
            with conditional_forward_cache(model, enabled=True):
                model(value).double().square().mean().backward()
    finally:
        handle.remove()
    assert all("forward" not in layer.__dict__ for layer in layers)
    assert parametrize._cache_enabled == 0
    model.zero_grad(set_to_none=True)
    with conditional_forward_cache(model, enabled=True):
        model(value).double().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize("layout", ["nested", "unknown", "preexisting"])
def test_unsupported_checkpoint_layouts_fail_before_any_forward_is_modified(layout):
    pytest.importorskip("tabpfn")
    if layout == "preexisting":
        model = CheckpointedTransformerFixture()
        first = model.encoder.layers[0]
        first.forward = first.forward
    else:
        model = CheckpointedTransformerFixture(nested=layout == "nested", unknown=layout == "unknown")
    before = tuple(layer.__dict__.get("forward") for layer in model.encoder.layers)
    with pytest.raises(ModelQualificationError, match="unsupported|unknown|nested|preexisting"):
        with conditional_forward_cache(model, enabled=True):
            pass
    assert tuple(layer.__dict__.get("forward") for layer in model.encoder.layers) == before
    assert parametrize._cache_enabled == 0
