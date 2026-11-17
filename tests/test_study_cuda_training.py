"""CPU-only guards/envelope tests; CUDA numerical proof belongs to the GPU pilot."""
from copy import deepcopy
from dataclasses import asdict
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from posteriorforge.study.model import CANONICAL_FULL_SUPPORT, ModelQualificationError
from posteriorforge.study import training
from posteriorforge.study.training import TrainingConfig, TrainingState, make_optimizer, save_training_checkpoint, load_training_checkpoint
from posteriorforge.study.updates import stagewise_cuda_qualification_update
from posteriorforge.study.cuda_training import QualificationTrainingBackend, configure_cuda_determinism


def network():
    return nn.Sequential(nn.Linear(2, 3), nn.Dropout(.25), nn.Linear(3, 1))


def test_cuda_update_requires_explicit_opt_in_and_rejects_cpu_without_device_probe(monkeypatch):
    model = network(); optimizer = make_optimizer(model, TrainingConfig())
    def forbidden():
        raise AssertionError('CPU negative test must not probe CUDA')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    with pytest.raises(ModelQualificationError, match='opt-in'):
        stagewise_cuda_qualification_update(model, optimizer, [], TrainingState(), law=CANONICAL_FULL_SUPPORT)
    with pytest.raises(ModelQualificationError, match='CUDA-only'):
        stagewise_cuda_qualification_update(model, optimizer, [], TrainingState(), law=CANONICAL_FULL_SUPPORT, allow_cuda_qualification=True)


def test_cuda_backend_requires_explicit_opt_in_and_rejects_cpu_before_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: (_ for _ in ()).throw(AssertionError('no GPU probe')))
    with pytest.raises(ModelQualificationError, match='opt-in'):
        QualificationTrainingBackend(None, 'missing')
    with pytest.raises(ModelQualificationError, match='CUDA device'):
        QualificationTrainingBackend(None, 'missing', device='cpu', allow_cuda_qualification=True)


def test_cuda_math_policy_requires_predeclared_cublas_and_available_device(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ModelQualificationError, match='CUBLAS_WORKSPACE_CONFIG'):
        configure_cuda_determinism()
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ModelQualificationError, match='available CUDA'):
        configure_cuda_determinism()


def test_cuda_backend_preserves_effective_batch_and_one_query_before_gpu_work():
    backend = object.__new__(QualificationTrainingBackend)
    backend.spec = SimpleNamespace(effective_batch=16)
    with pytest.raises(ModelQualificationError, match='effective batch'):
        backend.update([], TrainingState())
    items = [SimpleNamespace(query=SimpleNamespace(theta=np.zeros((2, 4))))] * 16
    with pytest.raises(ModelQualificationError, match='independent query'):
        backend.update(items, TrainingState())


def _mock_cuda_envelope(monkeypatch):
    # Only the serialization envelope is simulated; all arithmetic remains CPU.
    policy = {'device':'cuda:0', 'visible_devices':[{'ordinal':0},{'ordinal':1}],
              'deterministic_algorithms':True, 'neural_precision':'float32'}
    states = [torch.tensor([1, 3, 5], dtype=torch.uint8), torch.tensor([2, 4, 6], dtype=torch.uint8)]
    monkeypatch.setattr(training, '_model_device', lambda model: torch.device('cuda:0'))
    monkeypatch.setattr(training, '_cuda_execution_metadata', lambda model: deepcopy(policy))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [s.clone() for s in states])
    def restore(values):
        states[:] = [s.clone() for s in values]
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', restore)
    return policy, states


def _step(model, optimizer, state, episode_rng):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 2)
    scale = 1 + np.random.random() + random.random() + episode_rng.random()
    loss = (model(x) * scale).square().mean()
    loss.backward(); optimizer.step()
    state.update += 1; state.data_position += 16


def _equal_optimizer(left, right):
    assert left['param_groups'] == right['param_groups']
    for key, state in left['state'].items():
        for name, value in state.items():
            assert torch.equal(value, right['state'][key][name])


def test_mock_cuda_checkpoint_envelope_restores_all_rngs_flags_optimizer_and_cursor(tmp_path, monkeypatch):
    policy, cuda_states = _mock_cuda_envelope(monkeypatch)
    torch.manual_seed(71); np.random.seed(72); random.seed(73)
    config = TrainingConfig(max_updates=2); model = network(); model.train(); model[0].eval()
    optimizer = make_optimizer(model, config); state = TrainingState(stream_id='engineering-stream')
    episode_rng = np.random.default_rng(74)
    _step(model, optimizer, state, episode_rng)
    checkpoint_flags = {name: m.training for name, m in model.named_modules()}
    manifest = save_training_checkpoint(tmp_path/'checkpoint', model, optimizer, state, config=config, episode_rng=episode_rng)
    assert manifest['schema'] == 'finetuneforge.training_checkpoint.cuda_qualification.v1'
    assert manifest['execution'] == policy and manifest['scientifically_qualified'] is False
    _step(model, optimizer, state, episode_rng)
    expected_weights = deepcopy(model.state_dict()); expected_optimizer = deepcopy(optimizer.state_dict())
    future = (torch.rand(3), np.random.rand(3), random.random(), episode_rng.random(3))
    cuda_states[:] = [torch.zeros(3, dtype=torch.uint8)] * 2
    fresh = network(); fresh.eval(); fresh_optimizer = make_optimizer(fresh, config); fresh_rng = np.random.default_rng(0)
    restored = load_training_checkpoint(tmp_path/'checkpoint', fresh, fresh_optimizer, config=config,
        expected_stream_id='engineering-stream', episode_rng=fresh_rng)
    assert restored.update == 1 and restored.data_position == 16
    assert {name: m.training for name, m in fresh.named_modules()} == checkpoint_flags
    assert [s.tolist() for s in cuda_states] == [[1,3,5],[2,4,6]]
    _step(fresh, fresh_optimizer, restored, fresh_rng)
    assert restored == state
    assert all(torch.equal(v, fresh.state_dict()[name]) for name,v in expected_weights.items())
    _equal_optimizer(expected_optimizer, fresh_optimizer.state_dict())
    assert torch.equal(future[0], torch.rand(3)) and np.array_equal(future[1], np.random.rand(3))
    assert future[2] == random.random() and np.array_equal(future[3], fresh_rng.random(3))
    assert all(value.device.type == 'cpu' for values in fresh_optimizer.state.values() for value in values.values() if isinstance(value, torch.Tensor))


def test_mock_cuda_policy_change_is_rejected_before_weight_mutation(tmp_path, monkeypatch):
    policy, states = _mock_cuda_envelope(monkeypatch)
    config = TrainingConfig(); original = network(); optimizer = make_optimizer(original, config)
    save_training_checkpoint(tmp_path/'checkpoint', original, optimizer, TrainingState(stream_id='x'), config=config)
    fresh = network(); before = deepcopy(fresh.state_dict()); fresh_optimizer = make_optimizer(fresh, config)
    policy['deterministic_algorithms'] = False
    with pytest.raises(ModelQualificationError, match='numerical policy'):
        load_training_checkpoint(tmp_path/'checkpoint', fresh, fresh_optimizer, config=config, expected_stream_id='x')
    assert all(torch.equal(value, fresh.state_dict()[name]) for name,value in before.items())


def test_cuda_checkpoint_cannot_be_restored_as_cpu_v1(tmp_path, monkeypatch):
    original_model_device = training._model_device
    _mock_cuda_envelope(monkeypatch)
    config = TrainingConfig(); model = network(); optimizer = make_optimizer(model, config)
    save_training_checkpoint(tmp_path/'checkpoint', model, optimizer, TrainingState(stream_id='x'), config=config)
    monkeypatch.setattr(training, '_model_device', original_model_device)
    with pytest.raises(ModelQualificationError, match='configuration/device'):
        load_training_checkpoint(tmp_path/'checkpoint', model, optimizer, config=config, expected_stream_id='x')


def test_mock_cuda_rng_inventory_is_validated_before_mutation(tmp_path, monkeypatch):
    _mock_cuda_envelope(monkeypatch)
    config = TrainingConfig(); model = network(); optimizer = make_optimizer(model, config)
    save_training_checkpoint(tmp_path/'checkpoint', model, optimizer, TrainingState(stream_id='x'), config=config)
    path = tmp_path/'checkpoint'/'training.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True); payload['cuda_rng'] = [torch.tensor([1], dtype=torch.uint8)]
    torch.save(payload, path)
    manifest_path = tmp_path/'checkpoint'/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    manifest['files']['training.pt'] = training._hash(path); manifest_path.write_text(json.dumps(manifest))
    before=deepcopy(model.state_dict())
    with pytest.raises(ModelQualificationError, match='CUDA RNG'):
        load_training_checkpoint(tmp_path/'checkpoint', model, optimizer, config=config, expected_stream_id='x')
    assert all(torch.equal(value, model.state_dict()[name]) for name,value in before.items())


def test_cpu_checkpoint_keeps_v1_and_does_not_touch_cuda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU checkpoint must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    config = TrainingConfig(); model = network(); optimizer = make_optimizer(model, config)
    manifest = save_training_checkpoint(tmp_path/'checkpoint', model, optimizer, TrainingState(stream_id='x'), config=config)
    assert manifest['schema'] == 'finetuneforge.training_checkpoint.v1' and 'execution' not in manifest
    restored=load_training_checkpoint(tmp_path/'checkpoint', model, optimizer, config=config, expected_stream_id='x')
    assert restored == TrainingState(stream_id='x')


def test_campaign_source_binding_covers_cuda_training():
    from posteriorforge.study.campaign import source_identity
    assert len(source_identity()['cuda_training.py']) == 64
