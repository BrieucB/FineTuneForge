"""CPU-only policy checks for the opt-in CUDA qualification interfaces."""
import pytest

torch = pytest.importorskip('torch')
from posteriorforge.study.model import (resolve_inference_device, ModelQualificationError,
    PreparedMember, TabPFNBackbone)
from posteriorforge.study.sampling_backend import TabPFNConditionalBackend


def test_device_policy_rejects_non_cuda_and_unavailable_cuda(monkeypatch):
    assert resolve_inference_device() == torch.device('cpu')
    for value in ('mps', 'meta', 'cpu:0', 'invalid'):
        with pytest.raises(ValueError, match='device'): resolve_inference_device(value)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ModelQualificationError, match='unavailable'):
        resolve_inference_device('cuda')
    with pytest.raises(ModelQualificationError, match='unavailable'):
        resolve_inference_device('cuda:0')


def test_device_index_policy_without_gpu_access(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    assert str(resolve_inference_device('cuda')) == 'cuda:1'
    assert str(resolve_inference_device('cuda:0')) == 'cuda:0'
    with pytest.raises(ModelQualificationError, match='unavailable'):
        resolve_inference_device('cuda:2')


def test_prepared_member_rejects_wrong_precision_before_network():
    net = TabPFNBackbone(torch.nn.Linear(2, 2), torch.tensor([0.,1.,2.]), checkpoint_sha256='fixture')
    member = PreparedMember(torch.ones((2,2),dtype=torch.float64), torch.ones(2), torch.ones((1,2)))
    with pytest.raises(ModelQualificationError, match='FP32 inputs'):
        net.forward_member(member)


def test_cpu_default_is_still_explicit():
    assert TabPFNConditionalBackend.backend_kind == 'stock_tabpfn_cpu'
    assert TabPFNConditionalBackend.sampling_law == 'canonical_half_normal_full_support.v1'
