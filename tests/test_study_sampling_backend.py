import pytest
pytest.importorskip('torch')
from posteriorforge.study.model import ModelQualificationError
from posteriorforge.study.sampling_backend import TabPFNConditionalBackend

def test_missing_checkpoint_is_never_downloaded(tmp_path):
    with pytest.raises(ModelQualificationError, match='pinned cached'):
        TabPFNConditionalBackend(tmp_path/'missing.ckpt', object(), expected_model_sha256='a'*64)

def test_backend_class_does_not_accept_unverified_checkpoint(tmp_path):
    bad = tmp_path/'checkpoint.ckpt'
    bad.write_bytes(b'not the pinned model')
    with pytest.raises(ModelQualificationError, match='pinned cached'):
        TabPFNConditionalBackend(bad, object(), expected_model_sha256='a'*64)

def test_real_cached_checkpoint_two_queries_uses_current_lora_weights():
    import os
    import time
    import numpy as np
    import torch
    from posteriorforge.study.banks import prepare_bank
    from posteriorforge.study.model import (TabPFNBackbone, attach_lora, weights_fingerprint,
        prepare_fitted_regressor_members)
    from posteriorforge.study.science import encode_prior
    from test_study_banks import problem
    path = os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:
        pytest.skip('real checkpoint test is opt-in and never downloads')
    torch.set_num_threads(2)
    torch.manual_seed(71)
    bank = prepare_bank(problem(), 16, 2)
    network = TabPFNBackbone.from_checkpoint(path)
    t = time.monotonic()
    fingerprint = weights_fingerprint(network)
    print(f'CPU model fingerprint seconds={time.monotonic()-t:.6f}', flush=True)
    backend = TabPFNConditionalBackend(path, network, expected_model_sha256=fingerprint)
    reg = backend.prepare(bank.observations, encode_prior(bank.spec, bank.theta)[:,0])
    assert reg.model_ is None and reg.executor_.model is None
    queries = np.array([[.1, .15], [-.2, .1]], dtype=np.float64)
    ensemble = prepare_fitted_regressor_members(reg, queries, device='cpu', source_policy='cancelled_zero_width')
    assert len(ensemble.members) == 8 and len(ensemble.cancel_masks) == 8
    assert ensemble.source_policy == 'cancelled_zero_width'
    uniforms = np.array([.3, .7], dtype=np.float64)
    original = backend.sample(reg, queries, uniforms)
    assert original.shape == (2,) and np.isfinite(original).all()
    separate = np.concatenate([backend.sample(reg, queries[i:i+1], uniforms[i:i+1]) for i in range(2)])
    np.testing.assert_allclose(original, separate, rtol=2e-5, atol=2e-6)
    network.backbone = attach_lora(network.backbone, rank=4)
    with torch.no_grad():
        for name, value in network.backbone.named_parameters():
            if 'lora_B' in name:
                value.normal_(0, .03)
    updated_hash = weights_fingerprint(network)
    assert updated_hash != fingerprint
    updated_backend = TabPFNConditionalBackend(path, network, expected_model_sha256=updated_hash)
    updated = updated_backend.sample(reg, queries, uniforms)
    assert np.isfinite(updated).all()
    print(f'original={original.tolist()} updated={updated.tolist()} max_difference={np.max(np.abs(original-updated)):.9g}', flush=True)
    assert not np.allclose(original, updated, atol=1e-6, rtol=1e-6)
    with pytest.raises(Exception, match='model weights changed'):
        backend.verify_identity()
    updated_backend.verify_identity()

def test_stock_backend_declares_canonical_law():
    from posteriorforge.study.sampling_backend import TabPFNConditionalBackend
    assert TabPFNConditionalBackend.sampling_law == 'canonical_half_normal_full_support.v1'
    assert TabPFNConditionalBackend.source_policy == 'cancelled_zero_width'
    assert TabPFNConditionalBackend.backend_kind == 'stock_tabpfn_cpu'
