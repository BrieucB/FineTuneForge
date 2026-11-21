import numpy as np
import pytest
from scipy.special import ndtri
from posteriorforge.study.banks import prepare_bank, PairedBank
from posteriorforge.study.sampling import sample_posterior
from posteriorforge.study.science import encode_prior
from test_study_banks import problem

IDS = dict(model_sha256='a'*64, bank_manifest_sha256='b'*64, protocol_sha256='c'*64)

class Conditional:
    def __init__(self, fail_at=None): self.inputs=[]; self.calls=0; self.fail_at=fail_at
    def prepare(self, x, y):
        self.inputs.append((x.copy(), y.copy()))
        return len(self.inputs)-1
    def sample(self, stage, q, u):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError('interrupted')
        # Exact latent joint: z0|y ~ N(y0,1), z1|z0,y ~ N(.6*z0+y1, .8^2).
        assert q.shape[1] == 2 + stage
        return (q[:,0] + ndtri(u)) if stage == 0 else (.6*q[:,2] + q[:,1] + .8*ndtri(u))

def run(bank, backend, **kw):
    return sample_posterior(bank, np.array([.3,-.2]), backend, seed=17, microbatch_size=kw.pop('microbatch_size', 4000), identities=IDS, diagnostic_cpu=True, **kw)

def test_joint_law_context_conditioning_and_replay():
    bank = prepare_bank(problem(), 32, 5)
    backend = Conditional()
    x, meta = run(bank, backend)
    z = encode_prior(bank.spec, x)
    np.testing.assert_allclose(z.mean(0), [.3, -.02], atol=.018)
    np.testing.assert_allclose(np.cov(z.T), [[1., .6], [.6, 1.]], atol=.025)
    context_z = encode_prior(bank.spec, bank.theta)
    np.testing.assert_array_equal(backend.inputs[0][0], bank.obs)
    np.testing.assert_array_equal(backend.inputs[1][0], np.column_stack((bank.obs,context_z[:,0])))
    np.testing.assert_array_equal(backend.inputs[1][1], context_z[:,1])
    assert meta['posterior_sampling_forward_evaluations'] == 0
    assert x.shape == (50000,2) and bank.spec.support_mask(x).all()
    other, _ = run(bank, Conditional(), microbatch_size=7000)
    np.testing.assert_array_equal(x, other)
    changed, _ = sample_posterior(bank, np.array([.8,-.2]), Conditional(), seed=17, microbatch_size=4000, identities=IDS, diagnostic_cpu=True)
    assert np.mean(encode_prior(bank.spec, changed)[:,0]) - np.mean(z[:,0]) == pytest.approx(.5)

def test_resume_and_identity_tampering(tmp_path):
    bank = prepare_bank(problem(), 16, 4)
    path = tmp_path/'recovery.npz'
    with pytest.raises(RuntimeError, match='interrupted'):
        run(bank, Conditional(fail_at=4), resume_path=path)
    resumed, _ = run(bank, Conditional(), resume_path=path)
    direct, _ = run(bank, Conditional())
    np.testing.assert_array_equal(resumed, direct)
    with pytest.raises(ValueError, match='identity'):
        sample_posterior(bank, np.array([.4,-.2]), Conditional(), seed=17, microbatch_size=4000, identities=IDS, diagnostic_cpu=True, resume_path=path)
    with np.load(path, allow_pickle=False) as data:
        x, state = data['samples'].copy(), data['state'].copy()
    x[0,0] += 1
    np.savez(path, samples=x, state=state)
    with pytest.raises(ValueError, match='recovery'):
        run(bank, Conditional(), resume_path=path)

def test_guards_and_malformed_conditional():
    bank = prepare_bank(problem(), 8, 3)
    with pytest.raises(ValueError, match='diagnostic'):
        sample_posterior(bank, [.3,-.2], Conditional(), seed=0, microbatch_size=10, identities=IDS)
    class Bad(Conditional):
        def sample(self, stage, q, u): return np.full(len(u)-1, np.nan)
    with pytest.raises(ValueError, match='incomplete'):
        run(bank, Bad())
    with pytest.raises(ValueError, match='observation'):
        sample_posterior(bank, [.3], Conditional(), seed=0, microbatch_size=10, identities=IDS, diagnostic_cpu=True)

def test_decode_failure_records_only_complete_rows(tmp_path):
    import json
    bank = prepare_bank(problem(), 8, 3)
    class Overflow(Conditional):
        def __init__(self, overflow): super().__init__(); self.overflow = overflow
        def sample(self, stage, q, u):
            if stage == 1 and self.overflow: return np.full(len(u), 1e308)
            return super().sample(stage, q, u)
    path = tmp_path/'failed.npz'
    with pytest.raises((ValueError, FloatingPointError, OverflowError)):
        run(bank, Overflow(True), resume_path=path)
    with np.load(path, allow_pickle=False) as data:
        state = json.loads(data['state'].item())
        assert state['status'] == 'failed' and state['failure']['completed_rows'] == 0
        assert data['samples'].shape == (0, bank.spec.dimension)
    recovered, _ = run(bank, Overflow(False), resume_path=path)
    direct, _ = run(bank, Overflow(False))
    np.testing.assert_array_equal(recovered, direct)

def test_open_uniform_endpoints_are_excluded():
    from posteriorforge.study.sampling import _open_uniform
    class EndpointGenerator:
        def integers(self, low, high, *, size, dtype):
            assert (low, high, size, dtype) == (1, 1 << 53, 2, np.uint64)
            return np.array([low, high-1], dtype=dtype)
    result = _open_uniform(EndpointGenerator(), 2)
    assert result.dtype == np.float64 and np.all((result > 0) & (result < 1))
    assert result[0] == 2.0**-53 and result[1] == 1.0 - 2.0**-53

def test_recovery_rejects_changed_backend_implementation(tmp_path):
    bank = prepare_bank(problem(), 8, 3)
    path = tmp_path/'recovery.npz'
    with pytest.raises(RuntimeError): run(bank, Conditional(fail_at=4), resume_path=path)
    class Changed(Conditional):
        def sample(self, stage, q, u): return super().sample(stage, q, u)
    with pytest.raises(ValueError, match='identity'):
        run(bank, Changed(), resume_path=path)

def test_completed_recovery_rechecks_backend_weights(tmp_path):
    class Verifiable(Conditional):
        def __init__(self): super().__init__(); self.valid = True
        def verify_identity(self):
            if not self.valid: raise ValueError('model changed')
    bank = prepare_bank(problem(), 8, 3)
    path = tmp_path/'recovery.npz'
    model = Verifiable()
    run(bank, model, resume_path=path)
    model.valid = False
    with pytest.raises(ValueError, match='model changed'): run(bank, model, resume_path=path)

def test_invalid_decoded_rows_never_committed(tmp_path, monkeypatch):
    import json
    import posteriorforge.study.sampling as sampling
    bank = prepare_bank(problem(), 8, 3)
    path = tmp_path/'failed.npz'
    monkeypatch.setattr(sampling, 'decode_prior', lambda spec, z: np.zeros((len(z), spec.dimension)))
    with pytest.raises(ValueError, match='supported'): run(bank, Conditional(), resume_path=path)
    with np.load(path, allow_pickle=False) as data:
        assert data['samples'].shape == (0, 2)
        assert json.loads(data['state'].item())['status'] == 'failed'

def test_sampler_import_does_not_require_torch():
    import subprocess
    import sys
    script = '''\nimport builtins\nreal_import = builtins.__import__\ndef blocked(name, *args, **kwargs):\n    if name == "torch" or name.startswith("torch."): raise ImportError("torch deliberately blocked")\n    return real_import(name, *args, **kwargs)\nbuiltins.__import__ = blocked\nimport posteriorforge.study.sampling\n'''
    subprocess.run([sys.executable, '-c', script], check=True)

def test_declared_backend_law_and_recovery_binding(tmp_path):
    class Analytic(Conditional):
        sampling_law = 'analytic_zero_sensor_posterior.v1'
        source_policy = 'analytic_control'
    bank = prepare_bank(problem(), 8, 3)
    analytic = Analytic()
    path = tmp_path/'state.npz'
    _, meta = run(bank, analytic, resume_path=path)
    assert meta['binding']['requested_sampling_law'] == 'canonical_half_normal_full_support.v1'
    assert meta['binding']['law'] == analytic.sampling_law
    assert meta['binding']['source_policy'] == analytic.source_policy
    analytic.sampling_law = 'changed-law'
    with pytest.raises(ValueError, match='identity'): run(bank, analytic, resume_path=path)
    _, other = run(bank, Conditional())
    assert other['binding']['law'] == other['binding']['source_policy'] == 'diagnostic_backend_unspecified'
    analytic.sampling_law = ''
    with pytest.raises(ValueError, match='nonempty'): run(bank, analytic)
