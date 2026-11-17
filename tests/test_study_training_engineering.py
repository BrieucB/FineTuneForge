from contextlib import nullcontext
import json
from types import SimpleNamespace
import pytest
from posteriorforge.study import training_engineering as eng


def test_scoped_call_restores_instance_even_on_exception():
    class Example:
        def action(self, value): return value + 1
    obj=Example(); labels=[]
    def record(label): labels.append(label); return nullcontext()
    with pytest.raises(RuntimeError):
        with eng.scoped_calls(obj, 'action', 'scope', record):
            assert obj.action(3)==4
            raise RuntimeError('failure')
    assert obj.action(4)==5 and 'action' not in obj.__dict__ and labels==['scope']


def test_binding_rejects_bad_input_checksum(tmp_path):
    p=tmp_path/'manifest.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        eng.verify_bindings(p,'wrong',tmp_path,'commit')


def test_worker_rejects_unaccounted_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        eng.worker('profile',inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')


def test_launch_requires_distinct_devices(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID','123'); monkeypatch.setenv('SLURM_JOB_ACCOUNT','eu-26-17')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,0')
    monkeypatch.setattr(eng,'verify_bindings',lambda *a: {})
    with pytest.raises(RuntimeError,match='distinct'):
        eng.launch(inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')


def test_qualification_launch_requires_four_unique_devices(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID','123'); monkeypatch.setenv('SLURM_JOB_ACCOUNT','eu-26-17')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,2')
    monkeypatch.setattr(eng,'verify_bindings',lambda *a: {})
    with pytest.raises(RuntimeError,match='four distinct'):
        eng.launch_qualification(inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')


def test_qualification_worker_rejects_wrong_lane_before_cuda(tmp_path):
    with pytest.raises(ValueError,match='lane'):
        eng.qualification_worker('unknown',inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')


def test_qualification_cli_routes_worker(monkeypatch, capsys):
    calls=[]
    monkeypatch.setattr(eng,'qualification_worker',lambda **kwargs: calls.append(kwargs) or {'ok':True})
    eng.main(['qualification-worker','--lane','cache_r4','--inputs','a','--input-sha256','b',
              '--output','c','--repo','d','--source-commit','e'])
    assert calls[0]['lane']=='cache_r4'
    assert json.loads(capsys.readouterr().out)=={'ok':True}


def test_conditional_parity_exercises_cache_and_restores(monkeypatch):
    import torch
    from contextlib import contextmanager
    from posteriorforge.study import model, training_execution
    monkeypatch.setattr(eng,'_rng_snapshot',lambda: 'rng')
    monkeypatch.setattr(eng,'_rng_restore',lambda _: None)
    monkeypatch.setattr(model,'configure_differentiable_forward',lambda _: None)
    active=[]
    @contextmanager
    def cache(_, *, enabled):
        active.append(enabled)
        yield
    monkeypatch.setattr(training_execution,'conditional_forward_cache',cache)
    class Density:
        def __init__(self, x): self.logits=x; self.weights=x
        def log_prob(self, targets, **kwargs): return self.logits.sum().reshape(1)
    class Ensemble:
        def adjusted_logits(self, network): return [network.weight.reshape(1)]*8
        def density(self, network, **kwargs): return Density(torch.stack(self.adjusted_logits(network)))
    class Backend:
        def __init__(self):
            self.model=torch.nn.Linear(1,1,bias=False)
            self.optimizer=torch.optim.SGD(self.model.parameters(),lr=0.1)
        def prepare(self, item):
            return SimpleNamespace(stages=[SimpleNamespace(index=0,parameter_name='a',ensemble=ensemble,
                                                          targets=torch.zeros(1),decoder_log_jac=torch.zeros(1))])
    ensemble=Ensemble(); backend=Backend()
    result=eng._conditional_parity(backend,object())
    assert result['passed'] and result['gradient_comparison']['passed'] and active==[False,True]
    assert 'adjusted_logits' not in ensemble.__dict__ and backend.model.weight.grad is None
