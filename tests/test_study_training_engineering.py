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
    torch = pytest.importorskip('torch')
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


def test_fusion_checkpoint_rejects_manifest_before_loading(tmp_path, monkeypatch):
    from posteriorforge.study import campaign
    p=tmp_path/'checkpoint';p.mkdir();(p/'manifest.json').write_text('{}')
    monkeypatch.setattr(campaign,'read_campaign_checkpoint',lambda _: pytest.fail('must not load bad manifest'))
    with pytest.raises(ValueError,match='checksum'):
        eng.verify_fusion_checkpoint(p,'bad','source',None,'stream','inputs')


def test_fusion_comparison_gates_outputs_and_gradients():
    torch = pytest.importorskip('torch')
    def result():
        return {'logits':[torch.ones(2) for _ in range(8)],'cdf':torch.tensor([.5],dtype=torch.float64),
                'quantiles':torch.tensor([1.],dtype=torch.float64),'loss':torch.tensor(1.,dtype=torch.float64),
                'gradients':{'a':torch.ones(1)},'wall_seconds':1.}
    left=result(); right=result()
    assert eng._fusion_compare(left,right)['passed']
    right['cdf']+=.01
    assert not eng._fusion_compare(left,right)['passed']
    right=result();right['gradients']['a']+=.1
    assert not eng._fusion_compare(left,right)['passed']


def test_fusion_cli_routes_arguments(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(eng,'fusion_preflight',lambda **kwargs: calls.append(kwargs) or {'ok':True})
    eng.main(['fusion-preflight','--inputs','a','--input-sha256','b','--output','c',
              '--repo','d','--source-commit','e','--checkpoint','f','--checkpoint-sha256','g',
              '--checkpoint-source-commit','h'])
    assert calls[0]['checkpoint_source_commit']=='h'
    assert json.loads(capsys.readouterr().out)=={'ok':True}


def test_triton_fp_fusion_policy_default_and_explicit():
    assert eng.triton_fp_fusion_policy({}) == {'environment_value':None,'effective_value':'1','enabled':True,
        'meaning':'Triton compiler default; explicit operations and external kernels may differ'}
    p=eng.triton_fp_fusion_policy({'TRITON_DEFAULT_FP_FUSION':'0'})
    assert p['effective_value']=='0' and p['enabled'] is False
    assert eng.triton_fp_fusion_policy({'TRITON_DEFAULT_FP_FUSION':'1'})['enabled'] is True
    with pytest.raises(ValueError,match='must be 0 or 1'):
        eng.triton_fp_fusion_policy({'TRITON_DEFAULT_FP_FUSION':'false'})


def test_fusion_invalid_policy_fails_before_import_and_auth(monkeypatch,tmp_path):
    import builtins
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_JOB_ACCOUNT','eu-26-17')
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION','invalid')
    monkeypatch.setattr(eng,'verify_bindings',lambda *args: pytest.fail('authentication must not run'))
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name=='torch': pytest.fail('torch must not be imported')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    with pytest.raises(ValueError,match='TRITON_DEFAULT_FP_FUSION'):
        eng.fusion_preflight(inputs='x',input_sha256='x',output=tmp_path/'out',repo='x',
                             source_commit='x',checkpoint='x',checkpoint_sha256='x',checkpoint_source_commit='x')
    assert not (tmp_path/'out').exists()


def test_fusion_failure_receipt_records_effective_policy(monkeypatch,tmp_path):
    pytest.importorskip("torch")
    from posteriorforge.study import cuda_training
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_JOB_ACCOUNT','eu-26-17')
    monkeypatch.setenv('TRITON_DEFAULT_FP_FUSION','0')
    monkeypatch.setattr(eng,'verify_bindings',lambda *args: {'protocol_sha256':'a'*64,'episodes':[]})
    monkeypatch.setattr(eng,'verify_fusion_checkpoint',lambda *args: {'weights_sha256':'b'*64})
    def stop(): raise RuntimeError('fixture stop before CUDA')
    monkeypatch.setattr(cuda_training,'configure_cuda_determinism',stop)
    out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='fixture stop'):
        eng.fusion_preflight(inputs='x',input_sha256='x',output=out,repo='x',source_commit='x',
                             checkpoint='x',checkpoint_sha256='x',checkpoint_source_commit='x')
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='failed' and result['failure']['message']=='fixture stop before CUDA'
    assert result['triton_fp_fusion']['environment_value']=='0'
    assert result['triton_fp_fusion']['effective_value']=='0'
    assert result['triton_fp_fusion']['enabled'] is False
