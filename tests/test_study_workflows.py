from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.special import ndtri
from posteriorforge.study import workflows as w
from posteriorforge.study.artifacts import load_samples
from posteriorforge.study.banks import prepare_bank, save_bank
from posteriorforge.study.science import encode_prior
from test_study_banks import problem


class Conditional:
    model_sha256 = 'a'*64
    def __init__(self, fail_at=None):
        self.calls=0
        self.fail_at=fail_at
    def prepare(self, x, target):
        return x.shape[1]-2
    def sample(self, stage, query, uniforms):
        self.calls+=1
        if self.calls==self.fail_at:
            raise RuntimeError('unit interruption')
        return query[:,0]+ndtri(uniforms) if stage==0 else .6*query[:,2]+query[:,1]+.8*ndtri(uniforms)


@pytest.fixture
def posterior_request(tmp_path):
    receipt=save_bank(prepare_bank(problem(),64,7),tmp_path/'bank')
    np.save(tmp_path/'query.npy',np.array([.3,-.2],dtype=np.float64))
    value=w.PosteriorRequest('bank',receipt['manifest_sha256'],'query.npy',w.file_sha256(tmp_path/'query.npy'),
        {'base_checkpoint':'unused-by-injected-test.ckpt','model_sha256':'a'*64},'b'*64,
        context_size=16,seed=12,microbatch_size=4096)
    return value


def backend_factory(fail_at=None):
    def factory(model, *, base_directory):
        return Conditional(fail_at),{'model_sha256':model['model_sha256'],
            'checkpoint_manifest_sha256':'c'*64,'kind':'frozen','scientifically_qualified':False}
    return factory


def run(posterior_request,tmp_path,**kwargs):
    return w.execute_posterior_request(posterior_request,output=tmp_path/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=kwargs.pop('backend_factory',backend_factory()),**kwargs)


def test_request_plan_and_policy_are_detached_without_enabling_execution(posterior_request):
    plan=posterior_request.plan()
    assert plan['posterior_draws']==50000 and plan['gpu_jobs_submitted']==0
    assert w.PosteriorRequest.from_dict(posterior_request.to_dict()).fingerprint==posterior_request.fingerprint
    with pytest.raises(TypeError):posterior_request.model['model_sha256']='d'*64
    with pytest.raises(ValueError,match='50,000'):replace(posterior_request,posterior_draws=10000)
    with pytest.raises(ValueError):w.PosteriorRequest.from_dict(dict(posterior_request.to_dict(),device='cuda'))
    first=w.protocol_identity();first['orchestration']['quick_validation_contexts'].append(4096)
    assert w.protocol_identity()['orchestration']['quick_validation_contexts']==[16,64,256,1024]


def test_complete_file_backed_posterior_and_cache_replay(posterior_request,tmp_path):
    record=run(posterior_request,tmp_path)
    assert record['completed_draws']==50000 and not record['scientifically_qualified']
    spec,samples,query,manifest=load_samples(tmp_path/'run/posterior',expected_manifest_sha256=record['artifact_manifest_sha256'])
    assert samples.shape==(50000,2) and manifest['bank_cost']['context_forward_evaluations']<manifest['bank_cost']['shared_bank_generation_forward_evaluations']
    assert manifest['provenance']['posterior_seed']==12 and np.array_equal(query,[.3,-.2])
    np.testing.assert_allclose(encode_prior(spec,samples).mean(0),[.3,-.02],atol=.02)
    with pytest.raises(FileExistsError):run(posterior_request,tmp_path)
    def should_not_load(*a,**k):raise AssertionError('complete cache must not reload model')
    again=run(posterior_request,tmp_path,resume=True,backend_factory=should_not_load)
    assert again['reused_complete_artifact'] and again['artifact_manifest_sha256']==record['artifact_manifest_sha256']


def test_failure_keeps_complete_prefix_and_resumes_exactly(posterior_request,tmp_path):
    with pytest.raises(RuntimeError,match='interruption'):
        run(posterior_request,tmp_path,backend_factory=backend_factory(fail_at=4))
    status=json.loads((tmp_path/'run/run.json').read_text())
    assert status['status']=='failed' and status['artifact_directory'] is None
    assert not (tmp_path/'run/posterior').exists()
    with np.load(tmp_path/'run/recovery.npz',allow_pickle=False) as state:
        assert state['samples'].shape==(4096,2)
    finished=run(posterior_request,tmp_path,resume=True)
    other=tmp_path/'independent';other.mkdir()
    baseline=w.execute_posterior_request(posterior_request,output=other/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=backend_factory())
    np.testing.assert_array_equal(load_samples(tmp_path/'run/posterior')[1],load_samples(other/'run/posterior')[1])
    attempts=[json.loads(p.read_text()) for p in (tmp_path/'run/attempts').glob('*.json')]
    assert {a['status'] for a in attempts}=={'failed','samples_complete_unassessed'}
    assert finished['completed_draws']==baseline['completed_draws']==50000


def test_atomic_artifact_publication_recovers_after_writer_failure(posterior_request,tmp_path,monkeypatch):
    from posteriorforge.study import artifacts
    original=artifacts.save_samples
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        raise OSError('unit publication interruption')
    monkeypatch.setattr(artifacts,'save_samples',interrupted)
    with pytest.raises(OSError):run(posterior_request,tmp_path)
    assert not (tmp_path/'run/posterior').exists()
    assert list((tmp_path/'run').glob('posterior-staging-*'))
    monkeypatch.setattr(artifacts,'save_samples',original)
    record=run(posterior_request,tmp_path,resume=True)
    assert record['completed_draws']==50000


def test_guards_fail_before_backend_and_preserve_failure(posterior_request,tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='opt-in'):
        w.execute_posterior_request(posterior_request,output=tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()
    monkeypatch.setenv('SLURM_JOB_ID','123')
    with pytest.raises(RuntimeError,match='Slurm'):run(posterior_request,tmp_path)
    assert not (tmp_path/'run').exists()
    monkeypatch.delenv('SLURM_JOB_ID')
    (tmp_path/'query.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):run(posterior_request,tmp_path)
    assert json.loads((tmp_path/'run/run.json').read_text())['status']=='failed'


def test_standalone_test_case_sampling_requires_frozen_evaluation(posterior_request,tmp_path):
    test_spec=replace(problem(),split='test')
    receipt=save_bank(prepare_bank(test_spec,64,7),tmp_path/'test-bank')
    test_request=replace(posterior_request,bank_directory='test-bank',bank_manifest_sha256=receipt['manifest_sha256'])
    with pytest.raises(RuntimeError,match='frozen selection'):run(test_request,tmp_path)
    assert not (tmp_path/'run/posterior').exists()


def test_loaded_evaluation_variant_must_match_actual_checkpoint_metadata():
    binding={'protocol_sha256':'b'*64,'model':{'method':'lora','rank':4,'training_seed':0,'update':1000,'checkpoint_manifest_sha256':'e'*64}}
    valid={'method':'lora_r4','rank':4,'training_replica':0,'completed_updates':1000,
           'training_phase':'main','training_protocol_sha256':'b'*64,'checkpoint_manifest_sha256':'e'*64}
    w._validate_loaded_evaluation_model(binding,valid)
    for field,bad in [('method','full_finetune'),('rank',16),('training_replica',1),
                      ('completed_updates',200),('checkpoint_manifest_sha256','f'*64),('training_phase','pilot'),('training_protocol_sha256','c'*64)]:
        with pytest.raises(ValueError):w._validate_loaded_evaluation_model(binding,dict(valid,**{field:bad}))


def test_cached_artifact_rejects_even_self_consistent_manifest_edits(posterior_request,tmp_path):
    run(posterior_request,tmp_path)
    path=tmp_path/'run/posterior/manifest.json'
    receipt=json.loads(path.read_text());receipt['provenance']['extra_note']='edited'
    path.write_text(json.dumps(receipt))
    for _ in range(2):
        with pytest.raises(ValueError,match='manifest checksum'):
            run(posterior_request,tmp_path,resume=True)



def test_frozen_checkpoint_receipt_is_part_of_evaluation_identity():
    binding={'protocol_sha256':'b'*64,'model':{'method':'frozen','rank':None,'training_seed':None,
                                            'update':0,'checkpoint_manifest_sha256':'c'*64}}
    valid={'method':'frozen','rank':None,'training_replica':None,'completed_updates':0,
           'checkpoint_manifest_sha256':'c'*64}
    w._validate_loaded_evaluation_model(binding,valid)
    with pytest.raises(ValueError,match='checkpoint receipt'):
        w._validate_loaded_evaluation_model(binding,dict(valid,checkpoint_manifest_sha256='d'*64))


def test_published_artifact_retains_sampler_and_backend_implementation(posterior_request,tmp_path):
    record=run(posterior_request,tmp_path)
    _,_,_,receipt=load_samples(tmp_path/'run/posterior')
    provenance=receipt['provenance']
    assert provenance['sampler_binding']==record['timings']['binding']
    assert provenance['sampler_binding']['seed']==posterior_request.seed
    assert provenance['sampler_binding']['source_sha256']
    assert provenance['sampling_law']==provenance['sampler_binding']['law']
    assert provenance['sampling_law']=='diagnostic_backend_unspecified'


def test_completed_artifact_cannot_be_relabelled_as_a_different_selected_rank(posterior_request,tmp_path):
    from posteriorforge.study.evaluation_campaign import build_validation_plan
    from posteriorforge.study.reference import observation_digest
    spec=replace(problem(),split='validation')
    bank=save_bank(prepare_bank(spec,64,7),tmp_path/'validation-bank')
    versions=[{'version_id':'base','checkpoint_id':'base','model_sha256':'d'*64,
               'checkpoint_manifest_sha256':'e'*64,'method':'frozen','rank':None,'training_seed':None,'update':0},
              {'version_id':'trained','checkpoint_id':'trained','model_sha256':'a'*64,
               'checkpoint_manifest_sha256':'c'*64,'method':'lora','rank':16,'training_seed':0,'update':1000}]
    cases=[{'setup_id':'unit','case_id':'case','split':'validation','problem_fingerprint':spec.fingerprint,
            'observation_sha256':observation_digest(np.array([.3,-.2])),'reference_receipt_sha256':None}]
    banks=[{'setup_id':'unit','bank_seed':0,'bank_manifest_sha256':bank['manifest_sha256']},
           {'setup_id':'unit','bank_seed':1,'bank_manifest_sha256':'f'*64}]
    plan=build_validation_plan(posterior_request.protocol_sha256,versions,cases,banks,scope='diagnostic',context_sizes=(16,))
    selected=next(r for r in plan['requests'] if r['model']['version_id']=='trained' and r['bank']['bank_seed']==0)
    requested=replace(posterior_request,bank_directory='validation-bank',bank_manifest_sha256=bank['manifest_sha256'],
        model={'base_checkpoint':'unused','model_sha256':'a'*64,'campaign_checkpoint':'unused-trained',
               'checkpoint_manifest_sha256':'c'*64},seed=selected['posterior_seed'])
    def original_rank_four(model,*,base_directory):
        return Conditional(),{'model_sha256':'a'*64,'checkpoint_manifest_sha256':'c'*64,'method':'lora_r4',
            'rank':4,'training_replica':0,'completed_updates':1000,'training_phase':'main',
            'training_protocol_sha256':requested.protocol_sha256}
    run(requested,tmp_path,backend_factory=original_rank_four)
    with pytest.raises(ValueError,match='method, rank, replica or update'):
        w.execute_posterior_request(requested,output=tmp_path/'run',base_directory=tmp_path,
            allow_unqualified_cpu_diagnostic=True,resume=True,backend_factory=original_rank_four,
            evaluation_authorization={'plan':plan,'request_id':selected['request_id'],'selection_lock':None})



def test_frozen_request_cannot_supply_a_campaign_manifest(posterior_request):
    with pytest.raises(ValueError,match='omit campaign manifest'):
        replace(posterior_request,model=dict(posterior_request.model,checkpoint_manifest_sha256='c'*64))
