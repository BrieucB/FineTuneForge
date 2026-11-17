"""CPU orchestration uses injected tiny Torch backend, never a substitute PFN."""
from copy import deepcopy
from dataclasses import replace
import json,random
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from torch import nn
from posteriorforge.study.config import StudyConfig
from posteriorforge.study.spec import ProblemSpec,PriorSpec,ParameterSpec,FamilySpec,DatasetSpec
from posteriorforge.study.episodes import EpisodeStream,QuickValidationStream,BankStore
from posteriorforge.study.model import ModelQualificationError,weights_fingerprint
from posteriorforge.study.training import TrainingConfig,make_optimizer,export_full_weights
from posteriorforge.study.campaign import (CampaignSpec,METHODS,LEARNING_RATES,StockTrainingBackend,CampaignQualificationError,quick_milestones,
    plan_six_campaigns,plan_pilot_campaigns,select_method_learning_rates,select_main_allowance,shortlist_checkpoints,
    run_cpu_diagnostic,run_production_campaign,read_campaign_checkpoint,restore_campaign_checkpoint,campaign_identity)

P=StudyConfig().fingerprint

def problem(index=0):
    return ProblemSpec(f'campaign-{index}','diagnostic',(ParameterSpec('mu','population',PriorSpec('normal',.1*index,1.)),ParameterSpec('tau','population',PriorSpec('lognormal',.4,.1)),ParameterSpec('a','physical',PriorSpec('normal','mu','tau')),ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.03))),(FamilySpec('scalar','scalar','sigma'),),(DatasetSpec('data','scalar','specimen',{'a':'a'}),))

def diagnostic_spec():return CampaignSpec('lora_r4',0,'main',1e-4,4,P,effective_batch=2,diagnostic=True)

def streams(spec):
    return EpisodeStream((problem(),problem(1)),protocol_sha256=P,replica=spec.replica,phase=spec.phase,diagnostic=True),QuickValidationStream((problem(9),),protocol_sha256=P,diagnostic=True)

class TinyBackend:
    backend_kind='injected_torch_diagnostic_fixture'
    def __init__(self,spec,fail_update=None):
        self.spec=spec;self.fail_update=fail_update
        torch.manual_seed(63);self.model=nn.Linear(1,1,dtype=torch.float64)
        self.training_config=TrainingConfig(learning_rate=spec.learning_rate,max_updates=spec.updates,effective_batch=spec.effective_batch,seed=spec.initialization_seed)
        self.optimizer=make_optimizer(self.model,self.training_config);self.episode_rng=np.random.default_rng(64)
        self.identity={'backend_kind':self.backend_kind,'initial_weights_sha256':weights_fingerprint(self.model),'base_checkpoint_sha256':'b'*64}
    def update(self,items,state):
        if state.update+1==self.fail_update:raise RuntimeError('injected attempt failure')
        x=torch.tensor(np.stack([e.query.observations[0] for e in items]),dtype=torch.float64)
        y=torch.tensor([e.query.theta[0,e.query.spec.parameter_order.index('a')] for e in items],dtype=torch.float64)[:,None]
        random_factor=torch.rand((),dtype=torch.float64)+np.random.random()+random.random()+self.episode_rng.random()
        loss=(self.model(x)-y).square().mean()*(1+.01*random_factor)
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        state.update+=1;state.data_position+=len(items)
        return {'loss':float(loss.detach()),'update':state.update,'data_position':state.data_position,'scientifically_qualified':False}
    def quick_validate(self,items):
        records=[]
        with torch.no_grad():
            for item in items:
                x=torch.tensor(item.query.observations.copy(),dtype=torch.float64)
                target=torch.tensor(item.query.theta[:,item.query.spec.parameter_order.index('a')].copy(),dtype=torch.float64)[:,None]
                loss=.5*(self.model(x)-target).square().mean()
                records.append({'score':float(loss),'request_id':item.receipt['request']['request_id']})
        return {'score':sum(r['score'] for r in records)/len(records),'score_count':len(records),'records':records,'posterior_draws':0}
    def export(self,directory):
        directory=Path(directory);directory.mkdir()
        export_full_weights(directory/'fixture.safetensors',self.model)
        return {'backend_kind':self.backend_kind,'format':'diagnostic_fixture_only','path':'fixture.safetensors','weights_sha256':weights_fingerprint(self.model),'scientifically_qualified':False}


def test_planners_endpoint_lr_selection_and_common_forecast():
    assert quick_milestones(200)==tuple(range(20,201,20))
    assert quick_milestones(1000)==tuple(range(100,1001,100))
    assert len(plan_six_campaigns(protocol_sha256=P))==6
    assert len(plan_pilot_campaigns(protocol_sha256=P))==12
    records=[]
    for method in METHODS:
        low,high=LEARNING_RATES[method]
        records += [{'method':method,'replica':0,'learning_rate':low,'score':0.,'update':200},{'method':method,'replica':1,'learning_rate':low,'score':10.,'update':200},{'method':method,'replica':0,'learning_rate':high,'score':4.,'update':200},{'method':method,'replica':1,'learning_rate':high,'score':4.,'update':200}]
    chosen=select_method_learning_rates(records)
    assert all(chosen['learning_rates'][m]==LEARNING_RATES[m][1] for m in METHODS)
    with pytest.raises(ValueError):select_method_learning_rates([dict(r,update=180) for r in records])
    with pytest.raises(ValueError):select_method_learning_rates([dict(r,replica=bool(r['replica'])) for r in records])
    keys=[f'{m}/seed{r}' for m in METHODS for r in (0,1)]
    forecast=select_main_allowance(update_gpu_hours=dict.fromkeys(keys,.1),quick_check_gpu_hours=dict.fromkeys(keys,.1),pilot_gpu_hours=200.,measurement_sha256='c'*64)
    assert forecast['selected_updates']==3000
    assert forecast['forecasts'][1]['with25percent_margin_gpu_hours']==pytest.approx(2257.5)
    assert all(r['restart']=='original_pretrained_weights_and_fresh_adapter' for r in plan_six_campaigns(protocol_sha256=P,learning_rates=chosen['learning_rates'],main_updates=3000))
    assert [r['update'] for r in shortlist_checkpoints([{'quick_score':1.,'update':20,'checkpoint':'b'},{'quick_score':1.,'update':10,'checkpoint':'a'},{'quick_score':2.,'update':1,'checkpoint':'c'}])]==[10,20]


def test_end_to_end_restart_restores_model_optimizer_all_rngs_cursor_and_fixed_validation(tmp_path):
    spec=diagnostic_spec();stream,validation=streams(spec)
    full=TinyBackend(spec)
    expected=run_cpu_diagnostic(spec,stream=stream,validation=validation,store=BankStore(tmp_path/'full-banks'),backend=full,output=tmp_path/'full',allow_unqualified_cpu_diagnostic=True)
    expected_state=deepcopy(full.model.state_dict());expected_optimizer=deepcopy(full.optimizer.state_dict())
    future=(torch.rand(3),np.random.random(3),random.random(),full.episode_rng.random(3))
    part=TinyBackend(spec);store=BankStore(tmp_path/'resume-banks')
    partial=run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=part,output=tmp_path/'resume',allow_unqualified_cpu_diagnostic=True,stop_after_update=2)
    restored=TinyBackend(spec)
    actual=run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=restored,output=tmp_path/'resume',allow_unqualified_cpu_diagnostic=True,resume_from=partial['last_checkpoint'])
    assert actual['state']==expected['state'] and actual['state']['data_position']==8
    assert actual['history']==expected['history']
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in expected_state.items())
    for key,left in expected_optimizer['state'].items():
        for name,value in left.items():assert torch.equal(value,restored.optimizer.state_dict()['state'][key][name])
    assert torch.equal(future[0],torch.rand(3)) and np.array_equal(future[1],np.random.random(3))
    assert future[2]==random.random() and np.array_equal(future[3],restored.episode_rng.random(3))
    assert actual['costs']['from_scratch_accessible_forward_evaluations']>0
    assert actual['costs']['unknown_or_incomplete_cost_records']==0
    assert actual['posterior_draws']==actual['gpu_jobs_submitted']==0
    assert len(actual['shortlist'])==2 and not actual['scientifically_qualified']
    metadata=read_campaign_checkpoint(actual['last_checkpoint'])
    assert metadata['state']==actual['state'] and metadata['deployment']['format']=='diagnostic_fixture_only'
    bad=deepcopy(metadata['identity']);bad['source_sha256']='d'*64
    with pytest.raises(CampaignQualificationError,match='identity'):read_campaign_checkpoint(actual['last_checkpoint'],expected_identity=bad)
    artifact=Path(actual['last_checkpoint'])/'training'/'training.pt'
    with artifact.open('ab') as out:out.write(b'corrupt')
    with pytest.raises(CampaignQualificationError,match='checksum'):read_campaign_checkpoint(actual['last_checkpoint'])


def test_failed_attempt_data_compute_retained_and_successful_retry_uses_exact_replay(tmp_path):
    spec=diagnostic_spec();stream,validation=streams(spec);store=BankStore(tmp_path/'banks')
    backend=TinyBackend(spec,fail_update=2)
    with pytest.raises(RuntimeError,match='injected attempt'):
        run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=backend,output=tmp_path/'run',allow_unqualified_cpu_diagnostic=True)
    records=[json.loads(p.read_text()) for p in (tmp_path/'run'/'attempts').glob('*/attempt.json')]
    failed=next(r for r in records if r['status']=='failed')
    assert failed['data_events'] and failed['before']['update']==1 and failed['after']['update']==1
    assert all(p.grad is None for p in backend.model.parameters())
    restart=TinyBackend(spec)
    result=run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=restart,output=tmp_path/'run',allow_unqualified_cpu_diagnostic=True,resume_from=tmp_path/'run'/'checkpoints'/'update-000001')
    assert result['costs']['failed_or_incomplete_attempts']==1
    assert result['costs']['attempts']==5
    records=[json.loads(p.read_text()) for p in (tmp_path/'run'/'attempts').glob('*/attempt.json')]
    retried=next(r for r in records if r['status']=='complete' and r['before']['update']==1)
    assert retried['requests']==failed['requests']
    assert all(e['incurred_forward_evaluations']==0 for e in retried['data_events'])


def test_production_and_unbounded_or_implicit_execution_remain_disabled(tmp_path):
    spec=diagnostic_spec();stream,validation=streams(spec);backend=TinyBackend(spec)
    with pytest.raises(ModelQualificationError):run_production_campaign()
    with pytest.raises(ValueError):replace(spec,updates=11)
    with pytest.raises(ModelQualificationError):run_cpu_diagnostic(spec,stream=stream,validation=validation,store=BankStore(tmp_path/'banks'),backend=backend,output=tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('method',METHODS)
def test_actual_cached_model_factory_and_deployment_export_are_cpu_only(tmp_path,method):
    import os
    checkpoint=os.environ.get('FTF_CPU_CHECKPOINT')
    if not checkpoint:pytest.skip('actual cached model constructor/export fixture is opt-in; no download')
    spec=CampaignSpec(method,0,'main',LEARNING_RATES[method][0],1,P,effective_batch=1,diagnostic=True)
    backend=StockTrainingBackend.from_checkpoint_factory(spec,checkpoint)
    assert all(p.device.type=='cpu' for p in backend.model.parameters())
    assert sum(p.numel() for p in backend.model.parameters() if p.requires_grad)>0
    deployment=backend.export(tmp_path/'deployment')
    assert deployment['weights_sha256']==weights_fingerprint(backend.model)
    assert deployment['training_method']==('full_finetune' if method=='full_finetune' else 'lora')
    assert deployment['format']==('full_tabpfn_safetensors' if method=='full_finetune' else 'peft_adapter')
    assert not deployment['scientifically_qualified']


def test_planning_and_authenticated_metadata_read_work_with_torch_import_blocked(tmp_path):
    import os,subprocess,sys
    directory=tmp_path/'metadata';directory.mkdir()
    metadata={'schema':'finetuneforge.campaign_checkpoint.v1','identity':{'fixture':True},'scientifically_qualified':False}
    raw=json.dumps(metadata).encode();(directory/'campaign.json').write_bytes(raw)
    from hashlib import sha256
    (directory/'manifest.json').write_text(json.dumps({'schema':'finetuneforge.campaign_checkpoint.files.v1','files':{'campaign.json':sha256(raw).hexdigest()}}))
    code='''import importlib.abc,sys
class BlockAccelerators(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','peft','tabpfn'):raise ImportError('accelerator library forbidden in pure planner')
sys.meta_path.insert(0,BlockAccelerators())
from posteriorforge.study.campaign import plan_six_campaigns,plan_pilot_campaigns,read_campaign_checkpoint
assert len(plan_six_campaigns(protocol_sha256='a'*64))==6
assert len(plan_pilot_campaigns(protocol_sha256='a'*64))==12
assert read_campaign_checkpoint(sys.argv[1])['identity']=={'fixture':True}
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code,str(directory)],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr


def test_first_failed_attempt_can_resume_authenticated_initial_checkpoint(tmp_path):
    spec=diagnostic_spec();stream,validation=streams(spec);store=BankStore(tmp_path/'banks')
    with pytest.raises(RuntimeError):
        run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=TinyBackend(spec,fail_update=1),output=tmp_path/'run',allow_unqualified_cpu_diagnostic=True)
    initial=tmp_path/'run'/'checkpoints'/'update-000000'
    assert read_campaign_checkpoint(initial)['state']['update']==0
    result=run_cpu_diagnostic(spec,stream=stream,validation=validation,store=store,backend=TinyBackend(spec),output=tmp_path/'run',allow_unqualified_cpu_diagnostic=True,resume_from=initial)
    assert result['state']['update']==4 and result['costs']['failed_or_incomplete_attempts']==1
    assert result['costs']['initial_checkpoint_seconds']>=0


def test_pilot_qualification_cost_is_not_double_charged_to_main_and_caps_remain():
    keys=[f'{m}/seed{r}' for m in METHODS for r in (0,1)]
    kwargs={'update_gpu_hours':dict.fromkeys(keys,.0395),'quick_check_gpu_hours':dict.fromkeys(keys,.1),'measurement_sha256':'c'*64}
    result=select_main_allowance(**kwargs,pilot_gpu_hours=500.)
    assert result['selected_updates']==10000
    row=result['forecasts'][-1]
    assert row['with25percent_margin_gpu_hours']==pytest.approx(2970.)
    assert row['project_total_with_reserved_other_phases_gpu_hours']==pytest.approx(9970.)
    assert not row['pilot_included_in_main_forecast'] and row['pilot_cost_phase']=='qualification'
    assert result['qualification']['fits_envelope']
    assert sum(result['phase_envelopes_gpu_hours'].values())==result['project_cap_gpu_hours']==10000
    exceeded=select_main_allowance(**kwargs,pilot_gpu_hours=500.01)
    assert not exceeded['qualification']['fits_envelope'] and exceeded['selected_updates'] is None
    overhead=select_main_allowance(**kwargs,pilot_gpu_hours=0.,other_gpu_hours=100.)
    assert overhead['selected_updates']==3000


@pytest.mark.parametrize('field,value',[('replica',True),('replica',False),('replica',0.),('master_seed',True),('master_seed',1.),('master_seed',-1),('diagnostic',1),('diagnostic','false'),('diagnostic',None)])
def test_campaign_identity_flags_reject_boolean_integer_aliases(field,value):
    with pytest.raises(ValueError):replace(diagnostic_spec(),**{field:value})
    if field=='master_seed':
        with pytest.raises(ValueError):plan_six_campaigns(protocol_sha256=P,master_seed=value)
        with pytest.raises(ValueError):plan_pilot_campaigns(protocol_sha256=P,master_seed=value)


@pytest.mark.parametrize('mismatched',['training','validation'])
def test_campaign_identity_rejects_different_training_validation_master_seeds(mismatched):
    spec=diagnostic_spec();stream,validation=streams(spec);backend=TinyBackend(spec)
    if mismatched=='training':stream=EpisodeStream(stream.setups,protocol_sha256=P,replica=spec.replica,phase=spec.phase,master_seed=spec.master_seed+1,diagnostic=True)
    else:validation=QuickValidationStream(validation.setups,protocol_sha256=P,master_seed=spec.master_seed+1,diagnostic=True)
    with pytest.raises(ValueError,match='master-seed mismatch'):campaign_identity(spec,stream,validation,backend)


def test_actual_stock_backend_one_optimizer_unit_and_trained_deployment_two_query_roundtrip(tmp_path):
    """One real optimizer unit, not a campaign or a neural posterior request."""
    import os,time,uuid
    from dataclasses import asdict
    from posteriorforge.study.banks import prepare_bank,save_bank,array_descriptor
    from posteriorforge.study.episodes import MaterializedEpisode,digest
    from posteriorforge.study.science import encode_prior
    from posteriorforge.study.training import TrainingState,score_teacher_forced_episode,balanced_problem_loss
    from posteriorforge.study.model import CANONICAL_FULL_SUPPORT,prepare_fitted_regressor_members,canonical_full_support_mixture
    from posteriorforge.study.campaign import save_campaign_checkpoint
    from posteriorforge.study.model_loading import load_inference_model
    checkpoint=os.environ.get('FTF_CPU_CHECKPOINT')
    if not checkpoint:pytest.skip('one actual cached CPU optimizer unit is opt-in; no download or campaign')
    torch.set_num_threads(2)
    evidence_root=os.environ.get('FTF_CPU_UNIT_EVIDENCE_DIR')
    directory=(Path(evidence_root)/('unit-'+uuid.uuid4().hex)) if evidence_root else tmp_path/'unit-evidence'
    directory.mkdir(parents=True)
    started=time.perf_counter();spec=CampaignSpec('lora_r4',0,'main',1e-4,1,P,effective_batch=1,diagnostic=True)
    scientific=problem()
    context=prepare_bank(scientific,16,2026111701);query=prepare_bank(scientific,1,2026111702)
    context_manifest=save_bank(context,directory/'context-bank');query_manifest=save_bank(query,directory/'query-bank')
    data={'scope':'direct_materialized_unit_fixture_not_episode_stream_campaign','context_worlds':16,'independent_training_query_cases':1,'unknowns':4,'roles':list(scientific.roles),'context_manifest_sha256':context_manifest['manifest_sha256'],'query_manifest_sha256':query_manifest['manifest_sha256'],'context_cost':context.manifest['cost'],'query_cost':query.manifest['cost'],'query_theta':array_descriptor(query.theta),'query_observations':array_descriptor(query.observations),'extra_prediction_query':'fixed observation-only perturbation for parity; no simulator evaluation or generating label used'}
    item=MaterializedEpisode(context,query,data)
    backend=StockTrainingBackend.from_checkpoint_factory(spec,checkpoint)
    state=TrainingState(stream_id=digest({'fixture':'single-real-optimizer-unit','data':data['query_theta'],'context':context_manifest['theta']}))
    before=weights_fingerprint(backend.model)
    original_trainable={name:p.detach().clone() for name,p in backend.model.named_parameters() if p.requires_grad}
    checked=time.perf_counter();prepared=backend.prepare(item)
    with torch.no_grad():
        score=score_teacher_forced_episode(backend.model,prepared,law=CANONICAL_FULL_SUPPORT)
        expected=float(balanced_problem_loss(list(score.parameter_losses.unbind(0)),[score.roles]))
    del prepared,score
    expected_score_seconds=time.perf_counter()-checked
    print(json.dumps({'event':'real_unit_update_started','context':16,'query_cases':1,'unknowns':4,'rank':4,'posterior_draws':0}),flush=True)
    updated=time.perf_counter();report=backend.update([item],state);unit_seconds=time.perf_counter()-updated
    after=weights_fingerprint(backend.model)
    assert np.isfinite(report['loss']) and report['loss']==pytest.approx(expected,abs=2e-6,rel=2e-6)
    assert state.update==1 and state.data_position==1 and report['optimizer_steps']==1
    assert report['backward_calls']==4 and np.isfinite(report['gradient_norm_before_clip'])
    deltas={name:float((p.detach()-original_trainable[name]).abs().max()) for name,p in backend.model.named_parameters() if name in original_trainable}
    changed=sum(value>0 for value in deltas.values())
    assert changed>0 and before!=after
    print(json.dumps({'event':'real_unit_update_complete','seconds':unit_seconds,'loss':report['loss'],'expected_loss':expected,'updated_trainable_tensors':changed}),flush=True)
    roundtrip=time.perf_counter()
    # The campaign checkpoint container is reused only as an authenticated
    # deployment/restart format; no campaign/stream execution is claimed.
    identity={'spec':spec.to_dict(),'fixture_scope':'one_optimizer_unit_only','training_config':asdict(backend.training_config),'backend':backend.identity,'stream_id':state.stream_id}
    artifact=directory/'unit-checkpoint'
    manifest_sha=save_campaign_checkpoint(artifact,backend,state,identity=identity,history=[])
    loaded,load_report=load_inference_model(checkpoint,expected_model_sha256=after,campaign_checkpoint=artifact,expected_checkpoint_manifest_sha256=manifest_sha)
    assert weights_fingerprint(loaded)==after
    # Same evaluation flags avoid comparing different CPU Linear dispatch paths.
    backend.model.requires_grad_(False);backend.model.eval()
    reg=backend.regressor_factory();reg.fit(context.observations.copy(),encode_prior(scientific,context.theta)[:,0])
    two_queries=np.concatenate((query.observations,query.observations+.03),axis=0)
    ensemble=prepare_fitted_regressor_members(reg,two_queries,source_policy='cancelled_zero_width')
    with torch.no_grad():
        left=ensemble.adjusted_logits(backend.model);right=ensemble.adjusted_logits(loaded)
        assert len(left)==len(right)==8 and all(value.shape==(2,5000) for value in left)
        differences=[]
        for a,b in zip(left,right):
            assert torch.equal(torch.isneginf(a),torch.isneginf(b))
            finite=torch.isfinite(a)&torch.isfinite(b)
            differences.append(float((a[finite]-b[finite]).abs().max()))
            assert torch.equal(a,b)
        d1=canonical_full_support_mixture(left,[m.source_borders for m in ensemble.members],ensemble.canonical_borders,sampling_borders=ensemble.sampling_borders)
        d2=canonical_full_support_mixture(right,[m.source_borders for m in ensemble.members],ensemble.canonical_borders,sampling_borders=ensemble.sampling_borders)
        assert torch.equal(d1.weights,d2.weights)
        cdf_delta=float((d1.cdf(torch.tensor([0.,.25]))-d2.cdf(torch.tensor([0.,.25]))).abs().max())
        assert cdf_delta==0
    receipt={'schema':'finetuneforge.actual_one_optimizer_cpu_unit.v1','scope':'one actual LoRA-r4 optimizer unit; no campaign, deployment upload or posterior generation','data':data,'state':asdict(state),'expected_role_balanced_physical_nll':expected,'actual_role_balanced_physical_nll':report['loss'],'absolute_loss_difference':abs(expected-report['loss']),'report':report,'updated_trainable_tensors':changed,'maximum_trainable_parameter_change':max(deltas.values()),'weights_before':before,'weights_after':after,'checkpoint_manifest_sha256':manifest_sha,'checkpoint_directory':str(artifact),'reload':load_report,'prediction_parameter_stage':0,'conditional_prediction_queries':2,'all_eight_members_retained':True,'max_logit_roundtrip_difference':max(differences),'max_cdf_roundtrip_difference':cdf_delta,'timings':{'expected_score_seconds':expected_score_seconds,'optimizer_unit_seconds':unit_seconds,'deployment_roundtrip_and_prediction_seconds':time.perf_counter()-roundtrip,'total_cpu_wall_seconds':time.perf_counter()-started},'torch_version':torch.__version__,'torch_cuda_build':torch.version.cuda,'cpu_threads':torch.get_num_threads(),'posterior_draws':0,'gpu_jobs_submitted':0,'production_training_run':False,'scientifically_qualified':False}
    (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'event':'real_unit_fixture_complete','receipt':str(directory/'receipt.json'),'seconds':receipt['timings']['total_cpu_wall_seconds'],'max_logit_delta':max(differences),'max_cdf_delta':cdf_delta}),flush=True)
