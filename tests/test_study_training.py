from copy import deepcopy
import random
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from torch import nn
from posteriorforge.study.model import LEGACY_FINITE,CANONICAL_FULL_SUPPORT,ModelQualificationError
from posteriorforge.study.training import (TrainingConfig,TrainingState,TargetSupportMismatch,balanced_problem_loss,conditional_nll,make_optimizer,train_update,save_training_checkpoint,load_training_checkpoint,export_full_weights)


def test_role_balance_and_equal_problem_influence():
    a=torch.tensor([3.,3.,6.,9.],requires_grad=True)
    # Population and batch belong to the same group; duplicating physicals does not dominate.
    b=torch.tensor([3.,6.,6.,6.,9.],requires_grad=True)
    loss=balanced_problem_loss([a,b],[['population','batch','physical','noise'],['population','physical','physical','physical','noise']])
    assert loss.item()==6.
    loss.backward();assert torch.isfinite(a.grad).all()
    assert b.grad[1:4].sum()==pytest.approx(b.grad[0].item())


def test_score_has_width_jacobian_and_support_not_classifier_cross_entropy():
    logits=torch.zeros(1,2,requires_grad=True)
    result=conditional_nll(logits,torch.tensor([0.,1.,3.]),torch.tensor([2.]),law=LEGACY_FINITE,log_abs_det_decode=np.log(3))
    assert result.item()==pytest.approx(np.log(12))
    with pytest.raises(TargetSupportMismatch):conditional_nll(logits,torch.tensor([0.,1.,3.]),torch.tensor([4.]),law=LEGACY_FINITE)
    with pytest.raises(TargetSupportMismatch):balanced_problem_loss([torch.tensor([0.,float('inf'),0.])],[['population','physical','noise']])
    with pytest.raises(ValueError):balanced_problem_loss([torch.zeros(2)],[['physical','noise']])


def test_production_qualification_cannot_be_asserted_by_config_boolean():
    with pytest.raises(ModelQualificationError):TrainingConfig().require_production_qualification()
    with pytest.raises(ModelQualificationError):TrainingConfig(sampler_law_qualified=True).validate()


def _losses(model,generator):
    x=torch.randn(3,2)+torch.from_numpy(generator.normal(size=(3,2))).float()
    target=torch.tensor([.1,.2,.3]);loss=(model(x).squeeze(-1)-target).square()
    return [loss],[['population','physical','noise']]


def test_checkpoint_continuation_restores_optimizer_rng_and_episode_position(tmp_path):
    torch.manual_seed(71);np.random.seed(72);random.seed(73)
    config=TrainingConfig(max_updates=2);model=nn.Linear(2,1)
    optimizer=make_optimizer(model,config);state=TrainingState(stream_id='same-stream');generator=np.random.default_rng(74)
    losses,roles=_losses(model,generator);train_update(model,optimizer,losses,roles,state)
    save_training_checkpoint(tmp_path/'checkpoint',model,optimizer,state,config=config,episode_rng=generator)
    losses,roles=_losses(model,generator);expected=train_update(model,optimizer,losses,roles,state)
    parameters=deepcopy(model.state_dict());future=(torch.rand(2),np.random.rand(2),random.random(),generator.normal(size=2))
    fresh=nn.Linear(2,1);new_optimizer=make_optimizer(fresh,config);new_generator=np.random.default_rng(0)
    resumed=load_training_checkpoint(tmp_path/'checkpoint',fresh,new_optimizer,config=config,expected_stream_id='same-stream',episode_rng=new_generator)
    assert resumed.update==1 and resumed.data_position==1
    losses,roles=_losses(fresh,new_generator);actual=train_update(fresh,new_optimizer,losses,roles,resumed)
    assert actual['loss']==expected['loss'] and resumed.update==2 and resumed.data_position==2
    assert all(torch.equal(value,fresh.state_dict()[name]) for name,value in parameters.items())
    actual_future=(torch.rand(2),np.random.rand(2),random.random(),new_generator.normal(size=2))
    assert torch.equal(future[0],actual_future[0])
    assert np.array_equal(future[1],actual_future[1]) and future[2]==actual_future[2] and np.array_equal(future[3],actual_future[3])
    with pytest.raises(ModelQualificationError):load_training_checkpoint(tmp_path/'checkpoint',fresh,new_optimizer,config=config,expected_stream_id='other',episode_rng=new_generator)
    with pytest.raises(FileExistsError):save_training_checkpoint(tmp_path/'checkpoint',fresh,new_optimizer,resumed,config=config,episode_rng=new_generator)


def test_corrupt_checkpoint_and_disconnected_loss_refuse_updates(tmp_path):
    config=TrainingConfig();model=nn.Linear(2,1);optimizer=make_optimizer(model,config);state=TrainingState(stream_id='a')
    before=deepcopy(model.state_dict())
    with pytest.raises(ModelQualificationError):train_update(model,optimizer,[torch.ones(3)],[['population','physical','noise']],state)
    assert state.update==0 and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    save_training_checkpoint(tmp_path/'checkpoint',model,optimizer,state,config=config)
    with (tmp_path/'checkpoint'/'rng.json').open('a') as stream:stream.write('corrupt')
    with pytest.raises(ModelQualificationError,match='checksum'):load_training_checkpoint(tmp_path/'checkpoint',model,optimizer,config=config,expected_stream_id='a')


def test_full_checkpoint_export_uses_safetensors(tmp_path):
    pytest.importorskip('safetensors')
    from safetensors.torch import load_file
    model=nn.Linear(2,1);manifest=export_full_weights(tmp_path/'model.safetensors',model)
    loaded=load_file(tmp_path/'model.safetensors')
    assert all(torch.equal(v,loaded[k]) for k,v in model.state_dict().items())
    assert manifest['scientifically_qualified'] is False


def _teacher_banks(split='train', specimens=1, query_count=2):
    from posteriorforge.study.spec import ProblemSpec,ParameterSpec,PriorSpec,FamilySpec,DatasetSpec
    from posteriorforge.study.banks import prepare_bank
    parameters=[ParameterSpec('mean','population',PriorSpec('normal',0.,1.)),ParameterSpec('spread','population',PriorSpec('lognormal',.4,.1))]
    parameters += [ParameterSpec(f'a{i}','physical',PriorSpec('normal','mean','spread')) for i in range(specimens)]
    parameters += [ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.03))]
    spec=ProblemSpec(f'teacher-{specimens}',split,tuple(parameters),(FamilySpec('readout','scalar','sigma'),),tuple(DatasetSpec(f'd{i}','readout',f's{i}',{'a':f'a{i}'}) for i in range(specimens)))
    return prepare_bank(spec,16,2026111701),prepare_bank(spec,query_count,2026111702)


@pytest.mark.parametrize('specimens',[1,2])
def test_teacher_forcing_uses_only_observations_and_true_predecessors(specimens):
    from posteriorforge.study.training import teacher_forced_inputs
    from posteriorforge.study.science import encode_prior,log_abs_det_decode
    context,query=_teacher_banks(specimens=specimens)
    actual=teacher_forced_inputs(context,query)
    cz=encode_prior(context.spec,context.theta);qz=encode_prior(query.spec,query.theta)
    jac=log_abs_det_decode(query.spec,qz,per_parameter=True)
    assert len(actual)==query.spec.dimension
    for j,stage in enumerate(actual):
        assert stage.context_features.shape==(16,context.spec.observation_dimension+j)
        assert stage.query_features.shape==(len(query.theta),query.spec.observation_dimension+j)
        assert np.array_equal(stage.context_features[:,:context.spec.observation_dimension],context.observations)
        assert np.array_equal(stage.query_features[:,query.spec.observation_dimension:],qz[:,:j])
        assert np.array_equal(stage.context_targets,cz[:,j])
        assert np.array_equal(stage.query_targets,qz[:,j])
        assert np.array_equal(stage.decoder_log_jac,jac[:,j])
        assert not stage.query_features.flags.writeable
    assert np.allclose(np.stack([x.decoder_log_jac for x in actual],-1).sum(-1),log_abs_det_decode(query.spec,qz))


def test_teacher_forcing_refuses_context_leakage_and_final_test_cases():
    from posteriorforge.study.training import teacher_forced_inputs
    context,query=_teacher_banks()
    with pytest.raises(ModelQualificationError,match='occur in the context'):teacher_forced_inputs(context,context.prefix(1))
    context,query=_teacher_banks(split='test')
    with pytest.raises(ModelQualificationError,match='test cases'):teacher_forced_inputs(context,query)


def test_optimizer_rejects_validation_test_scores_before_backward():
    from posteriorforge.study.training import TeacherForcedScore,train_teacher_forced_update
    model=nn.Linear(2,4);optimizer=make_optimizer(model,TrainingConfig());state=TrainingState(stream_id='unit')
    losses=model(torch.ones(2,2)).square()
    before=deepcopy(model.state_dict());roles=('population','population','physical','noise')
    for split in ('validation','test','diagnostic'):
        score=TeacherForcedScore('fixture','fingerprint',split,roles,losses,LEGACY_FINITE)
        with pytest.raises(ModelQualificationError):train_teacher_forced_update(model,optimizer,[score],state)
        assert state.update==0 and all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    score=TeacherForcedScore('fixture','fingerprint','diagnostic',roles,losses,LEGACY_FINITE)
    result=train_teacher_forced_update(model,optimizer,[score],state,allow_diagnostic_cpu=True)
    assert result['update']==1 and result['data_position']==2


@pytest.mark.parametrize("law,context_count",[(LEGACY_FINITE,64),(CANONICAL_FULL_SUPPORT,16)])
def test_real_teacher_forced_episode_uses_all_parameters_and_stock_preprocessors(monkeypatch,law,context_count):
    import os
    from posteriorforge.study.model import TabPFNBackbone
    from posteriorforge.study.training import prepare_teacher_forced_episode,score_teacher_forced_episode
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real cached-checkpoint episode is opt-in, with no download')
    from tabpfn import TabPFNRegressor
    context,query=_teacher_banks(split='diagnostic',query_count=1)
    # Same nested prior bank: B16 has retained cancelled duplicate-border
    # evidence; B64 is the smallest tested strictly valid all-stage fixture.
    from posteriorforge.study.banks import prepare_bank
    larger=prepare_bank(context.spec,context_count,2026111701)
    assert np.array_equal(larger.theta[:16],context.theta)
    context=larger
    def factory():
        return TabPFNRegressor(model_path=path,device='cpu',n_estimators=8,random_state=0,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    prepared=prepare_teacher_forced_episode(context,query,regressor_factory=factory,law=law)
    assert len(prepared.stages)==context.spec.dimension==4
    assert all(len(stage.ensemble.members)==8 for stage in prepared.stages)
    network=TabPFNBackbone.from_checkpoint(path)
    from posteriorforge.study.model import BarDensity
    original_log_prob=BarDensity.log_prob
    witnessed=[]
    def witness(density,values,*,log_abs_det_decode=0.0):
        witnessed.append((-original_log_prob(density,values),torch.as_tensor(log_abs_det_decode)))
        return original_log_prob(density,values,log_abs_det_decode=log_abs_det_decode)
    monkeypatch.setattr(BarDensity,'log_prob',witness)
    with torch.no_grad():score=score_teacher_forced_episode(network,prepared,law=law)
    latent_losses=torch.stack([item[0] for item in witnessed],-1)
    jac=torch.stack([stage.decoder_log_jac for stage in prepared.stages],-1)
    assert all(torch.equal(item[1],stage.decoder_log_jac) for item,stage in zip(witnessed,prepared.stages))
    assert score.parameter_losses.shape==(1,4) and torch.isfinite(score.parameter_losses).all()
    assert torch.allclose(score.parameter_losses,latent_losses+jac,atol=1e-13,rtol=1e-13)
    assert torch.allclose(score.physical_log_prob,-score.parameter_losses.sum(-1))
    assert score.split=='diagnostic' and score.density_law==law
    assert prepared.source_policy==('legacy_strict' if law==LEGACY_FINITE else 'cancelled_zero_width')


def test_real_b16_default_source_failure_is_retained_and_zero_mass_primitive_preserves_heads():
    import os
    from posteriorforge.study.model import TabPFNBackbone,prepare_fitted_regressor_members,full_support_cdf_grid,canonical_full_support_mixture,attach_lora
    from posteriorforge.study.training import teacher_forced_inputs
    from posteriorforge import execution_probability_v3 as legacy_policy
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real source-grid negative fixture is opt-in, with no download')
    from tabpfn import TabPFNRegressor
    context,query=_teacher_banks(split='diagnostic',query_count=1)
    stage=teacher_forced_inputs(context,query)[1]
    reg=TabPFNRegressor(model_path=path,device='cpu',n_estimators=8,random_state=0,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    reg.fit(stage.context_features.copy(),stage.context_targets.copy())
    with pytest.raises(ModelQualificationError,match='non-strict source'):prepare_fitted_regressor_members(reg,stage.query_features.copy())
    prepared=prepare_fitted_regressor_members(reg,stage.query_features.copy(),source_policy='cancelled_zero_width')
    assert len(prepared.members)==8
    network=TabPFNBackbone(reg.executor_.model,reg.bardist_.borders,checkpoint_sha256='fixture-cached-base')
    network.backbone.requires_grad_(False)
    network.backbone=attach_lora(network.backbone,rank=4)
    grid=torch.tensor([-1500.,-2.,0.,2.,1500.],dtype=torch.float64)
    outputs=prepared.adjusted_logits(network)
    for value in outputs:value.retain_grad()
    collapsed_members=[]
    for i,(member,logits) in enumerate(zip(prepared.members,outputs)):
        zero=torch.diff(member.source_borders)==0
        if zero.any():
            collapsed_members.append(i)
            assert int(zero.sum())==213 and torch.isneginf(logits[...,zero]).all()
            with pytest.raises(RuntimeError,match='strict finite original'):legacy_policy.from_adjusted_logits(logits,member.source_borders,prepared.canonical_borders,prepared.sampling_borders)
        cdf,sf=full_support_cdf_grid(logits,member.source_borders,grid)
        assert torch.isfinite(cdf).all() and torch.isfinite(sf).all()
        assert torch.allclose(cdf+sf,torch.ones_like(cdf),atol=2e-14,rtol=2e-14)
    assert collapsed_members==[2,3,6,7]
    candidate=canonical_full_support_mixture(outputs,[m.source_borders for m in prepared.members],prepared.canonical_borders,sampling_borders=prepared.sampling_borders)
    loss=-candidate.log_prob(torch.tensor(stage.query_targets,dtype=torch.float64)).mean()
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in outputs)
    gradients={n:p.grad for n,p in network.backbone.named_parameters() if p.requires_grad}
    assert all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert any(bool((g!=0).any()) for n,g in gradients.items() if 'lora_B' in n and '.parametrizations.' in n)


def test_selected_full_tail_default_retains_guard_and_pending_cannot_fallback():
    from posteriorforge.study.model import PENDING_FULL_TAIL
    from posteriorforge.study.training import prepare_teacher_forced_episode
    assert TrainingConfig().density_law==CANONICAL_FULL_SUPPORT
    with pytest.raises(ModelQualificationError):TrainingConfig().require_production_qualification()
    config=TrainingConfig(density_law=PENDING_FULL_TAIL)
    assert config.density_law==PENDING_FULL_TAIL
    config.validate()  # Hyperparameter/metadata inspection does not execute a law.
    with pytest.raises(ModelQualificationError):config.require_production_qualification()
    with pytest.raises(ModelQualificationError,match='no finite fallback'):
        conditional_nll(torch.zeros(1,2),torch.tensor([0.,1.,2.]),torch.tensor([.5]),law=config.density_law)
    context,query=_teacher_banks()
    called=[]
    def factory():called.append(True);raise AssertionError('pending must fail before any model access')
    with pytest.raises(ModelQualificationError,match='no finite fallback'):
        prepare_teacher_forced_episode(context,query,regressor_factory=factory,law=config.density_law)
    assert not called
