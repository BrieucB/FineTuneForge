"""Optimizer-objective/lifecycle fixtures, not replacement foundation models."""
from copy import deepcopy
from dataclasses import replace
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from posteriorforge.study.model import BarDensity,CANONICAL_FULL_SUPPORT,LEGACY_FINITE,ModelQualificationError
from posteriorforge.study.training import (PreparedTeacherForcedEpisode,PreparedTeacherForcedStage,
    TrainingConfig,TrainingState,make_optimizer,score_teacher_forced_episode,train_teacher_forced_update)
from posteriorforge.study.updates import stagewise_cpu_update


class Tracker:
    def __init__(self,enforce=False):
        self.enforce=enforce;self.events=[];self.active=0;self.maximum=0;self.steps=0;self.initial=None


class BackwardWitness(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,tracker,label,bad_gradient):
        if tracker.enforce:assert tracker.active==0,'next conditional started before prior backward'
        tracker.active+=1;tracker.maximum=max(tracker.maximum,tracker.active)
        tracker.events.append(('forward',label))
        ctx.tracker=tracker;ctx.label=label;ctx.bad_gradient=bad_gradient
        return value
    @staticmethod
    def backward(ctx,gradient):
        ctx.tracker.events.append(('backward',ctx.label));ctx.tracker.active-=1
        return (gradient*float('nan') if ctx.bad_gradient else gradient),None,None,None


class FixtureEnsemble:
    """Tiny density fixture for accumulation algebra; no model-substitution claim."""
    def __init__(self,features,tracker,label,failure=None):
        self.features=features;self.tracker=tracker;self.label=label;self.failure=failure
    def density(self,model,*,law):
        if self.tracker.initial is None:self.tracker.initial=deepcopy(model.state_dict())
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in self.tracker.initial.items()),'weights changed before final optimizer step'
        if self.failure=='forward':raise ModelQualificationError('injected conditional failure')
        logits=torch.zeros((len(self.features),3),dtype=torch.float64) if self.failure=='disconnected' else model(self.features)
        logits=BackwardWitness.apply(logits,self.tracker,self.label,self.failure=='gradient')
        return BarDensity(logits,torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64),law=law)


def fixtures(tracker,*,law=CANONICAL_FULL_SUPPORT,query_counts=(1,3),dimensions=(4,6),failure=None):
    generator=torch.Generator().manual_seed(314)
    episodes=[]
    for i,(queries,dimension) in enumerate(zip(query_counts,dimensions)):
        roles=('population','batch')+('physical',)*(dimension-3)+('noise',)
        names=tuple(f'p{j}' for j in range(dimension));stages=[]
        for j,role in enumerate(roles):
            key=f'{i}/{j}';bad=failure[1] if failure is not None and failure[0]==key else None
            ensemble=FixtureEnsemble(torch.randn(queries,2,generator=generator,dtype=torch.float64),tracker,key,bad)
            targets=torch.linspace(-1.5,1.5,queries,dtype=torch.float64)+.08*j
            jac=torch.linspace(-.3,.4,queries,dtype=torch.float64)+.07*j
            stages.append(PreparedTeacherForcedStage(j,names[j],role,ensemble,targets,jac))
        episodes.append(PreparedTeacherForcedEpisode(f'unit-{i}',f'fp-{i}','train',names,roles,16,queries,law,'legacy_strict' if law==LEGACY_FINITE else 'cancelled_zero_width',tuple(stages)))
    return tuple(episodes)


def model_and_optimizer():
    torch.manual_seed(85)
    model=nn.Linear(2,3,dtype=torch.float64)
    return model,make_optimizer(model,TrainingConfig(learning_rate=1e-3))


def count_step(optimizer,tracker,monkeypatch):
    original=optimizer.step
    def step(*args,**kwargs):
        assert tracker.active==0
        tracker.steps+=1;tracker.events.append(('optimizer','batch'))
        return original(*args,**kwargs)
    monkeypatch.setattr(optimizer,'step',step)


@pytest.mark.parametrize('law',[LEGACY_FINITE,CANONICAL_FULL_SUPPORT])
def test_stagewise_matches_existing_full_graph_loss_gradients_optimizer_and_unequal_queries(law,monkeypatch):
    reference,reference_optimizer=model_and_optimizer();actual,actual_optimizer=model_and_optimizer()
    reference_tracker=Tracker();actual_tracker=Tracker(enforce=True)
    reference_episodes=fixtures(reference_tracker,law=law)
    actual_episodes=fixtures(actual_tracker,law=law)
    reference_state=TrainingState(update=4,data_position=7,stream_id='matched')
    actual_state=TrainingState(update=4,data_position=7,stream_id='matched')
    scores=[score_teacher_forced_episode(reference,episode,law=law) for episode in reference_episodes]
    expected=train_teacher_forced_update(reference,reference_optimizer,scores,reference_state,gradient_clip=.1)
    count_step(actual_optimizer,actual_tracker,monkeypatch)
    observed=stagewise_cpu_update(actual,actual_optimizer,actual_episodes,actual_state,law=law,gradient_clip=.1,allow_unqualified_cpu_diagnostic=True)
    assert observed['loss']==pytest.approx(expected['loss'],abs=3e-14,rel=3e-14)
    assert observed['gradient_norm_before_clip']==pytest.approx(expected['gradient_norm_before_clip'],abs=3e-14,rel=3e-14)
    for (name,left),(_,right) in zip(reference.named_parameters(),actual.named_parameters()):
        assert torch.allclose(left.grad,right.grad,atol=3e-14,rtol=3e-14),name
        assert torch.allclose(left,right,atol=3e-14,rtol=3e-14),name
    for key,left in reference_optimizer.state_dict()['state'].items():
        for name,value in left.items():
            right=actual_optimizer.state_dict()['state'][key][name]
            if isinstance(value,torch.Tensor):assert torch.allclose(value,right,atol=3e-14,rtol=3e-14)
            else:assert value==right
    assert actual_state==reference_state and actual_state.update==5 and actual_state.data_position==11
    assert observed['episode_count']==2 and observed['query_case_count']==4
    assert observed['backward_calls']==observed['conditional_count']==10
    assert observed['optimizer_steps']==actual_tracker.steps==1
    assert actual_tracker.maximum==1 and actual_tracker.active==0
    assert actual_tracker.events[-1]==('optimizer','batch')
    expected_events=[(kind,f'{i}/{j}') for i,d in enumerate((4,6)) for j in range(d) for kind in ('forward','backward')]
    assert actual_tracker.events[:-1]==expected_events
    assert not observed['scientifically_qualified']


@pytest.mark.parametrize('split',['validation','test'])
def test_preflight_rejects_any_validation_test_before_forward_and_clears_stale_gradients(split,monkeypatch):
    model,optimizer=model_and_optimizer();tracker=Tracker(enforce=True);episodes=fixtures(tracker)
    episodes=(episodes[0],replace(episodes[1],split=split))
    state=TrainingState(update=2,data_position=5);before=deepcopy(model.state_dict())
    for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
    count_step(optimizer,tracker,monkeypatch)
    with pytest.raises(ModelQualificationError,match='validation/test'):
        stagewise_cpu_update(model,optimizer,episodes,state,law=CANONICAL_FULL_SUPPORT,allow_unqualified_cpu_diagnostic=True)
    assert not tracker.events and tracker.steps==0
    assert state.update==2 and state.data_position==5
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())


@pytest.mark.parametrize('failure',['forward','gradient','disconnected'])
def test_failed_conditionals_clear_accumulation_without_step_or_counter_change(failure,monkeypatch):
    model,optimizer=model_and_optimizer();tracker=Tracker(enforce=True)
    episodes=fixtures(tracker,failure=('1/1',failure));state=TrainingState(update=3,data_position=8)
    before=deepcopy(model.state_dict());count_step(optimizer,tracker,monkeypatch)
    with pytest.raises(ModelQualificationError):
        stagewise_cpu_update(model,optimizer,episodes,state,law=CANONICAL_FULL_SUPPORT,allow_unqualified_cpu_diagnostic=True)
    assert any(kind=='backward' for kind,_ in tracker.events)
    assert tracker.steps==0 and not optimizer.state
    assert state.update==3 and state.data_position==8
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())


def test_opt_in_and_cpu_device_are_mandatory_without_production_fallback(monkeypatch):
    model,optimizer=model_and_optimizer();tracker=Tracker(enforce=True);episodes=fixtures(tracker)
    with pytest.raises(ModelQualificationError,match='opt-in'):
        stagewise_cpu_update(model,optimizer,episodes,TrainingState(),law=CANONICAL_FULL_SUPPORT)
    assert not tracker.events
    # Meta tensors test the device guard without CUDA initialization/access.
    model.register_buffer('non_cpu_fixture',torch.empty(1,device='meta'))
    with pytest.raises(ModelQualificationError,match='CPU-only'):
        stagewise_cpu_update(model,optimizer,episodes,TrainingState(),law=CANONICAL_FULL_SUPPORT,allow_unqualified_cpu_diagnostic=True)
    assert not tracker.events


def test_two_shards_sum_global_weighted_gradients_before_clipping():
    from posteriorforge.study.updates import _stagewise_update

    reference, reference_optimizer = model_and_optimizer()
    ranks = [model_and_optimizer(), model_and_optimizer()]
    states = [TrainingState(), TrainingState()]
    reference_report = stagewise_cpu_update(
        reference, reference_optimizer, fixtures(Tracker()), TrainingState(),
        law=CANONICAL_FULL_SUPPORT, gradient_clip=.1, allow_unqualified_cpu_diagnostic=True,
    )
    assert reference_report['gradient_norm_before_clip'] > .1
    contributions = []
    rhs_reports = []

    def collect_rank_zero(left):
        def sum_for_rank_one(right):
            for left_parameter, right_parameter in zip(left, right, strict=True):
                left_parameter.grad.add_(right_parameter.grad)
                right_parameter.grad.copy_(left_parameter.grad)

        rhs_reports.append(_stagewise_update(
            *ranks[1], (fixtures(Tracker())[1],), states[1],
            law=CANONICAL_FULL_SUPPORT, gradient_clip=.1, allow_unqualified_cpu_diagnostic=True,
            global_query_cases=4, sum_rank_gradients=sum_for_rank_one,
        ))
        contributions.append(1)

    lhs = _stagewise_update(
        *ranks[0], (fixtures(Tracker())[0],), states[0],
        law=CANONICAL_FULL_SUPPORT, gradient_clip=.1, allow_unqualified_cpu_diagnostic=True,
        global_query_cases=4, sum_rank_gradients=collect_rank_zero,
    )
    rhs, = rhs_reports
    assert contributions == [1]
    assert lhs['loss'] + rhs['loss'] == pytest.approx(reference_report['loss'], abs=3e-14)
    assert lhs['gradient_norm_before_clip'] == pytest.approx(reference_report['gradient_norm_before_clip'], abs=3e-14)
    assert [r['local_query_case_count'] for r in (lhs, rhs)] == [1, 3]
    assert [r['backward_calls'] for r in (lhs, rhs)] == [4, 6]
    assert states == [TrainingState(update=1, data_position=4)] * 2
    for model, optimizer in ranks:
        for expected, actual in zip(reference.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-14, rtol=3e-14)
            for key, tensor in reference_optimizer.state[expected].items():
                torch.testing.assert_close(optimizer.state[actual][key], tensor, atol=3e-14, rtol=3e-14)


def test_split_case_preserves_full_role_weights_and_the_global_case_denominator():
    from posteriorforge.study.updates import _stagewise_update

    model, optimizer = model_and_optimizer()
    a, b = fixtures(Tracker())
    baseline = stagewise_cpu_update(model, optimizer, (a, b), TrainingState(), law=CANONICAL_FULL_SUPPORT,
                                    gradient_clip=.1, allow_unqualified_cpu_diagnostic=True)
    peers = [model_and_optimizer(), model_and_optimizer()]
    states = [TrainingState(), TrainingState()]
    first_a, _ = fixtures(Tracker())
    second_a, second_b = fixtures(Tracker())
    left = (replace(first_a, stages=(first_a.stages[1],)),)
    right = (replace(second_a, stages=second_a.stages[:1]+second_a.stages[2:]), second_b)
    options = dict(law=CANONICAL_FULL_SUPPORT, gradient_clip=.1, allow_unqualified_cpu_diagnostic=True,
                   global_query_cases=4, allow_conditional_shards=True)
    saved = []

    def reduce_left(left_parameters):
        def reduce_right(right_parameters):
            for left, right in zip(left_parameters, right_parameters, strict=True):
                left.grad.add_(right.grad)
                right.grad.copy_(left.grad)
        saved.append(_stagewise_update(*peers[1], right, states[1], sum_rank_gradients=reduce_right, **options))

    first = _stagewise_update(*peers[0], left, states[0], sum_rank_gradients=reduce_left, **options)
    second, = saved
    assert first['conditional_count'] == 1 and second['conditional_count'] == 9
    assert first['loss'] + second['loss'] == pytest.approx(baseline['loss'], abs=3e-14)
    assert first['gradient_norm_before_clip'] == pytest.approx(baseline['gradient_norm_before_clip'], abs=3e-14)
    assert states == [TrainingState(update=1, data_position=4)] * 2
    for actual_model, _ in peers:
        for expected, actual in zip(model.parameters(), actual_model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-14, rtol=3e-14)


def test_regular_update_refuses_partial_cases():
    model, optimizer = model_and_optimizer()
    case, _ = fixtures(Tracker())
    state = TrainingState()
    with pytest.raises(ModelQualificationError, match='incomplete ordered'):
        stagewise_cpu_update(model, optimizer, (replace(case, stages=case.stages[1:]),), state,
                             law=CANONICAL_FULL_SUPPORT, allow_unqualified_cpu_diagnostic=True)
    assert state == TrainingState()


def test_sampled_role_weights_target_the_exhaustive_preclip_gradient():
    from itertools import combinations, product
    import math

    from posteriorforge.study.conditional_sampling import select_half_conditionals
    from posteriorforge.study.updates import _stagewise_update

    def evaluate(request_id=None):
        model, optimizer = model_and_optimizer()
        episode, = fixtures(Tracker(enforce=True), query_counts=(1,), dimensions=(6,))
        selections = None
        if request_id is not None:
            selection = select_half_conditionals(episode.roles, request_id)
            episode = replace(episode, stages=tuple(episode.stages[i] for i in selection.stage_indices))
            selections = (selection,)
        state = TrainingState()
        report = _stagewise_update(model, optimizer, (episode,), state, law=CANONICAL_FULL_SUPPORT,
            allow_unqualified_cpu_diagnostic=True, gradient_clip=100., conditional_selections=selections)
        assert state.update == state.data_position == report["optimizer_steps"] == 1
        assert report["gradient_norm_before_clip"] < 100.
        return report, torch.cat([p.grad.ravel().detach() for p in model.parameters()])

    full, expected = evaluate()
    test_episode, = fixtures(Tracker(), query_counts=(1,), dimensions=(6,))
    requests_by_selection = {}
    for i in range(100):
        request_id = f"{i:064x}"
        selection = select_half_conditionals(test_episode.roles, request_id)
        requests_by_selection.setdefault(selection.stage_indices, request_id)
    expected_subsets = {(p, *q, 5) for p, q in product((0, 1), combinations((2, 3, 4), 2))}
    assert set(requests_by_selection) == expected_subsets

    reports, gradients = zip(*(evaluate(request_id) for request_id in requests_by_selection.values()))
    assert all(r["conditional_count"] == 4 for r in reports)
    assert math.fsum(r["loss"] for r in reports) / len(reports) == pytest.approx(full["loss"], abs=3e-14)
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected, rtol=0, atol=3e-14)


def test_invalid_sampled_role_counts_fail_before_forward():
    from posteriorforge.study.conditional_sampling import select_half_conditionals
    from posteriorforge.study.updates import _stagewise_update

    tracker = Tracker(enforce=True)
    original, = fixtures(tracker, query_counts=(1,), dimensions=(6,))
    selection = select_half_conditionals(original.roles, "1" * 64)
    sampled = replace(original, stages=tuple(original.stages[i] for i in selection.stage_indices))
    forged = replace(selection, selected_role_counts=(2, 2, 1))
    model, optimizer = model_and_optimizer()
    state = TrainingState()
    with pytest.raises(ModelQualificationError, match="replay policy"):
        _stagewise_update(model, optimizer, (sampled,), state, law=CANONICAL_FULL_SUPPORT,
                          allow_unqualified_cpu_diagnostic=True, conditional_selections=(forged,))
    assert state == TrainingState() and not optimizer.state and not tracker.events
