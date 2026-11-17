"""CPU control-flow checks for opt-in checkpoint-safe training scopes."""
from contextlib import contextmanager
from copy import deepcopy

import pytest

torch=pytest.importorskip('torch')
from posteriorforge.study import updates,training_execution
from posteriorforge.study.model import CANONICAL_FULL_SUPPORT,ModelQualificationError
from posteriorforge.study.training import TrainingState
from test_study_updates import Tracker,FixtureEnsemble,BackwardWitness,fixtures,model_and_optimizer


@pytest.mark.parametrize('failure',[None,'forward','gradient'])
def test_cache_wrappers_span_backward_and_end_before_guards_step_or_failure(failure,monkeypatch):
    model,optimizer=model_and_optimizer();tracker=Tracker(enforce=True)
    episodes=fixtures(tracker,failure=None if failure is None else ('1/1',failure))
    state=TrainingState();before=deepcopy(model.state_dict());scope={'active':False,'entered':0,'exited':0}
    @contextmanager
    def context(actual,*,enabled):
        assert actual is model and enabled is True and not scope['active']
        scope['active']=True;scope['entered']+=1
        try:yield
        finally:scope['active']=False;scope['exited']+=1
    original_density=FixtureEnsemble.density
    def density(self,*a,**kw):
        assert scope['active'];return original_density(self,*a,**kw)
    original_backward=BackwardWitness.backward
    def backward(ctx,gradient):
        assert scope['active'];return original_backward(ctx,gradient)
    original_check=updates._require_finite_gradients
    def check(parameters):
        assert not scope['active'];return original_check(parameters)
    original_step=optimizer.step
    def step(*a,**kw):
        assert not scope['active'];tracker.steps+=1;return original_step(*a,**kw)
    monkeypatch.setattr(training_execution,'conditional_forward_cache',context)
    monkeypatch.setattr(FixtureEnsemble,'density',density)
    monkeypatch.setattr(BackwardWitness,'backward',staticmethod(backward))
    monkeypatch.setattr(updates,'_require_finite_gradients',check)
    monkeypatch.setattr(optimizer,'step',step)
    def run():
        return updates._stagewise_update(model,optimizer,episodes,state,law=CANONICAL_FULL_SUPPORT,
            allow_unqualified_cpu_diagnostic=True,cache_parametrizations=True)
    if failure is None:
        report=run();assert report['conditional_count']==scope['entered']==10
        assert state.update==tracker.steps==1 and state.data_position==4
    else:
        with pytest.raises(ModelQualificationError):run()
        assert state.update==tracker.steps==0 and not optimizer.state
        assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
        assert all(p.grad is None for p in model.parameters())
    assert scope['entered']==scope['exited'] and not scope['active']


def test_disabled_route_never_enters_optimization_context(monkeypatch):
    model,optimizer=model_and_optimizer()
    monkeypatch.setattr(training_execution,'conditional_forward_cache',
        lambda *a,**kw:(_ for _ in ()).throw(AssertionError('stock route must remain untouched')))
    report=updates._stagewise_update(model,optimizer,fixtures(Tracker()),TrainingState(),
        law=CANONICAL_FULL_SUPPORT,allow_unqualified_cpu_diagnostic=True)
    assert report['optimizer_steps']==1 and report['conditional_count']==10
