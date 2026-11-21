from dataclasses import replace
import json
import numpy as np
import pytest
from posteriorforge.study.spec import ProblemSpec,PriorSpec,ParameterSpec,FamilySpec,DatasetSpec
from posteriorforge.study.episodes import EpisodeStream,QuickValidationStream,BankStore,materialize_episode

P='a'*64

def problem(index=0,split='diagnostic'):
    return ProblemSpec(f'episode-{index}',split,(ParameterSpec('mu','population',PriorSpec('normal',.1*index,1.)),ParameterSpec('tau','population',PriorSpec('lognormal',.4,.1)),ParameterSpec('a','physical',PriorSpec('normal','mu','tau')),ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.03))),(FamilySpec('scalar','scalar','sigma'),),(DatasetSpec('data','scalar','specimen',{'a':'a'}),))

def test_matched_stream_replays_cursor_and_separates_replicas_and_phases():
    setups=(problem(),problem(1))
    streams=[EpisodeStream(setups,protocol_sha256=P,diagnostic=True) for _ in range(3)]
    expected=[streams[0].request(i) for i in range(100)]
    assert all([stream.request(i) for i in range(100)]==expected for stream in streams)
    assert {r.context_size for r in expected}=={16,64,256,1024}
    assert len({r.context_seed for r in expected})==100
    assert all(r.context_seed!=r.query_seed and r.query_bank_count==1 for r in expected)
    assert streams[0].batch(17,3)==tuple(expected[17:20])
    for kwargs in ({'replica':1},{'phase':'pilot'}):
        other=EpisodeStream(setups,protocol_sha256=P,diagnostic=True,**kwargs)
        assert other.identity!=streams[0].identity and other.request(0).context_seed!=expected[0].context_seed
    with pytest.raises(ValueError):EpisodeStream((problem(split='test'),),protocol_sha256=P,diagnostic=True)
    with pytest.raises(ValueError):EpisodeStream(setups,protocol_sha256=P)


def test_fresh_visits_and_exact_shared_replay_account_for_all_data(tmp_path):
    stream=EpisodeStream((problem(),),protocol_sha256=P,diagnostic=True);store=BankStore(tmp_path/'banks')
    a=materialize_episode(stream,stream.request(0),store);replay=materialize_episode(stream,stream.request(0),store)
    b=materialize_episode(stream,stream.request(1),store)
    assert np.array_equal(a.context.theta,replay.context.theta) and np.array_equal(a.query.observations,replay.query.observations)
    assert replay.receipt['incurred_forward_evaluations']==0
    assert all(d['from_scratch_forward_evaluations']>0 for d in replay.receipt['datasets'])
    assert a.receipt['incurred_forward_evaluations']==len(a.context.theta)+1
    assert a.receipt['ordering'] and a.query.theta.shape==(1,4)
    assert a.receipt['datasets'][0]['dataset_id']!=b.receipt['datasets'][0]['dataset_id']
    assert a.receipt['datasets'][1]['dataset_id']!=b.receipt['datasets'][1]['dataset_id']
    assert not any(np.array_equal(row,a.query.theta[0]) for row in a.context.theta)
    with pytest.raises(ValueError):materialize_episode(stream,replace(stream.request(0),context_seed=42),store)


def test_quick_validation_is_exact320_fixed_scores_with_nested_shared_data(tmp_path):
    setups=tuple(problem(i,split='validation') for i in range(20))
    stream=QuickValidationStream(setups,protocol_sha256=P);requests=list(stream)
    assert len(stream)==len(requests)==320 and requests==list(stream)
    assert {r.context_size for r in requests}=={16,64,256,1024}
    assert len({r.query_seed for r in requests})==20
    assert len({r.context_seed for r in requests})==40
    store=BankStore(tmp_path/'banks')
    a=materialize_episode(stream,requests[0],store);other_case=materialize_episode(stream,requests[1],store);larger=materialize_episode(stream,requests[2],store)
    assert a.receipt['ordering'] is None
    assert np.array_equal(a.query.theta,larger.query.theta)
    assert not np.array_equal(a.query.theta,other_case.query.theta)
    assert np.array_equal(a.context.theta,larger.context.theta[:16])
    assert a.receipt['incurred_forward_evaluations']==1026 and larger.receipt['incurred_forward_evaluations']==0
    assert len({d['dataset_id'] for item in (a,other_case,larger) for d in item.receipt['datasets']})==2


def test_cache_corruption_and_failed_provider_are_retained_without_zero_cost(tmp_path):
    store=BankStore(tmp_path/'banks');bank,event=store.obtain(problem(),16,31)
    path=tmp_path/'banks'/event['dataset_id']/'bank.npz'
    with path.open('ab') as out:out.write(b'corrupt')
    with pytest.raises(ValueError):store.obtain(problem(),16,31)
    assert store.events[-1]['incurred_forward_evaluations'] is None
    def broken(*args):raise RuntimeError('provider failed after unknown partial work')
    bad=BankStore(tmp_path/'broken',provider=broken,provider_identity='b'*64)
    with pytest.raises(RuntimeError):bad.obtain(problem(),16,1)
    record=json.loads(next((tmp_path/'broken'/'failures').glob('*.json')).read_text())
    assert record['incurred_forward_evaluations'] is None and record['forward_evaluation_upper_bound']==16


@pytest.mark.parametrize('field,value',[('replica',True),('replica',False),('replica',0.),('master_seed',True),('master_seed',1.),('master_seed',-1),('diagnostic',1),('diagnostic','false'),('diagnostic',None)])
def test_episode_stream_identity_flags_require_exact_integer_boolean_types(field,value):
    kwargs={'protocol_sha256':P,'diagnostic':True,field:value}
    with pytest.raises(ValueError):EpisodeStream((problem(),),**kwargs)
    if field!='replica':
        with pytest.raises(ValueError):QuickValidationStream((problem(),),**kwargs)
