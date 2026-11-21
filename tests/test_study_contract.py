import json
from dataclasses import replace
import numpy as np
import pytest
from posteriorforge.study.spec import PriorSpec, ParameterSpec, FamilySpec, DatasetSpec, ProblemSpec
from posteriorforge.study.config import StudyConfig
from posteriorforge.study.seeds import derive_seed


def fixture_spec():
    return ProblemSpec('unit','diagnostic',(
        ParameterSpec('mu','population',PriorSpec('normal',0,1)),
        ParameterSpec('a','physical',PriorSpec('normal','mu',.2)),
        ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.02))),
        (FamilySpec('f','scalar','sigma'),),(DatasetSpec('d','f','s',{'a':'a'}),),{'labels':['known']})


def test_contract_roundtrip_is_immutable_and_support_is_complete():
    spec=fixture_spec()
    assert ProblemSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).fingerprint==spec.fingerprint
    with pytest.raises(TypeError): spec.datasets[0].parameters['a']='mu'
    with pytest.raises(TypeError): spec.metadata['labels']='changed'
    assert spec.support_mask(np.array([[0,1,.1],[0,1,-.1],[0,np.nan,.1]])).tolist()==[True,False,False]
    with pytest.raises(ValueError):spec.support_mask(np.ones((2,2)))


def test_invalid_parents_unused_targets_and_unknown_family_fail_before_simulation():
    spec=fixture_spec()
    with pytest.raises(ValueError,match='precede'):replace(spec,parameters=(spec.parameters[1],spec.parameters[0],spec.parameters[2]))
    with pytest.raises(ValueError,match='unused'):replace(spec,parameters=spec.parameters+(ParameterSpec('unused','population',PriorSpec('normal',0,1)),))
    with pytest.raises(ValueError):FamilySpec('f','unimplemented','sigma')
    with pytest.raises(ValueError):PriorSpec('normal',True,1)
    with pytest.raises(ValueError):PriorSpec('lognormal',0,1)


def test_full_study_plan_counts_and_scientific_invariants():
    cfg=StudyConfig();p=cfg.plan()
    assert p['maximum_npe_pfn_requests']==10800
    assert p['prior_only_requests']==120 and p['maximum_complete_vectors']==546000000
    assert p['gpu_jobs_submitted']==0 and p['estimated_gpu_hours'] is None
    assert StudyConfig.from_dict(cfg.to_dict()).fingerprint==cfg.fingerprint
    for kwargs in ({'posterior_draws':10000},{'cap_gpu_hours':200},{'bank_seeds':True},{'execution_mode':'gpu'},{'checkpoint_shortlist':3}):
        with pytest.raises(ValueError):StudyConfig(**kwargs)
    with pytest.raises(ValueError):StudyConfig.from_dict({'silent_typo':1})
    assert StudyConfig(checkpoint_shortlist=1).plan()['maximum_npe_pfn_requests']==8400


def test_named_seed_streams_replay_and_separate_identity_components():
    base=derive_seed(20261117,'bank',setup=4,bank=0)
    assert base==derive_seed(20261117,'bank',setup=4,bank=0)
    assert len({base,derive_seed(20261117,'query',setup=4),derive_seed(20261117,'bank',setup=5),derive_seed(20261117,'bank',setup=4,rank=1)})==4
    with pytest.raises(ValueError):derive_seed(True,'bank')


def test_distinct_families_cannot_silently_share_measurement_noise():
    spec=fixture_spec()
    with pytest.raises(ValueError,match='distinct'):
        replace(spec,families=spec.families+(FamilySpec('g','scalar','sigma'),),datasets=spec.datasets+(DatasetSpec('e','g','s',{'a':'a'}),))


def test_unknown_coefficients_and_signed_positive_coefficients_are_rejected():
    spec=fixture_spec()
    with pytest.raises(ValueError,match='coefficient'):
        replace(spec,datasets=(DatasetSpec('d','f','s',{'unused_coefficient':'a'}),))
    with pytest.raises(ValueError,match='positive'):
        replace(spec,families=(FamilySpec('f','spring','sigma'),),datasets=(DatasetSpec('d','f','s',{'k':'a'},tuple(range(10)),{'c':1}),))


def test_imported_specs_cannot_bypass_the_heldout_physics_boundary():
    p=ProblemSpec('spring-anchor','diagnostic',(
        ParameterSpec('k','physical',PriorSpec('lognormal',1,.2)),
        ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.02))),
        (FamilySpec('f','spring','sigma'),),(DatasetSpec('d','f','s',{'k':'k'},tuple(range(10)),{'c':1}),))
    for split in ('train','validation'):
        with pytest.raises(ValueError,match='held-out'):replace(p,split=split)


def test_incomplete_coefficients_and_inferred_saturation_exponents_fail_early():
    spec=fixture_spec()
    with pytest.raises(ValueError,match='explicitly'):
        replace(spec,families=(FamilySpec('f','quadratic','sigma'),),datasets=(DatasetSpec('d','f','s',{'a':'a'},tuple(range(10))),))
