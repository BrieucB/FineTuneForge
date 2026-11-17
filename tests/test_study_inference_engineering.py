import pytest
from posteriorforge.study import inference_engineering as e


def test_variant_control_does_not_match_unmerged_by_substring():
    assert e.context_variant('c4k_r4_unmerged')==('lora_r4','unmerged',False)
    assert e.context_variant('c4k_r4_merged_tiled')==('lora_r4','merged_lora',True)
    assert e.context_variant('c4k_frozen_stock')==('frozen','unmerged',False)
    with pytest.raises(ValueError):e.context_variant('invalid')


def test_launch_stops_outside_allocation_before_effects(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'):
        e.run('missing','a'*64,tmp_path/'output',tmp_path,'profile')
    assert not (tmp_path/'output').exists()


def test_full_sample_distribution_flags_detect_degradation():
    import numpy as np
    a=np.random.default_rng(32).normal(size=(50000,4))
    b=a.copy();d=e.compare_samples(a,b)
    assert d['max_pairwise_cdf']==0 and not d['investigation_flag']
    b[:,0]+=1
    assert e.compare_samples(a,b)['investigation_flag']
    with pytest.raises(ValueError,match='50000'):
        e.compare_samples(a[:10000],b[:10000])


def test_manifest_scope_rejects_reduced_draws_without_torch(tmp_path,monkeypatch):
    from posteriorforge.study.cuda_qualification import write_json,file_hash
    p=tmp_path/'manifest.json';write_json(p,{'schema':'finetuneforge.inference_engineering.v1','posterior_draws':10000,'timing_repetitions':3,'training_pilots_enabled':False})
    with pytest.raises(ValueError,match='contract'):e.verify(p,file_hash(p),tmp_path)
