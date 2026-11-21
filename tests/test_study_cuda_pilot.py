"""CPU-only boundary tests for the CUDA qualification orchestrator."""
import pytest
from posteriorforge.study.sampling import sample_posterior


def test_modes_are_explicit_and_exclusive():
    for flags in ({}, {"diagnostic_cpu": True, "cuda_qualification": True}, {"cuda_qualification": 1}):
        with pytest.raises(ValueError, match="exactly one"):
            sample_posterior(None, None, object(), seed=0, microbatch_size=1024, identities={}, **flags)


def test_cuda_sampling_requires_slurm_and_authenticated_backend(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        sample_posterior(None, None, object(), seed=0, microbatch_size=1024, identities={}, cuda_qualification=True)
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-test-no-submission")
    with pytest.raises(ValueError, match="authenticated"):
        sample_posterior(None, None, object(), seed=0, microbatch_size=1024, identities={}, cuda_qualification=True)


def test_launch_requires_real_allocation_before_effects(monkeypatch, tmp_path):
    from posteriorforge.study.cuda_qualification import launch_workers
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    output=tmp_path/'must-not-exist'
    with pytest.raises(RuntimeError, match='Slurm'):
        launch_workers(inputs=tmp_path/'missing.json',input_sha256='a'*64,output=output,repo=tmp_path)
    assert not output.exists()


def test_manifest_binds_sources_query_and_exact_draw_count(monkeypatch,tmp_path):
    import json
    from posteriorforge.study import cuda_qualification as q
    base=tmp_path/'base';base.write_bytes(b'test-only-model-identity')
    obs=tmp_path/'observation.npy';obs.write_bytes(b'test-only-observation-identity')
    source={'unit.py':'a'*64};monkeypatch.setattr(q,'source_files',lambda repo:source)
    v={'schema':'finetuneforge.cuda_qualification_inputs.v1','posterior_draws':50000,'lanes':list(q.LANES),
       'source_files':dict(source),'base_checkpoint':str(base),'base_checkpoint_sha256':q.file_hash(base),
       'observation_sha256':q.file_hash(obs)}
    p=tmp_path/'manifest.json';q.write_json(p,v)
    assert q.verify_inputs(p,q.file_hash(p),tmp_path)['posterior_draws']==50000
    source['unit.py']='b'*64
    with pytest.raises(ValueError,match='source'):
        q.verify_inputs(p,q.file_hash(p),tmp_path)
    source['unit.py']='a'*64
    obs.write_bytes(b'changed-query')
    with pytest.raises(ValueError,match='query'):
        q.verify_inputs(p,q.file_hash(p),tmp_path)
    v['posterior_draws']=49999;q.write_json(p,v)
    with pytest.raises(ValueError,match='contract'):
        q.verify_inputs(p,q.file_hash(p),tmp_path)


def test_probe_requires_accounted_slurm_before_outputs(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    from posteriorforge.study.cuda_qualification import run_probes
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'):
        run_probes(inputs=tmp_path/'input',input_sha256='a'*64,output=tmp_path/'output',repo=tmp_path,
                   campaign_checkpoint=tmp_path/'checkpoint',checkpoint_manifest_sha256='b'*64)
    assert not (tmp_path/'output').exists()
