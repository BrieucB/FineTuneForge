"""CPU integrity checks; no model loads, sampling, downloads or GPU calls."""
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/qualify-training-posteriors.py'
spec=importlib.util.spec_from_file_location('training_posterior_qualification_fixture',path)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@pytest.fixture
def posterior_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'source',lambda *a:{'source':'a'*64})
    for name in ('base','observation'):(tmp_path/name).write_bytes(name.encode())
    template={'posterior_draws':50000,'small_bank':{'worlds':256,'dimension':4},'protocol_sha256':'b'*64,
        'base_checkpoint':str(tmp_path/'base'),'base_checkpoint_sha256':p.digest(tmp_path/'base'),
        'frozen_model_sha256':'c'*64,'observation':str(tmp_path/'observation'),
        'observation_sha256':p.digest(tmp_path/'observation')}
    (tmp_path/'template.json').write_text(json.dumps(template))
    for variant in p.VARIANTS:
        directory=tmp_path/variant;directory.mkdir()
        (directory/'campaign.json').write_text(json.dumps({'schema':'finetuneforge.campaign_checkpoint.v1',
            'identity':{'spec':{'method':'lora_r4','effective_batch':16,'protocol_sha256':'b'*64}},
            'state':{'update':2,'data_position':32},'deployment':{'rank':4,'format':'peft_adapter'},'weights_sha256':variant[0]*64}))
        (directory/'manifest.json').write_text('{}')
    r=p.prepare(tmp_path/'template.json',tmp_path/'stock',tmp_path/'cached',tmp_path/'inputs',tmp_path,'d'*40)
    return tmp_path,r


def test_fixed_count_and_distinct_origins_are_authenticated(posterior_fixture):
    root,r=posterior_fixture;v=p.verify(r['path'],r['sha256'],root,'d'*40)
    assert v['posterior_draws']==50000 and v['timing_repetitions']==1
    assert v['checkpoints']['stock']['model_sha256']!=v['checkpoints']['cached']['model_sha256']
    assert v['new_forward_evaluations']==0 and not v['training_pilots_enabled']


@pytest.mark.parametrize('field,value',[('posterior_draws',10000),('posterior_draws',50000.),
    ('microbatch_size',10000),('seed',5),('training_pilots_enabled',True)])
def test_contract_tampering_is_rejected_before_gpu_import(posterior_fixture,field,value):
    root,r=posterior_fixture;f=Path(r['path']);v=json.loads(f.read_text());v[field]=value;f.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='contract'):p.verify(f,p.digest(f),root,'d'*40)


def test_checkpoint_mutation_invalidates_complete_request(posterior_fixture):
    root,r=posterior_fixture
    f=root/'cached/campaign.json';v=json.loads(f.read_text());v['state']['update']=3;f.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='two-update'):p.verify(r['path'],r['sha256'],root,'d'*40)


def test_unallocated_launch_stops_without_subprocess_or_gpu(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(p.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('no children')))
    with pytest.raises(RuntimeError,match='accounted'):p.run(inputs='missing',input_sha256='x',output=tmp_path/'unused',repo=tmp_path,source_commit='x')
    assert not (tmp_path/'unused').exists()


def test_rank16_has_distinct_method_and_supports_fixed_draws(posterior_fixture):
    root,_=posterior_fixture
    for name in p.VARIANTS:
        f=root/name/'campaign.json';v=json.loads(f.read_text())
        v['identity']['spec']['method']='lora_r16';v['deployment']['rank']=16;f.write_text(json.dumps(v))
    r=p.prepare(root/'template.json',root/'stock',root/'cached',root/'inputs-r16',root,'d'*40,rank=16)
    v=p.verify(r['path'],r['sha256'],root,'d'*40)
    assert v['rank']==16 and v['posterior_draws']==50000
