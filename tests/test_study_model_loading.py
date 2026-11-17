import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from posteriorforge.study.model_loading import _deployment, load_inference_model, BASE_SHA256, LAW, SOURCE_POLICY


def deployment(method='lora_r4'):
    train, rank, fmt, path = {'lora_r4':('lora',4,'peft_adapter','adapter'),
      'lora_r16':('lora',16,'peft_adapter','adapter'),
      'full_finetune':('full_finetune',None,'full_tabpfn_safetensors','full_model.safetensors')}[method]
    return {'backend_kind':'stock_tabpfn_cpu','format':fmt,'path':path,'campaign_method':method,
      'training_method':train,'rank':rank,'base_checkpoint_sha256':BASE_SHA256,'weights_sha256':'a'*64,
      'sampling_law':LAW,'source_policy':SOURCE_POLICY,'scientifically_qualified':False}


def test_lazy_import_and_untrusted_metadata(tmp_path):
    script = '''import builtins\nr=builtins.__import__\ndef blocked(name,*a,**kw):\n if name == "torch" or name.startswith("torch."): raise ImportError("blocked")\n return r(name,*a,**kw)\nbuiltins.__import__=blocked\nimport posteriorforge.study.model_loading\n'''
    subprocess.run([sys.executable,'-c',script],check=True)
    deploy=tmp_path/'deployment'; deploy.mkdir(); (deploy/'adapter').mkdir()
    metadata={'deployment':deployment(),'weights_sha256':'a'*64,'identity':{'spec':{'method':'lora_r4'}}}
    _deployment(tmp_path,metadata,'a'*64)
    for key, val in [('path','../../outside'),('rank',16),('campaign_method','full_finetune'),('base_checkpoint_sha256','b'*64),('scientifically_qualified',True)]:
        bad=dict(metadata['deployment']);bad[key]=val
        with pytest.raises(ValueError): _deployment(tmp_path,dict(metadata,deployment=bad),'a'*64)
    with pytest.raises(ValueError): _deployment(tmp_path,metadata,'b'*64)
    (deploy/'adapter').rmdir(); (deploy/'adapter').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='unsafe'): _deployment(tmp_path,metadata,'a'*64)
    with pytest.raises(ValueError, match='pinned cached'):
        load_inference_model(tmp_path/'missing',expected_model_sha256='a'*64)


def _campaign(root, network, method):
    from posteriorforge.study.model import weights_fingerprint
    from posteriorforge.study.training import save_adapter, export_full_weights
    root.mkdir(); directory=root/'deployment';directory.mkdir()
    record=deployment(method); record['weights_sha256']=weights_fingerprint(network)
    if method.startswith('lora'):save_adapter(directory/'adapter', network.backbone, checkpoint_sha256=BASE_SHA256)
    else:export_full_weights(directory/'full_model.safetensors',network)
    metadata={'schema':'finetuneforge.campaign_checkpoint.v1','scientifically_qualified':False,
      'identity':{'spec':{'method':method}},'deployment':record,'weights_sha256':record['weights_sha256']}
    (root/'campaign.json').write_text(json.dumps(metadata))
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (root/'manifest.json').write_text(json.dumps({'schema':'finetuneforge.campaign_checkpoint.files.v1','files':files}))
    return record['weights_sha256'],hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()


def test_real_cached_frozen_and_campaign_roundtrip(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('peft')
    import torch
    from posteriorforge.study.model import TabPFNBackbone, attach_lora, weights_fingerprint
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real checkpoint is opt-in and never downloaded')
    torch.set_num_threads(2);torch.manual_seed(19)
    network=TabPFNBackbone.from_checkpoint(path)
    fingerprint=weights_fingerprint(network)
    frozen,meta=load_inference_model(path,expected_model_sha256=fingerprint)
    assert meta['kind']=='frozen' and meta['checkpoint_identity_format']=='base_checkpoint_file_sha256'
    assert weights_fingerprint(frozen)==fingerprint and not any(p.requires_grad for p in frozen.parameters())
    with pytest.raises(ValueError,match='fingerprint'):load_inference_model(path,expected_model_sha256='a'*64)
    network.backbone=attach_lora(network.backbone,rank=4)
    with torch.no_grad():
        for name,p in network.backbone.named_parameters():
            if 'lora_B' in name:p.normal_(0,.01)
    expected,receipt=_campaign(tmp_path/'adapter-checkpoint',network,'lora_r4')
    loaded,meta=load_inference_model(path,expected_model_sha256=expected,campaign_checkpoint=tmp_path/'adapter-checkpoint',expected_checkpoint_manifest_sha256=receipt)
    assert weights_fingerprint(loaded)==expected and meta['rank']==4 and meta['method']=='lora_r4'
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError,match='manifest'):load_inference_model(path,expected_model_sha256=expected,campaign_checkpoint=tmp_path/'adapter-checkpoint',expected_checkpoint_manifest_sha256='a'*64)
    adapter=tmp_path/'adapter-checkpoint'/'deployment'/'adapter'/'adapter_model.safetensors'
    with adapter.open('ab') as f:f.write(b'corrupt')
    with pytest.raises(Exception,match='checksum'):load_inference_model(path,expected_model_sha256=expected,campaign_checkpoint=tmp_path/'adapter-checkpoint',expected_checkpoint_manifest_sha256=receipt)
    fresh=TabPFNBackbone.from_checkpoint(path)
    with torch.no_grad():next(fresh.parameters()).add_(.001)
    expected,receipt=_campaign(tmp_path/'full-checkpoint',fresh,'full_finetune')
    loaded,meta=load_inference_model(path,expected_model_sha256=expected,campaign_checkpoint=tmp_path/'full-checkpoint',expected_checkpoint_manifest_sha256=receipt)
    assert weights_fingerprint(loaded)==expected and meta['rank'] is None and meta['method']=='full_finetune'

def test_campaign_manifest_required_and_authenticated_before_loading(tmp_path):
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real checkpoint is opt-in and never downloaded')
    with pytest.raises(ValueError,match='expected campaign manifest'):
        load_inference_model(path,expected_model_sha256='a'*64,campaign_checkpoint=tmp_path)
    with pytest.raises(ValueError,match='manifest checksum'):
        load_inference_model(path,expected_model_sha256='a'*64,campaign_checkpoint=tmp_path,expected_checkpoint_manifest_sha256='b'*64)
    (tmp_path/'manifest.json').write_text('corrupt')
    with pytest.raises(ValueError,match='manifest checksum'):
        load_inference_model(path,expected_model_sha256='a'*64,campaign_checkpoint=tmp_path,expected_checkpoint_manifest_sha256='b'*64)
