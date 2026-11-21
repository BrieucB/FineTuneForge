"""Frozen-deployment guards and opt-in real adapter composition on CPU only."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from posteriorforge.study.inference_deployment import load_frozen_deployment


def test_import_is_lazy_and_modes_reject_before_model_access():
    script = '''import builtins
original=builtins.__import__
def blocked(name,*args,**kwargs):
 if name == 'torch' or name.startswith('torch.') or name == 'peft': raise ImportError('blocked')
 return original(name,*args,**kwargs)
builtins.__import__=blocked
from posteriorforge.study.inference_deployment import load_frozen_deployment
try: load_frozen_deployment('missing',expected_model_sha256='a'*64,mode='invalid')
except ValueError: pass
else: raise AssertionError('mode accepted')
'''
    subprocess.run([sys.executable,'-c',script],check=True)
    with pytest.raises(ValueError,match='authenticated campaign'):
        load_frozen_deployment('missing',expected_model_sha256='a'*64,mode='merged_lora')


def _unit_network(rank=4):
    torch=pytest.importorskip('torch');pytest.importorskip('peft')
    from torch import nn
    from posteriorforge.study.model import TabPFNBackbone, attach_lora, BASE_CHECKPOINT_SHA256
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self._w_qkv=nn.Parameter(torch.randn(3,2,2,4))
            self._w_out=nn.Parameter(torch.randn(2,2,4))
        def compute_qkv(self,x):
            return torch.einsum('...i,shdi->...shd',x,self._w_qkv)
        def forward(self,x):
            q,k,v=self.compute_qkv(x).unbind(-3)
            return torch.einsum('...hd,hdo->...o',(q+k+v).tanh(),self._w_out)
    class Base(nn.Module):
        def __init__(self):
            super().__init__();self.attention=Attention();self.encoder=nn.Module()
            self.encoder.mlp=nn.Sequential(nn.Linear(4,4),nn.GELU())
            self.decoder_dict=nn.ModuleDict({'standard':nn.Linear(4,2)})
        def forward(self,x):return self.decoder_dict['standard'](self.encoder.mlp(self.attention(x)))
    network=TabPFNBackbone(Base().eval(),torch.tensor([-1.,0.,1.]),checkpoint_sha256=BASE_CHECKPOINT_SHA256)
    network.backbone=attach_lora(network.backbone,rank=rank)
    network.requires_grad_(False);network.eval()
    return network


def test_train_mode_unsupported_adapter_and_wrong_origin_are_rejected(monkeypatch):
    torch=pytest.importorskip('torch')
    import posteriorforge.study.inference_deployment as deployment
    from posteriorforge.study.model import weights_fingerprint, ModelQualificationError
    network=_unit_network();finger=weights_fingerprint(network)
    metadata={'model_sha256':finger,'rank':4,'method':'lora_r4'}
    monkeypatch.setattr(deployment,'load_inference_model',lambda *a,**kw:(network,metadata))
    network.train()
    with pytest.raises(ModelQualificationError,match='frozen eval'):
        load_frozen_deployment('unused',expected_model_sha256=finger,campaign_checkpoint='unused',mode='merged_lora')
    network.eval()
    metadata['rank']=16
    with pytest.raises(ModelQualificationError,match='rank and checkpoint method'):
        load_frozen_deployment('unused',expected_model_sha256=finger,campaign_checkpoint='unused',mode='merged_lora')
    metadata['rank']=4;metadata['model_sha256']='f'*64
    with pytest.raises(ModelQualificationError,match='origin fingerprint'):
        load_frozen_deployment('unused',expected_model_sha256=finger,campaign_checkpoint='unused',mode='merged_lora')


def test_multiple_adapters_are_not_silently_merged(monkeypatch):
    pytest.importorskip('torch')
    import posteriorforge.study.inference_deployment as deployment
    from posteriorforge.study.model import weights_fingerprint, ModelQualificationError
    network=_unit_network();finger=weights_fingerprint(network)
    metadata={'model_sha256':finger,'rank':4,'method':'lora_r4'}
    network.backbone.peft_config['other']=network.backbone.peft_config['default']
    monkeypatch.setattr(deployment,'load_inference_model',lambda *a,**kw:(network,metadata))
    with pytest.raises(ModelQualificationError,match='one default'):
        load_frozen_deployment('unused',expected_model_sha256=finger,campaign_checkpoint='unused',mode='merged_lora')


def test_disabled_adapter_is_not_silently_composed(monkeypatch):
    pytest.importorskip('torch');pytest.importorskip('peft')
    from peft.tuners.lora.layer import LoraLayer
    import posteriorforge.study.inference_deployment as deployment
    from posteriorforge.study.model import weights_fingerprint, ModelQualificationError
    network=_unit_network()
    next(m for m in network.backbone.modules() if isinstance(m,LoraLayer)).enable_adapters(False)
    finger=weights_fingerprint(network);metadata={'model_sha256':finger,'rank':4,'method':'lora_r4'}
    monkeypatch.setattr(deployment,'load_inference_model',lambda *a,**kw:(network,metadata))
    with pytest.raises(ModelQualificationError,match='disabled'):
        load_frozen_deployment('unused',expected_model_sha256=finger,campaign_checkpoint='unused',mode='merged_lora')


def _hash_tree(root):
    return {str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob('*') if p.is_file()}


@pytest.mark.parametrize('rank',[4,16])
def test_actual_cached_trained_adapter_composition_preserves_predictions_and_files(rank):
    torch=pytest.importorskip('torch');pytest.importorskip('peft');pytest.importorskip('tabpfn')
    base=os.environ.get('FTF_CPU_CHECKPOINT');campaign_root=os.environ.get('FTF_INFERENCE_CAMPAIGN_ROOT')
    if not base or not campaign_root:
        pytest.skip('actual cached trained adapter test is opt-in; no downloads or optimizer updates')
    from posteriorforge.study.model import PreparedMember, canonical_full_support_mixture, CANONICAL_FULL_SUPPORT, weights_fingerprint
    torch.set_num_threads(2)
    checkpoint=Path(campaign_root)/f'lora_r{rank}'/'after-two-updates'
    manifest=checkpoint/'manifest.json';manifest_sha=sha256(manifest.read_bytes()).hexdigest()
    expected=json.loads((checkpoint/'campaign.json').read_text())['weights_sha256']
    before_files=_hash_tree(checkpoint);before_base=sha256(Path(base).read_bytes()).hexdigest()
    original,original_meta=load_frozen_deployment(base,expected_model_sha256=expected,
        campaign_checkpoint=checkpoint,expected_checkpoint_manifest_sha256=manifest_sha,mode='unmerged')
    original_state=weights_fingerprint(original)
    merged,merged_meta=load_frozen_deployment(base,expected_model_sha256=expected,
        campaign_checkpoint=checkpoint,expected_checkpoint_manifest_sha256=manifest_sha,mode='merged_lora')
    generator=torch.Generator().manual_seed(20261117)
    x=torch.randn(16,3,generator=generator);y=torch.randn(16,generator=generator);q=torch.randn(2,3,generator=generator)
    members=[PreparedMember(x[:,torch.roll(torch.arange(3),i%3)],y,q[:,torch.roll(torch.arange(3),i%3)]) for i in range(8)]
    with torch.no_grad():
        reference=original(members);candidate=merged(members)
        assert len(reference)==len(candidate)==8
        assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(reference,candidate))
        a=canonical_full_support_mixture(reference,[original.borders]*8,original.borders)
        b=canonical_full_support_mixture(candidate,[merged.borders]*8,merged.borders)
        points=torch.tensor([-.5,.5],dtype=torch.float64)
        assert torch.allclose(a.cdf(points),b.cdf(points),atol=1e-6,rtol=1e-6)
    assert original_meta['model_sha256']==expected==weights_fingerprint(original)==original_state
    assert merged_meta['model_sha256']==weights_fingerprint(merged)!=expected
    assert merged_meta['origin_model_sha256']==expected
    assert merged_meta['checkpoint_manifest_sha256']==manifest_sha
    assert merged_meta['inventory_before']['lora_linear_modules']>0
    assert merged_meta['inventory_before']['packed_projection_modules']>0
    assert merged_meta['inventory_after']['lora_linear_modules']==0
    assert merged_meta['inventory_after']['packed_projection_modules']==0
    assert merged_meta['inventory_after']['parametrized_modules']==0
    assert original_meta['inventory_after']['feature_basis_hook_seeds']==[0]
    assert merged_meta['inventory_after']['feature_basis_hook_seeds']==[0]
    assert merged_meta['trainable_parameter_count']==merged_meta['trainable_tensor_count']==0
    assert not any(p.requires_grad or p.grad is not None for p in merged.parameters())
    assert all(p.dtype==torch.float32 for p in merged.parameters()) and merged.borders.dtype==torch.float64
    assert merged_meta['equivalence_status']=='requires_execution_qualification'
    assert not merged_meta['scientifically_qualified']
    assert _hash_tree(checkpoint)==before_files
    assert sha256(Path(base).read_bytes()).hexdigest()==before_base
