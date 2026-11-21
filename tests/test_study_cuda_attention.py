"""Bounded, CPU-only parity for the opt-in inference attention adapter."""
import os
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from posteriorforge.study.cuda_attention import tiled_attention, install_tiled_attention

@pytest.mark.parametrize('h,kh,nq,nk', [(4,4,7,13),(4,1,7,13),(6,2,9,5),(3,3,1,31)])
def test_exact_chunked_sdpa_all_keys(h,kh,nq,nk):
    torch.manual_seed(31)
    q=torch.randn(3,nq,h,8);k=torch.randn(3,nk,kh,8);v=torch.randn(3,nk,kh,6)
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.repeat_interleave(h//kh,2).transpose(1,2),v.repeat_interleave(h//kh,2).transpose(1,2),dropout_p=0).transpose(1,2)
    for budget in (4*h*nk, 4*h*nk*3, 4*h*nk*nq*2):
        actual=tiled_attention(q,k,v,max_score_bytes=budget)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


def test_unsupported_semantics_fail_closed():
    q=torch.randn(1,2,4,3);k=torch.randn(1,3,2,3);v=k.clone()
    for kw in ({'is_causal':True},{'attn_mask':torch.ones(2,3,dtype=torch.bool)},{'dropout_p':.1},{'max_score_bytes':1}):
        with pytest.raises(ValueError):tiled_attention(q,k,v,**kw)
    with pytest.raises(ValueError):tiled_attention(q,k[:,:,:1],v[:,:,:2].double())
    with pytest.raises(ValueError):tiled_attention(q,k,v,max_score_bytes=True)


def test_instance_patch_fast_path_tiling_gradients_and_restoration():
    from tabpfn.model.multi_head_attention import MultiHeadAttention
    torch.manual_seed(7)
    layer=MultiHeadAttention(input_size=16,output_size=16,d_k=4,d_v=4,nhead=4,device=torch.device('cpu'),dtype=torch.float32,share_kv_across_n_heads=2)
    x=torch.randn(3,7,16);kv=torch.randn(3,11,16)
    with torch.no_grad(): reference=layer(x,kv)
    grad_input=x.clone().requires_grad_();layer(grad_input,kv).sum().backward();gradient=grad_input.grad.clone();layer.zero_grad(set_to_none=True)
    with install_tiled_attention(layer,max_score_bytes=4*4*11*2) as handle:
        identity=handle.metadata.copy()
        with torch.no_grad(): actual=layer(x,kv)
        torch.testing.assert_close(actual,reference,rtol=2e-5,atol=2e-6)
        assert handle.statistics['tiled_inference'] > 0
        assert handle.statistics['max_tile_score_bytes'] <= handle.metadata['max_score_bytes']
        assert handle.metadata == identity
        incoming=x.clone().requires_grad_();layer(incoming,kv).sum().backward()
        torch.testing.assert_close(incoming.grad,gradient,rtol=0,atol=0)
        assert handle.statistics['stock_gradient_path'] > 0
        with pytest.raises(ValueError,match='custom'):install_tiled_attention(layer)
    with torch.no_grad():torch.testing.assert_close(layer(x,kv),reference,rtol=0,atol=0)
    with install_tiled_attention(layer,max_score_bytes=1<<28) as fast:
        with torch.no_grad():torch.testing.assert_close(layer(x,kv),reference,rtol=0,atol=0)
        assert fast.statistics['stock_fast_path'] > 0 and fast.statistics['tiled_inference']==0


def test_real_eight_member_stock_outputs_under_forced_tiling():
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('cached real checkpoint optional; no downloads')
    from posteriorforge.study.model import TabPFNBackbone, prepare_fitted_regressor_members, weights_fingerprint
    from posteriorforge.study.banks import prepare_bank
    from posteriorforge.study.science import encode_prior
    from tabpfn import TabPFNRegressor
    from test_study_banks import problem
    torch.set_num_threads(2)
    bank=prepare_bank(problem(),16,3); z=encode_prior(bank.spec,bank.theta)
    reg=TabPFNRegressor(model_path=path,device='cpu',n_estimators=8,random_state=0,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    reg.fit(bank.observations.copy(),z[:,0].copy())
    network=TabPFNBackbone.from_checkpoint(path).eval();before=weights_fingerprint(network)
    ensemble=prepare_fitted_regressor_members(reg,np.array([[.1,.2],[-.2,.3]]),device='cpu',source_policy='cancelled_zero_width')
    with torch.no_grad():expected=network(ensemble.members)
    with install_tiled_attention(network,max_score_bytes=2048) as handle:
        with torch.no_grad():actual=network(ensemble.members)
        assert handle.statistics['tiled_inference'] > 0
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-4)
        assert weights_fingerprint(network)==before
    assert len(actual)==8
