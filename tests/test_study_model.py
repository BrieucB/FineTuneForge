"""Density-law and real packed-projection CPU checks; no production inference."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from posteriorforge.study.model import (
    BarDensity, CacheIdentity, LEGACY_FINITE, NATIVE_FULL_SUPPORT, CANONICAL_FULL_SUPPORT,
    ModelQualificationError, PreparedMember, TabPFNBackbone, attach_lora,
    canonical_finite_mixture, merge_lora, weights_fingerprint,
)


def test_finite_density_is_normalized_with_bin_widths_and_jacobian():
    logits = torch.tensor([0., 0.], dtype=torch.float64, requires_grad=True)
    d = BarDensity(logits, torch.tensor([0., 1., 3.]), law=LEGACY_FINITE)
    values = torch.tensor([.5, 2.], dtype=torch.float64)
    assert torch.allclose(d.log_prob(values).exp(), torch.tensor([.5, .25], dtype=torch.float64))
    assert torch.allclose(d.log_prob(values, log_abs_det_decode=np.log(2)), d.log_prob(values)-np.log(2))
    assert torch.isneginf(d.log_prob(torch.tensor([-1., 4.]))).all()
    (-d.log_prob(values).mean()).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose((d.log_prob(values).exp()*torch.tensor([1.,2.])).sum(), torch.tensor(1.,dtype=torch.float64))


@pytest.mark.parametrize("law", [LEGACY_FINITE, NATIVE_FULL_SUPPORT])
def test_inverse_cdf_and_density_use_the_same_law(law):
    d = BarDensity(torch.tensor([-.4, .7, .2], dtype=torch.float64), torch.tensor([-2., -1., 1., 3.]), law=law)
    u = torch.linspace(.0001, .9999, 1001, dtype=torch.float64)
    x = d.icdf(u)
    assert torch.allclose(d.cdf(x), u, atol=2e-13, rtol=2e-13)
    # Numerical differentiation of the CDF verifies the separately coded PDF.
    points = torch.tensor([-4., -1.5, 0., 2., 5.],dtype=torch.float64)
    points = points if law == NATIVE_FULL_SUPPORT else points[1:4]
    delta=1e-5
    numerical=(d.cdf(points+delta)-d.cdf(points-delta))/(2*delta)
    assert torch.allclose(numerical, d.log_prob(points).exp(),atol=2e-10,rtol=1e-7)


def test_finite_zero_uniform_policy_and_invalid_full_support_uniform():
    d=BarDensity(torch.tensor([-torch.inf,0.,0.]),torch.tensor([0.,1.,2.,3.]),law=LEGACY_FINITE)
    assert d.icdf(torch.tensor(0.)).item()==1.
    with pytest.raises(ValueError):d.icdf(torch.tensor(1.))
    full=BarDensity(torch.zeros(2),torch.tensor([0.,1.,2.]),law=NATIVE_FULL_SUPPORT)
    with pytest.raises(ValueError,match="open uniforms"):full.icdf(torch.tensor(0.))


def test_native_full_support_math_agrees_with_checkpoint_criterion_away_from_edges():
    pytest.importorskip("tabpfn")
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    b=torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64)
    y=torch.tensor([-5.,-.4,0.,2.,6.],dtype=torch.float64)
    logits=torch.tensor([.2,-.5,.8],dtype=torch.float64).expand(len(y),-1)
    criterion=FullSupportBarDistribution(b)
    d=BarDensity(logits,b,law=NATIVE_FULL_SUPPORT)
    assert torch.allclose(-d.log_prob(y),criterion(logits,y),atol=3e-6,rtol=3e-6)
    finite=BarDensity(logits,b,law=LEGACY_FINITE)
    assert torch.isneginf(finite.log_prob(y)[[0,-1]]).all()
    assert torch.isfinite(d.log_prob(y)).all()


def test_corrected_canonical_mixture_preserves_all_members_and_gradients():
    logits=[torch.tensor([[.2,-.4]],dtype=torch.float64,requires_grad=True),torch.tensor([[-.8,.5]],dtype=torch.float64,requires_grad=True)]
    borders=[torch.tensor([-2.,0.,2.]),torch.tensor([-1.,1.,3.])]
    canonical=torch.tensor([-2.,0.,2.])
    mixture=canonical_finite_mixture(logits,borders,canonical,expected_members=2)
    p0=torch.softmax(logits[0],-1)[0,0]
    p1=torch.softmax(logits[1],-1)[0,0]
    assert torch.allclose(mixture.weights[0],torch.stack(((p0+.5*p1)/2,1-(p0+.5*p1)/2)))
    (-mixture.log_prob(torch.tensor([.5]))).sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in logits)
    with pytest.raises(ValueError,match="all 8"):canonical_finite_mixture(logits,borders,canonical)


class ProjectionFixture(nn.Module):
    """Small actual TabPFN attention module, solely for adapter-layout fixtures."""
    def __init__(self):
        super().__init__()
        from tabpfn.model.multi_head_attention import MultiHeadAttention
        self.attention=MultiHeadAttention(input_size=4,output_size=4,d_k=2,d_v=2,nhead=2,device=torch.device('cpu'),dtype=torch.float32)
        self.encoder=nn.Module()
        self.encoder.mlp=nn.Sequential(nn.Linear(4,8,bias=False),nn.GELU(),nn.Linear(8,4,bias=False))
        self.decoder_dict=nn.ModuleDict({'standard':nn.Linear(4,3,bias=False)})
    def forward(self,x):
        return self.decoder_dict['standard'](self.encoder.mlp(self.attention(x)))


@pytest.mark.parametrize("rank", [4, 16])
def test_peft_packed_projection_zero_adapter_gradients_save_load_and_merge(tmp_path,rank):
    pytest.importorskip('peft');pytest.importorskip('tabpfn')
    from posteriorforge.study.training import save_adapter,load_adapter
    torch.manual_seed(19)
    base=ProjectionFixture().eval().requires_grad_(False);base_state=deepcopy(base.state_dict())
    x=torch.randn(2,3,4);before=base(x).detach()
    model=attach_lora(base,rank=rank)
    assert torch.equal(before,model(x).detach())
    a_parameters=[p for n,p in model.named_parameters() if 'lora_A' in n and '.parametrizations.' in n]
    assert a_parameters and all(p.shape[0]==rank for p in a_parameters)
    loss=model(x).square().mean();loss.backward()
    changed=[p for n,p in model.named_parameters() if 'lora_B' in n and p.grad is not None and bool((p.grad!=0).any())]
    assert changed and all(torch.isfinite(p.grad).all() for p in changed)
    with torch.no_grad():
        for p in changed:p.add_(-.01*p.grad)
    after=model(x).detach()
    save_adapter(tmp_path/'adapter',model,checkpoint_sha256='fixture-base')
    fresh=ProjectionFixture().eval();fresh.load_state_dict(base_state)
    restored=load_adapter(fresh,tmp_path/'adapter',expected_checkpoint_sha256='fixture-base')
    assert torch.allclose(after,restored(x).detach(),atol=1e-6,rtol=1e-6)
    merged=merge_lora(restored)
    assert torch.allclose(after,merged(x).detach(),atol=1e-6,rtol=1e-6)
    assert not any('lora_' in n or '.parametrizations.' in n for n,_ in merged.named_parameters())


def test_cache_identity_refuses_changed_weights_or_preprocessing():
    original=CacheIdentity('weights','preprocessing','context','protocol')
    original.require_same(original)
    with pytest.raises(ModelQualificationError):original.require_same(CacheIdentity('adapter-changed','preprocessing','context','protocol'))
    with pytest.raises(ModelQualificationError):original.require_same(CacheIdentity('weights','other','context','protocol'))


def test_prepared_member_contains_no_query_target_and_checks_width():
    member=PreparedMember(torch.ones(2,3),torch.zeros(2),torch.ones(1,3))
    member.validate()
    assert not hasattr(member,'query_y')
    with pytest.raises(ValueError):PreparedMember(torch.ones(2,3),torch.zeros(2),torch.ones(1,4)).validate()


@pytest.mark.parametrize("rank", [4, 16])
def test_real_checkpoint_cpu_zero_adapter_all_member_gradient_and_merge(tmp_path,rank):
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real cached checkpoint qualification is explicitly opt-in; no CI download')
    pytest.importorskip('peft')
    torch.set_num_threads(2);torch.manual_seed(23)
    network=TabPFNBackbone.from_checkpoint(Path(path))
    assert sum(p.numel() for p in network.backbone.parameters())==11081864
    x=torch.randn(16,3);y=torch.randn(16);q=torch.randn(2,3)
    members=[PreparedMember(x[:,torch.roll(torch.arange(3),i%3)],y,q[:,torch.roll(torch.arange(3),i%3)]) for i in range(8)]
    # Full-fine-tuning control: actual low-level attention weights get gradients.
    from posteriorforge.study.training import configure_full_finetune,save_adapter,load_adapter
    configure_full_finetune(network.backbone)
    full_logits=network.forward_member(members[0])
    full_loss=-BarDensity(full_logits,network.borders,law=LEGACY_FINITE).log_prob(torch.tensor([0.,.25])).mean()
    full_loss.backward()
    assert any(p.grad is not None and bool((p.grad!=0).any()) for name,p in network.backbone.named_parameters() if '_w_q' in name)
    network.backbone.zero_grad(set_to_none=True)
    network.backbone.requires_grad_(False)
    with torch.no_grad():before=network(members)
    network.backbone=attach_lora(network.backbone,rank=rank)
    with torch.no_grad():zero_adapter=network(members)
    assert len(zero_adapter)==8 and all(torch.equal(a,b) for a,b in zip(before,zero_adapter))
    actual=network(members)
    density=canonical_finite_mixture(actual,[network.borders]*8,network.borders)
    loss=-density.log_prob(torch.tensor([0.,.25])).mean();loss.backward()
    gradients={name:p.grad for name,p in network.backbone.named_parameters() if p.requires_grad}
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert any(bool((g!=0).any()) for name,g in gradients.items() if 'lora_B' in name and '.parametrizations.' in name)
    assert any(bool((g!=0).any()) for name,g in gradients.items() if 'lora_B' in name and 'decoder_dict' in name)
    with torch.no_grad():
        for name,p in network.backbone.named_parameters():
            if p.requires_grad and p.grad is not None:p.add_(-1e-4*p.grad)
        adapted=network(members)
    save_adapter(tmp_path/'real-adapter',network.backbone,checkpoint_sha256=network.checkpoint_sha256)
    fresh=TabPFNBackbone.from_checkpoint(Path(path))
    fresh.backbone=load_adapter(fresh.backbone,tmp_path/'real-adapter',expected_checkpoint_sha256=network.checkpoint_sha256)
    with torch.no_grad():
        restored=fresh(members)
        assert all(torch.equal(a,b) for a,b in zip(adapted,restored))
        fresh.backbone=merge_lora(fresh.backbone)
        merged=fresh(members)
    assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(adapted,merged))
    assert not network.metadata()['scientifically_qualified']


def test_differentiable_finite_head_matches_actual_corrected_v3_probability_policy():
    from posteriorforge import execution_probability_v3 as policy
    from posteriorforge.study.model import translate_finite_logits
    canonical=torch.tensor([-3.,-.2,.7,3.],dtype=torch.float32)
    api=torch.tensor([-6.,-.4,1.4,6.],dtype=torch.float32)
    logits=[];borders=[];reference=[]
    for i in range(8):
        x=torch.tensor([[.2+i*.07,-.4,.8],[-.2,.3-i*.02,.1]],dtype=torch.float32,requires_grad=True)
        b=torch.tensor([-2.-i*.03,-.5,.9,2.7+i*.04],dtype=torch.float32)
        logits.append(x);borders.append(b)
        old=policy.from_adjusted_logits(x.detach(),b,canonical,api)
        assert torch.allclose(translate_finite_logits(x,b,canonical),old['raw_weights64'],atol=2e-15,rtol=2e-15)
        reference.append(old)
    expected=policy.assemble_shared_raw64(reference,stage0=True)
    actual=canonical_finite_mixture(logits,borders,canonical,sampling_borders=api)
    assert torch.allclose(actual.weights,expected['sampling_interval_weights64'],atol=2e-15,rtol=2e-15)
    assert torch.allclose(actual.weights.cumsum(-1),expected['sampling_cdf64'],atol=2e-15,rtol=2e-15)
    targets=torch.tensor([-.9,2.],dtype=torch.float64)
    indices=torch.searchsorted(api.double(),targets,right=True)-1
    expected_log=expected['sampling_interval_weights64'].gather(-1,indices[:,None]).squeeze(-1).log()-torch.diff(api.double())[indices].log()
    assert torch.allclose(actual.log_prob(targets),expected_log,atol=2e-15,rtol=2e-15)
    (-actual.log_prob(targets).mean()).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in logits)


def test_real_stock_preprocessing_bridge_preserves_eight_member_probability_path():
    path=os.environ.get('FTF_CPU_CHECKPOINT')
    if not path:pytest.skip('real checkpoint bridge qualification is opt-in and never downloads')
    from tabpfn import TabPFNRegressor
    from posteriorforge import execution_probability_v3 as policy
    from posteriorforge.study.model import prepare_fitted_regressor_members,translate_finite_logits
    rng=np.random.default_rng(20261117)
    reg=TabPFNRegressor(model_path=path,device='cpu',n_estimators=8,random_state=0,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    reg.fit(rng.normal(size=(16,3)),rng.normal(size=16))
    prepared=prepare_fitted_regressor_members(reg,rng.normal(size=(2,3)))
    assert len(prepared.members)==8 and all(len(m.context_y)==16 for m in prepared.members)
    network=TabPFNBackbone(reg.executor_.model,reg.bardist_.borders,checkpoint_sha256='fixture-already-pinned')
    outputs=prepared.adjusted_logits(network)
    reference=[policy.from_adjusted_logits(x.detach(),m.source_borders,prepared.canonical_borders,prepared.sampling_borders) for x,m in zip(outputs,prepared.members)]
    expected=policy.assemble_shared_raw64(reference,stage0=True)
    actual=canonical_finite_mixture(outputs,[m.source_borders for m in prepared.members],prepared.canonical_borders,sampling_borders=prepared.sampling_borders)
    assert torch.allclose(actual.weights,expected['sampling_interval_weights64'],atol=2e-14,rtol=2e-13)
    target=torch.tensor([0.,.25],dtype=torch.float64)
    loss=-actual.log_prob(target).mean();loss.backward()
    assert any(p.grad is not None and bool((p.grad!=0).any()) for n,p in network.backbone.named_parameters() if '_w_q' in n)


def test_native_full_tail_pdf_normalization_survival_and_unclipped_sampling():
    from scipy.integrate import quad
    logits=torch.tensor([.2,-.5,.8],dtype=torch.float64)
    d=BarDensity(logits,torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64),law=NATIVE_FULL_SUPPORT)
    integral=sum(quad(lambda x:float(d.pdf(torch.tensor(x,dtype=torch.float64))),left,right,epsabs=1e-10)[0] for left,right in [(-np.inf,-1.),(-1.,1.),(1.,np.inf)])
    assert integral==pytest.approx(1.,abs=2e-10)
    values=torch.tensor([-30.,-2.,0.,3.,35.],dtype=torch.float64)
    assert torch.allclose(d.cdf(values)+d.sf(values),torch.ones_like(values),atol=2e-15,rtol=2e-15)
    assert d.sf(values[-1])>0 and 1-d.cdf(values[-1])==0
    a=d.sample((128,),generator=torch.Generator().manual_seed(34))
    b=d.sample((128,),generator=torch.Generator().manual_seed(34))
    assert torch.equal(a,b) and torch.isfinite(a).all()
    assert a.min() < d.borders[0] and a.max() > d.borders[-1]


def test_native_tail_density_retains_mass_below_cdf_rounding_precision():
    logits=torch.tensor([0.,0.,-50.],dtype=torch.float64,requires_grad=True)
    borders=torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64)
    full=BarDensity(logits,borders,law=NATIVE_FULL_SUPPORT)
    finite=BarDensity(logits,borders,law=LEGACY_FINITE)
    assert full.weights[-1]>0 and finite.weights[-1]==0
    loss=-full.log_prob(torch.tensor(30.,dtype=torch.float64));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_full_tail_grid_preserves_exactly_cancelled_collapsed_source_bins():
    from posteriorforge.study.model import full_support_cdf_grid
    logits=torch.tensor([[.2,-torch.inf,-.5,.8],[-.3,-torch.inf,.2,.4]],dtype=torch.float64,requires_grad=True)
    borders=torch.tensor([-2.,-1.,-1.,1.,3.],dtype=torch.float64)
    grid=torch.tensor([-35.,-2.,-1.,0.,1.,3.,35.],dtype=torch.float64)
    cdf,sf=full_support_cdf_grid(logits,borders,grid)
    compact=BarDensity(logits[:,[0,2,3]],torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64),law=NATIVE_FULL_SUPPORT)
    for j in range(2):
        reference=BarDensity(compact.logits[j],compact.borders,law=NATIVE_FULL_SUPPORT)
        assert torch.allclose(cdf[j],reference.cdf(grid),atol=2e-15,rtol=2e-15)
        assert torch.allclose(sf[j],reference.sf(grid),atol=2e-15,rtol=2e-15)
    (-torch.log(cdf[:,3])).sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad[:,1],torch.zeros(2,dtype=torch.float64))
    bad=logits.detach().clone();bad[:,1]=-1000.
    with pytest.raises(ModelQualificationError,match='positive-mass zero-width'):full_support_cdf_grid(bad,borders,grid)
    with pytest.raises(ModelQualificationError,match='outer-tail scales'):full_support_cdf_grid(logits.detach(),torch.tensor([-1.,-1.,-1.,1.,3.]),grid)


def test_canonical_full_tail_preserves_all_eight_projected_masses_and_gradients():
    from posteriorforge.study.model import canonical_full_support_mixture,translate_full_support_logits
    canonical=torch.tensor([-3.,-.5,.8,3.],dtype=torch.float64)
    members=[];borders=[];independent=[]
    for i in range(8):
        logits=torch.tensor([[.2+.03*i,-.4,.7],[-.3,.2,.1+.04*i]],dtype=torch.float64,requires_grad=True)
        source=torch.tensor([-2.5-.1*i,-.7,.9,2.8+.03*i],dtype=torch.float64)
        rows=[]
        for row in logits:
            native=BarDensity(row,source,law=NATIVE_FULL_SUPPORT)
            c=native.cdf(canonical[1:-1]);s=native.sf(canonical[1:-1])
            rows.append(torch.stack((c[0],c[1]-c[0],s[-1])))
        raw=torch.stack(rows)
        assert torch.allclose(translate_full_support_logits(logits,source,canonical),raw,atol=2e-15,rtol=2e-15)
        members.append(logits);borders.append(source);independent.append(raw)
    density=canonical_full_support_mixture(members,borders,canonical)
    expected=torch.stack(independent).mean(0);expected=expected/expected.sum(-1,keepdim=True)
    assert density.law==CANONICAL_FULL_SUPPORT
    assert torch.allclose(density.weights,expected,atol=2e-15,rtol=2e-15)
    points=torch.tensor([.1,-.2],dtype=torch.float64)
    (-density.log_prob(points).mean()).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and bool((x.grad!=0).any()) for x in members)
    delta=1e-5
    derivative=(density.cdf(points+delta)-density.cdf(points-delta))/(2*delta)
    assert torch.allclose(derivative,density.pdf(points),atol=2e-10,rtol=1e-7)
    u=torch.tensor([.00001,.99999],dtype=torch.float64)
    assert torch.allclose(density.cdf(density.icdf(u)),u,atol=2e-13,rtol=2e-13)
    with pytest.raises(ValueError,match='all 8'):canonical_full_support_mixture(members[:-1],borders[:-1],canonical)


def test_canonical_full_tail_identity_affine_and_native_criterion_parity():
    from posteriorforge.study.model import canonical_full_support_mixture
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    logits=torch.tensor([.2,-.5,.8],dtype=torch.float64)
    borders=torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64)
    native=BarDensity(logits,borders,law=NATIVE_FULL_SUPPORT)
    density=canonical_full_support_mixture([logits]*8,[borders]*8,borders)
    values=torch.tensor([-20.,-.5,0.,2.,25.],dtype=torch.float64)
    assert torch.allclose(density.log_prob(values),native.log_prob(values),atol=3e-14,rtol=3e-14)
    criterion=FullSupportBarDistribution(borders)
    assert torch.allclose(-density.log_prob(values),criterion(logits.expand(len(values),-1),values),atol=3e-5,rtol=3e-6)
    scale=2.;shift=.5
    affine=canonical_full_support_mixture([logits],[borders],borders,expected_members=1,sampling_borders=scale*borders+shift)
    assert torch.allclose(affine.log_prob(scale*values+shift),native.log_prob(values)-np.log(scale),atol=3e-14,rtol=3e-14)
    assert torch.allclose(affine.cdf(scale*values+shift),native.cdf(values),atol=3e-15,rtol=3e-15)
    u=torch.linspace(.0001,.9999,101,dtype=torch.float64)
    assert torch.allclose(affine.icdf(u),scale*native.icdf(u)+shift,atol=3e-13,rtol=3e-13)


def test_canonical_projection_retains_direct_upper_survival_below_cdf_resolution():
    from posteriorforge.study.model import canonical_full_support_mixture,translate_full_support_logits
    logits=torch.tensor([0.,0.,0.],dtype=torch.float64,requires_grad=True)
    source=torch.tensor([-2.,-1.,1.,3.],dtype=torch.float64)
    canonical=torch.tensor([-100.,20.,30.,100.],dtype=torch.float64)
    native=BarDensity(logits,source,law=NATIVE_FULL_SUPPORT)
    assert native.cdf(canonical[-2])==1 and native.sf(canonical[-2])>0
    raw=translate_full_support_logits(logits,source,canonical)
    assert raw[-1]>0 and raw[-1]==native.sf(canonical[-2])
    density=canonical_full_support_mixture([logits],[source],canonical,expected_members=1)
    loss=-density.log_prob(torch.tensor(130.,dtype=torch.float64));loss.backward()
    assert density.weights[-1]>0 and torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_canonical_projection_recovers_upper_mass_when_cdf_difference_loses_it():
    from posteriorforge.study.model import translate_full_support_logits

    logits = torch.tensor([0., 3., 0., -45., -46.], dtype=torch.float32, requires_grad=True)
    source = torch.tensor([-1., 0., 1., 2., 3., 4.], dtype=torch.float32)
    canonical = torch.tensor([-1., 0., 2.99, 3., 4.], dtype=torch.float32)
    raw = translate_full_support_logits(logits, source, canonical)
    expected = torch.softmax(logits.double(), -1)[-2] * (3 - canonical.double()[2])
    assert 0 < raw[2] < 1e-22
    torch.testing.assert_close(raw[2], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(raw.sum(), raw.new_tensor(1.), atol=2e-15, rtol=0)


def test_canonical_projection_integrates_uncertain_bins_in_batched_center():
    from posteriorforge.study.model import translate_full_support_logits

    dtype = torch.float64
    logits = torch.tensor([[[0., -50., 0.], [.3, 1., -.2]], [[-.1, 2., .2], [0., -60., 0.]]],
                          dtype=dtype, requires_grad=True)
    source = torch.tensor([0., 1., 2., 3.], dtype=dtype)
    canonical = torch.tensor([0., 1.1, 1.2, 3.], dtype=dtype)
    raw = translate_full_support_logits(logits, source, canonical)
    probability = torch.softmax(logits.detach(), -1)
    expected = probability[..., 1] * (canonical[2] - canonical[1])
    torch.testing.assert_close(raw[..., 1], expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(raw.sum(-1), torch.ones_like(expected), atol=2e-15, rtol=0)
    (-raw[..., 1].log().sum()).backward()
    probability[..., 1] -= 1
    torch.testing.assert_close(logits.grad, probability, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize('direction', [-1, 1])
def test_canonical_projection_integrates_narrow_half_normal_slices(direction):
    import math
    from posteriorforge.study.model import _HALF_NORMAL_MEDIAN, translate_full_support_logits

    dtype = torch.float64
    logits = torch.tensor([0., 0., 0.], dtype=dtype, requires_grad=True)
    source = torch.tensor([-1., 0., 1., 2.], dtype=dtype)
    canonical = torch.tensor([-2., -1e-19, 0., 1., 3.], dtype=dtype)
    if direction == 1:
        source, canonical = -source.flip(-1), -canonical.flip(-1)
    mass = translate_full_support_logits(logits, source, canonical)[1 if direction == -1 else 2]
    expected = math.erf(1e-19 * _HALF_NORMAL_MEDIAN / math.sqrt(2)) / 3
    assert expected > 0 and mass.item() == pytest.approx(expected, abs=0, rel=1e-13)
    (-mass.log()).backward()
    assert torch.isfinite(logits.grad).all()


def test_canonical_projection_recovers_outer_mass_from_small_half_normal_pieces():
    import math
    from posteriorforge.study.model import _HALF_NORMAL_MEDIAN, translate_full_support_logits

    logits = torch.tensor([-50., 0.], dtype=torch.float64)
    source = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    canonical = torch.tensor([-2., 1e-19, 2.], dtype=torch.float64)
    projected = translate_full_support_logits(logits, source, canonical)
    probability = torch.softmax(logits, -1)
    expected = probability[0] + probability[1] * math.erf(1e-19 * _HALF_NORMAL_MEDIAN / math.sqrt(2))
    torch.testing.assert_close(projected[0], expected, rtol=1e-13, atol=0)


def test_canonical_full_tail_cancelled_source_sampling_is_coherent_and_unclipped():
    from scipy.integrate import quad
    from posteriorforge.study.model import canonical_full_support_mixture,sampling_law_metadata
    logits=torch.tensor([.2,-torch.inf,-.5,.8],dtype=torch.float64)
    source=torch.tensor([-2.,-1.,-1.,1.,3.],dtype=torch.float64)
    canonical=torch.tensor([-3.,-.5,.8,3.],dtype=torch.float64)
    density=canonical_full_support_mixture([logits]*8,[source]*8,canonical)
    integral=sum(quad(lambda x:float(density.pdf(torch.tensor(x,dtype=torch.float64))),a,b,epsabs=1e-10)[0] for a,b in [(-np.inf,-.5),(-.5,.8),(.8,np.inf)])
    assert integral==pytest.approx(1.,abs=2e-10)
    values=density.sample((128,),generator=torch.Generator().manual_seed(981))
    assert torch.isfinite(values).all() and values.min()<canonical[0] and values.max()>canonical[-1]
    assert torch.allclose(density.cdf(density.icdf(torch.tensor([1e-8,.5,1-1e-8],dtype=torch.float64))),torch.tensor([1e-8,.5,1-1e-8],dtype=torch.float64),atol=2e-13,rtol=2e-13)
    metadata=sampling_law_metadata(CANONICAL_FULL_SUPPORT)
    assert metadata['source_policy']=='cancelled_zero_width' and metadata['members']==8
    assert not metadata['scientifically_qualified'] and not metadata['positive_mass_floor']
