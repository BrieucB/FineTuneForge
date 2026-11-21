import json
from pathlib import Path
import numpy as np
import pytest
from posteriorforge.study.catalogue import generate_setup
from posteriorforge.study.science import draw_prior
from posteriorforge.study.artifacts import save_samples,load_samples
from posteriorforge.study.banks import prepare_bank, save_bank


def test_prior_control_artifact_has_fifty_thousand_rows_and_authenticates(tmp_path):
    spec=generate_setup(0,split='diagnostic',size_band='small',physics=['scalar'])
    samples=draw_prior(spec,50000,12)
    m=save_samples(spec,samples,tmp_path/'control',kind='prior_control',provenance={'forward_evaluations':0})
    s,x,y,loaded=load_samples(tmp_path/'control',expected_manifest_sha256=m['manifest_sha256'])
    assert s.fingerprint==spec.fingerprint and np.array_equal(x,samples) and y is None
    assert loaded['status']=='samples_complete_unassessed' and not loaded['scientifically_qualified']
    with pytest.raises(FileExistsError):save_samples(spec,samples,tmp_path/'control',kind='prior_control',provenance={'forward_evaluations':0})
    (tmp_path/'control'/'samples.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):load_samples(tmp_path/'control')


def test_artifact_writer_cannot_accept_incomplete_or_unsupported_draws(tmp_path):
    spec=generate_setup(1,split='diagnostic',size_band='small',physics=['scalar'])
    x=draw_prior(spec,16,12)
    with pytest.raises(ValueError,match='50,000'):save_samples(spec,x,tmp_path/'partial')
    x=draw_prior(spec,50000,12)
    with pytest.raises(ValueError,match='provenance'):save_samples(spec,x,tmp_path/'unbound',observation=np.zeros(spec.observation_dimension))
    x[0,-1]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):save_samples(spec,x,tmp_path/'invalid',kind='prior_control',provenance={'forward_evaluations':0})


@pytest.mark.parametrize("kind", ["posterior", "prior_control"])
def test_load_revalidates_provenance_and_never_upgrades_completeness_to_accuracy(tmp_path, kind):
    spec=generate_setup(0,split='diagnostic',size_band='small',physics=['scalar'])
    samples=draw_prior(spec,50000,12)
    bank_dir = tmp_path / 'bank'
    bank_receipt = save_bank(prepare_bank(spec, 16, 7), bank_dir)
    provenance=({'model_sha256':'a'*64,'bank_manifest_sha256':bank_receipt['manifest_sha256'],'protocol_sha256':'c'*64}
                if kind=='posterior' else {'forward_evaluations':0})
    observation=np.zeros(spec.observation_dimension) if kind=='posterior' else None
    root=tmp_path/'record'
    receipt=save_samples(spec,samples,root,kind=kind,provenance=provenance,observation=observation,
                         **({'bank_directory':bank_dir, 'context_size':16} if kind=='posterior' else {}))
    _,_,_,loaded=load_samples(root,expected_manifest_sha256=receipt['manifest_sha256'])
    assert loaded['scientifically_qualified'] is False
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['provenance']={}
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='provenance|forward-call'):
        load_samples(root)
    manifest['provenance']=provenance
    manifest['status']='pass'
    manifest['scientifically_qualified']=True
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='manifest'):
        load_samples(root)


def test_posterior_retains_verified_prefix_and_shared_generation_cost(tmp_path):
    spec = generate_setup(0, split='diagnostic', size_band='small', physics=['scalar'])
    samples = draw_prior(spec, 50000, 12)
    bank_dir = tmp_path / 'bank'
    bank = prepare_bank(spec, 64, 7)
    receipt = save_bank(bank, bank_dir)
    provenance = {'model_sha256':'a'*64, 'bank_manifest_sha256':receipt['manifest_sha256'], 'protocol_sha256':'c'*64}
    kwargs = dict(observation=np.zeros(spec.observation_dimension), provenance=provenance,
                  bank_directory=bank_dir, context_size=16)
    saved = save_samples(spec, samples, tmp_path/'posterior', **kwargs)
    _, _, _, loaded = load_samples(tmp_path/'posterior', expected_manifest_sha256=saved['manifest_sha256'])
    cost = loaded['bank_cost']
    assert cost['context_forward_evaluations'] == bank.prefix(16).manifest['cost']['forward_evaluations']
    assert cost['shared_bank_generation_forward_evaluations'] == bank.manifest['cost']['forward_evaluations']
    assert cost['shared_bank_generation_forward_evaluations'] > cost['context_forward_evaluations']
    assert cost['posterior_sampling_forward_evaluations'] == 0
    manifest_path = tmp_path/'posterior'/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['bank_cost']['context_forward_evaluations'] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cost mismatch'):
        load_samples(tmp_path/'posterior')
    with pytest.raises(ValueError, match='context'):
        save_samples(spec, samples, tmp_path/'bad_context', **dict(kwargs, context_size=256))


def test_posterior_rejects_missing_cost_trace_and_tampered_bank_manifest(tmp_path):
    from hashlib import sha256
    spec = generate_setup(0, split='diagnostic', size_band='small', physics=['scalar'])
    samples = draw_prior(spec, 50000, 12)
    bank_dir = tmp_path/'bank'
    receipt = save_bank(prepare_bank(spec, 16, 7), bank_dir)
    provenance = {'model_sha256':'a'*64, 'bank_manifest_sha256':receipt['manifest_sha256'], 'protocol_sha256':'c'*64}
    kwargs = dict(observation=np.zeros(spec.observation_dimension), provenance=provenance,
                  bank_directory=bank_dir, context_size=16)
    save_samples(spec, samples, tmp_path/'good', **kwargs)
    retained = tmp_path/'good'/'bank_manifest.json'
    retained.write_bytes(retained.read_bytes()+b' ')
    with pytest.raises(ValueError, match='checksum'):
        load_samples(tmp_path/'good')
    bank_manifest = bank_dir/'manifest.json'
    payload = json.loads(bank_manifest.read_text())
    del payload['cost']['prefix_forward_evaluations']
    bank_manifest.write_text(json.dumps(payload))
    provenance['bank_manifest_sha256'] = sha256(bank_manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='cost trace'):
        save_samples(spec, samples, tmp_path/'missing', **kwargs)
