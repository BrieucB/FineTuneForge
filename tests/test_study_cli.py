import json
import subprocess
import sys
from pathlib import Path
import pytest
from posteriorforge.study.entrypoint import main
from posteriorforge.study.spec import ProblemSpec


def test_dry_plan_cli_preserves_fifty_thousand_and_never_submits(capsys):
    main(['study','plan'])
    result=json.loads(capsys.readouterr().out)
    assert result['maximum_npe_pfn_requests']==10800
    assert result['posterior_draws_per_request']==50000 and result['gpu_jobs_submitted']==0


def test_gpu_command_is_explicitly_disabled():
    with pytest.raises(RuntimeError,match='disabled'):main(['study','submit'])


def test_cli_and_contract_import_without_torch_or_jax():
    p=subprocess.run([sys.executable,'-c','import sys; from posteriorforge.study.entrypoint import _parser; _parser(); assert "torch" not in sys.modules and "jax" not in sys.modules'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_entrypoint_delegates_historical_commands_without_rewriting_them(monkeypatch):
    from posteriorforge import cli as historical
    calls=[]
    monkeypatch.setattr(historical, "main", lambda args: calls.append(args))
    main(["doctor"])
    assert calls == [["doctor"]]


def test_combined_help_includes_study(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "study" in capsys.readouterr().out



def test_prior_control_cli_is_readable_by_the_shared_artifact_loader(tmp_path, capsys):
    from posteriorforge.study.spec import PriorSpec,ParameterSpec,FamilySpec,DatasetSpec
    from posteriorforge.study.artifacts import load_samples
    spec=ProblemSpec('cli-unit','train',(
        ParameterSpec('mu','population',PriorSpec('normal',0,1)),
        ParameterSpec('a','physical',PriorSpec('normal','mu',.2)),
        ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.02))),
        (FamilySpec('f','scalar','sigma'),),(DatasetSpec('d','f','s',{'a':'a'}),))
    path=tmp_path/'spec.json'
    path.write_text(json.dumps(spec.to_dict()))
    main(['study','prior-check','--spec',str(path),'--seed','17','--output',str(tmp_path/'prior')])
    receipt=json.loads(capsys.readouterr().out)
    retained,x,y,manifest=load_samples(tmp_path/'prior',expected_manifest_sha256=receipt['manifest_sha256'])
    assert retained.fingerprint==spec.fingerprint and x.shape==(50000,3) and y is None
    assert manifest['provenance']['forward_evaluations']==0
    assert manifest['scientifically_qualified'] is False



def test_study_module_entrypoint_runs_without_accelerator_imports():
    result=subprocess.run([sys.executable,'-m','posteriorforge.study','plan'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert plan['posterior_draws_per_request']==50000
    assert plan['gpu_jobs_submitted']==0
