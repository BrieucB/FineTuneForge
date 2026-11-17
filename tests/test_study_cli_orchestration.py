from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from posteriorforge.study.entrypoint import main
from posteriorforge.study import workflows as w
from posteriorforge.study.banks import prepare_bank,save_bank
from posteriorforge.study.reference import observation_digest
from test_study_workflows import posterior_request,backend_factory
from test_study_banks import problem


def test_matched_campaign_planning_works_with_heavy_imports_blocked():
    code='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','peft','tabpfn','jax'):raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block())
from posteriorforge.study.entrypoint import main
main(['study','train-plan'])
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    plan=json.loads(result.stdout)
    assert len(plan['main_campaigns'])==6 and len(plan['pilot_campaigns'])==12
    assert plan['orchestration']['quick_validation_scores_per_check']==320
    assert plan['orchestration']['invalid_decoded_vectors']=='fail_request_preserve_evidence'
    assert not plan['scientifically_qualified'] and plan['gpu_jobs_submitted']==0


def test_posterior_cli_runs_and_recovers_exact_fifty_thousand(posterior_request,tmp_path,capsys,monkeypatch):
    path=tmp_path/'posterior-request.json';path.write_text(json.dumps(posterior_request.to_dict()))
    main(['study','posterior-plan','--config',str(path)])
    assert json.loads(capsys.readouterr().out)['posterior_draws']==50000
    with pytest.raises(RuntimeError,match='opt-in'):
        main(['study','posterior','--config',str(path),'--output',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
    monkeypatch.setattr(w,'make_posterior_backend',backend_factory())
    args=['study','posterior','--config',str(path),'--output',str(tmp_path/'out'),'--cpu-diagnostic']
    main(args)
    record=json.loads(capsys.readouterr().out)
    assert record['completed_draws']==50000 and record['gpu_jobs_submitted']==0
    main(args+['--resume'])
    again=json.loads(capsys.readouterr().out)
    assert again['reused_complete_artifact'] and again['artifact_manifest_sha256']==record['artifact_manifest_sha256']


def test_evaluation_cli_retains_reference_skips_and_seals_diagnostic_inputs(tmp_path,capsys,monkeypatch):
    spec=replace(problem(),split='validation')
    y=np.array([.3,-.2],dtype=np.float64)
    (tmp_path/'spec.json').write_text(json.dumps(spec.to_dict()));np.save(tmp_path/'query.npy',y)
    banks=[]
    directories={}
    for index,seed in enumerate((17,29)):
        name='bank-'+str(index)
        receipt=save_bank(prepare_bank(spec,64,seed),tmp_path/name)
        banks.append({'setup_id':'unit','bank_seed':index,'bank_manifest_sha256':receipt['manifest_sha256']})
        directories['unit/'+str(index)]={'directory':name}
    model={'version_id':'base','checkpoint_id':'base-checkpoint','model_sha256':'a'*64,
           'checkpoint_manifest_sha256':'b'*64,'method':'frozen','rank':None,'training_seed':None,'update':0}
    case={'setup_id':'unit','case_id':'case0','split':'validation','problem_fingerprint':spec.fingerprint,
          'observation_sha256':observation_digest(y),'reference_receipt_sha256':None}
    planning={'schema':'finetuneforge.evaluation_plan_request.v1','split':'validation','protocol_sha256':'c'*64,
              'models':[model],'cases':[case],'banks':banks,'scope':'diagnostic','context_sizes':[16]}
    plan_input=tmp_path/'planning.json';plan_input.write_text(json.dumps(planning))
    main(['study','evaluation-plan','--config',str(plan_input),'--output',str(tmp_path/'plan.json')])
    plan_record=json.loads(capsys.readouterr().out)
    assert plan_record['requests']==2
    value={'schema':'finetuneforge.evaluation_request.v1','plan_file':'plan.json',
           'plan_file_sha256':w.file_sha256(tmp_path/'plan.json'),
           'cases':{'unit/case0':{'spec_file':'spec.json','spec_file_sha256':w.file_sha256(tmp_path/'spec.json'),
               'observation_file':'query.npy','observation_file_sha256':w.file_sha256(tmp_path/'query.npy')}},
           'banks':directories,'models':{'base':{'base_checkpoint':'not-loaded.ckpt','model_sha256':'a'*64}}}
    request_path=tmp_path/'evaluation.json';request_path.write_text(json.dumps(value))
    def should_not_load(*a,**k):raise AssertionError('no qualified reference: sampler must not run')
    monkeypatch.setattr(w,'make_posterior_backend',should_not_load)
    command=['study','evaluate','--config',str(request_path),'--output',str(tmp_path/'out'),'--cpu-diagnostic']
    main(command)
    record=json.loads(capsys.readouterr().out)
    assert record['record_count']==2 and record['headline_savings_claim'] is False
    assert not (tmp_path/'out/posterior-runs').exists()
    records_dir=tmp_path/'out'/record['assessment_directory']
    main(['study','seal-selection','--records',str(records_dir),
        '--manifest-sha256',record['assessment_manifest_sha256'],'--plan-sha256',record['plan_sha256'],
        '--test-catalogue-sha256','d'*64,'--output',str(tmp_path/'selection.json')])
    sealed=json.loads(capsys.readouterr().out)
    assert sealed['scientifically_qualified'] is False
    assert len(sealed['selection_lock_sha256'])==64
    main(command+['--resume'])
    again=json.loads(capsys.readouterr().out)
    assert again['assessment_directory']!=record['assessment_directory']
    assert records_dir.exists()
