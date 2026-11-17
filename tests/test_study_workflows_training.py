import json
from pathlib import Path
import pytest
pytest.importorskip('torch')
from posteriorforge.study import workflows as w
from test_study_campaign import TinyBackend,problem,diagnostic_spec


def request_file(tmp_path):
    for name,values in [('train',[problem(),problem(1)]),('validation',[problem(9)])]:
        (tmp_path/(name+'.json')).write_text(json.dumps({'setups':[s.to_dict() for s in values]}))
    return w.TrainingRequest('train.json',w.file_sha256(tmp_path/'train.json'),
        'validation.json',w.file_sha256(tmp_path/'validation.json'),'unused.ckpt','shared-cache',diagnostic_spec().to_dict())


def test_training_request_plan_and_complete_resume_via_file_driver(tmp_path):
    request=request_file(tmp_path)
    assert request.plan()['policy']['quick_validation_scores_per_check']==320
    factory=lambda spec,path:TinyBackend(spec)
    partial=w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=factory,stop_after_update=2)
    assert partial['completed_updates']==2 and partial['quick_validation_scores_per_check']==16
    complete=w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=factory,resume_from=partial['last_checkpoint'])
    assert complete['completed_updates']==4 and complete['data_cursor']==8
    assert complete['gpu_jobs_submitted']==complete['posterior_draws']==0
    assert Path(complete['last_checkpoint']).is_dir()
    again=w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=factory,resume_from=complete['last_checkpoint'])
    assert again['reused_complete_campaign'] and again['model_sha256']==complete['model_sha256']


def test_training_file_driver_retains_failed_attempt_and_recovers_from_update_zero(tmp_path):
    request=request_file(tmp_path)
    with pytest.raises(RuntimeError,match='injected'):
        w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
            allow_unqualified_cpu_diagnostic=True,backend_factory=lambda spec,path:TinyBackend(spec,fail_update=1))
    initial=tmp_path/'run/campaign/checkpoints/update-000000'
    assert initial.is_dir()
    assert json.loads((tmp_path/'run/run.json').read_text())['status']=='failed'
    recovered=w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
        allow_unqualified_cpu_diagnostic=True,backend_factory=lambda spec,path:TinyBackend(spec),resume_from=initial)
    assert recovered['completed_updates']==4 and recovered['costs']['failed_or_incomplete_attempts']==1
    assert len(list((tmp_path/'run/driver-attempts').glob('*.json')))==2


def test_training_request_checks_catalogue_identity_before_model_loading(tmp_path):
    request=request_file(tmp_path)
    (tmp_path/'train.json').write_text('{}')
    def should_not_load(*a,**k):raise AssertionError('not reached')
    with pytest.raises(ValueError,match='catalogue checksum'):
        w.execute_training_request(request,output=tmp_path/'run',base_directory=tmp_path,
            allow_unqualified_cpu_diagnostic=True,backend_factory=should_not_load)
    assert json.loads((tmp_path/'run/run.json').read_text())['status']=='failed'


def test_training_cli_routes_stop_and_resume_flags(tmp_path,capsys,monkeypatch):
    from posteriorforge.study.entrypoint import main
    from posteriorforge.study import campaign
    request=request_file(tmp_path)
    path=tmp_path/'request.json';path.write_text(json.dumps(request.to_dict()))
    monkeypatch.setattr(campaign.StockTrainingBackend,'from_checkpoint_factory',lambda spec,path:TinyBackend(spec))
    main(['study','train-plan','--config',str(path)])
    assert json.loads(capsys.readouterr().out)['campaign']['diagnostic'] is True
    args=['study','train','--config',str(path),'--output',str(tmp_path/'cli-run'),'--cpu-diagnostic']
    main(args+['--stop-after-update','2'])
    partial=json.loads(capsys.readouterr().out)
    assert partial['completed_updates']==2
    main(args+['--resume-from',partial['last_checkpoint']])
    complete=json.loads(capsys.readouterr().out)
    assert complete['completed_updates']==4 and complete['posterior_draws']==0
