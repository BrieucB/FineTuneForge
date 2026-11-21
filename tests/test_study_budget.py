import json
import pytest
from posteriorforge.budget import BudgetLedger as LegacyBudget
from posteriorforge.study.budget import StudyBudgetLedger, BudgetError


def test_distinct_cap_does_not_widen_or_rewrite_legacy_ledger(tmp_path):
    path=tmp_path/'old.json';old=LegacyBudget(path)
    old.reserve(gpus=1,walltime_seconds=30,purpose='unit test')
    before=path.read_bytes()
    with pytest.raises(BudgetError):StudyBudgetLedger(path).summary()
    assert path.read_bytes()==before and old.summary()['cap_gpu_hours']==200
    new=StudyBudgetLedger(tmp_path/'new.json')
    assert new.summary()['cap_gpu_hours']==10000
    with pytest.raises(ValueError):StudyBudgetLedger(tmp_path/'bad',cap_gpu_hours=200)


def test_cpu_submission_guard_runs_before_files_reservations_or_subprocess(tmp_path,monkeypatch):
    import posteriorforge.study.budget as b
    monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:pytest.fail('No Slurm call is permitted'))
    ledger=StudyBudgetLedger(tmp_path/'ledger.json')
    with pytest.raises(BudgetError,match='disabled'):
        ledger.submit('/no/script',gpus=1,walltime_seconds=60,purpose='not authorized',job_name='pf-disabled',log_directory=tmp_path/'logs')
    assert not ledger.path.exists() and not (tmp_path/'logs').exists()


def test_actual_allocated_gpu_cost_and_uncertain_state_remain_conservative(tmp_path):
    ledger=StudyBudgetLedger(tmp_path/'new.json')
    held=ledger.reserve(gpus=1,walltime_seconds=3600,purpose='fake scheduler unit fixture')
    ledger.attach_job(held['reservation_id'],'123')
    record={'JobIDRaw':'123','Account':'eu-26-17','Partition':'qgpu','State':'OUT_OF_MEMORY','ElapsedRaw':'3600','AllocTRES':'cpu=64,gres/gpu=8'}
    ledger.reconcile(held['reservation_id'],record)
    assert ledger.summary()['actual_gpu_hours']==8
    assert ledger.summary()['reserved_gpu_hours']==0
    again=ledger.reserve(gpus=1,walltime_seconds=1,purpose='fake overrun fixture')
    ledger.attach_job(again['reservation_id'],'124')
    ledger.reconcile(again['reservation_id'],dict(record,JobIDRaw='124',ElapsedRaw=str(10000*3600),AllocTRES='gres/gpu=1'))
    assert ledger.summary()['cap_exceeded']
    with pytest.raises(BudgetError):ledger.reserve(gpus=1,walltime_seconds=1,purpose='must refuse')


@pytest.mark.parametrize('outcome', ['timeout', 'nonzero'])
def test_failed_submission_holds_uncertain_reservation(tmp_path, monkeypatch, outcome):
    import subprocess
    import posteriorforge.study.budget as budget

    script = tmp_path / 'job.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    ledger = StudyBudgetLedger(tmp_path / 'ledger.json', allow_gpu_jobs=True)

    def fake_run(argv, **kwargs):
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return subprocess.CompletedProcess(argv, 1, stdout='', stderr='failure')

    monkeypatch.setattr(budget.subprocess, 'run', fake_run)
    with pytest.raises(budget.SubmissionUncertain):
        ledger.submit(script, gpus=2, walltime_seconds=60, purpose='mock submission',
                      job_name='pf-test', log_directory=tmp_path / 'logs')
    reloaded = StudyBudgetLedger(ledger.path)
    entry, = reloaded.inspect()['reservations']
    assert entry['state'] == 'uncertain'
    assert reloaded.summary()['reserved_gpu_seconds'] == 120
    with pytest.raises(BudgetError, match='cannot be released'):
        reloaded.release_unsubmitted(entry['reservation_id'], reason='unsafe release')
    assert StudyBudgetLedger(ledger.path).summary()['reserved_gpu_seconds'] == 120


def test_submission_persists_reservation_before_scheduler_call(tmp_path, monkeypatch):
    import subprocess
    import posteriorforge.study.budget as budget

    script = tmp_path / 'job.sh'
    script.write_text('#!/bin/sh\nexit 0\n')
    ledger = StudyBudgetLedger(tmp_path / 'ledger.json', allow_gpu_jobs=True)
    observed = []

    def fake_run(argv, **kwargs):
        # Read the durable file directly: submit holds the ledger lock during this call.
        entry, = json.loads(ledger.path.read_text())['reservations']
        assert entry['state'] == 'submitting'
        assert entry['reserved_gpu_seconds'] == 120
        assert entry['submitted_argv'] == argv
        observed.append(entry['reservation_id'])
        return subprocess.CompletedProcess(argv, 0, stdout='12345\n', stderr='')

    monkeypatch.setattr(budget.subprocess, 'run', fake_run)
    entry = ledger.submit(script, gpus=2, walltime_seconds=60, purpose='mock submission',
                          job_name='pf-test', log_directory=tmp_path / 'logs')
    assert observed == [entry['reservation_id']]
    persisted, = StudyBudgetLedger(ledger.path).inspect()['reservations']
    assert persisted['state'] == 'submitted' and persisted['job_id'] == '12345'
    assert StudyBudgetLedger(ledger.path).summary()['reserved_gpu_seconds'] == 120


def test_concurrent_reservations_cannot_exceed_cap_after_reload(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from posteriorforge.study.budget import CAP_GPU_SECONDS

    path = tmp_path / 'ledger.json'
    ledger = StudyBudgetLedger(path)
    seed = ledger.reserve(gpus=1, walltime_seconds=1, purpose='prior allocation')
    ledger.attach_job(seed['reservation_id'], '123')
    ledger.reconcile(seed['reservation_id'], {
        'JobIDRaw': '123', 'Account': 'eu-26-17', 'Partition': 'qgpu',
        'State': 'COMPLETED', 'ElapsedRaw': str(CAP_GPU_SECONDS - 4),
        'AllocTRES': 'gres/gpu=1',
    })
    barrier = Barrier(8)

    def attempt(_):
        barrier.wait()
        try:
            return StudyBudgetLedger(path).reserve(gpus=1, walltime_seconds=1, purpose='concurrent')['reservation_id']
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(8)))
    accepted = [identity for identity in results if identity is not None]
    assert len(accepted) == len(set(accepted)) == 4
    reloaded = StudyBudgetLedger(path)
    summary = reloaded.summary()
    assert summary['actual_gpu_seconds'] == CAP_GPU_SECONDS - 4
    assert summary['reserved_gpu_seconds'] == 4
    assert summary['remaining_gpu_seconds'] == 0
    assert {entry['reservation_id'] for entry in reloaded.inspect()['reservations']} == set(accepted) | {seed['reservation_id']}


@pytest.mark.parametrize(("gpus", "nodes", "per_node"), [(4, 1, 4), (8, 1, 8), (16, 2, 8), (32, 4, 8)])
def test_qgpu_full_node_layout_reserves_total_and_reconciles(gpus, nodes, per_node, tmp_path, monkeypatch):
    import subprocess
    from posteriorforge.study import budget

    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    ledger = StudyBudgetLedger(tmp_path / "ledger.json", allow_gpu_jobs=True)

    def submit(argv, **kwargs):
        assert f"--nodes={nodes}" in argv and f"--ntasks={nodes}" in argv
        assert f"--gpus-per-node={per_node}" in argv and f"--cpus-per-task={16 * per_node}" in argv
        row, = json.loads(ledger.path.read_text())["reservations"]
        assert row["gpus"] == gpus and row["reserved_gpu_seconds"] == 60 * gpus
        return subprocess.CompletedProcess(argv, 0, stdout="12345\n", stderr="")

    monkeypatch.setattr(budget.subprocess, "run", submit)
    entry = ledger.submit(script, gpus=gpus, walltime_seconds=60, purpose="mock scaling check",
                          job_name="pf-test-scale", log_directory=tmp_path / "log")
    record = {"JobIDRaw": "12345", "Account": "eu-26-17", "Partition": "qgpu",
              "State": "COMPLETED", "ElapsedRaw": "10", "AllocTRES": f"gres/gpu={gpus}"}
    assert ledger.reconcile(entry["reservation_id"], record)["actual_gpu_seconds"] == gpus * 10


def test_qgpu_rejects_explicit_single_node_for_two_node_replay(tmp_path):
    script = tmp_path / "worker.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    ledger = StudyBudgetLedger(tmp_path / "ledger.json", allow_gpu_jobs=True)
    with pytest.raises(ValueError, match="single explicit node"):
        ledger.submit(script, gpus=16, walltime_seconds=60, purpose="mock scaling check",
                      job_name="pf-test-scale", log_directory=tmp_path / "log", node_name="acn05")
    assert not ledger.path.exists()
