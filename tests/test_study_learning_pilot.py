"""CPU fixtures for the separately authorized learning pilot workflow."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from posteriorforge.study import campaign, workflows
from posteriorforge.study.budget import StudyBudgetLedger
from posteriorforge.study.catalogue import MASTER_SEED


EXECUTION = {
    "backend": "stock_tabpfn_cuda_qualification",
    "parametrization_cache_policy": "per_layer_forward_and_checkpoint_recompute.v1",
    "fusion_enabled": False,
}
SOURCE = "a" * 40
JOB_ID = "123456"


@pytest.fixture
def payload():
    spec = campaign.CampaignSpec(
        "lora_r4", 0, "pilot", 1e-4, 200,
        workflows.protocol_identity()["protocol_sha256"], MASTER_SEED,
    )
    return {
        "schema": "finetuneforge.training_request.v2",
        "training_catalogue_file": "train.json",
        "training_catalogue_sha256": "b" * 64,
        "validation_catalogue_file": "validation.json",
        "validation_catalogue_sha256": "c" * 64,
        "base_checkpoint": "unused.ckpt",
        "data_cache_directory": "shared-cache",
        "campaign": spec.to_dict(),
        "execution": dict(EXECUTION),
    }


@pytest.mark.parametrize("field, value", [
    ("backend", "stock_tabpfn_cpu"),
    ("parametrization_cache_policy", "stock_uncached"),
    ("fusion_enabled", True),
    ("fusion_enabled", 0),
    ("additional_flag", False),
])
def test_v2_execution_contract_is_closed(payload, field, value):
    payload["execution"][field] = value
    with pytest.raises(ValueError):
        workflows.TrainingRequest.from_dict(payload)


def test_tiled_attention_is_an_explicit_request_identity(payload, tmp_path, monkeypatch):
    from posteriorforge.study import pilot

    stock = workflows.TrainingRequest.from_dict(payload)
    payload["execution"] = dict(pilot.CUDA_TILED_ATTENTION_EXECUTION)
    tiled = workflows.TrainingRequest.from_dict(payload)
    assert tiled.fingerprint != stock.fingerprint
    assert tiled.plan()["execution"] == pilot.CUDA_TILED_ATTENTION_EXECUTION
    assert tiled.plan()["execution"]["fusion_enabled"] is False
    for invalid in (True, float(512 * 1024 * 1024), 1024):
        invalid_request = dict(payload, execution=dict(payload["execution"], attention_max_score_bytes=invalid))
        with pytest.raises(ValueError, match="cache policy"):
            workflows.TrainingRequest.from_dict(invalid_request)

    def make_backend(spec, checkpoint, *, attention_max_score_bytes):
        assert attention_max_score_bytes == 512 * 1024 * 1024
        return "tiled"

    monkeypatch.setattr(pilot, "authorize_cuda_pilot", lambda **_: {})
    monkeypatch.setattr(workflows, "_make_cuda_pilot_backend", make_backend)
    monkeypatch.setattr(workflows, "_execute_training_request",
                        lambda request, *, spec, factory, **_: factory(spec, tmp_path / "base.ckpt"))
    assert workflows.execute_cuda_pilot_request(tiled, output=tmp_path / "tiled", ledger="unused.json") == "tiled"


def test_member_checkpointing_requires_tiled_attention_and_changes_request_identity(payload, tmp_path, monkeypatch):
    from posteriorforge.study import pilot

    checkpointed_policy = pilot.CUDA_MEMBER_RECOMPUTE_EXECUTION
    invalid = dict(payload, execution=dict(payload["execution"],
                    ensemble_member_policy=checkpointed_policy["ensemble_member_policy"]))
    with pytest.raises(ValueError, match="cache policy"):
        workflows.TrainingRequest.from_dict(invalid)
    tiled = workflows.TrainingRequest.from_dict(dict(payload, execution=pilot.CUDA_TILED_ATTENTION_EXECUTION))
    checkpointed = workflows.TrainingRequest.from_dict(dict(payload, execution=checkpointed_policy))
    assert checkpointed.fingerprint != tiled.fingerprint

    def factory(spec, checkpoint, *, attention_max_score_bytes, recompute_ensemble_members):
        assert attention_max_score_bytes == 512 * 1024 * 1024
        assert recompute_ensemble_members is True
        return "checkpointed"

    monkeypatch.setattr(pilot, "authorize_cuda_pilot", lambda **_: {})
    monkeypatch.setattr(workflows, "_make_cuda_pilot_backend", factory)
    monkeypatch.setattr(workflows, "_execute_training_request",
                        lambda request, *, spec, factory, **_: factory(spec, tmp_path / "base.ckpt"))
    assert workflows.execute_cuda_pilot_request(
        checkpointed, output=tmp_path / "recomputed", ledger="unused.json") == "checkpointed"


@pytest.mark.parametrize("changes", [
    {"phase": "main", "updates": 1000},
    {"method": "lora_r16"},
    {"method": "full_finetune", "learning_rate": 1e-5},
    {"diagnostic": True, "updates": 1},
    {"master_seed": MASTER_SEED + 1},
    {"protocol_sha256": "d" * 64},
    {"effective_batch": 15},
    {"updates": 199},
])
def test_v2_is_limited_to_declared_rank_four_learning_pilots(payload, changes):
    payload["campaign"].update(changes)
    with pytest.raises(ValueError):
        workflows.TrainingRequest.from_dict(payload)


def test_v2_round_trip_and_driver_isolation(payload, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    request = workflows.TrainingRequest.from_dict(payload)
    assert request.to_dict() == payload
    assert request.fingerprint == workflows.canonical_sha256(payload)
    plan = request.plan()
    assert plan["posterior_draws"] == plan["gpu_jobs_submitted"] == 0
    assert plan["scientifically_qualified"] is False
    changed = dict(payload, training_catalogue_sha256="d" * 64)
    assert workflows.TrainingRequest.from_dict(changed).fingerprint != request.fingerprint

    with pytest.raises(ValueError):
        workflows.TrainingRequest.from_dict({k: v for k, v in payload.items() if k != "execution"})
    with pytest.raises(ValueError):
        workflows.TrainingRequest.from_dict(dict(payload, schema="finetuneforge.training_request.v1"))
    with pytest.raises((ValueError, RuntimeError)):
        workflows.execute_training_request(
            request, output=tmp_path / "cpu", allow_unqualified_cpu_diagnostic=True,
            backend_factory=lambda *_: pytest.fail("CPU backend must stay unloaded"),
        )
    assert not (tmp_path / "cpu").exists()

    legacy = {k: v for k, v in payload.items() if k != "execution"}
    legacy["schema"] = "finetuneforge.training_request.v1"
    assert workflows.TrainingRequest.from_dict(legacy).to_dict() == legacy
    with pytest.raises((ValueError, RuntimeError)):
        workflows.execute_cuda_pilot_request(legacy, output=tmp_path / "cuda", ledger=tmp_path / "ledger.json")
    assert not (tmp_path / "cuda").exists()


def test_plan_and_unauthorized_execution_do_not_import_accelerator_libraries(payload, tmp_path):
    script = """\
import importlib.abc
import json
import os
from pathlib import Path
import sys

class BlockAcceleratorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'torch', 'peft', 'tabpfn', 'transformers'}:
            raise AssertionError('unexpected accelerator library import: ' + fullname)

sys.meta_path.insert(0, BlockAcceleratorImports())
os.environ.pop('SLURM_JOB_ID', None)
from posteriorforge.study.workflows import TrainingRequest, execute_cuda_pilot_request

request = TrainingRequest.from_dict(json.load(sys.stdin))
assert request.plan()['posterior_draws'] == 0
root = Path(sys.argv[1])
try:
    execute_cuda_pilot_request(request, output=root / 'run', ledger=root / 'ledger.json')
except RuntimeError as exc:
    assert 'Slurm' in str(exc), str(exc)
else:
    raise AssertionError('accepted CUDA pilot outside Slurm')
assert list(root.iterdir()) == []
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], input=json.dumps(payload),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    from posteriorforge.study import pilot

    for name, value in {
        "SLURM_JOB_ID": JOB_ID,
        "SLURM_JOB_ACCOUNT": "eu-26-17",
        "SLURM_JOB_PARTITION": "qgpu",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(pilot, "_source_identity", lambda: (tmp_path, SOURCE))
    return StudyBudgetLedger(tmp_path / "ledger.json")


def reserve(ledger, request, output, *, resume_from=None, stop_after_update=None,
            gpus=1, job_id=JOB_ID, metadata_changes=None):
    metadata = {
        "training_request_sha256": request.fingerprint,
        "source_commit": SOURCE,
        "output": str(output.resolve()),
        "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
        "stop_after_update": stop_after_update,
    }
    metadata.update(metadata_changes or {})
    reservation = ledger.reserve(gpus=gpus, walltime_seconds=60, purpose="CPU test fixture",
                                 metadata=metadata)
    if job_id is not None:
        ledger.attach_job(reservation["reservation_id"], job_id)
    return reservation


def test_authorization_accepts_exact_resume_and_stop_binding(payload, allocation, tmp_path):
    from posteriorforge.study.pilot import authorize_cuda_pilot

    request = workflows.TrainingRequest.from_dict(payload)
    output = tmp_path / "pilot"
    resume = output / "campaign/checkpoints/update-000020"
    entry = reserve(allocation, request, output, resume_from=resume, stop_after_update=40)
    before = allocation.path.read_bytes()
    receipt = authorize_cuda_pilot(
        request_sha256=request.fingerprint, ledger=allocation.path, output=output,
        resume_from=resume, stop_after_update=40,
    )
    assert receipt["job_id"] == JOB_ID
    assert receipt["reservation_id"] == entry["reservation_id"]
    assert receipt["source_commit"] == SOURCE
    assert allocation.path.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("environment, reservation", [
    ({"SLURM_JOB_ID": None}, {}),
    ({"SLURM_JOB_ID": "123456_1"}, {}),
    ({"SLURM_JOB_ACCOUNT": "different"}, {}),
    ({"SLURM_JOB_PARTITION": "qcpu"}, {}),
    ({"CUDA_VISIBLE_DEVICES": "0,1"}, {}),
    ({"CUBLAS_WORKSPACE_CONFIG": None}, {}),
    ({}, {"gpus": 2}),
    ({}, {"job_id": None}),
    ({}, {"job_id": "987654"}),
    ({}, {"metadata_changes": {"training_request_sha256": "d" * 64}}),
    ({}, {"metadata_changes": {"source_commit": "b" * 40}}),
    ({}, {"metadata_changes": {"output": "/different"}}),
    ({}, {"metadata_changes": {"resume_from": "/different/checkpoint"}}),
    ({}, {"metadata_changes": {"stop_after_update": 20}}),
])
def test_rejected_allocations_never_create_run_or_load_backend(
    payload, allocation, tmp_path, monkeypatch, environment, reservation,
):
    request = workflows.TrainingRequest.from_dict(payload)
    output = tmp_path / "pilot"
    reserve(allocation, request, output, **reservation)
    for name, value in environment.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def forbidden(*args, **kwargs):
        pytest.fail("unauthorized execution reached the catalogue or model loader")

    monkeypatch.setattr(workflows, "_pilot_catalogue", forbidden)
    monkeypatch.setattr(workflows, "_make_cuda_pilot_backend", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        workflows.execute_cuda_pilot_request(request, output=output, ledger=allocation.path)
    assert not output.exists()


def test_cuda_backend_factory_selects_cached_stock_path(payload, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from posteriorforge.study import cuda_training

    spec = campaign.CampaignSpec(**payload["campaign"])
    checkpoint = tmp_path / "unused.ckpt"
    chosen = {}

    def record_backend(self, actual_spec, actual_checkpoint, **kwargs):
        chosen.update(spec=actual_spec, checkpoint=actual_checkpoint, kwargs=kwargs)
        self.identity = {"parametrization_cache_policy": EXECUTION["parametrization_cache_policy"]}

    monkeypatch.setenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cuda_training.QualificationTrainingBackend, "__init__", record_backend)
    backend = workflows._make_cuda_pilot_backend(spec, checkpoint)
    assert isinstance(backend, cuda_training.CudaLearningPilotBackend)
    assert chosen["spec"] == spec
    assert chosen["checkpoint"] == checkpoint
    assert chosen["kwargs"] == {"allow_cuda_qualification": True, "cache_parametrizations": True,
                                "device": "cuda:0"}
    assert backend.identity["fusion_enabled"] is False


def test_cli_routes_cuda_pilot_arguments(payload, monkeypatch, tmp_path, capsys):
    from posteriorforge.study.entrypoint import main

    config = tmp_path / "study" / "request.json"
    config.parent.mkdir()
    config.write_text(json.dumps(payload))
    ledger = tmp_path / "ledger.json"
    output = tmp_path / "run"
    checkpoint = output / "campaign/checkpoints/update-000020"

    def execute(value, **kwargs):
        assert value == payload
        assert kwargs == {
            "base_directory": config.parent,
            "ledger": str(ledger),
            "output": str(output),
            "resume_from": str(checkpoint),
            "stop_after_update": 40,
        }
        return {"posterior_draws": 0}

    monkeypatch.setattr(workflows, "execute_cuda_pilot_request", execute)
    main(["study", "train-cuda-pilot", "--config", str(config), "--ledger", str(ledger),
          "--output", str(output), "--resume-from", str(checkpoint), "--stop-after-update", "40"])
    assert json.loads(capsys.readouterr().out) == {"posterior_draws": 0}


def test_gpu_pilot_backend_guard_rejects_cpu_model(payload):
    pytest.importorskip("torch")
    from test_study_campaign import TinyBackend

    spec = campaign.CampaignSpec(**payload["campaign"])
    backend = TinyBackend(spec)
    backend.backend_kind = EXECUTION["backend"]
    backend.identity.update(backend_kind=EXECUTION["backend"],
                            parametrization_cache_policy=EXECUTION["parametrization_cache_policy"])
    with pytest.raises(RuntimeError):
        campaign._require_cuda_pilot_backend(spec, backend)


def test_cpu_diagnostic_rejects_cuda_parameters_before_run(tmp_path):
    pytest.importorskip("torch")
    from test_study_campaign import diagnostic_spec

    parameter = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    backend = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter((parameter,))))
    output = tmp_path / "cpu"
    with pytest.raises(RuntimeError, match="GPU execution disabled"):
        campaign.run_cpu_diagnostic(
            diagnostic_spec(), stream=None, validation=None, store=None, backend=backend,
            output=output, allow_unqualified_cpu_diagnostic=True,
        )
    assert not output.exists()


def test_pilot_catalogue_authenticates_generated_definitions(tmp_path, monkeypatch):
    from posteriorforge.study.catalogue import generate_catalogue

    validation = generate_catalogue("validation")
    path = tmp_path / "validation.json"
    payload = {
        "schema": "posteriorforge.study.catalogue.v1",
        "split": "validation",
        "seed": MASTER_SEED,
        "generated_observation_cases": 0,
        "problem_fingerprints": [setup.fingerprint for setup in validation],
        "setups": [setup.to_dict() for setup in validation],
    }
    # CI generates its own fixture. The pilot ships the exact pinned catalogue.
    fixture_digest = workflows.canonical_sha256(payload["problem_fingerprints"])
    monkeypatch.setitem(workflows.PILOT_CATALOGUE_SHA256, "validation", fixture_digest)
    path.write_text(json.dumps(payload))
    canonical_sha256 = workflows.file_sha256(path)
    loaded = workflows._pilot_catalogue(path, canonical_sha256, split="validation")
    assert tuple(setup.fingerprint for setup in loaded) == tuple(setup.fingerprint for setup in validation)

    payload["setups"].reverse()
    payload["problem_fingerprints"].reverse()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        workflows._pilot_catalogue(path, canonical_sha256, split="validation")
    with pytest.raises(ValueError, match="fixed protocol listing"):
        workflows._pilot_catalogue(path, workflows.file_sha256(path), split="validation")


def test_pilot_checkpoint_resume_preserves_cursor_and_fixed_validation(
    payload, allocation, tmp_path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    from posteriorforge.study import model_loading
    from test_study_campaign import TinyBackend, problem

    request = workflows.TrainingRequest.from_dict(payload)
    output = tmp_path / "pilot"
    example = problem()
    training = tuple(replace(example, id=f"train-{i}", split="train") for i in range(10000))
    validation = tuple(replace(example, id=f"validation-{i}", split="validation") for i in range(20))
    sources = {"train": training, "validation": validation}
    base_checkpoint = tmp_path / payload["base_checkpoint"]
    base_checkpoint.write_bytes(b"CPU test checkpoint identity")
    monkeypatch.setattr(model_loading, "BASE_SHA256", workflows.file_sha256(base_checkpoint))

    def load_fixture(path, expected_sha256, *, split):
        name = "training" if split == "train" else "validation"
        assert path == tmp_path / payload[f"{name}_catalogue_file"]
        assert expected_sha256 == payload[f"{name}_catalogue_sha256"]
        return sources[split]

    class PilotFixture(TinyBackend):
        def __init__(self, spec):
            super().__init__(spec)
            self.identity.update(EXECUTION, backend_kind=EXECUTION["backend"])

        def update(self, items, state):
            assert len(items) == 16
            positions = [item.receipt["request"]["position"] for item in items]
            assert positions == list(range(state.data_position, state.data_position + 16))
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(torch.ones(1, 1, dtype=torch.float64)).square().sum()
            loss.backward()
            self.optimizer.step()
            state.update += 1
            state.data_position += 16
            return {"loss": float(loss.detach()), "update": state.update,
                    "data_position": state.data_position, "scientifically_qualified": False}

        def quick_validate(self, items):
            records = [
                {"request_id": item.receipt["request"]["request_id"],
                 "context_size": item.receipt["request"]["context_size"], "score": 1.,
                 "posterior_draws": 0}
                for item in items
            ]
            return {"score": 1., "score_count": len(records), "records": records, "posterior_draws": 0}

    monkeypatch.setattr(workflows, "_pilot_catalogue", load_fixture)
    monkeypatch.setattr(workflows, "_make_cuda_pilot_backend", lambda spec, _: PilotFixture(spec))
    monkeypatch.setattr(campaign, "_require_cuda_pilot_backend", lambda spec, backend: None)
    monkeypatch.setattr(campaign, "materialize_episode", lambda stream, episode, store:
                        SimpleNamespace(receipt={"request": episode.to_dict(), "posterior_draws": 0}))

    reserve(allocation, request, output, stop_after_update=20)
    partial = workflows.execute_cuda_pilot_request(
        request, output=output, ledger=allocation.path, base_directory=tmp_path, stop_after_update=20,
    )
    assert partial["status"] == "cuda_learning_pilot_checkpointed"
    assert partial["completed_updates"] == 20
    assert partial["data_cursor"] == 320
    checkpoint = partial["last_checkpoint"]

    next_job_id = "123457"
    monkeypatch.setenv("SLURM_JOB_ID", next_job_id)
    reserve(allocation, request, output, resume_from=checkpoint, job_id=next_job_id)
    complete = workflows.execute_cuda_pilot_request(
        request, output=output, ledger=allocation.path, base_directory=tmp_path, resume_from=checkpoint,
    )
    assert complete["status"] == "cuda_learning_pilot_complete"
    assert complete["completed_updates"] == 200
    assert complete["data_cursor"] == 3200
    assert complete["quick_validation_scores_per_check"] == 320
    assert complete["posterior_draws"] == complete["gpu_jobs_submitted"] == 0
    assert complete["scientifically_qualified"] is False
    assert complete["costs"]["gpu_hours"] is None
    assert partial["allocation"]["job_id"] == JOB_ID
    assert complete["allocation"]["job_id"] == next_job_id
    assert complete["history"][:20] == partial["history"]
    assert [row["update"] for row in complete["history"] if "quick_score" in row] == list(range(20, 201, 20))
    endpoint = Path(complete["last_checkpoint"])
    retained = campaign.read_campaign_checkpoint(endpoint)
    assert retained["identity"]["backend"]["parametrization_cache_policy"] == EXECUTION["parametrization_cache_policy"]
    identity = retained["identity"]["cuda_learning_pilot"]
    assert identity["execution"] == EXECUTION
    assert identity["source_commit"] == SOURCE
    assert identity["request_sha256"] == request.fingerprint
    assert retained["weights_sha256"] == complete["model_sha256"]
    assert (endpoint / "deployment" / retained["deployment"]["path"]).is_file()

    checks = []
    for path in sorted((output / "campaign/attempts").glob("*/attempt.json")):
        check = json.loads(path.read_text()).get("quick_validation")
        if check is not None:
            assert check["posterior_draws"] == 0
            checks.append([row["request_id"] for row in check["records"]])
    assert len(checks) == 10
    assert len(set(checks[0])) == 320
    assert all(check == checks[0] for check in checks)


def test_half_sampling_is_bound_to_the_explicit_request(payload, tmp_path, monkeypatch):
    from posteriorforge.study import pilot

    control = workflows.TrainingRequest.from_dict(dict(payload, execution=pilot.CUDA_MEMBER_RECOMPUTE_EXECUTION))
    policy = dict(pilot.CUDA_HALF_CONDITIONAL_EXECUTION)
    sampled = workflows.TrainingRequest.from_dict(dict(payload, execution=policy))
    assert sampled.plan()["execution"] == policy
    assert sampled.fingerprint != control.fingerprint
    assert control.to_dict()["execution"] == pilot.CUDA_MEMBER_RECOMPUTE_EXECUTION

    for changed in (
        dict(policy, conditional_sampling_percent=50.0),
        dict(policy, conditional_sampling_percent=25),
        dict(policy, conditional_sampling_policy="different.v1"),
        {key: value for key, value in policy.items() if key != "ensemble_member_policy"},
    ):
        with pytest.raises(ValueError, match="cache policy"):
            workflows.TrainingRequest.from_dict(dict(payload, execution=changed))

    def factory(spec, checkpoint, *, attention_max_score_bytes, recompute_ensemble_members, conditional_sampling_policy):
        assert attention_max_score_bytes == 512 * 1024 * 1024
        assert recompute_ensemble_members is True
        assert conditional_sampling_policy == policy["conditional_sampling_policy"]
        return "sampled"

    monkeypatch.setattr(pilot, "authorize_cuda_pilot", lambda **_: {})
    monkeypatch.setattr(workflows, "_make_cuda_pilot_backend", factory)
    monkeypatch.setattr(workflows, "_execute_training_request",
                        lambda request, *, spec, factory, **_: factory(spec, tmp_path / "base.ckpt"))
    assert workflows.execute_cuda_pilot_request(sampled, output=tmp_path / "run", ledger="unused.json") == "sampled"


def test_sampled_cuda_pilot_command_restarts_exactly_with_cpu_fixture(
    payload, allocation, tmp_path, monkeypatch, capsys,
):
    torch = pytest.importorskip("torch")
    from posteriorforge.study import cuda_training, model_loading, pilot
    from posteriorforge.study.conditional_sampling import select_half_conditionals
    from posteriorforge.study.entrypoint import main
    from posteriorforge.study.model import weights_fingerprint
    from posteriorforge.study.training import make_optimizer
    from posteriorforge.study.updates import _stagewise_update
    from test_study_campaign import TinyBackend, problem
    from test_study_updates import FixtureEnsemble, Tracker, fixtures

    policy = dict(pilot.CUDA_HALF_CONDITIONAL_EXECUTION)
    payload["execution"] = policy
    request = workflows.TrainingRequest.from_dict(payload)
    config = tmp_path / "request.json"
    config.write_text(json.dumps(payload))
    base = tmp_path / payload["base_checkpoint"]
    base.write_bytes(b"CPU fixture for sampled CUDA request")
    monkeypatch.setattr(model_loading, "BASE_SHA256", workflows.file_sha256(base))
    training = tuple(replace(problem(), id=f"train-{i}", split="train") for i in range(10_000))
    validation = tuple(replace(problem(), id=f"validation-{i}", split="validation") for i in range(20))
    monkeypatch.setattr(workflows, "_pilot_catalogue",
                        lambda _path, _checksum, *, split: {"train": training, "validation": validation}[split])

    sample_episode, = fixtures(Tracker(), query_counts=(1,), dimensions=(4,))
    roles = sample_episode.roles

    def materialize(stream, episode, store):
        return SimpleNamespace(
            context=SimpleNamespace(spec=SimpleNamespace(roles=roles)),
            query=SimpleNamespace(theta=[[0.0]]),
            receipt={"request": episode.to_dict(), "posterior_draws": 0},
        )

    monkeypatch.setattr(campaign, "materialize_episode", materialize)
    monkeypatch.setattr(campaign, "quick_milestones", lambda updates: (2, updates))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class CpuEnvelope(cuda_training.CudaLearningPilotBackend):
        def __init__(self, spec, checkpoint, *, attention_max_score_bytes, recompute_ensemble_members,
                     conditional_sampling_policy):
            assert checkpoint == base
            assert attention_max_score_bytes == policy["attention_max_score_bytes"] and recompute_ensemble_members
            assert conditional_sampling_policy == policy["conditional_sampling_policy"]
            TinyBackend.__init__(self, spec)
            self.model = torch.nn.Linear(2, 3, dtype=torch.float64)
            self.optimizer = make_optimizer(self.model, self.training_config)
            self.cache_parametrizations = True
            self.conditional_sampling_policy = conditional_sampling_policy
            self.device = torch.device("cpu")
            self.identity.update(policy, backend_kind=policy["backend"], qualification_only=True,
                                 initial_weights_sha256=weights_fingerprint(self.model))

        export = TinyBackend.export

        def _measure(self, operation, *args):
            return operation(*args)

        def prepare(self, item, *, stage_indices=None):
            req = item.receipt["request"]
            tracker = Tracker()
            episode, = fixtures(tracker, query_counts=(1,), dimensions=(4,))
            if req["phase"] == "quick_validation":
                assert stage_indices is None
                return episode
            assert stage_indices == select_half_conditionals(roles, req["request_id"]).stage_indices
            # Exercise checkpoint restoration of both global and backend-local RNG.
            scale = 1.0 + torch.rand(()).item() + self.episode_rng.random()
            stages = tuple(
                replace(stage, ensemble=FixtureEnsemble(stage.ensemble.features * scale, tracker, stage.index))
                for stage in episode.stages if stage.index in stage_indices
            )
            return replace(episode, stages=stages)

    def cpu_update(model, optimizer, episodes, state, *, allow_cuda_qualification, expected_device,
                   cache_parametrizations, **kwargs):
        assert allow_cuda_qualification and expected_device.type == "cpu" and cache_parametrizations
        return _stagewise_update(model, optimizer, episodes, state, allow_unqualified_cpu_diagnostic=True, **kwargs)

    monkeypatch.setattr(cuda_training, "CudaLearningPilotBackend", CpuEnvelope)
    monkeypatch.setattr(cuda_training, "stagewise_cuda_qualification_update", cpu_update)

    def invoke(output, job, stop, checkpoint=None):
        monkeypatch.setenv("SLURM_JOB_ID", str(job))
        reserve(allocation, request, output, resume_from=checkpoint, stop_after_update=stop, job_id=str(job))
        args = ["study", "train-cuda-pilot", "--config", str(config), "--output", str(output),
                "--ledger", str(allocation.path), "--stop-after-update", str(stop)]
        if checkpoint is not None:
            args.extend(("--resume-from", checkpoint))
        main(args)
        return json.loads(capsys.readouterr().out)

    uninterrupted = invoke(tmp_path / "full", 7651, 2)
    first = invoke(tmp_path / "segmented", 7652, 1)
    resumed = invoke(tmp_path / "segmented", 7653, 2, first["last_checkpoint"])
    left, right = (Path(run["last_checkpoint"]) for run in (uninterrupted, resumed))
    retained = campaign.read_campaign_checkpoint(right)
    assert retained["identity"]["cuda_learning_pilot"]["execution"] == policy
    assert retained["identity"]["cuda_learning_pilot"]["request_sha256"] == request.fingerprint
    assert retained["history"][0]["conditional_sampling_plan"] == first["history"][0]["conditional_sampling_plan"]
    assert retained["history"] == resumed["history"]
    assert [{k: v for k, v in row.items() if k != "checkpoint"} for row in resumed["history"]] == [
        {k: v for k, v in row.items() if k != "checkpoint"} for row in uninterrupted["history"]
    ]
    assert resumed["data_cursor"] == uninterrupted["data_cursor"] == 32
    assert resumed["model_sha256"] == uninterrupted["model_sha256"] and resumed["posterior_draws"] == 0
    assert "quick_score" in retained["history"][1] and resumed["quick_validation_scores_per_check"] == 320
    plans = retained["history"][1]["conditional_sampling_plan"]["episodes"]
    assert len({row["request_id"] for row in plans}) == 16
    assert all(len(row["stage_indices"]) == 3 for row in plans)

    a, b = (torch.load(p / "training" / "training.pt", weights_only=True) for p in (left, right))
    for name, value in a["model"].items():
        assert torch.equal(value, b["model"][name])
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for param, state in a["optimizer"]["state"].items():
        assert all(torch.equal(value, b["optimizer"]["state"][param][key]) for key, value in state.items())
    incompatible = json.loads(json.dumps(retained["identity"]))
    incompatible["cuda_learning_pilot"]["execution"] = pilot.CUDA_MEMBER_RECOMPUTE_EXECUTION
    with pytest.raises(campaign.CampaignQualificationError, match="identity mismatch"):
        campaign.read_campaign_checkpoint(right, expected_identity=incompatible)
