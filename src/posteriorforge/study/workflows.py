"""File-backed CPU orchestration; no GPU authorization or scheduler access."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import MappingProxyType
import uuid

import numpy as np
from .banks import CONTEXT_SIZES, load_bank
from .config import StudyConfig


POLICY = {
    "schema": "finetuneforge.orchestration_policy.v1",
    "training_bank_policy": "fresh_per_visit",
    "training_query_policy": "fresh_independent_one_case_per_episode",
    "matched_streams": "same_replica_and_phase_across_all_methods",
    "phase_streams": "pilot_and_main_disjoint",
    "quick_validation_cases": 40,
    "quick_validation_bank_seeds": 2,
    "quick_validation_contexts": [16, 64, 256, 1024],
    "quick_validation_scores_per_check": 320,
    "quick_validation_data": "fixed_across_milestones_and_methods",
    "pilot_updates": 200,
    "pilot_score": "mean_endpoint_score_across_both_replicas",
    "learning_rate_policy": "one_per_method_tie_smaller",
    "main_initialization": "restart_pretrained_fresh_adapter",
    "pilot_updates_in_main_allowance": False,
    "main_update_candidates": [1000, 3000, 10000],
    "effective_batch_cases": 16,
    "quick_validation_milestones": 10,
    "invalid_decoded_vectors": "fail_request_preserve_evidence",
    "posterior_draws": 50000,
    "selection_with_no_qualified_primary": "freeze_diagnostic_inputs_without_claims",
    "execution": "cpu_diagnostic_only_no_slurm",
}


def canonical_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " must be an explicit SHA256")
    return value


def protocol_identity():
    payload = {"study": StudyConfig().to_dict(), "orchestration": json.loads(json.dumps(POLICY))}
    return {"protocol_sha256": canonical_sha256(payload), **payload}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_guard(allowed):
    if allowed is not True:
        raise RuntimeError("execution requires explicit unqualified CPU-diagnostic opt-in; production remains unqualified")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU orchestration diagnostics cannot run in Slurm jobs")


def _resolve(value, base):
    p = Path(value)
    return p if p.is_absolute() else Path(base) / p


@dataclass(frozen=True)
class PosteriorRequest:
    bank_directory: str
    bank_manifest_sha256: str
    observation_file: str
    observation_file_sha256: str
    model: dict
    protocol_sha256: str
    context_size: int = 256
    seed: int = 0
    microbatch_size: int = 1024
    posterior_draws: int = 50000

    def __post_init__(self):
        for name in ("bank_directory", "observation_file"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + " is required")
        for name in ("bank_manifest_sha256", "observation_file_sha256", "protocol_sha256"):
            _sha(getattr(self, name), name)
        if type(self.context_size) is not int or self.context_size not in CONTEXT_SIZES:
            raise ValueError("one declared context size is required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("posterior seed must be a nonnegative integer")
        if type(self.microbatch_size) is not int or not 1 <= self.microbatch_size <= 50000:
            raise ValueError("microbatch size must lie in1..50000")
        if type(self.posterior_draws) is not int or self.posterior_draws != 50000:
            raise ValueError("every posterior request requires exactly50,000 complete draws")
        if not isinstance(self.model, Mapping) or set(self.model) - {"base_checkpoint", "model_sha256", "campaign_checkpoint", "checkpoint_manifest_sha256"}:
            raise ValueError("unsupported model specification")
        if not isinstance(self.model.get("base_checkpoint"), str) or not self.model["base_checkpoint"]:
            raise ValueError("pinned base_checkpoint path is required")
        _sha(self.model.get("model_sha256"), "model_sha256")
        if self.model.get("campaign_checkpoint") is not None:
            if not isinstance(self.model["campaign_checkpoint"], str) or not self.model["campaign_checkpoint"]:
                raise ValueError("campaign checkpoint path required")
            _sha(self.model.get("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256")
        elif self.model.get("checkpoint_manifest_sha256") is not None:
            raise ValueError("frozen inference requests omit campaign manifest fields; the base file identity is pinned")
        # Detach nested caller-owned state. This JSON is also the run identity.
        object.__setattr__(self, "model", MappingProxyType(json.loads(json.dumps(dict(self.model), sort_keys=True))))

    @classmethod
    def from_dict(cls, value):
        v = dict(value)
        if v.pop("schema", None) != "finetuneforge.posterior_request.v1":
            raise ValueError("unsupported posterior request schema")
        try:
            return cls(**v)
        except TypeError as exc:
            raise ValueError("missing or unsupported request fields") from exc

    def to_dict(self):
        from dataclasses import fields
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["model"] = dict(self.model)
        return {"schema": "finetuneforge.posterior_request.v1", **result}

    @property
    def fingerprint(self):
        return canonical_sha256(self.to_dict())

    def plan(self):
        return {"status": "planned_unqualified", "request_sha256": self.fingerprint,
                "posterior_draws": 50000, "context_size": self.context_size,
                "microbatch_size": self.microbatch_size,
                "complete_microbatches": (50000 + self.microbatch_size - 1) // self.microbatch_size,
                "invalid_decoded_vectors": "fail_request_preserve_evidence",
                "sampling_law": StudyConfig().sampling_law, "scientifically_qualified": False,
                "gpu_jobs_submitted": 0, "estimated_gpu_seconds": None}


def _read_observation(request, base):
    path = _resolve(request.observation_file, base)
    if file_sha256(path) != request.observation_file_sha256:
        raise ValueError("observation file checksum mismatch")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("one finite float64 observation-only vector is required")
    return value


def _authorize_evaluation(request, bank, observation, authorization):
    if authorization is None:
        if bank.spec.split == "test":
            raise RuntimeError("final-test sampling requires a frozen selection lock and verified evaluation request")
        return
    from .evaluation_campaign import verify_evaluation_request
    from .reference import observation_digest
    if not isinstance(authorization, dict) or set(authorization) != {"plan", "request_id", "selection_lock"}:
        raise ValueError("verified evaluation authorization required")
    r = verify_evaluation_request(authorization["plan"], authorization["request_id"],
                                  selection_lock=authorization["selection_lock"])
    expected = (request.protocol_sha256, bank.spec.fingerprint, observation_digest(observation),
                request.bank_manifest_sha256, request.model["model_sha256"], request.context_size,
                request.seed, 50000)
    actual = (r["protocol_sha256"], r["case"]["problem_fingerprint"], r["case"]["observation_sha256"],
              r["bank"]["bank_manifest_sha256"], r["model"]["model_sha256"], r["context_size"],
              r["posterior_seed"], r["posterior_draws"])
    if expected != actual or r["split"] != bank.spec.split:
        raise ValueError("posterior request does not match the frozen evaluation binding")
    trained = r["model"]["method"] != "frozen"
    if trained != bool(request.model.get("campaign_checkpoint")):
        raise ValueError("frozen/trained model role differs from evaluation selection")
    if trained and request.model.get("checkpoint_manifest_sha256") != r["model"]["checkpoint_manifest_sha256"]:
        raise ValueError("trained checkpoint differs from frozen evaluation selection")
    return {**r, "evaluation_scope": authorization["plan"]["scope"]}


def _validate_loaded_evaluation_model(binding, metadata):
    if binding is None:
        return
    expected = binding["model"]
    if metadata.get("checkpoint_manifest_sha256") != expected["checkpoint_manifest_sha256"]:
        raise ValueError("loaded checkpoint receipt differs from frozen evaluation selection")
    method = metadata.get("method")
    method = "lora" if method in ("lora_r4", "lora_r16", "lora") else "full" if method == "full_finetune" else method
    actual = (method, metadata.get("rank"), metadata.get("training_replica"), metadata.get("completed_updates"))
    wanted = (expected["method"], expected["rank"], expected["training_seed"], expected["update"])
    if actual != wanted:
        raise ValueError("loaded method, rank, replica or update differs from frozen evaluation model")
    if expected["method"] != "frozen" and binding.get("evaluation_scope") == "study" and metadata.get("training_diagnostic") is not False:
        raise ValueError("diagnostic training weights cannot represent a production study campaign")
    if expected["method"] != "frozen" and metadata.get("training_phase") != "main":
        raise ValueError("evaluation must use the selected main-training checkpoint")
    if expected["method"] != "frozen" and metadata.get("training_protocol_sha256") != binding["protocol_sha256"]:
        raise ValueError("trained checkpoint belongs to a different evaluation protocol")


def make_posterior_backend(model, *, base_directory):
    from .model_loading import load_inference_model
    from .sampling_backend import TabPFNConditionalBackend
    base = _resolve(model["base_checkpoint"], base_directory)
    campaign = model.get("campaign_checkpoint")
    network, metadata = load_inference_model(base, expected_model_sha256=model["model_sha256"],
        campaign_checkpoint=None if campaign is None else _resolve(campaign, base_directory),
        expected_checkpoint_manifest_sha256=model.get("checkpoint_manifest_sha256"))
    backend = TabPFNConditionalBackend(base, network, expected_model_sha256=model["model_sha256"])
    return backend, {**metadata, "backend_kind": backend.backend_kind}


@contextmanager
def _output_lock(output, request_sha256, *, resume):
    root = Path(output)
    if root.exists() and not resume:
        raise FileExistsError("output exists; explicit resume is required")
    if resume and not (root / "run.json").is_file():
        raise ValueError("no retained run to resume")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns this run directory") from exc
        try:
            if (root / "run.json").exists():
                old = json.loads((root / "run.json").read_text())
                if old.get("request_sha256") != request_sha256:
                    raise ValueError("resume request identity mismatch")
                if old.get("orchestration_source_sha256") != file_sha256(__file__):
                    raise ValueError("resume orchestration source identity mismatch")
            yield root
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute_posterior_request(request, *, output, base_directory=".", allow_unqualified_cpu_diagnostic=False,
                              resume=False, backend_factory=None, evaluation_authorization=None):
    """Run the complete file-backed50k path; failures never create a success artifact."""
    _cpu_guard(allow_unqualified_cpu_diagnostic)
    if not isinstance(request, PosteriorRequest):
        request = PosteriorRequest.from_dict(request)
    with _output_lock(output, request.fingerprint, resume=resume) as root:
        started = time.monotonic()
        attempt = uuid.uuid4().hex
        previous_record = json.loads((root / "run.json").read_text()) if (root / "run.json").exists() else {}
        previous_artifact_sha = (previous_record.get("artifact_manifest_sha256") or
            previous_record.get("pending_artifact_manifest_sha256") or previous_record.get("previous_artifact_manifest_sha256"))
        record = {"schema": "finetuneforge.posterior_run.v1", "request_sha256": request.fingerprint,
                  "request": request.to_dict(), "attempt_id": attempt, "status": "running",
                  "started_utc": datetime.now(timezone.utc).isoformat(), "requested_draws": 50000,
                  "scientifically_qualified": False, "gpu_jobs_submitted": 0, "allocated_gpu_hours": 0,
                  "orchestration_source_sha256": file_sha256(__file__),
                  "previous_artifact_manifest_sha256": previous_artifact_sha}
        _write_json(root / "run.json", record)
        try:
            bank_path = _resolve(request.bank_directory, base_directory)
            bank = load_bank(bank_path, expected_sha256=request.bank_manifest_sha256)
            observation = _read_observation(request, base_directory)
            if observation.shape != (bank.spec.observation_dimension,):
                raise ValueError("query observation order/dimension mismatch")
            evaluation_binding = _authorize_evaluation(request, bank, observation, evaluation_authorization)
            prefix = bank.prefix(request.context_size)
            artifact = root / "posterior"
            from .artifacts import load_samples, save_samples
            if artifact.exists():
                if not resume:
                    raise FileExistsError(artifact)
                if previous_artifact_sha is None:
                    raise ValueError("retained posterior is missing its committed manifest receipt")
                spec, samples, query, receipt = load_samples(artifact, expected_manifest_sha256=previous_artifact_sha)
                p = receipt["provenance"]
                _validate_loaded_evaluation_model(evaluation_binding, {
                    "method": p.get("model_method"), "rank": p.get("model_rank"),
                    "training_replica": p.get("training_replica"), "completed_updates": p.get("completed_updates"),
                    "training_diagnostic": p.get("training_diagnostic"),
                    "training_phase": p.get("training_phase"), "training_protocol_sha256": p.get("training_protocol_sha256"),
                    "checkpoint_manifest_sha256": p.get("checkpoint_manifest_sha256")})
                if (spec.fingerprint != bank.spec.fingerprint or not np.array_equal(query, observation)
                        or p.get("request_sha256") != request.fingerprint or p.get("posterior_seed") != request.seed
                        or any(p.get(k) != v for k, v in {"model_sha256": request.model["model_sha256"],
                            "bank_manifest_sha256": request.bank_manifest_sha256,
                            "protocol_sha256": request.protocol_sha256}.items())):
                    raise ValueError("retained posterior artifact does not match resumed request")
                record.update(status="samples_complete_unassessed", completed_draws=len(samples),
                    artifact_directory="posterior", artifact_manifest_sha256=file_sha256(artifact / "manifest.json"),
                    reused_complete_artifact=True)
            else:
                cold = time.monotonic()
                factory = make_posterior_backend if backend_factory is None else backend_factory
                backend, model_metadata = factory(request.model, base_directory=base_directory)
                cold = time.monotonic() - cold
                _validate_loaded_evaluation_model(evaluation_binding, model_metadata)
                if model_metadata.get("model_sha256") != request.model["model_sha256"]:
                    raise ValueError("loaded model identity mismatch")
                from .sampling import sample_posterior
                identities = {"model_sha256": request.model["model_sha256"],
                    "bank_manifest_sha256": request.bank_manifest_sha256, "protocol_sha256": request.protocol_sha256}
                samples, timing = sample_posterior(prefix, observation, backend, seed=request.seed,
                    microbatch_size=request.microbatch_size, identities=identities,
                    resume_path=root / "recovery.npz", diagnostic_cpu=True)
                timing["cold_model_load_seconds"] = cold
                provenance = {**identities, "posterior_seed": request.seed, "request_sha256": request.fingerprint,
                              "requested_sampling_law": StudyConfig().sampling_law,
                              "sampling_law": timing["binding"]["law"],
                              "source_policy": timing["binding"]["source_policy"],
                              "backend_kind": model_metadata.get("backend_kind", type(backend).__module__ + "." + type(backend).__qualname__),
                              "model_method": model_metadata.get("method"),
                              "model_rank": model_metadata.get("rank"),
                              "training_replica": model_metadata.get("training_replica"),
                              "completed_updates": model_metadata.get("completed_updates"),
                              "training_diagnostic": model_metadata.get("training_diagnostic"),
                              "training_phase": model_metadata.get("training_phase"),
                              "training_protocol_sha256": model_metadata.get("training_protocol_sha256"),
                              "loader_source_sha256": model_metadata.get("loader_source_sha256"),
                              "sampler_binding": timing["binding"],
                              "checkpoint_manifest_sha256": model_metadata.get("checkpoint_manifest_sha256"),
                              "orchestration_source_sha256": record["orchestration_source_sha256"]}
                staging = root / ("posterior-staging-" + attempt)
                receipt = save_samples(bank.spec, samples, staging, observation=observation,
                    provenance=provenance, bank_directory=bank_path, context_size=request.context_size)
                record["pending_artifact_manifest_sha256"] = receipt["manifest_sha256"]
                _write_json(root / "run.json", record)
                os.replace(staging, artifact)
                record.update(status="samples_complete_unassessed", completed_draws=len(samples),
                    artifact_directory="posterior", artifact_manifest_sha256=receipt["manifest_sha256"],
                    timings=timing, model=model_metadata, bank_cost=receipt["bank_cost"],
                    reused_complete_artifact=False)
        except BaseException as exc:
            record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                          artifact_directory=None, artifact_manifest_sha256=None)
            (root / ("traceback-" + attempt + ".txt")).write_text(traceback.format_exc())
            raise
        finally:
            record["wall_seconds_current_attempt"] = time.monotonic() - started
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(root / "attempts" / (attempt + ".json"), record)
            _write_json(root / "run.json", record)
        return {**record, "run_manifest_sha256": file_sha256(root / "run.json")}


@dataclass(frozen=True)
class TrainingRequest:
    training_catalogue_file: str
    training_catalogue_sha256: str
    validation_catalogue_file: str
    validation_catalogue_sha256: str
    base_checkpoint: str
    data_cache_directory: str
    campaign: dict

    def __post_init__(self):
        for name in ("training_catalogue_file", "validation_catalogue_file", "base_checkpoint", "data_cache_directory"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(name + " is required")
        for name in ("training_catalogue_sha256", "validation_catalogue_sha256"):
            _sha(getattr(self, name), name)
        from .campaign import CampaignSpec
        spec = CampaignSpec(**self.campaign)
        object.__setattr__(self, "campaign", MappingProxyType(spec.to_dict()))

    @classmethod
    def from_dict(cls, value):
        v = dict(value)
        if v.pop("schema", None) != "finetuneforge.training_request.v1":
            raise ValueError("unsupported training request schema")
        try:
            return cls(**v)
        except TypeError as exc:
            raise ValueError("missing or unsupported request fields") from exc

    def to_dict(self):
        from dataclasses import fields
        v = {f.name: getattr(self, f.name) for f in fields(self)}
        v["campaign"] = dict(self.campaign)
        return {"schema": "finetuneforge.training_request.v1", **v}

    @property
    def fingerprint(self):
        return canonical_sha256(self.to_dict())

    def plan(self):
        return {"status": "planned_unqualified", "request_sha256": self.fingerprint,
                "campaign": dict(self.campaign), "posterior_draws": 0, "gpu_jobs_submitted": 0,
                "policy": json.loads(json.dumps(POLICY)), "scientifically_qualified": False,
                "production_execution_enabled": False}


def _catalogue(path, expected_sha256):
    from .spec import ProblemSpec
    if file_sha256(path) != expected_sha256:
        raise ValueError("catalogue checksum mismatch")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not isinstance(value.get("setups"), list):
        raise ValueError("catalogue needs setup definitions")
    setups = tuple(ProblemSpec.from_dict(v) for v in value["setups"])
    if not setups:
        raise ValueError("empty catalogue")
    return setups


def execute_training_request(request, *, output, base_directory=".", allow_unqualified_cpu_diagnostic=False,
                             resume_from=None, stop_after_update=None, backend_factory=None):
    _cpu_guard(allow_unqualified_cpu_diagnostic)
    if not isinstance(request, TrainingRequest):
        request = TrainingRequest.from_dict(request)
    from .campaign import CampaignSpec, StockTrainingBackend, read_campaign_checkpoint, run_cpu_diagnostic
    from .episodes import EpisodeStream, QuickValidationStream, BankStore
    spec = CampaignSpec(**request.campaign)
    if not spec.diagnostic:
        raise RuntimeError("production campaigns remain blocked pending I07; use a declared bounded CPU diagnostic")
    with _output_lock(output, request.fingerprint, resume=resume_from is not None) as root:
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        record = {"schema": "finetuneforge.training_run.v1", "request_sha256": request.fingerprint,
                  "request": request.to_dict(), "attempt_id": attempt, "status": "running",
                  "scientifically_qualified": False, "gpu_jobs_submitted": 0, "posterior_draws": 0,
                  "orchestration_source_sha256": file_sha256(__file__)}
        _write_json(root / "run.json", record)
        try:
            if resume_from is not None:
                resume_from = Path(resume_from).resolve()
                if not resume_from.is_relative_to((root / "campaign" / "checkpoints").resolve()):
                    raise ValueError("resume must use a checkpoint retained in this run directory")
            train = _catalogue(_resolve(request.training_catalogue_file, base_directory), request.training_catalogue_sha256)
            validation = _catalogue(_resolve(request.validation_catalogue_file, base_directory), request.validation_catalogue_sha256)
            stream = EpisodeStream(train, protocol_sha256=spec.protocol_sha256, master_seed=spec.master_seed,
                                   replica=spec.replica, phase=spec.phase, diagnostic=True)
            checks = QuickValidationStream(validation, protocol_sha256=spec.protocol_sha256,
                                           master_seed=spec.master_seed, diagnostic=True)
            store = BankStore(_resolve(request.data_cache_directory, base_directory))
            factory = StockTrainingBackend.from_checkpoint_factory if backend_factory is None else backend_factory
            backend = factory(spec, _resolve(request.base_checkpoint, base_directory))
            result_path = root / "campaign" / "result.json"
            previous = json.loads(result_path.read_text()) if result_path.exists() else None
            if resume_from is not None and previous is not None and previous.get("state", {}).get("update") == spec.updates:
                from .campaign import campaign_identity
                retained = read_campaign_checkpoint(resume_from,
                    expected_identity=campaign_identity(spec, stream, checks, backend))
                if retained["state"] != previous["state"] or Path(previous["last_checkpoint"]).resolve() != resume_from:
                    raise ValueError("completed campaign checkpoint/result mismatch")
                result = previous
                record["reused_complete_campaign"] = True
            else:
                result = run_cpu_diagnostic(spec, stream=stream, validation=checks, store=store, backend=backend,
                    output=root / "campaign", allow_unqualified_cpu_diagnostic=True,
                    resume_from=resume_from, stop_after_update=stop_after_update)
                record["reused_complete_campaign"] = False
            checkpoint = Path(result["last_checkpoint"])
            retained = read_campaign_checkpoint(checkpoint)
            record.update(status=result["status"], result_file="campaign/result.json",
                result_sha256=file_sha256(root / "campaign" / "result.json"),
                completed_updates=result["state"]["update"], data_cursor=result["state"]["data_position"],
                last_checkpoint=str(checkpoint), checkpoint_manifest_sha256=file_sha256(checkpoint / "manifest.json"),
                model_sha256=retained["weights_sha256"], costs=result["costs"],
                quick_validation_scores_per_check=len(checks), history=result["history"], shortlist=result["shortlist"])
        except BaseException as exc:
            record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            (root / ("traceback-" + attempt + ".txt")).write_text(traceback.format_exc())
            raise
        finally:
            record["wall_seconds_current_attempt"] = time.monotonic() - started
            _write_json(root / "driver-attempts" / (attempt + ".json"), record)
            _write_json(root / "run.json", record)
        return {**record, "run_manifest_sha256": file_sha256(root / "run.json")}


def _execute_evaluation_request_unlocked(value, *, output, base_directory=".", allow_unqualified_cpu_diagnostic=False,
                               backend_factory=None, resume=False):
    """Consume a frozen matrix using file providers and the actual50k sampler."""
    _cpu_guard(allow_unqualified_cpu_diagnostic)
    required = {"schema", "plan_file", "plan_file_sha256", "cases", "banks", "models"}
    optional = {"selection_lock_file", "selection_lock_sha256", "test_catalogue_sha256"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        raise ValueError("evaluation request has unsupported/missing fields")
    if value["schema"] != "finetuneforge.evaluation_request.v1":
        raise ValueError("unsupported evaluation request schema")
    plan_path = _resolve(value["plan_file"], base_directory)
    if file_sha256(plan_path) != _sha(value["plan_file_sha256"], "plan_file_sha256"):
        raise ValueError("evaluation plan file checksum mismatch")
    plan = json.loads(plan_path.read_text())
    from .evaluation_campaign import (consume_evaluation_plan, verify_evaluation_request,
        load_selection_lock, summarize_validation, summarize_test)
    from .spec import ProblemSpec
    lock = None
    if plan["split"] == "test":
        if not optional <= set(value):
            raise ValueError("test evaluation requires an authenticated frozen selection lock")
        lock = load_selection_lock(_resolve(value["selection_lock_file"], base_directory),
            expected_sha256=value["selection_lock_sha256"], protocol_sha256=plan["protocol_sha256"],
            test_catalogue_sha256=value["test_catalogue_sha256"])
    # Verify a complete plan before creating outputs or invoking any provider.
    if not plan.get("requests"):
        raise ValueError("nonempty complete evaluation plan required")
    verify_evaluation_request(plan, plan["requests"][0]["request_id"], selection_lock=lock)
    outer = Path(output)
    if outer.exists() and not resume:
        raise FileExistsError("evaluation output exists; explicit resume required")
    outer.mkdir(parents=True, exist_ok=True)
    def case_key(case):
        return case["setup_id"] + "/" + case["case_id"]
    def case_provider(case):
        binding = value["cases"][case_key(case)]
        path = _resolve(binding["spec_file"], base_directory)
        if file_sha256(path) != binding["spec_file_sha256"]:
            raise ValueError("case spec checksum mismatch")
        spec = ProblemSpec.from_dict(json.loads(path.read_text()))
        query = _resolve(binding["observation_file"], base_directory)
        if file_sha256(query) != binding["observation_file_sha256"]:
            raise ValueError("case observation checksum mismatch")
        observation = np.load(query, allow_pickle=False)
        if not isinstance(observation, np.ndarray) or observation.dtype != np.float64:
            raise ValueError("float64 observation-only case required")
        return spec, observation
    def bank_provider(bank):
        return _resolve(value["banks"][bank["setup_id"] + "/" + str(bank["bank_seed"])]["directory"], base_directory)
    def reference_provider(case):
        binding = value["cases"][case_key(case)]
        if binding.get("reference_file") is None:
            return None
        path = _resolve(binding["reference_file"], base_directory)
        if file_sha256(path) != binding.get("reference_file_sha256"):
            raise ValueError("reference file checksum mismatch")
        return json.loads(path.read_text())
    def posterior_provider(r):
        case = value["cases"][case_key(r["case"])]
        model = value["models"][r["model"]["version_id"]]
        bank = value["banks"][r["bank"]["setup_id"] + "/" + str(r["bank"]["bank_seed"])]["directory"]
        request = PosteriorRequest(bank, r["bank"]["bank_manifest_sha256"], case["observation_file"],
            case["observation_file_sha256"], model, r["protocol_sha256"],
            context_size=r["context_size"], seed=r["posterior_seed"])
        run_path = outer / "posterior-runs" / r["request_id"]
        result = execute_posterior_request(request, output=run_path, base_directory=base_directory,
            allow_unqualified_cpu_diagnostic=True, resume=resume and run_path.exists(), backend_factory=backend_factory,
            evaluation_authorization={"plan": plan, "request_id": r["request_id"], "selection_lock": lock})
        return {"directory": run_path / "posterior", "manifest_sha256": result["artifact_manifest_sha256"]}
    assessment_name = "assessment-" + uuid.uuid4().hex
    result = consume_evaluation_plan(plan, case_provider=case_provider, bank_provider=bank_provider,
        posterior_provider=posterior_provider, reference_provider=reference_provider,
        output_directory=outer / assessment_name, selection_lock=lock,
        reference_preflight="require_qualified")
    summary = (summarize_test(plan, result["records"], selection_lock=lock) if plan["split"] == "test"
               else summarize_validation(plan, result["records"]))
    summary_name = assessment_name + "-summary.json"
    _write_json(outer / summary_name, summary)
    return {"status": "cpu_evaluation_records_complete", "plan_sha256": plan["plan_sha256"],
            "record_count": len(result["records"]), "assessment_directory": assessment_name,
            "assessment_manifest_sha256": result["manifest_sha256"],
            "summary_file": summary_name, "gpu_jobs_submitted": 0,
            "scientifically_qualified": False, "headline_savings_claim": False}


def plan_campaigns():
    from .campaign import plan_six_campaigns, plan_pilot_campaigns
    p = protocol_identity()
    return {"status": "planned_unqualified", **p,
            "main_campaigns": plan_six_campaigns(protocol_sha256=p["protocol_sha256"]),
            "pilot_campaigns": [r.to_dict() for r in plan_pilot_campaigns(protocol_sha256=p["protocol_sha256"])],
            "scientifically_qualified": False, "gpu_jobs_submitted": 0,
            "main_allowance_selection": "retained measured forecast required; no allowance selected"}


def plan_evaluation_request(value, *, base_directory="."):
    from .evaluation_campaign import build_validation_plan, build_test_plan, load_selection_lock
    common = {"schema", "split", "protocol_sha256", "cases", "banks"}
    optional = {"models", "master_seed", "scope", "context_sizes", "selection_lock_file",
                "selection_lock_sha256", "test_catalogue_sha256"}
    if not isinstance(value, dict) or not common <= set(value) or set(value) - common - optional:
        raise ValueError("unsupported evaluation plan request")
    if value["schema"] != "finetuneforge.evaluation_plan_request.v1":
        raise ValueError("unsupported evaluation plan schema")
    if value["split"] == "validation":
        if any(k in value for k in ("selection_lock_file", "selection_lock_sha256", "test_catalogue_sha256")):
            raise ValueError("validation planning cannot consume a test selection lock")
        return build_validation_plan(value["protocol_sha256"], value["models"], value["cases"], value["banks"],
            master_seed=value.get("master_seed", 20261117), scope=value.get("scope", "study"),
            context_sizes=tuple(value.get("context_sizes", CONTEXT_SIZES)))
    if value["split"] != "test" or any(k in value for k in ("models", "scope", "context_sizes")):
        raise ValueError("test plan inherits model versions, scope and context sizes from its frozen lock")
    lock = load_selection_lock(_resolve(value["selection_lock_file"], base_directory),
        expected_sha256=value["selection_lock_sha256"], protocol_sha256=value["protocol_sha256"],
        test_catalogue_sha256=value["test_catalogue_sha256"])
    return build_test_plan(value["protocol_sha256"], value["cases"], value["banks"], selection_lock=lock,
                           master_seed=value.get("master_seed"))


def inspect_model(base_checkpoint, *, campaign_checkpoint=None):
    """Establish a cached model's content identity; this is not qualification."""
    from .model_loading import load_inference_model
    from .campaign import read_campaign_checkpoint
    from .model import TabPFNBackbone, weights_fingerprint
    if campaign_checkpoint is None:
        network = TabPFNBackbone.from_checkpoint(base_checkpoint)
        expected = weights_fingerprint(network)
        del network
        manifest_sha = None
    else:
        expected = read_campaign_checkpoint(campaign_checkpoint)["deployment"]["weights_sha256"]
        manifest_sha = file_sha256(Path(campaign_checkpoint) / "manifest.json")
    _, metadata = load_inference_model(base_checkpoint, expected_model_sha256=expected,
        campaign_checkpoint=campaign_checkpoint, expected_checkpoint_manifest_sha256=manifest_sha)
    return metadata



def execute_evaluation_request(value, *, output, base_directory=".", allow_unqualified_cpu_diagnostic=False,
                               backend_factory=None, resume=False):
    _cpu_guard(allow_unqualified_cpu_diagnostic)
    request_sha = canonical_sha256(value)
    with _output_lock(output, request_sha, resume=resume) as root:
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        record = {"schema": "finetuneforge.evaluation_run.v1", "request_sha256": request_sha,
                  "attempt_id": attempt, "status": "running", "scientifically_qualified": False,
                  "gpu_jobs_submitted": 0, "orchestration_source_sha256": file_sha256(__file__)}
        _write_json(root / "run.json", record)
        try:
            report = _execute_evaluation_request_unlocked(value, output=root, base_directory=base_directory,
                allow_unqualified_cpu_diagnostic=True, backend_factory=backend_factory, resume=True)
            record.update(report)
        except BaseException as exc:
            record.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            (root / ("traceback-" + attempt + ".txt")).write_text(traceback.format_exc())
            raise
        finally:
            record["wall_seconds_current_attempt"] = time.monotonic() - started
            _write_json(root / "driver-attempts" / (attempt + ".json"), record)
            _write_json(root / "run.json", record)
        return {**record, "run_manifest_sha256": file_sha256(root / "run.json")}
