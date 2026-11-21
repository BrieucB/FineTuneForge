"""Versioned study design and non-executing campaign expansion."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    study_id: str = "hierarchical-ft-v1"
    master_seed: int = 20261117
    training_setups: int = 10000
    validation_setups: int = 20
    test_setups: int = 40
    cases_per_setup: int = 2
    bank_seeds: int = 2
    training_replicas: int = 2
    checkpoint_shortlist: int = 2
    posterior_draws: int = 50000
    training_contexts: tuple[int, ...] = (16, 64, 256, 1024)
    evaluation_contexts: tuple[int, ...] = (16, 64, 256, 1024, 4096)
    cap_gpu_hours: int = 10000
    qualification_gpu_hours: int = 500
    training_gpu_hours: int = 3000
    evaluation_gpu_hours: int = 4500
    engineering_gpu_hours: int = 1000
    reserve_gpu_hours: int = 1000
    execution_mode: str = "cpu-only"
    sampling_law: str = "canonical_half_normal_full_support.v1"
    source_grid_policy: str = "cancelled_zero_width"

    def __post_init__(self):
        for f in fields(self):
            if f.name in ("study_id", "execution_mode", "sampling_law", "source_grid_policy", "training_contexts", "evaluation_contexts"):
                continue
            x = getattr(self, f.name)
            if type(x) is not int or x < (0 if f.name == "master_seed" else 1):
                raise ValueError(f"{f.name} must be an explicit positive integer (seed may be zero)")
        for name in ("training_contexts", "evaluation_contexts"):
            value = tuple(getattr(self, name))
            if any(type(v) is not int for v in value):
                raise ValueError("context sizes must be explicit integers")
            object.__setattr__(self, name, value)
        fixed = {"training_setups": 10000, "validation_setups": 20, "test_setups": 40,
                 "cases_per_setup": 2, "bank_seeds": 2, "training_replicas": 2,
                 "posterior_draws": 50000, "cap_gpu_hours": 10000,
                 "training_contexts": (16, 64, 256, 1024),
                 "evaluation_contexts": (16, 64, 256, 1024, 4096)}
        if self.study_id != "hierarchical-ft-v1" or any(getattr(self, k) != v for k, v in fixed.items()):
            raise ValueError("changing the agreed scientific design requires a new protocol")
        if self.checkpoint_shortlist not in (1, 2):
            raise ValueError("at most two full-validation checkpoints per run")
        if sum((self.qualification_gpu_hours, self.training_gpu_hours, self.evaluation_gpu_hours,
                self.engineering_gpu_hours, self.reserve_gpu_hours)) != self.cap_gpu_hours:
            raise ValueError("phase envelopes must sum to the immutable study cap")
        if self.sampling_law != "canonical_half_normal_full_support.v1" or self.source_grid_policy != "cancelled_zero_width":
            raise ValueError("the full-tail common-backend policy is fixed; production qualification remains pending")
        if self.execution_mode != "cpu-only":
            raise ValueError("this software stage authorizes CPU checks only; GPU execution is disabled")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        names = {f.name for f in fields(cls)}
        if not isinstance(value, dict) or set(value) - names:
            raise ValueError("unknown study configuration field")
        return cls(**value)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def plan(self):
        models = 1 + 3 * self.training_replicas
        validation_versions = 1 + 3 * self.training_replicas * self.checkpoint_shortlist
        val_per = self.validation_setups * self.cases_per_setup * self.bank_seeds * len(self.evaluation_contexts)
        test_per = self.test_setups * self.cases_per_setup * self.bank_seeds * len(self.evaluation_contexts)
        val_requests, test_requests = validation_versions * val_per, models * test_per
        prior = (self.validation_setups + self.test_setups) * self.cases_per_setup
        return {"schema": "posteriorforge.study.plan.v1", "study_id": self.study_id,
                "protocol_sha256": self.fingerprint, "status": "planned_unqualified",
                "execution_mode": self.execution_mode, "gpu_jobs_submitted": 0,
                "validation_model_versions": validation_versions, "selected_test_model_versions": models,
                "maximum_validation_requests": val_requests, "test_requests": test_requests,
                "maximum_npe_pfn_requests": val_requests + test_requests,
                "prior_only_requests": prior, "reference_cases": prior,
                "posterior_draws_per_request": self.posterior_draws,
                "maximum_complete_vectors": (val_requests + test_requests + prior) * self.posterior_draws,
                "estimated_gpu_hours": None, "cap_gpu_hours": self.cap_gpu_hours,
                "configuration": self.to_dict(),
                "blocking_qualification": ["reference precision", "actual model/adapter numerical behavior",
                    "sampling-law qualification", "representative runtime and memory", "full-study cost forecast"],
                "note": "No timing, speedup, accuracy or GPU qualification is implied by a dry-run plan."}
