"""Complete variable-size paired banks with authenticated nested prefixes."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .spec import ProblemSpec
from .science import draw_prior, simulate

CONTEXT_SIZES = (16, 64, 256, 1024, 4096)
SCHEMA = "finetuneforge.study.bank.v1"


def array_descriptor(value: np.ndarray) -> dict:
    a = np.ascontiguousarray(value)
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": sha256(a.tobytes()).hexdigest()}


def _json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class PairedBank:
    spec: ProblemSpec
    theta: np.ndarray
    observations: np.ndarray
    manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        theta = np.array(self.theta, dtype=np.float64, order="C", copy=True)
        obs = np.array(self.observations, dtype=np.float64, order="C", copy=True)
        if theta.ndim != 2 or theta.shape[1] != self.spec.dimension or len(theta) < 1:
            raise ValueError("complete nonempty physical rows required")
        if obs.shape != (len(theta), self.spec.observation_dimension):
            raise ValueError("observation rows must stay paired with complete parameter rows")
        if not self.spec.support_mask(theta).all() or not np.isfinite(obs).all():
            raise ValueError("banks must be finite and within declared prior support")
        theta.setflags(write=False)
        obs.setflags(write=False)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "observations", obs)
        # JSON round-trip detaches the metadata from mutable caller state.
        manifest = json.loads(json.dumps(self.manifest, sort_keys=True, allow_nan=False))
        if "problem_fingerprint" in manifest and manifest["problem_fingerprint"] != self.spec.fingerprint:
            raise ValueError("bank/problem fingerprint mismatch")
        object.__setattr__(self, "manifest", manifest)

    @property
    def obs(self):
        return self.observations

    def prefix(self, count: int) -> "PairedBank":
        if type(count) is not int or not 1 <= count <= len(self.theta):
            raise ValueError("prefix count must be in the stored row range")
        metadata = dict(self.manifest)
        metadata.update(worlds=count, parent_worlds=len(self.theta), parent_theta=array_descriptor(self.theta))
        cost = dict(metadata.get("cost", {}))
        if "prefix_forward_evaluations" in cost:
            cumulative = cost["prefix_forward_evaluations"]
            if len(cumulative) != len(self.theta) + 1:
                raise ValueError("invalid retained prefix cost trace")
            cost["actual_generation_forward_evaluations"] = cost.get("actual_generation_forward_evaluations", cost["forward_evaluations"])
            cost["actual_generation_worlds"] = cost.get("actual_generation_worlds", cost.get("worlds", len(self.theta)))
            cost["context_forward_evaluations"] = cumulative[count]
            cost["forward_evaluations"] = cumulative[count]
            cost["worlds"] = count
            cost["requested_local_vectors"] = count * len(self.spec.datasets)
            cost["reused_local_vectors"] = cost["requested_local_vectors"] - cumulative[count]
            cost["actual_generation_noise_values"] = cost.get("actual_generation_noise_values", cost.get("noise_values_generated", 0))
            cost["noise_values_generated"] = count * self.spec.observation_dimension if cost.get("noise_included") else 0
            if "prefix_forward_evaluations_by_family" in cost:
                per_family = cost["prefix_forward_evaluations_by_family"]
                cost["actual_generation_per_family_forward_evaluations"] = cost.get("actual_generation_per_family_forward_evaluations", cost["per_family_forward_evaluations"])
                cost["per_family_forward_evaluations"] = {key: values[count] for key, values in per_family.items()}
                cost["prefix_forward_evaluations_by_family"] = {key: values[:count + 1] for key, values in per_family.items()}
            cost["prefix_forward_evaluations"] = cumulative[:count + 1]
        metadata["cost"] = cost
        return PairedBank(self.spec, self.theta[:count], self.observations[:count], metadata)


def prepare_bank(spec: ProblemSpec, count: int, seed: int = 0) -> PairedBank:
    """Generate one bank; callers create all smaller contexts with prefix()."""
    if type(seed) is not int or seed < 0:
        raise ValueError("bank seed must be a nonnegative integer")
    streams = np.random.SeedSequence([seed, 20261117]).spawn(2)
    prior_seed, noise_seed = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams]
    theta = draw_prior(spec, count, prior_seed)
    obs, cost = simulate(spec, theta, seed=noise_seed)
    return PairedBank(spec, theta, obs, {"schema": SCHEMA, "problem_fingerprint": spec.fingerprint,
        "seed": seed, "prior_seed": prior_seed, "noise_seed": noise_seed, "worlds": count,
        "parameter_order": list(spec.parameter_order), "dataset_order": [d.id for d in spec.datasets],
        "sampling": "full specified hierarchical prior; no truth-dependent filtering", "cost": cost})


def save_bank(bank: PairedBank, output) -> dict:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "spec.json", bank.spec.to_dict())
    with (root / "bank.npz").open("xb") as stream:
        np.savez(stream, theta=bank.theta, observations=bank.observations)
    result = dict(bank.manifest, schema=SCHEMA, problem_fingerprint=bank.spec.fingerprint,
        theta=array_descriptor(bank.theta), observations=array_descriptor(bank.observations),
        parameter_order=list(bank.spec.parameter_order), dataset_order=[d.id for d in bank.spec.datasets],
        worlds=len(bank.theta), files={name: sha256((root / name).read_bytes()).hexdigest() for name in ("spec.json", "bank.npz")})
    _json(root / "manifest.json", result)
    return dict(result, manifest_sha256=sha256((root / "manifest.json").read_bytes()).hexdigest())


def load_bank(output, *, expected_sha256: str | None = None) -> PairedBank:
    root = Path(output)
    manifest_bytes = (root / "manifest.json").read_bytes()
    if expected_sha256 is not None and sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ValueError("bank manifest checksum mismatch")
    m = json.loads(manifest_bytes)
    if m.get("schema") != SCHEMA or set(m.get("files", {})) != {"spec.json", "bank.npz"}:
        raise ValueError("unsupported or incomplete bank manifest")
    for name, digest in m["files"].items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("bank payload checksum mismatch")
    spec = ProblemSpec.from_dict(json.loads((root / "spec.json").read_text()))
    with np.load(root / "bank.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"theta", "observations"}:
            raise ValueError("unexpected bank arrays")
        theta, obs = saved["theta"], saved["observations"]
    if array_descriptor(theta) != m["theta"] or array_descriptor(obs) != m["observations"]:
        raise ValueError("bank array descriptor mismatch")
    if m["parameter_order"] != list(spec.parameter_order) or m["dataset_order"] != [d.id for d in spec.datasets]:
        raise ValueError("bank ordering mismatch")
    return PairedBank(spec, theta, obs, m)
