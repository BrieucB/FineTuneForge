"""Immutable scientific contracts for paired hierarchical inference problems.

Means and standard deviations below are physical moments, also for log-normal
populations. A string reference names an earlier inferred scalar. No query truth
is part of a problem definition. Historical fixed-shape interfaces stay intact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

PHYSICS = ("sensor", "quadratic", "scalar", "saturation", "decay", "oscillation", "spring", "contact")
TRAIN_PHYSICS = PHYSICS[:6]
TEST_PHYSICS = PHYSICS[6:]
ROLES = ("population", "batch", "physical", "noise")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class PriorSpec:
    distribution: str
    mean: float | str
    sd: float | str

    def __post_init__(self):
        if self.distribution not in ("normal", "lognormal"):
            raise ValueError("prior distribution must be normal or lognormal")
        for name in ("mean", "sd"):
            value = getattr(self, name)
            if isinstance(value, str):
                if not value:
                    raise ValueError("prior references cannot be empty")
            else:
                value = _finite(value, name)
                if name == "sd" and value <= 0:
                    raise ValueError("prior SD must be positive")
                if name == "mean" and self.distribution == "lognormal" and value <= 0:
                    raise ValueError("log-normal physical mean must be positive")
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    role: str
    prior: PriorSpec

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.role not in ROLES:
            raise ValueError("invalid parameter name or role")
        if not isinstance(self.prior, PriorSpec):
            raise ValueError("prior must be a PriorSpec")


@dataclass(frozen=True)
class FamilySpec:
    id: str
    physics: str
    noise_parameter: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or self.physics not in PHYSICS or not isinstance(self.noise_parameter, str) or not self.noise_parameter:
            raise ValueError("invalid experiment family")


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    family_id: str
    specimen_id: str
    parameters: Mapping[str, str]
    controls: tuple[float, ...] = ()
    known: Mapping[str, float] = field(default_factory=dict)
    matrix: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.id, self.family_id, self.specimen_id)):
            raise ValueError("dataset, family and specimen identities are required")
        params = dict(self.parameters)
        if not params or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in params.items()):
            raise ValueError("physical parameter links must be explicit nonempty names")
        if set(params) & set(self.known):
            raise ValueError("a coefficient cannot be both inferred and known")
        object.__setattr__(self, "parameters", MappingProxyType(params))
        object.__setattr__(self, "known", MappingProxyType({str(k): _finite(v, str(k)) for k, v in self.known.items()}))
        object.__setattr__(self, "controls", tuple(_finite(v, "control") for v in self.controls))
        object.__setattr__(self, "matrix", tuple(tuple(_finite(v, "matrix") for v in row) for row in self.matrix))


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    split: str
    parameters: tuple[ParameterSpec, ...]
    families: tuple[FamilySpec, ...]
    datasets: tuple[DatasetSpec, ...]
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ("parameters", "families", "datasets"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        if not isinstance(self.id, str) or not self.id or self.split not in ("train", "validation", "test", "diagnostic"):
            raise ValueError("invalid problem identity or split")
        if not 1 <= len(self.parameters) <= 96 or not 1 <= len(self.families) <= 6:
            raise ValueError("problem exceeds unknown/family bounds")
        seen = {}
        for param in self.parameters:
            if param.name in seen:
                raise ValueError("duplicate parameter name")
            for key in ("mean", "sd"):
                ref = getattr(param.prior, key)
                if isinstance(ref, str):
                    if ref not in seen:
                        raise ValueError("prior parents must precede descendants")
                    if (key == "sd" or param.prior.distribution == "lognormal") and seen[ref].prior.distribution != "lognormal":
                        raise ValueError("positive prior moments need positive parent support")
            seen[param.name] = param
        if self.split in ("train", "validation") and any(f.physics in TEST_PHYSICS for f in self.families):
            raise ValueError("held-out physics cannot enter training or validation")
        family = {f.id: f for f in self.families}
        if len(family) != len(self.families):
            raise ValueError("duplicate family identity")
        if len({f.noise_parameter for f in self.families}) != len(self.families):
            raise ValueError("different families need distinct inferred noise levels")
        for f in self.families:
            noise = seen.get(f.noise_parameter)
            if noise is None or noise.role != "noise" or noise.prior.distribution != "lognormal":
                raise ValueError("each family needs an inferred positive noise SD")
        if len({d.id for d in self.datasets}) != len(self.datasets):
            raise ValueError("duplicate dataset identity")
        counts = Counter(d.family_id for d in self.datasets)
        if set(counts) != set(family) or any(not 1 <= n <= 12 for n in counts.values()):
            raise ValueError("each active family needs 1..12 datasets")
        used = {f.noise_parameter for f in self.families}
        repeated_links = {}
        coefficients = {"quadratic": {"a", "b", "d"}, "scalar": {"a"}, "saturation": {"d", "A", "K", "h"},
                        "decay": {"d", "A_fast", "tau_fast", "A_slow", "delta_tau"},
                        "oscillation": {"d", "A", "B", "gamma", "omega", "v"}, "spring": {"k", "c"}, "contact": {"d", "k", "u0"}}
        positive = {"saturation": {"A", "K"}, "decay": {"A_fast", "tau_fast", "A_slow", "delta_tau"},
                    "oscillation": {"gamma", "omega"}, "spring": {"k", "c"}, "contact": {"k", "u0"}}
        for dataset in self.datasets:
            f = family[dataset.family_id]
            allowed = {f"p{i}" for i in range(len(dataset.matrix[0]))} if f.physics == "sensor" and dataset.matrix else coefficients.get(f.physics, set())
            if (set(dataset.parameters) | set(dataset.known)) - allowed:
                raise ValueError("unknown physical coefficient name")
            declared = set(dataset.parameters) | set(dataset.known)
            required = {"quadratic": {"a", "b", "d"}, "scalar": {"a"}, "saturation": {"d", "A", "K", "h"},
                        "decay": {"d", "A_fast", "tau_fast"}, "oscillation": {"d", "A", "B", "gamma", "omega", "v"},
                        "spring": {"k", "c"}, "contact": {"d", "k", "u0"}}.get(f.physics, allowed)
            if required - declared:
                raise ValueError("all active coefficients must be explicitly known or inferred")
            if f.physics == "decay" and bool("A_slow" in declared) != bool("delta_tau" in declared):
                raise ValueError("a second decay needs both amplitude and time gap")
            if f.physics == "saturation" and "h" in dataset.parameters:
                raise ValueError("the saturation exponent is known, not inferred")
            if f.physics == "oscillation" and ({"gamma", "omega"} & set(dataset.parameters)) and not ({"A", "B"} & set(dataset.parameters)) and dataset.known.get("A", 0) == dataset.known.get("B", 0) == 0:
                raise ValueError("unknown frequency/damping needs a nonzero oscillation")
            for coefficient in positive.get(f.physics, set()):
                if coefficient in dataset.parameters and seen.get(dataset.parameters[coefficient]) is not None and seen[dataset.parameters[coefficient]].prior.distribution != "lognormal":
                    raise ValueError("positive physical coefficients require positive prior support")
                if coefficient in dataset.known and dataset.known[coefficient] <= 0:
                    raise ValueError("known positive physical coefficients must be positive")
            if f.physics == "saturation" and dataset.known.get("h", 1) not in (1, 2):
                raise ValueError("saturation exponent must be known 1 or 2")
            identity = (dataset.family_id, dataset.specimen_id)
            links = dict(dataset.parameters)
            if identity in repeated_links and repeated_links[identity] != links:
                raise ValueError("repeated measurements must preserve specimen parameter links")
            repeated_links[identity] = links
            if not 1 <= len(set(dataset.parameters.values())) <= 6:
                raise ValueError("physical parameter dimension must be 1..6")
            for name in dataset.parameters.values():
                if name not in seen or seen[name].role != "physical":
                    raise ValueError("dataset physical links must name inferred physical parameters")
                used.add(name)
            if f.physics == "scalar":
                if len(dataset.controls) not in (0, 1):
                    raise ValueError("scalar experiments have one reading")
            elif f.physics == "sensor":
                if len(dataset.matrix) != 10 or len({len(r) for r in dataset.matrix}) != 1 or not 1 <= len(dataset.matrix[0]) <= 6:
                    raise ValueError("sensor matrix must have ten channels and 1..6 properties")
            elif len(dataset.controls) != 10:
                raise ValueError("curve experiments require ten control points")
        for p in reversed(self.parameters):
            if p.name in used:
                used.update(v for v in (p.prior.mean, p.prior.sd) if isinstance(v, str))
        if used != set(seen):
            raise ValueError("unused inferred quantities are not allowed")
        if self.dimension + self.observation_dimension > 450:
            raise ValueError("measurements plus unknowns must be at most 450")
        json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @property
    def dimension(self):
        return len(self.parameters)

    @property
    def observation_dimension(self):
        fs = {f.id: f for f in self.families}
        return sum(1 if fs[d.family_id].physics == "scalar" else 10 for d in self.datasets)

    @property
    def parameter_order(self):
        return tuple(p.name for p in self.parameters)

    @property
    def roles(self):
        return tuple(p.role for p in self.parameters)

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def support_mask(self, theta):
        value = np.asarray(theta, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.dimension:
            raise ValueError("complete parameter rows required")
        good = np.isfinite(value).all(axis=1)
        for j, p in enumerate(self.parameters):
            if p.prior.distribution == "lognormal":
                good &= value[:, j] > 0
        return good

    def to_dict(self):
        return _plain(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["parameters"] = tuple(ParameterSpec(p["name"], p["role"], PriorSpec(**p["prior"])) for p in data["parameters"])
        data["families"] = tuple(FamilySpec(**f) for f in data["families"])
        data["datasets"] = tuple(DatasetSpec(**d) for d in data["datasets"])
        return cls(**data)
