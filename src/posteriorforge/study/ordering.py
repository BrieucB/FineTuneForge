"""Reversible listing augmentation without changing a hierarchical problem.

Only the serialized order changes. Physical identities, prior references,
sharing links and within-dataset controls remain intact. This is a CPU data
primitive, not a training campaign or evidence of relabelling robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .banks import PairedBank
from .spec import ProblemSpec

SCHEMA = "finetuneforge.study.ordering.v1"


def _permutation(value, name: str) -> tuple[int, ...]:
    result = tuple(value)
    if not result or any(type(i) is not int for i in result) or sorted(result) != list(range(len(result))):
        raise ValueError(f"{name} must be a complete integer permutation")
    return result


def _inverse(value: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(value)
    for target, source in enumerate(value):
        result[source] = target
    return tuple(result)


def _columns(value: Any, indices: tuple[int, ...]) -> np.ndarray:
    rows = np.asarray(value)
    if rows.ndim < 1 or rows.shape[-1] != len(indices):
        raise ValueError("array's final dimension must match the complete column mapping")
    return np.take(rows, indices, axis=-1)


@dataclass(frozen=True)
class OrderingMap:
    """Every forward index maps a target position to its source position.

    Thus target_theta = source_theta[:, parameter_columns]. The inverse maps
    explicitly restore the source listing; neither direction changes row order.
    Fingerprints bind the maps to the complete source and target definitions.
    """

    source_fingerprint: str
    target_fingerprint: str
    parameter_columns: tuple[int, ...]
    observation_columns: tuple[int, ...]
    family_indices: tuple[int, ...]
    dataset_indices: tuple[int, ...]
    seed: int | None = None

    def __post_init__(self):
        for name in ("source_fingerprint", "target_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("ordering fingerprints must be lowercase SHA256 digests")
        for name in ("parameter_columns", "observation_columns", "family_indices", "dataset_indices"):
            object.__setattr__(self, name, _permutation(getattr(self, name), name))
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("ordering seed must be a nonnegative integer or None")

    @property
    def inverse_parameter_columns(self):
        return _inverse(self.parameter_columns)

    @property
    def inverse_observation_columns(self):
        return _inverse(self.observation_columns)

    def inverse(self) -> "OrderingMap":
        return OrderingMap(self.target_fingerprint, self.source_fingerprint,
            self.inverse_parameter_columns, self.inverse_observation_columns,
            _inverse(self.family_indices), _inverse(self.dataset_indices), self.seed)

    def parameters_to_target(self, values: Any) -> np.ndarray:
        return _columns(values, self.parameter_columns)

    def parameters_to_source(self, values: Any) -> np.ndarray:
        return _columns(values, self.inverse_parameter_columns)

    def observations_to_target(self, values: Any) -> np.ndarray:
        return _columns(values, self.observation_columns)

    def observations_to_source(self, values: Any) -> np.ndarray:
        return _columns(values, self.inverse_observation_columns)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint, "seed": self.seed,
            "parameter_columns": list(self.parameter_columns),
            "observation_columns": list(self.observation_columns),
            "family_indices": list(self.family_indices), "dataset_indices": list(self.dataset_indices),
            "inverse_parameter_columns": list(self.inverse_parameter_columns),
            "inverse_observation_columns": list(self.inverse_observation_columns)}

    @classmethod
    def from_dict(cls, value: dict) -> "OrderingMap":
        data = dict(value)
        if data.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported ordering schema")
        inverse_parameters = data.pop("inverse_parameter_columns", None)
        inverse_observations = data.pop("inverse_observation_columns", None)
        result = cls(**data)
        if inverse_parameters != list(result.inverse_parameter_columns) or inverse_observations != list(result.inverse_observation_columns):
            raise ValueError("inconsistent retained inverse ordering")
        return result


def _observation_columns(spec: ProblemSpec, dataset_indices: tuple[int, ...]) -> tuple[int, ...]:
    physics = {family.id: family.physics for family in spec.families}
    blocks, offset = [], 0
    for dataset in spec.datasets:
        width = 1 if physics[dataset.family_id] == "scalar" else 10
        blocks.append(tuple(range(offset, offset + width)))
        offset += width
    return tuple(column for i in dataset_indices for column in blocks[i])


def _rebuild(spec: ProblemSpec, parameters, families, datasets) -> ProblemSpec:
    return replace(spec, parameters=tuple(spec.parameters[i] for i in parameters),
        families=tuple(spec.families[i] for i in families), datasets=tuple(spec.datasets[i] for i in datasets))


def _parameter_order(spec: ProblemSpec, rng: np.random.Generator) -> tuple[int, ...]:
    positions = {name: j for j, name in enumerate(spec.parameter_order)}
    predecessors = [{positions[ref] for ref in (p.prior.mean, p.prior.sd) if isinstance(ref, str)} for p in spec.parameters]
    population_batch = {i for i, p in enumerate(spec.parameters) if p.role in ("population", "batch")}
    for i, p in enumerate(spec.parameters):
        if p.role == "physical":
            predecessors[i].update(population_batch)
    remaining, selected = set(range(spec.dimension)), []
    while remaining:
        eligible = sorted(i for i in remaining if not predecessors[i] & remaining)
        if not eligible:
            raise ValueError("prior links cannot place every population/batch quantity before physical properties")
        chosen = eligible[int(rng.integers(len(eligible)))]
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def reorder_problem(spec: ProblemSpec, seed: int = 0) -> tuple[ProblemSpec, OrderingMap]:
    """Randomly reorder eligible parameters, families and whole datasets.

    Parameter parents always precede children. All population/batch quantities
    precede every physical property; remaining simultaneously eligible nodes
    may exchange positions. Noise quantities retain their identity and links.
    No labels, controls, truth values or simulation banks are generated.
    """
    if not isinstance(spec, ProblemSpec):
        raise ValueError("a ProblemSpec is required")
    if type(seed) is not int or seed < 0:
        raise ValueError("ordering seed must be a nonnegative integer")
    streams = np.random.SeedSequence([seed, 20261117, 5198404]).spawn(3)
    parameters = _parameter_order(spec, np.random.default_rng(streams[0]))
    families = tuple(int(i) for i in np.random.default_rng(streams[1]).permutation(len(spec.families)))
    datasets = tuple(int(i) for i in np.random.default_rng(streams[2]).permutation(len(spec.datasets)))
    target = _rebuild(spec, parameters, families, datasets)
    ordering = OrderingMap(spec.fingerprint, target.fingerprint, parameters,
        _observation_columns(spec, datasets), families, datasets, seed)
    return target, ordering


def apply_ordering(spec: ProblemSpec, ordering: OrderingMap) -> ProblemSpec:
    """Replay a checked map; its inverse restores the original exact listing."""
    if not isinstance(spec, ProblemSpec) or not isinstance(ordering, OrderingMap):
        raise ValueError("a ProblemSpec and OrderingMap are required")
    if spec.fingerprint != ordering.source_fingerprint:
        raise ValueError("ordering source fingerprint mismatch")
    sizes = (spec.dimension, spec.observation_dimension, len(spec.families), len(spec.datasets))
    if sizes != tuple(len(getattr(ordering, field)) for field in ("parameter_columns", "observation_columns", "family_indices", "dataset_indices")):
        raise ValueError("ordering dimensions do not match the source problem")
    if ordering.observation_columns != _observation_columns(spec, ordering.dataset_indices):
        raise ValueError("observation columns must preserve every complete dataset's control order")
    target = _rebuild(spec, ordering.parameter_columns, ordering.family_indices, ordering.dataset_indices)
    if target.fingerprint != ordering.target_fingerprint:
        raise ValueError("ordering target fingerprint mismatch")
    return target


def reorder_bank(bank: PairedBank, ordering: OrderingMap) -> PairedBank:
    """Permute stored arrays only: no forward calls or fresh noise generation.

    Source generation and nested-prefix costs remain verbatim, in the source
    coordinate system identified by their retained fingerprint. Stale payload
    checksums move into provenance; save_bank authenticates the new payload.
    """
    if not isinstance(bank, PairedBank):
        raise ValueError("a PairedBank is required")
    target = apply_ordering(bank.spec, ordering)
    metadata = dict(bank.manifest)
    provenance = ordering.to_dict()
    provenance.update(source_parameter_order=list(bank.spec.parameter_order),
        target_parameter_order=list(target.parameter_order),
        source_family_order=[f.id for f in bank.spec.families], target_family_order=[f.id for f in target.families],
        source_dataset_order=[d.id for d in bank.spec.datasets], target_dataset_order=[d.id for d in target.datasets],
        added_forward_evaluations=0, added_noise_values=0, operation="column permutation only")
    artifacts = {key: metadata.pop(key) for key in ("theta", "observations", "files", "manifest_sha256", "parent_theta") if key in metadata}
    if artifacts:
        provenance["source_artifacts"] = artifacts
    history = list(metadata.get("ordering_history", []))
    history.append(provenance)
    metadata.update(problem_fingerprint=target.fingerprint,
        parameter_order=list(target.parameter_order), dataset_order=[d.id for d in target.datasets],
        ordering_history=history)
    return PairedBank(target, ordering.parameters_to_target(bank.theta),
        ordering.observations_to_target(bank.observations), metadata)
