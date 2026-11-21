"""Analytical hierarchical providers and causal prior transport.

No model weights or numerical reference posteriors are used here. Log-normal
parameters are physical arithmetic means and standard deviations throughout.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from typing import Any

import numpy as np

from .spec import ProblemSpec

PHYSICS_ALIASES = {
    "sensor_mixtures": "sensor", "scalar_readout": "scalar",
    "damped_oscillation": "oscillation", "nonlinear_spring": "spring",
    "contact_threshold": "contact",
}


def _count(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("count must be a positive integer")
    return value


def _rows(spec: ProblemSpec, values: Any) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != spec.dimension or not len(rows):
        raise ValueError("complete nonempty rows with the problem parameter width required")
    return rows


def _resolved(value: float | str, theta: np.ndarray, positions: dict[str, int]):
    if isinstance(value, str):
        return theta[:, positions[value]]
    return np.full(len(theta), float(value), dtype=np.float64)


def _moments(prior, theta: np.ndarray, positions: dict[str, int]):
    mean = _resolved(prior.mean, theta, positions)
    sd = _resolved(prior.sd, theta, positions)
    return mean, sd


def _lognormal(mean: np.ndarray, sd: np.ndarray):
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(sd)) or np.any(mean <= 0) or np.any(sd <= 0):
        raise ValueError("log-normal physical means and SDs must be finite and positive")
    variance_log = np.logaddexp(0.0, 2.0 * (np.log(sd) - np.log(mean)))
    return np.log(mean) - 0.5 * variance_log, np.sqrt(variance_log)


def decode_prior(spec: ProblemSpec, latent: Any) -> np.ndarray:
    """Map independent standard-normal rows into complete physical prior worlds."""
    z = _rows(spec, latent)
    if not np.isfinite(z).all():
        raise ValueError("latent rows must be finite")
    theta = np.empty_like(z)
    positions = {name: j for j, name in enumerate(spec.parameter_order)}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for j, parameter in enumerate(spec.parameters):
            mean, sd = _moments(parameter.prior, theta, positions)
            if np.any(~np.isfinite(sd)) or np.any(sd <= 0) or np.any(~np.isfinite(mean)):
                raise ValueError("conditional prior moments must be finite with positive SD")
            if parameter.prior.distribution == "normal":
                theta[:, j] = mean + sd * z[:, j]
            elif parameter.prior.distribution == "lognormal":
                loc, scale = _lognormal(mean, sd)
                theta[:, j] = np.exp(loc + scale * z[:, j])
            else:
                raise ValueError("unsupported prior distribution")
    if not np.isfinite(theta).all() or not spec.support_mask(theta).all():
        raise ValueError("prior decoding produced nonfinite or unsupported values")
    return theta


def encode_prior(spec: ProblemSpec, theta: Any) -> np.ndarray:
    """Invert the causal prior transform without clipping or discarding rows."""
    rows = _rows(spec, theta)
    if not spec.support_mask(rows).all():
        raise ValueError("prior encoding requires finite supported physical rows")
    positions = {name: j for j, name in enumerate(spec.parameter_order)}
    z = np.empty_like(rows)
    for j, parameter in enumerate(spec.parameters):
        mean, sd = _moments(parameter.prior, rows, positions)
        if parameter.prior.distribution == "normal":
            z[:, j] = (rows[:, j] - mean) / sd
        else:
            loc, scale = _lognormal(mean, sd)
            z[:, j] = (np.log(rows[:, j]) - loc) / scale
    if not np.isfinite(z).all():
        raise ValueError("nonfinite encoded coordinates")
    return z


def log_abs_det_decode(spec: ProblemSpec, latent: Any, *, per_parameter: bool = False) -> np.ndarray:
    """Causal decoder log Jacobian, summed by default or one term per scalar.

    Each column is the log absolute diagonal derivative d(theta_j)/d(z_j)
    holding preceding coordinates fixed. The causal map is triangular, so the
    column sum is its full log determinant. Per-scalar terms permit the same
    physical-density correction when balancing parameter groups in training.
    """
    if type(per_parameter) is not bool:
        raise ValueError("per_parameter must be an explicit boolean")
    theta = decode_prior(spec, latent)
    positions = {name: j for j, name in enumerate(spec.parameter_order)}
    result = np.zeros(len(theta), dtype=np.float64)
    terms = np.empty_like(theta) if per_parameter else None
    for j, parameter in enumerate(spec.parameters):
        mean, sd = _moments(parameter.prior, theta, positions)
        if parameter.prior.distribution == "normal":
            contribution = np.log(sd)
        else:
            _, scale = _lognormal(mean, sd)
            contribution = np.log(scale) + np.log(theta[:, j])
        result += contribution
        if terms is not None:
            terms[:, j] = contribution
    return terms if per_parameter else result


def draw_prior(spec: ProblemSpec, count: int, seed: int = 0) -> np.ndarray:
    """Draw parameters only: this function never evaluates a forward response."""
    count = _count(count)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return decode_prior(spec, np.random.default_rng(seed).normal(size=(count, spec.dimension)))


def log_prior(spec: ProblemSpec, theta: Any) -> np.ndarray:
    rows = _rows(spec, theta)
    supported = spec.support_mask(rows)
    result = np.full(len(rows), -np.inf, dtype=np.float64)
    if supported.any():
        selected = rows[supported]
        z = encode_prior(spec, selected)
        result[supported] = -0.5 * np.sum(z * z + math.log(2.0 * math.pi), axis=1) - log_abs_det_decode(spec, z)
    return result


def _response(physics: str, coefficients: dict[str, float], controls, matrix) -> np.ndarray:
    """One complete deterministic local measurement vector at one physical setting."""
    physics = PHYSICS_ALIASES.get(physics, physics)
    c = coefficients
    u = np.asarray(controls, dtype=np.float64)
    if physics == "sensor":
        h = np.asarray(matrix, dtype=np.float64)
        if h.ndim != 2 or h.shape[0] != 10 or h.shape[1] < 1:
            raise ValueError("sensor matrix must have ten rows and at least one property")
        values = [c[f"p{k}"] if f"p{k}" in c else c[f"theta_{k+1}"] for k in range(h.shape[1])]
        answer = h @ np.asarray(values)
    elif physics == "scalar":
        answer = np.asarray([c["a"]])
    elif physics == "quadratic":
        answer = c["a"] * u + c["b"] * u * u + c["d"]
    elif physics == "saturation":
        if c["A"] <= 0 or c["K"] <= 0 or c.get("h", 1) not in (1, 2) or np.any(u < 0):
            raise ValueError("saturation requires positive A/K, nonnegative controls and known h=1/2")
        h = c.get("h", 1)
        answer = c["d"] + c["A"] * u**h / (c["K"]**h + u**h)
    elif physics == "decay":
        if c["A_fast"] <= 0 or c["tau_fast"] <= 0 or np.any(u < 0):
            raise ValueError("decay amplitudes/times must be positive")
        answer = c["d"] + c["A_fast"] * np.exp(-u / c["tau_fast"])
        if "A_slow" in c:
            if c["A_slow"] <= 0 or c["delta_tau"] <= 0:
                raise ValueError("second decay amplitude/time gap must be positive")
            answer += c["A_slow"] * np.exp(-u / (c["tau_fast"] + c["delta_tau"]))
    elif physics == "oscillation":
        if c["gamma"] <= 0 or c["omega"] <= 0 or np.any(u < 0):
            raise ValueError("oscillation damping/frequency must be positive")
        answer = c["d"] + np.exp(-c["gamma"] * u) * (c["A"] * np.cos(c["omega"] * u) + c["B"] * np.sin(c["omega"] * u)) + c["v"] * u
    elif physics == "spring":
        if c["k"] <= 0 or c["c"] <= 0:
            raise ValueError("spring stiffness coefficients must be positive")
        answer = 2.0 * np.sqrt(c["k"] / (3.0 * c["c"])) * np.sinh(np.arcsinh(1.5 * u / c["k"] * np.sqrt(3.0 * c["c"] / c["k"])) / 3.0)
    elif physics == "contact":
        if c["k"] <= 0 or c["u0"] <= 0:
            raise ValueError("contact slope/threshold must be positive")
        answer = c["d"] + c["k"] * np.maximum(u - c["u0"], 0.0)
    else:
        raise ValueError(f"unknown physics {physics!r}")
    answer = np.asarray(answer, dtype=np.float64)
    expected = 1 if physics == "scalar" else 10
    if answer.shape != (expected,) or not np.isfinite(answer).all():
        raise ValueError("local response must be a complete finite measurement vector")
    return answer


def simulate(spec: ProblemSpec, theta: Any, seed: int = 0, *, noisy: bool = True) -> tuple[np.ndarray, dict]:
    """Evaluate full worlds; reuse equivalent noiseless local calls only.

    Repeated measurements receive independent errors even when their physical
    response was reused. The cache contains no learned-model state or query data.
    """
    started = time.perf_counter()
    rows = _rows(spec, theta)
    if not spec.support_mask(rows).all():
        raise ValueError("simulation requires finite prior-supported complete rows")
    if type(seed) is not int or seed < 0 or type(noisy) is not bool:
        raise ValueError("invalid simulation seed/noise flag")
    positions = {name: j for j, name in enumerate(spec.parameter_order)}
    families = {f.id: f for f in spec.families}
    obs = np.empty((len(rows), spec.observation_dimension), dtype=np.float64)
    cache: dict[tuple, np.ndarray] = {}
    calls = Counter()
    first_rows: dict[tuple, int] = {}
    offset = 0
    for dataset in spec.datasets:
        family = families[dataset.family_id]
        physics = PHYSICS_ALIASES.get(family.physics, family.physics)
        width = 1 if physics == "scalar" else 10
        controls = np.asarray(dataset.controls, dtype=np.float64)
        matrix = np.asarray(dataset.matrix, dtype=np.float64)
        for i, row in enumerate(rows):
            c = {key: float(value) for key, value in dataset.known.items()}
            c.update({key: float(row[positions[name]]) for key, name in dataset.parameters.items()})
            key = (family.id, physics, tuple(sorted((k, v.hex()) for k, v in c.items())), controls.tobytes(), matrix.shape, matrix.tobytes())
            first_rows[key] = min(i, first_rows.get(key, i))
            if key not in cache:
                cache[key] = _response(physics, c, controls, matrix)
                calls[family.id] += 1
            obs[i, offset:offset + width] = cache[key]
        if noisy:
            sigma = rows[:, positions[family.noise_parameter]]
            if np.any(sigma <= 0):
                raise ValueError("measurement noise must have positive SD")
            role = np.frombuffer(hashlib.sha256(dataset.id.encode()).digest()[:8], dtype="<u4").tolist()
            rng = np.random.default_rng(np.random.SeedSequence([seed, *role]))
            obs[:, offset:offset + width] += rng.normal(size=(len(rows), width)) * sigma[:, None]
        offset += width
    if not np.isfinite(obs).all():
        raise ValueError("simulation generated nonfinite observations")
    requested = len(rows) * len(spec.datasets)
    first_counts = np.bincount(list(first_rows.values()), minlength=len(rows))
    prefix_cost = np.concatenate(([0], np.cumsum(first_counts))).tolist()
    per_family_prefix = {f.id: np.concatenate(([0], np.cumsum(np.bincount([row for key, row in first_rows.items() if key[0] == f.id], minlength=len(rows))))).tolist() for f in spec.families}
    cost = {"schema": "finetuneforge.local-forward-cost.v1", "worlds": len(rows),
            "requested_local_vectors": requested, "forward_evaluations": sum(calls.values()),
            "reused_local_vectors": requested - sum(calls.values()),
            "noise_values_generated": int(obs.size) if noisy else 0,
            "per_family_forward_evaluations": dict(sorted(calls.items())),
            "elapsed_seconds": time.perf_counter() - started,
            "call_unit": "one complete noiseless local curve or measurement vector",
            "noise_included": noisy, "problem_fingerprint": spec.fingerprint,
            "prefix_forward_evaluations": prefix_cost,
            "prefix_forward_evaluations_by_family": per_family_prefix}
    return obs, cost
