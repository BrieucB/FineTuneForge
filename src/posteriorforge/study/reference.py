"""Independent float64 references for variable-size synthetic hierarchies.

Sampling is not qualification. Production qualification is derived from retained
refinement and independently seeded runs, with explicit CDF/moment uncertainty.
The uncertainty contract is an empirical numerical diagnostic, not a theorem
about arbitrary posterior functionals. No function submits jobs or uses a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

SCHEMA = "posteriorforge.study.reference.v1"
CONTRACT = "retained-refinement-independent-v1"
PRECISION_LIMIT = .005
MOMENT_PRECISION_LIMIT = .02


class ReferenceUnavailable(RuntimeError):
    """An optional reference dependency or required qualification is unavailable."""


def _rows(spec, theta):
    x = np.asarray(theta, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != spec.dimension or not len(x):
        raise ValueError("nonempty complete parameter rows required")
    return x


def observation_digest(observation):
    x = np.ascontiguousarray(observation, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("one finite observation vector required")
    return sha256(x.tobytes()).hexdigest()


def _array_digest(array):
    x = np.ascontiguousarray(array, dtype=np.float64)
    return sha256(str(x.shape).encode() + x.tobytes()).hexdigest()


def source_digest():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _seed_words(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer reference seed required")
    # Use the complete entropy instead of silently truncating 64-bit seeds to
    # JAX's legacy 32-bit PRNGKey argument.
    return np.random.SeedSequence(seed).generate_state(2, dtype=np.uint32)


def _moment(value, known, xp=np):
    return known[value] if isinstance(value, str) else xp.asarray(value)


def _lognormal(mean, sd, xp=np):
    variance = xp.log1p((sd / mean) ** 2)
    return xp.log(mean) - variance / 2, xp.sqrt(variance)


def log_prior(spec, theta):
    """Independent physical-coordinate prior density, including log Jacobians."""
    x = _rows(spec, theta)
    result = np.full(len(x), -np.inf)
    good = spec.support_mask(x)
    if not good.any():
        return result
    y, values = x[good], {}
    lp = np.zeros(len(y), dtype=np.float64)
    for j, parameter in enumerate(spec.parameters):
        mean = np.broadcast_to(_moment(parameter.prior.mean, values), (len(y),))
        sd = np.broadcast_to(_moment(parameter.prior.sd, values), (len(y),))
        if np.any(sd <= 0) or not np.isfinite(sd).all():
            raise ValueError("invalid conditional prior SD")
        if parameter.prior.distribution == "lognormal":
            if np.any(mean <= 0):
                raise ValueError("invalid conditional log-normal mean")
            loc, scale = _lognormal(mean, sd)
            lp += norm.logpdf(np.log(y[:, j]), loc, scale) - np.log(y[:, j])
        else:
            lp += norm.logpdf(y[:, j], mean, sd)
        values[parameter.name] = y[:, j]
    result[good] = lp
    return result


def _decode(spec, z):
    x, values = np.empty_like(z, dtype=np.float64), {}
    for j, parameter in enumerate(spec.parameters):
        mean = _moment(parameter.prior.mean, values)
        sd = _moment(parameter.prior.sd, values)
        if parameter.prior.distribution == "lognormal":
            loc, scale = _lognormal(mean, sd)
            x[:, j] = np.exp(loc + scale * z[:, j])
        else:
            x[:, j] = mean + sd * z[:, j]
        values[parameter.name] = x[:, j]
    return x


def _response(spec, theta, xp=np):
    """Independent vectorized equations; no dependency on the bank simulator."""
    values = {p.name: theta[:, j] for j, p in enumerate(spec.parameters)}
    families = {f.id: f for f in spec.families}
    outputs = []
    for dataset in spec.datasets:
        physics = families[dataset.family_id].physics
        coeff = {k: values[v] for k, v in dataset.parameters.items()}
        coeff.update({k: xp.full((theta.shape[0],), v) for k, v in dataset.known.items()})
        def c(name, default=None):
            if name in coeff:
                return coeff[name][:, None]
            if default is None:
                raise ValueError(f"missing coefficient {name} for {physics}")
            return xp.full((theta.shape[0], 1), default)
        u = xp.asarray(dataset.controls)[None, :]
        if physics == "sensor":
            keys = sorted(coeff, key=lambda k: (int(''.join(v for v in k if v.isdigit()) or 0), k))
            h = xp.asarray(dataset.matrix)
            if len(keys) != h.shape[1]:
                raise ValueError("sensor coefficient count does not match matrix")
            y = xp.stack([coeff[k] for k in keys], axis=1) @ h.T
        elif physics == "quadratic":
            y = c("a") * u + c("b", 0.) * u ** 2 + c("d", 0.)
        elif physics == "scalar":
            y = c("a")
        elif physics == "saturation":
            h = float(dataset.known.get("h", 1.))
            y = c("d", 0.) + c("A") * u ** h / (c("K") ** h + u ** h)
        elif physics == "decay":
            y = c("d", 0.) + c("A_fast") * xp.exp(-u / c("tau_fast"))
            if "A_slow" in coeff or "delta_tau" in coeff:
                y = y + c("A_slow") * xp.exp(-u / (c("tau_fast") + c("delta_tau")))
        elif physics == "oscillation":
            y = c("d", 0.) + xp.exp(-c("gamma") * u) * (c("A") * xp.cos(c("omega") * u) + c("B", 0.) * xp.sin(c("omega") * u)) + c("v", 0.) * u
        elif physics == "spring":
            k, cubic = c("k"), c("c")
            y = 2 * xp.sqrt(k / (3 * cubic)) * xp.sinh(xp.arcsinh(3 * u / (2 * k) * xp.sqrt(3 * cubic / k)) / 3)
        elif physics == "contact":
            y = c("d", 0.) + c("k") * xp.maximum(u - c("u0"), 0.)
        else:
            raise ValueError(f"unknown physics {physics}")
        outputs.append(y)
    return xp.concatenate(outputs, axis=1)


def expected_observations(spec, theta):
    x = _rows(spec, theta)
    if not spec.support_mask(x).all():
        raise ValueError("finite prior-supported parameter rows required")
    result = np.asarray(_response(spec, x), dtype=np.float64)
    if result.shape != (len(x), spec.observation_dimension) or not np.isfinite(result).all():
        raise ValueError("invalid deterministic reference response")
    return result


def log_likelihood(spec, theta, observation):
    x = _rows(spec, theta)
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (spec.observation_dimension,) or not np.isfinite(observation).all():
        raise ValueError("one finite observation vector of the declared width required")
    result = np.full(len(x), -np.inf)
    good = spec.support_mask(x)
    if not good.any():
        return result
    predicted = expected_observations(spec, x[good])
    index = {p.name: j for j, p in enumerate(spec.parameters)}
    families = {f.id: f for f in spec.families}
    scales = []
    for dataset in spec.datasets:
        family = families[dataset.family_id]
        width = 1 if family.physics == "scalar" else 10
        scales.append(np.repeat(x[good, index[family.noise_parameter]][:, None], width, axis=1))
    sigma = np.concatenate(scales, axis=1)
    result[good] = norm.logpdf(observation[None, :], predicted, sigma).sum(axis=1)
    return result


def log_joint(spec, theta, observation):
    return log_prior(spec, theta) + log_likelihood(spec, theta, observation)


def predefined_directions(dimension, *, seed=20261117, scales=None):
    if type(dimension) is not int or dimension < 2:
        raise ValueError("at least two dimensions required for joint projections")
    units = np.ones(dimension) if scales is None else np.asarray(scales, dtype=np.float64)
    if units.shape != (dimension,) or not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("positive declared reference units required")
    rng = np.random.default_rng(seed)
    joint = rng.normal(size=(32, dimension))
    while np.any(joint == 0):
        invalid = np.any(joint == 0, axis=1)
        joint[invalid] = rng.normal(size=(int(invalid.sum()), dimension))
    joint /= np.sqrt(np.sum(joint * joint, axis=1))[:, None]
    joint *= np.where(joint[:, :1] < 0, -1., 1.)
    return np.vstack([np.eye(dimension), joint]) / units[None, :]


@dataclass(frozen=True)
class ReferenceRun:
    """Retained reference samples; this object carries no qualified/pass flag."""
    method: str
    samples: np.ndarray
    seed: int
    problem_fingerprint: str
    observation_sha256: str
    diagnostics: Mapping
    source_sha256: str

    def __post_init__(self):
        _seed_words(self.seed)
        if self.method not in ("numpyro_nuts", "tempered_smc"):
            raise ValueError("unsupported reference method identity")
        samples = np.array(self.samples, dtype=np.float64, copy=True)
        if samples.ndim != 2 or len(samples) < 2 or not np.isfinite(samples).all():
            raise ValueError("finite retained reference samples[n,D], n>=2 required")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        diagnostic = dict(self.diagnostics)
        invalid = {key: str(value) for key, value in diagnostic.items()
                   if isinstance(value, (float, np.floating)) and not np.isfinite(value)}
        if invalid:
            diagnostic.update(state="incomplete", reason="nonfinite_sampler_diagnostics", nonfinite_diagnostics=invalid)
            for key in invalid:
                diagnostic[key] = None
        object.__setattr__(self, "diagnostics", diagnostic)

    def manifest(self):
        return {"method": self.method, "seed": self.seed, "problem_fingerprint": self.problem_fingerprint,
                "observation_sha256": self.observation_sha256, "source_sha256": self.source_sha256,
                "sample_sha256": _array_digest(self.samples), "shape": list(self.samples.shape),
                "diagnostics": dict(self.diagnostics)}


def run_smc_reference(spec, observation, *, particles=1024, seed=0, moves=8,
                      max_temperatures=128, deadline_seconds=60.):
    """Bounded CPU tempered SMC in noncentred normal coordinates.

    Always returns sampled_unqualified diagnostics. Only retained independent
    refinement can qualify these samples. Logical reference forward evaluations
    count all actually evaluated local dataset responses, including repeats.
    """
    for value, name, minimum in ((particles, "particles", 16), (moves, "moves", 1), (max_temperatures, "max_temperatures", 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {name}")
    if not np.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("positive reference deadline required")
    query_sha = observation_digest(observation)
    started = time.monotonic()
    _seed_words(seed)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(particles, spec.dimension))
    theta = _decode(spec, z)
    ll = log_likelihood(spec, theta, observation)
    if not np.isfinite(ll).any():
        raise ReferenceUnavailable("all initial reference-particle likelihoods are nonfinite")
    if np.isnan(ll).any() or np.isposinf(ll).any():
        raise ReferenceUnavailable("invalid initial reference-particle likelihood")
    rows = particles
    ancestry = np.arange(particles)
    beta, evidence, step = 0., 0., .7 / np.sqrt(spec.dimension)
    history = []
    reason = None
    def weights(delta):
        logw = delta * ll
        logw -= logsumexp(logw)
        w = np.exp(logw)
        return w, 1. / np.dot(w, w)
    for iteration in range(max_temperatures):
        if time.monotonic() - started >= deadline_seconds:
            reason = "deadline_exhausted"
            break
        remaining = 1. - beta
        _, ess = weights(remaining)
        if ess >= .8 * particles:
            delta = remaining
        else:
            low, high = 0., remaining
            for _ in range(50):
                mid = (low + high) / 2
                if weights(mid)[1] >= .8 * particles:
                    low = mid
                else:
                    high = mid
            delta = low
        if delta <= np.finfo(float).eps:
            reason = "tempering_precision_exhausted"
            break
        w, ess = weights(delta)
        evidence += logsumexp(delta * ll) - np.log(particles)
        beta = min(1., beta + delta)
        # One random systematic offset; the final CDF edge is arithmetic closure,
        # not movement or clipping of physical parameter samples.
        edges = np.cumsum(w)
        edges[-1] = 1.
        indices = np.searchsorted(edges, (rng.random() + np.arange(particles)) / particles, side="right")
        z, ll, ancestry = z[indices].copy(), ll[indices].copy(), ancestry[indices]
        accepted = 0
        for _ in range(moves):
            if time.monotonic() - started >= deadline_seconds:
                reason = "deadline_exhausted"
                break
            proposal = z + step * rng.normal(size=z.shape)
            proposed_ll = log_likelihood(spec, _decode(spec, proposal), observation)
            rows += particles
            log_alpha = -.5 * (np.sum(proposal ** 2, 1) - np.sum(z ** 2, 1)) + beta * (proposed_ll - ll)
            take = np.log(rng.random(particles)) < log_alpha
            z[take], ll[take] = proposal[take], proposed_ll[take]
            accepted += int(take.sum())
        history.append({"beta": float(beta), "ess_before_resampling": float(ess), "acceptance": accepted / (particles * moves), "step": float(step)})
        step *= np.exp(np.clip(history[-1]["acceptance"] - .234, -.25, .25))
        if reason or beta == 1.:
            break
    if beta < 1. and reason is None:
        reason = "tempering_limit_exhausted"
    theta = _decode(spec, z)
    diagnostics = {"state": "sampled_unqualified" if beta == 1. and reason is None else "incomplete",
                   "reason": reason, "beta": float(beta), "history": history,
                   "effective_sample_size": int(len(np.unique(ancestry))),
                   "ess_contract": "conservative retained initial-ancestor count, not iid certification",
                   "likelihood_parameter_rows": rows, "local_forward_calls": rows * len(spec.datasets),
                   "platform": "cpu", "log_evidence_estimate": float(evidence),
                   "elapsed_seconds": time.monotonic() - started}
    return ReferenceRun("tempered_smc", theta, seed, spec.fingerprint, query_sha, diagnostics, source_digest())


def run_nuts_reference(spec, observation, *, seed=0, warmup=500, draws=2000, chains=4):
    """Optional CPU-only NumPyro implementation; never marks a run qualified."""
    for value, name, minimum in ((warmup, "warmup", 1), (draws, "draws", 2), (chains, "chains", 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {name}")
    query_sha = observation_digest(observation)
    if np.asarray(observation).shape != (spec.observation_dimension,):
        raise ValueError("observation width mismatch")
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from numpyro.diagnostics import summary
    except ImportError as exc:
        raise ReferenceUnavailable("NumPyro reference environment has not been prepared/qualified") from exc
    def model():
        values, columns = {}, []
        for j, parameter in enumerate(spec.parameters):
            z = numpyro.sample(f"latent_{j:04d}", dist.Normal(0., 1.))
            mean = _moment(parameter.prior.mean, values, jnp)
            sd = _moment(parameter.prior.sd, values, jnp)
            if parameter.prior.distribution == "lognormal":
                loc, scale = _lognormal(mean, sd, jnp)
                physical = jnp.exp(loc + scale * z)
            else:
                physical = mean + sd * z
            values[parameter.name] = numpyro.deterministic(f"physical_{j:04d}", physical)
            columns.append(physical)
        theta = jnp.stack(columns)[None, :]
        predicted = _response(spec, theta, jnp)[0]
        sigmas = []
        families = {f.id: f for f in spec.families}
        for dataset in spec.datasets:
            family = families[dataset.family_id]
            sigmas.extend([values[family.noise_parameter]] * (1 if family.physics == "scalar" else 10))
        numpyro.sample("observations", dist.Normal(predicted, jnp.stack(sigmas)).to_event(1), obs=jnp.asarray(observation))
    started = time.monotonic()
    seed_words = _seed_words(seed)
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(True):
        mcmc = MCMC(NUTS(model, target_accept_prob=.9, max_tree_depth=10), num_warmup=warmup,
                     num_samples=draws, num_chains=chains, chain_method="vectorized", progress_bar=False)
        mcmc.run(jnp.asarray(seed_words, dtype=jnp.uint32))
        sites = mcmc.get_samples(group_by_chain=True)
        diagnostic = summary(sites, group_by_chain=True)
        samples = np.stack([np.asarray(sites[f"physical_{j:04d}"]).reshape(-1) for j in range(spec.dimension)], axis=1)
        divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    d = {"state": "sampled_unqualified", "chains": chains, "draws_per_chain": draws,
         "effective_sample_size": float(min(len(samples), min(v["n_eff"] for v in diagnostic.values()))),
         "max_rhat": float(max(v["r_hat"] for v in diagnostic.values())), "divergences": divergences,
         "platform": "cpu", "jax_x64": True, "noncentred": True,
         "elapsed_seconds": time.monotonic() - started,
         "likelihood_parameter_rows": None, "local_forward_calls": None,
         "cost_status": "logical NUTS gradient/initialization calls are not yet instrumented; not zero; elapsed and scheduler allocation are separate",
         "physical_parameter_order": list(spec.parameter_order)}
    return ReferenceRun("numpyro_nuts", samples, seed, spec.fingerprint, query_sha, d, source_digest())


def _empirical_cdfs(samples, directions, queries):
    projected = samples @ directions.T
    return np.asarray([np.searchsorted(np.sort(projected[:, j]), queries[j], side="right") / len(samples)
                       for j in range(len(directions))])


def reference_receipt(reference):
    """Digest all numerical witnesses and qualification metadata for loaders."""
    h = sha256()
    for key in ("directions", "queries", "cdfs", "mean", "sd"):
        h.update(key.encode())
        h.update(_array_digest(reference[key]).encode())
    meta = {k: reference[k] for k in ("schema", "problem_fingerprint", "observation_sha256", "qualification")}
    h.update(json.dumps(meta, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def qualify_reference_witnesses(spec, observation, refinements: Sequence[ReferenceRun],
                                independent: ReferenceRun | None = None, *,
                                directions=None, seed=20261117, grid_points=513,
                                require_second_method=False):
    """Derive reference qualification from retained numerical runs, never a flag.

    Precision is the maximum retained grid discrepancy and an ESS-based Monte
    Carlo diagnostic. For correlated MCMC/SMC output that diagnostic is not an
    iid DKW guarantee. Difficult-case method/geometry audits remain mandatory.
    """
    if not refinements:
        raise ValueError("at least one retained reference run required")
    if type(grid_points) is not int or grid_points < 129:
        raise ValueError("at least 129 witness points required")
    runs = list(refinements) + ([] if independent is None else [independent])
    query_sha = observation_digest(observation)
    if np.asarray(observation).shape != (spec.observation_dimension,):
        raise ValueError("observation width mismatch")
    for run in runs:
        if not isinstance(run, ReferenceRun) or run.problem_fingerprint != spec.fingerprint or run.observation_sha256 != query_sha:
            raise ValueError("reference problem/query binding mismatch")
        if run.samples.shape[1] != spec.dimension or not spec.support_mask(run.samples).all():
            raise ValueError("reference samples violate complete-row/support contract")
        if run.source_sha256 != source_digest():
            raise ValueError("reference source identity mismatch")
    last = refinements[-1]
    units = spec.metadata.get("parameter_units", np.ones(spec.dimension))
    if isinstance(units, Mapping):
        if set(units) != set(spec.parameter_order):
            raise ValueError("declared projection units must cover all parameters")
        units = [units[name] for name in spec.parameter_order]
    expected_directions = predefined_directions(spec.dimension, seed=seed, scales=units)
    directions = expected_directions if directions is None else np.asarray(directions, dtype=np.float64)
    if not np.array_equal(directions, expected_directions):
        raise ValueError("directions must match the predefined seed and declared physical units")
    if directions.shape != (spec.dimension + 32, spec.dimension) or not np.isfinite(directions).all():
        raise ValueError("D physical axes and 32 joint directions required")
    projected = last.samples @ directions.T
    queries = []
    for j in range(len(directions)):
        ordered = np.sort(projected[:, j])
        span = ordered[-1] - ordered[0]
        if not span > 0:
            raise ValueError("reference projection has zero spread")
        grid = np.concatenate([[ordered[0] - .05 * span], np.quantile(ordered, np.linspace(.0001, .9999, grid_points - 2)), [ordered[-1] + .05 * span]])
        for k in range(1, len(grid)):
            if grid[k] <= grid[k - 1]:
                grid[k] = np.nextafter(grid[k - 1], np.inf)
        queries.append(grid)
    queries = np.asarray(queries)
    cdfs = _empirical_cdfs(last.samples, directions, queries)
    mean, sd = last.samples.mean(0), last.samples.std(0, ddof=1)
    if np.any(sd <= 0) or not np.isfinite(sd).all():
        raise ValueError("reference marginal SD must be positive")
    enough = len(refinements) >= 2 and len(last.samples) > len(refinements[-2].samples)
    independent_ok = independent is not None and independent.seed != last.seed and _array_digest(independent.samples) != _array_digest(last.samples)
    if require_second_method:
        independent_ok = independent_ok and independent.method != last.method
    def finite_number(value, default=0.):
        try:
            result = float(value)
            return result if np.isfinite(result) else default
        except (TypeError, ValueError):
            return default
    completed = all(r.diagnostics.get("state") == "sampled_unqualified" and r.diagnostics.get("divergences", 0) == 0
                    and finite_number(r.diagnostics.get("max_rhat", 1.), default=2.) <= 1.01 for r in runs)
    counts = [finite_number(r.diagnostics.get("effective_sample_size", 0.)) for r in runs]
    ess_ok = all(1 < v <= len(run.samples) for v, run in zip(counts, runs))
    precision = 1.
    mean_error, sd_error = np.full(spec.dimension, 1.), np.full(spec.dimension, 1.)
    if enough and independent_ok and ess_ok:
        comparisons = [refinements[-2], independent]
        cdf_changes = [float(np.max(np.abs(cdfs - _empirical_cdfs(r.samples, directions, queries)))) for r in comparisons]
        # The explicit finite-witness diagnostic is retained with its limitations;
        # no precision is inferred from sample count or Rhat alone.
        mc = float(np.sqrt(np.log(2 * len(directions) / .01) / (2 * min(counts))))
        precision = max([mc] + cdf_changes)
        mean_error = np.maximum.reduce([np.abs(r.samples.mean(0) - mean) / sd for r in comparisons] + [np.full(spec.dimension, 3 / np.sqrt(min(counts)))])
        sd_error = np.maximum.reduce([np.abs(r.samples.std(0, ddof=1) - sd) / sd for r in comparisons] + [np.full(spec.dimension, 3 / np.sqrt(2 * (min(counts) - 1)))])
    moment_passed = bool(np.all(mean_error <= MOMENT_PRECISION_LIMIT) and np.all(sd_error <= MOMENT_PRECISION_LIMIT))
    passed = bool(enough and independent_ok and completed and ess_ok and precision <= PRECISION_LIMIT and moment_passed)
    qualification = {"state": "qualified" if passed else "inconclusive", "contract": CONTRACT,
        "precision": float(precision), "moment_relative_precision_limit": MOMENT_PRECISION_LIMIT,
        "projection_seed": seed, "projection_units": np.asarray(units, dtype=np.float64).tolist(),
        "projection_contract": "fixed-seed dense Gaussian unit vectors, first-coordinate-positive sign, declared physical units; independent of model outcomes",
        "retained_refinement_passed": bool(enough),
        "independent_passed": bool(independent_ok), "completed_runs": bool(completed),
        "second_method_required": bool(require_second_method),
        "moment_gate_passed": moment_passed,
        "moment_precision": {"mean_error_reference_sd": mean_error.tolist(), "sd_error_reference_sd": sd_error.tolist()},
        "evidence": [r.manifest() for r in runs],
        "precision_scope": "retained finite-grid discrepancies plus ESS diagnostic; not a certified iid bound for correlated samples"}
    result = {"schema": SCHEMA, "problem_fingerprint": spec.fingerprint,
              "observation_sha256": query_sha, "directions": directions, "queries": queries,
              "cdfs": cdfs, "mean": mean, "sd": sd, "qualification": qualification}
    result["receipt_sha256"] = reference_receipt(result)
    return result


build_reference_witnesses = qualify_reference_witnesses
