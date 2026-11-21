"""Dimension-aware, fail-closed posterior gates for the new study.

The public boundary requires exactly 50,000 vectors. Reference receipts bind
CDF/moment witnesses and qualification evidence to the problem and observation.
No row is clipped, replaced or dropped; unresolved numerical precision cannot
produce a model pass.
"""
from __future__ import annotations

import numpy as np

from ..validation import cdf_distance_bounds, sample_integrity
from .reference import CONTRACT, SCHEMA, observation_digest, predefined_directions, reference_receipt, source_digest

DRAW_COUNT = 50_000
CDF_LIMIT = .10
BIAS_LIMIT = .20
SD_LIMITS = (.8, 1.2)
REFERENCE_LIMIT = .005
MOMENT_LIMIT = .02


def reference_moment_errors_valid(mean_error, sd_error, dimension):
    """Numerical moment stability is separate from the .005 CDF contract."""
    me, se = np.asarray(mean_error, dtype=np.float64), np.asarray(sd_error, dtype=np.float64)
    return bool(me.shape == (dimension,) and se.shape == (dimension,)
                and np.isfinite(me).all() and np.isfinite(se).all()
                and np.all(me >= 0) and np.all(se >= 0)
                and np.all(me <= MOMENT_LIMIT) and np.all(se <= MOMENT_LIMIT))


def _reference_arrays(reference, spec, observation):
    if not isinstance(reference, dict):
        raise ValueError("reference mapping required")
    required = ("directions", "queries", "cdfs", "mean", "sd", "qualification")
    if any(k not in reference for k in required):
        raise ValueError("incomplete numerical reference witnesses")
    d = spec.dimension
    directions, queries, cdfs, mean, sd = (np.asarray(reference[k], dtype=np.float64)
        for k in ("directions", "queries", "cdfs", "mean", "sd"))
    if directions.shape != (d + 32, d) or not np.isfinite(directions).all():
        raise ValueError("D axes and 32 complete-vector projections required")
    first = directions[:d]
    if not np.array_equal(first, np.diag(np.diag(first))) or np.any(np.diag(first) <= 0) or np.any(directions[d:] == 0):
        raise ValueError("invalid physical axes or joint projections")
    if queries.ndim != 2 or queries.shape[0] != d + 32 or queries.shape[1] < 129 or not np.isfinite(queries).all() or np.any(np.diff(queries, axis=1) <= 0):
        raise ValueError("strictly increasing finite per-projection witness grids required")
    if cdfs.shape != queries.shape or not np.isfinite(cdfs).all() or np.any(cdfs < 0) or np.any(cdfs > 1) or np.any(np.diff(cdfs, axis=1) < 0):
        raise ValueError("monotone reference CDF probabilities required")
    if mean.shape != (d,) or sd.shape != (d,) or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("finite reference mean and positive SD required")
    q = reference["qualification"]
    if not isinstance(q, dict):
        raise ValueError("reference qualification mapping required")
    query_matches = observation is not None and reference.get("observation_sha256") == observation_digest(observation)
    binding = reference.get("schema") == SCHEMA and reference.get("problem_fingerprint") == spec.fingerprint and query_matches
    if reference.get("receipt_sha256") != reference_receipt(reference):
        raise ValueError("reference numerical receipt mismatch")
    try:
        precision = float(q["precision"])
        me = np.asarray(q["moment_precision"]["mean_error_reference_sd"], dtype=np.float64)
        se = np.asarray(q["moment_precision"]["sd_error_reference_sd"], dtype=np.float64)
    except (KeyError, ValueError, TypeError):
        precision, me, se = 1., np.full(d, 1.), np.full(d, 1.)
    moment_ok = reference_moment_errors_valid(me, se, d)
    evidence = q.get("evidence", [])
    valid_evidence = isinstance(evidence, list) and len(evidence) >= 3
    if valid_evidence:
        for e in evidence:
            valid_evidence = valid_evidence and isinstance(e, dict) and (
                e.get("method") in ("numpyro_nuts", "tempered_smc")
                and e.get("source_sha256") == source_digest()
                and e.get("problem_fingerprint") == spec.fingerprint
                and e.get("observation_sha256") == reference.get("observation_sha256")
                and len(e.get("sample_sha256", "")) == 64
                and len(e.get("shape", [])) == 2 and e["shape"][1] == d
                and isinstance(e.get("diagnostics"), dict)
                and e["diagnostics"].get("state") == "sampled_unqualified")
    if valid_evidence:
        try:
            units = spec.metadata.get("parameter_units", np.ones(d))
            if isinstance(units, dict) or hasattr(units, "keys"):
                units = [units[name] for name in spec.parameter_order]
            expected_directions = predefined_directions(d, seed=q["projection_seed"], scales=units)
            valid_evidence = np.array_equal(directions, expected_directions) and np.array_equal(q["projection_units"], units)
        except (KeyError, TypeError, ValueError):
            valid_evidence = False
    if valid_evidence:
        # Flags cannot replace actual refinement/independence relationships.
        primary, previous, independent = evidence[-2], evidence[-3], evidence[-1]
        valid_evidence = (previous["shape"][0] < primary["shape"][0]
            and primary["seed"] != independent["seed"]
            and primary["sample_sha256"] != independent["sample_sha256"]
            and (not q.get("second_method_required") or primary["method"] != independent["method"]))
        effective = []
        for e in evidence:
            try:
                ess = float(e["diagnostics"]["effective_sample_size"])
                rhat = float(e["diagnostics"].get("max_rhat", 1.))
                good = np.isfinite(ess) and 1 < ess <= e["shape"][0] and np.isfinite(rhat) and rhat <= 1.01 and e["diagnostics"].get("divergences", 0) == 0
            except (KeyError, TypeError, ValueError):
                ess, good = 0., False
            valid_evidence = valid_evidence and good
            effective.append(ess)
        if valid_evidence:
            least = min(effective)
            floor = np.sqrt(np.log(2 * len(directions) / .01) / (2 * least))
            valid_evidence = (precision >= floor and np.all(me >= 3 / np.sqrt(least))
                              and np.all(se >= 3 / np.sqrt(2 * (least - 1))))
    qualified = bool(binding and moment_ok and valid_evidence and q.get("state") == "qualified"
        and q.get("contract") == CONTRACT and np.isfinite(precision) and 0 <= precision <= REFERENCE_LIMIT
        and all(q.get(k) is True for k in ("retained_refinement_passed", "independent_passed", "completed_runs", "moment_gate_passed")))
    return directions, queries, cdfs, mean, sd, precision, me, se, qualified


def moment_intervals(sample_mean, sample_sd, reference_mean, reference_sd, mean_error, sd_error):
    """Conservative normalized mean/SD intervals with numerical reference error."""
    m, s, rm, rs, me, se = (np.asarray(v, dtype=np.float64) for v in
                           (sample_mean, sample_sd, reference_mean, reference_sd, mean_error, sd_error))
    if any(v.shape != rm.shape for v in (m, s, rs, me, se)) or not all(np.isfinite(v).all() for v in (m, s, rm, rs, me, se)):
        raise ValueError("finite matching moment vectors required")
    if np.any(rs <= 0) or np.any(s < 0) or np.any(me < 0) or np.any(se < 0) or np.any(se >= 1):
        raise ValueError("invalid moment uncertainties or standard deviations")
    bias, ratio = np.abs(m - rm) / rs, s / rs
    bl, bu = np.maximum(0., bias - me) / (1 + se), (bias + me) / (1 - se)
    rl, ru = ratio / (1 + se), ratio / (1 - se)
    return {"absolute_bias_reference_sd": bias.tolist(), "sd_ratio": ratio.tolist(),
            "bias_lower": bl.tolist(), "bias_upper": bu.tolist(), "ratio_lower": rl.tolist(), "ratio_upper": ru.tolist(),
            "passed": bool(np.all(bu <= BIAS_LIMIT) and np.all(rl >= SD_LIMITS[0]) and np.all(ru <= SD_LIMITS[1])),
            "conclusively_failed": bool(np.any(bl > BIAS_LIMIT) or np.any(ru < SD_LIMITS[0]) or np.any(rl > SD_LIMITS[1]))}


def assess_samples(samples, reference, *, spec, observation=None):
    """Assess one complete 50,000-vector posterior; no sampling is performed."""
    values = np.asarray(samples)
    shape_ok = values.ndim == 2 and values.shape == (DRAW_COUNT, spec.dimension)
    support = spec.support_mask(values) if shape_ok else None
    integrity = sample_integrity(values, expected_count=DRAW_COUNT, dimension=spec.dimension, support_mask=support)
    result = {"schema": "posteriorforge.study.posterior_gate.v1", "status": "invalid_samples", "passed": False,
              "problem_fingerprint": spec.fingerprint, "integrity": integrity,
              "reference_qualified": False, "expected_count": DRAW_COUNT,
              "limits": {"cdf": CDF_LIMIT, "absolute_bias_reference_sd": BIAS_LIMIT, "sd_ratio": list(SD_LIMITS), "reference_precision": REFERENCE_LIMIT, "reference_moment_relative_precision": MOMENT_LIMIT},
              "scope": "finite_synthetic_case_diagnostic_not_global_calibration"}
    if not integrity["passed"]:
        return result
    result["status"] = "reference_inconclusive"
    if reference is None:
        result["reason"] = "missing_reference"
        return result
    directions, queries, cdfs, rm, rs, precision, me, se, qualified = _reference_arrays(reference, spec, observation)
    result["reference_qualified"] = qualified
    result["reference_qualification"] = reference["qualification"]
    values = np.asarray(values, dtype=np.float64)
    sm, ss = values.mean(0), values.std(0, ddof=1)
    if me.shape == (spec.dimension,) and se.shape == (spec.dimension,) and np.all(se < 1):
        moments = moment_intervals(sm, ss, rm, rs, me, se)
    else:
        moments = {"passed": False, "conclusively_failed": False, "reason": "unqualified_moment_uncertainty"}
    result["moments"] = dict(moments, sample_mean=sm.tolist(), sample_sd=ss.tolist(), reference_mean=rm.tolist(), reference_sd=rs.tolist())
    projected = values @ directions.T
    bounds = [cdf_distance_bounds(projected[:, j:j + 1], queries[j], cdfs[j:j + 1],
                                  precision=precision if np.isfinite(precision) and 0 <= precision <= 1 else 0.)
              for j in range(len(directions))]
    for label, ids in (("marginal_cdf", range(spec.dimension)), ("joint_projection_cdf", range(spec.dimension, spec.dimension + 32))):
        lower = [bounds[j]["lower"][0] for j in ids]
        upper = [bounds[j]["upper"][0] for j in ids]
        result[label] = {"lower": lower, "upper": upper, "max_lower": max(lower), "max_upper": max(upper), "passed": bool(max(upper) <= CDF_LIMIT)}
    if not qualified:
        return result
    if moments["conclusively_failed"] or result["marginal_cdf"]["max_lower"] > CDF_LIMIT or result["joint_projection_cdf"]["max_lower"] > CDF_LIMIT:
        result["status"] = "model_failure"
    elif not moments["passed"]:
        result["status"] = "moment_inconclusive"
    elif result["marginal_cdf"]["passed"] and result["joint_projection_cdf"]["passed"]:
        result.update(status="pass", passed=True)
    else:
        result["status"] = "cdf_inconclusive"
    return result
