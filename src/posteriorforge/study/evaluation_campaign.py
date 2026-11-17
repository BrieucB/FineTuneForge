"""Deterministic artifact evaluation and validation-to-test selection sealing.

This driver reads artifacts; it never trains, simulates, samples, launches a job,
filters posterior rows, or changes a checkpoint. Providers locate already prepared
artifacts. A diagnostic lock freezes inputs without qualifying an inference model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import load_samples, SCHEMA as SAMPLE_SCHEMA
from .banks import load_bank, array_descriptor
from .evaluation import (BANK_SIZES, paired_call_savings, passing_budgets, select_checkpoint,
                         select_default_bank, select_primary_rank)
from .metrics import (assess_samples, _reference_arrays, moment_intervals,
                      reference_moment_errors_valid, CDF_LIMIT, BIAS_LIMIT, SD_LIMITS,
                      REFERENCE_LIMIT, MOMENT_LIMIT)
from .reference import (observation_digest, reference_receipt, _array_digest,
                        source_digest, predefined_directions, CONTRACT, SCHEMA as REFERENCE_SCHEMA)
from .spec import ProblemSpec

PLAN_SCHEMA = "finetuneforge.evaluation_plan.v1"
RECORD_SCHEMA = "finetuneforge.evaluation_record.v1"
LOCK_SCHEMA = "finetuneforge.selection_lock.v1"
DRAW_COUNT = 50_000
STUDY_SAMPLING_LAW = "canonical_half_normal_full_support.v1"
STUDY_SOURCE_POLICY = "cancelled_zero_width"


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return sha256(_bytes(value)).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"explicit {label} SHA256 required")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} required")
    return value


def _json_copy(value):
    return json.loads(_bytes(value))


def _without(value, *names):
    return {key: item for key, item in value.items() if key not in names}


def _models(values, *, scope):
    result, seen, checkpoints = [], set(), set()
    groups = defaultdict(list)
    for item in values:
        row = dict(item)
        required = {"version_id", "checkpoint_id", "model_sha256", "checkpoint_manifest_sha256",
                    "method", "rank", "training_seed", "update"}
        if set(row) != required:
            raise ValueError("model version needs exact checkpoint, method and seed bindings")
        for name in ("version_id", "checkpoint_id"):
            _text(row[name], name)
        for name in ("model_sha256", "checkpoint_manifest_sha256"):
            _sha(row[name], name)
        if row["version_id"] in seen or row["checkpoint_id"] in checkpoints:
            raise ValueError("duplicate model/checkpoint identity")
        seen.add(row["version_id"]); checkpoints.add(row["checkpoint_id"])
        if type(row["update"]) is not int or row["update"] < 0:
            raise ValueError("explicit nonnegative checkpoint update required")
        method, rank, seed = row["method"], row["rank"], row["training_seed"]
        if method == "frozen":
            if rank is not None or seed is not None or row["update"] != 0:
                raise ValueError("frozen baseline has no training rank/seed/update")
        elif method in ("lora", "full"):
            if type(seed) is not int or seed < 0:
                raise ValueError("trained checkpoint needs a nonnegative training seed")
            if (method == "lora" and (type(rank) is not int or rank not in (4, 16))) or (method == "full" and rank is not None):
                raise ValueError("LoRA rank must be 4/16; full fine-tuning has no LoRA rank")
        else:
            raise ValueError("method must be frozen, lora or full")
        groups[(method, rank, seed)].append(row)
        result.append(row)
    frozen = [v for v in result if v["method"] == "frozen"]
    if len(frozen) != 1 or any(len(group) > (1 if key[0] == "frozen" else 2) for key, group in groups.items()):
        raise ValueError("one baseline and at most two shortlisted versions per training run required")
    if scope == "study":
        seeds = [{k[2] for k in groups if k[:2] == pair} for pair in (("lora", 4), ("lora", 16), ("full", None))]
        if not all(len(s) == 2 for s in seeds) or not seeds[0] == seeds[1] == seeds[2] or len(groups) != 7:
            raise ValueError("study needs both matched training replicas of LoRA4/16 and full tuning")
    return sorted(result, key=lambda v: v["version_id"])


def _cases(values, *, split, scope):
    result, seen = [], set()
    for item in values:
        row = dict(item)
        required = {"setup_id", "case_id", "split", "problem_fingerprint", "observation_sha256", "reference_receipt_sha256"}
        if not required <= set(row) or set(row) - required - {"test_group", "cluster_id"}:
            raise ValueError("case needs problem/query/reference bindings and explicit identities")
        for name in ("setup_id", "case_id"):
            _text(row[name], name)
        if row["split"] != split:
            raise ValueError(f"{split}-only case catalogue required")
        for name in ("problem_fingerprint", "observation_sha256"):
            _sha(row[name], name)
        if row["reference_receipt_sha256"] is not None:
            _sha(row["reference_receipt_sha256"], "reference receipt")
        key = (row["setup_id"], row["case_id"])
        if key in seen:
            raise ValueError("duplicate case identity")
        seen.add(key); result.append(row)
    if not result:
        raise ValueError("explicit nonempty case catalogue required")
    by_setup = defaultdict(list)
    for case in result:
        by_setup[case["setup_id"]].append(case)
    if scope == "study" and (len(by_setup) != (20 if split == "validation" else 40) or any(len(v) != 2 for v in by_setup.values())):
        raise ValueError("study requires 20 validation/40 test setups with two cases each")
    for members in by_setup.values():
        if len({v["problem_fingerprint"] for v in members}) != 1:
            raise ValueError("cases within one setup must share the same problem")
        for name in ("test_group", "cluster_id"):
            if len({v.get(name) for v in members}) != 1:
                raise ValueError("test group and bootstrap cluster must be constant within a setup")
            if name in members[0]:
                _text(members[0][name], name)
    if scope == "study" and split == "test":
        counts = defaultdict(int)
        for members in by_setup.values():
            counts[members[0].get("test_group")] += 1
        if dict(counts) != {name: 8 for name in ("familiar", "upper_sizes", "sharing", "heldout_physics", "combined")}:
            raise ValueError("study test catalogue requires eight setups in each of the five groups")
        pairs = defaultdict(set)
        for setup_id, members in by_setup.items():
            if members[0]["test_group"] == "sharing":
                if not members[0].get("cluster_id"):
                    raise ValueError("sharing tests require four explicit matched-pair cluster IDs")
                pairs[members[0]["cluster_id"]].add(setup_id)
        if len(pairs) != 4 or any(len(setups) != 2 for setups in pairs.values()):
            raise ValueError("sharing tests require four matched-pair clusters with exactly two setups each")
        other_clusters = [members[0].get("cluster_id", setup_id) for setup_id, members in by_setup.items() if members[0]["test_group"] != "sharing"]
        if any(cluster in pairs for cluster in other_clusters) or len(set(other_clusters)) != len(other_clusters):
            raise ValueError("independent test setups must not share clusters outside the four declared sharing pairs")
    return sorted(result, key=lambda v: (v["setup_id"], v["case_id"]))


def case_catalogue_sha256(cases):
    """Hash sealed metadata only; no test observations are opened."""
    return _digest(sorted([dict(v) for v in cases], key=lambda v: (v["setup_id"], v["case_id"])))


def _banks(values, cases):
    result, seen = [], set()
    setups = {v["setup_id"] for v in cases}
    for item in values:
        row = dict(item)
        if set(row) != {"setup_id", "bank_seed", "bank_manifest_sha256"}:
            raise ValueError("bank needs setup/seed and manifest SHA256")
        if row["setup_id"] not in setups or type(row["bank_seed"]) is not int or row["bank_seed"] not in (0, 1):
            raise ValueError("bank setup must match catalogue, with seed identities 0/1")
        _sha(row["bank_manifest_sha256"], "bank manifest")
        key = (row["setup_id"], row["bank_seed"])
        if key in seen:
            raise ValueError("duplicate bank identity")
        seen.add(key); result.append(row)
    if seen != {(setup, seed) for setup in setups for seed in (0, 1)}:
        raise ValueError("both matched bank seeds are required for every setup")
    for setup in setups:
        if len({row["bank_manifest_sha256"] for row in result if row["setup_id"] == setup}) != 2:
            raise ValueError("independent bank replicas require different manifest SHA256 receipts")
    return sorted(result, key=lambda v: (v["setup_id"], v["bank_seed"]))


def _requests(protocol_sha256, models, cases, banks, contexts, master_seed):
    by_setup = defaultdict(list)
    for bank in banks:
        by_setup[bank["setup_id"]].append(bank)
    result = []
    for model in models:
        for case in cases:
            for bank in by_setup[case["setup_id"]]:
                for size in contexts:
                    body = {"protocol_sha256": protocol_sha256, "split": case["split"], "model": model,
                            "case": case, "bank": bank, "context_size": size, "posterior_draws": DRAW_COUNT}
                    # Full namespaced identity, not Python's randomized hash.
                    body["posterior_seed"] = int(_digest({"role": "posterior-draws", "master_seed": master_seed, "request": body})[:16], 16)
                    result.append(dict(body, request_id=_digest(body)))
    return result


def _make_plan(protocol_sha256, models, cases, banks, *, split, master_seed, scope, contexts, selection_lock_sha256=None):
    _sha(protocol_sha256, "protocol")
    if scope not in ("study", "diagnostic") or type(master_seed) is not int or master_seed < 0:
        raise ValueError("explicit scope and nonnegative master seed required")
    contexts = tuple(contexts)
    if not contexts or tuple(sorted(set(contexts))) != contexts or any(type(v) is not int or v not in BANK_SIZES for v in contexts):
        raise ValueError("ordered directly tested context sizes required")
    if scope == "study" and contexts != BANK_SIZES:
        raise ValueError("study requires all five agreed bank sizes")
    models = _models(models, scope=scope)
    cases = _cases(cases, split=split, scope=scope)
    banks = _banks(banks, cases)
    payload = {"schema": PLAN_SCHEMA, "protocol_sha256": protocol_sha256, "split": split,
               "scope": scope, "master_seed": master_seed, "posterior_draws": DRAW_COUNT,
               "context_sizes": list(contexts), "models": models, "cases": cases, "banks": banks,
               "case_catalogue_sha256": case_catalogue_sha256(cases),
               "selection_lock_sha256": selection_lock_sha256,
               "requests": _requests(protocol_sha256, models, cases, banks, contexts, master_seed)}
    return dict(payload, plan_sha256=_digest(payload))


def build_validation_plan(protocol_sha256, models, cases, banks, *, master_seed=20261117, scope="study", context_sizes=BANK_SIZES):
    """Expand baseline + up to 12 shortlisted versions without opening data."""
    return _make_plan(protocol_sha256, models, cases, banks, split="validation", master_seed=master_seed,
                      scope=scope, contexts=context_sizes)


def _verify_plan(plan):
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("authenticated evaluation plan required")
    if plan.get("plan_sha256") != _digest(_without(plan, "plan_sha256")):
        raise ValueError("evaluation plan checksum mismatch")
    rebuilt = _make_plan(plan["protocol_sha256"], plan["models"], plan["cases"], plan["banks"],
        split=plan["split"], master_seed=plan["master_seed"], scope=plan["scope"], contexts=plan["context_sizes"],
        selection_lock_sha256=plan["selection_lock_sha256"])
    if rebuilt != dict(plan):
        raise ValueError("evaluation matrix does not match declared plan")
    if plan["split"] not in ("validation", "test"):
        raise ValueError("only validation/test plans can be evaluated")
    return rebuilt


def verify_evaluation_request(plan, request_id, *, selection_lock=None):
    """Return one detached request only after phase and immutable bindings pass."""
    plan = _verify_plan(plan)
    if plan["split"] == "test":
        _check_test_lock(plan, selection_lock)
    _sha(request_id, "evaluation request")
    for request in plan["requests"]:
        if request["request_id"] == request_id:
            return _json_copy(request)
    raise ValueError("request is not in the authenticated evaluation matrix")


class _ReferenceSkipped(Exception):
    pass


def _bank_costs(bank, request):
    spec, manifest, size = bank.spec, bank.manifest, request["context_size"]
    if spec.fingerprint != request["case"]["problem_fingerprint"]:
        raise ValueError("bank/problem binding mismatch")
    worlds, cost = manifest.get("worlds"), manifest.get("cost", {})
    trace = cost.get("prefix_forward_evaluations")
    if (type(worlds) is not int or worlds < size or not isinstance(trace, list) or len(trace) != worlds + 1
            or any(type(v) is not int or v < 0 for v in trace) or trace[0] != 0
            or any(a > b for a, b in zip(trace, trace[1:])) or cost.get("forward_evaluations") != trace[-1]):
        raise ValueError("authenticated prefix forward-call trace required")
    full = cost.get("actual_generation_forward_evaluations", trace[-1])
    if type(full) is not int or full < trace[-1]:
        raise ValueError("full bank-generation cost must cover retained prefix")
    return {"forward_calls": trace[size], "bank_generation_forward_calls": full,
            "posterior_sampling_forward_calls": 0, "bank_cost_unit": "one complete noiseless local curve or measurement vector"}


def _bank_replica_certificate(plan, setup_id, banks):
    replicas = []
    for binding in (row for row in plan["banks"] if row["setup_id"] == setup_id):
        bank = banks[(setup_id, binding["bank_seed"])]
        manifest = bank.manifest
        seeds = {name: manifest.get(name) for name in ("seed", "prior_seed", "noise_seed")}
        if any(type(value) is not int or value < 0 for value in seeds.values()):
            raise ValueError("bank must retain its actual generation/prior/noise seeds")
        replicas.append(dict(binding, generation_seeds=seeds,
                             theta=array_descriptor(bank.theta), observations=array_descriptor(bank.observations)))
    certificate = {"setup_id": setup_id, "replicas": sorted(replicas, key=lambda row: row["bank_seed"])}
    _verify_bank_replica_certificate(certificate, setup_id, plan["banks"])
    return certificate


def _verify_bank_replica_certificate(certificate, setup_id, bindings):
    if not isinstance(certificate, dict) or certificate.get("setup_id") != setup_id:
        raise ValueError("independent bank replica evidence required")
    replicas = certificate.get("replicas", [])
    expected = {row["bank_seed"]: row["bank_manifest_sha256"] for row in bindings if row["setup_id"] == setup_id}
    if len(replicas) != 2 or {row.get("bank_seed") for row in replicas} != {0, 1}:
        raise ValueError("two complete bank replica receipts required")
    for row in replicas:
        if row.get("bank_manifest_sha256") != expected.get(row["bank_seed"]):
            raise ValueError("bank replica certificate/plan mismatch")
        seeds = row.get("generation_seeds", {})
        if set(seeds) != {"seed", "prior_seed", "noise_seed"} or any(type(v) is not int or v < 0 for v in seeds.values()):
            raise ValueError("actual bank generation seeds required; index 0/1 is not a generation seed")
        for name in ("theta", "observations"):
            descriptor = row.get(name, {})
            _sha(descriptor.get("sha256"), "bank array")
            if not isinstance(descriptor.get("shape"), list) or len(descriptor["shape"]) != 2:
                raise ValueError("complete bank array descriptor required")
    for name in ("seed", "prior_seed", "noise_seed"):
        if replicas[0]["generation_seeds"][name] == replicas[1]["generation_seeds"][name]:
            raise ValueError("independent bank replicas must have different actual generation seeds")
    if any(replicas[0][name]["sha256"] == replicas[1][name]["sha256"] for name in ("theta", "observations")):
        raise ValueError("independent bank replicas must not duplicate physical parameter/output arrays")


def _reference_certificate(reference):
    keys = ("directions", "queries", "cdfs", "mean", "sd")
    return {"schema": reference["schema"], "problem_fingerprint": reference["problem_fingerprint"],
            "observation_sha256": reference["observation_sha256"], "qualification": _json_copy(reference["qualification"]),
            "arrays": {key: {"sha256": _array_digest(reference[key]), "shape": list(np.asarray(reference[key]).shape)} for key in keys},
            "receipt_sha256": reference["receipt_sha256"]}


def _numeric_vector(value, size, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"finite complete {label} vector required")
    return array


def _reference_replay_binding(certificate, gate, request, dimension):
    """Reconstruct the original pinned receipt from compact array digests.

    The receipt binds qualification metadata and reference moments to the same
    full numerical witness arrays validated on first consumption. It is not a
    replacement for retained reference artifacts or their convergence review.
    """
    if not isinstance(certificate, dict):
        raise ValueError("pass requires retained reference numerical receipt evidence")
    arrays = certificate.get("arrays", {})
    keys = ("directions", "queries", "cdfs", "mean", "sd")
    if not isinstance(arrays, dict) or set(arrays) != set(keys) or any(not isinstance(arrays[k], dict) for k in keys):
        raise ValueError("all five reference witness array digests required")
    h = sha256()
    for key in keys:
        h.update(key.encode()); h.update(_sha(arrays[key].get("sha256"), "reference array").encode())
    meta = {key: certificate.get(key) for key in ("schema", "problem_fingerprint", "observation_sha256", "qualification")}
    h.update(json.dumps(meta, sort_keys=True, allow_nan=False).encode())
    expected = request["case"]["reference_receipt_sha256"]
    if (not expected or h.hexdigest() != expected or certificate.get("receipt_sha256") != expected
            or meta["schema"] != REFERENCE_SCHEMA or meta["problem_fingerprint"] != request["case"]["problem_fingerprint"]
            or meta["observation_sha256"] != request["case"]["observation_sha256"]):
        raise ValueError("reference numerical receipt/metadata binding mismatch")
    q = meta["qualification"]
    if q != gate.get("reference_qualification") or not isinstance(q, dict):
        raise ValueError("gate qualification must match the pinned reference receipt")
    query_shape = arrays["queries"].get("shape")
    if (arrays["directions"].get("shape") != [dimension + 32, dimension]
            or not isinstance(query_shape, list) or len(query_shape) != 2
            or query_shape[0] != dimension + 32 or type(query_shape[1]) is not int or query_shape[1] < 129
            or arrays["cdfs"].get("shape") != query_shape
            or any(arrays[k].get("shape") != [dimension] for k in ("mean", "sd"))):
        raise ValueError("reference projection/grid/moment witness shapes are inconsistent")
    return q, arrays


def _qualified_reference_replay(certificate, gate, request, dimension):
    q, arrays = _reference_replay_binding(certificate, gate, request, dimension)
    if (q.get("state") != "qualified" or q.get("contract") != CONTRACT
            or any(q.get(k) is not True for k in ("retained_refinement_passed", "independent_passed", "completed_runs", "moment_gate_passed"))):
        raise ValueError("qualified retained refinement/independent reference evidence required")
    precision = q.get("precision")
    if isinstance(precision, bool) or not isinstance(precision, (int, float)) or not isfinite(precision) or not 0 <= precision <= REFERENCE_LIMIT:
        raise ValueError("reference CDF precision must be at most 0.005")
    moment = q.get("moment_precision", {})
    mean_error = moment.get("mean_error_reference_sd"); sd_error = moment.get("sd_error_reference_sd")
    if not reference_moment_errors_valid(mean_error, sd_error, dimension) or q.get("moment_relative_precision_limit") != MOMENT_LIMIT:
        raise ValueError("finite stable reference moment uncertainty is required")
    units = _numeric_vector(q.get("projection_units"), dimension, "projection units")
    seed = q.get("projection_seed")
    if np.any(units <= 0) or type(seed) is not int or seed < 0:
        raise ValueError("predefined projection seed/units required")
    directions = predefined_directions(dimension, seed=seed, scales=units)
    if arrays["directions"]["sha256"] != _array_digest(directions):
        raise ValueError("reference joint projections must match their predefined seed/units")
    evidence = q.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 3:
        raise ValueError("reference needs refinement and independent numerical runs")
    effective = []
    for run in evidence:
        shape = run.get("shape", []) if isinstance(run, dict) else []
        diagnostics = run.get("diagnostics", {}) if isinstance(run, dict) else {}
        if not isinstance(shape, list):
            shape = []
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        if (len(shape) != 2 or type(shape[0]) is not int or shape[0] < 2 or shape[1] != dimension
                or run.get("method") not in ("numpyro_nuts", "tempered_smc")
                or run.get("source_sha256") != source_digest()
                or run.get("problem_fingerprint") != request["case"]["problem_fingerprint"]
                or run.get("observation_sha256") != request["case"]["observation_sha256"]
                or type(run.get("seed")) is not int or run["seed"] < 0
                or diagnostics.get("state") != "sampled_unqualified" or diagnostics.get("divergences", 0) != 0):
            raise ValueError("reference numerical run/source/query evidence is invalid")
        _sha(run.get("sample_sha256"), "retained reference samples")
        ess, rhat = diagnostics.get("effective_sample_size"), diagnostics.get("max_rhat", 1.)
        if (isinstance(ess, bool) or not isinstance(ess, (int, float)) or not isfinite(ess) or not 1 < ess <= shape[0]
                or isinstance(rhat, bool) or not isinstance(rhat, (int, float)) or not isfinite(rhat) or not 0 < rhat <= 1.01):
            raise ValueError("reference precision needs finite supported ESS/Rhat evidence")
        effective.append(ess)
    previous, primary, independent = evidence[-3:]
    if (previous["shape"][0] >= primary["shape"][0] or primary["seed"] == independent["seed"]
            or primary["sample_sha256"] == independent["sample_sha256"]
            or (q.get("second_method_required") and primary["method"] == independent["method"])):
        raise ValueError("reference refinement/independent evidence is inconsistent")
    least = min(effective)
    floor = np.sqrt(np.log(2 * (dimension + 32) / .01) / (2 * least))
    if precision < floor or np.any(np.asarray(mean_error) < 3 / np.sqrt(least)) or np.any(np.asarray(sd_error) < 3 / np.sqrt(2 * (least - 1))):
        raise ValueError("reference precision cannot understate retained numerical uncertainty")
    return precision, mean_error, sd_error, arrays


def _replayed_integrity(gate):
    integrity = gate.get("integrity", {})
    dimension = integrity.get("expected_dimension")
    if (type(dimension) is not int or not 2 <= dimension <= 96 or integrity.get("expected_count") != DRAW_COUNT
            or integrity.get("clipping_or_repair_performed") is not False):
        raise ValueError("complete-vector integrity evidence with no clipping is required")
    shape = integrity.get("shape")
    shape_ok = shape == [DRAW_COUNT, dimension]
    if integrity.get("complete_shape") is not shape_ok:
        raise ValueError("integrity shape and complete-shape flag contradict")
    if shape_ok:
        counts = [integrity.get(name) for name in ("nonfinite_row_count", "out_of_support_row_count")]
        if any(type(value) is not int or not 0 <= value <= DRAW_COUNT for value in counts):
            raise ValueError("finite/support integrity row counts required")
        finite = counts[0] == 0
        defined = integrity.get("prior_support_defined") is True
        supported = defined and counts[1] == 0
        if integrity.get("finite") is not finite or integrity.get("in_prior_support") is not supported:
            raise ValueError("integrity counts and finite/support flags contradict")
        passed = finite and supported
    else:
        passed = False
    if integrity.get("passed") is not passed:
        raise ValueError("integrity pass flag contradicts complete finite supported vectors")
    return dimension, passed


def _replayed_cdf_bounds(gate, dimension, precision):
    maxima = []
    for name, size in (("marginal_cdf", dimension), ("joint_projection_cdf", 32)):
        bounds = gate.get(name, {})
        lower = _numeric_vector(bounds.get("lower"), size, name + " lower bounds")
        upper = _numeric_vector(bounds.get("upper"), size, name + " upper bounds")
        # The underlying whole-line bound adds precision to the point error;
        # a positive lower already subtracts that same precision. Both bounds
        # must therefore include this 2p gap, except where the upper clips at1.
        consistent_upper = np.where(lower > 0, np.minimum(1., lower + 2 * precision), precision)
        if (np.any(lower < 0) or np.any(lower > upper) or np.any(upper > 1)
                or np.any(lower > 1 - precision + 1e-12)
                or np.any(upper + 1e-12 < consistent_upper)
                or bounds.get("max_lower") != float(lower.max()) or bounds.get("max_upper") != float(upper.max())):
            raise ValueError("retained numeric CDF bounds are inconsistent with reference uncertainty")
        actual_pass = bool(np.all(upper <= CDF_LIMIT))
        if bounds.get("passed") is not actual_pass:
            raise ValueError("CDF pass flag contradicts retained numeric bounds")
        maxima.append((float(lower.max()), float(upper.max())))
    return maxima


def _replayed_moments(gate, dimension, mean_error, sd_error, arrays):
    moments = gate.get("moments", {})
    actual = {name: _numeric_vector(moments.get(name), dimension, name) for name in
              ("sample_mean", "sample_sd", "reference_mean", "reference_sd")}
    if np.any(actual["sample_sd"] < 0) or np.any(actual["reference_sd"] <= 0):
        raise ValueError("finite nonnegative sample SD and positive reference SD required")
    if any(_array_digest(actual["reference_" + name]) != arrays[name]["sha256"] for name in ("mean", "sd")):
        raise ValueError("gate reference moments must match pinned numerical witnesses")
    me, se = np.asarray(mean_error), np.asarray(sd_error)
    if me.shape != (dimension,) or se.shape != (dimension,) or not np.all(se < 1):
        if moments.get("passed") is not False or moments.get("conclusively_failed") is not False or moments.get("reason") != "unqualified_moment_uncertainty":
            raise ValueError("invalid reference moment uncertainty must remain inconclusive")
        return {"passed": False, "conclusively_failed": False}
    recomputed = moment_intervals(actual["sample_mean"], actual["sample_sd"], actual["reference_mean"], actual["reference_sd"], me, se)
    for name in ("absolute_bias_reference_sd", "sd_ratio", "bias_lower", "bias_upper", "ratio_lower", "ratio_upper"):
        retained = _numeric_vector(moments.get(name), dimension, name)
        if not np.allclose(retained, recomputed[name], rtol=1e-12, atol=1e-12):
            raise ValueError("retained moment bounds do not match numerical uncertainty propagation")
    if any(moments.get(name) is not recomputed[name] for name in ("passed", "conclusively_failed")):
        raise ValueError("moment pass/failure flags contradict numeric bounds")
    return recomputed


def _verify_numeric_assessment(record, request):
    """Replay exactly assess_samples' ordering, including conservative labels."""
    gate = record["gate"]
    limits = {"cdf": CDF_LIMIT, "absolute_bias_reference_sd": BIAS_LIMIT, "sd_ratio": list(SD_LIMITS),
              "reference_precision": REFERENCE_LIMIT, "reference_moment_relative_precision": MOMENT_LIMIT}
    if (gate.get("schema") != "posteriorforge.study.posterior_gate.v1" or gate.get("limits") != limits
            or gate.get("expected_count") != DRAW_COUNT):
        raise ValueError("record must use the unchanged absolute scientific gates")
    dimension, valid_samples = _replayed_integrity(gate)
    if not valid_samples:
        expected = "invalid_samples"
        if gate.get("reference_qualified") is not False or any(name in gate for name in ("reference_qualification", "moments", "marginal_cdf", "joint_projection_cdf")):
            raise ValueError("invalid samples cannot have assessed reference/CDF/moment evidence")
    else:
        if record.get("artifact_draw_count") != DRAW_COUNT:
            raise ValueError("assessment requires all 50,000 complete artifact vectors")
        _sha(record.get("artifact_manifest_sha256"), "posterior manifest")
        certificate = record.get("reference_certificate")
        if certificate is None:
            expected = "reference_inconclusive"
            if (gate.get("reference_qualified") is not False or gate.get("reason") != "missing_reference"
                    or record.get("reference_binding_verified") is not False or record.get("reference_receipt_sha256") is not None
                    or any(name in gate for name in ("reference_qualification", "moments", "marginal_cdf", "joint_projection_cdf"))):
                raise ValueError("missing reference numerical evidence must remain explicitly inconclusive")
        else:
            if record.get("reference_binding_verified") is not True or record.get("reference_receipt_sha256") != request["case"]["reference_receipt_sha256"]:
                raise ValueError("reference receipt binding evidence is required")
            q, arrays = _reference_replay_binding(certificate, gate, request, dimension)
            try:
                precision, mean_error, sd_error, _ = _qualified_reference_replay(certificate, gate, request, dimension)
                qualified = True
            except ValueError:
                qualified = False
                try:
                    precision = float(q["precision"])
                    mean_error = np.asarray(q["moment_precision"]["mean_error_reference_sd"], dtype=np.float64)
                    sd_error = np.asarray(q["moment_precision"]["sd_error_reference_sd"], dtype=np.float64)
                except (KeyError, ValueError, TypeError):
                    precision, mean_error, sd_error = 1., np.ones(dimension), np.ones(dimension)
            if gate.get("reference_qualified") is not qualified:
                raise ValueError("reference-qualified flag contradicts retained numerical evidence")
            used_precision = precision if np.isfinite(precision) and 0 <= precision <= 1 else 0.
            bounds = _replayed_cdf_bounds(gate, dimension, used_precision)
            moments = _replayed_moments(gate, dimension, mean_error, sd_error, arrays)
            if not qualified:
                expected = "reference_inconclusive"
            elif moments["conclusively_failed"] or any(lower > CDF_LIMIT for lower, _ in bounds):
                expected = "model_failure"
            elif not moments["passed"]:
                expected = "moment_inconclusive"
            elif all(upper <= CDF_LIMIT for _, upper in bounds):
                expected = "pass"
            else:
                expected = "cdf_inconclusive"
    if record["status"] != expected or gate.get("passed") is not (expected == "pass"):
        raise ValueError("retained numeric evidence does not support the declared assessment status")


def _validate_study_provenance(provenance, model, *, scope):
    """Admit the actual common backend and production training state only.

    Diagnostic fixtures may exercise another explicit analytical law. A frozen
    baseline has no training state; trained study checkpoints must carry the
    literal false diagnostic flag written by their verified checkpoint loader.
    """
    if scope == "diagnostic":
        return
    if scope != "study" or not isinstance(provenance, Mapping):
        raise ValueError("study artifact requires actual backend/training provenance")
    if provenance.get("sampling_law") != STUDY_SAMPLING_LAW or provenance.get("source_policy") != STUDY_SOURCE_POLICY:
        raise ValueError("study artifact must use the fixed canonical sampling law and source policy")
    if model["method"] != "frozen" and provenance.get("training_diagnostic") is not False:
        raise ValueError("trained study artifact must come from non-diagnostic training")


def _validate_record_artifact_provenance(record, request, *, scope):
    if scope == "diagnostic":
        return
    raw = record.get("artifact_manifest_json")
    if not isinstance(raw, str) or sha256(raw.encode("utf-8")).hexdigest() != record.get("artifact_manifest_sha256"):
        raise ValueError("study replay needs the actual artifact manifest checksum evidence")
    manifest = json.loads(raw)
    provenance = record.get("artifact_provenance")
    if (manifest.get("schema") != SAMPLE_SCHEMA or manifest.get("kind") != "posterior"
            or manifest.get("requested_count") != DRAW_COUNT
            or manifest.get("problem_fingerprint") != request["case"]["problem_fingerprint"]
            or manifest.get("provenance") != provenance or not isinstance(provenance, Mapping)):
        raise ValueError("retained artifact provenance must match the pinned manifest")
    expected = {"model_sha256": request["model"]["model_sha256"],
                "protocol_sha256": request["protocol_sha256"],
                "bank_manifest_sha256": request["bank"]["bank_manifest_sha256"],
                "posterior_seed": request["posterior_seed"]}
    if request["model"]["method"] != "frozen":
        expected["checkpoint_manifest_sha256"] = request["model"]["checkpoint_manifest_sha256"]
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("study replay artifact model/bank/protocol/seed binding mismatch")
    _validate_study_provenance(provenance, request["model"], scope=scope)


def _record(request):
    return {"schema": RECORD_SCHEMA, "request_id": request["request_id"],
            "protocol_sha256": request["protocol_sha256"], "split": request["split"],
            "version_id": request["model"]["version_id"], "checkpoint_id": request["model"]["checkpoint_id"],
            "model_sha256": request["model"]["model_sha256"],
            "setup_id": request["case"]["setup_id"], "case_id": request["case"]["case_id"],
            "bank_seed": request["bank"]["bank_seed"], "context_size": request["context_size"],
            "posterior_draws": DRAW_COUNT, "artifact_draw_count": 0, "status": "failed", "gate": None,
            "forward_calls": None, "bank_generation_forward_calls": None,
            "artifact_manifest_sha256": None, "artifact_provenance": None, "artifact_manifest_json": None,
            "reference_receipt_sha256": None,
            "reference_binding_verified": False, "reference_certificate": None,
            "bank_replica_certificate": None, "error": None}


def _finish_record(record):
    return dict(record, record_sha256=_digest(record))


def _verify_record(record, request):
    if record.get("schema") != RECORD_SCHEMA or record.get("record_sha256") != _digest(_without(record, "record_sha256")):
        raise ValueError("evaluation record checksum mismatch")
    initial = _record(request)
    for name in ("request_id", "protocol_sha256", "split", "version_id", "checkpoint_id", "model_sha256", "setup_id", "case_id", "bank_seed", "context_size", "posterior_draws"):
        if record.get(name) != initial[name]:
            raise ValueError("evaluation record/request binding mismatch")
    if record.get("status") not in ("pass", "model_failure", "invalid_samples", "reference_inconclusive", "moment_inconclusive", "cdf_inconclusive", "failed", "incomplete"):
        raise ValueError("explicit evaluation status required")
    gate = record.get("gate")
    if gate is not None and (not isinstance(gate, dict) or gate.get("status") != record["status"] or gate.get("problem_fingerprint") != request["case"]["problem_fingerprint"]):
        raise ValueError("assessment/status binding mismatch")
    if gate is None:
        if record["status"] not in ("failed", "incomplete", "reference_inconclusive"):
            raise ValueError("assessed scientific status requires retained numeric gate evidence, not a boolean")
        return
    if record["status"] in ("failed", "incomplete"):
        raise ValueError("backend failure/incomplete record must not contradict an assessed gate")
    _verify_numeric_assessment(record, request)


def consume_evaluation_plan(plan, *, case_provider, bank_provider, posterior_provider, reference_provider,
                            output_directory=None, selection_lock=None,
                            reference_preflight="retain_available_artifacts"):
    """Assess existing artifacts, retaining exactly one record per request.

    Providers: case(case_binding)->(ProblemSpec, observation); bank(bank_binding)
    ->directory; posterior(request)->{directory, manifest_sha256} or None;
    reference(case_binding)->witness mapping or None. They must locate prepared
    artifacts; this function performs no model or simulator evaluations.
    """
    plan = _verify_plan(plan)
    if reference_preflight not in ("retain_available_artifacts", "require_qualified"):
        raise ValueError("explicit reference preflight policy required")
    if plan["split"] == "test":
        _check_test_lock(plan, selection_lock)  # Before any provider can see data.
    root = None if output_directory is None else Path(output_directory)
    if root is not None:
        root.mkdir(parents=True, exist_ok=False)
        (root / "plan.json").write_bytes(_bytes(plan) + b"\n")
    queries, banks, references, records = {}, {}, {}, []
    def get_reference(case, key):
        if key not in references:
            try:
                reference = reference_provider(_json_copy(case))
                expected = case["reference_receipt_sha256"]
                if reference is not None and (expected is None or reference.get("receipt_sha256") != expected or reference_receipt(reference) != expected):
                    raise ValueError("reference receipt binding mismatch")
                references[key] = reference, None
            except Exception as exc:
                references[key] = None, {"stage": "reference", "reason": type(exc).__name__, "detail": str(exc)}
        return references[key]
    for request in plan["requests"]:
        record, stage = _record(request), "bank"
        try:
            binding = request["bank"]
            bank_key = (binding["setup_id"], binding["bank_seed"])
            if bank_key not in banks:
                banks[bank_key] = load_bank(bank_provider(_json_copy(binding)), expected_sha256=binding["bank_manifest_sha256"])
            record.update(_bank_costs(banks[bank_key], request))
            for paired in (row for row in plan["banks"] if row["setup_id"] == binding["setup_id"]):
                paired_key = (paired["setup_id"], paired["bank_seed"])
                if paired_key not in banks:
                    banks[paired_key] = load_bank(bank_provider(_json_copy(paired)), expected_sha256=paired["bank_manifest_sha256"])
            record["bank_replica_certificate"] = _bank_replica_certificate(plan, binding["setup_id"], banks)
            stage = "query"
            case = request["case"]; key = (case["setup_id"], case["case_id"])
            if key not in queries:
                spec, observation = case_provider(_json_copy(case))
                y = np.asarray(observation, dtype=np.float64)
                if (not isinstance(spec, ProblemSpec) or spec.fingerprint != case["problem_fingerprint"]
                        or spec.split != plan["split"] or y.shape != (spec.observation_dimension,)
                        or not np.isfinite(y).all() or observation_digest(y) != case["observation_sha256"]):
                    raise ValueError("query/spec checksum binding mismatch")
                queries[key] = spec, y.copy()
            spec, y = queries[key]
            if reference_preflight == "require_qualified":
                stage = "reference_preflight"
                reference, error = get_reference(case, key)
                try:
                    qualified = reference is not None and _reference_arrays(reference, spec, y)[-1]
                except Exception as exc:
                    qualified = False
                    error = {"stage": stage, "reason": type(exc).__name__, "detail": str(exc)}
                if not qualified:
                    record.update(status="reference_inconclusive", error=error or {"stage": stage, "reason": "missing_or_unqualified_reference"},
                                  posterior_execution="skipped_unqualified_reference", reference_binding_verified=reference is not None,
                                  reference_receipt_sha256=None if reference is None else reference["receipt_sha256"])
                    raise _ReferenceSkipped()
            stage = "posterior"
            location = posterior_provider(_json_copy(request))
            if location is None:
                record.update(status="incomplete", error={"stage": stage, "reason": "posterior_artifact_missing"})
            else:
                if not isinstance(location, Mapping) or set(location) != {"directory", "manifest_sha256"}:
                    raise ValueError("posterior provider must return directory and expected manifest checksum")
                digest = _sha(location["manifest_sha256"], "posterior manifest")
                actual_spec, values, actual_y, manifest = load_samples(location["directory"], expected_manifest_sha256=digest)
                provenance = manifest["provenance"]
                record["artifact_provenance"] = _json_copy(provenance)
                expected_provenance = {"model_sha256": request["model"]["model_sha256"],
                    "protocol_sha256": plan["protocol_sha256"], "bank_manifest_sha256": binding["bank_manifest_sha256"]}
                if (manifest["kind"] != "posterior" or actual_spec.fingerprint != spec.fingerprint
                        or not np.array_equal(actual_y, y) or manifest["bank_cost"]["context_size"] != request["context_size"]
                        or any(provenance.get(k) != v for k, v in expected_provenance.items())
                        or type(provenance.get("posterior_seed")) is not int or provenance["posterior_seed"] != request["posterior_seed"]
                        or manifest["bank_cost"]["context_forward_evaluations"] != record["forward_calls"]
                        or manifest["bank_cost"]["shared_bank_generation_forward_evaluations"] != record["bank_generation_forward_calls"]):
                    raise ValueError("posterior request/model/query/bank/context binding mismatch")
                # Additional training receipt is required for trained versions.
                if request["model"]["method"] != "frozen" and provenance.get("checkpoint_manifest_sha256") != request["model"]["checkpoint_manifest_sha256"]:
                    raise ValueError("trained checkpoint manifest binding mismatch")
                _validate_study_provenance(provenance, request["model"], scope=plan["scope"])
                manifest_bytes = (Path(location["directory"]) / "manifest.json").read_bytes()
                if sha256(manifest_bytes).hexdigest() != digest:
                    raise ValueError("sample manifest changed while assessing the artifact")
                record["artifact_manifest_json"] = manifest_bytes.decode("utf-8")
                record["artifact_manifest_sha256"] = digest
                record["artifact_draw_count"] = DRAW_COUNT
                stage = "reference"
                reference, error = get_reference(case, key)
                record["reference_binding_verified"] = reference is not None
                record["reference_receipt_sha256"] = None if reference is None else reference["receipt_sha256"]
                record["reference_certificate"] = None if reference is None else _reference_certificate(reference)
                stage = "assessment"
                try:
                    gate = assess_samples(values, reference, spec=spec, observation=y)
                    record.update(status=gate["status"], gate=gate, error=error)
                except Exception as exc:
                    record.update(status="reference_inconclusive", error={"stage": stage, "reason": type(exc).__name__, "detail": str(exc)})
        except _ReferenceSkipped:
            pass
        except Exception as exc:
            record.update(status="failed", error={"stage": stage, "reason": type(exc).__name__, "detail": str(exc)})
        record = _finish_record(record)
        records.append(record)
        if root is not None:
            with (root / (request["request_id"] + ".json")).open("xb") as handle:
                handle.write(_bytes(record) + b"\n")
    result = {"schema": "finetuneforge.evaluation_consumption.v1", "plan_sha256": plan["plan_sha256"],
              "record_count": len(records), "records_sha256": _digest(records), "records": records,
              "gpu_jobs_submitted": 0, "posterior_draws_per_request": DRAW_COUNT}
    if root is not None:
        manifest = _without(result, "records")
        with (root / "manifest.json").open("xb") as handle:
            handle.write(_bytes(manifest) + b"\n")
        result["manifest_sha256"] = sha256((root / "manifest.json").read_bytes()).hexdigest()
    return result


def load_evaluation_records(directory, *, expected_manifest_sha256, expected_plan_sha256):
    """Load immutable per-cell files through an externally retained manifest hash."""
    root = Path(directory)
    raw = (root / "manifest.json").read_bytes()
    if sha256(raw).hexdigest() != _sha(expected_manifest_sha256, "evaluation manifest"):
        raise ValueError("evaluation manifest checksum mismatch")
    manifest = json.loads(raw)
    plan = _verify_plan(json.loads((root / "plan.json").read_bytes()))
    if (plan["plan_sha256"] != _sha(expected_plan_sha256, "evaluation plan")
            or manifest.get("schema") != "finetuneforge.evaluation_consumption.v1"
            or manifest.get("plan_sha256") != expected_plan_sha256):
        raise ValueError("evaluation manifest/plan binding mismatch")
    records = [json.loads((root / (request["request_id"] + ".json")).read_bytes()) for request in plan["requests"]]
    if manifest.get("record_count") != len(records) or manifest.get("records_sha256") != _digest(records):
        raise ValueError("evaluation record matrix checksum mismatch")
    _validated_records(plan, records)
    return plan, records


def _validated_records(plan, records):
    _verify_plan(plan)
    rows = list(records)
    index = {}
    for row in rows:
        key = row.get("request_id")
        if key in index:
            raise ValueError("duplicate evaluation record")
        index[key] = row
    if set(index) != {r["request_id"] for r in plan["requests"]}:
        raise ValueError("complete validation request matrix required; absent cells cannot be sealed")
    ordered = []
    for request in plan["requests"]:
        record = index[request["request_id"]]
        _verify_record(record, request)
        if record.get("artifact_manifest_sha256") is not None:
            _validate_record_artifact_provenance(record, request, scope=plan["scope"])
        if record.get("bank_replica_certificate") is not None:
            _verify_bank_replica_certificate(record["bank_replica_certificate"], request["case"]["setup_id"], plan["banks"])
        elif record["status"] == "pass":
            raise ValueError("pass requires independent bank replica evidence")
        ordered.append(record)
    return ordered


def _worst_violation(rows):
    values = []
    for row in rows:
        gate = row.get("gate") or {}
        try:
            moments = gate["moments"]
            values.extend([gate["marginal_cdf"]["max_upper"] / .1 - 1,
                           gate["joint_projection_cdf"]["max_upper"] / .1 - 1,
                           max(moments["bias_upper"]) / .2 - 1,
                           max(.8 / v - 1 for v in moments["ratio_lower"]),
                           max(v / 1.2 - 1 for v in moments["ratio_upper"])])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    value = max([0.] + values)
    return float(value) if isfinite(value) else None


def summarize_validation(plan, records):
    """Use only complete validation records; scientific failures remain visible."""
    if plan.get("split") != "validation":
        raise ValueError("validation-only checkpoint selection required")
    rows = _validated_records(plan, records)
    unavailable = [r["request_id"] for r in rows if type(r.get("forward_calls")) is not int or r["forward_calls"] < 0 or r.get("bank_replica_certificate") is None]
    if unavailable:
        return {"status": "selection_unavailable", "reason": "unavailable_authenticated_bank_costs",
                "requests": unavailable, "selected_models": [], "primary_rank": None,
                "scientifically_qualified": False}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["version_id"]].append(row)
    cases = [{"setup_id": c["setup_id"], "case_id": c["case_id"]} for c in plan["cases"]]
    baseline = next(v for v in plan["models"] if v["method"] == "frozen")
    baseline_summary = passing_budgets(grouped[baseline["version_id"]], cases=cases)
    baseline_pass = [v for v in baseline_summary if v["minimum"] is not None]
    runs, selected, checkpoints, defaults = defaultdict(list), [baseline], [], {}
    by_checkpoint = {v["checkpoint_id"]: v for v in plan["models"]}
    for model in plan["models"]:
        if model["method"] != "frozen":
            runs[(model["method"], model["rank"], model["training_seed"])].append(model)
    rank_rows = {4: [], 16: []}
    for run, models in sorted(runs.items(), key=lambda item: str(item[0])):
        candidates = [{"checkpoint_id": v["checkpoint_id"], "update": v["update"], "training_seed": v["training_seed"],
                       "records": grouped[v["version_id"]], "worst_normalized_violation": _worst_violation(grouped[v["version_id"]])} for v in models]
        choice = select_checkpoint(candidates, cases=cases, baseline_qualified_cases=baseline_pass)
        model = by_checkpoint[choice["checkpoint_id"]]
        selected.append(model)
        checkpoints.append(dict(choice, version_id=model["version_id"], method=run[0], rank=run[1]))
        if run[0] == "lora":
            rank_rows[run[1]].append(choice)
    for model in selected:
        default = select_default_bank(grouped[model["version_id"]], cases=cases)
        if plan["scope"] == "diagnostic" and default["status"] == "qualified":
            default = dict(default, status="unqualified", diagnostic_candidate_context_size=default["context_size"],
                           context_size=None, reason="diagnostic_matrix_scope")
        defaults[model["version_id"]] = default
    if all(len(v) == 2 for v in rank_rows.values()):
        primary = select_primary_rank(rank_rows, cases=cases)
    else:
        primary = {"candidate_rank": None, "primary_rank": None, "status": "diagnostic_only",
                   "lock_reasons": ["both_training_replicas_of_both_ranks_required"]}
    if plan["scope"] == "diagnostic":
        primary = dict(primary, primary_rank=None, status="diagnostic_only", primary_rank_qualified_for_lock=False,
                       lock_reasons=list(primary.get("lock_reasons", [])) + ["diagnostic_matrix_scope"])
    return {"status": "validation_selection_complete", "selected_models": sorted(selected, key=lambda v: v["version_id"]),
            "all_model_versions": plan["models"], "checkpoints": checkpoints, "primary": primary,
            "primary_rank": primary["primary_rank"], "default_banks": defaults,
            "validation_case_count": len(cases), "validation_record_count": len(rows),
            "scientifically_qualified": False, "savings_claim": None,
            "reporting_policy": "retain_all_methods_training_replicas_and_tested_budgets"}


@dataclass(frozen=True)
class SelectionLock:
    """Immutable raw bytes authenticated by an externally retained file checksum."""
    raw: bytes
    sha256: str

    def __post_init__(self):
        _sha(self.sha256, "selection lock")
        if not isinstance(self.raw, bytes) or sha256(self.raw).hexdigest() != self.sha256:
            raise ValueError("selection lock file checksum mismatch")
        document = json.loads(self.raw)
        if (document.get("schema") != LOCK_SCHEMA or document.get("lock_receipt_sha256") != _digest(_without(document, "lock_receipt_sha256"))
                or document.get("status") not in ("locked_diagnostic_inputs", "locked_validation_selected_inputs")
                or document.get("scientifically_qualified") is not False or document.get("savings_claim") is not None):
            raise ValueError("invalid selection lock receipt")

    @property
    def document(self):
        return json.loads(self.raw)


def seal_selection(path, *, validation_plan, records, test_catalogue_sha256):
    """Freeze all selected replicas after complete validation, never claim a pass.

    Failed/inconclusive cells are completed records. Missing cells are rejected.
    A missing qualified primary/default does not prevent a diagnostic input lock.
    """
    _sha(test_catalogue_sha256, "sealed test catalogue")
    rows = _validated_records(validation_plan, records)
    summary = summarize_validation(validation_plan, rows)
    if summary["status"] != "validation_selection_complete":
        raise ValueError("cannot select checkpoints without authenticated bank costs")
    diagnostic = summary["primary_rank"] is None
    body = {"schema": LOCK_SCHEMA, "status": "locked_diagnostic_inputs" if diagnostic else "locked_validation_selected_inputs",
            "protocol_sha256": validation_plan["protocol_sha256"], "scope": validation_plan["scope"],
            "validation_plan_sha256": validation_plan["plan_sha256"],
            "validation_case_catalogue_sha256": validation_plan["case_catalogue_sha256"],
            "validation_cases": validation_plan["cases"], "master_seed": validation_plan["master_seed"],
            "validation_records_sha256": _digest(rows), "validation_record_count": len(rows),
            "test_case_catalogue_sha256": test_catalogue_sha256,
            "all_model_versions": validation_plan["models"], "selected_models": summary["selected_models"],
            "selection": summary, "diagnostic_only": diagnostic, "primary_rank": summary["primary_rank"],
            "scientifically_qualified": False, "savings_claim": None,
            "test_selection_allowed": False, "checkpoint_or_seed_substitution_allowed": False}
    document = dict(body, lock_receipt_sha256=_digest(body))
    raw = _bytes(document) + b"\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(raw)
    target.chmod(0o444)
    return SelectionLock(raw, sha256(raw).hexdigest())


def load_selection_lock(path, *, expected_sha256, protocol_sha256, test_catalogue_sha256):
    lock = SelectionLock(Path(path).read_bytes(), expected_sha256)
    document = lock.document
    if document["protocol_sha256"] != _sha(protocol_sha256, "protocol") or document["test_case_catalogue_sha256"] != _sha(test_catalogue_sha256, "test catalogue"):
        raise ValueError("selection lock protocol/test catalogue mismatch")
    return lock


def build_test_plan(protocol_sha256, cases, banks, *, selection_lock, master_seed=None):
    if not isinstance(selection_lock, SelectionLock):
        raise ValueError("checksum-authenticated selection lock required before test planning")
    document = selection_lock.document
    cases = list(cases)
    catalogue = case_catalogue_sha256(cases)
    if master_seed is not None and master_seed != document["master_seed"]:
        raise ValueError("test posterior seed policy was frozen before test access")
    master_seed = document["master_seed"]
    prior_cases = document["validation_cases"]
    prior_identities = {(v["setup_id"], v["case_id"]) for v in prior_cases}
    prior_queries = {v["observation_sha256"] for v in prior_cases}
    if any((v["setup_id"], v["case_id"]) in prior_identities or v["observation_sha256"] in prior_queries for v in cases):
        raise ValueError("fresh sealed test cases must not reuse validation identities or queries")
    if document["protocol_sha256"] != protocol_sha256 or document["test_case_catalogue_sha256"] != catalogue:
        raise ValueError("selection lock protocol/test catalogue mismatch")
    plan = _make_plan(protocol_sha256, document["selected_models"], cases, banks, split="test", master_seed=master_seed,
                      scope=document["scope"], contexts=BANK_SIZES, selection_lock_sha256=selection_lock.sha256)
    _check_test_lock(plan, selection_lock)
    return plan


def _check_test_lock(plan, lock):
    if not isinstance(lock, SelectionLock):
        raise ValueError("checksum-authenticated selection lock required before test access")
    document = lock.document
    prior = document["validation_cases"]
    previous_ids = {(v["setup_id"], v["case_id"]) for v in prior}
    previous_queries = {v["observation_sha256"] for v in prior}
    if any((v["setup_id"], v["case_id"]) in previous_ids or v["observation_sha256"] in previous_queries for v in plan["cases"]):
        raise ValueError("fresh sealed test cases must not reuse validation identities or queries")
    if (plan["selection_lock_sha256"] != lock.sha256 or plan["protocol_sha256"] != document["protocol_sha256"]
            or plan["case_catalogue_sha256"] != document["test_case_catalogue_sha256"]
            or plan["models"] != document["selected_models"] or plan["scope"] != document["scope"]
            or plan["master_seed"] != document["master_seed"]):
        raise ValueError("test plan/selection lock binding mismatch; late model/seed substitution forbidden")


def summarize_test(plan, records, *, selection_lock, bootstrap_repetitions=10000, seed=None):
    """Report locked replicas and groups without selecting with test outcomes.

    Intervals are per training replica, with bank seeds paired inside cases and
    cases clustered by setup/matched pair. They are not pooled as independent
    training replicas. This report never manufactures a headline transfer claim.
    """
    if plan.get("split") != "test":
        raise ValueError("sealed-test plan required for test reporting")
    _check_test_lock(plan, selection_lock)
    frozen_seed = selection_lock.document["master_seed"]
    if seed is not None and seed != frozen_seed:
        raise ValueError("bootstrap seed policy was frozen before test access")
    seed = frozen_seed
    if plan["scope"] == "study" and bootstrap_repetitions != 10000:
        raise ValueError("study inference requires the fixed 10,000 cluster-bootstrap repetitions")
    rows = _validated_records(plan, records)
    if any(type(r.get("forward_calls")) is not int or r["forward_calls"] < 0 or r.get("bank_replica_certificate") is None for r in rows):
        return {"status": "inconclusive", "reason": "unavailable_authenticated_bank_costs", "savings_claim": None}
    by_version = defaultdict(list)
    for row in rows:
        by_version[row["version_id"]].append(row)
    summaries = {version: passing_budgets(values, cases=plan["cases"], split="test") for version, values in by_version.items()}
    baseline = next(v["version_id"] for v in plan["models"] if v["method"] == "frozen")
    cluster_by_setup = {v["setup_id"]: v.get("cluster_id", v["setup_id"]) for v in plan["cases"]}
    groups = {name: {(v["setup_id"], v["case_id"]) for v in plan["cases"] if v.get("test_group") == name}
              for name in ("familiar", "upper_sizes", "sharing", "heldout_physics", "combined")}
    def compare(version, identities=None):
        b = summaries[baseline]; a = summaries[version]
        if identities is not None:
            b = [r for r in b if (r["setup_id"], r["case_id"]) in identities]
            a = [r for r in a if (r["setup_id"], r["case_id"]) in identities]
        if not b:
            return {"status": "inconclusive", "reason": "no_predefined_group_cases", "ratio": None}
        return paired_call_savings(b, a, cluster_by_setup=cluster_by_setup,
                                   bootstrap_repetitions=bootstrap_repetitions, seed=seed)
    defaults = selection_lock.document["selection"]["default_banks"]
    comparisons = {}
    for version in sorted(by_version):
        if version == baseline:
            continue
        comparisons[version] = {"overall": compare(version), "groups": {name: compare(version, ids) for name, ids in groups.items()},
                                "interpretation": "per_replica_descriptive_evidence; no_test_driven_seed_or_rank_selection"}
    default_results = {}
    for version in sorted(by_version):
        default = defaults[version]
        size = default["context_size"] if default["status"] == "qualified" else None
        selected = [r for r in by_version[version] if r["context_size"] == size] if size is not None else []
        default_results[version] = {"validation_chosen_context_size": size, "status": "not_validated" if size is None else "evaluated_unchanged",
                                    "passed_requests": sum(r["status"] == "pass" for r in selected), "total_requests": len(selected)}
    return {"status": "sealed_test_report_complete", "selection_lock_sha256": selection_lock.sha256,
            "diagnostic_only": selection_lock.document["diagnostic_only"], "primary_rank": selection_lock.document["primary_rank"],
            "comparisons": comparisons, "case_summaries": summaries, "fixed_defaults": default_results,
            "reporting_policy": "all_selected_methods_replicas_budgets_and_failures_retained",
            "savings_claim": None, "claim_scope": "review_overall_and_heldout_physics_evidence_without_selecting_best_test_replica",
            "scientifically_qualified": False}
