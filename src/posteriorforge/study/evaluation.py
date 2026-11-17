"""Pure validation-only selection and paired simulation-cost analysis.

This module never trains, samples, submits jobs or drops difficult cases.
Record lists are explicit: absent bank/case rows cannot become a passing default.
"""
from __future__ import annotations

from collections import defaultdict
from math import exp, isfinite, log
from typing import Mapping, Sequence

import numpy as np

BANK_SIZES = (16, 64, 256, 1024, 4096)


def shortlist_checkpoints(records: Sequence[Mapping], *, limit=2):
    if type(limit) is not int or not 1 <= limit <= 2:
        raise ValueError("one or two shortlisted checkpoints allowed")
    by_id = {}
    for r in records:
        if r.get("split") != "validation":
            raise ValueError("checkpoint selection requires validation-only scores")
        cid, score, update = r.get("checkpoint_id"), r.get("score"), r.get("update")
        if not isinstance(cid, str) or not cid or type(update) is not int or update < 0 or isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
            raise ValueError("finite checkpoint score, identity and update required")
        if cid in by_id:
            raise ValueError("duplicate checkpoint score identity")
        by_id[cid] = dict(r)
    return sorted(by_id.values(), key=lambda r: (r["score"], r["update"], r["checkpoint_id"]))[:limit]


def _case_identity(value):
    return (str(value["setup_id"]), str(value["case_id"]))


def _validate_rows(records, *, split):
    seen = set()
    for r in records:
        if r.get("split") != split:
            raise ValueError(f"{split}-only records required")
        case = _case_identity(r)
        key = (case, r.get("bank_seed"), r.get("context_size"))
        if key in seen:
            raise ValueError("duplicate case/bank/budget record")
        seen.add(key)
        if r.get("context_size") not in BANK_SIZES or type(r.get("bank_seed")) is not int:
            raise ValueError("declared bank size and integer bank seed required")
        calls = r.get("forward_calls")
        if type(calls) is not int or calls < 0:
            raise ValueError("nonnegative actual forward-call count required")
        if r.get("status") not in ("pass", "model_failure", "invalid_samples", "reference_inconclusive", "moment_inconclusive", "cdf_inconclusive", "failed", "incomplete"):
            raise ValueError("explicit scientific status required")


def passing_budgets(records, *, cases=None, bank_seeds=(0, 1), split="validation"):
    """One method/checkpoint's per-case smallest tested joint-seed passing budget."""
    rows = list(records)
    _validate_rows(rows, split=split)
    expected = tuple(_case_identity(r) if isinstance(r, Mapping) else tuple(map(str, r)) for r in cases) if cases is not None else tuple(sorted({_case_identity(r) for r in rows}))
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate expected case identity")
    seeds = tuple(bank_seeds)
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(type(s) is not int for s in seeds):
        raise ValueError("two independent integer bank seeds required")
    index = {(_case_identity(r), r["bank_seed"], r["context_size"]): r for r in rows}
    result = []
    for case in expected:
        passing = []
        statuses = []
        conclusively_failed = []
        budget_records = []
        generation_costs = {str(seed): None for seed in seeds}
        for size in BANK_SIZES:
            current = [index.get((case, seed, size)) for seed in seeds]
            by_seed = {str(seed): None if row is None else row["forward_calls"] for seed, row in zip(seeds, current)}
            budget_records.append({"context_size": size, "forward_calls_by_seed": by_seed,
                                   "status_by_seed": {str(seed): "missing" if row is None else row["status"] for seed, row in zip(seeds, current)}})
            for seed, row in zip(seeds, current):
                if row is None or row.get("bank_generation_forward_calls") is None:
                    continue
                total = row["bank_generation_forward_calls"]
                if type(total) is not int or total < row["forward_calls"]:
                    raise ValueError("full-bank generation count must be an integer covering the prefix count")
                key = str(seed)
                if generation_costs[key] is not None and generation_costs[key] != total:
                    raise ValueError("inconsistent full-bank generation cost across prefixes")
                generation_costs[key] = total
            statuses.extend("missing" if r is None else r["status"] for r in current)
            conclusively_failed.append(any(r is not None and r["status"] in ("model_failure", "invalid_samples", "failed", "incomplete") for r in current))
            if all(r is not None and r["status"] == "pass" for r in current):
                passing.append({"context_size": size, "forward_calls": max(r["forward_calls"] for r in current),
                                "forward_calls_by_seed": dict(by_seed),
                                "forward_calls_reduction": "max_over_required_bank_seeds"})
        unresolved_reference = "reference_inconclusive" in statuses
        best = min(passing, key=lambda r: r["context_size"]) if passing and not unresolved_reference else None
        result.append({"setup_id": case[0], "case_id": case[1], "status": "qualified" if best else "no_qualified_budget",
                       "passing_budgets": passing, "minimum": best, "bank_seeds": list(seeds),
                       "budget_records": budget_records, "bank_generation_forward_calls_by_seed": generation_costs,
                       "bank_generation_cost_source": "explicit_record_metadata_only",
                       "has_unresolved_reference": unresolved_reference, "has_missing_records": "missing" in statuses,
                       "all_budgets_conclusively_failed": bool(all(conclusively_failed) and not unresolved_reference and "missing" not in statuses)})
    return result


def select_default_bank(records, *, cases, bank_seeds=(0, 1)):
    rows = list(records)
    expected = tuple(_case_identity(r) if isinstance(r, Mapping) else tuple(map(str, r)) for r in cases)
    if not expected:
        raise ValueError("explicit nonempty validation case catalogue required")
    summaries = passing_budgets(rows, cases=expected, bank_seeds=bank_seeds)
    if any(s["has_missing_records"] for s in summaries):
        return {"status": "unqualified", "context_size": None, "reason": "missing_validation_runs", "cases": summaries}
    # Any reference-inconclusive case blocks a universal scientific default,
    # even if another request happened to carry a contradictory pass label.
    if any(s["has_unresolved_reference"] for s in summaries):
        return {"status": "unqualified", "context_size": None, "reason": "unresolved_validation_reference", "cases": summaries}
    common = set(BANK_SIZES)
    for summary in summaries:
        common &= {b["context_size"] for b in summary["passing_budgets"]}
    if not common:
        return {"status": "unqualified", "context_size": None, "reason": "no_bank_passes_every_case_and_seed", "cases": summaries}
    return {"status": "qualified", "context_size": min(common), "cases": summaries, "future_task_guarantee": False}


def select_checkpoint(candidates, *, cases, baseline_qualified_cases=(), bank_seeds=(0, 1)):
    """Select among at most two complete, validation-only checkpoint matrices.

    Cost tie-breaks use the common qualified case cohort, preventing a candidate
    from appearing cheaper merely because it failed a difficult expensive case.
    """
    if not candidates or len(candidates) > 2:
        raise ValueError("one or two shortlisted checkpoints required")
    baseline = {_case_identity(c) if isinstance(c, Mapping) else tuple(map(str, c)) for c in baseline_qualified_cases}
    prepared = []
    for candidate in candidates:
        if not candidate.get("checkpoint_id") or type(candidate.get("update")) is not int or candidate["update"] < 0:
            raise ValueError("checkpoint identity/update required")
        summaries = passing_budgets(candidate["records"], cases=cases, bank_seeds=bank_seeds)
        if any(r["has_missing_records"] for r in summaries):
            raise ValueError("complete full-validation matrices required before checkpoint selection")
        qualified = {(r["setup_id"], r["case_id"]) for r in summaries if r["minimum"] is not None}
        violation = candidate.get("worst_normalized_violation", float("inf"))
        if violation is None:
            violation = float("inf")
        if isinstance(violation, bool) or not isinstance(violation, (int, float)) or np.isnan(violation) or violation < 0:
            raise ValueError("nonnegative normalized gate violation required")
        prepared.append((candidate, summaries, qualified, violation))
    common = set.intersection(*(r[2] for r in prepared))
    if baseline:
        common &= baseline
    ranked = []
    for candidate, summaries, qualified, violation in prepared:
        calls = [r["minimum"]["forward_calls"] for r in summaries if (r["setup_id"], r["case_id"]) in common]
        if any(c <= 0 for c in calls):
            raise ValueError("trained checkpoint budgets require positive forward-call counts")
        cost = float(exp(np.mean(np.log(calls)))) if calls else float("inf")
        key = (len(baseline - qualified), -len(qualified), cost, violation, candidate["update"], candidate["checkpoint_id"])
        status = "validation_inconclusive" if any(r["has_unresolved_reference"] for r in summaries) else "selected" if qualified else "selected_without_qualified_budget"
        ranked.append((key, {"checkpoint_id": candidate["checkpoint_id"], "update": candidate["update"],
            "selection_status": status, "source_split": "validation", "training_seed": candidate.get("training_seed"),
            "validation_case_count": len(summaries), "regressions": key[0], "qualified_cases": len(qualified),
            "geometric_mean_calls": cost if np.isfinite(cost) else None,
            "cost_comparison_case_count": len(common),
            "worst_normalized_violation": violation if np.isfinite(violation) else None, "cases": summaries}))
    return min(ranked, key=lambda r: r[0])[1]


def select_primary_rank(selected_by_rank, *, cases=None):
    """Distinguish a diagnostic candidate from a conclusive primary selection.

    Every comparison reference/matrix must be conclusive. The chosen rank needs
    a qualified budget in each replica and zero regressions; shared conclusive
    hard failures may remain. Universal case success is only a default-bank
    requirement. This pure function does not persist an immutable selection lock.
    """
    if set(selected_by_rank) != {4, 16}:
        raise ValueError("both LoRA ranks 4 and 16 required")
    expected = None if cases is None else {_case_identity(c) if isinstance(c, Mapping) else tuple(map(str, c)) for c in cases}
    if expected is not None and not expected:
        raise ValueError("nonempty expected validation case catalogue required")
    scored, rows_by_rank, seed_sets = [], {}, []
    for rank, replicas in selected_by_rank.items():
        if len(replicas) != 2:
            raise ValueError("two independent validation replicas required per rank")
        if any(r.get("source_split") != "validation" for r in replicas):
            raise ValueError("primary-rank selection requires validation-only checkpoint summaries")
        seeds = [r.get("training_seed") for r in replicas]
        identities = [r.get("checkpoint_id") for r in replicas]
        if any(type(v) is not int for v in seeds) or len(set(seeds)) != 2 or any(not isinstance(v, str) or not v for v in identities) or len(set(identities)) != 2:
            raise ValueError("two distinct training-seed and checkpoint identities required")
        seed_sets.append(set(seeds))
        calls = [r.get("geometric_mean_calls") for r in replicas]
        violations = [r.get("worst_normalized_violation") for r in replicas]
        key = (sum(r["regressions"] for r in replicas), -sum(r["qualified_cases"] for r in replicas),
               float(np.mean(np.log(calls))) if all(c is not None and c > 0 for c in calls) else float("inf"),
               max(violations) if all(v is not None for v in violations) else float("inf"), rank)
        scored.append((key, rank))
        rows_by_rank[rank] = list(replicas)
        if expected is None:
            for replica in replicas:
                details = replica.get("cases", [])
                if details:
                    expected = {_case_identity(c) for c in details}
                    break
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("matched training-seed sets required across LoRA ranks")
    comparison_reasons, qualification = [], {}
    for rank in (4, 16):
        replica_checks = []
        for replica in rows_by_rank[rank]:
            details = replica.get("cases", [])
            reasons = []
            identities = [_case_identity(c) for c in details]
            if not expected or set(identities) != expected or len(identities) != len(set(identities)):
                reasons.append("missing_or_mismatched_validation_case_catalogue")
            if any(c.get("has_missing_records", True) for c in details):
                reasons.append("missing_validation_records")
            if any(c.get("has_unresolved_reference", True) for c in details) or replica.get("selection_status") == "validation_inconclusive":
                reasons.append("unresolved_validation_reference")
            if any(c.get("minimum") is None and not c.get("all_budgets_conclusively_failed", False) for c in details):
                reasons.append("unresolved_model_gate")
            if replica.get("selection_status") not in ("selected", "selected_without_qualified_budget", "validation_inconclusive"):
                reasons.append("unsupported_checkpoint_selection_status")
            if replica.get("qualified_cases") != sum(c.get("minimum") is not None for c in details):
                reasons.append("qualified_case_count_mismatch")
            comparison_reasons.extend(f"rank{rank}/seed{replica['training_seed']}:{reason}" for reason in reasons)
            replica_checks.append({"training_seed": replica["training_seed"], "checkpoint_id": replica["checkpoint_id"],
                                   "comparison_conclusive": not reasons, "reasons": reasons})
        chosen_reasons = []
        if any(r.get("selection_status") != "selected" or r.get("qualified_cases", 0) <= 0 for r in rows_by_rank[rank]):
            chosen_reasons.append("no_qualified_budget_in_every_replica")
        if any(r.get("regressions", 0) != 0 for r in rows_by_rank[rank]):
            chosen_reasons.append("new_failures_on_baseline_qualified_cases")
        qualification[str(rank)] = {"replicas": replica_checks, "chosen_rank_reasons": chosen_reasons}
    candidate = min(scored)[1]
    primary_reasons = comparison_reasons + qualification[str(candidate)]["chosen_rank_reasons"]
    primary = candidate if not primary_reasons else None
    return {"candidate_rank": candidate, "primary_rank": primary,
            "status": "qualified_for_lock" if primary is not None else "diagnostic_only",
            "primary_rank_qualified_for_lock": primary is not None, "selection_locked": False,
            "lock_reasons": primary_reasons, "rank_qualification": qualification,
            "source_split": "validation", "training_replicas_per_rank": 2,
            "reporting_policy": "retain_all_methods_and_training_replicas",
            "test_seed_selection_performed": False,
            "qualification_scope": "validation_selection_only_not_global_accuracy_or_transfer_certification"}


def _bank_generation_costs(baseline, adapted):
    """Count explicit complete-bank generation once per setup/bank, not per run."""
    known, expected = {}, set()
    for row in list(baseline) + list(adapted):
        seeds = row.get("bank_seeds", [])
        values = row.get("bank_generation_forward_calls_by_seed", {})
        for seed in seeds:
            key = (str(row["setup_id"]), str(seed))
            expected.add(key)
            value = values.get(str(seed))
            if value is None:
                continue
            if type(value) is not int or value < 0:
                raise ValueError("nonnegative explicit full-bank generation count required")
            if key in known and known[key] != value:
                raise ValueError("inconsistent shared bank-generation cost")
            known[key] = value
    complete = bool(expected) and expected == set(known)
    return {"status": "recorded" if complete else "incomplete", "total_forward_calls": sum(known.values()) if complete else None,
            "known_forward_calls": sum(known.values()),
            "banks": [{"setup_id": setup, "bank_seed": seed, "forward_calls": known.get((setup, seed))} for setup, seed in sorted(expected)],
            "scope": "unique_target_context_banks_in_this_comparison; training_query_reference_costs_are_separate",
            "counted_once_per_setup_and_bank_seed": True, "included_in_minimum_prefix_ratio": False}


def _matching_seed_costs(summary, minimum):
    values = minimum.get("forward_calls_by_seed")
    if not isinstance(values, Mapping):
        return None
    values = {str(key): value for key, value in values.items()}
    seeds = {str(seed) for seed in summary.get("bank_seeds", [])}
    if len(values) != 2 or len(seeds) != 2 or set(values) != seeds:
        return None
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("matched simulator-call ratios require positive actual counts for each bank seed")
    return values


def paired_call_savings(baseline, adapted, *, cluster_by_setup=None, bootstrap_repetitions=10000, seed=20261117):
    """Retain all cases, but form finite ratios only for jointly qualified budgets.

    Inputs are passing_budgets summaries. Match bank seeds before forming ratios,
    then geometrically average within a case and across cases. Bank seeds are
    correlated repetitions, never independent bootstrap units. Recovery is never
    assigned an invented baseline cost. Cluster IDs can bind sharing pairs.
    """
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 100:
        raise ValueError("at least 100 cluster bootstrap repetitions required")
    baseline, adapted = list(baseline), list(adapted)
    def indexed(rows):
        result = {}
        for row in rows:
            key = _case_identity(row)
            if key in result:
                raise ValueError("duplicate per-case budget summary")
            result[key] = row
        return result
    b, a = indexed(baseline), indexed(adapted)
    if set(b) != set(a):
        raise ValueError("baseline/adapted case catalogues must match; missing cases cannot disappear")
    clusters, output = defaultdict(list), []
    for case in sorted(b):
        bm, am = b[case].get("minimum"), a[case].get("minimum")
        unresolved = any(r.get("has_unresolved_reference", False) or r.get("has_missing_records", False) or
                         (r.get("minimum") is None and r.get("all_budgets_conclusively_failed") is False) for r in (b[case], a[case]))
        seed_ratios = None
        if unresolved:
            ratio, outcome = None, "inconclusive"
        elif bm is not None and am is not None:
            bc, ac = _matching_seed_costs(b[case], bm), _matching_seed_costs(a[case], am)
            if bc is None or ac is None or set(bc) != set(ac):
                ratio, outcome = None, "inconclusive"
            else:
                seed_ratios = {seed: ac[seed] / bc[seed] for seed in sorted(bc)}
                log_ratio = float(np.mean(np.log(list(seed_ratios.values()))))
                ratio = float(np.exp(log_ratio))
                cluster = str(cluster_by_setup.get(case[0], case[0])) if cluster_by_setup is not None else case[0]
                clusters[cluster].append(log_ratio)
                outcome = "jointly_qualified"
        else:
            ratio = None
            outcome = "recovered_failure" if bm is None and am is not None else "regression" if bm is not None else "neither_qualified"
        output.append({"setup_id": case[0], "case_id": case[1], "outcome": outcome, "adapted_over_baseline_calls": ratio,
                       "baseline": bm, "adapted": am, "matched_bank_seed_ratios": seed_ratios})
    result = {"cases": output, "recovered_failures": sum(r["outcome"] == "recovered_failure" for r in output),
              "new_failures": sum(r["outcome"] == "regression" for r in output),
              "inconclusive_cases": sum(r["outcome"] == "inconclusive" for r in output), "cluster_count": len(clusters),
              "ratio": None, "ratio_interval_95": None, "status": "inconclusive",
              "bank_generation_costs": _bank_generation_costs(baseline, adapted),
              "ratio_aggregation": "geometric_mean_of_matched_seed_ratios_within_case_then_equal_case_geometric_mean",
              "bank_seed_repetitions_per_case": 2, "resampling_unit": "setup_or_declared_matched_pair"}
    if not clusters:
        return result
    # Resampling units are independent setups/pairs, but matched pairs retain
    # both setups in the point estimator rather than being downweighted.
    sums = np.asarray([np.sum(v) for _, v in sorted(clusters.items())])
    counts = np.asarray([len(v) for _, v in sorted(clusters.items())])
    logs = sums / counts
    result["ratio"] = float(np.exp(sums.sum() / counts.sum()))
    if len(logs) < 2:
        result["reason"] = "fewer_than_two_independent_clusters"
        return result
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_repetitions)
    for start in range(0, bootstrap_repetitions, 1000):
        n = min(1000, bootstrap_repetitions - start)
        indices = rng.integers(0, len(logs), size=(n, len(logs)))
        boot[start:start + n] = np.exp(sums[indices].sum(1) / counts[indices].sum(1))
    interval = np.quantile(boot, [.025, .975]).tolist()
    result["ratio_interval_95"] = interval
    result["status"] = "savings_demonstrated" if interval[1] < 1 and result["new_failures"] == 0 and result["inconclusive_cases"] == 0 else "no_qualified_savings_claim"
    return result
