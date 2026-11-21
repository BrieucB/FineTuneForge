import numpy as np
import pytest

from posteriorforge.study.evaluation import (BANK_SIZES, paired_call_savings, passing_budgets,
    select_checkpoint, select_default_bank, shortlist_checkpoints)

CASES = (("s1", "c0"), ("s2", "c0"))


def records(passes=(256, 1024), cases=CASES):
    return [dict(setup_id=setup, case_id=case, split="validation", bank_seed=seed,
                 context_size=size, forward_calls=size * 3,
                 status="pass" if size in passes else "model_failure")
            for setup, case in cases for seed in (0, 1) for size in BANK_SIZES]


def test_quick_selection_is_validation_only_and_at_most_two():
    rows = [dict(checkpoint_id="b", split="validation", update=20, score=1.),
            dict(checkpoint_id="a", split="validation", update=10, score=1.),
            dict(checkpoint_id="c", split="validation", update=30, score=2.)]
    assert [r["checkpoint_id"] for r in shortlist_checkpoints(rows)] == ["a", "b"]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        shortlist_checkpoints(rows)


def test_default_checks_all_cases_both_seeds_and_nonmonotone_sizes():
    rows = records(passes=(64, 1024))
    result = select_default_bank(rows, cases=CASES)
    assert result["context_size"] == 64
    # Larger-bank failure does not erase a directly tested smaller-bank pass.
    assert 256 not in [b["context_size"] for b in result["cases"][0]["passing_budgets"]]
    for r in rows:
        if r["setup_id"] == "s2" and r["context_size"] == 64 and r["bank_seed"] == 1:
            r["status"] = "model_failure"
    assert select_default_bank(rows, cases=CASES)["context_size"] == 1024
    rows[0]["status"] = "reference_inconclusive"
    assert select_default_bank(rows, cases=CASES)["status"] == "unqualified"


def test_missing_cases_and_duplicate_runs_cannot_create_default():
    rows = records(cases=CASES[:1])
    assert select_default_bank(rows, cases=CASES)["status"] == "unqualified"
    with pytest.raises(ValueError, match="duplicate"):
        passing_budgets(rows + rows[:1])


def test_checkpoint_prefers_no_regression_over_new_easy_cases():
    good_rows, ambitious_rows = records(), records()
    for row in good_rows:
        if row["setup_id"] == "s2": row["status"] = "model_failure"
    for row in ambitious_rows:
        if row["setup_id"] == "s1": row["status"] = "model_failure"
    good = dict(checkpoint_id="conservative", update=20, records=good_rows, worst_normalized_violation=0.)
    ambitious = dict(checkpoint_id="regression", update=10, records=ambitious_rows, worst_normalized_violation=0.)
    result = select_checkpoint([good, ambitious], cases=CASES, baseline_qualified_cases=[CASES[0]])
    assert result["checkpoint_id"] == "conservative"


def summary(setup, count, *, seed_costs=None, generation_costs=None, context_size=256):
    costs = {"0": count, "1": count} if seed_costs is None else {str(k): v for k, v in seed_costs.items()}
    return dict(setup_id=setup, case_id="c0", bank_seeds=[0, 1],
                bank_generation_forward_calls_by_seed={"0": None, "1": None} if generation_costs is None else generation_costs,
                minimum=None if count is None else {"forward_calls": count, "context_size": context_size, "forward_calls_by_seed": costs})


def test_recoveries_have_no_invented_ratio_and_failures_stay_visible():
    base = [summary("s1", 100), summary("s2", None), summary("s3", 100)]
    adapted = [summary("s1", 25), summary("s2", 25), summary("s3", None)]
    result = paired_call_savings(base, adapted, bootstrap_repetitions=100)
    assert result["recovered_failures"] == 1
    assert result["new_failures"] == 1
    assert result["cases"][1]["adapted_over_baseline_calls"] is None
    assert result["status"] != "savings_demonstrated"
    with pytest.raises(ValueError, match="catalogues"):
        paired_call_savings(base, adapted[:-1], bootstrap_repetitions=100)


def test_matched_sharing_pairs_remain_one_bootstrap_cluster():
    base = [summary("a", 100), summary("b", 100), summary("c", 100), summary("d", 100)]
    adapted = [summary("a", 25), summary("b", 25), summary("c", 50), summary("d", 50)]
    result = paired_call_savings(base, adapted, cluster_by_setup={"a": "pair1", "b": "pair1", "c": "pair2", "d": "pair2"}, bootstrap_repetitions=1000)
    assert result["cluster_count"] == 2
    assert result["status"] == "savings_demonstrated"
    assert result["ratio"] == pytest.approx(np.sqrt(.25 * .5))
    assert result["ratio_interval_95"][1] < 1
    assert result == paired_call_savings(base, adapted, cluster_by_setup={"a": "pair1", "b": "pair1", "c": "pair2", "d": "pair2"}, bootstrap_repetitions=1000)


def test_missing_budget_records_and_inconclusive_cases_do_not_disappear():
    rows = records()
    rows.pop()
    assert select_default_bank(rows, cases=CASES)["reason"] == "missing_validation_runs"
    base = passing_budgets(records(passes=()), cases=CASES)
    adapted = passing_budgets(records(passes=(64,)), cases=CASES)
    assert paired_call_savings(base, adapted, bootstrap_repetitions=100)["recovered_failures"] == 2
    uncertain = records(passes=())
    uncertain[0]["status"] = "reference_inconclusive"
    result = paired_call_savings(passing_budgets(uncertain, cases=CASES), adapted, bootstrap_repetitions=100)
    assert result["inconclusive_cases"] == 1
    assert result["recovered_failures"] == 1
    assert result["status"] == "inconclusive"


def test_matched_pairs_do_not_lose_the_weight_of_their_two_setups():
    base = [summary("a", 100), summary("b", 100), summary("c", 100)]
    adapted = [summary("a", 25), summary("b", 25), summary("c", 100)]
    result = paired_call_savings(base, adapted, cluster_by_setup={"a": "pair1", "b": "pair1"}, bootstrap_repetitions=100)
    assert result["ratio"] == pytest.approx(.25 ** (2 / 3))


def selected_replica(seed, *, rows=None, baseline_cases=CASES, checkpoint_prefix="seed"):
    rows = records() if rows is None else rows
    return select_checkpoint([dict(checkpoint_id=f"{checkpoint_prefix}{seed}", training_seed=seed, update=10,
                                   records=rows, worst_normalized_violation=0.)],
                             cases=CASES, baseline_qualified_cases=baseline_cases)


def test_checkpoint_selection_rejects_partial_matrix_and_primary_rank_uses_both_replicas():
    from posteriorforge.study.evaluation import select_primary_rank
    incomplete = dict(checkpoint_id="partial", update=10, records=records()[:-1])
    with pytest.raises(ValueError, match="complete"):
        select_checkpoint([incomplete], cases=CASES)
    selected, second = selected_replica(0), selected_replica(1)
    result = select_primary_rank({4: [selected, second], 16: [selected, second]}, cases=CASES)
    assert result["candidate_rank"] == result["primary_rank"] == 4
    assert result["status"] == "qualified_for_lock"
    assert result["selection_locked"] is False  # Persistence belongs to the coordinator.
    with pytest.raises(ValueError, match="two"):
        select_primary_rank({4: [selected], 16: [selected, second]})
    with pytest.raises(ValueError, match="distinct"):
        select_primary_rank({4: [selected, selected], 16: [selected, second]})
    with pytest.raises(ValueError, match="validation"):
        select_primary_rank({4: [dict(selected, source_split="test"), second], 16: [selected, second]})


def test_inconclusive_reference_blocks_primary_but_keeps_diagnostic_candidate():
    from posteriorforge.study.evaluation import select_primary_rank
    good = [selected_replica(0), selected_replica(1)]
    uncertain = records()
    uncertain[0]["status"] = "reference_inconclusive"
    other = [selected_replica(0, rows=uncertain), selected_replica(1)]
    result = select_primary_rank({4: good, 16: other}, cases=CASES)
    assert result["candidate_rank"] == 4
    assert result["primary_rank"] is None
    assert result["status"] == "diagnostic_only"
    assert any("unresolved_validation_reference" in reason for reason in result["lock_reasons"])
    assert set(result["rank_qualification"]) == {"4", "16"}
    assert result["reporting_policy"] == "retain_all_methods_and_training_replicas"


def test_shared_hard_failures_do_not_impose_the_default_bank_rule_on_rank_selection():
    from posteriorforge.study.evaluation import select_primary_rank
    rows = records()
    for row in rows:
        if row["setup_id"] == "s2":
            row["status"] = "model_failure"
    replicas = [selected_replica(seed, rows=rows, baseline_cases=CASES[:1]) for seed in (0, 1)]
    result = select_primary_rank({4: replicas, 16: replicas}, cases=CASES)
    assert result["primary_rank"] == 4
    assert select_default_bank(rows, cases=CASES)["status"] == "unqualified"
    assert replicas[0]["qualified_cases"] == 1  # The hard second case stays visible.


def test_conclusively_failed_rank_can_be_compared_but_cannot_become_primary():
    from posteriorforge.study.evaluation import select_primary_rank
    failed = [selected_replica(seed, rows=records(passes=())) for seed in (0, 1)]
    no_primary = select_primary_rank({4: failed, 16: failed}, cases=CASES)
    assert no_primary["candidate_rank"] == 4
    assert no_primary["primary_rank"] is None
    good = [selected_replica(0), selected_replica(1)]
    assert select_primary_rank({4: good, 16: failed}, cases=CASES)["primary_rank"] == 4


def test_regression_in_either_chosen_replica_prevents_primary_lock():
    from posteriorforge.study.evaluation import select_primary_rank
    rows = records()
    for row in rows:
        if row["setup_id"] == "s2":
            row["status"] = "model_failure"
    candidate = [selected_replica(0), selected_replica(1, rows=rows)]
    failed = [selected_replica(seed, rows=records(passes=())) for seed in (0, 1)]
    result = select_primary_rank({4: candidate, 16: failed}, cases=CASES)
    assert result["candidate_rank"] == 4
    assert result["primary_rank"] is None
    assert "new_failures_on_baseline_qualified_cases" in result["lock_reasons"]


def test_matching_bank_seed_ratios_precede_geometric_case_aggregation():
    base = [summary(setup, 1000, seed_costs={0: 1000, 1: 400}, context_size=4096) for setup in ("a", "b")]
    adapted = [summary(setup, 300, seed_costs={0: 100, 1: 300}, context_size=1024) for setup in ("a", "b")]
    result = paired_call_savings(base, adapted, bootstrap_repetitions=100)
    expected = np.sqrt((100 / 1000) * (300 / 400))
    assert result["ratio"] == pytest.approx(expected)
    assert result["ratio"] != pytest.approx(300 / 1000)  # Ratio of maxima pairs different seeds.
    assert result["cases"][0]["matched_bank_seed_ratios"] == {"0": .1, "1": .75}
    assert result["cluster_count"] == 2  # The four bank repetitions are not four setups.


def test_missing_matched_seed_costs_are_inconclusive_not_maximum_fallback():
    base = [summary("a", 100), summary("b", 100)]
    adapted = [summary("a", 25), summary("b", 25)]
    adapted[0]["minimum"]["forward_calls_by_seed"].pop("1")
    result = paired_call_savings(base, adapted, bootstrap_repetitions=100)
    assert result["inconclusive_cases"] == 1
    assert result["cases"][0]["matched_bank_seed_ratios"] is None
    assert result["status"] != "savings_demonstrated"


def test_full_bank_generation_is_retained_and_deduplicated_separately_from_prefix_costs():
    cases = (("s1", "c0"), ("s1", "c1"), ("s2", "c0"), ("s2", "c1"))
    baseline, adapted = records(passes=(1024,), cases=cases), records(passes=(256,), cases=cases)
    for row in baseline + adapted:
        multiplier = 10 if row["bank_seed"] == 0 else 20
        row["forward_calls"] = row["context_size"] * multiplier
        row["bank_generation_forward_calls"] = 4096 * multiplier
    b = passing_budgets(baseline, cases=cases)
    a = passing_budgets(adapted, cases=cases)
    assert b[0]["minimum"]["forward_calls_by_seed"] == {"0": 10240, "1": 20480}
    assert len(b[0]["budget_records"]) == 5
    result = paired_call_savings(b, a, bootstrap_repetitions=100)
    assert result["ratio"] == pytest.approx(.25)
    generation = result["bank_generation_costs"]
    assert generation["status"] == "recorded"
    assert generation["total_forward_calls"] == 2 * (40960 + 81920)
    assert len(generation["banks"]) == 4  # Not once per case, method or selected prefix.
    assert generation["included_in_minimum_prefix_ratio"] is False


def test_generation_cost_is_unknown_when_not_explicitly_recorded():
    result = paired_call_savings([summary("a", 100), summary("b", 100)],
                                 [summary("a", 25), summary("b", 25)], bootstrap_repetitions=100)
    assert result["bank_generation_costs"]["status"] == "incomplete"
    assert result["bank_generation_costs"]["total_forward_calls"] is None
