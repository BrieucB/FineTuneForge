"""CPU artifact-driver fixtures; no trained model or qualified reference is run."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from posteriorforge.study import evaluation_campaign as c
from posteriorforge.study.artifacts import save_samples
from posteriorforge.study.banks import prepare_bank, save_bank
from posteriorforge.study.reference import observation_digest
from posteriorforge.study.science import draw_prior
from posteriorforge.study.spec import PriorSpec, ParameterSpec, FamilySpec, DatasetSpec, ProblemSpec

PROTOCOL = "a" * 64


def digest(label):
    return sha256(label.encode()).hexdigest()


def model(version="frozen", *, method="frozen", rank=None, seed=None, update=0):
    return dict(version_id=version, checkpoint_id="checkpoint-" + version,
                model_sha256=digest("weights-" + version), checkpoint_manifest_sha256=digest("manifest-" + version),
                method=method, rank=rank, training_seed=seed, update=update)


def models(*, shortlisted=1):
    result = [model()]
    for method, rank in (("lora", 4), ("lora", 16), ("full", None)):
        for seed in (0, 1):
            for checkpoint in range(shortlisted):
                result.append(model(f"{method}-{rank}-{seed}-{checkpoint}", method=method, rank=rank, seed=seed, update=10 + checkpoint))
    return result


def fake_metadata(split="validation", setups=20):
    cases, banks = [], []
    for setup in range(setups):
        sid = f"{split}-setup-{setup}"
        for i in range(2):
            cases.append(dict(setup_id=sid, case_id=f"case-{i}", split=split,
                              problem_fingerprint=digest(sid), observation_sha256=digest(sid + str(i)),
                              reference_receipt_sha256=None))
        for seed in (0, 1):
            banks.append(dict(setup_id=sid, bank_seed=seed, bank_manifest_sha256=digest(sid + "bank" + str(seed))))
    return cases, banks


def problem(split="validation"):
    return ProblemSpec("toy-" + split, split, (
        ParameterSpec("mean", "population", PriorSpec("normal", 0., 1.)),
        ParameterSpec("spread", "population", PriorSpec("lognormal", .2, .05)),
        ParameterSpec("value", "physical", PriorSpec("normal", "mean", "spread")),
        ParameterSpec("noise", "noise", PriorSpec("lognormal", .1, .03))),
        (FamilySpec("f", "scalar", "noise"),),
        (DatasetSpec("d", "f", "s", {"a": "value"}, (0.,), {}),), {})


@pytest.fixture
def toy(tmp_path):
    spec = problem()
    y = np.array([.25])  # Observed numerical fixture; no hidden query values supplied.
    cases = [dict(setup_id="validation-toy", case_id="case-0", split="validation", problem_fingerprint=spec.fingerprint,
                  observation_sha256=observation_digest(y), reference_receipt_sha256=None)]
    bindings, directories = [], {}
    for seed in (0, 1):
        path = tmp_path / f"bank-{seed}"
        receipt = save_bank(prepare_bank(spec, 4096, seed=seed), path)
        # save_bank's returned checksum is byte-based and anchored in the request.
        checksum = sha256((path / "manifest.json").read_bytes()).hexdigest()
        bindings.append(dict(setup_id="validation-toy", bank_seed=seed, bank_manifest_sha256=checksum))
        directories[seed] = path
    values = draw_prior(spec, 50000, seed=73)
    return dict(spec=spec, y=y, cases=cases, banks=bindings, directories=directories, values=values, root=tmp_path)


def plan(toy, versions=None, contexts=c.BANK_SIZES):
    return c.build_validation_plan(PROTOCOL, [model()] if versions is None else versions, toy["cases"], toy["banks"],
                                   scope="diagnostic", context_sizes=contexts)


def providers(toy, *, posterior=None, reference=None):
    return dict(case_provider=lambda binding: (toy["spec"], toy["y"]),
                bank_provider=lambda binding: toy["directories"][binding["bank_seed"]],
                posterior_provider=(lambda request: None) if posterior is None else posterior,
                reference_provider=(lambda case: None) if reference is None else reference)


def artifact_provider(toy, *, mutate=None):
    def provide(request):
        provenance = dict(model_sha256=request["model"]["model_sha256"], protocol_sha256=PROTOCOL,
                          checkpoint_manifest_sha256=request["model"]["checkpoint_manifest_sha256"],
                          posterior_seed=request["posterior_seed"],
                          bank_manifest_sha256=request["bank"]["bank_manifest_sha256"])
        if mutate is not None:
            mutate(provenance)
        path = toy["root"] / ("samples-" + request["request_id"])
        receipt = save_samples(toy["spec"], toy["values"], path, observation=toy["y"], provenance=provenance,
                               bank_directory=toy["directories"][request["bank"]["bank_seed"]], context_size=request["context_size"])
        return dict(directory=path, manifest_sha256=receipt["manifest_sha256"])
    return provide


def test_study_request_matrix_is_deterministic_and_keeps_every_50000_draw_cell():
    cases, banks = fake_metadata()
    versions = models(shortlisted=2)
    result = c.build_validation_plan(PROTOCOL, versions, cases, banks)
    assert len(result["requests"]) == 5200  # 13 versions × 40 cases × 2 banks × 5 budgets.
    assert {v["posterior_draws"] for v in result["requests"]} == {50000}
    assert len({v["request_id"] for v in result["requests"]}) == 5200
    assert result == c.build_validation_plan(PROTOCOL, list(reversed(versions)), list(reversed(cases)), list(reversed(banks)))
    assert len({v["posterior_seed"] for v in result["requests"]}) == 5200
    with pytest.raises(ValueError, match="two shortlisted"):
        c.build_validation_plan(PROTOCOL, versions + [model("extra", method="lora", rank=4, seed=0, update=99)], cases, banks)
    with pytest.raises(ValueError, match="five"):
        c.build_validation_plan(PROTOCOL, versions, cases, banks, context_sizes=(16,))


def test_actual_artifacts_are_assessed_50000_without_inventing_reference_pass(toy, monkeypatch):
    requested = plan(toy, contexts=(16,))
    called, bank_reads = [], []
    real = c.assess_samples
    def assessed(values, reference, *, spec, observation):
        called.append(values.shape)
        return real(values, reference, spec=spec, observation=observation)
    monkeypatch.setattr(c, "assess_samples", assessed)
    p = providers(toy, posterior=artifact_provider(toy))
    old = p["bank_provider"]
    p["bank_provider"] = lambda binding: bank_reads.append(binding["bank_seed"]) or old(binding)
    out = c.consume_evaluation_plan(requested, **p, output_directory=toy["root"] / "evaluation")
    assert called == [(50000, 4), (50000, 4)]
    assert bank_reads == [0, 1]
    assert {r["status"] for r in out["records"]} == {"reference_inconclusive"}
    assert all(r["forward_calls"] == 16 and r["bank_generation_forward_calls"] == 4096 for r in out["records"])
    assert all(r["gate"]["passed"] is False for r in out["records"])
    assert len(list((toy["root"] / "evaluation").glob("*.json"))) == 4  # Plan, manifest, 2 records.
    assert out["posterior_draws_per_request"] == 50000
    restored_plan, restored = c.load_evaluation_records(toy["root"] / "evaluation", expected_manifest_sha256=out["manifest_sha256"],
                                                       expected_plan_sha256=requested["plan_sha256"])
    assert restored_plan == requested and restored == out["records"]
    record_path = toy["root"] / "evaluation" / (requested["requests"][0]["request_id"] + ".json")
    record_path.write_text(record_path.read_text().replace('reference_inconclusive', 'failed'))
    with pytest.raises(ValueError, match="matrix checksum"):
        c.load_evaluation_records(toy["root"] / "evaluation", expected_manifest_sha256=out["manifest_sha256"],
                                  expected_plan_sha256=requested["plan_sha256"])


def test_mismatched_artifacts_fail_and_missing_or_failed_cells_are_retained(toy):
    requested = plan(toy)
    def provide(request):
        if request["context_size"] == 16:
            raise RuntimeError("fixture backend failed")
        return None
    out = c.consume_evaluation_plan(requested, **providers(toy, posterior=provide))
    assert len(out["records"]) == 10
    assert [r["status"] for r in out["records"]].count("failed") == 2
    assert [r["status"] for r in out["records"]].count("incomplete") == 8
    assert all(r["forward_calls"] == r["context_size"] for r in out["records"])
    mismatch = c.consume_evaluation_plan(plan(toy, contexts=(16,)),
        **providers(toy, posterior=artifact_provider(toy, mutate=lambda p: p.update(model_sha256="0" * 64))))
    assert {r["status"] for r in mismatch["records"]} == {"failed"}
    assert all("binding" in r["error"]["detail"] for r in mismatch["records"])


def test_reference_boolean_cannot_qualify_and_query_bank_receipts_are_verified(toy):
    requested = plan(toy, contexts=(16,))
    requested["cases"][0]["reference_receipt_sha256"] = "b" * 64
    requested = c.build_validation_plan(PROTOCOL, requested["models"], requested["cases"], requested["banks"],
                                        scope="diagnostic", context_sizes=(16,))
    out = c.consume_evaluation_plan(requested, **providers(toy, posterior=artifact_provider(toy),
        reference=lambda case: {"receipt_sha256": "b" * 64, "qualified": True}))
    assert {r["status"] for r in out["records"]} == {"reference_inconclusive"}
    assert all(not r["reference_binding_verified"] for r in out["records"])
    bad = providers(toy)
    bad["case_provider"] = lambda case: (toy["spec"], np.array([9.]))
    out = c.consume_evaluation_plan(plan(toy, contexts=(16,)), **bad)
    assert all(r["error"]["stage"] == "query" for r in out["records"])
    bad = providers(toy)
    bad["bank_provider"] = lambda binding: toy["directories"][1 - binding["bank_seed"]]
    out = c.consume_evaluation_plan(plan(toy), **bad)
    assert all(r["forward_calls"] is None and r["status"] == "failed" for r in out["records"])
    assert c.summarize_validation(plan(toy), out["records"])["status"] == "selection_unavailable"


def diagnostic_selection(toy):
    versions = models()
    requested = plan(toy, versions=versions)
    # A completed attempt can be failed or inconclusive. It is not an absent cell.
    provide = artifact_provider(toy)
    def unavailable(request):
        if request["context_size"] == 16:
            raise RuntimeError("fixture run failed")
        if request["context_size"] == 64:
            return provide(request)
        return None
    out = c.consume_evaluation_plan(requested, **providers(toy, posterior=unavailable))
    return requested, out["records"]


def test_complete_diagnostic_failures_can_freeze_all_seeds_without_qualified_primary_or_default(toy):
    requested, rows = diagnostic_selection(toy)
    summary = c.summarize_validation(requested, rows)
    assert {r["status"] for r in rows} == {"failed", "reference_inconclusive", "incomplete"}
    assert len(summary["selected_models"]) == 7
    assert summary["primary_rank"] is None
    assert all(v["status"] == "unqualified" for v in summary["default_banks"].values())
    test_cases, test_banks = fake_metadata("test", 1)
    path = toy["root"] / "selection.json"
    lock = c.seal_selection(path, validation_plan=requested, records=rows,
                            test_catalogue_sha256=c.case_catalogue_sha256(test_cases))
    document = lock.document
    assert document["diagnostic_only"] is True
    assert document["primary_rank"] is None and document["savings_claim"] is None
    assert document["scientifically_qualified"] is False
    assert document["all_model_versions"] == requested["models"]
    assert len(document["selected_models"]) == 7
    loaded = c.load_selection_lock(path, expected_sha256=lock.sha256, protocol_sha256=PROTOCOL,
                                  test_catalogue_sha256=c.case_catalogue_sha256(test_cases))
    test = c.build_test_plan(PROTOCOL, test_cases, test_banks, selection_lock=loaded)
    assert len(test["requests"]) == 140  # 7 selected versions × 2 toy cases × 2 banks × 5.
    assert {v["posterior_draws"] for v in test["requests"]} == {50000}
    with pytest.raises(FileExistsError):
        c.seal_selection(path, validation_plan=requested, records=rows, test_catalogue_sha256=c.case_catalogue_sha256(test_cases))
    with pytest.raises(ValueError, match="validation-only"):
        c.summarize_validation(test, [])


def test_absent_or_tampered_validation_cells_cannot_be_sealed(toy):
    requested, rows = diagnostic_selection(toy)
    with pytest.raises(ValueError, match="absent"):
        c.seal_selection(toy["root"] / "missing.json", validation_plan=requested, records=rows[:-1], test_catalogue_sha256="b" * 64)
    forged = json.loads(json.dumps(rows))
    forged[0]["status"] = "pass"
    with pytest.raises(ValueError, match="checksum"):
        c.summarize_validation(requested, forged)
    forged[0]["record_sha256"] = c._digest(c._without(forged[0], "record_sha256"))
    with pytest.raises(ValueError, match="not a boolean"):
        c.summarize_validation(requested, forged)


def test_test_gate_runs_before_providers_and_forbids_late_model_seed_or_catalogue_changes(toy):
    requested, rows = diagnostic_selection(toy)
    test_cases, test_banks = fake_metadata("test", 1)
    lock = c.seal_selection(toy["root"] / "lock.json", validation_plan=requested, records=rows,
                            test_catalogue_sha256=c.case_catalogue_sha256(test_cases))
    test = c.build_test_plan(PROTOCOL, test_cases, test_banks, selection_lock=lock)
    calls = []
    def forbidden(value):
        calls.append(value)
        raise AssertionError("test data must remain sealed")
    with pytest.raises(ValueError, match="lock required"):
        c.consume_evaluation_plan(test, case_provider=forbidden, bank_provider=forbidden,
            posterior_provider=forbidden, reference_provider=forbidden, selection_lock=True)
    assert not calls
    with pytest.raises(ValueError, match="seed policy"):
        c.build_test_plan(PROTOCOL, test_cases, test_banks, selection_lock=lock, master_seed=999)
    changed = json.loads(json.dumps(test_cases)); changed[0]["observation_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="catalogue"):
        c.build_test_plan(PROTOCOL, changed, test_banks, selection_lock=lock)
    changed_plan = json.loads(json.dumps(test))
    changed_plan["models"][1]["model_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="checksum"):
        c.consume_evaluation_plan(changed_plan, **providers(toy), selection_lock=lock)
    with pytest.raises(ValueError, match="checksum"):
        c.load_selection_lock(toy["root"] / "lock.json", expected_sha256="0" * 64,
            protocol_sha256=PROTOCOL, test_catalogue_sha256=c.case_catalogue_sha256(test_cases))


def test_test_reporting_preserves_all_replicas_groups_and_fixed_defaults_without_test_selection(toy):
    requested, rows = diagnostic_selection(toy)
    test_cases, test_banks = fake_metadata("test", 1)
    lock = c.seal_selection(toy["root"] / "test-report-lock.json", validation_plan=requested, records=rows,
                            test_catalogue_sha256=c.case_catalogue_sha256(test_cases))
    test = c.build_test_plan(PROTOCOL, test_cases, test_banks, selection_lock=lock)
    # Missing bank artifacts are retained and cannot be given a savings claim.
    out = c.consume_evaluation_plan(test, **providers(toy), selection_lock=lock)
    assert len(out["records"]) == len(test["requests"])
    report = c.summarize_test(test, out["records"], selection_lock=lock, bootstrap_repetitions=100)
    assert report["status"] == "inconclusive" and report["savings_claim"] is None
    assert all(r["forward_calls"] is None for r in out["records"])


def test_request_verification_and_reference_preflight_do_not_expose_or_sample_unqualified_cases(toy):
    requested = plan(toy)
    first = requested["requests"][0]
    bound = c.verify_evaluation_request(requested, first["request_id"])
    assert bound == first
    bound["model"]["model_sha256"] = "0" * 64
    assert requested["requests"][0]["model"]["model_sha256"] != "0" * 64
    with pytest.raises(ValueError, match="not in"):
        c.verify_evaluation_request(requested, "0" * 64)
    calls = []
    out = c.consume_evaluation_plan(requested, **providers(toy, posterior=lambda request: calls.append(request)),
                                    reference_preflight="require_qualified")
    assert not calls
    assert len(out["records"]) == len(requested["requests"])
    assert all(r["status"] == "reference_inconclusive" and r["posterior_execution"] == "skipped_unqualified_reference"
               and r["artifact_draw_count"] == 0 and r["posterior_draws"] == 50000 for r in out["records"])
    assert all(r["forward_calls"] == r["context_size"] and r["bank_generation_forward_calls"] == 4096 for r in out["records"])


def test_complete_test_report_keeps_six_trained_replicas_and_does_not_invent_default_or_savings(toy):
    requested, rows = diagnostic_selection(toy)
    spec = problem("test"); observation = np.array([.45])
    cases = [dict(setup_id="test-toy", case_id="case-0", split="test", problem_fingerprint=spec.fingerprint,
                  observation_sha256=observation_digest(observation), reference_receipt_sha256=None, test_group="heldout_physics")]
    bindings, directories = [], {}
    for seed in (0, 1):
        directory = toy["root"] / f"test-bank-{seed}"
        save_bank(prepare_bank(spec, 4096, seed=seed), directory)
        bindings.append(dict(setup_id="test-toy", bank_seed=seed,
                             bank_manifest_sha256=sha256((directory / "manifest.json").read_bytes()).hexdigest()))
        directories[seed] = directory
    lock = c.seal_selection(toy["root"] / "complete-report-lock.json", validation_plan=requested, records=rows,
                            test_catalogue_sha256=c.case_catalogue_sha256(cases))
    test = c.build_test_plan(PROTOCOL, cases, bindings, selection_lock=lock)
    out = c.consume_evaluation_plan(test, case_provider=lambda case: (spec, observation),
        bank_provider=lambda bank: directories[bank["bank_seed"]], posterior_provider=lambda request: None,
        reference_provider=lambda case: None, selection_lock=lock)
    report = c.summarize_test(test, out["records"], selection_lock=lock, bootstrap_repetitions=100)
    assert report["status"] == "sealed_test_report_complete"
    assert len(report["comparisons"]) == 6 and len(report["case_summaries"]) == 7
    assert report["savings_claim"] is None and report["primary_rank"] is None and report["diagnostic_only"] is True
    assert all(v["status"] == "not_validated" for v in report["fixed_defaults"].values())
    assert all(set(v["groups"]) == {"familiar", "upper_sizes", "sharing", "heldout_physics", "combined"} for v in report["comparisons"].values())
    assert all(v["overall"]["bank_generation_costs"]["total_forward_calls"] == 8192 for v in report["comparisons"].values())
    with pytest.raises(ValueError, match="bootstrap seed policy"):
        c.summarize_test(test, out["records"], selection_lock=lock, bootstrap_repetitions=100, seed=99)
    with pytest.raises(ValueError, match="lock required"):
        c.verify_evaluation_request(test, test["requests"][0]["request_id"])
    assert c.verify_evaluation_request(test, test["requests"][0]["request_id"], selection_lock=lock) == test["requests"][0]


def test_incomplete_artifact_or_wrong_seed_is_never_counted_as_complete_50000(toy):
    requested = plan(toy, contexts=(16,))
    def partial(request):
        save_samples(toy["spec"], toy["values"][:-1], toy["root"] / ("partial-" + request["request_id"]))
    result = c.consume_evaluation_plan(requested, **providers(toy, posterior=partial))
    assert all(r["status"] == "failed" and r["artifact_draw_count"] == 0 for r in result["records"])
    result = c.consume_evaluation_plan(requested, **providers(toy,
        posterior=artifact_provider(toy, mutate=lambda p: p.update(posterior_seed=1))))
    assert all(r["status"] == "failed" and r["artifact_draw_count"] == 0 for r in result["records"])


def replay_component_fixture(toy, *, qualification_change=None):
    """Internally consistent metadata control, NOT an empirical reference/run.

    It tests receipt/inequality replay only. No production result or convergence
    claim is created from these illustrative retained-run descriptors.
    """
    d = toy["spec"].dimension
    units = np.ones(d); directions = c.predefined_directions(d, scales=units)
    query = observation_digest(toy["y"])
    evidence = []
    for i, count in enumerate((500000, 1000000, 800000)):
        evidence.append(dict(method="tempered_smc" if i == 2 else "numpyro_nuts", source_sha256=c.source_digest(),
            problem_fingerprint=toy["spec"].fingerprint, observation_sha256=query,
            sample_sha256=digest("illustrative-run-descriptor-" + str(i)), shape=[count, d], seed=30 + i,
            diagnostics=dict(state="sampled_unqualified", effective_sample_size=300000., max_rhat=1., divergences=0)))
    qualification = dict(state="qualified", contract=c.CONTRACT, precision=.0049,
        moment_relative_precision_limit=.02, projection_seed=20261117, projection_units=units.tolist(),
        retained_refinement_passed=True, independent_passed=True, completed_runs=True,
        second_method_required=True, moment_gate_passed=True,
        moment_precision=dict(mean_error_reference_sd=[.01] * d, sd_error_reference_sd=[.01] * d), evidence=evidence)
    if qualification_change is not None:
        qualification_change(qualification)
    reference = dict(schema=c.REFERENCE_SCHEMA, problem_fingerprint=toy["spec"].fingerprint,
        observation_sha256=query, qualification=qualification, directions=directions,
        queries=np.broadcast_to(np.linspace(-4, 4, 129), (d + 32, 129)).copy(),
        cdfs=np.broadcast_to(np.linspace(0., 1., 129), (d + 32, 129)).copy(),
        mean=np.array([0., .2, 0., .1]), sd=np.full(d, .1))
    reference["receipt_sha256"] = c.reference_receipt(reference)
    cases = json.loads(json.dumps(toy["cases"]))
    cases[0]["reference_receipt_sha256"] = reference["receipt_sha256"]
    requested = c.build_validation_plan(PROTOCOL, [model()], cases, toy["banks"], scope="diagnostic", context_sizes=(16,))
    request = requested["requests"][0]
    moments = c.moment_intervals(reference["mean"], reference["sd"], reference["mean"], reference["sd"],
        qualification["moment_precision"]["mean_error_reference_sd"], qualification["moment_precision"]["sd_error_reference_sd"])
    moments.update(sample_mean=reference["mean"].tolist(), sample_sd=reference["sd"].tolist(),
                   reference_mean=reference["mean"].tolist(), reference_sd=reference["sd"].tolist())
    gate = dict(schema="posteriorforge.study.posterior_gate.v1", status="pass", passed=True,
        problem_fingerprint=toy["spec"].fingerprint, expected_count=50000, reference_qualified=True,
        integrity=dict(passed=True, expected_count=50000, expected_dimension=d, shape=[50000, d], complete_shape=True,
                       finite=True, in_prior_support=True, prior_support_defined=True, nonfinite_row_count=0,
                       out_of_support_row_count=0, clipping_or_repair_performed=False),
        limits=dict(cdf=.1, absolute_bias_reference_sd=.2, sd_ratio=[.8, 1.2], reference_precision=.005, reference_moment_relative_precision=.02),
        reference_qualification=qualification, moments=moments,
        marginal_cdf=dict(lower=[.02] * d, upper=[.04] * d, max_lower=.02, max_upper=.04, passed=True),
        joint_projection_cdf=dict(lower=[.02] * 32, upper=[.04] * 32, max_lower=.02, max_upper=.04, passed=True))
    record = c._record(request)
    record.update(status="pass", artifact_draw_count=50000, gate=gate,
        artifact_manifest_sha256=digest("illustrative-artifact-receipt-only"), reference_binding_verified=True,
        reference_receipt_sha256=reference["receipt_sha256"], reference_certificate=c._reference_certificate(reference))
    banks = {("validation-toy", seed): c.load_bank(toy["directories"][seed]) for seed in (0, 1)}
    record["bank_replica_certificate"] = c._bank_replica_certificate(requested, "validation-toy", banks)
    record.update(c._bank_costs(banks[("validation-toy", 0)], request))
    return requested, request, c._finish_record(record)


def test_replayed_pass_requires_numeric_cdf_bias_sd_and_reference_receipt_evidence(toy):
    requested, request, control = replay_component_fixture(toy)
    c._verify_record(control, request)  # Positive consistency control, not reference qualification evidence.
    for kind in ("marginal_cdf", "joint_projection_cdf", "mean_bias", "sd_ratio", "missing_certificate"):
        altered = json.loads(json.dumps(control))
        if kind in ("marginal_cdf", "joint_projection_cdf"):
            size = len(altered["gate"][kind]["upper"])
            altered["gate"][kind].update(upper=[.9] * size, max_upper=.9, passed=True)
        elif kind in ("mean_bias", "sd_ratio"):
            moments = altered["gate"]["moments"]
            if kind == "mean_bias":
                moments["sample_mean"] = (np.asarray(moments["sample_mean"]) + 1.).tolist()
            else:
                moments["sample_sd"] = (np.asarray(moments["sample_sd"]) * 5.).tolist()
            moments.update(c.moment_intervals(moments["sample_mean"], moments["sample_sd"], moments["reference_mean"],
                moments["reference_sd"], [.01] * 4, [.01] * 4))
            moments["passed"] = True  # Audit counterexample: a flag cannot override bounds.
        else:
            altered["reference_certificate"] = None
        altered = c._finish_record(c._without(altered, "record_sha256"))
        with pytest.raises(ValueError, match="numeric|evidence|bounds"):
            c._verify_record(altered, request)
    # The compact certificate reconstructs the existing receipt exactly.
    original = control["reference_certificate"]["receipt_sha256"]
    changed = json.loads(json.dumps(control))
    changed["reference_certificate"]["qualification"]["precision"] = .0048
    changed["gate"]["reference_qualification"]["precision"] = .0048
    changed = c._finish_record(c._without(changed, "record_sha256"))
    assert original == request["case"]["reference_receipt_sha256"]
    with pytest.raises(ValueError, match="receipt"):
        c._verify_record(changed, request)


@pytest.mark.parametrize("change", [
    lambda q: q.update(precision=.006),
    lambda q: q.update(precision=.001),
    lambda q: q.update(evidence=q["evidence"][:2]),
    lambda q: q["evidence"][-1].update(seed=q["evidence"][-2]["seed"]),
    lambda q: q["evidence"][0]["diagnostics"].update(effective_sample_size=10.),
    lambda q: q["evidence"][1]["diagnostics"].update(max_rhat=1.1),
    lambda q: q["evidence"][1].update(source_sha256="0" * 64),
    lambda q: q["moment_precision"].update(mean_error_reference_sd=[.03] * 4),
])
def test_pass_flags_cannot_override_reference_precision_or_independent_numerical_evidence(toy, change):
    _, request, altered = replay_component_fixture(toy, qualification_change=change)
    with pytest.raises(ValueError, match="reference"):
        c._verify_record(altered, request)


def test_duplicate_bank_receipts_seeds_and_arrays_are_rejected_without_confusing_replica_indices(toy):
    from posteriorforge.study.banks import PairedBank
    duplicate = json.loads(json.dumps(toy["banks"]))
    duplicate[1]["bank_manifest_sha256"] = duplicate[0]["bank_manifest_sha256"]
    with pytest.raises(ValueError, match="different manifest"):
        c.build_validation_plan(PROTOCOL, [model()], toy["cases"], duplicate, scope="diagnostic")
    original = c.load_bank(toy["directories"][0])
    second = c.load_bank(toy["directories"][1])
    for kind in ("same_seed", "same_arrays"):
        metadata = json.loads(json.dumps(second.manifest))
        if kind == "same_seed":
            metadata["seed"] = original.manifest["seed"]
            copied = PairedBank(second.spec, second.theta, second.observations, metadata)
        else:
            copied = PairedBank(original.spec, original.theta, original.observations, metadata)
        directory = toy["root"] / kind
        receipt = save_bank(copied, directory)
        bindings = json.loads(json.dumps(toy["banks"]))
        bindings[1]["bank_manifest_sha256"] = receipt["manifest_sha256"]
        request = c.build_validation_plan(PROTOCOL, [model()], toy["cases"], bindings, scope="diagnostic", context_sizes=(16,))
        source = providers(toy)
        source["bank_provider"] = lambda bank, path=directory: toy["directories"][0] if bank["bank_seed"] == 0 else path
        consumed = c.consume_evaluation_plan(request, **source)
        assert all(row["status"] == "failed" and row["bank_replica_certificate"] is None for row in consumed["records"])
        assert c.summarize_validation(request, consumed["records"])["status"] == "selection_unavailable"
    # Actual generation seeds are 64-bit streams, not replica indices 0/1.
    bindings, directories = [], {}
    for index, actual_seed in enumerate((2**63 + 17, 2**63 + 19)):
        directory = toy["root"] / f"derived-bank-{index}"
        receipt = save_bank(prepare_bank(toy["spec"], 16, seed=actual_seed), directory)
        bindings.append(dict(setup_id="validation-toy", bank_seed=index, bank_manifest_sha256=receipt["manifest_sha256"]))
        directories[index] = directory
    request = c.build_validation_plan(PROTOCOL, [model()], toy["cases"], bindings, scope="diagnostic", context_sizes=(16,))
    source = providers(toy); source["bank_provider"] = lambda bank: directories[bank["bank_seed"]]
    consumed = c.consume_evaluation_plan(request, **source)
    assert all(row["status"] == "incomplete" for row in consumed["records"])
    assert consumed["records"][0]["bank_replica_certificate"]["replicas"][0]["generation_seeds"]["seed"] == 2**63 + 17


def test_study_sharing_metadata_requires_four_two_setup_clusters_in_every_case():
    cases, _ = fake_metadata("test", 40)
    group_names = ("familiar", "upper_sizes", "sharing", "heldout_physics", "combined")
    for case in cases:
        setup_index = int(case["setup_id"].rsplit("-", 1)[1])
        case["test_group"] = group_names[setup_index // 8]
    with pytest.raises(ValueError, match="explicit matched-pair"):
        c._cases(cases, split="test", scope="study")
    for case in cases:
        if case["test_group"] == "sharing":
            setup_index = int(case["setup_id"].rsplit("-", 1)[1]) - 16
            case["cluster_id"] = f"sharing-pair-{setup_index // 2}"
    valid = c._cases(cases, split="test", scope="study")
    sharing = [row for row in valid if row["test_group"] == "sharing"]
    assert len(sharing) == 16 and len({row["cluster_id"] for row in sharing}) == 4
    for pair in {row["cluster_id"] for row in sharing}:
        assert len({row["setup_id"] for row in sharing if row["cluster_id"] == pair}) == 2
        assert len([row for row in sharing if row["cluster_id"] == pair]) == 4
    broken = json.loads(json.dumps(cases))
    for row in broken:
        if row["setup_id"] == "test-setup-16":
            row["cluster_id"] = "extra-independent-cluster"
    with pytest.raises(ValueError, match="exactly two"):
        c._cases(broken, split="test", scope="study")


@pytest.mark.parametrize("status", ["model_failure", "moment_inconclusive", "cdf_inconclusive"])
def test_replay_accepts_supported_scientific_statuses_and_rejects_false_conservative_labels(toy, status):
    _, request, control = replay_component_fixture(toy)
    forged = json.loads(json.dumps(control))
    forged["status"] = status; forged["gate"].update(status=status, passed=False)
    forged = c._finish_record(c._without(forged, "record_sha256"))
    with pytest.raises(ValueError, match="declared assessment status"):
        c._verify_record(forged, request)
    supported = json.loads(json.dumps(control))
    supported["status"] = status; supported["gate"].update(status=status, passed=False)
    if status == "model_failure":
        supported["gate"]["marginal_cdf"].update(lower=[.2] * 4, upper=[.3] * 4, max_lower=.2, max_upper=.3, passed=False)
    elif status == "moment_inconclusive":
        moments = supported["gate"]["moments"]
        moments["sample_mean"][0] += .0195  # .195 referenceSD, straddling .2 once uncertainty is included.
        moments.update(c.moment_intervals(moments["sample_mean"], moments["sample_sd"], moments["reference_mean"],
                                         moments["reference_sd"], [.01] * 4, [.01] * 4))
    else:
        supported["gate"]["marginal_cdf"].update(lower=[.09] * 4, upper=[.11] * 4, max_lower=.09, max_upper=.11, passed=False)
    supported = c._finish_record(c._without(supported, "record_sha256"))
    c._verify_record(supported, request)


def test_replay_cannot_hide_qualified_reference_or_invalid_samples_behind_status_flags(toy):
    _, request, control = replay_component_fixture(toy)
    for status in ("reference_inconclusive", "invalid_samples"):
        altered = json.loads(json.dumps(control))
        altered["status"] = status; altered["gate"].update(status=status, passed=False, reference_qualified=False)
        altered = c._finish_record(c._without(altered, "record_sha256"))
        with pytest.raises(ValueError, match="reference-qualified"):
            c._verify_record(altered, request)
    # A real invalid-array component verdict is preserved without accepting it.
    altered = json.loads(json.dumps(control))
    gate = c.assess_samples(toy["values"][:-1], None, spec=toy["spec"], observation=toy["y"])
    altered.update(status=gate["status"], gate=gate, artifact_draw_count=0, artifact_manifest_sha256=None,
                   reference_certificate=None, reference_binding_verified=False, reference_receipt_sha256=None)
    altered = c._finish_record(c._without(altered, "record_sha256"))
    c._verify_record(altered, request)
    for status in ("failed", "incomplete"):
        altered = json.loads(json.dumps(control)); altered["status"] = status; altered["gate"]["status"] = status
        altered = c._finish_record(c._without(altered, "record_sha256"))
        with pytest.raises(ValueError, match="must not contradict"):
            c._verify_record(altered, request)


def test_cdf_bounds_must_include_precision_twice_when_lower_is_positive(toy):
    _, request, control = replay_component_fixture(toy)
    altered = json.loads(json.dumps(control))
    altered["gate"]["marginal_cdf"].update(lower=[.099] * 4, upper=[.1] * 4, max_lower=.099, max_upper=.1, passed=True)
    altered = c._finish_record(c._without(altered, "record_sha256"))
    # With p=.0049, actual U must be at least .1088; this cannot be a pass.
    with pytest.raises(ValueError, match="inconsistent with reference uncertainty"):
        c._verify_record(altered, request)
    clipped = json.loads(json.dumps(control))
    clipped.update(status="model_failure"); clipped["gate"].update(status="model_failure", passed=False)
    clipped["gate"]["marginal_cdf"].update(lower=[.9951] * 4, upper=[1.] * 4, max_lower=.9951, max_upper=1., passed=False)
    clipped = c._finish_record(c._without(clipped, "record_sha256"))
    c._verify_record(clipped, request)  # Clipping at probability1 remains mathematically valid.
    impossible = json.loads(json.dumps(clipped))
    impossible["gate"]["marginal_cdf"].update(lower=[.999] * 4, max_lower=.999)
    impossible = c._finish_record(c._without(impossible, "record_sha256"))
    with pytest.raises(ValueError, match="inconsistent with reference uncertainty"):
        c._verify_record(impossible, request)  # L+p would exceed the largest possible CDF error1.


def test_real_tiny_unqualified_reference_and_complete_artifacts_remain_inconclusive_on_replay(toy):
    from posteriorforge.study.reference import run_smc_reference, qualify_reference_witnesses
    # Bounded genuine CPU diagnostic, too small to meet numerical precision.
    run = run_smc_reference(toy["spec"], toy["y"], particles=128, moves=2, seed=109, deadline_seconds=3)
    reference = qualify_reference_witnesses(toy["spec"], toy["y"], [run], grid_points=129)
    assert reference["qualification"]["state"] == "inconclusive"
    cases = json.loads(json.dumps(toy["cases"]))
    cases[0]["reference_receipt_sha256"] = reference["receipt_sha256"]
    requested = c.build_validation_plan(PROTOCOL, [model()], cases, toy["banks"], scope="diagnostic", context_sizes=(16,))
    result = c.consume_evaluation_plan(requested, **providers(toy, posterior=artifact_provider(toy), reference=lambda case: reference))
    assert {row["status"] for row in result["records"]} == {"reference_inconclusive"}
    assert all(row["artifact_draw_count"] == 50000 for row in result["records"])
    for request, record in zip(requested["requests"], result["records"]):
        c._verify_record(record, request)


@pytest.mark.parametrize("method,rank,seed", [("frozen", None, None), ("lora", 4, 0), ("lora", 16, 1), ("full", None, 0)])
def test_study_acceptance_requires_actual_canonical_law_and_source_policy(method, rank, seed):
    version = model("guard-control", method=method, rank=rank, seed=seed, update=0 if method == "frozen" else 10)
    provenance = dict(sampling_law=c.STUDY_SAMPLING_LAW, source_policy=c.STUDY_SOURCE_POLICY, training_diagnostic=False)
    c._validate_study_provenance(provenance, version, scope="study")
    for key, value in (("sampling_law", "analytic_test_law.v1"), ("sampling_law", None),
                       ("source_policy", "different_source_policy"), ("source_policy", None)):
        altered = dict(provenance); altered[key] = value
        with pytest.raises(ValueError, match="fixed canonical"):
            c._validate_study_provenance(altered, version, scope="study")
    for key in ("sampling_law", "source_policy"):
        altered = dict(provenance); altered.pop(key)
        with pytest.raises(ValueError, match="fixed canonical"):
            c._validate_study_provenance(altered, version, scope="study")


@pytest.mark.parametrize("flag", [True, 0, None, "missing"])
def test_trained_study_artifact_requires_literal_non_diagnostic_flag(flag):
    version = model("lora-guard", method="lora", rank=4, seed=0, update=10)
    provenance = dict(sampling_law=c.STUDY_SAMPLING_LAW, source_policy=c.STUDY_SOURCE_POLICY)
    if flag != "missing":
        provenance["training_diagnostic"] = flag
    with pytest.raises(ValueError, match="non-diagnostic training"):
        c._validate_study_provenance(provenance, version, scope="study")
    # The frozen baseline has no training process or training diagnostic flag.
    c._validate_study_provenance(dict(sampling_law=c.STUDY_SAMPLING_LAW, source_policy=c.STUDY_SOURCE_POLICY), model(), scope="study")


def test_diagnostic_analytical_artifacts_retain_actual_law_without_becoming_main_study_provenance(toy, monkeypatch):
    requested = plan(toy, contexts=(16,))
    calls = []
    real_guard = c._validate_study_provenance
    def checked(provenance, version, *, scope):
        calls.append((scope, provenance["sampling_law"]))
        return real_guard(provenance, version, scope=scope)
    monkeypatch.setattr(c, "_validate_study_provenance", checked)
    provider = artifact_provider(toy, mutate=lambda provenance: provenance.update(
        sampling_law="analytic_independent_test_law.v1", source_policy="analytic_test_control", training_diagnostic=True))
    result = c.consume_evaluation_plan(requested, **providers(toy, posterior=provider))
    assert calls == [("diagnostic", "analytic_independent_test_law.v1")] * 2
    assert {row["status"] for row in result["records"]} == {"reference_inconclusive"}
    assert all(row["artifact_draw_count"] == 50000 for row in result["records"])
    assert all(row["artifact_provenance"]["sampling_law"] == "analytic_independent_test_law.v1" for row in result["records"])
    with pytest.raises(ValueError, match="fixed canonical"):
        real_guard(result["records"][0]["artifact_provenance"], model(), scope="study")


def test_study_replay_cannot_relabel_actual_analytical_artifact_provenance(toy):
    requested = plan(toy, contexts=(16,))
    provider = artifact_provider(toy, mutate=lambda p: p.update(sampling_law="analytic_test_only.v1", source_policy="analytic_control", training_diagnostic=True))
    result = c.consume_evaluation_plan(requested, **providers(toy, posterior=provider))
    request, original = requested["requests"][0], result["records"][0]
    assert sha256(original["artifact_manifest_json"].encode()).hexdigest() == original["artifact_manifest_sha256"]
    c._validate_record_artifact_provenance(original, request, scope="diagnostic")
    with pytest.raises(ValueError, match="fixed canonical"):
        c._validate_record_artifact_provenance(original, request, scope="study")
    changed = json.loads(json.dumps(original))
    changed["artifact_provenance"].update(sampling_law=c.STUDY_SAMPLING_LAW, source_policy=c.STUDY_SOURCE_POLICY, training_diagnostic=False)
    changed = c._finish_record(c._without(changed, "record_sha256"))
    with pytest.raises(ValueError, match="match the pinned manifest"):
        c._validate_record_artifact_provenance(changed, request, scope="study")
    changed["artifact_manifest_json"] = changed["artifact_manifest_json"].replace("analytic_test_only.v1", c.STUDY_SAMPLING_LAW)
    changed = c._finish_record(c._without(changed, "record_sha256"))
    with pytest.raises(ValueError, match="manifest checksum"):
        c._validate_record_artifact_provenance(changed, request, scope="study")
