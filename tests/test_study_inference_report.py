"""CPU-only report fixtures; no model runs, new simulation banks or posteriors."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report-inference-engineering.py"
spec = importlib.util.spec_from_file_location("inference_report_component_fixture", SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def h(text):
    return sha256(text.encode()).hexdigest()


def values():
    # A constant in-memory schema/rendering fixture, NOT inferred samples.
    return np.broadcast_to(np.array([[0., .2, 0., .1]]), (50000, 4))


def metrics(context=256):
    deployment = {"model_sha256": h("actual-merged"), "origin_model_sha256": h("origin"),
        "checkpoint_manifest_sha256": h("checkpoint"), "deployment_mode": "merged_lora",
        "source_files": {name: h(name) for name in ("inference_deployment.py", "model.py", "model_loading.py", "training.py")}}
    binding = {"count": 50000, "invalid_policy": "fail", "problem": h("problem"), "seed": 5,
        "microbatch_size": 1024, "uniform_law": "uint53-open.v1", "law": "canonical", "source_policy": "source",
        "source_sha256": {name: h(name) for name in ("sampling.py", "science.py", "spec.py", "banks.py", "config.py")},
        "theta": {"shape": [context, 4], "dtype": "<f8", "sha256": h("theta" + str(context))},
        "observations": {"shape": [context, 10], "dtype": "<f8", "sha256": h("obs" + str(context))},
        "query": {"shape": [10], "dtype": "<f8", "sha256": sha256(np.zeros(10, dtype=np.float64).tobytes()).hexdigest()},
        "identities": {"model_sha256": h("actual-merged"), "bank_manifest_sha256": h("bank" + str(context)), "protocol_sha256": h("protocol")},
        "backend": {"model_sha256": h("actual-merged"), "deployment": deployment, "attention": {"policy": "stock"},
            "engineering_source_sha256": h("inference_engineering.py"),
            "source_sha256": {name: h(name) for name in ("model.py", "sampling_backend.py")}}}
    return {"posterior_draws": 50000, "scientifically_qualified": False, "sample_sha256": h("payload"),
        "peak_allocated_bytes": 1_000_000_000, "peak_reserved_bytes": 2_000_000_000,
        "sampling": {"status": "samples_complete_unassessed", "binding": binding,
            "posterior_sampling_forward_evaluations": 0, "wall_seconds": 10.,
            "context_preparation_seconds_this_run": 1., "sampling_seconds_this_run": 8.9}}


def write_run(root, row):
    root.mkdir()
    (root / "samples.npz").write_bytes(b"payload")
    (root / "metrics.json").write_text(json.dumps(row))


def test_load_run_validates_sample_and_metrics_hashes_counts_and_origin(tmp_path, monkeypatch):
    row = metrics(); root = tmp_path / "run"; write_run(root, row)
    monkeypatch.setattr(r, "read_samples", lambda path: values())
    compact, loaded, _ = r.load_run(root, expected_context=256, expected_metrics_sha256=r.file_sha256(root / "metrics.json"))
    assert loaded.shape == (50000, 4) and compact["model_sha256"] != compact["origin_model_sha256"]
    assert compact["metrics_sha256"] == r.file_sha256(root / "metrics.json")
    with pytest.raises(ValueError, match="metrics SHA256"):
        r.load_run(root, expected_context=256, expected_metrics_sha256="0" * 64)
    (root / "samples.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="samples SHA256"):
        r.load_run(root, expected_context=256)


@pytest.mark.parametrize("change", [
    lambda row: row.update(posterior_draws=49999),
    lambda row: row["sampling"].update(status="incomplete"),
    lambda row: row["sampling"]["binding"]["theta"].update(shape=[1024, 4]),
    lambda row: row["sampling"]["binding"]["backend"]["deployment"].update(model_sha256=h("wrong")),
    lambda row: row["sampling"].update(wall_seconds=-1.),
])
def test_malformed_or_partial_requests_are_never_plotted(tmp_path, monkeypatch, change):
    row = metrics(); change(row); root = tmp_path / "run"; write_run(root, row)
    monkeypatch.setattr(r, "read_samples", lambda path: values())
    with pytest.raises(ValueError): r.load_run(root, expected_context=256)


@pytest.mark.parametrize("bad", [np.zeros((49999, 4)), np.zeros((50000, 3)), np.ones((50000, 4), dtype=np.float32),
                                     np.broadcast_to(np.array([[0., -1., 0., .1]]), (50000, 4))])
def test_actual_array_shape_dtype_and_support_are_checked(monkeypatch, bad):
    class Payload:
        files = ("samples",)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key): return bad
    monkeypatch.setattr(r.np, "load", lambda *a, **kw: Payload())
    with pytest.raises(ValueError): r.read_samples("not-opened-fixture.npz")


def test_comparator_delegation_and_retained_metric_hash_match(monkeypatch):
    from posteriorforge.study import inference_engineering
    called = []
    def compare(a, b):
        called.append((a, b))
        return {"original_manifest_sha256": h("a"), "candidate_manifest_sha256": h("b"), "investigation_flag": False}
    monkeypatch.setattr(inference_engineering, "compare_files", compare)
    result = r.paired_comparison({"directory": "a", "metrics_sha256": h("a")}, {"directory": "b", "metrics_sha256": h("b")})
    assert called == [("a", "b")] and not result["investigation_flag"]
    assert len(result["comparison_source_sha256"]) == 64
    with pytest.raises(ValueError, match="metrics SHA256"):
        r.paired_comparison({"directory": "a", "metrics_sha256": h("wrong")}, {"directory": "b", "metrics_sha256": h("b")})


def memory_fixture():
    rows = [{"stage": stage, "queries": q, "status": "completed", "finite": True,
        "peak_allocated_bytes": int(gb * 1e9), "conditional_seconds": seconds}
        for stage, q, gb, seconds in ((74, 1, 12.06, 103.), (0, 1024, 11.11, 94.), (37, 1024, 13.11, 113.), (74, 1024, 15.11, 131.))]
    return {"stock_oom": {"requested_allocation_gib": 129., "recorded_peak_allocated_bytes": 31_763_671_552},
        "tiled_probes": rows, "memory_reduction_ratio": None,
        "large_75d_conditional_proxy": {"estimated_single_gpu_hours_range": [95.8, 134.2], "measured_full_posterior": False}}


def group_fixture(name="frozen", context=256, reps=3, tiled=False, inputs=None):
    mode = "merged_lora" if name == "merged" else "unmerged"
    method = "frozen" if name == "frozen" else "lora_r4"
    row = metrics(context)
    origin = h("frozen") if method == "frozen" else h("origin")
    cp = h("base") if method == "frozen" else h("checkpoint")
    row["sampling"]["binding"]["backend"]["deployment"].update(origin_model_sha256=origin, checkpoint_manifest_sha256=cp, deployment_mode=mode)
    if inputs is not None:
        descriptor = inputs["small_bank" if context == 256 else "control_4096"]
        row["sampling"]["binding"]["identities"]["bank_manifest_sha256"] = descriptor["manifest_sha256"]
    if tiled:
        row["sampling"]["binding"]["backend"]["attention"] = {"policy": r.TILED_POLICY, "max_score_bytes": 536870912,
            "source_sha256": h("cuda_attention.py"), "scientifically_qualified": False,
            "training_path": "stock_unchanged", "attention_instances": 24}
    runs = [{"directory": f"{name}-{context}-{i}", "metrics_sha256": h(f"{name}-{context}-{i}"), "samples_sha256": h(name),
        "binding": deepcopy(row["sampling"]["binding"]), "model_sha256": h(name), "origin_model_sha256": origin,
        "checkpoint_manifest_sha256": cp, "deployment_mode": mode,
        "bank_manifest_sha256": row["sampling"]["binding"]["identities"]["bank_manifest_sha256"],
        "attention": deepcopy(row["sampling"]["binding"]["backend"]["attention"])} for i in range(reps)]
    seconds = [10. + i for i in range(reps)]
    return {"runs": runs, "seconds": seconds, "median_seconds": float(np.median(seconds)), "range_seconds": [min(seconds), max(seconds)],
        "method": method, "mode": mode, "tiled": tiled, "completed_updates": None if method == "frozen" else 2,
        "repetitions": reps, "context_worlds": context}, [values()] * reps


def write_inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs-01").mkdir(exist_ok=True)
    inputs = {"source_commit": "fixture", "frozen_model_sha256": h("frozen"), "base_checkpoint_sha256": h("base"),
        "models": {"lora_r4": {"model_sha256": h("origin"), "checkpoint_manifest_sha256": h("checkpoint")}},
        "attention_score_budget_bytes": 536870912,
        "source_files": {"src/posteriorforge/study/" + name: h(name) for name in
            ("sampling.py", "science.py", "spec.py", "banks.py", "config.py", "model.py", "sampling_backend.py",
             "inference_deployment.py", "model_loading.py", "training.py", "inference_engineering.py", "cuda_attention.py")},
        "large_probe": {"context": {"manifest_sha256": h("large-context")}, "query": {"manifest_sha256": h("large-query")}}}
    for context, key in ((256, "small_bank"), (4096, "control_4096")):
        directory = root / ("bank-" + str(context)); directory.mkdir()
        binding = metrics(context)["sampling"]["binding"]
        value = {"theta": binding["theta"], "observations": binding["observations"], "problem_fingerprint": binding["problem"]}
        (directory / "manifest.json").write_text(json.dumps(value))
        inputs[key] = {"directory": str(directory), "manifest_sha256": r.file_sha256(directory / "manifest.json")}
    np.save(root / "observation.npy", np.zeros(10, dtype=np.float64), allow_pickle=False)
    inputs.update(observation=str(root / "observation.npy"), observation_sha256=r.file_sha256(root / "observation.npy"))
    (root / "inputs-01/manifest.json").write_text(json.dumps(inputs))
    return inputs


def test_partial_collection_keeps_complete_repeats_and_all_requested_comparisons(tmp_path, monkeypatch):
    inputs = write_inputs(tmp_path)
    parity_path = tmp_path / "profile-01/merged"; parity_path.mkdir(parents=True)
    (parity_path / "merge-parity.json").write_text(json.dumps([{"stage": i, "passed": True} for i in range(4)]))
    calls = []
    def load(path, *, repetitions, context):
        path = Path(path); tiled = path.name == "tiled-control"
        name = "frozen" if tiled else path.name
        return group_fixture(name, context, repetitions, tiled, inputs=inputs)
    def compare(a, b):
        calls.append((a, b))
        return {"original_manifest_sha256": h(a), "candidate_manifest_sha256": h(b), "investigation_flag": False}
    monkeypatch.setattr(r, "load_group", load)
    monkeypatch.setattr(r, "compare_run_files", compare)
    monkeypatch.setattr(r, "memory_evidence", lambda root: memory_fixture())
    summary, arrays = r.collect_report(tmp_path, partial=True)
    assert summary["status"] == "partial_c256_report" and len(summary["pending_c4096"]) == 4
    assert len(calls) == 4 and len(summary["relative_distribution_comparisons"]["c256_merged_vs_unmerged"]) == 3
    assert set(arrays["c256"]) == {"frozen", "unmerged", "merged"}
    assert not summary["scientifically_qualified"] and not summary["c4096_speedup_claim"]
    with pytest.raises(ValueError, match="incomplete 4,096"):
        r.collect_report(tmp_path)
    for lane in r.FOUR_K_PATHS.values():
        directory = tmp_path / "context4096-01" / lane; directory.mkdir(parents=True)
        (directory / "posterior-result.json").write_text("{}")
    def load_all(path, *, repetitions, context):
        if context == 256: return load(path, repetitions=repetitions, context=context)
        lane = Path(path).name
        name = "frozen" if "frozen" in lane else "merged" if "merged_tiled" in lane else "unmerged"
        return group_fixture(name, context, repetitions, lane.endswith("tiled"), inputs=inputs)
    monkeypatch.setattr(r, "load_group", load_all)
    summary, arrays = r.collect_report(tmp_path)
    assert summary["status"] == "complete_engineering_report" and not summary["pending_c4096"]
    assert "c4096_frozen_stock_vs_tiled" in summary["relative_distribution_comparisons"]
    assert "c4096_r4_unmerged_vs_merged_tiled" in summary["relative_distribution_comparisons"]
    assert all(v.shape == (50000, 4) for v in arrays["c4096"].values())


def test_renderer_accepts_complete_fixture_metrics_without_claiming_learning(tmp_path):
    pytest.importorskip("matplotlib")
    summary = {"c256": {name: group_fixture(name)[0] for name in ("frozen", "unmerged", "merged")}}
    r.render_latency(summary, tmp_path / "latency.png")
    r.render_posteriors({name: values() for name in ("frozen", "unmerged", "merged")}, tmp_path / "posterior.png", context=256)
    r.render_memory(memory_fixture(), tmp_path / "memory.png")
    for p in tmp_path.glob("*.png"):
        assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and p.stat().st_size > 10000
    assert len(list(tmp_path.glob("*.png"))) == 3
    with pytest.raises(ValueError, match="complete"):
        r.render_posteriors({"frozen": np.zeros((49999, 4))}, tmp_path / "bad.png", context=256)


@pytest.mark.parametrize("change", [
    lambda g: g["runs"][0].update(bank_manifest_sha256=h("different-bank")),
    lambda g: g["runs"][0]["binding"]["theta"].update(sha256=h("different-theta")),
    lambda g: g["runs"][0]["binding"]["query"].update(sha256=h("different-query")),
    lambda g: g["runs"][0]["binding"]["source_sha256"].update(**{"sampling.py": h("wrong-source")}),
    lambda g: g["runs"][0]["binding"]["backend"].update(engineering_source_sha256=h("wrong-driver")),
    lambda g: g["runs"][0]["binding"]["backend"]["deployment"]["source_files"].update(**{"training.py": h("wrong-training")}),
    lambda g: g["runs"][0]["attention"].update(policy="stock"),
    lambda g: g["runs"][0]["attention"].update(max_score_bytes=1024),
    lambda g: g["runs"][0]["attention"].update(source_sha256=h("wrong-attention")),
])
def test_pinned_input_bank_query_source_and_attention_tampering_rejected(tmp_path, change):
    inputs = write_inputs(tmp_path)
    group, _ = group_fixture("frozen", tiled=True, inputs=inputs)
    r.validate_group_identity(group, inputs, frozen=True, tiled=True, merged=False)
    altered = deepcopy(group); change(altered)
    with pytest.raises(ValueError):
        r.validate_group_identity(altered, inputs, frozen=True, tiled=True, merged=False)


def test_stock_attention_and_observation_file_have_explicit_pins(tmp_path):
    inputs = write_inputs(tmp_path)
    group, _ = group_fixture("frozen", inputs=inputs)
    r.validate_group_identity(group, inputs, frozen=True, tiled=False, merged=False)
    group["runs"][0]["attention"] = {"policy": "stock", "max_score_bytes": 512}
    with pytest.raises(ValueError, match="stock attention"):
        r.validate_group_identity(group, inputs, frozen=True, tiled=False, merged=False)
    group, _ = group_fixture("frozen", inputs=inputs)
    Path(inputs["observation"]).write_bytes(b"tampered observation")
    with pytest.raises(ValueError, match="observation FILE"):
        r.validate_group_identity(group, inputs, frozen=True, tiled=False, merged=False)


def write_memory_receipts(root, inputs, monkeypatch):
    probes = root / "profile-01/probes"; probes.mkdir(parents=True)
    rows = memory_fixture()["tiled_probes"]
    forecast = {"posterior_draws": 50000, "stages": 75, "microbatches": 49, "full_75d_run_enabled": False,
        "estimated_sampler_seconds_range": [75 * 49 * 94., 75 * 49 * 131.]}
    fitting = {"passed": True, "cdf_max_abs": 0.}
    for name, row in zip(("probe-74-1.json", "probe-0-1024.json", "probe-37-1024.json", "probe-74-1024.json"), rows):
        (probes / name).write_text(json.dumps(row))
    (probes / "large-forecast.json").write_text(json.dumps(forecast))
    (probes / "fitting-stock-tiled-parity.json").write_text(json.dumps(fitting))
    parent = {"status": "completed", "lane": "probes", "scientifically_qualified": False, "training_pilots_started": False,
        "source_commit": inputs["source_commit"], "input_manifest_sha256": r.file_sha256(root / "inputs-01/manifest.json"),
        "checks": {"rows": rows, "forecast": forecast, "fitting_parity": fitting}}
    (probes / "result.json").write_text(json.dumps(parent))
    old = {"status": "out_of_memory", "error": "CUDA out of memory. Tried to allocate 129.00 GiB.",
        "peak_allocated_bytes": 31_763_671_552,
        "probe": {"context": {"dimension": 75, "observations": 200, "worlds": 4096, "manifest_sha256": h("large-context")},
                  "query": {"manifest_sha256": h("large-query")}}}
    original_json, original_sha = r._json, r.file_sha256
    monkeypatch.setattr(r, "_json", lambda p: deepcopy(old) if Path(p).name == "memory-probe-2.json" else original_json(p))
    monkeypatch.setattr(r, "file_sha256", lambda p: h("old-oom") if Path(p).name == "memory-probe-2.json" else original_sha(p))
    return probes


@pytest.mark.parametrize("target,key,value", [
    ("result.json", "input_manifest_sha256", "0" * 64),
    ("result.json", "source_commit", "different-source"),
    ("result.json", "status", "failed"),
    ("probe-74-1.json", "peak_allocated_bytes", 1),
    ("large-forecast.json", "full_75d_run_enabled", True),
    ("fitting-stock-tiled-parity.json", "passed", False),
])
def test_memory_receipts_cannot_detach_from_completed_parent_input_and_source(tmp_path, monkeypatch, target, key, value):
    inputs = write_inputs(tmp_path)
    probes = write_memory_receipts(tmp_path, inputs, monkeypatch)
    control = r.memory_evidence(tmp_path)
    assert control["memory_reduction_ratio"] is None
    assert control["input_manifest_sha256"] == r.file_sha256(tmp_path / "inputs-01/manifest.json")
    assert control["producing_source_commit"] == inputs["source_commit"]
    path = probes / target
    altered = json.loads(path.read_text()); altered[key] = value; path.write_text(json.dumps(altered))
    with pytest.raises(ValueError): r.memory_evidence(tmp_path)
