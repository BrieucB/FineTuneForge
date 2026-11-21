"""CPU-only receipt, integrity, and rendering tests for the training report."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from posteriorforge.study.banks import PairedBank, array_descriptor, save_bank
from posteriorforge.study.campaign import CampaignQualificationError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report-training-engineering.py"
spec = importlib.util.spec_from_file_location("training_engineering_report", SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def h(value):
    return sha256(value.encode()).hexdigest()


def describe(path):
    return {"path": str(path), "sha256": r.file_hash(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return describe(path)


def training_fixture(tmp_path):
    checkpoint = tmp_path / "base.ckpt"
    checkpoint.write_text("test weights")
    # A fixture bank of constant supported rows, with no simulator invocation.
    problem = r.tiny_problem(posterior=True)
    bank_root = tmp_path / "bank"
    saved = save_bank(PairedBank(problem, np.broadcast_to([0., .2, 0., .1], (256, 4)),
                                np.zeros((256, 10))), bank_root)
    observation = tmp_path / "observation.npy"
    np.save(observation, np.zeros(10, dtype=np.float64), allow_pickle=False)
    small_bank = {"directory": str(bank_root), "manifest_sha256": saved["manifest_sha256"],
                  "worlds": 256, "dimension": 4, "observations": 10, "problem": problem.fingerprint}
    episodes = [{"position": i, "context": {"worlds": 16, "dimension": 4, "observations": 1},
                 "query": {"worlds": 1, "dimension": 4, "observations": 1}} for i in range(32)]
    return {"schema": "finetuneforge.cuda_qualification_inputs.v1", "base_checkpoint": str(checkpoint),
            "base_checkpoint_sha256": r.file_hash(checkpoint), "frozen_model_sha256": h("frozen"),
            "training_effective_batch": 16, "posterior_draws": r.COUNT, "engineering_updates": 2,
            "protocol_sha256": h("protocol"), "observation_sha256": r.file_hash(observation),
            "source_commit": h("training-predecessor")[:40], "episodes": episodes,
            "posterior_bank": {**small_bank, "directory": "bank"}}


def make_trace(path):
    events = [{"cat": "kernel", "dur": 6.}, {"cat": "kernel", "dur": 5.},
              {"cat": "gpu_user_annotation", "dur": 9_000.},
              {"cat": "cpu_op", "name": "aten::mm", "dur": 40.},
              {"cat": "cpu_op", "name": "aten::bmm", "dur": 20.},
              {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 30.},
              *[{"cat": "user_annotation", "name": f"engineering::{name}", "dur": 100.} for name in r.SCOPES]]
    trace = write(path, {"traceEvents": events})
    row = {"episode_position": 0, "stage_index": 0, "parameter_name": "mu", "trace": path.name,
           "trace_sha256": trace["sha256"], "cuda_kernel_events": {"count": 2, "duration_us": 11.},
           "operator_summary": [{"name": name, "count": 1, "cpu_total_us": 100.}
                                for name in (*r.PROFILE_NAMES, *(f"engineering::{name}" for name in r.SCOPES))],
           "profiled_preparation_and_conditional_wall_seconds": .01}
    return trace, row


def updates(policy):
    return [{"update": i, "data_position": i * 16, "episode_count": 16, "query_case_count": 16,
             "conditional_count": 64, "backward_calls": 64, "optimizer_steps": 1,
             "neural_precision": "float32", "probability_precision": "float64", "sampling_law": r.LAW,
             "mixed_precision": False, "preprocessing_includes_host_to_device": True,
             "scientifically_qualified": False, "parametrization_cache_policy": policy,
             "preprocessing_seconds": 1., "update_seconds": 3., "loss": -.9 * i,
             "gradient_norm_before_clip": .1} for i in (1, 2)]


def training_receipt(training, input_sha, method, *, cached):
    return {"spec": {"method": method, "replica": 0, "phase": "pilot", "updates": 2,
                     "master_seed": 20261117, "effective_batch": 16, "diagnostic": True,
                     "protocol_sha256": training["protocol_sha256"],
                     "learning_rate": 1e-5 if method == "full_finetune" else 1e-4},
            "status": "completed", "scientifically_qualified": False, "input_manifest_sha256": input_sha,
            "base_checkpoint_sha256": training["base_checkpoint_sha256"], "posterior_requests": 0,
            "posterior_draws_contract": r.COUNT, "source_commit": h("source")[:40],
            "hardware": {"name": "NVIDIA A100-SXM4-40GB", "memory_bytes": 40 * 1024**3,
                         "uuid": h(str(cached))[:40], "cuda_runtime": "12.4"},
            "peak_allocated_bytes": 2, "peak_reserved_bytes": 3,
            "unprofiled_update_wall_seconds": [5., 4.5],
            "updates": updates(r.CACHE_POLICY if cached else "stock_uncached")}


def write_checkpoint(path, receipt, training, input_sha, *, cached):
    rank = {"lora_r4": 4, "lora_r16": 16, "full_finetune": None}[receipt["spec"]["method"]]
    model = h(str(path))
    identity = {"spec": receipt["spec"], "stream_id": r.digest(training["episodes"]),
                "protocol_sha256": training["protocol_sha256"], "qualification_inputs_sha256": input_sha,
                "source_commit": receipt["source_commit"], "cache_parametrizations": cached}
    meta = {"schema": "finetuneforge.campaign_checkpoint.v1", "identity": identity,
            "state": {"update": 2, "data_position": 32, "stream_id": identity["stream_id"]},
            "weights_sha256": model, "scientifically_qualified": False,
            "history": [{"update": u["update"], "loss": u["loss"]} for u in receipt["updates"]],
            "deployment": {"campaign_method": receipt["spec"]["method"], "rank": rank, "weights_sha256": model,
                           "base_checkpoint_sha256": training["base_checkpoint_sha256"], "sampling_law": r.LAW,
                           "scientifically_qualified": False,
                           "format": "peft_adapter" if rank else "full_tabpfn_safetensors"}}
    metadata = write(path / "campaign.json", meta)
    weights = path / "weights.fixture"
    weights.write_bytes(b"immutable checkpoint fixture")
    manifest = write(path / "manifest.json", {"schema": "finetuneforge.campaign_checkpoint.files.v1",
                     "files": {"campaign.json": metadata["sha256"], weights.name: r.file_hash(weights)}})
    receipt.update(checkpoint_manifest_sha256=manifest["sha256"], final_model_sha256=model)
    binding = {"campaign_checkpoint": str(path), "checkpoint_manifest_sha256": manifest["sha256"],
               "campaign_metadata_sha256": metadata["sha256"], "model_sha256": model}
    return manifest, binding


def parity():
    comparison = {"passed": True, **r.TOLERANCES, "missing": [], "unexpected": [], "failed": [],
                  "tensor_count": 1, "tensors": {"weight": {"passed": True}}, "max_absolute_error": 0.}
    return [{"episode_position": 0, "stage_index": 0, "parameter_name": "mu", "members": 8,
             "passed": True, "logits_density_loss_bitwise_equal": True, "gradient_comparison": deepcopy(comparison)}
            for _ in range(2)]


def make_training_layer(root, manifest, training):
    input_sha = manifest["training_inputs"]["sha256"]
    old = training_receipt(training, input_sha, "lora_r4", cached=False)
    old.update(schema="finetuneforge.training_engineering.v1", mode="baseline", job_id="1",
               original_source_commit=training["source_commit"])
    trace, record = make_trace(root / "baseline/trace.json")
    cat, _, _ = r.trace_metrics(trace)
    manifest["baseline"] = {"result": write(root / "baseline/update.json", old), "trace": trace,
                            "profile": write(root / "baseline/profile.json", {**old, "mode": "profile", "conditional_profile": record}),
                            "profile_categories": write(root / "baseline/audit.json", {"trace_sha256": trace["sha256"],
                                "source_commit": old["source_commit"], "categories": cat})}
    common = {"source_commit": old["source_commit"], "hostname": "test-node", "job_id": "2"}
    allocation = {**common, "schema": "finetuneforge.training_engineering_qualification_allocation.v1",
                  "input_manifest_sha256": input_sha, "worker_returncodes": dict.fromkeys(r.METHODS, 0),
                  "scientifically_qualified": False, "posterior_requests": 0}
    lanes, checkpoints = {}, {}
    for lane, method in r.METHODS.items():
        cached = lane != "stock_r4"
        folder = root / lane
        row = training_receipt(training, input_sha, method, cached=cached)
        row.update(common, schema=r.QUALIFICATION_SCHEMA, lane=lane)
        checkpoint, binding = write_checkpoint(folder / "after-two-update", row, training, input_sha, cached=cached)
        descriptors = {"checkpoint_manifest": checkpoint}
        if cached:
            row.update(restart=dict.fromkeys(r.RESTART_CHECKS, True), parity=parity())
        if lane in r.R4_LANES:
            descriptors["trace"], row["conditional_profile"] = make_trace(folder / "trace.json")
        descriptors["result"] = write(folder / "result.json", row)
        lanes[lane], checkpoints[lane] = descriptors, binding
    manifest["qualification"] = {"allocation": write(root / "allocation.json", allocation), "lanes": lanes}
    return checkpoints


def test_native_trace_counts_exclude_annotations_and_verify_receipt(tmp_path):
    trace, record = make_trace(tmp_path / "trace.json")
    summary = r.profile({"conditional_profile": record}, trace)
    assert summary["cuda_kernel_count"] == 2
    assert summary["sum_cuda_kernel_elapsed_seconds"] == 11e-6
    record["cuda_kernel_events"]["duration_us"] += 9_000
    with pytest.raises(ValueError, match="kernel totals"):
        r.profile({"conditional_profile": record}, trace)
    Path(trace["path"]).write_text("changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        r.profile({"conditional_profile": record}, trace)


def test_checkpoint_binding_authenticates_payload_without_model_loading(tmp_path):
    training = training_fixture(tmp_path)
    input_sha = h("inputs")
    receipt = training_receipt(training, input_sha, "lora_r4", cached=True)
    checkpoint = tmp_path / "after-two-update"
    descriptor, wanted = write_checkpoint(checkpoint, receipt, training, input_sha, cached=True)
    assert r.checkpoint_summary(descriptor, receipt, training, input_sha, cached=True) == wanted
    with pytest.raises(CampaignQualificationError, match="identity mismatch"):
        r.checkpoint_summary(descriptor, receipt, training, input_sha, cached=False)
    (checkpoint / "weights.fixture").write_bytes(b"tampered")
    with pytest.raises(CampaignQualificationError, match="checksum"):
        r.checkpoint_summary(descriptor, receipt, training, input_sha, cached=True)


def test_parity_requires_original_tolerance_complete_gradients_and_eight_members():
    initial_and_post_one = parity()
    assert [item["phase"] for item in r.parity_summary(initial_and_post_one)] == ["initial", "after_update_1"]
    for change in (lambda row: row.update(members=7), lambda row: row["gradient_comparison"].update(atol=2e-6),
                   lambda row: row["gradient_comparison"]["tensors"]["weight"].update(passed=False)):
        mutated = deepcopy(initial_and_post_one)
        change(mutated[1])
        with pytest.raises(ValueError, match="conditional parity"):
            r.parity_summary(mutated)


def make_posterior_layer(root, training, by_variant, *, rank=4, source_commit=None):
    analytic = r.load_helper("fixture_analytic_posterior", "check-study-orchestration-cpu.py")
    trial_source = source_commit or h("source")[:40]
    bank_dir = root / training["posterior_bank"]["directory"]
    bank_json = json.loads((bank_dir / "manifest.json").read_bytes())
    bank = {**training["posterior_bank"], "directory": str(bank_dir)}
    observation = root / "observation.npy"
    settings = {k: training[k] for k in ("base_checkpoint", "base_checkpoint_sha256", "protocol_sha256",
                                         "frozen_model_sha256", "observation_sha256")}
    settings.update(observation=str(observation), small_bank=bank)
    template = write(root / "template.json", settings)
    core = ("sampling.py", "science.py", "spec.py", "banks.py", "config.py")
    backend = ("model.py", "sampling_backend.py")
    deploy = ("inference_deployment.py", "model.py", "model_loading.py", "training.py")
    core_files = "src/posteriorforge/study/"
    sources = {name: r.file_hash(r.REPO / core_files / name)
               for name in (*core, *backend, *deploy, "inference_engineering.py")}
    inputs = {**settings, "schema": r.POSTERIOR_SCHEMA, "rank": rank, "seed": 20265117, "microbatch_size": 1024,
              "mode": "merged_lora", "posterior_draws": r.COUNT, "timing_repetitions": 1,
              "attention_policy": "stock", "training_pilots_enabled": False, "new_forward_evaluations": 0,
              "scientifically_qualified": False, "source_commit": trial_source, "template_path": template["path"],
              "template_sha256": template["sha256"], "checkpoints": by_variant,
              "source_files": {core_files + name: digest for name, digest in sources.items()}}
    input_descriptor = write(root / "posterior/inputs.json", inputs)
    common = {"schema": r.POSTERIOR_SCHEMA, "scientifically_qualified": False, "source_commit": trial_source,
              "input_manifest_sha256": input_descriptor["sha256"]}
    allocation = dict(common, worker_returncodes=dict.fromkeys(r.POSTERIOR_LANES, 0), posterior_draws_per_request=r.COUNT)
    result = {"inputs": input_descriptor, "allocation": write(root / "posterior/allocation.json", allocation), "lanes": {}}
    u = np.linspace(.01, 1., r.COUNT)
    # Complete synthetic arrays for checksum and plotting tests, with distinct trained origins.
    for variant, checkpoint in by_variant.items():
        samples = np.column_stack((u - .5 + (1e-6 if variant == "cached" else 0.), .2 + u * .05,
                                   u - .4, .1 + u * .03))
        folder = root / "posterior" / variant
        rep = folder / "repetition-0"
        rep.mkdir(parents=True)
        np.savez(rep / "samples.npz", samples=samples)
        merged_model = h("merged " + variant)
        deployment = {"origin_model_sha256": checkpoint["model_sha256"], "model_sha256": merged_model,
                      "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
                      "deployment_mode": "merged_lora", "source_files": {name: sources[name] for name in deploy}}
        identities = {"model_sha256": merged_model, "bank_manifest_sha256": bank["manifest_sha256"],
                      "protocol_sha256": training["protocol_sha256"]}
        binding = {"count": r.COUNT, "invalid_policy": "fail", "problem": bank["problem"], "seed": inputs["seed"],
                   "microbatch_size": inputs["microbatch_size"], "uniform_law": "uint53-open.v1", "law": r.LAW,
                   "source_policy": "cancelled_zero_width", "source_sha256": {name: sources[name] for name in core},
                   "theta": bank_json["theta"], "observations": bank_json["observations"],
                   "query": array_descriptor(np.zeros(10)), "identities": identities,
                   "backend": {"model_sha256": merged_model, "attention": {"policy": "stock"},
                               "deployment": deployment, "source_sha256": {name: sources[name] for name in backend},
                               "engineering_source_sha256": sources["inference_engineering.py"]}}
        sampling = {"status": "samples_complete_unassessed", "binding": binding, "posterior_sampling_forward_evaluations": 0,
                    "wall_seconds": 10., "context_preparation_seconds_this_run": .2, "sampling_seconds_this_run": 9.8}
        row = {"posterior_draws": r.COUNT, "sampling": sampling, "sample_sha256": r.file_hash(rep / "samples.npz"),
               "peak_allocated_bytes": 8, "peak_reserved_bytes": 9, "scientifically_qualified": False,
               "analytical_diagnostics": analytic.analyze_samples(samples)}
        metrics = write(rep / "metrics.json", row)
        container = write(folder / "posterior-result.json", {"method": f"lora_r{rank}", "mode": "merged_lora", "tiled": False,
            "context": bank, "deployment": {**deployment, "completed_updates": 2}, "scientifically_qualified": False,
            "runs": [row], "seconds": [10.], "median_seconds": 10., "range_seconds": [10., 10.]})
        receipt = write(folder / "result.json", {**common, "variant": variant, "status": "completed", "job_id": "3",
                         "checkpoint": checkpoint, "posterior_draws": r.COUNT, "posterior_result_sha256": container["sha256"]})
        result["lanes"][variant] = {"result": receipt, "container": container, "metrics": metrics}
    return result


def report_fixture(tmp_path):
    training = training_fixture(tmp_path)
    manifest = {"schema": r.INPUT_SCHEMA, "training_inputs": write(tmp_path / "train.json", training)}
    checkpoints = make_training_layer(tmp_path / "train", manifest, training)
    origins = {variant: checkpoints[lane] for variant, lane in r.POSTERIOR_LANES.items()}
    manifest["posterior"] = make_posterior_layer(tmp_path, training, origins)
    audit = tmp_path / "cpu.xml"
    audit.write_text('<testsuite tests="3" errors="0" failures="0" skipped="0"/>')
    rejected = tmp_path / "failed.xml"
    rejected.write_text('<testsuite tests="2" errors="0" failures="2" skipped="0"><failure>CheckpointError</failure></testsuite>')
    manifest["audits"] = {"cpu": describe(audit), "rejected_cache": describe(rejected)}
    return manifest


def test_report_preserves_two_batches_distinct_origins_all_vectors_and_figures(tmp_path):
    manifest = report_fixture(tmp_path)
    inputs = write(tmp_path / "report-inputs.json", manifest)
    output = tmp_path / "figures"
    report = json.loads(r.build_report(inputs["path"], inputs["sha256"], output).read_bytes())
    assert not report["scientifically_qualified"]
    summary = report["training"]
    assert [row["batch"] for row in summary["stock_cached_r4_comparison_by_batch"]] == [1, 2]
    assert summary["lanes"]["cache_r4"]["restart"]["passed"]
    assert report["audits"]["cpu"]["tests"] == 3
    assert report["audits"]["rejected_cache"]["failures"] == 2
    posterior = report["posterior"]
    a, b = (posterior["runs"][lane]["request"] for lane in ("stock", "cached"))
    assert a["origin_model_sha256"] != b["origin_model_sha256"]
    assert a["posterior_draws"] == b["posterior_draws"] == r.COUNT
    assert len(posterior["comparison"]["axis_and32projection_cdf"]) == 36
    assert not posterior["runs"]["stock"]["analytic_control"]["control_checks_passed"]
    assert not posterior["runs"]["stock"]["exact_marginal_study_thresholds"]["full_study_matrix_run"]
    assert set(report["figures"]) == {"training-latency.png", "training-profile.png", "training-posteriors.png"}
    assert all(r.file_hash(output / name) == digest for name, digest in report["figures"].items())


def test_report_rejects_swapped_trained_origin_even_when_receipt_rehashed(tmp_path):
    manifest = report_fixture(tmp_path)
    lanes = manifest["posterior"]["lanes"]
    stock_path = Path(lanes["stock"]["result"]["path"])
    stock = json.loads(stock_path.read_bytes())
    stock["checkpoint"] = json.loads(Path(lanes["cached"]["result"]["path"]).read_bytes())["checkpoint"]
    lanes["stock"]["result"] = write(stock_path, stock)
    inputs = write(tmp_path / "report-inputs.json", manifest)
    with pytest.raises(ValueError, match="its authenticated training checkpoint"):
        r.collect_report(inputs["path"], inputs["sha256"])


def test_report_rejects_modified_samples_against_retained_metrics(tmp_path):
    manifest = report_fixture(tmp_path)
    metrics = Path(manifest["posterior"]["lanes"]["cached"]["metrics"]["path"])
    np.savez(metrics.with_name("samples.npz"), samples=np.ones((r.COUNT, 4)))
    inputs = write(tmp_path / "report-inputs.json", manifest)
    with pytest.raises(ValueError, match="samples SHA256 mismatch"):
        r.collect_report(inputs["path"], inputs["sha256"])


def test_separate_rank16_report_keeps_failed_weight_check_and_older_source(tmp_path):
    training = training_fixture(tmp_path)
    training_input = write(tmp_path / "training.json", training)
    input_sha = training_input["sha256"]
    origins = {}
    for side in ("stock", "cached"):
        cached = side == "cached"
        row = training_receipt(training, input_sha, "lora_r16", cached=cached)
        _, origins[side] = write_checkpoint(tmp_path / side / "after-two-update", row, training, input_sha, cached=cached)
    source_commit = h("source")[:40]
    failed_rank16 = {"reference_kind": "authenticated_historical_stock_same_data",
                     "reference_checkpoint": origins["stock"], "candidate_checkpoint": origins["cached"],
                     "model": {"passed": False}, "optimizer": {"passed": True}, "passed": False}
    earlier = {"schema": "finetuneforge.training_engineering_report.v1", "scientifically_qualified": False,
               "evidence": {"training_inputs": training_input}, "posterior": {"source_commit": source_commit},
               "final_state_comparison": {"source_commit": source_commit, **r.TOLERANCES,
                                          "artifact": training_input, "cases": {"cache_r16": failed_rank16}},
               "training": {"source_commit": source_commit, "lanes": {"cache_r16": {
                   "checkpoint": origins["cached"], "restart": dict.fromkeys(r.RESTART_CHECKS, True)}}}}
    first_report = write(tmp_path / "rank4/report.json", earlier)
    r16_source = h("rank-16 posterior worker")[:40]
    posterior = make_posterior_layer(tmp_path, training, origins, rank=16, source_commit=r16_source)
    inputs = {"schema": r.RANK16_INPUT_SCHEMA, "rank4_report": first_report, "posterior": posterior}
    manifest = write(tmp_path / "rank16-inputs.json", inputs)
    output = tmp_path / "rank16-report"
    report = json.loads(r.build_report(manifest["path"], manifest["sha256"], output, rank16=True).read_bytes())
    assert report["earlier_rank4_posterior_source_commit"] == source_commit
    assert report["posterior"]["source_commit"] == r16_source
    assert report["posterior"]["rank"] == 16
    assert not report["final_state_comparison"]["model"]["passed"]
    assert report["cached_restart"]["passed"] and not report["posterior"]["comparison"]["investigation_flag"]
    assert set(report["figures"]) == {"rank16-posteriors.png"}
    assert r.file_hash(first_report["path"]) == first_report["sha256"]
    receipt = Path(posterior["lanes"]["cached"]["result"]["path"])
    row = json.loads(receipt.read_bytes())
    row["source_commit"] = source_commit
    posterior["lanes"]["cached"]["result"] = write(receipt, row)
    manifest = write(tmp_path / "rank16-inputs.json", inputs)
    with pytest.raises(ValueError, match="its authenticated training checkpoint"):
        r.collect_rank16_report(manifest["path"], manifest["sha256"])
