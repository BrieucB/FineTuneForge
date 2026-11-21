#!/usr/bin/env python3
"""Build CPU-only evidence for the two-update training execution comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from posteriorforge.study.campaign import read_campaign_checkpoint
from posteriorforge.study.cuda_qualification import tiny_problem
from posteriorforge.study.episodes import digest
from posteriorforge.study.metrics import BIAS_LIMIT, CDF_LIMIT, SD_LIMITS
from posteriorforge.study import inference_engineering

INPUT_SCHEMA = "finetuneforge.training_engineering_report_inputs.v1"
RANK16_INPUT_SCHEMA = "finetuneforge.training_rank16_posterior_report_inputs.v1"
POSTERIOR_SCHEMA = "finetuneforge.training_posterior_qualification.v1"
QUALIFICATION_SCHEMA = "finetuneforge.training_engineering_qualification.v1"
CACHE_POLICY = "per_layer_forward_and_checkpoint_recompute.v1"
LAW = "canonical_half_normal_full_support.v1"
COUNT = 50_000
METHODS = {"stock_r4": "lora_r4", "cache_r4": "lora_r4", "cache_r16": "lora_r16",
           "full_finetune": "full_finetune"}
POSTERIOR_LANES = {"stock": "stock_r4", "cached": "cache_r4"}
R4_LANES = ("stock_r4", "cache_r4")
PROFILE_NAMES = ("aten::mm", "aten::bmm", "cudaLaunchKernel")
SCOPES = ("preprocess_and_h2d", "network_forward", "density_including_network",
          "density_log_prob", "conditional_backward")
RESTART_CHECKS = ("passed", "weights_bitwise_equal", "optimizer_bitwise_equal",
                  "future_all_rngs_equal", "state_equal", "deployment_bitwise_equal")
TOLERANCES = {"atol": 2e-7, "rtol": 2e-5}
REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    with Path(path).open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    digest_ = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest_.update(chunk)
    return digest_.hexdigest()


def hexadecimal(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def descriptor_path(descriptor, label):
    require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"},
            f"{label}: expected path and SHA256 descriptor")
    require(isinstance(descriptor["path"], str) and Path(descriptor["path"]).is_absolute()
            and hexadecimal(descriptor["sha256"], 64), f"{label}: expected absolute path and SHA256")
    return Path(descriptor["path"])


def read_bytes(descriptor, label):
    path = descriptor_path(descriptor, label)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == descriptor["sha256"], f"{label}: SHA256 mismatch")
    return path, raw


def read_json(descriptor, label):
    path, raw = read_bytes(descriptor, label)
    return path, json.loads(raw)


def finite(value, label, *, minimum=None):
    require(type(value) in (float, int) and math.isfinite(value)
            and (minimum is None or value >= minimum), f"invalid {label}")
    return value


def training_binding(row, training, input_sha256):
    require(row.get("status") == "completed" and row.get("scientifically_qualified") is False
            and row.get("input_manifest_sha256") == input_sha256
            and row.get("base_checkpoint_sha256") == training["base_checkpoint_sha256"]
            and row.get("posterior_requests") == 0 and row.get("posterior_draws_contract") == COUNT
            and hexadecimal(row.get("source_commit"), 40), "training receipt identity mismatch")
    require(row["hardware"]["name"].startswith("NVIDIA A100")
            and row["hardware"]["memory_bytes"] >= 39 * 1024**3, "A100 hardware receipt required")


def training_times(row, method, protocol, *, cache_policy=None):
    expected_spec = {"method": method, "replica": 0, "phase": "pilot", "updates": 2,
                     "master_seed": 20261117, "effective_batch": 16, "diagnostic": True,
                     "protocol_sha256": protocol,
                     "learning_rate": 1e-5 if method == "full_finetune" else 1e-4}
    require(row.get("spec") == expected_spec, "two-update training specification mismatch")
    updates, clocks = row["updates"], row["unprofiled_update_wall_seconds"]
    require(len(updates) == len(clocks) == 2, "two distinct fixed-batch timings required")
    result = []
    for position, (update, clock) in enumerate(zip(updates, clocks), start=1):
        expected = {"update": position, "data_position": 16 * position, "episode_count": 16,
                    "query_case_count": 16, "conditional_count": 64, "backward_calls": 64,
                    "optimizer_steps": 1, "neural_precision": "float32",
                    "probability_precision": "float64", "sampling_law": LAW,
                    "mixed_precision": False, "preprocessing_includes_host_to_device": True,
                    "scientifically_qualified": False}
        if cache_policy is not None:
            expected["parametrization_cache_policy"] = cache_policy
        require(all(update.get(k) == v for k, v in expected.items()), "training update contract mismatch")
        wall = finite(clock, "wall seconds", minimum=1e-12)
        prep = finite(update["preprocessing_seconds"], "preparation seconds", minimum=0)
        compute = finite(update["update_seconds"], "compute seconds", minimum=1e-12)
        require(prep + compute <= wall, "training timing components exceed wall time")
        result.append({"batch": position, "wall_seconds": wall, "preparation_and_h2d_seconds": prep,
                       "update_compute_seconds": compute, "other_wall_seconds": wall - (prep + compute),
                       "loss": finite(update["loss"], "loss"),
                       "gradient_norm_before_clip": finite(update["gradient_norm_before_clip"],
                                                           "gradient norm", minimum=0)})
    return result


def trace_metrics(descriptor):
    _, trace = read_json(descriptor, "conditional Chrome trace")
    counts, durations, operators = Counter(), Counter(), Counter()
    cpu_scopes = {}
    for event in trace["traceEvents"]:
        if "dur" not in event:
            continue
        category = event.get("cat")
        if category:
            counts[category] += 1
            durations[category] += event["dur"]
        name = event.get("name", "")
        if category in ("cpu_op", "cuda_runtime") and name in PROFILE_NAMES:
            operators[name] += 1
        if category == "user_annotation" and name.removeprefix("engineering::") in SCOPES:
            require(name not in cpu_scopes, f"duplicate CPU scope: {name}")
            cpu_scopes[name] = event["dur"]
    return ({name: {"count": count, "sum_duration_us": durations[name]} for name, count in counts.items()},
            dict(operators), cpu_scopes)


def profile(row, trace, *, baseline_audit=None):
    record = row["conditional_profile"]
    require(record.get("episode_position") == 0 and record.get("stage_index") == 0
            and record.get("parameter_name") == "mu"
            and Path(trace["path"]).name == record["trace"], "matched first conditional profile required")
    categories, trace_ops, trace_scopes = trace_metrics(trace)
    kernel = categories["kernel"]
    if baseline_audit is None:
        require(record["trace_sha256"] == trace["sha256"], "profile trace SHA256 mismatch")
        claimed = record["cuda_kernel_events"]
        require(claimed["count"] == kernel["count"]
                and math.isclose(claimed["duration_us"], kernel["sum_duration_us"], rel_tol=1e-10),
                "retained GPU kernel totals mismatch")
    else:
        require(baseline_audit["trace_sha256"] == trace["sha256"]
                and baseline_audit["source_commit"] == row["source_commit"]
                and baseline_audit["categories"] == categories, "baseline trace category audit mismatch")
    native = {op["name"]: op["count"] for op in record["operator_summary"] if op["name"] in PROFILE_NAMES}
    require(set(native) == set(PROFILE_NAMES) and trace_ops == native, "native profile operator counts mismatch")
    scopes = {}
    for name in SCOPES:
        key = f"engineering::{name}"
        rows = [r for r in record["operator_summary"] if r["name"] == key and r["cpu_total_us"] > 0]
        require(len(rows) == 1 and rows[0]["count"] == 1 and key in trace_scopes
                and math.isclose(rows[0]["cpu_total_us"], trace_scopes[key], abs_tol=.002),
                f"CPU profiler scope differs from raw CPU annotation: {name}")
        scopes[name] = trace_scopes[key] / 1e6
    return {"trace_sha256": trace["sha256"], "native_operator_counts": native,
            "cuda_kernel_count": kernel["count"], "sum_cuda_kernel_elapsed_seconds": kernel["sum_duration_us"] / 1e6,
            "cpu_scope_seconds": scopes, "traced_wall_seconds": finite(record["profiled_preparation_and_conditional_wall_seconds"], "traced wall seconds", minimum=1e-12),
            "scope": "Episode 0, stage 0 (mu). Sum of CUDA kernel event durations can include overlap. CPU scopes can nest."}


def audit_summary(descriptor, *, rejected):
    _, raw = read_bytes(descriptor, "rejected cache audit" if rejected else "CPU audit")
    if raw.lstrip().startswith(b"<"):
        root = ET.fromstring(raw)
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        count = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                 for key in ("tests", "errors", "failures", "skipped")}
        if rejected:
            require(count["failures"] > 0 and b"CheckpointError" in raw,
                    "rejected whole-conditional cache CheckpointError required")
        else:
            require(count["tests"] > 0 and all(count[key] == 0 for key in ("errors", "failures", "skipped")),
                    "passing CPU audit required")
        return dict(count, artifact=descriptor, passed=not rejected)
    value = json.loads(raw)
    require(value.get("passed") is (not rejected), "CPU audit outcome does not match its evidence role")
    if rejected:
        require("CheckpointError" in json.dumps(value.get("failure")),
                "rejected whole-conditional cache CheckpointError required")
    return {"artifact": descriptor, "receipt": value, "passed": not rejected}


def parity_summary(parity):
    require(len(parity) == 2, "initial and post-update-one parity records required")
    compact = []
    for phase, row in zip(("initial", "after_update_1"), parity):
        result = row["gradient_comparison"]
        require(row.get("episode_position") == 0 and row.get("stage_index") == 0
                and row.get("parameter_name") == "mu" and row.get("members") == 8
                and row.get("passed") is True and row.get("logits_density_loss_bitwise_equal") is True
                and result.get("passed") is True and all(result.get(k) == v for k, v in TOLERANCES.items())
                and all(result.get(key) == [] for key in ("missing", "unexpected", "failed"))
                and type(result.get("tensor_count")) is int and result["tensor_count"] > 0
                and len(result["tensors"]) == result["tensor_count"]
                and all(tensor.get("passed") is True for tensor in result["tensors"].values()),
                "same-weight gradient/eight-member conditional parity failed")
        compact.append({"phase": phase, "tensors": result["tensor_count"], "members": 8,
                        "logits_density_loss_bitwise_equal": True, **TOLERANCES,
                        "max_absolute_error": finite(result["max_absolute_error"], "gradient delta", minimum=0)})
    return compact


def checkpoint_summary(descriptor, receipt, training, input_sha256, *, cached):
    manifest_path, manifest = read_json(descriptor, "final training checkpoint manifest")
    require(manifest_path.name == "manifest.json" and descriptor["sha256"] == receipt["checkpoint_manifest_sha256"],
            "training checkpoint receipt SHA256 mismatch")
    identity = {"spec": receipt["spec"], "stream_id": digest(training["episodes"]),
                "protocol_sha256": training["protocol_sha256"], "qualification_inputs_sha256": input_sha256,
                "source_commit": receipt["source_commit"], "cache_parametrizations": cached}
    meta = read_campaign_checkpoint(manifest_path.parent, expected_identity=identity)
    require(file_hash(manifest_path) == descriptor["sha256"], "training checkpoint manifest changed during validation")
    model = receipt["final_model_sha256"]
    method = receipt["spec"]["method"]
    rank = {"lora_r4": 4, "lora_r16": 16, "full_finetune": None}[method]
    deploy = meta["deployment"]
    require(meta["state"]["update"] == 2 and meta["state"]["data_position"] == 32
            and meta["state"]["stream_id"] == identity["stream_id"]
            and meta["history"] == [{"update": u["update"], "loss": u["loss"]} for u in receipt["updates"]]
            and hexadecimal(model, 64) and model == meta["weights_sha256"] == deploy["weights_sha256"]
            and deploy["campaign_method"] == method and deploy["rank"] == rank
            and deploy["format"] == ("peft_adapter" if rank else "full_tabpfn_safetensors")
            and deploy["base_checkpoint_sha256"] == training["base_checkpoint_sha256"]
            and deploy["sampling_law"] == LAW and deploy["scientifically_qualified"] is False,
            "two-update checkpoint content does not match training receipt")
    return {"campaign_checkpoint": str(manifest_path.parent.resolve()),
            "checkpoint_manifest_sha256": descriptor["sha256"],
            "campaign_metadata_sha256": manifest["files"]["campaign.json"], "model_sha256": model}


def collect_training(inputs, training, training_sha256):
    baseline = inputs["baseline"]
    _, old = read_json(baseline["result"], "earlier stock baseline")
    _, old_profile = read_json(baseline["profile"], "earlier stock profile")
    _, profile_audit = read_json(baseline["profile_categories"], "earlier stock profile category audit")
    for row, mode in ((old, "baseline"), (old_profile, "profile")):
        training_binding(row, training, training_sha256)
        require(row.get("schema") == "finetuneforge.training_engineering.v1" and row.get("mode") == mode
                and row.get("original_source_commit") == training["source_commit"]
                and row.get("job_id") == old["job_id"] and row.get("source_commit") == old["source_commit"]
                and row.get("spec") == old["spec"], "legacy baseline/profile receipt mismatch")
    prior = {"source_commit": old["source_commit"], "job_id": old["job_id"],
             "updates": training_times(old, "lora_r4", training["protocol_sha256"]),
             "profile": profile(old_profile, baseline["trace"], baseline_audit=profile_audit)}
    qualification = inputs["qualification"]
    _, allocation = read_json(qualification["allocation"], "same-node training allocation")
    require(allocation.get("schema") == "finetuneforge.training_engineering_qualification_allocation.v1"
            and allocation.get("worker_returncodes") == dict.fromkeys(METHODS, 0)
            and allocation.get("input_manifest_sha256") == training_sha256
            and allocation.get("scientifically_qualified") is False
            and allocation.get("posterior_requests") == 0 and isinstance(allocation.get("hostname"), str)
            and bool(allocation["hostname"]) and set(qualification["lanes"]) == set(METHODS),
            "completed four-lane allocation required")
    lanes = {}
    for lane, method in METHODS.items():
        descriptor = qualification["lanes"][lane]
        _, row = read_json(descriptor["result"], f"{lane} training receipt")
        training_binding(row, training, training_sha256)
        require(row.get("schema") == QUALIFICATION_SCHEMA and row.get("lane") == lane
                and all(row.get(k) == allocation[k] for k in ("job_id", "hostname", "source_commit")),
                "training receipt must match the same-node allocation")
        cached = lane != "stock_r4"
        controls = {key: row[key] for key in ("hardware", "peak_allocated_bytes", "peak_reserved_bytes")}
        record = dict(controls, checkpoint=checkpoint_summary(descriptor["checkpoint_manifest"], row, training,
                                                               training_sha256, cached=cached),
                      updates=training_times(row, method, training["protocol_sha256"],
                                             cache_policy=CACHE_POLICY if cached else "stock_uncached"))
        if cached:
            record["same_weight_parity"] = parity_summary(row["parity"])
            restart = row["restart"]
            require(all(restart.get(k) is True for k in RESTART_CHECKS), "cached restart parity failed")
            record["restart"] = {k: restart[k] for k in RESTART_CHECKS}
        if lane in R4_LANES:
            record["profile"] = profile(row, descriptor["trace"])
        lanes[lane] = record
    stock, cached = (lanes[name] for name in R4_LANES)
    hardware_keys = ("name", "memory_bytes", "cuda_runtime")
    require(all(stock["hardware"][k] == cached["hardware"][k] for k in hardware_keys)
            and stock["hardware"]["uuid"] != cached["hardware"]["uuid"],
            "pair requires distinct GPUs of the same model, capacity and CUDA runtime")
    comparison = []
    for original, candidate in zip(stock["updates"], cached["updates"]):
        comparison.append({"batch": original["batch"],
                           "wall_ratio_stock_over_cached": original["wall_seconds"] / candidate["wall_seconds"],
                           "compute_ratio_stock_over_cached": original["update_compute_seconds"] / candidate["update_compute_seconds"],
                           "loss_delta_cached_minus_stock": candidate["loss"] - original["loss"],
                           "gradient_norm_delta_cached_minus_stock": candidate["gradient_norm_before_clip"] - original["gradient_norm_before_clip"]})
    return {"earlier_stock_baseline": prior, "lanes": lanes, "stock_cached_r4_comparison_by_batch": comparison,
            "execution_policy": CACHE_POLICY, "source_commit": allocation["source_commit"],
            "job_id": allocation["job_id"], "hostname": allocation["hostname"],
            "scope": "Two different fixed episode batches per lane, timed without profiler instrumentation. "
                     "The same-node rank-four comparison uses the same initial model, data and settings. The cached lane "
                     "also ran separate conditional parity checks before its timed regions."}



def final_state_comparison(descriptor, performance):
    _, receipt = read_json(descriptor, "final model and optimizer comparison")
    comparators = {"cache_r4": "same_allocation_current_source",
                   "cache_r16": "authenticated_historical_stock_same_data",
                   "full_finetune": "authenticated_historical_stock_same_data"}
    require(receipt.get("schema") == "finetuneforge.training_final_state_comparison.v1"
            and receipt.get("qualification_source_commit") == performance["source_commit"]
            and all(receipt.get(k) == v for k, v in TOLERANCES.items())
            and set(receipt["cases"]) == set(comparators), "final-state comparison identity or tolerances mismatch")
    cases = {}
    for lane, reference_kind in comparators.items():
        row = receipt["cases"][lane]
        candidate = performance["lanes"][lane]["checkpoint"]
        reference_path = Path(row["reference_path"])
        reference_sha = row["reference_checkpoint_manifest_sha256"]
        require(row.get("reference_kind") == reference_kind and row.get("spec_and_data_state_equal") is True
                and row.get("candidate_path") == candidate["campaign_checkpoint"]
                and row.get("candidate_checkpoint_manifest_sha256") == candidate["checkpoint_manifest_sha256"]
                and reference_path.is_absolute() and file_hash(reference_path / "manifest.json") == reference_sha,
                "final-state candidate/reference checkpoint binding mismatch")
        if lane == "cache_r4":
            stock = performance["lanes"]["stock_r4"]["checkpoint"]
            require(str(reference_path) == stock["campaign_checkpoint"]
                    and reference_sha == stock["checkpoint_manifest_sha256"], "final-state rank-four reference is not the same-allocation stock checkpoint")
        reference = read_campaign_checkpoint(reference_path)
        candidate_meta = json.loads((Path(candidate["campaign_checkpoint"]) / "campaign.json").read_bytes())
        require(reference["state"] == candidate_meta["state"]
                and reference["identity"]["spec"] == candidate_meta["identity"]["spec"]
                and all(reference["deployment"][k] == candidate_meta["deployment"][k]
                        for k in ("base_checkpoint_sha256", "campaign_method", "rank", "sampling_law")),
                "final-state reference differs in spec, deployment, or data state")
        result = {"reference_kind": reference_kind, "reference_checkpoint": {
                    "campaign_checkpoint": str(reference_path), "checkpoint_manifest_sha256": reference_sha,
                    "campaign_metadata_sha256": file_hash(reference_path / "campaign.json"),
                    "model_sha256": reference["weights_sha256"]},
                  "candidate_checkpoint": candidate}
        for key in ("model", "optimizer"):
            comparison = row[key]
            tensors = comparison["tensors"]
            failed = sorted(name for name, item in tensors.items() if item.get("passed") is not True)
            passed = bool(tensors) and not failed and not comparison["missing"] and not comparison["unexpected"]
            require(all(comparison.get(k) == v for k, v in TOLERANCES.items())
                    and comparison.get("tensor_count") == len(tensors) > 0
                    and sorted(comparison["failed"]) == failed and comparison.get("passed") is passed
                    and comparison.get("max_absolute_error") == max(item.get("max_absolute_error") or 0. for item in tensors.values()),
                    "final-state tensor aggregate mismatch")
            result[key] = {"passed": passed, "bitwise_equal": passed and all(item.get("exact") is True for item in tensors.values()),
                           "tensor_count": len(tensors), "failed_tensors": failed,
                           "max_absolute_error": finite(comparison["max_absolute_error"], "final-state tensor delta", minimum=0)}
        result["passed"] = result["model"]["passed"] and result["optimizer"]["passed"]
        require(row.get("passed") is result["passed"], "final-state case outcome mismatch")
        cases[lane] = result
    passed = all(row["passed"] for row in cases.values())
    require(receipt.get("passed") is passed, "final-state overall outcome mismatch")
    return {"artifact": descriptor, "source_commit": receipt["qualification_source_commit"],
            **TOLERANCES, "passed": passed, "cases": cases,
            "scope": "Two-update stock-versus-cached model and optimizer tensors. Cached restart parity is recorded in the training lanes."}


def exact_marginal_study_thresholds(absolute):
    limits = {"cdf": CDF_LIMIT, "absolute_bias_reference_sd": BIAS_LIMIT, "sd_ratio": list(SD_LIMITS)}
    marginals = {}
    for name, row in absolute["marginals"].items():
        checks = {"cdf": row["cdf_sup_error"] <= CDF_LIMIT,
                  "mean": row["mean_error_in_analytic_sd"] <= BIAS_LIMIT,
                  "sd": SD_LIMITS[0] <= row["sd_ratio"] <= SD_LIMITS[1]}
        marginals[name] = {"checks": checks, "passed": all(checks.values())}
    return {"limits": limits, "marginals": marginals, "full_study_matrix_run": False,
            "scope": "Original study thresholds applied to the exact mu, tau, and sigma marginal diagnostics for this single control."}

def diagnostic_agrees(recorded, recomputed):
    """Allow roundoff across CPU BLAS libraries when rechecking statistics."""
    if isinstance(recomputed, dict):
        return (isinstance(recorded, dict) and recorded.keys() == recomputed.keys()
                and all(diagnostic_agrees(recorded[k], v) for k, v in recomputed.items()))
    if type(recomputed) is float:
        return type(recorded) is float and math.isclose(recorded, recomputed, rel_tol=2e-10, abs_tol=2e-12)
    return type(recorded) is type(recomputed) and recorded == recomputed


def validate_posterior_binding(group, inputs, bank, checkpoint, helper, rank):
    run = group["runs"][0]
    require(group["method"] == f"lora_r{rank}" and group["mode"] == "merged_lora"
            and group["tiled"] is False and group["completed_updates"] == 2
            and run["deployment_mode"] == "merged_lora", "posterior method/deployment mode mismatch")
    require(run["origin_model_sha256"] == checkpoint["model_sha256"]
            and run["checkpoint_manifest_sha256"] == checkpoint["checkpoint_manifest_sha256"],
            "posterior deployment origin differs from its authenticated checkpoint")
    binding = run["binding"]
    backend = binding["backend"]
    require(run["bank_manifest_sha256"] == inputs["small_bank"]["manifest_sha256"]
            and all(binding[k] == bank.manifest[k] for k in ("theta", "observations"))
            and binding["problem"] == bank.spec.fingerprint, "posterior request bank arrays or problem mismatch")
    pinned_sources = ((binding["source_sha256"], ("sampling.py", "science.py", "spec.py", "banks.py", "config.py")),
                      (backend["source_sha256"], ("model.py", "sampling_backend.py")),
                      (backend["deployment"]["source_files"], ("inference_deployment.py", "model.py", "model_loading.py", "training.py")))
    require(all(sources == {name: helper._input_source(inputs, name) for name in names}
                for sources, names in pinned_sources)
            and backend.get("engineering_source_sha256") == helper._input_source(inputs, "inference_engineering.py"),
            "posterior request differs from the input revision's pinned core sources")
    helper.validate_attention_policy(run["attention"], inputs, tiled=False)


def collect_posteriors(descriptors, training, checkpoints, *, rank=4):
    _, inputs = read_json(descriptors["inputs"], "posterior input manifest")
    _, allocation = read_json(descriptors["allocation"], "posterior allocation")
    posterior_input_sha = descriptors["inputs"]["sha256"]
    expected = {"schema": POSTERIOR_SCHEMA, "seed": 20265117, "microbatch_size": 1024, "mode": "merged_lora",
                "posterior_draws": COUNT, "timing_repetitions": 1, "attention_policy": "stock",
                "training_pilots_enabled": False, "new_forward_evaluations": 0, "scientifically_qualified": False}
    require(all(inputs.get(k) == v for k, v in expected.items())
            and type(inputs.get("rank", 4)) is int and inputs.get("rank", 4) == rank
            and hexadecimal(inputs.get("source_commit"), 40)
            and all(inputs[k] == training[k] for k in ("protocol_sha256", "base_checkpoint", "base_checkpoint_sha256",
                                                         "observation_sha256", "frozen_model_sha256"))
            and set(inputs["checkpoints"]) == set(descriptors["lanes"]) == set(POSTERIOR_LANES),
            "posterior fixed experiment identity mismatch")
    descriptor = inputs["small_bank"]
    require(all(descriptor[k] == training["posterior_bank"][k] for k in ("worlds", "dimension", "observations",
                                                                         "manifest_sha256", "problem"))
            and (descriptor["worlds"], descriptor["dimension"], descriptor["observations"]) == (256, 4, 10),
            "posterior bank differs from the fixed D4/M10/C256 control")
    require(allocation == {"schema": POSTERIOR_SCHEMA, "source_commit": inputs["source_commit"],
                           "input_manifest_sha256": posterior_input_sha, "posterior_draws_per_request": COUNT,
                           "scientifically_qualified": False, "worker_returncodes": dict.fromkeys(POSTERIOR_LANES, 0)},
            "posterior allocation identity or completion mismatch")
    template = Path(inputs["template_path"])
    require(file_hash(template) == inputs["template_sha256"], "posterior control template SHA256 mismatch")
    prior = json.loads(template.read_bytes())
    require(all(prior[k] == inputs[k] for k in ("small_bank", "base_checkpoint", "base_checkpoint_sha256",
                                                "frozen_model_sha256", "observation", "observation_sha256", "protocol_sha256")),
            "posterior control changed compared with its pinned template")
    from posteriorforge.study.banks import load_bank
    analytic = load_helper("training_report_analytic_control", "check-study-orchestration-cpu.py")
    previous_report = load_helper("training_report_inference_helpers", "report-inference-engineering.py")
    bank = load_bank(descriptor["directory"], expected_sha256=descriptor["manifest_sha256"])
    require(bank.spec.fingerprint == descriptor["problem"] == tiny_problem(posterior=True).fingerprint,
            "analytic zero-sensor control problem required")
    require(file_hash(inputs["base_checkpoint"]) == inputs["base_checkpoint_sha256"], "base checkpoint SHA256 mismatch")
    query = previous_report._query_descriptor(inputs)
    require(np.all(np.load(inputs["observation"], allow_pickle=False) == 0), "zero-sensor query must be exactly zero")
    records, values = {}, {}
    for name in POSTERIOR_LANES:
        descriptor = descriptors["lanes"][name]
        path, row = read_json(descriptor["result"], f"{name} posterior receipt")
        container_path, container = read_json(descriptor["container"], f"{name} posterior container")
        checkpoint = checkpoints[name]
        expected_receipt = {"schema": POSTERIOR_SCHEMA, "variant": name, "status": "completed", "scientifically_qualified": False,
                            "source_commit": inputs["source_commit"], "input_manifest_sha256": posterior_input_sha,
                            "checkpoint": checkpoint, "posterior_draws": COUNT, "posterior_result_sha256": descriptor["container"]["sha256"]}
        require(all(row.get(k) == v for k, v in expected_receipt.items()) and inputs["checkpoints"][name] == checkpoint,
                f"{name} posterior does not originate from its authenticated training checkpoint")
        require(path == container_path.with_name("result.json") and container_path.name == "posterior-result.json"
                and descriptor_path(descriptor["metrics"], f"{name} metrics") == container_path.parent / "repetition-0/metrics.json"
                and container.get("context") == inputs["small_bank"] and container.get("scientifically_qualified") is False,
                "posterior group artifact/contract mismatch")
        group, samples = previous_report.load_group(container_path.parent, repetitions=1, context=256)
        run = group["runs"][0]
        require(run["metrics_sha256"] == descriptor["metrics"]["sha256"]
                and group["container_sha256"] == descriptor["container"]["sha256"], "posterior metric/container SHA256 mismatch")
        validate_posterior_binding(group, inputs, bank, checkpoint, previous_report, rank)
        binding = run["binding"]
        require(all(container["deployment"].get(k) == value for k, value in binding["backend"]["deployment"].items()),
                "posterior deployment differs from request identity")
        require(binding["query"] == query and binding["seed"] == inputs["seed"]
                and binding["microbatch_size"] == inputs["microbatch_size"] and binding["law"] == LAW
                and binding["source_policy"] == "cancelled_zero_width" and binding["uniform_law"] == "uint53-open.v1"
                and binding["identities"]["protocol_sha256"] == inputs["protocol_sha256"],
                "posterior query/protocol/uniform/conditional law mismatch")
        values[name] = samples[0]
        absolute = analytic.analyze_samples(values[name])
        require(diagnostic_agrees(run["analytical_diagnostics"], absolute),
                "retained exact diagnostic differs from the complete posterior")
        records[name] = {"request": run, "analytic_control": absolute, "job_id": row["job_id"],
                         "exact_marginal_study_thresholds": exact_marginal_study_thresholds(absolute),
                         "sample_moments": {"parameter_order": ["mu", "tau", "a", "sigma"],
                                            "mean": values[name].mean(axis=0).tolist(),
                                            "sd": values[name].std(axis=0, ddof=1).tolist()}}
    stock = records["stock"]; cached = records["cached"]
    require(stock["job_id"] == cached["job_id"]
            and previous_report.request_key(stock["request"]["binding"]) == previous_report.request_key(cached["request"]["binding"]),
            "relative posterior comparison requires identical problem, bank, query, seed, law, protocol and core sources")
    comparator_sha = file_hash(inference_engineering.__file__)
    require(comparator_sha == previous_report._input_source(inputs, "inference_engineering.py"),
            "posterior comparator source differs from pinned engineering source")
    comparison = inference_engineering.compare_samples(values["stock"], values["cached"])
    return {"rank": rank, "source_commit": inputs["source_commit"], "comparison_source_sha256": comparator_sha,
            "analytic_control_source_sha256": file_hash(analytic.__file__), "draws_per_request": COUNT,
            "analytic_control_scope": "The 12 exact-control checks use tight Monte Carlo tolerances for this control. Original study thresholds are shown separately for three exact marginals. The full scientific matrix was outside this experiment.",
            "runs": records, "comparison": comparison}, values


def collect_report(manifest_path, input_sha256):
    descriptor = {"path": str(Path(manifest_path).resolve()), "sha256": input_sha256}
    _, inputs = read_json(descriptor, "report input manifest")
    require(inputs.get("schema") == INPUT_SCHEMA, "training engineering report input schema mismatch")
    _, training = read_json(inputs["training_inputs"], "fixed training input manifest")
    require(training.get("schema") == "finetuneforge.cuda_qualification_inputs.v1"
            and training.get("training_effective_batch") == 16 and training.get("posterior_draws") == COUNT
            and training.get("engineering_updates") == 2 and len(training.get("episodes", [])) == 32
            and all(episode["position"] == i and all((episode[k]["worlds"], episode[k]["dimension"],
                                                       episode[k]["observations"]) == (worlds, 4, 1)
                                                      for k, worlds in (("context", 16), ("query", 1)))
                    for i, episode in enumerate(training["episodes"])), "fixed D4/C16 training episodes required")
    audits = {key: audit_summary(inputs["audits"][key], rejected=(key == "rejected_cache"))
              for key in ("cpu", "rejected_cache")}
    performance = collect_training(inputs, training, inputs["training_inputs"]["sha256"])
    checkpoints = {name: performance["lanes"][lane]["checkpoint"] for name, lane in POSTERIOR_LANES.items()}
    posterior, values = collect_posteriors(inputs["posterior"], training, checkpoints)
    report = {"schema": "finetuneforge.training_engineering_report.v1", "scientifically_qualified": False,
              "report_inputs": descriptor, "evidence": inputs, "training": performance, "audits": audits,
              "posterior": posterior, "scope": "Two bounded training updates and one complete posterior request per rank-four variant. "
              "These results do not establish fine-tuning gains, cross-problem transfer, or campaign-scale resource savings."}
    if "final_state_comparison" in inputs:
        report["final_state_comparison"] = final_state_comparison(inputs["final_state_comparison"], performance)
    return report, values



def collect_rank16_report(manifest_path, input_sha256):
    descriptor = {"path": str(Path(manifest_path).resolve()), "sha256": input_sha256}
    _, inputs = read_json(descriptor, "rank-16 report input manifest")
    require(inputs.get("schema") == RANK16_INPUT_SCHEMA, "rank-16 report input schema mismatch")
    _, previous = read_json(inputs["rank4_report"], "retained rank-4 report")
    require(previous.get("schema") == "finetuneforge.training_engineering_report.v1"
            and previous.get("scientifically_qualified") is False
            and previous["posterior"].get("rank", 4) == 4, "retained rank-4 report identity mismatch")
    _, training = read_json(previous["evidence"]["training_inputs"], "fixed training input manifest")
    comparative = previous["final_state_comparison"]
    performance = previous["training"]
    lane = performance["lanes"]["cache_r16"]
    state = comparative["cases"]["cache_r16"]
    require(comparative["source_commit"] == performance["source_commit"]
            and all(comparative[k] == v for k, v in TOLERANCES.items())
            and state["reference_kind"] == "authenticated_historical_stock_same_data"
            and state["candidate_checkpoint"] == lane["checkpoint"]
            and state["model"]["passed"] is False and state["optimizer"]["passed"] is True
            and all(lane["restart"].get(k) is True for k in RESTART_CHECKS),
            "rank-16 report requires the retained failed final-weight comparison and passed cached restart")
    checkpoints = {"stock": state["reference_checkpoint"], "cached": state["candidate_checkpoint"]}
    for checkpoint in checkpoints.values():
        path = Path(checkpoint["campaign_checkpoint"])
        require(path.is_absolute() and file_hash(path / "manifest.json") == checkpoint["checkpoint_manifest_sha256"]
                and file_hash(path / "campaign.json") == checkpoint["campaign_metadata_sha256"],
                "rank-16 posterior origin SHA256 mismatch")
        meta = read_campaign_checkpoint(path)
        require(meta["weights_sha256"] == checkpoint["model_sha256"]
                and meta["deployment"]["weights_sha256"] == checkpoint["model_sha256"]
                and meta["deployment"]["campaign_method"] == "lora_r16" and meta["deployment"]["rank"] == 16
                and meta["deployment"]["format"] == "peft_adapter"
                and meta["deployment"]["base_checkpoint_sha256"] == training["base_checkpoint_sha256"]
                and meta["identity"]["spec"]["protocol_sha256"] == training["protocol_sha256"]
                and meta["state"] == {"update": 2, "data_position": 32, "stream_id": digest(training["episodes"])},
                "rank-16 posterior origin differs from the matched two-update checkpoint")
    posterior, values = collect_posteriors(inputs["posterior"], training, checkpoints, rank=16)
    return {"schema": "finetuneforge.training_rank16_posterior_report.v1", "scientifically_qualified": False,
            "report_inputs": descriptor, "evidence": inputs, "earlier_rank4_posterior_source_commit": previous["posterior"]["source_commit"],
            "training_source_commit": performance["source_commit"],
            "final_state_comparison": {"artifact": comparative["artifact"], **TOLERANCES, **state},
            "cached_restart": lane["restart"], "posterior": posterior,
            "scope": "One fixed 50,000-vector request per rank-16 checkpoint. The historical-stock weight diagnostic remains failed regardless of the relative posterior result."}, values

def plot_training(performance, output):
    plt = load_helper("training_report_plot_helpers", "report-inference-engineering.py")._plotting()
    lanes = performance["lanes"]
    labels = ("Earlier stock r4", "Same-node stock r4", "Cached r4", "Cached r16", "Cached full fine-tune")
    rows = [performance["earlier_stock_baseline"], *[lanes[key] for key in METHODS]]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 4.1), constrained_layout=True)
    metrics = (("preparation_and_h2d_seconds", "Preparation and host-to-device", "#6998B8"),
               ("update_compute_seconds", "Update and optimizer", "#286A79"),
               ("other_wall_seconds", "Other measured wall time", "#B1BBC2"))
    for i, ax in enumerate(axes):
        offset = np.zeros(len(rows))
        for key, label, color in metrics:
            widths = np.array([row["updates"][i][key] for row in rows])
            ax.barh(y, widths, left=offset, color=color, label=label, height=.62)
            offset += widths
        for j, row in enumerate(rows):
            seconds = row["updates"][i]["wall_seconds"]
            ax.annotate(f"{seconds:.1f} s", (seconds, j), xytext=(4, 0), textcoords="offset points", va="center")
        ax.set(title=f"Fixed episode batch {i + 1}", xlabel="Wall seconds per update", yticks=y, yticklabels=labels)
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(u["wall_seconds"] for row in rows for u in row["updates"]) * 1.23)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", frameon=False, ncols=3)
    fig.suptitle("Training update latency on Karolina\nD4 · 16 context worlds · 16 episodes per update · A100")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_profiles(performance, output):
    plt = load_helper("training_report_profile_helpers", "report-inference-engineering.py")._plotting()
    profiles = [performance["earlier_stock_baseline"]["profile"],
                *[performance["lanes"][lane]["profile"] for lane in R4_LANES]]
    width = .24
    positions = np.arange(4)
    fig, ax = plt.subplots(figsize=(9.2, 4.3), constrained_layout=True)
    for i, (name, color) in enumerate(zip(("Earlier stock r4", "Same-node stock r4", "Cached r4"),
                                          ("#9EB6C3", "#2467A6", "#A95F1C"))):
        record = profiles[i]
        counts = [*[record["native_operator_counts"][k] for k in PROFILE_NAMES], record["cuda_kernel_count"]]
        ax.bar(positions + (i - 1) * width, counts, width, color=color, label=name)
    ax.set(xticks=positions, xticklabels=[*PROFILE_NAMES, "CUDA kernel events"], ylabel="Event or operator count",
           title="Work issued for the first stage-0 conditional\nNative Torch operators, CUDA launches and actual GPU kernel events")
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_posteriors(values, output, *, rank=4):
    helpers = load_helper("training_report_posterior_helpers", "report-inference-engineering.py")
    plt = helpers._plotting()
    exact = helpers.exact_distributions()
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.2), constrained_layout=True)
    columns = (("mu", "Population mean μ", (-4., 4.)), ("tau", "Population spread τ", (.025, .45)),
               ("a", "Specimen property a", (-4., 4.)), ("sigma", "Measurement-noise SD σ", (.005, .20)))
    for i, (name, label, limits) in enumerate(columns):
        grid = np.linspace(*limits, 601)
        for lane, title, color in (("stock", f"Stock-trained r{rank}", "#2467A6"),
                                   ("cached", f"Cache-trained r{rank}", "#A95F1C")):
            observed = values[lane][:, i]
            # Keep all draws in the denominator and include tail bins beyond the display window.
            edges = np.unique(np.r_[min(observed.min(), limits[0]), np.linspace(*limits, 81),
                                    max(observed.max(), limits[1])])
            counts, _ = np.histogram(observed, bins=edges)
            axes[0, i].stairs(counts / (COUNT * np.diff(edges)), edges, color=color, lw=1.3, label=title)
            axes[1, i].plot(np.sort(observed), np.arange(1, COUNT + 1) / COUNT, color=color, lw=1.2)
        if name in exact:
            axes[0, i].plot(grid, exact[name].pdf(grid), color="#182A3A", ls="--", lw=1.7, label="Exact posterior")
            axes[1, i].plot(grid, exact[name].cdf(grid), color="#182A3A", ls="--", lw=1.7)
        axes[0, i].set(xlim=limits, title=label)
        axes[1, i].set(xlim=limits, ylim=(0, 1), xlabel=label)
        for ax in axes[:, i]:
            ax.grid(alpha=.14)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Cumulative probability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=3)
    fig.suptitle(f"Posterior after two LoRA-r{rank} updates\n50,000 complete vectors per variant · 256 context worlds · ten zero readings")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(manifest_path, input_sha256, output, *, rank16=False):
    collect = collect_rank16_report if rank16 else collect_report
    result, values = collect(manifest_path, input_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if rank16:
        plot_posteriors(values, output / "rank16-posteriors.png", rank=16)
    else:
        plot_training(result["training"], output / "training-latency.png")
        plot_profiles(result["training"], output / "training-profile.png")
        plot_posteriors(values, output / "training-posteriors.png")
    result["figures"] = {path.name: file_hash(path) for path in sorted(output.glob("*.png"))}
    (output / "report.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return output / "report.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank16", action="store_true", help="Build the separate rank-16 posterior report")
    args = parser.parse_args(argv)
    print(build_report(args.inputs, args.input_sha256, args.output, rank16=args.rank16))


if __name__ == "__main__":
    main()
