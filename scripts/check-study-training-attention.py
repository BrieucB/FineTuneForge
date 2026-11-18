#!/usr/bin/env python3
"""Measure three retained rank-4 pilot conditionals in one authorized A100 job."""
from argparse import ArgumentParser
from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback

from posteriorforge.study.banks import PairedBank, array_descriptor, load_bank
from posteriorforge.study.campaign import CampaignSpec
from posteriorforge.study.ordering import reorder_bank, reorder_problem
from posteriorforge.study.pilot import CUDA_TILED_ATTENTION_EXECUTION, authorize_cuda_pilot
from posteriorforge.study.workflows import TrainingRequest, _write_json, file_sha256

SCRATCH = Path("/scratch/project/eu-26-17/eubrieucb/posteriorforge")
GPU_ENVIRONMENT = SCRATCH / "envs/study-training-gpu"
RETAINED_ROOT = SCRATCH / "results/hierarchical-ft-v1-learning-pilot-r4-seed0-lr1e-4"
RETAINED_ATTEMPT_SHA256 = "cacda01cdda6365edc4f6f73618a15a8716bf9a45ca02acc0c8feeea1da873af"
RETAINED_INITIAL_WEIGHTS_SHA256 = "53924b6e90586130ae704b1277cde6f5293b8dd16216a4f3cec403bb46ac06ae"
CHECKPOINT = SCRATCH / "cache/model/tabpfn-v2-regressor.ckpt"
CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
ATOL, RTOL = 2e-7, 2e-5
MEMBER_COUNT = 8
SMALL_SCORE_CAP = 4 * 1024**2
SCORE_CAP = CUDA_TILED_ATTENTION_EXECUTION["attention_max_score_bytes"]
# Each width occurs in four of the eight ordered stock preprocessors.
CASES = (
    (2, 0, "train-02493", "population-1.within_sd", 64, (244, 119)),
    (7, 0, "train-09424", "population-2.within_sd", 1024, (251, 101)),
    (11, 68, "train-08858", "population-3.family-1-specimen-2", 1024, (720, 309)),
)


def load_retained_episodes(request, config_path, attempt_path):
    """Authenticate raw banks, stored ordering, and all selected ordered arrays."""
    spec = CampaignSpec(**request.campaign)
    if (spec.method, spec.replica, spec.phase, spec.learning_rate) != ("lora_r4", 0, "pilot", 1e-4):
        raise ValueError("qualification requires the retained rank-4 pilot identity")
    base = (config_path.parent / request.base_checkpoint).resolve(strict=True)
    if base != CHECKPOINT or file_sha256(base) != CHECKPOINT_SHA256:
        raise ValueError("pinned cached checkpoint identity mismatch")
    if (config_path.parent / request.data_cache_directory).resolve() != RETAINED_ROOT / "data-cache":
        raise ValueError("qualification must bind the retained bank cache")
    if (attempt_path.name != "attempt.json"
            or attempt_path.parent.parent != RETAINED_ROOT / "run-01/campaign/attempts"):
        raise ValueError("qualification must bind the retained attempt")
    source = attempt_path.read_bytes()
    if sha256(source).hexdigest() != RETAINED_ATTEMPT_SHA256:
        raise ValueError("retained attempt checksum mismatch")
    attempt = json.loads(source)
    episodes = {item["request"]["position"]: item for item in attempt["episodes"]}
    return spec, tuple(load_case(case, episodes[case[0]]) for case in CASES)


def load_case(case, episode):
    position, stage, setup, parameter, context_size, _ = case
    request = episode["request"]
    if (request["position"], request["context_size"]) != (position, context_size):
        raise ValueError("retained position or context-size mismatch")
    context_desc, query_desc = episode["datasets"]

    def load(desc):
        path = Path(desc["path"]).resolve(strict=True)
        if path.parent != RETAINED_ROOT / "data-cache" or path.name != desc["dataset_id"]:
            raise ValueError("retained bank path/dataset identity mismatch")
        return load_bank(path, expected_sha256=desc["bank_manifest_sha256"])

    context, queries = load(context_desc), load(query_desc)
    if (len(context.theta), len(queries.theta)) != (request["context_bank_count"], request["query_bank_count"]):
        raise ValueError("retained bank size mismatch")
    if any(bank.spec.fingerprint != request["setup_fingerprint"] for bank in (context, queries)):
        raise ValueError("retained bank setup fingerprint mismatch")
    query_index = request["query_index"]
    context = context.prefix(context_size)
    query = PairedBank(queries.spec, queries.theta[query_index:query_index + 1],
                       queries.observations[query_index:query_index + 1], {})
    _, ordering = reorder_problem(context.spec, request["ordering_seed"])
    if ordering.to_dict() != episode["ordering"]:
        raise ValueError("retained parameter/observation ordering mismatch")
    context, query = reorder_bank(context, ordering), reorder_bank(query, ordering)
    if (context.spec.id, context.spec.split, context.spec.parameters[stage].name) != (setup, "train", parameter):
        raise ValueError("retained setup or teacher-forced stage identity mismatch")
    for label, bank in (("context", context), ("query", query)):
        for field in ("theta", "observations"):
            if array_descriptor(getattr(bank, field)) != episode[f"{label}_{field}"]:
                raise ValueError(f"retained {label} {field} checksum mismatch")
    return case, episode, context, query


def cuda_memory(torch, device):
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def fresh_model(torch, spec, device):
    from posteriorforge.study.model import TabPFNBackbone, attach_lora, weights_fingerprint

    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(spec.initialization_seed)
        model = TabPFNBackbone.from_checkpoint(CHECKPOINT)
        model.backbone = attach_lora(model.backbone, rank=spec.rank)
    if weights_fingerprint(model) != RETAINED_INITIAL_WEIGHTS_SHA256:
        raise ValueError("fresh model weights differ from the retained pilot initialization")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 neural weights are required")
    return model.to(device)


def prepare_stage(torch, device, item):
    from tabpfn import TabPFNRegressor
    from posteriorforge.study.model import prepare_fitted_regressor_members
    from posteriorforge.study.training import teacher_forced_inputs

    case, _, context, query = item
    _, stage_index, _, _, size, widths = case
    stage = teacher_forced_inputs(context, query)[stage_index]
    estimator = TabPFNRegressor(
        model_path=CHECKPOINT, device="cpu", n_estimators=MEMBER_COUNT, random_state=0,
        n_jobs=1, inference_precision=torch.float32, fit_mode="fit_preprocessors",
        softmax_temperature=.9, average_before_softmax=False,
    )
    estimator.fit(stage.context_features.copy(), stage.context_targets.copy())
    ensemble = prepare_fitted_regressor_members(
        estimator, stage.query_features.copy(), device=device, source_policy="cancelled_zero_width",
    )
    shapes = [tuple(m.context_x.shape) for m in ensemble.members]
    if shapes != [(size, width) for width in widths for _ in range(MEMBER_COUNT // 2)]:
        raise ValueError(f"prepared member shapes differ from retained forensic geometry: {shapes}")
    if any(config.subsample_ix is not None for config in estimator.executor_.ensemble_configs):
        raise ValueError("qualification requires the complete retained context")
    targets = torch.tensor(stage.query_targets, dtype=torch.float64, device=device)
    jacobian = torch.tensor(stage.decoder_log_jac, dtype=torch.float64, device=device)
    return ensemble, targets, jacobian


def gradient_vector(torch, model, row):
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    used = tuple(name for name, grad in gradients if grad is not None)
    if not used:
        raise RuntimeError("no LoRA gradients")
    vector = torch.cat([grad.detach().cpu().reshape(-1) for _, grad in gradients if grad is not None])
    row.update(trainable_tensors=len(gradients), gradient_tensors=len(used),
               gradient_elements=vector.numel())
    if vector.dtype != torch.float32 or not bool(torch.isfinite(vector).all()):
        raise RuntimeError("the training check requires finite FP32 LoRA gradients")
    row.update(gradient_nonzero_elements=int(torch.count_nonzero(vector)),
               gradient_l2=float(torch.linalg.vector_norm(vector.double())))
    if row["gradient_nonzero_elements"] == 0:
        raise RuntimeError("the training check produced only zero gradients")
    return used, vector


def measure(torch, device, spec, item, score_cap, report, save):
    from posteriorforge.study.cuda_attention import install_training_tiled_attention
    from posteriorforge.study.model import CANONICAL_FULL_SUPPORT
    from posteriorforge.study.training_execution import conditional_forward_cache

    case, episode, _, _ = item
    position, stage, setup, parameter, context_size, widths = case
    row = {
        "position_zero_based": position,
        "stage_zero_based": stage,
        "setup_id": setup,
        "parameter": parameter,
        "context_size": context_size,
        "members": MEMBER_COUNT,
        "item_score_shapes": [[(width + 1) // 2 + 1, 6, context_size, context_size]
                              for width in widths for _ in range(MEMBER_COUNT // 2)],
        "retained_episode": {key: episode[key] for key in
                             ("request", "datasets", "ordering", "context_theta",
                              "context_observations", "query_theta", "query_observations")},
        "execution": "stock" if score_cap is None else "tiled",
        "score_cap_bytes": score_cap,
    }
    report["measurements"].append(row)
    row["phase"] = "preparation"
    save()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    attention = None
    started = time.perf_counter()
    try:
        ensemble, targets, jacobian = prepare_stage(torch, device, item)
        model = fresh_model(torch, spec, device)
        torch.cuda.synchronize(device)
        row["preparation_seconds"] = time.perf_counter() - started
        row["preparation_memory"] = cuda_memory(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        row["forward_backward_baseline_memory"] = cuda_memory(torch, device)
        attention = (nullcontext() if score_cap is None
                     else install_training_tiled_attention(model, max_score_bytes=score_cap))
        with attention, conditional_forward_cache(model, enabled=True):
            row["phase"] = "forward_and_density"
            save()
            started = time.perf_counter()
            density = ensemble.density(model, law=CANONICAL_FULL_SUPPORT)
            loss = -density.log_prob(targets, log_abs_det_decode=jacobian)
            if loss.shape != (1,) or loss.dtype != torch.float64 or not bool(torch.isfinite(loss).all()):
                raise RuntimeError("the retained target must have one finite FP64 physical NLL")
            row["loss"] = float(loss.detach())
            torch.cuda.synchronize(device)
            row["forward_and_density_seconds"] = time.perf_counter() - started
            row["forward_and_density_memory"] = cuda_memory(torch, device)

            row["phase"] = "backward"
            save()
            started = time.perf_counter()
            loss.sum().backward()
            torch.cuda.synchronize(device)
            row["backward_seconds"] = time.perf_counter() - started
            row["forward_and_backward_memory"] = cuda_memory(torch, device)
        if score_cap is not None:
            stats = attention.statistics
            if (stats["tiled_training"] <= 0 or stats["checkpointed_tiles"] <= 0
                    or stats["recomputed_tiles"] <= 0
                    or stats["max_estimated_score_bytes"] <= score_cap
                    or not 0 < stats["max_tile_score_bytes"] <= score_cap):
                raise RuntimeError("required bounded training attention and recomputation were not observed")
        used, vector = gradient_vector(torch, model, row)
        row["phase"] = "complete"
        return row, used, vector
    finally:
        row["last_allocator_counters"] = cuda_memory(torch, device)
        if score_cap is not None and attention is not None:
            row["attention"] = attention.statistics
            row["attention_policy"] = attention.metadata
        save()


def compare(torch, stock, tiled):
    stock_row, stock_names, stock_grad = stock
    tiled_row, tiled_names, tiled_grad = tiled
    if stock_names != tiled_names:
        raise RuntimeError("stock and tiled checks used different LoRA gradient tensors")
    gradient_difference = (tiled_grad - stock_grad).abs()
    allowed_gradient_difference = ATOL + RTOL * stock_grad.abs()
    loss_difference = abs(tiled_row["loss"] - stock_row["loss"])
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "loss_absolute_difference": loss_difference,
        "loss_passed": loss_difference <= ATOL + RTOL * abs(stock_row["loss"]),
        "gradient_max_absolute_difference": float(gradient_difference.max()),
        "gradient_max_tolerance_ratio": float((gradient_difference / allowed_gradient_difference).max()),
        "gradients_passed": bool(torch.all(gradient_difference <= allowed_gradient_difference)),
    }


def execute(spec, items, report, save):
    import torch
    from posteriorforge.study.cuda_training import configure_cuda_determinism
    from posteriorforge.study.model import CANONICAL_FULL_SUPPORT

    if torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("pinned Torch 2.6.0 CUDA 12.4 build required")
    configure_cuda_determinism()
    torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("the training attention check requires exactly one visible CUDA device")
    props = torch.cuda.get_device_properties(device)
    report["runtime"] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": props.name,
        "device_total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "autocast": torch.is_autocast_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
        "torch_compile_disable": os.environ["TORCH_COMPILE_DISABLE"],
    }
    if "A100" not in props.name or torch.is_autocast_enabled():
        raise RuntimeError("the training attention check requires an A100 and FP32 neural execution")
    report.update(rank=spec.rank, initialization_seed=spec.initialization_seed,
                  initial_weights_sha256=RETAINED_INITIAL_WEIGHTS_SHA256,
                  checkpoint_sha256=CHECKPOINT_SHA256, density_law=CANONICAL_FULL_SUPPORT)
    save()
    stock = measure(torch, device, spec, items[0], None, report, save)
    tiled = measure(torch, device, spec, items[0], SMALL_SCORE_CAP, report, save)
    report["stock_tiled_comparison"] = compare(torch, stock, tiled)
    save()
    # Complete the two capacity measurements before enforcing the numerical gate.
    for item in items[1:]:
        measure(torch, device, spec, item, SCORE_CAP, report, save)
    result = report["stock_tiled_comparison"]
    if not result["loss_passed"] or not result["gradients_passed"]:
        raise RuntimeError("the matched stock and tiled loss or LoRA gradients exceed tolerance")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    output = args.output.resolve()
    attempt = args.attempt.resolve()
    request = TrainingRequest.from_dict(json.loads(config.read_bytes()))
    if request.execution != CUDA_TILED_ATTENTION_EXECUTION:
        raise ValueError("a v2 tiled CUDA training request is required")
    allocation = authorize_cuda_pilot(
        request_sha256=request.fingerprint, ledger=args.ledger, output=output,
        resume_from=None, stop_after_update=1,
    )
    checkout = Path(allocation["source_checkout"])
    if Path(__file__).resolve() != checkout / "scripts/check-study-training-attention.py":
        raise RuntimeError("the check script must belong to its authorized checkout")
    if output.is_relative_to(RETAINED_ROOT):
        raise ValueError("qualification output must be separate from the retained pilot")
    output.mkdir(exist_ok=False)
    report = {
        "schema": "posteriorforge.study.training_attention_check.v1",
        "allocation": allocation,
        "retained_source": {"job_id": "5451614", "attempt_path": str(attempt),
                            "attempt_sha256": RETAINED_ATTEMPT_SHA256},
        "status": "running",
        "measurements": [],
        "optimizer_steps": 0,
        "posterior_draws": 0,
        "scientifically_qualified": False,
    }

    def save():
        _write_json(output / "qualification.json", report)

    save()
    try:
        if "torch" in sys.modules:
            raise RuntimeError("CUDA pilot authorization and source validation must precede Torch import")
        if (Path(sys.prefix).resolve() != GPU_ENVIRONMENT or sys.version_info[:2] != (3, 11)
                or version("tabpfn") != "2.0.9" or version("peft") != "0.21.3"):
            raise RuntimeError("the verified study-training-gpu interpreter is required")
        required_settings = {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TABPFN_DISABLE_TELEMETRY": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TORCH_COMPILE_DISABLE": "1", "NVIDIA_TF32_OVERRIDE": "0",
            "TABPFN_ALLOW_CPU_LARGE_DATASET": "1",
        }
        if any(os.getenv(key) != value for key, value in required_settings.items()):
            raise RuntimeError("offline FP32 qualification environment is required")
        spec, items = load_retained_episodes(request, config, attempt)
        execute(spec, items, report, save)
    except BaseException as error:
        report.update(status="failed", exception_type=type(error).__name__, exception=str(error),
                      traceback=traceback.format_exc())
        save()
        raise
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
