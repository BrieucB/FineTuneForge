#!/usr/bin/env python3
"""CPU artifact validation and compact figures for the bounded inference study.

No model, simulator, optimizer or GPU work is performed. Every plotted posterior
must contain all 50,000 vectors. Timing repetitions are paired execution repeats,
not independent scientific cases. The two-update adapters remain unqualified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy.stats import lognorm, norm

COUNT = 50_000
CORE_PATHS = {"frozen": "profile-01/frozen", "unmerged": "profile-01/unmerged", "merged": "profile-01/merged"}
FOUR_K_PATHS = {"frozen_stock": "c4k_frozen_stock", "frozen_tiled": "c4k_frozen_tiled",
                "r4_unmerged": "c4k_r4_unmerged", "r4_merged_tiled": "c4k_r4_merged_tiled"}
PARAMETERS = ("mu", "tau", "a", "sigma")


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_bytes())


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"finite positive {name} required")
    return float(value)


def read_samples(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"samples"}:
            raise ValueError("only the complete samples array is accepted")
        values = np.array(data["samples"], copy=True)
    if values.dtype != np.float64 or values.shape != (COUNT, 4) or not np.isfinite(values).all():
        raise ValueError("exactly 50,000 complete finite float64 D4 vectors required")
    if np.any(values[:, (1, 3)] <= 0):
        raise ValueError("tau and sigma must remain inside positive prior support; no repair")
    return values


def load_run(directory, *, expected_context, expected_metrics_sha256=None):
    """Authenticate one immutable completed request before it may enter a plot."""
    root = Path(directory)
    raw = (root / "metrics.json").read_bytes()
    metrics_sha = sha256(raw).hexdigest()
    if expected_metrics_sha256 is not None and metrics_sha != expected_metrics_sha256:
        raise ValueError("metrics SHA256 mismatch")
    row = json.loads(raw)
    if (type(row.get("posterior_draws")) is not int or row["posterior_draws"] != COUNT
            or row.get("scientifically_qualified") is not False):
        raise ValueError("an explicitly unqualified complete 50,000-vector engineering record is required")
    sampling = row["sampling"]; binding = sampling["binding"]
    if (sampling.get("status") != "samples_complete_unassessed" or binding.get("count") != COUNT
            or binding.get("invalid_policy") != "fail" or sampling.get("posterior_sampling_forward_evaluations") != 0
            or binding["theta"].get("shape") != [expected_context, 4]
            or binding["observations"].get("shape") != [expected_context, 10]
            or binding["query"].get("shape") != [10]):
        raise ValueError("complete D4/M10/context request binding required")
    for name in ("theta", "observations", "query"):
        if not _sha(binding[name].get("sha256")):
            raise ValueError("array SHA256 binding required")
    identities = binding["identities"]; backend = binding["backend"]; deployment = backend["deployment"]
    for name in ("model_sha256", "bank_manifest_sha256", "protocol_sha256"):
        if not _sha(identities.get(name)):
            raise ValueError("model/bank/protocol SHA256 identities required")
    if (backend.get("model_sha256") != identities["model_sha256"]
            or deployment.get("model_sha256") != identities["model_sha256"]
            or not _sha(deployment.get("origin_model_sha256"))
            or not _sha(deployment.get("checkpoint_manifest_sha256"))):
        raise ValueError("actual deployment and original checkpoint identities must both be retained")
    if not _sha(row.get("sample_sha256")) or file_sha256(root / "samples.npz") != row["sample_sha256"]:
        raise ValueError("samples SHA256 mismatch")
    values = read_samples(root / "samples.npz")
    if file_sha256(root / "metrics.json") != metrics_sha:
        raise ValueError("metrics changed during artifact read")
    timing = _positive(sampling["wall_seconds"], "total sampling time")
    context_seconds = sampling["context_preparation_seconds_this_run"]
    warm_seconds = sampling["sampling_seconds_this_run"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in (context_seconds, warm_seconds)) or context_seconds + warm_seconds > timing + .1:
        raise ValueError("finite nonnegative context/warm timings must fit total latency")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError("explicit nonnegative peak memory bytes required")
    compact = {"directory": str(root), "metrics_sha256": metrics_sha, "samples_sha256": row["sample_sha256"],
        "posterior_draws": COUNT, "context_worlds": expected_context, "dimension": 4, "observations": 10,
        "sampling_seconds": timing, "context_preparation_seconds": float(context_seconds),
        "warm_sampling_seconds": float(warm_seconds), "peak_allocated_bytes": row["peak_allocated_bytes"],
        "peak_reserved_bytes": row["peak_reserved_bytes"], "model_sha256": identities["model_sha256"],
        "origin_model_sha256": deployment["origin_model_sha256"],
        "checkpoint_manifest_sha256": deployment["checkpoint_manifest_sha256"],
        "bank_manifest_sha256": identities["bank_manifest_sha256"], "deployment_mode": deployment["deployment_mode"],
        "attention": backend["attention"], "binding": binding, "analytical_diagnostics": row.get("analytical_diagnostics"),
        "scientifically_qualified": False}
    return compact, values, row


def request_key(binding):
    fields = ("problem", "query", "theta", "observations", "seed", "microbatch_size", "uniform_law", "law", "source_policy", "source_sha256")
    value = {key: binding[key] for key in fields}
    value["protocol_sha256"] = binding["identities"]["protocol_sha256"]
    value["bank_manifest_sha256"] = binding["identities"]["bank_manifest_sha256"]
    return value


def compare_run_files(original, candidate):
    """Use the producing driver's matched-request and 36-projection comparison."""
    from posteriorforge.study import inference_engineering
    result = inference_engineering.compare_files(original, candidate)
    return dict(result, comparison_source_sha256=file_sha256(inference_engineering.__file__))


def load_group(directory, *, repetitions, context):
    root = Path(directory)
    container_path = root / "posterior-result.json"
    container = _json(container_path)
    runs, samples = [], []
    if len(container.get("runs", [])) != repetitions:
        raise ValueError("completed timing group must retain every declared repetition")
    for rep in range(repetitions):
        compact, values, raw = load_run(root / f"repetition-{rep}", expected_context=context)
        if raw != container["runs"][rep]:
            raise ValueError("metrics/container repetition mismatch")
        runs.append(compact); samples.append(values)
    if any(request_key(row["binding"]) != request_key(runs[0]["binding"]) or
           row["model_sha256"] != runs[0]["model_sha256"] for row in runs):
        raise ValueError("timing repetitions must replay one identical request and deployed model")
    seconds = [row["sampling_seconds"] for row in runs]
    if container.get("seconds") != seconds or not math.isclose(container["median_seconds"], float(np.median(seconds)), rel_tol=1e-12):
        raise ValueError("retained timing summary does not match completed requests")
    if container.get("range_seconds") != [min(seconds), max(seconds)]:
        raise ValueError("retained timing range does not match completed requests")
    return {"repetitions": repetitions, "seconds": seconds, "median_seconds": float(np.median(seconds)),
        "range_seconds": [min(seconds), max(seconds)], "runs": runs,
        "container_sha256": file_sha256(container_path), "method": container["method"],
        "mode": container["mode"], "tiled": container["tiled"], "context_worlds": context,
        "completed_updates": container["deployment"].get("completed_updates"),
        "identical_sample_bytes_across_timing_repeats": len({r["samples_sha256"] for r in runs}) == 1,
        "scientifically_qualified": False}, samples


def paired_comparison(original, candidate):
    value = compare_run_files(original["directory"], candidate["directory"])
    if (value.get("original_manifest_sha256") != original["metrics_sha256"]
            or value.get("candidate_manifest_sha256") != candidate["metrics_sha256"]):
        raise ValueError("comparison metrics SHA256 mismatch")
    return value


def memory_evidence(root):
    repo = Path(__file__).resolve().parents[1]
    old_path = repo / "evidence/cuda-qualification-v1/memory/memory-probe-2.json"
    old = _json(old_path)
    probe_root = Path(root) / "profile-01/probes"
    rows = [_json(probe_root / name) for name in ("probe-74-1.json", "probe-0-1024.json", "probe-37-1024.json", "probe-74-1024.json")]
    if old.get("status") != "out_of_memory" or old["probe"]["context"]["dimension"] != 75 or old["probe"]["context"]["worlds"] != 4096:
        raise ValueError("retained exact-shape stock OOM evidence required")
    match = re.search(r"Tried to allocate ([0-9.]+) GiB", old["error"])
    if not match:
        raise ValueError("failed allocation request is distinct from peak memory and must be retained")
    for row, pair in zip(rows, ((74, 1), (0, 1024), (37, 1024), (74, 1024))):
        if (row.get("status") != "completed" or row.get("finite") is not True
                or (row.get("stage"), row.get("queries")) != pair):
            raise ValueError("all declared finite conditional probes must be complete")
        _positive(row["peak_allocated_bytes"], "conditional allocated peak")
    forecast_path = probe_root / "large-forecast.json"
    forecast = _json(forecast_path)
    expected = [75 * 49 * f([r["conditional_seconds"] for r in rows if r["queries"] == 1024]) for f in (min, max)]
    if (forecast.get("full_75d_run_enabled") is not False or forecast.get("posterior_draws") != COUNT
            or forecast.get("stages") != 75 or forecast.get("microbatches") != 49
            or not np.allclose(forecast["estimated_sampler_seconds_range"], expected, rtol=1e-12, atol=1e-9)):
        raise ValueError("large-run proxy must stay bound to the measured conditional recipe")
    fitting = _json(probe_root / "fitting-stock-tiled-parity.json")
    return {"shape": {"unknowns": 75, "observations": 200, "context_worlds": 4096},
        "stock_oom": {"status": "out_of_memory", "requested_allocation_gib": float(match[1]),
            "recorded_peak_allocated_bytes": old["peak_allocated_bytes"], "evidence_sha256": file_sha256(old_path),
            "scope": "failed allocation request; recorded peak precedes failure and is not required capacity"},
        "tiled_probes": rows, "fitting_stock_tiled_conditional_parity": fitting,
        "large_75d_conditional_proxy": dict(forecast, estimated_single_gpu_hours_range=[v / 3600 for v in expected],
            measured_full_posterior=False, scope="conditional-cost proxy only; full 75D posterior not run; excludes initial preprocessing and IO"),
        "memory_reduction_ratio": None,
        "comparison_scope": "capacity demonstration at one shape; stock requested allocation and successful measured peak are incomparable quantities; no reduction ratio",
        "source_files_sha256": {name: file_sha256(probe_root / name) for name in
            ("probe-74-1.json", "probe-0-1024.json", "probe-37-1024.json", "probe-74-1024.json", "large-forecast.json", "fitting-stock-tiled-parity.json")}}


def validate_group_identity(group, inputs, *, frozen, tiled, merged):
    expected_method = "frozen" if frozen else "lora_r4"
    expected_mode = "merged_lora" if merged else "unmerged"
    wanted = inputs["frozen_model_sha256"] if frozen else inputs["models"]["lora_r4"]["model_sha256"]
    checkpoint = inputs["base_checkpoint_sha256"] if frozen else inputs["models"]["lora_r4"]["checkpoint_manifest_sha256"]
    if group["method"] != expected_method or group["mode"] != expected_mode or group["tiled"] is not tiled:
        raise ValueError("reported lane method/deployment/attention mode mismatch")
    if not frozen and group["completed_updates"] != 2:
        raise ValueError("these adapters must retain exactly two engineering updates")
    for row in group["runs"]:
        if row["origin_model_sha256"] != wanted or row["checkpoint_manifest_sha256"] != checkpoint:
            raise ValueError("actual deployed fingerprint must retain the pinned originating model/checkpoint")
        if row["deployment_mode"] != expected_mode:
            raise ValueError("deployed mode does not match the labelled control")


def collect_report(root, *, partial=False):
    """Collect complete C256 repeats; partial mode never plots incomplete C4k runs."""
    root = Path(root).resolve()
    inputs_path = root / "inputs-01/manifest.json"
    inputs = _json(inputs_path)
    groups, arrays = {}, {}
    for name, path in CORE_PATHS.items():
        groups[name], arrays[name] = load_group(root / path, repetitions=3, context=256)
    for name, group in groups.items():
        validate_group_identity(group, inputs, frozen=name == "frozen", tiled=False, merged=name == "merged")
    reference = request_key(groups["frozen"]["runs"][0]["binding"])
    if any(request_key(group["runs"][0]["binding"]) != reference for group in groups.values()):
        raise ValueError("frozen/unmerged/merged overlays require the identical context, query and random stream")
    comparisons = {"c256_merged_vs_unmerged": [paired_comparison(groups["unmerged"]["runs"][i], groups["merged"]["runs"][i]) for i in range(3)]}
    tiled, _ = load_group(root / "profile-01/frozen/tiled-control", repetitions=1, context=256)
    validate_group_identity(tiled, inputs, frozen=True, tiled=True, merged=False)
    comparisons["c256_frozen_stock_vs_tiled"] = paired_comparison(groups["frozen"]["runs"][0], tiled["runs"][0])
    controls, control_arrays, pending = {}, {}, []
    for name, path in FOUR_K_PATHS.items():
        directory = root / "context4096-01" / path
        if not (directory / "posterior-result.json").is_file():
            pending.append(name)
            continue
        controls[name], control_arrays[name] = load_group(directory, repetitions=1, context=4096)
        validate_group_identity(controls[name], inputs, frozen=name.startswith("frozen"),
            tiled=name.endswith("tiled"), merged=name == "r4_merged_tiled")
    if pending and not partial:
        raise ValueError("incomplete 4,096-context controls: " + ", ".join(pending))
    for label, first, second in (("c4096_frozen_stock_vs_tiled", "frozen_stock", "frozen_tiled"),
                                 ("c4096_r4_unmerged_vs_merged_tiled", "r4_unmerged", "r4_merged_tiled")):
        if first in controls and second in controls:
            comparisons[label] = paired_comparison(controls[first]["runs"][0], controls[second]["runs"][0])
    if controls:
        control_key = request_key(next(iter(controls.values()))["runs"][0]["binding"])
        if any(request_key(group["runs"][0]["binding"]) != control_key for group in controls.values()):
            raise ValueError("4,096-context controls must use matched banks, observations and random streams")
    merge_parity = _json(root / "profile-01/merged/merge-parity.json")
    if len(merge_parity) != 4 or {r["stage"] for r in merge_parity} != set(range(4)):
        raise ValueError("all four conditional merge-parity stages required")
    summary = {"schema": "finetuneforge.inference_engineering_report.v1", "status": "partial_c256_report" if pending else "complete_engineering_report",
        "raw_root": str(root), "input_manifest_sha256": file_sha256(inputs_path),
        "producing_source_commit": inputs["source_commit"], "posterior_draws_per_request": COUNT, "parameters": list(PARAMETERS),
        "c256": groups, "c256_frozen_tiled": tiled, "c4096": controls, "pending_c4096": pending,
        "relative_distribution_comparisons": comparisons, "conditional_merge_parity": merge_parity,
        "memory": memory_evidence(root), "scientifically_qualified": False, "learning_gain_claim": False,
        "two_engineering_updates_only": True, "large_75d_full_posterior_executed": False,
        "c4096_speedup_claim": False,
        "timing_scope": "C256: median and min/max of three same-request timing repetitions; C4096: one observation per control, no repeated speedup claim; excludes model loading and profiler/artifact output",
        "accuracy_scope": "absolute scientific gates remain unqualified; relative execution preservation does not imply posterior accuracy or learning gain",
        "gpu_jobs_submitted_by_report": 0, "report_source_sha256": file_sha256(__file__)}
    return summary, {"c256": {k: v[0] for k, v in arrays.items()}, "c4096": {k: v[0] for k, v in control_arrays.items()}}


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def render_latency(summary, path):
    plt = _plotting()
    keys = ("frozen", "unmerged", "merged")
    labels = ("Frozen", "LoRA-r4\nunmerged", "LoRA-r4\nmerged")
    groups = [summary["c256"][k] for k in keys]
    medians = np.asarray([g["median_seconds"] for g in groups])
    lower = medians - np.asarray([g["range_seconds"][0] for g in groups])
    upper = np.asarray([g["range_seconds"][1] for g in groups]) - medians
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.bar(labels, medians, color=("#2467A6", "#D58A28", "#427D55"), width=.58,
        yerr=np.vstack((lower, upper)), capsize=6, alpha=.85)
    for x, group in enumerate(groups):
        ax.scatter(np.full(3, x), group["seconds"], color="#182A3A", s=22, zorder=3)
        ax.text(x, max(group["seconds"]) + .035 * max(medians), "%.2f s\n[%.2f–%.2f]" %
            (group["median_seconds"], *group["range_seconds"]), ha="center", fontsize=10)
    ax.set(ylabel="Seconds for 50,000 complete vectors", ylim=(0, float(max(g["range_seconds"][1] for g in groups)) * 1.22),
        title="Matched single-GPU request latency\nD4 · 10 observations · 256 worlds · three timing repetitions")
    ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Median with min–max range; same bank/query/seed repeats · two engineering updates, no learning-gain claim", fontsize=9)
    fig.savefig(path, dpi=180); plt.close(fig)


def exact_distributions():
    sigma_v = math.log1p((.03 / .1) ** 2)
    tau_v = math.log1p((.05 / .2) ** 2)
    return {"mu": norm(), "tau": lognorm(s=math.sqrt(tau_v), scale=math.exp(math.log(.2) - .5 * tau_v)),
        "sigma": lognorm(s=math.sqrt(sigma_v), scale=math.exp(math.log(.1) - 10.5 * sigma_v))}


def render_posteriors(samples, path, *, context):
    plt = _plotting()
    if any(v.shape != (COUNT, 4) or not np.isfinite(v).all() for v in samples.values()):
        raise ValueError("plots require complete 50,000-vector requests")
    if context == 256:
        names = (("frozen", "Frozen", "#2467A6"), ("unmerged", "r4 unmerged", "#D58A28"), ("merged", "r4 merged", "#427D55"))
    else:
        names = (("frozen_stock", "Frozen stock", "#2467A6"), ("frozen_tiled", "Frozen tiled", "#589CB0"),
                 ("r4_unmerged", "r4 unmerged", "#D58A28"), ("r4_merged_tiled", "r4 merged+tiled", "#427D55"))
    exact = exact_distributions()
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.1), constrained_layout=True)
    for column, (name, index, label, limits) in enumerate((("mu", 0, "Population mean μ", (-4., 4.)),
            ("tau", 1, "Population spread τ", (.025, .45)), ("sigma", 3, "Measurement-noise SD σ", (.005, .20)))):
        grid = np.linspace(*limits, 700)
        for key, text, color in names:
            values = samples[key][:, index]
            axes[0, column].hist(values, bins=np.linspace(*limits, 81), density=True, histtype="step", lw=1.4, label=text, color=color)
            ordered = np.sort(values)
            axes[1, column].plot(ordered, np.arange(1, COUNT + 1) / COUNT, color=color, lw=1.2)
        axes[0, column].plot(grid, exact[name].pdf(grid), color="#182A3A", lw=1.9, ls="--", label="Exact posterior")
        axes[1, column].plot(grid, exact[name].cdf(grid), color="#182A3A", lw=1.9, ls="--")
        axes[0, column].set(xlim=limits, xlabel=label, ylabel="Density")
        axes[1, column].set(xlim=limits, ylim=(0., 1.), xlabel=label, ylabel="Cumulative probability")
        for row in (0, 1): axes[row, column].grid(alpha=.14)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Neural posterior agreement · context {context:,} · 50,000 complete vectors per curve\nTwo engineering updates only; exact control remains an accuracy diagnostic, no learning-gain claim", fontsize=13)
    fig.savefig(path, dpi=180); plt.close(fig)


def render_memory(memory, path):
    plt = _plotting()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.1, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1.25]})
    left.axis("off")
    old = memory["stock_oom"]
    q1 = next(r for r in memory["tiled_probes"] if r["queries"] == 1)
    left.text(0., .98, "Same large conditional shape\n75 unknowns · 200 observations · 4,096 worlds", va="top", fontsize=12, weight="bold")
    left.text(0., .76, "STOCK: OUT OF MEMORY\nFailed allocation REQUEST: %.2f GiB\nRecorded pre-failure allocated peak: %.2f GB\nNeither number estimates required total capacity." %
        (old["requested_allocation_gib"], old["recorded_peak_allocated_bytes"] / 1e9), va="top", fontsize=10,
        bbox={"facecolor": "#FBE9E5", "edgecolor": "none", "pad": 9})
    left.text(0., .43, "TILED: COMPLETED Q1 CONDITIONAL\nMeasured allocated peak: %.2f GB\nDifferent quantities: no memory-reduction ratio.\nThis is not a complete 75D posterior." % (q1["peak_allocated_bytes"] / 1e9),
        va="top", fontsize=10, bbox={"facecolor": "#E6F0E9", "edgecolor": "none", "pad": 9})
    hours = memory["large_75d_conditional_proxy"]["estimated_single_gpu_hours_range"]
    left.text(0., .10, "Full 75D/50k conditional-cost proxy: %.1f–%.1f GPU-hours\nNot measured end-to-end; setup/IO excluded; full run not executed." % tuple(hours), va="top", fontsize=9)
    rows = memory["tiled_probes"]
    labels = [f"Stage {r['stage']}, Q={r['queries']:,}" for r in rows]
    values = [r["peak_allocated_bytes"] / 1e9 for r in rows]
    right.barh(labels, values, color="#427D55", alpha=.85)
    for i, value in enumerate(values): right.text(value + .15, i, f"{value:.2f} GB", va="center", fontsize=9)
    right.set(xlim=(0., max(values) * 1.23), xlabel="Measured peak allocated memory (decimal GB)",
        title="Completed tiled conditional probes\nQ = query rows; all eight ensemble members")
    right.grid(axis="x", alpha=.15)
    fig.suptitle("Attention capacity evidence · successful peaks and failed requests remain separate", fontsize=13)
    fig.savefig(path, dpi=180); plt.close(fig)


def build_report(root, output, *, partial=False):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU reporting must run outside GPU/Slurm allocations")
    summary, arrays = collect_report(root, partial=partial)
    target = Path(output); target.mkdir(parents=True, exist_ok=False)
    render_latency(summary, target / "latency-context256.png")
    render_posteriors(arrays["c256"], target / "posterior-cdf-context256.png", context=256)
    render_memory(summary["memory"], target / "memory-conditional-support.png")
    if not summary["pending_c4096"]:
        render_posteriors(arrays["c4096"], target / "posterior-cdf-context4096.png", context=4096)
    summary["figures"] = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(target.glob("*.png"))}
    (target / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; compact JSON/PNG only")
    parser.add_argument("--partial", action="store_true", help="Require complete C256 groups, explicitly defer unfinished C4096 controls")
    args = parser.parse_args(argv)
    value = build_report(args.root, args.output, partial=args.partial)
    print(json.dumps({"status": value["status"], "pending_c4096": value["pending_c4096"],
        "figures": list(value["figures"]), "scientifically_qualified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
