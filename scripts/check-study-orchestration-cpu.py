#!/usr/bin/env python3
"""Exact analytical control of the file-backed 50,000-vector CPU driver.

This does not evaluate or qualify TabPFN, a transformer, or a numerical reference.
Run with PYTHONPATH=src and a CPU environment. All generated artifacts belong in
an explicit output directory outside Git; existing directories are never replaced.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import lognorm, norm

from posteriorforge.study.artifacts import load_samples
from posteriorforge.study.banks import prepare_bank, save_bank
from posteriorforge.study.science import draw_prior
from posteriorforge.study.spec import DatasetSpec, FamilySpec, ParameterSpec, PriorSpec, ProblemSpec
from posteriorforge.study.workflows import PosteriorRequest, execute_posterior_request, file_sha256, protocol_identity

COUNT = 50_000
LAW = "analytic_zero_sensor_posterior.v1"
TITLE = "Analytical control · CPU orchestration"
EQUATIONS = {
    "prior": "mu~Normal(0,1); tau~Lognormal(physical mean=.2,sd=.05); a|mu,tau~Normal(mu,tau); sigma~Lognormal(physical mean=.1,sd=.03)",
    "observation": "y[0:10]=0; H[10,1]=0; y|a,sigma~Normal(H*a,sigma^2*I)",
    "likelihood": "p(y=0|sigma) proportional to sigma^(-10)",
    "posterior": "log(sigma)|y~Normal(prior_log_mean-10*prior_log_sd^2,prior_log_sd^2); (mu,tau,a)|y retains its joint prior",
    "prior_z_conditionals": "z0,z1,z2 independent Normal(0,1); z3~Normal(-10*prior_log_sd,1)",
}


def build_problem() -> ProblemSpec:
    """A diagnostic hierarchy with always-inferred positive measurement noise."""
    return ProblemSpec("analytical-zero-sensor-control", "diagnostic", (
        ParameterSpec("mu", "population", PriorSpec("normal", 0., 1.)),
        ParameterSpec("tau", "population", PriorSpec("lognormal", .2, .05)),
        ParameterSpec("a", "physical", PriorSpec("normal", "mu", "tau")),
        ParameterSpec("sigma", "noise", PriorSpec("lognormal", .1, .03))),
        (FamilySpec("zero-sensor", "sensor", "sigma"),),
        (DatasetSpec("ten-zero-readings", "zero-sensor", "specimen-0", {"p0": "a"}, matrix=((0.,),) * 10),),
        {"purpose": "exact analytical orchestration control; no model quality claim",
         "zero_response": "deliberately uninformative about physical properties; noise remains inferred"})


def analytic_parameters() -> dict:
    """Convert PHYSICAL log-normal moments and complete the sigma likelihood tilt."""
    log_variance = math.log1p((.03 / .1) ** 2)
    log_sd = math.sqrt(log_variance)
    prior_log_mean = math.log(.1) - .5 * log_variance
    posterior_log_mean = prior_log_mean - 10 * log_variance
    scale = math.exp(-10 * log_variance)
    return {"sigma_prior_log_mean": prior_log_mean, "sigma_log_sd": log_sd,
        "sigma_posterior_log_mean": posterior_log_mean, "sigma_posterior_mean": .1 * scale,
        "sigma_posterior_sd": .03 * scale, "sigma_latent_posterior_mean": -10 * log_sd,
        "mu_a_covariance": 1., "a_variance": 1. + .2 ** 2 + .05 ** 2,
        "tau_log_sd": math.sqrt(math.log1p((.05 / .2) ** 2)),
        "tau_log_mean": math.log(.2) - .5 * math.log1p((.05 / .2) ** 2)}


def analytic_model_sha256() -> str:
    payload = {"law": LAW, "equations": EQUATIONS, "script_sha256": file_sha256(__file__)}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class AnalyticConditionalBackend:
    """Exact posterior conditionals for THIS control; no fitting or neural model."""
    backend_kind = "analytic_control"
    sampling_law = LAW
    source_policy = "analytic_control"

    def __init__(self, *, fail_at_call: int | None = None):
        self.model_sha256 = analytic_model_sha256()
        self.calls = 0
        self.fail_at_call = fail_at_call  # Explicit failure injection for recovery tests only.

    def identity_metadata(self):
        return {"type": "AnalyticConditionalBackend", "backend_kind": self.backend_kind,
            "model_sha256": self.model_sha256, "source_sha256": file_sha256(__file__),
            "sampling_law": self.sampling_law, "source_policy": self.source_policy,
            "scope": "exact zero-sensor analytical control; no transformer evaluated"}

    def verify_identity(self):
        if self.model_sha256 != analytic_model_sha256():
            raise ValueError("analytical equations/source changed during sampling")

    def prepare(self, context_features, context_targets):
        features = np.asarray(context_features)
        targets = np.asarray(context_targets)
        if (features.ndim != 2 or not 10 <= features.shape[1] <= 13
                or targets.shape != (len(features),) or not np.isfinite(features).all() or not np.isfinite(targets).all()):
            raise ValueError("complete finite control-context rows required")
        return features.shape[1] - 10

    def sample(self, stage, query_features, uniforms):
        query = np.asarray(query_features)
        u = np.asarray(uniforms, dtype=np.float64)
        if (type(stage) is not int or not 0 <= stage < 4 or query.shape != (len(u), 10 + stage)
                or u.ndim != 1 or not np.isfinite(query).all() or not np.isfinite(u).all()
                or np.any((u <= 0) | (u >= 1)) or np.any(query[:, :10] != 0)):
            raise ValueError("this analytical conditional supports only ten zero observations")
        self.calls += 1
        if self.calls == self.fail_at_call:
            raise RuntimeError("analytical control recovery interruption")
        shift = analytic_parameters()["sigma_latent_posterior_mean"] if stage == 3 else 0.
        return ndtri(u) + shift


def analytic_backend_factory(model, *, base_directory):
    """Explicit injection: the request's base_checkpoint field is NEVER opened."""
    backend = AnalyticConditionalBackend()
    if model["model_sha256"] != backend.model_sha256:
        raise ValueError("analytical model identity mismatch")
    return backend, {"model_sha256": backend.model_sha256,
        "checkpoint_manifest_sha256": backend.model_sha256, "kind": "analytic_control", "backend_kind": "analytic_control",
        "sampling_law": LAW, "source_policy": "analytic_control", "scientifically_qualified": False,
        "base_checkpoint_used": False, "neural_model_evaluations": 0}


def cdf_sup_error(values, cdf) -> float:
    """Exact empirical-versus-continuous-CDF distance, checking both step sides."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    probabilities = np.asarray(cdf(ordered), dtype=np.float64)
    if ordered.ndim != 1 or not len(ordered) or not np.isfinite(ordered).all() or probabilities.shape != ordered.shape:
        raise ValueError("finite one-dimensional samples and matching CDF required")
    right = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
    left = right - 1. / len(ordered)
    return float(max(np.max(right - probabilities), np.max(probabilities - left)))


def analyze_samples(samples) -> dict:
    """Check marginal laws AND physical dependence against exact mathematics."""
    values = np.asarray(samples)
    spec = build_problem()
    if values.dtype != np.float64 or values.shape != (COUNT, 4) or not spec.support_mask(values).all():
        raise ValueError("exactly 50,000 complete supported float64 vectors required")
    expected = analytic_parameters()
    distributions = {"mu": norm(),
        "tau": lognorm(s=expected["tau_log_sd"], scale=math.exp(expected["tau_log_mean"])),
        "sigma": lognorm(s=expected["sigma_log_sd"], scale=math.exp(expected["sigma_posterior_log_mean"]))}
    marginal = {}
    for name, index in (("mu", 0), ("tau", 1), ("sigma", 3)):
        distribution = distributions[name]
        mean, sd = float(distribution.mean()), float(distribution.std())
        measured_mean, measured_sd = float(values[:, index].mean()), float(values[:, index].std(ddof=1))
        marginal[name] = {"cdf_sup_error": cdf_sup_error(values[:, index], distribution.cdf),
            "sample_mean": measured_mean, "analytic_mean": mean, "sample_sd": measured_sd,
            "analytic_sd": sd, "mean_error_in_analytic_sd": abs(measured_mean - mean) / sd,
            "sd_ratio": measured_sd / sd}
    covariance = float(np.cov(values[:, 0], values[:, 2], ddof=1)[0, 1])
    variance_a = float(values[:, 2].var(ddof=1))
    checks = {name + "_cdf": row["cdf_sup_error"] <= .012 for name, row in marginal.items()}
    checks.update({name + "_mean": row["mean_error_in_analytic_sd"] <= .02 for name, row in marginal.items()})
    checks.update({name + "_sd": .98 <= row["sd_ratio"] <= 1.02 for name, row in marginal.items()})
    checks.update(mu_a_covariance=abs(covariance - expected["mu_a_covariance"]) <= .035,
                  a_variance=abs(variance_a - expected["a_variance"]) <= .035,
                  a_mean=abs(float(values[:, 2].mean())) <= .02 * math.sqrt(expected["a_variance"]))
    return {"sample_count": COUNT, "complete_vectors": True, "dimension": 4, "parameter_order": list(spec.parameter_order),
        "marginals": marginal, "mu_a_covariance": {"sample": covariance, "analytic": 1., "absolute_mc_tolerance": .035},
        "a_variance": {"sample": variance_a, "analytic": expected["a_variance"], "absolute_mc_tolerance": .035},
        "monte_carlo_tolerances": {"cdf_sup_error": .012, "mean_error_in_analytic_sd": .02, "sd_ratio": [.98, 1.02]},
        "checks": checks, "control_checks_passed": all(checks.values()),
        "scope": "fixed-seed mathematical control, not global calibration or numerical-reference qualification"}


def make_figure(samples, prior_sigma, output, metrics):
    """Publication-style diagnostic PNG; all 50,000 posterior rows are plotted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    p = analytic_parameters()
    blue, orange, ink = "#2467A6", "#D58A28", "#182A3A"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": ink, "text.color": ink, "xtick.color": ink,
        "ytick.color": ink, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(TITLE + "\n50,000 complete posterior vectors · exact analytical backend · no transformer evaluated", fontsize=15)
    mu, a, sigma = samples[:, 0], samples[:, 2], samples[:, 3]
    axes[0].hist(mu, bins=65, density=True, color=blue, alpha=.65, label="Posterior: 50,000 draws")
    grid = np.linspace(-4., 4., 500)
    axes[0].plot(grid, norm.pdf(grid), color=ink, lw=2, label="Exact posterior = prior")
    axes[0].set(xlabel="Population mean μ", ylabel="Density", xlim=(-4., 4.), title="Uninformed properties stay uncertain")
    axes[0].legend(frameon=False, fontsize=8)
    limit = float(lognorm.ppf(.999, s=p["sigma_log_sd"], scale=math.exp(p["sigma_prior_log_mean"])))
    bins = np.linspace(0., max(float(np.max(prior_sigma)), float(np.max(sigma))), 90)
    axes[1].hist(prior_sigma, bins=bins, density=True, color=orange, alpha=.4, label="Prior: parameter-only draws")
    axes[1].hist(sigma, bins=bins, density=True, color=blue, alpha=.65, label="Posterior: 50,000 draws")
    grid = np.linspace(1e-5, limit, 600)
    axes[1].plot(grid, lognorm.pdf(grid, s=p["sigma_log_sd"], scale=math.exp(p["sigma_prior_log_mean"])), color=orange, lw=1.7, ls="--", label="Exact prior")
    axes[1].plot(grid, lognorm.pdf(grid, s=p["sigma_log_sd"], scale=math.exp(p["sigma_posterior_log_mean"])), color=ink, lw=2, label="Exact posterior")
    axes[1].set(xlabel="Measurement-noise SD σ", ylabel="Density", xlim=(0., limit), title="Ten zero readings update unknown noise")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].scatter(mu, a, s=.8, alpha=.05, color=blue, linewidths=0, rasterized=True)
    axes[2].plot([-4, 4], [-4, 4], ls="--", lw=1.4, color=ink)
    axes[2].set(xlabel="Population mean μ", ylabel="Specimen property a", xlim=(-4., 4.), ylim=(-4., 4.),
        title="Hierarchy retains shared uncertainty")
    axes[2].text(.03, .97, "All 50,000 points\nCov(μ, a): %.4f (exact: 1)\nVar(a): %.4f (exact: 1.0425)" %
        (metrics["mu_a_covariance"]["sample"], metrics["a_variance"]["sample"]),
        transform=axes[2].transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    for ax in axes:
        ax.grid(alpha=.12)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def run_control(output, *, seed=20261117, microbatch_size=4096, make_plot=True) -> dict:
    """Run only this analytical diagnostic, keeping raw artifacts outside Git."""
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("analytical CPU control must not run inside Slurm")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative control seed required")
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("generated control artifacts must be outside the Git worktree")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    spec = build_problem()
    bank_receipt = save_bank(prepare_bank(spec, 16, seed=seed + 1), output / "bank")
    observation = np.zeros(10, dtype=np.float64)
    np.save(output / "observation.npy", observation, allow_pickle=False)
    request = PosteriorRequest("bank", bank_receipt["manifest_sha256"], "observation.npy", file_sha256(output / "observation.npy"),
        {"base_checkpoint": "unused-by-explicit-analytical-control-callback.ckpt", "model_sha256": analytic_model_sha256()},
        protocol_identity()["protocol_sha256"], context_size=16, seed=seed, microbatch_size=microbatch_size)
    _json(output / "request.json", request.to_dict())
    record = execute_posterior_request(request, output=output / "posterior-run", base_directory=output,
        allow_unqualified_cpu_diagnostic=True, backend_factory=analytic_backend_factory)
    actual_spec, samples, actual_y, artifact = load_samples(output / "posterior-run" / "posterior",
        expected_manifest_sha256=record["artifact_manifest_sha256"])
    if actual_spec.fingerprint != spec.fingerprint or not np.array_equal(actual_y, observation):
        raise ValueError("analytical control artifact/query mismatch")
    metrics = analyze_samples(samples)
    # Parameter-only draws for a PRIOR histogram; these never call the simulator.
    prior_sigma = draw_prior(spec, COUNT, seed=seed + 2)[:, 3]
    np.savez(output / "prior-plot-samples.npz", sigma=prior_sigma)
    figure = output / "posterior-control.png"
    if make_plot:
        make_figure(samples, prior_sigma, figure, metrics)
    study_dir = Path(__import__("posteriorforge.study.science", fromlist=["__file__"]).__file__).parent
    versions = {}
    for name in ("numpy", "scipy", "matplotlib"):
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = None
    metrics.update(schema="finetuneforge.analytic_orchestration_control.v1", title=TITLE,
        status="analytical_control_passed" if metrics["control_checks_passed"] else "analytical_control_failed",
        equations=EQUATIONS, analytic_parameters=analytic_parameters(), model_sha256=analytic_model_sha256(),
        backend_kind="analytic_control", sampling_law=LAW, source_policy="analytic_control",
        transformer_evaluated=False, numerical_reference_claim=False, scientifically_qualified=False,
        seed=seed, bank_seed=seed + 1, prior_plot_seed=seed + 2, prior_plot_count=COUNT,
        prior_plot_forward_evaluations=0, observation_generation_forward_evaluations=0,
        observation="ten deliberately fixed zeros, independent of the context bank; no hidden query truth supplied",
        context_size=16, posterior_draws=COUNT, microbatch_size=microbatch_size,
        bank_cost=bank_receipt["cost"], posterior_sampling_forward_evaluations=0,
        request_sha256=request.fingerprint, artifact_manifest_sha256=record["artifact_manifest_sha256"],
        run_manifest_sha256=record["run_manifest_sha256"], gpu_jobs_submitted=0, allocated_gpu_hours=0,
        elapsed_cpu_wall_seconds=time.monotonic() - started, python=platform.python_version(), versions=versions,
        source_sha256={**{"script": file_sha256(__file__)}, **{name: file_sha256(study_dir / name) for name in
            ("workflows.py", "sampling.py", "science.py", "spec.py", "banks.py", "artifacts.py")}},
        figure=None if not make_plot else {"file": figure.name, "sha256": file_sha256(figure)},
        scope="exact analytic control of conditioning, physical dependence, 50,000 complete vectors and file-backed orchestration; not TabPFN quality, transfer or performance qualification")
    _json(output / "metrics.json", metrics)
    files = {str(path.relative_to(output)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != ".lock"}
    receipt = {"schema": "finetuneforge.analytic_orchestration_receipt.v1", "output": str(output),
        "files": files, "metrics_sha256": file_sha256(output / "metrics.json"), "script_sha256": file_sha256(__file__),
        "control_checks_passed": metrics["control_checks_passed"], "posterior_draws": COUNT,
        "scientifically_qualified": False, "gpu_jobs_submitted": 0}
    _json(output / "receipt.json", receipt)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh artifact directory outside Git")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--microbatch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    metrics = run_control(args.output, seed=args.seed, microbatch_size=args.microbatch_size)
    print(json.dumps({"status": metrics["status"], "posterior_draws": COUNT,
        "marginal_cdf_errors": {k: v["cdf_sup_error"] for k, v in metrics["marginals"].items()},
        "figure": str(args.output / "posterior-control.png"), "gpu_jobs_submitted": 0}, sort_keys=True))
    return 0 if metrics["control_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
