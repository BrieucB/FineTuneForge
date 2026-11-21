"""Genuine exact-posterior CPU orchestration control, never a transformer claim."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import lognorm

from posteriorforge.study import science, workflows
from posteriorforge.study.artifacts import load_samples
from posteriorforge.study.workflows import PosteriorRequest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-study-orchestration-cpu.py"
spec = importlib.util.spec_from_file_location("check_study_orchestration_cpu_control", SCRIPT)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_closed_form_sigma_update_follows_bayes_with_physical_moments():
    p = control.analytic_parameters()
    variance = np.log1p((.03 / .1) ** 2)
    assert p["sigma_log_sd"] ** 2 == pytest.approx(variance)
    assert p["sigma_posterior_log_mean"] == pytest.approx(np.log(.1) - 10.5 * variance)
    prior = lognorm(s=p["sigma_log_sd"], scale=np.exp(p["sigma_prior_log_mean"]))
    posterior = lognorm(s=p["sigma_log_sd"], scale=np.exp(p["sigma_posterior_log_mean"]))
    sigma = np.geomspace(.005, .5, 31)
    # Exact density ratio is the sigma^-10 likelihood divided by its known
    # log-normal prior expectation. This checks the posterior independently of
    # the conditional sampler and the sampled moments.
    log_evidence = -10 * p["sigma_prior_log_mean"] + 50 * variance
    np.testing.assert_allclose(posterior.logpdf(sigma), prior.logpdf(sigma) - 10 * np.log(sigma) - log_evidence, rtol=2e-14, atol=2e-14)
    assert posterior.mean() == pytest.approx(.1 * 1.09 ** -10)
    assert posterior.std() == pytest.approx(.03 * 1.09 ** -10)
    assert p["a_variance"] == pytest.approx(1.0425)
    problem = control.build_problem()
    assert problem.parameter_order == ("mu", "tau", "a", "sigma")
    assert problem.roles[-1] == "noise" and np.array_equal(problem.datasets[0].matrix, np.zeros((10, 1)))


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("analytical-orchestration") / "control"
    return root, control.run_control(root, make_plot=False)


def test_file_backed_50000_control_matches_exact_marginals_and_shared_uncertainty(completed):
    root, metrics = completed
    actual_spec, samples, observation, manifest = load_samples(root / "posterior-run/posterior",
        expected_manifest_sha256=metrics["artifact_manifest_sha256"])
    assert samples.shape == (50000, 4) and samples.dtype == np.float64
    assert actual_spec.support_mask(samples).all() and np.array_equal(observation, np.zeros(10))
    assert metrics["control_checks_passed"] and all(metrics["checks"].values())
    assert metrics["transformer_evaluated"] is False and metrics["scientifically_qualified"] is False
    assert metrics["numerical_reference_claim"] is False and metrics["gpu_jobs_submitted"] == 0
    assert metrics["bank_cost"]["forward_evaluations"] == 16
    assert metrics["posterior_sampling_forward_evaluations"] == metrics["prior_plot_forward_evaluations"] == 0
    assert manifest["provenance"]["sampling_law"] == control.LAW
    assert metrics["sampling_law"] == control.LAW
    with np.load(root / "posterior-run/recovery.npz", allow_pickle=False) as recovery:
        assert recovery["samples"].shape == (50000, 4)
    receipt = json.loads((root / "receipt.json").read_text())
    for name, item in receipt["files"].items():
        path = root / name
        assert sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert path.stat().st_size == item["bytes"]


def test_control_rejects_lost_sharing_and_frozen_prior_noise(completed):
    root, _ = completed
    samples = load_samples(root / "posterior-run/posterior")[1]
    lost_sharing = samples.copy()
    lost_sharing[:, 2] -= lost_sharing[:, 0]
    checks = control.analyze_samples(lost_sharing)["checks"]
    assert not checks["mu_a_covariance"] and not checks["a_variance"]
    wrong_noise = samples.copy()
    wrong_noise[:, 3] *= np.exp(10 * control.analytic_parameters()["sigma_log_sd"] ** 2)
    assert not control.analyze_samples(wrong_noise)["checks"]["sigma_cdf"]
    with pytest.raises(ValueError, match="50,000"):
        control.analyze_samples(samples[:-1])


def test_backend_rejects_nonzero_observations_and_never_opens_base_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("analytical injection must never load neural weights")
    monkeypatch.setattr(workflows, "make_posterior_backend", forbidden)
    backend, metadata = control.analytic_backend_factory({"model_sha256": control.analytic_model_sha256(),
        "base_checkpoint": "not-a-real-file"}, base_directory="missing-directory")
    assert metadata["base_checkpoint_used"] is False and metadata["neural_model_evaluations"] == 0
    uniforms = np.array([.25, .5, .75])
    for j in range(4):
        stage = backend.prepare(np.zeros((16, 10 + j)), np.zeros(16))
        result = backend.sample(stage, np.zeros((3, 10 + j)), uniforms)
        offset = control.analytic_parameters()["sigma_latent_posterior_mean"] if j == 3 else 0.
        np.testing.assert_allclose(result, control.ndtri(uniforms) + offset)
    with pytest.raises(ValueError, match="zero observations"):
        backend.sample(0, np.ones((3, 10)), uniforms)


def test_analytical_recovery_retains_complete_vectors_and_replays_exactly(completed, tmp_path):
    root, _ = completed
    request = PosteriorRequest.from_dict(json.loads((root / "request.json").read_text()))
    output = tmp_path / "interrupted"
    def interrupted(model, *, base_directory):
        _, metadata = control.analytic_backend_factory(model, base_directory=base_directory)
        return control.AnalyticConditionalBackend(fail_at_call=6), metadata
    with pytest.raises(RuntimeError, match="recovery interruption"):
        workflows.execute_posterior_request(request, output=output, base_directory=root,
            allow_unqualified_cpu_diagnostic=True, backend_factory=interrupted)
    assert not (output / "posterior").exists()
    with np.load(output / "recovery.npz", allow_pickle=False) as saved:
        assert saved["samples"].shape == (4096, 4)
    record = workflows.execute_posterior_request(request, output=output, base_directory=root,
        allow_unqualified_cpu_diagnostic=True, backend_factory=control.analytic_backend_factory, resume=True)
    assert record["completed_draws"] == 50000
    np.testing.assert_array_equal(load_samples(output / "posterior")[1], load_samples(root / "posterior-run/posterior")[1])


def test_prior_plot_sampling_adds_no_forward_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("parameter-only prior must not call physics")
    monkeypatch.setattr(science, "_response", forbidden)
    values = control.draw_prior(control.build_problem(), 8, seed=3)
    assert values.shape == (8, 4) and control.build_problem().support_mask(values).all()


def test_figure_is_generated_from_all_complete_posterior_rows(completed, tmp_path):
    pytest.importorskip("matplotlib")
    root, metrics = completed
    samples = load_samples(root / "posterior-run/posterior")[1]
    with np.load(root / "prior-plot-samples.npz", allow_pickle=False) as saved:
        prior_sigma = saved["sigma"]
    output = tmp_path / "posterior.png"
    control.make_figure(samples, prior_sigma, output, metrics)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and output.stat().st_size > 10000
