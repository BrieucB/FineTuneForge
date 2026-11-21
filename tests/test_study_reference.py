"""Mathematical/component CPU checks, not production reference qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from posteriorforge.study.spec import DatasetSpec, FamilySpec, ParameterSpec, PriorSpec, ProblemSpec
from posteriorforge.study import reference as r


def problem(physics="scalar", *, repeats=1):
    positive = physics not in ("quadratic", "scalar", "oscillation")
    p = (ParameterSpec("mu", "population", PriorSpec("lognormal" if positive else "normal", 2., .4)),
         ParameterSpec("spread", "population", PriorSpec("lognormal", .3, .08)),
         ParameterSpec("theta", "physical", PriorSpec("lognormal" if positive else "normal", "mu", "spread")),
         ParameterSpec("sigma", "noise", PriorSpec("lognormal", .1, .02)))
    controls = tuple(np.linspace(-1 if physics == "quadratic" else 0, 2, 10))
    data = {
        "sensor": ("p0", {}, (), tuple((v,) for v in np.linspace(.1, 1., 10))),
        "quadratic": ("a", {"b": .25, "d": -.2}, controls, ()),
        "scalar": ("a", {}, (), ()),
        "saturation": ("A", {"d": .2, "K": 1., "h": 2.}, controls, ()),
        "decay": ("A_fast", {"d": .1, "tau_fast": .5, "A_slow": .7, "delta_tau": 1.}, controls, ()),
        "oscillation": ("A", {"d": -.1, "gamma": .2, "omega": 2., "B": .4, "v": .1}, controls, ()),
        "spring": ("k", {"c": .2}, controls, ()),
        "contact": ("k", {"u0": .5, "d": -.1}, controls, ()),
    }
    coefficient, known, u, matrix = data[physics]
    return ProblemSpec("ref-" + physics, "diagnostic", p, (FamilySpec("f", physics, "sigma"),),
        tuple(DatasetSpec(str(i), "f", "s", {coefficient: "theta"}, u, known, matrix) for i in range(repeats)))


@pytest.mark.parametrize("physics", ("sensor", "quadratic", "scalar", "saturation", "decay", "oscillation", "spring", "contact"))
def test_independent_response_equations(physics):
    spec = problem(physics)
    theta = np.array([[2., .3, 1.5, .1]])
    actual = r.expected_observations(spec, theta)[0]
    u = np.asarray(spec.datasets[0].controls)
    expected = {
        "sensor": lambda: 1.5 * np.linspace(.1, 1., 10),
        "quadratic": lambda: 1.5 * u + .25 * u ** 2 - .2,
        "scalar": lambda: np.array([1.5]),
        "saturation": lambda: .2 + 1.5 * u ** 2 / (1 + u ** 2),
        "decay": lambda: .1 + 1.5 * np.exp(-u / .5) + .7 * np.exp(-u / 1.5),
        "oscillation": lambda: -.1 + np.exp(-.2 * u) * (1.5 * np.cos(2 * u) + .4 * np.sin(2 * u)) + .1 * u,
        "spring": lambda: np.array([brentq(lambda x: 1.5 * x + .2 * x ** 3 - value, -10., 10.) for value in u]),
        "contact": lambda: -.1 + 1.5 * np.maximum(u - .5, 0),
    }[physics]()
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_conditional_prior_density_uses_physical_lognormal_moments():
    spec = problem()
    theta = np.array([[2., .3, 1.5, .1], [1.1, .2, 1.2, .13]])
    expected = []
    for mu, spread, physical, sigma in theta:
        result = norm.logpdf(mu, 2., .4) + norm.logpdf(physical, mu, spread)
        for value, mean, sd in ((spread, .3, .08), (sigma, .1, .02)):
            v = np.log1p((sd / mean) ** 2)
            result += norm.logpdf(np.log(value), np.log(mean) - v / 2, np.sqrt(v)) - np.log(value)
        expected.append(result)
    np.testing.assert_allclose(r.log_prior(spec, theta), expected)
    theta[0, -1] = -1
    assert r.log_prior(spec, theta)[0] == -np.inf


def test_repeated_readings_share_noise_but_not_error_realizations():
    spec = problem(repeats=3)
    theta = np.array([[2., .3, 1.5, .1]])
    observed = np.array([1.4, 1.6, 1.7])
    expected = norm.logpdf(observed, 1.5, .1).sum()
    assert r.log_likelihood(spec, theta, observed)[0] == pytest.approx(expected)
    assert r.log_joint(spec, theta, observed)[0] == pytest.approx(expected + r.log_prior(spec, theta)[0])


def test_reference_run_is_retained_and_never_self_qualified():
    spec = problem()
    run = r.run_smc_reference(spec, np.array([1.5]), particles=128, moves=2, seed=12, deadline_seconds=3)
    assert run.samples.shape == (128, 4)
    assert not run.samples.flags.writeable
    assert run.diagnostics["state"] in ("sampled_unqualified", "incomplete")
    assert run.diagnostics["local_forward_calls"] >= 128
    assert spec.support_mask(run.samples).all()
    ref = r.qualify_reference_witnesses(spec, np.array([1.5]), [run])
    assert ref["qualification"]["state"] == "inconclusive"
    assert ref["qualification"]["precision"] == 1.
    assert ref["receipt_sha256"] == r.reference_receipt(ref)


def test_reference_precision_needs_refinement_and_independent_evidence():
    spec = problem()
    q = np.array([1.5])
    first = r.run_smc_reference(spec, q, particles=64, moves=2, seed=2, deadline_seconds=3)
    second = r.run_smc_reference(spec, q, particles=128, moves=2, seed=2, deadline_seconds=3)
    # Merely adding retained rows or a second method name cannot waive precision.
    ref = r.qualify_reference_witnesses(spec, q, [first, second], independent=first)
    assert ref["qualification"]["state"] == "inconclusive"
    assert not ref["qualification"]["independent_passed"]
    with pytest.raises(ValueError, match="binding"):
        r.qualify_reference_witnesses(spec, np.array([1.6]), [first])
    with pytest.raises(ValueError, match="source"):
        r.qualify_reference_witnesses(spec, q, [replace(first, source_sha256="0" * 64)])


def test_optional_nuts_never_silently_falls_back():
    import importlib.util
    if importlib.util.find_spec("numpyro") is not None:
        pytest.skip("real optional NUTS test is exercised in the dedicated reference environment")
    with pytest.raises(r.ReferenceUnavailable, match="NumPyro"):
        r.run_nuts_reference(problem(), np.array([1.5]), warmup=2, draws=2, chains=2)


def test_predefined_projection_inventory_is_fixed_and_full_vector():
    directions = r.predefined_directions(7)
    assert directions.shape == (39, 7)
    np.testing.assert_array_equal(directions[:7], np.eye(7))
    assert np.all(directions[7:] != 0)
    assert len(np.unique(r.predefined_directions(4)[4:], axis=0)) == 32
    assert np.all(directions[7:, 0] > 0)
    np.testing.assert_allclose(np.sum(directions[7:] ** 2, axis=1), 1.)
    np.testing.assert_array_equal(directions, r.predefined_directions(7))
    with pytest.raises(ValueError):
        r.predefined_directions(2, scales=[1, 0])


def test_actual_tiny_nuts_cpu_runner_returns_unqualified_samples():
    pytest.importorskip("numpyro")
    spec = problem()
    run = r.run_nuts_reference(spec, np.array([1.5]), warmup=16, draws=16, chains=2, seed=41)
    assert run.method == "numpyro_nuts"
    assert run.samples.shape == (32, spec.dimension)
    assert spec.support_mask(run.samples).all()
    assert run.diagnostics["platform"] == "cpu"
    assert run.diagnostics["noncentred"] is True
    assert run.diagnostics["local_forward_calls"] is None
    assert run.diagnostics["state"] == "sampled_unqualified"
    assert r.qualify_reference_witnesses(spec, np.array([1.5]), [run])["qualification"]["state"] == "inconclusive"


def test_reference_seed_retains_full_entropy():
    assert not np.array_equal(r._seed_words(7), r._seed_words(7 + 2**40))
    np.testing.assert_array_equal(r._seed_words(7), r._seed_words(7))
    with pytest.raises(ValueError):
        r._seed_words(True)


@pytest.mark.parametrize("physics", ("sensor", "quadratic", "scalar", "saturation", "decay", "oscillation", "spring", "contact"))
def test_independent_reference_matches_bank_provider_contract(physics):
    from posteriorforge.study import science
    spec = problem(physics)
    theta = science.draw_prior(spec, 16, seed=701)
    generated, _ = science.simulate(spec, theta, seed=702, noisy=False)
    np.testing.assert_allclose(generated, r.expected_observations(spec, theta), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(science.log_prior(spec, theta), r.log_prior(spec, theta), rtol=1e-11, atol=1e-11)


def test_independent_reference_handles_mixed_cross_family_sharing():
    from posteriorforge.study import science, catalogue
    spec = catalogue.generate_setup(7001, split="diagnostic", physics=["sensor", "quadratic", "oscillation"], sharing="cross", size_band="small")
    theta = science.draw_prior(spec, 16, seed=703)
    generated, cost = science.simulate(spec, theta, seed=704, noisy=False)
    np.testing.assert_allclose(generated, r.expected_observations(spec, theta), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(science.log_prior(spec, theta), r.log_prior(spec, theta), rtol=1e-11, atol=1e-11)
    assert cost["forward_evaluations"] < cost["requested_local_vectors"]


def test_smc_cannot_normalize_an_all_nonfinite_likelihood(monkeypatch):
    monkeypatch.setattr(r, "log_likelihood", lambda spec, theta, observation: np.full(len(theta), -np.inf))
    with pytest.raises(r.ReferenceUnavailable, match="nonfinite"):
        r.run_smc_reference(problem(), np.array([1.5]), particles=16, moves=1)
