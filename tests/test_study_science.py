"""Focused mathematics/provider tests; no model downloads or posterior runs."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import norm, lognorm

from posteriorforge.study.spec import PriorSpec, ParameterSpec, FamilySpec, DatasetSpec, ProblemSpec
from posteriorforge.study.science import draw_prior, encode_prior, decode_prior, log_abs_det_decode, log_prior, simulate, _response


def scalar_problem(repeated=False):
    pars = (ParameterSpec("mu", "population", PriorSpec("normal", .5, .4)),
            ParameterSpec("spread", "population", PriorSpec("lognormal", .2, .05)),
            ParameterSpec("a", "physical", PriorSpec("normal", "mu", "spread")),
            ParameterSpec("sigma", "noise", PriorSpec("lognormal", .1, .02)))
    data = [DatasetSpec("reading-0", "f", "specimen", {"a": "a"})]
    if repeated:
        data.append(DatasetSpec("reading-1", "f", "specimen", {"a": "a"}))
    return ProblemSpec("toy", "diagnostic", pars, (FamilySpec("f", "scalar", "sigma"),), tuple(data))


def test_causal_roundtrip_density_and_jacobian():
    p = scalar_problem()
    z = np.random.default_rng(9).normal(size=(256, p.dimension))
    theta = decode_prior(p, z)
    np.testing.assert_allclose(encode_prior(p, theta), z, rtol=2e-13, atol=2e-13)
    expected = norm.logpdf(theta[:, 0], .5, .4)
    for col, mean, sd in ((1, .2, .05), (3, .1, .02)):
        sl = np.sqrt(np.log1p((sd / mean)**2))
        ml = np.log(mean) - .5 * sl**2
        expected += lognorm.logpdf(theta[:, col], sl, scale=np.exp(ml))
    expected += norm.logpdf(theta[:, 2], theta[:, 0], theta[:, 1])
    np.testing.assert_allclose(log_prior(p, theta), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(log_prior(p, theta), norm.logpdf(z).sum(axis=1) - log_abs_det_decode(p, z), atol=1e-12)


def test_log_normal_physical_moments():
    p = ProblemSpec("moments", "diagnostic", (
        ParameterSpec("a", "physical", PriorSpec("lognormal", 2., .3)),
        ParameterSpec("sigma", "noise", PriorSpec("lognormal", .1, .02))),
        (FamilySpec("f", "scalar", "sigma"),), (DatasetSpec("d", "f", "s", {"a": "a"}),))
    values = draw_prior(p, 50000, 123)
    assert abs(values[:, 0].mean() - 2.) < .008
    assert abs(values[:, 0].std() - .3) < .006


def test_prior_only_performs_no_response_call_and_is_deterministic():
    p = scalar_problem()
    with patch("posteriorforge.study.science._response", side_effect=AssertionError("forward called")):
        a = draw_prior(p, 16, 3)
        b = draw_prior(p, 16, 3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, draw_prior(p, 16, 4))


def test_invalid_physical_values_are_not_clipped_or_discarded():
    p = scalar_problem()
    x = draw_prior(p, 8, 1)
    x[0, 3] = -1
    assert log_prior(p, x)[0] == -np.inf
    with pytest.raises(ValueError):
        encode_prior(p, x)
    with pytest.raises(ValueError):
        simulate(p, x)


def test_repeated_measurements_reuse_physics_but_not_errors():
    p = scalar_problem(repeated=True)
    theta = draw_prior(p, 32, 2)
    clean, cost = simulate(p, theta, 3, noisy=False)
    np.testing.assert_array_equal(clean[:, 0], clean[:, 1])
    assert cost["forward_evaluations"] == 32
    assert cost["requested_local_vectors"] == 64
    assert cost["reused_local_vectors"] == 32
    assert cost["prefix_forward_evaluations"] == list(range(33))
    y, cost = simulate(p, theta, 3)
    assert not np.array_equal(y[:, 0], y[:, 1])
    assert cost["noise_values_generated"] == 64
    small, _ = simulate(p, theta[:5], 3)
    np.testing.assert_array_equal(small, y[:5])


def test_dataset_order_permutation_preserves_noisy_dataset_values():
    p = scalar_problem(repeated=True)
    theta = draw_prior(p, 12, 3)
    y, _ = simulate(p, theta, 4)
    q = replace(p, datasets=tuple(reversed(p.datasets)))
    reordered, _ = simulate(q, theta, 4)
    np.testing.assert_array_equal(reordered, y[:, ::-1])


def test_conditional_measurement_noise_has_declared_gaussian_scale():
    p = scalar_problem(repeated=True)
    theta = draw_prior(p, 10000, 10)
    y, _ = simulate(p, theta, 11)
    residual = (y - theta[:, 2, None]) / theta[:, 3, None]
    assert abs(residual.mean()) < .03
    assert abs(residual.std() - 1.) < .03
    assert abs(np.corrcoef(residual.T)[0, 1]) < .03


@pytest.mark.parametrize("physics,c", [
    ("quadratic", {"a": 2., "b": 3., "d": 1.}),
    ("saturation", {"d": .2, "A": 2., "K": .5, "h": 2}),
    ("decay", {"d": .1, "A_fast": 1., "tau_fast": .3, "A_slow": .5, "delta_tau": 1.}),
    ("oscillation", {"d": .1, "A": 1., "B": .5, "gamma": .1, "omega": 2., "v": .02}),
    ("spring", {"k": 2., "c": .4}),
    ("contact", {"d": .2, "k": 3., "u0": .5}),
])
def test_response_laws_are_complete_and_follow_physics(physics, c):
    u = np.linspace(0., 2., 10)
    y = _response(physics, c, u, ())
    assert y.shape == (10,) and np.isfinite(y).all()
    if physics == "spring":
        np.testing.assert_allclose(c["k"] * y + c["c"] * y**3, u, rtol=3e-14, atol=3e-14)
    elif physics == "quadratic":
        np.testing.assert_allclose(y, 2*u + 3*u*u + 1.)
    elif physics == "contact":
        np.testing.assert_array_equal(y[u < .5], np.full((u < .5).sum(), .2))
    elif physics == "decay":
        assert np.all(np.diff(y) < 0)
    elif physics == "saturation":
        assert np.all(np.diff(y) > 0) and y[-1] < 2.2


def test_sensor_channels_and_scalar_recipe():
    h = np.arange(20, dtype=float).reshape(10, 2) / 20
    np.testing.assert_allclose(_response("sensor", {"p0": 2., "p1": 3.}, (), h), h @ [2., 3.])
    np.testing.assert_array_equal(_response("scalar", {"a": 2.}, (), ()), [2.])


@pytest.mark.parametrize("physics,c", [("spring", {"k": -1., "c": 1.}), ("decay", {"d": 0., "A_fast": 1., "tau_fast": 0.}), ("contact", {"d": 0., "k": 1., "u0": -1.})])
def test_invalid_physics_fails_instead_of_repair(physics, c):
    with pytest.raises(ValueError):
        _response(physics, c, np.linspace(0, 1, 10), ())


def test_per_parameter_jacobian_matches_sum_and_causal_finite_differences():
    p = scalar_problem()
    z = np.random.default_rng(7).normal(size=(8, p.dimension))
    terms = log_abs_det_decode(p, z, per_parameter=True)
    assert terms.shape == z.shape and terms.dtype == np.float64
    np.testing.assert_allclose(terms.sum(axis=1), log_abs_det_decode(p, z), rtol=2e-15, atol=2e-15)
    theta = decode_prior(p, z)
    np.testing.assert_allclose(encode_prior(p, theta), z, atol=2e-13)
    eps = 1e-6
    for j in range(p.dimension):
        plus, minus = z[:1].copy(), z[:1].copy()
        plus[0, j] += eps; minus[0, j] -= eps
        derivative = (decode_prior(p, plus)[0, j] - decode_prior(p, minus)[0, j]) / (2 * eps)
        np.testing.assert_allclose(np.log(abs(derivative)), terms[0, j], rtol=3e-8, atol=3e-8)
    with pytest.raises(ValueError, match="boolean"):
        log_abs_det_decode(p, z, per_parameter=1)
