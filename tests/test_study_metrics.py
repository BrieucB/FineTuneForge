"""CPU gate component checks; no production posterior/reference is created."""
import numpy as np
import pytest

from posteriorforge.study import metrics as m
from posteriorforge.study import reference as r
from test_study_reference import problem


def samples(spec, n=50000):
    z = np.random.default_rng(23).normal(size=(n, spec.dimension))
    # This is a numerical support fixture, not a posterior inference run.
    z[:, 0] = 2 + .4 * z[:, 0]
    z[:, 1] = np.exp(-1.2 + .2 * z[:, 1])
    z[:, 2] = 1.5 + .1 * z[:, 2]
    z[:, 3] = np.exp(-2.3 + .2 * z[:, 3])
    return z


@pytest.mark.parametrize("count", [1, 49999, 50001])
def test_public_boundary_rejects_non_50000_draws(count):
    spec = problem()
    result = m.assess_samples(samples(spec, count), None, spec=spec)
    assert result["status"] == "invalid_samples"
    assert not result["passed"]


def test_complete_finite_support_is_required_and_missing_reference_never_passes():
    spec = problem()
    values = samples(spec)
    result = m.assess_samples(values, None, spec=spec)
    assert result["status"] == "reference_inconclusive"
    assert result["integrity"]["passed"]
    values[17, -1] = -.1
    result = m.assess_samples(values, None, spec=spec)
    assert result["status"] == "invalid_samples"
    values[17, -1] = np.nan
    assert m.assess_samples(values, None, spec=spec)["status"] == "invalid_samples"


def test_reference_receipt_and_query_binding_are_checked():
    spec = problem()
    observation = np.array([1.5])
    run = r.run_smc_reference(spec, observation, particles=128, moves=2, seed=2, deadline_seconds=3)
    ref = r.qualify_reference_witnesses(spec, observation, [run])
    result = m.assess_samples(samples(spec), ref, spec=spec, observation=observation)
    assert result["status"] == "reference_inconclusive"
    assert not result["reference_qualified"]
    ref["mean"][0] += 1e-5
    with pytest.raises(ValueError, match="receipt"):
        m.assess_samples(samples(spec), ref, spec=spec, observation=observation)


def test_boolean_reference_pass_flag_is_insufficient():
    spec = problem()
    observation = np.array([1.5])
    run = r.run_smc_reference(spec, observation, particles=128, moves=2, seed=3, deadline_seconds=3)
    ref = r.qualify_reference_witnesses(spec, observation, [run])
    q = ref["qualification"]
    q.update(state="qualified", precision=0., retained_refinement_passed=True, independent_passed=True,
             completed_runs=True, moment_gate_passed=True)
    q["moment_precision"] = {"mean_error_reference_sd": [0.] * 4, "sd_error_reference_sd": [0.] * 4}
    q["evidence"] = q["evidence"] * 3
    ref["receipt_sha256"] = r.reference_receipt(ref)
    result = m.assess_samples(samples(spec), ref, spec=spec, observation=observation)
    assert not result["reference_qualified"]
    assert not result["passed"]


def test_reference_moment_uncertainty_can_change_a_borderline_verdict():
    exact = m.moment_intervals([.199], [1.], [0.], [1.], [0.], [0.])
    uncertain = m.moment_intervals([.199], [1.], [0.], [1.], [.004], [.004])
    assert exact["passed"]
    assert not uncertain["passed"]
    assert not uncertain["conclusively_failed"]
    bad = m.moment_intervals([.3], [1.], [0.], [1.], [.004], [.004])
    assert bad["conclusively_failed"]


def test_stable_moment_error_is_retained_separately_from_cdf_precision():
    assert m.reference_moment_errors_valid([.015], [.015], 1)
    assert not m.reference_moment_errors_valid([.021], [.015], 1)
    assert not m.reference_moment_errors_valid([np.nan], [.015], 1)
    assert not m.reference_moment_errors_valid([.015, .015], [.015], 1)
    interval = m.moment_intervals([.17], [1.], [0.], [1.], [.015], [.015])
    assert interval["passed"]
    assert interval["bias_upper"][0] > interval["absolute_bias_reference_sd"][0]
