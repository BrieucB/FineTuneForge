from dataclasses import replace
import json

import numpy as np
import pytest

from posteriorforge.study import banks, science
from posteriorforge.study.banks import load_bank, prepare_bank, save_bank
from posteriorforge.study.ordering import OrderingMap, apply_ordering, reorder_bank, reorder_problem
from posteriorforge.study.spec import DatasetSpec, FamilySpec, ParameterSpec, PriorSpec, ProblemSpec


def problem():
    """Small diagnostic only: two batches, repeated and cross-family links."""
    parameters = (
        ParameterSpec("mu", "population", PriorSpec("normal", 0., 1.)),
        ParameterSpec("within", "population", PriorSpec("lognormal", .3, .06)),
        ParameterSpec("between", "population", PriorSpec("lognormal", .2, .04)),
        ParameterSpec("batch0", "batch", PriorSpec("normal", "mu", "between")),
        ParameterSpec("batch1", "batch", PriorSpec("normal", "mu", "between")),
        ParameterSpec("x0", "physical", PriorSpec("normal", "batch0", "within")),
        ParameterSpec("x1", "physical", PriorSpec("normal", "batch1", "within")),
        ParameterSpec("sigma_curve", "noise", PriorSpec("lognormal", .1, .02)),
        ParameterSpec("sigma_scalar", "noise", PriorSpec("lognormal", .15, .03)),
    )
    families = (FamilySpec("curve", "quadratic", "sigma_curve"), FamilySpec("readout", "scalar", "sigma_scalar"))
    controls = tuple(np.linspace(-1., 1., 10))
    datasets = (
        DatasetSpec("c0", "curve", "s0", {"a": "x0"}, controls, {"b": .2, "d": .1}),
        DatasetSpec("c1", "curve", "s1", {"a": "x1"}, controls, {"b": .2, "d": .1}),
        DatasetSpec("c0_repeat", "curve", "s0", {"a": "x0"}, controls, {"b": .2, "d": .1}),
        DatasetSpec("r0", "readout", "s0", {"a": "x0"}),
        DatasetSpec("r1", "readout", "s1", {"a": "x1"}),
    )
    return ProblemSpec("ordering-diagnostic", "diagnostic", parameters, families, datasets,
        {"fixture": "no catalogue queries or posterior samples", "nested": {"value": [1, 2]}})


def test_seeded_reordering_preserves_identities_links_and_parent_order():
    source = problem()
    outcomes = set()
    for seed in range(8):
        target, mapping = reorder_problem(source, seed)
        assert target.id == source.id and target.split == source.split
        assert target.to_dict()["metadata"] == source.to_dict()["metadata"]
        assert {p.name: p for p in target.parameters} == {p.name: p for p in source.parameters}
        assert {f.id: f for f in target.families} == {f.id: f for f in source.families}
        assert {d.id: d for d in target.datasets} == {d.id: d for d in source.datasets}
        positions = {p.name: j for j, p in enumerate(target.parameters)}
        for j, parameter in enumerate(target.parameters):
            for parent in (parameter.prior.mean, parameter.prior.sd):
                if isinstance(parent, str):
                    assert positions[parent] < j
        assert max(i for i, p in enumerate(target.parameters) if p.role in ("population", "batch")) < min(i for i, p in enumerate(target.parameters) if p.role == "physical")
        repeated, repeated_mapping = reorder_problem(source, seed)
        assert repeated.to_dict() == target.to_dict() and repeated_mapping == mapping
        assert apply_ordering(source, mapping).to_dict() == target.to_dict()
        outcomes.add((mapping.parameter_columns, mapping.family_indices, mapping.dataset_indices))
    assert len(outcomes) > 1
    assert len({item[0] for item in outcomes}) > 1
    assert len({item[1] for item in outcomes}) > 1
    assert len({item[2] for item in outcomes}) > 1


def test_prior_transport_physics_and_dataset_noise_are_equivariant():
    source = problem()
    target, mapping = reorder_problem(source, 19)
    theta = science.draw_prior(source, 12, 29)
    mapped = mapping.parameters_to_target(theta)
    np.testing.assert_allclose(science.log_prior(target, mapped), science.log_prior(source, theta), rtol=2e-14, atol=2e-14)
    latent = science.encode_prior(source, theta)
    np.testing.assert_allclose(science.encode_prior(target, mapped), mapping.parameters_to_target(latent), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(science.decode_prior(target, mapping.parameters_to_target(latent)), mapped, rtol=2e-14, atol=2e-14)
    for noisy in (False, True):
        source_obs, source_cost = science.simulate(source, theta, seed=73, noisy=noisy)
        target_obs, target_cost = science.simulate(target, mapped, seed=73, noisy=noisy)
        np.testing.assert_array_equal(target_obs, mapping.observations_to_target(source_obs))
        for key in ("forward_evaluations", "reused_local_vectors", "noise_values_generated", "per_family_forward_evaluations", "prefix_forward_evaluations", "prefix_forward_evaluations_by_family"):
            assert source_cost[key] == target_cost[key]


def test_inverse_and_json_map_restore_exact_spec_and_array_columns():
    source = problem()
    target, mapping = reorder_problem(source, 3)
    restored = apply_ordering(target, mapping.inverse())
    assert restored.to_dict() == source.to_dict()
    assert restored.fingerprint == source.fingerprint
    assert mapping.inverse().inverse() == mapping
    assert OrderingMap.from_dict(json.loads(json.dumps(mapping.to_dict()))) == mapping
    theta = np.arange(3 * source.dimension).reshape(3, source.dimension)
    obs = np.arange(3 * source.observation_dimension).reshape(3, source.observation_dimension)
    np.testing.assert_array_equal(mapping.parameters_to_source(mapping.parameters_to_target(theta)), theta)
    np.testing.assert_array_equal(mapping.observations_to_source(mapping.observations_to_target(obs)), obs)
    np.testing.assert_array_equal(mapping.parameters_to_source(mapping.parameters_to_target(theta[0])), theta[0])
    assert tuple(mapping.parameter_columns[i] for i in mapping.inverse_parameter_columns) == tuple(range(source.dimension))
    assert tuple(mapping.observation_columns[i] for i in mapping.inverse_observation_columns) == tuple(range(source.observation_dimension))


def test_existing_bank_permutation_adds_no_calls_and_preserves_prefix_costs(monkeypatch, tmp_path):
    source = prepare_bank(problem(), 16, 41).prefix(8)
    save_bank(source, tmp_path / "source")
    bank = load_bank(tmp_path / "source")
    target, mapping = reorder_problem(bank.spec, 77)
    original_cost = json.loads(json.dumps(bank.manifest["cost"]))
    original_manifest = json.loads(json.dumps(bank.manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("ordering must not sample or evaluate any forward response")

    for module, name in ((science, "draw_prior"), (science, "simulate"), (science, "_response"), (banks, "draw_prior"), (banks, "simulate")):
        monkeypatch.setattr(module, name, forbidden)
    permuted = reorder_bank(bank, mapping)
    assert permuted.spec.fingerprint == target.fingerprint
    np.testing.assert_array_equal(permuted.theta, mapping.parameters_to_target(bank.theta))
    np.testing.assert_array_equal(permuted.obs, mapping.observations_to_target(bank.obs))
    assert not permuted.theta.flags.writeable and not permuted.obs.flags.writeable
    assert permuted.manifest["cost"] == original_cost
    assert bank.manifest == original_manifest
    record = permuted.manifest["ordering_history"][-1]
    assert record["added_forward_evaluations"] == record["added_noise_values"] == 0
    assert record["source_fingerprint"] == bank.spec.fingerprint
    assert record["source_parameter_order"] == list(bank.spec.parameter_order)
    assert record["source_artifacts"]["files"] == bank.manifest["files"]
    assert record["source_artifacts"]["parent_theta"] == bank.manifest["parent_theta"]
    assert not ({"theta", "observations", "files", "parent_theta"} & permuted.manifest.keys())
    assert permuted.prefix(4).manifest["cost"]["actual_generation_forward_evaluations"] == original_cost["actual_generation_forward_evaluations"]
    assert permuted.prefix(4).manifest["cost"]["forward_evaluations"] == original_cost["prefix_forward_evaluations"][4]
    inverse = reorder_bank(permuted, mapping.inverse())
    np.testing.assert_array_equal(inverse.theta, bank.theta)
    np.testing.assert_array_equal(inverse.obs, bank.obs)
    assert inverse.manifest["cost"] == original_cost
    assert len(inverse.manifest["ordering_history"]) == 2
    saved = save_bank(permuted, tmp_path / "permuted")
    loaded = load_bank(tmp_path / "permuted", expected_sha256=saved["manifest_sha256"])
    np.testing.assert_array_equal(loaded.theta, permuted.theta)
    np.testing.assert_array_equal(loaded.obs, permuted.obs)
    assert loaded.manifest["cost"] == original_cost


def test_reject_incompatible_maps_partial_columns_and_noncausal_roles():
    source = problem()
    _, mapping = reorder_problem(source, 1)
    with pytest.raises(ValueError, match="source fingerprint"):
        apply_ordering(replace(source, id="other"), mapping)
    with pytest.raises(ValueError, match="target fingerprint"):
        apply_ordering(source, replace(mapping, target_fingerprint="0" * 64))
    broken = list(mapping.observation_columns)
    broken[0], broken[1] = broken[1], broken[0]
    with pytest.raises(ValueError, match="complete dataset"):
        apply_ordering(source, replace(mapping, observation_columns=tuple(broken)))
    for field in ("parameter_columns", "observation_columns", "family_indices", "dataset_indices"):
        with pytest.raises(ValueError, match="permutation"):
            replace(mapping, **{field: (0, 0)})
    with pytest.raises(ValueError, match="dimension"):
        mapping.parameters_to_target(np.zeros((2, source.dimension - 1)))
    with pytest.raises(ValueError, match="dimensions"):
        apply_ordering(source, replace(mapping, parameter_columns=(0,)))
    serialized = mapping.to_dict()
    serialized["inverse_parameter_columns"] = list(reversed(serialized["inverse_parameter_columns"]))
    with pytest.raises(ValueError, match="inverse"):
        OrderingMap.from_dict(serialized)
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            reorder_problem(source, seed)
    # This valid diagnostic prior ordering cannot satisfy the stronger role
    # policy: its population depends on a physical value. Fail rather than
    # silently violating either ancestry or the requested role ordering.
    incompatible = ProblemSpec("noncausal-role-diagnostic", "diagnostic", (
        ParameterSpec("a", "physical", PriorSpec("normal", 0., 1.)),
        ParameterSpec("mean", "population", PriorSpec("normal", "a", 1.)),
        ParameterSpec("b", "physical", PriorSpec("normal", "mean", 1.)),
        ParameterSpec("noise", "noise", PriorSpec("lognormal", .1, .02))),
        (FamilySpec("f", "quadratic", "noise"),),
        (DatasetSpec("d", "f", "s", {"a": "a", "b": "b"}, tuple(np.linspace(-1, 1, 10)), {"d": 0.}),))
    with pytest.raises(ValueError, match="population/batch"):
        reorder_problem(incompatible, 1)
