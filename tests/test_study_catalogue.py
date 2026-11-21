from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from posteriorforge.study.catalogue import generate_setup, generate_catalogue, matched_sharing_pair, structural_signature, heldout_sharing_signatures, SIZE_BANDS
from posteriorforge.study.science import draw_prior, simulate
from posteriorforge.study.spec import ProblemSpec, TRAIN_PHYSICS, TEST_PHYSICS


@pytest.mark.parametrize("band", SIZE_BANDS)
def test_definition_size_strata_and_determinism(band):
    p = generate_setup(9, size_band=band)
    q = generate_setup(9, size_band=band)
    assert p.fingerprint == q.fingerprint
    assert SIZE_BANDS[band][0] <= p.dimension <= SIZE_BANDS[band][1]
    assert p.dimension + p.observation_dimension <= 450
    assert set(f.physics for f in p.families) <= set(TRAIN_PHYSICS)
    assert set(p.roles) >= {"population", "physical", "noise"}
    assert p.metadata["query_truth_used_for_controls"] is False
    theta = draw_prior(p, 2, 3)
    y, _ = simulate(p, theta, 4)
    assert y.shape == (2, p.observation_dimension)


@pytest.mark.parametrize("physics", TRAIN_PHYSICS + TEST_PHYSICS)
def test_each_family_recipe_generates_finite_data(physics):
    split = "test" if physics in TEST_PHYSICS else "train"
    p = generate_setup(12, split=split, physics=[physics])
    theta = draw_prior(p, 3, 9)
    y, cost = simulate(p, theta, 10)
    assert np.isfinite(y).all() and cost["forward_evaluations"] > 0
    assert len(p.families) == 1
    assert sum(x.role == "noise" for x in p.parameters) == 1


@pytest.mark.parametrize("split", ("train", "validation"))
def test_reserved_physics_never_enters_development(split):
    with pytest.raises(ValueError, match="reserved"):
        generate_setup(0, split=split, physics=["spring"])


@pytest.mark.parametrize("pattern", ("distinct", "repeated", "cross", "batch", "overlap"))
def test_sharing_and_batches_are_explicit(pattern):
    p = generate_setup(7, physics=["sensor", "quadratic"], sharing=pattern)
    assert p.metadata["sharing"] == pattern
    if pattern in ("batch", "overlap"):
        assert 1 <= p.metadata["batch_count"] <= 3
        assert "batch" in p.roles
    else:
        assert p.metadata["batch_count"] == 0
    assert ProblemSpec.from_dict(p.to_dict()).fingerprint == p.fingerprint


def test_matched_pairs_change_links_not_priors_or_size():
    for i in range(4):
        a, b = matched_sharing_pair(i)
        assert a.parameters == b.parameters
        assert a.dimension == b.dimension
        assert a.observation_dimension == b.observation_dimension
        assert [d.controls for d in a.datasets] == [d.controls for d in b.datasets]
        assert [d.matrix for d in a.datasets] == [d.matrix for d in b.datasets]
        assert [dict(d.parameters) for d in a.datasets] != [dict(d.parameters) for d in b.datasets]
        assert structural_signature(a) != structural_signature(b)
        assert a.metadata["matched_pair"] == b.metadata["matched_pair"] == i


def test_structural_signature_ignores_all_identity_labels():
    p = matched_sharing_pair(0)[0]
    data = p.to_dict()
    rename = {x["name"]: "new-" + x["name"] for x in data["parameters"]}
    for x in data["parameters"]:
        x["name"] = rename[x["name"]]
        for k in ("mean", "sd"):
            if isinstance(x["prior"][k], str):
                x["prior"][k] = rename[x["prior"][k]]
    fr = {f["id"]: "new-" + f["id"] for f in data["families"]}
    for f in data["families"]:
        f["id"] = fr[f["id"]]; f["noise_parameter"] = rename[f["noise_parameter"]]
    for d in data["datasets"]:
        d["id"] = "new-" + d["id"]
        d["family_id"] = fr[d["family_id"]]
        d["specimen_id"] = "new-" + d["specimen_id"]
        d["parameters"] = {k: rename[v] for k, v in d["parameters"].items()}
    data["id"] = "new-id"
    q = ProblemSpec.from_dict(data)
    assert p.fingerprint != q.fingerprint
    assert structural_signature(p) == structural_signature(q)


def test_small_development_catalogue_strata_and_holdouts():
    specs = generate_catalogue("train", count=6, seed=9)
    assert Counter(p.metadata["size_band"] for p in specs) == {"small": 2, "medium": 2, "large": 2}
    reserved = heldout_sharing_signatures(9)
    assert all(structural_signature(p) not in reserved for p in specs)
    validation = generate_catalogue("validation", count=3, seed=9)
    assert not set(p.fingerprint for p in specs) & set(p.fingerprint for p in validation)


def test_test_group_definitions_without_query_or_bank_generation():
    specs = generate_catalogue("test", count=40, seed=33)
    counts = Counter(p.metadata["test_group"] for p in specs)
    assert set(counts.values()) == {8} and len(counts) == 5
    for p in specs:
        kinds = {f.physics for f in p.families}
        if p.metadata["test_group"] == "heldout_physics":
            assert kinds <= set(TEST_PHYSICS)
        elif p.metadata["test_group"] == "combined":
            assert kinds & set(TRAIN_PHYSICS) and kinds & set(TEST_PHYSICS)
    # This test constructs definitions only: no query truth, bank or reference.
    assert all("query_theta" not in p.metadata for p in specs)


def test_large_stratum_can_keep_a_small_family_anchor_without_fixing_an_impossible_subset():
    p = generate_setup(3, size_band="large", required_physics="scalar", seed=4)
    assert "scalar" in {f.physics for f in p.families}
    assert p.dimension >= 65 and len(p.families) >= 2
    with pytest.raises(ValueError, match="cannot reach"):
        generate_setup(0, size_band="large", physics=["scalar"])


def test_cross_label_requires_a_real_cross_family_physical_link():
    for seed in range(8):
        p = generate_setup(1, physics=["sensor", "saturation"], sharing="cross", seed=seed)
        users = {}
        for d in p.datasets:
            for parameter in d.parameters.values():
                users.setdefault(parameter, set()).add(d.family_id)
        assert any(len(families) > 1 for families in users.values())
    with pytest.raises(ValueError, match="more than one"):
        generate_setup(0, physics=["scalar"], sharing="cross")


def test_scalar_readouts_have_no_curve_coverage_label():
    p = generate_setup(5, physics=["scalar"])
    assert p.metadata["coverage"] == "not_applicable"
    assert set(p.metadata["dataset_coverage"].values()) == {"not_applicable"}
