from pathlib import Path

import numpy as np
import pytest

from posteriorforge.study.banks import PairedBank, prepare_bank, save_bank, load_bank
from posteriorforge.study.spec import PriorSpec, ParameterSpec, FamilySpec, DatasetSpec, ProblemSpec


def problem():
    return ProblemSpec("bank-test", "diagnostic", (
        ParameterSpec("a", "physical", PriorSpec("normal", 0., 1.)),
        ParameterSpec("sigma", "noise", PriorSpec("lognormal", .1, .02))),
        (FamilySpec("f", "scalar", "sigma"),),
        (DatasetSpec("d0", "f", "s", {"a": "a"}), DatasetSpec("d1", "f", "s", {"a": "a"})))


def test_nested_prefixes_and_distinct_cost_units():
    bank = prepare_bank(problem(), 16, 5)
    prefix = bank.prefix(4)
    np.testing.assert_array_equal(prefix.theta, bank.theta[:4])
    np.testing.assert_array_equal(prefix.obs, bank.obs[:4])
    assert prefix.manifest["cost"]["forward_evaluations"] == 4
    assert prefix.manifest["cost"]["actual_generation_forward_evaluations"] == 16
    assert prefix.manifest["cost"]["requested_local_vectors"] == 8
    assert prefix.prefix(2).manifest["cost"]["actual_generation_forward_evaluations"] == 16
    assert not bank.theta.flags.writeable and not bank.obs.flags.writeable


def test_shorter_generation_has_exact_prefixes():
    p = problem()
    whole, short = prepare_bank(p, 16, 5), prepare_bank(p, 4, 5)
    np.testing.assert_array_equal(whole.theta[:4], short.theta)
    np.testing.assert_array_equal(whole.obs[:4], short.obs)


def test_save_load_authentication(tmp_path):
    bank = prepare_bank(problem(), 8, 1)
    saved = save_bank(bank, tmp_path / "bank")
    loaded = load_bank(tmp_path / "bank", expected_sha256=saved["manifest_sha256"])
    np.testing.assert_array_equal(loaded.theta, bank.theta)
    np.testing.assert_array_equal(loaded.obs, bank.obs)
    assert loaded.spec.fingerprint == bank.spec.fingerprint
    with pytest.raises(FileExistsError):
        save_bank(bank, tmp_path / "bank")
    with pytest.raises(ValueError):
        load_bank(tmp_path / "bank", expected_sha256="0" * 64)
    with (tmp_path / "bank" / "bank.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bank(tmp_path / "bank")


def test_mispaired_or_unsupported_bank_fails():
    p = problem()
    with pytest.raises(ValueError):
        PairedBank(p, np.zeros((4, 2)), np.zeros((3, 2)))
    x = prepare_bank(p, 4, 1)
    bad = x.theta.copy(); bad[0, 1] = -1
    with pytest.raises(ValueError):
        PairedBank(p, bad, x.obs)
    with pytest.raises(ValueError):
        x.prefix(5)


def test_prefix_costs_remain_internally_consistent():
    whole = prepare_bank(problem(), 16, 2)
    cost = whole.prefix(4).manifest["cost"]
    assert sum(cost["per_family_forward_evaluations"].values()) == cost["forward_evaluations"] == 4
    assert cost["noise_values_generated"] == 8
    assert cost["actual_generation_noise_values"] == 32
    assert cost["actual_generation_per_family_forward_evaluations"] == {"f": 16}
