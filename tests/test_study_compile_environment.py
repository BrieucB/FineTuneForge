import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/prepare-study-compile-environment.py'
spec=importlib.util.spec_from_file_location('compile_env',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_reject_original_or_overlapping_destinations(tmp_path):
    original=tmp_path/'original';original.mkdir()
    with pytest.raises((ValueError,FileExistsError)):
        module.validate_paths(tmp_path,original,original,tmp_path/'python',tmp_path/'receipt')
    with pytest.raises(ValueError,match='overlap'):
        module.validate_paths(tmp_path,original,tmp_path/'new',tmp_path/'new'/'python',tmp_path/'receipt')


def test_reject_outside_scratch_and_existing(tmp_path):
    original=tmp_path/'old';original.mkdir()
    with pytest.raises(ValueError,match='scratch'):
        module.validate_paths(tmp_path,original,tmp_path.parent/'outside',tmp_path/'python',tmp_path/'receipt')
    with pytest.raises(FileExistsError):
        module.validate_paths(tmp_path,original,tmp_path/'old',tmp_path/'python',tmp_path/'receipt')


def test_pinned_archive_contract():
    assert '3.11.13%2B20251007' in module.ARCHIVE_URL
    assert len(module.ARCHIVE_SHA256)==64
