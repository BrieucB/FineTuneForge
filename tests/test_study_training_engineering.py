from contextlib import nullcontext
import json
from types import SimpleNamespace
import pytest
from posteriorforge.study import training_engineering as eng


def test_scoped_call_restores_instance_even_on_exception():
    class Example:
        def action(self, value): return value + 1
    obj=Example(); labels=[]
    def record(label): labels.append(label); return nullcontext()
    with pytest.raises(RuntimeError):
        with eng.scoped_calls(obj, 'action', 'scope', record):
            assert obj.action(3)==4
            raise RuntimeError('failure')
    assert obj.action(4)==5 and 'action' not in obj.__dict__ and labels==['scope']


def test_binding_rejects_bad_input_checksum(tmp_path):
    p=tmp_path/'manifest.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        eng.verify_bindings(p,'wrong',tmp_path,'commit')


def test_worker_rejects_unaccounted_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        eng.worker('profile',inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')


def test_launch_requires_distinct_devices(monkeypatch, tmp_path):
    monkeypatch.setenv('SLURM_JOB_ID','123'); monkeypatch.setenv('SLURM_JOB_ACCOUNT','eu-26-17')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,0')
    monkeypatch.setattr(eng,'verify_bindings',lambda *a: {})
    with pytest.raises(RuntimeError,match='distinct'):
        eng.launch(inputs='none',input_sha256='none',output=tmp_path/'out',repo='none',source_commit='none')
