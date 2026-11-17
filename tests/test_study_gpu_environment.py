"""Public lock/ownership checks; no Torch import, network, model or GPU is needed."""
from pathlib import Path
import re
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def requirements(path):
    return dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;]+)',path.read_text(),re.M))


def test_gpu_lock_preserves_cpu_dependencies_and_adds_cuda_runtime():
    cpu=requirements(ROOT/'environments/study-training-cpu.lock')
    gpu=requirements(ROOT/'environments/study-training-gpu.lock')
    assert len(cpu)>100
    assert gpu['torch']=='2.6.0'
    assert cpu['torch']=='2.6.0+cpu'
    assert {k:v for k,v in cpu.items() if k!='torch'}=={k:gpu[k] for k in cpu if k!='torch'}
    assert gpu['nvidia-cuda-runtime-cu12']=='12.4.127'
    assert gpu['nvidia-cublas-cu12']=='12.4.5.8'
    assert gpu['triton']=='3.2.0'
    ancestry=hashlib.sha256((ROOT/'environments/study-training-cpu.lock').read_bytes()).hexdigest()
    assert ancestry in (ROOT/'environments/study-training-gpu.in').read_text()


def test_every_locked_requirement_has_a_sha256():
    text=(ROOT/'environments/study-training-gpu.lock').read_text()
    blocks=re.split(r'(?=^[A-Za-z0-9_.-]+==)',text,flags=re.M)[1:]
    assert len(blocks)==181
    for block in blocks:
        assert re.search(r'--hash=sha256:[0-9a-f]{64}',block),block.splitlines()[0]
    assert 'http://' not in text
    assert not re.search(r'(^|\s)-e\s|git\+|file://',text)


def test_preparation_refuses_to_mutate_unowned_existing_environment(tmp_path):
    existing=tmp_path/'historical-env';existing.mkdir()
    sentinel=existing/'untouched';sentinel.write_text('legacy')
    result=subprocess.run(['bash',str(ROOT/'scripts/prepare-study-gpu-env.sh'),
        '--environment',str(existing),'--python',__import__('sys').executable,
        '--uv','/does/not/exist','--cache',str(tmp_path/'cache'),
        '--receipt',str(tmp_path/'receipt.json')],text=True,capture_output=True)
    assert result.returncode!=0
    assert 'Refusing to mutate an existing environment' in result.stderr
    assert sentinel.read_text()=='legacy'
    assert not (tmp_path/'receipt.json').exists()
