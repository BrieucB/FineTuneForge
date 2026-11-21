#!/usr/bin/env python3
"""Create a separate, hash-locked GPU compile environment without using a GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ARCHIVE_URL='https://github.com/astral-sh/python-build-standalone/releases/download/20251007/cpython-3.11.13%2B20251007-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
ARCHIVE_SHA256='43bfc42529843ecd1d9c08c4a239ede348f96ff0acaef2ec24b28dc059f4f0c3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_paths(scratch, original, env, interpreter, receipt):
    root=Path(scratch).resolve();original=Path(original).resolve()
    paths=[Path(env),Path(interpreter),Path(receipt)]
    for p in paths:
        if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('all destinations must be non-symlink paths under scratch')
        if p.exists():raise FileExistsError(p)
        if p.resolve()==original or p.resolve().is_relative_to(original) or original.is_relative_to(p.resolve()):
            raise ValueError('destination overlaps original environment')
    resolved=[p.resolve() for p in paths]
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(resolved) for b in resolved[i+1:]):
        raise ValueError('destinations overlap')


def main(argv=None):
    p=argparse.ArgumentParser()
    for key in ('scratch','original','environment','interpreter','receipt','uv','cache','repo'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args(argv);validate_paths(a.scratch,a.original,a.environment,a.interpreter,a.receipt)
    repo=Path(a.repo).resolve(); output=Path(a.interpreter); output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    # Keep the upstream archive as a permanent, hash-verified provenance artifact.
    archive=output.parent/(output.name+'-cpython-3.11.13-20251007.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    subprocess.run(['curl','-fL','--retry','2','--max-time','180','--output',str(archive),ARCHIVE_URL],check=True,env=env)
    if sha(archive)!=ARCHIVE_SHA256:raise ValueError('managed CPython archive checksum mismatch')
    output.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        tar.extractall(output,filter='data')
    python=output/'python/bin/python3.11'
    if not python.is_file():raise ValueError('managed Python executable missing')
    script=repo/'scripts/prepare-study-gpu-env.sh'
    prep_receipt=Path(a.receipt).with_name(Path(a.receipt).name+'.gpu-lock.json')
    if prep_receipt.exists():raise FileExistsError(prep_receipt)
    subprocess.run([str(script),'--environment',a.environment,'--uv',a.uv,'--python',str(python),
                    '--cache',a.cache,'--receipt',str(prep_receipt),'--offline'],check=True,env=env)
    probe=r'''import hashlib,importlib.util,json,os,platform,subprocess,sys,sysconfig,tempfile
from pathlib import Path
if sys.version_info[:3]!=(3,11,13):raise RuntimeError('wrong CPython patch')
include=Path(sysconfig.get_path('include')); header=include/'Python.h';config=include/'pyconfig.h'
if not header.is_file() or not config.is_file():raise RuntimeError('Python development headers missing')
source=''' + repr('#include <Python.h>\nstatic PyObject *answer(PyObject *s, PyObject *a) { return PyLong_FromLong(42); }\nstatic PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, ""}, {NULL, NULL, 0, NULL}};\nstatic struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "ftf_compile_probe", NULL, -1, methods};\nPyMODINIT_FUNC PyInit_ftf_compile_probe(void) { return PyModule_Create(&mod); }\n') + r'''
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='ftf-c-extension-') as d:
 src=Path(d)/'probe.c';src.write_text(source);out=Path(d)/('ftf_compile_probe'+sysconfig.get_config_var('EXT_SUFFIX'))
 command=['gcc','-shared','-fPIC','-O2','-I'+str(include),str(src),'-o',str(out)]
 subprocess.run(command,check=True)
 spec=importlib.util.spec_from_file_location('ftf_compile_probe',out);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if m.answer()!=42:raise RuntimeError('compiled module returned unexpected result')
 result={'python':sys.version,'executable':sys.executable,'base_prefix':sys.base_prefix,'platform':platform.platform(),
 'include':str(include),'python_h_sha256':h(header),'pyconfig_h_sha256':h(config),
 'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
 'compiler_path':subprocess.check_output(['which','gcc'],text=True).strip(),
 'compiler_sha256':h(subprocess.check_output(['which','gcc'],text=True).strip()),
 'source_sha256':h(src),'extension_sha256':h(out),'compile_command':command,'extension_import_value':m.answer()}
import torch
if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized during compile probe')
result['cuda_initialized']=False;print(json.dumps(result,sort_keys=True))
'''
    probe_output=subprocess.check_output([str(Path(a.environment)/'bin/python'),'-c',probe],text=True,env=env)
    result={'schema':'finetuneforge.compile_environment.v1','environment':str(Path(a.environment).resolve()),
            'interpreter_install':str(output.resolve()),'archive_url':ARCHIVE_URL,'archive_sha256':ARCHIVE_SHA256,
            'archive_path':str(archive.resolve()),'gpu_lock_receipt':str(prep_receipt.resolve()),
            'gpu_lock_receipt_sha256':sha(prep_receipt),'lock_sha256':sha(repo/'environments/study-training-gpu.lock'),
            'probe':json.loads(probe_output),'offline_locked_packages':True,'gpu_jobs_submitted':0,'scientifically_qualified':False}
    receipt=Path(a.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'receipt':str(receipt),'python':result['probe']['executable'],'cuda_initialized':False},sort_keys=True))


if __name__=='__main__':main()
