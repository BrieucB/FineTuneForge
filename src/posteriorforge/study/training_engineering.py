"""Bounded CUDA update timing and one-conditional CPU/CUDA profiler.

This is engineering evidence only; it neither samples posteriors nor selects LR.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from .cuda_qualification import file_hash, write_json, _load_bank, SEED, DRAW_COUNT


def verify_bindings(inputs, input_sha256, repo, source_commit):
    """Authenticate the old immutable bank contract and current committed tree."""
    path = Path(inputs)
    if file_hash(path) != input_sha256:
        raise ValueError('input manifest checksum mismatch')
    value = json.loads(path.read_text())
    if (value.get('schema') != 'finetuneforge.cuda_qualification_inputs.v1'
            or value.get('posterior_draws') != DRAW_COUNT
            or value.get('training_effective_batch') != 16
            or len(value.get('episodes', [])) != 32):
        raise ValueError('original input contract mismatch')
    if file_hash(value['base_checkpoint']) != value['base_checkpoint_sha256']:
        raise ValueError('original base checkpoint checksum mismatch')
    root = Path(repo).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != source_commit:
        raise ValueError('source commit mismatch')
    # Include untracked source as well as tracked changes; output belongs outside the tree.
    status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--',
                                      'src', 'scripts', 'slurm', 'environments'], text=True)
    if status.strip():
        raise ValueError('source snapshot has uncommitted changes')
    return value


@contextmanager
def scoped_calls(obj, name, label, record_function):
    """Temporarily wrap an instance method; restore even on failure."""
    had = name in getattr(obj, '__dict__', {})
    previous = obj.__dict__.get(name) if had else None
    original = getattr(obj, name)
    def wrapped(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)
    object.__setattr__(obj, name, wrapped)
    try:
        yield
    finally:
        if had:
            object.__setattr__(obj, name, previous)
        else:
            object.__delattr__(obj, name)


def _profile_conditional(backend, batch, out):
    """Trace exactly the first episode/stage without an optimizer update."""
    import torch
    from .model import CANONICAL_FULL_SUPPORT
    record = torch.profiler.record_function
    started = time.perf_counter()
    with scoped_calls(backend.model, 'forward', 'engineering::network_forward', record):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, with_stack=False, profile_memory=False) as prof:
            t = time.perf_counter()
            with record('engineering::preprocess_and_h2d'):
                prepared = backend.prepare(batch[0])
                torch.cuda.synchronize()
            preprocessing = time.perf_counter() - t
            stage = prepared.stages[0]
            with record('engineering::density_including_network'):
                density = stage.ensemble.density(backend.model, law=CANONICAL_FULL_SUPPORT)
            with record('engineering::density_log_prob'):
                loss = -density.log_prob(stage.targets, log_abs_det_decode=stage.decoder_log_jac).sum()
            with record('engineering::conditional_backward'):
                loss.backward()
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    backend.optimizer.zero_grad(set_to_none=True)
    prof.export_chrome_trace(str(out / 'first-conditional-trace.json'))
    events = prof.key_averages()
    (out / 'operators.txt').write_text(events.table(sort_by='self_cuda_time_total', row_limit=100))
    rows = []
    for e in events:
        key = e.key
        if (key in ('aten::mm','aten::bmm','aten::addmm','cudaLaunchKernel') or key.startswith('engineering::') or 'backward' in key.lower()
                or 'memcpy' in key.lower() or 'copy_' in key.lower()
                or 'synchroniz' in key.lower() or 'item' in key.lower()):
            rows.append({'name': key, 'count': e.count,
                         'cpu_total_us': e.cpu_time_total, 'cpu_self_us': e.self_cpu_time_total,
                         'device_total_us': getattr(e, 'device_time_total', 0),
                         'device_self_us': getattr(e, 'self_device_time_total', 0)})
    return {'episode_position':0, 'stage_index':stage.index, 'parameter_name':stage.parameter_name,
            'preprocessing_and_h2d_wall_seconds':preprocessing,
            'profiled_preparation_and_conditional_wall_seconds':elapsed, 'operator_summary':rows,
            'trace':'first-conditional-trace.json', 'operator_table':'operators.txt',
            'scope':'one conditional only; no optimizer step; not a full update'}


def worker(mode, *, inputs, input_sha256, output, repo, source_commit):
    import numpy as np
    import torch
    from .campaign import CampaignSpec
    from .cuda_training import QualificationTrainingBackend, configure_cuda_determinism
    from .episodes import digest
    from .training import TrainingState
    if mode not in ('profile', 'baseline'):
        raise ValueError('invalid worker mode')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    value = verify_bindings(inputs, input_sha256, repo, source_commit)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    result = {'schema':'finetuneforge.training_engineering.v1', 'mode':mode, 'status':'running',
              'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'source_commit':source_commit,
              'input_manifest_sha256':input_sha256, 'original_source_commit':value['source_commit'],
              'base_checkpoint_sha256':value['base_checkpoint_sha256'], 'job_id':os.environ['SLURM_JOB_ID'],
              'posterior_requests':0, 'posterior_draws_contract':DRAW_COUNT, 'scientifically_qualified':False}
    write_json(out/'result.json', result)
    start = time.perf_counter()
    try:
        configure_cuda_determinism(); torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one visible GPU per worker required')
        props = torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory < 39*1024**3:
            raise RuntimeError('A100 40GB or larger required')
        result['hardware'] = {'name':props.name, 'memory_bytes':props.total_memory,
                              'uuid':str(getattr(props, 'uuid', 'unavailable')), 'cuda_runtime':torch.version.cuda}
        root = Path(inputs).parent
        batches = [[SimpleNamespace(context=_load_bank(root, row['context']),
                                     query=_load_bank(root, row['query']), receipt=row)
                    for row in value['episodes'][start:start+16]] for start in (0,16)]
        spec = CampaignSpec('lora_r4', 0, 'pilot', 1e-4, 2, value['protocol_sha256'], effective_batch=16, diagnostic=True)
        backend = QualificationTrainingBackend(spec, value['base_checkpoint'], device='cuda:0', allow_cuda_qualification=True)
        backend.episode_rng = np.random.default_rng(SEED+5000)
        torch.manual_seed(spec.initialization_seed); torch.cuda.manual_seed_all(spec.initialization_seed)
        np.random.seed(SEED); random.seed(SEED)
        state = TrainingState(stream_id=digest(value['episodes']))
        result['spec'] = asdict(spec)
        torch.cuda.reset_peak_memory_stats()
        if mode == 'profile':
            result['conditional_profile'] = _profile_conditional(backend, batches[0], out)
        else:
            reports=[]; durations=[]
            for batch in batches:
                t=time.perf_counter(); reports.append(backend.update(batch, state)); torch.cuda.synchronize()
                durations.append(time.perf_counter()-t)
            result.update(updates=reports, unprofiled_update_wall_seconds=durations,
                          timing_note='first update cold; second update warm, using next fixed episode batch')
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), status='completed')
    except BaseException as exc:
        result.update(status='failed', failure={'type':type(exc).__name__, 'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        result['worker_wall_seconds'] = time.perf_counter()-start
        write_json(out/'result.json', result)
    return result


def launch(*, inputs, input_sha256, output, repo, source_commit):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    verify_bindings(inputs, input_sha256, repo, source_commit)
    devices = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise RuntimeError('at least two distinct allocated GPUs required')
    out=Path(output); out.mkdir(parents=True, exist_ok=False); jobs=[]
    try:
        for mode, device in zip(('profile','baseline'), devices):
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            log=(out/(mode+'.log')).open('w')
            cmd=[sys.executable, str(Path(repo)/'scripts/engineer-study-training.py'), 'worker',
                 '--mode', mode, '--inputs', str(inputs), '--input-sha256', input_sha256,
                 '--output', str(out/mode), '--repo', str(repo), '--source-commit', source_commit]
            jobs.append((mode,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        codes={mode:proc.wait() for mode,proc,_ in jobs}
    finally:
        for _,proc,log in jobs:
            if proc.poll() is None: proc.terminate(); proc.wait()
            log.close()
    summary={'schema':'finetuneforge.training_engineering_allocation.v1','worker_returncodes':codes,
             'input_manifest_sha256':input_sha256,'source_commit':source_commit,
             'job_id':os.environ['SLURM_JOB_ID'], 'posterior_requests':0,
             'scientifically_qualified':False}
    write_json(out/'allocation.json',summary)
    if any(code != 0 for code in codes.values()): raise RuntimeError('engineering worker failed; inspect retained evidence')
    return summary


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='operation',required=True)
    for name in ('run','worker'):
        q=sub.add_parser(name)
        for arg in ('inputs','input-sha256','output','repo','source-commit'): q.add_argument('--'+arg,required=True)
        if name=='worker': q.add_argument('--mode',choices=('profile','baseline'),required=True)
    args=vars(p.parse_args(argv)); op=args.pop('operation')
    result=launch(**args) if op=='run' else worker(**args)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
