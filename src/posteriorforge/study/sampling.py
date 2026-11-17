"""Complete-vector autoregressive posterior generation with explicit qualification modes.

Checkpoint files retain only complete microbatches, never partial parameter rows.
They are recovery state, not qualified sample artifacts.
"""
from __future__ import annotations
from hashlib import sha256
import json
import inspect
import copy
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .banks import PairedBank, array_descriptor
from .config import StudyConfig
from .science import decode_prior, encode_prior

COUNT = 50000
SCHEMA = 'finetuneforge.study.posterior-recovery.v2'
UNIFORM_LAW = 'uint53-open.v1'


def _source_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _backend_identity(backend):
    if hasattr(backend, 'identity_metadata'):
        identity = backend.identity_metadata()
    else:
        cls = type(backend)
        try:
            source = inspect.getsource(cls).encode()
        except (OSError, TypeError) as exc:
            raise ValueError('backend must expose stable identity metadata or inspectable source') from exc
        identity = {'type': cls.__module__ + '.' + cls.__qualname__, 'source_sha256': sha256(source).hexdigest()}
    if not isinstance(identity, dict) or not identity:
        raise ValueError('backend identity metadata required')
    _digest(identity)
    return identity


def _open_uniform(rng, count):
    # Every integer is exactly representable and maps strictly inside (0, 1).
    return rng.integers(1, 1 << 53, size=count, dtype=np.uint64).astype(np.float64) * (2.0 ** -53)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _hash_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _save(path, binding, samples, rngs, stage_seconds, elapsed, failure=None):
    state = {'schema': SCHEMA, 'status': 'failed' if failure else ('incomplete' if len(samples) < COUNT else 'complete_unassessed'),
             'binding': binding, 'rows': len(samples), 'samples': array_descriptor(samples),
             'rng': [r.bit_generator.state for r in rngs], 'stage_seconds': stage_seconds,
             'wall_seconds': elapsed, 'failure': failure}
    state['checksum'] = _digest(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.posterior-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, samples=samples, state=np.array(json.dumps(state, sort_keys=True, allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _load(path, binding, spec, rngs):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'samples', 'state'}: raise ValueError('invalid recovery payload')
        samples = np.array(data['samples'], copy=True)
        state = json.loads(str(data['state'].item()))
    checksum = state.pop('checksum', None)
    if checksum != _digest(state) or state.get('schema') != SCHEMA or state.get('binding') != binding:
        raise ValueError('recovery identity or checksum mismatch')
    n = len(samples)
    if (samples.dtype != np.float64 or samples.shape != (n, spec.dimension) or n > COUNT
        or state.get('rows') != n or state.get('samples') != array_descriptor(samples)
        or state.get('status') not in (('incomplete', 'failed') if n < COUNT else ('complete_unassessed',))
        or not spec.support_mask(samples).all() or len(state.get('rng', [])) != spec.dimension):
        raise ValueError('invalid or incomplete recovery rows')
    if n < COUNT and n % binding['microbatch_size'] != 0:
        raise ValueError('recovery must contain complete microbatches')
    for rng, saved in zip(rngs, state['rng']): rng.bit_generator.state = saved
    return samples, state['stage_seconds'], state['wall_seconds']


def sample_posterior(bank, observation, backend, *, seed, microbatch_size, identities,
                     resume_path=None, diagnostic_cpu=False, cuda_qualification=False):
    """Produce exactly 50,000 complete supported vectors and diagnostic metadata.

    backend.prepare(context_x, context_y) fits a conditional and backend.sample
    consumes that stage, query features, and independent FP64 uniforms. The query
    contains observations and generated predecessors only. An exception is a failed
    request; an existing recovery file remains explicitly incomplete and resumable.
    """
    if type(diagnostic_cpu) is not bool or type(cuda_qualification) is not bool or diagnostic_cpu == cuda_qualification:
        raise ValueError('select exactly one explicit diagnostic_cpu or cuda_qualification mode')
    if diagnostic_cpu:
        if os.environ.get('SLURM_JOB_ID'):
            raise ValueError('posterior diagnostics are forbidden inside Slurm jobs')
    else:
        if not os.environ.get('SLURM_JOB_ID'):
            raise ValueError('CUDA qualification requires an accounted Slurm allocation')
        if getattr(backend, 'backend_kind', None) != 'stock_tabpfn_cuda_qualification':
            raise ValueError('CUDA qualification requires the authenticated CUDA backend')
        import torch
        if not torch.cuda.is_available() or torch.device(getattr(backend, 'device', 'cpu')).type != 'cuda':
            raise ValueError('CUDA qualification requires an available CUDA device')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('FP32 qualification requires TF32 disabled')
    if not isinstance(bank, PairedBank): raise TypeError('paired bank required')
    spec = bank.spec
    y = np.asarray(observation, dtype=np.float64)
    if y.shape != (spec.observation_dimension,) or not np.isfinite(y).all():
        raise ValueError('one finite observation-only query required')
    if type(seed) is not int or seed < 0 or type(microbatch_size) is not int or not 1 <= microbatch_size <= COUNT:
        raise ValueError('nonnegative seed and valid microbatch size required')
    required = {'model_sha256', 'bank_manifest_sha256', 'protocol_sha256'}
    if not isinstance(identities, dict) or set(identities) != required or not all(_hash_ok(identities[k]) for k in required):
        raise ValueError('explicit model, bank and protocol SHA256 identities required')
    if getattr(backend, 'model_sha256', identities['model_sha256']) != identities['model_sha256']:
        raise ValueError('backend model identity mismatch')
    if hasattr(backend, 'verify_identity'):
        backend.verify_identity()
    source_dir = Path(__file__).parent
    source_hashes = {name: _source_sha(source_dir / name) for name in ('sampling.py', 'science.py', 'spec.py', 'banks.py', 'config.py')}
    backend_law = getattr(backend, 'sampling_law', 'diagnostic_backend_unspecified')
    backend_policy = getattr(backend, 'source_policy', 'diagnostic_backend_unspecified')
    if any(not isinstance(value, str) or not value.strip() for value in (backend_law, backend_policy)):
        raise ValueError('backend sampling law and source policy must be nonempty strings')
    binding = {'identities': dict(identities), 'problem': spec.fingerprint,
        'source_sha256': source_hashes, 'backend': _backend_identity(backend), 'uniform_law': UNIFORM_LAW,
        'theta': array_descriptor(bank.theta), 'observations': array_descriptor(bank.observations),
        'query': array_descriptor(y), 'seed': seed, 'microbatch_size': microbatch_size,
        'requested_sampling_law': StudyConfig().sampling_law, 'law': backend_law, 'source_policy': backend_policy, 'invalid_policy': 'fail', 'count': COUNT}
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(spec.dimension)]
    samples = np.empty((0, spec.dimension), dtype=np.float64)
    stage_seconds = [0.0] * spec.dimension
    elapsed = 0.0
    if resume_path is not None and Path(resume_path).exists():
        samples, stage_seconds, elapsed = _load(resume_path, binding, spec, rngs)
    start = time.monotonic()
    context_z = encode_prior(spec, bank.theta)
    stages = []
    context_start = time.monotonic()
    if len(samples) < COUNT:
        try:
            for j in range(spec.dimension):
                t = time.monotonic()
                stages.append(backend.prepare(np.concatenate((bank.observations, context_z[:, :j]), axis=1), context_z[:, j].copy()))
                stage_seconds[j] += time.monotonic() - t
        except Exception as exc:
            if resume_path is not None:
                _save(resume_path, binding, samples, rngs, stage_seconds, elapsed + time.monotonic() - start,
                      failure={'type': type(exc).__name__, 'message': str(exc), 'completed_rows': len(samples)})
            raise
    context_seconds = time.monotonic() - context_start
    warm_start = time.monotonic()
    while len(samples) < COUNT:
        n = min(microbatch_size, COUNT - len(samples))
        prior_rng = [copy.deepcopy(r.bit_generator.state) for r in rngs]
        try:
            z = np.empty((n, spec.dimension), dtype=np.float64)
            for j, stage in enumerate(stages):
                t = time.monotonic()
                query = np.concatenate((np.broadcast_to(y, (n, len(y))), z[:, :j]), axis=1)
                values = np.asarray(backend.sample(stage, query, _open_uniform(rngs[j], n)), dtype=np.float64)
                if values.shape != (n,) or not np.isfinite(values).all():
                    raise ValueError('conditional returned incomplete or nonfinite rows')
                z[:, j] = values
                stage_seconds[j] += time.monotonic() - t
            physical = np.asarray(decode_prior(spec, z))
            if physical.dtype != np.float64 or physical.shape != (n, spec.dimension) or not spec.support_mask(physical).all():
                raise ValueError('decoded microbatch must contain complete supported float64 rows')
            if len(samples) + n == COUNT and hasattr(backend, 'verify_identity'):
                backend.verify_identity()
        except Exception as exc:
            for rng, state in zip(rngs, prior_rng): rng.bit_generator.state = state
            if resume_path is not None:
                _save(resume_path, binding, samples, rngs, stage_seconds, elapsed + time.monotonic() - start,
                      failure={'type': type(exc).__name__, 'message': str(exc), 'completed_rows': len(samples)})
            raise
        samples = np.concatenate((samples, physical), axis=0)
        if resume_path is not None: _save(resume_path, binding, samples, rngs, stage_seconds, elapsed + time.monotonic() - start)
    if not stages and hasattr(backend, 'verify_identity'):
        backend.verify_identity()
    cost = bank.manifest.get('cost', {})
    metadata = {'status': 'samples_complete_unassessed', 'scientifically_qualified': False,
        'binding': binding, 'stage_seconds': stage_seconds, 'wall_seconds': elapsed + time.monotonic() - start,
        'context_preparation_seconds_this_run': context_seconds, 'sampling_seconds_this_run': time.monotonic() - warm_start,
        'cold_model_load_seconds': None,
        'posterior_sampling_forward_evaluations': 0,
        'context_forward_evaluations': cost.get('context_forward_evaluations', cost.get('forward_evaluations')),
        'shared_bank_generation_forward_evaluations': cost.get('actual_generation_forward_evaluations', cost.get('forward_evaluations'))}
    return samples, metadata
