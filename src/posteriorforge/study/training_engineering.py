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
import socket
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
    from .model import CANONICAL_FULL_SUPPORT, configure_differentiable_forward
    configure_differentiable_forward(backend.model)
    record = torch.profiler.record_function
    started = time.perf_counter()
    from .training_execution import conditional_forward_cache
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, with_stack=False, profile_memory=False) as prof:
        t = time.perf_counter()
        with record('engineering::preprocess_and_h2d'):
            prepared = backend.prepare(batch[0])
            torch.cuda.synchronize()
        preprocessing = time.perf_counter() - t
        stage = prepared.stages[0]
        with scoped_calls(stage.ensemble, 'adjusted_logits', 'engineering::network_forward', record):
            with conditional_forward_cache(backend.model, enabled=backend.cache_parametrizations):
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
                         'native_operator_device_total_us': getattr(e, 'device_time_total', 0) if not key.startswith('engineering::') else None,
                         'native_operator_device_self_us': getattr(e, 'self_device_time_total', 0) if not key.startswith('engineering::') else None})
    trace_path = out / 'first-conditional-trace.json'
    trace = json.loads(trace_path.read_text())
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and 'dur' in e]
    kernel_summary = {'count':len(kernels), 'duration_us':sum(e['dur'] for e in kernels),
                      'scope':'sum of Chrome kernel event durations; excludes gpu_user_annotation; may overlap'}
    return {'trace_sha256':file_hash(trace_path), 'cuda_kernel_events':kernel_summary, 'episode_position':0, 'stage_index':stage.index, 'parameter_name':stage.parameter_name,
            'preprocessing_and_h2d_wall_seconds':preprocessing,
            'profiled_preparation_and_conditional_wall_seconds':elapsed, 'operator_summary':rows,
            'trace':'first-conditional-trace.json', 'operator_table':'operators.txt',
            'scope':'one conditional only; no optimizer step; not a full update'}


def worker(mode, *, inputs, input_sha256, output, repo, source_commit):
    if mode not in ('profile', 'baseline'):
        raise ValueError('invalid worker mode')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    import numpy as np
    import torch
    from .campaign import CampaignSpec
    from .cuda_training import QualificationTrainingBackend, configure_cuda_determinism
    from .episodes import digest
    from .training import TrainingState
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
    for name in ('run','worker','qualify','qualification-worker','fusion-preflight'):
        q=sub.add_parser(name)
        for arg in ('inputs','input-sha256','output','repo','source-commit'): q.add_argument('--'+arg,required=True)
        if name=='worker': q.add_argument('--mode',choices=('profile','baseline'),required=True)
        if name=='qualification-worker': q.add_argument('--lane',choices=('cache_r4','cache_r16','full_finetune','stock_r4'),required=True)
        if name=='fusion-preflight':
            q.add_argument('--checkpoint',required=True)
            q.add_argument('--checkpoint-sha256',required=True)
            q.add_argument('--checkpoint-source-commit',required=True)
    args=vars(p.parse_args(argv)); op=args.pop('operation')
    result={'run':launch,'worker':worker,'qualify':launch_qualification,'qualification-worker':qualification_worker,'fusion-preflight':fusion_preflight}[op](**args)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


def _rng_snapshot():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())


def _rng_restore(snapshot):
    import numpy as np
    import torch
    random.setstate(snapshot[0]); np.random.set_state(snapshot[1])
    torch.set_rng_state(snapshot[2]); torch.cuda.set_rng_state(snapshot[3])


def _conditional_parity(backend, item):
    """Same weights/inputs, one complete eight-member conditional, no step."""
    import torch
    from .model import CANONICAL_FULL_SUPPORT, configure_differentiable_forward
    from .training_execution import conditional_forward_cache
    from .training_numerics import compare_named_tensors, trainable_gradient_snapshot
    rng=_rng_snapshot()
    try:
        prepared=backend.prepare(item); stage=prepared.stages[0]
        outcomes=[]
        for enabled in (False, True):
            _rng_restore(rng)
            backend.optimizer.zero_grad(set_to_none=True)
            configure_differentiable_forward(backend.model)
            captured=[]; original=stage.ensemble.adjusted_logits
            def grab(*args, **kwargs):
                values=original(*args, **kwargs)
                captured.extend(x.detach().cpu().clone() for x in values)
                return values
            object.__setattr__(stage.ensemble,'adjusted_logits',grab)
            try:
                with conditional_forward_cache(backend.model,enabled=enabled):
                    density=stage.ensemble.density(backend.model,law=CANONICAL_FULL_SUPPORT)
                    score=-density.log_prob(stage.targets,log_abs_det_decode=stage.decoder_log_jac).sum()
                    evidence={'logits':captured,'density_logits':density.logits.detach().cpu().clone(),
                              'density_weights':density.weights.detach().cpu().clone(),
                              'loss':score.detach().cpu().clone()}
                    score.backward()
                evidence['gradients']=trainable_gradient_snapshot(backend.model)
                outcomes.append(evidence)
            finally:
                object.__delattr__(stage.ensemble,'adjusted_logits')
        left,right=outcomes
        exact=(len(left['logits'])==8 and len(right['logits'])==8 and
               all(torch.equal(x,y) for x,y in zip(left['logits'],right['logits'])) and
               all(torch.equal(left[k],right[k]) for k in ('density_logits','density_weights','loss')))
        gradients=compare_named_tensors(left['gradients'],right['gradients'],atol=2e-7,rtol=2e-5)
        return {'episode_position':0,'stage_index':stage.index,'parameter_name':stage.parameter_name,
                'members':8,'logits_density_loss_bitwise_equal':exact,'gradient_comparison':gradients,
                'passed':exact and gradients['passed']}
    finally:
        backend.optimizer.zero_grad(set_to_none=True)
        _rng_restore(rng)


def qualification_worker(lane, *, inputs, input_sha256, output, repo, source_commit):
    lanes={'cache_r4':('lora_r4',True),'cache_r16':('lora_r16',True),
           'full_finetune':('full_finetune',True),'stock_r4':('lora_r4',False)}
    if lane not in lanes: raise ValueError('unknown qualification lane')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    import numpy as np
    import torch
    from .campaign import CampaignSpec, save_campaign_checkpoint, restore_campaign_checkpoint
    from .cuda_training import QualificationTrainingBackend, configure_cuda_determinism
    from .cuda_qualification import _state_digest, _rng_probe
    from .episodes import digest
    from .model import weights_fingerprint
    from .training import TrainingState
    value=verify_bindings(inputs,input_sha256,repo,source_commit)
    out=Path(output); out.mkdir(parents=True,exist_ok=False); method,cached=lanes[lane]
    result={'schema':'finetuneforge.training_engineering_qualification.v1','lane':lane,'status':'running',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
            'job_id':os.environ['SLURM_JOB_ID'],'source_commit':source_commit,
            'input_manifest_sha256':input_sha256,'base_checkpoint_sha256':value['base_checkpoint_sha256'],
            'posterior_requests':0,'posterior_draws_contract':DRAW_COUNT,'scientifically_qualified':False}
    write_json(out/'result.json',result); started=time.perf_counter()
    try:
        configure_cuda_determinism();torch.set_num_threads(4)
        if torch.cuda.device_count()!=1: raise RuntimeError('exactly one visible GPU per worker required')
        props=torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory<39*1024**3: raise RuntimeError('A100 40GB or larger required')
        result['hardware']={'name':props.name,'memory_bytes':props.total_memory,
                            'uuid':str(getattr(props,'uuid','unavailable')),'cuda_runtime':torch.version.cuda}
        root=Path(inputs).parent
        batches=[[SimpleNamespace(context=_load_bank(root,row['context']),query=_load_bank(root,row['query']),receipt=row)
                  for row in value['episodes'][start:start+16]] for start in (0,16)]
        rate=1e-5 if method=='full_finetune' else 1e-4
        spec=CampaignSpec(method,0,'pilot',rate,2,value['protocol_sha256'],effective_batch=16,diagnostic=True)
        stream_id=digest(value['episodes']); result['spec']=asdict(spec)
        identity={'spec':asdict(spec),'stream_id':stream_id,'protocol_sha256':value['protocol_sha256'],
                  'qualification_inputs_sha256':input_sha256,'source_commit':source_commit,'cache_parametrizations':cached}
        def fresh():
            b=QualificationTrainingBackend(spec,value['base_checkpoint'],device='cuda:0',allow_cuda_qualification=True,
                                           cache_parametrizations=cached)
            b.episode_rng=np.random.default_rng(SEED+5000)
            torch.manual_seed(spec.initialization_seed);torch.cuda.manual_seed_all(spec.initialization_seed)
            np.random.seed(SEED);random.seed(SEED)
            return b
        backend=fresh();state=TrainingState(stream_id=stream_id)
        torch.cuda.reset_peak_memory_stats()
        if lane in ('cache_r4','stock_r4'):
            rng=_rng_snapshot()
            try: result['conditional_profile']=_profile_conditional(backend,batches[0],out)
            finally: _rng_restore(rng);backend.optimizer.zero_grad(set_to_none=True)
        parity=[]
        if cached:
            parity.append(_conditional_parity(backend,batches[0][0]))
            result['parity']=parity;write_json(out/'result.json',result)
            if not parity[-1]['passed']: raise RuntimeError('initial conditional parity failed')
        reports=[];times=[]
        t=time.perf_counter();reports.append(backend.update(batches[0],state));torch.cuda.synchronize();times.append(time.perf_counter()-t)
        if cached:
            parity.append(_conditional_parity(backend,batches[0][0]))
            result['parity']=parity;write_json(out/'result.json',result)
            if not parity[-1]['passed']: raise RuntimeError('nonzero adapter conditional parity failed')
        save_campaign_checkpoint(out/'after-one-update',backend,state,identity=identity,history=[{'update':1,'loss':reports[0]['loss']}])
        t=time.perf_counter();reports.append(backend.update(batches[1],state));torch.cuda.synchronize();times.append(time.perf_counter()-t)
        result.update(updates=reports,unprofiled_update_wall_seconds=times,parity=parity,
                      final_model_sha256=weights_fingerprint(backend.model))
        final_sha=save_campaign_checkpoint(out/'after-two-update',backend,state,identity=identity,
                   history=[{'update':i+1,'loss':r['loss']} for i,r in enumerate(reports)])
        result['checkpoint_manifest_sha256']=final_sha
        if cached:
            expected_optimizer=_state_digest(backend.optimizer.state_dict());expected_rng=_rng_probe(backend.episode_rng)
            del backend;torch.cuda.empty_cache()
            resumed=fresh(); restored,_=restore_campaign_checkpoint(out/'after-one-update',resumed,expected_identity=identity)
            replay=resumed.update(batches[1],restored);torch.cuda.synchronize()
            # Match the export side effects that precede the continuous RNG probe.
            save_campaign_checkpoint(out/'replay-two-update',resumed,restored,identity=identity,
                history=[{'update':1,'loss':reports[0]['loss']},{'update':2,'loss':replay['loss']}])
            restart={'weights_bitwise_equal':weights_fingerprint(resumed.model)==result['final_model_sha256'],
                     'optimizer_bitwise_equal':_state_digest(resumed.optimizer.state_dict())==expected_optimizer,
                     'future_all_rngs_equal':_rng_probe(resumed.episode_rng)==expected_rng,
                     'state_equal':asdict(restored)==asdict(state),'replay_update':replay}
            restart['passed']=all(restart[k] for k in ('weights_bitwise_equal','optimizer_bitwise_equal','future_all_rngs_equal','state_equal'))
            from .model_loading import load_inference_model
            loaded,_=load_inference_model(value['base_checkpoint'],expected_model_sha256=result['final_model_sha256'],
                campaign_checkpoint=out/'after-two-update',expected_checkpoint_manifest_sha256=final_sha,device='cuda:0')
            resumed.model.requires_grad_(False); resumed.model.eval()
            prepared=resumed.prepare(batches[0][0])
            with torch.no_grad():
                reference=resumed.model(prepared.stages[0].ensemble.members)
                deployment=loaded(prepared.stages[0].ensemble.members)
            restart['deployment_bitwise_equal']=len(reference)==8 and all(torch.equal(x,y) for x,y in zip(reference,deployment))
            restart['passed']=restart['passed'] and restart['deployment_bitwise_equal']
            result['restart']=restart
            if not restart['passed']: raise RuntimeError('cached interrupted replay failed')
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),status='completed')
    except BaseException as exc:
        result.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        result['worker_wall_seconds']=time.perf_counter()-started;write_json(out/'result.json',result)
    return result


def launch_qualification(*, inputs, input_sha256, output, repo, source_commit):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT') != 'eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    verify_bindings(inputs,input_sha256,repo,source_commit)
    devices=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if len(devices)!=4 or len(set(devices))!=4: raise RuntimeError('four distinct allocated GPUs required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);jobs=[]
    try:
        for lane,device in zip(('cache_r4','cache_r16','full_finetune','stock_r4'),devices):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
            log=(out/(lane+'.log')).open('w')
            cmd=[sys.executable,str(Path(repo)/'scripts/engineer-study-training.py'),'qualification-worker',
                 '--lane',lane,'--inputs',str(inputs),'--input-sha256',input_sha256,
                 '--output',str(out/lane),'--repo',str(repo),'--source-commit',source_commit]
            jobs.append((lane,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        codes={lane:proc.wait() for lane,proc,_ in jobs}
    finally:
        for _,proc,log in jobs:
            if proc.poll() is None: proc.terminate();proc.wait()
            log.close()
    summary={'schema':'finetuneforge.training_engineering_qualification_allocation.v1',
             'worker_returncodes':codes,'input_manifest_sha256':input_sha256,'source_commit':source_commit,
             'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],
             'posterior_requests':0,'scientifically_qualified':False}
    write_json(out/'allocation.json',summary)
    if any(code!=0 for code in codes.values()): raise RuntimeError('qualification worker failed; inspect retained evidence')
    return summary


def verify_fusion_checkpoint(checkpoint, checkpoint_sha256, checkpoint_source_commit, spec, stream_id, input_sha256):
    """Check manifest, complete inventory and identity before restoring any payload."""
    from .campaign import read_campaign_checkpoint
    path=Path(checkpoint)
    if path.is_symlink() or (path/'manifest.json').is_symlink() or file_hash(path/'manifest.json') != checkpoint_sha256:
        raise ValueError('fusion checkpoint manifest checksum mismatch')
    metadata=read_campaign_checkpoint(path)
    identity=metadata['identity']
    if (identity.get('spec')!=asdict(spec) or identity.get('stream_id')!=stream_id
            or identity.get('source_commit')!=checkpoint_source_commit
            or identity.get('qualification_inputs_sha256')!=input_sha256
            or identity.get('cache_parametrizations') is not True
            or metadata['state'].get('update')!=2 or metadata['state'].get('data_position')!=32
            or metadata.get('deployment',{}).get('campaign_method')!='lora_r4'):
        raise ValueError('fusion checkpoint identity mismatch')
    return metadata


def _fusion_stage(backend, stage):
    import torch
    from .model import CANONICAL_FULL_SUPPORT, configure_differentiable_forward
    from .training_execution import conditional_forward_cache
    from .training_numerics import trainable_gradient_snapshot
    configure_differentiable_forward(backend.model)
    backend.optimizer.zero_grad(set_to_none=True)
    raw=[]; original=backend.model.forward_member
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);raw.append(value.detach().cpu().clone());return value
    object.__setattr__(backend.model,'forward_member',capture)
    started=time.perf_counter()
    try:
        with conditional_forward_cache(backend.model,enabled=True):
            density=stage.ensemble.density(backend.model,law=CANONICAL_FULL_SUPPORT)
            score=-density.log_prob(stage.targets,log_abs_det_decode=stage.decoder_log_jac).sum()
            grid=torch.unique(torch.cat((torch.linspace(-8.,8.,257,dtype=torch.float64,device='cuda:0'),density.borders)))
            levels=torch.tensor([1e-9,1e-5,.01,.5,.99,1-1e-5,1-1e-9],dtype=torch.float64,device='cuda:0')
            cdf=density.cdf(grid).detach().cpu();quantiles=density.icdf(levels).detach().cpu()
            loss=score.detach().cpu().clone()
            score.backward()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        return {'logits':raw,'cdf':cdf,'quantiles':quantiles,'loss':loss,
                'gradients':trainable_gradient_snapshot(backend.model),'wall_seconds':elapsed}
    finally:
        object.__delattr__(backend.model,'forward_member')
        backend.optimizer.zero_grad(set_to_none=True)


def _fusion_compare(reference, candidate):
    import torch
    from .training_numerics import compare_named_tensors
    logits=compare_named_tensors({str(i):x for i,x in enumerate(reference['logits'])},
                                  {str(i):x for i,x in enumerate(candidate['logits'])},atol=1e-4,rtol=1e-3)
    gradients=compare_named_tensors(reference['gradients'],candidate['gradients'],atol=2e-7,rtol=2e-5)
    cdf_error=float((reference['cdf']-candidate['cdf']).abs().max())
    quantile_error=float((reference['quantiles']-candidate['quantiles']).abs().max())
    qpass=bool(torch.allclose(reference['quantiles'],candidate['quantiles'],atol=1e-3,rtol=1e-3))
    loss_error=float((reference['loss']-candidate['loss']).abs().max())
    loss_pass=bool(torch.allclose(reference['loss'],candidate['loss'],atol=1e-3,rtol=0))
    result={'members':len(reference['logits']),'logits':logits,'gradients':gradients,
            'cdf_max_absolute_error':cdf_error,'cdf_atol':1e-4,
            'quantile_max_absolute_error':quantile_error,'quantile_passed':qpass,
            'quantile_atol':1e-3,'quantile_rtol':1e-3,
            'loss_max_absolute_error':loss_error,'loss_atol':1e-3,'loss_passed':loss_pass,
            'eager_wall_seconds':reference['wall_seconds'],'compiled_wall_seconds':candidate['wall_seconds'],
            'timing_scope':'conditional with instrumentation, CDF/quantiles, backward and synchronization; excludes preprocessing and gradient snapshot'}
    result['passed']=(len(reference['logits'])==8 and len(candidate['logits'])==8 and logits['passed'] and
                      gradients['passed'] and cdf_error<=1e-4 and qpass and loss_pass)
    return result


def triton_fp_fusion_policy(environ):
    """Record Triton's pinned compiler default before importing Torch."""
    selected = environ.get('TRITON_DEFAULT_FP_FUSION', '1')
    if selected not in ('0', '1'):
        raise ValueError('TRITON_DEFAULT_FP_FUSION must be 0 or 1')
    return {'environment_value':environ.get('TRITON_DEFAULT_FP_FUSION'),
            'effective_value':selected, 'enabled':selected=='1',
            'meaning':'Triton compiler default; explicit operations and external kernels may differ'}


def fusion_preflight(*, inputs, input_sha256, output, repo, source_commit,
                     checkpoint, checkpoint_sha256, checkpoint_source_commit):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    fp_fusion_policy=triton_fp_fusion_policy(os.environ)
    import numpy as np
    import torch
    from .campaign import CampaignSpec, restore_campaign_checkpoint
    from .cuda_training import QualificationTrainingBackend, configure_cuda_determinism
    from .episodes import digest
    from .model import weights_fingerprint
    from .training_compilation import compiled_training_mlps
    value=verify_bindings(inputs,input_sha256,repo,source_commit)
    spec=CampaignSpec('lora_r4',0,'pilot',1e-4,2,value['protocol_sha256'],effective_batch=16,diagnostic=True)
    stream_id=digest(value['episodes'])
    checkpoint_metadata=verify_fusion_checkpoint(checkpoint,checkpoint_sha256,checkpoint_source_commit,spec,stream_id,input_sha256)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result={'schema':'finetuneforge.training_fusion_preflight.v1','status':'running',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
            'job_id':os.environ['SLURM_JOB_ID'],'source_commit':source_commit,'input_manifest_sha256':input_sha256,
            'checkpoint_manifest_sha256':checkpoint_sha256,'checkpoint_source_commit':checkpoint_source_commit,
            'checkpoint_weights_sha256':checkpoint_metadata['weights_sha256'], 'optimizer_updates':0,'posterior_requests':0,
            'scientifically_qualified':False,'policies':{'cache':'per_layer_forward_and_checkpoint_recompute.v1',
            'neural':'fp32','probability':'fp64','tf32':False,'amp':False,'sdpa':'math','fullgraph':True,'dynamic':False},
            'compile_cache_path':os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
            'triton_fp_fusion':fp_fusion_policy}
    write_json(out/'result.json',result);started=time.perf_counter()
    try:
        configure_cuda_determinism();torch.set_num_threads(4)
        if torch.cuda.device_count()!=1: raise RuntimeError('exactly one visible GPU required')
        props=torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory<39*1024**3: raise RuntimeError('A100 40GB or larger required')
        if not result['compile_cache_path']:raise RuntimeError('explicit TORCHINDUCTOR_CACHE_DIR required')
        result['hardware']={'name':props.name,'memory_bytes':props.total_memory,'uuid':str(getattr(props,'uuid','unavailable'))}
        backend=QualificationTrainingBackend(spec,value['base_checkpoint'],device='cuda:0',allow_cuda_qualification=True,cache_parametrizations=True)
        backend.episode_rng=np.random.default_rng(SEED+5000)
        torch.manual_seed(spec.initialization_seed);torch.cuda.manual_seed_all(spec.initialization_seed)
        np.random.seed(SEED);random.seed(SEED)
        root=Path(inputs).parent;row=value['episodes'][0]
        item=SimpleNamespace(context=_load_bank(root,row['context']),query=_load_bank(root,row['query']),receipt=row)
        prepared=backend.prepare(item)
        if len(prepared.stages)!=4:raise RuntimeError('four teacher-forced conditional stages required')
        rng=_rng_snapshot();initial={k:v.detach().cpu().clone() for k,v in backend.model.state_dict().items()}
        initial_hash=weights_fingerprint(backend.model);result['initial_weights_sha256']=initial_hash
        if initial_hash==checkpoint_metadata['weights_sha256']:raise RuntimeError('checkpoint did not change initial model weights')
        def load_initial():
            backend.model.load_state_dict(initial,strict=True);_rng_restore(rng)
            if weights_fingerprint(backend.model)!=initial_hash:raise RuntimeError('initial weights restore failed')
        def load_nonzero():
            restored,_=restore_campaign_checkpoint(checkpoint,backend,expected_identity=checkpoint_metadata['identity'])
            if restored.update!=2 or weights_fingerprint(backend.model)!=checkpoint_metadata['weights_sha256']:
                raise RuntimeError('nonzero checkpoint restore failed')
            _rng_restore(rng)
        eager={}
        for name,load in (('initial',load_initial),('nonzero',load_nonzero)):
            load();eager[name]=[]
            for stage in prepared.stages:
                result['progress']={'phase':'eager_reference','weights':name,'stage':stage.index};write_json(out/'result.json',result)
                _rng_restore(rng);eager[name].append(_fusion_stage(backend,stage))
        from torch._dynamo.utils import counters
        counters.clear();torch._dynamo.reset()
        load_initial()
        with compiled_training_mlps(backend.model,enabled=True,backend='inductor') as metadata:
            result['compile_policy']=metadata;result['comparisons']={};result['warmup_wall_seconds']={}
            for name,load in (('initial',load_initial),('nonzero',load_nonzero)):
                load();t=time.perf_counter()
                for stage in prepared.stages:
                    result['progress']={'phase':'compiled_warmup','weights':name,'stage':stage.index};write_json(out/'result.json',result)
                    _rng_restore(rng);_fusion_stage(backend,stage)
                result['warmup_wall_seconds'][name]=time.perf_counter()-t
                rows=[]
                for stage,reference in zip(prepared.stages,eager[name]):
                    result['progress']={'phase':'compiled_comparison','weights':name,'stage':stage.index};write_json(out/'result.json',result)
                    _rng_restore(rng);candidate=_fusion_stage(backend,stage)
                    comparison=_fusion_compare(reference,candidate)
                    comparison.update(stage_index=stage.index,parameter_name=stage.parameter_name)
                    rows.append(comparison)
                result['comparisons'][name]=rows
                result['dynamo_counters']={k:dict(v) for k,v in counters.items()}
                write_json(out/'result.json',result)
                if not all(x['passed'] for x in rows):raise RuntimeError(name+' compiled numerical parity failed')
            result['dynamo_counters']={k:dict(v) for k,v in counters.items()}
            if result['dynamo_counters'].get('graph_break'):raise RuntimeError('Dynamo graph break detected')
            if result['dynamo_counters'].get('stats',{}).get('unique_graphs',0)<1:raise RuntimeError('no compiled Dynamo graphs recorded')
        result.pop('progress',None);result['status']='preflight_passed'
    except BaseException as exc:
        result.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        result['wall_seconds']=time.perf_counter()-started;write_json(out/'result.json',result)
    return result
