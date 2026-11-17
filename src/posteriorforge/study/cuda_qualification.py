"""Explicit I07 engineering pilot; neither production training nor test selection.

Four independent GPU workers qualify numerical parity, training/restart and
complete 50,000-vector sampling. Large-shape memory failures remain evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

LANES = ('frozen', 'lora_r4', 'lora_r16', 'full_finetune')
DRAW_COUNT = 50000
SEED = 20261117


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_files(repo):
    repo = Path(repo)
    names = sorted((repo / 'src/posteriorforge/study').glob('*.py'))
    names += [repo / 'scripts/qualify-study-cuda.py', repo / 'environments/study-training-gpu.lock',
              repo / 'scripts/check-study-orchestration-cpu.py', repo / 'slurm/run-study-qualification.sh']
    return {p.relative_to(repo).as_posix(): file_hash(p) for p in names}


def tiny_problem(*, posterior=False):
    from .spec import ProblemSpec, ParameterSpec, PriorSpec, FamilySpec, DatasetSpec
    params = (ParameterSpec('mu','population',PriorSpec('normal',0.,1.)),
              ParameterSpec('tau','population',PriorSpec('lognormal',.2,.05)),
              ParameterSpec('a','physical',PriorSpec('normal','mu','tau')),
              ParameterSpec('sigma','noise',PriorSpec('lognormal',.1,.03)))
    if posterior:
        return ProblemSpec('cuda-zero-sensor-control','diagnostic',params,
            (FamilySpec('zero-sensor','sensor','sigma'),),
            (DatasetSpec('ten-zero-readings','zero-sensor','specimen-0',{'p0':'a'},matrix=((0.,),)*10),),
            {'purpose':'exact analytical development control; no transfer claim'})
    return ProblemSpec('cuda-training-scalar','diagnostic',params,
        (FamilySpec('scalar','scalar','sigma'),),
        (DatasetSpec('one-reading','scalar','specimen-0',{'a':'a'}),),
        {'purpose':'matched bounded GPU engineering; no production update'})


def prepare_inputs(output, *, repo, base_checkpoint, source_commit):
    from .banks import prepare_bank, save_bank
    from .catalogue import generate_setup
    from .workflows import protocol_identity
    from .model import TabPFNBackbone, weights_fingerprint
    from .spec import ProblemSpec
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    def bank(spec,count,seed,name):
        b=prepare_bank(spec,count,seed); p=root/name; m=save_bank(b,p)
        return {'directory':name,'manifest_sha256':m['manifest_sha256'],
                'worlds':count,'seed':seed,'problem':spec.fingerprint,
                'dimension':spec.dimension,'observations':spec.observation_dimension,
                'forward_evaluations':b.manifest['cost']['forward_evaluations']}
    model=TabPFNBackbone.from_checkpoint(base_checkpoint)
    value={'schema':'finetuneforge.cuda_qualification_inputs.v1','source_commit':source_commit,
           'source_files':source_files(repo),'base_checkpoint':str(Path(base_checkpoint).resolve()),
           'base_checkpoint_sha256':file_hash(base_checkpoint),'frozen_model_sha256':weights_fingerprint(model),
           'protocol_sha256':protocol_identity()['protocol_sha256'], 'posterior_draws':DRAW_COUNT,
           'lanes':list(LANES),'master_seed':SEED,'scientifically_qualified':False,
           'fp32_policy':{'tf32':False,'deterministic_algorithms':True,'sdpa':'math'},
           'posterior_context_worlds':256,'posterior_microbatch_size':1024,
           'training_effective_batch':16,'engineering_updates':2,
           'pilot_learning_rate_selection':False,'main_campaign_enabled':False,
           'reference_scope':'analytical zero-sensor marginals only, not full study gate'}
    train=tiny_problem(); value['episodes']=[]
    for pos in range(32):
        value['episodes'].append({'position':pos,
            'context':bank(train,16,SEED+100+2*pos,f'episodes/{pos:03d}/context'),
            'query':bank(train,1,SEED+101+2*pos,f'episodes/{pos:03d}/query')})
    post=tiny_problem(posterior=True)
    value['posterior_bank']=bank(post,256,SEED+1000,'posterior-bank')
    import numpy as np
    np.save(root/'observation.npy',np.zeros(10,dtype=np.float64),allow_pickle=False)
    value['observation_sha256']=file_hash(root/'observation.npy')
    value['probes']=[]
    for index,band,contexts in [(501,'medium',(1024,)),(502,'large',(1024,4096))]:
        original=generate_setup(index,split='validation',size_band=band)
        d=original.to_dict(); d['split']='diagnostic'; spec=ProblemSpec.from_dict(d)
        for count in contexts:
            name=f'probes/{band}-{count}'
            value['probes'].append({'band':band,'context':bank(spec,count,SEED+2000+index+count,name+'/context'),
                                    'query':bank(spec,1,SEED+3000+index+count,name+'/query')})
    value['data_forward_evaluations']=sum(row[k]['forward_evaluations'] for row in value['episodes'] for k in ('context','query'))+value['posterior_bank']['forward_evaluations']+sum(row[k]['forward_evaluations'] for row in value['probes'] for k in ('context','query'))
    write_json(root/'manifest.json',value)
    return {'input_manifest':str(root/'manifest.json'),'input_manifest_sha256':file_hash(root/'manifest.json'),
            'forward_evaluations':value['data_forward_evaluations'],'posterior_draws_per_request':DRAW_COUNT}


def verify_inputs(path, expected_sha, repo):
    path=Path(path)
    if file_hash(path)!=expected_sha: raise ValueError('qualification input manifest checksum mismatch')
    value=json.loads(path.read_text())
    if value.get('schema')!='finetuneforge.cuda_qualification_inputs.v1' or value.get('posterior_draws')!=DRAW_COUNT or value.get('lanes')!=list(LANES):
        raise ValueError('qualification input contract mismatch')
    if source_files(repo)!=value['source_files']: raise ValueError('qualification source/environment bytes changed')
    if file_hash(value['base_checkpoint'])!=value['base_checkpoint_sha256']:raise ValueError('checkpoint changed')
    if file_hash(path.parent/'observation.npy')!=value['observation_sha256']:raise ValueError('query changed')
    return value


def _load_bank(root, desc):
    from .banks import load_bank
    return load_bank(Path(root)/desc['directory'],expected_sha256=desc['manifest_sha256'])


def _analytic_module(repo):
    spec=importlib.util.spec_from_file_location('ftf_exact_analytic_control',Path(repo)/'scripts/check-study-orchestration-cpu.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _state_digest(value):
    import torch
    import numpy as np
    h=sha256()
    def feed(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous(); h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());feed(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:feed(x)
        else:h.update(repr(v).encode())
    feed(value); return h.hexdigest()


def _rng_probe(episode_rng):
    import torch
    import numpy as np
    return _state_digest((torch.rand(8),torch.rand(8,device='cuda'),np.random.random(8),random.random(),episode_rng.random(8)))


def _parity(value, root, output):
    import torch
    import numpy as np
    from .model import TabPFNBackbone, CANONICAL_FULL_SUPPORT
    from .training import prepare_teacher_forced_episode
    from tabpfn import TabPFNRegressor
    cp=value['base_checkpoint']; item=value['episodes'][0]
    context=_load_bank(root,item['context']); query=_load_bank(root,item['query'])
    factory=lambda:TabPFNRegressor(model_path=cp,device='cpu',n_estimators=8,random_state=0,n_jobs=1,
        inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    cpu=TabPFNBackbone.from_checkpoint(cp).eval(); gpu=TabPFNBackbone.from_checkpoint(cp).to('cuda').eval()
    a=prepare_teacher_forced_episode(context,query,regressor_factory=factory,law=CANONICAL_FULL_SUPPORT,device='cpu')
    b=prepare_teacher_forced_episode(context,query,regressor_factory=factory,law=CANONICAL_FULL_SUPPORT,device='cuda')
    rows=[]
    with torch.no_grad():
        for left,right in zip(a.stages,b.stages):
            raw_cpu=cpu(left.ensemble.members); raw_gpu=gpu(right.ensemble.members)
            if len(raw_cpu)!=8 or len(raw_gpu)!=8:raise RuntimeError('all eight member predictions required')
            member_errors=[float((l-r.cpu()).abs().max()) for l,r in zip(raw_cpu,raw_gpu)]
            member_pass=all(torch.allclose(l,r.cpu(),rtol=1e-3,atol=1e-4) for l,r in zip(raw_cpu,raw_gpu))
            x=left.ensemble.density(cpu,law=CANONICAL_FULL_SUPPORT); y=right.ensemble.density(gpu,law=CANONICAL_FULL_SUPPORT)
            p=torch.tensor([1e-9,1e-5,.01,.5,.99,1-1e-5,1-1e-9],dtype=torch.float64)
            grid=torch.unique(torch.cat((torch.linspace(-8.,8.,257,dtype=torch.float64),x.borders.cpu())))
            cdf=float((x.cdf(grid)-y.cdf(grid.to('cuda')).cpu()).abs().max())
            score=float((x.log_prob(left.targets)-y.log_prob(right.targets).cpu()).abs().max())
            qcpu=x.icdf(p);qgpu=y.icdf(p.to('cuda')).cpu()
            quantile=float((qcpu-qgpu).abs().max());quantile_pass=bool(torch.allclose(qcpu,qgpu,rtol=1e-3,atol=1e-3))
            ok=bool(member_pass) and cdf<=1e-4 and score<=1e-3 and quantile_pass
            rows.append({'stage':left.index,'eight_raw_member_max_abs_logits':member_errors,
                'max_abs_cdf_over_grid':cdf,'cdf_grid_points':len(grid),'max_abs_log_prob_at_truth':score,
                'max_abs_quantile':quantile,'quantile_levels':p.tolist(),'passed':ok})
    result={'scope':'same fixed inputs, all eight pristine member predictions, FP32 neural/FP64 law',
        'cdf_grid':'257 equally spaced values in [-8,8] plus all shared output borders',
        'tolerances':{'member_logits_rtol':1e-3,'member_logits_atol':1e-4,'cdf_atol':1e-4,'log_prob_atol':1e-3,'quantile_rtol':1e-3,'quantile_atol':1e-3},
        'rows':rows,'passed':all(r['passed'] for r in rows)}
    write_json(output/'cpu-cuda-parity.json',result)
    if not result['passed']:raise RuntimeError('CPU/CUDA numerical parity failed')
    del a,b,cpu,gpu;torch.cuda.empty_cache()
    return result


def _profile_conditional(value, root, backend, output):
    import torch
    import numpy as np
    from .science import encode_prior
    context=_load_bank(root,value['posterior_bank']); z=encode_prior(context.spec,context.theta)
    stage=backend.prepare(context.observations,z[:,0])
    query=np.zeros((32,context.spec.observation_dimension))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as trace:
        backend.sample(stage,query,np.linspace(.01,.99,32));torch.cuda.synchronize()
    trace.export_chrome_trace(str(output/'conditional-trace.json'))
    (output/'conditional-profile.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))


def plot_neural_posterior(samples, prior_sigma, output, metrics, *, method):
    """Explicit neural-output figure with exact analytical curves as references."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import norm, lognorm
    logvar=np.log1p((.03/.1)**2); logsd=np.sqrt(logvar)
    priorloc=np.log(.1)-.5*logvar; postloc=priorloc-10*logvar
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    fig.suptitle(f'CUDA neural posterior control: {method}\n50,000 complete vectors; engineering pilot, no transfer claim')
    axes[0].hist(samples[:,0],bins=70,density=True,alpha=.6,label='Neural posterior')
    x=np.linspace(-4,4,500);axes[0].plot(x,norm.pdf(x),color='black',label='Exact posterior = prior')
    axes[0].set(xlabel='Population mean mu',ylabel='Density',xlim=(-4,4));axes[0].legend()
    axes[1].hist(prior_sigma,bins=80,density=True,alpha=.35,label='Prior')
    axes[1].hist(samples[:,3],bins=80,density=True,alpha=.6,label='Neural posterior')
    x=np.linspace(.0001,.25,500)
    axes[1].plot(x,lognorm.pdf(x,s=logsd,scale=np.exp(postloc)),color='black',label='Exact posterior')
    axes[1].set(xlabel='Inferred measurement-noise SD',ylabel='Density',xlim=(0,.25));axes[1].legend()
    axes[2].scatter(samples[:,0],samples[:,2],s=.6,alpha=.06,rasterized=True)
    axes[2].plot([-4,4],[-4,4],ls='--',color='black')
    axes[2].set(xlabel='Population mean mu',ylabel='Specimen property a',xlim=(-4,4),ylim=(-4,4))
    axes[2].set_title('All 50,000 vectors; shared population uncertainty')
    fig.savefig(output,dpi=160);plt.close(fig)


def _posterior(value, root, model, output, *, repo, method):
    import torch
    import numpy as np
    from .model import weights_fingerprint
    from .sampling_backend import TabPFNConditionalBackend
    from .sampling import sample_posterior
    context=_load_bank(root,value['posterior_bank']); observation=np.load(Path(root)/'observation.npy',allow_pickle=False)
    model.eval(); digest=weights_fingerprint(model)
    backend=TabPFNConditionalBackend(value['base_checkpoint'],model,expected_model_sha256=digest,device='cuda:0')
    _profile_conditional(value,root,backend,output)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    samples,metadata=sample_posterior(context,observation,backend,seed=SEED+4000,microbatch_size=value['posterior_microbatch_size'],
        identities={'model_sha256':digest,'bank_manifest_sha256':value['posterior_bank']['manifest_sha256'],'protocol_sha256':value['protocol_sha256']},
        resume_path=output/'recovery.npz',cuda_qualification=True)
    torch.cuda.synchronize()
    if samples.shape!=(DRAW_COUNT,context.spec.dimension):raise RuntimeError('complete-vector contract failed')
    # Save all raw complete vectors; this diagnostic is not admitted to study selection.
    np.savez(output/'samples.npz',samples=samples)
    control=_analytic_module(repo); metrics=control.analyze_samples(samples)
    rng=np.random.default_rng(SEED+4001)
    prior_sigma=rng.lognormal(control.analytic_parameters()['sigma_prior_log_mean'],control.analytic_parameters()['sigma_log_sd'],DRAW_COUNT)
    plot_neural_posterior(samples,prior_sigma,output/'posterior.png',metrics,method=method)
    summary={'method':method,'posterior_draws':len(samples),'dimension':context.spec.dimension,
        'wall_seconds':time.perf_counter()-started,'sampling':metadata,'analytical_diagnostics':metrics,
        'samples_sha256':file_hash(output/'samples.npz'),'figure_sha256':file_hash(output/'posterior.png'),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'scientifically_qualified':False,'scope':'development analytical marginal control, not full scientific screen or simulation savings'}
    write_json(output/'posterior.json',summary);return summary


def _training(value, root, output, method):
    import torch
    import numpy as np
    from types import SimpleNamespace
    from .campaign import CampaignSpec, save_campaign_checkpoint, restore_campaign_checkpoint
    from .cuda_training import QualificationTrainingBackend
    from .training import TrainingState
    from .model import weights_fingerprint
    from .episodes import digest
    rate=1e-5 if method=='full_finetune' else 1e-4
    spec=CampaignSpec(method,0,'pilot',rate,2,value['protocol_sha256'],effective_batch=16,diagnostic=True)
    stream_id=digest(value['episodes'])
    identity={'spec':asdict(spec),'stream_id':stream_id,'protocol_sha256':value['protocol_sha256'],'qualification_inputs_sha256':file_hash(Path(root)/'manifest.json'),'source_files':value['source_files']}
    batches=[]
    for start in (0,16):
        batches.append([SimpleNamespace(context=_load_bank(root,r['context']),query=_load_bank(root,r['query']),receipt=r) for r in value['episodes'][start:start+16]])
    def fresh():
        result=QualificationTrainingBackend(spec,value['base_checkpoint'],device='cuda:0',allow_cuda_qualification=True)
        result.episode_rng=np.random.default_rng(SEED+5000)
        torch.manual_seed(spec.initialization_seed);torch.cuda.manual_seed_all(spec.initialization_seed)
        np.random.seed(SEED);random.seed(SEED)
        return result
    full=fresh(); before=weights_fingerprint(full.model);state=TrainingState(stream_id=stream_id);history=[]
    # Profile one conditional backward without stepping or perturbing training RNG.
    from .model import CANONICAL_FULL_SUPPORT
    py_rng=random.getstate();np_rng=np.random.get_state()
    with torch.random.fork_rng(devices=[0]):
        prepared=full.prepare(batches[0][0]);stage=prepared.stages[0]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            density=stage.ensemble.density(full.model,law=CANONICAL_FULL_SUPPORT)
            loss=-density.log_prob(stage.targets,log_abs_det_decode=stage.decoder_log_jac).sum()
            loss.backward();torch.cuda.synchronize()
        prof.export_chrome_trace(str(output/'training-conditional-trace.json'))
        (output/'training-conditional-profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=30))
        full.optimizer.zero_grad(set_to_none=True);del prepared,stage,density,loss
    random.setstate(py_rng);np.random.set_state(np_rng)

    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for batch in batches:history.append(full.update(batch,state))
    torch.cuda.synchronize();continuous_seconds=time.perf_counter()-started
    expected_weights=weights_fingerprint(full.model);expected_optimizer=_state_digest(full.optimizer.state_dict()); expected_rng=_rng_probe(full.episode_rng)
    peak={'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
    del full;torch.cuda.empty_cache()
    partial=fresh();partstate=TrainingState(stream_id=stream_id);first=partial.update(batches[0],partstate)
    checkpoint=output/'after-one-update';receipt=save_campaign_checkpoint(checkpoint,partial,partstate,identity=identity,history=[{'update':1,'loss':first['loss']}])
    del partial;torch.cuda.empty_cache()
    resumed=fresh();restored,_=restore_campaign_checkpoint(checkpoint,resumed,expected_identity=identity)
    second=resumed.update(batches[1],restored);torch.cuda.synchronize()
    actual_weights=weights_fingerprint(resumed.model);actual_optimizer=_state_digest(resumed.optimizer.state_dict());actual_rng=_rng_probe(resumed.episode_rng)
    result={'method':method,'engineering_updates':2,'effective_batch':16,'posterior_draws':0,
        'training_data_positions':restored.data_position,'fresh_training_contexts':32,'same_banks_across_methods':True,
        'continuous_seconds':continuous_seconds,'continuous_updates':history,'resumed_second_update':second,
        'initial_weights_sha256':before,'weights_changed':before!=actual_weights,
        'continuous_weights_sha256':expected_weights,'resumed_weights_sha256':actual_weights,
        'weights_bitwise_equal':expected_weights==actual_weights,'optimizer_bitwise_equal':expected_optimizer==actual_optimizer,
        'future_all_rngs_equal':expected_rng==actual_rng,'state':asdict(restored),'checkpoint_manifest_sha256':receipt,
        'peak_memory':peak,'scientifically_qualified':False,'pilot_learning_rate_selected':False}
    result['passed']=result['weights_changed'] and result['weights_bitwise_equal'] and result['optimizer_bitwise_equal'] and result['future_all_rngs_equal'] and restored.update==2 and restored.data_position==32
    write_json(output/'training-restart.json',result)
    if not result['passed']:raise RuntimeError('GPU training/restart qualification failed')
    final=output/'after-two-updates'
    final_sha=save_campaign_checkpoint(final,resumed,restored,identity=identity,history=[{'update':2,'loss':second['loss']}])
    from .model_loading import load_inference_model
    loaded,load_metadata=load_inference_model(value['base_checkpoint'],expected_model_sha256=actual_weights,
        campaign_checkpoint=final,expected_checkpoint_manifest_sha256=final_sha,device='cuda:0')
    resumed.model.requires_grad_(False);resumed.model.eval()
    prepared=resumed.prepare(batches[0][0])
    with torch.no_grad():
        expected=resumed.model(prepared.stages[0].ensemble.members)
        actual=loaded(prepared.stages[0].ensemble.members)
    result['deployment_reload_bitwise_equal']=all(torch.equal(x,y) for x,y in zip(expected,actual))
    result['deployment_loader']=load_metadata
    result['passed']=result['passed'] and result['deployment_reload_bitwise_equal']
    write_json(output/'training-restart.json',result)
    if not result['passed']:raise RuntimeError('CUDA deployment reload changed predictions')
    del resumed,prepared,expected,actual;torch.cuda.empty_cache()
    return loaded,result


def _memory_probes(value, root, model, output, *, selected):
    import torch
    import numpy as np
    from .model import CANONICAL_FULL_SUPPORT, weights_fingerprint
    from .sampling_backend import TabPFNConditionalBackend
    from .science import encode_prior
    rows=[]
    for index in selected:
        probe=value['probes'][index]; row={'probe':probe,'mode':'last_parameter_conditional_q1','status':'running'}
        write_json(output/f'memory-probe-{index}.json',row)
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            context=_load_bank(root,probe['context']);query=_load_bank(root,probe['query'])
            z=encode_prior(context.spec,context.theta);q=encode_prior(query.spec,query.theta)
            backend=TabPFNConditionalBackend(value['base_checkpoint'],model,expected_model_sha256=weights_fingerprint(model),device='cuda:0')
            stage=backend.prepare(np.concatenate((context.observations,z[:,:-1]),axis=1),z[:,-1])
            features=np.concatenate((query.observations,q[:,:-1]),axis=1)
            out=backend.sample(stage,features,np.array([.5]));torch.cuda.synchronize()
            row.update(status='completed',wall_seconds=time.perf_counter()-start,finite=bool(np.isfinite(out).all()))
            del stage,backend
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status='out_of_memory',error=str(exc));torch.cuda.empty_cache()
        finally:
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(output/f'memory-probe-{index}.json',row);rows.append(row)
    return rows


def worker(lane, *, inputs, input_sha256, output, repo):
    import torch
    import numpy as np
    from .cuda_training import configure_cuda_determinism
    from .model import TabPFNBackbone
    if lane not in LANES:raise ValueError('unknown qualification lane')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='eu-26-17':
        raise RuntimeError('qualification requires accounted eu-26-17 Slurm allocation')
    value=verify_inputs(inputs,input_sha256,repo);root=Path(inputs).parent;out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();record={'lane':lane,'status':'running','input_manifest_sha256':input_sha256,'source_commit':value['source_commit'],
        'job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions':{},'scientifically_qualified':False,'posterior_draws_contract':DRAW_COUNT}
    write_json(out/'result.json',record)
    try:
        if lane!='frozen':
            gate=out.parent/'parity-gate.json'
            while not gate.exists():time.sleep(.2)
            result=json.loads(gate.read_text())
            if result.get('input_manifest_sha256')!=input_sha256 or result.get('passed') is not True:
                raise RuntimeError('CPU/CUDA parity prerequisite failed; training not started')
        configure_cuda_determinism();torch.set_num_threads(4)
        if torch.cuda.device_count()!=1:raise RuntimeError('one independent visible GPU per worker required')
        props=torch.cuda.get_device_properties(0)
        if 'A100' not in props.name or props.total_memory<39*1024**3:raise RuntimeError('declared A100 40GB hardware required')
        record['hardware']={'name':props.name,'memory_bytes':props.total_memory,'capability':list(torch.cuda.get_device_capability()),'cuda_runtime':torch.version.cuda,'uuid':str(getattr(props,'uuid','unavailable'))}
        from importlib.metadata import version
        record['versions']={x:version(x) for x in ('torch','tabpfn','peft','transformers','numpy','scipy')}
        write_json(out/'result.json',record)
        if lane=='frozen':
            record['parity']=_parity(value,root,out)
            write_json(out.parent/'parity-gate.json',{'passed':True,'input_manifest_sha256':input_sha256})
            model=TabPFNBackbone.from_checkpoint(value['base_checkpoint']).to('cuda').eval()
            record['posterior']=_posterior(value,root,model,out,repo=repo,method=lane)
        else:
            model,record['training']=_training(value,root,out,lane)
            if lane=='lora_r4':record['posterior']=_posterior(value,root,model,out,repo=repo,method=lane)
            elif lane=='lora_r16':record['memory_probes']=_memory_probes(value,root,model,out,selected=[0])
            else:record['memory_probes']=_memory_probes(value,root,model,out,selected=[1,2])
        record['status']='engineering_checks_passed'
    except BaseException as exc:
        record.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)})
        (out/'traceback.txt').write_text(traceback.format_exc())
        if lane=='frozen' and not (out.parent/'parity-gate.json').exists():
            write_json(out.parent/'parity-gate.json',{'passed':False,'input_manifest_sha256':input_sha256})
        raise
    finally:
        record['wall_seconds']=time.perf_counter()-started;write_json(out/'result.json',record)
    return record


def launch_workers(*, inputs,input_sha256,output,repo):
    # Parent never initializes CUDA; children receive disjoint physical devices.
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    verify_inputs(inputs,input_sha256,repo)
    devices=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if len(devices)!=4 or len(set(devices))!=4:raise RuntimeError('this frozen pilot requires four allocated GPUs')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);processes=[]
    for lane,device in zip(LANES,devices):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1')
        stream=(root/(lane+'.log')).open('w')
        args=[sys.executable,str(Path(repo)/'scripts/qualify-study-cuda.py'),'worker','--lane',lane,'--inputs',str(inputs),'--input-sha256',input_sha256,'--output',str(root/lane),'--repo',str(repo)]
        processes.append((lane,subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT),stream))
    codes={}
    try:
        for lane,p,stream in processes:
            codes[lane]=p.wait();stream.close()
            if lane=='frozen' and not (root/'parity-gate.json').exists():
                write_json(root/'parity-gate.json',{'passed':False,'input_manifest_sha256':input_sha256})
    finally:
        for lane,p,stream in processes:
            if p.poll() is None:p.terminate()
            stream.close()
    summary={'schema':'finetuneforge.cuda_qualification_allocation.v1','job_id':os.environ['SLURM_JOB_ID'],
        'input_manifest_sha256':input_sha256,'worker_returncodes':codes,'all_worker_checks_passed':all(x==0 for x in codes.values()),
        'posterior_draws_per_request':DRAW_COUNT,'scientifically_qualified':False,
        'parallelism':'four independent complete-model qualification workers; no distributed training or tensor parallelism',
        'remaining':'I07 full scientific/reference and LR pilots remain unrun'}
    write_json(root/'allocation.json',summary)
    if not summary['all_worker_checks_passed']:raise RuntimeError('qualification has failed lanes; retain evidence')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='operation',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',required=True);prep.add_argument('--repo',required=True);prep.add_argument('--base-checkpoint',required=True);prep.add_argument('--source-commit',required=True)
    for command in ('run','worker'):
        q=sub.add_parser(command);q.add_argument('--inputs',required=True);q.add_argument('--input-sha256',required=True);q.add_argument('--output',required=True);q.add_argument('--repo',required=True)
        if command=='worker':q.add_argument('--lane',choices=LANES,required=True)
    a=p.parse_args();args=vars(a);op=args.pop('operation')
    result=prepare_inputs(**args) if op=='prepare' else (launch_workers(**args) if op=='run' else worker(**args))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
