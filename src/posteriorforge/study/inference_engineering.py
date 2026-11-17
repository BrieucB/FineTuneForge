"""Source-bound attention/adapter inference qualification, with training disabled."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .cuda_qualification import file_hash, write_json, source_files as original_sources

COUNT=50000
REPS=3
BUDGET=536870912


def source_files(repo):
    value=original_sources(repo)
    for name in ('scripts/engineer-study-inference.py','slurm/run-study-inference.sh'):
        value[name]=file_hash(Path(repo)/name)
    return value


def prepare(output,repo,source_commit,previous):
    from .banks import load_bank,prepare_bank,save_bank
    old=Path(previous);v=json.loads((old/'inputs-01/manifest.json').read_text())
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    def descriptor(d):
        return dict(d,directory=str(old/'inputs-01'/d['directory']))
    small=load_bank(old/'inputs-01/posterior-bank',expected_sha256=v['posterior_bank']['manifest_sha256'])
    large=prepare_bank(small.spec,4096,20261117+1000)
    saved=save_bank(large,root/'control-4096')
    pq=prepare_bank(small.spec,1,20269117);pqm=save_bank(pq,root/'parity-query')
    bank4k={'directory':str(root/'control-4096'),'manifest_sha256':saved['manifest_sha256'],'worlds':4096,'dimension':4,'observations':10}
    models={}
    for method in ('lora_r4','lora_r16','full_finetune'):
        cp=old/'allocation-02'/method/'after-two-updates'
        data=json.loads((cp/'campaign.json').read_text())
        models[method]={'campaign_checkpoint':str(cp),'checkpoint_manifest_sha256':file_hash(cp/'manifest.json'),'model_sha256':data['weights_sha256']}
    data={'schema':'finetuneforge.inference_engineering.v1','source_commit':source_commit,'source_files':source_files(repo),
        'previous_evidence':str(old),'base_checkpoint':v['base_checkpoint'],'base_checkpoint_sha256':v['base_checkpoint_sha256'],
        'frozen_model_sha256':v['frozen_model_sha256'],'models':models,'protocol_sha256':v['protocol_sha256'],
        'small_bank':descriptor(v['posterior_bank']),'control_4096':bank4k,
        'parity_query':{'directory':str(root/'parity-query'),'manifest_sha256':pqm['manifest_sha256']},
        'large_probe':{k:descriptor(v['probes'][2][k]) for k in ('context','query')},
        'fitting_probe':{k:descriptor(v['probes'][1][k]) for k in ('context','query')},
        'observation':str(old/'inputs-01/observation.npy'),'observation_sha256':file_hash(old/'inputs-01/observation.npy'),
        'posterior_draws':COUNT,'timing_repetitions':REPS,'microbatch_size':1024,'attention_score_budget_bytes':BUDGET,
        'new_analytical_forward_evaluations':large.manifest['cost']['forward_evaluations']+pq.manifest['cost']['forward_evaluations'],
        'training_pilots_enabled':False,'precision':'FP32 neural / FP64 density and decoding; math SDPA only',
        'scope':'exact failed75D conditional plus50k small controls; full75D posterior needs measured forecast and scope decision'}
    write_json(root/'manifest.json',data)
    return {'manifest':str(root/'manifest.json'),'sha256':file_hash(root/'manifest.json'),'new_forward_evaluations':data['new_analytical_forward_evaluations']}


def verify(inputs,input_sha256,repo):
    p=Path(inputs)
    if file_hash(p)!=input_sha256:raise ValueError('input manifest checksum mismatch')
    v=json.loads(p.read_text())
    if v.get('schema')!='finetuneforge.inference_engineering.v1' or v.get('posterior_draws')!=COUNT or v.get('timing_repetitions')!=REPS or v.get('training_pilots_enabled') is not False:
        raise ValueError('engineering scope/count contract changed')
    if v['source_files']!=source_files(repo):raise ValueError('source/lock bytes changed')
    for field,expected in [('base_checkpoint',v['base_checkpoint_sha256']),('observation',v['observation_sha256'])]:
        if file_hash(v[field])!=expected:raise ValueError(field+' identity changed')
    return v


def configure():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='eu-26-17':raise RuntimeError('accounted Slurm execution required')
    import torch
    from .cuda_training import configure_cuda_determinism
    configure_cuda_determinism()
    if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:raise RuntimeError('one independent visible GPU per worker required')
    if 'A100' not in torch.cuda.get_device_properties(0).name:raise RuntimeError('declared A100 hardware required')


def load(v,method,mode='unmerged'):
    from .inference_deployment import load_frozen_deployment
    args={'expected_model_sha256':v['frozen_model_sha256']} if method=='frozen' else {
        'expected_model_sha256':v['models'][method]['model_sha256'],'campaign_checkpoint':v['models'][method]['campaign_checkpoint'],
        'expected_checkpoint_manifest_sha256':v['models'][method]['checkpoint_manifest_sha256']}
    return load_frozen_deployment(v['base_checkpoint'],**args,mode=mode,device='cuda:0')


def backend(v,network,deployment,handle=None):
    from .sampling_backend import TabPFNConditionalBackend
    stable={k:deployment[k] for k in ('origin_model_sha256','model_sha256','checkpoint_manifest_sha256','deployment_mode','merge_policy','source_files','feature_embedding_policy')}
    class EngineeringBackend(TabPFNConditionalBackend):
        def identity_metadata(self):
            return dict(super().identity_metadata(),engineering_source_sha256=file_hash(__file__),deployment=stable,
                        attention={'policy':'stock'} if handle is None else handle.metadata)
        def verify_identity(self):
            super().verify_identity()
            if handle is not None and (handle.closed or any(getattr(m._compute,'__func__',None) is not replacement for m,original,replacement in handle.modules)):
                raise ValueError('attention installation changed during request')
    return EngineeringBackend(v['base_checkpoint'],network,expected_model_sha256=deployment['model_sha256'],device='cuda:0')


def bank(d):
    from .banks import load_bank
    return load_bank(d['directory'],expected_sha256=d['manifest_sha256'])


def one_prepared(b,context,query,stage,nq):
    import numpy as np
    from .science import encode_prior
    z=encode_prior(context.spec,context.theta);q=encode_prior(query.spec,query.theta)
    prepared=b.prepare(np.concatenate((context.observations,z[:,:stage]),1),z[:,stage])
    features=np.concatenate((query.observations,q[:,:stage]),1)
    return prepared,np.repeat(features,nq,axis=0)


def compare_prepared(net_a,net_b,prep,features,*,attention_b=None):
    import torch
    from .model import prepare_fitted_regressor_members,CANONICAL_FULL_SUPPORT,BarDensity
    ensemble=prepare_fitted_regressor_members(prep,features,device='cuda:0',source_policy='cancelled_zero_width')
    with torch.no_grad():
        la=net_a(ensemble.members)
        lb=net_b(ensemble.members)
        da=ensemble.density(net_a,law=CANONICAL_FULL_SUPPORT)
        db=ensemble.density(net_b,law=CANONICAL_FULL_SUPPORT)
        ps=torch.linspace(.00001,.99999,len(features),dtype=torch.float64,device='cuda:0')
        qa=da.icdf(ps);qb=db.icdf(ps)
        grid=torch.linspace(-8.,8.,65,dtype=torch.float64,device='cuda:0')
        # Densities have a query dimension; evaluate a scalar grid location for every query.
        cdf=max(float((da.cdf(t.expand(len(features)))-db.cdf(t.expand(len(features)))).abs().max()) for t in grid)
        aa=BarDensity(da.logits[:1],da.borders,law=CANONICAL_FULL_SUPPORT)
        bb=BarDensity(db.logits[:1],db.borders,law=CANONICAL_FULL_SUPPORT)
        fullgrid=torch.unique(torch.cat((aa.borders,bb.borders,torch.linspace(-8.,8.,257,dtype=torch.float64,device='cuda:0'))))
        fullcdf=max(float((aa.cdf(g)-bb.cdf(g)).abs().max()) for g in fullgrid.split(128))
        tails=torch.tensor([1e-9,1e-5,.01,.5,.99,1-1e-5,1-1e-9],dtype=torch.float64,device='cuda:0')
        ta=aa.icdf(tails);tb=bb.icdf(tails)
        tail_pass=bool(torch.allclose(ta,tb,rtol=1e-3,atol=1e-3))
        cdf=max(cdf,fullcdf)
        errors=[float((a-b).abs().max()) for a,b in zip(la,lb)]
        passed=len(la)==len(lb)==8 and all(torch.allclose(a,b,rtol=1e-3,atol=1e-4) for a,b in zip(la,lb)) and cdf<=1e-4 and bool(torch.allclose(qa,qb,rtol=1e-3,atol=1e-3)) and tail_pass
    return {'passed':bool(passed),'raw_member_max_abs_logits':errors,'cdf_max_abs':cdf,'quantile_max_abs':max(float((qa-qb).abs().max()),float((ta-tb).abs().max())),'cdf_full_grid_points':len(fullgrid),'tail_levels':tails.tolist(),'queries':len(features),
            'limits':{'logits_rtol':1e-3,'logits_atol':1e-4,'cdf_atol':1e-4,'quantile_rtol':1e-3,'quantile_atol':1e-3}}


def profile(v,b,context,out):
    import torch,numpy as np
    from .science import encode_prior
    from .model import prepare_fitted_regressor_members,canonical_full_support_mixture
    z=encode_prior(context.spec,context.theta);prep=b.prepare(context.observations,z[:,0]);features=np.zeros((32,context.spec.observation_dimension))
    with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as trace:
        with torch.profiler.record_function('query_preprocessing_and_h2d'):
            ensemble=prepare_fitted_regressor_members(prep,features,device='cuda:0',source_policy=b.source_policy)
        with torch.profiler.record_function('transformer_ensemble'):
            logits=ensemble.adjusted_logits(b.network)
        with torch.profiler.record_function('canonical_cdf_mixture_and_icdf'):
            density=canonical_full_support_mixture(logits,[m.source_borders for m in ensemble.members],ensemble.canonical_borders,sampling_borders=ensemble.sampling_borders)
            values=density.icdf(torch.linspace(.01,.99,32,dtype=torch.float64,device='cuda:0'))
        with torch.profiler.record_function('d2h'):values.cpu();torch.cuda.synchronize()
    trace.export_chrome_trace(str(out/'conditional-trace.json'))
    averages=trace.key_averages();(out/'profile.txt').write_text(averages.table(sort_by='self_cuda_time_total',row_limit=40))
    counts={e.key:{'count':e.count,'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total} for e in averages if e.key in ('aten::mm','aten::bmm','aten::addmm','cudaLaunchKernel','cudaStreamSynchronize')}
    write_json(out/'profile-counts.json',counts)
    return counts


def posterior(v,method,mode,tiled,context_desc,out,*,repetitions):
    import torch,numpy as np
    from .cuda_attention import install_tiled_attention
    from .sampling import sample_posterior
    from .cuda_qualification import _analytic_module,plot_neural_posterior
    net,meta=load(v,method,mode);handle=install_tiled_attention(net,max_score_bytes=v['attention_score_budget_bytes']) if tiled else None
    b=backend(v,net,meta,handle);context=bank(context_desc);y=np.load(v['observation'],allow_pickle=False)
    write_json(out/'deployment.json',meta);profile(v,b,context,out)
    rows=[]
    for rep in range(repetitions):
        path=out/f'repetition-{rep}';path.mkdir()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        values,record=sample_posterior(context,y,b,seed=20265117,microbatch_size=v['microbatch_size'],
          identities={'model_sha256':meta['model_sha256'],'bank_manifest_sha256':context_desc['manifest_sha256'],'protocol_sha256':v['protocol_sha256']},
          resume_path=path/'recovery.npz',cuda_qualification=True)
        np.savez(path/'samples.npz',samples=values)
        row={'posterior_draws':len(values),'sampling':record,'sample_sha256':file_hash(path/'samples.npz'),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'attention_stats':None if handle is None else handle.statistics,'scientifically_qualified':False}
        if rep==0:
            repo=Path(__file__).resolve().parents[3];control=_analytic_module(repo);metrics=control.analyze_samples(values)
            row['analytical_diagnostics']=metrics
            p=control.analytic_parameters();prior=np.random.default_rng(20265118).lognormal(p['sigma_prior_log_mean'],p['sigma_log_sd'],50000)
            plot_neural_posterior(values,prior,path/'posterior.png',metrics,method=f'{method}/{mode}/tiled={tiled}/C={len(context.theta)}')
        write_json(path/'metrics.json',row);rows.append(row)
    result={'method':method,'mode':mode,'tiled':tiled,'context':context_desc,'deployment':meta,'runs':rows,
            'seconds':[r['sampling']['wall_seconds'] for r in rows],'median_seconds':float(np.median([r['sampling']['wall_seconds'] for r in rows])),
            'range_seconds':[min(r['sampling']['wall_seconds'] for r in rows),max(r['sampling']['wall_seconds'] for r in rows)],'scientifically_qualified':False}
    write_json(out/'posterior-result.json',result)
    if handle is not None:handle.close()
    return result


def probes(v,out):
    import torch,numpy as np
    from .cuda_attention import install_tiled_attention
    from .model import prepare_fitted_regressor_members
    net,meta=load(v,'full_finetune');b=backend(v,net,meta)
    context=bank(v['large_probe']['context']);query=bank(v['large_probe']['query'])
    rows=[]
    for index,nq in [(74,1),(0,1024),(37,1024),(74,1024)]:
        handle=install_tiled_attention(net,max_score_bytes=v['attention_score_budget_bytes']);b=backend(v,net,meta,handle)
        row={'stage':index,'queries':nq,'status':'running'};path=out/f'probe-{index}-{nq}.json';write_json(path,row)
        try:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            prepared,features=one_prepared(b,context,query,index,nq)
            prep_seconds=time.perf_counter()-start;start=time.perf_counter()
            result=b.sample(prepared,features,np.linspace(.01,.99,nq));torch.cuda.synchronize()
            row.update(status='completed',preparation_seconds=prep_seconds,conditional_seconds=time.perf_counter()-start,finite=bool(np.isfinite(result).all()),attention_stats=handle.statistics)
        except BaseException as exc:
            row.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)});raise
        finally:
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());write_json(path,row);rows.append(row);handle.close();torch.cuda.empty_cache()
    # Existing fitting1024 shape allows a genuine stock-vs-tiled comparison at high feature count.
    ca=bank(v['fitting_probe']['context']);qu=bank(v['fitting_probe']['query']);ref,ref_meta=load(v,'full_finetune')
    candidate,candidate_meta=load(v,'full_finetune');handle=install_tiled_attention(candidate,max_score_bytes=BUDGET)
    prep,features=one_prepared(backend(v,ref,ref_meta),ca,qu,74,1)
    comparison=compare_prepared(ref,candidate,prep,features);write_json(out/'fitting-stock-tiled-parity.json',comparison);handle.close()
    if not comparison['passed']:raise RuntimeError('fitting stock/tiled parity failed')
    # A measured forecast, not an execution authorization for the full75D posterior.
    sampled=[r['conditional_seconds'] for r in rows if r['queries']==1024]
    forecast={'estimated_sampler_seconds_range':[75*49*min(sampled),75*49*max(sampled)],'measured_stages':[0,37,74],
              'posterior_draws':50000,'microbatches':49,'stages':75,'assumptions':'representative repeated-conditional costs; excludes extra initial preprocessing/IO; no context KV cache',
              'full_75d_run_enabled':False}
    write_json(out/'large-forecast.json',forecast)
    return {'rows':rows,'fitting_parity':comparison,'forecast':forecast}


def parity(v,out):
    import torch,numpy as np
    from .cuda_attention import install_tiled_attention
    net,meta=load(v,'lora_r4','unmerged');merged,mm=load(v,'lora_r4','merged_lora');ctx=bank(v['small_bank'])
    # Use complete recorded query cases for conditional-only checks; no posterior output count is changed.
    q=bank(v['parity_query'])
    rows=[]
    for stage in range(4):
        prep,features=one_prepared(backend(v,net,meta),ctx,q,stage,32)
        x=compare_prepared(net,merged,prep,features);x['stage']=stage;rows.append(x)
    write_json(out/'merge-parity.json',rows)
    if not all(x['passed'] for x in rows):raise RuntimeError('merged conditional parity failed')
    return rows


def context_variant(lane):
    variants={'c4k_frozen_stock':('frozen','unmerged',False),'c4k_frozen_tiled':('frozen','unmerged',True),
              'c4k_r4_unmerged':('lora_r4','unmerged',False),'c4k_r4_merged_tiled':('lora_r4','merged_lora',True)}
    if lane not in variants:raise ValueError('unknown context variant')
    return variants[lane]


def compare_samples(original,candidate):
    """Execution-preservation flags on complete matched control posteriors."""
    import numpy as np
    from scipy.stats import ks_2samp
    from .reference import predefined_directions
    a=np.asarray(original);b=np.asarray(candidate)
    if a.shape!=(50000,4) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('two complete finite50000x4 control samples required')
    sig_sd=.03*np.exp(-10*np.log1p((.03/.1)**2))
    scales=np.array([1.,.05,np.sqrt(1.0425),sig_sd])
    directions=predefined_directions(4,seed=20261117,scales=scales)
    distances=[float(ks_2samp(a@d,b@d).statistic) for d in directions]
    mean=float(np.max(np.abs(a.mean(0)-b.mean(0))/scales))
    if np.any(a.std(0,ddof=1)<=0):raise ValueError('nondegenerate original sample required')
    sd=float(np.max(np.abs(b.std(0,ddof=1)/a.std(0,ddof=1)-1)))
    return {'axis_and32projection_cdf':distances,'max_pairwise_cdf':max(distances),'max_mean_shift_reference_sd':mean,
            'max_relative_sd_change':sd,'limits':{'cdf':.01,'mean_reference_sd':.05,'relative_sd':.05},
            'investigation_flag':max(distances)>.01 or mean>.05 or sd>.05,'scientifically_qualified':False,
            'scope':'relative execution preservation; absolute scientific accuracy is assessed separately and remains unqualified'}


def compare_files(original,candidate):
    """Authenticate matched request artifacts before a relative distribution check."""
    import numpy as np
    pa,pb=Path(original),Path(candidate)
    a=json.loads((pa/'metrics.json').read_text());b=json.loads((pb/'metrics.json').read_text())
    for p,row in ((pa,a),(pb,b)):
        if file_hash(p/'samples.npz')!=row['sample_sha256'] or row['posterior_draws']!=50000:
            raise ValueError('sample artifact identity/count mismatch')
    x,y=a['sampling']['binding'],b['sampling']['binding']
    for field in ('problem','query','theta','observations','seed','microbatch_size','uniform_law','law','source_policy'):
        if x[field]!=y[field]:raise ValueError('comparison request mismatch: '+field)
    if x['identities']['protocol_sha256']!=y['identities']['protocol_sha256'] or x['source_sha256']!=y['source_sha256']:
        raise ValueError('comparison protocol/core-source mismatch')
    for field in ('origin_model_sha256','checkpoint_manifest_sha256'):
        if x['backend']['deployment'][field]!=y['backend']['deployment'][field]:raise ValueError('comparison origin mismatch')
    if x['identities']['bank_manifest_sha256']!=y['identities']['bank_manifest_sha256']:raise ValueError('comparison bank mismatch')
    with np.load(pa/'samples.npz',allow_pickle=False) as f:av=f['samples']
    with np.load(pb/'samples.npz',allow_pickle=False) as f:bv=f['samples']
    return dict(compare_samples(av,bv),original_manifest_sha256=file_hash(pa/'metrics.json'),candidate_manifest_sha256=file_hash(pb/'metrics.json'),
                samples_bitwise_equal=bool(np.array_equal(av,bv)))


def worker(lane,inputs,input_sha256,output,repo):
    v=verify(inputs,input_sha256,repo);configure();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record={'lane':lane,'status':'running','input_manifest_sha256':input_sha256,'source_commit':v['source_commit'],
            'training_pilots_started':False,'scientifically_qualified':False};write_json(out/'result.json',record)
    started=time.perf_counter()
    try:
        if lane=='probes':record['checks']=probes(v,out)
        elif lane=='merged':
            record['parity']=parity(v,out);record['posterior']=posterior(v,'lora_r4','merged_lora',False,v['small_bank'],out,repetitions=REPS)
        elif lane=='unmerged':record['posterior']=posterior(v,'lora_r4','unmerged',False,v['small_bank'],out,repetitions=REPS)
        elif lane=='frozen':
            record['posterior']=posterior(v,'frozen','unmerged',False,v['small_bank'],out,repetitions=REPS)
            extra=out/'tiled-control';extra.mkdir()
            record['tiled_control']=posterior(v,'frozen','unmerged',True,v['small_bank'],extra,repetitions=1)
        elif lane.startswith('c4k_'):
            method,mode,tiled=context_variant(lane)
            record['posterior']=posterior(v,method,mode,tiled,v['control_4096'],out,repetitions=1)
        else:raise ValueError('unknown lane')
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)});(out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:record['wall_seconds']=time.perf_counter()-started;write_json(out/'result.json',record)
    return record


def run(inputs,input_sha256,output,repo,phase):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    verify(inputs,input_sha256,repo)
    lanes=('probes','unmerged','merged','frozen') if phase=='profile' else ('c4k_frozen_stock','c4k_frozen_tiled','c4k_r4_unmerged','c4k_r4_merged_tiled')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=4 or len(set(devices))!=4:raise RuntimeError('four independent GPUs required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);children=[]
    for lane,dev in zip(lanes,devices):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=dev,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1',TABPFN_ALLOW_CPU_LARGE_DATASET='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        stream=(root/(lane+'.log')).open('w');args=[sys.executable,str(Path(repo)/'scripts/engineer-study-inference.py'),'worker','--lane',lane,'--inputs',str(inputs),'--input-sha256',input_sha256,'--output',str(root/lane),'--repo',str(repo)]
        children.append((lane,subprocess.Popen(args,env=env,stdout=stream,stderr=subprocess.STDOUT),stream))
    codes={}
    try:
        for lane,p,stream in children:codes[lane]=p.wait();stream.close()
    finally:
        for lane,p,stream in children:
            if p.poll() is None:p.terminate()
            stream.close()
    value={'phase':phase,'codes':codes,'source_commit':json.loads(Path(inputs).read_text())['source_commit'],'scientifically_qualified':False}
    write_json(root/'allocation.json',value)
    if any(codes.values()):raise RuntimeError('qualification lane failed; evidence retained')
    return value


def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='operation',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--repo',required=True);q.add_argument('--source-commit',required=True);q.add_argument('--previous',required=True)
    for op in ('run','worker'):
        q=sub.add_parser(op)
        for name in ('inputs','input-sha256','output','repo'):q.add_argument('--'+name,required=True)
        if op=='run':q.add_argument('--phase',choices=('profile','context4096'),required=True)
        else:q.add_argument('--lane',required=True)
    a=vars(p.parse_args(argv));op=a.pop('operation');value=prepare(**a) if op=='prepare' else (run(**a) if op=='run' else worker(**a));print(json.dumps(value,sort_keys=True))
