#!/usr/bin/env python3
"""Exactly 50,000-draw checks after bounded training execution qualification."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

COUNT=50000
SCHEMA='finetuneforge.training_posterior_qualification.v1'
VARIANTS=('stock','cached')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def source(repo,commit):
    repo=Path(repo)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('source revision mismatch')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--','src','scripts','slurm','environments'],text=True).strip():
        raise ValueError('source snapshot must be clean')
    return {str(p.relative_to(repo)):digest(p) for p in sorted((repo/'src/posteriorforge/study').glob('*.py'))} | {
        name:digest(repo/name) for name in ('scripts/qualify-training-posteriors.py','slurm/run-study-training-posteriors.sh','environments/study-training-gpu.lock')}


def checkpoint(directory,protocol):
    path=Path(directory);meta=json.loads((path/'campaign.json').read_text())
    spec=meta['identity']['spec']
    if (meta.get('schema')!='finetuneforge.campaign_checkpoint.v1' or meta['state']['update']!=2
            or meta['state']['data_position']!=32 or spec['method']!='lora_r4'
            or spec['effective_batch']!=16 or spec['protocol_sha256']!=protocol
            or meta['deployment']['rank']!=4 or meta['deployment']['format']!='peft_adapter'):
        raise ValueError('matched two-update rank4 checkpoint required')
    return {'campaign_checkpoint':str(path.resolve()),'checkpoint_manifest_sha256':digest(path/'manifest.json'),
        'campaign_metadata_sha256':digest(path/'campaign.json'),'model_sha256':meta['weights_sha256']}


def prepare(template,stock_checkpoint,cached_checkpoint,output,repo,source_commit):
    if os.environ.get('SLURM_JOB_ID'):raise RuntimeError('prepare inputs outside GPU allocations')
    old=json.loads(Path(template).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if type(old.get('posterior_draws')) is not int or old['posterior_draws']!=COUNT or old['small_bank']['worlds']!=256 or old['small_bank']['dimension']!=4:
        raise ValueError('fixed D4/C256/50,000 control required')
    value={'schema':SCHEMA,'source_commit':source_commit,'source_files':source(repo,source_commit),
        'template_path':str(Path(template).resolve()),'template_sha256':digest(template),'posterior_draws':COUNT,
        'protocol_sha256':old['protocol_sha256'],'base_checkpoint':old['base_checkpoint'],
        'base_checkpoint_sha256':old['base_checkpoint_sha256'],'frozen_model_sha256':old['frozen_model_sha256'],
        'small_bank':old['small_bank'],'observation':old['observation'],'observation_sha256':old['observation_sha256'],
        'microbatch_size':1024,'seed':20265117,'timing_repetitions':1,'mode':'merged_lora','attention_policy':'stock',
        'training_pilots_enabled':False,'new_forward_evaluations':0,'scientifically_qualified':False,
        'checkpoints':{name:checkpoint(path,old['protocol_sha256']) for name,path in zip(VARIANTS,(stock_checkpoint,cached_checkpoint))}}
    write(out/'manifest.json',value)
    return {'path':str(out/'manifest.json'),'sha256':digest(out/'manifest.json')}


def verify(inputs,input_sha256,repo,source_commit):
    if digest(inputs)!=input_sha256:raise ValueError('input manifest checksum mismatch')
    v=json.loads(Path(inputs).read_text())
    if (v.get('schema')!=SCHEMA or type(v.get('posterior_draws')) is not int or v.get('posterior_draws')!=COUNT or v.get('timing_repetitions')!=1
            or v.get('seed')!=20265117 or v.get('microbatch_size')!=1024
            or v.get('mode')!='merged_lora' or v.get('attention_policy')!='stock'
            or v.get('training_pilots_enabled') is not False or v['source_commit']!=source_commit
            or set(v.get('checkpoints',{}))!=set(VARIANTS)):
        raise ValueError('posterior engineering contract changed')
    if source(repo,source_commit)!=v['source_files']:raise ValueError('source bytes changed')
    for path,expected in ((v['template_path'],v['template_sha256']),(v['base_checkpoint'],v['base_checkpoint_sha256']),
                           (v['observation'],v['observation_sha256'])):
        if digest(path)!=expected:raise ValueError('pinned input bytes changed')
    for d in v['checkpoints'].values():
        if checkpoint(d['campaign_checkpoint'],v['protocol_sha256'])!=d:raise ValueError('checkpoint binding changed')
    return v


def worker(variant,*,inputs,input_sha256,output,repo,source_commit):
    from posteriorforge.study.inference_engineering import configure,posterior
    v=verify(inputs,input_sha256,repo,source_commit);configure()
    if variant not in VARIANTS:raise ValueError('unknown variant')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    v['models']={'lora_r4':v['checkpoints'][variant]}
    result={'schema':SCHEMA,'variant':variant,'status':'running','source_commit':source_commit,
        'input_manifest_sha256':input_sha256,'checkpoint':v['checkpoints'][variant],
        'posterior_draws':COUNT,'scientifically_qualified':False,'job_id':os.environ['SLURM_JOB_ID']}
    write(out/'result.json',result)
    try:
        group=posterior(v,'lora_r4','merged_lora',False,v['small_bank'],out,repetitions=1)
        if len(group['runs'])!=1 or group['runs'][0]['posterior_draws']!=COUNT:
            raise RuntimeError('incomplete fixed-size posterior request')
        result['status']='completed';result['posterior_result_sha256']=digest(out/'posterior-result.json')
    except BaseException as error:
        result['status']='failed';result['error']={'type':type(error).__name__,'message':str(error)}
        (out/'traceback.txt').write_text(traceback.format_exc());raise
    finally:write(out/'result.json',result)
    return result


def run(*,inputs,input_sha256,output,repo,source_commit):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='eu-26-17':
        raise RuntimeError('accounted eu-26-17 Slurm allocation required')
    verify(inputs,input_sha256,repo,source_commit)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or not all(devices) or len(set(devices))!=2:raise RuntimeError('two independent GPUs required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);children=[]
    try:
        for variant,device in zip(VARIANTS,devices):
            log=(out/(variant+'.log')).open('w')
            cmd=[sys.executable,str(Path(repo)/'scripts/qualify-training-posteriors.py'),'worker','--variant',variant,
                '--inputs',str(inputs),'--input-sha256',input_sha256,'--output',str(out/variant),
                '--repo',str(repo),'--source-commit',source_commit]
            children.append((variant,subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT),log))
        codes={name:p.wait() for name,p,_ in children}
    finally:
        for _,p,log in children:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    value={'schema':SCHEMA,'worker_returncodes':codes,'source_commit':source_commit,
        'input_manifest_sha256':input_sha256,'posterior_draws_per_request':COUNT,'scientifically_qualified':False}
    write(out/'allocation.json',value)
    if any(codes.values()):raise RuntimeError('posterior qualification worker failed')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);q=p.add_subparsers(dest='operation',required=True)
    a=q.add_parser('prepare')
    for name in ('template','stock-checkpoint','cached-checkpoint','output','repo','source-commit'):a.add_argument('--'+name,required=True)
    for operation in ('run','worker'):
        a=q.add_parser(operation)
        for name in ('inputs','input-sha256','output','repo','source-commit'):a.add_argument('--'+name,required=True)
        if operation=='worker':a.add_argument('--variant',choices=VARIANTS,required=True)
    args=vars(p.parse_args());operation=args.pop('operation')
    result={'prepare':prepare,'run':run,'worker':worker}[operation](**args)
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
