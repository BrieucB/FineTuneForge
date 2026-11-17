"""Training planning and bounded CPU orchestration; production stays I07-gated.

Pilots and main runs have disjoint matched data streams. Main runs restart the
original pretrained model with fresh adapters; pilot data/compute remain costs.
No posterior draws, GPU execution, scheduler submission or upload occurs here.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json,math,os,random,time,uuid
import numpy as np
from .episodes import EpisodeStream,QuickValidationStream,BankStore,materialize_episode,digest,_sha
from .seeds import derive_seed

METHODS=('lora_r4','lora_r16','full_finetune')
LEARNING_RATES={'lora_r4':(1e-4,3e-4),'lora_r16':(1e-4,3e-4),'full_finetune':(1e-5,3e-5)}
MAIN_ALLOWANCES=(1000,3000,10000)

class CampaignQualificationError(RuntimeError):
    '''Pure metadata/identity validation error; importing it requires no Torch.'''

def _write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def source_identity():
    names=('campaign.py','episodes.py','model.py','training.py','updates.py','science.py','banks.py','ordering.py','spec.py','seeds.py','cuda_training.py','training_execution.py')
    return {name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

def quick_milestones(updates):
    if type(updates) is not int or updates<1:raise ValueError('positive update allowance required')
    return tuple(sorted({math.ceil(updates*k/10) for k in range(1,11)}))

@dataclass(frozen=True)
class CampaignSpec:
    method:str
    replica:int
    phase:str
    learning_rate:float
    updates:int
    protocol_sha256:str
    master_seed:int=20261117
    effective_batch:int=16
    diagnostic:bool=False
    def __post_init__(self):
        if self.method not in METHODS or type(self.replica) is not int or self.replica not in (0,1) or self.phase not in ('pilot','main') or not _sha(self.protocol_sha256):raise ValueError('invalid campaign identity')
        if type(self.master_seed) is not int or self.master_seed<0 or type(self.diagnostic) is not bool:raise ValueError('integer master seed and explicit boolean diagnostic flag required')
        if self.learning_rate not in LEARNING_RATES[self.method]:raise ValueError('learning rate outside agreed pilot grid')
        if type(self.updates) is not int or self.updates<1 or type(self.effective_batch) is not int or not 1<=self.effective_batch<=16:raise ValueError('invalid update/batch allowance')
        if self.diagnostic:
            if self.updates>10:raise ValueError('bounded CPU diagnostics permit at most10 updates')
        elif self.effective_batch!=16 or (self.phase=='pilot' and self.updates!=200) or (self.phase=='main' and self.updates not in MAIN_ALLOWANCES):raise ValueError('scientific campaign allowance changed')
    @property
    def rank(self):return None if self.method=='full_finetune' else int(self.method.rsplit('r',1)[1])
    @property
    def training_method(self):return 'full_finetune' if self.rank is None else 'lora'
    @property
    def initialization_seed(self):return derive_seed(self.master_seed,'model-init/'+self.phase,replica=self.replica,rank=0 if self.rank is None else self.rank)
    def to_dict(self):return asdict(self)

def plan_six_campaigns(*,protocol_sha256,learning_rates=None,main_updates=None,master_seed=20261117):
    if not _sha(protocol_sha256):raise ValueError('protocol SHA256 required')
    if type(master_seed) is not int or master_seed<0:raise ValueError('nonnegative integer master seed required')
    if main_updates is not None and main_updates not in MAIN_ALLOWANCES:raise ValueError('main allowance requires measured selection')
    rows=[]
    for method in METHODS:
        if learning_rates is not None and learning_rates.get(method) not in LEARNING_RATES[method]:raise ValueError('invalid selected learning rate')
        for replica in (0,1):
            rows.append({'method':method,'replica':replica,'phase':'main','learning_rate':None if learning_rates is None else learning_rates[method],'updates':main_updates,'effective_batch':16,'protocol_sha256':protocol_sha256,'master_seed':master_seed,'restart':'original_pretrained_weights_and_fresh_adapter','main_stream':'disjoint_from_pilot','pilot_updates_excluded_from_main':True,'pilot_costs_charged_separately':True,'status':'planned_unqualified','gpu_execution_enabled':False})
    return rows

def select_method_learning_rates(records):
    """Select one LR/method from mean endpoint score across both pilot seeds."""
    selected={};evidence={}
    for method in METHODS:
        means={}
        for rate in LEARNING_RATES[method]:
            rows=[r for r in records if r['method']==method and r['learning_rate']==rate]
            if len(rows)!=2 or any(type(r['replica']) is not int for r in rows) or {r['replica'] for r in rows}!={0,1} or any(type(r.get('update')) is not int or r['update']!=200 or not math.isfinite(r['score']) for r in rows):raise ValueError('two finite200-update endpoint scores per LR required')
            means[rate]=math.fsum(r['score'] for r in rows)/2
        selected[method]=min(means,key=lambda r:(means[r],r));evidence[method]=means
    if len(records)!=12:raise ValueError('exactly12 pilot endpoint records required')
    return {'learning_rates':selected,'mean_scores':evidence,'score_rule':'200-update endpoint averaged over both replicas','tie_rule':'lower learning rate','scientifically_qualified':False}

def select_main_allowance(*,update_gpu_hours,quick_check_gpu_hours,pilot_gpu_hours,other_gpu_hours=0.,measurement_sha256):
    """Forecast main training separately from qualification pilots.

    Pilot costs belong to the500-hour qualification envelope. Main updates,
    monitoring and main overhead receive25% margin against3000 hours. The full
    qualification envelope remains reserved in the immutable10000-hour project
    plan, so pilot costs are neither lost nor charged twice to main training.
    """
    keys={f'{m}/seed{r}' for m in METHODS for r in (0,1)}
    if set(update_gpu_hours)!=keys or set(quick_check_gpu_hours)!=keys or not _sha(measurement_sha256):raise ValueError('measured costs for all6 campaigns and retained measurement hash required')
    values=[*update_gpu_hours.values(),*quick_check_gpu_hours.values(),pilot_gpu_hours,other_gpu_hours]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values) or not all(v>0 for v in update_gpu_hours.values()):raise ValueError('finite positive measured update costs required')
    envelopes={'qualification':500,'training':3000,'evaluation':4500,'engineering':1000,'reserve':1000}
    qualification_fits=pilot_gpu_hours<=envelopes['qualification']
    rows=[]
    for allowance in MAIN_ALLOWANCES:
        training=allowance*math.fsum(update_gpu_hours.values());monitoring=10*math.fsum(quick_check_gpu_hours.values())
        main_total=1.25*(training+monitoring+other_gpu_hours)
        project_total=main_total+sum(v for k,v in envelopes.items() if k!='training')
        rows.append({'updates':allowance,'main_training_gpu_hours':training,'quick_validation_gpu_hours':monitoring,'pilot_gpu_hours':pilot_gpu_hours,'pilot_cost_phase':'qualification','pilot_included_in_main_forecast':False,'other_gpu_hours':other_gpu_hours,'other_cost_phase':'main_training','with25percent_margin_gpu_hours':main_total,'fits_training_envelope':main_total<=envelopes['training'],'project_total_with_reserved_other_phases_gpu_hours':project_total,'fits_project_cap':project_total<=10000})
    accepted=[r['updates'] for r in rows if r['fits_training_envelope'] and r['fits_project_cap'] and qualification_fits]
    return {'selected_updates':max(accepted) if accepted else None,'forecasts':rows,'training_envelope_gpu_hours':3000,'qualification':{'pilot_gpu_hours':pilot_gpu_hours,'envelope_gpu_hours':500,'fits_envelope':qualification_fits,'remaining_for_other_qualification_gpu_hours':max(0.,500-pilot_gpu_hours)},'phase_envelopes_gpu_hours':envelopes,'project_cap_gpu_hours':10000,'measurement_sha256':measurement_sha256,'status':'forecast_only_not_execution_authorization'}

def shortlist_checkpoints(history):
    valid=[r for r in history if r.get('quick_score') is not None and r.get('checkpoint') and math.isfinite(r['quick_score'])]
    return sorted(valid,key=lambda r:(r['quick_score'],r['update']))[:2]

class StockTrainingBackend:
    """Actual cached TabPFN/PEFT backend; constructor does no training."""
    backend_kind='stock_tabpfn_cpu'
    def __init__(self,spec:CampaignSpec,checkpoint):
        import torch
        from .model import TabPFNBackbone,CANONICAL_FULL_SUPPORT,BASE_CHECKPOINT_SHA256,attach_lora,weights_fingerprint
        from .training import TrainingConfig,configure_full_finetune,make_optimizer
        self.spec=spec;self.checkpoint=Path(checkpoint)
        self.training_config=TrainingConfig(method=spec.training_method,rank=4 if spec.rank is None else spec.rank,learning_rate=spec.learning_rate,max_updates=spec.updates,effective_batch=spec.effective_batch,seed=spec.initialization_seed,density_law=CANONICAL_FULL_SUPPORT)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.initialization_seed)
            self.model=TabPFNBackbone.from_checkpoint(self.checkpoint)
            if spec.rank is None:configure_full_finetune(self.model.backbone)
            else:self.model.backbone=attach_lora(self.model.backbone,rank=spec.rank)
        self.optimizer=make_optimizer(self.model,self.training_config)
        self.identity={'backend_kind':self.backend_kind,'base_checkpoint_sha256':BASE_CHECKPOINT_SHA256,'initialization_seed':spec.initialization_seed,'initial_weights_sha256':weights_fingerprint(self.model),'versions':{n:version(n) for n in ('torch','tabpfn','peft','transformers','numpy','scipy')}}
    @classmethod
    def from_checkpoint_factory(cls,spec,checkpoint):return cls(spec,checkpoint)
    def regressor_factory(self):
        import torch
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(model_path=self.checkpoint,device='cpu',n_estimators=8,random_state=0,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',softmax_temperature=.9,average_before_softmax=False)
    def prepare(self,item):
        from .model import CANONICAL_FULL_SUPPORT
        from .training import prepare_teacher_forced_episode
        return prepare_teacher_forced_episode(item.context,item.query,regressor_factory=self.regressor_factory,law=CANONICAL_FULL_SUPPORT)
    def update(self,items,state):
        from .model import CANONICAL_FULL_SUPPORT
        from .updates import stagewise_cpu_update
        started=time.perf_counter();prepared=[self.prepare(x) for x in items];preparation=time.perf_counter()-started
        report=stagewise_cpu_update(self.model,self.optimizer,prepared,state,law=CANONICAL_FULL_SUPPORT,gradient_clip=1.,allow_unqualified_cpu_diagnostic=True)
        return dict(report,preprocessing_seconds=preparation,update_seconds=time.perf_counter()-started-preparation)
    def quick_validate(self,items):
        import torch
        from .model import CANONICAL_FULL_SUPPORT
        from .training import score_teacher_forced_episode,balanced_problem_loss
        rows=[]
        with torch.no_grad():
            for item in items:
                episode=self.prepare(item);score=score_teacher_forced_episode(self.model,episode,law=CANONICAL_FULL_SUPPORT)
                loss=balanced_problem_loss(list(score.parameter_losses.unbind(0)),[score.roles])
                rows.append({'request_id':item.receipt['request']['request_id'],'score':float(loss),'context_size':item.receipt['request']['context_size'],'posterior_draws':0})
        if not rows:raise ValueError('empty quick validation')
        return {'score':math.fsum(r['score'] for r in rows)/len(rows),'score_count':len(rows),'records':rows,'posterior_draws':0}
    def export(self,directory):
        from .model import CANONICAL_FULL_SUPPORT,BASE_CHECKPOINT_SHA256,weights_fingerprint
        from .training import save_adapter,export_full_weights
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        if self.spec.rank is None:
            export_full_weights(directory/'full_model.safetensors',self.model);format_='full_tabpfn_safetensors';path='full_model.safetensors'
        else:
            save_adapter(directory/'adapter',self.model.backbone,checkpoint_sha256=BASE_CHECKPOINT_SHA256);format_='peft_adapter';path='adapter'
        return {'backend_kind':self.backend_kind,'format':format_,'path':path,'campaign_method':self.spec.method,'training_method':self.spec.training_method,'rank':self.spec.rank,'base_checkpoint_sha256':BASE_CHECKPOINT_SHA256,'weights_sha256':weights_fingerprint(self.model),'sampling_law':CANONICAL_FULL_SUPPORT,'source_policy':'cancelled_zero_width','scientifically_qualified':False}

def campaign_identity(spec,stream,validation,backend):
    if spec.protocol_sha256!=stream.protocol_sha256 or spec.replica!=stream.replica or spec.phase!=stream.phase:raise ValueError('campaign/stream mismatch')
    if validation.protocol_sha256!=spec.protocol_sha256:raise ValueError('validation protocol mismatch')
    if spec.master_seed!=stream.master_seed or spec.master_seed!=validation.master_seed:raise ValueError('campaign/training/validation master-seed mismatch')
    if getattr(backend,'spec',spec)!=spec:raise ValueError('backend/campaign specification mismatch')
    return {'spec':spec.to_dict(),'protocol_sha256':spec.protocol_sha256,'stream_id':stream.identity,'quick_validation_id':validation.identity,'source_sha256':digest(source_identity()),'source_files':source_identity(),'backend':backend.identity,'training_config':asdict(backend.training_config)}

def save_campaign_checkpoint(directory,backend,state,*,identity,history):
    from .model import weights_fingerprint
    from .training import save_training_checkpoint
    directory=Path(directory)
    if directory.exists():raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True);temporary=directory.with_name(directory.name+'.tmp-'+uuid.uuid4().hex);temporary.mkdir()
    # Export precedes RNG snapshot so any exporter RNG use is included exactly.
    deployment=backend.export(temporary/'deployment')
    save_training_checkpoint(temporary/'training',backend.model,backend.optimizer,state,config=backend.training_config,episode_rng=getattr(backend,'episode_rng',None))
    metadata={'schema':'finetuneforge.campaign_checkpoint.v1','identity':identity,'state':asdict(state),'history':history,'deployment':deployment,'weights_sha256':weights_fingerprint(backend.model),'scientifically_qualified':False}
    _write(temporary/'campaign.json',metadata)
    inventory={str(p.relative_to(temporary)):sha256(p.read_bytes()).hexdigest() for p in temporary.rglob('*') if p.is_file()}
    _write(temporary/'manifest.json',{'schema':'finetuneforge.campaign_checkpoint.files.v1','files':inventory})
    os.replace(temporary,directory)
    return sha256((directory/'manifest.json').read_bytes()).hexdigest()

def read_campaign_checkpoint(directory,*,expected_identity=None):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!='finetuneforge.campaign_checkpoint.files.v1':raise CampaignQualificationError('checkpoint manifest schema mismatch')
    inventory={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    if set(manifest['files'])!=inventory:raise CampaignQualificationError('checkpoint inventory mismatch')
    for name,checksum in manifest['files'].items():
        path=directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or sha256(path.read_bytes()).hexdigest()!=checksum:raise CampaignQualificationError('checkpoint checksum/path mismatch')
    metadata=json.loads((directory/'campaign.json').read_text())
    if metadata.get('schema')!='finetuneforge.campaign_checkpoint.v1' or metadata.get('scientifically_qualified') is not False:raise CampaignQualificationError('checkpoint qualification/schema mismatch')
    if expected_identity is not None and metadata['identity']!=expected_identity:raise CampaignQualificationError('source/protocol/stream/model identity mismatch')
    return metadata

def restore_campaign_checkpoint(directory,backend,*,expected_identity):
    from .model import ModelQualificationError,weights_fingerprint
    from .training import load_training_checkpoint
    metadata=read_campaign_checkpoint(directory,expected_identity=expected_identity)
    state=load_training_checkpoint(Path(directory)/'training',backend.model,backend.optimizer,config=backend.training_config,expected_stream_id=expected_identity['stream_id'],episode_rng=getattr(backend,'episode_rng',None))
    if asdict(state)!=metadata['state'] or weights_fingerprint(backend.model)!=metadata['weights_sha256']:raise ModelQualificationError('checkpoint state/weights mismatch')
    return state,metadata

def run_cpu_diagnostic(spec,*,stream,validation,store,backend,output,allow_unqualified_cpu_diagnostic=False,resume_from=None,stop_after_update=None):
    """Complete bounded CPU loop with real/injected backend and retained attempts."""
    import torch
    from .model import ModelQualificationError,weights_fingerprint
    from .training import TrainingState
    if not allow_unqualified_cpu_diagnostic or not spec.diagnostic:raise ModelQualificationError('explicit bounded CPU-diagnostic campaign required; production is I07-gated')
    if any(p.device.type!='cpu' for p in backend.model.parameters()):raise ModelQualificationError('GPU execution disabled')
    identity=campaign_identity(spec,stream,validation,backend)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runfile=output/'run.json'
    if runfile.exists() and json.loads(runfile.read_text())['identity']!=identity:raise ModelQualificationError('run identity mismatch')
    if not runfile.exists():_write(runfile,{'identity':identity,'scientifically_qualified':False,'gpu_jobs_submitted':0})
    if resume_from is None:
        if (output/'attempts').exists() and any((output/'attempts').iterdir()):raise ModelQualificationError('existing attempts require explicit checkpoint resume')
        if weights_fingerprint(backend.model)!=backend.identity.get('initial_weights_sha256'):raise ModelQualificationError('fresh original-weights model/adapter initialization required')
        torch.manual_seed(spec.initialization_seed);np.random.seed(spec.initialization_seed%(2**32));random.seed(spec.initialization_seed)
        state=TrainingState(stream_id=stream.identity);history=[]
        initial=output/'checkpoints'/'update-000000'
        if not initial.exists():
            checkpointed=time.perf_counter();initial_sha=save_campaign_checkpoint(initial,backend,state,identity=identity,history=history)
            _write(runfile,{'identity':identity,'scientifically_qualified':False,'gpu_jobs_submitted':0,'initial_checkpoint':str(initial),'initial_checkpoint_manifest_sha256':initial_sha,'initial_checkpoint_seconds':time.perf_counter()-checkpointed})
        else:read_campaign_checkpoint(initial,expected_identity=identity)
    else:state,previous=restore_campaign_checkpoint(resume_from,backend,expected_identity=identity);history=list(previous['history'])
    target=spec.updates if stop_after_update is None else stop_after_update
    if type(target) is not int or not state.update<target<=spec.updates:raise ValueError('diagnostic stop must advance within allowance')
    milestones=quick_milestones(spec.updates);last_checkpoint=None
    attempts=output/'attempts';attempts.mkdir(exist_ok=True)
    while state.update<target:
        before=asdict(state);event_start=len(store.events);started=time.perf_counter()
        attempt=attempts/(f'update-{state.update+1:06d}-'+uuid.uuid4().hex);attempt.mkdir()
        record={'status':'running','before':before,'requests':[r.to_dict() for r in stream.batch(state.data_position,spec.effective_batch)],'timings':{},'posterior_draws':0,'gpu_jobs_submitted':0,'scientifically_qualified':False}
        _write(attempt/'attempt.json',record)
        try:
            data_started=time.perf_counter();items=[materialize_episode(stream,stream.request(before['data_position']+i),store) for i in range(spec.effective_batch)]
            record['episodes']=[x.receipt for x in items];record['timings']['training_data_seconds']=time.perf_counter()-data_started
            update_started=time.perf_counter();report=backend.update(items,state);record['timings']['training_update_seconds']=time.perf_counter()-update_started;record['update']=report
            if state.update!=before['update']+1 or state.data_position!=before['data_position']+spec.effective_batch:raise ModelQualificationError('backend counter/cursor mismatch')
            row={'update':state.update,'data_position':state.data_position,'training_loss':report['loss']}
            if state.update in milestones:
                checked=time.perf_counter();quick=backend.quick_validate(materialize_episode(validation,r,store) for r in validation)
                if quick.get('posterior_draws')!=0 or quick.get('score_count')!=len(validation) or not math.isfinite(quick['score']):raise ModelQualificationError('incomplete/nonfinite quick validation; no posterior draws allowed')
                record['timings']['quick_validation_seconds']=time.perf_counter()-checked;record['quick_validation']=quick;row['quick_score']=quick['score']
            save=state.update in milestones or state.update==target
            if save:row['checkpoint']=f'checkpoints/update-{state.update:06d}'
            history.append(row)
            if save:
                checkpointed=time.perf_counter();last_checkpoint=output/row['checkpoint'];record['checkpoint_manifest_sha256']=save_campaign_checkpoint(last_checkpoint,backend,state,identity=identity,history=history)
                record['timings']['checkpoint_seconds']=time.perf_counter()-checkpointed
            record['status']='complete';record['after']=asdict(state)
        except BaseException as exc:
            backend.optimizer.zero_grad(set_to_none=True);record.update(status='failed',exception_type=type(exc).__name__,exception=str(exc),after=asdict(state))
            raise
        finally:
            record['data_events']=store.events[event_start:];record['timings']['allocated_cpu_wall_seconds']=time.perf_counter()-started
            _write(attempt/'attempt.json',record)
    result={'status':'cpu_diagnostic_complete' if state.update==spec.updates else 'cpu_diagnostic_checkpointed','identity':identity,'state':asdict(state),'history':history,'shortlist':shortlist_checkpoints(history),'costs':summarize_attempt_costs(output),'last_checkpoint':str(last_checkpoint),'posterior_draws':0,'gpu_jobs_submitted':0,'scientifically_qualified':False}
    _write(output/'result.json',result)
    return result

def run_production_campaign(*args,**kwargs):
    from .training import TrainingConfig
    TrainingConfig().require_production_qualification()

def plan_pilot_campaigns(*,protocol_sha256,master_seed=20261117):
    return [CampaignSpec(method,replica,'pilot',rate,200,protocol_sha256,master_seed) for method in METHODS for rate in LEARNING_RATES[method] for replica in (0,1)]

def summarize_attempt_costs(output):
    """Include failed/retried attempts; deduplicate accessible raw datasets only."""
    datasets={};incurred=0;unknown=0;wall=0.;attempts=0;failed=0
    runfile=Path(output)/'run.json'
    initial_seconds=json.loads(runfile.read_text()).get('initial_checkpoint_seconds',0.) if runfile.exists() else 0.
    for path in sorted((Path(output)/'attempts').glob('*/attempt.json')):
        record=json.loads(path.read_text());attempts+=1;failed+=record['status']!='complete'
        wall+=record.get('timings',{}).get('allocated_cpu_wall_seconds',0.)
        if record['status']=='running':unknown+=1
        for event in record.get('data_events',[]):
            if event['incurred_forward_evaluations'] is None:unknown+=1
            else:incurred+=event['incurred_forward_evaluations']
            if event['status']=='complete':
                previous=datasets.setdefault(event['dataset_id'],{'worlds':event['worlds'],'forward_evaluations':event['from_scratch_forward_evaluations'],'bank_manifest_sha256':event['bank_manifest_sha256']})
                if previous['forward_evaluations']!=event['from_scratch_forward_evaluations']:raise CampaignQualificationError('dataset cost identity mismatch')
    return {'attempts':attempts,'failed_or_incomplete_attempts':failed,'unique_raw_datasets':len(datasets),'from_scratch_accessible_worlds':sum(d['worlds'] for d in datasets.values()),'from_scratch_accessible_forward_evaluations':sum(d['forward_evaluations'] for d in datasets.values()),'actual_incurred_forward_evaluations_known':incurred,'unknown_or_incomplete_cost_records':unknown,'allocated_cpu_wall_seconds':wall+initial_seconds,'initial_checkpoint_seconds':initial_seconds,'datasets':datasets,'gpu_hours':0.,'note':'unknown failed-provider work is not zero; cached shared data retains from-scratch cost'}
