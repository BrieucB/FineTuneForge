"""Matched fresh-world training episodes and fixed probability-validation data.

Banks are cached only for exact replay across methods/restarts. Different setup
visits use different context and query seeds. Cache hits never make the data's
from-scratch scientific cost disappear from its dataset provenance.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from hashlib import sha256
from pathlib import Path
import fcntl,json,os,time,uuid
import numpy as np
from .banks import PairedBank,prepare_bank,save_bank,load_bank,array_descriptor
from .ordering import reorder_problem,reorder_bank
from .seeds import derive_seed

CONTEXTS=(16,64,256,1024)

def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def _json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

@dataclass(frozen=True)
class EpisodeRequest:
    stream_id:str
    position:int
    setup_index:int
    setup_fingerprint:str
    phase:str
    context_size:int
    context_seed:int
    query_seed:int
    ordering_seed:int|None
    context_bank_count:int
    query_bank_count:int=1
    query_index:int=0
    def __post_init__(self):
        if not _sha(self.stream_id) or not _sha(self.setup_fingerprint):raise ValueError('SHA256 stream/setup identities required')
        for name in ('position','setup_index','context_seed','query_seed','query_index'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError('nonnegative integer episode identities required')
        if self.phase not in ('pilot','main','quick_validation'):raise ValueError('unknown episode phase')
        if any(type(getattr(self,n)) is not int for n in ('context_size','context_bank_count','query_bank_count')):raise ValueError('explicit integer context/query counts required')
        if self.ordering_seed is not None and (type(self.ordering_seed) is not int or self.ordering_seed<0):raise ValueError('integer ordering seed required')
        if self.context_size not in CONTEXTS or self.context_bank_count<self.context_size or not 0<=self.query_index<self.query_bank_count:raise ValueError('invalid context/query dimensions')
        if self.phase!='quick_validation' and (self.context_bank_count!=self.context_size or self.query_bank_count!=1 or self.query_index!=0 or self.ordering_seed is None):raise ValueError('training requires fresh chosen-size bank, Q=1 and listing augmentation')
        if self.context_seed==self.query_seed:raise ValueError('independent context and query seeds required')
    @property
    def identity(self):return digest(asdict(self))
    def to_dict(self):return dict(asdict(self),request_id=self.identity)

class EpisodeStream:
    def __init__(self,setups,*,protocol_sha256,master_seed=20261117,replica=0,phase='main',diagnostic=False):
        self.setups=tuple(setups)
        if not self.setups or not _sha(protocol_sha256) or type(replica) is not int or replica not in (0,1) or phase not in ('pilot','main'):raise ValueError('invalid matched stream')
        if type(master_seed) is not int or master_seed<0 or type(diagnostic) is not bool:raise ValueError('integer master seed and explicit boolean diagnostic flag required')
        if not diagnostic and (len(self.setups)!=10000 or any(s.split!='train' for s in self.setups)):raise ValueError('production catalogue must contain all10000 training setups')
        if any(s.split not in ('train','diagnostic') for s in self.setups):raise ValueError('validation/test setup in training stream')
        if len({s.fingerprint for s in self.setups})!=len(self.setups):raise ValueError('duplicate training setup')
        self.master_seed=master_seed;self.replica=replica;self.phase=phase;self.protocol_sha256=protocol_sha256;self.diagnostic=diagnostic
        self.identity=digest({'schema':'finetuneforge.episode_stream.v1','protocol':protocol_sha256,'setups':[s.fingerprint for s in self.setups],'master_seed':master_seed,'replica':replica,'phase':phase,'contexts':CONTEXTS,'context_policy':'fresh_per_visit','query_cases':1,'ordering':'fresh_topological_per_visit'})
    def request(self,position):
        if type(position) is not int or position<0:raise ValueError('nonnegative episode cursor required')
        seed=derive_seed(self.master_seed,'episode-choice/'+self.phase,case=position,replica=self.replica)
        rng=np.random.default_rng(seed);index=int(rng.integers(len(self.setups)));size=CONTEXTS[int(rng.integers(4))]
        seeds=[derive_seed(self.master_seed,'episode-'+role+'/'+self.phase,setup=index,case=position,replica=self.replica) for role in ('context','query','ordering')]
        return EpisodeRequest(self.identity,position,index,self.setups[index].fingerprint,self.phase,size,*seeds,size)
    def batch(self,cursor,count):
        if type(count) is not int or count<1:raise ValueError('positive effective-batch size required')
        return tuple(self.request(cursor+i) for i in range(count))

class QuickValidationStream:
    """Same40 cases ×2 bank seeds ×4 sizes =320 probability scores/check."""
    def __init__(self,setups,*,protocol_sha256,master_seed=20261117,diagnostic=False):
        self.setups=tuple(setups)
        if not self.setups or not _sha(protocol_sha256):raise ValueError('validation setup/protocol identities required')
        if type(master_seed) is not int or master_seed<0 or type(diagnostic) is not bool:raise ValueError('integer master seed and explicit boolean diagnostic flag required')
        if not diagnostic and (len(self.setups)!=20 or any(s.split!='validation' for s in self.setups)):raise ValueError('quick validation requires the20 validation setups')
        if any(s.split not in ('validation','diagnostic') for s in self.setups):raise ValueError('train/test setup in quick validation')
        self.master_seed=master_seed;self.protocol_sha256=protocol_sha256
        self.identity=digest({'schema':'finetuneforge.quick_validation.v1','protocol':protocol_sha256,'setups':[s.fingerprint for s in self.setups],'seed':master_seed,'cases_per_setup':2,'banks':2,'contexts':CONTEXTS,'listing':'fixed_catalogue_listing'})
    def __iter__(self):
        position=0
        for i,spec in enumerate(self.setups):
            query=derive_seed(self.master_seed,'quick-validation-cases',setup=i)
            for bank in range(2):
                context=derive_seed(self.master_seed,'quick-validation-context',setup=i,bank=bank)
                for size in CONTEXTS:
                    for case in range(2):
                        yield EpisodeRequest(self.identity,position,i,spec.fingerprint,'quick_validation',size,context,query,None,1024,2,case)
                        position+=1
    def __len__(self):return len(self.setups)*2*4*2

class BankStore:
    """Authenticated replay cache, with incurred and accessible-cost records."""
    def __init__(self,directory,*,provider=prepare_bank,provider_identity=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.provider=provider
        if provider_identity is None:
            provider_identity=digest({name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('science.py','banks.py','spec.py')})
        if not _sha(provider_identity):raise ValueError('explicit provider SHA256 required')
        self.provider_identity=provider_identity;self.events=[]
    def obtain(self,spec,count,seed):
        if type(count) is not int or count<1 or type(seed) is not int or seed<0:raise ValueError('positive bank count and nonnegative seed required')
        descriptor={'provider_sha256':self.provider_identity,'problem_fingerprint':spec.fingerprint,'worlds':count,'seed':seed}
        key=digest(descriptor);target=self.directory/key;lockdir=self.directory/'.locks';lockdir.mkdir(exist_ok=True)
        started=time.perf_counter()
        with (lockdir/key).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            generated=False
            try:
                if not target.exists():
                    temporary=self.directory/('.inprogress-'+key+'-'+uuid.uuid4().hex)
                    bank=self.provider(spec,count,seed)
                    manifest=save_bank(bank,temporary)
                    _json(temporary/'cache_entry.json',dict(descriptor,dataset_id=key,bank_manifest_sha256=manifest['manifest_sha256']))
                    os.replace(temporary,target);generated=True
                entry=json.loads((target/'cache_entry.json').read_text())
                if entry!={**descriptor,'dataset_id':key,'bank_manifest_sha256':entry.get('bank_manifest_sha256')} or not _sha(entry.get('bank_manifest_sha256')):raise ValueError('cache identity mismatch')
                bank=load_bank(target,expected_sha256=entry['bank_manifest_sha256'])
                if bank.spec.fingerprint!=spec.fingerprint or len(bank.theta)!=count:raise ValueError('cache bank mismatch')
                cost=bank.manifest['cost']['forward_evaluations']
                event={'status':'complete','dataset_id':key,'worlds':count,'from_scratch_forward_evaluations':cost,'incurred_forward_evaluations':cost if generated else 0,'generated_now':generated,'bank_manifest_sha256':entry['bank_manifest_sha256'],'elapsed_seconds':time.perf_counter()-started,'path':str(target)}
                self.events.append(event)
                return bank,event
            except BaseException as exc:
                failure={'status':'failed','dataset_id':key,'descriptor':descriptor,'incurred_forward_evaluations':None,'forward_evaluation_upper_bound':count*len(spec.datasets),'elapsed_seconds':time.perf_counter()-started,'exception_type':type(exc).__name__,'exception':str(exc),'note':'partial provider work is unknown, never reported as zero'}
                failed=self.directory/'failures';failed.mkdir(exist_ok=True);_json(failed/(key+'-'+uuid.uuid4().hex+'.json'),failure);self.events.append(failure)
                raise

@dataclass(frozen=True)
class MaterializedEpisode:
    context:PairedBank
    query:PairedBank
    receipt:dict

def materialize_episode(stream,request:EpisodeRequest,store:BankStore):
    if request.stream_id!=stream.identity or not 0<=request.setup_index<len(stream.setups):raise ValueError('request/stream identity mismatch')
    spec=stream.setups[request.setup_index]
    if spec.fingerprint!=request.setup_fingerprint:raise ValueError('setup fingerprint mismatch')
    if isinstance(stream,EpisodeStream) and request!=stream.request(request.position):raise ValueError('episode does not match its deterministic cursor')
    context,context_cost=store.obtain(spec,request.context_bank_count,request.context_seed)
    queries,query_cost=store.obtain(spec,request.query_bank_count,request.query_seed)
    context=context.prefix(request.context_size)
    q=request.query_index
    query=PairedBank(spec,queries.theta[q:q+1],queries.observations[q:q+1],{'source_query_bank':query_cost,'query_index':q})
    ordering=None
    if request.ordering_seed is not None:
        _,mapping=reorder_problem(spec,request.ordering_seed)
        context=reorder_bank(context,mapping);query=reorder_bank(query,mapping);ordering=mapping.to_dict()
    receipt={'request':request.to_dict(),'context_theta':array_descriptor(context.theta),'context_observations':array_descriptor(context.observations),'query_theta':array_descriptor(query.theta),'query_observations':array_descriptor(query.observations),'datasets':[context_cost,query_cost],'ordering':ordering,'context_forward_evaluations':context.manifest['cost']['forward_evaluations'],'incurred_forward_evaluations':context_cost['incurred_forward_evaluations']+query_cost['incurred_forward_evaluations'],'query_cases':1,'posterior_draws':0}
    return MaterializedEpisode(context,query,receipt)
