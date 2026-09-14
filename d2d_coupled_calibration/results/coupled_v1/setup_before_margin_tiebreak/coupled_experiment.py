"""Frozen, measurement-budget comparison for capacitively coupled D2D lanes."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import ndtr
from coupled_link import ROOT,LEVELS,CoupledContext,CoupledSetting,payloads,simulate,phase_table,fingerprint,link

OUT=ROOT/'results/coupled_v1'

def config():
    p=OUT/'config.json'
    if p.exists():return json.loads(p.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1409263301);contexts=[]
    for _ in range(48):
        common={'vdd':float(rng.uniform(.95,1.10)),'temperature_c':float(rng.uniform(10,75)),'rate_gbps':2.}
        lanes=[link.Context(**common,resistance_ohm=float(rng.uniform(15,55)),
                 inductance_nh=float(rng.uniform(.15,.5)),capacitance_pf=float(rng.uniform(.18,.4)),
                 rx_capacitance_pf=float(rng.uniform(.04,.09)),clock_offset_ui=float(rng.uniform(-.20,.20))) for _ in range(2)]
        contexts.append(asdict(CoupledContext(*lanes,coupling_pf=float(rng.uniform(.02,.14)))))
    cfg={'version':'coupled_v1','contexts':contexts,'train_ids':list(range(24)),
         'validation_ids':list(range(24,32)),'test_ids':list(range(32,48)),
         'strength_levels':list(LEVELS),'horizon':32,'budgets':[4,8,16,32],'primary_budget':16,
         'ratios':[.95,.9,.85],'primary_ratio':.9,'margin':.08,'guard_ui':.1,
         'calibration_bits':64,'calibration_seed_base':31609,
         'confirmation_bits':128,'confirmation_seed_base':59191,
         'confirmation_patterns':['random_a','random_b','opposite_alternating'],
         'points_per_ui':32,'train_seeds':[0,1,2],'alphas':[.1,2.5],'checkpoint_steps':[4000,8000],
         'width':128,'batch_size':256,'lr':.0003,'discount':.95,'tau':.005,
         'policy_noise':.2,'noise_clip':.5,'policy_frequency':2,'episodes_per_context':32,
         'bo_lengthscale':.6,'bo_candidates':1024,'ga_population':8,'ga_mutation_prob':.35,
         'baseline_sweeps':{'traditional':[{'phase_step':8},{'phase_step':16}],
             'bo':[{'lengthscale':x} for x in [.3,.6,1.2]],
             'ga':[{'population':p,'mutation_prob':m} for p in [4,8] for m in [.2,.5]]},
         'observation':'Last four (setting,measurement) records, incumbent setting/measurement, VDD, rate, margin, guard, remaining budget; 50 dimensions.',
         'reward':'Change in best observed utility minus 0.01. Fixed 32-step horizon, no success early termination.',
         'utility':'2-energy_ratio if quality passes; otherwise -2-4*sum(margin deficits)-4*mean(error fractions)-0.1*energy_ratio.',
         'energy':'Sum of both transmitter supply energies / total bits across both lanes. Paired reference on identical packet.',
         'confirmation':'One frozen setting must meet margins and zero errors on ALL 3 held-out packets; energy ratio uses summed packet energies to the matching 8+8 reference.',
         'selection':'Validation mean joint quality-energy success across all budgets/ratios/seeds, tie final utility. No test selection.',
         'budget_note':'One query measures both lanes, nine phase probes per lane. Initial reference query is additional. Identical observation costs for all methods.',
         'scope':'Lumped coupled-RLC toy; PTM predictive MOS model, behavioral RX; no PDK, extracted interposer, CDR, noise, compliance or real-time feedback claim.'}
    p.write_text(json.dumps(cfg,indent=2)+'\n');return cfg

def packet_for(i,which='calibration'):
    cfg=config()
    if which=='calibration':return payloads(cfg['calibration_seed_base']+1009*i,cfg['calibration_bits'])
    if which=='opposite_alternating':
        p=np.tile([0,1],cfg['confirmation_bits']//2).astype(np.uint8)
        return np.stack([p,1-p])
    return payloads(cfg['confirmation_seed_base']+1009*i+(0 if which=='random_a' else 900001),cfg['confirmation_bits'])

def cached_wave(i,a,b,which='calibration'):
    cfg=config();c=CoupledContext.from_dict(cfg['contexts'][i]);bits=packet_for(i,which)
    folder=OUT/'waves'/which;folder.mkdir(parents=True,exist_ok=True)
    p=folder/f'c{i:02d}_s{a}{b}.npz';strengths=(LEVELS[a],LEVELS[b])
    if p.exists():
        with np.load(p) as z:w={k:z[k].item() if z[k].ndim==0 else z[k] for k in z.files}
        assert w['fingerprint']==fingerprint(c,strengths,bits,cfg['points_per_ui'])
    else:
        w=simulate(c,strengths,bits,points_per_ui=cfg['points_per_ui'])
        temporary=p.with_suffix('.tmp.npz');np.savez_compressed(temporary,**w);temporary.replace(p)
    return c,bits,w

def prepare(split,workers=6):
    cfg=config();ids=cfg[split+'_ids'];t=time.perf_counter()
    margin=np.empty((len(ids),4,4,2,65));errors=np.empty_like(margin,dtype=np.int16)
    energy=np.empty((len(ids),4,4));seconds=[]
    def one(i,a,b):
        c,bits,w=cached_wave(i,a,b);m,e=phase_table(w,c,bits,cfg['guard_ui'])
        return i,a,b,m,e,w['energy_pj_bit'],w['seconds']
    with ThreadPoolExecutor(workers) as pool:
        futures=[pool.submit(one,i,a,b) for i in ids for a in range(4) for b in range(4)]
        for k,f in enumerate(as_completed(futures)):
            i,a,b,m,e,en,sec=f.result();j=ids.index(i)
            margin[j,a,b]=m;errors[j,a,b]=e;energy[j,a,b]=en;seconds.append(sec)
            if (k+1)%16==0:print(split,k+1,'/',len(futures),'SPICE transients',round(time.perf_counter()-t,1),'seconds',flush=True)
    np.savez_compressed(OUT/f'{split}_table.npz',ids=ids,margin=margin,errors=errors,energy=energy)
    (OUT/f'{split}_simulation.json').write_text(json.dumps({'transients':len(seconds),'elapsed_seconds':time.perf_counter()-t,
         'sum_transient_seconds':sum(seconds),'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()},indent=2)+'\n')

def quality(m,margin=.08):return bool(min(m['margins'])>=margin and sum(m['errors'])==0)
def features(m):return np.array([*m['margins'],*(np.array(m['errors'])/m['bits_per_lane']),m['energy_ratio']],np.float32)
def utility(m,margin=.08):
    if quality(m,margin):return float(2-m['energy_ratio'])
    return float(-2-4*np.maximum(0,margin-np.array(m['margins'])).sum()-4*np.mean(np.array(m['errors'])/m['bits_per_lane'])-.1*m['energy_ratio'])

class Table:
    def __init__(self,split):
        self.cfg=config();self.split=split;self.measurements=0
        with np.load(OUT/f'{split}_table.npz') as z:
            self.ids=z['ids'].tolist();self.margin=z['margin'];self.errors=z['errors'];self.energy=z['energy']
    def evaluate(self,i,s):
        self.measurements+=1;j=self.ids.index(i);a,b=s.strength0,s.strength1
        return {'margins':[float(self.margin[j,a,b,k,p]) for k,p in enumerate([s.phase0,s.phase1])],
                'errors':[int(self.errors[j,a,b,k,p]) for k,p in enumerate([s.phase0,s.phase1])],
                'energy_pj_bit':float(self.energy[j,a,b]),'energy_ratio':float(self.energy[j,a,b]/self.energy[j,3,3]),
                'bits_per_lane':self.cfg['calibration_bits']}

class Environment:
    observation_dim=50;action_dim=4
    def __init__(self,table,i,initial=CoupledSetting()):
        self.table=table;self.i=i;self.cfg=table.cfg;self.steps=0
        reference=CoupledSetting();ref_measurement=table.evaluate(i,reference)
        self.history=[(reference,ref_measurement)]
        self.setting=initial;self.measurement=ref_measurement if initial==reference else table.evaluate(i,initial)
        if initial!=reference:self.history.append((initial,self.measurement))
        self.best=max(self.history,key=lambda sm:utility(sm[1],self.cfg['margin']));self.done=False
    def obs(self):
        rows=[np.r_[s.action(),features(m)] for s,m in self.history[-4:]]
        rows=[rows[0]]*(4-len(rows))+rows
        s,m=self.best;c=self.cfg['contexts'][self.i]['lane0']
        return np.r_[np.concatenate(rows),s.action(),features(m),c['vdd'],c['rate_gbps']/4,
                     self.cfg['margin'],self.cfg['guard_ui'],(self.cfg['horizon']-self.steps)/self.cfg['horizon']].astype(np.float32)
    def step(self,action):
        if self.done:raise RuntimeError('Episode ended')
        before=utility(self.best[1],self.cfg['margin']);s=CoupledSetting.from_action(action)
        m=self.table.evaluate(self.i,s);self.setting,self.measurement=s,m;self.history.append((s,m));self.steps+=1
        if utility(m,self.cfg['margin'])>before:self.best=(s,m)
        reward=utility(self.best[1],self.cfg['margin'])-before-.01
        self.done=self.steps==self.cfg['horizon']
        return self.obs(),float(reward),False,self.done,m

def random_setting(rng):return CoupledSetting(int(rng.integers(4)),int(rng.integers(65)),int(rng.integers(4)),int(rng.integers(65)))

class Search:
    """Only public action codes and measurements explicitly observed so far."""
    def __init__(self,method,seed=0,params=None):
        self.method=method;self.rng=np.random.default_rng(seed);self.seen={};self.traces=[];self.cfg=config()
        self.params=params or {}
        self.generator=None;self.pending=None
    def observe(self,s,m):
        self.seen[s]=m;self.traces.append((s,m))
    def best(self):return max(self.seen,key=lambda s:utility(self.seen[s],self.cfg['margin']))
    def coordinate(self):
        s=self.best();d=self.params.get('phase_step',16);deltas=[1,d,1,d]
        while True:
            improved=False
            for axis in [0,2,1,3]:
                base=s
                for direction in [-1,1]:
                    x=list(base.codes());x[axis]+=direction*deltas[axis]
                    if not 0<=x[axis]<=[3,64,3,64][axis]:continue
                    p=CoupledSetting(*x)
                    if p in self.seen:continue
                    yield p
                    best=self.best()
                    if utility(self.seen[best])>utility(self.seen[s]):s=best;improved=True
            if not improved:
                if deltas[1]==1:return
                deltas[1]=max(1,deltas[1]//2);deltas[3]=deltas[1]
    def suggest(self):
        if self.method in ['traditional','perturbed']:
            if self.method=='perturbed' and self.rng.random()<.35:return random_setting(self.rng)
            if self.generator is None:self.generator=self.coordinate()
            try:return next(self.generator)
            except StopIteration:return self.best()
        if self.method=='random':return random_setting(self.rng)
        if self.method=='ga':return self.ga()
        if self.method=='bo':return self.bo()
        if self.method=='fixed8':return CoupledSetting()
        raise ValueError(self.method)
    def ga(self):
        popsize=self.params.get('population',self.cfg['ga_population'])
        pop=sorted(self.seen,key=lambda s:utility(self.seen[s]),reverse=True)[:popsize]
        if len(pop)<popsize:return random_setting(self.rng)
        def tournament():
            q=self.rng.choice(len(pop),size=3,replace=False)
            return pop[int(q.min())]
        for _ in range(64):
            a,b=tournament(),tournament();x=np.where(self.rng.random(4)<.5,a.codes(),b.codes()).astype(int)
            for j,hi in enumerate([3,64,3,64]):
                if self.rng.random()<self.params.get('mutation_prob',self.cfg['ga_mutation_prob']):
                    x[j]=int(self.rng.integers(hi+1)) if self.rng.random()<.25 else int(np.clip(x[j]+self.rng.choice([-1,1])*(1 if hi==3 else self.rng.choice([4,8,16])),0,hi))
            s=CoupledSetting(*x.tolist())
            if s not in self.seen:return s
        return random_setting(self.rng)
    def bo(self):
        if len(self.seen)<5:return random_setting(self.rng)
        ss=list(self.seen);x=np.stack([s.action() for s in ss]).astype(float)
        y=np.array([utility(self.seen[s]) for s in ss]);y=(y-y.mean())/(y.std()+1e-6)
        # Matérn-5/2 GP; fixed hyperparameters predeclared in config, nugget 1e-5.
        def kernel(a,b):
            r=np.sqrt(np.maximum(((a[:,None,:]-b[None,:,:])**2).sum(-1),0))/self.params.get('lengthscale',self.cfg['bo_lengthscale'])
            z=np.sqrt(5)*r;return (1+z+z*z/3)*np.exp(-z)
        k=kernel(x,x)+np.eye(len(x))*1e-5;cf=cho_factor(k,lower=True,check_finite=False)
        best=self.best();candidates={random_setting(self.rng) for _ in range(self.cfg['bo_candidates'])}
        for _ in range(128):
            candidates.add(CoupledSetting.from_action(best.action()+self.rng.normal(0,.25,4)))
        candidates=list(candidates-set(ss));candidates.sort(key=lambda s:s.codes())
        if not candidates:return best
        z=np.stack([s.action() for s in candidates]);kx=kernel(x,z)
        mu=kx.T@cho_solve(cf,y,check_finite=False)
        var=np.maximum(1-np.sum(kx*cho_solve(cf,kx,check_finite=False),axis=0),1e-12)
        sd=np.sqrt(var);improvement=mu-y.max()-.01;t=improvement/sd
        ei=improvement*ndtr(t)+sd*np.exp(-.5*t*t)/np.sqrt(2*np.pi)
        return candidates[int(np.argmax(ei))]

def collect():
    cfg=config();table=Table('train');rng=np.random.default_rng(602491)
    keys=['observations','actions','rewards','next_observations','terminals','timeouts','context_ids','episode_ids','behavior_ids']
    data={k:[] for k in keys};ep=0
    for i in table.ids:
        for rep in range(cfg['episodes_per_context']):
            method=['random','traditional','perturbed'][rep%3]
            initial=CoupledSetting() if rep%2 else random_setting(rng)
            env=Environment(table,i,initial);search=Search(method,seed=10000*i+rep+1123);search.observe(initial,env.measurement)
            obs=env.obs()
            for step in range(cfg['horizon']):
                setting=search.suggest();action=setting.action();nxt,r,t,timeout,m=env.step(action);search.observe(setting,m)
                for k,v in zip(keys,[obs,action,r,nxt,t,timeout,i,ep,rep%3]):data[k].append(v)
                obs=nxt
            ep+=1
    arr={k:np.asarray(v) for k,v in data.items()}
    for k in ['observations','actions','rewards','next_observations']:arr[k]=arr[k].astype(np.float32)
    np.savez_compressed(OUT/'dataset.npz',**arr)
    meta={'transitions':len(arr['rewards']),'episodes':ep,'logical_measurement_bundles':table.measurements,
          'behavior':'random / coordinate search / randomly perturbed coordinate search, train contexts only',
          'dataset_sha256':hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest()}
    (OUT/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','prepare','collect']);p.add_argument('--split',default='train');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.command=='init':print(config())
    elif a.command=='prepare':prepare(a.split,a.workers)
    else:collect()
