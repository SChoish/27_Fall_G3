"""Physical width coordinates, deterministic proposal banks and isolated SPICE measurements."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from alarm_pilot import TARGETS
from audit import checks
from circuit import Circuit, simulate
from experiment import arguments, upper

DEV={k:TARGETS[k] for k in ("fork","mixed_gf","load_sky","seven_sky")}
PAIR=replace(TARGETS["load_sky"],loads=(.72,1.38),coupling=2.4)
TREE=Circuit("sky",(-1,0,1,2,1,4,-2,6),(1,1.8,2.3,3.7,1.55,2.8,1,2.2),
             (3,5,7),(.75,1.20,1.45),2.4)
TEST={"pair_new_sky":PAIR,"pair_new_gf":replace(PAIR,pdk="gf"),
      "tree_new_sky":TREE,"tree_new_gf":replace(TREE,pdk="gf")}

def span(c):
    return np.log(upper(c)/.58)

def bank(c):
    hi=upper(c); dim=len(hi);n=dim//2
    # Same relative perturbations for geometrically matched SKY/GF twins.
    rng=np.random.default_rng(1881+dim)
    term=np.repeat([i in c.outputs for i in range(n)],2)
    load=np.repeat([c.loads[c.outputs.index(i)] if i in c.outputs else 1. for i in range(n)],2)
    pol=np.tile([-1.,1.],n)
    xs=[];labels=[]
    def put(x,tag):
        v=np.round(np.clip(x,.58,hi),6)
        if not any(np.array_equal(v,p) for p in xs):
            xs.append(v);labels.append(tag)
    put(np.ones(dim),"nominal");put(np.where(term,.84,1.),"role")
    for a,b in ((.91,.86),(.96,.74),(.89,.94),(.95,.90),(.80,1.),(.91,.98),(.94,1.),(.88,.97)):
        put(np.where(term,a,b),"role")
    for _ in range(64):
        a,b=rng.uniform([.78,.85],[.97,1.02]);load_slope=rng.uniform(-.035,.14)
        skew=rng.uniform(-.045,.045);jitter=rng.normal(0,.018,dim)
        put(np.where(term,a,b)*np.exp(term*load_slope*np.log(load)+pol*skew+jitter),"load")
    for _ in range(36):
        put(np.where(term,.88,.96)*np.exp(rng.uniform(-.095,.095,dim)),"local")
    for _ in range(30):
        put(np.exp(rng.uniform(np.log(.58),np.log(hi))),"global")
    return np.array(xs),np.array(labels)

class Store:
    def __init__(self,name,c,args):
        self.name,self.c,self.args=name,c,args
        self.root=args.out/"warm"/"raw"/name
        self.root.mkdir(parents=True,exist_ok=True)
    def read(self,x,k):
        x=np.round(x,6).tolist()
        key=hashlib.sha256(json.dumps([x,k]).encode()).hexdigest()+".json"
        dest=self.root/key
        if dest.exists():return json.loads(dest.read_text())["raw"]
        old=self.args.out/"alarm"/"physical"/self.name/key
        reuse=self.name in DEV and old.exists()
        raw=json.loads(old.read_text())["raw"] if reuse else simulate(self.c,x,k,self.args)
        import tempfile,os
        with tempfile.NamedTemporaryFile(mode="w",dir=self.root,delete=False) as f:
            json.dump(dict(x=x,k=k,raw=raw,reuse=reuse),f);tmp=f.name
        os.replace(tmp,dest)
        return raw
    def pair(self,x,k):
        return [self.read(x,j) for j in (k,k+1)]

def acquire(name,c,args):
    store=Store(name,c,args);X,tags=bank(c)
    ref=store.pair(X[0],0)+store.pair(X[0],2)
    if any("error" in a for a in ref):raise ValueError((name,ref))
    def one(x):
        dev=store.pair(x,0)
        ok,e=checks(dev,ref[:2])
        val=store.pair(x,2) if ok and e<=.95 else None
        return dict(x=x.tolist(),dev=dev,val=val)
    with ThreadPoolExecutor(2) as pool:rec=list(pool.map(one,X))
    (args.out/"warm"/(name+".json")).write_text(json.dumps(dict(reference=ref,records=rec,tags=tags.tolist())))
    print(name,len(rec),flush=True)

if __name__=="__main__":
    args=arguments()
    domains=DEV if args.mode=="source" else TEST
    if domains is TEST and not (args.out/"warm"/"lock.json").exists():
        raise RuntimeError("Lock tuning choices before measuring the test netlists")
    with ThreadPoolExecutor(4) as pool:list(pool.map(lambda it:acquire(*it,args),domains.items()))
