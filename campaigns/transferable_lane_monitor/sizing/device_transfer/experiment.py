"""Sequential searches on continuous device widths. Reuse only identical raw observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from circuit import CIRCUITS,SOURCES,TARGETS,simulate,ratios

METHODS=("random","cold_bo","polarity","mlp","mean","gat","gat_frozen","gat_new")
GOALS=(.95,.90)
CAP=40

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["source","search"])
    for flag in ("sky","gf","wasm","out"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--workers",type=int,default=4)
    return p.parse_args()

def upper(c):
    return np.repeat([1. if p<0 else 1.3 for p in c.parents],2)

def structure(c):
    # Parallel N/P devices share their output; signal nets join upstream and downstream devices.
    n=2*len(c.parents)
    edge=np.zeros((n,n),int)
    feats=[]
    for i,p in enumerate(c.parents):
        depth=1
        a=p
        while a>=0:
            depth+=1
            a=c.parents[a]
        terminal=i in c.outputs
        load=c.loads[c.outputs.index(i)] if terminal else 0
        for j in range(2):
            k=2*i+j
            edge[k,k]=1
            edge[k,2*i+1-j]=2
            if p>=0:
                edge[k,2*p:2*p+2]=3
                edge[2*p:2*p+2,k]=4
            feats.append([j,np.log2(c.sizes[i]*(1+j))/3,depth/4,float(p<0),
                          float(terminal),load,c.parents.count(i)/2,1.])
    if c.coupling:
        a,b=c.outputs
        edge[2*a:2*a+2,2*b:2*b+2]=5
        edge[2*b:2*b+2,2*a:2*a+2]=5
    return np.array(feats),edge

class Meter:
    def __init__(self,name,args):
        self.name,self.c,self.args=name,CIRCUITS[name],args
        self.cache=args.out/"physical"/name
        self.cache.mkdir(parents=True,exist_ok=True)
    def read(self,x,k):
        # Quantize to six decimals before both hashing and generating SPICE.
        x=np.round(x,6).tolist()
        key=hashlib.sha256(json.dumps([x,k]).encode()).hexdigest()
        path=self.cache/(key+".json")
        if path.exists(): return json.loads(path.read_text())["raw"]
        r=simulate(self.c,x,k,self.args)
        path.write_text(json.dumps(dict(x=x,corner=k,raw=r)))
        return r
    def pair(self,x,start):
        return [self.read(x,k) for k in (start,start+1)]

def source(name,args):
    m=Meter(name,args)
    c=m.c
    n=2*len(c.parents)
    ref=m.pair(np.ones(n),0)
    y0=ratios(ref,ref)
    rng=np.random.default_rng(672+n)
    designs=[]
    for j in range(n):
        x=np.ones(n);x[j]=.84;designs.append(x)
    for _ in range(30):
        baseline=rng.uniform(.69,1.)
        designs.append(np.clip(baseline*np.exp(rng.normal(0,.14,n)),.58,upper(c)))
    with ThreadPoolExecutor(args.workers) as pool:
        raw=list(pool.map(lambda x:m.pair(x,0),designs))
    if any("error" in r for rows in raw for r in rows):raise RuntimeError("Source measurements missing")
    ys=np.stack([ratios(row,ref) for row in raw])
    gradients=(ys[:n]-y0)/np.log(.84)
    loss=ys[n:,0]+7*np.maximum(ys[n:,1:],0).sum(1)
    elite=np.argsort(loss)[:4]
    actor=np.log(np.array(designs[n:])[elite]).mean(0)
    (args.out/(name+"_source.json")).write_text(json.dumps(dict(designs=np.array(designs).tolist(),
        labels=ys.tolist(),per_device=np.column_stack([gradients,actor]).tolist(),ref=y0.tolist())))
    print(name,len(designs)*2+2,flush=True)

def pool(c,rng,inc):
    n=len(inc)
    # Groupwise common movement plus independent per-device perturbations.
    global_moves=np.exp(rng.uniform(-.34,.025,(160,n))+.11*rng.normal(size=(160,n)))
    local=inc*np.exp(.12*rng.normal(size=(160,n)))
    return np.clip(np.vstack([global_moves,local]),.58,upper(c))

def search_domain(name,args,models):
    from proposal import propose
    m=Meter(name,args)
    n=2*len(m.c.parents)
    ref=m.pair(np.ones(n),0)+m.pair(np.ones(n),2)
    if any("error" in r for r in ref):raise RuntimeError((name,ref))
    for pair in (ref[:2],ref[2:]):
        assert max(ratios(pair,pair)[1:])<=0,(name,pair)
    for seed in (21,22,23):
        for method in METHODS:
            rng=np.random.default_rng(seed)
            history=[np.ones(n)];labels=[ratios(ref[:2],ref[:2])]
            cost=4;hits={};trace=[]
            while cost+2<=CAP and len(hits)<2:
                loss=np.array(labels)
                index=np.argmin(loss[:,0]+7*np.maximum(loss[:,1:],0).sum(1))
                candidates=pool(m.c,rng,history[index])
                x=propose(method,models[seed],m.c,np.array(history),loss,candidates,rng)
                x=np.round(x,6)
                dev=m.pair(x,0);cost+=2
                y=ratios(dev,ref[:2])
                record=dict(x=x.tolist(),y=y.tolist(),verified=False)
                if y[0]<=np.log(.95) and max(y[1:])<=0 and cost+2<=CAP:
                    val=ratios(m.pair(x,2),ref[2:]);cost+=2
                    record.update(verified=True,validation=val.tolist())
                    for g in GOALS:
                        if max(y[0],val[0])<=np.log(g) and max(val[1:])<=0:
                            hits.setdefault(g,cost)
                history.append(x);labels.append(y)
                trace.append(dict(record,cost=cost))
            path=args.out/f"{name}_{method}_{seed}.json"
            path.write_text(json.dumps(dict(name=name,method=method,seed=seed,cost=cost,hits=hits,trace=trace)))
        print(name,seed,flush=True)

if __name__=="__main__":
    args=arguments()
    args.out.mkdir(exist_ok=True,parents=True)
    if args.mode=="source":
        for name in SOURCES:source(name,args)
    else:
        from proposal import train
        models={s:train(args.out,s) for s in (21,22,23)}
        with ThreadPoolExecutor(args.workers) as pool_:
            list(pool_.map(lambda name:search_domain(name,args,models),TARGETS))
