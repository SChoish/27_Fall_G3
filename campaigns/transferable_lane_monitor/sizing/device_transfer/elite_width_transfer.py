"""Confirm mixed proposals on a new circuit and test source-elite width transfer."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np

from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_policy import rank_candidates
from blind_width_run import GOALS, run
from century_engine import enable
from circuit import Circuit
from corner_transfer import DOMAINS as EARLIER, seeds as earlier_seeds, selector
from experiment import arguments
from raw_netlist import geometry
from raw_netlist_run import train

ROOT=Path(__file__).parent
ARCHIVE=ROOT/"elite_width_transfer_measurements.zip"
FORK=Circuit("sky",(-1,0,1,2,1,4,0),(1.,1.75,2.1,3.15,1.45,2.55,1.95),
             (3,5,6),(1.35,.75,1.5),2.15)
DOMAINS={f"corner_stem_{p}":replace(FORK,pdk=p) for p in ("sky","gf")}
SEEDS=(101,102,103)
ARMS=("mlp200","mixed","retrieved")


def descriptors(c):
    value=geometry(c,connected=False).features
    mos=value[(value[:,1]+value[:,2])>0]
    polar=mos[:,1]
    width=mos[:,5]
    # Midranks within each polarity are invariant to device ordering and absolute scale.
    rank=np.empty(len(mos))
    for p in np.unique(polar):
        idx=polar==p
        w=width[idx]
        rank[idx]=((w[:,None]>w).sum(1)+.5*(w[:,None]==w).sum(1))/len(w)
    return polar,rank


def templates(source,target,random=False):
    p,r=descriptors(target)
    mapped=[]
    for index,row in enumerate(source):
        ps,rs=descriptors(replace(SOURCE_NETS[row["name"]],pdk=row["pdk"]))
        distance=(r[:,None]-rs)**2
        weights=np.exp(-.5*distance/.2**2)*(p[:,None]==ps)
        weights/=weights.sum(1,keepdims=True)
        y=np.array(row["y"]);z=np.log(row["x"])
        elite=np.argsort(y[1:,0]+4*np.maximum(y[1:,1:],0).sum(1),kind="stable")[:3]+1
        if random:
            elite=np.random.default_rng([4703,index]).choice(np.arange(1,len(z)),size=3,replace=False)
        mapped.extend(z[elite]@weights.T)
    return np.stack(mapped)


def policies(source,domains=DOMAINS,trial_seeds=None):
    baseline=selector(source,use_correction=False)
    net=train(source,only=("anon400",))["anon400"]
    def for_domain(n,c):
        banks={a:templates(source,c,random=(a=="random_templates")) for a in ("retrieved","random_templates")}
        d=banks["retrieved"].shape[1]
        seeds=trial_seeds if trial_seeds is not None else (earlier_seeds(n) if n in EARLIER else SEEDS)
        reordered={(a,s):bank[:,np.random.default_rng([s,886]).permutation(d)]
                   for a,bank in banks.items() for s in seeds}
        def pick(arm,hist,y,lo,hi,seed,unused):
            if arm not in ("retrieved","random_templates"):
                return baseline(arm,hist,y,lo,hi,seed,None)
            rng=np.random.default_rng([seed,len(hist),6481])
            if len(hist)<=4:return rng.uniform(lo,hi)
            def local(z,y,lo,hi,radius,stream):
                incumbent=z[np.argmin(y[:,0]+4*np.maximum(y[:,1:],0).sum(1))]
                mid=len(radius)//2
                base=stream.uniform(np.maximum(lo,incumbent-radius[:mid]),np.minimum(hi,incumbent+radius[:mid]))
                prior=reordered[arm,seed]
                count=len(radius)-mid
                anchors=np.clip(prior[np.arange(count)%len(prior)],lo,hi)
                draws=stream.uniform(np.maximum(lo,anchors-radius[mid:]),np.minimum(hi,anchors+radius[mid:]))
                draws[:len(prior)]=anchors[:len(prior)]
                return np.vstack((base,draws))
            return rank_candidates(np.asarray(hist),np.asarray(y),lo,hi,rng,network=net,local_builder=local)
        return pick
    return {n:for_domain(n,c) for n,c in domains.items()}


def verify(source,runs,choose):
    assert len(source)==4 and all(r["pdk"]=="sky" for r in source) and len(runs)==36
    return {n:audit(source,[r for r in runs if r["name"]==n],ARMS,{a:(choose[n],None) for a in ARMS},
                    {n:c},SEEDS,GOALS)[1] for n,c in DOMAINS.items()}


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as f:
            source,runs,stored=[json.loads(f.read(k+".json")) for k in ("source","runs","audit")]
        stats=verify(source,runs,policies(source));assert stats==stored
        return stats
    args=arguments();enable()
    with ZipFile(ROOT/"anonymous_width_measurements.zip") as f:
        source=[dict(r,pdk="sky") for r in json.loads(f.read("source.json"))]
    choose=policies(source)
    root=args.out/"elite_width_transfer";root.mkdir(exist_ok=True,parents=True)
    def pair(job):
        n,a,s=job
        rows=[]
        for goal in GOALS:
            run((n,DOMAINS[n],a,s,goal),args,None,root,choose[n])
            rows.append(json.loads((root/f"{n}-{a}-{s}-{goal}.json").read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs=[r for block in pool.map(pair,[(n,a,s) for s in SEEDS for n in DOMAINS for a in ARMS]) for r in block]
    stats=verify(source,runs,choose)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as f:
        for key,data in (("source",source),("runs",runs),("audit",stats)):
            f.writestr(key+".json",json.dumps(data,separators=(",",":"),allow_nan=False))
    return stats


if __name__=="__main__":
    print(main())
