"""Transfer a source-trained slew correction to acquisition at an unseen PDK."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from adaptive_prior_run import TEST
from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_policy import rank_candidates
from blind_width_run import GOALS, run
from century_engine import enable
from circuit import Circuit, ratios
from experiment import arguments
from raw_netlist_run import train
from unseen_elite_metric import local_proposals
from warm_space import Store

ROOT = Path(__file__).parent
ARCHIVE = ROOT/"corner_transfer_measurements.zip"
NEW = "corner_fourout_gf"
DOMAINS = {k: TEST[k] for k in ("adapt_lane_gf","adapt_branch_gf")}
DOMAINS[NEW] = Circuit("gf", (-1,0,0,1,3,-2,5), (1.,1.65,2.55,1.4,2.7,1.05,2.0),
                       (1,2,4,6), (.8,1.35,1.1,1.55), 2.8)
ARMS = ("mlp200","slew_mlp","elite_natural","slew_elite")


def seeds(name):
    return (91,92,93) if name==NEW else (81,82,83)


def source_pairs(args):
    with ZipFile(ROOT/"anonymous_width_measurements.zip") as f:
        rows=json.loads(f.read("source.json"))
    def acquire(row):
        c=SOURCE_NETS[row["name"]]
        assert c.pdk=="sky"
        store=Store("corner_source_"+row["name"],c,args)
        val=[store.pair(x,2) for x in row["x"]]
        assert not any("error" in v for pair in val for v in pair)
        return dict(row,pdk="sky",confirmation=val)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(acquire,rows))


def features(z, y, reference):
    return np.column_stack((y[:,0]-reference[0], y[:,1]-reference[1],
                            y[:,2]-reference[2], z.mean(1), z.std(1)))


def fit_slew(rows):
    X=[];Y=[]
    for row in rows:
        y=np.asarray(row["y"])
        val=row["confirmation"]
        other=np.array([ratios(pair,val[0])[2] for pair in val])
        X.append(features(np.log(row["x"]),y,y[0]))
        Y.append(other-y[:,2])
    x=np.vstack(X)
    return np.linalg.solve(x.T@x+.01*np.eye(x.shape[1]),x.T@np.concatenate(Y))


def selector(source, use_correction=True):
    model=train(source,only=("anon400",))["anon400"]
    beta=fit_slew(source) if use_correction else None
    def select(arm,hist,y,lo,hi,seed,unused):
        rng=np.random.default_rng([seed,len(hist),6481])
        if len(hist)<=4:return rng.uniform(lo,hi)
        z=np.asarray(hist); obs=np.asarray(y)
        def local(z,y,lo,hi,radius,stream):
            if arm=="mixed":
                center=z[np.argmin(y[:,0]+4*np.maximum(y[:,1:],0).sum(1))]
                mid=len(radius)//2
                base=stream.uniform(np.maximum(lo,center-radius[:mid]),np.minimum(hi,center+radius[:mid]))
                directions=local_proposals(z,y,lo,hi,radius[mid:],stream,model,True,True,True)
                return np.vstack((base,directions))
            return local_proposals(z,y,lo,hi,radius,stream,model,True,True,True)
        def adjust(candidates, predicted):
            corrected=predicted.copy()
            gap=features(candidates,predicted,obs[0])@beta
            corrected[:,2]+=np.maximum(0.,gap)
            return corrected
        return rank_candidates(z,obs,lo,hi,rng,network=model,
                               local_builder=local if arm in ("elite_natural","slew_elite","mixed") else None,
                               adjust=adjust if arm in ("slew_mlp","slew_elite") else None)
    return select


def verify(source,runs,arms=ARMS):
    mixed = arms==("mixed",)
    assert len(source)==4 and all(r["pdk"]=="sky" and (mixed or len(r["confirmation"])==25) for r in source)
    assert len(runs)==sum(len(seeds(n))*len(arms)*len(GOALS) for n in DOMAINS)
    sel=selector(source,not mixed)
    stats={}
    for n,c in DOMAINS.items():
        _,stats[n]=audit(source,[r for r in runs if r["name"]==n],arms,{a:(sel,None) for a in arms},
                         {n:c},seeds(n),GOALS)
    return stats


def acquire(mixed=False):
    args=arguments();enable()
    root=args.out/"corner_transfer"
    root.mkdir(exist_ok=True,parents=True)
    arms=("mixed",) if mixed else ARMS
    if mixed:
        with ZipFile(ROOT/"anonymous_width_measurements.zip") as archive:
            source=[dict(r,pdk="sky") for r in json.loads(archive.read("source.json"))]
    else:
        source=source_pairs(args)
    sel=selector(source,not mixed)
    def pair(job):
        name,arm,seed=job
        rows=[]
        for g in GOALS:
            f=f"{name}-{arm}-{seed}-{g}.json"
            old=args.out/"unseen_elite_metric"/f
            if name!=NEW and arm in ("mlp200","elite_natural") and old.exists():
                rows.append(json.loads(old.read_text()))
                continue
            run((name,DOMAINS[name],arm,seed,g),args,None,root,sel)
            rows.append(json.loads((root/f).read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs=[r for block in pool.map(pair,[(n,a,s) for n in DOMAINS for s in seeds(n) for a in arms]) for r in block]
    stats=verify(source,runs,arms)
    with ZipFile(archive_path(mixed),"w",compression=ZIP_DEFLATED,compresslevel=9) as f:
        for key,data in (("source",source),("runs",runs),("audit",stats)):
            f.writestr(key+".json",json.dumps(data,separators=(",",":"),allow_nan=False))
    return stats


def archive_path(mixed):
    return ROOT/"mixed_transfer_measurements.zip" if mixed else ARCHIVE


def replay(mixed=False):
    with ZipFile(archive_path(mixed)) as f:
        source,runs,expected=[json.loads(f.read(k+".json")) for k in ("source","runs","audit")]
    stats=verify(source,runs,("mixed",) if mixed else ARMS)
    assert stats==expected
    return stats


if __name__=="__main__":
    mixed="--mixed" in sys.argv
    if mixed:sys.argv.remove("--mixed")
    print(replay(mixed) if sys.argv[1:]==["--replay"] else acquire(mixed))
