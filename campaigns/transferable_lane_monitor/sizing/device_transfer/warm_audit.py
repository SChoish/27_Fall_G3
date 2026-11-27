"""Reconstruct costs and stop decisions, then re-simulate distinct first-hit witnesses."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json,time
import numpy as np
from audit import checks
from circuit import simulate
from experiment import arguments,upper
from warm_space import bank,DEV,TEST,TARGETS

CIRCUITS={**DEV,**TEST,"seven_gf":TARGETS["seven_gf"]}
def read(file):
    return json.loads(file.read_text())
def gather(root):
    groups={}
    for split,tag in (("development_runs","development"),("sensitivity_runs","holdout_bank"),("test_runs","lock_aliases")):
        f=root/(split+".json")
        if f.exists():
            for key,rs in read(f).items():groups[tag,key]=rs
    for pattern,tag in (("continuous-*.json","continuous"),("strict-*.json","strict")):
        for f in root.glob(pattern):
            try:r=read(f)
            except json.JSONDecodeError:continue
            groups.setdefault((tag,r["arm"]),[]).append(r)
    return groups

def targets(groups,xs):
    pts=set()
    for runs in groups.values():
        for r in runs:
            name=r["circuit"]
            for stop in set(r["hits"].values()):
                step=next(s for s in r["trace"] if s["calls"]==stop)
                x=step["x"] if "x" in step else xs[name][step["i"]]
                pts.add((name,tuple(x)))
    return pts

def audit(args,groups,xs,fine):
    root=args.out/"warm"
    raw={};expenses={}
    for d in (root/"raw").iterdir():
        if not d.is_dir():continue
        cache={};reused=errors=0
        for f in d.glob("*.json"):
            r=read(f);cache[tuple(r["x"]),r["k"]]=r["raw"];reused+=int(r["reuse"])
            errors+=int("error" in r["raw"])
        raw[d.name]=cache;expenses[d.name]=dict(unique=len(cache),reused=reused,errors=errors)
    counts={};precision={};bad=[];verified={};points=set()
    for (tag,key),runs in groups.items():
        if tag=="lock_aliases":continue  # same bank trial already present in the sensitivity surface
        for r in runs:
            name=r["circuit"];nom=tuple(np.ones(len(upper(CIRCUITS[name]))))
            def rows(x,k):return [raw[name][x,j] for j in (k,k+1)]
            ref=rows(nom,0)+rows(nom,2)
            calls=4;hits={}
            for step in r["trace"]:
                x=tuple(step["x"] if "x" in step else xs[name][step["i"]]);calls+=2
                ok,v=checks(rows(x,0),ref[:2]);qual=ok and v<=.95 and calls+2<=40
                assert qual==step["confirmed"]
                if qual:
                    calls+=2;good,e=checks(rows(x,2),ref[2:])
                    for goal in (.95,.90):
                        if good and max(v,e)<=goal:hits.setdefault(str(goal),(calls,x))
                assert calls==step["calls"] and calls<=40
            assert {g:t for g,(t,x) in hits.items()}==r["hits"] and calls==r["calls"]
            acc=counts.setdefault(tag,dict(trials=0,standalone=0));acc["trials"]+=1;acc["standalone"]+=calls
            for goal,(t,x) in hits.items():
                points.add((name,x))
                a,b=fine[name,x];pas=bool(a and b<=float(goal))
                rc=precision.setdefault(tag,[0,0]);rc[0]+=1;rc[1]+=int(pas)
                if not pas:bad.append(dict(tag=tag,group=key,circuit=name,seed=r["seed"],goal=goal,calls=t,
                                          fine_edp=b if a else None))
                verified[name,x]=[bool(a),b if a else None]
    ledger=dict(phase_counts=counts,physical=expenses,first_hit_records_and_reverified=precision,
        unique_first_hits=len(points),fine_calls=4*len(list((root/"fine").glob("*.json"))),
        failures=bad)
    (root/"audit.json").write_text(json.dumps(ledger))
    print("audit",counts,"unique first hit",len(points),"finer failures",len(bad),flush=True)

def main():
    args=arguments();args.step_ps=.25;root=args.out/"warm";saved=root/"fine";saved.mkdir(exist_ok=True)
    xs={n:bank(c)[0] for n,c in CIRCUITS.items()}
    def measured(point):
        n,x=point;key=sha256(json.dumps([n,x]).encode()).hexdigest()
        file=saved/(key+".json")
        if not file.exists():
            rec=[simulate(CIRCUITS[n],x,k,args) for k in range(4)]
            file.write_text(json.dumps(dict(name=n,x=x,raw=rec)))
        return read(file)["raw"]
    done={};refs={}
    with ThreadPoolExecutor(4) as pool:
        while True:
            groups=gather(root);points=targets(groups,xs)
            missing=sorted(points-done.keys())
            for name in sorted(set(n for n,x in missing)-refs.keys()):
                refs[name]=measured((name,tuple(np.ones(len(upper(CIRCUITS[name]))))))
            for (name,x),record in zip(missing,pool.map(measured,missing)):
                ref=refs[name]
                pair=[checks(record[k:k+2],ref[k:k+2]) for k in (0,2)]
                done[name,x]=(all(p[0] for p in pair),max(p[1] for p in pair))
            ready=((root/"sensitivity_runs.json").exists() and len(list(root.glob("continuous-*.json")))==108
                   and len(list(root.glob("strict-*.json")))==66)
            if ready and targets(gather(root),xs)<=done.keys():
                audit(args,gather(root),xs,done);return
            time.sleep(8)
if __name__=="__main__":main()
