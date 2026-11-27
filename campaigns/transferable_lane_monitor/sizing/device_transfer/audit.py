"""Verify the cost ledger and each target hit directly from per-corner measurements."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from circuit import CIRCUITS,SOURCES,TARGETS,deck
from experiment import METHODS,structure,upper
from extend import SOURCE
from linesearch import HOLD,HELD,METHODS as HOLD_METHODS
from proposal import train

ALL_METHODS=(*METHODS,"rule","mlp_plus","mean_plus","gat_plus")

def checks(pair,reference):
    if any("error" in r for r in pair):
        return False, float("inf")
    best=0.
    for t,(r,b) in enumerate(zip(pair,reference)):
        for i,(p,q) in enumerate(zip(r["ports"],b["ports"])):
            if min(p[:4])<=0 or p[4]<.9 or p[5]>.1:
                return False,float("inf")
            if any(pi>1.10*qi for pi,qi in zip(p[:4],q[:4])):
                return False,float("inf")
            if t:
                p0=pair[0]["ports"][i]
                q0=reference[0]["ports"][i]
                shift=max(abs(p[j]-p0[j]) for j in (0,1))
                if shift>1.05*max(abs(q[j]-q0[j]) for j in (0,1)):
                    return False,float("inf")
            best=max(best,r["energy"]*max(p[:2])/(b["energy"]*max(q[:2])))
    return True,best

def read_raw(root,name,x,k):
    key=hashlib.sha256(json.dumps([np.round(x,6).tolist(),k]).encode()).hexdigest()
    d=json.loads((root/"physical"/name/(key+".json")).read_text())
    np.testing.assert_array_equal(d["x"],np.round(x,6))
    assert d["corner"]==k
    return d["raw"]

def audit(root):
    summaries=[]
    for name in (*TARGETS,HOLD):
        c=HELD if name==HOLD else CIRCUITS[name];n=2*len(c.parents)
        ref=[read_raw(root,name,np.ones(n),k) for k in range(4)]
        for method in (HOLD_METHODS if name==HOLD else ALL_METHODS):
            rows=[]
            for seed in (21,22,23):
                d=json.loads((root/f"{name}_{method}_{seed}.json").read_text())
                calls=4;hits={}
                for step in d["trace"]:
                    x=np.array(step["x"]);assert len(x)==n and np.all((.58<=x)&(x<=upper(c)))
                    pair=[read_raw(root,name,x,k) for k in (0,1)];calls+=2
                    ok,edp=checks(pair,ref[:2])
                    should=ok and edp<=.95 and calls+2<=40
                    assert should==step["verified"]
                    if should:
                        val=[read_raw(root,name,x,k) for k in (2,3)];calls+=2
                        ok,val_edp=checks(val,ref[2:])
                        for g in (.95,.9):
                            if ok and max(edp,val_edp)<=g:hits.setdefault(str(g),calls)
                    assert calls==step["cost"]
                assert hits==d["hits"] and calls==d["cost"] and calls<=40
                rows.append(d)
            for g in (.95,.9):
                times=[r["hits"].get(str(g)) for r in rows]
                summaries.append(dict(circuit=name,method=method,improvement=int(round((1-g)*100)),
                                      successes=sum(t is not None for t in times),first_hits=times.count(8),
                                      rmst=sum(t or 40 for t in times)/3,per_seed=times))
    with (root/"summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    costs={};errors=Counter()
    for p in (root/"physical").iterdir():
        records=[json.loads(k.read_text()) for k in p.glob("*.json")]
        costs[p.name]=len(records)
        errors[p.name]=sum("error" in r["raw"] for r in records)
    audit=dict(runs=len(summaries)*3//2,physical=costs,physical_failures=dict(errors),
               source_base=sum(costs[n] for n in SOURCES),source_expanded=sum(costs[n] for n in (*SOURCES,"source_pair")),
               target_physical=sum(costs[n] for n in (*TARGETS,HOLD)),
               stand_alone_target_cost=sum(json.loads(p.read_text())["cost"] for p in root.glob("*_2[123].json")))
    (root/"audit.json").write_text(json.dumps(audit,indent=2))
    print(audit)

def invariance(root):
    models=train(root,21);rng=np.random.default_rng(928)
    for c in (*CIRCUITS.values(),SOURCE,HELD):
        f,e=structure(c);perm=rng.permutation(len(f))
        for mode in ("mlp","mean","gat"):
            y,_=models[mode].outputs(f,e)
            yp,_=models[mode].outputs(f[perm],e[perm][:,perm])
            np.testing.assert_allclose(yp,y[perm],rtol=1e-8,atol=1e-8)
        net,_=deck(c,np.ones(len(f)),0)
        for i in range(len(f)):
            x=np.ones(len(f));x[i]=.9
            modified,_=deck(c,x,0)
            assert sum(a!=b for a,b in zip(net.splitlines(),modified.splitlines()))==1
    print("equivariance and individual device edits passed")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("--invariants",action="store_true")
    a=p.parse_args()
    if a.invariants:invariance(a.root)
    else:audit(a.root)
