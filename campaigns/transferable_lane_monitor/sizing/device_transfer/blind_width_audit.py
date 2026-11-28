"""Replay anonymous policy proposals and every charged SPICE call from raw observations."""
import csv
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from blind_width_policy import fit_prior,moments,propose
from blind_width_run import ARMS,GOALS,NEW,SEEDS
from audit import checks
from circuit import ratios
from experiment import upper


def audit(source,runs,arms=ARMS,selectors=None,domains=NEW,seeds=SEEDS,goals=GOALS,width_upper=upper,*,allow_invalid=False,replay_proposals=True):
    examples=[]
    for r in source:
        x=np.asarray(r["x"])
        assert len(x)==25 and len(r["raw"])==25
        raw=r["raw"]
        y=np.array([ratios(pair,raw[0]) for pair in raw])
        np.testing.assert_allclose(r["y"],y)
        z=np.log(x)
        np.testing.assert_allclose(moments(z),moments(z[:,::-1]),atol=1e-12)
        examples.append((z,y-y[0]))
    prior=fit_prior(examples)
    seen={};keys=set();prefix={}
    assert len(runs)==len(domains)*len(arms)*len(seeds)*len(goals)
    fine_witnesses=0
    for r in runs:
        n,arm,seed,g=(r[k] for k in ("name","arm","seed","goal"))
        assert (n,arm,seed,g) not in keys and n in domains and arm in arms and seed in seeds and g in goals
        keys.add((n,arm,seed,g))
        p=np.array(r["perm"]);dim=len(p);inv=np.argsort(p)
        np.testing.assert_array_equal(p,np.random.default_rng([seed,886]).permutation(dim))
        def record(x,start,values,fine=False):
            for k,v in enumerate(values,start):
                assert allow_invalid or "error" not in v
                key=(n,tuple(x),k,fine)
                if allow_invalid and "error" in v:
                    continue
                assert seen.setdefault(key,v)==v
        base=np.ones(dim);record(base,0,r["ref"])
        lo=np.full(dim,np.log(.58));hi=np.log(width_upper(domains[n])[p])
        hist=[np.zeros(dim)];labels=[ratios(r["ref"][:2],r["ref"][:2])]
        calls=4;fine_ref=None;hit=None
        for t,s in enumerate(r["trace"]):
            assert hit is None
            if replay_proposals or t<4:
                if replay_proposals and selectors and arm in selectors:
                    select,net=selectors[arm]
                    z=select(arm,hist,labels,lo,hi,seed,net)
                else:
                    z=propose(arm,hist,labels,lo,hi,seed,prior if arm=="transfer_bo" else None)
                x=np.round(np.exp(z)[inv],6);np.testing.assert_allclose(x,s["x"],rtol=0,atol=1e-6)
            x=np.array(s["x"])
            if t<4:assert prefix.setdefault((n,seed,t),x.tolist())==x.tolist()
            assert np.all((x>=.58)&(x<=width_upper(domains[n])))
            record(x,0,s["dev"]);calls+=2
            ok,e=checks(s["dev"],r["ref"][:2])
            assert (s["val"] is not None)==bool(ok and e<=g and calls+2<=100)
            need=False
            if s["val"] is not None:
                calls+=2;record(x,2,s["val"])
                ok,e=checks(s["val"],r["ref"][2:])
                need=ok and e<=g and calls+(8 if fine_ref is None else 4)<=100
            assert (s["fine"] is not None)==bool(need)
            if need:
                if fine_ref is None:
                    fine_ref=r["fine_ref"];record(base,0,fine_ref,True);calls+=4
                record(x,0,s["fine"],True);calls+=4
                if all(ok and val<=g for ok,val in
                       [checks(s["fine"][k:k+2],fine_ref[k:k+2]) for k in (0,2)]):
                    hit=calls;fine_witnesses+=1
            assert s["calls"]==calls and calls<=100
            hist.append(np.log(x[p]));labels.append(ratios(s["dev"],r["ref"][:2]))
        assert r["hit"]==hit and r["calls"]==calls and (hit is not None or calls==100)
    summary=[]
    for n in ("combined",*domains):
        for a in arms:
            for g in goals:
                ts=[r["hit"] for r in runs if r["arm"]==a and r["goal"]==g and n in ("combined",r["name"])]
                for cap in (40,100):
                    summary.append(dict(domain=n,method=a,goal=g,cap=cap,n=len(ts),
                                        successes=sum(t is not None and t<=cap for t in ts),
                                        mean=float(np.mean([min(t or cap,cap) for t in ts]))))
    meta=dict(runs=len(runs),standalone_target=sum(r["calls"] for r in runs),unique_target_rows=len(seen),
              source_rows=sum(len(r["x"])*2 for r in source),fine_first_hits=fine_witnesses)
    return summary,meta


if __name__=="__main__":
    path=Path("runs/anonymous_width")
    archive=Path("anonymous_width_measurements.zip")
    if sys.argv[1:]==["replay"]:
        with ZipFile(archive) as z:
            read=lambda k:json.loads(z.read(k+".json"))
            summary,stats=audit(read("source"),read("runs"))
            assert stats==read("audit") and summary==read("summary")
    else:
        source=json.loads((path/"source.json").read_text())
        runs=[json.loads(f.read_text()) for f in sorted(path.glob("long100*.json"))]
        summary,stats=audit(source,runs)
        with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=8) as z:
            for name,obj in (("source",source),("runs",runs),("audit",stats),("summary",summary)):
                z.writestr(name+".json",json.dumps(obj,separators=(",",":"),allow_nan=False))
        with (path/"summary.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    print(stats)
