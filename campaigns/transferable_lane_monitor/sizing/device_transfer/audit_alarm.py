"""Reconstruct time-to-target, buffer use and costs from the serialized SPICE ledger."""
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from alarm_pilot import TARGETS, METHODS, SEEDS, train_source
from alarm_model import inputs, features, direction
from audit import checks
from circuit import simulate
from experiment import arguments, upper

ALL=(*METHODS,"role40","keep_head","reset_head","prefix_attention")
GOALS=(.95,.90)

def check_learning(root):
    model,beta=train_source(root,21)
    assert np.linalg.norm(beta)>0
    for c in TARGETS.values():
        x=np.full(2*len(c.parents),.9);desc=features(c)
        def pred(z):return model.predict(inputs(x*np.exp(z),desc))[0]
        h=1.e-5
        direct=np.stack([(pred(h*u)-pred(-h*u))/(2*h) for u in np.eye(len(x))],axis=1)
        np.testing.assert_allclose(direct,model.jacobian(x,desc),atol=1e-7)
        for kind in ("euclid","scalar","attention","pullback"):
            delta,grad=direction(model,x,desc,kind,beta)
            assert delta@grad < 0
        perm=np.random.default_rng(932).permutation(len(x))
        np.testing.assert_allclose(model.predict(inputs(x,desc)),model.predict(inputs(x[perm],(desc[0][perm],desc[1]))),atol=1e-12)
    clone=model.copy();before=[p.copy() for p in clone.params];y=np.array([[.1,.2,.3,.4,.5]])
    clone.fit(inputs(x,desc),y,4,.002,True)
    for i in (0,1):np.testing.assert_array_equal(clone.params[i],before[i])
    for i in (2,3):assert not np.array_equal(clone.params[i],before[i])
    for p,q in zip(model.params,before):np.testing.assert_array_equal(p,q)

def raw(root,name,x,k):
    key=hashlib.sha256(json.dumps([np.round(x,6).tolist(),k]).encode()).hexdigest()
    p=root/"physical"/name/(key+".json")
    rec=json.loads(p.read_text())
    assert rec["corner"]==k and rec["x"]==list(x)
    return rec["raw"]

def audit(root):
    detailed=[];witnesses={};total=0;buffer={};effects=[]
    for name,c in TARGETS.items():
        nominal=np.ones(len(upper(c)))
        ref=[raw(root,name,nominal,k) for k in range(4)]
        target={}
        for method in ALL:
            trials=[]
            for seed in ([21] if method in ("role","role40") else SEEDS):
                row=json.loads((root/f"{name}-{method}-{seed}.json").read_text())
                calls=4;hits={};old=[];selections=[]
                for i,s in enumerate(row["trace"]):
                    x=s["x"];calls+=2
                    assert np.all(np.asarray(x)>=.58) and np.all(np.asarray(x)<=upper(c))
                    if i==0:
                        want=np.repeat([.84 if j in c.outputs else 1. for j in range(len(c.parents))],2)
                        np.testing.assert_array_equal(x,want)
                    if s["origin"] is not None:
                        assert s["origin"]<i and "buffer" in method
                        selections.append(i)
                        if s["origin"]<i-1:old.append(i)
                    ok,e=checks([raw(root,name,x,k) for k in (0,1)],ref[:2])
                    assert s["confirmed"]==(ok and e<=.95 and calls+2<=40)
                    if s["confirmed"]:
                        calls+=2
                        ok,v=checks([raw(root,name,x,k) for k in (2,3)],ref[2:])
                        for g in GOALS:
                            if ok and max(e,v)<=g:
                                hits.setdefault(str(g),calls)
                                key=(name,g)
                                rank=(calls,method,seed)
                                if key not in witnesses or rank<witnesses[key][0]:
                                    witnesses[key]=(rank,x)
                    assert s["calls"]==calls
                assert hits==row["hits"] and calls==row["calls"] and calls<=40
                total+=calls
                if "buffer" in method:
                    bucket=buffer.setdefault(method,[0,0])
                    bucket[0]+=len(selections);bucket[1]+=len(old)
                trials.append(hits)
            for g in GOALS:
                times=[r.get(str(g)) for r in trials]
                avg=float(np.mean([40 if e is None else e for e in times]))
                target[method,g]=avg
                detailed.append(dict(circuit=name,method=method,goal=g,seeds=len(times),
                                     hits=sum(e is not None for e in times),mean_capped=avg,times=times))
        for left,right in (("single","elite"),("elite","scalar"),("scalar","attention"),
                           ("attention","pullback"),("attention","random_buffer"),("attention","scored_buffer"),
                           ("random_buffer","scored_buffer"),("attention","prefix_attention"),
                           ("keep_head","reset_head"),("role40","scored_buffer")):
            for g in GOALS:
                effects.append(dict(circuit=name,goal=g,baseline=left,variant=right,
                                    calls_saved=target[left,g]-target[right,g]))
    costs={}
    for name in TARGETS:
        records=[json.loads(p.read_text()) for p in (root/"physical"/name).glob("*.json")]
        costs[name]=dict(unique=len(records),reused=sum(r["prior_cache"] for r in records),
                         errors=sum("error" in r["raw"] for r in records))
    for file,rows in (("summary.csv",detailed),("contrasts.csv",effects)):
        with (root/file).open("w") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    ledger=dict(trials=sum(r["seeds"] for r in detailed)//2,standalone_calls=total,source_calls=324,
                cost_by_circuit=costs,snapshot_selections_and_older=buffer)
    (root/"audit.json").write_text(json.dumps(ledger,indent=2))
    print(ledger,flush=True)
    return witnesses

def refine(args,choices):
    root=args.out/"alarm";args.step_ps=.25
    def check(name):
        c=TARGETS[name];n=len(upper(c))
        ref=[simulate(c,np.ones(n),k,args) for k in range(4)]
        points={}
        def entry(x,goal,method,calls):
            key=tuple(x)
            if key not in points:points[key]=[simulate(c,x,k,args) for k in range(4)]
            raw=points[key]
            result=[checks(raw[:2],ref[:2]),checks(raw[2:],ref[2:])]
            good=all(p[0] for p in result)
            worst=max(p[1] for p in result)
            return dict(method=method,goal=goal,calls=calls,x=x,edp=worst if good else None,passes=good and worst<=goal)
        checks_out=[]
        for goal in GOALS:
            if (name,goal) in choices:
                (calls,method,seed),x=choices[name,goal]
                checks_out.append(entry(x,goal,method+"-"+str(seed),calls))
        diagnostic=json.loads((root/f"witness_{name}.json").read_text())
        x=diagnostic["best_x"]
        checks_out.append(entry(x,.90,"independent_witness",diagnostic["calls"]))
        return dict(circuit=name,checks=checks_out,calls=4*(1+len(points)),reference=ref,
                    records=[dict(x=k,raw=v) for k,v in points.items()])
    with ThreadPoolExecutor(3) as pool:results=list(pool.map(check,TARGETS))
    (root/"finer.json").write_text(json.dumps(results))
    for r in results:print(r["circuit"],[(s["method"],s["goal"],s["edp"],s["passes"]) for s in r["checks"]])

if __name__=="__main__":
    args=arguments()
    check_learning(args.out)
    choices=audit(args.out/"alarm")
    refine(args,choices)
