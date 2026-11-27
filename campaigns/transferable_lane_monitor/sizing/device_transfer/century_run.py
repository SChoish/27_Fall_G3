"""Paired cost-to-target benchmark, with independent first hits for each goal."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
from audit import checks
from circuit import Circuit, ratios
from experiment import arguments, upper
from warm_space import Store, TEST
from warm_policy import Config, warm_prefix, source_prior
from warm_continuous import Continuous
from century_heuristics import Heuristic

FRESH = Circuit("sky", (-1,0,1,2,-2,4,0), (1,2.1,1.8,3.4,1.05,2.45,2.85),
                (6,3,5), (1.45,.85,1.25), 1.65)
DOMAINS = {**TEST, "parallel100_sky":FRESH, "parallel100_gf":replace(FRESH,pdk="gf")}
HEURISTICS = ("random","bo","acor","de","group_acor","group_random")
LEARNED = ("residual","warm4","rank2","shuffled")
SEEDS = (31,32,33)
CAP = 100


def trial(name, circuit, arm, seed, args, model=None, metric=None, goals=(.95,.90), bo_goal=.90):
    store = Store(name, circuit, args)
    x0 = np.ones(len(upper(circuit)))
    ref = store.pair(x0,0) + store.pair(x0,2)
    if any("error" in row for row in ref):
        raise RuntimeError((name,ref))
    hist = [x0]
    labels = [ratios(ref[:2], ref[:2])]
    policy = Heuristic(arm,circuit,seed,bo_goal) if arm in HEURISTICS else None
    if policy is None:
        space = Continuous(circuit)
        space.inverse_direction = metric
        cfg = Config(residual=True, warm=4 if arm=="warm4" else 0,
                     first="warm" if arm=="warm4" else "role")
        order = warm_prefix(space,cfg,seed)
        indices = [0]
        model = model.copy()
    calls, hits, trace = 4, {}, []
    while calls+2<=CAP and len(hits)<len(goals):
        t = len(trace)
        if policy:
            x = policy.ask(np.array(hist), np.array(labels))
        else:
            if t<len(order):
                index=order[t]
            else:
                model.fit(space.encoded[indices],np.array(labels),20,.0018)
                index=space.rank(model,indices,labels,cfg,seed,t)
            indices.append(index)
            x=space.x[index]
        x=np.round(x,6)
        assert np.all((x>=.58)&(x<=upper(circuit)))
        dev=store.pair(x,0)
        ok,edp=checks(dev,ref[:2])
        calls+=2
        confirmed=bool(ok and edp<=max(goals) and calls+2<=CAP)
        val=None
        if confirmed:
            val=store.pair(x,2)
            valid,e=checks(val,ref[2:])
            calls+=2
            for goal in goals:
                if valid and max(e,edp)<=goal:
                    hits.setdefault(str(goal),calls)
        hist.append(x)
        labels.append(ratios(dev,ref[:2]))
        trace.append(dict(x=x.tolist(),calls=calls,dev=dev,validation=val))
    return dict(circuit=name,arm=arm,seed=seed,calls=calls,hits=hits,ref=ref,trace=trace)


def main():
    args=arguments()
    from century_engine import enable
    enable()
    root=args.out/"century"
    root.mkdir(parents=True,exist_ok=True)
    methods=HEURISTICS if args.mode=="search" else LEARNED
    models,metrics={},{}
    if args.mode=="source":
        from century_metric import fit_all
        models={s:source_prior(args.out,s) for s in SEEDS}
        metrics=fit_all(args,models)
    def run(job):
        name,arm,seed=job
        file=root/f"{name}-{arm}-{seed}.json"
        if file.exists():return
        result=trial(name,DOMAINS[name],arm,seed,args,models.get(seed),metrics.get((arm,seed)))
        tmp=file.with_suffix(".part")
        tmp.write_text(json.dumps(result,allow_nan=False))
        os.replace(tmp,file)
        print(name,arm,seed,result["hits"],flush=True)
    jobs=[(name,arm,s) for arm in methods for s in SEEDS for name in DOMAINS]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))


if __name__=="__main__":
    main()
