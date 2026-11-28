"""Actual SPICE first-hit pilot with blinded coordinates and fully charged verification."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path
import json
import numpy as np
from alarm_pilot import SOURCE_NETS
from audit import checks
from blind_width_policy import propose,fit_prior
from century_engine import enable
from century_portfolio import NEW
from circuit import ratios
from experiment import upper,arguments
from warm_space import Store

ARMS=("random","acor","bo","transfer_bo")
SEEDS=(71,72,73)
GOALS=(.95,.90)


def source(args,root):
    def sample(item):
        index,(name,c)=item
        lo=np.log(.58);hi=np.log(upper(c))
        rng=np.random.default_rng([281,index])
        store=Store("blind_source_"+name,c,args)
        xs=[np.ones(len(hi))]
        for i in range(24):
            # Alternate whole-box and nominal-centered draws. All coordinates follow the same law.
            scale=1. if i%2==0 else rng.uniform(.1,.6)
            xs.append(np.round(np.exp(rng.uniform(lo*scale,hi*scale)),6))
        raw=[store.pair(x,0) for x in xs]
        y=np.array([ratios(pair,raw[0]) for pair in raw])
        return dict(name=name,x=np.asarray(xs).tolist(),y=y.tolist(),raw=raw)
    with ThreadPoolExecutor(4) as pool:
        records=list(pool.map(sample,enumerate(SOURCE_NETS.items())))
    (root/"source.json").write_text(json.dumps(records))


def run(job,args,prior,root):
    name,c,arm,seed,goal=job
    dest=root/f"{name}-{arm}-{seed}-{goal}.json"
    if dest.exists():return
    # Only the simulator adapter has the netlist, physical order and PDK.
    dim=len(upper(c)); perm=np.random.default_rng([seed,886]).permutation(dim)
    inv=np.argsort(perm);lo=np.full(dim,np.log(.58));hi=np.log(upper(c)[perm])
    coarse=Store(name,c,args)
    fine_args=copy(args);fine_args.step_ps=.25
    fine=Store("blind_fine_"+name,c,fine_args)
    nominal=np.ones(dim)
    ref=coarse.pair(nominal,0)+coarse.pair(nominal,2)
    calls=4;fine_ref=None;hit=None;trace=[]
    hist=[np.zeros(dim)];y=[ratios(ref[:2],ref[:2])]
    while calls+2<=100 and hit is None:
        z=propose(arm,hist,y,lo,hi,seed,prior if arm=="transfer_bo" else None)
        x=np.round(np.exp(z)[inv],6)
        dev=coarse.pair(x,0);calls+=2
        ok,e=checks(dev,ref[:2]);val=fraw=None
        if ok and e<=goal and calls+2<=100:
            val=coarse.pair(x,2);calls+=2
            ok,e=checks(val,ref[2:])
            if ok and e<=goal and calls+(8 if fine_ref is None else 4)<=100:
                if fine_ref is None:
                    fine_ref=fine.pair(nominal,0)+fine.pair(nominal,2);calls+=4
                fraw=fine.pair(x,0)+fine.pair(x,2);calls+=4
                results=[checks(fraw[k:k+2],fine_ref[k:k+2]) for k in (0,2)]
                if all(p and v<=goal for p,v in results):hit=calls
        hist.append(np.log(x[perm]));y.append(ratios(dev,ref[:2]))
        trace.append(dict(x=x.tolist(),dev=dev,val=val,fine=fraw,calls=calls))
    dest.write_text(json.dumps(dict(name=name,arm=arm,seed=seed,goal=goal,perm=perm.tolist(),
                                    ref=ref,fine_ref=fine_ref,trace=trace,calls=calls,hit=hit),allow_nan=False))
    print(name,arm,seed,goal,hit,flush=True)


if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"anonymous_width";root.mkdir(exist_ok=True)
    if args.mode=="source":source(args,root)
    else:
        rec=json.loads((root/"source.json").read_text())
        prior=fit_prior([(np.log(r["x"]),np.array(r["y"])-r["y"][0]) for r in rec])
        jobs=[(n,c,a,s,g) for a in ARMS for s in SEEDS for n,c in NEW.items() for g in GOALS]
        with ThreadPoolExecutor(args.workers) as pool:list(pool.map(lambda j:run(j,args,prior,root),jobs))
