"""Train and evaluate with the same log-width interval for every MOS."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import numpy as np
from adaptive_prior_run import TEST
from alarm_pilot import SOURCE_NETS
from blind_width_policy import propose
from blind_width_run import run
from century_engine import enable
from circuit import ratios
from experiment import arguments
from raw_netlist_run import bind,select,train
from warm_space import Store

ARMS=("bo","devices400","wires400")
SEEDS=(81,82)
GOALS=(.95,)


def uniform_upper(c):
    return np.full(2*len(c.parents),1.3)


def sample_source(item,args):
    index,name,pdk=item
    c=replace(SOURCE_NETS[name],pdk=pdk)
    store=Store(f"uniform_source_{pdk}_{name}",c,args)
    hi=np.log(uniform_upper(c));lo=np.log(.58)
    rng=np.random.default_rng([281,index])
    xs=[np.ones(len(hi))]
    for i in range(24):
        scale=1. if i%2==0 else rng.uniform(.1,.6)
        xs.append(np.round(np.exp(rng.uniform(lo*scale,hi*scale)),6))
    raw=[store.pair(x,0) for x in xs]
    y=[ratios(pair,raw[0]).tolist() for pair in raw]
    return dict(name=name,pdk=pdk,x=np.asarray(xs).tolist(),y=y,raw=raw)


def choose(arm,hist,y,lo,hi,seed,net):
    if arm=="bo":return propose(arm,hist,y,lo,hi,seed)
    return select(arm,hist,y,lo,hi,seed,net)


if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"uniform_netlist";root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        if args.mode=="source":
            jobs=[(i,n,p) for i,n in enumerate(SOURCE_NETS) for p in ("sky","gf")]
            src=list(pool.map(lambda it:sample_source(it,args),jobs))
            (root/"source.json").write_text(json.dumps(src))
        else:
            models=train(json.loads((root/"source.json").read_text()),only=ARMS[1:])
            def one(job):
                name,c,arm,seed,goal=job
                model=None if arm=="bo" else bind(arm,models[arm],c,seed)
                run(job,args,model,root,choose,uniform_upper)
            jobs=[(n,c,a,s,g) for s in SEEDS for n,c in TEST.items() for a in ARMS for g in GOALS]
            list(pool.map(one,jobs))
