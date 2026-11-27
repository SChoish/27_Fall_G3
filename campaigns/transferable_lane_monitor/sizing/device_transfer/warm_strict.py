"""Secondary selection in continuous space where the strict goal has known witnesses."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import numpy as np
from alarm_pilot import TARGETS
from experiment import arguments
from warm_continuous import Continuous,Live
from warm_policy import Config,rollout,source_prior
from warm_space import TEST,Store

CANDIDATES={
    "scalar":Config(),
    "local1":Config(warm=1),
    "local2":Config(warm=2),
    "global2":Config(warm=2,region="global"),
    "small_random":Config(cap=2,reuse=.5,select="random"),
    "small_guard":Config(cap=2,reuse=.5,select="guard"),
    "large_score":Config(cap=8,reuse=1.,select="score"),
    "unit":Config(unit=True),
    "calibrated_small":Config(residual=True,cap=2,reuse=.5,select="random")}

def run(args,name,c,alias,cfg,seed,prior):
    root=args.out/"warm";path=root/f"strict-{name}-{alias}-{seed}.json"
    if path.exists():return json.loads(path.read_text())
    space=Continuous(c);live=Live(space,Store(name,c,args))
    r=rollout(space,live,prior,cfg,seed)
    for s in r["trace"]:s["x"]=space.x[s["i"]].tolist()
    r=dict(circuit=name,arm=alias,**r);path.write_text(json.dumps(r));return r

if __name__=="__main__":
    args=arguments();root=args.out/"warm"
    if args.mode=="source":
        seeds=(21,22,23);domains={n:TARGETS[n] for n in ("mixed_gf","seven_gf")}
        opts=CANDIDATES
    else:
        seeds=(31,32,33);domains=TEST
        selected=json.loads((root/"strict_lock.json").read_text())["config"]
        opts={"strict_selected":Config(**selected)}
    priors={s:source_prior(args.out,s) for s in seeds}
    def study(item):
        name,c=item
        results=[run(args,name,c,m,cfg,s,priors[s]) for s in seeds for m,cfg in opts.items()]
        print("strict",args.mode,name,flush=True);return results
    with ThreadPoolExecutor(4) as pool:rows=sum(pool.map(study,domains.items()),[])
    if args.mode=="source":
        def times(m):
            group=[r for r in rows if r["arm"]==m]
            return [float(np.mean([r["hits"].get(g,40) for r in group])) for g in ("0.9","0.95")]
        winner=min(opts,key=lambda m:(*times(m),opts[m].warm,opts[m].cap,m))
        (root/"strict_lock.json").write_text(json.dumps(dict(winner=winner,config=asdict(opts[winner]),
                  scores={m:times(m) for m in opts},seeds=seeds,domains=list(domains))))
