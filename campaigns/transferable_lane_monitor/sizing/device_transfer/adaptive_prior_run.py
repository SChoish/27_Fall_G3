"""Calibrate source predictive means using only paid, anonymous target observations."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from anonymous_network_run import train,select
from blind_width_policy import propose,rank_candidates
from blind_width_run import run,NEW,SEEDS,GOALS
from century_engine import enable
from circuit import Circuit
from experiment import arguments

BRANCH=Circuit("sky",(-1,0,0,-2,3),(1.,1.65,2.65,1.1,2.05),(1,2,4),(.85,1.15,1.40),2.3)
SHIFT=replace(NEW["long100_sky"],loads=(.82,1.55),coupling=3.35)
TEST={f"adapt_{name}_{pdk}":replace(c,pdk=pdk) for name,c in (("lane",SHIFT),("branch",BRANCH)) for pdk in ("sky","gf")}
TEST_SEEDS=(81,82,83)
BASES=("bo","mlp_bo","gat_bo")
ADAPT=("attenuate","blend")
ARMS=(*BASES,*ADAPT)


def calibrated_mean(arm,nets,z,candidates,y,K):
    def outputs(net):
        return net.readout(net.encode(z)),net.readout(net.encode(candidates))
    m,c=outputs(nets["mlp"])
    if arm=="blend":
        other,other_c=outputs(nets["gat"])
        d,dc=other-m,other_c-c
    elif arm=="attenuate":
        d,dc=-m,-c
    else:
        raise ValueError(arm)
    # Per-output minimizer of the regularized GLS loss on paid development observations.
    inverse_d=np.linalg.solve(K,d)
    residual=y-y[0]-m
    variance=np.maximum(y.std(0),.02)**2
    alpha=np.clip((inverse_d*residual).sum(0)/((d*inverse_d).sum(0)+2*variance),0.,1.)
    return y[0]+m+alpha*d,y[0]+c+alpha*dc


def policy(arm,hist,y,lo,hi,seed,nets):
    if arm=="bo":return propose("bo",hist,y,lo,hi,seed)
    if arm in BASES:return select(arm,hist,y,lo,hi,seed,nets[arm.split("_")[0]])
    rng=np.random.default_rng([seed,len(hist),6481])
    if len(hist)<=4:return rng.uniform(lo,hi)
    return rank_candidates(np.asarray(hist),np.asarray(y),lo,hi,rng,
                           calibrated=lambda z,cand,obs,K:calibrated_mean(arm,nets,z,cand,obs,K))


if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"adaptive_prior";root.mkdir(exist_ok=True)
    source=json.loads((args.out/"anonymous_width"/"source.json").read_text())
    nets=train(source)
    # Reference runs on the old target are already archived and independently replayed.
    jobs=[(n,c,a,s,g) for circuits,seeds,arms in ((NEW,SEEDS,ADAPT),(TEST,TEST_SEEDS,ARMS))
          for a in arms for s in seeds for n,c in circuits.items() for g in GOALS]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job:run(job,args,nets,root,policy),jobs))
