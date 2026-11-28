"""Matched neural surrogate and BO prior tests using the existing role-free source data."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from anonymous_network import WidthNet
from blind_width_policy import rank_candidates
from blind_width_run import run,NEW,SEEDS,GOALS
from century_engine import enable
from experiment import arguments

ARMS=("mlp","gat","mlp_bo","gat_bo")

def train(source):
    examples=[(np.log(r["x"]),np.asarray(r["y"])-r["y"][0]) for r in source]
    return {name:WidthNet(name=="gat").fit(examples) for name in ("mlp","gat")}

def select(arm,hist,y,lo,hi,seed,net):
    rng=np.random.default_rng([seed,len(hist),6481])
    if len(hist)<=4:return rng.uniform(lo,hi)
    return rank_candidates(np.asarray(hist),np.asarray(y),lo,hi,rng,network=net,gp=arm.endswith("_bo"))

if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"anonymous_network";root.mkdir(exist_ok=True)
    source=json.loads((args.out/"anonymous_width"/"source.json").read_text())
    nets=train(source)
    jobs=[(n,c,arm,s,g) for arm in ARMS for s in SEEDS for n,c in NEW.items() for g in GOALS]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda j:run(j,args,nets[j[2].split("_")[0]],root,select),jobs))
