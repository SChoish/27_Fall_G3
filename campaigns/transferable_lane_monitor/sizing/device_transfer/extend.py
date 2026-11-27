"""Train a second prior on a physically coupled source and include a simple sizing rule."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import circuit
import experiment as exp
import proposal

NAME="source_pair"
SOURCE=circuit.Circuit("sky",(-1,0,1,2,-2,4,5,6),(1,2,2,4,1,1.5,3,3.5),
                       (3,7),(.65,1.15),1.0)

def run():
    args=exp.arguments()
    exp.CIRCUITS[NAME]=SOURCE
    exp.SOURCES=(*exp.SOURCES,NAME)
    proposal.SOURCES=exp.SOURCES
    if args.mode=="source":
        exp.source(NAME,args)
        return
    import sys
    # All runs reuse the same proposal module imported by search_domain.
    original=proposal.propose
    def choose(method,models,c,hist,y,candidates,rng):
        if method=="rule":
            # Relative size schedule referenced to stage role. Both polarities share a move.
            terminal=np.repeat([i in c.outputs for i in range(len(c.parents))],2)
            target=np.ones(len(terminal))
            fraction=[(.84,1.),(.91,.86),(.96,.74),(.89,.94),(.95,.90)][(len(hist)-1)%5]
            target[terminal]=fraction[0];target[~terminal]=fraction[1]
            if len(hist)>5:
                target*=np.exp(rng.normal(0,.045,len(target)))
            return np.clip(target,.58,exp.upper(c))
        return original(method.removesuffix("_plus"),models,c,hist,y,candidates,rng)
    proposal.propose=choose
    exp.METHODS=("rule","mlp_plus","mean_plus","gat_plus")
    models={s:proposal.train(args.out,s) for s in (21,22,23)}
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda name:exp.search_domain(name,args,models),exp.TARGETS))

if __name__=="__main__":
    run()
