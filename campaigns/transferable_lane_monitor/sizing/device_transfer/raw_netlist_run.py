"""Physical metadata without hand-labeled device roles or hand-coded sizing actions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import numpy as np
from adaptive_prior_run import TEST,TEST_SEEDS
from alarm_pilot import SOURCE_NETS
from anonymous_network import WidthNet
from blind_width_policy import rank_candidates
from blind_width_run import GOALS,run
from century_engine import enable
from experiment import arguments
from raw_netlist import BoundNet,DeviceNet,Graph,geometry

ARMS=("anon400","devices400","dense400","wires400")
SKY_ONLY="wires200"


def graphs(c):
    graph=geometry(c)
    return graph,Graph(graph.features,None)


def train(records,only=(*ARMS,SKY_ONLY)):
    examples=[];dense=[];anonymous=[];sky=[]
    for r in records:
        c=replace(SOURCE_NETS[r["name"]],pdk=r["pdk"])
        graph,unmasked=graphs(c)
        z=np.log(r["x"]);y=np.asarray(r["y"])-r["y"][0]
        anonymous.append((z,y));examples.append((z,y,graph));dense.append((z,y,unmasked))
        if c.pdk=="sky":sky.append(examples[-1])
    specifications={"anon400":(WidthNet,False,anonymous),"devices400":(DeviceNet,False,dense),
                    "dense400":(DeviceNet,True,dense),"wires400":(DeviceNet,True,examples),
                    SKY_ONLY:(DeviceNet,True,sky)}
    models={}
    for arm in only:
        constructor,attention,data=specifications[arm]
        models[arm]=constructor(attention).fit(data)
    return models


def bind(arm,net,c,seed):
    if arm=="anon400":return net
    graph=geometry(c,connected=arm in ("wires400",SKY_ONLY))
    if arm==SKY_ONLY and c.pdk!="sky":
        x=graph.features.copy();x[:,8:]=0.
        graph=Graph(x,graph.mask)
    perm=np.random.default_rng([seed,886]).permutation(2*len(c.parents))
    return BoundNet(net,graph.reorder(perm))


def select(arm,hist,y,lo,hi,seed,net):
    rng=np.random.default_rng([seed,len(hist),6481])
    if len(hist)<=4:return rng.uniform(lo,hi)
    return rank_candidates(np.asarray(hist),np.asarray(y),lo,hi,rng,network=net)


if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"raw_netlist"
    models=train(json.loads((root/"source.json").read_text()))
    def evaluate(job):
        n,c,a,s,g=job
        run(job,args,bind(a,models[a],c,s),root,select)
    # Goals sharing a policy seed use the same prefix. Run them sequentially to reuse each physical result.
    def pair(job):
        for g in GOALS:evaluate((*job,g))
    jobs=[(n,c,a,s) for a in (*ARMS,SKY_ONLY) for s in TEST_SEEDS for n,c in TEST.items()
          if a!=SKY_ONLY or c.pdk=="gf"]
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(pair,jobs))
