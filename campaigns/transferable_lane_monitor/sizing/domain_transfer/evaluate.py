"""Blind finite-pool SPICE replay with a common candidate grid and explicit source costs."""
import argparse
import csv
import json
from math import erf
from pathlib import Path
import numpy as np
from model import Encoder, inputs, neighbors, adapt, choose_score
from data import GRID,DOMAINS,campaign

METHODS=("cold_bo","source_rank","random_head","mlp_frozen","mlp_new_head",
         "mlp_head_adapt","gat_frozen","gat_new_head","gat_head_adapt","gat_no_edges")
GOALS=(.95,.90)


def unpack(data):
    c,r=data["corners"],data["reference"]
    labels=[(campaign.metrics(d["raw"][:2],r[:2],c[:2]),
             campaign.metrics(d["raw"][2:],r[2:],c[2:])) for d in data["designs"]]
    assert np.allclose(np.array([d["x"] for d in data["designs"]]),GRID)
    return map(np.array,zip(*labels))


def cold(already,y):
    x=np.array(GRID)
    xseen=x[already] if len(already) else np.empty((0,2))
    xseen=np.vstack([[1,1],xseen])
    base=REFERENCE
    y=np.vstack([base,np.asarray(y).reshape(-1,5)])
    mu,sd=campaign.predict(xseen,y,x)
    sigma=np.maximum(sd[:,None],.05)*np.maximum(y.std(axis=0),[.18,.12,.15,.12,.04])
    z=(np.array([np.log(.90),0,0,0,0])-mu)/sigma
    return -np.log(np.maximum(.5*(1+np.vectorize(erf)(z/np.sqrt(2))),1e-15)).sum(axis=1)


REFERENCE=np.log([1,1/1.1,1/1.1,1/1.05,.9])


def run(method,seed,prior,e,dev,verify):
    history=[]
    labels=[]
    calls=4
    hits={}
    first=None
    order=np.random.default_rng(seed).permutation(len(GRID))
    while calls+2<=40 and len(hits)<2:
        if method=="cold_bo":
            score=cold(history,labels)
        elif method=="source_rank" or method.endswith("_frozen"):
            score=choose_score(prior)
        elif method.endswith("new_head") or method=="random_head":
            if not history:
                score=np.zeros(len(GRID))
            else:
                baseline=np.broadcast_to(REFERENCE,prior.shape)
                score=choose_score(adapt(e,history,np.array(labels),baseline))
        else:
            pred=adapt(e,history,np.array(labels),prior) if history else prior
            score=choose_score(pred)
        score[history]=np.inf
        j=int(order[score[order].argmin()])
        history.append(j)
        d=dev[j]
        labels.append(d)
        calls+=2
        checked=d[0]<=np.log(.95) and max(d[1:])<=0 and calls+2<=40
        if checked:
            calls+=2
            worst=np.maximum(d,verify[j])
            for goal in GOALS:
                if worst[0]<=np.log(goal) and max(worst[1:])<=0:
                    hits.setdefault(str(goal),calls)
        if first is None:
            first=j
    return dict(first=first,first5=hits.get("0.95")==8,first10=hits.get("0.9")==8,
                hit5=hits.get("0.95"),hit10=hits.get("0.9"),order=history,charged=calls)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    a=p.parse_args()
    data={n:json.loads((a.root/(n+".json")).read_text()) for n in DOMAINS}
    source=["sky2","sky4","sky2_light","sky4_light"]
    ds={n:tuple(unpack(v)) for n,v in data.items()}
    batches=[(inputs(data[n]["stages"],GRID,data[n]["load"]),ds[n][0]) for n in source]
    records=[]
    oracle={}
    for seed in range(11,17):
        models={k:Encoder(k,seed) for k in ("mlp","gat")}
        for net in models.values():
            net.fit(batches,seed)
        for domain in [k for k in DOMAINS if k not in source]:
            v=data[domain]
            f=inputs(v["stages"],GRID,v["load"])
            dev,verify=ds[domain]
            worst=np.maximum(dev,verify)
            oracle[domain]=[int(((worst[:,0]<=np.log(g))&(worst[:,1:].max(axis=1)<=0)).sum()) for g in GOALS]
            for method in METHODS:
                k="gat" if method.startswith("gat") else "mlp"
                net=Encoder("mlp",seed) if method=="random_head" else models[k]
                e=net.embedding(f)
                if method=="gat_no_edges":
                    # Inference-time edge ablation, same source-trained weights.
                    emb,_=net.forward(f,np.eye(f.shape[1],dtype=bool))
                    e=np.column_stack([emb,np.ones(len(f))])
                if method=="source_rank":
                    prior=np.mean([y for _,y in batches],axis=0)
                elif method=="random_head":
                    prior=np.broadcast_to(REFERENCE,(len(f),5))
                else:
                    prior=net.predict(e)
                records.append(dict(domain=domain,method=method,seed=seed,
                                    **run(method,seed,prior,e,dev,verify)))
            perm=np.random.default_rng(seed).permutation(f.shape[1])
            g=models["gat"]
            adj=neighbors(f.shape[1])
            np.testing.assert_allclose(g.forward(f,adj)[0],g.forward(f[:,perm],adj[perm][:,perm])[0],
                                       rtol=1e-9,atol=1e-9)
    root=a.root.parent
    (root/"replays.json").write_text(json.dumps(dict(coverage=oracle,records=records),separators=(",",":")))
    with (root/"summary.csv").open("w") as out:
        w=csv.writer(out)
        w.writerow(["domain","method","grid_hits_5","grid_hits_10","first5","first10","success5","cost5","success10","cost10"])
        for n,(o5,o10) in oracle.items():
            for m in METHODS:
                rows=[r for r in records if r["domain"]==n and r["method"]==m]
                vals=[]
                for goal in ("5","10"):
                    hits=[r["hit"+goal] for r in rows]
                    vals.extend([sum(x is not None for x in hits),sum(x or 40 for x in hits)/len(hits)])
                w.writerow([n,m,o5,o10,sum(r["first5"] for r in rows),sum(r["first10"] for r in rows),*vals])


if __name__=="__main__":
    main()
