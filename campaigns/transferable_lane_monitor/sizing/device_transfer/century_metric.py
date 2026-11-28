"""Learn a shared inverse metric from measured equal-length source displacements."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from alarm_model import features, inputs
from alarm_pilot import SOURCE_NETS
from circuit import ratios
from warm_space import Store


def acquire(args):
    file=args.out/"century"/"source_directions.json"
    if file.exists():
        return json.loads(file.read_text())
    out={}
    for name,c in SOURCE_NETS.items():
        store=Store("direction_"+name,c,args)
        dim=2*len(c.parents)
        rng=np.random.default_rng(6712+dim)
        ref=store.pair(np.ones(dim),0)
        batches=[]
        for a in range(3):
            center=rng.uniform(np.log(.78),np.log(.90),dim)
            direction=rng.normal(size=(8,dim))
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            xs=np.round(np.exp(center+.10*direction),6)
            with ThreadPoolExecutor(4) as pool:
                raw=list(pool.map(lambda x:store.pair(x,0),xs))
            if any("error" in row for pair in [ref,*raw] for row in pair):
                raise RuntimeError(name)
            labels=[ratios(pair,ref).tolist() for pair in raw]
            batches.append(dict(center=center.tolist(),x=xs.tolist(),y=labels,raw=raw))
        out[name]=dict(ref=ref,batches=batches)
    file.write_text(json.dumps(out))
    return out


def diagonal(g):
    return .7*g*g + .015*np.mean(g*g) + .0001


def preparation(g,roles):
    scale=1/np.sqrt(diagonal(g))
    # The feature map is shared across widths and topologies.
    R=roles/np.sqrt(len(g))
    V=scale[:,None]*R
    return -g*scale**2,V,R.T@(scale*g)


class InverseMetric:
    def __init__(self,B):
        self.B=B

    def __call__(self,g,roles):
        base,V,q=preparation(g,roles)
        return base-(V@self.B)@(self.B.T@q)


def training_pairs(data,model,shuffle):
    rows=[]
    for i,(name,c) in enumerate(SOURCE_NETS.items()):
        desc=features(c)
        for j,batch in enumerate(data[name]["batches"]):
            center=np.array(batch["center"])
            x=np.exp(center)
            yhat=model.predict(inputs(x,desc))[0]
            weight=3/(1+np.exp(-np.clip(60*(yhat[1:]+.01),-40,40)))
            g=np.r_[1.,weight]@model.jacobian(x,desc)
            y=np.array(batch["y"])
            # Feasible first, then minimum total positive log violation, then EDP.
            viol=np.maximum(y[:,1:],0).sum(1)
            rank=list(zip(viol>0,viol,y[:,0]))
            if shuffle:
                perm=np.random.default_rng([991,i,j]).permutation(len(rank))
                rank=[rank[k] for k in perm]
            d=np.log(batch["x"])-center
            d/=np.linalg.norm(d,axis=1,keepdims=True)
            pairs=np.array([d[k]-d[l] for k in range(8) for l in range(8) if rank[k]<rank[l]])
            rows.append((name,*preparation(g,desc[0]),pairs))
    return rows


def objective(rows,B):
    loss=0.
    grad=np.zeros_like(B)
    correct=count=0
    for _,base,V,q,pairs in rows:
        p=base-(V@B)@(B.T@q)
        norm=max(np.linalg.norm(p),1e-12)
        s=p/norm
        score=4*pairs@s
        loss+=np.logaddexp(0,-score).sum()
        correct+=np.sum(score>0)
        count+=len(score)
        ds=(-4/(1+np.exp(score)))@pairs
        dp=(ds-s*(ds@s))/norm
        v=V.T@dp
        grad-=np.outer(v,q@B)+np.outer(q,v@B)
    return loss/count + .002*np.sum(B*B), grad/count+.004*B, correct/count


def fit(rows):
    B=np.random.default_rng(514).normal(0,.12,(10,2))
    m=np.zeros_like(B)
    v=np.zeros_like(B)
    for t in range(1,161):
        _,grad,_=objective(rows,B)
        m=.9*m+.1*grad
        v=.999*v+.001*grad**2
        B-=.025*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return B


def fit_all(args,models):
    data=acquire(args)
    output={}
    learned={}
    for seed,model in models.items():
        true=training_pairs(data,model,False)
        for arm in ("rank2","shuffled"):
            rows=true if arm=="rank2" else training_pairs(data,model,True)
            B=fit(rows)
            metric=InverseMetric(B)
            learned[arm,seed]=metric
            cv={}
            for name in SOURCE_NETS:
                train=[r for r in rows if r[0]!=name]
                test=[r for r in true if r[0]==name]
                cv[name]=[objective(test,b)[2] for b in (np.zeros_like(B),fit(train))]
            # Existing SPICE surrogate was fit on all four source circuits. CV only holds out direction labels.
            output[f"{arm}-{seed}"]=dict(B=B.tolist(),source_ordinal_accuracy=objective(true,B)[2],
                                          held_out_direction_accuracy=cv)
    (args.out/"century"/"metrics.json").write_text(json.dumps(output))
    return learned
