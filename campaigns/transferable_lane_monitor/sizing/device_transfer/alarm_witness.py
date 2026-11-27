"""Independent four-corner feasibility search. Its observations are never used by a trial."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from alarm_pilot import TARGETS
from circuit import ratios, simulate
from experiment import arguments, upper

def search(name,args):
    c=TARGETS[name];n=2*len(c.parents)
    role=np.repeat([int(i in c.outputs) for i in range(len(c.parents))],2)*2+np.arange(n)%2
    hi=upper(c); rng=np.random.default_rng(8104)
    rows={}
    def query(x):
        x=np.round(x,6);key=tuple(x)
        if key not in rows:
            rows[key]=[simulate(c,x,k,args) for k in range(4)]
        return rows[key]
    ref=query(np.ones(n))
    def evaluate(z):
        x=np.clip(np.exp(z[role]),.58,hi)
        raw=query(x)
        y=np.maximum(ratios(raw[:2],ref[:2]),ratios(raw[2:],ref[2:]))
        return y,x
    def score(y):
        return y[...,0]+6*np.maximum(0,y[...,1:]+.002).sum(-1)
    z=np.log([.95,.95,.91,.91]); radius=.11
    for _ in range(6):
        y,_=evaluate(z)
        jac=np.stack([(evaluate(z-.018*v)[0]-y)/(-.018) for v in np.eye(4)])
        moves=rng.uniform(-radius,radius,size=(3500,4))
        pred=y+moves@jac
        delta=moves[np.argmin(score(pred))]
        candidate=z+delta
        new,x=evaluate(candidate)
        if np.max(new[1:])<=0 and new[0]<=np.log(.9):
            break
        if score(new)<score(y):
            z=candidate; radius=min(.14,radius*1.2)
        else:
            radius*=.6
    best=None
    for x,raw in rows.items():
        y=np.maximum(ratios(raw[:2],ref[:2]),ratios(raw[2:],ref[2:]))
        if np.max(y[1:])<=0 and (best is None or y[0]<best[0]):
            best=(float(y[0]),x)
    return dict(circuit=name,best_x=best[1],edp_ratio=np.exp(best[0]),calls=4*len(rows),
                records=[dict(x=x,raw=raw) for x,raw in rows.items()])

if __name__=="__main__":
    args=arguments()
    root=args.out/"alarm";root.mkdir(exist_ok=True)
    with ThreadPoolExecutor(min(args.workers,3)) as pool:
        def save(name):
            path=root/f"witness_{name}.json"
            if not path.exists():
                r=search(name,args)
                path.write_text(json.dumps(r))
                print(name,r["edp_ratio"],r["calls"],flush=True)
        list(pool.map(save,TARGETS))
