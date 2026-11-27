"""Deterministic electrical role policy followed by coordinate polling in four groups."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from adaptive import POLICY,size_move
from alarm_model import cost
from audit import checks
from century_run import DOMAINS,CAP
from circuit import ratios
from experiment import arguments,upper
from warm_space import Store

def search(name,c,args,goals=(.95,.90)):
    store=Store(name,c,args)
    x0=np.ones(len(upper(c)))
    ref=store.pair(x0,0)+store.pair(x0,2)
    hist=[x0];labels=[ratios(ref[:2],ref[:2])]
    groups=np.repeat([2*(j in c.outputs) for j in range(len(c.parents))],2)+np.arange(len(x0))%2
    calls,hits,trace=4,{},[]
    seen={tuple(x0)}
    queue=[]
    center=None
    radius=.10
    while calls+2<=CAP and len(hits)<len(goals):
        t=len(trace)
        if t<len(POLICY):
            x=size_move(c,POLICY,t,hist,labels)
        else:
            while not queue:
                best=hist[int(np.argmin(cost(labels)))]
                if center is not None and np.array_equal(best,center):radius*=.5
                if radius<1e-5:break
                center=best
                for j in range(4):
                    for sign in (-1,1):
                        x=np.round(np.clip(center*np.exp(sign*radius*(groups==j)),.58,upper(c)),6)
                        if tuple(x) not in seen:queue.append(x)
            if not queue:break
            x=queue.pop(0)
        x=np.round(x,6)
        seen.add(tuple(x))
        raw=store.pair(x,0)
        ok,e=checks(raw,ref[:2]);calls+=2
        val=None
        if ok and e<=max(goals) and calls+2<=CAP:
            val=store.pair(x,2)
            ok,v=checks(val,ref[2:]);calls+=2
            for goal in goals:
                if ok and max(e,v)<=goal:hits.setdefault(str(goal),calls)
        hist.append(x);labels.append(ratios(raw,ref[:2]))
        trace.append(dict(x=x.tolist(),calls=calls,dev=raw,validation=val))
    return dict(circuit=name,arm="role_poll",seed=0,calls=calls,hits=hits,ref=ref,trace=trace)

if __name__=="__main__":
    args=arguments()
    from century_engine import enable
    enable()
    def run(item):
        name,c=item
        file=args.out/"century"/f"{name}-role_poll-0.json"
        if file.exists():return
        r=search(name,c,args)
        file.write_text(json.dumps(r))
        print(name,r["hits"],flush=True)
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(run,DOMAINS.items()))
