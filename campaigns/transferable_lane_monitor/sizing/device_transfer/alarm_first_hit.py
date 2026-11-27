"""Repeat each distinct reported first-hit design at 0.25 ps, without optimizer feedback."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from alarm_pilot import TARGETS
from audit import checks
from circuit import simulate
from experiment import arguments, upper


def main():
    args=arguments()
    args.step_ps=.25
    root=args.out/"alarm"
    cache=args.out/"alarm_precision";cache.mkdir(exist_ok=True)
    ref={}
    def lookup(point):
        name,x=point
        key=hashlib.sha256(json.dumps([name,x]).encode()).hexdigest()
        path=cache/(key+".json")
        if not path.exists():
            result=[simulate(TARGETS[name],x,k,args) for k in range(4)]
            path.write_text(json.dumps(dict(name=name,x=x,raw=result)))
        return json.loads(path.read_text())["raw"]
    for name,c in TARGETS.items():
        ref[name]=lookup((name,tuple(np.ones(len(upper(c))))))
    assessed={}
    with ThreadPoolExecutor(2) as pool:
        while True:
            trials=[];needed=set()
            for path in sorted(root.glob("*-*-*.json")):
                try:r=json.loads(path.read_text())
                except json.JSONDecodeError:continue
                trials.append(r)
                for stop in set(r["hits"].values()):
                    x=next(s["x"] for s in r["trace"] if s["calls"]==stop)
                    needed.add((r["circuit"],tuple(x)))
            points=sorted(needed-assessed.keys())
            for p,raw in zip(points,pool.map(lookup,points)):
                base=ref[p[0]]
                pairs=[checks(raw[k:k+2],base[k:k+2]) for k in (0,2)]
                assessed[p]=(all(v[0] for v in pairs),max(v[1] for v in pairs))
            if len(trials)==175:break
            time.sleep(8)
    verified=[]
    for r in trials:
        for goal,t in r["hits"].items():
            x=next(s["x"] for s in r["trace"] if s["calls"]==t)
            ok,val=assessed[r["circuit"],tuple(x)]
            verified.append(dict(circuit=r["circuit"],method=r["method"],seed=r["seed"],
                                 goal=goal,calls=t,passes=ok and val<=float(goal),edp=val if ok else None))
    raw=[json.loads(f.read_text()) for f in sorted(cache.glob("*.json"))]
    report=dict(unique_candidates=len(assessed),spice_calls=4*len(raw),tests=verified,raw=raw)
    (root/"all_first_hits.json").write_text(json.dumps(report))
    print(len(assessed),len(verified),sum(v["passes"] for v in verified),report["spice_calls"])

if __name__=="__main__":main()
