"""Independent finer-step checks of one locked successful design per target."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from experiment import arguments, TARGETS, CIRCUITS, simulate
from audit import checks
from linesearch import HOLD,HELD

if __name__=="__main__":
    args=arguments()
    args.step_ps=.25
    refinements=[]
    for name in (*TARGETS,HOLD):
        c=HELD if name==HOLD else CIRCUITS[name]
        records=[json.loads(p.read_text()) for p in args.out.glob(name+"_*_2[123].json")]
        records=[r for r in records if r["name"]==name]
        options=[(float(g),t,r["method"],r["seed"],r) for r in records for g,t in r["hits"].items()]
        if not options: continue
        goal,t,method,seed,r=min(options,key=lambda v:v[:4])
        x=next(s["x"] for s in r["trace"] if s["cost"]==t)
        with ThreadPoolExecutor(2) as pool:
            rows=list(pool.map(lambda z:simulate(c,*z,args),[(a,k) for a in (np.ones(len(x)),x) for k in range(4)]))
        pairs=[checks(rows[4:6],rows[:2]),checks(rows[6:],rows[2:4])]
        edp=max(v[1] for v in pairs)
        refinements.append(dict(circuit=name,goal=goal,method=method,seed=seed,calls=t,x=x,
            measurements=rows,edp_ratio=edp if np.isfinite(edp) else None,
            passes=bool(all(v[0] for v in pairs) and edp<=goal)))
    (args.out/"refinement.json").write_text(json.dumps(refinements,indent=2))
    print([(r["circuit"],r["method"],r["passes"],r["edp_ratio"]) for r in refinements])
