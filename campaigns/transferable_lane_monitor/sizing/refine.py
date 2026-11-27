"""Quarter-step remeasurement of the two jointly verified representative designs."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from campaign import CORNERS, NOMINAL, transient, metrics

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--wasm",type=Path)
    p.add_argument("--node",default="node")
    p.add_argument("--ngspice",default="ngspice")
    args=p.parse_args()
    args.step_ps=.25
    results=[]
    for n,tag,target in [(2,"joint2",.95),(4,"joint10_4",.9)]:
        x=json.loads((args.dir/tag/"summary.json").read_text())["x"]
        with ThreadPoolExecutor(2) as pool:
            rows=list(pool.map(lambda pair:transient(n,*pair,args),
                               [(a,c) for a in (NOMINAL,x) for c in CORNERS]))
        ref,val=rows[:4],rows[4:]
        worst=np.maximum(metrics(val[:2],ref[:2],CORNERS[:2]),metrics(val[2:],ref[2:],CORNERS[2:]))
        results.append(dict(stages=n,x=x,step_ps=.25,physical_evals=8,
            reference=ref,candidate=val,max_edp_ratio=np.exp(worst[0]),max_slew_ratio=1.10*np.exp(worst[2]),
            passes=bool(worst[0]<=np.log(target) and max(worst[1:])<=0)))
    with (args.dir/"refinement.json").open("x") as f:
        json.dump(results,f,indent=2)
    print([(r["stages"],r["max_edp_ratio"],r["max_slew_ratio"],r["passes"]) for r in results])
