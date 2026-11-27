"""Independent 20-lane noninferiority check for two preselected circuit designs."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
from campaign import transient, NOMINAL
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/local_monitor"))
from benchmark_transfer import characterize_circuit, CONDITIONS
from modes import BASE, hidden_model, panel, run_modes
from study import load_delay_table


def choose(record):
    hit=record["hits"].get("0.9")
    if hit is not None:
        return next(r["x"] for r in record["rows"] if r["calls"]==hit)
    feasible=[r for r in record["rows"] if max(r["dev_log_ratios"][1:])<=0]
    return min(feasible,key=lambda r:r["dev_log_ratios"][0])["x"]


def sweep(stages,x,args,path):
    corners=[(v,c,27) for c in (4,6,8,10,12) for v in (1.62,1.66,1.70,1.74,1.78,1.80)]
    with ThreadPoolExecutor(2) as pool:
        output=list(pool.map(lambda c:transient(stages,x,c,args),corners))
    with path.open("x") as f:
        w=csv.writer(f)
        w.writerow(["load_fF","vdd_V","rise_ps","fall_ps"])
        for (v,c,_),row in zip(corners,output):
            w.writerow([c,v,row["rise"],row["fall"]])
    return load_delay_table(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--stages",type=int,choices=(2,4),required=True)
    p.add_argument("--wasm",type=Path)
    p.add_argument("--node",default="node")
    p.add_argument("--ngspice",default="ngspice")
    args=p.parse_args()
    n=args.stages
    method={2:"kernel",4:"residual"}[n]
    record=json.loads((args.dir/f"s{n}_{method}_11.json").read_text())
    x=choose(record)
    out=args.dir/f"link{n}"
    out.mkdir(exist_ok=False)
    tables=[sweep(n,design,args,out/(name+".csv"))
            for design,name in [(NOMINAL,"reference"),(x,"candidate")]]
    char,_,_=characterize_circuit("original",2,824000)
    model=hidden_model(tables[0])
    q=np.load(BASE/"belief_weights.npz")["q"]
    summary={}
    with (out/"trajectories.csv").open("x") as f:
        writer=csv.writer(f)
        writer.writerow(["condition","seed","reference_failure","candidate_failure"])
        for mode in CONDITIONS:
            pairs=[]
            for chunk in range(0,256,16):
                seeds=np.arange(951000+chunk,951016+chunk)
                rates=[]
                for tbl in tables:
                    data=panel(2,seeds,mode,tbl,len(char[-1]))
                    _,metrics=run_modes(data,char,q,model)
                    rates.append(metrics[:,0])
                pairs.extend(np.column_stack(rates))
                writer.writerows((mode,int(s),a,b) for s,a,b in zip(seeds,*rates))
            values=np.array(pairs)
            delta=values[:,1]-values[:,0]
            rng=np.random.default_rng(43321)
            boot=delta[rng.integers(256,size=(10000,256))].mean(axis=1)
            upper=float(np.quantile(boot,1-.05/4))
            summary[mode]=dict(reference=values[:,0].mean(),candidate=values[:,1].mean(),
                               difference=delta.mean(),upper=upper,pass_screen=upper<=.01)
    (out/"summary.json").write_text(json.dumps(dict(stages=n,x=x,source_method=method,
        fresh_spice_evals=60,conditions=summary),indent=2)+"\n")
    print(summary,flush=True)


if __name__=="__main__":
    main()
