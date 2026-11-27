"""First hits if every finer reference and witness transient must fit the same cap."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from century_audit import verified
from century_portfolio_check import key
from century_portfolio import NEW
from century_run import DOMAINS

def main():
    root=Path("runs/century")
    extension=json.loads((root/"precision_continuation.json").read_text())
    rows=[]
    with ZipFile("century_measurements.zip") as z:
        fine={(r["circuit"],tuple(r["x"])):r["raw"] for r in json.loads(z.read("fine.json"))}
        for file in z.namelist():
            if not file.startswith("traces/"):continue
            r=json.loads(z.read(file));n=r["circuit"]
            goal=.90 if r["protocol"]=="strict" else .95
            base=r["hits"].get(str(goal))
            if base is not None:
                x=tuple(next(x for x,t,v in r["trace"] if t==base))
                assert verified(fine[n,x],fine[n,(1.,)*len(x)],goal)
            rows.append((r["protocol"],n,r["arm"],goal,base,None if base is None else base+8))
    with ZipFile("century_portfolio_measurements.zip") as z:
        fine=json.loads(z.read("fine.json"));fine.update(extension["fine"])
        for r in json.loads(z.read("records.json")):
            n,arm,seed,g=[r[k] for k in ("circuit","method","seed","goal")]
            matches=[v for v in extension["continued"] if
                     (v["circuit"],v["method"],v["seed"],v["goal"])==(n,arm,seed,g)]
            path=matches[0] if matches else r
            extra=0;hit=None
            for step in path["trace"]:
                val=step["validation"]
                if val is None or not verified(step["dev"]+val,r["ref"],g):continue
                x=tuple(step["x"])
                extra=8 if extra==0 else extra+4
                if verified(fine[key(n,x)],fine[key(n,(1.,)*len(x))],g):
                    hit=step["calls"]+extra;break
            assert path["hit"]==(None if hit is None else hit-extra)
            rows.append(("portfolio",n,arm,g,r["hit"],hit))
    result=[]
    for p,ns in [("joint",DOMAINS),("strict",DOMAINS),("bo95",DOMAINS),("portfolio",DOMAINS),("portfolio",NEW)]:
        subset=[r for r in rows if r[0]==p and r[1] in ns]
        for arm,g in sorted(set((r[2],r[3]) for r in subset)):
            ts=[r[-1] or 1000 for r in subset if (r[2],r[3])==(arm,g)]
            for cap in (40,100):
                result.append(dict(protocol=p,domains="fresh_two" if ns is NEW else "main_six",
                    method=arm,goal=g,cap=cap,n=len(ts),hits=sum(t<=cap for t in ts),
                    mean=float(np.minimum(ts,cap).mean())))
    with (root/"charged_precision.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    print(len(result),"fully charged comparisons")
if __name__=="__main__":main()
