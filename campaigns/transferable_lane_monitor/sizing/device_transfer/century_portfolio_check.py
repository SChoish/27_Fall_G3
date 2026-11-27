"""Validate and package the two-step portfolio, including the locked PDK twins."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import csv
import json
from pathlib import Path
import sys
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from audit import checks
from century_audit import verified
from century_portfolio import NEW,NEW_SEEDS
from century_run import DOMAINS,SEEDS
from century_engine import simulate
from experiment import arguments,upper

ALL={**DOMAINS,**NEW}
MODES=("role","warm4","prefix_warm4")
def key(n,x):return sha256(json.dumps([n,x]).encode()).hexdigest()


def verify(records,fine):
    pts=set();fail=[];hit_count=0
    expected={(n,m,s,g) for n in ALL for m in MODES for s in
              ((0,) if m=="role" else (NEW_SEEDS if n in NEW else SEEDS)) for g in (.95,.90)}
    assert expected=={(r["circuit"],r["method"],r["seed"],r["goal"]) for r in records}
    assert len(expected)==len(records)
    for r in records:
        calls=4;hit=None
        n=r["circuit"];g=r["goal"]
        seen=set()
        for step in r["trace"]:
            assert hit is None
            x=tuple(step["x"]);dev=step["dev"]
            assert np.all((np.array(x)>=.58)&(x<=upper(ALL[n])))
            if r["method"]=="prefix_warm4":assert x not in seen
            seen.add(x)
            ok,e=checks(dev,r["ref"][:2]);calls+=2
            assert (step["validation"] is not None)==bool(ok and e<=g and calls+2<=100)
            if step["validation"] is not None:
                calls+=2
                if verified(dev+step["validation"],r["ref"],g):
                    hit=calls;pts.add((n,x))
                    if not verified(fine[key(n,x)],fine[key(n,tuple(np.ones(len(x))))],g):
                        fail.append([n,r["method"],r["seed"],g,hit])
                    hit_count+=1
            assert calls==step["calls"] and calls<=100
        assert r["hit"]==hit and calls==r["calls"] and (hit is not None or calls==100)
    summary=[]
    for domain,names in (("previous",DOMAINS),("new",NEW),*((n,[n]) for n in ALL)):
        for mode in MODES:
            for goal in (.95,.90):
                times=[r["hit"] or 101 for r in records if r["circuit"] in names and r["method"]==mode and r["goal"]==goal]
                for cap in (40,100):
                    summary.append(dict(domain=domain,method=mode,goal=goal,cap=cap,n=len(times),
                                        successes=sum(t<=cap for t in times),mean=float(np.mean(np.minimum(times,cap))),
                                        times=[t if t<=cap else None for t in times]))
    return dict(count=len(records),hits=hit_count,unique_witnesses=len(pts),fine_failures=fail),summary


def main():
    archive=Path("century_portfolio_measurements.zip")
    if len(sys.argv)==2 and sys.argv[1]=="replay":
        with ZipFile(archive) as z:
            audit,summary=verify(json.loads(z.read("records.json")),json.loads(z.read("fine.json")))
            assert audit==json.loads(z.read("audit.json")) and summary==json.loads(z.read("summary.json"))
            print(audit)
        return
    args=arguments();args.step_ps=.25
    root=args.out/"century";src=root/"portfolio"
    records=[json.loads(f.read_text()) for f in sorted(src.glob("run-*.json"))]
    points={(r["circuit"],tuple(r["trace"][-1]["x"])) for r in records if r["hit"] is not None}
    points|={(n,tuple(np.ones(len(x)))) for n,x in points}
    folder=src/"fine"
    folder.mkdir(exist_ok=True)
    def get(point):
        n,x=point;digest=key(n,x);file=folder/(digest+".json")
        if not file.exists():
            old=root/"fine"/file.name
            val=json.loads(old.read_text())["raw"] if old.exists() else [simulate(ALL[n],x,k,args) for k in range(4)]
            file.write_text(json.dumps(val))
        return digest,json.loads(file.read_text())
    with ThreadPoolExecutor(args.workers) as pool:fine=dict(pool.map(get,sorted(points)))
    audit,summary=verify(records,fine)
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=8) as z:
        for name,v in (("records",records),("fine",fine),("summary",summary),("audit",audit)):
            z.writestr(name+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    with (root/"portfolio_summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    (root/"portfolio_audit.json").write_text(json.dumps(audit,indent=2))
    print(audit)

if __name__=="__main__":main()
