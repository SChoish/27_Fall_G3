"""Audit costs from corner records, remeasure first hits and archive a compact raw ledger."""
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from audit import checks
from circuit import simulate
from experiment import arguments, upper
from century_run import CAP, DOMAINS, SEEDS, HEURISTICS, LEARNED
from warm_space import TEST

GOALS=(.95,.90)
ARMS=(*HEURISTICS,*LEARNED)


def verified(record,reference,goal):
    p=[checks(record[k:k+2],reference[k:k+2]) for k in (0,2)]
    return all(ok for ok,e in p) and max(e for ok,e in p)<=goal


def overview(runs):
    rows=[]
    for arm in ARMS:
        for domain,names in (("all",DOMAINS),("previous",TEST),("fresh",set(DOMAINS)-set(TEST)),
                             *((n,[n]) for n in DOMAINS)):
            trials=[r for r in runs if r["arm"]==arm and r["circuit"] in names]
            for goal in GOALS:
                times=[r["hits"].get(str(goal),CAP+1) for r in trials]
                for limit in (40,100):
                    rows.append(dict(arm=arm,domain=domain,goal=goal,limit=limit,n=len(times),
                                     hits=sum(t<=limit for t in times),
                                     mean_calls=float(np.mean(np.minimum(times,limit))),
                                     times=" ".join(str(t) if t<=limit else "-" for t in times)))
    return rows


def main():
    args=arguments()
    root=args.out/"century"
    runs=[]
    keys=set()
    data={n:{} for n in DOMAINS}
    def register(n,x,k,raw):
        key=(tuple(x),k)
        prior=data[n].setdefault(key,raw)
        assert prior==raw
    for n,c in DOMAINS.items():
        for arm in ARMS:
            for seed in SEEDS:
                r=json.loads((root/f"{n}-{arm}-{seed}.json").read_text())
                assert (r["circuit"],r["arm"],r["seed"])==(n,arm,seed)
                x0=tuple(np.ones(len(upper(c))))
                for k,raw in enumerate(r["ref"]):register(n,x0,k,raw)
                calls=4
                hits={}
                for i,step in enumerate(r["trace"]):
                    x=step["x"]
                    assert len(x)==len(x0) and np.all((np.array(x)>=.58)&(np.array(x)<=upper(c)))
                    if i==0 and arm!="warm4":
                        np.testing.assert_array_equal(x,np.repeat([.84 if j in c.outputs else 1.
                                                                    for j in range(len(c.parents))],2))
                    for k,raw in enumerate(step["dev"]):register(n,x,k,raw)
                    calls+=2
                    ok,e=checks(step["dev"],r["ref"][:2])
                    assert (step["validation"] is not None)==bool(ok and e<=.95 and calls+2<=CAP)
                    if step["validation"] is not None:
                        for k,raw in enumerate(step["validation"],2):register(n,x,k,raw)
                        calls+=2
                        for goal in GOALS:
                            if verified(step["dev"]+step["validation"],r["ref"],goal):
                                hits.setdefault(str(goal),calls)
                    assert calls==step["calls"] and calls<=CAP
                assert hits==r["hits"] and calls==r["calls"] and (calls==CAP or len(hits)==2)
                if n in TEST and arm=="warm4":
                    old_file=args.out/"warm"/f"continuous-{n}-nonzero_warm-{seed}.json"
                    if old_file.exists():
                        past=json.loads(old_file.read_text())
                        assert {g:t for g,t in hits.items() if t<=40}==past["hits"]
                        for a,b in zip(r["trace"],past["trace"]):
                            assert a["x"]==b["x"]
                for t in hits.values():
                    x=next(st["x"] for st in r["trace"] if st["calls"]==t)
                    keys.add((n,tuple(x)))
                runs.append(r)
    fine_dir=root/"fine"
    fine_dir.mkdir(exist_ok=True)
    def fine(point):
        n,x=point
        file=fine_dir/(hashlib.sha256(json.dumps([n,x]).encode()).hexdigest()+".json")
        if not file.exists():
            old=args.out/"warm"/"fine"/file.name
            raw=json.loads(old.read_text())["raw"] if old.exists() else [simulate(DOMAINS[n],x,k,args) for k in range(4)]
            file.write_text(json.dumps(dict(circuit=n,x=x,raw=raw,reused=old.exists())))
        return json.loads(file.read_text())
    args.step_ps=.25
    points=sorted(keys|{(n,tuple(np.ones(len(upper(DOMAINS[n]))))) for n,x in keys})
    with ThreadPoolExecutor(args.workers) as workers:
        fine_data=dict(zip(points,workers.map(fine,points)))
    fails=[]
    hit_count=0
    for r in runs:
        for g,t in r["hits"].items():
            n=r["circuit"]
            x=next(st["x"] for st in r["trace"] if st["calls"]==t)
            ref=fine_data[n,tuple(np.ones(len(x)))]["raw"]
            if not verified(fine_data[n,tuple(x)]["raw"],ref,float(g)):
                fails.append([n,r["arm"],r["seed"],g,t])
            hit_count+=1
    overview_rows=overview(runs)
    with (root/"summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=overview_rows[0]);w.writeheader();w.writerows(overview_rows)
    def compact(o):return json.dumps(o,separators=(",",":"),allow_nan=False)
    archive=Path("century_measurements.zip")
    with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=7) as z:
        for name,table in data.items():
            z.writestr("raw/"+name+".json",compact([[x,k,raw] for (x,k),raw in table.items()]))
        for r in runs:
            z.writestr("traces/"+r["circuit"]+"-"+r["arm"]+"-"+str(r["seed"])+".json",
                       compact({**{k:r[k] for k in ("circuit","arm","seed","calls","hits")},
                                "trace":[[s["x"],s["calls"],s["validation"] is not None] for s in r["trace"]]}))
        z.writestr("fine.json",compact(list(fine_data.values())))
        for name in ("source_directions.json","metrics.json","summary.csv"):
            z.write(root/name,name)
    stats=dict(runs=len(runs),charged_target=sum(r["calls"] for r in runs),
               distinct_target_corner_rows=sum(len(t) for t in data.values()),
               target_errors=sum("error" in row for t in data.values() for row in t.values()),
               hits=hit_count,distinct_first_hit_designs=len(keys),precision_checks=4*len(points),
               precision_reused=4*sum(row["reused"] for row in fine_data.values()),precision_failures=fails)
    (root/"audit.json").write_text(json.dumps(stats,indent=2))
    print(stats,flush=True)

if __name__=="__main__":
    main()
