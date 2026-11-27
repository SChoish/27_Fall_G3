"""Reconstruct hits and counts directly from physical per-transient measurements."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import defaultdict


def passes(rows,refs):
    if any("error" in r for r in rows):
        return False,None
    ratios=[]
    for pair,(r,ref) in enumerate(zip(rows,refs)):
        v=(1.8,1.66,1.8,1.62)[pair]
        dt=max(r["rise"],r["fall"])
        ratios.append(dt*r["energy_fJ"]/(max(ref["rise"],ref["fall"])*ref["energy_fJ"]))
        if not (r["low"]<=v*.1 and r["high"]>=v*.9 and r["energy_fJ"]>0):
            return False,None
        if any(not 0 < r[k] <= 1.10*ref[k] for k in ("rise","fall","slew_rise","slew_fall")):
            return False,None
    for j in (0,2):
        def droop(data):
            return max(abs(data[j][key]-data[j+1][key]) for key in ("rise","fall"))
        if droop(rows)>1.05*droop(refs):
            return False,None
    return True,max(ratios)


def audit(path):
    data=json.loads(path.read_text())
    method=data["method"] if data["method"]!="target_bo" else path.parent.name
    stages,seed=data["stages"],data["seed"]
    ref=data["rows"][0]
    refs=ref["dev"]+ref["verify"]
    calls=failed=0
    hits={}
    for r in data["rows"]:
        physical=r["dev"]+r.get("verify",[])
        calls+=len(physical)
        failed+=sum("error" in row for row in physical)
        assert calls==r["calls"],path
        if len(physical)==4:
            valid,ratio=passes(physical,refs)
            if valid:
                for threshold in (.95,.90,.80):
                    if ratio<=threshold:
                        hits.setdefault(str(threshold),calls)
    assert hits==data["hits"] and calls==data["physical_calls"],path
    return stages,method,seed,calls,failed,data["source_calls"],hits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    args=p.parse_args()
    paths=sorted(args.root.rglob("s[24]_*.json"))
    records=[audit(path) for path in paths]
    assert len(records)==66
    groups=defaultdict(list)
    for stage,method,seed,calls,failed,pre,hits in records:
        groups[stage,method].append((seed,pre,hits))
    with (args.root/"summary.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["stages","method","target","seeds","successes","restricted_mean_calls",
                    "median_hit_calls","mean_source_calls","per_seed_hits"])
        for (stage,method),rows in sorted(groups.items()):
            assert {r[0] for r in rows}==set(range(11,17))
            for target in (.95,.90,.80):
                times=[r[2].get(str(target)) for r in sorted(rows)]
                reached=sorted(t for t in times if t is not None)
                # Empirical first CDF crossing of 50 percent, inclusive.
                median=reached[2] if len(reached)>=3 else None
                w.writerow([stage,method,target,6,len(reached),
                            sum(t if t is not None else 40 for t in times)/6,
                            median,statistics.mean(r[1] for r in rows),times])
    audit_log=dict(optimizer_runs=len(records),physical_evals=sum(r[3] for r in records),
                   failed_evals=sum(r[4] for r in records),all_calls_and_hits_reconstructed=True)
    (args.root/"audit.json").write_text(json.dumps(audit_log,indent=2)+"\n")
    print(audit_log)


if __name__=="__main__":
    main()
