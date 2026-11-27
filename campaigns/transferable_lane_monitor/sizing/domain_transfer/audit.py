"""Reconstruct all first-hit counts from physical measurements without acquisition code."""
import json
from pathlib import Path
import numpy as np

def eligible(rows, refs, conditions):
    feasible=True
    ratios=[]
    for r, base, (volts,_,_) in zip(rows,refs,conditions):
        feasible &= all(0<r[s]<=1.10*base[s] for s in ("rise","fall","slew_rise","slew_fall"))
        feasible &= r["low"]<=.1*volts and r["high"]>=.9*volts
        ratios.append(r["energy_fJ"]*max(r["rise"],r["fall"])/
                      (base["energy_fJ"]*max(base["rise"],base["fall"])))
    drift=lambda pair:max(abs(pair[1][k]-pair[0][k]) for k in ("rise","fall"))
    feasible &= drift(rows)<=1.05*drift(refs)
    return feasible,max(ratios)

def main():
    data={p.stem:json.loads(p.read_text()) for p in Path("measured").glob("*.json")}
    records=json.loads(Path("replays.json").read_text())["records"]
    for name,d in data.items():
        energies=[row["raw"][0]["energy_fJ"] for row in d["designs"]]
        assert len(set(energies))>=10, f"Parameter sweep has no effect on energy in {name}"
    for r in records:
        d=data[r["domain"]]
        calls=4
        hits={}
        assert len(set(r["order"]))==len(r["order"])
        for j in r["order"]:
            rows=d["designs"][j]["raw"]
            ok,edp=eligible(rows[:2],d["reference"][:2],d["corners"][:2])
            calls+=2
            if ok and edp<=.95 and calls+2<=40:
                calls+=2
                held,second=eligible(rows[2:],d["reference"][2:],d["corners"][2:])
                for g in (.95,.90):
                    if held and max(edp,second)<=g:
                        hits.setdefault(g,calls)
        assert (calls,hits.get(.95),hits.get(.90))==(r["charged"],r["hit5"],r["hit10"])
    live=json.loads(Path("verify.json").read_text())
    if Path("verify_wire.json").exists():
        live=[r for r in live if r["domain"]!="wire"]+json.loads(Path("verify_wire.json").read_text())
    result=dict(replays=len(records),domains=len(data),physical_grid_evals=sum(len(d["reference"])+sum(
        len(v["raw"]) for v in d["designs"]) for d in data.values()),additional_physical_checks=sum(
        r["calls"] for r in live),live_domains=[r["domain"] for r in live],all_hits_match=True)
    Path("audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)

if __name__=="__main__":
    main()
