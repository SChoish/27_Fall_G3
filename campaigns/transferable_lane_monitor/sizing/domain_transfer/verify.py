"""Audit replay counts independently and rerun four graph-adaptation trajectories in ngspice."""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from data import campaign, corners, make_deck, DOMAINS, GRID


def live(record,args):
    name=record["domain"]
    pdk,n,load=DOMAINS[name]
    args.model=args.gf if pdk=="gf" else args.sky
    campaign.deck=lambda *parts:make_deck(pdk,*parts,wire=name=="wire")
    cs=corners(pdk,load)
    ref=[campaign.transient(n,[1,1],c,args) for c in cs]
    hits={}
    count=4
    checked=[]
    for idx in record["order"]:
        x=GRID[idx]
        dev=[campaign.transient(n,x,c,args) for c in cs[:2]]
        d=campaign.metrics(dev,ref[:2],cs[:2])
        count+=2
        row=dict(x=x,dev=dev)
        if max(d[1:])<=0 and d[0]<=np.log(.95) and count+2<=40:
            val=[campaign.transient(n,x,c,args) for c in cs[2:]]
            v=campaign.metrics(val,ref[2:],cs[2:])
            count+=2
            row["confirm"]=val
            for g in (.95,.9):
                if max(d[1:])<=0 and max(v[1:])<=0 and max(d[0],v[0])<=np.log(g):
                    hits.setdefault(g,count)
        checked.append(row)
    assert (count,hits.get(.95),hits.get(.9))==(record["charged"],record["hit5"],record["hit10"])
    return dict(domain=name,calls=count,hits=hits,reference=ref,rows=checked)


if __name__=="__main__":
    p=ArgumentParser()
    for key in ("sky","gf","wasm"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--domain",choices=list(DOMAINS))
    args=p.parse_args()
    records=json.loads(Path("replays.json").read_text())["records"]
    chosen=[r for r in records if r["seed"]==11 and r["method"]=="gat_head_adapt"]
    if args.domain:
        chosen=[r for r in chosen if r["domain"]==args.domain]
    result=[live(r,args) for r in chosen]
    Path(f"verify_{args.domain}.json" if args.domain else "verify.json").write_text(json.dumps(result,separators=(",",":")))
    print([(r["domain"],r["calls"],r["hits"]) for r in result])
