"""Reconstruct every reported first hit solely from archived raw electrical measurements."""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from audit import checks
from century_audit import PROTOCOLS, arms_for, overview, verified
from century_run import CAP, DOMAINS, SEEDS
from experiment import upper


def replay(path):
    with ZipFile(path) as z:
        def read(file):return json.loads(z.read(file))
        raw={n:{(tuple(x),k):v for x,k,v in read("raw/"+n+".json")} for n in DOMAINS}
        fine={(r["circuit"],tuple(r["x"])):r["raw"] for r in read("fine.json")}
        stored=list(csv.DictReader(io.StringIO(z.read("summary.csv").decode())))
        runs=[]
        failed=[]
        count=0
        for protocol,goals in PROTOCOLS.items():
            for n in DOMAINS:
                refx=tuple(np.ones(len(upper(DOMAINS[n]))))
                def pair(x,k):return [raw[n][x,j] for j in (k,k+1)]
                ref=pair(refx,0)+pair(refx,2)
                for arm in arms_for(protocol):
                    for seed in ((0,) if arm=="role_poll" else SEEDS):
                        r=read(f"traces/{protocol}/{n}-{arm}-{seed}.json")
                        assert (r["circuit"],r["arm"],r["seed"])==(n,arm,seed)
                        calls=4;hits={}
                        for x,t,validated in r["trace"]:
                            x=tuple(x)
                            dev=pair(x,0)
                            ok,edp=checks(dev,ref[:2])
                            calls+=2
                            assert validated==bool(ok and edp<=max(goals) and calls+2<=CAP)
                            if validated:
                                allrows=dev+pair(x,2)
                                calls+=2
                                for goal in goals:
                                    if str(goal) not in hits and verified(allrows,ref,goal):
                                        hits[str(goal)]=calls
                                        count+=1
                                        if not verified(fine[n,x],fine[n,refx],goal):
                                            failed.append((protocol,n,arm,seed,goal,calls))
                            assert calls==t
                        assert hits==r["hits"] and calls==r["calls"] and (calls==CAP or len(hits)==len(goals))
                        runs.append(r)
        rows=[row for protocol in PROTOCOLS for row in overview(
            [r for r in runs if r["protocol"]==protocol],protocol)]
        assert len(rows)==len(stored)
        for row,given in zip(rows,stored):
            assert all(str(v)==given[k] for k,v in row.items())
        print(f"{len(runs)} trajectories, {len(rows)} summary rows, {count} first hits, {len(failed)} fine failures")
        return runs,failed


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("archive",type=Path,nargs="?",default=Path("century_measurements.zip"))
    replay(parser.parse_args().archive)
