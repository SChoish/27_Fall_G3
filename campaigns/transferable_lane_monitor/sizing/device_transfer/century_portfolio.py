"""A fixed electrical prefix with a local-warmup critic fallback. Separate new-netlist test."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import numpy as np
from audit import checks
from circuit import Circuit, ratios
from experiment import arguments,upper
from century_run import DOMAINS,SEEDS,trial
from century_role import search
from century_engine import enable,simulate
from warm_policy import source_prior
from warm_space import Store

CHAIN=Circuit("sky",(-1,0,1,2,3,4,-2,6),(1,1.55,2.1,2.7,3.3,3.8,1.05,2.55),(5,7),(1.32,.78),2.15)
NEW={"long100_sky":CHAIN,"long100_gf":replace(CHAIN,pdk="gf")}
NEW_SEEDS=(51,52,53)


def compose(name,c,seed,goal,mode,role,warm,store):
    ref=role["ref"]
    if warm is not None:assert ref==warm["ref"]
    if mode=="role":sequence=role["trace"]
    elif mode=="warm4":sequence=warm["trace"]
    else:sequence=[*role["trace"][:2],*warm["trace"]]
    calls=4;trace=[];hit=None
    seen=set()
    for step in sequence:
        if calls+2>100:break
        x=tuple(step["x"])
        # Skip duplicated measurements only in the composed policy. Keep both original controls intact.
        if mode=="prefix_warm4" and x in seen:continue
        seen.add(x)
        dev=step["dev"]
        calls+=2
        ok,e=checks(dev,ref[:2])
        val=None
        if ok and e<=goal and calls+2<=100:
            val=step["validation"] or store.pair(x,2)
            calls+=2
            ok,e2=checks(val,ref[2:])
            if ok and e2<=goal:hit=calls
        trace.append(dict(x=x,calls=calls,dev=dev,validation=val))
        if hit is not None:break
    if hit is None and calls<100:
        raise RuntimeError(("insufficient prefix/suffix observations",name,seed,mode,goal,calls))
    return dict(circuit=name,method=mode,seed=seed,goal=goal,calls=calls,hit=hit,ref=ref,trace=trace)


def main():
    args=arguments()
    enable()
    root=args.out/"century"
    out=root/"portfolio"
    out.mkdir(exist_ok=True)
    names=NEW if args.mode=="source" else DOMAINS
    seeds=NEW_SEEDS if args.mode=="source" else SEEDS
    models={s:source_prior(args.out,s) for s in seeds} if names is NEW else {}
    def get(file,make):
        if not file.exists():
            rec=make()
            temp=file.with_suffix(".part")
            temp.write_text(json.dumps(rec,allow_nan=False))
            temp.replace(file)
        return json.loads(file.read_text())
    # One job per circuit and target. A single deterministic role trajectory is shared across seeds.
    def work(job):
        n,c,g=job
        folder=root if g==.95 else root/"strict"
        if n in NEW:
            role=get(out/f"base-{n}-{g}-role.json",lambda:search(n,c,args,goals=(g,)))
        else:
            role=json.loads((folder/f"{n}-role_poll-0.json").read_text())
        store=Store(n,c,args)
        get(out/f"run-{n}-{g}-role-0.json",lambda:compose(n,c,0,g,"role",role,None,store))
        for seed in seeds:
            if n in NEW:
                warm=get(out/f"base-{n}-{g}-warm4-{seed}.json",
                         lambda:trial(n,c,"warm4",seed,args,models[seed],goals=(g,)))
            else:
                warm=json.loads((folder/f"{n}-warm4-{seed}.json").read_text())
            for mode in ("warm4","prefix_warm4"):
                get(out/f"run-{n}-{g}-{mode}-{seed}.json",
                    lambda:compose(n,c,seed,g,mode,role,warm,store))
        print(n,g,flush=True)
    jobs=[(n,c,g) for n,c in names.items() for g in (.95,.90)]
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(work,jobs))

if __name__=="__main__":main()
