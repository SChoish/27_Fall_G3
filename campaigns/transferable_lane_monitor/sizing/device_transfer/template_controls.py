"""Replay-only comparison on prior circuits and an untuned random-donor counterfactual."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from blind_width_audit import audit
from blind_width_run import GOALS, run
from century_engine import enable
from corner_transfer import DOMAINS as EARLIER, seeds
from elite_width_transfer import ROOT, DOMAINS, SEEDS, policies
from experiment import arguments

CIRCUITS={**EARLIER,**DOMAINS}
ARCHIVE=ROOT/"template_controls_measurements.zip"


def settings(name):
    return ("retrieved",seeds(name)) if name in EARLIER else ("random_templates",SEEDS)


def verify(source,runs,choose):
    assert len(runs)==30 and len(source)==4 and all(r["pdk"]=="sky" for r in source)
    stats={}
    for n,c in CIRCUITS.items():
        arm,seq=settings(n)
        stats[n]=audit(source,[r for r in runs if r["name"]==n],(arm,),{arm:(choose[n],None)},
                       {n:c},seq,GOALS)[1]
    return stats


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as f:
            source,runs,stored=[json.loads(f.read(k+".json")) for k in ("source","runs","audit")]
        stats=verify(source,runs,policies(source,CIRCUITS));assert stats==stored
        return stats
    args=arguments();enable()
    with ZipFile(ROOT/"anonymous_width_measurements.zip") as f:
        source=[dict(r,pdk="sky") for r in json.loads(f.read("source.json"))]
    choose=policies(source,CIRCUITS)
    root=args.out/"elite_width_transfer";root.mkdir(exist_ok=True,parents=True)
    def pair(job):
        n,s=job;arm,_=settings(n)
        rows=[]
        for g in GOALS:
            run((n,CIRCUITS[n],arm,s,g),args,None,root,choose[n])
            rows.append(json.loads((root/f"{n}-{arm}-{s}-{g}.json").read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs=[r for block in pool.map(pair,[(n,s) for n in CIRCUITS for s in settings(n)[1]]) for r in block]
    stats=verify(source,runs,choose)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as f:
        for k,v in (("source",source),("runs",runs),("audit",stats)):
            f.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    return stats


if __name__=="__main__":print(main())
