"""Prospective paired test of the electrical-MAE head rule on a new three-output circuit."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from zipfile import ZipFile, ZIP_DEFLATED

from blind_width_audit import audit
from blind_width_run import run
from century_engine import enable
from circuit import Circuit
from experiment import arguments
from ppo_sizing import ROOT
from surrogate_buffer import selector
from surrogate_buffer_run import models

TOPOLOGY = Circuit("sky",(-1,0,0,0),(1.,1.55,2.05,2.65),(1,2,3),(.72,1.07,1.41),2.75)
TARGETS = {f"surrogate_three_{p}":replace(TOPOLOGY,pdk=p) for p in ("sky","gf")}
PLANNED = ("mlp200","fresh","mae")
ARMS = (*PLANNED,"source_guard")
SEEDS = (131,132,133)
GOAL = .95
ARCHIVE = ROOT/"surrogate_buffer_transfer_measurements.zip"


def source_pdk(circuit):
    return "gf" if circuit.pdk=="sky" else "sky"


def verify(sources,runs):
    nets=models(sources)
    assert len(runs)==len(TARGETS)*len(SEEDS)*len(ARMS)
    reports={}
    for n,c in TARGETS.items():
        p=source_pdk(c)
        group=[r for r in runs if r["name"]==n]
        sels={a:(selector(nets[p],GOAL),None) for a in ARMS}
        reports[n]=audit(sources[p],group,ARMS,sels,{n:c},SEEDS,(GOAL,))[1]
    return reports


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as z:
            sources,runs,stats=[json.loads(z.read(k+".json")) for k in ("source","runs","audit")]
        assert verify(sources,runs)==stats
        print("Replayed 18 preselected runs and 6 exploratory source-fallback runs.")
        return
    args=arguments()
    enable()
    root=args.out/"surrogate_buffer_transfer"
    root.mkdir(parents=True,exist_ok=True)
    with ZipFile(ROOT/"ppo_sizing_measurements.zip") as z:
        sources=json.loads(z.read("source.json"))
    nets=models(sources)
    def evaluate(job):
        n,s=job
        c=TARGETS[n]
        for a in ARMS:
            run((n,c,a,s,GOAL),args,None,root,selector(nets[source_pdk(c)],GOAL))
        return [json.loads((root/f"{n}-{a}-{s}-{GOAL}.json").read_text()) for a in ARMS]
    with ThreadPoolExecutor(args.workers) as pool:
        runs=[r for group in pool.map(evaluate,[(n,s) for n in TARGETS for s in SEEDS]) for r in group]
    stats=verify(sources,runs)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for k,d in (("source",sources),("runs",runs),("audit",stats)):
            z.writestr(k+".json",json.dumps(d,separators=(",",":"),allow_nan=False))
    print(stats)


if __name__=="__main__":
    main()
