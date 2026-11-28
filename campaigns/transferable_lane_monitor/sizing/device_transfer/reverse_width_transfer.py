"""Evaluate frozen donor mapping from GF180 to two new SKY130 circuits."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_policy import propose
from blind_width_run import GOALS, run
from century_engine import enable
from circuit import Circuit
from elite_width_transfer import ROOT, policies
from experiment import arguments

ARCHIVE = ROOT / "reverse_width_transfer_measurements.zip"
SEEDS = (111, 112, 113)
ARMS = ("bo", "mlp200", "retrieved", "random_templates")
DOMAINS = {
    "reverse_split_sky": Circuit("sky", (-1,0,1,1,2,3), (1.,1.7,1.45,2.,2.5,3.4),
                                 (4,5), (.86,1.42), 2.35),
    "reverse_dual_sky": Circuit("sky", (-1,0,1,2,-2,4,5,6,6), (1.,1.4,2.15,3.1,1.1,1.65,2.3,2.8,3.6),
                                (3,7,8), (1.25,.82,1.50), 2.95),
}


def selectors(source):
    assert len(source)==4 and {r["name"] for r in source}==set(SOURCE_NETS)
    assert all(r["pdk"]=="gf" and len(r["x"])==25 for r in source)
    learned = policies(source,DOMAINS,trial_seeds=SEEDS)
    def cold(arm, hist, y, lo, hi, seed, unused):
        return propose("bo",hist,y,lo,hi,seed)
    return {n:{a:(cold if a=="bo" else learned[n], None) for a in ARMS} for n in DOMAINS}


def verify(source,runs,choose):
    assert len(runs)==len(DOMAINS)*len(ARMS)*len(SEEDS)*len(GOALS)
    return {n:audit(source,[r for r in runs if r["name"]==n],ARMS,choose[n],{n:c},SEEDS,GOALS)[1]
            for n,c in DOMAINS.items()}


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as f:
            source,runs,expected=[json.loads(f.read(k+".json")) for k in ("source","runs","audit")]
        stats=verify(source,runs,selectors(source))
        assert stats==expected
        return stats
    args=arguments()
    enable()
    data=json.loads((args.out/"raw_netlist"/"source.json").read_text())
    source=[r for r in data if r["pdk"]=="gf"]
    choose=selectors(source)
    root=args.out/"reverse_transfer"
    root.mkdir(parents=True,exist_ok=True)
    def pair(job):
        n,a,s=job
        rows=[]
        for g in GOALS:
            run((n,DOMAINS[n],a,s,g),args,None,root,choose[n][a][0])
            rows.append(json.loads((root/f"{n}-{a}-{s}-{g}.json").read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs=[r for block in pool.map(pair,[(n,a,s) for s in SEEDS for n in DOMAINS for a in ARMS]) for r in block]
    stats=verify(source,runs,choose)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as f:
        for k,v in (("source",source),("runs",runs),("audit",stats)):
            f.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    return stats


if __name__=="__main__":
    print(main())
