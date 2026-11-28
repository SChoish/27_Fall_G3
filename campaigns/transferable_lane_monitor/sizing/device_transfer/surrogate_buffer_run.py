"""Cross-PDK first-hit experiment and independent proposal/measurement replay."""
import json
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile, ZIP_DEFLATED
import sys
from blind_width_audit import audit
from blind_width_run import GOALS, run
from century_engine import enable
from experiment import arguments
from ppo_sizing import DOMAINS, ROOT, seeds, source_for, train
from surrogate_buffer import ARMS, selector

ARCHIVE = ROOT / "surrogate_buffer_measurements.zip"


def models(sources):
    return {p:train(rows,only=("anon400",))["anon400"] for p,rows in sources.items()}


def verify(sources,runs):
    nets = models(sources)
    assert len(runs)==len(ARMS)*24
    stats, decisions = {}, {}
    for name,circuit in DOMAINS.items():
        p = source_for(name)
        assert p != circuit.pdk
        for goal in GOALS:
            group = [r for r in runs if r["name"]==name and r["goal"]==goal]
            logs = {a:[] for a in ARMS}
            sels = {a:(selector(nets[p],goal,logs[a]),None) for a in ARMS}
            _, stats[f"{name}_{goal}"] = audit(sources[p],group,ARMS,sels,{name:circuit},seeds(name),(goal,))
            for arm, entries in logs.items():
                decisions[f"{name}_{goal}_{arm}"] = entries
    return stats, decisions


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as z:
            sources,runs,stats,decisions=[json.loads(z.read(k+".json")) for k in ("source","runs","audit","decisions")]
        assert verify(sources,runs)==(stats,decisions)
        print(f"Replayed {len(runs)} runs including all selected head ages.")
        return
    guard_only="--guard-only" in sys.argv
    if guard_only:
        sys.argv.remove("--guard-only")
    args=arguments()
    enable()
    with ZipFile(ROOT/"ppo_sizing_measurements.zip") as z:
        sources=json.loads(z.read("source.json"))
        baselines=[r for r in json.loads(z.read("runs.json")) if r["arm"]=="mlp200"]
    nets=models(sources)
    root=args.out/"surrogate_buffer"
    root.mkdir(exist_ok=True,parents=True)

    def evaluate(job):
        name,seed=job
        local=[]
        for arm in (("source_guard",) if guard_only else ARMS[1:]):
            for goal in GOALS:
                run((name,DOMAINS[name],arm,seed,goal),args,None,root,selector(nets[source_for(name)],goal))
                local.append(json.loads((root/f"{name}-{arm}-{seed}-{goal}.json").read_text()))
        return local

    with ThreadPoolExecutor(args.workers) as pool:
        runs=baselines+[r for block in pool.map(evaluate,[(n,s) for n in DOMAINS for s in seeds(n)]) for r in block]
    if guard_only:
        return
    stats, decisions=verify(sources,runs)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as z:
        for key,data in (("source",sources),("runs",runs),("audit",stats),("decisions",decisions)):
            z.writestr(key+".json",json.dumps(data,separators=(",",":"),allow_nan=False))
    print(f"Verified {len(runs)} runs.")


if __name__=="__main__":
    main()
