"""Train the surrogate selector, run the matched SPICE trials, and replay the archive."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sys
from zipfile import ZipFile, ZIP_LZMA

import numpy as np
from blind_width_audit import audit
from blind_width_run import GOALS, run
from century_engine import enable
from circuit import Circuit
from experiment import arguments
from ppo_sizing import DOMAINS, ROOT, seeds
from surrogate_buffer import selector
from surrogate_buffer_run import models
from surrogate_buffer_transfer import TARGETS, SEEDS as THREE_SEEDS, source_pdk
from surrogate_gate import EXPERTS, chooser, fit

NEW_LAYOUT = Circuit("sky",(-1,0,0,2,3),(1.,2.15,1.4,1.85,2.9),(1,4),(.91,1.47),2.95)
NEW = {f"gate_asym_{p}":replace(NEW_LAYOUT,pdk=p) for p in ("sky","gf")}
ALL = {**DOMAINS,**TARGETS,**NEW}
ARMS = ("mlp200",*EXPERTS,"coin","myopic_gate","ppo_gate")
ARCHIVE = ROOT/"surrogate_gate_measurements.zip"


def conditions(n):
    if n in NEW:
        return (151,152,153),GOALS
    if n in TARGETS:
        return THREE_SEEDS,(.95,)
    return seeds(n),GOALS


def sources_and_baselines():
    result=[]
    for file in ("surrogate_buffer_measurements.zip","surrogate_buffer_transfer_measurements.zip"):
        with ZipFile(ROOT/file) as f:
            sources=json.loads(f.read("source.json"))
            result.extend(r for r in json.loads(f.read("runs.json")) if r["arm"] in ("mlp200",*EXPERTS))
    return sources,result


def verify(sources,trained,runs):
    nets=models(sources)
    stats,decisions={},{}
    assert len(runs)==sum(len(conditions(n)[0])*len(conditions(n)[1])*len(ARMS) for n in ALL)
    for n,c in ALL.items():
        p=source_pdk(c)
        assert p!=c.pdk
        trial_seeds,goals=conditions(n)
        for g in goals:
            subset=[r for r in runs if r["name"]==n and r["goal"]==g]
            logs={a:[] for a in ARMS}
            sels={a:(selector(nets[p],g) if a=="mlp200" else chooser(nets[p],trained[p],g,logs[a]),None)
                  for a in ARMS}
            _,stats[f"{n}_{g}"]=audit(sources[p],subset,ARMS,sels,{n:c},trial_seeds,(g,))
            decisions[f"{n}_{g}"]=logs
    return stats,decisions


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as z:
            src,trained,rows,stats,actions=[json.loads(z.read(k+".json")) for k in ("source","training","runs","audit","actions")]
        for p,source in src.items():
            rebuilt=fit(source)
            for a,group in rebuilt.items():
                for s,reference in zip(group,trained[p][a],strict=True):
                    np.testing.assert_allclose(s["weights"],reference["weights"],atol=1e-10,rtol=0)
                    assert all(s[k]==reference[k] for k in ("decisions","warm_hits","policy_hits"))
        assert verify(src,trained,rows)==(stats,actions)
        print(f"Reproduced source policies and {len(rows)} SPICE trajectories.")
        return
    sources,archived=sources_and_baselines()
    prep="--controls-only" in sys.argv
    gates_only="--gates-only" in sys.argv
    if gates_only:
        sys.argv.remove("--gates-only")
    if prep:
        sys.argv.remove("--controls-only")
    args=arguments()
    enable()
    root=args.out/"surrogate_gate"
    root.mkdir(parents=True,exist_ok=True)
    trained={}
    for p,rows in sources.items():
        if prep:
            trained[p]={}
        else:
            path=root/(p+"_training.json")
            if not path.exists():
                path.write_text(json.dumps(fit(rows)))
            trained[p]=json.loads(path.read_text())
    nets=models(sources)
    def evaluate(job):
        n,s=job
        c=ALL[n];p=source_pdk(c)
        out=[]
        for a in ARMS:
            if gates_only and not a.endswith("_gate"):
                continue
            if (a in ("mlp200",*EXPERTS) and n not in NEW) or (prep and a.endswith("_gate")):
                continue
            for g in conditions(n)[1]:
                pick=selector(nets[p],g) if a=="mlp200" else chooser(nets[p],trained[p],g)
                run((n,c,a,s,g),args,None,root,pick)
                out.append(json.loads((root/f"{n}-{a}-{s}-{g}.json").read_text()))
        return out
    jobs=[(n,s) for n in (*NEW,*DOMAINS,*TARGETS) for s in conditions(n)[0]]
    with ThreadPoolExecutor(args.workers) as pool:
        rows=archived+[r for group in pool.map(evaluate,jobs) for r in group]
    if prep or gates_only:
        print("Finished selected arms.")
        return
    stats,actions=verify(sources,trained,rows)
    with ZipFile(ARCHIVE,"w",compression=ZIP_LZMA) as z:
        for k,v in (("source",sources),("training",trained),("runs",rows),("actions",actions),("audit",stats)):
            z.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    print(f"Verified {len(rows)} SPICE trajectories.")


if __name__=="__main__":
    main()
