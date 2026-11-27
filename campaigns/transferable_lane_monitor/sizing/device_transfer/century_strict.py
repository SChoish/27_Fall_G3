"""10%-only first hit. Reuse raw overlaps and spend saved confirmation calls on new candidates."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from time import sleep
from century_engine import enable
from century_metric import InverseMetric
from century_role import search
from century_run import DOMAINS,SEEDS,HEURISTICS,LEARNED,trial
from warm_policy import source_prior
from experiment import arguments
import numpy as np


def main():
    args=arguments()
    enable()
    root=args.out/"century"
    dest=root/"strict"
    dest.mkdir(exist_ok=True)
    models={s:source_prior(args.out,s) for s in SEEDS}
    parameters=json.loads((root/"metrics.json").read_text())
    metrics={k:InverseMetric(np.array(v["B"])) for k,v in parameters.items()}
    def work(file):
        r=json.loads(file.read_text())
        name,arm,seed=r["circuit"],r["arm"],r["seed"]
        # With no 5%-only confirmation this strict-only trial is exactly the joint trial.
        from audit import checks
        saved=any(step["validation"] is not None and checks(step["dev"],r["ref"][:2])[1]>.90 for step in r["trace"])
        if not saved:
            result={**r,"hits":{g:t for g,t in r["hits"].items() if g=="0.9"}}
        elif arm=="role_poll":
            result=search(name,DOMAINS[name],args,goals=(.90,))
        else:
            result=trial(name,DOMAINS[name],arm,seed,args,models.get(seed),metrics.get(f"{arm}-{seed}"),(.90,))
        assert all(a["x"]==b["x"] for a,b in zip(result["trace"],r["trace"]))
        temp=(dest/file.name).with_suffix(".part")
        temp.write_text(json.dumps(result,allow_nan=False))
        os.replace(temp,dest/file.name)
    expected={root/f"{n}-{a}-{s}.json" for n in DOMAINS for a in (*HEURISTICS,*LEARNED,"role_poll")
              for s in ((0,) if a=="role_poll" else SEEDS)}
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            waiting=[f for f in expected if not (dest/f.name).exists()]
            if not waiting:break
            ready=[f for f in sorted(waiting) if f.exists()]
            if ready:list(pool.map(work,ready))
            else:sleep(5)
    print("strict completed",len(expected),flush=True)

if __name__=="__main__":main()
