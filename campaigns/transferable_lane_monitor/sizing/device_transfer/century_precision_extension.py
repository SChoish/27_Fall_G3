"""Continue numerically marginal first hits to the original 100-call ceiling."""
import json
from pathlib import Path
import numpy as np
from zipfile import ZipFile
from century_audit import verified
from century_engine import enable,simulate
from century_portfolio import compose,NEW
from century_portfolio_check import key
from century_run import trial
from circuit import ratios
from experiment import arguments
from warm_policy import source_prior
from warm_space import Store


def main():
    args=arguments();enable()
    root=args.out/"century";src=root/"portfolio"
    with ZipFile("century_portfolio_measurements.zip") as z:
        records=json.loads(z.read("records.json"))
        fail=json.loads(z.read("audit.json"))["fine_failures"]
        fine=json.loads(z.read("fine.json"))
    initial=set(fine)
    output=[]
    diagnostics=[]
    for n,arm,s,g,t in fail:
        c=NEW[n];model=source_prior(args.out,s)
        args.step_ps=1.
        suffix=trial(n,c,"warm4",s,args,model)
        role=json.loads((src/f"base-{n}-{g}-role.json").read_text())
        store=Store(n,c,args)
        ref=suffix["ref"]
        nominal=tuple(np.ones(len(suffix["trace"][0]["x"])))
        fine_ref=fine[key(n,nominal)]
        def passes(x):
            digest=key(n,x)
            if digest not in fine:
                import copy
                opts=copy.copy(args);opts.step_ps=.25
                fine[digest]=[simulate(c,x,k,opts) for k in range(4)]
            ok=verified(fine[digest],fine_ref,g)
            diagnostics.append(dict(circuit=n,method=arm,seed=s,goal=g,x=x,passes=ok,
                coarse_edp=float(np.exp(max(ratios(store.pair(x,k),ref[k:k+2])[0] for k in (0,2)))),
                fine_edp=float(np.exp(max(ratios(fine[digest][k:k+2],fine_ref[k:k+2])[0] for k in (0,2))))))
            return ok
        result=compose(n,c,s,g,arm,role,suffix,store,precision_pass=passes)
        original=next(r for r in records if (r["circuit"],r["method"],r["seed"],r["goal"])==(n,arm,s,g))
        assert all(list(a["x"])==b["x"] and a["calls"]==b["calls"] for a,b in zip(result["trace"],original["trace"]))
        output.append(result)
    adjusted=[]
    for r in records:
        match=[v for v in output if all(r[k]==v[k] for k in ("circuit","method","seed","goal"))]
        adjusted.append(match[0] if match else r)
    summary=[]
    for nset,names in (("new",set(NEW)),*((n,{n}) for n in NEW)):
        for arm in ("role","warm4","prefix_warm4"):
            for g in (.95,.9):
                ts=[r["hit"] or 101 for r in adjusted if r["circuit"] in names and r["method"]==arm and r["goal"]==g]
                summary.append(dict(domain=nset,method=arm,goal=g,n=len(ts),hits=sum(t<=100 for t in ts),
                                    capped_mean=float(np.minimum(ts,100).mean()),times=ts))
    result=dict(continued=output,diagnostics=diagnostics,fine={k:v for k,v in fine.items() if k not in initial},
                new_fine_calls=4*len(set(fine)-initial),summary=summary)
    (root/"precision_continuation.json").write_text(json.dumps(result,allow_nan=False))
    print([(v["method"],v["calls"],v["hit"]) for v in output],result["new_fine_calls"])
if __name__=="__main__":main()
