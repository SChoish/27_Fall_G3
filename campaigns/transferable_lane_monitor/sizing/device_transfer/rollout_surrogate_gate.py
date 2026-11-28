"""One-step rollout policy improvement from paired source returns on measured SPICE tables."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sys
from zipfile import ZipFile,ZIP_LZMA

import numpy as np
from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_run import run
from century_engine import enable
from experiment import arguments,upper
from ppo_sizing import ROOT,train
from surrogate_buffer import merit
from surrogate_buffer_run import models
from surrogate_buffer_transfer import source_pdk
from surrogate_gate import HORIZON,SEEDS,chooser,proposals
from surrogate_gate_run import ALL,NEW,conditions,sources_and_baselines

ARM="rollout_gate"
ARCHIVE=ROOT/"rollout_surrogate_gate_measurements.zip"


def collect(row,net,fold,seed):
    x,y=np.log(row["x"]),np.asarray(row["y"])
    hi=np.log(upper(replace(SOURCE_NETS[row["name"]],pdk=row["pdk"])))
    lo=np.full_like(hi,np.log(.58))
    states=[];gains=[]
    for g in (.95,.90):
        success=merit(y,g)<=0
        cache={}
        def actions(hist):
            if hist not in cache:
                available=[i for i in range(1,len(x)) if i not in hist]
                ids,F=proposals(net,x[list(hist)],y[list(hist)],lo,hi,x[available],g)
                cache[hist]=([available[j] for j in ids],F)
            return cache[hist]
        def value(hist,first,left):
            # Same future reference policy for both first actions: source-guard.
            for step in range(1,left+1):
                i=actions(hist)[0][first if step==1 else 1]
                if success[i]:
                    return 1-step/HORIZON
                hist=(*hist,i)
            return -left/HORIZON
        for rep in range(12):
            rng=np.random.default_rng([seed,fold,rep,41573])
            hist=(0,*rng.choice(np.arange(1,len(x)),4,replace=False).tolist())
            exploration=rng.integers(2,size=9)
            if success[list(hist)].any():
                continue
            for t in range(9):
                ids,F=actions(hist)
                if t in (0,4,8) and ids[0]!=ids[1]:
                    gain=value(hist,1,HORIZON-t)-value(hist,0,HORIZON-t)
                    states.append(F);gains.append(gain)
                i=ids[exploration[t]]
                if success[i]:
                    break
                hist=(*hist,i)
    return states,gains


def fit_rollout(rows):
    folds=[train(rows[:i]+rows[i+1:],only=("anon400",))["anon400"] for i in range(len(rows))]
    out=[]
    for seed in SEEDS:
        x=[];y=[]
        for i,(row,net) in enumerate(zip(rows,folds)):
            F,gains=collect(row,net,i,seed)
            x.extend(F);y.extend(gains)
        X=np.stack(x);Y=np.asarray(y)
        weights=np.linalg.solve(X.T@X+.005*len(Y)*np.eye(X.shape[1]),X.T@Y)
        out.append(dict(weights=weights.tolist(),count=len(Y),fresh_better=int((Y<0).sum()),
                        guard_better=int((Y>0).sum()),ties=int((Y==0).sum())))
    return {ARM:out}


def verify(src,fitted,rows):
    nets=models(src)
    assert len(rows)==sum(len(conditions(n)[0])*len(conditions(n)[1]) for n in ALL)
    stats={};actions={}
    for n,c in ALL.items():
        p=source_pdk(c)
        for g in conditions(n)[1]:
            group=[r for r in rows if r["name"]==n and r["goal"]==g]
            log=[]
            _,stats[f"{n}_{g}"]=audit(src[p],group,(ARM,),{ARM:(chooser(nets[p],fitted[p],g,log),None)},
                                       {n:c},conditions(n)[0],(g,))
            actions[f"{n}_{g}"]=log
    return stats,actions


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as z:
            src,fitted,rows,stats,actions=[json.loads(z.read(k+".json")) for k in ("source","training","runs","audit","actions")]
        for p,records in src.items():
            for a,b in zip(fit_rollout(records)[ARM],fitted[p][ARM],strict=True):
                np.testing.assert_allclose(a["weights"],b["weights"],atol=1e-10,rtol=0)
                assert [a[k] for k in ("count","ties","fresh_better","guard_better")]==[b[k] for k in ("count","ties","fresh_better","guard_better")]
        assert verify(src,fitted,rows)==(stats,actions)
        print(f"Reproduced paired source returns and {len(rows)} target trials.")
        return
    args=arguments();enable()
    src,_=sources_and_baselines();nets=models(src)
    root=args.out/ARM;root.mkdir(parents=True,exist_ok=True)
    fitted={}
    for p,records in src.items():
        path=root/(p+"_training.json")
        if not path.exists():
            path.write_text(json.dumps(fit_rollout(records)))
        fitted[p]=json.loads(path.read_text())
    def evaluate(job):
        n,s=job;c=ALL[n];p=source_pdk(c);rows=[]
        for g in conditions(n)[1]:
            run((n,c,ARM,s,g),args,None,root,chooser(nets[p],fitted[p],g))
            rows.append(json.loads((root/f"{n}-{ARM}-{s}-{g}.json").read_text()))
        return rows
    jobs=[(n,s) for n in (*NEW,*(n for n in ALL if n not in NEW)) for s in conditions(n)[0]]
    with ThreadPoolExecutor(args.workers) as pool:
        rows=[r for group in pool.map(evaluate,jobs) for r in group]
    stats,actions=verify(src,fitted,rows)
    with ZipFile(ARCHIVE,"w",compression=ZIP_LZMA) as z:
        for k,v in (("source",src),("training",fitted),("runs",rows),("audit",stats),("actions",actions)):
            z.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    print(f"Verified {len(rows)} paired-advantage trials.")


if __name__=="__main__":
    main()
