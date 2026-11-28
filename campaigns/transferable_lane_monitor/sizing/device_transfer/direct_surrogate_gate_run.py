"""SPICE evaluation and independent replay of learned and random direct head selection."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from zipfile import ZipFile,ZIP_LZMA

import numpy as np
from blind_width_audit import audit
from blind_width_run import run
from century_engine import enable
from direct_surrogate_gate import ARMS,chooser_direct,fit_direct
from experiment import arguments
from ppo_sizing import ROOT
from surrogate_buffer_run import models
from surrogate_buffer_transfer import source_pdk
from surrogate_gate_run import ALL,NEW,conditions,sources_and_baselines

ARCHIVE=ROOT/"direct_surrogate_gate_measurements.zip"


def verify(sources,fitted,rows):
    assert len(rows)==sum(len(conditions(n)[0])*len(conditions(n)[1])*len(ARMS) for n in ALL)
    stats,actions={},{}
    nets=models(sources)
    for n,c in ALL.items():
        p=source_pdk(c)
        for g in conditions(n)[1]:
            logs={a:[] for a in ARMS}
            sels={a:(chooser_direct(nets[p],fitted[p],g,logs[a]),None) for a in ARMS}
            group=[r for r in rows if r["name"]==n and r["goal"]==g]
            _,stats[f"{n}_{g}"]=audit(sources[p],group,ARMS,sels,{n:c},conditions(n)[0],(g,))
            actions[f"{n}_{g}"]=logs
    return stats,actions


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as z:
            src,fitted,rows,stats,actions=[json.loads(z.read(k+".json")) for k in ("source","training","runs","audit","actions")]
        for p,records in src.items():
            for new,old in zip(fit_direct(records),fitted[p],strict=True):
                np.testing.assert_allclose(new["weights"],old["weights"],atol=1e-10,rtol=0)
                assert all(new[k]==old[k] for k in ("decisions","warm_hits","policy_hits"))
        assert verify(src,fitted,rows)==(stats,actions)
        print(f"Reproduced direct policies and {len(rows)} SPICE trials.")
        return
    args=arguments();enable()
    src,_=sources_and_baselines()
    nets=models(src)
    root=args.out/"direct_surrogate_gate";root.mkdir(exist_ok=True,parents=True)
    fitted={}
    for p,records in src.items():
        f=root/(p+"_training.json")
        if not f.exists():
            f.write_text(json.dumps(fit_direct(records)))
        fitted[p]=json.loads(f.read_text())
    def evaluate(job):
        n,s=job;c=ALL[n];p=source_pdk(c);rows=[]
        for a in ARMS:
            for g in conditions(n)[1]:
                run((n,c,a,s,g),args,None,root,chooser_direct(nets[p],fitted[p],g))
                rows.append(json.loads((root/f"{n}-{a}-{s}-{g}.json").read_text()))
        return rows
    jobs=[(n,s) for n in (*NEW,*(n for n in ALL if n not in NEW)) for s in conditions(n)[0]]
    with ThreadPoolExecutor(args.workers) as pool:
        rows=[r for group in pool.map(evaluate,jobs) for r in group]
    stats,actions=verify(src,fitted,rows)
    with ZipFile(ARCHIVE,"w",compression=ZIP_LZMA) as z:
        for k,v in (("source",src),("training",fitted),("runs",rows),("audit",stats),("actions",actions)):
            z.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    print(f"Verified {len(rows)} direct head-selection trials.")


if __name__=="__main__":
    main()
