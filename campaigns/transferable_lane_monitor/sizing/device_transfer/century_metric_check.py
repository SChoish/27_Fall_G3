"""Finite differences and source rotation angles for the learned inverse metric."""
import json
from pathlib import Path
import numpy as np
from century_metric import objective,training_pairs
from warm_policy import source_prior


def main():
    root=Path("runs/century")
    data=json.loads((root/"source_directions.json").read_text())
    weights=json.loads((root/"metrics.json").read_text())
    rows=[]
    for seed in (31,32,33):
        net=source_prior(root.parent,seed)
        for arm in ("rank2","shuffled"):
            pairs=training_pairs(data,net,arm=="shuffled")
            B=np.array(weights[f"{arm}-{seed}"]["B"])
            _,grad,_=objective(pairs,B)
            err=0.
            for ij in np.ndindex(B.shape):
                high=B.copy();high[ij]+=1e-5
                low=B.copy();low[ij]-=1e-5
                fd=(objective(pairs,high)[0]-objective(pairs,low)[0])/2e-5
                err=max(err,float(abs(fd-grad[ij])))
            angles=[]
            for _,base,V,q,_ in pairs:
                updated=base-V@B@(B.T@q)
                cosine=updated@base/(np.linalg.norm(updated)*np.linalg.norm(base))
                angles.append(float(np.rad2deg(np.arccos(np.clip(cosine,-1,1)))))
            assert err<1e-7
            rows.append(dict(seed=seed,arm=arm,pair_count=sum(len(r[-1]) for r in pairs),
                             gradient_error=err,source_rotation_degrees=angles))
    (root/"metric_diagnostics.json").write_text(json.dumps(rows))
    print(max(r["gradient_error"] for r in rows))
    for arm in ("rank2","shuffled"):
        angles=[a for r in rows if r["arm"]==arm for a in r["source_rotation_degrees"]]
        print(arm,np.percentile(angles,[0,50,100]))
if __name__=="__main__":main()
