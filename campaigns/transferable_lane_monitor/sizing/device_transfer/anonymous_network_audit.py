"""Verify gradients, permutation invariance, raw records and paid first hits."""
import csv
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from anonymous_network_run import ARMS,train,select
from anonymous_network import WidthNet
from blind_width_audit import audit


def verify(source,runs):
    examples=[(np.log(r["x"])[1:6],np.asarray(r["y"])[1:6]-r["y"][0]) for r in source]
    nets=train(source)
    errors={}
    for name,net in nets.items():
        dummy=WidthNet(name=="gat")
        _,grads=dummy.loss_grad(examples)
        rng=np.random.default_rng(33)
        direction={k:rng.normal(size=v.shape) for k,v in dummy.params.items()}
        eps=1e-6
        for k,val in dummy.params.items():val+=eps*direction[k]
        hi=dummy.loss_grad(examples)[0]
        for k,val in dummy.params.items():val-=2*eps*direction[k]
        lo=dummy.loss_grad(examples)[0]
        fd=abs((hi-lo)/(2*eps)-sum(np.sum(v*direction[k]) for k,v in grads.items()))
        perm=max(float(abs(net.readout(net.encode(z[:,::-1]))-net.readout(net.encode(z))).max()) for z,y in examples)
        assert fd<1e-7 and perm<1e-10
        errors[name]=dict(gradient_error=fd,permutation_error=perm,parameters=sum(x.size for x in net.params.values()))
    summary,stats=audit(source,runs,ARMS,{arm:(select,nets[arm.split("_")[0]]) for arm in ARMS})
    return summary,{**stats,"encoder_checks":errors}


if __name__=="__main__":
    root=Path("runs/anonymous_network")
    archive=Path("anonymous_network_measurements.zip")
    if sys.argv[1:]==["replay"]:
        with ZipFile(archive) as z:
            read=lambda k:json.loads(z.read(k+".json"))
            summary,stats=verify(read("source"),read("runs"))
            assert summary==read("summary")
            for k,v in stats.items():
                if k!="encoder_checks":assert v==read("audit")[k]
    else:
        source=json.loads(Path("runs/anonymous_width/source.json").read_text())
        runs=[json.loads(p.read_text()) for p in sorted(root.glob("long100*.json"))]
        summary,stats=verify(source,runs)
        with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=8) as z:
            for k,v in (("source",source),("runs",runs),("summary",summary),("audit",stats)):
                z.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
        with (root/"summary.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    print(stats)
