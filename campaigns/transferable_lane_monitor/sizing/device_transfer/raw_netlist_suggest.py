"""Produce one candidate from a supported raw deck, numerical limits and measured observations."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from circuit import ratios
from raw_netlist import BoundNet,extract
from raw_netlist_run import select,train


def suggest(deck_text,pdk,record,seed,model,arm):
    # Device names only align the measurement vectors with netlist instances.
    names=[s.split()[0] for s in deck_text.splitlines()[1:] if s.split() and s.split()[0][0].lower() in ("m","x")]
    if len(names)!=len(set(n.lower() for n in names)):raise ValueError("Duplicate MOS instance name")
    if pdk not in ("sky","gf"):raise ValueError("Expected sky or gf")
    if ("limits" in record)==("common_limits" in record):
        raise ValueError("Supply either common_limits or limits keyed by MOS instance")
    if "common_limits" in record:
        limits=np.array([record["common_limits"] for _ in names],float)
    else:
        if set(record["limits"])!=set(names):raise ValueError("Limits must cover exactly the MOS instances")
        limits=np.array([record["limits"][k] for k in names],float)
    if limits.shape!=(len(names),2) or not names:raise ValueError("Expected one [lower, upper] pair per MOS")
    if not np.all(np.isfinite(limits)) or not np.all((limits[:,0]>0)&(limits[:,0]<=1)&(limits[:,1]>=1)&(limits[:,0]<limits[:,1])):
        raise ValueError("Expected positive lower/upper relative widths enclosing nominal")
    graph=extract(deck_text,pdk,connected=arm=="wires400")
    if not record["observations"]:
        return dict(zip(names,[1.]*len(names)))
    hist=[];y=[]
    for obs in record["observations"]:
        widths=np.array([obs["ratios"][k] for k in names],float)
        if ("dev" in obs)==("log_metrics" in obs):
            raise ValueError("Supply dev transient records or five log_metrics per observation")
        if "dev" in obs:
            metrics=ratios(obs["dev"],record["observations"][0]["dev"])
        else:
            metrics=np.array(obs["log_metrics"],float)
        if not np.all((limits[:,0]<=widths)&(widths<=limits[:,1])) or metrics.shape!=(5,) or not np.isfinite(metrics).all():
            raise ValueError("Observation outside design space or invalid metrics")
        hist.append(np.log(widths));y.append(metrics)
    if np.any(hist[0]!=0):raise ValueError("First observation must be the nominal design")
    perm=np.random.default_rng([seed,886]).permutation(len(names))
    net=BoundNet(model,graph.reorder(perm))
    lo,hi=np.log(limits[perm]).T
    z=select(arm,np.array(hist)[:,perm],y,lo,hi,seed,net)
    return dict(zip(names,np.round(np.exp(z[np.argsort(perm)]),6).tolist()))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("deck",type=Path)
    p.add_argument("--pdk",required=True,choices=("sky","gf","sky130","gf180"))
    p.add_argument("--history",required=True,type=Path)
    p.add_argument("--seed",type=int,default=81)
    p.add_argument("--method",choices=("devices400","dense400","wires400"),default="wires400")
    p.add_argument("--source",type=Path,help="Source measurement JSON or ZIP for the trained encoder")
    a=p.parse_args()
    record=json.loads(a.history.read_text())
    model=None
    if len(record["observations"])>4:
        source=a.source or Path(__file__).with_name("runs")/"raw_netlist"/"source.json"
        if not source.is_file():
            p.error("Source measurements required after warm-up. Pass --source or run raw_netlist_source.py source first.")
        if source.suffix==".zip":
            with ZipFile(source) as z:rows=json.loads(z.read("source.json"))
        else:rows=json.loads(source.read_text())
        model=train(rows,only=(a.method,))[a.method]
    pdk={"sky130":"sky","gf180":"gf"}.get(a.pdk,a.pdk)
    print(json.dumps(suggest(a.deck.read_text(),pdk,record,a.seed,model,a.method),indent=2))
