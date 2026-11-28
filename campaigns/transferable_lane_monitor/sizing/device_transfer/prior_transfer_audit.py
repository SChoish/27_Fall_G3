"""Independent trajectory replay for the adaptive anonymous and physical-netlist comparisons."""
import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np

from adaptive_prior_run import ARMS as ADAPT_ARMS,TEST,TEST_SEEDS,policy
from anonymous_network_run import train as anonymous_train
from audit import checks
from blind_width_audit import audit
from blind_width_run import NEW,SEEDS,GOALS
from circuit import deck
from raw_netlist import extract,DeviceNet
from raw_netlist_run import ARMS,SKY_ONLY,bind,select,train
from alarm_pilot import SOURCE_NETS
import uniform_netlist_run as uniform


def validate_graphs(source):
    rng=np.random.default_rng(1901)
    example=source[0];c=replace(SOURCE_NETS[example["name"]],pdk=example["pdk"])
    text,_=deck(c,np.ones(2*len(c.parents)),0)
    graph=extract(text,c.pdk)
    # Renaming nets in each primitive preserves features and the incidence-derived graph.
    translated=[]
    for i,line in enumerate(text.splitlines()):
        words=line.split()
        if i and words and words[0][0].lower() in ("m","x","v","c"):
            last=5 if words[0][0].lower() in ("m","x") else 3
            words[1:last]=["renamed_"+s for s in words[1:last]]
            line=" ".join(words)
        translated.append(line)
    renamed=extract("\n".join(translated),c.pdk)
    np.testing.assert_array_equal(graph.features,renamed.features)
    np.testing.assert_array_equal(graph.mask,renamed.mask)
    z=np.log(example["x"])[1:7];y=np.array(example["y"])[1:7]-example["y"][0]
    net=DeviceNet(True)
    perm=rng.permutation(z.shape[1]);x=net.encode(z,graph)
    gap=float(np.max(abs(x-net.encode(z[:,perm],graph.reorder(perm)))))
    _,grad=net.loss_grad([(z,y,graph)])
    direction={k:rng.normal(size=v.shape) for k,v in net.params.items()}
    eps=1e-6
    for k,p in net.params.items():p+=eps*direction[k]
    hi=net.loss_grad([(z,y,graph)])[0]
    for k,p in net.params.items():p-=2*eps*direction[k]
    lo=net.loss_grad([(z,y,graph)])[0]
    diff=abs((hi-lo)/(2*eps)-sum(np.sum(direction[k]*g) for k,g in grad.items()))
    assert gap<1e-10 and diff<1e-7
    return dict(permutation_error=gap,loss_gradient_error=float(diff))


def verify(group,runs,source):
    for r in source:
        assert not any("error" in v for pair in r["raw"] for v in pair)
    for r in runs:
        for k in (0,2):assert checks(r["ref"][k:k+2],r["ref"][k:k+2])[0]
    stats={}
    if group=="adaptive":
        nets=anonymous_train(source)
        for split,domains,seeds in (("development",NEW,SEEDS),("transfer",TEST,TEST_SEEDS)):
            subset=[r for r in runs if r["name"] in domains]
            selectors={a:(policy,nets) for a in ADAPT_ARMS}
            _,stats[split]=audit(source,subset,ADAPT_ARMS,selectors,domains,seeds)
    elif group=="uniform":
        stats["graph"]=validate_graphs(source)
        nets=train(source,only=uniform.ARMS[1:])
        for name,c in TEST.items():
            bound={(a,s):bind(a,nets[a],c,s) for a in uniform.ARMS[1:] for s in uniform.SEEDS}
            def choose(arm,hist,y,lo,hi,seed,unused):
                return uniform.choose(arm,hist,y,lo,hi,seed,bound.get((arm,seed)))
            subset=[r for r in runs if r["name"]==name]
            selectors={a:(choose,None) for a in uniform.ARMS}
            _,stats[name]=audit(source,subset,uniform.ARMS,selectors,{name:c},uniform.SEEDS,uniform.GOALS,uniform.uniform_upper)
        for r in source:
            assert np.all((np.asarray(r["x"])>=.58)&(np.asarray(r["x"])<=1.3))
            mate=next(other for other in source if other["name"]==r["name"] and other["pdk"]!=r["pdk"])
            np.testing.assert_array_equal(r["x"],mate["x"])
    else:
        stats["graph"]=validate_graphs(source)
        nets=train(source)
        for name,c in TEST.items():
            arms=(*ARMS,SKY_ONLY) if c.pdk=="gf" else ARMS
            bound={(a,s):bind(a,nets[a],c,s) for a in arms for s in TEST_SEEDS}
            def choose(arm,hist,y,lo,hi,seed,unused):
                return select(arm,hist,y,lo,hi,seed,bound[arm,seed])
            selectors={a:(choose,None) for a in arms}
            subset=[r for r in runs if r["name"]==name]
            _,stats[name]=audit(source,subset,arms,selectors,{name:c},TEST_SEEDS)
    return stats


def summarize(runs):
    rows=[]
    for r in runs:
        assert r["calls"]==100 or r["hit"] is not None
    for split,names in (("previous",NEW),("transfer",TEST),("lane",[k for k in TEST if "lane" in k]),
                        ("branch",[k for k in TEST if "branch" in k]),("gf_transfer",[k for k in TEST if k.endswith("_gf")]),
                        *((n,[n]) for n in [*NEW,*TEST])):
        grouped=defaultdict(list)
        for r in runs:
            if r["name"] in names:grouped[r["arm"],r["goal"]].append(r["hit"])
        for (a,g),times in sorted(grouped.items()):
            for cap in (40,100):
                rows.append(dict(split=split,arm=a,goal=g,cap=cap,n=len(times),
                                 hits=sum(t is not None and t<=cap for t in times),
                                 mean=float(np.mean([min(t or cap,cap) for t in times]))))
    return rows


def collect(group,path):
    if group=="adaptive":
        source=json.loads((path/"anonymous_width"/"source.json").read_text())
        runs=[json.loads(f.read_text()) for f in sorted((path/"adaptive_prior").glob("*-*.json"))]
        for a,d in (("bo","anonymous_width"),("mlp_bo","anonymous_network"),("gat_bo","anonymous_network")):
            runs.extend(json.loads(f.read_text()) for f in sorted((path/d).glob(f"long100*-{a}-*.json")))
    else:
        location=path/("uniform_netlist" if group=="uniform" else "raw_netlist")
        source=json.loads((location/"source.json").read_text())
        runs=[json.loads(f.read_text()) for f in sorted(location.glob("adapt_*-*.json"))]
    return source,runs


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("group",choices=["adaptive","physical","uniform"])
    parser.add_argument("--replay",action="store_true")
    opt=parser.parse_args();group=opt.group
    archive=Path(f"{group}_transfer_measurements.zip")
    root=Path("runs")
    if opt.replay:
        with ZipFile(archive) as z:
            data=lambda k:json.loads(z.read(k+".json"))
            runs=data("runs");source=data("source")
            stats=verify(group,runs,source)
            assert summarize(runs)==data("summary")
            for k,v in stats.items():
                if k!="graph":assert v==data("audit")[k]
    else:
        source,runs=collect(group,root)
        stats=verify(group,runs,source);rows=summarize(runs)
        with ZipFile(archive,"w",compression=ZIP_DEFLATED,compresslevel=8) as z:
            for k,v in (("source",source),("runs",runs),("summary",rows),("audit",stats)):
                z.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
        out=root/{"adaptive":"adaptive_prior","physical":"raw_netlist","uniform":"uniform_netlist"}[group]
        with (out/"summary.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(group,len(runs),stats,flush=True)
