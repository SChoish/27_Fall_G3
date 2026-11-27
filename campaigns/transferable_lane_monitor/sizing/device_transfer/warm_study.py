"""Development-only configuration selection, followed by locked paired holdout runs."""
from dataclasses import asdict,replace
from pathlib import Path
from time import sleep
import json
import numpy as np
from alarm_pilot import trial
from audit import checks
from circuit import ratios
from experiment import arguments
from warm_space import DEV,TEST,Store
from warm_policy import Config,Space,rollout,source_prior

def write(path,value):path.write_text(json.dumps(value,allow_nan=False))
def ident(cfg):return json.dumps(asdict(cfg),sort_keys=True)

class Table:
    def __init__(self,path):
        data=json.loads(path.read_text());r=data["reference"]
        self.nominal_label=ratios(r[:2],r[:2]);self.dev=[];self.val=[]
        for row in data["records"]:
            pair=row["dev"]
            self.dev.append((ratios(pair,r[:2]),*checks(pair,r[:2])))
            self.val.append(None if row["val"] is None else checks(row["val"],r[2:]))
    def development(self,idx):return self.dev[idx]
    def confirmation(self,idx):
        result=self.val[idx]
        if result is None:raise AssertionError("unqualified target verification")
        return result

class Benchmark:
    def __init__(self,args,domains,seeds):
        self.root=args.out/"warm";self.seeds=seeds
        self.models={s:source_prior(args.out,s) for s in seeds}
        self.domains={n:(Space(c),Table(self.root/(n+".json"))) for n,c in domains.items()}
        self.runs={}
    def evaluate(self,cfg):
        key=ident(cfg)
        if key not in self.runs:
            self.runs[key]=[dict(circuit=n,**rollout(space,table,self.models[s],cfg,s))
                for n,(space,table) in self.domains.items() for s in self.seeds]
        return self.runs[key]
    def score(self,cfg):
        rs=self.evaluate(cfg)
        return tuple(sum(r["hits"].get(g,40) for r in rs)/len(rs) for g in ("0.95","0.9"))
    def best(self,configs):
        # On equal measured cost, prefer fewer warm samples, then no/smaller buffer.
        return min(configs,key=lambda c:(*self.score(c),c.warm,c.cap,c.reuse,ident(c)))

def tune(args):
    from itertools import product
    b=Benchmark(args,DEV,(21,22,23))
    base=Config()
    geometries=[replace(base,unit=u,residual=r) for u,r in product((False,True),repeat=2)]
    geom=b.best(geometries)
    warms=[geom]+[replace(geom,warm=k,region=g,first=f) for k,g,f in product((1,2,4),("local","global"),("role","random"))]
    warm=b.best(warms);forced_warm=b.best(warms[1:])
    def buffers(parent):
        return [parent]+[replace(parent,cap=c,reuse=p,select=r) for c,p,r in product((2,4,8),(.25,.5,1.),("random","score","guard"))]
    opts=buffers(geom)
    buffer=b.best(opts);forced_buffer=b.best(opts[1:])
    # Use all buffer settings with the best nonzero warm-up, including negative interactions.
    combo=buffers(forced_warm)
    final=b.best([*opts,*combo])
    arms={"fixed_bank_base":base,"geometry":geom,"warm":warm,"nonzero_warm":forced_warm,
          "buffer":buffer,"nonzero_buffer":forced_buffer,"combined":final,
          "opposite_coordinate":replace(final,unit=not final.unit),
          "no_correction":replace(final,residual=False),"no_buffer":replace(final,cap=0,reuse=0.,select="score"),
          "no_warm":replace(final,warm=0,first="role",region="local")}
    # Matched single-factor controls for exactly the same warm points and buffer gates.
    arms["opposite_order"]=replace(forced_warm,first="random" if forced_warm.first=="role" else "role")
    arms["same_buffer_random"]=replace(forced_buffer,select="random")
    arms["same_buffer_scored"]=replace(forced_buffer,select="score")
    arms["same_buffer_guarded"]=replace(forced_buffer,select="guard")
    for k in (0,1,2,4):arms[f"warm_count_{k}"]=replace(forced_warm,warm=k)
    # Lock the 10%-target candidate separately from the primary 5% recommendation.
    pool=[*opts,*combo,*warms,*geometries]
    arms["strict_goal"]=min(pool,key=lambda c:(b.score(c)[1],b.score(c)[0],c.warm,c.cap,c.reuse,ident(c)))
    write(b.root/"development_runs.json",b.runs)
    write(b.root/"lock.json",dict(arms={n:asdict(c) for n,c in arms.items()},
          scores={n:b.score(c) for n,c in arms.items()},selection_domains=list(DEV),seeds=list(b.seeds)))
    print("development settings",len(b.runs),"selected",asdict(final),b.score(final),flush=True)

def holdout(args):
    b=Benchmark(args,TEST,(31,32,33,34,35))
    lock=json.loads((b.root/"lock.json").read_text())
    for cfg in lock["arms"].values():b.evaluate(Config(**cfg))
    write(b.root/"test_runs.json",b.runs)
    print("held-out settings",len(b.runs),flush=True)

if __name__=="__main__":
    args=arguments()
    (tune if args.mode=="source" else holdout)(args)
