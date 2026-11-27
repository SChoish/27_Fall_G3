"""Paired sequential SPICE benchmark for elite, metric and snapshot reuse."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import sleep
import hashlib
import json

import numpy as np
import experiment as exp
from adaptive import POLICY, size_move
from audit import checks
from circuit import Circuit, ratios, simulate
from extend import SOURCE
from proposal import propose
from alarm_model import Critic, anchors, cost, features, fit_attention, inputs, local_search

NEW_FORK=Circuit("sky",(-1,0,1,2,2,-2,5),(1.,1.75,2.25,3.5,2.,1.,2.5),
                 (3,4,6),(.65,1.25,1.50),2.7)
TARGETS={"fork":exp.CIRCUITS["fork"],"mixed_gf":exp.CIRCUITS["mixed_gf"],
         "load_sky":replace(exp.CIRCUITS["pair_sky"],loads=(.55,1.55),coupling=3.5),
         "seven_sky":NEW_FORK,"seven_gf":replace(NEW_FORK,pdk="gf")}
SOURCE_NETS={k:exp.CIRCUITS[k] for k in exp.SOURCES}
SOURCE_NETS["source_pair"]=SOURCE
METHODS=("role","bo","single","elite","scalar","attention","pullback","random_buffer","scored_buffer")
SEEDS=(21,22,23)

def train_source(root,seed):
    xx=[];yy=[];examples=[]
    for name,c in SOURCE_NETS.items():
        data=json.loads((root/(name+"_source.json")).read_text())
        desc=features(c);x=np.array(data["designs"]);x0=np.ones(x.shape[1])
        xx.extend(inputs(np.vstack((x0,x)),desc));yy.extend([data["ref"],*data["labels"]])
        examples.extend([(z,desc,exp.upper(c)) for z in [x0,*x[-6:]]])
    model=Critic(seed)
    model.fit(np.array(xx),np.array(yy),500,.006)
    return model,fit_attention(model,examples)

class Ledger:
    def __init__(self,name,args):
        self.name=name;self.args=args;self.c=TARGETS[name]
        self.path=args.out/"alarm"/"physical"/name
        self.path.mkdir(parents=True,exist_ok=True)
        self.old=args.out/"physical"/name
    def read(self,x,k):
        x=np.round(x,6).tolist()
        key=hashlib.sha256(json.dumps([x,k]).encode()).hexdigest()+".json"
        dest=self.path/key
        if dest.exists():return json.loads(dest.read_text())["raw"]
        old=self.old/key
        if self.name in ("fork","mixed_gf") and old.exists():
            raw=json.loads(old.read_text())["raw"];reuse=True
        else:
            raw=simulate(self.c,x,k,self.args);reuse=False
        dest.write_text(json.dumps(dict(x=x,corner=k,raw=raw,prior_cache=reuse)))
        return raw
    def pair(self,x,k):
        return [self.read(x,j) for j in (k,k+1)]

def choose_snapshot(buffer, center, current_cost, rng, scored):
    if not scored:return int(rng.integers(len(buffer)))
    def rank(item):
        _,past,base,error,improvement,_=item
        dist=np.linalg.norm(np.log(center/past))/np.sqrt(len(past))
        return error+improvement+dist+.35*abs(current_cost-base)
    return min(range(len(buffer)),key=lambda j:rank(buffer[j]))

def trial(ledger,prior,beta,method,seed):
    c=ledger.c;desc=features(c);hi=exp.upper(c)
    nominal=np.ones(len(hi))
    ref=ledger.pair(nominal,0)+ledger.pair(nominal,2)
    if any("error" in r for r in ref):raise RuntimeError(ledger.name)
    history=[nominal];labels=[ratios(ref[:2],ref[:2])]
    model=prior.copy();bank=[];rng=np.random.default_rng(seed)
    if method=="reset_head":
        model.params[2][:]=0.
        model.params[3][:]=labels[0]
    trace=[];hits={};calls=4
    while calls+2<=40 and len(hits)<2:
        t=len(trace);snapshot=None;origin=None
        if method=="role40" and t>=len(POLICY):
            pairs=[(.92,.9,1.),(.90,.94,.96),(.88,.98,1.04),(.95,.86,.96),(.90,.90,1.04),(.85,.95,.96),(.94,.82,1.04),(.89,.96,1.)]
            a,b,p=pairs[(t-len(POLICY))%len(pairs)]
            terminal=np.repeat([j in c.outputs for j in range(len(c.parents))],2)
            x=np.clip(np.where(terminal,a,b)*np.tile([p,1/p],len(c.parents)),.58,hi)
        elif t==0 or method in ("role","role40") or (method=="prefix_attention" and t==1):
            if t==len(POLICY):break
            x=size_move(c,POLICY,t,history,labels)
        elif method=="bo":
            incumbent=history[np.argmin(cost(labels))]
            candidates=exp.pool(c,rng,incumbent)
            x=propose("cold_bo",{},c,np.array(history),np.array(labels),candidates,rng)
        else:
            center=history[np.argmin(cost(labels))];base=float(min(cost(labels)))
            if "buffer" in method and bank:
                snapshot=choose_snapshot(bank,center,base,rng,method=="scored_buffer")
                origin=bank[snapshot][-1]
                model=bank[snapshot][0].copy()
            model.fit(inputs(history,desc),np.array(labels),20,.0018,freeze_encoder=method in ('keep_head','reset_head'))
            kind=method if method in ("scalar","attention","pullback") else "euclid"
            if "buffer" in method or method in ("keep_head","reset_head","prefix_attention"):kind="attention"
            x=local_search(model,desc,hi,history,labels,kind,1 if method=="single" else 3,beta,seed,t)
        x=np.round(x,6)
        dev=ledger.pair(x,0);calls+=2
        y=ratios(dev,ref[:2]);ok,edp=checks(dev,ref[:2]);verified=False
        if ok and edp<=.95 and calls+2<=40:
            confirm=ledger.pair(x,2);calls+=2;verified=True
            good,val=checks(confirm,ref[2:])
            for goal in (.95,.90):
                if good and max(edp,val)<=goal:hits.setdefault(str(goal),calls)
        if "buffer" in method and t>0:
            pred=model.predict(inputs(x,desc))[0]
            improvement=float(cost(y)-base)
            # Shared deterministic insertion coin, independent of snapshot selection RNG.
            insert=np.random.default_rng([seed,t,443]).random()<.4
            if not bank or improvement<0 or insert:
                bank.append((model.copy(),center.copy(),base,float(np.mean(abs(pred-y))),improvement,t))
                bank=bank[-4:]
        history.append(x);labels.append(y)
        trace.append(dict(x=x.tolist(),calls=calls,confirmed=verified,origin=origin))
    return dict(circuit=ledger.name,method=method,seed=seed,hits=hits,calls=calls,trace=trace)

def main(args=None, wait_for_primary=False):
    if args is None:args=exp.arguments()
    root=args.out/"alarm";root.mkdir(exist_ok=True)
    models={s:train_source(args.out,s) for s in SEEDS}
    (root/"attention.json").write_text(json.dumps({s:b.tolist() for s,(_,b) in models.items()}))
    def run(name):
        if wait_for_primary:
            while not (root/f"{name}-scored_buffer-23.json").exists():sleep(5)
        ledger=Ledger(name,args)
        for seed in SEEDS:
            for method in METHODS:
                if method in ("role","role40") and seed!=21:continue
                path=root/f"{name}-{method}-{seed}.json"
                if path.exists():continue
                result=trial(ledger,*models[seed],method,seed)
                path.write_text(json.dumps(result))
            print(name,seed,flush=True)
    with ThreadPoolExecutor(args.workers) as pool:list(pool.map(run,TARGETS))

if __name__=="__main__":main()
