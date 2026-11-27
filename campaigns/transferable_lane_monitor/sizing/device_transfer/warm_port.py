"""Fresh PDK pair for electrical, per-port and polarity-aware second-step sizing."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import numpy as np
from alarm_model import inputs
from circuit import Circuit
from experiment import arguments
from warm_continuous import Continuous,Live
from warm_policy import Config,rollout,source_prior
from warm_space import Store

NEW=Circuit("sky",(-1,0,-2,2,3,4),(1.,2.15,1.1,1.55,2.3,3.65),(1,5),(.85,1.60),2.9)
DOMAINS={"edge_pair_sky":NEW,"edge_pair_gf":replace(NEW,pdk="gf")}
ARMS=("base","electrical","electrical_probe")
class PortSpace(Continuous):
    def __init__(self,c,arm):
        super().__init__(c);self.c=c;self.arm=arm;self.observations=[]
    def rank(self,model,h,y,cfg,seed,t):
        if self.arm=="base" or t>2 or (t==2 and self.arm!="electrical_probe"):
            return super().rank(model,h,y,cfg,seed,t)
        ref,probe=self.observations[:2]
        if t==1 and not any("error" in r for r in ref+probe):
            x=np.ones(len(self.hi))
            for j,node in enumerate(self.c.outputs):
                a=np.array([d["ports"][j] for d in ref])
                b=np.array([d["ports"][j] for d in probe])
                # A rising output is charged by PMOS, a falling output is discharged by NMOS.
                for edge,pol in ((0,1),(1,0)):
                    ratio=np.log(b[:,[edge,edge+2]]/a[:,[edge,edge+2]]).ravel()
                    limits=[np.log(1.1)/d for d in ratio if d>1e-8]
                    base_swing=max(abs(a[0,:2]-a[1,:2]))
                    test_swing=abs(b[0,edge]-b[1,edge])
                    original=abs(a[0,edge]-a[1,edge])
                    if test_swing>original+1e-8:
                        limits.append((1.05*base_swing-original)/(test_swing-original))
                    alpha=np.clip(.94*min([1.6,*limits]),.25,1.5)
                    x[2*node+pol]=.84**alpha
            self.center=x
        elif t==2 and hasattr(self,"center"):
            noise=np.random.default_rng([seed,1961]).uniform(-.045,.045,len(self.hi))
            x=self.center*np.exp(noise)
        else:
            return super().rank(model,h,y,cfg,seed,t)
        x=np.round(np.clip(x,.58,self.hi),6)
        self.x=np.vstack([self.x,x]);self.encoded=np.vstack([self.encoded,inputs(x,self.desc)])
        return len(self.x)-1

class PortLive(Live):
    def __init__(self,space,store):
        super().__init__(space,store);space.observations.append(self.ref[:2])
    def development(self,i):
        if len(self.space.observations)==1:self.space.observations.append(self.store.pair(self.space.x[i],0))
        return super().development(i)

if __name__=="__main__":
    args=arguments();root=args.out/"warm";seeds=(41,42,43)
    priors={s:source_prior(args.out,s) for s in seeds}
    def run(item):
        name,c=item;store=Store(name,c,args)
        for s in seeds:
            for arm in ARMS:
                space=PortSpace(c,arm)
                r=rollout(space,PortLive(space,store),priors[s],Config(),s)
                for step in r["trace"]:step["x"]=space.x[step["i"]].tolist()
                (root/f"port-{name}-{arm}-{s}.json").write_text(json.dumps(dict(circuit=name,arm=arm,**r)))
        print("electrical",name,flush=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(run,DOMAINS.items()))
