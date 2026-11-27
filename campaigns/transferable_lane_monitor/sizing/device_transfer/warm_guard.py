"""Test whether randomization helps specifically near a measured feasibility boundary."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import sleep
import json
import numpy as np
from experiment import arguments
from warm_policy import Config,rollout,source_prior
from warm_continuous import Continuous,Live
from warm_space import TEST,Store,DEV

class Guard(Continuous):
    def __init__(self,c,noise):
        super().__init__(c);self.noise=noise;self.triggered=False
    def rank(self,model,h,y,cfg,seed,t):
        if t==1 and y[1][0]<=np.log(.95) and np.max(y[1][1:])>0:
            delta=y[1][1:]-y[0][1:];rising=delta>1e-6
            bound=min(-y[0][1:][rising]/delta[rising])
            alpha=np.clip(.94*bound,.2,1.6)
            z=alpha*np.log(self.x[1])
            if self.noise:
                z+=np.random.default_rng([seed,830]).normal(0,.025,len(z))
            x=np.round(np.clip(np.exp(z),.58,self.hi),6)
            from alarm_model import inputs
            self.x=np.vstack([self.x,x]);self.encoded=np.vstack([self.encoded,inputs(x,self.desc)])
            self.triggered=True
            return len(self.x)-1
        return super().rank(model,h,y,cfg,seed,t)

if __name__=="__main__":
    args=arguments();root=args.out/"warm"
    priors={s:source_prior(args.out,s) for s in (31,32,33)}
    def run(item):
        name,c=item
        for seed,prior in priors.items():
            while not (root/f"continuous-{name}-strict_goal-{seed}.json").exists():sleep(3)
            for key,cfg in (("scaled_only",Config(unit=True)),("residual_only",Config(residual=True)),
                            ("guard_secant",Config()),("guard_random",Config())):
                store=Store(name,c,args)
                space=Guard(c,key=="guard_random") if key.startswith("guard") else Continuous(c)
                r=rollout(space,Live(space,store),prior,cfg,seed)
                for step in r["trace"]:step["x"]=space.x[step["i"]].tolist()
                (root/f"continuous-{name}-{key}-{seed}.json").write_text(json.dumps(
                    dict(circuit=name,arm=key,triggered=getattr(space,"triggered",False),**r)))
        print("coordinate and guarded probes",name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,TEST.items()))
