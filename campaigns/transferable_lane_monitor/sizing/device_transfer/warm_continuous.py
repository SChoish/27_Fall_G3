"""Independent continuous-width test of the settings selected by bank screening."""
from concurrent.futures import ThreadPoolExecutor
import json
from time import sleep
import numpy as np
from alarm_model import anchors,cost,inputs,local_search
from audit import checks
from circuit import ratios
from experiment import arguments,upper
from warm_policy import Config,Space,rollout,source_prior
from warm_space import TEST,Store

class Continuous(Space):
    def __init__(self,c):
        super().__init__(c);self.hi=upper(c)

    def rank(self,model,h,y,cfg,seed,t):
        hist=self.x[h];features=self.encoded[h];obs=np.array(y)
        if not cfg.residual and not cfg.unit:
            x=local_search(model,self.desc,self.hi,hist,obs,"scalar",3,None,seed,t)
        else:
            logh=np.log(hist)
            def kernel(xs):
                diff=np.log(np.atleast_2d(xs))[:,None]-logh[None,:]
                return np.exp(-.5*np.sum(diff*diff,axis=-1)/self.bandwidth2)
            weights=None
            if cfg.residual:
                weights=np.linalg.solve(kernel(hist)+.06*np.eye(len(h)),obs-model.predict(features))
            def predict(xs):
                v=model.predict(inputs(xs,self.desc))
                return v if weights is None else v+kernel(xs)@weights
            scale=self.span if cfg.unit else np.ones_like(self.span)
            ids=anchors(obs,3);cand=[];scores=[]
            rng=np.random.default_rng([seed,t,2851])
            for j,noise in enumerate(rng.normal(size=(12,len(self.hi)))):
                anchor=hist[ids[j%len(ids)]]
                yp=predict(anchor)[0]
                J=model.jacobian(anchor,self.desc)
                if weights is not None:
                    J+=weights.T@(kernel(anchor)[0,:,None]*(logh-np.log(anchor))/self.bandwidth2)
                pen=3/(1+np.exp(-np.clip(60*(yp[1:]+.01),-40,40)))
                gz=np.r_[1.,pen]@J;gu=gz*scale
                pu=-gu/(.7*gu*gu+.015*np.mean(gu*gu)+.0001)
                pz=pu*scale
                pz/=max(np.max(abs(pz)),1e-10)
                dz=pz+(0 if j<3 else .45)*noise
                dz/=max(np.max(abs(dz)),1e-10)
                xs=np.clip(anchor*np.exp(np.array([.28,.18,.11,.065,.035])[:,None]*dz),.58,self.hi)
                v=cost(predict(xs));dot=np.log(xs/anchor)@gz
                fail=(dot>=0)|(v>cost(yp)+1e-4*dot)
                cand.extend(xs);scores.extend(v+2*fail)
            cand=np.array(cand);scores=np.array(scores)
            scores[np.max(abs(cand[:,None]-hist),axis=2).min(1)<1e-5]=np.inf
            x=(cand[scores.argmin()] if np.isfinite(scores).any() else
               np.clip(hist[ids[0]]*np.exp(rng.normal(0,.045,len(self.hi))),.58,self.hi))
        x=np.round(x,6);self.x=np.vstack([self.x,x])
        self.encoded=np.vstack([self.encoded,inputs(x,self.desc)])
        return len(self.x)-1

class Live:
    def __init__(self,space,store):
        self.space=space;self.store=store
        x=space.x[0];self.ref=store.pair(x,0)+store.pair(x,2)
        self.nominal_label=ratios(self.ref[:2],self.ref[:2])
    def development(self,i):
        raw=self.store.pair(self.space.x[i],0)
        return ratios(raw,self.ref[:2]),*checks(raw,self.ref[:2])
    def confirmation(self,i):
        return checks(self.store.pair(self.space.x[i],2),self.ref[2:])

if __name__=="__main__":
    args=arguments();root=args.out/"warm"
    arms=json.loads((root/"lock.json").read_text())["arms"]
    keys=("fixed_bank_base","nonzero_warm","nonzero_buffer","combined","strict_goal")
    cfgs={key:Config(**arms[key]) for key in keys}
    seeds=(31,32,33);priors={s:source_prior(args.out,s) for s in seeds}
    def run(item):
        name,c=item
        while not (root/(name+".json")).exists():sleep(2)
        store=Store(name,c,args)
        for seed in seeds:
            cache={}
            for key,cfg in cfgs.items():
                if cfg not in cache:
                    space=Continuous(c)
                    r=rollout(space,Live(space,store),priors[seed],cfg,seed)
                    for step in r["trace"]:step["x"]=space.x[step["i"]].tolist()
                    cache[cfg]=r
                path=root/f"continuous-{name}-{key}-{seed}.json"
                path.write_text(json.dumps(dict(circuit=name,arm=key,**cache[cfg])))
        print("continuous",name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,TEST.items()))
