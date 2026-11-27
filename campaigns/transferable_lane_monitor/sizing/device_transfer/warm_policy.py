"""Closed-loop proposals using a scalar ALARM metric without attention."""
from dataclasses import dataclass
import numpy as np

from alarm_model import Critic, anchors, cost, features, inputs
from warm_space import bank,span

@dataclass(frozen=True)
class Config:
    unit: bool=False
    residual: bool=False
    warm: int=0
    region: str="local"
    first: str="role"
    cap: int=0
    reuse: float=0.
    select: str="score"

def source_prior(root,seed):
    from alarm_pilot import SOURCE_NETS
    import json
    x=[];y=[]
    for name,c in SOURCE_NETS.items():
        r=json.loads((root/(name+"_source.json")).read_text())
        widths=np.array(r["designs"])
        x.extend(inputs(np.vstack([np.ones(widths.shape[1]),widths]),features(c)))
        y.extend([r["ref"],*r["labels"]])
    out=Critic(seed);out.fit(np.array(x),np.array(y),500,.006)
    return out

class Space:
    def __init__(self,c):
        self.x,self.tags=bank(c)
        self.z=np.log(self.x)
        self.desc=features(c);self.encoded=inputs(self.x,self.desc)
        self.span=span(c)
        dz=self.z[:,None]-self.z[None,:]
        self.bandwidth2=.11**2*self.z.shape[1]
        self.kernel=np.exp(-.5*(dz**2).sum(-1)/self.bandwidth2)

    def rank(self,model,history,labels,cfg,seed,t):
        y=np.array(labels); h=np.array(history)
        prediction=model.predict(self.encoded)
        weights=None
        if cfg.residual:
            # Independent log electrical outputs, target-only ridge residual.
            weights=np.linalg.solve(self.kernel[np.ix_(h,h)]+.06*np.eye(len(h)),y-prediction[h])
            prediction=prediction+self.kernel[:,h]@weights
        scale=self.span if cfg.unit else np.ones_like(self.span)
        avail=np.ones(len(self.x),bool);avail[h]=False
        short=set()
        for j in anchors(y,3):
            anchor=h[j];z=self.z[anchor]
            jac=model.jacobian(self.x[anchor],self.desc)
            if weights is not None:
                dk=self.kernel[anchor,h,None]*(self.z[h]-z)/self.bandwidth2
                jac=jac+weights.T@dk
            yp=prediction[anchor]
            penalties=3/(1+np.exp(-np.clip(60*(yp[1:]+.01),-40,40)))
            g=(np.r_[1.,penalties]@jac)*scale
            # For a scalar gradient, (diag(d)+gg^T)^-1 g is parallel to g/d.
            d=.7*g*g+.015*np.mean(g*g)+.0001
            p=-g/d
            p=p/max(np.max(abs(p*scale)),1e-12)
            du=(self.z-z)/scale
            for radius in (.045,.10,.18):
                distance=np.mean((du-radius*p)**2,axis=1)
                distance[~avail]=np.inf
                short.update(np.argsort(distance,kind="stable")[:6].tolist())
        ids=np.array(sorted(short))
        return int(ids[np.argmin(cost(prediction[ids]))])

def warm_prefix(space,cfg,seed):
    rng=np.random.default_rng([seed,91])
    ids=rng.permutation(np.flatnonzero(space.tags==cfg.region))[:cfg.warm].tolist()
    return [1,*ids] if cfg.first=="role" else [*ids,1]

def rollout(space,env,prior,cfg,seed):
    model=prior.copy()
    order=warm_prefix(space,cfg,seed)
    hist=[0];y=[env.nominal_label];calls=4;hits={};trace=[];snapshots=[]
    while calls+2<=40 and len(hits)<2:
        t=len(trace); restored=None;origins=[];gate=False
        if t<len(order):
            idx=order[t];stage="role" if idx==1 else "warm"
        else:
            eligible=[r for r in snapshots if r[0]<t-1]
            if cfg.cap and eligible:
                rng=np.random.default_rng([seed,t,5411])
                gate=rng.random()<cfg.reuse
                if gate:
                    origins=[s[0] for s in eligible]
                    latest=space.encoded[hist[-1]]
                    def err(net):
                        norm=np.array([.15,.10,.10,.12,.04])
                        return float(np.mean(abs((net.predict(latest)-y[-1])/norm)))
                    if cfg.select=="random":
                        selected=eligible[int(rng.integers(len(eligible)))]
                    else:
                        selected=min(eligible,key=lambda s:err(s[1]))
                    if cfg.select!="guard" or err(selected[1])<err(model):
                        restored,net=selected;model=net.copy()
            model.fit(space.encoded[hist],np.array(y),20,.0018)
            idx=space.rank(model,hist,y,cfg,seed,t);stage="metric"
            if cfg.cap:
                snapshots.append((t,model.copy()))
                snapshots=snapshots[-(cfg.cap+1):]  # cap eligible older snapshots plus the live one
        label,ok,dev=env.development(idx);calls+=2
        confirmed=bool(ok and dev<=.95 and calls+2<=40)
        if confirmed:
            valid,test=env.confirmation(idx);calls+=2
            for g in (.95,.90):
                if valid and max(dev,test)<=g:hits.setdefault(str(g),calls)
        hist.append(idx);y.append(label)
        trace.append(dict(i=idx,stage=stage,calls=calls,confirmed=confirmed,restored=restored,
                          candidates=origins,gate=bool(gate)))
    return dict(seed=seed,hits=hits,calls=calls,trace=trace)
