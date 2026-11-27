"""Shared per-device action and electrical sensitivity heads for variable netlists."""
import json
from math import erf
import numpy as np
from experiment import structure,CIRCUITS,SOURCES,upper

class Network:
    def __init__(self,mode,seed):
        self.mode=mode
        rng=np.random.default_rng(seed)
        self.w={"in":rng.normal(0,.35,(8,12)),"b":np.zeros(12),
                "out":rng.normal(0,.19,(12,6)),"c":np.zeros(6)}
        if mode!="mlp":
            self.w["msg"]=rng.normal(0,.16,(12,12))
            self.w["edgeval"]=rng.normal(0,.12,(6,12))
        if mode=="gat":
            self.w["q"]=rng.normal(0,.12,(12,4));self.w["k"]=rng.normal(0,.12,(12,4))
            self.w["edgebias"]=np.zeros(6)

    def forward(self,f,edge):
        w=self.w
        h=np.tanh(f@w["in"]+w["b"])
        if self.mode=="mlp":
            return h,(h,)
        mask=edge>0
        if self.mode=="gat":
            q,k=h@w["q"],h@w["k"]
            logits=np.where(mask,q@k.T/2+w["edgebias"][edge],-1e10)
            a=np.exp(logits-logits.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
        else:
            a=mask/mask.sum(1,keepdims=True)
        value=h[None,:,:]+w["edgeval"][edge]
        msg=(a[:,:,None]*value).sum(1)
        z=np.tanh(h+msg@w["msg"])
        return z,(h,a,msg,value)

    def fit(self,examples):
        yy=np.vstack([y for _,_,y in examples])
        self.avg=yy.mean(0);self.scale=np.maximum(yy.std(0),.035)
        mom={k:0. for k in self.w};vel=mom.copy()
        for t in range(1,451):
            grad={k:np.zeros_like(v) for k,v in self.w.items()}
            for f,edge,y in examples:
                w=self.w
                z,cache=self.forward(f,edge);h=cache[0]
                dp=2*(z@w["out"]+w["c"]-(y-self.avg)/self.scale)/yy.size
                grad["out"]+=z.T@dp;grad["c"]+=dp.sum(0)
                dh=dp@w["out"].T
                if self.mode!="mlp":
                    _,a,msg,value=cache
                    du=dh*(1-z*z)
                    grad["msg"]+=msg.T@du
                    dm=du@w["msg"].T
                    dv=a[:,:,None]*dm[:,None,:]
                    np.add.at(grad["edgeval"],edge,dv)
                    dh=du+dv.sum(0)
                    if self.mode=="gat":
                        da=(value*dm[:,None,:]).sum(2)
                        ds=a*(da-(a*da).sum(1,keepdims=True))
                        np.add.at(grad["edgebias"],edge,ds)
                        dq=ds@(h@w["k"])/2;dk=ds.T@(h@w["q"])/2
                        grad["q"]+=h.T@dq;grad["k"]+=h.T@dk
                        dh+=dq@w["q"].T+dk@w["k"].T
                dh*=1-h*h
                grad["in"]+=f.T@dh;grad["b"]+=dh.sum(0)
            for k,g in grad.items():
                mom[k]=.9*mom[k]+.1*g;vel[k]=.999*vel[k]+.001*g*g
                self.w[k]-=.006*(mom[k]/(1-.9**t))/(np.sqrt(vel[k]/(1-.999**t))+1e-8)

    def outputs(self,f,edge):
        e=self.forward(f,edge)[0]
        y=(e@self.w["out"]+self.w["c"])*self.scale+self.avg
        return y,e

def train(root,seed):
    examples=[]
    for name in SOURCES:
        d=json.loads((root/(name+"_source.json")).read_text())
        examples.append((*structure(CIRCUITS[name]),np.array(d["per_device"])))
    models={}
    for mode in ("mlp","mean","gat"):
        net=Network(mode,seed);net.fit(examples)
        models[mode]=net
    models["polarity"]=np.stack([np.vstack([y[f[:,0]==j] for f,e,y in examples]).mean(0) for j in (0,1)])
    return models

def loss(y):
    return y[:,0]+7*np.maximum(y[:,1:],0).sum(1)

def propose(method,models,c,hist,y,candidates,rng):
    if method=="random":
        return rng.uniform(.58,upper(c))
    if method=="cold_bo":
        delta=(candidates[:,None,:]-hist[None,:,:])/.24
        cross=np.exp(-.5*(delta*delta).sum(2)/np.sqrt(hist.shape[1]))
        delta=(hist[:,None,:]-hist[None,:,:])/.24
        kern=np.exp(-.5*(delta*delta).sum(2)/np.sqrt(hist.shape[1]))+.004*np.eye(len(y))
        weights=np.linalg.solve(kern,cross.T).T
        pred=y.mean(0)+weights@(y-y.mean(0))
        sd=np.sqrt(np.maximum(.006,1-(cross*weights).sum(1)))[:,None]
        sig=sd*np.maximum(y.std(0),[.12,.10,.10,.12,.035])
        z=(np.array([np.log(.90),0,0,0,0])-pred)/sig
        score=-np.log(np.maximum(.5*(1+np.vectorize(erf,otypes=[float])(z/np.sqrt(2))),1e-14)).sum(1)
        return candidates[score.argmin()]

    f,edge=structure(c)
    if method=="polarity":
        out=models["polarity"][f[:,0].astype(int)]
        e=np.eye(2)[f[:,0].astype(int)]
    else:
        mode=method.split("_")[0]
        out,e=models[mode].outputs(f,edge)
    # The same action head is decoded at every device; the number of returned moves varies.
    action=np.clip(np.exp(out[:,5]),.58,upper(c))
    if method=="gat_new":
        action=np.ones_like(action)
    if len(hist)==1 and method!="gat_new":
        return action
    candidates=np.vstack([candidates,action, np.clip(action*np.exp(rng.normal(0,.08,(96,len(action)))),.58,upper(c))])
    d=np.log(candidates)
    train=np.log(hist)
    J=out[:,:5] if method!="gat_new" else np.zeros_like(out[:,:5])
    pred=y[0]+d@J
    if method!="gat_frozen" and len(y)>1:
        # Fit a ridge residual around the source sensitivities using observed development metrics.
        feature=np.column_stack([e,np.ones(len(e))])/np.sqrt(len(e))
        tx=train@feature;cx=d@feature
        residual=y-y[0]-train@J
        pred+=cx@tx.T@np.linalg.solve(tx@tx.T+.005*np.eye(len(tx)),residual)
    score=loss(pred)
    seen=np.max(abs(candidates[:,None,:]-hist[None,:,:]),axis=2).min(axis=1)<2e-5
    score[seen]=np.inf
    return candidates[score.argmin()]
