"""Permutation-invariant anonymous-width encoders. NumPy training for small source datasets."""
import numpy as np


class WidthNet:
    def __init__(self,attention,seed=1807):
        self.attention=attention
        rng=np.random.default_rng(seed)
        self.params={
            "a":rng.normal(size=12),"b":rng.normal(0,.25,12),
            "h":rng.normal(0,.25,(12,12)),"c":np.zeros(12),
            "out":rng.normal(0,.2,(12,5))}
        if attention:
            self.params.update(q=rng.normal(0,.25,12), k=rng.normal(0,.25,12))
        self.scale=np.ones(5)

    def forward(self,z):
        p=self.params
        x=(np.atleast_2d(z)/.3)[...,None]
        h=np.tanh(x*p["a"]+p["b"])
        logits=weights=None
        if self.attention:
            logits=(h@p["q"])[:,:,None]+(h@p["k"])[:,None,:]
            a=np.where(logits>=0,logits,.2*logits)
            weights=np.exp(a-a.max(2,keepdims=True))
            weights/=weights.sum(2,keepdims=True)
            mixed=.5*(h+weights@h)
        else:
            mixed=h
        v=np.tanh(mixed@p["h"]+p["c"])
        return v.mean(1),(x,h,logits,weights,mixed,v)

    def encode(self,z):
        return self.forward(z)[0]-self.forward(np.zeros((1,np.shape(z)[-1])))[0]

    def readout(self,encoded):
        return (encoded@self.params["out"])*self.scale

    def backward(self,d,cache):
        x,h,logits,weights,mixed,v=cache
        dv=d[:,None,:]*(1-v*v)/v.shape[1]
        grad={"h":mixed.reshape(-1,12).T@dv.reshape(-1,12),"c":dv.sum((0,1))}
        dm=dv@self.params["h"].T
        dh=dm
        if self.attention:
            dw=.5*(dm@h.swapaxes(1,2))
            dl=weights*(dw-(dw*weights).sum(2,keepdims=True))*np.where(logits>=0,1.,.2)
            dq=dl.sum(2);dk=dl.sum(1)
            grad["q"]=(dq[...,None]*h).sum((0,1))
            grad["k"]=(dk[...,None]*h).sum((0,1))
            dh=.5*(dm+weights.swapaxes(1,2)@dm)+dq[...,None]*self.params["q"]+dk[...,None]*self.params["k"]
        dp=dh*(1-h*h)
        grad.update(a=(dp*x).sum((0,1)),b=dp.sum((0,1)))
        return grad

    def loss_grad(self,examples):
        grad={k:np.zeros_like(v) for k,v in self.params.items()}
        loss=0.
        for z,y in examples:
            pooled,cache=self.forward(z)
            zero,anchor=self.forward(np.zeros((1,z.shape[1])))
            encoded=pooled-zero
            err=encoded@self.params["out"]-y/self.scale
            loss+=np.mean(err*err)/len(examples)
            dy=2*err/(len(examples)*err.size)
            grad["out"]+=encoded.T@dy
            d=dy@self.params["out"].T
            for flow,ctx in ((d,cache),(-d.sum(0,keepdims=True),anchor)):
                for k,v in self.backward(flow,ctx).items():grad[k]+=v
        return loss,grad

    def fit(self,examples):
        self.scale=np.maximum(np.vstack([y for z,y in examples]).std(0),.015)
        m={k:np.zeros_like(v) for k,v in self.params.items()}
        v={k:np.zeros_like(w) for k,w in self.params.items()}
        for t in range(1,301):
            _,grad=self.loss_grad(examples)
            for k,w in self.params.items():
                m[k]=.9*m[k]+.1*grad[k]
                v[k]=.999*v[k]+.001*grad[k]**2
                w-=.003*(m[k]/(1-.9**t))/(np.sqrt(v[k]/(1-.999**t))+1e-8)
        return self
