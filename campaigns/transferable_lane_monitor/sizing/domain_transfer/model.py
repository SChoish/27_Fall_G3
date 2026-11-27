"""Small supervised circuit encoders with analytic backpropagation and ridge readouts."""
import numpy as np
from data import WIDTHS


def inputs(stages, xs, load):
    # Node features in signal order. No PDK name, target labels or target normalization.
    pos=np.linspace(0,1,stages)
    x=np.array(xs)
    scale=np.exp((1-pos)*np.log(x[:,:1])+pos*np.log(x[:,1:]))
    return np.stack(np.broadcast_arrays(scale-1, np.log2(WIDTHS[stages])/2,
                    pos, pos==0, pos==1, np.log2(load)*(pos==1),
                    np.full(stages,stages/4)),axis=-1).astype(float)


def neighbors(n):
    return abs(np.arange(n)[:,None]-np.arange(n))<=1


class Encoder:
    def __init__(self,kind,seed):
        self.kind=kind
        rng=np.random.default_rng(seed)
        self.w={"node":rng.normal(0,.40,(7,12)), "bias":np.zeros(12),
                "head":rng.normal(0,.18,(24,5)), "offset":np.zeros(5)}
        if kind=="gat":
            self.w["neighbor"]=rng.normal(0,.17,(12,12))
            self.w["attention"]=rng.normal(0,.20,12)

    def forward(self,f,adj):
        h=np.tanh(f@self.w["node"]+self.w["bias"])
        if self.kind=="gat":
            # Single-head local attention including self; shared across nodes.
            logits=np.where(adj,(h@self.w["attention"])[:,None,:],-1e9)
            a=np.exp(logits-logits.max(axis=-1,keepdims=True))
            a/=a.sum(axis=-1,keepdims=True)
            m=a@h
            z=np.tanh(h+m@self.w["neighbor"])
        else:
            a=m=None
            z=h
        out=f[:,:,4:5]
        e=np.concatenate([z.mean(axis=1),(z*out).sum(axis=1)],axis=1)
        return e,(f,h,a,m,z,out)

    def fit(self,batches,seed):
        # Normalize labels only on source domains. All corners used in a source label
        # are the two development corners; confirmation remains evaluator-only.
        labels=np.concatenate([y for f,y in batches])
        self.mean=labels.mean(axis=0)
        self.scale=np.maximum(labels.std(axis=0),.04)
        mom={k:0 for k in self.w}
        vel=mom.copy()
        for t in range(1,701):
            grad={k:np.zeros_like(v) for k,v in self.w.items()}
            for f,y in batches:
                e,(_,h,a,m,z,out)=self.forward(f,neighbors(f.shape[1]))
                dp=2*(e@self.w["head"]+self.w["offset"]-(y-self.mean)/self.scale)/(len(labels)*5)
                grad["head"]+=e.T@dp
                grad["offset"]+=dp.sum(axis=0)
                de=dp@self.w["head"].T
                dz=de[:,:12][:,None,:]/f.shape[1]+out*de[:,12:][:,None,:]
                if self.kind=="gat":
                    du=dz*(1-z*z)
                    grad["neighbor"]+=m.reshape(-1,12).T@du.reshape(-1,12)
                    dm=du@self.w["neighbor"].T
                    da=dm@h.swapaxes(1,2)
                    ds=(a*(da-(da*a).sum(axis=2,keepdims=True))).sum(axis=1)
                    grad["attention"]+=np.einsum("bnh,bn->h",h,ds)
                    dh=du+a.swapaxes(1,2)@dm+ds[:,:,None]*self.w["attention"]
                else:
                    dh=dz
                dh*=1-h*h
                grad["node"]+=f.reshape(-1,7).T@dh.reshape(-1,12)
                grad["bias"]+=dh.sum(axis=(0,1))
            for k,g in grad.items():
                mom[k]=.9*mom[k]+.1*g
                vel[k]=.999*vel[k]+.001*g*g
                self.w[k]-=.009*(mom[k]/(1-.9**t))/(np.sqrt(vel[k]/(1-.999**t))+1e-8)

    def embedding(self,f):
        e,_=self.forward(f,neighbors(f.shape[1]))
        return np.column_stack([e,np.ones(len(e))])

    def predict(self,e):
        return (e[:,:-1]@self.w["head"]+self.w["offset"])*self.scale+self.mean


def adapt(e,observed,y,prior,alpha=.015):
    train=e[observed]
    kernel=train@train.T+alpha*np.eye(len(train))
    return prior+(e@train.T)@np.linalg.solve(kernel,y-prior[observed])


def choose_score(pred,goal=.90):
    # Shared by all representation arms. Include a conservative margin on constraints.
    return np.maximum(pred[:,0]-np.log(goal),0)+4*np.maximum(pred[:,1:]+.008,0).sum(axis=1)+.01*pred[:,0]
