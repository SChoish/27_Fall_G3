"""A shared, size-independent circuit critic and continuous local search."""
import numpy as np
from experiment import structure, upper

def features(c):
    f, edges = structure(c)
    n = len(f)
    roles = np.column_stack((np.ones(n), f[:,0], f[:,1], f[:,2], f[:,3],
                            f[:,4], f[:,0]*f[:,4], f[:,5], f[:,6],
                            (edges==5).any(axis=1)))
    context = np.array([n/16, len(c.outputs)/3, np.mean(c.loads), c.coupling/4])
    return roles, context

def inputs(x, descriptor):
    roles, context = descriptor
    z = np.log(np.atleast_2d(x))
    scale = 3/len(roles)
    return np.column_stack((scale*z@roles, scale*(z*z)@roles,
                            np.tile(context,(len(z),1))))

class Critic:
    def __init__(self, seed):
        rng=np.random.default_rng(seed)
        self.params=[rng.normal(0,.22,(24,24)),np.zeros(24),
                     rng.normal(0,.14,(24,5)),np.zeros(5)]

    def copy(self):
        result=object.__new__(Critic)
        result.params=[p.copy() for p in self.params]
        return result

    def predict(self,x):
        w,b,v,d=self.params
        return np.tanh(x@w+b)@v+d

    def fit(self,x,y,steps,lr):
        m=[np.zeros_like(p) for p in self.params]
        v=[np.zeros_like(p) for p in self.params]
        for i in range(1,steps+1):
            w,b,out,d=self.params
            h=np.tanh(x@w+b)
            delta=2*(h@out+d-y)/y.size
            inner=(delta@out.T)*(1-h*h)
            grads=(x.T@inner,inner.sum(0),h.T@delta,delta.sum(0))
            for j,g in enumerate(grads):
                m[j]=.9*m[j]+.1*g
                v[j]=.999*v[j]+.001*g*g
                self.params[j]-=lr*(m[j]/(1-.9**i))/(np.sqrt(v[j]/(1-.999**i))+1e-8)

    def jacobian(self,x,desc):
        roles,_=desc
        w,b,v,_=self.params
        h=np.tanh(inputs(x,desc)[0]@w+b)
        feature_jac=(w*((1-h*h)[None,:]))@v
        return (3/len(x))*(roles@feature_jac[:10]+2*np.log(x)[:,None]*(roles@feature_jac[10:20])).T

def cost(y):
    # A smooth hinge gives a differentiable cost at the electrical boundary.
    y=np.asarray(y)
    return y[...,0]+3*np.logaddexp(0,(y[...,1:]+.01)*60).sum(axis=-1)/60

def direction(net,x,desc,kind,beta):
    y=net.predict(inputs(x,desc))[0]
    jac=net.jacobian(x,desc)
    weight=1/(1+np.exp(-np.clip(60*(y[1:]+.01),-40,40)))
    g=np.r_[1.,3*weight]@jac
    if kind=="euclid":
        p=-g
    else:
        floor=.015*np.mean(g*g)+.0001
        if kind=="pullback":
            metric=jac.T@(np.r_[1.,.15+3*weight][:,None]*jac)+floor*np.eye(len(x))
        else:
            logits=desc[0]@beta if kind=="attention" else np.zeros(len(x))
            a=np.exp(logits-logits.max());a/=a.mean()
            metric=np.outer(g,g)+np.diag(.7*a*g*g+floor)
        p=-np.linalg.solve(metric,g)
    return p/max(np.max(np.abs(p)),1e-10),g

def fit_attention(net, examples):
    # Coordinate descent on source-only one-step surrogate cost, critic held fixed.
    beta=np.zeros(10)
    def objective(b):
        total=0.
        for x,desc,hi in examples:
            p,_=direction(net,x,desc,"attention",b)
            trial=np.clip(x*np.exp(.10*p),.58,hi)
            total+=float(cost(net.predict(inputs(trial,desc))[0]))
        return total/len(examples)+.0002*np.dot(b,b)
    for step in (.6,.3):
        for j in range(1,len(beta)):
            grid=[beta.copy() for _ in range(3)]
            grid[1][j]+=step;grid[2][j]-=step
            beta=min(grid,key=objective)
    return beta

def anchors(y,k):
    first=int(np.argmin(cost(y)))
    slack=np.column_stack((y[:,0],y[:,2],np.maximum(y[:,1:],0).max(1)))
    keep=[i for i in range(len(y)) if not np.any(np.all(slack<=slack[i],axis=1)&np.any(slack<slack[i],axis=1))]
    chosen=[first]
    while len(chosen)<k and any(i not in chosen for i in keep):
        chosen.append(max((i for i in keep if i not in chosen),key=lambda i:np.linalg.norm(slack[chosen]-slack[i],axis=1).min()))
    return chosen

def local_search(net, desc, hi, history, labels, kind, k, beta, seed, index):
    hist=np.array(history); y=np.array(labels)
    centers=anchors(y,k)
    rng=np.random.default_rng([seed,index,2851])
    candidates=[]; scores=[]
    # 12 directions, 5 trial step lengths per direction in every learned arm.
    for j,noise in enumerate(rng.normal(size=(12,len(hi)))):
        x=hist[centers[j%len(centers)]]
        p,g=direction(net,x,desc,kind,beta)
        delta=p+(0 if j<3 else .45)*noise
        delta/=max(np.max(np.abs(delta)),1e-10)
        steps=np.array([.28,.18,.11,.065,.035])
        trial=np.clip(x*np.exp(steps[:,None]*delta),.58,hi)
        val=cost(net.predict(inputs(trial,desc)))
        base=float(cost(net.predict(inputs(x,desc))[0]))
        moved=np.log(trial/x)
        accepted=(val<=base+1e-4*(moved@g)) & ((moved@g)<0)
        candidates.extend(trial)
        scores.extend(val+np.where(accepted,0.,2.))
    cand=np.array(candidates); score=np.array(scores)
    seen=np.max(abs(cand[:,None]-hist),axis=2).min(1)<1e-5
    score[seen]=np.inf
    if np.isinf(score).all():
        return np.clip(hist[centers[0]]*np.exp(rng.normal(0,.045,len(hi))),.58,hi)
    return cand[score.argmin()]
