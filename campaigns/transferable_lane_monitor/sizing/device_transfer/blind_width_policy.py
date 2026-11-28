"""Search anonymous numerical coordinates from bounds and past black-box evaluations."""
import numpy as np


def moments(z):
    # Fixed-size permutation-invariant summary. Differences vanish at nominal z=0.
    x = np.atleast_2d(z)
    return np.column_stack([x.mean(1), x.std(1), np.mean(x*x,axis=1), x.mean(1)**2])


def fit_prior(examples):
    X = np.vstack([moments(z) for z,y in examples])
    Y = np.vstack([y for z,y in examples])
    return np.linalg.solve(X.T@X + .01*np.eye(X.shape[1]), X.T@Y)


def kernel(x,y):
    distance = np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)/x.shape[1]
    return np.exp(-.5*distance/.23**2)


def propose(arm,history,observations,lower,upper,seed,prior=None):
    z = np.array(history)
    y = np.array(observations)
    t,dim = z.shape
    rng = np.random.default_rng([seed,t,6481])
    if arm=="random" or t<=4 or (arm=="acor" and t<6):
        return rng.uniform(lower,upper)
    penalty = lambda values: values[...,0]+4*np.maximum(values[...,1:],0).sum(axis=-1)
    if arm=="acor":
        pop=z[np.argsort(penalty(y),kind="stable")[:6]]
        weights=np.exp(-.5*(np.arange(6)/2.4)**2)
        anchor=pop[rng.choice(6,p=weights/weights.sum())]
        sd=.85*np.abs(pop-anchor).sum(axis=0)/5
        sample=rng.normal(anchor,sd)
        span=upper-lower
        return lower+span-np.abs((sample-lower)%(2*span)-span)
    if arm not in ("bo","transfer_bo"):raise ValueError(arm)
    if (arm=="transfer_bo")!=(prior is not None):raise ValueError("source prior mismatch")
    return rank_candidates(z,y,lower,upper,rng,prior=prior)


def rank_candidates(z,y,lower,upper,rng,prior=None,network=None,gp=True,calibrated=None):
    t,dim=z.shape
    penalty=lambda values: values[...,0]+4*np.maximum(values[...,1:],0).sum(axis=-1)
    center=z[np.argmin(penalty(y))]
    span=upper-lower
    radius=rng.uniform(.04,.55,(256,1))*span
    local=rng.uniform(np.maximum(lower,center-radius),np.minimum(upper,center+radius))
    candidates=np.vstack((local,rng.uniform(lower,upper,(256,dim))))
    if network is not None:
        feature=network.encode(z)
        test_feature=network.encode(candidates)
        mean=y[0]+network.readout(feature)
        pred=y[0]+network.readout(test_feature)
        if not gp:
            ridge=feature.T@feature+.01*np.eye(feature.shape[1])
            head=np.linalg.solve(ridge,feature.T@(y-mean))
            return candidates[np.argmin(penalty(pred+test_feature@head))]
    u=(z-lower)/span
    v=(candidates-lower)/span
    K=kernel(u,u)+.004*np.eye(t)
    cross=kernel(v,u)
    w=np.linalg.solve(K,cross.T).T
    if calibrated is not None:
        mean,pred=calibrated(z,candidates,y,K)
    elif network is None:
        mean=np.broadcast_to(y[0],y.shape)
        pred=y[0]+np.zeros((len(v),y.shape[1]))
    if prior is not None:
        mean=mean+moments(z)@prior
        pred+=moments(candidates)@prior
    pred+=w@(y-mean)
    sig=np.sqrt(np.maximum(0.,1-(w*cross).sum(1)))
    scale=np.maximum(y.std(axis=0),.02)
    score=penalty(pred-.7*sig[:,None]*scale)
    return candidates[np.argmin(score)]
