"""Probability of attaining the primary constrained target with a calibrated margin."""
from math import erf
import numpy as np


def target_proposal(rng, x, y, source):
    from campaign import kernel
    points = rng.uniform([.5,.5],[1.,1.6],size=(768,2))
    if source is not None:
        points = np.vstack([points,source[0]])
    points = points[np.linalg.norm(points[:,None,:]-x,axis=2).min(axis=1)>1e-6]

    def model(train, labels, query):
        k = kernel(train,train)+.005*np.eye(len(train))
        cross = kernel(query,train)
        mean = labels.mean(axis=0)
        mu = mean + cross @ np.linalg.solve(k,labels-mean)
        sd = np.sqrt(np.maximum(1-(cross*np.linalg.solve(k,cross.T).T).sum(axis=1),.005))
        return mu,sd[:,None]

    prior = 0.
    residual = y
    if source is not None:
        prior,_ = model(*source,points)
        at_x,_ = model(*source,x)
        residual = y-at_x
    mu,sd = model(x,residual,points)
    scale = np.maximum(residual.std(axis=0),[.18,.12,.15,.12,.04])
    z = (np.array([np.log(.90),0,0,0,0])-prior-mu)/(sd*scale)
    cdf = .5*(1+np.vectorize(erf,otypes=[float])(z/np.sqrt(2)))
    return points[np.log(np.maximum(cdf,1e-15)).sum(axis=1).argmax()]
