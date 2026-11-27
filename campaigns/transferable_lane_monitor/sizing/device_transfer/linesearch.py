"""Exploratory controlled comparison on a new forked two-input circuit."""
import numpy as np
import experiment as exp
import proposal
from circuit import Circuit
from extend import NAME,SOURCE

HOLD="fork_pair_sky"
HELD=Circuit("sky",(-1,0,0,-2,3),(1.25,2.5,1.5,1.,2.5),(1,2,4),(.85,.65,1.25),1.5)
METHODS=("rule","cold_bo","gat_plus","role_line","gat_line")
ORIGINAL=proposal.propose

def line(method,models,c,hist,y,candidates,rng):
    if method in ("cold_bo","gat_plus"):
        return ORIGINAL("cold_bo" if method=="cold_bo" else "gat",models,c,hist,y,candidates,rng)
    terminal=np.repeat([i in c.outputs for i in range(len(c.parents))],2)
    if method=="rule":
        a,b=[(.84,1.),(.91,.86),(.96,.74),(.89,.94),(.95,.90)][(len(hist)-1)%5]
        x=np.where(terminal,a,b)
        if len(hist)>5:x*=np.exp(rng.normal(0,.045,len(x)))
        return np.clip(x,.58,exp.upper(c))
    f,e=exp.structure(c)
    if method=="gat_line":
        out,embedding=models["gat"].outputs(f,e)
    # Start with a broad electrical role direction. Subsequent directions can be per device.
    directions=[np.where(terminal,np.log(.80),0.),np.where(terminal,np.log(.90),np.log(.80))]
    if method=="gat_line":
        directions+=[out[:,5], .6*directions[0]+.4*out[:,5]]
    else:
        directions+=[np.where(terminal,0.,np.log(.76)), np.where(terminal,np.log(.86),np.log(.90))]
    if len(hist)==1: return np.exp(directions[0])
    # Reconstruct each tried direction's linearized feasible interval from dev measurements.
    t=len(hist)-1
    if t%2==1:
        last=np.log(hist[-1]);diff=y[-1]-y[0]
        bound=np.min(np.r_[1.6, -y[0,1:][diff[1:]>1e-6]/diff[1:][diff[1:]>1e-6]])
        x=np.clip(np.exp(last*np.clip(.94*bound,.2,1.6)),.58,exp.upper(c))
        if np.linalg.norm(hist-x,axis=1).min()>1e-3:return x
    if t<8:return np.clip(np.exp(directions[((t+1)//2)%4]),.58,exp.upper(c))
    candidates=np.vstack([candidates,*[np.clip(np.exp(d*np.arange(.25,1.6,.1)[:,None]),.58,exp.upper(c)) for d in directions]])
    logs=np.log(hist);d=np.log(candidates)
    if method=="gat_line":
        J=out[:,:5];feats=np.column_stack([embedding,np.ones(len(e))])/np.sqrt(len(e))
    else:
        J=np.zeros((len(e),5));feats=np.column_stack([terminal,~terminal])/np.sqrt(len(e))
    tx=logs@feats
    pred=y[0]+d@J+(d@feats)@tx.T@np.linalg.solve(tx@tx.T+.004*np.eye(len(y)),y-y[0]-logs@J)
    score=proposal.loss(pred)
    score[np.linalg.norm(candidates[:,None]-hist,axis=2).min(1)<1e-4]=np.inf
    return candidates[score.argmin()]

if __name__=="__main__":
    args=exp.arguments()
    exp.CIRCUITS[NAME]=SOURCE
    exp.CIRCUITS[HOLD]=HELD
    proposal.SOURCES=(*exp.SOURCES,NAME)
    proposal.propose=line
    exp.METHODS=METHODS
    models={s:proposal.train(args.out,s) for s in (21,22,23)}
    exp.search_domain(HOLD,args,models)
