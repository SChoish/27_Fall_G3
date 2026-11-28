"""Categorical PPO chooses a SPICE surrogate snapshot before paying for its proposal."""
from dataclasses import replace
import numpy as np

from alarm_pilot import SOURCE_NETS
from blind_width_policy import generate_candidates, kernel
from experiment import upper
from ppo_sizing import train
from surrogate_buffer import WINDOW, fit_head, merit
from surrogate_gate import HORIZON, SEEDS

ARMS=("random_snapshot","ppo_snapshot")


def options(net,z,y,lo,hi,candidates,goal):
    h=net.encode(z); test=net.encode(candidates)
    base=y[0]+net.readout(h); unadjusted=y[0]+net.readout(test)
    residual=y-base
    n=len(y)
    # The head fitted only on the nominal sample is identically the source head.
    ends=[0,n,*range(n-WINDOW,max(1,n-WINDOW-8),-2)]
    u=(z-lo)/(hi-lo)
    cross=kernel((candidates-lo)/(hi-lo),u)
    w=np.linalg.solve(kernel(u,u)+.004*np.eye(n),cross.T).T
    sigma=np.sqrt(np.maximum(0.,1-(w*cross).sum(1)))
    explore=.7*sigma[:,None]*np.maximum(y.std(0),.02)
    forecasts=[];ids=[]
    for end in ends:
        if end:
            head=fit_head(h,residual,end)
            pred=(unadjusted+test@head)+w@(residual-h@head)
        else:
            pred=unadjusted+w@residual
        conservative=pred-explore
        score=conservative[:,0]+4*np.maximum(conservative[:,1:],0).sum(1)
        ids.append(int(np.argmin(score)))
        forecasts.append(merit(pred,goal))
    m=np.tanh(np.asarray(forecasts)[:,ids]/.1)
    self_margin=m.diagonal()
    avg=m.mean(0)
    ends=np.asarray(ends)
    progress=min((n-5)/16,1.)
    gap=np.tanh(merit(y,goal).min()/.1)
    X=np.column_stack((ends==0,ends==n,ends/n,self_margin,avg,m.std(0),sigma[ids],
                       self_margin*gap,(ends==n)*progress,(ends==0)*float(goal==.9)))
    return ids,ends,X


def probabilities(logits):
    p=np.exp(logits-logits.max())
    return p/p.sum()


def learn(tasks,seed):
    theta=np.zeros(10); m=np.zeros(10);v=np.zeros(10);critic=np.zeros(4)
    step=warm_hits=wins=choices=0
    for epoch in range(20):
        samples=[]
        for ep in range(24):
            net,x,y,lo,hi=tasks[ep%4]
            goal=(.95,.90)[ep//4%2]
            rng=np.random.default_rng([seed,epoch,ep,11627])
            hist=[0,*rng.choice(np.arange(1,len(x)),4,replace=False).tolist()]
            uniforms=rng.random(HORIZON)
            if merit(y[hist],goal).min()<=0:
                warm_hits+=1
                continue
            available=[i for i in range(1,len(x)) if i not in hist]
            traj=[]
            for t in range(HORIZON):
                ids,ends,F=options(net,x[hist],y[hist],lo,hi,x[available],goal)
                p=probabilities(F@theta)
                a=min(int(np.searchsorted(p.cumsum(),uniforms[t])),len(p)-1)
                state=[1.,t/HORIZON,float(goal==.9),np.tanh(merit(y[hist],goal).min()/.1)]
                i=available.pop(ids[a])
                hit=merit(y[i:i+1],goal)[0]<=0
                hist.append(i)
                traj.append((F,a,p[a],state,len(set(ids))>1,float(hit)-1/HORIZON))
                if hit:
                    wins+=1
                    break
            ret=0.
            for F,a,old,state,distinct,reward in reversed(traj):
                ret+=reward
                if distinct:
                    samples.append((F,a,old,state,ret))
        if not samples:
            continue
        choices+=len(samples)
        V=np.array([s[3] for s in samples]);G=np.array([s[4] for s in samples])
        adv=G-V@critic
        critic=np.linalg.solve(V.T@V+.1*np.eye(4),V.T@G)
        adv=(adv-adv.mean())/max(adv.std(),1e-8)
        for _ in range(4):
            grad=np.zeros_like(theta)
            for (F,a,old,_,_),advantage in zip(samples,adv):
                p=probabilities(F@theta);ratio=p[a]/old
                if (advantage>=0 and ratio>1.2) or (advantage<0 and ratio<.8):
                    continue
                grad+=ratio*advantage*(F[a]-p@F)
            grad/=len(samples);step+=1
            m=.9*m+.1*grad;v=.999*v+.001*grad**2
            theta+=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return dict(weights=theta.tolist(),decisions=choices,warm_hits=warm_hits,policy_hits=wins)


def fit_direct(rows):
    tasks=[]
    for i,row in enumerate(rows):
        net=train(rows[:i]+rows[i+1:],only=("anon400",))["anon400"]
        hi=np.log(upper(replace(SOURCE_NETS[row["name"]],pdk=row["pdk"])))
        tasks.append((net,np.log(row["x"]),np.array(row["y"]),np.full_like(hi,np.log(.58)),hi))
    return [learn(tasks,s) for s in SEEDS]


def chooser_direct(net,models,goal,log=None):
    weights=np.mean([r["weights"] for r in models],axis=0) if models else None
    def select(arm,hist,labels,lo,hi,seed,unused):
        rng=np.random.default_rng([seed,len(hist),6481])
        if len(hist)<=4:
            return rng.uniform(lo,hi)
        z,y=map(np.asarray,(hist,labels))
        c=generate_candidates(z,y,lo,hi,rng)
        ids,ends,F=options(net,z,y,lo,hi,c,goal)
        if arm=="ppo_snapshot":
            a=int(np.argmax(F@weights))
        elif arm=="random_snapshot":
            a=int(np.random.default_rng([seed,len(hist),3001]).integers(len(ids)))
        else:
            raise ValueError(arm)
        if log is not None:
            log.append(dict(end=int(ends[a]),observed=len(hist),distinct=len(set(ids))))
        return c[ids[a]]
    return select
