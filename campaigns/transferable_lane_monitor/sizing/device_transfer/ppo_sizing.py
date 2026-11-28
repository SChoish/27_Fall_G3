"""On-policy clipped policy-gradient acquisition trained on measured source lookup episodes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys
import numpy as np

from adaptive_prior_run import TEST
from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_policy import evaluate_candidates, generate_candidates, rank_candidates
from blind_width_run import GOALS, run
from century_engine import enable
from experiment import arguments, upper
from raw_netlist_run import train
from reverse_width_transfer import DOMAINS as SKY_TARGETS, SEEDS as SKY_SEEDS

ROOT=Path(__file__).parent
ARCHIVE=ROOT/"ppo_sizing_measurements.zip"
DOMAINS={**{n:TEST[n] for n in ("adapt_lane_gf","adapt_branch_gf")},**SKY_TARGETS}
ARMS=("mlp200","ppo","bandit")
EPOCHS=32
NFEATURES=10


def action_features(pred,sig,score,goal,count):
    delta=pred.copy()
    delta[:,0]-=np.log(goal)
    scaled=np.tanh(delta/0.10)
    progress=min(count/48,1.)
    return -score/.12,np.column_stack((scaled,sig,sig*progress,sig*scaled[:,0],
                                       scaled[:,0]*np.max(scaled[:,1:],axis=1),
                                       sig*float(goal==.90)))


def observed_features(history, outcomes, candidates, lo, hi, goal):
    # Every coordinate is a log width relative to its own nominal value.
    # All reductions are invariant to simultaneous permutations of the devices.
    scores=outcomes[:,0]+4*np.maximum(outcomes[:,1:],0).sum(axis=1)
    anchor=history[np.argmin(scores)]
    relative=candidates/(hi-lo)
    mean=relative.mean(axis=1)
    spread=relative.std(axis=1)
    motion=(candidates-anchor)/(hi-lo)
    distance=np.sqrt(np.mean(motion**2,axis=1))
    novelty=np.sqrt(np.mean(((candidates[:,None,:]-history)/(hi-lo))**2,axis=2)).min(axis=1)
    unmet=np.maximum(outcomes[:,0]-np.log(goal),0)+np.maximum(outcomes[:,1:],0).sum(axis=1)
    gap=np.tanh(unmet.min()/.1)
    progress=min(len(history)/48,1.)
    features=np.column_stack((mean,spread,mean**2,motion.mean(axis=1),distance,novelty,
                              mean*gap,distance*gap,novelty*progress,mean*float(goal==.90)))
    return np.zeros(len(candidates)),features


def softmax(v):
    p=np.exp(v-v.max())
    return p/p.sum()


def hit(y,goal):
    return (y[...,0]<=np.log(goal)) & (y[...,1:]<=0).all(axis=-1)


def train_actor(rows,models,gamma,buffer_size=1):
    rng=np.random.default_rng(14019)
    actor=np.zeros(NFEATURES)
    adam1=np.zeros_like(actor);adam2=adam1.copy()
    critic=np.zeros(4);step=0;success=0;warm=0;actions=0;clipped=0;total_updates=0
    history=[critic.copy()]
    critic_age_counts=np.zeros(buffer_size,dtype=int)
    tasks=[]
    for row,model in zip(rows,models):
        c=replace(SOURCE_NETS[row["name"]],pdk=row["pdk"])
        hi=np.log(upper(c));lo=np.full(len(hi),np.log(.58))
        tasks.append((model,np.log(row["x"]),np.asarray(row["y"]),lo,hi))
    for epoch in range(EPOCHS):
        samples=[];returns=[];folds=[]
        for episode in range(24):
            model,x,y,lo,hi=tasks[episode%len(tasks)]
            goal=GOALS[(episode//len(tasks))%2]
            initial=rng.choice(np.arange(1,len(x)),size=4,replace=False)
            visited=[0,*initial]
            if hit(y[initial],goal).any():
                warm+=1
                continue
            available=[i for i in range(1,len(x)) if i not in visited]
            trajectory=[]
            for t in range(16):
                def features(_,pred,sig,score):
                    return action_features(pred,sig,score,goal,len(visited))
                if model is None:
                    base,F=observed_features(x[visited],y[visited],x[available],lo,hi,goal)
                else:
                    base,F=evaluate_candidates(x[visited],y[visited],lo,hi,x[available],network=model,decision=features)
                probs=softmax(base+F@actor)
                choice=int(rng.choice(len(available),p=probs))
                new=available.pop(choice)
                success_now=bool(hit(y[new],goal))
                sy=y[visited]
                gap=np.maximum(sy[:,0]-np.log(goal),0)+np.maximum(sy[:,1:],0).sum(1)
                state=np.array([1.,t/16,float(goal==.90),np.tanh(gap.min()/.1)])
                trajectory.append((base,F,choice,probs[choice],state,-.05+float(success_now)))
                visited.append(new)
                if success_now:
                    success+=1
                    break
            discounted=0.;ret=[]
            for transition in reversed(trajectory):
                discounted=transition[-1]+gamma*discounted;ret.append(discounted)
            returns.extend(reversed(ret));samples.extend(trajectory)
            folds.extend([(episode%8+episode//8+epoch)%2]*len(trajectory))
        states=np.stack([s[4] for s in samples]);returns=np.array(returns)
        adv=returns-states@critic
        if buffer_size>1:
            predictions=states@np.stack(history[::-1],axis=1)
            fold_ids=np.asarray(folds)
            for fold in (0,1):
                held=fold_ids==fold
                if not np.any(held) or np.all(held):
                    continue
                mse=np.mean((returns[~held,None]-predictions[~held])**2,axis=0)
                age=int(np.argmin(mse))
                critic_age_counts[age]+=1
                adv[held]=returns[held]-predictions[held,age]
        critic=np.linalg.solve(states.T@states+.1*np.eye(4),states.T@returns)
        history=(history+[critic])[-buffer_size:]
        adv=(adv-adv.mean())/max(adv.std(),1e-8)
        actions+=len(samples)
        for _ in range(4):
            grad=np.zeros_like(actor)
            for (base,F,choice,old,_,_),a in zip(samples,adv):
                p=softmax(base+F@actor);ratio=p[choice]/old
                saturated=(a>=0 and ratio>1.2) or (a<0 and ratio<.8)
                clipped+=int(saturated);total_updates+=1
                if not saturated:
                    grad+=a*ratio*(F[choice]-p@F)
            grad/=len(samples)
            step+=1;adam1=.9*adam1+.1*grad;adam2=.999*adam2+.001*grad**2
            actor+=.025*(adam1/(1-.9**step))/(np.sqrt(adam2/(1-.999**step))+1e-8)
    return dict(weights=actor.tolist(),episodes=EPOCHS*24,decisions=actions,warm_hits=warm,
                policy_hits=success,clip_fraction=clipped/total_updates,critic_age_counts=critic_age_counts.tolist())


def build(rows):
    networks=[train(rows[:i]+rows[i+1:],only=("anon400",))["anon400"] for i in range(len(rows))]
    return {arm:train_actor(rows,networks,gamma) for arm,gamma in (("ppo",1.),("bandit",0.))}


def chooser(model,weights,goal):
    def select(arm,hist,y,lo,hi,seed,unused):
        rng=np.random.default_rng([seed,len(hist),6481])
        if len(hist)<=4:return rng.uniform(lo,hi)
        if arm=="standalone":
            z=np.asarray(hist);outcomes=np.asarray(y)
            candidates=generate_candidates(z,outcomes,lo,hi,rng)
            _,features=observed_features(z,outcomes,candidates,lo,hi,goal)
            return candidates[np.argmax(features@weights[arm])]
        def decide(candidates,pred,sig,score):
            base,F=action_features(pred,sig,score,goal,len(hist))
            return candidates[np.argmax(base+F@weights[arm])]
        return rank_candidates(np.asarray(hist),np.asarray(y),lo,hi,rng,network=model,
                               decision=None if arm=="mlp200" else decide)
    return select


def load_baselines():
    runs=[]
    for file in ("unseen_elite_metric_measurements.zip","branch_transfer_measurements.zip",
                 "reverse_width_transfer_measurements.zip"):
        with ZipFile(ROOT/file) as f:
            runs.extend(r for r in json.loads(f.read("runs.json")) if r["arm"]=="mlp200")
    return runs


def seeds(name):
    return SKY_SEEDS if name in SKY_TARGETS else (81,82,83)


def policies(sources,training):
    result={}
    for p,rows in sources.items():
        assert len(rows)==4 and all(r["pdk"]==p and len(r["x"])==25 for r in rows)
        model=None if set(training[p])=={"standalone"} else train(rows,only=("anon400",))["anon400"]
        weights={a:np.array(v["weights"]) for a,v in training[p].items()}
        result[p]={g:chooser(model,weights,g) for g in GOALS}
    return result


def source_for(n):
    return "gf" if n in SKY_TARGETS else "sky"


def verify(sources,training,runs):
    choose=policies(sources,training);stats={}
    assert len(runs)==72
    for n,c in DOMAINS.items():
        p=source_for(n)
        assert p!=c.pdk
        for g in GOALS:
            subset=[r for r in runs if r["goal"]==g and r["name"]==n]
            _,stats[f"{n}_{g}"]=audit(sources[p],subset,ARMS,{a:(choose[p][g],None) for a in ARMS},
                                      {n:c},seeds(n),(g,))
    return stats


def main():
    if sys.argv[1:]==["--replay"]:
        with ZipFile(ARCHIVE) as f:
            sources,training,runs,expected=[json.loads(f.read(k+".json")) for k in ("source","training","runs","audit")]
        for p,rows in sources.items():
            recreated=build(rows)
            for arm in recreated:np.testing.assert_allclose(recreated[arm]["weights"],training[p][arm]["weights"],atol=1e-10)
        stats=verify(sources,training,runs)
        assert stats==expected
        return stats
    args=arguments();enable()
    rows=json.loads((args.out/"raw_netlist"/"source.json").read_text())
    sources={p:[r for r in rows if r["pdk"]==p] for p in ("sky","gf")}
    root=args.out/"ppo_sizing";root.mkdir(exist_ok=True,parents=True)
    training={p:build(rs) for p,rs in sources.items()}
    print("trained",training,flush=True)
    choose=policies(sources,training)
    def pair(job):
        n,a,s=job;rows=[]
        for g in GOALS:
            run((n,DOMAINS[n],a,s,g),args,None,root,choose[source_for(n)][g])
            rows.append(json.loads((root/f"{n}-{a}-{s}-{g}.json").read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs=load_baselines()+[r for block in pool.map(pair,[(n,a,s) for n in DOMAINS
                         for s in seeds(n) for a in ("ppo","bandit")]) for r in block]
    stats=verify(sources,training,runs)
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=9) as f:
        for key,data in (("source",sources),("runs",runs),("training",training),("audit",stats)):
            f.writestr(key+".json",json.dumps(data,separators=(",",":"),allow_nan=False))
    return stats

if __name__=="__main__":
    print(main())
