"""Source-trained, binary PPO policy for choosing the next SPICE surrogate proposal."""
from dataclasses import replace
import numpy as np

from alarm_pilot import SOURCE_NETS
from blind_width_policy import evaluate_candidates, generate_candidates
from experiment import upper
from ppo_sizing import train
from surrogate_buffer import merit, select_head

EXPERTS = ("fresh", "source_guard")
SEEDS = (41, 42, 43)
HORIZON = 16
NFEATURES = 10


def proposals(net, z, y, lo, hi, candidates, goal):
    features = net.encode(z)
    test = net.encode(candidates)
    source_hist = y[0] + net.readout(features)
    source_test = y[0] + net.readout(test)
    idx, forecast = [], []
    for expert in EXPERTS:
        head, _ = select_head(expert, net, z, y, lo, hi, goal, 0)
        def prior(*unused):
            return source_hist + features @ head, source_test + test @ head
        def capture(_, pred, uncertainty, score):
            return int(np.argmin(score)), merit(pred, goal), uncertainty
        j, margin, sigma = evaluate_candidates(z, y, lo, hi, candidates,
                                                calibrated=prior, decision=capture)
        idx.append(j)
        forecast.append(margin)
    observed = merit(y, goal)
    normalized = np.tanh(np.array(forecast)[:, idx] / .1)
    state = np.array([1., min((len(z)-5)/16, 1.), float(goal==.90),
                      np.tanh(observed.min()/.1),
                      np.tanh((observed[:-3].min()-observed.min())/.1),
                      normalized[0,0], normalized[0,1], normalized[1,0], normalized[1,1],
                      sigma[idx[1]]-sigma[idx[0]]])
    return idx, state


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))


def train_actor(tasks, seed, gamma):
    # Episode-indexed streams keep the warmups and uniforms matched even after an early hit.
    weights = np.zeros(NFEATURES)
    critic = np.zeros(4)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    step = wins = choices = warm_hits = 0
    for epoch in range(20):
        states, actions, probs, targets = [], [], [], []
        for ep in range(24):
            task = tasks[ep%4]
            goal = (.95,.90)[ep//4%2]
            net,x,y,lo,hi = task
            rng = np.random.default_rng([seed,epoch,ep,11627])
            history = [0,*rng.choice(np.arange(1,len(x)),4,replace=False).tolist()]
            uniforms = rng.random(HORIZON)
            if np.min(merit(y[history],goal))<=0:
                warm_hits += 1
                continue
            available = [i for i in range(1,len(x)) if i not in history]
            trajectory = []
            for t in range(HORIZON):
                idx, state = proposals(net,x[history],y[history],lo,hi,x[available],goal)
                prob = float(sigmoid(state@weights))
                arm = int(uniforms[t]<prob)
                i = available.pop(idx[arm])
                success = bool(merit(y[i:i+1],goal)[0]<=0)
                trajectory.append((state,arm,prob,idx[0]!=idx[1],float(success)-1/HORIZON))
                history.append(i)
                if success:
                    wins += 1
                    break
            ret = 0.
            for s,a,p,distinct,r in reversed(trajectory):
                ret = r + gamma*ret
                if distinct:
                    states.append(s); actions.append(a); probs.append(p); targets.append(ret)
        if not states:
            continue
        X = np.stack(states)
        A, old, returns = map(np.asarray,(actions,probs,targets))
        choices += len(A)
        V = X[:,:4]
        adv = returns - V@critic
        critic = np.linalg.solve(V.T@V + .1*np.eye(4), V.T@returns)
        adv = (adv-adv.mean()) / max(adv.std(),1e-8)
        old_prob = np.where(A,old,1-old)
        for _ in range(4):
            p = sigmoid(X@weights)
            ratio = np.where(A,p,1-p)/old_prob
            active = ~(((adv>=0)&(ratio>1.2)) | ((adv<0)&(ratio<.8)))
            grad = X.T@(active*adv*ratio*(A-p))/len(X)
            step += 1
            m=.9*m+.1*grad
            v=.999*v+.001*grad**2
            weights+=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return dict(weights=weights.tolist(),decisions=choices,warm_hits=warm_hits,policy_hits=wins)


def fit(source):
    tasks=[]
    for i,row in enumerate(source):
        model=train(source[:i]+source[i+1:],only=("anon400",))["anon400"]
        c=replace(SOURCE_NETS[row["name"]],pdk=row["pdk"])
        hi=np.log(upper(c))
        tasks.append((model,np.log(row["x"]),np.asarray(row["y"]),np.full_like(hi,np.log(.58)),hi))
    return {arm:[train_actor(tasks,seed,gamma) for seed in SEEDS]
            for arm,gamma in (("ppo_gate",1.),("myopic_gate",0.))}


def chooser(net, fitted, goal, log=None):
    weights={arm:np.mean([r["weights"] for r in models],axis=0) for arm,models in fitted.items()}
    def choose(arm,hist,labels,lo,hi,seed,unused):
        rng=np.random.default_rng([seed,len(hist),6481])
        if len(hist)<=4:
            return rng.uniform(lo,hi)
        z,y=map(np.asarray,(hist,labels))
        candidates=generate_candidates(z,y,lo,hi,rng)
        ids,state=proposals(net,z,y,lo,hi,candidates,goal)
        if arm=="coin":
            pick=int(np.random.default_rng([seed,len(hist),3001]).integers(2))
        elif arm in EXPERTS:
            pick=EXPERTS.index(arm)
        else:
            pick=int(state@weights[arm]>0)
        if log is not None:
            log.append(dict(chosen=pick,distinct=ids[0]!=ids[1]))
        return candidates[ids[pick]]
    return choose
