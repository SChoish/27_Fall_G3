"""Search device sizes on measured timing-limited channels and lock before transfer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from dataclasses import replace

import numpy as np
from scipy.spatial.distance import cdist

from experiments.run_phase_feedback import CODES, curve, designs, d, suite, transient
from experiments.constrained_search import ConstrainedSearch
from experiments.run_three_lane import HIGH_DRIVE
from experiments.d2d_replay_search import propose


def pick_channels(bank):
    # Four development strata, including the hardest case by best attainable shared phase.
    relevant=[r for r in bank if r['design']=='bo11']
    ordered=sorted(relevant,key=lambda r:max(r['quality']))
    return [ordered[i]['channel'] for i in np.rint(np.linspace(0,len(ordered)-1,4)).astype(int)]


def init(seed):
    rng=np.random.default_rng(seed)
    old=designs()
    return [np.asarray(x) for x in (old['bo11'],HIGH_DRIVE,old['archive'],[1.8,7.5,3.5,2.2,2.8,3.],
                                     *(d.decode(u) for u in rng.uniform(size=(4,6))))]


def oracle(panels):
    cs=[]
    for packets in panels:
        margins=np.min([p['margin'] for p in packets],axis=0)
        best=int(np.argmax(margins.min(axis=0)))
        residuals=1-margins[:,best]/.08
        over=max(p['overshoot'] for p in packets)
        cs.append(dict(phase=int(CODES[best]),margins=margins[:,best].tolist(),overshoot=over,
                       residuals=[*residuals.tolist(),over/.2-1],
                       energy=float(np.mean([p['energy'] for p in packets]))))
    g=np.asarray([r['residuals'] for r in cs])
    return dict(valid=True,feasible=bool(g.max()<=0),energy_pj_bit=float(np.mean([r['energy'] for r in cs])),
                margins=np.min([r['margins'] for r in cs],axis=0).tolist(),
                overshoot_fraction=max(r['overshoot'] for r in cs), residuals=g.flatten().tolist(),cases=cs)


def rank(row):
    v=max(0.,max(row['oracle']['residuals']))
    return v,row['oracle']['energy_pj_bit']


def bracket(rows,seed):
    # Build measured feasible/infeasible segments in log sizing coordinates, then
    # predict the first feasible position on each with a small margin reserve.
    rng=np.random.default_rng(np.random.SeedSequence([seed,len(rows),8101]))
    X=np.array([d.encode(r['x']) for r in rows])
    G=np.array([r['oracle']['residuals'] for r in rows])
    E=np.array([r['oracle']['energy_pj_bit'] for r in rows])
    good=np.flatnonzero(G.max(axis=1)<=0)
    if len(good):
        pairs=[]
        for i in good:
            for j in np.flatnonzero(E<E[i]):
                change=G[j]-G[i]
                limiting=change>1e-6
                step=np.clip(np.min((-.04-G[i,limiting])/change[limiting]),.06,.9) if limiting.any() else .85
                u=X[i]+step*(X[j]-X[i])
                dist=cdist([u],X).min()
                if dist>.035:
                    pairs.append((E[i]+step*(E[j]-E[i]),u))
        if pairs and len(rows)%3!=1:
            return d.decode(min(pairs,key=lambda item:item[0])[1])
    anchor=X[min(range(len(rows)),key=lambda i:rank(rows[i]))]
    options=np.tile(anchor,(256,1))
    axes=rng.integers(6,size=(len(options),2))
    np.add.at(options,(np.arange(len(options))[:,None],axes),rng.uniform(-.22,.22,axes.shape))
    options=np.clip(options,0,1)
    distance=cdist(options,X)
    # Cheap local linear residual model; screen extrapolations with distance to data.
    A=np.c_[np.ones(len(X)),X-anchor]
    weights=np.exp(-np.sum((X-anchor)**2,axis=1)/.5)
    B=np.linalg.solve(A.T@(A*weights[:,None])+np.diag([.001]+[.08]*6),
                      A.T@(np.c_[np.log(E),G]*weights[:,None]))
    pred=np.c_[np.ones(len(options)),options-anchor]@B
    violation=np.maximum(0,pred[:,1:].max(axis=1)+.5*distance.min(axis=1))
    eligible=np.flatnonzero(distance.min(axis=1)>.012)
    best=eligible[np.lexsort((pred[eligible,0],violation[eligible]))[0]]
    return d.decode(options[best])


def execute(args):
    model=json.loads(args.model.read_text())
    channels=pick_channels(model['bank'])
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    keys=('bo','bracket','mlp_local')
    manifest=dict(seed=args.seed,channels=channels,root_seed=model['dev_seed'],model_sha=d.file_hash(args.model),
                  solver_sha=d.file_hash(__file__),methods=keys,short_calls=72,long_calls=24)
    manifest=json.loads(json.dumps(manifest))
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=manifest:
        raise ValueError('Campaign identity differs')
    d.write_json(output/'protocol.json',manifest)
    cache={}
    def evaluate(x,long,pool):
        key=(tuple(x),long)
        if key not in cache:
            cases=[]
            for c in channels:
                base=suite(model['dev_seed'],c,128)[0]
                for k in range(2 if long else 1):
                    cases.append(replace(base,seed=900000+100*c+10*int(long)+k,
                        bits=128 if long else 64,correlated=k==0))
            data=list(pool.map(lambda c:curve(transient(x,c),c),cases))
            n=2 if long else 1
            cache[key]=oracle([data[i:i+n] for i in range(0,len(cases),n)])
        return cache[key]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for method in keys:
            path=output/(method+'.json')
            if path.exists():
                finished=json.loads(path.read_text())
                for r in finished['rows']:
                    cache[(tuple(r['x']),False)]=r['oracle']
                for r in finished['validation']:
                    cache[(tuple(r['x']),True)]=r['oracle']
                continue
            history=[]
            bo=ConstrainedSearch(6,16,args.seed)
            for step in range(18):
                if step<8:
                    x=init(args.seed)[step]
                    u=np.clip(d.encode(x),0,1)
                elif method=='bo':
                    u=bo.ask()
                    x=d.decode(u)
                elif method=='bracket':
                    x=bracket(history,args.seed)
                else:
                    x=propose(history,[],args.seed,local_edits=True)
                row=evaluate(x,False,pool)
                history.append(dict(x=x.tolist(),oracle=row))
                if method=='bo':
                    bo.observe(u,row['energy_pj_bit'],row['residuals'])
            ordered=sorted(history,key=rank)
            # Reserve one slot for largest measured quality slack.
            safe=min(history,key=lambda r:max(r['oracle']['residuals']))
            shortlist=[]
            for r in ordered[:2]+[safe]+ordered:
                if r['x'] not in [s['x'] for s in shortlist]:
                    shortlist.append(r)
                if len(shortlist)==3:
                    break
            checked=[dict(x=r['x'],oracle=evaluate(r['x'],True,pool)) for r in shortlist]
            selected=min(checked,key=rank)
            d.write_json(path,dict(rows=history,validation=checked,selected=selected,
                       feasible_on_development=selected['oracle']['feasible'],calls=72+8*len(checked)))
    d.write_json(output/'locked.json',{k:json.loads((output/(k+'.json')).read_text())['selected'] for k in keys})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,default=4)
    execute(p.parse_args())
