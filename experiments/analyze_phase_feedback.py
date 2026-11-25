"""Learn calibration using development channels, then score policies through a three-query interface."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.run_phase_feedback import CODES, d

GRID = [int(np.flatnonzero(CODES==c)[0]) for c in (16,32,48)]
METHODS = ('fixed32','trained_static','grid','quadratic','prior_grid','prior_adaptive')


def load_episodes(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    out=[]
    for channel in range(protocol['channels']):
        for name in protocol['designs']:
            e=json.loads((folder/f'{channel:03d}-{name}.json').read_text())
            out.append(e)
    return protocol,out


def prototype(episode):
    # Known training pattern; three lane-specific signed aperture margins at a chosen code.
    return dict(design=episode['design'],channel=episode['channel'],
        probes=np.asarray([p['margin'] for p in episode['probe']]).round(7).tolist(),
        quality=np.min([np.asarray(p['margin']).min(axis=0) for p in episode['validation']],axis=0).round(7).tolist(),
        overshoot=max(p['overshoot'] for p in episode['validation']))


def choose(method,bank,observe):
    target=np.asarray([p['quality'] for p in bank])
    # Conservative development score; upper bound from invariant overshoot recorded separately.
    static=int(np.argmax(np.quantile(target,.2,axis=0)))
    if method=='fixed32':
        return GRID[1],[]
    if method=='trained_static':
        return static,[]
    ref=np.asarray([p['probes'] for p in bank])
    observed=[]
    weights=np.ones(len(bank))/len(bank)
    fitted=static
    for step in range(3):
        if method!='prior_adaptive':
            code=GRID[step]
        elif step==0:
            code=GRID[1]
        else:
            # Select the code whose response best distinguishes uncertain optimal phases.
            # Weighted covariance with the development target's argmax across channels.
            phase=np.argmax(target,axis=1).astype(float)
            centered=ref[:,step]-np.einsum('b,blc->lc',weights,ref[:,step])
            cov=np.einsum('b,b,blc->lc',weights,phase-weights@phase,centered)
            var=np.einsum('b,blc->lc',weights,centered**2)
            information=(cov**2/(var+.02**2)).mean(axis=0)
            information[[j for j,_ in observed]]=-np.inf
            code=int(np.argmax(information))
        observation=np.asarray(observe(step,code))
        observed.append((code,observation))
        if method.startswith('prior_'):
            distances=sum(np.mean((ref[:,s,:,j]-v)**2,axis=1) for s,(j,v) in enumerate(observed))
            logits=-distances/(2*.045**2)
            weights=np.exp(logits-logits.max())
            weights/=weights.sum()
            mu=weights@target
            sigma=np.sqrt(weights@((target-mu)**2))
            fitted=int(np.argmax(mu-.6*sigma))
    scores=np.asarray([v.min() for _,v in observed])
    winner=observed[int(np.argmax(scores))][0]
    if method=='quadratic':
        a,b,_=np.polyfit(CODES[GRID],scores,2)
        if a < -1e-7:
            winner=int(np.argmin(np.abs(CODES-np.clip(-b/(2*a),CODES[0],CODES[-1]))))
    if method.startswith('prior_'):
        winner=fitted
    return winner,[int(CODES[j]) for j,_ in observed]


def fit(dev):
    protocol,episodes=load_episodes(dev)
    return dict(dev_seed=protocol['root_seed'],dev_channels=protocol['channels'],codes=CODES.tolist(),
                design_vectors=protocol['designs'],bank=[prototype(e) for e in episodes],
                methods=list(METHODS),code_sha256=d.file_hash(Path(__file__)))


def score(episode,code):
    packets=episode['validation']
    min_margin=min(min(np.asarray(p['margin'])[:,code]) for p in packets)
    errors=sum(sum(np.asarray(p['errors'])[:,code]) for p in packets)
    overshoot=max(p['overshoot'] for p in packets)
    passed=sum(min(np.asarray(p['margin'])[:,code])>=.08 and sum(np.asarray(p['errors'])[:,code])==0
               and p['overshoot']<=.2 for p in packets)
    return dict(passes=int(passed),all_pass=bool(passed==len(packets)),margin=float(min_margin),errors=int(errors),
                overshoot=overshoot,energy=float(np.mean([p['energy'] for p in packets])))


def evaluate(model,test,cross_validate=False):
    protocol,episodes=load_episodes(test)
    if (protocol['root_seed']==model['dev_seed'])!=cross_validate or protocol['designs']!=model['design_vectors']:
        raise ValueError('Expected disjoint test or explicitly requested development cross-validation')
    rows=[]
    for episode in episodes:
        # Restrict references to the known manufactured design and omit the held-out development channel.
        bank=[r for r in model['bank'] if r['design']==episode['design'] and
              (not cross_validate or r['channel']!=episode['channel'])]
        record=dict(channel=episode['channel'],design=episode['design'],methods={})
        locked={}
        def observe(step,code):
            return np.asarray(episode['probe'][step]['margin'])[:,code]
        for method in METHODS:
            locked[method]=choose(method,bank,observe)
        # The oracle opens test labels only after every method has committed to its phase.
        oracle_scores=[score(episode,i) for i in range(len(CODES))]
        oracle=int(np.argmax([r['margin'] for r in oracle_scores]))
        locked['oracle']=(oracle,[])
        for method,(code,probes) in locked.items():
            record['methods'][method]=dict(code=int(CODES[code]),probes=probes,**oracle_scores[code])
        rows.append(record)
    counts={k:sum(r['methods'][k]['all_pass'] for r in rows) for k in (*METHODS,'oracle')}
    return dict(root_seed=protocol['root_seed'],channels=protocol['channels'],design_count=len(protocol['designs']),
                long_bits=protocol['long_bits'],physical_transients=len(rows)*7,passed=counts,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev',type=Path)
    p.add_argument('--test',type=Path)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True)
    args=p.parse_args()
    if args.dev:
        model=fit(args.dev)
        d.write_json(args.model,model)
        result=evaluate(model,args.dev,True)
    else:
        model=json.loads(args.model.read_text())
        if model['code_sha256']!=d.file_hash(Path(__file__)):
            raise ValueError('Policy differs from the development lock')
        result=evaluate(model,args.test)
    d.write_json(args.result,result)
    print(result['passed'])


if __name__=='__main__':
    main()
