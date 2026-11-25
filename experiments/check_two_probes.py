"""Two calibrated probes on a frozen BO circuit with fixed versus adaptive second phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments.analyze_phase_feedback import GRID, score
from experiments.run_phase_feedback import CODES, curve, d, designs, suite, transient


def posterior(weights, samples, observation):
    distances = np.mean((samples-observation)**2,axis=1)
    logits = np.log(weights.clip(1e-300)) - distances/(2*.045**2)
    unnormalized = np.exp(logits-logits.max())
    return unnormalized/unnormalized.sum()


def route(bank, observe, second=None, one=False):
    probes = np.array([r['probes'] for r in bank])
    quality = np.array([r['quality'] for r in bank])
    weights = posterior(np.ones(len(bank)),probes[:,0,:,GRID[1]],observe(0,GRID[1]))
    # Bounded utility favours threshold crossings and avoids rewarding surplus eye height.
    utility = np.clip((quality-.08)/.04,-1,1)
    if one:
        return int(np.argmax(weights@utility)), None
    if second is None:
        def expected_value(code):
            responses = probes[:,1,:,code]
            value = [np.max(posterior(weights,responses,y)@utility) for y in responses]
            return weights@value
        second = max((GRID[0],GRID[2]),key=expected_value)
    weights = posterior(weights,probes[:,1,:,second],observe(1,second))
    return int(np.argmax(weights@utility)), second


def choose_fixed_direction(bank):
    results = []
    for code in (GRID[0],GRID[2]):
        held = []
        for r in bank:
            train = [p for p in bank if p['channel']!=r['channel']]
            final,_ = route(train,lambda k,j:np.array(r['probes'])[k,:,j],second=code)
            held.append(r['quality'][final])
        results.append((sum(v>=.08 for v in held),float(np.mean(held)),code))
    return max(results)[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    model=json.loads(args.model.read_text())
    bank=[p for p in model['bank'] if p['design']=='bo13']
    fixed=choose_fixed_direction(bank)
    name='bo13'
    root=49106
    x=designs()[name]
    proto=dict(seed=root,channels=12,x=x,model_sha=d.file_hash(args.model),source=d.file_hash(__file__),
               fixed_second=int(CODES[fixed]),first=int(CODES[GRID[1]]),probe_bits=64,test_bits=256)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=proto:
        raise ValueError('Incompatible protocol')
    d.write_json(path,proto)

    def trial(i):
        path=out/f'{i:03d}.json'
        if path.exists():
            return json.loads(path.read_text())
        cases=suite(root,i,256)
        probes=[curve(transient(x,c),c) for c in cases[:2]]
        def observe(step,code):
            return np.array(probes[step]['margin'])[:,code]
        phases={'fixed32':(GRID[1],None),'one':route(bank,observe,one=True),
                'two_fixed':route(bank,observe,second=fixed),'two_adaptive':route(bank,observe)}
        episode=dict(validation=[curve(transient(x,c),c) for c in cases[3:]])
        result=dict(channel=i, probe_pj_link=[3*p['bits']*p['energy'] for p in probes],
                    methods={k:dict(code=int(CODES[j]),second=int(CODES[z]) if z is not None else None,
                                 **score(episode,j)) for k,(j,z) in phases.items()})
        d.write_json(path,result)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(trial,range(proto['channels'])))
    d.write_json(out/'summary.json',dict(protocol=proto,physical_calls=len(rows)*6,rows=rows))


if __name__ == '__main__':
    main()
