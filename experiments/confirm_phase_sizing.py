"""Blind common-channel transfer for locked sizing candidates using a deployable three-probe rule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np
from experiments.analyze_phase_feedback import choose,score
from experiments.run_phase_feedback import CODES,curve,d,designs,suite,transient


def run(args):
    if args.root_seed in (49101,49102):
        raise ValueError('Use new root for confirmation')
    out=args.output
    out.mkdir(exist_ok=True,parents=True)
    candidates={'previous_bo11':designs()['bo11']}
    locks={}
    for src in args.runs:
        current=json.loads((src/'locked.json').read_text())
        seed=json.loads((src/'protocol.json').read_text())['seed']
        locks[str(seed)]=d.file_hash(src/'locked.json')
        candidates.update({f'{k}_s{seed}':v['x'] for k,v in current.items()})
    unique=list(dict.fromkeys(tuple(x) for x in candidates.values()))
    members={k:unique.index(tuple(x)) for k,x in candidates.items()}
    protocol=dict(locks=locks,channels=args.channels,root_seed=args.root_seed,bits=args.bits,
                  methods=['fixed32','grid','quadratic','oracle'],designs=[list(u) for u in unique],members=members,
                  code=d.file_hash(__file__))
    manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('Existing panel differs')
    d.write_json(manifest,protocol)
    jobs=[(i,j) for i in range(args.channels) for j in range(len(unique)) if not (out/f'{i:03d}-{j}.json').exists()]
    def trial(job):
        i,j=job
        panel=suite(args.root_seed,i,args.bits)
        probes=[curve(transient(unique[j],c),c) for c in panel[:3]]
        def observe(step,code):
            return np.asarray(probes[step]['margin'])[:,code]
        selected={k:choose(k,[],observe)[0] for k in ('fixed32','grid','quadratic')}
        episode=dict(validation=[curve(transient(unique[j],c),c) for c in panel[3:]])
        measurements=[score(episode,k) for k in range(len(CODES))]
        selected['oracle']=int(np.argmax([r['margin'] for r in measurements]))
        values={k:dict(code=int(CODES[v]),**measurements[v]) for k,v in selected.items()}
        d.write_json(out/f'{i:03d}-{j}.json',dict(channel=i,design=j,methods=values))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(trial,jobs))
    rows=[json.loads((out/f'{i:03d}-{j}.json').read_text()) for i in range(args.channels) for j in range(len(unique))]
    d.write_json(out/'summary.json',dict(protocol=protocol,rows=rows,physical_calls=len(rows)*7))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--channels',type=int,default=16)
    p.add_argument('--root-seed',type=int,default=49103)
    p.add_argument('--bits',type=int,default=512)
    p.add_argument('--workers',type=int,default=7)
    run(p.parse_args())
