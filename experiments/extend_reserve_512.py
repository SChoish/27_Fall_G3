"""Measure only missing frozen reserve designs on the primary 512-bit grid benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from experiments.analyze_phase_feedback import choose, score
from experiments.run_phase_feedback import CODES, curve, d, suite, transient


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    base=read(args.primary/'protocol.json')
    if base['root_seed']!=49103 or base['bits']!=512:
        raise ValueError('Expected primary 512-bit panel')
    designs=list(base['designs'])
    members=dict(base['members'])
    hashes={}
    for directory in args.runs:
        seed=read(directory/'protocol.json')['seed']
        path=directory/'locked.json'
        x=read(path)['reserve']['x']
        hashes[str(seed)]=d.file_hash(path)
        if x not in designs:
            designs.append(x)
        members[f'reserve_s{seed}']=designs.index(x)
    proto=dict(primary_protocol=d.file_hash(args.primary/'protocol.json'),locks=hashes,
               designs=designs,members=members,channels=base['channels'],root_seed=base['root_seed'],
               bits=base['bits'],source=d.file_hash(__file__),policy='grid')
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists() and read(path)!=proto:
        raise ValueError('Incompatible locked extension')
    d.write_json(path,proto)

    def trial(job):
        i,j=job
        path=out/f'{i:03d}-{j}.json'
        if path.exists():
            return read(path)
        cases=suite(proto['root_seed'],i,proto['bits'])
        probes=[curve(transient(designs[j],c),c) for c in cases[:3]]
        phase,_=choose('grid',[],lambda step,code:np.asarray(probes[step]['margin'])[:,code])
        valid=[curve(transient(designs[j],c),c) for c in cases[3:]]
        grid=dict(code=int(CODES[phase]),**score(dict(validation=valid),phase))
        row=dict(channel=i,design=j,methods=dict(grid=grid))
        d.write_json(path,row)
        return row

    jobs=[(i,j) for i in range(proto['channels']) for j in range(len(base['designs']),len(designs))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(trial,jobs))
    d.write_json(out/'extra.json',dict(protocol=proto,physical_calls=7*len(rows),rows=rows))


if __name__ == '__main__':
    main()
