"""Evaluate frozen archive-only, archive BO and archive MLP on a new common RC panel."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np

from experiments.robust_search_ablation import d2d, file_hash, new_suites, Measurer


def panel():
    templates = new_suites('d2d')[0][::2]
    rng = np.random.default_rng(261125995)
    fresh = []
    for i,template in enumerate(templates):
        ctx = template.context
        vdd,temp,r,c,cc = rng.uniform([.95,27,30,.30,.06],[1.02,65,40,.34,.10])
        lanes = [replace(lane, vdd=vdd, temperature_c=temp, resistance_ohm=r+2*j, capacitance_pf=c)
                 for j,lane in enumerate((ctx.lane0,ctx.lane1))]
        fresh.append(replace(template,context=replace(ctx,lane0=lanes[0],lane1=lanes[1],coupling_pf=cc)))
    return {split:[replace(base, name=f'{split}_{i}_{k}',bits=128,points_per_ui=128,
                  pattern='random', seed=261130000+1000*j+100*i+k)
                for i,base in enumerate(bases) for k in range(8)]
            for j,(split,bases) in enumerate([('same',templates),('fresh',fresh)])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('campaigns'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    locks = {}
    designs = {}
    for seed in range(6,11):
        path = args.root/f'neural_replay_archive_s{seed}'/'locked.json'
        locks[seed] = file_hash(path)
        for m,r in json.loads(path.read_text()).items():
            designs[seed,m] = r['selection']['design'] if r['selection'] else None
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    suites = panel()
    d2d.write_json(out/'protocol.json',dict(locks=locks,suites={k:[asdict(c) for c in v]
                   for k,v in suites.items()},designs=[dict(seed=k[0],method=k[1],x=v) for k,v in designs.items()]))
    details = {}
    costs = {}
    for split,cases in suites.items():
        measure = Measurer('d2d',cases,out/split,args.workers)
        for k,x in designs.items():
            result = measure(x) if x is not None else None
            details.setdefault(k,{})[split] = result
        costs[split] = measure.accounting()
    rows = []
    for (seed,m),splits in details.items():
        valid = all(r and r['valid'] for r in splits.values())
        rows.append(dict(seed=seed,method=m,all_pass=all(bool(r and r['feasible']) for r in splits.values()),
             energy=float(np.mean([r['energy_pj_bit'] for r in splits.values()])) if valid else None,
             splits=splits))
    d2d.write_json(out/'summary.json',dict(rows=rows,costs=costs))


if __name__=='__main__':
    main()
