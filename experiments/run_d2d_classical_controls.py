"""Fresh D2D random and steady-state GA controls at the same short/long budgets."""
import argparse
from pathlib import Path
import json

import numpy as np
from experiments.robust_search_ablation import (
    d2d, initial_designs, Measurer, shortlist, validated_selection, file_hash)


def draw_ga(rows, rng):
    # Binary tournament on the eight best observed designs; bounded blend + mutation.
    survivors = sorted(rows, key=lambda row: d2d.cost(row['oracle']))[:8]
    parents = []
    for _ in range(2):
        i, j = rng.integers(len(survivors), size=2)
        parents.append(d2d.encode(survivors[min(i,j)]['x']))
    blend = rng.uniform(-.25, 1.25, size=6)
    mutation = rng.uniform(size=6) < 1/6
    mutation[rng.integers(6)] = True
    return np.clip(blend*parents[0]+(1-blend)*parents[1]+mutation*rng.normal(0,.12,6), 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.reference/'protocol.json').read_text())
    suites = {k:[d2d.Case.from_dict(c) for c in cases] for k,cases in protocol['suites'].items()}
    short = Measurer('d2d',suites.pop('short'),out/'short',args.workers)
    long = Measurer('d2d',suites.pop('long'),out/'long',args.workers)
    if file_hash(short.backend.binary) != protocol['binary']:
        raise ValueError('Reference ngspice binary mismatch')
    seed = protocol['seed']
    locked = {}
    for name in ('random','ga'):
        rows = []
        rng = np.random.default_rng(np.random.SeedSequence([seed, 924]))
        for x in initial_designs('d2d',seed):
            rows.append(dict(x=x.tolist(),oracle=short(x)))
        while len(rows)<18:
            u = rng.uniform(size=6) if name=='random' else draw_ga(rows,rng)
            x = d2d.decode(u)
            if any(np.array_equal(x,r['x']) for r in rows):
                continue
            rows.append(dict(x=x.tolist(), oracle=short(x)))
        candidates = shortlist('d2d',rows)
        measured = [long(row['x']) for row in candidates]
        d2d.write_json(out/name/'trace.json',dict(rows=rows,checked=candidates,measured=measured))
        locked[name] = dict(selection=validated_selection('d2d',candidates,measured),
                            logical_cases=72+8*len(measured),case_payload_bits=4608+1024*len(measured))
    d2d.write_json(out/'locked.json',locked)
    results = {}
    costs = {}
    for split,cases in suites.items():
        evaluator = Measurer('d2d',cases,out/split,args.workers)
        results[split] = {m:evaluator(v['selection']['design']) if v['selection'] else None
                          for m,v in locked.items()}
        costs[split] = evaluator.accounting()
    d2d.write_json(out/'summary.json',dict(seed=seed,results=results,short=short.accounting(),
        long=long.accounting(),test_costs=costs,protocol=file_hash(args.reference/'protocol.json'),
        locked=file_hash(out/'locked.json')))


if __name__ == '__main__':
    main()
