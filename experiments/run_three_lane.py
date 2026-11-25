"""Run the three-lane difficulty calibration or equal-budget, held-out sizing comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

from experiments.d2d_three_lane import d, make_case, simulate
from experiments.constrained_search import ConstrainedSearch
from experiments.d2d_replay_search import propose
from experiments.robust_search_ablation import initial_designs, response

HIGH_DRIVE = [1.4, 6., 3., 2., 2., 3.]
CALIBRATION = dict(nominal=d.NOMINAL.tolist(), high_drive=HIGH_DRIVE,
                   bo3=[1.5510420522815604,3.4837503954530593,.9259558396027472,2.327681738929573,2.,5.000000000000001],
                   archive=[1.3945718754465593,2.064068978908268,.822009391179969,2.3126135544533652,4.,21.308311930369285])


def calibrate(folder, workers):
    jobs = [(r,c,k) for r in (2.,2.15,2.3) for c in (.05,.085) for k in CALIBRATION]
    def trial(job):
        rate, coupling, key = job
        result = simulate(CALIBRATION[key], make_case(rate, coupling, 12067, 64, True))
        return dict(rate=rate, coupling=coupling, design=key, **result)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(trial, jobs))
    d.write_json(folder/'calibration.json', dict(designs=CALIBRATION, rows=rows))


def panel(stage):
    if stage == 'test':
        # Distinct unseen combinations and a fixed sampling offset, after locking.
        return [make_case(r,c,180000+j*20+k,128,k%2==1,skew)
                for j,(r,c,skew) in enumerate([(2.12,.081,-.01),(2.19,.071,.012)])
                for k in range(8)]
    long = stage == 'long'
    return [make_case(r,c,170000+j*20+k,128 if long else 64,k%2==1)
            for j,(r,c) in enumerate([(2.10,.075),(2.15,.085)])
            for k in range(4 if long else 2)]


def run(folder, seed, workers):
    start = monotonic()
    suites = {name:panel(name) for name in ('short','long','test')}
    d.write_json(folder/'protocol.json', dict(seed=seed, high_drive=HIGH_DRIVE,
        source_sha256={p:d.file_hash(Path(__file__).with_name(p)) for p in
                       ['run_three_lane.py','d2d_three_lane.py','d2d_replay_search.py']},
        suites={k:[asdict(c) for c in v] for k,v in suites.items()}))
    cache = {}
    def evaluate(x, stage, executor):
        key = (tuple(x),stage)
        if key not in cache:
            rows = list(executor.map(lambda c: simulate(x,c), suites[stage]))
            cache[key] = d.aggregate(rows)
        return cache[key]

    initial = initial_designs('d2d', seed)
    initial[1] = np.asarray(HIGH_DRIVE)
    locked = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for method in ('bo','replay','replay_local'):
            rows = []
            bo = ConstrainedSearch(6, 4*len(suites['short']), seed)
            for step in range(18):
                if step < 8:
                    x = initial[step]
                    u = np.clip(d.encode(x),0,1)
                elif method == 'bo':
                    u = bo.ask()
                    x = d.decode(u)
                else:
                    x = propose(rows, [], seed, local_edits=method == 'replay_local')
                result = evaluate(x, 'short', executor)
                if method == 'bo':
                    bo.observe(u, *response('d2d',result))
                rows.append(dict(x=x.tolist(), oracle=result))
            # Every method gets the same three validation slots and return rule.
            shortlist = sorted(rows, key=lambda row:d.cost(row['oracle']))[:3]
            checked = [dict(x=row['x'], oracle=evaluate(row['x'],'long',executor)) for row in shortlist]
            valid = [row for row in checked if row['oracle']['feasible']]
            winner = min(valid, key=lambda row:row['oracle']['energy_pj_bit']) if valid else None
            locked[method] = dict(x=winner['x'] if winner else None,
                                  calls=18*len(suites['short'])+3*len(suites['long']))
            d.write_json(folder/(method+'.json'),dict(rows=rows,checked=checked))
        d.write_json(folder/'locked.json',locked)
        scores = {k:evaluate(v['x'],'test',executor) if v['x'] is not None else None for k,v in locked.items()}
    d.write_json(folder/'summary.json',dict(seed=seed, locked=locked, scores=scores,
        wall_seconds=monotonic()-start, unique_transients=sum(len(suites[stage]) for _,stage in cache)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.calibrate:
        calibrate(args.output, args.workers)
    else:
        run(args.output, args.seed, args.workers)
