"""Paired ngspice experiment: BO reference, neural replay with and without long feedback."""
import argparse
from dataclasses import replace, asdict
from pathlib import Path
import subprocess
import time

import numpy as np
from experiments.robust_search_ablation import (Measurer, initial_designs, shortlist,
    validated_selection, new_suites, response, file_hash, ROOT, d2d)
from experiments.constrained_search import ConstrainedSearch
from experiments.d2d_replay_search import propose, targets


def run_arm(method, seed, measure, validate, output):
    bo = ConstrainedSearch(6, 12, seed)
    rows, checked, measured, pairs = [], [], [], []
    initial = initial_designs('d2d', seed)
    schedule = (18, 18, 18) if method == 'bo' else (14, 16, 18)
    for step in range(1, 19):
        if step <= 8:
            x = initial[step-1]
            point = np.clip(d2d.encode(x), 0, 1)
        elif method == 'bo':
            point = bo.ask()
            x = d2d.decode(point)
        else:
            x = propose(rows, pairs if method == 'replay_feedback' else [], seed)
        result = measure(x)
        if method == 'bo':
            bo.observe(point, *response('d2d', result))
        rows.append(dict(x=x.tolist(), oracle=result))
        if step in schedule:
            if method == 'bo':
                candidates = shortlist('d2d', rows)
            else:
                eligible = [row for row in rows if row['oracle']['feasible'] and
                            row['x'] not in [c['x'] for c in checked]]
                key = (lambda r: max(response('d2d',r['oracle'])[1])) if step == 18 else (
                    lambda r: r['oracle']['energy_pj_bit'])
                candidates = sorted(eligible, key=key)[:1]
            for row in candidates:
                high = validate(row['x'])
                checked.append(row)
                measured.append(high)
                if high['valid']:
                    pairs.append((d2d.encode(row['x']), targets(high)[1:]-targets(row['oracle'])[1:]))
        d2d.write_json(output/'trace.json', dict(rows=rows, checked=checked, measured=measured))
    return dict(selection=validated_selection('d2d', checked, measured),
                logical_cases=72+8*len(checked), case_payload_bits=4608+1024*len(checked))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--packet-offset', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    short, val, old_test = new_suites('d2d')
    long = [replace(c, bits=128) for c in val]
    tests = {name:[replace(c, seed=c.seed+args.packet_offset) for c in cases]
             for name,cases in old_test.items()}
    if len({c.seed for cases in tests.values() for c in cases}) != 16 or any(
            c.seed in {k.seed for k in short+long+sum(old_test.values(),[])}
            for cases in tests.values() for c in cases):
        raise ValueError('Final packets must be distinct from old and development packets')
    methods = ('bo', 'replay', 'replay_feedback')
    s = Measurer('d2d', short, out/'short', args.workers)
    h = Measurer('d2d', long, out/'long', args.workers)
    d2d.write_json(out/'protocol.json',dict(seed=args.seed, methods=methods,
        suites={k:[asdict(c) for c in cases] for k,cases in dict(short=short,long=long,**tests).items()},
        source={p:file_hash(ROOT/p) for p in ['experiments/run_d2d_replay.py','experiments/d2d_replay_search.py']},
        git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        simulator=s.backend.simulator_version, binary=file_hash(s.backend.binary)))
    started = time.monotonic()
    locked = {m:run_arm(m,args.seed,s,h,out/m) for m in methods}
    d2d.write_json(out/'locked.json', locked)
    scores = {}
    for split,cases in tests.items():
        eval_test = Measurer('d2d', cases, out/split, args.workers)
        scores[split] = {m:eval_test(v['selection']['design']) if v['selection'] else None
                         for m,v in locked.items()}
    d2d.write_json(out/'summary.json', dict(seed=args.seed, locked=file_hash(out/'locked.json'),
        results=scores, short=s.accounting(), long=h.accounting(), elapsed=time.monotonic()-started))


if __name__ == '__main__':
    main()
