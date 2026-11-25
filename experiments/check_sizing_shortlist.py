"""Change one long-validation slot while reusing the measured bracket search trace."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from experiments.run_phase_sizing import curve, d, oracle, rank, suite, transient


def shortlist(rows):
    order = sorted(rows, key=rank)
    safe = min(rows, key=lambda r: max(r['oracle']['residuals']))
    # A quarter of the normalized threshold means margin >= 0.10 and overshoot <= 0.15.
    reserve = [r for r in order if max(r['oracle']['residuals']) <= -.25 and r['x'] != order[0]['x']]
    picks = []
    for r in [order[0], *(reserve[:1] or order[1:2]), safe, *order]:
        if r['x'] not in [p['x'] for p in picks]:
            picks.append(r)
        if len(picks) == 3:
            break
    if len(picks) != 3:
        raise ValueError('Need three different designs')
    return picks


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.source/'protocol.json').read_text())
    arms = {k: json.loads((args.source/f'{k}.json').read_text()) for k in ('bo','bracket')}
    locks = {k: v['selected'] for k,v in arms.items()}
    candidates = shortlist(arms['bracket']['rows'])
    cases = []
    for channel in protocol['channels']:
        base = suite(protocol['root_seed'],channel,128)[0]
        cases.extend(replace(base,seed=900010+100*channel+k,bits=128,correlated=k==0) for k in range(2))
    checked = []
    cached = {tuple(r['x']):r for r in arms['bracket']['validation']}
    extra = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for candidate in candidates:
            x = candidate['x']
            key = tuple(x)
            if key in cached:
                checked.append(cached[key])
                continue
            packets = list(pool.map(lambda c:curve(transient(x,c),c),cases))
            checked.append(dict(x=x,oracle=oracle([packets[i:i+2] for i in range(0,len(cases),2)])))
            extra += len(cases)
    locks['reserve'] = min(checked,key=rank)
    d.write_json(output/'protocol.json', dict(seed=protocol['seed'], extra_calls=extra,
        short_calls=72,long_calls=24,source_checksums={k:d.file_hash(args.source/f'{k}.json') for k in arms},
        checked=checked))
    d.write_json(output/'locked.json', locks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers',type=int,default=2)
    run(parser.parse_args())
