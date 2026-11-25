"""Audit common initialization and summarize locked designs on identical final packets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from d2d_sizing.circuit import write_json


def read(path):
    return json.loads(path.read_text())


def audit(root, seeds):
    output = []
    for seed in seeds:
        first_initial = None
        for group in ('pilot', 'controls'):
            path = root / f'neural_replay_{group}_s{seed}'
            scores = read(path/'summary.json')
            locked = read(path/'locked.json')
            if scores['locked'] != hashlib.sha256((path/'locked.json').read_bytes()).hexdigest():
                raise ValueError(f'Invalid selection lock at {path}')
            for method, result in locked.items():
                selection = result['selection']
                trace = read(path/method/'trace.json')
                initial = [row['x'] for row in trace['rows'][:8]]
                if first_initial is None:
                    first_initial = initial
                if initial != first_initial or len(trace['rows']) != 18:
                    raise ValueError('Unmatched initialization or short budget')
                n_long = len(trace['checked'])
                if not 0 <= n_long <= 3 or result['logical_cases'] != 72 + 8*n_long:
                    raise ValueError('Invalid long budget')
                entry = dict(seed=seed, method=method, returned=selection is not None,
                             calls=result['logical_cases'],case_payload_bits=result['case_payload_bits'],
                             design=None if selection is None else selection['design'],
                             all_pass=False, energy=None, cases=[])
                for split, methods in scores['results'].items():
                    data = methods[method]
                    if data is None:
                        if selection is not None:
                            raise ValueError('Missing final evaluation')
                        continue
                    for c in data['cases']:
                        entry['cases'].append(dict(split=split, packet_seed=c['case']['seed'],
                            valid=c['valid'], feasible=c['feasible'], energy=c.get('energy_pj_bit'),
                            margins=c.get('margins'), errors=c.get('errors'), overshoot=c.get('overshoot_fraction')))
                if selection is not None:
                    if len(entry['cases']) != 16:
                        raise ValueError('Incomplete panel')
                    entry['all_pass'] = all(c['feasible'] for c in entry['cases'])
                    if all(c['valid'] for c in entry['cases']):
                        entry['energy'] = float(np.mean([c['energy'] for c in entry['cases']]))
                output.append(entry)
    panels = {tuple((c['split'],c['packet_seed']) for c in r['cases']) for r in output if r['returned']}
    if len(panels) > 1:
        raise ValueError('Final packet panels differ')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('campaigns'))
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.root, args.seeds))
