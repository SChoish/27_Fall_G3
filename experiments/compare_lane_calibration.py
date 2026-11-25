"""Compare three deployable common-clock rules on a frozen BO design and fresh packets."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from experiments.analyze_phase_feedback import choose, score
from experiments.confirm_phase_sizing import lane_quadratic
from experiments.run_phase_feedback import CODES, curve, d, designs, suite, transient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = 49104
    design = designs()['bo11']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(seed=root, design=design, channels=8, probe_bits=64, test_bits=128,
                    sources={p.name: d.file_hash(p) for p in (Path(__file__),
                        Path(__file__).with_name('confirm_phase_sizing.py'),
                        Path(__file__).with_name('analyze_phase_feedback.py'),
                        Path(__file__).with_name('run_phase_feedback.py'))})
    protocol = output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != manifest:
        raise ValueError('Protocol changed')
    d.write_json(protocol, manifest)

    def trial(i):
        path = output/f'{i:03d}.json'
        if path.exists():
            return json.loads(path.read_text())
        cases = suite(root, i, 128)
        probes = [curve(transient(design, c), c) for c in cases[:3]]

        def observe(step, code):
            return np.asarray(probes[step]['margin'])[:, code]

        selected = {k: choose(k, [], observe)[0] for k in ('fixed32', 'grid', 'quadratic')}
        selected['lane_quadratic'] = lane_quadratic(observe)
        validation = dict(validation=[curve(transient(design, c), c) for c in cases[3:]])
        rows = {k: dict(code=int(CODES[v]), **score(validation, v)) for k, v in selected.items()}
        result = dict(channel=i, line='RLC' if i%2 else 'RC', methods=rows)
        d.write_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial, range(manifest['channels'])))
    d.write_json(output/'summary.json', dict(protocol=manifest, physical_calls=7*len(rows), rows=rows))


if __name__ == '__main__':
    main()
