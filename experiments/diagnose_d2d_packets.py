"""Separate packet coverage from transient resolution on frozen returned designs."""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from experiments.robust_search_ablation import adc, d2d, file_hash
from experiments.report_robust_search import read


@dataclass(frozen=True)
class PrefixCase(d2d.Case):
    master_bits: int = 128

    def payload(self):
        if self.pattern != 'random' or not 0 < self.bits <= self.master_bits:
            raise ValueError('Expected a random prefix of the master packet')
        # Drawing (2,64) would change lane 1, so slice a shared (2,128) draw.
        master = np.random.default_rng(self.seed).integers(0, 2, (2, self.master_bits), dtype=np.uint8)
        return master[:, :self.bits]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output')
    summary = read(args.campaign/'summary.json')
    protocol = read(args.campaign/'protocol.json')
    case = d2d.Case.from_dict(protocol['test']['same_channels'][6])
    prefix = PrefixCase(**{k: v for k, v in case.__dict__.items()})
    np.testing.assert_array_equal(prefix.payload(), case.payload())
    cases = [replace(prefix, name=f'bits{bits}_ppui{ppui}', bits=bits, points_per_ui=ppui)
             for bits in (64, 128) for ppui in (64, 128)]
    evaluator = d2d.Evaluator(cases, args.output/'measurements', workers=4)
    result = []
    for seed, method in [('0', 'legacy'), ('0', 'legacy_validated'), ('2', 'legacy'), ('1', 'legacy')]:
        design = summary['results'][seed]['results'][method]['selection']['design']
        measured = evaluator.evaluate(design)
        result.append(dict(seed=int(seed), method=method, design=design, measurement=measured))
        print(seed, method, [(c['case']['name'], min(c['margins']), c['feasible'])
                             for c in measured['cases']], flush=True)
    adc.write_json(args.output/'summary.json', dict(source_sha256=file_hash(__file__),
        source_campaign_sha256=file_hash(args.campaign/'summary.json'), results=result,
        interpretation='Same exact two-lane master packet; 64-bit prefix versus full 128, crossed with solver resolution',
        cost=evaluator.accounting()))


if __name__ == '__main__':
    main()
