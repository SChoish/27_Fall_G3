"""Long-packet validation follow-up; use new final packets after stage-one diagnosis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from experiments.report_robust_search import read
from experiments.robust_search_ablation import Measurer, adc, d2d, file_hash, new_suites, validated_selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output directory')
    previous = read(args.campaign/'summary.json')
    old_protocol = read(args.campaign/'protocol.json')
    if old_protocol['kind'] != 'd2d' or old_protocol['budget'] != 24:
        p.error('Expected the completed 24-query D2D ablation')
    _, short_validation, tests = new_suites('d2d')
    validation = [replace(c, bits=128) for c in short_validation]
    tests = {split: [replace(c, seed=c.seed+100000, name=c.name+'_stage2') for c in cases]
             for split, cases in tests.items()}
    args.output.mkdir(parents=True)
    adc.write_json(args.output/'protocol.json', dict(
        parent_summary_sha256=file_hash(args.campaign/'summary.json'),
        source_sha256=file_hash(__file__), validation=[asdict(c) for c in validation],
        tests={k: [asdict(c) for c in v] for k, v in tests.items()},
        reason='Stage-one test is now diagnostic/development evidence. All stage-two test seeds are fresh.',
        comparison='Same 96 logical case calls; long validation has twice the simulated bits, so not equal wall time.',
        methods=['legacy', 'legacy_validated_64', 'legacy_validated_128']))

    def select(seed):
        folder = args.output/f'seed_{seed}'
        prior = previous['results'][seed]['results']
        candidates = read(args.campaign/f'seed_{seed}'/'legacy_shortlist_locked.json')
        adc.write_json(folder/'shortlist_locked.json', candidates)
        evaluate = Measurer('d2d', validation, folder/'validation', workers=2)
        measured = [evaluate(row['x']) for row in candidates]
        selection = validated_selection('d2d', candidates, measured)
        adc.write_json(folder/'validation.json', measured)
        result = dict(returns=dict(legacy=prior['legacy']['selection'],
            legacy_validated_64=prior['legacy_validated']['selection'], legacy_validated_128=selection),
            long_validation_cost=evaluate.accounting(),
            logical_case_calls=dict(legacy=96, legacy_validated_64=96,
                                    legacy_validated_128=18*4+len(measured)*8))
        adc.write_json(folder/'returns.json', result)
        print(seed, 'long validation passed', sum(r['feasible'] for r in measured), '/', len(measured), flush=True)
        return result

    seeds = list(previous['results'])
    with ThreadPoolExecutor(3) as pool:
        selections = dict(zip(seeds, pool.map(select, seeds)))
    lock = args.output/'selections_locked.json'
    adc.write_json(lock, selections)
    digest = file_hash(lock)
    print('Locked all stage-two returns', flush=True)

    def test(seed):
        evaluators = {split: Measurer('d2d', cases, args.output/'test'/seed/split, workers=2)
                      for split, cases in tests.items()}
        result = {}
        for method, selection in selections[seed]['returns'].items():
            result[method] = dict(selection=selection, test={} if selection is None else
                {split: evaluate(selection['design']) for split, evaluate in evaluators.items()})
        return dict(results=result, cost={k: v.accounting() for k, v in evaluators.items()})

    with ThreadPoolExecutor(3) as pool:
        final = dict(zip(seeds, pool.map(test, seeds)))
    assert file_hash(lock) == digest
    adc.write_json(args.output/'summary.json', dict(results=final,
        locked_selection_sha256=digest, protocol_sha256=file_hash(args.output/'protocol.json')))
    print('Completed', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
