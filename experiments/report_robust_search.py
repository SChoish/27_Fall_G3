"""Check completed ablations, publish compact measurements, and summarize results."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robust_search_ablation import energy, file_hash, shortlist, validated_selection, METHODS
from experiments.benchmark_constrained_search import response
from experiments.constrained_search import Surrogate
from adc_driver import circuit as adc
from d2d_sizing import circuit as d2d
from adc_driver.circuit import write_json


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != 'cycles'}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def forecast_diagnostics(kind, trace, cls):
    """Prequential forecasts: fit only rows preceding each measured proposal.

    The legacy value is a heuristic score. Neither value estimates unseen-packet
    reliability; these diagnostics concern measured development constraints.
    """
    circuit = adc if kind == 'adc' else d2d
    nconstraints = len(trace[0]['oracle']['contexts']) if kind == 'adc' else 3*len(trace[0]['oracle']['cases'])
    search = cls(len(circuit.NAMES), nconstraints, 0)
    predictions = []
    for index, row in enumerate(trace):
        point = np.clip(circuit.encode(row['x']), 0, 1)
        if index >= 8:
            valid = [j for j, r in enumerate(search.results) if r is not None]
            if len(valid) >= 3:
                x = np.asarray(search.points)[valid]
                y = np.asarray([search.results[j] for j in valid])
                mean, std = Surrogate(x, search.model_targets(y), search.radius).predict(point[None, :])
                score = float(np.exp(search.feasibility_log_score(mean, std)[0]))
                predictions.append(dict(query=index+1, predicted_score=score,
                    valid=row['oracle']['valid'], feasible=row['oracle']['feasible']))
            search.pending = point.copy()
        search.observe(point, *response(kind, row['oracle']))
        if 'radius' in row:
            assert np.isclose(search.radius, row['radius'])
    confident = [r for r in predictions if r['predicted_score'] >= .9]
    return dict(predictions=predictions, score_at_least_point9=len(confident),
                failures_among_these=sum(not r['feasible'] for r in confident),
                scope='Development feasibility, not unseen-packet generalization; legacy is a heuristic score')


def check_and_summarize(root, destination):
    protocol = read(root/'protocol.json')
    summary = read(root/'summary.json')
    lock = read(root/'selections_locked.json')
    assert summary['protocol_sha256'] == file_hash(root/'protocol.json')
    assert summary['locked_selection_sha256'] == file_hash(root/'selections_locked.json')
    assert lock['protocol_sha256'] == summary['protocol_sha256']
    for path, digest in protocol['source_sha256'].items():
        assert file_hash(root/'source'/path) == digest, path
    kind = protocol['kind']
    rows, failures, forecasts = [], [], {}
    for seed, run in summary['results'].items():
        for method, record in run['results'].items():
            original = lock['runs'][seed]['returns'][method]
            assert {k: v for k, v in record.items() if k != 'test'} == original
            base = method.removesuffix('_validated')
            trace = read(root/f'seed_{seed}'/f'{base}.json')
            if not method.endswith('_validated'):
                forecasts[f'{seed}/{method}'] = forecast_diagnostics(kind, trace, METHODS[base])
            assert len(trace) == protocol['budget']
            assert record['logical_case_calls'] <= protocol['logical_case_budget']
            selection = record['selection']
            if method.endswith('_validated'):
                candidates = shortlist(kind, trace[:record['search_queries']])
                measured = read(root/f'seed_{seed}'/f'{base}_validation.json')
                assert candidates == read(root/f'seed_{seed}'/f'{base}_shortlist_locked.json')
                assert validated_selection(kind, candidates, measured) == selection
                assert record['logical_case_calls'] == (record['search_queries']*len(protocol['development'])
                    +len(candidates)*len(protocol['validation']))
            elif selection is not None:
                eligible = [r for r in trace if r['oracle']['feasible']]
                assert selection['design'] == min(eligible, key=lambda r: energy(kind, r['oracle']))['x']
            case_key = 'contexts' if kind == 'adc' else 'cases'
            split_stats = {}
            for split, result in record['test'].items():
                cases = result[case_key]
                split_stats[split] = dict(pass_all=result['feasible'], passed=sum(c['feasible'] for c in cases),
                    total=len(cases), energy=energy(kind, result) if result['valid'] else None)
                for index, case in enumerate(cases):
                    if not case['feasible']:
                        failures.append(dict(seed=int(seed), method=method, split=split, case=index,
                                             measurement=compact(case)))
            rows.append(dict(seed=int(seed), method=method, returned=selection is not None,
                pass_all=selection is not None and all(r['feasible'] for r in record['test'].values()),
                test=split_stats, logical_case_calls=record['logical_case_calls']))
    methods = sorted({r['method'] for r in rows})
    aggregate = {}
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        means = [np.mean([v['energy'] for v in r['test'].values()]) for r in group
                 if r['returned'] and all(v['energy'] is not None for v in r['test'].values())]
        aggregate[method] = dict(seeds=len(group), returns=sum(r['returned'] for r in group),
            passed_all=sum(r['pass_all'] for r in group),
            mean_energy_all_valid_returns=float(np.mean(means)) if means else None,
            passed_cases=sum(v['passed'] for r in group for v in r['test'].values()),
            tested_cases=sum(v['total'] for r in group for v in r['test'].values()),
            logical_case_calls=[r['logical_case_calls'] for r in group])
    destination.mkdir(parents=True)
    for name in ('protocol.json', 'summary.json', 'selections_locked.json'):
        write_json(destination/name, compact(read(root/name)))
    for seed in protocol['seeds']:
        for path in sorted((root/f'seed_{seed}').glob('*.json')):
            write_json(destination/path.parent.name/path.name, compact(read(path)))
    # Compacting changes hashes: retain the original byte hashes separately.
    manifest = dict(original={name: file_hash(root/name) for name in
        ('protocol.json', 'summary.json', 'selections_locked.json')},
        compact={str(p.relative_to(destination)): file_hash(p) for p in sorted(destination.rglob('*.json'))},
        checks='Completed source snapshots, selection locks, prefix-only shortlists, budget and return rules verified',
        omitted='Per-cycle ADC rows and raw transient waveforms; aggregate and per-context measurements retained')
    write_json(destination/'verification.json', manifest)
    result = dict(kind=kind, aggregate=aggregate, seed_results=rows, test_failures=failures,
                  prequential_diagnostics=forecasts, source_sha256=protocol['source_sha256'])
    write_json(destination/'analysis.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adc', type=Path, required=True)
    p.add_argument('--d2d', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh publication directory')
    for kind in ('adc', 'd2d'):
        result = check_and_summarize(getattr(args, kind), args.output/kind)
        print(kind, json.dumps(result['aggregate']), flush=True)


if __name__ == '__main__':
    main()
