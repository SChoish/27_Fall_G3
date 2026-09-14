"""Run a matched 2x2 attention/line-search ablation using real SPICE queries.

Example: python compare_attention.py --budget 32 --seeds 0 1 2 --workers 2
Every variant pays for its own initial eight designs. No oracle cache is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from circuit import ROOT, cost, write_json


def summarize(paths, output):
    summary = []
    initial_by_seed = {}
    for name, seed, path in paths:
        run = json.loads(path.read_text())
        rows = run['trace']
        assert len(rows) == run['design_queries'] == run['budget']
        assert run['transient_queries'] == len(rows) * len(run['contexts'])
        assert [r['query'] for r in rows] == list(range(1, len(rows) + 1))
        assert all(abs(r['cost'] - cost(r['oracle'])) < 1e-12 for r in rows)
        designs = [r['u'] for r in rows[:8]]
        if seed in initial_by_seed:
            assert designs == initial_by_seed[seed], 'Mismatched initialization'
        initial_by_seed[seed] = designs
        observed_best = min(rows, key=lambda r: r['cost'])
        assert observed_best == run['best']
        first = next((r['query'] for r in rows if r['oracle']['feasible']), None)
        assert first == run['first_success']
        best = run['best']['oracle']
        initial_good = [r['oracle']['mean_driver_energy_pj'] for r in rows[:8] if r['oracle']['feasible']]
        reference = min(initial_good) if initial_good else None
        proposals = [r['proposal'] for r in rows if 'proposal' in r]
        for p in proposals:
            assert p['additional_spice_queries'] == p['attention']['additional_spice_queries'] == 0
            assert p['metric_min_eigenvalue'] > 0
            assert abs(sum(p['coordinate_probabilities']) - 1) < 1e-5
            if p['line_search'] == 'none': assert p['line_search_trials'] == 0
        summary.append(dict(
            variant=name, seed=seed, design_queries=len(rows), transient_queries=run['transient_queries'],
            source_design_queries=run['source_design_queries'], first_success=first,
            feasible=best['feasible'], best_error_mv=best['worst_hold_error_mv'],
            best_energy_pj=best['mean_driver_energy_pj'], initial_feasible_energy_pj=reference,
            energy_improvement_pct=(100 * (reference - best['mean_driver_energy_pj']) / reference
                                    if reference and best['feasible'] else None),
            metric_selected=sum(p['selected_origin'] == 'metric_step' for p in proposals),
            post_initial_queries=len(rows)-8,
            attention_updates=sum(p['attention']['updates'] for p in proposals),
            line_search_trials=sum(p['line_search_trials'] for p in proposals),
            surrogate_objective_evaluations=sum(p['surrogate_objective_evaluations'] for p in proposals),
            elapsed_seconds=run['elapsed_seconds']))
    output = Path(output)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary)
    write_json(output / 'summary.json', dict(
        rows=summary, matching_initial_designs=True,
        total_design_queries=sum(r['design_queries'] for r in summary),
        python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'pymoo')},
        note='Warm-start integration ablation; the 24-design bundled source dataset is a smoke dataset. '
             'Success at query 3 can be the supplied feasible reference. This is not evidence of general superiority.'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['regression', 'rnc'], default='rnc')
    parser.add_argument('--checkpoint', help='Defaults to the bundled checkpoint for --method')
    parser.add_argument('--contexts')
    parser.add_argument('--budget', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--include-baselines', action='store_true', help='Also run random and GA at the same budget')
    parser.add_argument('--output', default=str(ROOT / 'results/attention_demo'))
    args = parser.parse_args()
    if args.workers < 1 or args.budget < 16 or args.budget % 8:
        parser.error('workers >= 1; budget a multiple of 8 and >= 16')
    if len(set(args.seeds)) != len(args.seeds): parser.error('Seeds must be unique')
    out = Path(args.output).resolve()
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}; choose a new output directory')
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / f'checkpoints/{args.method}.pt'
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    variants = [(f'{a}_{ls}', args.method, a, ls) for a in ('off', 'learned') for ls in ('none', 'armijo')]
    if args.include_baselines: variants.extend([(m, m, 'off', 'none') for m in ('random', 'ga')])
    jobs = [(name, method, a, ls, seed) for seed in args.seeds for name, method, a, ls in variants]
    out.mkdir(parents=True)
    def job(item):
        name, method, attention, line_search, seed = item
        folder = out / f'{name}_s{seed}'
        cmd = [sys.executable, str(ROOT/'run.py'), 'optimize', '--method', method,
               '--seed', str(seed), '--budget', str(args.budget), '--output', str(folder)]
        if method in ('regression', 'rnc'):
            cmd += ['--checkpoint', str(checkpoint), '--attention', attention, '--line-search', line_search]
        if args.contexts: cmd += ['--contexts', str(Path(args.contexts).resolve())]
        with (out / f'{name}_s{seed}.log').open('w') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f'Completed {name} seed {seed}', flush=True)
        return name, seed, folder/'trace.json'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        paths = list(pool.map(job, jobs))
    print(json.dumps(summarize(paths, out), indent=2))


if __name__ == '__main__':
    main()
