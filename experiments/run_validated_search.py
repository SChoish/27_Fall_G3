"""Run one optimizer with a genuinely reserved, charged validation budget."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from experiments.benchmark_constrained_search import response
from experiments.robust_search_ablation import (
    ROOT, METHODS, Measurer, adc, d2d, file_hash, initial_designs, new_suites,
    shortlist, validated_selection,
)


def run(kind, seed, budget, method, development, validation, evaluate, validate, output):
    """Stop search at the reserved prefix; no hidden full-budget search is run."""
    circuit = adc if kind == 'adc' else d2d
    reserve = 3*len(validation)
    if reserve % len(development) or budget-reserve//len(development) < 8:
        raise ValueError('Insufficient budget for eight initial points and validation')
    search_budget = budget-reserve//len(development)
    search = METHODS[method](len(circuit.NAMES), len(development)*(1 if kind == 'adc' else 3), seed)
    rows = []
    for x in initial_designs(kind, seed):
        result = evaluate(x)
        search.observe(np.clip(circuit.encode(x), 0, 1), *response(kind, result))
        rows.append(dict(x=x.tolist(), oracle=result))
    adc.write_json(output/'initial.json', rows)
    with (output/'search.jsonl').open('x') as log:
        while len(rows) < search_budget:
            point = search.ask()
            x = circuit.decode(point)
            result = evaluate(x)
            search.observe(point, *response(kind, result))
            row = dict(x=x.tolist(), oracle=result, radius=search.radius)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
    adc.write_json(output/'search.json', rows)
    candidates = shortlist(kind, rows)
    adc.write_json(output/'shortlist_locked.json', candidates)
    measured = [validate(r['x']) for r in candidates]
    adc.write_json(output/'validation.json', measured)
    result = dict(selection=validated_selection(kind, candidates, measured),
                  search_queries=len(rows), validation_queries=len(candidates),
                  logical_case_calls=len(rows)*len(development)+len(measured)*len(validation),
                  logical_case_budget=budget*len(development),
                  shortlist_sha256=file_hash(output/'shortlist_locked.json'))
    adc.write_json(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('adc', 'd2d'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--method', choices=tuple(METHODS), default='legacy')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--validation-bits', type=int, choices=(64, 128),
                        help='D2D validation packet length (default: 128; does not change search packets)')
    args = parser.parse_args()
    budget = args.budget if args.budget is not None else (40 if args.kind == 'adc' else 24)
    if args.output.exists() or budget < 14 or args.seed < 0 or args.workers < 1:
        parser.error('Use a fresh output, budget >=14, seed >=0 and workers >=1')
    development, validation, _ = new_suites(args.kind)
    if args.kind == 'd2d':
        validation = [replace(c, bits=args.validation_bits or 128) for c in validation]
    elif args.validation_bits is not None:
        parser.error('--validation-bits applies only to D2D')
    runtime = adc.verify_runtime()
    sources = [Path(__file__).resolve(), ROOT/'experiments/robust_search_ablation.py',
               ROOT/'experiments/constrained_search.py', ROOT/'experiments/benchmark_constrained_search.py',
               ROOT/'adc_driver/circuit.py', ROOT/'d2d_sizing/circuit.py',
               ROOT/'d2d_coupled_calibration/link.py', ROOT/'d2d_coupled_calibration/coupled_link.py',
               ROOT/'adc_driver/models/sky130_1v8_tt.spice', ROOT/'d2d_coupled_calibration/assets/45nm_bulk.txt']
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    args.output.mkdir(parents=True)
    adc.write_json(args.output/'protocol.json', dict(kind=args.kind, seed=args.seed, budget=budget,
        method=args.method, development=[asdict(c) for c in development],
        validation=[asdict(c) for c in validation], runtime=runtime,
        binary_sha256=file_hash(runtime['binary']), source_sha256=hashes))
    evaluate = Measurer(args.kind, development, args.output/'development', args.workers)
    validate = Measurer(args.kind, validation, args.output/'validation_measurements', args.workers)
    run(args.kind, args.seed, budget, args.method, development, validation, evaluate, validate, args.output)
    for path, digest in hashes.items():
        if file_hash(ROOT/path) != digest:
            raise RuntimeError(f'Source changed during execution: {path}')
    adc.write_json(args.output/'physical_cost.json', dict(development=evaluate.accounting(),
                                                        validation=validate.accounting()))
    print(args.output/'summary.json')


if __name__ == '__main__':
    main()
