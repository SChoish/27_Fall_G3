"""Run the constrained optimizer on the existing ADC or D2D circuit evaluator."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.benchmark_constrained_search import (
    ADC_MULTI_WARM_START, adc, d2d, read, run_new, sha, suites,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('adc','adc-multi','d2d'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, help='ADC contexts or D2D cases in JSON')
    parser.add_argument('--budget', type=int, help='Design queries including eight initial points')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3, help='Parallel D2D case simulations')
    args = parser.parse_args()
    if args.budget is None:
        args.budget = {'adc':32,'adc-multi':40,'d2d':24}[args.kind]
    if args.budget < 8 or args.workers < 1 or args.seed < 0:
        parser.error('Use a budget of at least eight, a nonnegative seed and positive worker count')
    if args.output.exists():
        parser.error('Use a fresh output directory')

    kind = 'adc' if args.kind=='adc-multi' else args.kind
    if kind == 'adc':
        circuit = adc
        contexts = (suites('adc',multiple_inputs=True)[0] if args.kind=='adc-multi' and not args.contexts
                    else adc.load_contexts(args.contexts))
        if not contexts:
            parser.error('Provide at least one ADC context')
        rng = np.random.default_rng(args.seed+2718)
        designs = [adc.NOMINAL,adc.PRIOR_OPAMP,adc.ADC_REFERENCE,
                   *(adc.decode(u) for u in rng.uniform(0,1,(5,9)))]
        if args.kind=='adc-multi':
            designs[3] = np.asarray(ADC_MULTI_WARM_START)
        runtime = adc.verify_runtime()

        def evaluate(x):
            return adc.evaluate_suite(x,contexts)

        args.output.mkdir(parents=True)
    else:
        circuit = d2d
        contexts = ([d2d.Case.from_dict(c) for c in read(args.contexts)] if args.contexts else
                    d2d.default_suite(channel_model='rc'))
        if not contexts or any((c.margin_min,c.overshoot_max) != (.08,.2) for c in contexts):
            parser.error('Use a nonempty D2D suite with the existing .08 margin and .2 overshoot thresholds')
        rng = np.random.default_rng(np.random.SeedSequence([args.seed,47011]))
        designs = [d2d.NOMINAL,*(d2d.decode(u) for u in rng.uniform(0,1,(7,6)))]
        evaluator = d2d.Evaluator(contexts,args.output/'measurements',workers=args.workers)
        runtime = evaluator.simulator_version
        evaluate = evaluator.evaluate

    circuit.write_json(args.output/'protocol.json', dict(kind=args.kind,seed=args.seed,budget=args.budget,
        contexts=[asdict(c) for c in contexts], runtime=runtime, initial=[x.tolist() for x in designs],
        optimizer_sha256=sha(Path(__file__).with_name('constrained_search.py'))))
    initial = [dict(x=x.tolist(),oracle=evaluate(x)) for x in designs]
    circuit.write_json(args.output/'initial.json',initial)
    summary = run_new(kind,initial,evaluate,args.seed,args.budget,args.output)
    summary['cost'] = (dict(design_queries=args.budget,spice_transient_attempts=args.budget*len(contexts))
                       if kind=='adc' else evaluator.accounting())
    circuit.write_json(args.output/'summary.json',summary)
    circuit.write_json(args.output/'best_design.json',dict(zip(circuit.NAMES,summary['design'])))
    print(args.output/'summary.json')


if __name__ == '__main__':
    main()
