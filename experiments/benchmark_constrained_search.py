"""Compare the constrained optimizer with G3 baselines on unchanged SPICE tasks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

from adc_driver import circuit as adc
from d2d_sizing import circuit as d2d
from experiments.constrained_search import ConstrainedSearch


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT/'implementation_results/20260915_d2d_alarm'
ADC_DIR = ROOT/'adc_driver'
# Selected on the earlier two-pattern development run; all arms pay for it.
ADC_MULTI_WARM_START = [11.341023176380707,2.116961422843628,3.1512093002313466,
    50.551431566422906,14.09396679267611,.3,19.33290486877235,47.42333484821684,1.]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response(kind, row):
    if not row['valid']:
        return None,None
    if kind == 'adc':
        return row['mean_driver_energy_pj'], [np.log(max(c['worst_hold_error_mv'],1e-8)/
            c['context']['error_limit_mv']) for c in row['contexts']]
    return row['energy_pj_bit'], [g for c in row['cases'] for g in (
        *[1-m/.08 for m in c['margins']],c['overshoot_fraction']/.2-1)]


def trace_summary(rows, circuit):
    incumbent = min(rows,key=lambda r:circuit.cost(r['oracle']))
    initial = min(rows[:8],key=lambda r:circuit.cost(r['oracle']))
    energy_key = 'mean_driver_energy_pj' if circuit is adc else 'energy_pj_bit'
    improvement = circuit.cost(incumbent['oracle'])<circuit.cost(initial['oracle'])
    first_improvement = next((j for j,r in enumerate(rows,1) if j>8 and
                             circuit.cost(r['oracle'])<circuit.cost(initial['oracle'])), None)
    return dict(design=incumbent['x'], development=incumbent['oracle'],
                development_energy=incumbent['oracle'].get(energy_key), initial_energy=initial['oracle'].get(energy_key),
                improved_initial=improvement, first_improvement=first_improvement, queries=len(rows))


def run_new(kind, initial, evaluate, seed, budget, output):
    circuit = adc if kind == 'adc' else d2d
    constraint_count = len(initial[0]['oracle']['contexts']) if kind == 'adc' else 3*len(initial[0]['oracle']['cases'])
    optimizer = ConstrainedSearch(len(circuit.NAMES),constraint_count,seed)
    rows = list(initial)
    for r in rows:
        optimizer.observe(np.clip(circuit.encode(r['x']),0,1),*response(kind,r['oracle']))
    start = time.perf_counter()
    optimizer_seconds = 0.
    with (output/'constrained_search.jsonl').open('x') as trace:
        while len(rows)<budget:
            begin = time.perf_counter()
            u = optimizer.ask()
            optimizer_seconds += time.perf_counter()-begin
            x = circuit.decode(u)
            result = evaluate(x)
            optimizer.observe(u,*response(kind,result))
            row = dict(x=x.tolist(), oracle=result, radius=float(optimizer.radius))
            rows.append(row)
            trace.write(json.dumps(row,allow_nan=False)+'\n')
            trace.flush()
            if len(rows)%8==0:
                best = trace_summary(rows,circuit)
                print(kind,seed,len(rows),best['development_energy'], flush=True)
    circuit.write_json(output/'constrained_search.json',rows)
    return dict(**trace_summary(rows,circuit), optimizer_seconds=optimizer_seconds,
                elapsed_seconds=time.perf_counter()-start)


def adc_run(root, seed, budget, contexts, warm_start=None):
    # Existing ADC optimizer modules use script-relative imports.
    from search import initial, optimize
    result = {}
    points = initial(seed)
    if warm_start is not None:
        points[3] = adc.encode(warm_start)
    evaluations = [dict(x=adc.decode(u).tolist(),oracle=adc.evaluate_suite(adc.decode(u),contexts)) for u in points]
    adc.write_json(root/'initial.json',evaluations)
    result['constrained_search'] = run_new('adc',evaluations,lambda x:adc.evaluate_suite(x,contexts),seed,budget,root)
    methods = ('random','ga') if warm_start is not None else ('random','ga','rnc')
    for method in methods:
        checkpoint = ADC_DIR/'checkpoints/rnc.pt' if method=='rnc' else None
        record = optimize(method,budget,seed,contexts,root/method,checkpoint=checkpoint,warm_start=warm_start)
        result[method] = trace_summary(record['trace'],adc)
        result[method]['elapsed_seconds'] = record['elapsed_seconds']
    for name in methods:
        original = read(root/name/'trace.json')['trace']
        assert np.allclose([r['x'] for r in original[:8]],[r['x'] for r in evaluations])
    return result


def d2d_run(root, seed, budget, contexts):
    methods = ('alarm','regression','random')
    recorded = {m:read(PILOT/f'traces/{m}_s{seed}.json')['rows'] for m in methods}
    initial = recorded['alarm'][:8]
    for name in methods:
        assert len(recorded[name])==budget and recorded[name][:8]==initial
    d2d.write_json(root/'initial_from_pilot.json',initial)
    result = {m:trace_summary(rows,d2d) for m,rows in recorded.items()}
    evaluator = d2d.Evaluator(contexts,root/'measurements',workers=3)
    result['constrained_search'] = run_new('d2d',initial,evaluator.evaluate,seed,budget,root)
    d2d.write_json(root/'new_spice_cost.json',evaluator.accounting())
    return result


def suites(kind, multiple_inputs=False):
    if kind == 'adc':
        c = adc.Context()
        development = [c]
        if multiple_inputs:
            for index, levels in enumerate(((.8,1.1,.8,1.1,.8,1.1),
                    (.81,1.02,.86,1.08,.82,.94),(.91,.83,1.06,.89,1.1,1.02))):
                development.append(replace(c,name=f'development_{index}',levels_v=levels))
        matched = [replace(context,maxstep_ns=.5) for context in development]
        seeds = range(26112810,26112822) if multiple_inputs else range(26112710,26112718)
        heldout = [replace(c,name=f'packet_{seed}',maxstep_ns=.5,
                           levels_v=tuple(np.random.default_rng(seed).uniform(.8,1.1,6)))
                   for seed in seeds]
    else:
        protocol = read(PILOT/'protocol.json')
        development = [d2d.Case.from_dict(c) for c in protocol['development']]
        matched = [replace(d2d.Case.from_dict(c),points_per_ui=128) for c in protocol['confirmation']]
        heldout = [replace(c,seed=c.seed+261127,points_per_ui=128)
                   for c in d2d.default_suite('test',channel_model='rc')]
    return development, matched, heldout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('adc','adc-multi','d2d'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('Use at least one worker')
    multiple_inputs = args.kind=='adc-multi'
    kind = 'adc' if multiple_inputs else args.kind
    circuit = adc if kind=='adc' else d2d
    if kind=='adc':
        sys.path.insert(0,str(ADC_DIR))
    if kind=='adc' and not multiple_inputs:
        import torch
        torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True)
    dev, matched, heldout = suites(kind,multiple_inputs)
    budget = 40 if multiple_inputs else 32 if kind=='adc' else 24
    seeds = list(range(3))
    file_sources = [Path(__file__).resolve(),Path(__file__).with_name('constrained_search.py'),Path(circuit.__file__)]
    if kind=='d2d':
        file_sources += [PILOT/'protocol.json',*sorted((PILOT/'traces').glob('*.json'))]
    else:
        file_sources += [ADC_DIR/'search.py',ADC_DIR/'learning.py',ADC_DIR/'checkpoints/rnc.pt']
    circuit.write_json(out/'protocol.json', dict(kind=args.kind,seeds=seeds,budget=budget,initial=8,
        shared_warm_start=ADC_MULTI_WARM_START if multiple_inputs else None,
        development=[asdict(c) for c in dev], matched=[asdict(c) for c in matched],
        heldout=[asdict(c) for c in heldout],runtime=adc.verify_runtime(),
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in file_sources},
        data_access=('D2D uses the published initial measurements and baseline histories from the same pilot. '
                     'All returned designs are simulated again for comparison.' if kind=='d2d' else
                     'All arms share the charged warm start; no offline training. The bundled RNC matches only the single-input protocol.'
                     if multiple_inputs else 'ADC baselines are run fresh; RNC has the bundled 24-design source set. New search, random, and GA do not.'),
        selection='Existing circuit cost on development only. All returns locked before any confirmation simulation.'))

    def fit(seed):
        path = out/f'seed_{seed}'
        path.mkdir()
        if multiple_inputs:
            return adc_run(path,seed,budget,dev,ADC_MULTI_WARM_START)
        run = adc_run if kind=='adc' else d2d_run
        return run(path,seed,budget,dev)

    with ThreadPoolExecutor(args.jobs) as pool:
        results = dict(zip(seeds,pool.map(fit,seeds)))
    lock = out/'selections_locked.json'
    circuit.write_json(lock,dict(protocol_sha256=sha(out/'protocol.json'),results=results))
    print(kind,'locked all returns',flush=True)
    unique = {}
    for seed, methods in results.items():
        for method, r in methods.items():
            key = tuple(np.round(r['design'],10))
            unique.setdefault(key,[]).append((seed,method))

    def confirm(item):
        design, tags = item
        name = f'{tags[0][0]}_{tags[0][1]}'
        if kind=='adc':
            a = adc.evaluate_suite(design,matched,out/'confirmation'/name/'matched')
            b = adc.evaluate_suite(design,heldout,out/'confirmation'/name/'heldout')
            return tags,a,b,dict(matched_attempts=len(matched),heldout_attempts=len(heldout))
        a = d2d.Evaluator(matched,out/'confirmation'/name/'matched',workers=2)
        b = d2d.Evaluator(heldout,out/'confirmation'/name/'heldout',workers=2)
        return tags,a.evaluate(design),b.evaluate(design),dict(matched=a.accounting(),heldout=b.accounting())

    costs = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for tags,a,b,cost in pool.map(confirm,unique.items()):
            costs.append(dict(used_by=tags,**cost))
            for seed, method in tags:
                results[seed][method].update(matched=a,heldout=b)
    circuit.write_json(out/'summary.json',dict(locked_selection_sha256=sha(lock),results=results,
                                              confirmation_cost=costs))
    print(kind,'completed',out/'summary.json',flush=True)


if __name__ == '__main__':
    main()
