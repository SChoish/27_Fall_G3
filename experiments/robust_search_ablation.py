"""Paired search/validation ablation with charged SPICE calls and locked tests.

All four policies get the same number of case evaluations, not equal wall time.
The validation variants spend 24 calls on up to three candidates on eight new
development stimuli. They use only a prefix of the full search trajectory;
later search rows are never consulted for their shortlist or return decision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import scipy

from adc_driver import circuit as adc
from d2d_sizing import circuit as d2d
from experiments.benchmark_constrained_search import ADC_MULTI_WARM_START, response, suites
from experiments.constrained_search import ConstrainedSearch, MaxConstraintSearch

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'legacy': ConstrainedSearch, 'max_constraint': MaxConstraintSearch}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_suites(kind):
    if kind == 'adc':
        development = suites('adc', multiple_inputs=True)[0]
        c = adc.Context()
        patterns = [tuple(np.random.default_rng(s).uniform(.8, 1.1, 6))
                    for s in range(261125310, 261125314)]
        patterns += [(.8,)*6, (1.1,)*6, tuple(np.linspace(.8, 1.1, 6)),
                     tuple(np.linspace(1.1, .8, 6))]
        validation = [replace(c, name=f'validation_{i}', levels_v=p)
                      for i, p in enumerate(patterns)]
        test = [replace(c, name=f'test_{s}', maxstep_ns=.5,
                        levels_v=tuple(np.random.default_rng(s).uniform(.8, 1.1, 6)))
                for s in range(261125910, 261125926)]
        return development, validation, {'fresh_inputs': test}
    development = d2d.default_suite(channel_model='rc')
    bases = development[::2]
    validation = [replace(c, name=f'validation_{j}_{k}', bits=64, pattern=pattern,
                          seed=261125310+101*j+k)
                  for j, c in enumerate(bases)
                  for k, pattern in enumerate(('random', 'random', 'same', 'runs'))]
    # Fresh channel values fixed before the campaign, within the original envelope.
    new_bases = []
    for j, (vdd, temp, res, cap, coupling) in enumerate([
            (.97, 48., 37., .325, .085), (1.01, 35., 33., .315, .075)]):
        old = bases[j]
        lanes = [replace(lane, vdd=vdd, temperature_c=temp, resistance_ohm=res+2*k,
                         capacitance_pf=cap)
                 for k, lane in enumerate((old.context.lane0, old.context.lane1))]
        new_bases.append(replace(old, context=replace(old.context, lane0=lanes[0],
                                                     lane1=lanes[1], coupling_pf=coupling)))
    test = {}
    for split, contexts, offset in [('same_channels', bases, 0), ('new_channels', new_bases, 1000)]:
        test[split] = [replace(c, name=f'{split}_{j}_{k}', bits=128, points_per_ui=128,
                              pattern='random', seed=261125910+offset+101*j+k)
                       for j, c in enumerate(contexts) for k in range(4)]
    return development, validation, test


def initial_designs(kind, seed):
    if kind == 'adc':
        rng = np.random.default_rng(seed+2718)
        designs = [adc.NOMINAL, adc.PRIOR_OPAMP, adc.ADC_REFERENCE,
                   *(adc.decode(u) for u in rng.uniform(0, 1, (5, 9)))]
        designs[3] = np.asarray(ADC_MULTI_WARM_START)
        return designs
    rng = np.random.default_rng(np.random.SeedSequence([seed, 47011]))
    return [d2d.NOMINAL, *(d2d.decode(u) for u in rng.uniform(0, 1, (7, 6)))]


def energy(kind, row):
    return row['mean_driver_energy_pj' if kind == 'adc' else 'energy_pj_bit']


def shortlist(kind, rows, count=3):
    """Two lowest-energy feasible designs plus the largest measured slack.

    Invalid/infeasible rows cannot be returned. Deduplicate by physical design.
    Every decision here depends only on search rows, never validation or test.
    """
    feasible = [r for r in rows if r['oracle']['feasible']]
    if not feasible:
        return []
    ordered = sorted(feasible, key=lambda r: energy(kind, r['oracle']))
    safe = min(feasible, key=lambda r: max(response(kind, r['oracle'])[1]))
    candidates = ordered[:max(0, count-1)]+[safe]+ordered
    result, seen = [], set()
    for row in candidates:
        key = tuple(row['x'])
        if key not in seen:
            result.append(row)
            seen.add(key)
        if len(result) == count:
            break
    return result


def validated_selection(kind, candidates, measurements):
    if len(candidates) != len(measurements):
        raise ValueError('Every shortlisted design needs its validation measurement')
    passed = [(r, v) for r, v in zip(candidates, measurements)
              if r['oracle']['feasible'] and v['feasible']]
    if not passed:
        return None
    row, validation = min(passed, key=lambda p: energy(kind, p[0]['oracle']))
    return dict(design=row['x'], development=row['oracle'], validation=validation)


class Measurer:
    def __init__(self, kind, contexts, output, workers):
        self.kind, self.contexts, self.output = kind, contexts, Path(output)
        self.cache = {}
        self.requests = self.hits = 0
        self.seconds = 0.
        self.backend = d2d.Evaluator(contexts, output, workers=workers) if kind == 'd2d' else None

    def __call__(self, x):
        self.requests += 1
        key = tuple(np.asarray(x, dtype=float))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        start = time.perf_counter()
        result = (self.backend.evaluate(x) if self.backend else
                  adc.evaluate_suite(x, self.contexts))
        self.seconds += time.perf_counter()-start
        self.cache[key] = result
        adc.write_json(self.output/'records'/f'{len(self.cache):05d}.json',
                       dict(x=list(key), oracle=result))
        return result

    def accounting(self):
        result = dict(requests=self.requests, shared_design_cache_hits=self.hits,
                      unique_designs=len(self.cache), elapsed_seconds=self.seconds)
        result['physical'] = (self.backend.accounting() if self.backend else
                              dict(spice_transient_attempts=len(self.cache)*len(self.contexts)))
        return result


def fit_seed(kind, seed, budget, development, validation, output, workers):
    folder = output/f'seed_{seed}'
    folder.mkdir()
    circuit = adc if kind == 'adc' else d2d
    evaluate = Measurer(kind, development, folder/'development', workers)
    validate = Measurer(kind, validation, folder/'validation', workers)
    initial = [dict(x=x.tolist(), oracle=evaluate(x)) for x in initial_designs(kind, seed)]
    adc.write_json(folder/'initial.json', initial)
    constraint_count = len(development)*(1 if kind == 'adc' else 3)
    prefix = budget-3*len(validation)//len(development)
    if prefix < len(initial) or (budget-prefix)*len(development) != 3*len(validation):
        raise ValueError('Cannot allocate this budget exactly')
    returns = {}
    for name, cls in METHODS.items():
        optimizer = cls(len(circuit.NAMES), constraint_count, seed)
        rows = list(initial)
        for row in rows:
            optimizer.observe(np.clip(circuit.encode(row['x']), 0, 1), *response(kind, row['oracle']))
        started = time.perf_counter()
        with (folder/f'{name}.jsonl').open('x') as trace:
            while len(rows) < budget:
                point = optimizer.ask()
                x = circuit.decode(point)
                oracle = evaluate(x)
                optimizer.observe(point, *response(kind, oracle))
                row = dict(x=x.tolist(), oracle=oracle, radius=optimizer.radius)
                rows.append(row)
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                trace.flush()
                if len(rows) % 8 == 0:
                    print(kind, seed, name, len(rows), flush=True)
        adc.write_json(folder/f'{name}.json', rows)
        feasible = [r for r in rows if r['oracle']['feasible']]
        best = min(feasible, key=lambda r: energy(kind, r['oracle'])) if feasible else None
        returns[name] = dict(selection=None if best is None else dict(design=best['x'], development=best['oracle']),
                             logical_case_calls=budget*len(development), search_queries=budget,
                             validation_queries=0, search_elapsed_seconds=time.perf_counter()-started)
        candidates = shortlist(kind, rows[:prefix])
        adc.write_json(folder/f'{name}_shortlist_locked.json', candidates)
        measured = [validate(r['x']) for r in candidates]
        adc.write_json(folder/f'{name}_validation.json', measured)
        returns[name+'_validated'] = dict(selection=validated_selection(kind, candidates, measured),
            logical_case_calls=prefix*len(development)+len(candidates)*len(validation),
            search_queries=prefix, validation_queries=len(candidates),
            validation_pass_count=sum(m['feasible'] for m in measured),
            shortlist_sha256=file_hash(folder/f'{name}_shortlist_locked.json'))
    adc.write_json(folder/'returns.json', returns)
    return dict(returns=returns, physical_cost=dict(development=evaluate.accounting(),
                                                  validation=validate.accounting()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('adc', 'd2d'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--budget', type=int)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    budget = args.budget or (40 if args.kind == 'adc' else 24)
    if budget < 14 or min(args.jobs, args.workers) < 1 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Invalid budget, seed list or worker count')
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a fresh output directory')
    development, validation, test = new_suites(args.kind)
    runtime = adc.verify_runtime()
    sources = [Path(__file__), ROOT/'experiments/constrained_search.py',
               ROOT/'experiments/benchmark_constrained_search.py', ROOT/'adc_driver/circuit.py',
               ROOT/'d2d_sizing/circuit.py', ROOT/'d2d_coupled_calibration/link.py',
               ROOT/'d2d_coupled_calibration/coupled_link.py',
               ROOT/'adc_driver/models/sky130_1v8_tt.spice', ROOT/'d2d_coupled_calibration/assets/45nm_bulk.txt']
    output.mkdir(parents=True)
    protocol = dict(kind=args.kind, seeds=args.seeds, budget=budget,
        logical_case_budget=budget*len(development), initial=8, validation_reserve=24,
        development=[asdict(c) for c in development], validation=[asdict(c) for c in validation],
        test={k: [asdict(c) for c in v] for k, v in test.items()},
        runtime=runtime, binary_sha256=file_hash(runtime['binary']),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        methods=list(METHODS), shortlist='Two lowest development energies plus largest observed slack, deduplicated',
        selection='Lowest development energy passing every validation case; abstain if none passes',
        reuse='Shared initial measurements, search prefixes and identical designs are physically reused; logical cost charged per arm',
        comparison='Same case-call budget; report wall time separately. Validation is development data, not test.')
    adc.write_json(output/'protocol.json', protocol)
    for source in sources:
        frozen = output/'source'/source.relative_to(ROOT)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes())
    with ThreadPoolExecutor(args.jobs) as pool:
        runs = dict(zip(args.seeds, pool.map(lambda seed: fit_seed(args.kind, seed, budget,
            development, validation, output, args.workers), args.seeds)))
    lock = output/'selections_locked.json'
    adc.write_json(lock, dict(protocol_sha256=file_hash(output/'protocol.json'), runs=runs))
    lock_hash = file_hash(lock)
    print(args.kind, 'locked all returns before final test', flush=True)

    def confirm(seed):
        evaluators = {split: Measurer(args.kind, contexts, output/'test'/str(seed)/split, args.workers)
                      for split, contexts in test.items()}
        result = {}
        for name, record in runs[seed]['returns'].items():
            selection = record['selection']
            result[name] = dict(**record, test={} if selection is None else
                               {split: evaluate(selection['design']) for split, evaluate in evaluators.items()})
        return dict(results=result, physical_test_cost={k: v.accounting() for k, v in evaluators.items()})

    with ThreadPoolExecutor(args.jobs) as pool:
        final = dict(zip(args.seeds, pool.map(confirm, args.seeds)))
    if file_hash(lock) != lock_hash:
        raise RuntimeError('Locked selections changed')
    for path, digest in protocol['source_sha256'].items():
        if file_hash(ROOT/path) != digest:
            raise RuntimeError(f'Source changed while campaign ran: {path}')
    adc.write_json(output/'summary.json', dict(protocol_sha256=file_hash(output/'protocol.json'),
                                              locked_selection_sha256=lock_hash, results=final))
    print(args.kind, 'completed', output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
