"""Paired ALARM/regression/random pilot on actual ngspice D2D RC sizing."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from d2d_sizing import circuit, surrogate
from d2d_sizing.alarm_adapter import make_optimizer, metrics, UPSTREAM_COMMIT

METHODS = ('alarm', 'regression', 'random')


def row(x, result):
    return dict(x=np.asarray(x).tolist(), u=circuit.encode(x).tolist(), oracle=result, split='train')


def best(rows):
    return min(rows, key=lambda r: circuit.cost(r['oracle']))


def confirmation_suite():
    # New packet seeds, same development channel contexts. Not new PVT corners.
    base = circuit.default_suite(channel_model='rc')
    return [replace(base[2*j], name=f'confirmation_{j}_{k}', pattern=pattern,
                    seed=940001+101*j+k, bits=128)
            for j in range(2) for k,pattern in enumerate(('random','random','same','runs'))]


def run_seed(task):
    output, seed, initial_count, additional, workers = task
    torch.set_num_threads(1)
    root = Path(output)/f'seed_{seed}'
    root.mkdir()
    evaluator = circuit.Evaluator(circuit.default_suite(channel_model='rc'), root/'measurements', workers=workers)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 47011]))
    designs = [circuit.NOMINAL.copy()] + [circuit.decode(u) for u in rng.uniform(0,1,(initial_count-1,6))]
    initial = [row(x,evaluator.evaluate(x)) for x in designs]
    circuit.write_json(root/'initial.json', initial)
    if not all(r['oracle']['valid'] for r in initial):
        raise RuntimeError('Initial simulation failure; pilot stopped without inventing imported labels')
    if not initial[0]['oracle']['feasible']:
        raise RuntimeError('Nominal design fails fixed development quality constraints')
    selected = {}
    for method in METHODS:
        rows = copy.deepcopy(initial)
        policy_seconds = 0.
        start = time.perf_counter()
        if method == 'alarm':
            optimizer = make_optimizer(seed, initial_count, additional)
            optimizer.seed_population(designs, [metrics(r['oracle']) for r in initial])
        elif method == 'regression':
            payload = surrogate.train(initial, 'regression', seed, epochs=100, checkpoint='last')
            optimizer = surrogate.Search(payload, initial, mode='buffer', attention='learned', online_epochs=20)
        policy_seconds += time.perf_counter()-start
        proposal_rng = np.random.default_rng(np.random.SeedSequence([seed, 47012]))
        events=[]
        for step in range(additional):
            start = time.perf_counter()
            if method == 'alarm':
                x = optimizer.ask()
                if x is None:
                    raise RuntimeError('Unexpected early stop')
                source = optimizer._pending_source
                search = optimizer.last_search
                event = dict(source=source, accepted_steps=search.accepted_steps if search else 0,
                             backtracks=search.backtracks if search else 0)
            elif method == 'regression':
                u,event = optimizer.propose(rows, proposal_rng)
                x = circuit.decode(u)
                predictions = optimizer.predictions(u, rows)
            else:
                x = circuit.decode(proposal_rng.uniform(0,1,6))
                event = dict(source='random')
            policy_seconds += time.perf_counter()-start
            measured = row(x,evaluator.evaluate(x))
            start = time.perf_counter()
            if method == 'alarm':
                if measured['oracle']['valid']:
                    optimizer.tell(metrics(measured['oracle']))
                else:
                    optimizer.reject('D2D evaluation invalid; see measurement receipts')
            elif method == 'regression':
                optimizer.observe(measured,predictions)
            policy_seconds += time.perf_counter()-start
            rows.append(measured)
            events.append(dict(query=initial_count+step+1, **event))
            circuit.write_json(root/f'{method}_progress.json', dict(completed=step+1, incumbent=best(rows)))
            if (step+1) % 4 == 0:
                b=best(rows)['oracle']
                print(json.dumps(dict(seed=seed,method=method,additional=step+1,energy=b['energy_pj_bit'],feasible=b['feasible'])),flush=True)
        incumbent = best(rows)
        if method == 'alarm':
            assert optimizer.finished and optimizer.stats['attempts'] == initial_count+additional
            assert optimizer.best_feasible is not None
            assert np.allclose(optimizer.best_feasible.design, incumbent['x'],rtol=1e-10,atol=1e-12)
            stats,config=optimizer.stats,optimizer.settings
        elif method == 'regression':
            stats=dict(online_updates=len(optimizer.events))
            config=dict(offline_epochs=100,checkpoint='last',mode='buffer',attention='learned',online_epochs=20)
        else:
            stats,config={},{}
        result=dict(seed=seed,method=method,initial_designs=initial_count,additional=additional,
                    equivalent_design_queries=initial_count+additional,
                    equivalent_search_case_measurements=4*(initial_count+additional),
                    policy_seconds=policy_seconds,rows=rows,events=events,selected=incumbent,
                    stats=stats,config=config)
        circuit.write_json(root/f'{method}.json',result)
        selected[method]=dict(x=incumbent['x'],development=incumbent['oracle'],
                              policy_seconds=policy_seconds,stats=stats)
    cost=evaluator.accounting()
    circuit.write_json(root/'cost.json',cost)
    return dict(seed=seed,selected=selected,cost=cost)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--initial',type=int,default=8)
    p.add_argument('--additional',type=int,default=16)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    if args.initial<4 or args.additional<1 or min(args.jobs,args.workers)<1 or len(set(args.seeds))!=len(args.seeds):p.error('Invalid sizes or duplicate seeds')
    args.output.mkdir(parents=True)
    source_paths=list((REPO/'d2d_sizing').glob('*.py'))+list((REPO/'adc_driver').glob('*.py'))+list((REPO/'third_party/alarm/alarm_torch').glob('*.py'))+[Path(__file__),REPO/'d2d_coupled_calibration/link.py',REPO/'d2d_coupled_calibration/coupled_link.py',REPO/'d2d_coupled_calibration/assets/45nm_bulk.txt']
    protocol=dict(alarm_commit=UPSTREAM_COMMIT,g3_base_commit='eaaabef2d749c0c50f0ed97a73c44777721e88b4',
                  seeds=args.seeds,initial=args.initial,additional=args.additional,methods=METHODS,
                  alarm_config=make_optimizer(args.seeds[0],args.initial,args.additional).settings,
                  regression=dict(epochs=100,checkpoint='last',online_epochs=20,attention='learned',mode='buffer'),
                  development=[asdict(c) for c in circuit.default_suite(channel_model='rc')],
                  confirmation=[asdict(c) for c in confirmation_suite()],
                  source_sha256={str(x.relative_to(REPO)):file_sha(x) for x in source_paths},
                  python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                  primary='Measured returned design energy subject to every confirmation case passing',
                  no_hyperparameter_tuning=True,extra_offline_data=0)
    circuit.write_json(args.output/'protocol.json',protocol)
    tasks=[(str(args.output.resolve()),s,args.initial,args.additional,args.workers) for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results=list(pool.map(run_seed,tasks))
    # Lock every return design before any confirmation waveform is evaluated.
    circuit.write_json(args.output/'selections_locked.json',results)
    lock=file_sha(args.output/'selections_locked.json')
    confirm=circuit.Evaluator(confirmation_suite(),args.output/'confirmation',workers=min(8,args.jobs*args.workers))
    nominal=confirm.evaluate(circuit.NOMINAL)
    final=[]
    for seed_result in results:
        for method in METHODS:
            selection=seed_result['selected'][method]
            checked=confirm.evaluate(selection['x'])
            saving=(1-checked['energy_pj_bit']/nominal['energy_pj_bit'])*100 if checked['valid'] and nominal['valid'] else None
            final.append(dict(seed=seed_result['seed'],method=method,x=selection['x'],confirmation=checked,
                              savings_percent=saving,unchanged_nominal=bool(np.allclose(selection['x'],circuit.NOMINAL,rtol=1e-10,atol=1e-12)),
                              policy_seconds=selection['policy_seconds'],stats=selection['stats']))
    assert lock==file_sha(args.output/'selections_locked.json')
    summary=dict(protocol_sha256=file_sha(args.output/'protocol.json'),selection_sha256=lock,
                 nominal=nominal,results=final,search_costs=[r['cost'] for r in results],confirmation_cost=confirm.accounting())
    circuit.write_json(args.output/'summary.json',summary)
    print(json.dumps({'finished':True,'results':[dict(seed=r['seed'],method=r['method'],pass_all=r['confirmation']['feasible'],savings=r['savings_percent']) for r in final]}),flush=True)


if __name__=='__main__':main()
