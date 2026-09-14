"""ADC mixed-only input/margin and critic ablations, with separate final inputs.

Run init, diagnose, prepare-c, compare, attention, then test. Every stage uses a
frozen protocol and records fresh transients separately from reused measurements.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import uuid
import time
import circuit
from pathlib import Path

import numpy as np

from circuit import (Context, ADC_REFERENCE, encode, decode, evaluate_suite, evaluate,
                     aggregate, cost, write_json, verify_runtime)
from search import fingerprint, physical_fingerprint, optimize

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def install_accounting(out):
    """Persist each attempt before invocation, including interrupted/retried work."""
    original=evaluate
    def charged(x,c=Context(),output=None,timeout_s=60):
        path=out/'spice_attempts'/f'{uuid.uuid4().hex}.json'
        record=dict(started_at=time.time(),context=asdict(c),x=np.asarray(x).tolist(),completed=False)
        write_json(path,record)
        r=original(x,c,output,timeout_s)
        record.update(completed=True,valid=r['valid'],elapsed_seconds=r['elapsed_seconds'])
        write_json(path,record)
        return r
    circuit.evaluate=charged
    return charged

def contexts(protocol,condition):return [Context.from_dict(c) for c in protocol['conditions'][condition]]


def relabel(oracle,requested):
    records=copy.deepcopy(oracle['contexts'])
    if len(records)!=len(requested):raise ValueError('Context count differs')
    for r,c in zip(records,requested):
        old=Context.from_dict(r['context'])
        if physical_fingerprint([old])!=physical_fingerprint([c]):
            raise ValueError('Cannot reuse measurements across different physical inputs')
        r['context']=asdict(c)
        r['feasible']=bool(r['valid'] and r.get('worst_hold_error_mv',1e9)<=c.error_limit_mv)
        r['normalized_violation']=max(0.,r.get('worst_hold_error_mv',1e9)/c.error_limit_mv-1)
    return aggregate(records)


def init(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a fresh study directory')
    source=Path(args.source).resolve();data=read(source)
    base=Context()
    if physical_fingerprint([Context.from_dict(c) for c in data['contexts']])!=physical_fingerprint([base]):
        raise ValueError('Round 2 requires the previous nominal mixed dataset')
    dev=[base,replace(base,name='development_alternating',levels_v=(.8,1.1,.8,1.1,.8,1.1))]
    dev += [replace(base,name=f'development_{s}',levels_v=tuple(np.random.default_rng(s).uniform(.8,1.1,6)))
            for s in (271091,271092)]
    # Fixed independently of diagnostics and optimizer outputs; same operating range.
    tests=[replace(base,name=f'test_{s}',levels_v=tuple(np.random.default_rng(s).uniform(.8,1.1,6)))
           for s in (9152601,9152602,9152603,9152604,9152605,9152606,9152607,9152608)]
    assert not set(c.levels_v for c in tests)&set(c.levels_v for c in dev)
    protocol=dict(version=1,source=str(source),source_sha256=sha(source),quality='mixed',
                  conditions={'A':[asdict(base)],'B':[asdict(replace(base,error_limit_mv=.8))],
                              'C':[asdict(replace(c,error_limit_mv=.8)) for c in dev]},
                  final_contexts=[asdict(c) for c in tests],final_error_limit_mv=1.,
                  interpretation='A→B margin only; B→C direct input diversity. 0.8 mV is an experimental margin, not a generalization guarantee.',
                  source_hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.glob('*.py'))},
                  model_hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'models').rglob('*')) if f.is_file()},
                  runtime=verify_runtime(),
                  comparison=dict(seeds=args.seeds,spice_budget=args.spice_budget,epochs=args.epochs,
                                  critic_epochs=args.critic_epochs,attention='learned',line_search='none'),
                  diagnostic=dict(neighbors=args.neighbors,seed=9152600,radius=.06,
                                  anchor=None if not args.diagnostic_anchor else read(args.diagnostic_anchor),
                                  anchor_path=None if not args.diagnostic_anchor else str(Path(args.diagnostic_anchor).resolve()),
                                  anchor_sha256=None if not args.diagnostic_anchor else sha(args.diagnostic_anchor)),
                  prepared_c=None if not args.prepared_c else dict(path=str(Path(args.prepared_c).resolve()),sha256=sha(args.prepared_c)),
                  test_rule='Lock all comparison and selected attention endpoints before any final SPICE test. No test-driven selection.',
                  selection_rule='Development all-input 1 mV success fraction, then passing mean energy; deterministic label tie break.',
                  budget_rule='Equal actual online transient budget for GA/regression/RNC and all critic modes. Offline, diagnostic, development confirmation and final costs reported separately.')
    if args.spice_budget<64 or args.spice_budget%32:raise ValueError('SPICE budget must be >=64 and a multiple of 32')
    if args.neighbors<1 or args.epochs<1 or args.critic_epochs<1:raise ValueError('Positive counts required')
    if args.diagnostic_anchor:
        from circuit import validate_design,NAMES
        anchor=protocol['diagnostic']['anchor']
        validate_design([anchor[k] for k in NAMES] if isinstance(anchor,dict) else anchor)
    prepared=None
    if args.prepared_c:
        prepared=read(args.prepared_c)
        if prepared.get('reuse',{}).get('source_sha256')!=sha(source):raise ValueError('Prepared C source mismatch')
        expected={r['index']:(r['u'],r['split']) for r in data['records']}
        actual={r['index']:(r['u'],r['split']) for r in prepared['records']}
        if actual!=expected:raise ValueError('Prepared C design membership/splits differ')
        for row in prepared['records']:relabel(row['oracle'],contexts(protocol,'C'))
        if prepared['context_hash']!=fingerprint(contexts(protocol,'C')):raise ValueError('Prepared C context mismatch')
    out.mkdir(parents=True);write_json(out/'protocol.json',protocol)
    write_json(out/'protocol_lock.json',dict(sha256=sha(out/'protocol.json')))
    for label in ('A','B'):
        d=copy.deepcopy(data);cs=contexts(protocol,label)
        for r in d['records']:r['oracle']=relabel(r['oracle'],cs)
        d.update(contexts=[asdict(c) for c in cs],context_hash=fingerprint(cs),
                 reuse=dict(source_sha256=sha(source),additional_transient_queries=0,
                            original_transient_queries=data['transient_queries']))
        write_json(out/f'data_{label}.json',d)
    artifacts={f'data_{label}.json':sha(out/f'data_{label}.json') for label in ('A','B')}
    if prepared is not None:
        (out/'data_C.json').write_bytes(Path(args.prepared_c).read_bytes())
        artifacts['data_C.json']=sha(out/'data_C.json')
    write_json(out/'artifacts.json',artifacts)
    print(json.dumps(dict(stage='init',output=str(out),records=len(data['records']),fresh_transients=0)))


def load_protocol(out):
    p=read(out/'protocol.json')
    if sha(out/'protocol.json')!=read(out/'protocol_lock.json')['sha256']:raise ValueError('Protocol changed')
    if sha(p['source'])!=p['source_sha256']:raise ValueError('Original mixed source changed')
    for name,h in {**p['source_hashes'],**p['model_hashes']}.items():
        if sha(ROOT/name)!=h:raise ValueError(f'Study source changed: {name}; initialize a fresh study')
    for name,h in read(out/'artifacts.json').items():
        if sha(out/name)!=h:raise ValueError(f'Study artifact changed: {name}')
    if verify_runtime()!=p['runtime']:raise ValueError('SPICE runtime changed')
    return p


def diagnostics(out,p,workers):
    if (out/'diagnostic.json').exists():raise FileExistsError('Diagnostic already complete')
    rng=np.random.default_rng(p['diagnostic']['seed']);n=p['diagnostic']['neighbors']
    from circuit import NAMES
    anchor=p['diagnostic'].get('anchor')
    if isinstance(anchor,dict):anchor=[anchor[k] for k in NAMES]
    center=encode(ADC_REFERENCE if anchor is None else anchor)
    points=[encode(ADC_REFERENCE)]
    if anchor is not None:points.append(center)
    points += [np.clip(center+rng.normal(0,p['diagnostic']['radius'],len(ADC_REFERENCE)),0,1) for _ in range(n)]
    cs=contexts(p,'C')
    def job(pair):
        i,u=pair;r=evaluate_suite(decode(u),cs,out/'diagnostic_waveforms'/str(i))
        row=dict(index=i,u=u.tolist(),x=decode(u).tolist(),oracle=r,cost=cost(r))
        write_json(out/'diagnostic_rows'/f'{i}.json',row)
        return row
    # Durable rows mean interrupted diagnostics need not lose their measured evidence.
    def cached(pair):
        i,u=pair;path=out/'diagnostic_rows'/f'{i}.json'
        if path.exists():
            r=read(path)
            if not np.allclose(r['u'],u,rtol=0,atol=1e-15):raise ValueError('Diagnostic point changed')
            return r
        return job(pair)
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(cached,enumerate(points)))
    feasible=[r for r in rows if r['oracle']['feasible']]
    chosen=min(feasible,key=lambda r:r['cost']) if feasible else None
    # Load/fast acquisition are separate diagnostic controls, not input-diversity C.
    stress_path=out/'stress_diagnostic.json'
    if stress_path.exists():stress=read(stress_path)
    else:
        stress=evaluate_suite(ADC_REFERENCE,[replace(Context(),name='large_cap',csample_pf=30.),
                                            replace(Context(),name='fast_acquisition',acquisition_ns=100.)],out/'stress_waveforms')
        write_json(stress_path,stress)
    result=dict(rows=rows,feasible_count=len(feasible),warm_start=None if chosen is None else chosen['x'],
                best=min(rows,key=lambda r:r['cost']),stress=stress,
                design_evaluations=len(rows)+1,transient_queries=len(rows)*len(cs)+2,
                scope='Reference + fixed local neighbors; bounded feasibility diagnosis, not an impossibility proof.')
    write_json(out/'diagnostic.json',result)
    artifacts=read(out/'artifacts.json');artifacts['diagnostic.json']=sha(out/'diagnostic.json');write_json(out/'artifacts.json',artifacts)
    print(json.dumps(dict(stage='diagnose',feasible=len(feasible),designs=len(rows),transients=result['transient_queries'])))


def prepare_c(out,p,workers):
    if (out/'data_C.json').exists():raise FileExistsError('C data already prepared')
    data=read(out/'data_A.json');cs=contexts(p,'C')
    def job(row):
        path=out/'additional_inputs'/f"{row['index']}.json"
        if path.exists():return read(path)
        nominal=relabel(row['oracle'],cs[:1])['contexts']
        extra=[evaluate(decode(row['u']),c) for c in cs[1:]]
        new=copy.deepcopy(row);new['oracle']=aggregate(nominal+extra)
        write_json(path,new);return new
    with ThreadPoolExecutor(max_workers=workers) as pool:records=list(pool.map(job,data['records']))
    added=len(records)*(len(cs)-1)
    data.update(records=records,contexts=[asdict(c) for c in cs],context_hash=fingerprint(cs),
                transient_queries=data['transient_queries']+added,
                reuse=dict(source_sha256=p['source_sha256'],additional_transient_queries=added,
                           original_transient_queries=data['transient_queries'],
                           additional_design_points=len(records),quality_membership_and_splits='unchanged'))
    write_json(out/'data_C.json',data)
    artifacts=read(out/'artifacts.json');artifacts['data_C.json']=sha(out/'data_C.json');write_json(out/'artifacts.json',artifacts)
    print(json.dumps(dict(stage='prepare-c',additional_transients=added)))


def compare(out,p,attention_only=False):
    from attention import SearchOptions
    from critics import CriticOptions
    from learning import train
    if (out/'endpoints.json').exists():raise ValueError('Final endpoints already locked')
    diag=read(out/'diagnostic.json')
    if not diag['warm_start']:
        raise ValueError('No C-feasible design in the diagnostic; inspect settling/turnoff/hold before optimization')
    if not (out/'data_C.json').exists():raise ValueError('Run prepare-c before comparison')
    cfg=p['comparison'];index_path=out/'runs.json'
    index=read(index_path) if index_path.exists() else []
    if attention_only:
        if not (out/'comparison_complete.json').exists():raise ValueError('Complete primary comparison first')
        chosen=read(out/'comparison_complete.json')['attention_selection']
        labels=[chosen['condition']];variants=[(chosen['method'],chosen['mode'],'off')]
    else:
        labels=['A','B','C']
        variants=[('ga','frozen','learned')]+[(m,c,'learned') for m in ('regression','rnc') for c in ('frozen','latest','buffer')]
    for label in labels:
        cs=contexts(p,label);data=out/f'data_{label}.json'
        if not data.exists():raise ValueError('Run prepare-c first')
        for seed in cfg['seeds']:
            for method,mode,att in variants:
                name=f'{label}_{method}_{mode}_{att}_s{seed}';folder=out/'runs'/name
                ckpt=None;options=None;critic=None
                if method!='ga':
                    ckpt=out/'checkpoints'/f'{label}_{method}_s{seed}.pt'
                    if not ckpt.exists():train(data,method,seed,ckpt,cfg['epochs'])
                    options=SearchOptions(attention=att,line_search='none')
                    critic=CriticOptions(mode=mode,source_data=str(data),epochs=cfg['critic_epochs'])
                trace=folder/'trace.json'
                if not trace.exists():
                    if (folder/'partial.json').exists():raise ValueError(f'Partial run at {folder}; preserve it and use a fresh study to avoid hiding spent SPICE')
                    optimize(method,cfg['spice_budget']//len(cs),seed,cs,folder,ckpt,options,critic,diag['warm_start'])
                run=read(trace)
                confirm=folder/'development.json'
                if not confirm.exists():
                    dev_cs=[replace(c,error_limit_mv=1.) for c in contexts(p,'C')]
                    if label=='C':r=relabel(run['best']['oracle'],dev_cs);fresh=0
                    else:
                        first=relabel(run['best']['oracle'],dev_cs[:1])['contexts']
                        r=aggregate(first+[evaluate(run['best']['x'],c) for c in dev_cs[1:]]);fresh=3
                    write_json(confirm,dict(oracle=r,transient_queries=fresh))
                item=dict(name=name,condition=label,method=method,mode=mode,attention=att,seed=seed,
                          trace=str(trace.resolve()),development=str(confirm.resolve()),trace_sha256=sha(trace),
                          development_sha256=sha(confirm))
                index=[r for r in index if r['name']!=name]+[item];write_json(index_path,index)
    if not attention_only:
        groups={}
        for item in index:
            if item['method']=='ga' or item['attention']!='learned':continue
            key=(item['condition'],item['method'],item['mode']);groups.setdefault(key,[]).append(read(item['development'])['oracle'])
        def rank(key):
            rs=groups[key];passed=[r for r in rs if r['feasible']]
            return (-len(passed)/len(rs),np.mean([r['mean_driver_energy_pj'] for r in passed]) if passed else float('inf'),key)
        key=min(groups,key=rank)
        write_json(out/'comparison_complete.json',dict(attention_selection=dict(zip(('condition','method','mode'),key)),rule=p['selection_rule']))
    else:write_json(out/'attention_complete.json',dict(selected=chosen))


def final_test(out,p):
    if not (out/'attention_complete.json').exists():raise ValueError('Complete primary and selected attention comparison before final test')
    index=read(out/'runs.json');lock=out/'endpoints.json'
    endpoints=[dict(**r,x=read(r['trace'])['best']['x']) for r in index]
    for r in index:
        if sha(r['trace'])!=r['trace_sha256'] or sha(r['development'])!=r['development_sha256']:
            raise ValueError('A search/development result changed')
    if lock.exists():
        if read(lock)!=endpoints:raise ValueError('Locked endpoints changed')
    else:write_json(lock,endpoints)
    cs=[Context.from_dict(c) for c in p['final_contexts']];groups={};total=0
    for item in endpoints:
        path=out/'final_tests'/f"{item['name']}.json"
        if path.exists():r=read(path)
        else:
            r=evaluate_suite(item['x'],cs);write_json(path,r)
        run=read(item['trace']);total+=len(cs)
        key=f"{item['condition']}_{item['method']}_{item['mode']}_{item['attention']}"
        groups.setdefault(key,[]).append(dict(seed=item['seed'],all_inputs_pass=r['feasible'],
                         energy_pj=r['mean_driver_energy_pj'] if r['feasible'] else None,
                         worst_hold_error_mv=r['worst_hold_error_mv'],
                         search_transients=run['transient_queries'],returned_design_query=run['best']['query'],
                         returned_design_found_at_transients=run['best']['query']*len(run['contexts']),
                         first_success_design_query=run['first_success'],
                         first_development_success_transients=None if run['first_success'] is None else run['first_success']*len(run['contexts'])))
    summary={}
    for key,rs in groups.items():
        energies=[r['energy_pj'] for r in rs if r['all_inputs_pass']]
        summary[key]=dict(designs=len(rs),all_inputs_pass_rate=len(energies)/len(rs),
                          passing_energy_mean_pj=float(np.mean(energies)) if energies else None,
                          passing_energies_pj=energies,runs=rs)
    costs=dict(original_shared_offline_transients=read(p['source'])['transient_queries'],
               added_C_offline_transients=read(out/'data_C.json')['reuse']['additional_transient_queries'],
               new_C_offline_transients_this_study=0 if p.get('prepared_c') else read(out/'data_C.json')['reuse']['additional_transient_queries'],
               diagnostic_transients=read(out/'diagnostic.json')['transient_queries'],
               online_search_transients=sum(read(r['trace'])['transient_queries'] for r in index),
               development_confirmation_transients=sum(read(r['development'])['transient_queries'] for r in index),
               final_test_transients=total)
    attempts=[read(f) for f in (out/'spice_attempts').glob('*.json')]
    costs.update(recorded_spice_attempts=len(attempts),completed_spice_attempts=sum(r['completed'] for r in attempts),
                 interrupted_or_pending_attempts=sum(not r['completed'] for r in attempts))
    write_json(out/'summary.json',dict(groups=summary,costs=costs,final_limit_mv=1.,
               note='Passing-design energy is conditional; no pass means null, never zero. Final input suite is fixed and finite.'))
    print(json.dumps(dict(stage='test',summary=str(out/'summary.json'),costs=costs)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['init','diagnose','prepare-c','compare','attention','test'])
    parser.add_argument('--output',required=True);parser.add_argument('--source')
    parser.add_argument('--diagnostic-anchor',help='Prior development design JSON to include and center diagnostic neighbors on')
    parser.add_argument('--prepared-c',help='Already measured C dataset from this exact mixed source; no new offline SPICE')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--neighbors',type=int,default=32)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--spice-budget',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=500);parser.add_argument('--critic-epochs',type=int,default=20)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.stage=='init':
        if not args.source:parser.error('init requires --source')
        init(args);return
    out=Path(args.output);p=load_protocol(out)
    global evaluate
    evaluate=install_accounting(out)
    if args.stage=='diagnose':diagnostics(out,p,args.workers)
    elif args.stage=='prepare-c':prepare_c(out,p,args.workers)
    elif args.stage=='compare':compare(out,p)
    elif args.stage=='attention':compare(out,p,True)
    else:final_test(out,p)


if __name__=='__main__':main()
