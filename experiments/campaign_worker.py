"""Workers for a frozen, multi-seed physical training campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np


def adc_collect(args):
    from scipy.stats import qmc
    from circuit import (NAMES, NOMINAL, PRIOR_OPAMP, ADC_REFERENCE, encode, decode,
                         evaluate_suite, load_contexts, write_json, verify_runtime)
    from search import fingerprint
    contexts = load_contexts(args.contexts)
    verify_runtime()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = path.with_suffix('.jsonl')
    n = args.n
    u = qmc.LatinHypercube(len(NAMES), seed=args.seed).random(n)
    anchors = np.stack([encode(x) for x in (NOMINAL, PRIOR_OPAMP, ADC_REFERENCE)])
    u[:3] = anchors
    local_count = n // 4
    rng = np.random.default_rng(args.seed + 51091)
    for k in range(local_count):
        u[n-local_count+k] = np.clip(anchors[k % 3] + rng.normal(0, [.08, .16][k % 2], len(NAMES)), 0, 1)
    order = np.random.default_rng(args.seed + 319).permutation(n)
    split = np.full(n, 2, dtype=int)
    split[order[:int(.7*n)]] = 0
    split[order[int(.7*n):int(.85*n)]] = 1
    rows = {}
    if journal.exists():
        lines = journal.read_text().splitlines()
        valid_lines = []
        for j, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if j != len(lines)-1:
                    raise
                break  # A terminated final write is not a completed measurement.
            i = row['index']
            if row['u'] != u[i].tolist() or row['context_hash'] != fingerprint(contexts):
                raise ValueError('Collection journal contract mismatch')
            rows[i] = row
            valid_lines.append(line)
        journal.write_text('\n'.join(valid_lines) + ('\n' if valid_lines else ''))
    started = time.perf_counter()
    def one(i):
        return dict(index=i, u=u[i].tolist(), split=int(split[i]),
                    context_hash=fingerprint(contexts),
                    sampling='fixed_reference' if i < 3 else 'local_reference' if i >= n-local_count else 'latin_hypercube',
                    oracle=evaluate_suite(decode(u[i]), contexts))
    with journal.open('a', buffering=1) as log, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, i) for i in range(n) if i not in rows]
        for f in as_completed(futures):
            row = f.result()
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            rows[row['index']] = row
            if len(rows) % 32 == 0:
                print(json.dumps(dict(stage='adc_collect', completed=len(rows), total=n,
                      valid=sum(r['oracle']['valid'] for r in rows.values()),
                      feasible=sum(r['oracle']['feasible'] for r in rows.values()),
                      seconds=time.perf_counter()-started)), flush=True)
    result = dict(seed=args.seed, contexts=[asdict(c) for c in contexts], context_hash=fingerprint(contexts),
                  names=NAMES, records=[rows[i] for i in range(n)], design_queries=n,
                  transient_queries=n*len(contexts), split_codes={'train':0,'validation':1,'test':2},
                  sampling_protocol='Three documented references, 25% local perturbations around these references, remaining Latin hypercube. Fixed before measurements.',
                  elapsed_seconds_this_invocation=time.perf_counter()-started)
    write_json(path, result)


def adc_partition(args):
    import hashlib
    from circuit import cost, write_json
    source = Path(args.input)
    data = json.loads(source.read_text())
    training = sorted([r for r in data['records'] if r['split'] == 0], key=lambda r: (cost(r['oracle']), r['index']))
    heldout = [r for r in data['records'] if r['split'] != 0]
    size = len(training)//3
    tiers = {'high': training[:size], 'medium': training[size:2*size], 'low': training[-size:]}
    rng = np.random.default_rng(739201)
    mixed = []
    for j, quality in enumerate(('low','medium','high')):
        count = size//3 + (j < size%3)
        mixed += [tiers[quality][int(k)] for k in rng.choice(size, count, replace=False)]
    tiers['mixed'] = mixed
    for quality, rows in tiers.items():
        output = dict(data)
        output.update(records=sorted(rows+heldout,key=lambda r:r['index']), quality=quality,
                      source_pool_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      quality_definition='Training-only observed constraint-first cost tertiles; mixed balanced across tiers. Validation/test identical and untouched.',
                      visible_training_designs=len(rows), visible_validation_test_designs=len(heldout),
                      collection_cost_note='design_queries/transient_queries charge the entire shared source pool, including records excluded by quality filtering.')
        write_json(Path(args.output)/f'{quality}.json', output)
        print(quality, 'training',len(rows),'feasible',sum(r['oracle']['feasible'] for r in rows),flush=True)


def d2d_collect(args):
    from coupled_experiment import OUT, config, Table, Environment, Search, random_setting, utility
    from coupled_link import CoupledSetting
    from provenance import write_metadata, file_hash
    cfg=config();table=Table('train');rng=np.random.default_rng(602491)
    keys=['observations','actions','rewards','next_observations','terminals','timeouts','context_ids','episode_ids','behavior_ids']
    episodes=[];ep=0
    for i in table.ids:
        for rep in range(12):
            method=['random','traditional','perturbed'][rep%3]
            initial=CoupledSetting() if rep%2 else random_setting(rng)
            env=Environment(table,i,initial);search=Search(method,seed=10000*i+rep+1123)
            search.observe(initial,env.measurement);obs=env.obs();rows={k:[] for k in keys};scores=[]
            for step in range(cfg['horizon']):
                setting=search.suggest();action=setting.action()
                nxt,r,t,timeout,m=env.step(action);search.observe(setting,m)
                for k,v in zip(keys,[obs,action,r,nxt,t,timeout,i,ep,rep%3]):rows[k].append(v)
                scores.append(utility(m,cfg['margin']));obs=nxt
            episodes.append(dict(context_id=i,episode_id=ep,behavior=method,
                quality_score=float(np.mean(scores)),final_best_utility=utility(env.best[1],cfg['margin']),arrays=rows))
            ep+=1
        print('Collected offline episodes',ep,'/',len(table.ids)*12,flush=True)
    def save(path, selected):
        arrays={k:np.concatenate([np.asarray(e['arrays'][k]) for e in selected],axis=0) for k in keys}
        for k in ['observations','actions','rewards','next_observations']:arrays[k]=arrays[k].astype(np.float32)
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
        write_metadata(path,cfg,dependencies=[OUT/'train_table.npz'])
        return arrays
    save(OUT/'dataset_pool.npz',episodes)
    tiers={q:[] for q in ('low','medium','high','mixed')}
    for i in table.ids:
        ordered=sorted([e for e in episodes if e['context_id']==i],key=lambda e:(e['quality_score'],e['episode_id']))
        groups=[ordered[:4],ordered[4:8],ordered[8:]]
        for q,g in zip(('low','medium','high'),groups):tiers[q].extend(g)
        # Four episodes per context; extra tier rotates to balance the full set.
        tiers['mixed'].extend([g[0] for g in groups]+[groups[i%3][1]])
    for quality,selected in tiers.items():
        selected.sort(key=lambda e:e['episode_id'])
        folder=OUT/'quality_datasets'/quality;arrays=save(folder/'dataset.npz',selected)
        meta=dict(quality=quality,transitions=len(arrays['rewards']),episodes=len(selected),
            source_pool_transitions=sum(len(e['arrays']['rewards']) for e in episodes),
            source_pool_episodes=len(episodes),logical_measurement_bundles=table.measurements,
            dataset_sha256=file_hash(folder/'dataset.npz'),source_pool_sha256=file_hash(OUT/'dataset_pool.npz'),
            quality_definition='Within each training context, rank complete episodes by mean measured per-query utility. Four bottom/middle/top episodes; mixed reuses one per tier plus a rotating extra tier. No validation or test feedback.',
            quality_score_mean=float(np.mean([e['quality_score'] for e in selected])),
            final_best_utility_mean=float(np.mean([e['final_best_utility'] for e in selected])),
            episode_ids=[e['episode_id'] for e in selected],
            source_cost_note='Every type is charged the full shared pool acquisition cost; mixed adds no measurements.')
        (folder/'dataset.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta,flush=True)


def adc_search(args):
    from circuit import load_contexts
    from search import optimize
    from attention import SearchOptions
    options = None if args.method in ('random', 'ga') else SearchOptions(attention=args.attention, line_search=args.line_search)
    optimize(args.method, args.budget, args.seed, load_contexts(args.contexts), args.output, args.checkpoint, options)


def adc_lookup(args):
    from circuit import cost, decode, evaluate_suite, load_contexts, write_json
    data=json.loads(Path(args.input).read_text());contexts=load_contexts(args.contexts)
    selected=min((r for r in data['records'] if r['split']==0),key=lambda r:cost(r['oracle']))
    x=decode(selected['u']);oracle=evaluate_suite(x,contexts)
    row=dict(query=1,u=selected['u'],x=x.tolist(),oracle=oracle,cost=cost(oracle))
    write_json(Path(args.output)/'trace.json',dict(method='lookup',budget=1,design_queries=1,
        transient_queries=len(contexts),contexts=[asdict(c) for c in contexts],trace=[row],best=row,
        first_success=1 if oracle['feasible'] else None,source_design_queries=data['design_queries'],
        source_index=selected['index']))


def adc_confirm(args):
    from circuit import Context, evaluate_suite, write_json
    run = json.loads(Path(args.input).read_text())
    contexts = [Context.from_dict(c) for c in run['contexts']]
    design = run['best']['x']
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    dense = evaluate_suite(design, [replace(c, maxstep_ns=c.maxstep_ns/4) for c in contexts])
    # Prespecified new input sequences; never used to choose search endpoints.
    patterns = [[.8, 1.1, .8, 1.1, .8, 1.1]]
    patterns += [np.random.default_rng(seed).uniform(.8, 1.1, 6).tolist() for seed in (271091, 271092)]
    heldout = [evaluate_suite(design, [replace(c, levels_v=tuple(levels)) for c in contexts]) for levels in patterns]
    write_json(folder/'confirmation.json', dict(dense=dense, heldout=heldout,
               independent_quality_all=all(r['feasible'] for r in heldout),
               endpoint_fixed_before_confirmation=True))


def d2d_oracle(args):
    from coupled_experiment import config, OUT, cached_wave, phase_table
    cfg = config()
    rows = []
    for i in cfg['test_ids']:
        def one(pair):
            a, b = pair
            vals = []
            for pattern in cfg['confirmation_patterns']:
                c, bits, wave = cached_wave(i, a, b, pattern)
                margins, errors = phase_table(wave, c, bits, cfg['guard_ui'])
                vals.append((margins, errors, float(wave['energy_pj_bit'])))
            valid = np.stack([(m >= cfg['margin']) & (e == 0) for m,e,en in vals]).all(axis=0)
            return a,b,bool(valid.any(axis=-1).all()),sum(v[2] for v in vals)
        with ThreadPoolExecutor(args.workers) as pool:
            options = list(pool.map(one, [(a,b) for a in range(4) for b in range(4)]))
        reference = next(en for a,b,ok,en in options if (a,b)==(3,3))
        energies = [en/reference for a,b,ok,en in options if ok]
        best = min(energies) if energies else None
        rows.append(dict(context_id=i, minimum_energy_ratio=best,
                         feasible_by_ratio={str(r): best is not None and best <= r for r in cfg['ratios']}))
        (OUT/'independent_oracle_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('Independent grid',len(rows),'/',len(cfg['test_ids']),flush=True)
    (OUT/'independent_oracle.json').write_text(json.dumps(dict(
        scope='Post-evaluation full 4x4x65x65 grid on the same three independent packets; no calibration eligibility restriction, never policy input.',
        records=rows),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command', choices=['adc_collect','adc_partition','adc_search','adc_lookup','adc_confirm','d2d_collect','d2d_oracle'])
    p.add_argument('--project',required=True)
    p.add_argument('--output');p.add_argument('--input');p.add_argument('--contexts')
    p.add_argument('--n',type=int,default=1024);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--budget',type=int,default=128)
    p.add_argument('--method');p.add_argument('--checkpoint')
    p.add_argument('--attention',default='learned');p.add_argument('--line-search',default='none')
    args=p.parse_args();sys.path.insert(0,str(Path(args.project).resolve()))
    globals()[args.command](args)

if __name__=='__main__':main()
