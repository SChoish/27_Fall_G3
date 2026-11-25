"""Matched historical development data for BO and neural sizing proposals."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from experiments.constrained_search import ConstrainedSearch
from experiments.d2d_replay_search import propose
from experiments.robust_search_ablation import (
    ROOT, d2d, file_hash, initial_designs, new_suites, response, Measurer,
    shortlist, validated_selection)


SOURCE = ROOT/'implementation_results/20261125_robust_ablation/d2d'


def load_archive(cases, binary):
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    expected = [asdict(c) for c in cases]
    if protocol['development'] != expected or protocol['binary_sha256'] != file_hash(binary):
        raise ValueError('Archive development suite or ngspice differs')
    for name, digest in protocol['source_sha256'].items():
        if name.startswith(('d2d_sizing/circuit', 'd2d_coupled_calibration/')) and file_hash(ROOT/name) != digest:
            raise ValueError(f'Archive circuit/model mismatch: {name}')
    unique = {}
    inputs = {}
    for seed in (0,1,2):
        for method in ('legacy','max_constraint'):
            path = SOURCE/f'seed_{seed}'/f'{method}.json'
            inputs[str(path.relative_to(ROOT))] = file_hash(path)
            for item in json.loads(path.read_text()):
                row = dict(x=item['x'], oracle=item['oracle'])
                if [c['case'] for c in row['oracle']['cases']] != expected:
                    raise ValueError('Unexpected archive case')
                key = tuple(row['x'])
                if key in unique and unique[key]['oracle'] != row['oracle']:
                    raise ValueError('Inconsistent repeated archive measurement')
                unique[key] = row
    return list(unique.values()), inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=2)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    short,val,old_test = new_suites('d2d')
    long = [replace(c,bits=128) for c in val]
    s = Measurer('d2d',short,out/'short',args.workers)
    h = Measurer('d2d',long,out/'long',args.workers)
    archive,sources = load_archive(short,s.backend.binary)
    tests = {k:[replace(c,seed=c.seed+99300) for c in v] for k,v in old_test.items()}
    d2d.write_json(out/'protocol.json',dict(seed=args.seed,archive_sources=sources,
        archive_designs=len(archive),archive_unique_case_equivalents=4*len(archive),
        binary=file_hash(s.backend.binary), suites={k:[asdict(c) for c in v]
        for k,v in dict(short=short,long=long,**tests).items()}))
    locked = {}
    for name in ('archive_only','archive_bo','archive_mlp'):
        bo = ConstrainedSearch(6,12,args.seed)
        if name=='archive_bo':
            for r in archive:
                bo.observe(np.clip(d2d.encode(r['x']),0,1),*response('d2d',r['oracle']))
        known = {tuple(r['x']):r for r in archive}
        fresh = []
        initial = initial_designs('d2d',args.seed)
        for step in range(0 if name=='archive_only' else 18):
            if step<8:
                x = initial[step]
                u = np.clip(d2d.encode(x),0,1)
            elif name=='archive_bo':
                u = bo.ask()
                x = d2d.decode(u)
            else:
                x = propose(list(known.values()), [], args.seed)
            row = dict(x=x.tolist(),oracle=s(x))
            if name=='archive_bo' and tuple(x) not in known:
                bo.observe(u,*response('d2d',row['oracle']))
            known[tuple(x)] = row
            fresh.append(row)
        candidates = shortlist('d2d',list(known.values()))
        validations = [h(r['x']) for r in candidates]
        d2d.write_json(out/name/'trace.json',dict(fresh=fresh,checked=candidates,measured=validations))
        locked[name] = dict(selection=validated_selection('d2d',candidates,validations),
                            logical_cases=4*len(fresh)+8*len(validations))
    d2d.write_json(out/'locked.json',locked)
    results = {}
    for split,cases in tests.items():
        t = Measurer('d2d',cases,out/split,args.workers)
        results[split] = {m:t(v['selection']['design']) if v['selection'] else None
                          for m,v in locked.items()}
    d2d.write_json(out/'summary.json',dict(results=results,seed=args.seed,
        locked=file_hash(out/'locked.json'), short=s.accounting(), long=h.accounting()))


if __name__ == '__main__':
    main()
