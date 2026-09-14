"""Recompute the supplied demonstration evidence without new SPICE calls."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import importlib.metadata

import numpy as np

from circuit import ROOT,MODEL,MODEL_SHA256,Context,cost,write_json,measure
from search import fingerprint,initial
from plots import searches


def omit_time(value):
    if isinstance(value,dict):return {k:omit_time(v) for k,v in value.items() if k!='elapsed_seconds'}
    if isinstance(value,list):return [omit_time(v) for v in value]
    return value


def main():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA256
    methods=['random','ga','regression','rnc'];runs={};summary=[]
    for method in methods:
        p=ROOT/f'results/demo/{method}/trace.json';r=json.loads(p.read_text());runs[method]=r
        assert r['context_hash']==fingerprint([Context()])
        assert r['design_queries']==r['transient_queries']==len(r['trace'])==16
        best=None;first=None
        for j,row in enumerate(r['trace'],1):
            assert row['query']==j
            assert np.all((np.array(row['u'])>=0)&(np.array(row['u'])<=1))
            oracle=row['oracle']
            for result in oracle['contexts']:
                if result['valid']:
                    assert len(result['cycles'])==len(result['context']['levels_v'])
                    error=max(c['worst_hold_error_mv'] for c in result['cycles'])
                    energy=np.mean([c['amplifier_energy_pj']+c['clock_energy_pj'] for c in result['cycles']])
                    assert abs(error-result['worst_hold_error_mv'])<1e-10
                    assert abs(energy-result['mean_driver_energy_pj'])<1e-8
                    assert result['feasible']==(error<=result['context']['error_limit_mv'])
            assert abs(cost(oracle)-row['cost'])<1e-12
            if best is None or row['cost']<best['cost']:best=row
            if oracle['feasible'] and first is None:first=j
            assert row['first_success']==first and abs(row['best_cost']-best['cost'])<1e-12
        assert r['best']==best
        summary.append(dict(method=method,budget=16,first_success=first,
            final_error_mv=best['oracle']['worst_hold_error_mv'],
            final_energy_pj=best['oracle']['mean_driver_energy_pj'],
            improves_reference=best['oracle']['mean_driver_energy_pj']<r['trace'][2]['oracle']['mean_driver_energy_pj']-1e-9))
    matched=0
    for method in methods[1:]:
        for a,b in zip(runs['random']['trace'][:8],runs[method]['trace'][:8]):
            assert a['u']==b['u'] and omit_time(a['oracle'])==omit_time(b['oracle'])
            matched+=1
    source=json.loads((ROOT/'data/source_smoke.json').read_text())
    assert source['design_queries']==len(source['records'])==24
    train=[r for r in source['records'] if r['split']==0]
    feasible=[r for r in train if r['oracle']['feasible']]
    lookup=min(feasible,key=lambda r:r['oracle']['mean_driver_energy_pj']) if feasible else None
    dense=json.loads((ROOT/'results/validation/reference_dense/summary.json').read_text())
    base=runs['random']['trace'][2]['oracle']
    assert base['feasible'] and dense['feasible']
    error_delta=abs(dense['worst_hold_error_mv']-base['worst_hold_error_mv'])
    assert error_delta<.05
    standalone=json.loads((ROOT/'results/validation/standalone/metrics.json').read_text())
    assert abs(standalone['worst_hold_error_mv']-base['worst_hold_error_mv'])<1e-9
    assert abs(standalone['mean_driver_energy_pj']-base['mean_driver_energy_pj'])<1e-9
    # Inspect exact stored raw data, independently of the summary JSON.
    raw=np.loadtxt(ROOT/'results/validation/standalone/waveform.dat',skiprows=1)
    direct=measure(raw,Context())
    assert abs(direct['worst_hold_error_mv']-standalone['worst_hold_error_mv'])<1e-12
    stress=json.loads((ROOT/'results/validation/stress/summary.json').read_text())
    stress_rows=[dict(name=r['context']['name'],feasible=r['feasible'],error_mv=r.get('worst_hold_error_mv'),
                      energy_pj=r.get('mean_driver_energy_pj')) for r in stress['contexts']]
    learning_checks={}
    try:
        import torch
        from rnc import rank_n_contrast,rnc_reference
        torch.set_num_threads(1);torch.manual_seed(77)
        z=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[0,1],[0,1],[1,0],[2,1],[3,.5]],dtype=torch.float64)
        a=rank_n_contrast(z,y);b=rnc_reference(z,y)
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
        assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
        assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-12)
        for method in ['regression','rnc']:
            p=torch.load(ROOT/f'checkpoints/{method}.pt',weights_only=False,map_location='cpu')
            assert p['source_sha256']==hashlib.sha256((ROOT/'data/source_smoke.json').read_bytes()).hexdigest()
            assert p['valid_counts']=={'train':16,'validation':4,'test':4}
        learning_checks=dict(rnc_value_and_gradient_match_reference=True,checkpoint_data_splits_match=True)
    except ImportError:
        learning_checks=dict(skipped='Optional torch dependency is not installed')
    result=dict(design_queries_source=24,design_queries_search=64,initial_pairs_matched=matched,
                source_training_feasible_designs=len(feasible),
                source_lookup_index=None if lookup is None else lookup['index'],
                source_lookup_energy_pj=None if lookup is None else lookup['oracle']['mean_driver_energy_pj'],
                dense_error_mv=dense['worst_hold_error_mv'],dense_minus_base_abs_error_mv=error_delta,
                dense_energy_pj=dense['mean_driver_energy_pj'],standalone_replay_matches=True,
                methods=summary,stress=stress_rows,learning=learning_checks)
    write_json(ROOT/'results/validation/audit.json',result)
    with (ROOT/'results/methods.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    with (ROOT/'results/stress.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stress_rows[0]));writer.writeheader();writer.writerows(stress_rows)
    searches([ROOT/f'results/demo/{m}/trace.json' for m in methods],ROOT/'results/demo/searches.png')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
