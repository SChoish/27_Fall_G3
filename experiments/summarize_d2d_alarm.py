"""Summarize a completed D2D ALARM campaign without additional simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(root):
    root=Path(root)
    s=json.loads((root/'summary.json').read_text())
    p=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==s['protocol_sha256']
    assert hashlib.sha256((root/'selections_locked.json').read_bytes()).hexdigest()==s['selection_sha256']
    n=len(p['seeds'])
    initial_rows={seed:json.loads((root/f'seed_{seed}'/'initial.json').read_text()) for seed in p['seeds']}
    initial_returns=lambda r:any(np.allclose(r['x'],a['x'],rtol=1e-10,atol=1e-12) for a in initial_rows[r['seed']])
    rows=[]
    for method in p['methods']:
        runs=[r for r in s['results'] if r['method']==method]
        good=[r for r in runs if r['confirmation']['feasible']]
        values=[r['savings_percent'] for r in good]
        rows.append(dict(method=method,passed=len(good),total=n,
                         energy_mean=float(np.mean([r['confirmation']['energy_pj_bit'] for r in good])) if good else None,
                         savings_mean=float(np.mean(values)) if good else None,
                         savings_std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                         savings_by_seed=[r['savings_percent'] for r in runs],
                         improved_and_passed=sum(r['confirmation']['feasible'] and r['savings_percent']>1e-6 for r in runs),
                         nominal_returns=sum(r['unchanged_nominal'] for r in runs),
                         initial_set_returns=sum(initial_returns(r) for r in runs),
                         policy_seconds=sum(r['policy_seconds'] for r in runs)))
    costs=s['search_costs']+[s['confirmation_cost']]
    total={k:sum(c[k] for c in costs) for k in ('design_queries','logical_measurements','spice_transient_attempts','cache_hits','failed_transients','completed_transients')}
    def fmt(x,d=2):return '—' if x is None else f'{x:.{d}f}'
    lines=['# D2D RC sizing: ALARM pilot','',
           'ALARM, the existing regression+buffer+attention optimizer, and random search were compared on the same transistor-level RC circuit.',
           f'Three methods share {p["initial"]} measured initial designs per seed, including nominal. Each receives {p["additional"]} additional design queries. No external offline dataset is used.',
           '','| Method | Confirmation pass | Passed and improved | Mean energy pJ/bit (pass only) | Savings mean ± sample std % (pass only) | Initial-set returns |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["passed"]}/{n} | {r["improved_and_passed"]}/{n} | {fmt(r["energy_mean"],6)} | {fmt(r["savings_mean"])} ± {fmt(r["savings_std"])} | {r["initial_set_returns"]}/{n} |')
    lines+=['','## Seed-level results','','| Seed | Method | All confirmation inputs pass | Savings % | Energy pJ/bit |','|---:|---|---|---:|---:|']
    for r in s['results']:
        lines.append(f'| {r["seed"]} | {r["method"]} | {r["confirmation"]["feasible"]} | {fmt(r["savings_percent"])} | {fmt(r["confirmation"].get("energy_pj_bit"),6)} |')
    lines+=['','## Scope and fairness','',
           '- Six sizing variables, two coupled RC lanes, 72 MOS and behavioral RX. PTM MOS model; no new PDK extraction.',
           '- All algorithms use the same initial measured rows, logarithmic parameter bounds, four development cases, physical evaluator and feasible return rule. Failed simulator calls count against the budget.',
           '- ALARM uses the pinned upstream basic Config with a custom scalarizer that exactly preserves circuit.cost, including margin, overshoot and error constraints. The original optimizer code is not edited. No Lu VCO weights or profile are used.',
           '- Regression uses 100 initial epochs with the last checkpoint, then 20 epochs per update, learned attention and buffer. This is a current equal-data pilot; older sizing campaigns used different offline data and are not directly comparable.',
           '- Returning a design from the common initial set is not an improvement produced by the additional search. Initial-set return counts are reported explicitly.',
           '- This compares complete algorithms, not an isolated buffer, attention or line-search ablation. Neural architectures and training computation differ.',
           '- The eight confirmation cases use predefined packets at the same two development channel contexts. All return designs are locked before any confirmation evaluation. New-channel or PVT generalization is not established.',
           '- Three optimization seeds are not independent physical circuits. Means over passing designs are conditional; failures remain failures.',
           '', '## ALARM execution','','| Seed | Critic fits | Attention updates | Buffer pushes | Head transfers | Duplicate fallback |','|---:|---:|---:|---:|---:|---:|']
    for r in s['results']:
        if r['method']=='alarm':
            t=r['stats'];lines.append(f'| {r["seed"]} | {t["critic_fits"]} | {t["attention_updates"]} | {t["buffer_pushes"]} | {t["head_transfers"]} | {t["duplicate_fallbacks"]} |')
    lines+=['','## Cost','',
            f'Each method has {4*(p["initial"]+p["additional"])} logical case measurements per seed for initialization plus search. Shared initial measurements are physically collected only once per seed.',
            'Actual total costs, including confirmation:', '', '```json',json.dumps(total,indent=2),'```','',
            'Cache hits remain logical requests and do not count as new SPICE transients. Surrogate backtracking does not query SPICE.',
            '', 'Protocol, locked return designs, aggregate and per-query results are retained. No hyperparameter sweep or post-result checkpoint selection was performed.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'aggregate.json').write_text(json.dumps(dict(rows=rows,total_cost=total),indent=2)+'\n')
    print(json.dumps(dict(rows=rows,total_cost=total),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    summarize(p.parse_args().output)
