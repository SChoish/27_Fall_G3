"""Create scientific figures from a completed campaign; never select or train models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    summary=json.loads((root/'summary.json').read_text());out=root/'figures';out.mkdir(exist_ok=True)
    qualities=['low','medium','high','mixed'];colors=['#9a607f','#4777a8','#408a69','#bf8532']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ax=axes[0,0]
    vals=[summary['d2d_quality_data'][q]['quality_score_mean'] for q in qualities]
    ax.bar(qualities,vals,color=colors);ax.set(title='D2D: observed source-episode quality',ylabel='Mean measured utility (higher is better)')
    ax=axes[0,1]
    for j,q in enumerate(qualities):
        rows=[r for r in summary['adc_runs'] if r['task']=='nominal' and r['quality']==q and r['method']=='rnc' and r['attention']=='learned' and r['line_search']=='none']
        ys=[r['energy_pj'] for r in rows if r['feasible']]
        ax.scatter(j+np.linspace(-.1,.1,len(ys)),ys,color=colors[j],s=45)
        independent=sum(r['independent_quality'] for r in rows)
        ax.text(j,.98,f'Original: {len(ys)}/{len(rows)}\nNew inputs: {independent}/{len(rows)}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
    ax.set(xticks=range(4),xticklabels=qualities,title='ADC: RNC + attention, original-input energy',ylabel='Nominal-feasible energy (pJ/sample)')
    ax=axes[1,0]
    for method,shift,color in [('bc',-.18,'#64748b'),('td3bc',.18,'#3c6c96')]:
        values=[next(r for r in summary['d2d'] if r['quality']==q and r['method']==method) for q in qualities]
        ax.bar(np.arange(4)+shift,[r['joint_success_percent'] for r in values],width=.34,label=method,color=color)
        for j,r in enumerate(values):
            ys=[v['success_percent'] for v in r['seeds']]
            ax.scatter(j+shift+np.linspace(-.06,.06,len(ys)),ys,color='black',s=14,zorder=3)
    baseline=next(r['joint_success_percent'] for r in summary['d2d'] if r['quality']=='mixed' and r['method']=='traditional')
    ceiling=100*summary['d2d_independent_grid_feasible_counts']['0.9']/32
    ax.axhline(baseline,color='#bd742f',ls='--',label='Coordinate search')
    ax.axhline(ceiling,color='#888888',ls=':',label='Independent-grid feasibility')
    ax.set(xticks=range(4),xticklabels=qualities,title='D2D: independent quality + 10% saving',ylabel='Success (%) at 16 additional queries',ylim=(-1,max(10,ceiling+5)))
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for q,color in zip(qualities,colors):
        runs=[r for r in summary['adc_runs'] if r['task']=='nominal' and r['quality']==q and r['method']=='rnc' and r['attention']=='learned' and r['line_search']=='none']
        curves=[]
        for r in runs:
            trace=json.loads((root/r['path']/'trace.json').read_text())['trace'];best=np.nan;curve=[]
            for row in trace:
                m=row['oracle']
                if m['feasible']:best=min(best,m['mean_driver_energy_pj']) if np.isfinite(best) else m['mean_driver_energy_pj']
                curve.append(best)
            curves.append(curve)
        arr=np.array(curves);mask=np.isfinite(arr).any(axis=0)
        ax.plot(np.arange(1,arr.shape[1]+1)[mask],np.nanmedian(arr[:,mask],axis=0),label=q,color=color)
    ax.set(title='ADC: search progress on original input',xlabel='Charged design queries',ylabel='Median incumbent feasible energy (pJ/sample)');ax.legend()
    for ax in axes.flat:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Quality datasets | shared paid source pools | three seeds',fontsize=14)
    fig.savefig(out/'quality_comparison.png',dpi=180);fig.savefig(out/'quality_comparison.pdf');plt.close(fig)
    costs=dict(summary['costs'])
    d2d=root/'d2d_coupled_calibration/results/coupled_v1'
    costs['d2d_source_preparation_spice_transients']=json.loads((d2d/'train_simulation.json').read_text())['transients']
    costs['d2d_validation_preparation_spice_transients']=json.loads((d2d/'validation_simulation.json').read_text())['transients']
    costs['d2d_test_preparation_spice_transients']=json.loads((d2d/'test_simulation.json').read_text())['transients']
    costs['d2d_independent_grid_configured_transients']=32*16*3
    costs['d2d_dense_verification_transients']=4*3
    costs['adc_final_verification_transients']=sum((4 if r['task']=='robust' else 1)*4 for r in summary['adc_runs'])
    (out/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
    inputs=[root/'summary.json']
    for r in summary['adc_runs']:
        if r['task']=='nominal' and r['method']=='rnc' and r['attention']=='learned' and r['line_search']=='none':inputs.append(root/r['path']/'trace.json')
    (out/'provenance.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
