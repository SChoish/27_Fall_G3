"""Post-hoc diagnosis of frozen results; never trains, selects policies or calls SPICE."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=read(root/'source_hashes.json');manifest=read(root/'execution/manifest.json')
    source_checks={name:digest(root/name)==checksum for name,checksum in frozen.items()}
    input_checks={x['path']:digest(root/x['path'])==x['sha256']==x['sha256_after'] for x in manifest['input_artifacts']}
    assert all(source_checks.values()) and all(input_checks.values())
    versions={}
    for name in frozen:
        if not name.endswith('.py'):continue
        repo_name=name if '/' in name else 'experiments/'+name
        row={'snapshot_sha256':digest(root/name)}
        for commit in ('fba8520','eda37ac'):
            r=subprocess.run(['git','show',commit+':'+repo_name],cwd=a.repo,capture_output=True)
            row[commit]=dict(exists=r.returncode==0,sha256=hashlib.sha256(r.stdout).hexdigest() if r.returncode==0 else None,
                             identical=r.returncode==0 and r.stdout==(root/name).read_bytes())
        versions[name]=row
    runs=read(root/'adc_runs.json');adc=[]
    for desc in runs:
        if desc['task']!='nominal':continue
        trace=read(root/desc['path']/'trace.json');conf=read(root/desc['path']/'confirmation.json')
        cases=[]
        for i,suite in enumerate(conf['heldout']):
            context=max(suite['contexts'],key=lambda x:x['worst_hold_error_mv'])
            cycle=max(context['cycles'],key=lambda x:x['worst_hold_error_mv'])
            cases.append(dict(pattern=i,feasible=suite['feasible'],error_mv=suite['worst_hold_error_mv'],
                acquisition_error_at_worst_cycle_mv=cycle['acquisition_error_mv'],
                aperture_error_at_worst_cycle_mv=cycle['aperture_error_mv'],turnoff_change_at_worst_cycle_mv=cycle['turnoff_change_mv']))
        adc.append(dict(**desc,energy_pj=trace['best']['oracle']['mean_driver_energy_pj'],original_feasible=trace['best']['oracle']['feasible'],
            dense_feasible=conf['dense']['feasible'],independent_success=conf['independent_quality_all'],patterns=cases,
            objective=trace.get('surrogate_objective')))
    pool=read(root/'adc_driver/data/robust/pool.json');valid=[r['oracle'] for r in pool['records'] if r['oracle']['valid']]
    robust=[]
    for j,context in enumerate(pool['contexts']):
        values=[r['contexts'][j] for r in valid];best=min(values,key=lambda x:x['worst_hold_error_mv'])
        robust.append(dict(name=context['name'],valid_joint_designs=len(valid),feasible_designs=sum(x['feasible'] for x in values),
            minimum_hold_error_mv=best['worst_hold_error_mv'],max_acquisition_error_at_best_design_mv=best['max_acquisition_error_mv']))
    sys.path.insert(0,str(root/'d2d_coupled_calibration'))
    from coupled_experiment import packet_for,quality,rank
    from coupled_link import CoupledContext,phase_table
    base=root/'d2d_coupled_calibration/results/coupled_v1';cfg=read(base/'config.json');oracle=read(base/'independent_oracle.json')
    eligible=[x['context_id'] for x in oracle['records'] if x['feasible_by_ratio']['0.9']]
    calibration=np.load(base/'test_table.npz');grids={};feasibility=[]
    for context_id in eligible:
        masks=np.ones((4,4,2,65),dtype=bool);energies=np.zeros((4,4))
        c=CoupledContext.from_dict(cfg['contexts'][context_id])
        for sa in range(4):
            for sb in range(4):
                for pattern in cfg['confirmation_patterns']:
                    # Directly read the complete historical cache. Missing files fail;
                    # no cached_wave()/simulate() fallback or new measurement is allowed.
                    with np.load(base/f'waves/{pattern}/c{context_id:02d}_s{sa}{sb}.npz') as z:w={k:z[k] for k in z.files}
                    margins,errors=phase_table(w,c,packet_for(context_id,pattern),cfg['guard_ui'])
                    masks[sa,sb]&=(margins>=cfg['margin'])&(errors==0);energies[sa,sb]+=float(w['energy_pj_bit'])
        ratios=energies/energies[3,3];pairs=masks.any(-1).all(-1)&(ratios<=cfg['primary_ratio'])
        index=calibration['ids'].tolist().index(context_id)
        cal_mask=(calibration['margin'][index]>=cfg['margin'])&(calibration['errors'][index]==0)
        intersection=masks&cal_mask;intersection_pairs=intersection.any(-1).all(-1)&(ratios<=cfg['primary_ratio'])
        row=next(x for x in oracle['records'] if x['context_id']==context_id)
        assert abs(float(ratios[pairs].min())-row['minimum_energy_ratio'])<1e-12
        feasibility.append(dict(context_id=context_id,minimum_independent_energy_ratio=float(ratios[pairs].min()),
            independent_strength_pairs=np.argwhere(pairs).tolist(),
            calibration_and_independent_pairs=np.argwhere(intersection_pairs).tolist(),
            calibration_reference_feasible=bool(cal_mask[3,3,:,32].all()),
            feasible_phase_pairs=int(sum(masks[i,j,0].sum()*masks[i,j,1].sum() for i,j in np.argwhere(pairs)))))
        grids[context_id]=(masks,ratios,pairs)
    def joint(context_id,setting):
        masks,ratios,pairs=grids[context_id];i,j=setting['strength0'],setting['strength1']
        return bool(ratios[i,j]<=cfg['primary_ratio'] and masks[i,j,0,setting['phase0']] and masks[i,j,1,setting['phase1']])
    diagnostics=[];summary=[]
    for dataset in ('low','medium','high','mixed'):
        data=read(root/f'd2d_qualities/{dataset}/results/coupled_v1/test.json')
        trajectories={(t['context_id'],t['method'],t['seed']):t for t in data['trajectories']}
        for result in data['records']:
            context_id=result['context_id']
            if context_id not in eligible or result['budget']!=16:continue
            trajectory=trajectories[context_id,result['method'],result['seed']]
            initial={'setting':trajectory['initial_setting'],'measurement':trajectory['initial_measurement']}
            visits=[initial]+trajectory['trajectory']
            first_visited=None;first_incumbent=None;best=initial
            for step,v in enumerate(visits):
                if joint(context_id,v['setting']) and first_visited is None:first_visited=step
                if rank(v['measurement'],cfg['margin'])>rank(best['measurement'],cfg['margin']):best=v
                if joint(context_id,best['setting']) and first_incumbent is None:first_incumbent=step
                if step==16 or (result['method']=='fixed8' and step==0):assert best['setting']==result['setting']
            success=joint(context_id,result['setting'])
            assert success==(result['confirmation']['quality_all_patterns'] and result['confirmation']['energy_ratio']<=cfg['primary_ratio'])
            prefix=visits[:17];masks,ratios,pairs=grids[context_id]
            correct_strength=any(pairs[v['setting']['strength0'],v['setting']['strength1']] for v in prefix)
            if success:reason='success'
            elif any(joint(context_id,v['setting']) for v in prefix):reason='successful_setting_visited_but_not_returned'
            elif correct_strength:reason='right_strength_pair_but_no_successful_phase_pair_visited'
            else:reason='no_independent_feasible_strength_pair_visited'
            unique=len({tuple(v['setting'][k] for k in ('strength0','phase0','strength1','phase1')) for v in prefix})
            diagnostics.append(dict(quality=dataset,method=result['method'],seed=result['seed'],context_id=context_id,
                success=success,reason=reason,distinct_settings_through_16=unique,visits_through_16=len(prefix),
                first_visited_joint_success_posthoc=first_visited,first_incumbent_joint_success_posthoc=first_incumbent,
                returned_setting=result['setting'],returned_energy_ratio=result['confirmation']['energy_ratio'],
                returned_quality=result['confirmation']['quality_all_patterns']))
        for method in ('traditional','ga','bc','td3bc'):
            rows=[x for x in diagnostics if x['quality']==dataset and x['method']==method]
            allrows=[x for x in data['records'] if x['budget']==16 and x['method']==method]
            count=sum(x['success'] for x in rows)
            assert count==sum(x['confirmation']['quality_all_patterns'] and x['confirmation']['energy_ratio']<=cfg['primary_ratio'] for x in allrows)
            summary.append(dict(quality=dataset,method=method,successes=count,eligible_context_seed_pairs=len(rows),all_context_seed_pairs=len(allrows),
                conditional_success_percent=100*count/len(rows),overall_success_percent=100*count/len(allrows),
                reasons=dict(Counter(x['reason'] for x in rows))))
    result=dict(version=dict(recorded_repository=read(root/'source_repository.json'),source_checks=source_checks,execution_input_checks=input_checks,
        comparisons=versions,learned_search_objectives=sorted({x['objective'] for x in adc if x['objective']})),
        adc=adc,adc_robust_source_diagnosis=robust,d2d_feasible_contexts=feasibility,d2d_summary=summary,d2d_trajectories=diagnostics,
        limitations=['Post-hoc diagnosis on already used test data; not a new generalization evaluation.',
            'D2D feasibility applies to this discrete grid, these packets and constraints only.',
            'First-success indices are retrospective independent-cache evaluations, not independent online feedback or stopping times.',
            'ADC error components describe stored waveforms; they do not uniquely establish a physical cause.'])
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(objectives=result['version']['learned_search_objectives'],feasible_contexts=feasibility,d2d_summary=summary),indent=2))

if __name__=='__main__':main()
