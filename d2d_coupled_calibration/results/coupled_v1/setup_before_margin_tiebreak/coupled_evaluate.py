"""Select on validation, freeze settings, then confirm on independent patterns."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import hashlib
import json
import time
import numpy as np
from coupled_experiment import OUT,config,Table,Environment,Search,quality,utility,cached_wave,phase_table,CoupledSetting
from coupled_train import Policy,label

METHODS=['fixed8','traditional','bo','ga','random','bc','td3bc']

def rollout(table,i,method,seed=0,policy=None,params=None):
    cfg=config();env=Environment(table,i);search=Search(method,seed=seed*100000+1009*i+4991,params=params)
    search.observe(env.setting,env.measurement);records=[];trajectory=[];compute=0.
    if method=='fixed8':
        records=[{'context_id':i,'method':method,'seed':seed,'budget':b,'setting':asdict(env.setting),
            'calibration':env.measurement,'calibration_quality':quality(env.measurement,cfg['margin']),
            'measurement_bundles':1,'phase_probes':18,'bit_decisions':18*cfg['calibration_bits'],
            'controller_compute_seconds':0.,'confirmation':None} for b in cfg['budgets']]
        return records,{'context_id':i,'method':method,'seed':seed,'initial_setting':asdict(env.setting),
                        'initial_measurement':env.measurement,'trajectory':[]}
    for step in range(1,cfg['horizon']+1):
        start=time.perf_counter()
        action=policy(env.obs()) if policy else search.suggest().action()
        compute+=time.perf_counter()-start
        _,reward,term,timeout,m=env.step(action);search.observe(env.setting,m)
        trajectory.append({'step':step,'setting':asdict(env.setting),'measurement':m,'reward':reward})
        if step in cfg['budgets']:
            s,best=env.best
            records.append({'context_id':i,'method':method,'seed':seed,'budget':step,
                'setting':asdict(s),'calibration':best,'calibration_quality':quality(best,cfg['margin']),
                'measurement_bundles':step+1,'phase_probes':18*(step+1),
                'bit_decisions':18*(step+1)*cfg['calibration_bits'],'controller_compute_seconds':compute,
                'confirmation':None})
    return records,{'context_id':i,'method':method,'seed':seed,'initial_setting':asdict(CoupledSetting()),
                    'initial_measurement':env.history[0][1],'trajectory':trajectory}

def score(records):
    cfg=config()
    rates=[r['calibration_quality'] and r['calibration']['energy_ratio']<=rho for r in records for rho in cfg['ratios']]
    return [float(np.mean(rates)),float(np.mean([utility(r['calibration']) for r in records if r['budget']==cfg['horizon']]))]

def validate():
    cfg=config();table=Table('validation');trials=[]
    for method,alphas in [('bc',[0.]),('td3bc',cfg['alphas'])]:
        for alpha in alphas:
            for step in cfg['checkpoint_steps']:
                rows=[]
                for seed in cfg['train_seeds']:
                    policy=Policy(OUT/'models'/label(method,seed,alpha)/f'step{step}.pt')
                    for i in table.ids:rows.extend(rollout(table,i,method,seed,policy)[0])
                trial={'method':method,'alpha':alpha,'step':step,'score':score(rows),'records':rows};trials.append(trial)
                print(method,alpha,step,trial['score'],flush=True)
    selection={}
    for m in ['bc','td3bc']:
        t=max((r for r in trials if r['method']==m),key=lambda r:(*r['score'],-r['step'],-r['alpha']))
        selection[m]={k:t[k] for k in ['method','alpha','step','score']}
    for method,grid in cfg['baseline_sweeps'].items():
        for params in grid:
            rows=[]
            for seed in ([0] if method=='traditional' else cfg['train_seeds']):
                for i in table.ids:rows.extend(rollout(table,i,method,seed,params=params)[0])
            trial={'method':method,'params':params,'score':score(rows),'records':rows};trials.append(trial)
            print(method,params,trial['score'],flush=True)
        t=max((r for r in trials if r['method']==method),key=lambda r:tuple(r['score']))
        selection[method]={k:t[k] for k in ['method','params','score']}
    (OUT/'validation.json').write_text(json.dumps(trials,indent=2)+'\n')
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print(selection,flush=True)

def calibration():
    cfg=config();table=Table('test');selection=json.loads((OUT/'selection.json').read_text());records=[];traces=[]
    for method in METHODS:
        seeds=[0] if method in ['fixed8','traditional'] else cfg['train_seeds']
        for seed in seeds:
            policy=None
            if method in ['bc','td3bc']:
                p=selection[method];policy=Policy(OUT/'models'/label(method,seed,p['alpha'])/f'step{p["step"]}.pt')
            for i in table.ids:
                rr,tt=rollout(table,i,method,seed,policy,selection.get(method,{}).get('params'));records.extend(rr);traces.append(tt)
            print('Test calibration',method,seed,score([r for r in records if r['method']==method and r['seed']==seed]),flush=True)
    out={'records':records,'trajectories':traces,'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
         'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
         'note':'All endpoint settings frozen before any independent confirmation results are computed.'}
    (OUT/'test_calibration.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Frozen endpoints',len(records),flush=True)

def confirm(workers=6):
    cfg=config();source=OUT/'test_calibration.json';p=json.loads(source.read_text());rows=p['records']
    # All final settings are verified, including those that failed calibration.
    triples={(r['context_id'],r['setting']['strength0'],r['setting']['strength1']) for r in rows}
    triples|={(i,3,3) for i in cfg['test_ids']}
    tasks=[(*x,pattern) for x in sorted(triples) for pattern in cfg['confirmation_patterns']]
    measured={};start=time.perf_counter()
    def one(i,a,b,pattern):
        c,bits,w=cached_wave(i,a,b,pattern);m,e=phase_table(w,c,bits,cfg['guard_ui'])
        return (i,a,b,pattern),m,e,w['energy_pj_bit']
    with ThreadPoolExecutor(workers) as pool:
        fs=[pool.submit(one,*task) for task in tasks]
        for k,f in enumerate(as_completed(fs)):
            key,m,e,en=f.result();measured[key]=(m,e,en)
            if (k+1)%24==0:print('Independent verification',k+1,'/',len(fs),round(time.perf_counter()-start,1),'seconds',flush=True)
    for r in rows:
        i=r['context_id'];s=CoupledSetting(**r['setting']);details=[]
        for pattern in cfg['confirmation_patterns']:
            m,e,en=measured[i,s.strength0,s.strength1,pattern];ref=measured[i,3,3,pattern][2]
            phases=[s.phase0,s.phase1]
            details.append({'pattern':pattern,'margins':[float(m[j,phases[j]]) for j in range(2)],
                'errors':[int(e[j,phases[j]]) for j in range(2)],'energy_pj_bit':float(en),
                'reference_energy_pj_bit':float(ref),'energy_ratio':float(en/ref),'bits_per_lane':cfg['confirmation_bits']})
        valid=all(quality(m,cfg['margin']) for m in details)
        ratio=sum(m['energy_pj_bit'] for m in details)/sum(m['reference_energy_pj_bit'] for m in details)
        r['confirmation']={'patterns':details,'quality_all_patterns':valid,'energy_ratio':float(ratio),
            'worst_margin_vdd':float(min(min(m['margins']) for m in details)),
            'errors_total':int(sum(sum(m['errors']) for m in details))}
    p['calibration_file_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    p['confirmation_unique_transients']=len(tasks)
    p['note']='Prespecified paired-energy metric. All frozen endpoints independently checked; no test retuning or fallback.'
    (OUT/'test.json').write_text(json.dumps(p,indent=2)+'\n')
    arrays={}
    for (i,a,b,pattern),(m,e,en) in measured.items():
        key=f'c{i}_s{a}{b}_{pattern}';arrays[key+'_margin']=m;arrays[key+'_errors']=e;arrays[key+'_energy']=en
    np.savez_compressed(OUT/'confirmation_tables.npz',**arrays)
    print('Independent confirmation completed',len(rows),flush=True)

def oracle():
    cfg=config();table=Table('test');rows=[]
    for i in table.ids:
        j=table.ids.index(i);valid=(table.margin[j]>=cfg['margin'])&(table.errors[j]==0)
        # RX phases do not affect physical waveforms and may be chosen independently.
        pair_ok=valid.any(axis=-1).all(axis=-1)
        energy=table.energy[j]/table.energy[j,3,3]
        e=float(energy[pair_ok].min()) if pair_ok.any() else None
        rows.append({'context_id':i,'calibration_feasible':bool(pair_ok.any()),'minimum_energy_ratio':e,
                     'feasible_by_ratio':{str(r):bool(e is not None and e<=r) for r in cfg['ratios']}})
    (OUT/'oracle_feasibility.json').write_text(json.dumps({'scope':'Calibration packet, full 4x4x65x65 action grid; no independent-pattern oracle claim. Never supplied to policies.',
                                                         'records':rows},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','calibration','confirm','oracle']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.command=='confirm':confirm(a.workers)
    else:globals()[a.command]()
