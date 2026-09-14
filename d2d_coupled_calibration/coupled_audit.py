"""Circuit controls, data contracts, independent measurement and endpoint audit."""
from dataclasses import replace
import argparse
import hashlib
import json
import numpy as np
from coupled_link import *
from coupled_experiment import OUT,config,Table,quality,utility,rank

def physical():
    cfg=config();p=OUT/'development';p.mkdir(parents=True,exist_ok=True)
    c=CoupledContext(link.Context(),link.Context(),.1);bits=payloads(501,64)
    # Recreate physical controls if the bundled fixtures were removed.
    qb=np.stack([np.zeros(64,np.uint8),np.tile([0,1],32).astype(np.uint8)])
    quiet=[]
    for cc in [0.,.06,.12]:
        folder=p/f'quiet_cc{cc:g}'
        if not (folder/'wave.npz').exists():simulate(replace(c,coupling_pf=cc),(8,8),qb,keep=folder)
        with np.load(folder/'wave.npz') as z:w={k:z[k] for k in z.files}
        ix=(w['t']>=24*c.lane0.ui)&(w['t']<=78*c.lane0.ui)
        quiet.append({'coupling_pf':cc,'victim_peak_to_peak_mv':float(1000*np.ptp(w['rx'][ix,0]))})
    (p/'quiet_measurements.json').write_text(json.dumps(quiet,indent=2)+'\n')
    if not (p/'decoupled_random/wave.npz').exists():simulate(replace(c,coupling_pf=0),(8,8),bits,keep=p/'decoupled_random')
    n=netlist(c,(1,8),bits);n2=netlist(c,(8,1),bits)
    names=lambda t:{x.split()[0] for x in t.splitlines() if x.startswith('M')}
    assert len(names(n))==72 and names(n)==names(n2)
    assert sum(x.startswith('CCOUP') for x in n.splitlines())==3
    assert 'CCOUP' not in netlist(replace(c,coupling_pf=0),(1,8),bits)
    peaks=[r['victim_peak_to_peak_mv'] for r in quiet]
    assert peaks[0]<.001 and peaks[1]>10 and peaks[2]>peaks[1]
    with np.load(p/'decoupled_random/wave.npz') as z:w={k:z[k] for k in z.files}
    if (p/'independent_single_lane.npz').exists():
        with np.load(p/'independent_single_lane.npz') as z:single={k:z[k] for k in z.files}
    else:
        single=link.simulate(c.lane0,8,bits[0],points_per_ui=32)
        np.savez_compressed(p/'independent_single_lane.npz',**single)
    ts=np.linspace(24*c.lane0.ui,78*c.lane0.ui,2000)
    err=float(1000*np.max(np.abs(np.interp(ts,w['t'],w['rx'][:,0])-np.interp(ts,single['t'],single['rx']))))
    assert err<1.0
    result={'fixed_mos_count':72,'coupling_capacitors':3,'quiet_victim':quiet,
            'decoupled_vs_single_lane_max_error_mv':err,'passed':True}
    (OUT/'physics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

def data_audit():
    cfg=config();d=np.load(OUT/'dataset.npz');n=len(d['rewards'])
    assert set(d['context_ids'])==set(cfg['train_ids'])
    assert not set(cfg['train_ids'])&set(cfg['validation_ids']) and not set(cfg['test_ids'])&set(cfg['train_ids']+cfg['validation_ids'])
    assert d['observations'].shape==(n,50) and d['actions'].shape==(n,4)
    assert not d['terminals'].any() and int(d['timeouts'].sum())==n//cfg['horizon']
    same=d['episode_ids'][1:]==d['episode_ids'][:-1]
    assert np.array_equal(d['next_observations'][:-1][same],d['observations'][1:][same])
    assert np.allclose(d['actions'],d['next_observations'][:,27:31],atol=1e-7)
    for axis,levels in [(0,[-1,-1/3,1/3,1]),(2,[-1,-1/3,1/3,1])]:
        assert np.min(np.abs(d['actions'][:,axis,None]-np.array(levels)[None,:]),axis=1).max()<1e-6
    def score_from_obs(obs):
        m=obs[:,40:42].astype(float);e=obs[:,42:44].astype(float);en=obs[:,44].astype(float)
        return np.where((m.min(1)>=cfg['margin'])&(e.sum(1)==0),2-en,-2-4*np.maximum(0,cfg['margin']-m).sum(1)-4*e.mean(1)-.1*en)
    reward=score_from_obs(d['next_observations'])-score_from_obs(d['observations'])-.01
    rerr=float(np.max(np.abs(reward-d['rewards'])));assert rerr<3e-6
    table=Table('train');rng=np.random.default_rng(9319);max_error=0.
    for i in [0,7,19]:
        c=CoupledContext.from_dict(cfg['contexts'][i]);bits=__import__('coupled_experiment').packet_for(i)
        for a,b in [(0,3),(2,2)]:
            with np.load(OUT/'waves/calibration'/f'c{i:02d}_s{a}{b}.npz') as z:w={k:z[k] for k in z.files}
            for _ in range(5):
                s=CoupledSetting(a,int(rng.integers(65)),b,int(rng.integers(65)));m=table.evaluate(i,s)
                for j,lane in enumerate([c.lane0,c.lane1]):
                    sw={'t':w['t'],'rx':w['rx'][:,j],'energy_pj_bit':float(w['energy_lane_pj_bit'][j])}
                    mm=link.measure(sw,lane,link.Setting(LEVELS[[a,b][j]],16,[s.phase0,s.phase1][j]),bits[j])
                    assert mm['bit_errors']==m['errors'][j]
                    max_error=max(max_error,abs(mm['guard_margin_vdd']-m['margins'][j]))
    assert max_error<1e-12
    result={'transitions_checked':n,'reward_reconstruction_max_error':rerr,'table_vs_independent_measurement_margin_error':max_error,'passed':True}
    (OUT/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

def evaluation():
    from coupled_evaluate import rollout
    from coupled_train import Policy,label
    cfg=config();p=json.loads((OUT/'test.json').read_text());rows=p['records'];selection=json.loads((OUT/'selection.json').read_text())
    assert p['selection_sha256']==hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    assert p['config_sha256']==hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest()
    traces={(t['context_id'],t['method'],t['seed']):t for t in p['trajectories']}
    for r in rows:
        t=traces[r['context_id'],r['method'],r['seed']]
        candidates=[{'setting':t['initial_setting'],'measurement':t['initial_measurement']}]+t['trajectory'][:r['budget']]
        best=max(candidates,key=lambda q:rank(q['measurement']))
        assert best['setting']==r['setting'] and best['measurement']==r['calibration']
        bundles=1 if r['method']=='fixed8' else r['budget']+1
        assert r['measurement_bundles']==bundles and r['phase_probes']==18*bundles
        assert r['confirmation']['quality_all_patterns']==all(quality(m) for m in r['confirmation']['patterns'])
        ms=r['confirmation']['patterns'];ratio=sum(m['energy_pj_bit'] for m in ms)/sum(m['reference_energy_pj_bit'] for m in ms)
        assert abs(ratio-r['confirmation']['energy_ratio'])<1e-12
    selected=selection['td3bc'];policy=Policy(OUT/'models'/label('td3bc',0,selected['alpha'])/f'step{selected["step"]}.pt')
    table=Table('test')
    for i in table.ids:
        _,trace=rollout(table,i,'td3bc',0,policy)
        old=traces[i,'td3bc',0]
        assert trace['trajectory']==old['trajectory']
    dense=[]
    for method in ['traditional','bo','td3bc']:
        eligible=[r for r in rows if r['method']==method and r['seed']==0 and r['budget']==cfg['primary_budget']]
        r=min(eligible,key=lambda x:abs(x['confirmation']['worst_margin_vdd']-cfg['margin']))
        worst=min(r['confirmation']['patterns'],key=lambda m:min(m['margins']))
        i=r['context_id'];s=CoupledSetting(**r['setting']);c=CoupledContext.from_dict(cfg['contexts'][i])
        bits=__import__('coupled_experiment').packet_for(i,worst['pattern'])
        w=simulate(c,s.strengths(),bits,keep=OUT/'dense_checks'/method,points_per_ui=64)
        m=measure(w,c,s,bits);margin_change=float(np.max(np.abs(np.array(m['margins'])-worst['margins'])))
        en_change=float(abs(m['energy_pj_bit']/worst['energy_pj_bit']-1))
        same=quality(m)==quality(worst)
        dense.append({'method':method,'context_id':i,'pattern':worst['pattern'],'margin_abs_change':margin_change,
                      'energy_relative_change':en_change,'quality_classification_unchanged':same})
        assert margin_change<.002 and en_change<.01 and same
    result={'records_checked':len(rows),'frozen_selection_unchanged':True,'td3bc_seed0_exact_replay':True,'dense_verification':dense,'passed':True}
    (OUT/'evaluation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['physical','data_audit','evaluation']);a=p.parse_args();globals()[a.command]()
