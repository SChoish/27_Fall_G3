"""Read-only run audit, explicitly privileged gate diagnosis, and plots."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from d2d_sizing.circuit import file_hash,write_json
from .physics import PhaseBank
from .environment import Config,Environment,drift_path,metrics,action_id,ACTIONS
from .run import verify,read


def oracle_diagnosis(root,cfg,p):
    """Minimum failed blocks on 17x17 reachable phase states; NOT a controller.

    Uses future drift and full waveform labels. Deliberately isolated from all
    policy/surrogate training and main comparisons. Zero monitor cost is a
    privileged lower bound, not an implementable energy result.
    """
    receipt=read(root/'physics/x0/development/receipt.json')
    bank=PhaseBank.load(root/'physics/x0/development/bank.npz',receipt['bank_sha256'])
    rows=[];start=time.time()
    for environment in range(2):
        for seed in p['development_drift_seeds']:
            d=drift_path(seed,cfg);good=[]
            for t in range(cfg.horizon):
                lane_good=np.zeros((2,17),bool);case=environment*2+t%2
                for i in range(17):
                    margin,error=bank.sample(case,np.full(2,i/16)-d[t])
                    lane_good[:,i]=(margin>=cfg.margin_min)&(error==0)
                good.append(lane_good[0,:,None]&lane_good[1,None,:]&(bank.overshoot[case]<=cfg.overshoot_max))
            dp=np.full((17,17),1e6);dp[8,8]=0;parents=[]
            for t in range(cfg.horizon):
                new=np.full((17,17),1e6);parent=np.zeros((17,17,2),int)
                for i in range(17):
                    for j in range(17):
                        options=[(dp[ii,jj]+.0001*((i!=ii)+(j!=jj)),ii,jj)
                                 for ii in range(max(0,i-1),min(17,i+2))
                                 for jj in range(max(0,j-1),min(17,j+2))]
                        val,ii,jj=min(options);new[i,j]=val+int(not good[t][i,j]);parent[i,j]=[ii,jj]
                dp=new;parents.append(parent)
            pos=np.array(np.unravel_index(dp.argmin(),dp.shape));path=[]
            for parent in parents[::-1]:path.append(pos.copy()*4);pos=parent[tuple(pos)]
            env=Environment(bank,d,environment,cfg)
            for phase in path[::-1]:env.step(action_id(phase-env.phase,False))
            m=metrics(env.records,cfg)
            assert sum(not r['quality'] for r in env.records)==int(np.floor(dp.min()+1e-6))
            rows.append(dict(environment=environment,drift_seed=seed,minimum_failure_blocks=round(m['violation_rate']*cfg.horizon),
                any_phase_impossible_blocks=sum(not g.any() for g in good),metrics=m,blocks=env.records))
    write_json(root/'analysis/oracle_diagnosis.json',dict(label='POST-HOC PRIVILEGED DIAGNOSTIC ONLY; cannot deploy or compare as a policy',
        algorithm='Dynamic programming, known future drift, 17x17 states, +/-4 code per block; tie break minimizes phase changes',
        phase_bank_lookups=bank.lookups,spice_transients=0,elapsed_seconds=time.time()-start,episodes=rows))
    return rows,bank.lookups


def audit(root,p,cfg,costs):
    verify(root,read(root/'input_hashes.json'));checks=[]
    devseeds=set(p['development_drift_seeds']);testseeds=set(p['test_drift_seeds'])
    assert not devseeds&testseeds;checks.append('development/test drift seeds disjoint')
    for path in sorted((root/'physics').glob('*/*/receipt.json')):
        r=read(path);PhaseBank.load(path.parent/'bank.npz',r['bank_sha256'])
        receipts=sorted((path.parent/'receipts').glob('*.json'))
        for item in receipts:
            receipt=read(item);folder=path.parent/'cache'/receipt['fingerprint']
            assert file_hash(folder/'measurement.json')==receipt['measurement_sha256']
        waves=[file_hash(path.parent/'cache'/read(item)['fingerprint']/'wave.npz') for item in receipts]
        assert waves==r['waveform_hashes']
        for netlist in (path.parent/'cache').glob('*/circuit.cir'):
            assert sum(line.lstrip().lower().startswith('m') for line in netlist.read_text().splitlines())==72
    checks.append('banks, measured data, raw waveform hashes and 72 MOS netlists verified')
    for entry in costs['evaluation']:
        path=root/entry['path'];assert file_hash(path)==entry['sha256'];data=read(path)
        assert entry['phase_bank_lookups']==entry['blocks']+3*entry['calibration_measurements']
        for episode in data['episodes']:
            recomputed=metrics(episode['blocks'],cfg)
            for key,value in recomputed.items():assert np.isclose(value,episode['metrics'][key])
            previous=np.array([32,32]);drift=drift_path(episode['drift_seed'],cfg)
            for t,block in enumerate(episode['blocks']):
                current=np.clip(previous+ACTIONS[block['requested_action'],:2],0,64)
                np.testing.assert_array_equal(current,block['phase_codes'])
                np.testing.assert_allclose(current/64-drift[t],block['effective_phase'])
                assert block['block']==t;previous=current
    checks.append('all evaluation metrics and persistent phase transitions recomputed; lookup accounting checked')
    if (root/'locked_selection.json').exists():
        receipt=read(root/'selection_receipt.json');assert file_hash(root/'locked_selection.json')==receipt['lock_sha256']
        for row in read(root/'locked_selection.json').values():
            r=read(root/'physics'/row['design']/'test/receipt.json');assert row['x']==r['x']
            if row['policy_seed'] is not None:assert file_hash(root/'policies'/f"v2_s{row['policy_seed']}"/'policy.pt')==row['policy_sha256']
        checks.append('selected design and evaluation bank/policy match locked return')
    else:checks.append('no returned design or final evaluation: feasibility gate stopped execution')
    write_json(root/'analysis/verification.json',dict(passed=True,checks=checks,postprocessing_source_sha256=file_hash(__file__)))


def generate(root):
    root=Path(root).resolve();p=read(root/'protocol.json');cfg=Config(**p['config']);costs=read(root/'costs.json');state=read(root/'status.json')
    (root/'analysis').mkdir(exist_ok=True);audit(root,p,cfg,costs)
    results=read(root/'final_results.json') if (root/'final_results.json').exists() else read(root/'classical_gate.json')
    prefix='final_' if (root/'final_results.json').exists() else 'A_nominal_'
    sensitivities=[]
    for mode in results:
        episodes=read(root/'rollouts'/f'{prefix}{mode}.json')['episodes']
        for energy_scale in p['sensitivity']['monitor_energy_scales']:
            for time_scale in p['sensitivity']['calibration_time_scales']:
                items=[]
                for ep in episodes:
                    blocks=ep['blocks'];data=sum(r['data_seconds'] for r in blocks)
                    pause=sum(r['data_seconds']*time_scale*r['measured']+(cfg.phase_update_ui/64*r['data_seconds'] if r['phase_changes'] else 0) for r in blocks)
                    energy=sum(r['total_energy_pj']+(energy_scale-1)*r['monitor_energy_pj'] for r in blocks)
                    items.append(dict(energy=energy,downtime=pause/(pause+data),feasible=ep['metrics']['violation_rate']<=cfg.violation_limit and pause/(pause+data)<=cfg.downtime_limit))
                sensitivities.append(dict(controller=mode,monitor_energy_scale=energy_scale,calibration_time_scale=time_scale,
                    passing_episodes=sum(i['feasible'] for i in items),mean_energy_pj=np.mean([i['energy'] for i in items]),worst_downtime=max(i['downtime'] for i in items)))
    write_json(root/'analysis/sensitivity.json',dict(scope='Fixed recorded actions and block-index drift, no policy tuning/reselection; timing sensitivity does not resimulate drift during pauses',rows=sensitivities))
    oracle=[];diagnostic_lookups=0
    if state['state']=='gate_failed':oracle,diagnostic_lookups=oracle_diagnosis(root,cfg,p)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=list(results) if prefix=='A_nominal_' else ['static_classical','joint_classical','static_rl_s0','joint_rl_s0']
    fig,axes=plt.subplots(4,len(modes),figsize=(4*len(modes),10),squeeze=False,sharex=True)
    for col,mode in enumerate(modes):
        ep=read(root/'rollouts'/f'{prefix}{mode}.json')['episodes'][0];b=ep['blocks'];t=np.arange(len(b))
        phase=np.array([r['phase_codes'] for r in b])/64;drift=np.array([r['drift_ui'] for r in b]);margin=np.array([r['margins'] for r in b])
        for lane in range(2):
            axes[0,col].step(t,phase[:,lane],where='post',label=f'phase L{lane}')
            axes[0,col].plot(t,drift[:,lane],':',label=f'drift L{lane} (private)')
            axes[1,col].plot(t,margin[:,lane],label=f'L{lane}')
        axes[0,col].set_title(mode);axes[0,col].legend(fontsize=6)
        axes[1,col].axhline(cfg.margin_min,color='red',linestyle='--');axes[1,col].set_ylim(-.55,.55)
        axes[2,col].stem(t,[r['measured'] for r in b],basefmt=' ')
        axes[3,col].plot(t,np.cumsum([r['tx_energy_pj'] for r in b]),label='TX')
        axes[3,col].plot(t,np.cumsum([r['total_energy_pj'] for r in b]),label='total');axes[3,col].legend(fontsize=7);axes[3,col].set_xlabel('Data block')
    for i,label in enumerate(['Phase / drift (UI)','Signed margin / VDD','Paid measurement','Cumulative energy (pJ)']):axes[i,0].set_ylabel(label)
    fig.suptitle('First predefined environment / drift seed; drift shown for diagnosis only');fig.tight_layout()
    fig.savefig(root/'analysis/tracking.png',dpi=150);fig.savefig(root/'analysis/tracking.pdf');plt.close(fig)
    sums=dict(spice_transients=sum(r['spice_transient_attempts'] for r in costs['physical']),cache_hits=sum(r['cache_hits'] for r in costs['physical']),
        design_characterizations=len(costs['physical']),evaluation_blocks=sum(r['blocks'] for r in costs['evaluation']),
        evaluation_calibration_measurements=sum(r['calibration_measurements'] for r in costs['evaluation']),
        evaluation_phase_bank_lookups=sum(r['phase_bank_lookups'] for r in costs['evaluation']),
        training_blocks=sum(r['blocks'] for r in costs['training']),training_gradient_steps=sum(r['gradient_steps'] for r in costs['training']),
        training_phase_bank_lookups=sum(r['phase_bank_lookups'] for r in costs['training']),
        training_calibration_measurements=sum(r['calibration_measurements'] for r in costs['training']),
        diagnostic_phase_bank_lookups=diagnostic_lookups,diagnostic_spice_transients=0,
        numerical_grid_validation_lookups=4*len(costs['physical']))
    write_json(root/'analysis/cost_summary.json',sums)
    lines=['# D2D dynamic co-design pilot results','',f"Execution status: **{state['state']}**. Source and assumptions were frozen before characterization.",'',
        'All figures below use the new dynamic environment; no earlier sizing/SAC/IQL result is counted.', '',
        '| Controller | Passing episodes | Mean / worst violation | Longest failure | TX / control energy (pJ/episode) | Measurements/episode | Mean / worst downtime | Throughput (Gb/s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['passing_episodes']}/{r['episodes']} | {100*r['mean_violation_rate']:.2f}% / {100*r['worst_violation_rate']:.2f}% | {r['longest_failure_run']} | {r['mean_tx_energy_pj']:.2f} / {r['mean_calibration_energy_pj']:.2f} | {r['mean_calibration_measurements']:.2f} | {100*r['mean_downtime_fraction']:.2f}% / {100*r['worst_downtime_fraction']:.2f}% | {r['mean_effective_throughput_bps']/1e9:.3f} |")
    if oracle:
        lines+=['','## Interpretation and gate diagnosis','',
            'No classical method passed all eight development episodes under both fixed constraints (quality violation ≤5%, downtime ≤25%). The gate therefore stopped PPO training, circuit search and final testing. No thresholds were relaxed and no extra learning was added.',
            f"The post-hoc full-information dynamic program found minimum failure-block counts {[r['minimum_failure_blocks'] for r in oracle]} across the same eight trajectories while respecting ±4-code movement and phase bounds. This uses future drift/full phase labels and is **not a deployable controller or a fair energy baseline**.",
            'Every-block paid sensing tracks quality, but its pilot overhead violates downtime. Periodic sensing respects downtime, but loses quality on some trajectories. This identifies a sensing/cadence tradeoff for the implemented baselines; it does not prove that all causal controllers are infeasible.',
            'The first next step is a separately preregistered development study of causal prediction from paid measurement history and a physically supported monitor implementation. The present experiment does not establish an RL or co-design advantage.',
            '', '## Executed versus unexecuted','',
            '- Executed: fresh nominal transistor characterization, four classical controllers, paired development trajectories, integrity audit, bounded privileged diagnosis and fixed-trajectory cost sensitivity.',
            '- Implemented but not physically executed: three-seed PPO pilot, twelve-design alternating search, all four joint comparisons, final held-out inputs/drift, RNC option. PPO and design integration are checked separately with synthetic data, which are software checks, not physical performance.',
            '- The locked test packets and drift seeds were not evaluated; there is no selected return design.']
    lines+=['','## Cost ledger','', '```json',json.dumps(sums,indent=2),'```','',
        'SPICE runs characterize fixed waveforms. A phase-bank lookup is a two-lane numerical query, not SPICE. Each data block uses one lookup and each paid monitor request uses three additional lookups. Raw interpolation validation and privileged diagnostics are recorded separately. Calibration energy/time describes modeled hardware operation, not machine training power.',
        '', '## Model and sensitivity limits','',
        'PTM predictive 45 nm bulk devices at fixed 90 nm gate length, lumped coupled RLC channels, 2 Gb/s/lane and behavioral RX. Voltage/temperature are fixed within each waveform; correlated timing displacement is quasi-static, not an actual PVT transient. Phase interpolation is bounded and has no wrap. The finite 64-bit margin/error check is not a demonstrated BER specification.',
        'The ideal noiseless early/center/late amplitude monitor uses one known pilot packet with 27 parallel probes per lane. Its readout/aggregation energy is an assumed monitor budget (2 fJ per sample), not characterized silicon power; exact analog margins are stronger information than a one-bit early/late detector. Pilot TX and phase updates are charged separately. A realistic monitor may cost more and reveal less.',
        'Sensitivity covers monitor energy ×0.1/1/10 and pilot downtime ×0.5/1/2 on fixed trajectories. TX, pilot TX and phase-update energy are unchanged. None of these post-hoc accounting scenarios replaces the primary criteria. Drift advances by data block, so the model omits extra drift during interruption.',
        '', '## Artifacts','',
        '- [Frozen protocol](protocol.json), [source hashes](input_hashes.json), [raw logs](execution.log), [costs](costs.json).',
        '- [Verification](analysis/verification.json), [sensitivity](analysis/sensitivity.json), [plot PNG](analysis/tracking.png), [plot PDF](analysis/tracking.pdf).',
        '- [Privileged diagnosis](analysis/oracle_diagnosis.json) is development-only and never used for policy learning or design selection.' if oracle else '- Final selected designs are in locked_selection.json.',
        '', 'See `d2d_joint/SPEC.md` and `REFERENCES.md` for observation order, cost assumptions, algorithm scope and accessible-source limitations.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'analysis/artifact_hashes.json',{str(f.relative_to(root)):file_hash(f) for f in sorted((root/'analysis').glob('*')) if f.is_file() and f.name!='artifact_hashes.json'})
    print(json.dumps(sums,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);generate(p.parse_args().output)
