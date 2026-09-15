"""Preregister, run feasibility gate, PPO, and bounded alternating co-design."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from scipy.stats import qmc
from d2d_sizing import circuit
from d2d_sizing.circuit import encode,decode,NOMINAL,Case,file_hash,write_json,cost
from .physics import suites,characterize
from .environment import Config,Environment,drift_path,metrics
from .controllers import Classical
from .ppo import train,Policy
from .design import DesignSearch,score


def read(p):return json.loads(Path(p).read_text())
def verify(root,items):
    for name,h in items.items():
        if file_hash(root/name)!=h:raise ValueError('Frozen input changed: '+name)


def validate_selected(row,bank,policy_path=None):
    if row['x']!=bank.x.tolist():
        raise ValueError('Evaluated design differs from selected design')
    if row['policy_seed'] is not None:
        if policy_path is None or file_hash(policy_path)!=row['policy_sha256']:
            raise ValueError('Policy version differs')


def initialize(output,binary,representation='regression'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    initial=np.clip(encode(NOMINAL)+(qmc.LatinHypercube(6,seed=6217).random(6)-.5)*.36,0,1);initial[0]=encode(NOMINAL)
    development,test=suites();binary=Path(binary).resolve()
    version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True).stdout.strip()
    files=list((repo/'d2d_joint').glob('*.py'))+[repo/'d2d_sizing'/n for n in ['__init__.py','circuit.py','surrogate.py']]
    files += [repo/'adc_driver'/n for n in ['attention.py','rnc.py']]
    files += [repo/'d2d_coupled_calibration'/n for n in ['link.py','coupled_link.py','assets/45nm_bulk.txt']]
    for f in files:
        dest=output/'code'/f.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    write_json(output/'protocol.json',dict(schema=1,topic='Low-power D2D circuit/control co-design under quasi-static correlated timing drift',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        worktree_before=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        config=asdict(Config()),binary=str(binary),binary_sha256=file_hash(binary),simulator_version=version,
        initial_designs={f'x{i}':decode(u).tolist() for i,u in enumerate(initial)},designs_initial=6,
        proposals_per_arm=2,designs_total=12,representation=representation,
        design_arms=['static','classical','rl'],design_optimizer='Sizing regression backbone and parent-reuse critic buffer; 200 initial + 100 steps per update; RNC optional',
        development=[asdict(c) for c in development],test=[asdict(c) for c in test],
        development_drift_seeds=[601001,601002,601003,601004],test_drift_seeds=list(range(901001,901009)),
        ppo_seeds=[0,1,2],ppo_steps_per_stage=7680,ppo_stages=2,ppo=dict(rollout=256,minibatch=128,epochs=4,lr=.0003,gamma=.99,gae_lambda=.95,clip=.2,entropy_coefficient=.01,value_coefficient=.5),
        stage_A='Nominal design: none/periodic/event/bang-bang, eight development episodes each. Proceed only if at least one satisfies the unchanged episode constraints.',
        stage_B='Three nominal-design PPO pilots, 7680 blocks each; four-frame history and argmax evaluation',
        stage_C='Freeze V1 for all proposal scores; two proposals per arm after common six designs. Train V2 once on all 12 designs, 10 episodes per design per seed; re-score ALL designs under each fixed V2 before final selection.',
        selection='Static arm uses fixed-phase SPICE metric; classical and RL arms use actual per-block operational metrics, constraints first then energy. Each arm may select only common six plus its own two proposals.',
        cost_assumptions='Parallel 3-center x 9-aperture monitor on a known 64-bit/lane pilot; pilot TX energy from bank; comparator 0.002 pJ/sample, phase change 0.1 pJ/lane + two UI pause. No measured monitor silicon power claim.',
        sensitivity=dict(monitor_energy_scales=[.1,1,10],calibration_time_scales=[.5,1,2],scope='Reaccount fixed selected trajectories/actions; no retraining or reselection'),
        gate_failure='Save classical diagnostics and do not start joint optimization; no relaxed thresholds or extra training',
        claims='Finite alternating pilot, not exact bilevel optimum or real PVT transient / full SerDes reproduction',created_at=time.time()))
    write_json(output/'input_hashes.json',{str(f.relative_to(output)):file_hash(f) for f in output.rglob('*') if f.is_file()})


def evaluate(bank,factory,seeds,cfg,output,tag):
    episodes=[];lookup_before=bank.lookups
    for env_id in range(2):
        for seed in seeds:
            controller=factory();env=Environment(bank,drift_path(seed,cfg),env_id,cfg)
            while env.t<cfg.horizon:env.step(controller(env.observation()))
            episodes.append(dict(environment=env_id,drift_seed=seed,metrics=metrics(env.records,cfg),blocks=env.records))
    m=[e['metrics'] for e in episodes]
    result=dict(worst_violation_rate=max(r['violation_rate'] for r in m),worst_downtime_fraction=max(r['downtime_fraction'] for r in m),
        mean_violation_rate=float(np.mean([r['violation_rate'] for r in m])),mean_downtime_fraction=float(np.mean([r['downtime_fraction'] for r in m])),
        mean_total_energy_pj=float(np.mean([r['total_energy_pj'] for r in m])),longest_failure_run=max(r['longest_failure_run'] for r in m),
        feasible=all(r['feasible'] for r in m),passing_episodes=sum(r['feasible'] for r in m),episodes=len(m),
        mean_calibration_measurements=float(np.mean([r['calibration_measurements'] for r in m])),
        mean_effective_throughput_bps=float(np.mean([r['effective_throughput_bps'] for r in m])),
        mean_tx_energy_pj=float(np.mean([r['tx_energy_pj'] for r in m])),
        mean_calibration_energy_pj=float(np.mean([r['pilot_tx_energy_pj']+r['monitor_energy_pj']+r['phase_update_energy_pj'] for r in m])),
        mean_interruption_seconds=float(np.mean([r['interruption_seconds'] for r in m])),phase_bank_lookups=bank.lookups-lookup_before)
    write_json(output,dict(tag=tag,result=result,episodes=episodes));return result


def average_results(results):
    # Constraints apply to every paired development rollout and every V1 seed.
    return dict(worst_violation_rate=max(r['worst_violation_rate'] for r in results),
        worst_downtime_fraction=max(r['worst_downtime_fraction'] for r in results),
        mean_total_energy_pj=float(np.mean([r['mean_total_energy_pj'] for r in results])),
        longest_failure_run=max(r['longest_failure_run'] for r in results))


def execute(output):
    output=Path(output).resolve();p=read(output/'protocol.json');verify(output,read(output/'input_hashes.json'))
    if Path(__file__).resolve()!=output/'code/d2d_joint/run.py':raise ValueError('Run the frozen code copy')
    if (output/'status.json').exists():raise FileExistsError('No silent resume or overwrite')
    if file_hash(p['binary'])!=p['binary_sha256']:raise ValueError('Simulator changed')
    os.environ['NGSPICE_BIN']=p['binary'];cfg=Config(**p['config']);start=time.time()
    state=dict(state='running',stage='A',pid=os.getpid(),started_at=start);write_json(output/'status.json',state)
    banks={};static={};designs=dict(p['initial_designs']);costs=[];eval_ledger=[]
    cases=[Case.from_dict(c) for c in p['development']]
    def add_design(name,x):
        bank,result,account=characterize(x,cases,output/'physics'/name/'development',workers=2)
        if account['simulator_version']!=p['simulator_version']:raise ValueError('Wrong simulator')
        banks[name]=bank;static[name]=result;designs[name]=np.asarray(x).tolist();costs.append(dict(design=name,split='development',**account))
        print('CHARACTERIZED',name,'static pass',result['feasible'],flush=True)
    def evaluate_one(name,factory,tag,phase='development',bank=None):
        selected=banks[name] if bank is None else bank
        seeds=p['development_drift_seeds'] if phase=='development' else p['test_drift_seeds']
        path=output/'rollouts'/f'{tag}.json'
        r=evaluate(selected,factory,seeds,cfg,path,tag)
        eval_ledger.append(dict(tag=tag,design=name,split=phase,blocks=r['episodes']*cfg.horizon,calibration_measurements=r['mean_calibration_measurements']*r['episodes'],
            phase_bank_lookups=r['phase_bank_lookups'],path=str(path.relative_to(output)),sha256=file_hash(path)))
        return r
    try:
        add_design('x0',designs['x0']);classical={}
        for mode in ['none','periodic','event','bang_bang']:
            classical[mode]=evaluate_one('x0',lambda m=mode:Classical(m,cfg.horizon),'A_nominal_'+mode)
        write_json(output/'classical_gate.json',classical)
        feasible=[m for m in classical if classical[m]['feasible']]
        if not feasible:
            state.update(state='gate_failed',reason='No classical controller meets fixed nominal development constraints; joint optimization not run')
            write_json(output/'costs.json',dict(physical=costs,evaluation=eval_ledger,training=[]));write_json(output/'status.json',state)
            print('GATE FAILED',json.dumps(classical),flush=True);return
        controller=min(feasible,key=lambda m:score(classical[m]));write_json(output/'selected_classical.json',dict(mode=controller,selected_on='nominal development only'))
        print('GATE PASSED',controller,flush=True);state['stage']='B';write_json(output/'status.json',state)
        v1={};v1_hash={};nominal_ppo={}
        for seed in p['ppo_seeds']:
            v1[seed]=train({'x0':banks['x0']},seed,1,output/'policies'/f'v1_s{seed}',cfg,p['ppo_steps_per_stage'])
            v1_hash[str(seed)]=file_hash(output/'policies'/f'v1_s{seed}'/'policy.pt')
            nominal_ppo[str(seed)]=evaluate_one('x0',lambda s=seed:Policy(v1[s]),f'B_nominal_ppo_s{seed}')
            print('PPO V1 COMPLETE',seed,nominal_ppo[str(seed)]['passing_episodes'],'/8',flush=True)
        write_json(output/'nominal_ppo.json',nominal_ppo);write_json(output/'v1_lock.json',v1_hash)
        state['stage']='C';write_json(output/'status.json',state)
        for name,x in p['initial_designs'].items():
            if name!='x0':add_design(name,x)
        classical_scores={};rl_scores={}
        def operational(name):
            classical_scores[name]=evaluate_one(name,lambda:Classical(controller,cfg.horizon),'C_classical_'+name)
            rl_scores[name]=[evaluate_one(name,lambda s=seed:Policy(v1[s]),f'C_v1_s{seed}_{name}') for seed in p['ppo_seeds']]
        for name in p['initial_designs']:operational(name)
        pools={arm:list(p['initial_designs']) for arm in p['design_arms']};events={}
        for arm in p['design_arms']:
            def response(name):return static[name] if arm=='static' else classical_scores[name] if arm=='classical' else average_results(rl_scores[name])
            rows=[dict(u=encode(designs[name]).tolist(),result=response(name)) for name in pools[arm]]
            optimizer=DesignSearch(rows,method=p['representation'],seed=491,mode='static' if arm=='static' else 'operational')
            for query in range(p['proposals_per_arm']):
                u,pred=optimizer.propose([encode(x) for x in designs.values()]);name=f'{arm}_{query}'
                write_json(output/f'pending_{arm}.json',dict(u=u.tolist(),predictions=pred,policy_version_hashes=v1_hash))
                add_design(name,decode(u));operational(name)
                optimizer.observe(dict(u=u.tolist(),result=response(name)),pred);pools[arm].append(name)
                (output/f'pending_{arm}.json').unlink()
            events[arm]=optimizer.events
            torch.save(dict(state_dict=optimizer.current.model.state_dict(),mean=optimizer.mean,std=optimizer.std,
                mode=optimizer.mode,representation=optimizer.method,buffer=[dict(id=e.id,state_dict=e.model.state_dict(),errors=e.errors) for e in optimizer.entries]),output/f'design_surrogate_{arm}.pt')
        write_json(output/'design_search.json',dict(designs=designs,pools=pools,events=events,static=static,classical_scores=classical_scores,v1_rl_scores=rl_scores))
        for seed in p['ppo_seeds']:
            if file_hash(output/'policies'/f'v1_s{seed}'/'policy.pt')!=v1_hash[str(seed)]:raise ValueError('V1 policy changed')
        v2={};v2_scores={};v2_hash={}
        for seed in p['ppo_seeds']:
            v2[seed]=train(banks,seed,2,output/'policies'/f'v2_s{seed}',cfg,p['ppo_steps_per_stage'],v1[seed])
            v2_hash[str(seed)]=file_hash(output/'policies'/f'v2_s{seed}'/'policy.pt')
            v2_scores[str(seed)]={name:evaluate_one(name,lambda s=seed:Policy(v2[s]),f'C_v2_s{seed}_{name}') for name in banks}
            print('PPO V2 COMPLETE',seed,flush=True)
        write_json(output/'v2_scores.json',v2_scores)
        x_static=min(pools['static'],key=lambda n:cost(static[n]));x_classic=min(pools['classical'],key=lambda n:score(classical_scores[n]))
        selection={'static_classical':dict(design=x_static,controller=controller,policy_seed=None),
                   'joint_classical':dict(design=x_classic,controller=controller,policy_seed=None)}
        for seed in p['ppo_seeds']:
            x_rl=min(pools['rl'],key=lambda n:score(v2_scores[str(seed)][n]))
            selection[f'static_rl_s{seed}']=dict(design=x_static,controller='ppo',policy_seed=seed,policy_sha256=v2_hash[str(seed)])
            selection[f'joint_rl_s{seed}']=dict(design=x_rl,controller='ppo',policy_seed=seed,policy_sha256=v2_hash[str(seed)])
        for row in selection.values():row['x']=designs[row['design']]
        write_json(output/'locked_selection.json',selection);lock_hash=file_hash(output/'locked_selection.json')
        write_json(output/'selection_receipt.json',dict(lock_sha256=lock_hash,policy_hashes=v2_hash,
            v2_scores_sha256=file_hash(output/'v2_scores.json'),design_search_sha256=file_hash(output/'design_search.json'),locked_at=time.time()))
        state['stage']='final';write_json(output/'status.json',state)
        final_banks={};final_results={}
        for name in sorted({r['design'] for r in selection.values()}):
            bank,_,account=characterize(designs[name],[Case.from_dict(c) for c in p['test']],output/'physics'/name/'test',workers=2)
            final_banks[name]=bank;costs.append(dict(design=name,split='test',**account))
        for key,row in selection.items():
            if file_hash(output/'locked_selection.json')!=lock_hash:raise ValueError('Locked return changed')
            seed=row['policy_seed']
            validate_selected(row,final_banks[row['design']],output/'policies'/f'v2_s{seed}'/'policy.pt' if seed is not None else None)
            factory=(lambda:Classical(controller,cfg.horizon)) if seed is None else (lambda s=seed:Policy(v2[s]))
            final_results[key]=evaluate_one(row['design'],factory,'final_'+key,'test',final_banks[row['design']])
        write_json(output/'final_results.json',final_results)
        training=[dict(path=str(f.relative_to(output)),sha256=file_hash(f),**read(f)) for f in sorted((output/'policies').glob('*/training.json'))]
        write_json(output/'costs.json',dict(physical=costs,evaluation=eval_ledger,training=training))
        verify(output,read(output/'input_hashes.json'))
        write_json(output/'receipt.json',dict(protocol_sha256=file_hash(output/'protocol.json'),lock_sha256=lock_hash,
            final_results_sha256=file_hash(output/'final_results.json'),costs_sha256=file_hash(output/'costs.json'),
            elapsed_seconds=time.time()-start,policy_hashes=v2_hash))
        state.update(state='completed',finished_at=time.time(),receipt_sha256=file_hash(output/'receipt.json'));write_json(output/'status.json',state)
        print('COMPLETED',json.dumps(final_results),flush=True)
    except Exception as exc:
        state.update(state='failed',error=str(exc),finished_at=time.time());write_json(output/'status.json',state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['initialize','execute']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path);p.add_argument('--representation',choices=['regression','rnc'],default='regression');a=p.parse_args()
    if a.command=='initialize':
        if not a.binary:p.error('--binary required')
        initialize(a.output,a.binary,a.representation)
    else:execute(a.output)
