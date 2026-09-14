"""Matched prototype IQL vs frozen TD3+BC on the same mixed terminal-success task."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(source,root,td3bc_execution=None,expectile=.7,temperature=3.,iql_execution=None):
    if not np.isfinite(expectile) or not 0<expectile<1 or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid IQL expectile/beta')
    if iql_execution:
        prior_root=iql_execution.parent.parent
        for name,digest in json.loads((prior_root/'input_hashes.json').read_text()).items():
            if sha(prior_root/name)!=digest:raise ValueError('Baseline input changed: '+name)
        for p in (prior_root/'d2d').glob('*.py'):
            if sha(p)!=sha(REPO/'d2d_coupled_calibration'/p.name):raise ValueError('IQL baseline training/evaluation source differs: '+p.name)
    root.mkdir(parents=True,exist_ok=False);project=root/'d2d';project.mkdir()
    for p in (REPO/'d2d_coupled_calibration').glob('*.py'):shutil.copy2(p,project/p.name)
    shutil.copytree(REPO/'d2d_coupled_calibration/assets',project/'assets')
    out=project/'results/coupled_v1';out.mkdir(parents=True)
    names=['config.json','dataset.npz','dataset.npz.metadata.json','dataset.json']
    names += [f'{split}_table.npz{ext}' for split in ('train','validation') for ext in ('','.metadata.json')]
    for name in names:shutil.copy2(source/name,out/name)
    shutil.copy2(__file__,root/'train_iql_compare.py')
    protocol=dict(quality='mixed',source=str(source),source_hashes={name:sha(source/name) for name in names},
        method='iql',reward_mode='terminal_success',seeds=[0,1,2],expectile=expectile,temperature=temperature,
        budget=16,eta=1.,discount=.95,steps=[5000,15000,50000],primary_step=50000,
        source_transitions=3072,selected_transitions=1536,episodes=96,
        fixed_data='Same first 16 transitions per episode as the TD3+BC reward ablation; no new acquisition.',
        clock='Keep original time feature encoding (32-step scale); the task stops at 16.',
        validation='Existing 12 held-out validation contexts, calibration packet only; not a fresh independent test.',
        primary_metrics=['returned_incumbent_success','logical_measurements'],
        non_primary=['No full-grid Q ranking as a success criterion','No IQL+AMO','No CQL','No data or margin auxiliary change'],
        td3bc_reference='references/td3bc.json' if td3bc_execution else None,
        iql_reference='references/iql.json' if iql_execution else None,
        policy_description='IQL, deterministic actor / advantage-weighted MSE',
        beta_definition='Multiplicative inverse temperature: min(exp(beta * (Q - V)), 100)',
        hyperparams=f'Predeclared pair tau={expectile}, beta={temperature}; compare a joint setting, not separate causal effects.',
        paper_reference='https://arxiv.org/html/2110.06169v1#A2',
        resource_cap=dict(workers=3,threads_per_worker=1,total_timeout_seconds=2400,new_spice_calls=0))
    for name,path in [('td3bc',td3bc_execution),('iql',iql_execution)]:
        if path:
            ref=json.loads(path.read_text());validate_reference(ref,protocol,iql=name=='iql')
            target=root/'references'/f'{name}.json';target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    write(root/'protocol.json',protocol)
    files=[p for p in root.rglob('*') if p.is_file()]
    write(root/'input_hashes.json',{str(p.relative_to(root)):sha(p) for p in files})


def validate_reference(ref,protocol,controls=None,iql=False):
    p=ref['protocol']
    for key in ('quality','source_hashes','seeds','budget','eta','discount','primary_step'):
        if p[key]!=protocol[key]:raise ValueError('Reference comparison contract mismatch: '+key)
    if iql and (p['method']!='iql' or p['reward_mode']!=protocol['reward_mode'] or p['steps']!=protocol['steps']):
        raise ValueError('Reference IQL task or checkpoint schedule mismatch')
    if controls is not None:
        audits=list(ref['reward_audits'].values())
        if not audits or any(a['nonreward_array_hashes']!=controls for a in audits):
            raise ValueError('Reference nonreward training arrays differ')


def _rollout_row(policy,table,context_id,budget,margin,ratio):
    from coupled_experiment import Environment,quality,CoupledSetting
    env=Environment(table,context_id)
    queried=[]
    for _ in range(budget):
        env.step(policy(env.obs()));queried.append(env.setting.codes())
    setting,m=env.best
    return dict(context_id=context_id,setting=list(setting.codes()),
        quality=quality(m,margin),energy_ratio=m['energy_ratio'],
        success=quality(m,margin) and m['energy_ratio']<=ratio,
        policy_queries=budget,reference_queries=1,
        logical_measurements=budget+1,unique_settings=len(set(queried)),
        incumbent_replaced=int(setting!=CoupledSetting()),
        returned_initial=setting==CoupledSetting())


def execute(root):
    started=time.perf_counter();project=root/'d2d';out=project/'results/coupled_v1'
    protocol=json.loads((root/'protocol.json').read_text())
    for name,digest in json.loads((root/'input_hashes.json').read_text()).items():
        if sha(root/name)!=digest:raise ValueError('Frozen IQL experiment input changed: '+name)
    sys.path.insert(0,str(project))
    from coupled_experiment import config,Table
    from coupled_train import Policy,label
    from reward_design import RewardSpec
    cfg=config()
    assert cfg['discount']==protocol['discount'] and cfg['checkpoint_steps']==protocol['steps']
    assert not set(cfg['validation_ids'])&set(cfg['train_ids']+cfg['test_ids'])
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    logs=root/'logs';logs.mkdir(exist_ok=True)
    if (root/'status.json').exists() or (out/'models').exists():raise FileExistsError('Existing training outputs; refusing unverified rerun')
    write(root/'status.json',dict(state='running',pid=os.getpid(),started_at=time.time()))
    spec=RewardSpec(protocol['reward_mode'],protocol['budget'],cfg['discount'],protocol['eta'],cfg['margin'],cfg['primary_ratio'])
    def one(seed):
        name=f'iql_s{seed}';log=logs/f'{name}.log'
        command=[sys.executable,'coupled_train.py','--method','iql','--seed',str(seed),
                 '--expectile',str(protocol['expectile']),'--temperature',str(protocol['temperature']),
                 '--reward-mode',protocol['reward_mode'],'--reward-budget',str(protocol['budget']),
                 '--reward-eta',str(protocol['eta'])]
        before=time.perf_counter();print('START',name,flush=True)
        receipt=logs/f'{name}.receipt.json'
        write(receipt,dict(state='running',command=command,protocol_sha256=sha(root/'protocol.json')))
        with log.open('x') as f:subprocess.run(command,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
        folder=out/'models'/(label('iql',seed,expectile=protocol['expectile'],temperature=protocol['temperature'])+spec.suffix())
        write(receipt,dict(state='completed',command=command,protocol_sha256=sha(root/'protocol.json'),
                           output_hashes={str(p.relative_to(root)):sha(p) for p in folder.iterdir() if p.is_file()}))
        print('COMPLETED',name,flush=True)
        return dict(seed=seed,command=command,seconds=time.perf_counter()-before)
    with ThreadPoolExecutor(protocol['resource_cap']['workers']) as pool:runs=list(pool.map(one,protocol['seeds']))
    table=Table('validation');rows=[];audits={}
    for seed in protocol['seeds']:
        folder=out/'models'/(label('iql',seed,expectile=protocol['expectile'],temperature=protocol['temperature'])+spec.suffix())
        audit=json.loads((folder/'reward_audit.json').read_text());audits[f's{seed}']=audit
        for step in protocol['steps']:
            policy=Policy(folder/f'step{step}.pt');assert policy.reward_spec==audit['spec']
            for context_id in table.ids:
                row=_rollout_row(policy,table,context_id,protocol['budget'],cfg['margin'],cfg['primary_ratio'])
                row.update(method='iql',seed=seed,step=step);rows.append(row)
            print('VALIDATED','iql',seed,step,flush=True)
    controls=[a['nonreward_array_hashes'] for a in audits.values()]
    assert all(h==controls[0] for h in controls)
    summary=[]
    for step in protocol['steps']:
        rr=[r for r in rows if r['step']==step]
        per_seed=[sum(r['success'] for r in rr if r['seed']==s) for s in protocol['seeds']]
        summary.append(dict(method='iql',step=step,successes_per_seed=per_seed,total=len(rr),
            success_percent=100*sum(per_seed)/len(rr),quality_percent=100*sum(r['quality'] for r in rr)/len(rr),
            mean_energy_ratio=float(np.mean([r['energy_ratio'] for r in rr])),
            passing_mean_energy_ratio=float(np.mean([r['energy_ratio'] for r in rr if r['success']])) if any(r['success'] for r in rr) else None,
            mean_logical_measurements=float(np.mean([r['logical_measurements'] for r in rr])),
            mean_unique_settings=float(np.mean([r['unique_settings'] for r in rr])),
            returned_initial=int(sum(r['returned_initial'] for r in rr))))
    td3bc=None
    if protocol.get('td3bc_reference'):
        ref=json.loads((root/protocol['td3bc_reference']).read_text());validate_reference(ref,protocol,controls[0])
        td3bc=[s for s in ref.get('summary',[]) if s.get('mode')=='terminal_success']
    baseline=None
    if protocol.get('iql_reference'):
        baseline=json.loads((root/protocol['iql_reference']).read_text());validate_reference(baseline,protocol,controls[0],iql=True)
    result=dict(iql_baseline_reference=baseline,protocol=protocol,runs=runs,reward_audits=audits,nonreward_arrays_identical=True,
        validation=rows,summary=summary,td3bc_terminal_reference=td3bc,
        seconds=time.perf_counter()-started,new_spice_calls=0,
        validation_logical_measurements=sum(r['logical_measurements'] for r in rows if r['step']==protocol['primary_step']))
    execution=root/'execution';write(execution/'summary.json',result)
    primary=next(s for s in summary if s['step']==protocol['primary_step'])
    lines=['# D2D: IQL 설정 비교 (종료 성공, mixed, 예산 16)','',
        '**IQL, deterministic actor / advantage-weighted MSE**','',
        '동일 mixed 데이터 1,536전이, 동일 terminal_success 보상, 동일 16회 예산에서 원형 IQL만 학습했다.',
        f"expectile τ={protocol['expectile']}, advantage 계수 β={protocol['temperature']}을 실행 전에 고정했다. β는 나누는 온도가 아니라 exp(β × (Q−V))의 곱하는 계수이며 가중치 상한은 100이다.",
        '원문 Appendix B는 locomotion에 (0.7,3), AntMaze에 (0.9,10)을 쓴다. 두 값을 함께 바꾸는 설정 비교이며 각 파라미터의 효과나 D2D 최적성을 입증하지 않는다. [IQL Appendix B](https://arxiv.org/html/2110.06169v1#A2)',
        '원문의 Gaussian 정책과 구분되는 deterministic actor를 유지했다. 주 평가 시점은 50k이며 5k/15k는 보조 기록이다.',
        '평가는 반환 incumbent 성공과 논리 측정 횟수다. 전수 격자 Q 순위는 성공 기준이 아니다.','',
        '| method | 고정 50k seed별 성공 / 12 | 성공률 % | 품질 % | 평균 에너지 비 | 조건당 논리 측정 |',
        '|---|---|---:|---:|---:|---:|',
        f"| IQL ({protocol['expectile']}, {protocol['temperature']}) | {primary['successes_per_seed']} | {primary['success_percent']:.3f} | {primary['quality_percent']:.3f} | {primary['mean_energy_ratio']:.4f} | {primary['mean_logical_measurements']:.1f} |"]
    if baseline:
        old=next(r for r in baseline['summary'] if r['step']==protocol['primary_step'])
        bp=baseline['protocol']
        lines.append(f"| IQL ({bp['expectile']}, {bp['temperature']}; 기존 결과) | {old['successes_per_seed']} | {old['success_percent']:.3f} | {old['quality_percent']:.3f} | {old['mean_energy_ratio']:.4f} | {old['mean_logical_measurements']:.1f} |")
    if td3bc:
        t=next(s for s in td3bc if s['step']==protocol['primary_step'])
        lines.append(f"| td3bc (동일 과제, 기존 실행) | {t['successes_per_seed']} | {t['success_percent']:.3f} | {t['quality_percent']:.3f} | {t['mean_energy_ratio']:.4f} | 17 |")
    lines+=['','IQL은 데이터 행동의 Q로 expectile V를 학습하고, 부트스트랩은 V(h\')만 사용한다. 정책은 advantage 가중 BC다.',
        '추가 SPICE는 0회다. 독립 test 패킷은 읽지 않았다.','']
    lines += ['', '36개 seed×조건 결과는 12개 validation 조건을 3개 seed로 반복한 것이다. 독립 조건 36개나 새 입력 일반화 성능으로 해석하지 않는다.', '기존 설정은 재학습하지 않고 동결 결과를 재사용했다. 새 학습은 3개 seed × 50k step이다.']
    (execution/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)
    write(root/'status.json',dict(state='completed',pid=os.getpid(),seconds=time.perf_counter()-started,new_spice_calls=0,summary=str(execution/'summary.json'),summary_sha256=sha(execution/'summary.json')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--td3bc-execution',type=Path,help='Existing reward-ablation execution/summary.json')
    p.add_argument('--expectile',type=float,default=.7);p.add_argument('--temperature',type=float,default=3.,help='Multiplicative beta in exp(beta * advantage)')
    p.add_argument('--iql-execution',type=Path,help='Frozen IQL baseline execution/summary.json')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        try:execute(a.output.resolve())
        except Exception as exc:
            if (a.output/'status.json').exists():write(a.output/'failure.json',dict(error=str(exc),time=time.time()))
            raise
    else:
        if a.source is None:p.error('--source is required for initialization')
        initialize(a.source.resolve(),a.output.resolve(),a.td3bc_execution.resolve() if a.td3bc_execution else None,a.expectile,a.temperature,a.iql_execution.resolve() if a.iql_execution else None)

if __name__=='__main__':main()
