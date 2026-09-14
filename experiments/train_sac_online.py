"""Online SAC on train-table lookups; no mixed offline dataset or new reward."""
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
SOURCE_NAMES=['config.json']+[f'{s}_table.npz{e}' for s in ('train','validation') for e in ('','.metadata.json')]
PER_LANE=dict(coordinate_4axis_rank={'successes':0,'total':12,'note':'Existing 4-axis coordinate search at rho=0.9'},
              lock_4plus4_lane_step8={'successes':3,'total':12,'note':'Preview: lock 4+4 then per-lane phase step 8; grid cap'})


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_tables(source,ref):
    hashes=ref['protocol']['source_hashes']
    for name in ('config.json','train_table.npz','validation_table.npz'):
        if sha(source/name)!=hashes[name]:raise ValueError('Shared lookup table changed: '+name)


def initialize(source,root,iql_execution=None,iql_tau09_execution=None,td3bc_execution=None):
    root.mkdir(parents=True,exist_ok=False);project=root/'d2d';project.mkdir()
    for p in (REPO/'d2d_coupled_calibration').glob('*.py'):shutil.copy2(p,project/p.name)
    shutil.copytree(REPO/'d2d_coupled_calibration/assets',project/'assets')
    out=project/'results/coupled_v1';out.mkdir(parents=True)
    for name in SOURCE_NAMES:shutil.copy2(source/name,out/name)
    shutil.copy2(__file__,root/'train_sac_online.py')
    protocol=dict(quality='mixed_tables_only',source=str(source),source_hashes={name:sha(source/name) for name in SOURCE_NAMES},
        method='sac',reward_mode='terminal_success',seeds=[0,1,2],budget=16,eta=1.,discount=.95,
        steps=[5000,15000,50000],primary_step=50000,environment_steps=50000,updates_per_env_step=1,
        offline_dataset=False,no_margin_auxiliary=True,no_new_reward=True,
        clock='Keep original time feature encoding (32-step scale); the task stops at 16.',
        learning='Query new quantized actions on train contexts only; one SAC update per environment step.',
        validation='Existing 12 held-out validation contexts, calibration packet only; not a fresh independent test.',
        evaluation='Frozen weights; one reference measurement plus 16 policy queries.',
        primary_metrics=['returned_incumbent_success','logical_measurements','training_success_episodes'],
        cost_note='SAC 50k environment steps are not the same cost as IQL or TD3+BC 50k gradient updates on a fixed dataset. A gain is first evidence that extra interaction helped.',
        policy_description='SAC, squashed Gaussian actor; evaluation uses tanh(mean)',
        paper_reference='https://proceedings.mlr.press/v80/haarnoja18b.html',
        per_lane_reference=PER_LANE,
        iql_reference='references/iql.json' if iql_execution else None,
        iql_tau09_reference='references/iql_tau09.json' if iql_tau09_execution else None,
        td3bc_reference='references/td3bc.json' if td3bc_execution else None,
        resource_cap=dict(workers=3,threads_per_worker=1,per_job_timeout_seconds=3600,new_spice_calls=0))
    for name,path in [('iql',iql_execution),('iql_tau09',iql_tau09_execution),('td3bc',td3bc_execution)]:
        if path:
            ref=json.loads(path.read_text());shared_tables(source,ref)
            target=root/'references'/f'{name}.json';target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    write(root/'protocol.json',protocol)
    write(root/'input_hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})


def _rollout_row(policy,table,context_id,budget,margin,ratio):
    from coupled_experiment import Environment,quality,CoupledSetting
    env=Environment(table,context_id);queried=[]
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


def _summary(rows,seeds,method,step):
    per_seed=[sum(r['success'] for r in rows if r['seed']==s) for s in seeds]
    return dict(method=method,step=step,successes_per_seed=per_seed,total=len(rows),
        success_percent=100*sum(per_seed)/len(rows) if rows else 0.,
        quality_percent=100*sum(r['quality'] for r in rows)/len(rows) if rows else 0.,
        mean_energy_ratio=float(np.mean([r['energy_ratio'] for r in rows])) if rows else None,
        passing_mean_energy_ratio=float(np.mean([r['energy_ratio'] for r in rows if r['success']])) if any(r['success'] for r in rows) else None,
        mean_logical_measurements=float(np.mean([r['logical_measurements'] for r in rows])) if rows else None,
        mean_unique_settings=float(np.mean([r['unique_settings'] for r in rows])) if rows else None,
        returned_initial=int(sum(r['returned_initial'] for r in rows)))


def execute(root):
    started=time.perf_counter();project=root/'d2d';out=project/'results/coupled_v1'
    protocol=json.loads((root/'protocol.json').read_text())
    for name,digest in json.loads((root/'input_hashes.json').read_text()).items():
        if sha(root/name)!=digest:raise ValueError('Frozen SAC experiment input changed: '+name)
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
        name=f'sac_s{seed}';log=logs/f'{name}.log'
        command=[sys.executable,'coupled_train.py','--method','sac','--seed',str(seed),
                 '--reward-mode',protocol['reward_mode'],'--reward-budget',str(protocol['budget']),
                 '--reward-eta',str(protocol['eta']),'--checkpoint-steps',*[str(n) for n in protocol['steps']]]
        before=time.perf_counter();print('START',name,flush=True)
        receipt=logs/f'{name}.receipt.json'
        write(receipt,dict(state='running',command=command,protocol_sha256=sha(root/'protocol.json')))
        with log.open('x') as f:
            subprocess.run(command,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,
                           timeout=protocol['resource_cap']['per_job_timeout_seconds'])
        folder=out/'models'/(label('sac',seed)+spec.suffix())
        write(receipt,dict(state='completed',command=command,protocol_sha256=sha(root/'protocol.json'),
                           output_hashes={str(p.relative_to(root)):sha(p) for p in folder.iterdir() if p.is_file()}))
        print('COMPLETED',name,flush=True)
        return dict(seed=seed,command=command,seconds=time.perf_counter()-before)
    with ThreadPoolExecutor(protocol['resource_cap']['workers']) as pool:runs=list(pool.map(one,protocol['seeds']))
    val_table=Table('validation');train_table=Table('train');val_rows=[];train_rows=[];learning={}
    for seed in protocol['seeds']:
        folder=out/'models'/(label('sac',seed)+spec.suffix())
        learn=json.loads((folder/'training.json').read_text());learning[f's{seed}']=learn
        assert learn['environment_steps']==protocol['environment_steps']
        assert learn['updates_per_env_step']==1 and not learn['offline_dataset']
        assert learn['new_spice_calls']==0 and learn['reference_queries']==learn['training_episodes']
        assert learn['table_lookups']==learn['reference_queries']+learn['policy_queries']
        for step in protocol['steps']:
            policy=Policy(folder/f'step{step}.pt');assert policy.reward_spec['mode']=='terminal_success'
            for context_id in val_table.ids:
                row=_rollout_row(policy,val_table,context_id,protocol['budget'],cfg['margin'],cfg['primary_ratio'])
                row.update(method='sac',seed=seed,step=step,split='validation');val_rows.append(row)
            for context_id in train_table.ids:
                row=_rollout_row(policy,train_table,context_id,protocol['budget'],cfg['margin'],cfg['primary_ratio'])
                row.update(method='sac',seed=seed,step=step,split='train_frozen');train_rows.append(row)
            print('VALIDATED','sac',seed,step,flush=True)
    summary=[_summary([r for r in val_rows if r['step']==step],protocol['seeds'],'sac',step) for step in protocol['steps']]
    train_summary=[_summary([r for r in train_rows if r['step']==step],protocol['seeds'],'sac_train_frozen',step) for step in protocol['steps']]
    def load_ref(key):
        if not protocol.get(key):return None
        return json.loads((root/protocol[key]).read_text())
    iql=load_ref('iql_reference');iql09=load_ref('iql_tau09_reference');td3bc=load_ref('td3bc_reference')
    result=dict(protocol=protocol,runs=runs,learning=learning,validation=val_rows,train_frozen=train_rows,
        summary=summary,train_frozen_summary=train_summary,
        iql_terminal_reference=iql,iql_tau09_reference=iql09,td3bc_terminal_reference=td3bc,
        per_lane_reference=protocol['per_lane_reference'],
        seconds=time.perf_counter()-started,new_spice_calls=0,
        validation_logical_measurements=sum(r['logical_measurements'] for r in val_rows if r['step']==protocol['primary_step']),
        training_reference_queries=sum(learning[f's{s}']['reference_queries'] for s in protocol['seeds']),
        training_policy_queries=sum(learning[f's{s}']['policy_queries'] for s in protocol['seeds']))
    execution=root/'execution';write(execution/'summary.json',result)
    primary=next(s for s in summary if s['step']==protocol['primary_step'])
    train_primary=next(s for s in train_summary if s['step']==protocol['primary_step'])
    learn_success=[learning[f's{s}']['training_success_episodes'] for s in protocol['seeds']]
    lines=['# D2D: 온라인 SAC (종료 성공, 기존 mixed 데이터 없음, 예산 16)','',
        '**SAC, squashed Gaussian actor; 평가는 tanh(mean)**','',
        '기존 mixed offline 데이터 없이 train 조건 lookup 표에서만 새 행동을 질의하고 학습했다.',
        '보상은 기존 비교와 같은 terminal_success이며 마진 보조 학습이나 새 보상은 넣지 않았다.',
        '각 seed는 환경 step 50,000회이고 step당 업데이트 1회다. 기준 측정은 별도로 집계하며, 캐시에서 읽은 질의도 학습 측정으로 센다.',
        'SAC 50k 환경 step은 IQL·TD3+BC의 50k 업데이트와 같은 비용이 아니다. 잘되어도 우선 결론은 추가 상호작용을 허용했을 때의 변화다.',
        '[SAC 원논문](https://proceedings.mlr.press/v80/haarnoja18b.html)','']
    lines+=['| method | 고정 50k seed별 성공 / 12 | 성공률 % | 품질 % | 평균 에너지 비 | 조건당 논리 측정 |',
        '|---|---|---:|---:|---:|---:|',
        f"| SAC (온라인, mixed 없음) | {primary['successes_per_seed']} | {primary['success_percent']:.3f} | {primary['quality_percent']:.3f} | {primary['mean_energy_ratio']:.4f} | {primary['mean_logical_measurements']:.1f} |"]
    if iql:
        old=next(r for r in iql['summary'] if r['step']==protocol['primary_step'])
        lines.append(f"| IQL (0.7, 3; 기존 결과) | {old['successes_per_seed']} | {old['success_percent']:.3f} | {old['quality_percent']:.3f} | {old['mean_energy_ratio']:.4f} | {old['mean_logical_measurements']:.1f} |")
    if iql09:
        old=next(r for r in iql09['summary'] if r['step']==protocol['primary_step'])
        bp=iql09['protocol']
        lines.append(f"| IQL ({bp['expectile']}, {bp['temperature']}; 기존 결과) | {old['successes_per_seed']} | {old['success_percent']:.3f} | {old['quality_percent']:.3f} | {old['mean_energy_ratio']:.4f} | {old['mean_logical_measurements']:.1f} |")
    if td3bc:
        t=next(s for s in td3bc.get('summary',[]) if s.get('mode')=='terminal_success' and s['step']==protocol['primary_step'])
        lines.append(f"| td3bc (동일 과제, 기존 실행) | {t['successes_per_seed']} | {t['success_percent']:.3f} | {t['quality_percent']:.3f} | {t['mean_energy_ratio']:.4f} | 17 |")
    lane=protocol['per_lane_reference']
    lines+=[f"| 4축 좌표 탐색 (기존) | {lane['coordinate_4axis_rank']['successes']} / {lane['coordinate_4axis_rank']['total']} | — | — | — | 17 |",
        f"| 4+4 고정 + 레인별 step 8 (미리보기) | {lane['lock_4plus4_lane_step8']['successes']} / {lane['lock_4plus4_lane_step8']['total']} | — | — | — | 17 |",
        '','학습 중 train 조건에서 종료 성공한 episode 수는 seed별로 '
        f"{learn_success} / {learning['s0']['training_episodes']}다.",
        f"가중치를 고정한 뒤 train 조건을 다시 굴리면 50k에서 seed별 성공은 {train_primary['successes_per_seed']} / {len(train_table.ids)}다.",
        '학습 중에도 성공을 거의 못 찾으면 온라인 탐색으로도 희소한 성공 경험을 확보하지 못한 것이다.',
        '학습 조건에서는 성공하고 validation에서는 실패하면 조건 간 일반화를 먼저 본다.',
        '추가 SPICE는 0회다. 독립 test 패킷은 읽지 않았다.','',
        '36개 seed×조건 결과는 12개 validation 조건을 3개 seed로 반복한 것이다. 독립 조건 36개로 해석하지 않는다.']
    (execution/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)
    write(root/'status.json',dict(state='completed',pid=os.getpid(),seconds=time.perf_counter()-started,new_spice_calls=0,
        summary=str(execution/'summary.json'),summary_sha256=sha(execution/'summary.json')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iql-execution',type=Path);p.add_argument('--iql-tau09-execution',type=Path)
    p.add_argument('--td3bc-execution',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        try:execute(a.output.resolve())
        except Exception as exc:
            if (a.output/'status.json').exists():write(a.output/'failure.json',dict(error=str(exc),time=time.time()))
            raise
    else:
        if a.source is None:p.error('--source is required for initialization')
        initialize(a.source.resolve(),a.output.resolve(),
                   a.iql_execution.resolve() if a.iql_execution else None,
                   a.iql_tau09_execution.resolve() if a.iql_tau09_execution else None,
                   a.td3bc_execution.resolve() if a.td3bc_execution else None)

if __name__=='__main__':main()
