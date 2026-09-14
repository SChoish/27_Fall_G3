"""Matched TD3+BC reward experiment on copied train/validation artifacts only."""
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
MODES=('legacy','terminal_success','potential_success')


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(source,root):
    root.mkdir(parents=True,exist_ok=False);project=root/'d2d';project.mkdir()
    for p in (REPO/'d2d_coupled_calibration').glob('*.py'):shutil.copy2(p,project/p.name)
    shutil.copytree(REPO/'d2d_coupled_calibration/assets',project/'assets')
    out=project/'results/coupled_v1';out.mkdir(parents=True)
    names=['config.json','dataset.npz','dataset.npz.metadata.json','dataset.json']
    names += [f'{split}_table.npz{ext}' for split in ('train','validation') for ext in ('','.metadata.json')]
    for name in names:shutil.copy2(source/name,out/name)
    shutil.copy2(__file__,root/'train_reward_ablation.py')
    protocol=dict(quality='mixed',source=str(source),source_hashes={name:sha(source/name) for name in names},
        reward_modes=list(MODES),seeds=[0,1,2],alpha=.1,budget=16,eta=1.,discount=.95,
        steps=[5000,15000,50000],primary_step=50000,
        source_transitions=3072,selected_transitions=1536,episodes=96,
        fixed_data='Same first 16 transitions per episode, unchanged observations/actions/context IDs across all modes; no new acquisition.',
        clock='Keep original time feature encoding (32-step scale) in training and rollout; the new task always stops at 16.',
        validation='Existing 12 held-out validation contexts, calibration packet only; not a fresh independent test. Primary checkpoint is fixed at 50k; no reward-specific tuning.',
        objective='Success of returned incumbent on training calibration packet at step 16; independent-packet success remains a generalization question.',
        non_claims=['No BC reward effect','No fresh test performance claim','No reward-specific hyperparameter search','Not directly comparable to original 32-step training runs'],
        resource_cap=dict(workers=3,threads_per_worker=1,total_timeout_seconds=2400,new_spice_calls=0))
    write(root/'protocol.json',protocol)
    files=[p for p in root.rglob('*') if p.is_file()]
    write(root/'input_hashes.json',{str(p.relative_to(root)):sha(p) for p in files})


def execute(root):
    started=time.perf_counter();project=root/'d2d';out=project/'results/coupled_v1'
    protocol=json.loads((root/'protocol.json').read_text())
    for name,digest in json.loads((root/'input_hashes.json').read_text()).items():
        if sha(root/name)!=digest:raise ValueError('Frozen reward experiment input changed: '+name)
    sys.path.insert(0,str(project))
    from coupled_experiment import config,Table,Environment,quality
    from coupled_train import Policy,label
    from reward_design import RewardSpec
    cfg=config()
    assert cfg['discount']==protocol['discount'] and cfg['checkpoint_steps']==protocol['steps']
    assert not set(cfg['validation_ids'])&set(cfg['train_ids']+cfg['test_ids'])
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    logs=root/'logs';logs.mkdir(exist_ok=True)
    def one(job):
        mode,seed=job;name=f'{mode}_s{seed}';log=logs/f'{name}.log'
        command=[sys.executable,'coupled_train.py','--method','td3bc','--alpha',str(protocol['alpha']),
                 '--seed',str(seed),'--reward-mode',mode,'--reward-budget',str(protocol['budget']),'--reward-eta',str(protocol['eta'])]
        before=time.perf_counter();print('START',name,flush=True)
        with log.open('w') as f:subprocess.run(command,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
        print('COMPLETED',name,flush=True)
        return dict(mode=mode,seed=seed,command=command,seconds=time.perf_counter()-before)
    with ThreadPoolExecutor(3) as pool:runs=list(pool.map(one,[(m,s) for m in MODES for s in protocol['seeds']]))
    audits={};rows=[]
    for mode in MODES:
        spec=RewardSpec(mode,protocol['budget'],cfg['discount'],protocol['eta'],cfg['margin'],cfg['primary_ratio'])
        for seed in protocol['seeds']:
            folder=out/'models'/(label('td3bc',seed,protocol['alpha'])+spec.suffix())
            audit=json.loads((folder/'reward_audit.json').read_text());audits[f'{mode}_s{seed}']=audit
            for step in protocol['steps']:
                policy=Policy(folder/f'step{step}.pt');assert policy.reward_spec==audit['spec']
                table=Table('validation')
                for context_id in table.ids:
                    e=Environment(table,context_id)
                    for _ in range(protocol['budget']):e.step(policy(e.obs()))
                    setting,m=e.best
                    rows.append(dict(mode=mode,seed=seed,step=step,context_id=context_id,setting=list(setting.codes()),
                        quality=quality(m,cfg['margin']),energy_ratio=m['energy_ratio'],
                        success=quality(m,cfg['margin']) and m['energy_ratio']<=cfg['primary_ratio']))
                print('VALIDATED',mode,seed,step,flush=True)
    controls=[a['nonreward_array_hashes'] for a in audits.values()]
    assert all(h==controls[0] for h in controls)
    summary=[]
    for mode in MODES:
        for step in protocol['steps']:
            rr=[r for r in rows if r['mode']==mode and r['step']==step]
            per_seed=[sum(r['success'] for r in rr if r['seed']==s) for s in protocol['seeds']]
            train_logs=[json.loads((out/'models'/(label('td3bc',s,protocol['alpha'])+RewardSpec(mode,protocol['budget'],cfg['discount'],protocol['eta'],cfg['margin'],cfg['primary_ratio']).suffix())/'training.json').read_text()) for s in protocol['seeds']]
            last=[next(x for x in t['logs'] if x['step']==step) for t in train_logs]
            summary.append(dict(mode=mode,step=step,successes_per_seed=per_seed,total=len(rr),
                success_percent=100*sum(per_seed)/len(rr),quality_percent=100*sum(r['quality'] for r in rr)/len(rr),
                mean_energy_ratio=float(np.mean([r['energy_ratio'] for r in rr])),
                q_abs_mean=float(np.mean([x['q_abs_mean'] for x in last])),q_weight_mean=float(np.mean([x['q_weight'] for x in last])),
                bc_loss_mean=float(np.mean([x['bc_loss'] for x in last]))))
    result=dict(protocol=protocol,runs=runs,reward_audits=audits,nonreward_arrays_identical=True,
        validation=rows,summary=summary,seconds=time.perf_counter()-started,new_spice_calls=0,
        validation_logical_measurements=len(rows)*(protocol['budget']+1))
    execution=root/'execution';write(execution/'summary.json',result)
    lines=['# D2D reward 비교: 기존 mixed 데이터와 개발용 validation','',
        '세 reward × 세 seed로 TD3+BC를 각각 50,000 step 학습했다. alpha=0.1, gamma=0.95, eta=1, budget=16을 공통 고정했다. reward 이외의 입력 배열 해시가 9개 실행에서 동일하다.',
        '', '원래 96 episode의 첫 16개 전이(1,536개)를 공통 사용했다. 기존 32-step 시간 feature도 모든 군과 rollout에서 동일하게 유지하며 16회에서 종료·bootstrap 차단한다. 원천 수집 비용은 그대로이고 추가 SPICE 실행은 0회다.',
        '', '이 결과는 기존 validation 12조건의 calibration 패킷 기준이다. 독립 test 패킷은 복사하거나 읽지 않았다. 기존 32-step 본학습과 직접 성능 비교하지 않는다.',
        '', '| reward | 고정 50k에서 seed별 성공 / 12 | 평균 성공률 % | 품질 충족률 % | 평균 에너지 비 |',
        '|---|---|---:|---:|---:|']
    for s in summary:
        if s['step']==protocol['primary_step']:lines.append(f"| {s['mode']} | {s['successes_per_seed']} | {s['success_percent']:.3f} | {s['quality_percent']:.3f} | {s['mean_energy_ratio']:.4f} |")
    lines+=['','## 학습 신호와 해석','',
        f"학습 endpoint 성공 episode는 {audits['legacy_s0']['terminal_success_episodes']}/96개다. terminal-only는 중간 reward가 0이며, potential은 마지막 측정의 실패 정도를 연속 신호로 전달한다.",
        'potential의 종료값은 0이다. 할인 episode return은 gamma^(B-1) × 성공 - eta × 초기 potential과 일치하는지 자동 검사했다. 이 목적 보존은 TD3+BC 함수 근사·최적화·Q 절댓값 정규화의 불변성 또는 성능 향상을 보장하지 않는다.',
        '', '| reward | 마지막 batch 평균 abs(Q) | 평균 Q 가중치 | 평균 BC MSE |',
        '|---|---:|---:|---:|']
    for s in summary:
        if s['step']==protocol['primary_step']:lines.append(f"| {s['mode']} | {s['q_abs_mean']:.6f} | {s['q_weight_mean']:.6f} | {s['bc_loss_mean']:.6f} |")
    lines+=['','5k/15k checkpoint 결과, 전체 validation 기록, 보상 통계와 배열 해시는 summary.json에 보존했다. 주 결과의 checkpoint는 사전에 50k로 고정했으며 validation에서 가장 높은 시점을 골라 바꾸지 않았다.',
        '', 'BC는 reward를 사용하지 않으므로 이 비교에서 재학습하지 않았다. ADC 목적함수나 기존 평가 데이터도 바꾸지 않았다. 다음 독립 성능 확인에는 새로 고정한 평가 입력이 필요하다.']
    (execution/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute(a.output.resolve())
    else:
        if a.source is None:p.error('--source is required for initialization')
        initialize(a.source.resolve(),a.output.resolve())

if __name__=='__main__':main()
