"""Report raw held-out outcomes without hiding failed or infeasible cases."""
import csv
import hashlib
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy,torch
from coupled_experiment import ROOT,OUT,config,quality
from coupled_evaluate import METHODS

NAMES={'fixed8':'Fixed 8+8','traditional':'Coordinate search','bo':'Bayesian optimization',
       'ga':'Genetic algorithm','random':'Random search','bc':'BC','td3bc':'TD3+BC'}

def aggregate(rows,rho,confirmation=True):
    seeds=[]
    for seed in sorted({r['seed'] for r in rows}):
        rs=[r for r in rows if r['seed']==seed]
        good=[r['confirmation']['quality_all_patterns'] if confirmation else r['calibration_quality'] for r in rs]
        energy=[r['confirmation']['energy_ratio'] if confirmation else r['calibration']['energy_ratio'] for r in rs]
        seeds.append({'seed':seed,'joint_success_percent':float(100*np.mean(np.array(good)&(np.array(energy)<=rho))),
                      'quality_success_percent':float(100*np.mean(good)),
                      'controller_compute_seconds_mean':float(np.mean([r['controller_compute_seconds'] for r in rs]))})
    vals=np.array([s['joint_success_percent'] for s in seeds])
    return {'mean':float(vals.mean()),'std':float(vals.std(ddof=1)) if len(vals)>1 else None,
            'quality_mean':float(np.mean([s['quality_success_percent'] for s in seeds])),
            'compute_seconds_mean':float(np.mean([s['controller_compute_seconds_mean'] for s in seeds])),
            'seeds':seeds}

def formatted(s):return f'{s["mean"]:.2f}%' if s['std'] is None else f'{s["mean"]:.2f} ± {s["std"]:.2f}%'

def run():
    cfg=config();data=json.loads((OUT/'test.json').read_text());rows=data['records']
    selection=json.loads((OUT/'selection.json').read_text());dataset=json.loads((OUT/'dataset.json').read_text())
    physics=json.loads((OUT/'physics_audit.json').read_text());oracle=json.loads((OUT/'oracle_feasibility.json').read_text())
    robust_oracle=json.loads((OUT/'oracle_confirmation.json').read_text())
    assert robust_oracle['source_test_sha256']==hashlib.sha256((OUT/'test.json').read_bytes()).hexdigest()
    audits={name:json.loads((OUT/f'{name}_audit.json').read_text()) for name in ['physics','data','evaluation']}
    assert all(a['passed'] for a in audits.values())
    results={}
    for b in cfg['budgets']:
        results[str(b)]={}
        for rho in cfg['ratios']:
            results[str(b)][str(rho)]={m:{'calibration':aggregate([r for r in rows if r['method']==m and r['budget']==b],rho,False),
                    'confirmation':aggregate([r for r in rows if r['method']==m and r['budget']==b],rho)} for m in METHODS}
    feasibility={str(rho):sum(r['feasible_by_ratio'][str(rho)] for r in oracle['records']) for rho in cfg['ratios']}
    robust_feasibility={str(rho):sum(r['feasible_by_ratio'][str(rho)] for r in robust_oracle['records']) for rho in cfg['ratios']}
    summary={'primary_budget':cfg['primary_budget'],'primary_ratio':cfg['primary_ratio'],'results':results,
             'calibration_grid_feasible_count':feasibility,'test_contexts':len(cfg['test_ids']),
             'calibration_and_confirmation_feasible_count':robust_feasibility,'robust_oracle_status':robust_oracle['status'],
             'selection':selection,'dataset':dataset,'physics':physics,
             'confirmation_unique_transients':data['confirmation_unique_transients'],
             'notes':['Three seeds reuse the same 16 conditions; seed std is not a confidence interval.',
                      'Fixed 8+8 makes only the initial measurement; active methods have an additional query allowance.',
                      'Calibration-grid oracle feasibility is not independent-packet feasibility.',
                      'Every frozen endpoint, including calibration failures, is evaluated on all confirmation packets.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'episodes.csv').open('w',newline='') as f:
        fields=['method','seed','context_id','budget','strength0','phase0','strength1','phase1',
                'calibration_quality','calibration_energy_ratio','confirmed_quality','confirmed_energy_ratio',
                'confirmed_worst_margin_vdd','confirmed_errors','measurement_bundles','controller_compute_seconds']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            c=r['confirmation'];writer.writerow({**{k:r[k] for k in ['method','seed','context_id','budget','calibration_quality','measurement_bundles','controller_compute_seconds']},
                **r['setting'],'calibration_energy_ratio':r['calibration']['energy_ratio'],'confirmed_quality':c['quality_all_patterns'],
                'confirmed_energy_ratio':c['energy_ratio'],'confirmed_worst_margin_vdd':c['worst_margin_vdd'],'confirmed_errors':c['errors_total']})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figp,(ax,ay)=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    for cc,color in [(0.,'#A6ADB8'),(.06,'#193B65'),(.12,'#822D48')]:
        with np.load(OUT/'development'/f'quiet_cc{cc:g}/wave.npz') as z:w={k:z[k] for k in z.files}
        ix=(w['t']>=12e-9)&(w['t']<=16e-9)
        ax.plot(w['t'][ix]*2e9-24,1000*w['rx'][ix,0],label=f'{1000*cc:.0f} fF coupling',color=color,lw=1.5)
        if cc==.12:ay.plot(w['t'][ix]*2e9-24,w['rx'][ix,1],color='#193B65')
    ax.set(ylabel='Quiet victim RX (mV)',ylim=(-50,65),title='Capacitive injection into a quiet lane');ax.legend(frameon=False,ncol=3,loc='upper right')
    ay.set(ylabel='Switching aggressor RX (V)',xlabel='Time (UI)',xlim=(0,8))
    for a in [ax,ay]:a.grid(alpha=.12)
    figp.savefig(OUT/'coupling_physics.png',dpi=170);plt.close(figp)
    colors={'traditional':'#193B65','bo':'#7E728D','ga':'#7F91A4','bc':'#BB929F','td3bc':'#822D48','random':'#B8BEC7'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for m in ['traditional','bo','ga','bc','td3bc']:
        vals=[results[str(b)]['0.9'][m]['confirmation']['mean'] for b in cfg['budgets']]
        axes[0].plot(cfg['budgets'],vals,'o-',color=colors[m],label=NAMES[m],lw=2)
    axes[0].set(xlabel='Additional setting measurements',ylabel='Joint quality + 10% energy saving (%)',xticks=cfg['budgets'],ylim=(-.5,15),title='Independent patterns: progress during calibration')
    axes[0].axhline(100*robust_feasibility['0.9']/16,ls='--',lw=1,color='#555555',label=f'Feasibility diagnostic ({robust_feasibility["0.9"]}/16)')
    axes[0].legend(frameon=False,fontsize=8,loc='best');axes[0].grid(alpha=.12)
    pm=['traditional','bo','ga','td3bc'];width=.19
    for j,m in enumerate(pm):
        x=np.arange(3)+(j-1.5)*width
        ss=[results[str(cfg['primary_budget'])][str(rho)][m]['confirmation'] for rho in cfg['ratios']]
        vals=[s['mean'] for s in ss];axes[1].bar(x,vals,width=width,color=colors[m],label=NAMES[m])
        for xx,s in zip(x,ss):
            ps=[r['joint_success_percent'] for r in s['seeds']]
            axes[1].scatter(xx+np.linspace(-.025,.025,len(ps)),ps,color='black',s=10,zorder=3)
    axes[1].set(xticks=np.arange(3),xticklabels=['5% saving','10% saving','15% saving'],ylabel='Independent joint success (%)',ylim=(-1,40),title='16 measurements: quality-energy tradeoff')
    axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    axes[1].grid(axis='y',alpha=.12)
    fig.suptitle('Capacitively coupled D2D lanes | 16 unseen circuit conditions',fontweight='bold')
    fig.text(.5,.025,'Both lanes must pass all three held-out patterns; energy uses the same-pattern 8+8 reference.\nDashed line: post-evaluation feasibility with calibration + held-out packets. Dots: seeds on the same 16 conditions.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.21,wspace=.28);fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    primary=results[str(cfg['primary_budget'])][str(cfg['primary_ratio'])]
    tab='\n'.join(f'| {NAMES[m]} | {formatted(primary[m]["calibration"])} | {formatted(primary[m]["confirmation"])} | {primary[m]["confirmation"]["quality_mean"]:.2f}% |' for m in METHODS)
    best=max(['traditional','bo','ga'],key=lambda m:primary[m]['confirmation']['mean'])
    td=primary['td3bc']['confirmation']['mean'];classic=primary[best]['confirmation']['mean']
    five=results[str(cfg['primary_budget'])]['0.95']
    judgement=('TD3+BC의 평균 성공률이 가장 높은 고전 기준선보다 높았습니다. 다만 소규모 파일럿이며 일반적 우위나 통계적 유의성을 확정하지 않습니다.' if td>classic else
               '이번 파일럿에서는 TD3+BC가 가장 높은 성공률을 보인 고전 기준선을 넘지 못했습니다.')
    report=f'''# 기생 커패시턴스로 결합된 D2D 두 레인: 보정 실험

{judgement}

**용량성 누화 구현은 확인했지만, 이 설정에서 RL의 필요성이나 우위를 입증하지는 못했습니다.**
주 목표인 10% 절감을 보정·독립 검사 모두 통과하면서 달성할 수 있는 조건은 16개 중
{robust_feasibility['0.9']}개뿐입니다. {cfg['primary_budget']}회 측정에서 주 목표 성공률은
전통 좌표 탐색 {formatted(primary['traditional']['confirmation'])}, TD3+BC {formatted(primary['td3bc']['confirmation'])}였습니다.
5% 목표에서는 전통 방식 {formatted(five['traditional']['confirmation'])}, TD3+BC {formatted(five['td3bc']['confirmation'])}였습니다.
또한 기본 8+8 설정만으로도 독립 패킷 품질 충족률이 {primary['fixed8']['confirmation']['quality_mean']:.2f}%였습니다.
따라서 단순히 누화를 추가한 것만으로 고전 방식이 어려워하는 유용한 RL 과제가 되었다고 볼 수 없습니다.

## 회로와 실제 확인한 현상

고정된 MOS 36개 송신부 두 개(총 72개), 각 레인의 3구간 RLC 배선, 서로 대응하는
배선 노드를 연결한 커패시터 3개를 ngspice로 시뮬레이션했습니다. 각 커패시터는
총 결합량 Cc의 1/3입니다. Cc는 그라운드 정전용량과 별개의 배선 간 소자이며,
기본 토폴로지와 모든 출력 트랜지스터는 제어 코드와 무관하게 유지됩니다.

조용한 레인에 옆 레인의 스위칭으로 유도된 peak-to-peak 전압은 Cc=0에서 거의 0,
60 fF에서 46.61 mV, 120 fF에서 85.19 mV였습니다. 결합을 제거했을 때 단독 레인 회로와의
최대 파형 차이는 공통 시간축에서 {physics['decoupled_vs_single_lane_max_error_mv']:.3f} mV입니다.
이 검사는 용량성 누화를 확인합니다. 누화가 있다는 사실 자체가 고전 제어보다 RL이 우수하거나
이번 과제가 충분히 어렵다는 증거는 아닙니다.

## 주 결과

초기 기준 측정 1회 + 추가 설정 측정 16회, 동일 트래픽 대비 송신 에너지 10% 절감 기준입니다.
추가 측정 4/8/16/32회의 결과와 5/10/15% 절감 목표를 모두 summary.json에 기록했습니다.
각 budget은 같은 32-step 보정 궤적의 중간 결과입니다.

| 방법 | 보정 패킷: 품질+절감 | 독립 패킷: 품질+절감 | 독립 패킷: 품질만 |
|---|---:|---:|---:|
{tab}

시드 3개는 같은 회로 조건 16개를 반복 평가합니다. ±는 시드 사이의 표본 표준편차이며,
48개 독립 회로 조건이나 신뢰구간을 뜻하지 않습니다. BO·GA·무작위는 탐색 시드,
BC·TD3+BC는 학습 시드입니다. 전통 좌표 탐색과 고정 설정은 결정적입니다.

보정 패킷에서 전수 격자상 달성 가능한 조건 수(분모 16):
5% 절감 {feasibility['0.95']}, 10% 절감 {feasibility['0.9']}, 15% 절감 {feasibility['0.85']}.
불가능한 조건도 주 결과 분모에 남겼습니다. 이것은 보정 패킷의 가능성으로,
별도 3개 패킷에서의 가능성 상한을 증명하지 않습니다. 격자 정답은 정책이나 데이터 행동 생성에 제공하지 않았습니다.

독립 평가 후 추가 진단으로 **보정 패킷과 별도 세 패킷을 모두 같은 설정으로 통과**하는 경우도
계산했습니다. 5/10/15% 절감은 각각 {robust_feasibility['0.95']}/{robust_feasibility['0.9']}/{robust_feasibility['0.85']}개 조건에서 가능합니다.
현재 테스트의 초기 8+8은 보정 패킷에서 모두 품질을 만족하므로, 공통 incumbent 규칙상 반환하는
설정도 반드시 보정 품질을 만족합니다. 따라서 보정에서 어떤 위상으로도 통과할 수 없는 드라이버 조합은
이 추가 진단에서 제외할 수 있습니다. 이는 출력 가능한 설정에 대한 전수 진단이며,
보정 실패를 허용하는 순수한 3-packet 최적해를 계산했다고 주장하지 않습니다.
원래 test.json과 정책·체크포인트는 바꾸지 않았습니다. 패턴별 정규화 에너지 비율이 달라지므로,
보정 패킷과 독립 패킷의 에너지 목표 가능성 개수는 같을 필요가 없습니다.

## 다음 설계에서 고칠 부분

이번 제어는 1/2/4/8셀이라는 거친 4단계입니다. 배선 부하는 고정되어 있고, 비활성 출력 셀의
데이터 게이트도 계속 구동하므로 셀 수를 줄이는 것이 송신 에너지 감소에 그대로 비례하지 않습니다.
현재 자료는 알고리즘 튜닝보다 **제어 가능한 에너지·신호 품질 범위의 점검이 먼저**임을 보여줍니다.
다음 실험에서는 더 세밀한 출력 코드 또는 slew/보상 탭처럼 물리적으로 의미 있는 조절값을
검토하고, 학습·개발 조건에서 달성 가능성과 고전 보정의 측정 비용을 확인한 뒤 새 테스트를 고정해야 합니다.
이번 테스트를 반복 보면서 목표나 회로 범위를 조정해 성공률을 올리는 방식은 사용하지 않았습니다.

## 평가 기준

- 품질: 두 레인의 모든 확인 패킷에서 center bit error 0, 샘플링 위상 +/-0.10 UI의
  9개 검사점에 대한 최악 전압 여유 0.08 VDD 이상.
- 확인 패킷: 레인별 독립 난수 패턴 2개와 두 레인이 반대 방향으로 스위칭하는 교대 패턴 1개,
  각 128비트/레인. 이 패턴 목록과 판정 규칙은 학습·평가 전에 정했습니다.
- 에너지: 동일한 회로·동일한 패킷의 8+8셀 송신부 대비 총 송신 에너지 비율.
  분자는 세 패킷의 에너지 합, 분모는 동일 패킷의 기준 에너지 합입니다.
  두 레인에 전송한 전체 비트 수로 나눈 pJ/bit를 기록합니다.
- 각 예산 시점의 설정은 그때까지 관측한 보정 utility가 가장 큰 설정입니다. 모든 방법에 동일한
  선택 규칙을 적용했고, utility가 같으면 두 레인 중 더 나쁜 쪽의 전압 여유가 큰 설정을 택했습니다.
  독립 패킷을 본 뒤 다른 설정으로 바꾸거나 재학습하지 않았습니다.
  보정에 실패한 설정도 독립 패킷에서 실제로 확인했으며 실패 샘플을 삭제하지 않았습니다.
- 1회 측정은 두 레인 각각 64비트, 9개 phase 검사를 포함합니다. 따라서 18개 phase probe입니다.
  모델에 전체 파형·설정 격자·숨은 기생성분을 제공하지 않습니다. 이미 계산한 파형을 재사용해도
  새로운 설정 질의의 논리적 측정 비용은 그대로 셉니다. 고정 8+8은 초기 측정만 사용합니다.
- 지표 F_B(r)는 예산 B에서 반환한 설정이 품질을 만족하면서 에너지 비율 r 이하인 조건의 비율입니다.
  성공한 샘플끼리만 에너지 평균을 비교하는 선택 편향을 피합니다.

## 상태·행동·보상

행동은 각 레인의 strength code와 phase code, 총 4개입니다. 출력 셀 수는 1/2/4/8 중 하나,
위상은 0..1 UI를 65개 코드로 나눴습니다. Vref는 0.5 VDD로 고정했습니다.
Actor의 연속 출력을 실제 코드로 양자화합니다. 이 실험의 최적성은 이 이산 행동 집합에 한정됩니다.

상태 입력은 50차원입니다. 최근 네 번의 설정·측정(각 9차원), 지금까지 가장 좋은 설정·측정(9차원),
VDD·전송률·margin 목표·guard 폭·남은 예산(5차원)을 포함합니다. 배선 RLC, 결합 Cc,
각 레인의 clock skew는 숨은 조건입니다. 최근 네 번의 기록으로 완전한 Markov 상태가 된다고 주장하지 않습니다.
고전 탐색은 전체 관측 이력을 내부에 유지하므로 기억 구조까지 동일한 비교는 아닙니다.

Utility는 품질을 만족하면 2-에너지비율, 실패하면
-2-4×두 레인 margin 부족량의 합-4×평균 오류비율-0.1×에너지비율입니다.
보상은 `새 best utility - 이전 best utility - 0.01`입니다. 에피소드는 32회 후 종료하며
중간 성공으로 조기 종료하지 않습니다. 시간 예산도 과제의 종료이므로 bootstrap mask를 0으로 둡니다.

## 데이터와 비교 방법

새 회로 조건 48개를 train 24 / validation 8 / test 16으로 분리했습니다.
공급 0.95..1.10 V, 온도 10..75 C, 전송률 레인당 2 Gbit/s, Cc 20..140 fF,
각 레인 배선·부하·clock skew를 변화시켰습니다. 난수 패킷은 조건별로 다르며 학습·평가 비트열을 분리했습니다.
학습 데이터는 **{dataset['transitions']:,} transitions / {dataset['episodes']:,} episodes**입니다.
무작위, 좌표 탐색, 무작위를 섞은 좌표 탐색으로 수집했고, 성공 정답이나 전수 격자 최적 행동을 라벨로 넣지 않았습니다.
무작위 초기 설정을 쓰는 학습 에피소드도 8+8 기준을 별도로 측정하고 비용을 기록했습니다.

- 전통 보정: strength와 phase를 번갈아 조절하는 좌표 탐색, 위상 step을 coarse-to-fine으로 감소.
- BO: Matérn-5/2 GP + expected improvement, 전역/현재 최선 주변의 후보에서 다음 질의 선택.
- GA: 평가된 population의 tournament selection, uniform crossover, mutation. 각 개체 평가도 측정 1회로 계산.
- BC와 TD3+BC: 같은 offline 데이터, hidden 2×128 ReLU, LayerNorm 없음. batch 256, lr 3e-4,
  gamma .95, target tau .005, actor 매 2 step 업데이트, target noise .2, clip .5.
- TD3+BC alpha .1/2.5, checkpoint 4k/8k, 세 시드. 고전 방법도 phase step, BO lengthscale,
  GA population/mutation 후보를 validation에서 선택했습니다. 후보 목록은 config.json에 있고 선택은 시드 공통입니다.
- 선택 결과: {json.dumps(selection,ensure_ascii=False)}

RL·BC는 과거 회로의 측정 데이터를 사용한 반면 BO·GA·좌표 탐색에는 학습 데이터로 만든 초기화나
surrogate prior를 제공하지 않았습니다. 따라서 이 비교는 offline 비용을 투입해 새 조건의 보정을
줄이는 설정입니다. 과거 데이터 구축·학습 비용이 0이라는 의미가 아니며 총 비용 우위는 별도 판단해야 합니다.

## 검증과 한계

72 MOS 고정 토폴로지, Cc=0 대조군, 조용한 피해 레인의 간섭을 확인했습니다.
데이터의 조건 분리·episode 연결·보상 재계산·행동 양자화, 원파형과 측정표 일치,
모든 반환 설정의 선택 경로, TD3+BC 시드 0의 평가 궤적 재현을 검사했습니다.
전통/BO/TD3+BC에서 margin 경계에 가까운 설정의 최악 패킷을 시간 간격 절반으로 재확인했습니다.
상세 수치와 통과 여부는 physics_audit.json, data_audit.json, evaluation_audit.json에 있습니다.

공개 PTM predictive MOS 모델이며 foundry PDK가 아닙니다. 배선은 임의의 lumped RLC+Cc이며
실제 interposer 추출값이 아닙니다. 수신기·위상 검사·Vref 비교는 Python의 기능 모델입니다.
열잡음·랜덤 jitter·metastability·CDR·공유 전원 잡음·배선 상호 인덕턴스는 포함하지 않았습니다.
짧은 패킷의 무오류를 장기 BER 또는 UCIe 준수로 해석할 수 없습니다.
에너지는 송신 드라이버와 predriver만 포함하며 RX·clock·보정 모니터·controller 비용을 포함하지 않습니다.
각 질의는 설정을 고정한 별도 transient입니다. 비트 전송 중의 연속 동적 제어 실험은 아닙니다.

## 재현

```bash
python3 -m pip install -r requirements_coupled.txt
python3 coupled_experiment.py prepare --split train
python3 coupled_experiment.py prepare --split validation
python3 coupled_experiment.py collect
python3 coupled_train.py --all
python3 coupled_evaluate.py validate
python3 coupled_experiment.py prepare --split test
python3 coupled_evaluate.py calibration
python3 coupled_evaluate.py confirm
python3 coupled_evaluate.py oracle
python3 coupled_evaluate.py oracle_confirmation
python3 coupled_audit.py data_audit
python3 coupled_audit.py evaluation
python3 coupled_report.py
```

원래 실행의 결과·파형·데이터·학습 모델을 함께 제공하므로 결과 확인에 재실행은 필요하지 않습니다.
기존 설정을 바꿀 때는 결과 폴더를 별도 보관하고 새 실험으로 실행해야 합니다.

물리적 동기: [고밀도 D2D의 누화 연구](https://arxiv.org/html/2404.05119v1).
관련 최적화: [신호 무결성의 학습 기반 최적화와 탐색 기준선](https://arxiv.org/html/2606.07463v1).
알고리즘: [TD3+BC 저자 구현](https://github.com/sfujim/TD3_BC).
'''
    (ROOT/'COUPLED_REPORT.md').write_text(report)
    (OUT/'versions.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,
         'torch':torch.__version__,'scipy':scipy.__version__,'ngspice':'39.3 bundled runtime'},indent=2)+'\n')
    print({'primary':primary,'calibration_feasibility':feasibility,'conclusion':judgement},flush=True)

if __name__=='__main__':run()
