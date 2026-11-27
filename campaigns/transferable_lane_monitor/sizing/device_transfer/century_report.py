"""Generate the results tables and identify which constraint limits measured target designs."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from century_audit import ARMS, verified
from century_run import DOMAINS
from circuit import ratios
from experiment import upper
from warm_space import TEST

NAMES={"random":"전체 폭 랜덤","group_random":"역할 4좌표 랜덤","bo":"제약 확률 BO","de":"차분 진화 DE",
       "acor":"전체 폭 ACOR","group_acor":"역할 4좌표 ACOR","residual":"잔차 critic, 웜업 0",
       "warm4":"잔차 critic, 웜업 4","rank2":"방향 선호 rank-2","shuffled":"방향 라벨 셔플","role_poll":"역할 규칙 + 직접 탐색"}

def main():
    root=Path("runs/century")
    rows=list(csv.DictReader((root/"summary.csv").open()))
    index={(r["protocol"],r["arm"],r["domain"],float(r["goal"]),int(r["limit"])):r for r in rows}
    def row(p,a,g,c=100,d="all"):return index[p,a,d,g,c]
    def cell(p,a,g,c=100,d="all"):
        r=row(p,a,g,c,d)
        return f'{r["hits"]}/{r["n"]} · {float(r["mean_calls"]):.2f}'
    def make_table(header,rows):
        return "\n".join(["| "+" | ".join(header)+" |","|"+"---|"*len(header),
                          *("| "+" | ".join(map(str,r))+" |" for r in rows)])
    audit=json.loads((root/"audit.json").read_text())
    geometry=json.loads((root/"metrics.json").read_text())
    ranking=sorted(ARMS,key=lambda a:float(row("joint",a,.95)["mean_calls"]))
    header=["방법","5% ≤40회","5% ≤100회","10% 단독 ≤40회","10% 단독 ≤100회","10% 공동 ≤100회"]
    tables=[make_table(header,[[NAMES[a],cell("joint",a,.95,40),cell("joint",a,.95),
                               cell("strict",a,.9,40),cell("strict",a,.9),cell("joint",a,.9)] for a in ranking])]
    nets=list(DOMAINS)
    for p,g in (("joint",.95),("strict",.9)):
        tables.append(make_table(["방법",*nets],[[NAMES[a],*(cell(p,a,g,d=n) for n in nets)] for a in ranking]))
    split=make_table(["방법","기존 4조건 5%","신규 2조건 5%","기존 4조건 10% 단독","신규 2조건 10% 단독"],
                     [[NAMES[a],*(cell(p,a,g,d=d) for p,g in (("joint",.95),("strict",.9))
                                                for d in ("previous","fresh"))] for a in ranking])
    with ZipFile("century_measurements.zip") as z:
        def read(n):return json.loads(z.read(n))
        constraints=[]
        for n,c in DOMAINS.items():
            raw={(tuple(x),k):r for x,k,r in read("raw/"+n+".json")}
            ref=[raw[tuple(np.ones(len(upper(c)))),k] for k in range(4)]
            counts=[0]*6
            counts[0]=len([k for x,k in raw if k==0])-1
            witnesses=[]
            for x,k in raw:
                if k or x==tuple(np.ones(len(x))):continue
                y=ratios([raw[x,j] for j in range(2)],ref[:2])
                if y[0]<=np.log(.95):counts[1]+=1
                if max(y[1:])<=0 and y[0]<=np.log(.95):counts[2]+=1
                if (x,2) not in raw:continue
                counts[3]+=1
                arr=[raw[x,j] for j in range(4)]
                for i,g in enumerate((.95,.90)):
                    if verified(arr,ref,g):counts[4+i]+=1
                p=[ratios(arr[j:j+2],ref[j:j+2]) for j in (0,2)]
                if all(max(v[1:])<=0 for v in p):
                    witnesses.append(float(np.exp(max(v[0] for v in p))))
            constraints.append([n,*counts[:6],f"{min(witnesses):.5f}" if witnesses else "—"])
        diagnostics=make_table(["조건","개발 후보","개발 EDP ≤.95","개발 제약도 통과","확인한 후보","4코너 5%","4코너 10%","최저 확인 EDP"],constraints)
        trials=[read(k) for k in z.namelist() if k.startswith("traces/")]
    pairings=[]
    for a,b in (("rank2","residual"),("rank2","shuffled"),("warm4","residual"),("acor","random"),
                ("group_acor","group_random"),("group_acor","acor"),("de","random")):
        values=[]
        for p,g in (("joint",.95),("strict",.9)):
            times=lambda arm:{(r['circuit'],r['seed']):r['hits'].get(str(g),100)
                              for r in trials if r['arm']==arm and r['protocol']==p}
            x,y=times(a),times(b)
            diff=np.array([x[k]-y[k] for k in x]);assert len(diff)==18
            values.extend([f"{sum(diff<0)}/{sum(diff==0)}/{sum(diff>0)}",f"{diff.mean():+.2f}"])
        pairings.append([NAMES[a]+" vs "+NAMES[b],*values])
    comp=make_table(["대응 비교 A vs B","5% A 승/무/패","5% A−B 비용","10% A 승/무/패","10% A−B 비용"],pairings)
    accuracy=[]
    for a in ("rank2","shuffled"):
        for seed in (31,32,33):
            record=geometry[f"{a}-{seed}"]
            base,learned=np.mean(list(record["held_out_direction_accuracy"].values()),axis=0)
            accuracy.append([NAMES[a],seed,f'{record["source_ordinal_accuracy"]:.1%}',f"{base:.1%}",f"{learned:.1%}"])
    acc=make_table(["방향 모델","seed","전체 방향 자료 적합 후 정순위","보류 회로 기본 방향","보류 회로 학습 방향"],accuracy)
    lines=[
      "# 100 SPICE eval에서의 최초 FoM 도달과 회로 전이",
      "5%는 기존 기준과 같은 공동 추적의 첫 0.95 도달이고, 10% 단독은 0.90 가능 후보만 확인한다. 각 표는 성공 수/시행 수와 실패 포함 제한 평균 SPICE 호출 수다. 독립적인 회로 형상은 세 가지이며 각각 두 PDK를 짝지었다. 형상 두 가지와 부하 조건은 이전 실험에서 사용했고, 3출력 비대칭 병렬 회로 한 가지는 이번에 추가했다. 각 조건의 확률적 탐색은 3 seed, 결정적 역할 직접 탐색은 조건당 1회다. 방법 간 승패는 이 회로들의 기술 통계다.",
      "## 전체 탐색 비용",tables[0],
      "성공 여부는 동일한 소자 폭에서 개발·확인 코너의 에너지×최대 에지 지연과 모든 전기 제약을 실측하여 판정한다. 평균은 최초 도달까지의 호출 수와 미달 시 해당 상한으로 계산했다. 평균과 성공 수에는 100번째 호출에 성공한 경우가 서로 다르게 반영될 수 있다. 타깃의 기준 4회를 포함하고 워밍업과 확인도 유료다. 40회와 100회는 같은 실행에서 읽었다.",
      f"BO 획득함수를 5%에 맞춘 추가 대조군은 40회 {cell('bo95','bo',.95,40)}, 100회 {cell('bo95','bo',.95)}였다. 공동 BO의 획득함수 문턱은 10%다. 두 실험에서 후보 풀, 커널과 seed를 동일하게 유지했다.",
      "![최초 도달 누적 성공률](century_attainment.svg)",
      "## 전이 조건별 결과",split,"### 5% 목표, 100회 상한",tables[1],"### 10% 단독 목표, 100회 상한",tables[2],
      "## 같은 회로와 seed의 대응 비교",comp,
      "## 회로별 실측 가능성",diagnostics,
      "두 프로토콜과 모든 방법의 고유 후보를 합쳤다. 개발 EDP, 네 코너 제약 및 가장 낮은 확인 EDP는 서로 다른 열이다. 후보가 확인되지 않았거나 통과하지 못한 경우, 측정 범위를 넘어선 최적값은 알 수 없다.",
      "## 실제 SPICE 방향 선호로 학습한 metric",
      "각 SKY130 소스 회로에서 3개 anchor와 anchor당 8개 방향을 사용했다. 방향은 로그 폭 공간에서 L2 길이 0.10이고 실제 폭 반올림 오차는 허용했다. 같은 anchor의 28쌍은 8번의 후보 측정을 공유한다. 실측에서 허용 제약을 충족한 방향을 먼저 선호하고 양의 로그 제약 위반 합, EDP 순으로 비교했다. 12개 방향 집합이 제공한 336개 비교를 독립적인 실험 횟수로 세지 않는다.",
      "기본 대각 행렬 D, 역할 특징 Φ, 학습 행렬 B로 역 metric을 D^(-1/2) [I + Φ B Bᵀ Φᵀ] D^(-1/2)로 구성했다. 10×2 행렬 B는 측정 순위와 제안 방향 내적의 쌍대 로지스틱 손실로 학습했다. critic, 타깃 잔차 보정, 후보 수와 선탐색은 웜업 0과 같다. 표의 보류 조건은 한 소스 회로의 방향 라벨을 제외하고 B를 재학습한 결과다. 기반 critic은 기존 네 소스 회로의 324회 자료를 사용했다.",
      acc,
      "## 비용 검산과 재현",
      f"공동 및 단독 목표를 합쳐 {audit['runs']}개 프로토콜별 궤적을 원시값에서 재계산했다. 독립 실행으로 청구한 타깃 호출 수는 공동 {audit['charged_target']['joint']:,}회, 10% 단독 {audit['charged_target']['strict']:,}회, 목표를 맞춘 5% BO {audit['charged_target']['bo95']:,}회다. 동일한 설계의 코너값을 재사용했으며 두 프로토콜을 합친 고유 타깃 코너 측정은 {audit['distinct_target_corner_rows']:,}개다. 실패 측정은 {audit['target_errors']}개다. 기본 학습 비용은 소스 324회, 방향 선호에는 추가 후보 192회와 nominal 기준 재측정 8회를 썼다. 모델과 무관한 탐색은 소스 비용이 0이다.",
      f"0.25 ps로 최초 도달 설계 {audit['distinct_first_hit_designs']}개와 해당 기준 설계를 검산했다. 재검산 비용은 {audit['precision_checks']}회이며 이전 동일 정밀도 자료 {audit['precision_reused']}회를 포함한다. 최초 도달 기록 {audit['hits']}건 중 재검산 미통과는 {len(audit['precision_failures'])}건이다.",
      "기본 ngspice 실행과 컴파일 모듈을 재사용하는 실행기를 동일한 6개 회로 × 4코너에서 비교해 에너지, 지연, slew, 레일 출력 측정값의 일치를 확인했다. 각 eval은 독립적인 ngspice 메모리·회로·동작점에서 수행한다.",
      "기존 패키지의 ngspice 및 PDK 인자로 century_run.py search, century_run.py source, century_role.py search, century_strict.py search, century_bo_goal.py search, century_audit.py search 순서로 실행한다. python century_restore.py는 시뮬레이터 없이 압축 원자료에서 모든 최초 도달과 표를 재생한다.",
      "[평가 규약](century_protocol.md) · [전체 수치](runs/century/summary.csv) · [원시 측정과 궤적](century_measurements.zip) · [검산](runs/century/audit.json)",
      "[Socha & Dorigo, ACOR](https://doi.org/10.1016/j.ejor.2006.06.046) · [Storn & Price, DE](https://doi.org/10.1023/A:1008202821328)"
    ]
    Path("CENTURY_RESULTS_KO.md").write_text("\n\n".join(lines)+"\n")

if __name__=="__main__":main()
