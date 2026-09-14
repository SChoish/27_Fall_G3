# ALARM attention의 ADC 드라이버 적용

ALARM의 **좌표 self-attention과 학습되는 대각 metric**을 추가했습니다.
기본 탐색은 라인서치 없이 고정 길이의 방향을 투영합니다. 원본 VCO 실험의 완전한
재현은 아니며, 실제 SKY130 ADC 드라이버 평가기에 연결한 변형입니다.

검토 자료는 사용자 제공 `ALARM_V0.pdf` §4.2.2–4.2.3, Eqs. (24), (27),
그리고 `ALARM_Capstone_Technical_Report.pdf` §3.3–3.6입니다.
원문 PDF 자체를 이 저장소에 복제하지 않았습니다.

## 1. 구현한 attention

로그 정규화한 9차원 설계 벡터를 $u\in[0,1]^9$라고 합니다. 각 좌표를 스칼라
token으로 취급하고 원문의 $Q=K=V$ self-attention을 계산합니다.

$$
X=u[:,\mathrm{None}],\qquad
h(u)=\operatorname{softmax}(XX^\top)X,\qquad
p_\eta(u)=\operatorname{softmax}(W_\eta h(u)+b_\eta).
$$

각 token의 $d_k=1$이므로 $1/\sqrt{d_k}=1$입니다. 첫 softmax는 **한 설계 안의
좌표 축**에 적용합니다. 배치의 서로 다른 회로 설계를 섞지 않습니다.
학습되는 파라미터는 마지막 FC의 $W_\eta,b_\eta$입니다. 출력은 9개 좌표의 확률이고
초기 FC를 0으로 두어 균등 가중치에서 출발합니다. 별도의 Transformer 예측 모델이나
그래프 topology encoder를 추가한 것이 아닙니다.

현재 frozen surrogate의 스칼라 비용을 $\hat J(u)$, 그 gradient를
$g=\nabla_u\hat J(u)$라고 하면 다음 metric으로 방향을 계산합니다.

$$
w_i=9(1-\rho)p_{\eta,i}+\rho,\qquad \rho=0.02,
$$
$$
G=gg^\top+\operatorname{Diag}(w\odot g^2)
  +\left(10^{-4}+0.01\,\operatorname{mean}(g^2)\right)I,
\qquad d=-G^{-1}g.
$$

원문 Eq. (24)의 $\mu=9$로 두면 균등 확률 $p_i=1/9$일 때 기존의
$\operatorname{Diag}(g^2)$ 항과 일치합니다. $\rho$는 균등분포 혼합이고,
마지막 ridge는 gradient가 0인 좌표에서도 가역성을 확보하는 수치적 수정입니다.
`--attention off`는 정확히 균등 가중치를 사용합니다.

**큰 가중치는 그 좌표로의 이동에 더 큰 metric 비용을 부여합니다.** 이를 곧바로
“더 중요한 소자”나 회로의 인과적 중요도로 해석하면 안 됩니다.

## 2. Attention을 어떻게 학습하는가

먼저 지금까지 비용을 지불한 **모든 유효 실측**으로 residual ridge를 적합합니다.
그 뒤 encoder, regression/classification head와 ridge 계수는 고정합니다.
이미 평가한 유효 설계 중 최대 32개를 고르고, 설계당 한 번의 미분 가능한 이동을 만듭니다.

$$
u_i^+=\Pi_{[0,1]^9\cap[u_i-r,u_i+r]}
\left(u_i+s\frac{d_\eta(u_i)}{\|d_\eta(u_i)\|_2}\right),
\qquad
L_{\mathrm{att}}(\eta)=\frac1B\sum_i\hat J(u_i^+).
$$

기본값은 $s=0.15,r=0.25$입니다. 0에 가까운 방향의 norm에는 수치적 하한을 둡니다.
gradient는 endpoint surrogate → 투영 → 선형계 풀이 → attention으로 흐릅니다.
출발점과 frozen surrogate가 고정되어 있으므로 출발점의 $g$는 detach해도 이 **한 단계**
attention gradient는 정확합니다. 학습에는 Armijo의 이산적인 accept/reject를 넣지 않습니다.

Adam(lr=0.01)으로 40회 업데이트하며 해당 배치 비용이 가장 낮았던 attention을 보존합니다.
이 값은 **학습 배치의 surrogate 비용**이며 validation 성능이나 SPICE 개선율이 아닙니다.
attention 가중치는 다음 query로 이어지고 optimizer는 residual 재적합마다 새로 만듭니다.
attention 배치의 RNG를 후보 bank의 RNG와 분리해 on/off 비교에서 bank 난수가 달라지지
않도록 했습니다. attention에는 offline validation/test label을 전달하지 않습니다.

이 단계에는 추가 SPICE 호출, 수동 pos/neg 라벨, elite 전용 데이터셋이 없습니다.
기존 RNC는 성능 거리 순위에 기반한 별도 표현학습입니다.

## 3. 라인서치와 후보 선택

기본 `--line-search none`은 정해진 길이로 이동하고 설계 범위 및 시작점의 지역 반경으로
투영합니다. 각 시작점에서 최대 6회 이동하며 **모든 중간 후보를 보존**합니다.
상위 3개 실측 설계와 그 주변의 2개 교란 설계가 시작점입니다. 여기에 local random 64개와
global random 64개를 추가하고, 이미 평가한 후보를 제외한 뒤 surrogate 비용으로 하나를
선택합니다. 실제 성공·실패는 SPICE의 hold 오차와 에너지로 판정합니다.

`--line-search armijo`에서는 같은 방향에 대해 최대 8회, 배율 $2^{-k}$로
surrogate Armijo 검사를 합니다. 투영 후 실제 변위로 $g^\top(u^+-u)$를 계산합니다.
기존 코드는 마지막 iterate만 남겼지만 이번에는 두 모드 모두 중간 후보를 남깁니다.
따라서 `off + armijo`도 과거 번들과 완전히 동일한 replay라고 부르지 않습니다.

예측 모델의 오차 때문에 Armijo를 만족해도 실제 회로 성능은 나빠질 수 있습니다.
라인서치의 효과는 **동일 SPICE 예산의 최종 실측 성능**과 surrogate 계산량을 나누어
비교해야 합니다. 라인서치 유무와 관계없이 선택된 후보 하나만 평가 예산을 소비합니다.

## 4. 원문과 현재 구현의 범위

| 요소 | 현재 구현 |
|---|---|
| 좌표 self-attention → FC → softmax | 원문 구조를 9변수에 적용 |
| attention 가중 대각 metric | 적용; 균등 기준 맞춤, 확률 혼합과 ridge 추가 |
| 고정 critic으로 후보 비용을 낮추는 attention 학습 | 적용; 학습은 한 단계의 미분 가능한 이동 |
| 여러 surrogate 이동 후 SPICE 평가 | 적용; 고정 단계 기본, Armijo 선택 가능 |
| critic 학습 | offline regression/RNC + online residual ridge, 원문의 전체 재학습과 다름 |
| critic buffer | 미구현 |
| elite | 실측 상위 3개 시작점; 학습은 모든 유효 실측에서 선택 |
| 회로 topology | 14-MOS ADC 입력 드라이버 하나로 고정 |
| 미지 topology 일반화 | 구현·검증하지 않음 |

이 파이프라인은 새로운 SPICE 평가를 추가하는 **surrogate 기반 설계 최적화**입니다.
고정 데이터만 사용하는 offline RL 실험은 아닙니다.

## 5. 실행과 검증

```bash
python -m pip install -r requirements-learning.txt
python -m unittest discover -s tests -v
python run.py optimize --method rnc --checkpoint checkpoints/rnc.pt --budget 32 --output results/my_attention
python compare_attention.py --budget 32 --seeds 0 1 2 --workers 2 --include-baselines --output results/my_ablation
```

`compare_attention.py`는 2×2 조합과 선택적 GA·무작위를 별도 프로세스에서 실행합니다.
각 run은 초기 8개까지 새로 평가하며 실패 query도 셉니다. summary의 offline source 비용은
각 방법이 이용한 사전 데이터 규모이고, 동일 source를 재사용했다면 run 수만큼 재수집한
비용으로 중복 합산하면 안 됩니다. `attention.pt`는 run 종료 시점의 attention snapshot입니다.

수치 검증에는 metric의 양의 정부호성, 0 gradient, box 경계, 극단 가중치,
균등 가중치의 기존 metric 일치, meta-gradient와 유한차분 일치, 배치 간 독립성,
고정 critic 보존 및 attention의 실제 후보 변경을 포함합니다.

새 SPICE 결과는 `results/attention_demo/summary.csv`와 `summary.json`, 상세 기록은
각 조합 폴더의 `trace.json`에서 확인할 수 있습니다. 포함된 source는 24개뿐이고 초기
설계에 알려진 feasible reference가 있으므로, 이 작은 실행은 통합 검증이며 알고리즘
우위나 미지 회로 일반화를 입증하는 실험이 아닙니다.

## 6. 이번 실제 실행 결과

ngspice 39 / SKY130 TT, RNC source 24개, seed 0, 설정별 **16 design queries**로
실행했습니다. 초기 8개를 포함하며 총 6설정에서 새 SPICE transient 96회를 수행했습니다.
surrogate는 기존 RNC checkpoint를 공유했고 attention은 각 run에서 새로 시작했습니다.

| 설정 | Attention 업데이트 | Armijo 검사 | 초기 이후 metric 후보 선택 | 최종 feasible 에너지 |
|---|---:|---:|---:|---:|
| attention off, line search none | 0 | 0 | 8/8 | 169.298197 pJ/sample |
| attention off, Armijo | 0 | 262 | 8/8 | 169.298197 pJ/sample |
| attention learned, line search none | 320 | 0 | 8/8 | 169.298197 pJ/sample |
| attention learned, Armijo | 320 | 241 | 8/8 | 169.298197 pJ/sample |
| 무작위 | 0 | 0 | 해당 없음 | 169.298197 pJ/sample |
| GA | 0 | 0 | 해당 없음 | 169.298197 pJ/sample |

모든 run의 최종 설계는 query 3에서 새로 평가한 초기 feasible reference였습니다.
오차는 0.981292 mV이고 **초기 대비 에너지 개선은 모두 0%**입니다.

- 라인서치를 끈 상태에서 attention on/off의 새 제안은 8개 모두 달랐습니다.
- attention을 켠 두 run은 Armijo 유무와 관계없이 새 제안 8개가 정확히 같았습니다.
  Armijo를 켠 쪽은 surrogate 평가를 241개 더 수행했습니다(9,056 → 9,297).
- attention을 끈 두 run에서는 Armijo 유무에 따라 5/8개 제안이 달랐지만 최선 실측은
  같았습니다. 따라서 “라인서치는 항상 무의미하다”는 결론은 이 결과로 지지되지 않습니다.
- 첫 attention 업데이트의 동일 학습 배치 surrogate 비용은 1.623079 → 1.230547로
  낮아졌습니다. 이는 학습이 연결됐다는 확인이며 실측 성능 개선을 뜻하지 않습니다.

현재 결과는 라인서치를 기본적으로 끄는 간단한 구성과 양립합니다. 효과 비교에는 더 큰
사전 데이터와 여러 seed·예산이 필요합니다. 실행 시간은 3개 프로세스의 동시 실행에서
측정했으므로 이 표를 wall-clock 속도 비교로 사용하면 안 됩니다.
