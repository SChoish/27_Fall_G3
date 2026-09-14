# D2D: 고정 예산 성공 목적과 reward 비교

현재 학습기는 `--reward-mode`로 세 보상을 비교한다. 옵션을 생략하면 기존 학습 경로를
사용한다. 기존 Environment와 원천 데이터는 보존하며, TD3+BC와 IQL 입력에서 보상을 다시
계산한다. BC는 보상을 사용하지 않으므로 이 옵션을 거부한다.

| mode | 보상 |
|---|---|
| `legacy` | 기존 incumbent utility 개선량 − 0.01 |
| `terminal_success` | 마지막 전이에서 반환 incumbent의 성공 여부 |
| `potential_success` | 종료 성공 + eta × (gamma × 다음 potential − 현재 potential) |

## 성공과 보조 신호

성공 S는 반환 incumbent가 **학습용 calibration 패킷**에서 양쪽 마진 0.08 이상,
오류 0, 기준 대비 에너지 비율 0.9 이하인 경우다. 독립 패킷의 품질을 reward에 넣지
않는다. 따라서 독립 입력 성공률과의 차이는 일반화 문제로 남는다.

potential은 incumbent가 아니라 **가장 최근 측정**을 사용한다. 좋은 기준 설계가
있어도 실패 정도의 차이가 중간 학습 신호에 나타나도록 하기 위해서다.

```text
D = sum_i max(0, (margin_required - margin_i) / margin_required)
    + mean_i(error_fraction_i)
    + max(0, energy_ratio / success_energy_ratio - 1)
Phi_t = -D / (1 + D), t < B
Phi_B = 0
```

원시 관측 50차원 중 마지막 측정은 31:36, 반환 incumbent 측정은 40:45를 사용한다.
potential은 [-1, 0]으로 제한되며, eta는 보조 신호의 크기다. 현재 비교는 eta=1을
사전에 고정했다. 계수가 TD3+BC에 최적이라는 주장은 하지 않는다.

모든 episode가 같은 B에서 종료되므로 경로별 할인 합은 다음과 같다.

```text
sum_{t=0}^{B-1} gamma^t r_t
    = gamma^(B-1) S(h_B) - eta Phi_0(h_0)
```

초기 상태 분포가 정책과 무관하면, 두 번째 항은 정책에 따른 기대 성공률의 순서를
바꾸지 않는다. 고정 B에서는 첫 번째 항의 할인 계수도 공통이다. 이는 중간 개선을
언제 얻는지에 가중치를 주는 기존 delta utility와 다른 목적이다.

[Ng, Harada, Russell (ICML 1999), 저자 제공 원문](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)의
Theorem 1 / 식 (2)에 제시된 potential 차분 조건을 확인했다. 위 유한 예산 구현은
종료 potential을 0으로 두고 경로별 telescoping 식을 별도로 검증한다.
논문의 정책 불변성 결과를 TD3+BC의 함수 근사나 최적화 과정의 불변성으로 해석하지
않는다. 특히 Q 값의 이동은 `alpha / mean(abs(Q))` 정규화와 BC 항의 상대적 영향을
바꿀 수 있다. 마지막 학습 batch의 Q 절댓값, Q 가중치와 BC MSE를 기록한다.

## 비교 조건과 검증

2026-09-15 비교는 기존 mixed 데이터 96 episode에서 각각 첫 16개 전이,
총 1,536개를 사용한다. 세 보상 모두 같은 전이를 사용하고 관측·행동·context·종료
마스크 배열 해시가 일치하는지 검사한다. 원래 32회 수집 데이터의 나머지 절반을
사용하지 않아도, 이미 지불한 원천 수집 비용이 줄어드는 것은 아니다.

- 모든 비교군은 16번째 전이에서 terminal=true, timeout=false로 bootstrap을 끊는다.
  이는 새로 정의한 유한 예산 과제의 종료다.
- 관측의 기존 남은 시간 인코딩 (32−t)/32는 학습과 평가에서 공통 유지한다. 고정 16회 과제에서도
  시간은 구분되며, 평가는 정확히 16회 측정 후 종료한다.
- seed 0/1/2, alpha=0.1, gamma=0.95, eta=1, 50,000 학습 step을 공통 고정한다.
  보상군 간 모델 초기화와 batch 추출의 seed를 맞춘다.
- 주 결과는 사전 고정한 50k checkpoint다. 5k/15k 결과는 경과 확인용으로 보존한다.
- 기존 validation 12조건의 calibration 패킷에서 평가한다. 독립 test 테이블과 패킷은
  복사하거나 읽지 않는다. 새 SPICE 실행은 없다.
- 기존 32회 과제의 본학습 결과와 직접 성능 차이를 계산하지 않는다.

`tests/test_reward_design.py`는 실패 정도의 신호, 반환 incumbent 성공 판정,
종료 potential, 여러 예산·할인율에서의 할인 합, 공통 데이터와 종료 마스크,
잘못된 episode 연결 거부를 검증한다. 실제 데이터에서도 할인 합 오차를 검사한다.
reward 명세는 checkpoint에 저장하고, 학습 코드와 보상 코드 해시를 출처 의존성에
추가하여 다른 구현의 체크포인트를 조용히 재사용하지 않도록 한다.

## 재현

저장소 루트에서 기존 mixed 수집 결과를 새 디렉터리에 복사하고 실행한다.

```bash
python experiments/train_reward_ablation.py \
  --source campaigns/20260914_quality_s0-2/d2d_qualities/mixed/results/coupled_v1 \
  --output campaigns/new_reward_comparison
python campaigns/new_reward_comparison/train_reward_ablation.py \
  --output campaigns/new_reward_comparison --execute
```

실행기는 입력 해시를 확인하고 3개 worker로 9개 학습을 수행한다. 기존 폴더에
초기화하거나 서로 다른 조건의 결과를 이어 쓰지 않는다. 완료 보고서는
`execution/REPORT.md`, 상세 데이터는 `execution/summary.json`에 생성된다.
2026-09-15 실제 실행의 자원 제한과 고정 입력 목록은 해당 캠페인의 `launcher.json`,
`computation_contract.json`, bounded runner의 manifest에 기록한다.

## a4bb6d6 리뷰 후 학습량 진단

1,536개 전이, batch 256, 50,000 step은 총 12,800,000번 추출이고 전이당 기대
추출 횟수는 약 8,333.33회다. 이는 과적합의 증명이나 적정 학습량의 근거가 아니다.
새 `training.json`과 학습 로그에는 실제 전이 수, batch 크기, 학습 step과 actor/critic 각각의 optimizer step,
총 추출 횟수와 전이당 평균 추출 횟수를 남긴다.

기존 `q_abs_mean`, `q_weight`, `bc_loss`에 더해 actor loss의 Q 항 값과
**가중된 Q 항/BC 항 각각의 actor-parameter gradient norm**을 기록한다.
정규화 계수만으로 실제 영향력을 추측하는 한계를 줄이기 위한 진단이다.
진단용 autograd는 parameter gradient에 누적하지 않는다. 동일 seed의 1,000-step
테스트에서 진단 계산 유무에 따른 actor/critic 최종 가중치가 정확히 같음을 확인했다.
보상 수식과 기본 TD3+BC 업데이트는 바꾸지 않았다.

후속 학습량 실험은 아래 규칙을 새 프로토콜에 고정했다. 기존 50k 결과에 소급 적용하지 않는다.

- 관찰 시점: 1k / 3k / 5k / 10k / 15k / 30k / 50k.
- reward별로 같은 validation 규칙: 3 seed 평균 성공률 우선, 동률이면 통과 설계 평균
  에너지, 다시 동률이면 더 이른 checkpoint. 통과가 모두 없으면 에너지 순위를 만들지 않는다.
- 전체 12조건 결과를 유지하며, 개발 분석에서 달성 가능하다고 알려진 조건의 결과는
  보조 지표로 별도 보고한다. 같은 조건의 seed 반복을 독립 회로 조건으로 세지 않는다.
- Q 정규화 보조 실험은 모든 reward에서 데이터·학습량·선택 규칙을 맞추고 정규화만
  바꾸는 별도 축으로 둔다. Q/BC gradient norm과 실제 성공률을 함께 확인한다.
- 선택 이후의 일반화 주장은 별도로 고정한 새 입력에서만 평가한다.

`train_reward_ablation.py --checkpoint-steps 1000 3000 5000 10000 15000 30000 50000`으로
새 시점과 선택 규칙을 연결했다. 학습 schedule은 데이터의 config/provenance를 바꾸지 않고
별도 인자로 전달하며 checkpoint와 training.json에 보존한다. 동일 seed의 최종 가중치가
중간 checkpoint 추가로 달라지지 않는지 테스트했다.

2026-09-15 전체 후속 실행은 `campaigns/20260915_full_review_s0-2/d2d_reward`에서
세 보상 × 세 seed × 50k step으로 시작했다. 고정 50k 결과와 validation으로 선택한
시점을 별도로 보고하며, 7시점의 Q/BC gradient와 전이당 추출량을 저장한다.
진행 상태는 상위 `status.json`, 결과는 `execution/summary.json`과 `REPORT.md`에 기록한다.
Q 정규화 변경은 이 실행에 포함하지 않는다. 과거 고정 50k 비교의 세 보상 성공 0 결과와
새 실행 결과를 구분한다.

## 원형 IQL

TD3+BC 실패만으로 다른 가치 학습이 실패한다고 보지 않는다. `--method iql`은
Kostrikov, Nair, Levine (2021) 원형이다. 같은 mixed 데이터, 같은
`terminal_success` 보상, 같은 16회 예산에서 TD3+BC와 비교한다. 데이터나 마진
보조 학습은 동시에 바꾸지 않는다.

| | TD3+BC | IQL |
|---|---|---|
| 부트스트랩 | \(r+\gamma Q_{\mathrm{targ}}(h',\pi_{\mathrm{targ}}(h'))\) | \(r+\gamma V(h')\) |
| 가치 개선 | actor가 제안한 행동의 Q | 데이터 행동의 Q로 expectile \(V\) |
| 정책 추출 | Q를 높이는 actor 업데이트 | 데이터 행동을 advantage로 가중한 BC |

결정적 tanh actor에는 가우시안 log-probability 대신 advantage 가중 MSE를 쓴다.
expectile 0.7, temperature 3은 원문 locomotion 기본값이며 validation에서 고르지
않았다. 학습 중 critic은 actor 행동을 평가하지 않는다. 주 지표는 반환
incumbent 성공과 논리 측정 횟수다. 전수 격자 Q 순위는 정책 성공 기준이 아니다.

```bash
python experiments/train_iql_compare.py \
  --source campaigns/20260914_quality_s0-2/d2d_qualities/mixed/results/coupled_v1 \
  --td3bc-execution campaigns/20260915_reward_s0-2/execution/summary.json \
  --output campaigns/20260915_iql_terminal_s0-2
python campaigns/20260915_iql_terminal_s0-2/train_iql_compare.py \
  --output campaigns/20260915_iql_terminal_s0-2 --execute
```

CQL, Diffusion-QL, IQL+AMO는 이 비교에 넣지 않는다.


## IQL 추가 설정 비교: τ=0.9, β=10

`campaigns/20260915_iql_tau09_beta10_s0-2`에서 새로운 설정만 seed 0/1/2로 각각
50k step 학습한다. 기존 `(0.7,3)` 결과는 재학습하지 않고 원본 입력 해시를 확인한 후
별도 reference로 복사·동결한다. train/validation 데이터, 보상, 예산 16, actor 구조,
평가 코드, 50k 주 평가 시점은 유지한다. 두 파라미터 각각의 효과를 분리하는 비교는 아니다.

보고서의 알고리즘 표기는 **IQL, deterministic actor / advantage-weighted MSE**다.
CLI의 `temperature`는 β를 뜻하며 `min(exp(β * (Q - V)), 100)`으로 곱한다.
실제 한 step 학습에서 같은 advantage에 β=3과 β=10을 적용했을 때 지수 입력의 비가
10/3인지 테스트했다. [원문 Appendix B](https://arxiv.org/html/2110.06169v1#A2)의
AntMaze 설정이 후보 선정 근거이며 D2D의 최적 설정이라는 뜻은 아니다.

```bash
python experiments/train_iql_compare.py \
  --source campaigns/20260914_quality_s0-2/d2d_qualities/mixed/results/coupled_v1 \
  --iql-execution campaigns/20260915_iql_terminal_s0-2/execution/summary.json \
  --td3bc-execution campaigns/20260915_reward_s0-2/execution/summary.json \
  --expectile 0.9 --temperature 10 \
  --output campaigns/20260915_iql_tau09_beta10_s0-2
python campaigns/20260915_iql_tau09_beta10_s0-2/train_iql_compare.py \
  --output campaigns/20260915_iql_tau09_beta10_s0-2 --execute
```

추가 SPICE는 0회이며 기존 12개 개발 validation 조건을 3개 seed로 반복한다.
각 실행의 receipt와 출력 해시를 저장하고 기존 학습 폴더가 있으면 덮어쓰지 않는다.

추가 설정 실행 완료: 50k seed별 성공은 (0.7,3)=[1,0,0]/12, (0.9,10)=[0,1,0]/12다.
두 설정 모두 총 1/36이며 성공 조건은 context 27이다. 안정적인 개선은 확인되지 않았다.
새 설정의 5k=[0,0,1]/12, 15k=[0,0,0]/12는 보조 기록으로 남겼다.
비교 보고서는 `campaigns/20260915_iql_tau09_beta10_s0-2/execution/REPORT.md`다.

## 온라인 SAC

고정 mixed 데이터에서 성공 전이가 2/96뿐이라, 같은 16회 종료·같은
`terminal_success` 보상에서 **새 행동을 직접 질의하며** 학습이 되는지를
먼저 본다. `--method sac`는 Haarnoja et al., ICML 2018의 확률적 정책
SAC다. 기존 mixed 데이터는 열지 않고 train lookup 표에서만 질의한다.
캐시 hit도 학습 측정으로 세고, episode마다의 기준 측정은 따로 집계한다.
마진 보조 학습이나 새 보상은 넣지 않는다.

| | IQL / TD3+BC | SAC |
|---|---|---|
| 데이터 | 고정 mixed 1,536전이 | 빈 replay, train 조건만 질의 |
| 정책 | 결정적 tanh / advantage 가중 MSE 또는 TD3 | 스쿼시 가우시안, 평가는 \(\tanh(\mu)\) |
| 50k의 의미 | 경사 업데이트 50k | 환경 step 50k, step당 업데이트 1회 |

50k 환경 step은 50k offline 업데이트와 같은 비용이 아니다. SAC가 좋아져도
우선 결론은 “추가 상호작용을 허용했을 때 달라졌다”이다.

```bash
python experiments/train_sac_online.py \
  --source campaigns/20260914_quality_s0-2/d2d_qualities/mixed/results/coupled_v1 \
  --iql-execution campaigns/20260915_iql_terminal_s0-2/execution/summary.json \
  --iql-tau09-execution campaigns/20260915_iql_tau09_beta10_s0-2/execution/summary.json \
  --td3bc-execution campaigns/20260915_reward_s0-2/execution/summary.json \
  --output campaigns/20260915_sac_online_s0-2
python campaigns/20260915_sac_online_s0-2/train_sac_online.py \
  --output campaigns/20260915_sac_online_s0-2 --execute
```

학습 중 성공이 거의 없으면 온라인 탐색으로도 희소한 성공을 못 찾은 것이다.
train 조건에서는 되고 validation에서 실패하면 조건 간 일반화를 먼저 본다.
[SAC 원논문](https://proceedings.mlr.press/v80/haarnoja18b.html)
