# D2D: 고정 예산 성공 목적과 reward 비교

현재 학습기는 `--reward-mode`로 세 보상을 비교한다. 옵션을 생략하면 기존 학습 경로를
사용한다. 기존 Environment와 원천 데이터는 보존하며, TD3+BC 입력에서 보상을 다시
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
