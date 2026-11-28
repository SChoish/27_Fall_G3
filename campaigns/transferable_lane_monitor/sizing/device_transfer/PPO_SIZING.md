# 회로 사이징에서의 PPO 후보 선택

**모델 용어.** MLP + BO의 MLP는 설계별 EDP와 네 전기 제약 지표를 예측하는 **SPICE surrogate**다. PPO의 **value critic**은 탐색 상태에서 기대하는 누적 보상을 예측한다. 아래 표의 PPO + value critic buffer는 과거 상태가치 모델을 보관한다.

2026-11-28. 학습 전에 평가 조건을 고정했다. 주 비교는 타깃 EDP 5% 절감까지 정밀 검증을 마친 최초 SPICE 횟수와 성공 수다. 10%도 각각 독립 추적한다. SKY130 원천 → 기존 GF180 결합·분기(seed 81–83), GF180 원천 → 기존 SKY130 2출력·3출력 분기(seed 111–113)에 동일한 방법을 적용한다. 기존 방법 개발에서 사용했던 네 타깃이며, 이번 정책 가중치·하이퍼파라미터 선택에 타깃 실행 결과는 쓰지 않는다.

소스 회로당 25개 설계 × 2개 개발 코너, 네 회로 총 200 SPICE를 쓴다. 소스 표를 실제 측정 결과로 응답하는 순차 선택 환경으로 구성한다. 매 에피소드에서 nominal 및 무작위 4개 설계를 먼저 관측한 후 남은 항목 중 하나를 고른다. 실제로 선택한 설계의 측정값으로 이력이 늘어나고 GP가 갱신된다. 성공은 개발 코너의 EDP와 네 제약을 만족할 때다. 질의마다 −0.05, 최초 성공에 +1, 추가 질의 최대 16회다. 이미 웜업에서 성공한 에피소드에는 정책 업데이트가 없다. 각 에피소드의 MLP prior는 그 회로를 제외한 세 소스 회로로 학습해 해당 회로의 미측정 라벨이 예측기로 새어 들어가는 것을 막는다. 학습 표에는 확인·정밀 결과가 없다.

정책은 후보별 BO 점수에 선형 residual logit을 더하는 categorical actor다. 후보 특징은 대리모델의 EDP 목표 대비 예측, 네 제약 여유, GP 불확실성 및 상태와의 상호작용이다. 초기 residual은 0으로 BO와 greedy 선택이 같다. 학습 중에는 정책에서 표본을 뽑고 성공 또는 종료까지 discounted return을 계산한다. 직전 상태가치 baseline을 빼고 표준화한 advantage로 PPO clipped objective(0.2)를 4회 업데이트한다. 업데이트 32번 × 24에피소드, Adam 0.025다. value critic은 상태 특징에 대한 ridge 회귀다. 같은 코드·모델의 γ=1(PPO)과 γ=0(즉시보상 bandit 대조)을 비교한다. 소스 PDK별 학습 seed는 하나로 고정했다.

타깃에서는 네 소스 회로 전부로 다시 피팅한 같은 MLP, 같은 잔차 GP를 사용한다. 네 랜덤 웜업 후 매번 국소 256 + 전역 256개 후보를 생성한다. 동결한 actor가 예측 점수가 가장 높은 폭을 선택한다. 기준선은 같은 이력에서 같은 후보와 같은 GP를 받는 기존 MLP + BO다. 기준선은 EDP와 제약을 연속 점수화하는 기존 규칙이며 PPO와 bandit은 목표 난이도도 입력받는다. 타깃 확인 결과를 정책에 주지 않는다. 기준·확인·정밀 비용을 포함하고 미달은 100회로 계산한다. 소스 표 재조회는 추가 transient가 없다. 이 실험은 대리모델을 쓰는 RL 획득함수와 기존 BO 규칙의 비교이며, 소자 폭을 직접 출력하는 연속제어 SAC/TD3나 20-lane 측정 정책 평가와 별개다.

## 이전 PPO value critic을 재선택하는 비교

단일 value critic PPO와 같은 actor, 입력, 소스 예산, 학습 seed를 쓴다. 직전 네 업데이트의 ridge value critic을 보관한다. 현재 on-policy 에피소드를 두 묶음으로 나누고, 반대 묶음의 관측 return에 대한 평균제곱오차가 가장 작은 과거 value critic으로 각 묶음의 advantage를 계산한다. 최신 value critic도 선택 가능하며 오차 동률이면 최신 것을 쓴다. actor의 업데이트 횟수, 다음 value critic의 피팅, 타깃 평가 방식은 단일 value critic과 같다. 이전 value critic 선택 빈도 및 네 타깃에 걸친 도달 시간을 함께 보고한다. 용량은 4로 고정한다. 이 buffer는 정책경사 baseline을 선택하며, 설계의 측정값을 예측하는 SPICE surrogate buffer와는 역할이 다르다.

## 대리모델 없는 PPO 대조

512개 폭 후보와 공통 웜업까지 고정하고 MLP·GP·BO 점수를 모두 제거한다. 후보의 로그 폭 평균·표준편차, 현재 측정 최선점과의 거리 및 이동량, 이미 관측한 설계와의 최근접 거리, 관측 이력의 잔여 목표 오차·측정 횟수·목표와의 상호작용으로 10차원 actor를 학습한다. 소자 순서를 바꾸어도 점수는 같다. 확률정책 초기값은 균등하다. 같은 두 소스 표, 보상, PPO 업데이트, 단일 value critic과 타깃 argmax 배포를 쓴다. 공통 국소·전역 후보 생성기 위에서 폭을 고르는 독립 정책이다. 연속 폭 벡터를 직접 생성하는 actor는 이 비교에 포함되지 않는다.

## 두 공정 방향에서 관측한 결과

각 셀은 정밀 검증 성공 수와 100회 제한평균 SPICE 비용이다. 네 타깃 × 세 시작 seed × 두 목표 × 다섯 정책으로 120개 실행을 비교했다. 이 중 기준선 24개는 앞선 실험 기록이고 나머지 96개는 이번 정책들의 SPICE 궤적이다. 단독 PPO는 관측 이력과 폭 좌표로 공통 후보를 고른다. 아래의 PPO, PPO + value critic buffer 및 bandit은 MLP + 잔차 GP의 후보 점수를 보정한다.

| 소스 → 타깃 | EDP 절감 | MLP + BO | 단독 PPO | 즉시보상 bandit | PPO | PPO + value critic buffer |
|---|---:|---:|---:|---:|---:|---:|
| SKY130 → GF180 | 5% | 6/6 · 45.33 | 6/6 · 61.33 | 5/6 · 78.33 | 2/6 · 93.67 | 4/6 · 63.00 |
| SKY130 → GF180 | 10% | 4/6 · 84.67 | 0/6 · 100.00 | 1/6 · 95.33 | 0/6 · 100.00 | 1/6 · 96.00 |
| GF180 → SKY130 | 5% | 3/6 · 88.00 | 0/6 · 100.00 | 0/6 · 100.00 | 1/6 · 93.67 | 3/6 · 87.67 |
| GF180 → SKY130 | 10% | 0/6 · 100.00 | 0/6 · 100.00 | 0/6 · 100.00 | 0/6 · 100.00 | 0/6 · 100.00 |
| 전체 | 5% | 9/12 · 66.67 | 6/12 · 80.67 | 5/12 · 89.17 | 3/12 · 93.67 | 7/12 · 75.33 |
| 전체 | 10% | 4/12 · 92.33 | 0/12 · 100.00 | 1/12 · 97.67 | 0/12 · 100.00 | 1/12 · 98.00 |

단일 PPO와 buffer의 동일 회로·seed별 비교에서 제한비용이 감소·동률·증가한 횟수

- 5% 목표에서 7 · 4 · 1

- 10% 목표에서 1 · 11 · 0

동일 시작점에서 MLP + BO와 비교한 5% 제한비용 (감소 · 동률 · 증가)

- 단독 PPO는 2 · 3 · 7

- 즉시보상 bandit는 0 · 3 · 9

- PPO는 1 · 2 · 9

- PPO + value critic buffer는 5 · 2 · 5

## value critic 선택 기록

- SKY 소스에서 최신 · 1회 전 · 2회 전 · 3회 전 선택은 [19, 16, 12, 17]회. 이전 value critic 재사용은 45/64회.

- GF 소스에서 최신 · 1회 전 · 2회 전 · 3회 전 선택은 [16, 19, 15, 14]회. 이전 value critic 재사용은 48/64회.

## 같은 이력에서 첫 정책 제안의 개발 코너 통과 수

기준점 및 네 웜업 뒤 실제로 다음 후보를 제안한 대응 사례만 센다. 이후 궤적은 알고리즘마다 달라진다.

| 목표 | 비교 수 | MLP + BO | 단독 PPO | 즉시보상 bandit | PPO | PPO + value critic buffer |
|---|---:|---:|---:|---:|---:|---:|
| 5% | 12 | 0 | 0 | 0 | 0 | 0 |
| 10% | 12 | 0 | 0 | 0 | 0 | 0 |

5% 목표의 웜업 직후 선택이 전역 후보 256개에서 나온 횟수 (나머지는 국소 후보)

- SKY130 → GF180에서 MLP + BO 2/6, 단독 PPO 0/6, 즉시보상 bandit 5/6, PPO 0/6, PPO + value critic buffer 4/6

- GF180 → SKY130에서 MLP + BO 3/6, 단독 PPO 6/6, 즉시보상 bandit 4/6, PPO 4/6, PPO + value critic buffer 1/6

## 개별 최초 정밀 통과 SPICE 횟수

| 타깃 | seed | EDP 절감 | MLP + BO | 단독 PPO | 즉시보상 bandit | PPO | PPO + value critic buffer |
|---|---:|---:|---:|---:|---:|---:|---:|
| adapt_branch_gf | 81 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| adapt_branch_gf | 81 | 5% | 30 | 44 | 64 | 미달 | 미달 |
| adapt_branch_gf | 82 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| adapt_branch_gf | 82 | 5% | 52 | 78 | 미달 | 미달 | 38 |
| adapt_branch_gf | 83 | 10% | 94 | 미달 | 미달 | 미달 | 미달 |
| adapt_branch_gf | 83 | 5% | 50 | 48 | 80 | 미달 | 미달 |
| adapt_lane_gf | 81 | 10% | 46 | 미달 | 미달 | 미달 | 76 |
| adapt_lane_gf | 81 | 5% | 38 | 80 | 78 | 미달 | 62 |
| adapt_lane_gf | 82 | 10% | 92 | 미달 | 미달 | 미달 | 미달 |
| adapt_lane_gf | 82 | 5% | 50 | 46 | 86 | 86 | 36 |
| adapt_lane_gf | 83 | 10% | 76 | 미달 | 72 | 미달 | 미달 |
| adapt_lane_gf | 83 | 5% | 52 | 72 | 62 | 76 | 42 |
| reverse_dual_sky | 111 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_dual_sky | 111 | 5% | 미달 | 미달 | 미달 | 미달 | 88 |
| reverse_dual_sky | 112 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_dual_sky | 112 | 5% | 미달 | 미달 | 미달 | 62 | 미달 |
| reverse_dual_sky | 113 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_dual_sky | 113 | 5% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_split_sky | 111 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_split_sky | 111 | 5% | 84 | 미달 | 미달 | 미달 | 54 |
| reverse_split_sky | 112 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_split_sky | 112 | 5% | 66 | 미달 | 미달 | 미달 | 미달 |
| reverse_split_sky | 113 | 10% | 미달 | 미달 | 미달 | 미달 | 미달 |
| reverse_split_sky | 113 | 5% | 78 | 미달 | 미달 | 미달 | 84 |

## 측정 비용

타깃 독립 실행 기준 과금 합계는 10722회다. 기존 기준선은 1908회이며 원자료를 재사용했다. 공통 기준 및 같은 설계·코너의 기존 측정도 각 정책의 독립 비용에 포함했다.

- adapt_lane_gf에서 같은 seed·목표의 BO와 한 번이라도 다른 폭을 선택한 횟수 (6쌍) standalone 6, bandit 6, ppo 6, ppo_buffer 6

- adapt_branch_gf에서 같은 seed·목표의 BO와 한 번이라도 다른 폭을 선택한 횟수 (6쌍) standalone 6, bandit 6, ppo 6, ppo_buffer 6

- reverse_split_sky에서 같은 seed·목표의 BO와 한 번이라도 다른 폭을 선택한 횟수 (6쌍) standalone 6, bandit 6, ppo 6, ppo_buffer 6

- reverse_dual_sky에서 같은 seed·목표의 BO와 한 번이라도 다른 폭을 선택한 횟수 (6쌍) standalone 6, bandit 6, ppo 6, ppo_buffer 6


## 해석 및 재현

5%에서 MLP + BO가 네 회로 합산 9/12 · 66.67회다. 단독 PPO는 SKY130 → GF180에서 6/6이지만 GF180 → SKY130에서 0/6이었다. PPO에 이전 value critic을 다시 선택하도록 하면 5%에서 단일 value critic 대비 평균 18.33회 줄고 성공이 3/12에서 7/12로 늘었다. 대응 실행 12개 중 7개는 빨라지고 1개는 늦어졌다. MLP + BO와 비교하면 이 buffer 정책은 5개에서 빨랐고 5개에서 느렸으며 2개에서 동률이었다. 10%에서는 MLP + BO 4/12, buffer 정책 1/12였다. 네 타깃과 PDK별 정책 학습 seed 하나에서 얻은 기술통계다.

소스의 non-nominal 설계 96개 중 개발 코너 5% 통과는 SKY130 3개, GF180 4개였으며, 10%는 각각 1개와 2개였다. 768개 소스 에피소드가 이 표를 재사용한다. 학습 보상은 개발 코너 기준이며 타깃 성적은 정밀 확인까지의 비용이다. 정책 학습과 배포 시의 후보 개수도 다르다. 이 자료로 판단할 수 있는 범위는 이 후보 생성기·보상·소스 데이터에서 학습한 정책이다.

같은 시작점에서 MLP + BO와 buffer 학습 정책이 이긴 사례가 각각 다섯 개여서 상황별 선택을 연구할 근거가 있다. 다음 비교에서는 BO의 원래 제안과 buffer로 학습한 정책의 제안 사이만 선택하게 하고, 같은 관측 이력이 주어진 고정 규칙·무작위 혼합과 대조할 수 있다. 어떤 상태에서 전환할지 소스에서 학습하고 미사용 타깃에서 검증해야 한다. 여기서 value critic 선택은 반대 에피소드 묶음에서 측정한 return 예측오차를 사용했다. value critic 선택기 자체를 RL로 학습한 결과는 아직 없다.

[기준선 및 잔차 PPO·bandit 측정](ppo_sizing_measurements.zip) · [value critic buffer 측정](ppo_critic_buffer_measurements.zip) · [단독 PPO 측정](ppo_standalone_measurements.zip)

```sh
python ppo_sizing.py --replay
python ppo_critic_buffer.py --replay
python ppo_standalone.py --replay
python ppo_sizing_report.py
```
