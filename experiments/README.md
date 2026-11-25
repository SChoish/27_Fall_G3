# 품질별 ADC / D2D 본학습

2026-09-14 본학습·독립 평가를 완료했습니다(실행 약 57분).
[전체 결과](../campaigns/20260914_quality_s0-2/REPORT.md),
[선택 체크포인트](../campaigns/20260914_quality_s0-2/MODEL_INDEX.md),
[비교 그림](../campaigns/20260914_quality_s0-2/figures/quality_comparison.pdf),
[실측 비용](../campaigns/20260914_quality_s0-2/figures/costs.json)을 확인할 수 있습니다.
2026-09-15 [검토 반영 및 실패 진단](../campaigns/20260914_quality_s0-2/REVIEW_ADDENDUM.md)을 추가했습니다.
`analyze_campaign_failures.py`는 기존 실행 소스와 파형 캐시만 읽어 버전·ADC 입력 민감도·D2D 달성 가능 조건의 궤적을 분석합니다. 새 SPICE 실행이나 모델 학습은 하지 않습니다.

`20260914_quality_s0-2` 캠페인은 작은 실행 테스트와 별도로 데이터를 새로 수집하고,
low / medium / high / mixed에서 각 3개 seed를 학습·평가합니다.
기존 체크포인트를 이어 학습하지 않습니다.

## 실제 offline 수집 비용

| 항목 | 원천 수집 | 각 품질의 학습 데이터 | 공통 평가 |
|---|---|---|---|
| ADC nominal | 설계 512개 / transient 512회 | 119개 설계 | validation 77개 / test 77개 |
| ADC robust | 설계 512개 / 4개 조건, transient 2,048회 | 119개 설계 | validation 77개 / test 77개 |
| D2D | 24개 학습 조건 × 12 episode × 32 step = 9,216개 전이 + 초기 측정 | 96 episode / 3,072개 전이 | 새 validation 12조건 / test 32조건 |

ADC는 문서화된 세 기준 설계, 그 주변의 지역 샘플 25%, 나머지 Latin hypercube로
원천 풀을 만듭니다. 데이터 분할을 먼저 고정하고 학습 부분의 실측 제약 우선 비용만으로
품질을 나눕니다. 각 유형의 validation/test 설계는 같습니다.

D2D는 random / coordinate / perturbed coordinate로 원천 episode를 수집합니다.
각 학습 조건 안에서 episode 전체의 평균 실측 utility로 하위·중위·상위 네 episode를
선택합니다. mixed는 각 구간의 episode를 균형 있게 재사용합니다. 입력 조건 난이도가
데이터 품질 구분과 혼동되지 않도록 모든 유형에 같은 학습 조건이 포함됩니다.

**high는 해당 유료 수집 풀에서의 상대적 품질이며, 전문가 최적해나 사양 충족을
보장하는 이름이 아닙니다.** high를 선별할 때 버린 샘플도 원천 수집 비용에 포함합니다.
유형별 학습 크기와 원천 수집 비용을 구분하고, mixed 생성 비용을 중복 합산하지 않습니다.
validation 튜닝 및 test 측정 비용도 최종 보고서에 별도로 남깁니다.

## 학습과 평가

- ADC: 회귀/RNC, 유형별 3개 seed, 500 epoch 중 validation으로 checkpoint 선택.
  attention learned + fixed step을 기본 비교하며, mixed RNC에는 attention/Armijo
  조합도 비교합니다. 탐색 예산은 공통 초기 8개를 포함한 128회입니다.
  random/GA와 학습 데이터 최선 설계 lookup을 포함합니다.
- ADC 모든 반환 설계: 촘촘한 시간 간격과 사전 고정한 별도 입력 세 가지로 확인합니다.
- D2D: BC/TD3+BC, 유형별 3개 seed, 5k/15k/50k checkpoint를 validation에서 선택합니다.
  TD3+BC alpha .1/2.5와 고전 방법의 기존 validation 후보 목록을 고정합니다.
- D2D 모든 유형의 test 설정을 먼저 고정한 뒤 독립 세 패킷의 전체 격자와 반환 설정을
  평가합니다. 원파형 캐시는 공유하지만 정책의 논리적 측정 비용은 별도로 셉니다.
- 서로 다른 품질의 ADC 정규화 MSE는 비교하지 않습니다. 공통 test 설계에서 복원된
  log10 오차/에너지 MAE를 함께 보고합니다.

## 파일과 재개

`train_campaign.py`는 새 폴더에 소스·프로토콜을 동결하고 두 파이프라인을 실행합니다.
`campaign_worker.py`는 원천 수집, 품질 구분과 실제 평가 작업자이고,
`campaign_report.py`는 완료한 실행만 집계합니다.

현재 실행 위치:

```text
campaigns/20260914_quality_s0-2/
  protocol.json          고정한 데이터/학습/평가 규칙
  status.json            단계 상태와 실행 명령·완료 출력 해시
  events.jsonl           단계별 이벤트
  logs/                  실제 수집·학습·탐색 로그
  adc_driver/            새 ADC 데이터, 모델과 설계 탐색
  d2d_qualities/         품질별 D2D 데이터와 모델
  execution/             시간·CPU 상한을 둔 실행의 manifest와 최종 결과
  REPORT.md              모든 학습·평가가 끝나면 생성
```

새 캠페인을 시작할 때 실행 파일을 직접 지정할 수 있습니다.

```bash
NGSPICE_BIN=/absolute/path/to/ngspice python experiments/train_campaign.py \
  --output campaigns/new_quality_run
```

선택 순서는 **사용자 지정 `NGSPICE_BIN` → `PATH`의 ngspice → 실행 가능한 번들 파일**입니다.
번들이 없는 새 clone에서도 시스템 설치를 사용합니다. 명시한 경로가 잘못되면 이를 알리고
중단하며, 다른 실행 파일로 조용히 바꾸지 않습니다. 선택한 절대 경로와 실행 파일 해시는
`execution_environment.json`에 저장합니다.

이 수정으로 새로 만든 캠페인은 동일한 동결 코드로 재개할 수 있습니다.

```bash
python campaigns/new_quality_run/train_campaign.py \
  --output campaigns/new_quality_run --resume
```

- `source_hashes.json`은 ADC의 `configs/stress.json` 등 전체 조건 파일, 프로토콜,
  회로 모델, 소스와 번들을 고정합니다. D2D 품질별 복사본도 원본과 일치해야 합니다.
- `status.json`은 작업별 명령·작업 디렉터리·입력 파일/디렉터리의 내용 해시·출력 목록과
  완료 출력 해시를 기록합니다. 학습 데이터, 체크포인트, 출처 sidecar와 평가 입력을 포함합니다.
- 재개 전 기존 작업 전체의 입력과 완료 출력을 검사하고, 각 작업을 호출할 때 명령까지
  비교합니다. 실행 중 입력이 바뀐 작업도 완료 처리하지 않습니다.
- 일치하는 완료 작업은 건너뛰고, 미완료 작업은 같은 계약일 때만 재시도합니다.
  ADC 수집 journal과 D2D 파형 캐시를 재사용할 수 있지만, 모델 학습은 작업 처음부터
  재시작합니다.
- 설정·명령·입력·출력·실행 환경이 달라지면 원인을 알리고 중단합니다. 변경한 실험은
  **새 캠페인 폴더**에서 시작하세요. 검사 기록이 부족한 구버전 캠페인은 새 실행기로
  재개하지 않습니다. 기존 완료 결과와 동결된 소스는 보존합니다.

회귀 테스트:

```bash
python -m unittest discover -s experiments/tests -v
```

현재 실행은 bounded runner로 전체 6시간, CPU affinity 12개, 수치 라이브러리 thread 1개를
설정했습니다. ngspice 작업 최대 8개를 병행하며 새 작업 시작 전 여유 공간 6GiB를
확인합니다. 상한 적용 명령과 PID는 캠페인 폴더의 `launcher.json`에 있습니다.

## D2D reward 비교 — 2026-09-15

[보상 설계와 검증](../d2d_coupled_calibration/REWARD_DESIGN.md),
[완료 결과](../campaigns/20260915_reward_s0-2/execution/REPORT.md),
[상세 수치](../campaigns/20260915_reward_s0-2/execution/summary.json).

기존 보상 / 종료 성공 / 종료 성공 + potential을 TD3+BC에서 각 3개 seed로
50,000 step 학습했다. 공통 mixed 데이터의 첫 16회 전이 1,536개를 사용했고,
보상 외 입력 배열은 일치한다. 기존 validation 12조건의 calibration 패킷에서만
평가했다. 9개 학습과 27개 checkpoint 평가에 약 294초가 걸렸으며 추가 SPICE는 0회다.

사전 고정한 50k에서는 세 보상 모두 seed별 성공 0/12였다. 보상 수정으로 성능이
개선됐다는 근거는 얻지 못했다. 15k에는 terminal-only가 [1, 1, 0]/12,
potential이 [0, 1, 0]/12였지만 이를 주 결과로 바꿔 선택하지 않는다.
학습 endpoint 성공은 2/96 episode이며, 제한된 성공 예제와 학습 경과의 변동을
함께 점검할 근거다. 이 관찰만으로 실패 원인을 확정하지 않는다.

후속 진단은 [DIAGNOSTIC.md](../campaigns/20260915_reward_s0-2/DIAGNOSTIC.md)에
있다. validation 12조건 중 10% 절감 가능은 3개(모두 4+4)이고, 좌표 탐색도
ρ=0.9에서 0/12다. 학습 성공 2/96은 16회 안에 새로 찾은 예이며 초기부터 성공은
0, 17–32회 성공 3개는 잘려 빠졌다. 50k 정책은 중복 질의 없이 탐색하지만 4+4를
가능 조건에서 거의 밟지 않아 반환 에너지 비가 ~1이 된다. 추가 50k나 성공
샘플 확대보다 이 세 숫자를 먼저 고정한다. 최종 일반화 확인은 별도로 고정한
새 평가 입력에서 수행해야 한다.

## D2D 온라인 SAC — 2026-09-15

고정 mixed 데이터 없이, 같은 관측·행동 양자화·16회 종료·종료 성공 보상으로
SAC를 train 조건 lookup 표에서만 학습한다. seed 3개 × 환경 step 50,000,
step당 업데이트 1회. 평가는 가중치 고정 후 validation에서 기준 1회 + 정책 16회다.
캐시 lookup도 학습 측정으로 세고 기준 측정은 따로 집계한다. 마진 보조나
새 보상은 넣지 않는다. 50k 환경 step을 IQL/TD3+BC 50k 업데이트와 같은
비용으로 비교하지 않는다.

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

## 제약 베이지안 탐색

기존 ADC와 D2D 회로 평가기에 `ConstrainedSearch`를 연결했다. 탐색 중 관측한
에너지와 품질 지표를 각각 예측하고, 품질 통과 가능성과 예상 에너지 개선을 함께
고려해 다음 SPICE 질의를 정한다. 시작점 8개와 실패한 시뮬레이션도 예산에 포함한다.

```bash
python -m experiments.run_constrained_search adc-multi --output campaigns/adc_constrained
python -m experiments.run_constrained_search d2d --budget 24 --output campaigns/d2d_constrained
python -m experiments.benchmark_constrained_search adc-multi --output campaigns/adc_multi_bench
```

실행에 NumPy, SciPy, ngspice가 필요하다. `adc`와 `adc-multi` 비교는 기존 GA를 위해
`pymoo`를 사용하며, 단일 입력 `adc` 비교의 RNC 기준선은 PyTorch 체크포인트를 로드한다.
비교 모드는 공통 초기점과 동일 질의 예산을 쓰고, 모든 반환점을 잠근 뒤 새 입력에서 평가한다.
재사용하는 D2D 기준선의 원본 기록과 실제 재평가 비용은 실행 결과에 남는다.

## 최악 제약 모델과 예산 내 검증 비교

`MaxConstraintSearch`는 입력별 잔차 `g_j(x)`의 최댓값 `h(x)`를 하나의 GP로
예측한다. `h(x) <= 0`은 개발 조건 전체 통과와 동치이므로, 서로 상관된
제약의 통과 점수를 임의로 합치는 단계를 제거한다. GP 확률은 모델 근사이며
새 입력에서의 통과율 보장은 아니다. 기존 `ConstrainedSearch`의 기본 동작은
유지한다. 목적함수는 기존과 같은 log-energy expected improvement다.

아래 비교는 기존/최악 제약 모델 × 별도 개발 입력 검증 유무의 네 조건이다.
검증 조건은 전체 예산에서 개별 SPICE 24회를 미리 떼어, 저에너지 후보 두 개와
측정한 제약 여유가 가장 큰 후보 한 개를 각각 새 개발 입력 8개로 검사한다.
통과 후보 중 개발 에너지가 가장 낮은 후보를 반환하며 전부 실패하면 반환하지
않는다. 검증 결과로 최종 test 후보를 다시 선택하지 않는다.

| 회로 | 검증 없는 탐색 | 검증 포함 탐색 | 조건별 개별 SPICE 예산 |
|---|---|---|---|
| ADC | 40설계 × 4입력 | 34설계 × 4입력 + 최대 3설계 × 8입력 | 160 |
| D2D | 24설계 × 4조건 | 18설계 × 4조건 + 최대 3설계 × 8조건 | 96 |

모든 조건에서 초기 8개도 예산에 포함한다. 검증용 후보가 3개 미만이면 남은
예산은 사용하지 않고 그 사실을 기록한다. 시간은 별도로 측정하며 동일 실행
시간을 주장하지 않는다. 초기 측정과 같은 탐색 prefix는 물리적으로 공유하지만,
조건별 논리적 비용은 전부 부과한다. 이 비교의 검증은 개발 데이터에 해당한다.

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.robust_search_ablation adc \
  --output campaigns/robust_adc --seeds 0 1 2
OPENBLAS_NUM_THREADS=1 python -m experiments.robust_search_ablation d2d \
  --output campaigns/robust_d2d --seeds 0 1 2
python -m experiments.report_robust_search \
  --adc campaigns/robust_adc --d2d campaigns/robust_d2d \
  --output implementation_results/robust_ablation
python -m unittest experiments.tests.test_robust_constrained_search -v
```

프로토콜은 개발/검증/test 입력, 실행 파일과 코드 해시, 라이브러리 버전을
실행 전에 기록한다. 모든 반환 설계를 잠근 후에만 촘촘한 시간 격자의 새
test 파형을 평가한다. D2D는 기존 채널의 새 무작위 패킷과 새 채널의 새 패킷을
따로 평가한다. 고정된 `same`/`runs` 패턴은 seed만 바꿔 새 패킷이라고 부르지
않으며 여기서는 검증에만 쓴다. 보고 도구는 소스 동결, 선택 잠금, 예산 및
prefix만을 사용한 선택 규칙을 확인한 뒤 집계한다.

네 조건을 비교하지 않고 검증을 포함한 한 조건만 실행하려면 다음을 사용한다.
이 실행기는 실제 탐색을 ADC 34점 / D2D 18점에서 중단하고 검증 예산을 사용한다.

```bash
python -m experiments.run_validated_search adc --output campaigns/adc_validated
python -m experiments.run_validated_search d2d --validation-bits 128 \
  --output campaigns/d2d_validated_long
```

`--method max_constraint`로 실험적 GP 대안을 선택할 수 있다. 기본은 기존 BO다.
D2D의 `--validation-bits` 기본값은 후속 실험을 반영한 128이다. 1단계의 짧은 검증을
재현할 때는 64를 명시한다. 탐색 패킷 길이는 64로 유지한다. SPICE 호출 수는 같아도
검증에서 모사한 비트 수와 실행 시간은 증가한다.
검증 통과는 미관측 패킷 전체의 안전 보장이 아니다.

[2026-11-25 실패 분석·추가 실험 결과](../implementation_results/20261125_robust_ablation/REPORT_KO.md)
에서 네 조건 비교, 동일 패킷 prefix 대조 및 독립 후속 시험을 확인할 수 있다.

```bash
python -m experiments.diagnose_d2d_packets --campaign campaigns/robust_d2d \
  --output campaigns/packet_diagnostic
python -m experiments.refine_d2d_validation --campaign campaigns/robust_d2d \
  --output campaigns/long_validation
```
