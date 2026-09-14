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

우선 보상 비교 옵션을 유지하고, 다음 실험에서는 성공과 근접 실패를 연결하는
위상 이동 전이의 데이터 포함 여부를 분석한다. 그 결과를 바탕으로 동일 수집 예산
안에서 데이터 구성을 조정하고 validation 선택 규칙을 미리 고정한다. 최종 일반화
확인은 별도로 고정한 새 평가 입력에서 수행해야 한다.
