# ADC 2차 실험: 입력 다양성, 오차 여유, critic 재사용

목표는 **여러 입력에서 최대 hold 오차 1 mV를 만족하는 저전력 설계**다.
회로 topology, 스위치 크기, 기본 부하 15 pF와 acquisition 200 ns는 유지한다.
탐색은 입력별 평균 공급 에너지의 평균을 최소화하고, 입력 전체의 최대 hold 오차를
설계 한도로 제약한다. 실패는 오차 위반량, 성공은 에너지로 비교한다.

| 조건 | 탐색 입력 | 설계 한도 | 최종 테스트 한도 |
|---|---|---:|---:|
| A | 기존 1종 | 1.0 mV | 1.0 mV |
| B | 기존 1종 | 0.8 mV | 1.0 mV |
| C | 기존 + 이전 독립 입력 3종 | 0.8 mV | 1.0 mV |

사용자 요청 본문에 생략된 A/B/C 표를 위와 같이 해석했다. A→B는 여유,
B→C는 입력 다양성 효과다. 0.8 mV가 일반화를 보장하지 않는다.
C는 기존 `configs/stress.json`의 부하·속도·온도 조합과 다른 실험이다.

`round2.py init`은 이전 mixed의 설계점·품질 멤버십·train/validation/test 분할을
보존한다. A/B는 저장된 물리 측정값을 재채점한다. C는 같은 설계점에 개발 입력
3종을 추가 측정한다. 현재 mixed는 학습 119개와 검증/테스트 154개를 포함한
273개 설계점이므로 C의 추가 비용은 **819 transient**다. 최초 source pool의
512회 수집 비용은 공유 비용으로 한 번 기록한다.

기존 교대 입력과 seed 271091/271092 입력은 개발용이다. 최종 입력은 같은
0.8–1.1 V 범위에서 생성한 별도 고정 seed 8개이며 `protocol.json`에 실제 전압을
저장한다. 최종 입력으로 설정을 선택하거나 재학습하지 않는다.

## 실행 순서

저장소 최상위에서 실행한다. `python`은 requirements-learning과 pymoo가 설치된
환경을 사용하고, 필요한 경우 `NGSPICE_BIN`을 지정한다.

```bash
python adc_driver/round2.py init \
  --source campaigns/20260914_quality_s0-2/adc_driver/data/nominal/mixed.json \
  --output campaigns/my_adc_round2
python adc_driver/round2.py diagnose --output campaigns/my_adc_round2 --workers 4
python adc_driver/round2.py prepare-c --output campaigns/my_adc_round2 --workers 4
python adc_driver/round2.py compare --output campaigns/my_adc_round2
python adc_driver/round2.py attention --output campaigns/my_adc_round2
python adc_driver/round2.py test --output campaigns/my_adc_round2
```

`init`의 기본값은 seed 0/1/2, offline 500 epoch, online critic 20 epoch/update,
**각 run 실제 탐색 transient 512회**다. 따라서 A/B는 512 설계, C는 128 설계다.
같은 조건의 GA·회귀·RNC·critic 3모드는 입력 묶음과 실제 SPICE 예산이 같다.
작은 통합 검증에는 init에서 `--spice-budget 64 --epochs 10 --critic-epochs 1 --seeds 0`을
쓸 수 있으나 본 실험과 별도 폴더에 둔다. 예산은 64 이상, 32의 배수여야 한다.

진단은 기준 설계와 로그 정규화 공간의 고정 주변 설계 32개, 입력 4종을 측정한다.
별도로 기준 설계의 30 pF, 100 ns 조건도 측정한다. 총 134 transient다.
`diagnostic_waveforms/`와 `diagnostic.json`에는 다음이 저장된다.

- acquisition 종료 직전 signed 오차 및 최대 절댓값
- turn-off 전후 signed 전압 변화 및 최대 절댓값
- read 시점 저장값 대비 hold 중 최대 추가 변화 `hold_drift_mv`
- hold 전체의 목표 입력 대비 최대 오차와 공급 에너지

turn-off 변화는 두 시점의 전압 차이이며 순수 charge injection 분리가 아니다.
세 최대값은 다른 cycle에서 나올 수 있으므로 최악 cycle도 같이 읽어야 한다.

C의 0.8 mV를 만족한 진단 설계가 없으면 `compare`는 중단한다. 이는 이 작은
탐색의 실패이지 사양 불가능성 증명이 아니다. 확보된 설계는 **모든 방법·조건의
동일한 초기 8개 중 첫 무작위 설계를 대체**하고 다시 SPICE 비용을 지불한다.
따라서 warm-start 실험이며, 사양을 처음부터 찾아냈다고 해석하지 않는다.

## Critic 비교

| CLI 모드 | 학습 동작 |
|---|---|
| `frozen` (기본값) | 기존 고정 모델 + 실측 residual ridge |
| `latest` | 최신 전체 critic을 이어 학습 + 같은 residual ridge |
| `buffer` | 최대 4개 전체 critic에서 선택해 이어 학습 + 같은 residual ridge |

회귀/RNC와 critic 3모드 모두 attention learned, line search none을 고정한다.
GA도 동일 입력·초기 설계·SPICE 예산을 사용한다. 주 비교 완료 후 개발 입력의
1 mV 동시 성공률, 통과 에너지, 고정 이름 순서로 가장 유망한 학습 설정을 정하고
그 설정에서만 attention off를 추가 실행한다. 모든 반환 설계를 고정한 후
최종 테스트를 실행한다.

독립 실행 예시:

```bash
python adc_driver/run.py optimize --method rnc \
  --checkpoint campaigns/my_adc_round2/checkpoints/C_rnc_s0.pt \
  --contexts my_C_contexts.json --budget 128 --output results/my_buffer \
  --critic-mode buffer --source-data campaigns/my_adc_round2/data_C.json \
  --critic-capacity 4 --critic-epochs 20 --critic-temperature 0.25
```

`--source-data`는 checkpoint가 학습한 파일과 SHA256이 일치해야 한다. 온라인
replay에는 그 파일의 split 0과 지금까지 새로 지불한 탐색 데이터만 들어간다.
각 epoch에서 offline/online을 같은 개수로 섞는다. 두 집합의 크기가 다르면
작은 쪽을 복원 추출한다. encoder, regressor, validity classifier, RNC projector를
함께 재사용하고 갱신한다. 정규화는 offline train 통계를 계속 유지한다.
RNC projector는 RNC 모드에서만 contrastive loss로 갱신한다.

모델 버전마다 SPICE **이전**에 오차·에너지 log 예측과 validity 확률을 저장한다.
`pending_prediction.json`을 먼저 쓴 뒤 oracle을 호출하고, 완료 후 `trace.json`의
`critic_predictions`와 `critics.pt`의 evidence로 옮긴다. 잔차 보정 전 각 전체
critic 자체의 예측을 평가하므로 모델 버전별 점수의 의미가 고정된다.
후보 생성에 실제 사용한 잔차 보정 비용은 기존 proposal에 별도 기록한다.

점수는 최근 최대 32점의 정규화된 log-target MSE + 0.2 × validity Brier error다.
최근 query일수록, 현재 최선 설계 근처일수록 가중치를 높인다. 실제 제안으로 얻은
개선은 작고 시간에 따라 감소하는 보조 점수로만 반영한다. 선택은
`softmax(-score / temperature)`다. 점수가 낮을수록 선택 확률이 높다.
새 child에는 학습 이전 evidence를 복사하지 않으며, 아직 예측 기록이 없을 때만
부모의 당시 점수를 prior로 쓴다. 버퍼가 차면 새 child는 보존하고 기존 모델 중
현재 점수가 가장 나쁜 것을 제거한다. 이 선택·제거 규칙은 사전에 정한 구현 선택이며
ALARM 원문의 식을 그대로 재현했다는 주장이 아니다.

`critics.pt`에는 전체 state_dict, 정규화, 생성 query, 탐색 위치, 부모 ID,
실측 예측 오차·개선 기록과 선택 event를 저장한다. optimizer는 매 갱신 새로 만든다.
critic을 고정한 뒤 attention을 다시 학습하고 기존 후보 생성기를 실행한다.
actor나 추가 N² 설계 데이터 증강은 도입하지 않았다.

## 결과와 비용

`summary.json`은 각 조건/방법/critic/attention별로 다음을 보고한다.

- 새로운 입력 8종을 **모두** 통과한 반환 설계 비율
- 통과 설계들의 테스트 평균 에너지와 개별 에너지; 통과가 없으면 null
- 실제 탐색 transient, 반환 설계가 관측된 query와 그 시점까지 transient

offline 원천 수집, C 추가 입력, 진단, 온라인 탐색, 개발 확인, 최종 테스트 비용을
분리한다. `spice_attempts/`는 실행 전과 완료 후 개별 기록을 써서 중단·재시도도
추적한다. 요약에는 완료/미완료 attempt 수가 함께 나온다. 초기/실패 평가도 비용이다.

프로토콜, 원천 데이터, 코드, 모델, 생성 데이터·진단의 해시를 확인한다. 완료된
단계의 결과는 재사용하고, 부분 탐색은 비용 누락을 막기 위해 자동 재실행하지 않는다.
기존 완료 캠페인의 동결 소스·결과는 소급 수정하지 않는다.

```bash
python -m unittest discover -s adc_driver/tests -v
```

## 이번 작업에서 준비한 후속 실행

[검증 결과](ROUND2_VALIDATION.md)를 확인하세요. `campaigns/20260915_adc_round2_ready/`는
기존 C 측정값을 재사용하고 개발 과정에서 확보한 설계 주변을 진단한 폴더입니다.
34개 중 20개가 C를 통과했고 공통 초기점도 저장되어 있습니다. 다음 단계는 다음과 같습니다.

```bash
python campaigns/20260915_adc_round2_ready/execution_source/round2.py compare \
  --output campaigns/20260915_adc_round2_ready
python campaigns/20260915_adc_round2_ready/execution_source/round2.py attention \
  --output campaigns/20260915_adc_round2_ready
python campaigns/20260915_adc_round2_ready/execution_source/round2.py test \
  --output campaigns/20260915_adc_round2_ready
```

새 후속 폴더는 init의 `--prepared-c <data_C.json>`과
`--diagnostic-anchor <design.json>`으로 준비할 수 있습니다. prepared C는 원천 파일
해시, 설계점/분할, 물리 context가 일치해야 하며 추가 SPICE가 들지 않습니다.
anchor는 새로운 진단에 포함하고 그 주변을 샘플링합니다. 해당 anchor를 발견한 비용은
별도 개발 기록에서 가져와 공유 준비비에 합쳐야 합니다. 이번 작업의 전체 준비비는
`campaigns/20260915_adc_round2_ready/development_costs.json`에 기록했습니다.
그 안의 C 추가 819회는 후속 summary의 inherited C 비용과 중복 합산하지 않습니다.

실행 소스는 각 폴더의 `execution_source/`에 보존했습니다. 이전 두 진단/통합 실행은
현재 실행기보다 앞선 구현이므로 재검토할 때 해당 폴더의 동결 소스를 사용합니다.
