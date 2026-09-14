# ADC 2차 수정 검증 — 2026-09-15

요청된 입력/여유 조건과 critic 3모드를 구현하고 실제 ngspice로 확인했다.
기존 캠페인 실행 프로세스는 작업 시작 시 없었으며, 기존 완료 결과와 D2D 변경은
수정하지 않았다. 회로 topology와 소자 연결은 유지했다.

## 결과

기준 설계의 입력별 최대 hold 오차는 0.981292, 1.430657, 1.001749,
1.157001 mV로 기존 관찰을 재현했다. 최초 기준+주변 32개에서 C(4입력, 0.8 mV)
통과는 0개였고 최선은 0.849050 mV, 199.542729 pJ/sample이었다.

기준 설계의 large_cap(30 pF), fast_acquisition(100 ns) hold 오차는 각각
18.644913, 19.091444 mV, 획득 종료 전 최대 오차는 19.883511, 20.562444 mV였다.
기본 부하에서 기준 설계의 hold 중 추가 변화량은 최대 약 0.000077 mV였다.
정착 구간을 우선 진단할 근거이며, 이 수치만으로 순수 charge injection이나 다른
원인을 분리했다고 주장하지 않는다.

기존 nominal mixed 273개 설계점에 입력 3종을 추가 측정했다. 학습 split의
119개 중 C를 통과한 것은 0개였다. 이 결과에 validation/test 설계의 라벨을
온라인 학습에 넣지 않았다. 새 최종 입력 8종은 생성·고정만 했으며 아직 평가하지 않았다.

회귀/RNC × frozen/latest/buffer의 통합 확인은 각 16개 설계 × 4입력,
seed 0, offline 50 epoch, online 1 epoch/update, attention 4 update/proposal로
실행했다. **본 비교 기본값과 다른 짧은 동작 확인**이다. 여섯 실행 모두 정상 종료했다.
latest/buffer는 각각 8번 전체 critic을 갱신했고, 버퍼는 4개를 유지하며 이전 모델을
다시 부모로 선택했다. query 전 모델별 예측과 query 후 실측 비교 기록이 저장됐다.

이 과정의 frozen RNC 실행에서 0.495930 mV, 268.936251 pJ/sample 설계를 얻었다.
나머지 다섯 실행은 초기 0.849050 mV 설계를 유지했다. 이 1-seed의 짧은 결과로
RNC 우위 또는 온라인 갱신/버퍼의 효과를 결론내리지 않는다.

확보한 0.495930 mV 설계를 중심으로 기준 설계·anchor·주변 32개를 다시 진단했다.
**34개 중 20개가 C를 통과**했다. 통과 설계 중 최저 에너지인 다음 설계를 후속
모든 조건/방법의 공통 초기점으로 지정했다.

- 4개 개발 입력의 최대 hold 오차: **0.777016 mV**
- 입력별 평균 공급 에너지의 평균: **182.719778 pJ/sample**
- 설계 파일: `campaigns/20260915_adc_round2_ready/warm_start.json`

이는 개발 입력의 유효 설계 확보 결과다. 새 입력 일반화 성능은 아직 측정하지 않았다.
본 비교는 3 seed, 실제 transient 512회/run, mixed-only, 동일 attention/line search로
준비되어 있고, 선택한 조건의 attention off 및 새 입력 테스트는 그 다음 단계다.

## 비용과 검증

| 이번 추가 작업 | 실제 transient |
|---|---:|
| 최초 진단(33×4 + 부하/속도 2) | 134 |
| 기존 설계점의 새 개발 입력(273×3) | 819 |
| critic 여섯 통합 실행(6×16×4) | 384 |
| 확보한 설계 주변 진단(34×4 + 부하/속도 2) | 138 |
| 합계 | **1,475** |

기존 공유 source pool 512회는 위 추가 비용에서 제외했다. 이를 포함한 누적 준비비는
1,987회다. 후속 폴더는 기존 C 데이터를 재사용하므로 819회를 새로 쓰지 않았다.
`development_costs.json`과 후속 결과의 inherited C 비용은 중복 합산하지 않는다.

자동 테스트 25개를 통과했다. 기존 측정/attention/제약 우선 목적함수에 더해
전체 모델 갱신, validation/test replay 배제, 모델 버전별 prequential evidence,
음의 score softmax, checkpoint 저장, margin 재채점, 실제 평가 전 예측 선기록,
실패 비용, 최종 입력 선고정과 통과 설계 에너지 집계를 확인했다.
`git diff --check`도 통과했다.

원자료:

- `campaigns/20260915_adc_round2/diagnostic.json`, `data_C.json`, `spice_attempts/`
- `campaigns/20260915_adc_critic_smoke/summary.json`, 각 실행의 `trace.json` / `critics.pt`
- `campaigns/20260915_adc_round2_ready/diagnostic.json`, `protocol.json`, `development_costs.json`

세 폴더의 `execution_source/`는 각 프로토콜에 기록된 Python 소스 해시와 일치한다.
