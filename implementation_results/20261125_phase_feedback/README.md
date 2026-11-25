# D2D 수신 시점 보정 실험 (2026-11-25)

기존 3-lane BO seed 11–13의 회로와 이전 저에너지 버퍼 설계 하나를 고정한다. 각 채널은 3-lane 108 MOS, 합성 RC 또는 RLC이고, 전원·온도·임피던스·배선 결합·전송 속도·클록 오차를 채널별로 추출한다. 같은 채널을 모든 설계와 정책이 공유한다. 통계 단위는 독립 채널이다. 같은 채널의 네 설계와 네 패킷은 반복 측정이다.

짧은 훈련 패킷 3개(각 배선당 64비트)에서 매번 선택한 수신 위상 하나의 세 배선별 최악 정규화 전압 마진을 관측한다. 실제 수신기에 이 값을 반환할 마진 모니터가 있다고 가정한다. 기존 장치의 지원 범위 0..64 중 코드 8..56을 4 단위로 사용한다. 장착 전 오프라인 학습에는 개발 회로 파형의 모든 코드를 허용한다. 링크를 켠 뒤 보정에는 선택한 위상의 관측만 허용한다. 독립 패킷 4개 전체에서 모든 배선의 aperture margin ≥ 0.08 V/VDD, 중심 오류 0, overshoot ≤ 0.2 VDD를 요구한다. 수신 위상 변경은 수동 측정이므로 동일 회로에서 TX 에너지는 같다.

- fixed32, trained_static은 운영 중 훈련 비트가 필요 없다.
- grid, quadratic은 같은 세 코드에서 얻은 관측으로 각각 최고 측정점 또는 포물선 꼭짓점을 반환한다.
- prior_grid와 prior_adaptive는 같은 개발 채널의 참조 곡선으로 미래 패킷의 마진을 추정한다. prior_adaptive는 마진을 관측할 때마다 다음 질의 코드를 고른다.
- oracle은 독립 패킷을 열고 같은 위상 코드 하나를 모든 패킷·배선에 적용한 이론적 진단 상한이다.
- 각 채널·설계 조합마다 물리 transient 7개를 모든 방법이 공유한다. 16 개발 채널 448회, 24 시험 채널 672회. 수신기와 보정 회로 전력은 포함하지 않는다.

저장소 루트에서 시스템 ngspice를 쓰거나 NGSPICE_BIN을 설정한다. 설치한 Python 의존성은 experiments의 기존 requirements를 따른다. 다른 작업자가 같은 출력 디렉터리에 쓰고 있는지 먼저 확인한다. 완료된 조합은 실행 재개 시 재사용한다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=implementation_results/20261125_phase_feedback
python -m experiments.run_phase_feedback --output campaigns/phase_dev --root-seed 49101 --channels 16 --workers 7
python -m experiments.analyze_phase_feedback --dev campaigns/phase_dev --model campaigns/phase_model.json --result campaigns/phase_dev_scores.json
python -m experiments.run_phase_feedback --output campaigns/phase_test --root-seed 49102 --channels 24 --workers 4
python -m experiments.analyze_phase_feedback --test campaigns/phase_test --model campaigns/phase_model.json --result campaigns/phase_test_scores.json
```

개발 결과로 실패 원인을 가른다. 같은 수신 시점을 모든 배선에 썼을 때 실패하면 배선별 위상 상한을 계산한다. 여기서도 부족한 경우 overshoot와 개별 배선의 마진 부족을 분리한다. 기존 시험 성적을 읽고 수정한 정책은 별도의 새 채널 seed에서 판정한다.

## 개발 실패를 반영한 회로 크기 탐색

첫 개발 채널의 기존 BO 설계 3개는 13개의 공통 위상 코드 중 최선에서도 기준에 미달했다. 개발 16채널을 기존 bo11 회로의 사후 최적 마진으로 정렬해 네 분위의 대표 채널을 고른다. 6개 사이징 변수를 다시 탐색한다. BO, 통과·실패 설계 사이의 제약 경계를 보간하는 bracket, 기존 국소 수정 MLP를 비교한다. 동일 초기 설계 8개, 짧은 18개 × 4조건 × 64비트, 긴 3개 × 8조건 × 128비트다. 배선당 7,680비트, seed·방법당 96회다. 사전 개발 데이터 비용은 이 예산과 별도로 공개한다. 제조 전 시뮬레이션에서는 모든 위상을 읽고, 출하 후 보정에는 같은 짧은 훈련 비트만 사용한다. 개발에서 반환한 설계가 품질에 실패해도 가장 위반이 작은 후보를 진단용으로 남기며 `feasible_on_development`에 실패를 명시한다.

```bash
out=implementation_results/20261125_phase_feedback
python -m experiments.report_phase_feedback dev campaigns/phase_dev_scores.json "$out/DEV_KO.md"
python -m experiments.report_phase_feedback test campaigns/phase_test_scores.json "$out/TEST_KO.md"
for seed in 21 22 23; do
  python -m experiments.run_phase_sizing --model campaigns/phase_model.json --output "campaigns/phase_size_s$seed" --seed "$seed"
done
python -m experiments.confirm_phase_sizing --output campaigns/phase_transfer --root-seed 49103 --channels 16 --bits 512 --workers 7 campaigns/phase_size_s{21,22,23}
python -m experiments.report_phase_feedback sizing campaigns/phase_transfer/summary.json "$out/SIZING_KO.md"
```

## 2026-11-25 실행 인계

18:49 KST 시작, 21:49 KST 마감이다. 기존 scratch의 저장소에서 위 캠페인을 실행 중이며 `/tmp/g3_phase_dev.log`, `/tmp/g3_phase_test.log`, `/tmp/g3_phase_supervise.log`, `/tmp/g3_phase_size_*.log`에 로그를 남긴다. 개발 완료 → 모델 및 교차검증 동결 → 새 회로 세 seed → 512비트 완전 독립 시험 순서로 자동 진행한다. 24채널 기존 회로 평가 패킷은 별도로 생성하고 동결 후 정책에 적용한다. GitHub에서 이어받을 때 이미 올라온 DEV_KO, TEST_KO, SIZING_KO 보고서 및 캠페인 디렉터리를 먼저 확인한다. 종료된 단계의 소형 결과와 재현 명령을 같은 작업 브랜치에 보관한다. 동일한 실행이 진행 중이면 그 출력 디렉터리를 공유하는 두 번째 실행을 만들지 않는다.

각 개발·독립 평가 후 `python -m experiments.diagnose_phase_limits campaigns/phase_dev implementation_results/20261125_phase_feedback/DEV_LIMITS_KO.md`를 실행한다. 독립 평가에는 phase_test와 TEST_LIMITS_KO.md를 사용한다. 고정 위상 통과, 공통 위상으로 회복, 배선별 독립 위상으로만 회복, 한 배선 이상에서 마진/오류 미달, 레일 초과를 배타적으로 집계한다.

민감도 대조군을 재현하려면 `python -m experiments.check_phase_failure --dev campaigns/phase_dev --output implementation_results/20261125_phase_feedback`를 실행한다. 개발 채널 0과 1에서 패킷을 맞추고 결합, 비트율, 고구동 기준 설계의 효과를 각각 측정한다. `dev_model.json`이 있으면 고정된 모델로 독립 채널 평가와 다음 사이징 탐색을 재개할 수 있다.

### 512비트 독립 비교 사전 지정 (49103)

개발 실패 분석에서 공통 위상 최적화의 병목이 일부 배선의 마진으로 나타났다. 새 후보 lane_quadratic은 코드 16·32·48에서 관측한 배선별 최악 마진에 각각 이차식을 맞춘 뒤, 세 예측값의 최솟값이 가장 큰 공통 코드를 반환한다. 외삽 없이 16~48을 사용한다. 고정 크기 회로에서 grid 및 기존 quadratic과 세 패킷·192비트/배선·TX 파형을 정확히 공유한다. 배선별 클록을 요구하지 않는다. 49103의 16채널(8 RC, 8 RLC) × 네 검증 패킷에서 통과 수와 같은 채널에서의 득실을 비교한다. 위의 confirm_phase_sizing 및 report_phase_feedback sizing 명령으로 재현한다.

고정 bo11 회로에서도 설계 변경과 보정 효과를 분리한다. 보조 시험 seed 49104는 8개 채널에 128비트 패킷 네 개를 사용한다. 각 방법은 같은 세 관측을 사용하며 fixed32는 무보정 대조군이다. 새 사이징의 주 평가는 위에 지정한 49103이다.

```bash
python -m experiments.compare_lane_calibration --output campaigns/phase_lane_49104 --workers 4
python -m experiments.report_phase_costs campaigns implementation_results/20261125_phase_feedback/COSTS_KO.md
```

### 동일 탐색 기록에서 긴 검증 후보 선택 비교

개발 bracket seed 21·23에서 에너지가 가장 낮은 두 후보 모두 짧은 패킷의 제약 여유가 작았다. 첫 슬롯과 최대 여유 슬롯은 유지하고, 두 번째 슬롯은 모든 정규화 잔차 ≤ −0.25인 후보 중 가장 저렴한 별도 설계에 준다. 후보 부족 시 기존 규칙으로 돌아간다. 두 bracket 방법은 같은 짧은 18설계를 보며, BO를 포함한 세 방법 모두 동일한 검증 조건 8개 × 3설계를 지불한다. 배치 시 grid를 주 지표로 고정한다. 최종 비교는 별도 seed 49105, 12채널, 256비트 시험으로 진행한다.

```bash
for seed in 21 22 23; do
  python -m experiments.check_sizing_shortlist "campaigns/phase_size_s$seed" "campaigns/phase_reserve_s$seed"
done
python -m experiments.confirm_phase_sizing --output campaigns/phase_reserve_transfer --root-seed 49105 --channels 12 --bits 256 --workers 4 campaigns/phase_reserve_s{21,22,23}
python -m experiments.report_phase_feedback sizing campaigns/phase_reserve_transfer/summary.json implementation_results/20261125_phase_feedback/RESERVE_KO.md
```

### 결과 및 실행 상태 확인

[동결 설계와 개발 비용](SELECTION_KO.md), [고정 BO 보조 시험](LANE_49104_KO.md), [보정 패킷의 송신 에너지](DEV_ENERGY_KO.md)를 기록했다. 이 단계의 독립 시험 출력은 campaigns/phase_test (96조합), campaigns/phase_transfer (144조합), campaigns/phase_reserve_transfer (96조합)이다. 작업 공간 간 PID가 보이지 않을 수 있으므로 재시작 전 출력 파일의 최근 수정 시각과 개수 변화도 확인한다. 완료된 독립 시험만 최종 통계에 포함한다.

```bash
out=implementation_results/20261125_phase_feedback
python -m experiments.report_probe_energy campaigns/phase_dev "$out/DEV_ENERGY_KO.md"
python -m experiments.report_probe_energy campaigns/phase_test "$out/TEST_ENERGY_KO.md"
python -m experiments.report_reserve_transfer campaigns/phase_reserve_transfer/summary.json "$out/RESERVE_PAIRED_KO.md"
```

### 보정 비트를 128로 줄인 동등 비용 대조

새 채널 seed 49106의 bo13 고정 회로에서 첫 64비트 패킷은 코드 32, 두 번째 64비트는 코드 16 또는 48로 보정한다. 고정 경로는 개발 채널 leave-one-out으로 한 방향을 고르고, 적응 경로는 첫 응답을 본 뒤 각 후보의 두 번째 관측이 가져올 개발 사전모형의 기대 효용을 비교한다. 두 방법은 같은 사전모형·최종 판정·128비트·TX 파형을 사용한다. 0회 및 1회 관측도 기록한다. 12채널(6 RC, 6 RLC), 채널당 4개 × 256비트의 새 검증 패킷에서 평가한다. 모든 설계·모형·규칙은 49106 결과를 보기 전에 확정했다.

```bash
python -m experiments.check_two_probes --model implementation_results/20261125_phase_feedback/dev_model.json --output campaigns/phase_two_49106 --workers 3
```

시간 적분 확인에서는 49102의 RC/RLC별 grid 임계 통과·실패 각 1건에서 같은 설계·위상·패킷을 UI/128로 다시 계산한다. 통과율에는 원래 사전 지정한 UI/64 결과를 사용한다.

```bash
out=implementation_results/20261125_phase_feedback
python -m experiments.check_phase_resolution campaigns/phase_test_scores.json "$out/RESOLUTION_KO.md"
python -m experiments.report_two_probes campaigns/phase_two_49106/summary.json "$out/TWO_PROBES_KO.md"
python -m experiments.report_phase_pairs campaigns/phase_transfer/summary.json "$out/SIZING_PAIRED_KO.md"
python -m experiments.report_phase_pairs campaigns/phase_reserve_transfer/summary.json "$out/RESERVE_INTERVAL_KO.md"
```
