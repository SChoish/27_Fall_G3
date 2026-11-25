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
