# 공개 논문 베이스라인 비교

회로 A·B, 타깃 SKY130·GF180, 균형·비대칭 부하에서 최악 EDP 5% 개선까지 필요한 SPICE를 비교한다. 기존 전기 제약과 0.25 ps 정밀 검증을 동일하게 적용한다. 1회는 설계 하나의 코너 transient 하나다. 명목 설계, 공통 무작위 초기 설계 4개, 개발·확인·정밀 측정과 실패한 코너를 모두 100회 상한에 계상한다. 총 8개 조건 × 12개 시작 seed × 6개 방법으로 576개 궤적이다.

| 실행 ID | 정책 | 재사용하는 반대편 PDK의 소스 호출 |
| --- | --- | ---: |
| `scbo` | 타깃 ARD Matérn-5/2 GP, trust region, 제약 Thompson sampling | 0 |
| `kato` | 정렬 GP와 타깃 NeukGP, 제약 MACE, 선택적 전이 | 50 |
| `dnnopt` | 타깃 측정에서 만든 쌍별 pseudo-sample, surrogate와 actor, elite 후보 제안 | 0 |
| `mlp200` | 고정 소스 MLP와 타깃 잔차 GP | 200 |
| `random_templates` | `mlp200` + 무작위 소스 폭 매핑 | 200 |
| `retrieved` | `mlp200` + 소스 elite 폭 매핑 | 200 |

소스 기록은 기존 `direct_surrogate_gate_measurements.zip`에서 읽는다. 표의 비용은 각 정책에 제공한 소스 기록의 원래 비용이다. 여러 seed와 arm이 재사용한 소스 측정을 반복 청구하지 않는다. 타깃 궤적은 각 arm이 명목점부터 따로 실행한다. 최적화기는 개발 코너 측정만 받는다.

## 논문 구현과 G3 설정

- **SCBO.** [AISTATS 2021 논문](https://proceedings.mlr.press/v130/eriksson21a.html)의 [BoTorch 순차 튜토리얼](https://botorch.org/docs/v0.14.0/tutorials/scalable_constrained_bo)을 참조해, 목적과 네 제약 각각의 결합 posterior draw로 후보를 비교한다. feasible 표본이 없으면 양의 제약 위반 합이 가장 작은 후보를 고른다. G3에서는 batch 1, 매 결정 256개 Sobol 후보, SciPy GP 적합, 공통 초기점을 사용한다.
- **KATO.** [DAC 2024 논문](https://doi.org/10.1145/3649329.3657380)의 저예산 순차 재구현이다. 소스 GP를 고정하고, 타깃 encoder·decoder를 Delta-method likelihood로 적합한다. 타깃 전용 NeukGP도 유지한다. 가중치에 따라 둘 중 하나를 뽑고, 512개 고정 크기 후보 풀의 feasibility-weighted UCB·PI·EI Pareto front에서 다음 점을 뽑는다. 새로운 개발 측정이 incumbent를 개선한 경우 선택된 모델의 가중치를 올린다. 원 논문의 NSGA-II와 병렬 batch 대신 이 순차 후보 풀을 사용한다. 소스는 반대편 PDK의 네 회로 중 폭 차원 수, 출력 수, 이름 순서로 정한 한 회로의 25개 설계다. 원 논문 제약 실험의 200개 타깃 초기 설계 대신 공통 초기점을 쓴다.
- **DNN-Opt.** [DAC 2021 논문](https://doi.org/10.1109/DAC18074.2021.9586139) Algorithm 1의 타깃 전용 재구현이다. 관측된 설계 쌍에서 `(x_i, x_j - x_i) -> y_j`를 학습한다. 매 결정에 네트워크를 다시 초기화하고, surrogate를 고정한 뒤 elite 점에서 actor의 변화량을 학습한다. 노이즈를 더한 제안을 surrogate로 순위화한다. 폭만 고정된 비교 문제라 추가 SPICE를 요구하는 민감도 기반 차원 제거는 적용하지 않는다. 네트워크 크기, 학습 스텝, 탐색 노이즈는 G3에서 고정했다.
- 세 논문 방법이 6자리로 반올림했을 때 이미 측정한 폭을 다시 내면, 그 결정의 시드로 타깃 범위에서 새 폭을 제안하고 `duplicate_fallback`을 기록한다. 지표 검증과 SPICE 비용은 공통 실행기가 판정한다.

## 로컬 실행

저장소 루트에서 시작한다. 기존 G3 실험과 같은 SKY130/GF180 모델 파일, Node 24, ngspice WASM ESM을 사용한다. `GF_MODEL`과 `NGSPICE_MJS`는 해당 로컬 파일의 절대경로로 설정한다. [시뮬레이터 안내](../README.md)

```bash
python -m pip install numpy scipy torch
cd campaigns/transferable_lane_monitor/sizing/device_transfer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python published_baselines_run.py --out runs --plan

python published_baselines_run.py \
  --sky ../../../../adc_driver/models/sky130_1v8_tt.spice \
  --gf "${GF_MODEL:?set GF_MODEL}" --wasm "${NGSPICE_MJS:?set NGSPICE_MJS}" \
  --node "$(command -v node)" --compiled --workers 4 --out runs
```

한 seed씩 실제 동작을 먼저 확인하려면 실행 명령의 `--out runs`를 `--out runs/published-smoke --seeds-per-condition 1`로 바꾼다. 세 외부 방법만 실행하려면 별도 출력 폴더를 지정하고 `--arms scbo kato dnnopt`를 추가한다. 중단 후에는 같은 명령으로 완료 궤적과 디스크에 기록된 코너를 재사용한다. `--out`을 같게 유지하면 실행 manifest가 변경을 검출한다.

완료 시 `runs/published_baselines_95/summary.json`에 조건별·PDK별·전체 성공 수, 제한평균, 누적 달성 수와 대응 seed bootstrap 구간을 저장한다. 미달은 100회 비용으로 계산한다. `reference_minus_control`의 음수는 기준 방법이 더 적은 호출을 썼다는 뜻이다. `--report-only --out runs`으로 원자료의 초기점, 전기적 통과 여부와 코너 계수를 재검산한다. 이 집계는 신경망과 GP를 다시 학습하지 않는다.

SPICE 없는 구현 검산은 `python published_baselines_check.py`로 실행한다. 고정 GP가 들어 있는 정렬 모델의 공분산, feasibility 우선순위, 제약 베이지안 탐색을 포함한 합성 측정 56개의 시드 대응과 호출 수, 중단 후 KATO 궤적 재개를 확인한다.

## 20-lane 배선 50개 × 사이징 7개 결과

위 버퍼 비교와 별도로, 20-lane 패키지 모델 50개에 동일한 2단 SKY130 TT 송신 버퍼를
연결한 조건에서 여섯 기준선과 제약 베이지안 탐색을 비교했다. 7개 방법에 같은 초기 폭,
32 Gb/s 패킷과 드리프트, 최대 100회 타깃 호출 예산을 적용했다. 타깃의 소자 폭 후보는
ngspice로 측정하고, 그 지연 변화를 20-lane 채널 모델에 넣어 최악 레인의 마진을 확인했다.
링크 모니터는 모든 방법에서 동일하게 128블록당 32회 측정한다.

성공은 송신 버퍼 EDP를 5% 이상 줄이고 네 셀 제약을 모두 지키며, 확인한 세 드리프트
조건에서 링크 마진 미달률을 nominal보다 1%p 넘게 늘리지 않는 경우다. 독립 패킷과
0.25 ps 정밀 검증을 통과해야 최초 성공으로 센다. 미달은 비교 평균에 100회로 넣는다.
동일 폭·코너·해상도의 ngspice 결과를 재사용해 독립 실행 예산상 18,422회 중 5,338회를
실제로 계산했다. 각 방법이 사용한 기존 소스 측정은 타깃 비용과 별도로 표시한다.

| 방법 | 성공 배선 | 평균 타깃 호출 | 기존 소스 측정 |
| --- | ---: | ---: | ---: |
| 무작위 소스 매핑 | 50/50 | 46.92 | 200 |
| 제약 베이지안 탐색 | 40/50 | 47.60 | 0 |
| MLP200 | 50/50 | 48.68 | 200 |
| SCBO | 47/50 | 51.88 | 0 |
| DNN-Opt | 48/50 | 52.52 | 0 |
| Elite 소스 매핑 | 40/50 | 56.20 | 200 |
| KATO | 47/50 | 64.64 | 50 |

[집계](../../../../implementation_results/20261128_20lane_sizing/summary.json) ·
[프로토콜](../../../../implementation_results/20261128_20lane_sizing/protocol.json) ·
[배선 출처 및 해시](../../../../implementation_results/20261128_20lane_sizing/layouts.json) ·
[350개 원본 압축](../../../../implementation_results/20261128_20lane_sizing/trajectories.tar.gz)

채널 데이터는 Lihao Liu 등의 [Chiplet-SI](https://github.com/Koyomi1999/Chiplet-SI/tree/a9408b5ed373ee76b41bb000c412cc8fd39154a7)
(© 2024 Lihao Liu, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))에서 추출했다.
각 원본 M1 배선의 중앙 20개 레인에서 0–60 GHz 전방 경로를 골라 복소수 행렬로 변환했다.
50개 행렬을 함께 제공하며, [추출 코드](../../experiments/package20_monitor/channel_bank.py)는
기존 3개 참조 행렬을 원본에서 정확히 재현한 뒤 새 배선을 고른다. 이 실험의 EDP는 버퍼
범위이며, 링크 판정은 공급전압에 따른 실측 버퍼 지연을 사용하는 유한 패킷 모델이다.

저장소 루트에서 고정된 채널 파일과 Node 24.22.3을 사용해 다시 실행한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python campaigns/transferable_lane_monitor/sizing/device_transfer/link20_sizing_benchmark.py \
  --channels implementation_results/20261128_20lane_sizing \
  --sky adc_driver/models/sky130_1v8_tt.spice \
  --wasm "${NGSPICE_MJS:?set path to pinned ngspice ESM}" \
  --node "${NODE24:?set path to Node 24.22.3}" \
  --out campaigns/20lane_sizing_rerun --workers 8
```

### 목표 난이도와 예산별 순위

동일한 350개 궤적을 재집계했다. 공통 초기 후보 네 개로 공동 통과한 사례와 20회 안에 통과한 사례는 각각 0개였다.

| 방법 | 40회 | 60회 | 100회 |
|---|---:|---:|---:|
| 제약 베이지안 탐색 | 30/50 | 40/50 | 40/50 |
| SCBO | 24/50 | 39/50 | 47/50 |
| MLP200 | 20/50 | 34/50 | 50/50 |
| 무작위 소스 매핑 | 18/50 | 46/50 | 50/50 |
| DNN-Opt | 15/50 | 38/50 | 48/50 |
| Elite 소스 매핑 | 10/50 | 40/50 | 40/50 |
| KATO | 9/50 | 22/50 | 47/50 |

각 예산 열은 원래 100회 프로토콜의 누적 최초 통과를 뜻한다. 40회 이하 결과는 탐색기나 확인 단계에 미리 40회 상한을 준 재실행과 구분한다.

개발 코너에서 셀 기준을 통과한 후보 1,170개 중 링크 거절은 6개였다. 확인 코너에서는 셀 기준을 통과한 324개 중 링크 거절이 0개였다. 정밀 단계에서는 셀 기준을 통과한 322개 중 링크 거절이 0개였다. 터미널 미달 28개 중 27개는 확인 코너에서 셀 기준을 통과한 후보가 없었고, 한 개는 정밀 확인을 시작할 호출 예산이 부족했다. 이 수치는 실제로 방문해 해당 단계까지 평가된 후보에 대한 것이다.

같은 네 개의 폭을 탐색하는 SKY130 TT 버퍼를 50개 채널에 공유했다. 최적화 초기화 시드는 다섯 개를 순환했다. 제약 BO의 미달 10개는 모두 시드 26112911, elite 매핑의 미달 10개는 모두 시드 26112913에서 나왔다. 100회에서 무작위 매핑 대비 elite의 평균 호출 차이 +9.28회는 배선별 대응 재표집으로는 95% 구간 +3.28~+15.80회다. 시작 시드를 블록으로 묶은 탐색적 재표집에서는 −4.32~+31.36회다. 후자는 독립 블록이 다섯 개뿐이며, 새 최적화 시드에 대한 순위가 민감함을 보여준다.

```bash
python campaigns/transferable_lane_monitor/sizing/device_transfer/link20_sizing_difficulty.py \
  implementation_results/20261128_20lane_sizing > /tmp/g3_link20_difficulty.json
```

[재집계 JSON](../../../../implementation_results/20261128_20lane_sizing/difficulty.json)

### 10% 목표 비교

5%에서 정밀 확인된 최선의 개선은 6.74%였다. 5% 궤적은 그 목표를 처음 통과하는 순간 끝났으므로 10% 달성률은 새로 측정한다. 위의 20-lane 실행 명령에 `--goal 0.90`을 추가하고 `--out campaigns/20lane_sizing_goal90`으로 새 결과를 저장한다. 같은 50개 채널, 시작 시드, 네 셀 제약, 링크 1%p 허용치와 100회 상한을 고정한다. 분석기는 압축 아카이브와 새 실행 폴더를 모두 읽는다.
