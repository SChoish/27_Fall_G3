# SKY130 ADC 입력 드라이버 실험 번들

기존 2단 OPAMP에 **입력 RC, MOS transmission gate, 샘플링 커패시터**를 연결한
트랜지스터 수준 ngspice 실험입니다. 샘플링 오차와 실제 공급 에너지를 측정하고,
사양을 만족하면서 에너지를 줄이도록 소자 크기·바이어스·RC를 최적화합니다.

**바로 실행할 회로, Python 평가기, 데이터 수집, GA·무작위 탐색, 회귀·RNC 학습과
지역 탐색, 실행 기록·파형이 포함되어 있습니다.** 확인한 결과는 `RESULTS.md`에 있습니다.

## 1. 설치와 첫 실행

Python 3.11 이상과 ngspice가 필요합니다. 이 번들은 ngspice 39에서 실행했습니다.
SKY130 모델은 라이선스와 함께 포함되어 있어 별도 PDK 설치가 필요하지 않습니다.

```bash
# Ubuntu/Debian 예시. 관리자 권한이 없다면 기존 ngspice 경로를 사용하세요.
sudo apt-get install ngspice
python -m pip install -r requirements.txt
python run.py doctor
python run.py evaluate --design examples/adc_reference.json --output results/my_evaluation
```

ngspice가 PATH에 없으면 해당 실행 파일의 **절대 경로**를 지정합니다.

```bash
export NGSPICE_BIN=/absolute/path/to/ngspice
python run.py doctor
```

`results/my_evaluation/summary.json`에 결과가 나오며,
`results/my_evaluation/00_nominal/waveform.png`에서 입력·앰프 출력·저장 전압·클록·오차를
볼 수 있습니다. 원본 SPICE netlist와 로그, 원시 파형도 같은 폴더에 저장됩니다.

독립 SPICE 파일도 제공합니다. 모델이 파일 안에 포함되어 Python 없이 실행할 수 있습니다.

```bash
mkdir -p results/standalone_user
cd results/standalone_user
ngspice -b ../../examples/adc_reference.cir
```

위 명령 후 프로젝트 최상위 폴더로 돌아와 다음 Python 명령들을 실행하세요.

## 2. 회로와 고정한 실험 조건

| 부분 | 구현 |
|---|---|
| 앰프 | SKY130 1.8V NFET/PFET 8개, 2단 Miller OPAMP, unity feedback |
| 입력 네트워크 | 앰프 출력의 2pF 부하, 직렬 RISO, 접지 CFILTER |
| 샘플링 | 실제 NMOS+PMOS transmission gate, 고정 15pF CSAMPLE |
| 클록 | 실제 CMOS 인버터 두 단(4 MOS), 이상적인 외부 클록 자극 |
| 전체 MOS 수 | **14개** |
| 전원·온도·공정 | 1.8V, 27°C, SKY130 TT |
| 샘플링 | 1MS/s, acquisition 200ns, 외부 클록 edge 2ns |
| 입력 | 0.8, 1.1, 0.85, 1.05, 0.8, 0.9V 순서; 다음 acquisition 100ns 전에 변경 |
| 초기화 | 2us 바이어스 정착 + 0.9V dummy 샘플 두 번; 그 뒤 여섯 샘플을 평가 |
| 정확도 조건 | 각 hold 구간의 입력 기준 최대 절대 오차 ≤ **1mV** |

TG 폭은 NMOS 4um/PMOS 8um, 길이 0.15um입니다. 클록 인버터의 폭은 2um/4um,
길이는 0.15um입니다. 앰프 MOS 길이는 0.5um이고 reference MOS 폭은 5um입니다.
MOS bulk를 전원/접지에 연결하고, 확산 길이 0.29um 가정으로 접합 면적·둘레를 지정합니다.
샘플링 노드에는 초기 DC 해와 유한 누설 경로를 위한 1TΩ 저항이 있습니다.

15pF와 200ns는 실제
[TI ADS7042 데이터시트](https://www.ti.com/lit/ds/symlink/ads7042.pdf)의 입력 샘플링
용량과 acquisition time에서 참고했습니다. **그 칩의 회로나 성능을 재현한 것은 아닙니다.**
현재 전압 범위와 1mV 오차는 이 번들의 실험 조건이며, 12비트 ADC 성능을 의미하지 않습니다.

## 3. 무엇을 측정하는가

외부 클록이 낮아진 후 20ns에 저장 전압을 읽고, 그 시점부터 다음 acquisition 시작
10ns 전까지의 **최대 hold 오차**를 측정합니다. 이 구간에는 다음 입력 전압으로 바뀌는
시간도 들어가므로 스위치가 끊긴 뒤의 입력 결합도 관찰합니다.

오차의 기준은 그 샘플에서 저장해야 했던 입력 전압입니다. 최종 DC 출력으로 기준을
옮기거나 정적 오프셋을 빼지 않습니다. 다음 입력으로 바뀌었다고 이전 샘플의 목표를
바꾸지도 않습니다. 코드의 `worst_hold_error_mv`가 최종 정확도 지표입니다.

함께 기록하는 지표는 acquisition 끝 직전 오차, 스위치 turn-off 전후 전압 변화,
읽는 시점의 오차, 증폭기·클록 회로 각각의 에너지입니다. `turnoff_change_mv`는 두 시점
사이의 실제 전압 차이이며, 순수한 charge injection만 분리해 추정한 값은 아닙니다.

각 평가 주기의 공급 에너지는 다음과 같이 계산합니다.

\[
E_k=-\int_{t_k}^{t_k+T_s}V_{DD}\,[I_{\mathrm{AVDD}}(t)+I_{\mathrm{CLKDD}}(t)]\,dt.
\]

SPICE 전압원 전류의 부호를 반영한 **순 공급 에너지**입니다. 바이어스와 반복 스위칭
소모를 모두 포함합니다. `mean_driver_energy_pj`는 여섯 주기의 평균 pJ/sample이고,
평균 전력은 이를 샘플링 빈도와 곱한 값입니다. 이상적인 입력·외부 클록 자극원의
에너지는 `mean_stimulus_energy_pj`로 별도 기록하며 최적화 목적에 포함하지 않습니다.

성공은 유한한 파형, 평가 구간 내 전압 범위, 읽는 시점의 TG off 상태, 양의 공급
에너지와 최대 오차 조건으로 판정합니다. 이는 이 입력 시퀀스의 기능 검사이며,
모든 동작점에서의 소신호 안정성 증명을 대신하지 않습니다.

## 4. 설계 변수와 최적화

| 변수 | 의미 | 범위 |
|---|---|---:|
| wi_um | 입력 차동쌍 폭 | 2–30um |
| wp_um | 1단 PMOS 전류미러 폭 | 2–30um |
| wt_um | tail MOS 폭 | 2–30um |
| wop_um | 2단 PMOS 폭 | 5–80um |
| won_um | 2단 NMOS 폭 | 5–80um |
| cc_pf | Miller 보상 용량 | 0.3–6pF |
| ibias_ua | reference current | 3–30uA |
| riso_ohm | 출력 직렬 저항 | 20–2000Ω |
| cfilter_pf | ADC 입력 RC의 커패시터 | 1–100pF |

연결 topology와 샘플링 스위치 크기는 각 탐색 중 고정입니다. 샘플링 용량·속도·온도는
설계값이 아닌 평가 조건입니다. 입력 RC를 바꿔도 고정된 15pF 샘플링 용량은 줄어들지 않습니다.

목적은 **오차 조건을 만족하면서 평균 공급 에너지를 최소화**하는 것입니다.
유효하지 않은 평가, 정확도 실패, 정확도 성공을 구분하며 실패 평가도 query를 소비합니다.
GA는 오차를 부등식 제약으로, 에너지를 목적함수로 사용합니다. 모든 방법이 같은
로그 정규화 설계 범위에서 탐색합니다.

```bash
python run.py demo --budget 16 --output results/my_demo
python run.py optimize --method ga --budget 32 --seed 0 --output results/my_ga
python run.py optimize --method random --budget 32 --seed 0 --output results/my_random
```

GA는 population 8, offspring 8, SBX 교차, polynomial mutation, 제약 우선 생존 선택을
사용합니다. budget은 8의 배수로 16 이상입니다. 초기 8개도 예산에 포함하며, 성공 후에도
나머지 예산으로 에너지를 줄이는 탐색을 계속합니다.

초기 설계는 nominal, 이전 OPAMP sizing, 개발 중 얻은 ADC reference, 무작위 5개입니다.
같은 seed이면 모든 방법에 동일합니다. **이미 알려진 feasible reference도 포함하는
warm-start 실험**이며, 반환만으로 성공률을 높였다는 주장을 하지 않습니다. 그 reference보다
에너지를 줄였는지가 목적입니다. 모든 초기 설계를 새 SPICE 평가로 확인하고 비용을 셉니다.
설계 query 1회는 context 하나당 transient 1회이므로 context가 네 개면 4회 실행입니다.

## 5. 데이터와 contrastive learning

새 샘플링 지표용 데이터를 수집하고 새 모델을 학습합니다. 이전의 gain/UGF용 OPAMP
체크포인트는 사용하지 않습니다. 기본 학습은 두 타깃 **log10(최대 hold 오차),
log10(평균 에너지)**를 예측하고, RNC는 이 정규화된 성능 벡터 간 거리 순위로 비교를
구성합니다. 사람이 positive/negative를 정하지 않습니다.

```bash
python -m pip install -r requirements-learning.txt
python run.py collect --n 128 --workers 4 --output data/my_source.json
python run.py train --data data/my_source.json --method regression --output checkpoints/my_regression.pt
python run.py train --data data/my_source.json --method rnc --output checkpoints/my_rnc.pt
python run.py optimize --method regression --checkpoint checkpoints/my_regression.pt --budget 32 --output results/my_regression
python run.py optimize --method rnc --checkpoint checkpoints/my_rnc.pt --budget 32 --output results/my_rnc
```

학습/검증/테스트는 고정 무작위 분할 70/15/15%입니다. 정규화는 유효 학습 데이터만,
checkpoint 선택은 유효 검증 데이터만 사용합니다. 사전 데이터 평가 비용은 탐색 budget에
포함되지 않고 별도로 기록됩니다. 같은 데이터에서 최선의 feasible 설계를 선택하는
기준선도 연구 평가에 포함해야 합니다. 번들의 `validate_bundle.py`가 이를 보고합니다.

지역 탐색은 이전 **ALARM-inspired subset**을 연결했습니다: gradient outer product와
대각 damping으로 만든 metric, metric-preconditioned descent, surrogate Armijo,
실측 상위 3개에서 시작하는 탐색, 모든 유효 실측을 이용한 residual ridge 보정입니다.
원본 ALARM의 attention metric과 critic buffer는 구현하지 않았습니다. RNC는 별도
표현학습 항입니다. 이 모델은 고정된 context 목록을 입력에 내장한 모델이므로, 평가
조건이 학습 조건과 다르면 체크포인트 사용을 거부합니다.

실행 확인용 `data/source_smoke.json`은 24개뿐입니다. 함께 제공한 모델과 1개 seed 결과는
동작 확인용이고, RNC의 우위·미지 topology 전이를 검증한 결과가 아닙니다.

## 6. 부하·속도·온도를 바꾸기

```bash
python run.py evaluate --design examples/adc_reference.json --contexts configs/stress.json --output results/my_stress
python run.py optimize --method ga --contexts configs/stress.json --budget 32 --output results/my_robust_ga
```

`configs/stress.json`은 nominal, 30pF, 100ns acquisition, 85°C의 네 조건입니다.
이는 지원 기능을 보여주는 별도 평가이며 nominal 설계가 전부 통과한다고 가정하지 않습니다.
복수 조건의 탐색은 **모든 조건에서 정확도를 만족**하면서 조건별 평균 에너지를 줄입니다.
회귀/RNC를 사용하려면 그 context 목록으로 데이터를 새로 수집해 학습해야 합니다.

## 7. 검증과 파일 안내

```bash
python -m unittest discover -s tests -v
python validate_bundle.py
```

첫 명령은 저장 전압의 목표 유지, 정적 오차 보존, 에너지 단위, 정확한 적분 구간,
클록 오류 검사를 수행합니다. 둘째 명령은 제공된 결과·데이터의 비용과 초기 설계,
checkpoint 데이터 분할, dense 및 독립 SPICE 실행 결과를 다시 점검합니다.

| 파일 | 역할 |
|---|---|
| circuit.py | MOS 회로 생성, ngspice 실행, 정확도·에너지 측정 |
| run.py | 사용자 명령 진입점 |
| search.py | 데이터 수집, GA·무작위·학습 기반 탐색 |
| learning.py / rnc.py | 회귀·RNC 학습, ALARM-inspired 지역 탐색 |
| configs/ | nominal·stress 조건 |
| examples/ | 기준 설계 세 개, 독립 SPICE netlist |
| models/ | 실제 SKY130 TT 모델, Apache-2.0 라이선스·출처 |
| data/ / checkpoints/ | 실행 확인용 데이터와 모델 |
| results/ | 개발 기록, 네 방법 실행 기록, 검증 파형과 집계 |

이 회로는 **SAR ADC 입력을 단순화한 track-and-hold 부하**입니다. ADC 변환 로직,
capacitor DAC의 변환 중 재배치, 비교기 kickback은 포함하지 않습니다. MOS 모델의
비선형·기생 성분과 스위칭 전하 효과를 계산하지만, 열잡음 transient, mismatch,
process corner sweep, 추출 배선 기생, ENOB/SNR 검증은 수행하지 않았습니다.
R/C·입력 자극·reference current source는 이상적인 요소입니다.

모델 원본은 [SkyWater SKY130 primitive library](https://github.com/google/skywater-pdk-libs-sky130_fd_pr)의
commit `f62031a1be9aefe902d6d54cddd6f59b57627436`입니다. 모델 계수는 변경하지 않았고,
기존 번들에서 사용한 include 통합본과 라이선스를 그대로 가져왔습니다.
