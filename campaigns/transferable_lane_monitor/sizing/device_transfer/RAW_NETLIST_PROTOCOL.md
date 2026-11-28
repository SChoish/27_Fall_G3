# 원시 물리 정보를 이용하는 회로 전이

사용자는 회로 파일, PDK/모델, 설계변수 및 허용 범위, 목표와 측정 조건을 제공한다. 이 시험의 어댑터는 생성된 MOS/커패시터/독립 전압원 SPICE 인스턴스를 읽는다. 소자 유형, nominal W/L, PDK ID, 정전용량 또는 전원/펄스 크기와 공통 net 접속을 자동 추출한다. 이름은 net 일치 여부와 모델 식별에만 사용한다. 알고리즘에 출력단/내부단 레이블, 깊이, 팬아웃 역할, 폭 그룹, 수작업 초기 설계, 도메인별 행동 규칙은 주지 않는다. 각 MOS의 폭은 독립적으로 움직인다.

그래프 노드는 primitive 인스턴스다. 같은 net을 하나 이상 공유하면 연결한다. 전원·접지 연결과 자기 연결을 포함한다. 이 표현은 어느 핀끼리 연결되었는지, 몇 개의 net을 공유하는지까지 구별하지 않는다. 입력 모델 파일은 허용된 두 PDK와 사용한 primitive 문법 범위에서 해석한다.

소스는 기존 SKY130 200회 및 같은 네 토폴로지·같은 100개의 설계의 GF180 200회다. 이 400회에서 학습한 익명 MLP, 물리 속성 MLP, 물리 속성 + 완전 그래프 attention, 물리 속성 + 실제 net 공유 GAT를 비교한다. 마지막 두 군은 동일 아키텍처와 초기화에서 인접 마스크만 다르다. 모든 탐색기는 소스 신경망 평균 + 기존 타깃 잔차 GP를 사용한다. 별도로 SKY130의 200회만으로 물리 GAT를 학습하여 GF180 두 조건에서 미관측 PDK 전이를 시험한다. 이때 GF180 one-hot 열에 소스 근거가 없으므로 미관측 ID를 all-zero로 마스킹한다. 사용한 L, 전원 등의 연속 물리값은 그대로 제공한다.

두 평가 회로군은 부하·결합 변경 6단·2단과 새 5인버터 3출력 분기다. 각 PDK, seed 81–83, 5%/10% 단독 목표다. 네 익명 랜덤 웜업, 제약, 정밀 확인까지 100회 상한을 유지한다. 주요 비교 지표는 5% 도달의 제한 평균과 성공 수, 보조 지표는 10%다. 전체 결론은 PDK별 결과와 함께 보고한다. 모델 구조와 설정을 고정한 뒤 이 물리 입력 방법들의 결과를 측정한다.

## 넷리스트로 후보를 요청하는 방법

`python raw_netlist_suggest.py design.cir --pdk sky --history history.json`으로 다음 SPICE 후보의 소자별 nominal 폭 배율을 얻는다. `--method devices400`으로 물리 MLP, `--method wires400`으로 실제 net 공유 GAT를 선택한다. 학습에 필요한 기본 소스 데이터는 raw_netlist_source.py가 생성하는 runs/raw_netlist/source.json에서 읽는다. --source로 다른 실험의 측정 JSON이나 source.json이 포함된 결과 ZIP을 지정할 수 있다. 회로 그래프나 소자 역할 목록을 만들 필요가 없다.

history.json의 limits는 MOS 인스턴스 이름별 [최소, 최대] 폭 배율이다. observations가 비어 있으면 기준 설계를 반환한다. 이후 각 관측에는 소자 이름별 ratios와 개발 코너의 log_metrics 다섯 값(기준 대비 EDP, 허용치 대비 지연·slew·공급 민감도·레일 위반 비율의 로그)을 넣는다. 이 벤치마크에서는 기존 circuit.ratios가 시뮬레이터 출력으로 계산한다. 확인 코너 관측은 입력하지 않는다. 이 진입점은 본 시험의 평탄화된 두 PDK primitive deck용 후보 제안기이며 시뮬레이터 실행과 비용 집계는 평가 어댑터가 담당한다.

### 소자 역할과 무관한 공통 폭 범위

기존 설계 영역은 최초 입력에 붙은 소자에 1.0배, 나머지에 1.3배 상한을 준다. 실제 납품 사양에서 입력 부하 제한을 받는 경우와 별개로, 이 벤치마크의 탐색 범위가 역할 단서가 되는지 고려했다. 독립 보조 실험에서는 모든 MOS에 0.58–1.30배를 똑같이 적용한다. 소스의 네 회로도 같은 공통 범위에서 25개씩 재추출하고 각 PDK에서 측정한다(400 transient). 두 회로군 × 두 PDK × 같은 seed 81, 82에서 타깃 전용 BO, 물리 속성 MLP+BO, net 공유 GAT+BO의 5% 최초 도달을 비교한다. 제약, 고정 nominal 회로 및 타깃 100회 예산은 같다. 기존 시험과 탐색 공간 및 소스가 함께 바뀌므로 차이를 단일 요인에 귀속하지 않는다.


PDK 인자는 `sky130`과 `gf180` 표기도 받는다. 모든 MOS의 허용 배율이 같으면 history에 `"common_limits": [0.58, 1.30]` 한 쌍만 넣는다. 각 관측의 `dev`에 개발 코너별 측정값(각각 energy와 출력별 rise, fall, sr, sf, hi, lo)을 주면 nominal 대비 로그 지표를 계산한다. 첫 제안용 전체 입력은 `{"common_limits":[0.58,1.30],"observations":[]}`다. nominal 및 네 랜덤 웜업 후보는 소스 파일 없이도 요청할 수 있다.

### 실행과 독립 검증

README의 모델 및 ngspice 경로를 이용해 정의한 bash 배열 `opts`를 사용한다. 두 실행기는 같은 출력 폴더에서 완료된 탐색 기록을 건너뛴다.

```bash
python raw_netlist_source.py source "${opts[@]}"
python raw_netlist_run.py search "${opts[@]}"
python prior_transfer_audit.py physical
python prior_transfer_audit.py physical --replay

python uniform_netlist_run.py source "${opts[@]}"
python uniform_netlist_run.py search "${opts[@]}"
python prior_transfer_audit.py uniform
python prior_transfer_audit.py uniform --replay
```

첫 소스 실행은 기존 SKY130 소스가 로컬에 없으면 저장소의 anonymous_width_measurements.zip에서 읽는다. uniform 실험에는 common_limits와 함께 `--source runs/uniform_netlist/source.json`을 사용한다. 벤치마크는 기준 회로와 기존 측정 테스트벤치를 제공한다. 새 회로에서는 관심 출력, 자극, 사양과 개발·검증 조건을 담은 테스트벤치가 필요하다. 현재 입력 어댑터는 이 시험의 SKY130·GF180 모델을 지원하며 W를 탐색하고 L을 읽는다.
