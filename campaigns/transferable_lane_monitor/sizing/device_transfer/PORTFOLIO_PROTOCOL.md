# 역할 탐색 두 번 뒤 국소 웜업으로 전환하는 정책

앞서 비교한 6조건의 실행 결과 일부를 본 뒤 아래 정책을 고정했다. 새 검증은 6단과 2단의 서로 다른 버퍼를 정전 결합한 2출력, 8인버터 회로에서 수행한다. 부모 (-1,0,1,2,3,4,-2,6), nominal 크기 (1,1.55,2.1,2.7,3.3,3.8,1.05,2.55), 출력 (5,7), 부하 (1.32,.78), 상호 용량 2.15 fF를 SKY130과 GF180에 동일하게 적용한다. 새 회로 seed 51,52,53은 기존 6조건의 31,32,33과 다르다.

확인할 단일 조합은 역할 정책의 첫 두 개(출력 폭 0.84, 개발 제약으로 보정한 같은 방향의 두 번째 폭)를 시도한 뒤, 종전의 웜업 4개 + 잔차 critic을 실행한다. 동일 설계 재제안은 메모리에서 개발 관측을 읽는다. 개발 관측만으로 다음 폭을 만든다. 5%, 10%를 각각 단독 목표로 평가해 목표 가능성이 있는 경우에만 확인 코너 2회를 지불한다. 최초 확인에서 종료하거나 100회까지 진행한다. 역할 직접 탐색과 웜업 4개 단독이 대조군이다. 성공과 실패 포함 비용을 기존 6조건과 미사용 2조건에 각각 보고한다. 기초 실측을 공유한 조합별 궤적을 독립적인 물리 시뮬레이션 수로 더하지 않는다.

## 수치 검산과 재실행

원래 1 ps 평가의 독립 0.25 ps 확인에서 경계 설계가 뒤집히면 원래 100회 탐색 상한까지 같은 순서의 후속 후보를 측정한다. 더 미세한 시간 간격의 기준·설계 재측정을 탐색 예산과 별개로 기록하고, 같은 상한 안에 포함하는 비용 민감도도 보고한다. 후자는 해당 목표에 1 ps로 도달한 첫 설계에서 정밀 기준 4회와 정밀 설계 4회를 지불하고, 다음 적격 설계마다 정밀 설계 4회를 지불한다. 1 ps 개발·확인 제약을 통과하지 않은 설계는 정밀도 검산 후보가 될 수 없다.

주 실행 완료 후 기존 모델/시뮬레이터 opts를 사용해 아래 순서로 재생한다.

```bash
python century_portfolio.py search "${opts[@]}"
python century_portfolio.py source "${opts[@]}"
python century_portfolio_check.py search "${opts[@]}"
python century_portfolio_check.py replay
python century_precision_extension.py search "${opts[@]}"
python century_precision_cost.py
python century_portfolio_report.py
```

archive와 저장된 precision_continuation.json이 있으면 century_restore.py, century_portfolio_check.py replay와 century_precision_cost.py는 SPICE 없이 실행된다.
