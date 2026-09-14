# 27 Fall G3

두 개의 ngspice 실험 번들입니다. 시스템 `ngspice`를 쓰거나 `NGSPICE_BIN`으로 경로를 지정하세요.

## D2D coupled calibration

기생 커패시턴스로 결합된 D2D 두 레인 보정입니다.

- 실행: [`d2d_coupled_calibration/README.md`](d2d_coupled_calibration/README.md)
- 요약: [`d2d_coupled_calibration/COUPLED_REPORT.md`](d2d_coupled_calibration/COUPLED_REPORT.md)

원본 번들의 Linux ngspice 런타임은 저장소에 넣지 않았습니다.

## SKY130 ADC input driver

SKY130 TT MOS로 만든 ADC 입력 드라이버(2단 OPAMP + transmission gate + 샘플링)입니다.

- 실행: [`adc_driver/README.md`](adc_driver/README.md)
- 결과: [`adc_driver/RESULTS.md`](adc_driver/RESULTS.md)
- ALARM attention 구현·비교: [`adc_driver/ALARM_ATTENTION.md`](adc_driver/ALARM_ATTENTION.md)

ADC 드라이버의 regression/RNC 탐색은 학습되는 좌표 attention을 사용합니다.
라인서치는 기본적으로 끄며 `--line-search armijo`로 비교할 수 있습니다.
