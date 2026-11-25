"""Reduce full ngspice traces to per-seed held-out scores and auditable search receipts."""
import argparse
from pathlib import Path
import json

import numpy as np

from d2d_sizing.circuit import write_json


def read(path):
    return json.loads(path.read_text())


def collect(folders):
    rows, traces, protocols = [], {}, {}
    for folder in folders:
        summary = read(folder/'summary.json')
        seed = summary['seed']
        protocol = read(folder/'protocol.json')
        protocols[str(seed)] = {k:v for k,v in protocol.items() if k != 'suites'}
        packets = protocol['suites']['test']
        for method, result in summary['scores'].items():
            lock = summary['locked'][method]
            trace = read(folder/(method+'.json'))
            records = result['cases'] if result else []
            stage_rows = {}
            for stage in ('rows','checked'):
                stage_rows[stage] = [dict(x=r['x'], feasible=r['oracle']['feasible'],
                    energy=r['oracle']['energy_pj_bit'], margins=r['oracle']['margins'],
                    overshoot=r['oracle']['overshoot_fraction']) for r in trace[stage]]
            traces[f'{seed}-{method}'] = stage_rows
            rows.append(dict(seed=seed, method=method, design=lock['x'], calls=lock['calls'],
                case_payload_bits_per_lane=18*4*64+3*8*128,
                test_pass=sum(c['feasible'] for c in records), test_count=len(packets),
                all_pass=bool(result and result['feasible']), energy=result['energy_pj_bit'] if result else None,
                weakest_margin=min(result['margins']) if result else None,
                errors=sum(sum(c['errors']) for c in records) if records else None,
                cases=[dict(seed=p['seed'], correlated=p['correlated'], rate=p['context']['lane0']['rate_gbps'],
                    coupling=p['context']['coupling_pf'], skew=p['context']['lane0']['clock_offset_ui'],
                    **c) for p,c in zip(packets,records)]))
    return dict(rows=rows, traces=traces, protocols=protocols,
                suites=protocol['suites'])


def report(data, calibration, effect):
    lines = ['# 3개 배선 D2D 송신기 탐색', '',
        '가운데 송신선이 두 이웃과 각각 세 지점에서 용량성 결합을 갖는다. 세 송신기(총 108 MOS)는 같은 여섯 설계변수를 공유한다. 합성 RC 배선, PTM 트랜지스터 모델과 고정 수신 시점에서 ngspice로 측정했다.',
        '', '## 난이도 보정', '',
        '같은 64비트 패킷에서 양옆 배선이 가운데와 반대로 전환하도록 하고, 4가지 기존 설계를 측정했다. 아래 수치는 세 배선 중 가장 작은 마진(V/VDD)이다. 0.08 이상이어야 통과한다.',
        '', '|속도(Gb/s)|이웃당 결합(pF)|기본|기존 BO seed 3|기존 버퍼 설계|고구동 설계|','|---:|---:|---:|---:|---:|---:|']
    for rate,coupling in sorted({(r['rate'],r['coupling']) for r in calibration['rows']}):
        group={r['design']:min(r['margins']) for r in calibration['rows'] if (r['rate'],r['coupling'])==(rate,coupling)}
        lines.append(f'|{rate:.2f}|{coupling:.3f}|'+ '|'.join(f'{group[k]:.4f}' for k in ('nominal','bo3','archive','high_drive'))+'|')
    lines += ['', '같은 2.15 Gb/s, 이웃당 0.085 pF, 같은 패킷에서 두 배선 회로와 세 배선 회로를 직접 재측정했다. 가운데 배선 마진의 변화를 보면 추가 이웃의 영향을 분리할 수 있다.',
        '', '|설계|이웃 1개|이웃 2개|', '|---|---:|---:|']
    for row in effect['rows']:
        lines.append(f"|{row['design']}|{row['one_neighbour_margin']:.5f}|{row['two_neighbour_margin']:.5f}|")
    lines += ['', '## 동일 예산 비교', '',
        '세 방법은 초기 8개(고구동 기준 설계 포함), 짧은 평가 18개 × 4조건 × 64비트, 마지막 검증 3개 × 8조건 × 128비트를 공유했다. 방법과 seed마다 96회, 배선당 7,680개 payload bit 상당이다. 독립 데이터와 양옆이 가운데와 반대로 전환하는 데이터를 모두 포함한다. 기존 패킷의 접두부를 긴 개발 검증에 포함하고, 추가 패킷도 평가했다. 최종 시험용 128비트 패킷 16개 및 속도·결합·클록 오프셋 조합 2개는 탐색 후 고정한 설계에만 적용했다.',
        '', 'BO는 조건별 제약을 예측한다. 두 MLP는 5개 회귀 모델로 배선별 최저 마진과 에너지를 예측하고, 예측된 마진 위반량을 우선 줄인다. replay_local은 상위 설계의 변수 한두 개만 바꿔 근처 후보를 만든다. 두 MLP는 같은 수의 후보를 비교한다.',
        '', '|방법|전체 통과 seed|통과 seed의 평균 TX pJ/bit|','|---|---:|---:|']
    for method in ('bo','replay','replay_local'):
        group=[r for r in data['rows'] if r['method']==method]
        energies=[r['energy'] for r in group if r['all_pass']]
        avg=f'{np.mean(energies):.6f}' if energies else '—'
        lines.append(f'|{method}|{len(energies)}/{len(group)}|{avg}|')
    lines += ['', '|seed|방법|통과 패킷|TX pJ/bit|최저 마진|중심 비트 오류|',
              '|---:|---|---:|---:|---:|---:|']
    for r in data['rows']:
        metrics='—|—|—' if r['energy'] is None else f"{r['energy']:.6f}|{r['weakest_margin']:.5f}|{r['errors']}"
        lines.append(f"|{r['seed']}|{r['method']}|{r['test_pass']}/{r['test_count']}|{metrics}|")
    lines += ['', '에너지는 세 송신기의 공급 에너지를 배선 비트당 평균낸 값이다. 최종 통과에는 모든 배선에서 중심 오류 0개, 판정 시점 ±0.1 UI 구간의 최저 마진 0.08 V/VDD 이상, overshoot 0.2 VDD 이하가 필요하다. 동일한 설계를 반환한 경우 시험 결과도 동일하다. 전체 링크 전력 및 실리콘 BER은 이 시뮬레이션에서 측정하지 않았다.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('runs',type=Path,nargs='+')
    a=p.parse_args()
    data=collect(a.runs)
    write_json(a.output/'results.json',data)
    (a.output/'REPORT_KO.md').write_text(report(data,read(a.output/'calibration.json'),read(a.output/'neighbour_effect.json')))
