"""Summarize matched manufacturing budgets and paired blind channel outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def report(campaigns, out):
    data = read(campaigns/'phase_transfer/summary.json')
    protocol = data['protocol']
    channels = protocol['channels']
    lines = ['# 동일 예산 회로 탐색 및 독립 검증', '',
             '검증 단위는 채널이다. 표의 통과는 세 배선이 512비트 패킷 네 개에서 모두 제약을 만족한 경우다. '
             '에너지는 실패 채널까지 포함한 전체 검증 패킷의 송신 에너지다. 위상 보정에는 192비트/배선을 사용했다.',
             '', '|seed|탐색|개발 통과|개발 TX pJ/bit|그리드 RC|그리드 RLC|포물선|배선별 곡선|전체 TX pJ/bit|BO 대비 그리드 득/실|',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups = {name: sorted([r for r in data['rows'] if r['design']==idx], key=lambda r:r['channel'])
              for name, idx in protocol['members'].items()}

    def passed(row, method):
        return int(row['methods'][method]['all_pass'])

    costs = []
    for seed in sorted(protocol['locks']):
        seen = set()
        for name in ('bo', 'bracket', 'mlp_local'):
            arm = read(campaigns/f'phase_size_s{seed}'/f'{name}.json')
            group = groups[f'{name}_s{seed}']
            delta = [passed(a,'grid')-passed(b,'grid') for a,b in zip(group,groups[f'bo_s{seed}'])]
            rc,rlc = [sum(passed(r,'grid') for r in group if r['channel']%2==k) for k in (0,1)]
            quad,lane = [sum(passed(r,m) for r in group) for m in ('quadratic','lane_quadratic')]
            energy = np.mean([r['methods']['grid']['energy'] for r in group])
            selected = arm['selected']['oracle']
            lines.append(f"|{seed}|{name}|{'통과' if selected['feasible'] else '실패'}|{selected['energy_pj_bit']:.5f}|{rc}/{channels//2}|{rlc}/{channels//2}|{quad}/{channels}|{lane}/{channels}|{energy:.5f}|{delta.count(1)}/{delta.count(-1)}|")
            for stage in ('rows', 'validation'):
                seen.update((stage,tuple(r['x'])) for r in arm[stage])
        costs.append((seed,sum(4 if stage=='rows' else 8 for stage,_ in seen)))
    lines += ['', '탐색은 seed·방법당 짧은 18설계 × 4조건, 긴 3설계 × 8조건으로 96회, 배선당 7,680비트다. '
              '세 방법 합계는 seed당 288회, 세 seed 합계는 864회다. 같은 seed에서 동일 설계·패킷은 캐시를 공유한다.',
              '', '|seed|캐시 중복 제거 후 고유 transient|', '|---:|---:|']
    lines += [f'|{seed}|{n}|' for seed,n in costs]
    lines += ['', f"새 채널 seed {protocol['root_seed']}에서 고유 설계 {len(protocol['designs'])}개 × {channels}채널 × 7패킷을 측정해 독립 검증에 {data['physical_calls']}회가 들었다. "
              '재시작 전 체크포인트에 남지 않은 부분 실행의 비용은 복원할 수 없다. 개발 데이터 획득 448회, 물리 요인 대조 24회, 기존 설계 독립 시험 672회는 별도다. 수신기 회로 전력은 측정하지 않았다.']
    out.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report(args.campaigns, args.output)
