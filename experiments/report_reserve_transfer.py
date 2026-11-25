"""Paired energy and all-lane yield on shared RC/RLC channels for the shortlist ablation."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(data, out):
    proto = data['protocol']
    members = proto['members']
    table = {(r['design'], r['channel']):r['methods']['grid'] for r in data['rows']}
    n = proto['channels']
    lines = ['# 마진 여유 슬롯의 독립 검증', '',
             f"채널 seed {proto['root_seed']}, 독립 채널 {n}개. 주 지표는 세 번 관측한 grid 위상에서 네 패킷 × {proto['bits']}비트 × 세 배선을 모두 통과한 채널 수다. 평균 TX 에너지에는 실패 채널도 포함했다. 공동 통과 에너지 차이는 같은 채널의 reserve − 대조군이다.",
             '', '|seed|탐색|RC 통과|RLC 통과|전체|전체 TX pJ/bit|', '|---:|---|---:|---:|---:|---:|']
    outcomes = {}
    energies = {}
    for seed in sorted(proto['locks']):
        for method in ('bo', 'bracket', 'reserve'):
            key = f'{method}_s{seed}'
            cells = [table[members[key], i] for i in range(n)]
            ys = outcomes[key] = np.array([r['all_pass'] for r in cells], dtype=int)
            es = energies[key] = np.array([r['energy'] for r in cells])
            lines.append(f'|{seed}|{method}|{ys[::2].sum()}/{len(ys[::2])}|{ys[1::2].sum()}/{len(ys[1::2])}|{ys.sum()}/{n}|{es.mean():.6f}|')
    lines += ['', '|seed|reserve 대조군|추가 통과|새 실패|공동 통과|공동 통과 Δ TX pJ/bit|', '|---:|---|---:|---:|---:|---:|']
    for seed in sorted(proto['locks']):
        arm = f'reserve_s{seed}'
        for method in ('bo', 'bracket'):
            ref = f'{method}_s{seed}'
            diff = outcomes[arm]-outcomes[ref]
            both = (outcomes[arm]*outcomes[ref]).astype(bool)
            delta = f'{(energies[arm]-energies[ref])[both].mean():+.6f}' if both.any() else '—'
            lines.append(f'|{seed}|{method}|{(diff>0).sum()}|{(diff<0).sum()}|{both.sum()}|{delta}|')
    lines += ['', f"세 방법의 탐색·검증 배정량은 각 seed에서 방법당 96회다. 기존 bracket과 짧은 탐색 기록 및 중복된 긴 후보를 공유한 결과, 여유 슬롯을 확인하기 위한 추가 개발 시뮬레이션은 seed 21/22/23에서 8/0/8회였다. 새 채널 독립 검증은 {data['physical_calls']}개 transient를 사용했다. 같은 채널을 세 seed가 공유했다."]
    out.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report(json.loads(args.input.read_text()), args.output)
