"""Paired yield and TX deltas over independent channels, conditional on three fixed search seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(panel, arms, out):
    proto = panel['protocol']
    table = {(r['design'],r['channel']):r['methods']['grid'] for r in panel['rows']}
    seeds = list(proto['locks'])
    channels = np.arange(proto['channels'])
    rng = np.random.default_rng(37491)
    samples = np.concatenate([rng.choice(ids, (10000,len(ids)),replace=True)
                              for ids in (channels[::2],channels[1::2])],axis=1)
    lines = ['# 같은 채널에서 비교한 강건성과 송신 에너지', '',
             f"채널 seed {proto['root_seed']}, 공통 3점 grid 보정. 각 채널에서 같은 탐색 seed끼리 차이를 낸 뒤 세 seed에 대해 평균했다. RC와 RLC 개수를 유지한 채 채널을 10,000회 재표집한 percentile 95% 구간이다. 탐색 seed 세 개는 고정했으며 실패 채널 에너지도 포함했다.",
             '', '|비교|통과율 차이 %p [95%]|TX pJ/bit 차이 [95%]|','|---|---:|---:|']
    for arm, ref in arms:
        differences = []
        for i in channels:
            values = []
            for seed in seeds:
                a = table[proto['members'][f'{arm}_s{seed}'],i]
                b = table[proto['members'][f'{ref}_s{seed}'],i]
                values.append([100*(int(a['all_pass'])-int(b['all_pass'])),a['energy']-b['energy']])
            differences.append(np.mean(values,axis=0))
        d = np.asarray(differences)
        lo,hi = np.quantile(d[samples].mean(axis=1),[.025,.975],axis=0)
        y,e=d.mean(axis=0)
        lines.append(f'|{arm} − {ref}|{y:+.1f} [{lo[0]:+.1f}, {hi[0]:+.1f}]|{e:+.6f} [{lo[1]:+.6f}, {hi[1]:+.6f}]|')
    lines += ['', '통과율과 에너지는 각각 해석한다. 같은 비트율로 송신해도 사양에 미달한 채널에서는 그 에너지를 정상 전달 비용으로 볼 수 없다. 표본이 작아 관측된 득실이 전혀 없을 때 재표집 구간이 0으로 퇴화할 수 있다. 이는 동등성 판정이 아니다.']
    out.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    panel=json.loads(args.panel.read_text())
    members=panel['protocol']['members']
    arms = [('reserve','bracket'),('reserve','bo')] if 'reserve_s21' in members else [('mlp_local','bo'),('bracket','bo')]
    summarize(panel,arms,args.output)
