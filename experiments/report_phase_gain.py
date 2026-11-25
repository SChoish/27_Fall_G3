"""Compare deployment policies on the same circuits and independent channels."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    proto=data['protocol']
    n=proto['channels']
    cells={(r['channel'],r['design']):r['methods'] for r in data['rows']}
    rng=np.random.default_rng(37491)
    samples=np.concatenate([rng.choice(ids,(10000,len(ids)),replace=True)
                            for ids in (np.arange(0,n,2),np.arange(1,n,2))],axis=1)
    lines=['# 고정한 회로에서 세 관측을 사용하는 방법의 효과','',
           '채널 seed 49103. 각 방법에서 제조 seed 21·22·23을 고정했다. 같은 채널·회로·세 훈련 패킷의 grid와 포물선 결과를 짝지었다. 신뢰구간은 독립 채널을 RC/RLC 층별로 재표집했다.',
           '', '|회로 탐색|grid|포물선|추가 통과 / 새 실패|포물선 − grid %p [95%]|배선별 곡선과 포물선 판정 불일치|',
           '|---|---:|---:|---:|---:|---:|']
    for method in ('bo','mlp_local','bracket'):
        r=[[cells[i,proto['members'][f'{method}_s{seed}']] for seed in sorted(proto['locks'])] for i in range(n)]
        vals={k:np.array([[int(s[k]['all_pass']) for s in row] for row in r])
              for k in ('grid','quadratic','lane_quadratic')}
        dif=vals['quadratic']-vals['grid']
        lo,hi=np.quantile(100*dif.mean(axis=1)[samples].mean(axis=1),[.025,.975])
        mismatches=np.count_nonzero(vals['quadratic']!=vals['lane_quadratic'])
        lines.append(f"|{method}|{vals['grid'].sum()}/{dif.size}|{vals['quadratic'].sum()}/{dif.size}|{(dif>0).sum()} / {(dif<0).sum()}|{100*dif.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]|{mismatches}/{dif.size}|")
    lines+=['', '위상 보정의 세 관측과 송신 파형이 같으므로 보정법 간 추가 TX 비용은 0이다. 48개 채널·seed 조합은 독립 채널 16개를 공유한다.']
    args.output.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
