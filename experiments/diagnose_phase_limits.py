"""Classify validation failures using counterfactual timing choices on the same packets."""
import argparse
from pathlib import Path
import json

import numpy as np
from experiments.analyze_phase_feedback import GRID,load_episodes
from experiments.run_phase_feedback import CODES


def classify(episode):
    packets=episode['validation']
    M=np.min([v['margin'] for v in packets],axis=0)
    E=np.sum([v['errors'] for v in packets],axis=0)
    rail=max(p['overshoot'] for p in packets)
    ok=(M>=.08)&(E==0)
    best=int(M.min(axis=0).argmax())
    local=M.argmax(axis=1)
    if rail>.2:
        label='rail_limit'
    elif ok[:,GRID[1]].all():
        label='fixed_pass'
    elif ok.all(axis=0).any():
        label='shared_recovery'
    elif ok.any(axis=1).all():
        label='lane_specific_recovery'
    else:
        label='insufficient_margin_or_errors'
    available=np.min(np.max(M,axis=1))
    return dict(channel=episode['channel'],design=episode['design'],category=label,
        fixed=float(M[:,GRID[1]].min()),shared=float(M[:,best].min()),individual=float(available),
        best_shared=int(CODES[best]),best_individual=CODES[local].tolist(),rail=rail,
        limiting_lane=int(M.max(axis=1).argmin()),
        energy=float(np.mean([v['energy'] for v in packets])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    protocol,episodes=load_episodes(a.folder)
    rows=[classify(e) for e in episodes]
    lines=['# 수신 위상으로 회복 가능한 실패의 범위','',
           f"채널 seed {protocol['root_seed']}, 독립 채널 {protocol['channels']}개, 설계 {len(protocol['designs'])}개다. 같은 설계와 모든 검증 패킷을 고정하고 위상만 바꿔 각 조합을 아래 다섯 경우 중 하나로 분류했다.",
           '', '|설계|고정 위상 통과|공통 위상으로 추가 회복|배선별 위상으로 추가 회복|한 배선 이상에서 마진/오류 미달|레일 초과|평균 TX pJ/bit|',
           '|---|---:|---:|---:|---:|---:|---:|']
    categories=['fixed_pass','shared_recovery','lane_specific_recovery','insufficient_margin_or_errors','rail_limit']
    for name in protocol['designs']:
        g=[r for r in rows if r['design']==name]
        counts=[str(sum(r['category']==c for r in g)) for c in categories]
        lines.append('|'+ '|'.join([name,*counts,f"{np.mean([r['energy'] for r in g]):.6f}"])+'|')
    lines+=['','상한은 허용한 13개 코드(8~56, 4 단위)에 한정된다. 배선별 위상은 세 배선마다 독립 코드를 쓸 수 있다고 가정한 사후 진단이다. 레일 초과는 위상으로 회복할 수 없어 먼저 분류했다.']
    a.output.write_text('\n'.join(lines)+'\n')
    a.output.with_suffix('.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
