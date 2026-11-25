"""Report all four frozen manufacturing algorithms using exactly the same grid evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.report_phase_pairs import summarize
from experiments.run_phase_feedback import d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('primary',type=Path)
    parser.add_argument('extension',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    base=json.loads((args.primary/'summary.json').read_text())
    extra=json.loads((args.extension/'extra.json').read_text())
    proto=extra['protocol']
    if proto['primary_protocol']!=d.file_hash(args.primary/'protocol.json'):
        raise ValueError('Wrong primary panel')
    rows=[*base['rows'],*extra['rows']]
    n=proto['channels']
    table={(r['channel'],r['design']):r['methods']['grid'] for r in rows}
    if len(table)!=len(rows) or len(rows)!=n*len(proto['designs']):
        raise ValueError('Duplicated or missing channel-design combinations')
    lines=['# 같은 512비트 패킷에서 비교한 네 회로 탐색법', '',
           f"홀드아웃 {proto['root_seed']}의 {n}개 채널, 세 번 관측한 grid 위상을 모든 방법에 적용했다. 채널 통과는 긴 패킷 네 개에서 세 배선 모두 기준을 만족한 경우다. TX pJ/bit는 실패를 포함한 전체 채널 평균이다.",
           '', '|탐색 seed|방법|RC|RLC|전체|TX pJ/bit|BO 대비 추가 통과 / 새 실패|BO와 공동 통과 시 Δ pJ/bit|',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for seed in sorted(proto['locks']):
        for method in ('bo','mlp_local','bracket','reserve'):
            idx=proto['members'][f'{method}_s{seed}']
            ref=proto['members'][f'bo_s{seed}']
            arm=[table[i,idx] for i in range(n)]
            bo=[table[i,ref] for i in range(n)]
            y=np.array([int(r['all_pass']) for r in arm])
            diff=y-np.array([int(r['all_pass']) for r in bo])
            e=np.mean([r['energy'] for r in arm])
            joint=[a['energy']-b['energy'] for a,b in zip(arm,bo) if a['all_pass'] and b['all_pass']]
            delta=f'{np.mean(joint):+.6f}' if joint else '—'
            lines.append(f'|{seed}|{method}|{sum(y[::2])}/{len(y[::2])}|{sum(y[1::2])}/{len(y[1::2])}|{sum(y)}/{n}|{e:.6f}|{sum(diff>0)} / {sum(diff<0)}|{delta}|')
    lines+=['',f"이미 측정한 기본 설계 {len(base['protocol']['designs'])}개와 새 설계 {len(proto['designs'])-len(base['protocol']['designs'])}개를 합쳤다. 기본 시험 {base['physical_calls']}회 + 추가 {extra['physical_calls']}회다. seed 21·23의 기존 bracket은 같은 회로이고 seed 22의 reserve도 기존 bracket과 같다. 실험 단위는 채널로, RC/RLC 채널들은 L 이외의 물리 조건도 서로 다르다."]
    args.output.write_text('\n'.join(lines)+'\n')
    summarize(dict(protocol=proto,rows=rows),[('mlp_local','bo'),('bracket','bo'),('reserve','bo'),('reserve','bracket')],args.output.with_name('ALL_PAIRED_KO.md'))
    args.output.with_suffix('.json').write_text(json.dumps(extra,separators=(',',':'))+'\n')


if __name__ == '__main__':
    main()
