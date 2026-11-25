import argparse
import json
from pathlib import Path

import numpy as np


def report(data, path):
    rows = data['rows']
    proto = data['protocol']
    lines = ['# 관측 두 번으로 보정한 독립 채널 결과', '',
             f"bo13 고정 회로, 새 채널 seed {proto['seed']}, 채널별 256비트 검증 패킷 네 개. 개발에서 선택한 고정 경로는 32→{proto['fixed_second']}이다. 적응 경로는 첫 64비트 응답을 보고 16/48 중에서 고른다. 한 번만 읽는 대조군도 같은 개발 참조 자료를 쓴다.",
             '', '|방법|보정 비트/배선|RC|RLC|전체 통과|보정 TX pJ/링크|', '|---|---:|---:|---:|---:|---:|']
    for name,n in [('fixed32',0),('one',1),('two_fixed',2),('two_adaptive',2)]:
        passed = [int(r['methods'][name]['all_pass']) for r in rows]
        energy = np.mean([sum(r['probe_pj_link'][:n]) for r in rows])
        lines.append(f'|{name}|{64*n}|{sum(passed[::2])}/{len(passed[::2])}|{sum(passed[1::2])}/{len(passed[1::2])}|{sum(passed)}/{len(rows)}|{energy:.2f}|')
    diff = [int(r['methods']['two_adaptive']['all_pass'])-int(r['methods']['two_fixed']['all_pass']) for r in rows]
    directions = [r['methods']['two_adaptive']['second'] for r in rows]
    e = np.mean([r['methods']['one']['energy'] for r in rows])
    lines += ['', f"적응 경로는 코드 16을 {directions.count(16)}회, 48을 {directions.count(48)}회 사용했다. 고정 경로 대비 추가 통과 {diff.count(1)}개, 새 실패 {diff.count(-1)}개였다. 실제 검증 payload 송신 에너지는 공통 {e:.6f} pJ/bit다. 총 {data['physical_calls']}개 transient를 공유했다. 표의 보정 비용은 세 배선 payload 전류 적분값으로, 수신기 전력과 프리앰블은 제외했다."]
    path.write_text('\n'.join(lines)+'\n')
    path.with_suffix('.json').write_text(json.dumps(data,separators=(',',':'))+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    a=parser.parse_args()
    report(json.loads(a.input.read_text()),a.output)
