"""Reintegrate boundary cases at half the time step, holding chosen receiver codes fixed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json

from experiments.analyze_phase_feedback import score
from experiments.run_phase_feedback import CODES, curve, d, designs, suite, transient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    panel=json.loads(args.scores.read_text())
    checks=[]
    candidates=designs()
    lines=['# 판정 경계의 시간 해상도 검증','',
           '49102 독립 시험에서 RC/RLC별로 grid 통과와 실패 각각 마진 0.08에 가장 가까운 조합 하나를 골랐다. 기존 보정 코드·회로·채널·패킷은 고정하고 ngspice 최대 시간 간격만 UI/64에서 UI/128로 줄였다. 재보정 없이 검증용 네 패킷을 다시 적분했다.',
           '', '|채널|설계|고정 코드|원래 마진|세밀 마진|원래 통과 패킷|세밀 통과 패킷|TX pJ/bit 변화|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for parity in (0,1):
            for passed in (True,False):
                group=[r for r in panel['rows'] if r['channel']%2==parity and r['methods']['grid']['all_pass']==passed]
                row=min(group,key=lambda r:abs(r['methods']['grid']['margin']-.08))
                channel,name=row['channel'],row['design']
                old=row['methods']['grid']
                cases=[replace(c,points_per_ui=128) for c in suite(panel['root_seed'],channel,panel['long_bits'])[3:]]
                waves=list(pool.map(lambda c:curve(transient(candidates[name],c),c),cases))
                new=score(dict(validation=waves),CODES.tolist().index(old['code']))
                line='RLC' if parity else 'RC'
                checks.append(dict(channel=channel,line=line,design=name,old=old,new=new))
                lines.append(f"|{line} {channel}|{name}|{old['code']}|{old['margin']:.6f}|{new['margin']:.6f}|{old['passes']}/4|{new['passes']}/4|{new['energy']-old['energy']:+.6f}|")
    flips=sum(r['old']['all_pass']!=r['new']['all_pass'] for r in checks)
    lines+=['',f'16개 transient를 추가했다. 네 조합 중 {flips}개에서 전체 통과 판정이 바뀌었다. 경계 표본을 의도적으로 골랐으므로 이 비율로 전체 수치를 보정할 수 없다.']
    args.output.write_text('\n'.join(lines)+'\n')
    d.write_json(args.output.with_suffix('.json'),dict(physical_calls=16,checks=checks))


if __name__ == '__main__':
    main()
