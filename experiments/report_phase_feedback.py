"""Compact per-channel outcomes for phase calibration, manufacturing search, and blind transfer."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.run_phase_feedback import d


def read(p):
    return json.loads(p.read_text())


def phase_report(data,path,title):
    names=list(data['rows'][0]['methods'])
    designs=list(dict.fromkeys(r['design'] for r in data['rows']))
    lines=[f'# {title}', '', f"독립 채널 {data['channels']}개, 회로 {data['design_count']}개, 조합당 긴 패킷 4개 × {data['long_bits']}비트/배선이다. 각 방법은 같은 파형을 공유했다. ngspice 실행은 총 {data['physical_transients']}회다.",
           '', '|방법|보정 비트/배선|'+ '|'.join(designs)+'|총 통과 조합|','|---|---:|'+'---:|'*(len(designs)+1)]
    for method in names:
        groups=[[r['methods'][method] for r in data['rows'] if r['design']==name] for name in designs]
        counts=[sum(s['all_pass'] for s in g) for g in groups]
        bits='사후 상한' if method=='oracle' else ('0' if method in ('fixed32','trained_static') else '192')
        cells=[f'{n}/{len(g)}' for n,g in zip(counts,groups)]
        lines.append('|'+ '|'.join([method,bits,*cells,f"{sum(counts)}/{len(data['rows'])}"])+'|')
    lines+=['', '같은 채널에서 서로 다른 설계의 결과는 상관될 수 있다. 신뢰구간을 계산할 때 채널 단위로 짝을 유지한다. oracle은 시험 패킷을 본 뒤 전 배선 공통 위상을 고른 상한이다.',
            '', '|방법|고정 32 대비 회복 조합|새로 잃은 조합|grid 대비 통과율 차이(pp), 채널 bootstrap 95%|',
            '|---|---:|---:|---:|']
    for method in names:
        if method=='oracle':
            continue
        wins=losses=0
        diffs={}
        for r in data['rows']:
            a=int(r['methods'][method]['all_pass'])
            fixed=int(r['methods']['fixed32']['all_pass'])
            wins+=a>fixed
            losses+=a<fixed
            diffs.setdefault(r['channel'],[]).append(a-int(r['methods']['grid']['all_pass']))
        ds=np.array([np.mean(g) for g in diffs.values()])*100
        rng=np.random.default_rng(1709)
        lo,hi=np.quantile(rng.choice(ds,(4000,len(ds)),replace=True).mean(axis=1),[.025,.975])
        lines.append(f'|{method}|{wins}|{losses}|{ds.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]|')
    path.write_text('\n'.join(lines)+'\n')
    raw=[dict(channel=r['channel'],design=r['design'],energy=r['methods']['fixed32']['energy'],
              methods={k:[m['code'],m['passes'],round(m['margin'],6),m['errors']] for k,m in r['methods'].items()})
         for r in data['rows']]
    path.with_suffix('.json').write_text(json.dumps(dict(columns=['code','passed_packets','worst_margin','errors'],rows=raw),
                                                    separators=(',',':'))+'\n')


def sizing_report(data,path):
    pt=data['protocol']
    lane_header = '배선별 곡선 보정 통과|' if 'lane_quadratic' in pt['methods'] else ''
    lines=['# 새 회로 크기와 수신 위상 보정 독립 검증','',
        f"독립 채널 {pt['channels']}개, 채널별 보정 192비트/배선, 검증 패킷 4개 × {pt['bits']}비트/배선. 고유 설계 {len(pt['designs'])}개를 실제 ngspice로 {data['physical_calls']}회 측정했다.",
        '', f'|설계|고정 위상 통과 채널|3점 그리드 통과|포물선 보정 통과|{lane_header}공통 위상 사후 상한|그리드 통과 채널 평균 TX pJ/bit|',
        '|---|' + '---:|'*(len(pt['methods'])+1)]
    for name,idx in pt['members'].items():
        group=[r for r in data['rows'] if r['design']==idx]
        counts=[str(sum(r['methods'][m]['all_pass'] for r in group))+'/'+str(len(group)) for m in pt['methods']]
        good=[r['methods']['grid']['energy'] for r in group if r['methods']['grid']['all_pass']]
        energy=f'{np.mean(good):.6f}' if good else '—'
        lines.append('|'+ '|'.join([name,*counts,energy])+'|')
    lines+=['','|같은 seed의 BO 대비|함께 통과한 채널|그리드 에너지 차이 pJ/bit|추가 통과|새 실패|',
            '|---|---:|---:|---:|---:|']
    for name,idx in pt['members'].items():
        if not name.startswith(('bracket_s','mlp_local_s','reserve_s','midpoint_s')):
            continue
        baseline=pt['members']['bo_s'+name.rsplit('_s',1)[1]]
        arm={r['channel']:r['methods']['grid'] for r in data['rows'] if r['design']==idx}
        bo={r['channel']:r['methods']['grid'] for r in data['rows'] if r['design']==baseline}
        common=[arm[c]['energy']-bo[c]['energy'] for c in arm if arm[c]['all_pass'] and bo[c]['all_pass']]
        delta=f'{np.mean(common):+.6f}' if common else '—'
        gains=sum(arm[c]['all_pass'] and not bo[c]['all_pass'] for c in arm)
        losses=sum(bo[c]['all_pass'] and not arm[c]['all_pass'] for c in arm)
        lines.append(f'|{name}|{len(common)}|{delta}|{gains}|{losses}|')
    lines+=['','에너지 평균의 분모는 방법별로 통과한 채널이다. 같은 채널에서 에너지 차이를 해석하려면 두 방법이 함께 통과한 채널을 사용한다. 결과 JSON은 채널별 값을 포함한다.']
    path.write_text('\n'.join(lines)+'\n')
    path.with_suffix('.json').write_text(json.dumps(data,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['dev','test','sizing'])
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    data=read(a.input)
    if a.kind=='sizing':
        sizing_report(data,a.output)
    else:
        phase_report(data,a.output,'개발 채널 교차검증' if a.kind=='dev' else '새 채널 위상 보정 검증')
