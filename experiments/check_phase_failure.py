"""One-factor ngspice counterfactuals for the first RC and first RLC development channels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from experiments.diagnose_phase_limits import classify
from experiments.run_phase_feedback import curve,d,designs,suite,transient
from experiments.run_three_lane import HIGH_DRIVE


def run(dev,output,workers):
    protocol=json.loads((dev/'protocol.json').read_text())
    rows=[]
    for index in (0,1):
        baseline=json.loads((dev/f'{index:03d}-bo11.json').read_text())
        rows.append(dict(intervention='original',**classify(baseline)))
        original_cases=suite(protocol['root_seed'],index,protocol['long_bits'])[3:]
        for name in ('half_coupling','ten_percent_slower','high_drive'):
            x=HIGH_DRIVE if name=='high_drive' else designs()['bo11']
            cases=[]
            for c in original_cases:
                ctx=c.context
                if name=='half_coupling':
                    ctx=replace(ctx,coupling_pf=ctx.coupling_pf*.5)
                if name=='ten_percent_slower':
                    ctx=replace(ctx,lane0=replace(ctx.lane0,rate_gbps=ctx.lane0.rate_gbps*.9),
                                lane1=replace(ctx.lane1,rate_gbps=ctx.lane1.rate_gbps*.9))
                cases.append(replace(c,context=ctx))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                packets=list(pool.map(lambda c:curve(transient(x,c),c),cases))
            e=dict(channel=index,design=name if name=='high_drive' else 'bo11',validation=packets)
            rows.append(dict(intervention=name,**classify(e)))
    output.mkdir(parents=True,exist_ok=True)
    d.write_json(output/'sensitivity.json',dict(dev_seed=protocol['root_seed'],physical_new_transients=24,
                 bits=protocol['long_bits'],rows=rows))
    lines=['# 첫 두 개발 채널의 단일 요인 실험','',
           '기존 bo11 설계와 같은 네 개의 256비트 검증 패킷, 같은 수신 판정 조건을 사용했다. 기준 파형은 개발 캠페인에서 재사용했다. 나머지 24개 ngspice 파형은 새로 계산했다. 각 마진은 네 패킷과 세 배선 중 최솟값이다.',
           '', '|채널|변경|고정 위상 마진|최선 공통 위상 마진|평균 TX pJ/bit|','|---|---|---:|---:|---:|']
    labels={'original':'기준','half_coupling':'이웃 결합 절반','ten_percent_slower':'비트율 10% 감소','high_drive':'고구동 기준 설계로 교체'}
    for r in rows:
        lines.append(f"|{r['channel']}|{labels[r['intervention']]}|{r['fixed']:.4f}|{r['shared']:.4f}|{r['energy']:.5f}|")
    (output/'SENSITIVITY_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    run(a.dev,a.output,a.workers)
