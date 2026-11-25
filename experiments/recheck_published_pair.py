"""Cold rerun of a discordant 512-bit BO/MLP pair from the committed summary alone."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from experiments.analyze_phase_feedback import choose,score
from experiments.run_phase_feedback import CODES,curve,d,suite,transient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    source=args.results/'SIZING_KO.json'
    data=json.loads(source.read_text())
    proto=data['protocol']
    by_id={(row['channel'],row['design']):row['methods']['grid'] for row in data['rows']}
    ids=[proto['members'][name] for name in ('bo_s22','mlp_local_s22')]
    chan=next(i for i in range(proto['channels']) if by_id[i,ids[0]]['all_pass']!=by_id[i,ids[1]]['all_pass'])
    cases=suite(proto['root_seed'],chan,proto['bits'])

    def repeat(idx):
        x=proto['designs'][idx]
        probes=[curve(transient(x,c),c) for c in cases[:3]]
        phase,_=choose('grid',[],lambda step,code:np.asarray(probes[step]['margin'])[:,code])
        valid=[curve(transient(x,c),c) for c in cases[3:]]
        return dict(code=int(CODES[phase]),**score(dict(validation=valid),phase))

    with ThreadPoolExecutor(max_workers=2) as pool:
        verified=list(pool.map(repeat,ids))
    lines=['# 원격 체크아웃의 물리 재현 검증','',
           f"저장된 주 시험에서 seed 22 BO와 국소 MLP의 grid 통과 여부가 처음 달라진 곳은 채널 번호 {chan}이다. 채널 seed({proto['root_seed']}) 및 설계 벡터로 보정용 64비트 세 패킷과 검증용 512비트 네 패킷을 처음부터 재생성했다. 체크포인트나 기존 파형을 사용하지 않았다.",
           '', '|회로|위상|통과 패킷|오류 수|최악 마진|TX pJ/bit|', '|---|---:|---:|---:|---:|---:|']
    records=[]
    for name,idx,fresh in zip(('BO','국소 MLP'),ids,verified):
        old=by_id[chan,idx]
        for key in ('code','passes','all_pass','errors'):
            if old[key]!=fresh[key]:
                raise ValueError(f'Reproduction differs: {name} {key}')
        for key in ('margin','overshoot','energy'):
            np.testing.assert_allclose(old[key],fresh[key],atol=1e-9,rtol=0)
        lines.append(f"|{name}|{fresh['code']}|{fresh['passes']}/4|{fresh['errors']}|{fresh['margin']:.8f}|{fresh['energy']:.8f}|")
        records.append(dict(design=name,original=old,repeated=fresh))
    lines+=['','두 설계의 위상·오류 수·통과 여부가 원본과 일치했고, 마진·overshoot·에너지는 절대오차 10⁻⁹ 이내였다. 이 검산에 14회가 들었다. 같은 시험의 재측정이므로 통과율의 표본 수는 늘지 않는다.']
    (args.results/'COLD_RERUN_KO.md').write_text('\n'.join(lines)+'\n')
    d.write_json(args.results/'COLD_RERUN_KO.json',dict(channel=chan,source_sha256=d.file_hash(source),
                                                        physical_calls=14,records=records))


if __name__=='__main__':
    main()
