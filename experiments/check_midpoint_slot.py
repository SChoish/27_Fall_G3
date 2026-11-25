"""A one-seed manufacturing pilot with a geometric midpoint replacing the reserve slot."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from experiments.check_sizing_shortlist import shortlist
from experiments.run_phase_sizing import curve,d,oracle,rank,suite,transient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('reserve',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    proto=json.loads((args.source/'protocol.json').read_text())
    if proto['seed']!=21:
        raise ValueError('Prespecified pilot seed is 21')
    trace=json.loads((args.source/'bracket.json').read_text())
    picks=shortlist(trace['rows'])
    midpoint=np.sqrt(np.asarray(picks[1]['x'])*np.asarray(picks[2]['x'])).tolist()
    cases=[replace(suite(proto['root_seed'],i,128)[0],seed=900010+100*i+k,bits=128,correlated=k==0)
           for i in proto['channels'] for k in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        packets=list(pool.map(lambda c:curve(transient(midpoint,c),c),cases))
    middle=dict(x=midpoint,oracle=oracle([packets[i:i+2] for i in range(0,len(cases),2)]))
    reused={tuple(r['x']):r for r in trace['validation']}
    checked=[reused[tuple(picks[0]['x'])],middle,reused[tuple(picks[2]['x'])]]
    locks=json.loads((args.reserve/'locked.json').read_text())
    locks['midpoint']=min(checked,key=rank)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    d.write_json(out/'protocol.json',dict(seed=21,checked=checked,short_calls=72,long_calls=24,
                incremental_transients=8,source=d.file_hash(__file__)))
    d.write_json(out/'locked.json',locks)


if __name__=='__main__':
    main()
