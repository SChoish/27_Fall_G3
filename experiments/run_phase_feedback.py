"""Acquire phase response curves on fixed sizing designs. One transient per packet.

All phase codes below are passive receiver measurements of a shared transient.
Run each output folder from only one process. Completed episodes can be resumed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import time

import numpy as np
from experiments.d2d_three_lane import d, make_case, measure_transient, transient
from coupled_link import phase_table

ROOT = Path(__file__).resolve().parents[1]
CODES = np.arange(8, 57, 4)
ARCHIVE = [1.3945718754465593,2.064068978908268,.822009391179969,2.3126135544533652,4.,21.308311930369285]


def designs():
    rows=json.loads((ROOT/'implementation_results/20261125_three_lane/results.json').read_text())['rows']
    return {**{f"bo{r['seed']}":r['design'] for r in rows if r['method']=='bo'}, 'archive':ARCHIVE}


def channel(root_seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([root_seed,index]))
    return dict(rate=rng.uniform(1.92,2.38), coupling=rng.uniform(.045,.105),
                skew=rng.uniform(-.18,.18), outer_skew=rng.uniform(-.05,.05),
                temp=rng.uniform(23,74), vdd=rng.uniform(.945,1.025),
                resistance=rng.uniform(26,44), cap=rng.uniform(.275,.355),
                inductance=rng.uniform(.18,.42) if index%2 else 0.)


def suite(root_seed, index, long_bits):
    settings=channel(root_seed,index)
    out=[]
    for k in range(7):
        packet_seed=int(np.random.SeedSequence([3197,root_seed,index,k]).generate_state(1)[0])
        c=make_case(settings['rate'], settings['coupling'], packet_seed,64 if k<3 else long_bits,
                    k in (0,2,4,6),settings['skew'])
        center=replace(c.context.lane0,temperature_c=settings['temp'],vdd=settings['vdd'],
            resistance_ohm=settings['resistance'],capacitance_pf=settings['cap'],
            inductance_nh=settings['inductance'])
        outer=replace(center,resistance_ohm=center.resistance_ohm+2,
                      clock_offset_ui=center.clock_offset_ui+settings['outer_skew'])
        out.append(replace(c,context=d.CoupledContext(center,outer,settings['coupling'])))
    return out


def curve(raw,case):
    base=measure_transient(raw,case)
    if not base['valid']:
        raise ValueError('Invalid transmitter waveform')
    margins,errors=[],[]
    for third,offset in ((False,3),(True,6)):
        pair=replace(case,third=third)
        wave=dict(t=raw[:,0],rx=raw[:,[2,2+offset]])
        m,e=phase_table(wave,case.context,pair.payload(),case.guard_ui)
        keep=[1] if third else [0,1]
        margins.extend(m[keep][:,CODES].tolist())
        errors.extend(e[keep][:,CODES].tolist())
    ref=CODES.tolist().index(32)
    np.testing.assert_allclose(np.asarray(margins)[:,ref],base['margins'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(np.asarray(errors)[:,ref],base['errors'])
    return dict(margin=margins,errors=errors,overshoot=base['overshoot_fraction'],
                energy=base['energy_pj_bit'], seed=case.seed,bits=case.bits,correlated=case.correlated)


def acquire(args):
    args.output.mkdir(parents=True,exist_ok=True)
    binary,env=d.link.simulator()
    chosen=designs()
    protocol=dict(root_seed=args.root_seed, channels=args.channels, long_bits=args.long_bits,
        designs=chosen,codes=CODES.tolist(),probe_bits=[64]*3,
        validation_packets=4,model_sha256=d.file_hash(d.LINK_ROOT/'assets/45nm_bulk.txt'),
        simulator_sha256=d.file_hash(binary), simulator_version=subprocess.check_output(
            [binary,'--version'],env=env,text=True).strip(),source_sha256={str(p.relative_to(ROOT)):d.file_hash(p)
            for p in (Path(__file__).resolve(),ROOT/'experiments/d2d_three_lane.py',Path(d.__file__),
                      d.LINK_ROOT/'coupled_link.py',d.LINK_ROOT/'link.py')})
    manifest=args.output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('Resume protocol differs')
    d.write_json(manifest,protocol)
    jobs=[(idx,name,x) for idx in range(args.channels) for name,x in chosen.items()
          if not (args.output/f'{idx:03d}-{name}.json').exists()]
    def run(job):
        idx,name,x=job
        start=time.monotonic()
        panel=suite(args.root_seed,idx,args.long_bits)
        responses=[curve(transient(x,c),c) for c in panel]
        d.write_json(args.output/f'{idx:03d}-{name}.json',
            dict(channel=idx,design=name,settings=channel(args.root_seed,idx),
                 probe=responses[:3],validation=responses[3:],seconds=time.monotonic()-start))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run,jobs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root-seed',type=int,required=True)
    p.add_argument('--channels',type=int,required=True)
    p.add_argument('--long-bits',type=int,default=256)
    p.add_argument('--workers',type=int,default=6)
    acquire(p.parse_args())
