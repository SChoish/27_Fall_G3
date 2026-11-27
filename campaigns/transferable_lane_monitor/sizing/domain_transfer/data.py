"""Materialize a small physics-labelled benchmark, distinct from online algorithm cost."""
import argparse
import itertools
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import re
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import campaign

GRID=list(itertools.product([.64,.76,.88,.96,1.],[.65,.76,.85,.94,1.1]))
DOMAINS={"sky2":("sky",2,1),"sky4":("sky",4,1),"sky2_light":("sky",2,.75),"sky4_light":("sky",4,.75),"lane":("sky",4,2),
         "wire":("sky",4,1),"pdk":("gf",4,1),"both":("gf",4,2),"topology":("gf",6,2)}
WIDTHS={2:[1,2],4:[1,2,2,4],6:[1,1,2,2,3,4]}
base_deck=campaign.deck


def make_deck(pdk,stages,x,corner,model,step_ps=1,wire=False):
    # Same measurement harness, with a variable-length non-inverting chain.
    harness=base_deck(2,(1.,1.),corner,model,step_ps)
    harness=(harness.replace("3.8n","4.45n").replace("from=1.75n to=3.75n","from=2.40n to=4.40n")
             .replace("at=3.79n","at=4.39n"))
    body=[]
    for i,w in enumerate(np.array(WIDTHS[stages])*np.geomspace(*x,stages)):
        prev=f"n{i}" if i else "in"
        out=f"n{i+1}" if i<stages-1 else ("driver" if wire else "out")
        for pol,factor,rail in (("n",1,"0"),("p",2,"dd")):
            if pdk=="gf":
                line=f"M{pol}{i} {out} {prev} {rail} {rail} {pol}mos_3p3 w={w*factor:.9f}u l=.28u"
            else:
                line=f"X{pol}{i} {out} {prev} {rail} {rail} sky130_fd_pr__{pol}fet_01v8 w={w*factor:.9f} l=.15"
            body.append(line)
    harness,count=re.subn(r"^Xn0.*?(?=^Cload)","\n".join(body)+"\n",harness,flags=re.M|re.S)
    assert count==1 and len(body)==2*stages
    if wire:
        capacitance=corner[1]/2
        harness=harness.replace(f"Cload out 0 {corner[1]}f",
                                f"Cnear driver 0 {capacitance}f\nRline driver out 120\nCfar out 0 {capacitance}f")
        assert "Rline driver out 120" in harness
    return harness



def corners(pdk,load):
    return [(v*(3.3/1.8 if pdk=="gf" else 1),c*load,t) for v,c,t in campaign.CORNERS]


def collect(name,args):
    pdk,stages,load=DOMAINS[name]
    campaign.deck=lambda *parts:make_deck(pdk,*parts,wire=name=="wire")
    args.model=args.gf if pdk=="gf" else args.sky
    conditions=corners(pdk,load)
    inputs=[((1.,1.),c) for c in conditions]+[(x,c) for x in GRID for c in conditions]
    with ThreadPoolExecutor(args.workers) as pool:
        rows=list(pool.map(lambda v:campaign.transient(stages,*v,args),inputs))
    for i,r in enumerate(rows):
        if "error" in r or min(r["energy_fJ"],r["rise"],r["fall"],r["slew_rise"],r["slew_fall"])<=0:
            raise ValueError((name,i,r))
    result=dict(name=name,pdk=pdk,stages=stages,load=load,corners=conditions,
                reference=rows[:4],designs=[dict(x=x,raw=rows[4+i*4:8+i*4]) for i,x in enumerate(GRID)])
    path=args.out/(name+".json")
    path.write_text(json.dumps(result,separators=(",",":")))
    print(name,len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("sky","gf","wasm","out"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--workers",default=4,type=int)
    p.add_argument("--names",nargs="+",default=list(DOMAINS),choices=DOMAINS)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    for name in args.names:
        if (args.out/(name+".json")).exists():
            raise FileExistsError(name)
        collect(name,args)
