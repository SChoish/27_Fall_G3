"""Parameterized fanouts and coupled lanes with independent P/N widths."""
from pathlib import Path
from dataclasses import dataclass
import json
import re
import tempfile
import subprocess
import numpy as np

@dataclass(frozen=True)
class Circuit:
    pdk: str
    parents: tuple  # negative IDs encode distinct signal inputs
    sizes: tuple
    outputs: tuple
    loads: tuple
    coupling: float = 0  # mutual capacitance in fF

CIRCUITS={
    "short":Circuit("sky",(-1,0),(1,2),(1,),(1,)),
    "chain":Circuit("sky",(-1,0,1,2),(1,2,2,4),(3,),(1,)),
    "source_fork":Circuit("sky",(-1,0,1,2,2),(1,2,2,4,2),(3,4),(1,.65)),
    "fork":Circuit("sky",(-1,0,1,2,1,4),(1,2,2,4,1.5,3),(3,5),(.85,1.35)),
    "pair_sky":Circuit("sky",(-1,0,-2,2),(1,2,1,2),(1,3),(1,.8),2.0),
    "pair_gf":Circuit("gf",(-1,0,-2,2),(1,2,1,2),(1,3),(1,.8),2.0),
    "mixed_gf":Circuit("gf",(-1,0,-2,2,3,4),(1,2,1,2,2,4),(1,5),(1.35,.8),3.0),
}
SOURCES=("short","chain","source_fork")
TARGETS=("fork","pair_sky","pair_gf","mixed_gf")
MEAS=("rise","fall","sr","sf","hi","lo")
BASE=Path(__file__).resolve().parents[1]

def deck(c,x,k):
    nominal=1.8 if c.pdk=="sky" else 3.3
    v=nominal*(1,1.66/1.8,1, .90)[k]
    temp=(27,27,70,70)[k]
    cl=(8,8,12,12)[k]
    lines=["device-level buffer transfer",'.include "/model.spice"',f".temp {temp}",f"Vdd dd 0 {v}"]
    for root in sorted({-p for p in c.parents if p<0}):
        a,b=(v,0) if root==2 and k<2 else (0,v)
        lines.append(f"Vin{root} in{root} 0 pulse({a} {b} .5n 10p 10p 1n 2n)")
    for i,(p,w) in enumerate(zip(c.parents,c.sizes)):
        upstream=f"in{-p}" if p<0 else f"n{p}"
        for j,(pol,rail,scale) in enumerate([("n","0",1),("p","dd",2)]):
            width=w*scale*x[2*i+j]
            typ="M" if c.pdk=="gf" else "X"
            spec=(f"{pol}mos_3p3 w={width:.9f}u l=.28u" if c.pdk=="gf" else
                  f"sky130_fd_pr__{pol}fet_01v8 w={width:.9f} l=.15")
            lines.append(f"{typ}{pol}{i} n{i} {upstream} {rail} {rail} {spec}")
    for i,load in zip(c.outputs,c.loads):
        lines.append(f"Cload{i} n{i} 0 {cl*load}f")
    if c.coupling:
        a,b=c.outputs[0],c.outputs[-1]
        lines.append(f"Cmut n{a} n{b} {c.coupling}f")
    lines+=[".tran 1p 4.49n",".meas tran charge integ i(Vdd) from=2.45n to=4.45n"]
    for o,i in enumerate(c.outputs):
        p=i
        depth=0
        while p>=0:
            depth+=1
            p=c.parents[p]
        assert depth%2==0
        root=-p
        # The inverted input pulse starts high: synchronize all edge events by TD,
        # measuring the first occurrence of the desired polarity *after* 2.4 ns.
        inv=root==2 and k<2
        hi,lo=("4.45n","3.43n") if inv else ("3.43n","4.45n")
        lines.extend([f".meas tran hi{o} find v(n{i}) at={hi}",
                      f".meas tran lo{o} find v(n{i}) at={lo}"])
        for edge,name,a,b in [("rise","sr",.2,.8),("fall","sf",.8,.2)]:
            trig=f"td=2.4n {edge}=1"
            lines.extend([f".meas tran {edge}{o} trig v(in{root}) val={v/2} {trig} targ v(n{i}) val={v/2} {trig}",
                          f".meas tran {name}{o} trig v(n{i}) val={v*a} {trig} targ v(n{i}) val={v*b} {trig}"])
    return "\n".join(lines+[".end",""]),v

def simulate(c,x,k,args):
    net,v=deck(c,x,k)
    model=(args.sky if c.pdk=="sky" else args.gf).resolve()
    wasm=args.wasm.resolve()
    with tempfile.TemporaryDirectory() as folder:
        path=(Path(folder)/"design.cir").resolve()
        path.write_text(net)
        cmd=[args.node,"--permission",*(f"--allow-fs-read={p}" for p in
             (BASE,wasm.parent,model.parent,path.parent)),str(BASE/"wasm_eval.mjs"),str(wasm),str(model),str(path)]
        try:
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            return {"error":"timeout"}
        def read(key):
            match=re.search(r"^"+key+r"\s*=\s*([-+.\deE]+)",run.stdout,re.M)
            if not match:
                raise ValueError(key)
            return float(match[1])
        try:
            if run.returncode: raise ValueError("ngspice")
            energy=-read("charge")*v*1e15
            ports=[[read(m+str(o))*(1e12 if m in MEAS[:4] else 1/v) for m in MEAS] for o in range(len(c.outputs))]
            if energy<=0 or np.min(np.array(ports)[:,:4])<=0: raise ValueError("timing")
            return dict(energy=energy,ports=ports)
        except ValueError as e:
            return dict(error=str(e),log=(run.stdout+run.stderr)[-600:])

def ratios(rows,refs):
    if any("error" in r for r in rows): return np.full(5,3.)
    p,b=[np.array([r["ports"] for r in side]) for side in [rows,refs]]
    e=np.array([r["energy"]/s["energy"] for r,s in zip(rows,refs)])
    edp=e[:,None]*p[:,:,:2].max(2)/b[:,:,:2].max(2)
    shift=np.abs(p[0,:,:2]-p[1,:,:2]).max(1)
    base_shift=np.abs(b[0,:,:2]-b[1,:,:2]).max(1)
    return np.log([edp.max(),(p[:,:,:2]/b[:,:,:2]).max()/1.10,
                   (p[:,:,2:4]/b[:,:,2:4]).max()/1.10,(shift/base_shift).max()/1.05,
                   max((.9/np.maximum(p[:,:,4],1e-6)).max(),(p[:,:,5]/.1).max())])
