"""Cost-to-target benchmark for transistor-level buffer sizing. Requires NumPy and ngspice."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent
GOALS = (.95, .90, .80)
CORNERS = [(1.80,8,27), (1.66,8,27), (1.80,12,70), (1.62,12,70)]
NOMINAL = [1., 1.]
CAP = 40


def deck(stages, x, corner, model, step_ps=1):
    v, load, temp = corner
    widths = np.array([1,2] if stages == 2 else [1,2,2,4])*np.geomspace(*x, stages)
    lines = ["SKY130 buffer width search", f'.include "{model}"', f".temp {temp}",
             f"Vdd dd 0 {v}", f"Vin in 0 pulse(0 {v} 0.5n 10p 10p 1n 2n)"]
    upstream = "in"
    for i, width in enumerate(widths):
        node = "out" if i == stages-1 else f"internal{i}"
        lines.extend([f"Xn{i} {node} {upstream} 0 0 sky130_fd_pr__nfet_01v8 w={width:.9f} l=.15",
                      f"Xp{i} {node} {upstream} dd dd sky130_fd_pr__pfet_01v8 w={2*width:.9f} l=.15"])
        upstream = node
    lines.extend([f"Cload out 0 {load}f", f".tran {step_ps:g}p 3.8n",
                  ".meas tran charge integ i(Vdd) from=1.75n to=3.75n",
                  ".meas tran high find v(out) at=3.40n",
                  ".meas tran low find v(out) at=3.79n"])
    for edge, a, b in [("rise",.2,.8),("fall",.8,.2)]:
        lines.extend([f".meas tran {edge} trig v(in) val={v/2} {edge}=2 targ v(out) val={v/2} {edge}=2",
                      f".meas tran slew_{edge} trig v(out) val={v*a} {edge}=2 targ v(out) val={v*b} {edge}=2"])
    return "\n".join(lines+[".end",""])


def transient(stages, x, corner, args):
    model = args.model.resolve()
    with tempfile.TemporaryDirectory() as temp:
        path = (Path(temp)/"design.cir").resolve()
        wasm = args.wasm is not None
        path.write_text(deck(stages,x,corner,"/model.spice" if wasm else model, getattr(args,"step_ps",1)))
        if wasm:
            module = args.wasm.resolve()
            cmd = [args.node,"--permission",*(f"--allow-fs-read={p}" for p in
                    (ROOT,module.parent,model.parent,path.parent)),str(ROOT/"wasm_eval.mjs"),
                   str(module),str(model),str(path)]
        else:
            cmd = [args.ngspice,"-b",str(path)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except subprocess.TimeoutExpired:
            return {"error":"timeout"}
        result = {}
        for name in ("rise","fall","slew_rise","slew_fall","charge","high","low"):
            m = re.search(r"^"+name+r"\s*=\s*([-+.\deE]+)",p.stdout,re.M)
            if m is None or p.returncode:
                return {"error":(p.stdout+p.stderr)[-1000:]}
            result[name] = float(m[1])
        result["energy_fJ"] = -corner[0]*result.pop("charge")*1e15
        for name in ("rise","fall","slew_rise","slew_fall"):
            result[name] *= 1e12
        return result


def metrics(pair, reference, corners):
    """Log ratios: EDP, edge delay/limit, slew/limit, low-V shift/limit, swing/limit."""
    if any("error" in row for row in pair):
        return np.full(5, 4.)
    def matrix(rows, names):
        return np.array([[r[n] for n in names] for r in rows])
    delay, ref_delay = (matrix(rows,("rise","fall")) for rows in (pair,reference))
    slew, ref_slew = (matrix(rows,("slew_rise","slew_fall")) for rows in (pair,reference))
    energy, ref_energy = (matrix(rows,("energy_fJ",))[:,0] for rows in (pair,reference))
    if min(delay.min(),slew.min(),energy.min()) <= 0:
        return np.full(5, 4.)
    swing = max(max(.9*c[0]/max(r["high"],1e-10),r["low"]/(.1*c[0])) for c,r in zip(corners,pair))
    values = [max(energy*delay.max(axis=1)/(ref_energy*ref_delay.max(axis=1))),
              (delay/ref_delay).max()/1.1, (slew/ref_slew).max()/1.1,
              abs(delay[1]-delay[0]).max()/(1.05*abs(ref_delay[1]-ref_delay[0]).max()), swing]
    return np.log(np.maximum(values,1e-10))


def kernel(a,b):
    u = (a[:,None,:]-b[None,:,:])/np.array([.23,.40])
    return np.exp(-.5*(u*u).sum(axis=2))


def predict(x,y,candidates):
    k = kernel(x,x) + .005*np.eye(len(x))
    cross = kernel(candidates,x)
    mean = y.mean(axis=0)
    pred = mean + cross @ np.linalg.solve(k,y-mean)
    var = np.maximum(1.-(cross*np.linalg.solve(k,cross.T).T).sum(axis=1),0)
    return pred,np.sqrt(var)


def rank(y):
    # Scores use only development EDP and hard-constraint excesses.
    return y[:,0] + 6*np.maximum(y[:,1:],0).sum(axis=1)


def proposal(method, rng, x, y, source):
    if len(x)==1 or method=="uniform":
        return rng.uniform([.5,.5],[1.,1.6])
    if method=="target_bo":
        from target_bo import target_proposal
        return target_proposal(rng,x,y,source)
    if method=="sweep":
        mesh = np.array([[a,b] for a in (.55,.7,.85,1.) for b in (.55,.75,1.,1.3,1.6)])
        remaining = mesh[np.min(np.linalg.norm(mesh[:,None,:]-x,axis=2),axis=1)>1e-6]
        return remaining[0]
    candidates = rng.uniform([.5,.5],[1.,1.6],size=(512,2))
    if source is not None:
        sx,sy = source
        if method=="replay":
            for j in np.argsort(rank(sy)):
                if np.min(np.linalg.norm(x-sx[j],axis=1))>1e-6:
                    return sx[j]
        else:
            candidates = np.vstack([candidates,sx])
            candidates = candidates[np.min(np.linalg.norm(candidates[:,None,:]-x,axis=2),axis=1)>1e-6]
            prior,_ = predict(sx,sy,candidates)
            fitted,_ = predict(sx,sy,x)
            residual,sd = predict(x,y-fitted,candidates)
            return candidates[np.argmin(rank(prior+residual)-.06*sd)]
    pred,sd = predict(x,y,candidates)
    return candidates[np.argmin(rank(pred)-.06*sd)]


def run(stages,seed,method,args,source_rows=None):
    rng = np.random.default_rng(seed)
    source = None if source_rows is None else (
        np.array([r["x"] for r in source_rows]),
        np.array([r["dev_log_ratios"] for r in source_rows]))
    refs = [transient(stages,NOMINAL,c,args) for c in CORNERS]
    if any("error" in r for r in refs):
        raise RuntimeError(f"Reference failed: {refs}")
    calls = 4
    row0 = dict(x=NOMINAL,dev=refs[:2],verify=refs[2:],calls=calls,
                dev_log_ratios=metrics(refs[:2],refs[:2],CORNERS[:2]).tolist())
    rows = [row0]
    hits = {}
    while calls+2<=CAP and len(hits)<len(GOALS):
        x = proposal(method,rng,np.array([r["x"] for r in rows]),
                     np.array([r["dev_log_ratios"] for r in rows]),source)
        dev = [transient(stages,x,c,args) for c in CORNERS[:2]]
        calls += 2
        d = metrics(dev,refs[:2],CORNERS[:2])
        row = dict(x=x.tolist(),dev=dev,dev_log_ratios=d.tolist())
        if d[0]<=np.log(.95) and max(d[1:])<=0 and calls+2<=CAP:
            val = [transient(stages,x,c,args) for c in CORNERS[2:]]
            calls += 2
            v = metrics(val,refs[2:],CORNERS[2:])
            row.update(verify=val,verify_log_ratios=v.tolist())
            for g in GOALS:
                if g not in hits and max(d[0],v[0])<=np.log(g) and max(v[1:])<=0:
                    hits[g] = calls
        rows.append(dict(row,calls=calls))
    record = dict(stages=stages,method=method,seed=seed,physical_calls=calls,
                  source_calls=source_rows[-1]["calls"] if source_rows else 0,hits=hits,rows=rows)
    path = args.out/f"s{stages}_{method}_{seed}.json"
    with path.open("x") as f:
        json.dump(record,f,separators=(",",":"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--ngspice",default="ngspice")
    p.add_argument("--wasm",type=Path)
    p.add_argument("--node",default="node")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    def fit(seed):
        source = run(2,seed,"kernel",args)
        for stages,method in [(2,"uniform"),(2,"sweep"),(4,"uniform"),(4,"sweep"),(4,"kernel"),
                              (4,"replay"),(4,"residual")]:
            run(stages,seed,method,args,source if method in ("replay","residual") else None)
        print("finished seed",seed,flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(fit,range(11,17)))


if __name__=="__main__":
    main()
