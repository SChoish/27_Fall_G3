"""Optional compiled-module cache; use a fresh isolated simulator for every transient."""
import atexit
import json
from pathlib import Path
import select
import subprocess
import threading
from circuit import deck, MEAS
import re
import numpy as np

local=threading.local()
workers=[]
lock=threading.Lock()

def stop():
    for p in workers:
        if p.poll() is None:p.terminate()

atexit.register(stop)


def simulate(c,x,k,args):
    net,v=deck(c,x,k,getattr(args,"step_ps",1))
    model=(args.sky if c.pdk=="sky" else args.gf).resolve()
    wasm=args.wasm.resolve()
    key=(model,wasm,args.node)
    if getattr(local,"key",None)!=key or local.proc.poll() is not None:
        if hasattr(local,"proc") and local.proc.poll() is None:
            local.proc.terminate()
        script=Path(__file__).with_name("century_worker.mjs").resolve()
        cmd=[args.node,"--permission",*(f"--allow-fs-read={p}" for p in
             (script,model,wasm)),str(script),str(wasm),str(model)]
        local.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        local.key=key
        with lock:workers.append(local.proc)
    p=local.proc
    p.stdin.write(json.dumps(net)+"\n")
    p.stdin.flush()
    if not select.select([p.stdout],[],[],30)[0]:
        p.kill()
        raise RuntimeError("ngspice worker timed out")
    status,log=json.loads(p.stdout.readline())
    if status:
        raise RuntimeError(log[-600:])
    def read(name):
        match=re.search("^"+name+r"\s*=\s*([-+.\deE]+)",log,re.M)
        if match is None:raise RuntimeError(log[-600:])
        return float(match[1])
    energy=-read("charge")*v*1e15
    ports=[[read(m+str(i))*(1e12 if m in MEAS[:4] else 1/v) for m in MEAS] for i in range(len(c.outputs))]
    if energy<=0 or np.min(np.array(ports)[:,:4])<=0:raise RuntimeError("nonpositive timing or energy")
    return dict(energy=energy,ports=ports)


def enable():
    import warm_space
    warm_space.simulate=simulate


if __name__=="__main__":
    from experiment import arguments
    from century_run import DOMAINS
    from circuit import simulate as reference
    from time import monotonic
    args=arguments()
    times=[0.,0.]
    for name,c in DOMAINS.items():
        x=np.random.default_rng(935).uniform(.7,.98,2*len(c.parents))
        for k in range(4):
            records=[]
            for i,fn in enumerate([reference,simulate]):
                t=monotonic()
                records.append(fn(c,x,k,args))
                times[i]+=monotonic()-t
            assert records[0]==records[1],(name,k,records)
    print("24 exact corner records; isolated and cached seconds:",times,flush=True)
