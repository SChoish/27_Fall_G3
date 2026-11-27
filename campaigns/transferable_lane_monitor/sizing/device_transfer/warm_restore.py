"""Reconstruct the archived ledger in a new directory, then audit it without SPICE."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
from audit import checks
from warm_audit import CIRCUITS,audit,gather
from warm_space import bank
from experiment import upper

def restore(dest):
    root=dest/"warm";root.mkdir(parents=True,exist_ok=False)
    with ZipFile("warm_traces.zip") as z:
        for name in z.namelist():
            if len(Path(name).parts)!=1 or Path(name).suffix not in (".json",".csv"):
                raise ValueError(name)
            (root/name).write_bytes(z.read(name))
    with ZipFile("warm_spice.zip") as z:
        for member in z.namelist():
            finer=member=="finer.jsonl"
            if not finer and member not in {f"raw/{n}.jsonl" for n in CIRCUITS}:raise ValueError(member)
            sub=root/"fine" if finer else root/"raw"/Path(member).stem
            sub.mkdir(parents=True)
            for line in z.read(member).splitlines():
                v=json.loads(line)
                payload=[v["name"],v["x"]] if finer else [v["x"],v["k"]]
                key=sha256(json.dumps(payload).encode()).hexdigest()
                (sub/(key+".json")).write_bytes(line)
    before=json.loads((root/"audit.json").read_text())
    raw={}
    for f in (root/"fine").glob("*.json"):
        v=json.loads(f.read_text());raw[v["name"],tuple(v["x"])]=v["raw"]
    fine={}
    for (n,x),rs in raw.items():
        ref=raw[n,tuple(np.ones(len(upper(CIRCUITS[n]))))]
        ck=[checks(rs[k:k+2],ref[k:k+2]) for k in (0,2)]
        fine[n,x]=all(ok for ok,_ in ck),max(v for _,v in ck)
    audit(SimpleNamespace(out=dest),gather(root),{n:bank(c)[0] for n,c in CIRCUITS.items()},fine)
    after=json.loads((root/"audit.json").read_text())
    assert before==after
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("new_directory",type=Path);restore(p.parse_args().new_directory)
