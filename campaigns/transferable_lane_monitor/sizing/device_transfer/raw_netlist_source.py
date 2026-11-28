"""Acquire matching GF180 source labels at the 100 already sampled SKY130 designs."""
import json
from pathlib import Path
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
from alarm_pilot import SOURCE_NETS
from century_engine import enable
from experiment import arguments
from warm_space import Store
from circuit import ratios

if __name__=="__main__":
    args=arguments();enable()
    root=args.out/"raw_netlist";root.mkdir(exist_ok=True)
    cached=args.out/"anonymous_width"/"source.json"
    if cached.is_file():
        sky=json.loads(cached.read_text())
    else:
        with ZipFile(Path(__file__).with_name("anonymous_width_measurements.zip")) as archive:
            sky=json.loads(archive.read("source.json"))
    def sample(r):
        c=replace(SOURCE_NETS[r["name"]],pdk="gf")
        store=Store("raw_source_gf_"+r["name"],c,args)
        raw=[store.pair(x,0) for x in r["x"]]
        assert all("error" not in v for pair in raw for v in pair)
        y=[ratios(pair,raw[0]).tolist() for pair in raw]
        return dict(name=r["name"],pdk="gf",x=r["x"],y=y,raw=raw)
    with ThreadPoolExecutor(4) as pool:gf=list(pool.map(sample,sky))
    (root/"source.json").write_text(json.dumps([*({**r,"pdk":"sky"} for r in sky),*gf]))
