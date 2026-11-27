"""Run target-directed acquisitions against the same frozen suites and source cost."""
from argparse import Namespace, ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from campaign import run

if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--wasm",type=Path)
    p.add_argument("--node",default="node")
    p.add_argument("--ngspice",default="ngspice")
    args=p.parse_args()
    def evaluate(seed):
        for stages,label in [(2,"target_cold"),(4,"target_cold"),(4,"target_transfer")]:
            out=args.dir/label
            out.mkdir(exist_ok=True)
            src = None
            if label=="target_transfer":
                src=json.loads((args.dir/f"s2_kernel_{seed}.json").read_text())["rows"]
            run(stages,seed,"target_bo",Namespace(**vars(args),out=out),src)
    with ThreadPoolExecutor(3) as executor:
        list(executor.map(evaluate,range(11,17)))
