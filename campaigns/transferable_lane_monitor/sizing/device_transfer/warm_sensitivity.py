"""Post-lock paired sensitivity surface; does not participate in selecting a winner."""
import json
from warm_study import Benchmark,write
from warm_policy import Config
from warm_space import TEST
from experiment import arguments

if __name__=="__main__":
    args=arguments();root=args.out/"warm"
    b=Benchmark(args,TEST,(31,32,33,34,35))
    b.runs=json.loads((root/"test_runs.json").read_text())
    for key in json.loads((root/"development_runs.json").read_text()):
        b.evaluate(Config(**json.loads(key)))
    write(root/"sensitivity_runs.json",b.runs)
