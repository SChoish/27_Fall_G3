"""Measure optimizer CPU time while reproducing complete trajectories from the locked ledger."""
from collections import defaultdict
from functools import wraps
import json
from time import thread_time
from pathlib import Path

import alarm_pilot as pilot
from audit_alarm import ALL


class Replay:
    def __init__(self,name,records):
        self.name=name;self.c=pilot.TARGETS[name];self.records=records

    def pair(self,x,k):
        return [self.records[(tuple(x),j)] for j in (k,k+1)]


def main():
    args=pilot.exp.arguments()
    root=args.out/"alarm"
    times=defaultdict(float)
    def watch(owner,key,kind):
        original=getattr(owner,key)
        @wraps(original)
        def clock(*a,**kw):
            begin=thread_time()
            result=original(*a,**kw)
            times[kind]+=thread_time()-begin
            return result
        setattr(owner,key,clock)
    watch(pilot.Critic,"fit","fit")
    watch(pilot,"fit_attention","source_attention_fit")
    for owner,key in ((pilot,"local_search"),(pilot,"propose"),(pilot,"size_move"),
                      (pilot.exp,"pool"),(pilot,"choose_snapshot")):
        watch(owner,key,"proposal")
    models={s:pilot.train_source(args.out,s) for s in pilot.SEEDS}
    source=dict(times)
    counts=defaultdict(int);sums=defaultdict(lambda:defaultdict(float))
    traces=0
    for name in pilot.TARGETS:
        data={}
        for path in (root/"physical"/name).glob("*.json"):
            rec=json.loads(path.read_text())
            data[tuple(rec["x"]),rec["corner"]]=rec["raw"]
        replay=Replay(name,data)
        for method in ALL:
            for seed in ((21,) if method in ("role","role40") else pilot.SEEDS):
                lock=json.loads((root/f"{name}-{method}-{seed}.json").read_text())
                times.clear()
                start=thread_time()
                new=pilot.trial(replay,*models[seed],method,seed)
                total=thread_time()-start
                assert lock==new,(name,method,seed)
                traces+=1;counts[method]+=1
                for k,v in times.items():sums[method][k]+=v
                sums[method]["total"]+=total
    output=dict(traces_exact=traces,source_per_seed_seconds={k:v/len(models) for k,v in source.items()},
                per_trial_seconds={m:{k:v/counts[m] for k,v in vals.items()} for m,vals in sums.items()})
    (root/"timing.json").write_text(json.dumps(output,indent=2))
    print("Exact policy replays",traces)
if __name__=="__main__":main()
