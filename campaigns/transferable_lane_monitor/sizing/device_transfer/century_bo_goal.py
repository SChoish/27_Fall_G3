"""Check BO using the exact acquisition threshold for each requested FoM."""
from concurrent.futures import ThreadPoolExecutor
import json
from century_engine import enable
from century_run import trial, DOMAINS, SEEDS
from experiment import arguments

def main():
    args=arguments();enable()
    root=args.out/"century"/"bo95"
    root.mkdir(exist_ok=True)
    def run(job):
        name,seed=job
        file=root/f"{name}-bo-{seed}.json"
        if file.exists():return
        r=trial(name,DOMAINS[name],"bo",seed,args,goals=(.95,),bo_goal=.95)
        temp=file.with_suffix(".part")
        temp.write_text(json.dumps(r,allow_nan=False))
        temp.replace(file)
        print(name,seed,r["hits"],flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run,[(n,s) for s in SEEDS for n in DOMAINS]))

if __name__=="__main__":main()
