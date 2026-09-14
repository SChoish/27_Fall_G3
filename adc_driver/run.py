"""Command-line entry point. Run `python run.py --help`."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path
import json

import numpy as np

from circuit import (ROOT,NAMES,NOMINAL,PRIOR_OPAMP,Context,load_contexts,
                     evaluate_suite,netlist,verify_runtime,write_json)


def design(path):
    if path is None:return NOMINAL.copy()
    data=json.loads(Path(path).read_text())
    if isinstance(data,dict):return np.array([data[n] for n in NAMES],dtype=float)
    return np.array(data,dtype=float)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Check ngspice and the bundled licensed SKY130 models')
    ev=sub.add_parser('evaluate',help='Run actual transistor transients and save waveforms')
    ev.add_argument('--design');ev.add_argument('--contexts');ev.add_argument('--output',default='results/evaluation')
    ev.add_argument('--dense',action='store_true',help='Reduce maximum time step by a factor of four')
    export=sub.add_parser('export',help='Write a self-contained, directly runnable SPICE netlist')
    export.add_argument('--design');export.add_argument('--contexts');export.add_argument('--output',default='examples/standalone.cir')
    collect=sub.add_parser('collect',help='Collect fixed, charged physical source data')
    collect.add_argument('--n',type=int,default=128);collect.add_argument('--seed',type=int,default=140926)
    collect.add_argument('--workers',type=int,default=4);collect.add_argument('--contexts')
    collect.add_argument('--output',default='data/source.json')
    fit=sub.add_parser('train',help='Train a fresh regression or RNC model for the ADC task')
    fit.add_argument('--data',required=True);fit.add_argument('--method',choices=['regression','rnc'],required=True)
    fit.add_argument('--seed',type=int,default=0);fit.add_argument('--epochs',type=int,default=150);fit.add_argument('--output',required=True)
    opt=sub.add_parser('optimize',help='Run search with every initial and failed SPICE query charged')
    opt.add_argument('--method',choices=['random','ga','regression','rnc'],required=True)
    opt.add_argument('--budget',type=int,default=32);opt.add_argument('--seed',type=int,default=0)
    opt.add_argument('--contexts');opt.add_argument('--checkpoint');opt.add_argument('--output',required=True)
    opt.add_argument('--attention',choices=['off','learned'],default='learned',help='Surrogate search only; learned coordinate metric by default')
    opt.add_argument('--line-search',choices=['none','armijo'],default='none',help='Surrogate search only; fixed projected steps by default')
    opt.add_argument('--attention-steps',type=int,default=40,help='Attention updates per paid-data residual refit')
    opt.add_argument('--step-size',type=float,default=.15,help='Metric direction length in normalized design coordinates')
    opt.add_argument('--search-steps',type=int,default=6,help='Metric steps per start before selecting one SPICE candidate')
    demo=sub.add_parser('demo',help='Evaluate two reference designs and run random/GA examples')
    demo.add_argument('--budget',type=int,default=16);demo.add_argument('--seed',type=int,default=0)
    demo.add_argument('--workers',type=int,default=2);demo.add_argument('--contexts');demo.add_argument('--output',default='results/demo')
    args=parser.parse_args()
    if args.command=='doctor':print(json.dumps(verify_runtime(),indent=2));return
    if args.command=='train':
        from learning import train
        print(json.dumps(train(args.data,args.method,args.seed,args.output,args.epochs),indent=2));return
    contexts=load_contexts(args.contexts)
    if args.command=='export':
        if len(contexts)!=1:raise ValueError('Export accepts exactly one context')
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(netlist(design(args.design),contexts[0],standalone=True));print(str(p.resolve()));return
    if args.command=='evaluate':
        if args.dense:contexts=[replace(c,maxstep_ns=c.maxstep_ns/4) for c in contexts]
        verify_runtime();r=evaluate_suite(design(args.design),contexts,args.output)
        write_json(Path(args.output)/'summary.json',r)
        from plots import waveform
        for i,(c,rr) in enumerate(zip(contexts,r['contexts'])):
            if rr['valid']:waveform(Path(args.output)/f'{i:02d}_{c.name}')
        print(json.dumps({k:v for k,v in r.items() if k!='contexts'},indent=2));return
    from search import collect,optimize
    if args.command=='collect':collect(args.n,args.seed,contexts,args.output,args.workers);return
    if args.command=='optimize':
        options=None
        if args.method in ['regression','rnc']:
            from attention import SearchOptions
            options=SearchOptions(attention=args.attention,line_search=args.line_search,
                                  attention_steps=args.attention_steps,step_size=args.step_size,
                                  search_steps=args.search_steps)
        optimize(args.method,args.budget,args.seed,contexts,args.output,args.checkpoint,options);return
    verify_runtime();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    from plots import waveform,searches
    for name,x in [('nominal',NOMINAL),('prior_opamp',PRIOR_OPAMP)]:
        r=evaluate_suite(x,contexts,out/name);write_json(out/name/'summary.json',r)
        for i,(c,rr) in enumerate(zip(contexts,r['contexts'])):
            if rr['valid']:waveform(out/name/f'{i:02d}_{c.name}')
    def job(method):return optimize(method,args.budget,args.seed,contexts,out/method)
    with ThreadPoolExecutor(max_workers=min(2,args.workers)) as pool:list(pool.map(job,['random','ga']))
    searches([out/m/'trace.json' for m in ['random','ga']],out/'searches.png')


if __name__=='__main__':main()
