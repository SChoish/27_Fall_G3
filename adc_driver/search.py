"""Charged SPICE search and fixed source-data collection."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.stats import qmc

from circuit import (NAMES,LOW,HIGH,NOMINAL,PRIOR_OPAMP,ADC_REFERENCE,encode,decode,
                     evaluate_suite,cost,write_json,verify_runtime)


def fingerprint(contexts):
    payload=json.dumps([asdict(c) for c in contexts],sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def collect(n,seed,contexts,output,workers=4):
    if n<12: raise ValueError('Use at least 12 source designs for a split')
    verify_runtime()
    if Path(output).exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    u=qmc.LatinHypercube(len(NAMES),seed=seed).random(n)
    # The fixed reference designs are charged, and explicitly identified.
    u[0]=encode(NOMINAL);u[1]=encode(PRIOR_OPAMP);u[2]=encode(ADC_REFERENCE)
    start=time.perf_counter()
    def job(v): return evaluate_suite(decode(v),contexts)
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(job,u)):
            records.append(dict(index=i,u=u[i].tolist(),oracle=r))
            if (i+1)%8==0: print(json.dumps(dict(stage='collect',completed=i+1,n=n,seconds=time.perf_counter()-start)),flush=True)
    split=np.full(n,2,dtype=int);order=np.random.default_rng(seed+319).permutation(n)
    split[order[:int(.7*n)]]=0;split[order[int(.7*n):int(.85*n)]]=1
    for r,s in zip(records,split): r['split']=int(s)
    result=dict(seed=seed,contexts=[asdict(c) for c in contexts],context_hash=fingerprint(contexts),
                names=NAMES,records=records,design_queries=n,transient_queries=n*len(contexts),
                split_codes={'train':0,'validation':1,'test':2},elapsed_seconds=time.perf_counter()-start)
    write_json(output,result)
    return result


def initial(seed):
    rng=np.random.default_rng(seed+2718)
    return np.vstack([encode(NOMINAL),encode(PRIOR_OPAMP),encode(ADC_REFERENCE),rng.uniform(0,1,(5,len(NAMES)))])


def optimize(method,budget,seed,contexts,output,checkpoint=None,search_options=None):
    if budget<16 or budget%8: raise ValueError('Budget must be a multiple of 8, at least 16')
    if method not in ['random','ga','regression','rnc']: raise ValueError(method)
    surrogate=None
    if method in ['regression','rnc']:
        if checkpoint is None: raise ValueError('A matching trained checkpoint is required')
        from learning import LocalSearch
        surrogate=LocalSearch(checkpoint,contexts,search_options)
        if surrogate.payload['method']!=method: raise ValueError('Checkpoint method does not match search method')
    elif search_options is not None:
        raise ValueError('Attention/metric options apply only to regression or RNC search')
    verify_runtime()
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'trace.json').exists(): raise FileExistsError(f'Refusing to overwrite {folder}/trace.json')
    plan=dict(method=method,seed=seed,budget=budget,contexts=[asdict(c) for c in contexts],
              context_hash=fingerprint(contexts),parameter_names=NAMES,low=LOW.tolist(),high=HIGH.tolist(),
              initial_queries=8,initial_rule='Nominal + prior OPAMP + developed ADC reference + five random designs, all charged',
              prior_opamp_rule='Earlier OPAMP sizing is a documented warm start; its ADC evaluation is new',
              objective='Minimize mean driver energy per sample subject to worst hold error in each context',
              checkpoint=str(checkpoint) if checkpoint else None,
              checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if surrogate else None,
              search_options=asdict(surrogate.options) if surrogate else None,
              source_design_queries=surrogate.payload['source_design_queries'] if surrogate else 0,
              source_sha256=surrogate.payload['source_sha256'] if surrogate else None,
              source_data_cost_in_search_budget=False,
              ga=dict(population=8,offspring=8,crossover='SBX(prob=.9,eta=15)',
                      mutation='Polynomial(prob=1,prob_var=1/9,eta=20)',selection='binary tournament',
                      survival='constraint violation then energy'),
              full_budget_after_success=True)
    write_json(folder/'protocol.json',plan)
    rows=[];best=None;first=None;begin=time.perf_counter()
    def query(u,proposal_report=None):
        nonlocal best,first
        assert len(rows)<budget
        r=evaluate_suite(decode(u),contexts);value=cost(r)
        row=dict(query=len(rows)+1,u=np.asarray(u).tolist(),x=decode(u).tolist(),cost=value,oracle=r)
        if proposal_report is not None:row['proposal']=proposal_report
        if best is None or value<best['cost']: best=row
        if r['feasible'] and first is None: first=row['query']
        row['best_cost']=best['cost'];row['first_success']=first
        rows.append(row)
        # A completed query is durably checkpointed before proposing the next one.
        write_json(folder/'partial.json',dict(**plan,trace=rows))
        return r
    if method=='ga':
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        class Problem(ElementwiseProblem):
            def __init__(self): super().__init__(n_var=len(NAMES),n_obj=1,n_ieq_constr=1,xl=0.,xu=1.)
            def _evaluate(self,u,out,*args,**kwargs):
                r=query(u)
                out['F']=r['mean_driver_energy_pj'] if r['valid'] else 1e12
                out['G']=max(v['worst_hold_error_mv']/v['context']['error_limit_mv']-1
                             for v in r['contexts']) if r['valid'] else 1e6
        algorithm=GA(pop_size=8,n_offsprings=8,sampling=initial(seed),
                     crossover=SBX(prob=.9,eta=15),mutation=PM(prob=1.,prob_var=1/len(NAMES),eta=20),
                     eliminate_duplicates=True)
        result=minimize(Problem(),algorithm,('n_eval',budget),seed=seed,verbose=False)
        assert result.algorithm.evaluator.n_eval==len(rows)==budget
    else:
        rng=np.random.default_rng(seed+90101)
        for k in range(budget):
            if k<8: u=initial(seed)[k]
            elif surrogate: u=surrogate.propose(rows,rng)
            else: u=rng.uniform(0,1,len(NAMES))
            query(u,surrogate.last_report if surrogate and k>=8 else None)
    output_record=dict(**plan,trace=rows,best=best,first_success=first,
                       design_queries=len(rows),transient_queries=len(rows)*len(contexts),
                       elapsed_seconds=time.perf_counter()-begin)
    write_json(folder/'trace.json',output_record)
    write_json(folder/'best_design.json',dict(zip(NAMES,best['x'])))
    if surrogate:surrogate.save_attention(folder/'attention.pt')
    (folder/'partial.json').unlink(missing_ok=True)
    print(json.dumps(dict(stage='search',method=method,seed=seed,first_success=first,
                          error_mv=best['oracle']['worst_hold_error_mv'],
                          energy_pj=best['oracle']['mean_driver_energy_pj'])),flush=True)
    return output_record
