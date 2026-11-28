"""Paired seven-method sizing with actual SPICE and a 20-lane link gate."""
import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np

from audit import checks
from century_engine import simulate
from circuit import CIRCUITS, ratios
from experiment import upper
from link_sizing_panel import LINK_ALLOWANCE, REGIMES, delay_swing, link_residual, panels
from published_baselines_run import (ARMS, ARCHIVE, ROOT, digest, get_kato_donors, load_sources,
                                     mapped_local, models, prepare, rank_candidates, read, templates, write)


GOAL, CAP = .95, 100
CIRCUIT = CIRCUITS["short"]
TRANSFER = ("mlp200","random_templates","retrieved")
SEEDS = (26112910,26112911,26112912,26112913,26112914)


class Spice:
    """Physically execute an identical transient once; charge every standalone trial."""

    def __init__(self, args):
        self.args = args
        self.entries = {}
        self.lock = Lock()

    def measure(self, x, corner, step):
        design = tuple(float(v) for v in x)
        key = (design,corner,step)
        with self.lock:
            prior = self.entries.get(key)
            if prior is None:
                owner = Future()
                self.entries[key] = owner
        if prior is not None:
            return prior.result(), True
        try:
            arguments = copy(self.args)
            arguments.step_ps = step
            try:
                raw = simulate(CIRCUIT,x,corner,arguments)
            except RuntimeError as error:
                raw = {"error":str(error)[-500:]}
            owner.set_result(raw)
            return raw, False
        except Exception as error:
            owner.set_exception(error)
            raise


def policy_factory(source):
    external = tuple(a for a in ARMS if a not in TRANSFER)
    targets = [("tx20",CIRCUIT,SEEDS)]
    donor = get_kato_donors(source,targets)
    create = prepare(targets,external,source,donor)
    denoiser = models(source)["gf"]
    banks = {a:templates(source["gf"],CIRCUIT,random=a=="random_templates") for a in TRANSFER[1:]}

    def choose(arm, log):
        if arm not in TRANSFER:
            return create(arm,CIRCUIT,GOAL,log)

        def proposal(_arm,hist,values,lo,hi,seed,_prior):
            z = np.asarray(hist)
            y = np.asarray(values)
            rng = np.random.default_rng([seed,len(hist),6481])
            if len(hist)<=4:
                return rng.uniform(lo,hi)

            def calibrated(obs,candidates,labels,_kernel):
                def predict(points):
                    cell = denoiser.readout(denoiser.encode(points))
                    # Source archive contains five cell outputs, no link labels.
                    return labels[0]+np.c_[cell,np.zeros(len(points))]
                return predict(obs), predict(candidates)

            bank = banks.get(arm)
            local = None
            if bank is not None:
                perm = np.random.default_rng([seed,886]).permutation(z.shape[1])
                local = mapped_local(bank[:,perm])
            return rank_candidates(z,y,lo,hi,rng,calibrated=calibrated,local_builder=local)
        return proposal
    return choose, {a:(2*len(donor["tx20"]["x"]) if a=="kato" else 200 if a in TRANSFER else 0)
                    for a in ARMS}


def measured(panel, raw):
    if any("error" in r for r in raw):
        return None
    return panel.failure(delay_swing(raw))


def qualifies(rows, refs, failure, baseline):
    ok, edp = checks(rows,refs)
    if not ok or failure is None:
        return False, None
    return bool(edp<=GOAL and link_residual(failure,baseline)<=0),float(edp)


def run_trial(layout,arm,seed,panel,meter,factory,out):
    start = perf_counter()
    dev, test = panel
    calls = 0
    hits = 0
    failed = 0
    ledger = []

    def pair(x, corner, step=1.):
        nonlocal calls, hits, failed
        result = []
        for k in (corner,corner+1):
            raw, hit = meter.measure(x,k,step)
            result.append(raw)
            ledger.append(dict(call=len(ledger)+1,x=x.tolist(),corner=k,step_ps=step,
                               cache_hit=hit,raw=raw))
            calls += 1
            hits += hit
            failed += "error" in raw
        return result

    n = len(upper(CIRCUIT))
    nominal = np.ones(n)
    ref = pair(nominal,0)+pair(nominal,2)
    if not all(checks(ref[k:k+2],ref[k:k+2])[0] for k in (0,2)):
        raise ValueError("Invalid nominal SPICE")
    base_dev = measured(dev,ref[:2])
    base_test = measured(test,ref[2:])
    info = []
    choose = factory(arm,info)
    perm = np.random.default_rng([seed,886]).permutation(n)
    inv = np.argsort(perm)
    lo,hi = np.full(n,np.log(.58)),np.log(upper(CIRCUIT)[perm])
    hist = [np.zeros(n)]
    labels = [np.r_[ratios(ref[:2],ref[:2]),-LINK_ALLOWANCE]]
    fine_ref = fine_baselines = None
    trace = []
    hit = None
    while calls+2<=CAP and hit is None:
        z = choose(arm,hist,labels,lo,hi,seed,None)
        x = np.round(np.exp(z)[inv],6)
        actual = pair(x,0)
        link = measured(dev,actual)
        if link is None:
            y = np.full(6,3.)
        else:
            y = np.r_[ratios(actual,ref[:2]),link_residual(link,base_dev)]
        q, edp = qualifies(actual,ref[:2],link,base_dev)
        check = final = None
        heldout_link = fine_link = None
        if q and calls+2<=CAP:
            check = pair(x,2)
            heldout_link = measured(test,check)
            q, _ = qualifies(check,ref[2:],heldout_link,base_test)
            if q and calls+(8 if fine_ref is None else 4)<=CAP:
                if fine_ref is None:
                    fine_ref = pair(nominal,0,.25)+pair(nominal,2,.25)
                    fine_baselines = [measured(dev,fine_ref[:2]),measured(test,fine_ref[2:])]
                    if any(v is None for v in fine_baselines):
                        raise ValueError("Invalid fine nominal SPICE")
                final = pair(x,0,.25)+pair(x,2,.25)
                fine_link = [measured(dev,final[:2]),measured(test,final[2:])]
                if all(qualifies(final[2*i:2*i+2],fine_ref[2*i:2*i+2],fine_link[i],
                                 fine_baselines[i])[0] for i in (0,1)):
                    hit = calls
        trace.append(dict(x=x.tolist(),y=y.tolist(),dev=actual,dev_link=link,dev_edp=edp,
                          check=check,check_link=heldout_link,fine=final,fine_link=fine_link,calls=calls))
        hist.append(np.log(x[perm]))
        labels.append(y)
    if hit is None and calls!=CAP:
        raise AssertionError("Missed trial consumed the wrong budget")
    row = dict(layout=layout,arm=arm,seed=seed,calls=calls,cache_hits=hits,
               physical_transients=calls-hits,invalid_attempts=failed,hit=hit,ref=ref,
               ref_link=[base_dev,base_test],fine_ref=fine_ref,fine_ref_link=fine_baselines,
               trace=trace,actions=info,ledger=ledger,elapsed_seconds=perf_counter()-start)
    write(out/f"layout{layout:02d}-{arm}.json",row)
    print(layout,arm,hit,calls,flush=True)
    return row


def audit_trial(row, panel):
    dev,this_test = panel
    calls = 4
    found = None
    reference = row["ref"]
    fine_reference_charged = False
    for trial in row["trace"]:
        assert found is None
        calls += 2
        link = measured(dev,trial["dev"])
        assert link==trial["dev_link"]
        q, _ = qualifies(trial["dev"],reference[:2],link,row["ref_link"][0])
        assert (trial["check"] is not None)==(q and calls+2<=CAP)
        if trial["check"] is not None:
            calls += 2
            link = measured(this_test,trial["check"])
            assert link==trial["check_link"]
            q, _ = qualifies(trial["check"],reference[2:],link,row["ref_link"][1])
            assert (trial["fine"] is not None)==(q and calls+(4 if fine_reference_charged else 8)<=CAP)
        if trial["fine"] is not None:
            calls += 4 if fine_reference_charged else 8
            fine_reference_charged = True
            fine = trial["fine"]
            values = [measured(dev,fine[:2]),measured(this_test,fine[2:])]
            assert values==trial["fine_link"]
            if all(qualifies(fine[2*i:2*i+2],row["fine_ref"][2*i:2*i+2],values[i],
                             row["fine_ref_link"][i])[0] for i in (0,1)):
                found = calls
        assert calls==trial["calls"]<=CAP
    assert len(row["ledger"])==row["calls"]==calls
    assert row["hit"]==found and (found is not None or calls==CAP)
    assert sum(r["cache_hit"] for r in row["ledger"])==row["cache_hits"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("channels","out","sky","wasm"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--node",required=True)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--limit",type=int,choices=range(1,51),default=50)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    raw_out = args.out/"trajectories"
    raw_out.mkdir()
    catalog = read(args.channels/"layouts.json")
    with np.load(args.channels/"channels.npz") as f:
        h = f["h"]
    assert h.shape==(50,20,20,61) and len({r["path"] for r in catalog["layouts"]})==50
    for matrix,row in zip(h,catalog["layouts"]):
        assert sha256(matrix.tobytes()).hexdigest()==row["matrix_sha256"]
    import torch
    torch.set_num_threads(1)
    factory,source = policy_factory(load_sources())
    meter = Spice(args)
    with ThreadPoolExecutor(args.workers) as pool:
        panel = list(pool.map(lambda i:panels(h[i],i),range(args.limit)))
        def execute(job):
            i,arm = job
            return run_trial(i,arm,SEEDS[i%len(SEEDS)],panel[i],meter,factory,raw_out)

        results = list(pool.map(execute,((0,a) for a in ARMS)))
        for pilot in results:
            audit_trial(pilot,panel[0])
        print("Audited the first physical 20-lane circuit against all seven methods.",flush=True)
        results.extend(pool.map(execute,((i,a) for i in range(1,args.limit) for a in ARMS)))
        list(pool.map(lambda r:audit_trial(r,panel[r["layout"]]),results))
    assert len(results)==args.limit*7
    stats = {}
    for i in range(args.limit):
        paired = [r for r in results if r["layout"]==i]
        seed = SEEDS[i%len(SEEDS)]
        bounds = upper(CIRCUIT)
        order = np.random.default_rng([seed,886]).permutation(len(bounds))
        common = [np.round(np.exp(np.random.default_rng([seed,step,6481])
                                      .uniform(np.log(.58),np.log(bounds[order])))[np.argsort(order)],6)
                  for step in range(1,5)]
        for r in paired:
            np.testing.assert_array_equal([step["x"] for step in r["trace"][:4]],common[:len(r["trace"][:4])])
    for arm in ARMS:
        subset = [r for r in results if r["arm"]==arm]
        stats[arm] = dict(n=len(subset), link_qualified_hits=sum(r["hit"] is not None for r in subset),
                          mean_standalone_calls=float(np.mean([r["calls"] for r in subset])),
                          source_spice=source[arm])
    physical = sum(r["physical_transients"] for r in results)
    assert physical==len(meter.entries)==sum(not corner["cache_hit"] for r in results for corner in r["ledger"])
    report = dict(layouts=args.limit,lanes=20,regimes=REGIMES,quarter_ps=True,goal=GOAL,cap=CAP,
                  source_archive_sha256=digest(ARCHIVE),circuit=asdict(CIRCUIT),methods=stats,
                  logical_target_calls=sum(r["calls"] for r in results),physical_target_transients=physical,
                  cache_hits=sum(r["cache_hits"] for r in results),
                  invalid_attempts=sum(r["invalid_attempts"] for r in results),
                  completed_by_layout=Counter(r["layout"] for r in results))
    write(args.out/"summary.json",report)
    print(stats,flush=True)


if __name__=="__main__":
    main()
