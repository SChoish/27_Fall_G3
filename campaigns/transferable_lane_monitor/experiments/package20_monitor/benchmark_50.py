"""Compare causal 20-lane monitor policies over 50 held-out package layouts."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from pilot import (AR, Channel, FAMILIES, GRID, LANES, N, POLICIES, Q, R, SAMPLES,
                   T, TARGET, curves, drift_scale, evaluate, risk, scheduler, trajectory)


PERIOD = 4
MODES = ("mild","stationary","bursts")
NEW = ("centered_margin","quantized_risk")
EPISODES = 8
DEV_PACKET_SEED = 611100
TEST_PACKET_SEED = 941100
NOISE, WEIGHTS = np.polynomial.hermite.hermgauss(5)
NOISE *= np.sqrt(2.)
WEIGHTS /= np.sqrt(np.pi)


def characterize_all(channel):
    # Calibration sees a disjoint packet bank and measures all 20 eye widths.
    dev = np.minimum.reduce([curves(channel,DEV_PACKET_SEED+s,f) for s in range(8) for f in FAMILIES])
    phases = np.arange(2,15)
    nominal = phases[dev[:,phases-GRID[0]].argmax(axis=1)]
    tolerance = np.zeros((2,N))
    for lane,phase in enumerate(nominal):
        central = phase-GRID[0]
        eye = dev[lane]
        if eye[central] <= TARGET:
            continue
        for axis,direction in enumerate((-1,1)):
            i = central
            while eye[i+direction] >= TARGET:
                i += direction
            crossing = (eye[i]-TARGET)/(eye[i]-eye[i+direction])
            tolerance[axis,lane] = (abs(i-central)+crossing)/SAMPLES
    heldout = np.array([curves(channel,TEST_PACKET_SEED+s,f) for s in range(8) for f in FAMILIES])
    return nominal,*tolerance,heldout


def choose_quantized(mean, covariance, left, right, process_cov):
    sigma_obs = np.sqrt(covariance.diagonal()+R)
    gain_sigma = covariance/sigma_obs
    after_var = covariance.diagonal()[:,None]-gain_sigma**2
    # Counterfactual outcomes depend only on the predicted observation law.
    posterior = mean[None,:,None]+NOISE[:,None,None]*gain_sigma
    included = left+right>0
    scores = np.zeros(N)
    for h in (0,PERIOD-1):
        future = AR**h*posterior
        dy = sum(AR**(2*k) for k in range(h))*process_cov.diagonal()
        sd = np.sqrt(np.maximum(AR**(2*h)*after_var+dy[:,None],1e-12))
        correction = np.rint(SAMPLES*(future+.5*(right-left)[None,:,None]))/SAMPLES
        p = ndtr((correction-right[None,:,None]-future)/sd)
        p += ndtr((future-correction-left[None,:,None])/sd)
        # A product of marginal pass probabilities is a ranking approximation.
        # Actual validation propagates the joint drift and checks all 20 lanes.
        log_pass = np.log1p(-np.minimum(p[:,included,:],1-1e-12)).sum(axis=1)
        scores += WEIGHTS @ (-np.expm1(log_pass))
    return int(scores.argmin())


def monitor(policy, readings, left, right, q):
    mean = np.zeros(N)
    cov = np.zeros((N,N))
    correction = np.empty_like(readings)
    queried = np.full(T,-1)
    for t in range(T):
        mean *= AR
        cov = AR**2*cov+q
        if t%PERIOD==PERIOD//2:
            if policy=="quantized_risk":
                lane = choose_quantized(mean,cov,left,right,q)
            else:
                var = cov.diagonal()
                lane = int(risk(var[:,None]-cov**2/(var+R),left[:,None],right[:,None]).sum(axis=0).argmin())
            queried[t] = lane
            col = cov[:,lane].copy()
            denom = col[lane]+R
            mean += col*((readings[t,lane]-mean[lane])/denom)
            cov -= np.outer(col,col)/denom
        correction[t] = np.rint(SAMPLES*(mean+.5*(right-left)))
    return correction,queried


def layout_trial(item):
    index,h = item
    channel = Channel(h,32,True)
    nominal,left,right,bank = characterize_all(channel)
    rows = []
    audited = 0
    for scenario_index,scenario in enumerate(MODES):
        q = drift_scale(scenario)**2*Q
        for replicate in range(EPISODES):
            seed = 914000+1000*scenario_index+replicate
            truth,readings,packet = trajectory(index,seed,scenario,len(bank))
            for name in (*POLICIES,*NEW,"static"):
                if name in NEW:
                    correction,queried = monitor(name,readings,left,right,q)
                    count = int(np.sum(queried>=0))
                    if replicate==0:
                        # Replaying with every free observation changed verifies
                        # that lane choices and corrections use paid probes only.
                        altered = np.full_like(readings,1e5)
                        time = np.flatnonzero(queried>=0)
                        altered[time,queried[time]] = readings[time,queried[time]]
                        again,lanes = monitor(name,altered,left,right,q)
                        np.testing.assert_array_equal(again,correction)
                        np.testing.assert_array_equal(lanes,queried)
                        audited += 1
                elif name=="static":
                    correction,count = np.zeros_like(truth),0
                else:
                    correction,count = scheduler(name,PERIOD,readings,left,right,q)
                assert count==(0 if name=="static" else T//PERIOD)
                score = evaluate(correction,truth,packet,nominal,bank)
                if score["offgrid"]!=0:
                    raise ValueError(f"Phase outside characterized support on layout {index}: {name}")
                rows.append(dict(layout=index,scenario=scenario,seed=seed,method=name,probes=count,
                                 **{k:float(v) for k,v in score.items()}))
    return rows, dict(layout=index, dev_open_lanes=int(np.count_nonzero(left+right)),
                      heldout_packets=len(bank), observation_replays=audited)


def summarize(rows, layouts, diagnostics):
    by = defaultdict(list)
    for row in rows:
        by[row["layout"],row["method"],row["scenario"]].append(row)
    methods = (*POLICIES,*NEW,"static")
    assert len(rows)==len(layouts)*len(methods)*len(MODES)*EPISODES
    assert len(by)==len(layouts)*len(methods)*len(MODES)
    assert all(len(group)==len({r["seed"] for r in group})==EPISODES for group in by.values())
    stats = {}
    all_layouts = [r["layout"] for r in layouts]
    for key,scenarios in (("all",MODES),*((s,(s,)) for s in MODES)):
        def layout_means(method,metric):
            return np.array([np.mean([r[metric] for s in scenarios for r in by[i,method,s]])
                             for i in all_layouts])

        stats[key] = {name:{field:float(layout_means(name,field).mean())
                            for field in ("bad20","bad25","bad30","mean_margin","p05")} for name in methods}
        reference = layout_means("margin","bad25")
        rng = np.random.default_rng(26112822)
        sample = rng.integers(len(layouts),size=(10000,len(layouts)))
        for name in NEW:
            paired = layout_means(name,"bad25")-reference
            stats[key][name].update(delta_vs_margin=float(paired.mean()),
                layout_resample_ci95=np.percentile(paired[sample].mean(axis=1),[2.5,97.5]).tolist(),
                layouts_better_vs_margin=int(np.sum(paired<0)),
                layouts_equal_vs_margin=int(np.sum(paired==0)))
    return dict(layouts=len(layouts),lanes=N,rate_gbps=32,blocks=T,local_probes=T//PERIOD,
                extra_common_mode_observations=0,cardinality_by_method=len(layouts)*len(MODES)*EPISODES,
                results=stats,diagnostics=diagnostics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=8)
    args = p.parse_args()
    with np.load(args.channels/"channels.npz") as loaded:
        h = loaded["h"]
    catalog = json.loads((args.channels/"layouts.json").read_text())
    matrices = catalog["layouts"]
    assert h.shape==(50,20,20,61)
    assert len({r["path"] for r in matrices})==len({r["matrix_sha256"] for r in matrices})==50
    for layout,matrix in zip(matrices,h):
        assert sha256(matrix.tobytes()).hexdigest()==layout["matrix_sha256"]
        assert layout["through_paths"]==20 and layout["crosstalk_paths"]>0
    args.out.mkdir(parents=True,exist_ok=True)
    rows, diagnostics = [], []
    with (args.out/"episodes.csv").open("x") as f, ProcessPoolExecutor(args.workers) as pool:
        writer = None
        for new,validation in pool.map(layout_trial,enumerate(h)):
            if writer is None:
                writer = csv.DictWriter(f,fieldnames=list(new[0]))
                writer.writeheader()
            writer.writerows(new)
            f.flush()
            rows.extend(new)
            diagnostics.append(validation)
            print(f"Completed layout {validation['layout']+1}/50",flush=True)
    report = summarize(rows,matrices,diagnostics)
    report["catalog_sha256"] = sha256((args.channels/"layouts.json").read_bytes()).hexdigest()
    (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
