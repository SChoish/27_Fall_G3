"""A deterministic role-based feasible-step policy and a final held-out PDK transfer."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import numpy as np
import experiment as exp
from audit import checks
from linesearch import HOLD, HELD
from proposal import propose

BLIND = "fork_pair_gf"
CIRCUITS = {**exp.CIRCUITS, HOLD: HELD, BLIND: replace(HELD, pdk="gf")}
POLICY = ((.84, 1.), None, (.91, .86), (.96, .74), (.89, .94), (.95, .90),
          (.80, 1.), None)


def size_move(c, schedule, index, hist, observations):
    terminal = np.repeat([i in c.outputs for i in range(len(c.parents))], 2)
    move = schedule[index]
    if move is not None:
        return np.where(terminal, move[0], move[1])
    # Fit each normalized log constraint along the previous log-width change;
    # choose the tightest estimated boundary, retaining six percent headroom.
    change = observations[-1][1:] - observations[0][1:]
    rising = change > 1e-6
    bounds = -observations[0][1:][rising] / change[rising]
    fraction = np.clip(.94 * min([1.6, *bounds]), .2, 1.6)
    return np.clip(hist[-1] ** fraction, .58, exp.upper(c))


def run_trial(args, name, method, seed):
    c = CIRCUITS[name]
    meter = exp.Meter(name, args)
    ref_x = np.ones(2*len(c.parents))
    ref = meter.pair(ref_x, 0) + meter.pair(ref_x, 2)
    hist, obs = [ref_x], [exp.ratios(ref[:2], ref[:2])]
    cost, hit, trace = 4, None, []
    rng = np.random.default_rng(seed)
    while cost + 2 <= 40:
        index = len(trace)
        if method == "adaptive":
            if index == len(POLICY):
                break
            x = size_move(c, POLICY, index, hist, obs)
        else:
            y = np.array(obs)
            best = np.argmin(y[:,0]+7*np.maximum(y[:,1:],0).sum(1))
            candidates = exp.pool(c,rng,hist[best])
            if method == "cold_bo":
                x = propose(method,{},c,np.array(hist),y,candidates,rng)
            else:
                cycle = (POLICY[0], *POLICY[2:6])
                x = size_move(c, cycle, index % 5, hist, obs)
                if index >= 5:
                    x = np.clip(x*np.exp(rng.normal(0,.045,len(x))),.58,exp.upper(c))
        x = np.round(x,6)
        dev = meter.pair(x,0)
        cost += 2
        ok, edp = checks(dev,ref[:2])
        val = None
        if ok and edp <= .95 and cost+2 <= 40:
            val = meter.pair(x,2)
            cost += 2
            valid, confirm = checks(val,ref[2:])
            if valid and max(edp,confirm) <= .95:
                hit = cost
        trace.append(dict(x=x.tolist(),cost=cost,dev=dev,confirmation=val))
        hist.append(x)
        obs.append(exp.ratios(dev,ref[:2]))
        if hit is not None:
            break
    return dict(name=name,method=method,seed=seed,first_five=hit,cost=cost,reference=ref,trace=trace)


if __name__ == "__main__":
    args = exp.arguments()
    exp.CIRCUITS.update(CIRCUITS)
    root = args.out
    args.out = root / "adaptive_check"
    groups = [(n, [("adaptive",21)]) for n in (*exp.TARGETS,HOLD)]
    # Keep trials on the same new target sequential so identical cache entries are reused.
    groups.append((BLIND,[("adaptive",21), *((m,s) for m in ("rule","cold_bo") for s in (21,22,23))]))
    def run_group(group):
        name, trials = group
        return [run_trial(args,name,m,s) for m,s in trials]
    with ThreadPoolExecutor(4) as pool:
        records = [r for group in pool.map(run_group,groups) for r in group]
    count = len(list((args.out/"physical").glob("*/*.json")))
    (root/"adaptive.json").write_text(json.dumps(dict(unique_measurements=count,trials=records)))
    print([(r["name"],r["method"],r["seed"],r["first_five"]) for r in records])
