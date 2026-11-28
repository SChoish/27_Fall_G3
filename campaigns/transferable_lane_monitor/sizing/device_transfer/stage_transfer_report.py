"""Audit raw measurements and report paired restricted first-hit costs."""
import argparse
from pathlib import Path

import numpy as np
from blind_width_audit import audit
from stage_transfer_run import ARMS, choose, conditions, policies, read, source_pdk, write


def verify_group(name, circuit, seeds, goal, rows, sources, nets, actors):
    logs = {a:[] for a in ARMS}
    selectors = {a:(choose(a,circuit,nets,actors,goal,logs[a]),None) for a in ARMS}
    audit(sources[source_pdk(circuit)],rows,ARMS,selectors,{name:circuit},seeds,(goal,),
          allow_invalid=True)
    for arm in ARMS:
        replay = iter(logs[arm])
        for r in rows:
            if r["arm"] != arm:
                continue
            for saved in r["actions"]:
                entry = next(replay)
                for field in ("actor_inputs","logits"):
                    expected, actual = entry.get(field), saved.get(field)
                    if expected is not None:
                        np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)
                    else:
                        assert actual is None
                assert all(entry[k]==saved[k] for k in ("end","observed","distinct","ends","proposal_ids","chosen"))
        assert next(replay,None) is None


def summarize(hits):
    # Axes are layout/load stratum, matched seed, target PDK, arm.
    cost = np.where(hits > 0, hits, 100)
    rng = np.random.default_rng(99181)
    indices = rng.integers(hits.shape[1],size=(10000,hits.shape[0],hits.shape[1]))
    strata = np.arange(hits.shape[0])[:,None]
    result = {}
    for tag, pdks in (("all",slice(None)),("sky",slice(0,1)),("gf",slice(1,2))):
        c, h = cost[:,:,pdks], hits[:,:,pdks]
        summary = {}
        for a, arm in enumerate(ARMS):
            times = h[...,a].ravel()
            summary[arm] = dict(n=times.size, successes=int((times>0).sum()),
                                mean_cost=float(c[...,a].mean()),
                                cumulative_hits=[int(((times>0)&(times<=t)).sum()) for t in range(1,101)])
        comparisons = {}
        for a, arm in enumerate(ARMS[:-1]):
            diff = (c[...,-1]-c[...,a]).mean(axis=2)
            boots = diff[strata,indices].mean(axis=(1,2))
            comparisons[arm] = dict(ppo_minus_control=float(diff.mean()),
                ci95=np.percentile(boots,[2.5,97.5]).tolist(),
                ppo_only=int(((h[...,-1]>0)&(h[...,a]==0)).sum()),
                control_only=int(((h[...,a]>0)&(h[...,-1]==0)).sum()))
        result[tag] = dict(methods=summary, paired=comparisons)
    return result


def report(root):
    goal = read(root/"manifest.json")["goal"]
    sources, nets, actors = policies()
    hits = np.empty((4,12,2,len(ARMS)), dtype=int)
    diagnostics = {}
    for idx,(name,circuit,seeds) in enumerate(conditions()):
        rows = [read(root/f"{name}-{a}-{s}-{goal}.json") for s in seeds for a in ARMS]
        verify_group(name,circuit,seeds,goal,rows,sources,nets,actors)
        hits[idx//2,:,idx%2,:] = np.array([r["hit"] or 0 for r in rows]).reshape(len(seeds),-1)
        diagnostics[name] = {}
        for arm in ARMS:
            group = [r for r in rows if r["arm"]==arm]
            actions = [entry for r in group for entry in r["actions"]]
            distinct = [entry for entry in actions if entry["distinct"]>1]
            diagnostics[name][arm] = dict(n=len(group), successes=sum(r["hit"] is not None for r in group),
                mean_cost=float(np.mean([r["calls"] for r in group])), decisions=len(actions),different_widths=len(distinct),
                source=sum(e["end"]==0 for e in distinct),
                fresh=sum(e["end"]==e["observed"] for e in distinct),
                past=sum(0<e["end"]<e["observed"] for e in distinct),
                invalid_corners=sum(r["invalid_corners"] for r in group))
    write(root/"summary.json",dict(results=summarize(hits),head_selection=diagnostics))
    print("Audited 384 trajectories:",root/"summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
