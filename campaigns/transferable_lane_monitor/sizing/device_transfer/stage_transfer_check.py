"""Check actor masking and identical counterfactual proposals on archived target histories."""
from zipfile import ZipFile
import json
import argparse
import numpy as np

from circuit import ratios
from direct_surrogate_gate import fit_direct
from experiment import upper
from stage_transfer_run import ARCHIVE, ARMS, choose, policies
from surrogate_gate_run import ALL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="reproduce the six source-only stage actors")
    args = parser.parse_args()
    sources, nets, actors = policies()
    if args.retrain:
        for pdk, records in sources.items():
            rebuilt = fit_direct(records, "ppo_stage")
            for new, old in zip(rebuilt, actors["ppo_stage"][pdk], strict=True):
                np.testing.assert_allclose(new["weights"], old["weights"], rtol=0, atol=1e-9)
                assert all(new[k]==old[k] for k in ("decisions","warm_hits","policy_hits"))
        print("Reproduced six frozen stage actors.")
    with ZipFile(ARCHIVE) as archive:
        rows = json.loads(archive.read("runs.json"))
    for group in actors["ppo_stage"].values():
        assert len(group)==3
        for model in group:
            assert model["decisions"]>0 and np.all(np.asarray(model["weights"])[3:8]==0)
    histories = different = 0
    for row in rows:
        c, seed, goal = ALL[row["name"]], row["seed"], row["goal"]
        perm = np.asarray(row["perm"])
        ref = row["ref"][:2]
        labels = [ratios(ref,ref)]
        hist = [np.zeros(len(perm))]
        lo, hi = np.full(len(perm),np.log(.58)), np.log(upper(c)[perm])
        logs = {a:[] for a in ARMS}
        selectors = {a:choose(a,c,nets,actors,goal,logs[a]) for a in ARMS}
        for t,step in enumerate(row["trace"][:13]):
            if t<4:
                warms = [selectors[a](a,hist,labels,lo,hi,seed,None) for a in ARMS]
                for warm in warms:
                    np.testing.assert_array_equal(warm,warms[0])
            else:
                for a in ("ppo_snapshot","ppo_stage"):
                    selectors[a](a,hist,labels,lo,hi,seed,None)
                full, stage = logs["ppo_snapshot"][-1],logs["ppo_stage"][-1]
                assert full["ends"]==stage["ends"] and full["proposal_ids"]==stage["proposal_ids"]
                F = np.asarray(full["actor_inputs"])
                F[:,3:8] = 0
                np.testing.assert_array_equal(F, stage["actor_inputs"])
                histories += 1
                different += int(full["distinct"]>1)
            hist.append(np.log(np.asarray(step["x"])[perm]))
            labels.append(ratios(step["dev"],ref))
    assert different>0
    print(f"Verified warm-ups, actor inputs and all head proposals on {histories} decisions; "
          f"{different} had distinct proposals.")


if __name__ == "__main__":
    main()
