"""Recheck 5%-hit accounting and rerun each successful adaptive witness with finer timesteps."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from adaptive import CIRCUITS
from audit import checks
from circuit import simulate
from experiment import arguments


def passes(rows, ref):
    ok, edp = checks(rows, ref)
    return ok and edp <= .95


def main():
    args = arguments()
    args.step_ps = .25
    trials = json.loads((args.out/"adaptive.json").read_text())["trials"]
    successes = []
    for trial in trials:
        ref = trial["reference"]
        cost, hit = 4, None
        for step in trial["trace"]:
            cost += 2
            verify = passes(step["dev"], ref[:2]) and cost+2 <= 40
            assert verify == (step["confirmation"] is not None)
            if verify:
                cost += 2
                if passes(step["confirmation"], ref[2:]):
                    hit = cost
            assert cost == step["cost"]
        assert cost == trial["cost"] and hit == trial["first_five"]
        if trial["method"] == "adaptive" and hit is not None:
            successes.append(trial)

    def finer(trial):
        c = CIRCUITS[trial["name"]]
        x = trial["trace"][-1]["x"]
        raw = [simulate(c, widths, k, args) for widths in (np.ones(len(x)), x) for k in range(4)]
        pairs = [checks(raw[4:6],raw[:2]), checks(raw[6:],raw[2:4])]
        worst = max(p[1] for p in pairs)
        return dict(name=trial["name"],first_five=trial["first_five"],
                    passes=all(p[0] for p in pairs) and worst<=.95,
                    edp_ratio=worst if np.isfinite(worst) else None,measurements=raw)
    with ThreadPoolExecutor(3) as pool:
        rows = list(pool.map(finer,successes))
    (args.out/"adaptive_refinement.json").write_text(json.dumps(rows,indent=2))
    print([(r["name"],r["passes"],r["edp_ratio"]) for r in rows])


if __name__ == "__main__":
    main()
