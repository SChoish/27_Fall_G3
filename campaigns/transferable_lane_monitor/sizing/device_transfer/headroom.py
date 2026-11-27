"""Compare identical first role-based edits and locate the limiting electrical constraints."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from circuit import CIRCUITS, TARGETS, ratios
from linesearch import HELD, HOLD


def headroom(root):
    output = {}
    for name in (*TARGETS, HOLD):
        circuit = HELD if name == HOLD else CIRCUITS[name]
        rows = defaultdict(dict)
        for path in (root / "physical" / name).glob("*.json"):
            item = json.loads(path.read_text())
            rows[tuple(item["x"])][item["corner"]] = item["raw"]
        n = 2 * len(circuit.parents)
        nominal = rows[(1.,) * n]
        ref = [nominal[k] for k in range(4)]
        first_rule = tuple(.84 if i // 2 in circuit.outputs else 1. for i in range(n))
        best = (float("inf"), None)
        rule = {}
        for x, obs in rows.items():
            if 0 not in obs or 1 not in obs:
                continue
            dev = np.exp(ratios([obs[0], obs[1]], ref[:2]))
            if x == first_rule:
                rule["development"] = dev.tolist()
                if 2 in obs and 3 in obs:
                    rule["confirmation"] = np.exp(ratios([obs[2], obs[3]], ref[2:])).tolist()
            if max(dev[1:]) <= 1 and dev[0] < best[0]:
                best = (dev[0], x)
        output[name] = dict(first_rule=rule, best_development_edp=best[0],
                            best_development_x=best[1])
    (root / "headroom.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    headroom(Path("runs"))
