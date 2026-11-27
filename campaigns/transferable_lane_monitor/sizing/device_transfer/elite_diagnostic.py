"""Count primary search designs that qualified in development and passed confirmation."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from circuit import TARGETS, ratios
from linesearch import HOLD


def summarize(root):
    for name in (*TARGETS, HOLD):
        by_design = {}
        for path in (root / "physical" / name).glob("*.json"):
            row = json.loads(path.read_text())
            by_design.setdefault(tuple(row["x"]), {})[row["corner"]] = row["raw"]
        reference = next(rows for x, rows in by_design.items() if all(v == 1 for v in x))
        counts = Counter()
        for rows in by_design.values():
            if len(rows) != 4:
                continue
            dev = ratios([rows[0], rows[1]], [reference[0], reference[1]])
            if dev[0] > np.log(.95) or max(dev[1:]) > 0:
                continue
            counts["confirmed"] += 1
            if any("error" in rows[k] for k in (2, 3)):
                counts["simulation_error"] += 1
                continue
            values = ratios([rows[2], rows[3]], [reference[2], reference[3]])
            violations = values > [np.log(.95), 0, 0, 0, 0]
            if not violations.any():
                counts["success"] += 1
            for key, fails in zip(("edp", "delay", "slew", "supply", "rail"), violations):
                counts[key] += int(fails)
        yield name, counts


if __name__ == "__main__":
    for name, counts in summarize(Path("runs")):
        print(name, dict(counts))
