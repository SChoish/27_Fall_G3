"""Apply the same two locked, role-based moves to the four primary transfer targets."""
import json
import numpy as np
from experiment import arguments, Meter, TARGETS, ratios
from linesearch import line


def replay(name, args):
    meter = Meter(name, args)
    c = meter.c
    n = 2 * len(c.parents)
    x0 = np.ones(n)
    reference = meter.pair(x0, 0) + meter.pair(x0, 2)
    hist, y = [x0], [ratios(reference[:2], reference[:2])]
    calls, hits, trace = 4, {}, []
    # First two directions in role_line are seed-independent. If interpolation
    # repeats exactly, line() already has a fixed role direction fallback.
    rng = np.random.default_rng(21)
    for _ in range(2):
        x = np.round(line("role_line", {}, c, np.array(hist), np.array(y), None, rng), 6)
        dev = ratios(meter.pair(x, 0), reference[:2])
        calls += 2
        val = None
        if dev[0] <= np.log(.95) and max(dev[1:]) <= 0:
            val = ratios(meter.pair(x, 2), reference[2:])
            calls += 2
            for goal in (.95, .90):
                if max(val[1:]) <= 0 and max(dev[0], val[0]) <= np.log(goal):
                    hits.setdefault(goal, calls)
        trace.append(dict(x=x.tolist(), cost=calls, dev=dev.tolist(),
                          confirmation=None if val is None else val.tolist()))
        y.append(dev)
        hist.append(x)
    raw = [json.loads(p.read_text()) for p in sorted(meter.cache.glob("*.json"))]
    return dict(name=name, cost=calls, hits=hits, trace=trace, measurements=raw)


if __name__ == "__main__":
    args = arguments()
    # Keep post-hoc raw SPICE records outside the primary search cache.
    original = args.out
    args.out = original / "backtrack_check"
    results = [replay(name, args) for name in TARGETS]
    (original / "backtrack.json").write_text(json.dumps(results, indent=2))
    print([(r["name"], r["hits"]) for r in results])
