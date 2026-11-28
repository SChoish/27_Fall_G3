"""Read verified first-hit traces and compare matched designs at smaller budgets."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def field(obj, path):
    for key in path.split("."):
        obj = obj[key]
    return obj


def integer(value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected an integer >= {minimum}, got {value!r}")
    return value


def read_row(raw, args):
    design = tuple(field(raw, key) for key in args.design_keys)
    if any(type(value) not in (str, int) for value in design):
        raise ValueError(f"invalid design key {design!r}")
    method = field(raw, args.method_key)
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty string")
    calls = integer(field(raw, args.calls_key), minimum=1)
    if args.success_key:
        success = field(raw, args.success_key)
        if type(success) is not bool:
            raise ValueError("success key must point to a boolean")
        hit = calls if success else None
    else:
        hit = field(raw, args.hit_key)
        if hit is not None:
            integer(hit, minimum=1)
    if (hit is not None and hit > calls) or calls > args.cap:
        raise ValueError(f"inconsistent first hit / total calls / cap: {hit!r}, {calls}, {args.cap}")
    if hit is None and calls < args.cap:
        raise ValueError(f"unresolved trajectory terminated at {calls} before cap {args.cap}")
    warm_hit = None
    if args.warmup_steps is not None:
        cumulative = [integer(field(step, args.step_calls_key), minimum=1)
                      for step in field(raw, args.trace_key)]
        if (any(a >= b for a, b in zip(cumulative, cumulative[1:]))
                or (cumulative and cumulative[-1] > calls)):
            raise ValueError("invalid cumulative calls in trace")
        limit = min(args.warmup_steps, len(cumulative))
        if limit < args.warmup_steps and hit is None:
            raise ValueError("incomplete warmup in trajectory without a verified hit")
        warm_hit = bool(limit and hit is not None and hit <= cumulative[limit - 1])
    return design, method, hit, warm_hit


def aggregate(rows, budgets, cap, reference=None, expected=None):
    matched = defaultdict(dict)
    for design, method, hit, warm in rows:
        if method in matched[design]:
            raise ValueError(f"duplicate trajectory for {design!r}, {method!r}")
        matched[design][method] = (hit, warm)
    if not matched:
        raise ValueError("no trajectories")
    arms = sorted({method for runs in matched.values() for method in runs})
    shape = (len(matched), len(arms))
    if expected is not None and shape != tuple(expected):
        raise ValueError(f"expected {tuple(expected)} (designs, methods), found {shape}")
    for key, runs in matched.items():
        missing = set(arms) - runs.keys()
        if missing:
            raise ValueError(f"incomplete paired design {key!r}: missing {sorted(missing)}")
    if reference is not None and reference not in arms:
        raise ValueError(f"reference method {reference!r} is absent")
    order = sorted(matched, key=repr)
    hits = np.array([[matched[key][arm][0] or 0 for arm in arms] for key in order])
    output = {"paired_designs": len(order), "reference": reference, "budgets": {}}
    warmups = [[matched[key][a][1] for key in order] for a in arms]
    if all(v is not None for arm in warmups for v in arm):
        per_case = np.array(warmups, dtype=bool)
        output["warmup_successes"] = {arm: int(per_case[i].sum()) for i, arm in enumerate(arms)}
        output["warmup_status_disagrees_on_pairs"] = int((per_case != per_case[:1]).any(axis=0).sum())
    resample = np.random.default_rng(816).integers(len(order), size=(10000, len(order)))
    for budget in budgets:
        passed = (hits > 0) & (hits <= budget)
        cost = np.where(passed, hits, budget)
        table, compared = {}, {}
        for index, arm in enumerate(arms):
            table[arm] = {"successes": int(passed[:, index].sum()), "n": len(order),
                          "restricted_mean_calls": float(cost[:, index].mean())}
            if budget == cap:
                h = hits[passed[:, index], index]
                table[arm]["mean_calls_given_success"] = float(h.mean()) if len(h) else None
            if reference is not None and arm != reference:
                j = arms.index(reference)
                delta = cost[:, index] - cost[:, j]
                compared[arm] = {
                    "method_minus_reference": float(delta.mean()),
                    "paired_bootstrap_ci95": np.percentile(delta[resample].mean(axis=1), [2.5, 97.5]).tolist(),
                    "method_only_success": int((passed[:, index] & ~passed[:, j]).sum()),
                    "reference_only_success": int((passed[:, j] & ~passed[:, index]).sum()),
                }
        output["budgets"][str(budget)] = {"methods": table, "paired": compared}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, help="folder containing one JSON record per trajectory")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=(20, 40, 60, 100))
    parser.add_argument("--design-keys", nargs="+", default=("name", "seed"))
    parser.add_argument("--method-key", default="arm")
    parser.add_argument("--calls-key", default="calls")
    hit = parser.add_mutually_exclusive_group()
    hit.add_argument("--hit-key", default="hit", help="verified first-hit calls as integer or null")
    hit.add_argument("--success-key", help="boolean field, only use for runners that stop on their first hit")
    parser.add_argument("--trace-key", default="trace")
    parser.add_argument("--step-calls-key", default="calls")
    parser.add_argument("--warmup-steps", type=int, help="number of common initial proposals in trace")
    parser.add_argument("--reference", help="method ID for paired differences")
    parser.add_argument("--expected", type=int, nargs=2, metavar=("DESIGNS", "METHODS"))
    args = parser.parse_args()
    if any(b < 1 or b > args.cap for b in args.budgets):
        parser.error("all budgets must be positive and at most the cap")
    if args.warmup_steps is not None and args.warmup_steps < 1:
        parser.error("--warmup-steps must be positive")

    rows = []
    for path in sorted(args.trajectories.glob(args.pattern)):
        if not path.is_file():
            continue
        try:
            rows.append(read_row(json.loads(path.read_text()), args))
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"{path}: {error}") from error
    try:
        report = aggregate(rows, sorted(set(args.budgets)), args.cap, args.reference, args.expected)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "attainment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.out / "attainment.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["budget", "method", "successes", "n", "restricted_mean_calls"])
        for budget, entry in report["budgets"].items():
            for method, values in entry["methods"].items():
                writer.writerow([budget, method, values["successes"], values["n"],
                                 values["restricted_mean_calls"]])
    print(f"{report['paired_designs']} paired designs: {args.out / 'attainment.json'}")


if __name__ == "__main__":
    main()
