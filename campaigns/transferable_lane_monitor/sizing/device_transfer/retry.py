"""Retry failed transients serially and check affected proposals without changing search history."""
import json


from audit import checks
from circuit import CIRCUITS, simulate
from experiment import arguments
from linesearch import HELD, HOLD


def main():
    args = arguments()
    circuits = {**CIRCUITS, HOLD: HELD}
    measurements = {}
    failed = set()
    supplemental = []
    for path in sorted((args.out / "physical").glob("*/*.json")):
        row = json.loads(path.read_text())
        name = path.parent.name
        key = (name, tuple(row["x"]), row["corner"])
        measurements[key] = row["raw"]
        if "error" in row["raw"]:
            failed.add(key)

    def acquire(key):
        name, x, corner = key
        row = simulate(circuits[name], x, corner, args)
        supplemental.append(dict(name=name, x=x, corner=corner, raw=row))
        measurements[key] = row

    for key in sorted(failed):
        acquire(key)
    affected = []
    for path in sorted(args.out.glob("*_2[123].json")):
        run = json.loads(path.read_text())
        name = run["name"]
        for step in run["trace"]:
            x = tuple(step["x"])
            keys = [(name, x, k) for k in range(4)]
            used = keys if step["verified"] else keys[:2]
            if not failed.intersection(used):
                continue
            ref = [measurements[name, (1.,)*len(x), k] for k in range(4)]
            ok, edp = checks([measurements[k] for k in keys[:2]], ref[:2])
            dev_hit = ok and edp <= .95
            # A previously rejected development candidate would require two extra calls.
            can_verify = step["verified"] or step["cost"] + 2 <= 40
            hit = []
            if dev_hit and can_verify:
                for key in keys[2:]:
                    if key not in measurements:
                        acquire(key)
                valid, check_edp = checks([measurements[k] for k in keys[2:]], ref[2:])
                hit = [g for g in (.95, .90) if valid and max(edp, check_edp) <= g]
            affected.append(dict(run=path.stem, cost=step["cost"], dev_hit=dev_hit,
                                 feasible_goals=hit, existing_hits=run["hits"]))
    result = dict(failed_unique=len(failed), remaining=sum("error" in measurements[k] for k in failed),
                  supplemental_calls=len(supplemental), affected=affected, measurements=supplemental)
    (args.out / "retries.json").write_text(json.dumps(result, indent=2))
    print({k:v for k,v in result.items() if k!="measurements"})


if __name__ == "__main__":
    main()
