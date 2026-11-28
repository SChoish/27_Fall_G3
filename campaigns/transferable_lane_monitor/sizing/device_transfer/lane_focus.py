"""Run the paired 5% lane-load comparison or replay its paid observations."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from adaptive_prior_run import TEST
from blind_width_audit import audit
from blind_width_policy import propose
from blind_width_run import run
from century_engine import enable
from experiment import arguments
from raw_netlist_run import ARMS, bind, select, train

DOMAINS = {key: TEST[key] for key in ("adapt_lane_sky", "adapt_lane_gf")}
METHODS = ("bo", *ARMS)
SEEDS = (81, 82, 83)
GOAL = .95
ARCHIVE = Path(__file__).with_name("lane_focus_measurements.zip")


def selector(method, circuit, models):
    if method == "bo":
        return lambda arm, hist, y, lo, hi, seed, _: propose(arm, hist, y, lo, hi, seed)
    nets = {s: bind(method, models[method], circuit, s) for s in SEEDS}
    return lambda arm, hist, y, lo, hi, seed, _: select(arm, hist, y, lo, hi, seed, nets[seed])


def verify(source, runs, models):
    assert len(runs) == len(METHODS) * len(DOMAINS) * len(SEEDS)
    stats = {}
    for domain, circuit in DOMAINS.items():
        selectors = {arm: (selector(arm, circuit, models), None) for arm in METHODS}
        rows = [row for row in runs if row["name"] == domain]
        _, stats[domain] = audit(source, rows, METHODS, selectors, {domain: circuit}, SEEDS, (GOAL,))
    return stats


def acquire():
    args = arguments()
    enable()
    raw = args.out / "raw_netlist"
    source = json.loads((raw / "source.json").read_text())
    models = train(source, only=ARMS)

    def evaluate(job):
        name, arm, seed = job
        circuit = DOMAINS[name]
        folder = args.out / "adaptive_prior" if arm == "bo" else raw
        folder.mkdir(parents=True, exist_ok=True)
        run((name, circuit, arm, seed, GOAL), args, None, folder, selector(arm, circuit, models))
        return json.loads((folder / f"{name}-{arm}-{seed}-{GOAL}.json").read_text())

    with ThreadPoolExecutor(args.workers) as pool:
        runs = list(pool.map(evaluate, [(n, a, s) for s in SEEDS for a in METHODS for n in DOMAINS]))
    stats = verify(source, runs, models)
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for key, data in (("source", source), ("runs", runs), ("audit", stats)):
            archive.writestr(key + ".json", json.dumps(data, separators=(",", ":"), allow_nan=False))
    return stats


def replay():
    with ZipFile(ARCHIVE) as archive:
        source = json.loads(archive.read("source.json"))
        runs = json.loads(archive.read("runs.json"))
        stored = json.loads(archive.read("audit.json"))
    stats = verify(source, runs, train(source, only=ARMS))
    assert stored == stats
    return stats


if __name__ == "__main__":
    print(replay() if sys.argv[1:] == ["--replay"] else acquire())
