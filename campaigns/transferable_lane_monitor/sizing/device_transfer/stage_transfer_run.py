"""Paired frozen-actor cross-PDK trial. Each completed corner belongs to one arm and run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from audit import checks
from blind_width_run import run
from century_engine import simulate as compiled_simulate
from circuit import Circuit, simulate
from direct_surrogate_gate import chooser_direct
from ppo_sizing import ROOT
from surrogate_buffer import selector
from surrogate_buffer_run import models
from surrogate_buffer_transfer import source_pdk

ARMS = ("mlp200", "random_snapshot", "ppo_stage", "ppo_snapshot")
LAYOUTS = (
    Circuit("sky", (-1,0,1,2,2,4,5), (1,1.4,2,2.8,1.6,2,3.1), (3,6), (1,1), 2.5),
    Circuit("sky", (-1,0,1,2,-2,4,5,6,7,8), (1,1.3,1.8,2.7,1,1.25,1.55,1.9,2.3,3),
            (3,9), (1,1), 2.5),
)
ARCHIVE = ROOT / "direct_surrogate_gate_measurements.zip"
STAGE = ROOT / "stage_transfer_weights.json"


def conditions():
    for i, layout in enumerate(LAYOUTS):
        for j, loads in enumerate(((1,1), (.75,1.5))):
            seeds = list(range(2000+100*i+20*j, 2012+100*i+20*j))
            for pdk in ("sky", "gf"):
                yield f"stage_{i}_{j}_{pdk}", replace(layout, pdk=pdk, loads=loads), seeds


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policies():
    with ZipFile(ARCHIVE) as archive:
        sources = json.loads(archive.read("source.json"))
        full = json.loads(archive.read("training.json"))
    stage = read(STAGE)
    if stage["archive_sha256"] != digest(ARCHIVE):
        raise ValueError("Stage weights and source archive differ")
    return sources, models(sources), {"ppo_snapshot":full, "ppo_stage":stage["training"]}


def choose(arm, circuit, nets, actors, goal, log):
    src = source_pdk(circuit)
    if arm == "mlp200":
        return selector(nets[src], goal)
    trained = actors[arm][src] if arm in actors else None
    return chooser_direct(nets[src], trained, goal, log, details=True)


def evaluate(job, args, root, nets, actors):
    name, circuit, arm, seed, goal = job
    stem = f"{name}-{arm}-{seed}-{goal}"
    dest = root / (stem+".json")
    if dest.exists():
        return
    # Keep a private ordered ledger until the trajectory is committed. Replay all
    # completed corners on restart, including invalid attempts, without new SPICE.
    ledger = root / (stem+".partial")
    ledger.mkdir(exist_ok=True)
    calls = 0
    invalid = 0

    class Corners:
        def __init__(self, name, c, sim_args):
            self.args = sim_args

        def pair(self, x, start):
            nonlocal calls, invalid
            values = []
            for k in (start, start+1):
                calls += 1
                key = [x.tolist(), k, getattr(self.args, "step_ps", 1)]
                path = ledger / f"{calls:03}.json"
                if path.exists():
                    saved = read(path)
                    if saved["key"] != key:
                        raise ValueError(f"Cannot replay corner {path}")
                    raw = saved["raw"]
                else:
                    if getattr(self.args, "compiled", False):
                        try:
                            raw = compiled_simulate(circuit, x, k, self.args)
                        except RuntimeError as error:
                            raw = {"error":"ngspice", "log":str(error)}
                    else:
                        raw = simulate(circuit, x, k, self.args)
                    write(path, {"key":key, "raw":raw})
                invalid += int("error" in raw)
                values.append(raw)
            if np.all(x == 1) and (not checks(values, values)[0]):
                raise ValueError(f"Invalid nominal reference in {stem}, corner pair {start}")
            return values

    actions = []
    pick = choose(arm, circuit, nets, actors, goal, actions)
    with TemporaryDirectory(dir=root) as tmp:
        run(job, args, None, Path(tmp), pick, store_factory=Corners)
        row = read(Path(tmp)/dest.name)
    if calls != row["calls"] or calls > 100 or (row["hit"] is None and calls != 100):
        raise ValueError(f"SPICE ledger mismatch in {stem}")
    row.update(actions=actions, invalid_corners=invalid)
    write(dest, row)
    shutil.rmtree(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("sky","gf","wasm","out"):
        parser.add_argument("--"+arg, type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compiled", action="store_true", help="reuse compiled WASM; instantiate a fresh simulator per corner")
    parser.add_argument("--goal", type=float, choices=(.95,.90), default=.95)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.out / f"stage_transfer_{round(100*args.goal)}"
    root.mkdir(parents=True, exist_ok=True)
    with (root/"campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        targets = list(conditions())
        files = [ROOT/n for n in ("stage_transfer_run.py","direct_surrogate_gate.py","blind_width_run.py",
                 "circuit.py","surrogate_buffer.py","blind_width_policy.py","experiment.py","audit.py",
                 "raw_netlist_run.py","anonymous_network.py","../wasm_eval.mjs")]
        if args.compiled:
            files.extend([ROOT/"century_engine.py", ROOT/"century_worker.mjs"])
        manifest = dict(schema=1, compiled=args.compiled, goal=args.goal, cap=100, arms=ARMS, source_cost_per_pdk=200,
                        circuits={n:dict(circuit=asdict(c),seeds=s) for n,c,s in targets},
                        weights=digest(STAGE), archive=digest(ARCHIVE), numpy=np.__version__,
                        node=subprocess.check_output([args.node,"--version"], text=True).strip(),
                        code={p.name:digest(p) for p in files},
                        simulator={k:digest(getattr(args,k)) for k in ("sky","gf","wasm")})
        path = root/"manifest.json"
        if path.exists():
            previous = read(path)
            if "compiled" not in previous and not args.compiled:
                previous["compiled"] = False
                if previous["code"].get("stage_transfer_run.py") == "6149867a25dcf42a1a9381835ee61add0085bf4b11e8cfea0680297c49e85a8b":
                    previous["code"]["stage_transfer_run.py"] = manifest["code"]["stage_transfer_run.py"]
        if path.exists() and previous != json.loads(json.dumps(manifest)):
            raise ValueError("Manifest changed. Use a separate --out directory.")
        write(path, manifest)
        _, nets, actors = policies()
        jobs = [(n,c,a,s,args.goal) for n,c,seeds in targets for s in seeds for a in ARMS]
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(lambda j:evaluate(j,args,root,nets,actors), jobs))
        from stage_transfer_report import report
        report(root)


if __name__ == "__main__":
    main()
