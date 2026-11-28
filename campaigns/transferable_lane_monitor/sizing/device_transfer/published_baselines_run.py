"""Run paired 5% EDP trials of published sizing optimizers and G3 controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np

from alarm_pilot import SOURCE_NETS
from blind_width_audit import audit
from blind_width_policy import rank_candidates
from elite_width_transfer import mapped_local, templates
from experiment import upper
from stage_transfer_run import ARCHIVE, ROOT, conditions, digest, evaluate, read, source_pdk, write
from surrogate_buffer_run import models

ARMS = ("scbo", "kato", "dnnopt", "mlp200", "random_templates", "retrieved")
DEPENDENCIES = (
    "published_baselines_run.py", "published_scbo.py", "published_kato.py", "published_dnnopt.py",
    "stage_transfer_run.py", "blind_width_run.py", "blind_width_policy.py", "blind_width_audit.py",
    "experiment.py", "audit.py", "circuit.py", "elite_width_transfer.py", "raw_netlist.py",
    "raw_netlist_run.py", "anonymous_network.py", "alarm_pilot.py", "surrogate_buffer_run.py",
    "surrogate_buffer_transfer.py", "../wasm_eval.mjs",
)


def load_sources():
    with ZipFile(ARCHIVE) as archive:
        return json.loads(archive.read("source.json"))


def get_kato_donors(sources, targets):
    from published_kato import donor_for

    sizes = {name: len(c.outputs) for name, c in SOURCE_NETS.items()}
    return {name: donor_for(sources[source_pdk(c)], len(upper(c)), len(c.outputs), sizes)
            for name, c, _ in targets}


def prepare(targets, arms, sources, donors):
    source_models = models(sources) if set(arms) & {"mlp200", "random_templates", "retrieved"} else {}
    donor_gps = {}
    if "kato" in arms:
        from published_kato import source_gp

        for row in donors.values():
            key = row["pdk"], row["name"]
            if key not in donor_gps:
                circuit = replace(SOURCE_NETS[row["name"]], pdk=row["pdk"])
                donor_gps[key] = source_gp(row, np.log(.58), np.log(upper(circuit)))

    def factory(arm, circuit, goal, actions):
        src = source_pdk(circuit)
        if arm in ("mlp200", "random_templates", "retrieved"):
            net = source_models[src]
            bank = None if arm == "mlp200" else templates(sources[src], circuit, random=arm == "random_templates")

            def g3_pick(x, labels, lo, hi, rng, seed):
                builder = None
                if bank is not None:
                    perm = np.random.default_rng([seed, 886]).permutation(x.shape[1])
                    builder = mapped_local(bank[:, perm])
                return rank_candidates(x, labels, lo, hi, rng, network=net, local_builder=builder)
            external = None
        elif arm == "scbo":
            from published_scbo import select_scbo as external
        elif arm == "dnnopt":
            from published_dnnopt import select_dnnopt as external
        else:
            from published_kato import select_kato

            name = next(n for n, c, _ in targets if c == circuit)
            row = donors[name]
            source = donor_gps[row["pdk"], row["name"]]

            def external(x, y, rng, log):
                candidate, decision = select_kato(source, x, y, rng, log)
                return candidate, dict(decision, donor=row["name"], source_pdk=row["pdk"])

        def pick(_arm, hist, labels, lo, hi, seed, _prior):
            rng = np.random.default_rng([seed, len(hist), 6481])
            if len(hist) <= 4:
                return rng.uniform(lo, hi)
            x, y = np.asarray(hist), np.asarray(labels)
            if external is None:
                return g3_pick(x, y, lo, hi, rng, seed)
            point, decision = external(np.clip((x - lo) / (hi - lo), 0., 1.), y, rng, actions)
            if point.shape != lo.shape or not np.isfinite(point).all() or np.any((point < 0) | (point > 1)):
                raise ValueError(f"{arm} proposed invalid normalized widths")
            proposed = lo + point * (hi - lo)
            measured = np.round(np.exp(x), 6)
            duplicate = np.any(np.all(measured == np.round(np.exp(proposed), 6), axis=1))
            if duplicate:
                proposed = rng.uniform(lo, hi)
            actions.append(dict(decision, observed=len(hist), duplicate_fallback=bool(duplicate)))
            return proposed
        return pick
    return factory


def job_list(targets, arms, goal):
    return [(name, circuit, arm, seed, goal)
            for name, circuit, seeds in targets for seed in seeds for arm in arms]


def summarize(hits, arms):
    # Axes: layout/load, paired seed, target PDK, method.
    cost = np.where(hits > 0, hits, 100)
    rng = np.random.default_rng(99181)
    indices = rng.integers(hits.shape[1], size=(10000, *hits.shape[:2]))
    strata = np.arange(hits.shape[0])[:, None]
    reference = next((a for a in ("retrieved", "mlp200") if a in arms), arms[0])
    primary = arms.index(reference)
    result = {}
    for tag, pdks in (("all", slice(None)), ("sky", slice(0, 1)), ("gf", slice(1, 2))):
        h, c = hits[:, :, pdks], cost[:, :, pdks]
        methods, comparisons = {}, {}
        for i, arm in enumerate(arms):
            observed = h[..., i].ravel()
            methods[arm] = dict(n=observed.size, successes=int((observed > 0).sum()),
                                mean_cost=float(c[..., i].mean()), total_target_calls=int(c[..., i].sum()),
                                cumulative_hits=[int(((observed > 0) & (observed <= t)).sum())
                                                 for t in range(1, 101)])
            if i != primary:
                diff = (c[..., primary] - c[..., i]).mean(axis=2)
                boots = diff[strata, indices].mean(axis=(1, 2))
                comparisons[arm] = dict(reference_minus_control=float(diff.mean()),
                                        ci95=np.percentile(boots, [2.5, 97.5]).tolist(),
                                        reference_only=int(((h[..., primary] > 0) & (h[..., i] == 0)).sum()),
                                        control_only=int(((h[..., i] > 0) & (h[..., primary] == 0)).sum()))
        result[tag] = dict(methods=methods, paired=comparisons)
    return dict(reference=reference, results=result)


def report(root, targets, sources):
    manifest = read(root / "manifest.json")
    goal, arms = manifest["goal"], manifest["arms"]
    if digest(ARCHIVE) != manifest["archive"]:
        raise ValueError("Source archive differs from the recorded campaign")
    for name, circuit, _ in targets:
        if json.loads(json.dumps(asdict(circuit))) != manifest["circuits"][name]["circuit"]:
            raise ValueError(f"Current definition of {name} differs from the recorded campaign")
    targets = [(n, c, manifest["circuits"][n]["seeds"]) for n, c, _ in targets]
    jobs = job_list(targets, arms, goal)
    missing = sum(not (root / f"{n}-{a}-{s}-{g}.json").is_file() for n, _, a, s, g in jobs)
    if missing:
        raise ValueError(f"{missing} of {len(jobs)} trajectories are incomplete")
    hits = np.empty((4, len(targets[0][2]), 2, len(arms)), dtype=int)
    conditions_report = {}
    for index, (name, circuit, seeds) in enumerate(targets):
        rows = [read(root / f"{name}-{a}-{s}-{goal}.json") for s in seeds for a in arms]
        audit(sources[source_pdk(circuit)], rows, arms, domains={name: circuit},
              seeds=seeds, goals=(goal,), allow_invalid=True, replay_proposals=False)
        hits[index // 2, :, index % 2] = np.array([r["hit"] or 0 for r in rows]).reshape(len(seeds), len(arms))
        stats = {}
        for arm in arms:
            subset = [r for r in rows if r["arm"] == arm]
            stats[arm] = dict(n=len(subset), successes=sum(r["hit"] is not None for r in subset),
                              mean_cost=float(np.mean([r["calls"] for r in subset])),
                              invalid_corners=sum(r["invalid_corners"] for r in subset))
        conditions_report[name] = stats
    data = summarize(hits, arms)
    data.update(conditions=conditions_report, source_calls_by_condition=manifest["source_calls_by_condition"])
    write(root / "summary.json", data)
    for arm, stats in data["results"]["all"]["methods"].items():
        print(f'{arm:17} {stats["successes"]:2}/{stats["n"]:2}  {stats["mean_cost"]:.2f} SPICE')
    print(f"Audited {len(jobs)} trajectories: {root / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sky", "gf", "wasm", "out"):
        parser.add_argument(f"--{name}", type=Path, required=name == "out")
    parser.add_argument("--node")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--goal", type=float, choices=(.95, .90), default=.95)
    parser.add_argument("--seeds-per-condition", type=int, choices=range(1, 13), default=12,
                        metavar="1..12")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="show trajectory count and exit")
    mode.add_argument("--report-only", action="store_true", help="audit completed raw traces")
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.arms)) != len(args.arms):
        parser.error("workers must be positive and arms must be unique")
    targets = [(n, c, seeds[:args.seeds_per_condition]) for n, c, seeds in conditions()]
    jobs = job_list(targets, args.arms, args.goal)
    root = args.out / f"published_baselines_{round(100 * args.goal)}"
    if args.plan:
        print(f"{len(jobs)} trajectories, at most {100 * len(jobs):,} target SPICE transients")
        print(root)
        return
    if not args.report_only and (not args.node or not all(p and p.is_file() for p in
                                                         (args.sky, args.gf, args.wasm))):
        parser.error("provide --node and existing --sky, --gf, --wasm files")
    sources = load_sources()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.report_only:
            report(root, targets, sources)
            return
        if set(args.arms) & {"kato", "dnnopt"}:
            import torch

            torch.set_num_threads(1)
        donors = get_kato_donors(sources, targets) if "kato" in args.arms else {}
        code = DEPENDENCIES + (("century_engine.py", "century_worker.mjs") if args.compiled else ())
        packages = ["numpy"]
        if set(args.arms) & {"kato", "scbo"}:
            packages.append("scipy")
        if set(args.arms) & {"kato", "dnnopt"}:
            packages.append("torch")
        source_calls = {}
        for name, c, _ in targets:
            calls = 2 * sum(len(row["x"]) for row in sources[source_pdk(c)])
            source_calls[name] = {a: (2 * len(donors[name]["x"]) if a == "kato" else
                                      0 if a in ("scbo", "dnnopt") else calls) for a in args.arms}
        manifest = dict(schema=1, cap=100, goal=args.goal, compiled=args.compiled, arms=list(args.arms),
                        circuits={n: dict(circuit=asdict(c), seeds=s) for n, c, s in targets},
                        archive=digest(ARCHIVE), source_calls_by_condition=source_calls,
                        kato_donors={n: [row["pdk"], row["name"]] for n, row in donors.items()},
                        packages={p: version(p) for p in packages},
                        node=subprocess.check_output([args.node, "--version"], text=True).strip(),
                        code={p: digest(ROOT / p) for p in code},
                        simulator={k: digest(getattr(args, k)) for k in ("sky", "gf", "wasm")})
        path = root / "manifest.json"
        if path.exists() and read(path) != json.loads(json.dumps(manifest)):
            raise ValueError("Manifest differs. Restore the original options or choose a new --out")
        write(path, manifest)
        factory = prepare(targets, args.arms, sources, donors)

        def work(job):
            evaluate(job, args, root, None, None, pick_factory=factory)
        with ThreadPoolExecutor(args.workers) as pool:
            for _ in pool.map(work, jobs):
                pass
        report(root, targets, sources)


if __name__ == "__main__":
    main()
