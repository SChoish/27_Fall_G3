"""Ablate historical PPO value baselines on the same source episodes and target circuits."""
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys

import numpy as np
from blind_width_audit import audit
from blind_width_run import GOALS, run
from century_engine import enable
from experiment import arguments
from ppo_sizing import DOMAINS, ROOT, policies, seeds, source_for, train, train_actor

ARCHIVE = ROOT / "ppo_critic_buffer_measurements.zip"
ARM = "ppo_buffer"


def learn(rows):
    models = [train(rows[:i] + rows[i+1:], only=("anon400",))["anon400"]
              for i in range(len(rows))]
    return {ARM: train_actor(rows, models, gamma=1., buffer_size=4)}


def verify(sources, training, runs, arm):
    select = policies(sources, training)
    stats = {}
    for name, circuit in DOMAINS.items():
        pdk = source_for(name)
        assert pdk != circuit.pdk
        for goal in GOALS:
            group = [r for r in runs if r["goal"] == goal and r["name"] == name]
            _, stats[f"{name}_{goal}"] = audit(
                sources[pdk], group, (arm,), {arm: (select[pdk][goal], None)},
                {name: circuit}, seeds(name), (goal,))
    assert len(runs) == 24
    return stats


def main(arm=ARM, archive=ARCHIVE, fit=learn, directory="ppo_critic_buffer"):
    if sys.argv[1:] == ["--replay"]:
        with ZipFile(archive) as f:
            sources, training, runs, expected = [
                json.loads(f.read(k + ".json")) for k in ("source", "training", "runs", "audit")]
        for pdk, rows in sources.items():
            fitted = fit(rows)[arm]
            np.testing.assert_allclose(fitted["weights"], training[pdk][arm]["weights"], atol=1e-10)
            assert fitted["critic_age_counts"] == training[pdk][arm]["critic_age_counts"]
        assert verify(sources, training, runs, arm) == expected
        print(f"Replayed all 24 {arm} runs and both source policies.")
        return
    args = arguments()
    enable()
    records = json.loads((args.out / "raw_netlist" / "source.json").read_text())
    sources = {p: [r for r in records if r["pdk"] == p] for p in ("sky", "gf")}
    training = {p: fit(rows) for p, rows in sources.items()}
    print("trained", training, flush=True)
    select = policies(sources, training)
    root = args.out / directory
    root.mkdir(exist_ok=True, parents=True)

    def evaluate(job):
        name, seed = job
        group = []
        for goal in GOALS:
            run((name, DOMAINS[name], arm, seed, goal), args, None, root, select[source_for(name)][goal])
            group.append(json.loads((root / f"{name}-{arm}-{seed}-{goal}.json").read_text()))
        return group

    with ThreadPoolExecutor(args.workers) as pool:
        runs = [r for group in pool.map(evaluate, [(n,s) for n in DOMAINS for s in seeds(n)]) for r in group]
    stats = verify(sources, training, runs, arm)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as f:
        for key, data in (("source",sources), ("training",training), ("runs",runs), ("audit",stats)):
            f.writestr(key+".json", json.dumps(data, separators=(",",":"), allow_nan=False))
    print(stats)


if __name__ == "__main__":
    main()
