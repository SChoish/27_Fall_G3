"""Matched SKY130-only priors and a 2 x 2 elite/metric intervention on GF180."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from adaptive_prior_run import TEST
from blind_width_audit import audit
from blind_width_policy import kernel, propose, rank_candidates
from blind_width_run import GOALS, run
from century_engine import enable
from experiment import arguments
from lane_focus import SEEDS
from raw_netlist_run import SKY_ONLY, bind, train

DOMAIN = "adapt_lane_gf"
ARMS = ("bo", "mlp200", "wires200", "elite", "metric", "elite_metric")
DIRECTION_ARMS = ("elite_gradient", "elite_natural")
ARCHIVE = Path(__file__).with_name("unseen_elite_metric_measurements.zip")


def local_proposals(z, y, lo, hi, radius, rng, model, elite, metric, directional=False):
    # Share the incumbent and the random unit-cube samples across interventions.
    order = np.argsort(y[:, 0] + 4 * np.maximum(y[:, 1:], 0).sum(1), kind="stable")
    keep = order[:3 if elite else 1]
    group = np.arange(len(radius)) % len(keep)
    anchors = z[keep]
    centers = anchors[group]
    local = rng.uniform(np.maximum(lo, centers-radius), np.minimum(hi, centers+radius))
    if not metric and not directional:
        return local
    span = hi-lo
    normalized = (z-lo)/span
    source_mean = y[0] + model.readout(model.encode(z))
    K = kernel(normalized, normalized) + .004*np.eye(len(z))
    residual = np.linalg.solve(K, y-source_mean)

    def cost(points):
        mean = y[0] + model.readout(model.encode(lo+points*span)) + kernel(points, normalized) @ residual
        return mean[:, 0] + (4/60)*np.logaddexp(0, 60*mean[:, 1:]).sum(1)

    step = 1e-4
    eye = np.eye(len(span))
    for i, anchor in enumerate(anchors):
        u = (anchor-lo)/span
        gradient = (cost(u+step*eye)-cost(u-step*eye))/(2*step)
        damping = .015*np.mean(gradient**2) + .0001
        geometry = np.outer(gradient, gradient) + np.diag(.7*gradient**2+damping)
        idx = group == i
        if directional:
            descent = -np.linalg.solve(geometry, gradient) if metric else -gradient
            descent /= max(np.max(np.abs(descent)), 1e-12)
            # Identical bounded perturbations and step lengths in both directional controls.
            lower = np.maximum(lo, anchor-radius[idx])
            upper = np.minimum(hi, anchor+radius[idx])
            noise = 2*(local[idx]-lower)/(upper-lower)-1
            noise[0] = 0.
            direction = descent+.45*noise
            direction /= np.maximum(np.max(np.abs(direction), axis=1, keepdims=True), 1e-12)
            local[idx] = np.clip(anchor+radius[idx]*direction, lo, hi)
            continue
        values, vectors = np.linalg.eigh(geometry)
        delta = (local[idx]-anchor)/span
        shaped = ((delta @ vectors)/np.sqrt(values)) @ vectors.T
        shaped *= np.linalg.norm(delta, axis=1, keepdims=True)/np.maximum(np.linalg.norm(shaped, axis=1, keepdims=True), 1e-15)
        # Reflect at the common box bounds after the direction change.
        local[idx] = lo + span*(1-np.abs((u+shaped)%2-1))
    return local


def models_from(source):
    assert len(source)==4 and all(r["pdk"]=="sky" for r in source)
    assert sum(len(r["x"])*2 for r in source)==200
    return train(source, only=("anon400", SKY_ONLY))


def chooser(models):
    mask_models = {s: bind(SKY_ONLY, models[SKY_ONLY], TEST[DOMAIN], s) for s in SEEDS}
    def select(arm, history, obs, lo, hi, seed, unused):
        if arm == "bo":
            return propose("bo", history, obs, lo, hi, seed)
        rng = np.random.default_rng([seed, len(history), 6481])
        if len(history)<=4:
            return rng.uniform(lo, hi)
        model = mask_models[seed] if arm == SKY_ONLY else models["anon400"]
        builder = None
        if arm in ("elite", "metric", "elite_metric", *DIRECTION_ARMS):
            def builder(z, y, lower, upper, radius, stream):
                return local_proposals(z, y, lower, upper, radius, stream, model,
                                       arm in ("elite", "elite_metric", *DIRECTION_ARMS),
                                       arm in ("metric", "elite_metric", "elite_natural"), arm in DIRECTION_ARMS)
        return rank_candidates(np.asarray(history), np.asarray(obs), lo, hi, rng,
                               network=model, local_builder=builder)
    return select


def verify(source, runs, select, arms=ARMS):
    selectors = {a: (select, None) for a in arms}
    return audit(source, runs, arms, selectors, {DOMAIN: TEST[DOMAIN]}, SEEDS, GOALS)[1]


def acquire(arms=ARMS, archive_path=ARCHIVE):
    args = arguments()
    enable()
    root = args.out/"unseen_elite_metric"
    root.mkdir(parents=True, exist_ok=True)
    # The archive contains the original SKY-only source labels and no GF180 source labels.
    with ZipFile(Path(__file__).with_name("anonymous_width_measurements.zip")) as archive:
        source = [dict(r, pdk="sky") for r in json.loads(archive.read("source.json"))]
    select = chooser(models_from(source))
    def pair(job):
        arm, seed = job
        rows = []
        for goal in GOALS:
            name = f"{DOMAIN}-{arm}-{seed}-{goal}.json"
            historical = args.out/"adaptive_prior"/name
            if arm=="bo" and historical.exists():
                rows.append(json.loads(historical.read_text()))
                continue
            run((DOMAIN, TEST[DOMAIN], arm, seed, goal), args, None, root, select)
            rows.append(json.loads((root/name).read_text()))
        return rows
    with ThreadPoolExecutor(args.workers) as pool:
        runs = [r for group in pool.map(pair, [(a,s) for s in SEEDS for a in arms]) for r in group]
    stats = verify(source, runs, select, arms)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as out:
        for k, v in (("source",source),("runs",runs),("audit",stats)):
            out.writestr(k+".json",json.dumps(v,separators=(",",":"),allow_nan=False))
    return stats


def replay(arms=ARMS, archive_path=ARCHIVE):
    with ZipFile(archive_path) as f:
        source, runs, expected = [json.loads(f.read(k+".json")) for k in ("source","runs","audit")]
    stats = verify(source, runs, chooser(models_from(source)), arms)
    assert stats==expected
    return stats


if __name__=="__main__":
    arms, archive_path = ARMS, ARCHIVE
    if "--directional" in sys.argv:
        sys.argv.remove("--directional")
        arms, archive_path = DIRECTION_ARMS, ARCHIVE.with_name("unseen_directional_measurements.zip")
    print(replay(arms, archive_path) if sys.argv[1:]==["--replay"] else acquire(arms, archive_path))
