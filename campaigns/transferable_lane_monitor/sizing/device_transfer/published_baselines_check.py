"""No-SPICE integration check for the six methods and the charged corner ledger."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiment import upper
from published_baselines_run import (
    ARMS, ARCHIVE, conditions, digest, get_kato_donors, job_list, load_sources, prepare, report, summarize,
)
from published_kato import AlignedGP, NeuralGP, mace_front
from published_scbo import posterior_draw, preferred, trust_region
from stage_transfer_run import evaluate, read, write


def check_numerics():
    # Feasibility beats a stronger unconstrained objective; among failures
    # the sum of positive violations determines the order.
    data = np.array([[-.6, .2, .1], [-.2, -.1, -.1], [-.8, .1, 0.], [-.3, -.1, 0.]])
    np.testing.assert_array_equal(preferred(data), [3, 1, 2, 0])
    np.testing.assert_equal(trust_region(np.tile(data[1], (19, 1)), 14), (.4, 0))
    for value in (-.2, .3):
        rng = np.random.default_rng(4)
        x = rng.random((8, 3))
        np.testing.assert_allclose(posterior_draw(x, np.full(8, value), x, rng), value, atol=1e-5)
    front = mace_front(np.array([[-.3, -.1], [-.2, -.1], [-.3, .2]]), np.full((3, 2), .02), -.2)
    np.testing.assert_array_equal(front, [0])

    rng = np.random.default_rng(23)
    sx, sy, tx, ty = (rng.random(shape) for shape in ((10, 4), (10, 5), (7, 6), (7, 5)))
    source = NeuralGP(sx, sy, rng, steps=5)
    before, _ = source.posterior(torch.from_numpy(sx))
    gp = AlignedGP(source, torch.from_numpy(tx), torch.from_numpy(ty), rng)
    point = torch.from_numpy(tx[:1])
    with torch.no_grad():
        source_mean, source_var = source.posterior(gp.encoder(point).sigmoid())
    jacobian = torch.autograd.functional.jacobian(lambda z: gp.decoder(z)[0], source_mean)[:, 0]
    expected = source_var[0] * (jacobian @ jacobian.T)
    expected += gp.eye * gp.log_noise.clamp(-9., 2.).exp()
    _, covariance = gp.posterior(point)
    torch.testing.assert_close(covariance[0], expected)
    after, _ = source.posterior(torch.from_numpy(sx))
    torch.testing.assert_close(before, after, rtol=0., atol=0.)
    demo = summarize(np.array([[[[0, 20], [60, 0]]]]), ("retrieved", "mlp200"))
    totals = demo["results"]["all"]["methods"]
    assert totals["retrieved"]["mean_cost"] == 80. and totals["mlp200"]["mean_cost"] == 60.


def main():
    torch.set_num_threads(1)
    check_numerics()
    targets = [(n, c, seeds[:1]) for n, c, seeds in conditions()]
    sources = load_sources()
    donors = get_kato_donors(sources, targets)
    factory = prepare(targets, ARMS, sources, donors)
    cold = {}
    for name, circuit, (seed,) in targets:
        bounds = upper(circuit)
        perm = np.random.default_rng([seed, 886]).permutation(len(bounds))
        cold[circuit] = {
            tuple(np.round(np.exp(np.random.default_rng([seed, t, 6481])
                                  .uniform(np.log(.58), np.log(bounds[perm])))[np.argsort(perm)], 6))
            for t in range(1, 5)
        }

    def synthetic(circuit, widths, corner, args):
        nominal = np.all(widths == 1.)
        delay = 1. + .07 * corner
        energy = 1. if nominal else ((.98 + .015 * widths.mean()) if tuple(widths) in cold[circuit] else .9)
        return {"energy": float(energy), "ports": [[delay, delay, 1., 1., .99, .01]
                                                   for _ in circuit.outputs]}

    args = SimpleNamespace(compiled=True)
    jobs = job_list(targets, ARMS, .95)
    with TemporaryDirectory() as a, TemporaryDirectory() as b:
        full_root, resumed_root = Path(a), Path(b)
        def impossible(circuit, widths, corner, args):
            return dict(synthetic(circuit, widths, corner, args), energy=1.)

        miss = next(j for j in jobs if j[2] == "mlp200")
        with patch("stage_transfer_run.compiled_simulate", side_effect=impossible) as simulator:
            evaluate(miss, args, resumed_root, None, None, pick_factory=factory)
        name, _, arm, seed, goal = miss
        failed = read(resumed_root / f"{name}-{arm}-{seed}-{goal}.json")
        assert failed["hit"] is None and failed["calls"] == simulator.call_count == 100

        # Repeat a KATO trajectory across an interruption after 16 paid corners.
        job = next(j for j in jobs if j[2] == "kato")
        calls = []

        def interrupt(*x):
            if len(calls) == 16:
                raise InterruptedError("synthetic interruption")
            calls.append(x)
            return synthetic(*x)

        def count(*x):
            calls.append(x)
            return synthetic(*x)

        with patch("stage_transfer_run.compiled_simulate", side_effect=interrupt):
            try:
                evaluate(job, args, resumed_root, None, None, pick_factory=factory)
            except InterruptedError:
                pass
            else:
                raise AssertionError("did not interrupt")
        calls.clear()
        with patch("stage_transfer_run.compiled_simulate", side_effect=count):
            evaluate(job, args, resumed_root, None, None, pick_factory=factory)
        assert len(calls) == 8
        with patch("stage_transfer_run.compiled_simulate", side_effect=synthetic):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda trial: evaluate(trial, args, full_root, None, None,
                                                     pick_factory=factory), jobs))
        name, _, arm, seed, goal = job
        fname = f"{name}-{arm}-{seed}-{goal}.json"
        assert read(full_root / fname) == read(resumed_root / fname)
        for name, circuit, (seed,) in targets:
            rows = [read(full_root / f"{name}-{arm}-{seed}-0.95.json")
                    for arm in ARMS]
            assert all(row["hit"] == row["calls"] == 24 for row in rows)
            assert {row["arm"] for row in rows if row["actions"]} == {
                "kato", "scbo", "dnnopt", "constrained_search"}
        write(full_root / "manifest.json", dict(goal=.95, arms=ARMS, archive=digest(ARCHIVE),
              circuits={name: dict(circuit=asdict(c), seeds=seeds) for name, c, seeds in targets},
              source_calls_by_condition={}))
        report(full_root, targets, sources)
        summary = read(full_root / "summary.json")["results"]["all"]["methods"]
        assert all(row["successes"] == 8 and row["mean_cost"] == 24. for row in summary.values())
    print(f"Passed numerical checks, all {len(jobs)} synthetic trajectories, reporting and interruption/resume.")


if __name__ == "__main__":
    main()
