"""Paired commissioning and evaluation for a single shared timing monitor."""
import argparse
import csv
from pathlib import Path
import numpy as np
from modes import (BASE, ROOT, Channel, FAMILIES, GRID, LANES, PERIOD, SAMPLES,
                   TARGET, characterize, curves, hidden_model, panel, run_modes)
from study import load_delay_table, episode
from transfer import ParticleMonitor


# (rate in Gb/s, receiver RC time constant in ps, mapping to measured load classes)
CIRCUITS = {"original": (32, 0, [0, 1, 2, 3, 4]),
            "rc40": (32, 2, [0, 1, 2, 3, 4]),
            "rc100_24g": (24, 5, [4, 2, 0, 3, 1])}
CONDITIONS = ("quiet", "persistent", "fast", "offset")


def characterize_circuit(circuit, layout, packet_seed):
    rate, tau, loads = CIRCUITS[circuit]
    path = ROOT / f"implementation_results/20261125_package20/channels/layout{layout}.npz"
    h = np.load(path)["h"]
    # Unilateral first-order receiver, after the matched package transfer matrix.
    h = h / (1+2j*np.pi*np.arange(h.shape[-1])*tau/1000)
    channel = Channel(h, rate, True)
    nom, left, right, _ = characterize(channel)
    bank = np.array([curves(channel, packet_seed+s, f) for s in range(8) for f in FAMILIES])
    table = load_delay_table(BASE / "delay.csv").copy()
    # The reference panel and legacy estimator convert ps -> UI at 32 Gb/s.
    # Adapt their table in memory, equivalent to true delay/(1000/rate).
    for field in ("rise_ps", "fall_ps"):
        table[field] = (table[field].reshape(5, 6)[loads]*rate/32).ravel()
    delay = .5*(table["rise_ps"]+table["fall_ps"]).reshape(5, 6)
    amp = (delay[:, 2]-delay[:, -1])[LANES % 5]/31.25
    return (nom, left, right, bank), table, amp


def score(codes, truth, packet, char):
    nominal, _, _, bank = char
    x = nominal + codes - SAMPLES*truth - GRID[0]
    if not np.all((x >= 0) & (x < len(GRID)-1)):
        raise ValueError("phase outside characterized range")
    i = x.astype(int)
    v = bank[packet[:, None], LANES, i]
    w = bank[packet[:, None], LANES, i+1]
    return (v+(x-i)*(w-v)).min(axis=1) < TARGET


def particle_roll(data, char, q, amp, seeds, probe, particles):
    truth, readings, common, packets = data
    # One reproducible stream per fixed evaluation chunk of 16 trajectories.
    b, steps, lanes = truth.shape
    m = ParticleMonitor(q, amp, *char[1:3], b, np.random.default_rng([7289, int(seeds[0])]),
                        particles=particles, probe=probe)
    counts = np.zeros((2, b))
    nprobes = 0
    for t in range(steps):
        m.advance(common[:, t])
        if t % PERIOD == 2:
            a = m.choose()
            m.observe(a, readings[np.arange(b), t, a])
            nprobes += 1
        pred = (np.exp(m.logw)[:, :, None]*(m.base+m.shift)).sum(axis=1)
        for i, offset in enumerate([0, .5*(char[2]-char[1])]):
            counts[i] += score(np.rint(SAMPLES*(pred+offset)), truth[:, t], packets[:, t], char)
    assert nprobes == 32
    return counts/steps


def evaluate(args):
    layout = 0 if args.split == "dev" else 2
    seed0 = 310000 if args.split == "dev" else 910000
    packet_seed = 214000 if args.split == "dev" else 814000
    char, table, amp = characterize_circuit(args.circuit, layout, packet_seed)
    model = hidden_model(table)
    with np.load(BASE/"belief_weights.npz") as w:
        q, weights = w["q"], w["weights"]
    q_static = np.load(BASE/"models.npz")["process_cov"]
    center = lambda mean: np.rint(SAMPLES*(mean + .5*(char[2]-char[1])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        writer = csv.writer(f)
        writer.writerow(["circuit","split","condition","seed","policy","failure"])
        for condition in CONDITIONS:
            for start in range(0, args.episodes, 16):
                seeds = seed0+np.arange(start, min(start+16,args.episodes))
                data = panel(layout, seeds, condition, table, len(char[-1]))
                def save(name, values):
                    writer.writerows((args.circuit,args.split,condition,s,name,v)
                                     for s,v in zip(seeds,values))
                for name, correction in [("greedy",None), ("greedy_center",center)]:
                    _, met = run_modes(data,char,q,model,correction=correction)
                    save(name,met[:,0])
                random = np.stack([np.random.default_rng([297,int(s)]).integers(len(LANES),size=32)
                                   for s in seeds])
                for name, schedule in [("random",random),("round_robin",
                                        np.broadcast_to(np.arange(32)%len(LANES),random.shape))]:
                    _, met = run_modes(data,char,q,model,replay=schedule)
                    save(name,met[:,0])
                for name, par in zip(("ppo19","ppo23","ppo29"), weights):
                    _, met = run_modes(data,char,q,model,weights=par)
                    save(name,met[:,0])
                _, met, _ = episode(data,char,q_static,"static_trace")
                save("static_kalman",met[:,0])
                for probe in ("trace","eye"):
                    raw, centered = particle_roll(data,char,q,amp,seeds,probe,args.particles)
                    save("particle_"+probe,raw)
                    save("particle_"+probe+"_center",centered)
            f.flush()
            print(args.circuit,condition,"finished",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split",choices=("dev","test"),required=True)
    p.add_argument("--circuit",choices=CIRCUITS,required=True)
    p.add_argument("--episodes",type=int,default=256)
    p.add_argument("--particles",type=int,default=96)
    p.add_argument("--out",type=Path,required=True)
    evaluate(p.parse_args())
