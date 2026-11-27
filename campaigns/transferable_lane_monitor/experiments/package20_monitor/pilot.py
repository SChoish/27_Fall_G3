"""Shared-monitor pilot on measured-in-simulation channel responses and synthetic drift."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/package20'))
from run import BITS, Channel, FAMILIES, LANES, SAMPLES, traffic

GRID = np.arange(-24, 41)
N = len(LANES)
T = 128
AR = .97
R = .025 ** 2
TARGET = .25
GROUP = (LANES[:, None] // 5 == LANES[None, :] // 5)
STATIONARY = .11**2 + .08**2 * GROUP + .06**2 * np.eye(N)
Q = STATIONARY * (1 - AR**2)
ROOT_Q = np.linalg.cholesky(Q)
POLICIES = ('round_robin', 'mee', 'trace', 'margin', 'margin_event')


def curves(channel, seed, family):
    bits = traffic(seed, family)
    x = np.fft.rfft(np.repeat(bits * .5, SAMPLES, axis=1))
    waves = np.fft.irfft(x[channel.tx] * channel.response, n=BITS * SAMPLES)
    samples = (np.arange(BITS)[:, None] * SAMPLES + channel.coarse + GRID) % (BITS * SAMPLES)
    score = np.empty((N, len(GRID)))
    for lane, edges in enumerate(channel.incoming):
        signed = waves[edges].sum(axis=0)[samples] * bits[lane, :, None]
        minimum = signed.min(axis=0)
        score[lane, 1:-1] = np.minimum.reduce([minimum[:-2], minimum[1:-1], minimum[2:]])
        score[lane, [0, -1]] = -1
    return score


def characterize(channel):
    dev = np.minimum.reduce([curves(channel, 11000 + s, f) for s in range(8) for f in FAMILIES])
    codes = np.arange(2, 15)
    nominal = codes[dev[:, codes - GRID[0]].argmax(axis=1)]
    low, high = [], []
    for lane, center in enumerate(nominal):
        index = center - GRID[0]
        assert dev[lane, index] > TARGET
        distances = []
        for d in [-1, 1]:
            i = index
            while dev[lane, i + d] >= TARGET:
                i += d
            frac = (dev[lane, i] - TARGET) / (dev[lane, i] - dev[lane, i + d])
            distances.append((abs(index - i) + frac) / SAMPLES)
        low.append(distances[0])
        high.append(distances[1])
    # Fixed held-out packet bank; the scheduler never sees the selected packet.
    bank = np.array([curves(channel, 37000 + s, f) for s in range(8) for f in FAMILIES])
    return nominal, np.array(low), np.array(high), bank


def drift_scale(scenario):
    return .5 if scenario == 'mild' else 1.


def trajectory(layout, seed, scenario, packets):
    rng = np.random.default_rng([30111, layout, seed])
    kick = drift_scale(scenario) * (rng.standard_normal((T, N)) @ ROOT_Q.T)
    if scenario == 'bursts':
        # Unannounced 8-block volatility burst in each half, unchanged monitor model.
        for start in (rng.integers(12, 48), rng.integers(76, 112)):
            kick[start:start + 8] *= 3
    drift = np.zeros(N)
    history = []
    for delta in kick:
        drift = AR * drift + delta
        history.append(drift)
    truth = np.array(history)
    return truth, truth + rng.normal(0, np.sqrt(R), truth.shape), rng.integers(packets, size=T)


def risk(var, left, right):
    from scipy.special import ndtr
    sd = np.sqrt(np.maximum(var, 1e-12))
    return ndtr(-left / sd) + ndtr(-right / sd)


def scheduler(policy, period, readings, left, right, process_cov, common=None):
    mean = np.zeros(N)
    cov = np.zeros((N, N))
    corrections = np.empty_like(readings)
    nprobe = 0
    surprise = False
    for t in range(T):
        mean *= AR
        cov = AR**2 * cov + process_cov
        if common is not None:
            # Hypothetical global monitor measures the spatial mean before the lane probe.
            column = cov.mean(axis=1)
            denom = column.mean() + R
            mean += column * ((common[t] - mean.mean()) / denom)
            cov -= np.outer(column, column) / denom
        if period and t % period == 0:
            # Once per window. A large previous innovation advances the next probe.
            offset = 0 if policy == 'margin_event' and surprise else period // 2
        if period and t % period == offset:
            var = cov.diagonal().copy()
            shrink = cov**2 / (var + R)
            if policy == 'round_robin':
                i = nprobe % N
            elif policy == 'mee':
                # Ramakanth et al., Eq. 33, evaluated at the allowed probe slots.
                i = (var / np.sqrt(Q.diagonal())).argmax()
            elif policy == 'trace':
                i = shrink.sum(axis=0).argmax()
            else:
                after = var[:, None] - shrink
                i = risk(after, left[:, None], right[:, None]).sum(axis=0).argmin()
            innovation = readings[t, i] - mean[i]
            surprise = abs(innovation) > 1.5 * np.sqrt(var[i] + R)
            column = cov[:, i].copy()
            mean += column * innovation / (var[i] + R)
            cov -= np.outer(column, column) / (var[i] + R)
            nprobe += 1
        corrections[t] = np.rint(mean * SAMPLES)
    assert nprobe == (T // period if period else 0)
    return corrections, nprobe


def evaluate(correction, truth, packet, nominal, bank):
    position = nominal + correction - SAMPLES * truth
    idx = np.clip(position - GRID[0], 0, len(GRID) - 1)
    i = idx.astype(int)
    bank_rows = bank[packet[:, None], LANES, i]
    following = bank[packet[:, None], LANES, np.minimum(i + 1, len(GRID) - 1)]
    margin = (bank_rows + (idx - i) * (following - bank_rows)).min(axis=1)
    return dict(bad20=np.mean(margin < .2), bad25=np.mean(margin < TARGET),
                bad30=np.mean(margin < .3), mean_margin=margin.mean(),
                p05=np.quantile(margin, .05), offgrid=np.sum(idx != position - GRID[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=48)
    parser.add_argument('--out', type=Path, default=ROOT / 'implementation_results/20261125_monitor')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'episodes.csv').open('w') as stream:
        writer = None
        for layout in range(3):
            path = ROOT / f'implementation_results/20261125_package20/channels/layout{layout}.npz'
            channel = Channel(np.load(path)['h'], 32, True)
            nominal, left, right, bank = characterize(channel)
            for scenario in ('mild', 'stationary', 'bursts'):
                for seed in range(args.episodes):
                    truth, readings, packet = trajectory(layout, seed, scenario, len(bank))
                    cases = [('static', 0, np.zeros_like(truth), 0),
                             ('known_drift', 0, np.rint(truth * SAMPLES), 0)]
                    for period in (2, 4, 8):
                        for policy in POLICIES:
                            correction, probes = scheduler(policy, period, readings, left, right, drift_scale(scenario)**2 * Q)
                            cases.append((policy, period, correction, probes))
                    # A second architecture receives a common-mode observation in every block.
                    ref_rng = np.random.default_rng([8901, layout, seed])
                    common = truth.mean(axis=1) + ref_rng.normal(0, np.sqrt(R), T)
                    for policy, period in [('only', 0), ('round_robin', 4), ('trace', 4), ('margin', 4)]:
                        correction, probes = scheduler(policy, period, readings, left, right,
                                                        drift_scale(scenario)**2 * Q, common=common)
                        cases.append(('common_' + policy, period, correction, probes))
                    for policy, period, correction, probes in cases:
                        row = dict(layout=layout, scenario=scenario, seed=seed, policy=policy,
                                   period=period, probes=probes, common_reads=T if policy.startswith('common_') else 0,
                                   **evaluate(correction, truth, packet, nominal, bank))
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=row)
                            writer.writeheader()
                        writer.writerow(row)
                print(layout, scenario, flush=True)


if __name__ == '__main__':
    main()
