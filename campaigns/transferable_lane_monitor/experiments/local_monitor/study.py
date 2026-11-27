"""Cross-layout test of history-dependent lane scheduling with a shared timing monitor."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/package20_monitor'))
from pilot import AR, Channel, GRID, LANES, N, Q, R, SAMPLES, T, TARGET, characterize, trajectory

BASE = ROOT / 'implementation_results/local_monitor'
GROUPS = LANES // 5
MEMBERS = np.eye(4)[GROUPS].T
GROUP_KERNEL = MEMBERS[:, :, None] * MEMBERS[:, None, :]
PERIOD = 4
STEPS = T // PERIOD
NFEAT = 9


def load_delay_table(path):
    table = np.genfromtxt(path, delimiter=',', names=True)
    if table.dtype.names != ('load_fF', 'vdd_V', 'rise_ps', 'fall_ps') or table.size != 30:
        raise ValueError(f'Expected 30 load/voltage points in {path}')
    table = np.sort(table, order=['load_fF', 'vdd_V'])
    np.testing.assert_allclose(table['load_fF'], np.repeat([4, 6, 8, 10, 12], 6))
    np.testing.assert_allclose(table['vdd_V'], np.tile([1.62, 1.66, 1.70, 1.74, 1.78, 1.80], 5))
    for edge in ('rise_ps', 'fall_ps'):
        if not np.all(np.isfinite(table[edge]) & (table[edge] > 0)):
            raise ValueError(f'Invalid {edge} in {path}')
    return table


def panel(layout, seeds, scenario, table, n_packets):
    truths, obs, refs, packets = [], [], [], []
    # Each lane gets a fixed slow-edge load class. The five classes repeat across groups.
    loads = LANES % 5
    vdd = np.unique(table['vdd_V'])
    curves = ((table['rise_ps'] + table['fall_ps']) / 2).reshape(5, -1)
    for seed in seeds:
        truth, readings, packet = trajectory(layout, int(seed), 'mild', n_packets)
        rng = np.random.default_rng([15593, layout, int(seed)])
        ref_noise = rng.normal(0, np.sqrt(R), T)
        shift = np.zeros_like(truth)
        if scenario != 'quiet':
            bounds = (24, 41) if scenario in ('persistent', 'offset') else (8, 17)
            t = int(rng.integers(8, 17))
            droop = np.zeros((T, 4))
            group = int(rng.integers(4))
            while t < T:
                length = int(rng.integers(*bounds))
                droop[t:t + length, group] = rng.uniform(.06, .14)
                t += length
                group = (group + int(rng.integers(1, 4))) % 4
            filtered = np.zeros(4)
            volts = []
            for target in droop:
                filtered += (.25 if scenario == 'offset' else .45) * (target - filtered)
                volts.append(1.8 - filtered)
            for i, load in enumerate(loads):
                physical_group = ((i + 2) % N) // 5 if scenario == 'offset' else GROUPS[i]
                delays = np.interp(np.asarray(volts)[:, physical_group], vdd, curves[load])
                shift[:, i] = (delays - curves[load, -1]) / 31.25
        truths.append(truth + shift)
        obs.append(readings + shift)
        refs.append((truth + shift).mean(axis=1) + ref_noise)
        packets.append(packet)
    return tuple(np.stack(x) for x in (truths, obs, refs, packets))


def estimate_q(data):
    truth = data[0]
    previous = np.pad(truth[:, :-1], ((0, 0), (1, 0), (0, 0)))
    residual = (truth - AR * previous).reshape(-1, N)
    return residual.T @ residual / len(residual)


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def longest(row):
    edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
    return max(edges[1::2] - edges[::2], default=0)


def episode(data, char, q, policy, weights=None, replay=None, rng=None, collect=False):
    truth, readings, common, packets = data
    nominal, left, right, bank = char
    b = len(truth)
    batch = np.arange(b)
    mean = np.zeros((b, N))
    cov = np.zeros((b, N, N))
    alarm = np.zeros((b, 4))
    age = np.zeros((b, N))
    actions = np.empty((b, STEPS), int)
    margins = np.empty((b, T))
    rewards = np.zeros((b, STEPS))
    records = []
    for t in range(T):
        mean *= AR
        cov = AR**2 * cov + q
        # Extra shared process noise for groups with a large recent probe innovation.
        if policy != 'static_trace':
            cov += .00045 * (alarm @ GROUP_KERNEL.reshape(4, -1)).reshape(b, N, N)
        alarm *= .94
        age += 1
        col = cov.mean(axis=2)
        denom = col.mean(axis=1) + R
        mean += col * ((common[:, t] - mean.mean(axis=1)) / denom)[:, None]
        cov -= col[:, :, None] * col[:, None, :] / denom[:, None, None]
        if t % PERIOD == 2:
            step = t // PERIOD
            var = cov.diagonal(axis1=1, axis2=2)
            gain = (cov**2).sum(axis=1) / (var + R)
            norm_gain = np.log(gain / gain.mean(axis=1, keepdims=True))
            if replay is not None:
                action = replay[:, step]
            elif policy in ('static_trace', 'adaptive_trace'):
                action = gain.argmax(axis=1)
            elif policy == 'actor':
                group_mean = (mean @ MEMBERS.T / 5)[:, GROUPS]
                features = np.stack(np.broadcast_arrays(
                    norm_gain, np.minimum(age / 32, 2), alarm[:, GROUPS],
                    group_mean / .15, abs(group_mean - mean.mean(axis=1, keepdims=True)) / .15,
                    (mean - group_mean) / .08, (1 / (left + right))[None, :] / 3,
                    var / .006, (np.min(age.reshape(b, 4, 5), axis=2) / 24)[:, GROUPS]
                ), axis=-1)
                probs = softmax(features @ weights)
                if rng is None:
                    action = probs.argmax(axis=1)
                else:
                    action = np.minimum((rng.random(b)[:, None] > probs.cumsum(axis=1)).sum(axis=1), N-1)
                if collect:
                    records.append((features, probs[batch, action]))
            else:
                raise ValueError(policy)
            actions[:, step] = action
            col = cov[batch, :, action]
            denom = var[batch, action] + R
            innovation = readings[batch, t, action] - mean[batch, action]
            group = GROUPS[action]
            alarm[batch, group] = np.maximum(alarm[batch, group],
                                              np.clip(innovation**2 / denom - 1, 0, 6))
            mean += col * (innovation / denom)[:, None]
            cov -= col[:, :, None] * col[:, None, :] / denom[:, None, None]
            age[batch, action] = 0
        position = nominal + np.rint(SAMPLES * mean) - SAMPLES * truth[:, t] - GRID[0]
        assert position.min() >= 0 and position.max() < len(GRID) - 1
        idx = position.astype(int)
        packet = packets[:, t, None]
        a = bank[packet, LANES, idx]
        z = bank[packet, LANES, idx + 1]
        margins[:, t] = (a + (position - idx) * (z - a)).min(axis=1)
        if t >= 2:
            rewards[:, (t - 2) // PERIOD] -= (margins[:, t] < TARGET) / PERIOD
    bad = margins < TARGET
    metrics = np.column_stack([bad.mean(axis=1), [longest(row) for row in bad], margins.mean(axis=1)])
    if collect:
        features, probs = zip(*records)
        rollout = (np.stack(features, axis=1), np.stack(probs, axis=1), actions, rewards)
    else:
        rollout = None
    return actions, metrics, rollout


def ppo_update(roll, w, v, adam_m, adam_v, n, rng):
    f, old, act, rew = roll
    states = np.concatenate([f.mean(axis=2), np.ones((*rew.shape, 1))], axis=-1)
    values = states @ v
    adv = np.empty_like(rew)
    carry = 0
    future = 0
    for k in reversed(range(STEPS)):
        carry = rew[:, k] + .97 * future - values[:, k] + .97 * .95 * carry
        adv[:, k] = carry
        future = values[:, k]
    target = (adv + values).ravel()
    x = states.reshape(-1, len(v))
    v = np.linalg.solve(x.T @ x + .5 * np.eye(len(v)), x.T @ target)
    adv = (adv.ravel() - adv.mean()) / (adv.std() + 1e-8)
    f, act, old = f.reshape(-1, N, NFEAT), act.ravel(), old.ravel()
    for _ in range(4):
        for ix in np.array_split(rng.permutation(len(act)), 4):
            fx, ax = f[ix], act[ix]
            p = softmax(fx @ w)
            ratio = p[np.arange(len(ix)), ax] / old[ix]
            active = np.where(adv[ix] >= 0, ratio <= 1.2, ratio >= .8)
            loggrad = fx[np.arange(len(ix)), ax] - (p[:, :, None] * fx).sum(axis=1)
            grad = ((adv[ix] * ratio * active)[:, None] * loggrad).mean(axis=0)
            n += 1
            adam_m = .9 * adam_m + .1 * grad
            adam_v = .999 * adam_v + .001 * grad**2
            w += .009 * (adam_m / (1 - .9**n)) / (np.sqrt(adam_v / (1 - .999**n)) + 1e-8)
    return w, v, adam_m, adam_v, n


def train(chars, table, q, seed, updates):
    # Categorical clipped PPO, shared linear lane scorer, linear state-value baseline.
    rng = np.random.default_rng(seed)
    w = np.zeros(NFEAT)
    w[0] = 3
    v = np.zeros(NFEAT + 1)
    adam_m = np.zeros_like(w)
    adam_v = np.zeros_like(w)
    n = 0
    for u in range(updates):
        layout = u % 2
        mode = 'quiet' if (u // 2) % 4 == 3 else 'persistent'
        data = panel(layout, rng.integers(100000, 2000000000, 16), mode, table, len(chars[layout][-1]))
        _, _, roll = episode(data, chars[layout], q, 'actor', weights=w, rng=rng, collect=True)
        w, v, adam_m, adam_v, n = ppo_update(roll, w, v, adam_m, adam_v, n, rng)
    return w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', default=64, type=int)
    parser.add_argument('--test', default=128, type=int)
    parser.add_argument('--out', default=BASE, type=Path)
    parser.add_argument('--delay-table', type=Path, default=BASE / 'delay.csv')
    args = parser.parse_args()
    assert args.test >= 2 and args.test % 2 == 0
    args.out.mkdir(exist_ok=True, parents=True)
    table = load_delay_table(args.delay_table)
    chars = []
    for layout in range(3):
        path = ROOT / f'implementation_results/20261125_package20/channels/layout{layout}.npz'
        chars.append(characterize(Channel(np.load(path)['h'], 32, True)))
    qs = [estimate_q(panel(l, range(100, 132), mode, table, len(chars[l][-1])))
          for l in (0, 1) for mode in ('quiet', 'persistent', 'persistent', 'persistent')]
    q = np.mean(qs, axis=0)
    weights = [train(chars, table, q, s, args.updates) for s in (19, 23, 29)]
    np.savez(args.out / 'models.npz', weights=weights, process_cov=q)
    with (args.out / 'results.csv').open('w') as f:
        out = csv.writer(f)
        out.writerow(['layout', 'condition', 'policy', 'seed', 'bad', 'longest', 'margin', 'probe_diff_vs_fixed'])
        for mode in ('quiet', 'persistent', 'fast'):
            data = panel(2, np.arange(8000, 8000 + args.test), mode, table, len(chars[2][-1]))
            # Exchange schedules across independent episodes within the same condition.
            partner = np.arange(args.test) ^ 1
            fixed, metrics, _ = episode(data, chars[2], q, 'static_trace')
            def save(name, actions, metrics):
                for i, row in enumerate(metrics):
                    out.writerow([2, mode, name, 8000+i, *row, np.mean(actions[i] != fixed[i])])
            save('fixed', fixed, metrics)
            for name, policy, w in [('adaptive', 'adaptive_trace', None)] + [
                    (f'ppo{s}', 'actor', w) for s, w in zip((19, 23, 29), weights)]:
                own, metrics, _ = episode(data, chars[2], q, policy, weights=w)
                save(name, own, metrics)
                yoked, metrics, _ = episode(data, chars[2], q, policy, replay=own[partner])
                save(name + '_replay', yoked, metrics)
            actions, metrics, _ = episode(data, chars[2], q, 'adaptive_trace', replay=fixed)
            save('fixed_adaptive_filter', actions, metrics)
            print(mode, 'finished', flush=True)


if __name__ == '__main__':
    main()
