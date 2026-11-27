"""Independent evaluation with a discrete supply-droop model and held-out packet seeds."""
import argparse
import csv
from pathlib import Path
import numpy as np
from study import load_delay_table
from study import (AR, BASE, Channel, GRID, GROUPS, LANES, N, PERIOD, R, ROOT, SAMPLES,
                   STEPS, T, TARGET, characterize, episode, longest, panel, softmax)
from pilot import curves, FAMILIES


def hidden_model(table):
    volts = np.unique(table['vdd_V'])
    delay = ((table['rise_ps'] + table['fall_ps']) / 2).reshape(5, -1)
    targets = [np.zeros(N)]
    for group in range(4):
        for v in (1.68, 1.72):
            delta = np.array([(np.interp(v, volts, d) - d[-1]) / 31.25 for d in delay])
            targets.append((GROUPS == group) * delta[LANES % 5])
    # States are no droop or (group, magnitude). Priors are fixed before evaluation.
    k = len(targets)
    trans = np.eye(k) * .965
    trans[0, 1:] = .035 / (k-1)
    for i in range(1, k):
        trans[i, 0] = .007
        other = [j for j in range(1, k) if (j-1)//2 != (i-1)//2]
        trans[i, other] = .028 / len(other)
    np.testing.assert_allclose(trans.sum(axis=1), 1)
    return np.array(targets), trans


def run_modes(data, char, q, model, replay=None, weights=None, rng=None, collect=False):
    truth, readings, common, packets = data
    nominal, left, right, bank = char
    targets, transition = model
    b, k = len(truth), len(targets)
    batch = np.arange(b)
    prob = np.zeros((b, k))
    prob[:, 0] = 1
    base = np.zeros((b, k, N))
    shift = np.zeros_like(base)
    cov = np.zeros((b, N, N))
    margins = np.empty((b, T))
    actions = np.zeros((b, STEPS), int)
    age = np.zeros((b, N))
    records = []
    for t in range(T):
        age += 1
        flow = prob[:, :, None] * transition
        prob = flow.sum(axis=1)
        mix = flow.swapaxes(1, 2) / np.maximum(prob[:, :, None], 1e-100)
        base = AR * (mix @ base)
        shift = .55 * (mix @ shift) + .45 * targets
        cov = AR**2 * cov + q
        column = cov.mean(axis=2)
        denom = column.mean(axis=1) + R
        error = common[:, t, None] - (base + shift).mean(axis=2)
        prob = softmax(np.log(prob + 1e-100) - .5 * error**2 / denom[:, None])
        base += error[:, :, None] * (column / denom[:, None])[:, None, :]
        cov -= column[:, :, None] * column[:, None, :] / denom[:, None, None]
        if t % PERIOD == 2:
            step = t // PERIOD
            if replay is None:
                # Approximate marginal state covariance includes between-mode dispersion.
                modes = base + shift
                center = (prob[:, None, :] @ modes)
                deviations = modes - center
                total = cov + deviations.swapaxes(1, 2) @ (prob[:, :, None] * deviations)
                var = total.diagonal(axis1=1, axis2=2)
                gain = (total**2).sum(axis=1) / (var + R)
                if weights is None:
                    action = gain.argmax(axis=1)
                else:
                    group_p = prob[:, 1:].reshape(b, 4, 2).sum(axis=2)[:, GROUPS]
                    features = np.stack(np.broadcast_arrays(
                        np.log(gain / gain.mean(axis=1, keepdims=True)), group_p,
                        group_p * (1 - group_p), (var - cov.diagonal(axis1=1, axis2=2)) / .006,
                        np.minimum(age / 32, 2), var / .006,
                        (1 / (left + right))[None, :] / 3,
                        (center[:, 0] - center.mean(axis=2)) / .15,
                        (np.min(age.reshape(b, 4, 5), axis=2) / 24)[:, GROUPS]
                    ), axis=-1)
                    probs = softmax(features @ weights)
                    if rng is None:
                        action = probs.argmax(axis=1)
                    else:
                        action = np.minimum((rng.random(b)[:, None] > probs.cumsum(axis=1)).sum(axis=1), N-1)
                    if collect:
                        records.append((features, probs[batch, action]))
            else:
                action = replay[:, step]
            actions[:, step] = action
            age[batch, action] = 0
            column = cov[batch, :, action]
            denom = column[batch, action] + R
            error = readings[batch, t, action, None] - (base + shift)[batch, :, action]
            prob = softmax(np.log(prob + 1e-100) - .5 * error**2 / denom[:, None])
            base += error[:, :, None] * (column / denom[:, None])[:, None, :]
            cov -= column[:, :, None] * column[:, None, :] / denom[:, None, None]
        mean = (prob[:, None, :] @ (base + shift))[:, 0]
        position = nominal + np.rint(SAMPLES * mean) - SAMPLES * truth[:, t] - GRID[0]
        assert position.min() >= 0 and position.max() < len(GRID) - 1
        idx = position.astype(int)
        a = bank[packets[:, t, None], LANES, idx]
        z = bank[packets[:, t, None], LANES, idx + 1]
        margins[:, t] = (a + (position - idx) * (z - a)).min(axis=1)
    bad = margins < TARGET
    metrics = np.column_stack([bad.mean(axis=1), [longest(row) for row in bad], margins.mean(axis=1)])
    if collect:
        features, probs = zip(*records)
        rewards = -np.pad(bad[:, 2:].astype(float), ((0, 0), (0, 2))).reshape(b, STEPS, PERIOD).mean(axis=2)
        return actions, metrics, (np.stack(features, axis=1), np.stack(probs, axis=1), actions, rewards)
    return actions, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=BASE)
    p.add_argument('--test', type=int, default=256)
    p.add_argument('--delay-table', type=Path, default=BASE / 'delay.csv')
    args = p.parse_args()
    assert args.test % 2 == 0 and args.test >= 2
    fits = np.load(args.out / 'models.npz')
    table = load_delay_table(args.delay_table)
    model = hidden_model(table)
    channel = Channel(np.load(ROOT / 'implementation_results/20261125_package20/channels/layout2.npz')['h'], 32, True)
    nominal, left, right, _ = characterize(channel)
    bank = np.array([curves(channel, 57000+s, f) for s in range(8) for f in FAMILIES])
    char = (nominal, left, right, bank)
    # Estimate nuisance covariance only from quiet development trajectories.
    from study import estimate_q
    q0 = np.mean([estimate_q(panel(l, range(100, 132), 'quiet', table, len(bank))) for l in (0, 1)], axis=0)
    q = fits['process_cov']
    with (args.out / 'confirmation.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'seed', 'policy', 'bad', 'longest', 'margin'])
        for mode in ('quiet', 'persistent', 'fast', 'offset'):
            for start in range(0, args.test, 32):
                seeds = 18000 + np.arange(start, min(start + 32, args.test))
                partner = np.arange(len(seeds)) ^ 1
                data = panel(2, seeds, mode, table, len(bank))
                def save(name, metrics):
                    for seed, row in zip(seeds, metrics):
                        writer.writerow([mode, seed, name, *row])
                fixed, metrics, _ = episode(data, char, q, 'static_trace')
                save('fixed', metrics)
                for name, policy, w in [('adaptive', 'adaptive_trace', None)] + [
                        (f'ppo{s}', 'actor', w) for s, w in zip((19, 23, 29), fits['weights'])]:
                    own, metrics, _ = episode(data, char, q, policy, weights=w)
                    save(name, metrics)
                    _, metrics, _ = episode(data, char, q, policy, replay=own[partner])
                    save(name + '_replay', metrics)
                _, metrics, _ = episode(data, char, q, 'adaptive_trace', replay=fixed)
                save('fixed_adaptive', metrics)
                own, metrics = run_modes(data, char, q0, model)
                save('mode_adaptive', metrics)
                _, metrics = run_modes(data, char, q0, model, replay=own[partner])
                save('mode_replay', metrics)
                _, metrics = run_modes(data, char, q0, model, replay=fixed)
                save('mode_fixed', metrics)
            print(mode, 'confirmed', flush=True)


if __name__ == '__main__':
    main()
