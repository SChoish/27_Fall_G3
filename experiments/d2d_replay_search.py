"""Online neural response ensemble for budgeted D2D sizing, with measured long-packet corrections."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor

from d2d_sizing import circuit


def targets(result):
    """Log TX energy and signed worst violations, keeping the two lanes distinct."""
    return np.array([np.log(result['energy_pj_bit']),
                     *(1 - np.asarray(result['margins']) / .08),
                     result['overshoot_fraction'] / .2 - 1])


def long_correction(candidates, pairs, dimensions=3):
    """Transfer the observed scenario-set mismatch locally; zero away from observations."""
    if not pairs:
        return np.zeros((len(candidates), dimensions))
    anchors, differences = zip(*pairs)
    weights = np.exp(-.5 * cdist(candidates, anchors, 'sqeuclidean') / .4**2)
    return weights @ np.maximum(differences, 0) / np.maximum(1., weights.sum(axis=1, keepdims=True))


def propose(rows, pairs, seed, *, local_edits=False):
    rng = np.random.default_rng(np.random.SeedSequence([seed, len(rows), 371]))
    history = np.array([circuit.encode(row['x']) for row in rows])
    good = [i for i, row in enumerate(rows) if row['oracle']['valid']]
    pool = rng.uniform(size=(1024, len(circuit.NAMES)))
    if len(good) < 4:
        return circuit.decode(pool[0])
    order = sorted(good, key=lambda i: circuit.cost(rows[i]['oracle']))
    anchors = history[rng.choice(order[:3], size=1024)]
    pool = np.vstack([pool, np.clip(anchors + rng.normal(0, .12, anchors.shape), 0, 1)])
    if local_edits:
        # Keep promising joint settings intact while varying one or two device sizes.
        # Reserve the same number of candidate predictions as the unmodified proposer.
        bases = anchors.copy()
        axes = rng.integers(len(circuit.NAMES), size=(len(bases), 2))
        delta = rng.uniform(-.24, .16, size=axes.shape)
        delta[:len(bases)//2, 1] = 0
        np.add.at(bases, (np.arange(len(bases))[:, None], axes), delta)
        pool[1024:] = np.clip(bases, 0, 1)
    pool = pool[cdist(pool, history, 'chebyshev').min(axis=1) > 1e-5]
    y = np.array([targets(rows[i]['oracle']) for i in good])
    center = y.mean(axis=0)
    scale = y.std(axis=0).clip(.05)
    estimates = []
    for _ in range(5):
        indices = rng.choice(len(good), size=max(4, int(.85*len(good))), replace=False)
        # Small regularized models; each fits only already-paid design observations.
        net = MLPRegressor(hidden_layer_sizes=(12,), activation='tanh', solver='lbfgs',
                           alpha=.3, max_iter=350, random_state=int(rng.integers(2**31)))
        net.fit(history[good][indices], ((y-center)/scale)[indices])
        estimates.append(net.predict(pool)*scale + center)
    estimates = np.asarray(estimates)
    estimates[:, :, 1:] += long_correction(pool, pairs, y.shape[1]-1)
    # Rank violation first. Among predicted feasible candidates minimize median log energy.
    violation = np.maximum(0., np.quantile(estimates[:, :, 1:], .75, axis=0).max(axis=1))
    log_energy = np.median(estimates[:, :, 0], axis=0)
    return circuit.decode(pool[np.lexsort((log_energy, violation))[0]])
