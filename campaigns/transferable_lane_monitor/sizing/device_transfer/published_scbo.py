"""Sequential SCBO search in the nominal-normalized width box."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def preferred(values):
    """SCBO's feasible-first order, with summed positive violation otherwise."""
    violation = np.maximum(values[:, 1:], 0).sum(axis=1)
    return np.lexsort((np.where(violation == 0, values[:, 0], violation), violation > 0))


def improvement(before, after, tolerance=0.):
    old, new = (np.maximum(row[1:], 0).sum() for row in (before, after))
    if new or old:
        return new < old
    return after[0] < before[0] - tolerance * abs(before[0])


def trust_region(values, dimension):
    # All runs start optimization after the nominal evaluation and four paid starts.
    length, successes, failures = .8, 0, 0
    reset_at = 0
    best = values[preferred(values[:5])[0]]
    for index, value in enumerate(values[5:], start=5):
        if improvement(best, value, tolerance=.001):
            successes, failures = successes + 1, 0
            best = value
        else:
            successes, failures = 0, failures + 1
        if successes == 10:
            length, successes = min(1.6, 2 * length), 0
        if failures == max(4, dimension):
            length, failures = length / 2, 0
        if length < .5 ** 7:
            length, successes, failures = .8, 0, 0
            reset_at, best = index, value
    return length, reset_at


def matern(distance):
    q = np.sqrt(5.) * distance
    return (1 + q + q ** 2 / 3) * np.exp(-q)


def posterior_draw(x, targets, candidates, rng):
    """One joint GP draw, preserving correlations between candidate points."""
    center = targets.mean()
    scale = max(targets.std(), 1e-6)
    y = (targets - center) / scale
    # Bounded ARD Matérn-5/2 GP with learned noise, in standardized units.
    differences = (x[:, None, :] - x[None, :, :]) ** 2
    diagonal = np.diag_indices(len(x))

    def objective(params):
        inverse_squared = np.exp(-2 * params[:-2])
        delta = np.sqrt(np.maximum(differences @ inverse_squared, 0.))
        amplitude = np.exp(params[-2])
        signal = amplitude * matern(delta)
        noise = np.exp(params[-1])
        k = signal.copy()
        k[diagonal] += noise
        factor = cho_factor(k, lower=True, check_finite=False)
        alpha = cho_solve(factor, y, check_finite=False)
        loss = .5 * (y @ alpha) + np.log(factor[0].diagonal()).sum()
        inverse = cho_solve(factor, np.eye(len(x)), check_finite=False)
        weight = .5 * (inverse - np.outer(alpha, alpha))
        q = np.sqrt(5.) * delta
        length_gradient = ((5. / 3.) * (1 + q) * np.exp(-q))[..., None]
        derivatives = amplitude * length_gradient * differences * inverse_squared
        gradient = np.concatenate(((weight[..., None] * derivatives).sum(axis=(0, 1)),
                                   [(weight * signal).sum(), noise * np.trace(weight)]))
        return loss, gradient

    lower = np.log([.005] * x.shape[1] + [.01, 1e-8])
    upper = np.log([4.] * x.shape[1] + [100., 1e-3])
    initial = np.log([.4] * x.shape[1] + [1., 1e-4])
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B",
                   bounds=list(zip(lower, upper)), options={"maxiter": 35})
    params = fit.x if np.all(np.isfinite(fit.x)) else initial
    length = np.exp(params[:-2])
    amplitude = np.exp(params[-2])
    obs = x / length
    test = candidates / length
    train_kernel = amplitude * matern(cdist(obs, obs))
    train_kernel[diagonal] += np.exp(params[-1])
    chol = cholesky(train_kernel, lower=True, check_finite=False)
    cross = amplitude * matern(cdist(obs, test))
    whitened = solve_triangular(chol, cross, lower=True, check_finite=False)
    mean = whitened.T @ solve_triangular(chol, y, lower=True, check_finite=False)
    covariance = amplitude * matern(cdist(test, test)) - whitened.T @ whitened
    covariance.flat[::len(test) + 1] += 1e-8
    sample = cholesky(covariance, lower=True, check_finite=False) @ rng.standard_normal(len(test))
    return center + scale * (mean + sample)


def select_scbo(x, values, rng, _decisions):
    dimension = x.shape[1]
    length, restart_at = trust_region(values, dimension)
    x, values = x[restart_at:], values[restart_at:]
    center = x[preferred(values)[0]]
    lo = np.maximum(0., center - length / 2)
    hi = np.minimum(1., center + length / 2)
    sampler = qmc.Sobol(dimension, scramble=True, seed=int(rng.integers(2**32)))
    perturb = lo + sampler.random_base2(8) * (hi - lo)
    mask = rng.random(perturb.shape) < min(20 / dimension, 1.)
    empty = np.flatnonzero(~mask.any(axis=1))
    mask[empty, rng.integers(dimension, size=len(empty))] = True
    candidates = np.where(mask, perturb, center)
    draws = np.column_stack([posterior_draw(x, col, candidates, rng) for col in values.T])
    return candidates[preferred(draws)[0]], {"trust_region_length": length, "restart_at": restart_at}
