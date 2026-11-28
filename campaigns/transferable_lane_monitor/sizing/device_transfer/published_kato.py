"""Low-budget sequential adaptation of KATO (DAC 2024)."""
import numpy as np
from scipy.special import log_ndtr, ndtr
import torch
from torch import nn

from blind_width_policy import generate_candidates
from published_dnnopt import mlp
from published_scbo import improvement, preferred


class NeuralKernel(nn.Module):
    """An exponentiated nonnegative sum of kernels on learned projections."""

    def __init__(self, dimension, rng):
        super().__init__()
        projections = np.repeat(np.eye(dimension)[None, ...], 3, axis=0)
        projections += rng.normal(0., .025, size=projections.shape)
        self.projections = nn.Parameter(torch.from_numpy(projections))
        self.log_length = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.log_weight = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.log_amplitude = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, left, right):
        a = left @ self.projections
        b = right @ self.projections
        square = (a[:, :, None, :] - b[:, None, :, :]).square().mean(dim=-1)
        norm = square / self.log_length.clamp(-4., 3.).exp()[:, None, None].square()
        q = (5. * norm[1] + 1e-20).sqrt()
        bases = torch.stack(((-.5 * norm[0]).exp(),
                             (1 + q + q.square() / 3) * (-q).exp(),
                             1 / (1 + .5 * norm[2])))
        weights = self.log_weight.softmax(dim=0)[:, None, None]
        return (self.log_amplitude.clamp(-5., 3.) + (weights * (bases - 1)).sum(dim=0)).exp()


class NeuralGP:
    def __init__(self, x, y, rng, steps):
        self.x = torch.from_numpy(x)
        raw = torch.from_numpy(y)
        self.center = raw.mean(0)
        self.scale = raw.std(0, unbiased=False).clamp_min(.02)
        z = (raw - self.center) / self.scale
        self.kernel = NeuralKernel(x.shape[1], rng)
        log_noise = nn.Parameter(torch.tensor(-4., dtype=torch.float64))
        eye = torch.eye(len(x), dtype=torch.float64)
        optimizer = torch.optim.Adam([*self.kernel.parameters(), log_noise], lr=.025)

        def factor():
            return torch.linalg.cholesky(self.kernel(self.x, self.x) +
                                         eye * log_noise.clamp(-10., 1.).exp())

        for _ in range(steps):
            chol = factor()
            white = torch.linalg.solve_triangular(chol, z, upper=False)
            loss = (.5 * white.square().sum() + z.shape[1] * chol.diagonal().log().sum()) / len(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.kernel.requires_grad_(False)
        with torch.no_grad():
            self.chol = factor()
            self.white = torch.linalg.solve_triangular(self.chol, z, upper=False)

    def posterior(self, x):
        k = self.kernel(self.x, x)
        projected = torch.linalg.solve_triangular(self.chol, k, upper=False)
        diag = self.kernel.log_amplitude.clamp(-5., 3.).exp()
        return projected.T @ self.white, (diag - projected.square().sum(0)).clamp_min(1e-8)

    def predict(self, x):
        mean, var = self.posterior(x)
        return self.center + mean * self.scale, var.sqrt()[:, None] * self.scale


class AlignedGP:
    def __init__(self, source, x, y, rng):
        self.source = source
        self.encoder = mlp(rng, (x.shape[1], 32, source.x.shape[1]), nn.Sigmoid)
        self.decoder = mlp(rng, (len(source.center), 32, y.shape[1]), nn.Sigmoid)
        self.center = y.mean(0)
        self.scale = y.std(0, unbiased=False).clamp_min(.02)
        self.eye = torch.eye(y.shape[1], dtype=torch.float64)
        self.log_noise = nn.Parameter(torch.tensor(-3., dtype=torch.float64))
        params = [*self.encoder.parameters(), *self.decoder.parameters(), self.log_noise]
        optimizer = torch.optim.Adam(params, lr=.02)
        standardized = (y - self.center) / self.scale
        for _ in range(65):
            mean, covariance = self.posterior(x)
            chol = torch.linalg.cholesky(covariance)
            white = torch.linalg.solve_triangular(chol, (standardized - mean)[..., None], upper=False)
            loss = (.5 * white.square().sum() + chol.diagonal(dim1=-2, dim2=-1).log().sum()) / len(x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.)
            optimizer.step()

    def posterior(self, x):
        source_mean, source_variance = self.source.posterior(self.encoder(x).sigmoid())
        hidden = self.decoder[0](source_mean).sigmoid()
        mean = self.decoder[2](hidden)
        # J diag(source variance) J^T. The five standardised source outputs
        # are modeled independently and share the same scalar predictive variance.
        jacobian = (self.decoder[2].weight * (hidden * (1. - hidden))[:, None, :]) @ self.decoder[0].weight
        covariance = (jacobian @ jacobian.mT) * source_variance[:, None, None]
        covariance = covariance + self.eye * self.log_noise.clamp(-9., 2.).exp()
        return mean, covariance

    def predict(self, x):
        mean, covariance = self.posterior(x)
        return self.center + mean * self.scale, covariance.diagonal(dim1=-2, dim2=-1).sqrt() * self.scale


def donor_for(rows, target_dimension, output_count, output_sizes):
    return min(rows, key=lambda row: (abs(len(row["x"][0]) - target_dimension),
                                      abs(output_sizes[row["name"]] - output_count), row["name"]))


def source_gp(row, lower, upper):
    x = (np.log(np.array(row["x"])) - lower) / (upper - lower)
    x = np.clip(x, 0., 1.)
    return NeuralGP(x, np.asarray(row["y"]), np.random.default_rng(7183), steps=95)


def mace_front(mean, std, best):
    std = np.maximum(std, 1e-9)
    diff = best - mean[:, 0]
    u = diff / std[:, 0]
    probability = ndtr(u)
    ei = diff * probability + std[:, 0] * np.exp(-.5 * u ** 2) / np.sqrt(2 * np.pi)
    # Put the optimistic score on a nonnegative scale before feasibility weighting.
    optimistic = -mean[:, 0] + 2. * std[:, 0]
    optimistic -= optimistic.min()
    log_feasible = log_ndtr(-mean[:, 1:] / std[:, 1:]).sum(axis=1)
    tiny = np.finfo(float).tiny
    scores = np.column_stack((np.log(np.maximum(optimistic, tiny)), log_ndtr(u),
                              np.log(np.maximum(ei, tiny)))) + log_feasible[:, None]
    dominates = (scores[:, None, :] >= scores[None, :, :]).all(axis=2)
    dominates &= (scores[:, None, :] > scores[None, :, :]).any(axis=2)
    return np.flatnonzero(~dominates.any(axis=0))


def select_kato(source, x, values, rng, decisions):
    weights = np.array([len(source.x), 5], dtype=float)
    incumbent = values[preferred(values[:5])[0]]
    for decision, outcome in zip(decisions, values[5:]):
        if improvement(incumbent, outcome):
            weights[0 if decision["model"] == "aligned" else 1] += 1
            incumbent = outcome
    # For batch size one, sample the proposal model with the paper's batch proportions.
    model = ("aligned", "target")[rng.choice(2, p=weights / weights.sum())]
    if model == "aligned":
        predictor = AlignedGP(source, torch.from_numpy(x), torch.from_numpy(values), rng)
    else:
        predictor = NeuralGP(x, values, rng, steps=50)
    candidates = generate_candidates(x, values, np.zeros(x.shape[1]), np.ones(x.shape[1]), rng)
    with torch.no_grad():
        mean, std = predictor.predict(torch.from_numpy(candidates))
    front = mace_front(mean.numpy(), std.numpy(), incumbent[0])
    return candidates[rng.choice(front)], {"model": model, "weights": weights.tolist(), "front_size": len(front)}
