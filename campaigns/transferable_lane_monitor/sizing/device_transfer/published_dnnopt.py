"""DNN-Opt Algorithm 1 adapted to the G3 single-design query contract."""
import numpy as np
import torch
from torch import nn


def mlp(rng, sizes, activation):
    # Explicit local initialization so concurrent trials and ledger replay agree.
    layers = []
    for index, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:])):
        linear = nn.Linear(inputs, outputs, dtype=torch.float64)
        limit = np.sqrt(6 / (inputs + outputs))
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(rng.uniform(-limit, limit, (outputs, inputs))))
            linear.bias.zero_()
        layers.append(linear)
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def merit(values):
    # G3 labels are log ratios. Use positive, clipped violations of the four limits.
    return values[..., 0] + torch.clamp(4 * values[..., 1:], min=0., max=1.).sum(dim=-1)


def select_dnnopt(x, values, rng, _decisions):
    widths, labels = torch.from_numpy(x), torch.from_numpy(values)
    dimension = x.shape[1]
    n = len(x)
    critic = mlp(rng, (2 * dimension, 64, 64, labels.shape[1]), nn.SiLU)
    actor = mlp(rng, (dimension, 64, 64, dimension), nn.SiLU)
    # Targets for (x_i, x_j-x_i) are the actual metrics of x_j.
    first = widths.repeat_interleave(n, dim=0)
    second = widths.repeat((n, 1))
    pairs = torch.cat((first, second - first), dim=-1)
    targets = labels.repeat((n, 1))
    center = labels.mean(dim=0)
    scale = labels.std(dim=0, unbiased=False).clamp_min(.02)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.006)
    for _ in range(100):
        idx = rng.integers(n * n, size=min(128, n * n))
        prediction = critic(pairs[idx])
        loss = (prediction - (targets[idx] - center) / scale).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    critic.requires_grad_(False)
    elite_idx = torch.argsort(merit(labels), stable=True)[:max(3, int(np.ceil(n * .2)))]
    elite = widths[elite_idx]
    lower, upper = elite.amin(dim=0), elite.amax(dim=0)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.005)
    for _ in range(60):
        delta = .5 * torch.tanh(actor(elite))
        point = elite + delta
        violation = (lower - point).clamp_min(0.) + (point - upper).clamp_min(0.)
        prediction = center + scale * critic(torch.cat((elite, delta), dim=-1))
        loss = (merit(prediction) + 50 * torch.linalg.vector_norm(violation, dim=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        noisy = elite + .5 * torch.tanh(actor(elite))
        noisy += torch.from_numpy(rng.normal(0., .08, size=tuple(elite.shape)))
        candidates = noisy.clamp(0., 1.)
        pred = center + scale * critic(torch.cat((elite, candidates - elite), dim=-1))
        index = int(merit(pred).argmin())
    return candidates[index].numpy(), {"elite_count": len(elite)}
