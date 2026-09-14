"""ALARM coordinate attention and a differentiable, damped metric step.

ALARM_V0, Sections 4.2.2--4.2.3, Eqs. (24), (27): scalar-coordinate
self-attention -> fully connected logits -> diagonal softmax weights.
This is not a Transformer surrogate or a topology encoder.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class SearchOptions:
    attention: str = 'learned'
    line_search: str = 'none'
    attention_steps: int = 40
    attention_lr: float = .01
    step_size: float = .15
    search_steps: int = 6
    trust_radius: float = .25
    attention_floor: float = .02
    damping: float = 1e-4
    relative_damping: float = .01
    backtracks: int = 8
    armijo_c1: float = 1e-4

    def __post_init__(self):
        if self.attention not in ('off', 'learned'):
            raise ValueError('attention must be off or learned')
        if self.line_search not in ('none', 'armijo'):
            raise ValueError('line_search must be none or armijo')
        for name in ('attention_steps', 'search_steps', 'backtracks'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('attention_lr', 'step_size', 'trust_radius', 'damping'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 <= self.attention_floor < 1:
            raise ValueError('attention_floor must be in [0, 1)')
        if not math.isfinite(self.relative_damping) or self.relative_damping < 0:
            raise ValueError('relative_damping must be finite and nonnegative')
        if not 0 < self.armijo_c1 < 1:
            raise ValueError('armijo_c1 must be in (0, 1)')


class CoordinateAttention(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.logits = nn.Linear(dimensions, dimensions)
        # Every run starts at the matched uniform-metric ablation.
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, u):
        # A design has d coordinate tokens, each with scalar Q=K=V (d_k=1).
        # Softmax is across coordinates, never across independent designs.
        tokens = u.unsqueeze(-1)
        scores = tokens @ tokens.transpose(-1, -2)
        attended = (scores.softmax(dim=-1) @ tokens).squeeze(-1)
        return self.logits(attended).softmax(dim=-1)


def metric_matrix(gradient, probabilities, options):
    """Eq. (24), with mu=d and an explicit strictly positive ridge.

    mu=d makes uniform p_i=1/d recover the old diagonal coefficient 1.
    The uniform mixture prevents extreme weights; the ridge protects inactive
    coordinates. No claim that the undamped rank-one pullback is invertible.
    """
    dimensions = gradient.shape[-1]
    weights = (1 - options.attention_floor) * dimensions * probabilities + options.attention_floor
    square = gradient.square()
    diagonal = weights * square + options.relative_damping * square.mean(-1, keepdim=True) + options.damping
    return gradient.unsqueeze(-1) * gradient.unsqueeze(-2) + torch.diag_embed(diagonal)


def metric_direction(gradient, probabilities, options):
    metric = metric_matrix(gradient, probabilities, options)
    direction = -torch.linalg.solve(metric, gradient.unsqueeze(-1)).squeeze(-1)
    return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8) * options.step_size


def project(candidate, center, options):
    lower = (center - options.trust_radius).clamp(0, 1)
    upper = (center + options.trust_radius).clamp(0, 1)
    return torch.maximum(torch.minimum(candidate, upper), lower)
