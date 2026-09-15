"""Differentiable n-step projected Armijo search, Eq. (24)-(27).

Discrete backtracking decisions are detached. The selected continuous branch retains
the gradient and Hessian paths through every step when differentiable=True.
"""
from dataclasses import dataclass, field
import torch
from .config import Config


def metric_diagonal(gradient, probabilities, config: Config):
    delta = config.metric_floor * (1 + gradient.square().mean(dim=-1, keepdim=True))
    return config.mu * probabilities * gradient.square() + delta


def natural_direction(gradient, probabilities, config: Config):
    if config.euclidean:
        direction = -gradient
    else:
        diagonal = metric_diagonal(gradient, probabilities, config)
        v = gradient / diagonal
        direction = -v / (1 + (gradient * v).sum(dim=-1, keepdim=True))
    if config.direction_limit > 0:
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        direction = direction * (config.direction_limit / norm.clamp_min(1e-30)).clamp(max=1)
    return direction


@dataclass
class SearchResult:
    x: torch.Tensor
    start_value: torch.Tensor
    final_value: torch.Tensor
    accepted_steps: int = 0  # summed over samples
    backtracks: int = 0
    trace: list = field(default_factory=list)


@torch.enable_grad()
def line_search(critic, attention, start, gamma, config: Config, *, differentiable=False,
                exploration_factors=None, trace=False):
    if start.ndim not in (1, 2) or not start.is_floating_point():
        raise ValueError("start must have shape [d] or [batch, d]")
    squeeze = start.ndim == 1
    x = start.unsqueeze(0) if squeeze else start
    if not torch.isfinite(x).all() or torch.any((x < 0) | (x > 1)):
        raise ValueError("start must be finite and inside [0, 1]")
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)
    batch = x.shape[0]
    scale = torch.as_tensor(gamma, dtype=x.dtype, device=x.device).detach()
    scale = torch.broadcast_to(scale, (batch,)).abs().clamp_min(config.gamma_floor)
    if not torch.isfinite(scale).all():
        raise ValueError("gamma must be finite")
    factors = torch.ones((config.search_steps, batch), dtype=x.dtype, device=x.device)
    if exploration_factors is not None:
        factors = torch.broadcast_to(torch.as_tensor(exploration_factors, dtype=x.dtype,
                                     device=x.device), factors.shape).detach()
        if not torch.isfinite(factors).all() or not (factors > 0).all():
            raise ValueError("exploration factors must be positive and finite")
    initial = critic(x).detach()
    # A_eta(x_i) is held fixed within one LSA call, as the Eq. (26) tuple specifies.
    probabilities = (attention(x) if attention is not None and config.use_attention
                     and not config.euclidean else torch.full_like(x, 1 / x.shape[-1]))
    if not differentiable:
        probabilities = probabilities.detach()
    accepted_count = backtrack_count = 0
    records = []
    for step in range(config.search_steps):
        if not differentiable:
            x = x.detach().requires_grad_(True)
        if step and config.attention_refresh and attention is not None and config.use_attention and not config.euclidean:
            probabilities = attention(x)
            if not differentiable:
                probabilities = probabilities.detach()
        value = critic(x)
        gradient, = torch.autograd.grad(value.sum(), x, create_graph=differentiable)
        direction = natural_direction(gradient, probabilities, config)
        # Rejected candidate graphs are unnecessary. Only alpha and an accept mask
        # cross this no_grad boundary; recompute the selected candidate below.
        with torch.no_grad():
            alpha = config.step_size * scale * factors[step]
            selected = torch.zeros_like(alpha)
            searching = torch.ones(batch, dtype=torch.bool, device=x.device)
            before = x.detach()
            for _ in range(config.max_backtracks):
                candidate = (before + alpha[:, None] * direction.detach()).clamp(0, 1)
                displacement = candidate - before
                slope = (gradient.detach() * displacement).sum(dim=-1)
                finite = torch.isfinite(candidate).all(dim=-1) & torch.isfinite(slope)
                movable = (displacement.square().sum(dim=-1) > 1e-24) & (slope < 0) & finite
                searching &= movable
                if not bool(searching.any()):
                    break
                candidate_value = critic(candidate)
                accept = searching & torch.isfinite(candidate_value) & (
                    candidate_value <= value.detach() + config.armijo * slope)
                selected = torch.where(accept, alpha, selected)
                searching &= ~accept
                backtrack_count += int(searching.sum())
                if not bool(searching.any()):
                    break
                alpha = torch.where(searching, alpha * config.rho, alpha)
            accepted = selected > 0
            accepted_count += int(accepted.sum())
        if not bool(accepted.any()):
            break
        # Retains d(grad_x Q)/dx and every earlier accepted step during training.
        candidate = (x + selected[:, None] * direction).clamp(0, 1)
        new_x = torch.where(accepted[:, None], candidate, x)
        if trace:
            with torch.no_grad():
                records.append({"before": x.detach().clone(), "after": new_x.detach().clone(),
                                "value_before": value.detach().clone(),
                                "value_after": critic(new_x).detach().clone(),
                                "slope": (gradient.detach() * (new_x-x)).sum(-1).detach().clone(),
                                "alpha": selected.clone(), "accepted": accepted.clone()})
        x = new_x
    final = critic(x)
    if not differentiable:
        x, final = x.detach(), final.detach()
    if squeeze:
        x, initial, final = x.squeeze(0), initial.squeeze(0), final.squeeze(0)
    return SearchResult(x, initial, final, accepted_count, backtrack_count, records)
