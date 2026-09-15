"""Critic and coordinate-token self-attention from Figure 6."""
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, dimension: int, width: int = 32, *, hidden=None, activation='tanh',
                 dropout=0.0, standardize_inputs=False):
        super().__init__()
        widths = hidden or (width, width)
        layers = []
        previous = dimension
        for index, size in enumerate(widths):
            layers.append(nn.Linear(previous, size))
            if index == 1 and dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.GELU())
            previous = size
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)
        self.standardize_inputs = standardize_inputs
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_scale", torch.tensor(1.0))

    def standardized(self, x):
        if self.standardize_inputs:
            x = (x-0.5) * (12 ** 0.5)
        return self.head(self.features(x)).squeeze(-1)

    def forward(self, x):
        # All search gradients/values stay in actual scalar-objective units.
        return self.standardized(x) * self.target_scale + self.target_mean

    @torch.no_grad()
    def set_target_stats(self, values, normalize=True):
        self.target_mean.copy_(values.mean() if normalize else values.new_tensor(0.0))
        self.target_scale.copy_(values.std(unbiased=False).clamp_min(1e-8)
                                if normalize else values.new_tensor(1.0))

    @torch.no_grad()
    def transfer_head(self, previous):
        """Copy only the output layer, adjusting its units to new target statistics."""
        self.head.weight.copy_(previous.head.weight * previous.target_scale / self.target_scale)
        self.head.bias.copy_((previous.head.bias * previous.target_scale
                              + previous.target_mean - self.target_mean) / self.target_scale)


class Attention(nn.Module):
    def __init__(self, dimension: int, width: int = 0):
        super().__init__()
        self.fc = (nn.Linear(dimension, dimension) if width == 0 else
                   nn.Sequential(nn.Linear(dimension, width), nn.Tanh(), nn.Linear(width, dimension)))

    def forward(self, x):
        # Coordinate i is scalar token [x_i]; Q=K=V=x, d_k=1.
        # Supports independent designs [batch, d], never mixes the batch axis.
        scores = x.unsqueeze(-1) * x.unsqueeze(-2)
        attended = (torch.softmax(scores, dim=-1) @ x.unsqueeze(-1)).squeeze(-1)
        return torch.softmax(self.fc(attended), dim=-1)


class ProjectedAttention(nn.Module):
    """Reference Q/K/V projections with key-axis softmax and nondegenerate normalization."""
    def __init__(self, dimension, dk=256, dropout=0.1):
        super().__init__()
        self.dk = dk
        self.query = nn.Linear(1, dk)
        self.key = nn.Linear(1, dk)
        self.value = nn.Linear(1, 1)
        self.drop_out = nn.Dropout(dropout)
        # A single scalar token cannot be normalized over its one feature: it
        # would become identically zero. Normalize the vector of design tokens.
        self.layer_norm = nn.LayerNorm(dimension) if dimension > 1 else nn.Identity()
        self.linear_out = nn.Linear(dimension, dimension)

    def forward(self, x):
        tokens = x.unsqueeze(-1)
        q, k, v = self.query(tokens), self.key(tokens), self.value(tokens)
        weights = torch.softmax((q @ k.transpose(-1, -2)) / self.dk**0.5, dim=-1)
        residual = x + self.drop_out((weights @ v).squeeze(-1))
        return torch.softmax(self.linear_out(self.layer_norm(residual)), dim=-1)


def make_networks(dimension, config, seed):
    # Construct on CPU in an isolated RNG context: no change to caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        critic = Critic(dimension, config.critic_width, hidden=config.critic_hidden,
                        activation=config.critic_activation, dropout=config.critic_dropout,
                        standardize_inputs=config.standardize_inputs)
        attention = (Attention(dimension, config.attention_width) if config.attention_kind == 'scalar'
                     else ProjectedAttention(dimension, config.attention_dk, config.attention_dropout))
        if config.network_init == 'kaiming_relu':
            for module in list(critic.modules()) + list(attention.modules()):
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)
    dtype = getattr(torch, config.dtype)
    return (critic.to(device=config.device, dtype=dtype),
            attention.to(device=config.device, dtype=dtype))
