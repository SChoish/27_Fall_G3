"""Implementation defaults; the manuscript does not supply tuned hyperparameters."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Config:
    initial_designs: int = 60
    max_evaluations: int = 100
    critic_epochs: int = 200
    attention_epochs: int = 8
    critic_width: int = 32
    critic_hidden: tuple[int, ...] | None = None
    critic_activation: str = "tanh"
    critic_dropout: float = 0.0
    standardize_inputs: bool = False
    network_init: str = "default"
    critic_batch: int = 0
    training_rounds: int = 1
    attention_width: int = 0  # 0: the single FC layer in Figure 6
    attention_kind: str = "scalar"
    attention_dk: int = 256
    attention_dropout: float = 0.1
    attention_batch: int = 0  # 0: full observed population
    attention_full_epoch: bool = False
    attention_search_steps: int = 0  # 0: use search_steps
    attention_refresh: bool = False
    attention_gamma: str = 'sample'
    search_steps: int = 3
    max_backtracks: int = 35
    buffer_capacity: int = 8
    buffer_score_mode: str = "anchor"
    seed: int = 1
    euclidean: bool = False
    use_attention: bool = True
    use_buffer: bool = True
    normalize_targets: bool = True
    critic_lr: float = 0.003
    attention_lr: float = 0.003
    mu: float = 1.0
    metric_floor: float = 1e-4
    step_size: float = 1.0
    rho: float = 0.5
    armijo: float = 1e-4
    gamma_floor: float = 0.05
    direction_limit: float = 1.0  # 0 disables this optional trust-radius bound
    exploration_probability: float = 0.1
    exploration_rate: float = 2.0
    push_probability: float = 0.2
    buffer_temperature: float = 1.0
    kappa_error: float = 1.0
    kappa_improvement: float = 1.0
    kappa_distance: float = 1.0
    distance_mu: float = 1.0
    duplicate_tolerance: float = 1e-8
    goal: float | None = None
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self):
        positive_ints = ("initial_designs", "max_evaluations", "critic_epochs", "critic_width",
                         "search_steps", "max_backtracks", "buffer_capacity", "training_rounds", "attention_dk")
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("attention_epochs", "attention_width", "attention_batch", "seed",
                     "critic_batch", "attention_search_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 2 <= self.initial_designs <= self.max_evaluations:
            raise ValueError("budget must cover at least two initial designs")
        for name in ("euclidean", "use_attention", "use_buffer", "normalize_targets",
                     "standardize_inputs", "attention_full_epoch", "attention_refresh"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be Boolean")
        for name in ("critic_lr", "attention_lr", "mu", "metric_floor", "step_size",
                     "gamma_floor", "exploration_rate", "buffer_temperature", "kappa_error",
                     "kappa_improvement", "kappa_distance", "distance_mu", "duplicate_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("rho", "armijo"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("exploration_probability", "push_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.direction_limit) or self.direction_limit < 0:
            raise ValueError("direction_limit must be nonnegative and finite")
        if self.duplicate_tolerance >= 1:
            raise ValueError("duplicate_tolerance must be smaller than the unit domain")
        if self.goal is not None and not math.isfinite(self.goal):
            raise ValueError("goal must be finite or None")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.critic_hidden is not None:
            if not self.critic_hidden or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                                             for v in self.critic_hidden):
                raise ValueError("critic_hidden must contain positive integer widths")
            object.__setattr__(self, 'critic_hidden', tuple(self.critic_hidden))
        for name in ('critic_dropout', 'attention_dropout'):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f'{name} must be in [0, 1)')
        for name, values in [('critic_activation', ('tanh', 'gelu')), ('network_init', ('default', 'kaiming_relu')),
                             ('attention_kind', ('scalar', 'projected')),
                             ('attention_gamma', ('sample', 'best')),
                             ('buffer_score_mode', ('anchor', 'proposal'))]:
            if getattr(self, name) not in values:
                raise ValueError(f'{name} must be one of {values}')

    def to_dict(self):
        return asdict(self)
