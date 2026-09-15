"""Simulator-space mapping and Eq. (21) scalarization (all objectives minimize)."""
from dataclasses import dataclass
import numpy as np


class Domain:
    def __init__(self, lower, upper, log_scale=None):
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        if (self.lower.ndim != 1 or self.lower.size == 0 or self.upper.shape != self.lower.shape
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or not (self.lower < self.upper).all()):
            raise ValueError("lower/upper must be equally sized finite vectors with lower < upper")
        flags = np.zeros(self.lower.size, dtype=bool) if log_scale is None else np.asarray(log_scale)
        if flags.shape != self.lower.shape or not np.isin(flags, [0, 1]).all():
            raise ValueError("log_scale must have one Boolean flag per coordinate")
        self.log_scale = flags.astype(bool, copy=True)
        if np.any(self.lower[self.log_scale] <= 0):
            raise ValueError("logarithmic bounds must be positive")
        self._lo = self.lower.copy()
        self._hi = self.upper.copy()
        self._lo[self.log_scale] = np.log(self._lo[self.log_scale])
        self._hi[self.log_scale] = np.log(self._hi[self.log_scale])
        self.dimension = len(self.lower)
        if not np.isfinite(self._hi - self._lo).all():
            raise ValueError("domain range overflows float64")

    def physical(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != self.dimension or not np.isfinite(z).all():
            raise ValueError("invalid normalized design")
        if np.any((z < 0) | (z > 1)):
            raise ValueError("normalized coordinates must be in [0, 1]")
        x = self._lo + z * (self._hi - self._lo)
        x[..., self.log_scale] = np.exp(x[..., self.log_scale])
        return np.clip(x, self.lower, self.upper)  # clamp exp/log endpoint roundoff

    def normalize(self, x):
        x = np.array(x, dtype=np.float64, copy=True)
        if x.ndim < 1 or x.shape[-1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("invalid physical design")
        if np.any((x < self.lower) | (x > self.upper)):
            raise ValueError("physical design is outside the bounds")
        x[..., self.log_scale] = np.log(x[..., self.log_scale])
        return np.clip((x - self._lo) / (self._hi - self._lo), 0, 1)


class Scalarizer:
    def __init__(self, weights, clips):
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.clips = np.array(clips, dtype=np.float64, copy=True)
        if (self.weights.ndim != 1 or self.clips.ndim != 1
                or len(self.weights) != len(self.clips) + 1
                or not np.isfinite(self.weights).all() or not np.isfinite(self.clips).all()
                or not (self.weights > 0).all() or not (self.clips > 0).all()):
            raise ValueError("positive finite weights=[w0,...,wm], clips=[c1,...,cm] required")

    def __call__(self, metrics):
        f = np.asarray(metrics, dtype=np.float64)
        if f.shape != self.weights.shape or not np.isfinite(f).all():
            raise ValueError("invalid metrics; use reject() for failed simulations")
        with np.errstate(over="ignore", invalid="ignore"):
            value = self.weights[0] * f[0] + np.minimum(
                self.clips, np.maximum(0, self.weights[1:] * f[1:])).sum()
        if not np.isfinite(value):
            raise ValueError("scalar objective overflows float64")
        return float(value), bool(np.all(f[1:] <= 0))


@dataclass(frozen=True)
class Observation:
    design: tuple[float, ...]
    normalized: tuple[float, ...]
    metrics: tuple[float, ...] | None
    objective: float | None
    feasible: bool
    valid: bool
    source: str
    error: str | None = None

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)
