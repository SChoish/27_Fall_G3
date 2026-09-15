"""Adapt the upstream ALARM optimizer to the existing D2D sizing objective."""
from pathlib import Path
import sys
import numpy as np
from .circuit import LOW, HIGH, cost

VENDOR = Path(__file__).resolve().parents[1] / 'third_party' / 'alarm'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from alarm_torch import ALARM, Config

UPSTREAM_COMMIT = 'e407e8f1c616dc3a645b0793cf87ecb19a97433e'


def metrics(result):
    """Energy, two worst margins, worst overshoot, total observed bit errors."""
    if not result.get('valid', False):
        raise ValueError('Invalid simulation: use optimizer.reject instead')
    errors = sum(sum(case['errors']) for case in result['cases'])
    return np.array([result['energy_pj_bit'], *result['margins'],
                     result['overshoot_fraction'], errors], dtype=float)


class D2DObjective:
    """Preserve circuit.cost and its quality constraints without extra penalties."""
    def __init__(self, margin=.08, overshoot=.2):
        self.margin, self.overshoot = margin, overshoot

    def __call__(self, values):
        v = np.asarray(values, float)
        if v.shape != (5,) or not np.isfinite(v).all() or v[0] <= 0 or v[3] < 0 or v[4] < 0 or v[4] != int(v[4]):
            raise ValueError('Expected finite physical D2D metrics')
        feasible = bool(min(v[1:3]) >= self.margin and v[3] <= self.overshoot and v[4] == 0)
        result = dict(valid=True, feasible=feasible, energy_pj_bit=float(v[0]),
                      margins=v[1:3].tolist(), overshoot_fraction=float(v[3]))
        return cost(result, self.margin, self.overshoot), feasible


def make_optimizer(seed=0, initial_designs=8, additional=16, margin=.08, overshoot=.2):
    return ALARM(LOW, HIGH, log_scale=[True]*len(LOW),
                 scalarizer=D2DObjective(margin, overshoot),
                 config=Config(seed=seed, initial_designs=initial_designs,
                               max_evaluations=initial_designs+additional))
