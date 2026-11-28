"""Connect the ADC/D2D constrained search to the paired log-width benchmark."""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[4]))
from experiments.constrained_search import ConstrainedSearch


def response(values):
    # circuit.ratios emits this sentinel when a development transient fails.
    if np.all(values == 3.):
        return None, None
    # ratios returns log(EDP) and four signed log residuals. The optimizer
    # accepts linear objective values and log-transforms them internally.
    return float(np.exp(values[0])), values[1:]


def make_selector():
    optimizer = None

    def select(widths, values, rng, _decisions):
        nonlocal optimizer
        if optimizer is None:
            optimizer = ConstrainedSearch(widths.shape[1], values.shape[1]-1,
                                          int(rng.integers(2**32)))
            for point, value in zip(widths, values):
                optimizer.observe(point, *response(value))
        else:
            if len(widths) != len(optimizer.points)+1:
                raise ValueError("Expected one new development observation")
            optimizer.observe(optimizer.pending, *response(values[-1]), measured_point=widths[-1])
        return optimizer.ask(), {"radius": optimizer.radius}

    return select
