"""FIFO critic snapshots and lower-is-better scoring, Eq. (28)-(31)."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BufferEntry:
    critic: torch.nn.Module
    anchor: np.ndarray
    anchor_objective: float
    gamma: float
    proposal: np.ndarray
    proposal_objective: float
    proposal_prediction: float


class CriticBuffer:
    def __init__(self, config, coordinate_scale=None):
        self.config = config
        self.coordinate_scale = 1.0 if coordinate_scale is None else np.array(coordinate_scale, copy=True)
        self.entries = deque(maxlen=config.buffer_capacity)

    def __len__(self):
        return len(self.entries)

    def push(self, critic, anchor, anchor_objective, proposal, proposal_objective,
             proposal_prediction, rng):
        if not self.config.use_buffer:
            return False
        if proposal_objective >= anchor_objective and rng.random() >= self.config.push_probability:
            return False
        snapshot = deepcopy(critic).eval().requires_grad_(False)
        self.entries.append(BufferEntry(snapshot, np.array(anchor, copy=True), anchor_objective,
                                        anchor_objective, np.array(proposal, copy=True),
                                        proposal_objective, proposal_prediction))
        return True

    @torch.no_grad()
    def scores(self, anchor, objective):
        result = []
        c = self.config
        for entry in self.entries:
            if c.buffer_score_mode == 'proposal':
                error = entry.proposal_prediction-entry.proposal_objective
                distance = np.square((np.asarray(anchor)-entry.proposal)*self.coordinate_scale).sum()
                result.append(c.kappa_error*error**2
                              + c.kappa_improvement*(entry.proposal_objective-entry.gamma)
                              + c.kappa_distance*distance)
                continue
            parameter = next(entry.critic.parameters())
            x = torch.as_tensor(anchor, dtype=parameter.dtype, device=parameter.device)
            # Ground truth already exists here. No speculative SPICE evaluations.
            error = float(entry.critic(x)) - objective
            distance = np.sqrt(np.square(np.asarray(anchor) - entry.anchor).sum()
                               + c.distance_mu * (objective-entry.anchor_objective)**2)
            result.append(c.kappa_error * error**2
                          + c.kappa_improvement * (entry.proposal_objective-entry.gamma)
                          + c.kappa_distance * distance)
        return np.array(result, dtype=np.float64)

    def probabilities(self, anchor, objective):
        scores = self.scores(anchor, objective)
        if not len(scores):
            return scores
        if not np.isfinite(scores).all():
            raise FloatingPointError("nonfinite critic-buffer scores; rescale simulator metrics")
        logits = -(scores - scores.min()) / self.config.buffer_temperature
        weights = np.exp(logits)
        return weights / weights.sum()

    def select(self, anchor, objective, rng):
        if not self.config.use_buffer or not self.entries:
            return None
        return self.entries[int(rng.choice(len(self.entries), p=self.probabilities(anchor, objective)))]
