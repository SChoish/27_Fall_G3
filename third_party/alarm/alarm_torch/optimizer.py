"""Sequential ask/tell ALARM with optional import of previously measured samples."""
from typing import Callable
from dataclasses import replace
import numpy as np
import torch
from .buffer import CriticBuffer
from .config import Config
from .networks import make_networks
from .problem import Domain, Observation, Scalarizer
from .search import line_search


class ALARM:
    def __init__(self, lower, upper, *, weights=None, clips=None, scalarizer=None, log_scale=None,
                 config: Config | None = None,
                 success: Callable[[Observation], bool] | None = None, **settings):
        if config is not None and settings:
            raise ValueError("pass either config=Config(...) or keyword settings")
        self.config = config or Config(**settings)
        self.domain = Domain(lower, upper, log_scale)
        if scalarizer is not None:
            if weights is not None or clips is not None or not callable(scalarizer):
                raise ValueError('pass a callable scalarizer or weights/clips, not both')
            self.scalarizer = scalarizer
        else:
            if weights is None or clips is None:
                raise ValueError('weights/clips or a scalarizer are required')
            self.scalarizer = Scalarizer(weights, clips)
        self.success_condition = success
        if success is not None and not callable(success):
            raise TypeError("success must be callable or None")
        self.device = torch.device(self.config.device)
        self.dtype = getattr(torch, self.config.dtype)
        if self.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if self.device.type == "mps" and (self.dtype == torch.float64 or not torch.backends.mps.is_available()):
            raise ValueError("MPS requires an available MPS device and dtype='float32'")
        # Training/exploration never advance the initial-design RNG.
        self._initial_rng = np.random.default_rng(self.config.seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, 1]))
        self._history = []
        self._pending = None
        self._pending_source = None
        self._pending_anchor = None
        self._pending_prediction = None
        self._best = self._best_feasible = self._solution = None
        self.critic = self.attention = None
        self.buffer = CriticBuffer(self.config, self.domain._hi-self.domain._lo)
        self.last_search = None
        self._stats = dict(critic_fits=0, critic_optimizer_steps=0, attention_updates=0, buffer_pushes=0, head_transfers=0,
                           duplicate_fallbacks=0, critic_loss=None, attention_loss=None,
                           attention_gradient_norm=None)

    @property
    def history(self):
        return tuple(self._history)

    @property
    def best(self):
        return self._best

    @property
    def best_feasible(self):
        return self._best_feasible

    @property
    def solution(self):
        """First measured feasible observation that satisfies the requested stop condition."""
        return self._solution

    @property
    def finished(self):
        return (len(self._history) >= self.config.max_evaluations or
                (len(self._history) >= self.config.initial_designs and self._solution is not None))

    @property
    def settings(self):
        return self.config.to_dict()

    @property
    def stats(self):
        return dict(self._stats, attempts=len(self._history),
                    imported_evaluations=sum(o.source == 'imported' for o in self._history),
                    new_evaluations=sum(o.source != 'imported' for o in self._history),
                    valid_evaluations=sum(o.valid for o in self._history),
                    failed_evaluations=sum(not o.valid for o in self._history),
                    buffer_size=len(self.buffer), success=self._solution is not None)

    def _tensor(self, values):
        result = torch.as_tensor(np.asarray(values), dtype=self.dtype, device=self.device)
        if not torch.isfinite(result).all():
            raise FloatingPointError("input exceeds the chosen torch dtype; rescale or use float64")
        return result

    @torch.enable_grad()
    def _fit(self):
        seed = int(self._rng.integers(0, 2**63 - 1))
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            self._fit_with_seed(seed)

    def _batch_indices(self, count, size):
        if size == 0 or size >= count:
            yield torch.arange(count, device=self.device)
            return
        order = self._rng.permutation(count)
        for offset in range(0, count, size):
            yield torch.as_tensor(order[offset:offset+size], dtype=torch.long, device=self.device)

    def _fit_with_seed(self, seed):
        c = self.config
        valid = [o for o in self._history if o.valid]
        x = self._tensor([o.normalized for o in valid])
        values = self._tensor([o.objective for o in valid])
        self.critic, self.attention = make_networks(self.domain.dimension, c, seed)
        self.critic.set_target_stats(values, c.normalize_targets)
        previous = self.buffer.select(self._best.normalized, self._best.objective, self._rng)
        if previous is not None:
            self.critic.transfer_head(previous.critic)
            self._stats['head_transfers'] += 1
        target = (values-self.critic.target_mean) / self.critic.target_scale
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr)
        attention_optimizer = torch.optim.Adam(self.attention.parameters(), lr=c.attention_lr)
        search_config = replace(c, search_steps=c.attention_search_steps or c.search_steps)
        self._stats['attention_loss'] = None
        self._stats['attention_gradient_norm'] = None
        for _ in range(c.training_rounds):
            self.critic.train().requires_grad_(True)
            for _ in range(c.critic_epochs):
                for indices in self._batch_indices(len(valid), c.critic_batch):
                    critic_optimizer.zero_grad(set_to_none=True)
                    loss = (self.critic.standardized(x[indices])-target[indices]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite critic loss; rescale objective or reduce critic_lr')
                    loss.backward()
                    critic_optimizer.step()
                    self._stats['critic_optimizer_steps'] += 1
            self.critic.eval().requires_grad_(False)
            for parameter in self.critic.parameters():
                parameter.grad = None
            if c.use_attention and not c.euclidean and c.attention_epochs:
                self.attention.train()
                for _ in range(c.attention_epochs):
                    if c.attention_full_epoch:
                        batches = self._batch_indices(len(valid), c.attention_batch)
                    else:
                        size = min(c.attention_batch or len(valid), len(valid))
                        indices = self._rng.choice(len(valid), size=size, replace=False)
                        batches = [torch.as_tensor(indices, dtype=torch.long, device=self.device)]
                    for indices in batches:
                        gamma = self._best.objective if c.attention_gamma == 'best' else values[indices]
                        search = line_search(self.critic, self.attention, x[indices], gamma,
                                             search_config, differentiable=True)
                        attention_optimizer.zero_grad(set_to_none=True)
                        loss = ((search.final_value-self.critic.target_mean) / self.critic.target_scale).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite attention loss')
                        loss.backward()
                        params = list(self.attention.parameters())
                        grad_sq = sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None)
                        if not np.isfinite(grad_sq):
                            raise FloatingPointError('nonfinite attention gradient')
                        if any(p.grad is not None for p in params):
                            torch.nn.utils.clip_grad_norm_(params, 10.0)
                            attention_optimizer.step()
                            self._stats['attention_updates'] += 1
                        self._stats['attention_loss'] = float(search.final_value.detach().mean())
                        self._stats['attention_gradient_norm'] = float(np.sqrt(grad_sq))
        with torch.no_grad():
            mse = (self.critic(x)-values).square().mean()
            if not torch.isfinite(mse):
                raise FloatingPointError('critic fit diverged')
            self._stats['critic_loss'] = float(mse)
        self._stats['critic_fits'] += 1
        self.attention.eval().requires_grad_(False)

    def _evaluate_metrics(self, metrics):
        array = np.array(metrics, dtype=np.float64, copy=True)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError('metrics must be a finite vector')
        objective, feasible = self.scalarizer(array)
        if not np.isfinite(objective):
            raise ValueError('scalar objective must be finite')
        return array, float(objective), bool(feasible)

    def _meets_goal(self, observation):
        if not observation.feasible:
            return False
        if self.success_condition is not None:
            return bool(self.success_condition(observation))
        return self.config.goal is not None and observation.objective <= self.config.goal

    def seed_population(self, designs, metrics):
        """Import exactly initial_designs measured samples into an empty optimizer.

        Counts toward the total budget. No simulator calls or fabricated targets;
        validate the entire population before committing any observations.
        """
        if self._history or self._pending is not None:
            raise RuntimeError('seed_population requires an empty optimizer without a pending request')
        designs = np.asarray(designs, dtype=np.float64)
        metrics = np.asarray(metrics, dtype=np.float64)
        n = self.config.initial_designs
        if designs.shape != (n, self.domain.dimension) or metrics.ndim != 2 or len(metrics) != n:
            raise ValueError('population shapes must match initial_designs and the design dimension')
        normalized = self.domain.normalize(designs)
        observations = []
        solution = None
        for x, z, values in zip(designs, normalized, metrics):
            array, value, feasible = self._evaluate_metrics(values)
            record = Observation(tuple(x), tuple(z), tuple(array), value, feasible, True, 'imported')
            if self._meets_goal(record) and solution is None:
                solution = record
            observations.append(record)
        self._history = observations
        self._best = min(observations, key=lambda o: o.objective)
        self._best_feasible = min((o for o in observations if o.feasible), key=lambda o: o.objective, default=None)
        self._solution = solution
        return self.history

    def _duplicate(self, normalized):
        if not self._history:
            return False
        # Include failures: a failed simulation is still an attempted design.
        points = np.array([o.normalized for o in self._history])
        distance = np.linalg.norm(points-normalized, axis=1)
        return bool(np.any(distance <= self.config.duplicate_tolerance))

    def _unique(self, candidate, *, local):
        if not self._duplicate(candidate):
            return candidate, False
        for attempt in range(256):
            if local and attempt < 64:
                proposal = np.clip(candidate + self._rng.normal(0, 0.02, candidate.shape), 0, 1)
            else:
                proposal = self._initial_rng.random(self.domain.dimension)
            if not self._duplicate(proposal):
                return proposal, True
        raise RuntimeError("cannot find an untried design; reduce duplicate_tolerance")

    def ask(self):
        """Return physical parameters; repeat calls return a copy of the pending design."""
        if self._pending is not None:
            return self.domain.physical(self._pending)
        if self.finished:
            return None
        self._pending_anchor = self._pending_prediction = None
        if len(self._history) < self.config.initial_designs or self._best is None:
            candidate = self._initial_rng.random(self.domain.dimension)
            candidate, duplicate = self._unique(candidate, local=False)
            source = "initial" if len(self._history) < self.config.initial_designs else "no_valid_data"
            if duplicate:
                source += "_duplicate_fallback"
        else:
            self._fit()
            anchor = self._best
            c = self.config
            factors = np.ones((c.search_steps, 1))
            explore = self._rng.random((c.search_steps, 1)) < c.exploration_probability
            factors[explore] = c.exploration_probability * c.exploration_rate
            self.last_search = line_search(self.critic, self.attention, self._tensor(anchor.normalized),
                                           anchor.objective, c, exploration_factors=factors)
            candidate = self.last_search.x.cpu().numpy().astype(np.float64)
            candidate, duplicate = self._unique(candidate, local=True)
            source = "duplicate_fallback" if duplicate else "line_search"
            if duplicate:
                self._stats['duplicate_fallbacks'] += 1
            self._pending_anchor = anchor
            with torch.no_grad():
                self._pending_prediction = float(self.critic(self._tensor(candidate)))
        self._pending = candidate.copy()
        self._pending_source = source
        return self.domain.physical(self._pending)

    def tell(self, metrics):
        """Commit one successful actual evaluation. Invalid metrics keep the request pending."""
        if self._pending is None:
            raise RuntimeError("call ask() before tell()")
        metrics, objective, feasible = self._evaluate_metrics(metrics)
        normalized = self._pending.copy()
        observation = Observation(tuple(self.domain.physical(normalized)), tuple(normalized),
                                  tuple(float(v) for v in metrics), objective, feasible, True,
                                  self._pending_source)
        meets = self._meets_goal(observation)
        if self._pending_anchor is not None:
            anchor = self._pending_anchor
            pushed = self.buffer.push(self.critic, anchor.normalized, anchor.objective, normalized,
                                      objective, self._pending_prediction, self._rng)
            self._stats['buffer_pushes'] += int(pushed)
        self._history.append(observation)
        if self._best is None or objective < self._best.objective:
            self._best = observation
        if feasible and (self._best_feasible is None or objective < self._best_feasible.objective):
            self._best_feasible = observation
        if meets and self._solution is None:
            self._solution = observation
        self._pending = self._pending_anchor = self._pending_prediction = self._pending_source = None
        return observation

    def reject(self, reason="simulation failed"):
        """Count one failed simulation; do not insert a fabricated critic target."""
        if self._pending is None:
            raise RuntimeError("call ask() before reject()")
        observation = Observation(tuple(self.domain.physical(self._pending)), tuple(self._pending),
                                  None, None, False, False, self._pending_source, str(reason))
        self._history.append(observation)
        self._pending = self._pending_anchor = self._pending_prediction = self._pending_source = None
        return observation
