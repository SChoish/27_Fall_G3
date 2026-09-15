"""Measured-response regression/RNC, residual fitting and critic reuse.

This is supervised circuit optimization, not a Bellman critic. Buffer scores
the same residual-corrected predictor used for proposals, before measurement.
"""
import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adc_driver.attention import CoordinateAttention, SearchOptions, metric_direction, project
from adc_driver.rnc import rank_n_contrast
from .circuit import NAMES, cost

torch.set_num_threads(1)


def target(result):
    # Signed margins must NOT be logged or clipped to a positive epsilon.
    return np.array([*result['margins'], np.log10(result['energy_pj_bit']),
                     result['overshoot_fraction']], dtype=np.float32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(len(NAMES), 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.regressor = nn.Linear(64, 4)
        self.validity = nn.Linear(64, 1)
        self.projector = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 16))

    def forward(self, u):
        h = self.encoder(u)
        return self.regressor(h), self.validity(h)[:, 0], h, self.projector(h)


def loss_components(model, rows, mean, std, method):
    u = torch.tensor(np.array([r['u'] for r in rows]), dtype=torch.float32)
    good = torch.tensor([r['oracle']['valid'] for r in rows])
    pred, valid, _, z = model(u)
    reg = pred.sum()*0
    contrast = z.sum()*0
    if good.any():
        y = torch.tensor(np.array([target(r['oracle']) for r in rows if r['oracle']['valid']]))
        yn = (y-mean)/std
        reg = F.mse_loss(pred[good], yn)
        if method == 'rnc':
            contrast = rank_n_contrast(z[good], yn)
    bce = F.binary_cross_entropy_with_logits(valid, good.float())
    return reg, bce, contrast


def train(rows, method, seed, epochs=100, checkpoint='validation'):
    if method not in ('regression', 'rnc') or epochs < 1:
        raise ValueError('Invalid training method or epochs')
    if checkpoint not in ('validation', 'last'):
        raise ValueError('Invalid checkpoint rule')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed+7001)
    training = [r for r in rows if r.get('split', 'train') == 'train']
    validation = [r for r in rows if r.get('split') == 'validation' and r['oracle']['valid']]
    y = np.array([target(r['oracle']) for r in training if r['oracle']['valid']])
    if len(y) < 4:
        raise ValueError('Need >=4 valid train designs')
    if checkpoint == 'validation' and not validation:
        raise ValueError('Need >=1 valid validation design')
    mean = torch.tensor(y.mean(0))
    std = torch.tensor(np.maximum(y.std(0, ddof=1), [.05, .05, .05, .02]), dtype=torch.float32)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, logs = math.inf, None, []
    scored = validation if checkpoint == 'validation' else [r for r in training if r['oracle']['valid']]
    for epoch in range(epochs):
        for ix in np.array_split(rng.permutation(len(training)), max(1, math.ceil(len(training)/32))):
            reg, bce, rnc = loss_components(model, [training[i] for i in ix], mean, std, method)
            loss = reg + .2*bce + .1*rnc
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite surrogate loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            score = float(loss_components(model, scored, mean, std, 'regression')[0])
        logs.append(dict(epoch=epoch+1, validation_mse=score if checkpoint == 'validation' else None,
                         train_mse=score if checkpoint == 'last' else None,
                         last_regression=float(reg.detach()), last_rnc=float(rnc.detach())))
        if checkpoint == 'last' or score < best:
            best, state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch+1
    return dict(state_dict=state, mean=mean, std=std, method=method, seed=seed,
                checkpoint=checkpoint, best_epoch=best_epoch, validation_mse=best if checkpoint == 'validation' else None,
                train_mse=best if checkpoint == 'last' else None, logs=logs)


class Predictor:
    def __init__(self, model, mean, std, rows):
        self.model = model.eval().requires_grad_(False)
        self.mean, self.std, self.coef = mean, std, None
        good = [r for r in rows if r['oracle']['valid']]
        if len(good) >= 2:
            with torch.no_grad():
                pred, _, h, _ = model(torch.tensor(np.array([r['u'] for r in good]), dtype=torch.float32))
                self.hm, self.hs = h.mean(0), h.std(0).clamp_min(.2)
                phi = torch.cat([(h-self.hm)/self.hs, torch.ones(len(h), 1)], 1)
                y = torch.tensor(np.array([target(r['oracle']) for r in good]))
                residual = (y-mean)/std-pred
                self.coef = phi.T@torch.linalg.solve(phi@phi.T + .5*torch.eye(len(phi)), residual)

    def predict(self, u):
        pred, valid, h, _ = self.model(u)
        if self.coef is not None:
            phi = torch.cat([(h-self.hm)/self.hs, torch.ones(len(h), 1)], 1)
            pred = pred+phi@self.coef
        return pred*self.std+self.mean, valid

    def objective(self, u, margin=.08, overshoot=.2):
        y, valid = self.predict(u)
        violation = ((margin-y[:, :2])/margin).clamp_min(0).sum(-1)
        violation = violation + ((y[:, 3]-overshoot)/overshoot).clamp_min(0)
        energy = torch.sigmoid(math.log(10)*y[:, 2])  # E / (1 + E)
        return torch.where(valid < 0, 3+torch.sigmoid(-valid),
                           torch.where(violation > 0, 1+violation/(1+violation), energy))


@dataclass
class Entry:
    id: int
    model: Model
    errors: list
    prior: float = 1.


class Search:
    def __init__(self, payload, offline, mode='buffer', attention='learned', online_epochs=20,
                 margin=.08, overshoot=.2):
        if mode not in ('frozen', 'latest', 'buffer') or online_epochs < 1:
            raise ValueError('Invalid critic update options')
        self.payload, self.offline, self.mode = payload, offline, mode
        self.online_epochs, self.margin, self.overshoot = online_epochs, margin, overshoot
        model = Model()
        model.load_state_dict(payload['state_dict'])
        self.entries = [Entry(0, model, [])]
        self.current, self.next_id = self.entries[0], 1
        self.mean, self.std = payload['mean'], payload['std']
        self.options = SearchOptions(attention=attention)
        self.attention = CoordinateAttention(len(NAMES)) if attention == 'learned' else None
        self.events = []

    @staticmethod
    def score(entry):
        if not entry.errors:
            return entry.prior
        errors = entry.errors[-32:]
        weights = np.exp(-np.arange(len(errors)-1, -1, -1)/32)
        return float(np.average(errors, weights=weights))

    def _update(self, rows):
        if self.mode == 'frozen':
            return dict(mode=self.mode, updated=False, selected=self.current.id)
        scores = np.array([self.score(e) for e in self.entries])
        rng = np.random.default_rng(np.random.SeedSequence([self.payload['seed'], len(rows), 87231]))
        probabilities = np.exp(-(scores-scores.min())/.25)
        probabilities /= probabilities.sum()
        parent = self.entries[int(rng.choice(len(self.entries), p=probabilities))] if self.mode == 'buffer' else self.current
        model = copy.deepcopy(parent.model).train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        count = max(len(self.offline), len(rows))
        for _ in range(self.online_epochs):
            offline = rng.choice(len(self.offline), count, replace=len(self.offline) < count)
            online = rng.choice(len(rows), count, replace=len(rows) < count)
            replay = [self.offline[i] for i in offline] + [rows[i] for i in online]
            for ix in np.array_split(rng.permutation(len(replay)), max(1, math.ceil(len(replay)/32))):
                reg, bce, rnc = loss_components(model, [replay[i] for i in ix], self.mean, self.std, self.payload['method'])
                loss = reg + .2*bce + .1*rnc
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite online surrogate loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
        child = Entry(self.next_id, model.eval(), [], self.score(parent))
        self.next_id += 1
        event = dict(mode=self.mode, updated=True, parent=parent.id, selected=child.id,
                     parent_scores=scores.tolist(), probabilities=probabilities.tolist())
        if self.mode == 'latest':
            self.entries = [child]
        else:
            if len(self.entries) >= 4:
                self.entries.pop(int(scores.argmax()))
            self.entries.append(child)
        self.current = child
        return event

    def predictions(self, u, rows):
        # Must be called before the oracle. Each candidate gets its own residual
        # fit using exactly the already-paid rows, matching the proposer.
        result = []
        point = torch.tensor(np.asarray(u)[None], dtype=torch.float32)
        for entry in self.entries:
            predictor = Predictor(entry.model, self.mean, self.std, rows)
            with torch.no_grad():
                y, valid = predictor.predict(point)
            result.append(dict(id=entry.id, targets=y[0].tolist(), valid_probability=float(valid.sigmoid()[0])))
        return result

    def observe(self, row, predictions):
        entries = {e.id: e for e in self.entries}
        for prediction in predictions:
            valid = row['oracle']['valid']
            error = .2*(prediction['valid_probability']-float(valid))**2
            if valid:
                error += float(np.mean(((np.asarray(prediction['targets'])-target(row['oracle']))/self.std.numpy())**2))
            entries[prediction['id']].errors.append(error)

    def propose(self, rows, rng):
        event = self._update(rows)
        predictor = Predictor(self.current.model, self.mean, self.std, rows)
        objective = lambda u: predictor.objective(u, self.margin, self.overshoot)
        attention_delta = 0.
        if self.attention is not None:
            good = [r for r in rows if r['oracle']['valid']]
            if good:
                arng = np.random.default_rng(np.random.SeedSequence([self.payload['seed'], len(rows), 41871]))
                selected = arng.choice(len(good), min(32, len(good)), replace=False)
                points = torch.tensor(np.array([good[i]['u'] for i in selected]), dtype=torch.float32, requires_grad=True)
                gradient = torch.autograd.grad(objective(points).sum(), points)[0].detach()
                points = points.detach()
                active = gradient.norm(dim=-1) > 1e-8
                points, gradient = points[active], gradient[active]
                if len(points):
                    optimizer = torch.optim.Adam(self.attention.parameters(), lr=self.options.attention_lr)
                    def attention_loss():
                        direction = metric_direction(gradient, self.attention(points), self.options)
                        return objective(project(points+direction, points, self.options)).mean()
                    with torch.no_grad():
                        before = best = float(attention_loss())
                    state = copy.deepcopy(self.attention.state_dict())
                    for _ in range(self.options.attention_steps):
                        loss = attention_loss()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.attention.parameters(), 5., error_if_nonfinite=True)
                        optimizer.step()
                        with torch.no_grad():
                            score = float(attention_loss())
                        if score < best:
                            best, state = score, copy.deepcopy(self.attention.state_dict())
                    self.attention.load_state_dict(state)
                    attention_delta = before-best
        elites = sorted(rows, key=lambda r: cost(r['oracle'], self.margin, self.overshoot))[:3]
        centers = [np.asarray(r['u']) for r in elites]
        starts = centers + [np.clip(centers[0]+rng.normal(0, .15, len(NAMES)), 0, 1) for _ in range(2)]
        candidates = []
        for start in starts:
            u = torch.tensor(start, dtype=torch.float32)
            center = u.clone()
            for _ in range(self.options.search_steps):
                u = u.detach().requires_grad_(True)
                gradient = torch.autograd.grad(objective(u[None]).sum(), u)[0]
                if not torch.isfinite(gradient).all() or gradient.norm() < 1e-8:
                    break
                with torch.no_grad():
                    p = self.attention(u) if self.attention is not None else torch.ones_like(u)/len(u)
                    u = project(u+metric_direction(gradient, p, self.options), center, self.options)
                    candidates.append(u.numpy())
        candidates.extend(np.clip(centers[0]+rng.normal(0, .15, (64, len(NAMES))), 0, 1))
        candidates.extend(rng.uniform(0, 1, (64, len(NAMES))))
        a = np.asarray(candidates)
        old = np.array([r['u'] for r in rows])
        a = a[np.min(np.linalg.norm(a[:, None]-old[None], axis=-1), axis=1) > 1e-6]
        if not len(a):
            raise RuntimeError('No new design candidate')
        with torch.no_grad():
            scores = objective(torch.tensor(a, dtype=torch.float32)).numpy()
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite candidate score')
        selected = int(scores.argmin())
        event.update(after_query=len(rows), predicted_cost=float(scores[selected]), candidates=len(a),
                     attention_objective_improvement=attention_delta,
                     selection_evidence='premeasurement residual-corrected prediction error')
        self.events.append(event)
        return a[selected], event
