"""Numerical and integration checks for the learned metric (no new SPICE)."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from attention import CoordinateAttention, SearchOptions, metric_direction, metric_matrix, project
    from circuit import Context, NAMES, ROOT
    from learning import LocalSearch


@unittest.skipIf(torch is None, 'Install requirements-learning.txt for attention tests')
class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uniform_matches_legacy_metric_and_inactive_coordinates_are_safe(self):
        g = torch.tensor([[0., 1., -2.], [0., 0., 0.]], dtype=torch.float64)
        p = torch.ones_like(g) / 3
        config = SearchOptions()
        actual = metric_matrix(g, p, config)
        expected = g[:, :, None] * g[:, None, :] + torch.diag_embed(
            g.square() + .01 * g.square().mean(-1, keepdim=True) + 1e-4)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool((torch.linalg.eigvalsh(actual) > 0).all()))
        direction = metric_direction(g, p, config)
        self.assertLess(float((g[0] * direction[0]).sum()), 0)
        torch.testing.assert_close(direction[1], torch.zeros_like(g[1]))

    def test_self_attention_does_not_mix_independent_designs(self):
        model = CoordinateAttention(3).double()
        with torch.no_grad():
            model.logits.weight.copy_(torch.eye(3, dtype=torch.float64))
        u = torch.tensor([[.1, .4, .8], [.8, .3, .2]], dtype=torch.float64)
        torch.testing.assert_close(model(u), torch.stack([model(row) for row in u]))
        torch.testing.assert_close(model(u).sum(-1), torch.ones(2, dtype=torch.float64))
        self.assertFalse(torch.allclose(model(u)[0], model(u)[1]))

    def test_meta_gradient_through_solve_matches_finite_difference(self):
        model = CoordinateAttention(3).double()
        config = SearchOptions(step_size=.05)
        u = torch.tensor([[.4, .5, .6]], dtype=torch.float64)
        target = torch.tensor([[.15, .8, .3]], dtype=torch.float64)
        curvature = torch.tensor([1., 3., 5.], dtype=torch.float64)
        g = 2 * curvature * (u - target)
        def objective():
            v = project(u + metric_direction(g, model(u), config), u, config)
            return (curvature * (v - target).square()).sum()
        exact = torch.autograd.grad(objective(), model.logits.bias)[0][0].item()
        eps = 1e-5
        with torch.no_grad():
            model.logits.bias[0] = eps; plus = float(objective())
            model.logits.bias[0] = -eps; minus = float(objective())
            model.logits.bias[0] = 0
        self.assertGreater(abs(exact), 1e-7)
        self.assertAlmostEqual(exact, (plus - minus) / (2 * eps), places=8)

    def test_boundary_projection_and_extreme_weights(self):
        config = SearchOptions()
        u = torch.tensor([[0., 1., .5]], dtype=torch.float64)
        g = torch.tensor([[-3., 2., 0.]], dtype=torch.float64)
        p = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
        direction = metric_direction(g, p, config)
        self.assertTrue(bool(torch.isfinite(direction).all()))
        v = project(u + direction, u, config)
        self.assertTrue(bool(((v >= 0) & (v <= 1)).all()))
        self.assertLessEqual(float((v - u).abs().max()), config.trust_radius)

    def test_attention_trains_and_changes_direction_without_changing_critic(self):
        rows = json.loads((ROOT / 'results/demo/rnc/trace.json').read_text())['trace'][:8]
        off = LocalSearch(ROOT / 'checkpoints/rnc.pt', [Context()], SearchOptions(attention='off'))
        learned = LocalSearch(ROOT / 'checkpoints/rnc.pt', [Context()])
        before = copy.deepcopy(learned.model.state_dict())
        # The proposer must not simulate any uncharged designs.
        with patch('circuit.evaluate_suite', side_effect=AssertionError('Unexpected SPICE call')):
            a = off.propose(rows, np.random.default_rng(90101))
            b = learned.propose(rows, np.random.default_rng(90101))
        report = learned.last_report['attention']
        self.assertGreater(report['parameter_delta_l2'], 0)
        self.assertGreater(report['max_gradient_norm'], 0)
        self.assertLess(report['loss_after'], report['loss_before'])
        # Both searches may correctly select the same winning random-bank
        # candidate. Verify the learned movement, not an accidental argmin tie.
        u = torch.tensor(rows[2]['u'], dtype=torch.float32, requires_grad=True)
        gradient = torch.autograd.grad(learned.loss(u)[0], u)[0].detach()
        with torch.no_grad():
            uniform_step = metric_direction(gradient, off.probabilities(u), off.options)
            learned_step = metric_direction(gradient, learned.probabilities(u), learned.options)
        self.assertFalse(torch.allclose(uniform_step, learned_step))
        self.assertEqual(learned.last_report['line_search_trials'], 0)
        self.assertGreater(learned.last_report['accepted_metric_steps'], 0)
        for name, value in learned.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in learned.model.parameters()))
        for v in (a, b):
            self.assertTrue(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all())
            self.assertGreater(np.linalg.norm(np.array([r['u'] for r in rows]) - v, axis=1).min(), 1e-6)

    def test_context_mismatch_is_rejected_before_search(self):
        with self.assertRaisesRegex(ValueError, 'context differs'):
            LocalSearch(ROOT / 'checkpoints/rnc.pt', [Context(temperature_c=85)])


if __name__ == '__main__':
    unittest.main()
