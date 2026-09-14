"""Constraint ordering and gradients used by attention and candidate selection."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from learning import surrogate_cost


@unittest.skipIf(torch is None, 'Optional learning dependencies required')
class SearchObjectiveTests(unittest.TestCase):
    def cost(self, targets, logits=None):
        y = torch.tensor(np.log10(targets), dtype=torch.float64, requires_grad=True)
        logits = torch.tensor([5.] * len(targets) if logits is None else logits, dtype=torch.float64)
        return y, surrogate_cost(y, logits, 0.)

    def test_feasible_energy_beats_unneeded_accuracy_and_confidence(self):
        _, score = self.cost([[1., 169.2982], [.1, 300.]], [.01, 20.])
        self.assertLess(score[0], score[1])
        self.assertAlmostEqual(float(score[0].detach()), 169.2982 / 1169.2982)

    def test_feasibility_and_validity_have_priority_over_energy(self):
        _, score = self.cost([[1., 1e5], [1.00001, .001], [.1, .001]], [2., 2., -.01])
        self.assertLess(score[0], score[1])
        self.assertLess(score[1], score[2])

    def test_feasible_gradient_spends_no_effort_on_excess_accuracy(self):
        y, score = self.cost([[.5, 200.]])
        grad = torch.autograd.grad(score.sum(), y)[0]
        self.assertEqual(grad[0, 0], 0.)
        self.assertGreater(grad[0, 1], 0.)

    def test_infeasible_gradient_reduces_error(self):
        y, score = self.cost([[2., 200.]])
        grad = torch.autograd.grad(score.sum(), y)[0]
        self.assertGreater(grad[0, 0], 0.)
        self.assertEqual(grad[0, 1], 0.)


if __name__ == '__main__':
    unittest.main()
