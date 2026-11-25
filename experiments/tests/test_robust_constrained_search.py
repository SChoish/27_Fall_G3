import unittest
import numpy as np
from scipy.special import ndtr

from experiments.constrained_search import ConstrainedSearch, MaxConstraintSearch
from experiments.robust_search_ablation import new_suites, shortlist, validated_selection


def row(x, energy, constraints, feasible=True):
    contexts = [dict(worst_hold_error_mv=float(np.exp(g)), context=dict(error_limit_mv=1.))
                for g in constraints]
    return dict(x=[x], oracle=dict(valid=True, feasible=feasible,
        mean_driver_energy_pj=energy, contexts=contexts))


class RobustSearchTests(unittest.TestCase):
    def test_max_target_preserves_exact_feasible_set(self):
        y = np.array([[1., -.2, -.5], [2., -.1, 0.], [3., -.1, .001]])
        targets = MaxConstraintSearch(1, 2, 0).model_targets(y)
        np.testing.assert_array_equal(targets[:, 1] <= 0, np.all(y[:, 1:] <= 0, axis=1))
        np.testing.assert_array_equal(targets[:, 0], y[:, 0])

    def test_scalar_gaussian_probability(self):
        search = MaxConstraintSearch(1, 2, 0)
        mean = np.array([[0., 0.], [0., 1.], [0., -1.]])
        actual = np.exp(search.feasibility_log_score(mean, np.ones_like(mean)))
        np.testing.assert_allclose(actual, [.5, ndtr(-1), ndtr(1)])

    def test_duplicate_constraints_do_not_change_proposals(self):
        a, b = MaxConstraintSearch(2, 2, 7), MaxConstraintSearch(2, 3, 7)
        for x in ([.1,.2], [.5,.5], [.8,.9]):
            g = [x[0]-.6, .1-x[1]]
            a.observe(x, 1+x[0], g)
            b.observe(x, 1+x[0], [*g, g[0]])
        np.testing.assert_array_equal(a.ask(), b.ask())

    def test_failed_simulations_count_but_are_not_targets(self):
        search = MaxConstraintSearch(2, 1, 5)
        x = search.ask()
        with self.assertRaises(RuntimeError):
            search.ask()
        with self.assertRaises(ValueError):
            search.observe(1-x, 1., [-1.])
        search.observe(x)
        self.assertEqual(len(search.points), 1)
        self.assertIsNone(search.results[0])
        self.assertTrue(np.all((search.ask() >= 0) & (search.pending <= 1)))

    def test_shortlist_includes_safe_backup_and_excludes_failures(self):
        rows = [row(.1, 1, [-.001]), row(.2, 2, [-.002]), row(.3, 3, [-.01]),
                row(.4, 10, [-1.]), row(.5, .1, [.1], False)]
        chosen = shortlist('adc', rows)
        self.assertEqual([r['x'] for r in chosen], [[.1], [.2], [.4]])
        validation = [dict(feasible=False), dict(feasible=False), dict(feasible=True)]
        selected = validated_selection('adc', chosen, validation)
        self.assertEqual(selected['design'], [.4])
        self.assertIsNone(validated_selection('adc', chosen, [dict(feasible=False)]*3))

    def test_selection_ranks_passing_designs_by_development_energy(self):
        candidates = [row(.1, 1, [-.1]), row(.2, 2, [-.1])]
        measured = [dict(feasible=True, mean_driver_energy_pj=100.),
                    dict(feasible=True, mean_driver_energy_pj=.01)]
        self.assertEqual(validated_selection('adc', candidates, measured)['design'], [.1])

    def test_fresh_test_inputs_and_fixed_validation_cost(self):
        for kind in ('adc', 'd2d'):
            dev, validation, test = new_suites(kind)
            self.assertEqual((len(dev), len(validation)), (4, 8))
            if kind == 'adc':
                seen = {c.levels_v for c in dev+validation}
                self.assertTrue(all(c.levels_v not in seen for c in test['fresh_inputs']))
            else:
                seen = {c.seed for c in dev+validation}
                self.assertTrue(all(c.seed not in seen and c.pattern == 'random'
                                    for cases in test.values() for c in cases))


if __name__ == '__main__':
    unittest.main()
