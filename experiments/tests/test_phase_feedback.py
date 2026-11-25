import unittest
import numpy as np

from experiments.analyze_phase_feedback import choose, GRID
from experiments.check_sizing_shortlist import shortlist
from experiments.confirm_phase_sizing import lane_quadratic
from experiments.run_phase_feedback import CODES, suite
from experiments.run_phase_sizing import oracle, bracket, init


class CalibrationTests(unittest.TestCase):
    def test_budget_and_quadratic(self):
        calls=[]
        def measure(step,idx):
            calls.append((step,idx))
            return np.array([.24-.001*(CODES[idx]-40)**2]*3)
        self.assertEqual(choose('fixed32',[],measure),(GRID[1],[]))
        self.assertEqual(calls,[])
        code,queries=choose('quadratic',[],measure)
        self.assertEqual(CODES[code],40)
        self.assertEqual(queries,[16,32,48])
        self.assertEqual(calls,list(enumerate(GRID)))

    def test_lane_curve_uses_only_allowed_observations(self):
        queries = []
        def measure(step, index):
            queries.append((step, index))
            phase = CODES[index]
            return np.array([.3 - .001*(phase-24)**2, .3 - .001*(phase-48)**2, .5])
        self.assertEqual(CODES[lane_quadratic(measure)], 36)
        self.assertEqual(queries, list(enumerate(GRID)))

    def test_reserve_replaces_only_one_slot(self):
        rows = [dict(x=[i], oracle=dict(residuals=[-slack], energy_pj_bit=i+1))
                for i, slack in enumerate([.04, .05, .3, .9])]
        self.assertEqual([row['x'] for row in shortlist(rows)], [[0], [2], [3]])
        rows[2]['oracle']['residuals'] = [-.1]
        rows[3]['oracle']['residuals'] = [-.2]
        self.assertEqual([row['x'] for row in shortlist(rows)], [[0], [1], [3]])

    def test_splits_and_geometry(self):
        a=suite(49101,1,256)
        b=suite(49102,1,256)
        self.assertEqual([c.bits for c in a],[64]*3+[256]*4)
        self.assertEqual(len({c.seed for c in a+b}),14)
        self.assertGreater(a[0].context.lane0.inductance_nh,0)
        self.assertEqual(suite(49101,0,256)[0].context.lane0.inductance_nh,0)

    def test_return_requires_joint_phase(self):
        packet=dict(margin=[[.2,0.],[0.,.2],[.2,.2]],overshoot=0.,energy=.2)
        measured=oracle([[packet]])
        self.assertFalse(measured['feasible'])
        self.assertEqual(min(measured['margins']),0.)

    def test_proposal_stays_in_range(self):
        rows=[]
        for k,x in enumerate(init(21)):
            rows.append(dict(x=x.tolist(),oracle=dict(residuals=[.5-k/8]*4,energy_pj_bit=.1+k/9)))
        p=bracket(rows,21)
        self.assertEqual(p.shape,(6,))
        self.assertTrue(np.isfinite(p).all())
