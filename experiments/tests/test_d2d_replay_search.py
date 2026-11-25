import unittest

import numpy as np
from experiments.d2d_replay_search import long_correction, targets


class ReplayTests(unittest.TestCase):
    def test_constraints_keep_lane_identity_and_sign(self):
        x = targets(dict(energy_pj_bit=.2,margins=[.04,.16],overshoot_fraction=.1))
        np.testing.assert_allclose(x,[np.log(.2),.5,-1.,-.5])

    def test_local_correction_does_not_cancel_an_observed_failure(self):
        anchor = np.zeros(6)
        candidate = np.array([anchor, np.ones(6)])
        value = long_correction(candidate, [(anchor, [.4,-.2,.1])])
        np.testing.assert_allclose(value[0], [.4,0.,.1])
        self.assertLess(value[1,0],.01)
        np.testing.assert_array_equal(long_correction(candidate,[]),np.zeros((2,3)))
