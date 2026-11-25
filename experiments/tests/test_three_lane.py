import unittest
from dataclasses import replace
import numpy as np
from experiments.d2d_three_lane import d, deck, make_case


class ThreeLaneTests(unittest.TestCase):
    def test_topology(self):
        case = make_case(2.15,.085,101,64,True)
        lines = deck(d.NOMINAL,case).splitlines()
        self.assertEqual(sum(s.startswith('M') for s in lines),108)
        for index,node in enumerate(('ch0','ch1','rx')):
            self.assertTrue(any(s.startswith(f'CCOUP{index} {node}_0 {node}_1 ') for s in lines))
            self.assertTrue(any(s.startswith(f'CRIGHT{index} {node}_0 {node}_2 ') for s in lines))
        self.assertEqual(sum(s.startswith('RTERM_') for s in lines),3)
        self.assertTrue(any('i(vdd_2)' in s for s in lines))

    def test_patterns_and_prefixes(self):
        case = make_case(2.15,.085,101,64,False)
        for third in (True,False):
            pair = replace(case,third=third)
            np.testing.assert_array_equal(pair.payload(),replace(pair,bits=128).payload()[:,:64])
            a,b = replace(pair,correlated=True).payload()
            np.testing.assert_array_equal(1-a,b)
        self.assertFalse(np.array_equal(case.payload()[1],replace(case,third=True).payload()[1]))
