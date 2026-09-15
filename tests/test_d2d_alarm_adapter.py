import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from d2d_sizing import circuit
from d2d_sizing.alarm_adapter import D2DObjective, metrics, make_optimizer

class AdapterTests(unittest.TestCase):
    def test_cost_and_feasibility_match(self):
        for energy in (.1,.3,2.):
            for margin in (-.1,.079,.08,.4):
                for overshoot in (0.,.2,.21):
                    for errors in (0,1):
                        feasible=margin>=.08 and overshoot<=.2 and errors==0
                        result=dict(valid=True,feasible=feasible,energy_pj_bit=energy,margins=[margin,.3],overshoot_fraction=overshoot,cases=[dict(errors=[errors,0])])
                        objective,flag=D2DObjective()(metrics(result))
                        self.assertEqual(flag,feasible)
                        self.assertEqual(objective,circuit.cost(result))
    def test_invalid_measurement_not_labeled(self):
        with self.assertRaises(ValueError):metrics(dict(valid=False))
        with self.assertRaises(ValueError):D2DObjective()([.2,.1,.2,0,np.nan])
    def test_log_coordinates_match(self):
        opt=make_optimizer()
        for u in (np.zeros(6),np.ones(6),np.linspace(0,1,6)):
            np.testing.assert_allclose(opt.domain.physical(u),circuit.decode(u),rtol=1e-12)
    def test_budget_and_no_oracle_early_stop(self):
        opt=make_optimizer(seed=17,initial_designs=4,additional=1)
        designs=[circuit.decode(np.full(6,i/3)) for i in range(4)]
        opt.seed_population(designs,[[.3,.1,.1,0,0]]*4)
        self.assertFalse(opt.finished)
        x=opt.ask();np.testing.assert_array_equal(opt.ask(),x)
        opt.tell([.2,.1,.1,0,0])
        self.assertTrue(opt.finished)
        self.assertEqual(opt.stats['attempts'],5)
        self.assertIsNone(opt.ask())

if __name__=='__main__':unittest.main()
