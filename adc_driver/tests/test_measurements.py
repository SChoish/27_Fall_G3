"""Independent signal/energy checks; run with Python's standard unittest."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from circuit import Context,SIGNALS,schedule,measure,integral


def synthetic(offset=0.,track_during_hold=False):
    c=Context();levels,starts,ends,stop=schedule(c)
    t=np.unique(np.r_[np.linspace(0,stop,20001),starts,ends,stop])
    vin=np.full(len(t),c.initial_v);stored=np.full(len(t),c.initial_v)
    gate=np.zeros(len(t))
    for value,start,end in zip(levels,starts,ends):
        vin[t>=start-c.input_lead_ns*1e-9]=value
        stored[t>=start]=value
        gate[(t>=start)&(t<end)]=c.vdd_v
    signals=dict(vin=vin,vout=vin,vfilter=vin,vsample=(vin if track_during_hold else stored)+offset,
                 clkn=gate,clkp=c.vdd_v-gate,clkcmd=gate,
                 i_avdd=np.full(len(t),-100e-6/c.vdd_v),
                 i_clkdd=np.full(len(t),-2e-6/c.vdd_v),
                 i_vin=np.zeros(len(t)),i_clkcmd=np.zeros(len(t)))
    return c,np.column_stack([t]+[signals[n] for n in SIGNALS])


class Measurements(unittest.TestCase):
    def test_true_sample_and_hold_is_accurate(self):
        c,a=synthetic();r=measure(a,c)
        self.assertTrue(r['feasible'])
        self.assertLess(r['worst_hold_error_mv'],1e-9)

    def test_hold_target_does_not_follow_next_input(self):
        c,a=synthetic(track_during_hold=True);r=measure(a,c)
        self.assertFalse(r['feasible'])
        self.assertGreater(r['worst_hold_error_mv'],200.)

    def test_static_offset_is_not_subtracted(self):
        c,a=synthetic(offset=.002);r=measure(a,c)
        self.assertFalse(r['feasible'])
        self.assertAlmostEqual(r['worst_hold_error_mv'],2.,places=8)

    def test_supply_energy_and_unit_conversion(self):
        c,a=synthetic();r=measure(a,c)
        self.assertAlmostEqual(r['mean_driver_energy_pj'],102.,places=8)
        self.assertAlmostEqual(r['mean_driver_power_uw'],102.,places=8)
        self.assertAlmostEqual(r['mean_clock_energy_pj'],2.,places=8)

    def test_integration_inserts_exact_window_boundaries(self):
        t=np.array([0.,.1,.4,.9,1.]);y=2*t+3
        self.assertAlmostEqual(integral(t,y,.15,.75),(.75**2+3*.75)-(.15**2+3*.15),places=12)

    def test_clock_fault_is_rejected(self):
        c,a=synthetic();a[:,SIGNALS.index('clkn')+1]=c.vdd_v
        self.assertFalse(measure(a,c)['valid'])


if __name__=='__main__':unittest.main()
