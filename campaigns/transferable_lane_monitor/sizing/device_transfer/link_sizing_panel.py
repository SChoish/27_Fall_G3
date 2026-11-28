"""20-lane matched-port link panel driven by measured transmitter timing."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]/"experiments/package20_monitor"))
from benchmark_50 import Channel, characterize_all
from pilot import AR, FAMILIES, GRID, LANES, N, Q, R, SAMPLES, T, TARGET, curves, drift_scale, trajectory


RATE = 32
PERIOD = 4
REGIMES = ("mild","stationary","bursts")
LINK_ALLOWANCE = .01


def probe_estimate(readings, q):
    """The fixed MEE estimator is linear in its paid observations."""
    cov = np.zeros((N,N))
    mean = np.zeros((len(readings),N))
    result = np.empty_like(readings)
    probes = 0
    for t in range(T):
        mean *= AR
        cov = AR**2*cov+q
        if t%PERIOD==PERIOD//2:
            var = cov.diagonal()
            lane = int((var/np.sqrt(Q.diagonal())).argmax())
            column = cov[:,lane].copy()
            denom = var[lane]+R
            mean += ((readings[:,t,lane]-mean[:,lane])/denom)[:,None]*column
            cov -= np.outer(column,column)/denom
            probes += 1
        result[:,t] = mean
    assert probes == 32
    return result


def supply_fraction(layout, seed, mode):
    # Fraction of the measured nominal-to-low-supply interval, not extrapolation.
    if mode=="mild":
        return np.zeros((T,N))
    rng = np.random.default_rng([layout,seed,5521])
    group = int(rng.integers(4))
    droop = np.zeros((T,4))
    start = int(rng.integers(8,17))
    while start<T:
        duration = int(rng.integers(24,41) if mode=="stationary" else rng.integers(8,17))
        droop[start:start+duration,group] = rng.uniform(.35,.9)
        start += duration
        group = (group+int(rng.integers(1,4)))%4
    past = np.zeros(4)
    filtered = []
    for row in droop:
        past = .55*past+.45*row
        filtered.append(past.copy())
    return np.array(filtered)[:,LANES//5]


@dataclass
class Panel:
    nominal: np.ndarray
    center: np.ndarray
    bank: np.ndarray
    groups: list

    def failure(self, delay_swing_ps):
        swing = delay_swing_ps*RATE/1000
        results = []
        for truth, supply, packet, prediction, supply_prediction in self.groups:
            phase = self.nominal + np.rint(SAMPLES*(prediction+swing*supply_prediction+self.center))
            position = phase-SAMPLES*(truth+swing*supply)-GRID[0]
            offgrid = (position<0)|(position>=len(GRID)-1)
            position = np.clip(position,0,len(GRID)-1)
            i = position.astype(int)
            a = self.bank[packet[:,:,None],LANES,i]
            b = self.bank[packet[:,:,None],LANES,np.minimum(i+1,len(GRID)-1)]
            worst = np.where(offgrid,-1.,a+(position-i)*(b-a)).min(axis=2)
            results.append(float((worst<TARGET).mean()))
        return results


def panels(h, layout):
    assert h.shape==(20,20,61)
    channel = Channel(h,RATE,True)
    nominal,left,right,_ = characterize_all(channel)

    def build(packet_seed, trajectory_seed, packets, episodes):
        bank = np.array([curves(channel,packet_seed+i,f) for i in range(packets) for f in FAMILIES])
        groups = []
        for j,regime in enumerate(REGIMES):
            seeds = trajectory_seed+100*j+np.arange(episodes)
            truth,obs,packet = (np.stack(r) for r in zip(*(trajectory(layout,int(seed),regime,len(bank))
                                                           for seed in seeds)))
            supply = np.stack([supply_fraction(layout,int(seed),regime) for seed in seeds])
            q = drift_scale(regime)**2*Q
            groups.append((truth,supply,packet,probe_estimate(obs,q),probe_estimate(supply,q)))
        return Panel(nominal,.5*(right-left),bank,groups)

    return build(672100,672500,2,2), build(982100,982500,4,8)


def delay_swing(pair):
    return float(np.mean(pair[1]["ports"][0][:2])-np.mean(pair[0]["ports"][0][:2]))


def link_residual(failure, reference):
    delta = np.asarray(failure)-np.asarray(reference)
    return float(np.max(delta-LINK_ALLOWANCE))
