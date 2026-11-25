"""Three 36-MOS transmitters, centre lane 0 with two separately coupled neighbours."""
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from d2d_sizing import circuit as d
from coupled_link import lane_body


@dataclass(frozen=True)
class Pair(d.Case):
    third: bool = False
    correlated: bool = False

    def payload(self):
        # Per-lane streams give 64-bit and 128-bit tests identical prefixes.
        a, b, c = [np.random.default_rng(np.random.SeedSequence([self.seed, i]))
                   .integers(0, 2, self.bits, dtype=np.uint8) for i in range(3)]
        if self.correlated:
            b = c = 1 - a
        return np.stack([a, c if self.third else b])


def deck(x, case):
    bits = case.payload()
    lines = d.link_netlist(case.context, (8, 8), bits, case.points_per_ui).splitlines()
    idx = lines.index('.control')
    right = replace(case, third=True).payload()[1]
    additions = lane_body(case.context.lane1, 8, right, case.points_per_ui, 2)
    for i, node in enumerate(('ch0', 'ch1', 'rx')):
        additions.append(f'CRIGHT{i} {node}_0 {node}_2 {case.context.coupling_pf/3:.12g}p')
    lines[idx:idx] = additions
    idx = next(i for i, line in enumerate(lines) if line.startswith('wrdata'))
    lines[idx] += ' v(tx_2) v(rx_2) i(vdd_2)'
    return d.sized_netlist(x, '\n'.join(lines), 3)


def transient(x, case):
    binary, env = d.link.simulator()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path/'link.cir').write_text(deck(x, case))
        process = subprocess.run([binary, '-b', 'link.cir'], cwd=path, env=env,
                                 capture_output=True, text=True, timeout=120)
        if process.returncode or not (path/'wave.txt').exists():
            raise RuntimeError((process.stdout + process.stderr)[-1500:])
        raw = np.loadtxt(path/'wave.txt', skiprows=1, ndmin=2)
    if raw.shape[1] != 10:
        raise ValueError('Expected time and three signals per lane')
    raw = raw[np.r_[np.diff(raw[:, 0]) > 0, True]]
    return raw


def measure_transient(raw, case):
    pairs = []
    for third, offset in [(False, 3), (True, 6)]:
        wave = dict(t=raw[:, 0], rx=raw[:, [2, 2+offset]], ivdd=raw[:, [3, 3+offset]])
        pairs.append(d.measure(wave, replace(case, third=third)))
    a, b = pairs
    energy = a['energy_lane_pj_bit'] + b['energy_lane_pj_bit'][1:]
    return dict(valid=all(p['valid'] for p in pairs), feasible=all(p['feasible'] for p in pairs),
                margins=a['margins'] + b['margins'][1:], errors=a['errors'] + b['errors'][1:],
                overshoot_fraction=max(p['overshoot_fraction'] for p in pairs),
                energy_lane_pj_bit=energy, energy_pj_bit=float(np.mean(energy)))


def simulate(x, case):
    return measure_transient(transient(x, case), case)


def make_case(rate, coupling, seed, bits, correlated, skew=0.):
    center = d.link.Context(rate_gbps=rate, inductance_nh=0, clock_offset_ui=skew)
    outer = replace(center, resistance_ohm=32.)
    return Pair('three_lane', context=d.CoupledContext(center, outer, coupling),
                bits=bits, seed=seed, correlated=correlated)
