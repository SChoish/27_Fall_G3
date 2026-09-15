"""Six sizing variables, 72 MOS, coupled RLC channels and a fixed RX aperture.

Uses the repository's PTM predictive MOS model, not a foundry PDK or a UCIe
compliance model. Every design is physically re-simulated; calibration lookup
tables are never used. Source excitation and behavioral RX power are excluded.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LINK_ROOT = REPO / 'd2d_coupled_calibration'
if str(LINK_ROOT) not in sys.path:
    sys.path.insert(0, str(LINK_ROOT))
from coupled_link import CoupledContext, link, netlist as link_netlist

NAMES = ('pre1_n_um', 'pre2_n_um', 'output_n_um', 'pn_ratio',
         'enable_width_ratio', 'series_resistance_ohm')
LOW = np.array([.4, 1.5, .25, 1.4, 1., 1.])
HIGH = np.array([2.5, 10., 4., 3., 4., 80.])
NOMINAL = np.array([1., 4., 2., 2., 2., 5.])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_design(x):
    x = np.asarray(x, float)
    if x.shape != LOW.shape or not np.isfinite(x).all() or (x < LOW).any() or (x > HIGH).any():
        raise ValueError('Expected six finite sizing variables within LOW/HIGH')
    return x


def encode(x):
    return np.log(validate_design(x) / LOW) / np.log(HIGH / LOW)


def decode(u):
    u = np.asarray(u, float)
    if u.shape != LOW.shape or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError('Expected six normalized variables in [0,1]')
    return np.clip(LOW * (HIGH / LOW) ** u, LOW, HIGH)


@dataclass(frozen=True)
class Case:
    name: str
    context: CoupledContext = field(default_factory=lambda: CoupledContext(link.Context(), link.Context(), .06))
    pattern: str = 'random'
    seed: int = 71001
    bits: int = 64
    # Fixed before optimization. No per-design phase maximization or alignment.
    phase_ui: float = .5
    guard_ui: float = .1
    margin_min: float = .08
    overshoot_max: float = .2
    points_per_ui: int = 64

    def __post_init__(self):
        if self.pattern not in ('random', 'opposite', 'same', 'runs'):
            raise ValueError('Unknown data pattern')
        if type(self.bits) is not int or self.bits < 8 or self.bits % 8:
            raise ValueError('bits must be a multiple of eight')
        if not 0 <= self.phase_ui <= 1 or not 0 < self.guard_ui < .5:
            raise ValueError('Invalid fixed sampling aperture')
        if not 0 < self.margin_min < .5 or not 0 < self.overshoot_max < 1:
            raise ValueError('Invalid quality limits')
        if type(self.points_per_ui) is not int or self.points_per_ui < 32:
            raise ValueError('Use at least 32 transient points per UI')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['context'] = CoupledContext.from_dict(value['context'])
        return cls(**value)

    def payload(self):
        if self.pattern == 'random':
            return np.random.default_rng(self.seed).integers(0, 2, (2, self.bits), dtype=np.uint8)
        if self.pattern == 'runs':
            a = np.tile(np.r_[np.zeros(4), np.ones(4)], self.bits // 8).astype(np.uint8)
        else:
            a = np.tile([0, 1], self.bits // 2).astype(np.uint8)
        return np.stack([a, a if self.pattern == 'same' else 1 - a])


def default_suite(split='development'):
    """Small predeclared input suites, not statistically exhaustive PVT corners."""
    if split not in ('development', 'test'):
        raise ValueError(split)
    development = split == 'development'
    settings = [(1., 27., 30., .30, .06), (.95, 65., 40., .34, .10)] if development else [
        (.98, 40., 35., .32, .08), (1.02, 55., 38., .33, .09)]
    result = []
    for j, (vdd, temp, resistance, cap, coupling) in enumerate(settings):
        lanes = [link.Context(vdd=vdd, temperature_c=temp, resistance_ohm=resistance + 2*k,
                              capacitance_pf=cap, inductance_nh=.3, rx_capacitance_pf=.06,
                              clock_offset_ui=0.) for k in range(2)]
        context = CoupledContext(*lanes, coupling_pf=coupling)
        patterns = ('random', 'opposite') if development else ('random', 'random', 'same', 'runs')
        for k, pattern in enumerate(patterns):
            result.append(Case(f'{split}_{j}_{k}_{pattern}', context=context, pattern=pattern,
                               seed=(71001 if development else 171001) + 101*j + k,
                               bits=64 if development else 128))
    return result


def suite_hash(cases):
    return hashlib.sha256(json.dumps([asdict(c) for c in cases], sort_keys=True).encode()).hexdigest()


def netlist(x, case):
    p1, p2, out, pn, en, rs = validate_design(x)
    lines = link_netlist(case.context, (8, 8), case.payload(), case.points_per_ui).splitlines()
    lines[0] = 'D2D transmitter sizing: shared six-variable design, 72 fixed MOS'
    mos_count = 0
    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        name = tokens[0]
        if name.startswith('M'):
            base = name.rsplit('_', 1)[0]
            if base.startswith('MPRE1'):
                width = p1 * (pn if base.endswith('P') else 1.)
            elif base.startswith('MPRE2'):
                width = p2 * (pn if base.endswith('P') else 1.)
            elif base.startswith('MH'):
                width = out * pn * en
            elif base.startswith('MF'):
                width = out * en
            elif base.startswith('MP'):
                width = out * pn
            elif base.startswith('MN'):
                width = out
            else:
                raise ValueError(f'Unrecognized MOS: {name}')
            lines[i] = re.sub(r'\bw=\S+', f'w={width:.12g}u', line)
            mos_count += 1
        elif name.startswith('RCH0_'):
            lane = name.rsplit('_', 1)[1]
            tokens[1] = 'pad_' + lane
            lines[i] = ' '.join(tokens)
    if mos_count != 72:
        raise ValueError('Unexpected transmitter topology')
    control = lines.index('.control')
    lines[control:control] = [f'RTERM_{k} tx_{k} pad_{k} {rs:.12g}' for k in range(2)]
    return '\n'.join(lines) + '\n'


def area_proxy(x):
    p1, p2, out, pn, en, _ = validate_design(x)
    return float(2 * .09 * (1 + pn) * (p1 + p2 + 8*out*(1 + en)))


def measure(wave, case):
    t, rx, currents = wave['t'], wave['rx'], wave['ivdd']
    bits = case.payload()
    c = case.context.lane0
    if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(rx).all() or not np.isfinite(currents).all():
        raise ValueError('Nonfinite or incomplete waveform')
    if not (np.diff(t) > 0).all():
        raise ValueError('Timestamps must be strictly increasing')
    t0, t1 = 16*c.ui, (16 + case.bits)*c.ui
    ts = np.r_[t0, t[(t > t0) & (t < t1)], t1]
    energy = [float(np.trapezoid(-c.vdd*np.interp(ts, t, currents[:, k]), ts) / case.bits*1e12)
              for k in range(2)]
    margins, errors = [], []
    for k, lane in enumerate((case.context.lane0, case.context.lane1)):
        times = (np.arange(16, 16+case.bits)[:, None] + lane.coarse_latency_ui + case.phase_ui +
                 lane.clock_offset_ui + np.linspace(-case.guard_ui, case.guard_ui, 9)[None, :])*lane.ui
        if times.min() < t[0] or times.max() > t[-1] or t[-1] < t1:
            raise ValueError('Incomplete sampling window')
        v = np.interp(times.ravel(), t, rx[:, k]).reshape(times.shape) / lane.vdd
        signed = (2*bits[k].astype(float)[:, None]-1)*(v-.5)
        margins.append(float(signed.min()))
        errors.append(int(np.count_nonzero((v[:, 4] >= .5) != bits[k])))
    samples = rx[(t >= t0) & (t <= t1)] / c.vdd
    overshoot = float(max(0., samples.max()-1., -samples.min()))
    valid = bool(min(energy) > 0 and np.isfinite(energy).all() and np.abs(samples).max() < 10)
    feasible = bool(valid and min(margins) >= case.margin_min and sum(errors) == 0 and
                    overshoot <= case.overshoot_max)
    return dict(valid=valid, feasible=feasible, margins=margins, errors=errors,
                overshoot_fraction=overshoot, energy_lane_pj_bit=energy,
                energy_pj_bit=float(np.mean(energy)), case=asdict(case))


def aggregate(records):
    valid = all(r.get('valid', False) for r in records)
    if not valid:
        return dict(valid=False, feasible=False, cases=records)
    margins = np.min([r['margins'] for r in records], axis=0).tolist()
    return dict(valid=True, feasible=all(r['feasible'] for r in records), margins=margins,
                energy_pj_bit=float(np.mean([r['energy_pj_bit'] for r in records])),
                overshoot_fraction=float(max(r['overshoot_fraction'] for r in records)), cases=records)


def cost(result, margin=.08, overshoot=.2):
    if not result['valid']:
        return 1e6
    if not result['feasible']:
        violation = np.maximum(0., (margin-np.asarray(result['margins']))/margin).sum()
        violation += max(0., (result['overshoot_fraction']-overshoot)/overshoot)
        return float(1 + violation)
    e = result['energy_pj_bit']
    return float(e/(1+e))


class Evaluator:
    """Each request counts; physical transients and cache hits are separate."""
    def __init__(self, cases, output, workers=1):
        self.cases = list(cases)
        if not self.cases or len({(c.margin_min,c.overshoot_max) for c in cases}) != 1:
            raise ValueError('A nonempty suite with common quality limits is required')
        if workers < 1:
            raise ValueError('workers must be positive')
        self.workers = workers
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.calls = self.transients = self.cache_hits = self.failed = 0
        self.design_queries = 0
        self.binary, self.env = link.simulator()
        self.simulator_version = subprocess.run([self.binary, '--version'], env=self.env,
            capture_output=True, text=True, check=True).stdout.strip()
        self.code_hash = file_hash(__file__)

    def _case(self, x, case, query, case_index):
        text = netlist(x, case)
        model_hash = file_hash(LINK_ROOT/'assets/45nm_bulk.txt')
        identity = dict(netlist=text.replace(str(LINK_ROOT), 'LINK_ROOT'), code=self.code_hash,
                        simulator=self.simulator_version, model=model_hash, case=asdict(case))
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        folder = self.output/'cache'/digest
        cached = folder/'measurement.json'
        receipt = self.output/'receipts'/f'q{query:05d}_c{case_index:02d}.json'
        if receipt.exists():
            raise FileExistsError(f'Existing evaluator request: {receipt}')
        hit = cached.exists()
        write_json(receipt, dict(status='pending', fingerprint=digest, cache_hit=hit, query=query,
                                 case_index=case_index, x=np.asarray(x).tolist()))
        if hit:
            data = json.loads(cached.read_text())
            if data['fingerprint'] != digest:
                raise ValueError('Measurement cache mismatch')
            write_json(receipt, dict(status='complete', fingerprint=digest, cache_hit=True,
                                     measurement_sha256=file_hash(cached)))
            return data['measurement'], True, False
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'circuit.cir').write_text(text)
        start = time.perf_counter()
        try:
            proc = subprocess.run([self.binary, '-b', 'circuit.cir'], cwd=folder, env=self.env,
                                  capture_output=True, text=True, timeout=120)
            log = proc.stdout + proc.stderr
            (folder/'ngspice.log').write_text(log)
            if proc.returncode or re.search(r'(?im)^\s*(?:fatal error|error on line|error:|fatal:)', log):
                raise RuntimeError(log[-1500:])
            raw = np.loadtxt(folder/'wave.txt', skiprows=1, ndmin=2)
            raw = raw[np.r_[np.diff(raw[:, 0]) > 0, True]]
            if raw.shape[1] != 7:
                raise ValueError('Unexpected SPICE columns')
            wave = dict(t=raw[:, 0], tx=raw[:, [1, 4]], rx=raw[:, [2, 5]], ivdd=raw[:, [3, 6]])
            result = measure(wave, case)
            result['area_proxy_um2'] = area_proxy(x)
            np.savez_compressed(folder/'wave.npz', **wave)
            (folder/'wave.txt').unlink()
            write_json(cached, dict(fingerprint=digest, measurement=result))
            write_json(receipt, dict(status='complete', fingerprint=digest, cache_hit=False,
                                     measurement_sha256=file_hash(cached), seconds=time.perf_counter()-start))
            return result, False, False
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            result = dict(valid=False, feasible=False, reason=str(error), case=asdict(case))
            write_json(receipt, dict(status='failed', fingerprint=digest, cache_hit=False,
                                     seconds=time.perf_counter()-start, reason=str(error)))
            return result, False, True

    def evaluate(self, x):
        x = validate_design(x)
        self.design_queries += 1
        jobs = [(x, c, self.design_queries, k) for k, c in enumerate(self.cases)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(lambda args: self._case(*args), jobs))
        self.calls += len(results)
        self.cache_hits += sum(hit for _, hit, _ in results)
        self.transients += sum(not hit for _, hit, _ in results)
        self.failed += sum(failed for _, _, failed in results)
        return aggregate([r for r, _, _ in results])

    def accounting(self):
        return dict(design_queries=self.design_queries, logical_measurements=self.calls,
                    spice_transient_attempts=self.transients, cache_hits=self.cache_hits,
                    failed_transients=self.failed, completed_transients=self.transients-self.failed,
                    suite_sha256=suite_hash(self.cases), simulator_version=self.simulator_version)
