"""Small D2D-inspired link: CMOS with RC or RLC transient, behavioral receiver.

This is not a UCIe compliance model. Settings stay fixed during each packet;
each measurement starts a fresh ngspice transient after a warmup sequence.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Context:
    vdd: float = 1.0
    temperature_c: float = 27.0
    rate_gbps: float = 2.0
    resistance_ohm: float = 30.0
    inductance_nh: float = 0.3  # Zero selects RC: no inductor is emitted.
    capacitance_pf: float = 0.3
    rx_capacitance_pf: float = 0.06
    clock_offset_ui: float = 0.0
    # Fixed coarse word alignment, chosen as part of the toy architecture.
    # Phase tuning covers [1,2] UI after the input launch. No oracle alignment.
    coarse_latency_ui: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Context must be finite')
        if min(self.vdd, self.rate_gbps, self.resistance_ohm,
               self.capacitance_pf, self.rx_capacitance_pf) <= 0:
            raise ValueError('Supply, rate, resistance and capacitances must be positive')
        if self.inductance_nh < 0:
            raise ValueError('Inductance must be nonnegative; zero selects RC')
        if abs(self.clock_offset_ui) > .25:
            raise ValueError('This packet alignment model supports at most +/-0.25 UI skew')
        if not 0 <= self.coarse_latency_ui <= 2:
            raise ValueError('Coarse alignment must be between 0 and 2 UI')

    @property
    def ui(self):
        return 1e-9 / self.rate_gbps

@dataclass(frozen=True)
class Setting:
    cells: int = 8
    vref_code: int = 16
    phase_code: int = 32

    def __post_init__(self):
        for x, lo, hi in [(self.cells,1,8),(self.vref_code,0,32),(self.phase_code,0,64)]:
            if not isinstance(x, (int,np.integer)) or not lo <= x <= hi:
                raise ValueError('Expected integer cells=1..8, vref_code=0..32, phase_code=0..64')

    @property
    def vref_fraction(self):
        return .25 + .5 * self.vref_code / 32

    @property
    def phase_ui(self):
        return self.phase_code / 64

    def action(self):
        return np.array([2*(self.cells-1)/7-1, self.vref_code/16-1,
                         self.phase_code/32-1], dtype=np.float32)

    @classmethod
    def from_action(cls, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('Action must have three finite values')
        a = np.clip(a,-1,1)
        return cls(int(np.rint((a[0]+1)*3.5))+1,
                   int(np.rint((a[1]+1)*16)), int(np.rint((a[2]+1)*32)))

def bits_for(seed=123, n_bits=256):
    """Seeded pseudorandom test data, not a standardized maximal-length PRBS."""
    if n_bits < 8 or n_bits % 8:
        raise ValueError('Payload size must be a positive multiple of 8')
    return np.random.default_rng(seed).integers(0,2,n_bits,dtype=np.uint8)

def serialize_bytes(data):
    """Functional 8:1 serializer; MSB-first, no coding/scrambling overhead."""
    return np.unpackbits(np.frombuffer(bytes(data),dtype=np.uint8),bitorder='big')

def deserialize_bits(bits):
    """Functional 1:8 deserializer with known byte alignment."""
    b=np.asarray(bits)
    if b.ndim!=1 or len(b)%8 or not np.isin(b,[0,1]).all():
        raise ValueError('Expected aligned, binary bits in multiples of eight')
    return np.packbits(b.astype(np.uint8),bitorder='big').tobytes()

def simulator():
    binary = os.environ.get('NGSPICE_BIN') or shutil.which('ngspice')
    runtime = ROOT/'runtime'
    if not binary:
        for r in [runtime, ROOT.parent/'opamp_offline_pilot/runtime']:
            if (r/'usr/bin/ngspice').exists():
                runtime = r
                binary = str(r/'usr/bin/ngspice')
                break
    if not binary:
        raise RuntimeError('Install ngspice or set NGSPICE_BIN')
    env = dict(os.environ)
    lib = runtime/'usr/lib/x86_64-linux-gnu'
    if lib.exists():
        env['LD_LIBRARY_PATH'] = str(lib) + ':' + env.get('LD_LIBRARY_PATH','')
    return str(Path(binary).resolve()), env

def packet(payload):
    # Known bit alignment, no clock acquisition or word alignment algorithm.
    return np.concatenate([np.tile([0,1],8), payload, [0,1,0,1]]).astype(np.uint8)

def netlist(context, cells, payload, points_per_ui=64, sections=3):
    c = context
    if not 1 <= cells <= 8 or points_per_ui < 16 or sections < 1:
        raise ValueError('Invalid simulator configuration')
    bits = packet(payload)
    # Two predriver inversions + one output inversion. Source is complementary.
    src = (1-bits)*c.vdd
    rise = .04*c.ui
    pwl = [(0.,float(src[0]))]
    for i in range(1,len(src)):
        if src[i] != src[i-1]:
            pwl.extend([(i*c.ui,float(src[i-1])),(i*c.ui+rise,float(src[i]))])
    stop = len(bits)*c.ui
    pwl.append((stop,float(src[-1])))
    lines = [
        'D2D-inspired segmented CMOS driver, 36 MOS, behavioral RX in Python',
        f'.include "{ROOT / "assets/45nm_bulk.txt"}"',
        f'.temp {c.temperature_c:.12g}',
        '.options reltol=1e-5 abstol=1e-12 vntol=1e-7',
        f'VDD vdd 0 {c.vdd:.12g}',
        'VIN din 0 PWL(',
        *[f'+ {t:.15g} {v:.12g}' for t,v in pwl], '+ )',
        'MPRE1P pre din vdd vdd pmos w=2u l=90n',
        'MPRE1N pre din 0 0 nmos w=1u l=90n',
        'MPRE2P drive pre vdd vdd pmos w=8u l=90n',
        'MPRE2N drive pre 0 0 nmos w=4u l=90n',
    ]
    for k in range(8):
        en = c.vdd if k < cells else 0.
        # Header/footer isolate disabled cells; they are NOT removed from netlist.
        lines.extend([
            f'VEN{k} en{k} 0 {en:.12g}',
            f'VENB{k} enb{k} 0 {c.vdd-en:.12g}',
            f'MH{k} hp{k} enb{k} vdd vdd pmos w=8u l=90n',
            f'MP{k} tx drive hp{k} vdd pmos w=4u l=90n',
            f'MN{k} tx drive fn{k} 0 nmos w=2u l=90n',
            f'MF{k} fn{k} en{k} 0 0 nmos w=4u l=90n',
        ])
    prev = 'tx'
    for k in range(sections):
        node = 'rx' if k == sections-1 else f'ch{k}'
        if c.inductance_nh == 0:
            # Connect the resistor directly; removing L without reconnecting
            # its terminals would leave an open channel.
            lines.append(f'RCH{k} {prev} {node} {c.resistance_ohm/sections:.12g}')
        else:
            lines += [f'RCH{k} {prev} rl{k} {c.resistance_ohm/sections:.12g}',
                      f'LCH{k} rl{k} {node} {c.inductance_nh/sections:.12g}n']
        lines.append(f'CCH{k} {node} 0 {c.capacitance_pf/sections:.12g}p')
        prev = node
    lines += [f'CRX rx 0 {c.rx_capacitance_pf:.12g}p',
              '.control', 'set noaskquit', 'set wr_vecnames', 'set wr_singlescale',
              'option numdgt=15',
              f'tran {c.ui/points_per_ui:.15g} {stop:.15g} 0 {c.ui/points_per_ui:.15g}',
              'wrdata wave.txt v(tx) v(rx) i(vdd)', 'quit', '.endc', '.end']
    return '\n'.join(lines)+'\n'

def simulate(context, cells, payload, keep=None, points_per_ui=64, sections=3):
    binary, env = simulator()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='d2d_') as tmp:
        path = Path(keep) if keep is not None else Path(tmp)
        path.mkdir(parents=True,exist_ok=True)
        text = netlist(context,cells,payload,points_per_ui,sections)
        if keep is not None:
            # Kept cases remain executable after moving the whole project.
            relative_model = os.path.relpath(ROOT/'assets/45nm_bulk.txt',path.resolve())
            text = text.replace(str(ROOT/'assets/45nm_bulk.txt'),relative_model)
        (path/'wave.txt').unlink(missing_ok=True)
        (path/'wave.npz').unlink(missing_ok=True)
        (path/'circuit.cir').write_text(text)
        proc = subprocess.run([binary,'-b','circuit.cir'],cwd=path,env=env,
                              capture_output=True,text=True,timeout=120)
        log = proc.stdout+proc.stderr
        if keep is not None:
            (path/'ngspice.log').write_text(log)
            (path/'context.json').write_text(json.dumps(asdict(context),indent=2))
            np.save(path/'payload.npy',payload)
        if proc.returncode or re.search(r'(?im)^\s*(?:fatal error|error on line|error:|fatal:)', log) or not (path/'wave.txt').exists():
            raise RuntimeError(f'ngspice failed: {log[-1800:]}')
        raw = np.loadtxt(path/'wave.txt',skiprows=1,ndmin=2)
    # Discard duplicate timestamps if the simulator emits any at breakpoints.
    raw = raw[np.r_[np.diff(raw[:,0])>0,True]]
    if raw.shape[1] != 4 or not np.isfinite(raw).all() or len(raw)<len(payload)*16:
        raise ValueError('Invalid transient output')
    if raw[-1,0] < (16+len(payload)+3)*context.ui:
        raise ValueError('Truncated transient')
    t0,t1 = 16*context.ui,(16+len(payload))*context.ui
    inside = (raw[:,0]>t0)&(raw[:,0]<t1)
    time_points = np.r_[t0,raw[inside,0],t1]
    power = -context.vdd*np.interp(time_points,raw[:,0],raw[:,3])
    # Net energy at VDD includes two predrivers + eight fixed output cells.
    energy = np.trapezoid(power,time_points)/len(payload)*1e12
    if energy <= 0:
        raise ValueError('Nonpositive supply energy')
    return {'t':raw[:,0], 'tx':raw[:,1], 'rx':raw[:,2], 'ivdd':raw[:,3],
            'energy_pj_bit':float(energy), 'seconds':time.perf_counter()-start,
            'netlist_sha256':hashlib.sha256(text.encode()).hexdigest()}

def measure(wave, context, setting, payload, guard_ui=.10):
    """Measured packet errors and 9-point guard-window voltage margin.

    Guard sampling is a software diagnostic using a known training sequence;
    it would require extra phase probes or an eye monitor in real hardware.
    No random jitter, thermal noise, metastability or clock recovery is modeled.
    """
    if not 0 <= guard_ui <= .25:
        raise ValueError('Guard half-width must be in [0,0.25] UI')
    base = (np.arange(16,16+len(payload)) + context.coarse_latency_ui
            + setting.phase_ui + context.clock_offset_ui)
    offsets = np.linspace(-guard_ui,guard_ui,9)
    times = (base[:,None]+offsets[None,:])*context.ui
    voltages = np.interp(times.ravel(),wave['t'],wave['rx']).reshape(len(payload),9)
    vref = context.vdd*setting.vref_fraction
    recovered = (voltages[:,4] >= vref).astype(np.uint8)
    errors = int(np.count_nonzero(recovered != payload))
    signed = (2*payload.astype(float)[:,None]-1)*(voltages-vref)/context.vdd
    return {'bit_errors':errors, 'bits':len(payload), 'packet_error_fraction':errors/len(payload),
            'center_margin_vdd':float(signed[:,4].min()),
            'guard_margin_vdd':float(signed.min()), 'guard_ui':guard_ui,
            'energy_pj_bit':wave['energy_pj_bit'],
            'recovered_bytes_hex':deserialize_bits(recovered).hex()}

def feasible(metrics, margin=.08, max_energy_pj=1.0):
    return bool(metrics['bit_errors']==0 and metrics['guard_margin_vdd'] >= margin
                and metrics['energy_pj_bit'] <= max_energy_pj)

class LinkSimulator:
    """Cache only repeats of the SAME circuit + packet. Not a learned surrogate."""
    def __init__(self):
        self.cache = {}
        self.spice_calls = 0
        self.measurements = 0

    def wave(self,context,cells,payload):
        key = (context,cells,payload.tobytes())
        if key not in self.cache:
            self.cache[key] = simulate(context,cells,payload)
            self.spice_calls += 1
        return self.cache[key]

    def evaluate(self,context,setting,payload,guard_ui=.10):
        self.measurements += 1
        return measure(self.wave(context,setting.cells,payload),context,setting,payload,guard_ui)
