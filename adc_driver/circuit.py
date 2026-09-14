"""SKY130 transistor-level amplifier driving a MOS track-and-hold input.

This is an ADC input-load testbench, not a transistor implementation of a SAR
converter. MOS switches and clock inverters are physical BSIM4 devices. Passive
R/C, input sources, and the amplifier reference-current source are ideal.
"""
from dataclasses import asdict, dataclass, fields
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
MODEL = ROOT/'models/sky130_1v8_tt.spice'
MODEL_SHA256 = '20b049c1a68762d62c49ba4249e2b75e5889ac97739c942cbe4934a71426f2f2'
NAMES = ['wi_um','wp_um','wt_um','wop_um','won_um','cc_pf','ibias_ua','riso_ohm','cfilter_pf']
LOW = np.array([2,2,2,5,5,.3,3,20,1.],dtype=float)
HIGH = np.array([30,30,30,80,80,6,30,2000,100.],dtype=float)
NOMINAL = np.array([10,5,10,20,20,1,10,200,10.],dtype=float)
PRIOR_OPAMP = np.array([5.625785739096434,3.2114120679559917,2.8129326293580106,
                        80.,14.656307309420036,.7846010306532256,
                        17.540315892636617,200.,10.])
ADC_REFERENCE = PRIOR_OPAMP.copy()
ADC_REFERENCE[7:] = [500.,1.]  # Documented development sizing; evaluated afresh in each search.
SIGNALS = ['vin','vout','vfilter','vsample','clkn','clkp','clkcmd',
           'i_avdd','i_clkdd','i_vin','i_clkcmd']


def write_json(path, value):
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(tuple,list,np.ndarray)): return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
        if isinstance(v,np.generic): return v.item()
        return v
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


@dataclass(frozen=True)
class Context:
    name: str = 'nominal'
    vdd_v: float = 1.8
    temperature_c: float = 27.
    csample_pf: float = 15.
    fs_hz: float = 1e6
    acquisition_ns: float = 200.
    edge_ns: float = 2.
    input_lead_ns: float = 100.
    read_delay_ns: float = 20.
    maxstep_ns: float = 2.
    warmup_us: float = 2.
    dummy_cycles: int = 2
    initial_v: float = .9
    levels_v: tuple = (.8,1.1,.85,1.05,.8,.9)
    error_limit_mv: float = 1.
    # Fixed switch sizing: one complementary transmission gate, not an ideal S.
    switch_wn_um: float = 4.
    switch_wp_um: float = 8.

    def validate(self):
        numeric=[v for k,v in asdict(self).items() if isinstance(v,(int,float))]
        if not np.isfinite(numeric).all(): raise ValueError('Nonfinite context')
        if not 1.6 <= self.vdd_v <= 1.89: raise ValueError('Supply outside configured SKY130 experiment range')
        if not 0 < self.csample_pf or not 0 < self.fs_hz: raise ValueError('Positive load and sampling rate required')
        period_ns=1e9/self.fs_hz
        if self.acquisition_ns <= 4*self.edge_ns: raise ValueError('Acquisition must include a flat tracking interval')
        if self.input_lead_ns <= self.edge_ns: raise ValueError('Input change must finish before acquisition')
        if period_ns-self.acquisition_ns <= self.input_lead_ns+self.read_delay_ns+10:
            raise ValueError('Insufficient hold time for stimulus and measurement')
        if self.read_delay_ns <= 2*self.edge_ns or self.maxstep_ns <= 0: raise ValueError('Invalid read delay/time step')
        if self.error_limit_mv <= 0 or self.dummy_cycles < 1 or len(self.levels_v)<2:
            raise ValueError('Positive error limit, dummy cycle, and multiple samples required')
        if self.warmup_us*1e3 <= self.input_lead_ns: raise ValueError('Warmup too short')
        if not all(0 < v < self.vdd_v for v in (self.initial_v,*self.levels_v)):
            raise ValueError('Input levels must be within the supply rails')
        return self

    @classmethod
    def from_dict(cls, data):
        unknown=set(data)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f'Unknown context keys: {sorted(unknown)}')
        d=dict(data)
        if 'levels_v' in d: d['levels_v']=tuple(d['levels_v'])
        return cls(**d).validate()


def encode(x): return np.log(np.asarray(x)/LOW)/np.log(HIGH/LOW)
def decode(u): return np.clip(LOW*np.exp(np.asarray(u)*np.log(HIGH/LOW)),LOW,HIGH)


def validate_design(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(len(NAMES),) or not np.isfinite(x).all(): raise ValueError('Expected nine finite design values')
    if (x<LOW-1e-10).any() or (x>HIGH+1e-10).any(): raise ValueError('Design outside declared bounds')
    return x


def mos(name,d,g,s,b,kind,w,l=.15):
    return (f'X{name} {d} {g} {s} {b} sky130_fd_pr__{kind}_01v8 '
            f'w={w:.12g} l={l:.12g} ad={w*.29:.12g} as={w*.29:.12g} '
            f'pd={2*(w+.29):.12g} ps={2*(w+.29):.12g}')


def amplifier(x):
    wi,wp,wt,wop,won,cc,ib=x[:7]
    return '\n'.join([
        '.subckt opamp inp inn out vdd',f'IREF vdd bias {ib:.12g}u',
        mos('REF','bias','bias','0','0','nfet',5,.5),
        mos('TAIL','tail','bias','0','0','nfet',wt,.5),
        mos('IN1','n1','inn','tail','0','nfet',wi,.5),
        mos('IN2','n2','inp','tail','0','nfet',wi,.5),
        mos('P1','n1','n1','vdd','vdd','pfet',wp,.5),
        mos('P2','n2','n1','vdd','vdd','pfet',wp,.5),
        mos('P3','out','n2','vdd','vdd','pfet',wop,.5),
        mos('N3','out','bias','0','0','nfet',won,.5),
        f'CC n2 out {cc:.12g}p','.ends opamp'])


def schedule(c):
    c.validate()
    levels=[c.initial_v]*c.dummy_cycles+list(c.levels_v)
    starts=c.warmup_us*1e-6+np.arange(len(levels))/c.fs_hz
    return levels,starts,starts+c.acquisition_ns*1e-9,starts[-1]+1/c.fs_hz


def pwl(points):
    return 'PWL('+' '.join(f'{t:.15g} {v:.12g}' for t,v in points)+')'


def netlist(x, c=Context(), standalone=False):
    x=validate_design(x);c.validate(); levels,starts,ends,stop=schedule(c)
    edge=c.edge_ns*1e-9
    vp=[(0,c.initial_v)]; previous=c.initial_v
    for level,start in zip(levels,starts):
        change=start-c.input_lead_ns*1e-9
        vp.extend([(change,previous),(change+edge,level)])
        previous=level
    vp.append((stop,previous))
    cp=[(0,0.)]
    for start,end in zip(starts,ends):
        cp.extend([(start,0.),(start+edge,c.vdd_v),(end-edge,c.vdd_v),(end,0.)])
    cp.append((stop,0.))
    model=MODEL.read_text() if standalone else '.include "'+MODEL.as_posix()+'"'
    return '\n'.join([
        'SKY130 ADC input driver: 8-MOS amplifier, CMOS TG and clock buffer',
        model, '.options numdgt=15 reltol=1e-5 abstol=1e-12 vntol=1e-8',
        f'.temp {c.temperature_c:.12g}',amplifier(x),
        f'VAVDD avdd 0 {c.vdd_v:.12g}',f'VCLKDD clkdd 0 {c.vdd_v:.12g}',
        'VIN vin 0 '+pwl(vp),'VCLK clkcmd 0 '+pwl(cp),
        'XAMP vin out out avdd opamp',
        'COUT out 0 2p',f'RISO out adc_in {x[7]:.12g}',
        f'CFILTER adc_in 0 {x[8]:.12g}p',
        # Two actual CMOS inverters make complementary TG clocks, including skew.
        mos('CK0N','clkp','clkcmd','0','0','nfet',2),
        mos('CK0P','clkp','clkcmd','clkdd','clkdd','pfet',4),
        mos('CK1N','clkn','clkp','0','0','nfet',2),
        mos('CK1P','clkn','clkp','clkdd','clkdd','pfet',4),
        mos('SWN','adc_in','clkn','sample','0','nfet',c.switch_wn_um),
        mos('SWP','adc_in','clkp','sample','avdd','pfet',c.switch_wp_um),
        f'CSAMPLE sample 0 {c.csample_pf:.12g}p',
        # Explicit 1 Tohm leakage reference for a finite initial DC solution.
        'RLEAK sample 0 1e12',
        '.control','set noaskquit','set wr_vecnames','set wr_singlescale','set numdgt=15',
        f'tran {c.maxstep_ns*1e-9:.12g} {stop:.15g} 0 {c.maxstep_ns*1e-9:.12g}',
        'wrdata waveform.dat v(vin) v(out) v(adc_in) v(sample) v(clkn) v(clkp) v(clkcmd) '
        'i(vavdd) i(vclkdd) i(vin) i(vclk)',
        'quit','.endc','.end',''])


def binary_env():
    binary=os.environ.get('NGSPICE_BIN') or shutil.which('ngspice')
    if not binary: raise RuntimeError('ngspice is missing; install it or set NGSPICE_BIN to its executable')
    binary=Path(binary).expanduser().resolve()
    if not binary.is_file(): raise RuntimeError(f'ngspice executable does not exist: {binary}')
    env=os.environ.copy()
    lib=binary.parent.parent/'lib/x86_64-linux-gnu'
    if lib.exists(): env['LD_LIBRARY_PATH']=str(lib)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    return str(binary),env


def verify_runtime():
    if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=MODEL_SHA256: raise RuntimeError('SKY130 model checksum mismatch')
    binary,env=binary_env()
    p=subprocess.run([binary,'--version'],env=env,capture_output=True,text=True,timeout=10)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    return dict(binary=binary,version=p.stdout.strip(),model_sha256=MODEL_SHA256)


def interval(t,y,a,b):
    if not t[0]<=a<b<=t[-1]+1e-15: raise ValueError('Incomplete waveform for requested interval')
    interior=(t>a)&(t<b)
    return np.r_[a,t[interior],b],np.r_[np.interp(a,t,y),y[interior],np.interp(b,t,y)]


def integral(t,y,a,b):
    tt,yy=interval(t,y,a,b)
    return float(np.trapezoid(yy,tt))


def measure(a,c):
    c.validate();levels,starts,ends,stop=schedule(c)
    if a.ndim!=2 or a.shape[1]!=len(SIGNALS)+1 or not np.isfinite(a).all():
        raise ValueError('Incomplete or nonfinite transient data')
    t=a[:,0]
    if np.any(np.diff(t)<=0) or t[-1]<stop-1e-13: raise ValueError('Nonmonotone or truncated time series')
    sig={name:a[:,j+1] for j,name in enumerate(SIGNALS)}
    cycles=[]
    for k in range(c.dummy_cycles,len(levels)):
        target=levels[k];end=ends[k];next_start=starts[k]+1/c.fs_hz
        before=end-(c.edge_ns+3)*1e-9
        sample_t=end+c.read_delay_ns*1e-9
        cap=float(np.interp(sample_t,t,sig['vsample']))
        _,held=interval(t,sig['vsample'],sample_t,next_start-10e-9)
        off_n=float(np.interp(sample_t,t,sig['clkn']))
        off_p=float(np.interp(sample_t,t,sig['clkp']))
        energy_amp=integral(t,-c.vdd_v*sig['i_avdd'],starts[k],next_start)*1e12
        energy_clock=integral(t,-c.vdd_v*sig['i_clkdd'],starts[k],next_start)*1e12
        energy_stim=integral(t,-sig['vin']*sig['i_vin']-sig['clkcmd']*sig['i_clkcmd'],starts[k],next_start)*1e12
        cycles.append(dict(cycle=k-c.dummy_cycles,target_v=target,sample_time_s=sample_t,
            stored_v=cap,aperture_error_mv=(cap-target)*1000,
            acquisition_error_mv=(float(np.interp(before,t,sig['vsample']))-target)*1000,
            turnoff_change_mv=(cap-float(np.interp(before,t,sig['vsample'])))*1000,
            worst_hold_error_mv=float(np.max(abs(held-target))*1000),
            switch_off=bool(off_n<.1*c.vdd_v and off_p>.9*c.vdd_v),
            amplifier_energy_pj=energy_amp,clock_energy_pj=energy_clock,
            driver_energy_pj=energy_amp+energy_clock,stimulus_energy_pj=energy_stim))
    worst=max(r['worst_hold_error_mv'] for r in cycles)
    energy=float(np.mean([r['driver_energy_pj'] for r in cycles]))
    mask=t>=starts[c.dummy_cycles]
    bounded=all(sig[name][mask].min()>-.1 and sig[name][mask].max()<c.vdd_v+.1
                for name in ['vout','vfilter','vsample'])
    valid=bool(bounded and energy>0 and all(r['switch_off'] for r in cycles))
    return dict(valid=valid,reason='ok' if valid else 'Unbounded voltage, nonpositive energy, or switch not off',
        feasible=bool(valid and worst<=c.error_limit_mv),worst_hold_error_mv=worst,
        max_aperture_error_mv=max(abs(r['aperture_error_mv']) for r in cycles),
        max_acquisition_error_mv=max(abs(r['acquisition_error_mv']) for r in cycles),
        mean_driver_energy_pj=energy,mean_driver_power_uw=energy*c.fs_hz*1e-6,
        mean_amplifier_energy_pj=float(np.mean([r['amplifier_energy_pj'] for r in cycles])),
        mean_clock_energy_pj=float(np.mean([r['clock_energy_pj'] for r in cycles])),
        mean_stimulus_energy_pj=float(np.mean([r['stimulus_energy_pj'] for r in cycles])),
        normalized_violation=max(0.,worst/c.error_limit_mv-1),cycles=cycles)


def evaluate(x,c=Context(),output=None,timeout_s=60):
    x=validate_design(x);c.validate();binary,env=binary_env()
    tmp=tempfile.TemporaryDirectory(prefix='adc_driver_') if output is None else None
    folder=Path(tmp.name if tmp else output);folder.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    r=dict(design=dict(zip(NAMES,x.tolist())),x=x.tolist(),context=asdict(c),valid=False,feasible=False)
    try:
        (folder/'waveform.dat').unlink(missing_ok=True)
        (folder/'waveform.npz').unlink(missing_ok=True)
        (folder/'circuit.cir').write_text(netlist(x,c))
        p=subprocess.run([binary,'-b','circuit.cir'],cwd=folder,env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout_s)
        (folder/'ngspice.log').write_text(p.stdout)
        if p.returncode or re.search(r'(?im)^\s*(fatal error|error on line|error:|fatal:)',p.stdout):
            raise RuntimeError(p.stdout[-1800:])
        a=np.loadtxt(folder/'waveform.dat',skiprows=1)
        r.update(measure(a,c))
        if output is not None:
            np.savez_compressed(folder/'waveform.npz',time=a[:,0],**{n:a[:,j+1] for j,n in enumerate(SIGNALS)})
    except (subprocess.TimeoutExpired,RuntimeError,ValueError,OSError) as e:
        r['reason']=f'{type(e).__name__}: {e}'
    finally:
        r['elapsed_seconds']=time.perf_counter()-begin
        if output is not None: write_json(folder/'metrics.json',r)
        if tmp: tmp.cleanup()
    return r


def aggregate(records):
    valid=all(r.get('valid',False) for r in records)
    return dict(valid=valid,feasible=valid and all(r['feasible'] for r in records),
                worst_hold_error_mv=max(r.get('worst_hold_error_mv',1e9) for r in records),
                normalized_violation=max(r.get('normalized_violation',1e9) for r in records),
                mean_driver_energy_pj=float(np.mean([r.get('mean_driver_energy_pj',1e12) for r in records])),
                contexts=records)


def evaluate_suite(x,contexts,output=None):
    return aggregate([evaluate(x,c,None if output is None else Path(output)/f'{i:02d}_{c.name}')
                      for i,c in enumerate(contexts)])


def cost(r):
    if not r['valid']: return 1e6
    if not r['feasible']: return 1+float(np.log1p(r['normalized_violation']))
    e=r['mean_driver_energy_pj']
    return e/(e+1000.)


def load_contexts(path=None):
    if path is None: return [Context()]
    data=json.loads(Path(path).read_text())
    return [Context.from_dict(c) for c in (data if isinstance(data,list) else [data])]
