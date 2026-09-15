"""Two real 36-MOS transmitters with distributed capacitive lane coupling.

Receiver sampling remains behavioral. No fabricated-PDK accuracy, thermal
noise, mutual inductance, supply droop, CDR or dynamic register switching claim.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import numpy as np
import link

ROOT=link.ROOT
LEVELS=(1,2,4,8)

@dataclass(frozen=True)
class CoupledContext:
    lane0: link.Context
    lane1: link.Context
    coupling_pf: float=.10
    def __post_init__(self):
        if not np.isfinite(self.coupling_pf) or self.coupling_pf<0:
            raise ValueError('Nonnegative finite coupling required')
        for k in ['vdd','temperature_c','rate_gbps']:
            if getattr(self.lane0,k)!=getattr(self.lane1,k):
                raise ValueError('Shared supply, temperature and rate required')
    @classmethod
    def from_dict(cls,d):
        return cls(link.Context(**d['lane0']),link.Context(**d['lane1']),d['coupling_pf'])

@dataclass(frozen=True)
class CoupledSetting:
    strength0: int=3
    phase0: int=32
    strength1: int=3
    phase1: int=32
    def __post_init__(self):
        for x,hi in zip(self.codes(),[3,64,3,64]):
            if not isinstance(x,(int,np.integer)) or not 0<=x<=hi:
                raise ValueError('Strength codes 0..3; phase codes 0..64')
    def codes(self):return (self.strength0,self.phase0,self.strength1,self.phase1)
    def action(self):return np.array(self.codes(),np.float32)/np.array([1.5,32,1.5,32],np.float32)-1
    def strengths(self):return (LEVELS[self.strength0],LEVELS[self.strength1])
    @classmethod
    def from_action(cls,a):
        a=np.asarray(a,float)
        if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('Four finite action entries required')
        return cls(*np.rint((np.clip(a,-1,1)+1)*[1.5,32,1.5,32]).astype(int).tolist())

def payloads(seed,n):return np.stack([link.bits_for(seed,n),link.bits_for(seed+100003,n)])

def netlist(context,strengths,bits,points_per_ui=32):
    if np.shape(bits)[0]!=2 or not np.isin(bits,[0,1]).all():raise ValueError('Two binary packets required')
    if len(bits[0])!=len(bits[1]):raise ValueError('Equal lane packet sizes required')
    c=context.lane0
    lines=['Two capacitively coupled lanes, 72 fixed MOS devices',
           f'.include "{ROOT / "assets/45nm_bulk.txt"}"',f'.temp {c.temperature_c:.12g}',
           '.options reltol=1e-5 abstol=1e-12 vntol=1e-7']
    for j,(lane,n,payload) in enumerate(zip([context.lane0,context.lane1],strengths,bits)):
        source=link.netlist(lane,n,payload,points_per_ui).splitlines()
        body=source[source.index(next(x for x in source if x.startswith('VDD '))):source.index('.control')]
        for line in body:
            if line.startswith('+'):lines.append(line);continue
            tok=line.split();device=tok[0][0];tok[0]+=f'_{j}'
            terminals=4 if device=='M' else 2
            for k in range(1,terminals+1):
                if tok[k]!='0':tok[k]+=f'_{j}'
            lines.append(' '.join(tok))
    if context.coupling_pf>0:
        for k,node in enumerate(['ch0','ch1','rx']):
            lines.append(f'CCOUP{k} {node}_0 {node}_1 {context.coupling_pf/3:.12g}p')
    stop=len(link.packet(bits[0]))*c.ui
    lines+=['.control','set num_threads=1','set noaskquit','set wr_vecnames','set wr_singlescale','option numdgt=15',
            f'tran {c.ui/points_per_ui:.15g} {stop:.15g} 0 {c.ui/points_per_ui:.15g}',
            'wrdata wave.txt v(tx_0) v(rx_0) i(vdd_0) v(tx_1) v(rx_1) i(vdd_1)',
            'quit','.endc','.end']
    return '\n'.join(lines)+'\n'

def fingerprint(context,strengths,bits,points_per_ui=32):
    model=ROOT/'assets/45nm_bulk.txt'
    # Thread scheduling is not an electrical-model parameter; exact before/after
    # waveform equivalence is recorded in thread_equivalence.json.
    text=netlist(context,strengths,bits,points_per_ui).replace(str(model),'MODEL_FILE').replace('set num_threads=1\n','')
    return hashlib.sha256(text.encode()+model.read_bytes()).hexdigest()

def simulate(context,strengths,bits,keep=None,points_per_ui=32):
    binary,env=link.simulator();start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='d2d_coupled_') as tmp:
        path=Path(keep) if keep is not None else Path(tmp);path.mkdir(parents=True,exist_ok=True)
        text=netlist(context,strengths,bits,points_per_ui)
        if keep is not None:
            text=text.replace(str(ROOT/'assets/45nm_bulk.txt'),os.path.relpath(ROOT/'assets/45nm_bulk.txt',path.resolve()))
        (path/'wave.txt').unlink(missing_ok=True)
        (path/'wave.npz').unlink(missing_ok=True)
        (path/'circuit.cir').write_text(text)
        proc=subprocess.run([binary,'-b','circuit.cir'],cwd=path,env=env,capture_output=True,text=True,timeout=120)
        log=proc.stdout+proc.stderr
        if keep is not None:(path/'ngspice.log').write_text(log)
        if proc.returncode or re.search(r'(?im)^\s*(?:fatal error|error on line|error:|fatal:)',log) or not (path/'wave.txt').exists():
            raise RuntimeError(log[-2000:])
        raw=np.loadtxt(path/'wave.txt',skiprows=1,ndmin=2)
        if keep is not None:
            (path/'ngspice.log').write_text(proc.stdout+proc.stderr)
            (path/'context.json').write_text(json.dumps(asdict(context),indent=2)+'\n')
            np.save(path/'payload.npy',bits)
            (path/'wave.txt').unlink()
    raw=raw[np.r_[np.diff(raw[:,0])>0,True]]
    if raw.shape[1]!=7 or not np.isfinite(raw).all():raise ValueError('Invalid SPICE waveform')
    c=context.lane0;n=len(bits[0]);t0=16*c.ui;t1=(16+n)*c.ui
    if raw[-1,0]<(16+n+3)*c.ui:raise ValueError('Incomplete simulation')
    ts=np.r_[t0,raw[(raw[:,0]>t0)&(raw[:,0]<t1),0],t1]
    energy=np.array([np.trapezoid(-c.vdd*np.interp(ts,raw[:,0],raw[:,k]),ts)/n*1e12 for k in [3,6]])
    if np.any(energy<=0):raise ValueError('Nonpositive supply energy')
    out={'t':raw[:,0],'tx':raw[:,[1,4]],'rx':raw[:,[2,5]],'ivdd':raw[:,[3,6]],
         'energy_lane_pj_bit':energy,'energy_pj_bit':float(energy.mean()),
         'seconds':time.perf_counter()-start,'simulator_threads':1,'fingerprint':fingerprint(context,strengths,bits,points_per_ui)}
    if keep is not None:np.savez_compressed(Path(keep)/'wave.npz',**out)
    return out

def phase_table(wave,context,bits,guard_ui=.10):
    margins=np.empty((2,65));errors=np.empty((2,65),np.int16)
    for j,lane in enumerate([context.lane0,context.lane1]):
        times=(np.arange(16,16+len(bits[j]))[:,None,None]+lane.coarse_latency_ui+
               np.arange(65)[None,:,None]/64+lane.clock_offset_ui+np.linspace(-guard_ui,guard_ui,9)[None,None,:])*lane.ui
        v=np.interp(times.ravel(),wave['t'],wave['rx'][:,j]).reshape(times.shape)/lane.vdd
        signed=(2*bits[j].astype(float)[:,None,None]-1)*(v-.5)
        margins[j]=signed.min(axis=(0,2))
        errors[j]=np.count_nonzero((v[:,:,4]>=.5)!=bits[j][:,None],axis=0)
    return margins,errors

def measure(wave,context,setting,bits,guard_ui=.10):
    margins,errors=phase_table(wave,context,bits,guard_ui)
    phases=[setting.phase0,setting.phase1]
    return {'margins':[float(margins[j,phases[j]]) for j in range(2)],
            'errors':[int(errors[j,phases[j]]) for j in range(2)],
            'energy_pj_bit':float(wave['energy_pj_bit']),'bits_per_lane':len(bits[0])}
