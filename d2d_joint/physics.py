"""Fresh transistor simulations, then bounded receiver-time interpolation."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from d2d_sizing import circuit
from d2d_sizing.circuit import Case, Evaluator, file_hash, write_json


def suites():
    development=[];test=[]
    for split,settings,base,dest in [('development',[(1.,27.,30.,.30,.06),(.95,65.,40.,.34,.10)],641000,development),
                                    ('test',[(.98,42.,35.,.32,.08),(1.02,55.,38.,.33,.09)],971000,test)]:
        for env,(v,temp,r,c,coupling) in enumerate(settings):
            lanes=[circuit.link.Context(vdd=v,temperature_c=temp,resistance_ohm=r+2*k,
                 capacitance_pf=c,inductance_nh=.3,rx_capacitance_pf=.06,clock_offset_ui=0.) for k in range(2)]
            for packet in range(2):
                dest.append(Case(f'{split}_e{env}_p{packet}',context=circuit.CoupledContext(*lanes,coupling),
                    seed=base+env*101+packet,bits=64,pattern='random',points_per_ui=64))
    return development,test


class PhaseBank:
    def __init__(self,phase,voltage,bits,energy,overshoot,cases,x):
        self.phase=np.asarray(phase);self.voltage=np.asarray(voltage);self.bits=np.asarray(bits)
        self.energy=np.asarray(energy);self.overshoot=np.asarray(overshoot);self.cases=cases
        self.x=np.asarray(x,float).copy();self.x.setflags(write=False);self.lookups=0
    def sample(self,case_index,phase,guard=.1):
        """Two lanes at requested centers; no outside-support interpolation."""
        self.lookups+=1
        positions=np.asarray(phase)[:,None]+np.linspace(-guard,guard,9)[None,:]
        if not np.isfinite(positions).all() or positions.min()<self.phase[0] or positions.max()>self.phase[-1]:
            raise ValueError('Requested sampling times outside characterized support')
        loc=(positions-self.phase[0])/(self.phase[1]-self.phase[0]);lo=np.floor(loc).astype(int)
        lo=np.minimum(lo,len(self.phase)-2);w=loc-lo;margins=[];errors=[]
        for lane in range(2):
            curve=self.voltage[case_index,lane]
            v=curve[:,lo[lane]]*(1-w[lane])+curve[:,lo[lane]+1]*w[lane]
            signed=(2*self.bits[case_index,lane].astype(float)[:,None]-1)*(v-.5)
            margins.append(float(signed.min()));errors.append(int(np.sum((v[:,4]>=.5)!=self.bits[case_index,lane])))
        return np.asarray(margins),np.asarray(errors)
    def save(self,path):
        np.savez_compressed(path,phase=self.phase,voltage=self.voltage,bits=self.bits,energy=self.energy,
            overshoot=self.overshoot,x=self.x,cases_json=json.dumps([asdict(c) for c in self.cases]))
    @classmethod
    def load(cls,path,expected_hash):
        if file_hash(path)!=expected_hash:raise ValueError('Phase bank checksum mismatch')
        with np.load(path) as z:
            return cls(z['phase'],z['voltage'],z['bits'],z['energy'],z['overshoot'],
                       [Case.from_dict(c) for c in json.loads(str(z['cases_json']))],z['x'])


def characterize(x,cases,output,workers=2):
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite or silently resume characterization')
    evaluator=Evaluator(cases,output,workers);result=evaluator.evaluate(x)
    if not result['valid']:raise ValueError('Invalid transistor characterization; see receipts')
    phase=np.linspace(-.5,1.5,513);voltage=[];bits=[];energies=[];overshoot=[];waves=[]
    for j,case in enumerate(cases):
        receipt=json.loads((output/'receipts'/f'q00001_c{j:02d}.json').read_text())
        folder=output/'cache'/receipt['fingerprint'];measurement=folder/'measurement.json'
        if file_hash(measurement)!=receipt['measurement_sha256']:raise ValueError('Characterization receipt mismatch')
        with np.load(folder/'wave.npz') as z:wave={k:z[k] for k in z.files}
        curves=[]
        for lane,c in enumerate([case.context.lane0,case.context.lane1]):
            times=(np.arange(16,16+case.bits)[:,None]+c.coarse_latency_ui+phase[None,:]+c.clock_offset_ui)*c.ui
            if times.min()<wave['t'][0] or times.max()>wave['t'][-1]:raise ValueError('Incomplete characterized waveform')
            curves.append(np.interp(times.ravel(),wave['t'],wave['rx'][:,lane]).reshape(times.shape)/c.vdd)
        voltage.append(curves);bits.append(case.payload());energies.append(result['cases'][j]['energy_pj_bit'])
        overshoot.append(result['cases'][j]['overshoot_fraction']);waves.append(file_hash(folder/'wave.npz'))
    bank=PhaseBank(phase,np.asarray(voltage,np.float32),bits,energies,overshoot,cases,x)
    deltas=[]
    for j,c in enumerate(cases):
        m,e=bank.sample(j,[c.phase_ui]*2,c.guard_ui)
        deltas.append(float(np.max(np.abs(m-result['cases'][j]['margins']))))
        if deltas[-1]>.002 or e.tolist()!=result['cases'][j]['errors']:raise ValueError('Phase-grid numerical check failed')
    bank.lookups=0;bank.save(output/'bank.npz')
    write_json(output/'receipt.json',dict(bank_sha256=file_hash(output/'bank.npz'),x=np.asarray(x).tolist(),
        static_result=result,accounting=evaluator.accounting(),max_margin_grid_error=max(deltas),
        waveform_hashes=waves,scope='Fixed waveforms; receiver sampling-time displacement only'))
    return bank,result,evaluator.accounting()
