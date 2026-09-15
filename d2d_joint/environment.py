"""Partially observed persistent phase control; private physical diagnostics."""
from dataclasses import dataclass,asdict
from itertools import product
import numpy as np
from d2d_sizing.circuit import encode

ACTIONS=np.array(list(product([-4,0,4],[-4,0,4],[0,1])),dtype=int)
FRAME_DIM=24


@dataclass(frozen=True)
class Config:
    horizon:int=64
    history:int=4
    drift_bound:float=.32
    drift_velocity_rho:float=.96
    drift_velocity_sigma:float=.004
    margin_min:float=.08
    overshoot_max:float=.2
    violation_limit:float=.05
    downtime_limit:float=.25
    dither_ui:float=.0625
    monitor_pj_per_comparison:float=.002
    phase_update_pj_per_lane:float=.1
    phase_update_ui:float=2.
    reward_energy_reference_pj_bit:float=1.
    quality_weight:float=1.


def drift_path(seed,cfg):
    rng=np.random.default_rng(seed);d=rng.uniform(-.08,.08,2);v=rng.uniform(-.008,.008,2)
    rows=[]
    for t in range(cfg.horizon):
        rows.append(d.copy());v=cfg.drift_velocity_rho*v+rng.normal(0,cfg.drift_velocity_sigma,2)
        nxt=d+v;hit=np.abs(nxt)>cfg.drift_bound
        d=np.clip(nxt,-cfg.drift_bound,cfg.drift_bound);v[hit]*=-.5
    return np.array(rows)


def action_id(delta,measure):
    matches=np.flatnonzero(np.all(ACTIONS==[*delta,int(measure)],axis=1))
    if len(matches)!=1:raise ValueError('Invalid discrete control action')
    return int(matches[0])


class Environment:
    def __init__(self,bank,trajectory,environment_id=0,cfg=Config()):
        self.bank=bank;self.cfg=cfg;self._drift=np.asarray(trajectory,float).copy()
        if self._drift.shape!=(cfg.horizon,2) or np.max(np.abs(self._drift))>cfg.drift_bound+1e-12:raise ValueError('Invalid drift trajectory')
        self._environment_id=environment_id;self.x=bank.x.copy();self.x.setflags(write=False)
        self.t=0;self.phase=np.array([32,32]);self.last_phase=np.array([0,0]);self.last_measurement=np.zeros((2,3))
        self.last_measurement_t=None;self.previous_action=np.zeros(3);self.frames=[];self.records=[]
        self._append_frame()
    def _frame(self):
        exists=self.last_measurement_t is not None
        age=self.t-1-self.last_measurement_t if exists else self.cfg.horizon
        matches=(self.phase==self.last_phase).astype(float) if exists else np.zeros(2)
        return np.r_[encode(self.x),self.phase/64,self.last_phase/64,self.last_measurement.ravel(),
            min(age,self.cfg.horizon)/self.cfg.horizon,float(exists),matches,self.t/self.cfg.horizon,self.previous_action].astype(np.float32)
    def _append_frame(self):
        f=self._frame();assert len(f)==FRAME_DIM
        self.frames.append(f);self.frames=self.frames[-self.cfg.history:]
    def observation(self):
        # Neither reward nor info, actual drift, packet labels or full curves are included.
        return np.concatenate([np.zeros(FRAME_DIM,np.float32)]*(self.cfg.history-len(self.frames))+self.frames).copy()
    def step(self,action):
        if self.t>=self.cfg.horizon:raise RuntimeError('Episode already ended')
        if not isinstance(action,(int,np.integer)) or not 0<=action<len(ACTIONS):raise ValueError('Invalid action index')
        request=ACTIONS[action];old=self.phase.copy();self.phase=np.clip(self.phase+request[:2],0,64)
        changed=int(np.sum(self.phase!=old));measure=bool(request[2])
        case_index=self._environment_id*2+self.t%2;case=self.bank.cases[case_index]
        effective=self.phase/64-self._drift[self.t]
        margins,errors=self.bank.sample(case_index,effective)
        quality=bool(margins.min()>=self.cfg.margin_min and errors.sum()==0 and self.bank.overshoot[case_index]<=self.cfg.overshoot_max)
        bits=case.bits;tx=float(self.bank.energy[case_index]*2*bits)
        monitor_energy=pilot_tx=0.;measurement_time=0.
        if measure:
            measured=np.stack([self.bank.sample(case_index,effective+offset)[0] for offset in [-self.cfg.dither_ui,0,self.cfg.dither_ui]],axis=1)
            self.last_measurement=measured;self.last_phase=self.phase.copy();self.last_measurement_t=self.t
            # One known pilot packet; 3 phase centers x 9 aperture probes in parallel on both lanes.
            pilot_tx=tx;monitor_energy=2*bits*3*9*self.cfg.monitor_pj_per_comparison
            measurement_time=bits*case.context.lane0.ui
        phase_energy=changed*self.cfg.phase_update_pj_per_lane
        update_time=self.cfg.phase_update_ui*case.context.lane0.ui if changed else 0.
        data_time=bits*case.context.lane0.ui
        deficit=float(np.maximum(0,(self.cfg.margin_min-margins)/self.cfg.margin_min).clip(0,1).mean())
        quality_signal=max(deficit,float(errors.sum()/(2*bits)),float(self.bank.overshoot[case_index]>self.cfg.overshoot_max))
        total=tx+pilot_tx+monitor_energy+phase_energy
        reward=-total/(2*bits*self.cfg.reward_energy_reference_pj_bit)-self.cfg.quality_weight*quality_signal
        record=dict(block=self.t,phase_codes=self.phase.tolist(),effective_phase=effective.tolist(),drift_ui=self._drift[self.t].tolist(),
            requested_action=int(action),measured=measure,phase_changes=changed,margins=margins.tolist(),errors=errors.tolist(),
            quality=quality,tx_energy_pj=tx,pilot_tx_energy_pj=pilot_tx,monitor_energy_pj=monitor_energy,
            phase_update_energy_pj=phase_energy,total_energy_pj=total,data_seconds=data_time,
            interruption_seconds=measurement_time+update_time,delivered_bits=2*bits if quality else 0,
            reward=float(reward),quality_shaping=quality_signal,case_index=case_index)
        self.records.append(record);self.previous_action=np.r_[(self.phase-old)/4,float(measure)]
        self.t+=1;self._append_frame()
        return self.observation(),float(reward),self.t==self.cfg.horizon,record.copy()


def metrics(records,cfg=Config()):
    failures=[not r['quality'] for r in records];run=longest=0
    for bad in failures:run=run+1 if bad else 0;longest=max(longest,run)
    data=sum(r['data_seconds'] for r in records);stop=sum(r['interruption_seconds'] for r in records)
    keys=['tx_energy_pj','pilot_tx_energy_pj','monitor_energy_pj','phase_update_energy_pj','total_energy_pj','delivered_bits']
    out={k:sum(r[k] for r in records) for k in keys}
    out.update(blocks=len(records),violation_rate=float(np.mean(failures)),longest_failure_run=longest,
        calibration_measurements=sum(r['measured'] for r in records),phase_changes=sum(r['phase_changes'] for r in records),
        interruption_seconds=stop,downtime_fraction=stop/(data+stop),elapsed_seconds=data+stop,
        effective_throughput_bps=out['delivered_bits']/(data+stop))
    out['feasible']=out['violation_rate']<=cfg.violation_limit and out['downtime_fraction']<=cfg.downtime_limit
    return out
