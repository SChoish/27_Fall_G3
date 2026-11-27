"""Finite-packet, matched-port 20-lane traffic calibration experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SAMPLES = 16
BITS = 256
PHASE = np.arange(2,15,2)
FAMILIES = ['independent','aligned','opposing','lower_half','upper_half']
LANES = np.arange(20)
DEV, TEST = 12, 24


def traffic(seed, family):
    rng = np.random.default_rng(seed)
    base = rng.choice([-1,1],(20,BITS))
    bus = rng.choice([-1,1],BITS)
    error = np.where(rng.random((20,BITS))<.1,-1,1)
    sign = np.where(LANES%2,-1,1)
    if family=='independent':
        return base
    group = LANES
    if family=='lower_half': group=LANES[:10]
    if family=='upper_half': group=LANES[10:]
    polarity = np.ones(20) if family=='aligned' else sign
    base[group]=(polarity[group,None]*bus*error[group]).astype(int)
    return base


class Channel:
    def __init__(self,h,rate,xtalk):
        # Common coarse delay from median low-frequency main-path group delay.
        main=h[LANES,LANES]
        delay=-np.angle(main[:,1]/main[:,0])/(2*np.pi)
        self.coarse=round(np.median(delay)*rate*SAMPLES)
        freq=np.fft.rfftfreq(BITS*SAMPLES,d=1/(rate*SAMPLES))
        # Taper the last 10 GHz to zero rather than extrapolate above 60 GHz.
        window=np.where(freq<=50,1,(1+np.cos(np.pi*np.clip((freq-50)/10,0,1)))/2)
        edges=np.any(np.abs(h)>0,axis=-1)
        if not xtalk:edges &= np.eye(20,dtype=bool)
        # Convert the sampled rectangle's Fourier series to the continuous held NRZ pulse.
        # This corrects the half-sample phase and sinc envelope.
        relative=freq/(rate*SAMPLES)
        hold=np.exp(-1j*np.pi*relative)*np.sinc(relative)
        self.rx,self.tx=np.where(edges)
        self.response=np.array([np.interp(freq,np.arange(61),h[i,j])*window*hold
                                for i,j in zip(self.rx,self.tx)])
        self.points=((np.arange(BITS)[:,None]*SAMPLES+self.coarse+np.arange(1,16))%(BITS*SAMPLES)).ravel()
        self.incoming=[np.flatnonzero(self.rx==i) for i in LANES]

    def scores(self,bits):
        # Ideal matched incident NRZ, ±0.5, normalized to one full launched swing.
        x=np.fft.rfft(np.repeat(bits*.5,SAMPLES,axis=1))
        y=np.fft.irfft(x[self.tx]*self.response,n=BITS*SAMPLES)[:,self.points]
        table=np.empty((21,20,len(PHASE)))
        def margin(samples,signs):
            v=(samples.reshape(BITS,15)*signs[:,None]).min(axis=0)
            return np.minimum.reduce([v[PHASE-2],v[PHASE-1],v[PHASE]])
        for i,edges in enumerate(self.incoming):
            total=y[edges].sum(axis=0)
            table[:,i]=margin(total,bits[i])
            for edge in edges:
                # A single selected TX may reduce its matched incident amplitude by 12.5%.
                table[1+self.tx[edge],i]=margin(total-.125*y[edge],bits[i])
        return table


def select(score):
    # score[action, lane, phase], with ties favoring no backoff and centreward phases
    order=np.argsort(abs(PHASE-8),kind='stable')
    phase=order[np.argmax(score[:,:,order],axis=2)]
    quality=score[np.arange(len(score))[:,None],LANES[None,:],phase].min(axis=1)
    a=np.argmax(quality)
    return a,phase[a]


def evaluate(score,action):
    a,phases=action
    v=score[a,LANES,phases]
    return float(v.min()),int(v.argmin())


def classify(score, fixed, per_family):
    oracle=select(score)
    pos,bottleneck=evaluate(score,fixed)
    oracle_margin,_=evaluate(score,oracle)
    by_family,_=evaluate(score,per_family)
    phase_only=select(score[:1])
    fixed_drive_phase=select(score[fixed[0]:fixed[0]+1])
    fixed_drive_opt,_=evaluate(score,(fixed[0],fixed_drive_phase[1]))
    return dict(fixed=pos, bottleneck=bottleneck, family=by_family,
                oracle=oracle_margin, phase_only=evaluate(score,phase_only)[0],
                fixed_drive_opt=fixed_drive_opt, oracle_tx=int(oracle[0])-1,
                gap=float(np.diff(np.sort(score[fixed[0],LANES,fixed[1]])[:2])[0]),
                **{'phase_'+str(i):int(PHASE[k]) for i,k in enumerate(oracle[1])})


def campaign(data,out):
    rows=[]; pairs=[]; rules=[]
    for layout in range(3):
        h=np.load(data/f'layout{layout}.npz')['h']
        for rate in [16,32]:
            for xt in [True,False]:
                channel=Channel(h,rate,xt)
                groups={}
                for f in FAMILIES:
                    groups[f]=np.minimum.reduce([channel.scores(traffic(11000+s,f)) for s in range(DEV)])
                training=np.minimum.reduce(list(groups.values()))
                all_phase=training[0].min(axis=0)
                common_phase=int(np.argmax(all_phase))
                # Truly uniform source and phase (no selected lane for backoff).
                global_action=(0,np.full(20,common_phase))
                fixed=select(training)
                settings={f:select(groups[f]) for f in FAMILIES}
                rules.append(dict(layout=layout,rate=rate,xt=xt,global_phase=int(PHASE[common_phase]),
                                  fixed_tx=int(fixed[0])-1,fixed_phase=PHASE[fixed[1]].tolist(),
                                  family_tx={f:int(a[0])-1 for f,a in settings.items()}))
                for s in range(TEST):
                    all_scores=[channel.scores(traffic(27000+s,f)) for f in FAMILIES]
                    actions=[select(q) for q in all_scores]
                    packet_rows=[]
                    for f,score in zip(FAMILIES,all_scores):
                        r=classify(score,fixed,settings[f])
                        r.update(layout=layout,rate=rate,xt=xt,seed=s,traffic=f,
                                 uniform=evaluate(score,global_action)[0])
                        packet_rows.append(r); rows.append(r)
                    for i in range(len(FAMILIES)):
                        for j in range(i+1,len(FAMILIES)):
                            lost_i=packet_rows[i]['oracle']-evaluate(all_scores[i],actions[j])[0]
                            lost_j=packet_rows[j]['oracle']-evaluate(all_scores[j],actions[i])[0]
                            common=np.minimum(all_scores[i]-packet_rows[i]['oracle'],
                                              all_scores[j]-packet_rows[j]['oracle'])
                            pair_regret=max(0.,-evaluate(common,select(common))[0])
                            pairs.append(dict(shared_regret=pair_regret,layout=layout,rate=rate,xt=xt,seed=s,
                                first=FAMILIES[i],second=FAMILIES[j],
                                changed_lane=packet_rows[i]['bottleneck']!=packet_rows[j]['bottleneck'],
                                loss_first=lost_i,loss_second=lost_j))
                print('done',layout,rate,xt,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    for name,records in [('packets',rows),('switches',pairs)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=records[0],lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'fixed_rules.json').write_text(json.dumps(rules,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('channels',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--phase-step',type=int,choices=(1,2),default=2,help='Sample intervals between phase codes (1/16 UI each)')
    a=p.parse_args()
    PHASE = np.arange(2,15,a.phase_step)
    campaign(a.channels,a.out)
