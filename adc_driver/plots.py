from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def waveform(folder,output=None):
    folder=Path(folder);data=np.load(folder/'waveform.npz')
    metrics=json.loads((folder/'metrics.json').read_text());c=metrics['context']
    t=data['time']*1e6;begin=c['warmup_us']+c['dummy_cycles']/c['fs_hz']*1e6
    mask=t>=begin-.15
    fig,axes=plt.subplots(3,1,figsize=(10,7),gridspec_kw={'height_ratios':[2,1,1]},sharex=True)
    for name,label,color,style in [('vin','Input','#777777','--'),('vout','Amplifier output','#233d61','-'),
                                   ('vsample','Stored sample','#873447','-')]:
        axes[0].plot(t[mask],data[name][mask],style,label=label,color=color,lw=1.25)
    for row in metrics['cycles']:
        axes[0].plot(row['sample_time_s']*1e6,row['stored_v'],'o',color='#873447',ms=3)
    axes[0].set(ylabel='Voltage (V)',title='MOS sampling circuit: measured transistor-level transient')
    axes[0].legend(frameon=False,ncol=3,fontsize=9)
    axes[1].plot(t[mask],data['clkn'][mask],color='#233d61',lw=1,label='NMOS gate')
    axes[1].plot(t[mask],data['clkp'][mask],color='#873447',lw=1,label='PMOS gate')
    axes[1].set(ylabel='TG clocks (V)');axes[1].legend(frameon=False,ncol=2,fontsize=8)
    for row in metrics['cycles']:
        start=row['sample_time_s']*1e6;end=start-c['read_delay_ns']*.001+1e6/c['fs_hz']-c['acquisition_ns']*.001-.01
        select=(t>=start)&(t<=end)
        axes[2].plot(t[select],(data['vsample'][select]-row['target_v'])*1000,color='#873447')
    for sign in [-1,1]: axes[2].axhline(sign*c['error_limit_mv'],color='#777777',ls='--',lw=1)
    axes[2].set(ylabel='Hold error (mV)',xlabel='Time (us)')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output or folder/'waveform.png',dpi=170);plt.close(fig)


def searches(paths,output):
    colors=['#233d61','#873447','#757575','#5685a1'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for path,color in zip(paths,colors):
        r=json.loads(Path(path).read_text());rows=r['trace'];n=np.arange(1,len(rows)+1)
        energy=[];best=np.inf
        for row in rows:
            if row['oracle']['feasible']:best=min(best,row['oracle']['mean_driver_energy_pj'])
            energy.append(best if np.isfinite(best) else np.nan)
        axes[0].plot(n,[row['best_cost'] for row in rows],label=r['method'],color=color)
        axes[1].plot(n,energy,label=r['method'],color=color)
    axes[0].set(ylabel='Best measured constrained cost')
    axes[1].set(ylabel='Best feasible energy (pJ/sample)')
    for ax in axes:
        ax.set(xlabel='New design evaluations');ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle('Execution example; one seed, no algorithm ranking claim')
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)
