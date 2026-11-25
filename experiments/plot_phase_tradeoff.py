"""Plot paired robustness and TX energy using the two compact committed result files."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    folder=args.results
    base=json.loads((folder/'SIZING_KO.json').read_text())
    extra=json.loads((folder/'ALL_512_KO.json').read_text())
    rows=[*base['rows'],*extra['rows']]
    members=extra['protocol']['members']
    names=['BO','Local MLP','Bracket','Reserve slot']
    methods=['bo','mlp_local','bracket','reserve']
    colors=['#294766','#257d79','#aa5235','#6b55a7']
    fig,axes=plt.subplots(1,2,figsize=(9,5.5))
    y=np.arange(12,dtype=float)+np.repeat([0,.65,1.3],4)
    for j,(name,method,color) in enumerate(zip(names,methods,colors)):
        for i,seed in enumerate((21,22,23)):
            vals=[r['methods']['grid'] for r in rows if r['design']==members[f'{method}_s{seed}']]
            count=sum(s['all_pass'] for s in vals)
            energy=np.mean([s['energy'] for s in vals])
            pos=y[i*4+j]
            for ax,value,label,pad in zip(axes,[count,energy],[str(count),f'{energy:.3f}'],[.25,.004]):
                ax.barh(pos,value,.66,color=color)
                ax.text(value+pad,pos,label,fontsize=8,va='center',color='#243346')
    for k,ax in enumerate(axes):
        ax.set_ylim(y[-1]+.8,-.9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#bdc4cd')
        ax.set_yticks(y,([f'Seed {seed}    {name}' if j==0 else name for seed in (21,22,23)
                            for j,name in enumerate(names)] if k==0 else ['']*12),fontsize=8)
        ax.tick_params(axis='y',length=0,pad=6)
        ax.tick_params(axis='x',length=0,labelsize=8,color='#667687')
        ax.grid(axis='x',color='#e0e4ea',linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_xlim(0,17.8)
    axes[0].set_xticks([0,4,8,12,16])
    axes[0].set_title('Passing channels out of 16',fontsize=12,loc='left')
    axes[1].set_xlim(0,.39)
    axes[1].set_xticks([0,.1,.2,.3])
    axes[1].set_title('Transmitter energy (pJ/bit)',fontsize=12,loc='left')
    fig.subplots_adjust(top=.85,bottom=.12,left=.16,right=.98,wspace=.12)
    fig.text(.16,.95,'512-bit transfer on unseen RC and RLC channels',fontsize=15,weight='bold',color='#18334c')
    fig.text(.16,.035,'Same 16 channels, three-probe grid calibration. Energy includes passing and failing channels.',fontsize=8,color='#47596b')
    fig.savefig(folder/'TRADEOFF_512.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
