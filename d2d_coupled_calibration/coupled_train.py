"""Vanilla TD3+BC or BC on the frozen coupled-lane offline dataset."""
import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from coupled_experiment import OUT,config,Environment

torch.set_num_threads(1)

class Actor(nn.Module):
    def __init__(self):
        super().__init__();w=config()['width']
        self.net=nn.Sequential(nn.Linear(50,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,4),nn.Tanh())
    def forward(self,x):return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__();w=config()['width']
        def net():return nn.Sequential(nn.Linear(54,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,1))
        self.q1,self.q2=net(),net()
    def forward(self,s,a):
        sa=torch.cat([s,a],-1);return self.q1(sa),self.q2(sa)

def label(method,seed,alpha=0):return f'{method}_s{seed}'+(f'_a{alpha:g}' if method=='td3bc' else '')

def train(method,seed,alpha):
    cfg=config();folder=OUT/'models'/label(method,seed,alpha);folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed)
    with np.load(OUT/'dataset.npz') as d:
        obs=d['observations'];mean=obs.mean(0);std=obs.std(0)+.001
        arrays=[(obs-mean)/std,d['actions'],(d['next_observations']-mean)/std,d['rewards'][:,None],
                (~(d['terminals']|d['timeouts'])).astype(np.float32)[:,None]]
    data=[torch.tensor(a,dtype=torch.float32) for a in arrays]
    actor=Actor();critic=Critic();at=copy.deepcopy(actor);ct=copy.deepcopy(critic)
    oa=torch.optim.Adam(actor.parameters(),lr=cfg['lr']);oc=torch.optim.Adam(critic.parameters(),lr=cfg['lr'])
    logs=[];start=time.perf_counter();al=torch.tensor(0.)
    for step in range(1,max(cfg['checkpoint_steps'])+1):
        ix=torch.randint(len(data[0]),(cfg['batch_size'],));s,a,ns,r,mask=[t[ix] for t in data]
        if method=='bc':
            al=F.mse_loss(actor(s),a);oa.zero_grad();al.backward();oa.step();cl=torch.tensor(0.)
        else:
            with torch.no_grad():
                noise=(torch.randn_like(a)*cfg['policy_noise']).clamp(-cfg['noise_clip'],cfg['noise_clip'])
                na=(at(ns)+noise).clamp(-1,1);q1,q2=ct(ns,na)
                target=r+cfg['discount']*mask*torch.minimum(q1,q2)
            q1,q2=critic(s,a);cl=F.mse_loss(q1,target)+F.mse_loss(q2,target)
            oc.zero_grad();cl.backward();oc.step()
            if step%cfg['policy_frequency']==0:
                for p in critic.parameters():p.requires_grad_(False)
                pi=actor(s);q=critic.q1(torch.cat([s,pi],-1));weight=alpha/q.abs().mean().detach().clamp_min(1e-6)
                al=-weight*q.mean()+F.mse_loss(pi,a);oa.zero_grad();al.backward();oa.step()
                for p in critic.parameters():p.requires_grad_(True)
                with torch.no_grad():
                    for p,t in zip(actor.parameters(),at.parameters()):t.lerp_(p,cfg['tau'])
                    for p,t in zip(critic.parameters(),ct.parameters()):t.lerp_(p,cfg['tau'])
        if step%1000==0:
            if not torch.isfinite(al) or not torch.isfinite(cl):raise ValueError('Nonfinite training loss')
            logs.append({'step':step,'actor_loss':float(al.detach()),'critic_loss':float(cl.detach()),'seconds':time.perf_counter()-start})
            print(label(method,seed,alpha),logs[-1],flush=True)
        if step in cfg['checkpoint_steps']:
            checkpoint={'actor':actor.state_dict(),'critic':critic.state_dict(),'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                        'method':method,'seed':seed,'alpha':alpha,'step':step}
            buffer=io.BytesIO();torch.save(checkpoint,buffer)
            tmp=folder/f'step{step}.tmp.pt';tmp.write_bytes(buffer.getvalue());tmp.replace(folder/f'step{step}.pt')
    (folder/'training.json').write_text(json.dumps({'logs':logs,'seconds':time.perf_counter()-start,
        'dataset_sha256':hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
        'spice_calls_during_learning':0,'torch_version':torch.__version__},indent=2)+'\n')

class Policy:
    def __init__(self,path):
        p=torch.load(path,map_location='cpu',weights_only=True);self.actor=Actor();self.actor.load_state_dict(p['actor']);self.actor.eval()
        self.mean=p['mean'];self.std=p['std']
    def __call__(self,obs):
        with torch.no_grad():return self.actor((torch.as_tensor(obs,dtype=torch.float32)-self.mean)/self.std).numpy()

def all_runs():
    cfg=config();jobs=[('bc',s,0.) for s in cfg['train_seeds']]+[('td3bc',s,a) for a in cfg['alphas'] for s in cfg['train_seeds']]
    def one(job):
        m,s,a=job;name=label(m,s,a);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (OUT/f'{name}.log').open('w') as f:
            subprocess.run([sys.executable,__file__,'--method',m,'--seed',str(s),'--alpha',str(a)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('Completed',name,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(one,jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--method',choices=['bc','td3bc'],default='td3bc')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--alpha',type=float,default=2.5);a=p.parse_args()
    if a.all:all_runs()
    else:train(a.method,a.seed,a.alpha)
