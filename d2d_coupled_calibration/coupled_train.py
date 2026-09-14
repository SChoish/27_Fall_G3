"""Vanilla TD3+BC, BC, prototype IQL, or online SAC on the coupled-lane task."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from coupled_experiment import OUT,config,Environment,Table
from coupled_link import CoupledSetting

from provenance import verify_artifact, write_metadata

torch.set_num_threads(1)

class Actor(nn.Module):
    def __init__(self):
        super().__init__();w=config()['width']
        self.net=nn.Sequential(nn.Linear(50,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,4),nn.Tanh())
    def forward(self,x):return self.net(x)

class GaussianActor(nn.Module):
    """Squashed Gaussian policy used by online SAC; eval uses tanh(mean)."""
    def __init__(self):
        super().__init__();w=config()['width']
        self.trunk=nn.Sequential(nn.Linear(50,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU())
        self.mean=nn.Linear(w,4);self.log_std=nn.Linear(w,4)
    def forward(self,x):
        h=self.trunk(x);return self.mean(h),self.log_std(h).clamp(-20,2)
    def sample(self,x):
        mean,log_std=self.forward(x);normal=torch.distributions.Normal(mean,log_std.exp());u=normal.rsample()
        action=torch.tanh(u);correction=2*(math.log(2)-u-F.softplus(-2*u))
        return action,(normal.log_prob(u)-correction).sum(-1,keepdim=True)
    def mean_action(self,x):
        mean,_=self.forward(x);return torch.tanh(mean)

class Critic(nn.Module):
    def __init__(self):
        super().__init__();w=config()['width']
        def net():return nn.Sequential(nn.Linear(54,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,1))
        self.q1,self.q2=net(),net()
    def forward(self,s,a):
        sa=torch.cat([s,a],-1);return self.q1(sa),self.q2(sa)

class Value(nn.Module):
    def __init__(self):
        super().__init__();w=config()['width']
        self.net=nn.Sequential(nn.Linear(50,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,1))
    def forward(self,x):return self.net(x)

def expectile_loss(diff,expectile):
    if not np.isfinite(expectile) or not 0<expectile<1:raise ValueError('expectile must lie in (0,1)')
    weight=torch.where(diff>0,expectile,1-expectile)
    return (weight*diff.square()).mean()

def label(method,seed,alpha=0,expectile=.7,temperature=3.):
    if method=='td3bc':return f'{method}_s{seed}_a{alpha:g}'
    if method=='iql':return f'{method}_s{seed}_t{expectile:g}_b{temperature:g}'
    if method=='sac':return f'{method}_s{seed}'
    return f'{method}_s{seed}'

def _checkpoint_steps(cfg,checkpoint_steps):
    steps=list(cfg['checkpoint_steps'] if checkpoint_steps is None else checkpoint_steps)
    if not steps or any(type(n) is not int or n < 1 for n in steps) or steps!=sorted(set(steps)):
        raise ValueError('Checkpoint steps must be strictly increasing positive integers')
    return steps

def train(method,seed,alpha,reward_spec=None,checkpoint_steps=None,expectile=.7,temperature=3.):
    cfg=config();steps=_checkpoint_steps(cfg,checkpoint_steps)
    if reward_spec is not None:
        if method not in ('td3bc','iql','sac'):raise ValueError('Reward ablations apply to TD3+BC, IQL and SAC; BC does not consume reward')
        if reward_spec.gamma!=cfg['discount'] or reward_spec.margin!=cfg['margin'] or reward_spec.energy_ratio!=cfg['primary_ratio']:
            raise ValueError('Reward spec must match configured discount and success thresholds')
        if reward_spec.budget>cfg['horizon']:raise ValueError('Reward budget exceeds source horizon')
    if method=='sac':
        _train_sac(seed,reward_spec,steps);return
    if method=='iql':
        _train_iql(seed,reward_spec,steps,expectile,temperature);return
    name=label(method,seed,alpha)+(reward_spec.suffix() if reward_spec is not None else '')
    folder=OUT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed)
    verify_artifact(OUT/'dataset.npz',cfg,dependencies=[OUT/'train_table.npz'])
    with np.load(OUT/'dataset.npz') as source:d={k:source[k] for k in source.files}
    reward_audit=None
    if reward_spec is not None:
        from reward_design import relabel
        d,reward_audit=relabel(d,reward_spec)
        (folder/'reward_audit.json').write_text(json.dumps(reward_audit,indent=2)+'\n')
    obs=d['observations'];mean=obs.mean(0);std=obs.std(0)+.001
    arrays=[(obs-mean)/std,d['actions'],(d['next_observations']-mean)/std,d['rewards'][:,None],
            (~(d['terminals']|d['timeouts'])).astype(np.float32)[:,None]]
    data=[torch.tensor(a,dtype=torch.float32) for a in arrays]
    actor=Actor();critic=Critic();at=copy.deepcopy(actor);ct=copy.deepcopy(critic)
    oa=torch.optim.Adam(actor.parameters(),lr=cfg['lr']);oc=torch.optim.Adam(critic.parameters(),lr=cfg['lr'])
    logs=[];start=time.perf_counter();al=torch.tensor(0.);q_abs=0.;q_weight=0.;bc_loss=0.;q_term_value=0.;q_grad_norm=0.;bc_grad_norm=0.
    for step in range(1,max(steps)+1):
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
                bc=F.mse_loss(pi,a);q_term=-weight*q.mean();al=q_term+bc
                if step%1000==0:
                    parameters=tuple(actor.parameters())
                    q_grads=torch.autograd.grad(q_term,parameters,retain_graph=True)
                    bc_grads=torch.autograd.grad(bc,parameters,retain_graph=True)
                    q_grad_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in q_grads)))
                    bc_grad_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in bc_grads)))
                q_term_value=float(q_term.detach());oa.zero_grad();al.backward();oa.step()
                q_abs=float(q.detach().abs().mean());q_weight=float(weight.detach());bc_loss=float(bc.detach())
                for p in critic.parameters():p.requires_grad_(True)
                with torch.no_grad():
                    for p,t in zip(actor.parameters(),at.parameters()):t.lerp_(p,cfg['tau'])
                    for p,t in zip(critic.parameters(),ct.parameters()):t.lerp_(p,cfg['tau'])
        if step%1000==0:
            if not torch.isfinite(al) or not torch.isfinite(cl):raise ValueError('Nonfinite training loss')
            logs.append({'step':step,'actor_loss':float(al.detach()),'critic_loss':float(cl.detach()),'q_abs_mean':q_abs,'q_weight':q_weight,'bc_loss':bc_loss,'actor_q_term':q_term_value,'actor_q_gradient_norm':q_grad_norm,
                         'actor_bc_gradient_norm':bc_grad_norm,'sampled_transition_draws':step*cfg['batch_size'],
                         'mean_draws_per_transition':step*cfg['batch_size']/len(obs),'seconds':time.perf_counter()-start})
            print(name,logs[-1],flush=True)
        if step in steps:
            checkpoint={'actor':actor.state_dict(),'critic':critic.state_dict(),'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                        'method':method,'seed':seed,'alpha':alpha,'step':step,'checkpoint_steps':steps}
            if reward_spec is not None:checkpoint['reward_spec']=reward_audit['spec']
            buffer=io.BytesIO();torch.save(checkpoint,buffer)
            tmp=folder/f'step{step}.tmp.pt';tmp.write_bytes(buffer.getvalue());tmp.replace(folder/f'step{step}.pt')
            dependencies=[OUT/'dataset.npz']
            if reward_spec is not None:dependencies += [Path(__file__),Path(__file__).with_name('reward_design.py')]
            write_metadata(folder/f'step{step}.pt',cfg,dependencies=dependencies)
    (folder/'training.json').write_text(json.dumps({'logs':logs,'seconds':time.perf_counter()-start,
        'dataset_sha256':hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
        'spice_calls_during_learning':0,'training_transitions':len(obs),'batch_size':cfg['batch_size'],
        'training_steps':max(steps),'checkpoint_steps':steps,
        'critic_optimizer_steps':max(steps) if method=='td3bc' else 0,
        'actor_optimizer_steps':max(steps)//cfg['policy_frequency'] if method=='td3bc' else max(steps),
        'sampled_transition_draws':max(steps)*cfg['batch_size'],
        'mean_draws_per_transition':max(steps)*cfg['batch_size']/len(obs),'torch_version':torch.__version__,'reward_audit':reward_audit},indent=2)+'\n')

def _train_iql(seed,reward_spec,steps,expectile,temperature):
    """Prototype IQL: expectile V on data actions, Q bootstrap through V, AWR actor.

    Kostrikov, Nair, Levine, Implicit Q-Learning (2021). The deterministic actor
    uses advantage-weighted MSE in place of a Gaussian log-probability. Expectile
    0.7 and temperature 3 are the paper's default locomotion pair, frozen before
    validation. The critic never evaluates actor actions.
    """
    if not np.isfinite(expectile) or not 0<expectile<1:raise ValueError('expectile must lie in (0,1)')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be positive')
    cfg=config();name=label('iql',seed,expectile=expectile,temperature=temperature)
    name+=reward_spec.suffix() if reward_spec is not None else ''
    folder=OUT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed)
    verify_artifact(OUT/'dataset.npz',cfg,dependencies=[OUT/'train_table.npz'])
    with np.load(OUT/'dataset.npz') as source:d={k:source[k] for k in source.files}
    reward_audit=None
    if reward_spec is not None:
        from reward_design import relabel
        d,reward_audit=relabel(d,reward_spec)
        (folder/'reward_audit.json').write_text(json.dumps(reward_audit,indent=2)+'\n')
    obs=d['observations'];mean=obs.mean(0);std=obs.std(0)+.001
    arrays=[(obs-mean)/std,d['actions'],(d['next_observations']-mean)/std,d['rewards'][:,None],
            (~(d['terminals']|d['timeouts'])).astype(np.float32)[:,None]]
    data=[torch.tensor(a,dtype=torch.float32) for a in arrays]
    actor=Actor();critic=Critic();value=Value();target_critic=copy.deepcopy(critic)
    oa=torch.optim.Adam(actor.parameters(),lr=cfg['lr'])
    oc=torch.optim.Adam(critic.parameters(),lr=cfg['lr'])
    ov=torch.optim.Adam(value.parameters(),lr=cfg['lr'])
    logs=[];start=time.perf_counter();al=torch.tensor(0.);cl=torch.tensor(0.);vl=torch.tensor(0.)
    q_abs=0.;v_mean=0.;exp_adv_mean=0.;exp_adv_clip_frac=0.;bc_loss=0.
    for step in range(1,max(steps)+1):
        ix=torch.randint(len(data[0]),(cfg['batch_size'],));s,a,ns,r,mask=[t[ix] for t in data]
        with torch.no_grad():
            tq1,tq2=target_critic(s,a);target_q=torch.minimum(tq1,tq2)
        v=value(s);vl=expectile_loss(target_q-v,expectile)
        ov.zero_grad();vl.backward();ov.step()
        with torch.no_grad():
            next_v=value(ns);target=r+cfg['discount']*mask*next_v
        q1,q2=critic(s,a);cl=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        oc.zero_grad();cl.backward();oc.step()
        with torch.no_grad():
            q=torch.minimum(*critic(s,a));adv=q-value(s)
            exp_adv=torch.exp(temperature*adv).clamp(max=100.)
        pi=actor(s);bc=F.mse_loss(pi,a,reduction='none').mean(-1,keepdim=True)
        al=(exp_adv*bc).mean()
        oa.zero_grad();al.backward();oa.step()
        with torch.no_grad():
            for p,t in zip(critic.parameters(),target_critic.parameters()):t.lerp_(p,cfg['tau'])
        if step%1000==0:
            if not torch.isfinite(al) or not torch.isfinite(cl) or not torch.isfinite(vl):
                raise ValueError('Nonfinite training loss')
            q_abs=float(q.detach().abs().mean());v_mean=float(value(s).detach().mean())
            exp_adv_mean=float(exp_adv.mean());exp_adv_clip_frac=float((exp_adv>=100.).float().mean())
            bc_loss=float(bc.mean())
            logs.append({'step':step,'actor_loss':float(al.detach()),'critic_loss':float(cl.detach()),
                         'value_loss':float(vl.detach()),'q_abs_mean':q_abs,'v_mean':v_mean,
                         'exp_adv_mean':exp_adv_mean,'exp_adv_clip_frac':exp_adv_clip_frac,'bc_loss':bc_loss,
                         'sampled_transition_draws':step*cfg['batch_size'],
                         'mean_draws_per_transition':step*cfg['batch_size']/len(obs),
                         'seconds':time.perf_counter()-start})
            print(name,logs[-1],flush=True)
        if step in steps:
            checkpoint={'actor':actor.state_dict(),'critic':critic.state_dict(),'value':value.state_dict(),
                        'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'method':'iql','seed':seed,
                        'expectile':expectile,'temperature':temperature,'step':step,'checkpoint_steps':steps}
            if reward_spec is not None:checkpoint['reward_spec']=reward_audit['spec']
            buffer=io.BytesIO();torch.save(checkpoint,buffer)
            tmp=folder/f'step{step}.tmp.pt';tmp.write_bytes(buffer.getvalue());tmp.replace(folder/f'step{step}.pt')
            dependencies=[OUT/'dataset.npz']
            if reward_spec is not None:dependencies += [Path(__file__),Path(__file__).with_name('reward_design.py')]
            write_metadata(folder/f'step{step}.pt',cfg,dependencies=dependencies)
    (folder/'training.json').write_text(json.dumps({'logs':logs,'seconds':time.perf_counter()-start,
        'dataset_sha256':hashlib.sha256((OUT/'dataset.npz').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
        'spice_calls_during_learning':0,'training_transitions':len(obs),'batch_size':cfg['batch_size'],
        'training_steps':max(steps),'checkpoint_steps':steps,'critic_optimizer_steps':max(steps),
        'value_optimizer_steps':max(steps),'actor_optimizer_steps':max(steps),
        'sampled_transition_draws':max(steps)*cfg['batch_size'],
        'mean_draws_per_transition':max(steps)*cfg['batch_size']/len(obs),'expectile':expectile,
        'temperature':temperature,'bootstrap':'r + gamma * V(next) on data actions only',
        'policy_extraction':'advantage-weighted MSE on dataset actions','torch_version':torch.__version__,
        'reward_audit':reward_audit},indent=2)+'\n')

class RunningNorm:
    def __init__(self,dim):
        self.n=0;self.mean=np.zeros(dim,np.float64);self.m2=np.zeros(dim,np.float64)
    def update(self,x):
        x=np.asarray(x,np.float64);self.n+=1;delta=x-self.mean;self.mean+=delta/self.n;self.m2+=delta*(x-self.mean)
    def stats(self):
        var=self.m2/max(self.n-1,1)
        return self.mean.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32)+.001

def _train_sac(seed,reward_spec,steps):
    """Online SAC from Haarnoja et al., ICML 2018, with automatic entropy temperature.

    Starts with an empty replay buffer. Train-table lookups are the only environment
    interactions; cached reads still count as learning measurements. Evaluation uses
    tanh(mean). No mixed offline dataset, margin auxiliary, or new reward.
    """
    from reward_design import transition_reward,success
    if reward_spec is None or reward_spec.mode!='terminal_success':
        raise ValueError('SAC first run uses terminal_success only; no margin auxiliary or new reward')
    cfg=config();name=label('sac',seed)+reward_spec.suffix()
    folder=OUT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed)
    verify_artifact(OUT/'train_table.npz',cfg)
    table=Table('train');ids=list(table.ids)
    if not ids:raise ValueError('Train table is empty')
    if set(ids)&set(cfg['validation_ids']+cfg['test_ids']):raise ValueError('Train table overlaps held-out ids')
    cap=max(steps);obs_b=np.zeros((cap,50),np.float32);act_b=np.zeros((cap,4),np.float32)
    nxt_b=np.zeros((cap,50),np.float32);rew_b=np.zeros((cap,1),np.float32);mask_b=np.zeros((cap,1),np.float32)
    filled=0;norm=RunningNorm(50)
    actor=GaussianActor();critic=Critic();target_critic=copy.deepcopy(critic)
    oa=torch.optim.Adam(actor.parameters(),lr=cfg['lr']);oc=torch.optim.Adam(critic.parameters(),lr=cfg['lr'])
    log_alpha=torch.zeros(1,requires_grad=True);oalpha=torch.optim.Adam([log_alpha],lr=cfg['lr'])
    target_entropy=-4.;updates=0;env=None;obs=None;episodes=0;success_episodes=0
    success_contexts=[];logs=[];start=time.perf_counter()
    al=torch.tensor(0.);cl=torch.tensor(0.);alpha_loss=torch.tensor(0.)
    q_abs=0.;entropy=0.;alpha_value=1.
    audit=dict(spec=dict(mode=reward_spec.mode,budget=reward_spec.budget,gamma=reward_spec.gamma,
        eta=reward_spec.eta,margin=reward_spec.margin,energy_ratio=reward_spec.energy_ratio),
        offline_dataset=False,reward='terminal * incumbent_success; no potential or margin auxiliary')
    (folder/'reward_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for step in range(1,cap+1):
        if env is None or env.steps>=reward_spec.budget:
            env=Environment(table,ids[int(np.random.randint(len(ids)))]);obs=env.obs();norm.update(obs);episodes+=1
        mean,std=norm.stats()
        with torch.no_grad():
            raw,_=actor.sample(((torch.as_tensor(obs,dtype=torch.float32)-torch.as_tensor(mean))/torch.as_tensor(std)).unsqueeze(0))
        action=CoupledSetting.from_action(raw.squeeze(0).numpy()).action()
        nxt,_,_,_,_=env.step(action);done=env.steps==reward_spec.budget
        reward=float(np.asarray(transition_reward(obs[None],nxt[None],np.array([done]),reward_spec)).reshape(-1)[0])
        norm.update(nxt);obs_b[filled]=obs;act_b[filled]=action;nxt_b[filled]=nxt
        rew_b[filled]=reward;mask_b[filled]=0. if done else 1.;filled+=1
        if done and bool(success(nxt,reward_spec)):
            success_episodes+=1;success_contexts.append(int(env.i))
        obs=nxt
        if filled>=cfg['batch_size']:
            ix=np.random.randint(0,filled,cfg['batch_size']);mean,std=norm.stats()
            s=torch.as_tensor((obs_b[ix]-mean)/std);a=torch.as_tensor(act_b[ix])
            ns=torch.as_tensor((nxt_b[ix]-mean)/std);r=torch.as_tensor(rew_b[ix]);mask=torch.as_tensor(mask_b[ix])
            alpha=log_alpha.exp()
            with torch.no_grad():
                na,nlog=actor.sample(ns);tq1,tq2=target_critic(ns,na)
                target=r+cfg['discount']*mask*(torch.minimum(tq1,tq2)-alpha*nlog)
            q1,q2=critic(s,a);cl=F.mse_loss(q1,target)+F.mse_loss(q2,target)
            oc.zero_grad();cl.backward();oc.step()
            for p in critic.parameters():p.requires_grad_(False)
            pi,logp=actor.sample(s);q=torch.minimum(*critic(s,pi));al=(alpha.detach()*logp-q).mean()
            oa.zero_grad();al.backward();oa.step()
            for p in critic.parameters():p.requires_grad_(True)
            alpha_loss=-(log_alpha*(logp+target_entropy).detach()).mean()
            oalpha.zero_grad();alpha_loss.backward();oalpha.step()
            with torch.no_grad():
                for p,t in zip(critic.parameters(),target_critic.parameters()):t.lerp_(p,cfg['tau'])
            updates+=1;alpha_value=float(alpha.detach());entropy=float((-logp).mean().detach());q_abs=float(q.detach().abs().mean())
        if step%1000==0:
            if not torch.isfinite(al) or not torch.isfinite(cl):raise ValueError('Nonfinite training loss')
            logs.append({'step':step,'environment_steps':step,'gradient_updates':updates,'actor_loss':float(al.detach()),
                         'critic_loss':float(cl.detach()),'alpha':alpha_value,'entropy':entropy,'q_abs_mean':q_abs,
                         'replay':filled,'training_success_episodes':success_episodes,
                         'table_lookups':table.measurements,'seconds':time.perf_counter()-start})
            print(name,logs[-1],flush=True)
        if step in steps:
            mean,std=norm.stats()
            checkpoint={'actor':actor.state_dict(),'critic':critic.state_dict(),'log_alpha':log_alpha.detach().clone(),
                        'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'method':'sac','seed':seed,
                        'step':step,'checkpoint_steps':steps,'environment_steps':step,'gradient_updates':updates,
                        'reward_spec':audit['spec']}
            buffer=io.BytesIO();torch.save(checkpoint,buffer)
            tmp=folder/f'step{step}.tmp.pt';tmp.write_bytes(buffer.getvalue());tmp.replace(folder/f'step{step}.pt')
            write_metadata(folder/f'step{step}.pt',cfg,dependencies=[OUT/'train_table.npz',Path(__file__),
                Path(__file__).with_name('reward_design.py')])
    (folder/'training.json').write_text(json.dumps({'logs':logs,'seconds':time.perf_counter()-start,
        'train_table_sha256':hashlib.sha256((OUT/'train_table.npz').read_bytes()).hexdigest(),
        'config_sha256':hashlib.sha256((OUT/'config.json').read_bytes()).hexdigest(),
        'method':'sac','policy':'squashed Gaussian; eval uses tanh(mean)',
        'offline_dataset':False,'reward_mode':'terminal_success','no_margin_auxiliary':True,
        'environment_steps':cap,'updates_per_env_step':1,'gradient_updates':updates,
        'warmup_env_steps':min(cfg['batch_size']-1,cap),'batch_size':cfg['batch_size'],
        'training_episodes':episodes,'training_success_episodes':success_episodes,
        'training_success_contexts':success_contexts,
        'training_success_context_counts':{str(k):success_contexts.count(k) for k in sorted(set(success_contexts))},
        'reference_queries':episodes,'policy_queries':cap,'table_lookups':table.measurements,
        'cached_lookups_counted':True,'new_spice_calls':0,'replay_transitions':filled,
        'held_out_during_learning':True,'torch_version':torch.__version__,'reward_audit':audit,
        'cost_note':'SAC environment steps are not the same cost as offline gradient updates on a fixed dataset.',
        'paper_reference':'https://proceedings.mlr.press/v80/haarnoja18b.html'},indent=2)+'\n')

class Policy:
    def __init__(self,path):
        p=torch.load(path,map_location='cpu',weights_only=True);self.method=p.get('method')
        if self.method=='sac':
            verify_artifact(path,config(),dependencies=[OUT/'train_table.npz'])
            self.actor=GaussianActor()
        else:
            verify_artifact(path,config(),dependencies=[OUT/'dataset.npz'])
            self.actor=Actor()
        self.actor.load_state_dict(p['actor']);self.actor.eval()
        self.mean=p['mean'];self.std=p['std'];self.reward_spec=p.get('reward_spec')
    def __call__(self,obs):
        with torch.no_grad():
            x=(torch.as_tensor(obs,dtype=torch.float32)-self.mean)/self.std
            if self.method=='sac':return self.actor.mean_action(x).numpy()
            return self.actor(x).numpy()

def all_runs():
    cfg=config();jobs=[('bc',s,0.) for s in cfg['train_seeds']]+[('td3bc',s,a) for a in cfg['alphas'] for s in cfg['train_seeds']]
    def one(job):
        m,s,a=job;name=label(m,s,a);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (OUT/f'{name}.log').open('w') as f:
            subprocess.run([sys.executable,__file__,'--method',m,'--seed',str(s),'--alpha',str(a)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
        print('Completed',name,flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(one,jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--method',choices=['bc','td3bc','iql','sac'],default='td3bc')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--alpha',type=float,default=2.5)
    p.add_argument('--expectile',type=float,default=.7);p.add_argument('--temperature',type=float,default=3.)
    p.add_argument('--reward-mode',choices=['legacy','terminal_success','potential_success'])
    p.add_argument('--checkpoint-steps',type=int,nargs='+',help='Training schedule override; leaves the dataset contract unchanged')
    p.add_argument('--reward-budget',type=int,default=16);p.add_argument('--reward-eta',type=float,default=1.)
    a=p.parse_args()
    if a.all:
        if a.checkpoint_steps:p.error('--checkpoint-steps requires an individual training run')
        if a.reward_mode:p.error('--all is the original experiment; use individual runs for reward ablations')
        all_runs()
    else:
        spec=None
        if a.reward_mode:
            from reward_design import RewardSpec
            cfg=config();spec=RewardSpec(a.reward_mode,a.reward_budget,cfg['discount'],a.reward_eta,cfg['margin'],cfg['primary_ratio'])
        train(a.method,a.seed,a.alpha,spec,a.checkpoint_steps,a.expectile,a.temperature)
