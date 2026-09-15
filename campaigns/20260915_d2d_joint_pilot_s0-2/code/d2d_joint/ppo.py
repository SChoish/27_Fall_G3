"""Categorical PPO; four-frame history, no privileged critic input."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .environment import Environment,drift_path,ACTIONS,FRAME_DIM,Config,metrics
from d2d_sizing.circuit import write_json,file_hash

class ActorCritic(nn.Module):
    def __init__(self,dim=96):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Linear(64,len(ACTIONS));self.value=nn.Linear(64,1)
    def forward(self,obs):
        h=self.encoder(obs);return Categorical(logits=self.actor(h)),self.value(h).squeeze(-1)

class Policy:
    def __init__(self,model):self.model=model.eval()
    def __call__(self,obs):
        with torch.inference_mode():dist,_=self.model(torch.as_tensor(obs,dtype=torch.float32));return int(dist.logits.argmax())

def gae(rewards,values,next_values,dones,gamma=.99,lam=.95):
    adv=np.zeros(len(rewards),np.float32);last=0.
    for i in range(len(rewards)-1,-1,-1):
        mask=1.-dones[i];delta=rewards[i]+gamma*mask*next_values[i]-values[i]
        last=delta+gamma*lam*mask*last;adv[i]=last
    return adv,adv+values

def train(banks,seed,stage,output,cfg=Config(),steps=7680,initial=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if steps%256 or steps%(len(banks)*cfg.horizon):raise ValueError('Budget must balance complete design episodes')
    torch.set_num_threads(1);torch.manual_seed(seed+stage*1000)
    model=ActorCritic(FRAME_DIM*cfg.history)
    if initial is not None:model.load_state_dict(initial.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);logs=[];episodes=[];ep=0;env=None
    design_ids=list(banks);counts={k:0 for k in banks};lookup_before=sum(b.lookups for b in banks.values())
    for update in range(steps//256):
        obs_rows=[];actions=[];rewards=[];values=[];next_values=[];dones=[];logps=[]
        for _ in range(256):
            if env is None:
                design=design_ids[ep%len(design_ids)];environment=(ep//len(design_ids))%2
                trajectory_seed=1000000+stage*100000+seed*10000+ep
                env=Environment(banks[design],drift_path(trajectory_seed,cfg),environment,cfg);counts[design]+=1;ep+=1
            obs=env.observation()
            with torch.no_grad():dist,v=model(torch.from_numpy(obs));action=dist.sample();lp=dist.log_prob(action)
            nxt,reward,done,_=env.step(int(action))
            with torch.no_grad():_,nv=model(torch.from_numpy(nxt))
            obs_rows.append(obs);actions.append(int(action));rewards.append(reward);values.append(float(v));next_values.append(float(nv));dones.append(float(done));logps.append(float(lp))
            if done:
                episodes.append(dict(episode=ep-1,design=design,environment=environment,drift_seed=trajectory_seed,metrics=metrics(env.records,cfg)));env=None
        adv,returns=gae(np.asarray(rewards),np.asarray(values,np.float32),np.asarray(next_values,np.float32),np.asarray(dones))
        x=torch.tensor(np.array(obs_rows));a=torch.tensor(actions);old=torch.tensor(logps);ret=torch.tensor(returns)
        advantage=torch.tensor((adv-adv.mean())/(adv.std()+1e-8))
        for epoch in range(4):
            for ix in torch.randperm(256).split(128):
                dist,v=model(x[ix]);new=dist.log_prob(a[ix]);ratio=(new-old[ix]).exp()
                pg=-torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix]).mean()
                vl=(v-ret[ix]).square().mean();entropy=dist.entropy().mean();loss=pg+.5*vl-.01*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
        logs.append(dict(environment_steps=(update+1)*256,reward_mean=float(np.mean(rewards)),policy_loss=float(pg.detach()),
            value_loss=float(vl.detach()),entropy=float(entropy.detach()),approx_kl=float((old[ix]-new).mean().detach()),gradient_steps=(update+1)*8))
    assert env is None
    torch.save(dict(state_dict=model.state_dict(),seed=seed,stage=stage,environment_steps=steps,observation_dim=FRAME_DIM*cfg.history,
        policy='Categorical PPO / four-frame MLP / argmax evaluation',designs=design_ids),output/'policy.pt')
    write_json(output/'training.json',dict(logs=logs,episodes=episodes,per_design_episodes=counts,blocks=steps,
        phase_bank_lookups=sum(b.lookups for b in banks.values())-lookup_before,
        calibration_measurements=sum(r['metrics']['calibration_measurements'] for r in episodes),
        tx_energy_pj=sum(r['metrics']['tx_energy_pj'] for r in episodes),
        control_energy_pj=sum(r['metrics']['total_energy_pj']-r['metrics']['tx_energy_pj'] for r in episodes),
        interrupted_seconds=sum(r['metrics']['interruption_seconds'] for r in episodes),gradient_steps=steps//256*8,
        checkpoint_sha256=file_hash(output/'policy.pt'),new_spice_calls=0))
    return model
