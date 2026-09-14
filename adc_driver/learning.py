"""Optional regression/RNC surrogate for this new sampling task.

Old OPAMP gain/UGF checkpoints are not loaded: the targets are now measured
hold-error and dynamic energy. This module learns fresh models from source data.
"""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from circuit import NAMES,cost,write_json
from rnc import rank_n_contrast
from attention import (CoordinateAttention, SearchOptions, metric_direction,
                       metric_matrix, project)


SURROGATE_OBJECTIVE = 'validity_feasibility_energy_v2'


def surrogate_cost(log_targets, logits, log_error_limit):
    """Predicted valid+feasible < valid+infeasible < invalid.

    Feasible cost is E/(E+1000), matching circuit.cost, independent of excess
    accuracy or validity confidence. Infeasible cost increases monotonically
    with log error. The bounded tiers preserve priority even at extreme values.
    This piecewise objective is differentiable within each tier; crossing a
    validity/feasibility boundary may jump and is checked at the endpoint.
    """
    violation = (log_targets[:, 0] - log_error_limit).clamp_min(0)
    energy = torch.sigmoid(np.log(10.) * (log_targets[:, 1] - 3.))
    infeasible = 1. + violation / (1. + violation)
    invalid = 2. + torch.sigmoid(-logits)
    return torch.where(logits < 0, invalid,
                       torch.where(violation > 0, infeasible, energy))


def target(r):
    return np.log10([max(r['worst_hold_error_mv'],1e-6),max(r['mean_driver_energy_pj'],1e-6)])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(len(NAMES),64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.regressor=nn.Linear(64,2);self.classifier=nn.Linear(64,1)
        self.projector=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,16))
    def forward(self,u):
        h=self.encoder(u)
        return dict(h=h,pred=self.regressor(h),logit=self.classifier(h)[:,0],z=self.projector(h))


def train(data_path,method,seed,output,epochs=150):
    if method not in ['regression','rnc']: raise ValueError(method)
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed+7001)
    data=json.loads(Path(data_path).read_text());rows=data['records']
    u=torch.tensor(np.array([r['u'] for r in rows]),dtype=torch.float32)
    valid=np.array([r['oracle']['valid'] for r in rows]);split=np.array([r['split'] for r in rows])
    y=torch.tensor(np.array([target(r['oracle']) for r in rows]),dtype=torch.float32)
    tr=np.flatnonzero(split==0);good=np.flatnonzero((split==0)&valid)
    va=np.flatnonzero((split==1)&valid);te=np.flatnonzero((split==2)&valid)
    if len(good)<4 or len(va)<1: raise ValueError('Need >=4 valid training and >=1 valid validation observations')
    mean=y[good].mean(0);std=y[good].std(0).clamp_min(.05);yn=(y-mean)/std
    model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');state=None;history=[]
    for epoch in range(epochs):
        model.train()
        for ix in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/32)))):
            out=model(u[ix]);good_batch=torch.tensor(valid[ix])
            reg=F.mse_loss(out['pred'][good_batch],yn[ix][good_batch]) if good_batch.any() else out['pred'].sum()*0
            classification=F.binary_cross_entropy_with_logits(out['logit'],torch.tensor(valid[ix],dtype=torch.float32))
            rnc=rank_n_contrast(out['z'][good_batch],yn[ix][good_batch]) if method=='rnc' else reg*0
            loss=reg+.2*classification+.1*rnc
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        model.eval()
        with torch.no_grad(): mse=F.mse_loss(model(u[va])['pred'],yn[va]).item()
        history.append(mse)
        if mse<best:best=mse;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(state)
    with torch.no_grad():
        test=None if len(te)==0 else float(F.mse_loss(model(u[te])['pred'],yn[te]))
    payload=dict(state_dict=state,mean=mean,std=std,method=method,seed=seed,epochs=epochs,
                 best_epoch=best_epoch,validation_mse=best,test_mse=test,
                 valid_counts={'train':len(good),'validation':len(va),'test':len(te)},
                 context_hash=data['context_hash'],contexts=data['contexts'],
                 source_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                 source_design_queries=data['design_queries'],history=history)
    Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output)
    report={k:v for k,v in payload.items() if k not in ['state_dict','mean','std','history']}
    write_json(str(output)+'.json',report)
    return report


class LocalSearch:
    """Frozen surrogate, paid-data residual fit and learned coordinate metric.

    Attention learns through a differentiable one-step proposal with the critic
    fixed, following ALARM Eq. (27). Execution uses fixed projected steps by
    default; surrogate Armijo is optional. Critic updates are independently configurable.
    """
    def __init__(self,path,contexts,options=None,critic_options=None):
        from search import physical_fingerprint
        from circuit import Context
        torch.set_num_threads(1)
        p=torch.load(path,map_location='cpu',weights_only=False)
        if physical_fingerprint([Context.from_dict(c) for c in p['contexts']])!=physical_fingerprint(contexts): raise ValueError('Checkpoint context differs from the requested task')
        if len({c.error_limit_mv for c in contexts})!=1:
            raise ValueError('Surrogate search requires one common error limit across contexts')
        self.payload=p;self.model=Model().eval();self.model.load_state_dict(p['state_dict'])
        self.model.requires_grad_(False)
        self.mean=p['mean'];self.std=p['std'];self.limit=np.log10(contexts[0].error_limit_mv);self.coef=None
        self.options=options or SearchOptions()
        self.attention=CoordinateAttention(len(NAMES)) if self.options.attention=='learned' else None
        self.last_report={};self.objective_evaluations=0
        from critics import CriticBuffer
        self.critics=CriticBuffer(p,critic_options)

    def fit(self,rows):
        good=[r for r in rows if r['oracle']['valid']]
        self.coef=None
        if len(good)<2:return
        with torch.no_grad():
            out=self.model(torch.tensor(np.array([r['u'] for r in good]),dtype=torch.float32))
            h=out['h'];self.hm=h.mean(0);self.hs=h.std(0).clamp_min(.2)
            phi=torch.cat([(h-self.hm)/self.hs,torch.ones(len(h),1)],1)
            y=torch.tensor(np.array([target(r['oracle']) for r in good]),dtype=torch.float32)
            residual=(y-self.mean)/self.std-out['pred']
            self.coef=phi.T@torch.linalg.solve(phi@phi.T+.5*torch.eye(len(phi)),residual)

    def loss(self,u):
        out=self.model(u.reshape(-1,len(NAMES)));pred=out['pred']
        self.objective_evaluations+=len(pred)
        if self.coef is not None:
            phi=torch.cat([(out['h']-self.hm)/self.hs,torch.ones(len(pred),1)],1)
            pred=pred+phi@self.coef
        y=pred*self.std+self.mean
        return surrogate_cost(y,out['logit'],self.limit)

    def probabilities(self,u):
        if self.attention is None: return torch.ones_like(u)/len(NAMES)
        return self.attention(u)

    def update_attention(self,rows,rng):
        """All paid valid observations are eligible, not just elite starts.

        No new SPICE calls, held-out labels, or pos/neg annotations. Gradients
        flow through the solve, projection and endpoint surrogate to eta. The
        initial gradient is fixed because these starting designs are fixed.
        """
        good=[r for r in rows if r['oracle']['valid']]
        report=dict(enabled=self.attention is not None,eligible_observations=len(good),
                    training_observations=0,updates=0,additional_spice_queries=0,
                    objective='Mean frozen-surrogate cost after one fixed projected metric step')
        if self.attention is None or not good: return report
        indices=rng.choice(len(good),size=min(32,len(good)),replace=False)
        points=torch.tensor(np.array([good[i]['u'] for i in indices]),dtype=torch.float32,requires_grad=True)
        base_loss=self.loss(points)
        gradient=torch.autograd.grad(base_loss.sum(),points)[0].detach()
        points=points.detach()
        active=torch.isfinite(gradient).all(-1)&torch.isfinite(base_loss)&(gradient.norm(dim=-1)>1e-8)
        points=points[active];gradient=gradient[active]
        if len(points)==0:return report
        before={k:v.detach().clone() for k,v in self.attention.state_dict().items()}
        optimizer=torch.optim.Adam(self.attention.parameters(),lr=self.options.attention_lr)
        def objective():
            direction=metric_direction(gradient,self.probabilities(points),self.options)
            return self.loss(project(points+direction,points,self.options)).mean()
        with torch.no_grad(): first=float(objective())
        if not np.isfinite(first): raise FloatingPointError('Non-finite attention training objective')
        best=first;state=copy.deepcopy(self.attention.state_dict());grad_norm=0.;updates=0
        for _ in range(self.options.attention_steps):
            value=objective()
            if not torch.isfinite(value):raise FloatingPointError('Non-finite attention training objective')
            optimizer.zero_grad(set_to_none=True);value.backward()
            norm=nn.utils.clip_grad_norm_(self.attention.parameters(),5,error_if_nonfinite=True)
            grad_norm=max(grad_norm,float(norm));optimizer.step();updates+=1
            with torch.no_grad(): score=float(objective())
            if np.isfinite(score) and score<best:
                best=score;state=copy.deepcopy(self.attention.state_dict())
        self.attention.load_state_dict(state)
        delta=sum(float((v-before[k]).square().sum()) for k,v in state.items())**.5
        report.update(training_observations=len(points),updates=updates,
                      loss_before=first,loss_after=best,max_gradient_norm=grad_norm,parameter_delta_l2=delta)
        return report

    def save_attention(self,path):
        if self.attention is None:return
        torch.save(dict(state_dict=self.attention.state_dict(),options=asdict(self.options),
                        surrogate_objective=SURROGATE_OBJECTIVE,
                        parameter_names=NAMES,context_hash=self.payload['context_hash'],
                        surrogate_source_sha256=self.payload['source_sha256'],
                        note='Final attention snapshot; not a resumable search or a standalone surrogate'),path)

    def propose(self,rows,rng):
        if not rows:raise ValueError('Local search requires paid observations')
        self.objective_evaluations=0
        critic_report=self.critics.update(rows)
        self.model.load_state_dict(self.critics.current["state_dict"])
        self.mean=self.critics.current["mean"];self.std=self.critics.current["std"]
        self.model.eval().requires_grad_(False)
        self.fit(rows)
        # Independent RNG stream: toggling attention does not consume the
        # candidate-bank RNG and confound the matched attention ablation.
        attention_rng=np.random.default_rng(np.random.SeedSequence([self.payload['seed'],len(rows),41871]))
        attention_report=self.update_attention(rows,attention_rng)
        elites=sorted(rows,key=lambda r:cost(r['oracle']))[:3]
        centers=[np.array(r['u']) for r in elites]
        starts=centers+[np.clip(centers[0]+rng.normal(0,.15,len(NAMES)),0,1) for _ in range(2)]
        candidates=[];origins=[];line_trials=0;accepted_steps=0
        for start in starts:
            u=torch.tensor(start,dtype=torch.float32);center=u.clone()
            for _ in range(self.options.search_steps):
                u=u.detach().requires_grad_(True);f=self.loss(u)[0];g=torch.autograd.grad(f,u)[0]
                if not torch.isfinite(f) or not torch.isfinite(g).all() or g.norm()<1e-8:break
                accepted=False
                with torch.no_grad():
                    direction=metric_direction(g,self.probabilities(u),self.options)
                    if self.options.line_search=='none':
                        proposal=project(u+direction,center,self.options)
                        accepted=bool((proposal-u).norm()>1e-8)
                    else:
                        for k in range(self.options.backtracks):
                            proposal=project(u+(0.5**k)*direction,center,self.options)
                            line_trials+=1
                            if (proposal-u).norm()<=1e-8:break
                            if self.loss(proposal)[0]<=f+self.options.armijo_c1*torch.dot(g,proposal-u):
                                accepted=True;break
                    if accepted:u=proposal;accepted_steps+=1
                if not accepted:break
                # Retain every iterate so fixed steps need no decrease test.
                candidates.append(u.detach().numpy());origins.append('metric_step')
        local=np.clip(centers[0]+rng.normal(0,.15,(64,len(NAMES))),0,1)
        candidates.extend(local);origins.extend(['local_random']*len(local))
        global_bank=rng.uniform(0,1,(64,len(NAMES)))
        candidates.extend(global_bank);origins.extend(['global_random']*len(global_bank))
        a=np.array(candidates);old=np.array([r['u'] for r in rows]);origins=np.array(origins)
        keep=np.isfinite(a).all(1)&(np.min(np.linalg.norm(a[:,None,:]-old[None,:,:],axis=2),axis=1)>1e-6)
        a=a[keep];origins=origins[keep]
        if len(a)==0:raise RuntimeError('No unevaluated finite candidate')
        with torch.no_grad(): scores=self.loss(torch.tensor(a,dtype=torch.float32)).numpy()
        scores=np.where(np.isfinite(scores),scores,np.inf)
        if not np.isfinite(scores).any():raise FloatingPointError('No finite surrogate candidate score')
        selected=int(np.argmin(scores));incumbent=torch.tensor(centers[0],dtype=torch.float32,requires_grad=True)
        g=torch.autograd.grad(self.loss(incumbent)[0],incumbent)[0]
        with torch.no_grad():
            p=self.probabilities(incumbent);eigenvalues=torch.linalg.eigvalsh(metric_matrix(g,p,self.options))
        self.last_report=dict(critic=critic_report,surrogate_objective=SURROGATE_OBJECTIVE,attention=attention_report,incumbent_query=elites[0]['query'],
                              elite_queries=[r['query'] for r in elites],
                              coordinate_probabilities=p.tolist(),parameter_names=NAMES,
                              metric_min_eigenvalue=float(eigenvalues.min()),
                              selected_origin=str(origins[selected]),predicted_loss=float(scores[selected]),
                              candidates=len(a),accepted_metric_steps=accepted_steps,
                              line_search=self.options.line_search,line_search_trials=line_trials,
                              surrogate_objective_evaluations=self.objective_evaluations,
                              additional_spice_queries=0)
        return a[selected]
