"""Optional regression/RNC surrogate for this new sampling task.

Old OPAMP gain/UGF checkpoints are not loaded: the targets are now measured
hold-error and dynamic energy. This module learns fresh models from source data.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from circuit import NAMES,cost,write_json
from rnc import rank_n_contrast


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
    """Frozen encoder, paid-data residual fit, metric descent and Armijo.

This is the earlier ALARM-inspired subset, not full ALARM. No attention metric
or critic buffer. Top-three starts and a candidate bank are explicit additions.
"""
    def __init__(self,path,contexts):
        from search import fingerprint
        torch.set_num_threads(1)
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['context_hash']!=fingerprint(contexts): raise ValueError('Checkpoint context differs from the requested task')
        if len({c.error_limit_mv for c in contexts})!=1:
            raise ValueError('Surrogate search requires one common error limit across contexts')
        self.payload=p;self.model=Model().eval();self.model.load_state_dict(p['state_dict'])
        self.mean=p['mean'];self.std=p['std'];self.limit=np.log10(contexts[0].error_limit_mv);self.coef=None

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
        if self.coef is not None:
            phi=torch.cat([(out['h']-self.hm)/self.hs,torch.ones(len(pred),1)],1)
            pred=pred+phi@self.coef
        y=pred*self.std+self.mean
        return (F.softplus(5*(y[:,0]-self.limit))/5).square()+.02*y[:,1]+.2*F.softplus(-out['logit'])

    def propose(self,rows,rng):
        self.fit(rows);elites=sorted(rows,key=lambda r:cost(r['oracle']))[:3]
        centers=[np.array(r['u']) for r in elites]
        starts=centers+[np.clip(centers[0]+rng.normal(0,.15,len(NAMES)),0,1) for _ in range(2)]
        candidates=[]
        for start in starts:
            u=torch.tensor(start,dtype=torch.float32);center=u.clone()
            for _ in range(6):
                u=u.detach().requires_grad_(True);f=self.loss(u)[0];g=torch.autograd.grad(f,u)[0]
                if not torch.isfinite(g).all() or g.norm()<1e-8:break
                metric=torch.outer(g,g)+torch.diag(g.square()+.01*g.square().mean()+1e-4)
                direction=-torch.linalg.solve(metric,g);direction=direction/direction.norm().clamp_min(1e-8)*.15
                accepted=False
                with torch.no_grad():
                    for k in range(8):
                        proposal=(u+(0.5**k)*direction).clamp(0,1)
                        proposal=torch.maximum(torch.minimum(proposal,center+.25),center-.25).clamp(0,1)
                        if self.loss(proposal)[0]<=f+1e-4*torch.dot(g,proposal-u):
                            u=proposal;accepted=True;break
                if not accepted:break
            candidates.append(u.detach().numpy())
        candidates.extend(np.clip(centers[0]+rng.normal(0,.15,(64,len(NAMES))),0,1))
        candidates.extend(rng.uniform(0,1,(64,len(NAMES))))
        a=np.array(candidates);old=np.array([r['u'] for r in rows])
        a=a[np.min(np.linalg.norm(a[:,None,:]-old[None,:,:],axis=2),axis=1)>1e-6]
        with torch.no_grad(): scores=self.loss(torch.tensor(a,dtype=torch.float32)).numpy()
        return a[np.argmin(scores)]
