"""Operational-response regression and small parent-reuse critic buffer."""
import copy
import numpy as np
import torch
from d2d_sizing.surrogate import Model,Entry,Search
from adc_driver.rnc import rank_n_contrast
from d2d_sizing.surrogate import target as static_target
from d2d_sizing.circuit import cost as static_cost

def vector(r):
    return np.array([r['worst_violation_rate'],r['worst_downtime_fraction'],np.log10(r['mean_total_energy_pj']),r['longest_failure_run']/64],np.float32)

def score(r,quality_limit=.05,downtime_limit=.25):
    violation=max(0,r['worst_violation_rate']/quality_limit-1)+max(0,r['worst_downtime_fraction']/downtime_limit-1)
    e=r['mean_total_energy_pj'];return 1+violation if violation>0 else e/(8192+e)

class DesignSearch:
    def __init__(self,rows,method='regression',seed=491,mode='operational'):
        if method not in ('regression','rnc'):raise ValueError(method)
        self.method=method;self.seed=seed;self.rows=list(rows);self.events=[]
        self.vector=static_target if mode=='static' else vector;self.score=static_cost if mode=='static' else score;self.mode=mode
        y=np.array([self.vector(r['result']) for r in rows]);self.mean=torch.tensor(y.mean(0));self.std=torch.tensor(np.maximum(y.std(0),[.05,.05,.05,.05]),dtype=torch.float32)
        torch.manual_seed(seed);model=Model();self.entries=[Entry(0,model,[])];self.current=self.entries[0];self.next_id=1
        self._fit(model,self.rows,200)
    def _fit(self,model,rows,steps):
        model.train().requires_grad_(True);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        u=torch.tensor(np.array([r['u'] for r in rows]),dtype=torch.float32)
        y=(torch.tensor(np.array([self.vector(r['result']) for r in rows]))-self.mean)/self.std
        for _ in range(steps):
            pred,_,_,z=model(u);loss=(pred-y).square().mean()
            if self.method=='rnc':loss=loss+.1*rank_n_contrast(z,y)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        model.eval().requires_grad_(False)
    def predict(self,model,u):return model(u)[0]*self.std+self.mean
    def propose(self,excluded):
        rng=np.random.default_rng(self.seed+len(self.rows)*1009);center=np.array(min(self.rows,key=lambda r:self.score(r['result']))['u'])
        pool=np.r_[np.clip(center+rng.normal(0,.12,(128,6)),0,1),rng.uniform(0,1,(128,6))]
        pool=pool[np.min(np.linalg.norm(pool[:,None]-np.array(excluded)[None],axis=-1),axis=1)>1e-5]
        with torch.no_grad():y=self.predict(self.current.model,torch.tensor(pool,dtype=torch.float32)).numpy()
        v=(np.maximum(0,(.08-y[:,:2])/.08).sum(1)+np.maximum(0,(y[:,3]-.2)/.2) if self.mode=='static' else np.maximum(0,y[:,0]/.05-1)+np.maximum(0,y[:,1]/.25-1));e=10**np.clip(y[:,2],-5,8)
        u=pool[int(np.where(v>0,1+v,e/((1 if self.mode=='static' else 8192)+e)).argmin())]
        with torch.no_grad():pred=[dict(id=x.id,prediction=self.predict(x.model,torch.tensor(u[None],dtype=torch.float32))[0].tolist()) for x in self.entries]
        return u,pred
    def observe(self,row,predictions):
        target=self.vector(row['result']);old={e.id:e for e in self.entries}
        for r in predictions:old[r['id']].errors.append(float(np.mean(((np.array(r['prediction'])-target)/self.std.numpy())**2)))
        self.rows.append(row);scores=np.array([Search.score(e) for e in self.entries]);prob=np.exp(-(scores-scores.min())/.25);prob/=prob.sum()
        rng=np.random.default_rng(self.seed+len(self.rows));parent=self.entries[int(rng.choice(len(self.entries),p=prob))]
        child=Entry(self.next_id,copy.deepcopy(parent.model),[],Search.score(parent));self.next_id+=1;self._fit(child.model,self.rows,100)
        if len(self.entries)==4:self.entries.pop(int(scores.argmax()))
        self.entries.append(child);self.current=child
        self.events.append(dict(parent=parent.id,child=child.id,premeasurement_predictions=predictions,
            normalized_error_scores=scores.tolist(),parent_probabilities=prob.tolist(),gradient_steps=100,
            note='Heuristic parent reuse; child has no independent adoption test'))
