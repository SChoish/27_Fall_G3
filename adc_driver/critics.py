"""Full-state critic reuse with prequential (before learning) evidence.

The score below is a documented adaptation, not a reproduction of ALARM.
Normalization stays fixed to offline training statistics across online versions.
No held-out design labels enter replay or critic selection.
"""
import copy
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CriticOptions:
    mode: str = 'frozen'
    source_data: str | None = None
    capacity: int = 4
    epochs: int = 20
    lr: float = 3e-4
    temperature: float = .25
    recent: int = 32
    locality: float = .3

    def __post_init__(self):
        if self.mode not in ('frozen','latest','buffer'):
            raise ValueError('critic mode must be frozen, latest, or buffer')
        for name in ('capacity','epochs','recent'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1: raise ValueError(name)
        for name in ('lr','temperature','locality'):
            v=getattr(self,name)
            if not np.isfinite(v) or v<=0: raise ValueError(name)
        if self.mode!='frozen' and not self.source_data:
            raise ValueError('Online critic updates require the matching offline source data')


def selection_probabilities(scores,temperature):
    a=-np.asarray(scores,dtype=float)/temperature
    a-=a.max()
    p=np.exp(a)
    return p/p.sum()


class CriticBuffer:
    def __init__(self,payload,options=None):
        self.options=options or CriticOptions()
        self.method=payload['method'];self.seed=payload['seed'];self.source_hash=payload['source_sha256']
        self.offline=[]
        if self.options.source_data:
            path=Path(self.options.source_data)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.source_hash:
                raise ValueError('Offline replay file does not match checkpoint source SHA256')
            data=json.loads(path.read_text())
            if data['context_hash']!=payload['context_hash']: raise ValueError('Replay context mismatch')
            self.offline=[r for r in data['records'] if r['split']==0]
            if not self.offline: raise ValueError('Empty offline training split')
        self.entries=[dict(id=0,state_dict=copy.deepcopy(payload['state_dict']),
                           mean=payload['mean'].clone(),std=payload['std'].clone(),
                           created_after_query=0,center=None,evidence=[],improvements=[],prior_score=1.)]
        self.current=self.entries[0];self.next_id=1;self.last_update=-1
        self.events=[]

    def predict(self,u):
        from learning import Model
        points=torch.tensor(np.asarray(u).reshape(1,-1),dtype=torch.float32)
        records=[]
        for entry in self.entries:
            model=Model().eval();model.load_state_dict(entry['state_dict'])
            with torch.no_grad():
                out=model(points);y=out['pred'][0]*entry['std']+entry['mean']
            records.append(dict(critic_id=entry['id'],log_targets=y.tolist(),
                                valid_probability=float(out['logit'].sigmoid()[0])))
        return records

    def observe(self,row,predictions,previous_best):
        from learning import target
        by_id={e['id']:e for e in self.entries}
        for pred in predictions:
            e=by_id[pred['critic_id']]
            # The immutable model version existed BEFORE this oracle call.
            valid=bool(row['oracle']['valid']);prob=pred['valid_probability']
            error=.2*(prob-float(valid))**2
            if valid:
                error+=float(np.mean(((np.asarray(pred['log_targets'])-target(row['oracle']))/e['std'].numpy())**2))
            e['evidence'].append(dict(query=row['query'],u=row['u'],error=error,
                                      observed_valid=valid,observed_cost=row['cost'],
                                      observed_log_targets=target(row['oracle']).tolist() if valid else None))
            e['evidence']=e['evidence'][-self.options.recent:]
        # Only actual generated proposals receive improvement credit; not warm starts.
        if row['query']>8 and previous_best is not None:
            gain=max(0.,previous_best-row['cost'])/max(1.,abs(previous_best))
            self.current['improvements'].append(dict(query=row['query'],gain=gain))
            self.current['improvements']=self.current['improvements'][-self.options.recent:]

    def score(self,entry,center,query):
        evidence=entry['evidence']
        if not evidence:return entry['prior_score']
        weights=np.array([np.exp(-(query-r['query'])/self.options.recent)*
                          (.1+.9*np.exp(-np.mean((np.asarray(r['u'])-center)**2)/(2*self.options.locality**2)))
                          for r in evidence])
        prediction=float(np.average([r['error'] for r in evidence],weights=weights))
        gains=[r['gain']*np.exp(-(query-r['query'])/self.options.recent) for r in entry['improvements']]
        # Small bounded tie influence; one lucky improvement cannot dominate accuracy.
        return prediction-.05*(sum(gains)/max(1,len(evidence)))

    def update(self,rows):
        if self.options.mode=='frozen' or len(rows)==self.last_update:
            return dict(mode=self.options.mode,updated=False,selected_id=self.current['id'])
        from learning import Model,target
        from rnc import rank_n_contrast
        self.last_update=len(rows)
        center=np.asarray(min(rows,key=lambda r:r['cost'])['u'])
        scores=[self.score(e,center,len(rows)) for e in self.entries]
        probabilities=selection_probabilities(scores,self.options.temperature)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,len(rows),87231]))
        selected=(int(rng.choice(len(self.entries),p=probabilities)) if self.options.mode=='buffer'
                  else next(i for i,e in enumerate(self.entries) if e['id']==self.current['id']))
        parent=self.entries[selected]
        model=Model();model.load_state_dict(parent['state_dict']);model.train()
        mean=parent['mean'];std=parent['std']
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.options.lr,weight_decay=1e-4)
        # Balanced offline/online replay each epoch, all accumulated online points eligible.
        for _ in range(self.options.epochs):
            count=max(len(self.offline),len(rows))
            a=rng.choice(len(self.offline),count,replace=len(self.offline)<count)
            b=rng.choice(len(rows),count,replace=len(rows)<count)
            replay=[self.offline[i] for i in a]+[rows[i] for i in b]
            for ix in np.array_split(rng.permutation(len(replay)),max(1,int(np.ceil(len(replay)/32)))):
                batch=[replay[i] for i in ix]
                u=torch.tensor(np.array([r['u'] for r in batch]),dtype=torch.float32)
                good=torch.tensor([r['oracle']['valid'] for r in batch],dtype=torch.bool)
                out=model(u)
                reg=out['pred'].sum()*0;contrast=reg
                if good.any():
                    y=torch.tensor(np.array([target(r['oracle']) for r in batch if r['oracle']['valid']]),dtype=torch.float32)
                    yn=(y-mean)/std
                    reg=F.mse_loss(out['pred'][good],yn)
                    if self.method=='rnc':contrast=rank_n_contrast(out['z'][good],yn)
                loss=reg+.2*F.binary_cross_entropy_with_logits(out['logit'],good.float())+.1*contrast
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite online critic loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        fresh=dict(id=self.next_id,state_dict=copy.deepcopy(model.state_dict()),mean=mean.clone(),std=std.clone(),
                   created_after_query=len(rows),center=center.tolist(),evidence=[],improvements=[],
                   prior_score=scores[selected],parent_id=parent['id'])
        self.next_id+=1
        event=dict(mode=self.options.mode,updated=True,after_query=len(rows),parent_id=parent['id'],
                   selected_id=fresh['id'],candidate_ids=[e['id'] for e in self.entries],
                   scores=scores,selection_probabilities=probabilities.tolist(),
                   offline_training_rows=len(self.offline),online_rows=len(rows),epochs=self.options.epochs,
                   normalization='fixed offline training statistics',additional_spice_queries=0)
        if self.options.mode=='latest':self.entries=[fresh]
        else:
            self.entries.append(fresh)
            if len(self.entries)>self.options.capacity:
                # Keep the just-trained child for prospective evaluation; evict worst old score.
                worst=max(range(len(scores)),key=lambda i:(scores[i],-self.entries[i]['created_after_query']))
                event['evicted_id']=self.entries.pop(worst)['id']
        self.current=fresh;self.events.append(event)
        return event

    def save(self,path):
        torch.save(dict(options=asdict(self.options),entries=self.entries,current_id=self.current['id'],
                        next_id=self.next_id,events=self.events,method=self.method,source_sha256=self.source_hash,
                        note='Full encoder, heads, RNC projector and normalization; evidence is prequential. Optimizer is reset on each update.'),path)
