"""Replay fitted ensembles on their next actual development query, without new SPICE runs."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from experiments import d2d_replay_search as replay
from experiments.run_phase_sizing import d


def audit(folder):
    seed=json.loads((folder/'protocol.json').read_text())['seed']
    rows=json.loads((folder/'mlp_local.json').read_text())['rows']
    records=[]
    original_fit=MLPRegressor.fit
    for step in range(8,len(rows)):
        history=rows[:step]
        models=[]
        def keep_fit(model,*args,**kwargs):
            result=original_fit(model,*args,**kwargs)
            models.append(model)
            return result
        with warnings.catch_warnings(record=True) as caught, patch.object(MLPRegressor,'fit',keep_fit):
            warnings.simplefilter('always',ConvergenceWarning)
            proposed=replay.propose(history,[],seed,local_edits=True)
        np.testing.assert_allclose(proposed,rows[step]['x'],rtol=0,atol=1e-10)
        x=d.encode(proposed)
        observations=np.array([replay.targets(r['oracle']) for r in history])
        y=replay.targets(rows[step]['oracle'])
        scale=observations.std(axis=0).clip(.05)
        forecasts=np.array([model.predict(x[None])[0]*scale+observations.mean(axis=0) for model in models])
        mlp=np.r_[np.median(forecasts[:,0]),np.quantile(forecasts[:,1:],.75,axis=0)]
        nearest=np.argmin([np.linalg.norm(x-d.encode(r['x'])) for r in history])
        record=dict(step=step,warnings=sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        for name,pred in [('mlp',mlp),('nearest',observations[nearest])]:
            record[name]=dict(relative_energy_error=float(abs(np.exp(pred[0]-y[0])-1)),
                              predicted_feasible=bool(max(pred[1:])<=0))
        record['actual_feasible']=bool(max(y[1:])<=0)
        records.append(record)
    return seed,records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('folders',type=Path,nargs='+')
    args=p.parse_args()
    data=dict(audit(folder) for folder in args.folders)
    lines=['# 국소 MLP의 개발 중 다음 제안 예측 검산','',
           '각 seed의 공통 초기 8개 이후 10개 제안에서 당시까지 지불한 짧은 측정만 사용해 5개 MLP 앙상블을 복원했다. 실제 선택한 설계 벡터와 30/30 일치하는지 확인하고, 그 설계에서 당시 예측과 저장된 새 SPICE 관측을 비교했다. 근접 대조는 같은 과거 데이터에서 정규화한 로그 설계변수 거리 기준 최근접 관측이다. 시험 패킷 사용 및 추가 SPICE 실행은 없다.',
           '', '|seed|예측기|에너지 상대절대오차 평균|예측 통과|예측 통과 중 실제 실패|실제 통과|MLP 수렴 경고/50|',
           '|---:|---|---:|---:|---:|---:|---:|']
    for seed,records in data.items():
        for name in ('mlp','nearest'):
            energy=np.mean([r[name]['relative_energy_error'] for r in records])
            passed=sum(r[name]['predicted_feasible'] for r in records)
            false=sum(r[name]['predicted_feasible'] and not r['actual_feasible'] for r in records)
            good=sum(r['actual_feasible'] for r in records)
            warn=str(sum(r['warnings'] for r in records))+'/50' if name=='mlp' else '—'
            lines.append(f'|{seed}|{name}|{100*energy:.2f}%|{passed}/10|{false}|{good}/10|{warn}|')
    lines+=['','제약에는 원래 제안기와 같은 앙상블 75% 분위, 에너지에는 중앙값을 사용했다. 평가 표본은 MLP가 제안한 회로에 국한된다. 경고 횟수와 제약 예측 오차 사이의 인과관계는 이 검산으로 식별할 수 없다.']
    args.output.write_text('\n'.join(lines)+'\n')
    d.write_json(args.output.with_suffix('.json'),data)
