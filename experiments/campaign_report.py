"""Summarize actual quality-dataset training and prespecified evaluations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np


def mean(xs):return float(np.mean(xs)) if xs else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    protocol=json.loads((root/'protocol.json').read_text())
    runs=json.loads((root/'adc_runs.json').read_text());adc=[]
    for desc in runs:
        folder=root/desc['path'];run=json.loads((folder/'trace.json').read_text())
        confirmation=json.loads((folder/'confirmation.json').read_text());best=run['best']['oracle']
        reference=[r['oracle']['mean_driver_energy_pj'] for r in run['trace'][:8] if r['oracle']['feasible']]
        adc.append(dict(**desc,design_queries=run['design_queries'],transient_queries=run['transient_queries'],
                   feasible=best['feasible'],energy_pj=best['mean_driver_energy_pj'],error_mv=best['worst_hold_error_mv'],
                   dense_feasible=confirmation['dense']['feasible'],independent_quality=confirmation['independent_quality_all'],
                   energy_improvement_percent=100*(1-best['mean_driver_energy_pj']/min(reference)) if reference and best['feasible'] and desc['method']!='lookup' else None))
    grouped=defaultdict(list)
    for r in adc:grouped[(r['task'],r['quality'],r['method'],r['attention'],r['line_search'])].append(r)
    adc_summary=[]
    for key,rows in grouped.items():
        adc_summary.append(dict(zip(('task','quality','method','attention','line_search'),key),runs=len(rows),
            feasible=sum(r['feasible'] for r in rows),dense_feasible=sum(r['dense_feasible'] for r in rows),
            independent_quality=sum(r['independent_quality'] for r in rows),
            feasible_energy_pj_mean=mean([r['energy_pj'] for r in rows if r['feasible']]),
            improvement_percent_mean=mean([r['energy_improvement_percent'] for r in rows if r['energy_improvement_percent'] is not None])))
    # All qualities share the same test designs. Compare restored log targets,
    # not normalized MSEs whose scale depends on the training quality subset.
    sys.path.insert(0,str(root/'adc_driver'))
    import torch
    from learning import Model,target
    torch.set_num_threads(1)
    predictions=[]
    for task in ('nominal','robust'):
        for quality in ('low','medium','high','mixed'):
            data=json.loads((root/f'adc_driver/data/{task}/{quality}.json').read_text())
            test=[r for r in data['records'] if r['split']==2 and r['oracle']['valid']]
            for method in ('regression','rnc'):
                for seed in (0,1,2):
                    path=root/f'adc_driver/checkpoints/{task}/{quality}/{method}_s{seed}.pt'
                    payload=torch.load(path,weights_only=False,map_location='cpu');model=Model().eval();model.load_state_dict(payload['state_dict'])
                    with torch.no_grad():
                        out=model(torch.tensor(np.array([r['u'] for r in test]),dtype=torch.float32))['pred']
                        pred=(out*payload['std']+payload['mean']).numpy()
                    actual=np.array([target(r['oracle']) for r in test]);mae=np.abs(pred-actual).mean(0)
                    predictions.append(dict(task=task,quality=quality,method=method,seed=seed,best_epoch=payload['best_epoch'],
                        valid_test_designs=len(test),log10_error_mae=float(mae[0]),log10_energy_mae=float(mae[1])))
    d2d=[];quality_data={};selection={};validation_cost={};test_cost={}
    for quality in ('low','medium','high','mixed'):
        out=root/f'd2d_qualities/{quality}/results/coupled_v1'
        data=json.loads((out/'test.json').read_text());rows=data['records']
        cfg=json.loads((out/'config.json').read_text());quality_data[quality]=json.loads((out/'dataset.json').read_text())
        selection[quality]=json.loads((out/'selection.json').read_text())
        trials=json.loads((out/'validation.json').read_text())
        validation_cost[quality]=sum(sum(r['measurement_bundles'] for r in t['records'] if r['budget']==cfg['horizon']) for t in trials)
        test_cost[quality]=sum(r['measurement_bundles'] for r in rows if r['budget']==cfg['horizon'])
        for method in ('fixed8','traditional','bo','ga','random','bc','td3bc'):
            rr=[r for r in rows if r['method']==method and r['budget']==cfg['primary_budget']]
            per_seed=[]
            for seed in sorted({r['seed'] for r in rr}):
                subset=[r for r in rr if r['seed']==seed]
                per_seed.append(dict(seed=seed,success_percent=100*mean([r['confirmation']['quality_all_patterns'] and r['confirmation']['energy_ratio']<=cfg['primary_ratio'] for r in subset]),
                    quality_percent=100*mean([r['confirmation']['quality_all_patterns'] for r in subset])))
            scores=[r['success_percent'] for r in per_seed]
            d2d.append(dict(quality=quality,method=method,joint_success_percent=mean(scores),
                seed_std_percent=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                quality_percent=mean([r['quality_percent'] for r in per_seed]),seeds=per_seed))
    oracle=json.loads((root/'d2d_coupled_calibration/results/coupled_v1/independent_oracle.json').read_text())
    oracle_counts={str(r):sum(row['feasible_by_ratio'][str(r)] for row in oracle['records']) for r in (.95,.9,.85)}
    source={task:json.loads((root/f'adc_driver/data/{task}/pool.json').read_text()) for task in ('nominal','robust')}
    summary=dict(protocol=protocol,adc=adc_summary,adc_runs=adc,adc_prediction=predictions,d2d=d2d,
        d2d_quality_data=quality_data,d2d_selection=selection,d2d_independent_grid_feasible_counts=oracle_counts,
        costs=dict(adc_source_designs=sum(v['design_queries'] for v in source.values()),
            adc_source_transients=sum(v['transient_queries'] for v in source.values()),
            adc_online_design_queries=sum(r['design_queries'] for r in adc),
            adc_online_transients=sum(r['transient_queries'] for r in adc),
            d2d_source_transitions=quality_data['mixed']['source_pool_transitions'],
            d2d_source_logical_measurements=quality_data['mixed']['logical_measurement_bundles'],
            d2d_validation_logical_measurements=validation_cost,d2d_test_calibration_logical_measurements=test_cost),
        interpretation='Quality types reuse one paid source pool. ADC quality uses training-only design cost; D2D quality uses within-context episode mean utility. All types share held-out evaluation. Seed variation is not variation across independent circuit populations.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# ADC / D2D 품질별 본학습 결과','',
        '새 SPICE 데이터와 3개 seed로 학습·탐색·독립 확인을 완료한 결과입니다. 작은 실행 테스트의 checkpoint를 재사용하지 않았습니다.',
        '', '## Offline 수집 비용과 데이터 구성','',
        f"- ADC: 원천 설계 {summary['costs']['adc_source_designs']}개, 실제 transient {summary['costs']['adc_source_transients']}회. 조건군별 512개 중 학습용 성능 분위수로 품질을 구분했습니다.",
        '- ADC 각 품질: 학습 설계 119개와 공통 validation/test. high 선별을 위해 평가한 전체 원천 풀도 비용에 포함합니다.',
        f"- D2D: 원천 전이 {summary['costs']['d2d_source_transitions']}개, 초기 측정을 포함한 논리적 질의 {summary['costs']['d2d_source_logical_measurements']}회. 각 품질은 96개 episode / 3,072개 전이입니다.",
        '- mixed는 기존 풀의 부분집합을 섞습니다. 네 유형의 수집 비용을 중복 합산하지 않습니다. validation 튜닝·test 측정 비용은 summary.json에 별도로 기록했습니다.',
        '', '## ADC: 실제 128회 탐색 결과','',
        'lookup은 학습 데이터의 최선 설계를 한 번 새로 평가합니다. 나머지 방법은 동일한 초기 8개를 포함한 128회 예산입니다. 에너지는 사양을 만족한 run의 조건부 평균이므로 성공 수와 함께 해석해야 합니다.',
        '', '| 조건 | 품질 | 방법 | attention / line search | 사양 충족 | 독립 입력 충족 | feasible 평균 pJ |',
        '|---|---|---|---|---:|---:|---:|']
    for r in adc_summary:
        energy='—' if r['feasible_energy_pj_mean'] is None else f"{r['feasible_energy_pj_mean']:.3f}"
        lines.append(f"| {r['task']} | {r['quality']} | {r['method']} | {r['attention']} / {r['line_search']} | {r['feasible']}/{r['runs']} | {r['independent_quality']}/{r['runs']} | {energy} |")
    lines += ['', '예측 정확도는 공통 test 설계에서 원래 log10 단위의 MAE로 비교했으며 summary.json의 adc_prediction에 기록했습니다. 품질별 정규화 MSE를 서로 직접 비교하지 않습니다.',
        '', '## D2D: 추가 측정 16회, 독립 패킷 품질 + 10% 에너지 절감','',
        '| 데이터 품질 | 방법 | 성공률 % | seed 표준편차 | 품질만 충족 % |', '|---|---|---:|---:|---:|']
    for r in d2d:
        sd='—' if r['seed_std_percent'] is None else f"{r['seed_std_percent']:.2f}"
        lines.append(f"| {r['quality']} | {r['method']} | {r['joint_success_percent']:.2f} | {sd} | {r['quality_percent']:.2f} |")
    lines += ['', f'독립 패킷 전체 격자의 5/10/15% 절감 가능 조건 수(분모 32): {oracle_counts}. 이 격자는 모든 평가 설정을 고정한 뒤 계산했습니다.',
        '', '3개 seed는 같은 회로 조건을 공유합니다. 반복 seed를 독립된 새 회로 표본으로 해석하지 않습니다. high는 이 수집 풀에서 상대적인 실측 품질을 뜻하며 최적 정책 또는 모든 사양을 만족하는 데이터를 의미하지 않습니다.',
        '', '검증 시 선택한 모델·단계·하이퍼파라미터는 summary.json의 d2d_selection, 개별 실행 명령·해시는 status.json, 상세 로그는 logs/에서 확인할 수 있습니다.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['costs'],indent=2),flush=True)

if __name__=='__main__':main()
