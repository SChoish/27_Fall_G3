"""Summarize by circuit and seed; package raw SPICE separately from cheap policy replays."""
from dataclasses import replace,asdict
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import csv,json
import numpy as np
from warm_audit import gather
from warm_policy import Config
from warm_space import DEV,TEST
from warm_study import ident

ROOT=Path("runs/warm")
def read(name):return json.loads((ROOT/name).read_text())
def stats(rows,goal):
    values=[r["hits"].get(goal) for r in rows]
    by={}
    for r,t in zip(rows,values):by.setdefault(r["circuit"],[]).append(40 if t is None else t)
    return sum(t is not None for t in values),len(values),float(np.mean([np.mean(v) for v in by.values()]))
def cell(rows,g):
    a,b,v=stats(rows,g)
    return f"{a}/{b} · {v:.2f}"
def write_csv(name,rows):
    with (ROOT/name).open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def report():
    group=gather(ROOT);lock=read("lock.json");strict=read("strict_lock.json");audit=read("audit.json")
    arms={n:Config(**v) for n,v in lock["arms"].items()}
    out=[]
    for (tag,key),records in group.items():
        for n in sorted({r["circuit"] for r in records}|{"전체"}):
            rs=records if n=="전체" else [r for r in records if r["circuit"]==n]
            for g in ("0.95","0.9"):
                a,b,v=stats(rs,g);out.append(dict(phase=tag,method=key,circuit=n,goal=g,successes=a,
                       trials=b,mean_capped=v,first_hits=[r["hits"].get(g) for r in rs]))
    write_csv("summary.csv",out)
    configs=[]
    for (tag,key),rs in group.items():
        if tag not in ("holdout_bank","development"):continue
        c=json.loads(key);selected=sum(s["restored"] is not None for r in rs for s in r["trace"])
        cfg={**c,"phase":tag}
        for goal,suffix in (("0.95","5"),("0.9","10")):
            a,b,v=stats(rs,goal);cfg.update({f"hits_{suffix}":a,f"rmst_{suffix}":v})
        cfg.update(trials=len(rs),restores=selected,paid_warm_moves=sum(s["stage"]=="warm" for r in rs for s in r["trace"]))
        configs.append(cfg)
    write_csv("sensitivity.csv",configs)
    s=["# Attention 없는 ALARM의 폭 좌표, 랜덤 웜업과 critic buffer","","표의 값은 **성공 수/시행 수 · 40회 제한 평균 SPICE eval**입니다. 동일 회로의 seed 평균을 먼저 계산합니다. 기준 4회와 후보 개발 2회, 필요할 때의 확인 2회를 포함합니다. 웜업 중 통과도 최초 도달로 셉니다. 5%와 10% 목표를 같은 궤적에서 평가했습니다.","",
       "## 미사용 회로, 연속 폭 탐색","","4개 조건 × 3개 대응 seed입니다. 한 쌍은 같은 2-lane 회로의 부하·결합을 바꾼 SKY130/GF180이고, 다른 쌍은 새 8-inverter 분기·결합 회로의 SKY130/GF180입니다. PDK 전이에 관한 관측치는 서로 다른 두 공정이며 네 개의 독립 공정을 뜻하지 않습니다.","","| 방법 | 5% 목표 | 10% 목표 |","|---|---:|---:|"]
    labels={"fixed_bank_base":"기존 연속 scalar ALARM", "scaled_only":"허용 로그 범위로 좌표 정규화",
            "residual_only":"타깃 측정으로 critic 잔차 보정","nonzero_warm":"개발에서 고른 유료 웜업 구성",
            "nonzero_buffer":"개발에서 고른 유료 웜업 없는 buffer 구성","combined":"5% 개발 선정 구성",
            "guard_secant":"초기 제약 위반 시 결정적 보폭 보정","guard_random":"동일 보폭 보정 + 무작위 perturbation",
            "strict_selected":"연속 탐색에서 따로 고른 10% 구성"}
    for key,label in labels.items():
        rs=group["strict" if key=="strict_selected" else "continuous",key]
        s.append(f"| {label} | {cell(rs,'0.95')} | {cell(rs,'0.9')} |")
    s+=["","일부 행은 같은 파라미터와 궤적을 공유합니다. 잔차 보정은 타깃에서 관측한 전기 성능 벡터와 MLP 예측의 차이를 로그 폭 공간에서 국소 보간합니다. 보폭 보정은 첫 출력단 축소가 개발 EDP를 만족하고 전기 제약을 넘겼을 때만 동작합니다. 확인 조건의 측정값을 보폭이나 critic 갱신에 사용하지 않았습니다.","","## 좌표와 물리량","",
        "각 소자에 대해 W=Wnom·exp(z)를 사용합니다. 입력을 받는 첫 인버터의 소자는 W/Wnom ∈ [0.58,1.00], 나머지는 [0.58,1.30]입니다. 허용 로그 폭은 각각 0.5447과 0.8071입니다. 비교한 정규화 좌표 u=(z−log(0.58))/log(upper/0.58)는 두 범위를 [0,1]로 옮깁니다. 두 방식 모두 물리적인 최대 로그 이동량과 critic 입력을 일치시켰습니다. SKY130 L=0.15 µm, GF180 L=0.28 µm이며 M은 인스턴스에서 생략해 기본값 1입니다. L과 M은 탐색하지 않았습니다.",
        "","한 소자의 폭을 µm와 m로 표현한 로그 비율은 같았습니다. 정규화 좌표의 analytic Jacobian과 중앙 유한차분 최대 오차는 1.21e−11입니다. 대각 damping을 제거한 scalar metric은 좌표를 원래 물리 공간으로 되돌렸을 때 방향 차이가 1.2e−16 미만이었습니다. 실제 damping을 포함하면 검사 지점 여덟 개에서 방향 차이가 약 2.0–6.9도였습니다. 예측 출력도 nominal과 전기 허용치에 대한 로그 비율입니다. 소스 출력 표준편차(EDP, 지연, slew, 전원 민감도, rail)는 각각 0.092, 0.052, 0.109, 0.068, 약 0입니다. 무차원화는 했으며 동일 분산으로 whitening하지 않았습니다.","",
        "## 랜덤 웜업 길이·범위·순서","","동일한 140개 물리적 폭 후보에서 비교한 미사용 4조건 × 5 seed입니다. 매 런에는 질의한 개발 결과만 공개했습니다. 0·1·2·4는 랜덤 설계 수입니다. 같은 seed·분포에서는 처음 k개 표본이 동일합니다. local은 역할 기반 폭 근처 ±0.095 로그 perturbation, global은 소자별 물리 허용 범위의 로그 균등표본입니다. 위치와 관계없이 역할 제안도 한 번 수행합니다.","",
        "| 분포 · 최초 순서 | 랜덤 0개 | 1개 | 2개 | 4개 |","|---|---:|---:|---:|---:|"]
    geom=arms["geometry"]
    for region in ("local","global"):
        for order in ("role","random"):
            cells=[]
            for k in (0,1,2,4):
                cfg=replace(geom,warm=k,first=order,region=region) if k else geom
                cells.append(cell(group["holdout_bank",ident(cfg)],"0.95"))
            s.append("| "+" | ".join([region+" · "+order,*cells])+" |")
    s+=["","다음은 개발에서 고른 국소 랜덤 우선 순서에서 5% 성공 또는 40회 중단까지 지불한 평균 웜업 비용입니다. 0개 대비 총비용 증가량에서 웜업 비용을 빼면 뒤의 역할·metric 제안에 지불한 비용 변화를 볼 수 있습니다. 서로 다른 궤적 사이의 비용 분해입니다.","","| 랜덤 설계 | 평균 유료 웜업 호출 | 5%를 웜업에서 직접 달성 | 0개 대비 전체 호출 변화 |","|---:|---:|---:|---:|"]
    base_rmst=stats(group["holdout_bank",ident(geom)],"0.95")[2]
    for k in (0,1,2,4):
        rs=group["holdout_bank",ident(arms[f"warm_count_{k}"])]
        spend=direct=0
        for r in rs:
            limit=r["hits"].get("0.95",40);prev=4
            for step in r["trace"]:
                if step["calls"]>limit:break
                if step["stage"]=="warm":
                    spend+=step["calls"]-prev
                    direct+=int(step["calls"]==r["hits"].get("0.95"))
                prev=step["calls"]
        s.append(f"| {k} | {spend/len(rs):.2f} | {direct}/{len(rs)} | {stats(rs,'0.95')[2]-base_rmst:+.2f} |")
    s+=["","## Critic buffer 민감도","","같은 기본 geometry, 랜덤 웜업 0개에서의 미사용 후보표 결과입니다. cap은 현재 모델과 별도로 보관하는 과거 snapshot 수입니다. 매 모델 단계 20회 업데이트를 일치시켰습니다. score는 마지막 신규 개발 설계의 정규화된 예측오차로 과거 모델을 선택하고, guard는 그 모델의 오차가 현재 모델보다 낮을 때만 복귀합니다. 복귀 확률 난수와 웜업 난수는 분리했습니다.","","| 복귀 방식 | cap | 확률 0.25 | 확률 0.5 | 확률 1 |","|---|---:|---:|---:|---:|"]
    for sel in ("random","score","guard"):
        for cap in (2,4,8):
            cells=[cell(group["holdout_bank",ident(replace(geom,cap=cap,reuse=p,select=sel))],"0.95") for p in (.25,.5,1.)]
            s.append("| "+" | ".join([sel,str(cap),*cells])+" |")
    s+=["","## 별도의 미사용 2단·4단 lane 쌍에서 출력별 보폭 검증","","앞선 네 회로 결과와 분리한 추가 실험입니다. 같은 SPICE 파형에서 출력별 상승 지연·slew는 PMOS, 하강 지연·slew는 NMOS에 연결해 두 번째 폭을 각자 계산했습니다. 최초 역할 제안과 그 이후의 scalar critic, 동일한 seed를 공유합니다.","","| 방법 | 5% 목표 | 10% 목표 |","|---|---:|---:|"]
    for key in ("base","electrical","electrical_probe"):
        rs=group["port",key]
        s.append(f"| {key} | {cell(rs,'0.95')} | {cell(rs,'0.9')} |")
    s+=["","## 선택과 비용","","개발 회로 4개와 seed 21–23에서 설정을 고르고, 미사용 쌍과 seed 31–35에서 고정된 구성을 비교했습니다. 연속 평가는 seed 31–33입니다. 개발 후보표에는 10% 목표를 통과하는 설계가 없어, 이를 달성한 전력이 있는 기존 GF180 두 회로에서 9개 설정을 별도 연속 탐색으로 비교했습니다. 이 10% 선정 결과는 GF180 개발 표본에 의존합니다.","",
        "5% 선정 설정은 `"+ident(arms["combined"])+"`입니다. 강제 웜업 설정은 `"+ident(arms["nonzero_warm"])+"`입니다. 10% 별도 개발에서 선정한 설정은 `"+strict["winner"]+"`이며 `"+json.dumps(strict["config"],sort_keys=True)+"`입니다.","",
        "고정 후보표 제한, 연속 검색, source 측정과 재검산 비용을 구분했습니다. 표를 만드는 데 사용한 모든 SPICE 값을 각 온라인 방법에 제공하지 않았습니다. 원래의 연속 검색에서 같은 설정을 별칭으로 저장한 행도 있으며 독립 실행 환산 비용에 포함됐습니다. 정확히 같은 측정은 캐시에서 재사용했습니다. 연속 검색 원시 자료와 수치감사 결과는 아래에 있습니다.","","| 구간 | 시행 | 독립 실행 환산 SPICE |","|---|---:|---:|"]
    for phase,values in audit["phase_counts"].items():s.append(f'| {phase} | {values["trials"]} | {values["standalone"]} |')
    from audit import checks
    s+=["","## 고정 후보표가 포함한 실제 통과 설계 수","","각 행은 정책 시행 횟수와 별개로, 사전에 만든 140개 설계를 전수 확인한 결과입니다. 0이면 그 후보표에서 어떤 순서를 써도 해당 목표에 도달할 수 없습니다.","","| 회로 | 5% 통과 설계 | 10% 통과 설계 |","|---|---:|---:|"]
    for name in (*DEV,*TEST):
        batch=read(name+".json");ref=batch["reference"]
        ok=[]
        for row in batch["records"]:
            if row["val"] is None:continue
            d,e=checks(row["dev"],ref[:2]);v,f=checks(row["val"],ref[2:])
            if d and v:ok.append(max(e,f))
        s.append(f"| {name} | {sum(x<=.95 for x in ok)} | {sum(x<=.90 for x in ok)} |")
    strict_dev=[r for (phase,k),rows in group.items() if phase=="strict" and k!="strict_selected" for r in rows]
    strict_test=group["strict","strict_selected"]
    s+=["",f"별도 연속 10% 튜닝 구간에서 개발 {len(strict_dev)}시행에 독립 환산 {sum(r['calls'] for r in strict_dev)}회, 고정 설정 테스트 {len(strict_test)}시행에 {sum(r['calls'] for r in strict_test)}회를 사용했습니다. 기존 scalar 개발 실행은 동일 2개 회로 × 3 seed의 이전 폭 궤적과 대조했습니다."]
    for r in group["strict","scalar"]:
        old=json.loads((ROOT.parent/"alarm"/f"{r['circuit']}-scalar-{r['seed']}.json").read_text())
        assert old['hits']==r['hits'] and [s['x'] for s in old['trace']]==[s['x'] for s in r['trace']]
    physical=audit["physical"]
    s+=["",f'소스 학습은 기존 324회입니다. 이 연구에 저장된 고유 타깃·개발 단일 동작점 측정은 {sum(v["unique"] for v in physical.values()):,}개이고, 이전 실험 캐시에서 읽은 것은 {sum(v["reused"] for v in physical.values()):,}개입니다. 이는 동일점 동시 실행을 식별할 수 없는 고유 키 기준입니다. 실패 측정은 {sum(v["errors"] for v in physical.values())}개입니다.',
        "",f'모든 기록의 기준·후보·확인 호출 장부를 1 ps 원시값에서 다시 계산했습니다. 최초 도달에 쓰인 서로 다른 후보 {audit["unique_first_hits"]}개를 0.25 ps로 다시 실행했고 기준을 포함한 별도 수치 검산 비용은 {audit["fine_calls"]}회입니다. 재검산에서 원래 목표를 놓친 기록은 {len(audit["failures"])}건입니다. 이는 재검산 시점의 재최적화 결과가 아닙니다.',
        "","[설정과 평가 규약](WARM_PROTOCOL.md) · [전체 표](runs/warm/summary.csv) · [민감도](runs/warm/sensitivity.csv) · [감사](runs/warm/audit.json) · [탐색 궤적](warm_traces.zip) · [원시 SPICE](warm_spice.zip)"]
    Path("WARM_RESULTS_KO.md").write_text("\n".join(s)+"\n")

def package():
    for output,physical in [("warm_spice.zip",True),("warm_traces.zip",False)]:
        with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=9) as z:
            if physical:
                for d in sorted((ROOT/"raw").iterdir()):
                    z.writestr("raw/"+d.name+".jsonl",b"\n".join(p.read_bytes() for p in sorted(d.glob("*.json")))+b"\n")
                z.writestr("finer.jsonl",b"\n".join(p.read_bytes() for p in sorted((ROOT/"fine").glob("*.json")))+b"\n")
            else:
                for p in sorted(ROOT.iterdir()):
                    if p.is_file():z.write(p,p.name)
        with ZipFile(output) as z:assert z.testzip() is None

if __name__=="__main__":
    report()
    package()
