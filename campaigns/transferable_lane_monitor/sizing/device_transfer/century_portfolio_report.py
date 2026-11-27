"""Report paired results of the exploratory portfolio and its independent test."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

def main():
    root=Path("runs/century")
    rows=list(csv.DictReader((root/"portfolio_summary.csv").open()))
    ix={(r["domain"],r["method"],float(r["goal"]),int(r["cap"])):r for r in rows}
    def value(n,m,g,cap):
        r=ix[n,m,g,cap]
        return f"{r['successes']}/{r['n']} · {float(r['mean']):.2f}"
    groups=[("previous","선택에 사용한 6조건"),("new","고정 후 신규 2조건"),("long100_sky","신규 SKY130"),("long100_gf","신규 GF180")]
    names={"role":"역할 규칙 + 직접 탐색","warm4":"웜업 4회 + 잔차 critic","prefix_warm4":"역할 2회 → 웜업 4회 + 잔차 critic"}
    tab=["| 조건 | 방법 | 5% ≤40 | 5% ≤100 | 10% ≤40 | 10% ≤100 |","|---|---|---|---|---|---|"]
    for n,label in groups:
        for m,title in names.items():
            tab.append("| "+" | ".join([label,title,*(value(n,m,g,v) for g in (.95,.9) for v in (40,100))])+" |")
    with ZipFile("century_portfolio_measurements.zip") as z:
        records=json.loads(z.read("records.json"))
        fine=json.loads(z.read("fine.json"))
    new=[r for r in records if r["circuit"].startswith("long100")]
    calls={}
    for r in new:
        n=r["circuit"]
        for k,raw in enumerate(r["ref"]):calls[n,(1.,)*len(r["trace"][0]["x"]),k]=raw
        for step in r["trace"]:
            for k,raw in enumerate(step["dev"]+(step["validation"] or [])):
                key=n,tuple(step["x"]),k
                assert calls.setdefault(key,raw)==raw
    audit=json.loads((root/"portfolio_audit.json").read_text())
    by={(r["circuit"],r["method"],r["seed"],r["goal"]):r["hit"] for r in records}
    compare=[]
    for fresh in (False,True):
        for goal in (.95,.9):
            diff=[(t or 100)-(by[n,"warm4",s,g] or 100) for (n,m,s,g),t in by.items()
                  if n.startswith("long100")==fresh and m=="prefix_warm4" and g==goal]
            compare.append([fresh,goal,sum(v<0 for v in diff),sum(v==0 for v in diff),sum(v>0 for v in diff),np.mean(diff)])
    ext=json.loads((root/"precision_continuation.json").read_text())
    charged=list(csv.DictReader((root/"charged_precision.csv").open()))
    rigorous=["| 방법 | 5% 정밀 검산 별도 | 5% 정밀 검산 포함 | 10% 정밀 검산 별도 | 10% 정밀 검산 포함 |","|---|---|---|---|---|"]
    for m,title in names.items():
        cells=[]
        for goal in (.95,.90):
            r=next(v for v in ext["summary"] if v["domain"]=="new" and v["method"]==m and v["goal"]==goal)
            q=next(v for v in charged if v["domains"]=="fresh_two" and v["method"]==m
                   and float(v["goal"])==goal and int(v["cap"])==100)
            cells.extend([f"{r['hits']}/{r['n']} · {r['capped_mean']:.2f}",f"{q['hits']}/{q['n']} · {float(q['mean']):.2f}"])
        rigorous.append("| "+" | ".join([title,*cells])+" |")
    lines=["# 역할 제안 두 번을 먼저 쓰는 조합의 독립 전이 검증",
           "개발 결과를 본 뒤 결정한 조합이다. 첫 제안은 출력 폭을 nominal의 0.84배로 줄인다. 두 번째는 측정한 개발 전기 제약에 맞춰 같은 방향의 보폭을 조절한다. 이후 기존 네 국소 웜업과 잔차 critic을 사용한다. 같은 설계를 만나면 기존 개발 관측을 반환한다. 모든 실행은 4회 기준 측정을 포함해 최대 100회다.",
           "신규 회로는 6단과 2단 버퍼를 결합한 2출력 구조다. 정책·회로·부하·seed를 고정하고 SKY130과 GF180을 평가했다. 독립 구조 한 가지, 두 PDK, 확률적 방법당 3 seed다. 표 안의 값은 성공 수/시행 수와 제한 평균 최초 도달 비용이다.",
           "## 신규 두 PDK 조건의 정밀 확인 결과",
           "\n".join(rigorous),
           "신규 SKY130의 seed 53에서 같은 폭의 최악 EDP가 1 ps에서 0.94992289, 0.25 ps에서 0.95003132로 측정됐다. 두 방법의 원래 최초 통과가 경계에서 뒤집혀 100회까지 탐색을 재개했다. 다음 설계는 양쪽 정밀도에서 통과했다. 웜업 단독은 탐색 86→90회, 조합은 88→92회다. 마지막 설계에 추가로 새로 수행한 정밀 transient는 4회다. 모든 정밀 비용을 같은 예산 안에서 지불하면 실패 후보와 성공 후보를 모두 확인해야 한다. 정밀 기준 4회와 두 설계 8회를 더한 102회와 104회는 상한 100을 넘으므로 이 두 시행은 미달로 집계한다.",
           "## 원래 1 ps 평가의 최초 도달",
           "\n".join(tab),
           "5%와 10%를 각각 직접 겨냥한다. 확률적 설계와 결정적 역할 정책의 성공 수는 분모가 다르며 모든 평균에서 각 회로 조건의 가중치는 같다.",
           "## 같은 seed에서 조합과 웜업 단독의 차이",
           "| 조건 | 목표 | 조합 승/무/패 | 조합 − 웜업 평균 호출 |\n|---|---|---|---|\n"+"\n".join(
           f"| {'신규 2조건' if fresh else '기존 6조건'} | {100*(1-goal):.0f}% | {w}/{t}/{l} | {d:+.2f} |" for fresh,goal,w,t,l,d in compare),
           f"새 2조건은 총 {len(calls):,}개의 서로 다른 타깃 단일 코너 측정을 포함한다. 개발·확인 원시값에서 {audit['count']}개 정책 궤적을 재생했고 최초 도달 기록 {audit['hits']}건, 고유 도달 설계 {audit['unique_witnesses']}개를 확인했다. 기준을 포함해 정밀도 검산에 대응하는 단일 코너 기록은 {4*len(fine)}개다. 0.25 ps 검산 미통과는 {len(audit['fine_failures'])}건이다. 기존 6조건의 기준 정책과 주 실험은 같은 데이터를 공유한다.",
           "기존 조합은 저장된 역할·웜업 정책의 제안 순서를 실제 비용 장부에 따라 합성했다. 신규 회로에서도 각 기초 정책을 ngspice와 폐루프로 실행하고, 같은 실측값을 쓰는 고정된 조합의 궤적을 재구성했다. 확인 결과는 종료 판정에만 사용한다.",
           "[고정 규약](PORTFOLIO_PROTOCOL.md) · [1 ps 수치 CSV](runs/century/portfolio_summary.csv) · [정밀 검증을 비용에 포함한 결과](runs/century/charged_precision.csv) · [경계 후보 재탐색 기록](runs/century/precision_continuation.json) · [원시 기록과 궤적](century_portfolio_measurements.zip) · [주 실험](CENTURY_RESULTS_KO.md)"]
    Path("PORTFOLIO_RESULTS_KO.md").write_text("\n\n".join(lines)+"\n")
if __name__=="__main__":main()
