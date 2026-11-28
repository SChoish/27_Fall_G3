"""Report prospectively specified and prior-circuit first-hit costs and routing behavior."""
import json
from statistics import mean
from zipfile import ZipFile

from surrogate_gate_run import ARCHIVE, ARMS, NEW

LABELS={"mlp200":"소스 MLP 고정","fresh":"최신 head 고정","source_guard":"소스 복귀 규칙 고정",
        "coin":"균등 무작위 전환","myopic_gate":"즉시보상 전환","ppo_gate":"PPO 전환"}


def summarize(rows,actions,trained):
    index={(r["name"],r["seed"],r["goal"],r["arm"]):r for r in rows}
    assert len(index)==len(rows)
    print("## 최초 정밀 통과까지의 타깃 SPICE 비용\n")
    print("각 셀은 성공 수/실행 수와 100회 제한평균 비용이다. 기존 조건에서 5%는 6조건, 10%는 4조건이다.\n")
    print("| 방법 | 신규 5% | 신규 10% | 기존 5% | 기존 10% |")
    print("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        cells=[]
        for new,goal in ((True,.95),(True,.90),(False,.95),(False,.90)):
            sample=[r for r in rows if (r["name"] in NEW)==new and r["goal"]==goal and r["arm"]==arm]
            cells.append(f"{sum(r['hit'] is not None for r in sample)}/{len(sample)} · {mean(r['calls'] for r in sample):.2f}")
        print("| "+" | ".join([LABELS[arm],*cells])+" |")
    print("\n## 신규 회로 개별 최초 통과 횟수\n")
    print("| 타깃 PDK | EDP 절감 | seed | "+" | ".join(LABELS[a] for a in ARMS)+" |")
    print("|---|---:|---:|"+"---:|"*len(ARMS))
    for name,seed,g in sorted({k[:3] for k in index if k[0] in NEW}):
        hits=[index[name,seed,g,a]["hit"] for a in ARMS]
        print(f"| {NEW[name].pdk} | {1-g:.0%} | {seed} | "+" | ".join(str(h) if h is not None else "미달" for h in hits)+" |")
    print("\n## 두 surrogate의 제안이 달랐을 때의 선택\n")
    print("| 타깃 | 목표 | 전체 결정 | 서로 다른 제안 | PPO의 소스 복귀 제안 선택 | 즉시보상 정책의 소스 복귀 제안 선택 |")
    print("|---|---:|---:|---:|---:|---:|")
    for key,table in actions.items():
        name,goal=key.rsplit("_",1)
        ppo=table["ppo_gate"]
        cells=[]
        for arm in ("ppo_gate","myopic_gate"):
            distinct=[r for r in table[arm] if r["distinct"]]
            cells.append(f"{sum(r['chosen'] for r in distinct)}/{len(distinct)}")
        print(f"| {name} | {1-float(goal):.0%} | {len(ppo)} | {sum(r['distinct'] for r in ppo)} | "+" | ".join(cells)+" |")
    print("\n## 대응 비교\n")
    print("| 타깃 | 목표 | PPO 대비 기준 | PPO 빠름/동률/느림 |")
    print("|---|---:|---|---:|")
    for new,g in ((True,.95),(True,.90),(False,.95),(False,.90)):
        keys=sorted({k[:3] for k in index if (k[0] in NEW)==new and k[2]==g})
        for a in ("fresh","source_guard","coin","myopic_gate"):
            diffs=[index[t+("ppo_gate",)]["calls"]-index[t+(a,)]["calls"] for t in keys]
            print(f"| {'신규' if new else '기존'} | {1-g:.0%} | {LABELS[a]} | "
                  f"{sum(d<0 for d in diffs)}/{sum(d==0 for d in diffs)}/{sum(d>0 for d in diffs)} |")
    print("\n## 소스 학습\n")
    for p,arms in trained.items():
        for a,group in arms.items():
            print(f"- {p}, {LABELS[a]}의 seed별 웜업 통과 / 이후 통과 / 비강제 행동 수는 "
                  +", ".join(f"{r['warm_hits']}/{r['policy_hits']}/{r['decisions']}" for r in group)+"이다.")
    print(f"\n{len(rows)}개 독립 실행 비용의 합은 {sum(r['calls'] for r in rows):,} SPICE eval이다. "
          "공통 기준과 동일 측정 캐시 재사용은 각 정책의 독립 실행 비용에 포함했다.")


if __name__=="__main__":
    with ZipFile(ARCHIVE) as z:
        summarize(*(json.loads(z.read(k+".json")) for k in ("runs","actions","training")))
