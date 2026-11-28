"""Compare direct PPO routing to nonlearned policies on the same tasks and seeds."""
from collections import Counter
import json
from statistics import mean
from zipfile import ZipFile
from direct_surrogate_gate_run import ARCHIVE as DIRECT
from surrogate_gate_run import ARCHIVE as BINARY, NEW
from rollout_surrogate_gate import ARCHIVE as ROLLOUT
from surrogate_gate_report import LABELS

NAMES={**LABELS,"random_snapshot":"버퍼 head 무작위","ppo_snapshot":"버퍼 head 직접 PPO","rollout_gate":"대응 return 정책 개선"}


def main():
    with ZipFile(BINARY) as z:
        rows=json.loads(z.read("runs.json"))
    with ZipFile(DIRECT) as z:
        rows.extend(json.loads(z.read("runs.json")))
        actions=json.loads(z.read("actions.json"))
    with ZipFile(ROLLOUT) as z:
        rows.extend(json.loads(z.read("runs.json")))
        rollout_actions=json.loads(z.read("actions.json"))
    idx={(r["name"],r["seed"],r["goal"],r["arm"]):r for r in rows}
    assert len(idx)==len(rows)==378
    for n,s,g in {k[:3] for k in idx}:
        history=[x["x"] for x in idx[n,s,g,"fresh"]["trace"][:4]]
        assert all([x["x"] for x in idx[n,s,g,a]["trace"][:4]]==history for a in NAMES)
    print("## 직접 head 선택을 포함한 전체 비교\n")
    print("| 방법 | 신규 5% | 신규 10% | 기존 5% | 기존 10% |")
    print("|---|---:|---:|---:|---:|")
    for a,label in NAMES.items():
        cells=[]
        for new,g in ((True,.95),(True,.90),(False,.95),(False,.90)):
            matched=[r for r in rows if (r["name"] in NEW)==new and r["goal"]==g and r["arm"]==a]
            cells.append(f"{sum(r['hit'] is not None for r in matched)}/{len(matched)} · {mean(r['calls'] for r in matched):.2f}")
        print("| "+" | ".join([label,*cells])+" |")
    print("\n| 타깃 PDK | 목표 | seed | 직접 PPO | 같은 목록에서 무작위 |")
    print("|---|---:|---:|---:|---:|")
    for n,s,g in sorted({k[:3] for k in idx if k[0] in NEW}):
        costs=[idx[n,s,g,a]["hit"] for a in ("ppo_snapshot","random_snapshot")]
        print(f"| {NEW[n].pdk} | {1-g:.0%} | {s} | "+" | ".join(str(v or "미달") for v in costs)+" |")
    print("\n두 개 이상의 서로 다른 폭 제안이 있었을 때 직접 PPO가 택한 head 종류와 횟수\n")
    for key,table in actions.items():
        active=[r for r in table["ppo_snapshot"] if r["distinct"]>1]
        counts=Counter("소스" if r["end"]==0 else "최신" if r["end"]==r["observed"] else "과거" for r in active)
        print(f"- {key}에서 소스 {counts['소스']}회, 최신 {counts['최신']}회, 과거 {counts['과거']}회였다.")
    print("\n## 신규 회로에서 실제로 선택이 달라진 횟수\n")
    print("두 제안이 다른 경우에 한해, 해당 방법 자신의 방문 상태에서 센 횟수다. 직접 PPO는 서로 다른 폭이 적어도 2개인 결정을 사용한다.\n")
    with ZipFile(BINARY) as z:
        binary_actions=json.loads(z.read("actions.json"))
    for goal in (.95,.90):
        print(f"- {1-goal:.0%} 목표")
        for a in ("coin","myopic_gate","ppo_gate","rollout_gate"):
            decisions=[e for n in NEW for e in (rollout_actions[f"{n}_{goal}"] if a=="rollout_gate" else binary_actions[f"{n}_{goal}"][a]) if e["distinct"]]
            print(f"  - {NAMES[a]} 정책은 소스 복귀 제안을 {sum(e['chosen'] for e in decisions)}/{len(decisions)}회 선택했다.")
        ds=[e for n in NEW for e in actions[f"{n}_{goal}"]["ppo_snapshot"] if e["distinct"]>1]
        print(f"  - 직접 PPO의 소스 원본·최신·과거 선택은 {sum(e['end']==0 for e in ds)} · "
              f"{sum(e['end']==e['observed'] for e in ds)} · {sum(0<e['end']<e['observed'] for e in ds)}회였다.")
    print(f"\n총 {len(rows)}개 실행의 독립 예산 합계는 {sum(r['calls'] for r in rows):,}회다.")


if __name__=="__main__":
    main()
