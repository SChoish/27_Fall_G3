"""Summarize matched, fine-confirmed hits from the three independently replayable archives."""
from collections import Counter
import json
from statistics import mean
from zipfile import ZipFile

from ppo_sizing import ARCHIVE, DOMAINS, GOALS, source_for
from ppo_critic_buffer import ARCHIVE as BUFFER_ARCHIVE
from ppo_standalone import ARCHIVE as STANDALONE_ARCHIVE

ARMS = ("mlp200", "standalone", "bandit", "ppo", "ppo_buffer")
NAMES = ("MLP + BO", "단독 PPO", "즉시보상 bandit", "PPO", "PPO + critic buffer")


def read(archive, key):
    with ZipFile(archive) as f:
        return json.loads(f.read(key + ".json"))


def summarize():
    rows = read(ARCHIVE, "runs") + read(BUFFER_ARCHIVE, "runs") + read(STANDALONE_ARCHIVE, "runs")
    index = {(r["name"], r["seed"], r["goal"], r["arm"]): r for r in rows}
    assert len(index) == len(rows) == 120
    trials = sorted({key[:3] for key in index})
    for t in trials:
        histories = [index[t + (a,)]["trace"] for a in ARMS]
        assert all([s["x"] for s in h[:4]] == [s["x"] for s in histories[0][:4]] for h in histories)
    print("| 소스 → 타깃 | EDP 절감 | " + " | ".join(NAMES) + " |")
    print("|---|---:|" + "---:|" * len(ARMS))
    for pdk, label in (("sky","SKY130 → GF180"), ("gf","GF180 → SKY130"), (None,"전체")):
        for g in GOALS:
            subset = [t for t in trials if t[2] == g and (pdk is None or source_for(t[0]) == pdk)]
            cells = []
            for arm in ARMS:
                hits = [index[t + (arm,)]["hit"] for t in subset]
                cells.append(f"{sum(h is not None for h in hits)}/{len(hits)} · {mean(h or 100 for h in hits):.2f}")
            print(f"| {label} | {1-g:.0%} | " + " | ".join(cells) + " |")
    print("\n단일 PPO와 buffer의 동일 회로·seed별 비교에서 제한비용이 감소·동률·증가한 횟수")
    for g in GOALS:
        pairs = [(index[t + ("ppo",)]["calls"], index[t + ("ppo_buffer",)]["calls"])
                 for t in trials if t[2] == g]
        print(f"- {1-g:.0%} 목표에서 {sum(b<a for a,b in pairs)} · {sum(b==a for a,b in pairs)} · {sum(b>a for a,b in pairs)}")
    print("\n## Critic 선택 기록")
    train = read(BUFFER_ARCHIVE, "training")
    for p, methods in train.items():
        counts = methods["ppo_buffer"]["critic_age_counts"]
        print(f"- {p.upper()} 소스에서 최신 · 1회 전 · 2회 전 · 3회 전 선택은 {counts}회. 이전 critic 재사용은 {sum(counts[1:])}/{sum(counts)}회.")
    print("\n## 개별 최초 정밀 통과 SPICE 횟수")
    print("| 타깃 | seed | EDP 절감 | " + " | ".join(NAMES) + " |")
    print("|---|---:|---:|" + "---:|" * len(ARMS))
    for t in trials:
        hits = [str(index[t + (arm,)]["hit"] or "미달") for arm in ARMS]
        print(f"| {t[0]} | {t[1]} | {1-t[2]:.0%} | " + " | ".join(hits) + " |")
    print("\n## 측정 비용")
    print(f"타깃 독립 실행 기준 과금 합계는 {sum(r['calls'] for r in rows)}회다. "
          f"기존 기준선은 {sum(r['calls'] for r in rows if r['arm']=='mlp200')}회이며 원자료를 재사용했다. "
          "공통 기준 및 같은 설계·코너의 기존 측정도 각 정책의 독립 비용에 포함했다.")
    for name in DOMAINS:
        changed = Counter()
        for t in trials:
            if t[0] != name:
                continue
            baseline = index[t + ("mlp200",)]["trace"]
            for arm in ARMS[1:]:
                trace = index[t + (arm,)]["trace"]
                changed[arm] += any(x["x"] != y["x"] for x,y in zip(baseline,trace))
        print(f"- {name}에서 같은 seed·목표의 BO와 한 번이라도 다른 폭을 선택한 횟수 (6쌍) "
              + ", ".join(f"{arm} {changed[arm]}" for arm in ARMS[1:]))


if __name__ == "__main__":
    summarize()
