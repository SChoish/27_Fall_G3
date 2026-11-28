"""Summarize matched, fine-confirmed hits from the three independently replayable archives."""
from collections import Counter, defaultdict
import json
from statistics import mean
from zipfile import ZipFile

import numpy as np
from audit import checks
from circuit import ratios
from experiment import upper
from ppo_sizing import action_features, observed_features, rank_candidates, train

from ppo_sizing import ARCHIVE, DOMAINS, GOALS, source_for
from ppo_critic_buffer import ARCHIVE as BUFFER_ARCHIVE
from ppo_standalone import ARCHIVE as STANDALONE_ARCHIVE

ARMS = ("mlp200", "standalone", "bandit", "ppo", "ppo_buffer")
NAMES = ("MLP + BO", "단독 PPO", "즉시보상 bandit", "PPO", "PPO + value critic buffer")


def read(archive, key):
    with ZipFile(archive) as f:
        return json.loads(f.read(key + ".json"))


def proposal_origins(index,trials):
    """Reconstruct the first post-warmup selection for each arm from the common history."""
    sources=read(ARCHIVE,"source")
    models={p:train(rows,only=("anon400",))["anon400"] for p,rows in sources.items()}
    weights={}
    for archive in (ARCHIVE,BUFFER_ARCHIVE,STANDALONE_ARCHIVE):
        for p,arms in read(archive,"training").items():
            weights.update({(p,a):np.array(v["weights"]) for a,v in arms.items()})
    origin=defaultdict(list)
    for t in trials:
        name,seed,goal=t
        r=index[t+("mlp200",)]
        if goal!=.95 or len(r["trace"])<5:
            continue
        p=source_for(name)
        perm=r["perm"]
        hi=np.log(upper(DOMAINS[name])[perm]);lo=np.full_like(hi,np.log(.58))
        ref=r["ref"][:2]
        z=np.array([np.zeros_like(hi)]+[np.log(np.array(s["x"])[perm]) for s in r["trace"][:4]])
        y=np.array([ratios(ref,ref)]+[ratios(s["dev"],ref) for s in r["trace"][:4]])
        def collect(candidates,pred,sigma,score):
            base,features=action_features(pred,sigma,score,goal,len(z))
            _,solo=observed_features(z,y,candidates,lo,hi,goal)
            for arm in ARMS:
                if arm=="mlp200": logit=base
                elif arm=="standalone": logit=solo@weights[p,arm]
                else: logit=base+features@weights[p,arm]
                choice=int(np.argmax(logit))
                width=np.round(np.exp(candidates[choice])[np.argsort(perm)],6)
                np.testing.assert_allclose(width,index[t+(arm,)]["trace"][4]["x"],atol=1e-6,rtol=0)
                origin[p,arm].append(choice>=256)
        rank_candidates(z,y,lo,hi,np.random.default_rng([seed,len(z),6481]),
                        network=models[p],decision=collect)
    print("\n5% 목표의 웜업 직후 선택이 전역 후보 256개에서 나온 횟수 (나머지는 국소 후보)")
    for p,label in (("sky","SKY130 → GF180"),("gf","GF180 → SKY130")):
        print(f"- {label}에서 " + ", ".join(f"{name} {sum(origin[p,a])}/{len(origin[p,a])}"
                                          for a,name in zip(ARMS,NAMES)))


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
    print("\n동일 시작점에서 MLP + BO와 비교한 5% 제한비용 (감소 · 동률 · 증가)")
    for method,label in zip(ARMS[1:],NAMES[1:]):
        pairs = [(index[t + ("mlp200",)]["calls"], index[t + (method,)]["calls"])
                 for t in trials if t[2] == .95]
        print(f"- {label}는 {sum(b<a for a,b in pairs)} · {sum(b==a for a,b in pairs)} · {sum(b>a for a,b in pairs)}")
    print("\n## value critic 선택 기록")
    train = read(BUFFER_ARCHIVE, "training")
    for p, methods in train.items():
        counts = methods["ppo_buffer"]["critic_age_counts"]
        print(f"- {p.upper()} 소스에서 최신 · 1회 전 · 2회 전 · 3회 전 선택은 {counts}회. 이전 value critic 재사용은 {sum(counts[1:])}/{sum(counts)}회.")
    print("\n## 같은 이력에서 첫 정책 제안의 개발 코너 통과 수")
    print("기준점 및 네 웜업 뒤 실제로 다음 후보를 제안한 대응 사례만 센다. 이후 궤적은 알고리즘마다 달라진다.")
    print("| 목표 | 비교 수 | " + " | ".join(NAMES) + " |")
    print("|---|---:|" + "---:|" * len(ARMS))
    for g in GOALS:
        ts = [t for t in trials if t[2] == g and all(len(index[t+(a,)]["trace"])>4 for a in ARMS)]
        values = []
        for arm in ARMS:
            outcomes = []
            for t in ts:
                r = index[t+(arm,)]
                outcomes.append(checks(r["trace"][4]["dev"],r["ref"][:2]))
            values.append(str(sum(ok and edp<=g for ok,edp in outcomes)))
        print(f"| {1-g:.0%} | {len(ts)} | " + " | ".join(values) + " |")
    proposal_origins(index,trials)
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
