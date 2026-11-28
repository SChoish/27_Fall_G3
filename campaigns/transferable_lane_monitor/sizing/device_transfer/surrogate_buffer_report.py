"""Summarize verified first hits and one-step forecasts on fixed baseline trajectories."""
from collections import defaultdict
import json
from statistics import mean
from zipfile import ZipFile

import numpy as np
from blind_width_policy import kernel
from circuit import ratios
from experiment import upper
from ppo_sizing import DOMAINS, GOALS, source_for
from surrogate_buffer import ARMS, WINDOW, margins, select_head
from surrogate_buffer_run import ARCHIVE, models

NAMES = dict(zip(ARMS,("소스 MLP 고정","최신 전체","최신 적격","무작위","평균 오차","목표 경계 가중 오차","설계 순위","거짓 통과 비용","소스 복귀")))


def score(rows):
    hits = [r["hit"] for r in rows]
    return f"{sum(h is not None for h in hits)}/{len(rows)} · {mean(h or 100 for h in hits):.2f}"


def diagnostics(sources,runs):
    """All heads score the very same next design on each archived frozen-MLP path."""
    trained = models(sources)
    stats = defaultdict(list)
    for r in runs:
        if r["arm"]!="mlp200" or r["goal"]!=.95:
            continue
        p = np.asarray(r["perm"])
        y0 = ratios(r["ref"][:2],r["ref"][:2])
        net = trained[source_for(r["name"])]
        lo = np.full(len(p),np.log(.58))
        hi = np.log(upper(DOMAINS[r["name"]])[p])
        trace = r["trace"]
        z = np.vstack((np.zeros(len(p)),[np.log(np.array(s["x"])[p]) for s in trace]))
        y = np.vstack((y0,[ratios(s["dev"],r["ref"][:2]) for s in trace]))
        feat = net.encode(z)
        base = y0 + net.readout(feat)
        u = (z-lo)/(hi-lo)
        for stop in range(5,len(y)):
            w = np.linalg.solve(kernel(u[:stop],u[:stop])+.004*np.eye(stop),kernel(u[stop:stop+1],u[:stop]).T).T[0]
            for arm in ARMS:
                if arm=="mlp200":
                    biased=base[:stop+1]
                else:
                    head,_=select_head(arm,net,z[:stop],y[:stop],lo,hi,.95,r["seed"])
                    biased=base[:stop+1]+feat[:stop+1]@head
                pred=biased[stop]+w@(y[:stop]-biased[:stop])
                err = np.mean(np.abs(pred-y[stop])/np.maximum(.02,net.scale))
                predicted_pass = (margins(pred[None,:],.95) <= 0).all()
                actual_pass = (margins(y[stop:stop+1],.95) <= 0).all()
                stats[arm].append((err, predicted_pass and not actual_pass, actual_pass and not predicted_pass))
    return stats


def main():
    with ZipFile(ARCHIVE) as z:
        sources,runs,decisions=[json.loads(z.read(k+".json")) for k in ("source","runs","decisions")]
    print("## 정밀 검증까지의 성공 수와 제한평균 SPICE 횟수\n")
    print("| 방법 | 5% 전체 | SKY130 → GF180 | GF180 → SKY130 | 10% 전체 | SKY130 → GF180 | GF180 → SKY130 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        cells=[]
        for g in GOALS:
            rows=[r for r in runs if r["arm"]==arm and r["goal"]==g]
            cells.extend(score([r for r in rows if p is None or source_for(r["name"])==p]) for p in (None,"sky","gf"))
        print("| "+" | ".join([NAMES[arm],*cells])+" |")
    print("\n## 5% 동일 회로·seed 대응 비교\n")
    print("| 기준 | 최신 적격 대비 빠름/동률/느림 | 소스 고정 대비 빠름/동률/느림 | 여러 head가 있던 결정 중 과거 head 선택 |")
    print("|---|---:|---:|---:|")
    paired={(r["name"],r["seed"],r["arm"]):r["calls"] for r in runs if r["goal"]==.95}
    keys=sorted({k[:2] for k in paired})
    for arm in ARMS[3:]:
        cells=[]
        for base in ("latest","mlp200"):
            diffs=np.array([paired[t+(arm,)]-paired[t+(base,)] for t in keys])
            cells.append("/".join(map(str,(sum(diffs<0),sum(diffs==0),sum(diffs>0)))))
        rows=[item for n in DOMAINS for item in decisions[f"{n}_0.95_{arm}"] if item["observed"]-WINDOW>=3]
        cells.append(f"{sum(0<r['trained']<r['observed']-WINDOW for r in rows)}/{len(rows)}")
        print("| "+" | ".join([NAMES[arm],*cells])+" |")
    for g in GOALS:
        rows=[row for n in DOMAINS for row in decisions[f"{n}_{g}_source_guard"]]
        print(f"\n소스 복귀 기준은 {1-g:.0%} 목표에서 소스 원본을 {sum(r['trained']==0 for r in rows)}/{len(rows)}번 사용했다.")
    print("\n## 같은 다음 설계를 예측하는 진단\n")
    errors=diagnostics(sources,runs)
    print("기준 MLP의 5% 평가 궤적에서 다음 후보를 관측하기 직전, 같은 이력과 후보에 대한 다섯 지표의 소스 스케일 정규화 절대오차를 비교했다. 개발 두 코너만 사용했다. 측정점에 같은 가중치를 둔 기술통계다.\n")
    print("| 방법 | 다음 설계 수 | 평균 예측오차 | 설계 전체 거짓 통과 | 설계 전체 거짓 탈락 |")
    print("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        mae, fp, fn = np.array(errors[arm]).T
        print(f"| {NAMES[arm]} | {len(mae)} | {mae.mean():.4f} | {int(fp.sum())} | {int(fn.sum())} |")
    print("\n각 방법을 독립 실행으로 계산한 타깃 SPICE 예산 합계는 "
          f"{sum(r['calls'] for r in runs):,}회다. 동일 측정 재사용분도 각 방법의 비용에 포함했다.")


def transfer_report():
    from surrogate_buffer_transfer import ARCHIVE as holdout_archive, ARMS as holdout_arms
    with ZipFile(holdout_archive) as z:
        rows=json.loads(z.read("runs.json"))
    print("\n## 신규 3출력 회로의 5% 목표\n")
    print("| 방법 | SKY130 → GF180 | GF180 → SKY130 | 전체 |")
    print("|---|---:|---:|---:|")
    for arm in holdout_arms:
        group=[r for r in rows if r["arm"]==arm]
        cells=[score([r for r in group if r["name"].endswith(p)]) for p in ("gf","sky")]
        print("| "+" | ".join([NAMES[arm],*cells,score(group)])+" |")
    print("\n| 타깃 | seed | "+" | ".join(NAMES[a] for a in holdout_arms)+" |")
    print("|---|---:|---:|---:|---:|")
    for name,seed in sorted({(r["name"],r["seed"]) for r in rows}):
        hits={r["arm"]:r["hit"] for r in rows if r["name"]==name and r["seed"]==seed}
        print("| "+" | ".join([name,str(seed),*(str(hits[a] or "미달") for a in holdout_arms)])+" |")


if __name__=="__main__":
    main()
    transfer_report()
