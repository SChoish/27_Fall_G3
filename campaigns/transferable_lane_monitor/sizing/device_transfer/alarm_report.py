"""Publish Korean result tables and a restorable archive of the SPICE traces."""
import ast
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

from audit_alarm import ALL, GOALS
from alarm_pilot import TARGETS

NAMES=dict(role="기존 역할 규칙",role40="역할 규칙 (40회)",bo="동일 초기 제안 BO",
    single="시작점 1개 + 일반 경사",elite="Elite + 일반 경사",scalar="Elite + ALARM 균일 대각 가중치",
    attention="Elite + ALARM 학습 대각",pullback="Elite + 다중 지표 metric",
    random_buffer="학습 대각 + 무작위 buffer",scored_buffer="학습 대각 + 점수 buffer",
    keep_head="encoder 고정 + head 유지",reset_head="encoder 고정 + head 초기화",
    prefix_attention="역할 제안 2회 + 학습 대각")
NEW=("load_sky","seven_sky","seven_gf")
root=Path("runs/alarm")

def report():
    rows=list(csv.DictReader((root/"summary.csv").open()))
    data={(r["circuit"],r["method"],float(r["goal"])):r for r in rows}
    def values(names,m,g):
        subset=[data[n,m,g] for n in names]
        return (sum(int(r["hits"]) for r in subset),sum(int(r["seeds"]) for r in subset),
                sum(float(r["mean_capped"]) for r in subset)/len(subset))
    def cell(names,m,g):
        a,b,v=values(names,m,g)
        return f"{a}/{b} · {v:.1f}"
    curve=[]
    for group,domains in (("all",tuple(TARGETS)),("new",NEW)):
        for m in ALL:
            for goal in GOALS:
                for stop in (8,10,12,16,20,24,32,40):
                    vals=[]
                    for n in domains:
                        ts=ast.literal_eval(data[n,m,goal]["times"])
                        vals.append(sum(t is not None and t<=stop for t in ts)/len(ts))
                    curve.append(dict(domains=group,method=m,goal=goal,calls=stop,
                                      reached_fraction=sum(vals)/len(vals)))
    with (root/"hit_curves.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=curve[0])
        writer.writeheader()
        writer.writerows(curve)
    ledger=json.loads((root/"audit.json").read_text())
    fine=json.loads((root/"finer.json").read_text())
    s=["# ALARM 구성요소의 목표 도달 비용", "",
       "표는 성공 수/시행 수와 40회 제한 평균 SPICE 호출 수입니다. 미도달도 40회로 포함했으며 실제 40회째 성공은 성공 수에 포함합니다. 회로에 동일 가중치를 주었습니다. 목표는 모든 출력·동작점에서 EDP를 기준 대비 각각 5%, 10% 줄이면서 지연·slew·레일·전원 민감도 제약을 만족하는 것입니다. 최초 4회는 기준, 개발 2회와 확인 2회를 포함해 최솟값은 8회입니다.", "",
       "## 방법별 결과","","| 방법 | 전체 5% | 새 회로 조건 5% | 전체 10% | 새 회로 조건 10% |","|---|---:|---:|---:|---:|"]
    s[2:2]=["5% 목표에서 다섯 조건의 제한 평균은 elite + ALARM 균일 대각 가중치가 19.20회로 가장 낮았고, 새 세 조건에서는 elite + 일반 경사가 21.33회로 가장 낮았습니다. 10% 목표는 다중 지표 metric이 34.40회로 가장 낮았으나 성공은 4/15회였고 모두 GF180에서 나왔습니다. 두 성과는 서로 다른 목표에 대한 것입니다.","","같은 일반 경사에 elite 시작점을 추가하면 새 조건에서 5% 도달 비용이 27.33회에서 21.33회로 줄었습니다. 학습 대각과 과거 SPICE surrogate의 점수 재사용은 5% 전체 평균을 개선하지 못했습니다. 기준 역할 규칙과 ALARM 균일 대각 방법의 5% 전체 타깃 비용차는 3.60회입니다. 같은 회로 분포와 평균 절감량을 가정하면 324회 소스 비용을 새로 지불하는 경우 약 90건의 타깃이 필요합니다. 이미 소스 데이터를 보유했다면 신규 타깃의 추가 비용을 따로 볼 수 있습니다.",""]
    for m in ALL:s.append("| "+" | ".join([NAMES[m],cell(TARGETS,m,.95),cell(NEW,m,.95),cell(TARGETS,m,.9),cell(NEW,m,.9)])+" |")
    s+=["","새 조건은 기존 SKY130 결합 2-lane의 부하·결합 용량을 바꾼 load_sky와, 동일한 새 7-inverter 배선을 두 PDK에서 평가한 seven_sky·seven_gf입니다. fork와 mixed_gf는 이전 연구에서 진단에 사용했습니다. 세 seed는 공통 회로 내에 중첩되어 있습니다. 결정적 역할 규칙은 조건당 한 번만 실행했습니다.",""]
    for g in GOALS:
        s += [f"## 회로별 {100*(1-g):.0f}% 목표","", "| 방법 | "+" | ".join(TARGETS)+" |","|---|---:|---:|---:|---:|---:|"]
        for m in ALL:s.append("| "+" | ".join([NAMES[m],*[cell([n],m,g) for n in TARGETS]])+" |")
        s.append("")
    contrasts=list(csv.DictReader((root/"contrasts.csv").open()))
    s+=["## 같은 회로 안에서 바꾼 구성요소","","양수는 변경 후 절약한 횟수입니다. 승/무/패는 각 회로에서 seed 평균을 낸 뒤 비교했습니다. 둘 다 미도달한 회로도 무승부에 포함합니다.","","| 기준 → 변경 | 전체 5% 절약 | 5% 승/무/패 | 전체 10% 절약 | 10% 승/무/패 |","|---|---:|---:|---:|---:|"]
    for a,b in dict.fromkeys((r['baseline'],r['variant']) for r in contrasts):
        cells=[]
        for g in GOALS:
            vs=[float(r["calls_saved"]) for r in contrasts if r['baseline']==a and r['variant']==b and float(r["goal"])==g]
            cells.extend([f"{sum(vs)/len(vs):+.2f}","/".join(str(v) for v in (sum(n>1e-6 for n in vs),sum(abs(n)<1e-6 for n in vs),sum(n< -1e-6 for n in vs)))])
        s.append("| "+" | ".join([NAMES[a]+" → "+NAMES[b],*cells])+" |")
    s+=["","## 0.25 ps 재검산","","각 회로와 목표에서 최초 도달이 가장 빨랐던 설계를 재검산했습니다. 동률이면 방법명·seed의 사전순으로 하나를 선택했습니다. 참조 탐색은 네 동작점을 모두 사용해 4개 역할·극성 변수에 대해 유한차분을 수행한 별도 실행입니다. 해당 탐색의 실패만으로 10% 달성 불가능을 판단할 수 없습니다.","","| 회로 | 목표 | 방법 | 최초 도달 또는 참조 탐색 비용 | 재검산 EDP / 기준 | 통과 |","|---|---:|---|---:|---:|---|"]
    for rr in fine:
        for t in rr['checks']:
            edp="전기 제약 초과" if t['edp'] is None else f"{t['edp']:.6f}"
            s.append(f"| {rr['circuit']} | {100*(1-t['goal']):.0f}% | {t['method']} | {t['calls']} | {edp} | {'예' if t['passes'] else '아니요'} |")
    precision=json.loads((root/"all_first_hits.json").read_text())
    tests=precision["tests"]
    failures=[r for r in tests if not r["passes"]]
    s+=["","## 모든 최초 도달 설계 재검산","",
        f"알고리즘·회로·seed·목표별 최초 도달 기록 {len(tests)}개에 대해, 중복 제거한 후보 {precision['unique_candidates']}개를 모두 0.25 ps로 재실행했습니다. {len(tests)-len(failures)}개 기록이 같은 목표를 다시 통과했습니다. 비용은 기준 재계산을 포함해 {precision['spice_calls']}회이며 위 표의 재검산 비용과 별도입니다."]
    if failures:
        s+=["","| 회로 | 방법 | seed | 목표 | 원 호출 수 | 재검산 EDP / 기준 |","|---|---|---:|---:|---:|---:|"]
        for t in failures:
            result="전기 제약 초과" if t["edp"] is None else f"{t['edp']:.6f}"
            s.append(f"| {t['circuit']} | {NAMES[t['method']]} | {t['seed']} | {100*(1-float(t['goal'])):.0f}% | {t['calls']} | {result} |")
    probe=json.loads((root/"asymmetric_load_probe.json").read_text())
    fine_load="전기 제약 초과" if probe["fine_edp"] is None else f"{probe['fine_edp']:.6f}"
    s+=["","## 비대칭 부하의 달성 가능성","",
        f"load_sky에서 같은 역할의 두 lane을 같은 폭으로 묶은 참조 탐색이 목표를 놓친 뒤, 독립적인 8개 폭을 조절하는 사후 참조 탐색을 수행했습니다. 네 동작점 모두 사용한 비용은 {probe['calls']}회이고, 가장 낮은 실측 EDP/기준은 {probe['edp_ratio']:.6f}입니다. 추가 8회로 검산한 값은 {fine_load}입니다. 이 확인은 타깃 알고리즘의 도달 기록에 포함하지 않았습니다."]
    zoom=json.loads((root/"load_oracle_refined.json").read_text())
    zoom_edp="전기 제약 초과" if zoom["fine_edp"] is None else f"{zoom['fine_edp']:.6f}"
    s+=["",f"5% 경계 부근을 별도로 재탐색했습니다. 소자별 제약 선형화와 더 작은 차분폭을 사용한 추가 비용 {zoom['additional_spice_calls']}회, 최종 시간 간격 재검산 {zoom['fine_calls']}회에서 재검산 EDP/기준은 {zoom_edp}였습니다. 기존 두 참조 탐색 기록도 모두 남겼습니다."]
    costs=ledger["cost_by_circuit"].values()
    oracle=sum(json.loads((root/f"witness_{n}.json").read_text())["calls"] for n in TARGETS)
    s+=["","## 비용과 재현","",
        f"- 원시 측정에서 {ledger['trials']}개 시행의 모든 제약 판정과 최초 도달 호출 수를 다시 계산했습니다. 방법별 독립 실행 환산 비용은 {ledger['standalone_calls']:,}회입니다. 타깃 원시 측정 {sum(c['unique'] for c in costs):,}개 중 기존 실측 캐시에서 가져온 것은 {sum(c['reused'] for c in costs)}개이며, 시뮬레이터 오류는 {sum(c['errors'] for c in costs)}개였습니다.",
        f"- 역할 규칙과 BO는 소스 비용 0회, 학습 방법은 같은 SKY130 소스 324회입니다. 참조 탐색 {oracle}회, 0.25 ps 재검산 {sum(r['calls'] for r in fine)}회는 별도입니다."]
    timing=json.loads((root/"timing.json").read_text())
    trained=timing["source_per_seed_seconds"]
    s.append(f"- 저장한 시뮬레이션 값을 메모리에서 읽어 {timing['traces_exact']}개 알고리즘 궤적을 완전히 재생했습니다. seed당 소스 SPICE surrogate 학습은 {trained['fit']:.3f} CPU초, 좌표 가중치 학습은 {trained['source_attention_fit']:.3f} CPU초였습니다. 전체 재생과 방법별 타깃 SPICE surrogate 학습·후보 제안 시간은 [시간 로그](runs/alarm/timing.json)에 있습니다. SPICE 실행 시간은 이 CPU 시간에 포함하지 않았습니다.")
    for m,(all_n,old_n) in ledger["snapshot_selections_and_older"].items():
        s.append(f"- {NAMES[m]}가 snapshot을 초기값으로 선택해 다시 학습한 횟수는 {all_n}회입니다. 직전 갱신보다 더 오래된 모델을 선택한 경우는 {old_n}회였습니다.")
    s+=["- SPICE surrogate의 해석적 Jacobian을 중앙 유한차분과 대조했고, 네 방향의 하강성, 소자 순열 불변성, encoder 고정 및 snapshot 복사 독립성을 확인했습니다.",
        "- ALARM 원문을 바탕으로 만든 변형입니다. 역할 특성을 소자 전체에 집계한 MLP, 소스에서 학습한 좌표별 softmax 대각 가중치, 12개 방향 × 5개 스텝 후보를 사용합니다. 허용 범위에서 Armijo를 만족한 후보에 우선순위를 주고, 전부 탈락하면 예측 비용으로 결정합니다. 후보 슬롯 수는 같아도 초기 elite 개수에 따라 중복 좌표 수는 다를 수 있습니다. buffer 점수는 과거 벡터 예측오차·실측 비용 변화·현재 중심과의 거리를 사용합니다. 이식한 buffer는 한 타깃 안에서만 모델을 다시 고릅니다.",
        "- 해당 비교는 셀 수준 SPICE입니다. 검증 대상은 부하·결합·회로 배선·PDK 변화이며 전체 패키지 signoff는 포함하지 않습니다.",
        "", "[설정](ALARM_PILOT.md) · [회로/seed별 결과](runs/alarm/summary.csv) · [대응 효과](runs/alarm/contrasts.csv) · [최초 달성 누적분포](runs/alarm/hit_curves.csv) · [비용 장부](runs/alarm/audit.json) · [원시 데이터](alarm_measurements.zip)"]
    Path("ALARM_RESULTS_KO.md").write_text("\n".join(s)+"\n")

def pack():
    with TemporaryDirectory(dir=".") as scratch:
        pending=Path(scratch)/"alarm_measurements.zip"
        with ZipFile(pending,"w",compression=ZIP_DEFLATED,compresslevel=9) as archive:
            for f in sorted(root.iterdir()):
                if f.is_file():archive.write(f,"runs/"+f.name)
            for d in sorted((root/"physical").iterdir()):
                archive.writestr("physical/"+d.name+".jsonl",b"\n".join(f.read_bytes() for f in sorted(d.glob("*.json")))+b"\n")
        with ZipFile(pending) as archive:assert archive.testzip() is None
        pending.replace("alarm_measurements.zip")


if __name__=="__main__":
    report()
    pack()
