"""On identical unused warmup lists, measure whether source experts differ in first hits."""
from dataclasses import replace
from statistics import mean
import json
from zipfile import ZipFile

import numpy as np
from alarm_pilot import SOURCE_NETS
from experiment import upper
from ppo_sizing import ROOT, train
from surrogate_buffer import merit
from surrogate_gate import EXPERTS, HORIZON, proposals


def first(net,x,y,lo,hi,warm,goal,mode):
    hist=[0,*warm]
    if np.any(merit(y[hist],goal)<=0):
        return 0,True
    available=[i for i in range(1,len(x)) if i not in hist]
    for step in range(1,HORIZON+1):
        ids,_=proposals(net,x[hist],y[hist],lo,hi,x[available],goal)
        i=available.pop(ids[mode])
        hist.append(i)
        if merit(y[i:i+1],goal)[0]<=0:
            return step,True
    return HORIZON,False


def main():
    with ZipFile(ROOT/"ppo_sizing_measurements.zip") as z:
        sources=json.loads(z.read("source.json"))
    print("## 소스 데이터에서 두 고정 제안자의 구별 가능성\n")
    print("각 소스 회로·목표에 학습에 쓰인 난수 일정과 독립인 웜업 24세트를 대응시켰다. 같은 네 소스 회로와 설계 표를 재사용한 진단이다. 비용은 웜업 뒤 개발 설계 질의 수이며 미달은 16, 웜업 통과는 0이다. 각 설계 질의는 개발 두 코너에 해당한다.\n")
    print("| 소스 | 목표 | 대응 사례 | 성공 여부 불일치 | 제한비용 다름 | 최신 head 평균 | 복귀 규칙 평균 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for p,rows in sources.items():
        pairs={.95:[],.90:[]}
        for j,row in enumerate(rows):
            net=train(rows[:j]+rows[j+1:],only=("anon400",))["anon400"]
            hi=np.log(upper(replace(SOURCE_NETS[row["name"]],pdk=p)));lo=np.full_like(hi,np.log(.58))
            x,y=np.log(row["x"]),np.asarray(row["y"])
            for rep in range(24):
                warm=np.random.default_rng([234801,j,rep]).choice(np.arange(1,len(x)),4,replace=False).tolist()
                for g in pairs:
                    pairs[g].append([first(net,x,y,lo,hi,warm,g,k) for k in range(len(EXPERTS))])
        for g,group in pairs.items():
            print(f"| {p} | {1-g:.0%} | {len(group)} | {sum(a[1]!=b[1] for a,b in group)} | "
                  f"{sum(a[0]!=b[0] for a,b in group)} | {mean(a[0] for a,b in group):.3f} | {mean(b[0] for a,b in group):.3f} |")


if __name__=="__main__":
    main()
