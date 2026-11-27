"""Empirical cost-to-target curves for every method; requires ReportLab."""
import json
from zipfile import ZipFile
import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, String
from reportlab.lib.colors import HexColor

LABELS={"random":("Random","#8e9296"),"de":("DE","#7656a8"),"bo":("BO","#d55e00"),
        "acor":("ACOR","#1877ab"),"group_acor":("Grouped ACOR","#098570"),"group_random":("Grouped random","#ad7918"),
        "role_poll":("Role + poll","#151a24"),"residual":("Residual critic","#1877ab"),
        "warm4":("4 local probes","#d55e00"),"rank2":("Rank-2 metric","#098570"),
        "shuffled":("Shuffled labels","#7656a8")}


def main():
    with ZipFile("century_measurements.zip") as z:
        curves=[json.loads(z.read(f)) for f in z.namelist() if f.startswith("traces/")]
    d=Drawing(840,550)
    d.add(String(34,522,"SPICE evaluations to first verified target",fontName="Helvetica-Bold",fontSize=18))
    d.add(String(34,502,"Three circuit layouts, each evaluated on SKY130 and GF180. Success requires all four operating corners.",fontSize=10))
    rows=(("Source-free search",("random","group_random","de","bo","acor","group_acor","role_poll")),
          ("Source-trained search",("residual","warm4","rank2","shuffled","role_poll")))
    for row,(label,arms) in enumerate(rows):
        y=315-row*225
        d.add(String(34,y+155,label,fontName="Helvetica-Bold",fontSize=12))
        for col,(protocol,goal,subtitle) in enumerate([("joint","0.95","5% EDP improvement"),("strict","0.9","10% EDP improvement, direct target")]):
            chart=LinePlot()
            chart.x=66+col*396;chart.y=y;chart.width=320;chart.height=123
            d.add(String(chart.x,y+132,subtitle,fontSize=10))
            data=[]
            for i,arm in enumerate(arms):
                rs=[r for r in curves if r["protocol"]==protocol and r["arm"]==arm]
                assert len(rs)==(6 if arm=="role_poll" else 18)
                times=np.array([r["hits"].get(goal,101) for r in rs])
                points=[(0,0)]
                for t in sorted(set(times[times<=100])):
                    points.extend([(int(t),points[-1][1]),(int(t),100*float(np.mean(times<=t)))])
                points.append((100,points[-1][1]))
                data.append(points)
                chart.lines[i].strokeColor=HexColor(LABELS[arm][1])
                chart.lines[i].strokeWidth=1.8
                if arm in ("role_poll","shuffled","group_random","random"):chart.lines[i].strokeDashArray=[4,2]
            chart.data=data
            for axis in (chart.xValueAxis,chart.yValueAxis):
                axis.valueMin=0;axis.valueMax=100
                axis.labels.fontSize=8
                axis.strokeColor=HexColor("#7b8494")
            chart.xValueAxis.valueSteps=[0,20,40,60,80,100]
            chart.yValueAxis.valueSteps=[0,25,50,75,100]
            chart.yValueAxis.labelTextFormat="%d%%"
            chart.yValueAxis.visibleGrid=1
            chart.yValueAxis.gridStrokeColor=HexColor("#e1e5e8")
            chart.yValueAxis.gridStrokeWidth=.5
            d.add(Line(chart.x+.4*chart.width,chart.y,chart.x+.4*chart.width,chart.y+chart.height,
                       strokeColor=HexColor("#c0c5cb"),strokeDashArray=[2,2]))
            d.add(chart)
            d.add(String(chart.x+chart.width/2,y-25,"Target SPICE evaluations",fontSize=8,textAnchor="middle"))
        x=35
        for arm in arms:
            name,color=LABELS[arm]
            line=Line(x,y-45,x+15,y-45,strokeColor=HexColor(color),strokeWidth=2)
            if arm in ("role_poll","shuffled","group_random","random"):line.strokeDashArray=[4,2]
            d.add(line)
            d.add(String(x+20,y-48,name,fontSize=9))
            x+=36+len(name)*5.0
    d.add(String(34,13,"Vertical axis: fraction of trials reaching the target. Dotted vertical: former 40-evaluation cap. Each circuit carries equal weight.",fontSize=8))
    renderSVG.drawToFile(d,"century_attainment.svg")


if __name__=="__main__":main()
