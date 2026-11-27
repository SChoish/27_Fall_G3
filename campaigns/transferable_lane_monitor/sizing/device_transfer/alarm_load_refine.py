"""Second-stage oracle check of 5% attainability near the asymmetric-load boundary."""
import json
import numpy as np
from alarm_load_probe import constraints
from alarm_pilot import TARGETS
from audit import checks
from circuit import simulate
from experiment import arguments,upper

def project(a,b,lower,upper):
    # Alternating projections onto linearized constraint halfspaces and box.
    move=np.zeros(a.shape[1])
    for _ in range(260):
        violation=a@move-b
        if np.max(violation)<2e-5:break
        k=int(np.argmax(violation))
        move=np.clip(move-(violation[k]/max(a[k]@a[k],1e-12))*a[k],lower,upper)
    return move,np.max(a@move-b)

if __name__=="__main__":
    args=arguments();name="load_sky";c=TARGETS[name];hi=np.log(upper(c))
    root=args.out/"alarm"
    old=json.loads((root/"asymmetric_load_probe.json").read_text())
    rows={tuple(d["x"]):d["raw"] for d in old["records"]};new={}
    ref=rows[tuple(np.ones(len(hi)))]
    def evaluate(z):
        x=tuple(np.round(np.exp(np.clip(z,np.log(.58),hi)),6))
        if x not in rows:
            rows[x]=[simulate(c,x,k,args) for k in range(4)]
            new[x]=rows[x]
        return constraints(rows[x],ref)[0],x
    best=(old["edp_ratio"],tuple(old["x"]));z=np.log(best[1]);radius=.04
    n=4*len(c.outputs)
    for _ in range(6):
        y,_=evaluate(z);derivatives=[]
        for i in range(len(z)):
            h=.004 if z[i]+.004<=hi[i] else -.004
            v=z.copy();v[i]+=h
            derivatives.append((evaluate(v)[0]-y)/h)
        a=np.array(derivatives).T;low=np.maximum(-radius,np.log(.58)-z);high=np.minimum(radius,hi-z)
        left,right=np.log(.90),max(y[:n].max(),np.log(.96))
        selected=None
        for _ in range(10):
            level=(left+right)/2
            cap=np.r_[np.full(n,level),np.full(len(y)-n,-.0008)]
            delta,violation=project(a,cap-y,low,high)
            if violation<.00012:right=level;selected=delta
            else:left=level
        if selected is None:radius*=.65;continue
        trial,x=evaluate(z+selected)
        for step in (1.,.65):
            trial,x=evaluate(z+step*selected)
            if trial[n:].max()<=0 and np.exp(trial[:n].max())<best[0]:
                best=(np.exp(trial[:n].max()),x)
        z=np.log(best[1]);radius*=.72
        if best[0]<.949:break
    args.step_ps=.25
    fine=[simulate(c,x,k,args) for x in (np.ones(len(hi)),best[1]) for k in range(4)]
    vals=[checks(fine[4+k:6+k],fine[k:k+2]) for k in (0,2)]
    val=max(b for a,b in vals) if all(a for a,b in vals) else None
    result=dict(edp_ratio=best[0],x=best[1],additional_spice_calls=len(new)*4,fine_calls=8,
                fine_edp=val,fine=fine,records=[dict(x=x,raw=r) for x,r in new.items()])
    (root/"load_oracle_refined.json").write_text(json.dumps(result))
    print(best[0],val,len(new)*4)
