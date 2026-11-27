"""Post-hoc attainability check on the asymmetric load, using eight independent widths."""
import json
import numpy as np
from alarm_pilot import TARGETS
from audit import checks
from circuit import simulate
from experiment import arguments, upper

def constraints(rows,ref):
    values=[];edp=[]
    for k,(r,b) in enumerate(zip(rows,ref)):
        for i,(p,q) in enumerate(zip(r["ports"],b["ports"])):
            edp.append(np.log(r["energy"]*max(p[:2])/(b["energy"]*max(q[:2]))))
            values.extend(np.log(np.array(p[:4])/q[:4]/1.10))
            values.extend((np.log(.90/p[4]),np.log(max(p[5],1e-9)/.10)))
            if k%2:
                dp=np.abs(np.array(p[:2])-rows[k-1]["ports"][i][:2])
                dq=np.abs(np.array(q[:2])-ref[k-1]["ports"][i][:2])
                values.extend(np.log(np.maximum(dp,1e-10)/(1.05*max(dq))))
    return np.array(edp+values),len(edp)

def value(y,n):
    return y[...,:n].max(-1)+5*np.maximum(0,y[...,n:]+.0015).max(-1)

if __name__=="__main__":
    args=arguments();c=TARGETS["load_sky"];hi=np.log(upper(c));dim=len(hi)
    rows={}
    def eval_at(z):
        x=tuple(np.round(np.exp(np.clip(z,np.log(.58),hi)),6))
        if x not in rows:rows[x]=[simulate(c,x,k,args) for k in range(4)]
        return rows[x]
    ref=eval_at(np.zeros(dim));widths=np.log(np.repeat([.95,.91,.95,.91],2))
    rng=np.random.default_rng(479);radius=.095
    for _ in range(8):
        cur,n=constraints(eval_at(widths),ref)
        jac=[]
        for i in range(dim):
            v=widths.copy()
            h=.014 if widths[i]+.014<=hi[i] else -.014
            v[i]+=h
            jac.append((constraints(eval_at(v),ref)[0]-cur)/h)
        jac=np.array(jac)
        moves=rng.uniform(-radius,radius,size=(10000,dim))
        # Refine promising perturbations against the linearized per-port constraints.
        for scale in (.35,.12,.04):
            moves=np.clip(widths+moves,np.log(.58),hi)-widths
            pred=cur+moves@jac
            elite=moves[np.argsort(value(pred,n))[:32]]
            moves=np.vstack((elite,np.repeat(elite,32,axis=0)+rng.normal(0,scale*radius,(1024,dim))))
        moves=np.clip(widths+moves,np.log(.58),hi)-widths
        step=moves[np.argmin(value(cur+moves@jac,n))]
        y,_=constraints(eval_at(widths+step),ref)
        if value(y,n)<value(cur,n):
            widths+=step;radius=min(.15,radius*1.15)
        else:radius*=.6
        if y[:n].max()<=np.log(.9) and y[n:].max()<=0:break
    best=(1.,tuple(np.ones(dim)))
    for x,raw in rows.items():
        results=[checks(raw[k:k+2],ref[k:k+2]) for k in (0,2)]
        if all(a for a,b in results):best=min(best,(max(b for a,b in results),x))
    calls=len(rows)*4
    args.step_ps=.25
    fine=[simulate(c,x,k,args) for x in (np.ones(dim),best[1]) for k in range(4)]
    results=[checks(fine[4+k:6+k],fine[k:k+2]) for k in (0,2)]
    val=max(b for a,b in results) if all(a for a,b in results) else None
    out=dict(circuit="load_sky",calls=calls,edp_ratio=best[0],x=best[1],fine_edp=val,fine_calls=8,
             fine=fine,records=[dict(x=x,raw=r) for x,r in rows.items()])
    (args.out/"alarm"/"asymmetric_load_probe.json").write_text(json.dumps(out))
    print(calls,best,val)
