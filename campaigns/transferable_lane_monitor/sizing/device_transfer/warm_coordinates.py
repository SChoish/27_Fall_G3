"""Check chart/PDK semantics and actual source electrical scales."""
from pathlib import Path
import json
import re
import numpy as np
from alarm_model import features,inputs
from alarm_pilot import SOURCE_NETS
from circuit import deck
from experiment import upper
from warm_space import DEV,TEST,span
from warm_policy import source_prior

def audit(root):
    net=source_prior(root,21);pdk={};err=0.;unit_error=0.;cov_error=0.;angles=[]
    for c in (*DEV.values(),*TEST.values()):
        dim=len(upper(c));lo=.58;s=span(c)
        x=np.exp(np.log(lo)+.45*s);u=np.log(x/lo)/s
        np.testing.assert_allclose(lo*np.exp(u*s),x)
        unit_error=max(unit_error,float(abs(np.log((x*1e-6)/(1e-6))-np.log(x)).max()))
        desc=features(c);J=net.jacobian(x,desc)*s
        h=.00001
        def f(u):return net.predict(inputs(lo*np.exp(u*s),desc))[0]
        grad=np.stack([(f(u+h*v)-f(u-h*v))/(2*h) for v in np.eye(dim)],axis=1)
        err=max(err,float(abs(grad-J).max()))
        yp=net.predict(inputs(x,desc))[0]
        penalty=3/(1+np.exp(-np.clip(60*(yp[1:]+.01),-40,40)))
        g=np.r_[1.,penalty]@net.jacobian(x,desc)
        def mapped(scale,damped):
            gu=g*scale
            floor=.015*np.mean(gu*gu)+.0001 if damped else 0.
            physical=-scale*gu/(.7*gu*gu+floor)
            return physical/np.linalg.norm(physical)
        raw=mapped(np.ones(dim),False)
        cov_error=max(cov_error,float(np.max(abs(raw-mapped(s,False)))))
        cos=np.clip(mapped(np.ones(dim),True)@mapped(s,True),-1,1)
        angles.append(float(np.degrees(np.arccos(cos))))
        text,_=deck(c,x,0)
        rows=[t for t in text.splitlines() if re.match(r'^[MX][np]\d+ ',t)]
        assert len(rows)==dim
        for i,line in enumerate(rows):
            w,l=re.search(r' w=([.0-9]+u?) l=([.0-9]+u?)',line).groups()
            assert not re.search(r'\s(?:m|mult|nf)=',line,re.I)
            desired=c.sizes[i//2]*(1+i%2)*x[i]
            assert abs(float(w.rstrip('u'))-desired)<1e-8
            assert l==('.28u' if c.pdk=='gf' else '.15')
        pdk[c.pdk]=dict(length_um=float(l.rstrip("u")),multiplicity="implicit 1, no instance sweep")
    assert err<1e-7 and unit_error<1e-12 and cov_error<1e-10
    labels=np.vstack([json.loads((root/(n+"_source.json")).read_text())["labels"] for n in SOURCE_NETS])
    result=dict(pdks=pdk,log_spans=[np.log(1/.58),np.log(1.3/.58)],jacobian_max_error=err,
                unit_invariance_error=unit_error,undamped_mapped_direction_error=cov_error,
                damped_angles_deg=angles,source_label_std=np.std(labels,axis=0).tolist(),
                source_label_range=np.ptp(labels,axis=0).tolist())
    (root/"warm"/"coordinate_audit.json").write_text(json.dumps(result,indent=2))
    print(result)
if __name__=="__main__":audit(Path("runs"))
