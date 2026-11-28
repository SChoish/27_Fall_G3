"""Read primitive attributes and connectivity directly from the supported SPICE decks."""
from dataclasses import dataclass
import re
import numpy as np
from anonymous_network import WidthNet
from circuit import deck

@dataclass(frozen=True)
class Graph:
    features: np.ndarray
    mask: np.ndarray | None

    def reorder(self, perm):
        order=np.r_[perm,np.arange(len(perm),len(self.features))]
        return Graph(self.features[order],None if self.mask is None else self.mask[np.ix_(order,order)])


def extract(text,pdk,connected=True):
    mos=[];others=[]
    for line in text.splitlines()[1:]:
        fields=line.lower().split()
        if not fields or fields[0][0] in ".*":continue
        kind=fields[0][0]
        # Width ratio is filled in at query time. Other columns are immutable raw metadata.
        f=np.zeros(10);f[8+(pdk=="gf")]=1.
        if kind in ("x","m"):
            identifier=fields[5]
            polarity={"nmos_3p3":1,"pmos_3p3":2,"sky130_fd_pr__nfet_01v8":1,"sky130_fd_pr__pfet_01v8":2}[identifier]
            f[polarity]=1
            params=dict(x.split("=") for x in fields[6:] if "=" in x)
            # Both deck conventions encode W and L in micrometres (GF has an explicit u suffix).
            f[5]=np.log(float(params["w"].removesuffix("u")))
            f[6]=np.log(float(params["l"].removesuffix("u")))
            mos.append((f,set(fields[1:5])))
        elif kind=="c":
            f[3]=1.
            f[7]=np.log1p(float(fields[3].removesuffix("f")))
            others.append((f,set(fields[1:3])))
        elif kind=="v":
            f[4]=1.
            value=" ".join(fields[3:])
            if value.startswith("pulse("):
                levels=re.match(r"pulse\(([^ ]+) ([^ ]+)",value)
                f[7]=abs(float(levels[1])-float(levels[2]))
            else:f[7]=float(value)
            others.append((f,set(fields[1:3])))
        else:raise ValueError(f"Unsupported primitive: {fields[0]}")
    parts=mos+others
    mask=np.array([[bool(a&b) for _,b in parts] for _,a in parts]) if connected else None
    return Graph(np.array([f for f,_ in parts]),mask)


def geometry(c,connected=True):
    text,_=deck(c,np.ones(2*len(c.parents)),0)
    return extract(text,c.pdk,connected)


class DeviceNet(WidthNet):
    def __init__(self,attention):
        super().__init__(attention)
        self.params["a"]=np.random.default_rng(884).normal(0,.35,(10,12))

    def embed(self,z,context):
        z=np.atleast_2d(z)
        x=np.broadcast_to(context.features,(len(z),*context.features.shape)).copy()
        x[:,:z.shape[1],0]=z/.3
        return x,np.tanh(x@self.params["a"]+self.params["b"])

    def input_gradient(self,x,dp):
        return x.reshape(-1,x.shape[-1]).T@dp.reshape(-1,dp.shape[-1])


class BoundNet:
    def __init__(self,net,graph):
        self.net=net;self.graph=graph

    def encode(self,z):
        return self.net.encode(z,self.graph)

    def readout(self,h):
        return self.net.readout(h)
