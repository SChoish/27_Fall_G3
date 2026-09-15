"""Classical controllers consume only the exact policy observation vector."""
import numpy as np
from .environment import FRAME_DIM,action_id


class Classical:
    def __init__(self,mode,horizon=64):
        if mode not in ('none','periodic','event','bang_bang'):raise ValueError(mode)
        self.mode=mode;self.horizon=horizon;self.last_used=-2
    def __call__(self,observation):
        f=np.asarray(observation)[-FRAME_DIM:];t=int(round(float(f[20])*self.horizon))
        age=int(round(float(f[16])*self.horizon));exists=f[17]>.5
        last=t-1-age;delta=np.zeros(2,int);m=f[10:16].reshape(2,3)
        if self.mode=='none':return action_id(delta,False)
        if exists and last!=self.last_used:
            for lane in range(2):
                if f[18+lane]>.5 and (m[lane,1]<.18 or min(m[lane,0],m[lane,2])<.1):
                    difference=m[lane,2]-m[lane,0]
                    delta[lane]=4 if difference>.01 else -4 if difference<-.01 else 0
            self.last_used=last
        if self.mode=='periodic':request=t%4==0
        elif self.mode=='bang_bang':request=True
        else:request=not exists or age>=7 or (age>=3 and m[:,1].min()<.16)
        return action_id(delta,request)
