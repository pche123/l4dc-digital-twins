from __future__ import annotations
import numpy as np


def rel_move(a,b): return float(np.linalg.norm(b-a)/(np.linalg.norm(a)+1e-12))

def _rescale_to_move(src,cand,target):
    d=cand-src; m=np.linalg.norm(d)/(np.linalg.norm(src)+1e-12)
    if m<1e-12: return src.copy()
    return src + d*(target/m)

def jitter(src,rng,target):
    d=rng.normal(size=src.shape); d/=np.linalg.norm(d)+1e-12
    return src+d*(target*np.linalg.norm(src))

def colored_jitter(src,rng,target):
    d=rng.normal(size=src.shape)
    ker=np.array([1,2,3,2,1],float); ker/=ker.sum()
    for c in range(src.shape[0]): d[c]=np.convolve(d[c],ker,mode='same')
    d/=np.linalg.norm(d)+1e-12
    return src+d*(target*np.linalg.norm(src))

def scaling(src,rng,target):
    factors=1+rng.normal(0,0.2,size=(src.shape[0],1)); cand=src*factors
    return _rescale_to_move(src,cand,target)

def time_warp(src,rng,target):
    n=src.shape[1]; t=np.arange(n,dtype=float)
    phase=rng.uniform(0,2*np.pi); warp=t+1.6*np.sin(2*np.pi*t/max(n-1,1)+phase)
    warp=np.clip(warp,0,n-1)
    cand=np.vstack([np.interp(t,warp,src[c]) for c in range(src.shape[0])])
    return _rescale_to_move(src,cand,target)

def mixup(src,other,rng,target):
    lam=rng.uniform(0.1,0.4); cand=(1-lam)*src+lam*other
    return _rescale_to_move(src,cand,target)
