#!/usr/bin/env python3
"""Optimization-only v0.14.2c SIGMA core.

Scientific behavior is inherited from v0.14.2. The optimization is that beta
search evaluates movement without recomputing singular spectra. Full orbit and
emitted-spectrum diagnostics are computed exactly once for the final beta.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
import json, math
from pathlib import Path
from typing import Any
import numpy as np

EPS = 1e-12

@dataclass(frozen=True)
class MVConfig:
    delay_fraction: float = 0.10
    delay_max: int = 32
    factor_mix_u: float = 0.50
    spectral_focus: float = 0.0
    retraction: str = "polar"
    movement_tol: float = 2e-4
    beta_max: float = 8.0
    scale_floor: float = 1e-6

@dataclass
class PreparedMV:
    y: np.ndarray; mean: np.ndarray; scale: np.ndarray; z: np.ndarray
    h: np.ndarray; weights: np.ndarray; u: np.ndarray; s: np.ndarray
    v: np.ndarray; delay: int; cfg: MVConfig


def with_config(prep: PreparedMV, cfg: MVConfig) -> PreparedMV:
    # Reuses the source-local standardization, Hankelization and one SVD.
    if cfg.delay_fraction != prep.cfg.delay_fraction or cfg.delay_max != prep.cfg.delay_max:
        raise ValueError("cannot reuse PreparedMV when delay settings change")
    return replace(prep, cfg=cfg)


def _as_2d(y):
    y=np.asarray(y,float)
    if y.ndim!=2 or y.shape[0]<1 or y.shape[1]<8: raise ValueError(f"expected (channels,time>=8), got {y.shape}")
    if not np.isfinite(y).all(): raise ValueError("input contains NaN or infinity")
    return y

def fit_channel_standardizer(y, scale_floor=1e-6):
    y=_as_2d(y); mean=y.mean(1,keepdims=True); scale=y.std(1,ddof=1,keepdims=True)
    fallback=np.maximum(np.linalg.norm(y-mean,axis=1,keepdims=True)/math.sqrt(max(y.shape[1]-1,1)),scale_floor)
    return mean,np.maximum(np.where(scale>scale_floor,scale,fallback),scale_floor)

def standardize(y,mean,scale): return (_as_2d(y)-mean)/scale

def unstandardize(z,mean,scale): return np.asarray(z,float)*scale+mean

def choose_delay(n,cfg): return int(min(max(4,int(round(cfg.delay_fraction*n))),cfg.delay_max,max(4,n//2),n-2))
def hankel_weights(n,d):
    k=n-d+1; w=np.zeros(n)
    for lag in range(d): w[lag:lag+k]+=1
    return w

def normalized_block_hankel(z,d):
    z=_as_2d(z);m,n=z.shape;k=n-d+1;w=hankel_weights(n,d);h=np.empty((m*d,k))
    for lag in range(d):
        idx=np.arange(lag,lag+k);h[lag*m:(lag+1)*m]=z[:,idx]/np.sqrt(w[idx])[None,:]
    return h,w

def block_hankel_adjoint(h,c,n,w):
    h=np.asarray(h,float);d=h.shape[0]//c;k=n-d+1;z=np.zeros((c,n))
    for lag in range(d):
        idx=np.arange(lag,lag+k);z[:,idx]+=h[lag*c:(lag+1)*c]/np.sqrt(w[idx])[None,:]
    return z

def project_block_hankel(h,c,n,w):
    z=block_hankel_adjoint(h,c,n,w);hp,_=normalized_block_hankel(z,h.shape[0]//c);return z,hp

def rel_movement(a,b): return float(np.linalg.norm(b-a)/(np.linalg.norm(a)+EPS))
def polar_retract(q,delta):
    a=q+delta;g=(a.T@a+(a.T@a).T)/2;lam,vec=np.linalg.eigh(g);lam=np.maximum(lam,1e-12)
    return a@((vec*(1/np.sqrt(lam))[None,:])@vec.T)
def qr_retract(q,delta):
    q2,r=np.linalg.qr(q+delta,mode="reduced");s=np.sign(np.diag(r));s[s==0]=1;return q2*s[None,:]
def _mode_weights(s,focus):
    if abs(focus)<1e-15:return np.ones_like(s)
    q=np.clip(s/(s[0]+EPS),0,1);base=np.maximum(q if focus>0 else 1-q,1e-8)**abs(focus);return .1+.9*base
def tangent_direction(q,s,rng,focus):
    r=q.shape[1];w=_mode_weights(s,focus);g=rng.normal(size=(r,r));sk=.5*(g-g.T);sk*=np.sqrt(w[:,None]*w[None,:]);d=q@sk
    if q.shape[0]>r:
        e=rng.normal(size=q.shape);d+=(e-q@(q.T@e))*w[None,:]
    norm=np.linalg.norm(d)
    if norm>EPS:d=d/norm*math.sqrt(r)
    if np.linalg.norm(q.T@d+d.T@q)>1e-8:raise RuntimeError("tangent projection failed")
    return d

def prepare_sigma(y,cfg=MVConfig()):
    y=_as_2d(y);mean,scale=fit_channel_standardizer(y,cfg.scale_floor);z=standardize(y,mean,scale);d=choose_delay(y.shape[1],cfg);h,w=normalized_block_hankel(z,d);u,s,vt=np.linalg.svd(h,full_matrices=False)
    return PreparedMV(y,mean,scale,z,h,w,u,s,vt.T,d,cfg)
def sigma_direction(prep,rng):
    mix=float(np.clip(prep.cfg.factor_mix_u,0,1));du=tangent_direction(prep.u,prep.s,rng,prep.cfg.spectral_focus);dv=tangent_direction(prep.v,prep.s,rng,prep.cfg.spectral_focus)
    return math.sqrt(mix)*du,math.sqrt(1-mix)*dv

def _orbit_and_projection(prep,du,dv,beta):
    retract=polar_retract if prep.cfg.retraction=="polar" else qr_retract
    u2=retract(prep.u,beta*du) if np.linalg.norm(du)>EPS else prep.u
    v2=retract(prep.v,beta*dv) if np.linalg.norm(dv)>EPS else prep.v
    orbit=(u2*prep.s[None,:])@v2.T
    z2,hp=project_block_hankel(orbit,prep.y.shape[0],prep.y.shape[1],prep.weights)
    return orbit,z2,hp

def movement_only(prep,du,dv,beta):
    _,z2,_=_orbit_and_projection(prep,du,dv,beta)
    return rel_movement(prep.z,z2)
def emit_full(prep,du,dv,beta):
    orbit,z2,hp=_orbit_and_projection(prep,du,dv,beta);y2=unstandardize(z2,prep.mean,prep.scale)
    sv_orbit=np.linalg.svd(orbit,compute_uv=False);sv_proj=np.linalg.svd(hp,compute_uv=False)
    denom=np.max(prep.s)+EPS
    return y2,{"beta":float(beta),"movement":rel_movement(prep.z,z2),
        "projection_residual":float(np.linalg.norm(orbit-hp)/(np.linalg.norm(orbit)+EPS)),
        "orbit_sv_relerr":float(np.max(np.abs(sv_orbit-prep.s))/denom),
        "emitted_sv_relerr":float(np.max(np.abs(sv_proj-prep.s))/denom),
        "isometry_error":float(abs(np.linalg.norm(prep.h)-np.linalg.norm(prep.z)))}
def matched_candidate(prep,rng,target=.05):
    du,dv=sigma_direction(prep,rng);cache={}
    def mov(beta):
        beta=float(np.clip(beta,0,prep.cfg.beta_max));key=round(beta,13)
        if key not in cache:cache[key]=movement_only(prep,du,dv,beta)
        return cache[key]
    lo,hi=0.,.05;mhi=mov(hi)
    while mhi<target and hi<prep.cfg.beta_max:
        lo=hi;hi=min(prep.cfg.beta_max,hi*2);mhi=mov(hi)
    if mhi<target:
        y,meta=emit_full(prep,du,dv,hi);meta.update({"target_movement":float(target),"movement_error":abs(meta["movement"]-target),"matched":False,"failure":"target_not_reached_before_beta_max"});return y,meta
    iterations=0
    for iterations in range(1,61):
        mid=.5*(lo+hi);mm=mov(mid)
        if abs(mm-target)<=prep.cfg.movement_tol:lo=hi=mid;break
        if mm<target:lo=mid
        else:hi=mid
    beta=.5*(lo+hi);y,meta=emit_full(prep,du,dv,beta)
    meta.update({"target_movement":float(target),"movement_error":float(abs(meta["movement"]-target)),"matched":bool(abs(meta["movement"]-target)<=prep.cfg.movement_tol),"delay":prep.delay,"channels":prep.y.shape[0],"timepoints":prep.y.shape[1],"factor_mix_u":prep.cfg.factor_mix_u,"spectral_focus":prep.cfg.spectral_focus,"retraction":prep.cfg.retraction,"beta_search_evaluations":len(cache),"beta_search_iterations":iterations,"full_svd_diagnostic_calls":1})
    return y,meta

def synthetic_coupled_system(seed=7,channels=5,n_time=180):
    rng=np.random.default_rng(seed);a=rng.normal(scale=.12,size=(channels,channels));a=.72*np.eye(channels)+(a+a.T)/2;a*=.94/max(max(abs(np.linalg.eigvals(a))),.94);y=np.zeros((channels,n_time));y[:,0]=rng.normal(size=channels);forcing=rng.normal(size=(channels,2))
    for t in range(n_time-1):
        u=np.array([math.sin(2*math.pi*t/31),math.cos(2*math.pi*t/47)]);y[:,t+1]=a@y[:,t]+.12*forcing@u+.025*rng.normal(size=channels)
    return y

def self_test(output=None):
    rng=np.random.default_rng(123);y=synthetic_coupled_system();records=[]
    for mix in [0.,.5,1.]:
        cfg=MVConfig(factor_mix_u=mix);prep=prepare_sigma(y,cfg);y2,meta=matched_candidate(prep,rng,.05);records.append(meta);assert meta["matched"] and meta["orbit_sv_relerr"]<1e-9 and np.isfinite(y2).all() and meta["full_svd_diagnostic_calls"]==1
    out={"status":"pass","records":records}
    if output:Path(output).write_text(json.dumps(out,indent=2),encoding="utf-8")
    return out
if __name__=="__main__":
    import argparse;p=argparse.ArgumentParser();p.add_argument("--output");a=p.parse_args();print(json.dumps(self_test(a.output),indent=2))
