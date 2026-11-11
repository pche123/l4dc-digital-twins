from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from sigma_core_legacy import MVConfig, prepare_sigma, normalized_block_hankel, block_hankel_adjoint

EPS=1e-12

TRUE_SYM=np.diag([-1.0,-1.0,1.0])
FALSE_SYM=np.diag([-1.0,1.0,1.0])


def lorenz_rhs(v, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x,y,z=v
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z], float)


def integrate_lorenz(x0, n, dt=0.01, burn=0, obs_noise=0.0, seed=0):
    rng=np.random.default_rng(seed)
    x=np.asarray(x0,float).copy(); out=[]
    for t in range(burn+n):
        k1=lorenz_rhs(x); k2=lorenz_rhs(x+0.5*dt*k1); k3=lorenz_rhs(x+0.5*dt*k2); k4=lorenz_rhs(x+dt*k3)
        x=x+(dt/6.0)*(k1+2*k2+2*k3+k4)
        if t>=burn: out.append(x.copy())
    a=np.asarray(out).T
    if obs_noise>0: a=a+rng.normal(0,obs_noise,a.shape)
    return a


def lorenz_reservoir(seed:int, n:int=6000, dt:float=0.01, burn:int=3000, obs_noise:float=0.0):
    rng=np.random.default_rng(seed)
    x0=np.array([1.,1.,1.])+rng.normal(0,0.25,3)
    return integrate_lorenz(x0,n=n,dt=dt,burn=burn,obs_noise=obs_noise,seed=seed+991)


def extract_wing_windows(arr:np.ndarray, length:int, sign:int, count:int, min_fraction:float=0.95,
                         stride:int=4, min_separation:int|None=None, start_offset:int=0):
    """Deterministically extract mostly-one-wing windows from a reservoir.

    sign=+1 for x>0; sign=-1 for x<0. Windows are non-overlapping by at least
    min_separation starts to reduce dependence.
    """
    if min_separation is None: min_separation=length
    x=arr[0]; chosen=[]; last=-10**9
    for s in range(start_offset, arr.shape[1]-length+1, stride):
        if s-last < min_separation: continue
        w=arr[:,s:s+length]
        frac=float(np.mean((w[0]>0) if sign>0 else (w[0]<0)))
        if frac>=min_fraction:
            chosen.append(w.copy()); last=s
            if len(chosen)>=count: break
    if len(chosen)<count:
        raise RuntimeError(f'Only found {len(chosen)}/{count} wing windows sign={sign}, length={length}')
    return chosen


def build_lorenz_split(seed:int, length:int, n_tune_each:int=5, n_head_each:int=5, n_test_each:int=10,
                       obs_noise:float=0.015, min_fraction:float=0.95):
    """Independent reservoirs for training, tuning, headroom, test, and oracle arms.

    TRAIN is intentionally one-wing (+x). Evaluation sets are balanced across wings.
    This design makes the exact Lorenz symmetry external information relative to TRAIN.
    """
    # independent deterministic reservoirs prevent accidental overlap
    rtrain=lorenz_reservoir(100000+seed*31, obs_noise=obs_noise)
    rtune=lorenz_reservoir(200000+seed*31, obs_noise=obs_noise)
    rhead=lorenz_reservoir(300000+seed*31, obs_noise=obs_noise)
    rtest=lorenz_reservoir(400000+seed*31, obs_noise=obs_noise)
    rextra=lorenz_reservoir(500000+seed*31, obs_noise=obs_noise)
    roracle=lorenz_reservoir(600000+seed*31, obs_noise=obs_noise)
    train=extract_wing_windows(rtrain,length,+1,2,min_fraction=min_fraction,min_separation=2*length)
    tune=(extract_wing_windows(rtune,length,+1,n_tune_each,min_fraction=min_fraction,min_separation=length)+
          extract_wing_windows(rtune,length,-1,n_tune_each,min_fraction=min_fraction,min_separation=length,start_offset=2))
    head=(extract_wing_windows(rhead,length,+1,n_head_each,min_fraction=min_fraction,min_separation=length)+
          extract_wing_windows(rhead,length,-1,n_head_each,min_fraction=min_fraction,min_separation=length,start_offset=2))
    test=(extract_wing_windows(rtest,length,+1,n_test_each,min_fraction=min_fraction,min_separation=length)+
          extract_wing_windows(rtest,length,-1,n_test_each,min_fraction=min_fraction,min_separation=length,start_offset=2))
    extra=extract_wing_windows(rextra,length,-1,1,min_fraction=min_fraction,min_separation=length)[0]
    oracle=extract_wing_windows(roracle,length,-1,1,min_fraction=min_fraction,min_separation=length)[0]
    return {'train':train,'tune':tune,'headroom':head,'test':test,'extra_real':extra,'oracle':oracle}


def fit_standardizer(series_list):
    A=np.concatenate(series_list,axis=1)
    mu=A.mean(axis=1,keepdims=True)
    sd=np.maximum(A.std(axis=1,ddof=1,keepdims=True),1e-6)
    return mu,sd


def apply_standardizer(xs,mu,sd): return [(np.asarray(x)-mu)/sd for x in xs]


def invert_standardizer(xs,mu,sd): return [np.asarray(x)*sd+mu for x in xs]


def rel_move(a,b): return float(np.linalg.norm(b-a)/(np.linalg.norm(a)+EPS))


def spectrum_project_delta(source_z:np.ndarray, candidate_z:np.ndarray, delay_fraction:float=0.10,
                           delay_max:int=32, tol:float=1e-10):
    """Project an arbitrary trajectory perturbation onto the first-order nullspace
    of the singular-value map of the normalized block-Hankel embedding.

    This is deliberately *not* a SIGMA orbit move. It is the causal ablation:
    take method X's trajectory-space perturbation and remove the components that
    change the source Hankel singular values to first order.
    """
    cfg=MVConfig(delay_fraction=delay_fraction,delay_max=delay_max)
    prep=prepare_sigma(source_z,cfg)
    H=prep.h; U=prep.u; s=prep.s; V=prep.v
    d=prep.delay; c,n=source_z.shape
    # grad sigma_i wrt standardized trajectory z is H^*(u_i v_i^T)
    grads=[]
    for i in range(len(s)):
        outer=np.outer(U[:,i],V[:,i])
        g=block_hankel_adjoint(outer,c,n,prep.weights)
        grads.append(g.reshape(-1))
    G=np.stack(grads,axis=0)
    delta=(candidate_z-source_z).reshape(-1)
    # orthogonal projection onto ker(G)
    GG=G@G.T
    coeff=np.linalg.pinv(GG,rcond=tol)@(G@delta)
    dp=delta-G.T@coeff
    # preserve the original trajectory movement dose exactly when possible
    nd=np.linalg.norm(delta); npj=np.linalg.norm(dp)
    if npj>EPS and nd>EPS: dp*=nd/npj
    projected=source_z+dp.reshape(source_z.shape)
    # diagnostics
    linear_before=float(np.linalg.norm(G@delta))
    linear_after=float(np.linalg.norm(G@dp))
    rank=int(np.linalg.matrix_rank(G,tol=1e-9))
    return projected, {
        'sv_jacobian_rows':int(G.shape[0]),'sv_jacobian_rank':rank,
        'kernel_dimension':int(G.shape[1]-rank),'linear_sv_change_before':linear_before,
        'linear_sv_change_after':linear_after,'projection_fraction':float(np.linalg.norm(dp-delta)/(nd+EPS)),
        'movement_before':rel_move(source_z,candidate_z),'movement_after':rel_move(source_z,projected),
        'delay':int(d)
    }


def apply_symmetry(x:np.ndarray, kind:str):
    if kind=='true': return TRUE_SYM@x
    if kind=='false': return FALSE_SYM@x
    raise KeyError(kind)
