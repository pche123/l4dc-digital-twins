"""Independent label-free feature coverage proxy audit for existing v1.0 one-step experiment.

Exactly matches the sampling, smoothing, and feature construction in coverage_value_sweep.py.
Uses synthetic blocks' inputs only for leverage; val_vs_duplicate is loaded solely for evaluation.
Note: leverage does NOT establish synthetic-target correctness or forecast improvement.
"""
import os, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from selective_gate_sweep import windows,basis,rk4
from coverage_value_sweep import standardizer,rows,fit

def one(system,seed,n,lam=1e-4):
    noise=.015 if system==0 else .01
    degree=2 if system==0 else 3
    W=[savgol_filter(a,9,3,axis=0,mode='interp') for a in windows(system,seed,n,noise)]
    mu,sd=standardizer(W)
    X,Y=rows(W,mu,sd,degree)
    H=X.T@X + lam*np.eye(X.shape[1])
    B=np.linalg.solve(H,X.T@Y)
    tr=np.array((-1,-1,1) if system==0 else (-1,-1))
    fl=np.array((-1,1,1) if system==0 else (-1,1))
    independent=[savgol_filter(a,9,3,axis=0,mode='interp') for a in windows(system,seed+500000,1,noise)]
    blocks={'duplicate':[W[0]],'exact':[W[0]*tr], 'false':[W[0]*fl],
           'same':[independent[0]],'missing':[independent[0]*tr]}
    out=[]
    for arm, wb in blocks.items():
        C,U=rows(wb,mu,sd,degree)
        # The eigenvalues of Z=CH^-1 C' characterize novel feature-space coverage.
        Z=C@np.linalg.solve(H,C.T)
        z_eig=np.maximum(np.linalg.eigvalsh((Z+Z.T)/2),0)
        residual=U-C@B
        upd=np.linalg.solve(H+C.T@C,C.T@residual)
        # Effective incremental degrees of freedom; monotone bounded representation of leverage.
        info=float(np.sum(z_eig/(1+z_eig)))
        out.append(dict(system='Lorenz' if system==0 else 'Duffing',seed=seed,n_windows=n,arm=arm,
            leverage_mean=float(np.trace(Z)/len(C)),leverage_effdf=info,
            leverage_top=float(z_eig[-1]), residual_rms=float(np.sqrt(np.mean(residual**2))),
            update_fro=float(np.linalg.norm(upd)), nrows=len(C)))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=6000);p.add_argument('--nseeds',type=int,default=40)
    p.add_argument('--out',default='data/coverage_proxy_audit.csv');a=p.parse_args()
    rk4(0,np.array([1.,1.,1.]),1,.01)
    out=[]
    for s in (0,1):
        for n in (2,4,8,16):
            for i in range(a.nseeds):
                out.extend(one(s,a.start+i,n))
            print('completed',s,n,flush=True)
    df=pd.DataFrame(out)
    df.to_csv(a.out,index=False)
    print(df.groupby(['system','n_windows','arm'])[['leverage_mean','leverage_effdf','residual_rms']].median().round(4).to_string())
