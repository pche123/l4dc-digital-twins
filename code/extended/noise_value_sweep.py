"""Prospective stress-test extension of v1.0 one-step study across measurement noise.
Standalone protocol, new seeds 11000--11039. Same candidate size, ridge, test as coverage sweep.
Noise multiplier m affects training *and* synthetic observed windows, not clean deployment reference.
Do not combine with v0.7 recursive results or claim tuning/preregistration.
"""
import os,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.signal import savgol_filter
from selective_gate_sweep import windows,basis,rk4
from coverage_value_sweep import standardizer,rows,fit,target_values

def one(system,seed,mult,n=8,lam=1e-4):
    noise=(.015 if system==0 else .01)*mult
    degree=2 if system==0 else 3
    W=[savgol_filter(a,9,3,axis=0,mode='interp') for a in windows(system,seed,n,noise)]
    mu,sd=standardizer(W)
    X,Y=rows(W,mu,sd,degree)
    Strue=np.array((-1,-1,1) if system==0 else (-1,-1))
    Sfalse=np.array((-1,1,1) if system==0 else (-1,1))
    blocks={'duplicate':[W[0]],'exact':[W[0]*Strue],'false':[W[0]*Sfalse]}
    Wt=windows(system,seed+1000000,8,0.0)
    xp,yp=target_values(system,Wt)
    xm,ym=target_values(system,[a*Strue for a in Wt])
    errors={}
    for arm,Wb in blocks.items():
        C,U=rows(Wb,mu,sd,degree)
        B=fit(np.vstack((X,C)),np.vstack((Y,U)),lam)
        by=[]
        for states,targets in [(xp,yp),(xm,ym)]:
            p=basis((states-mu)/sd,degree)@B*sd
            by.append(float(np.mean(((p-targets)/sd)**2)))
        errors[arm]=sum(by)/2
    return [{'system':'Lorenz' if system==0 else 'Duffing','seed':seed,'n_windows':n,'noise_multiplier':mult,'arm':arm,'err_balanced':errors[arm], 'val_vs_duplicate':errors['duplicate']-errors[arm]} for arm in blocks]

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--start',type=int,default=11000);a.add_argument('--nseeds',type=int,default=40);a.add_argument('--out',default='data/noise_value_sweep.csv');arg=a.parse_args()
    rk4(0,np.array([1.,1.,1.]),1,.01)
    rows_out=[]
    for sys in (0,1):
        for noise in (0.,1.,2.,4.):
            for k in range(arg.nseeds):
                rows_out.extend(one(sys,arg.start+k,noise))
            print('complete',sys,noise,flush=True)
    d=pd.DataFrame(rows_out);d.to_csv(arg.out,index=False)
    print(d.groupby(['system','noise_multiplier','arm']).val_vs_duplicate.agg(['median',lambda x:(x>0).sum()]).to_string())
