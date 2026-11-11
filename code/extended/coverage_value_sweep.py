"""Independent one-step deployment-loss sweep for polynomial dynamics learners.
Not same evaluation or tuning as v0.7; independent noise-free test distribution.
"""
import os,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.signal import savgol_filter
from selective_gate_sweep import windows,basis,rk4

def standardizer(w):
    a=np.concatenate(w)
    return a.mean(0),np.maximum(a.std(0,ddof=1),1e-5)

def rows(w,mu,sd,degree):
    x=np.concatenate([a[:-1] for a in w]); y=np.concatenate([a[1:] for a in w]);
    return basis((x-mu)/sd,degree), (y-x)/sd

def fit(X,Y,ridge):
    return np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@Y)

def target_values(system, w):
    x=np.concatenate([a[:-1] for a in w])
    dt=.01 if system==0 else .1
    out=np.array([rk4(system,np.array(a,dtype=float),1,dt)[0] for a in x])
    return x,out-x

def one(system,seed,n,method,lam):
    noise=.015 if system==0 else .01
    degree=2 if system==0 else 3
    W=windows(system,seed,n,noise)
    if method=='smooth':W=[savgol_filter(a,9,3,axis=0,mode='interp') for a in W]
    mu,sd=standardizer(W)
    X,Y=rows(W,mu,sd,degree)
    B0=fit(X,Y,lam)
    Strue=np.array((-1,-1,1) if system==0 else (-1,-1))
    Sfalse=np.array((-1,1,1) if system==0 else (-1,1))
    independent=windows(system,seed+500000,1,noise)
    if method=='smooth':independent=[savgol_filter(a,9,3,axis=0,mode='interp') for a in independent]
    blocks={'duplicate':[W[0]],'exact':[W[0]*Strue], 'false':[W[0]*Sfalse],
            'same':[independent[0]], 'missing':[independent[0]*Strue]}
    # Evaluation states are new clean orbits, positive and actual symmetry images.
    Wt=windows(system,seed+1000000,8,0.)
    xp,yp=target_values(system,Wt)
    xm,ym=target_values(system,[a*Strue for a in Wt])
    teststates={'positive':(xp,yp),'missing':(xm,ym)}
    predictors={'base':B0}
    for name,wb in blocks.items():
        C,U=rows(wb,mu,sd,degree)
        predictors[name]=fit(np.vstack((X,C)),np.vstack((Y,U)),lam)
    errors={}
    for arm,B in predictors.items():
        by={}
        for region,(xt,yt) in teststates.items():
            z=basis((xt-mu)/sd,degree)
            p=z@B*sd
            # target increments, no derivative division.
            by[region]=float(np.mean(((p-yt)/sd)**2))
        errors[arm]=by
    rows_out=[]
    for arm in predictors:
        e=errors[arm]
        rows_out.append(dict(system='Lorenz' if system==0 else 'Duffing',seed=seed,n_windows=n,method=method,arm=arm,err_positive=e['positive'],err_missing=e['missing'],err_balanced=(e['positive']+e['missing'])/2,
          val_vs_duplicate=errors['duplicate']['positive']/2+errors['duplicate']['missing']/2-(e['positive']+e['missing'])/2))
    return rows_out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=6000);p.add_argument('--nseeds',type=int,default=40);p.add_argument('--method',default='smooth');p.add_argument('--out',default='data/coverage_value_sweep.csv');a=p.parse_args()
    rk4(0,np.array([1.,1.,1.]),1,.01)
    R=[]
    for sys in (0,1):
        for n in (2,4,8,16):
            for k in range(a.nseeds):
                R.extend(one(sys,a.start+k,n,a.method,1e-4))
            print('completed',sys,n,flush=True)
    df=pd.DataFrame(R);df.to_csv(a.out,index=False)
    print(df.groupby(['system','n_windows','arm'])[['err_balanced','val_vs_duplicate']].median().round(4).to_string())
