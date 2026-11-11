"""Standalone positive-wing/well symmetry screening sweep (not original phase3 protocol).
Frozen comparisons: ratio only <=.5 and ratio + absolute best defect <=.5.
New independent seeds supplied by caller. Selective acceptance means candidate correctness
is assessed on known simulation truth only *after* selection. No ground truth in gate.
"""
import os, itertools, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np, pandas as pd
from numba import njit
from scipy.signal import savgol_filter

@njit(cache=True)
def vf(system,x):
    if system==0:
        a,b,c=x[0],x[1],x[2]
        return np.array([10*(b-a),a*(28-c)-b,a*b-8/3*c])
    a,b=x[0],x[1]
    if system==1: return np.array([b,a-a*a*a])
    return np.array([b,a-a*a*a-.3*b+.15])
@njit(cache=True)
def rk4(system, x0,n,dt):
    out=np.empty((n,len(x0)))
    x=x0.copy()
    for i in range(n):
        k1=vf(system,x);k2=vf(system,x+.5*dt*k1)
        k3=vf(system,x+.5*dt*k2);k4=vf(system,x+dt*k3)
        x=x+dt*(k1+2*k2+2*k3+k4)/6
        out[i]=x
    return out

def windows(system, seed,nw,noise):
    rng=np.random.default_rng(seed)
    w=[]
    if system==0:
        # Distinct simulated trajectories; each window short and >=95% positive wing.
        for j in range(nw):
            s=np.array([1+rng.normal(scale=.4),1+rng.normal(scale=.4),1+rng.normal(scale=.4)])
            tr=rk4(system,s,6000,.01)[1800:]
            good=np.where(np.convolve((tr[:,0]>0).astype(float),np.ones(36),mode='valid')>=35)[0]
            if not len(good): raise RuntimeError('No Lorenz wing window')
            k=int(rng.choice(good));w.append(tr[k:k+36]+rng.normal(0,noise,(36,3)))
    else:
        for j in range(nw):
            for k in range(100):
                a=rng.uniform(.7,1.25);b=rng.normal(0,.12)
                E=.5*b*b-.5*a*a+.25*a**4
                if -.24<E<-.04: break
            tr=rk4(system,np.array([a,b]),120,.1)
            k=int(rng.integers(5,60))
            w.append(tr[k:k+40]+rng.normal(0,noise,(40,2)))
    return w

def basis(X,degree):
    d=X.shape[1]
    cl=[np.ones(X.shape[0])]
    for order in range(1,degree+1):
        for ids in itertools.combinations_with_replacement(range(d),order):
            cl.append(np.prod(X[:,ids],axis=1))
    return np.stack(cl,axis=1)

def evaluate(system,seed,nw,method='raw',lam=1e-4):
    noise=.015 if system==0 else .01
    d=3 if system==0 else 2
    degree=2 if system==0 else 3
    dt=.01 if system==0 else .1
    w=windows(system,seed,nw,noise)
    if method=='smooth':
        w=[savgol_filter(a,9,3,axis=0,mode='interp') for a in w]
    xx=np.concatenate([a[:-1] for a in w])
    yy=np.concatenate([a[1:]-a[:-1] for a in w])
    mu=np.concatenate(w).mean(0); sd=np.concatenate(w).std(0,ddof=1)
    sd=np.maximum(sd,1e-5)
    C=basis((xx-mu)/sd,degree)
    B=np.linalg.solve(C.T@C+lam*np.eye(C.shape[1]),C.T@(yy/sd))
    def fhat(x):return basis((x-mu)/sd,degree)@B*sd/dt
    F=fhat(xx); den=np.sqrt(np.mean(F**2))+1e-12
    cand=list(itertools.product([1,-1],repeat=d))[1:]
    defects=np.array([np.sqrt(np.mean((fhat(xx*np.array(s))-F*np.array(s))**2))/den for s in cand])
    order=np.argsort(defects)
    chosen=tuple(cand[order[0]])
    ratio=float(defects[order[0]]/max(defects[order[1]],1e-12))
    absolute=float(defects[order[0]])
    truth={0:(-1,-1,1),1:(-1,-1),2:None}[system]
    return dict(system=['Lorenz','Duffing','Biased damped Duffing'][system],seed=seed,n_windows=nw,method=method,degree=degree,chosen=str(chosen),truth=str(truth),correct=int(chosen==truth),defect=absolute,second=float(defects[order[1]]),ratio=ratio,ratio_accept=int(ratio<=.5),selective_accept=int(ratio<=.5 and absolute<=.5),
    ratio_correct=int(ratio<=.5 and chosen==truth),ratio_wrong=int(ratio<=.5 and chosen!=truth),selective_correct=int(ratio<=.5 and absolute<=.5 and chosen==truth),selective_wrong=int(ratio<=.5 and absolute<=.5 and chosen!=truth))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=3000);p.add_argument('--nseeds',type=int,default=35);p.add_argument('--out',default='data/selective_gate_sweep.csv');p.add_argument('--methods',default='raw,smooth');a=p.parse_args()
    rows=[]
    # compile before testing
    rk4(0,np.array([1.,1.,1.]),10,.01)
    for s in [0,1,2]:
        for n in [2,4,8,16]:
            for method in a.methods.split(','):
                for k in range(a.nseeds):
                    try:rows.append(evaluate(s,a.start+k,n,method))
                    except Exception as ex: print('FAIL',s,n,method,k,str(ex),flush=True)
                print('completed',s,n,method,flush=True)
    pd.DataFrame(rows).to_csv(a.out,index=False)
    df=pd.DataFrame(rows)
    print(df.groupby(['system','method','n_windows'])[['ratio_correct','ratio_wrong','selective_correct','selective_wrong']].sum().to_string())
