from __future__ import annotations
import numpy as np

def make_lagged(series_list, lag:int=4):
    X=[];Y=[]
    for y in series_list:
        y=np.asarray(y,float); c,n=y.shape
        for t in range(lag,n): X.append(y[:,t-lag:t].T.reshape(-1));Y.append(y[:,t])
    return np.asarray(X),np.asarray(Y)

def fit_ridge_var(series_list,lag=4,alpha=1e-3):
    X,Y=make_lagged(series_list,lag); return np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@Y)

def forecast_ridge(B,history,horizon,lag=4):
    vals=[history[:,i].copy() for i in range(history.shape[1])];out=[]
    for _ in range(horizon):
        y=np.stack(vals[-lag:],axis=0).reshape(-1)@B; out.append(y.copy());vals.append(y.copy())
    return np.asarray(out).T

def fit_dmd(series_list,obs,lam=1e-3):
    X=[];Y=[]
    for x in series_list:
        z=obs.transform(x); X.append(z[:,:-1].T);Y.append(z[:,1:].T)
    X=np.vstack(X);Y=np.vstack(Y); V=lam*np.eye(X.shape[1])+X.T@X
    return np.linalg.solve(V,X.T@Y).T

def forecast_dmd(K,history,horizon,obs):
    z=obs.transform(history)[:,-1].copy(); out=[]
    for _ in range(horizon): z=K@z; out.append(obs.decode(z).copy())
    return np.asarray(out).T

def _origins(y, min_hist, h, max_origins=5):
    if y.shape[1] <= min_hist+h: return []
    return np.linspace(min_hist,y.shape[1]-h-1,min(max_origins,y.shape[1]-min_hist-h),dtype=int)

def recursive_nrmse_ridge(B,series_list,lag=4,horizons=(1,4,8,16)):
    vals=[];byh={h:[] for h in horizons}
    for y in series_list:
        scale=float(np.std(y)+1e-8)
        for h in horizons:
            for t in _origins(y,lag,h):
                pred=forecast_ridge(B,y[:,:t],h,lag);truth=y[:,t:t+h]
                e=float(np.sqrt(np.mean((pred-truth)**2))/scale); vals.append(e);byh[h].append(e)
    return float(np.mean(vals)),{h:(float(np.mean(v)) if v else np.nan) for h,v in byh.items()}

def recursive_nrmse_dmd(K,series_list,obs,horizons=(1,4,8,16)):
    vals=[];byh={h:[] for h in horizons}; min_hist=max(4,obs.delay)
    for y in series_list:
        scale=float(np.std(y)+1e-8)
        for h in horizons:
            for t in _origins(y,min_hist,h):
                pred=forecast_dmd(K,y[:,:t],h,obs);truth=y[:,t:t+h]
                e=float(np.sqrt(np.mean((pred-truth)**2))/scale);vals.append(e);byh[h].append(e)
    return float(np.mean(vals)),{h:(float(np.mean(v)) if v else np.nan) for h,v in byh.items()}

def train_only_select_observable(train,obs_lib,horizons=(1,4,8),lam=1e-3):
    # source-level 2-fold CV: each real training trajectory serves once as held-out source.
    scores=[]
    for obs in obs_lib:
        fold=[]
        try:
            for i in range(len(train)):
                tr=[train[j] for j in range(len(train)) if j!=i]; va=[train[i]]
                K=fit_dmd(tr,obs,lam); loss,_=recursive_nrmse_dmd(K,va,obs,horizons)
                fold.append(loss)
            score=float(np.mean(fold))
        except (np.linalg.LinAlgError,ValueError,FloatingPointError): score=float('inf')
        scores.append((score,obs.name))
    scores.sort(key=lambda z:(z[0],z[1])); return scores[0][1],scores
