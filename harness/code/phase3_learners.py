from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from learners import make_lagged, fit_ridge_var, forecast_ridge, recursive_nrmse_ridge


def fit_mlp_var(series_list,lag=4,alpha=1e-3,seed=0,hidden=(32,32)):
    X,Y=make_lagged(series_list,lag)
    model=MLPRegressor(hidden_layer_sizes=hidden,activation='tanh',solver='adam',alpha=float(alpha),
                       max_iter=600,random_state=int(seed),tol=1e-5,early_stopping=True,
                       validation_fraction=0.15,n_iter_no_change=20,learning_rate_init=1e-3)
    model.fit(X,Y)
    return model


def forecast_mlp(model,history,horizon,lag=4):
    vals=[history[:,i].copy() for i in range(history.shape[1])];out=[]
    for _ in range(horizon):
        x=np.stack(vals[-lag:],axis=0).reshape(1,-1)
        y=np.asarray(model.predict(x)).reshape(-1)
        out.append(y.copy());vals.append(y.copy())
    return np.asarray(out).T


def _origins(y,min_hist,h,max_origins=5):
    if y.shape[1] <= min_hist+h: return []
    return np.linspace(min_hist,y.shape[1]-h-1,min(max_origins,y.shape[1]-min_hist-h),dtype=int)


def recursive_nrmse_mlp(model,series_list,lag=4,horizons=(1,4,8)):
    vals=[]
    for y in series_list:
        scale=float(np.std(y)+1e-8)
        for h in horizons:
            for t in _origins(y,lag,h):
                pred=forecast_mlp(model,y[:,:t],h,lag);truth=y[:,t:t+h]
                vals.append(float(np.sqrt(np.mean((pred-truth)**2))/scale))
    return float(np.mean(vals)) if vals else float('inf')


def tune_ridge(train,tune,alphas,lag,horizons):
    rec=[]
    for a in alphas:
        B=fit_ridge_var(train,lag,float(a));loss=recursive_nrmse_ridge(B,tune,lag,horizons)[0]
        rec.append((float(loss),float(a)))
    rec.sort();return rec[0][1],rec


def tune_mlp(train,tune,alphas,lag,horizons,seed):
    rec=[]
    for a in alphas:
        try:
            M=fit_mlp_var(train,lag,float(a),seed=seed);loss=recursive_nrmse_mlp(M,tune,lag,horizons)
        except Exception:
            loss=float('inf')
        rec.append((float(loss),float(a)))
    rec.sort();return rec[0][1],rec


def eval_learner(kind,train,evalset,alpha,lag,horizons,seed=0):
    if kind=='ridge':
        B=fit_ridge_var(train,lag,float(alpha)); return recursive_nrmse_ridge(B,evalset,lag,horizons)[0],B
    M=fit_mlp_var(train,lag,float(alpha),seed=seed); return recursive_nrmse_mlp(M,evalset,lag,horizons),M


def fitted_ridge_generator(train,source,alpha,lag):
    """Generate deterministic synthetic trajectory from the fitted ridge VAR itself.

    The first lag states are taken from source; all later rows satisfy the fitted
    recurrence to numerical precision, making this the exact same-model
    self-consistency arm for the ridge learner.
    """
    B=fit_ridge_var(train,lag,float(alpha))
    hist=source[:,:lag]
    pred=forecast_ridge(B,hist,source.shape[1]-lag,lag)
    return np.concatenate([hist,pred],axis=1),B

def recursive_nrmse_mlp_batched(model,series_list,lag=4,horizons=(1,4,8),max_origins=5):
    """Numerically equivalent recursive NRMSE with batched origins to reduce predict overhead."""
    vals=[]
    for y in series_list:
        scale=float(np.std(y)+1e-8)
        for h in horizons:
            origins=_origins(y,lag,h,max_origins=max_origins)
            if len(origins)==0: continue
            # one independent rolling history per origin
            histories=[[y[:,i].copy() for i in range(t)] for t in origins]
            preds=[[] for _ in origins]
            for _step in range(h):
                X=np.stack([np.stack(hist[-lag:],axis=0).reshape(-1) for hist in histories],axis=0)
                Y=np.asarray(model.predict(X))
                if Y.ndim==1: Y=Y[:,None]
                for j,yy in enumerate(Y):
                    histories[j].append(yy.copy());preds[j].append(yy.copy())
            for j,t in enumerate(origins):
                pred=np.asarray(preds[j]).T;truth=y[:,t:t+h]
                vals.append(float(np.sqrt(np.mean((pred-truth)**2))/scale))
    return float(np.mean(vals)) if vals else float('inf')
