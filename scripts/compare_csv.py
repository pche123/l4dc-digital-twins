"""Compare two experiment CSVs without sorting or changing the archived data."""
from __future__ import annotations
import argparse, sys
import numpy as np
import pandas as pd

def compare(a,b,tol=1e-10):
    x,y=pd.read_csv(a),pd.read_csv(b)
    issues=[]
    if x.shape != y.shape: issues.append(f'row/column count differs: {x.shape} vs {y.shape}')
    if list(x.columns)!=list(y.columns): issues.append('column names/order differs')
    if issues:return False, '; '.join(issues),float('inf')
    # Seed/joblib completion order can change row order even when each experiment is identical.
    # Sort on identifying columns; this is NOT a change to the archived datasets on disk.
    id_candidates=['system','seed','learner','arm','length','noise','eps','verifier','transform','wmax','law','rho','n','sigma','wells','cand','regime','noise_from','tau','r','setting']
    ids=[c for c in id_candidates if c in x.columns]
    if ids:
        x=x.sort_values(ids,kind='mergesort').reset_index(drop=True)
        y=y.sort_values(ids,kind='mergesort').reset_index(drop=True)
    maxdiff=0.
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            aa,bb=x[c].to_numpy(dtype=float),y[c].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(aa),np.isnan(bb)):
                issues.append(f'{c}: NaN locations differ');continue
            with np.errstate(invalid="ignore"):
                diff=np.abs(aa-bb)
            if np.isinf(aa).any() or np.isinf(bb).any():
                if not np.array_equal(np.isinf(aa),np.isinf(bb)):
                    issues.append(f'{c}: infinity locations differ')
                diff[np.isinf(aa)&np.isinf(bb)&(np.sign(aa)==np.sign(bb))]=0.
            diff=diff[~np.isnan(diff)]
            m=float(np.max(diff)) if len(diff) else 0.
            maxdiff=max(maxdiff,m)
            if m>tol:issues.append(f'{c}: maximum absolute error {m:.3g}')
        else:
            aa=x[c].fillna('__MISSING__').astype(str)
            bb=y[c].fillna('__MISSING__').astype(str)
            if not aa.equals(bb):issues.append(f'{c}: nonnumeric values differ')
    return not issues, ('OK' if not issues else '; '.join(issues)),maxdiff

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archived');p.add_argument('regenerated');p.add_argument('--tol',type=float,default=1e-10)
    a=p.parse_args();ok,msg,v=compare(a.archived,a.regenerated,a.tol)
    print(f'{"PASS" if ok else "FAIL"}: {msg} | max absolute numerical difference = {v:.3g}')
    sys.exit(0 if ok else 1)
