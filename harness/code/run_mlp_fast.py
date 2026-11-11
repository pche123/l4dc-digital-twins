from __future__ import annotations
import argparse,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from baselines import time_warp
from phase3_common import build_lorenz_split,fit_standardizer,apply_standardizer,apply_symmetry,spectrum_project_delta,rel_move
from phase3_learners import fit_mlp_var,recursive_nrmse_mlp_batched,fitted_ridge_generator
from learners import fit_ridge_var
warnings.filterwarnings('ignore')

def fit_grid(train,tune,alphas,lag,H,seed):
    records=[]
    for a in alphas:
        M=fit_mlp_var(train,lag,float(a),seed=seed)
        loss=recursive_nrmse_mlp_batched(M,tune,lag,H)
        records.append((float(loss),float(a),M))
    records.sort(key=lambda z:(z[0],z[1]));return records

def run_one(seed,m):
    raw=build_lorenz_split(seed,m['trajectory_length'],m['tuning_windows_each_wing'],m['headroom_windows_each_wing'],m['test_windows_each_wing'],m['observational_noise_sd'],m['minimum_training_wing_fraction'])
    mu,sd=fit_standardizer(raw['train']); z={k:(apply_standardizer(v,mu,sd) if isinstance(v,list) else apply_standardizer([v],mu,sd)[0]) for k,v in raw.items()}
    z['true_symmetry']=apply_standardizer([apply_symmetry(raw['train'][0],'true')],mu,sd)[0];z['false_symmetry']=apply_standardizer([apply_symmetry(raw['train'][0],'false')],mu,sd)[0]
    lag=m['lag'];H=tuple(m['forecast_horizons']);grid=m['mlp_alpha_grid'];source=z['train'][0]
    # generator alpha from tuned ridge base, as preregistered
    best_ra=None;best_rl=np.inf
    for a in m['ridge_alpha_grid']:
        B=fit_ridge_var(z['train'],lag,float(a))
        from learners import recursive_nrmse_ridge
        L=recursive_nrmse_ridge(B,z['tune'],lag,H)[0]
        if L<best_rl:best_rl=L;best_ra=float(a)
    fitted,_=fitted_ridge_generator(z['train'],source,best_ra,lag)
    rng=np.random.default_rng(901003+seed);tw=time_warp(source,rng,m['time_warp_target_relative_movement']);twp,pmeta=spectrum_project_delta(source,tw)
    arms={'base':None,'duplicate':source.copy(),'fitted_model':fitted,'false_symmetry':z['false_symmetry'],'true_symmetry':z['true_symmetry'],'extra_real':z['extra_real'],'oracle_simulator':z['oracle'],'time_warp':tw,'time_warp_spectrum_projected':twp}
    # Base diagnostic model uses its own tuned alpha
    base_grid=fit_grid(z['train'],z['tune'],grid,lag,H,seed*1000+1); base_model=base_grid[0][2];base_alpha=base_grid[0][1]
    from learners import make_lagged
    rows=[]
    for j,(arm,cand) in enumerate(arms.items()):
        tr=z['train'] if cand is None else z['train']+[cand]
        rec=base_grid if arm=='base' else fit_grid(tr,z['tune'],grid,lag,H,seed*1000+101+j)
        tune_loss,alpha,M=rec[0]
        head=recursive_nrmse_mlp_batched(M,z['headroom'],lag,H);test=recursive_nrmse_mlp_batched(M,z['test'],lag,H)
        if cand is None: rf=rr=np.nan
        else:
            X,Y=make_lagged([cand],lag);R=Y-base_model.predict(X);rf=float(np.linalg.norm(R));rr=float(np.sqrt(np.mean(R*R)))
        rows.append({'seed':seed,'learner':'mlp','mode':'retuned_alpha','arm':arm,'alpha':alpha,'tune_loss':tune_loss,'headroom_loss':head,'test_loss':test,'candidate_residual_fro_under_base':rf,'candidate_residual_rmse_under_base':rr,'train_positive_wing_fraction':float(np.mean(np.concatenate(raw['train'],axis=1)[0]>0)),'meta_linear_sv_after':pmeta['linear_sv_change_after'] if arm=='time_warp_spectrum_projected' else np.nan})
    return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--jobs',type=int,default=8);p.add_argument('--n-seeds',type=int,default=None);a=p.parse_args();m=json.loads(Path(a.manifest).read_text())['lorenz']; n=int(a.n_seeds or m['seeds'])
 outs=Parallel(n_jobs=a.jobs,backend='loky')(delayed(run_one)(s,m) for s in range(n))
 rows=[x for r in outs for x in r];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'lorenz_symmetry_results.csv',index=False);print({'rows':len(rows),'seeds':n})
