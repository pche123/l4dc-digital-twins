"""Direct imposition on one-wing Lorenz (standardised coordinates, same protocol/units as model_class_exp.py).
Equivariant model: g~(h)=1/2(g(h)+A(g(S'h)-b)) for next-state AR, f~(y)=1/2(f(y)+A f(S'y)) for the increment library, with
S'(y)=A y+b the symmetry expressed in standardised coordinates. usage: python code/imposed_lorenz.py harness data"""
import sys, numpy as np, pandas as pd
import model_class_exp as MC
from model_class_exp import build_lorenz_split, fit_standardizer, apply_standardizer

def sym_map(raw_sym, mu, sd):
    S = np.diag(raw_sym).astype(float); A = S.copy(); b = ((S @ mu - mu) / sd).reshape(-1); return A, b  # sd diagonal => A=S

def Sp(y, A, b): return A @ y + b[:, None]

def eq_design(L, series, A, b):
    X, Y = L.design(series); XS, _ = L.design([Sp(y, A, b) for y in series]); return X, XS, Y

def fit_eq(L, series, alpha, A, b):
    X, XS, Y = eq_design(L, series, A, b); cols = []
    for k in range(3):
        a = A[k, k]; Z = .5 * (X + a * XS)
        tgt = Y[:, k] - (.5 * (-a * b[k]) if L.kind == 'linear' else 0.)
        cols.append(np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ tgt))
    return np.array(cols).T

def pred_eq(L, B, hist, A, b):
    # hist (lag,3) most recent last -> next state
    h = hist[None]; hs = (hist @ A.T + b)[None]
    if L.kind == 'linear':
        g = h.reshape(1, -1) @ B; gs = hs.reshape(1, -1) @ B
        return (.5 * (g + (gs - b) @ A.T))[0]
    last = hist[-1][None]; lasts = hs[:, -1, :]
    f = MC.quad_features(last) @ B; fs = MC.quad_features(lasts) @ B
    return last[0] + .5 * (f + fs @ A.T)[0]

def roll_eq(L, B, init, n, A, b):
    h = [init[:, i] for i in range(init.shape[1])]
    for _ in range(n):
        nxt = pred_eq(L, B, np.stack(h[-MC.LAG:], 0), A, b)
        h.append(np.clip(np.nan_to_num(nxt, nan=1e3, posinf=1e3, neginf=-1e3), -1e3, 1e3))
    return np.stack(h, 1)

def ev_eq(L, B, series, A, b):
    vals = []
    for y in series:
        sc = float(np.std(y) + 1e-8)
        for h in MC.H:
            for t in MC._origins(y, MC.LAG, h):
                p = roll_eq(L, B, y[:, t - MC.LAG:t], h, A, b)[:, MC.LAG:]
                vals.append(min(float(np.sqrt(np.mean((p - y[:, t:t + h]) ** 2)) / sc), 10.0))
    return float(np.mean(vals))

def one(seed):
    Lr = MC.M['trajectory_length']; noise = MC.M['observational_noise_sd']; frac = MC.M['minimum_training_wing_fraction']
    raw = build_lorenz_split(seed, Lr, MC.M['tuning_windows_each_wing'], MC.M['headroom_windows_each_wing'], MC.NTEST, noise, frac)
    mu, sd = fit_standardizer(raw['train'])
    z = {k: (apply_standardizer(v, mu, sd) if isinstance(v, list) else apply_standardizer([v], mu, sd)[0]) for k, v in raw.items()}
    rows = []; pos, neg = z['test'][:MC.NTEST], z['test'][MC.NTEST:]
    for name in ('linear', 'quadratic'):
        L = MC.Learner(name)
        for arm, kind in [('impose_true', 'true'), ('impose_false', 'false')]:
            A, b = sym_map(np.array([-1., -1., 1.]) if kind == 'true' else np.array([-1., 1., 1.]), mu, sd)
            def fitf(a): return fit_eq(L, z['train'], a, A, b)
            best = sorted((ev_eq(L, fitf(a), z['tune'], A, b), a) for a in MC.GRID)[0][1]; B = fitf(best)
            rows.append(dict(seed=seed, learner=name, arm=arm, alpha=best, balanced_loss=ev_eq(L, B, z['test'], A, b),
                             positive_loss=ev_eq(L, B, pos, A, b), negative_loss=ev_eq(L, B, neg, A, b), headroom_loss=ev_eq(L, B, z['headroom'], A, b)))
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=2)(delayed(one)(s) for s in range(int(MC.M['seeds'])))
    pd.DataFrame([r for rr in out for r in rr]).to_csv(MC.OUT / 'imposed_symmetry_lorenz.csv', index=False); print('done')
