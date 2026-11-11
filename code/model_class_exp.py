"""Model-class experiment: local linear AR(4) vs global quadratic-library learner on one-wing Lorenz.

Both learners are linear in their parameters, so the exact block-update identity applies to both.
Splits, seeds, symmetries, tuning grid and eligibility rule follow the frozen Phase-3 protocol.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

PH3 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PH3 / 'code'))
from phase3_common import (build_lorenz_split, fit_standardizer, apply_standardizer, apply_symmetry,
                           lorenz_reservoir, extract_wing_windows)
from baselines import time_warp

M = json.loads((PH3 / 'protocol/phase3_protocol.json').read_text())['lorenz']
LAG = M['lag']; H = tuple(M['forecast_horizons']); GRID = M['ridge_alpha_grid']
NTEST = M['test_windows_each_wing']

# ---------------- learners (both linear in parameters) ----------------
def quad_features(x):  # x: (n,3) -> (n,10): 1, x, upper-triangular x_i x_j
    n = x.shape[0]; cols = [np.ones((n, 1)), x]
    for i in range(3):
        for j in range(i, 3):
            cols.append((x[:, i] * x[:, j])[:, None])
    return np.hstack(cols)

class Learner:
    def __init__(self, kind): self.kind = kind
    def design(self, series_list):
        X, Y = [], []
        for y in series_list:
            y = np.asarray(y, float); n = y.shape[1]
            for t in range(LAG, n):
                if self.kind == 'linear':
                    X.append(y[:, t - LAG:t].T.reshape(-1)); Y.append(y[:, t])
                else:
                    X.append(quad_features(y[:, t - 1][None, :])[0]); Y.append(y[:, t] - y[:, t - 1])
        return np.asarray(X), np.asarray(Y)
    def fit(self, series_list, alpha):
        X, Y = self.design(series_list)
        return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    def step(self, B, hist):  # hist: list of state vectors (most recent last), batched: (k, lag, 3)
        if self.kind == 'linear':
            return hist.reshape(hist.shape[0], -1) @ B
        last = hist[:, -1, :]
        return last + quad_features(last) @ B
    def rollout(self, B, init_hist, n):  # init_hist (3, LAG)
        h = [init_hist[:, i] for i in range(init_hist.shape[1])]
        for _ in range(n):
            nxt = self.step(B, np.stack(h[-LAG:], 0)[None])[0]
            if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e3: nxt = np.clip(np.nan_to_num(nxt, posinf=1e3, neginf=-1e3), -1e3, 1e3)
            h.append(nxt)
        return np.stack(h, 1)

def _origins(y, min_hist, h, max_origins=5):
    if y.shape[1] <= min_hist + h: return []
    return np.linspace(min_hist, y.shape[1] - h - 1, min(max_origins, y.shape[1] - min_hist - h), dtype=int)

def evaluate(L, B, series_list):
    vals = []; byh = {h: [] for h in H}
    for y in series_list:
        scale = float(np.std(y) + 1e-8)
        for h in H:
            for t in _origins(y, LAG, h):
                pred = L.rollout(B, y[:, t - LAG:t], h)[:, LAG:]
                e = float(np.sqrt(np.mean((pred - y[:, t:t + h]) ** 2)) / scale)
                e = min(e, 10.0)  # cap diverged rollouts (logged via count)
                vals.append(e); byh[h].append(e)
    return float(np.mean(vals)), {h: float(np.mean(v)) for h, v in byh.items()}

def tune(L, train, tuneset):
    rec = sorted((evaluate(L, L.fit(train, a), tuneset)[0], a) for a in GRID)
    return rec[0][1]

# ---------------- one seed ----------------
def one_seed(seed):
    Lr = M['trajectory_length']; noise = M['observational_noise_sd']; frac = M['minimum_training_wing_fraction']
    raw = build_lorenz_split(seed, Lr, M['tuning_windows_each_wing'], M['headroom_windows_each_wing'], NTEST, noise, frac)
    mu, sd = fit_standardizer(raw['train'])
    z = {k: (apply_standardizer(v, mu, sd) if isinstance(v, list) else apply_standardizer([v], mu, sd)[0]) for k, v in raw.items()}
    src = z['train'][0]
    rsame = lorenz_reservoir(700000 + seed * 31, obs_noise=noise)
    same = apply_standardizer([extract_wing_windows(rsame, Lr, +1, 1, min_fraction=frac, min_separation=Lr)[0]], mu, sd)[0]
    tw = time_warp(src, np.random.default_rng(901003 + seed), M['time_warp_target_relative_movement'])
    learners = {'linear': Learner('linear'), 'quadratic': Learner('quadratic')}
    gens = {}
    for name, L in learners.items():
        a0 = tune(L, z['train'], z['tune'])
        gens[name] = L.rollout(L.fit(z['train'], a0), src[:, :LAG], src.shape[1] - LAG)
    arms = {
        'base': None, 'duplicate': src.copy(),
        'false_symmetry': apply_standardizer([apply_symmetry(raw['train'][0], 'false')], mu, sd)[0],
        'true_symmetry': apply_standardizer([apply_symmetry(raw['train'][0], 'true')], mu, sd)[0],
        'same_wing_real': same, 'opposite_wing_real': z['extra_real'], 'oracle': z['oracle'], 'time_warp': tw,
    }
    rows = []
    pos, neg = z['test'][:NTEST], z['test'][NTEST:]
    for name, L in learners.items():
        A = dict(arms)
        A['fitted_model'] = gens[name]                                  # same-model generator
        A['other_model'] = gens['quadratic' if name == 'linear' else 'linear']  # generator-learner gap
        Bbase = L.fit(z['train'], tune(L, z['train'], z['tune']))
        for arm, cand in A.items():
            tr = z['train'] if cand is None else z['train'] + [cand]
            a = tune(L, tr, z['tune']); B = L.fit(tr, a)
            # learner-relative residual of the candidate under the base learner
            if cand is None: rres = np.nan
            else:
                C, U = L.design([cand]); rres = float(np.sqrt(np.mean((U - C @ Bbase) ** 2)))
            row = {'seed': seed, 'learner': name, 'arm': arm, 'alpha': a, 'residual_rmse_under_base': rres,
                   'headroom_loss': evaluate(L, B, z['headroom'])[0]}
            for tag, S in [('balanced', z['test']), ('positive', pos), ('negative', neg)]:
                l, lh = evaluate(L, B, S); row[f'{tag}_loss'] = l
                for h in H: row[f'{tag}_h{h}'] = lh[h]
            rows.append(row)
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=2, verbose=5)(delayed(one_seed)(s) for s in range(int(M['seeds'])))
    df = pd.DataFrame([r for rr in outs for r in rr]); df.to_csv(OUT / 'model_class_results.csv', index=False)
    print('done', df.shape)
