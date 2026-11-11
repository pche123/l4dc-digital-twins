"""Second system: Hamiltonian double-well Duffing oscillator, x'' = x - x^3.

Same protocol as the Lorenz study: train on one well, balanced tuning/calibration/test,
per-arm ridge tuning, 10% calibration-headroom eligibility, recursive NRMSE at h in {1,4,8}.
Learners: local linear AR(4) and a cubic library (contains the true vector field).
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd

PH3 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PH3 / 'code'))
from baselines import time_warp

DT, LEN, NOISE, LAG, H = 0.1, 40, 0.01, 4, (1, 4, 8)
GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
S_TRUE = np.diag([-1., -1.]); S_FALSE = np.diag([-1., 1.])

def f(s): return np.array([s[1], s[0] - s[0] ** 3])

def integrate(s0, n):
    s = np.array(s0, float); out = []
    for _ in range(n):
        k1 = f(s); k2 = f(s + .5 * DT * k1); k3 = f(s + .5 * DT * k2); k4 = f(s + DT * k3)
        s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(s.copy())
    return np.array(out).T

def window(rng, sign):
    # energy E = v^2/2 - x^2/2 + x^4/4 in (-0.24, -0.03): bounded orbit inside one well
    while True:
        x0 = sign * rng.uniform(0.35, 1.38); v0 = rng.normal(0, 0.15)
        E = v0 ** 2 / 2 - x0 ** 2 / 2 + x0 ** 4 / 4
        if -0.24 < E < -0.03: break
    burn = int(rng.integers(0, 60))
    tr = integrate([x0, v0], burn + LEN)[:, burn:]
    return tr + rng.normal(0, NOISE, tr.shape)

def split(seed):
    r = lambda k: np.random.default_rng(seed * 1000 + k)
    return {'train': [window(r(1), +1), window(r(2), +1)],
            'tune': [window(r(10 + i), s) for i in range(5) for s in (+1, -1)],
            'headroom': [window(r(30 + i), s) for i in range(5) for s in (+1, -1)],
            'test': [window(r(50 + i), +1) for i in range(10)] + [window(r(70 + i), -1) for i in range(10)],
            'same': window(r(90), +1), 'opposite': window(r(91), -1), 'oracle': window(r(92), -1)}

def cubic_features(x):
    a, b = x[:, 0:1], x[:, 1:2]
    return np.hstack([np.ones_like(a), a, b, a * a, a * b, b * b, a ** 3, a * a * b, a * b * b, b ** 3])

class Learner:
    def __init__(self, kind): self.kind = kind
    def design(self, series):
        X, Y = [], []
        for y in series:
            for t in range(LAG, y.shape[1]):
                if self.kind == 'linear': X.append(y[:, t - LAG:t].T.reshape(-1)); Y.append(y[:, t])
                else: X.append(cubic_features(y[:, t - 1][None])[0]); Y.append(y[:, t] - y[:, t - 1])
        return np.array(X), np.array(Y)
    def fit(self, series, alpha):
        X, Y = self.design(series); return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    def rollout(self, B, init, n):
        h = [init[:, i] for i in range(init.shape[1])]
        for _ in range(n):
            if self.kind == 'linear': nxt = np.stack(h[-LAG:], 0).reshape(-1) @ B
            else: nxt = h[-1] + cubic_features(h[-1][None])[0] @ B
            h.append(np.clip(np.nan_to_num(nxt, nan=1e3), -1e3, 1e3))
        return np.stack(h, 1)

def origins(y, h): return np.linspace(LAG, y.shape[1] - h - 1, min(5, y.shape[1] - LAG - h), dtype=int)

def evaluate(L, B, series):
    vals = []; byh = {h: [] for h in H}
    for y in series:
        sc = float(np.std(y) + 1e-8)
        for h in H:
            for t in origins(y, h):
                p = L.rollout(B, y[:, t - LAG:t], h)[:, LAG:]
                e = min(float(np.sqrt(np.mean((p - y[:, t:t + h]) ** 2)) / sc), 10.0); vals.append(e); byh[h].append(e)
    return float(np.mean(vals)), {h: float(np.mean(v)) for h, v in byh.items()}

def tune(L, tr, tu): return sorted((evaluate(L, L.fit(tr, a), tu)[0], a) for a in GRID)[0][1]

def one_seed(seed):
    raw = split(seed); A = np.concatenate(raw['train'], 1)
    mu = A.mean(1, keepdims=True); sd = A.std(1, ddof=1, keepdims=True)
    st = lambda x: (x - mu) / sd
    z = {k: ([st(w) for w in v] if isinstance(v, list) else st(v)) for k, v in raw.items()}
    src = z['train'][0]
    learners = {'linear': Learner('linear'), 'cubic': Learner('cubic')}
    gens = {n: L.rollout(L.fit(z['train'], tune(L, z['train'], z['tune'])), src[:, :LAG], LEN - LAG) for n, L in learners.items()}
    arms = {'base': None, 'duplicate': src.copy(), 'false_symmetry': st(S_FALSE @ raw['train'][0]),
            'true_symmetry': st(S_TRUE @ raw['train'][0]), 'same_wing_real': z['same'], 'opposite_wing_real': z['opposite'],
            'oracle': z['oracle'], 'time_warp': time_warp(src, np.random.default_rng(901003 + seed), 0.03)}
    rows = []; pos, neg = z['test'][:10], z['test'][10:]
    for name, L in learners.items():
        Aa = dict(arms); Aa['fitted_model'] = gens[name]; Aa['other_model'] = gens['cubic' if name == 'linear' else 'linear']
        Bb = L.fit(z['train'], tune(L, z['train'], z['tune']))
        for arm, cand in Aa.items():
            tr = z['train'] if cand is None else z['train'] + [cand]
            a = tune(L, tr, z['tune']); B = L.fit(tr, a)
            if cand is None: rres = np.nan
            else: C, U = L.design([cand]); rres = float(np.sqrt(np.mean((U - C @ Bb) ** 2)))
            row = dict(seed=seed, learner=name, arm=arm, alpha=a, residual_rmse_under_base=rres, headroom_loss=evaluate(L, B, z['headroom'])[0])
            for tag, S in [('balanced', z['test']), ('positive', pos), ('negative', neg)]:
                l, lh = evaluate(L, B, S); row[f'{tag}_loss'] = l
                for h in H: row[f'{tag}_h{h}'] = lh[h]
            rows.append(row)
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=2)(delayed(one_seed)(s) for s in range(30))
    df = pd.DataFrame([r for rr in outs for r in rr]); df.to_csv(OUT / 'duffing_results.csv', index=False); print(df.shape)
