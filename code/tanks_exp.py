"""Real-data case study on the cascaded-tanks benchmark (Schoukens et al., 2016/2017): two tanks, pump voltage u in,
lower-tank level y out, Ts = 4 s, one 1024-sample estimation record and one validation record.
Bernoulli model:  x1' = -k1 g1(x1) + k4 u,   x2' = k2 g1(x1) - k3 g2(x2),   y = x2 + e,   g = sqrt (Torricelli) or identity.
Nothing in the model knows about overflow: the output saturates near 10 when the upper tank overflows.

Question: can the low-level part of the data certify the outflow law (sqrt vs linear), and does a twin built on the certified
law make synthetic high-level data that help a NARX learner?  Training sees only rows with y <= LC ("low level").
Regions of the held-out record, by the output level at the origin: low (<= LC), high (LC < y < OV, law valid), overflow (y >= OV).
Two replicates: estimation->validation and validation->estimation (the same device, so not independent).

usage: python code/tanks_exp.py <dataBenchmark.csv> data      writes data/tanks_laws.csv, data/tanks_results.csv
"""
import sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')

TS, SUB = 4.0, 8
LC, OV = 6.0, 9.5
NY, NU = 2, 2
ALPHAS = [1e-4, 1e-2, 1, 1e2]
H = (1, 5, 20)
LAWS = {'sqrt-sqrt': (.5, .5), 'sqrt-lin': (.5, 1.), 'lin-sqrt': (1., .5), 'lin-lin': (1., 1.)}   # (exponent of x1 outflow, of x2 outflow)

def power(x, p): return np.sqrt(np.maximum(x, 0)) if p == .5 else x

def simulate(theta, p, u, x1_0, x2_0):
    k1, k2, k3, k4 = theta; dt = TS / SUB; x1, x2 = x1_0, x2_0; out = np.empty(len(u))
    for t in range(len(u)):
        out[t] = x2
        for _ in range(SUB):
            g1 = power(x1, p[0]); d1 = -k1 * g1 + k4 * u[t]; d2 = k2 * g1 - k3 * power(x2, p[1])
            x1 = max(x1 + dt * d1, 0.); x2 = max(x2 + dt * d2, 0.)
    return out

def segments(y, lo_only=True, minlen=12):
    m = y <= LC; segs, s = [], None
    for i, v in enumerate(m):
        if v and s is None: s = i
        if (not v or i == len(m) - 1) and s is not None:
            e = i + 1 if v else i
            if e - s >= minlen: segs.append((s, e))
            s = None
    return segs

def resid(par, p, u, y, segs):
    theta = np.exp(par[:4]); r = []
    for j, (a, b) in enumerate(segs):
        sim = simulate(theta, p, u[a:b], np.exp(par[4 + j]), y[a]); r.append(sim - y[a:b])
    return np.concatenate(r)

def fit_law(p, u, y, segs):
    best = None
    for k0 in [(.05, .05, .05, .05), (.02, .1, .1, .05), (.1, .02, .05, .2)]:
        for x10 in (4., 12.):
            par0 = np.concatenate([np.log(k0), np.full(len(segs), np.log(x10))])
            try: sol = least_squares(resid, par0, args=(p, u, y, segs), bounds=(-12, 6), max_nfev=60)
            except Exception: continue
            if best is None or sol.cost < best.cost: best = sol
    return np.exp(best.x[:4])

def eval_law(theta, p, u, y, segs):
    """simulation RMSE of a fixed law on given segments; the unmeasured x1(0) of each segment is a nuisance parameter."""
    se, n = 0., 0
    for a, b in segs:
        f = lambda lx: np.sum((simulate(theta, p, u[a:b], np.exp(lx), y[a]) - y[a:b]) ** 2)
        r = minimize_scalar(f, bounds=(-3, 6), method='bounded', options={'xatol': 1e-2}); se += r.fun; n += b - a
    return float(np.sqrt(se / n))

def rows(y, u):
    t = np.arange(max(NY, NU), len(y))
    X = np.hstack([np.stack([y[t - k] for k in range(1, NY + 1)], 1), np.stack([u[t - k] for k in range(NU)], 1)]); return X, y[t], t

POW = [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(NY + NU), d)]
def poly(X): return np.hstack([np.ones((len(X), 1))] + [np.prod(X[:, list(c)], 1, keepdims=True) for c in POW])
class Ridge:
    def fit(self, X, y, a): F = poly(X); self.B = np.linalg.solve(F.T @ F + a * np.eye(F.shape[1]), F.T @ y); return self
    def predict(self, X): return poly(X) @ self.B
class MLP:
    def __init__(self, seed): self.seed = seed
    def fit(self, X, y, a):
        self.m = MLPRegressor(hidden_layer_sizes=(32, 32), activation='tanh', alpha=a, max_iter=500, early_stopping=True, random_state=self.seed, n_iter_no_change=25).fit(X, y); return self
    def predict(self, X): return self.m.predict(X)

def rollout_err(model, y, u, h, origins, sd):
    hist = np.stack([y[origins - k] for k in range(1, NY + 1)], 1); err = np.zeros(len(origins))
    for k in range(h):
        t = origins + k; X = np.hstack([hist, np.stack([u[t - j] for j in range(NU)], 1)])
        p = np.clip(model.predict(X), -5, 25); err += (p - y[t]) ** 2; hist = np.hstack([p[:, None], hist[:, :-1]])
    return float(np.sqrt(np.mean(err / h)) / sd)

def multisine(rng, n, mean, sd):
    """random-phase multisine on the benchmark band (0-0.0144 Hz at Ts=4 s: 59 harmonics of 1/4096 s), 1/f-tilted amplitudes"""
    k = np.arange(1, 60); amp = 1. / np.sqrt(k); ph = rng.uniform(0, 2 * np.pi, len(k)); t = np.arange(n) * TS
    u = sum(a * np.cos(2 * np.pi * kk * t / (n * TS) + p) for a, kk, p in zip(amp, k, ph)); return mean + sd * (u - u.mean()) / u.std()

def main(csv, out):
    d = pd.read_csv(csv); rec = {'est': (d.uEst.values, d.yEst.values), 'val': (d.uVal.values, d.yVal.values)}
    laws, res = [], []
    for tr, te in [('est', 'val'), ('val', 'est')]:
        (ut, yt), (ue, ye) = rec[tr], rec[te]; sd_te = ye.std()
        segs_tr, segs_te = segments(yt), segments(ye)
        # ---- certify laws on low-level data only
        fits, dfs = {}, {}
        for name, p in LAWS.items():
            th = fit_law(p, ut, yt, segs_tr); fits[name] = th
            dfs[name] = eval_law(th, p, ue, ye, segs_te) / ye[np.concatenate([np.arange(a, b) for a, b in segs_te])].std()
        o = sorted(dfs, key=dfs.get); acc = o[0] if dfs[o[0]] <= .5 and dfs[o[0]] <= .5 * dfs[o[1]] else None
        # twin extrapolation error on the high-level (law-valid) and overflow segments of the held-out record
        hi = [(a, b) for a, b in _runs(ye, LC, OV)]; ov = [(a, b) for a, b in _runs(ye, OV, 99)]
        for name, p in LAWS.items():
            laws.append(dict(train=tr, law=name, theta=list(np.round(fits[name], 4)), defect_low=dfs[name], accepted=int(name == acc),
                             rmse_high=eval_law(fits[name], p, ue, ye, hi) / sd_te if hi else np.nan,
                             rmse_overflow=eval_law(fits[name], p, ue, ye, ov) / sd_te if ov else np.nan))
        print(tr, '->', te, {k: round(v, 3) for k, v in dfs.items()}, 'accepted', acc, flush=True)
        # ---- learners
        Xl, tl, idx = rows(yt, ut); low = yt[idx] <= LC; Xl, tl = Xl[low], tl[low]
        Xv, tv, _ = rows(ye, ue)                                   # tuning: held-out record, low rows only
        lowv = ye[np.arange(max(NY, NU), len(ye))] <= LC; Xv, tv = Xv[lowv], tv[lowv]
        for seed in range(6):
            rng = np.random.default_rng(1000 * (tr == 'val') + seed)
            noise = float(np.std(resid(np.log(np.r_[fits['sqrt-sqrt'], np.ones(len(segs_tr)) * 8]), LAWS['sqrt-sqrt'], ut, yt, segs_tr)) * .3)
            syn = {}
            for name, p in LAWS.items():
                Xs, ts = [], []
                for r in range(4):
                    us = multisine(rng, 1024, rng.uniform(3.0, 4.2), .8); ys = simulate(fits[name], p, us, 8., 5.)[100:]
                    ys = ys + rng.normal(0, noise, len(ys)); X, t, _ = rows(ys, us[100:]); Xs.append(X); ts.append(t)
                syn[name] = (np.vstack(Xs), np.concatenate(ts))
            arms = {'base': None, 'duplicate': (Xl, tl), 'twin sqrt-sqrt': syn['sqrt-sqrt'], 'twin lin-lin': syn['lin-lin'],
                    'twin sqrt-lin': syn['sqrt-lin'], 'twin lin-sqrt': syn['lin-sqrt']}
            arms['gated'] = syn[acc] if acc else arms['duplicate']
            Xr, tr_, _ = rows(yt, ut); hi_rows = (yt[np.arange(max(NY, NU), len(yt))] > LC) & (yt[np.arange(max(NY, NU), len(yt))] < OV)
            arms['real high-level'] = (Xr[hi_rows], tr_[hi_rows])
            for lname, make in {'poly NARX': lambda: Ridge(), 'MLP': lambda: MLP(seed)}.items():
                for arm, extra in arms.items():
                    XA, tA = (Xl, tl) if extra is None else (np.vstack([Xl, extra[0]]), np.concatenate([tl, extra[1]]))
                    m = min((make().fit(XA, tA, a) for a in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - tv) ** 2))
                    for region, (lo_, hi_) in {'low': (0, LC), 'high': (LC, OV), 'overflow': (OV, 99)}.items():
                        org = np.arange(max(NY, NU), len(ye) - max(H)); org = org[(ye[org] > lo_) & (ye[org] <= hi_)] if region != 'high' else org[(ye[org] > LC) & (ye[org] < OV)]
                        if region == 'low': org = org[ye[org] <= LC]
                        row = dict(train=tr, seed=seed, learner=lname, arm=arm, accepted=acc or 'abstain', region=region, n=len(org))
                        for h in H: row[f'nrmse_h{h}'] = rollout_err(m, ye, ue, h, org, sd_te) if len(org) else np.nan
                        res.append(row)
        print('  done', tr, flush=True)
    pd.DataFrame(laws).to_csv(Path(out) / 'tanks_laws.csv', index=False); pd.DataFrame(res).to_csv(Path(out) / 'tanks_results.csv', index=False)

def _runs(y, lo, hi, minlen=8):
    m = (y > lo) & (y < hi); segs, s = [], None
    for i, v in enumerate(m):
        if v and s is None: s = i
        if (not v or i == len(m) - 1) and s is not None:
            e = i + 1 if v else i
            if e - s >= minlen: segs.append((s, e))
            s = None
    return segs

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
