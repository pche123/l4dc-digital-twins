"""Real-data approximate-symmetry case on the Wiener-Hammerstein benchmark (Schoukens, Suykens and Ljung, SYSID 2009).
The system is linear - diode - linear, and the 1N4148 diode makes the static nonlinearity asymmetric (the output is skewed),
so (-y,-u) is at best an approximate symmetry and (-y,+u), (+y,-u) are not. The gate should accept at most an approximate one, and
the value of mirrored data should track how well it holds. Training sees only rows with y_{t-1}>0; the estimation part is split into 5 blocks (replicates);
the test part is the benchmark's held-out segment. Series are decimated by 8 (51.2 kHz -> 6.4 kHz).

usage: python code/wh_exp.py <WienerHammerBenchmark.csv> data     writes data/wh_results.csv, data/wh_gate.csv
"""
import sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import decimate
warnings.filterwarnings('ignore')

NY, NU = 4, 4                      # y lags 1..4, u lags 0..3
DEC = 8; NEST = 100000 // DEC; NBLK = 5
ALPHAS = [1e-6, 1e-4, 1e-2, 1, 1e2]
H = (1, 10, 50)
BL = NEST // NBLK
REAL = [(i * BL, (i + 1) * BL) for i in range(NBLK)]
ARROW = (NEST, NEST + 8000)

def rows(y, u):
    t = np.arange(max(NY, NU), len(y))
    Y = np.stack([y[t - k] for k in range(1, NY + 1)], 1); U = np.stack([u[t - k] for k in range(NU)], 1)
    return np.hstack([Y, U]), y[t]

POW = [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(NY + NU), d)]
def poly(X): return np.hstack([np.ones((len(X), 1))] + [np.prod(X[:, list(c)], 1, keepdims=True) for c in POW])
def lin(X): return np.hstack([np.ones((len(X), 1)), X])

class Ridge:
    def __init__(self, feat): self.feat = feat
    def fit(self, X, y, a):
        F = self.feat(X); self.B = np.linalg.solve(F.T @ F + a * np.eye(F.shape[1]), F.T @ y); return self
    def predict(self, X): return self.feat(X) @ self.B
LEARNERS = {'linear ARX': lambda: Ridge(lin), 'cubic NARX': lambda: Ridge(poly)}

def flip(X, y, sy, su):
    Z = X.copy(); Z[:, :NY] *= sy; Z[:, NY:] *= su; return Z, sy * y
CANDS = {'exact (-y,-u)': (-1, -1), 'false (-y,+u)': (-1, 1), 'input-only (+y,-u)': (1, -1)}

def recursive(model, y, u, h):
    o = np.arange(NY + NU, len(y) - h - 1, 50); hist = np.stack([y[o - k] for k in range(1, NY + 1)], 1); err = np.zeros(len(o))
    for k in range(h):
        t = o + k; X = np.hstack([hist, np.stack([u[t - j] for j in range(NU)], 1)])
        p = np.clip(model.predict(X), -20, 20); err += (p - y[t]) ** 2
        hist = np.hstack([p[:, None], hist[:, :-1]])
    return float(np.sqrt(np.mean(err / h)) / np.std(y))

def main(csv, out):
    d = pd.read_csv(csv).iloc[:, :2].dropna(); u_all = decimate(d.iloc[:, 0].values, DEC, zero_phase=True); y_all = decimate(d.iloc[:, 1].values, DEC, zero_phase=True)
    u_all = u_all - u_all.mean(); y_all = y_all - y_all.mean()
    res, gate = [], []
    yt, ut = y_all[ARROW[0]:ARROW[1]], u_all[ARROW[0]:ARROW[1]]
    for r, (a, b) in enumerate(REAL):
        y, u = y_all[a:b], u_all[a:b]
        sy, su = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(u ** 2))          # scale only: keeps the symmetry exact
        X, t = rows(y / sy, u / su); pos = X[:, 0] > 0
        Xp, tp, Xn, tn = X[pos], t[pos], X[~pos], t[~pos]
        rng = np.random.default_rng(r); keep = rng.choice(len(Xn), size=min(len(Xn), len(Xp)), replace=False)
        a2, b2 = REAL[(r + 1) % len(REAL)]; Xv, tv = rows(y_all[a2:b2] / sy, u_all[a2:b2] / su)
        pv = Xv[:, 0] > 0; Xv, tv = Xv[pv], tv[pv]                     # tuning sees the one-sided region only
        ys, us = yt / sy, ut / su
        # symbolic gate on the one-sided rows
        lib = min((Ridge(poly).fit(Xp, tp, al) for al in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - tv) ** 2))
        scale = np.sqrt(np.mean(lib.predict(Xp) ** 2)); dfs = {}
        for name, (fy, fu) in CANDS.items():
            Z, _ = flip(Xp, tp, fy, fu); dfs[name] = float(np.sqrt(np.mean((lib.predict(Z) - fy * lib.predict(Xp)) ** 2)) / scale)
        o = sorted(dfs, key=dfs.get); acc = o[0] if dfs[o[0]] <= .5 and dfs[o[0]] <= .5 * dfs[o[1]] else None
        # mirror-region error of the library (vs real negative rows): the quantity Proposition 8 says the defect tracks
        e_mirror = float(np.sqrt(np.mean((lib.predict(Xn) - tn) ** 2)) / scale); e_obs = float(np.sqrt(np.mean((lib.predict(Xp) - tp) ** 2)) / scale)
        for name in CANDS: gate.append(dict(realisation=r, candidate=name, defect=dfs[name], accepted=int(name == acc), e_obs=e_obs, e_mirror=e_mirror))
        arms = {'base': None, 'duplicate': (Xp, tp), 'exact': flip(Xp, tp, -1, -1), 'false': flip(Xp, tp, -1, 1),
                'real negative': (Xn[keep], tn[keep])}
        arms['gated'] = arms['exact'] if acc == 'exact (-y,-u)' else (flip(Xp, tp, *CANDS[acc]) if acc else arms['duplicate'])
        for lname, make in LEARNERS.items():
            for arm, extra in arms.items():
                XA, tA = (Xp, tp) if extra is None else (np.vstack([Xp, extra[0]]), np.concatenate([tp, extra[1]]))
                m = min((make().fit(XA, tA, al) for al in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - tv) ** 2))
                row = dict(realisation=r, learner=lname, arm=arm, accepted=acc or 'abstain')
                for h in H: row[f'nrmse_h{h}'] = recursive(m, ys, us, h)
                res.append(row)
        print('realisation', r, 'accepted', acc, {k: round(v, 3) for k, v in dfs.items()}, 'e_obs %.3f e_mirror %.3f' % (e_obs, e_mirror), flush=True)
    pd.DataFrame(res).to_csv(Path(out) / 'wh_results.csv', index=False); pd.DataFrame(gate).to_csv(Path(out) / 'wh_gate.csv', index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
