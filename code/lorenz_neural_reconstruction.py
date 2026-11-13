"""Independent one-step neural reconstruction on Lorenz (Supplement D and the verification-value table).

Protocol: 20 seeds (12300-12319); eight positive-wing training windows from one simulation (observation noise
0.015), eight noise-free test windows per wing from two further independent simulations; lag four; an MLP with two
tanh layers of 64 units trained by full-batch Adam for 400 epochs (learning rate 0.003, weight decay 1e-4) on
standardized one-step targets. Arms add the same number of rows: duplicate (training windows repeated), true_sym
(mirrored under (-x,-y,z)), false_sym (mirrored under (-x,y,z)) and selective (the gate's accepted candidate, else
duplicate). The gate fits a quadratic ridge field to the training windows and accepts the best axis sign flip iff
its defect is <= 0.5 and at most half the runner-up's. All arms of a seed share one network initialization.
Outputs: data/lorenz_documented_protocol_results.csv and data/lorenz_documented_protocol_summary.csv.

usage: python code/lorenz_neural_reconstruction.py harness data
"""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(sys.argv[1]) / 'code'))
from phase3_common import extract_wing_windows, fit_standardizer, lorenz_reservoir

SEEDS = range(12300, 12320)
LEN, LAG, NWIN, NOISE, DT = 36, 4, 8, 0.015, 0.01
HIDDEN, EPOCHS, LR, WD = 64, 400, 3e-3, 1e-4
TRUE_S, FALSE_S = np.diag([-1., -1., 1.]), np.diag([-1., 1., 1.])
CANDS = {s: np.diag(s) for s in itertools.product([1., -1.], repeat=3) if not all(v > 0 for v in s)}


def rows(series):
    X, Y = [], []
    for w in series:
        for t in range(LAG, w.shape[1]):
            X.append(w[:, t - LAG:t].T.reshape(-1)); Y.append(w[:, t])
    return np.array(X), np.array(Y)


def train_mlp(X, Y, seed):
    """Two tanh hidden layers; full-batch Adam with L2 weight decay added to the gradient (PyTorch Adam convention)."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], HIDDEN, HIDDEN, Y.shape[1]]
    P = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        k = 1 / np.sqrt(a); P += [rng.uniform(-k, k, (a, b)), rng.uniform(-k, k, b)]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(X)
    for it in range(1, EPOCHS + 1):
        h1 = np.tanh(X @ P[0] + P[1]); h2 = np.tanh(h1 @ P[2] + P[3]); out = h2 @ P[4] + P[5]
        g = 2 * (out - Y) / (n * Y.shape[1])
        gW3, gb3 = h2.T @ g, g.sum(0); d2 = (g @ P[4].T) * (1 - h2 ** 2)
        gW2, gb2 = h1.T @ d2, d2.sum(0); d1 = (d2 @ P[2].T) * (1 - h1 ** 2)
        gW1, gb1 = X.T @ d1, d1.sum(0)
        G = [gW1, gb1, gW2, gb2, gW3, gb3]
        for i in range(6):
            gi = G[i] + WD * P[i]
            m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi ** 2
            P[i] = P[i] - LR * (m[i] / (1 - b1 ** it)) / (np.sqrt(v[i] / (1 - b2 ** it)) + eps)
    return lambda Z: np.tanh(np.tanh(Z @ P[0] + P[1]) @ P[2] + P[3]) @ P[4] + P[5]


def quad(Z):
    cols = [np.ones((len(Z), 1)), Z] + [(Z[:, i] * Z[:, j])[:, None] for i in range(3) for j in range(i, 3)]
    return np.hstack(cols)


def gate(train_raw, mu, sd):
    """Quadratic ridge field on the training windows; equivariance defect of every axis sign flip."""
    Z = [(w - mu) / sd for w in train_raw]
    F = np.vstack([quad(z[:, :-1].T) for z in Z]); dY = np.vstack([(z[:, 1:] - z[:, :-1]).T for z in Z])
    B = np.linalg.solve(F.T @ F + 1e-6 * np.eye(F.shape[1]), F.T @ dY)
    f = lambda X: (quad((X - mu[:, 0]) / sd[:, 0]) @ B) * sd[:, 0] / DT
    X = np.concatenate(train_raw, axis=1).T; F0 = f(X); scale = np.sqrt(np.mean(F0 ** 2))
    d = {s: float(np.sqrt(np.mean((f(X @ S.T) - F0 @ S.T) ** 2)) / scale) for s, S in CANDS.items()}
    o = sorted(d, key=d.get); d1, d2 = d[o[0]], d[o[1]]
    accepted = o[0] if (d1 <= 0.5 and d1 <= 0.5 * d2) else None
    return accepted, o[0], d1, d1 / d2, d[(-1., -1., 1.)]


def one_seed(seed):
    tr = extract_wing_windows(lorenz_reservoir(100000 + seed * 31, obs_noise=NOISE), LEN, +1, NWIN, min_separation=LEN)
    tp = extract_wing_windows(lorenz_reservoir(400000 + seed * 31), LEN, +1, NWIN, min_separation=LEN)
    tn = extract_wing_windows(lorenz_reservoir(450000 + seed * 31), LEN, -1, NWIN, min_separation=LEN)
    mu, sd = fit_standardizer(tr); st = lambda w: (w - mu) / sd
    acc, best, d1, ratio, exact_def = gate(tr, mu, sd)
    blocks = {'duplicate': [st(w) for w in tr], 'true_sym': [st(TRUE_S @ w) for w in tr], 'false_sym': [st(FALSE_S @ w) for w in tr]}
    blocks['selective'] = [st(np.diag(acc) @ w) for w in tr] if acc is not None else blocks['duplicate']
    base = [st(w) for w in tr]; Xp, Yp = rows([st(w) for w in tp]); Xn, Yn = rows([st(w) for w in tn])
    out = []
    for arm, extra in blocks.items():
        X, Y = rows(base + extra); net = train_mlp(X, Y, seed)
        mp, mn = float(np.mean((net(Xp) - Yp) ** 2)), float(np.mean((net(Xn) - Yn) ** 2))
        out.append(dict(seed=seed, arm=arm, mse_pos=mp, mse_neg=mn, mse_bal=(mp + mn) / 2, gate_accept=acc is not None,
                        best_defect=d1, ratio=ratio, best_sign=str([int(v) for v in best]), exact_defect=exact_def))
    return out


def main(out):
    from joblib import Parallel, delayed
    R = pd.DataFrame([r for rr in Parallel(n_jobs=2)(delayed(one_seed)(s) for s in SEEDS) for r in rr])
    p = R.pivot(index='seed', columns='arm', values='mse_bal'); rng = np.random.default_rng(20261109); S = []
    for arm in ['true_sym', 'false_sym', 'selective']:
        g = (p['duplicate'] - p[arm]).to_numpy(); bm = g[rng.integers(0, len(g), (10000, len(g)))].mean(1)
        S.append(dict(arm=arm, n=len(g), mean_gain=g.mean(), median_gain=float(np.median(g)), ci_lo=np.quantile(bm, .025),
                      ci_hi=np.quantile(bm, .975), positive=int((g > 0).sum())))
    R.to_csv(Path(out) / 'lorenz_documented_protocol_results.csv', index=False)
    pd.DataFrame(S).to_csv(Path(out) / 'lorenz_documented_protocol_summary.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[2])
