"""Symbolic symmetry gate and verified augmentation on fresh seeds (Lorenz, Duffing, Lotka-Volterra control).

usage: python verify_exp.py <phase3> <out_dir> <mode>   mode in {main, lownoise, tilt, control}

For every cell (system, length, noise, tilt, seed) we
  1. fit a library to the one-sided training data and score each candidate axis reflection S by its
     equivariance defect d(S) = RMS(f(Sx) - S f(x)) / RMS(f(x)) over training states (no test data);
  2. record forecasting loss of the base learner and of learners augmented with the duplicate, the true
     symmetry, the false reflection, a genuine opposite-side trajectory, and any candidate passing the ratio test.
The paper's gate (accept the best candidate iff d1 <= 0.5 and d1/d2 <= 0.5) is applied in gate_summary.py.
"""
from __future__ import annotations
import sys, itertools, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

PH3, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
RULE = 0.5   # clear-winner ratio, fixed on the Lorenz study

def _load(name, argv):
    sys.argv = argv
    sp = importlib.util.spec_from_file_location(name, HERE / f'{name}.py'); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m
mc = _load('model_class_exp', ['x', str(PH3), str(OUT / '_t1')])
du = _load('duffing_exp', ['x', str(PH3), str(OUT / '_t2')])
from phase3_common import build_lorenz_split, fit_standardizer, apply_standardizer

# ---------------- systems ----------------
def flips(dim):
    out = {}
    for s in itertools.product([1., -1.], repeat=dim):
        if all(v > 0 for v in s): continue
        out['(%s)' % ','.join(('-' if v < 0 else '+') + c for v, c in zip(s, 'xyz'[:dim]))] = np.diag(s)
    return out

LOR_TRUE, LOR_FALSE = '(-x,-y,+z)', '(-x,+y,+z)'
DUF_TRUE, DUF_FALSE = '(-x,-y)', '(-x,+y)'      # (x, xdot): exact symmetry flips both; false flips position only

def duf_split(seed, length, noise, eps):
    """Double-well Duffing x'' = x - x^3 + eps (eps != 0 tilts the potential and breaks the symmetry)."""
    DT = du.DT
    f = lambda s: np.array([s[1], s[0] - s[0] ** 3 + eps])
    def integ(s0, n):
        s = np.array(s0, float); out = []
        for _ in range(n):
            k1 = f(s); k2 = f(s + .5 * DT * k1); k3 = f(s + .5 * DT * k2); k4 = f(s + DT * k3)
            s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(s.copy())
        return np.array(out).T
    def win(rng, sign):
        while True:
            x0 = sign * rng.uniform(0.35, 1.38); v0 = rng.normal(0, 0.15)
            E = v0 ** 2 / 2 - x0 ** 2 / 2 + x0 ** 4 / 4 - eps * x0
            if -0.24 < E < -0.03: break
        burn = int(rng.integers(0, 60)); tr = integ([x0, v0], burn + length)[:, burn:]
        return tr + rng.normal(0, noise, tr.shape)
    r = lambda k: np.random.default_rng(seed * 1000 + k)
    return {'train': [win(r(1), +1), win(r(2), +1)],
            'tune': [win(r(10 + i), s) for i in range(5) for s in (+1, -1)],
            'headroom': [win(r(30 + i), s) for i in range(5) for s in (+1, -1)],
            'test': [win(r(50 + i), +1) for i in range(10)] + [win(r(70 + i), -1) for i in range(10)],
            'extra_real': win(r(91), -1), 'oracle': win(r(92), -1)}

def defects(feat, B, mu, sd, dt, X, cands):
    """Equivariance defects of learned field f(x) = feat((x-mu)/sd) @ B * sd / dt over raw states X (n,d)."""
    f = lambda Z: (feat((Z - mu) / sd) @ B) * sd / dt
    F0 = f(X); scale = np.sqrt(np.mean(F0 ** 2))
    return {n: float(np.sqrt(np.mean((f(X @ S.T) - F0 @ S.T) ** 2)) / scale) for n, S in cands.items()}

LAM = 0.1   # STLSQ threshold relative to each equation's largest (column-normalised) coefficient; fixed a priori

def sparse_field(series, dt, feat, lam=LAM, ridge=1e-6, iters=6):
    """Sequentially thresholded least squares (SINDy-style) in raw coordinates; returns f(X) for X of shape (n,d)."""
    Phi = np.vstack([feat(w[:, :-1].T) for w in series]); Y = np.vstack([(w[:, 1:] - w[:, :-1]).T / dt for w in series])
    s = np.sqrt(np.mean(Phi ** 2, 0)) + 1e-12; P = Phi / s
    C = np.zeros((P.shape[1], Y.shape[1]))
    for k in range(Y.shape[1]):
        act = np.ones(P.shape[1], bool)
        for _ in range(iters):
            c = np.zeros(P.shape[1]); A = P[:, act]
            c[act] = np.linalg.solve(A.T @ A + ridge * len(A) * np.eye(A.shape[1]), A.T @ Y[:, k])
            new = np.abs(c) >= lam * np.abs(c).max()
            if (new == act).all(): break
            act = new
        C[:, k] = c / s
    return lambda X: feat(X) @ C

def callable_defects(f, X, cands):
    F0 = f(X); scale = np.sqrt(np.mean(F0 ** 2))
    return {n: float(np.sqrt(np.mean((f(X @ S.T) - F0 @ S.T) ** 2)) / scale) for n, S in cands.items()}

def clear_winner(d, rule=RULE):
    o = sorted(d.items(), key=lambda kv: kv[1])
    return o[0][0] if o[0][1] <= rule * o[1][1] else None

# ---------------- one cell ----------------
def run_cell(system, seed, length, noise, eps=0.0):
    if system == 'lorenz':
        import phase3_common as pc
        pc.lorenz_rhs = (lambda v, _e=eps: np.array([10.0 * (v[1] - v[0]) + _e, v[0] * (28.0 - v[2]) - v[1], v[0] * v[1] - 8.0 / 3.0 * v[2]]))
        raw = build_lorenz_split(seed, length, 5, 5, 10, noise, 0.95)
        mod, true, false, dim, dt = mc, LOR_TRUE, LOR_FALSE, 3, 0.01
        lib = {'linear': mc.Learner('linear'), 'quadratic': mc.Learner('quadratic')}; libname = 'quadratic'
        feat = mc.quad_features
    else:
        raw = duf_split(seed, length, noise, eps)
        mod, true, false, dim, dt = du, DUF_TRUE, DUF_FALSE, 2, du.DT
        lib = {'linear': du.Learner('linear'), 'cubic': du.Learner('cubic')}; libname = 'cubic'
        feat = du.cubic_features
    cands = flips(dim)
    mu, sd = fit_standardizer(raw['train'])
    st = lambda w: (w - mu) / sd
    z = {k: ([st(w) for w in v] if isinstance(v, list) else st(v)) for k, v in raw.items()}
    L = lib[libname]; a = mod.tune(L, z['train'], z['tune']); B = L.fit(z['train'], a)
    X = np.concatenate(raw['train'], axis=1).T
    dd = defects(feat, B, mu[:, 0], sd[:, 0], dt, X, cands)
    ds = callable_defects(sparse_field(raw['train'], dt, feat), X, cands)
    drows, accs = [], set()
    for ver, d in (('dense', dd), ('sparse', ds)):
        acc = clear_winner(d); accs |= {acc} if acc else set()
        drows += [dict(system=system, length=length, noise=noise, eps=eps, seed=seed, verifier=ver, transform=n, defect=v,
                       accepted=int(n == acc), is_true=int(n == true)) for n, v in d.items()]
    arm_names = {'duplicate', 'true', 'false', 'opposite', true, false} | accs
    arms = {'base': None, 'duplicate': z['train'][0].copy(), 'opposite': z['extra_real']}
    for n in arm_names:
        if n in cands: arms['T' + n] = st(cands[n] @ raw['train'][0])
    rows = []
    pos, neg = z['test'][:10], z['test'][10:]
    for lname, Lr in lib.items():
        for arm, cand in arms.items():
            tr = z['train'] if cand is None else z['train'] + [cand]
            al = mod.tune(Lr, tr, z['tune']); Bm = Lr.fit(tr, al)
            row = dict(system=system, length=length, noise=noise, eps=eps, seed=seed, learner=lname, arm=arm,
                       headroom_loss=mod.evaluate(Lr, Bm, z['headroom'])[0])
            for tag, S in [('balanced', z['test']), ('positive', pos), ('negative', neg)]:
                row[tag + '_loss'] = mod.evaluate(Lr, Bm, S)[0]
            rows.append(row)
    return rows, drows

# ---------------- no-symmetry control: Lotka-Volterra ----------------
def lv_cell(seed, length=40, noise=0.01):
    DT = 0.1; f = lambda s: np.array([s[0] - s[0] * s[1], s[0] * s[1] - s[1]])
    def integ(s0, n):
        s = np.array(s0, float); out = []
        for _ in range(n):
            k1 = f(s); k2 = f(s + .5 * DT * k1); k3 = f(s + .5 * DT * k2); k4 = f(s + DT * k3)
            s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(s.copy())
        return np.array(out).T
    rng = np.random.default_rng(seed + 777)
    tr = []
    for _ in range(2):
        x0 = rng.uniform(1.2, 2.2); w = integ([x0, 1.0], int(rng.integers(0, 40)) + length)[:, -length:]
        tr.append(w + rng.normal(0, noise, w.shape))
    mu, sd = fit_standardizer(tr); cands = flips(2)
    def feat(Z):
        a, b = Z[:, 0:1], Z[:, 1:2]; return np.hstack([np.ones_like(a), a, b, a * a, a * b, b * b])
    Xs, Ys = [], []
    for w in tr:
        y = (w - mu) / sd
        for t in range(1, y.shape[1]): Xs.append(feat(y[:, t - 1][None])[0]); Ys.append(y[:, t] - y[:, t - 1])
    Xs, Ys = np.array(Xs), np.array(Ys); B = np.linalg.solve(Xs.T @ Xs + 1e-4 * np.eye(Xs.shape[1]), Xs.T @ Ys)
    X = np.concatenate(tr, 1).T
    out = []
    for ver, d in (('dense', defects(feat, B, mu[:, 0], sd[:, 0], DT, X, cands)),
                   ('sparse', callable_defects(sparse_field(tr, DT, feat), X, cands))):
        acc = clear_winner(d)
        out += [dict(system='lv', length=length, noise=noise, eps=0.0, seed=seed, verifier=ver, transform=n, defect=v,
                     accepted=int(n == acc), is_true=0) for n, v in d.items()]
    return out

if __name__ == '__main__':
    from joblib import Parallel, delayed
    jobs = []
    if MODE == 'main':      # fresh seeds, paper defaults
        jobs += [('lorenz', s, 36, 0.015, 0.0) for s in range(1000, 1030)]
        jobs += [('duffing', s, 40, 0.01, 0.0) for s in range(1000, 1030)]
    elif MODE == 'lownoise':  # Duffing gate as the library becomes identifiable
        jobs += [('duffing', s, 40, nz, 0.0) for nz in (0.0, 0.0025, 0.005) for s in range(1000, 1030)]
    elif MODE == 'tilt':    # graded symmetry breaking: constant forcing eps added to dx/dt of Lorenz
        for eps in (2.0, 5.0, 10.0):
            jobs += [('lorenz', s, 36, 0.015, eps) for s in range(3000, 3020)]
    if MODE == 'control':
        rows = [r for s in range(1000, 1030) for r in lv_cell(s)]
        pd.DataFrame(rows).to_csv(OUT / 'control_lv_defects.csv', index=False)
        print(pd.DataFrame(rows).groupby('seed').accepted.sum().sum(), 'accepts / 30'); sys.exit()
    outs = Parallel(n_jobs=2, verbose=3)(delayed(run_cell)(*j) for j in jobs)
    name = 'duf_lownoise' if MODE == 'lownoise' else f'verify_{MODE}'
    pd.DataFrame([r for o in outs for r in o[0]]).to_csv(OUT / f'{name}_arms.csv', index=False)
    pd.DataFrame([r for o in outs for r in o[1]]).to_csv(OUT / f'{name}_defects.csv', index=False)
    print('done', MODE, len(jobs))
