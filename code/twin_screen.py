"""Digital twin with a symbolic screen on candidate degradation laws (see data/twin_screen_protocol.md).

usage: python code/twin_screen.py data          writes data/twin_screen_arms.csv and data/twin_screen_laws.csv
Two screens are recorded: a black-box one (the ridge learner's degradation signature) and a structured one
(the twin's own equations fitted per wear level). Both use the symmetry gate's thresholds.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

DT, T = 0.02, 300
ALPHAS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
PLANT = dict(m1=1.02, m2=0.68, k1=82., kc=58., c1=.75, cc=.40, k2=lambda w: 103 * (1 - .45 * w), c2=lambda w: .55 * (1 + 1.2 * w))
NOM = dict(m1=1.00, m2=0.70, k1=80., kc=60., c1=.80, cc=.35)
LAWS = {'none': (lambda w: 100., lambda w: .60),
        'damping-only': (lambda w: 100., lambda w: .60 * (1 + 1.2 * w)),
        'stiffness-only': (lambda w: 100 * (1 - .45 * w), lambda w: .60),
        'correct': (lambda w: 100 * (1 - .45 * w), lambda w: .60 * (1 + 1.2 * w)),
        'overstated': (lambda w: 100 * (1 - .9 * w), lambda w: .60 * (1 + 2.4 * w))}
RULE_ABS, RULE_RATIO = 0.5, 0.5
LP = butter(4, 5.0 / (0.5 / DT))

def gate(defects, names):
    d = {k: defects[k] for k in names}; o = sorted(d, key=d.get)
    return o[0] if (d[o[0]] <= RULE_ABS and d[o[0]] <= RULE_RATIO * d[o[1]]) else None

def structured_fit(train):
    """Per-wear-level k2/m2 and c2/m2 from filtered observations (the twin's own equations, unknown parameters)."""
    rows, ys, J = [], [], len(train)
    for j, d in enumerate(train):
        X = filtfilt(*LP, d['X'], axis=0); a2 = (X[2:, 3] - X[:-2, 3]) / (2 * DT); q1, q2, v1, v2 = X[1:-1].T
        R = np.zeros((len(a2), 2 * J + 2)); R[:, 2 * j], R[:, 2 * j + 1], R[:, -2], R[:, -1] = q2, v2, q2 - q1, v2 - v1
        rows.append(R[20:-20]); ys.append(a2[20:-20])
    R, y = np.vstack(rows), np.concatenate(ys)
    beta = np.linalg.lstsq(R, y, rcond=None)[0]; s2 = np.sum((y - R @ beta) ** 2) / (len(y) - R.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(R.T @ R)))
    kap, gam = -beta[0:2 * J:2], -beta[1:2 * J:2]
    rho = np.concatenate([kap[1:] / kap[0] - 1, gam[1:] / gam[0] - 1])
    wts = 1 / np.concatenate([se[2:2 * J:2] / abs(kap[0]), se[3:2 * J:2] / abs(gam[0])]) ** 2
    return rho, wts
H = (1, 5, 20)

def params(kind, w):
    if kind == 'plant': p = dict(PLANT); p['k2'], p['c2'] = PLANT['k2'](w), PLANT['c2'](w); return p
    p = dict(NOM); p['k2'], p['c2'] = LAWS[kind][0](w), LAWS[kind][1](w); return p

def rhs(x, u, p):   # x (...,4) = q1, q2, v1, v2
    q1, q2, v1, v2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    a1 = (-p['k1'] * q1 - p['c1'] * v1 - p['kc'] * (q1 - q2) - p['cc'] * (v1 - v2) + u) / p['m1']
    a2 = (-p['k2'] * q2 - p['c2'] * v2 - p['kc'] * (q2 - q1) - p['cc'] * (v2 - v1)) / p['m2']
    return np.stack([v1, v2, a1, a2], -1)

def step(x, u, p):
    k1 = rhs(x, u, p); k2 = rhs(x + DT / 2 * k1, u, p); k3 = rhs(x + DT / 2 * k2, u, p); k4 = rhs(x + DT * k3, u, p)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def excitation(rng):
    t = np.arange(T) * DT; f = rng.uniform(.4, 3.5, 4); ph = rng.uniform(0, 2 * np.pi, 4)
    u = np.sin(2 * np.pi * f[:, None] * t + ph[:, None]).sum(0); u /= np.sqrt(np.mean(u ** 2))
    prbs = np.repeat(rng.choice([-1., 1.], T // 10 + 1), 10)[:T]
    return u + .2 * prbs

def simulate(kind, w, u, x0, rng=None, noise=1.0):
    p = params(kind, w); X = np.empty((T, 4)); x = x0.copy()
    for t in range(T): X[t] = x; x = step(x, u[t], p)
    if rng is not None: X = X + rng.normal(0, 1, X.shape) * np.array([.0025, .0025, .01, .01]) * noise
    return X

def traj(kind, w, rng, noise=1.0, observe=True):
    u = excitation(rng); x0 = rng.normal(0, .01, 4)
    return dict(X=simulate(kind, w, u, x0, rng if observe else None, noise), U=u, w=w)

def feats(X, U, w):
    W = np.full((len(X), 1), w)
    return np.hstack([X, U[:, None], W * X, W * U[:, None], W, np.ones((len(X), 1))])

class Learner:
    def __init__(self, train):
        F = np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in train])
        self.mu, self.sd = F[:, :-1].mean(0), F[:, :-1].std(0) + 1e-12
        S = np.vstack([d['X'] for d in train]); self.xsd = S.std(0)
    def z(self, F): return np.hstack([(F[:, :-1] - self.mu) / self.sd, F[:, -1:]])
    def fit(self, data, alpha):
        F = self.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in data])); Y = np.vstack([d['X'][1:] for d in data])
        return np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)
    def predict(self, B, X, U, w): return self.z(feats(X, U, w)) @ B
    def onestep(self, B, data): return float(np.mean([np.mean(((self.predict(B, d['X'][:-1], d['U'][:-1], d['w']) - d['X'][1:]) / self.xsd) ** 2) for d in data]))
    def rollout(self, B, x0, U, w):
        X = np.empty((len(U), 4)); x = x0.copy()
        for t in range(len(U)):
            X[t] = x; x = np.clip(self.predict(B, x[None], U[t:t + 1], w)[0], -1e3, 1e3)
        return X
    def nrmse(self, B, data, h):
        errs = []
        for d in data:
            o = np.arange(0, T - h - 1, 10); x = d['X'][o].copy(); e = np.zeros(len(o))
            for k in range(h):
                x = np.clip(self.predict(B, x, d['U'][o + k], d['w']), -1e3, 1e3)
                e += np.mean(((x - d['X'][o + k + 1]) / self.xsd) ** 2, 1)
            errs.append(np.sqrt(e / h))
        return float(np.mean(np.concatenate(errs)))

def one_seed(seed, wmax=0.15, noise=1.0):
    rng = np.random.default_rng(seed)
    wtr = np.linspace(0, wmax, 4)
    train = [traj('plant', w, rng, noise) for w in wtr]
    tune = [traj('plant', w, rng, noise) for w in (wmax / 6, 5 * wmax / 6)]
    test = [traj('plant', w, rng, noise) for w in (.55, .70, .85) for _ in range(3)]
    L = Learner(train)
    alpha = min(ALPHAS, key=lambda a: L.onestep(L.fit(train, a), tune)); B0 = L.fit(train, alpha)
    # symbolic screen: does each law predict how the dynamics changed across the observed wear range?
    X = np.vstack([d['X'][:-1] for d in train]); U = np.concatenate([d['U'][:-1] for d in train])
    dh = (L.predict(B0, X, U, wmax) - L.predict(B0, X, U, 0.)) / L.xsd
    dtrue = (step(X, U, params('plant', wmax)) - step(X, U, params('plant', 0.))) / L.xsd
    rms = lambda A: float(np.sqrt(np.mean(A ** 2)))
    lawrows, defects = [], {}
    for name in LAWS:
        dA = (step(X, U, params(name, wmax)) - step(X, U, params(name, 0.))) / L.xsd
        defects[name] = rms(dh - dA) / rms(dh)
        lawrows.append(dict(seed=seed, wmax=wmax, noise=noise, law=name, defect=defects[name],
                            dist_to_true=rms(dtrue - dA) / rms(dh), fit_error=rms(dh - dtrue) / rms(dh)))
    rho, wts = structured_fit(train); sdef = {}
    for r in lawrows:
        kf, cf = LAWS[r['law']]
        rhoA = np.concatenate([[kf(w) / kf(0) - 1 for w in wtr[1:]], [cf(w) / cf(0) - 1 for w in wtr[1:]]])
        sdef[r['law']] = r['structured_defect'] = float(np.sqrt(np.sum(wts * (rho - rhoA) ** 2) / np.sum(wts * rho ** 2)))
    acc_learner, accepted, binary = gate(defects, LAWS), gate(sdef, LAWS), gate(sdef, ['correct', 'damping-only'])
    for r in lawrows:
        r['accepted_learner'] = int(r['law'] == acc_learner); r['accepted'] = int(r['law'] == accepted); r['binary'] = int(r['law'] == binary)
    # matched augmentation blocks at high wear: same inputs and initial states for every synthetic arm
    blocks = [dict(U=excitation(rng), x0=rng.normal(0, .01, 4), w=w) for w in (.65, .85)]
    mk = lambda kind: [dict(X=simulate(kind, b['w'], b['U'], b['x0']), U=b['U'], w=b['w']) for b in blocks]
    arms = {'duplicate': train[-2:], 'own_twin': [dict(X=L.rollout(B0, b['x0'], b['U'], b['w']), U=b['U'], w=b['w']) for b in blocks]}
    for name in LAWS: arms['law:' + name] = mk(name)
    arms['gated'] = arms['law:' + accepted] if accepted else arms['duplicate']
    arms['binary'] = arms['law:' + binary] if binary else arms['duplicate']
    arms['degraded_real'] = [dict(X=simulate('plant', b['w'], b['U'], b['x0'], rng, noise), U=b['U'], w=b['w']) for b in blocks]
    rows = []
    for arm, extra in arms.items():
        B = L.fit(train + extra, alpha)
        r = dict(seed=seed, wmax=wmax, noise=noise, arm=arm, accepted=accepted or 'abstain', binary=binary or 'abstain',
                 update_rel=float(np.linalg.norm(B - B0) / np.linalg.norm(B0)))
        for h in H: r[f'nrmse_h{h}'] = L.nrmse(B, test, h)
        rows.append(r)
    return rows, lawrows

RHOS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100)

def exposure_seed(seed, wmax=0.15, noise=1.0, laws=('correct', 'damping-only', 'none')):
    """Dose-response: weight rho on a fixed two-trajectory synthetic block (Theorem 4's exposure)."""
    rng = np.random.default_rng(seed)
    train = [traj('plant', w, rng, noise) for w in np.linspace(0, wmax, 4)]
    tune = [traj('plant', w, rng, noise) for w in (wmax / 6, 5 * wmax / 6)]
    test = [traj('plant', w, rng, noise) for w in (.55, .70, .85) for _ in range(3)]
    dep = [traj('plant', w, rng, observe=False) for w in (.55, .70, .85) for _ in range(3)]
    L = Learner(train); alpha = min(ALPHAS, key=lambda a: L.onestep(L.fit(train, a), tune))
    F = L.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in train])); Y = np.vstack([d['X'][1:] for d in train])
    Fd = L.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in dep])); Yd = np.vstack([d['X'][1:] for d in dep])
    th_star = np.linalg.lstsq(Fd, Yd, rcond=None)[0]
    risk = lambda B: float(np.mean((Fd @ (B - th_star) / L.xsd) ** 2))
    blocks = [dict(U=excitation(rng), x0=rng.normal(0, .01, 4), w=w) for w in (.65, .85)]
    rows = []
    for law in laws:
        syn = [dict(X=simulate(law, b['w'], b['U'], b['x0']), U=b['U'], w=b['w']) for b in blocks]
        Fs = L.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in syn])); Ys = np.vstack([d['X'][1:] for d in syn])
        for rho in RHOS:
            B = np.linalg.solve(F.T @ F + rho * Fs.T @ Fs + alpha * np.eye(F.shape[1]), F.T @ Y + rho * Fs.T @ Ys)
            rows.append(dict(seed=seed, wmax=wmax, noise=noise, law=law, rho=rho, onestep_risk=risk(B), nrmse_h1=L.nrmse(B, test, 1), nrmse_h20=L.nrmse(B, test, 20)))
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    OUT = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == 'exposure':
        outs = Parallel(n_jobs=2, verbose=2)(delayed(exposure_seed)(s, w, n) for (w, n) in ((0.15, 1.0), (0.30, 0.25), (0.30, 0.0)) for s in range(16000, 16040))
        pd.DataFrame([r for o in outs for r in o]).to_csv(OUT / 'twin_exposure.csv', index=False); print('done'); sys.exit()
    cells = list(itertools.product((0.05, 0.15, 0.30), (0.0, 0.25, 1.0, 4.0)))
    outs = Parallel(n_jobs=2, verbose=2)(delayed(one_seed)(s, w, n) for (w, n) in cells for s in range(15000, 15040))
    pd.DataFrame([r for o in outs for r in o[0]]).to_csv(OUT / 'twin_screen_arms.csv', index=False)
    pd.DataFrame([r for o in outs for r in o[1]]).to_csv(OUT / 'twin_screen_laws.csv', index=False)
    print('done')
