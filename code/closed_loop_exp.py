"""Closed-loop test: LQR designed on each learned model (at wear 0.7), evaluated on the true plant."""
import sys, numpy as np, pandas as pd
from scipy.linalg import solve_discrete_are
from twin_screen import *
import twin_screen as TS

WC = 0.70
QW = np.diag([1., 1., .1, .1]); RW = np.array([[0.01]])

def lin_model(pred):
    z = np.zeros((1, 4)); f0 = pred(z, 0.); A = np.zeros((4, 4))
    for i in range(4):
        e = np.zeros((1, 4)); e[0, i] = 1e-3; A[:, i] = (pred(e, 0.) - f0)[0] / 1e-3
    Bv = (pred(z, 1e-3) - f0)[0] / 1e-3
    return A, Bv[:, None]

def lqr(A, B):
    try:
        P = solve_discrete_are(A, B, QW, RW); return np.linalg.solve(RW + B.T @ P @ B, B.T @ P @ A)
    except Exception:
        return None

def cost(Ap, Bp, K, x0s, n=400):
    if K is None: return np.inf
    Acl = Ap - Bp @ K
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1: return np.inf
    J = 0.
    for x in x0s:
        for _ in range(n):
            u = -(K @ x); J += float(x @ QW @ x) + RW[0, 0] * float(u @ u); x = Acl @ x
    return J / len(x0s)

PLANT_AB = lin_model(lambda X, u: TS.step(X, u, TS.params('plant', WC)))
X0S = [np.array([.05, -.03, 0., 0.]), np.array([-.04, .06, .1, 0.]), np.array([.03, .03, 0., -.1])]
J_OR = cost(*PLANT_AB, lqr(*PLANT_AB), X0S)

def closed_loop(L, B):
    A, Bv = lin_model(lambda X, u: L.predict(B, X, np.full(len(X), u), WC))
    J = cost(*PLANT_AB, lqr(A, Bv), X0S)
    return dict(J_ratio=J / J_OR, unstable=int(not np.isfinite(J)))

def one_seed_cl(seed, wmax=0.15, noise=1.0):
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
        r = dict(seed=seed, wmax=wmax, noise=noise, arm=arm, accepted=accepted or 'abstain')
        r.update(closed_loop(L, B))
        rows.append(r)
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]
    rows = Parallel(n_jobs=2)(delayed(one_seed_cl)(s) for s in range(lo, hi))
    df = pd.DataFrame([r for rr in rows for r in rr]); df.to_csv(out, index=False); print(df.shape)
