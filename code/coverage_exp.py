"""Pseudo-confidence: coverage of nominal 90% one-step predictive intervals vs synthetic exposure on the 2-DOF twin.
A practitioner treats every row as evidence: ridge posterior with noise variance estimated from all (weighted) rows.
Arms: duplicate (copies of real rows), self (the learner's own one-step labels on the twin's states), laws (correct / damping-only / none),
real (fresh plant data at high wear, the honest reference). Exposure rho scales each synthetic block's weight.
Evaluated on unseen-regime deployment trajectories (wear .55-.85) and on fresh seen-regime trajectories (training wear).
usage: python code/coverage_exp.py 18000 18040 data/coverage_twin.csv"""
import sys, numpy as np, pandas as pd
from twin_screen import *
import twin_screen as TS
RHOS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
Z90 = 1.6449

def fitw(L, real, syn, rho, alpha, noise_from='all'):
    tgt = lambda d: d['Y'] if 'Y' in d else d['X'][1:]
    F = [L.z(feats(d['X'][:-1], d['U'][:-1], d['w'])) for d in real]; Y = [tgt(d) for d in real]; W = [np.ones(len(f)) for f in F]
    nreal = sum(len(f) for f in F)
    for d in syn:
        F.append(L.z(feats(d['X'][:-1], d['U'][:-1], d['w']))); Y.append(tgt(d)); W.append(np.full(len(F[-1]), rho))
    F = np.vstack(F); Y = np.vstack(Y); W = np.concatenate(W); isreal = np.arange(len(W)) < nreal
    keep = W > 0; F, Y, W, isreal = F[keep], Y[keep], W[keep], isreal[keep]
    Hm = (F * W[:, None]).T @ F + alpha * np.eye(F.shape[1]); B = np.linalg.solve(Hm, (F * W[:, None]).T @ Y)
    R = Y - F @ B; p = F.shape[1]
    if noise_from == 'all':      # naive: every weighted row counts as an observation
        s2 = (W[:, None] * R ** 2).sum(0) / max(W.sum() - p, 1.0)
    else:                        # residual variance from the real rows only
        s2 = (R[isreal] ** 2).sum(0) / max(isreal.sum() - p, 1.0)
    return B, np.linalg.inv(Hm), s2

def cover(L, B, Hi, s2, data):
    hit, err, wid = [], [], []
    for d in data:
        F = L.z(feats(d['X'][:-1], d['U'][:-1], d['w'])); P = F @ B; Y = d['X'][1:]
        lev = np.einsum('ij,jk,ik->i', F, Hi, F); sd = np.sqrt(s2[None, :] * (1 + lev[:, None]))
        hit.append(np.abs(Y - P) <= Z90 * sd); err.append(((Y - P) / L.xsd) ** 2); wid.append(sd / L.xsd)
    return float(np.mean(np.concatenate(hit))), float(np.sqrt(np.mean(np.concatenate(err)))), float(np.median(np.concatenate(wid)))

def one_seed_cov(seed, wmax=0.15, noise=1.0):
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
    seen = [traj('plant', w, rng, noise) for w in wtr]               # fresh trajectories in the seen regime
    arms['real'] = [dict(X=simulate('plant', b['w'], b['U'], b['x0'], rng, noise), U=b['U'], w=b['w']) for b in blocks]
    # information-closed self-labels: the correct-law states labelled by the fitted learner's own one-step predictions
    arms['self'] = [dict(X=d['X'], U=d['U'], w=d['w'], Y=L.predict(B0, d['X'][:-1], d['U'][:-1], d['w'])) for d in arms['law:correct']]
    rows = []
    for arm in ('duplicate', 'self', 'law:correct', 'law:damping-only', 'law:none', 'real'):
        for rho in RHOS:
            for nf in ('all', 'real'):
                B, Hi, s2 = fitw(L, train, arms[arm], rho, alpha, nf)
                for reg, data in (('deployment', test), ('seen', seen)):
                    c, e, wd = cover(L, B, Hi, s2, data)
                    rows.append(dict(seed=seed, arm=arm, rho=rho, noise_from=nf, regime=reg, coverage=c, rmse=e, width=wd, accepted=accepted or 'abstain'))
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]
    rows = Parallel(n_jobs=2)(delayed(one_seed_cov)(s) for s in range(lo, hi))
    pd.DataFrame([r for rr in rows for r in rr]).to_csv(out, index=False); print('done')
