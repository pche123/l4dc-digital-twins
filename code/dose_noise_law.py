"""Test of the noise-to-bias dosing law (Corollary, Supplement A): rho* = sigma^2 / b^2.

For a fixed twin (fixed bias) the law predicts that the best exposure grows with the square of the
sensor-noise level. Fresh seeds 19000-19039, wear seen up to 0.30, noise multipliers 1/8 ... 2;
the block, learner and one-step deployment risk are those of the dose-response study
(twin_screen.exposure_seed), evaluated on a finer exposure grid.
Output: data/dose_noise_law.csv

usage: python code/dose_noise_law.py data
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from twin_screen import ALPHAS, Learner, excitation, feats, simulate, traj

RHOS = np.concatenate([[0.0], np.logspace(-4, 3, 57)])
NOISES = (0.125, 0.25, 0.5, 1.0, 2.0)
LAWS = ('correct', 'damping-only', 'none')

SD = np.array([.0025, .0025, .01, .01])

def one(seed, noise, design, wmax=0.30):
    """design='sensor': noisy states, as in the paper (regressors and targets both noisy);
    design='equation': clean regressors, noise of the same size on the targets only."""
    rng = np.random.default_rng(seed)
    if design == 'sensor':
        train = [traj('plant', w, rng, noise) for w in np.linspace(0, wmax, 4)]
        tune = [traj('plant', w, rng, noise) for w in (wmax / 6, 5 * wmax / 6)]
    else:
        train = [traj('plant', w, rng, observe=False) for w in np.linspace(0, wmax, 4)]
        tune = [traj('plant', w, rng, observe=False) for w in (wmax / 6, 5 * wmax / 6)]
    dep = [traj('plant', w, rng, observe=False) for w in (.55, .70, .85) for _ in range(3)]
    L = Learner(train); alpha = min(ALPHAS, key=lambda a: L.onestep(L.fit(train, a), tune))
    Z = lambda data: L.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in data]))
    T = lambda data: np.vstack([d['X'][1:] for d in data])
    F, Y, Fd, Yd = Z(train), T(train), Z(dep), T(dep)
    if design == 'equation':
        Y = Y + rng.normal(0, 1, Y.shape) * SD * noise
        Ft, Yt = Z(tune), T(tune); Yt = Yt + rng.normal(0, 1, Yt.shape) * SD * noise
        ridge = lambda a: np.linalg.solve(F.T @ F + a * np.eye(F.shape[1]), F.T @ Y)
        alpha = min(ALPHAS, key=lambda a: float(np.mean(((Ft @ ridge(a) - Yt) / L.xsd) ** 2)))
    th_star = np.linalg.lstsq(Fd, Yd, rcond=None)[0]
    risk = lambda B: float(np.mean((Fd @ (B - th_star) / L.xsd) ** 2))
    blocks = [dict(U=excitation(rng), x0=rng.normal(0, .01, 4), w=w) for w in (.65, .85)]
    rows = []
    for law in LAWS:
        syn = [dict(X=simulate(law, b['w'], b['U'], b['x0']), U=b['U'], w=b['w']) for b in blocks]
        Fs, Ys = Z(syn), T(syn)
        r = [risk(np.linalg.solve(F.T @ F + rho * Fs.T @ Fs + alpha * np.eye(F.shape[1]), F.T @ Y + rho * Fs.T @ Ys)) for rho in RHOS]
        k = int(np.argmin(r))
        rows.append(dict(seed=seed, design=design, noise=noise, law=law, rho_star=RHOS[k], best_rel=r[k] / r[0], end_rel=r[-1] / r[0]))
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=2)(delayed(one)(s, n, d) for d in ('equation', 'sensor') for n in NOISES for s in range(19000, 19040))
    D = pd.DataFrame([r for o in outs for r in o]); D.to_csv(Path(sys.argv[1]) / 'dose_noise_law.csv', index=False)
    for design, Dd in D.groupby('design'):
        print(design); print(Dd.groupby(['law', 'noise']).rho_star.median().unstack(0).to_string())
        for law in LAWS:
            g = Dd[(Dd.law == law) & (Dd.rho_star > 0) & (Dd.rho_star < RHOS[-1])]
            slope = np.polyfit(np.log(g.noise), np.log(g.rho_star), 1)[0]
            print(f'  {law}: log-log slope of rho* against noise level = {slope:.2f} (law predicts 2); interior optima {len(g)}/{len(Dd[Dd.law == law])}')