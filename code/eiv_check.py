"""Excitation theorem with noisy regressors (errors-in-variables). Data y = x'theta* + eps, regressors observed as z = x + u.
Deployment on raw noisy measurements has excess risk (theta - theta_t)' Sz (theta - theta_t) with attenuated target
theta_t = Sz^{-1} Sx theta*, Sz = Sx + Su. The multi-direction exposure identity therefore holds with e = theta_hat - theta_t,
g = theta_g - theta_t, Q = Sz. A correct noiseless twin (theta_g = theta*) is biased for noisy deployment (g != 0) and has a finite
best exposure; it helps clean-input deployment (target theta*). Adding matched input noise to the twin's regressors realigns it.
usage: python code/eiv_check.py data"""
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(7); d, n, m, lam = 6, 60, 120, 1e-2
RHOS = [0, .01, .03, .1, .3, 1, 3, 10, 30]
rows = []; worst = 0
for trial in range(400):
    A = rng.normal(size=(d, d)); Sx = A @ A.T / d + .2 * np.eye(d); Su = np.diag(rng.uniform(.1, .8, d)); Sz = Sx + Su
    th = rng.normal(size=d); tt = np.linalg.solve(Sz, Sx @ th)
    Lx = np.linalg.cholesky(Sx); Lu = np.linalg.cholesky(Su)
    X = rng.normal(size=(n, d)) @ Lx.T; Z = X + rng.normal(size=(n, d)) @ Lu.T; y = X @ th + .3 * rng.normal(size=n)
    H = Z.T @ Z + lam * np.eye(d); thh = np.linalg.solve(H, Z.T @ y)
    C = rng.normal(size=(m, d)) @ Lx.T                      # twin rows at clean states
    Cn = C + rng.normal(size=(m, d)) @ Lu.T                 # twin rows with matched input noise
    for rho in RHOS:
        G = rho * C.T @ C
        tc = np.linalg.solve(H + G, Z.T @ y + rho * C.T @ (C @ th))                 # correct noiseless twin
        Gn = rho * Cn.T @ Cn; tn = np.linalg.solve(H + Gn, Z.T @ y + rho * Cn.T @ (C @ th))   # matched-noise twin (labels from clean states)
        risk_noisy = lambda t: (t - tt) @ Sz @ (t - tt); risk_clean = lambda t: (t - th) @ Sx @ (t - th)
        if rho > 0:   # identity check: predicted change for noisy deployment via multi-direction formula
            V, R = C.T, rho * np.eye(m); Hi = np.linalg.inv(H); e = thh - tt; g = th - tt
            K = np.linalg.inv(np.linalg.inv(R) + V.T @ Hi @ V); s = V.T @ (e - g); M = V.T @ Hi @ Sz @ e; Cc = V.T @ Hi @ Sz @ Hi @ V
            pred = -2 * s @ K @ M + s @ K @ Cc @ K @ s; act = risk_noisy(tc) - risk_noisy(thh)
            worst = max(worst, abs(pred - act) / (abs(act) + 1e-12))
        rows.append(dict(trial=trial, rho=rho, noisy_correct=risk_noisy(tc), clean_correct=risk_clean(tc), noisy_matched=risk_noisy(tn)))
D = pd.DataFrame(rows); D.to_csv(sys.argv[1] + '/eiv_check.csv', index=False)
base = D[D.rho == 0].set_index('trial')
for c in ['noisy_correct', 'clean_correct', 'noisy_matched']: D[c + '_rel'] = D[c] / D.trial.map(base[c])
print('max relative error of the identity over', 400 * (len(RHOS) - 1), 'cases:', worst)
print(D.groupby('rho')[['noisy_correct_rel', 'clean_correct_rel', 'noisy_matched_rel']].median().round(3).to_string())
best = D[D.rho > 0].loc[D[D.rho > 0].groupby('trial').noisy_correct.idxmin()].rho
print('interior optimum (noisy deployment, correct twin) in', round(((best > 0) & (best < max(RHOS))).mean(), 3), 'of trials; harmed at rho=30 in',
      round((D[D.rho == 30].set_index('trial').noisy_correct > base.noisy_correct).mean(), 3))
