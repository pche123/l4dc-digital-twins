"""Bayesian reading of a synthetic block (linear-Gaussian model). A noiseless synthetic block (Phi_g, theta_g) with exposure rho is
exactly the Gaussian prior N(theta_g, sigma^2/rho (Phi_g^T Phi_g)^-1). We check (i) the identity, (ii) credible-interval coverage when rollouts
are treated as evidence, and (iii) choosing the exposure rho by marginal likelihood (empirical Bayes) against fixed rho and the oracle best rho.
usage: python code/bayes_view.py data"""
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(2027); d, n, m, sig, lam = 8, 20, 40, 0.5, 1e-3
RHOS = np.array([0, .01, .03, .1, .3, 1, 3, 10, 30, 100.])
z90 = 1.6449
def run(bias, trials=600):
    out = {k: [] for k in ('mse_real', 'mse_rho1', 'mse_eb', 'mse_oracle', 'cov_real', 'cov_self1', 'cov_biased1', 'cov_eb', 'rho_eb', 'rho_or', 'ident')}
    for _ in range(trials):
        th = rng.normal(size=d); P = rng.normal(size=(n, d)); y = P @ th + sig * rng.normal(size=n)
        Pg = 1.5 * rng.normal(size=(m, d)); Pt = rng.normal(size=(200, d)); H = P.T @ P + lam * np.eye(d); G0 = Pg.T @ Pg
        thg = th + bias * rng.normal(size=d) / np.sqrt(d); b = P.T @ y
        def post(rho, tg):
            Ht = H + rho * G0; mu = np.linalg.solve(Ht, b + rho * G0 @ tg); return mu, Ht
        th0, _ = post(0, thg)
        # identity: block update == prior update
        mu1, _ = post(1., thg); e_G = np.linalg.solve(H + G0, H @ th0 + G0 @ thg); out['ident'].append(np.linalg.norm(mu1 - e_G))
        risk = lambda mu: float(np.mean((Pt @ (mu - th)) ** 2))
        mses = [risk(post(r, thg)[0]) for r in RHOS]
        # marginal likelihood of y under prior N(thg, sig^2/rho G0^-1) (+ base ridge ignored)
        ev = []
        for r in RHOS[1:]:
            C = sig ** 2 * (np.eye(n) + P @ np.linalg.solve(G0, P.T) / r); s, ld = np.linalg.slogdet(C); res = y - P @ thg
            ev.append(-.5 * (ld + res @ np.linalg.solve(C, res)))
        r_eb = RHOS[1:][int(np.argmax(ev))]; r_or = RHOS[int(np.argmin(mses))]
        out['mse_real'].append(mses[0]); out['mse_rho1'].append(risk(post(1., thg)[0])); out['mse_eb'].append(risk(post(r_eb, thg)[0])); out['mse_oracle'].append(min(mses))
        out['rho_eb'].append(r_eb); out['rho_or'].append(r_or)
        def cov(mu, Ht):
            sd = sig * np.sqrt(np.einsum('ij,jk,ik->i', Pt, np.linalg.inv(Ht), Pt)); return float(np.mean(np.abs(Pt @ (mu - th)) <= z90 * sd))
        out['cov_real'].append(cov(th0, H))
        selfg = th0  # rollouts of the fitted model: theta_g = posterior mean
        ms, Hs = post(1., selfg); out['cov_self1'].append(cov(ms, Hs))
        mb, Hb = post(1., thg); out['cov_biased1'].append(cov(mb, Hb))
        me, He = post(r_eb, thg); out['cov_eb'].append(cov(me, He))
    return {k: (np.median(v) if k.startswith(('mse', 'rho')) else float(np.mean(v))) for k, v in out.items()} | {'bias': bias}
rows = [run(b) for b in (0.0, 0.3, 1.0, 3.0)]
df = pd.DataFrame(rows); pd.set_option('display.width', 200); print(df.round(4).to_string())
df.to_csv(sys.argv[1] + '/bayes_view.csv', index=False)
