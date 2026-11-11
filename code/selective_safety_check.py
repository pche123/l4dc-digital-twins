"""Numerical check of the selective-safety theorem (Supplement): with probability >= 1 - delta, every candidate S satisfies
d_f(S) <= d(S) + r(S), r(S) = B (L_X + L_SX(S)) / ||f_hat||_X, with B = sqrt(sum_j beta_j^2) from the self-normalized bound and
L the RMS leverage ||F(x)||_{H^-1} on observed and on mirrored states. Duffing field f(x,v) = (v, x - x^3), cubic library (realizable),
noisy vector-field targets (equation error). A certified gate accepts S only if d(S) + r(S) <= Delta.
usage: python code/selective_safety_check.py data"""
import sys, itertools, numpy as np, pandas as pd
POW = [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(2), d)]
def F(X): return np.hstack([np.ones((len(X), 1))] + [np.prod(X[:, list(c)], 1, keepdims=True) for c in POW])
def f(X): return np.stack([X[:, 1], X[:, 0] - X[:, 0] ** 3], 1)
# The cubic Duffing field has exact coefficient norms (1, sqrt(2)) in this basis.
# M is a declared conservative PHYSICAL PRIOR bound, not an oracle norm used in the radius.
M = np.array([2.0, 2.0])
CANDS = {'exact (-x,-v)': np.diag([-1., -1.]), 'false (-x,+v)': np.diag([-1., 1.]), 'time reversal (+x,-v)': np.diag([1., -1.])}
LAM, DELTA, DEL = 1e-3, 0.05, 0.5
def trial(rng, n, sigma, wells):
    xs = rng.uniform(0.35, 1.4, n) * (rng.choice([-1, 1], n) if wells == 2 else 1); vs = rng.uniform(-.5, .5, n)
    X = np.stack([xs, vs], 1); Y = f(X) + sigma * rng.normal(size=(n, 2))
    P = F(X); d = P.shape[1]; H = P.T @ P + LAM * np.eye(d); Hi = np.linalg.inv(H); Th = Hi @ P.T @ Y
    fh = lambda Z: F(Z) @ Th; nf = np.sqrt(np.mean(np.sum(fh(X) ** 2, 1)))
    _, ld = np.linalg.slogdet(H)
    beta = [sigma * np.sqrt(ld - d * np.log(LAM) + 2 * np.log(2 / DELTA)) + np.sqrt(LAM) * M[j] for j in range(2)]
    B = np.sqrt(sum(b * b for b in beta)); lev = lambda Z: np.sqrt(np.mean(np.einsum('ij,jk,ik->i', F(Z), Hi, F(Z))))
    LX = lev(X); out = []
    for name, S in CANDS.items():
        SX = X @ S.T
        dS = np.sqrt(np.mean(np.sum((fh(SX) - fh(X) @ S.T) ** 2, 1))) / nf
        dfS = np.sqrt(np.mean(np.sum((f(SX) - f(X) @ S.T) ** 2, 1))) / nf
        r = B * (LX + lev(SX)) / nf
        out.append(dict(cand=name, d=dS, d_true=dfS, r=r, holds=dfS <= dS + r + 1e-12, true=name.startswith('exact')))
    o = pd.DataFrame(out); best = o.sort_values('d').iloc[0]; second = o.sort_values('d').iloc[1]
    std_acc = best.d <= .5 and best.d <= .5 * second.d
    cert = o[(o.d + o.r) <= DEL]
    return dict(n=n, sigma=sigma, wells=wells, holds_all=bool(o.holds.all()), r_exact=float(o[o.true].r.iloc[0]), d_exact=float(o[o.true].d.iloc[0]),
                std_accept=bool(std_acc), std_wrong=bool(std_acc and not best.true), cert_accept_true=bool(cert.true.any()), cert_wrong=bool((~cert.true).any()))
if __name__ == '__main__':
    rng = np.random.default_rng(2027); rows = []
    for n, sigma, wells in itertools.product([200, 2000], [0.01, 0.05, 0.2], [1, 2]):
        for _ in range(500): rows.append(trial(rng, n, sigma, wells))
    D = pd.DataFrame(rows); D.to_csv(sys.argv[1] + '/selective_safety_check.csv', index=False)
    g = D.groupby(['wells', 'n', 'sigma']).agg(bound_holds=('holds_all', 'mean'), r_exact=('r_exact', 'median'), d_exact=('d_exact', 'median'),
        std_accept=('std_accept', 'mean'), std_wrong=('std_wrong', 'mean'), cert_accept=('cert_accept_true', 'mean'), cert_wrong=('cert_wrong', 'mean'))
    pd.set_option('display.width', 200); print(g.round(4).to_string())
