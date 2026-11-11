"""Monte Carlo check of the finite-sample bound (Supplement, Proposition 8) for the augmented ridge learner.
Reports how often the bound fails (should be <= delta) and realised error / bound.   usage: python code/finite_bound_check.py data"""
import sys, numpy as np, pandas as pd
from pathlib import Path
CASES = {'correct twin': dict(mis=0.0), 'mismatched, noisy twin': dict(), 'mismatched, noiseless twin': dict(sigg=0.0, mis=0.5),
         'noisy data, small lambda': dict(sig=1.0, lam=0.1, n=30), 'higher dimension': dict(d=15, n=40, m=100)}
def run(d=6, n=60, m=60, lam=1.0, sig=0.3, sigg=0.1, mis=0.2, delta=0.05, T=4000, seed=0):
    rng = np.random.default_rng(seed); th = rng.normal(size=d); th /= np.linalg.norm(th); thg = th + mis * rng.normal(size=d) / np.sqrt(d)
    ld = lambda A: np.linalg.slogdet(A / lam)[1]; viol, ratio = 0, []
    for _ in range(T):
        Phi = rng.normal(size=(n, d)); C = 1.5 * rng.normal(size=(m, d)); Y = Phi @ th + sig * rng.normal(size=n); U = C @ thg + sigg * rng.normal(size=m)
        H = Phi.T @ Phi + lam * np.eye(d); G = C.T @ C + lam * np.eye(d); HC = H + C.T @ C
        e = np.linalg.solve(HC, Phi.T @ Y + C.T @ U) - th; lhs = np.sqrt(e @ HC @ e)
        B = sig * np.sqrt(ld(H) + 2 * np.log(2 / delta)) + sigg * np.sqrt(ld(G) + 2 * np.log(2 / delta)) + np.sqrt(lam) * np.linalg.norm(th) + np.linalg.norm(C @ (thg - th))
        viol += lhs > B; ratio.append(lhs / B)
    return dict(violation_rate=viol / T, median_ratio=float(np.median(ratio)), max_ratio=float(np.max(ratio)))
def mirrored(break_eps, d=3, n=80, lam=1.0, sig=0.2, delta=0.05, T=3000, seed=1):
    """mirrored rows (x -> -x) that share the real data's noise; library [x, x^3, x^2], the even term breaks the symmetry"""
    rng = np.random.default_rng(seed); th = np.array([1.0, -.5, break_eps]); F = lambda x: np.stack([x, x ** 3, x ** 2], 1)
    ld = lambda A: np.linalg.slogdet(A / lam)[1]; viol, ratio = 0, []
    for _ in range(T):
        x = rng.uniform(.3, 1.5, n); f = F(x) @ th; y = f + sig * rng.normal(size=n); Phi = F(x); C = F(-x); U = -y
        H = Phi.T @ Phi + lam * np.eye(d); G = C.T @ C + lam * np.eye(d); HC = H + C.T @ C
        e = np.linalg.solve(HC, Phi.T @ y + C.T @ U) - th; lhs = np.sqrt(e @ HC @ e)
        B = sig * (np.sqrt(ld(H) + 2 * np.log(2 / delta)) + np.sqrt(ld(G) + 2 * np.log(2 / delta))) + np.sqrt(lam) * np.linalg.norm(th) + np.linalg.norm(-f - C @ th)
        viol += lhs > B; ratio.append(lhs / B)
    return dict(violation_rate=viol / T, median_ratio=float(np.median(ratio)), max_ratio=float(np.max(ratio)))
def fitted_twin(sig_g, d=6, n=60, m=60, lam=1.0, sig=0.3, delta=0.05, T=4000, seed=2):
    """twin fitted on the SAME real data (ridge on the first d-2 features only, so it is misspecified), then simulating fresh rows with fresh noise"""
    rng = np.random.default_rng(seed); th = rng.normal(size=d); th /= np.linalg.norm(th); ld = lambda A: np.linalg.slogdet(A / lam)[1]; viol, ratio = 0, []
    for _ in range(T):
        Phi = rng.normal(size=(n, d)); Y = Phi @ th + sig * rng.normal(size=n); k = d - 2
        thg = np.zeros(d); thg[:k] = np.linalg.solve(Phi[:, :k].T @ Phi[:, :k] + lam * np.eye(k), Phi[:, :k].T @ Y)   # depends on the real noise
        C = 1.5 * rng.normal(size=(m, d)); U = C @ thg + sig_g * rng.normal(size=m)
        H = Phi.T @ Phi + lam * np.eye(d); G = C.T @ C + lam * np.eye(d); HC = H + C.T @ C
        e = np.linalg.solve(HC, Phi.T @ Y + C.T @ U) - th; lhs = np.sqrt(e @ HC @ e)
        B = sig * np.sqrt(ld(H) + 2 * np.log(2 / delta)) + sig_g * np.sqrt(ld(G) + 2 * np.log(2 / delta)) + np.sqrt(lam) * np.linalg.norm(th) + np.linalg.norm(C @ (thg - th))
        viol += lhs > B; ratio.append(lhs / B)
    return dict(violation_rate=viol / T, median_ratio=float(np.median(ratio)), max_ratio=float(np.max(ratio)))
FITTED = {'twin fitted on the same data, noisy': 0.2, 'twin fitted on the same data, deterministic': 0.0}
MIRROR = {'mirrored rows, true symmetry': 0.0, 'mirrored rows, weakly broken': 0.3, 'mirrored rows, strongly broken': 1.0}
if __name__ == '__main__':
    out = pd.DataFrame([dict(case=k, **run(**v)) for k, v in CASES.items()] + [dict(case=k, **mirrored(v)) for k, v in MIRROR.items()] + [dict(case=k, **fitted_twin(v)) for k, v in FITTED.items()]); print(out.round(3).to_string(index=False))
    out.to_csv(Path(sys.argv[1]) / 'finite_bound_check.csv', index=False)
