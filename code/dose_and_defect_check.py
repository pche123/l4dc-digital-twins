"""Randomized checks of two results in the supplement.

1. Dose-bias equivalence (Corollary, Supplement A): after a rank-one synthetic block the learner's
   error depends on the exposure rho and the generator g only through beta = alpha * s, so two
   different generators with matched beta leave the learner in the same state, and the largest
   attainable risk reduction along the direction is m^2 / c for every generator.
2. Verification-value inequality (Proposition, Supplement B): mirrored rows (F(SX), Sy) move the
   verifier's own ridge fit by at most sqrt(n) * ||fhat||_X * (d(S) + r_X) in the H_C norm, and
   change its predictions on the mirrored and on the observed states by at most the same amount.

3. Noise-to-bias dosing law (Corollary, Supplement A): for a least-squares learner with error
   covariance sigma^2 H^{-1} and a twin with bias b along v, the exposure minimizing the expected
   deployment risk is rho* = sigma^2 / b^2 for every deployment metric Q.

usage: python code/dose_and_defect_check.py
"""
import numpy as np

rng = np.random.default_rng(20261113)

# ---- 1. dose-bias equivalence
n_pairs, worst_state, worst_best = 0, 0.0, 0.0
for _ in range(5000):
    p = int(rng.integers(2, 9))
    A = rng.normal(size=(p + 3, p)); H = A.T @ A + 0.1 * np.eye(p)
    B = rng.normal(size=(p, p)); Q = B.T @ B
    e, v, g1, g2 = (rng.normal(size=p) for _ in range(4))
    Hi = np.linalg.inv(H); h = v @ Hi @ v; m = v @ Hi @ Q @ e; c = v @ Hi @ Q @ Hi @ v
    after = lambda rho, g: np.linalg.solve(H + rho * np.outer(v, v), H @ e + rho * v * (v @ g))
    rho1 = rng.uniform(0.01, 5); a1 = rho1 / (1 + rho1 * h); beta = a1 * (v @ (e - g1))
    a2 = beta / (v @ (e - g2))
    if 0 < a2 < 1 / h:                                   # generator 2 can reach the same beta
        e1, e2 = after(rho1, g1), after(a2 / (1 - a2 * h), g2)
        worst_state = max(worst_state, np.linalg.norm(e1 - e2) / np.linalg.norm(e1)); n_pairs += 1
    for g in (g1, g2):                                   # best attainable reduction is m^2/c
        s = v @ (e - g); a = m / (s * c)
        if 0 < a < 1 / h:
            er = after(a / (1 - a * h), g)
            worst_best = max(worst_best, abs((e @ Q @ e - er @ Q @ er) - m * m / c) / (m * m / c))
print(f'dose-bias equivalence: {n_pairs} matched pairs, largest relative state difference {worst_state:.1e}')
print(f'best reduction equals m^2/c: largest relative error {worst_best:.1e}')

# ---- 2. verification-value inequality (cubic library, one Duffing well, random sign flips)
def lib(X):
    x, v = X[:, 0], X[:, 1]
    return np.stack([np.ones_like(x), x, v, x * x, x * v, v * v, x ** 3, x * x * v, x * v * v, v ** 3], 1)
field = lambda X: np.stack([X[:, 1], X[:, 0] - X[:, 0] ** 3], 1)
ratios, violations, exact = [], 0, 0.0
for _ in range(5000):
    n = int(rng.integers(30, 300))
    X = np.stack([rng.uniform(.3, 1.4, n), rng.uniform(-.6, .6, n)], 1)
    Y = field(X) + rng.normal(0, rng.choice([0, .01, .1, .5]), (n, 2)); lam = 10 ** rng.uniform(-8, 0)
    S = np.diag(rng.choice([-1., 1.], 2))
    if S[0, 0] == 1 and S[1, 1] == 1: continue
    F = lib(X); H = F.T @ F + lam * np.eye(F.shape[1]); Th = np.linalg.solve(H, F.T @ Y); fh = F @ Th
    C = lib(X @ S); HC = H + C.T @ C; D = np.linalg.solve(HC, C.T @ (Y @ S - C @ Th))
    scale = np.linalg.norm(fh)                           # sqrt(n) * ||fhat||_X
    d = np.linalg.norm(C @ Th - fh @ S) / scale; rX = np.linalg.norm(Y - fh) / scale
    bound = scale * (d + rX); move = np.sqrt(np.trace(D.T @ HC @ D))
    K = C @ np.linalg.solve(H, C.T); R = Y @ S - C @ Th      # exact identity for the update energy
    exact = max(exact, abs(move ** 2 - np.sum(R * (K @ np.linalg.solve(np.eye(n) + K, R)))) / move ** 2)
    ratios.append(move / bound)
    violations += (move > bound * (1 + 1e-9)) + (np.linalg.norm(C @ D) > bound * (1 + 1e-9)) + (np.linalg.norm(F @ D) > bound * (1 + 1e-9))
print(f'verification-value inequality: {violations} violations in {len(ratios)} instances; '
      f'update/bound median {np.median(ratios):.2f}, max {max(ratios):.3f}')
print(f'update energy equals <R, K(I+K)^-1 R>: largest relative error {exact:.1e}')

# ---- 3. noise-to-bias dosing law: exact expected risk at rho* = sigma^2/b^2 versus a numerical search
from scipy.optimize import minimize_scalar
worst_gap, worst_value = 0.0, 0.0
for _ in range(2000):
    p = int(rng.integers(2, 9))
    A = rng.normal(size=(p + 3, p)); H = A.T @ A; B = rng.normal(size=(p, p)); Q = B.T @ B
    v, g = rng.normal(size=p), rng.normal(size=p); sig2 = 10 ** rng.uniform(-3, 1); b = v @ g
    Hi = np.linalg.inv(H); h = v @ Hi @ v; c = v @ Hi @ Q @ Hi @ v
    def expected_risk(logrho):                               # e_rho = M (H e + G g), Cov(e) = sig2 H^-1
        G = np.exp(logrho) * np.outer(v, v); M = np.linalg.inv(H + G); bias = M @ G @ g
        return bias @ Q @ bias + sig2 * np.trace(Q @ M @ H @ M)
    law = np.log(sig2 / b ** 2)
    opt = minimize_scalar(expected_risk, bounds=(law - 6, law + 6), method='bounded', options=dict(xatol=1e-10))
    worst_gap = max(worst_gap, (expected_risk(law) - opt.fun) / opt.fun)
    drop = sig2 ** 2 * c / (sig2 * h + b * b)                 # closed-form expected reduction at rho*
    worst_value = max(worst_value, abs((sig2 * np.trace(Q @ Hi) - expected_risk(law)) - drop) / drop)
print(f'dosing law: expected risk at rho* = sigma^2/b^2 exceeds the numerical minimum by at most {worst_gap:.1e} (relative); '
      f'closed-form reduction matches to {worst_value:.1e}')

# ---- 4. dosing law for a learner that is also biased (mean error a): alpha* = (sig2 c + (v'a - b) m_a) / (c (sig2 h + (v'a - b)^2))
worst = 0.0
for _ in range(2000):
    p = int(rng.integers(2, 9))
    A = rng.normal(size=(p + 3, p)); H = A.T @ A; B = rng.normal(size=(p, p)); Q = B.T @ B
    v, g, a = rng.normal(size=p), rng.normal(size=p), 0.3 * rng.normal(size=p); sig2 = 10 ** rng.uniform(-3, 1); b = v @ g
    Hi = np.linalg.inv(H); h = v @ Hi @ v; c = v @ Hi @ Q @ Hi @ v; ma = v @ Hi @ Q @ a; d = v @ a - b
    def expected_risk(alpha):                                 # E R_Q(e - alpha H^-1 v s), e ~ (a, sig2 H^-1)
        T = np.eye(p) - alpha * np.outer(Hi @ v, v); mean = T @ a + alpha * Hi @ v * b
        return mean @ Q @ mean + sig2 * np.trace(Q @ T @ Hi @ T.T)
    astar = (sig2 * c + d * ma) / (c * (sig2 * h + d * d))
    opt = minimize_scalar(expected_risk, bounds=(astar - 5, astar + 5), method='bounded', options=dict(xatol=1e-12))
    worst = max(worst, (expected_risk(astar) - opt.fun) / opt.fun)
print(f'biased-learner dosing law: expected risk at the closed-form alpha* exceeds the numerical minimum by at most {worst:.1e} (relative)')
