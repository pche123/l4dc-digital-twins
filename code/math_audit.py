"""Numerical checks of the paper's identities on random instances.  usage: python code/math_audit.py"""
import numpy as np
rng = np.random.default_rng(0)
def close(a, b, tol=1e-8): return np.allclose(a, b, atol=tol, rtol=1e-7)
checks = {}
for trial in range(200):
    p, k, n, m, nv = 6, 3, 40, 9, 25
    Phi, Y = rng.normal(size=(n, p)), rng.normal(size=(n, k)); lam = rng.uniform(.01, 1)
    H = Phi.T @ Phi + lam * np.eye(p); Th = np.linalg.solve(H, Phi.T @ Y)
    C = rng.normal(size=(m, p)); Thg = rng.normal(size=(p, k)); Eg = .1 * rng.normal(size=(m, k)); U = C @ Thg + Eg
    HC = H + C.T @ C; ThC = np.linalg.solve(HC, Phi.T @ Y + C.T @ U)
    checks.setdefault('Lemma 1 block update', []).append(close(ThC - Th, np.linalg.solve(HC, C.T @ (U - C @ Th))))
    Wr, Ws = np.linalg.solve(HC, H), np.linalg.solve(HC, C.T @ C)
    checks.setdefault('Eq. (4) pull identity', []).append(close(ThC, Wr @ Th + Ws @ Thg + np.linalg.solve(HC, C.T @ Eg)))
    ThS = np.linalg.solve(HC, Phi.T @ Y + C.T @ (C @ Th))
    checks.setdefault('Corollary 2 self-consistency', []).append(close(ThS, Th))
    Pv, Yv = rng.normal(size=(nv, p)), rng.normal(size=(nv, k)); Rv = Yv - Pv @ Th
    Da, Ds = rng.normal(size=(p, k)), rng.normal(size=(p, k))
    L = lambda T: np.sum((Yv - Pv @ T) ** 2) / nv
    V = 2 / nv * np.sum((Pv.T @ Rv) * (Da - Ds)) + (np.sum((Pv @ Ds) ** 2) - np.sum((Pv @ Da) ** 2)) / nv
    checks.setdefault('Eq. (5) value identity', []).append(close(L(Th + Ds) - L(Th + Da), V))
    # Theorem 4 (vectorised parameter)
    e, g, v = rng.normal(size=p), rng.normal(size=p), rng.normal(size=p); Q = (lambda B: B @ B.T)(rng.normal(size=(p, p))); rho = rng.uniform(.1, 5)
    Hi = np.linalg.inv(H); h, s_, mm, c = v @ Hi @ v, v @ (e - g), v @ Hi @ Q @ e, v @ Hi @ Q @ Hi @ v; al = rho / (1 + rho * h)
    erho = np.linalg.solve(H + rho * np.outer(v, v), H @ e + rho * np.outer(v, v) @ g)
    checks.setdefault('Theorem 4 rank-one update', []).append(close(erho, e - al * Hi @ v * s_))
    checks.setdefault('Theorem 4 risk change', []).append(close(erho @ Q @ erho - e @ Q @ e, -2 * al * s_ * mm + al ** 2 * s_ ** 2 * c))
    A = (lambda B: B @ B.T)(rng.normal(size=(p, p))); t = 1e-6
    R = lambda tt: (lambda et: et @ Q @ et)(np.linalg.solve(H + tt * A, H @ e + tt * A @ g))
    a_, b_ = Hi @ Q @ e, e - g
    checks.setdefault('Eq. (7) derivative', []).append(abs(-(R(t) - R(-t)) / (2 * t) - 2 * np.trace(A @ (np.outer(a_, b_) + np.outer(b_, a_)) / 2)) < 1e-4 * (1 + abs(R(0))))
    # Proposition (defect sandwich) with an S-equivariant f and arbitrary fitted fhat
    d = 3; S = np.diag(rng.choice([-1., 1.], d)); X = rng.normal(size=(30, d))
    Wl = rng.normal(size=(d, d)); Wl = (Wl + S @ Wl @ S) / 2          # S W S = W  =>  f(Sx) = S f(x)
    f = lambda Z: Z @ Wl.T + np.tanh(Z @ S @ Wl.T @ S + Z @ Wl.T)       # also equivariant (odd under S)
    fh = lambda Z: f(Z) + .3 * Z ** 2 + .1 * np.sin(3 * Z)          # arbitrary fitted field
    rms = lambda M: np.sqrt(np.mean(np.sum(M ** 2, 1)))
    eq = rms(f(X @ S.T) - f(X) @ S.T)
    dh, eX, eSX = rms(fh(X @ S.T) - fh(X) @ S.T), rms(fh(X) - f(X)), rms(fh(X @ S.T) - f(X @ S.T))
    checks.setdefault('Proposition sandwich (b)', []).append(eq < 1e-10 and abs(eSX - eX) <= dh + 1e-12 and dh <= eSX + eX + 1e-12)
    gx = lambda x: x - x ** 3; Z = rng.normal(size=(30, 2)); F = np.stack([Z[:, 1], gx(Z[:, 0])], 1); Sr = np.diag([1., -1.])
    FS = np.stack([(Z @ Sr)[:, 1], gx((Z @ Sr)[:, 0])], 1)
    checks.setdefault('Proposition sandwich (c)', []).append(close(rms(FS - F @ Sr) / rms(F), 2.0))
for k, v in checks.items(): print(f'{k:32s} {sum(v)}/{len(v)}')
assert all(all(v) for v in checks.values())
