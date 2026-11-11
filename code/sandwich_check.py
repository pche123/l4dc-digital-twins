"""Check the certification-value sandwich on the fresh seeds: for a true symmetry S,
|e(SX) - e(X)| <= d(S) <= e(SX) + e(X), where e(.) is the RMS error of the learned field (relative to RMS f_hat on X).

usage: python code/sandwich_check.py harness data      (writes data/sandwich_check.csv)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
PH3, OUT = sys.argv[1], Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
sys.argv = ['x', PH3, str(OUT / '_tmp'), 'none']
src = (HERE / 'verify_exp.py').read_text().split("if __name__")[0]
g = {'__file__': str(HERE / 'verify_exp.py'), '__name__': 'verify_exp'}; exec(src, g)

def lorenz(X): return np.stack([10 * (X[:, 1] - X[:, 0]), X[:, 0] * (28 - X[:, 2]) - X[:, 1], X[:, 0] * X[:, 1] - 8 / 3 * X[:, 2]], 1)
def duffing(X): return np.stack([X[:, 1], X[:, 0] - X[:, 0] ** 3], 1)

rows = []
for system, seeds, noise, length in [('lorenz', range(1000, 1030), 0.015, 36), ('duffing', range(1000, 1030), 0.01, 40), ('duffing', range(1000, 1030), 0.0, 40)]:
    for seed in seeds:
        if system == 'lorenz':
            raw = g['build_lorenz_split'](seed, length, 5, 5, 10, noise, 0.95); mod, feat, dt, f, S = g['mc'], g['mc'].quad_features, 0.01, lorenz, np.diag([-1., -1., 1.])
            L = mod.Learner('quadratic')
        else:
            raw = g['duf_split'](seed, length, noise, 0.0); mod, feat, dt, f, S = g['du'], g['du'].cubic_features, g['du'].DT, duffing, np.diag([-1., -1.])
            L = mod.Learner('cubic')
        mu, sd = g['fit_standardizer'](raw['train']); z = [(w - mu) / sd for w in raw['train']]
        zt = [(w - mu) / sd for w in raw['tune']]
        B = L.fit(z, mod.tune(L, z, zt))
        fh = lambda Y: (feat((Y - mu[:, 0]) / sd[:, 0]) @ B) * sd[:, 0] / dt
        X = np.concatenate(raw['train'], 1).T
        scale = np.sqrt(np.mean(fh(X) ** 2)); rms = lambda A: float(np.sqrt(np.mean(A ** 2)) / scale)
        d = rms(fh(X @ S.T) - fh(X) @ S.T)
        rows.append(dict(system=system, noise=noise, seed=seed, defect=d, err_X=rms(fh(X) - f(X)), err_SX=rms(fh(X @ S.T) - f(X @ S.T))))
df = pd.DataFrame(rows)
df['lower_ok'] = (df.err_SX - df.err_X).abs() <= df.defect + 1e-9
df['upper_ok'] = df.defect <= df.err_SX + df.err_X + 1e-9
df.to_csv(OUT / 'sandwich_check.csv', index=False)
print(df.groupby(['system', 'noise'])[['defect', 'err_X', 'err_SX']].median().round(3))
print('sandwich holds in', int((df.lower_ok & df.upper_ok).sum()), '/', len(df))
