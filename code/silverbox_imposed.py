"""Direct imposition on Silverbox: odd (equivariant) ridge vs exact mirror vs duplication, ridge learners.
usage: python code/silverbox_imposed.py <SNLS80mV.csv> data"""
import sys
import numpy as np, pandas as pd
import silverbox_exp as S
from silverbox_exp import *

def odd(F):  # keep only odd-degree terms: equivariance under (y,u)->(-y,-u) for an odd target
    keep = [0] if False else None
    return F
POWODD = [i + 1 for i, c in enumerate(S.POW) if len(c) % 2 == 1]
def poly_odd(X): return S.poly(X)[:, POWODD]
def lin_odd(X): return X
IMPOSED = {'linear ARX': lambda: Ridge(lin_odd), 'cubic NARX': lambda: Ridge(poly_odd)}

def main(csv, out):
    d = pd.read_csv(csv); u_all = d.V1.values - d.V1.values.mean(); y_all = d.V2.values - d.V2.values.mean(); res = []
    yt, ut = y_all[ARROW[0]:ARROW[1]], u_all[ARROW[0]:ARROW[1]]
    for r, (a, b) in enumerate(REAL):
        y, u = y_all[a:b], u_all[a:b]; sy, su = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(u ** 2))
        X, t = rows(y / sy, u / su); pos = X[:, 0] > 0; Xp, tp = X[pos], t[pos]
        a2, b2 = REAL[(r + 1) % len(REAL)]; Xv, tv = rows(y_all[a2:b2] / sy, u_all[a2:b2] / su); pv = Xv[:, 0] > 0; Xv, tv = Xv[pv], tv[pv]
        ys, us = yt / sy, ut / su; Xe, te = flip(Xp, tp, -1, -1)
        for lname in IMPOSED:
            arms = {'duplicate': (S.LEARNERS[lname], np.vstack([Xp, Xp]), np.concatenate([tp, tp])), 'exact': (S.LEARNERS[lname], np.vstack([Xp, Xe]), np.concatenate([tp, te])),
                    'imposed': (IMPOSED[lname], Xp, tp)}
            for arm, (make, XA, tA) in arms.items():
                m = min((make().fit(XA, tA, al) for al in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - tv) ** 2))
                row = dict(realisation=r, learner=lname, arm=arm)
                for h in H: row[f'nrmse_h{h}'] = recursive(m, ys, us, h)
                res.append(row)
        print('realisation', r, flush=True)
    pd.DataFrame(res).to_csv(Path(out) / 'silverbox_imposed.csv', index=False)
if __name__ == '__main__': main(sys.argv[1], sys.argv[2])
