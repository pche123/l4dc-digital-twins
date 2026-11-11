"""Real-data case study on the EMPS benchmark (Janot, Gautier and Brunot): a prismatic drive with ball screw, force F in,
motor position out, 1 kHz, two 25 s records (estimation and pulse-validation), closed loop.
Physics:  M a = F - Fv v - Fc sign(v) - offset, but the real friction is asymmetric (screw fatigue): Fv is about 44% larger
for negative velocity.  So the textbook mirror (v, F) -> (-v, -F) is approximately, not exactly, a symmetry.

Question: from positive-velocity data only, does the gate certify the mirror, and what do mirrored rows buy a learner on the
negative-velocity half, compared with genuine negative-velocity data?

Processing follows the benchmark scripts: 4th-order Butterworth at 100 Hz (zero phase), central differences, then 10x
decimation to 100 Hz.  Learners predict the velocity increment v_{k+1}-v_k from (v_k, v_{k-1}, F_k, F_{k-1}).
Replicates: train on the positive rows of one half of one record, tune on the positive rows of the other record,
test on the other record.   usage: python code/emps_exp.py <folder with DATA_EMPS*.mat> data
"""
import sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import butter, filtfilt, decimate
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')

ALPHAS = [1e-6, 1e-4, 1e-2, 1, 1e2]
H = (1, 10, 50)                       # 0.01 s, 0.1 s, 0.5 s at 100 Hz
CANDIDATES = {'exact (-v,-F)': (-1, -1, -1), 'false (-v,+F)': (-1, 1, -1), 'input-only (+v,-F)': (1, -1, 1)}  # (v sign, F sign, output sign)

def load(path):
    d = sio.loadmat(path); q = d['qm'].ravel(); F = d['gtau'].item() * d['vir'].ravel()
    b, a = butter(4, 100 / 500); qf = filtfilt(b, a, q)
    v = np.r_[qf[1] - qf[0], (qf[2:] - qf[:-2]) / 2, qf[-1] - qf[-2]] / 1e-3
    v, F = v[50:-50], F[50:-50]
    return decimate(v, 10, ftype='fir', zero_phase=True), decimate(F, 10, ftype='fir', zero_phase=True)

def rows(v, F):
    t = np.arange(1, len(v) - 1)
    return np.stack([v[t], v[t - 1], F[t], F[t - 1]], 1), v[t + 1] - v[t], t

def mirror(X, y, sv, sF, so):
    return X * np.array([sv, sv, sF, sF]), so * y

POW = [c for d in (1, 2, 3) for c in itertools.combinations_with_replacement(range(4), d)]
class Scaled:
    """fixed, uncentred scaling (keeps odd features odd) shared by real and synthetic rows"""
    def __init__(self, X, y): self.sx = np.sqrt(np.mean(X ** 2, 0)); self.sy = np.sqrt(np.mean(y ** 2))
class Poly(Scaled):
    def feats(self, X): Z = X / self.sx; return np.hstack([np.ones((len(Z), 1))] + [np.prod(Z[:, list(c)], 1, keepdims=True) for c in POW])
    def fit(self, X, y, a): Fm = self.feats(X); self.B = np.linalg.solve(Fm.T @ Fm + a * np.eye(Fm.shape[1]), Fm.T @ (y / self.sy)); return self
    def predict(self, X): return self.feats(X) @ self.B * self.sy
class Linear(Poly):
    def feats(self, X): Z = X / self.sx; return np.hstack([Z, np.ones((len(Z), 1))])
class Physics(Poly):
    """the textbook law as a library: F_k, v_k, sign(v_k), constant (Coulomb and offset are collinear on one side)"""
    def feats(self, X): Z = X / self.sx; return np.stack([Z[:, 2], Z[:, 0], np.sign(Z[:, 0]), np.ones(len(Z))], 1)
class MLP(Scaled):
    def __init__(self, X, y, seed): super().__init__(X, y); self.seed = seed
    def fit(self, X, y, a):
        self.m = MLPRegressor(hidden_layer_sizes=(32, 32), activation='tanh', alpha=a, max_iter=800, early_stopping=True,
                              random_state=self.seed, n_iter_no_change=30).fit(X / self.sx, y / self.sy); return self
    def predict(self, X): return self.m.predict(X / self.sx) * self.sy

def defect(model, X, sv, sF, so):
    f = model.predict(X); fs = model.predict(X * np.array([sv, sv, sF, sF]))
    return float(np.sqrt(np.mean((fs - so * f) ** 2)) / np.sqrt(np.mean(f ** 2)))

def rollout_err(model, v, F, h, origins, sd):
    vk, vkm = v[origins].copy(), v[origins - 1].copy(); err = np.zeros(len(origins))
    for k in range(h):
        t = origins + k; X = np.stack([vk, vkm, F[t], F[t - 1]], 1)
        vn = np.clip(vk + model.predict(X), -1., 1.); err += (vn - v[t + 1]) ** 2; vkm, vk = vk, vn
    return float(np.sqrt(np.mean(err / h)) / sd)

def main(folder, out):
    rec = {'est': load(Path(folder) / 'DATA_EMPS.mat'), 'val': load(Path(folder) / 'DATA_EMPS_PULSES.mat')}
    gate_rows, res = [], []
    for tr, te in [('est', 'val'), ('val', 'est')]:
        (vt, Ft), (ve, Fe) = rec[tr], rec[te]; sd = ve.std()
        Xa, ya, ta = rows(vt, Ft); Xe, ye, te_idx = rows(ve, Fe)
        posv = (Xe[:, 0] > 0) & (Xe[:, 1] > 0); Xv, yv = Xe[posv], ye[posv]               # one-sided tuning rows
        for half in (0, 1):
            seg = (ta < len(vt) // 2) if half == 0 else (ta >= len(vt) // 2)
            pos = seg & (Xa[:, 0] > 0) & (Xa[:, 1] > 0); neg = seg & (Xa[:, 0] < 0) & (Xa[:, 1] < 0)
            Xl, yl = Xa[pos], ya[pos]; rep = f'{tr}-{"first" if half == 0 else "second"} half'
            # ---- gate: cubic NARX verifier (as on Silverbox) and the textbook physics library, positive rows only
            for vname, V in {'cubic NARX': Poly(Xl, yl), 'physics library': Physics(Xl, yl)}.items():
                m = min((V.__class__(Xl, yl).fit(Xl, yl, a) for a in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - yv) ** 2))
                d = {c: defect(m, Xl, *s) for c, s in CANDIDATES.items()}; o = sorted(d, key=d.get)
                acc = o[0] if d[o[0]] <= .5 and d[o[0]] <= .5 * d[o[1]] else 'abstain'
                fx = np.sqrt(np.mean(m.predict(Xl) ** 2))
                e_X = np.sqrt(np.mean((m.predict(Xl) - yl) ** 2)) / fx
                e_SX = np.sqrt(np.mean((m.predict(Xa[neg]) - ya[neg]) ** 2)) / fx            # error on the real unseen half
                for c in CANDIDATES: gate_rows.append(dict(replicate=rep, verifier=vname, candidate=c, defect=d[c], accepted=int(c == acc), e_X=e_X, e_SX=e_SX))
                print(rep, vname, {k: round(x, 3) for k, x in d.items()}, '->', acc, 'e_X %.3f e_SX %.3f' % (e_X, e_SX), flush=True)
            # 'true' mirror defect: asymmetric-friction law fitted on BOTH halves of this record, defect evaluated on the seen rows
            Z = np.c_[Xa[:, 2], Xa[:, 0] * (Xa[:, 0] > 0), (Xa[:, 0] > 0), Xa[:, 0] * (Xa[:, 0] < 0), -(Xa[:, 0] < 0).astype(float), np.ones(len(Xa))]
            th = np.linalg.lstsq(Z[seg], ya[seg], rcond=None)[0]
            fz = lambda X: np.c_[X[:, 2], X[:, 0] * (X[:, 0] > 0), (X[:, 0] > 0), X[:, 0] * (X[:, 0] < 0), -(X[:, 0] < 0).astype(float), np.ones(len(X))] @ th
            dtrue = float(np.sqrt(np.mean((fz(-Xl) + fz(Xl)) ** 2)) / np.sqrt(np.mean(fz(Xl) ** 2)))
            gate_rows.append(dict(replicate=rep, verifier='asymmetric law, both halves (reference)', candidate='exact (-v,-F)', defect=dtrue, accepted=-1, e_X=np.nan, e_SX=np.nan))
            print('  reference true defect of the mirror %.3f' % dtrue, flush=True)
            arms = {'base': None, 'duplicate': (Xl, yl), 'genuine negative half': (Xa[neg], ya[neg])}
            for c, s in CANDIDATES.items(): arms[c] = mirror(Xl, yl, *s)
            org = te_idx[(Xe[:, 0] < 0) & (Xe[:, 1] < 0)]; org = org[(org >= 1) & (org < len(ve) - max(H) - 1)]
            orp = te_idx[posv]; orp = orp[(orp >= 1) & (orp < len(ve) - max(H) - 1)]
            makers = {'linear ARX': lambda s: Linear(Xl, yl), 'cubic NARX': lambda s: Poly(Xl, yl), 'physics library': lambda s: Physics(Xl, yl), 'MLP': lambda s: MLP(Xl, yl, s)}
            for lname, make in makers.items():
                for seed in (range(5) if lname == 'MLP' else [0]):
                    for arm, extra in arms.items():
                        XA, yA = (Xl, yl) if extra is None else (np.vstack([Xl, extra[0]]), np.concatenate([yl, extra[1]]))
                        m = min((make(seed).fit(XA, yA, a) for a in ALPHAS), key=lambda m: np.mean((m.predict(Xv) - yv) ** 2))
                        for region, o_ in {'negative (unseen)': org, 'positive (seen)': orp}.items():
                            row = dict(replicate=rep, learner=lname, seed=seed, arm=arm, region=region, n=len(o_))
                            for h in H: row[f'nrmse_h{h}'] = rollout_err(m, ve, Fe, h, o_, sd)
                            res.append(row)
            print('  done', rep, flush=True)
    pd.DataFrame(gate_rows).to_csv(Path(out) / 'emps_gate.csv', index=False); pd.DataFrame(res).to_csv(Path(out) / 'emps_results.csv', index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
