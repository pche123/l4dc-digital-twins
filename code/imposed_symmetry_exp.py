"""Direct-imposition baseline (Duffing): hard equivariant ridge vs mirrored augmentation vs duplication.
Equivariant model: f~(y)=1/2(f(y)+S f(Sy)) for diagonal involution S, fit by per-output ridge.
Usage: python code/imposed_symmetry_exp.py harness data"""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
import duffing_exp as D
from duffing_exp import *

def design_sym(L, series, S):
    """returns X (n,p) of f(y) features, XS (n,p) features at S y, Y targets"""
    X, Y = L.design(series); XS, _ = L.design([S @ y for y in series]); return X, XS, Y

def fit_eq(L, series, alpha, S):
    X, XS, Y = design_sym(L, series, S); B = []
    for k in range(Y.shape[1]):
        Z = .5 * (X + S[k, k] * XS); B.append(np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ Y[:, k]))
    return np.array(B).T

class EqModel:
    """wraps Learner so rollout uses symmetrised predictor"""
    def __init__(self, L, S): self.L, self.S = L, S
    def rollout(self, B, init, n):
        L, S = self.L, self.S; h = [init[:, i] for i in range(init.shape[1])]
        for _ in range(n):
            hs = [S @ v for v in h]
            if L.kind == 'linear':
                a = np.stack(h[-D.LAG:], 0).reshape(-1) @ B; b = np.stack(hs[-D.LAG:], 0).reshape(-1) @ B
                nxt = .5 * (a + np.diag(S) * b)
                
            else:
                a = D.cubic_features(h[-1][None])[0] @ B; b = D.cubic_features(hs[-1][None])[0] @ B
                nxt = h[-1] + .5 * (a + np.diag(S) * b)
            h.append(np.clip(np.nan_to_num(nxt, nan=1e3), -1e3, 1e3))
        return np.stack(h, 1)

def one(seed):
    raw = D.split(seed); A = np.concatenate(raw['train'], 1)
    mu = A.mean(1, keepdims=True); sd = A.std(1, ddof=1, keepdims=True); st = lambda x: (x - mu) / sd
    z = {k: ([st(w) for w in v] if isinstance(v, list) else st(v)) for k, v in raw.items()}
    src = z['train'][0]; rows = []; pos, neg = z['test'][:10], z['test'][10:]
    # symmetry acts in standardised coordinates: S commutes with diagonal standardisation only if mu~0; use raw-space map
    # standardised map: y' = (S(sd*y+mu)-mu)/sd
    def Sstd(S): 
        return lambda y: (S @ (sd * y + mu) - mu) / sd
    for name in ('linear', 'cubic'):
        L = D.Learner(name)
        def ev(M, B, S_): 
            out = {}
            for tag, ss in [('balanced', z['test']), ('positive', pos), ('negative', neg), ('headroom', z['headroom'])]:
                l, lh = D.evaluate(M, B, ss); out[f'{tag}_loss'] = l
            return out
        # baseline / duplication / mirrored augmentation (existing pipeline)
        for arm, cand in [('base', None), ('duplicate', src.copy()), ('mirror_true', st(D.S_TRUE @ raw['train'][0]))]:
            tr = z['train'] if cand is None else z['train'] + [cand]
            a = D.tune(L, tr, z['tune']); B = L.fit(tr, a); rows.append(dict(seed=seed, learner=name, arm=arm, alpha=a, **ev(L, B, None)))
        # same-unit reference arms in raw coordinates (symmetry exact there)
        for arm, cand in [('base_raw', None), ('duplicate_raw', raw['train'][0].copy()), ('mirror_raw', D.S_TRUE @ raw['train'][0])]:
            tr = raw['train'] if cand is None else raw['train'] + [cand]
            a = D.tune(L, tr, raw['tune']); B = L.fit(tr, a)
            rows.append(dict(seed=seed, learner=name, arm=arm, alpha=a, balanced_loss=D.evaluate(L, B, raw['test'])[0],
               positive_loss=D.evaluate(L, B, raw['test'][:10])[0], negative_loss=D.evaluate(L, B, raw['test'][10:])[0], headroom_loss=D.evaluate(L, B, raw['headroom'])[0]))
        # imposed (hard) equivariance; works in raw coordinates with mu=0 check -> apply to standardised series via affine map
        for arm, S in [('impose_true', D.S_TRUE), ('impose_false', D.S_FALSE)]:
            # work in raw space for exact symmetry; model fit on raw train, evaluated on raw test
            Lr = D.Learner(name); M = EqModel(Lr, S)
            def fitr(a, ser=raw['train']): return fit_eq(Lr, ser, a, S)
            def evr(B, ser): 
                tot = []; 
                for y in ser:
                    sc = float(np.std(y) + 1e-8)
                    for h in D.H:
                        for t in D.origins(y, h):
                            p = M.rollout(B, y[:, t - D.LAG:t], h)[:, D.LAG:]
                            tot.append(min(float(np.sqrt(np.mean((p - y[:, t:t + h]) ** 2)) / sc), 10.0))
                return float(np.mean(tot))
            best = sorted((evr(fitr(a), raw['tune']), a) for a in D.GRID)[0][1]; B = fitr(best)
            rows.append(dict(seed=seed, learner=name, arm=arm, alpha=best, balanced_loss=evr(B, raw['test']),
                             positive_loss=evr(B, raw['test'][:10]), negative_loss=evr(B, raw['test'][10:]), headroom_loss=evr(B, raw['headroom'])))
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=2)(delayed(one)(s) for s in range(30))
    df = pd.DataFrame([r for rr in out for r in rr]); df.to_csv(D.OUT / 'imposed_symmetry_duffing.csv', index=False); print(df.shape)
