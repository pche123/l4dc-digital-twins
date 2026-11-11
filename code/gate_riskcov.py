"""Risk-coverage of the symbolic gate on the stored per-seed defects (no new simulation).
For each absolute threshold tau (ratio rule r fixed), report per setting: decisions, acceptances, wrong acceptances
(Wilson 95% intervals; seeds within a setting are the unit, settings are never pooled into one rate), and the mean
duplicate-adjusted gain per decision of using the gate (0 when it abstains), where arms exist.
usage: python code/gate_riskcov.py data   -> data/gate_riskcov.csv"""
import sys, numpy as np, pandas as pd
from pathlib import Path
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 1e9]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))
def decide(g, tau, r):
    g = g.sort_values('defect'); d1, d2 = g.defect.iloc[0], g.defect.iloc[1]
    return (g['transform'].iloc[0], bool(g['is_true'].iloc[0])) if (d1 <= tau and d1 <= r * d2) else (None, None)
ARM = {'lorenz': lambda tr: 'T' + tr, 'duffing': lambda tr: 'T' + tr}
def main(out, r=0.5):
    out = Path(out); rows = []
    sets = []
    D = pd.read_csv(out / 'verify_main_defects.csv'); D = D[D.verifier == 'dense']
    A = pd.read_csv(out / 'verify_main_arms.csv')
    sets += [('Lorenz (exact symmetry)', D[D.system == 'lorenz'], A[A.system == 'lorenz']), ('Duffing, nominal noise', D[D.system == 'duffing'], A[A.system == 'duffing'])]
    L = pd.read_csv(out / 'duf_lownoise_defects.csv'); L = L[L.verifier == 'dense']
    for nz, lab in [(0.0, 'noise-free'), (0.0025, '1/4 noise'), (0.005, '1/2 noise')]: sets.append((f'Duffing, {lab}', L[L.noise == nz], None))
    T = pd.read_csv(out / 'verify_tilt_defects.csv'); T = T[T.verifier == 'dense']
    for e in (2.0, 5.0, 10.0): sets.append((f'Lorenz, broken by {e:g} (approx.)', T[T.eps == e], None))
    C = pd.read_csv(out / 'control_lv_defects.csv'); C = C[C.verifier == 'dense']; sets.append(('Lotka-Volterra (no symmetry)', C, None))
    for name, d, arms in sets:
        groups = list(d.groupby('seed'))
        for tau in TAUS:
            acc = wrong = 0; gains = []
            for seed, g in groups:
                tr, ok = decide(g, tau, r)
                if tr is not None:
                    acc += 1; wrong += (not ok)
                if arms is not None:
                    a = arms[arms.seed == seed]; gs = []
                    for lrn, al in a.groupby('learner'):
                        dup = al[al.arm == 'duplicate'].balanced_loss.iloc[0]
                        if tr is None: gs.append(0.0)
                        else:
                            m = al[al.arm == 'T' + tr]
                            gs.append(dup - m.balanced_loss.iloc[0] if len(m) else np.nan)
                    gains.append(np.mean(gs))
            n = len(groups); lo, hi = wilson(wrong, n)
            rows.append(dict(setting=name, tau=tau, r=r, decisions=n, accepted=acc, wrong=wrong, coverage=acc / n, wrong_rate=wrong / n,
                             wrong_lo=lo, wrong_hi=hi, gain_per_decision=np.nanmean(gains) if gains else np.nan))
    # twin five-law screen (structured verifier), 12 settings x 40 seeds
    W = pd.read_csv(out / 'twin_screen_laws.csv'); WA = pd.read_csv(out / 'twin_screen_arms.csv')
    W = W.drop(columns=['defect']).rename(columns={'structured_defect': 'defect', 'law': 'transform'}); W['is_true'] = (W['transform'] == 'correct').astype(int)
    for (wmax, nz), d in W.groupby(['wmax', 'noise']):
        name = f'Twin laws, wear<={wmax:g}, noise x{nz:g}'
        for tau in TAUS:
            acc = wrong = 0; gains = []
            for (seed,), g in d.groupby(['seed']):
                tr, ok = decide(g, tau, r)
                acc += tr is not None; wrong += (tr is not None and not ok)
                a = WA[(WA.seed == seed) & (WA.wmax == wmax) & (WA.noise == nz)].set_index('arm').nrmse_h20
                gains.append(0.0 if tr is None else a['duplicate'] - a['law:' + tr])
            n = d.seed.nunique(); lo, hi = wilson(wrong, n)
            rows.append(dict(setting=name, tau=tau, r=r, decisions=n, accepted=acc, wrong=wrong, coverage=acc / n, wrong_rate=wrong / n,
                             wrong_lo=lo, wrong_hi=hi, gain_per_decision=float(np.mean(gains))))
    df = pd.DataFrame(rows); df.to_csv(out / 'gate_riskcov.csv', index=False); return df
if __name__ == '__main__':
    df = main(sys.argv[1]); pd.set_option('display.width', 220)
    print(df[df.tau.isin([0.2, 0.5, 1.0, 1e9])].pivot_table(index='setting', columns='tau', values=['accepted', 'wrong'], aggfunc='sum').to_string())
