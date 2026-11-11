"""Sensitivity of the symbolic gate to its two thresholds (defect <= tau and best/second-best <= r) on the stored per-seed defects.
usage: python code/gate_sweep.py data   -> data/gate_sweep.csv"""
import sys, itertools, pandas as pd
from pathlib import Path
FILES = [('Lorenz', 'verify_main_defects', 'lorenz'), ('Duffing', 'verify_main_defects', 'duffing'), ('Duffing noise study', 'duf_lownoise_defects', 'duffing'), ('Lotka-Volterra control', 'control_lv_defects', 'lv')]
def sweep(d, tau, r):
    keys = ['system', 'noise', 'eps', 'seed', 'verifier']; acc = wrong = n = 0
    for _, g in d.groupby(keys):
        g = g.sort_values('defect'); d1, d2 = g.defect.iloc[0], g.defect.iloc[1]; n += 1
        a = d1 <= tau and d1 <= r * d2; acc += a; wrong += a and not g.is_true.iloc[0]
    return n, int(acc), int(wrong)
if __name__ == '__main__':
    out = Path(sys.argv[1]); rows = []
    for name, f, sysname in FILES:
        D = pd.read_csv(out / f'{f}.csv'); D = D[(D.eps == 0) & (D.system == sysname)]
        for ver in sorted(D.verifier.unique()):
            for tau, r in itertools.product([0.3, 0.5, 0.7, 1.0], [0.3, 0.5, 0.7]):
                n, a, w = sweep(D[D.verifier == ver], tau, r); rows.append(dict(system=name, verifier=ver, tau=tau, r=r, decisions=n, accepted=a, wrong_accepts=w))
    df = pd.DataFrame(rows); df.to_csv(out / 'gate_sweep.csv', index=False)
    print(df[df.r == 0.5].pivot_table(index=['system', 'verifier'], columns='tau', values=['accepted', 'wrong_accepts'], aggfunc='sum').to_string())
    print(df.groupby('verifier')[['decisions']].first())
