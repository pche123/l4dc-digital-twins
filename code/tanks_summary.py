"""Summaries for the cascaded-tanks case study. usage: python code/tanks_summary.py data"""
import sys; import numpy as np, pandas as pd
d = sys.argv[1]; L = pd.read_csv(f'{d}/tanks_laws.csv'); R = pd.read_csv(f'{d}/tanks_results.csv')
print(L.drop(columns=['theta']).round(3).to_string())
for h in (1, 5, 20):
    c = f'nrmse_h{h}'; print(f'\n=== h={h}: median normalised RMSE over 2 directions x seeds')
    print(R.groupby(['learner', 'region', 'arm'])[c].median().unstack('arm').round(3).to_string())
    dup = R[R.arm == 'duplicate'].set_index(['train', 'seed', 'learner', 'region'])[c]
    G = R.set_index(['train', 'seed', 'learner', 'region']); G['gain'] = dup.reindex(G.index) - G[c]
    print('--- duplicate-adjusted gain (positive = better than duplicate), median [wins/n]')
    for (l, rg), g in G.groupby(['learner', 'region']):
        print(l, rg, {a: f"{x.gain.median():+.3f} [{(x.gain>0).sum()}/{len(x)}]" for a, x in g.groupby('arm') if a not in ('base', 'duplicate')})
