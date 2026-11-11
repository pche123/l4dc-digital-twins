"""Clear three-panel Figure 5 for the main text.

The panels use the archived seed-level or aggregate CSVs;
only plot selection, layout, legends and typography are changed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
OUT = ROOT / 'figures' / 'risk_and_coverage.pdf'
C = {'teal': '#138A86', 'navy': '#26364A', 'orange': '#D46A3A',
     'purple': '#7356A0', 'gray': '#808A95', 'black': '#202934'}
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'pdf.fonttype':42,
    'font.size':8.3, 'axes.titlesize':9.0, 'axes.labelsize':8.1,
    'xtick.labelsize':7.6, 'ytick.labelsize':7.6,
    'legend.fontsize':7.2, 'axes.spines.top':False,
    'axes.spines.right':False,
})
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.70), gridspec_kw={'wspace': .53})
fig.subplots_adjust(left=.083, right=.945, top=.86, bottom=.34)
for ax in axes:
    ax.grid(axis='y', alpha=.16, lw=.5)
    ax.set_axisbelow(True)

# A: A small, explicitly identified set of representative settings.
risk = pd.read_csv(D / 'gate_riskcov.csv')
risk = risk[risk.tau < 1e8]
a = axes[0]
settings = [
  ('Lorenz (exact symmetry)', 'accepted', C['teal'], '-', 'Lorenz: correct'),
  ('Twin laws, wear<=0.3, noise x0.25', 'accepted', C['navy'], '-', 'Twin: correct law'),
  ('Duffing, nominal noise', 'wrong', C['orange'], '--', 'Duffing: wrong'),
  ('Lotka-Volterra (no symmetry)', 'wrong', C['orange'], ':', 'No symmetry: wrong')
]
for setting, numerator, color, ls, label in settings:
    v = risk[risk.setting == setting].sort_values('tau')
    a.plot(v.tau, v[numerator]/v.decisions, color=color, ls=ls, lw=1.8, label=label)
a.axvline(.5, color=C['gray'], lw=1, ls='--')
a.annotate('used', xy=(.5, .88), xytext=(.55, .92),
           color=C['gray'], fontsize=7.0, arrowprops=dict(arrowstyle='-', lw=.55, color=C['gray']))
a.set(xscale='log', ylim=(-.035, 1.06), xlabel=r'defect threshold $\tau$', ylabel='fraction of decisions')
a.set_title('(a) Gate risk', loc='left', weight='bold', pad=7)
a.legend(loc='upper center', bbox_to_anchor=(.51,-.29), ncol=2,
         frameon=False, columnspacing=.8, handlelength=2.0,
         handletextpad=.35, fontsize=6.75)

# B: each dot is a wear/noise condition; use a four-colour compact key.
arm = pd.read_csv(D / 'twin_screen_arms.csv')
law = pd.read_csv(D / 'twin_screen_laws.csv')
piv = arm.pivot_table(index=['wmax','noise','seed'], columns='arm', values='nrmse_h20')
gain = (piv['duplicate']-piv['law:correct']).groupby(['wmax','noise']).median()
acc = law[law.law == 'correct'].groupby(['wmax','noise']).accepted.mean()
color = {0.0:'#C7CED5',.25:C['teal'],1.0:C['purple'],4.0:C['black']}
b = axes[1]
for (wear,nz), val in gain.items():
    b.scatter(acc[(wear,nz)], val, s=31, color=color[nz], edgecolor='white', lw=.55, zorder=3)
b.axhline(0, color=C['gray'], lw=.85)
b.set(yscale='symlog', xlabel='fraction certifying correct law', ylabel=r'correct-law gain ($h=20$)', xlim=(-.08,1.06))
b.set_ylim(-.17, 5e4)
b.set_yticks([0,1,100,10000])
b.set_yticklabels(['0','1','100','10k'])
b.set_title('(b) Certification versus gain', loc='left', weight='bold', pad=7)
b.legend(handles=[Line2D([],[],marker='o',ls='',color=color[n],ms=5.2,label=f'noise {n:g}×')
                  for n in (0.0,.25,1.0,4.0)], loc='upper center',
         bbox_to_anchor=(.5,-.29), ncol=2, frameon=False,
         handletextpad=.15, columnspacing=.8, fontsize=6.75)

# Use a compact legend to distinguish each series without overlapping curves.
cv = pd.read_csv(D / 'coverage_twin.csv')
cv = cv[cv.regime == 'seen']
x = sorted(cv.rho.unique())
xx = np.array([.125 if a == 0 else a for a in x])
c = axes[2]
lines = [
 ('self','all', C['orange'],'-', 'self-labels'),
 ('law:correct','all', C['navy'],'-', 'law labels'),
 ('law:correct','real',C['navy'],'--','law / real noise'),
 ('real','all',C['teal'],'-','fresh real'),
]
for name, noise, col, ls, label in lines:
    g = cv[(cv.arm==name)&(cv.noise_from==noise)].groupby('rho').coverage
    med = g.median().reindex(x).to_numpy()
    p25 = g.quantile(.25).reindex(x).to_numpy()
    p75 = g.quantile(.75).reindex(x).to_numpy()
    c.plot(xx, med, color=col, ls=ls, lw=1.65, label=label)
    c.fill_between(xx, p25,p75, color=col,alpha=.10,lw=0)
k = [(1196-12)/(1196+v*598-12) for v in x]
pred = [2*norm.cdf(norm.ppf(.95)*np.sqrt(v))-1 for v in k]
c.plot(xx, pred, color=C['black'], lw=1.0, ls=':', label='closed-form')
c.axhline(.90,color=C['gray'],lw=.75,ls=':')
c.set(xscale='log', xlabel=r'synthetic exposure $\rho$', ylabel='coverage of nominal 90%', ylim=(.45,.98))
c.set_xticks(xx, ['0']+[f'{v:g}' for v in x[1:]])
c.set_title('(c) False confidence', loc='left', weight='bold', pad=7)
c.legend(loc='upper center', bbox_to_anchor=(.5,-.29), ncol=2,
         frameon=False, handlelength=1.65, handletextpad=.30,
         columnspacing=.65, fontsize=6.7)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT,dpi=300)
fig.savefig(OUT.with_suffix('.png'),dpi=250)
plt.close(fig)
print('saved',OUT)
