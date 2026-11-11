"""Figures 2, 4 (Lorenz neuro-symbolic), 5 (Duffing) and 7 (digital-twin evidence) of v2.4, from data/ only.

usage: python code/make_figures.py      (run from the package root)
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / 'data'; FIG = ROOT / 'figures'
sys.path.insert(0, str(ROOT / 'code'))
from gate_summary import gate

NAVY, TEAL, PURPLE, ORANGE, GREY, MID, DARK = '#26364A', '#138A86', '#7356A0', '#D46A3A', '#69727D', '#C7CED5', '#252A30'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.6, 'axes.titlesize': 9.2, 'axes.labelsize': 8.6,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.8, 'pdf.fonttype': 42})
W = 6.9   # inches; close to the L4DC text width, so fonts print at roughly their nominal size
rng = np.random.default_rng(0)

def ci(x):
    x = np.asarray(x, float); m = rng.choice(x, (5000, len(x))).mean(1); return np.quantile(m, [.025, .975])

def clean(ax, grid='y'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, alpha=.15, lw=.5)

def eligible(g, real='opposite_wing_real'):
    hr = g.pivot(index='seed', columns='arm', values='headroom_loss')
    e = (hr.duplicate - hr[real]) / hr.duplicate >= 0.10
    return e[e].index

# ---------------- Figure 2: support, horizon, identification quality ----------------
mc = pd.read_csv(DATA / 'model_class_results.csv'); nz = pd.read_csv(DATA / 'noise_value_sweep.csv')
fig, axs = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw={'wspace': .48, 'width_ratios': [1.0, 1.0, 1.1]})
ax = axs[0]; rows = []
for L, lab in [('linear', 'AR'), ('quadratic', 'Quad')]:
    g = mc[mc.learner == L]; el = eligible(g)
    for arm, an in [('true_symmetry', 'exact'), ('false_symmetry', 'false')]:
        o = g.pivot(index='seed', columns='arm', values='positive_h8').loc[el]; m = g.pivot(index='seed', columns='arm', values='negative_h8').loc[el]
        rows.append((f'{lab} {an}', (o.duplicate - o[arm]).mean(), (m.duplicate - m[arm]).mean(), an))
y = np.arange(len(rows)); ax.axvline(0, color=GREY, lw=.8)
for i, (lab, ob, mi, an) in enumerate(rows):
    ax.plot([ob, mi], [i, i], color=MID, lw=1.3, zorder=0)
    ax.plot(ob, i, 'o', mfc='white', mec=GREY, ms=5.5)
    ax.plot(mi, i, 'o', color=TEAL if an == 'exact' else ORANGE, ms=5.5)
ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlabel('gain vs duplicate ($h=8$)', labelpad=1)
ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec=GREY, label='observed wing'),
                   Line2D([], [], marker='o', ls='', color=GREY, label='missing wing')], frameon=False, loc='upper center',
          bbox_to_anchor=(.5, -.28), ncol=2, fontsize=7, handletextpad=.1, columnspacing=.6)
ax.set_title('(a) Where the gain lands', loc='left', fontweight='bold'); clean(ax, 'x'); ax.spines['left'].set_visible(False)
ax = axs[1]; H = [1, 4, 8]; ax.axhline(0, color=GREY, lw=.8)
for L, ls in [('linear', '-'), ('quadratic', '--')]:
    g = mc[mc.learner == L]; el = eligible(g)
    for arm, col in [('true_symmetry', TEAL), ('false_symmetry', ORANGE)]:
        v = [(g.pivot(index='seed', columns='arm', values=f'balanced_h{h}').loc[el].pipe(lambda w: w.duplicate - w[arm])).mean() for h in H]
        ax.plot(H, v, 'o' + ls, color=col, lw=1.8, ms=4)
ax.set_xticks(H); ax.set_xlabel('recursive horizon $h$'); ax.set_ylabel('balanced gain')
ax.legend(handles=[Line2D([], [], color=TEAL, lw=1.8, label='exact'), Line2D([], [], color=ORANGE, lw=1.8, label='false'),
                   Line2D([], [], color=GREY, lw=1.5, label='AR(4)'), Line2D([], [], color=GREY, lw=1.5, ls='--', label='quadratic')],
          frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.28), fontsize=7, columnspacing=.8, handlelength=1.6)
ax.set_title('(b) Recursion', loc='left', fontweight='bold'); clean(ax)
ax = axs[2]; d = nz[nz.system == 'Duffing']; mult = sorted(d.noise_multiplier.unique()); xs = np.arange(len(mult))
for arm, col, lab in [('duplicate', GREY, 'duplicate'), ('false', ORANGE, 'false reflection'), ('exact', TEAL, 'exact symmetry')]:
    q = d[d.arm == arm].groupby('noise_multiplier').err_balanced.quantile([.25, .5, .75]).unstack()
    lo = np.maximum(q[.25].to_numpy(), 1e-5)
    ax.plot(xs, np.maximum(q[.5].to_numpy(), 1e-6), 'o-', color=col, lw=1.8, ms=4, label=lab)
    ax.fill_between(xs, np.maximum(q[.25].to_numpy(), 1e-6), np.maximum(q[.75].to_numpy(), 1e-6), color=col, alpha=.15, lw=0)
ax.set_yscale('log'); ax.set_xticks(xs, [f'{m:g}×' for m in mult]); ax.set_xlabel('Duffing observation noise')
ax.set_ylabel('test error (median)')
div = d[d.arm == 'duplicate'].groupby('noise_multiplier').err_balanced.apply(lambda x: (x > 1).mean())
med = d[d.arm == 'duplicate'].groupby('noise_multiplier').err_balanced.median()
for x, m in zip(xs, mult):
    if div[m] > 0: ax.text(x - .08, med[m] * 4, f'{100*div[m]:.0f}%', ha='right' if x < xs[-1] else 'center', fontsize=6.8, color=GREY)
ax.set_ylim(1e-9, 1e5); ax.legend(frameon=False, loc='lower right', fontsize=6.5, borderaxespad=.1)
ax.set_title('(c) Noise breaks the baseline', loc='left', fontweight='bold'); clean(ax)
fig.savefig(FIG / 'synthetic_comparisons.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)

# ---------------- Figure 4: Lorenz gate -> neural payoff -> verification/value tension ----------------
G = gate(pd.read_csv(DATA / 'verify_main_defects.csv')); G = G[(G.verifier == 'dense')]
GL = G[G.system == 'lorenz'].set_index('seed')
M = pd.read_csv(DATA / 'mlp_lorenz_fresh.csv')
fig, axs = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw={'wspace': .45, 'width_ratios': [1.0, 1.05, 1.05]})
ax = axs[0]; acc = GL.acc == 1; wrongbest = GL.best_true == 0
ax.fill_between([0, .5], 0, .5, color=TEAL, alpha=.09, lw=0)
ax.axvline(.5, ls='--', lw=.8, color=GREY); ax.axhline(.5, ls='--', lw=.8, color=GREY)
d1 = GL.d1.clip(upper=2.4); r = (GL.d1 / GL.d2).clip(upper=1.2)
ax.scatter(d1[~acc & ~wrongbest], r[~acc & ~wrongbest], s=24, facecolors='white', edgecolors=GREY, lw=.9, label='abstain')
ax.scatter(d1[wrongbest], r[wrongbest], s=24, marker='x', color=ORANGE, lw=1.1, label='abstain (wrong best)')
ax.scatter(d1[acc], r[acc], s=26, color=TEAL, edgecolor='white', lw=.4, label='accept exact')
ax.set_xscale('log'); ax.set_xlim(.06, 3); ax.set_ylim(0, 1.25)
ax.set_xlabel('best defect $d_{(1)}$'); ax.set_ylabel('$d_{(1)}/d_{(2)}$')
ax.legend(frameon=False, loc='upper left', fontsize=6.8, handletextpad=.2, borderaxespad=.1)
ax.set_title('(a) Symbolic gate', loc='left', fontweight='bold'); clean(ax, None)
ax = axs[1]; H = [1, 4, 8]; off = {'exact': -.22, 'gated': 0, 'false': .22}
for nm, col in [('exact', TEAL), ('gated', NAVY), ('false', ORANGE)]:
    for k, h in enumerate(H):
        w = M.pivot(index='seed', columns='arm', values=f'balanced_h{h}')
        a = GL.acc.reindex(w.index) == 1
        c = {'exact': w.true, 'false': w.false, 'gated': w.true.where(a, w.duplicate)}[nm]
        g = (w.duplicate - c).to_numpy(); lo, hi = ci(g)
        ax.errorbar(k + off[nm], g.mean(), yerr=[[g.mean() - lo], [hi - g.mean()]], fmt='o', color=col, ms=4.5, capsize=2, lw=1.2,
                    label=nm if k == 0 else None)
ax.axhline(0, color=GREY, lw=.8); ax.set_xticks(range(3), [f'$h={h}$' for h in H]); ax.set_ylabel('MLP gain vs duplicate')
ax.set_ylim(-.17, .46); ax.legend(frameon=False, loc='upper right', fontsize=7, handletextpad=.1)
ax.set_title('(b) MLP, recursive', loc='left', fontweight='bold'); clean(ax)
ax = axs[2]
A = pd.read_csv(DATA / 'verify_main_arms.csv'); ar = A[(A.system == 'lorenz') & (A.learner == 'linear')]
wa = ar.pivot(index='seed', columns='arm', values='balanced_loss'); ga = wa.duplicate - wa['T(-x,-y,+z)']
wm = M.pivot(index='seed', columns='arm', values='balanced_loss'); gm = wm.duplicate - wm.true
for gg, mk, lab in [(ga, 's', 'linear AR'), (gm, 'o', 'MLP')]:
    td = GL.true_d.reindex(gg.index); a = GL.acc.reindex(gg.index) == 1
    rho = spearmanr(td, gg).correlation
    ax.scatter(td[a], gg[a], marker=mk, s=20, color=TEAL, edgecolor='white', lw=.3)
    ax.scatter(td[~a], gg[~a], marker=mk, s=20, facecolors='white', edgecolors=GREY, lw=.8)
    ax.scatter([], [], marker=mk, s=20, color=DARK, label=f'{lab} ($\\rho={rho:.2f}$)')
ax.set_xscale('log'); ax.axhline(0, color=GREY, lw=.8); ax.axvline(.5, ls='--', lw=.8, color=GREY)
ax.set_xlabel('defect of the exact symmetry'); ax.set_ylabel('exact-symmetry gain')
ax.set_ylim(-.17, 1.3); ax.legend(frameon=False, loc='upper left', fontsize=6.8, handletextpad=.1)
ax.set_title('(c) Certified ≠ needed', loc='left', fontweight='bold'); clean(ax)
fig.savefig(FIG / 'symmetry_verification.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)

# ---------------- Figure 5: Duffing geometry + gate vs identifiability ----------------
def rk4(f, x0, dt, n):
    out = np.empty((n, len(x0))); x = np.array(x0, float)
    for i in range(n):
        out[i] = x; k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3); x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return out
def arrow(ax, x, y, i, col):
    ax.annotate('', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1, mutation_scale=8))
D = rk4(lambda s: np.array([s[1], s[0] - s[0] ** 3]), [1.16, .10], .04, 900); Dex = D * [-1, -1]; Dfa = D * [-1, 1]
xx = np.linspace(-1.65, 1.65, 400); V = -.5 * xx ** 2 + .25 * xx ** 4
GD = gate(pd.concat([pd.read_csv(DATA / 'duf_lownoise_defects.csv'), pd.read_csv(DATA / 'verify_main_defects.csv')]))
GD = GD[(GD.system == 'duffing') & (GD.verifier == 'dense')]
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={'wspace': .42, 'width_ratios': [.9, 1.1, 1.15]})
ax = axs[0]; ax.plot(xx, V, color=NAVY, lw=1.6); ax.axvspan(.25, 1.55, color=NAVY, alpha=.07)
ax.text(1.0, .47, 'observed', ha='center', fontsize=7.4, color=NAVY, fontweight='bold'); ax.text(-1.0, .47, 'unseen', ha='center', fontsize=7.4, color=GREY)
ax.set_ylim(-.3, .56); ax.set_xlabel('$x$'); ax.set_ylabel('potential $V(x)$'); ax.set_title('(a) One observed well', loc='left', fontweight='bold'); clean(ax, None)
ax = axs[1]; ax.plot(D[:, 0], D[:, 1], color=NAVY, lw=1.5, label='observed')
ax.plot(Dex[:, 0], Dex[:, 1], color=TEAL, lw=1.8, label='exact $(-x,-v)$'); ax.plot(Dfa[:, 0], Dfa[:, 1], color=ORANGE, lw=1.4, ls='--', label='false $(-x,v)$')
for i in (100, 360, 650): arrow(ax, Dex[:, 0], Dex[:, 1], i, TEAL); arrow(ax, Dfa[:, 0], Dfa[:, 1], i, ORANGE)
ax.set_xlabel('$x$'); ax.set_ylabel('$v$'); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-.35, .35); ax.legend(frameon=False, loc='upper center', fontsize=6.6, ncol=1, handlelength=1.4, bbox_to_anchor=(.5, -.25))
ax.set_title('(b) Opposite time', loc='left', fontweight='bold'); clean(ax, None)
ax = axs[2]; ns = sorted(GD.noise.unique()); xs = np.arange(len(ns))
acc = [GD[GD.noise == n].acc.sum() for n in ns]; tot = [len(GD[GD.noise == n]) for n in ns]
rw = [(GD[GD.noise == n].racc * (1 - GD[GD.noise == n].best_true)).sum() for n in ns]
ax.bar(xs - .18, acc, .34, color=TEAL, label='gate accepts exact')
ax.bar(xs + .18, rw, .34, color=ORANGE, label='ratio-only: wrong accept')
for x, a, t in zip(xs, acc, tot): ax.text(x - .18, a + .8, f'{a}', ha='center', fontsize=7, color=TEAL, fontweight='bold')
ax.set_xticks(xs, ['0', '¼', '½', '1']); ax.set_xlabel('noise (× nominal)'); ax.set_ylabel('seeds (of 30)'); ax.set_ylim(0, 34)
ax.set_ylim(0, 44); ax.legend(frameon=False, loc='upper left', fontsize=6.6, ncol=1, borderaxespad=.1)
ax.set_title('(c) Gate vs identifiability', loc='left', fontweight='bold'); clean(ax)
fig.savefig(FIG / 'duffing_dynamics.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)

# ---------------- Figure 7: digital-twin evidence + degradation-law screen ----------------
sl = pd.read_csv(DATA / 'digital_twin_self_label.csv'); gn = pd.read_csv(DATA / 'digital_twin_gains.csv')
TL = pd.read_csv(DATA / 'twin_screen_laws.csv')
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={'wspace': .52, 'width_ratios': [.9, 1, 1.15]})
ax = axs[0]; j = np.random.default_rng(7)
ax.scatter(j.normal(0, .05, len(sl)), sl.self_update_rel, s=12, color=GREY, alpha=.7)
ax.scatter(1 + j.normal(0, .05, len(sl)), sl.phys_update_rel, s=12, color=TEAL, alpha=.7)
ax.set_yscale('log'); ax.set_xticks([0, 1], ['self-\nlabels', 'degradation-\nlaw labels']); ax.set_xlim(-.5, 1.5)
ax.set_ylabel('relative parameter update'); ax.set_title('(a) Same states', loc='left', fontweight='bold'); clean(ax)
ax = axs[1]; H = [1, 5, 20]
p = gn.pivot(index='seed', columns='arm', values=[f'gain_h{h}' for h in H])
for k, h in enumerate(H):
    dd = (p[(f'gain_h{h}', 'correct_physics')] - p[(f'gain_h{h}', 'wrong_physics')]).to_numpy(); lo, hi = ci(dd)
    ax.errorbar(h, dd.mean(), yerr=[[dd.mean() - lo], [hi - dd.mean()]], fmt='o', color=TEAL, ms=4.5, capsize=2)
    ax.text(h, hi + .01, f'{(dd > 0).sum()}/40', ha='center', fontsize=7, color=TEAL, fontweight='bold')
ax.plot(H, [(p[(f'gain_h{h}', 'correct_physics')] - p[(f'gain_h{h}', 'wrong_physics')]).mean() for h in H], color=TEAL, lw=1.2)
ax.axhline(0, color=GREY, lw=.8); ax.set_xticks(H); ax.set_xlabel('recursive horizon $h$'); ax.set_ylabel('correct $-$ wrong law gain'); ax.set_ylim(-.01, .19)
ax.set_title('(b) Correct law pays', loc='left', fontweight='bold'); clean(ax)
ax = axs[2]; ns = sorted(TL.noise.unique()); xs = np.arange(len(ns))
c = TL[TL.law == 'correct']
for wm, ls in [(0.15, '-'), (0.30, '--')]:
    cc = c[c.wmax == wm].groupby('noise')
    ax.plot(xs, cc.binary.sum().reindex(ns), 'o' + ls, color=TEAL, lw=1.7, ms=4, )
    ax.plot(xs, cc.accepted_learner.sum().reindex(ns), 's' + ls, color=GREY, lw=1.2, ms=3.5, )
ax.set_xticks(xs, ['0', '¼', '1', '4']); ax.set_xlabel('sensor noise (× nominal)'); ax.set_ylabel('seeds certifying correct law'); ax.set_ylim(-2, 46)
ax.legend(handles=[Line2D([], [], color=TEAL, marker='o', ms=3.5, label='structured fit'), Line2D([], [], color=GREY, marker='s', ms=3.5, label='black-box learner'),
                   Line2D([], [], color=DARK, lw=1.2, label='seen $w\\leq0.15$'), Line2D([], [], color=DARK, lw=1.2, ls='--', label='seen $w\\leq0.30$')],
          frameon=False, fontsize=6.6, loc='upper center', bbox_to_anchor=(.45, -.3), ncol=2, handlelength=1.8, columnspacing=.8)
ax.set_title('(c) Certify before use', loc='left', fontweight='bold'); clean(ax)
fig.savefig(FIG / 'twin_experiments.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)
# ---------------- Figure: dose-response (Theorem 4) ----------------
E = pd.read_csv(DATA / 'twin_exposure.csv')
cells = [((0.15, 1.0), '(a) Learner far from truth'), ((0.30, 0.25), '(b) Learner close'), ((0.30, 0.0), '(c) Learner already right')]
fig, axs = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw={'wspace': .30})
for ax, ((wm, nz), title) in zip(axs, cells):
    g = E[(E.wmax == wm) & (E.noise == nz)]
    base = g[g.rho == 0].groupby('seed').onestep_risk.first()
    rhos = sorted(r for r in g.rho.unique() if r > 0)
    for law, col, ls, lab in [('correct', TEAL, '-', 'correct law'), ('damping-only', ORANGE, '--', 'damping-only'), ('none', PURPLE, ':', 'no degradation')]:
        gl = g[g.law == law]
        rel = np.array([np.quantile(gl[gl.rho == r].set_index('seed').onestep_risk.reindex(base.index) / base, [.25, .5, .75]) for r in rhos])
        ax.plot(rhos, rel[:, 1], ls, color=col, lw=1.8, label=lab); ax.fill_between(rhos, rel[:, 0], rel[:, 2], color=col, alpha=.12, lw=0)
    ax.axhline(1, color=GREY, lw=.9); ax.text(rhos[0]*1.05, 1.18, 'no synthetic data', fontsize=7.0, color=GREY, va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'synthetic exposure $\rho$')
    ax.set_title(title, loc='left', fontweight='bold', fontsize=8.8); clean(ax)
axs[0].set_ylabel('deployment risk / base')
axs[0].legend(frameon=False, fontsize=7.2, loc='lower left', handlelength=1.9)
fig.savefig(FIG / 'exposure_study.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)
print('wrote v2.4 figures')
