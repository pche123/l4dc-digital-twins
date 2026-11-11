"""Figure 1 (hero): geometry or physics proposes structure, a symbolic test accepts or abstains, a learner consumes it.

usage: python code/make_hero.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

FIG = Path(__file__).resolve().parents[1] / 'figures'
NAVY, TEAL, ORANGE, GREY, MID, DARK, PURPLE = '#26364A', '#138A86', '#D46A3A', '#69727D', '#C7CED5', '#252A30', '#7356A0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.6, 'pdf.fonttype': 42})

def rk4(f, x0, dt, n):
    out = np.empty((n, len(x0))); x = np.array(x0, float)
    for i in range(n):
        out[i] = x; k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3); x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return out
L = rk4(lambda x: np.array([10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 8 / 3 * x[2]]), [1.01, 1, 1], .01, 14000)[4000:]
m = L[:, 0] > 2; runs, s = [], None
for i, v in enumerate(m):
    if v and s is None: s = i
    if (not v or i == len(m) - 1) and s is not None:
        if i - s > 90: runs.append((s, i))
        s = None
s, e = max(runs, key=lambda r: r[1] - r[0]); seg = L[s:min(e, s + 180)]
ex, fa = seg * [-1, -1, 1], seg * [-1, 1, 1]

def arrow_on(ax, x, y, i, col):
    ax.annotate('', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1, mutation_scale=9))

from PIL import Image
STRIP = Image.open(FIG / 'physical_system_strip.png').convert('RGB')
fig = plt.figure(figsize=(6.9, 2.45))
outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=.12)
top = outer[0].subgridspec(1, 4, wspace=.04)
tiles = [((30, 346), 'physical rig', NAVY), ((359, 679), 'its digital twin', TEAL), ((700, 1006), 'Silverbox circuit', NAVY)]
for j, ((x0, x1), name, col) in enumerate(tiles):
    a = fig.add_subplot(top[j]); a.imshow(STRIP.crop((x0, 18, x1, 308))); a.axis('off'); a.set_title(name, fontsize=8.4, fontweight='bold', color=col, pad=1)
ax = fig.add_subplot(top[3]); ax.plot(L[::5, 0], L[::5, 1], color=MID, lw=.4, alpha=.45)
ax.plot(seg[:, 0], seg[:, 1], color=NAVY, lw=2); ax.plot(ex[:, 0], ex[:, 1], color=TEAL, lw=1.9); ax.plot(fa[:, 0], fa[:, 1], color=ORANGE, lw=1.6, ls='--')
arrow_on(ax, ex[:, 0], ex[:, 1], 70, TEAL); arrow_on(ax, fa[:, 0], fa[:, 1], 70, ORANGE)
ax.text(.97, .90, 'observed', transform=ax.transAxes, ha='right', va='top', color=NAVY, fontweight='bold', fontsize=7.4)
ax.text(.03, .97, 'exact $S$', transform=ax.transAxes, va='top', color=TEAL, fontweight='bold', fontsize=7.4)
ax.text(.03, .85, 'false $\\widetilde S$', transform=ax.transAxes, va='top', color=ORANGE, fontweight='bold', fontsize=7.4)
ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
ax.set_title('Lorenz system', fontsize=8.4, fontweight='bold', color=NAVY, pad=1)

# bottom row: structure -> symbolic test -> accept / abstain -> learner
ax = fig.add_subplot(outer[1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
def box(x, y, w, h, fc, ec, txt, col, fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.004,rounding_size=.025', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=col, fontweight='bold' if bold else 'normal', linespacing=1.15)
def arr(x0, y0, x1, y1, col=GREY): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=10, lw=1.2, color=col))
box(.01, .22, .20, .56, '#F4F6F8', GREY, 'candidate structure:\nsymmetry $S$ or\nwear law', DARK, 7.4)
box(.265, .18, .25, .64, '#F4F6F8', GREY, 'symbolic screen:\nscore candidate\ndefect $d$', DARK, 7.0)
box(.58, .54, .20, .38, '#E6F3F2', TEAL, 'accept: train on\nsynthetic data', TEAL, 7.6, True)
box(.58, .06, .20, .38, 'white', GREY, 'abstain: duplicate\nreal data only', GREY, 7.6, True)
arr(.215, .5, .26, .5); arr(.52, .62, .575, .73, TEAL); arr(.52, .38, .575, .25)
arr(.785, .73, .84, .56, TEAL); arr(.785, .25, .84, .44)
for x0, n in [(.86, 3), (.915, 4), (.97, 2)]:
    for yy in np.linspace(.30, .70, n): ax.add_patch(Ellipse((x0, yy), .025, .085, fc='white', ec=NAVY, lw=.9, zorder=3))
for y1 in np.linspace(.30, .70, 3):
    for y2 in np.linspace(.30, .70, 4): ax.plot([.8725, .9025], [y1, y2], color=MID, lw=.5, zorder=0)
for y1 in np.linspace(.30, .70, 4):
    for y2 in np.linspace(.30, .70, 2): ax.plot([.9275, .9575], [y1, y2], color=MID, lw=.5, zorder=0)
ax.text(.915, .12, 'learner', ha='center', fontsize=7.8, color=NAVY, fontweight='bold')
fig.savefig(FIG / 'overview.pdf', bbox_inches='tight', pad_inches=.02, dpi=300)
print('wrote hero')
