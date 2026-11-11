"""EMPS digital-twin illustration (AI-generated base image, figures/emps_illustration_source.jpg) with the identified friction law
typeset on the screen panel.   usage: python code/make_emps_figure.py"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open('figures/emps_illustration_source.jpg'); W, Hh = im.size
fig = plt.figure(figsize=(6.9, 6.9 * Hh / W)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.axis('off')
x0, x1, y0, y1 = 2055, 2460, 440, 592                       # blank panel on the monitor (pixels)
cx = (x0 + x1) / 2; c = '#7ff3ee'
ax.text(cx, y0 + 0.30 * (y1 - y0), r'$M\dot v = F - F_v v - F_c\,\mathrm{sign}(v)$', color=c, ha='center', va='center', fontsize=4.9)
ax.text(cx, y0 + 0.72 * (y1 - y0), r'$F_v^{+}=168,\ \ F_v^{-}=242$', color=c, ha='center', va='center', fontsize=4.9)
fig.savefig('figures/emps_illustration.pdf', dpi=300); fig.savefig('figures/emps_illustration.png', dpi=200)
