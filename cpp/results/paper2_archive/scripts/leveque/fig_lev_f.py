# -*- coding: utf-8 -*-
"""Standalone error-by-body chart. The totals are nearly equal, but the split is not:
   the closed form is more accurate on the two smooth bodies and loses only on the
   discontinuous one."""
import os
from _paths import RESULTS, CACHE_DIR
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

OUT = os.path.join(CACHE_DIR, "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11,
    "axes.labelsize": 12, "xtick.labelsize": 11.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "ytick.major.width": 0.9, "ytick.minor.width": 0.7,
    "ytick.major.size": 4.0, "ytick.minor.size": 2.2, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 220, "savefig.dpi": 900,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C1, C2 = "#14456E", "#E07B00"
BODY = ["cone", "cosine hump", "slotted\ncylinder", "total"]
# run.log, paper2_3_data/leveque/{s1,s2}, N = 200, one revolution
v1 = np.array([4.2562e-04, 1.1742e-04, 1.7346e-03, 3.1816e-03])   # iterated tanh
v2 = np.array([3.1486e-04, 8.3344e-05, 1.8113e-03, 3.2393e-03])   # closed form
rel = 100.0 * (v2 - v1) / v1

fig, ax = plt.subplots(figsize=(6.8, 4.6))
x = np.arange(len(BODY)); w = 0.34

ax.set_axisbelow(True)
ax.yaxis.grid(True, which="major", color="0.88", lw=0.6)
ax.yaxis.grid(True, which="minor", color="0.94", lw=0.4)

b1 = ax.bar(x - w / 2, v1, w, color=C1, label="tanh kernel, iterated", zorder=3)
b2 = ax.bar(x + w / 2, v2, w, color=C2, label="closed form", zorder=3)

def fmt(v):
    m, e = f"{v:.2e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"

for r, v in zip(b1, v1):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.10, fmt(v), ha="center", va="bottom",
            fontsize=7.6, color=C1, rotation=90)
for r, v in zip(b2, v2):
    ax.text(r.get_x() + r.get_width() / 2, v * 1.10, fmt(v), ha="center", va="bottom",
            fontsize=7.6, color=C2, rotation=90)

# relative change of the closed form against the iterated kernel
for xi, a, b, r in zip(x, v1, v2, rel):
    good = r < 0
    ax.annotate(f"{r:+.0f}%", xy=(xi, max(a, b) * 4.6), ha="center", va="center",
                fontsize=11, color="#1B7F4B" if good else "#B03A2E",
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec="#1B7F4B" if good else "#B03A2E", lw=0.7))

ax.set_yscale("log")
ax.set_ylim(4e-5, 3.2e-2)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(x); ax.set_xticklabels(BODY, linespacing=1.25)
ax.tick_params(axis="x", length=0)
ax.set_ylabel(r"$L_1$ error after one revolution")

ax.axvline(2.5, color="0.75", lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(2.5 - 0.06, 2.1e-2, "per body", ha="right", va="center", fontsize=9.5,
        color="0.45")
ax.text(2.5 + 0.06, 2.1e-2, "overall", ha="left", va="center", fontsize=9.5,
        color="0.45")

ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.35, borderpad=0.4)

fig.savefig(os.path.join(OUT, "Fig_lev_error.png"))
fig.savefig(os.path.join(OUT, "Fig_lev_error.pdf"))
plt.close(fig)
print("saved Fig_lev_error")
for b, a, c, r in zip(BODY, v1, v2, rel):
    print(f"  {b.replace(chr(10),' '):18s} {a:.4e} -> {c:.4e}   {r:+.1f}%")
