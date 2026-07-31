# -*- coding: utf-8 -*-
"""Standalone full-field panel: the two kernels overlaid on the rotated profile,
   with the cells where they actually disagree picked out."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
d = np.load(os.path.join(S, "lev_cache.npz"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11,
    "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "legend.fontsize": 10, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 220, "savefig.dpi": 900,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

pts, cells = d["pts"], d["cells"]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)
G = {k: d[k + "_g"] for k in ("s1", "s2", "exact")}

def to_nodal(cv):
    acc = np.zeros(len(pts)); cnt = np.zeros(len(pts))
    for j in range(cells.shape[1]):
        np.add.at(acc, cells[:, j], cv); np.add.at(cnt, cells[:, j], 1.0)
    return acc / np.maximum(cnt, 1)

N = {k: to_nodal(v) for k, v in G.items()}
diff = G["s2"] - G["s1"]
THR = 0.02                       # a fiftieth of the jump; below this the two agree

C1, C2 = "#14456E", "#E07B00"
FIELD = LinearSegmentedColormap.from_list(
    "field", ["#FFFFFF", "#EDF2F8", "#D6E2EE", "#BFD2E5", "#A9C2DC"])

fig, ax = plt.subplots(figsize=(6.6, 6.6))

# pale field for context; the contours carry the comparison
ax.tripcolor(tri, facecolors=G["s2"], cmap=FIELD, vmin=0, vmax=1,
             shading="flat", rasterized=True, zorder=1)

# cells where the two kernels genuinely differ
mask = np.abs(diff) <= THR
CHL = "#C2185B"
hl = np.ma.array(np.ones_like(diff), mask=mask)
ax.tripcolor(tri, facecolors=hl, cmap=ListedColormap([CHL]), vmin=0, vmax=1,
             shading="flat", rasterized=True, zorder=2)

LV = [0.05, 0.5, 0.95]
ax.tricontour(tri, N["s1"], levels=LV, colors=C1, linewidths=1.15, zorder=4)
ax.tricontour(tri, N["s2"], levels=LV, colors=C2, linewidths=1.15,
              linestyles=[(0, (3.0, 2.2))], zorder=5)

ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xticks(np.arange(0, 1.01, 0.05), minor=True)
ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

frac = 100.0 * (~mask).sum() / len(diff)
ax.legend(handles=[
    Line2D([], [], color=C1, lw=1.15, label="tanh kernel"),
    Line2D([], [], color=C2, lw=1.15, ls=(0, (3.0, 2.2)), label="closed form"),
    Patch(facecolor=CHL, label=rf"$|\Delta g| > {THR:g}$   ({frac:.2f}% of cells)"),
], loc="upper left", bbox_to_anchor=(0.015, 0.985), handlelength=2.2,
    labelspacing=0.45, borderpad=0.5)

ax.text(0.985, 0.022,
        rf"contours at $g = 0.05,\ 0.5,\ 0.95$" "\n"
        rf"$\max|\Delta g| = {np.abs(diff).max():.2f}$,  "
        rf"r.m.s. $= {np.sqrt((diff**2).mean())*1e3:.1f}\times10^{{-3}}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.2,
        color="0.25", linespacing=1.5)

fig.savefig(os.path.join(OUT, "Fig_lev_field.png"))
fig.savefig(os.path.join(OUT, "Fig_lev_field.pdf"))
plt.close(fig)
print(f"saved Fig_lev_field   cells above threshold: {(~mask).sum()} / {len(diff)} "
      f"({frac:.3f}%)")
