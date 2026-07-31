# -*- coding: utf-8 -*-
"""Configuration 3: direct difference between the two schemes, |rho_S1 - rho_S2|."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
G = np.load(os.path.join(S, "cfg3_grid.npz"))
g = G["g"]
X, Y = np.meshgrid(g, g)

D = np.abs(G["s1_rho"] - G["s2_rho"])
print(f"|rho_S1 - rho_S2|: max={D.max():.4f}  mean={D.mean():.4e}  "
      f"rms={np.sqrt((D**2).mean()):.4e}")
for lo, hi, lab in ((0.30, 0.70, "central roll"), (0.35, 0.65, "core"), (0.0, 1.0, "whole")):
    m = (X >= lo) & (X <= hi) & (Y >= lo) & (Y <= hi)
    print(f"   {lab:14s} mean={D[m].mean():.4e}  max={D[m].max():.4f}  "
          f"area with >0.05: {100*(D[m] > 0.05).mean():.2f}%")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 16,
    "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 800,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.2, 6.4))
im = ax.pcolormesh(g, g, np.maximum(D, 1e-5), cmap="inferno_r",
                   norm=LogNorm(vmin=1e-3, vmax=3e-1), shading="auto", rasterized=True)
# faint density outline so the reader can place the difference against the structures
ax.contour(g, g, G["s1_rho"], levels=[0.3, 0.6, 0.9, 1.2], colors="#2E7DB5",
           linewidths=0.5, alpha=0.8)
ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticks(np.arange(0, 1.01, 0.05), minor=True)
ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"$|\rho_{\mathrm{tanh}}-\rho_{\mathrm{closed}}|$", fontsize=16, labelpad=10)
cb.ax.tick_params(labelsize=14, width=1.0, length=4)
cb.outline.set_linewidth(0.9)

fig.savefig(os.path.join(OUT, "Fig_cfg3_diff.png"))
fig.savefig(os.path.join(OUT, "Fig_cfg3_diff.pdf"))
plt.close(fig)
print("saved Fig_cfg3_diff")
