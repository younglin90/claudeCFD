# -*- coding: utf-8 -*-
"""Shock / mixing-layer interaction, t = 120, 64 000 triangles on [0,200]x[-20,20].
   Three standalone figures: density for each scheme, and their difference."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
G = np.load(os.path.join(S, "sm_grid.npz"))
gx, gy = G["gx"], G["gy"]

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

DENS = LinearSegmentedColormap.from_list(
    "dens", ["#FFFFFF", "#E4ECF4", "#C2D6E8", "#93B4D4", "#5D87B8", "#2E5A8E", "#14304F"])
LO, HI = 0.93, 3.60
FILL = np.linspace(LO, HI, 40)
LINES = np.linspace(1.0, 3.5, 26)

def style(ax):
    ax.set_aspect("equal")
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
    ax.set_xticks([0, 50, 100, 150, 200]); ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xticks(np.arange(0, 201, 10), minor=True)
    ax.set_yticks(np.arange(-20, 21, 5), minor=True)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

for k, tag in (("s1", "tanh"), ("s2", "closed")):
    fig, ax = plt.subplots(figsize=(13.0, 3.9))
    ax.contourf(gx, gy, G[f"{k}_rho"], levels=FILL, cmap=DENS, extend="both")
    ax.contour(gx, gy, G[f"{k}_rho"], levels=LINES, colors="#141414", linewidths=0.3)
    style(ax)
    fig.savefig(os.path.join(OUT, f"Fig_sm_density_{tag}.png"))
    fig.savefig(os.path.join(OUT, f"Fig_sm_density_{tag}.pdf"))
    plt.close(fig)
    print(f"saved Fig_sm_density_{tag}")

D = np.abs(G["s1_rho"] - G["s2_rho"])
fig, ax = plt.subplots(figsize=(13.6, 3.9))
im = ax.pcolormesh(gx, gy, np.maximum(D, 1e-5), cmap="inferno_r",
                   norm=LogNorm(vmin=1e-3, vmax=3e-1), shading="auto", rasterized=True)
ax.contour(gx, gy, G["s1_rho"], levels=[1.5, 2.2, 3.0], colors="#2E7DB5",
           linewidths=0.4, alpha=0.85)
style(ax)
cb = fig.colorbar(im, ax=ax, fraction=0.014, pad=0.015)
cb.set_label(r"$|\rho_{\mathrm{tanh}}-\rho_{\mathrm{closed}}|$", fontsize=16, labelpad=10)
cb.ax.tick_params(labelsize=14, width=1.0, length=4)
cb.outline.set_linewidth(0.9)
fig.savefig(os.path.join(OUT, "Fig_sm_diff.png"))
fig.savefig(os.path.join(OUT, "Fig_sm_diff.pdf"))
plt.close(fig)
print("saved Fig_sm_diff")
