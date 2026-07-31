# -*- coding: utf-8 -*-
"""Shock-vortex interaction, t = 0.35, 160 000 triangles on [0,2]x[0,1].
   Three standalone figures: density for each scheme, and their difference."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
G = np.load(os.path.join(S, "sv_grid.npz"))
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

# the acoustic field is weak, so the shading range is clipped just under the shock jump
LO, HI = 0.998, 1.29
FILL = np.linspace(LO, HI, 40)
LINES = np.linspace(1.00, 1.28, 30)

# x > 1.35 is the undisturbed post-shock state and carries nothing, so it is cropped
XC = 1.35

def style(ax):
    ax.set_aspect("equal")
    ax.set_xlim(gx[0], XC); ax.set_ylim(gy[0], gy[-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(np.arange(0, 1.36, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

for k, tag in (("s1", "tanh"), ("s2", "closed")):
    fig, ax = plt.subplots(figsize=(10.4, 8.0))
    ax.contourf(gx, gy, G[f"{k}_rho"], levels=FILL, cmap=DENS, extend="both")
    ax.contour(gx, gy, G[f"{k}_rho"], levels=LINES, colors="#141414", linewidths=0.4)
    style(ax)
    fig.savefig(os.path.join(OUT, f"Fig_sv_density_{tag}.png"))
    fig.savefig(os.path.join(OUT, f"Fig_sv_density_{tag}.pdf"))
    plt.close(fig)
    print(f"saved Fig_sv_density_{tag}")

D = np.abs(G["s1_rho"] - G["s2_rho"])
fig, ax = plt.subplots(figsize=(11.0, 8.0))
im = ax.pcolormesh(gx, gy, np.maximum(D, 1e-6), cmap="inferno_r",
                   norm=LogNorm(vmin=1e-4, vmax=1e-1), shading="auto", rasterized=True)
ax.contour(gx, gy, G["s1_rho"], levels=[1.02, 1.08, 1.16, 1.24], colors="#2E7DB5",
           linewidths=0.5, alpha=0.85)
style(ax)
cb = fig.colorbar(im, ax=ax, fraction=0.031, pad=0.02)
cb.set_label(r"$|\rho_{\mathrm{tanh}}-\rho_{\mathrm{closed}}|$", fontsize=16, labelpad=10)
cb.ax.tick_params(labelsize=14, width=1.0, length=4)
cb.outline.set_linewidth(0.9)
fig.savefig(os.path.join(OUT, "Fig_sv_diff.png"))
fig.savefig(os.path.join(OUT, "Fig_sv_diff.pdf"))
plt.close(fig)
print("saved Fig_sv_diff")
