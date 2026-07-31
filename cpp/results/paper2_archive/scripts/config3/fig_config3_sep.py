# -*- coding: utf-8 -*-
"""Configuration 3 as three standalone figures, sized for a paper column: density for
   each scheme, and the symmetry residual of both side by side. Larger type throughout."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
G = np.load(os.path.join(S, "cfg3_grid.npz"))
g = G["g"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 16,
    "axes.labelsize": 17, "axes.titlesize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
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
FILL = np.linspace(0.13, 1.43, 40)
LINES = np.linspace(0.15, 1.40, 26)

def axes_style(ax):
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")

# ---------------- 1 and 2: density, one file per scheme ----------------
for k, tag in (("s1", "tanh"), ("s2", "closed")):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.contourf(g, g, G[f"{k}_rho"], levels=FILL, cmap=DENS, extend="both")
    ax.contour(g, g, G[f"{k}_rho"], levels=LINES, colors="#141414", linewidths=0.4)
    axes_style(ax)
    fig.savefig(os.path.join(OUT, f"Fig_cfg3_density_{tag}.png"))
    fig.savefig(os.path.join(OUT, f"Fig_cfg3_density_{tag}.pdf"))
    plt.close(fig)
    print(f"saved Fig_cfg3_density_{tag}")

# ---------------- 3: symmetry residual, both schemes in one file ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.2))
for ax, k, tt in ((axs[0], "s1", "(a) tanh kernel"), (axs[1], "s2", "(b) closed form")):
    R = np.maximum(G[f"{k}_res"], 1e-5)
    im = ax.pcolormesh(g, g, R, cmap="inferno_r", norm=LogNorm(vmin=1e-3, vmax=3e-1),
                       shading="auto", rasterized=True)
    ax.contour(g, g, G[f"{k}_rho"], levels=[0.3, 0.6, 0.9, 1.2], colors="#2E7DB5",
               linewidths=0.5, alpha=0.8)
    axes_style(ax)
    ax.set_title(tt, pad=6)
axs[1].set_ylabel("")
fig.subplots_adjust(wspace=0.14, right=0.88)
cax = fig.add_axes([0.895, axs[1].get_position().y0, 0.020,
                    axs[1].get_position().height])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$|\rho(x,y)-\rho(y,x)|$", fontsize=16, labelpad=8)
cb.ax.tick_params(labelsize=14, width=1.0, length=4)
cb.outline.set_linewidth(0.9)
fig.savefig(os.path.join(OUT, "Fig_cfg3_residual.png"))
fig.savefig(os.path.join(OUT, "Fig_cfg3_residual.pdf"))
plt.close(fig)
print("saved Fig_cfg3_residual")
