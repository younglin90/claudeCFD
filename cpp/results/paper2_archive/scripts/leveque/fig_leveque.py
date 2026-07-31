# -*- coding: utf-8 -*-
"""Solid-body rotation (LeVeque-Zalesak), one revolution, N = 200 (160 000 triangles).
   Iterated tanh kernel versus the closed form: same mesh, same 11 819 steps, recipes
   differing in one flag."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "figures")
d = np.load(os.path.join(S, "lev_cache.npz"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
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

CMAP = LinearSegmentedColormap.from_list(
    "seq", ["#FFFFFF", "#DEE8F1", "#AFC8E0", "#6E9CC6", "#2E6098", "#163454"])
C1, C2, CE = "#1F4E79", "#D97706", "#B03A2E"

fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.80], hspace=0.42, wspace=0.28)

# ---------- (a) full field ----------
ax = fig.add_subplot(gs[0, 0])
ax.tripcolor(tri, facecolors=G["s2"], cmap=CMAP, vmin=0, vmax=1, shading="flat",
             rasterized=True)
ax.tricontour(tri, N["s2"], levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="0.35",
              linewidths=0.28)
ax.tricontour(tri, N["exact"], levels=[0.5], colors=CE, linewidths=0.5)
for (x0, y0, w, h) in [(0.31, 0.56, 0.38, 0.38)]:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="0.2", lw=0.5,
                               ls=(0, (3, 2))))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_title("(a)  closed form, one revolution", pad=3)

# ---------- (b) zoom, contour lines of both kernels ----------
ax = fig.add_subplot(gs[0, 1])
ax.tricontourf(tri, N["exact"], levels=[0.5, 10.0], colors=["#EEF3F8"])
ax.tricontour(tri, N["exact"], levels=[0.5], colors=CE, linewidths=0.8)
ax.tricontour(tri, N["s1"], levels=[0.05, 0.5, 0.95], colors=C1, linewidths=0.7)
ax.tricontour(tri, N["s2"], levels=[0.05, 0.5, 0.95], colors=C2, linewidths=0.7,
              linestyles=[(0, (2.4, 1.9))])
ax.set_xlim(0.32, 0.68); ax.set_ylim(0.57, 0.93); ax.set_aspect("equal")
ax.set_xticks([0.35, 0.5, 0.65]); ax.set_yticks([0.6, 0.75, 0.9])
ax.set_title("(b)  slotted cylinder, contours overlaid", pad=3)
ax.legend(handles=[Line2D([], [], color=CE, lw=0.8, label="exact"),
                   Line2D([], [], color=C1, lw=0.7, label="tanh kernel"),
                   Line2D([], [], color=C2, lw=0.7, ls=(0, (2.4, 1.9)),
                          label="closed form")],
          loc="lower left", handlelength=1.9, labelspacing=0.18, borderpad=0.2)

# ---------- (c) difference ----------
ax = fig.add_subplot(gs[0, 2])
diff = G["s2"] - G["s1"]
lim = 0.3
im = ax.tripcolor(tri, facecolors=diff, cmap="RdBu_r",
                  norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                  shading="flat", rasterized=True)
ax.tricontour(tri, N["exact"], levels=[0.5], colors="0.3", linewidths=0.35)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_title("(c)  closed form minus tanh kernel", pad=3)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-lim, 0, lim])
cb.ax.tick_params(labelsize=6.2, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
ax.text(0.5, -0.17, r"r.m.s. $6.3\times10^{-3}$; confined to the interfaces",
        transform=ax.transAxes, ha="center", fontsize=6.3, color="0.35")

# ---------- (d), (e) line cuts ----------
def cut(ax, const, along, title, xlabel):
    t = np.linspace(0.30, 0.70, 1200) if along == "x" else np.linspace(0.05, 0.95, 1600)
    xs, ys = (t, np.full_like(t, const)) if along == "x" else (np.full_like(t, const), t)
    for k, col, lw, ls, lab in (("exact", "#F1B6AE", 2.2, "-", "exact"),
                                ("s1", C1, 0.9, "-", "tanh kernel"),
                                ("s2", C2, 0.9, (0, (2.4, 1.9)), "closed form")):
        f = mtri.LinearTriInterpolator(tri, N[k])
        ax.plot(t, f(xs, ys), color=col, lw=lw, ls=ls, label=lab)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(-0.06, 1.30)
    ax.set_xlabel(xlabel, labelpad=1); ax.set_ylabel(r"$g$", labelpad=1)
    ax.set_title(title, pad=3)

ax = fig.add_subplot(gs[1, 0])
cut(ax, 0.78, "x", r"(d)  cut at $y = 0.78$, across the slot", r"$x$")
ax.legend(loc="upper center", ncol=3, handlelength=1.7, columnspacing=0.9,
          borderpad=0.15, labelspacing=0.15, bbox_to_anchor=(0.5, 1.02))

ax = fig.add_subplot(gs[1, 1])
cut(ax, 0.5, "y", r"(e)  cut at $x = 0.5$, through the cone apex", r"$y$")

# ---------- (f) per-body error ----------
ax = fig.add_subplot(gs[1, 2])
BODY = ["cone", "hump", "slotted\ncylinder", "total"]
v1 = [4.2562e-04, 1.1742e-04, 1.7346e-03, 3.1816e-03]
v2 = [3.1486e-04, 8.3344e-05, 1.8113e-03, 3.2393e-03]
x = np.arange(4); w = 0.36
ax.bar(x - w / 2, v1, w, color=C1, label="tanh kernel", zorder=3)
ax.bar(x + w / 2, v2, w, color=C2, label="closed form", zorder=3)
for xi, a, b in zip(x, v1, v2):
    ax.text(xi, max(a, b) * 1.35, f"{100*(b-a)/a:+.0f}%", ha="center", fontsize=6.0,
            color="0.35")
ax.set_yscale("log"); ax.set_ylim(4e-5, 1.6e-2)
ax.set_xticks(x); ax.set_xticklabels(BODY, linespacing=1.1)
ax.set_ylabel(r"$L_1$ error", labelpad=1)
ax.set_title("(f)  error by body", pad=3)
ax.legend(loc="upper left", handlelength=1.2, labelspacing=0.2, borderpad=0.2)
ax.tick_params(axis="x", length=0)

fig.savefig(os.path.join(OUT, "Fig_leveque_S1S2.png"))
fig.savefig(os.path.join(OUT, "Fig_leveque_S1S2.pdf"))
plt.close(fig)
print("saved Fig_leveque_S1S2")
