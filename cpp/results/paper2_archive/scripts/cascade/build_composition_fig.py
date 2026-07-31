# -*- coding: utf-8 -*-
"""Where the solver time actually goes: absolute composition, S1 vs S2.
   Companion to the cascade (which shows ratios) - this shows why those ratios are what they are."""
import os
from _paths import RESULTS, CACHE_DIR
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C = os.path.join(RESULTS, "paper2_cascade")
OUT = os.path.join(CACHE_DIR, "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "ytick.minor.width": 0.45, "xtick.major.size": 0,
    "ytick.major.size": 3.2, "ytick.minor.size": 1.8, "ytick.right": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

d2 = pd.read_csv(C + r"\cascade_2d.csv")
d3 = pd.read_csv(C + r"\cascade_3d.csv")

# stage colours: accelerated stages get the accent, shared work stays neutral
COL = {"shift": "#B03A2E", "face": "#E08E82", "shared": "#C8C8C8",
       "bvd": "#9A9A9A", "rest": "#EDEDED"}

CASES2 = ["leveque", "shockmixing", "shockvortex", "mach3", "config3", "doublemach"]
CASES3 = ["deform", "sphere", "2cyl"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5),
                         gridspec_kw={"width_ratios": [6, 3], "wspace": 0.30})

# ---------------- 2D : normalise each case to its own S1 solver wall ----------------
ax = axes[0]
w = 0.36
for i, case in enumerate(CASES2):
    for j, sch in enumerate(("s1", "s2")):
        r = d2[(d2.case == case) & (d2.scheme == sch)].iloc[0]
        ref = float(d2[(d2.case == case) & (d2.scheme == "s1")].solver_wall_s.iloc[0])
        shift = r.geom_s / ref          # cell-shift bucket (LSQ + cell-D)
        face = r.face_s / ref
        bvd = r.bvd_sel_s / ref
        muscl = r.muscl_s / ref
        rest = max(0.0, (r.solver_wall_s - r.recon_sum_s) / ref)
        x = i + (-w / 2 if j == 0 else w / 2)
        b = 0.0
        for v, key in ((shift, "shift"), (face, "face"), (muscl, "shared"),
                       (bvd, "bvd"), (rest, "rest")):
            ax.bar(x, v, w, bottom=b, color=COL[key], edgecolor="white", linewidth=0.35, zorder=3)
            b += v
        ax.text(x, b + 0.015, "S1" if j == 0 else "S2", ha="center", fontsize=6.2, color="0.3")
ax.set_xticks(range(len(CASES2)))
ax.set_xticklabels(["leveque", "shock\nmixing", "shock\nvortex", "mach 3", "config 3", "double\nmach"],
                   linespacing=1.15)
ax.set_ylabel("solver wall time, normalised to S1")
ax.set_ylim(0, 1.28)
ax.axhline(1.0, color="0.55", lw=0.6, ls=(0, (4, 2.5)), zorder=2)
ax.set_title("(a) two dimensions", pad=4)

# ---------------- 3D ----------------
ax = axes[1]
for i, case in enumerate(CASES3):
    for j, sch in enumerate(("s1", "s2")):
        r = d3[(d3.case == case) & (d3.scheme == sch)].iloc[0]
        ref = float(d3[(d3.case == case) & (d3.scheme == "s1")].wall.iloc[0]) * 1000.0
        shift = r.celld / ref
        face = r.facebvd / ref
        lsq = r.lsq / ref
        rest = max(0.0, (r.wall * 1000.0 - r.recon_sum) / ref)
        x = i + (-w / 2 if j == 0 else w / 2)
        b = 0.0
        for v, key in ((shift, "shift"), (face, "face"), (lsq, "shared"), (rest, "rest")):
            ax.bar(x, v, w, bottom=b, color=COL[key], edgecolor="white", linewidth=0.35, zorder=3)
            b += v
        ax.text(x, b + 0.015, "S1" if j == 0 else "S2", ha="center", fontsize=6.2, color="0.3")
ax.set_xticks(range(len(CASES3)))
ax.set_xticklabels(["deform", "sphere", "two\ncylinder"], linespacing=1.15)
ax.set_ylim(0, 1.28)
ax.axhline(1.0, color="0.55", lw=0.6, ls=(0, (4, 2.5)), zorder=2)
ax.set_title("(b) three dimensions", pad=4)

fig.legend(handles=[
    Patch(facecolor=COL["shift"], label="cell shift (accelerated)"),
    Patch(facecolor=COL["face"], label="face value (accelerated)"),
    Patch(facecolor=COL["shared"], label="quadratic surface / MUSCL (shared)"),
    Patch(facecolor=COL["bvd"], label="BVD selection (shared)"),
    Patch(facecolor=COL["rest"], label="flux, time integration, rest of solver"),
], loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=3,
    handlelength=1.3, columnspacing=1.6, handletextpad=0.6)

fig.savefig(OUT + r"\Fig_composition.png")
fig.savefig(OUT + r"\Fig_composition.pdf")
plt.close(fig)
print("saved Fig_composition")

# accelerated fraction, the number that caps every speed-up
print("\naccelerated fraction of the S1 solver wall (Amdahl f):")
for case in CASES2:
    r = d2[(d2.case == case) & (d2.scheme == "s1")].iloc[0]
    f = (r.geom_s + r.face_s) / r.solver_wall_s
    print(f"  2D {case:13s} f = {f:.3f}   ceiling 1/(1-f) = {1/(1-f):5.2f}x")
for case in CASES3:
    r = d3[(d3.case == case) & (d3.scheme == "s1")].iloc[0]
    f = (r.celld + r.facebvd) / (r.wall * 1000.0)
    print(f"  3D {case:13s} f = {f:.3f}   ceiling 1/(1-f) = {1/(1-f):5.2f}x")
