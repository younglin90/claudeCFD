# -*- coding: utf-8 -*-
"""Four-level speed-up cascade.

Both dimensions are reduced identically:
  * pair S1 and S2 by repeat index,
  * form the ratio inside each pair,
  * divide by the shared-work ratio of the same pair (the machine-speed estimate),
  * report the median over pairs.
The shared bucket executes identical instructions in both schemes, so its ratio is a
direct estimate of how much faster the machine happened to be during that pair.
"""
import os
from _paths import RESULTS, CACHE_DIR
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = os.path.join(RESULTS, "paper2_cascade")
OUT = os.path.join(CACHE_DIR, "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "ytick.minor.width": 0.45,
    "ytick.major.size": 3.2, "ytick.minor.size": 1.8, "xtick.major.size": 0,
    "ytick.right": True, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C2D, C3D = "#1F4E79", "#B03A2E"
K2D, K3D = 14.4, 80.0          # isolated RDTSC kernel, from ../gauss_paper_v3

P2 = pd.read_csv(os.path.join(D, "cascade_2d_perrep.csv"))
P3 = pd.read_csv(os.path.join(D, "cascade_3d_perrep.csv"))

# 2D: accelerated = geom (cell shift) + face; shared = MUSCL + BVD selection
P2["accel"] = P2.geom + P2.face
P2["shared"] = P2.muscl + P2.bvd_sel
# 3D: accelerated = cell-D + face/BVD; shared = the o2-LSQ gradient
P3["accel"] = P3.celld + P3.facebvd
P3["shared"] = P3.lsq

def reduce_case(P, case, wallcol, kernel, dim):
    g = P[P.case == case]
    s1 = g[g.scheme == "s1"].set_index("rep")
    s2 = g[g.scheme == "s2"].set_index("rep")
    reps = sorted(set(s1.index) & set(s2.index))
    if not reps:
        return None
    rec = []
    for r in reps:
        a, b = s1.loc[r], s2.loc[r]
        ctl = a.shared / b.shared
        rec.append(dict(rep=r, ctl=ctl,
                        stage=(a.accel / b.accel) / ctl,
                        recon=(a.recon_sum / b.recon_sum) / ctl,
                        solver=(a[wallcol] / b[wallcol]) / ctl if np.isfinite(a[wallcol]) else np.nan,
                        stage_raw=a.accel / b.accel,
                        solver_raw=a[wallcol] / b[wallcol] if np.isfinite(a[wallcol]) else np.nan))
    R = pd.DataFrame(rec)
    med = R.median(numeric_only=True)
    return dict(case=case, dim=dim, npair=len(reps), kernel=kernel,
                ctl=med.ctl, ctl_lo=R.ctl.min(), ctl_hi=R.ctl.max(),
                stage=med.stage, recon=med.recon, solver=med.solver,
                solver_lo=R.solver.min(), solver_hi=R.solver.max(),
                stage_raw=med.stage_raw, solver_raw=med.solver_raw)

rows = []
for c in ["leveque", "shockmixing", "shockvortex", "mach3", "config3", "doublemach"]:
    r = reduce_case(P2, c, "wall", K2D, "2D")
    if r: rows.append(r)
for c in ["deform", "sphere", "2cyl"]:
    r = reduce_case(P3, c, "wall", K3D, "3D")
    if r: rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(D, "cascade_final.csv"), index=False)
print(R.to_string(index=False))

NAME = {"leveque": "LeVeque", "shockmixing": "shock mixing", "shockvortex": "shock vortex",
        "mach3": "Mach 3 step", "config3": "configuration 3", "doublemach": "double Mach",
        "deform": "deformation", "sphere": "spherical blast", "2cyl": "two cylinder"}

# shockmixing has no per-scheme wall line, so it stops at the reconstruction level
has_solver = R.solver.notna()

fig, ax = plt.subplots(figsize=(6.4, 4.0))
LEVELS = ["isolated\nkernel", "accelerated\nstages", "reconstruction\ntotal", "solver\ntotal"]
X = np.arange(4)
ax.axhspan(0.5, 1.0, color="0.93", lw=0, zorder=0)
ax.axhline(1.0, color="0.45", lw=0.7, zorder=2)
ax.text(1.5, 0.60, "slower than the baseline", fontsize=6.4, color="0.48", ha="center")

for _, r in R.iterrows():
    col = C2D if r.dim == "2D" else C3D
    mk = "o" if r.dim == "2D" else "s"
    y = [r.kernel, r.stage, r.recon, r.solver]
    n = 4 if np.isfinite(r.solver) else 3
    ax.plot(X[:n], y[:n], mk + "-", ms=3.4, lw=0.85, color=col, mfc=col,
            mec="white", mew=0.4, alpha=0.88, zorder=4)
    if np.isfinite(r.solver) and r.npair > 1:
        ax.plot([3, 3], [r.solver_lo, r.solver_hi], color=col, lw=2.6, alpha=0.30,
                solid_capstyle="butt", zorder=3)

lab = R[has_solver].sort_values("solver", ascending=False).reset_index(drop=True)
ly = np.log10(lab.solver.to_numpy())
ly[0] = max(ly[0], np.log10(3.6))
for i in range(1, len(ly)):
    if ly[i - 1] - ly[i] < 0.078:
        ly[i] = ly[i - 1] - 0.078
for (_, r), yl in zip(lab.iterrows(), ly):
    col = C2D if r.dim == "2D" else C3D
    ax.plot([3.04, 3.20], [r.solver, 10 ** yl], color=col, lw=0.5, alpha=0.6,
            zorder=3, clip_on=False)
    ax.text(3.24, 10 ** yl, f"{NAME[r.case]}  {r.solver:.2f}×", fontsize=6.4,
            color=col, va="center", ha="left", clip_on=False)

for xi, txt in ((0.5, "shared work\nin the same stage"), (1.5, "remaining\nreconstruction"),
                (2.5, "flux, time\nintegration")):
    ax.annotate("", xy=(xi + 0.32, 128.0), xytext=(xi - 0.32, 128.0),
                arrowprops=dict(arrowstyle="->", color="0.55", lw=0.6))
    ax.text(xi, 143.0, txt, fontsize=6.1, color="0.42", ha="center", va="bottom",
            linespacing=1.15)

ax.set_yscale("log")
ax.set_xticks(X); ax.set_xticklabels(LEVELS, linespacing=1.2)
ax.set_xlim(-0.36, 3.32); ax.set_ylim(0.5, 330)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_ylabel("speed-up of the closed form over the baseline")
ax.legend(handles=[Line2D([], [], color=C2D, marker="o", ms=3.4, lw=0.85, label="two dimensions"),
                   Line2D([], [], color=C3D, marker="s", ms=3.4, lw=0.85, label="three dimensions"),
                   Line2D([], [], color="0.55", lw=2.6, alpha=0.35, label="range over repeats")],
          loc="lower left", handlelength=1.6, labelspacing=0.28, borderpad=0.3)

fig.savefig(os.path.join(OUT, "Fig_cascade_final.png"))
fig.savefig(os.path.join(OUT, "Fig_cascade_final.pdf"))
plt.close(fig)
print("\nsaved Fig_cascade_final")
print(f"\nmachine-speed control, median per case: {R.ctl.min():.3f} - {R.ctl.max():.3f}")
print(f"solver level, corrected : {R.solver.min():.2f} - {R.solver.max():.2f}x")
print(f"solver level, raw       : {R.solver_raw.min():.2f} - {R.solver_raw.max():.2f}x")
