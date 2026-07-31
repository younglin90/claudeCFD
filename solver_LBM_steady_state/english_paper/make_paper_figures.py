#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate top-tier-journal-quality figures for the MSA-LBM / AP-Schur paper.

Outputs (english_paper/figures/):
  fig1_mechanism.(png|pdf)        schematic of the moment-Schur correction
  fig2_convergence_main.(png|pdf) representative residual-vs-walltime
  fig3_grid_1x.(png|pdf)          9-case convergence grid (1x)
  fig4_speedup.(png|pdf)          per-case wall + LBE speedup bars
  fig5_accuracy.(png|pdf)         grid-convergence / accuracy
  fig6_centerline.(png|pdf)       cavity centerline vs Ghia 1982
  fig7_fields.(png|pdf)           velocity-magnitude + vorticity contours
  fig8_memory.(png|pdf)           measured peak RSS vs dense-Jacobian
  figA1_grid_2x / figA2_grid_3x   appendix convergence grids
  figA3_determinism / figA4_acceptance
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "papers_data_ap_schur_only_final_20260610_100954"
COMB = DATA / "_combined_histories"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# global publication style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# scheme -> (label, color, linestyle, linewidth, zorder)
SCHEMES = {
    "proposed":       ("Proposed (MSA-LBM)", "#B11226", "-",  2.4, 10),
    "picard":         ("Picard",             "#1f77b4", "--", 1.2, 3),
    "anderson":       ("Anderson",           "#2ca02c", "-.", 1.2, 4),
    "inexact_newton": ("Inexact Newton",     "#ff7f0e", ":",  1.3, 5),
    "preconditioned": ("Precond. LBM",       "#9467bd", "--", 1.2, 6),
    "dual_time_mg":   ("Dual-time MG",       "#8c564b", "-.", 1.2, 7),
}
PLOT_ORDER = ["picard", "anderson", "inexact_newton", "preconditioned",
              "dual_time_mg", "proposed"]

# D2Q9 velocity set (matches lbm_core)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

BASE_LABEL = {
    "channel_poiseuille": "Channel Poiseuille",
    "couette": "Couette",
    "cavity_re100": "Lid-driven cavity, Re=100",
    "cavity_re400": "Lid-driven cavity, Re=400",
    "cavity_re1000": "Lid-driven cavity, Re=1000",
    "multi_cylinder": "Multi-cylinder array",
    "backward_step": "Backward-facing step",
    "cylinder_wake": "Cylinder wake",
    "t_junction": "T-junction",
}
# fixed display order for grids / bars
BASE_ORDER = ["channel_poiseuille", "couette", "cavity_re100", "cavity_re400",
              "cavity_re1000", "multi_cylinder", "backward_step",
              "cylinder_wake", "t_junction"]


def base_key(case_id: str) -> str:
    # match the most specific (longest) prefix first so that, e.g.,
    # 'cavity_re1000_...' is not captured by 'cavity_re100'
    for k in sorted(BASE_ORDER, key=len, reverse=True):
        if case_id.startswith(k):
            return k
    return case_id.split("__")[0]


def level_of(case_id: str) -> str:
    tail = case_id.split("__")[-1]
    return tail if tail in ("1x", "2x", "3x") else "?"


def read_combined(case_id: str, kind: str):
    """kind in {residual, wall_seconds, wall_seconds_monotone}. Returns dict scheme->np.array."""
    p = COMB / f"{case_id}__{kind}.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        cols = {h: i for i, h in enumerate(header)}
        data = {h: [] for h in header}
        for row in rd:
            for h, i in cols.items():
                v = row[i] if i < len(row) else ""
                data[h].append(v)
    out = {}
    for s in header[1:]:
        arr = np.array([float(x) if str(x).strip() not in ("", "nan") else np.nan
                        for x in data[s]], dtype=float)
        out[s] = arr
    return out


def case_dirs_by_level():
    levels = {"1x": [], "2x": [], "3x": []}
    for d in sorted(DATA.iterdir()):
        if not (d / "histories").is_dir():
            continue
        cid = d.name
        lv = level_of(cid)
        if lv in levels:
            levels[lv].append(cid)
    # order each level by BASE_ORDER
    for lv in levels:
        levels[lv].sort(key=lambda c: BASE_ORDER.index(base_key(c))
                        if base_key(c) in BASE_ORDER else 99)
    return levels


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def plot_convergence_ax(ax, case_id, show_legend=False, title=None):
    res = read_combined(case_id, "residual")
    wall = read_combined(case_id, "wall_seconds_monotone")
    if not res or not wall:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    for s in PLOT_ORDER:
        if s not in res or s not in wall:
            continue
        lab, col, ls, lw, z = SCHEMES[s]
        x = wall[s]; y = res[s]
        m = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if m.sum() < 2:
            continue
        ax.semilogy(x[m], y[m], ls=ls, color=col, lw=lw, zorder=z,
                    label=lab, solid_capstyle="round")
    # mark the level the proposed method reached
    if "proposed" in res:
        yp = res["proposed"]
        yp = yp[np.isfinite(yp) & (yp > 0)]
        if yp.size:
            ax.axhline(yp[-1], color=SCHEMES["proposed"][1], ls=(0, (1, 1)),
                       lw=0.8, alpha=0.5, zorder=2)
    ax.set_title(title or BASE_LABEL.get(base_key(case_id), case_id), pad=3)
    if show_legend:
        ax.legend(loc="upper right", framealpha=0.92, ncol=1, handlelength=2.4)


# --------------------------------------------------------------------------- #
# Figure 1 : mechanism schematic
# --------------------------------------------------------------------------- #
def fig1_mechanism():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 8.2)

    def box(x, y, w, h, text, fc, ec="#222222", fs=8.6, tc="#111111"):
        b = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.1, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4)

    def arrow(x1, y1, x2, y2, text=None, color="#333333", rad=0.0,
              dx=0.0, dy=0.28, fs=7.6, ha="center"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=2)
        if text:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha=ha,
                    va="bottom", fontsize=fs, color=color, zorder=5)

    # ---- row y-coordinates (well separated) ----
    yT, yM, yB = 6.5, 3.5, 0.6   # top / middle / bottom rows
    hh = 1.3

    # top: native distribution-space loop
    box(0.4, yT, 2.6, hh, r"$f$" + "\n" + r"distribution functions ($9N$)", "#eef2f7")
    box(4.0, yT, 3.0, hh, r"native operator $G$" + "\n" +
        r"collide $\to$ stream $\to$ BC", "#dfe9f3")
    box(8.0, yT, 3.2, hh, r"residual" + "\n" + r"$R(f)=f-G(f)$", "#eef2f7")
    arrow(3.0, yT + hh / 2, 4.0, yT + hh / 2)
    arrow(7.0, yT + hh / 2, 8.0, yT + hh / 2)

    # middle: moment-space correction
    box(0.4, yM, 2.6, hh, r"project  $\mathbf{M}$" + "\n" + r"$R_m=\mathbf{M}R$",
        "#f6ece1")
    box(4.0, yM, 3.0, hh, r"moment Schur solve" + "\n" +
        r"$S_U\,\delta m=-R_m$  (GMRES)", "#f3dcc6")
    box(8.0, yM, 3.2, hh, r"lift  $\mathbf{T}$" + "\n" + r"$\delta f=\mathbf{T}\,\delta m$",
        "#f6ece1")
    arrow(3.0, yM + hh / 2, 4.0, yM + hh / 2)
    arrow(7.0, yM + hh / 2, 8.0, yM + hh / 2)
    # residual -> project (enter moment space), routed down the right then left
    arrow(9.6, yT, 1.7, yM + hh, color="#9a6a3a", rad=0.16)
    ax.text(6.0, yM + hh + 0.55, r"reduce to conserved-moment subspace ($3N$)",
            fontsize=7.6, color="#9a6a3a", style="italic", ha="center")

    # bottom: gate + accept / fallback
    box(8.0, yB, 3.2, hh, r"admissibility gate" + "\n" + r"$\rho>0,\ \|R\|\downarrow$",
        "#e7f0e3")
    box(4.0, yB, 3.0, hh, r"accept" + "\n" + r"$f\leftarrow f+\alpha\,\delta f$",
        "#d7e8cf")
    box(0.4, yB, 2.6, hh, r"reject $\to$" + "\n" + r"native Picard fallback", "#f1d9d9")
    arrow(9.6, yM, 9.6, yB + hh)                       # lift -> gate (down)
    arrow(8.0, yB + hh / 2, 7.0, yB + hh / 2, text="pass", color="#2a8a3e", dy=0.18)
    arrow(4.0, yB + hh / 2, 3.0, yB + hh / 2, text="fail", color="#b04646", dy=0.18)

    # updated state loop back to f: collect accept + reject on a bottom rail,
    # then run up the far-left channel into f (no box crossings)
    rail_y = yB - 0.55
    railc = "#555555"
    ax.plot([5.5, 5.5], [yB, rail_y], color=railc, lw=1.3, zorder=1)   # accept down
    ax.plot([1.7, 1.7], [yB, rail_y], color=railc, lw=1.3, zorder=1)   # reject down
    ax.plot([5.5, 0.18], [rail_y, rail_y], color=railc, lw=1.3, zorder=1)  # rail
    ax.annotate("", xy=(0.18, yT + 0.2), xytext=(0.18, rail_y),
                arrowprops=dict(arrowstyle="-|>", color=railc, lw=1.5),
                zorder=1)
    ax.text(-0.05, (rail_y + yT) / 2 + 0.3, "updated state", rotation=90,
            fontsize=7.4, va="center", ha="center", color=railc)

    ax.set_title("Moment-Schur acceleration of the native LBM steady-state iteration",
                 fontsize=10.5, pad=8)
    save(fig, "fig1_mechanism")


# --------------------------------------------------------------------------- #
# Figure 2 : representative convergence
# --------------------------------------------------------------------------- #
def fig2_convergence_main(case_id="cavity_re1000_n129__2x"):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    plot_convergence_ax(ax, case_id, show_legend=True,
                        title=None)
    ax.set_xlabel("Wall time  [s]")
    ax.set_ylabel(r"Macroscopic $L_2$ residual")
    ax.set_title(f"{BASE_LABEL.get(base_key(case_id), case_id)}  ({level_of(case_id)})",
                 pad=4)
    # annotate baseline plateau
    ax.text(0.97, 0.18, "baselines stall\non the hydrodynamic\nplateau",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    save(fig, "fig2_convergence_main")


# --------------------------------------------------------------------------- #
# Figures 3 / A1 / A2 : convergence grids
# --------------------------------------------------------------------------- #
def fig_grid(level, name):
    levels = case_dirs_by_level()
    cases = levels.get(level, [])
    if not cases:
        print(f"  [skip] no cases for {level}")
        return
    n = len(cases)
    ncol = 3
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.4, 2.7 * nrow),
                             sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for i, cid in enumerate(cases):
        plot_convergence_ax(axes[i], cid, show_legend=False)
        if i % ncol == 0:
            axes[i].set_ylabel(r"$L_2$ residual")
        if i // ncol == nrow - 1:
            axes[i].set_xlabel("Wall time [s]")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    # shared legend
    handles = [Line2D([0], [0], color=SCHEMES[s][1], ls=SCHEMES[s][2],
                      lw=SCHEMES[s][3] + 0.4, label=SCHEMES[s][0])
               for s in PLOT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Convergence histories at the {level} mesh level "
                 f"(all methods; monotone wall-time axis)", y=1.002, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, name)


# --------------------------------------------------------------------------- #
# Figure 4 : speedup bars (wall + LBE)
# --------------------------------------------------------------------------- #
def _load_summary():
    rows = list(csv.DictReader(
        (DATA / "summary_all_methods_with_latest_ap_schur_only.csv").open(newline="")))
    return rows


def fig4_speedup():
    rows = _load_summary()
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)

    def fval(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    cids = sorted(by_case, key=lambda c: (BASE_ORDER.index(base_key(c))
                  if base_key(c) in BASE_ORDER else 99, level_of(c)))
    labels, wall_ratio, lbe_ratio = [], [], []
    for cid in cids:
        recs = by_case[cid]
        prop = next((r for r in recs if r["method"] == "proposed"), None)
        if not prop:
            continue
        pw, pl = fval(prop["wall_seconds"]), fval(prop["lbe_calls"])
        base_w = [fval(r["wall_seconds"]) for r in recs
                  if r["method"] != "proposed" and str(r["converged"]) in ("1", "True")
                  and fval(r["wall_seconds"]) > 0]
        base_l = [fval(r["lbe_calls"]) for r in recs
                  if r["method"] != "proposed" and str(r["converged"]) in ("1", "True")
                  and fval(r["lbe_calls"]) > 0]
        wr = (min(base_w) / pw) if base_w and pw > 0 else np.nan
        lr = (min(base_l) / pl) if base_l and pl > 0 else np.nan
        labels.append(f"{BASE_LABEL.get(base_key(cid), cid).split(',')[0]}\n{level_of(cid)}")
        wall_ratio.append(wr)
        lbe_ratio.append(lr)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True)
    for ax, ratios, ttl in zip(
            axes, (wall_ratio, lbe_ratio),
            ("Wall-time speedup", "Operator-work (LBE-call) speedup")):
        ratios = np.array(ratios, dtype=float)
        colors = ["#2a8a3e" if (r >= 1 or not np.isfinite(r)) else "#b04646"
                  for r in ratios]
        rfin = np.where(np.isfinite(ratios), ratios, 0.0)
        bars = ax.bar(x, rfin, color=colors, edgecolor="#333", lw=0.5, width=0.74)
        ax.axhline(1.0, color="#222", lw=1.0, ls="--")
        med = np.nanmedian(ratios)
        ax.axhline(med, color="#1f5fae", lw=1.0, ls=":",
                   label=f"median = {med:.2f}×")
        ax.set_ylabel(ttl + r"  ($\times$)")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="x", alpha=0)
        for xi, r in zip(x, ratios):
            if not np.isfinite(r):
                ax.text(xi, 0.05, "n/a", ha="center", va="bottom", fontsize=6,
                        rotation=90, color="#888")
    axes[0].set_title("Per-case acceleration of the proposed method "
                      "(ratio > 1 = proposed faster; green = win)", fontsize=10)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=6.4)
    fig.tight_layout()
    save(fig, "fig4_speedup")


# --------------------------------------------------------------------------- #
# Figure 5 : accuracy / grid convergence  (paper-verified values)
# --------------------------------------------------------------------------- #
def fig5_accuracy():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    # (a) Poiseuille second-order
    Ny = np.array([32, 64, 96])
    h = 1.0 / Ny
    eP = np.array([9.37e-3, 2.27e-3, 1.00e-3])
    ax = axes[0]
    ax.loglog(h, eP, "o-", color="#B11226", mfc="white", ms=6, lw=1.6,
              label="Proposed")
    # slope-2 guide
    hg = np.array([h[0], h[-1]])
    cg = eP[0] / h[0] ** 2
    ax.loglog(hg, cg * hg ** 2, "--", color="#555", lw=1.0, label=r"slope $=2$")
    ax.set_xlabel(r"Grid spacing $h=1/N_y$")
    ax.set_ylabel(r"Velocity rel. $L_2$ error")
    ax.set_title("(a) Channel Poiseuille\n(observed order $\\approx 2.0$)")
    ax.legend(loc="lower right")
    for xi, yi, p in zip(h[1:], eP[1:], [2.04, 2.02]):
        ax.annotate(f"$p={p}$", (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5)

    # (b) Couette machine precision
    lv = np.array([1, 2, 3])
    eC = np.array([2.75e-9, 2.87e-8, 5.19e-8])
    ax = axes[1]
    ax.semilogy(lv, eC, "s-", color="#1f5fae", mfc="white", ms=6, lw=1.6,
                label="Proposed")
    ax.axhspan(1e-10, 1e-7, color="#cfe3f5", alpha=0.5, lw=0,
               label="machine-precision band")
    ax.set_xticks(lv); ax.set_xticklabels(["1x", "2x", "3x"])
    ax.set_xlabel("Mesh level")
    ax.set_ylabel(r"Velocity rel. $L_2$ error")
    ax.set_ylim(1e-10, 1e-2)
    ax.set_title("(b) Couette\n(exactly representable)")
    ax.legend(loc="upper left")

    # (c) cavity Ghia monotone decrease
    ax = axes[2]
    cav = {
        "Re=100": ([0.117, 0.0669, 0.0493], "#B11226", "o"),
        "Re=400": ([0.106, 0.0642, 0.0501], "#e08214", "s"),
        "Re=1000": ([0.0542, 0.0326, 0.0257], "#1f5fae", "^"),
    }
    for lab, (vals, col, mk) in cav.items():
        ax.plot(lv, vals, mk + "-", color=col, mfc="white", ms=6, lw=1.5, label=lab)
    ax.set_xticks(lv); ax.set_xticklabels(["1x", "2x", "3x"])
    ax.set_xlabel("Mesh level")
    ax.set_ylabel(r"Ghia centerline rel. $L_2$ error")
    ax.set_title("(c) Lid-driven cavity\n(monotone toward Ghia 1982)")
    ax.legend(loc="upper right", title="Reynolds")
    fig.tight_layout()
    save(fig, "fig5_accuracy")


# --------------------------------------------------------------------------- #
# helpers for field-based figures
# --------------------------------------------------------------------------- #
def load_proposed_field(case_id):
    npz = DATA / case_id / "npz" / f"{case_id}__proposed.npz"
    if not npz.exists():
        return None
    f = np.load(npz)["f"]  # (9, Ny, Nx)
    rho = f.sum(0)
    ux = (CX[:, None, None] * f).sum(0) / rho
    uy = (CY[:, None, None] * f).sum(0) / rho
    return rho, ux, uy


def load_mask(case_id):
    """fluid mask (Ny,Nx) from a baseline fields CSV (geometry identical)."""
    fd = DATA / case_id / "fields"
    csvs = sorted(fd.glob("*.csv"))
    if not csvs:
        return None
    rows = list(csv.DictReader(csvs[0].open(newline="")))
    iy = np.array([int(r["iy"]) for r in rows])
    ix = np.array([int(r["ix"]) for r in rows])
    fl = np.array([float(r.get("fluid", 1)) for r in rows])
    Ny, Nx = iy.max() + 1, ix.max() + 1
    mask = np.ones((Ny, Nx))
    mask[iy, ix] = fl
    return mask > 0.5


# --------------------------------------------------------------------------- #
# Figure 6 : cavity centerline vs Ghia
# --------------------------------------------------------------------------- #
GHIA_Y = np.array([0.0, .0547, .0625, .0703, .1016, .1719, .2813, .4531, .5,
                   .6172, .7344, .8516, .9531, .9609, .9688, .9766, 1.0])
GHIA_X = np.array([0.0, .0625, .0703, .0781, .0938, .1563, .2266, .2344, .5,
                   .8047, .8594, .9063, .9453, .9531, .9609, .9688, 1.0])
GHIA_U = {
    100: np.array([0, -.03717, -.04192, -.04775, -.06434, -.1015, -.15662,
                   -.2109, -.20581, -.13641, .00332, .23151, .68717, .73722,
                   .78871, .84123, 1.0]),
    400: np.array([0, -.08186, -.09266, -.10338, -.14612, -.24299, -.32726,
                   -.17119, -.11477, .02135, .16256, .29093, .55892, .61756,
                   .68439, .75837, 1.0]),
    1000: np.array([0, -.18109, -.20196, -.2222, -.2973, -.38289, -.27805,
                    -.10648, -.0608, .05702, .18719, .33304, .46604, .51117,
                    .57492, .65928, 1.0]),
}
GHIA_V = {
    100: np.array([0, .09233, .10091, .1089, .12317, .16077, .17507, .17527,
                   .05454, -.24533, -.22445, -.16914, -.10313, -.08864, -.07391,
                   -.05906, 0]),
    400: np.array([0, .1836, .19713, .2092, .22965, .28124, .30203, .30174,
                   .05186, -.38598, -.44993, -.33827, -.22847, -.19254, -.15663,
                   -.12146, 0]),
    1000: np.array([0, .27485, .29012, .30353, .32627, .37095, .33075, .32235,
                    .02526, -.31966, -.42665, -.5155, -.39188, -.33714, -.27669,
                    -.21388, 0]),
}
CAVITY_CASE = {100: "cavity_re100_n33__3x", 400: "cavity_re400_n49__3x",
               1000: "cavity_re1000_n129__3x"}


def fig6_centerline():
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    for j, Re in enumerate((100, 400, 1000)):
        fld = load_proposed_field(CAVITY_CASE[Re])
        if fld is None:
            continue
        rho, ux, uy = fld
        Ny, Nx = ux.shape
        mx, my = Nx // 2, Ny // 2
        Ulid = np.median(ux[-1, :])  # prescribed lid velocity
        y = np.linspace(0, 1, Ny)
        x = np.linspace(0, 1, Nx)
        # u(y) at vertical centerline
        ax = axes[0, j]
        ax.plot(ux[:, mx] / Ulid, y, "-", color="#B11226", lw=1.8, label="Proposed")
        ax.plot(GHIA_U[Re], GHIA_Y, "o", mfc="white", mec="#111", ms=5,
                mew=0.9, label="Ghia 1982")
        ax.set_xlabel(r"$u/U_{\mathrm{lid}}$"); ax.set_ylabel(r"$y/L$")
        ax.set_title(f"Re = {Re}   |   $u$ on $x=0.5$")
        if j == 0:
            ax.legend(loc="upper left")
        # v(x) at horizontal centerline
        ax = axes[1, j]
        ax.plot(x, uy[my, :] / Ulid, "-", color="#1f5fae", lw=1.8, label="Proposed")
        ax.plot(GHIA_X, GHIA_V[Re], "s", mfc="white", mec="#111", ms=5,
                mew=0.9, label="Ghia 1982")
        ax.set_xlabel(r"$x/L$"); ax.set_ylabel(r"$v/U_{\mathrm{lid}}$")
        ax.set_title(f"Re = {Re}   |   $v$ on $y=0.5$")
        if j == 0:
            ax.legend(loc="lower left")
    fig.suptitle("Cavity centerline velocity profiles vs. Ghia et al. (1982)  "
                 "(finest, 3x mesh)", y=1.01, fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig6_centerline")


# --------------------------------------------------------------------------- #
# Figure 7 : velocity-magnitude + vorticity contours
# --------------------------------------------------------------------------- #
# all nine canonical geometries at the finest (3x) mesh
FIELD_CASES = [
    ("channel_poiseuille_Ny96_Nx576__3x", "Channel Poiseuille"),
    ("couette_n32__3x", "Couette"),
    ("cavity_re100_n33__3x", "Cavity, Re=100"),
    ("cavity_re400_n49__3x", "Cavity, Re=400"),
    ("cavity_re1000_n129__3x", "Cavity, Re=1000"),
    ("multi_cylinder_n32__3x", "Multi-cylinder array"),
    ("backward_step_n64__3x", "Backward-facing step"),
    ("cylinder_wake_n64__3x", "Cylinder wake"),
    ("t_junction_Nx288_Ny192_W48__3x", "T-junction"),
]


def _field_components(cid):
    fld = load_proposed_field(cid)
    if fld is None:
        return None
    rho, ux, uy = fld
    mask = load_mask(cid)
    solid = (~mask) if (mask is not None and mask.shape == ux.shape) \
        else np.zeros_like(ux, dtype=bool)
    speed = np.sqrt(ux ** 2 + uy ** 2)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    Ny, Nx = ux.shape
    ext = [0, Nx / max(Ny, Nx), 0, Ny / max(Ny, Nx)]
    return ux, uy, speed, vort, solid, ext


def _field_grid(kind, name):
    """kind in {'speed','vort'}; draw a 3x3 grid over the nine geometries."""
    fig, axes = plt.subplots(3, 3, figsize=(9.6, 7.4))
    axes = axes.ravel()
    for k, (cid, title) in enumerate(FIELD_CASES):
        ax = axes[k]
        comp = _field_components(cid)
        if comp is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=8.8)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ux, uy, speed, vort, solid, ext = comp
        Ny, Nx = ux.shape
        if kind == "speed":
            field = np.ma.array(speed, mask=solid)
            im = ax.imshow(field, origin="lower", cmap="viridis", extent=ext,
                           aspect="equal")
            try:
                xs = np.linspace(0, ext[1], Nx); ys = np.linspace(0, ext[3], Ny)
                uxm = np.ma.array(ux, mask=solid).filled(0.0)
                uym = np.ma.array(uy, mask=solid).filled(0.0)
                ax.streamplot(xs, ys, uxm, uym, color="white", density=0.8,
                              linewidth=0.35, arrowsize=0.45)
            except Exception:
                pass
        else:
            field = np.ma.array(vort, mask=solid)
            vmax = (np.nanpercentile(np.abs(field.compressed()), 97)
                    if field.count() else 1.0) or 1.0
            im = ax.imshow(field, origin="lower", cmap="RdBu_r", extent=ext,
                           aspect="equal", vmin=-vmax, vmax=vmax)
        # shade solid obstacles for clarity
        if solid.any():
            ax.imshow(np.ma.array(np.ones_like(ux), mask=~solid), origin="lower",
                      cmap="Greys", vmin=0, vmax=1, extent=ext, aspect="equal",
                      alpha=0.55, zorder=2)
        ax.set_title(title, fontsize=8.8)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    ttl = ("Velocity magnitude with streamlines" if kind == "speed"
           else "Vorticity")
    fig.suptitle(f"{ttl} for all nine benchmark geometries "
                 f"(proposed-method solution, finest 3x mesh; post-processing only)",
                 y=1.005, fontsize=10)
    fig.tight_layout()
    save(fig, name)


def fig7_fields():
    _field_grid("speed", "fig7_fields")


def fig7b_vorticity():
    _field_grid("vort", "fig7b_vorticity")


# --------------------------------------------------------------------------- #
# Figure 8 : memory scaling
# --------------------------------------------------------------------------- #
def fig8_memory():
    N = np.array([96, 145, 192])
    Ndof = N ** 2
    marginal_MB = np.array([22.2, 50.3, 85.8])
    dense_GB = np.array([51.0, 267.0, 820.0])
    dense_MB = dense_GB * 1024.0
    field_MB = np.array([0.63, 1.44, 2.53])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(Ndof, dense_MB, "^--", color="#b04646", ms=7, mfc="white", lw=1.4,
              label="Dense Jacobian (would-be)")
    ax.loglog(Ndof, marginal_MB, "o-", color="#B11226", ms=7, mfc="white", lw=1.8,
              label="Proposed: measured marginal RSS")
    ax.loglog(Ndof, field_MB, "s:", color="#1f5fae", ms=6, mfc="white", lw=1.2,
              label=r"Single distribution field ($9N\!\times\!8$ B)")
    # O(N) and O(N^2) guides
    g = np.array([Ndof[0], Ndof[-1]])
    ax.loglog(g, marginal_MB[0] * (g / Ndof[0]), color="#888", lw=0.8, ls="-",
              alpha=0.6)
    ax.text(Ndof[-1], marginal_MB[0] * (Ndof[-1] / Ndof[0]) * 0.6,
            r"$\mathcal{O}(N)$", color="#666", fontsize=8)
    ax.loglog(g, dense_MB[0] * (g / Ndof[0]) ** 2, color="#888", lw=0.8, ls="-",
              alpha=0.6)
    ax.text(Ndof[1] * 1.05, dense_MB[0] * (Ndof[1] / Ndof[0]) ** 2 * 1.2,
            r"$\mathcal{O}(N^2)$", color="#666", fontsize=8)
    ax.set_xlabel(r"Degrees of freedom  $N=N_xN_y$")
    ax.set_ylabel("Memory  [MB]")
    ax.set_title("Memory footprint: matrix-free vs. dense Jacobian")
    # secondary x ticks as grid label
    ax.set_xticks(Ndof)
    ax.set_xticklabels([f"${n}^2$" for n in N])
    ax.legend(loc="center right", framealpha=0.92)
    ax.annotate("3–4 orders\nof magnitude", xy=(Ndof[-1], dense_MB[-1]),
                xytext=(Ndof[0] * 1.1, dense_MB[-1] * 0.5), fontsize=8,
                color="#444",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    save(fig, "fig8_memory")


# --------------------------------------------------------------------------- #
# Appendix A3 : determinism
# --------------------------------------------------------------------------- #
def figA3_determinism():
    cases = ["Couette\nn32", "Multi-cyl.\nn32", "Cavity Re100\nn33", "Cyl. wake\nn64"]
    cv = np.array([5.3, 3.9, 3.6, 6.8])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(cases))
    ax.bar(x, cv, color="#1f5fae", edgecolor="#222", lw=0.6, width=0.6,
           label="wall-time CV (7 runs)")
    ax.axhline(7, color="#b04646", ls="--", lw=1.0, label="7% noise ceiling")
    ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=7.5)
    ax.set_ylabel("Wall-time coefficient of variation  [%]")
    ax.set_ylim(0, 9)
    ax.set_title("Run-to-run timing variability vs. operator-work determinism")
    for xi, c in zip(x, cv):
        ax.text(xi, c + 0.15, f"{c:.1f}%", ha="center", fontsize=7.5)
    ax.text(0.5, 0.92, "LBE-call: bit-identical across all 7 runs (CV = 0)",
            transform=ax.transAxes, ha="center", fontsize=8.2, color="#2a8a3e",
            bbox=dict(boxstyle="round,pad=0.3", fc="#eaf5ea", ec="#2a8a3e"))
    ax.legend(loc="upper right")
    save(fig, "figA3_determinism")


# --------------------------------------------------------------------------- #
# Appendix A4 : AP acceptance rate
# --------------------------------------------------------------------------- #
def figA4_acceptance():
    levels = ["1x", "2x", "3x"]
    rate = np.array([78.0, 69.1, 65.1])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.arange(3)
    bars = ax.bar(x, rate, color=["#B11226", "#d1495b", "#e89aa3"],
                  edgecolor="#222", lw=0.6, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_ylabel("AP-Schur acceptance rate  [%]")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Mesh level")
    ax.set_title("Correction acceptance by mesh level\n(overall 71.0%; no zero-accept case)")
    for xi, r in zip(x, rate):
        ax.text(xi, r + 1.5, f"{r:.1f}%", ha="center", fontsize=8.5)
    save(fig, "figA4_acceptance")


# --------------------------------------------------------------------------- #
def main():
    print(f"data: {DATA}")
    print(f"out : {OUT}")
    figs = [
        ("fig1", fig1_mechanism),
        ("fig2", fig2_convergence_main),
        ("fig3", lambda: fig_grid("1x", "fig3_grid_1x")),
        ("fig4", fig4_speedup),
        ("fig5", fig5_accuracy),
        ("fig6", fig6_centerline),
        ("fig7", fig7_fields),
        ("fig7b", fig7b_vorticity),
        ("fig8", fig8_memory),
        ("figA1", lambda: fig_grid("2x", "figA1_grid_2x")),
        ("figA2", lambda: fig_grid("3x", "figA2_grid_3x")),
        ("figA3", figA3_determinism),
        ("figA4", figA4_acceptance),
    ]
    for name, fn in figs:
        try:
            print(f"[{name}]")
            fn()
        except Exception as e:
            import traceback
            print(f"  !! {name} FAILED: {e}")
            traceback.print_exc()
    print("done.")


if __name__ == "__main__":
    main()
