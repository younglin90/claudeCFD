"""V2 figures + tables generator.

Reads paper_revision_data/v2_final/summary.json plus per-case artifacts
(history.csv, centerline.csv, ghia_compare.csv, field.vtk) and produces all
publication figures: convergence histories, speedup bars, accuracy heatmap,
grid scaling, centerline comparisons, contour grid, Pareto plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("paper_revision_data/v2_final")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


CASE_LABELS = {
    "kolmogorov": "Kolmogorov",
    "channel": "Plane Poiseuille",
    "couette": "Couette",
    "cavity_re100": "Cavity Re=100",
    "cavity_re400": "Cavity Re=400",
    "cavity_re1000": "Cavity Re=1000 (PLBE)",
    "multi_cylinder": "Multi-cylinder mask",
    "backward_step": "Backward step",
    "cylinder_wake": "Cylinder wake",
}

METHODS = ["picard_lbm", "anderson_lbm", "preconditioned_lbm",
           "inexact_newton_lbe", "dual_time_mg_lbm", "proposed"]
METHOD_LABELS = {
    "picard_lbm": "Baseline Picard",
    "anderson_lbm": "Anderson [Walker-Ni 2011]",
    "preconditioned_lbm": "Preconditioned [PRE 70]",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "Safe-NN-SCMK v2 (proposed)",
}
COLORS = {
    "picard_lbm": "#808080",
    "anderson_lbm": "#1f77b4",
    "preconditioned_lbm": "#2ca02c",
    "inexact_newton_lbe": "#9467bd",
    "dual_time_mg_lbm": "#ff7f0e",
    "proposed": "#d62728",
}


def load_summary():
    return json.loads((ROOT / "summary.json").read_text())


def load_history(case, N, method):
    p = ROOT / case / f"N{N}" / method / "history.csv"
    if not p.exists():
        return None
    arr = np.genfromtxt(p, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_vtk_velocity(case, N, method):
    p = ROOT / case / f"N{N}" / method / "field.vtk"
    if not p.exists():
        return None
    text = p.read_text().splitlines()
    dims = next(l for l in text if l.startswith("DIMENSIONS"))
    nx, ny, _ = (int(s) for s in dims.split()[1:4])
    start = text.index("VECTORS velocity float") + 1
    vals = np.array([[float(x) for x in l.split()] for l in text[start : start + nx * ny]])
    ux = vals[:, 0].reshape(ny, nx)
    uy = vals[:, 1].reshape(ny, nx)
    return ux, uy


def fig_convergence(summary):
    cases = list(CASE_LABELS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, case in zip(axes.flat, cases):
        # use largest N for this case
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            ax.set_visible(False)
            continue
        N = Ns[-1]
        for m in METHODS:
            h = load_history(case, N, m)
            if h is None or h.shape[0] < 2:
                continue
            ax.semilogy(h[:, 2], h[:, 1], color=COLORS[m], label=METHOD_LABELS[m] if ax is axes[0, 0] else None, lw=1.3)
        ax.set_title(f"{CASE_LABELS[case]} (N={N})", fontsize=10)
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("residual / vel-chg")
        ax.grid(alpha=0.3)
    fig.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "fig_convergence_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_lbe_speedup(summary):
    cases = list(CASE_LABELS.keys())
    # speedup vs picard at largest N per case
    bars = {m: [] for m in METHODS if m != "picard_lbm"}
    xlab = []
    for case in cases:
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        xlab.append(f"{CASE_LABELS[case]}\nN={N}")
        rows = {r["method"]: r for r in summary if r["case_id"] == case and r["N"] == N}
        base = rows.get("picard_lbm", {}).get("total_lbe_calls", 0)
        for m in bars:
            r = rows.get(m, {})
            lbe = r.get("total_lbe_calls", 0)
            bars[m].append(base / lbe if lbe and r.get("converged") else 0)
    x = np.arange(len(xlab))
    w = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, m in enumerate(bars):
        ax.bar(x + (i - 2) * w, bars[m], w, label=METHOD_LABELS[m], color=COLORS[m])
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("LBE-call speedup vs Picard")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lbe_speedup.png", dpi=160)
    plt.close(fig)


def fig_wall_speedup(summary):
    cases = list(CASE_LABELS.keys())
    bars = {m: [] for m in METHODS if m != "picard_lbm"}
    xlab = []
    for case in cases:
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        xlab.append(f"{CASE_LABELS[case]}\nN={N}")
        rows = {r["method"]: r for r in summary if r["case_id"] == case and r["N"] == N}
        base = rows.get("picard_lbm", {}).get("wall_seconds", 0)
        for m in bars:
            r = rows.get(m, {})
            wall = r.get("wall_seconds", 0)
            bars[m].append(base / wall if wall and r.get("converged") else 0)
    x = np.arange(len(xlab))
    w = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, m in enumerate(bars):
        ax.bar(x + (i - 2) * w, bars[m], w, label=METHOD_LABELS[m], color=COLORS[m])
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("Wall-time speedup vs Picard")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wall_speedup.png", dpi=160)
    plt.close(fig)


def fig_accuracy_heatmap(summary):
    cases = list(CASE_LABELS.keys())
    # build matrix: rows = method, cols = case (largest N)
    mat = np.full((len(METHODS), len(cases)), np.nan)
    for j, case in enumerate(cases):
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        for i, m in enumerate(METHODS):
            rr = [r for r in summary if r["case_id"] == case and r["N"] == N and r["method"] == m]
            if not rr:
                continue
            r = rr[0]
            err = r.get("analytic_rel_l2")
            if err is None:
                err = max(r.get("ghia_u_rms", 0), r.get("ghia_v_rms", 0)) or r.get("vs_picard_rel_l2")
            if err is None:
                err = np.nan
            mat[i, j] = err
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(np.log10(np.maximum(mat, 1e-12)), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([CASE_LABELS[c] for c in cases], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
    for i in range(len(METHODS)):
        for j in range(len(cases)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.1e}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.05 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="log10(relative L2 / RMS error)")
    ax.set_title("Accuracy heatmap (lower = better)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_accuracy_heatmap.png", dpi=160)
    plt.close(fig)


def fig_stability_map(summary):
    cases = list(CASE_LABELS.keys())
    mat = np.zeros((len(METHODS), len(cases)))
    for j, case in enumerate(cases):
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        for i, m in enumerate(METHODS):
            rr = [r for r in summary if r["case_id"] == case and r["N"] == N and r["method"] == m]
            if rr and rr[0].get("converged"):
                mat[i, j] = 1
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([CASE_LABELS[c] for c in cases], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
    for i in range(len(METHODS)):
        for j in range(len(cases)):
            ax.text(j, i, "Y" if mat[i, j] else "N", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title("Convergence map (Y/N) at largest N per case")
    fig.tight_layout()
    fig.savefig(FIG / "fig_stability_map.png", dpi=160)
    plt.close(fig)


def fig_grid_scaling(summary):
    cases = list(CASE_LABELS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, case in zip(axes.flat, cases):
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if len(Ns) < 2:
            ax.set_visible(False)
            continue
        for m in METHODS:
            lbe = []
            for N in Ns:
                rr = [r for r in summary if r["case_id"] == case and r["N"] == N and r["method"] == m and r.get("converged")]
                lbe.append(rr[0]["total_lbe_calls"] if rr else np.nan)
            ax.loglog(Ns, lbe, "o-", color=COLORS[m], label=METHOD_LABELS[m] if ax is axes[0, 0] else None, lw=1.2)
        ax.set_title(f"{CASE_LABELS[case]}", fontsize=10)
        ax.set_xlabel("N")
        ax.set_ylabel("LBE calls (converged only)")
        ax.grid(which="both", alpha=0.3)
    fig.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "fig_grid_scaling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_centerlines_cavity(summary):
    cavities = [("cavity_re100", 100), ("cavity_re400", 400), ("cavity_re1000", 1000)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (case, Re) in enumerate(cavities):
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        for m in METHODS:
            p = ROOT / case / f"N{N}" / m / "ghia_compare.csv"
            if not p.exists():
                continue
            arr = np.genfromtxt(p, delimiter=",", skip_header=1)
            axes[0, col].plot(arr[:, 2], arr[:, 1], "o-", color=COLORS[m], label=METHOD_LABELS[m] if col == 0 else None, ms=3, lw=0.9)
            axes[1, col].plot(arr[:, 4], arr[:, 5], "o-", color=COLORS[m], ms=3, lw=0.9)
        # ghia reference
        p_ref = ROOT / case / f"N{N}" / "picard_lbm" / "ghia_compare.csv"
        if p_ref.exists():
            arr = np.genfromtxt(p_ref, delimiter=",", skip_header=1)
            axes[0, col].plot(arr[:, 3], arr[:, 1], "k*", ms=8, label="Ghia 1982" if col == 0 else None)
            axes[1, col].plot(arr[:, 4], arr[:, 6], "k*", ms=8)
        axes[0, col].set_title(f"Cavity Re={Re} u-centerline (N={N})", fontsize=10)
        axes[1, col].set_title(f"Cavity Re={Re} v-centerline (N={N})", fontsize=10)
        for r in range(2):
            axes[r, col].set_xlabel("y" if r == 0 else "x")
            axes[r, col].set_ylabel("u/U" if r == 0 else "v/U")
            axes[r, col].grid(alpha=0.3)
    fig.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "fig_cavity_centerlines.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_contour_grid_proposed(summary):
    cases = list(CASE_LABELS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, case in zip(axes.flat, cases):
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            ax.set_visible(False)
            continue
        N = Ns[-1]
        v = load_vtk_velocity(case, N, "proposed")
        if v is None:
            v = load_vtk_velocity(case, N, "picard_lbm")
        if v is None:
            ax.set_visible(False)
            continue
        ux, uy = v
        spd = np.sqrt(ux ** 2 + uy ** 2)
        im = ax.imshow(spd, origin="lower", cmap="turbo")
        ax.set_title(f"{CASE_LABELS[case]} N={N}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Velocity magnitude contours (proposed solver)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "fig_contour_grid_proposed.png", dpi=160)
    plt.close(fig)


def fig_pareto(summary):
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in METHODS:
        xs, ys = [], []
        for r in summary:
            if r["method"] != m or not r.get("converged"):
                continue
            acc = r.get("analytic_rel_l2") or r.get("ghia_u_rms") or r.get("vs_picard_rel_l2")
            wall = r.get("wall_seconds")
            if acc is None or wall is None:
                continue
            xs.append(wall); ys.append(acc)
        ax.loglog(xs, ys, "o", color=COLORS[m], label=METHOD_LABELS[m], alpha=0.75, ms=6)
    ax.set_xlabel("wall-time (s)")
    ax.set_ylabel("error (rel L2 vs analytic/ghia/picard)")
    ax.set_title("Pareto: accuracy vs wall-time across all (case, N)")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pareto.png", dpi=160)
    plt.close(fig)


def table_methods_summary(summary):
    out = []
    out.append("# Method-vs-case grand summary (largest N per case)")
    out.append("")
    out.append("| Case | N | Method | LBE calls | wall (s) | native res | vel-chg | rel L2 / Ghia | converged |")
    out.append("|---|---:|---|---:|---:|---:|---:|---:|:---:|")
    cases = list(CASE_LABELS.keys())
    for case in cases:
        Ns = sorted({r["N"] for r in summary if r["case_id"] == case})
        if not Ns:
            continue
        N = Ns[-1]
        for m in METHODS:
            rr = [r for r in summary if r["case_id"] == case and r["N"] == N and r["method"] == m]
            if not rr:
                continue
            r = rr[0]
            err = r.get("analytic_rel_l2") or r.get("ghia_u_rms") or r.get("vs_picard_rel_l2") or float("nan")
            out.append(
                f"| {CASE_LABELS[case]} | {N} | {METHOD_LABELS[m]} | "
                f"{r.get('total_lbe_calls','-')} | {r.get('wall_seconds',float('nan')):.1f} | "
                f"{r.get('native_residual',float('nan')):.2e} | "
                f"{r.get('tail_velocity_change',float('nan')):.2e} | "
                f"{err:.3e} | {'Y' if r.get('converged') else 'N'} |"
            )
    (ROOT / "grand_summary_table.md").write_text("\n".join(out), encoding="utf-8")


def main():
    summary = load_summary()
    print(f"loaded {len(summary)} rows")
    fig_convergence(summary)
    fig_lbe_speedup(summary)
    fig_wall_speedup(summary)
    fig_accuracy_heatmap(summary)
    fig_stability_map(summary)
    fig_grid_scaling(summary)
    fig_centerlines_cavity(summary)
    fig_contour_grid_proposed(summary)
    fig_pareto(summary)
    table_methods_summary(summary)
    print(f"figures saved to {FIG}")


if __name__ == "__main__":
    main()
