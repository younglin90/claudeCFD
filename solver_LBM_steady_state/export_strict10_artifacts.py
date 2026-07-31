"""Export paper-ready artifacts for the strict 10-case comparison.

Inputs:
  paper_revision_data/fixed10_strict/npz_cache/*.npz
  paper_revision_data/fixed10_strict/summary.csv

Outputs:
  paper_revision_data/fixed10_strict_paper_artifacts/
    histories/*.csv
    vtk/*.vtk
    figures/*.png

The script does not recompute solver results.  It converts cached fields and
histories into raw files suitable for later manuscript plotting/ParaView use.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_60case_benchmark import CASE_IDS, METHODS, case_factory, macro_of, write_vtk


SRC = Path("paper_revision_data") / "fixed10_strict"
CACHE = SRC / "npz_cache"
OUT = Path("paper_revision_data") / "fixed10_strict_paper_artifacts"
HIST_DIR = OUT / "histories"
VTK_DIR = OUT / "vtk"
FIG_DIR = OUT / "figures"
ALL_METHODS = [m for m in METHODS if m != "proposed"] + ["proposed"]


def load_npz(case_id: str, method: str):
    path = CACHE / f"{case_id}__{method}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["f"], data["hist"]


def load_summary_rows():
    path = SRC / "summary.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_history(case_id: str, method: str, hist):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    path = HIST_DIR / f"{case_id}__{method}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "residual", "lbe_calls", "wall_seconds"])
        for row in hist:
            wr.writerow([int(row[0]), float(row[1]), int(row[2]), float(row[3])])


def export_vtk_and_histories():
    exported = []
    VTK_DIR.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS:
        _, _, factory = case_factory(case_id)
        for method in ALL_METHODS:
            item = load_npz(case_id, method)
            if item is None:
                continue
            f, hist = item
            case = factory()
            write_history(case_id, method, hist)
            write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, f)
            exported.append({"case_id": case_id, "method": method})
    return exported


def plot_metric_bars(rows, metric, ylabel, filename):
    if not rows:
        return
    cases = CASE_IDS
    methods = ALL_METHODS
    value = {
        (r["case_id"], r["method"]): float(r[metric])
        for r in rows
        if r.get(metric) not in {None, "", "nan", "inf"}
    }
    x = np.arange(len(cases))
    width = 0.13
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for j, method in enumerate(methods):
        vals = [value.get((c, method), np.nan) for c in cases]
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width, label=method)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=35, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def plot_residual_histories():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for case_id in CASE_IDS:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        any_line = False
        for method in ALL_METHODS:
            item = load_npz(case_id, method)
            if item is None:
                continue
            _, hist = item
            if hist.size == 0:
                continue
            ax.semilogy(hist[:, 2], hist[:, 1], lw=1.8, label=method)
            any_line = True
        if any_line:
            ax.set_title(case_id)
            ax.set_xlabel("LBE calls")
            ax.set_ylabel("native residual")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"{case_id}__residual_history.png", dpi=220)
        plt.close(fig)


def plot_proposed_vs_picard_contours():
    for case_id in CASE_IDS:
        item_p = load_npz(case_id, "picard_lbm")
        item_s = load_npz(case_id, "proposed")
        if item_p is None or item_s is None:
            continue
        f_p, _ = item_p
        f_s, _ = item_s
        _, _, factory = case_factory(case_id)
        case = factory()
        _, ux_p, uy_p = macro_of(case, f_p)
        _, ux_s, uy_s = macro_of(case, f_s)
        speed_p = np.sqrt(ux_p * ux_p + uy_p * uy_p)
        speed_s = np.sqrt(ux_s * ux_s + uy_s * uy_s)
        diff = np.sqrt((ux_s - ux_p) ** 2 + (uy_s - uy_p) ** 2)
        vmax = max(float(np.nanmax(speed_p)), float(np.nanmax(speed_s)), 1.0e-12)
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
        for ax, arr, title, mx in [
            (axes[0], speed_p, "Picard |u|", vmax),
            (axes[1], speed_s, "Proposed |u|", vmax),
            (axes[2], diff, "|Proposed-Picard|", max(float(np.nanmax(diff)), 1.0e-12)),
        ]:
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0.0, vmax=mx)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(case_id)
        fig.savefig(FIG_DIR / f"{case_id}__picard_proposed_contours.png", dpi=220)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    exported = export_vtk_and_histories()
    rows = load_summary_rows()
    plot_metric_bars(rows, "lbe_calls", "LBE calls", "strict10_lbe_calls.png")
    plot_metric_bars(rows, "wall_seconds", "wall seconds", "strict10_wall_seconds.png")
    plot_metric_bars(rows, "rel_l2_vs_picard", "relative L2 vs Picard", "strict10_rel_l2_vs_picard.png")
    plot_residual_histories()
    plot_proposed_vs_picard_contours()
    manifest = {
        "source": str(SRC),
        "output": str(OUT),
        "exported_field_count": len(exported),
        "case_count": len(CASE_IDS),
        "method_count": len(ALL_METHODS),
        "vtk_dir": str(VTK_DIR),
        "history_dir": str(HIST_DIR),
        "figure_dir": str(FIG_DIR),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
