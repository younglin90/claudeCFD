"""Export VTK fields and publication-style figures for fixed30 scaling runs.

This script does not recompute solver results.  It converts cached fields and
history CSVs from ``verify_fixed30_scaling_strict.py`` into paper-facing raw
artifacts under ``paper_revision_data/fixed30_scaling_artifacts``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_60case_benchmark import METHODS, write_vtk
from verify_fixed30_scaling_strict import BASE_CASE_IDS, _cache_path, case_factory_scaled


SRC = Path("paper_revision_data") / "fixed30_scaling_strict"
OUT = Path("paper_revision_data") / "fixed30_scaling_artifacts"
VTK_DIR = OUT / "vtk"
FIG_DIR = OUT / "figures"
HIST_DIR = OUT / "histories"


def read_summary():
    path = SRC / "summary.csv"
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_history(case_id: str, method: str):
    path = SRC / "histories" / f"{case_id}__{method}.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "iter": float(row["iter"]),
                "residual": float(row["residual"]),
                "lbe_calls": float(row["lbe_calls"]),
                "wall_seconds": float(row["wall_seconds"]),
            })
    return rows


def latest_cached(case_id: str, method: str):
    path = _cache_path(case_id, method)
    if not path.exists():
        matches = sorted(
            (SRC / "npz_cache").glob(f"{case_id}__{method}__*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            return None
        path = matches[0]
    data = np.load(path, allow_pickle=False)
    return data["f"]


def export_histories(rows):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    for path in (SRC / "histories").glob("*.csv"):
        target = HIST_DIR / path.name
        target.write_bytes(path.read_bytes())


def export_vtk(rows):
    VTK_DIR.mkdir(parents=True, exist_ok=True)
    seen = {(r["base_case_id"], int(r["scaling_level"]), r["case_id"], r["method"]) for r in rows}
    for base_id, level, case_id, method in sorted(seen):
        cached = latest_cached(case_id, method)
        if cached is None:
            continue
        _, _, _, factory = case_factory_scaled(base_id, level)
        case = factory()
        write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, cached)


def plot_residuals(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = sorted({r["case_id"] for r in rows})
    for case_id in case_ids:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        any_line = False
        for method in METHODS:
            hist = read_history(case_id, method)
            if not hist:
                continue
            x = [h["lbe_calls"] for h in hist]
            y = [max(h["residual"], 1e-16) for h in hist]
            ax.plot(x, y, label=method, linewidth=1.6)
            any_line = True
        if any_line:
            ax.set_yscale("log")
            ax.set_xlabel("LBE calls")
            ax.set_ylabel("native residual")
            ax.set_title(case_id)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=7)
            fig.savefig(FIG_DIR / f"{case_id}__residual_vs_lbe.png", dpi=180)
            fig.savefig(FIG_DIR / f"{case_id}__residual_vs_lbe.pdf")
        plt.close(fig)


def plot_bars(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = sorted({r["case_id"] for r in rows})
    for quantity, ylabel, filename in [
        ("lbe_calls", "LBE calls", "lbe_calls"),
        ("wall_seconds", "wall seconds", "wall_seconds"),
        ("rel_l2_vs_picard", "relative L2 vs Picard", "rel_l2"),
    ]:
        for case_id in case_ids:
            case_rows = [r for r in rows if r["case_id"] == case_id]
            methods = [r["method"] for r in case_rows]
            vals = [float(r[quantity]) for r in case_rows]
            fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
            ax.bar(np.arange(len(methods)), vals)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(case_id)
            if quantity in {"lbe_calls", "wall_seconds", "rel_l2_vs_picard"}:
                ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", alpha=0.25)
            fig.savefig(FIG_DIR / f"{case_id}__{filename}_bar.png", dpi=180)
            fig.savefig(FIG_DIR / f"{case_id}__{filename}_bar.pdf")
            plt.close(fig)


def main():
    rows = read_summary()
    OUT.mkdir(parents=True, exist_ok=True)
    export_histories(rows)
    export_vtk(rows)
    plot_residuals(rows)
    plot_bars(rows)
    manifest = {
        "source": str(SRC),
        "output": str(OUT),
        "row_count": len(rows),
        "vtk_dir": str(VTK_DIR),
        "history_dir": str(HIST_DIR),
        "figure_dir": str(FIG_DIR),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
