#!/usr/bin/env python3
"""Run no-force Couette (N=32, 1x) benchmark and export papers-data artifacts.

Boundary conditions:
- left/right: periodic
- bottom: wall
- top: moving wall with U_wall

Exact solution:
- linear Couette shear profile, u_x(y) = U_wall * y / (N - 1)

This wrapper reuses the no-force scaling benchmark pipeline but isolates the
output under `papers_data/couette_flow_N32__1x/` and adds an exact-profile
comparison figure for the final fields.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_60case_benchmark_no_force_scaling as bench
from lbm_couette import CouetteCase


OUT_ROOT = Path("papers_data") / "couette_flow_N32__1x"
METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _backup_existing(root: Path) -> None:
    if not root.exists():
        return
    backup = Path("papers_data") / "_legacy"
    _ensure_dir(backup)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dst = backup / f"{root.name}__{ts}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(root), str(dst))


def _patch_output_paths() -> None:
    bench.OUT = OUT_ROOT
    bench.HIST_DIR = OUT_ROOT / "histories"
    bench.VTK_DIR = OUT_ROOT / "vtk"
    bench.CACHE_DIR = OUT_ROOT / "npz_cache"


def _load_latest_case_field(case_id: str, method: str):
    candidates = sorted((OUT_ROOT / "npz_cache").glob(f"{case_id}__{method}__*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"missing cache for {case_id} {method}")
    data = np.load(candidates[0], allow_pickle=False)
    hist = [tuple(row) for row in data["hist"].tolist()]
    return data["f"], hist


def _write_exact_comparison(case_id: str) -> None:
    methods = bench.METHODS
    ref_case = CouetteCase(32, nu=0.05, U_wall=0.05)
    y = np.arange(ref_case.N, dtype=np.float64)
    exact = ref_case.analytical_ux()[:, 0]

    fig_dir = OUT_ROOT / "figure"
    field_dir = OUT_ROOT / "fields"
    _ensure_dir(fig_dir)
    _ensure_dir(field_dir)

    rows = []
    plt.figure(figsize=(7.4, 4.8))
    plt.plot(y, exact, "k--", lw=2.2, label="exact linear shear")
    for method in methods:
        f, _hist = _load_latest_case_field(case_id, method)
        _, ux, _uy = ref_case.macro(f)
        mean_ux = ux.mean(axis=1)
        err = mean_ux - exact
        rel_l2 = float(np.sqrt(np.mean(err * err)) / max(float(np.sqrt(np.mean(exact * exact))), 1.0e-30))
        rows.append(
            {
                "method": method,
                "rel_l2_mean_ux": rel_l2,
                "rms_mean_ux": float(np.sqrt(np.mean(err * err))),
                "linf_mean_ux": float(np.max(np.abs(err))),
                "final_mean_ux": float(mean_ux[-1]),
            }
        )
        plt.plot(y, mean_ux, lw=1.5, label=METHOD_LABELS.get(method, method))
    plt.xlabel("y index")
    plt.ylabel("mean u_x across x")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "centerline_vs_exact.png", dpi=220)
    plt.close()

    with (OUT_ROOT / "exact_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["method", "rel_l2_mean_ux", "rms_mean_ux", "linf_mean_ux", "final_mean_ux"])
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    summary = json.loads((OUT_ROOT / "metrics.json").read_text(encoding="utf-8"))
    summary["exact_comparison_csv"] = str(OUT_ROOT / "exact_comparison.csv")
    summary["exact_reference"] = "Couette linear shear exact solution"
    (OUT_ROOT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    _backup_existing(OUT_ROOT)
    _patch_output_paths()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        "paper_60case_benchmark_no_force_scaling.py",
        "--levels",
        "1",
        "--base-cases",
        "couette_n32",
        "--methods",
        ",".join(bench.METHODS),
        "--no-resume",
        "--no-cache",
    ]
    bench.main()
    _write_exact_comparison("couette_n32__1x")
    print(f"[saved] {OUT_ROOT}")


if __name__ == "__main__":
    main()
