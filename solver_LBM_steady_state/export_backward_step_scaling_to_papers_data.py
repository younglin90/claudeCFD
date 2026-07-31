#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_60case_benchmark_no_force_scaling import (
    CASE_IDS,
    METHODS,
    METHODS as _METHODS,
    _cache_path,
    _load_cached,
    _save_cached,
    case_factory_scaled,
    macro_of,
    row_for,
    run_method_with_wall,
    max_steps_for_scaled,
)
from paper_60case_benchmark_no_force import _normalize_history_wall_axes

SRC = Path("paper_revision_data/no_force_scaling_benchmark")
METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_history(dst: Path, hist):
    rows = _normalize_history_wall_axes(hist)
    with dst.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "residual", "lbe_calls", "wall_seconds_raw", "wall_seconds", "accepted", "phase"])
        wr.writerows(rows)
    return rows


def _plot(histories, fig_dir: Path):
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1, int(r[2])) for r in rows], dtype=float)
        y = np.array([max(1e-16, _safe_float(r[1], 1.0)) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.7, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("LBE calls")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_lbe_calls.png", dpi=170)
    plt.close()

    max_end = 1e-3
    for m in METHODS:
        r = histories[m]
        if r:
            max_end = max(max_end, float(r[-1][3]))
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1e-3, _safe_float(r[3], 1e-3)) for r in rows], dtype=float)
        y = np.array([max(1e-16, _safe_float(r[1], 1.0)) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.7, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1e-3, right=max_end * 1.1)
    plt.xlabel("Wall seconds (raw)")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_wall_seconds.png", dpi=170)
    plt.close()


def export_level(level: int):
    base_id = "backward_step_n64"
    case_id, label, tol, factory = case_factory_scaled(base_id, level)
    out = Path("papers_data") / f"backward_facing_step_N64__{level}x"
    hist_dir = out / "histories"
    vtk_dir = out / "vtk"
    fig_dir = out / "figure"
    _ensure(hist_dir)
    _ensure(vtk_dir)
    _ensure(fig_dir)

    ref_case = factory()
    ref_f, ref_hist, ref_wall = _load_cached(case_id, "picard_lbm")

    rows = []
    histories = {}
    for method in METHODS:
        case = factory()
        if method == "proposed":
            # Proposed must come from current code-hash cache only.
            strict = _cache_path(case_id, method)
            if not strict.exists():
                tol = float(tol)
                max_steps = max_steps_for_scaled(base_id, level)
                f_new, hist_new, wall_new = run_method_with_wall(
                    method, case, tol=tol, max_steps=max_steps, verbose=False
                )
                _save_cached(case_id, method, f_new, [[int(a), float(b), int(c), float(d)] for a, b, c, d in hist_new], wall_new)
            f, hist, wall = _load_cached(case_id, method)
        else:
            f, hist, wall = _load_cached(case_id, method)
        if hist and int(hist[0][2]) > 0:
            # ensure explicit initial residual row
            r0 = case.residual(case.initial_field())
            chi = getattr(case, "chi", None)
            if chi is not None:
                fluid = chi > 0
                rn0 = float(np.sqrt(np.mean(r0[:, fluid] * r0[:, fluid])))
            else:
                rn0 = float(np.sqrt(np.mean(r0 * r0)))
            hist = [(0, rn0, 0, 0.0)] + list(hist)

        hnorm = _write_history(hist_dir / f"{case_id}__{method}.csv", hist)
        histories[method] = hnorm

        src_vtk = SRC / "vtk" / f"{case_id}__{method}.vtk"
        if src_vtk.exists():
            shutil.copy2(src_vtk, vtk_dir / src_vtk.name)

        row = row_for(base_id, case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall)
        rows.append(row)

    _plot(histories, fig_dir)

    fields = list(rows[0].keys())
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "base_case": base_id,
                "case_id": case_id,
                "level": level,
                "source_cache": str(SRC / "npz_cache"),
                "methods": _METHODS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {out}")


def main():
    export_level(2)
    export_level(3)


if __name__ == "__main__":
    main()
