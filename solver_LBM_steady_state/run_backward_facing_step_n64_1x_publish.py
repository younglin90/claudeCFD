#!/usr/bin/env python3
"""Run backward-facing-step N64 (1x) no-force benchmark and export artifacts.

Case setup:
- left boundary: inlet (Zou-He velocity)
- right boundary: outlet (zero-gradient extrapolation)
- top/bottom: wall via mask
- forcing term: disabled
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from no_force_suite.no_force_cases import make_case, SUPPORTED_CASES
from paper_60case_benchmark_no_force import (
    _normalize_history_wall_axes,
    macro_of,
    run_one,
    write_vtk,
)

CASE_ID = "backward_step_n64"
OUT_ROOT = Path("papers_data") / "backward_facing_step_N64__1x"
HIST_DIR = OUT_ROOT / "histories"
FIG_DIR = OUT_ROOT / "figure"
FIELD_DIR = OUT_ROOT / "fields"
VTK_DIR = OUT_ROOT / "vtk"

METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]

METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _backup_existing(out_root: Path) -> None:
    if not out_root.exists():
        return
    backup_root = Path("papers_data") / "_legacy"
    _ensure_dir(backup_root)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dst = backup_root / f"{out_root.name}__{ts}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(out_root), str(dst))
    print(f"[backup] {out_root} -> {dst}")


def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _fast_norm(arr: np.ndarray) -> float:
    r = arr.ravel()
    return float(np.sqrt(float(r @ r)))


def _norm_residual_over_fluid(case, f: np.ndarray) -> float:
    r = case.residual(f)
    chi = getattr(case, "chi", None)
    if chi is None:
        return float(np.sqrt(np.mean(r * r)))
    fluid = chi > 0.0
    if not np.any(fluid):
        return float("inf")
    return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))


def _prepend_initial_history(case, hist):
    f0 = case.initial_field()
    r0 = _norm_residual_over_fluid(case, f0)
    h = list(hist) if hist is not None else []
    if not h:
        return [(0, r0, 0, 0.0)]
    first = h[0]
    first_lbe = int(_safe_float(first[2], 0.0)) if len(first) >= 3 else 0
    first_wall = _safe_float(first[3], 0.0) if len(first) >= 4 else 0.0
    if first_lbe <= 0 and first_wall <= 1.0e-15:
        return h
    return [(0, r0, 0, 0.0)] + h


def _write_history_csv(path: Path, hist):
    rows = _normalize_history_wall_axes(hist)
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iter",
            "residual",
            "lbe_calls",
            "wall_seconds_raw",
            "wall_seconds",
            "accepted",
            "phase",
        ])
        for row in rows:
            wr.writerow(row)
    return rows


def _find_reattachment_x(case, ux: np.ndarray):
    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64))
    ny, nx = ux.shape
    x_step = int(round(nx / 3.0))
    # Reattachment is expected near the lower wall after the step.
    # Search multiple streamwise lines in lower half and pick the earliest valid crossing.
    j_candidates = list(range(max(1, ny // 8), max(2, ny // 2)))
    best_x = float("nan")
    best_j = -1
    for j in j_candidates:
        if not (chi[j, x_step + 1] > 0):
            continue
        line = ux[j, :]
        valid = np.where(chi[j, :] > 0)[0]
        valid = valid[valid >= x_step]
        if valid.size < 2:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            ua = line[a]
            ub = line[b]
            if ua <= 0.0 and ub > 0.0:
                t = 0.0 if abs(ub - ua) < 1.0e-14 else (-ua) / (ub - ua)
                xr = float(a + t)
                if not np.isfinite(best_x) or xr < best_x:
                    best_x = xr
                    best_j = j
                break
    return best_x, best_j


def _write_fields(case_id: str, method: str, case, f: np.ndarray, ref_ux: np.ndarray, ref_uy: np.ndarray):
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    err_u = np.sqrt((ux - ref_ux) ** 2 + (uy - ref_uy) ** 2)
    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64))
    ny, nx = ux.shape

    base = FIELD_DIR / f"{case_id}__{method}"
    with (base.with_name(f"{base.name}__macro.csv")).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "rho", "ux", "uy", "speed", "vorticity", "err_u", "chi"])
        for j in range(ny):
            for i in range(nx):
                wr.writerow([j, i, rho[j, i], ux[j, i], uy[j, i], speed[j, i], vort[j, i], err_u[j, i], chi[j, i]])

    midx = nx // 2
    midy = ny // 2
    with (base.with_name(f"{base.name}__centerline_y.csv")).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y_idx", "ux", "uy", "ux_ref", "uy_ref", "ux_err", "uy_err"])
        for j in range(ny):
            wr.writerow([j, ux[j, midx], uy[j, midx], ref_ux[j, midx], ref_uy[j, midx], ux[j, midx] - ref_ux[j, midx], uy[j, midx] - ref_uy[j, midx]])

    with (base.with_name(f"{base.name}__centerline_x.csv")).open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_idx", "ux", "uy", "ux_ref", "uy_ref", "ux_err", "uy_err"])
        for i in range(nx):
            wr.writerow([i, ux[midy, i], uy[midy, i], ref_ux[midy, i], ref_uy[midy, i], ux[midy, i] - ref_ux[midy, i], uy[midy, i] - ref_uy[midy, i]])


def _plot_residual_vs_lbe(histories: dict[str, list[list[float]]]):
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1, int(r[2])) for r in rows], dtype=np.float64)
        y = np.array([max(1.0e-16, _safe_float(r[1], 1.0)) for r in rows], dtype=np.float64)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("LBE calls")
    plt.ylabel("Residual norm")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_vs_lbe_calls.png", dpi=170)
    plt.close()


def _plot_residual_vs_wall(histories: dict[str, list[list[float]]]):
    max_end = 1.0e-3
    for m in METHODS:
        rows = histories[m]
        if rows:
            max_end = max(max_end, _safe_float(rows[-1][3], 0.0))
    xmax = max_end * 1.1
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1.0e-3, _safe_float(r[3], 1.0e-3)) for r in rows], dtype=np.float64)
        y = np.array([max(1.0e-16, _safe_float(r[1], 1.0)) for r in rows], dtype=np.float64)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1.0e-3, right=max(1.1e-3, xmax))
    plt.xlabel("Wall seconds (raw)")
    plt.ylabel("Residual norm")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_vs_wall_seconds.png", dpi=170)
    plt.close()


def _plot_reattachment_bar(summary_rows: list[dict]):
    labels = []
    vals = []
    for r in summary_rows:
        labels.append(METHOD_LABELS[r["method"]])
        vals.append(r["reattach_x_over_H"])
    plt.figure(figsize=(7.4, 4.8))
    xs = np.arange(len(labels))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=20)
    plt.ylabel("Reattachment length x_r/H (centerline sign change)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reattachment_length_comparison.png", dpi=170)
    plt.close()


def _plot_profiles(summary_rows: list[dict], profiles: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    plt.figure(figsize=(7.2, 4.8))
    for r in summary_rows:
        m = r["method"]
        y, ux_c, ux_ref, _ = profiles[m]
        plt.plot(ux_c, y, lw=1.6, label=METHOD_LABELS[m])
    y0, _, ux_ref0, _ = profiles[summary_rows[0]["method"]]
    plt.plot(ux_ref0, y0, "k--", lw=2.0, label="Tight Picard ref")
    plt.xlabel("u_x at mid-x")
    plt.ylabel("y index")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "centerline_profile_y.png", dpi=170)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    if CASE_ID not in SUPPORTED_CASES:
        raise ValueError(f"unsupported case id: {CASE_ID}")
    if not args.no_clean:
        _backup_existing(OUT_ROOT)
    for p in (OUT_ROOT, HIST_DIR, FIG_DIR, FIELD_DIR, VTK_DIR):
        _ensure_dir(p)

    case_label, n, tol, _ = SUPPORTED_CASES[CASE_ID]
    print(f"[case] {CASE_ID}: {case_label} (N={n}, tol={tol})")

    # tight reference: Picard no-cache rerun
    ref_case, _, _, ref_tol, ref_f, ref_hist, ref_wall = run_one(CASE_ID, "picard_lbm", use_cache=not args.no_cache)
    _, ref_ux, ref_uy = macro_of(ref_case, ref_f)

    rows = []
    histories_for_plot = {}
    profile_cache = {}

    for method in METHODS:
        if method == "picard_lbm":
            case, f, hist, wall = ref_case, ref_f, ref_hist, ref_wall
        else:
            case, _, _, _, f, hist, wall = run_one(CASE_ID, method, use_cache=not args.no_cache)

        hist = _prepend_initial_history(case, hist)
        hist_norm = _write_history_csv(HIST_DIR / f"{CASE_ID}__{method}.csv", hist)
        histories_for_plot[method] = hist_norm

        final_res = _safe_float(hist[-1][1], np.inf) if hist else np.inf
        lbe_calls = int(_safe_float(hist[-1][2], 0.0)) if hist else 0
        wall_raw_end = _safe_float(hist[-1][3], 0.0) if hist else 0.0
        if wall <= 0.0 and wall_raw_end > 0.0:
            wall = wall_raw_end

        rho, ux, uy = macro_of(case, f)
        chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64)) > 0.0
        du = ux[chi] - ref_ux[chi]
        dv = uy[chi] - ref_uy[chi]
        den = max(_fast_norm(np.sqrt(ref_ux[chi] ** 2 + ref_uy[chi] ** 2)), 1.0e-30)
        rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / den)
        rms = float(np.sqrt(np.mean(du * du + dv * dv)))
        linf = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))

        xr, sample_j = _find_reattachment_x(case, ux)
        H = max(1.0, n * 0.5)
        xr_over_h = float((xr - n / 3.0) / H) if np.isfinite(xr) else float("nan")

        row = {
            "case_id": CASE_ID,
            "case_label": case_label,
            "method": method,
            "N": int(case.N),
            "tol": float(ref_tol),
            "converged": int(np.isfinite(final_res) and final_res < 5.0 * ref_tol),
            "final_residual": float(final_res),
            "lbe_calls": int(lbe_calls),
            "wall_seconds": float(wall),
            "rel_l2_vs_tight_picard": rel_l2,
            "rms_vs_tight_picard": rms,
            "linf_vs_tight_picard": linf,
            "reattach_x_idx": float(xr),
            "reattach_x_over_H": float(xr_over_h),
            "reattach_sample_y_idx": int(sample_j),
        }
        rows.append(row)
        _write_fields(CASE_ID, method, case, f, ref_ux, ref_uy)
        write_vtk(VTK_DIR / f"{CASE_ID}__{method}.vtk", case, f)

        y = np.arange(case.N, dtype=np.float64)
        midx = case.N // 2
        profile_cache[method] = (y, ux[:, midx], ref_ux[:, midx], uy[:, midx])

        print(
            f"  {method:22s} lbe={row['lbe_calls']:7d} wall={row['wall_seconds']:.4f}s "
            f"res={row['final_residual']:.3e} relL2={row['rel_l2_vs_tight_picard']:.3e}"
        )

    _plot_residual_vs_lbe(histories_for_plot)
    _plot_residual_vs_wall(histories_for_plot)
    _plot_reattachment_bar(rows)
    _plot_profiles(rows, profile_cache)

    fields = list(rows[0].keys())
    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    # Experimental reference fallback note:
    # no local experimental raw dataset shipped in repo for this case.
    metrics = {
        "case_id": CASE_ID,
        "out_root": str(OUT_ROOT),
        "methods": METHODS,
        "reference_policy": "analytic if available; else literature reference; else tight_picard",
        "reference_used_here": "tight_picard",
        "experimental_dataset_used": False,
        "note": "No local experimental backward-facing-step table found in repository; tight Picard used as numerical reference.",
    }
    (OUT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[saved] {OUT_ROOT / 'summary.csv'}")
    print(f"[saved] {OUT_ROOT / 'metrics.json'}")


if __name__ == "__main__":
    main()
