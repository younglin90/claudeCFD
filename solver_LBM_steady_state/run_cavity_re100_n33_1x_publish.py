#!/usr/bin/env python3
"""Run lid-driven cavity Re=100, N=33 (1x) and export paper-ready artifacts.

BC/physics:
- Top wall moving in +x with U=0.1
- Other three walls no-slip
- No forcing term (LBMCavity native case, forcing-free).

Outputs:
- papers_data/lid_driven_Re100_N33/
  - summary.csv
  - summary.json
  - metrics.json
  - histories/*.csv
  - fields/*.csv
  - vtk/*.vtk
  - figure/*.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("NUMBA_NUM_THREADS", "24")
os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lbm_core import LBMCavity
from ghia_validation import extract_centerline, get_ghia_data
from paper_60case_benchmark import macro_of, max_steps_for, run_method, write_vtk

CASE_ID = "cavity_re100_n33"
CASE_LABEL = "Lid-driven cavity Re=100 N=33"
RE = 100
N = 33
U_WALL = 0.1
TOL = 5e-7

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

OUT_ROOT = Path("papers_data") / "lid_driven_Re100_N33"
FIELD_DIR = OUT_ROOT / "fields"
FIG_DIR = OUT_ROOT / "figure"
HIST_DIR = OUT_ROOT / "histories"
VTK_DIR = OUT_ROOT / "vtk"


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _vector_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


def _is_finite(v):
    try:
        return float(v) == float(v)
    except Exception:
        return False


def _normalize_history_for_plot(hist, spike_ratio: float = 2.0):
    """Normalize history to monotone axes and mark large residual spikes.

    Some solvers emit phase resets (iteration/wall time restarts), which can make
    raw residual plots visually misleading. This routine:
      - builds a global monotone iteration index,
      - builds monotone cumulative wall-time axis,
      - marks rows where residual jumps are likely rejected probes as non-accepted.
    """
    rows = [list(r) for r in hist if r]
    if not rows:
        return []

    out = []
    first = rows[0]
    it_v = int(first[0]) if _is_finite(first[0]) else 0
    res_v = float(first[1]) if _is_finite(first[1]) else float("nan")
    lbe_v = int(first[2]) if _is_finite(first[2]) else 0
    wall_v = float(first[3]) if len(first) >= 4 and _is_finite(first[3]) else 0.0

    phase = 0
    segment_base = wall_v
    segment_offset = 0.0
    prev_iter = it_v
    prev_wall_raw = wall_v
    prev_residual = res_v
    prev_wall_norm = 0.0

    out.append([0, it_v, res_v, lbe_v, wall_v, 0.0, 1, phase])

    global_iter = 1
    for row in rows[1:]:
        if len(row) < 4:
            continue
        it_v = int(row[0]) if _is_finite(row[0]) else global_iter
        res_v = float(row[1]) if _is_finite(row[1]) else float("nan")
        lbe_v = int(row[2]) if _is_finite(row[2]) else 0
        wall_v = float(row[3]) if _is_finite(row[3]) else float("nan")
        if not _is_finite(wall_v):
            continue

        is_reset = _is_finite(prev_iter) and _is_finite(prev_wall_raw) and (
            it_v < prev_iter or wall_v < prev_wall_raw - 1.0e-15
        )
        if is_reset:
            segment_offset = prev_wall_norm
            segment_base = wall_v
            phase += 1

        wall_local = wall_v - segment_base
        wall_local = wall_local if wall_local >= 0.0 else 0.0
        wall_norm = segment_offset + wall_local
        if wall_norm < prev_wall_norm:
            wall_norm = prev_wall_norm

        accepted = 1
        if _is_finite(prev_residual) and _is_finite(res_v) and prev_residual > 0.0:
            if res_v / prev_residual > spike_ratio:
                accepted = 0

        out.append([global_iter, it_v, res_v, lbe_v, wall_v, wall_norm, accepted, phase])

        global_iter += 1
        prev_iter = it_v
        prev_wall_raw = wall_v
        prev_residual = res_v
        prev_wall_norm = wall_norm

    return out


def _vorticity(ux: np.ndarray, uy: np.ndarray, dx: float = 1.0, dy: float = 1.0):
    du_dy = np.gradient(ux, axis=0, edge_order=1) / dy
    dv_dx = np.gradient(uy, axis=1, edge_order=1) / dx
    return du_dy - dv_dx


def _sha256(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        path = Path(p)
        if not path.exists():
            continue
        h.update(path.as_posix().encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def backup_existing():
    if not OUT_ROOT.exists():
        return
    backup_root = Path("papers_data") / "_legacy"
    _ensure(backup_root)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dst = backup_root / f"{OUT_ROOT.name}__{ts}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(OUT_ROOT), str(dst))
    print(f"[backup] {OUT_ROOT} -> {dst}")


def case_factory():
    return LBMCavity(N=N, Re=RE, U_wall=U_WALL)


def _run_one(method: str, tol: float, max_steps: int):
    case = case_factory()
    f, hist, wall = run_method(method, case, tol=tol, max_steps=max_steps, verbose=False)
    return case, f, hist, wall


def _accuracy_vs_ref(ref_case, f_ref, case, f):
    rho_ref, ux_ref, uy_ref = macro_of(ref_case, f_ref)
    rho, ux, uy = macro_of(case, f)
    du = ux - ux_ref
    dv = uy - uy_ref
    den = np.sqrt(np.sum(ux_ref * ux_ref + uy_ref * uy_ref))
    den = max(float(den), 1e-30)

    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    speed = np.sqrt(ux * ux + uy * uy)
    d_speed = speed - speed_ref

    return {
        "rel_l2_vs_picard": float(_vector_norm(np.hstack((du.ravel(), dv.ravel()))) / den),
        "linf_vs_picard": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
        "rms_vs_picard": float(np.sqrt(np.mean(du * du + dv * dv))),
        "speed_rms": float(np.sqrt(np.mean(d_speed * d_speed))),
        "speed_l2": float(_vector_norm(d_speed) / max(_vector_norm(speed_ref), 1e-30)),
        "mean_rho": float(np.mean(rho)),
    }


def _ghia_metrics(case, f):
    y_ref, u_ref, x_ref, v_ref = get_ghia_data(RE)
    y, u_vert, x, v_horiz = extract_centerline(f, case, case.U_wall)

    u_interp = np.interp(y_ref, y, u_vert)
    v_interp = np.interp(x_ref, x, v_horiz)
    du = u_interp - u_ref
    dv = v_interp - v_ref

    return {
        "y_ref": y_ref,
        "u_ref": u_ref,
        "x_ref": x_ref,
        "v_ref": v_ref,
        "y": y,
        "u_vert": u_vert,
        "x": x,
        "v_horiz": v_horiz,
        "ghia_u_rms": float(np.sqrt(np.mean(du * du))),
        "ghia_v_rms": float(np.sqrt(np.mean(dv * dv))),
        "ghia_u_linf": float(np.max(np.abs(du))),
        "ghia_v_linf": float(np.max(np.abs(dv))),
        "centerline_max": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
    }


def _write_history(path: Path, hist):
    _ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iteration", "residual", "lbe_calls", "wall_seconds"])
        for it, res, lbe, wall in hist:
            wr.writerow([int(it), _safe_float(res), int(_safe_float(lbe, 0.0)), _safe_float(wall)])


def _write_normalized_history(path: Path, hist_norm):
    _ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iter_global",
            "iter_local",
            "residual",
            "lbe_calls",
            "wall_seconds_raw",
            "wall_seconds",
            "accepted",
            "phase",
        ])
        for row in hist_norm:
            wr.writerow([
                int(row[0]),
                int(row[1]),
                _safe_float(row[2]),
                int(_safe_float(row[3], 0.0)),
                _safe_float(row[4]),
                _safe_float(row[5]),
                int(_safe_float(row[6], 0)),
                int(_safe_float(row[7], 0)),
            ])


def _wall_zoom_limits(results, target=None, eps=1.0e-12):
    """Return wall-time x-range focused on method convergence, ignoring long-tail outliers."""
    if target is None:
        target = 5.0 * TOL
    conv_walls = []
    any_walls = []
    for row in results.values():
        hist_norm = row.get("history_norm", [])
        if not hist_norm:
            continue
        for r in hist_norm:
            if int(r[6]) != 1:
                continue
            res = _safe_float(r[2])
            wall = _safe_float(r[5])
            if _is_finite(res) and _is_finite(wall):
                any_walls.append(wall)
            if _is_finite(res) and res <= target:
                conv_walls.append(wall)
                break

    if not any_walls:
        return 1.0e-12, 1.0
    if len(conv_walls) > 0:
        xmax = max(conv_walls)
        xmax = max(10.0 * eps, xmax * 2.0)
    else:
        q95 = float(np.quantile(any_walls, 0.95))
        xmax = max(10.0 * eps, q95 * 1.2)
    xmin = max(10.0 * eps, min(any_walls) * 0.8)
    if xmax <= xmin:
        xmax = xmin * 10.0
    return xmin, xmax


def _wall_full_limits(results, eps=1.0e-12, min_x=1.0e-3, margin_ratio=0.20):
    """Use actual final wall-time of each method to build a consistent x-range."""
    finals = []
    for row in results.values():
        wall_final = row.get("wall_seconds")
        if wall_final is None:
            hist = row.get("history", [])
            if hist:
                wall_final = _safe_float(hist[-1][3])
        if wall_final is None:
            continue
        if _is_finite(wall_final) and wall_final > 0.0:
            finals.append(float(wall_final))
    if not finals:
        return 1.0e-3, 1.0

    wall_min = min(finals)
    wall_max = max(finals)
    if not np.isfinite(wall_min) or not np.isfinite(wall_max):
        return min_x, 1.0

    if wall_min <= eps:
        wall_min = min_x
    if wall_max <= eps:
        return min_x, min_x * (1.0 + margin_ratio)

    # Full-range view should always start from 1e-3 on log x-axis
    # and extend to the slowest method's wall-time with margin.
    xmin = min_x
    xmax = max(
        wall_max + max(wall_max * margin_ratio, min_x),
        xmin * (1.0 + margin_ratio),
    )
    if xmax <= xmin:
        xmax = xmin * 10.0
    return xmin, xmax


def _write_field_csvs(case_id: str, method: str, case, f, rho_ref, ux_ref, uy_ref, ghia_data):
    _, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = _vorticity(ux, uy)
    vort_ref = _vorticity(ux_ref, uy_ref)
    ny, nx = ux.shape
    base = FIELD_DIR / f"{case_id}__{method}"

    # field macro
    with (base.with_name(f"{base.name}__macro.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "rho", "ux", "uy", "speed", "chi"])
        for j in range(ny):
            for i in range(nx):
                wr.writerow([j, i, float(i), float(j), 1.0, ux[j, i], uy[j, i], speed[j, i], 1.0])

    # vorticity field
    with (base.with_name(f"{base.name}__vorticity.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "vorticity", "vorticity_ref", "vorticity_error", "chi"])
        for j in range(ny):
            for i in range(nx):
                wr.writerow([j, i, float(i), float(j), vort[j, i], vort_ref[j, i], vort[j, i] - vort_ref[j, i], 1.0])

    # pointwise error
    with (base.with_name(f"{base.name}__error.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "ux_err", "uy_err", "speed_err", "speed_ref", "speed", "vort_err", "chi"])
        for j in range(ny):
            for i in range(nx):
                wr.writerow([
                    j, i, float(i), float(j),
                    ux[j, i] - ux_ref[j, i],
                    uy[j, i] - uy_ref[j, i],
                    speed[j, i] - speed_ref[j, i],
                    speed_ref[j, i],
                    speed[j, i],
                    vort[j, i] - vort_ref[j, i],
                    1.0,
                ])

    # Centerline data for Ghia benchmark comparison
    i_mid = nx // 2
    j_mid = ny // 2
    with (base.with_name(f"{base.name}__centerline_u_vertical.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y", "u", "u_ref_picard", "u_err_picard", "u_ref_ghia", "u_err_ghia"])
        for j in range(ny):
            y = j / max(ny - 1, 1)
            u_ghia = float(np.interp(y, ghia_data["y_ref"], ghia_data["u_ref"]))
            u_val = float(ux[j, i_mid])
            wr.writerow([
                y,
                u_val,
                float(ux_ref[j, i_mid]),
                float(u_val - ux_ref[j, i_mid]),
                u_ghia,
                float(u_val - u_ghia),
            ])

    with (base.with_name(f"{base.name}__centerline_v_horizontal.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "v", "v_ref_picard", "v_err_picard", "v_ref_ghia", "v_err_ghia"])
        for i in range(nx):
            x = i / max(nx - 1, 1)
            v_ghia = float(np.interp(x, ghia_data["x_ref"], ghia_data["v_ref"]))
            v_val = float(uy[j_mid, i])
            v_ref = float(uy_ref[j_mid, i])
            wr.writerow([
                x,
                v_val,
                v_ref,
                float(v_val - v_ref),
                v_ghia,
                float(v_val - v_ghia),
            ])

    # analytic error file (for uniformity with other artifacts)
    with (base.with_name(f"{base.name}__analytic_error.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "ux_err", "uy_err", "speed_err", "vort_err", "mask"])
        for j in range(ny):
            for i in range(nx):
                wr.writerow([
                    j, i, float(i), float(j),
                    ux[j, i] - ux_ref[j, i],
                    uy[j, i] - uy_ref[j, i],
                    speed[j, i] - speed_ref[j, i],
                    vort[j, i] - vort_ref[j, i],
                    1,
                ])


def _plot_residual_vs_iteration(results):
    # Accepted-only plot (clean curve for manuscript)
    fig_acc, ax_acc = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    # Raw history with phase/spike context for debugging
    fig_raw, ax_raw = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for method, row in results.items():
        hist_norm = row.get("history_norm", [])
        if not hist_norm:
            continue

        accepted = [r for r in hist_norm if int(r[6]) == 1]
        if accepted:
            xs = [int(r[0]) + 1 for r in accepted]
            ys = [_safe_float(r[2]) for r in accepted]
            ax_acc.plot(xs, [max(y, 1e-30) for y in ys], label=METHOD_LABELS.get(method, method), lw=1.3)

        if hist_norm:
            xs = [int(r[0]) + 1 for r in hist_norm]
            ys = [_safe_float(r[2]) for r in hist_norm]
            ax_raw.plot(xs, [max(y, 1e-30) for y in ys], label=METHOD_LABELS.get(method, method), lw=1.3, alpha=0.35)

        spikes = [r for r in hist_norm if int(r[6]) == 0]
        if spikes:
            ax_acc.scatter(
                [int(r[0]) + 1 for r in spikes],
                [max(_safe_float(r[2]), 1e-30) for r in spikes],
                marker="x",
                s=24,
                color="tab:red",
            )

    for a in (ax_acc, ax_raw):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("iteration")
        a.set_ylabel("residual norm")
        a.set_title("Lid-driven cavity Re=100, N=33: residual vs iteration")
        a.grid(True, which="both", alpha=0.35)
        a.legend(fontsize=8)

    fig_acc.savefig(FIG_DIR / "residual_vs_iteration.png", dpi=220)
    fig_raw.savefig(FIG_DIR / "residual_vs_iteration_raw.png", dpi=220)
    plt.close(fig_acc)
    plt.close(fig_raw)


def _plot_residual_vs_wall(results):
    xlim = _wall_zoom_limits(results)
    xlim_full = _wall_full_limits(results)
    fig_acc, ax_acc = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    fig_raw, ax_raw = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    fig_zoom, ax_zoom = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)

    for method, row in results.items():
        hist_norm = row.get("history_norm", [])
        if not hist_norm:
            continue

        accepted = [r for r in hist_norm if int(r[6]) == 1]
        if accepted:
            xs_acc = [max(_safe_float(r[5]), 1e-30) for r in accepted]
            ys_acc = [max(_safe_float(r[2]), 1e-30) for r in accepted]
            ax_acc.plot(xs_acc, ys_acc, label=METHOD_LABELS.get(method, method), lw=1.3)

        xs_raw = [max(_safe_float(r[4]), 1e-30) for r in hist_norm]
        ys_raw = [max(_safe_float(r[2]), 1e-30) for r in hist_norm]
        ax_raw.plot(xs_raw, ys_raw, label=METHOD_LABELS.get(method, method), lw=1.3, alpha=0.35)

        spikes = [r for r in hist_norm if int(r[6]) == 0]
        if spikes:
            ax_acc.scatter(
                [max(_safe_float(r[5]), 1e-30) for r in spikes],
                [max(_safe_float(r[2]), 1e-30) for r in spikes],
                marker="x",
                s=24,
                color="tab:red",
            )
        # Final marker for accepted history (use monotone cumulative axis)
        if accepted:
            acc_final_wall = _safe_float(accepted[-1][5])
            acc_final_res = _safe_float(accepted[-1][2])
            if _is_finite(acc_final_wall) and _is_finite(acc_final_res):
                ax_acc.axvline(acc_final_wall, color="0.4", ls=":", lw=0.9)
                ax_acc.scatter([acc_final_wall], [max(acc_final_res, 1e-30)], marker="o", s=16)

        # Final marker for raw history (use raw wall time from solver history)
        wall_final_raw = row.get("wall_seconds")
        if wall_final_raw is None and hist_norm:
            wall_final_raw = _safe_float(hist_norm[-1][4])
        res_final = row.get("final_residual", hist_norm[-1][2] if hist_norm else float("nan"))
        if _is_finite(wall_final_raw) and _is_finite(res_final):
            ax_raw.axvline(_safe_float(wall_final_raw), color="0.4", ls=":", lw=0.9)
            ax_raw.scatter(
                [_safe_float(wall_final_raw)],
                [max(_safe_float(res_final), 1e-30)],
                marker="o",
                s=16,
            )

    for a in (ax_acc, ax_raw):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("wall seconds")
        a.set_ylabel("residual norm")
        a.set_title("Lid-driven cavity Re=100, N=33: residual vs wall")
        a.set_xlim(xlim_full)
        a.grid(True, which="both", alpha=0.35)
        a.legend(fontsize=8)

    ax_zoom.set_xscale("log")
    ax_zoom.set_yscale("log")
    ax_zoom.set_xlabel("wall seconds")
    ax_zoom.set_ylabel("residual norm")
    ax_zoom.set_title("Lid-driven cavity Re=100, N=33: residual vs wall (zoom)")
    ax_zoom.set_xlim(xlim)
    ax_zoom.grid(True, which="both", alpha=0.35)
    handles, labels = ax_zoom.get_legend_handles_labels()
    if handles:
        ax_zoom.legend(handles, labels, fontsize=8)

    for method, row in results.items():
        hist_norm = row.get("history_norm", [])
        if not hist_norm:
            continue
        accepted = [r for r in hist_norm if int(r[6]) == 1]
        if accepted:
            xs_acc = [max(_safe_float(r[5]), 1e-30) for r in accepted]
            ys_acc = [max(_safe_float(r[2]), 1e-30) for r in accepted]
            ax_zoom.plot(xs_acc, ys_acc, label=METHOD_LABELS.get(method, method), lw=1.3)

    fig_acc.savefig(FIG_DIR / "residual_vs_wall_seconds.png", dpi=220)
    fig_raw.savefig(FIG_DIR / "residual_vs_wall_seconds_raw.png", dpi=220)
    fig_zoom.savefig(FIG_DIR / "residual_vs_wall_seconds_zoom.png", dpi=220)

    plt.close(fig_acc)
    plt.close(fig_raw)
    plt.close(fig_zoom)
    return


def _plot_ghia_profiles(results, ghia_data):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    y_ref = ghia_data["y_ref"]
    x_ref = ghia_data["x_ref"]
    u_ref = ghia_data["u_ref"]
    v_ref = ghia_data["v_ref"]

    # u at vertical centerline x=0.5
    ax[0].plot(u_ref, y_ref, "k*", label="Ghia 1982", ms=6)
    for method, row in results.items():
        y, u_vert, _, _ = extract_centerline(row["field"], row["case"], row["case"].U_wall)
        ax[0].plot(u_vert, y, label=METHOD_LABELS.get(method, method))
    ax[0].set_xlabel("u / U_lid")
    ax[0].set_ylabel("y")
    ax[0].set_title("Vertical centerline u(y), Re=100")
    ax[0].grid(True, alpha=0.35)
    ax[0].legend(fontsize=8)

    # v at horizontal centerline y=0.5
    ax[1].plot(x_ref, v_ref, "k*", label="Ghia 1982", ms=6)
    for method, row in results.items():
        _, _, x, v_horiz = extract_centerline(row["field"], row["case"], row["case"].U_wall)
        ax[1].plot(x, v_horiz, label=METHOD_LABELS.get(method, method))
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("v / U_lid")
    ax[1].set_title("Horizontal centerline v(x), Re=100")
    ax[1].grid(True, alpha=0.35)
    ax[1].legend(fontsize=8)

    fig.suptitle("Lid-driven cavity Re=100 centerline comparison vs Ghia")
    fig.savefig(FIG_DIR / "centerline_vs_ghia.png", dpi=220)
    plt.close(fig)


def _write_summary(path: Path, rows):
    fields = [
        "case_id", "method", "tol", "N",
        "final_residual", "lbe_calls", "wall_seconds", "converged",
        "rel_l2_vs_picard", "linf_vs_picard", "rms_vs_picard",
        "speed_rms", "speed_l2", "ghia_u_rms", "ghia_v_rms", "ghia_u_linf", "ghia_v_linf", "centerline_max",
    ]
    _ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def run_case(methods, do_clean=True):
    if do_clean:
        backup_existing()
    _ensure(OUT_ROOT)
    _ensure(FIELD_DIR)
    _ensure(FIG_DIR)
    _ensure(HIST_DIR)
    _ensure(VTK_DIR)

    started = time.perf_counter()

    # Reference (Picard) run and per-method solves
    ref_case, f_ref, hist_ref, wall_ref = _run_one("picard_lbm", TOL, max_steps_for(CASE_ID))
    # ref_case is the same case object that produced f_ref, so geometry/bc is aligned.
    rho_ref, ux_ref, uy_ref = macro_of(ref_case, f_ref)
    ghia_ref = _ghia_metrics(ref_case, f_ref)

    # For output consistency (picard as reference)
    results = {
        "picard_lbm": {
            "case": ref_case,
            "method": "picard_lbm",
            "tol": TOL,
            "field": f_ref,
            "history": hist_ref,
            "wall": wall_ref,
            "history_norm": _normalize_history_for_plot(hist_ref),
            "accuracy": {
                "rel_l2_vs_picard": 0.0,
                "linf_vs_picard": 0.0,
                "rms_vs_picard": 0.0,
                "speed_rms": 0.0,
                "speed_l2": 0.0,
                "mean_rho": 1.0,
            },
            "ghia": ghia_ref,
        }
    }

    for method in methods:
        if method == "picard_lbm":
            continue
        try:
            case, f, hist, wall = _run_one(method, TOL, max_steps_for(CASE_ID))
            acc = _accuracy_vs_ref(ref_case, f_ref, case, f)
            gh = _ghia_metrics(case, f)
            results[method] = {
                "case": case,
                "method": method,
                "field": f,
                "history": hist,
                "history_norm": _normalize_history_for_plot(hist),
                "wall": wall,
                "tol": TOL,
                "accuracy": acc,
                "ghia": gh,
            }
        except Exception as exc:
            print(f"[warn] {method} failed: {exc}")
            case = case_factory()
            f = case.initial_field()
            acc = _accuracy_vs_ref(ref_case, f_ref, case, f)
            gh = _ghia_metrics(case, f)
            results[method] = {
                "case": case,
                "method": method,
                "field": f,
                "history": [(0, float("nan"), 0, 0.0)],
                "history_norm": _normalize_history_for_plot([(0, float("nan"), 0, 0.0)]),
                "wall": 0.0,
                "tol": TOL,
                "accuracy": acc,
                "ghia": gh,
            }

    # Write per-method artifacts
    for method, row in results.items():
        case = row["case"]
        f = row["field"]
        hist = row["history"]
        wall = row["wall"]
        _write_history(HIST_DIR / f"{CASE_ID}__{method}.csv", hist)
        _write_normalized_history(HIST_DIR / f"{CASE_ID}__{method}__normalized.csv", row["history_norm"])
        write_vtk(VTK_DIR / f"{CASE_ID}__{method}.vtk", case, f)
        _write_field_csvs(CASE_ID, method, case, f, rho_ref, ux_ref, uy_ref, ghia_ref)

        res_last = _safe_float(hist[-1][1]) if hist else float("inf")
        row["final_residual"] = res_last
        row["lbe_calls"] = int(hist[-1][2]) if hist else 0
        row["wall_seconds"] = _safe_float(hist[-1][3], 0.0) if hist else 0.0
        row["converged"] = int(np.isfinite(res_last) and res_last < 5.0 * TOL)
        row["final_wall_total"] = wall

    # write summary and metrics
    pic = results["picard_lbm"]
    prop = results.get("proposed")
    summary_rows = []
    for method in ["picard_lbm"] + [m for m in methods if m != "picard_lbm"]:
        row = results.get(method)
        if row is None:
            continue
        acc = row["accuracy"]
        gh = row["ghia"]
        summary_rows.append({
            "case_id": CASE_ID,
            "method": method,
            "tol": _safe_float(row["tol"]),
            "N": N,
            "final_residual": row["final_residual"],
            "lbe_calls": row["lbe_calls"],
            "wall_seconds": row["wall_seconds"],
            "converged": row["converged"],
            "rel_l2_vs_picard": acc["rel_l2_vs_picard"],
            "linf_vs_picard": acc["linf_vs_picard"],
            "rms_vs_picard": acc["rms_vs_picard"],
            "speed_rms": acc["speed_rms"],
            "speed_l2": acc["speed_l2"],
            "ghia_u_rms": gh["ghia_u_rms"],
            "ghia_v_rms": gh["ghia_v_rms"],
            "ghia_u_linf": gh["ghia_u_linf"],
            "ghia_v_linf": gh["ghia_v_linf"],
            "centerline_max": gh["centerline_max"],
        })

    _write_summary(OUT_ROOT / "summary.csv", summary_rows)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    metrics = {
        "goal": "cavity_re100_n33_1x",
        "case_id": CASE_ID,
        "case_label": CASE_LABEL,
        "case_count": 1,
        "method_count": len(methods),
        "wall_seconds_elapsed": time.perf_counter() - started,
        "reference": "Picard (tight fixed-point) for full-field; Ghia 1982 for centerlines",
        "convergence": {
            method: {
                "final_residual": row["final_residual"],
                "lbe_calls": row["lbe_calls"],
                "wall_seconds": row["wall_seconds"],
                "converged": bool(row["converged"]),
            }
            for method, row in results.items()
        },
        "proposed": {
            "lbe_speedup_vs_picard": float(pic["lbe_calls"] / max(results["proposed"]["lbe_calls"], 1))
            if "proposed" in results else float("nan"),
            "wall_speedup_vs_picard": float(pic["wall_seconds"] / max(results["proposed"]["wall_seconds"], 1e-12))
            if "proposed" in results else float("nan"),
        },
        "artifact_signature": {
            "summary_csv": hashlib.sha256((OUT_ROOT / "summary.csv").read_bytes()).hexdigest(),
            "config_signature": _sha256([
                Path("paper_60case_benchmark.py"),
                Path("lbm_core.py"),
                Path("ghia_validation.py"),
                Path("solver_anderson.py"),
                Path("solver_baseline.py"),
                Path("paper_faithful_baselines.py"),
                Path("solver_proposed_single.py"),
            ]),
        },
    }
    (OUT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # figures
    _plot_residual_vs_iteration(results)
    _plot_residual_vs_wall(results)
    _plot_ghia_profiles(results, ghia_ref)

    print(f"[saved] {OUT_ROOT / 'summary.csv'}")
    print(f"[saved] {OUT_ROOT / 'summary.json'}")
    print(f"[saved] {OUT_ROOT / 'metrics.json'}")
    print(f"[saved] {OUT_ROOT / 'figure'}/*.png")


def parse_methods(raw):
    if not raw:
        return METHODS
    out = []
    for token in raw.split(","):
        t = token.strip()
        if t and t in METHODS and t not in out:
            out.append(t)
    if "picard_lbm" not in out:
        out.insert(0, "picard_lbm")
    if "proposed" not in out and "proposed" in METHODS:
        out.append("proposed")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated methods")
    parser.add_argument("--no-clean", action="store_true", help="do not back up existing papers_data/lid_driven_Re100_N33")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    run_case(methods=methods, do_clean=not args.no_clean)


if __name__ == "__main__":
    main()
