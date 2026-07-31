#!/usr/bin/env python3
"""Run no-force Channel (N=32) benchmark and export papers-data artifacts.

- Boundary: periodic-x, wall-y (NoForceChannelCase)
- Physics: no forcing term (inlet/outlet BC only)
- Reference: analytic Poiseuille profile (parabolic), used for all method comparisons
- Outputs: papers_data/channel_n32__1x/
    - summary.csv / metrics.json
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
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from no_force_suite.no_force_cases import make_case, SUPPORTED_CASES
from paper_60case_benchmark_no_force import macro_of, run_one
from paper_60case_benchmark_no_force import write_history_csv, write_vtk

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

OUT_ROOT = Path("papers_data") / "channel_n32__1x"
FIELD_DIR = OUT_ROOT / "fields"
FIG_DIR = OUT_ROOT / "figure"
HIST_DIR = OUT_ROOT / "histories"
VTK_DIR = OUT_ROOT / "vtk"


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def backup_existing(out_root: Path) -> None:
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


def _analytic_poiseuille(case) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(case.N)
    U_in = float(getattr(case, "U_in", 0.05))
    y = np.arange(N, dtype=np.float64)
    L = float(N - 1)
    if L <= 0:
        raise ValueError("invalid channel size")
    # Match prior project convention used for no-force/analytic comparison.
    ubar = U_in * L / max(N, 1.0)
    ux = 6.0 * ubar * (y / L) * (1.0 - y / L)
    ux = ux[:, None] * np.ones((N, N), dtype=np.float64)
    uy = np.zeros((N, N), dtype=np.float64)
    rho = np.ones((N, N), dtype=np.float64)
    return rho, ux, uy


def _vorticity_from_velocity(ux: np.ndarray, uy: np.ndarray, dx: float = 1.0, dy: float = 1.0):
    du_y_dx = np.gradient(uy, axis=1, edge_order=1) / dx
    du_x_dy = np.gradient(ux, axis=0, edge_order=1) / dy
    return du_y_dx - du_x_dy


def _vector_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


def _compute_accuracy(case, f: np.ndarray, analytic: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    rho_ref, ux_ref, uy_ref = analytic
    rho, ux, uy = macro_of(case, f)
    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64)) > 0.0
    mask = chi

    dux = (ux - ux_ref)[mask]
    duy = (uy - uy_ref)[mask]
    den = np.sqrt(np.sum(ux_ref[mask] ** 2 + uy_ref[mask] ** 2))
    den = max(float(den), 1.0e-30)

    rel_l2 = _vector_norm(dux) if dux.size else 0.0
    rel_l2 = rel_l2 / den

    linf = float(max(np.max(np.abs(dux)) if dux.size else 0.0, np.max(np.abs(duy)) if duy.size else 0.0))
    rms = float(np.sqrt(np.mean(dux * dux + duy * duy)) if dux.size else 0.0)

    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    speed_err = np.abs(speed - speed_ref)[mask]
    speed_rms = float(np.sqrt(np.mean(speed_err * speed_err)) if speed_err.size else 0.0)
    speed_rel_l2 = _vector_norm(speed_err) / max(_vector_norm(speed_ref[mask]), 1.0e-30)

    return {
        "rel_l2_velocity": float(rel_l2),
        "linf_velocity": float(linf),
        "rms_velocity": float(rms),
        "speed_rms": float(speed_rms),
        "speed_rel_l2": float(speed_rel_l2),
        "mean_rho": float(np.mean(rho[mask])),
    }


def _write_history_csv(path: Path, hist):
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "residual", "lbe_calls", "wall_seconds", "wall_seconds_raw"])
        for row in hist:
            if row is None or len(row) < 4:
                continue
            it, res, lbe, wall = row[:4]
            wr.writerow([int(it), _safe_float(res), int(_safe_float(lbe, 0.0)), _safe_float(wall), _safe_float(wall)])


def _write_field_csvs(case_id: str, method: str, case, f: np.ndarray, analytic: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    rho_ref, ux_ref, uy_ref = analytic
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = _vorticity_from_velocity(ux, uy)
    vort_ref = _vorticity_from_velocity(ux_ref, uy_ref)
    chi = getattr(case, "chi", np.ones_like(ux, dtype=np.float64))
    mask = chi > 0.0

    ny, nx = ux.shape

    base = FIELD_DIR / f"{case_id}__{method}"

    with (base.with_name(f"{base.name}__macro.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "rho", "ux", "uy", "speed", "chi"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                wr.writerow([j, i, float(i), y, rho[j, i], ux[j, i], uy[j, i], speed[j, i], float(chi[j, i])])

    with (base.with_name(f"{base.name}__vorticity.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "vorticity", "vorticity_ref", "vorticity_error", "chi"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                if mask[j, i]:
                    ve = vort[j, i] - vort_ref[j, i]
                else:
                    ve = float("nan")
                wr.writerow([j, i, float(i), y, vort[j, i], vort_ref[j, i], ve, float(chi[j, i])])

    with (base.with_name(f"{base.name}__error.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iy",
            "ix",
            "x",
            "y",
            "ux_err",
            "uy_err",
            "speed_err",
            "speed_ref",
            "speed",
            "vort_err",
            "chi",
        ])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                if mask[j, i]:
                    ux_err = ux[j, i] - ux_ref[j, i]
                    uy_err = uy[j, i] - uy_ref[j, i]
                    sp_err = speed[j, i] - speed_ref[j, i]
                    vo_err = vort[j, i] - vort_ref[j, i]
                else:
                    ux_err = float("nan")
                    uy_err = float("nan")
                    sp_err = float("nan")
                    vo_err = float("nan")
                wr.writerow([j, i, float(i), y, ux_err, uy_err, sp_err, speed_ref[j, i], speed[j, i], vo_err, float(chi[j, i])])

    with (base.with_name(f"{base.name}__analytic_error.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "x", "y", "ux_err", "uy_err", "speed_err", "vort_err", "mask"])
        for j in range(ny):
            y = float(j)
            for i in range(nx):
                if mask[j, i]:
                    ux_err = ux[j, i] - ux_ref[j, i]
                    uy_err = uy[j, i] - uy_ref[j, i]
                    sp_err = speed[j, i] - speed_ref[j, i]
                    vo_err = vort[j, i] - vort_ref[j, i]
                    m = 1
                else:
                    ux_err = float("nan")
                    uy_err = float("nan")
                    sp_err = float("nan")
                    vo_err = float("nan")
                    m = 0
                wr.writerow([j, i, float(i), y, ux_err, uy_err, sp_err, vo_err, m])

    with (base.with_name(f"{base.name}__centerline_x.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "uy", "uy_ref", "uy_err", "ux", "ux_ref", "ux_err", "speed", "speed_ref", "speed_err"])
        mid_y = ny // 2
        for i in range(nx):
            wr.writerow([
                i,
                uy[mid_y, i],
                uy_ref[mid_y, i],
                uy[mid_y, i] - uy_ref[mid_y, i],
                ux[mid_y, i],
                ux_ref[mid_y, i],
                ux[mid_y, i] - ux_ref[mid_y, i],
                speed[mid_y, i],
                speed_ref[mid_y, i],
                speed[mid_y, i] - speed_ref[mid_y, i],
            ])

    with (base.with_name(f"{base.name}__centerline_y.csv")).open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["y", "uy", "uy_ref", "uy_err", "ux", "ux_ref", "ux_err", "speed", "speed_ref", "speed_err"])
        mid_x = nx // 2
        for j in range(ny):
            wr.writerow([
                j,
                uy[j, mid_x],
                uy_ref[j, mid_x],
                uy[j, mid_x] - uy_ref[j, mid_x],
                ux[j, mid_x],
                ux_ref[j, mid_x],
                ux[j, mid_x] - ux_ref[j, mid_x],
                speed[j, mid_x],
                speed_ref[j, mid_x],
                speed[j, mid_x] - speed_ref[j, mid_x],
            ])


def _plot_residual_vs_iteration(result_rows):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for method in METHODS:
        row = result_rows.get(method)
        if not row:
            continue
        hist = row["history"]
        if not hist:
            continue
        it = [int(r[0]) for r in hist]
        res = [max(_safe_float(r[1], float("nan")), 1.0e-20) for r in hist]
        ax.plot(it, res, label=METHOD_LABELS.get(method, method))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual norm")
    ax.set_title("channel_n32__1x residual convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.savefig(FIG_DIR / "residual_vs_iteration.png", dpi=220)
    plt.close(fig)


def _plot_residual_vs_wall(result_rows):
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for method in METHODS:
        row = result_rows.get(method)
        if not row:
            continue
        hist = row["history"]
        if not hist:
            continue
        wall = [max(_safe_float(r[3], 0.0), 1.0e-12) for r in hist]
        res = [max(_safe_float(r[1], float("nan")), 1.0e-20) for r in hist]
        ax.plot(wall, res, label=METHOD_LABELS.get(method, method))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall seconds")
    ax.set_ylabel("residual norm")
    ax.set_title("channel_n32__1x residual vs wall time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.savefig(FIG_DIR / "residual_vs_wall_seconds.png", dpi=220)
    plt.close(fig)


def _plot_accuracy_compare(case_id: str, case, results, analytic):
    rho_ref, ux_ref, uy_ref = analytic
    ny = int(case.N)
    y = np.arange(ny, dtype=np.float64)
    mean_ux_ref = ux_ref.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    # x-축: velocity, y-축: y 좌표.
    # 이전에는 축이 반대로 들어가 실제 값이 시각적으로 어색하게 보였으므로
    # 분석 기준선과 방법별 선의 축 정의를 실제 물리량(ux(y))와 맞춰 고정한다.
    ax.plot(y, mean_ux_ref, "k--", lw=2.0, label="analytic")
    for method in METHODS:
        row = results.get(method)
        if not row:
            continue
        rho, ux, uy = macro_of(case, row["f"])
        mean_ux = ux.mean(axis=1)
        ax.plot(y, mean_ux, lw=1.4, label=METHOD_LABELS.get(method, method))

    ax.set_xlabel("y index")
    ax.set_ylabel("x-averaged u_x")
    ax.set_title(f"{case_id} mean streamwise profile")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.savefig(FIG_DIR / "accuracy_profile_xavg_ux.png", dpi=220)
    plt.close(fig)

    # Speed/vorticity/error contours per method
    for method in METHODS:
        row = results.get(method)
        if not row:
            continue
        f = row["f"]
        rho, ux, uy = macro_of(case, f)
        speed = np.sqrt(ux * ux + uy * uy)
        vort = _vorticity_from_velocity(ux, uy)
        chi = getattr(case, "chi", np.ones_like(speed, dtype=np.float64)) > 0.0

        fig2, ax2 = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
        c = ax2.imshow(np.ma.array(speed, mask=~chi), origin="lower", cmap="viridis")
        fig2.colorbar(c, ax=ax2, label="|u|")
        ax2.set_title(f"{METHOD_LABELS.get(method, method)} speed")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig2.savefig(FIG_DIR / f"speed_contour__{method}.png", dpi=200)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
        c3 = ax3.imshow(np.ma.array(vort, mask=~chi), origin="lower", cmap="coolwarm")
        fig3.colorbar(c3, ax=ax3, label="vorticity")
        ax3.set_title(f"{METHOD_LABELS.get(method, method)} vorticity")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        fig3.savefig(FIG_DIR / f"vorticity_contour__{method}.png", dpi=200)
        plt.close(fig3)

        err = speed - np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
        fig4, ax4 = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
        c4 = ax4.imshow(np.ma.array(err, mask=~chi), origin="lower", cmap="bwr")
        fig4.colorbar(c4, ax=ax4, label="speed error")
        ax4.set_title(f"{METHOD_LABELS.get(method, method)} speed error vs analytic")
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        fig4.savefig(FIG_DIR / f"speed_error_contour__{method}.png", dpi=200)
        plt.close(fig4)


def _write_summary_csv(path: Path, summary_rows):
    fields = [
        "case_id",
        "method",
        "tol",
        "N",
        "final_residual",
        "lbe_calls",
        "wall_seconds",
        "converged",
        "rel_l2_velocity",
        "linf_velocity",
        "rms_velocity",
        "speed_rms",
        "speed_rel_l2",
        "mean_rho",
        "wall_last",
    ]
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in summary_rows:
            wr.writerow({k: row.get(k, "") for k in fields})


def _sha256_of_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        path = Path(p)
        if not path.exists():
            continue
        h.update(path.as_posix().encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def run_case(case_id: str = "channel_n32", methods=None, no_cache: bool = False, clean: bool = True):
    if methods is None:
        methods = METHODS

    if case_id not in SUPPORTED_CASES:
        raise ValueError(f"unsupported case_id {case_id}")

    if clean:
        backup_existing(OUT_ROOT)

    _ensure_dir(FIELD_DIR)
    _ensure_dir(FIG_DIR)
    _ensure_dir(HIST_DIR)
    _ensure_dir(VTK_DIR)

    started = time.perf_counter()

    # Reference run (Picard) plus all requested methods.
    results = {}

    # Ensure case metadata and analytic reference are fixed.
    ref_case = make_case(case_id)
    analytic = _analytic_poiseuille(ref_case)

    for method in methods:
        try:
            case, _, label, tol, f, hist, wall = run_one(case_id, method, use_cache=not no_cache)
            _ensure_dir(HIST_DIR)
            _write_history_csv(HIST_DIR / f"{case_id}__{method}.csv", hist)
            write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, f)
            _write_field_csvs(case_id, method, case, f, analytic)

            res_last = _safe_float(hist[-1][1]) if hist else float("inf")
            lbe_last = int(hist[-1][2]) if hist else 0
            wall_last = _safe_float(hist[-1][3]) if hist else 0.0
            acc = _compute_accuracy(case, f, analytic)

            converged = int(np.isfinite(res_last) and res_last < 5.0 * tol)
            results[method] = {
                "case": case,
                "label": label,
                "tol": tol,
                "f": f,
                "history": hist,
                "wall": wall,
                "final_residual": res_last,
                "lbe_calls": lbe_last,
                "wall_seconds": wall_last,
                "converged": converged,
                "accuracy": acc,
            }

            print(
                f"[{method}] iter={len(hist):4d}  res={res_last:.3e} "
                f"lbe={lbe_last:8d} wall={wall_last:8.3f}s conv={converged}"
            )
        except Exception as exc:
            results[method] = None
            print(f"[{method}] failed: {exc}")

    # write summary
    summary_rows = []
    for method in methods:
        row = results.get(method)
        if row is None:
            summary_rows.append({
                "case_id": case_id,
                "method": method,
                "tol": float("nan"),
                "N": int(ref_case.N),
                "final_residual": float("inf"),
                "lbe_calls": 0,
                "wall_seconds": 0.0,
                "converged": 0,
                "rel_l2_velocity": float("inf"),
                "linf_velocity": float("inf"),
                "rms_velocity": float("inf"),
                "speed_rms": float("inf"),
                "speed_rel_l2": float("inf"),
                "mean_rho": 0.0,
                "wall_last": 0.0,
            })
            continue

        acc = row["accuracy"]
        summary_rows.append({
            "case_id": case_id,
            "method": method,
            "tol": _safe_float(row["tol"]),
            "N": int(row["case"].N),
            "final_residual": _safe_float(row["final_residual"]),
            "lbe_calls": int(row["lbe_calls"]),
            "wall_seconds": _safe_float(row["wall_seconds"]),
            "converged": int(row["converged"]),
            "rel_l2_velocity": _safe_float(acc["rel_l2_velocity"]),
            "linf_velocity": _safe_float(acc["linf_velocity"]),
            "rms_velocity": _safe_float(acc["rms_velocity"]),
            "speed_rms": _safe_float(acc["speed_rms"]),
            "speed_rel_l2": _safe_float(acc["speed_rel_l2"]),
            "mean_rho": _safe_float(acc["mean_rho"]),
            "wall_last": _safe_float(row["wall_seconds"]),
        })

    _write_summary_csv(OUT_ROOT / "summary.csv", summary_rows)

    # metrics JSON
    prop = results.get("proposed")
    pic = results.get("picard_lbm")
    metrics = {
    "goal": "channel_n32__1x no-force periodic-x + wall-y benchmark",
        "case_id": case_id,
        "case_count": 1,
        "method_count": len(methods),
        "wall_seconds": time.perf_counter() - started,
        "use_cache": bool(not no_cache),
        "ref_label": "analytic Poiseuille",
        "wall_reference": "periodic x + wall y",
        "convergence": {
            method: {
                "final_residual": row["final_residual"],
                "lbe_calls": row["lbe_calls"],
                "wall_seconds": row["wall_seconds"],
                "converged": bool(row["converged"]),
            }
            for method, row in results.items() if row is not None
        },
        "exact_metrics": {
            method: row["accuracy"] for method, row in results.items() if row is not None
        },
        "artifact_hash": {
            "summary_csv": hashlib.sha256((OUT_ROOT / "summary.csv").read_bytes()).hexdigest(),
            "config_signature": _sha256_of_files([
                Path("paper_60case_benchmark_no_force.py"),
                Path("no_force_suite/no_force_cases.py"),
                Path("no_force_suite/no_force_lb_core.py"),
                Path("solver_anderson.py"),
                Path("solver_baseline.py"),
                Path("solver_proposed_single.py"),
                Path("paper_faithful_baselines.py"),
            ]),
        },
    }
    OUT_ROOT.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Figures and exports for paper-ready comparison
    _plot_residual_vs_iteration(results)
    _plot_residual_vs_wall(results)
    _plot_accuracy_compare(case_id, ref_case, results, analytic)

    if prop is not None and pic is not None:
        with (OUT_ROOT / "summary_accuracy.csv").open("w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["method", "rel_l2_velocity", "linf_velocity", "rms_velocity", "speed_rms", "speed_rel_l2", "lbe_calls", "wall_seconds", "final_residual", "converged"])
            for method in methods:
                row = results.get(method)
                if row is None:
                    continue
                acc = row["accuracy"]
                wr.writerow([
                    method,
                    acc["rel_l2_velocity"],
                    acc["linf_velocity"],
                    acc["rms_velocity"],
                    acc["speed_rms"],
                    acc["speed_rel_l2"],
                    row["lbe_calls"],
                    row["wall_seconds"],
                    row["final_residual"],
                    int(row["converged"]),
                ])

    print(f"[saved] {OUT_ROOT / 'summary.csv'}")
    print(f"[saved] {OUT_ROOT / 'metrics.json'}")
    print(f"[saved] {OUT_ROOT / 'figure'}/*.png")


def parse_methods(raw):
    if not raw:
        return METHODS
    out = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(t)
    if "picard_lbm" not in out:
        out.insert(0, "picard_lbm")
    if "proposed" not in out:
        out.append("proposed")
    return [m for m in out if m in METHODS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="channel_n32", help="no-force case id")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma separated methods")
    parser.add_argument("--no-cache", action="store_true", help="ignore previous solver cache")
    parser.add_argument("--clean", action="store_true", help="backup existing papers_data/channel_n32__1x and regenerate")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    run_case(case_id=args.case, methods=methods, no_cache=args.no_cache, clean=args.clean)


if __name__ == "__main__":
    main()
