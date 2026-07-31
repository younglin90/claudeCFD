#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghia_validation import get_ghia_data
from paper_60case_benchmark_no_force import macro_of, write_history_csv, write_vtk
from paper_60case_benchmark_no_force_scaling import (
    _f_rms_residual_value,
    _macro_l2_residual_components,
    case_factory_scaled,
    run_method_with_wall,
)


def _history_rows(hist):
    rows = []
    for row in hist:
        if len(row) < 4:
            continue
        rows.append(
            {
                "iter": int(row[0]),
                "residual": float(row[1]),
                "lbe_calls": int(row[2]),
                "wall_seconds": float(row[3]),
                "residual_kind": "macro_l2_p_ux_uy_uz",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def _plot_history(case_id: str, rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("residual_vs_iteration.png", "iter", "Iteration"),
        ("residual_vs_lbe_calls.png", "lbe_calls", "LBE calls"),
        ("residual_vs_wall_seconds.png", "wall_seconds", "Wall seconds"),
    ]
    for filename, xkey, xlabel in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        if xkey == "iter":
            x = np.array([1.0 if int(r[xkey]) == 0 else float(r[xkey]) for r in rows], dtype=np.float64)
        elif xkey == "wall_seconds":
            x = np.array([max(1.0e-6, float(r[xkey])) for r in rows], dtype=np.float64)
        else:
            x = np.array([max(1.0, float(r[xkey])) for r in rows], dtype=np.float64)
        y = np.array([max(1.0e-16, float(r["residual"])) for r in rows], dtype=np.float64)
        ax.plot(x, y, lw=1.6, label="SafeNN proposed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("macro L2 residual")
        ax.set_title(case_id)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)
    if rows and "relative_macro_l2_residual" in rows[0]:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        x = np.array([max(1.0, float(r["lbe_calls"])) for r in rows], dtype=np.float64)
        y = np.array([max(1.0e-16, float(r["relative_macro_l2_residual"])) for r in rows], dtype=np.float64)
        ax.plot(x, y, lw=1.6, label="SafeNN proposed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("relative macro L2 residual")
        ax.set_title(case_id)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "relative_macro_l2_residual_vs_lbe_calls.png", dpi=220)
        plt.close(fig)


def _macro_vector(case, f):
    rho, ux, uy = macro_of(case, f)
    pressure = rho / 3.0
    return np.concatenate([pressure.ravel(), ux.ravel(), uy.ravel()])


def _relative_plateau_stats(rel_values: list[float], window: int):
    if len(rel_values) < window:
        return False, float("nan")
    tail = np.asarray(rel_values[-window:], dtype=np.float64)
    half = max(window // 2, 1)
    old = float(np.median(tail[:half]))
    new = float(np.median(tail[half:]))
    improvement = (old - new) / max(old, 1.0e-300)
    return bool(np.isfinite(improvement) and improvement <= 0.05), float(improvement)


def _relative_macro_l2_plateau_tail(
    case,
    f,
    hist,
    *,
    r0: float,
    min_lbe: int = 100000,
    max_lbe: int = 1000000,
    check_every: int = 1000,
    window: int = 50,
    eps_plateau: float = 0.05,
    eps_macro_change: float = 1.0e-4,
):
    rows = _history_rows(hist)
    lbe = int(rows[-1]["lbe_calls"]) if rows else 0
    t0_wall = float(rows[-1]["wall_seconds"]) if rows else 0.0
    t0 = time.perf_counter() - t0_wall
    rel_values = []
    macro_snapshots = []
    last_stats = {
        "relative_macro_l2": float("nan"),
        "plateau_improvement": float("nan"),
        "macro_change": float("nan"),
        "relative_plateau": 0,
        "macro_change_pass": 0,
        "min_lbe_pass": 0,
    }

    def record(iter_id, state, calls):
        rn = float(_macro_l2_residual_components(case, state)[0])
        rel = rn / max(float(r0), 1.0e-300)
        vec = _macro_vector(case, state)
        rel_values.append(float(rel))
        macro_snapshots.append(vec)
        if len(macro_snapshots) > window + 1:
            macro_snapshots.pop(0)
        plateau, improvement = _relative_plateau_stats(rel_values, window)
        if len(macro_snapshots) >= window:
            prev = macro_snapshots[0]
            cur = macro_snapshots[-1]
            macro_change = float(np.linalg.norm(cur - prev) / max(np.linalg.norm(cur), 1.0e-300))
        else:
            macro_change = float("inf")
        wall_now = time.perf_counter() - t0
        hist.append((int(iter_id), rn, int(calls), wall_now))
        last_stats.update(
            {
                "relative_macro_l2": float(rel),
                "plateau_improvement": float(improvement),
                "macro_change": float(macro_change),
                "relative_plateau": int(plateau),
                "macro_change_pass": int(np.isfinite(macro_change) and macro_change <= eps_macro_change),
                "min_lbe_pass": int(calls >= min_lbe),
            }
        )
        converged = bool(
            calls >= min_lbe
            and plateau
            and np.isfinite(macro_change)
            and macro_change <= eps_macro_change
        )
        return converged

    record(len(hist), f, lbe)
    step = 0
    while lbe < max_lbe:
        for _ in range(check_every):
            f = case.lbe_step(f)
        step += check_every
        lbe += check_every
        if record(len(hist), f, lbe):
            break
    return f, hist, last_stats


def _write_field_csv(path: Path, case, f):
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "rho", "pressure", "ux", "uy", "uz", "speed"])
        ny, nx = ux.shape
        for iy in range(ny):
            for ix in range(nx):
                wr.writerow([iy, ix, rho[iy, ix], rho[iy, ix] / 3.0, ux[iy, ix], uy[iy, ix], 0.0, speed[iy, ix]])


def _write_ghia(case_id: str, case, f, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rho, ux, uy = macro_of(case, f)
    re = int(getattr(case, "Re", 1000))
    y_g, u_g, x_g, v_g = get_ghia_data(re)
    n = int(getattr(case, "N"))
    grid = np.linspace(0.0, 1.0, n)
    mid = n // 2
    u_wall = max(float(getattr(case, "U_wall", 1.0)), 1.0e-30)
    u_line = ux[:, mid] / u_wall
    v_line = uy[mid, :] / u_wall
    u_i = np.interp(y_g, grid, u_line)
    v_i = np.interp(x_g, grid, v_line)
    du = u_i - u_g
    dv = v_i - v_g
    err = {
        "case_id": case_id,
        "method": "proposed",
        "u_centerline_rms": float(np.sqrt(np.mean(du * du))),
        "v_centerline_rms": float(np.sqrt(np.mean(dv * dv))),
        "u_centerline_linf": float(np.max(np.abs(du))),
        "v_centerline_linf": float(np.max(np.abs(dv))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].plot(u_g, y_g, "ko", ms=3.5, label="Ghia et al.")
    axes[0].plot(u_line, grid, lw=1.8, label="SafeNN proposed")
    axes[0].set_xlabel("u / U")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"{case_id}: vertical centerline")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(x_g, v_g, "ko", ms=3.5, label="Ghia et al.")
    axes[1].plot(grid, v_line, lw=1.8, label="SafeNN proposed")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v / U")
    axes[1].set_title(f"{case_id}: horizontal centerline")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_dir / "ghia_centerline_comparison.png", dpi=220)
    plt.close(fig)
    with (out_dir.parent / "ghia_centerline_error.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(err.keys()))
        wr.writeheader()
        wr.writerow(err)
    return err


def _save_level(level: int, out_root: Path, papers_root: Path):
    case_id, label, tol, factory = case_factory_scaled("cavity_re1000_n129", level)
    case = factory()
    level_dir = out_root / f"level_{level}x"
    paper_dir = papers_root / case_id
    for d in (level_dir, paper_dir):
        d.mkdir(parents=True, exist_ok=True)

    r0 = float(_macro_l2_residual_components(case, case.initial_field())[0])
    t0 = time.perf_counter()
    f, hist, wall = run_method_with_wall("proposed", case, tol=tol, max_steps=0, verbose=False)
    f, hist, plateau_stats = _relative_macro_l2_plateau_tail(case, f, hist, r0=r0)
    elapsed = time.perf_counter() - t0
    rows = _history_rows(hist)
    macro_total, macro_p, macro_ux, macro_uy, macro_uz = _macro_l2_residual_components(case, f)
    f_rms = _f_rms_residual_value(case, f)
    lbe = int(rows[-1]["lbe_calls"]) if rows else 0
    wall_final = float(rows[-1]["wall_seconds"]) if rows else float(wall)
    threshold = 5.0 * float(tol)
    relative_macro_l2 = float(macro_total / max(r0, 1.0e-300))
    converged = int(
        plateau_stats["min_lbe_pass"]
        and plateau_stats["relative_plateau"]
        and plateau_stats["macro_change_pass"]
    )

    if rows:
        rows[-1]["residual"] = float(macro_total)
        rows[-1]["wall_seconds"] = float(wall_final)
    for row in rows:
        row["relative_macro_l2_residual"] = float(row["residual"] / max(r0, 1.0e-300))

    summary = {
        "base_case_id": "cavity_re1000_n129",
        "case_id": case_id,
        "case_label": label,
        "scaling_level": int(level),
        "method": "proposed",
        "tol": float(tol),
        "residual_threshold": float(threshold),
        "initial_macro_l2_residual": float(r0),
        "relative_macro_l2_residual": float(relative_macro_l2),
        "relative_plateau_window": 50,
        "relative_plateau_eps": 0.05,
        "macro_change_eps": 1.0e-4,
        "min_lbe_calls": 100000,
        "plateau_improvement": float(plateau_stats["plateau_improvement"]),
        "macro_change": float(plateau_stats["macro_change"]),
        "relative_plateau": int(plateau_stats["relative_plateau"]),
        "macro_change_pass": int(plateau_stats["macro_change_pass"]),
        "min_lbe_pass": int(plateau_stats["min_lbe_pass"]),
        "final_residual_kind": "macro_l2_p_ux_uy_uz",
        "final_residual": float(macro_total),
        "final_macro_l2_residual": float(macro_total),
        "final_macro_l2_pressure": float(macro_p),
        "final_macro_l2_ux": float(macro_ux),
        "final_macro_l2_uy": float(macro_uy),
        "final_macro_l2_uz": float(macro_uz),
        "final_f_rms_residual": float(f_rms),
        "lbe_calls": int(lbe),
        "wall_seconds": float(wall_final),
        "elapsed_wall_seconds": float(elapsed),
        "residual_converged": int(converged),
        "converged": int(converged),
    }

    np.savez_compressed(level_dir / "field_and_history.npz", f=np.asarray(f), hist=np.asarray(hist, dtype=np.float64), wall=float(wall))
    _write_csv(level_dir / "summary.csv", [summary])
    _write_csv(level_dir / "history.csv", rows)
    _write_csv(level_dir / "residual_summary.csv", [summary])
    _write_field_csv(level_dir / "field_macro.csv", case, f)
    write_history_csv(level_dir / "benchmark_style_history.csv", hist)
    try:
        write_vtk(level_dir / "field.vtk", case, f)
    except Exception:
        pass
    _plot_history(case_id, rows, level_dir / "figure")
    ghia_error = _write_ghia(case_id, case, f, level_dir / "figure")
    summary.update(ghia_error)
    _write_csv(level_dir / "summary.csv", [summary])
    _write_csv(level_dir / "residual_summary.csv", [summary])
    (level_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if paper_dir.exists():
        shutil.rmtree(paper_dir)
    shutil.copytree(level_dir, paper_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--out-root", default="paper_revision_data/_coord_round97_macro_l2_cavity_re1000_proposed_direct")
    parser.add_argument("--papers-root", default="papers_data/proposed_macro_l2_cavity_re1000_direct")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    out_root = Path(args.out_root)
    papers_root = Path(args.papers_root)
    out_root.mkdir(parents=True, exist_ok=True)
    papers_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for level in levels:
        print(f"[start] cavity Re1000 {level}x", flush=True)
        row = _save_level(level, out_root, papers_root)
        all_rows.append(row)
        _write_csv(out_root / "summary_all_completed_so_far.csv", all_rows)
        print(
            f"[saved] {row['case_id']} lbe={row['lbe_calls']} "
            f"macro_l2={row['final_residual']:.3e} f_rms={row['final_f_rms_residual']:.3e} "
            f"conv={row['converged']}",
            flush=True,
        )
    _write_csv(out_root / "summary.csv", all_rows)


if __name__ == "__main__":
    main()
