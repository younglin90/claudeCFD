"""Run every acceleration method on the paper-faithful PLBE Re=1000 cavity,
using the paper's velocity-change convergence criterion.

All methods share the same PLBECavity (Guo-Zhao-Shi EDF, NEQ-extrapolation BC,
M*=0.058, gamma=0.0625, U_wall=0.0084, N=129). Each method's accelerator runs
first to drive the native residual down, then a warm-started PLBE Picard tail
finishes the steady-state to ||du||_2 / ||u||_2 over 100 steps < 1e-6. We then
score each final field against Ghia 1982 centerline data.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from ghia_validation import get_ghia_data
from lbm_plbe_cavity import PLBECavity
from lbm_periodic import build_spectral_schur
from paper_60case_benchmark import (
    solve_inexact_newton_identity,
    write_history_csv,
    write_vtk,
)
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_scmk import solve_scmk
from solver_scmk_direct import solve_scmk_direct
from solver_scmk_mg import solve_scmk_mg
from solver_unified_safe_nn import _residual_norm
from verify_plbe_re1000 import run_plbe, velocity_change_norm


OUT = Path("paper_revision_data/plbe_re1000_paper_mach/all_methods")
OUT.mkdir(parents=True, exist_ok=True)


def make_case() -> PLBECavity:
    mach_star = 0.058
    ratio = 0.25
    gamma = ratio * ratio
    mach = mach_star * ratio
    u_wall = mach / math.sqrt(3.0)
    return PLBECavity(N=129, Re=1000, U_wall=u_wall, gamma=gamma)


def ghia_metrics(case: PLBECavity, f: np.ndarray) -> dict:
    _, ux, uy = case.macro(f)
    y_g, u_g, x_g, v_g = get_ghia_data(1000)
    n = case.N
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    u_center = ux[:, n // 2] / case.U_wall
    v_center = uy[n // 2, :] / case.U_wall
    u_interp = np.interp(y_g, y, u_center)
    v_interp = np.interp(x_g, x, v_center)
    return {
        "u_rms": float(np.sqrt(np.mean((u_interp - u_g) ** 2))),
        "v_rms": float(np.sqrt(np.mean((v_interp - v_g) ** 2))),
        "u_linf": float(np.max(np.abs(u_interp - u_g))),
        "v_linf": float(np.max(np.abs(v_interp - v_g))),
        "centerline_max": float(max(np.max(np.abs(u_interp - u_g)), np.max(np.abs(v_interp - v_g)))),
    }


def verdict(metrics: dict) -> str:
    if not np.isfinite(metrics["centerline_max"]):
        return "DIVERGED"
    if metrics["centerline_max"] < 0.05:
        return "YES"
    if metrics["centerline_max"] < 0.10:
        return "MARGINAL"
    return "NO"


def picard_tail(case: PLBECavity, f: np.ndarray, max_steps: int, tol: float, check_every: int):
    """Warm-started PLBE Picard with the paper's velocity-change criterion."""
    f = f.copy()
    prev = f.copy()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    last_change = float("inf")
    for step in range(1, max_steps + 1):
        f = case.lbe_step(f)
        lbe += 1
        if not np.all(np.isfinite(f)):
            hist.append((step, float("nan"), lbe, time.perf_counter() - t0))
            return f, hist, float("nan")
        if step % check_every == 0:
            last_change = velocity_change_norm(case, f, prev)
            hist.append((step, last_change, lbe, time.perf_counter() - t0))
            if last_change < tol:
                break
            prev = f.copy()
    return f, hist, last_change


def run_accelerator(method: str, case: PLBECavity, tol_native: float, budget: int):
    if method == "picard_lbm":
        # Single-stage paper Picard with velocity-change criterion.
        f, hist = run_plbe(case, max_steps=budget, tol=1.0e-6, check_every=100, verbose=False)
        return f, hist
    if method == "anderson_lbm":
        return solve_anderson(
            case, max_iter=budget // 2, tol=tol_native, m=5, beta=0.8,
            safeguard=True, verbose=False, check_every=5,
        )
    if method == "preconditioned_lbm":
        return solve_scmk_direct(
            case, max_outer=400, tol=tol_native, kinetic_substeps=40,
            line_search_max=4, anderson_m=3, anderson_beta=0.8, verbose=False,
        )
    if method == "inexact_newton_lbe":
        return solve_inexact_newton_identity(
            case, max_outer=400, tol=tol_native, krylov_max=10,
            kinetic_substeps=40, verbose=False,
        )
    if method == "dual_time_mg_lbm":
        s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
        return solve_scmk_mg(
            case, s_inv, max_outer=400, tol=tol_native, krylov_max=10,
            K_smooth=20, K_post=20, line_search_max=3, verbose=False,
        )
    if method == "proposed_safenn_scmk":
        s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
        return solve_scmk(
            case, s_inv, max_outer=500, tol=tol_native, krylov_max=10,
            krylov_tol=1e-3, kinetic_substeps=40,
            line_search_max=5, verbose=False,
        )
    raise ValueError(method)


def run_method(method: str, case: PLBECavity, tol_native: float, budget: int):
    t0 = time.perf_counter()
    try:
        f, hist = run_accelerator(method, case, tol_native, budget)
        accel_lbe = int(hist[-1][2]) if hist else 0
        # Tail Picard with paper criterion (skip if picard_lbm already used it).
        if method == "picard_lbm":
            tail_lbe = 0
            tail_change = float(hist[-1][1]) if hist else float("nan")
        elif f is None or not np.all(np.isfinite(f)):
            tail_lbe = 0
            tail_change = float("nan")
        else:
            f, tail_hist, tail_change = picard_tail(
                case, f, max_steps=budget, tol=1.0e-6, check_every=100,
            )
            for s, ch, lb, w in tail_hist:
                hist.append((s, ch, accel_lbe + lb, w))
            tail_lbe = (tail_hist[-1][2] if tail_hist else 0)
        wall = time.perf_counter() - t0
        if f is None or not np.all(np.isfinite(f)):
            return None, hist, wall, accel_lbe, tail_lbe, tail_change, "non-finite field"
        return f, hist, wall, accel_lbe, tail_lbe, tail_change, None
    except Exception as exc:
        traceback.print_exc()
        return None, [], time.perf_counter() - t0, 0, 0, float("nan"), f"{type(exc).__name__}: {exc}"


def main():
    case = make_case()
    print(f"PLBE case: N={case.N} Re={case.Re} U_wall={case.U_wall:.6e} gamma={case.gamma} "
          f"tau={case.tau:.6f} omega={case.omega:.6f}", flush=True)
    tol_native = 1.0e-7
    budget = 300000
    methods = [
        "picard_lbm",
        "anderson_lbm",
        "preconditioned_lbm",
        "inexact_newton_lbe",
        "dual_time_mg_lbm",
        "proposed_safenn_scmk",
    ]
    rows = []
    for m in methods:
        print(f"\n=== {m} ===", flush=True)
        f, hist, wall, accel_lbe, tail_lbe, tail_change, err = run_method(
            m, make_case(), tol_native, budget,
        )
        row = {
            "method": m,
            "wall_seconds": wall,
            "accelerator_lbe_calls": accel_lbe,
            "tail_picard_lbe_calls": tail_lbe,
            "total_lbe_calls": accel_lbe + tail_lbe,
            "tail_velocity_change_100": tail_change,
            "error": err,
        }
        if f is not None:
            _, native_res = _residual_norm(case, f)
            row["native_residual"] = float(native_res)
            row["paper_converged"] = bool(np.isfinite(tail_change) and tail_change < 1.0e-6)
            gm = ghia_metrics(case, f)
            row.update(gm)
            row["verdict"] = verdict(gm)
            write_vtk(OUT / f"{m}_field.vtk", case, f)
            if hist:
                write_history_csv(OUT / f"{m}_history.csv", hist)
        else:
            row["paper_converged"] = False
            row["verdict"] = "DIVERGED"
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    summary = {
        "case": {
            "N": case.N, "Re": case.Re, "U_wall": case.U_wall,
            "gamma": case.gamma, "tau": case.tau, "omega": case.omega,
            "mach_star": 0.058, "mach": case.U_wall * math.sqrt(3.0),
        },
        "tol_native": tol_native,
        "tol_paper_velocity_change": 1.0e-6,
        "budget_steps": budget,
        "rows": rows,
    }
    (OUT / "all_methods_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Paper-faithful PLBE Re=1000 cavity: every method scored vs Ghia",
        "",
        f"PLBECavity N={case.N} Re={case.Re} M*=0.058 gamma={case.gamma} "
        f"U_wall={case.U_wall:.4e} tau={case.tau:.4f} omega={case.omega:.4f}.",
        "Each method = (accelerator) -> (PLBE Picard tail until paper criterion).",
        "Convergence criterion: ||du||_2/||u||_2 over 100 steps < 1e-6.",
        "",
        "| Method | accel LBE | tail LBE | total LBE | wall (s) | native res | tail vel-chg | u_rms | v_rms | max dev | converged | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['accelerator_lbe_calls']} | {r['tail_picard_lbe_calls']} | "
            f"{r['total_lbe_calls']} | {r['wall_seconds']:.1f} | "
            f"{r.get('native_residual',float('nan')):.3e} | "
            f"{r['tail_velocity_change_100']:.3e} | "
            f"{r.get('u_rms',float('nan')):.3e} | {r.get('v_rms',float('nan')):.3e} | "
            f"{r.get('centerline_max',float('nan')):.3e} | "
            f"{'YES' if r['paper_converged'] else 'NO'} | "
            f"{r['verdict']} |"
        )
    lines += [
        "",
        "Verdict thresholds (Ghia 1982 centerline): YES if max dev < 0.05,",
        "MARGINAL < 0.10, otherwise NO. DIVERGED = non-finite field.",
        "",
        "Methods differ only in the steady-state acceleration wrapper. The",
        "underlying kinetic step is always the Guo-Zhao-Shi PLBE with unknown-only",
        "nonequilibrium extrapolation walls. The PLBE Picard tail is applied to",
        "every accelerator output so that the paper's velocity-change criterion",
        "is the actual stopping rule for every method, not a private residual.",
    ]
    (OUT / "all_methods_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
