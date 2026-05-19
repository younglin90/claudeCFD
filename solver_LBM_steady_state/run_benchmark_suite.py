"""SCMK-LBM Phase-4 benchmark suite : multiple cases vs baseline LBM.

Cases :
    1. Lid-driven cavity Re=100   (moving top wall, 3 stationary walls)
    2. Lid-driven cavity Re=400
    3. Couette flow              (periodic-x, moving top, no-slip bottom)
    4. Re-summary of Kolmogorov + Channel (Phase-1, Phase-4)
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_core import LBMCavity
from lbm_couette import CouetteCase
from lbm_periodic import build_spectral_schur
from solver_scmk import solve_scmk, solve_baseline_periodic


def run_case(case, label, tol=1e-7, max_baseline_steps=200000, max_outer=200,
             krylov_max=15, kinetic_substeps=20, line_search_max=6, out_dir="results_suite"):
    print(f"\n========== {label} ==========")
    print(f"  N={case.N}  omega={case.omega:.4f}")
    if hasattr(case, "Re"):
        print(f"  Re={case.Re:.2f}")

    # baseline
    print("  --- baseline LBM ---")
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline_periodic(case, max_steps=max_baseline_steps, tol=tol,
                                           check_every=500, verbose=False)
    wall_b = time.perf_counter() - t0
    print(f"  baseline : {hist_b[-1][0]} step, {hist_b[-1][2]} LBE, {wall_b:.2f}s, res {hist_b[-1][1]:.3e}")

    # rebuild case for SCMK (clean state)
    case2 = type(case)(*case_args(case))

    print("  --- SCMK Phase-4 ---")
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    t0 = time.perf_counter()
    f_s, hist_s = solve_scmk(case2, S_inv, max_outer=max_outer, tol=tol,
                              krylov_max=krylov_max, krylov_tol=1e-3,
                              line_search_max=line_search_max,
                              kinetic_substeps=kinetic_substeps, verbose=False)
    wall_s = time.perf_counter() - t0
    print(f"  SCMK     : {hist_s[-1][0]} outer, {hist_s[-1][2]} LBE, {wall_s:.2f}s, res {hist_s[-1][1]:.3e}")

    speedup_lbe = hist_b[-1][2] / max(hist_s[-1][2], 1)
    speedup_wall = wall_b / max(wall_s, 1e-12)
    print(f"  ** speedup : {speedup_lbe:.1f}x LBE, {speedup_wall:.1f}x wall **")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bx = np.array([h[2] for h in hist_b]); by = np.array([h[1] for h in hist_b])
    sx = np.array([h[2] for h in hist_s]); sy = np.array([h[1] for h in hist_s])
    ax.semilogy(bx, by, "b-", lw=1.5, label=f"Baseline ({hist_b[-1][2]} LBE, {wall_b:.1f}s)")
    ax.semilogy(sx, sy, "ro-", ms=4, label=f"SCMK Phase-4 ({hist_s[-1][2]} LBE, {wall_s:.1f}s)")
    ax.axhline(tol, color="gray", ls="--", lw=0.8, label=f"tol={tol:.0e}")
    ax.set_xlabel("LBE residual evaluations")
    ax.set_ylabel(r"$\|R_f\|_{RMS}$")
    ax.set_title(f"{label}  |  {speedup_lbe:.1f}x LBE  /  {speedup_wall:.1f}x wall")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_png = f"{out_dir}/{label.lower().replace(' ', '_').replace('=','').replace(',','')}_conv.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Plot : {out_png}")

    # Field agreement
    _, ux_b, uy_b = macro_of(case, f_b)
    _, ux_s, uy_s = macro_of(case2, f_s)
    agree = np.linalg.norm(ux_b - ux_s) / max(np.linalg.norm(ux_b), 1e-30)
    print(f"  field agreement (rel L2 err_ux) : {agree:.3e}")

    return {
        "label": label,
        "N": case.N,
        "Re": getattr(case, "Re", None),
        "omega": case.omega,
        "tol": tol,
        "baseline": {"step": int(hist_b[-1][0]), "lbe": int(hist_b[-1][2]),
                      "wall_s": float(wall_b), "res": float(hist_b[-1][1])},
        "scmk": {"outer": int(hist_s[-1][0]), "lbe": int(hist_s[-1][2]),
                  "wall_s": float(wall_s), "res": float(hist_s[-1][1])},
        "speedup_lbe": float(speedup_lbe),
        "speedup_wall": float(speedup_wall),
        "field_agreement": float(agree),
    }


def case_args(case):
    """Reconstruct constructor args (for clean restart)."""
    if isinstance(case, LBMCavity):
        return (case.N, case.Re, case.U_wall)
    if isinstance(case, CouetteCase):
        return (case.N, case.nu, case.U_wall)
    raise NotImplementedError


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    from lbm_core import moments
    return moments(f)


def main():
    out_dir = "results_suite"
    os.makedirs(out_dir, exist_ok=True)
    summary = []

    # Case 1 : Lid-driven cavity Re=100
    cavity_100 = LBMCavity(N=33, Re=100, U_wall=0.1)
    summary.append(run_case(cavity_100, "Cavity Re=100 N=33",
                             tol=5e-7, max_baseline_steps=200000, out_dir=out_dir,
                             kinetic_substeps=25, max_outer=200))

    # Case 2 : Lid-driven cavity Re=400
    cavity_400 = LBMCavity(N=49, Re=400, U_wall=0.1)
    summary.append(run_case(cavity_400, "Cavity Re=400 N=49",
                             tol=5e-7, max_baseline_steps=200000, out_dir=out_dir,
                             kinetic_substeps=25, max_outer=200))

    # Case 3 : Couette flow
    couette = CouetteCase(N=64, nu=0.05, U_wall=0.05)
    summary.append(run_case(couette, "Couette N=64",
                             tol=1e-9, max_baseline_steps=200000, out_dir=out_dir,
                             kinetic_substeps=20, max_outer=200))

    # Write summary
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n\n========== BENCHMARK SUITE SUMMARY ==========")
    print(f"{'Case':<28} {'LBE speedup':>14} {'Wall speedup':>14} {'Field agree':>14}")
    print("-" * 72)
    for s in summary:
        print(f"{s['label']:<28} {s['speedup_lbe']:>13.1f}x {s['speedup_wall']:>13.1f}x {s['field_agreement']:>14.2e}")
    print()


if __name__ == "__main__":
    main()
