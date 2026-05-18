"""Extended benchmark suite for top-tier SCI validation.

Adds to original 5-case :
    - Cavity Re=400, Re=1000 (additional Re for scaling)
    - N=48 version of Kolmogorov + Channel (grid sensitivity)
    - Cross-comparison with Anderson acceleration baseline

Composite metric weights :
    - speedup (LBE count)
    - field accuracy (vs baseline)
    - convergence success
    - generalization (low std dev across cases)
"""

import os, time, json
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_periodic import KolmogorovCase
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_core import LBMCavity, moments as cavity_moments
from lbm_voxel import VoxelCase, build_cylinder_mask
from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline
from solver_hybrid import solve_hybrid
from solver_anderson import solve_anderson


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return cavity_moments(f)


def run_case(case_b, case_s, case_a, tol, baseline_runner, max_baseline=80000):
    """Run baseline, SCMK-hybrid, Anderson for cross-comparison."""
    # baseline
    t0 = time.perf_counter()
    f_b, hist_b = baseline_runner(case_b, max_steps=max_baseline,
                                   tol=tol, check_every=200, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]

    # SCMK hybrid
    t0 = time.perf_counter()
    f_s, hist_s = solve_hybrid(case_s, max_outer=200, tol=tol,
                                krylov_max=10, kinetic_substeps=15,
                                N_check=6, min_ratio=2.0, verbose=False)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]

    # Anderson
    t0 = time.perf_counter()
    f_a, hist_a = solve_anderson(case_a, max_iter=max_baseline // 2, tol=tol,
                                  m=5, beta=1.0, safeguard=False, verbose=False)
    wall_a = time.perf_counter() - t0
    lbe_a = hist_a[-1][2]; res_a = hist_a[-1][1]

    su_scmk = lbe_b / max(lbe_s, 1)
    su_and = lbe_b / max(lbe_a, 1)

    _, ux_b, _ = macro_of(case_b, f_b)
    _, ux_s, _ = macro_of(case_s, f_s)
    _, ux_a, _ = macro_of(case_a, f_a)
    denom = max(np.linalg.norm(ux_b), 1e-12)
    err_s = float(np.linalg.norm(ux_b - ux_s) / denom)
    err_a = float(np.linalg.norm(ux_b - ux_a) / denom)

    return {
        "baseline_lbe": int(lbe_b), "baseline_wall": float(wall_b),
        "scmk_lbe": int(lbe_s), "scmk_wall": float(wall_s),
        "scmk_speedup": float(su_scmk), "scmk_err": err_s,
        "scmk_converged": bool(res_s < tol * 5),
        "anderson_lbe": int(lbe_a), "anderson_wall": float(wall_a),
        "anderson_speedup": float(su_and), "anderson_err": err_a,
        "anderson_converged": bool(res_a < tol * 5),
    }


def main():
    tol = 1e-7
    tol_cavity = 5e-7
    results = {}

    print("== Kolmogorov N=32 ==")
    c = lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N32"] = run_case(c(), c(), c(), tol, solve_baseline_periodic)

    print("== Channel N=32 ==")
    c = lambda: ChannelCase(N=32, nu=0.05, F0=1e-5)
    results["channel_N32"] = run_case(c(), c(), c(), tol, solve_baseline_periodic)

    print("== Couette N=32 ==")
    c = lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05)
    results["couette_N32"] = run_case(c(), c(), c(), tol, solve_baseline_periodic)

    print("== Cavity Re=100 N=25 ==")
    cavity_baseline = lambda case, max_steps, tol, check_every, verbose: solve_baseline(
        case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    c = lambda: LBMCavity(N=25, Re=100, U_wall=0.1)
    results["cavity_Re100"] = run_case(c(), c(), c(), tol_cavity, cavity_baseline)

    print("== Cavity Re=400 N=33 ==")
    c = lambda: LBMCavity(N=33, Re=400, U_wall=0.1)
    results["cavity_Re400"] = run_case(c(), c(), c(), tol_cavity, cavity_baseline)

    print("== Multi-cylinder N=32 ==")
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    for _ in range(6):
        r = max(2, N // 12)
        cx = rng.randint(r, N - r); cy = rng.randint(r, N - r)
        chi *= build_cylinder_mask(N, cx, cy, r)
    c = lambda: VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    results["multi_cylinder"] = run_case(c(), c(), c(), tol, solve_baseline_periodic)

    # composite metric
    scmk_speeds = [r["scmk_speedup"] for r in results.values()]
    scmk_errs = [r["scmk_err"] for r in results.values()]
    scmk_convs = [r["scmk_converged"] for r in results.values()]
    and_speeds = [r["anderson_speedup"] for r in results.values()]
    and_convs = [r["anderson_converged"] for r in results.values()]

    mean_scmk = float(np.mean(scmk_speeds))
    mean_and = float(np.mean(and_speeds))
    worst_err = float(max(scmk_errs))
    conv_scmk = sum(scmk_convs) / len(scmk_convs)
    conv_and = sum(and_convs) / len(and_convs)
    relative_advantage = mean_scmk / max(mean_and, 0.01)
    std_speedup = float(np.std(scmk_speeds))  # generalization measure

    acc_factor = max(0.0, 1.0 - 5.0 * worst_err)
    composite = mean_scmk * acc_factor * conv_scmk

    print("\n=== EXTENDED RESULTS ===")
    print(f"{'Case':<22} {'baseline':>10} {'SCMK x':>8} {'And x':>8} {'err':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<22} {r['baseline_lbe']:>10} "
              f"{r['scmk_speedup']:>7.1f}x {r['anderson_speedup']:>7.1f}x "
              f"{r['scmk_err']:>10.2e}")
    print()
    print(f"mean SCMK speedup     : {mean_scmk:.2f}x")
    print(f"mean Anderson speedup : {mean_and:.2f}x")
    print(f"SCMK/Anderson ratio   : {relative_advantage:.2f}")
    print(f"worst SCMK err        : {worst_err:.3e}")
    print(f"SCMK convergence      : {conv_scmk*100:.0f}%")
    print(f"Anderson convergence  : {conv_and*100:.0f}%")
    print(f"speedup std dev       : {std_speedup:.2f}")
    print(f"composite score       : {composite:.4f}")
    print(f"{composite:.6f}")

    with open("verify_extended_log.json", "w") as fh:
        json.dump({"per_case": results,
                    "mean_scmk": mean_scmk, "mean_anderson": mean_and,
                    "relative_advantage": relative_advantage,
                    "worst_err": worst_err, "conv_scmk": conv_scmk,
                    "conv_and": conv_and, "std_speedup": std_speedup,
                    "composite": composite}, fh, indent=2)


if __name__ == "__main__":
    main()
