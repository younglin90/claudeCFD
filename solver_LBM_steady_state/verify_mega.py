"""Mega benchmark suite — 11 cases, 3 methods (baseline / SCMK-hybrid / Anderson).

Covers :
    2D periodic    : Kolmogorov N=32, 48
    2D wall        : Channel N=32, Couette N=32
    2D cavity      : Re=100, 400 (N=25, 33)
    2D voxel       : multi-cylinder, single cylinder Re=20, Re=40
    3D periodic    : Kolmogorov 3D N=16, 24

Cross-comparison vs Anderson acceleration (classical method).
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
from lbm_cylinder import make_cylinder_case, compute_drag
from lbm_3d import Kolmogorov3DCase

from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline
from solver_hybrid import solve_hybrid
from solver_scmk_3d import solve_baseline_3d, solve_scmk_3d
from solver_anderson import solve_anderson


def macro_of(case, f):
    if hasattr(case, "macro"):
        m = case.macro(f)
        if len(m) == 3:  # 2D
            return m
        # 3D : (rho, ux, uy, uz)
        return m[0], m[1], m[2]
    return cavity_moments(f)


def run_one(case_b, case_s, case_a, tol, baseline_runner,
             scmk_runner, max_baseline, is_3d=False):
    t0 = time.perf_counter()
    f_b, hist_b = baseline_runner(case_b, max_steps=max_baseline,
                                   tol=tol, check_every=200, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]

    t0 = time.perf_counter()
    if is_3d:
        f_s, hist_s = scmk_runner(case_s, max_outer=120, tol=tol,
                                    krylov_max=10, kinetic_substeps=15,
                                    N_check=6, min_ratio=2.0, verbose=False)
    else:
        f_s, hist_s = scmk_runner(case_s, max_outer=200, tol=tol,
                                    krylov_max=10, kinetic_substeps=15,
                                    N_check=6, min_ratio=2.0, verbose=False)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]

    if is_3d:
        f_a, hist_a = f_s, hist_s  # skip Anderson for 3D (not implemented)
        wall_a = wall_s; lbe_a = lbe_s; res_a = res_s
    else:
        t0 = time.perf_counter()
        f_a, hist_a = solve_anderson(case_a, max_iter=max_baseline // 2, tol=tol,
                                      m=5, beta=1.0, safeguard=False, verbose=False)
        wall_a = time.perf_counter() - t0
        lbe_a = hist_a[-1][2]; res_a = hist_a[-1][1]

    su_s = lbe_b / max(lbe_s, 1)
    su_a = lbe_b / max(lbe_a, 1)
    su_s_wall = wall_b / max(wall_s, 1e-12)

    macro_b = macro_of(case_b, f_b)
    macro_s = macro_of(case_s, f_s)
    ux_b, ux_s = macro_b[1], macro_s[1]
    denom = max(np.linalg.norm(ux_b), 1e-12)
    err_s = float(np.linalg.norm(ux_b - ux_s) / denom)

    return {
        "baseline_lbe": int(lbe_b), "baseline_wall": float(wall_b),
        "scmk_lbe": int(lbe_s), "scmk_wall": float(wall_s),
        "scmk_speedup_lbe": float(su_s), "scmk_speedup_wall": float(su_s_wall),
        "scmk_err": err_s, "scmk_converged": bool(res_s < tol * 5),
        "anderson_lbe": int(lbe_a),
        "anderson_speedup": float(su_a),
        "anderson_converged": bool(res_a < tol * 5),
    }


def main():
    tol = 1e-7
    tol_cav = 5e-7
    results = {}

    print("\n[1/11] Kolmogorov N=32 (2D periodic)")
    c = lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N32"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 50000)

    print("[2/11] Kolmogorov N=48 (2D periodic, larger)")
    c = lambda: KolmogorovCase(N=48, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N48"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 80000)

    print("[3/11] Channel N=32 (2D 2-wall)")
    c = lambda: ChannelCase(N=32, nu=0.05, F0=1e-5)
    results["channel_N32"] = run_one(c(), c(), c(), tol,
                                       solve_baseline_periodic, solve_hybrid, 50000)

    print("[4/11] Couette N=32 (2D wall+lid)")
    c = lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05)
    results["couette_N32"] = run_one(c(), c(), c(), tol,
                                       solve_baseline_periodic, solve_hybrid, 50000)

    print("[5/11] Cavity Re=100 N=25 (2D 4-wall)")
    cavity_runner = lambda case, max_steps, tol, check_every, verbose: solve_baseline(
        case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    c = lambda: LBMCavity(N=25, Re=100, U_wall=0.1)
    results["cavity_Re100"] = run_one(c(), c(), c(), tol_cav,
                                        cavity_runner, solve_hybrid, 80000)

    print("[6/11] Cavity Re=400 N=33 (2D 4-wall stiffer)")
    c = lambda: LBMCavity(N=33, Re=400, U_wall=0.1)
    results["cavity_Re400"] = run_one(c(), c(), c(), tol_cav,
                                        cavity_runner, solve_hybrid, 80000)

    print("[7/11] Multi-cylinder N=32 (2D 6 obstacles)")
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    for _ in range(6):
        r = max(2, N // 12)
        cx = rng.randint(r, N - r); cy = rng.randint(r, N - r)
        chi *= build_cylinder_mask(N, cx, cy, r)
    c = lambda: VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    results["multi_cylinder"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 50000)

    print("[8/11] Cylinder Re=20 N=64 (single obstacle)")
    c = lambda: make_cylinder_case(N=64, D=12, Re=20, U_target=0.05)
    results["cylinder_Re20"] = run_one(c(), c(), c(), tol,
                                         solve_baseline_periodic, solve_hybrid, 30000)

    print("[9/11] Cylinder Re=40 N=64 (single obstacle, higher Re)")
    c = lambda: make_cylinder_case(N=64, D=12, Re=40, U_target=0.05)
    results["cylinder_Re40"] = run_one(c(), c(), c(), tol,
                                         solve_baseline_periodic, solve_hybrid, 30000)

    print("[10/11] 3D Kolmogorov N=16 (small 3D)")
    c = lambda: Kolmogorov3DCase(N=16, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_3D_N16"] = run_one(c(), c(), c(), tol,
                                            solve_baseline_3d, solve_scmk_3d, 30000, is_3d=True)

    print("[11/11] 3D Kolmogorov N=24 (medium 3D)")
    c = lambda: Kolmogorov3DCase(N=24, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_3D_N24"] = run_one(c(), c(), c(), tol,
                                            solve_baseline_3d, solve_scmk_3d, 30000, is_3d=True)

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Case':<25} {'base LBE':>10} {'SCMK x':>9} {'And x':>9} {'wall x':>9} {'err':>10} {'conv':>6}")
    print("-" * 90)
    for name, r in results.items():
        conv = "✓" if r["scmk_converged"] else "✗"
        print(f"{name:<25} {r['baseline_lbe']:>10} {r['scmk_speedup_lbe']:>8.1f}x "
              f"{r['anderson_speedup']:>8.1f}x {r['scmk_speedup_wall']:>8.1f}x "
              f"{r['scmk_err']:>10.2e} {conv:>6}")

    speedups = [r["scmk_speedup_lbe"] for r in results.values()]
    errs = [r["scmk_err"] for r in results.values()]
    convs = [r["scmk_converged"] for r in results.values()]
    and_speeds = [r["anderson_speedup"] for r in results.values()]

    mean_su = float(np.mean(speedups))
    median_su = float(np.median(speedups))
    min_su = float(np.min(speedups))
    max_su = float(np.max(speedups))
    worst_err = float(max(errs))
    conv_frac = sum(convs) / len(convs)
    mean_and = float(np.mean(and_speeds))
    advantage = mean_su / max(mean_and, 0.01)

    print()
    print(f"Mean SCMK speedup    : {mean_su:.2f}x")
    print(f"Median SCMK speedup  : {median_su:.2f}x")
    print(f"Min SCMK speedup     : {min_su:.2f}x")
    print(f"Max SCMK speedup     : {max_su:.2f}x")
    print(f"Mean Anderson speedup: {mean_and:.2f}x  (SCMK / Anderson = {advantage:.2f})")
    print(f"Worst field err      : {worst_err:.3e}")
    print(f"Convergence rate     : {conv_frac*100:.0f}%")

    acc_factor = max(0.0, 1.0 - 5.0 * worst_err)
    composite = mean_su * acc_factor * conv_frac
    print(f"Composite score      : {composite:.4f}")
    print(f"{composite:.6f}")

    with open("verify_mega_log.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
