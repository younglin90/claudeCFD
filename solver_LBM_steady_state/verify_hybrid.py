"""Verify hybrid SCMK+baseline."""

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


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return cavity_moments(f)


def run_case(case_b, case_s, tol, baseline_runner, max_baseline=50000):
    t0 = time.perf_counter()
    f_b, hist_b = baseline_runner(case_b, max_steps=max_baseline,
                                   tol=tol, check_every=200, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]

    t0 = time.perf_counter()
    f_s, hist_s = solve_hybrid(case_s, max_outer=200, tol=tol,
                                krylov_max=10, kinetic_substeps=15,
                                N_check=6, min_ratio=2.0,
                                verbose=False)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]

    speedup = lbe_b / max(lbe_s, 1)
    _, ux_b, _ = macro_of(case_b, f_b)
    _, ux_s, _ = macro_of(case_s, f_s)
    err = float(np.linalg.norm(ux_b - ux_s) / max(np.linalg.norm(ux_b), 1e-12))
    return {"lbe_speedup": float(speedup),
            "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
            "field_err": err, "scmk_converged": bool(res_s < tol * 5),
            "baseline_lbe": int(lbe_b), "scmk_lbe": int(lbe_s)}


def main():
    tol = 1e-7
    results = {}
    c1 = lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov"] = run_case(c1(), c1(), tol, solve_baseline_periodic)
    c2 = lambda: ChannelCase(N=32, nu=0.05, F0=1e-5)
    results["channel"] = run_case(c2(), c2(), tol, solve_baseline_periodic)
    c3 = lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05)
    results["couette"] = run_case(c3(), c3(), tol, solve_baseline_periodic)
    c4 = lambda: LBMCavity(N=25, Re=100, U_wall=0.1)
    cavity_baseline = lambda case, max_steps, tol, check_every, verbose: solve_baseline(
        case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    results["cavity_re100"] = run_case(c4(), c4(), 5e-7, cavity_baseline, max_baseline=80000)
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    n_obs = 6; radius = max(2, N // 12)
    for _ in range(n_obs):
        cx = rng.randint(radius, N - radius); cy = rng.randint(radius, N - radius)
        chi *= build_cylinder_mask(N, cx, cy, radius)
    c5 = lambda: VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    results["multi_cylinder"] = run_case(c5(), c5(), tol, solve_baseline_periodic)

    speedups = [r["lbe_speedup"] for r in results.values()]
    errs = [r["field_err"] for r in results.values()]
    convs = [r["scmk_converged"] for r in results.values()]
    mean_speedup = float(np.mean(speedups))
    worst_err = float(max(errs))
    conv_frac = sum(convs) / len(convs)
    accuracy_factor = max(0.0, 1.0 - 5.0 * worst_err)
    composite = mean_speedup * accuracy_factor * (0.5 + 0.5 * conv_frac)

    print("\n--- per-case (HYBRID) ---")
    for name, r in results.items():
        print(f"  {name:18s} lbe_speedup={r['lbe_speedup']:6.2f}x  "
              f"field_err={r['field_err']:.2e}  converged={r['scmk_converged']}")
    print(f"\nmean_speedup     = {mean_speedup:.3f}")
    print(f"worst_field_err  = {worst_err:.3e}")
    print(f"converged_frac   = {conv_frac:.2f}")
    print(f"composite_score  = {composite:.4f}")
    print(f"{composite:.6f}")


if __name__ == "__main__":
    main()
