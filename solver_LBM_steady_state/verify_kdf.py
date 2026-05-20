"""Autoresearch verify command : run 5 benchmarks, print composite metric.

Composite = mean(LBE_speedup_5cases) * accuracy_factor

  accuracy_factor = min(1.0, 1 - 5*max(field_err_5cases))
                  (penalize cases where SCMK field deviates >20% from baseline)

Higher is better. Target output : single float on last stdout line.

Cases (small N for speed) :
    1. Kolmogorov N=32 periodic
    2. Channel N=32 walls
    3. Couette N=32 lid
    4. Cavity Re=100 N=25 four-walls
    5. Multi-cylinder N=32 voxel

Each case : baseline + SCMK Phase-4 (current). Tol = 1e-7.
"""

import os
import sys
import time
import json
import numpy as np

# silence matplotlib if imported anywhere
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_periodic import KolmogorovCase, build_spectral_schur
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_core import LBMCavity
from lbm_voxel import VoxelCase, build_cylinder_mask
from solver_kdf import solve_kdf
from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    from lbm_core import moments
    return moments(f)


def run_case(case_b, case_s, tol, baseline_runner, scmk_kw,
             max_baseline_steps=50000):
    """Run baseline + SCMK, return (lbe_speedup, field_err_vs_baseline, scmk_converged)."""
    t0 = time.perf_counter()
    f_b, hist_b = baseline_runner(case_b, max_steps=max_baseline_steps,
                                   tol=tol, check_every=200, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]

    
    t0 = time.perf_counter()
    f_s, hist_s = solve_kdf(case_s, tol=tol, verbose=False, **scmk_kw)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]

    speedup_lbe = lbe_b / max(lbe_s, 1)
    converged_s = res_s < tol * 5  # within factor 5 of tolerance

    _, ux_b, _ = macro_of(case_b, f_b)
    _, ux_s, _ = macro_of(case_s, f_s)
    denom = max(np.linalg.norm(ux_b), 1e-12)
    err = float(np.linalg.norm(ux_b - ux_s) / denom)

    return {"lbe_speedup": float(speedup_lbe), "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
            "field_err": err, "scmk_converged": bool(converged_s),
            "baseline_lbe": int(lbe_b), "scmk_lbe": int(lbe_s),
            "baseline_wall": float(wall_b), "scmk_wall": float(wall_s)}


scmk_kw = dict(max_outer=400, lbm_per_cycle=12, dmd_rank=6, kinetic_substeps=15)


def main():
    tol = 1e-7
    results = {}

    # Case 1 : Kolmogorov
    c1b = KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    c1s = KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov"] = run_case(c1b, c1s, tol, solve_baseline_periodic, scmk_kw)

    # Case 2 : Channel
    c2b = ChannelCase(N=32, nu=0.05, F0=1e-5)
    c2s = ChannelCase(N=32, nu=0.05, F0=1e-5)
    results["channel"] = run_case(c2b, c2s, tol, solve_baseline_periodic, scmk_kw)

    # Case 3 : Couette
    c3b = CouetteCase(N=32, nu=0.05, U_wall=0.05)
    c3s = CouetteCase(N=32, nu=0.05, U_wall=0.05)
    results["couette"] = run_case(c3b, c3s, tol, solve_baseline_periodic, scmk_kw)

    # Case 4 : Cavity Re=100 (uses solve_baseline cavity-specific)
    c4b = LBMCavity(N=25, Re=100, U_wall=0.1)
    c4s = LBMCavity(N=25, Re=100, U_wall=0.1)
    tol4 = 5e-7
    cavity_baseline = lambda case, max_steps, tol, check_every, verbose: solve_baseline(
        case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    results["cavity_re100"] = run_case(c4b, c4s, tol4, cavity_baseline, scmk_kw,
                                       max_baseline_steps=80000)

    # Case 5 : Multi-cylinder
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    n_obs = 6; radius = max(2, N // 12)
    for _ in range(n_obs):
        cx = rng.randint(radius, N - radius); cy = rng.randint(radius, N - radius)
        chi *= build_cylinder_mask(N, cx, cy, radius)
    c5b = VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    c5s = VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    results["multi_cylinder"] = run_case(c5b, c5s, tol, solve_baseline_periodic, scmk_kw)

    # Composite metric : mean speedup * accuracy penalty
    speedups = [r["lbe_speedup"] for r in results.values()]
    errs = [r["field_err"] for r in results.values()]
    convs = [r["scmk_converged"] for r in results.values()]
    mean_speedup = float(np.mean(speedups))
    worst_err = float(max(errs))
    conv_frac = sum(convs) / len(convs)
    # penalty : if worst_err > 0.2, accuracy_factor drops linearly to 0 at 1.0
    accuracy_factor = max(0.0, 1.0 - 5.0 * worst_err)
    composite = mean_speedup * accuracy_factor * (0.5 + 0.5 * conv_frac)

    print("\n--- per-case ---")
    for name, r in results.items():
        print(f"  {name:18s} lbe_speedup={r['lbe_speedup']:6.2f}x  "
              f"field_err={r['field_err']:.2e}  converged={r['scmk_converged']}")

    print(f"\nmean_speedup     = {mean_speedup:.3f}")
    print(f"worst_field_err  = {worst_err:.3e}")
    print(f"converged_frac   = {conv_frac:.2f}")
    print(f"accuracy_factor  = {accuracy_factor:.3f}")
    print(f"composite_score  = {composite:.4f}")

    # autoresearch parses last line
    print(f"{composite:.6f}")
    with open("verify_log.json", "w") as fh:
        json.dump({"per_case": results,
                    "mean_speedup": mean_speedup, "worst_err": worst_err,
                    "conv_frac": conv_frac, "composite": composite}, fh, indent=2)


if __name__ == "__main__":
    main()
