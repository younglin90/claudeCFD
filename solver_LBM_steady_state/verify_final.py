"""Final paper-ready benchmark : 12 cases.

2D periodic    : Kolmogorov N=32, N=48, N=64
2D wall        : Channel N=32, Couette N=32
2D cavity      : Re=100 N=25, Re=400 N=33, Re=1000 N=49
2D voxel       : multi-cylinder N=32
3D periodic    : Kolmogorov 3D N=16, N=24
3D wall        : Channel 3D N=24

Compares :
    Baseline LBM Picard
    Anderson acceleration (classical, m=5)
    SCMK Phase-4 hybrid (this paper)
"""

import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_periodic import KolmogorovCase
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_core import LBMCavity, moments as cavity_moments
from lbm_voxel import VoxelCase, build_cylinder_mask
from lbm_3d import Kolmogorov3DCase
from lbm_channel_3d import Channel3DCase

from solver_scmk import solve_baseline_periodic
from solver_baseline import solve_baseline
from solver_hybrid import solve_hybrid
from solver_scmk_3d import solve_baseline_3d, solve_scmk_3d
from solver_anderson import solve_anderson


def macro_of(case, f):
    if hasattr(case, "macro"):
        m = case.macro(f)
        if len(m) == 3:
            return m
        return m[0], m[1], m[2]
    return cavity_moments(f)


def run_one(case_b, case_s, case_a, tol, baseline_runner,
             scmk_runner, max_baseline, is_3d=False, do_anderson=True):
    try:
        t0 = time.perf_counter()
        f_b, hist_b = baseline_runner(case_b, max_steps=max_baseline,
                                       tol=tol, check_every=200, verbose=False)
        wall_b = time.perf_counter() - t0
        lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]
    except Exception as e:
        print(f"  baseline failed : {e}")
        wall_b = 0.0; lbe_b = 0; res_b = float("nan"); f_b = None

    try:
        t0 = time.perf_counter()
        if is_3d:
            f_s, hist_s = scmk_runner(case_s, max_outer=150, tol=tol,
                                        krylov_max=10, kinetic_substeps=15,
                                        N_check=6, min_ratio=2.0, verbose=False)
        else:
            f_s, hist_s = scmk_runner(case_s, max_outer=300, tol=tol,
                                        krylov_max=10, kinetic_substeps=15,
                                        N_check=6, min_ratio=2.0, verbose=False)
        wall_s = time.perf_counter() - t0
        lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]
    except Exception as e:
        print(f"  SCMK failed : {e}")
        wall_s = 0.0; lbe_s = 0; res_s = float("nan"); f_s = None

    if do_anderson and not is_3d:
        try:
            t0 = time.perf_counter()
            f_a, hist_a = solve_anderson(case_a, max_iter=max_baseline // 2, tol=tol,
                                           m=5, beta=1.0, safeguard=False, verbose=False)
            wall_a = time.perf_counter() - t0
            lbe_a = hist_a[-1][2]; res_a = hist_a[-1][1]
            and_su = lbe_b / max(lbe_a, 1)
            and_conv = bool(res_a < tol * 5)
        except Exception as e:
            print(f"  Anderson failed : {e}")
            lbe_a = 0; and_su = float("nan"); and_conv = False
    else:
        lbe_a = lbe_s; and_su = float("nan"); and_conv = False

    su_lbe = lbe_b / max(lbe_s, 1)
    su_wall = wall_b / max(wall_s, 1e-12)

    if f_b is None or f_s is None:
        err_s = float("nan")
    else:
        macro_b = macro_of(case_b, f_b)
        macro_s = macro_of(case_s, f_s)
        ux_b, ux_s = macro_b[1], macro_s[1]
        denom = max(np.linalg.norm(ux_b), 1e-12)
        err_s = float(np.linalg.norm(ux_b - ux_s) / denom)

    return {
        "baseline_lbe": int(lbe_b), "baseline_wall": float(wall_b),
        "baseline_res": float(res_b),
        "scmk_lbe": int(lbe_s), "scmk_wall": float(wall_s),
        "scmk_speedup_lbe": float(su_lbe), "scmk_speedup_wall": float(su_wall),
        "scmk_err": err_s, "scmk_converged": bool(res_s < tol * 5),
        "anderson_lbe": int(lbe_a),
        "anderson_speedup": float(and_su),
        "anderson_converged": and_conv,
    }


def main():
    tol = 1e-7
    tol_cav = 5e-7
    results = {}

    print("[ 1/12] Kolmogorov N=32")
    c = lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N32"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 50000)

    print("[ 2/12] Kolmogorov N=48")
    c = lambda: KolmogorovCase(N=48, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N48"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 80000)

    print("[ 3/12] Kolmogorov N=64")
    c = lambda: KolmogorovCase(N=64, nu=0.05, F0=2e-4, kf=1)
    results["kolmogorov_N64"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 100000)

    print("[ 4/12] Channel N=32")
    c = lambda: ChannelCase(N=32, nu=0.05, F0=1e-5)
    results["channel_N32"] = run_one(c(), c(), c(), tol,
                                       solve_baseline_periodic, solve_hybrid, 50000)

    print("[ 5/12] Couette N=32")
    c = lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05)
    results["couette_N32"] = run_one(c(), c(), c(), tol,
                                       solve_baseline_periodic, solve_hybrid, 50000)

    print("[ 6/12] Cavity Re=100")
    cavity_runner = lambda case, max_steps, tol, check_every, verbose: solve_baseline(
        case, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose)
    c = lambda: LBMCavity(N=25, Re=100, U_wall=0.1)
    results["cavity_Re100"] = run_one(c(), c(), c(), tol_cav,
                                        cavity_runner, solve_hybrid, 80000)

    print("[ 7/12] Cavity Re=400")
    c = lambda: LBMCavity(N=33, Re=400, U_wall=0.1)
    results["cavity_Re400"] = run_one(c(), c(), c(), tol_cav,
                                        cavity_runner, solve_hybrid, 100000)

    print("[ 8/12] Cavity Re=1000 (low U for BGK stability)")
    c = lambda: LBMCavity(N=65, Re=1000, U_wall=0.05)
    results["cavity_Re1000"] = run_one(c(), c(), c(), tol_cav,
                                         cavity_runner, solve_hybrid, 200000)

    print("[ 9/12] Multi-cylinder N=32")
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    for _ in range(6):
        r = max(2, N // 12)
        cx = rng.randint(r, N - r); cy = rng.randint(r, N - r)
        chi *= build_cylinder_mask(N, cx, cy, r)
    c = lambda: VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    results["multi_cylinder"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_periodic, solve_hybrid, 50000)

    print("[10/12] 3D Kolmogorov N=16")
    c = lambda: Kolmogorov3DCase(N=16, nu=0.05, F0=2e-4, kf=1)
    results["3d_kolmogorov_N16"] = run_one(c(), c(), c(), tol,
                                             solve_baseline_3d, solve_scmk_3d, 30000,
                                             is_3d=True, do_anderson=False)

    print("[11/12] 3D Kolmogorov N=24")
    c = lambda: Kolmogorov3DCase(N=24, nu=0.05, F0=2e-4, kf=1)
    results["3d_kolmogorov_N24"] = run_one(c(), c(), c(), tol,
                                             solve_baseline_3d, solve_scmk_3d, 30000,
                                             is_3d=True, do_anderson=False)

    print("[12/12] 3D Channel N=24")
    c = lambda: Channel3DCase(N=24, nu=0.05, F0=1e-4)
    results["3d_channel_N24"] = run_one(c(), c(), c(), tol,
                                          solve_baseline_3d, solve_scmk_3d, 30000,
                                          is_3d=True, do_anderson=False)

    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Case':<22} {'base LBE':>10} {'SCMK x':>9} {'wall x':>9} {'And x':>9} {'err':>10} {'conv':>6}")
    print("-" * 100)
    for name, r in results.items():
        conv = "✓" if r["scmk_converged"] else "✗"
        and_str = f"{r['anderson_speedup']:.1f}x" if not np.isnan(r["anderson_speedup"]) else "  N/A"
        print(f"{name:<22} {r['baseline_lbe']:>10} {r['scmk_speedup_lbe']:>8.1f}x "
              f"{r['scmk_speedup_wall']:>8.1f}x {and_str:>9} "
              f"{r['scmk_err']:>10.2e} {conv:>6}")

    speedups_lbe = [r["scmk_speedup_lbe"] for r in results.values()]
    speedups_wall = [r["scmk_speedup_wall"] for r in results.values()]
    errs = [r["scmk_err"] for r in results.values()]
    convs = [r["scmk_converged"] for r in results.values()]

    print()
    print(f"Arithmetic mean SCMK LBE speedup     : {np.mean(speedups_lbe):.2f}x")
    print(f"Geometric mean SCMK LBE speedup      : {np.exp(np.mean(np.log(speedups_lbe))):.2f}x")
    print(f"Median SCMK LBE speedup              : {np.median(speedups_lbe):.2f}x")
    print(f"Min SCMK LBE speedup                 : {np.min(speedups_lbe):.2f}x")
    print(f"Max SCMK LBE speedup                 : {np.max(speedups_lbe):.2f}x")
    print(f"Wall speedup mean                    : {np.mean(speedups_wall):.2f}x")
    print(f"Worst field err                      : {max(errs):.3e}")
    print(f"Convergence rate                     : {sum(convs)}/{len(convs)}")

    with open("verify_final_log.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(results.keys())
    xs = np.arange(len(names))
    scmk_su = [results[n]["scmk_speedup_lbe"] for n in names]
    and_su = [results[n]["anderson_speedup"] if not np.isnan(results[n]["anderson_speedup"]) else 0
              for n in names]
    width = 0.35
    ax.bar(xs - width/2, scmk_su, width, label="SCMK", color="C0")
    ax.bar(xs + width/2, and_su, width, label="Anderson m=5", color="C1")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LBE-call speedup")
    ax.set_title("SCMK vs Anderson: 12-case benchmark")
    ax.axhline(1.0, color="gray", ls="--", lw=0.7)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("verify_final_speedup.png", dpi=120)
    plt.close(fig)
    print("Plot : verify_final_speedup.png")


if __name__ == "__main__":
    main()
