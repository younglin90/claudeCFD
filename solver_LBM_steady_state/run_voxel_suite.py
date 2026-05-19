"""Phase-5 : vessel-like voxel mask benchmarks.

Cases :
    1. Clean periodic              (fluid_fraction = 1.00)  -- Phase-1 baseline
    2. Random voxel obstacles 5%
    3. Random voxel obstacles 10%
    4. Random voxel obstacles 20%
    5. Single cylindrical obstacle
    6. Multiple cylinders (porous-like)

Measure how SCMK Phase-4 speedup degrades with geometry complexity.
"""

import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_voxel import VoxelCase, build_random_obstacle_mask, build_cylinder_mask
from lbm_periodic import build_spectral_schur
from solver_scmk import solve_scmk, solve_baseline_periodic


def run_one(case, label, tol=1e-7, out_dir="results_voxel", **scmk_kw):
    print(f"\n========== {label}  (fluid_frac={case.fluid_fraction:.3f}) ==========")

    print("  baseline ...")
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline_periodic(case, max_steps=80000, tol=tol, check_every=500, verbose=False)
    wall_b = time.perf_counter() - t0
    lbe_b = hist_b[-1][2]; res_b = hist_b[-1][1]
    print(f"    {lbe_b} LBE, {wall_b:.2f}s, res {res_b:.3e}")

    # rebuild case for clean SCMK start
    case2 = VoxelCase(case.chi.copy(), nu=case.nu, F0=case.F0, kf=case.kf)

    print("  SCMK Phase-4 ...")
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    t0 = time.perf_counter()
    f_s, hist_s = solve_scmk(case2, S_inv, tol=tol, verbose=False, **scmk_kw)
    wall_s = time.perf_counter() - t0
    lbe_s = hist_s[-1][2]; res_s = hist_s[-1][1]
    print(f"    {hist_s[-1][0]} outer, {lbe_s} LBE, {wall_s:.2f}s, res {res_s:.3e}")

    su_lbe = lbe_b / max(lbe_s, 1)
    su_w = wall_b / max(wall_s, 1e-12)
    print(f"    ** speedup {su_lbe:.1f}x LBE / {su_w:.1f}x wall **")

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bx = [h[2] for h in hist_b]; by = [h[1] for h in hist_b]
    sx = [h[2] for h in hist_s]; sy = [h[1] for h in hist_s]
    ax.semilogy(bx, by, "b-", label=f"Baseline ({lbe_b} LBE, {wall_b:.1f}s)")
    ax.semilogy(sx, sy, "ro-", ms=3, label=f"SCMK ({lbe_s} LBE, {wall_s:.1f}s)")
    ax.axhline(tol, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("LBE calls"); ax.set_ylabel("res RMS")
    ax.set_title(f"{label}  |  fluid={case.fluid_fraction:.2f}  |  {su_lbe:.1f}x LBE")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fn = f"{out_dir}/{label.lower().replace(' ','_').replace('=','').replace('%','pct')}.png"
    plt.savefig(fn, dpi=100); plt.close(fig)

    # mask viz
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    axs[0].imshow(case.chi, cmap="gray", origin="lower")
    axs[0].set_title(f"mask (fluid_frac={case.fluid_fraction:.3f})")
    _, ux_b, _ = case.macro(f_b)
    axs[1].imshow(ux_b, cmap="RdBu_r", origin="lower")
    axs[1].set_title("baseline u_x")
    plt.tight_layout()
    fn_mask = fn.replace(".png", "_mask.png")
    plt.savefig(fn_mask, dpi=100); plt.close(fig)

    return {
        "label": label, "fluid_frac": case.fluid_fraction,
        "baseline_lbe": int(lbe_b), "baseline_wall": float(wall_b), "baseline_res": float(res_b),
        "scmk_outer": int(hist_s[-1][0]), "scmk_lbe": int(lbe_s),
        "scmk_wall": float(wall_s), "scmk_res": float(res_s),
        "speedup_lbe": float(su_lbe), "speedup_wall": float(su_w),
    }


def main():
    out = "results_voxel"; os.makedirs(out, exist_ok=True)
    N = 48; nu = 0.05; F0 = 2e-4; kf = 1
    tol = 1e-7

    base_scmk = dict(max_outer=80, krylov_max=10, krylov_tol=1e-3,
                     line_search_max=5, kinetic_substeps=20)

    summary = []

    # 1. Clean periodic
    chi = np.ones((N, N))
    summary.append(run_one(VoxelCase(chi, nu, F0, kf),
                            "clean periodic", tol, out, **base_scmk))

    # 2-4. Random scatter
    for d in (0.05, 0.10, 0.20):
        chi = build_random_obstacle_mask(N, density=d, seed=42)
        summary.append(run_one(VoxelCase(chi, nu, F0, kf),
                                f"random {int(d*100)}pct", tol, out, **base_scmk))

    # 5. Single cylinder
    chi = build_cylinder_mask(N, cx=N//2, cy=N//2, radius=N//6)
    summary.append(run_one(VoxelCase(chi, nu, F0, kf),
                            "cylinder", tol, out, **base_scmk))

    # 6. Multiple cylinders (porous-like)
    chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    n_obs = 8
    radius = N // 12
    for _ in range(n_obs):
        cx = rng.randint(radius, N - radius)
        cy = rng.randint(radius, N - radius)
        chi *= build_cylinder_mask(N, cx, cy, radius)
    summary.append(run_one(VoxelCase(chi, nu, F0, kf),
                            "multi-cylinder", tol, out, **base_scmk))

    # Write summary
    with open(f"{out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n========== PHASE-5 VOXEL SUITE SUMMARY ==========")
    print(f"{'Case':<22}{'fluid':>8}{'LBE speedup':>14}{'Wall speedup':>14}")
    print("-" * 60)
    for s in summary:
        print(f"{s['label']:<22}{s['fluid_frac']:>8.3f}{s['speedup_lbe']:>13.1f}x{s['speedup_wall']:>13.1f}x")


if __name__ == "__main__":
    main()
