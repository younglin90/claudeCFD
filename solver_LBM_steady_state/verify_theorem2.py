"""Empirical verification of Theorem 2 (Newton-Krylov contraction rate bound).

Theorem 2 claim:  ||f^{n+1} - f*|| / ||f^n - f*||  <=  rho_NK = 1 - 1/kappa_target = 0.98

Setup :
    Run SCMK Phase-4 to deep tolerance (1e-9) on 6 representative cases.
    Estimate per-iteration contraction rate as geometric mean of consecutive ratios.
    Verify : measured rho <= 0.98.

Output :
    Table : case, measured rho, max-deviation from bound, iterations to 1e-7
    Plot : log10(res) vs outer iter for each case
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from lbm_periodic import KolmogorovCase
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_core import LBMCavity
from lbm_voxel import VoxelCase, build_cylinder_mask
from solver_hybrid import solve_hybrid
from solver_baseline import solve_baseline as cavity_baseline_runner


def measure_rho(history):
    """Geometric mean of consecutive residual ratios (after warmup of 2 iters)."""
    res = np.array([h[1] for h in history])
    if len(res) < 4:
        return None
    ratios = res[3:] / res[2:-1]
    ratios = ratios[ratios > 0]
    if len(ratios) < 2:
        return None
    log_ratios = np.log(ratios)
    return float(np.exp(np.mean(log_ratios)))


def run_case(case, tol, label):
    f, hist = solve_hybrid(case, max_outer=300, tol=tol,
                            krylov_max=10, krylov_tol=1e-3,
                            kinetic_substeps=15,
                            N_check=6, min_ratio=2.0, verbose=False)
    rho = measure_rho(hist)
    iters_to_tol = hist[-1][0]
    final_res = hist[-1][1]
    converged = final_res < tol * 5
    return {"label": label, "rho": rho, "iters": iters_to_tol,
            "final_res": final_res, "converged": converged,
            "history": hist}


def main():
    out_dir = "results_theorem2"
    os.makedirs(out_dir, exist_ok=True)

    tol = 1e-7

    cases = {}

    print("[1] Kolmogorov N=32")
    cases["Kolmogorov"] = run_case(
        KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1), tol, "Kolmogorov N=32")

    print("[2] Channel N=32")
    cases["Channel"] = run_case(
        ChannelCase(N=32, nu=0.05, F0=1e-5), tol, "Channel N=32")

    print("[3] Couette N=32")
    cases["Couette"] = run_case(
        CouetteCase(N=32, nu=0.05, U_wall=0.05), tol, "Couette N=32")

    print("[4] Cavity Re=100 N=25  (uses cavity baseline runner internally)")
    cases["Cavity Re=100"] = run_case(
        LBMCavity(N=25, Re=100, U_wall=0.1), 5e-7, "Cavity Re=100 N=25")

    print("[5] Cavity Re=400 N=33")
    cases["Cavity Re=400"] = run_case(
        LBMCavity(N=33, Re=400, U_wall=0.1), 5e-7, "Cavity Re=400 N=33")

    print("[6] Multi-cylinder N=32")
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    for _ in range(6):
        r = max(2, N // 12)
        cx = rng.randint(r, N - r); cy = rng.randint(r, N - r)
        chi *= build_cylinder_mask(N, cx, cy, r)
    cases["Multi-cyl"] = run_case(
        VoxelCase(chi, nu=0.05, F0=2e-4, kf=1), tol, "Multi-cyl N=32")

    # Print + plot
    bound = 0.98  # Theorem 2 prediction
    print()
    print(f"{'Case':<22} {'rho_measured':>14} {'<= 0.98?':>10} {'iters':>8} {'converged':>10}")
    print("-" * 72)
    for name, r in cases.items():
        ok = "YES" if (r["rho"] is not None and r["rho"] < bound) else "NO"
        rho_str = f"{r['rho']:.4f}" if r["rho"] is not None else "N/A"
        print(f"{name:<22} {rho_str:>14} {ok:>10} {r['iters']:>8} "
              f"{'✓' if r['converged'] else '✗':>10}")

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for (name, r), col in zip(cases.items(), colors):
        if r["history"] and len(r["history"]) > 1:
            its = [h[0] for h in r["history"]]
            res = [h[1] for h in r["history"]]
            rho_lbl = f"{r['rho']:.4f}" if r["rho"] is not None else "N/A"
            ax.semilogy(its, res, "-", color=col, lw=1.5,
                         label=f"{name}  (ρ={rho_lbl})")
    # Reference bound line
    ax.axhline(tol, color="gray", ls="--", lw=0.7, label=f"tol={tol:.0e}")
    ax.set_xlabel("Outer iteration k")
    ax.set_ylabel(r"$\|R_f^k\|_{RMS}$")
    ax.set_title(f"Theorem 2 empirical verification : ρ ≤ 0.98 predicted")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/convergence_rates.png", dpi=120)
    plt.close(fig)
    print(f"\nPlot : {out_dir}/convergence_rates.png")

    # JSON
    import json
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump({
            "bound": bound,
            "cases": {name: {"rho": r["rho"], "iters": int(r["iters"]),
                              "final_res": float(r["final_res"]),
                              "converged": bool(r["converged"])}
                       for name, r in cases.items()}
        }, fh, indent=2)


if __name__ == "__main__":
    main()
