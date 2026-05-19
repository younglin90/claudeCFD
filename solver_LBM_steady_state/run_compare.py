"""Compare baseline LBM vs AP-MoMeNt-LBM on lid-driven cavity steady state.

Outputs:
    results/convergence.png   - residual vs LBE-call count (log)
    results/centerline.png    - u(y) on x=0.5 and v(x) on y=0.5
    results/summary.txt       - iteration counts, wall times, speedup
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_core import LBMCavity, moments
from solver_baseline import solve_baseline
from solver_apmnt import solve_apmnt


def macro_field(case, f):
    rho, ux, uy = moments(f)
    return rho, ux, uy


def main():
    os.makedirs("results", exist_ok=True)

    N = 65
    Re = 100.0
    U_wall = 0.1
    tol = 1e-6

    case_b = LBMCavity(N=N, Re=Re, U_wall=U_wall)
    case_a = LBMCavity(N=N, Re=Re, U_wall=U_wall)

    print(f"Case: lid-driven cavity, N={N}, Re={Re}, U_wall={U_wall}")
    print(f"      nu={case_b.nu:.5f}, omega={case_b.omega:.5f}, tol={tol:.0e}")
    print()

    print("--- BASELINE (LBM time-march) ---")
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline(case_b, max_steps=200000, tol=tol, check_every=200)
    wall_b = time.perf_counter() - t0

    print()
    print("--- AP-MoMeNt-LBM (Newton-Krylov) ---")
    t0 = time.perf_counter()
    f_a, hist_a = solve_apmnt(
        case_a,
        max_outer=120,
        tol=tol,
        krylov_max=5,
        krylov_tol=5e-2,
        kinetic_steps=20,
        schur_mode="apmnt",
        line_search_max=3,
    )
    wall_a = time.perf_counter() - t0

    res_b_final = hist_b[-1][1]
    res_a_final = hist_a[-1][1]
    lbe_b = hist_b[-1][2]
    lbe_a = hist_a[-1][2]
    outer_b = hist_b[-1][0]
    outer_a = hist_a[-1][0]

    speedup_wall = wall_b / max(wall_a, 1e-12)
    speedup_lbe = lbe_b / max(lbe_a, 1)

    summary = {
        "case": {"N": N, "Re": Re, "U_wall": U_wall, "tol": tol},
        "baseline": {
            "steps": int(outer_b),
            "lbe_calls": int(lbe_b),
            "wall_s": float(wall_b),
            "final_res": float(res_b_final),
        },
        "apmnt": {
            "outer_iters": int(outer_a),
            "lbe_calls": int(lbe_a),
            "wall_s": float(wall_a),
            "final_res": float(res_a_final),
        },
        "speedup": {"wall": float(speedup_wall), "lbe_calls": float(speedup_lbe)},
    }

    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("results/summary.txt", "w") as fh:
        fh.write(json.dumps(summary, indent=2))

    # Convergence plot vs LBE-call count
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bx = np.array([h[2] for h in hist_b])
    by = np.array([h[1] for h in hist_b])
    ax_x = np.array([h[2] for h in hist_a])
    ax_y = np.array([h[1] for h in hist_a])
    ax.semilogy(bx, by, "b-", label=f"Baseline LBM ({lbe_b} LBE calls)")
    ax.semilogy(ax_x, ax_y, "ro-", ms=4, label=f"AP-MoMeNt-LBM ({lbe_a} LBE calls)")
    ax.axhline(tol, color="gray", ls="--", lw=0.8, label=f"tol={tol:.0e}")
    ax.set_xlabel("LBE residual evaluations")
    ax.set_ylabel(r"$\|R_f\|_{RMS}$")
    ax.set_title(f"Cavity Re={Re:.0f}, N={N}  |  speedup x{speedup_lbe:.1f} (calls)  /  x{speedup_wall:.1f} (wall)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/convergence.png", dpi=120)
    plt.close(fig)
    print("Plot saved: results/convergence.png")

    # Centerline velocity comparison
    rho_b, ux_b, uy_b = macro_field(case_b, f_b)
    rho_a, ux_a, uy_a = macro_field(case_a, f_a)
    mid = N // 2
    y = np.arange(N) / (N - 1)
    x = np.arange(N) / (N - 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(ux_b[:, mid] / U_wall, y, "b-", label="Baseline")
    axs[0].plot(ux_a[:, mid] / U_wall, y, "ro", ms=4, label="AP-MoMeNt")
    axs[0].set_xlabel(r"$u_x / U_{wall}$")
    axs[0].set_ylabel("y / L")
    axs[0].set_title("Vertical centerline u_x")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(x, uy_b[mid, :] / U_wall, "b-", label="Baseline")
    axs[1].plot(x, uy_a[mid, :] / U_wall, "ro", ms=4, label="AP-MoMeNt")
    axs[1].set_xlabel("x / L")
    axs[1].set_ylabel(r"$u_y / U_{wall}$")
    axs[1].set_title("Horizontal centerline u_y")
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/centerline.png", dpi=120)
    plt.close(fig)
    print("Plot saved: results/centerline.png")

    # L2 difference between two solutions
    err_ux = np.linalg.norm(ux_b - ux_a) / np.linalg.norm(ux_b)
    err_uy = np.linalg.norm(uy_b - uy_a) / max(np.linalg.norm(uy_b), 1e-30)
    print(f"\nField agreement: rel L2 err_ux={err_ux:.3e}, err_uy={err_uy:.3e}")
    with open("results/summary.txt", "a") as fh:
        fh.write(f"\nField agreement: err_ux={err_ux:.3e}, err_uy={err_uy:.3e}\n")


if __name__ == "__main__":
    main()
