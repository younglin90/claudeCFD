"""Phase-2 : SCMK-LBM vs baseline LBM on channel flow with bounce-back walls.

Spectral PC built under periodic assumption is mismatched at walls.
Measure speedup degradation vs Phase-1 (fully periodic).
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_channel import ChannelCase
from lbm_periodic import build_spectral_schur
from solver_scmk import solve_scmk, solve_baseline_periodic

# Phase-4 : use mode (0,0) = I regularization in build_spectral_schur


def main():
    os.makedirs("results_channel_phase4", exist_ok=True)
    out = "results_channel_phase4"

    N = 64
    nu = 0.05
    F0 = 1.0e-6
    tol = 1e-9

    case_b = ChannelCase(N=N, nu=nu, F0=F0)
    case_s = ChannelCase(N=N, nu=nu, F0=F0)

    print(f"Channel : N={N}  nu={nu}  F0={F0}")
    print(f"  omega={case_b.omega:.4f}  U_max_analytic={case_b.U_max:.5f}  Re={case_b.Re:.2f}")
    print(f"  tol={tol:.0e}")

    print("\n--- BASELINE (LBM Picard) ---")
    t0 = time.perf_counter()
    f_b, hist_b = solve_baseline_periodic(case_b, max_steps=200000, tol=tol, check_every=500)
    wall_b = time.perf_counter() - t0

    print("\n--- SCMK-LBM (JFNK + spectral Schur PC, periodic assumption) ---")
    S_inv = build_spectral_schur(N, omega=case_s.omega, mode="ap")
    t0 = time.perf_counter()
    f_s, hist_s = solve_scmk(
        case_s, S_inv,
        max_outer=200,
        tol=tol,
        krylov_max=15,
        krylov_tol=1e-3,
        line_search_max=6,
        kinetic_substeps=20,
    )
    wall_s = time.perf_counter() - t0

    lbe_b = hist_b[-1][2]; lbe_s = hist_s[-1][2]
    res_b = hist_b[-1][1]; res_s = hist_s[-1][1]
    iter_b = hist_b[-1][0]; iter_s = hist_s[-1][0]

    speedup_wall = wall_b / max(wall_s, 1e-12)
    speedup_lbe = lbe_b / max(lbe_s, 1)

    _, ux_b, _ = case_b.macro(f_b)
    _, ux_s, _ = case_s.macro(f_s)
    ux_ref = case_b.analytical_ux()
    err_b = np.linalg.norm(ux_b - ux_ref) / np.linalg.norm(ux_ref)
    err_s = np.linalg.norm(ux_s - ux_ref) / np.linalg.norm(ux_ref)
    agree = np.linalg.norm(ux_b - ux_s) / np.linalg.norm(ux_ref)

    summary = {
        "case": {"N": N, "nu": nu, "F0": F0, "tol": tol,
                  "omega": case_b.omega, "U_max": case_b.U_max, "Re": case_b.Re},
        "baseline": {"steps": int(iter_b), "lbe_calls": int(lbe_b),
                     "wall_s": float(wall_b), "final_res": float(res_b),
                     "err_vs_analytical": float(err_b)},
        "scmk": {"outer_iters": int(iter_s), "lbe_calls": int(lbe_s),
                 "wall_s": float(wall_s), "final_res": float(res_s),
                 "err_vs_analytical": float(err_s)},
        "speedup": {"wall": float(speedup_wall), "lbe_calls": float(speedup_lbe)},
        "field_agreement": float(agree),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("results_channel_phase4/summary.txt", "w") as fh:
        fh.write(json.dumps(summary, indent=2))

    # Convergence
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bx = np.array([h[2] for h in hist_b]); by = np.array([h[1] for h in hist_b])
    sx = np.array([h[2] for h in hist_s]); sy = np.array([h[1] for h in hist_s])
    ax.semilogy(bx, by, "b-", lw=1.5, label=f"Baseline LBM ({lbe_b} LBE, {wall_b:.1f}s)")
    ax.semilogy(sx, sy, "ro-", ms=4, label=f"SCMK-LBM ({lbe_s} LBE, {wall_s:.1f}s)")
    ax.axhline(tol, color="gray", ls="--", lw=0.8, label=f"tol={tol:.0e}")
    ax.set_xlabel("LBE residual evaluations")
    ax.set_ylabel(r"$\|R_f\|_{RMS}$")
    ax.set_title(
        f"Channel (bounce-back walls)  N={N} Re={case_b.Re:.1f}  |  "
        f"speedup x{speedup_lbe:.1f} calls, x{speedup_wall:.1f} wall"
    )
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results_channel_phase4/convergence.png", dpi=120)
    plt.close(fig)
    print("Plot: results_channel/convergence.png")

    # Profile
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    y = np.arange(N)
    ax.plot(case_b.analytical_ux()[:, 0], y, "k--", lw=2, label="Analytical Poiseuille")
    ax.plot(ux_b[:, 0], y, "b-", lw=1.5, label=f"Baseline (err={err_b:.2e})")
    ax.plot(ux_s[:, 0], y, "ro", ms=3, label=f"SCMK (err={err_s:.2e})")
    ax.set_xlabel(r"$u_x$"); ax.set_ylabel("y (lattice)")
    ax.set_title("Channel steady velocity profile")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results_channel_phase4/profile.png", dpi=120)
    plt.close(fig)
    print("Plot: results_channel/profile.png")


if __name__ == "__main__":
    main()
