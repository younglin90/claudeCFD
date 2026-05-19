"""Ghia 1982 lid-driven cavity literature validation.

Ghia, Ghia, Shin (1982) "High-Re solutions for incompressible flow using the
Navier-Stokes equations and a multigrid method", JCP 48, 387-411.

Provides reference centerline velocity profiles for Re = 100, 400, 1000, 3200, 5000.

We compare SCMK steady-state result against Ghia table values for Re = 100, 400, 1000.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ghia 1982 Table I : u-velocity along vertical centerline x = 0.5
# y values (non-dimensional, 0=bottom, 1=top)
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000
])

# u/U_lid at Re=100
GHIA_U_RE100 = np.array([
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,  0.78871,  0.84123, 1.00000
])

# u/U_lid at Re=400
GHIA_U_RE400 = np.array([
    0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119,
    -0.11477,  0.02135,  0.16256,  0.29093,  0.55892,  0.61756,  0.68439,  0.75837, 1.00000
])

# u/U_lid at Re=1000
GHIA_U_RE1000 = np.array([
    0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648,
    -0.06080,  0.05702,  0.18719,  0.33304,  0.46604,  0.51117,  0.57492,  0.65928, 1.00000
])

# Ghia Table II : v-velocity along horizontal centerline y = 0.5
GHIA_X = np.array([
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000
])

GHIA_V_RE100 = np.array([
    0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,  0.17507,  0.17527,
    0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000
])

GHIA_V_RE400 = np.array([
    0.00000,  0.18360,  0.19713,  0.20920,  0.22965,  0.28124,  0.30203,  0.30174,
    0.05186, -0.38598, -0.44993, -0.33827, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000
])

GHIA_V_RE1000 = np.array([
    0.00000,  0.27485,  0.29012,  0.30353,  0.32627,  0.37095,  0.33075,  0.32235,
    0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000
])


def get_ghia_data(Re):
    """Returns (y_centerline_u, u_centerline, x_centerline_v, v_centerline) for given Re."""
    if Re == 100:
        return GHIA_Y, GHIA_U_RE100, GHIA_X, GHIA_V_RE100
    elif Re == 400:
        return GHIA_Y, GHIA_U_RE400, GHIA_X, GHIA_V_RE400
    elif Re == 1000:
        return GHIA_Y, GHIA_U_RE1000, GHIA_X, GHIA_V_RE1000
    else:
        raise ValueError(f"Ghia 1982 data unavailable for Re={Re}")


def extract_centerline(f, case, U_lid):
    """Extract u(y) at x_center and v(x) at y_center from cavity simulation.

    Returns normalized profiles : u_x / U_lid at vertical centerline,
                                  u_y / U_lid at horizontal centerline.
    """
    from lbm_core import moments
    rho, ux, uy = moments(f)
    N = case.N
    mid_x = N // 2
    mid_y = N // 2
    y_grid = np.linspace(0.0, 1.0, N)
    x_grid = np.linspace(0.0, 1.0, N)
    u_vert = ux[:, mid_x] / U_lid
    v_horiz = uy[mid_y, :] / U_lid
    return y_grid, u_vert, x_grid, v_horiz


def compare_with_ghia(f, case, Re, output_path):
    """Generate Ghia comparison plot."""
    U_lid = case.U_wall
    y_grid, u_vert, x_grid, v_horiz = extract_centerline(f, case, U_lid)
    y_g, u_g, x_g, v_g = get_ghia_data(Re)

    # interpolate SCMK at Ghia y-points for error metric
    u_interp = np.interp(y_g, y_grid, u_vert)
    v_interp = np.interp(x_g, x_grid, v_horiz)
    err_u = float(np.sqrt(np.mean((u_interp - u_g) ** 2)))
    err_v = float(np.sqrt(np.mean((v_interp - v_g) ** 2)))

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].plot(u_vert, y_grid, "b-", lw=1.5, label=f"SCMK (N={case.N})")
    axs[0].plot(u_g, y_g, "ro", ms=5, label="Ghia 1982")
    axs[0].set_xlabel(r"$u/U_{lid}$"); axs[0].set_ylabel(r"$y/L$")
    axs[0].set_title(f"Vertical centerline, Re={Re} (RMS err {err_u:.3e})")
    axs[0].legend(); axs[0].grid(alpha=0.3)

    axs[1].plot(x_grid, v_horiz, "b-", lw=1.5, label=f"SCMK (N={case.N})")
    axs[1].plot(x_g, v_g, "ro", ms=5, label="Ghia 1982")
    axs[1].set_xlabel(r"$x/L$"); axs[1].set_ylabel(r"$v/U_{lid}$")
    axs[1].set_title(f"Horizontal centerline, Re={Re} (RMS err {err_v:.3e})")
    axs[1].legend(); axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return err_u, err_v


def main():
    import time
    from lbm_core import LBMCavity
    from solver_baseline import solve_baseline
    from solver_hybrid import solve_hybrid

    out_dir = "results_ghia"
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("Re=100",  100,  65, 0.1, 5e-7, 100000),
        ("Re=400_N65",  400,  65, 0.1, 5e-7, 150000),
        ("Re=400_N129", 400, 129, 0.1, 5e-7, 200000),
    ]

    results = {}
    for label, Re, N, U_wall, tol, max_steps in cases:
        print(f"\n=== Cavity {label} N={N} U={U_wall} ===")
        case_b = LBMCavity(N=N, Re=Re, U_wall=U_wall)
        case_s = LBMCavity(N=N, Re=Re, U_wall=U_wall)

        t0 = time.perf_counter()
        f_b, hist_b = solve_baseline(case_b, max_steps=max_steps, tol=tol,
                                       check_every=500, verbose=False)
        wall_b = time.perf_counter() - t0
        print(f"  baseline: step {hist_b[-1][0]} lbe {hist_b[-1][2]} res {hist_b[-1][1]:.2e} wall {wall_b:.1f}s")

        t0 = time.perf_counter()
        f_s, hist_s = solve_hybrid(case_s, max_outer=300, tol=tol,
                                     krylov_max=10, kinetic_substeps=15,
                                     N_check=6, min_ratio=2.0, verbose=False)
        wall_s = time.perf_counter() - t0
        print(f"  SCMK    : outer {hist_s[-1][0]} lbe {hist_s[-1][2]} res {hist_s[-1][1]:.2e} wall {wall_s:.1f}s")
        print(f"  speedup : {hist_b[-1][2]/hist_s[-1][2]:.2f}x LBE, {wall_b/wall_s:.2f}x wall")

        err_u_s, err_v_s = compare_with_ghia(f_s, case_s, Re,
                                              f"{out_dir}/ghia_Re{Re}_SCMK.png")
        err_u_b, err_v_b = compare_with_ghia(f_b, case_b, Re,
                                              f"{out_dir}/ghia_Re{Re}_baseline.png")
        print(f"  Ghia err (RMS) baseline : u {err_u_b:.3e}, v {err_v_b:.3e}")
        print(f"  Ghia err (RMS) SCMK     : u {err_u_s:.3e}, v {err_v_s:.3e}")
        results[label] = {
            "Re": Re, "N": N, "U_wall": U_wall,
            "baseline_lbe": int(hist_b[-1][2]), "baseline_wall": float(wall_b),
            "baseline_res": float(hist_b[-1][1]),
            "scmk_lbe": int(hist_s[-1][2]), "scmk_wall": float(wall_s),
            "scmk_res": float(hist_s[-1][1]),
            "speedup_lbe": float(hist_b[-1][2]/hist_s[-1][2]),
            "speedup_wall": float(wall_b / wall_s),
            "ghia_err_u_baseline": err_u_b, "ghia_err_v_baseline": err_v_b,
            "ghia_err_u_scmk": err_u_s, "ghia_err_v_scmk": err_v_s,
        }

    print("\n" + "=" * 80)
    print(f"{'Case':<10} {'speedup':>10} {'wall':>9} {'baseline u-err':>16} {'SCMK u-err':>12}")
    print("-" * 80)
    for label, r in results.items():
        print(f"{label:<10} {r['speedup_lbe']:>9.1f}x {r['speedup_wall']:>8.1f}x "
              f"{r['ghia_err_u_baseline']:>16.3e} {r['ghia_err_u_scmk']:>12.3e}")

    import json
    with open(f"{out_dir}/summary.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
