"""Generate side-by-side baseline vs SCMK contour plots for all 12 cases.

Output: results_contours/{case}.png (each 2-panel : baseline ux, SCMK ux + streamlines).
"""

import os, time
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


def macro_of(case, f):
    if hasattr(case, "macro"):
        m = case.macro(f)
        return m
    return cavity_moments(f)


def plot_2d_pair(case_b, case_s, f_b, f_s, label, out_path,
                  mask=None, show_stream=True):
    """Side-by-side 2D ux contour : baseline (left), SCMK (right)."""
    macro_b = macro_of(case_b, f_b)
    macro_s = macro_of(case_s, f_s)
    ux_b, uy_b = macro_b[1], macro_b[2]
    ux_s, uy_s = macro_s[1], macro_s[2]

    if mask is not None:
        ux_b = np.ma.masked_where(mask < 0.5, ux_b)
        ux_s = np.ma.masked_where(mask < 0.5, ux_s)

    vmax = max(abs(ux_b).max(), abs(ux_s).max())
    vmin = -vmax if ux_b.min() < 0 else 0
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axs[0].imshow(ux_b, origin="lower", cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Baseline LBM : u_x")
    plt.colorbar(im0, ax=axs[0], shrink=0.85)
    im1 = axs[1].imshow(ux_s, origin="lower", cmap="RdBu_r",
                          vmin=vmin, vmax=vmax)
    axs[1].set_title(f"SCMK-LBM : u_x")
    plt.colorbar(im1, ax=axs[1], shrink=0.85)
    if show_stream and ux_b.ndim == 2:
        Y, X = np.mgrid[0:ux_b.shape[0], 0:ux_b.shape[1]]
        try:
            for ax_, ux_, uy_ in zip(axs, [ux_b, ux_s], [uy_b, uy_s]):
                ax_.streamplot(X, Y, ux_, uy_, color="k", density=1.2, linewidth=0.5,
                                arrowsize=0.6)
        except Exception:
            pass
    fig.suptitle(label, fontsize=12)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_3d_slice(case_b, case_s, f_b, f_s, label, out_path):
    """3D : mid-slice along z=N/2 of u_x."""
    macro_b = macro_of(case_b, f_b)
    macro_s = macro_of(case_s, f_s)
    ux_b, ux_s = macro_b[1], macro_s[1]
    z_mid = ux_b.shape[0] // 2
    slc_b = ux_b[z_mid]
    slc_s = ux_s[z_mid]
    vmax = max(abs(slc_b).max(), abs(slc_s).max())
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axs[0].imshow(slc_b, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
    axs[0].set_title(f"Baseline LBM (z=N/2)")
    plt.colorbar(im0, ax=axs[0], shrink=0.85)
    im1 = axs[1].imshow(slc_s, origin="lower", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
    axs[1].set_title(f"SCMK-LBM (z=N/2)")
    plt.colorbar(im1, ax=axs[1], shrink=0.85)
    fig.suptitle(label + "  (3D mid-z slice)", fontsize=12)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    out = "results_contours"
    os.makedirs(out, exist_ok=True)
    tol = 1e-7

    # ----------------- 2D periodic Kolmogorov -----------------
    print("[ 1] Kolmogorov N=32")
    c = KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    c2 = KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    f_b, _ = solve_baseline_periodic(c, max_steps=50000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=200, tol=tol, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Kolmogorov flow, N=32", f"{out}/kolmogorov_N32.png")

    # ----------------- Channel -----------------
    print("[ 2] Channel N=32")
    c = ChannelCase(N=32, nu=0.05, F0=1e-5)
    c2 = ChannelCase(N=32, nu=0.05, F0=1e-5)
    f_b, _ = solve_baseline_periodic(c, max_steps=50000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=200, tol=tol, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Channel Poiseuille flow, N=32",
                  f"{out}/channel_N32.png")

    # ----------------- Couette -----------------
    print("[ 3] Couette N=32")
    c = CouetteCase(N=32, nu=0.05, U_wall=0.05)
    c2 = CouetteCase(N=32, nu=0.05, U_wall=0.05)
    f_b, _ = solve_baseline_periodic(c, max_steps=50000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=200, tol=tol, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Couette flow, N=32", f"{out}/couette_N32.png")

    # ----------------- Cavity Re=100 -----------------
    print("[ 4] Cavity Re=100 N=65")
    c = LBMCavity(N=65, Re=100, U_wall=0.1)
    c2 = LBMCavity(N=65, Re=100, U_wall=0.1)
    f_b, _ = solve_baseline(c, max_steps=200000, tol=5e-8, check_every=2000, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=300, tol=5e-8, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Lid-driven cavity, Re=100, N=65",
                  f"{out}/cavity_Re100.png")

    # ----------------- Cavity Re=400 -----------------
    print("[ 5] Cavity Re=400 N=97")
    c = LBMCavity(N=97, Re=400, U_wall=0.1)
    c2 = LBMCavity(N=97, Re=400, U_wall=0.1)
    f_b, _ = solve_baseline(c, max_steps=400000, tol=5e-8, check_every=2000, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=500, tol=5e-8, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Lid-driven cavity, Re=400, N=97",
                  f"{out}/cavity_Re400.png")

    # ----------------- Multi-cylinder -----------------
    print("[ 6] Multi-cylinder N=32")
    N = 32; chi = np.ones((N, N))
    rng = np.random.RandomState(7)
    for _ in range(6):
        r = max(2, N // 12)
        cx = rng.randint(r, N - r); cy = rng.randint(r, N - r)
        chi *= build_cylinder_mask(N, cx, cy, r)
    c = VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    c2 = VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    f_b, _ = solve_baseline_periodic(c, max_steps=50000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_hybrid(c2, max_outer=200, tol=tol, krylov_max=10,
                            kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_2d_pair(c, c2, f_b, f_s, "Multi-cylinder voxel flow, N=32",
                  f"{out}/multi_cylinder.png", mask=chi)

    # ----------------- 3D Kolmogorov N=24 -----------------
    print("[ 7] 3D Kolmogorov N=24")
    c = Kolmogorov3DCase(N=24, nu=0.05, F0=2e-4, kf=1)
    c2 = Kolmogorov3DCase(N=24, nu=0.05, F0=2e-4, kf=1)
    f_b, _ = solve_baseline_3d(c, max_steps=30000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_scmk_3d(c2, max_outer=150, tol=tol, krylov_max=10,
                             kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_3d_slice(c, c2, f_b, f_s, "3D Kolmogorov flow, N=24", f"{out}/3d_kolmogorov.png")

    # ----------------- 3D Channel N=24 -----------------
    print("[ 8] 3D Channel N=24")
    c = Channel3DCase(N=24, nu=0.05, F0=1e-4)
    c2 = Channel3DCase(N=24, nu=0.05, F0=1e-4)
    f_b, _ = solve_baseline_3d(c, max_steps=30000, tol=tol, check_every=200, verbose=False)
    f_s, _ = solve_scmk_3d(c2, max_outer=150, tol=tol, krylov_max=10,
                             kinetic_substeps=15, N_check=6, min_ratio=2.0, verbose=False)
    plot_3d_slice(c, c2, f_b, f_s, "3D Channel Poiseuille flow, N=24",
                   f"{out}/3d_channel.png")

    print("Done. Contours saved to", out)


if __name__ == "__main__":
    main()
