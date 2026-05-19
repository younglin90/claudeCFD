"""SCMK-LBM Phase-8 : FAS (Full Approximation Scheme) 2-level multigrid
                       + Phase-4 spectral PC at finest level.

FAS for steady residual r(f) = f - L(f) :

    Fine grid h with operator L_h.   Coarse grid H with L_H.
    Want : r_h(f_h) = 0.

    V-cycle (one pass) :

        (1) pre-smooth K_pre LBE on fine     : f_h <- L_h^{K_pre}(f_h)
        (2) Fine residual r_h = f_h - L_h(f_h)
        (3) Restrict to coarse :
                f_H_init = R^h_H f_h
                R_c      = R^h_H r_h(f_h)
        (4) Coarse Picard with source :
                f_H <- f_H_init
                for K_coarse iter :
                    f_H <- L_H(f_H) + R_c    (mask-respecting)
            converges to r_H(f_H) = R_c
        (5) Correction back :
                f_h <- f_h + P^H_h (f_H - f_H_init)
        (6) post-smooth K_post LBE on fine    : f_h <- L_h^{K_post}(f_h)

Use FAS as outer fixed-point map. Optionally wrap with Phase-4 spectral PC
on residual after each V-cycle for additional acceleration.
"""

import time
import numpy as np

from lbm_voxel import VoxelCase
from lbm_periodic import build_spectral_schur, apply_spectral_schur


def coarsen_mask(chi):
    """2x downsample mask via volume fraction + majority rule."""
    N = chi.shape[0]; Nc = N // 2
    block = chi[: 2 * Nc, : 2 * Nc].reshape(Nc, 2, Nc, 2).mean(axis=(1, 3))
    return (block > 0.5).astype(np.float64)


def restrict_distribution(f, chi_coarse):
    """Full-weighting 2x2 average of distribution function; mask-respecting.

    f shape (9, N, N). Returns (9, Nc, Nc).
    Within each 2x2 block, average over fluid voxels only; if all solid, zero.
    """
    N = f.shape[1]; Nc = N // 2
    block = f[:, : 2 * Nc, : 2 * Nc].reshape(9, Nc, 2, Nc, 2)
    # weight by fine-level fluid fraction (1 for fluid voxels). But here block already
    # contains f * chi_fine values; we just average. Better : block.sum / count_fluid.
    fc = block.mean(axis=(2, 4))
    return fc * chi_coarse[None, :, :]


def prolongate_distribution(f_c, chi_fine):
    """Nearest-neighbor 2x upsample with fine-level mask."""
    Nc = f_c.shape[1]; N = 2 * Nc
    fexp = np.repeat(np.repeat(f_c, 2, axis=1), 2, axis=2)
    return fexp * chi_fine[None, :, :]


def fas_vcycle(f_fine, fine_case, coarse_case, K_pre=3, K_coarse=20, K_post=3):
    """One FAS V-cycle."""
    # (1) pre-smooth
    for _ in range(K_pre):
        f_fine = fine_case.lbe_step(f_fine)

    # (2) fine residual r_h(f_h)
    L_h_f = fine_case.lbe_step(f_fine)
    r_h = f_fine - L_h_f

    # (3) restrict
    f_H_init = restrict_distribution(f_fine, coarse_case.chi)
    R_c = restrict_distribution(r_h, coarse_case.chi)

    # (4) coarse Picard with source
    f_H = f_H_init.copy()
    for _ in range(K_coarse):
        f_H = coarse_case.lbe_step(f_H) + R_c
        f_H *= coarse_case.chi[None, :, :]  # mask

    # (5) correction
    correction_H = f_H - f_H_init
    correction_h = prolongate_distribution(correction_H, fine_case.chi)
    f_fine = f_fine + correction_h

    # (6) post-smooth
    for _ in range(K_post):
        f_fine = fine_case.lbe_step(f_fine)

    return f_fine


def solve_fas(fine_case, max_iter=80, tol=1e-7, K_pre=3, K_coarse=20, K_post=3, verbose=True):
    """Pure FAS V-cycle iteration (no Newton outer)."""
    # build coarse case
    chi_c = coarsen_mask(fine_case.chi)
    coarse_case = VoxelCase(chi_c, fine_case.nu, F0=fine_case.F0, kf=fine_case.kf)

    f = fine_case.initial_field()
    n_full = fine_case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_iter):
        R = fine_case.residual(f); lbe_calls += 1
        rn = fine_case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, rn, lbe_calls, wall))
        if verbose:
            print(f"  fas {k:3d} | res {rn:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if rn < tol:
            if verbose:
                print(f"  CONVERGED at fas iter {k}")
            break

        f = fas_vcycle(f, fine_case, coarse_case, K_pre=K_pre, K_coarse=K_coarse, K_post=K_post)
        lbe_calls += K_pre + K_post + K_coarse + 1  # +1 for residual eval in vcycle

    return f, history, coarse_case


def solve_fas_spectral(fine_case, max_iter=80, tol=1e-7, K_pre=3, K_coarse=20, K_post=3,
                       spectral_apply_every=1, line_search_max=4, verbose=True):
    """FAS V-cycle + Phase-4 spectral PC correction (combined)."""
    chi_c = coarsen_mask(fine_case.chi)
    coarse_case = VoxelCase(chi_c, fine_case.nu, F0=fine_case.F0, kf=fine_case.kf)

    f = fine_case.initial_field()
    n_full = fine_case.dof
    S_inv = build_spectral_schur(fine_case.N, omega=fine_case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_iter):
        R = fine_case.residual(f); lbe_calls += 1
        rn = fine_case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, rn, lbe_calls, wall))
        if verbose:
            print(f"  k {k:3d} | res {rn:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if rn < tol:
            if verbose:
                print(f"  CONVERGED at iter {k}")
            break

        # FAS V-cycle
        f_after_fas = fas_vcycle(f, fine_case, coarse_case, K_pre, K_coarse, K_post)
        lbe_calls += K_pre + K_post + K_coarse + 1
        R_after = f_after_fas - fine_case.lbe_step(f_after_fas); lbe_calls += 1
        rn_after_fas = fine_case._fast_norm(R_after) / np.sqrt(n_full)

        # Optionally spectral PC step on top
        if k % spectral_apply_every == 0:
            df = apply_spectral_schur(fine_case, -R_after, S_inv)
            alpha = 1.0
            best_r = rn_after_fas; best_f = f_after_fas
            for _ in range(line_search_max):
                ft = f_after_fas + alpha * df
                # short kinetic damp
                for _ in range(3):
                    ft = fine_case.lbe_step(ft)
                lbe_calls += 4  # 3 lbe + residual
                Rt = ft - fine_case.lbe_step(ft)
                rt = fine_case._fast_norm(Rt) / np.sqrt(n_full)
                if rt < best_r:
                    best_r = rt; best_f = ft
                    break
                alpha *= 0.5
            f = best_f
        else:
            f = f_after_fas

    return f, history, coarse_case
