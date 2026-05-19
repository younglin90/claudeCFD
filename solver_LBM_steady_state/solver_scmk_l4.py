"""SCMK-LBM Phase-9 Layer-4 : Phase-4 spectral PC + cylinder-local macro GS smoother.

Layer 4 (from advice doc) :
    각 cylinder 인근 fluid voxel 에서 추가 macro Gauss-Seidel sweep.
    Standard LBE smoother 는 모든 cell 균등 처리하지만, cylinder shear-layer /
    recirculation 잔차가 가장 큼 -> 국소 강한 smoother 가 필요.

Per fine-level smoother sweep :
    (a) 표준 LBE substep
    (b) 각 cylinder neighborhood 에서  K_gs sweeps of 5-point macro Jacobi
    (c) re-lift to distribution preserving kinetic null-space
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def build_cylinder_neighborhoods(chi, cylinder_positions, cylinder_radii, r_aug=2):
    """For each cylinder, mask of fluid voxels within (radius + r_aug) of center."""
    N = chi.shape[0]
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    masks = []
    for (cx, cy), r in zip(cylinder_positions, cylinder_radii):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        nb = (d2 < (r + r_aug) ** 2) & (chi > 0.5)
        masks.append(nb)
    return masks


def union_neighborhoods(neighborhood_masks):
    """Single combined neighborhood mask."""
    if not neighborhood_masks:
        return None
    m = neighborhood_masks[0].copy()
    for nb in neighborhood_masks[1:]:
        m |= nb
    return m


def apply_layer4_smoother(case, f, combined_nb_mask, n_gs=2, damp=0.3):
    """Damped local macro Jacobi sweep in cylinder neighborhoods.

    Updates : u <- u + damp*(avg - u) restricted to mask.
    Then re-lifts distribution preserving kinetic null-space.
    Small damp prevents disturbing local force balance.
    """
    if combined_nb_mask is None or n_gs <= 0:
        return f

    from lbm_periodic import equilibrium

    rho, ux, uy = case.macro(f)
    rho0 = rho.copy(); ux0 = ux.copy(); uy0 = uy.copy()
    feq_old = equilibrium(rho0, ux0, uy0)

    for _ in range(n_gs):
        u_avg = ux.copy(); v_avg = uy.copy()
        u_avg[1:-1, 1:-1] = 0.25 * (ux[2:, 1:-1] + ux[:-2, 1:-1] +
                                     ux[1:-1, 2:] + ux[1:-1, :-2])
        v_avg[1:-1, 1:-1] = 0.25 * (uy[2:, 1:-1] + uy[:-2, 1:-1] +
                                     uy[1:-1, 2:] + uy[1:-1, :-2])
        # Damped update : only in neighborhood
        ux = np.where(combined_nb_mask, ux + damp * (u_avg - ux), ux)
        uy = np.where(combined_nb_mask, uy + damp * (v_avg - uy), uy)

    feq_new = equilibrium(rho, ux, uy)
    return f - feq_old + feq_new


def solve_scmk_l4(case, neighborhood_masks=None, max_outer=80, tol=1e-7,
                  krylov_max=10, krylov_tol=1e-3, line_search_max=5,
                  kinetic_substeps=15, n_gs=2, verbose=True):
    """SCMK Phase-4 with optional Layer-4 cylinder smoother inside line search."""
    f = case.initial_field()
    n_full = case.dof
    N = case.N
    S_inv = build_spectral_schur(N, omega=case.omega, mode="ap")

    combined_nb = union_neighborhoods(neighborhood_masks) if neighborhood_masks else None

    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R_f = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  outer {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # FGMRES inner
        norm_f = case._fast_norm(f)
        probe = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R_f, norm_f_cached=norm_f).ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            return apply_spectral_schur(case, R, S_inv).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R_f.ravel()
        df_flat, info = gmres(Aop, rhs, M=Mop, rtol=krylov_tol,
                              atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                              maxiter=2, restart=krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df_flat)):
            print("  GMRES NaN")
            break
        df = df_flat.reshape(case.shape)

        # composite line search with Layer-4 smoother augmenting LBE smoother
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            f_trial = f + alpha * df
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
                if combined_nb is not None:
                    f_trial = apply_layer4_smoother(case, f_trial, combined_nb, n_gs=n_gs)
            lbe_calls += kinetic_substeps + 1
            R_trial = f_trial - case.lbe_step(f_trial)
            r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
            if r_trial < res_norm:
                f = f_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            for _ in range(kinetic_substeps):
                f = case.lbe_step(f)
                if combined_nb is not None:
                    f = apply_layer4_smoother(case, f, combined_nb, n_gs=n_gs)
            lbe_calls += kinetic_substeps
            if verbose:
                print(f"     line search fail -> kinetic-only")

    return f, history
