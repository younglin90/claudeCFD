"""SCMK-LBM Phase-6 : 2-level V-cycle preconditioner with kinetic-aware transfer.

PC structure:
    Outer FGMRES:  J f = -R
        right-precond  z = Vcycle(r) approximating J^{-1} r

Vcycle(r_fine):
    (1) Pre-smooth on fine via spectral Schur :
            z = T_f S_f^{-1} M_f r_fine          (Phase-4 spectral PC, fine N)

    (2) Compute defect on fine :
            d = r_fine - J z                       (1 JVP)

    (3) Restrict macro defect to coarse :
            d_macro = M d
            d_macro_coarse = full-weight 2x2 average  (kinetic dropped)

    (4) Coarse solve via spectral Schur on coarse N :
            df_coarse = T_c S_c^{-1} d_macro_coarse

    (5) Prolongate :
            df_macro_fine = bilinear upsample df_coarse
            z += T_f · (M_f df_macro_fine)         (extract macro, re-lift)

    (6) Post-smooth via spectral Schur on fine :
            d2 = r_fine - J z                      (1 JVP)
            z += T_f S_f^{-1} M_f d2

    return z

Cost per Vcycle :  3 fine LBE evaluations  (2 JVPs + 1 implicit in spectral apply chain)
                 + 1 coarse spectral apply (cheap : FFT on (N/2)^2)
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def restrict_macro(U_fine):
    """Full-weighting 2x2 restriction. U_fine shape (3, N, N), N even."""
    N = U_fine.shape[1]
    Nc = N // 2
    return U_fine[:, : 2 * Nc, : 2 * Nc].reshape(3, Nc, 2, Nc, 2).mean(axis=(2, 4))


def prolongate_macro(U_coarse, N_fine):
    """Bilinear-ish upsample (nearest-neighbor repeat). Returns shape (3, N_fine, N_fine)."""
    Nc = U_coarse.shape[1]
    assert 2 * Nc == N_fine
    return np.repeat(np.repeat(U_coarse, 2, axis=1), 2, axis=2)


def make_vcycle_pc(case, S_inv_fine, S_inv_coarse, f_base_holder, R_base_holder, norm_f_holder):
    """Build a function that applies 2-level V-cycle as preconditioner."""
    N_fine = case.N
    N_coarse = N_fine // 2

    # Macro projection helper for coarse-sized residuals (we only need linear lift/project
    # in macro space, geometry-free).
    def lift_macro_to_distribution(dU, target_N):
        """Apply linear lift (M T = I) at any size."""
        from lbm_periodic import CX, CY, W
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9, target_N, target_N), dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    def project_distribution_to_macro(f):
        from lbm_periodic import CX, CY
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def vcycle(r_fine):
        f_base = f_base_holder[0]
        R_base = R_base_holder[0]
        norm_f = norm_f_holder[0]

        # (1) Pre-smooth : fine spectral Schur
        z = apply_spectral_schur(case, r_fine, S_inv_fine)

        # (2) Defect : d = r - J z   (one JVP)
        Jz = case.jvp(z, f_base, R_base, norm_f_cached=norm_f)
        d = r_fine - Jz

        # (3) Restrict macro defect
        d_macro_fine = case.project(d)
        d_macro_coarse = restrict_macro(d_macro_fine)

        # (4) Coarse spectral Schur solve (in macro space directly, via lift/project)
        d_coarse_f = lift_macro_to_distribution(d_macro_coarse, N_coarse)
        # Apply spectral Schur on coarse :
        #   R_U_hat (FFT of macro)  -> S_c^{-1} -> dU_hat -> IFFT -> dU
        # Wrap into a temporary "case-like" interface for apply_spectral_schur
        class _CoarseCase:
            pass
        cc = _CoarseCase()
        cc.shape = (9, N_coarse, N_coarse)
        cc.project = project_distribution_to_macro
        cc.lift = lambda dU: lift_macro_to_distribution(dU, N_coarse)
        dz_coarse_f = apply_spectral_schur(cc, d_coarse_f, S_inv_coarse)
        dz_macro_coarse = project_distribution_to_macro(dz_coarse_f)

        # (5) Prolongate macro -> fine, lift to distribution, add
        dz_macro_fine = prolongate_macro(dz_macro_coarse, N_fine)
        z = z + case.lift(dz_macro_fine)

        # (6) Post-smooth
        Jz2 = case.jvp(z, f_base, R_base, norm_f_cached=norm_f)
        d2 = r_fine - Jz2
        z = z + apply_spectral_schur(case, d2, S_inv_fine)

        return z

    return vcycle


def solve_scmk_v6(case, max_outer=80, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                   line_search_max=5, kinetic_substeps=20, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    N_fine = case.N
    N_coarse = N_fine // 2

    # build both PCs
    S_inv_fine = build_spectral_schur(N_fine, omega=case.omega, mode="ap")
    S_inv_coarse = build_spectral_schur(N_coarse, omega=case.omega, mode="ap")

    f_holder = [f]
    R_holder = [None]
    norm_holder = [0.0]

    vcycle = make_vcycle_pc(case, S_inv_fine, S_inv_coarse,
                             f_holder, R_holder, norm_holder)

    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R_f = case.residual(f)
        lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  outer {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose:
                print(f"  CONVERGED at outer {k}")
            break

        # update holders for closure
        f_holder[0] = f
        R_holder[0] = R_f
        norm_holder[0] = case._fast_norm(f)

        probe_count = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe_count[0] += 1
            return case.jvp(w, f, R_f, norm_f_cached=norm_holder[0]).ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            # vcycle itself adds 2 JVPs internally
            probe_count[0] += 2
            return vcycle(R).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R_f.ravel()
        df_flat, info = gmres(Aop, rhs, M=Mop,
                              rtol=krylov_tol, atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                              maxiter=2, restart=krylov_max)
        lbe_calls += probe_count[0]

        if not np.all(np.isfinite(df_flat)):
            print("  GMRES NaN, abort")
            break

        df = df_flat.reshape(case.shape)

        # composite line search
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            f_trial = f + alpha * df
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps + 1
            R_trial = f_trial - case.lbe_step(f_trial)
            r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
            if r_trial < res_norm:
                f = f_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # pure smoother step
            for _ in range(kinetic_substeps):
                f = case.lbe_step(f)
            lbe_calls += kinetic_substeps
            if verbose:
                print(f"     line search fail -> smoother")

    return f, history
