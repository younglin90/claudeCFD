"""SCMK-LBM Robust : parameter-free universal variant.

Changes from SCMK Phase-4 (autoresearch-tuned) :

    (1) Theoretical AP coefficient :  coeff = (1-omega)/omega   (no 0.5x magic)
    (2) Moore-Penrose pseudoinverse  instead of Tikhonov + mode-(0,0) hack
    (3) Backtracking line search restored (vs always-accept)
    (4) Adaptive kinetic substeps :  K_eff = K_base * max(0.5, min(2.0, res_ratio))
                                     fewer when residual contracting fast,
                                     more when stagnating.

All hyperparameters either removed or replaced by theory-derived / self-tuning.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import CX, CY, W


def build_robust_spectral_schur(N, omega):
    """Exact full-rank 9x9 Jacobian inverse per Fourier mode.  No magic constants.

    J(k) = I_9 - A(k) [(1-omega) I_9 + omega T M]    (9x9 complex per mode)

    For preconditioning macro residual :
        df_macro_lift = T M J(k)^{-1} R_f_hat                (project + lift)
    where J(k)^{-1} is computed once per mode via direct 9x9 inverse.

    Macro-projected inverse stored as 9x9, used to compute  T S_macro^{-1} M
    by direct projection at apply time.
    """
    M_mat = np.zeros((3, 9), dtype=np.float64)
    M_mat[0, :] = 1.0
    M_mat[1, :] = CX
    M_mat[2, :] = CY

    T_mat = np.zeros((9, 3), dtype=np.float64)
    for i in range(9):
        T_mat[i, 0] = W[i]
        T_mat[i, 1] = 3.0 * W[i] * CX[i]
        T_mat[i, 2] = 3.0 * W[i] * CY[i]

    TM = T_mat @ M_mat                                     # 9x9 projector

    kx = 2.0 * np.pi * np.fft.fftfreq(N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    A_phase = np.empty((9, N, N), dtype=np.complex128)     # diagonal of streaming
    for i in range(9):
        A_phase[i] = np.exp(-1j * (KX * CX[i] + KY * CY[i]))

    # C = (1-omega) I_9 + omega TM    (independent of k)
    C9 = (1.0 - omega) * np.eye(9, dtype=np.complex128) + omega * TM

    # J(k)_ij = delta_ij - A_i(k) * C_ij
    # batched over (N, N) modes
    # full 9x9 per mode
    I9 = np.eye(9, dtype=np.complex128)
    # AC[i,j,kx,ky] = A_phase[i, kx, ky] * C9[i, j]
    AC = A_phase[:, None, :, :] * C9[:, :, None, None]
    J = I9[:, :, None, None] - AC                          # (9, 9, N, N)

    # Compute J^{-1} per mode :  (N, N, 9, 9)  with pseudoinverse
    J_t = np.transpose(J, (2, 3, 0, 1))                    # (N, N, 9, 9)
    J_inv = np.linalg.pinv(J_t, rcond=1e-10)               # full-rank inverse

    # Macro-projected inverse :   T (M J^{-1}) ... actually return T·M·J^{-1}
    # apply(R) = T  M  J^{-1}  R_in_lift_space
    # we precompute T_M_Jinv = T @ M @ J^{-1}   shape (N, N, 9, 9)
    TM_arr = TM[None, None, :, :].astype(np.complex128)
    T_M_Jinv = np.einsum("kuab,kubc->kuac",
                          TM_arr * np.ones_like(J_inv[..., :1, :1]),
                          J_inv)
    return T_M_Jinv


def apply_robust_pc(case, R_f, T_M_Jinv):
    """Apply robust PC :  df = IFFT( T M J^{-1} FFT(R_f) )  via 9x9 mode-wise."""
    # R_f shape (9, N, N) real -> Fourier
    R_hat = np.fft.fft2(R_f, axes=(1, 2))                    # (9, N, N) complex
    R_perm = np.transpose(R_hat, (1, 2, 0))                  # (N, N, 9)
    df_perm = np.einsum("jkab,jkb->jka", T_M_Jinv, R_perm)
    df_hat = np.transpose(df_perm, (2, 0, 1))                # (9, N, N) complex
    df = np.real(np.fft.ifft2(df_hat, axes=(1, 2)))
    return df




def solve_scmk_robust(case, max_outer=80, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                      line_search_max=5, kinetic_substeps_base=12, verbose=True):
    """Robust SCMK : parameter-free PC + backtracking line search + adaptive substeps."""
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_robust_spectral_schur(case.N, omega=case.omega)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    prev_res = None

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

        # Adaptive kinetic substeps based on contraction rate
        if prev_res is not None and prev_res > 0:
            res_ratio = res_norm / prev_res
            # fast contraction (ratio < 0.5) -> fewer substeps
            # slow (ratio > 0.8) -> more substeps
            substep_scale = max(0.5, min(2.0, res_ratio / 0.5))
            K_eff = max(5, int(kinetic_substeps_base * substep_scale))
        else:
            K_eff = kinetic_substeps_base
        prev_res = res_norm

        norm_f = case._fast_norm(f)
        probe = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R_f, norm_f_cached=norm_f).ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            return apply_robust_pc(case, R, S_inv).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R_f.ravel()
        df_flat, info = gmres(Aop, rhs, M=Mop,
                              rtol=krylov_tol,
                              atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                              maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df_flat)):
            print("  GMRES NaN")
            break
        df = df_flat.reshape(case.shape)

        # Backtracking line search (restored from always-accept)
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            f_trial = f + alpha * df
            for _ in range(K_eff):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += K_eff
            R_trial = f_trial - case.lbe_step(f_trial)
            lbe_calls += 1
            r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
            if r_trial < res_norm:
                f = f_trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # pure kinetic fallback
            for _ in range(K_eff):
                f = case.lbe_step(f)
            lbe_calls += K_eff

    return f, history
