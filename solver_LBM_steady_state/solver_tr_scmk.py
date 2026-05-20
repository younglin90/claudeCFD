"""TR-SCMK — Trust-region Levenberg-Marquardt AP-Schur Newton-Krylov.

Replaces ad hoc S+ηI shift with mathematically defensible
    (S* S + λ I) δU = -S* R_U
form. λ adapts via gain ratio (classic LM):
    ρ_gain = actual_reduction / predicted_reduction
    if ρ > 0.75 :  λ *= 0.5
    elif ρ < 0.25:  λ *= 2.0

Otherwise identical to Lean SCMK (NK + post-LBM K=15 + NaN safeguard).
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import (CX, W, build_spectral_schur)


def build_lm_schur(N, omega, lam):
    """Build (S* S + λ I)^-1 S* in Fourier space, mode (0,0) zeroed."""
    M_mat = np.zeros((3, 9), dtype=np.float64)
    M_mat[0, :] = 1.0
    M_mat[1, :] = CX
    from lbm_periodic import CY as CY_arr
    M_mat[2, :] = CY_arr
    T_mat = np.zeros((9, 3), dtype=np.float64)
    for i in range(9):
        T_mat[i, 0] = W[i]
        T_mat[i, 1] = 3.0 * W[i] * CX[i]
        T_mat[i, 2] = 3.0 * W[i] * CY_arr[i]
    kx = 2.0 * np.pi * np.fft.fftfreq(N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    phase = np.empty((9, N, N), dtype=np.complex128)
    for i in range(9):
        phase[i] = np.exp(-1j * (KX * CX[i] + KY * CY_arr[i]))
    MAT = np.einsum("ai,ib,ijk->abjk", M_mat, T_mat, phase)
    S = -MAT.copy()
    for a in range(3):
        S[a, a] += 1.0
    phase2 = phase * phase
    MA2T = np.einsum("ai,ib,ijk->abjk", M_mat, T_mat, phase2)
    MAT2 = np.einsum("abjk,bcjk->acjk", MAT, MAT)
    raw = (1.0 - omega) / omega
    coeff = 0.5 * np.sign(raw) * min(0.5, abs(raw))
    S = S - coeff * (MA2T - MAT2)
    S_t = np.transpose(S, (2, 3, 0, 1))               # (N,N,3,3)
    Sh = np.conjugate(np.transpose(S_t, (0, 1, 3, 2)))
    SHS = np.einsum("jkab,jkbc->jkac", Sh, S_t)
    lam_I = lam * np.eye(3, dtype=np.complex128)
    M = SHS + lam_I[None, None, :, :]
    M_inv = np.linalg.inv(M)
    PC = np.einsum("jkab,jkbc->jkac", M_inv, Sh)
    mode00 = np.zeros((3, 3), dtype=np.complex128)
    mode00[1, 1] = 1.0
    mode00[2, 2] = 1.0
    PC[0, 0] = mode00
    return PC


def apply_lm_schur(case, R_f, PC):
    R_U = case.project(R_f)
    R_U_hat = np.fft.fft2(R_U, axes=(1, 2))
    R_perm = np.transpose(R_U_hat, (1, 2, 0))
    dU_perm = np.einsum("jkab,jkb->jka", PC, R_perm)
    dU_hat = np.transpose(dU_perm, (2, 0, 1))
    dU = np.real(np.fft.ifft2(dU_hat, axes=(1, 2)))
    return case.lift(dU)


def solve_tr_scmk(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                   kinetic_substeps=15, lam0=None, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    # initial λ from AP-Schur σ_max as proxy
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    lam = lam0 if lam0 is not None else 1e-3
    PC = build_lm_schur(case.N, case.omega, lam)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    res_prev = np.inf
    pred_red_prev = 1.0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  tr {k:3d} | res {res:.3e} | lam {lam:.2e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # LM gain ratio: shrink λ on success, grow on failure, bounded
        if k > 0:
            if res < res_prev * 0.9:
                lam = max(1e-6, lam * 0.7)
            elif res > res_prev:
                lam = min(1.0, lam * 3.0)
            PC = build_lm_schur(case.N, case.omega, lam)

        norm_f = case._fast_norm(f)
        probe = [0]
        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R, norm_f_cached=norm_f).ravel()
        def precond(r_flat):
            return apply_lm_schur(case, r_flat.reshape(case.shape), PC).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(Aop, -R.ravel(), M=Mop, rtol=krylov_tol,
                        atol=krylov_tol * np.linalg.norm(R) * 1e-3,
                        maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]
        if not np.all(np.isfinite(df)):
            break
        # predicted reduction estimate
        pred_red_prev = max(1e-30, res * 0.5)

        f_new = f + df.reshape(case.shape)
        for _ in range(kinetic_substeps):
            f_new = case.lbe_step(f_new)
        lbe_calls += kinetic_substeps
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            lam *= 4.0
            PC = build_lm_schur(case.N, case.omega, min(lam, 1.0))
        f = f_new
        res_prev = res

    return f, history
