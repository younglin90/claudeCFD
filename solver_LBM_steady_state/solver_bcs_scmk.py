"""BCS-SCMK — Boundary-Corrected Spectral Schur Newton-Krylov.

Treats real Jacobian as J = J0 + E where J0 = periodic Schur (analytic),
E = wall/voxel-induced perturbation (low rank).

Collects secant pairs (Δf, Δy) from Newton steps:
    Δy ≈ J · Δf  via JVP
    z   = Δy - J0 · Δf      (boundary contribution)

Maintains last r=5 pairs. Applies Woodbury-corrected PC:
    PC = PC0 - PC0 Z (I + Δf^T PC0 Z)^{-1} Δf^T PC0
where PC0 = T S_U^{-1} M (FFT-Schur).

Reference: Eisenstat-Walker idea extended to LBM-specific boundary E.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def apply_J0(case, v, S_full):
    """J0 v = v - L'_periodic v. Approximate via FFT macro Jacobian.

    For computational simplicity use: J0 v ≈ v - apply_C_kinetic(v)
    where apply_C_kinetic uses S_full to model the macro action.

    Concretely we use J0 v = v - lift(invFFT(S_macro · FFT(macro(v))))
    which captures the periodic streaming-collision linearization.
    """
    # Use periodic linear Jacobian on macro projection
    v_U = case.project(v)
    v_U_hat = np.fft.fft2(v_U, axes=(1, 2))
    v_perm = np.transpose(v_U_hat, (1, 2, 0))    # (N, N, 3)
    Jv_perm = np.einsum("jkab,jkb->jka", S_full, v_perm)
    Jv_hat = np.transpose(Jv_perm, (2, 0, 1))
    Jv = np.real(np.fft.ifft2(Jv_hat, axes=(1, 2)))
    return case.lift(Jv)


def build_S_full(N, omega):
    """Build forward Schur S_U (not inverse) for J0*v."""
    from lbm_periodic import CX, CY, W
    M_mat = np.zeros((3, 9))
    M_mat[0, :] = 1.0
    M_mat[1, :] = CX
    M_mat[2, :] = CY
    T_mat = np.zeros((9, 3))
    for i in range(9):
        T_mat[i, 0] = W[i]
        T_mat[i, 1] = 3.0 * W[i] * CX[i]
        T_mat[i, 2] = 3.0 * W[i] * CY[i]
    kx = 2.0 * np.pi * np.fft.fftfreq(N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    phase = np.empty((9, N, N), dtype=np.complex128)
    for i in range(9):
        phase[i] = np.exp(-1j * (KX * CX[i] + KY * CY[i]))
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
    return np.transpose(S, (2, 3, 0, 1))   # (N, N, 3, 3)


def solve_bcs_scmk(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                    kinetic_substeps=15, rank=5, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    S_full = build_S_full(case.N, case.omega)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    sec_S = []   # store Δf vectors
    sec_Z = []   # store (J - J0) Δf
    n_corrupt = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  bcs {k:3d} | res {res:.3e} | rank {len(sec_S)} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        norm_f = case._fast_norm(f)
        probe = [0]
        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R, norm_f_cached=norm_f).ravel()

        # Build Woodbury-corrected PC
        if len(sec_S) > 0:
            S_mat = np.stack([s.ravel() for s in sec_S], axis=1)   # (n, r)
            Z_mat = np.stack([z.ravel() for z in sec_Z], axis=1)   # (n, r)
            PC0_Z_list = [apply_spectral_schur(case, z.reshape(case.shape), S_inv).ravel() for z in sec_Z]
            PC0_Z = np.stack(PC0_Z_list, axis=1)                   # (n, r)
            # Inner matrix: I + S^T PC0 Z
            inner = np.eye(len(sec_S)) + S_mat.T @ PC0_Z
            try:
                inner_inv = np.linalg.inv(inner)
            except np.linalg.LinAlgError:
                inner_inv = None
        else:
            PC0_Z = None
            inner_inv = None
            S_mat = None

        def precond(r_flat):
            r = r_flat.reshape(case.shape)
            d0 = apply_spectral_schur(case, r, S_inv).ravel()
            if inner_inv is not None:
                # correction: - PC0 Z (I + S^T PC0 Z)^-1 S^T d0
                gamma = inner_inv @ (S_mat.T @ d0)
                d0 = d0 - PC0_Z @ gamma
            return d0

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(Aop, -R.ravel(), M=Mop, rtol=krylov_tol,
                       atol=krylov_tol * np.linalg.norm(R) * 1e-3,
                       maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]
        if not np.all(np.isfinite(df)):
            n_corrupt += 1
            if n_corrupt > 3: break
            sec_S, sec_Z = [], []
            continue

        df_arr = df.reshape(case.shape)

        # Build secant pair: y = J·df via JVP, z = y - J0·df
        y = case.jvp(df_arr, f, R, norm_f_cached=norm_f); lbe_calls += 1
        z = y - apply_J0(case, df_arr, S_full)
        sec_S.append(df_arr.copy())
        sec_Z.append(z)
        if len(sec_S) > rank:
            sec_S.pop(0); sec_Z.pop(0)

        f_new = f + df_arr
        for _ in range(kinetic_substeps):
            f_new = case.lbe_step(f_new)
        lbe_calls += kinetic_substeps
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            sec_S, sec_Z = [], []
        f = f_new

    return f, history
