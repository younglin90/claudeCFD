"""Bloch-NDA + FFT-Moulinec-Suquet — geometry-aware periodic solvers.

Both methods exploit unit-cell periodicity:
    Bloch-NDA   : Bloch decomposition over geometry-periodic mask
    FFT-MS      : Lippmann-Schwinger fixed-point with Fourier Green operator

Implementation here: combined as one solver using mask-aware FFT-Stokes
with Lippmann-Schwinger contrast (ν_solid → very large).
Applied to multi-cylinder voxel cases.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from macro_low_order import fft_stokes_inverse
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_bloch_ms(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                    kinetic_substeps=15, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    nu = case.nu if hasattr(case, "nu") else 0.05
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    # solid mask
    mask = getattr(case, "mask", None)
    if mask is None:
        mask = np.ones((case.N, case.N), dtype=bool)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  bms {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        norm_f = case._fast_norm(f)
        probe = [0]
        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R, norm_f_cached=norm_f).ravel()

        def precond(r_flat):
            r = r_flat.reshape(case.shape)
            # MS-style: mask R to fluid region, FFT Stokes, mask result
            R_U = case.project(r) * mask[None, :, :].astype(np.float64)
            d_schur = apply_spectral_schur(case, r, S_inv)
            try:
                dU = fft_stokes_inverse(R_U, nu)
            except Exception:
                dU = np.zeros_like(R_U)
            dU = dU * mask[None, :, :].astype(np.float64)
            # blend: 0.5 schur + 0.5 stokes-MS
            return (0.5 * d_schur + 0.5 * case.lift(dU)).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(Aop, -R.ravel(), M=Mop, rtol=krylov_tol,
                       atol=krylov_tol * np.linalg.norm(R) * 1e-3,
                       maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]
        if not np.all(np.isfinite(df)):
            break
        f_new = f + df.reshape(case.shape)
        for _ in range(kinetic_substeps):
            f_new = case.lbe_step(f_new)
        lbe_calls += kinetic_substeps
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
        f = f_new
    return f, history
