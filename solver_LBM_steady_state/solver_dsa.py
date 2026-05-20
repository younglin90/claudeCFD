"""DSA-LBM — Diffusion Synthetic Acceleration / Chapman-Enskog PC.

Uses fft_stokes_inverse AS preconditioner inside Newton-Krylov loop.
CE-derived macro Stokes operator replaces the AP-Schur kinetic block.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from macro_low_order import fft_stokes_inverse


def solve_dsa(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
               kinetic_substeps=15, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    nu = case.nu if hasattr(case, "nu") else 0.05
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  dsa {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
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
            R_U = case.project(r)
            try:
                dU = fft_stokes_inverse(R_U, nu)
            except Exception:
                dU = np.zeros_like(R_U)
            return case.lift(dU).ravel()
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
