"""HKR-LBM — Hydro-Kinetic Reduced Newton with slaved closure.

Solves Newton in 3N² macro space U, using f_neq slaved via K LBM substeps.

    Given U:
        f0 = T·U + f_neq_old
        f_slave = L^K(f0)              # K kinetic relaxation steps
        R_U(U) = M(f_slave − L(f_slave))

    Macro Jacobian via JFNK (3N² × 3N² space).

    After macro Newton: f ← T·(U+δU) + (f_slave − T·M·f_slave)
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def slaved_residual(U_flat, case, f_neq, k_slave):
    U = U_flat.reshape(3, case.N, case.N)
    f0 = case.lift(U) + f_neq
    f_s = f0
    for _ in range(k_slave):
        f_s = case.lbe_step(f_s)
    R_f = f_s - case.lbe_step(f_s)
    return case.project(R_f).ravel()


def solve_hkr(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
               kinetic_substeps=15, k_slave=3, verbose=True):
    f = case.initial_field()
    n_macro = case.macro_dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    # Warm-up
    for _ in range(5):
        f = case.lbe_step(f); lbe_calls += 1
    U = case.project(f)
    f_neq = f - case.lift(U)

    for k in range(max_outer):
        R_f = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R_f) / np.sqrt(case.dof)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  hkr {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        U_flat = U.ravel()
        R_U_base = slaved_residual(U_flat, case, f_neq, k_slave); lbe_calls += k_slave + 1
        norm_U = float(np.linalg.norm(U_flat))
        eps = 1e-6 * max(1.0, norm_U)
        probe = [0]

        def matvec(v_flat):
            probe[0] += 1
            R_pert = slaved_residual(U_flat + eps * v_flat, case, f_neq, k_slave)
            return (R_pert - R_U_base) / eps

        def precond(r_flat):
            # FFT-Schur on macro residual directly
            R_U = r_flat.reshape(3, case.N, case.N)
            R_U_hat = np.fft.fft2(R_U, axes=(1, 2))
            R_perm = np.transpose(R_U_hat, (1, 2, 0))
            dU_perm = np.einsum("jkab,jkb->jka", S_inv, R_perm)
            dU_hat = np.transpose(dU_perm, (2, 0, 1))
            dU = np.real(np.fft.ifft2(dU_hat, axes=(1, 2)))
            return dU.ravel()

        Aop = LinearOperator((n_macro, n_macro), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_macro, n_macro), matvec=precond, dtype=np.float64)
        dU_flat, _ = gmres(Aop, -R_U_base, M=Mop, rtol=krylov_tol,
                            atol=krylov_tol * np.linalg.norm(R_U_base) * 1e-3,
                            maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0] * (k_slave + 1)

        if not np.all(np.isfinite(dU_flat)):
            break

        U = U + dU_flat.reshape(3, case.N, case.N)
        f = case.lift(U) + f_neq
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps
        if not np.all(np.isfinite(f)):
            break
        U = case.project(f)
        f_neq = f - case.lift(U)

    return f, history
