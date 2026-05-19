"""Direct Macro Newton (DMN) — novel SCMK variant.

Idea: skip distribution-level Newton entirely.
Define macro residual:
    G(U) = U − M · L_LBM(T · U)        where U = (ρ, ρu_x, ρu_y) ∈ ℝ^{3·N²}
                                        T · U = equilibrium lift
Solve G(U) = 0 via Newton-Krylov in 3·N² space.
Reconstruct f* = T · U* then K iter L_LBM for neq recovery.

Novelty vs SCMK Phase-4:
  - Newton variable count : 9N² → 3N² (3x smaller, 4x in 3D)
  - Krylov memory          : 3x smaller
  - No JVP on distribution : JVP only on macro field
  - Same FFT-Schur PC      : free, already 3x3 in macro
  - Conceptually cleaner   : steady state is macro problem (Chapman-Enskog)

Trade-off:
  - Each macro JVP requires (T → LBM → M) = 1 LBM call (same as distribution JVP)
  - Final reconstruction needs K LBM steps for neq
  - Untested in LBM literature
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def macro_residual(U_flat, case, f_neq=None):
    """G(U) = U − M · L(T · U + f_neq).  f_neq holds slow-manifold neq part.
    """
    U = U_flat.reshape(3, case.N, case.N)
    f = case.lift(U)
    if f_neq is not None:
        f = f + f_neq
    f_next = case.lbe_step(f)
    U_next = case.project(f_next)
    return (U - U_next).ravel()


def macro_jvp(v_flat, U_flat, case, R_base, f_neq=None, eps=None):
    """G'(U) · v ≈ [G(U + ε v) − G(U)] / ε."""
    v = v_flat.reshape(3, case.N, case.N)
    if eps is None:
        norm_U = np.linalg.norm(U_flat)
        norm_v = np.linalg.norm(v_flat)
        if norm_v < 1e-30:
            return np.zeros_like(U_flat)
        eps = 1e-7 * (norm_U + 1.0) / norm_v
    R_pert = macro_residual(U_flat + eps * v_flat, case, f_neq=f_neq)
    return (R_pert - R_base) / eps


def solve_macro(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                kinetic_substeps=15, verbose=True):
    """Direct Macro Newton-Krylov solver."""
    # Initial f → initial U
    f = case.initial_field()
    U = case.project(f)                           # (3, N, N)
    n_macro = 3 * case.N * case.N

    # Spectral macro Schur (same as SCMK but applied directly on U-space)
    from lbm_periodic import build_spectral_schur
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")

    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    # Initial f_neq from current f (after a few LBE warm steps)
    for _ in range(5):
        f = case.lbe_step(f); lbe_calls += 1
    U = case.project(f)
    f_neq = f - case.lift(U)

    for k in range(max_outer):
        U_flat = U.ravel()
        R = macro_residual(U_flat, case, f_neq=f_neq); lbe_calls += 1
        res_norm = np.linalg.norm(R) / np.sqrt(n_macro)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  macro {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        norm_U = np.linalg.norm(U_flat)
        probe = [0]

        def matvec(v_flat):
            probe[0] += 1
            return macro_jvp(v_flat, U_flat, case, R, f_neq=f_neq)

        def precond(r_flat):
            # Apply Ŝ_U^{-1} per Fourier mode directly on macro U
            r3 = r_flat.reshape(3, case.N, case.N)
            r_hat = np.fft.fft2(r3, axes=(1, 2))
            r_perm = np.transpose(r_hat, (1, 2, 0))
            dU_perm = np.einsum("jkab,jkb->jka", S_inv, r_perm)
            dU_hat = np.transpose(dU_perm, (2, 0, 1))
            dU = np.real(np.fft.ifft2(dU_hat, axes=(1, 2)))
            return dU.ravel()

        Aop = LinearOperator((n_macro, n_macro), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_macro, n_macro), matvec=precond, dtype=np.float64)
        rhs = -R

        dU_flat, _ = gmres(Aop, rhs, M=Mop, rtol=krylov_tol,
                            atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                            maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(dU_flat)):
            if verbose: print("  GMRES NaN, abort")
            break

        # Update macro
        U = U + dU_flat.reshape(3, case.N, case.N)

        # Build full f using updated macro + saved neq, then K LBM substeps
        f = case.lift(U) + f_neq
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps

        # Re-extract macro AND refresh f_neq
        U = case.project(f)
        f_neq = f - case.lift(U)

    # Final reconstruction
    f = case.lift(U)
    for _ in range(kinetic_substeps):
        f = case.lbe_step(f)
    lbe_calls += kinetic_substeps

    return f, history
