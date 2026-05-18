"""Adaptive hybrid solver : SCMK Phase-4 with automatic baseline fallback.

Idea : SCMK Phase-4 accelerates when spectral PC is valid (periodic, simple walls).
       When PC bias dominates (multi-obstacle, complex BC), residual reduction
       stalls. Detect stagnation and switch to baseline LBE for safety.

Robustness : never slower than baseline (worst case = baseline + small overhead).
Universality : SCMK acceleration where applicable, baseline elsewhere.

Detection : after N_check outer iter, if residual reduction ratio < min_ratio,
            switch to baseline mode for remainder.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import build_spectral_schur, apply_spectral_schur


def solve_hybrid(case, max_outer=100, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                  kinetic_substeps=15, N_check=3, min_ratio=10.0,
                  max_baseline_steps=50000, verbose=True):
    """SCMK Phase-4 with adaptive baseline fallback on stagnation.

    Phase A : SCMK Phase-4 outer Newton-Krylov.
    Detection : after N_check accepted outers, measure cumulative residual
                ratio. If < min_ratio, switch to baseline LBE.
    Phase B : pure baseline LBE until convergence.
    """
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    phase_b = False

    # Phase A : SCMK
    res_history = []
    for k in range(max_outer):
        R_f = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        res_history.append(res_norm)
        if verbose:
            mode_tag = "B" if phase_b else "A"
            print(f"  hyb[{mode_tag}] {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at iter {k}")
            break

        # Stagnation detection at iter N_check
        if not phase_b and k == N_check and res_history[0] > 0:
            ratio = res_history[0] / res_norm
            if ratio < min_ratio:
                if verbose:
                    print(f"  STAGNATED (ratio={ratio:.2f} < {min_ratio}), switch to baseline")
                phase_b = True

        if phase_b:
            # Phase B : pure baseline LBE
            for _ in range(50):  # batch 50 steps per "outer" for log granularity
                f = case.lbe_step(f)
            lbe_calls += 50
            continue

        # Phase A : SCMK step
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
                              maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df_flat)):
            if verbose: print("  GMRES NaN, switch to baseline")
            phase_b = True
            continue

        df = df_flat.reshape(case.shape)
        f = f + df
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps

    return f, history
