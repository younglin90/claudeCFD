"""Simplified solver : Spectral-preconditioned Picard (no FGMRES inner).

Algorithm (much simpler than SCMK Phase-4):

    for outer k:
        R = f - L(f)                          # 1 LBM call
        delta = T · S^{-1} · M · R            # FFT (one PC apply, no Krylov)
        f = f - alpha * delta                  # under-relaxed update
        f = L^K(f)                             # K_kinetic LBM steps
        adaptive : if residual stalls → alpha decrease; else increase

No FGMRES, no JVP, no line search. Total per-outer cost ~ K_kinetic + 2 LBM calls.
"""

import time
import numpy as np
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_simple(case, max_outer=300, tol=1e-7,
                  alpha_init=1.0, kinetic_substeps=15,
                  N_check=6, min_ratio=2.0, verbose=True):
    """Spectral-preconditioned Picard iteration.

    No Krylov inner loop. Adaptive alpha based on residual ratio.
    """
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    alpha = alpha_init
    prev_res = None
    phase_b = False
    res_history = []

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        res_history.append(res_norm)
        if verbose:
            print(f"  simple {k:3d} | res {res_norm:.3e} | alpha {alpha:.3f} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Stagnation detection
        if not phase_b and k == N_check and res_history[0] > 0:
            ratio = res_history[0] / res_norm
            if ratio < min_ratio:
                if verbose: print(f"  STAGNATED (ratio={ratio:.2f}), switch to baseline")
                phase_b = True

        if phase_b:
            for _ in range(50):
                f = case.lbe_step(f)
            lbe_calls += 50
            continue

        # Spectral PC apply (1 FFT pass, no Krylov)
        delta = apply_spectral_schur(case, R, S_inv)

        # Adaptive alpha :  prev contraction good -> increase ; bad -> decrease
        if prev_res is not None and prev_res > 0:
            contraction = res_norm / prev_res
            if contraction < 0.5:
                alpha = min(1.2, alpha * 1.1)
            elif contraction > 0.95:
                alpha = max(0.3, alpha * 0.7)
        prev_res = res_norm

        # Update
        f = f - alpha * delta

        # K_kinetic LBM post-processing
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps

    return f, history
