"""KS-LBM : Koopman-Spectral Steady Solver via Dynamic Mode Decomposition.

Idea : LBM step  f^{n+1} = L_LBM(f^n)  is a Koopman-like operator.
       For each slow mode (lambda, psi) of L_LBM near steady,
       geometric-series sum gives  f^infty = f + Σ ψ (λ/(1-λ)) <ψ^left, f>.

Algorithm (parameter-free except snapshot count) :

    1. Warmup K_warm LBE steps -> snapshot matrix X
    2. DMD on X :
         X' = X shifted, U Σ V* = SVD(X)
         rank r auto-selected from singular value drop (relative cutoff)
         A_tilde = U* X' V Σ^{-1}
         eigenvalues λ_i and DMD modes ψ_i = U w_i
    3. Project residual onto modes and extrapolate :
         for each contracting mode (|λ-1| > tol_marginal, |λ| < 1):
            amp_i = <ψ_i^left, current state>
            correction += ψ_i * (λ_i / (1 - λ_i)) * amp_i
    4. Polish K_polish LBE steps
    5. If not converged, refresh snapshots and goto 2.
"""

import time
import numpy as np


def solve_ks_lbm(case, max_iter=20, tol=1e-7, K_warm=20, K_polish=5,
                  sv_cutoff=1e-3, verbose=True):
    """Koopman-DMD steady solver.

    Parameters
    ----------
    K_warm : snapshot count per DMD refresh  (algorithmic, not physical)
    K_polish : LBE steps after each jump
    sv_cutoff : DMD rank auto-selection (singular value relative cutoff)
    """
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_iter):
        # current residual
        R_f = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  ks {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at iter {k}")
            break

        # 1. Warmup snapshots
        snapshots = [f.ravel().copy()]
        f_walk = f.copy()
        for _ in range(K_warm):
            f_walk = case.lbe_step(f_walk)
            snapshots.append(f_walk.ravel().copy())
        lbe_calls += K_warm

        # 2. DMD
        X = np.stack(snapshots[:-1], axis=1)     # (n_full, K_warm)
        X_p = np.stack(snapshots[1:], axis=1)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        # auto rank
        r = int(np.sum(S > sv_cutoff * S[0]))
        r = max(1, min(r, K_warm))
        U_r = U[:, :r]; S_r = S[:r]; V_r = Vh[:r, :].conj().T  # V (K_warm, r)
        A_tilde = U_r.conj().T @ X_p @ V_r / S_r[None, :]
        lam, w = np.linalg.eig(A_tilde)
        psi = U_r @ w                              # DMD modes (n_full, r)

        # 3. Extrapolate slow modes (|λ - 1| > tol_marginal AND |λ| < 1)
        # amplitudes via least-squares :  ψ amps = f_walk - f
        # use end state f_walk for residual projection
        f_current = f_walk.ravel()
        # solve ψ a = f_current  for amplitudes a
        # use complex solve
        try:
            amps, *_ = np.linalg.lstsq(psi, f_current.astype(np.complex128), rcond=None)
        except Exception as e:
            if verbose: print(f"  lstsq failed : {e}")
            f = f_walk
            continue

        # geometric extrapolation
        tol_marginal = 1e-3  # avoid modes near λ=1 (steady itself)
        contracting = (np.abs(lam - 1) > tol_marginal) & (np.abs(lam) < 1.0 - tol_marginal)
        geom_factor = np.zeros_like(lam, dtype=np.complex128)
        geom_factor[contracting] = lam[contracting] / (1.0 - lam[contracting])
        correction = np.real(psi @ (geom_factor * amps))
        f_jumped = f_walk + correction.reshape(case.shape)

        # safeguard : check residual
        R_jumped = case.residual(f_jumped); lbe_calls += 1
        r_jumped = case._fast_norm(R_jumped) / np.sqrt(n_full)
        # accept jump unless catastrophic (10x worse)
        if r_jumped < res_norm * 10:
            f = f_jumped
        else:
            f = f_walk

        # 4. Polish
        for _ in range(K_polish):
            f = case.lbe_step(f)
        lbe_calls += K_polish

    return f, history
