"""KDF-LBM — Koopman/DMD Deflated Fixed-Point.

Collect m=8 LBM residual snapshots, build DMD operator,
identify dominant slow eigenmodes |λ|≈1, deflate them:

    R sequence: r_0, r_1, ..., r_m
    DMD : R[:, 1:] ≈ A R[:, :-1]
    eigendecomp on rank-r truncation of POD
    slow modes φ_j with λ_j close to 1
    Extrapolate to fixed point:
        f_∞ ≈ f_m + Φ_slow (I - Λ_slow)^{-1} α
    where α = Φ_slow^† r_m

Apply periodically; in between, do plain LBM substeps.
"""
import time
import numpy as np


def dmd_extrap(snapshots, eps_slow=0.05, rank=6):
    """Given snapshots X shape (n, m), return extrapolation Δf to steady-state."""
    X = np.stack([s.ravel() for s in snapshots], axis=1)   # (n, m)
    X0 = X[:, :-1]
    X1 = X[:, 1:]
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vt[:r, :].T
    A_tilde = U_r.T @ X1 @ V_r * (1.0 / S_r)
    lam, W = np.linalg.eig(A_tilde)
    Phi = X1 @ V_r * (1.0 / S_r) @ W              # (n, r)

    # slow modes: 1 - eps_slow < |λ| < 1 + eps_slow
    slow = np.where((np.abs(lam - 1.0) < eps_slow) & (np.abs(lam) < 1.0))[0]
    if len(slow) == 0:
        return None
    Phi_s = Phi[:, slow]
    lam_s = lam[slow]
    r_last = X[:, -1].astype(np.complex128)
    alpha, *_ = np.linalg.lstsq(Phi_s, r_last, rcond=None)
    one_minus_L = 1.0 - lam_s
    if np.any(np.abs(one_minus_L) < 1e-6):
        return None
    delta = Phi_s @ (alpha / one_minus_L)
    return np.real(delta)


def solve_kdf(case, max_outer=400, tol=1e-7, lbm_per_cycle=12, dmd_rank=6,
               kinetic_substeps=15, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        # Collect snapshots
        snaps = [f.copy()]
        for _ in range(lbm_per_cycle):
            f = case.lbe_step(f); lbe_calls += 1
            snaps.append(f.copy())

        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  kdf {k:3d} | res {res:.3e} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # DMD extrapolation
        delta = dmd_extrap(snaps, rank=dmd_rank)
        if delta is not None and np.all(np.isfinite(delta)):
            f_trial = f + delta.reshape(case.shape)
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
            if np.all(np.isfinite(f_trial)):
                R_trial = case.residual(f_trial); lbe_calls += 1
                res_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
                if res_trial < res:           # accept only if better
                    f = f_trial

    return f, history
