"""RRE-LBM : Reduced Rank Extrapolation for steady-state LBM.

Sidi 1991 vector extrapolation method. Parameter-free quasi-Newton
acceleration for fixed-point iteration  f^{n+1} = L(f^n).

Algorithm (per cycle) :
    1. Generate K iterates  X = [f_0, f_1, ..., f_K]  via baseline LBE
    2. First differences   U[:, i] = f_{i+1} - f_i      (n_full x K)
    3. Second differences  W[:, i] = U[:, i+1] - U[:, i]  (n_full x (K-1))
    4. Solve least squares  W γ ≈ U[:, -1]      (K-1 unknowns)
    5. Extrapolated steady  f* ≈ f_0 - U[:, :-1] @ γ
    6. Polish  N_polish LBE steps
    7. If not converged, restart cycle from f*

No tunable physics parameters; only algorithmic K (cycle length) and N_polish.
Both can be auto-selected (K = sqrt(n_macro), N_polish = sqrt(K)) but explicit
values used here for reproducibility.
"""

import time
import numpy as np


def solve_rre_lbm(case, max_cycle=20, tol=1e-7, K=15, N_polish=5, verbose=True):
    """RRE-accelerated steady LBM.

    Parameters
    ----------
    K          : iterates per RRE cycle (algorithmic, not physical)
    N_polish   : LBE steps after each extrapolation
    """
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_cycle):
        R_f = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  rre {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at cycle {k}")
            break

        # 1. Generate K iterates
        f0 = f.copy()
        X = [f.ravel().copy()]
        for _ in range(K):
            f = case.lbe_step(f)
            X.append(f.ravel().copy())
        lbe_calls += K
        X = np.stack(X, axis=1)            # (n_full, K+1)

        # 2-3. Differences
        U = X[:, 1:] - X[:, :-1]            # (n_full, K)
        W = U[:, 1:] - U[:, :-1]            # (n_full, K-1)

        # 4. Least squares  W γ = U[:, -1]
        gamma, *_ = np.linalg.lstsq(W, U[:, -1], rcond=None)

        # 5. Extrapolated steady
        f_extrap = X[:, 0] - U[:, :-1] @ gamma
        f_extrap = f_extrap.reshape(case.shape)

        # check extrapolated residual; pick whichever is smaller
        R_ext = case.residual(f_extrap); lbe_calls += 1
        r_ext = case._fast_norm(R_ext) / np.sqrt(n_full)
        R_walk = case.residual(f); lbe_calls += 1
        r_walk = case._fast_norm(R_walk) / np.sqrt(n_full)

        f = f_extrap if r_ext < r_walk else f

        # 6. Polish
        for _ in range(N_polish):
            f = case.lbe_step(f)
        lbe_calls += N_polish

    return f, history
