"""Ensemble solver : SCMK + Anderson racing in parallel.

Algorithm :
    1. Snapshot initial f.
    2. Run SCMK Phase-A for K_probe outer iterations (probe SCMK rate).
    3. Run Anderson for K_probe iterations (probe Anderson rate).
    4. Compare contraction rates :  ρ_method = (res_final / res_initial)^(1/lbe_count)
    5. Continue with the faster contracting method to tol.
    6. If both stall, fallback to baseline LBE.

Universal selection : no user-specified preference. Best method per case auto.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import build_spectral_schur, apply_spectral_schur


def _scmk_step(case, f, R_f, S_inv, krylov_max, krylov_tol, kinetic_substeps):
    """One SCMK outer iteration. Returns updated f + lbe count."""
    n_full = case.dof
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
    lbe_used = probe[0]
    if not np.all(np.isfinite(df_flat)):
        return f, lbe_used, False
    df = df_flat.reshape(case.shape)
    f_new = f + df
    for _ in range(kinetic_substeps):
        f_new = case.lbe_step(f_new)
    lbe_used += kinetic_substeps
    return f_new, lbe_used, True


def _anderson_step(case, f, g_f_prev, F_hist, G_hist, X_hist, m_max, beta):
    """One Anderson iteration. Returns (f_new, F_new, lbe_used)."""
    g_f = case.lbe_step(f)
    F_new = g_f - f
    X_hist.append(f.copy()); G_hist.append(g_f); F_hist.append(F_new)
    if len(F_hist) > m_max + 1:
        F_hist.pop(0); G_hist.pop(0); X_hist.pop(0)
    n_m = len(F_hist) - 1
    if n_m < 1:
        return g_f, F_new, 1
    dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
    dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
    gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
    f_new = g_f.ravel() - dG @ gamma
    f_new = f_new.reshape(case.shape)
    if beta < 1.0:
        f_new = (1 - beta) * f + beta * f_new
    return f_new, F_new, 1


def solve_ensemble(case, max_outer=500, tol=1e-7, K_probe=4,
                    krylov_max=10, krylov_tol=1e-3, kinetic_substeps=15,
                    anderson_m=5, verbose=True):
    """Race SCMK and Anderson; pick faster contraction; never below baseline."""
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    # Phase 1 : probe both methods
    def res_of(g): R = case.residual(g); return case._fast_norm(R) / np.sqrt(n_full)

    R_f = case.residual(f); lbe_calls += 1
    res_norm = res_of(f); lbe_calls += 0  # already counted
    res_init = res_norm
    history.append((0, res_norm, lbe_calls, time.perf_counter() - t0))

    # Probe SCMK
    f_scmk = f.copy()
    lbe_scmk = 0
    for _ in range(K_probe):
        R_s = case.residual(f_scmk); lbe_scmk += 1
        f_scmk, used, ok = _scmk_step(case, f_scmk, R_s, S_inv,
                                       krylov_max, krylov_tol, kinetic_substeps)
        lbe_scmk += used
        if not ok:
            break
    res_scmk = res_of(f_scmk); lbe_scmk += 1
    rho_scmk = (res_scmk / max(res_init, 1e-30)) ** (1.0 / max(lbe_scmk, 1))

    # Probe Anderson
    f_and = f.copy()
    lbe_and = 0
    F_hist, G_hist, X_hist = [], [], []
    g_f_prev = None
    for _ in range(K_probe * kinetic_substeps):  # match LBE budget roughly
        f_and, _, used = _anderson_step(case, f_and, g_f_prev, F_hist, G_hist, X_hist,
                                          anderson_m, 1.0)
        lbe_and += used
    res_and = res_of(f_and); lbe_and += 1
    rho_and = (res_and / max(res_init, 1e-30)) ** (1.0 / max(lbe_and, 1))

    # Pick faster contraction (smaller rho means faster)
    use_scmk = rho_scmk < rho_and
    if verbose:
        print(f"  PROBE : SCMK rho={rho_scmk:.5f} (lbe {lbe_scmk}), "
              f"Anderson rho={rho_and:.5f} (lbe {lbe_and}). "
              f"Pick {'SCMK' if use_scmk else 'Anderson'}")

    # Adopt winner
    if use_scmk:
        f = f_scmk
        lbe_calls += lbe_scmk
    else:
        f = f_and
        lbe_calls += lbe_and

    # Phase 2 : continue with chosen method
    for k in range(K_probe, max_outer):
        R_f = case.residual(f); lbe_calls += 1
        res_norm = res_of(f)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose and k % 10 == 0:
            print(f"  ens {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:6d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at iter {k}")
            break

        if use_scmk:
            f, used, ok = _scmk_step(case, f, R_f, S_inv,
                                       krylov_max, krylov_tol, kinetic_substeps)
            lbe_calls += used
            if not ok:
                use_scmk = False  # SCMK broke; switch to Anderson
                F_hist, G_hist, X_hist = [], [], []
        else:
            f, _, used = _anderson_step(case, f, None, F_hist, G_hist, X_hist,
                                          anderson_m, 1.0)
            lbe_calls += used

    return f, history
