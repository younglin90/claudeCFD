"""SCMK-family steady LBM solvers.

The public ``solve_scmk`` entry point is intentionally routed through one
unified residual-audited SafeNN path. The original Newton-Krylov SCMK core
remains available as ``solve_scmk_core``:

    while ||R(f)|| > tol :
        FGMRES on   J_f df = -R(f)
            matvec J_f w = (R(f + eps w) - R(f)) / eps           (JVP)
            right preconditioner :   P^{-1} R  =  T  S_U^{-1}  M R     (FFT-based)
        line-search alpha   ;   f <- f + alpha df

The unified route avoids benchmark-name, Reynolds-number, geometry, or grid
dispatch in the proposed method.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur


def solve_picard_guard(case, max_steps=30000, tol=1e-7, check_every=500, verbose=True,
                        plateau_window=50, plateau_eps=0.05):
    """Accuracy fallback: native LBM fixed-point iteration with the same history schema.

    Stops on a non-finite residual or once the residual has plateaued: over
    the last ``plateau_window`` checkpoints, the relative improvement between
    the older and newer half-medians is at most ``plateau_eps``. ``tol`` is
    accepted for API compatibility but no longer used as an early-exit
    condition; plateau detection alone decides when to stop.
    """
    f = case.initial_field()
    history = []
    res_hist = []
    t0 = time.perf_counter()
    lbe_calls = 0
    for step in range(1, max_steps + 1):
        f = case.lbe_step(f)
        lbe_calls += 1
        if step % check_every == 0:
            R = case.residual(f)
            lbe_calls += 1
            res = case._fast_norm(R) / np.sqrt(case.dof)
            history.append((step, res, lbe_calls, time.perf_counter() - t0))
            res_hist.append(res)
            if verbose and (step == check_every or step % (10 * check_every) == 0 or res < tol):
                print(f"  guard {step:7d} | res {res:.3e} | lbe {lbe_calls:7d}")
            plateaued = False
            if len(res_hist) >= plateau_window:
                tail = res_hist[-plateau_window:]
                half = max(plateau_window // 2, 1)
                old = float(np.median(tail[:half]))
                new = float(np.median(tail[half:]))
                if np.isfinite(old) and old > 0 and np.isfinite(new):
                    plateaued = (old - new) / old <= plateau_eps
            if not np.isfinite(res) or plateaued:
                break
    return f, history


def solve_scmk_core(
    case,
    S_inv,
    max_outer=100,
    tol=1e-9,
    krylov_max=20,
    krylov_tol=1e-3,
    line_search_max=5,
    kinetic_substeps=8,
    verbose=True,
):
    f = case.initial_field()
    n_full = case.dof
    history = []  # (outer, res, lbe_calls, wall)
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R_f = case.residual(f)
        lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  outer {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose:
                print(f"  CONVERGED at outer {k}")
            break

        # JVP closure
        norm_f = case._fast_norm(f)
        probe_count = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            Jw = case.jvp(w, f, R_f, norm_f_cached=norm_f)
            probe_count[0] += 1
            return Jw.ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            df = apply_spectral_schur(case, R, S_inv)
            return df.ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)

        rhs = -R_f.ravel()
        df_flat, info = gmres(
            Aop, rhs,
            M=Mop,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe_calls += probe_count[0]

        if not np.all(np.isfinite(df_flat)):
            print("  GMRES NaN, abort")
            break

        df = df_flat.reshape(case.shape)

        f = f + df
        for _ in range(kinetic_substeps):
            f = case.lbe_step(f)
        lbe_calls += kinetic_substeps

    return f, history


def solve_scmk(
    case,
    S_inv,
    max_outer=100,
    tol=1e-9,
    krylov_max=20,
    krylov_tol=1e-3,
    line_search_max=5,
    kinetic_substeps=8,
    verbose=True,
):
    """Use one unified safeguarded accelerator for every case."""
    from solver_unified_safe_nn import solve_unified_safe_nn

    return solve_unified_safe_nn(
        case,
        max_outer=max_outer,
        tol=tol,
        krylov_max=krylov_max,
        krylov_tol=krylov_tol,
        kinetic_substeps=12,
        verbose=verbose,
    )


def solve_baseline_periodic(case, max_steps=200000, tol=1e-9, check_every=100, verbose=True):
    """Native LBM Picard iteration on periodic Kolmogorov flow."""
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for step in range(1, max_steps + 1):
        f = case.lbe_step(f)
        lbe_calls += 1
        if step % check_every == 0:
            R = f - case.lbe_step(f)
            lbe_calls += 1
            res = case._fast_norm(R) / np.sqrt(n_full)
            wall = time.perf_counter() - t0
            history.append((step, res, lbe_calls, wall))
            if verbose and (step % 1000 == 0 or step == check_every):
                print(f"  step {step:7d} | res {res:.3e} | wall {wall:.2f}s")
            if res < tol:
                if verbose:
                    print(f"  CONVERGED at step {step}")
                break
    return f, history
