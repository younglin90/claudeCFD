"""AP-MoMeNt-LBM: Asymptotic-Preserving Moment-Newton-Krylov for steady LBM.

Outer loop:
    S_U^AP  dU = -M R_f(f)
    f <- f + alpha * T dU      (line search)
    f <- L(f) repeated kinetic_steps times       (null-space damping)

Schur action choices :
    'galerkin' = M J_f T v                              (1 probe)
    'apmnt'    = M J_f T v - (1/omega) M J_f (I-TM) J_f T v  (2 probes)
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def solve_apmnt(
    case,
    max_outer=200,
    tol=1e-7,
    krylov_max=30,
    krylov_tol=1e-2,
    kinetic_steps=3,
    schur_mode="apmnt",          # 'apmnt' or 'galerkin'
    line_search_max=8,
    verbose=True,
):
    f = case.initial_field()
    N = case.N
    n_macro = case.macro_dof
    history = []  # (outer, res, lbe_calls, wall)
    t0 = time.perf_counter()
    lbe_calls = 0

    schur_fn = case.schur_apmnt if schur_mode == "apmnt" else case.schur_galerkin

    for k in range(max_outer):
        R_f = case.residual(f)
        lbe_calls += 1
        R_U = case.project(R_f)
        res_norm = np.sqrt((R_f * R_f).mean())
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))

        if verbose:
            print(f"  outer {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")

        if res_norm < tol:
            if verbose:
                print(f"  CONVERGED at outer {k}")
            break

        # Krylov solve for macro correction
        probes_per_matvec = 2 if schur_mode == "apmnt" else 1
        probe_count = [0]
        norm_f_cached = case._fast_norm(f)

        def matvec(v):
            dU = v.reshape(3, N, N)
            out = schur_fn(dU, f, R_f, norm_f_cached=norm_f_cached)
            probe_count[0] += probes_per_matvec
            return out.flatten()

        L_op = LinearOperator((n_macro, n_macro), matvec=matvec, dtype=np.float64)
        rhs = -R_U.flatten()
        rhs_norm = np.linalg.norm(rhs)

        dU_flat, info = gmres(
            L_op, rhs,
            rtol=krylov_tol,
            atol=krylov_tol * rhs_norm * 1e-3,
            maxiter=krylov_max,
            restart=min(krylov_max, 30),
        )
        lbe_calls += probe_count[0]

        if not np.all(np.isfinite(dU_flat)):
            if verbose:
                print(f"  GMRES returned NaN, abort")
            break

        dU = dU_flat.reshape(3, N, N)
        df_macro = case.lift(dU)

        # Backtracking line search on full residual norm
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            f_trial = f + alpha * df_macro
            R_trial = f_trial - case.lbe_step(f_trial)
            lbe_calls += 1
            r_trial = np.sqrt((R_trial * R_trial).mean())
            if r_trial < res_norm:
                f = f_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Take damped step anyway, recover via kinetic relaxation
            f = f + alpha * df_macro

        # Kinetic null-space relaxation
        for _ in range(kinetic_steps):
            f = case.lbe_step(f)
            lbe_calls += 1

    return f, history
