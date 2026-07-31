"""Safe-NN — Residual-monotone Nesterov + Newton-Krylov.

Difference from NN:
    y_k = f_k + β(f_k - f_{k-1})
    R(y_k) evaluated
    if ||R(y_k)|| > (1+ε) ||R(f_k)|| :
        β ← 0.5 β, y_k := f_k         (reject lookahead)
    Newton-Krylov step on accepted y_k

Prevents Cavity Re=400 NaN.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_safe_nn(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                   kinetic_substeps=15, beta_max=0.7, eps_accept=0.10,
                   line_search=False, line_search_max=4,
                   residual_growth_reject=0.15,
                   final_polish_tol=None, final_polish_max_steps=0,
                   final_polish_check_every=500, verbose=True):
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    beta = 0.1
    res_prev = np.inf
    reject_count = 0
    streak_no_reject = 0
    beta_cap = beta_max
    best_res = np.inf
    best_f = np.array(f, copy=True)
    best_lbe = 0
    best_wall = time.perf_counter() - t0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if res < best_res:
            best_res = float(res)
            best_f = np.array(f, copy=True)
            best_lbe = lbe_calls
            best_wall = wall
        if verbose:
            print(f"  snn {k:3d} | res {res:.3e} | beta {beta:.3f} | rej {reject_count} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Tentative β with dynamic cap (raise when stable smooth regime)
        if res > res_prev:
            beta = beta * 0.7
            streak_no_reject = 0
            beta_cap = beta_max
        else:
            beta = min(beta_cap, beta + 0.15)
            if streak_no_reject >= 2:
                beta_cap = min(0.95, beta_max + 0.2)

        # Nesterov lookahead with residual-safe acceptance
        if beta > 0.3:
            y = f + beta * (f - f_prev)
            R_y = y - case.lbe_step(y); lbe_calls += 1
            norm_R_y = case._fast_norm(R_y)
            norm_R = case._fast_norm(R)
            eps_eff = eps_accept + 0.2 * beta
            if norm_R_y > (1.0 + eps_eff) * norm_R or not np.all(np.isfinite(R_y)):
                y = f.copy()
                R_y = R
                beta = beta * 0.7
                reject_count += 1
                streak_no_reject = 0
                beta_cap = beta_max
            else:
                streak_no_reject += 1
        else:
            y = f
            R_y = R
            streak_no_reject += 1

        norm_y = case._fast_norm(y)
        probe = [0]
        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, y, R_y, norm_f_cached=norm_y).ravel()
        def precond(r_flat):
            return apply_spectral_schur(case, r_flat.reshape(case.shape),
                                          S_inv).ravel()
        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(Aop, -R_y.ravel(), M=Mop, rtol=krylov_tol,
                       atol=krylov_tol * np.linalg.norm(R_y) * 1e-3,
                       maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]
        if not np.all(np.isfinite(df)):
            if verbose: print("  GMRES NaN, abort")
            break

        # Adaptive K: fewer substeps when near convergence AND res decreasing
        if res < 3e-5 and res < res_prev:
            K_eff = max(5, kinetic_substeps // 2)
        else:
            K_eff = kinetic_substeps

        accepted = False
        candidate_res = None
        alpha = 1.0
        f_new = None
        fallback_alpha = 1.0 if line_search else 0.5
        fallback_steps = max(1, K_eff if line_search else max(1, K_eff // 2))
        trials = line_search_max if line_search else 4
        for _ in range(max(1, trials)):
            f_trial = y + alpha * df.reshape(case.shape)
            for _ in range(K_eff):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += K_eff

            if np.all(np.isfinite(f_trial)):
                R_trial = f_trial - case.lbe_step(f_trial)
                lbe_calls += 1
                r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
                monotone_cap = (1.0 + residual_growth_reject) * best_res
                if np.isfinite(r_trial) and r_trial <= max(monotone_cap, tol):
                    f_new = f_trial
                    accepted = True
                    candidate_res = float(r_trial)
                    break

            # Always dampen step when check failed.
            alpha *= 0.5
            if alpha < 1.0e-4:
                # short extra probe for very conservative recovery
                f_trial = y + fallback_alpha * df.reshape(case.shape)
                for _ in range(fallback_steps):
                    f_trial = case.lbe_step(f_trial)
                lbe_calls += fallback_steps
                if np.all(np.isfinite(f_trial)):
                    R_trial = f_trial - case.lbe_step(f_trial)
                    lbe_calls += 1
                    r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
                    monotone_cap = (1.0 + residual_growth_reject) * best_res
                    if np.isfinite(r_trial) and r_trial <= max(monotone_cap, tol):
                        f_new = f_trial
                        accepted = True
                        candidate_res = float(r_trial)
                        break
                fallback_alpha *= 0.5
                fallback_steps = max(1, fallback_steps // 2)
                alpha = fallback_alpha

        if not accepted or f_new is None or not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            beta = 0.0

        if accepted and candidate_res is None:
            R_new = case.residual(f_new)
            lbe_calls += 1
            candidate_res = case._fast_norm(R_new) / np.sqrt(n_full)
            if not np.all(np.isfinite(R_new)):
                candidate_res = float("inf")

        if accepted and np.isfinite(candidate_res) and candidate_res < best_res:
            best_res = float(candidate_res)
            best_f = np.array(f_new, copy=True)
            best_lbe = lbe_calls
            best_wall = time.perf_counter() - t0

        f_prev = f
        f = f_new
        res_prev = res

    if final_polish_tol is not None and final_polish_max_steps > 0:
        for step in range(1, final_polish_max_steps + 1):
            f = case.lbe_step(f)
            lbe_calls += 1
            if step % final_polish_check_every == 0:
                R = case.residual(f)
                lbe_calls += 1
                res = case._fast_norm(R) / np.sqrt(n_full)
                wall = time.perf_counter() - t0
                history.append((k + step / max(final_polish_max_steps, 1), res, lbe_calls, wall))
                if verbose:
                    print(f"  polish {step:7d} | res {res:.3e} | lbe {lbe_calls:5d}")
                if not np.isfinite(res) or res < final_polish_tol:
                    break

    if best_res < np.inf and (not np.isfinite(history[-1][1]) or history[-1][1] > best_res):
        history.append((len(history), best_res, best_lbe, best_wall))
    return best_f, history
