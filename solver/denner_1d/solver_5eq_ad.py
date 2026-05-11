# solver/denner_1d/solver_5eq_ad.py
# Ref: Solver_fully_coupled_5_equation.md — §14 (권장 알고리즘)
#
# Fully-coupled 5-equation backward Euler solver with autograd Jacobian.
#
# Algorithm (§14):
#   Outer Picard (s=0,1,...):
#     1. EOS inversion → recover (p, T, u) from conservative state
#     2. Compute MWI face velocity θ (lagged)
#     3. Compute K coefficient (lagged)
#     4. Compute CICSAM γ_f (lagged, if used)
#     Inner Newton (k=0,1,...):
#       a. Evaluate residual R(W_k) using frozen θ, K
#       b. Compute Jacobian J = autograd.jacobian(R)(W_k)   [4N×4N]
#       c. Solve J·ΔW = -R  (dense, scipy.linalg.solve)
#       d. Line search → ω_k
#       e. W_{k+1} = W_k + ω_k·ΔW
#       f. Enforce physics bounds: p>0, T>0, 0≤α₁≤1
#       g. Newton convergence check
#     Outer convergence check (θ, K changes small)
#
# Conservative variables are NOT the Newton unknowns here — we use
# primitive variables W = {p, u, T, α₁} for the autograd residual.
# This avoids EOS inversion inside every residual evaluation (which would
# break the autograd graph). The EOS is called *forward* only.
#
# Jacobian memory:
#   N=10  → 4N=40,  J is 40×40  → negligible
#   N=200 → 4N=800, J is 800×800 → ~5 MB float64  → OK
#
# Reference: Solver_fully_coupled_5_equation.md §3.2, §10, §14, §15.1

import numpy as np
import scipy.linalg

from .eos.eos_class import create_eos
from .assembly_5eq_ad import (
    make_residual_prim_ad,
    make_residual_cons_ad,
    compute_jacobian_ad,
    compute_K_ad,
    compute_face_velocity_ad,
    invert_primitives,
    pack_W, unpack_W,
    pack_Q, unpack_Q,
    _get_eos_params, _eos_rho_anp, _eos_evol_anp,
    _mixture_p_from_Q_anp, _mixture_T_from_Q_anp,
    _AUTOGRAD_AVAILABLE,
)

_P_FLOOR = 1.0    # Pa
_T_FLOOR = 1e-3   # K


# ---------------------------------------------------------------------------
# Fallback: finite-difference Jacobian (when autograd not available)
# ---------------------------------------------------------------------------

def _fd_jacobian(residual_func, W_k, eps_scale=1e-7):
    """Finite-difference Jacobian (fallback when autograd unavailable).

    Parameters
    ----------
    residual_func : callable  R(W) → (M,)
    W_k           : ndarray (M,)
    eps_scale     : float

    Returns
    -------
    J : ndarray (M, M)
    """
    M  = len(W_k)
    R0 = residual_func(W_k)
    J  = np.zeros((M, M))
    for j in range(M):
        eps_j = max(abs(W_k[j]) * eps_scale, 1e-8)
        W_p = W_k.copy(); W_p[j] += eps_j
        J[:, j] = (residual_func(W_p) - R0) / eps_j
    return J


# ---------------------------------------------------------------------------
# Linear solve with fallback
# ---------------------------------------------------------------------------

def _solve_linear(J, b):
    """Solve J·x = b for the Newton update.

    Approach:
    1) Row + column equilibration so the row/col max is O(1).
    2) Solve the scaled system using the *Levenberg-Marquardt damped
       normal equations*:  (J^T J + lambda I) x = J^T b,  with
       lambda = damping_frac * mean(|diag(J^T J)|).
       This bounds dW automatically even when J has a near-zero
       singular value: along that direction the solution is now
       J^T b / (eigval^2 + lambda), so the worst-case amplification
       is 1/lambda, not 1/eigval.  For our primitive-variable Jacobian
       eigval can be 1e-5 of the dominant scale, so the previous direct
       solve produced dW with one component ~1e5x its physical value;
       the LM solve eliminates that pathology in a single step.
    3) If LM produces NaN or fails, fall back to a least-squares
       pseudo-inverse for finite output.
    """
    row_scale = np.max(np.abs(J), axis=1)
    row_scale = np.where(row_scale > 1e-300, row_scale, 1.0)
    J_s = J / row_scale[:, None]
    b_s = b / row_scale

    col_scale = np.max(np.abs(J_s), axis=0)
    col_scale = np.where(col_scale > 1e-300, col_scale, 1.0)
    J_sc = J_s / col_scale[None, :]

    # (1) Levenberg-Marquardt damped normal equations — primary path.
    # Solving (J^T J + lambda I) x = J^T b stays well-posed even when J
    # is singular; the resulting x is a descent direction for ||J x - b||.
    #
    # Lambda choice (Marquardt 1963 §V):
    #   lam = damping_frac * mean(diag(J^T J))
    # The previous 1e-6 * ||J||_F was orders of magnitude too small for
    # the primitive-variable Jacobian: J had one tiny singular value
    # (the velocity-momentum coupling), so (J^T J + 1e-6 lam) inverted
    # by ~1e5 along that direction and dW blew up by 1e5 in u.
    # Using the diagonal scale of J^T J as the lambda seed makes the
    # damping commensurate with the Jacobian's own scaling and bounds
    # the worst-case singular inversion by 1/damping_frac.
    try:
        JtJ = J_sc.T @ J_sc
        Jtb = J_sc.T @ b_s
        diag_mean = float(np.mean(np.abs(np.diag(JtJ))))
        lam = max(1.0e-3 * diag_mean, 1.0e-12)
        x_s = scipy.linalg.solve(JtJ + lam * np.eye(JtJ.shape[0]), Jtb,
                                  assume_a='sym')
        x = x_s / col_scale
        if np.all(np.isfinite(x)):
            return x
    except Exception:
        pass

    # (2) Least-squares pseudo-inverse — final safety net.
    try:
        x_s, _, _, _ = np.linalg.lstsq(J_sc, b_s, rcond=None)
        x = x_s / col_scale
        if np.all(np.isfinite(x)):
            return x
    except Exception:
        pass

    return np.zeros_like(b)


# ---------------------------------------------------------------------------
# Physics bounds enforcement
# ---------------------------------------------------------------------------

def _enforce_bounds(p_k, u_k, T_k, a1_k):
    """Enforce physical bounds after Newton update."""
    p_k  = np.maximum(p_k, _P_FLOOR)
    T_k  = np.maximum(T_k, _T_FLOOR)
    a1_k = np.clip(a1_k, 1e-10, 1.0 - 1e-10)
    return p_k, u_k, T_k, a1_k


# ---------------------------------------------------------------------------
# Inner Newton with AD Jacobian
# ---------------------------------------------------------------------------

def _inner_newton_ad(p_k, u_k, T_k, a1_k,
                     p_n, u_n, T_n, a1_n,
                     N, dx, dt,
                     ph1_params, ph2_params,
                     bc_l, bc_r,
                     theta_lag, K_lag,
                     max_newton, newton_tol,
                     use_autograd=True,
                     verbose=False):
    """Inner Newton iteration with autograd (or FD fallback) Jacobian.

    Parameters
    ----------
    p_k, u_k, T_k, a1_k : ndarray (N,)  current outer Picard iterate
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    N, dx, dt            : grid/time
    ph1_params, ph2_params : EOS parameter tuples
    bc_l, bc_r           : str
    theta_lag            : ndarray (N+1,)  lagged face velocity
    K_lag                : ndarray (N,)   lagged K coefficient
    max_newton           : int
    newton_tol           : float
    use_autograd         : bool
    verbose              : bool

    Returns
    -------
    p_out, u_out, T_out, a1_out : ndarray (N,)  converged Newton iterate
    info : dict
    """
    # Build autograd residual closure
    res_func = make_residual_prim_ad(
        p_n, u_n, T_n, a1_n,
        N, dx, dt,
        ph1_params, ph2_params,
        bc_l, bc_r,
        theta_lag, K_lag)

    W_k = pack_W(p_k, u_k, T_k, a1_k)
    info = {'converged': False, 'newton_iters': 0, 'residuals': []}

    for niter in range(max_newton):
        # 1. Residual at current iterate
        try:
            R_k = np.array(res_func(W_k))
        except Exception as exc:
            if verbose:
                print(f"      [AD Newton {niter}] residual failed: {exc}")
            break
        R_norm = float(np.linalg.norm(R_k))
        info['residuals'].append(R_norm)

        if not np.isfinite(R_norm):
            if verbose:
                print(f"      [AD Newton {niter}] non-finite residual")
            break

        # Early exit: residual already at machine precision (e.g. PE-static
        # cases where W_k matches the steady state exactly).
        if R_norm < newton_tol * max(1.0, float(np.linalg.norm(W_k))):
            info['converged'] = True
            info['newton_iters'] = niter
            break

        # 2. Jacobian
        try:
            if use_autograd and _AUTOGRAD_AVAILABLE:
                J_k = np.array(compute_jacobian_ad(res_func, W_k))
            else:
                J_k = _fd_jacobian(res_func, W_k)
        except Exception as exc:
            if verbose:
                print(f"      [AD Newton {niter}] Jacobian failed: {exc}")
            J_k = _fd_jacobian(res_func, W_k)

        # 3. Solve J·ΔW = -R
        dW = _solve_linear(J_k, -R_k)

        if not np.all(np.isfinite(dW)):
            dW = np.zeros_like(W_k)

        # 3b. Trust-region clip on Newton step.
        # Even when the Jacobian is numerically exact, the primitive-
        # variable Newton system is mildly ill-conditioned because the
        # mass and energy block diagonals differ by ~10 orders of
        # magnitude (e.g. d/dp(rho_water) ~ 1e-7 vs d/dT(rho_water) ~
        # 1e-3).  Solving J dW = -R can therefore produce a dW whose
        # ∞-norm exceeds W by many orders of magnitude — typically the
        # velocity component blows up by 10^5x because the momentum
        # block is the most singular.  We cap the per-iteration
        # relative state change at TR_REL so that the line search
        # operates within the radius of quadratic convergence.
        TR_REL = 0.05  # 5% max relative change per Newton iter
        ratios = np.abs(dW) / (np.abs(W_k) + 1e-10)
        rmax = float(np.max(ratios))
        if rmax > TR_REL:
            dW = dW * (TR_REL / rmax)

        # 4. Backtracking line search (8 halvings)
        omega = 1.0
        ls_improved = False
        W_trial = W_k.copy()
        R_trial_norm = R_norm

        for _ls in range(8):
            W_trial = W_k + omega * dW

            # Enforce bounds in trial state
            p_tr, u_tr, T_tr, a1_tr = unpack_W(W_trial, N)
            p_tr, u_tr, T_tr, a1_tr = _enforce_bounds(p_tr, u_tr, T_tr, a1_tr)
            W_trial = pack_W(p_tr, u_tr, T_tr, a1_tr)

            if not np.all(np.isfinite(W_trial)):
                omega *= 0.5
                continue

            try:
                R_trial = np.array(res_func(W_trial))
                R_trial_norm = float(np.linalg.norm(R_trial))
                if np.isfinite(R_trial_norm) and R_trial_norm < R_norm:
                    ls_improved = True
                    break
            except Exception:
                pass
            omega *= 0.5

        if ls_improved:
            W_k = W_trial
        else:
            # --- Block Gauss-Seidel fallback ---
            # Full J solve failed (ill-conditioned due to mass-alpha proportionality).
            # Solve alpha block first, update residual, then solve remaining 3N.
            #
            # Block structure: W = [p(0:N), u(N:2N), T(2N:3N), a1(3N:4N)]
            #                  R = [R_m1, R_mom, R_en, R_a1]
            J_aa = J_k[3*N:4*N, 3*N:4*N]  # alpha-alpha block
            R_a  = R_k[3*N:4*N]            # alpha residual

            # Step 1: Solve alpha equation  J_aa · Δα₁ = -R_a
            try:
                da1 = np.linalg.solve(J_aa, -R_a)
            except np.linalg.LinAlgError:
                da1 = np.zeros(N)

            # Step 2: Update residual for remaining equations
            # R' = R + J[:, 3N:4N] · Δα₁  (effect of alpha change on all equations)
            J_xa = J_k[:3*N, 3*N:4*N]   # coupling: how alpha affects mass/mom/energy
            R_rest = R_k[:3*N] + J_xa @ da1

            # Step 3: Solve 3N system for (Δp, Δu, ΔT)
            J_xx = J_k[:3*N, :3*N]  # 3N×3N block
            try:
                dx_rest = _solve_linear(J_xx, -R_rest)
            except Exception:
                dx_rest = np.zeros(3*N)

            dW_bgs = np.concatenate([dx_rest, da1])

            # Step 4: Line search with block GS direction
            omega_bgs = 1.0
            ls_bgs_improved = False
            for _ls_bgs in range(12):
                W_trial_bgs = W_k + omega_bgs * dW_bgs
                p_tr, u_tr, T_tr, a1_tr = unpack_W(W_trial_bgs, N)
                p_tr, u_tr, T_tr, a1_tr = _enforce_bounds(p_tr, u_tr, T_tr, a1_tr)
                W_trial_bgs = pack_W(p_tr, u_tr, T_tr, a1_tr)

                if not np.all(np.isfinite(W_trial_bgs)):
                    omega_bgs *= 0.5
                    continue

                try:
                    R_trial_bgs = np.array(res_func(W_trial_bgs))
                    R_trial_bgs_norm = float(np.linalg.norm(R_trial_bgs))
                    if np.isfinite(R_trial_bgs_norm) and R_trial_bgs_norm < R_norm:
                        ls_bgs_improved = True
                        break
                except Exception:
                    pass
                omega_bgs *= 0.5

            if ls_bgs_improved:
                W_k = W_trial_bgs
                omega = omega_bgs
                dW = dW_bgs
                if verbose:
                    print(f"      [AD Newton {niter:3d}] BGS fallback: ω={omega_bgs:.3f}")
            else:
                # Both full + BGS line searches were rejected. Apply a
                # trust-region damped step: choose omega so that the
                # ∞-norm of the relative state change stays below tr_frac.
                # If even this micro step fails the bound test, give up
                # (Newton iteration terminates with the prior R, the outer
                # loop / time-step machinery can react).
                tr_frac = 1.0e-3
                dW_use = dW if np.all(np.isfinite(dW)) else dW_bgs
                if dW_use is None or not np.all(np.isfinite(dW_use)):
                    omega = 0.0
                else:
                    rel_dW = np.max(np.abs(dW_use) / (np.abs(W_k) + 1e-10))
                    omega_tr = tr_frac / max(rel_dW, tr_frac)  # ≤ 1
                    W_trial_tr = W_k + omega_tr * dW_use
                    p_tr, u_tr, T_tr, a1_tr = unpack_W(W_trial_tr, N)
                    p_tr, u_tr, T_tr, a1_tr = _enforce_bounds(p_tr, u_tr, T_tr, a1_tr)
                    W_trial_tr = pack_W(p_tr, u_tr, T_tr, a1_tr)
                    if np.all(np.isfinite(W_trial_tr)):
                        try:
                            R_tr = np.array(res_func(W_trial_tr))
                            R_tr_norm = float(np.linalg.norm(R_tr))
                        except Exception:
                            R_tr_norm = float('inf')
                        # Accept only if residual did not blow up (allow
                        # mild non-monotonic growth up to 1.5x — small
                        # primal-feasibility margin).
                        if np.isfinite(R_tr_norm) and R_tr_norm < 1.5 * R_norm:
                            W_k = W_trial_tr
                            omega = omega_tr
                            dW = dW_use
                            if verbose:
                                print(f"      [AD Newton {niter:3d}] trust-region ω={omega_tr:.1e}"
                                      f"  R_tr/R={R_tr_norm/max(R_norm,1e-300):.3f}")
                        else:
                            omega = 0.0
                    else:
                        omega = 0.0

        # 5. Convergence check (relative update norm)
        dW_rel = np.max(np.abs(omega * dW) / (np.abs(W_k) + 1e-10))
        info['newton_iters'] = niter + 1

        if verbose and (niter < 5 or niter % 10 == 0):
            print(f"      [AD Newton {niter:3d}]: |R|={R_norm:.3e}  "
                  f"|ΔW|_rel={dW_rel:.3e}  ω={omega:.3f}")

        if dW_rel < newton_tol and omega > 0:
            info['converged'] = True
            break

    # Final residual norm
    try:
        R_final = np.array(res_func(W_k))
        info['R_final_norm'] = float(np.linalg.norm(R_final))
    except Exception:
        info['R_final_norm'] = float('nan')

    p_out, u_out, T_out, a1_out = unpack_W(W_k, N)
    p_out, u_out, T_out, a1_out = _enforce_bounds(p_out, u_out, T_out, a1_out)
    return p_out, u_out, T_out, a1_out, info


# ---------------------------------------------------------------------------
# Outer Picard loop
# ---------------------------------------------------------------------------

def _outer_picard_5eq_ad(N, dx, dt,
                          p_n, u_n, T_n, a1_n,
                          ph1, ph2, bc_l, bc_r,
                          cfg,
                          theta_old=None, u_bar_old=None, rho_star_old=None):
    """5-equation fully-coupled solver: outer Picard + inner AD Newton.

    Implements §14 of Solver_fully_coupled_5_equation.md.

    Parameters
    ----------
    N, dx, dt    : int, float, float
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    ph1, ph2     : EOS dicts or objects
    bc_l, bc_r   : str
    cfg          : dict
    theta_old, u_bar_old, rho_star_old : MWI transient correction data

    Returns
    -------
    p_f, u_f, T_f, a1_f : ndarray (N,)  new-time primitives
    info : dict
    """
    max_outer   = cfg.get('max_outer', 3)
    max_newton  = cfg.get('max_newton', 20)
    newton_tol  = cfg.get('newton_tol', 1e-6)
    outer_tol   = cfg.get('outer_tol', 1e-6)
    use_ad      = cfg.get('use_autograd', True) and _AUTOGRAD_AVAILABLE
    verbose     = cfg.get('verbose_newton', False)

    # EOS parameter tuples for autograd-compatible functions
    ph1_params = _get_eos_params(ph1)
    ph2_params = _get_eos_params(ph2)

    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # Initial Picard iterate = old-time
    p_k  = p_n.copy()
    u_k  = u_n.copy()
    T_k  = T_n.copy()
    a1_k = np.clip(a1_n.copy(), 1e-10, 1.0 - 1e-10)

    info_out = {
        'converged':    False,
        'outer_iters':  0,
        'inner_iters':  [],
        'residuals':    [],
        'picard_iters': 0,
    }

    # Track best Picard state (smallest outer_change with converged Newton)
    best_state = (p_k.copy(), u_k.copy(), T_k.copy(), a1_k.copy())
    best_change = float('inf')
    prev_change = float('inf')

    for outer in range(max_outer):
        # ---------------------------------------------------------------
        # Step 1: Compute lagged MWI face velocity (θ)
        # ---------------------------------------------------------------
        a2_k = 1.0 - a1_k
        rho1_k = eos1.rho(p_k, T_k)
        rho2_k = eos2.rho(p_k, T_k)
        rho_k  = np.maximum(a1_k * rho1_k + a2_k * rho2_k, 1e-300)

        theta_lag, u_bar_lag, d_hat_lag, rho_star_lag = compute_face_velocity_ad(
            u_k, p_k, rho_k, dx, dt, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old)

        # ---------------------------------------------------------------
        # Step 2: Compute lagged K coefficient (Wood's)
        # ---------------------------------------------------------------
        K_lag = compute_K_ad(a1_k, p_k, T_k, ph1, ph2)

        # ---------------------------------------------------------------
        # Step 3: Inner Newton with AD Jacobian
        # ---------------------------------------------------------------
        p_new, u_new, T_new, a1_new, info_inner = _inner_newton_ad(
            p_k, u_k, T_k, a1_k,
            p_n, u_n, T_n, a1_n,
            N, dx, dt,
            ph1_params, ph2_params,
            bc_l, bc_r,
            theta_lag, K_lag,
            max_newton, newton_tol,
            use_autograd=use_ad,
            verbose=verbose)

        info_out['inner_iters'].append(info_inner['newton_iters'])
        info_out['residuals'].extend(info_inner['residuals'])

        if verbose:
            print(f"    [Picard {outer}] inner Newton: "
                  f"{info_inner['newton_iters']} iters, "
                  f"R_final={info_inner.get('R_final_norm', float('nan')):.3e}, "
                  f"converged={info_inner['converged']}")

        # ---------------------------------------------------------------
        # Step 4: Outer convergence check
        # ---------------------------------------------------------------
        dp   = float(np.max(np.abs(p_new  - p_k))  / (np.max(np.abs(p_k))  + 1.0))
        du   = float(np.max(np.abs(u_new  - u_k))  / (np.max(np.abs(u_k))  + 1e-6))
        dT   = float(np.max(np.abs(T_new  - T_k))  / (np.max(np.abs(T_k))  + 1.0))
        da1  = float(np.max(np.abs(a1_new - a1_k)))

        p_k, u_k, T_k, a1_k = p_new, u_new, T_new, a1_new
        info_out['outer_iters'] = outer + 1
        info_out['picard_iters'] = outer + 1

        outer_change = max(dp, du, dT, da1)
        if verbose:
            print(f"    [Picard {outer}] outer change: {outer_change:.3e}")

        # Track best converged state
        if info_inner['converged'] and outer_change < best_change:
            best_change = outer_change
            best_state = (p_k.copy(), u_k.copy(), T_k.copy(), a1_k.copy())

        if info_inner['converged'] and outer_change < outer_tol:
            info_out['converged'] = True
            break

        # Picard divergence detection: if outer_change grows significantly,
        # revert to best state and stop
        if outer > 0 and outer_change > 10.0 * prev_change and prev_change < 1.0:
            if verbose:
                print(f"    [Picard {outer}] divergence detected, reverting to best state")
            p_k, u_k, T_k, a1_k = best_state
            info_out['converged'] = (best_change < outer_tol)
            break

        prev_change = outer_change

    return p_k, u_k, T_k, a1_k, info_out


# ---------------------------------------------------------------------------
# Public API: step_5eq_ad
# ---------------------------------------------------------------------------

def step_5eq_ad(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg):
    """One time step of the 5-equation AD-Jacobian Newton solver.

    Interface compatible with solver_a.step.

    Parameters / Returns identical to solver_5eq.step_5eq.

    Notes
    -----
    Uses autograd.jacobian for Jacobian computation (falls back to
    finite-difference if autograd is not installed).

    cfg keys specific to this solver:
        use_autograd : bool  (default True)  — use autograd vs FD Jacobian
        max_outer    : int   (default 3)     — outer Picard iterations
        max_newton   : int   (default 20)    — inner Newton iterations
        newton_tol   : float (default 1e-6)  — Newton convergence tolerance
        outer_tol    : float (default 1e-6)  — Picard convergence tolerance
        verbose_newton : bool (default False) — print Newton iteration info
    """
    if cfg is None:
        cfg = {}

    N = len(state['p'])

    p_n  = state['p'].copy()
    u_n  = state['u'].copy()
    T_n  = state['T'].copy()
    a1_n = np.clip(state['psi'].copy(), 1e-10, 1.0 - 1e-10)

    # Transient MWI correction data
    is_first     = aux.get('is_first_step', True)
    theta_old    = aux.get('theta_old',    None)
    u_bar_old    = aux.get('u_bar_old',    None)
    rho_star_old = aux.get('rho_star_old', None)
    if is_first:
        theta_old = u_bar_old = rho_star_old = None

    # Run the outer Picard + inner Newton
    p_f, u_f, T_f, a1_f, info = _outer_picard_5eq_ad(
        N, dx, dt,
        p_n, u_n, T_n, a1_n,
        ph1, ph2, bc_l, bc_r, cfg,
        theta_old=theta_old, u_bar_old=u_bar_old, rho_star_old=rho_star_old)

    # Final physical bounds
    p_f  = np.maximum(p_f, _P_FLOOR)
    T_f  = np.maximum(T_f, _T_FLOOR)
    a1_f = np.clip(a1_f, 1e-10, 1.0 - 1e-10)
    a2_f = 1.0 - a1_f

    # Reconstruct derived quantities
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)
    rho1_f = eos1.rho(p_f, T_f)
    rho2_f = eos2.rho(p_f, T_f)
    rho_f  = np.maximum(a1_f * rho1_f + a2_f * rho2_f, 1e-300)

    rE_f = (a1_f * eos1.e_vol(p_f, T_f)
            + a2_f * eos2.e_vol(p_f, T_f)
            + 0.5 * rho_f * u_f**2)
    E_total_f = rE_f / rho_f

    # MWI for new_aux
    theta_f, u_bar_f, d_hat_f, rho_star_f = compute_face_velocity_ad(
        u_f, p_f, rho_f, dx, dt, bc_l, bc_r,
        theta_old=theta_old, u_bar_old=u_bar_old,
        rho_star_old=rho_star_old)

    new_state = {
        'p':       p_f,
        'u':       u_f,
        'T':       T_f,
        'psi':     a1_f,
        'rho':     rho_f,
        'E_total': E_total_f,
        'u_face':  theta_f,
    }

    new_aux = {
        'is_first_step':  False,
        'bdf_order':      1,
        'rho_nm1':        rho_f,
        'rhoU_nm1':       rho_f * u_f,
        'E_nm1':          E_total_f,
        'rho_face_acid':  None,
        'dt_prev':        dt,
        'theta_old':      theta_f,
        'u_bar_old':      u_bar_f,
        'rho_star_old':   rho_star_f,
    }

    info_out = {
        'converged':    info['converged'],
        'outer_iters':  info['outer_iters'],
        'inner_iters':  info['inner_iters'],
        'picard_iters': info['picard_iters'],
        'residuals':    info['residuals'],
    }

    return new_state, new_aux, info_out


# ===========================================================================
# Conservative-variable Newton solver (5N) with FD Jacobian
# ===========================================================================

def _inner_newton_cons(Q_k, Q_n, N, dx, dt, ph1_params, ph2_params,
                       bc_l, bc_r, theta_lag, K_lag, p_guess, T_guess,
                       max_newton=30, newton_tol=1e-8, verbose=False):
    """Newton solver in conservative variables Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}.

    Uses FD Jacobian on the conservative residual. EOS inversion happens
    inside the residual function (not autograd — standard numpy).
    """
    from .assembly_5eq_ad import (make_residual_cons, fd_jacobian_cons,
                                   pack_Q, unpack_Q, invert_primitives)

    res_func = make_residual_cons(Q_n, N, dx, dt, ph1_params, ph2_params,
                                   bc_l, bc_r, theta_lag, K_lag,
                                   p_guess, T_guess)

    Q = Q_k.copy()
    info = {'converged': False, 'newton_iters': 0, 'residuals': [],
            'R_final_norm': float('nan')}

    R0 = res_func(Q)
    R_norm = float(np.linalg.norm(R0))
    R_norm_0 = R_norm
    info['residuals'].append(R_norm)

    for niter in range(max_newton):
        if R_norm < 1e-10:
            info['converged'] = True
            info['newton_iters'] = niter
            break

        # FD Jacobian
        J = fd_jacobian_cons(res_func, Q, N)

        # Solve J·ΔQ = -R
        try:
            dQ = np.linalg.solve(J, -R0)
        except np.linalg.LinAlgError:
            dQ = np.linalg.lstsq(J, -R0, rcond=None)[0]

        if not np.all(np.isfinite(dQ)):
            if verbose:
                print(f"      [cons Newton {niter}] non-finite dQ")
            break

        # Backtracking line search
        omega = 1.0
        ls_ok = False
        for _ls in range(10):
            Q_trial = Q + omega * dQ
            # Bounds
            a1r1_t, a2r2_t, ru_t, rE_t, a1_t = unpack_Q(Q_trial, N)
            a1r1_t = np.maximum(a1r1_t, 1e-20)
            a2r2_t = np.maximum(a2r2_t, 1e-20)
            a1_t = np.clip(a1_t, 1e-14, 1.0 - 1e-14)
            Q_trial = pack_Q(a1r1_t, a2r2_t, ru_t, rE_t, a1_t)

            try:
                R_trial = res_func(Q_trial)
                Rn = float(np.linalg.norm(R_trial))
                if np.isfinite(Rn) and Rn < R_norm:
                    ls_ok = True
                    R0 = R_trial
                    R_norm = Rn
                    Q = Q_trial
                    break
            except Exception:
                pass
            omega *= 0.5

        if verbose and (niter < 5 or niter % 10 == 0):
            print(f"      [cons Newton {niter:3d}]: |R|={R_norm:.3e}  ω={omega:.4f}  ls={ls_ok}")

        info['residuals'].append(R_norm)
        info['newton_iters'] = niter + 1

        if not ls_ok:
            break

        # Convergence check (relative residual)
        if R_norm < newton_tol * R_norm_0 or R_norm < 1e-10:
            info['converged'] = True
            break

    info['R_final_norm'] = R_norm
    return Q, info


def _outer_picard_cons(N, dx, dt, p_n, u_n, T_n, a1_n,
                       ph1, ph2, bc_l, bc_r, cfg,
                       theta_old=None, u_bar_old=None, rho_star_old=None):
    """Conservative 5-eq solver: outer Picard + inner conservative Newton."""
    from .assembly_5eq_ad import (_get_eos_params, compute_face_velocity_ad,
                                   compute_K_ad, invert_primitives,
                                   pack_Q, unpack_Q)

    max_outer  = cfg.get('max_outer', 3)
    max_newton = cfg.get('max_newton', 30)
    newton_tol = cfg.get('newton_tol', 1e-8)
    verbose    = cfg.get('verbose_newton', False)

    ph1_params = _get_eos_params(ph1)
    ph2_params = _get_eos_params(ph2)
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # Initial conservative state from old-time primitives
    a1_n_c = np.clip(a1_n, 1e-14, 1.0 - 1e-14)
    a2_n = 1.0 - a1_n_c
    rho1_n = eos1.rho(p_n, T_n)
    rho2_n = eos2.rho(p_n, T_n)
    rho_n  = a1_n_c * rho1_n + a2_n * rho2_n

    a1r1_n = a1_n_c * rho1_n
    a2r2_n = a2_n   * rho2_n
    ru_n   = rho_n  * u_n
    evol1_n = eos1.e_vol(p_n, T_n)
    evol2_n = eos2.e_vol(p_n, T_n)
    rE_n   = a1_n_c * evol1_n + a2_n * evol2_n + 0.5 * rho_n * u_n**2

    Q_n = pack_Q(a1r1_n, a2r2_n, ru_n, rE_n, a1_n_c)
    Q_k = Q_n.copy()

    p_k, u_k, T_k, a1_k = p_n.copy(), u_n.copy(), T_n.copy(), a1_n_c.copy()

    info_out = {'converged': False, 'outer_iters': 0, 'inner_iters': [],
                'residuals': [], 'picard_iters': 0}

    for outer in range(max_outer):
        # Compute lagged theta and K from current primitives
        a2_k = 1.0 - a1_k
        rho1_k = eos1.rho(p_k, T_k)
        rho2_k = eos2.rho(p_k, T_k)
        rho_k  = np.maximum(a1_k * rho1_k + a2_k * rho2_k, 1e-300)

        theta_lag, u_bar_lag, d_hat_lag, rho_star_lag = compute_face_velocity_ad(
            u_k, p_k, rho_k, dx, dt, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old)

        K_lag = compute_K_ad(a1_k, p_k, T_k, ph1, ph2)

        # Inner Newton in conservative variables
        Q_new, info_inner = _inner_newton_cons(
            Q_k, Q_n, N, dx, dt, ph1_params, ph2_params,
            bc_l, bc_r, theta_lag, K_lag, p_k, T_k,
            max_newton, newton_tol, verbose)

        info_out['inner_iters'].append(info_inner['newton_iters'])
        info_out['residuals'].extend(info_inner['residuals'])

        # Recover primitives from new Q
        a1r1_new, a2r2_new, ru_new, rE_new, a1_new = unpack_Q(Q_new, N)
        a1_new = np.clip(a1_new, 1e-14, 1.0 - 1e-14)
        rho_new = np.maximum(a1r1_new + a2r2_new, 1e-300)
        u_new = ru_new / rho_new
        rho_e_new = rE_new - 0.5 * rho_new * u_new**2
        p_new, T_new = invert_primitives(
            a1_new, rho_new, rho_e_new, p_k, T_k, ph1_params, ph2_params)

        if verbose:
            print(f"    [Picard {outer}] cons Newton: "
                  f"{info_inner['newton_iters']} iters, "
                  f"R_final={info_inner.get('R_final_norm', float('nan')):.3e}, "
                  f"converged={info_inner['converged']}")

        # Outer convergence check
        dp  = float(np.max(np.abs(p_new - p_k)) / (np.max(np.abs(p_k)) + 1.0))
        du  = float(np.max(np.abs(u_new - u_k)) / (np.max(np.abs(u_k)) + 1e-6))
        da1 = float(np.max(np.abs(a1_new - a1_k)))
        outer_change = max(dp, du, da1)

        p_k, u_k, T_k, a1_k = p_new, u_new, T_new, a1_new
        Q_k = Q_new
        info_out['outer_iters'] = outer + 1
        info_out['picard_iters'] = outer + 1

        if verbose:
            print(f"    [Picard {outer}] outer change: {outer_change:.3e}")

        if info_inner['converged'] and outer_change < 1e-4:
            info_out['converged'] = True
            break

    return p_k, u_k, T_k, a1_k, info_out


def step_5eq_cons(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg):
    """One time step of the conservative 5-eq solver with FD Jacobian."""
    if cfg is None:
        cfg = {}

    N = len(state['p'])
    p_n  = state['p'].copy()
    u_n  = state['u'].copy()
    T_n  = state['T'].copy()
    a1_n = np.clip(state['psi'].copy(), 1e-14, 1.0 - 1e-14)

    is_first     = aux.get('is_first_step', True)
    theta_old    = aux.get('theta_old', None)
    u_bar_old    = aux.get('u_bar_old', None)
    rho_star_old = aux.get('rho_star_old', None)
    if is_first:
        theta_old = u_bar_old = rho_star_old = None

    p_f, u_f, T_f, a1_f, info = _outer_picard_cons(
        N, dx, dt, p_n, u_n, T_n, a1_n,
        ph1, ph2, bc_l, bc_r, cfg,
        theta_old=theta_old, u_bar_old=u_bar_old, rho_star_old=rho_star_old)

    p_f  = np.maximum(p_f, _P_FLOOR)
    T_f  = np.maximum(T_f, _T_FLOOR)
    a1_f = np.clip(a1_f, 1e-14, 1.0 - 1e-14)
    a2_f = 1.0 - a1_f

    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)
    rho1_f = eos1.rho(p_f, T_f)
    rho2_f = eos2.rho(p_f, T_f)
    rho_f  = np.maximum(a1_f * rho1_f + a2_f * rho2_f, 1e-300)

    rE_f = (a1_f * eos1.e_vol(p_f, T_f)
            + a2_f * eos2.e_vol(p_f, T_f)
            + 0.5 * rho_f * u_f**2)
    E_total_f = rE_f / rho_f

    from .assembly_5eq_ad import compute_face_velocity_ad
    theta_f, u_bar_f, d_hat_f, rho_star_f = compute_face_velocity_ad(
        u_f, p_f, rho_f, dx, dt, bc_l, bc_r,
        theta_old=theta_old, u_bar_old=u_bar_old,
        rho_star_old=rho_star_old)

    new_state = {
        'p': p_f, 'u': u_f, 'T': T_f, 'psi': a1_f,
        'rho': rho_f, 'E_total': E_total_f, 'u_face': theta_f,
    }
    new_aux = {
        'is_first_step': False, 'bdf_order': 1,
        'rho_nm1': rho_f, 'rhoU_nm1': rho_f * u_f, 'E_nm1': E_total_f,
        'rho_face_acid': None, 'dt_prev': dt,
        'theta_old': theta_f, 'u_bar_old': u_bar_f, 'rho_star_old': rho_star_f,
    }
    info_out = {
        'converged': info['converged'], 'outer_iters': info['outer_iters'],
        'inner_iters': info['inner_iters'], 'picard_iters': info['picard_iters'],
        'residuals': info['residuals'],
    }
    return new_state, new_aux, info_out


# ===========================================================================
# Conservative-variable Newton with autograd Jacobian (5N)
# ===========================================================================

def _inner_newton_cons_ad(Q_k, Q_n, N, dx, dt,
                           ph1_params, ph2_params,
                           bc_l, bc_r, theta_lag, K_lag,
                           p_lag=None, T_lag=None,
                           max_newton=30, newton_tol=1e-8,
                           use_autograd=True, verbose=False):
    """Newton solver in conservative Q with autograd (or FD-fallback) Jacobian.

    Uses make_residual_cons_ad with partially-lagged ACID: p,T from outer
    Picard are frozen constants; only direct conservative Q variables are
    live autograd variables. This gives clean Jacobian (temporal I/dt +
    upwind advection) while preserving PE through ACID.

    Parameters
    ----------
    Q_k           : ndarray (5N,)  current conservative iterate
    Q_n           : ndarray (5N,)  old-time conservative state
    N, dx, dt     : int, float, float
    ph1_params, ph2_params : tuple from _get_eos_params()
    bc_l, bc_r    : str
    theta_lag     : ndarray (N+1,)  lagged face velocity
    K_lag         : ndarray (N,)   lagged K coefficient
    p_lag         : ndarray (N,) or None  lagged pressure
    T_lag         : ndarray (N,) or None  lagged temperature
    max_newton    : int
    newton_tol    : float
    use_autograd  : bool
    verbose       : bool

    Returns
    -------
    Q_out : ndarray (5N,)  converged conservative state
    info  : dict
    """
    # Build autograd-differentiable residual closure
    res_func = make_residual_cons_ad(
        Q_n, N, dx, dt, ph1_params, ph2_params,
        bc_l, bc_r, theta_lag, K_lag,
        p_lag=p_lag, T_lag=T_lag)

    Q = Q_k.copy()
    info = {'converged': False, 'newton_iters': 0, 'residuals': [],
            'R_final_norm': float('nan')}

    # Initial residual
    try:
        R0 = np.array(res_func(Q))
    except Exception as exc:
        if verbose:
            print(f"      [cons AD Newton] initial residual failed: {exc}")
        return Q, info

    R_norm = float(np.linalg.norm(R0))
    R_norm_0 = R_norm + 1e-300
    info['residuals'].append(R_norm)

    for niter in range(max_newton):
        if not np.isfinite(R_norm):
            if verbose:
                print(f"      [cons AD Newton {niter}] non-finite residual")
            break

        if R_norm < 1e-10:
            info['converged'] = True
            info['newton_iters'] = niter
            break

        # --- Jacobian ---
        try:
            if use_autograd and _AUTOGRAD_AVAILABLE:
                J = np.array(compute_jacobian_ad(res_func, Q))
            else:
                J = _fd_jacobian(res_func, Q)
        except Exception as exc:
            if verbose:
                print(f"      [cons AD Newton {niter}] Jacobian failed: {exc}")
            J = _fd_jacobian(res_func, Q)

        # --- Solve J·ΔQ = -R ---
        dQ = _solve_linear(J, -R0)

        if not np.all(np.isfinite(dQ)):
            if verbose:
                print(f"      [cons AD Newton {niter}] non-finite dQ")
            break

        # --- Backtracking line search (8 halvings) ---
        omega = 1.0
        ls_ok = False
        for _ls in range(8):
            Q_trial = Q + omega * dQ

            # Enforce physical bounds
            a1r1_t, a2r2_t, ru_t, rE_t, a1_t = unpack_Q(Q_trial, N)
            a1r1_t = np.maximum(a1r1_t, 1e-20)
            a2r2_t = np.maximum(a2r2_t, 1e-20)
            a1_t   = np.clip(a1_t, 1e-10, 1.0 - 1e-10)
            Q_trial = pack_Q(a1r1_t, a2r2_t, ru_t, rE_t, a1_t)

            if not np.all(np.isfinite(Q_trial)):
                omega *= 0.5
                continue

            try:
                R_trial = np.array(res_func(Q_trial))
                Rn = float(np.linalg.norm(R_trial))
                if np.isfinite(Rn) and Rn < R_norm:
                    ls_ok = True
                    R0 = R_trial
                    R_norm = Rn
                    Q = Q_trial
                    break
            except Exception:
                pass
            omega *= 0.5

        if verbose and (niter < 5 or niter % 10 == 0):
            print(f"      [cons AD Newton {niter:3d}]: |R|={R_norm:.3e}  "
                  f"ω={omega:.4f}  ls={ls_ok}")

        info['residuals'].append(R_norm)
        info['newton_iters'] = niter + 1

        if not ls_ok:
            break

        # Convergence check: relative residual or absolute floor
        if R_norm < newton_tol * R_norm_0 or R_norm < 1e-10:
            info['converged'] = True
            break

    info['R_final_norm'] = R_norm
    return Q, info


def _outer_picard_cons_ad(N, dx, dt, p_n, u_n, T_n, a1_n,
                           ph1, ph2, bc_l, bc_r, cfg,
                           theta_old=None, u_bar_old=None, rho_star_old=None):
    """Conservative 5-eq solver: outer Picard + inner conservative AD Newton.

    Identical structure to _outer_picard_cons but uses _inner_newton_cons_ad
    (autograd Jacobian through algebraic p/T recovery) instead of FD Jacobian.

    Parameters
    ----------
    N, dx, dt      : int, float, float
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    ph1, ph2       : EOS dicts or objects
    bc_l, bc_r     : str
    cfg            : dict
    theta_old, u_bar_old, rho_star_old : MWI transient correction data

    Returns
    -------
    p_f, u_f, T_f, a1_f : ndarray (N,)
    info : dict
    """
    max_outer  = cfg.get('max_outer',  3)
    max_newton = cfg.get('max_newton', 30)
    newton_tol = cfg.get('newton_tol', 1e-8)
    verbose    = cfg.get('verbose_newton', False)
    use_ad     = cfg.get('use_autograd', True) and _AUTOGRAD_AVAILABLE

    ph1_params = _get_eos_params(ph1)
    ph2_params = _get_eos_params(ph2)
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # Build old-time conservative state
    a1_n_c = np.clip(a1_n, 1e-14, 1.0 - 1e-14)
    a2_n   = 1.0 - a1_n_c
    rho1_n = eos1.rho(p_n, T_n)
    rho2_n = eos2.rho(p_n, T_n)
    rho_n  = a1_n_c * rho1_n + a2_n * rho2_n
    a1r1_n = a1_n_c * rho1_n
    a2r2_n = a2_n   * rho2_n
    ru_n   = rho_n  * u_n
    rE_n   = (a1_n_c * eos1.e_vol(p_n, T_n)
              + a2_n * eos2.e_vol(p_n, T_n)
              + 0.5 * rho_n * u_n**2)

    Q_n = pack_Q(a1r1_n, a2r2_n, ru_n, rE_n, a1_n_c)
    Q_k = Q_n.copy()

    # Current primitive estimates (used for theta/K lagging)
    p_k  = p_n.copy()
    u_k  = u_n.copy()
    T_k  = T_n.copy()
    a1_k = a1_n_c.copy()

    info_out = {'converged': False, 'outer_iters': 0, 'inner_iters': [],
                'residuals': [], 'picard_iters': 0}

    for outer in range(max_outer):
        # ---------------------------------------------------------------
        # Step 1: Lagged MWI face velocity (theta)
        # ---------------------------------------------------------------
        a2_k   = 1.0 - a1_k
        rho1_k = eos1.rho(p_k, T_k)
        rho2_k = eos2.rho(p_k, T_k)
        rho_k  = np.maximum(a1_k * rho1_k + a2_k * rho2_k, 1e-300)

        theta_lag, u_bar_lag, d_hat_lag, rho_star_lag = compute_face_velocity_ad(
            u_k, p_k, rho_k, dx, dt, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old)

        # ---------------------------------------------------------------
        # Step 2: Lagged K coefficient (Wood's)
        # ---------------------------------------------------------------
        K_lag = compute_K_ad(a1_k, p_k, T_k, ph1, ph2)

        # ---------------------------------------------------------------
        # Step 3: Inner Newton with AD Jacobian (conservative variables)
        # ---------------------------------------------------------------
        Q_new, info_inner = _inner_newton_cons_ad(
            Q_k, Q_n, N, dx, dt, ph1_params, ph2_params,
            bc_l, bc_r, theta_lag, K_lag,
            p_lag=p_k, T_lag=T_k,
            max_newton=max_newton, newton_tol=newton_tol,
            use_autograd=use_ad, verbose=verbose)

        info_out['inner_iters'].append(info_inner['newton_iters'])
        info_out['residuals'].extend(info_inner['residuals'])

        # ---------------------------------------------------------------
        # Step 4: Recover primitives from new Q (plain numpy)
        # ---------------------------------------------------------------
        a1r1_new, a2r2_new, ru_new, rE_new, a1_new = unpack_Q(Q_new, N)
        a1_new = np.clip(a1_new, 1e-14, 1.0 - 1e-14)

        # Algebraic recovery (same formulas as residual, but numpy)
        p_new = _mixture_p_from_Q_anp(a1r1_new, a2r2_new, ru_new, rE_new,
                                       a1_new, ph1_params, ph2_params)
        p_new = np.array(p_new, dtype=float)
        T_new = _mixture_T_from_Q_anp(a1r1_new, a2r2_new, a1_new,
                                       p_new, ph1_params, ph2_params)
        T_new = np.array(T_new, dtype=float)
        rho_new = np.maximum(a1r1_new + a2r2_new, 1e-300)
        u_new   = ru_new / rho_new

        if verbose:
            print(f"    [cons AD Picard {outer}] inner Newton: "
                  f"{info_inner['newton_iters']} iters, "
                  f"R_final={info_inner.get('R_final_norm', float('nan')):.3e}, "
                  f"converged={info_inner['converged']}")

        # ---------------------------------------------------------------
        # Step 5: Outer convergence check
        # ---------------------------------------------------------------
        dp   = float(np.max(np.abs(p_new  - p_k))  / (np.max(np.abs(p_k))  + 1.0))
        du   = float(np.max(np.abs(u_new  - u_k))  / (np.max(np.abs(u_k))  + 1e-6))
        da1  = float(np.max(np.abs(a1_new - a1_k)))
        outer_change = max(dp, du, da1)

        p_k, u_k, T_k, a1_k = p_new, u_new, T_new, a1_new
        Q_k = Q_new
        info_out['outer_iters']  = outer + 1
        info_out['picard_iters'] = outer + 1

        if verbose:
            print(f"    [cons AD Picard {outer}] outer change: {outer_change:.3e}")

        if info_inner['converged'] and outer_change < 1e-4:
            info_out['converged'] = True
            break

    return p_k, u_k, T_k, a1_k, info_out


def step_5eq_cons_ad(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg):
    """One time step of the conservative 5-eq solver with autograd Jacobian.

    Uses algebraic p/T recovery from Q so the residual is fully
    autograd-differentiable — no iterative EOS inversion inside R(Q).

    Interface compatible with solver_a.step.

    cfg keys specific to this solver:
        use_autograd   : bool  (default True)  — use autograd vs FD Jacobian
        max_outer      : int   (default 3)     — outer Picard iterations
        max_newton     : int   (default 30)    — inner Newton iterations
        newton_tol     : float (default 1e-8)  — Newton convergence tolerance
        verbose_newton : bool  (default False)
    """
    if cfg is None:
        cfg = {}

    N = len(state['p'])
    p_n  = state['p'].copy()
    u_n  = state['u'].copy()
    T_n  = state['T'].copy()
    a1_n = np.clip(state['psi'].copy(), 1e-14, 1.0 - 1e-14)

    is_first     = aux.get('is_first_step', True)
    theta_old    = aux.get('theta_old',    None)
    u_bar_old    = aux.get('u_bar_old',    None)
    rho_star_old = aux.get('rho_star_old', None)
    if is_first:
        theta_old = u_bar_old = rho_star_old = None

    p_f, u_f, T_f, a1_f, info = _outer_picard_cons_ad(
        N, dx, dt, p_n, u_n, T_n, a1_n,
        ph1, ph2, bc_l, bc_r, cfg,
        theta_old=theta_old, u_bar_old=u_bar_old, rho_star_old=rho_star_old)

    # Final physical bounds
    p_f  = np.maximum(p_f, _P_FLOOR)
    T_f  = np.maximum(T_f, _T_FLOOR)
    a1_f = np.clip(a1_f, 1e-14, 1.0 - 1e-14)
    a2_f = 1.0 - a1_f

    # Reconstruct derived quantities
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)
    rho1_f = eos1.rho(p_f, T_f)
    rho2_f = eos2.rho(p_f, T_f)
    rho_f  = np.maximum(a1_f * rho1_f + a2_f * rho2_f, 1e-300)

    rE_f = (a1_f * eos1.e_vol(p_f, T_f)
            + a2_f * eos2.e_vol(p_f, T_f)
            + 0.5 * rho_f * u_f**2)
    E_total_f = rE_f / rho_f

    # MWI for new_aux
    theta_f, u_bar_f, d_hat_f, rho_star_f = compute_face_velocity_ad(
        u_f, p_f, rho_f, dx, dt, bc_l, bc_r,
        theta_old=theta_old, u_bar_old=u_bar_old, rho_star_old=rho_star_old)

    new_state = {
        'p':       p_f,
        'u':       u_f,
        'T':       T_f,
        'psi':     a1_f,
        'rho':     rho_f,
        'E_total': E_total_f,
        'u_face':  theta_f,
    }
    new_aux = {
        'is_first_step':  False,
        'bdf_order':      1,
        'rho_nm1':        rho_f,
        'rhoU_nm1':       rho_f * u_f,
        'E_nm1':          E_total_f,
        'rho_face_acid':  None,
        'dt_prev':        dt,
        'theta_old':      theta_f,
        'u_bar_old':      u_bar_f,
        'rho_star_old':   rho_star_f,
    }
    info_out = {
        'converged':    info['converged'],
        'outer_iters':  info['outer_iters'],
        'inner_iters':  info['inner_iters'],
        'picard_iters': info['picard_iters'],
        'residuals':    info['residuals'],
    }
    return new_state, new_aux, info_out
