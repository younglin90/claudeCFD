# solver/denner_1d/solver_5eq.py
# Ref: breezy-wishing-wall.md — 5-equation conservative Newton solver
#
# One time step of the 5-equation conservative Newton solver.
#
# Conservative variables: Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}
# Newton iteration: J(Q_k)·ΔQ = -R(Q_k),  Q_{k+1} = Q_k + ω·ΔQ
#
# Key advantage over 4-equation primitive Newton:
#   Temporal Jacobian = 1/dt·I → diagonal-dominant → no α/ζ ill-conditioning.

import numpy as np

from .eos.eos_class import create_eos
from .flux.mwi import harmonic_face_density, mwi_face_coeff_denner
from .boundary import apply_ghost, apply_ghost_velocity
from .assembly import solve_linear_system
from .timestepping import compute_dt_acoustic
from .assembly_5eq import (
    split5, compute_K, invert_5eq, residual_5eq, assemble_jacobian_5eq,
    residual_5eq_prim, assemble_jacobian_5eq_prim,
)


_P_FLOOR = 1.0    # Pa
_T_FLOOR = 1e-3   # K


def _momentum_diag(rho, dt):
    """Momentum equation diagonal: ρ/dt."""
    return rho / dt


def _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r, n_ghost=2,
                            theta_old=None, u_bar_old=None,
                            rho_star_old=None, dt=None):
    """MWI face velocity: θ_f = ū_f - d̂_f·(p_R - p_L)/dx
    + optional transient correction (Denner 2018 Eq. 20).

    Returns theta (N+1,), u_bar (N+1,).
    """
    from .boundary import apply_ghost, apply_ghost_velocity
    N = len(u_k)
    ng = n_ghost
    u_ext = apply_ghost_velocity(u_k, bc_l, bc_r, ng)
    p_ext = apply_ghost(p_k, bc_l, bc_r, ng)
    theta = np.empty(N + 1)
    u_bar = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        ub = 0.5 * (u_ext[iL] + u_ext[iR])
        dp = (p_ext[iR] - p_ext[iL]) / dx
        theta[f] = ub - d_hat[f] * dp
        u_bar[f] = ub
    # Transient correction (Denner 2018 Eq. 20)
    if (theta_old is not None and u_bar_old is not None
            and rho_star_old is not None and dt is not None):
        theta += d_hat * (rho_star_old / dt) * (theta_old - u_bar_old)
    return theta, u_bar


def _pack_Q(a1, p, T, u, eos1, eos2):
    """Pack primitive state into conservative vector Q (5N,)."""
    a2 = np.maximum(1.0 - a1, 0.0)
    rho1 = eos1.rho(p, T)
    rho2 = eos2.rho(p, T)
    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho  = a1r1 + a2r2
    ru   = rho * u
    # ρE = α₁ρ₁e₁ + α₂ρ₂e₂ + ½ρu²
    rE   = (a1 * eos1.e_vol(p, T)
            + a2 * eos2.e_vol(p, T)
            + 0.5 * rho * u**2)
    return np.concatenate([a1r1, a2r2, ru, rE, a1])


def _newton_5eq_prim(N, dx, dt,
                     p_n, u_n, T_n, a1_n,
                     ph1, ph2, bc_l, bc_r,
                     cfg,
                     theta_old=None, u_bar_old=None, rho_star_old=None):
    """5-equation primitive-variable Newton iteration.

    Unknowns: W = {p, u, T, α₁}  (4N total)
    Newton step: J(W_k)·ΔW = -R(W_k),  W_{k+1} = W_k + ω·ΔW

    EOS is used *forward* only — no EOS inversion required.

    Parameters
    ----------
    N, dx, dt    : grid/time
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    ph1, ph2     : EOS parameter dicts
    bc_l, bc_r   : str
    cfg          : dict  solver configuration
    theta_old, u_bar_old, rho_star_old : MWI transient correction data

    Returns
    -------
    p_k, u_k, T_k, a1_k : ndarray (N,)  converged primitive state
    info                 : dict
    """
    max_newton = cfg.get('max_newton', 50)
    newton_tol = cfg.get('newton_tol', 1e-6)

    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # Initial iterate = old-time primitives
    p_k  = p_n.copy()
    u_k  = u_n.copy()
    T_k  = T_n.copy()
    a1_k = a1_n.copy()

    info = {'converged': False, 'outer_iters': 0, 'residuals': []}

    for niter in range(max_newton):
        # 1. EOS forward — no inversion!
        a1_k = np.clip(a1_k, 1e-10, 1.0 - 1e-10)
        a2_k = 1.0 - a1_k
        rho1_k = eos1.rho(p_k, T_k)
        rho2_k = eos2.rho(p_k, T_k)
        rho_k  = a1_k * rho1_k + a2_k * rho2_k
        rho_k  = np.maximum(rho_k, 1e-300)

        # 2. K coefficient (Wood's sound speed)
        K_k = compute_K(a1_k, p_k, T_k, eos1, eos2)

        # 3. MWI face velocity
        rho_star_k = harmonic_face_density(rho_k, bc_l, bc_r)
        e_diag_k   = _momentum_diag(rho_k, dt)
        d_hat_k    = mwi_face_coeff_denner(e_diag_k, rho_star_k, dx, dt, bc_l, bc_r)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat_k, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)

        # 4. Residual R(W_k)  [4N]
        R_k = residual_5eq_prim(
            p_k, u_k, T_k, a1_k,
            p_n, u_n, T_n, a1_n,
            N, dx, dt, ph1, ph2, bc_l, bc_r,
            theta_k, K_k)
        R_norm = float(np.linalg.norm(R_k))

        # 5. Jacobian J(W_k)  [4N × 4N]
        J_k = assemble_jacobian_5eq_prim(
            p_k, u_k, T_k, a1_k, K_k, theta_k,
            N, dx, dt, ph1, ph2, bc_l, bc_r,
            d_hat_k=d_hat_k)

        # 6. Solve J·ΔW = -R
        p_ref  = float(max(float(np.mean(np.abs(p_k))), 1.0))
        u_ref  = float(max(float(np.mean(np.abs(u_k))), 1e-6))
        T_ref  = float(max(float(np.mean(np.abs(T_k))), 1.0))

        try:
            dW = solve_linear_system(J_k, -R_k,
                                     p_ref=p_ref, u_ref=u_ref, h_ref=T_ref,
                                     phi_ref=1.0, n_blocks=4)
        except Exception:
            import scipy.sparse.linalg as spla
            try:
                dW = spla.spsolve(J_k.tocsc(), -R_k)
                if not np.all(np.isfinite(dW)):
                    dW = np.zeros(4 * N)
            except Exception:
                dW = np.zeros(4 * N)

        if not np.all(np.isfinite(dW)):
            dW = np.zeros(4 * N)

        dp  = dW[0*N:1*N]
        du  = dW[1*N:2*N]
        dT  = dW[2*N:3*N]
        da1 = dW[3*N:4*N]

        # 7. Backtracking line search (8 halvings, NO EOS inversion!)
        omega = 1.0
        _ls_improved = False
        R_trial_norm = R_norm
        p_tr = p_k.copy(); u_tr = u_k.copy()
        T_tr = T_k.copy(); a1_tr = a1_k.copy()

        for ls in range(8):
            p_tr  = np.maximum(p_k  + omega * dp,  _P_FLOOR)
            u_tr  = u_k + omega * du
            T_tr  = np.maximum(T_k  + omega * dT,  _T_FLOOR)
            a1_tr = np.clip(a1_k + omega * da1, 1e-10, 1.0 - 1e-10)

            if not (np.all(np.isfinite(p_tr)) and np.all(np.isfinite(u_tr))
                    and np.all(np.isfinite(T_tr)) and np.all(np.isfinite(a1_tr))):
                omega *= 0.5
                continue

            # Recompute MWI for trial state (no inversion needed)
            a2_tr = 1.0 - a1_tr
            rho_tr = np.maximum(
                a1_tr * eos1.rho(p_tr, T_tr) + a2_tr * eos2.rho(p_tr, T_tr),
                1e-300)
            rho_star_tr = harmonic_face_density(rho_tr, bc_l, bc_r)
            e_diag_tr   = _momentum_diag(rho_tr, dt)
            d_hat_tr    = mwi_face_coeff_denner(e_diag_tr, rho_star_tr, dx, dt, bc_l, bc_r)
            theta_tr, _ = _compute_face_velocity(
                u_tr, p_tr, d_hat_tr, dx, bc_l, bc_r,
                theta_old=theta_old, u_bar_old=u_bar_old,
                rho_star_old=rho_star_old, dt=dt)
            K_tr = compute_K(a1_tr, p_tr, T_tr, eos1, eos2)

            R_trial = residual_5eq_prim(
                p_tr, u_tr, T_tr, a1_tr,
                p_n, u_n, T_n, a1_n,
                N, dx, dt, ph1, ph2, bc_l, bc_r,
                theta_tr, K_tr)
            R_trial_norm = float(np.linalg.norm(R_trial))

            if np.isfinite(R_trial_norm) and R_trial_norm < R_norm:
                _ls_improved = True
                break
            omega *= 0.5

        if _ls_improved:
            p_k, u_k, T_k, a1_k = p_tr, u_tr, T_tr, a1_tr
        else:
            omega = 0.0

        # 8. Convergence check (relative update norm)
        res_delta = max(
            np.max(np.abs(omega * dp)  / (np.abs(p_k)  + 1.0)),
            np.max(np.abs(omega * du)  / (max(float(np.max(np.abs(u_k))), 1e-6))),
            np.max(np.abs(omega * dT)  / (np.abs(T_k)  + 1.0)),
            np.max(np.abs(omega * da1)))

        info['residuals'].append(R_norm)
        info['outer_iters'] = niter + 1

        if niter < 5 or niter % 10 == 0:
            print(f"    5eq-prim-Newton {niter:3d}: |R|={R_norm:.3e}  "
                  f"|ΔW|_rel={res_delta:.3e}  ω={omega:.3e}")

        if res_delta < newton_tol:
            info['converged'] = True
            break

    return p_k, u_k, T_k, a1_k, info


def _newton_5eq(N, dx, dt,
                Q_n, ph1, ph2, bc_l, bc_r,
                p_n, T_n, u_n,
                cfg,
                theta_old=None, u_bar_old=None, rho_star_old=None):
    """5-equation conservative Newton iteration (legacy, kept for reference).

    J(Q_k)·ΔQ = -R(Q_k),  Q_{k+1} = Q_k + ω·ΔQ

    Parameters
    ----------
    N, dx, dt : grid/time
    Q_n       : ndarray (5N,)  old-time conservative state
    ph1, ph2  : EOS parameter dicts
    bc_l, bc_r: str
    p_n, T_n, u_n : ndarray (N,)  old-time primitives (for initial guess)
    cfg       : dict  solver configuration
    theta_old, u_bar_old, rho_star_old : MWI transient correction data

    Returns
    -------
    Q_k   : ndarray (5N,)  converged conservative state
    info  : dict
    """
    max_newton = cfg.get('max_newton', 50)
    newton_tol = cfg.get('newton_tol', 1e-6)

    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # Initial iterate = old-time state
    Q_k = Q_n.copy()

    # Initial primitive guess = old-time
    p_guess = p_n.copy()
    T_guess = T_n.copy()

    info = {'converged': False, 'outer_iters': 0, 'residuals': []}

    for niter in range(max_newton):
        # 1. EOS inversion: Q_k → (p_k, T_k, u_k)
        try:
            p_k, T_k, u_k = invert_5eq(Q_k, N, ph1, ph2, p_guess, T_guess)
        except Exception:
            p_k, T_k, u_k = p_guess.copy(), T_guess.copy(), (Q_k[2*N:3*N] / np.maximum(Q_k[:N] + Q_k[N:2*N], 1e-300))
        p_k = np.maximum(p_k, _P_FLOOR)
        T_k = np.maximum(T_k, _T_FLOOR)

        # Update guess for next iteration
        p_guess = p_k.copy()
        T_guess = T_k.copy()

        # 2. K coefficient (Wood's sound speed)
        a1_k = np.clip(Q_k[4*N:5*N], 0.0, 1.0)
        K_k = compute_K(a1_k, p_k, T_k, eos1, eos2)

        # 3. MWI face velocity
        rho_k = Q_k[:N] + Q_k[N:2*N]
        rho_k = np.maximum(rho_k, 1e-300)
        rho_star_k = harmonic_face_density(rho_k, bc_l, bc_r)
        e_diag_k   = _momentum_diag(rho_k, dt)
        d_hat_k    = mwi_face_coeff_denner(e_diag_k, rho_star_k, dx, dt, bc_l, bc_r)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat_k, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)

        # 4. Residual R(Q_k)
        R_k = residual_5eq(Q_k, Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r, theta_k, K_k, p_k, T_k, u_k)
        R_norm = float(np.linalg.norm(R_k))

        # 5. Jacobian J(Q_k)
        J_k = assemble_jacobian_5eq(Q_k, p_k, T_k, u_k, K_k, theta_k, N, dx, dt, ph1, ph2, bc_l, bc_r)

        # 6. Solve J·ΔQ = -R
        p_ref = float(max(float(np.mean(np.abs(p_k))), 1.0))
        u_ref = float(max(float(np.mean(np.abs(u_k))), 1e-6))
        rE_ref = float(max(float(np.mean(np.abs(Q_k[3*N:4*N]))), 1.0))

        try:
            dQ = solve_linear_system(J_k, -R_k,
                                     p_ref=p_ref, u_ref=u_ref, h_ref=rE_ref,
                                     phi_ref=1.0, n_blocks=5)
        except Exception:
            import scipy.sparse.linalg as spla
            try:
                dQ = spla.spsolve(J_k.tocsc(), -R_k)
                if not np.all(np.isfinite(dQ)):
                    dQ = np.zeros(5 * N)
            except Exception:
                dQ = np.zeros(5 * N)

        if not np.all(np.isfinite(dQ)):
            dQ = np.zeros(5 * N)

        # 7. Backtracking line search (8 halvings)
        omega = 1.0
        _ls_improved = False
        Q_trial = Q_k.copy()
        R_trial_norm = R_norm
        p_guess_ls = p_k.copy()
        T_guess_ls = T_k.copy()
        for ls in range(8):
            Q_trial = Q_k + omega * dQ
            Q_trial[4*N:5*N] = np.clip(Q_trial[4*N:5*N], 1e-10, 1.0 - 1e-10)
            Q_trial[0*N:1*N] = np.maximum(Q_trial[0*N:1*N], 1e-20)
            Q_trial[1*N:2*N] = np.maximum(Q_trial[1*N:2*N], 1e-20)

            rho_tr_check = Q_trial[:N] + Q_trial[N:2*N]
            rE_tr_check  = Q_trial[3*N:4*N]
            ru_tr_check  = Q_trial[2*N:3*N]
            rho_e_tr_check = rE_tr_check - 0.5 * ru_tr_check**2 / np.maximum(rho_tr_check, 1e-300)
            if (np.any(~np.isfinite(Q_trial))
                    or np.any(rho_tr_check <= 0.0)
                    or np.any(rho_e_tr_check < 0.0)):
                omega *= 0.5
                p_guess_ls = p_k.copy()
                T_guess_ls = T_k.copy()
                continue

            try:
                p_tr, T_tr, u_tr = invert_5eq(Q_trial, N, ph1, ph2, p_guess_ls, T_guess_ls)
                p_tr = np.maximum(p_tr, _P_FLOOR)
                T_tr = np.maximum(T_tr, _T_FLOOR)
                if not (np.all(np.isfinite(p_tr)) and np.all(np.isfinite(T_tr))
                        and np.all(p_tr > 0.0) and np.all(T_tr > 0.0)):
                    omega *= 0.5
                    p_guess_ls = p_k.copy()
                    T_guess_ls = T_k.copy()
                    continue
                p_guess_ls = p_tr.copy()
                T_guess_ls = T_tr.copy()
                a1_tr = np.clip(Q_trial[4*N:5*N], 0.0, 1.0)
                K_tr = compute_K(a1_tr, p_tr, T_tr, eos1, eos2)
                rho_tr = np.maximum(Q_trial[:N] + Q_trial[N:2*N], 1e-300)
                rho_star_tr = harmonic_face_density(rho_tr, bc_l, bc_r)
                e_diag_tr = _momentum_diag(rho_tr, dt)
                d_hat_tr  = mwi_face_coeff_denner(e_diag_tr, rho_star_tr, dx, dt, bc_l, bc_r)
                theta_tr, _ = _compute_face_velocity(
                    u_tr, p_tr, d_hat_tr, dx, bc_l, bc_r,
                    theta_old=theta_old, u_bar_old=u_bar_old,
                    rho_star_old=rho_star_old, dt=dt)
                R_trial = residual_5eq(Q_trial, Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                                       theta_tr, K_tr, p_tr, T_tr, u_tr)
                R_trial_norm = float(np.linalg.norm(R_trial))
                if np.isfinite(R_trial_norm) and R_trial_norm < R_norm:
                    _ls_improved = True
                    break
            except Exception:
                p_guess_ls = p_k.copy()
                T_guess_ls = T_k.copy()
            omega *= 0.5

        if _ls_improved:
            Q_k = Q_trial
        else:
            omega = 0.0

        # 8. Convergence check (relative update norm)
        dQ_rel = np.max(np.abs(omega * dQ) / (np.abs(Q_k) + 1e-300))
        info['residuals'].append(R_norm)
        info['outer_iters'] = niter + 1

        if niter < 5 or niter % 10 == 0:
            print(f"    5eq-Newton {niter:3d}: |R|={R_norm:.3e}  "
                  f"|ΔQ|_rel={dQ_rel:.3e}  ω={omega:.3e}")

        if dQ_rel < newton_tol:
            info['converged'] = True
            break

    return Q_k, info


def step_5eq(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg):
    """One time step of the 5-equation primitive-variable Newton solver.

    Parameters / Returns compatible with solver_a.step interface.

    Unknowns: W = {p, u, T, α₁} — EOS used forward only, no inversion.

    Parameters
    ----------
    state  : dict with 'p', 'u', 'T', 'psi', 'rho', 'E_total', 'u_face'
    ph1,ph2: EOS parameter dicts
    dx,dt  : float
    bc_l,bc_r : str
    aux    : dict with MWI transient data
    cfg    : dict with solver configuration

    Returns
    -------
    new_state, new_aux, info
    """
    if cfg is None:
        cfg = {}

    N = len(state['p'])

    p_n  = state['p'].copy()
    u_n  = state['u'].copy()
    T_n  = state['T'].copy()
    a1_n = state['psi'].copy()   # volume fraction = α₁

    # Transient correction from previous step
    is_first     = aux.get('is_first_step', True)
    theta_old    = aux.get('theta_old', None)
    u_bar_old    = aux.get('u_bar_old', None)
    rho_star_old = aux.get('rho_star_old', None)
    if is_first:
        theta_old = u_bar_old = rho_star_old = None

    # Primitive-variable Newton (no EOS inversion!)
    p_f, u_f, T_f, a1_f, info = _newton_5eq_prim(
        N, dx, dt,
        p_n, u_n, T_n, a1_n,
        ph1, ph2, bc_l, bc_r, cfg,
        theta_old=theta_old, u_bar_old=u_bar_old, rho_star_old=rho_star_old)

    # Final physical bounds
    p_f  = np.maximum(p_f,  _P_FLOOR)
    T_f  = np.maximum(T_f,  _T_FLOOR)
    a1_f = np.clip(a1_f, 1e-10, 1.0 - 1e-10)
    a2_f = 1.0 - a1_f

    # Reconstruct mixture density and total energy from final primitives (forward EOS)
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)
    rho1_f = eos1.rho(p_f, T_f)
    rho2_f = eos2.rho(p_f, T_f)
    rho_f  = a1_f * rho1_f + a2_f * rho2_f
    rho_f  = np.maximum(rho_f, 1e-300)

    rE_f = (a1_f * eos1.e_vol(p_f, T_f)
            + a2_f * eos2.e_vol(p_f, T_f)
            + 0.5 * rho_f * u_f**2)
    E_total_f = rE_f / rho_f

    # MWI for new_aux
    rho_star_f = harmonic_face_density(rho_f, bc_l, bc_r)
    e_diag_f   = _momentum_diag(rho_f, dt)
    d_hat_f    = mwi_face_coeff_denner(e_diag_f, rho_star_f, dx, dt, bc_l, bc_r)
    theta_f, u_bar_f = _compute_face_velocity(
        u_f, p_f, d_hat_f, dx, bc_l, bc_r,
        theta_old=theta_old, u_bar_old=u_bar_old,
        rho_star_old=rho_star_old, dt=dt)

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
        'inner_iters':  [],
        'picard_iters': info['outer_iters'],
        'residuals':    info['residuals'],
    }

    return new_state, new_aux, info_out
