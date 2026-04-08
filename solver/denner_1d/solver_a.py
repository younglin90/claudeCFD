# solver/denner_1d/solver_a.py
# Ref: Denner 2018 — Section 6 (barotropic loop), Eq. 20 (MWI), Eq. 25/29/30 (Newton)
#
# Newton linearisation + (p, u, h) variables + barotropic inner/outer loop.
# No under-relaxation.

import numpy as np

from .eos.base import (compute_mixture_props,
                       compute_specific_total_enthalpy,
                       recover_T_from_h)
from .flux.mwi import (acid_face_density,
                       harmonic_face_density,
                       mwi_face_coeff_denner)
from .boundary import apply_ghost, apply_ghost_velocity
from .assembly import assemble_newton_3N, solve_linear_system
from .vof_cn import vof_step


_P_FLOOR    = 1.0      # Pa
_T_FLOOR    = 1e-3     # K
_EPS_PSI    = 0.0   # no VOF clipping needed — Newton + ACID handles full density ratio

_MAX_OUTER  = 5
_MAX_INNER  = 10
_INNER_TOL  = 1e-6
_OUTER_TOL  = 1e-6


def _mixture_rho(p, T, psi, ph1, ph2):
    props = compute_mixture_props(p, np.zeros_like(p), T, psi, ph1, ph2)
    return props['rho']


def _mixture_zeta(p, T, psi, ph1, ph2):
    props = compute_mixture_props(p, np.zeros_like(p), T, psi, ph1, ph2)
    return props['zeta_v']


def _momentum_diagonal(rho_k, dx, dt):
    """
    Momentum diagonal e_P ≈ ρ_k/dt  (dominant temporal term).
    Used to build MWI Denner coefficient d_hat.
    """
    return rho_k / dt


def _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r, n_ghost=2):
    """
    MWI face velocity: ϑ_f = ū_f − d̂_f · (p_R − p_L)/dx
    Returns theta (N+1,).
    """
    N = len(u_k)
    ng = n_ghost
    u_ext = apply_ghost_velocity(u_k, bc_l, bc_r, ng)
    p_ext = apply_ghost(p_k, bc_l, bc_r, ng)
    theta = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        u_bar = 0.5 * (u_ext[iL] + u_ext[iR])
        dp    = (p_ext[iR] - p_ext[iL]) / dx
        theta[f] = u_bar - d_hat[f] * dp
    return theta


def step(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg=None):
    """
    One time step: Newton + barotropic inner/outer loop.

    Parameters / Returns identical to old solver_a.step interface.
    """
    if cfg is None:
        cfg = {}
    max_outer  = cfg.get('max_outer', _MAX_OUTER)
    max_inner  = cfg.get('max_inner', _MAX_INNER)
    inner_tol  = cfg.get('inner_tol', _INNER_TOL)
    outer_tol  = cfg.get('outer_tol', _OUTER_TOL)

    N = len(state['p'])
    p_n   = state['p'].copy()
    u_n   = state['u'].copy()
    T_n   = state['T'].copy()
    psi_n = state['psi'].copy()

    # ----------------------------------------------------------------
    # Step 1: VOF explicit update
    # ----------------------------------------------------------------
    psi_reg = np.clip(psi_n, _EPS_PSI, 1.0 - _EPS_PSI)
    psi_new, psi_face_vof, _ = vof_step(psi_reg, u_n, dx, dt, bc_l, bc_r)
    psi_new = np.clip(psi_new, _EPS_PSI, 1.0 - _EPS_PSI)

    # ----------------------------------------------------------------
    # Old-time quantities (ACID: use psi_new for temporal density)
    # ----------------------------------------------------------------
    props_old = compute_mixture_props(p_n, u_n, T_n, psi_new, ph1, ph2)
    rho_old = props_old['rho']
    h_old   = compute_specific_total_enthalpy(p_n, u_n, T_n, psi_new, ph1, ph2)

    # ----------------------------------------------------------------
    # Initialise Newton iterate from old state
    # ----------------------------------------------------------------
    p_k = p_n.copy()
    u_k = u_n.copy()
    T_k = T_n.copy()
    h_k = h_old.copy()

    info_outer = {'converged': False, 'outer_iters': 0, 'inner_iters': []}

    for outer in range(max_outer):
        rho_k   = _mixture_rho(p_k, T_k, psi_new, ph1, ph2)
        zeta_k  = _mixture_zeta(p_k, T_k, psi_new, ph1, ph2)

        # ACID face density and MWI coefficient
        props_k      = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)
        rho_face_acid = acid_face_density(
            props_k['rho'], props_k['c_mix'], psi_new, bc_l, bc_r)
        rho_star      = harmonic_face_density(rho_k, bc_l, bc_r)

        e_diag = _momentum_diagonal(rho_k, dx, dt)
        d_hat  = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
        theta_k = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r)

        # ---- Inner barotropic loop (T fixed, h frozen) ----
        inner_iters = 0
        for inner in range(max_inner):
            A, b_vec = assemble_newton_3N(
                N, dx, dt,
                rho_old, u_n, h_old, p_n,
                rho_k, u_k, h_k, p_k, T_k, psi_new,
                zeta_k,
                rho_face_acid, d_hat, theta_k,
                ph1, ph2, bc_l, bc_r,
                freeze_h=True,
            )

            x_k = np.concatenate([p_k, u_k, h_k])
            r   = b_vec - A.dot(x_k)

            p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
            u_ref = float(max(np.mean(np.abs(u_k)), 1e-6))
            h_ref = float(max(np.mean(np.abs(h_k)), 1.0))
            dx_vec = solve_linear_system(A, r, p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)

            dp = dx_vec[0:N]
            du = dx_vec[N:2*N]
            # h frozen in inner loop

            p_k = np.maximum(p_k + dp, _P_FLOOR)
            u_k = u_k + du

            # Update face velocity with new p, u
            theta_k = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r)

            res_p = np.max(np.abs(dp)) / p_ref
            res_u = np.max(np.abs(du)) / max(u_ref, 1e-6)
            inner_res = max(res_p, res_u)
            inner_iters += 1

            if inner_res < inner_tol:
                break

        info_outer['inner_iters'].append(inner_iters)

        # ---- Outer: full energy solve for h ----
        # Re-compute face quantities with updated p_k, u_k
        props_k      = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)
        rho_k        = props_k['rho']
        zeta_k       = props_k['zeta_v']
        rho_face_acid = acid_face_density(
            rho_k, props_k['c_mix'], psi_new, bc_l, bc_r)
        rho_star      = harmonic_face_density(rho_k, bc_l, bc_r)
        e_diag        = _momentum_diagonal(rho_k, dx, dt)
        d_hat         = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
        theta_k       = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r)

        A_full, b_full = assemble_newton_3N(
            N, dx, dt,
            rho_old, u_n, h_old, p_n,
            rho_k, u_k, h_k, p_k, T_k, psi_new,
            zeta_k,
            rho_face_acid, d_hat, theta_k,
            ph1, ph2, bc_l, bc_r,
            freeze_h=False,
        )

        x_k_full = np.concatenate([p_k, u_k, h_k])
        r_full   = b_full - A_full.dot(x_k_full)
        p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
        u_ref = float(max(np.mean(np.abs(u_k)), 1e-6))
        h_ref = float(max(np.mean(np.abs(h_k)), 1.0))
        dx_full = solve_linear_system(A_full, r_full, p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)

        dh = dx_full[2*N:3*N]
        h_k = h_k + dh

        # Update T from new h
        T_k_new = recover_T_from_h(h_k, u_k, p_k, psi_new, ph1, ph2, T_guess=T_k)
        T_k_new = np.maximum(T_k_new, _T_FLOOR)

        # Outer convergence: relative density change
        rho_k_new = _mixture_rho(p_k, T_k_new, psi_new, ph1, ph2)
        delta_rho  = np.max(np.abs(rho_k_new - rho_k)) / (np.mean(np.abs(rho_k)) + 1e-300)

        T_k = T_k_new
        rho_k = rho_k_new

        info_outer['outer_iters'] = outer + 1

        if delta_rho < outer_tol:
            info_outer['converged'] = True
            break

    # ----------------------------------------------------------------
    # Build output state
    # ----------------------------------------------------------------
    props_new = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)

    # Face velocity for diagnostics
    p_ext_new = apply_ghost(p_k, bc_l, bc_r)
    u_ext_new = apply_ghost_velocity(u_k, bc_l, bc_r)
    u_face_new = np.array([
        0.5 * (u_ext_new[2 + f - 1] + u_ext_new[2 + f])
        - d_hat[f] * (p_ext_new[2 + f] - p_ext_new[2 + f - 1]) / dx
        for f in range(N + 1)
    ])

    new_state = {
        'p':       p_k,
        'u':       u_k,
        'T':       T_k,
        'psi':     psi_new,
        'rho':     props_new['rho'],
        'E_total': props_new['E_total'],
        'u_face':  u_face_new,
    }

    new_aux = {
        'is_first_step': False,
        'bdf_order':     1,
        'rho_nm1':       rho_old,
        'rhoU_nm1':      rho_old * u_n,
        'E_nm1':         props_old['E_total'],
        'rho_face_acid': rho_face_acid,
        'dt_prev':       dt,
    }

    info = {
        'converged':    info_outer['converged'],
        'outer_iters':  info_outer['outer_iters'],
        'inner_iters':  info_outer['inner_iters'],
        # Picard-compatible alias for print_step_info
        'picard_iters': info_outer['outer_iters'],
        'residuals':    [],
    }

    return new_state, new_aux, info
