# solver/denner_1d/solver_a.py
# Ref: Denner 2018 — Section 6 (barotropic loop), Eq. 20 (MWI), Eq. 25/29/30 (Newton)
#
# Newton linearisation + (p, u, h) variables + barotropic inner/outer loop.
# No under-relaxation.

import numpy as np

from .eos.base import (compute_mixture_props,
                       compute_mixture_props_Y,
                       compute_mixture_props_Ns,
                       compute_specific_total_enthalpy,
                       compute_specific_total_enthalpy_Y,
                       compute_specific_total_enthalpy_Ns,
                       recover_T_from_h,
                       recover_T_from_h_Y)
from .flux.mwi import (acid_face_density,
                       harmonic_face_density,
                       mwi_face_coeff_denner)
from .boundary import apply_ghost, apply_ghost_velocity
from .assembly import assemble_newton_3N, assemble_newton_4N, assemble_newton_Ns, solve_linear_system
from .vof_cn import vof_step, mass_fraction_step
from .interface.cicsam import cicsam_face_beta
from .vof_cn import compute_compression_coefficients


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


def _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r, n_ghost=2,
                           theta_old=None, u_bar_old=None,
                           rho_star_old=None, dt=None):
    """
    MWI face velocity: ϑ_f = ū_f − d̂_f · (p_R − p_L)/dx
    + Denner 2018 Eq. 20 transient correction:
      ϑ_f += d̂_f · (ρ★_old / dt) · (ϑ_old − ū_old)

    Returns theta (N+1,), u_bar (N+1,).
    """
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
    # Transient correction (Denner 2018 Eq. 20, last term)
    if (theta_old is not None and u_bar_old is not None
            and rho_star_old is not None and dt is not None):
        theta += d_hat * (rho_star_old / dt) * (theta_old - u_bar_old)
    return theta, u_bar


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
    variable_set = cfg.get('variable_set', 'puh')  # 'puh' or 'puT'
    vof_type     = cfg.get('vof_type', 'volume')   # 'volume' or 'mass'
    use_K        = cfg.get('use_K', False)          # Denner 2018 compressibility K in VOF
    use_compress = cfg.get('use_compress', False)   # anti-diffusion compression term
    coupled      = cfg.get('coupled', False)        # fully coupled 4N system
    coupled_Ns   = cfg.get('coupled_Ns', False)     # fully coupled (2+N_s)N system

    N = len(state['p'])
    p_n   = state['p'].copy()
    u_n   = state['u'].copy()
    T_n   = state['T'].copy()
    psi_n = state['psi'].copy()

    # ----------------------------------------------------------------
    # Transient correction data from previous step (Mod 2)
    # ----------------------------------------------------------------
    is_first      = aux.get('is_first_step', True)
    theta_old     = aux.get('theta_old', None)
    u_bar_old     = aux.get('u_bar_old', None)
    rho_star_old  = aux.get('rho_star_old', None)
    if is_first:
        theta_old = u_bar_old = rho_star_old = None

    # ----------------------------------------------------------------
    # COUPLED_Ns PATH: fully coupled (2+N_s)N system (p, u, T, φ₀,..,φ_{N_s-2})
    # ----------------------------------------------------------------
    if coupled_Ns:
        from .eos.eos_class import create_eos
        phases    = cfg.get('phases', [ph1, ph2])
        N_s       = len(phases)
        use_mass  = (vof_type == 'mass')
        mixing    = 'mass' if use_mass else 'volume'
        max_newton = cfg.get('max_newton', 20)
        newton_tol = cfg.get('newton_tol', 1e-6)

        # phi_n: (N_s-1, N)
        if 'phi_arr' in state:
            phi_n = state['phi_arr'].copy()  # (N_s-1, N)
        else:
            # backward compat: 2-species
            if use_mass:
                from .eos.base import compute_phase_props
                from .vof_cn import psi_to_Y
                rho1_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), phases[0])['rho'])
                rho2_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), phases[1])['rho'])
                phi_n = np.atleast_2d(psi_to_Y(psi_n, rho1_s, rho2_s))
            else:
                phi_n = np.atleast_2d(psi_n.copy())

        phi_old = phi_n.copy()  # (N_s-1, N)

        # Old-time properties
        props_old = compute_mixture_props_Ns(p_n, u_n, T_n, phi_old, phases, mixing=mixing)
        rho_old   = props_old['rho']
        h_old     = compute_specific_total_enthalpy_Ns(p_n, u_n, T_n, phi_old, phases,
                                                        mixing=mixing)

        # Initialise iterate
        p_k   = p_n.copy()
        u_k   = u_n.copy()
        T_k   = T_n.copy()
        phi_k = phi_old.copy()  # (N_s-1, N)

        # Face velocity for CICSAM (frozen from old-time velocity)
        u_ext_vof = apply_ghost_velocity(u_n, bc_l, bc_r, 2)
        u_face_vof = np.array([0.5 * (u_ext_vof[2 + f - 1] + u_ext_vof[2 + f])
                                for f in range(N + 1)])

        info_outer = {'converged': False, 'outer_iters': 0, 'inner_iters': []}

        for niter in range(max_newton):
            props_k       = compute_mixture_props_Ns(p_k, u_k, T_k, phi_k, phases, mixing=mixing)
            rho_k_arr     = props_k['rho']
            zeta_k_vals   = props_k['zeta_v']
            phi_T_k_vals  = props_k['phi_v']
            alpha_k_list  = props_k['Delta_rho']        # list of N_s-1 arrays
            d_rho_h_dphi_list = props_k['d_rho_h_dphi']  # list of N_s-1 arrays

            h_k = compute_specific_total_enthalpy_Ns(p_k, u_k, T_k, phi_k, phases,
                                                      mixing=mixing)

            # Use first species fraction for ACID (2-species compat; general case uses phi_i_full)
            psi_for_acid = phi_k[0]
            rho_face_acid = acid_face_density(rho_k_arr, props_k['c_mix'], psi_for_acid,
                                               bc_l, bc_r)
            rho_star   = harmonic_face_density(rho_k_arr, bc_l, bc_r)
            e_diag     = _momentum_diagonal(rho_k_arr, dx, dt)
            d_hat      = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
            theta_k_face, u_bar_k = _compute_face_velocity(
                u_k, p_k, d_hat, dx, bc_l, bc_r,
                theta_old=theta_old, u_bar_old=u_bar_old,
                rho_star_old=rho_star_old, dt=dt)

            # Per-species CICSAM beta
            beta_k_list = []
            for k in range(N_s - 1):
                phi_ext_k = apply_ghost(phi_k[k], bc_l, bc_r, 2)
                beta_k_list.append(cicsam_face_beta(phi_ext_k, u_face_vof, dt, dx, n_ghost=2))

            # Per-species compression
            C_k_list = n_hat_list = None
            if use_compress:
                C_k_list = []
                n_hat_list = []
                for k in range(N_s - 1):
                    ck, nh, _ = compute_compression_coefficients(
                        phi_k[k], u_face_vof, dx, dt, bc_l, bc_r, n_ghost=2)
                    C_k_list.append(ck)
                    n_hat_list.append(nh)

            A_mat, b_vec = assemble_newton_Ns(
                N, dx, dt, N_s,
                rho_old, u_n, h_old, p_n, phi_old,
                rho_k_arr, u_k, h_k, p_k, T_k, phi_k,
                zeta_k_vals, phi_T_k_vals,
                alpha_k_list, d_rho_h_dphi_list,
                rho_face_acid, d_hat, theta_k_face,
                beta_k_list, phases, bc_l, bc_r,
                mixing_type=mixing,
                use_compress=use_compress,
                C_k_arr=C_k_list, n_hat_k_arr=n_hat_list, u_face_vof=u_face_vof)

            n_blocks = 2 + N_s
            x_k = np.concatenate([p_k, u_k, T_k] + [phi_k[k] for k in range(N_s - 1)])
            r_vec = b_vec - A_mat.dot(x_k)

            p_ref_ns = float(max(np.mean(np.abs(p_k)), 1.0))
            u_ref_ns = float(max(np.mean(np.abs(u_k)), 1e-6))
            T_ref_ns = float(max(np.mean(np.abs(T_k)), 1.0))
            dx_vec = solve_linear_system(A_mat, r_vec,
                                         p_ref=p_ref_ns, u_ref=u_ref_ns, h_ref=T_ref_ns,
                                         phi_ref=1.0, n_blocks=n_blocks)

            dp   = dx_vec[0:N]
            du   = dx_vec[N:2*N]
            dT   = dx_vec[2*N:3*N]
            dphi = [dx_vec[(3 + k) * N:(4 + k) * N] for k in range(N_s - 1)]

            omega = cfg.get('coupled_omega', 0.3)
            p_k = np.maximum(p_k + omega * dp, _P_FLOOR)
            u_k = u_k + omega * du
            T_k = np.maximum(T_k + omega * dT, _T_FLOOR)
            for k in range(N_s - 1):
                phi_k[k] = np.clip(phi_k[k] + omega * dphi[k], 0.0, 1.0)

            # Convergence check
            res = max(
                np.max(np.abs(omega * dp)) / p_ref_ns,
                np.max(np.abs(omega * du)) / max(u_ref_ns, 1e-6),
                np.max(np.abs(dT)) / max(float(np.mean(np.abs(T_k))), 1.0),
                max(np.max(np.abs(dphi[k])) for k in range(N_s - 1)),
            )
            info_outer['outer_iters'] = niter + 1
            if res < newton_tol:
                info_outer['converged'] = True
                break

        # Convert to psi for output
        if N_s == 2:
            if use_mass:
                from .eos.base import compute_phase_props
                from .vof_cn import Y_to_psi
                rho1_s = float(compute_phase_props(np.mean(p_k), np.mean(T_k), phases[0])['rho'])
                rho2_s = float(compute_phase_props(np.mean(p_k), np.mean(T_k), phases[1])['rho'])
                psi_new = Y_to_psi(phi_k[0], rho1_s, rho2_s)
            else:
                psi_new = phi_k[0].copy()
        else:
            psi_new = phi_k[0].copy()

        props_new = compute_mixture_props_Ns(p_k, u_k, T_k, phi_k, phases, mixing=mixing)
        u_face_new, u_bar_k = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r,
                                                      theta_old=theta_old,
                                                      u_bar_old=u_bar_old,
                                                      rho_star_old=rho_star_old,
                                                      dt=dt)

        new_state = {
            'p':       p_k,
            'u':       u_k,
            'T':       T_k,
            'psi':     psi_new,
            'rho':     props_new['rho'],
            'E_total': props_new['E_total'],
            'u_face':  u_face_new,
            'phi_arr': phi_k,  # full species array (N_s-1, N)
        }
        new_aux = {
            'is_first_step':  False,
            'bdf_order':      1,
            'rho_nm1':        rho_old,
            'rhoU_nm1':       rho_old * u_n,
            'E_nm1':          props_old['E_total'],
            'rho_face_acid':  rho_face_acid,
            'dt_prev':        dt,
            'theta_old':      u_face_new,
            'u_bar_old':      u_bar_k,
            'rho_star_old':   rho_star,
        }
        info = {
            'converged':    info_outer['converged'],
            'outer_iters':  info_outer['outer_iters'],
            'inner_iters':  [],
            'picard_iters': info_outer['outer_iters'],
            'residuals':    [],
        }
        return new_state, new_aux, info

    # ----------------------------------------------------------------
    # COUPLED PATH: fully coupled 4N×4N Newton system (p, u, T, φ)
    # ----------------------------------------------------------------
    if coupled:
        use_mass = (vof_type == 'mass')
        mixing_type = 'mass' if use_mass else 'volume'
        max_newton = cfg.get('max_newton', 20)
        newton_tol = cfg.get('newton_tol', 1e-6)

        # Convert initial psi to φ (volume or mass fraction)
        from .eos.base import compute_phase_props
        rho1_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), ph1)['rho'])
        rho2_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), ph2)['rho'])

        if use_mass:
            from .vof_cn import psi_to_Y
            phi_n = psi_to_Y(psi_n, rho1_s, rho2_s)
        else:
            phi_n = np.clip(psi_n.copy(), _EPS_PSI, 1.0 - _EPS_PSI)

        phi_old = phi_n.copy()

        def _mix_props_with_phi(p, u, T, phi):
            if use_mass:
                return compute_mixture_props_Y(p, u, T, phi, ph1, ph2)
            return compute_mixture_props(p, u, T, phi, ph1, ph2)

        def _mix_h_with_phi(p, u, T, phi):
            if use_mass:
                return compute_specific_total_enthalpy_Y(p, u, T, phi, ph1, ph2)
            return compute_specific_total_enthalpy(p, u, T, phi, ph1, ph2)

        # Face velocity for CICSAM (frozen from old-time velocity)
        u_ext_vof = apply_ghost_velocity(u_n, bc_l, bc_r, 2)
        u_face_vof = np.array([0.5 * (u_ext_vof[2 + f - 1] + u_ext_vof[2 + f])
                                for f in range(N + 1)])

        # --- Step A: Implicit VOF update (replaces explicit CICSAM) ---
        phi_ext_arr = apply_ghost(phi_n, bc_l, bc_r, 2)
        beta_k_face = cicsam_face_beta(phi_ext_arr, u_face_vof, dt, dx, n_ghost=2)

        import scipy.sparse as sp_vof
        A_vof = sp_vof.lil_matrix((N, N), dtype=float)
        b_vof = np.zeros(N)
        is_per = (bc_l == 'periodic')

        for i in range(N):
            fR = i + 1; fL = i
            iL_v = (N - 1 if is_per else 0) if (fL - 1) < 0 else (fL - 1)
            iR_v = (0 if is_per else N - 1) if fR >= N else fR
            tR_v = u_face_vof[fR]; tL_v = u_face_vof[fL]
            bR = float(beta_k_face[fR]); bL = float(beta_k_face[fL])

            A_vof[i, i] += 1.0 / dt
            b_vof[i]    += phi_n[i] / dt

            if tR_v >= 0:
                A_vof[i, i]    += (1.0 - bR) * tR_v / dx
                A_vof[i, iR_v] += bR * tR_v / dx
            else:
                A_vof[i, iR_v] += (1.0 - bR) * tR_v / dx
                A_vof[i, i]    += bR * tR_v / dx
            if tL_v >= 0:
                A_vof[i, iL_v] -= (1.0 - bL) * tL_v / dx
                A_vof[i, i]    -= bL * tL_v / dx
            else:
                A_vof[i, i]    -= (1.0 - bL) * tL_v / dx
                A_vof[i, iL_v] -= bL * tL_v / dx
            # Source: -ψ·∇·θ (volume fraction only; mass fraction has no source)
            if not use_mass:
                A_vof[i, i] -= (tR_v - tL_v) / dx

        import scipy.sparse.linalg as spla_vof
        phi_new = spla_vof.spsolve(A_vof.tocsr(), b_vof)
        if np.all(np.isfinite(phi_new)):
            phi_k = np.clip(phi_new, 0.0, 1.0)
        else:
            phi_k = phi_n.copy()

        if use_compress:
            from .vof_cn import _compression_flux_bounded
            comp_div = _compression_flux_bounded(
                phi_k, u_face_vof, dx, dt, bc_l, bc_r, n_ghost=2)
            phi_k = np.clip(phi_k - dt * comp_div, 0.0, 1.0)

        # Old-time quantities (ACID: use updated phi_k, same as segregated)
        props_old = _mix_props_with_phi(p_n, u_n, T_n, phi_k)
        rho_old = props_old['rho']
        h_old = _mix_h_with_phi(p_n, u_n, T_n, phi_k)

        # Initialise iterate
        p_k = p_n.copy()
        u_k = u_n.copy()
        T_k = T_n.copy()

        max_outer = cfg.get('max_outer', _MAX_OUTER)
        max_inner = cfg.get('max_inner', _MAX_INNER)
        inner_tol = cfg.get('inner_tol', _INNER_TOL)
        outer_tol = cfg.get('outer_tol', _OUTER_TOL)

        info_outer = {'converged': False, 'outer_iters': 0, 'inner_iters': []}

        for outer in range(max_outer):
            # --- Step B: 3N barotropic inner/outer loop for (p,u,T) with frozen φ ---
            props_k = _mix_props_with_phi(p_k, u_k, T_k, phi_k)
            rho_k_arr = props_k['rho']
            zeta_k_arr = props_k['zeta_v']
            h_k = _mix_h_with_phi(p_k, u_k, T_k, phi_k)
            rho_face_acid = acid_face_density(
                rho_k_arr, props_k['c_mix'], phi_k, bc_l, bc_r)
            rho_star = harmonic_face_density(rho_k_arr, bc_l, bc_r)
            e_diag = _momentum_diagonal(rho_k_arr, dx, dt)
            d_hat = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
            theta_k_face, u_bar_k = _compute_face_velocity(
                u_k, p_k, d_hat, dx, bc_l, bc_r,
                theta_old=theta_old, u_bar_old=u_bar_old,
                rho_star_old=rho_star_old, dt=dt)

            # Inner: freeze T, solve (p,u) via 3N
            inner_iters = 0
            phi_T_k_arr = props_k.get('phi_v', None)
            for inner in range(max_inner):
                A_3N, b_3N = assemble_newton_3N(
                    N, dx, dt,
                    rho_old, u_n, h_old, p_n,
                    rho_k_arr, u_k, h_k, p_k, T_k, phi_k,
                    zeta_k_arr,
                    rho_face_acid, d_hat, theta_k_face,
                    ph1, ph2, bc_l, bc_r,
                    freeze_h=True, third_var='T',
                    phi_k=phi_T_k_arr,
                    mixing_type=mixing_type)

                x_3N = np.concatenate([p_k, u_k, T_k])
                r_3N = b_3N - A_3N.dot(x_3N)
                p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
                u_ref = float(max(np.mean(np.abs(u_k)), 1e-6))
                dx_3N = solve_linear_system(A_3N, r_3N,
                                            p_ref=p_ref, u_ref=u_ref,
                                            h_ref=float(max(np.mean(np.abs(T_k)), 1.0)))
                dp = dx_3N[0:N]; du = dx_3N[N:2*N]
                p_k = np.maximum(p_k + dp, _P_FLOOR)
                u_k = u_k + du
                theta_k_face, u_bar_k = _compute_face_velocity(
                    u_k, p_k, d_hat, dx, bc_l, bc_r,
                    theta_old=theta_old, u_bar_old=u_bar_old,
                    rho_star_old=rho_star_old, dt=dt)
                inner_res = max(np.max(np.abs(dp))/p_ref,
                                np.max(np.abs(du))/max(u_ref, 1e-6))
                inner_iters += 1
                if inner_res < inner_tol:
                    break
            info_outer['inner_iters'].append(inner_iters)

            # Outer: full energy solve via 3N
            props_k = _mix_props_with_phi(p_k, u_k, T_k, phi_k)
            rho_k_arr = props_k['rho']
            zeta_k_arr = props_k['zeta_v']
            phi_T_k_arr = props_k['phi_v']
            h_k = _mix_h_with_phi(p_k, u_k, T_k, phi_k)
            rho_face_acid = acid_face_density(rho_k_arr, props_k['c_mix'], phi_k, bc_l, bc_r)
            rho_star = harmonic_face_density(rho_k_arr, bc_l, bc_r)
            e_diag = _momentum_diagonal(rho_k_arr, dx, dt)
            d_hat = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
            theta_k_face, u_bar_k = _compute_face_velocity(
                u_k, p_k, d_hat, dx, bc_l, bc_r,
                theta_old=theta_old, u_bar_old=u_bar_old,
                rho_star_old=rho_star_old, dt=dt)

            A_3N, b_3N = assemble_newton_3N(
                N, dx, dt,
                rho_old, u_n, h_old, p_n,
                rho_k_arr, u_k, h_k, p_k, T_k, phi_k,
                zeta_k_arr,
                rho_face_acid, d_hat, theta_k_face,
                ph1, ph2, bc_l, bc_r,
                freeze_h=False, third_var='T',
                phi_k=phi_T_k_arr,
                mixing_type=mixing_type)
            x_3N = np.concatenate([p_k, u_k, T_k])
            r_3N = b_3N - A_3N.dot(x_3N)
            dx_3N = solve_linear_system(A_3N, r_3N,
                                        p_ref=float(max(np.mean(np.abs(p_k)), 1.0)),
                                        u_ref=float(max(np.mean(np.abs(u_k)), 1e-6)),
                                        h_ref=float(max(np.mean(np.abs(T_k)), 1.0)))
            dT = dx_3N[2*N:3*N]
            T_k = np.maximum(T_k + dT, _T_FLOOR)

            # Outer convergence: density change
            rho_k_new = _mix_props_with_phi(p_k, u_k, T_k, phi_k)['rho']
            delta_rho = np.max(np.abs(rho_k_new - rho_k_arr)) / (np.mean(np.abs(rho_k_arr)) + 1e-300)
            rho_k_arr = rho_k_new

            info_outer['outer_iters'] = outer + 1
            if delta_rho < outer_tol:
                info_outer['converged'] = True
                break

        # Convert phi back to psi for output
        if use_mass:
            from .vof_cn import Y_to_psi
            psi_new = Y_to_psi(phi_k, rho1_s, rho2_s)
        else:
            psi_new = phi_k.copy()
        psi_new = np.clip(psi_new, _EPS_PSI, 1.0 - _EPS_PSI)

        vof_field_out = phi_k

        props_new = _mix_props_with_phi(p_k, u_k, T_k, phi_k)
        u_face_new, u_bar_k = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r,
                                                      theta_old=theta_old,
                                                      u_bar_old=u_bar_old,
                                                      rho_star_old=rho_star_old,
                                                      dt=dt)

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
            'is_first_step':  False,
            'bdf_order':      1,
            'rho_nm1':        rho_old,
            'rhoU_nm1':       rho_old * u_n,
            'E_nm1':          props_old['E_total'],
            'rho_face_acid':  rho_face_acid,
            'dt_prev':        dt,
            'theta_old':      u_face_new,
            'u_bar_old':      u_bar_k,
            'rho_star_old':   rho_star,
        }
        info = {
            'converged':    info_outer['converged'],
            'outer_iters':  info_outer['outer_iters'],
            'inner_iters':  info_outer.get('inner_iters', []),
            'picard_iters': info_outer['outer_iters'],
            'residuals':    [],
        }
        return new_state, new_aux, info

    # ----------------------------------------------------------------
    # SEGREGATED PATH (coupled=False): original code below
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Step 1: VOF / mass-fraction explicit update
    # ----------------------------------------------------------------
    psi_reg = np.clip(psi_n, _EPS_PSI, 1.0 - _EPS_PSI)
    use_mass = (vof_type == 'mass')

    if use_mass:
        from .eos.base import compute_phase_props
        rho1_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), ph1)['rho'])
        rho2_s = float(compute_phase_props(np.mean(p_n), np.mean(T_n), ph2)['rho'])
        # return_Y=True: get Y_new directly (no ψ→Y→ψ round-trip)
        Y_new, _, u_face_vof = mass_fraction_step(
            psi_reg, u_n, dx, dt, bc_l, bc_r, rho1_s, rho2_s,
            use_compress=use_compress, return_Y=True)
        Y_new = np.clip(Y_new, 0.0, 1.0)
        # psi_new is for display/output only; internally we work with Y_new
        from .vof_cn import Y_to_psi
        psi_new = Y_to_psi(Y_new, rho1_s, rho2_s)
        psi_new = np.clip(psi_new, _EPS_PSI, 1.0 - _EPS_PSI)
        # vof_field used for assembly = Y_new
        vof_field = Y_new
    else:
        vof_ph1 = ph1 if use_K else None
        vof_ph2 = ph2 if use_K else None
        vof_p   = p_n if use_K else None
        vof_T   = T_n if use_K else None
        psi_new, _, u_face_vof = vof_step(
            psi_reg, u_n, dx, dt, bc_l, bc_r,
            ph1=vof_ph1, ph2=vof_ph2, p=vof_p, T=vof_T,
            use_compress=use_compress)
        psi_new = np.clip(psi_new, _EPS_PSI, 1.0 - _EPS_PSI)
        vof_field = psi_new

    # ----------------------------------------------------------------
    # Old-time quantities (ACID: use updated vof_field for temporal density)
    # ----------------------------------------------------------------
    if use_mass:
        props_old = compute_mixture_props_Y(p_n, u_n, T_n, vof_field, ph1, ph2)
        rho_old   = props_old['rho']
        h_old     = compute_specific_total_enthalpy_Y(p_n, u_n, T_n, vof_field, ph1, ph2)
    else:
        props_old = compute_mixture_props(p_n, u_n, T_n, vof_field, ph1, ph2)
        rho_old   = props_old['rho']
        h_old     = compute_specific_total_enthalpy(p_n, u_n, T_n, vof_field, ph1, ph2)

    # ----------------------------------------------------------------
    # Helpers: mixture properties depending on mixing_type
    # ----------------------------------------------------------------
    mixing_type = 'mass' if use_mass else 'volume'

    def _mix_props(p, u, T):
        if use_mass:
            return compute_mixture_props_Y(p, u, T, vof_field, ph1, ph2)
        return compute_mixture_props(p, u, T, vof_field, ph1, ph2)

    def _mix_h(p, u, T):
        if use_mass:
            return compute_specific_total_enthalpy_Y(p, u, T, vof_field, ph1, ph2)
        return compute_specific_total_enthalpy(p, u, T, vof_field, ph1, ph2)

    def _recover_T(h, u, p, T_guess):
        if use_mass:
            return recover_T_from_h_Y(h, u, p, vof_field, ph1, ph2, T_guess=T_guess)
        return recover_T_from_h(h, u, p, vof_field, ph1, ph2, T_guess=T_guess)

    # ----------------------------------------------------------------
    # Initialise Newton iterate from old state
    # ----------------------------------------------------------------
    p_k = p_n.copy()
    u_k = u_n.copy()
    T_k = T_n.copy()
    h_k = h_old.copy()

    # For puT mode, phi_k is needed (dρ/dT)
    if variable_set == 'puT':
        phi_k_arr = _mix_props(p_k, u_k, T_k)['phi_v']
    else:
        phi_k_arr = None

    info_outer = {'converged': False, 'outer_iters': 0, 'inner_iters': []}

    for outer in range(max_outer):
        props_k_iter = _mix_props(p_k, u_k, T_k)
        rho_k   = props_k_iter['rho']
        zeta_k  = props_k_iter['zeta_v']

        # ACID face density and MWI coefficient
        rho_face_acid = acid_face_density(
            rho_k, props_k_iter['c_mix'], vof_field, bc_l, bc_r)
        rho_star = harmonic_face_density(rho_k, bc_l, bc_r)

        e_diag = _momentum_diagonal(rho_k, dx, dt)
        d_hat  = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)

        # ---- Inner barotropic loop (T fixed, h/T frozen) ----
        inner_iters = 0
        third_block = T_k if variable_set == 'puT' else h_k
        for inner in range(max_inner):
            A, b_vec = assemble_newton_3N(
                N, dx, dt,
                rho_old, u_n, h_old, p_n,
                rho_k, u_k, h_k, p_k, T_k, vof_field,
                zeta_k,
                rho_face_acid, d_hat, theta_k,
                ph1, ph2, bc_l, bc_r,
                freeze_h=True,
                third_var=variable_set[-1] if variable_set == 'puT' else 'h',
                phi_k=phi_k_arr,
                mixing_type=mixing_type,
            )

            x_k = np.concatenate([p_k, u_k, third_block])
            r   = b_vec - A.dot(x_k)

            p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
            u_ref = float(max(np.mean(np.abs(u_k)), 1e-6))
            h_ref = float(max(np.mean(np.abs(third_block)), 1.0))
            dx_vec = solve_linear_system(A, r, p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)

            dp = dx_vec[0:N]
            du = dx_vec[N:2*N]

            p_k = np.maximum(p_k + dp, _P_FLOOR)
            u_k = u_k + du

            # Update face velocity with new p, u (no transient correction in inner loop)
            theta_k, u_bar_k = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r,
                                                       theta_old=theta_old,
                                                       u_bar_old=u_bar_old,
                                                       rho_star_old=rho_star_old,
                                                       dt=dt)

            res_p = np.max(np.abs(dp)) / p_ref
            res_u = np.max(np.abs(du)) / max(u_ref, 1e-6)
            inner_res = max(res_p, res_u)
            inner_iters += 1

            if inner_res < inner_tol:
                break

        info_outer['inner_iters'].append(inner_iters)

        # ---- Outer: full energy solve ----
        props_k_outer = _mix_props(p_k, u_k, T_k)
        rho_k         = props_k_outer['rho']
        zeta_k        = props_k_outer['zeta_v']
        if variable_set == 'puT':
            phi_k_arr = props_k_outer['phi_v']
        rho_face_acid = acid_face_density(
            rho_k, props_k_outer['c_mix'], vof_field, bc_l, bc_r)
        rho_star      = harmonic_face_density(rho_k, bc_l, bc_r)
        e_diag        = _momentum_diagonal(rho_k, dx, dt)
        d_hat         = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)

        # Update h_k from current (p, u, T) for assembly consistency
        h_k = _mix_h(p_k, u_k, T_k)

        tv = 'T' if variable_set == 'puT' else 'h'
        A_full, b_full = assemble_newton_3N(
            N, dx, dt,
            rho_old, u_n, h_old, p_n,
            rho_k, u_k, h_k, p_k, T_k, vof_field,
            zeta_k,
            rho_face_acid, d_hat, theta_k,
            ph1, ph2, bc_l, bc_r,
            freeze_h=False,
            third_var=tv,
            phi_k=phi_k_arr,
            mixing_type=mixing_type,
        )

        third_block = T_k if variable_set == 'puT' else h_k
        x_k_full = np.concatenate([p_k, u_k, third_block])
        r_full   = b_full - A_full.dot(x_k_full)
        p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
        u_ref = float(max(np.mean(np.abs(u_k)), 1e-6))
        h_ref = float(max(np.mean(np.abs(third_block)), 1.0))
        dx_full = solve_linear_system(A_full, r_full, p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)

        d3 = dx_full[2*N:3*N]
        if variable_set == 'puT':
            T_k_new = np.maximum(T_k + d3, _T_FLOOR)
        else:
            h_k = h_k + d3
            T_k_new = _recover_T(h_k, u_k, p_k, T_k)
            T_k_new = np.maximum(T_k_new, _T_FLOOR)

        # Outer convergence: relative density change
        rho_k_new = _mix_props(p_k, u_k, T_k_new)['rho']
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
    props_new = _mix_props(p_k, u_k, T_k)

    # Face velocity for diagnostics
    u_face_new, _ = _compute_face_velocity(u_k, p_k, d_hat, dx, bc_l, bc_r,
                                           theta_old=theta_old,
                                           u_bar_old=u_bar_old,
                                           rho_star_old=rho_star_old,
                                           dt=dt)

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
        'is_first_step':  False,
        'bdf_order':      1,
        'rho_nm1':        rho_old,
        'rhoU_nm1':       rho_old * u_n,
        'E_nm1':          props_old['E_total'],
        'rho_face_acid':  rho_face_acid,
        'dt_prev':        dt,
        # Transient correction data (Mod 2)
        'theta_old':      u_face_new,
        'u_bar_old':      u_bar_k,
        'rho_star_old':   rho_star,
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
