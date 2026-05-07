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
from .assembly import (assemble_newton_3N, assemble_newton_4N, assemble_newton_Ns,
                       solve_linear_system, solve_schur_4N, solve_block_schur_4N,
                       residual_4N, solve_jfnk_4N, _ci)
from .vof_cn import vof_step, mass_fraction_step
from .interface.cicsam import cicsam_face_beta, cicsam_face_jacobian
from .vof_cn import compute_compression_coefficients


_P_FLOOR    = 1.0      # Pa — minimum pressure to prevent ρ=0 in ideal gas
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


def _newton_deltaQ_4N(N, dx_cell, dt,
                      p_n, u_n, T_n, phi_n,
                      rho_n, h_n,
                      ph1, ph2, bc_l, bc_r,
                      mixing_type, cfg,
                      theta_old, u_bar_old, rho_star_old):
    """ΔQ formulation: J(Q_k)·ΔQ = -R(Q_k), Q_{k+1} = Q_k + ΔQ.

    Proper Newton's method — J and R evaluated at same Q_k.
    All coefficients (ACID, MWI, CICSAM) updated at each Newton iteration.
    """
    max_newton = cfg.get('max_newton', 20)
    newton_tol = cfg.get('newton_tol', 1e-6)
    use_mass = (mixing_type == 'mass')
    third_var = cfg.get('third_var', 'T')   # 'T' (default) or 'h'
    picard_adv = cfg.get('picard_advection', False)  # Picard advection: skip spatial ACID Y-Jacobian

    # Initial guess = old-time state
    p_k   = p_n.copy()
    u_k   = u_n.copy()
    T_k   = T_n.copy()
    phi_k = phi_n.copy()

    # h-mode: compute initial specific total enthalpy h_k from (p, u, T, φ)
    if third_var == 'h':
        if use_mass:
            h_k_mode = compute_specific_total_enthalpy_Y(p_k, u_k, T_k, phi_k, ph1, ph2)
        else:
            h_k_mode = compute_specific_total_enthalpy(p_k, u_k, T_k, phi_k, ph1, ph2)
    else:
        h_k_mode = None   # unused in T-mode

    info = {'converged': False, 'outer_iters': 0, 'residuals': []}

    # --- Pseudo-transient continuation (PTC) initialisation ---
    psi_tau = dt        # pseudo-time step 초기값 (= physical dt)
    R_norm_prev = 0.0   # for SER update
    R_norm_initial = None  # for PTC terminal disable (Stage 1b)

    for niter in range(max_newton):
        # --- 1. Mixture properties at Q_k ---
        if use_mass:
            props_k = compute_mixture_props_Y(p_k, u_k, T_k, phi_k, ph1, ph2)
            h_k_props = compute_specific_total_enthalpy_Y(p_k, u_k, T_k, phi_k, ph1, ph2)
        else:
            props_k = compute_mixture_props(p_k, u_k, T_k, phi_k, ph1, ph2)
            h_k_props = compute_specific_total_enthalpy(p_k, u_k, T_k, phi_k, ph1, ph2)

        # h-mode: use h_k_mode (the Newton unknown) for assembly; T_k is the recovered value
        # T-mode: h_k used in assembly is h_k_props (derived from T_k)
        h_k_assem = h_k_mode if (third_var == 'h') else h_k_props

        rho_k        = props_k['rho']
        zeta_k       = props_k['zeta_v']
        phi_T_k      = props_k.get('phi_v', np.zeros(N))
        if use_mass:
            alpha_k    = props_k.get('d_rho_dY', np.zeros(N))
            drh_dphi_k = props_k.get('d_rho_h_dY', np.zeros(N))
        else:
            alpha_k    = props_k.get('Delta_rho_psi', np.zeros(N))
            drh_dphi_k = props_k.get('d_rho_h_dpsi', np.zeros(N))

        # --- 2. ACID, MWI, CICSAM at Q_k (updated each iteration) ---
        rho_face_k  = acid_face_density(rho_k, props_k['c_mix'], phi_k, bc_l, bc_r)
        rho_star_k  = harmonic_face_density(rho_k, bc_l, bc_r)
        e_diag_k    = _momentum_diagonal(rho_k, dx_cell, dt)
        d_hat_k     = mwi_face_coeff_denner(e_diag_k, rho_star_k, dx_cell, dt, bc_l, bc_r)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat_k, dx_cell, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)
        phi_ext_k   = apply_ghost(phi_k, bc_l, bc_r, 2)
        u_ext_k     = apply_ghost_velocity(u_k, bc_l, bc_r, 2)
        u_face_k    = np.array([0.5 * (u_ext_k[2 + f - 1] + u_ext_k[2 + f])
                                for f in range(N + 1)])
        beta_k      = cicsam_face_beta(phi_ext_k, u_face_k, dt, dx_cell, n_ghost=2)

        # --- 3. Jacobian J at Q_k ---
        # Full Newton: rho_n frozen (old-time), alpha_k full (spatial ACID Jacobian included)
        # picard_adv=True: spatial ACID Y-Jacobian terms are deferred (Picard advection mode)
        J, _ = assemble_newton_4N(
            N, dx_cell, dt,
            rho_n, u_n, h_n, p_n, phi_n,                   # old-time (frozen)
            rho_k, u_k, h_k_assem, p_k, T_k, phi_k,        # iterate Q_k
            zeta_k, phi_T_k,
            alpha_k, drh_dphi_k,                            # FULL alpha (not zero)
            rho_face_k, d_hat_k, theta_k,
            beta_k, ph1, ph2, bc_l, bc_r,
            mixing_type=mixing_type,
            third_var=third_var,
            picard_advection=picard_adv,
            use_acid=cfg.get('use_acid', True))

        # --- 3.5. Pseudo-transient continuation: add (M/Δτ) to diagonal ---
        # M scaled to match dominant off-diagonal coupling:
        #   Mass row:   M = |α_ref| ← matches α/dt in Y-column → ratio ≈ 1
        #   Mom row:    M = ρ_ref
        #   Energy row: M = ρ_ref (h-mode) or ρ_ref·cp (T-mode)
        #   Species row: M = ρ_ref
        # PTC is active in early iterations or while residual is still large.
        # After convergence enters terminal phase, PTC is disabled to allow
        # quadratic Newton convergence (Stage 1b).
        rho_ref_ptc = float(np.mean(rho_k))
        alpha_ref_ptc = float(np.max(np.abs(alpha_k))) + 1e-300

        # PTC activation: always on for niter==0 (no prior residual info yet).
        # For niter>=1, use R_norm_prev vs R_norm_initial to decide.
        if niter == 0:
            ptc_active = True
        elif R_norm_initial is not None and R_norm_prev < 1e-2 * R_norm_initial:
            ptc_active = False   # terminal phase: let pure Newton take over
        else:
            ptc_active = True

        if ptc_active:
            J_lil = J.tolil()
            for ii in range(N):
                J_lil[_ci(0, ii, N), _ci(0, ii, N)] += alpha_ref_ptc / psi_tau   # mass: match α
                J_lil[_ci(1, ii, N), _ci(1, ii, N)] += rho_ref_ptc / psi_tau     # momentum
                J_lil[_ci(2, ii, N), _ci(2, ii, N)] += rho_ref_ptc / psi_tau     # energy
                J_lil[_ci(3, ii, N), _ci(3, ii, N)] += rho_ref_ptc / psi_tau     # species
            J = J_lil.tocsr()

        # --- 4. Nonlinear residual R at Q_k ---
        # residual_4N always uses T as the 3rd variable in x_4N
        x_k = np.concatenate([p_k, u_k, T_k, phi_k])
        R_k = residual_4N(
            x_k, N, dx_cell, dt,
            rho_n, u_n, h_n, p_n, phi_n,            # old-time (frozen)
            ph1, ph2, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old,
            mixing_type=mixing_type,
            use_K=cfg.get('use_K', False),
            use_acid=cfg.get('use_acid', True))

        # Record initial residual for PTC terminal disable (Stage 1b)
        if R_norm_initial is None:
            R_norm_initial = float(np.linalg.norm(R_k))

        # --- 5. Solve J·ΔQ = -R_k ---
        p_ref   = float(max(np.mean(np.abs(p_k)), 1.0))
        u_ref   = float(max(np.mean(np.abs(u_k)), 1.0))
        if third_var == 'h':
            # h-mode: reference scale is specific enthalpy
            h3_ref = float(max(np.mean(np.abs(h_k_assem)), 1.0))
        else:
            h3_ref  = float(max(np.mean(np.abs(T_k)), 1.0))
        try:
            deltaQ = solve_block_schur_4N(J, -R_k, N,
                                          p_ref=p_ref, u_ref=u_ref, h_ref=h3_ref)
        except Exception:
            deltaQ = solve_linear_system(J, -R_k,
                                         p_ref=p_ref, u_ref=u_ref, h_ref=h3_ref,
                                         phi_ref=1.0, n_blocks=4)

        dp  = deltaQ[0:N]
        du  = deltaQ[N:2*N]
        dh3 = deltaQ[2*N:3*N]   # Δh (h-mode) or ΔT (T-mode)
        dY  = deltaQ[3*N:4*N]

        # --- 6. Backtracking line search + physics clamping ---
        R_norm = np.linalg.norm(R_k)
        omega = 1.0
        R_trial_norm = R_norm
        for ls_iter in range(8):
            p_trial  = np.maximum(p_k + omega * dp, _P_FLOOR)
            u_trial  = u_k + omega * du
            Y_trial  = np.clip(phi_k + omega * dY, 0.0, 1.0)
            if third_var == 'h':
                h_trial  = h_k_mode + omega * dh3
                T_trial  = recover_T_from_h_Y(h_trial, u_trial, p_trial, Y_trial, ph1, ph2, T_guess=T_k) if use_mass \
                           else recover_T_from_h(h_trial, u_trial, p_trial, Y_trial, ph1, ph2, T_guess=T_k)
                T_trial  = np.maximum(T_trial, _T_FLOOR)
            else:
                T_trial  = np.maximum(T_k + omega * dh3, _T_FLOOR)
            x_trial = np.concatenate([p_trial, u_trial, T_trial, Y_trial])
            R_trial = residual_4N(
                x_trial, N, dx_cell, dt,
                rho_n, u_n, h_n, p_n, phi_n,
                ph1, ph2, bc_l, bc_r,
                theta_old=theta_old, u_bar_old=u_bar_old,
                rho_star_old=rho_star_old,
                mixing_type=mixing_type,
                use_K=cfg.get('use_K', False),
                use_acid=cfg.get('use_acid', True))
            R_trial_norm = np.linalg.norm(R_trial)
            if R_trial_norm < R_norm:
                break
            omega *= 0.5
        omega = min(omega,
                    0.5 * p_ref / (np.max(np.abs(dp)) + 1e-300),
                    500.0 / (np.max(np.abs(du)) + 1e-300),
                    0.5 * h3_ref / (np.max(np.abs(dh3)) + 1e-300))

        # --- 7. Update Q_{k+1} = Q_k + ω·ΔQ ---
        p_k   = np.maximum(p_k   + omega * dp, _P_FLOOR)
        u_k   = u_k   + omega * du
        phi_k = np.clip(phi_k + omega * dY, 0.0, 1.0)
        if third_var == 'h':
            h_k_mode = h_k_mode + omega * dh3
            # Recover T_k from updated (h, u, p, φ) — needed for mixture props next iteration
            if use_mass:
                T_k = recover_T_from_h_Y(h_k_mode, u_k, p_k, phi_k, ph1, ph2, T_guess=T_k)
            else:
                T_k = recover_T_from_h(h_k_mode, u_k, p_k, phi_k, ph1, ph2, T_guess=T_k)
            T_k = np.maximum(T_k, _T_FLOOR)
        else:
            T_k   = np.maximum(T_k   + omega * dh3, _T_FLOOR)

        # --- 8. Convergence check ---
        res_norm  = R_norm   # use pre-update residual for convergence monitor
        res_delta = max(
            np.max(np.abs(omega * dp)) / p_ref,
            np.max(np.abs(omega * du)) / max(u_ref, 1e-6),
            np.max(np.abs(omega * dh3)) / max(h3_ref, 1e-6),
            np.max(np.abs(omega * dY)))
        info['residuals'].append(res_norm)
        info['outer_iters'] = niter + 1

        # --- 9. SER: update pseudo-time step ---
        R_norm_new = R_trial_norm
        if niter > 0 and R_norm_prev > 1e-300:
            psi_tau = psi_tau * R_norm_prev / max(R_norm_new, 1e-300)
            psi_tau = float(np.clip(psi_tau, dt * 0.01, dt * 1e12))
        R_norm_prev = R_norm_new

        if niter < 5 or niter % 10 == 0:
            print(f"    ΔQ-Newton {niter:3d}: |R|={res_norm:.3e}  "
                  f"|ΔQ|={res_delta:.3e}  ω={omega:.3e}  "
                  f"Δτ/dt={psi_tau/dt:.2e}")

        if res_delta < newton_tol:
            info['converged'] = True
            break

    x_new = np.concatenate([p_k, u_k, T_k, phi_k])
    return x_new, info


def _frozen_newton_4N(x0, N, dx_cell, dt, rho_old, u_old, h_old, p_old, phi_old,
                      ph1, ph2, bc_l, bc_r, mixing_type, cfg,
                      theta_old_face, u_bar_old_face, rho_star_old_face,
                      mix_props_fn, mix_h_fn):
    """Frozen-coefficient Newton for 4N coupled system.

    ALL linearization coefficients (rho_k, zeta_k, alpha_k, ACID, CICSAM)
    are frozen at old-time. Only face velocity θ is iterate-dependent.

    Ref: Denner 2018 — Section 6, frozen linearization approach.
    """
    # cicsam_face_beta is imported at module level

    max_newton = cfg.get('max_newton', 20)
    newton_tol = cfg.get('newton_tol', 1e-6)

    p_k   = x0[0:N].copy()
    u_k   = x0[N:2*N].copy()
    T_k   = x0[2*N:3*N].copy()
    phi_k = np.clip(x0[3*N:4*N].copy(), 0.0, 1.0)

    # Freeze ALL coefficients at old-time
    props_old = mix_props_fn(p_old, u_old, T_k, phi_old)
    rho_frozen       = rho_old.copy()
    zeta_frozen      = props_old['zeta_v']
    phi_T_frozen     = props_old.get('phi_v', np.zeros(N))
    alpha_frozen     = props_old.get('Delta_rho_psi', np.zeros(N))
    drh_dphi_frozen  = props_old.get('d_rho_h_dpsi', np.zeros(N))
    h_frozen         = h_old.copy()

    # Frozen ACID face density
    rho_face_frozen = acid_face_density(rho_frozen, props_old['c_mix'], phi_old, bc_l, bc_r)
    rho_star_frozen = harmonic_face_density(rho_frozen, bc_l, bc_r)
    e_diag_frozen   = _momentum_diagonal(rho_frozen, dx_cell, dt)
    d_hat_frozen    = mwi_face_coeff_denner(e_diag_frozen, rho_star_frozen, dx_cell, dt, bc_l, bc_r)

    # Frozen CICSAM beta
    u_ext_old  = apply_ghost_velocity(u_old, bc_l, bc_r, 2)
    u_face_old = np.array([0.5 * (u_ext_old[2 + f - 1] + u_ext_old[2 + f])
                            for f in range(N + 1)])
    phi_ext_old  = apply_ghost(phi_old, bc_l, bc_r, 2)
    beta_frozen  = cicsam_face_beta(phi_ext_old, u_face_old, dt, dx_cell, n_ghost=2)

    info = {'converged': False, 'outer_iters': 0, 'residuals': []}

    for niter in range(max_newton):
        # θ depends on current iterate p_k, u_k (only non-frozen part)
        theta_k, u_bar_k = _compute_face_velocity(
            u_k, p_k, d_hat_frozen, dx_cell, bc_l, bc_r,
            theta_old=theta_old_face, u_bar_old=u_bar_old_face,
            rho_star_old=rho_star_old_face, dt=dt)

        A_4N, b_4N = assemble_newton_4N(
            N, dx_cell, dt,
            rho_old, u_old, h_old, p_old, phi_old,
            rho_frozen, u_k, h_frozen, p_k, T_k, phi_k,
            zeta_frozen, phi_T_frozen,
            alpha_frozen, drh_dphi_frozen,
            rho_face_frozen, d_hat_frozen, theta_k,
            beta_frozen, ph1, ph2, bc_l, bc_r,
            mixing_type=mixing_type)

        x_4N = np.concatenate([p_k, u_k, T_k, phi_k])
        r_4N = b_4N - A_4N.dot(x_4N)

        p_ref   = float(max(np.mean(np.abs(p_k)), 1.0))
        u_ref_v = float(max(np.mean(np.abs(u_k)), 1.0))
        h_ref_v = float(max(np.mean(np.abs(T_k)), 1.0))

        dx_4N = solve_schur_4N(A_4N, r_4N, N,
                               p_ref=p_ref, u_ref=u_ref_v, h_ref=h_ref_v)

        dp   = dx_4N[0:N]
        du   = dx_4N[N:2*N]
        dT   = dx_4N[2*N:3*N]
        dphi = dx_4N[3*N:4*N]

        # Simple line search / damping
        omega_flow = min(1.0,
            0.5 * p_ref / (np.max(np.abs(dp)) + 1e-300),
            500.0 / (np.max(np.abs(du)) + 1e-300),
            0.5 * h_ref_v / (np.max(np.abs(dT)) + 1e-300))

        p_k   = np.maximum(p_k   + omega_flow * dp,   _P_FLOOR)
        u_k   = u_k   + omega_flow * du
        T_k   = np.maximum(T_k   + omega_flow * dT,   _T_FLOOR)
        phi_k = np.clip(phi_k + dphi, 0.0, 1.0)

        res = max(
            np.max(np.abs(omega_flow * dp))   / p_ref,
            np.max(np.abs(omega_flow * du))   / max(u_ref_v, 1e-6),
            np.max(np.abs(omega_flow * dT))   / max(h_ref_v, 1e-6),
            np.max(np.abs(dphi)))
        info['outer_iters'] = niter + 1
        info['residuals'].append(res)

        if res < newton_tol:
            info['converged'] = True
            break

    x_new = np.concatenate([p_k, u_k, T_k, phi_k])
    return x_new, info


def step(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg=None):
    """
    One time step: Newton + barotropic inner/outer loop.

    Parameters / Returns identical to old solver_a.step interface.
    """
    if cfg is None:
        cfg = {}

    # ----------------------------------------------------------------
    # 5-EQUATION AD-JACOBIAN PATH: autograd Jacobian Newton solver
    # ----------------------------------------------------------------
    if cfg.get('five_eq_cons_ad', False):
        from .solver_5eq_ad import step_5eq_cons_ad
        return step_5eq_cons_ad(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg)

    if cfg.get('five_eq_cons', False):
        from .solver_5eq_ad import step_5eq_cons
        return step_5eq_cons(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg)

    if cfg.get('five_eq_ad', False):
        from .solver_5eq_ad import step_5eq_ad
        return step_5eq_ad(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg)

    # ----------------------------------------------------------------
    # 5-EQUATION CONSERVATIVE PATH: Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}
    # ----------------------------------------------------------------
    if cfg.get('five_eq', False):
        from .solver_5eq import step_5eq
        return step_5eq(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg)

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

        def _recover_T(h, u, p, T_guess):
            if use_mass:
                return recover_T_from_h_Y(h, u, p, phi_k, ph1, ph2, T_guess=T_guess)
            return recover_T_from_h(h, u, p, phi_k, ph1, ph2, T_guess=T_guess)

        # --- Fully Coupled 4N: (p, u, T, ψ) ---
        max_newton  = cfg.get('max_newton', 50)
        newton_tol  = cfg.get('newton_tol', 1e-6)
        solver_mode = cfg.get('solver_mode', 'delta')

        # Old-time state
        props_old_init = _mix_props_with_phi(p_n, u_n, T_n, phi_n)
        rho_old = props_old_init['rho']
        h_old   = _mix_h_with_phi(p_n, u_n, T_n, phi_n)

        # Initial guess = old-time state
        x0 = np.concatenate([p_n, u_n, T_n, phi_n])

        if solver_mode == 'delta':
            # ---- ΔQ formulation: proper Newton J·ΔQ = -R ----
            x_new, jfnk_info = _newton_deltaQ_4N(
                N, dx, dt,
                p_n, u_n, T_n, phi_n,
                rho_old, h_old,
                ph1, ph2, bc_l, bc_r,
                mixing_type, cfg,
                theta_old, u_bar_old, rho_star_old)
        elif solver_mode == 'jfnk':
            # ---- JFNK path with optional ILU preconditioner ----
            def _res_fn(x):
                return residual_4N(
                    x, N, dx, dt,
                    rho_old, u_n, h_old, p_n, phi_n,
                    ph1, ph2, bc_l, bc_r,
                    theta_old=theta_old, u_bar_old=u_bar_old,
                    rho_star_old=rho_star_old,
                    mixing_type=mixing_type,
                    use_K=cfg.get('use_K', False))

            # Preconditioner factory: approximate Jacobian at current state
            def _precond_fn(x_curr):
                _p   = x_curr[0:N]
                _u   = x_curr[N:2*N]
                _T   = x_curr[2*N:3*N]
                _phi = np.clip(x_curr[3*N:4*N], 0.0, 1.0)
                _props = _mix_props_with_phi(_p, _u, _T, _phi)
                _h     = _mix_h_with_phi(_p, _u, _T, _phi)
                _rho_star = harmonic_face_density(_props['rho'], bc_l, bc_r)
                _e_diag   = _momentum_diagonal(_props['rho'], dx, dt)
                _d_hat    = mwi_face_coeff_denner(_e_diag, _rho_star, dx, dt, bc_l, bc_r)
                _theta, _ = _compute_face_velocity(
                    _u, _p, _d_hat, dx, bc_l, bc_r,
                    theta_old=theta_old, u_bar_old=u_bar_old,
                    rho_star_old=rho_star_old, dt=dt)
                _rho_face = acid_face_density(_props['rho'], _props['c_mix'], _phi, bc_l, bc_r)
                _u_ext  = apply_ghost_velocity(_u, bc_l, bc_r, 2)
                _u_face = np.array([0.5 * (_u_ext[2 + f - 1] + _u_ext[2 + f])
                                    for f in range(N + 1)])
                _phi_ext = apply_ghost(_phi, bc_l, bc_r, 2)
                _beta    = cicsam_face_beta(_phi_ext, _u_face, dt, dx, n_ghost=2)
                A_approx, _ = assemble_newton_4N(
                    N, dx, dt,
                    rho_old, u_n, h_old, p_n, phi_n,
                    _props['rho'], _u, _h, _p, _T, _phi,
                    _props['zeta_v'], _props.get('phi_v', np.zeros(N)),
                    _props.get('Delta_rho_psi', np.zeros(N)),
                    _props.get('d_rho_h_dpsi', np.zeros(N)),
                    _rho_face, _d_hat, _theta,
                    _beta, ph1, ph2, bc_l, bc_r,
                    mixing_type=mixing_type)
                return A_approx

            x_new, jfnk_info = solve_jfnk_4N(
                _res_fn, x0, N,
                max_newton=max_newton,
                newton_tol=cfg.get('newton_tol', 1e-6),
                max_gmres=cfg.get('max_gmres', 100),
                gmres_tol=cfg.get('gmres_tol', 1e-3),
                omega=cfg.get('newton_omega', 1.0),
                verbose=True,
                precond_fn=_precond_fn)
        else:
            # ---- Frozen-coefficient Newton path ----
            x_new, jfnk_info = _frozen_newton_4N(
                x0, N, dx, dt, rho_old, u_n, h_old, p_n, phi_n,
                ph1, ph2, bc_l, bc_r, mixing_type, cfg,
                theta_old, u_bar_old, rho_star_old,
                _mix_props_with_phi, _mix_h_with_phi)

        p_k = x_new[0:N]
        u_k = x_new[N:2*N]
        T_k = x_new[2*N:3*N]
        phi_k = np.clip(x_new[3*N:4*N], 0.0, 1.0)

        # Convert phi back to psi for output
        if use_mass:
            from .vof_cn import Y_to_psi
            psi_new = Y_to_psi(phi_k, rho1_s, rho2_s)
        else:
            psi_new = phi_k.copy()
        psi_new = np.clip(psi_new, _EPS_PSI, 1.0 - _EPS_PSI)

        props_new = _mix_props_with_phi(p_k, u_k, T_k, phi_k)
        rho_star_new = harmonic_face_density(props_new['rho'], bc_l, bc_r)
        e_diag_new = _momentum_diagonal(props_new['rho'], dx, dt)
        d_hat_new = mwi_face_coeff_denner(
            e_diag_new, rho_star_new, dx, dt, bc_l, bc_r)
        u_face_new, u_bar_new = _compute_face_velocity(
            u_k, p_k, d_hat_new, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)
        rho_face_acid_new = acid_face_density(
            props_new['rho'], props_new['c_mix'], phi_k, bc_l, bc_r)

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
            'E_nm1':          props_old_init['E_total'],
            'rho_face_acid':  rho_face_acid_new,
            'dt_prev':        dt,
            'theta_old':      u_face_new,
            'u_bar_old':      u_bar_new,
            'rho_star_old':   rho_star_new,
        }
        info = {
            'converged':    jfnk_info['converged'],
            'outer_iters':  jfnk_info['outer_iters'],
            'inner_iters':  [],
            'picard_iters': jfnk_info['outer_iters'],
            'residuals':    jfnk_info.get('residuals', []),
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
    # Old-time quantities: use actual state from previous time step.
    # ----------------------------------------------------------------
    # The standard ACID approach recomputes ρ_old = ρ(p_n, T_n, ψ_new),
    # which gives exact Abgrall conservation for uniform (p,T). However,
    # when ψ_new changes the dominant phase (e.g., air→water) but T_n
    # is still the old phase's temperature, the EOS gives unphysical
    # density (e.g., water at T=6.94K → ρ~43000). Using the actual old
    # state (ρ_n, h_n) avoids this entirely: ρ_old is always physical.
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

    # Newton loop for (p, u, T/h).
    # ρ̃ is always implicit (Full Newton) — NOT Picard (which freezes ρ̃).
    # Two strategies:
    #   1. Single Newton: solve full 3N simultaneously (fast, but requires
    #      non-zero T-diagonal — can fail for ideal gas / stiffened gas with q=0)
    #   2. Barotropic (Denner 2018 §6): inner=freeze T, solve (p,u); outer=update T
    #      (robust for stiffened gas, avoids ill-conditioned T-row)
    use_barotropic = cfg.get('use_barotropic', False)
    max_newton = cfg.get('max_newton', max_outer * max_inner)
    newton_tol = cfg.get('newton_tol', inner_tol)
    info_outer = {'converged': False, 'outer_iters': 0, 'inner_iters': [0]}

    tv = 'T' if variable_set == 'puT' else 'h'

    def _update_coeffs():
        """Recompute props, face quantities, and return them."""
        props_k_iter = _mix_props(p_k, u_k, T_k)
        rho_k   = props_k_iter['rho']
        zeta_k  = props_k_iter['zeta_v']
        phi_k_a = props_k_iter['phi_v'] if variable_set == 'puT' else None
        h_k_val = _mix_h(p_k, u_k, T_k)
        rfa = acid_face_density(rho_k, props_k_iter['c_mix'], vof_field, bc_l, bc_r)
        rs  = harmonic_face_density(rho_k, bc_l, bc_r)
        ed  = _momentum_diagonal(rho_k, dx, dt)
        dh  = mwi_face_coeff_denner(ed, rs, dx, dt, bc_l, bc_r)
        tk, ubk = _compute_face_velocity(
            u_k, p_k, dh, dx, bc_l, bc_r,
            theta_old=theta_old, u_bar_old=u_bar_old,
            rho_star_old=rho_star_old, dt=dt)
        return rho_k, zeta_k, phi_k_a, h_k_val, rfa, rs, dh, tk, ubk

    if use_barotropic:
        # --- Barotropic inner/outer (Denner 2018 §6, Fig.5) using puh ---
        # Inner: solve (p, u, h) simultaneously (h-diagonal = ρ/dt > 0, always
        #        well-conditioned). Density uses FROZEN T (barotropic assumption).
        # Outer: recover T from h, recompute ρ(p,T), check density convergence.
        # This avoids the ill-conditioned T-diagonal issue (d(ρh)/dT ≈ 0 for
        # ideal gas and stiffened gas with q=0).
        for outer in range(max_outer):
            rho_k, zeta_k, _, h_k, rho_face_acid, rho_star, d_hat, theta_k, u_bar_k = _update_coeffs()
            props_inner = _mix_props(p_k, u_k, T_k)
            for inner in range(max_inner):
                A_mat, b_vec = assemble_newton_3N(
                    N, dx, dt, rho_old, u_n, h_old, p_n,
                    rho_k, u_k, h_k, p_k, T_k, vof_field, zeta_k,
                    rho_face_acid, d_hat, theta_k, ph1, ph2, bc_l, bc_r,
                    freeze_h=False, third_var='h',   # puh: h-diagonal = ρ/dt
                    phi_k=None, mixing_type=mixing_type)
                x_3N = np.concatenate([p_k, u_k, h_k])
                r_3N = b_vec - A_mat.dot(x_3N)
                p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
                c_ref = float(np.mean(props_inner.get('c_mix', np.ones(N))))
                u_ref = float(max(np.mean(np.abs(u_k)), c_ref, 1.0))
                h_ref = float(max(np.mean(np.abs(h_k)), 1.0))
                dx_3N = solve_linear_system(A_mat, r_3N,
                    p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)
                dp = dx_3N[0:N]; du = dx_3N[N:2*N]; dh = dx_3N[2*N:3*N]
                p_k = np.maximum(p_k + dp, _P_FLOOR)
                u_k = u_k + du
                h_k = h_k + dh
                # Update coefficients with FROZEN T (barotropic)
                props_inner = _mix_props(p_k, u_k, T_k)
                rho_k   = props_inner['rho']
                zeta_k  = props_inner['zeta_v']
                h_k     = _mix_h(p_k, u_k, T_k)  # recompute h for consistency
                rho_face_acid = acid_face_density(rho_k, props_inner['c_mix'], vof_field, bc_l, bc_r)
                rho_star = harmonic_face_density(rho_k, bc_l, bc_r)
                e_diag = _momentum_diagonal(rho_k, dx, dt)
                d_hat = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)
                theta_k, u_bar_k = _compute_face_velocity(
                    u_k, p_k, d_hat, dx, bc_l, bc_r,
                    theta_old=theta_old, u_bar_old=u_bar_old,
                    rho_star_old=rho_star_old, dt=dt)
                res_inner = max(np.max(np.abs(dp))/p_ref,
                                np.max(np.abs(du))/max(u_ref,1e-6))
                if res_inner < inner_tol:
                    break
            info_outer['inner_iters'].append(inner + 1)
            # Outer: recover T from h, update density
            T_k = _recover_T(h_k, u_k, p_k, T_k)
            T_k = np.maximum(T_k, _T_FLOOR)
            rho_new = _mix_props(p_k, u_k, T_k)['rho']
            delta_rho = np.max(np.abs(rho_new - rho_k)) / (np.mean(np.abs(rho_k)) + 1e-300)
            info_outer['outer_iters'] = outer + 1
            if delta_rho < outer_tol:
                info_outer['converged'] = True
                break
    else:
        # --- Single Newton loop: solve full 3N simultaneously ---
        for niter in range(max_newton):
            rho_k, zeta_k, phi_k_arr, h_k, rho_face_acid, rho_star, d_hat, theta_k, u_bar_k = _update_coeffs()
            A_mat, b_vec = assemble_newton_3N(
                N, dx, dt, rho_old, u_n, h_old, p_n,
                rho_k, u_k, h_k, p_k, T_k, vof_field, zeta_k,
                rho_face_acid, d_hat, theta_k, ph1, ph2, bc_l, bc_r,
                freeze_h=False, third_var=tv, phi_k=phi_k_arr,
                mixing_type=mixing_type)
            third_block = T_k if variable_set == 'puT' else h_k
            x_k = np.concatenate([p_k, u_k, third_block])
            r = b_vec - A_mat.dot(x_k)
            p_ref = float(max(np.mean(np.abs(p_k)), 1.0))
            props_s = _mix_props(p_k, u_k, T_k)
            c_ref_s = float(np.mean(props_s.get('c_mix', np.ones(N))))
            u_ref = float(max(np.mean(np.abs(u_k)), c_ref_s, 1.0))
            h_ref = float(max(np.mean(np.abs(third_block)), 1.0))
            dx_vec = solve_linear_system(A_mat, r, p_ref=p_ref, u_ref=u_ref, h_ref=h_ref)
            dp = dx_vec[0:N]; du = dx_vec[N:2*N]; d3 = dx_vec[2*N:3*N]
            omega_nr = cfg.get('newton_omega', 1.0)
            c_max_s = float(np.max(props_s['c_mix']))
            omega_ls = omega_nr * min(1.0,
                0.5 * p_ref / (np.max(np.abs(dp)) + 1e-300),
                c_max_s / (np.max(np.abs(du)) + 1e-300),
                0.5 * h_ref / (np.max(np.abs(d3)) + 1e-300))
            p_k = np.maximum(p_k + omega_ls * dp, _P_FLOOR)
            u_k = u_k + omega_ls * du
            if variable_set == 'puT':
                T_k = np.maximum(T_k + omega_ls * d3, _T_FLOOR)
            else:
                h_k = h_k + omega_ls * d3
                T_k = _recover_T(h_k, u_k, p_k, T_k)
                T_k = np.maximum(T_k, _T_FLOOR)
            res = max(np.max(np.abs(omega_nr * dp))/p_ref,
                      np.max(np.abs(omega_nr * du))/max(u_ref,1e-6),
                      np.max(np.abs(omega_nr * d3))/max(h_ref,1e-6))
            info_outer['outer_iters'] = niter + 1
            if res < newton_tol:
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
