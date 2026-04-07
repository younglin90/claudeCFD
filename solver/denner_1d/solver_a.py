# solver/denner_1d/solver_a.py
# Ref: DENNER_SCHEME.md § 7 (Mode A)
#
# Mode A driver: segregated VOF (Crank-Nicolson) + coupled (p,u,T) BDF1/BDF2.

import numpy as np

from .eos.base import compute_mixture_props
from .eos.invert import invert_eos
from .flux.mwi import acid_face_density, mwi_face_coeff, mwi_face_velocity_components
from .flux.consistent import compute_all_face_quantities
from .boundary import apply_ghost, apply_ghost_velocity
from .assembly import assemble_3N, solve_linear_system
from .vof_cn import vof_step


# Default Picard solver settings
_MAX_PICARD  = 50
_PICARD_TOL  = 1e-6
_PICARD_OMEGA = 0.5   # under-relaxation factor (0 < ω ≤ 1); <1 damps oscillations
_P_FLOOR     = 1.0    # [Pa]
_T_FLOOR     = 1e-3   # [K]


def step(state, ph1, ph2, dx, dt, bc_l, bc_r, aux, cfg=None):
    """
    Perform one time step of Mode A.

    Parameters
    ----------
    state : dict
        p, u, T, psi : ndarray (N,)   current cell-centred state
        rho, E_total  : ndarray (N,)   derived quantities at current state
        u_face        : ndarray (N+1,) face velocity from previous step (or initial)
    ph1, ph2 : dict    NASG EOS parameters for phase 1 and 2
    dx, dt   : float
    bc_l, bc_r : str
    aux : dict
        is_first_step : bool
        bdf_order     : int  (1 or 2)
        rho_nm1       : (N,) or None   (for BDF2)
        rhoU_nm1      : (N,) or None
        E_nm1         : (N,) or None
    cfg : dict or None   (optional: max_picard_iter, picard_tol)

    Returns
    -------
    new_state : dict  (same keys as state, updated)
    new_aux   : dict  (updated for next step)
    info      : dict  (convergence info)
    """
    if cfg is None:
        cfg = {}
    max_picard  = cfg.get('max_picard_iter', _MAX_PICARD)
    picard_tol  = cfg.get('picard_tol', _PICARD_TOL)
    picard_omega = cfg.get('picard_omega', _PICARD_OMEGA)

    N = len(state['p'])
    p_n   = state['p'].copy()
    u_n   = state['u'].copy()
    T_n   = state['T'].copy()
    psi_n = state['psi'].copy()

    is_first_step = aux.get('is_first_step', True)
    bdf_order     = aux.get('bdf_order', 1)
    rho_nm1  = aux.get('rho_nm1',  None)
    rhoU_nm1 = aux.get('rhoU_nm1', None)
    E_nm1    = aux.get('E_nm1',    None)

    u_face_prev = state.get('u_face', None)

    # ------------------------------------------------------------------
    # Step 0: Mixture properties at old state (psi_n — old phase distribution)
    # ------------------------------------------------------------------
    # The "old" E^n, ρ^n must be evaluated at psi_n (not psi_new) to ensure
    # that r_e = (E_n - E_k)/dt + (H_L - H_R)*u/dx = 0 for a uniform state.
    # The matching H_face is computed via face-VOF blending (psi_face_vof from
    # the VOF step), NOT van Leer upwinding of cell-centre E_total.
    # This guarantees: temporal change (E_water→E_air) exactly equals the
    # energy flux carried away by the VOF advection.
    props_n = compute_mixture_props(p_n, u_n, T_n, psi_n, ph1, ph2)
    rho_n     = props_n['rho']
    E_total_n = props_n['E_total']
    rhoU_n    = rho_n * u_n
    rho_h_n   = props_n['rho_h']

    # ------------------------------------------------------------------
    # Step 1: VOF explicit update
    # ------------------------------------------------------------------
    psi_new, psi_face_vof, u_face_vof = vof_step(
        psi_n, u_n, dx, dt, bc_l, bc_r)

    # ------------------------------------------------------------------
    # Step 2: Picard iterations for (p, u, T)
    # ------------------------------------------------------------------
    # Initialise Picard state from old values
    p_k = p_n.copy()
    u_k = u_n.copy()
    T_k = T_n.copy()

    # Initial ACID face density from old state (for first iteration)
    props_init = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)
    rho_face_acid = acid_face_density(
        props_init['rho'], props_init['c_mix'], psi_new, bc_l, bc_r)
    d_hat = mwi_face_coeff(rho_face_acid, dt)

    # Store ACID face density for MWI transient correction
    if is_first_step or (u_face_prev is None):
        rho_face_acid_old = rho_face_acid.copy()
    else:
        rho_face_acid_old = aux.get('rho_face_acid', rho_face_acid.copy())

    converged = False
    res_hist  = []

    for it in range(max_picard):
        # ----  3a. Compute Picard thermodynamic properties  ----
        props_k      = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)
        rho_k        = props_k['rho']
        E_total_k    = props_k['E_total']   # Picard state total energy
        rho_h_k      = props_k['rho_h']     # Denner 2018 ρh at Picard state
        d_rho_h_dp_v = props_k['d_rho_h_dp_v']
        d_rho_h_dT_v = props_k['d_rho_h_dT_v']

        # ----  3b. Face quantities (deferred / Picard)  ----
        # ACID face density and MWI coefficient
        rho_face_acid = acid_face_density(
            rho_k, props_k['c_mix'], psi_new, bc_l, bc_r)
        d_hat = mwi_face_coeff(rho_face_acid, dt)

        # Volume face velocity (arithmetic, used as upwind indicator)
        u_ext = apply_ghost_velocity(u_k, bc_l, bc_r)
        u_face_vol = np.empty(N + 1)
        for f in range(N + 1):
            iL = 2 + f - 1
            iR = 2 + f
            u_face_vol[f] = 0.5 * (u_ext[iL] + u_ext[iR])

        # MWI transient correction
        _, _, mwi_correction, u_face_full = mwi_face_velocity_components(
            u_k, p_k,
            rho_face_acid, d_hat, dx,
            u_face_prev, rho_face_acid_old, dt,
            bc_l, bc_r,
            include_transient=(not is_first_step and u_face_prev is not None),
        )

        # All face quantities: pass psi_face_vof from the VOF step so
        # the continuity equation uses the SAME face VOF as the VOF advection,
        # guaranteeing exact mass-flux consistency (avoids ~47 kg/m³s residual).
        face_q = compute_all_face_quantities(
            p_k, u_k, T_k, psi_new,
            ph1, ph2,
            rho_face_acid, d_hat,
            u_face_vol, dx,
            bc_l, bc_r,
            dt=dt,
            psi_face_given=psi_face_vof,
        )
        rhoU_face = face_q['rhoU_face']
        H_face    = face_q['H_face']

        # ----  3c. Assemble 3N×3N system  ----
        A, b = assemble_3N(
            N, dx, dt,
            rho_n=rho_n, u_n=u_n, T_n=T_n, E_total_n=E_total_n,
            rho_k=rho_k, u_k=u_k, T_k=T_k, p_k=p_k, psi=psi_new,
            E_total_k=E_total_k,
            zeta_v=props_k['zeta_v'],
            phi_v=props_k['phi_v'],
            dEdp_v=props_k['dEdp_v'],
            dEdT_v=props_k['dEdT_v'],
            dEdu_v=props_k['dEdu_v'],
            rho_face=face_q['rho_face'],
            d_hat=d_hat,
            rhoU_face=rhoU_face,
            H_face=H_face,
            rho_face_mwi=rho_face_acid,
            mwi_correction=mwi_correction,
            bc_l=bc_l, bc_r=bc_r,
            bdf_order=bdf_order,
            rho_nm1=rho_nm1,
            rhoU_nm1=rhoU_nm1,
            E_nm1=E_nm1,
            # Denner 2018 ACID energy equation
            # ACID = Acoustically Conservative Interface Discretisation (Denner 2018)
            # ACID modifies the MWI face density (mwi.py) for acoustic conservation.
            # The energy equation uses the standard ∂E_total/∂t form with H̃_f
            # computed by face-VOF blending in consistent.py (psi_face_vof).
            # use_acid_energy=False: correct legacy E_total temporal path.
            ph1=ph1, ph2=ph2,
            rho_h_n=rho_h_n,
            rho_h_k=rho_h_k,
            d_rho_h_dp_v=d_rho_h_dp_v,
            d_rho_h_dT_v=d_rho_h_dT_v,
            p_n=p_n,
            use_acid_energy=False,
        )

        # ----  3d. Solve in CORRECTION form: A * δx = r, where r = b - A*x_k  ----
        #
        # The assembly gives A*x = b where b contains large temporal terms
        # (O(E_water/dt) ≈ 1e14 for water) and A*x_k ≈ b at near-equilibrium.
        # Solving A*x = b directly suffers from catastrophic cancellation:
        #   error(x) ≈ cond(A) × ε_machine × |b| / ‖A‖
        # which can be O(1e4 Pa) for cond(A)~1e14.
        #
        # In correction form the RHS is the true residual r = b - A*x_k ≈ 0,
        # so the solve error is ε_machine times smaller.
        p_ref = float(np.mean(np.abs(p_k))) if np.any(np.abs(p_k) > 0) else 1e5
        u_ref = float(np.mean(np.abs(u_k))) if np.any(np.abs(u_k) > 0) else 1.0
        T_ref = float(np.mean(np.abs(T_k))) if np.any(np.abs(T_k) > 0) else 300.0

        x_k_vec = np.concatenate([p_k, u_k, T_k])
        r = b - A.dot(x_k_vec)          # true residual (small near equilibrium)
        dvar = solve_linear_system(A, r, p_ref=max(p_ref, 1.0),
                                         u_ref=max(u_ref, 1e-6),
                                         T_ref=max(T_ref, 1.0))

        # ----  3e. Clamp T corrections for ill-conditioned energy rows  ----
        #
        # With Path B (explicit energy flux in b), any tiny velocity variation
        # δu from the p,u solve gets amplified by H_face/dx in the deferred
        # energy residual r_e ≈ H/dx * δu.  For air cells at a water-air
        # interface, H_face ≈ 1.83e9 J/m³ while dEdT_eff ≈ 834 J/(m³K),
        # so the condition ratio H/(alpha*dEdT*dx) ≈ 4.4e6 >> 1.  The solve
        # amplifies floating-point artefacts into δT ≈ 5e-3 K.
        #
        # Fix: zero δT for cells where the energy equation is ill-conditioned.
        # With Path B the energy rows are decoupled from p,u in A (no
        # off-diagonal flux entries), so clamping δT does not affect δp/δu.
        #
        # Threshold 1e6:
        #   Air at interface  → ratio ≈ 4.4e6 → CLAMPED
        #   Interior air      → ratio ≈ 619   → not clamped
        #   Water (any)       → ratio ≈ 2027  → not clamped
        #   Smooth interface  → ratio < 1e6   → not clamped (N=10 smooth test)
        _KV_FLOOR_CL   = 718.0   # [J/(kg·K)]  matches assembly _KV_FLOOR
        _COND_THRESH_T = 1.0e6
        _alpha_bdf1    = 1.0 / dt
        _H_abs   = np.abs(H_face)
        _Hmax    = np.maximum(_H_abs[:-1], _H_abs[1:])          # (N,)
        _dEdT_cl = np.maximum(np.abs(props_k['dEdT_v']),
                              rho_k * _KV_FLOOR_CL)             # (N,)
        _ill = _Hmax > (_COND_THRESH_T * _alpha_bdf1 * _dEdT_cl * dx)
        if np.any(_ill):
            dvar[2 * N:][_ill] = 0.0

        p_new = p_k + dvar[0:N]
        u_new = u_k + dvar[N:2*N]
        T_new = T_k + dvar[2*N:3*N]

        # Physical floor
        p_new = np.maximum(p_new, _P_FLOOR)
        T_new = np.maximum(T_new, _T_FLOOR)

        # ----  3f. Convergence check  ----
        res_p = np.max(np.abs(p_new - p_k)) / (np.mean(np.abs(p_k)) + 1e-300)
        res_u = np.max(np.abs(u_new - u_k)) / (np.mean(np.abs(u_k)) + 1e-300)
        res_T = np.max(np.abs(T_new - T_k)) / (np.mean(np.abs(T_k)) + 1e-300)
        res   = max(res_p, res_u, res_T)
        res_hist.append(res)

        # Under-relaxation: blend new and old iterate to damp oscillations
        p_k = picard_omega * p_new + (1.0 - picard_omega) * p_k
        u_k = picard_omega * u_new + (1.0 - picard_omega) * u_k
        T_k = picard_omega * T_new + (1.0 - picard_omega) * T_k
        p_k = np.maximum(p_k, _P_FLOOR)
        T_k = np.maximum(T_k, _T_FLOOR)

        if res < picard_tol:
            converged = True
            break

    # Step 4: EOS consistency — not applied here.
    # The Picard result already satisfies the EOS by construction: (p^{n+1}, T^{n+1})
    # are the primitive variables from which all thermodynamic quantities are derived.
    # Applying an additional explicit conservative update then inverting the EOS would
    # conflict with the implicit Picard solution and introduce spurious pressure jumps.

    # ------------------------------------------------------------------
    # Step 5: Store face velocity for next step (MWI transient correction)
    # ------------------------------------------------------------------
    # Update face velocity with implicit MWI
    p_ext_new = apply_ghost(p_k, bc_l, bc_r)
    u_ext_new = apply_ghost_velocity(u_k, bc_l, bc_r)
    u_face_new = np.empty(N + 1)
    for f in range(N + 1):
        iL = 2 + f - 1
        iR = 2 + f
        dp = (p_ext_new[iR] - p_ext_new[iL]) / dx
        u_face_new[f] = (0.5 * (u_ext_new[iL] + u_ext_new[iR])
                         - d_hat[f] * dp)

    # ------------------------------------------------------------------
    # Build new state and aux
    # ------------------------------------------------------------------
    props_new = compute_mixture_props(p_k, u_k, T_k, psi_new, ph1, ph2)

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
        'bdf_order':     1,  # keep BDF1: BDF2 is unstable when density jumps O(1) per step
        'rho_nm1':       rho_n,
        'rhoU_nm1':      rhoU_n,
        'E_nm1':         E_total_n,
        'rho_face_acid': rho_face_acid,
    }

    info = {
        'converged':    converged,
        'picard_iters': it + 1,
        'residuals':    res_hist,
    }

    return new_state, new_aux, info
