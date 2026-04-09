# solver/denner_1d/eos/base.py
# Ref: DENNER_SCHEME.md § 2, Eq.(5)-(9)
#
# NASG EOS: vectorized forward evaluation (p, T) -> thermodynamic properties.
# Ideal gas is NASG with pinf=0, b=0, eta=0.
#
# Phase dict keys: gamma, pinf, b, kv, eta

import numpy as np


def compute_phase_props(p, T, ph):
    """
    Compute NASG phase thermodynamic properties.

    Parameters
    ----------
    p : array_like
        Pressure [Pa], scalar or ndarray.
    T : array_like
        Temperature [K], scalar or ndarray.
    ph : dict
        EOS parameters with keys: gamma, pinf, b, kv, eta.

    Returns
    -------
    dict with keys:
        rho   : density [kg/m³]
        c     : speed of sound [m/s]
        h     : specific enthalpy [J/kg]
        E     : internal energy density (per unit volume) [J/m³]
        zeta  : ∂ρ/∂p |_T  [kg/(m³·Pa)]
        phi   : ∂ρ/∂T |_p  [kg/(m³·K)]
        dEdp  : ∂E/∂p |_T  [J/(m³·Pa)]
        dEdT  : ∂E/∂T |_p  [J/(m³·K)]
    """
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)

    gamma = float(ph['gamma'])
    pinf  = float(ph['pinf'])
    b     = float(ph['b'])
    kv    = float(ph['kv'])
    eta   = float(ph['eta'])

    # Eq. (5): A_k = kv*T*(gamma-1) + b*(p+pinf)
    A = kv * T * (gamma - 1.0) + b * (p + pinf)

    # rho_k = (p + pinf) / A   [Eq. 5]
    rho = (p + pinf) / A

    # co-volume factor
    one_minus_b_rho = 1.0 - b * rho  # = kv*T*(gamma-1)/A

    # speed of sound [Eq. 7]
    c = np.sqrt(gamma * (p + pinf) / (rho * one_minus_b_rho + 1e-300))

    # specific enthalpy [Eq. 7]
    h = gamma * kv * T + b * p + eta

    # internal energy density E_k = rho*(kv*T + eta) + pinf*(1 - rho*b)/(gamma-1)  [Eq. 6]
    E = rho * (kv * T + eta) + pinf * one_minus_b_rho / (gamma - 1.0)

    # TDU derivatives [Eq. 8a, 8b]
    A2 = A * A + 1e-300
    zeta = kv * T * (gamma - 1.0) / A2           # ∂ρ/∂p |_T
    phi  = -(p + pinf) * kv * (gamma - 1.0) / A2  # ∂ρ/∂T |_p

    # Energy derivatives [Eq. 9a, 9b]
    coef = kv * T + eta - pinf * b / (gamma - 1.0)
    dEdp = coef * zeta
    dEdT = rho * kv + coef * phi

    return {
        'rho':  rho,
        'c':    c,
        'h':    h,
        'E':    E,
        'zeta': zeta,
        'phi':  phi,
        'dEdp': dEdp,
        'dEdT': dEdT,
    }


def compute_mixture_props(p, u, T, psi, ph1, ph2):
    """
    Compute mixture thermodynamic properties at cell centres.

    Parameters
    ----------
    p, u, T : ndarray (N,)
    psi : ndarray (N,)  -- volume fraction of phase 1
    ph1, ph2 : dict     -- EOS parameters for each phase

    Returns
    -------
    dict with keys:
        rho1, rho2, E1, E2 : per-phase densities and internal energy densities
        c1, c2             : per-phase sound speeds
        zeta1, zeta2, phi1, phi2
        dEdp1, dEdT1, dEdp2, dEdT2
        rho                : mixture density
        E_int              : mixture internal energy density (no KE)
        E_total            : total energy density (with KE)
        c_mix              : mixture sound speed (volume-averaged)
        zeta_v, phi_v      : mixture density derivatives
        dEdp_v, dEdT_v     : mixture energy derivatives (including KE)
        dEdu_v             : ∂E_total/∂u = rho*u
        Delta_rho_psi      : rho1 - rho2
        dEdpsi             : ∂E_total/∂psi
    """
    props1 = compute_phase_props(p, T, ph1)
    props2 = compute_phase_props(p, T, ph2)

    rho1 = props1['rho']
    rho2 = props2['rho']
    E1   = props1['E']
    E2   = props2['E']

    rho   = psi * rho1 + (1.0 - psi) * rho2
    E_int = psi * E1   + (1.0 - psi) * E2
    E_total = E_int + 0.5 * rho * u * u

    # Mixture density derivatives
    zeta_v = psi * props1['zeta'] + (1.0 - psi) * props2['zeta']
    phi_v  = psi * props1['phi']  + (1.0 - psi) * props2['phi']

    # Mixture energy derivatives (total energy includes KE)
    dEdp_v = (psi * props1['dEdp'] + (1.0 - psi) * props2['dEdp']
              + 0.5 * u * u * zeta_v)
    dEdT_v = (psi * props1['dEdT'] + (1.0 - psi) * props2['dEdT']
              + 0.5 * u * u * phi_v)
    dEdu_v = rho * u  # ∂E_total/∂u

    # Mixture sound speed (volume-fraction weighted)
    c_mix = psi * props1['c'] + (1.0 - psi) * props2['c']

    Delta_rho_psi = rho1 - rho2
    dEdpsi = (E1 - E2) + 0.5 * u * u * Delta_rho_psi

    # ----------------------------------------------------------------
    # Denner 2018 ρh formulation (for ACID energy equation)
    # rho_h = ψ * ρ₁*h₁ + (1-ψ) * ρ₂*h₂   [J/m³]
    # ----------------------------------------------------------------
    h1 = props1['h']
    h2 = props2['h']
    rho_h = psi * rho1 * h1 + (1.0 - psi) * rho2 * h2
    d_rho_h_dpsi = rho1 * h1 - rho2 * h2  # d(ρh_static)/dψ

    # d(ρh)/dp = ψ*(ζ₁*h₁ + ρ₁*b₁) + (1-ψ)*(ζ₂*h₂ + ρ₂*b₂)
    # where ∂h/∂p|_T = b  (NASG: h = γ*κᵥ*T + b*p + η → ∂h/∂p=b)
    b1 = float(ph1['b'])
    b2 = float(ph2['b'])
    d_rho_h_dp_v = (psi * (props1['zeta'] * h1 + rho1 * b1) +
                    (1.0 - psi) * (props2['zeta'] * h2 + rho2 * b2))

    # d(ρh)/dT = ψ*(φ₁*h₁ + ρ₁*γ₁*κᵥ₁) + (1-ψ)*(φ₂*h₂ + ρ₂*γ₂*κᵥ₂)
    # where ∂h/∂T|_p = γ*κᵥ  (NASG: h = γ*κᵥ*T + b*p + η → ∂h/∂T=γ*κᵥ)
    g1kv1 = float(ph1['gamma']) * float(ph1['kv'])
    g2kv2 = float(ph2['gamma']) * float(ph2['kv'])
    d_rho_h_dT_v = (psi * (props1['phi'] * h1 + rho1 * g1kv1) +
                    (1.0 - psi) * (props2['phi'] * h2 + rho2 * g2kv2))

    return {
        'rho1': rho1, 'rho2': rho2,
        'E1': E1, 'E2': E2,
        'c1': props1['c'], 'c2': props2['c'],
        'zeta1': props1['zeta'], 'zeta2': props2['zeta'],
        'phi1':  props1['phi'],  'phi2':  props2['phi'],
        'dEdp1': props1['dEdp'], 'dEdp2': props2['dEdp'],
        'dEdT1': props1['dEdT'], 'dEdT2': props2['dEdT'],
        'rho': rho,
        'E_int': E_int,
        'E_total': E_total,
        'c_mix': c_mix,
        'zeta_v': zeta_v,
        'phi_v': phi_v,
        'dEdp_v': dEdp_v,
        'dEdT_v': dEdT_v,
        'dEdu_v': dEdu_v,
        'Delta_rho_psi': Delta_rho_psi,
        'dEdpsi': dEdpsi,
        # Denner 2018 ρh quantities
        'rho_h':          rho_h,
        'd_rho_h_dp_v':   d_rho_h_dp_v,
        'd_rho_h_dT_v':   d_rho_h_dT_v,
        'd_rho_h_dpsi':   d_rho_h_dpsi,
    }


def compute_mixture_props_Y(p, u, T, Y, ph1, ph2):
    """
    Compute mixture thermodynamic properties using mass fraction Y (phase 1).

    Parameters
    ----------
    p, u, T : ndarray (N,)
    Y       : ndarray (N,)  -- mass fraction of phase 1
    ph1, ph2 : dict         -- EOS parameters for each phase

    Returns
    -------
    dict with same keys as compute_mixture_props, plus 'rho1', 'rho2'.
    """
    props1 = compute_phase_props(p, T, ph1)
    props2 = compute_phase_props(p, T, ph2)

    rho1 = props1['rho']
    rho2 = props2['rho']
    E1   = props1['E']
    E2   = props2['E']

    # Mixture density: harmonic in volume fractions → 1/rho = Y/rho1 + (1-Y)/rho2
    inv_rho = Y / (rho1 + 1e-300) + (1.0 - Y) / (rho2 + 1e-300)
    rho = 1.0 / (inv_rho + 1e-300)

    # Mass-weighted specific internal energy: e_mix = Y*e1 + (1-Y)*e2
    e1 = E1 / (rho1 + 1e-300)
    e2 = E2 / (rho2 + 1e-300)
    E_int   = rho * (Y * e1 + (1.0 - Y) * e2)
    E_total = E_int + 0.5 * rho * u * u

    # ψ₁ = volume fraction of phase 1 = ρ·Y/ρ₁
    psi = rho * Y / (rho1 + 1e-300)

    # Mixture density derivatives (via volume fractions)
    zeta_v = psi * props1['zeta'] + (1.0 - psi) * props2['zeta']
    phi_v  = psi * props1['phi']  + (1.0 - psi) * props2['phi']

    # Mixture energy derivatives (total energy includes KE)
    dEdp_v = (psi * props1['dEdp'] + (1.0 - psi) * props2['dEdp']
              + 0.5 * u * u * zeta_v)
    dEdT_v = (psi * props1['dEdT'] + (1.0 - psi) * props2['dEdT']
              + 0.5 * u * u * phi_v)
    dEdu_v = rho * u

    # Mixture sound speed (volume-fraction weighted)
    c_mix = psi * props1['c'] + (1.0 - psi) * props2['c']

    Delta_rho_psi = rho1 - rho2
    dEdpsi = (E1 - E2) + 0.5 * u * u * Delta_rho_psi

    # ρh formulation (for ACID energy equation) — mass-weighted h
    h1 = props1['h']
    h2 = props2['h']
    h_mix = Y * h1 + (1.0 - Y) * h2             # mass-weighted static enthalpy
    rho_h = rho * h_mix
    d_rho_dY = -rho**2 * (1.0 / (rho1 + 1e-300) - 1.0 / (rho2 + 1e-300))
    d_rho_h_dY = rho * (h1 - h2) + d_rho_dY * h_mix

    b1 = float(ph1['b'])
    b2 = float(ph2['b'])
    g1kv1 = float(ph1['gamma']) * float(ph1['kv'])
    g2kv2 = float(ph2['gamma']) * float(ph2['kv'])

    # d(ρh)/dp: ρ² · (Y·ζ₁/ρ₁² + (1-Y)·ζ₂/ρ₂²) · h_mix + ρ·(Y·b₁ + (1-Y)·b₂)
    zeta_Y = rho * rho * (Y * props1['zeta'] / (rho1 * rho1 + 1e-300)
                          + (1.0 - Y) * props2['zeta'] / (rho2 * rho2 + 1e-300))
    d_rho_h_dp_v = zeta_Y * h_mix + rho * (Y * b1 + (1.0 - Y) * b2)

    # d(ρh)/dT: ρ² · (Y·φ₁/ρ₁² + (1-Y)·φ₂/ρ₂²) · h_mix + ρ·(Y·γ₁κᵥ₁ + (1-Y)·γ₂κᵥ₂)
    phi_Y = rho * rho * (Y * props1['phi'] / (rho1 * rho1 + 1e-300)
                         + (1.0 - Y) * props2['phi'] / (rho2 * rho2 + 1e-300))
    d_rho_h_dT_v = phi_Y * h_mix + rho * (Y * g1kv1 + (1.0 - Y) * g2kv2)

    return {
        'rho1': rho1, 'rho2': rho2,
        'E1': E1, 'E2': E2,
        'c1': props1['c'], 'c2': props2['c'],
        'zeta1': props1['zeta'], 'zeta2': props2['zeta'],
        'phi1':  props1['phi'],  'phi2':  props2['phi'],
        'dEdp1': props1['dEdp'], 'dEdp2': props2['dEdp'],
        'dEdT1': props1['dEdT'], 'dEdT2': props2['dEdT'],
        'rho': rho,
        'E_int': E_int,
        'E_total': E_total,
        'c_mix': c_mix,
        'zeta_v': zeta_v,
        'phi_v': phi_v,
        'dEdp_v': dEdp_v,
        'dEdT_v': dEdT_v,
        'dEdu_v': dEdu_v,
        'Delta_rho_psi': Delta_rho_psi,
        'dEdpsi': dEdpsi,
        # Denner 2018 ρh quantities
        'rho_h':          rho_h,
        'd_rho_h_dp_v':   d_rho_h_dp_v,
        'd_rho_h_dT_v':   d_rho_h_dT_v,
        'd_rho_dY':       d_rho_dY,
        'd_rho_h_dY':     d_rho_h_dY,
    }


def compute_specific_total_enthalpy(p, u, T, psi, ph1, ph2):
    """h_total = rho_h/rho + 0.5*u^2  (total specific enthalpy)."""
    props = compute_mixture_props(p, u, T, psi, ph1, ph2)
    rho = props['rho']
    rho_h = props['rho_h']
    h_static = rho_h / (rho + 1e-300)
    return h_static + 0.5 * u * u


def compute_specific_total_enthalpy_Y(p, u, T, Y, ph1, ph2):
    """h_total = Y*h1 + (1-Y)*h2 + 0.5*u^2  (mass-weighted total specific enthalpy)."""
    props1 = compute_phase_props(p, T, ph1)
    props2 = compute_phase_props(p, T, ph2)
    h_static = Y * props1['h'] + (1.0 - Y) * props2['h']
    return h_static + 0.5 * u * u


def recover_T_from_h(h_total, u, p, psi, ph1, ph2, T_guess=None, tol=1e-10, max_iter=50):
    """
    Newton iteration: find T such that h_static(p,T,psi) + 0.5*u^2 = h_total.
    Works element-wise on arrays.
    """
    h_target = h_total - 0.5 * u * u

    if T_guess is None:
        T = np.full_like(np.asarray(p, dtype=float), 300.0)
    else:
        T = np.asarray(T_guess, dtype=float).copy()

    for _ in range(max_iter):
        props = compute_mixture_props(p, u, T, psi, ph1, ph2)
        rho = props['rho']
        rho_h = props['rho_h']
        h_static = rho_h / (rho + 1e-300)

        d_rho_h_dT = props['d_rho_h_dT_v']
        phi = props['phi_v']
        dh_dT = (d_rho_h_dT * rho - rho_h * phi) / (rho * rho + 1e-300)

        residual = h_static - h_target
        dT = -residual / (dh_dT + 1e-300)
        T = np.maximum(T + dT, 1.0)

        if np.max(np.abs(dT)) < tol:
            break

    return T


def recover_T_from_h_Y(h_total, u, p, Y, ph1, ph2, T_guess=None, tol=1e-10, max_iter=50):
    """
    Newton iteration: find T such that Y*h1(p,T) + (1-Y)*h2(p,T) + 0.5*u^2 = h_total.
    Works element-wise on arrays.
    """
    h_target = h_total - 0.5 * u * u

    if T_guess is None:
        T = np.full_like(np.asarray(p, dtype=float), 300.0)
    else:
        T = np.asarray(T_guess, dtype=float).copy()

    g1kv1 = float(ph1['gamma']) * float(ph1['kv'])
    g2kv2 = float(ph2['gamma']) * float(ph2['kv'])

    for _ in range(max_iter):
        props1 = compute_phase_props(p, T, ph1)
        props2 = compute_phase_props(p, T, ph2)
        h_static = Y * props1['h'] + (1.0 - Y) * props2['h']
        # dh_static/dT = Y*γ₁κᵥ₁ + (1-Y)*γ₂κᵥ₂
        dh_dT = Y * g1kv1 + (1.0 - Y) * g2kv2

        residual = h_static - h_target
        dT = -residual / (dh_dT + 1e-300)
        T = np.maximum(T + dT, 1.0)

        if np.max(np.abs(dT)) < tol:
            break

    return T
