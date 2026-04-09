# solver/denner_1d/eos/base.py
# Ref: DENNER_SCHEME.md § 2, Eq.(5)-(9)
#
# General EOS: vectorized forward evaluation (p, T) -> thermodynamic properties.
# Delegates to EOS class objects via create_eos() factory.
# Backward compatible: ph may be a dict (NASG parameters) or an EOS instance.
#
# Phase dict keys (NASG): gamma, pinf, b, kv, eta

import numpy as np
from .eos_class import create_eos


def compute_phase_props(p, T, ph):
    """
    Compute phase thermodynamic properties via EOS class.

    Parameters
    ----------
    p : array_like
        Pressure [Pa], scalar or ndarray.
    T : array_like
        Temperature [K], scalar or ndarray.
    ph : dict or EOS
        EOS parameters (dict with keys: gamma, pinf, b, kv, eta)
        or an EOS instance directly.

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

    eos = create_eos(ph)

    rho  = eos.rho(p, T)
    c    = eos.c(p, T)
    h    = eos.h(p, T)
    E    = eos.e_vol(p, T)
    zeta = eos.drho_dp(p, T)
    phi  = eos.drho_dT(p, T)
    dEdp = eos.de_vol_dp(p, T)
    dEdT = eos.de_vol_dT(p, T)

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


def compute_mixture_props_Ns(p, u, T, phi_arr, phases, mixing='volume'):
    """
    General N_s-species mixture properties.

    Parameters
    ----------
    p, u, T  : ndarray (N,)
    phi_arr  : ndarray (N_s-1, N) — independent species fractions.
               Volume: ψ₁..ψ_{Ns-1}, ψ_{Ns} = 1 - Σψᵢ
               Mass:   Y₁..Y_{Ns-1}, Y_{Ns} = 1 - ΣYᵢ
    phases   : list of N_s EOS objects (or dicts — auto-converted)
    mixing   : 'volume' or 'mass'

    Returns
    -------
    dict with keys:
        rho, E_int, E_total, c_mix, zeta_v, phi_v,
        dEdp_v, dEdT_v, dEdu_v,
        rho_h, d_rho_h_dp_v, d_rho_h_dT_v,
        phase_props : list of per-phase dicts (from compute_phase_props)
        phi_full    : (N_s, N) complete fractions (sum=1)
        psi_full    : (N_s, N) volume fractions (derived, for mass mixing)
        Delta_rho   : list of (N,) — ∂ρ/∂φₖ for k=0..N_s-2
        d_rho_h_dphi : list of (N,) — ∂(ρh)/∂φₖ for k=0..N_s-2
    """
    N_s = len(phases)
    N = len(p)
    eos_list = [create_eos(ph) for ph in phases]

    phi_arr = np.atleast_2d(phi_arr)  # ensure (N_s-1, N)

    # Complete fractions (N_s, N)
    phi_full = np.zeros((N_s, N))
    for k in range(N_s - 1):
        phi_full[k] = phi_arr[k]
    phi_full[N_s - 1] = 1.0 - np.sum(phi_arr, axis=0)

    # Per-phase properties
    phase_props = [compute_phase_props(p, T, eos_list[k]) for k in range(N_s)]

    rho_k      = np.array([pp['rho']  for pp in phase_props])
    E_k        = np.array([pp['E']    for pp in phase_props])
    h_k        = np.array([pp['h']    for pp in phase_props])
    c_k        = np.array([pp['c']    for pp in phase_props])
    zeta_k     = np.array([pp['zeta'] for pp in phase_props])
    phi_rho_k  = np.array([pp['phi']  for pp in phase_props])
    dEdp_k     = np.array([pp['dEdp'] for pp in phase_props])
    dEdT_k     = np.array([pp['dEdT'] for pp in phase_props])

    if mixing == 'volume':
        psi = phi_full  # (N_s, N)
        # ρ = Σ ψₖρₖ
        rho    = np.sum(psi * rho_k,  axis=0)
        E_int  = np.sum(psi * E_k,    axis=0)
        zeta_v = np.sum(psi * zeta_k, axis=0)
        phi_v  = np.sum(psi * phi_rho_k, axis=0)
        dEdp_v = np.sum(psi * dEdp_k, axis=0) + 0.5 * u * u * zeta_v
        dEdT_v = np.sum(psi * dEdT_k, axis=0) + 0.5 * u * u * phi_v
        c_mix  = np.sum(psi * c_k,    axis=0)

        # ρh = Σ ψₖρₖhₖ
        rho_h = np.sum(psi * rho_k * h_k, axis=0)

        # ∂ρ/∂ψₖ = ρₖ - ρ_{Ns}  (k = 0..N_s-2)
        Delta_rho = [rho_k[k] - rho_k[N_s - 1] for k in range(N_s - 1)]

        # ∂(ρh)/∂ψₖ = ρₖhₖ - ρ_{Ns}h_{Ns}
        d_rho_h_dphi = [rho_k[k] * h_k[k] - rho_k[N_s - 1] * h_k[N_s - 1]
                        for k in range(N_s - 1)]

        # d(ρh)/dp = Σ ψₖ(ζₖhₖ + ρₖ·∂hₖ/∂p)
        dh_dp_k = np.array([np.broadcast_to(eos_list[k].dh_dp(p, T), N) for k in range(N_s)])
        d_rho_h_dp_v = np.sum(psi * (zeta_k * h_k + rho_k * dh_dp_k), axis=0)

        # d(ρh)/dT = Σ ψₖ(φₖhₖ + ρₖ·cpₖ)
        cp_k = np.array([np.broadcast_to(eos_list[k].cp(p, T), N) for k in range(N_s)])
        d_rho_h_dT_v = np.sum(psi * (phi_rho_k * h_k + rho_k * cp_k), axis=0)

        psi_full = psi

    else:  # mass
        Y = phi_full  # (N_s, N)
        # 1/ρ = Σ Yₖ/ρₖ
        inv_rho = np.sum(Y / (rho_k + 1e-300), axis=0)
        rho = 1.0 / (inv_rho + 1e-300)

        # ψₖ = ρ·Yₖ/ρₖ
        psi = rho * Y / (rho_k + 1e-300)
        psi_full = psi

        # e_mix = Σ Yₖ·eₖ (mass-weighted specific internal energy)
        e_k = E_k / (rho_k + 1e-300)
        E_int = rho * np.sum(Y * e_k, axis=0)

        # Derivatives via volume fractions
        zeta_v = np.sum(psi * zeta_k, axis=0)
        phi_v  = np.sum(psi * phi_rho_k, axis=0)
        dEdp_v = np.sum(psi * dEdp_k, axis=0) + 0.5 * u * u * zeta_v
        dEdT_v = np.sum(psi * dEdT_k, axis=0) + 0.5 * u * u * phi_v
        c_mix  = np.sum(psi * c_k,    axis=0)

        # ρh (mass-weighted): h_mix = Σ Yₖhₖ
        h_mix = np.sum(Y * h_k, axis=0)
        rho_h = rho * h_mix

        # ∂ρ/∂Yₖ = -ρ²·(1/ρₖ - 1/ρ_{Ns})
        Delta_rho = [
            -rho ** 2 * (1.0 / (rho_k[k] + 1e-300) - 1.0 / (rho_k[N_s - 1] + 1e-300))
            for k in range(N_s - 1)
        ]

        # ∂(ρh)/∂Yₖ = ρ·(hₖ - h_{Ns}) + ∂ρ/∂Yₖ · h_mix
        d_rho_h_dphi = [
            rho * (h_k[k] - h_k[N_s - 1]) + Delta_rho[k] * h_mix
            for k in range(N_s - 1)
        ]

        # d(ρh)/dp
        dh_dp_k = np.array([np.broadcast_to(eos_list[k].dh_dp(p, T), N) for k in range(N_s)])
        zeta_Y = rho * rho * np.sum(Y * zeta_k / (rho_k ** 2 + 1e-300), axis=0)
        d_rho_h_dp_v = zeta_Y * h_mix + rho * np.sum(Y * dh_dp_k, axis=0)

        # d(ρh)/dT
        cp_k = np.array([np.broadcast_to(eos_list[k].cp(p, T), N) for k in range(N_s)])
        phi_Y = rho * rho * np.sum(Y * phi_rho_k / (rho_k ** 2 + 1e-300), axis=0)
        d_rho_h_dT_v = phi_Y * h_mix + rho * np.sum(Y * cp_k, axis=0)

    E_total = E_int + 0.5 * rho * u * u
    dEdu_v = rho * u

    return {
        'rho': rho, 'E_int': E_int, 'E_total': E_total,
        'c_mix': c_mix, 'zeta_v': zeta_v, 'phi_v': phi_v,
        'dEdp_v': dEdp_v, 'dEdT_v': dEdT_v, 'dEdu_v': dEdu_v,
        'rho_h': rho_h,
        'd_rho_h_dp_v': d_rho_h_dp_v,
        'd_rho_h_dT_v': d_rho_h_dT_v,
        'phase_props': phase_props,
        'phi_full': phi_full,
        'psi_full': psi_full,
        'Delta_rho': Delta_rho,           # list of N_s-1 arrays
        'd_rho_h_dphi': d_rho_h_dphi,     # list of N_s-1 arrays
    }


def compute_mixture_props(p, u, T, psi, ph1, ph2):
    """
    Compute mixture thermodynamic properties at cell centres.

    Parameters
    ----------
    p, u, T : ndarray (N,)
    psi : ndarray (N,)       -- volume fraction of phase 1
    ph1, ph2 : dict or EOS   -- EOS parameters for each phase

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
    phi_arr = np.atleast_2d(psi)  # (1, N)
    result = compute_mixture_props_Ns(p, u, T, phi_arr, [ph1, ph2], mixing='volume')

    pp = result['phase_props']
    rho1 = pp[0]['rho']
    rho2 = pp[1]['rho']
    E1   = pp[0]['E']
    E2   = pp[1]['E']

    return {
        'rho1': rho1, 'rho2': rho2,
        'E1': E1, 'E2': E2,
        'c1': pp[0]['c'], 'c2': pp[1]['c'],
        'zeta1': pp[0]['zeta'], 'zeta2': pp[1]['zeta'],
        'phi1':  pp[0]['phi'],  'phi2':  pp[1]['phi'],
        'dEdp1': pp[0]['dEdp'], 'dEdp2': pp[1]['dEdp'],
        'dEdT1': pp[0]['dEdT'], 'dEdT2': pp[1]['dEdT'],
        'rho':     result['rho'],
        'E_int':   result['E_int'],
        'E_total': result['E_total'],
        'c_mix':   result['c_mix'],
        'zeta_v':  result['zeta_v'],
        'phi_v':   result['phi_v'],
        'dEdp_v':  result['dEdp_v'],
        'dEdT_v':  result['dEdT_v'],
        'dEdu_v':  result['dEdu_v'],
        'Delta_rho_psi': result['Delta_rho'][0],
        'dEdpsi': (E1 - E2) + 0.5 * u * u * result['Delta_rho'][0],
        # Denner 2018 ρh quantities
        'rho_h':          result['rho_h'],
        'd_rho_h_dp_v':   result['d_rho_h_dp_v'],
        'd_rho_h_dT_v':   result['d_rho_h_dT_v'],
        'd_rho_h_dpsi':   result['d_rho_h_dphi'][0],
    }


def compute_mixture_props_Y(p, u, T, Y, ph1, ph2):
    """
    Compute mixture thermodynamic properties using mass fraction Y (phase 1).

    Parameters
    ----------
    p, u, T : ndarray (N,)
    Y       : ndarray (N,)       -- mass fraction of phase 1
    ph1, ph2 : dict or EOS       -- EOS parameters for each phase

    Returns
    -------
    dict with same keys as compute_mixture_props, plus 'rho1', 'rho2',
    'd_rho_dY', 'd_rho_h_dY'.
    """
    phi_arr = np.atleast_2d(Y)  # (1, N)
    result = compute_mixture_props_Ns(p, u, T, phi_arr, [ph1, ph2], mixing='mass')

    pp = result['phase_props']
    rho1 = pp[0]['rho']
    rho2 = pp[1]['rho']
    E1   = pp[0]['E']
    E2   = pp[1]['E']

    # psi = volume fraction of phase 1 (derived inside Ns function)
    psi = result['psi_full'][0]
    Delta_rho_psi = rho1 - rho2
    dEdpsi = (E1 - E2) + 0.5 * u * u * Delta_rho_psi

    return {
        'rho1': rho1, 'rho2': rho2,
        'E1': E1, 'E2': E2,
        'c1': pp[0]['c'], 'c2': pp[1]['c'],
        'zeta1': pp[0]['zeta'], 'zeta2': pp[1]['zeta'],
        'phi1':  pp[0]['phi'],  'phi2':  pp[1]['phi'],
        'dEdp1': pp[0]['dEdp'], 'dEdp2': pp[1]['dEdp'],
        'dEdT1': pp[0]['dEdT'], 'dEdT2': pp[1]['dEdT'],
        'rho':     result['rho'],
        'E_int':   result['E_int'],
        'E_total': result['E_total'],
        'c_mix':   result['c_mix'],
        'zeta_v':  result['zeta_v'],
        'phi_v':   result['phi_v'],
        'dEdp_v':  result['dEdp_v'],
        'dEdT_v':  result['dEdT_v'],
        'dEdu_v':  result['dEdu_v'],
        'Delta_rho_psi': Delta_rho_psi,
        'dEdpsi':        dEdpsi,
        # Denner 2018 ρh quantities
        'rho_h':          result['rho_h'],
        'd_rho_h_dp_v':   result['d_rho_h_dp_v'],
        'd_rho_h_dT_v':   result['d_rho_h_dT_v'],
        'd_rho_dY':       result['Delta_rho'][0],
        'd_rho_h_dY':     result['d_rho_h_dphi'][0],
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

    # Create EOS objects once outside the Newton loop
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    for _ in range(max_iter):
        props1 = compute_phase_props(p, T, eos1)
        props2 = compute_phase_props(p, T, eos2)
        h_static = Y * props1['h'] + (1.0 - Y) * props2['h']
        # dh_static/dT = Y*cp₁ + (1-Y)*cp₂  (general EOS)
        dh_dT = Y * eos1.cp(p, T) + (1.0 - Y) * eos2.cp(p, T)

        residual = h_static - h_target
        dT = -residual / (dh_dT + 1e-300)
        T = np.maximum(T + dT, 1.0)

        if np.max(np.abs(dT)) < tol:
            break

    return T


# ---------------------------------------------------------------------------
# N_s-species helpers
# ---------------------------------------------------------------------------

def compute_specific_total_enthalpy_Ns(p, u, T, phi_arr, phases, mixing='volume'):
    """N_s-species total specific enthalpy  H = rho_h/rho + ½u²."""
    result = compute_mixture_props_Ns(p, u, T, phi_arr, phases, mixing=mixing)
    rho   = result['rho']
    rho_h = result['rho_h']
    return rho_h / (rho + 1e-300) + 0.5 * u * u


def recover_T_from_h_Ns(h_total, u, p, phi_arr, phases, mixing='volume',
                         T_guess=None, tol=1e-10, max_iter=50):
    """
    Newton: find T such that h_static(p, T, phi) + ½u² = h_total.
    Works element-wise on arrays.  Supports N_s species.
    """
    h_target = h_total - 0.5 * u * u
    N = len(p)

    if T_guess is None:
        T = np.full(N, 300.0)
    else:
        T = np.asarray(T_guess, dtype=float).copy()

    eos_list = [create_eos(ph) for ph in phases]
    N_s = len(phases)

    phi_arr2 = np.atleast_2d(phi_arr)  # (N_s-1, N)

    # Complete fractions (N_s, N)
    phi_full = np.zeros((N_s, N))
    for k in range(N_s - 1):
        phi_full[k] = phi_arr2[k]
    phi_full[N_s - 1] = 1.0 - np.sum(phi_arr2, axis=0)

    for _ in range(max_iter):
        h_k_arr  = np.array([eos_list[k].h(p, T)  for k in range(N_s)])
        cp_k_arr = np.array([eos_list[k].cp(p, T) for k in range(N_s)])

        if mixing == 'volume':
            rho_k_arr  = np.array([eos_list[k].rho(p, T)      for k in range(N_s)])
            phi_rho_k  = np.array([eos_list[k].drho_dT(p, T)  for k in range(N_s)])
            rho        = np.sum(phi_full * rho_k_arr, axis=0)
            rho_h      = np.sum(phi_full * rho_k_arr * h_k_arr, axis=0)
            h_static   = rho_h / (rho + 1e-300)
            d_rho_h_dT = np.sum(phi_full * (phi_rho_k * h_k_arr + rho_k_arr * cp_k_arr), axis=0)
            phi_v      = np.sum(phi_full * phi_rho_k, axis=0)
            dh_dT      = (d_rho_h_dT * rho - rho_h * phi_v) / (rho * rho + 1e-300)
        else:
            h_static = np.sum(phi_full * h_k_arr,  axis=0)
            dh_dT    = np.sum(phi_full * cp_k_arr, axis=0)

        residual = h_static - h_target
        dT = -residual / (dh_dT + 1e-300)
        T = np.maximum(T + dT, 1.0)

        if np.max(np.abs(dT)) < tol:
            break

    return T
