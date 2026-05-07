# solver/denner_1d/flux/slau2.py
# SLAU2 flux scheme (Kitamura & Shima, JCP 2013)
# Parameter-free, all-speed, shock-stable AUSM-family flux.
#
# Input:  left/right primitive states (rho, u, p, H, c) at each face
# Output: F_mass, F_momentum, F_energy at each face

import numpy as np


def _mach_plus(M):
    """AUSM+ Mach splitting M+(M)."""
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (M + np.abs(M)),
                    0.25 * (M + 1.0) ** 2)


def _mach_minus(M):
    """AUSM+ Mach splitting M-(M)."""
    return np.where(np.abs(M) >= 1.0,
                    0.5 * (M - np.abs(M)),
                    -0.25 * (M - 1.0) ** 2)


def _P_plus(M):
    """AUSM+ pressure splitting P+(M)."""
    return np.where(np.abs(M) >= 1.0,
                    np.where(M > 0, 1.0, 0.0),
                    0.25 * (M + 1.0) ** 2 * (2.0 - M))


def _P_minus(M):
    """AUSM+ pressure splitting P-(M)."""
    return np.where(np.abs(M) >= 1.0,
                    np.where(M < 0, 1.0, 0.0),
                    0.25 * (M - 1.0) ** 2 * (2.0 + M))


def slau2_flux(rho_L, rho_R, u_L, u_R, p_L, p_R, H_L, H_R, c_L, c_R):
    """
    Compute SLAU2 face fluxes for 1D Euler equations.

    Parameters
    ----------
    rho_L, rho_R : float or ndarray  density [kg/m³]
    u_L, u_R     : float or ndarray  velocity [m/s]
    p_L, p_R     : float or ndarray  pressure [Pa]
    H_L, H_R     : float or ndarray  total enthalpy h + 0.5*u² [J/kg]
    c_L, c_R     : float or ndarray  speed of sound [m/s]

    Returns
    -------
    F_mass   : float or ndarray  mass flux [kg/(m²·s)]
    F_mom    : float or ndarray  momentum flux [Pa]
    F_ener   : float or ndarray  energy flux [W/m²]
    """
    rho_L = np.asarray(rho_L, dtype=float)
    rho_R = np.asarray(rho_R, dtype=float)
    u_L = np.asarray(u_L, dtype=float)
    u_R = np.asarray(u_R, dtype=float)
    p_L = np.asarray(p_L, dtype=float)
    p_R = np.asarray(p_R, dtype=float)
    H_L = np.asarray(H_L, dtype=float)
    H_R = np.asarray(H_R, dtype=float)
    c_L = np.asarray(c_L, dtype=float)
    c_R = np.asarray(c_R, dtype=float)

    # Interface speed of sound
    c_bar = 0.5 * (c_L + c_R)
    c_bar = np.maximum(c_bar, 1e-300)

    # Mach numbers
    M_L = u_L / c_bar
    M_R = u_R / c_bar

    # Average Mach and chi parameter
    M_bar = np.sqrt(0.5 * (u_L ** 2 + u_R ** 2)) / c_bar
    M_hat = np.minimum(1.0, M_bar)
    chi = (1.0 - M_hat) ** 2

    # AUSM+ velocity splits
    u_plus_L = c_bar * _mach_plus(M_L)
    u_minus_R = c_bar * _mach_minus(M_R)

    # g function (expansion/stagnation detector)
    beta_L_minus = np.maximum(np.minimum(M_L, 0.0), -1.0)
    beta_R_plus = np.minimum(np.maximum(M_R, 0.0), 1.0)
    g = -beta_L_minus * beta_R_plus

    # Modified velocity splits
    V_plus_L = (1.0 - g) * u_plus_L + g * np.abs(u_L)
    V_minus_R = (1.0 - g) * u_minus_R - g * np.abs(u_R)

    # ---- Mass flux (identical for SLAU and SLAU2) ----
    m_dot = (0.5 * (rho_L * (u_L + np.abs(V_plus_L))
                     + rho_R * (u_R - np.abs(V_minus_R)))
             - chi / (2.0 * c_bar) * (p_R - p_L))

    # ---- Pressure flux (SLAU2 version) ----
    Pp_L = _P_plus(M_L)
    Pm_R = _P_minus(M_R)

    u_bar_mag = np.sqrt(0.5 * (u_L ** 2 + u_R ** 2))

    p_face = (0.5 * (p_L + p_R)
              + 0.5 * (Pp_L - Pm_R) * (p_L - p_R)
              + u_bar_mag * (Pp_L + Pm_R - 1.0) * 0.5 * (rho_L + rho_R) * c_bar)

    # ---- Face fluxes ----
    # Upwind convected quantities
    u_up = np.where(m_dot >= 0, u_L, u_R)
    H_up = np.where(m_dot >= 0, H_L, H_R)

    F_mass = m_dot
    F_mom = m_dot * u_up + p_face
    F_ener = m_dot * H_up

    return F_mass, F_mom, F_ener
