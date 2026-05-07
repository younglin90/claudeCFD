"""Exact Riemann solver for two-fluid stiffened gas EOS.

Toro 2009 Ch 4 Riemann solver, extended for stiffened gas (Saurel 2007):
- SG: p_bar = p + pinf, isentropic p_bar/rho^gamma = const
- Use p_bar instead of p in formulas

For two-fluid: left phase and right phase with different (gamma, pinf).
The contact discontinuity separates the two phases.
Star pressure p* is physical pressure (same on both sides).
In SG formulas, use (p* + pinf_K) for phase K.
"""
import numpy as np
from scipy.optimize import brentq


def sg_sound_speed(p, rho, gamma, pinf):
    """c = sqrt(gamma * (p + pinf) / rho)"""
    return np.sqrt(gamma * (p + pinf) / rho)


def f_K(p_star, p_K, rho_K, gamma_K, pinf_K):
    """Toro 2009 f_K function for SG.
    Returns pressure jump function, positive for shock, negative for rarefaction.
    """
    A_K = 2.0 / ((gamma_K + 1.0) * rho_K)
    B_K = (gamma_K - 1.0) / (gamma_K + 1.0) * (p_K + pinf_K)
    c_K = sg_sound_speed(p_K, rho_K, gamma_K, pinf_K)

    p_star_bar = p_star + pinf_K
    p_K_bar = p_K + pinf_K

    if p_star >= p_K:  # shock
        return (p_star - p_K) * np.sqrt(A_K / (p_star_bar + B_K))
    else:  # rarefaction
        return 2.0 * c_K / (gamma_K - 1.0) * (
            (p_star_bar / p_K_bar) ** ((gamma_K - 1.0) / (2.0 * gamma_K)) - 1.0
        )


def df_K(p_star, p_K, rho_K, gamma_K, pinf_K):
    """df_K/dp* for Newton iteration."""
    A_K = 2.0 / ((gamma_K + 1.0) * rho_K)
    B_K = (gamma_K - 1.0) / (gamma_K + 1.0) * (p_K + pinf_K)
    c_K = sg_sound_speed(p_K, rho_K, gamma_K, pinf_K)

    p_star_bar = p_star + pinf_K
    p_K_bar = p_K + pinf_K

    if p_star >= p_K:
        return np.sqrt(A_K / (p_star_bar + B_K)) * (
            1.0 - (p_star - p_K) / (2.0 * (p_star_bar + B_K))
        )
    else:
        return 1.0 / (rho_K * c_K) * (p_star_bar / p_K_bar) ** (
            -(gamma_K + 1.0) / (2.0 * gamma_K)
        )


def exact_riemann_star(pL, rhoL, uL, gammaL, pinfL, pR, rhoR, uR, gammaR, pinfR):
    """Solve for star region (p*, u*) via Newton iteration."""
    # Initial guess: two-rarefaction approximation
    cL = sg_sound_speed(pL, rhoL, gammaL, pinfL)
    cR = sg_sound_speed(pR, rhoR, gammaR, pinfR)
    p_guess = max(0.5 * (pL + pR), 1e-10)

    def f_total(p):
        return f_K(p, pL, rhoL, gammaL, pinfL) + f_K(p, pR, rhoR, gammaR, pinfR) + (uR - uL)

    # Bracket search
    p_low = max(1e-10, 0.01 * min(pL, pR))
    p_high = 100.0 * max(pL, pR)

    try:
        p_star = brentq(f_total, p_low, p_high, xtol=1e-8, rtol=1e-10)
    except ValueError:
        # Newton fallback
        p_star = p_guess
        for _ in range(100):
            f = f_total(p_star)
            df = df_K(p_star, pL, rhoL, gammaL, pinfL) + df_K(p_star, pR, rhoR, gammaR, pinfR)
            dp = -f / df
            p_star += dp
            p_star = max(p_star, 1e-10)
            if abs(dp) < 1e-6 * p_star:
                break

    u_star = 0.5 * (uL + uR) + 0.5 * (
        f_K(p_star, pR, rhoR, gammaR, pinfR) - f_K(p_star, pL, rhoL, gammaL, pinfL)
    )
    return p_star, u_star


def sample_solution(x, t, x0, pL, rhoL, uL, gammaL, pinfL, pR, rhoR, uR, gammaR, pinfR):
    """Sample the exact Riemann solution at (x, t).

    Returns (rho, u, p, phase) where phase=0 for left, 1 for right.
    Two-fluid: contact discontinuity separates phases.
    """
    if t <= 0:
        if x < x0:
            return rhoL, uL, pL, 0
        else:
            return rhoR, uR, pR, 1

    p_star, u_star = exact_riemann_star(pL, rhoL, uL, gammaL, pinfL,
                                         pR, rhoR, uR, gammaR, pinfR)
    s = (x - x0) / t  # self-similar variable
    cL = sg_sound_speed(pL, rhoL, gammaL, pinfL)
    cR = sg_sound_speed(pR, rhoR, gammaR, pinfR)

    # Left wave
    if p_star >= pL:  # left shock
        SL = uL - cL * np.sqrt((gammaL + 1.0) / (2.0 * gammaL) * (p_star + pinfL) / (pL + pinfL)
                                + (gammaL - 1.0) / (2.0 * gammaL))
        if s < SL:
            return rhoL, uL, pL, 0
        # Between SL and u_star (left star region)
        rho_starL = rhoL * ((p_star + pinfL) / (pL + pinfL) + (gammaL - 1.0) / (gammaL + 1.0)) / \
                           (((gammaL - 1.0) / (gammaL + 1.0)) * (p_star + pinfL) / (pL + pinfL) + 1.0)
        if s < u_star:
            return rho_starL, u_star, p_star, 0
    else:  # left rarefaction
        SHL = uL - cL
        rho_starL = rhoL * ((p_star + pinfL) / (pL + pinfL)) ** (1.0 / gammaL)
        c_starL = sg_sound_speed(p_star, rho_starL, gammaL, pinfL)
        STL = u_star - c_starL
        if s < SHL:
            return rhoL, uL, pL, 0
        if s < STL:
            # Inside rarefaction fan
            u_fan = 2.0 / (gammaL + 1.0) * (cL + (gammaL - 1.0) / 2.0 * uL + s)
            c_fan = 2.0 / (gammaL + 1.0) * (cL + (gammaL - 1.0) / 2.0 * (uL - s))
            rho_fan = rhoL * (c_fan / cL) ** (2.0 / (gammaL - 1.0))
            p_fan = (pL + pinfL) * (c_fan / cL) ** (2.0 * gammaL / (gammaL - 1.0)) - pinfL
            return rho_fan, u_fan, p_fan, 0
        if s < u_star:
            return rho_starL, u_star, p_star, 0

    # Right wave
    if p_star >= pR:  # right shock
        SR = uR + cR * np.sqrt((gammaR + 1.0) / (2.0 * gammaR) * (p_star + pinfR) / (pR + pinfR)
                                + (gammaR - 1.0) / (2.0 * gammaR))
        rho_starR = rhoR * ((p_star + pinfR) / (pR + pinfR) + (gammaR - 1.0) / (gammaR + 1.0)) / \
                           (((gammaR - 1.0) / (gammaR + 1.0)) * (p_star + pinfR) / (pR + pinfR) + 1.0)
        if s < SR:
            return rho_starR, u_star, p_star, 1
        return rhoR, uR, pR, 1
    else:  # right rarefaction
        rho_starR = rhoR * ((p_star + pinfR) / (pR + pinfR)) ** (1.0 / gammaR)
        c_starR = sg_sound_speed(p_star, rho_starR, gammaR, pinfR)
        STR = u_star + c_starR
        SHR = uR + cR
        if s < STR:
            return rho_starR, u_star, p_star, 1
        if s < SHR:
            u_fan = 2.0 / (gammaR + 1.0) * (-cR + (gammaR - 1.0) / 2.0 * uR + s)
            c_fan = 2.0 / (gammaR + 1.0) * (cR - (gammaR - 1.0) / 2.0 * (uR - s))
            rho_fan = rhoR * (c_fan / cR) ** (2.0 / (gammaR - 1.0))
            p_fan = (pR + pinfR) * (c_fan / cR) ** (2.0 * gammaR / (gammaR - 1.0)) - pinfR
            return rho_fan, u_fan, p_fan, 1
        return rhoR, uR, pR, 1


def exact_profile(x, t, x0, **kwargs):
    """Vectorized: return rho, u, p arrays sampled along x at time t."""
    rho = np.zeros_like(x, dtype=float)
    u = np.zeros_like(x, dtype=float)
    p = np.zeros_like(x, dtype=float)
    phase = np.zeros_like(x, dtype=int)
    for i, xi in enumerate(x):
        rho[i], u[i], p[i], phase[i] = sample_solution(xi, t, x0, **kwargs)
    return rho, u, p, phase


def mixture_eos_equiv(psi, ph_air, ph_water):
    """5-eq Kapila SG mixture equivalent EOS.
    psi: water volume fraction.
    Returns (gamma_mix, pinf_mix).
    """
    a = 1.0 - psi   # alpha_air
    gm1a = ph_air['gamma'] - 1.0
    gm1w = ph_water['gamma'] - 1.0
    Gamma_inv = a / gm1a + (1.0 - a) / gm1w
    Pi = (a * ph_air['gamma'] * ph_air['pinf'] / gm1a
          + (1.0 - a) * ph_water['gamma'] * ph_water['pinf'] / gm1w)
    gamma_mix = 1.0 + 1.0 / Gamma_inv
    pinf_mix = Pi / Gamma_inv
    return gamma_mix, pinf_mix


def mixture_rh_post_shock(Ms, p0, rho0, psi, ph_air, ph_water):
    """Rankine-Hugoniot post-shock state for 5-eq SG mixture.
    Returns: (p1, rho1, u1, V_shock)
    """
    gm, pinf_m = mixture_eos_equiv(psi, ph_air, ph_water)
    p0_bar = p0 + pinf_m
    c0 = np.sqrt(gm * p0_bar / max(rho0, 1e-30))
    Vs = Ms * c0
    p1 = (2.0 * gm * Ms**2 - (gm - 1.0)) / (gm + 1.0) * p0_bar - pinf_m
    rho1 = rho0 * (gm + 1.0) * Ms**2 / ((gm - 1.0) * Ms**2 + 2.0)
    u1 = Vs * (1.0 - rho0 / rho1)
    return p1, rho1, u1, Vs


def _sg_sound_speed(p, rho, g, pi):
    return np.sqrt(g * max(p + pi, 0.0) / max(rho, 1e-30))


def sample_profile(x_arr, t, x0,
                   pL, rhoL, uL, gL, piL,
                   pR, rhoR, uR, gR, piR):
    """Sample exact 1D Riemann solution at positions x_arr, time t.
    Discontinuity at x=x0 initially.
    Based on Toro 2009 Ch.4 (SG EOS generalization).
    Returns: p_ex, rho_ex, u_ex as numpy arrays.
    """
    p_star, u_star = exact_riemann_star(
        pL, rhoL, uL, gL, piL, pR, rhoR, uR, gR, piR)

    cL = _sg_sound_speed(pL, rhoL, gL, piL)
    cR = _sg_sound_speed(pR, rhoR, gR, piR)

    # Star densities
    if p_star > pL:   # left shock
        rhoL_s = rhoL * ((p_star + piL) / (pL + piL) + (gL - 1.0) / (gL + 1.0)) / \
                        ((gL - 1.0) / (gL + 1.0) * (p_star + piL) / (pL + piL) + 1.0)
    else:             # left rarefaction
        rhoL_s = rhoL * ((p_star + piL) / (pL + piL)) ** (1.0 / gL)

    if p_star > pR:   # right shock
        rhoR_s = rhoR * ((p_star + piR) / (pR + piR) + (gR - 1.0) / (gR + 1.0)) / \
                        ((gR - 1.0) / (gR + 1.0) * (p_star + piR) / (pR + piR) + 1.0)
    else:             # right rarefaction
        rhoR_s = rhoR * ((p_star + piR) / (pR + piR)) ** (1.0 / gR)

    cL_s = _sg_sound_speed(p_star, rhoL_s, gL, piL)
    cR_s = _sg_sound_speed(p_star, rhoR_s, gR, piR)

    # Wave speeds
    if p_star > pL:
        SL = uL - cL * np.sqrt((gL + 1.0) / (2.0 * gL) * (p_star + piL) / (pL + piL)
                                + (gL - 1.0) / (2.0 * gL))
        SL_head = SL; SL_tail = SL
    else:
        SL_head = uL - cL
        SL_tail = u_star - cL_s

    if p_star > pR:
        SR = uR + cR * np.sqrt((gR + 1.0) / (2.0 * gR) * (p_star + piR) / (pR + piR)
                                + (gR - 1.0) / (2.0 * gR))
        SR_head = SR; SR_tail = SR
    else:
        SR_head = uR + cR
        SR_tail = u_star + cR_s

    t_safe = max(t, 1e-30)
    S_arr = (x_arr - x0) / t_safe

    out_p   = np.empty_like(x_arr, dtype=float)
    out_rho = np.empty_like(x_arr, dtype=float)
    out_u   = np.empty_like(x_arr, dtype=float)

    for i, S in enumerate(S_arr):
        if S < SL_head:          # undisturbed left state
            out_p[i], out_rho[i], out_u[i] = pL, rhoL, uL
        elif p_star <= pL and SL_head <= S <= SL_tail:  # left rarefaction fan
            u_fan = (2.0 / (gL + 1.0)) * (cL + (gL - 1.0) / 2.0 * uL + S)
            c_fan = u_fan - S
            c_fan = abs(c_fan)
            rho_fan = rhoL * (c_fan / cL) ** (2.0 / (gL - 1.0))
            p_fan = (pL + piL) * (rho_fan / rhoL) ** gL - piL
            out_p[i], out_rho[i], out_u[i] = p_fan, rho_fan, u_fan
        elif S < u_star:         # left star region
            out_p[i], out_rho[i], out_u[i] = p_star, rhoL_s, u_star
        elif S < SR_tail:        # right star region
            out_p[i], out_rho[i], out_u[i] = p_star, rhoR_s, u_star
        elif p_star <= pR and SR_tail <= S <= SR_head:  # right rarefaction fan
            u_fan = (2.0 / (gR + 1.0)) * (-cR + (gR - 1.0) / 2.0 * uR + S)
            c_fan = S - u_fan
            c_fan = abs(c_fan)
            rho_fan = rhoR * (c_fan / cR) ** (2.0 / (gR - 1.0))
            p_fan = (pR + piR) * (rho_fan / rhoR) ** gR - piR
            out_p[i], out_rho[i], out_u[i] = p_fan, rho_fan, u_fan
        else:                    # undisturbed right state
            out_p[i], out_rho[i], out_u[i] = pR, rhoR, uR

    return out_p, out_rho, out_u


if __name__ == '__main__':
    # Test: Phase 2-1 Denner 2018 (HP Air / LP Water)
    print("Phase 2-1 Exact Riemann (HP Air / LP Water):")
    pL, pR = 1e9, 1e4
    T0 = 300.0
    rhoL = pL / ((1.4 - 1.0) * 717.5 * T0)  # air
    rhoR = (pR + 4.4e8) / ((4.1 - 1.0) * 474.2 * T0)  # water SG
    p_star, u_star = exact_riemann_star(pL, rhoL, 0.0, 1.4, 0.0,
                                        pR, rhoR, 0.0, 4.1, 4.4e8)
    print(f"  p_star={p_star:.4e}, u_star={u_star:.3f}")

    # Phase 2-2 Yoo & Sung (HP Water / LP Air)
    print("\nPhase 2-2 Exact Riemann (HP Water / LP Air):")
    rhoL_w = 1000.0; rhoR_a = 50.0
    pL_w, pR_a = 1e9, 1e5
    p_star, u_star = exact_riemann_star(pL_w, rhoL_w, 0.0, 4.4, 6e8,
                                        pR_a, rhoR_a, 0.0, 1.4, 0.0)
    print(f"  p_star={p_star:.4e}, u_star={u_star:.3f}")
