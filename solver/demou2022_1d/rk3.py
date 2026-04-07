# solver/demou2022_1d/rk3.py
"""
Mode A: Explicit RK3 + Fractional-Step time integration.

RK3 coefficients (DENNER_SCHEME.md §3.1):
    m=1: α̃=8/15, β̃=0,      γ̃=8/15
    m=2: α̃=5/12, β̃=-17/60, γ̃=3/20
    m=3: α̃=3/4,  β̃=-5/12,  γ̃=1/3

Per sub-step (DENNER_SCHEME.md §3.2):
    1. Compute ρ, c_mix, S^(2), S^(3) from (α₁, T, u, p)
    2. α₁^{m+1}  = α₁ - dt[α̃ R_α^m + β̃ R_α^{m-1}]   (CICSAM + S^(3))
    3. T^{m+1}   = T  - dt[α̃ R_T^m + β̃ R_T^{m-1}]   (van Leer + S^(3))
    4. u*        = u  - dt[α̃ R_u^m + β̃ R_u^{m-1}]   (van Leer, no ∇p)
    5. Helmholtz → p^{m+1}
    6. u^{m+1}  = u* - γ̃ dt / ρ * ∇p^{m+1}

Inviscid (Euler): dissipation source = 0 → S^(2) terms vanish.
"""

import numpy as np

from . import eos, source_terms, boundary, cicsam, flux_limiter, spatial, helmholtz

# RK3 coefficients [m=0,1,2]
_RK3_ALPHA = np.array([8.0/15.0,  5.0/12.0,  3.0/4.0])
_RK3_BETA  = np.array([0.0,       -17.0/60.0, -5.0/12.0])
_RK3_GAMMA = np.array([8.0/15.0,  3.0/20.0,   1.0/3.0])

_NG = 2   # ghost cell count


def _compute_state(alpha1, T, p, ph1, ph2):
    """Per-phase and mixture properties at given state."""
    pp1 = eos.phase_props(p, T, ph1)
    pp2 = eos.phase_props(p, T, ph2)
    c2_mix = source_terms.mixture_sound_speed_sq(alpha1, pp1, pp2, T)
    Sa3, ST3, Sa2, ST2, Sp2 = source_terms.source_coefficients(
        alpha1, pp1, pp2, T, c2_mix)
    rho    = alpha1 * pp1['rho'] + (1.0 - alpha1) * pp2['rho']
    c_mix  = np.sqrt(np.maximum(c2_mix, 0.0))
    return rho, c_mix, c2_mix, Sa3, ST3, Sa2, ST2, Sp2


def _residuals(alpha1, T, u, p, rho, Sa3, ST3,
               dx, dt, bc_l, bc_r):
    """
    Explicit residuals R_α, R_T, R_u at current sub-step.

    R_α = ∂(α₁u)/∂x + (Sa3 - α₁) ∂u/∂x
    R_T = ∂(Tu)/∂x  + (ST3 - T)  ∂u/∂x
    R_u = u ∂u/∂x   (convective form, van Leer for u face value)

    Returns R_alpha, R_T, R_u, u_face
    """
    # Ghost extensions
    u_ext  = boundary.apply_ghost_velocity(u,      bc_l, bc_r, _NG)
    a1_ext = boundary.apply_ghost(alpha1, bc_l, bc_r, _NG)
    T_ext  = boundary.apply_ghost(T,      bc_l, bc_r, _NG)

    # Arithmetic face velocity for flux (advecting velocity)
    u_face = spatial.face_velocity(u_ext, _NG)

    # ── α₁: CICSAM face values ────────────────────────────────────
    a1_face = cicsam.cicsam_face(a1_ext, u_face, dt=dt, dx=dx, n_ghost=_NG)

    # ── T: van Leer face values ───────────────────────────────────
    T_face = flux_limiter.van_leer_face(T_ext, u_face, _NG)

    # ── u: van Leer face values (for self-advection) ──────────────
    u_vl = flux_limiter.van_leer_face(u_ext, u_face, _NG)

    div_u = spatial.divergence(u_face, dx)

    # R_α: conservative flux divergence + S^(3) divergence correction
    R_a = (spatial.divergence(a1_face * u_face, dx)
           + (Sa3 - alpha1) * div_u)

    # R_T: same structure
    R_T = (spatial.divergence(T_face * u_face, dx)
           + (ST3 - T) * div_u)

    # R_u: u ∂u/∂x ≈ divergence of (u_vl * u_face)  (non-conservative)
    R_u = spatial.divergence(u_vl * u_face, dx)

    return R_a, R_T, R_u, u_face


def step(alpha1: np.ndarray, T: np.ndarray,
         u: np.ndarray, p: np.ndarray,
         ph1: dict, ph2: dict,
         dx: float, dt: float,
         bc_l: str, bc_r: str) -> tuple:
    """
    Advance one full time step with 3-stage RK3 + fractional-step.

    Parameters
    ----------
    alpha1, T, u, p : (N,) state arrays at time level n
    ph1, ph2        : phase EOS parameter dicts
    dx, dt          : cell size, time step
    bc_l, bc_r      : boundary condition types

    Returns
    -------
    alpha1_new, T_new, u_new, p_new : (N,) updated state at n+1
    """
    N = len(alpha1)

    # Previous-stage residuals (zero at t=0 or first sub-step)
    R_a_prev = np.zeros(N)
    R_T_prev = np.zeros(N)
    R_u_prev = np.zeros(N)

    a1 = alpha1.copy()
    Tv = T.copy()
    uv = u.copy()
    pv = p.copy()

    for m in range(3):
        am = _RK3_ALPHA[m]
        bm = _RK3_BETA[m]
        gm = _RK3_GAMMA[m]
        lam = gm * dt   # effective Helmholtz time scale

        # ── State properties ──────────────────────────────────────
        (rho, c_mix, c2_mix,
         Sa3, ST3, Sa2, ST2, Sp2) = _compute_state(a1, Tv, pv, ph1, ph2)

        # ── Explicit residuals ────────────────────────────────────
        R_a, R_T, R_u, u_face = _residuals(
            a1, Tv, uv, pv, rho, Sa3, ST3, dx, dt, bc_l, bc_r)

        # Dissipation source = 0 (inviscid)
        dissip = np.zeros(N)

        # ── Update α₁ ────────────────────────────────────────────
        a1_new = a1 - dt * (am * R_a + bm * R_a_prev)
        # S^(2) dissipation contribution (zero here, keep for future viscous)
        a1_new += lam * Sa2 * dissip
        a1_new = np.clip(a1_new, 1e-8, 1.0 - 1e-8)

        # ── Update T ─────────────────────────────────────────────
        T_new = Tv - dt * (am * R_T + bm * R_T_prev)
        T_new += lam * ST2 * dissip
        T_new = np.maximum(T_new, 1e-3)

        # ── Momentum predictor u* (no pressure gradient) ──────────
        u_star = uv - dt * (am * R_u + bm * R_u_prev)

        # ── Helmholtz pressure solve ──────────────────────────────
        u_star_ext  = boundary.apply_ghost_velocity(u_star, bc_l, bc_r, _NG)
        u_star_face = spatial.face_velocity(u_star_ext, _NG)
        div_u_star  = spatial.divergence(u_star_face, dx)

        rho_ext  = boundary.apply_ghost(rho, bc_l, bc_r, _NG)
        rho_face = spatial.face_density_harmonic(rho_ext, _NG)

        p_new = helmholtz.build_and_solve(
            p_old=pv,
            rho=rho,
            c2_mix=c2_mix,
            rho_face=rho_face,
            div_u_star=div_u_star,
            Sp2=Sp2,
            dissip_source=dissip,
            lam=lam,
            dx=dx,
            bc_l=bc_l,
            bc_r=bc_r,
        )

        # ── Velocity correction ────────────────────────────────────
        p_new_ext = boundary.apply_ghost(p_new, bc_l, bc_r, _NG)
        grad_p_new = spatial.gradient_cc(p_new_ext, dx, _NG)
        u_new = u_star - lam * grad_p_new / np.maximum(rho, 1e-300)

        # ── Rotate residuals and advance state ────────────────────
        R_a_prev = R_a
        R_T_prev = R_T
        R_u_prev = R_u

        a1 = a1_new
        Tv = T_new
        uv = u_new
        pv = p_new

    return a1, Tv, uv, pv
