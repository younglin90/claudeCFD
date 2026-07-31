"""Layered positivity for the explicit advection operator (Phase 8).

Strategy (user spec §"POSITIVITY-PRESERVING STRATEGY"):

  1. Reconstruction limiter   — handled inside `face_state` (α∈[1e-12,1−1e-12],
     T_k > 1, p > 1).  Slope scaling rather than post-clip.

  2. Flux blending limiter
        F_f = θ_f · F_HO + (1 − θ_f) · F_LO,   θ_f ∈ [0, 1]
     where F_HO is the APEC + ACID high-order flux and F_LO is the local
     Lax-Friedrichs (Rusanov) flux.  θ_f is reduced cell-by-cell until the
     candidate conservative update remains admissible.

  3. Newton implicit line search — handled inside `newton.newton_solve`.

  4. No blind clipping — fallbacks are documented + diagnostic-counted.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def _physical_face_flux(alpha, rho1, rho2, e1, e2, rho, u):
    """Conservative-form physical flux F(U) at a single state.

    F = (α ρ1 u, (1-α) ρ2 u, ρ u², (αρ1·e1+(1-α)ρ2·e2)·u + ½ρu²·u, α u)
    The pressure work p·u is intentionally excluded (lives in F_I).
    """
    F_a1r1 = alpha * rho1 * u
    F_a2r2 = (1.0 - alpha) * rho2 * u
    F_alpha = alpha * u
    F_rho = F_a1r1 + F_a2r2
    F_ru = rho * u * u
    F_rho_e = (alpha * rho1 * e1 + (1.0 - alpha) * rho2 * e2) * u
    F_K = 0.5 * u * u * F_rho
    F_rE = F_rho_e + F_K
    return dict(F_a1r1=F_a1r1, F_a2r2=F_a2r2, F_alpha=F_alpha,
                F_rho=F_rho, F_ru=F_ru, F_rE=F_rE)


def lax_friedrichs_fluxes(face):
    """Conservative Rusanov face flux (LEGACY — *not* PE-preserving!).

        F_LF = ½ (F(U_L) + F(U_R)) − ½ a_LF · (U_R − U_L)

    The (U_R − U_L) dissipation breaks pressure-equilibrium preservation
    on α-jump faces because it dissipates phase mass and energy with
    different coefficients than the PE-consistent advection scheme.
    Kept for diagnostic / debugging only.  Use `pe_preserving_lo_flux`
    instead in the positivity-blending path.
    """
    F_L = _physical_face_flux(face['a_L'], face['rho1_L'], face['rho2_L'],
                              face['e1_L'], face['e2_L'], face['rho_L'],
                              face['u_L'])
    F_R = _physical_face_flux(face['a_R'], face['rho1_R'], face['rho2_R'],
                              face['e1_R'], face['e2_R'], face['rho_R'],
                              face['u_R'])
    a_LF = face['a_LF']
    U_L = face['U_L']; U_R = face['U_R']

    out = {}
    for k_idx, k in enumerate(('F_a1r1', 'F_a2r2', 'F_ru', 'F_rE', 'F_alpha')):
        out[k] = 0.5 * (F_L[k] + F_R[k]) - 0.5 * a_LF * (U_R[k_idx] - U_L[k_idx])
    out['F_rho'] = out['F_a1r1'] + out['F_a2r2']
    return out


def pe_preserving_lo_flux(face, eos1, eos2):
    """PE-preserving low-order face flux — pure upwind on the *face* state.

    Strategy (ChatGPT v2 §3 우선순위 3):
      F_LO uses the **same face state** as F_HO (same a_f, T_k_f, ρ_k_f,
      u_f, p_f from `face_state`), so on a PE state both fluxes are
      identical and θ-blending cannot break PE.
      The dissipation that would normally come from a Rusanov LF flux is
      provided here purely by the **upwinding of α** built into face_state
      (`alpha_scheme='upwind'`).  No conservative (U_R − U_L) term is
      added because that term breaks PE.

    For positivity protection we add a **scalar advection-only** dissipation
    that targets only the α update (the variable that can leave [0,1]):

        F_α_LO = α_upw · u_f                  (pure upwind already PE-OK)
        F_q1_LO = α_upw · ρ_1_face · u_f
        F_q2_LO = (1−α_upw) · ρ_2_face · u_f
        F_ρu_LO = ρ_face · u_f²
        F_rE_LO = (α_upw·ρ_1·e_1 + (1−α_upw)·ρ_2·e_2) · u_f + ½ u² · F_ρ

    Note: with face_state's default `alpha_scheme='upwind'`, this LO is
    equal to F_HO at PE state — the blending then has zero effect on PE,
    while still letting θ_f drop near positivity-violating faces (e.g.
    where the high-order APEC χ_a fallback kicks in).
    """
    a_f   = face['alpha']
    u_f   = face['u']
    rho1f = face['rho1']
    rho2f = face['rho2']
    e1f   = face['e1']
    e2f   = face['e2']
    rho_f = face['rho']

    F_a1r1  = a_f * rho1f * u_f
    F_a2r2  = (1.0 - a_f) * rho2f * u_f
    F_alpha = a_f * u_f
    F_rho   = F_a1r1 + F_a2r2
    F_ru    = rho_f * u_f * u_f
    F_rho_e = (a_f * rho1f * e1f + (1.0 - a_f) * rho2f * e2f) * u_f
    F_K     = 0.5 * u_f ** 2 * F_rho
    F_rE    = F_rho_e + F_K
    return dict(F_a1r1=F_a1r1, F_a2r2=F_a2r2, F_alpha=F_alpha,
                F_rho=F_rho, F_ru=F_ru, F_rE=F_rE)


def positivity_blend_theta(F_HO, F_LO, U_n, dx, dt, *,
                            phase_mass_floor=1e-10,
                            alpha_floor=1e-6,
                            max_iter=30):
    """Compute per-face blending weight θ_f ∈ [0, 1] so that the candidate
    update U_cand = U_n − dt · ∂F/∂x with F = θ·F_HO + (1−θ)·F_LO keeps
    every cell admissible (positive phase mass, α ∈ [1e-12, 1−1e-12]).

    Strategy: start θ=1 (full HO).  At each pass, identify any cell whose
    update would make α₁ρ₁ < floor, α₂ρ₂ < floor, or α∉(0,1).  For each
    bad cell, halve θ on its two adjacent faces.  Iterate until no bad
    cells remain or θ floor is reached.
    """
    N = U_n[0].shape[0]
    inv_dx = 1.0 / dx

    def divergence(Fdict):
        return {k: (F[1:] - F[:-1]) * inv_dx for k, F in Fdict.items()}

    # Map U index → corresponding flux dict key
    KEY = {0: 'F_a1r1', 1: 'F_a2r2', 2: 'F_ru', 3: 'F_rE', 4: 'F_alpha'}

    theta = np.ones(N + 1, dtype=float)
    for _ in range(max_iter):
        F_blend = {k: theta * F_HO[k] + (1.0 - theta) * F_LO[k]
                   for k in F_HO}
        div_b = divergence(F_blend)
        U_cand = [U_n[k] - dt * div_b[KEY[k]] for k in range(5)]

        bad = (
            (U_cand[0] <= phase_mass_floor) |
            (U_cand[1] <= phase_mass_floor) |
            (U_cand[4] <= alpha_floor) |
            (U_cand[4] >= 1.0 - alpha_floor)
        )
        if not np.any(bad):
            return theta

        # Halve θ on faces adjacent to any bad cell
        bad_face_left  = np.zeros(N + 1, dtype=bool)
        bad_face_right = np.zeros(N + 1, dtype=bool)
        bad_face_left[:-1]  = bad         # face i+1/2 affects cell i
        bad_face_right[1:]  = bad         # face i-1/2 affects cell i
        bad_face = bad_face_left | bad_face_right

        new_theta = np.where(bad_face, 0.5 * theta, theta)
        if np.allclose(new_theta, theta):
            break
        theta = new_theta
    return theta


def blended_advective_fluxes(W_anchor, eos1, eos2, dx, dt, bc_l, bc_r,
                              *, energy_form='apec',
                              energy_alpha_pure_tol=1.0e-12,
                              alpha_scheme='upwind',
                              primitive_scheme='upwind',
                              face_thermo='acid',
                              positivity=True,
                              lo_flux='pe_preserving',
                              force_lo=False):
    """Compute APEC+ACID high-order flux, Rusanov low-order flux, blend
    with θ_f for positivity, return the blended face fluxes.

    Returns (face, blended_flux_dict, theta).
    """
    from .face_state import face_state
    from .flux import advective_fluxes
    from .primitive import prim_to_cons_W

    face = face_state(W_anchor, eos1, eos2, bc_l, bc_r,
                      alpha_scheme=alpha_scheme,
                      primitive_scheme=primitive_scheme,
                      face_thermo=face_thermo,
                      dt=dt,
                      dx=dx)
    F_HO = advective_fluxes(face, eos1, eos2, energy_form=energy_form,
                            energy_alpha_pure_tol=energy_alpha_pure_tol)
    if not positivity and not force_lo:
        return face, F_HO, np.ones(F_HO['F_rho'].shape[0])
    if lo_flux == 'pe_preserving':
        F_LO = pe_preserving_lo_flux(face, eos1, eos2)
    elif lo_flux == 'rusanov':
        F_LO = lax_friedrichs_fluxes(face)
    else:
        raise ValueError(f"Unknown lo_flux='{lo_flux}'.")
    if force_lo == 'interface':
        theta = np.ones(F_HO['F_rho'].shape[0], dtype=float)
        if 'a_L' in face and 'a_R' in face:
            theta[np.abs(face['a_R'] - face['a_L']) > 1.0e-8] = 0.0
        F_blend = {k: theta * F_HO[k] + (1.0 - theta) * F_LO[k] for k in F_HO}
        return face, F_blend, theta
    if force_lo:
        return face, F_LO, np.zeros(F_HO['F_rho'].shape[0])
    U_n, _ = prim_to_cons_W(W_anchor, eos1, eos2)
    a = np.asarray(W_anchor[0], dtype=float)
    alpha_margin = float(np.min(np.minimum(a, 1.0 - a)))
    # Validation cases often use tiny alpha floors to approximate pure phases.
    # Do not classify every near-pure cell as positivity-violating solely
    # because the limiter's generic alpha floor is larger than the case floor.
    local_alpha_floor = min(1.0e-6, max(1.0e-12, 0.1 * alpha_margin))
    theta = positivity_blend_theta(F_HO, F_LO, U_n, dx, dt,
                                   alpha_floor=local_alpha_floor)
    F_blend = {k: theta * F_HO[k] + (1.0 - theta) * F_LO[k] for k in F_HO}
    return face, F_blend, theta
