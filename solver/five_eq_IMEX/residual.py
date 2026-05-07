"""IMEX residual R(W) for one ARS-stage Newton solve.

Equation per cell (vector of length 5):

    R(W) = (U(W) − U_target)/(γΔt)
         + L_E(W^*)               # frozen at the stage anchor — does not depend on W
         + L_I(W)                  # implicit flux divergence ∇·F_I − S_I

    F_I(W) = (0, 0, p, p u, 0)^T
    L_I(W)[3]  = ∂p / ∂x          (momentum)
    L_I(W)[4]  = ∂(p u) / ∂x       (energy, pressure work)

The Kapila α-source can be split consistently with the acoustic subsystem:

    ∂t α + ∂x(αu) = (α + D_K) ∂x u
    ⇒ L_E,α = ∂x(αu) − α ∂x u,   L_I,α = −D_K^n ∂x u*

`kapila_source` below is therefore the frozen coefficient D_K^n, not
`α + D_K`.  Keeping only D_K implicit avoids double counting the
convective α∂x u part already present in L_E.

For Phase 3 we use a simple 2-point central face stencil for face pressure
and face velocity. Boundary conditions are handled via ghost cells in
`boundary.extend`.
"""
from __future__ import annotations
import numpy as np

from .boundary import extend
from .primitive import prim_to_cons_W

_EPS = 1e-30


def _mixture_impedance(W, eos1, eos2):
    """Cell-centered acoustic impedance rho*c from the Kapila mixture sound speed."""
    from .sound_speed import phase_sound_speed_sq, mixture_sound_speed_sq

    a1, T1, T2, _, p = W
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
    c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
    rho = np.maximum(a1 * rho1 + (1.0 - a1) * rho2, _EPS)
    c_mix_sq = mixture_sound_speed_sq(a1, rho1, c1_sq, rho2, c2_sq, kind='kapila')
    return np.maximum(rho * np.sqrt(np.maximum(c_mix_sq, _EPS)), _EPS)


def _apply_interface_acoustic_riemann(W, p_face, u_face, p_L, p_R, u_L, u_R,
                                      bc_l, bc_r, ng, eos1, eos2,
                                      alpha_jump_tol=1e-8):
    """Use linear acoustic Riemann face states at material alpha jumps."""
    if eos1 is None or eos2 is None:
        return p_face, u_face

    a1 = W[0]
    Z = _mixture_impedance(W, eos1, eos2)
    a_ext = extend(a1, bc_l, bc_r, ng=ng, odd=False)
    Z_ext = extend(Z, bc_l, bc_r, ng=ng, odd=False)

    n_face = p_face.shape[0]
    left = ng - 1
    a_L = a_ext[left:left + n_face]
    a_R = a_ext[left + 1:left + 1 + n_face]
    Z_L = Z_ext[left:left + n_face]
    Z_R = Z_ext[left + 1:left + 1 + n_face]

    mask = np.abs(a_R - a_L) > alpha_jump_tol
    if not np.any(mask):
        return p_face, u_face

    den = np.maximum(Z_L + Z_R, _EPS)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den

    p_out = p_face.copy()
    u_out = u_face.copy()
    p_out[mask] = p_star[mask]
    u_out[mask] = u_star[mask]
    return p_out, u_out


def _acoustic_riemann_faces(W, p_L, p_R, u_L, u_R,
                            bc_l, bc_r, ng, eos1, eos2, smooth=0.0):
    """Linear acoustic Riemann p/u face states on every face."""
    if eos1 is None or eos2 is None:
        return 0.5 * (p_L + p_R), 0.5 * (u_L + u_R)

    Z = _mixture_impedance(W, eos1, eos2)
    Z_ext = extend(Z, bc_l, bc_r, ng=ng, odd=False)
    n_face = p_L.shape[0]
    left = ng - 1
    Z_L = Z_ext[left:left + n_face]
    Z_R = Z_ext[left + 1:left + 1 + n_face]

    den = np.maximum(Z_L + Z_R, _EPS)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den
    if smooth > 0.0 and p_star.shape[0] > 2:
        w = min(max(float(smooth), 0.0), 0.49)
        p_sm = p_star.copy()
        u_sm = u_star.copy()
        p_sm[1:-1] = (1.0 - 2.0 * w) * p_star[1:-1] + w * (p_star[:-2] + p_star[2:])
        u_sm[1:-1] = (1.0 - 2.0 * w) * u_star[1:-1] + w * (u_star[:-2] + u_star[2:])
        p_star, u_star = p_sm, u_sm
    return p_star, u_star


def implicit_face_pu(W, bc_l, bc_r, *,
                     u_inlet=None, p_inlet=None, eos1=None, eos2=None,
                     rhie_chow=False, gamma_dt=None, dx=None,
                     dissipation=0.0, dissipation_form='biharmonic'):
    """Face pressure and face velocity for the implicit acoustic operator.

    Default: central 2-pt average (PE-preserving but odd-even decoupled).

    `dissipation` ∈ [0, 1] with `dissipation_form='biharmonic'` (default):
        p_face = ½(p_L + p_R) − D · (−p_LL + 3p_L − 3p_R + p_RR)/8

    On a uniform (or smooth) state this term vanishes (3-point smoothness
    test = 0).  For the 2-Δx (nyquist) mode (alternating ±1) it equals
        D · (−1 + 3 − 3 · −1 + 1)/8 = D · 8/8 = D · ε_nyquist
    so the nyquist pressure-mode is damped at rate D per face evaluation.

    `dissipation_form='upwind'` falls back to the sign-based 1st-order bias
    (cancels on alternating modes — kept for diagnostics only).
    """
    use_bih = (dissipation > 0.0 and dissipation_form == 'biharmonic')
    use_rc = bool(rhie_chow and bc_l == 'periodic' and bc_r == 'periodic'
                  and eos1 is not None and eos2 is not None
                  and gamma_dt is not None and dx is not None and dx > 0.0)
    ng = 2 if (use_bih or use_rc) else 1
    if bc_l == 'inlet_acoustic' and eos1 is not None and eos2 is not None:
        from .boundary import extend_W
        _, _, _, u_ext, p_ext = extend_W(
            W, bc_l, bc_r, ng=ng,
            u_inlet_l=u_inlet, p_inlet_l=p_inlet,
            eos1=eos1, eos2=eos2)
    else:
        _, _, _, u, p = W
        p_ext = extend(p, bc_l, bc_r, ng=ng, odd=False, dirichlet_l=p_inlet)
        u_ext = extend(u, bc_l, bc_r, ng=ng, odd=True,  dirichlet_l=u_inlet)

    if use_bih:
        # 4-point biharmonic stencil — face uses (p_LL, p_L, p_R, p_RR)
        N_face = p_ext.shape[0] - 3
        p_LL = p_ext[0:N_face]
        p_L  = p_ext[1:N_face+1]
        p_R  = p_ext[2:N_face+2]
        p_RR = p_ext[3:N_face+3]
        u_LL = u_ext[0:N_face]
        u_L  = u_ext[1:N_face+1]
        u_R  = u_ext[2:N_face+2]
        u_RR = u_ext[3:N_face+3]
        # Biharmonic kernel coefficients: [-1, 3, -3, 1] / 8 on (LL, L, R, RR)
        # Note: applied as p_face = central − D · biharmonic_term.
        bih_p = (-p_LL + 3.0 * p_L - 3.0 * p_R + p_RR) / 8.0
        bih_u = (-u_LL + 3.0 * u_L - 3.0 * u_R + u_RR) / 8.0
        p_face = 0.5 * (p_L + p_R) - dissipation * bih_p
        u_face = 0.5 * (u_L + u_R) - dissipation * bih_u
        p_face, u_face = _apply_interface_acoustic_riemann(
            W, p_face, u_face, p_L, p_R, u_L, u_R, bc_l, bc_r, ng, eos1, eos2)
        return p_face, u_face

    # No or upwind dissipation — 2-pt stencil
    if use_rc:
        a1, T1, T2, u, p = W
        rho1 = eos1.density(p, T1)
        rho2 = eos2.density(p, T2)
        rho = a1 * rho1 + (1.0 - a1) * rho2
        rho_ext = extend(rho, bc_l, bc_r, ng=ng, odd=False)

        p_im1 = p_ext[:-3]
        p_i   = p_ext[1:-2]
        p_ip1 = p_ext[2:-1]
        p_ip2 = p_ext[3:]
        u_i   = u_ext[1:-2]
        u_ip1 = u_ext[2:-1]
        rho_i = rho_ext[1:-2]
        rho_ip1 = rho_ext[2:-1]

        p_L = p_i
        p_R = p_ip1
        u_L = u_i
        u_R = u_ip1

        inv_dx = 1.0 / dx
        grad_p_f = (p_ip1 - p_i) * inv_dx
        grad_p_i = 0.5 * (p_ip1 - p_im1) * inv_dx
        grad_p_ip1 = 0.5 * (p_ip2 - p_i) * inv_dx
        grad_p_avg_f = 0.5 * (grad_p_i + grad_p_ip1)
        rho_f = np.maximum(0.5 * (rho_i + rho_ip1), _EPS)
        D_f = gamma_dt / rho_f

        p_face = 0.5 * (p_L + p_R)
        u_face = 0.5 * (u_L + u_R) - D_f * (grad_p_f - grad_p_avg_f)
        return p_face, u_face

    p_L = p_ext[:-1]; p_R = p_ext[1:]
    u_L = u_ext[:-1]; u_R = u_ext[1:]
    if dissipation_form == 'acoustic_riemann':
        return _acoustic_riemann_faces(
            W, p_L, p_R, u_L, u_R, bc_l, bc_r, ng, eos1, eos2,
            smooth=dissipation)
    p_face = 0.5 * (p_L + p_R)
    u_face = 0.5 * (u_L + u_R)
    if dissipation_form == 'upwind' and dissipation > 0.0:
        sign_u = np.where(u_face >= 0.0, 1.0, -1.0)
        p_face = p_face - dissipation * 0.5 * sign_u * (p_R - p_L)
        u_face = u_face - dissipation * 0.5 * sign_u * (u_R - u_L)
    p_face, u_face = _apply_interface_acoustic_riemann(
        W, p_face, u_face, p_L, p_R, u_L, u_R, bc_l, bc_r, ng, eos1, eos2)
    return p_face, u_face


def implicit_divergences(W, dx, bc_l, bc_r, *,
                         u_inlet=None, p_inlet=None, eos1=None, eos2=None,
                         rhie_chow=False, gamma_dt=None,
                         dissipation=0.0,
                         dissipation_form='biharmonic',
                         compact_lap_coeff=0.0):
    """L_I(W) row contributions to momentum (3) and energy (4)."""
    p_face, u_face = implicit_face_pu(W, bc_l, bc_r,
                                      u_inlet=u_inlet, p_inlet=p_inlet,
                                      eos1=eos1, eos2=eos2,
                                      rhie_chow=rhie_chow,
                                      gamma_dt=gamma_dt, dx=dx,
                                      dissipation=dissipation,
                                      dissipation_form=dissipation_form)
    inv_dx = 1.0 / dx
    grad_p   = (p_face[1:] - p_face[:-1]) * inv_dx
    div_u    = (u_face[1:] - u_face[:-1]) * inv_dx
    if compact_lap_coeff != 0.0 and bc_l == 'periodic' and bc_r == 'periodic':
        p = W[4]
        lap_p_over_dx = (np.roll(p, -1) - 2.0 * p + np.roll(p, 1)) * inv_dx * inv_dx * dx
        grad_p = grad_p + compact_lap_coeff * lap_p_over_dx
    # Conservative pressure-work divergence:
    #   div_pu = d_x (p_face * u_face)
    pu_face = p_face * u_face
    div_pu = (pu_face[1:] - pu_face[:-1]) * inv_dx
    return dict(grad_p=grad_p, div_pu=div_pu, div_u=div_u, p_face=p_face, u_face=u_face)


def residual(W, U_target, gamma_dt, L_E, eos1, eos2, dx, bc_l, bc_r, *,
             u_inlet=None, p_inlet=None,
             alpha_source_explicit=True, kapila_source=None,
             rhie_chow=False,
             imp_dissipation=0.0,
             imp_dissipation_form='biharmonic',
             imp_compact_lap_coeff=0.0,
             include_explicit_residual=False,
             pe_correct=False):
    """ARS-stage Newton residual (L_I-only, used by ars222 / be1).

        R(W) = (U(W) − U_target)/(γΔt) + L_I(W)
        L_I(W) = (0, 0, ∂p/∂x, ∂(p·u)/∂x, 0)

    `L_E` is accumulated outside the Newton (final ARS update).
    """
    U_now, _ = prim_to_cons_W(W, eos1, eos2)
    impl = implicit_divergences(W, dx, bc_l, bc_r,
                                u_inlet=u_inlet, p_inlet=p_inlet,
                                eos1=eos1, eos2=eos2,
                                rhie_chow=rhie_chow,
                                gamma_dt=gamma_dt,
                                dissipation=imp_dissipation,
                                dissipation_form=imp_dissipation_form,
                                compact_lap_coeff=imp_compact_lap_coeff)
    R = [None] * 5
    for k in range(5):
        R[k] = (U_now[k] - U_target[k]) / gamma_dt
        if include_explicit_residual:
            R[k] = R[k] + L_E[k]
    R[2] = R[2] + impl['grad_p']
    R[3] = R[3] + impl['div_pu']
    if not alpha_source_explicit:
        coeff = 0.0 if kapila_source is None else kapila_source
        R[4] = R[4] - coeff * impl['div_u']

    R_tuple = tuple(R)
    if pe_correct:
        # §6.4: project R onto PE-tangent by absorbing the PE-normal component
        # into the energy residual.  Should reduce the spectral PE-violating
        # eigenmode of the one-step amplification matrix.
        from .pe_correction import apply_pe_correction
        R_tuple, _pi = apply_pe_correction(R_tuple, W, eos1, eos2)
    return R_tuple, impl


def residual_full(W, U_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                  u_inlet=None, p_inlet=None,
                  alpha_scheme='upwind',
                  primitive_scheme='upwind',
                  energy_form='apec',
                  face_thermo='acid',
                  kapila_closure=False,
                  positivity=True):
    """Fully-implicit BE residual (mass + momentum + energy + α + ∇p + p·u
    all W-dependent):

        R(W) = (U(W) − U^n)/Δt + L_E(W) + L_I(W)

    L_E(W) uses the same ACID/APEC/upwind scheme as `explicit_residual`.
    L_I(W) is the central-stencil ∇p, ∂(p·u)/∂x.

    This is the standard non-IMEX backward-Euler form — used by `be_full_step`.
    Compared to the user-spec ARS222, the *advection* operator is now
    inside Newton, which is the only formulation that achieves machine-ε
    PE preservation across sharp α-jumps for arbitrary EOS.

    Returns (R_tuple, impl_dict, L_E_tuple).
    """
    U_now, _ = prim_to_cons_W(W, eos1, eos2)
    L_E, _face = explicit_residual(W, eos1, eos2, dx, bc_l, bc_r,
                                    alpha_scheme=alpha_scheme,
                                    primitive_scheme=primitive_scheme,
                                    energy_form=energy_form,
                                    face_thermo=face_thermo,
                                    kapila_closure=kapila_closure,
                                    positivity=positivity, dt=dt)
    impl = implicit_divergences(W, dx, bc_l, bc_r,
                                u_inlet=u_inlet, p_inlet=p_inlet)
    R = [None] * 5
    for k in range(5):
        R[k] = (U_now[k] - U_n[k]) / dt + L_E[k]
    R[2] = R[2] + impl['grad_p']
    R[3] = R[3] + impl['div_pu']
    return tuple(R), impl, L_E


def explicit_residual(W_anchor, eos1, eos2, dx, bc_l, bc_r, *,
                      alpha_scheme='upwind',
                      primitive_scheme='upwind',
                      energy_form='apec',
                      energy_alpha_pure_tol=1.0e-12,
                      face_thermo='acid',
                      kapila_closure=False,
                      positivity=True,
                      dt=None,
                      force_lo=False,
                      lo_flux='pe_preserving',
                      kapila_source_in_implicit=False):
    """Explicit operator L_E(W^*) = ∇·F_E − S_E.

    Phase 8: when `positivity=True` and `dt` is given, the high-order APEC+ACID
    flux is blended cell-wise with a Rusanov low-order flux so the candidate
    update remains admissible (no air cell gets negative phase mass).
    Allaire-Massoni S_E = (0,0,0,0,(α+D_K)·div(u)).
    Returns a 5-tuple of (N,) arrays plus the face dict.
    """
    from .face_state import face_state
    from .flux import advective_fluxes
    from .limiters import blended_advective_fluxes

    if (positivity or force_lo) and dt is not None:
        face, flx, _theta = blended_advective_fluxes(
            W_anchor, eos1, eos2, dx, dt, bc_l, bc_r,
            energy_form=energy_form,
            energy_alpha_pure_tol=energy_alpha_pure_tol,
            alpha_scheme=alpha_scheme,
            primitive_scheme=primitive_scheme,
            face_thermo=face_thermo,
            positivity=positivity,
            force_lo=force_lo,
            lo_flux=lo_flux)
    else:
        face = face_state(W_anchor, eos1, eos2, bc_l, bc_r,
                          alpha_scheme=alpha_scheme,
                          primitive_scheme=primitive_scheme,
                          face_thermo=face_thermo,
                          dt=dt,
                          dx=dx)
        flx = advective_fluxes(face, eos1, eos2, energy_form=energy_form,
                               energy_alpha_pure_tol=energy_alpha_pure_tol)

    inv_dx = 1.0 / dx
    div = {k: (F[1:] - F[:-1]) * inv_dx for k, F in flx.items()}

    # α-source S_E = (α + D_K)·div(u_face).
    #
    # If the Kapila acoustic split is enabled, only α·div(u) remains in the
    # explicit convective operator and D_K·div(u*) is put in the implicit
    # acoustic row.  Otherwise keep the legacy fully explicit (α+D_K) path.
    u_face = face['u']
    div_u = (u_face[1:] - u_face[:-1]) * inv_dx
    a1 = W_anchor[0]
    if kapila_closure:
        if kapila_source_in_implicit:
            B_cell = a1
        else:
            from .source_d1 import D_K_kapila_face
            B_face = face['alpha'] + D_K_kapila_face(face)
            B_cell = 0.5 * (B_face[1:] + B_face[:-1])
    else:
        B_cell = a1
    S_alpha = B_cell * div_u

    L_E = (
        div['F_a1r1'],
        div['F_a2r2'],
        div['F_ru'],
        div['F_rE'],
        div['F_alpha'] - S_alpha,
    )
    return L_E, face
