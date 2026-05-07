# solver/denner_1d/fraysse_common.py
# Common functions shared by all Fraysse solver variants.
# EOS, flux schemes, ghost cells, interface reconstruction (CICSAM, THINC).

import numpy as np

try:
    import autograd.numpy as anp
    from autograd import jacobian as ad_jacobian
    _AD = True
except ImportError:
    import numpy as anp
    _AD = False

# Export all functions including private helpers (needed by sub-modules)
__all__ = [
    'np', 'anp', '_AD', 'ad_jacobian',
    '_get_ph_params', 'pack_fraysse', 'unpack_fraysse',
    'mixture_eos_anp', 'physical_flux_anp',
    'rusanov_flux_anp', 'hllc_flux_anp',
    '_ghost_anp', '_ghost_anp2',
    '_cicsam_face_anp', '_thinc_face_anp',
]


# ---------------------------------------------------------------------------
# EOS parameter extraction
# ---------------------------------------------------------------------------

def _get_ph_params(ph):
    """Extract EOS params from phase dict, supporting key aliases."""
    gamma = float(ph['gamma'])
    pinf  = float(ph.get('pinf', ph.get('p_inf', 0.0)))
    b     = float(ph.get('b',    ph.get('b_covolume', 0.0)))
    kv    = float(ph.get('kappa_v', ph.get('kv', ph.get('cv', 717.5))))
    eta   = float(ph.get('eta', 0.0))
    return gamma, pinf, b, kv, eta


# ---------------------------------------------------------------------------
# pack / unpack helpers
# ---------------------------------------------------------------------------

def pack_fraysse(rhoY1, rhoY2, rho_u, rho_E):
    """Pack Q = [rhoY1(N)|rhoY2(N)|rho_u(N)|rho_E(N)]."""
    return anp.concatenate([rhoY1, rhoY2, rho_u, rho_E])


def unpack_fraysse(Q, N):
    """Unpack Q (4N,) -> rhoY1, rhoY2, rho_u, rho_E each (N,)."""
    return Q[0:N], Q[N:2*N], Q[2*N:3*N], Q[3*N:4*N]


# ---------------------------------------------------------------------------
# Mixture EOS  (autograd-compatible)
# ---------------------------------------------------------------------------

def mixture_eos_anp(rhoY1, rhoY2, rho_u, rho_E, ph1, ph2):
    """Compute p, T, u, c_wood from conserved variables (autograd-compatible)."""
    g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)

    _eps = 1e-300

    rho = rhoY1 + rhoY2
    u = rho_u / (rho + _eps)
    e = rho_E / (rho + _eps) - 0.5 * u ** 2

    Y1 = rhoY1 / (rho + _eps)
    Y1 = anp.maximum(Y1, 0.0)
    Y1 = anp.minimum(Y1, 1.0)
    Y2 = 1.0 - Y1

    e_hat  = e - Y1 * eta1 - Y2 * eta2
    V      = 1.0 / (rho + _eps) - Y1 * b1 - Y2 * b2
    kv_mix = Y1 * kv1 + Y2 * kv2
    A_mix  = Y1 * kv1 * (g1 - 1.0) + Y2 * kv2 * (g2 - 1.0)

    a_qd = V * kv_mix
    b_qd = V * (Y1 * kv1 * g1 + Y2 * kv2) * pinf1 - e_hat * A_mix
    c_qd = -e_hat * Y2 * kv2 * (g2 - 1.0) * pinf1

    discriminant = b_qd ** 2 - 4.0 * a_qd * c_qd
    discriminant = anp.maximum(discriminant, 1e-30)
    sqrt_disc = anp.sqrt(discriminant)

    p_form1 = (-b_qd + sqrt_disc) / (2.0 * a_qd + _eps)
    p_form2 = -2.0 * c_qd / (b_qd + sqrt_disc + _eps)
    p = anp.where(b_qd >= 0.0, p_form2, p_form1)
    p = anp.maximum(p, 1.0)

    G_p = Y1 * kv1 * (p + g1 * pinf1) / (p + pinf1 + _eps) + Y2 * kv2
    T = e_hat / (G_p + _eps)

    rho1 = (p + pinf1) / (kv1 * T * (g1 - 1.0) + b1 * (p + pinf1) + _eps)
    rho2 = (p + _eps)  / (kv2 * T * (g2 - 1.0) + _eps)

    alpha1 = Y1 * rho / (rho1 + _eps)
    alpha1 = anp.maximum(alpha1, 0.0)
    alpha1 = anp.minimum(alpha1, 1.0)
    alpha2 = 1.0 - alpha1

    c1_sq = g1 * (p + pinf1) / (rho1 * (1.0 - b1 * rho1) + _eps)
    c2_sq = g2 * p           / (rho2 + _eps)
    c1_sq = anp.maximum(c1_sq, _eps)
    c2_sq = anp.maximum(c2_sq, _eps)

    inv_rho_c2 = (alpha1 / (rho1 * c1_sq + _eps)
                + alpha2 / (rho2 * c2_sq + _eps))
    c_wood = anp.sqrt(1.0 / (rho * inv_rho_c2 + _eps))

    return p, T, u, c_wood


# ---------------------------------------------------------------------------
# Physical flux  (autograd-compatible)
# ---------------------------------------------------------------------------

def physical_flux_anp(rhoY1, rhoY2, rho_u, rho_E, p):
    """Compute physical flux F(Q)."""
    _eps = 1e-300
    rho = rhoY1 + rhoY2
    u = rho_u / (rho + _eps)

    F_Y1     = rhoY1 * u
    F_Y2     = rhoY2 * u
    F_mom    = rho_u * u + p
    F_energy = (rho_E + p) * u

    return F_Y1, F_Y2, F_mom, F_energy


# ---------------------------------------------------------------------------
# Rusanov (Local Lax-Friedrichs) numerical flux
# ---------------------------------------------------------------------------

def rusanov_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L, c_L,
                     rY1_R, rY2_R, ru_R, rE_R, p_R, c_R):
    """Rusanov numerical flux."""
    _eps = 1e-300
    rho_L = rY1_L + rY2_L
    rho_R = rY1_R + rY2_R
    u_L = ru_L / (rho_L + _eps)
    u_R = ru_R / (rho_R + _eps)

    lam_max = anp.maximum(anp.abs(u_L) + c_L, anp.abs(u_R) + c_R)

    FL = physical_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L)
    FR = physical_flux_anp(rY1_R, rY2_R, ru_R, rE_R, p_R)

    QL = [rY1_L, rY2_L, ru_L, rE_L]
    QR = [rY1_R, rY2_R, ru_R, rE_R]

    return tuple(0.5*(fL + fR) - 0.5*lam_max*(qR - qL)
                 for fL, fR, qL, qR in zip(FL, FR, QL, QR))


# ---------------------------------------------------------------------------
# HLLC numerical flux  (Toro 1994, autograd-compatible)
# ---------------------------------------------------------------------------

def hllc_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L, c_L,
                  rY1_R, rY2_R, ru_R, rE_R, p_R, c_R):
    """HLLC numerical flux (Toro 1994). Fully autograd-compatible."""
    _eps = 1e-300

    rho_L = rY1_L + rY2_L
    rho_R = rY1_R + rY2_R
    u_L = ru_L / (rho_L + _eps)
    u_R = ru_R / (rho_R + _eps)
    E_L = rE_L / (rho_L + _eps)
    E_R = rE_R / (rho_R + _eps)
    Y1_L = rY1_L / (rho_L + _eps)
    Y1_R = rY1_R / (rho_R + _eps)
    Y2_L = rY2_L / (rho_L + _eps)
    Y2_R = rY2_R / (rho_R + _eps)

    S_L = anp.minimum(u_L - c_L, u_R - c_R)
    S_R = anp.maximum(u_L + c_L, u_R + c_R)

    num_Sstar = (p_R - p_L
                 + rho_L * u_L * (S_L - u_L)
                 - rho_R * u_R * (S_R - u_R))
    den_Sstar = (rho_L * (S_L - u_L)
                 - rho_R * (S_R - u_R))
    S_star = num_Sstar / (den_Sstar + _eps)

    FL = physical_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L)
    FR = physical_flux_anp(rY1_R, rY2_R, ru_R, rE_R, p_R)

    coeff_L = rho_L * (S_L - u_L) / (S_L - S_star + _eps)
    coeff_R = rho_R * (S_R - u_R) / (S_R - S_star + _eps)

    Estar_factor_L = E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + _eps))
    Estar_factor_R = E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + _eps))

    Q_starL = [coeff_L * Y1_L, coeff_L * Y2_L,
               coeff_L * S_star, coeff_L * Estar_factor_L]
    Q_starR = [coeff_R * Y1_R, coeff_R * Y2_R,
               coeff_R * S_star, coeff_R * Estar_factor_R]

    QL = [rY1_L, rY2_L, ru_L, rE_L]
    QR = [rY1_R, rY2_R, ru_R, rE_R]

    result = []
    for k in range(4):
        F_hllcL = FL[k] + S_L * (Q_starL[k] - QL[k])
        F_hllcR = FR[k] + S_R * (Q_starR[k] - QR[k])

        F_k = anp.where(S_L >= 0, FL[k],
              anp.where(S_star >= 0, F_hllcL,
              anp.where(S_R > 0, F_hllcR, FR[k])))
        result.append(F_k)

    return tuple(result)


# ---------------------------------------------------------------------------
# Ghost cell extension
# ---------------------------------------------------------------------------

def _ghost_anp(arr, bc_l, bc_r, ng=1):
    """Extend array (N,) with ng ghost cells on each side."""
    if bc_l == 'periodic':
        left = arr[-ng:]
    else:
        left = arr[:ng]

    if bc_r == 'periodic':
        right = arr[:ng]
    else:
        right = arr[-ng:]

    return anp.concatenate([left, arr, right])


def _ghost_anp2(arr, bc_l, bc_r):
    """Extend array (N,) with ng=2 ghost cells on each side (autograd-compatible)."""
    if bc_l == 'periodic':
        left = arr[-2:]
    else:
        left = anp.concatenate([arr[0:1], arr[0:1]])

    if bc_r == 'periodic':
        right = arr[:2]
    else:
        right = anp.concatenate([arr[-1:], arr[-1:]])

    return anp.concatenate([left, arr, right])


# ---------------------------------------------------------------------------
# CICSAM face reconstruction (autograd-compatible)
# ---------------------------------------------------------------------------

def _cicsam_face_anp(Y1_ext, u_face, dt, dx):
    """Vectorized CICSAM Hyper-C face values. autograd-compatible."""
    ng = 2
    N_faces = len(u_face)
    N = N_faces - 1

    idx = np.arange(N_faces)

    D_pos  = ng + idx - 1
    A_pos  = ng + idx
    UU_pos = ng + idx - 2

    D_neg  = ng + idx
    A_neg  = ng + idx - 1
    UU_neg = ng + idx + 1

    psi_D  = anp.where(u_face >= 0.0, Y1_ext[D_pos],  Y1_ext[D_neg])
    psi_A  = anp.where(u_face >= 0.0, Y1_ext[A_pos],  Y1_ext[A_neg])
    psi_UU = anp.where(u_face >= 0.0, Y1_ext[UU_pos], Y1_ext[UU_neg])

    denom = psi_A - psi_UU
    is_uniform = anp.abs(denom) < 1e-10
    safe_denom = anp.where(is_uniform, 1.0, denom)

    psi_tilde_D = (psi_D - psi_UU) / safe_denom

    Co = anp.abs(u_face) * dt / dx
    safe_Co = anp.where(Co < 1e-10, 1e-10, Co)

    in_range = anp.where(
        (psi_tilde_D >= 0.0) * (psi_tilde_D <= 1.0),
        1.0, 0.0
    )
    psi_tilde_hypc = anp.minimum(psi_tilde_D / safe_Co, 1.0)
    psi_tilde_f = anp.where(in_range > 0.0, psi_tilde_hypc, psi_tilde_D)

    psi_face_interp = psi_UU + psi_tilde_f * denom
    psi_face = anp.where(is_uniform, psi_D, psi_face_interp)
    psi_face = anp.maximum(anp.minimum(psi_face, 1.0), 0.0)

    return psi_face


# ---------------------------------------------------------------------------
# THINC face reconstruction (autograd-compatible, smooth)
# ---------------------------------------------------------------------------

def _thinc_face_anp(Y1_ext, u_face, beta=2.5):
    """THINC (Tangent of Hyperbola for INterface Capturing) face values.
    Fully smooth (C^∞) — autograd-compatible with exact Jacobian."""
    ng = 2
    N_faces = len(u_face)
    N = N_faces - 1

    idx = np.arange(N_faces)

    D_pos  = ng + idx - 1;  A_pos  = ng + idx;     UU_pos = ng + idx - 2
    D_neg  = ng + idx;      A_neg  = ng + idx - 1;  UU_neg = ng + idx + 1

    Y_D  = anp.where(u_face >= 0.0, Y1_ext[D_pos],  Y1_ext[D_neg])
    Y_A  = anp.where(u_face >= 0.0, Y1_ext[A_pos],  Y1_ext[A_neg])
    Y_UU = anp.where(u_face >= 0.0, Y1_ext[UU_pos], Y1_ext[UU_neg])

    _gamma = 100.0
    sigma = anp.tanh(_gamma * (Y_A - Y_UU))
    u_sign = anp.tanh(_gamma * u_face)
    theta = sigma * u_sign

    Y_thinc_hi = 0.5 + 0.5 * anp.tanh(beta * Y_D)
    Y_thinc_lo = 0.5 - 0.5 * anp.tanh(beta * (1.0 - Y_D))

    alpha = 0.5 * (1.0 + theta)
    Y_thinc = alpha * Y_thinc_hi + (1.0 - alpha) * Y_thinc_lo

    w = 4.0 * Y_D * (1.0 - Y_D)
    w = anp.minimum(w, 1.0)

    Y_face = w * Y_thinc + (1.0 - w) * Y_D
    Y_face = anp.maximum(anp.minimum(Y_face, 1.0), 0.0)

    return Y_face
