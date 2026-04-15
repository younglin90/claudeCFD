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

def mixture_eos_anp(rhoY1, rhoY2, rho_u, rho_E, ph1, ph2, alpha1_in=None):
    """Compute p, T, u, c_wood from conserved variables (autograd-compatible).

    When alpha1_in is provided and both phases have b=0, eta=0 (pure SG),
    uses the direct α-based linear pressure formula to avoid catastrophic
    cancellation in the quadratic solver at extreme pressure ratios.
    """
    g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)

    _eps = 1e-300

    rho = rhoY1 + rhoY2
    u = rho_u / (rho + _eps)
    rho_e = rho_E - 0.5 * rho_u * u   # ρe = ρE - ½ρu²

    # --- Pressure computation ---
    use_direct = (alpha1_in is not None and b1 == 0 and b2 == 0
                  and eta1 == 0 and eta2 == 0)

    if use_direct:
        # Direct α-based linear formula for SG (no quadratic, no cancellation)
        # ρe = Σ α_k(p + γ_k P∞_k)/(γ_k-1)
        # → p = (ρe - Σ α_k γ_k P∞_k/(γ_k-1)) / (Σ α_k/(γ_k-1))
        a1 = alpha1_in
        a2 = 1.0 - a1
        Gamma_inv = a1 / (g1 - 1.0) + a2 / (g2 - 1.0)
        Pi = a1 * g1 * pinf1 / (g1 - 1.0) + a2 * g2 * pinf2 / (g2 - 1.0)
        p = (rho_e - Pi) / (Gamma_inv + _eps)
        p = anp.maximum(p, 1.0)

        # T from mixture density identity (no division by alpha, no switch):
        # ρ = Σ α_k(p+P∞_k)/((γ_k-1)*kv_k*T)  =>  T = Σ [...] / ρ
        T_numer = a1 * (p + pinf1) / ((g1 - 1.0) * kv1) \
                + a2 * (p + pinf2) / ((g2 - 1.0) * kv2)
        T = T_numer / (rho + _eps)
        T = anp.maximum(T, 1.0)

        rho1 = (p + pinf1) / (kv1 * T * (g1 - 1.0) + _eps)
        rho2 = (p + pinf2) / (kv2 * T * (g2 - 1.0) + _eps)
        alpha1 = a1
        alpha2 = a2
    else:
        # General NASG: Y-based quadratic formula
        e = rho_e / (rho + _eps)  # specific internal energy

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

    # Sound speed (common to both paths)
    c1_sq = g1 * (p + pinf1) / (rho1 * (1.0 - b1 * rho1) + _eps)
    c2_sq = g2 * (p + pinf2) / (rho2 + _eps)
    c1_sq = anp.maximum(c1_sq, _eps)
    c2_sq = anp.maximum(c2_sq, _eps)

    inv_rho_c2 = (alpha1 / (rho1 * c1_sq + _eps)
                + alpha2 / (rho2 * c2_sq + _eps))
    c_wood = anp.sqrt(1.0 / (rho * inv_rho_c2 + _eps))

    return p, T, u, c_wood


# ---------------------------------------------------------------------------
# Physical flux  (autograd-compatible)
# ---------------------------------------------------------------------------
# MUSCL-THINC-BVD reconstruction (Deng et al., JCP 371, 2018)
# ---------------------------------------------------------------------------

def _muscl_reconstruct_anp(q_ext, N):
    """MUSCL reconstruction with minmod limiter. Returns (qL, qR) at N+1 faces.
    q_ext: (N+2,) ghost-extended cell averages."""
    # Slopes
    dq_L = q_ext[1:N+1] - q_ext[0:N]      # q_i - q_{i-1}
    dq_R = q_ext[2:N+2] - q_ext[1:N+1]    # q_{i+1} - q_i

    # Minmod limiter (smooth for autograd)
    sigma = anp.where(dq_L * dq_R > 0,
                      anp.where(anp.abs(dq_L) < anp.abs(dq_R), dq_L, dq_R),
                      0.0)  # (N,)

    # L/R values at faces
    q_cells = q_ext[1:N+1]  # (N,)
    qL = q_cells + 0.5 * sigma      # left state at face i+1/2 = q_i + slope/2
    qR = q_cells - 0.5 * sigma      # right state at face i-1/2 = q_i - slope/2

    # Assemble face arrays: face j has L from cell j, R from cell j+1
    # qL_face[j] = qL[j] (left of face j+1/2 from cell j)
    # qR_face[j] = qR[j+1] (right of face j+1/2 from cell j+1)
    # But we need (N+1,) face values from (N,) cell values.
    # Use ghost-extended: qL_face[0..N] from cells 0..N, qR_face[0..N] from cells 1..N+1

    # Extend qL and qR with ghost
    sigma_ext_L = anp.concatenate([sigma[0:1], sigma])  # left ghost: copy first
    sigma_ext_R = anp.concatenate([sigma, sigma[-1:]])   # right ghost: copy last
    q_ext_L = q_ext[0:N+1] + 0.5 * sigma_ext_L  # (N+1,) left states at faces
    q_ext_R = q_ext[1:N+2] - 0.5 * sigma_ext_R  # (N+1,) right states at faces

    return q_ext_L, q_ext_R


def _thinc_reconstruct_anp(q_ext, N, beta=1.6):
    """Conservation-constrained THINC reconstruction (Deng 2018, Eq. 23-24).
    Returns (qL, qR) at N+1 faces.
    q_ext: (N+2,) ghost-extended cell averages."""
    _eps = 1e-20

    q_im1 = q_ext[0:N]      # q_{i-1}
    q_i   = q_ext[1:N+1]    # q_i (cell averages)
    q_ip1 = q_ext[2:N+2]    # q_{i+1}

    q_min = anp.minimum(q_im1, q_ip1)
    q_max = anp.maximum(q_im1, q_ip1)
    dq = q_max - q_min

    # Normalized variable C = (q_i - q_min) / (q_max - q_min)
    C = (q_i - q_min) / (dq + _eps)
    C = anp.maximum(anp.minimum(C, 1.0 - _eps), _eps)

    # Direction
    theta = anp.tanh(100.0 * (q_ip1 - q_im1))  # smooth sign

    # THINC face values (Eq. 24)
    B = anp.exp(theta * beta * (2.0 * C - 1.0))
    A = B / (anp.cosh(beta) - 1.0) * anp.tanh(beta)
    # Protect: A should not blow up
    A = anp.maximum(anp.minimum(A, 1e10), -1e10)

    q_half_sum = (q_min + q_max) / 2.0

    # Right face value of cell i (= right state at face i-1/2)
    qR_cell = q_half_sum * (1.0 + theta * A)

    # Left face value of cell i (= left state at face i+1/2)
    numer = anp.tanh(beta) + A
    denom = 1.0 + A * anp.tanh(beta)
    qL_cell = q_half_sum * (1.0 + theta * numer / (denom + 1e-30))

    # Clip to [q_min, q_max] for safety
    qL_cell = anp.maximum(anp.minimum(qL_cell, q_max), q_min)
    qR_cell = anp.maximum(anp.minimum(qR_cell, q_max), q_min)

    # Assemble face arrays (same structure as MUSCL)
    qL_ext = anp.concatenate([qL_cell[0:1], qL_cell])  # (N+1,)
    qR_ext = anp.concatenate([qR_cell, qR_cell[-1:]])   # (N+1,)

    return qL_ext, qR_ext


def _bvd_reconstruct_anp(q_ext, N, beta=1.6, delta=1e-4):
    """MUSCL-THINC-BVD reconstruction (Deng 2018, Eq. 26-27).
    Returns (qL_face, qR_face) each (N+1,).
    q_ext: (N+2,) ghost-extended cell averages."""

    # Get both reconstructions
    qL_mu, qR_mu = _muscl_reconstruct_anp(q_ext, N)     # (N+1,) each
    qL_th, qR_th = _thinc_reconstruct_anp(q_ext, N, beta)  # (N+1,) each

    # BVD criterion for each cell i: compare TBV of MUSCL vs THINC
    # TBV_i = |qL_{i-1/2} - qR_{i-1/2}| + |qL_{i+1/2} - qR_{i+1/2}|
    # For cell i: left face = i-1/2 (index i-1 in 0-based face array? No.)
    # Face j corresponds to face between cell j and cell j+1.
    # Cell i's left face = face i (in 0-based where face 0 = left boundary)
    # Actually face indexing: face j = boundary between cell j-1 and cell j.
    # Let's use: qL_face[j] = left state at face j, qR_face[j] = right state at face j.
    # Cell i contributes: qL at face i+1 (right boundary of cell i)
    #                     qR at face i   (left boundary of cell i)

    # TBV for MUSCL at cell i (using neighbor's reconstruction at the other side)
    # Simplified: TBV_MUSCL[i] = |jump at left face| + |jump at right face|
    #           = |qL_mu[i] - qR_mu[i]| + |qL_mu[i+1] - qR_mu[i+1]|
    # Wait, this isn't right. The TBV should compare jumps across faces.
    # Face j: jump = |qL[j] - qR[j]| where qL comes from cell j-1, qR from cell j.

    # For simplicity, compute face jumps for both reconstructions
    jump_mu = anp.abs(qL_mu - qR_mu)  # (N+1,) jump at each face for MUSCL
    jump_th = anp.abs(qL_th - qR_th)  # (N+1,) jump at each face for THINC

    # TBV for cell i = jump at face i + jump at face i+1
    # (face i = left boundary of cell i, face i+1 = right boundary)
    tbv_mu = jump_mu[0:N] + jump_mu[1:N+1]  # (N,)
    tbv_th = jump_th[0:N] + jump_th[1:N+1]  # (N,)

    # THINC activation condition (Eq. 25): monotone + not flat
    q_i = q_ext[1:N+1]
    q_im1 = q_ext[0:N]
    q_ip1 = q_ext[2:N+2]
    dq = anp.maximum(q_ip1, q_im1) - anp.minimum(q_ip1, q_im1)
    C = (q_i - anp.minimum(q_im1, q_ip1)) / (dq + 1e-20)
    is_monotone = (q_ip1 - q_i) * (q_i - q_im1)  # > 0 if monotone

    # Smooth selection: use THINC when conditions met AND TBV is smaller
    # w_thinc ≈ 1 when THINC preferred, ≈ 0 when MUSCL preferred
    cond_range = anp.tanh(100.0 * (C - delta)) * anp.tanh(100.0 * (1.0 - delta - C))
    cond_range = anp.maximum(cond_range, 0.0)  # 0 to 1
    cond_mono = anp.tanh(100.0 * is_monotone)
    cond_mono = anp.maximum(cond_mono, 0.0)
    cond_tbv = anp.tanh(100.0 * (tbv_mu - tbv_th))  # >0 when THINC has smaller TBV
    cond_tbv = anp.maximum(cond_tbv, 0.0)

    w_thinc = cond_range * cond_mono * cond_tbv  # (N,) per-cell weight

    # Blend L/R states per cell, then assemble faces
    # Cell i provides: qL at face i+1, qR at face i
    # For face j: qL[j] comes from cell j (its right boundary)
    #             qR[j] comes from cell j (its left boundary... no)
    # Actually: qL_face[j] = left state from cell j-1's right boundary
    #           qR_face[j] = right state from cell j's left boundary

    # Per-cell blended L (right boundary) and R (left boundary):
    # For cell i: qL_i = w*qL_th_cell[i] + (1-w)*qL_mu_cell[i]
    #             qR_i = w*qR_th_cell[i] + (1-w)*qR_mu_cell[i]

    # Extract per-cell L/R from face arrays:
    # qL_cell[i] = qL_face[i+1] (cell i's contribution to face i+1)
    # qR_cell[i] = qR_face[i]   (cell i's contribution to face i)
    # But our face arrays already have this structure from _muscl/_thinc.

    # Simpler: blend at the face level using the upstream cell's weight
    # For face j, left state from cell j-1: use w_thinc[j-1]
    # For face j, right state from cell j: use w_thinc[j]
    w_ext = anp.concatenate([w_thinc[0:1], w_thinc])  # (N+1,) for left states
    w_ext2 = anp.concatenate([w_thinc, w_thinc[-1:]])  # (N+1,) for right states

    qL_face = w_ext * qL_th + (1.0 - w_ext) * qL_mu
    qR_face = w_ext2 * qR_th + (1.0 - w_ext2) * qR_mu

    return qL_face, qR_face


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
    """HLLC numerical flux (Toro 1994). Fully autograd-compatible.

    Returns (F_rY1, F_rY2, F_ru, F_rE, u_face) — 5 arrays of shape (N+1,).
    u_face is the HLLC contact velocity (S* based), consistent with
    Johnsen & Colonius (2006) Eq. 25 for the α equation source term.
    """
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

    # HLLC face velocity consistent with Johnsen & Colonius (2006) Eq. 25
    # u_face = u_L + S_L*(S_L-u_L)/(S_L-S*) - 1)  if S* >= 0
    #        = u_R + S_R*(S_R-u_R)/(S_R-S*) - 1)  if S* < 0
    u_face_L = u_L + S_L * ((S_L - u_L) / (S_L - S_star + _eps) - 1.0)
    u_face_R = u_R + S_R * ((S_R - u_R) / (S_R - S_star + _eps) - 1.0)
    u_hllc = anp.where(S_star >= 0, u_face_L, u_face_R)

    return tuple(result) + (u_hllc,)


# ---------------------------------------------------------------------------
# AUSM+-up numerical flux (Liou 2006) — multi-species version
# ---------------------------------------------------------------------------

def ausm_plus_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L, c_L,
                       rY1_R, rY2_R, ru_R, rE_R, p_R, c_R):
    """AUSM+-up numerical flux for 2-species compressible flow.

    Same interface as hllc_flux_anp. Low-Mach fix included (Kp=0.25, Ku=0.75).

    Ref: Liou, JCP 214 (2006) 137-170.
    """
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
    H_L = E_L + p_L / (rho_L + _eps)
    H_R = E_R + p_R / (rho_R + _eps)

    # Interface sound speed (Liou 2006, Eq. 13)
    c_half = 0.5 * (c_L + c_R)
    c_half = anp.maximum(c_half, _eps)

    # Mach numbers
    M_L = u_L / c_half
    M_R = u_R / c_half

    # AUSM+ Mach splitting (smooth for autograd)
    abs_ML = anp.sqrt(M_L**2 + 1e-14)
    abs_MR = anp.sqrt(M_R**2 + 1e-14)

    # Smooth sub/supersonic blend
    _st = 20.0
    w_sub_L = 0.5 * (1.0 + anp.tanh(_st * (1.0 - abs_ML)))
    w_sub_R = 0.5 * (1.0 + anp.tanh(_st * (1.0 - abs_MR)))

    # M+ (subsonic: (M+1)²/4 + β(M²-1)², supersonic: (M+|M|)/2)
    beta_ausm = 1.0 / 8.0
    Mp_sub = 0.25 * (M_L + 1.0)**2 + beta_ausm * (M_L**2 - 1.0)**2
    Mp_sup = 0.5 * (M_L + abs_ML)
    Mm_sub = -0.25 * (M_R - 1.0)**2 - beta_ausm * (M_R**2 - 1.0)**2
    Mm_sup = 0.5 * (M_R - abs_MR)

    M_plus  = w_sub_L * Mp_sub + (1.0 - w_sub_L) * Mp_sup
    M_minus = w_sub_R * Mm_sub + (1.0 - w_sub_R) * Mm_sup

    # Low-Mach correction (AUSM+-up, Kp=0.25, Ku=0.75)
    M_bar_sq = 0.5 * (M_L**2 + M_R**2)
    M_0_sq = anp.minimum(1.0, anp.maximum(M_bar_sq, 1e-6))
    M_0 = anp.sqrt(M_0_sq)
    fa = M_0 * (2.0 - M_0)  # f(M_0), Eq. 16 in Liou 2006
    Kp = 0.25; Ku = 0.75
    rho_half = 0.5 * (rho_L + rho_R)

    M_half = M_plus + M_minus - Kp / fa * anp.maximum(1.0 - M_bar_sq, 0.0) * (p_R - p_L) / (rho_half * c_half**2 + _eps)

    # Mass flux
    m_dot = c_half * M_half * anp.where(M_half >= 0, rho_L, rho_R)

    # Pressure splitting (AUSM+ style, smooth)
    Pp_sub = 0.25 * (M_L + 1.0)**2 * (2.0 - M_L) + 3.0/16.0 * M_L * (M_L**2 - 1.0)**2
    Pp_sup = 0.5 * (1.0 + anp.tanh(100.0 * M_L))
    Pm_sub = 0.25 * (M_R - 1.0)**2 * (2.0 + M_R) - 3.0/16.0 * M_R * (M_R**2 - 1.0)**2
    Pm_sup = 0.5 * (1.0 - anp.tanh(100.0 * M_R))

    Pp_L = w_sub_L * Pp_sub + (1.0 - w_sub_L) * Pp_sup
    Pm_R = w_sub_R * Pm_sub + (1.0 - w_sub_R) * Pm_sup

    # Low-Mach pressure correction
    p_half = (Pp_L * p_L + Pm_R * p_R
              - Ku * Pp_L * Pm_R * (rho_L + rho_R) * fa * c_half * (u_R - u_L))

    # Upwind species/enthalpy
    is_pos = (m_dot >= 0)
    Y1_up = anp.where(is_pos, Y1_L, Y1_R)
    Y2_up = anp.where(is_pos, Y2_L, Y2_R)
    u_up  = anp.where(is_pos, u_L, u_R)
    H_up  = anp.where(is_pos, H_L, H_R)

    F_rY1 = m_dot * Y1_up
    F_rY2 = m_dot * Y2_up
    F_ru  = m_dot * u_up + p_half
    F_rE  = m_dot * H_up

    return (F_rY1, F_rY2, F_ru, F_rE)


# ---------------------------------------------------------------------------
# SLAU2 numerical flux (Kitamura & Shima 2013) — multi-species version
# ---------------------------------------------------------------------------

def slau2_flux_anp(rY1_L, rY2_L, ru_L, rE_L, p_L, c_L,
                   rY1_R, rY2_R, ru_R, rE_R, p_R, c_R):
    """SLAU2 numerical flux for 2-species compressible flow.

    Same interface as hllc_flux_anp: takes L/R conservative + primitive states,
    returns (F_rY1, F_rY2, F_ru, F_rE) tuple of face fluxes.

    Ref: Kitamura & Shima, JCP 245 (2013) 62-83.
    """
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
    H_L = E_L + p_L / (rho_L + _eps)
    H_R = E_R + p_R / (rho_R + _eps)

    # Interface sound speed
    c_bar = 0.5 * (c_L + c_R)
    c_bar = anp.maximum(c_bar, _eps)

    # Mach numbers
    M_L = u_L / c_bar
    M_R = u_R / c_bar

    # Average Mach and chi parameter
    M_bar = anp.sqrt(0.5 * (u_L**2 + u_R**2)) / c_bar
    M_hat = anp.minimum(1.0, M_bar)
    chi = (1.0 - M_hat)**2

    # Smooth |M| for autograd
    abs_ML = anp.sqrt(M_L**2 + 1e-14)
    abs_MR = anp.sqrt(M_R**2 + 1e-14)

    # Smooth subsonic/supersonic blending using tanh
    # w_sub ≈ 1 when |M| < 1, ≈ 0 when |M| > 1
    _st = 20.0  # steepness of transition
    w_sub_L = 0.5 * (1.0 + anp.tanh(_st * (1.0 - abs_ML)))
    w_sub_R = 0.5 * (1.0 + anp.tanh(_st * (1.0 - abs_MR)))

    # M+ and M- (smooth blend of sub/supersonic)
    Mp_sub = 0.25 * (M_L + 1.0)**2
    Mp_sup = 0.5 * (M_L + abs_ML)
    Mm_sub = -0.25 * (M_R - 1.0)**2
    Mm_sup = 0.5 * (M_R - abs_MR)

    M_plus  = w_sub_L * Mp_sub + (1.0 - w_sub_L) * Mp_sup
    M_minus = w_sub_R * Mm_sub + (1.0 - w_sub_R) * Mm_sup

    # Velocity splits
    u_plus_L  = c_bar * M_plus
    u_minus_R = c_bar * M_minus

    # g function (expansion/stagnation detector)
    beta_L_minus = anp.maximum(anp.minimum(M_L, 0.0), -1.0)
    beta_R_plus  = anp.minimum(anp.maximum(M_R, 0.0),  1.0)
    g = -beta_L_minus * beta_R_plus

    # Modified velocity splits
    V_plus_L  = (1.0 - g) * u_plus_L  + g * anp.sqrt(u_L**2 + 1e-14)
    V_minus_R = (1.0 - g) * u_minus_R - g * anp.sqrt(u_R**2 + 1e-14)

    # Mass flux
    m_dot = (0.5 * (rho_L * (u_L + anp.sqrt(V_plus_L**2 + 1e-14))
                   + rho_R * (u_R - anp.sqrt(V_minus_R**2 + 1e-14)))
             - chi / (2.0 * c_bar) * (p_R - p_L))

    # Pressure splitting (SLAU2) — smooth blend
    Pp_sub = 0.25 * (M_L + 1.0)**2 * (2.0 - M_L)
    Pp_sup = 0.5 * (1.0 + anp.tanh(100.0 * M_L))  # smooth Heaviside
    Pm_sub = 0.25 * (M_R - 1.0)**2 * (2.0 + M_R)
    Pm_sup = 0.5 * (1.0 - anp.tanh(100.0 * M_R))  # smooth Heaviside

    Pp_L = w_sub_L * Pp_sub + (1.0 - w_sub_L) * Pp_sup
    Pm_R = w_sub_R * Pm_sub + (1.0 - w_sub_R) * Pm_sup

    u_bar_mag = anp.sqrt(0.5 * (u_L**2 + u_R**2) + _eps)

    p_face = (0.5 * (p_L + p_R)
              + 0.5 * (Pp_L - Pm_R) * (p_L - p_R)
              + u_bar_mag * (Pp_L + Pm_R - 1.0) * 0.5 * (rho_L + rho_R) * c_bar)

    # Upwind-biased convective quantities
    is_pos = (m_dot >= 0)
    Y1_up = anp.where(is_pos, Y1_L, Y1_R)
    Y2_up = anp.where(is_pos, Y2_L, Y2_R)
    u_up  = anp.where(is_pos, u_L, u_R)
    H_up  = anp.where(is_pos, H_L, H_R)

    # Face fluxes (4-tuple matching HLLC interface)
    F_rY1 = m_dot * Y1_up
    F_rY2 = m_dot * Y2_up
    F_ru  = m_dot * u_up + p_face
    F_rE  = m_dot * H_up

    return (F_rY1, F_rY2, F_ru, F_rE)


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
