# solver/He2024/solver.py
# He2024 5-equation fully coupled implicit solver (standalone).
#
# Governing equations (2-species, 1D):
#   ∂(α₁ρ₁)/∂t + ∂(α₁ρ₁·u)/∂x = 0          [species 1 mass]
#   ∂(α₂ρ₂)/∂t + ∂(α₂ρ₂·u)/∂x = 0          [species 2 mass]
#   ∂(ρu)/∂t  + ∂(ρu²+p)/∂x  = 0            [momentum]
#   ∂(ρE)/∂t  + ∂((ρE+p)·u)/∂x = 0          [energy]
#   ∂α₁/∂t   + ∂(α₁·u)/∂x - (α₁+D₁)·∂u/∂x = 0  [volume fraction]
#
# Conservative variables: Q = {α₁ρ₁(N), α₂ρ₂(N), ρu(N), ρE(N), α₁(N)}
#
# Key insight: α₁ρ₁ = ρY₁ (partial density).
# Therefore, mixture_eos_anp(α₁ρ₁, α₂ρ₂, ρu, ρE, ph1, ph2) applies directly.
#
# Ref: CLAUDE.md, He & Tan (2024), Allaire-Massoni (2002), Kapila et al. (2001)

import numpy as np
from .common import (
    anp, _AD, ad_jacobian,
    _get_ph_params, mixture_eos_anp,
    hllc_flux_anp, rusanov_flux_anp, slau2_flux_anp, ausm_plus_flux_anp,
    physical_flux_anp, _bvd_reconstruct_anp,
    _ghost_anp, _ghost_anp2, _thinc_face_anp, _cicsam_face_anp,
)


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------

def pack_5eq(a1r1, a2r2, ru, rE, a1):
    """Pack Q = [α₁ρ₁(N) | α₂ρ₂(N) | ρu(N) | ρE(N) | α₁(N)]."""
    return anp.concatenate([a1r1, a2r2, ru, rE, a1])


def unpack_5eq(Q, N):
    """Unpack Q (5N,) → a1r1, a2r2, ru, rE, a1  each (N,)."""
    return Q[0:N], Q[N:2*N], Q[2*N:3*N], Q[3*N:4*N], Q[4*N:5*N]


# ---------------------------------------------------------------------------
# D₁ model coefficient (Kapila vs Allaire-Massoni)
# ---------------------------------------------------------------------------

def _compute_D1_anp(a1r1, a2r2, ru, rE, a1, ph1, ph2, model_type):
    """Compute D₁ coefficient for the α equation.

    Allaire-Massoni: D₁ = 0
    Kapila: D₁ = α₁ · (ρc²/(ρ₁c₁²) - 1)

    Returns array (N,).
    """
    _eps = 1e-300

    if model_type == 'allaire':
        return anp.zeros_like(a1)

    # Kapila: need individual phase sound speeds
    g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)

    p, T, u_vel, c_wood = mixture_eos_anp(a1r1, a2r2, ru, rE, ph1, ph2)

    rho = a1r1 + a2r2
    a2 = 1.0 - a1

    rho1 = anp.where(a1 > _eps, a1r1 / (a1 + _eps), a1r1 / _eps)
    rho2 = anp.where(a2 > _eps, a2r2 / (a2 + _eps), a2r2 / _eps)

    c1_sq = g1 * (p + pinf1) / (rho1 * (1.0 - b1 * rho1) + _eps)
    c2_sq = g2 * (p + pinf2) / (rho2 + _eps)
    c1_sq = anp.maximum(c1_sq, _eps)
    c2_sq = anp.maximum(c2_sq, _eps)

    # Wood mixture sound speed squared: 1/(ρc²) = α₁/(ρ₁c₁²) + α₂/(ρ₂c₂²)
    inv_rho_c2 = a1 / (rho1 * c1_sq + _eps) + a2 / (rho2 * c2_sq + _eps)
    rho_c2_mix = 1.0 / (inv_rho_c2 + _eps)

    D1 = a1 * (rho_c2_mix / (rho1 * c1_sq + _eps) - 1.0)
    return D1


# ---------------------------------------------------------------------------
# MMACM helper: smooth absolute value (autograd-compatible)
# ---------------------------------------------------------------------------

def _smooth_abs_anp(x, delta=1e-12):
    """Smooth approximation of |x| = sqrt(x^2 + delta^2).

    Ref: He & Tan 2024, JCP 513:113192 — MMACM interface detector.
    autograd-differentiable everywhere (no kink at x=0).
    """
    return anp.sqrt(x * x + delta * delta)


# ---------------------------------------------------------------------------
# MMACM-Ex: characteristic function H_k (Eq. 32, Zhao et al. 2025)
# ---------------------------------------------------------------------------

def _hk_characteristic_anp(a1, bc_l, bc_r, eps_interface=1e-4, steepness=200.0):
    """Compute MMACM-Ex characteristic function H_k at cell centers (N,).

    H_k ≈ 1 at monotone interface (r≈1), ≈ 0 in pure-phase or extrema.

    Eq. 32:  H_k = (1 - ((1-|r|)/(1+|r|))^4) · |n_x|
    where r = (α_i - α_{i-1}) / (α_{i+1} - α_i)
    In 1D, |n_x| = 1.

    Interface detection (Eq. 19): ε < α < 1-ε AND monotone (dL·dR > 0).
    Smooth versions for autograd compatibility.

    Ref: Zhao et al. 2025, Phys. Fluids 37:076157, Eq. 32.
    """
    N = len(a1)
    a1_ext = _ghost_anp(a1, bc_l, bc_r)   # (N+2,)
    dL = a1_ext[1:N+1] - a1_ext[0:N]      # α_i - α_{i-1}  (N,)
    dR = a1_ext[2:N+2] - a1_ext[1:N+1]    # α_{i+1} - α_i  (N,)

    # Slope ratio r = dL / dR (safe division preserving sign)
    abs_dR = _smooth_abs_anp(dR)
    sign_dR = anp.where(dR >= 0, 1.0, -1.0)
    r = dL * sign_dR / (abs_dR + 1e-30)
    abs_r = _smooth_abs_anp(r)

    # H_k = 1 - ((1-|r|)/(1+|r|))^4,  |n_x|=1 in 1D
    ratio = (1.0 - abs_r) / (1.0 + abs_r + 1e-30)
    H_raw = 1.0 - ratio ** 4

    # Interface detection (Eq. 19, smooth version for autograd)
    # (a) ε < α < 1-ε
    w_lo = 0.5 * (1.0 + anp.tanh(steepness * (a1 - eps_interface)))
    w_hi = 0.5 * (1.0 + anp.tanh(steepness * (1.0 - eps_interface - a1)))
    # (b) monotone: dL·dR > 0
    mono_arg = dL * dR / (dL * dL + dR * dR + 1e-30)
    mono = anp.maximum(anp.tanh(steepness * mono_arg), 0.0)

    H = H_raw * w_lo * w_hi * mono
    return anp.minimum(anp.maximum(H, 0.0), 1.0)


# ---------------------------------------------------------------------------
# MMACM (legacy): interface detector κ (Eq. 57, He & Tan 2024)
# ---------------------------------------------------------------------------

def _kappa_anp(a1, bc_l, bc_r, q_exp=2,
               eps_cutoff=1e-2, eps_sense=1e-12, steepness=200.0):
    """Compute MMACM interface indicator kappa (N,).

    kappa ≈ 1 near a sharp interface, ≈ 0 in pure-phase or smooth regions.

    Eq. 57:  kappa = 1 - |dR - dL|^q / (dL + dR)^q
    with smooth cutoff via tanh to suppress corrections in near-pure phases.

    Ref: He & Tan 2024, JCP 513:113192, Eq. 57.
    """
    N = len(a1)
    a1_ext = _ghost_anp(a1, bc_l, bc_r)   # (N+2,)
    dL = a1_ext[1:N+1] - a1_ext[0:N]      # Δα left  (N,)
    dR = a1_ext[2:N+2] - a1_ext[1:N+1]    # Δα right (N,)

    abs_dL = _smooth_abs_anp(dL)
    abs_dR = _smooth_abs_anp(dR)

    denom = (abs_dL + abs_dR) ** q_exp + eps_sense
    kappa_raw = 1.0 - _smooth_abs_anp(abs_dR - abs_dL) ** q_exp / denom

    # Smooth cutoff: suppress kappa in pure-phase regions
    w_lo = 0.5 * (1.0 + anp.tanh(steepness * (a1 - eps_cutoff)))
    w_hi = 0.5 * (1.0 + anp.tanh(steepness * (1.0 - eps_cutoff - a1)))
    kappa = kappa_raw * w_lo * w_hi

    return kappa


# ---------------------------------------------------------------------------
# MMACM: harmonic limiter downwind face reconstruction (Eq. 59-60)
# ---------------------------------------------------------------------------

def _harmonic_limiter_face_anp(a1, u_face, bc_l, bc_r, beta_lim=2.9):
    """Harmonic limiter downwind face reconstruction of alpha1 (N+1,).

    Uses the same stencil convention as _thinc_alpha_he2024 (ng=2 ghost cells).

    Eq. 59-60:  psi(r) = (|r| + r) / (1/beta + r),  a1_face = a_D + 0.5*psi*(a_A - a_D)
    where r = (a_D - a_UU) / (a_A - a_D) is the slope ratio (downwind bias).

    Ref: He & Tan 2024, JCP 513:113192, Eqs. 59-60.
    """
    N = len(a1)
    a1_ext2 = _ghost_anp2(a1, bc_l, bc_r)   # (N+4,)
    N_faces = N + 1
    ng = 2
    idx = np.arange(N_faces)

    # Stencil indices (identical to _thinc_alpha_he2024)
    D_pos  = ng + idx - 1;  A_pos  = ng + idx;      UU_pos = ng + idx - 2
    D_neg  = ng + idx;       A_neg  = ng + idx - 1;  UU_neg = ng + idx + 1

    a_D  = anp.where(u_face >= 0, a1_ext2[D_pos],  a1_ext2[D_neg])
    a_A  = anp.where(u_face >= 0, a1_ext2[A_pos],  a1_ext2[A_neg])
    a_UU = anp.where(u_face >= 0, a1_ext2[UU_pos], a1_ext2[UU_neg])

    # Slope ratio r = (a_D - a_UU) / (a_A - a_D)
    da_num = a_D - a_UU
    da_den = a_A - a_D
    # Safe denominator: avoid division by zero while preserving sign
    da_den_safe = anp.where(anp.abs(da_den) > 1e-30, da_den,
                            1e-30 * anp.where(da_den >= 0, 1.0, -1.0))
    r = da_num / da_den_safe

    # Harmonic limiter: psi(r) = (|r| + r) / (1/beta + r), clipped to [0, 2]
    abs_r = _smooth_abs_anp(r)
    psi_raw = (abs_r + r) / (1.0 / beta_lim + r + 1e-30)
    psi = anp.minimum(anp.maximum(psi_raw, 0.0), 2.0)

    # Downwind-biased face value (Eq. 60)
    a1_face_down = a_D + 0.5 * psi * (a_A - a_D)
    a1_face_down = anp.maximum(anp.minimum(a1_face_down, 1.0), 0.0)

    return a1_face_down


# ---------------------------------------------------------------------------
# MMACM: full correction flux
#   sharpening='mmacm_ex' → MMACM-Ex (Zhao 2025): H_k + pure downwind
#   sharpening='thinc'    → legacy MMACM (He & Tan 2024): κ + THINC
#   sharpening='cicsam'   → legacy MMACM: κ + CICSAM
# ---------------------------------------------------------------------------

def _mmacm_correction_flux(a1, a1r1, a2r2, p, T, u_vel, u_face,
                            a1_face_upwind, ph1, ph2, bc_l, bc_r,
                            mmacm_beta=4.0, kappa_q=2,
                            dt=None, dx=None, sharpening='mmacm_ex'):
    """Compute MMACM sharpening correction fluxes G at all faces (N+1,).

    sharpening='mmacm_ex':
        MMACM-Ex (Zhao et al. 2025, Phys. Fluids 37:076157)
        Step 1: H_k characteristic function (Eq. 32) → upwind face selection
        Step 2: Pure 1st-order downwind α₁ (Eq. 30)
        Step 3: J_k = H̃ · [u·α̂_down - u·α_upwind]  (Eq. 29)

    sharpening='thinc'|'cicsam'|'bvd':
        Legacy MMACM (He & Tan 2024, JCP 513:113192)
        κ function (Eq. 57) + THINC/CICSAM/BVD downwind

    Conservation consistency (Eq. 55/27) is identical for all variants.
    """
    _eps = 1e-300
    N = len(a1)

    if sharpening == 'mmacm_ex':
        # --- MMACM-Ex: H_k + pure downwind (Zhao 2025) ---
        H_cell = _hk_characteristic_anp(a1, bc_l, bc_r)  # (N,)
        H_ext = _ghost_anp(H_cell, bc_l, bc_r)           # (N+2,)
        # Upwind selection of H at faces (Eq. 28)
        char_face = anp.where(u_face >= 0, H_ext[0:N+1], H_ext[1:N+2])

        # Pure 1st-order downwind (Eq. 30)
        a1_ext1 = _ghost_anp(a1, bc_l, bc_r)  # (N+2,)
        a1_down = anp.where(u_face >= 0,
                            a1_ext1[1:N+2],    # α_{i+1} for u>0
                            a1_ext1[0:N+1])    # α_i     for u≤0
    else:
        # --- Legacy MMACM: κ + various downwind schemes ---
        kappa_cell = _kappa_anp(a1, bc_l, bc_r, q_exp=kappa_q)
        kappa_ext  = _ghost_anp(kappa_cell, bc_l, bc_r)
        char_face = 0.5 * (kappa_ext[0:N+1] + kappa_ext[1:N+2])  # avg

        a1_ext1 = _ghost_anp(a1, bc_l, bc_r)
        a1_ext2 = _ghost_anp2(a1, bc_l, bc_r)
        if sharpening == 'cicsam' and dt is not None and dx is not None:
            a1_down = _cicsam_face_anp(a1_ext2, u_face, dt, dx)
        elif sharpening == 'bvd':
            a1L_bvd, a1R_bvd = _bvd_reconstruct_anp(a1_ext1, N, beta=mmacm_beta)
            a1_down = anp.where(u_face >= 0, a1L_bvd, a1R_bvd)
        else:
            a1_down = _thinc_alpha_he2024(a1_ext2, u_face, beta=mmacm_beta)

    # --- G^α₁ = char_face · u · (α_down - α_upwind) ---
    G_alpha = char_face * u_face * (a1_down - a1_face_upwind)  # (N+1,)

    # Suppress correction in near-pure cells
    a1_up_check = anp.where(u_face >= 0,
                            _ghost_anp(a1, bc_l, bc_r)[0:N+1],
                            _ghost_anp(a1, bc_l, bc_r)[1:N+2])
    _m_eps = 1e-3; _m_steep = 100.0
    mask_lo = 0.5 * (1.0 + anp.tanh(_m_steep * (a1_up_check - _m_eps)))
    mask_hi = 0.5 * (1.0 + anp.tanh(_m_steep * (1.0 - _m_eps - a1_up_check)))
    G_alpha = G_alpha * mask_lo * mask_hi

    # --- Upwind selection of cell quantities for consistency (Eq. 55) ---
    def _upwind_ext(arr):
        ext = _ghost_anp(arr, bc_l, bc_r)   # (N+2,)
        return anp.where(u_face >= 0, ext[0:N+1], ext[1:N+2])

    a1_up   = _upwind_ext(a1)
    a1r1_up = _upwind_ext(a1r1)
    a2r2_up = _upwind_ext(a2r2)
    p_up    = _upwind_ext(p)
    T_up    = _upwind_ext(T)
    u_up    = _upwind_ext(u_vel)

    # Phase densities from upwind partial densities
    # Higher floor + clamp to prevent enormous phase densities at interface
    _alpha_floor = 1e-4
    rho1_up = a1r1_up / anp.maximum(a1_up,        _alpha_floor)
    rho2_up = a2r2_up / anp.maximum(1.0 - a1_up,  _alpha_floor)
    rho1_up = anp.minimum(rho1_up, 1e6)
    rho2_up = anp.minimum(rho2_up, 1e6)

    # Phase specific internal energies from EOS at (p_up, T_up)
    g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)

    e1_up = kv1 * T_up * (p_up + g1 * pinf1) / (p_up + pinf1 + _eps) + eta1
    e2_up = kv2 * T_up * (p_up + g2 * pinf2) / (p_up + pinf2 + _eps) + eta2

    # Phase total energies
    E1_up = e1_up + 0.5 * u_up ** 2
    E2_up = e2_up + 0.5 * u_up ** 2

    # Conservation consistency corrections (Eq. 55)
    G_a1r1 =  rho1_up * G_alpha
    G_a2r2 = -rho2_up * G_alpha
    G_ru   = (rho1_up - rho2_up) * u_up * G_alpha
    G_rE   = (rho1_up * E1_up - rho2_up * E2_up) * G_alpha

    return G_a1r1, G_a2r2, G_ru, G_rE, G_alpha


# ---------------------------------------------------------------------------
# THINC with global bounds [0, 1] — no local q_lo/q_hi limitation
# ---------------------------------------------------------------------------

def _thinc_alpha_he2024(a1_ext2, u_face, beta=4.0):
    """THINC face reconstruction for alpha1 with global bounds [0, 1].

    Uses direction detection from neighbor gradient (q_A - q_UU) to decide
    compression direction. Compresses toward 1 at rising interfaces and
    toward 0 at falling interfaces. No local-bounds limitation.

    Parameters
    ----------
    a1_ext2 : array (N+4,)
        Extended alpha1 with 2 ghost cells on each side.
    u_face : array (N+1,)
        Face velocities.
    beta : float
        Sharpness parameter (higher = sharper). Default 4.0.

    Returns
    -------
    a1_face : array (N+1,)
        THINC-reconstructed face alpha1 values.
    """
    ng = 2
    N_faces = len(u_face)
    idx = np.arange(N_faces)

    # Stencil: Donor (upwind of face), Acceptor (downwind), UU (upwind-upwind)
    D_pos  = ng + idx - 1;   A_pos  = ng + idx;      UU_pos = ng + idx - 2
    D_neg  = ng + idx;        A_neg  = ng + idx - 1;  UU_neg = ng + idx + 1

    q_D  = anp.where(u_face >= 0, a1_ext2[D_pos],  a1_ext2[D_neg])
    q_A  = anp.where(u_face >= 0, a1_ext2[A_pos],  a1_ext2[A_neg])
    q_UU = anp.where(u_face >= 0, a1_ext2[UU_pos], a1_ext2[UU_neg])

    # Direction detection: smooth sign of (q_A - q_UU) * sign(u)
    _gamma = 100.0
    sigma = anp.tanh(_gamma * (q_A - q_UU))
    u_sign = anp.tanh(_gamma * u_face)
    theta = sigma * u_sign

    # THINC compression with global bounds [0, 1]
    # theta > 0 (rising interface): compress toward 1
    # theta < 0 (falling interface): compress toward 0
    q_thinc_hi = 0.5 + 0.5 * anp.tanh(beta * q_D)
    q_thinc_lo = 0.5 - 0.5 * anp.tanh(beta * (1.0 - q_D))

    alpha_blend = 0.5 * (1.0 + theta)
    q_thinc = alpha_blend * q_thinc_hi + (1.0 - alpha_blend) * q_thinc_lo

    # Smooth activation: active near interface (q_D intermediate),
    # inactive in pure-phase regions (q_D ≈ 0 or 1)
    w_act = anp.tanh(50.0 * q_D * (1.0 - q_D) * 4.0)

    q_face = w_act * q_thinc + (1.0 - w_act) * q_D
    q_face = anp.maximum(anp.minimum(q_face, 1.0), 0.0)

    return q_face


# ---------------------------------------------------------------------------
# THINC cache helper: extract face values from current Q_k (numpy only)
# ---------------------------------------------------------------------------

def _compute_thinc_face_from_Q(Q_k, N, bc_l, bc_r, thinc_beta):
    """Compute THINC face values from current state Q_k (numpy only).

    Used by fd_sparse and jfnk methods to update the THINC cache at each
    Newton iteration rather than freezing it from Q_n.

    Parameters
    ----------
    Q_k : ndarray (5N,)
        Current Newton iterate (conservative variables).
    N : int
        Number of cells.
    bc_l, bc_r : str
        Boundary condition types.
    thinc_beta : float
        THINC sharpness parameter.

    Returns
    -------
    a1_face : ndarray (N+1,)
        THINC-reconstructed face alpha1.
    """
    a1_k = np.array(Q_k[4*N:5*N], dtype=float)
    if bc_l == 'periodic':
        a1_ext2 = np.zeros(N + 4)
        a1_ext2[2:N+2] = a1_k
        a1_ext2[0:2] = a1_k[-2:]
        a1_ext2[N+2:N+4] = a1_k[0:2]
    else:
        a1_ext2 = np.zeros(N + 4)
        a1_ext2[2:N+2] = a1_k
        a1_ext2[0:2] = a1_k[0]
        a1_ext2[N+2:N+4] = a1_k[-1]
    rho_k = np.array(Q_k[0:N] + Q_k[N:2*N], dtype=float)
    u_k = np.array(Q_k[2*N:3*N], dtype=float) / (rho_k + 1e-300)
    if bc_l == 'periodic':
        u_ext = np.concatenate([[u_k[-1]], u_k, [u_k[0]]])
    else:
        u_ext = np.concatenate([[u_k[0]], u_k, [u_k[-1]]])
    u_face = 0.5 * (u_ext[0:N+1] + u_ext[1:N+2])
    return np.array(_thinc_alpha_he2024(a1_ext2, u_face, beta=thinc_beta),
                    dtype=float)


# ---------------------------------------------------------------------------
# FD Sparse Jacobian: 15-color graph coloring for block-tridiagonal structure
# ---------------------------------------------------------------------------

def _fd_sparse_jacobian(res_func, Q_k, N, eps_fd=1e-7):
    """Sparse Jacobian via FD with 15-color graph coloring (block-tridiagonal).

    Uses 5 equations × 3-cell stride coloring: perturbs groups of non-adjacent
    columns simultaneously so each HLLC stencil (3-cell wide) sees at most one
    perturbation per group.

    Parameters
    ----------
    res_func : callable
        Residual function R(Q) → ndarray (5N,).
    Q_k : ndarray (5N,)
        Current Newton iterate.
    N : int
        Number of cells.
    eps_fd : float
        Finite difference step size (relative to max(|Q|, 1)).

    Returns
    -------
    J : scipy.sparse.csc_matrix  (5N, 5N)
    """
    from scipy.sparse import lil_matrix
    n_eq = 5
    n_dof = n_eq * N
    R0 = np.array(res_func(Q_k), dtype=float)
    J = lil_matrix((n_dof, n_dof))

    # THINC uses 5-cell stencil (UU, D, A + faces affect ±2 cells).
    # Use stride=5, giving 5 eq × 5 offsets = 25 colors.
    stride = 5
    for eq in range(n_eq):
        for offset in range(stride):
            cells = np.arange(offset, N, stride)
            if len(cells) == 0:
                continue
            col_indices = eq * N + cells
            Q_pert = Q_k.copy()
            eps_vec = eps_fd * np.maximum(np.abs(Q_k[col_indices]), 1.0)
            Q_pert[col_indices] += eps_vec
            R_pert = np.array(res_func(Q_pert), dtype=float)
            dR = R_pert - R0
            for k, cell_j in enumerate(cells):
                # 5-cell stencil: cells {j-2, j-1, j, j+1, j+2}
                for cell_i in range(max(0, cell_j - 2), min(N, cell_j + 3)):
                    for row_eq in range(n_eq):
                        row = row_eq * N + cell_i
                        val = dR[row] / eps_vec[k]
                        if abs(val) > 1e-30:
                            J[row, col_indices[k]] = val
    return J.tocsc()


# ---------------------------------------------------------------------------
# JFNK linear solve: matrix-free J*v with sparse preconditioner
# ---------------------------------------------------------------------------

def _jfnk_solve(res_func, Q_k, R0, prec_J_sp=None,
                gmres_tol=1e-8, gmres_maxiter=200):
    """JFNK: solve J*dQ = -R using matrix-free J*v products.

    The Jacobian-vector product J*v is approximated by a finite difference:
        J*v ≈ (R(Q_k + ε*v) - R(Q_k)) / ε

    An optional sparse preconditioner (from FD sparse Jacobian) can be
    supplied to accelerate GMRES convergence.

    Parameters
    ----------
    res_func : callable
        Residual function R(Q) → ndarray (5N,). Used for J*v products.
        Should use LIVE THINC (recomputed from Q each call).
    Q_k : ndarray (5N,)
        Current Newton iterate.
    R0 : ndarray (5N,)
        Residual at Q_k (pre-computed).
    prec_J_sp : scipy.sparse matrix or None
        Sparse Jacobian for ILU preconditioner (from frozen THINC residual).
    gmres_tol : float
        GMRES tolerance.
    gmres_maxiter : int
        Maximum GMRES iterations.

    Returns
    -------
    dQ : ndarray (5N,)
        Newton correction.
    info : int
        GMRES convergence info (0 = success).
    """
    from scipy.sparse.linalg import gmres, LinearOperator, spilu
    n_dof = len(Q_k)
    Q_norm = np.linalg.norm(Q_k)

    def matvec_Jv(v):
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-300:
            return np.zeros_like(v)
        eps = np.sqrt(np.finfo(float).eps) * max(Q_norm, 1.0) / v_norm
        R_pert = np.array(res_func(Q_k + eps * v), dtype=float)
        return (R_pert - R0) / eps

    J_op = LinearOperator((n_dof, n_dof), matvec=matvec_Jv)

    M = None
    if prec_J_sp is not None:
        try:
            ilu = spilu(prec_J_sp, fill_factor=10)
            M = LinearOperator((n_dof, n_dof), matvec=ilu.solve)
        except Exception:
            pass

    dQ, info = gmres(J_op, -R0, M=M, atol=gmres_tol, maxiter=gmres_maxiter)
    return dQ, info


# ---------------------------------------------------------------------------
# Residual factory (5-equation, autograd-compatible)
# ---------------------------------------------------------------------------

def make_residual_he2024(Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                          flux_type='hllc', model_type='allaire',
                          use_thinc=False, thinc_beta=50.0,
                          _thinc_cache=None, live_thinc=False,
                          use_mmacm=False, mmacm_beta=4.0,
                          mmacm_sharpening='thinc',
                          use_bvd=False, bvd_beta=1.6,
                          _bvd_cache=None):
    """Build autograd-differentiable residual for one timestep of He2024 solver.

    R = 0:
      R[0:N]   = (α₁ρ₁ - α₁ρ₁ⁿ)/dt + div(F_α₁ρ₁)/dx
      R[N:2N]  = (α₂ρ₂ - α₂ρ₂ⁿ)/dt + div(F_α₂ρ₂)/dx
      R[2N:3N] = (ρu   - ρuⁿ)   /dt + div(F_ρu)   /dx
      R[3N:4N] = (ρE   - ρEⁿ)   /dt + div(F_ρE)   /dx
      R[4N:5N] = (α₁   - α₁ⁿ)  /dt + div(α₁u)/dx - (α₁+D₁)·div(u)/dx

    The α₁ equation uses:
      div(α₁u)/dx  with upwind α₁ at faces
      div(u)/dx    with arithmetic-mean u at faces

    When use_thinc=True and _thinc_cache is provided:
      - _thinc_cache['a1_face'] holds lagged THINC α₁ face values (numpy)
      - These are treated as constants by autograd (not differentiated)
      - Newton updates THINC values at each iteration via the cache
      - TDU: (p,T,α₁_thinc) → EOS → consistent donor state → HLLC

    When use_thinc=True and live_thinc=True:
      - THINC α₁ face values are recomputed from Q's α₁ inside the residual.
      - Uses numpy for the THINC computation (not differentiable), so the
        result enters the anp computation as a constant.
      - Intended for JFNK J*v matrix-free products where Q changes each call.
      - live_thinc=True takes priority over _thinc_cache.

    Parameters
    ----------
    live_thinc : bool
        If True, recompute THINC from Q inside each residual evaluation.
        Only effective when use_thinc=True. Default False.
    """
    _Q_n = anp.array(Q_n, dtype=float)

    def residual(Q):
        a1r1, a2r2, ru, rE, a1 = unpack_5eq(Q, N)

        # --- EOS (pass alpha1 for direct SG formula) ---
        p, T, u_vel, c_wood = mixture_eos_anp(a1r1, a2r2, ru, rE, ph1, ph2,
                                               alpha1_in=a1)

        # --- HLLC/Rusanov flux for 4 conservation equations ---
        _mmacm_G_alpha = None  # set in else branch when use_mmacm=True
        _u_hllc = None         # HLLC face velocity for α equation (set in HLLC branch)
        if use_thinc and live_thinc:
            # Live THINC: recompute THINC from current Q's α₁ inside residual.
            # Used by JFNK for matrix-free J*v products.
            # _thinc_alpha_he2024 uses np.arange, so THINC is computed in numpy
            # (not differentiable); the result enters anp computation as constant.
            a1_np = np.array(a1, dtype=float)
            if bc_l == 'periodic':
                a1_ext2_live = np.zeros(N + 4)
                a1_ext2_live[2:N+2] = a1_np
                a1_ext2_live[0:2] = a1_np[-2:]
                a1_ext2_live[N+2:N+4] = a1_np[0:2]
            else:
                a1_ext2_live = np.zeros(N + 4)
                a1_ext2_live[2:N+2] = a1_np
                a1_ext2_live[0:2] = a1_np[0]
                a1_ext2_live[N+2:N+4] = a1_np[-1]
            u_np = np.array(u_vel, dtype=float)
            if bc_l == 'periodic':
                u_ext_live = np.concatenate([[u_np[-1]], u_np, [u_np[0]]])
            else:
                u_ext_live = np.concatenate([[u_np[0]], u_np, [u_np[-1]]])
            u_face_live = 0.5 * (u_ext_live[0:N+1] + u_ext_live[1:N+2])
            a1_thinc = np.array(
                _thinc_alpha_he2024(a1_ext2_live, u_face_live, beta=thinc_beta),
                dtype=float)

            # TDU donor reconstruction (same as frozen branch)
            u_ext1 = _ghost_anp(u_vel, bc_l, bc_r)
            u_face_thinc = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])

            p_ext = _ghost_anp(p, bc_l, bc_r)
            T_ext = _ghost_anp(T, bc_l, bc_r)

            p_don = anp.where(u_face_thinc >= 0, p_ext[0:N+1], p_ext[1:N+2])
            T_don = anp.where(u_face_thinc >= 0, T_ext[0:N+1], T_ext[1:N+2])

            a1_don = a1_thinc  # numpy constant — not differentiated
            a2_don = 1.0 - a1_don

            g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
            g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)
            _eps = 1e-300

            rho1_don = (p_don + pinf1) / (kv1 * (g1 - 1.0) * T_don + b1 * (p_don + pinf1) + _eps)
            rho2_don = (p_don + pinf2) / (kv2 * (g2 - 1.0) * T_don + b2 * (p_don + pinf2) + _eps)

            a1r1_don = a1_don * rho1_don
            a2r2_don = a2_don * rho2_don
            rho_don  = a1r1_don + a2r2_don

            u_don_v = anp.where(u_face_thinc >= 0, u_ext1[0:N+1], u_ext1[1:N+2])

            e1_don = kv1 * T_don * (p_don + g1 * pinf1) / (p_don + pinf1 + _eps) + eta1
            e2_don = kv2 * T_don * (p_don + g2 * pinf2) / (p_don + pinf2 + _eps) + eta2

            Y1_don = a1r1_don / (rho_don + _eps)
            e_mix_don = Y1_don * e1_don + (1.0 - Y1_don) * e2_don

            ru_don  = rho_don * u_don_v
            rE_don  = rho_don * (e_mix_don + 0.5 * u_don_v ** 2)

            c1_sq_don = g1 * (p_don + pinf1) / (rho1_don * (1.0 - b1 * rho1_don) + _eps)
            c2_sq_don = g2 * (p_don + pinf2) / (rho2_don + _eps)
            c1_sq_don = anp.maximum(c1_sq_don, _eps)
            c2_sq_don = anp.maximum(c2_sq_don, _eps)
            inv_rc2_don = a1_don / (rho1_don * c1_sq_don + _eps) + a2_don / (rho2_don * c2_sq_don + _eps)
            c_don = anp.sqrt(1.0 / (rho_don * inv_rc2_don + _eps))

            vars_ext = [_ghost_anp(v, bc_l, bc_r)
                        for v in [a1r1, a2r2, ru, rE, p, c_wood]]
            L_std = [v[0:N+1] for v in vars_ext]
            R_std = [v[1:N+2] for v in vars_ext]

            is_right = (u_face_thinc >= 0)
            new_L = [anp.where(is_right, a1r1_don, L_std[0]),
                     anp.where(is_right, a2r2_don, L_std[1]),
                     anp.where(is_right, ru_don,   L_std[2]),
                     anp.where(is_right, rE_don,   L_std[3]),
                     anp.where(is_right, p_don,    L_std[4]),
                     anp.where(is_right, c_don,    L_std[5])]
            new_R = [anp.where(is_right, R_std[0], a1r1_don),
                     anp.where(is_right, R_std[1], a2r2_don),
                     anp.where(is_right, R_std[2], ru_don),
                     anp.where(is_right, R_std[3], rE_don),
                     anp.where(is_right, R_std[4], p_don),
                     anp.where(is_right, R_std[5], c_don)]

            if flux_type == 'hllc':
                Ff_full = hllc_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                        new_L[4], new_L[5],
                                        new_R[0], new_R[1], new_R[2], new_R[3],
                                        new_R[4], new_R[5])
                Ff = Ff_full[:4]; _u_hllc = Ff_full[4]
            else:
                Ff = rusanov_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                      new_L[4], new_L[5],
                                      new_R[0], new_R[1], new_R[2], new_R[3],
                                      new_R[4], new_R[5])
                _u_hllc = None

            a1_face_adv = a1_thinc

        elif use_thinc and _thinc_cache is not None and _thinc_cache.get('a1_face') is not None:
            # TDU with lagged THINC: α₁ face values frozen from Q_k.
            # autograd sees them as constants → Jacobian stays well-conditioned.
            a1_thinc = _thinc_cache['a1_face']  # numpy array, frozen

            u_ext1  = _ghost_anp(u_vel, bc_l, bc_r)
            u_face_thinc = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])

            # TDU donor reconstruction: (p_don, T_don, α₁_thinc) → EOS → state
            p_ext  = _ghost_anp(p, bc_l, bc_r)
            T_ext  = _ghost_anp(T, bc_l, bc_r)

            p_don  = anp.where(u_face_thinc >= 0, p_ext[0:N+1],  p_ext[1:N+2])
            T_don  = anp.where(u_face_thinc >= 0, T_ext[0:N+1],  T_ext[1:N+2])

            # a1_thinc is numpy (frozen) — autograd treats as constant
            a1_don = a1_thinc
            a2_don = 1.0 - a1_don

            g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
            g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)
            _eps = 1e-300

            rho1_don = (p_don + pinf1) / (kv1 * (g1 - 1.0) * T_don + b1 * (p_don + pinf1) + _eps)
            rho2_don = (p_don + pinf2) / (kv2 * (g2 - 1.0) * T_don + b2 * (p_don + pinf2) + _eps)

            a1r1_don = a1_don * rho1_don
            a2r2_don = a2_don * rho2_don
            rho_don  = a1r1_don + a2r2_don

            u_don_v = anp.where(u_face_thinc >= 0, u_ext1[0:N+1], u_ext1[1:N+2])

            e1_don = kv1 * T_don * (p_don + g1 * pinf1) / (p_don + pinf1 + _eps) + eta1
            e2_don = kv2 * T_don * (p_don + g2 * pinf2) / (p_don + pinf2 + _eps) + eta2

            Y1_don = a1r1_don / (rho_don + _eps)
            e_mix_don = Y1_don * e1_don + (1.0 - Y1_don) * e2_don

            ru_don  = rho_don * u_don_v
            rE_don  = rho_don * (e_mix_don + 0.5 * u_don_v ** 2)

            c1_sq_don = g1 * (p_don + pinf1) / (rho1_don * (1.0 - b1 * rho1_don) + _eps)
            c2_sq_don = g2 * (p_don + pinf2) / (rho2_don + _eps)
            c1_sq_don = anp.maximum(c1_sq_don, _eps)
            c2_sq_don = anp.maximum(c2_sq_don, _eps)
            inv_rc2_don = a1_don / (rho1_don * c1_sq_don + _eps) + a2_don / (rho2_don * c2_sq_don + _eps)
            c_don = anp.sqrt(1.0 / (rho_don * inv_rc2_don + _eps))

            # Standard ghost extension for non-donor side
            vars_ext = [_ghost_anp(v, bc_l, bc_r)
                        for v in [a1r1, a2r2, ru, rE, p, c_wood]]
            L_std = [v[0:N+1] for v in vars_ext]
            R_std = [v[1:N+2] for v in vars_ext]

            is_right = (u_face_thinc >= 0)
            new_L = [anp.where(is_right, a1r1_don, L_std[0]),
                     anp.where(is_right, a2r2_don, L_std[1]),
                     anp.where(is_right, ru_don,   L_std[2]),
                     anp.where(is_right, rE_don,   L_std[3]),
                     anp.where(is_right, p_don,    L_std[4]),
                     anp.where(is_right, c_don,    L_std[5])]
            new_R = [anp.where(is_right, R_std[0], a1r1_don),
                     anp.where(is_right, R_std[1], a2r2_don),
                     anp.where(is_right, R_std[2], ru_don),
                     anp.where(is_right, R_std[3], rE_don),
                     anp.where(is_right, R_std[4], p_don),
                     anp.where(is_right, R_std[5], c_don)]

            if flux_type == 'hllc':
                Ff_full = hllc_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                        new_L[4], new_L[5],
                                        new_R[0], new_R[1], new_R[2], new_R[3],
                                        new_R[4], new_R[5])
                Ff = Ff_full[:4]; _u_hllc = Ff_full[4]
            else:
                Ff = rusanov_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                      new_L[4], new_L[5],
                                      new_R[0], new_R[1], new_R[2], new_R[3],
                                      new_R[4], new_R[5])
                _u_hllc = None

            # α₁ face values: same lagged THINC (consistent with HLLC donor)
            a1_face_adv = a1_thinc

        elif use_thinc:
            # Autograd path: w-blending THINC with donor reconstruction.
            # _thinc_face_anp (w-blending) is autograd-differentiable.
            a1_ext2 = _ghost_anp2(a1, bc_l, bc_r)
            u_ext1  = _ghost_anp(u_vel, bc_l, bc_r)
            u_face_thinc = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])
            a1_thinc = _thinc_face_anp(a1_ext2, u_face_thinc, beta=thinc_beta)

            p_ext  = _ghost_anp(p, bc_l, bc_r)
            T_ext  = _ghost_anp(T, bc_l, bc_r)
            p_don  = anp.where(u_face_thinc >= 0, p_ext[0:N+1],  p_ext[1:N+2])
            T_don  = anp.where(u_face_thinc >= 0, T_ext[0:N+1],  T_ext[1:N+2])

            a1_don = a1_thinc
            a2_don = 1.0 - a1_don

            g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
            g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)
            _eps = 1e-300

            rho1_don = (p_don + pinf1) / (kv1*(g1-1.0)*T_don + b1*(p_don+pinf1) + _eps)
            rho2_don = (p_don + pinf2) / (kv2*(g2-1.0)*T_don + b2*(p_don+pinf2) + _eps)
            a1r1_don = a1_don * rho1_don
            a2r2_don = a2_don * rho2_don
            rho_don  = a1r1_don + a2r2_don
            u_don_v = anp.where(u_face_thinc >= 0, u_ext1[0:N+1], u_ext1[1:N+2])

            e1_don = kv1*T_don*(p_don + g1*pinf1)/(p_don + pinf1 + _eps) + eta1
            e2_don = kv2*T_don*(p_don + g2*pinf2)/(p_don + pinf2 + _eps) + eta2
            Y1_don = a1r1_don / (rho_don + _eps)
            e_mix_don = Y1_don*e1_don + (1.0 - Y1_don)*e2_don
            ru_don  = rho_don * u_don_v
            rE_don  = rho_don * (e_mix_don + 0.5*u_don_v**2)

            c1_sq_don = g1*(p_don+pinf1)/(rho1_don*(1.0 - b1*rho1_don) + _eps)
            c2_sq_don = g2*(p_don+pinf2)/(rho2_don + _eps)
            c1_sq_don = anp.maximum(c1_sq_don, _eps)
            c2_sq_don = anp.maximum(c2_sq_don, _eps)
            inv_rc2_don = a1_don/(rho1_don*c1_sq_don+_eps) + a2_don/(rho2_don*c2_sq_don+_eps)
            c_don = anp.sqrt(1.0 / (rho_don*inv_rc2_don + _eps))

            vars_ext = [_ghost_anp(v, bc_l, bc_r) for v in [a1r1, a2r2, ru, rE, p, c_wood]]
            L_std = [v[0:N+1] for v in vars_ext]
            R_std = [v[1:N+2] for v in vars_ext]

            is_right = (u_face_thinc >= 0)
            new_L = [anp.where(is_right, a1r1_don, L_std[0]),
                     anp.where(is_right, a2r2_don, L_std[1]),
                     anp.where(is_right, ru_don,   L_std[2]),
                     anp.where(is_right, rE_don,   L_std[3]),
                     anp.where(is_right, p_don,    L_std[4]),
                     anp.where(is_right, c_don,    L_std[5])]
            new_R = [anp.where(is_right, R_std[0], a1r1_don),
                     anp.where(is_right, R_std[1], a2r2_don),
                     anp.where(is_right, R_std[2], ru_don),
                     anp.where(is_right, R_std[3], rE_don),
                     anp.where(is_right, R_std[4], p_don),
                     anp.where(is_right, R_std[5], c_don)]

            if flux_type == 'hllc':
                Ff_full = hllc_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                        new_L[4], new_L[5],
                                        new_R[0], new_R[1], new_R[2], new_R[3],
                                        new_R[4], new_R[5])
                Ff = Ff_full[:4]; _u_hllc = Ff_full[4]
            else:
                Ff = rusanov_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                      new_L[4], new_L[5],
                                      new_R[0], new_R[1], new_R[2], new_R[3],
                                      new_R[4], new_R[5])
                _u_hllc = None

            a1_face_adv = a1_thinc

        else:
            # Standard upwind reconstruction (conservative variables)
            vars_ext = [_ghost_anp(v, bc_l, bc_r)
                        for v in [a1r1, a2r2, ru, rE, p, c_wood]]

            if use_bvd and _bvd_cache is not None and _bvd_cache.get('L') is not None:
                L = _bvd_cache['L']
                R = _bvd_cache['R']
            elif use_bvd:
                bvd_pairs = [_bvd_reconstruct_anp(ve, N, beta=bvd_beta)
                             for ve in vars_ext]
                L = [pair[0] for pair in bvd_pairs]
                R = [pair[1] for pair in bvd_pairs]
            else:
                L = [v[0:N+1] for v in vars_ext]
                R = [v[1:N+2] for v in vars_ext]

            if flux_type == 'hllc':
                Ff_full = hllc_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                        R[0], R[1], R[2], R[3], R[4], R[5])
                Ff = Ff_full[:4]
                _u_hllc = Ff_full[4]  # HLLC face velocity for α equation
            elif flux_type == 'slau2':
                Ff = slau2_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                    R[0], R[1], R[2], R[3], R[4], R[5])
                _u_hllc = None
            elif flux_type == 'ausm+':
                Ff = ausm_plus_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                        R[0], R[1], R[2], R[3], R[4], R[5])
                _u_hllc = None
            else:
                Ff = rusanov_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                      R[0], R[1], R[2], R[3], R[4], R[5])
                _u_hllc = None

            u_ext1 = _ghost_anp(u_vel, bc_l, bc_r)
            u_face_std = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])

            if use_bvd and _bvd_cache is not None and _bvd_cache.get('a1_face') is not None:
                # Frozen BVD α₁ face values
                a1_face_adv = _bvd_cache['a1_face']
            elif use_bvd:
                # Live BVD α₁
                a1_ext1 = _ghost_anp(a1, bc_l, bc_r)
                a1L_bvd, a1R_bvd = _bvd_reconstruct_anp(a1_ext1, N, beta=bvd_beta)
                a1_face_adv = anp.where(u_face_std >= 0, a1L_bvd, a1R_bvd)
            else:
                a1_ext1 = _ghost_anp(a1, bc_l, bc_r)
                a1_face_adv = anp.where(u_face_std >= 0,
                                        a1_ext1[0:N+1], a1_ext1[1:N+2])
            u_face_thinc = u_face_std

            # --- MMACM sharpening correction (He & Tan 2024, Eqs. 55-56) ---
            # Applied on top of standard upwind HLLC; kappa=0 in pure phases
            # so G≈0 there (no effect on smooth/pure-phase regions).
            _mmacm_G_alpha = None
            if use_mmacm:
                G_a1r1, G_a2r2, G_ru, G_rE, _mmacm_G_alpha = \
                    _mmacm_correction_flux(
                        a1, a1r1, a2r2, p, T, u_vel,
                        u_face_std, a1_face_adv,
                        ph1, ph2, bc_l, bc_r,
                        mmacm_beta=mmacm_beta,
                        dt=dt, dx=dx,
                        sharpening=mmacm_sharpening)
                Ff = (Ff[0] + G_a1r1, Ff[1] + G_a2r2,
                      Ff[2] + G_ru,   Ff[3] + G_rE)

        # --- α₁ equation (non-conservative) ---
        # ∂α₁/∂t + ∂(α₁u)/∂x - (α₁+D₁)·∂u/∂x = 0
        #
        # Key: use HLLC face velocity û for BOTH α₁ advection flux and
        # div(u) source term (Johnsen & Colonius 2006). This ensures
        # the α equation is consistent with the HLLC conservation fluxes
        # and prevents pressure oscillation at material interfaces.
        #
        # Fallback to centered average if HLLC velocity is not available
        # (e.g., for Rusanov/SLAU2/AUSM+ flux).

        # Face velocity for α equation
        # Use centered average (consistent with existing validated results).
        # HLLC S*-based face velocity is available in _u_hllc but can cause
        # Newton convergence issues; reserved for future investigation.
        u_face = u_face_thinc

        # flux_alpha = α₁_face * u_face  (N+1,)
        flux_alpha = a1_face_adv * u_face
        if use_mmacm and _mmacm_G_alpha is not None:
            flux_alpha = flux_alpha + _mmacm_G_alpha

        # div(α₁u)/dx for cell i
        div_alpha_u = (flux_alpha[1:N+1] - flux_alpha[0:N]) / dx

        # div(u)/dx for cell i — MUST use same u_face as α advection
        div_u = (u_face[1:N+1] - u_face[0:N]) / dx

        # D₁ coefficient
        D1 = _compute_D1_anp(a1r1, a2r2, ru, rE, a1, ph1, ph2, model_type)

        # Unpack Q_n
        a1r1_n, a2r2_n, ru_n, rE_n, a1_n = unpack_5eq(_Q_n, N)

        # --- Residuals ---
        # Conservation equations (4 equations × N cells)
        R0 = (a1r1 - a1r1_n) / dt + (Ff[0][1:N+1] - Ff[0][0:N]) / dx
        R1 = (a2r2 - a2r2_n) / dt + (Ff[1][1:N+1] - Ff[1][0:N]) / dx
        R2 = (ru   - ru_n)   / dt + (Ff[2][1:N+1] - Ff[2][0:N]) / dx
        R3 = (rE   - rE_n)   / dt + (Ff[3][1:N+1] - Ff[3][0:N]) / dx

        # α₁ equation (non-conservative)
        R4 = (a1 - a1_n) / dt + div_alpha_u - (a1 + D1) * div_u

        return anp.concatenate([R0, R1, R2, R3, R4])

    return residual


# ---------------------------------------------------------------------------
# Newton solver (5-equation, GMRES+ILU, correction-based convergence)
# ---------------------------------------------------------------------------

def newton_he2024(Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                  max_newton=20, tol=1e-6, verbose=False,
                  flux_type='hllc', model_type='allaire',
                  use_thinc=False, thinc_beta=50.0,
                  jacobian_method='autograd',
                  use_mmacm=False, mmacm_beta=4.0,
                  mmacm_sharpening='thinc',
                  use_bvd=False, bvd_beta=1.6):
    """Newton solver for the He2024 5-equation system.

    Uses GMRES+ILU with row-column equilibration.
    Convergence: max relative correction of conservative variables.

    Parameters
    ----------
    jacobian_method : str
        One of:
        - 'autograd' (default): dense autograd Jacobian, works for all cases.
          THINC frozen from Q_n.
        - 'fd_sparse': FD sparse Jacobian with 15-color graph coloring.
          THINC updated at each Newton iteration from Q_k.
          Faster for large N when autograd is slow.
        - 'jfnk': Jacobian-Free Newton-Krylov. Matrix-free J*v via FD.
          Live THINC inside residual for J*v products.
          FD sparse of frozen THINC residual used as preconditioner.
          Best for TDU + THINC without w-blending.
    """
    if not _AD:
        raise ImportError("autograd required for newton_he2024")

    from scipy.sparse import csc_matrix, diags as sp_diags
    from scipy.sparse.linalg import spilu, gmres, LinearOperator

    # -----------------------------------------------------------------------
    # THINC cache setup
    # For 'autograd': no cache → residual computes _thinc_face_anp internally
    #   (w-blending, autograd-differentiable, original behaviour).
    # For 'fd_sparse'/'jfnk': cache with _thinc_alpha_he2024 (no w-blending),
    #   updated each Newton iteration from Q_k.
    # -----------------------------------------------------------------------
    # MMACM and THINC are mutually exclusive — MMACM takes priority
    if use_mmacm and use_thinc:
        use_thinc = False

    _thinc_cache = None
    if use_thinc:
        _thinc_cache = {
            'a1_face': _compute_thinc_face_from_Q(Q_n, N, bc_l, bc_r, thinc_beta)
        }

    # Frozen BVD cache: compute L/R states from Q_n once
    _bvd_cache = None
    if use_bvd:
        Q_n_arr = np.array(Q_n, dtype=float)
        a1r1_n, a2r2_n, ru_n, rE_n, a1_n = unpack_5eq(Q_n_arr, N)
        p_n, T_n, u_n, c_n = mixture_eos_anp(a1r1_n, a2r2_n, ru_n, rE_n, ph1, ph2,
                                                alpha1_in=a1_n)
        p_n = np.array(p_n); c_n = np.array(c_n)
        vars_n = [a1r1_n, a2r2_n, ru_n, rE_n, p_n, c_n]
        vars_ext_n = [np.array(_ghost_anp(v, bc_l, bc_r)) for v in vars_n]
        bvd_pairs = [_bvd_reconstruct_anp(ve, N, beta=bvd_beta) for ve in vars_ext_n]
        _bvd_cache = {
            'L': [np.array(pair[0]) for pair in bvd_pairs],
            'R': [np.array(pair[1]) for pair in bvd_pairs],
        }
        # α₁ face values
        a1_ext_n = np.array(_ghost_anp(np.array(a1_n), bc_l, bc_r))
        a1L_n, a1R_n = _bvd_reconstruct_anp(a1_ext_n, N, beta=bvd_beta)
        u_ext_n = np.array(_ghost_anp(np.array(u_n), bc_l, bc_r))
        u_face_n = 0.5 * (u_ext_n[0:N+1] + u_ext_n[1:N+2])
        _bvd_cache['a1_face'] = np.where(u_face_n >= 0, np.array(a1L_n), np.array(a1R_n))

    # Residual functions
    res_func = make_residual_he2024(
        Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
        flux_type=flux_type, model_type=model_type,
        use_thinc=use_thinc, thinc_beta=thinc_beta,
        _thinc_cache=_thinc_cache,
        use_mmacm=use_mmacm, mmacm_beta=mmacm_beta, mmacm_sharpening=mmacm_sharpening,
        use_bvd=use_bvd, bvd_beta=bvd_beta, _bvd_cache=_bvd_cache,
    )
    res_func_live = None
    if jacobian_method in ('fd_sparse', 'jfnk') and use_thinc:
        res_func_live = make_residual_he2024(
            Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
            flux_type=flux_type, model_type=model_type,
            use_thinc=True, thinc_beta=thinc_beta,
            live_thinc=True,
            use_mmacm=use_mmacm, mmacm_beta=mmacm_beta, mmacm_sharpening=mmacm_sharpening,
        use_bvd=use_bvd, bvd_beta=bvd_beta,
        )

    # The residual used for R evaluation and line search
    res_eval = res_func_live if res_func_live is not None else res_func

    # Only build autograd Jacobian when needed
    jac_func = None
    if jacobian_method == 'autograd':
        jac_func = ad_jacobian(res_func)

    Q_k = np.array(Q_n, dtype=float)

    # Reference scales for convergence criterion
    a1r1_n, a2r2_n, ru_n, rE_n, a1_n = unpack_5eq(Q_n, N)
    rho_n = a1r1_n + a2r2_n
    rho_ref  = max(float(np.max(np.abs(rho_n))),  1.0)
    rhou_ref = max(float(np.max(np.abs(ru_n))),   1.0)
    rhoE_ref = max(float(np.max(np.abs(rE_n))),   1.0)

    # Per-cell reference scale for column equilibration
    # Ensures minority phase cells (a1r1~eps) don't get swamped
    _q_ref = np.ones(5 * N)
    _q_ref[0:N]     = np.maximum(np.abs(a1r1_n), 1e-6)    # cell-wise a1r1 floor
    _q_ref[N:2*N]   = np.maximum(np.abs(a2r2_n), 1e-6)    # cell-wise a2r2 floor
    _q_ref[2*N:3*N] = np.maximum(np.abs(rho_n),  1.0)     # rho as ru ref (u~0)
    _q_ref[3*N:4*N] = np.maximum(np.abs(rE_n),   1.0)     # cell-wise rhoE
    _q_ref[4*N:5*N] = 1.0

    converged = False

    for niter in range(max_newton):
        # Update THINC cache from current Q_k for all Jacobian methods
        if use_thinc and _thinc_cache is not None:
            _thinc_cache['a1_face'] = _compute_thinc_face_from_Q(
                Q_k, N, bc_l, bc_r, thinc_beta)

        R = np.array(res_eval(Q_k), dtype=float)
        R_norm = np.linalg.norm(R)

        if verbose:
            print(f"    Newton {niter:2d}: |R| = {R_norm:.3e}")

        # ------------------------------------------------------------------
        # Compute Newton correction dQ based on jacobian_method
        # ------------------------------------------------------------------
        if jacobian_method == 'autograd':
            J = np.array(jac_func(Q_k), dtype=float)

            # Row-column equilibration with block-aware variable scaling
            D_row   = 1.0 / (np.max(np.abs(J), axis=1) + 1e-300)
            Q_scale = np.maximum(np.abs(Q_k), _q_ref)
            J_eq    = np.diag(D_row) @ J @ np.diag(Q_scale)
            b_eq    = -D_row * R

            try:
                J_sp = csc_matrix(J_eq)
                ilu  = spilu(J_sp, fill_factor=10)
                M    = LinearOperator(J_sp.shape, matvec=ilu.solve)
                dQ_eq, info_gmres = gmres(J_sp, b_eq, M=M, atol=1e-12,
                                          maxiter=200)
                if info_gmres != 0:
                    raise RuntimeError("GMRES did not converge")
            except Exception:
                try:
                    dQ_eq = np.linalg.solve(J_eq, b_eq)
                except np.linalg.LinAlgError:
                    return Q_k, {
                        'converged': False,
                        'newton_iters': niter,
                        'final_residual': float(R_norm),
                    }

            dQ = Q_scale * dQ_eq

        elif jacobian_method == 'fd_sparse':
            # Sparse FD Jacobian with graph coloring
            # Use res_eval (live THINC) so Jacobian captures full nonlinearity
            J_sp_fd = _fd_sparse_jacobian(res_eval, Q_k, N)

            # Sparse row-column equilibration with block-aware scaling
            abs_J = abs(J_sp_fd)
            row_max = np.array(abs_J.max(axis=1).todense()).ravel()
            D_row = 1.0 / (row_max + 1e-300)
            Q_scale = np.maximum(np.abs(Q_k), _q_ref)
            J_eq_sp = sp_diags(D_row) @ J_sp_fd @ sp_diags(Q_scale)
            b_eq = -D_row * R

            try:
                ilu = spilu(J_eq_sp.tocsc(), fill_factor=10)
                M   = LinearOperator(J_eq_sp.shape, matvec=ilu.solve)
                dQ_eq, info_gmres = gmres(J_eq_sp, b_eq, M=M, atol=1e-12,
                                          maxiter=200)
                if info_gmres != 0:
                    raise RuntimeError("GMRES did not converge")
            except Exception:
                try:
                    dQ_eq = np.linalg.solve(J_eq_sp.toarray(), b_eq)
                except np.linalg.LinAlgError:
                    return Q_k, {
                        'converged': False,
                        'newton_iters': niter,
                        'final_residual': float(R_norm),
                    }

            dQ = Q_scale * dQ_eq

        elif jacobian_method == 'jfnk':
            # JFNK: live THINC residual for J*v, frozen FD as preconditioner
            # res_func_live already built above (= res_eval)

            # Preconditioner: sparse FD of frozen-THINC residual
            prec_J = _fd_sparse_jacobian(res_func, Q_k, N)

            # Row-column equilibration for preconditioner
            abs_J = abs(prec_J)
            row_max = np.array(abs_J.max(axis=1).todense()).ravel()
            D_row = 1.0 / (row_max + 1e-300)
            Q_scale = np.maximum(np.abs(Q_k), 1.0)
            prec_J_eq = sp_diags(D_row) @ prec_J @ sp_diags(Q_scale)
            b_eq = -D_row * R

            # Build equilibrated live residual for JFNK
            def res_func_live_eq(Q_eq):
                Q_phys = Q_scale * Q_eq
                R_phys = np.array(res_eval(Q_phys), dtype=float)
                return D_row * R_phys

            Q_k_eq = Q_k / Q_scale
            R_eq = D_row * R

            dQ_eq, info_gmres = _jfnk_solve(
                res_func_live_eq, Q_k_eq, R_eq,
                prec_J_sp=prec_J_eq.tocsc())
            dQ = Q_scale * dQ_eq

        else:
            raise ValueError(
                f"Unknown jacobian_method: {jacobian_method!r}. "
                "Choose 'autograd', 'fd_sparse', or 'jfnk'.")

        # ------------------------------------------------------------------
        # Backtracking line search (common to all methods)
        # ------------------------------------------------------------------
        omega = 1.0
        for _ls in range(12):
            Q_trial = Q_k + omega * dQ
            # Physical bounds
            Q_trial[0:N]     = np.maximum(Q_trial[0:N],   1e-10)
            Q_trial[N:2*N]   = np.maximum(Q_trial[N:2*N], 1e-10)
            Q_trial[4*N:5*N] = np.clip(Q_trial[4*N:5*N], 0.0, 1.0)
            try:
                R_trial = np.array(res_eval(Q_trial), dtype=float)
                if np.linalg.norm(R_trial) < R_norm:
                    break
            except Exception:
                pass
            omega *= 0.5

        Q_k = Q_k + omega * dQ
        Q_k[0:N]     = np.maximum(Q_k[0:N],   1e-10)
        Q_k[N:2*N]   = np.maximum(Q_k[N:2*N], 1e-10)
        Q_k[4*N:5*N] = np.clip(Q_k[4*N:5*N], 0.0, 1.0)

        # ------------------------------------------------------------------
        # Convergence check (correction-based, minimum 3 iterations)
        # ------------------------------------------------------------------
        dQ_actual = omega * dQ
        corr_a1r1 = float(np.max(np.abs(dQ_actual[0:N])))     / rho_ref
        corr_a2r2 = float(np.max(np.abs(dQ_actual[N:2*N])))   / rho_ref
        corr_ru   = float(np.max(np.abs(dQ_actual[2*N:3*N]))) / rhou_ref
        corr_rE   = float(np.max(np.abs(dQ_actual[3*N:4*N]))) / rhoE_ref
        corr_a1   = float(np.max(np.abs(dQ_actual[4*N:5*N])))
        corr_max  = max(corr_a1r1, corr_a2r2, corr_ru, corr_rE, corr_a1)

        if verbose:
            print(f"      omega={omega:.3f} corr_max={corr_max:.2e}")

        if niter >= 2 and corr_max < tol:
            converged = True
            break

    R_final = float(np.linalg.norm(np.array(res_eval(Q_k), dtype=float)))
    return Q_k, {
        'converged': converged,
        'newton_iters': niter + 1,
        'final_residual': R_final,
    }


# ---------------------------------------------------------------------------
# Step function
# ---------------------------------------------------------------------------

def step_he2024(N, dx, Q_prev, ph1, ph2, bc_l, bc_r, cfg):
    """Advance solution by one timestep using the He2024 5-equation solver.

    Parameters
    ----------
    N : int
        Number of cells.
    dx : float
        Uniform cell width.
    Q_prev : ndarray (5N,)
        Previous-step conservative state packed as
        [α₁ρ₁(N) | α₂ρ₂(N) | ρu(N) | ρE(N) | α₁(N)].
    ph1, ph2 : dict
        EOS parameters for each phase (NASG/Ideal).
    bc_l, bc_r : str
        Boundary conditions ('periodic' or 'transmissive'/'outflow').
    cfg : dict
        Solver configuration keys:
          CFL             : float, default 0.5 (acoustic CFL for dt)
          dt_fixed        : float, optional (overrides CFL-based dt)
          flux_type       : 'hllc' (default) or 'rusanov'
          model_type      : 'allaire' (default, D_k=0) or 'kapila'
          use_thinc       : bool, default False
          thinc_beta      : float, default 50.0
          max_newton      : int, default 20
          newton_tol      : float, default 1e-6
          verbose         : bool, default False
          jacobian_method : 'autograd' (default) | 'fd_sparse' | 'jfnk'

    Returns
    -------
    Q_new : ndarray (5N,)
    dt    : float
    info  : dict  (converged, newton_iters, final_residual)
    """
    CFL = float(cfg.get('CFL', 0.5))

    a1r1_k, a2r2_k, ru_k, rE_k, a1_k = unpack_5eq(Q_prev, N)
    p_k, T_k, u_k, c_k = mixture_eos_anp(a1r1_k, a2r2_k, ru_k, rE_k, ph1, ph2,
                                           alpha1_in=a1_k)

    u_k = np.array(u_k, dtype=float)
    c_k = np.array(c_k, dtype=float)

    max_speed = float(np.max(np.abs(u_k) + c_k))
    if max_speed < 1e-300:
        max_speed = 1e-300

    if 'dt_fixed' in cfg:
        dt = float(cfg['dt_fixed'])
    else:
        dt = CFL * dx / max_speed

    flux_type       = cfg.get('flux_type',       'hllc')
    model_type      = cfg.get('model_type',      'allaire')
    use_thinc       = bool(cfg.get('use_thinc',  False))
    thinc_beta      = float(cfg.get('thinc_beta', 50.0))
    jacobian_method = cfg.get('jacobian_method', 'autograd')
    use_mmacm       = bool(cfg.get('use_mmacm',  False))
    mmacm_beta      = float(cfg.get('mmacm_beta', 4.0))
    mmacm_sharpening = cfg.get('mmacm_sharpening', 'mmacm_ex')
    use_bvd         = bool(cfg.get('use_bvd', False))
    bvd_beta        = float(cfg.get('bvd_beta', 1.6))

    Q_new, info = newton_he2024(
        Q_prev, N, dx, dt, ph1, ph2, bc_l, bc_r,
        max_newton      = int(cfg.get('max_newton',  20)),
        tol             = float(cfg.get('newton_tol', 1e-6)),
        verbose         = bool(cfg.get('verbose',    False)),
        flux_type       = flux_type,
        model_type      = model_type,
        use_thinc       = use_thinc,
        thinc_beta      = thinc_beta,
        jacobian_method = jacobian_method,
        use_mmacm       = use_mmacm,
        mmacm_beta      = mmacm_beta,
        mmacm_sharpening = mmacm_sharpening,
        use_bvd         = use_bvd,
        bvd_beta        = bvd_beta,
    )

    return Q_new, dt, info
