# solver/denner_1d/assembly_5eq.py
# Ref: breezy-wishing-wall.md — 5-equation conservative Newton solver
#
# 5-equation model: Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}
#
# Governing equations (residual form, BDF1):
#   R₁ = (α₁ρ₁)^{n+1}/dt - (α₁ρ₁)^n/dt + ∂(α₁ρ₁·u)/∂x = 0   [phase 1 mass]
#   R₂ = (α₂ρ₂)^{n+1}/dt - (α₂ρ₂)^n/dt + ∂(α₂ρ₂·u)/∂x = 0   [phase 2 mass]
#   R₃ = (ρu)^{n+1}/dt   - (ρu)^n/dt   + ∂(ρu²+p)/∂x = 0       [momentum]
#   R₄ = (ρE)^{n+1}/dt   - (ρE)^n/dt   + ∂(ρuH)/∂x = 0         [energy]
#   R₅ = α₁^{n+1}/dt     - α₁^n/dt     + u·∂α₁/∂x - K₁·∂u/∂x = 0 [alpha advection]
#
# Temporal Jacobian ∂Rₖ/∂Qₖ = 1/dt (diagonal) for all 5 equations.
# This eliminates the α/ζ ill-conditioning of primitive-variable formulations.

import numpy as np
import scipy.sparse as sp

from .eos.eos_class import create_eos
from .eos.invert import invert_eos
from .boundary import apply_ghost, apply_ghost_velocity


def _ci5(block, i, N):
    """Column index for block-ordered 5N system."""
    return block * N + i


def split5(Q_5N, N):
    """Split 5N vector into 5 blocks of length N."""
    a1r1 = Q_5N[0*N:1*N]
    a2r2 = Q_5N[1*N:2*N]
    ru   = Q_5N[2*N:3*N]
    rE   = Q_5N[3*N:4*N]
    a1   = Q_5N[4*N:5*N]
    return a1r1, a2r2, ru, rE, a1


def compute_K(a1, p, T, eos1, eos2):
    """Compute Wood's sound speed K coefficient.

    Wood's mixture compressibility:
        1/(ρ·c_Wood²) = α₁/(ρ₁·c₁²) + α₂/(ρ₂·c₂²)

    K coefficient (sum-to-1 condition satisfied):
        K₁ = α₁·ρ·c_Wood² / (ρ₁·c₁²)

    Parameters
    ----------
    a1   : ndarray (N,)  volume fraction of phase 1
    p    : ndarray (N,)  pressure
    T    : ndarray (N,)  temperature
    eos1, eos2 : EOS objects

    Returns
    -------
    K1 : ndarray (N,)
    """
    a2 = np.maximum(1.0 - a1, 0.0)

    rho1 = eos1.rho(p, T)
    c1   = eos1.c(p, T)
    rho2 = eos2.rho(p, T)
    c2   = eos2.c(p, T)

    rho = a1 * rho1 + a2 * rho2

    # Wood's compressibility: 1/(ρ·c_Wood²) = α₁/(ρ₁·c₁²) + α₂/(ρ₂·c₂²)
    rho1c1sq = rho1 * c1**2 + 1e-300
    rho2c2sq = rho2 * c2**2 + 1e-300
    inv_rho_c2 = a1 / rho1c1sq + a2 / rho2c2sq

    # ρ·c_Wood²
    rho_cw2 = 1.0 / (inv_rho_c2 + 1e-300)

    K1 = a1 * rho_cw2 / rho1c1sq
    return K1


def invert_5eq(Q_5N, N, ph1, ph2, p_guess=None, T_guess=None):
    """Recover primitive variables (p, T, u) from 5-equation conservative state Q.

    Given Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}, recover (p, T, u) via:
      ρ  = α₁ρ₁ + α₂ρ₂
      u  = ρu / ρ
      ρe = ρE - ½·ρu²   (internal energy density)
    Then call invert_eos(psi=α₁, rho_target=ρ, E_target=ρe, ...) for (p, T).

    Parameters
    ----------
    Q_5N     : ndarray (5N,)   conservative state
    N        : int              number of cells
    ph1, ph2 : dict             EOS parameter dicts
    p_guess  : ndarray (N,) or None
    T_guess  : ndarray (N,) or None

    Returns
    -------
    p : ndarray (N,)
    T : ndarray (N,)
    u : ndarray (N,)
    """
    a1r1, a2r2, ru, rE, a1 = split5(Q_5N, N)

    a1 = np.clip(a1, 0.0, 1.0)
    a2 = np.maximum(1.0 - a1, 0.0)

    rho = np.maximum(a1r1 + a2r2, 1e-300)
    u   = ru / rho
    # ρe = ρE - ½ρu²  (internal energy density [J/m³])
    rho_e = rE - 0.5 * ru * u

    # Provide default guesses if not supplied
    if p_guess is None:
        p_guess = np.full(N, 1e5)
    if T_guess is None:
        T_guess = np.full(N, 300.0)

    # Guard against unphysical internal energy
    if np.any(~np.isfinite(rho_e)) or np.any(rho_e < 0.0):
        # Fallback: reset guess to safe default values for problematic cells
        bad = ~np.isfinite(rho_e) | (rho_e < 0.0)
        p_guess = p_guess.copy()
        T_guess = T_guess.copy()
        p_guess[bad] = 1e5
        T_guess[bad] = 300.0
        rho_e = np.maximum(rho_e, 1.0)

    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        bad = ~np.isfinite(rho) | (rho <= 0.0)
        p_guess = p_guess.copy()
        T_guess = T_guess.copy()
        p_guess[bad] = 1e5
        T_guess[bad] = 300.0
        rho = np.maximum(rho, 1e-300)

    # EOS inversion: find (p, T) such that
    #   α₁·ρ₁(p,T) + α₂·ρ₂(p,T) = ρ  and
    #   α₁·ρ₁·e₁(p,T) + α₂·ρ₂·e₂(p,T) = ρe
    p, T = invert_eos(a1, rho, rho_e, p_guess, T_guess, ph1, ph2)

    # If inversion produced non-finite results, retry with safe default guess
    bad_out = ~np.isfinite(p) | ~np.isfinite(T) | (p <= 0.0) | (T <= 0.0)
    if np.any(bad_out):
        p_fallback = np.where(bad_out, 1e5, p)
        T_fallback = np.where(bad_out, 300.0, T)
        p2, T2 = invert_eos(a1, rho, rho_e, p_fallback, T_fallback, ph1, ph2)
        p = np.where(bad_out, p2, p)
        T = np.where(bad_out, T2, T)

    return p, T, u


def residual_5eq(Q_k, Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r, theta_k, K_k, p_k, T_k, u_k):
    """Compute 5N residual vector for the 5-equation model (BDF1).

    Face fluxes use standard first-order upwind with MWI face velocity theta_k.

    Parameters
    ----------
    Q_k     : ndarray (5N,)   current Newton iterate (conservative)
    Q_n     : ndarray (5N,)   old-time conservative state
    N       : int
    dx      : float
    dt      : float
    ph1,ph2 : dict
    bc_l,bc_r : str
    theta_k : ndarray (N+1,)  MWI face velocity
    K_k     : ndarray (N,)    Wood K coefficient (for alpha equation)
    p_k,T_k,u_k : ndarray (N,)  primitive variables at iterate

    Returns
    -------
    R : ndarray (5N,)
    """
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    a1r1_k, a2r2_k, ru_k, rE_k, a1_k = split5(Q_k, N)
    a1r1_n, a2r2_n, ru_n, rE_n, a1_n = split5(Q_n, N)

    a1_k = np.clip(a1_k, 0.0, 1.0)
    a2_k = np.maximum(1.0 - a1_k, 0.0)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def face_cells(f):
        iL = f - 1
        iR = f
        if iL < 0:
            iL = (N - 1) if is_per_l else 0
        if iR >= N:
            iR = 0 if is_per_r else N - 1
        return iL, iR

    # Ghost cell extensions for upwind access
    a1_ext  = apply_ghost(a1_k, bc_l, bc_r, 1)  # n_ghost=1 sufficient for 1st-order upwind
    p_ext   = apply_ghost(p_k,  bc_l, bc_r, 1)
    T_ext   = apply_ghost(T_k,  bc_l, bc_r, 1)
    ru_ext  = apply_ghost(ru_k, bc_l, bc_r, 1)
    u_ext   = apply_ghost(u_k,  bc_l, bc_r, 1)
    rE_ext  = apply_ghost(rE_k, bc_l, bc_r, 1)

    R = np.zeros(5 * N)

    for i in range(N):
        # Face indices
        f_R = i + 1
        f_L = i

        # Neighbor cells (with periodic/transmissive wrap)
        iL, _ = face_cells(f_L)
        _, iR  = face_cells(f_R)

        # Extended array indices (n_ghost=1)
        ng = 1
        iL_ext = ng + i - 1  # left cell in extended array
        i_ext  = ng + i       # cell i
        iR_ext = ng + i + 1   # right cell in extended array

        tR = theta_k[f_R]
        tL = theta_k[f_L]

        # --- Upwind selection ---
        # Right face: upwind=i if tR>=0, else upwind=iR
        # Left face:  upwind=iL if tL>=0, else upwind=i

        # Phase 1: upwind α₁ * phase density at (p,T) of upwind cell
        a1_upR  = a1_ext[i_ext]  if tR >= 0 else a1_ext[iR_ext]
        a1_upL  = a1_ext[iL_ext] if tL >= 0 else a1_ext[i_ext]
        p_upR   = p_ext[i_ext]   if tR >= 0 else p_ext[iR_ext]
        T_upR   = T_ext[i_ext]   if tR >= 0 else T_ext[iR_ext]
        p_upL   = p_ext[iL_ext]  if tL >= 0 else p_ext[i_ext]
        T_upL   = T_ext[iL_ext]  if tL >= 0 else T_ext[i_ext]

        r1_upR  = eos1.rho(p_upR, T_upR)
        r2_upR  = eos2.rho(p_upR, T_upR)
        r1_upL  = eos1.rho(p_upL, T_upL)
        r2_upL  = eos2.rho(p_upL, T_upL)

        # Bug 1 fix: use conservative variables directly for mass flux
        # (avoids EOS reconstruction inconsistency at faces)
        a1r1_fR = a1r1_k[i]  if tR >= 0 else a1r1_k[iR]
        a2r2_fR = a2r2_k[i]  if tR >= 0 else a2r2_k[iR]
        a1r1_fL = a1r1_k[iL] if tL >= 0 else a1r1_k[i]
        a2r2_fL = a2r2_k[iL] if tL >= 0 else a2r2_k[i]

        # R1: phase 1 mass conservation
        R[0*N + i] = (a1r1_k[i] - a1r1_n[i]) / dt + (a1r1_fR * tR - a1r1_fL * tL) / dx

        # R2: phase 2 mass conservation
        R[1*N + i] = (a2r2_k[i] - a2r2_n[i]) / dt + (a2r2_fR * tR - a2r2_fL * tL) / dx

        # R3: momentum — upwind ρu·u + arithmetic mean pressure
        pR_face = 0.5 * (p_k[i] + p_k[iR])
        pL_face = 0.5 * (p_k[iL] + p_k[i])

        ru_upR = ru_ext[i_ext]   if tR >= 0 else ru_ext[iR_ext]
        ru_upL = ru_ext[iL_ext]  if tL >= 0 else ru_ext[i_ext]
        u_upR  = u_ext[i_ext]    if tR >= 0 else u_ext[iR_ext]
        u_upL  = u_ext[iL_ext]   if tL >= 0 else u_ext[i_ext]

        # Bug 3 fix: use theta (MWI face velocity) instead of upwind u
        # for momentum flux to maintain theta consistency
        flux_mom_R = ru_upR * tR + pR_face
        flux_mom_L = ru_upL * tL + pL_face

        R[2*N + i] = (ru_k[i] - ru_n[i]) / dt + (flux_mom_R - flux_mom_L) / dx

        # R4: energy — total enthalpy flux H = (ρE + p)/ρ * ρ = ρE + p per unit vol
        # Total specific enthalpy H_spec = (E + p/ρ) = e + u²/2 + p/ρ
        # Energy flux F_E = ρu·H_spec = u·(ρE + p)

        # Use upwind cell for (ρE + p)·u
        rEp_upR = (rE_ext[i_ext]  + p_ext[i_ext])  if tR >= 0 else (rE_ext[iR_ext] + p_ext[iR_ext])
        rEp_upL = (rE_ext[iL_ext] + p_ext[iL_ext]) if tL >= 0 else (rE_ext[i_ext]  + p_ext[i_ext])

        # Bug 4 fix: use theta (MWI face velocity) instead of upwind u
        # for energy flux to maintain theta consistency
        flux_E_R = rEp_upR * tR
        flux_E_L = rEp_upL * tL

        R[3*N + i] = (rE_k[i] - rE_n[i]) / dt + (flux_E_R - flux_E_L) / dx

        # R5: alpha advection (non-conservative) with K compression term
        # ∂α₁/∂t + u·∂α₁/∂x = K₁·∂u/∂x
        # theta-consistent discretisation:
        #   ∂α₁/∂t + (θ_R·α₁_R - θ_L·α₁_L)/dx - α₁_i·(θ_R - θ_L)/dx = K₁·(θ_R - θ_L)/dx
        # (Ref: plan_report — FAIL 1: use face velocity theta instead of cell-centre u)
        K1_i = K_k[i]

        # First-order upwind α₁ at faces
        a1_advR = a1_k[i]  if tR >= 0 else a1_k[iR]
        a1_advL = a1_k[iL] if tL >= 0 else a1_k[i]

        # Divergence of theta at cell i (used for both advection and K term)
        div_theta = (tR - tL) / dx

        # Cell-centre α₁
        a1_i = a1_k[i]

        R[4*N + i] = ((a1_k[i] - a1_n[i]) / dt
                      + (tR * a1_advR - tL * a1_advL) / dx
                      - a1_i * div_theta
                      - K1_i * div_theta)

    return R


def assemble_jacobian_5eq(Q_k, p_k, T_k, u_k, K_k, theta_k, N, dx, dt, ph1, ph2, bc_l, bc_r):
    """Assemble 5N×5N sparse Jacobian for 5-equation conservative model.

    Jacobian strategy:
    - Temporal terms: exact (1/dt on diagonal for all 5 equations)
    - Spatial/advective terms: 1st-order upwind, phase mass and alpha
    - Pressure gradient (momentum): frozen face pressure (Picard for pressure)
    - Energy flux: frozen (Picard)
    - Alpha K term: partial — K frozen, ∂u/∂(ρu) = 1/ρ included

    Block ordering: [α₁ρ₁ | α₂ρ₂ | ρu | ρE | α₁]  (N cells each)

    Parameters
    ----------
    Q_k        : ndarray (5N,)
    p_k,T_k,u_k: ndarray (N,)   primitive variables at iterate
    K_k        : ndarray (N,)   Wood K coefficient
    theta_k    : ndarray (N+1,) MWI face velocity
    N,dx,dt    : grid/time params
    ph1,ph2    : EOS dicts
    bc_l,bc_r  : str

    Returns
    -------
    A_csr : scipy csr_matrix (5N, 5N)
    """
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    a1r1_k, a2r2_k, ru_k, rE_k, a1_k = split5(Q_k, N)
    a1_k = np.clip(a1_k, 0.0, 1.0)

    n5 = 5 * N
    A = sp.lil_matrix((n5, n5), dtype=float)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def face_cells(f):
        iL = f - 1
        iR = f
        if iL < 0:
            iL = (N - 1) if is_per_l else 0
        if iR >= N:
            iR = 0 if is_per_r else N - 1
        return iL, iR

    # ----------------------------------------------------------------
    # Pressure EOS Jacobian: dp_dQ[j, i] = ∂p_i / ∂Q_{j*N+i}
    # Computed by numerical finite difference (1 perturbation per DOF).
    # Only the dominant term ∂p/∂(ρE) (block j=3) is computed to limit
    # EOS inversion cost; other blocks are set to zero (frozen).
    # (Ref: plan_report — FAIL 3: add ∂p/∂Q to momentum and energy Jacobian)
    # ----------------------------------------------------------------
    dp_dQ = np.zeros((5, N))
    eps_scale = 1e-7
    for i in range(N):
        # Perturb only ρE (block 3): dominant contribution via ρe = ρE - ½ρu²
        idx = 3 * N + i
        dQ_eps = max(abs(Q_k[idx]) * eps_scale, 1e-10)
        Q_pert = Q_k.copy()
        Q_pert[idx] += dQ_eps
        try:
            p_pert, _, _ = invert_5eq(Q_pert, N, ph1, ph2, p_k.copy(), T_k.copy())
            dp_dQ[3, i] = (p_pert[i] - p_k[i]) / dQ_eps
        except Exception:
            dp_dQ[3, i] = 0.0
        # ρu perturbation (block 2): via kinetic energy
        idx2 = 2 * N + i
        dQ_eps2 = max(abs(Q_k[idx2]) * eps_scale, 1e-10)
        Q_pert2 = Q_k.copy()
        Q_pert2[idx2] += dQ_eps2
        try:
            p_pert2, _, _ = invert_5eq(Q_pert2, N, ph1, ph2, p_k.copy(), T_k.copy())
            dp_dQ[2, i] = (p_pert2[i] - p_k[i]) / dQ_eps2
        except Exception:
            dp_dQ[2, i] = 0.0

    for i in range(N):
        f_R = i + 1
        f_L = i
        iL, _ = face_cells(f_L)
        _, iR  = face_cells(f_R)

        tR = theta_k[f_R]
        tL = theta_k[f_L]

        rho_i = a1r1_k[i] + a2r2_k[i]

        # ----------------------------------------------------------------
        # Temporal terms: ∂Rₖ/∂Qₖ_ii = 1/dt for all 5 blocks
        # ----------------------------------------------------------------
        for blk in range(5):
            A[_ci5(blk, i, N), _ci5(blk, i, N)] += 1.0 / dt

        # ----------------------------------------------------------------
        # Phase 1 mass flux: ∂(α₁ρ₁·θ)/∂Q
        # Upwind α₁·ρ₁: α₁ comes from cell (i or iR for tR, etc.)
        # ρ₁ evaluated at upwind (p, T)
        # ----------------------------------------------------------------
        # Right face contribution to cell i's R1
        r1_upR = eos1.rho(p_k[i], T_k[i]) if tR >= 0 else eos1.rho(p_k[iR], T_k[iR])
        r2_upR = eos2.rho(p_k[i], T_k[i]) if tR >= 0 else eos2.rho(p_k[iR], T_k[iR])
        r1_upL = eos1.rho(p_k[iL], T_k[iL]) if tL >= 0 else eos1.rho(p_k[i], T_k[i])
        r2_upL = eos2.rho(p_k[iL], T_k[iL]) if tL >= 0 else eos2.rho(p_k[i], T_k[i])

        # Bug 2 fix: R1 flux = a1r1_upwind * tR → ∂R1_i/∂(a1r1_upwind) = tR/dx
        # column block must be 0 (a1r1), not 4 (alpha1)
        if tR >= 0:
            # upwind cell = i → a1r1_fR = a1r1_k[i]
            # ∂R1_i/∂(a1r1_i) from right face: +tR / dx
            A[_ci5(0, i, N), _ci5(0, i, N)] += tR / dx
        else:
            # upwind cell = iR → a1r1_fR = a1r1_k[iR]
            # ∂R1_i/∂(a1r1_iR) from right face: +tR / dx
            A[_ci5(0, i, N), _ci5(0, iR, N)] += tR / dx

        if tL >= 0:
            # upwind cell = iL → a1r1_fL = a1r1_k[iL]
            # ∂R1_i/∂(a1r1_iL) from left face: -tL / dx
            A[_ci5(0, i, N), _ci5(0, iL, N)] -= tL / dx
        else:
            # upwind cell = i → a1r1_fL = a1r1_k[i]
            # ∂R1_i/∂(a1r1_i) from left face: -tL / dx
            A[_ci5(0, i, N), _ci5(0, i, N)] -= tL / dx

        # ----------------------------------------------------------------
        # Phase 2 mass flux: symmetric to phase 1
        # Bug 2 fix: column block must be 1 (a2r2), not 4 (alpha1)
        # ----------------------------------------------------------------
        if tR >= 0:
            # upwind cell = i → a2r2_fR = a2r2_k[i]
            # ∂R2_i/∂(a2r2_i) from right face: +tR / dx
            A[_ci5(1, i, N), _ci5(1, i, N)] += tR / dx
        else:
            # upwind cell = iR → a2r2_fR = a2r2_k[iR]
            # ∂R2_i/∂(a2r2_iR) from right face: +tR / dx
            A[_ci5(1, i, N), _ci5(1, iR, N)] += tR / dx

        if tL >= 0:
            # upwind cell = iL → a2r2_fL = a2r2_k[iL]
            # ∂R2_i/∂(a2r2_iL) from left face: -tL / dx
            A[_ci5(1, i, N), _ci5(1, iL, N)] -= tL / dx
        else:
            # upwind cell = i → a2r2_fL = a2r2_k[i]
            # ∂R2_i/∂(a2r2_i) from left face: -tL / dx
            A[_ci5(1, i, N), _ci5(1, i, N)] -= tL / dx

        # ----------------------------------------------------------------
        # Momentum: ∂R3/∂(ρu) — upwind ρu·θ advection
        # Bug 3 fix: flux_mom = ru_upwind * theta → ∂/∂ru_upwind = tR/dx (not u*tR/dx)
        # Pressure gradient uses arithmetic mean (frozen p): no ∂p/∂Q added here
        # ----------------------------------------------------------------
        # Right face (momentum advection):
        if tR >= 0:
            # flux_mom_R = ru_i * tR → ∂/∂ru_i = tR
            A[_ci5(2, i, N), _ci5(2, i, N)] += tR / dx
        else:
            # flux_mom_R = ru_iR * tR → ∂/∂ru_iR = tR
            A[_ci5(2, i, N), _ci5(2, iR, N)] += tR / dx

        # Left face:
        if tL >= 0:
            # flux_mom_L = ru_iL * tL → ∂/∂ru_iL = tL
            A[_ci5(2, i, N), _ci5(2, iL, N)] -= tL / dx
        else:
            # flux_mom_L = ru_i * tL → ∂/∂ru_i = tL
            A[_ci5(2, i, N), _ci5(2, i, N)] -= tL / dx

        # Pressure gradient: ∂(p_face_R - p_face_L)/∂Q
        # p_face_R = 0.5*(p_i + p_iR): ∂p_face_R/∂Q_j_i  = 0.5*dp_dQ[j,i]
        #                               ∂p_face_R/∂Q_j_iR = 0.5*dp_dQ[j,iR]
        # p_face_L = 0.5*(p_iL + p_i): ∂p_face_L/∂Q_j_iL = 0.5*dp_dQ[j,iL]
        #                               ∂p_face_L/∂Q_j_i  = 0.5*dp_dQ[j,i]
        # Net: ∂R3/∂Q_j_i  += (+0.5 - 0.5)/dx = 0 (cancel)
        #      ∂R3/∂Q_j_iR += +0.5/dx
        #      ∂R3/∂Q_j_iL += -0.5/dx
        # (Ref: plan_report — FAIL 3: ∂p/∂Q in momentum pressure gradient)
        for j in range(5):
            if dp_dQ[j, iR] != 0.0:
                A[_ci5(2, i, N), j*N + iR] += 0.5 * dp_dQ[j, iR] / dx
            if dp_dQ[j, iL] != 0.0:
                A[_ci5(2, i, N), j*N + iL] -= 0.5 * dp_dQ[j, iL] / dx

        # ----------------------------------------------------------------
        # Energy: ∂R4/∂(ρE) — upwind energy flux
        # Bug 4 fix: flux_E = (ρE + p)_upwind * theta → frozen p, ∂(rEp)/∂(ρE) = 1
        # coefficient = tR/dx (not u*tR/dx)
        # ----------------------------------------------------------------
        if tR >= 0:
            A[_ci5(3, i, N), _ci5(3, i, N)] += tR / dx
        else:
            A[_ci5(3, i, N), _ci5(3, iR, N)] += tR / dx

        if tL >= 0:
            A[_ci5(3, i, N), _ci5(3, iL, N)] -= tL / dx
        else:
            A[_ci5(3, i, N), _ci5(3, i, N)] -= tL / dx

        # Energy flux: ∂((ρE+p)·θ)/∂Q — from p(Q) dependence
        # F_E = (rE + p)_upwind * theta
        # ∂F_E_R/∂Q_j_upwind = dp_dQ[j, upwind_R] * tR
        # upwind_R = i if tR>=0, else iR
        # (Ref: plan_report — FAIL 3: ∂p/∂Q in energy enthalpy flux)
        if tR >= 0:
            up_R = i
        else:
            up_R = iR
        if tL >= 0:
            up_L = iL
        else:
            up_L = i
        for j in range(5):
            if dp_dQ[j, up_R] != 0.0:
                A[_ci5(3, i, N), j*N + up_R] += dp_dQ[j, up_R] * tR / dx
            if dp_dQ[j, up_L] != 0.0:
                A[_ci5(3, i, N), j*N + up_L] -= dp_dQ[j, up_L] * tL / dx

        # ----------------------------------------------------------------
        # Alpha equation: ∂R5/∂α₁
        # R5_i = (a1_i - a1_n_i)/dt + (tR·a1_advR - tL·a1_advL)/dx
        #        - a1_i·(tR-tL)/dx - K1_i·(tR-tL)/dx
        #
        # theta is frozen (Picard for face velocity).
        # div_theta = (tR - tL)/dx (frozen scalar, no Q-derivative).
        #
        # ∂R5/∂α₁:
        #   From (tR·a1_advR)/dx:
        #     if tR >= 0: a1_advR = a1_i   → ∂/∂a1_i = +tR/dx
        #     else:       a1_advR = a1_iR  → ∂/∂a1_iR = +tR/dx
        #   From (-tL·a1_advL)/dx:
        #     if tL >= 0: a1_advL = a1_iL  → ∂/∂a1_iL = -tL/dx
        #     else:       a1_advL = a1_i   → ∂/∂a1_i  = -tL/dx
        #   From -a1_i·div_theta:
        #     ∂/∂a1_i = -div_theta (always diagonal)
        # (Ref: plan_report — FAIL 1 Jacobian: theta-consistent alpha Jacobian)
        # ----------------------------------------------------------------
        K1_i = float(K_k[i])
        div_theta_jac = (tR - tL) / dx

        # Right face upwind alpha
        if tR >= 0:
            A[_ci5(4, i, N), _ci5(4, i, N)] += tR / dx
        else:
            A[_ci5(4, i, N), _ci5(4, iR, N)] += tR / dx

        # Left face upwind alpha
        if tL >= 0:
            A[_ci5(4, i, N), _ci5(4, iL, N)] -= tL / dx
        else:
            A[_ci5(4, i, N), _ci5(4, i, N)] -= tL / dx

        # Subtract a1_i * div_theta (a1_i is cell i, frozen → diagonal)
        A[_ci5(4, i, N), _ci5(4, i, N)] -= div_theta_jac

        # K term: K1_i * div_theta — theta frozen, K1 frozen → no ∂/∂Q contribution
        # (K term Jacobian removed per plan_report FAIL 2)

    return A.tocsr()


# ---------------------------------------------------------------------------
# Primitive-variable Newton (W = {p, u, T, α₁})
# ---------------------------------------------------------------------------

def _ci4(block, i, N):
    """Column index for block-ordered 4N system: W = [p | u | T | α₁]."""
    return block * N + i


def residual_5eq_prim(p_k, u_k, T_k, a1_k,
                      p_n, u_n, T_n, a1_n,
                      N, dx, dt, ph1, ph2, bc_l, bc_r,
                      theta_k, K_k):
    """4N residual for primitive-variable 5-eq Newton (BDF1).

    Unknowns: W = {p[i], u[i], T[i], α₁[i]}  i=0..N-1
    Block ordering: R = [R_mass1(0..N-1) | R_mom(N..2N-1)
                         | R_energy(2N..3N-1) | R_alpha(3N..4N-1)]

    EOS is used *forward* — no inversion needed.

    Parameters
    ----------
    p_k, u_k, T_k, a1_k : ndarray (N,)  current Newton iterate
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    N, dx, dt            : int, float, float
    ph1, ph2             : EOS parameter dicts
    bc_l, bc_r           : str
    theta_k              : ndarray (N+1,)  MWI face velocity
    K_k                  : ndarray (N,)    Wood K coefficient

    Returns
    -------
    R : ndarray (4N,)
    """
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    a1_k  = np.clip(a1_k, 0.0, 1.0)
    a1_n  = np.clip(a1_n, 0.0, 1.0)
    a2_k  = 1.0 - a1_k
    a2_n  = 1.0 - a1_n

    # Cell-centre EOS evaluations at current iterate
    rho1_k = eos1.rho(p_k, T_k)
    rho2_k = eos2.rho(p_k, T_k)
    rho_k  = a1_k * rho1_k + a2_k * rho2_k

    # Cell-centre EOS evaluations at old time (forward — no inversion)
    rho1_n = eos1.rho(p_n, T_n)
    rho2_n = eos2.rho(p_n, T_n)
    rho_n  = a1_n * rho1_n + a2_n * rho2_n

    # Volumetric internal energies
    evol1_k = eos1.e_vol(p_k, T_k)
    evol2_k = eos2.e_vol(p_k, T_k)
    evol1_n = eos1.e_vol(p_n, T_n)
    evol2_n = eos2.e_vol(p_n, T_n)

    # Total energy density  ρE = α₁ρ₁e₁ + α₂ρ₂e₂ + ½ρu²
    rE_k = a1_k * evol1_k + a2_k * evol2_k + 0.5 * rho_k * u_k**2
    rE_n = a1_n * evol1_n + a2_n * evol2_n + 0.5 * rho_n * u_n**2

    # Phase-1 partial density
    a1r1_k = a1_k * rho1_k
    a1r1_n = a1_n * rho1_n

    # Momentum density
    ru_k = rho_k * u_k
    ru_n = rho_n * u_n

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def _wrap(il, ir):
        if il < 0:
            il = (N - 1) if is_per_l else 0
        if ir >= N:
            ir = 0 if is_per_r else N - 1
        return il, ir

    # Extended arrays (n_ghost=1) for upwind access
    ng = 1
    p_ext   = apply_ghost(p_k,   bc_l, bc_r, ng)
    T_ext   = apply_ghost(T_k,   bc_l, bc_r, ng)
    a1_ext  = apply_ghost(a1_k,  bc_l, bc_r, ng)
    u_ext   = apply_ghost(u_k,   bc_l, bc_r, ng)

    R = np.zeros(4 * N)

    for i in range(N):
        f_R = i + 1
        f_L = i
        iL, _ = _wrap(i - 1, i)
        _, iR  = _wrap(i, i + 1)

        iL_ext = ng + i - 1
        i_ext  = ng + i
        iR_ext = ng + i + 1

        tR = theta_k[f_R]
        tL = theta_k[f_L]

        # Upwind cell indices for left/right faces
        up_ext_R = i_ext  if tR >= 0 else iR_ext
        up_ext_L = iL_ext if tL >= 0 else i_ext

        p_upR = p_ext[up_ext_R];  T_upR = T_ext[up_ext_R];  a1_upR = a1_ext[up_ext_R]
        p_upL = p_ext[up_ext_L];  T_upL = T_ext[up_ext_L];  a1_upL = a1_ext[up_ext_L]
        u_upR = u_ext[up_ext_R]
        u_upL = u_ext[up_ext_L]

        # Phase-1 partial density at upwind cells
        r1_upR = eos1.rho(p_upR, T_upR)
        r1_upL = eos1.rho(p_upL, T_upL)
        r2_upR = eos2.rho(p_upR, T_upR)
        r2_upL = eos2.rho(p_upL, T_upL)
        rho_upR = a1_upR * r1_upR + (1.0 - a1_upR) * r2_upR
        rho_upL = a1_upL * r1_upL + (1.0 - a1_upL) * r2_upL

        # Phase-1 mass flux at faces: (α₁ρ₁)_up · θ
        a1r1_upR = a1_upR * r1_upR
        a1r1_upL = a1_upL * r1_upL

        # R1: phase 1 mass conservation
        R[0*N + i] = ((a1r1_k[i] - a1r1_n[i]) / dt
                      + (a1r1_upR * tR - a1r1_upL * tL) / dx)

        # Momentum flux: ρu_up · θ + p_face (arithmetic mean)
        ru_upR = rho_upR * u_upR
        ru_upL = rho_upL * u_upL
        pR_face = 0.5 * (p_k[i] + p_k[iR])
        pL_face = 0.5 * (p_k[iL] + p_k[i])

        # R2: momentum conservation
        R[1*N + i] = ((ru_k[i] - ru_n[i]) / dt
                      + (ru_upR * tR + pR_face - ru_upL * tL - pL_face) / dx)

        # Energy flux: (ρE + p)_up · θ
        evol1_upR = eos1.e_vol(p_upR, T_upR)
        evol2_upR = eos2.e_vol(p_upR, T_upR)
        evol1_upL = eos1.e_vol(p_upL, T_upL)
        evol2_upL = eos2.e_vol(p_upL, T_upL)
        a2_upR = 1.0 - a1_upR
        a2_upL = 1.0 - a1_upL
        rE_upR = a1_upR * evol1_upR + a2_upR * evol2_upR + 0.5 * rho_upR * u_upR**2
        rE_upL = a1_upL * evol1_upL + a2_upL * evol2_upL + 0.5 * rho_upL * u_upL**2
        flux_E_R = (rE_upR + p_upR) * tR
        flux_E_L = (rE_upL + p_upL) * tL

        # R3: energy conservation
        R[2*N + i] = ((rE_k[i] - rE_n[i]) / dt
                      + (flux_E_R - flux_E_L) / dx)

        # R4: alpha advection (non-conservative + K compression)
        # ∂α₁/∂t + (θ·α₁_up)_R/dx - (θ·α₁_up)_L/dx - α₁·div_θ - K·div_θ = 0
        a1_advR = a1_k[i]  if tR >= 0 else a1_k[iR]
        a1_advL = a1_k[iL] if tL >= 0 else a1_k[i]
        div_theta = (tR - tL) / dx
        R[3*N + i] = ((a1_k[i] - a1_n[i]) / dt
                      + (tR * a1_advR - tL * a1_advL) / dx
                      - a1_k[i] * div_theta
                      - K_k[i] * div_theta)

    return R


def assemble_jacobian_5eq_prim(p_k, u_k, T_k, a1_k,
                                K_k, theta_k,
                                N, dx, dt, ph1, ph2, bc_l, bc_r,
                                d_hat_k=None):
    """4N×4N sparse Jacobian for primitive-variable 5-eq Newton.

    Column ordering: W = [p(0..N-1) | u(N..2N-1) | T(2N..3N-1) | α₁(3N..4N-1)]
    Row    ordering: R = [R_mass1   | R_mom      | R_energy    | R_alpha    ]

    θ is frozen by default (Picard for MWI) → ∂θ/∂W = 0.
    If d_hat_k is provided, MWI face velocity Jacobian is included:
        θ_f = 0.5*(u_L + u_R) - d̂_f*(p_R - p_L)/dx
        ∂θ/∂u_L = 0.5,  ∂θ/∂u_R = 0.5
        ∂θ/∂p_L = +d̂/dx, ∂θ/∂p_R = -d̂/dx

    Parameters
    ----------
    p_k, u_k, T_k, a1_k : ndarray (N,)  current Newton iterate
    K_k                  : ndarray (N,)  Wood K coefficient
    theta_k              : ndarray (N+1,)  MWI face velocity
    N, dx, dt            : int, float, float
    ph1, ph2             : EOS parameter dicts
    bc_l, bc_r           : str

    Returns
    -------
    A_csr : scipy csr_matrix (4N, 4N)
    """
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    a1_k = np.clip(a1_k, 0.0, 1.0)
    a2_k = 1.0 - a1_k

    n4 = 4 * N
    A = sp.lil_matrix((n4, n4), dtype=float)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def _wrap(il, ir):
        if il < 0:
            il = (N - 1) if is_per_l else 0
        if ir >= N:
            ir = 0 if is_per_r else N - 1
        return il, ir

    ng = 1
    p_ext  = apply_ghost(p_k,  bc_l, bc_r, ng)
    T_ext  = apply_ghost(T_k,  bc_l, bc_r, ng)
    a1_ext = apply_ghost(a1_k, bc_l, bc_r, ng)

    for i in range(N):
        f_R = i + 1
        f_L = i
        iL, _ = _wrap(i - 1, i)
        _, iR  = _wrap(i, i + 1)

        iL_ext = ng + i - 1
        i_ext  = ng + i
        iR_ext = ng + i + 1

        tR = theta_k[f_R]
        tL = theta_k[f_L]

        up_ext_R = i_ext  if tR >= 0 else iR_ext
        up_ext_L = iL_ext if tL >= 0 else i_ext
        up_R = i  if tR >= 0 else iR
        up_L = iL if tL >= 0 else i

        p_upR  = p_ext[up_ext_R];   T_upR  = T_ext[up_ext_R];   a1_upR = a1_ext[up_ext_R]
        p_upL  = p_ext[up_ext_L];   T_upL  = T_ext[up_ext_L];   a1_upL = a1_ext[up_ext_L]

        rho1_i = eos1.rho(p_k[i], T_k[i])
        rho2_i = eos2.rho(p_k[i], T_k[i])
        rho_i  = a1_k[i] * rho1_i + a2_k[i] * rho2_i
        zeta1_i = eos1.drho_dp(p_k[i], T_k[i])
        zeta2_i = eos2.drho_dp(p_k[i], T_k[i])
        phi1_i  = eos1.drho_dT(p_k[i], T_k[i])
        phi2_i  = eos2.drho_dT(p_k[i], T_k[i])
        zeta_mix_i = a1_k[i] * zeta1_i + a2_k[i] * zeta2_i
        phi_mix_i  = a1_k[i] * phi1_i  + a2_k[i] * phi2_i

        evol1_i = eos1.e_vol(p_k[i], T_k[i])
        evol2_i = eos2.e_vol(p_k[i], T_k[i])
        de_vol1_dp_i = eos1.de_vol_dp(p_k[i], T_k[i])
        de_vol2_dp_i = eos2.de_vol_dp(p_k[i], T_k[i])
        de_vol1_dT_i = eos1.de_vol_dT(p_k[i], T_k[i])
        de_vol2_dT_i = eos2.de_vol_dT(p_k[i], T_k[i])

        u_i = u_k[i]

        # --------------------------------------------------------
        # TEMPORAL terms — diagonal cell i
        # --------------------------------------------------------
        # R1 (phase 1 mass): d(α₁ρ₁)/dt
        #   ∂(α₁ρ₁)/∂p = α₁·ζ₁
        #   ∂(α₁ρ₁)/∂T = α₁·φ₁
        #   ∂(α₁ρ₁)/∂α₁ = ρ₁
        A[_ci4(0, i, N), _ci4(0, i, N)] += a1_k[i] * zeta1_i / dt
        A[_ci4(0, i, N), _ci4(2, i, N)] += a1_k[i] * phi1_i  / dt
        A[_ci4(0, i, N), _ci4(3, i, N)] += rho1_i / dt

        # R2 (momentum): d(ρu)/dt
        #   ∂(ρu)/∂p = ζ_mix·u
        #   ∂(ρu)/∂u = ρ
        #   ∂(ρu)/∂T = φ_mix·u
        #   ∂(ρu)/∂α₁ = (ρ₁-ρ₂)·u
        A[_ci4(1, i, N), _ci4(0, i, N)] += zeta_mix_i * u_i / dt
        A[_ci4(1, i, N), _ci4(1, i, N)] += rho_i / dt
        A[_ci4(1, i, N), _ci4(2, i, N)] += phi_mix_i * u_i / dt
        A[_ci4(1, i, N), _ci4(3, i, N)] += (rho1_i - rho2_i) * u_i / dt

        # R3 (energy): d(ρE)/dt  where ρE = α₁·e_vol1 + α₂·e_vol2 + ½ρu²
        #   ∂(ρE)/∂p = α₁·de_vol1_dp + α₂·de_vol2_dp + ½u²·ζ_mix
        #   ∂(ρE)/∂u = ρ·u
        #   ∂(ρE)/∂T = α₁·de_vol1_dT + α₂·de_vol2_dT + ½u²·φ_mix
        #   ∂(ρE)/∂α₁ = e_vol1 - e_vol2 + ½u²·(ρ₁-ρ₂)
        drE_dp_i  = a1_k[i]*de_vol1_dp_i + a2_k[i]*de_vol2_dp_i + 0.5*u_i**2*zeta_mix_i
        drE_du_i  = rho_i * u_i
        drE_dT_i  = a1_k[i]*de_vol1_dT_i + a2_k[i]*de_vol2_dT_i + 0.5*u_i**2*phi_mix_i
        drE_da1_i = evol1_i - evol2_i + 0.5*u_i**2*(rho1_i - rho2_i)

        A[_ci4(2, i, N), _ci4(0, i, N)] += drE_dp_i  / dt
        A[_ci4(2, i, N), _ci4(1, i, N)] += drE_du_i  / dt
        A[_ci4(2, i, N), _ci4(2, i, N)] += drE_dT_i  / dt
        A[_ci4(2, i, N), _ci4(3, i, N)] += drE_da1_i / dt

        # R4 (alpha): d(α₁)/dt
        A[_ci4(3, i, N), _ci4(3, i, N)] += 1.0 / dt

        # --------------------------------------------------------
        # SPATIAL terms
        # --------------------------------------------------------
        # --- Phase-1 mass spatial flux: (α₁·ρ₁)_up · θ ---
        # Upwind cell for right face: up_R; for left face: up_L
        r1_upR  = eos1.rho(p_upR, T_upR)
        r1_upL  = eos1.rho(p_upL, T_upL)
        z1_upR  = eos1.drho_dp(p_upR, T_upR)
        z1_upL  = eos1.drho_dp(p_upL, T_upL)
        ph1_upR = eos1.drho_dT(p_upR, T_upR)
        ph1_upL = eos1.drho_dT(p_upL, T_upL)

        # Right face +tR/dx contribution to row i (mass eq)
        #   ∂(a1_upR · r1_upR · tR) / ∂p_up_R   = a1_upR · z1_upR · tR
        #   ∂(a1_upR · r1_upR · tR) / ∂T_up_R   = a1_upR · ph1_upR · tR
        #   ∂(a1_upR · r1_upR · tR) / ∂a1_up_R  = r1_upR · tR
        A[_ci4(0, i, N), _ci4(0, up_R, N)] += a1_upR * z1_upR * tR / dx
        A[_ci4(0, i, N), _ci4(2, up_R, N)] += a1_upR * ph1_upR * tR / dx
        A[_ci4(0, i, N), _ci4(3, up_R, N)] += r1_upR * tR / dx

        # Left face -tL/dx contribution to row i (mass eq)
        A[_ci4(0, i, N), _ci4(0, up_L, N)] -= a1_upL * z1_upL * tL / dx
        A[_ci4(0, i, N), _ci4(2, up_L, N)] -= a1_upL * ph1_upL * tL / dx
        A[_ci4(0, i, N), _ci4(3, up_L, N)] -= r1_upL * tL / dx

        # --- Momentum spatial flux: ρ_up·u_up·θ + p_face ---
        r2_upR  = eos2.rho(p_upR, T_upR)
        r2_upL  = eos2.rho(p_upL, T_upL)
        z2_upR  = eos2.drho_dp(p_upR, T_upR)
        z2_upL  = eos2.drho_dp(p_upL, T_upL)
        ph2_upR = eos2.drho_dT(p_upR, T_upR)
        ph2_upL = eos2.drho_dT(p_upL, T_upL)
        rho_upR = a1_upR * r1_upR + (1.0 - a1_upR) * r2_upR
        rho_upL = a1_upL * r1_upL + (1.0 - a1_upL) * r2_upL
        zeta_mix_upR = a1_upR * z1_upR + (1.0 - a1_upR) * z2_upR
        zeta_mix_upL = a1_upL * z1_upL + (1.0 - a1_upL) * z2_upL
        phi_mix_upR  = a1_upR * ph1_upR + (1.0 - a1_upR) * ph2_upR
        phi_mix_upL  = a1_upL * ph1_upL + (1.0 - a1_upL) * ph2_upL

        u_upR_val = u_k[up_R]
        u_upL_val = u_k[up_L]

        # Right face +tR/dx:  ρ_upR · u_upR · tR
        # ∂/∂p_upR  = zeta_mix_upR · u_upR · tR
        # ∂/∂u_upR  = rho_upR · tR
        # ∂/∂T_upR  = phi_mix_upR · u_upR · tR
        # ∂/∂a1_upR = (r1_upR - r2_upR) · u_upR · tR
        A[_ci4(1, i, N), _ci4(0, up_R, N)] += zeta_mix_upR * u_upR_val * tR / dx
        A[_ci4(1, i, N), _ci4(1, up_R, N)] += rho_upR * tR / dx
        A[_ci4(1, i, N), _ci4(2, up_R, N)] += phi_mix_upR * u_upR_val * tR / dx
        A[_ci4(1, i, N), _ci4(3, up_R, N)] += (r1_upR - r2_upR) * u_upR_val * tR / dx

        # Left face -tL/dx
        A[_ci4(1, i, N), _ci4(0, up_L, N)] -= zeta_mix_upL * u_upL_val * tL / dx
        A[_ci4(1, i, N), _ci4(1, up_L, N)] -= rho_upL * tL / dx
        A[_ci4(1, i, N), _ci4(2, up_L, N)] -= phi_mix_upL * u_upL_val * tL / dx
        A[_ci4(1, i, N), _ci4(3, up_L, N)] -= (r1_upL - r2_upL) * u_upL_val * tL / dx

        # Pressure gradient: ∂(p_R_face - p_L_face)/∂p
        # p_R_face = 0.5*(p_i + p_iR):  ∂/∂p_i = +0.5, ∂/∂p_iR = +0.5
        # p_L_face = 0.5*(p_iL + p_i):  ∂/∂p_iL = -0.5, ∂/∂p_i = -0.5  (net=0 at i)
        A[_ci4(1, i, N), _ci4(0, iR, N)] += 0.5 / dx
        A[_ci4(1, i, N), _ci4(0, iL, N)] -= 0.5 / dx

        # --- Energy spatial flux: (ρE + p)_up · θ ---
        ev1_upR  = eos1.e_vol(p_upR, T_upR);  ev2_upR  = eos2.e_vol(p_upR, T_upR)
        ev1_upL  = eos1.e_vol(p_upL, T_upL);  ev2_upL  = eos2.e_vol(p_upL, T_upL)
        dev1dp_upR = eos1.de_vol_dp(p_upR, T_upR); dev2dp_upR = eos2.de_vol_dp(p_upR, T_upR)
        dev1dp_upL = eos1.de_vol_dp(p_upL, T_upL); dev2dp_upL = eos2.de_vol_dp(p_upL, T_upL)
        dev1dT_upR = eos1.de_vol_dT(p_upR, T_upR); dev2dT_upR = eos2.de_vol_dT(p_upR, T_upR)
        dev1dT_upL = eos1.de_vol_dT(p_upL, T_upL); dev2dT_upL = eos2.de_vol_dT(p_upL, T_upL)

        a2_upR = 1.0 - a1_upR
        a2_upL = 1.0 - a1_upL
        rE_upR_val  = a1_upR*ev1_upR + a2_upR*ev2_upR + 0.5*rho_upR*u_upR_val**2
        rE_upL_val  = a1_upL*ev1_upL + a2_upL*ev2_upL + 0.5*rho_upL*u_upL_val**2

        # d(ρE+p)/dp at upwind cell (right face):
        #   d(α₁·ev1+α₂·ev2)/dp + ½u²·ζ_mix + 1 (from +p term)
        d_rEp_dp_R  = (a1_upR*dev1dp_upR + a2_upR*dev2dp_upR
                       + 0.5*u_upR_val**2*zeta_mix_upR + 1.0)
        d_rEp_dT_R  = (a1_upR*dev1dT_upR + a2_upR*dev2dT_upR
                       + 0.5*u_upR_val**2*phi_mix_upR)
        d_rEp_du_R  = rho_upR * u_upR_val
        d_rEp_da1_R = ev1_upR - ev2_upR + 0.5*u_upR_val**2*(r1_upR - r2_upR)

        d_rEp_dp_L  = (a1_upL*dev1dp_upL + a2_upL*dev2dp_upL
                       + 0.5*u_upL_val**2*zeta_mix_upL + 1.0)
        d_rEp_dT_L  = (a1_upL*dev1dT_upL + a2_upL*dev2dT_upL
                       + 0.5*u_upL_val**2*phi_mix_upL)
        d_rEp_du_L  = rho_upL * u_upL_val
        d_rEp_da1_L = ev1_upL - ev2_upL + 0.5*u_upL_val**2*(r1_upL - r2_upL)

        # Right face +tR/dx contribution
        A[_ci4(2, i, N), _ci4(0, up_R, N)] += d_rEp_dp_R  * tR / dx
        A[_ci4(2, i, N), _ci4(1, up_R, N)] += d_rEp_du_R  * tR / dx
        A[_ci4(2, i, N), _ci4(2, up_R, N)] += d_rEp_dT_R  * tR / dx
        A[_ci4(2, i, N), _ci4(3, up_R, N)] += d_rEp_da1_R * tR / dx

        # Left face -tL/dx contribution
        A[_ci4(2, i, N), _ci4(0, up_L, N)] -= d_rEp_dp_L  * tL / dx
        A[_ci4(2, i, N), _ci4(1, up_L, N)] -= d_rEp_du_L  * tL / dx
        A[_ci4(2, i, N), _ci4(2, up_L, N)] -= d_rEp_dT_L  * tL / dx
        A[_ci4(2, i, N), _ci4(3, up_L, N)] -= d_rEp_da1_L * tL / dx

        # --------------------------------------------------------
        # MWI FACE VELOCITY JACOBIAN: ∂θ/∂W terms
        # θ_f = 0.5*(u_L + u_R) - d̂_f*(p_R - p_L)/dx
        # ∂θ_f/∂u_L = 0.5,  ∂θ_f/∂u_R = 0.5
        # ∂θ_f/∂p_L = +d̂_f/dx,  ∂θ_f/∂p_R = -d̂_f/dx
        #
        # For each spatial flux term F_up * θ_f, the θ Jacobian adds:
        #   row via right face (+1/dx):  F_up * ∂θ_R/∂W * (1/dx)
        #   row via left face  (-1/dx): -F_up * ∂θ_L/∂W * (1/dx)
        # --------------------------------------------------------
        if d_hat_k is not None:
            d_R = d_hat_k[f_R]
            d_L = d_hat_k[f_L]

            # Precompute upwind flux "amplitudes" for each equation
            a1r1_upR_val = a1_upR * r1_upR
            a1r1_upL_val = a1_upL * r1_upL
            ru_upR_val   = rho_upR * u_upR_val
            ru_upL_val   = rho_upL * u_upL_val
            rEp_upR_val  = rE_upR_val + p_upR
            rEp_upL_val  = rE_upL_val + p_upL

            # --- R1 (phase-1 mass): spatial flux = a1r1_up * θ ---
            # Right face (+1/dx): ∂θ_R/∂u_i = 0.5, ∂θ_R/∂u_iR = 0.5
            #                     ∂θ_R/∂p_i = +d̂_R/dx, ∂θ_R/∂p_iR = -d̂_R/dx
            A[_ci4(0, i, N), _ci4(1, i,  N)] += a1r1_upR_val * 0.5 / dx
            A[_ci4(0, i, N), _ci4(1, iR, N)] += a1r1_upR_val * 0.5 / dx
            A[_ci4(0, i, N), _ci4(0, i,  N)] += a1r1_upR_val * (d_R / dx) / dx
            A[_ci4(0, i, N), _ci4(0, iR, N)] += a1r1_upR_val * (-d_R / dx) / dx
            # Left face (-1/dx): ∂θ_L/∂u_iL = 0.5, ∂θ_L/∂u_i = 0.5
            #                    ∂θ_L/∂p_iL = +d̂_L/dx, ∂θ_L/∂p_i = -d̂_L/dx
            A[_ci4(0, i, N), _ci4(1, iL, N)] -= a1r1_upL_val * 0.5 / dx
            A[_ci4(0, i, N), _ci4(1, i,  N)] -= a1r1_upL_val * 0.5 / dx
            A[_ci4(0, i, N), _ci4(0, iL, N)] -= a1r1_upL_val * (d_L / dx) / dx
            A[_ci4(0, i, N), _ci4(0, i,  N)] -= a1r1_upL_val * (-d_L / dx) / dx

            # --- R2 (momentum): spatial flux = ru_up * θ + p_face ---
            # Only the ru_up * θ part has θ coupling (p_face independent of θ)
            A[_ci4(1, i, N), _ci4(1, i,  N)] += ru_upR_val * 0.5 / dx
            A[_ci4(1, i, N), _ci4(1, iR, N)] += ru_upR_val * 0.5 / dx
            A[_ci4(1, i, N), _ci4(0, i,  N)] += ru_upR_val * (d_R / dx) / dx
            A[_ci4(1, i, N), _ci4(0, iR, N)] += ru_upR_val * (-d_R / dx) / dx

            A[_ci4(1, i, N), _ci4(1, iL, N)] -= ru_upL_val * 0.5 / dx
            A[_ci4(1, i, N), _ci4(1, i,  N)] -= ru_upL_val * 0.5 / dx
            A[_ci4(1, i, N), _ci4(0, iL, N)] -= ru_upL_val * (d_L / dx) / dx
            A[_ci4(1, i, N), _ci4(0, i,  N)] -= ru_upL_val * (-d_L / dx) / dx

            # --- R3 (energy): spatial flux = (ρE+p)_up * θ ---
            A[_ci4(2, i, N), _ci4(1, i,  N)] += rEp_upR_val * 0.5 / dx
            A[_ci4(2, i, N), _ci4(1, iR, N)] += rEp_upR_val * 0.5 / dx
            A[_ci4(2, i, N), _ci4(0, i,  N)] += rEp_upR_val * (d_R / dx) / dx
            A[_ci4(2, i, N), _ci4(0, iR, N)] += rEp_upR_val * (-d_R / dx) / dx

            A[_ci4(2, i, N), _ci4(1, iL, N)] -= rEp_upL_val * 0.5 / dx
            A[_ci4(2, i, N), _ci4(1, i,  N)] -= rEp_upL_val * 0.5 / dx
            A[_ci4(2, i, N), _ci4(0, iL, N)] -= rEp_upL_val * (d_L / dx) / dx
            A[_ci4(2, i, N), _ci4(0, i,  N)] -= rEp_upL_val * (-d_L / dx) / dx

            # --- R4 (alpha): θ_R*a1_advR/dx - θ_L*a1_advL/dx contribution ---
            # a1_advR = a1_k[i] if tR>=0 else a1_k[iR]  (upwind alpha)
            # a1_advL = a1_k[iL] if tL>=0 else a1_k[i]
            a1_advR_mwi = a1_k[i]  if tR >= 0 else a1_k[iR]
            a1_advL_mwi = a1_k[iL] if tL >= 0 else a1_k[i]

            A[_ci4(3, i, N), _ci4(1, i,  N)] += a1_advR_mwi * 0.5 / dx
            A[_ci4(3, i, N), _ci4(1, iR, N)] += a1_advR_mwi * 0.5 / dx
            A[_ci4(3, i, N), _ci4(0, i,  N)] += a1_advR_mwi * (d_R / dx) / dx
            A[_ci4(3, i, N), _ci4(0, iR, N)] += a1_advR_mwi * (-d_R / dx) / dx

            A[_ci4(3, i, N), _ci4(1, iL, N)] -= a1_advL_mwi * 0.5 / dx
            A[_ci4(3, i, N), _ci4(1, i,  N)] -= a1_advL_mwi * 0.5 / dx
            A[_ci4(3, i, N), _ci4(0, iL, N)] -= a1_advL_mwi * (d_L / dx) / dx
            A[_ci4(3, i, N), _ci4(0, i,  N)] -= a1_advL_mwi * (-d_L / dx) / dx

            # -α₁·div_θ - K·div_θ: θ Jacobian for div_θ = (θ_R - θ_L)/dx
            # ∂(div_θ)/∂u_iR = +0.5/dx,  ∂(div_θ)/∂u_iL = -0.5/dx
            # ∂(div_θ)/∂u_i  = 0.5/dx - 0.5/dx = 0 (cancel between two faces)
            # ∂(div_θ)/∂p terms: skip (d̂ is very small, negligible)
            coeff_div = -(a1_k[i] + K_k[i])
            A[_ci4(3, i, N), _ci4(1, iR, N)] += coeff_div * 0.5 / dx
            A[_ci4(3, i, N), _ci4(1, iL, N)] -= coeff_div * 0.5 / dx

        # --- Alpha equation spatial ---
        # (θ·α₁_up)_R/dx - (θ·α₁_up)_L/dx - α₁·div_θ - K·div_θ
        # Only ∂/∂α₁ (θ and K frozen):
        div_theta_i = (tR - tL) / dx

        if tR >= 0:
            # a1_advR = a1_k[i]
            A[_ci4(3, i, N), _ci4(3, i, N)]  += tR / dx
        else:
            # a1_advR = a1_k[iR]
            A[_ci4(3, i, N), _ci4(3, iR, N)] += tR / dx

        if tL >= 0:
            # a1_advL = a1_k[iL]
            A[_ci4(3, i, N), _ci4(3, iL, N)] -= tL / dx
        else:
            # a1_advL = a1_k[i]
            A[_ci4(3, i, N), _ci4(3, i, N)]  -= tL / dx

        # -α₁·div_θ: ∂/∂α₁_i = -div_theta (diagonal)
        A[_ci4(3, i, N), _ci4(3, i, N)] -= div_theta_i

    return A.tocsr()
