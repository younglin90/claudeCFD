"""Primitive W = (α₁, T₁, T₂, u, p) ↔ Conservative U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁) layer.

This module supplies the building blocks for an IMEX implicit Newton on the
five-equation Kapila / Allaire-Massoni model with general EOS:

    W = (alpha1, T1, T2, u, p)^T          # primitive
    U = (alpha1·rho1,                     # phase-1 partial mass
         alpha2·rho2,                     # phase-2 partial mass
         rho·u,                           # momentum
         rho·E,                           # total energy
         alpha1)^T                        # interface fraction

The 5×5 Jacobian dU/dW is closed-form and uses only the standard EOS
derivatives (∂ρ_k/∂p)_T, (∂ρ_k/∂T)_p, (∂e_k/∂p)_T, (∂e_k/∂T)_p — added in
Phase 1 to `solver/He2024/eos_general.py`.

The implementation is vectorised over cell arrays of shape (N,) so it can be
plugged into the existing solver loops without copying.
"""
from __future__ import annotations
import numpy as np

from .eos_general import to_eos


_EPS = 1e-30


# ─── primitive ↔ conservative ──────────────────────────────────────────────
def prim_to_cons_W(W, ph1, ph2):
    """W = (α, T1, T2, u, p)  →  U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁).

    All five components must be (N,) arrays. Returns the five U components
    in the same order with their phase densities, internal energies and the
    mixture density / specific internal energy as a diagnostic dict.
    """
    eos1 = to_eos(ph1); eos2 = to_eos(ph2)
    alpha1, T1, T2, u, p = (np.asarray(W[i], dtype=float) for i in range(5))
    beta = 1.0 - alpha1

    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1   = eos1.energy(rho1, p)
    e2   = eos2.energy(rho2, p)

    rho = alpha1 * rho1 + beta * rho2
    q = 0.5 * u * u
    rho_E = alpha1 * rho1 * (e1 + q) + beta * rho2 * (e2 + q)

    U = (alpha1 * rho1,
         beta   * rho2,
         rho * u,
         rho_E,
         alpha1)
    aux = dict(rho1=rho1, rho2=rho2, e1=e1, e2=e2, rho=rho)
    return U, aux


def cons_to_prim_W(U, ph1, ph2,
                   T1_init=None, T2_init=None,
                   tol=1e-9, max_iter=30):
    """U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)  →  W = (α₁, T₁, T₂, u, p).

    Uses Newton on the 3-vector (T1, T2, p), with α₁ and u recovered first:
        α₁ = U₅
        ρ  = U₁ + U₂      (positive)
        u  = U₃ / ρ
        ρe = U₄ − ½ ρu²

    Residual:
        F1(p, T1) = U₁ − α₁ · ρ₁(p, T₁)               = 0
        F2(p, T2) = U₂ − (1−α₁) · ρ₂(p, T₂)            = 0
        F3(p, T1, T2) = ρe − U₁·e₁(p,T₁) − U₂·e₂(p,T₂) = 0

    Jacobian uses analytic ∂ρ/∂p, ∂ρ/∂T, ∂e/∂p, ∂e/∂T from EOS layer.
    Falls back to per-phase 1-D Newton if the 3×3 system fails (e.g. pure-phase
    cells where one equation degenerates).
    """
    eos1 = to_eos(ph1); eos2 = to_eos(ph2)
    U1, U2, U3, U4, U5 = (np.asarray(U[i], dtype=float) for i in range(5))

    alpha1 = np.clip(U5, 0.0, 1.0)
    rho = np.maximum(U1 + U2, _EPS)
    u = U3 / rho
    rho_e = U4 - 0.5 * U3 * u

    # Initial guess for (T1, T2, p)
    if T1_init is None or T2_init is None:
        # Use a 1-bar / 300 K seed but scale T to majority phase
        rho1_seed = np.maximum(U1 / np.maximum(alpha1, 1e-8), _EPS)
        rho2_seed = np.maximum(U2 / np.maximum(1.0 - alpha1, 1e-8), _EPS)
        e1_seed = np.maximum(rho_e / np.maximum(rho, _EPS) / 2.0, 1.0)
        e2_seed = e1_seed
        T1 = np.maximum(eos1.temperature(rho1_seed, e1_seed), 1.0)
        T2 = np.maximum(eos2.temperature(rho2_seed, e2_seed), 1.0)
    else:
        T1 = np.maximum(np.asarray(T1_init, dtype=float), 1.0)
        T2 = np.maximum(np.asarray(T2_init, dtype=float), 1.0)

    # Pressure seed: from majority phase ρ + e
    use_phase1 = alpha1 >= 0.5
    rho1_g = np.maximum(U1 / np.maximum(alpha1, 1e-8), _EPS)
    rho2_g = np.maximum(U2 / np.maximum(1.0 - alpha1, 1e-8), _EPS)
    e1_g = eos1.energy(rho1_g, eos1.pressure_from_rhoT(rho1_g, T1))
    p_seed_1 = eos1.pressure(rho1_g, e1_g)
    p_seed_2 = eos2.pressure(rho2_g, eos2.energy(rho2_g, eos2.pressure_from_rhoT(rho2_g, T2)))
    p = np.where(use_phase1, p_seed_1, p_seed_2)
    p = np.maximum(p, 1.0)

    # 3×3 Newton (vectorised — each cell solved independently)
    for _ in range(max_iter):
        rho1 = eos1.density(p, T1); rho1 = np.maximum(rho1, _EPS)
        rho2 = eos2.density(p, T2); rho2 = np.maximum(rho2, _EPS)
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)

        F1 = U1 - alpha1 * rho1
        F2 = U2 - (1.0 - alpha1) * rho2
        F3 = rho_e - U1 * e1 - U2 * e2

        # Convergence test
        scale = np.maximum(np.abs(rho_e), 1.0)
        res = np.maximum.reduce([np.abs(F1) / np.maximum(np.abs(U1), 1.0),
                                 np.abs(F2) / np.maximum(np.abs(U2), 1.0),
                                 np.abs(F3) / scale])
        if np.max(res) < tol:
            break

        # Build 3×3 Jacobian J for (p, T1, T2)
        drho1_dp = eos1.drhodp_T(rho1, T1); drho1_dT = eos1.drhodT_p(rho1, T1)
        drho2_dp = eos2.drhodp_T(rho2, T2); drho2_dT = eos2.drhodT_p(rho2, T2)
        de1_dp   = eos1.dedp_T(rho1, T1);   de1_dT   = eos1.dedT_p(rho1, T1)
        de2_dp   = eos2.dedp_T(rho2, T2);   de2_dT   = eos2.dedT_p(rho2, T2)

        # ∂F/∂(p, T1, T2) :
        # F1: p ↦ -α·∂ρ1/∂p,  T1 ↦ -α·∂ρ1/∂T,  T2 ↦ 0
        # F2: p ↦ -(1-α)·∂ρ2/∂p, T1 ↦ 0, T2 ↦ -(1-α)·∂ρ2/∂T
        # F3: p ↦ -U1·∂e1/∂p − U2·∂e2/∂p, T1 ↦ -U1·∂e1/∂T, T2 ↦ -U2·∂e2/∂T
        J11 = -alpha1 * drho1_dp
        J12 = -alpha1 * drho1_dT
        J13 = np.zeros_like(J11)
        J21 = -(1.0 - alpha1) * drho2_dp
        J22 = np.zeros_like(J11)
        J23 = -(1.0 - alpha1) * drho2_dT
        J31 = -U1 * de1_dp - U2 * de2_dp
        J32 = -U1 * de1_dT
        J33 = -U2 * de2_dT

        # Closed-form 3×3 solve cell-wise (Cramer)
        # A = [[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]]
        det = (J11 * (J22 * J33 - J23 * J32)
               - J12 * (J21 * J33 - J23 * J31)
               + J13 * (J21 * J32 - J22 * J31))
        det = np.where(np.abs(det) < 1e-30, 1e-30 * np.sign(det + 1e-60), det)

        # solve J · dx = -F
        b1 = -F1; b2 = -F2; b3 = -F3
        d_p = (b1 * (J22 * J33 - J23 * J32)
               - J12 * (b2 * J33 - J23 * b3)
               + J13 * (b2 * J32 - J22 * b3)) / det
        d_T1 = (J11 * (b2 * J33 - J23 * b3)
                - b1 * (J21 * J33 - J23 * J31)
                + J13 * (J21 * b3 - b2 * J31)) / det
        d_T2 = (J11 * (J22 * b3 - b2 * J32)
                - J12 * (J21 * b3 - b2 * J31)
                + b1 * (J21 * J32 - J22 * J31)) / det

        # Damped update with positivity
        damp = 1.0
        p_new = p + damp * d_p
        T1_new = T1 + damp * d_T1
        T2_new = T2 + damp * d_T2

        # Line-search: shrink damp until p, T1, T2 > 0
        for _line in range(8):
            ok = (p_new > 1.0) & (T1_new > 1.0) & (T2_new > 1.0)
            if np.all(ok):
                break
            damp *= 0.5
            p_new = p + damp * d_p
            T1_new = T1 + damp * d_T1
            T2_new = T2 + damp * d_T2

        p = np.maximum(p_new, 1.0)
        T1 = np.maximum(T1_new, 1.0)
        T2 = np.maximum(T2_new, 1.0)

    W = (alpha1, T1, T2, u, p)
    return W


# ─── analytic dU/dW (5×5) ──────────────────────────────────────────────────
def dUdW_analytic(W, ph1, ph2, return_aux=False):
    """Compute dU/dW (5×5 per cell) in closed form.

    Following the row-by-row form from the user spec:
      Row 1: U1 = α·ρ1     (α derivative = ρ1; T1 = α·ρ1_T; T2=0; u=0; p = α·ρ1_p)
      Row 2: U2 = (1−α)·ρ2
      Row 3: U3 = ρ·u
      Row 4: U4 = ρE = α·ρ1(e1+q) + (1−α)·ρ2(e2+q),  q = ½ u²
      Row 5: U5 = α

    Output shape: (5, 5, N).  J[i, j, k] = ∂U_i / ∂W_j  at cell k.
    """
    eos1 = to_eos(ph1); eos2 = to_eos(ph2)
    alpha1, T1, T2, u, p = (np.asarray(W[i], dtype=float) for i in range(5))
    beta = 1.0 - alpha1
    q = 0.5 * u * u

    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1   = eos1.energy(rho1, p)
    e2   = eos2.energy(rho2, p)
    rho  = alpha1 * rho1 + beta * rho2

    rho1_p = eos1.drhodp_T(rho1, T1); rho1_T = eos1.drhodT_p(rho1, T1)
    rho2_p = eos2.drhodp_T(rho2, T2); rho2_T = eos2.drhodT_p(rho2, T2)
    e1_p   = eos1.dedp_T(rho1, T1);   e1_T   = eos1.dedT_p(rho1, T1)
    e2_p   = eos2.dedp_T(rho2, T2);   e2_T   = eos2.dedT_p(rho2, T2)

    N = alpha1.shape[0]
    J = np.zeros((5, 5, N), dtype=float)

    # Row 1: U1 = α · ρ1(p, T1)
    J[0, 0, :] = rho1
    J[0, 1, :] = alpha1 * rho1_T
    J[0, 2, :] = 0.0
    J[0, 3, :] = 0.0
    J[0, 4, :] = alpha1 * rho1_p

    # Row 2: U2 = (1 − α) · ρ2(p, T2)
    J[1, 0, :] = -rho2
    J[1, 1, :] = 0.0
    J[1, 2, :] = beta * rho2_T
    J[1, 3, :] = 0.0
    J[1, 4, :] = beta * rho2_p

    # Row 3: U3 = ρ · u
    J[2, 0, :] = u * (rho1 - rho2)
    J[2, 1, :] = alpha1 * u * rho1_T
    J[2, 2, :] = beta * u * rho2_T
    J[2, 3, :] = rho
    J[2, 4, :] = u * (alpha1 * rho1_p + beta * rho2_p)

    # Row 4: U4 = α · ρ1·(e1 + q) + (1-α) · ρ2·(e2 + q)
    h1 = e1 + q; h2 = e2 + q
    J[3, 0, :] = rho1 * h1 - rho2 * h2
    J[3, 1, :] = alpha1 * (h1 * rho1_T + rho1 * e1_T)
    J[3, 2, :] = beta * (h2 * rho2_T + rho2 * e2_T)
    J[3, 3, :] = rho * u
    J[3, 4, :] = (alpha1 * (h1 * rho1_p + rho1 * e1_p)
                  + beta   * (h2 * rho2_p + rho2 * e2_p))

    # Row 5: U5 = α
    J[4, 0, :] = 1.0
    J[4, 1, :] = 0.0
    J[4, 2, :] = 0.0
    J[4, 3, :] = 0.0
    J[4, 4, :] = 0.0

    if return_aux:
        return J, dict(rho1=rho1, rho2=rho2, e1=e1, e2=e2, rho=rho)
    return J


# ─── numerical Jacobian (validation only) ──────────────────────────────────
def dUdW_numerical(W, ph1, ph2, rel=1e-6):
    """5-point centered FD Jacobian on prim_to_cons_W. Validation only."""
    W_arr = [np.asarray(W[i], dtype=float).copy() for i in range(5)]
    N = W_arr[0].shape[0]
    J = np.zeros((5, 5, N), dtype=float)

    for j in range(5):
        wj = W_arr[j]
        # Per-component step that respects scale (α uses absolute step)
        if j == 0:                                # alpha
            dW = np.full_like(wj, rel)
        else:
            dW = np.maximum(np.abs(wj) * rel, rel)
        # +
        W_arr_p = [a.copy() for a in W_arr]; W_arr_p[j] = wj + dW
        Up, _ = prim_to_cons_W(W_arr_p, ph1, ph2)
        W_arr_m = [a.copy() for a in W_arr]; W_arr_m[j] = wj - dW
        Um, _ = prim_to_cons_W(W_arr_m, ph1, ph2)
        for i in range(5):
            J[i, j, :] = (Up[i] - Um[i]) / (2.0 * dW)
    return J
