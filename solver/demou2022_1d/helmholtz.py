# solver/demou2022_1d/helmholtz.py
"""
Pressure Helmholtz solver — Mode A, RK3 sub-step.

Equation (DENNER_SCHEME.md §3.2 Step 4):

    p^{m+1} - λ² ρc² ∇·(1/ρ ∇p^{m+1}) = f

where λ = γ̃_m Δt  (RK3 effective time step).

Discretised (cell i, interior):

    [1 + λ²(ρc²)_i (1/(ρ̄_e Δx²) + 1/(ρ̄_w Δx²))] p_i
    - λ²(ρc²)_i/(ρ̄_e Δx²) p_{i+1}
    - λ²(ρc²)_i/(ρ̄_w Δx²) p_{i-1}
    = f_i

RHS f_i (Demou2022):
    f_i = p^m_i + λ ρ_i c²_i [S_p^(2) (D_E + K)_i^m - div_u*_i]

Boundary conditions:
    transmissive : ∂p/∂n = 0  → ghost p = boundary p  (no off-diagonal at BC)
    periodic     : wrap-around (handled via dense system for general case)

Solved with Thomas algorithm (tridiagonal) for transmissive BC.
Periodic BC uses np.linalg.solve (dense, acceptable for 1D small N).
"""

import numpy as np
from scipy.linalg import solve_banded


def build_and_solve(p_old: np.ndarray,
                    rho: np.ndarray,
                    c2_mix: np.ndarray,
                    rho_face: np.ndarray,
                    div_u_star: np.ndarray,
                    Sp2: np.ndarray,
                    dissip_source: np.ndarray,
                    lam: float,
                    dx: float,
                    bc_l: str,
                    bc_r: str) -> np.ndarray:
    """
    Assemble and solve the Helmholtz pressure system.

    Parameters
    ----------
    p_old         : (N,) pressure at beginning of sub-step
    rho           : (N,) mixture density
    c2_mix        : (N,) mixture sound speed squared
    rho_face      : (N+1,) face densities (harmonic mean)
    div_u_star    : (N,) ∇·u* (velocity predictor divergence)
    Sp2           : (N,) S_p^(2) coefficient
    dissip_source : (N,) viscous dissipation + thermal conduction source (=0 for Euler)
    lam           : γ̃_m Δt  (effective RK3 time step for sub-step m)
    dx            : cell width
    bc_l, bc_r    : boundary condition type

    Returns
    -------
    p_new : (N,) updated pressure
    """
    N = len(p_old)
    lam2 = lam**2

    rho_e = rho_face[1:]    # right face of each cell
    rho_w = rho_face[:-1]   # left  face of each cell
    rho_e = np.maximum(rho_e, 1e-300)
    rho_w = np.maximum(rho_w, 1e-300)

    # Coefficients of off-diagonals
    ce = lam2 * rho * c2_mix / (rho_e * dx**2)   # coefficient of p_{i+1}
    cw = lam2 * rho * c2_mix / (rho_w * dx**2)   # coefficient of p_{i-1}
    cp = 1.0 + ce + cw                             # main diagonal (before BC)

    # RHS
    rhs = p_old + lam * rho * c2_mix * (Sp2 * dissip_source - div_u_star)

    # ── Boundary condition adjustments ─────────────────────────────
    if bc_l == 'transmissive':
        # Ghost p_0 = p[0] → left off-diagonal term = -cw[0]*p_ghost = -cw[0]*p[0]
        # moves to diagonal: cp[0] -= cw[0]
        cp[0] -= cw[0]
    if bc_r == 'transmissive':
        cp[-1] -= ce[-1]

    # ── Solve ──────────────────────────────────────────────────────
    if bc_l == 'periodic' and bc_r == 'periodic':
        # Dense system (small N acceptable for 1D)
        A = (np.diag(cp)
             - np.diag(ce[:-1], 1)
             - np.diag(cw[1:], -1))
        A[0, -1]  = -cw[0]    # periodic left wrap
        A[-1, 0]  = -ce[-1]   # periodic right wrap
        p_new = np.linalg.solve(A, rhs)
    else:
        # Banded (tridiagonal): ab[0]=upper, ab[1]=diag, ab[2]=lower
        ab = np.zeros((3, N))
        ab[1, :]  =  cp
        ab[0, 1:] = -ce[:-1]    # upper: p_{i+1} coefficients stored at column i+1
        ab[2, :-1]= -cw[1:]     # lower: p_{i-1} coefficients stored at column i-1
        p_new = solve_banded((1, 1), ab, rhs)

    return np.maximum(p_new, 1.0)   # pressure floor [Pa]
