# solver/demou2022_1d/spatial.py
"""
Spatial operators on a 1D uniform cell-centred grid.

All operators work on interior-cell arrays (length N).
Ghost-extended arrays (length N + 2*n_ghost) are expected where noted.

Operators
---------
face_velocity(u_ext)          : arithmetic face average
divergence(phi_face, dx)      : ∂(φ)/∂x at cell centres
gradient_cc(phi_ext, dx, ng)  : ∂φ/∂x at cell centres (central diff, 2nd order)
laplacian_helmholtz(...)      : ∇·(1/ρ ∇p) for Helmholtz equation
divergence_S3(phi_ext, Sa3_or_ST3, u_face, dx, ng)  : divergence source term
"""

import numpy as np


def face_velocity(u_ext: np.ndarray, n_ghost: int = 2) -> np.ndarray:
    """
    Arithmetic face velocity: ū_{i+1/2} = (u_i + u_{i+1}) / 2.

    Parameters
    ----------
    u_ext : (N + 2*n_ghost,) ghost-extended cell velocity

    Returns
    -------
    u_face : (N + 1,) face velocities (all interior + boundary faces)
    """
    N = len(u_ext) - 2 * n_ghost
    i = n_ghost  # index of first interior cell in extended array
    return 0.5 * (u_ext[i - 1: i + N] + u_ext[i: i + N + 1])


def divergence(phi_face: np.ndarray, dx: float) -> np.ndarray:
    """
    ∂φ/∂x|_i = (φ_{i+1/2} - φ_{i-1/2}) / Δx.

    Parameters
    ----------
    phi_face : (N + 1,) face values
    dx       : cell width

    Returns
    -------
    div : (N,) cell-centred divergence
    """
    return (phi_face[1:] - phi_face[:-1]) / dx


def gradient_cc(phi_ext: np.ndarray, dx: float, n_ghost: int = 2) -> np.ndarray:
    """
    Cell-centred gradient using central differences.

    ∂φ/∂x|_i = (φ_{i+1} - φ_{i-1}) / (2Δx)

    Parameters
    ----------
    phi_ext : (N + 2*n_ghost,) ghost-extended array

    Returns
    -------
    grad : (N,) cell-centred gradient
    """
    N = len(phi_ext) - 2 * n_ghost
    i = n_ghost
    return (phi_ext[i + 1: i + N + 1] - phi_ext[i - 1: i + N - 1]) / (2.0 * dx)


def face_density_harmonic(rho_ext: np.ndarray, n_ghost: int = 2) -> np.ndarray:
    """
    Harmonic-mean face density: 1/ρ̄ = (1/ρ_i + 1/ρ_{i+1}) / 2.

    Used in Helmholtz Laplacian (MWI-style).

    Returns
    -------
    rho_face : (N + 1,)
    """
    N = len(rho_ext) - 2 * n_ghost
    i = n_ghost
    rL = rho_ext[i - 1: i + N]
    rR = rho_ext[i:     i + N + 1]
    inv_rho = 0.5 * (1.0 / np.maximum(rL, 1e-300) + 1.0 / np.maximum(rR, 1e-300))
    return 1.0 / np.maximum(inv_rho, 1e-300)


def helmholtz_laplacian(p_ext: np.ndarray, rho_face: np.ndarray,
                        dx: float, n_ghost: int = 2) -> np.ndarray:
    """
    Discretised Helmholtz Laplacian: ∇·(1/ρ ∇p).

    = [(p_{i+1}-p_i)/(ρ_{i+1/2}Δx) - (p_i-p_{i-1})/(ρ_{i-1/2}Δx)] / Δx

    Parameters
    ----------
    p_ext    : (N + 2*n_ghost,) ghost-extended pressure
    rho_face : (N + 1,) face densities
    dx       : cell width

    Returns
    -------
    lap : (N,) Laplacian values
    """
    N = len(p_ext) - 2 * n_ghost
    i = n_ghost
    p_L  = p_ext[i - 1: i + N]
    p_C  = p_ext[i:     i + N]
    p_R  = p_ext[i + 1: i + N + 1]

    rho_e = rho_face[1:]    # right face density of each cell
    rho_w = rho_face[:-1]   # left  face density of each cell

    flux_e = (p_R - p_C) / (np.maximum(rho_e, 1e-300) * dx)
    flux_w = (p_C - p_L) / (np.maximum(rho_w, 1e-300) * dx)

    return (flux_e - flux_w) / dx


def advection_divergence_source(phi_ext: np.ndarray,
                                S3: np.ndarray,
                                u_face: np.ndarray,
                                phi_face: np.ndarray,
                                dx: float, n_ghost: int = 2) -> np.ndarray:
    """
    Compute the full RHS divergence terms for α₁ or T equations.

        R_φ = ∂(φ u)/∂x + (S^(3) - φ) ∂u/∂x

    where ∂(φu)/∂x uses face values from CICSAM/van Leer,
    and ∂u/∂x is computed from u_face.

    Parameters
    ----------
    phi_ext   : (N + 2*n_ghost,) ghost-extended φ
    S3        : (N,) S_φ^(3) coefficient (either Sa3 or ST3)
    u_face    : (N + 1,) face velocities
    phi_face  : (N + 1,) CICSAM/van Leer face values of φ
    dx        : cell width

    Returns
    -------
    R_phi : (N,) divergence residual
    """
    N = len(S3)
    i = n_ghost
    phi_C = phi_ext[i: i + N]   # cell-centre φ

    div_phi_u = divergence(phi_face * u_face, dx)   # ∂(φu)/∂x
    div_u     = divergence(u_face, dx)               # ∂u/∂x

    return div_phi_u + (S3 - phi_C) * div_u
