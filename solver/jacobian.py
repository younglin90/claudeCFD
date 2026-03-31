# Ref: CLAUDE.md § Jacobian 계산 (Backward Euler용)
"""
Numerical Jacobian computation via finite differences.

Phase 1 implementation (as per CLAUDE.md):
    dF/dU ≈ [F(U + ε·e_j) - F(U)] / ε,  ε = 1e-7 * |U_j|

This is the forward-difference (one-sided) approximation for the flux Jacobian,
used in the Newton iteration of the Backward Euler time integrator.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def numerical_jacobian(
    U: np.ndarray,
    flux_fn: Callable[[np.ndarray], np.ndarray],
    eps_rel: float = 1e-7,
    eps_abs: float = 1e-12,
) -> np.ndarray:
    """
    Compute Jacobian of flux_fn with respect to U using forward finite differences.

    J[i, j] = (flux_fn(U + eps_j * e_j)[i] - flux_fn(U)[i]) / eps_j

    where eps_j = eps_rel * |U[j]| + eps_abs  (component-wise scaling)

    Parameters
    ----------
    U : np.ndarray, shape (n,)
        State vector at which Jacobian is evaluated.
    flux_fn : callable
        Function that maps state vector to flux/residual vector of same shape.
        Signature: flux_fn(U: np.ndarray) -> np.ndarray
    eps_rel : float
        Relative perturbation factor. Default 1e-7 (per CLAUDE.md).
    eps_abs : float
        Absolute floor for perturbation to avoid division by zero.

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Numerical Jacobian matrix.
    """
    n = len(U)
    F0 = flux_fn(U)
    J = np.empty((n, n))

    for j in range(n):
        eps_j = eps_rel * abs(U[j]) + eps_abs
        U_pert = U.copy()
        U_pert[j] += eps_j
        F_pert = flux_fn(U_pert)
        J[:, j] = (F_pert - F0) / eps_j

    return J


def system_jacobian(
    U_cells: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    n_vars: int,
    eps_rel: float = 1e-7,
    eps_abs: float = 1e-12,
) -> np.ndarray:
    """
    Compute full system Jacobian for a 1D cell array.

    For the Backward Euler Newton iteration:
        R(U) = U^{n+1} - U^n + dt * L(U^{n+1}) = 0
    the system Jacobian is J = dR/dU.

    Parameters
    ----------
    U_cells : np.ndarray, shape (N_cells * n_vars,)
        Flattened conservative state vector for all cells.
    residual_fn : callable
        Signature: residual_fn(U_flat: np.ndarray) -> np.ndarray
        Returns residual R = U - U_old + dt * L(U), shape (N_cells * n_vars,).
    n_vars : int
        Number of variables per cell.
    eps_rel : float
        Relative perturbation factor.
    eps_abs : float
        Absolute floor.

    Returns
    -------
    J : np.ndarray, shape (N_total, N_total)
        Full system Jacobian (sparse in practice, dense here for Phase 1).
    """
    return numerical_jacobian(U_cells, residual_fn, eps_rel=eps_rel, eps_abs=eps_abs)
