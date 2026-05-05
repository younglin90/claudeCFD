"""Helpers for periodic pressure Helmholtz assembly/solve."""
from __future__ import annotations

import numpy as np

from .linear_solvers import solve_periodic_tridiag

_EPS = 1e-30


def assemble_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx):
    """Assemble periodic Helmholtz matrix coefficients for pressure update.

    Discrete form (cell-centered, periodic):
      (sigma_pp/(gamma_dt)) p_i
      - [gamma_dt/(rho_eff_{i+1/2} dx^2)] (p_{i+1} - p_i)
      + [gamma_dt/(rho_eff_{i-1/2} dx^2)] (p_i - p_{i-1})
    """
    sigma_pp = np.asarray(sigma_pp, dtype=float)
    rho_eff = np.asarray(rho_eff, dtype=float)
    n = sigma_pp.size
    if n < 2:
        raise ValueError("Helmholtz periodic assembly requires N >= 2.")
    if rho_eff.size != n:
        raise ValueError("sigma_pp and rho_eff must have the same size.")
    if gamma_dt <= 0.0 or dx <= 0.0:
        raise ValueError("gamma_dt and dx must be positive.")

    rho_face = 0.5 * (rho_eff + np.roll(rho_eff, -1))
    k_face = gamma_dt / (np.maximum(rho_face, _EPS) * dx * dx)

    diag = sigma_pp / gamma_dt + k_face + np.roll(k_face, 1)
    upper = -k_face[:-1]
    lower = -k_face[:-1]
    corner_lu = -k_face[-1]   # A[0, N-1]
    corner_ul = -k_face[-1]   # A[N-1, 0]
    return lower, diag, upper, float(corner_lu), float(corner_ul)


def solve_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx, rhs):
    """Solve periodic Helmholtz system using the cyclic tridiagonal solver."""
    lower, diag, upper, corner_lu, corner_ul = assemble_helmholtz_periodic(
        sigma_pp, rho_eff, gamma_dt, dx
    )
    return solve_periodic_tridiag(
        lower, diag, upper, np.asarray(rhs, dtype=float),
        corner_lu=corner_lu, corner_ul=corner_ul
    )


def assemble_helmholtz_periodic_legacy(a_pp, rho, gamma_dt, dx):
    """Compatibility wrapper for older call-sites."""
    return assemble_helmholtz_periodic(a_pp, rho, gamma_dt, dx)


def solve_helmholtz_periodic_legacy(a_pp, rho, gamma_dt, dx, rhs):
    """Compatibility wrapper for older call-sites."""
    return solve_helmholtz_periodic(a_pp, rho, gamma_dt, dx, rhs)
