"""Small linear solvers for 1D banded systems."""
from __future__ import annotations

import numpy as np


def solve_tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system with Thomas algorithm.

    Parameters
    ----------
    lower : (N-1,) array
        Sub-diagonal entries a_i = A[i, i-1], i=1..N-1.
    diag : (N,) array
        Diagonal entries b_i = A[i, i].
    upper : (N-1,) array
        Super-diagonal entries c_i = A[i, i+1], i=0..N-2.
    rhs : (N,) array
        Right-hand side.
    """
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = diag.size
    if n == 0:
        return np.empty(0, dtype=float)
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("Invalid tridiagonal dimensions.")

    c = upper.copy()
    d = rhs.copy()
    b = diag.copy()

    for i in range(1, n):
        piv = b[i - 1]
        if abs(piv) < 1e-30:
            raise np.linalg.LinAlgError("Zero pivot in tridiagonal solve.")
        w = lower[i - 1] / piv
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    x = np.empty(n, dtype=float)
    piv = b[-1]
    if abs(piv) < 1e-30:
        raise np.linalg.LinAlgError("Zero pivot in tridiagonal solve.")
    x[-1] = d[-1] / piv
    for i in range(n - 2, -1, -1):
        piv = b[i]
        if abs(piv) < 1e-30:
            raise np.linalg.LinAlgError("Zero pivot in tridiagonal solve.")
        x[i] = (d[i] - c[i] * x[i + 1]) / piv
    return x


def solve_periodic_tridiag(lower, diag, upper, rhs, *, corner_lu, corner_ul):
    """Solve periodic (cyclic) tridiagonal system with Woodbury reduction.

    System structure:
      A[i, i]   = diag[i]
      A[i, i-1] = lower[i-1], i=1..N-1
      A[i, i+1] = upper[i],   i=0..N-2
      A[0, N-1] = corner_lu
      A[N-1, 0] = corner_ul
    """
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    n = diag.size
    if n < 2:
        raise ValueError("Periodic tridiagonal requires N >= 2.")
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("Invalid periodic tridiagonal dimensions.")

    # A = T + U V^T, where T is non-cyclic tridiagonal and U,V are rank-2.
    # U = [e0, eN],  V^T rows place corner terms at opposite ends.
    x0 = solve_tridiag(lower, diag, upper, rhs)

    e0 = np.zeros(n, dtype=float); e0[0] = 1.0
    en = np.zeros(n, dtype=float); en[-1] = 1.0
    y0 = solve_tridiag(lower, diag, upper, e0)
    y1 = solve_tridiag(lower, diag, upper, en)
    Y = np.column_stack((y0, y1))            # T^{-1} U

    VT_x0 = np.array([corner_lu * x0[-1], corner_ul * x0[0]], dtype=float)
    M = np.array([
        [1.0 + corner_lu * Y[-1, 0], corner_lu * Y[-1, 1]],
        [corner_ul * Y[0, 0],        1.0 + corner_ul * Y[0, 1]],
    ], dtype=float)
    z = np.linalg.solve(M, VT_x0)
    return x0 - Y @ z
