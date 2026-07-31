"""Unit test for periodic tridiagonal solver."""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.linear_solvers import solve_periodic_tridiag


def _dense_from_periodic(lower, diag, upper, corner_lu, corner_ul):
    n = len(diag)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = diag[i]
    for i in range(n - 1):
        A[i + 1, i] = lower[i]
        A[i, i + 1] = upper[i]
    A[0, -1] = corner_lu
    A[-1, 0] = corner_ul
    return A


def _assert_rel_close(x, y, rtol=1e-12, atol=1e-12, label=""):
    err = np.max(np.abs(x - y))
    ref = max(np.max(np.abs(y)), 1.0)
    rel = err / ref
    if not (err <= atol or rel <= rtol):
        raise AssertionError(
            f"{label}: max_abs={err:.3e}, max_rel={rel:.3e}, "
            f"rtol={rtol}, atol={atol}"
        )


def test_random_diagonal_dominant():
    rng = np.random.default_rng(7)
    n = 24
    lower = rng.uniform(-0.2, 0.2, size=n - 1)
    upper = rng.uniform(-0.2, 0.2, size=n - 1)
    corner_lu = float(rng.uniform(-0.2, 0.2))
    corner_ul = float(rng.uniform(-0.2, 0.2))
    diag = 2.0 + rng.uniform(0.0, 0.5, size=n)
    rhs = rng.normal(size=n)

    A = _dense_from_periodic(lower, diag, upper, corner_lu, corner_ul)
    x_ref = np.linalg.solve(A, rhs)
    x = solve_periodic_tridiag(lower, diag, upper, rhs,
                               corner_lu=corner_lu, corner_ul=corner_ul)
    _assert_rel_close(x, x_ref, label="random")
    print("  [OK] random diagonal-dominant periodic system")


def test_helmholtz_constant_coeff():
    n = 32
    a = 0.15
    # A = I - a * L_periodic,   L p = p_{i+1} - 2p_i + p_{i-1}
    diag = np.full(n, 1.0 + 2.0 * a)
    lower = np.full(n - 1, -a)
    upper = np.full(n - 1, -a)
    corner_lu = -a
    corner_ul = -a

    i = np.arange(n)
    x_true = np.sin(2.0 * np.pi * i / n) + 0.3 * np.cos(6.0 * np.pi * i / n)
    A = _dense_from_periodic(lower, diag, upper, corner_lu, corner_ul)
    rhs = A @ x_true
    x = solve_periodic_tridiag(lower, diag, upper, rhs,
                               corner_lu=corner_lu, corner_ul=corner_ul)
    _assert_rel_close(x, x_true, label="helmholtz-const")
    print("  [OK] periodic Helmholtz constant-coefficient")


def test_nyquist_rhs():
    n = 20
    a = 0.23
    diag = np.full(n, 1.0 + 2.0 * a)
    lower = np.full(n - 1, -a)
    upper = np.full(n - 1, -a)
    corner_lu = -a
    corner_ul = -a
    rhs = np.array([1.0 if (k % 2 == 0) else -1.0 for k in range(n)], dtype=float)

    A = _dense_from_periodic(lower, diag, upper, corner_lu, corner_ul)
    x_ref = np.linalg.solve(A, rhs)
    x = solve_periodic_tridiag(lower, diag, upper, rhs,
                               corner_lu=corner_lu, corner_ul=corner_ul)
    _assert_rel_close(x, x_ref, label="nyquist")
    print("  [OK] nyquist forcing periodic Helmholtz")


if __name__ == '__main__':
    print("periodic tridiagonal solver tests")
    test_random_diagonal_dominant()
    test_helmholtz_constant_coeff()
    test_nyquist_rhs()
    print("All tests passed.")
