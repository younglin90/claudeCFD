"""Tests for periodic Helmholtz assembly/solve helpers."""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.helmholtz import (
    assemble_helmholtz_periodic,
    solve_helmholtz_periodic,
)


def _dense(lower, diag, upper, corner_lu, corner_ul):
    n = len(diag)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = diag[i]
    for i in range(n - 1):
        A[i, i + 1] = upper[i]
        A[i + 1, i] = lower[i]
    A[0, -1] = corner_lu
    A[-1, 0] = corner_ul
    return A


def _assert_close(name, x, y, rtol=1e-11, atol=1e-11):
    diff = float(np.max(np.abs(x - y)))
    ref = max(float(np.max(np.abs(y))), 1.0)
    rel = diff / ref
    if not (diff <= atol or rel <= rtol):
        raise AssertionError(f"{name}: max_abs={diff:.3e}, max_rel={rel:.3e}")


def test_variable_coefficients():
    n = 18
    x = (np.arange(n) + 0.5) / n
    rho = 0.8 + 0.4 * (1.0 + np.sin(2.0 * np.pi * x))
    a_pp = 2.0 + 0.2 * np.cos(4.0 * np.pi * x)
    gamma_dt = 2.5e-4
    dx = 1.0 / n
    rhs = np.cos(2.0 * np.pi * x) + 0.2 * np.sin(6.0 * np.pi * x)

    lower, diag, upper, c_lu, c_ul = assemble_helmholtz_periodic(
        a_pp, rho, gamma_dt, dx
    )
    A = _dense(lower, diag, upper, c_lu, c_ul)
    x_ref = np.linalg.solve(A, rhs)
    x_num = solve_helmholtz_periodic(a_pp, rho, gamma_dt, dx, rhs)
    _assert_close("variable", x_num, x_ref)
    print("  [OK] variable-coefficient periodic Helmholtz")


def test_nyquist_mode_forcing():
    n = 20
    rho = np.full(n, 1.2)
    a_pp = np.full(n, 2.3)
    gamma_dt = 1.0e-4
    dx = 1.0 / n
    rhs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)], dtype=float)

    lower, diag, upper, c_lu, c_ul = assemble_helmholtz_periodic(
        a_pp, rho, gamma_dt, dx
    )
    A = _dense(lower, diag, upper, c_lu, c_ul)
    x_ref = np.linalg.solve(A, rhs)
    x_num = solve_helmholtz_periodic(a_pp, rho, gamma_dt, dx, rhs)
    _assert_close("nyquist", x_num, x_ref)
    print("  [OK] nyquist forcing periodic Helmholtz")


if __name__ == '__main__':
    print("periodic Helmholtz helper tests")
    test_variable_coefficients()
    test_nyquist_mode_forcing()
    print("All tests passed.")
