"""FD Jacobian stencil consistency for the implicit pressure operator."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.primitive import prim_to_cons_W
from solver.five_eq_IMEX.jacobian import assemble_jacobian_fd


def _state(n):
    x = np.arange(n)
    return (
        0.45 + 0.05 * np.sin(2.0 * np.pi * x / n),
        np.full(n, 300.0),
        np.full(n, 310.0),
        np.full(n, 0.03),
        1.0e5 + 100.0 * np.sin(2.0 * np.pi * x / n),
    )


def _row_cell_strengths(J, n, col):
    dense = J[:, col].toarray().ravel().reshape(n, 5)
    return np.max(np.abs(dense), axis=1)


def test_fd_jacobian_stencil_width_tracks_biharmonic():
    eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
    eos2 = make_eos("ideal", gamma=1.67, kv=3120.0)
    n = 12
    dx = 1.0 / n
    W = _state(n)
    U, _ = prim_to_cons_W(W, eos1, eos2)
    zeros = tuple(np.zeros(n) for _ in range(5))
    ci = 6
    col_p = 5 * ci + 4

    J_narrow = assemble_jacobian_fd(
        W, U, 1e-3, zeros, eos1, eos2, dx, "periodic", "periodic",
        imp_dissipation=0.0,
    )
    s_narrow = _row_cell_strengths(J_narrow, n, col_p)
    outside_narrow = [i for i in range(n) if abs(i - ci) > 1]
    assert float(np.max(s_narrow[outside_narrow])) == 0.0

    J_wide = assemble_jacobian_fd(
        W, U, 1e-3, zeros, eos1, eos2, dx, "periodic", "periodic",
        imp_dissipation=0.5,
    )
    s_wide = _row_cell_strengths(J_wide, n, col_p)
    assert s_wide[ci - 2] > 1e-8
    assert s_wide[ci + 2] > 1e-8


if __name__ == "__main__":
    test_fd_jacobian_stencil_width_tracks_biharmonic()
    print("test_jacobian_stencil_consistency: PASS")
