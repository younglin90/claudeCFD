"""Near-pure primitive recovery smoke tests for five_eq_IMEX."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.primitive import prim_to_cons_W, cons_to_prim_W


def test_pure_phase_limit_recovery():
    eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
    eos2 = make_eos("sg", gamma=4.1, pinf=4.4e8, kv=474.2)
    alpha = np.array([1e-12, 1e-9, 1e-6, 1.0 - 1e-6, 1.0 - 1e-9, 1.0 - 1e-12])
    n = alpha.size
    W = (
        alpha,
        np.full(n, 302.0),
        np.full(n, 295.0),
        np.linspace(-0.02, 0.02, n),
        np.full(n, 1.2e5),
    )
    U, _ = prim_to_cons_W(W, eos1, eos2)
    W_rec = cons_to_prim_W(
        U, eos1, eos2,
        T1_init=W[1], T2_init=W[2],
        tol=1e-12, max_iter=80,
        alpha_pure_tol=1e-5,
    )
    assert all(np.all(np.isfinite(c)) for c in W_rec)
    assert np.all(W_rec[1] > 0.0)
    assert np.all(W_rec[2] > 0.0)
    assert np.all(W_rec[4] > 0.0)

    U_rec, _ = prim_to_cons_W(W_rec, eos1, eos2)
    rels = []
    for a, b in zip(U, U_rec):
        rels.append(float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))))
    assert max(rels) < 1e-8


if __name__ == "__main__":
    test_pure_phase_limit_recovery()
    print("test_pure_phase_limit_recovery: PASS")
