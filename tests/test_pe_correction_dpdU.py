"""Finite-difference check for dp/dU rows used by PE projection."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.primitive import prim_to_cons_W, cons_to_prim_W
from solver.five_eq_IMEX.pe_correction import dpdU


def test_dpdU_matches_recovered_pressure_fd():
    eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
    eos2 = make_eos("sg", gamma=4.1, pinf=4.4e8, kv=474.2)
    alpha = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    n = alpha.size
    W = (
        alpha,
        np.linspace(290.0, 330.0, n),
        np.linspace(285.0, 310.0, n),
        np.linspace(-0.1, 0.1, n),
        np.linspace(0.9e5, 1.4e5, n),
    )
    U, _ = prim_to_cons_W(W, eos1, eos2)
    analytic = dpdU(W, eos1, eos2)

    for comp in range(5):
        scale = np.maximum(np.abs(U[comp]), 1.0)
        if comp == 4:
            h = np.full(n, 1e-4)
        else:
            # Conservative-to-primitive Newton tolerances swamp tiny pressure
            # changes; use a still-linear but resolvable FD step.
            h = 1e-4 * scale
        Up = [np.asarray(c).copy() for c in U]
        Um = [np.asarray(c).copy() for c in U]
        Up[comp] = Up[comp] + h
        Um[comp] = Um[comp] - h
        if comp == 4:
            Up[comp] = np.clip(Up[comp], 1e-6, 1.0 - 1e-6)
            Um[comp] = np.clip(Um[comp], 1e-6, 1.0 - 1e-6)
            h_eff = 0.5 * (Up[comp] - Um[comp])
        else:
            h_eff = h
        Wp = cons_to_prim_W(tuple(Up), eos1, eos2, T1_init=W[1], T2_init=W[2],
                            tol=1e-13, max_iter=100)
        Wm = cons_to_prim_W(tuple(Um), eos1, eos2, T1_init=W[1], T2_init=W[2],
                            tol=1e-13, max_iter=100)
        fd = (Wp[4] - Wm[4]) / (2.0 * h_eff)
        denom = np.maximum(np.maximum(np.abs(fd), np.abs(analytic[comp])), 1.0)
        err = np.abs(fd - analytic[comp]) / denom
        assert float(np.max(err)) < 1e-3


def test_dpdU_near_pure_is_finite():
    eos = make_eos("ideal", gamma=1.4, kv=717.5)
    alpha = np.array([1e-9, 1.0 - 1e-9])
    W = (
        alpha,
        np.full(2, 300.0),
        np.full(2, 300.0),
        np.zeros(2),
        np.full(2, 1e5),
    )
    rows = dpdU(W, eos, eos)
    assert np.all(np.isfinite(rows))


if __name__ == "__main__":
    test_dpdU_matches_recovered_pressure_fd()
    test_dpdU_near_pure_is_finite()
    print("test_pe_correction_dpdU: PASS")
