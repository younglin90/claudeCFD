"""Longer single-phase acoustic propagation check.

This catches pressure waves that remain near the source instead of travelling
at the EOS acoustic speed over 07-like times.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.main import solve


def _periodic_dist(x, x0, L):
    d = np.abs(x - x0)
    return np.minimum(d, L - d)


def test_single_phase_acoustic_long_air():
    eos = make_eos("ideal", gamma=1.4, kv=717.5)
    n = 100
    length = 1.5
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    p0 = 1.0e5
    T0 = 300.0
    rho0 = float(eos.density(np.array([p0]), np.array([T0]))[0])
    c0 = float(np.sqrt(1.4 * p0 / rho0))
    u_peak = 0.02
    dp_peak = rho0 * c0 * u_peak
    sigma = 0.014
    x0 = 0.1
    t_end = 1.0e-3

    g = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    p_prime = dp_peak * g
    u_prime = p_prime / (rho0 * c0)
    T_prime = T0 * (1.4 - 1.0) / 1.4 * (p_prime / p0)
    W0 = (
        np.full(n, 1.0 - 1.0e-3),
        np.full(n, T0) + T_prime,
        np.full(n, T0) + T_prime,
        u_prime,
        np.full(n, p0) + p_prime,
    )
    out = solve(
        eos, eos, W0, dx, t_end,
        bc_l="transmissive", bc_r="transmissive",
        cfl=0.1, time_integrator="be1",
        pe_project_explicit=False,
        explicit_force_lo=True,
        imp_dissipation=0.02,
        dt_min=1e-10,
        max_steps=2000,
        newton_kwargs={"max_iter": 10, "rtol": 1e-7, "atol": 1e-11},
    )
    W = out["W"]
    assert out.get("terminated_reason") is None
    assert all(np.all(np.isfinite(c)) for c in W)
    x_peak = float(x[np.argmax(W[4] - p0)])
    expected = x0 + c0 * t_end
    assert abs(x_peak - expected) < 0.15
    assert float(np.max(W[4] - p0)) > 0.05 * dp_peak


if __name__ == "__main__":
    test_single_phase_acoustic_long_air()
    print("test_single_phase_acoustic_long: PASS")
