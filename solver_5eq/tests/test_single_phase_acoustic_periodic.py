"""Single-phase acoustic periodic smoke for five_eq_IMEX.

This is deliberately a smoke/diagnostic test: it should catch blow-up,
non-finite primitive recovery, or a completely stationary acoustic pulse
before full 07 interface physics is considered.
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


def test_single_phase_acoustic_periodic():
    eos = make_eos("ideal", gamma=1.4, kv=717.5)
    n = 80
    L = 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx

    p0 = 1.0e5
    T0 = 300.0
    rho0 = float(eos.density(np.array([p0]), np.array([T0]))[0])
    c0 = float(np.sqrt(1.4 * p0 / rho0))
    amp_p = 10.0
    sigma = 0.04
    x0 = 0.25

    g = np.exp(-_periodic_dist(x, x0, L) ** 2 / (2.0 * sigma ** 2))
    p_prime = amp_p * g
    u_prime = p_prime / (rho0 * c0)
    T_prime = T0 * (1.4 - 1.0) / 1.4 * (p_prime / p0)

    alpha = np.full(n, 1.0 - 1.0e-3)
    W0 = (
        alpha,
        np.full(n, T0) + T_prime,
        np.full(n, T0) + T_prime,
        u_prime,
        np.full(n, p0) + p_prime,
    )

    t_end = 1.0e-4
    out = solve(
        eos, eos, W0, dx, t_end,
        bc_l="periodic", bc_r="periodic",
        cfl=0.2, time_integrator="be1",
        pe_project_explicit=False,
        explicit_force_lo=True,
        imp_dissipation=0.02,
        dt_min=1e-12,
        newton_kwargs={"max_iter": 10, "rtol": 1e-7, "atol": 1e-11},
    )
    W = out["W"]
    finite = all(np.all(np.isfinite(v)) for v in W)
    assert finite
    assert out.get("terminated_reason") is None

    p_amp_final = float(np.max(np.abs(W[4] - p0)))
    assert 0.05 * amp_p < p_amp_final < 2.5 * amp_p

    x_peak = float(x[np.argmax(W[4] - p0)])
    x_expected = (x0 + c0 * t_end) % L
    err_x = float(_periodic_dist(np.array([x_peak]), x_expected, L)[0])
    assert err_x < 0.15


if __name__ == "__main__":
    test_single_phase_acoustic_periodic()
    print("test_single_phase_acoustic_periodic: PASS")

