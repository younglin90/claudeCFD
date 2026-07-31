"""Single-phase reflective acoustic smoke for boundary sign sanity."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.main import solve


def test_single_phase_acoustic_reflective():
    eos = make_eos("ideal", gamma=1.4, kv=717.5)
    n = 100
    L = 1.0
    dx = L / n
    x = (np.arange(n) + 0.5) * dx

    p0 = 1.0e5
    T0 = 300.0
    rho0 = float(eos.density(np.array([p0]), np.array([T0]))[0])
    c0 = float(np.sqrt(1.4 * p0 / rho0))
    amp_p = 5.0
    sigma = 0.035
    x0 = 0.15

    g = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    p_prime = amp_p * g
    # Left-going wave.  Reflection should keep pressure sign and flip velocity.
    u_prime = -p_prime / (rho0 * c0)
    T_prime = T0 * (1.4 - 1.0) / 1.4 * (p_prime / p0)
    W0 = (
        np.full(n, 1.0 - 1.0e-3),
        np.full(n, T0) + T_prime,
        np.full(n, T0) + T_prime,
        u_prime,
        np.full(n, p0) + p_prime,
    )

    t_end = 6.0e-4
    out = solve(
        eos, eos, W0, dx, t_end,
        bc_l="reflective", bc_r="transmissive",
        cfl=0.2, time_integrator="be1",
        pe_project_explicit=False,
        explicit_force_lo=True,
        imp_dissipation=0.02,
        dt_min=1e-12,
        newton_kwargs={"max_iter": 10, "rtol": 1e-7, "atol": 1e-11},
    )
    W = out["W"]
    assert all(np.all(np.isfinite(v)) for v in W)
    assert out.get("terminated_reason") is None

    p_dev = W[4] - p0
    assert float(np.max(p_dev)) > 0.1 * amp_p
    # After reflection from a left wall, the local velocity perturbation near
    # the reflected pressure peak should be right-going (positive).
    i = int(np.argmax(p_dev))
    assert W[3][i] > -1.0e-3


if __name__ == "__main__":
    test_single_phase_acoustic_reflective()
    print("test_single_phase_acoustic_reflective: PASS")

