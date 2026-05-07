"""Resolved stiffened-gas water acoustic propagation smoke test.

This test isolates the water-side acoustic block before the full 07-B
Air-Water interface case.  It intentionally uses a pulse wider than the 07-B
Air-Water Gaussian so that the check validates the EOS/acoustic propagation
path without conflating it with under-resolved interface transmission.
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


def _theta_from_eos(eos, p0: float, T0: float) -> float:
    rho = eos.density(np.array([p0]), np.array([T0]))
    rho_p = eos.drhodp_T(rho, np.array([T0]))
    rho_T = eos.drhodT_p(rho, np.array([T0]))
    e_p = eos.dedp_T(rho, np.array([T0]))
    e_T = eos.dedT_p(rho, np.array([T0]))
    pr2 = p0 / np.maximum(rho * rho, 1e-30)
    return float(((pr2 * rho_p - e_p) / (e_T - pr2 * rho_T))[0])


def test_resolved_single_phase_water_acoustic_propagates():
    p0 = 1.0e5
    gamma = 4.4
    pinf = 6.0e8
    kv = 1816.0
    rho_ref = 1000.0
    T0 = (p0 + pinf) / ((gamma - 1.0) * kv * rho_ref)
    eos = make_eos("sg", gamma=gamma, pinf=pinf, kv=kv)

    n = 50
    length = 1.5
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    rho0 = float(eos.density(np.array([p0]), np.array([T0]))[0])
    c0 = float(np.sqrt(gamma * (p0 + pinf) / rho0))
    u_peak = 2.0e-4
    dp_peak = rho0 * c0 * u_peak
    sigma = 0.04
    x0 = 0.3
    t_end = 2.0e-4

    g = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    p_prime = dp_peak * g
    u_prime = p_prime / (rho0 * c0)
    T_prime = _theta_from_eos(eos, p0, T0) * p_prime
    W0 = (
        np.full(n, 1.0 - 1.0e-8),
        np.full(n, T0) + T_prime,
        np.full(n, T0) + T_prime,
        u_prime,
        np.full(n, p0) + p_prime,
    )

    out = solve(
        eos, eos, W0, dx, t_end,
        bc_l="transmissive", bc_r="transmissive",
        cfl=0.2, time_integrator="be1", schur=False,
        pe_project_explicit=False, explicit_force_lo=True,
        imp_dissipation=0.02, dt_min=1e-10, max_steps=2000,
        pure_branch=True, alpha_pure_tol=1e-8,
        newton_kwargs={"max_iter": 12, "rtol": 1e-7, "atol": 1e-11},
    )

    W = out["W"]
    assert out.get("terminated_reason") is None
    assert all(np.all(np.isfinite(c)) for c in W)
    p_num = W[4] - p0
    x_peak = float(x[np.argmax(p_num)])
    x_expected = x0 + c0 * out["t_final"]
    assert abs(x_peak - x_expected) < 0.08
    assert float(np.max(p_num)) > 0.25 * dp_peak


if __name__ == "__main__":
    test_resolved_single_phase_water_acoustic_propagates()
    print("test_single_phase_acoustic_water: PASS")
