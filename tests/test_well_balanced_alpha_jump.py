"""Pressure-equilibrium alpha-jump regression for five_eq_IMEX."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.main import solve


def _alpha_jump_state(n, u0):
    alpha = np.full(n, 1e-3)
    alpha[: n // 2] = 1.0 - 1e-3
    return (
        alpha,
        np.full(n, 300.0),
        np.full(n, 300.0),
        np.full(n, u0),
        np.full(n, 1.0e5),
    )


def _run(W0, dx, t_end, dt):
    eos1 = make_eos("ideal", gamma=1.4, kv=717.5)
    eos2 = make_eos("nasg", gamma=1.187, pinf=7.028e8, kv=3610.0,
                    b=6.61e-4, eta=-1.177788e6)
    return solve(
        eos1, eos2, W0, dx, t_end,
        bc_l="periodic", bc_r="periodic",
        dt_fixed=dt,
        time_integrator="be1",
        imp_dissipation=0.02,
        pe_project_explicit=True,
        pe_projection_mode="contact",
        explicit_force_lo=True,
        dt_min=1e-14,
        max_steps=10000,
        newton_kwargs={"max_iter": 10, "rtol": 1e-7, "atol": 1e-11},
    )


def test_stationary_alpha_jump_one_step():
    n = 20
    W0 = _alpha_jump_state(n, u0=0.0)
    out = _run(W0, dx=1.0 / n, t_end=1e-3, dt=1e-3)
    W = out["W"]
    assert out.get("terminated_reason") is None
    assert all(np.all(np.isfinite(c)) for c in W)
    assert float(np.max(np.abs(W[4] - W0[4]) / W0[4])) < 1e-10
    assert float(np.max(np.abs(W[3] - W0[3]))) < 1e-10


def test_stationary_alpha_jump_many_step_smoke():
    n = 20
    W0 = _alpha_jump_state(n, u0=0.0)
    out = _run(W0, dx=1.0 / n, t_end=2e-2, dt=1e-3)
    W = out["W"]
    assert out.get("terminated_reason") is None
    assert all(np.all(np.isfinite(c)) for c in W)
    assert float(np.max(np.abs(W[4] - 1.0e5) / 1.0e5)) < 1e-6
    assert float(np.max(np.abs(W[3]))) < 1e-8


if __name__ == "__main__":
    test_stationary_alpha_jump_one_step()
    test_stationary_alpha_jump_many_step_smoke()
    print("test_well_balanced_alpha_jump: PASS")
