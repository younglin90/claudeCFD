"""Explicit time integrators for the FVM kernel.

Each integrator advances U by Δt using a `rhs(U) → ∂t U` callable that
the solver supplies (rhs encapsulates reconstruction + flux + divergence).

  forward_euler(U, dt, rhs) → U_next
  ssp_rk2     (U, dt, rhs) → U_next   (Heun, two-stage)
  ssp_rk3     (U, dt, rhs) → U_next   (Gottlieb-Shu three-stage)

No free parameters — coefficients are fixed by the SSP property.
"""
from __future__ import annotations
import numpy as np


def forward_euler(U, dt, rhs):
    return U + dt * rhs(U)


def ssp_rk2(U, dt, rhs):
    """Heun's SSP-RK2:
        U^(1)   = U^n + Δt L(U^n)
        U^{n+1} = ½ U^n + ½ (U^(1) + Δt L(U^(1)))
    """
    U1 = U + dt * rhs(U)
    return 0.5 * U + 0.5 * (U1 + dt * rhs(U1))


def ssp_rk3(U, dt, rhs):
    """Gottlieb-Shu SSP-RK3:
        U^(1)   = U^n + Δt L(U^n)
        U^(2)   = ¾ U^n + ¼ (U^(1) + Δt L(U^(1)))
        U^{n+1} = ⅓ U^n + ⅔ (U^(2) + Δt L(U^(2)))
    """
    U1 = U + dt * rhs(U)
    U2 = 0.75 * U + 0.25 * (U1 + dt * rhs(U1))
    return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * rhs(U2))


def get_integrator(name: str):
    table = {
        'euler':       forward_euler,
        'forward_euler': forward_euler,
        'ssp_rk2':     ssp_rk2,
        'heun':        ssp_rk2,
        'ssp_rk3':     ssp_rk3,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"unknown integrator '{name}'; available: {list(table)}")
    return table[name]
