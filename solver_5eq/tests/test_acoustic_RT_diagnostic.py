"""Air-water acoustic R/T diagnostic for the active implicit Riemann state.

This test checks the coefficient that `solver.five_eq_IMEX.residual` actually
uses in the acoustic Riemann face state.  It is intentionally faster and more
localized than the full 07-B nonlinear benchmark: if the interface impedance is
the dominant error, the active R/T coefficients should already be wrong here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.residual import _mixture_impedance
from solver.five_eq_IMEX.sound_speed import phase_sound_speed_sq


P0 = 1.0e5
ALPHA_FLOOR = 1.0e-5


def _temperature(p0: float, gamma: float, kv: float, rho: float, pinf: float = 0.0) -> float:
    return (p0 + pinf) / ((gamma - 1.0) * kv * rho)


def _air_water_state(alpha_floor: float = ALPHA_FLOOR):
    eos_air = make_eos("ideal", gamma=1.4, kv=717.5)
    eos_water = make_eos("sg", gamma=4.1, pinf=4.4e8, kv=474.2)

    T_air = _temperature(P0, 1.4, 717.5, 1.157, 0.0)
    T_water = _temperature(P0, 4.1, 474.2, 998.0, 4.4e8)

    # In 07-B Air-Water, phase 1 is air and phase 2 is water.
    W = (
        np.array([1.0 - alpha_floor, alpha_floor]),
        np.array([T_air, T_air]),
        np.array([T_water, T_water]),
        np.array([0.0, 0.0]),
        np.array([P0, P0]),
    )
    return W, eos_air, eos_water


def _pure_impedances(W, eos_air, eos_water):
    _, T1, T2, _, p = W
    rho_air = eos_air.density(p[:1], T1[:1])
    rho_water = eos_water.density(p[1:], T2[1:])
    c_air_sq = phase_sound_speed_sq(eos_air, rho_air, T1[:1])
    c_water_sq = phase_sound_speed_sq(eos_water, rho_water, T2[1:])
    return float(rho_air[0] * np.sqrt(c_air_sq[0])), float(rho_water[0] * np.sqrt(c_water_sq[0]))


def _coeffs(Z_left: float, Z_right: float):
    """Right-going acoustic pulse from left medium into right medium."""
    R_p = (Z_right - Z_left) / (Z_right + Z_left)
    T_p = 2.0 * Z_right / (Z_left + Z_right)
    R_u = -R_p
    T_u = 2.0 * Z_left / (Z_left + Z_right)
    return R_p, T_p, R_u, T_u


def test_air_water_active_acoustic_impedance_coefficients():
    W, eos_air, eos_water = _air_water_state()
    Z_active = _mixture_impedance(W, eos_air, eos_water)
    Z_air_pure, Z_water_pure = _pure_impedances(W, eos_air, eos_water)

    exact = _coeffs(Z_air_pure, Z_water_pure)
    active = _coeffs(float(Z_active[0]), float(Z_active[1]))

    # The alpha floor lowers the Kapila/Wood water-side impedance slightly, but
    # it does not collapse the air-water reflection coefficient to O(0.5).
    water_impedance_ratio = float(Z_active[1] / Z_water_pure)
    assert 0.90 < water_impedance_ratio < 1.01
    assert abs(active[0] - exact[0]) < 1.0e-3
    assert abs(active[1] - exact[1]) < 1.0e-3
    assert active[0] > 0.998


def main():
    W, eos_air, eos_water = _air_water_state()
    Z_active = _mixture_impedance(W, eos_air, eos_water)
    Z_air_pure, Z_water_pure = _pure_impedances(W, eos_air, eos_water)
    exact = _coeffs(Z_air_pure, Z_water_pure)
    active = _coeffs(float(Z_active[0]), float(Z_active[1]))

    print("Air-Water acoustic R/T diagnostic")
    print("---------------------------------")
    print(f"Z_air_pure      = {Z_air_pure:.6e}")
    print(f"Z_water_pure    = {Z_water_pure:.6e}")
    print(f"Z_air_active    = {float(Z_active[0]):.6e}")
    print(f"Z_water_active  = {float(Z_active[1]):.6e}")
    print(f"water Z ratio   = {float(Z_active[1] / Z_water_pure):.6f}")
    print(f"R_p exact/active= {exact[0]:.8f} / {active[0]:.8f}")
    print(f"T_p exact/active= {exact[1]:.8f} / {active[1]:.8f}")
    print(f"R_u exact/active= {exact[2]:.8f} / {active[2]:.8f}")
    print(f"T_u exact/active= {exact[3]:.8f} / {active[3]:.8f}")
    test_air_water_active_acoustic_impedance_coefficients()
    print("test_acoustic_RT_diagnostic: PASS")


if __name__ == "__main__":
    main()
