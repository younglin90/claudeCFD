from __future__ import annotations

import numpy as np
import pytest

from ehd_forces import normal_traction_jump_2d
from force_balance import max_abs_residual, static_normal_stress_residual


def test_static_normal_stress_residual_zero_for_balanced_flat_interface() -> None:
    eps_l = 9.0
    eps_g = 3.0
    e_l = (2.0, 0.0)
    e_g = (6.0, 0.0)
    electric_jump = normal_traction_jump_2d(eps_l, e_l, eps_g, e_g, normal=(1.0, 0.0))
    residual = static_normal_stress_residual(
        pressure_left=electric_jump,
        pressure_right=0.0,
        surface_tension=0.0,
        curvature=0.0,
        epsilon_left=eps_l,
        e_left=e_l,
        epsilon_right=eps_g,
        e_right=e_g,
        normal=(1.0, 0.0),
    )
    assert residual == pytest.approx(0.0)


def test_static_normal_stress_residual_includes_capillary_jump() -> None:
    residual = static_normal_stress_residual(
        pressure_left=12.0,
        pressure_right=2.0,
        surface_tension=0.5,
        curvature=20.0,
        epsilon_left=1.0,
        e_left=(0.0, 0.0),
        epsilon_right=1.0,
        e_right=(0.0, 0.0),
        normal=(1.0, 0.0),
    )
    assert residual == pytest.approx(0.0)


def test_max_abs_residual_reports_balance_error_norm() -> None:
    assert max_abs_residual(np.array([-1.0e-9, 3.0e-8, -2.0e-8])) == pytest.approx(3.0e-8)
