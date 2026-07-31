from __future__ import annotations

import math

import pytest

from nondimensional import (
    charge_relaxation_time,
    electric_capillary_number,
    electric_reynolds_number,
    flow_rate_parameter,
    ohnesorge_number,
)


def test_electric_capillary_number_scales_with_field_squared() -> None:
    base = electric_capillary_number(2.0, 3.0, 4.0, 5.0)
    doubled = electric_capillary_number(2.0, 6.0, 4.0, 5.0)
    assert base == pytest.approx(2.0 * 9.0 * 4.0 / 5.0)
    assert doubled == pytest.approx(4.0 * base)


def test_charge_relaxation_and_electric_reynolds_number() -> None:
    tau = charge_relaxation_time(8.0e-10, 2.0e-8)
    assert tau == pytest.approx(0.04)
    assert electric_reynolds_number(8.0e-10, velocity=0.5, length=0.01, conductivity=2.0e-8) == pytest.approx(2.0)


def test_ohnesorge_number_matches_capillary_viscous_scale() -> None:
    assert ohnesorge_number(0.01, 1000.0, 0.05, 1.0e-4) == pytest.approx(0.01 / math.sqrt(1000.0 * 0.05 * 1.0e-4))


def test_flow_rate_parameter_uses_capillary_inertial_scale() -> None:
    q = 1.0e-10
    radius = 20.0e-6
    gamma = 0.05
    rho = 1200.0
    expected = q / math.sqrt(gamma * radius**5 / rho)
    assert flow_rate_parameter(q, radius, gamma, rho) == pytest.approx(expected)
