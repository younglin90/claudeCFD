from __future__ import annotations

import math

import pytest

from microthruster import (
    electrical_power,
    ideal_exhaust_velocity,
    ideal_power_efficiency,
    kinetic_power,
    specific_charge,
    specific_impulse,
    specific_impulse_from_thrust,
    thrust_from_mass_flow,
    thrust_to_power,
)


def test_specific_charge_is_current_over_mass_flow() -> None:
    assert specific_charge(current=2.0e-6, mass_flow_rate=5.0e-10) == pytest.approx(4000.0)


def test_ideal_exhaust_velocity_and_isp_follow_electrostatic_scaling() -> None:
    xi = 4.0e5
    voltage = 1200.0
    velocity = ideal_exhaust_velocity(xi, voltage)

    assert velocity == pytest.approx(math.sqrt(2.0 * xi * voltage))
    assert specific_impulse(velocity) == pytest.approx(velocity / 9.80665)


def test_thrust_from_mass_flow_uses_momentum_flux() -> None:
    assert thrust_from_mass_flow(2.5e-9, 15000.0) == pytest.approx(3.75e-5)


def test_power_efficiency_links_beam_power_to_kinetic_power() -> None:
    current = 1.2e-6
    voltage = 1000.0
    mdot = 2.4e-9
    velocity = ideal_exhaust_velocity(specific_charge(current, mdot), voltage)

    assert electrical_power(current, voltage) == pytest.approx(1.2e-3)
    assert kinetic_power(mdot, velocity) == pytest.approx(electrical_power(current, voltage))
    assert ideal_power_efficiency(mdot, velocity, current, voltage) == pytest.approx(1.0)


def test_thrust_to_power_reports_performance_ratio() -> None:
    assert thrust_to_power(2.4e-6, 1.2e-3) == pytest.approx(2.0e-3)
    with pytest.raises(ValueError, match="power"):
        thrust_to_power(1.0e-6, 0.0)


def test_specific_impulse_from_thrust_uses_momentum_definition() -> None:
    assert specific_impulse_from_thrust(1.92e-6, 2.4e-9) == pytest.approx(800.0 / 9.80665)
    with pytest.raises(ValueError, match="mass_flow_rate"):
        specific_impulse_from_thrust(1.0e-6, 0.0)
