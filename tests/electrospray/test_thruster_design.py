from __future__ import annotations

import math

import pytest

from propellant import Propellant
from thruster_design import (
    array_current_density,
    extractor_field,
    extractor_open_area_fraction,
    operating_point,
)


def test_extractor_field_uses_voltage_gap_magnitude() -> None:
    assert extractor_field(-1200.0, 2.0e-3) == pytest.approx(6.0e5)


def test_extractor_open_area_fraction_is_bounded_by_pitch_cell() -> None:
    assert extractor_open_area_fraction(10.0e-6, 100.0e-6) == pytest.approx(math.pi * 0.01)
    with pytest.raises(ValueError, match="cannot exceed"):
        extractor_open_area_fraction(80.0e-6, 100.0e-6)


def test_array_current_density_averages_over_emitters_and_apertures() -> None:
    expected = 8.0e-6 / (4 * math.pi * (20.0e-6) ** 2)
    assert array_current_density(8.0e-6, aperture_radius=20.0e-6, emitter_count=4) == pytest.approx(expected)


def test_operating_point_links_propellant_flow_current_and_performance() -> None:
    propellant = Propellant("test", density=1200.0, viscosity=0.02, surface_tension=0.04, conductivity=1.0, permittivity=10.0)
    point = operating_point(propellant, volume_flow_rate=2.0e-12, current=1.2e-6, acceleration_voltage=1000.0)

    assert point.mass_flow_rate == pytest.approx(2.4e-9)
    assert point.charge_to_mass == pytest.approx(500.0)
    assert point.exhaust_velocity == pytest.approx(math.sqrt(1.0e6))
    assert point.thrust == pytest.approx(2.4e-6)
    assert point.specific_impulse == pytest.approx(1000.0 / 9.80665)
    assert point.electrical_power == pytest.approx(1.2e-3)
    assert point.kinetic_power == pytest.approx(1.2e-3)
    assert point.ideal_efficiency == pytest.approx(1.0)
    assert point.thrust_to_power == pytest.approx(2.0e-3)
