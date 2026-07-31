from __future__ import annotations

import math

import numpy as np
import pytest

from cone_geometry import (
    TaylorConeVoltageRampPoint,
    axisymmetric_cone_curvature,
    balanced_taylor_cone_normal_field,
    balanced_taylor_cone_voltage,
    cone_level_set,
    fit_cone_half_angle,
    maxwell_normal_pressure,
    normalized_taylor_cone_balance_residual,
    taylor_cone_half_angle,
    taylor_cone_level_set_balance_residual,
    taylor_cone_voltage_ramp_balance,
    taylor_cone_static_balance_residual,
)


def test_taylor_cone_half_angle_reference_value() -> None:
    assert taylor_cone_half_angle() == pytest.approx(49.292, abs=1.0e-12)
    assert taylor_cone_half_angle(radians=True) == pytest.approx(math.radians(49.292), abs=1.0e-14)


def test_cone_level_set_is_zero_on_analytic_surface() -> None:
    angle = math.radians(49.292)
    tip_z = 1.0
    z = np.linspace(0.1, 0.9, 20)
    r = (tip_z - z) * math.tan(angle)
    phi = cone_level_set(r, z, tip_z=tip_z, half_angle=angle)
    assert np.max(np.abs(phi)) < 1.0e-14


def test_fit_cone_half_angle_recovers_known_axisymmetric_cone() -> None:
    angle = math.radians(42.0)
    tip_z = 1.5
    z = np.linspace(0.2, 1.2, 30)
    r = (tip_z - z) * math.tan(angle)
    fitted = fit_cone_half_angle(r, z, tip_z=tip_z)
    assert math.degrees(fitted) == pytest.approx(42.0, abs=1.0e-12)


def test_axisymmetric_cone_curvature_matches_radius_relation() -> None:
    angle = math.radians(49.292)
    radius = np.array([0.2, 0.4, 0.8])
    assert axisymmetric_cone_curvature(radius, angle) == pytest.approx(np.cos(angle) / radius)


def test_taylor_cone_static_balance_residual_vanishes_for_balanced_field() -> None:
    angle = math.radians(49.292)
    radius = np.array([0.2, 0.4, 0.8])
    surface_tension = 1.5
    permittivity = 2.0
    curvature = axisymmetric_cone_curvature(radius, angle)
    normal_field = np.sqrt(2.0 * surface_tension * curvature / permittivity)

    assert maxwell_normal_pressure(permittivity, normal_field) == pytest.approx(surface_tension * curvature)
    assert np.max(np.abs(taylor_cone_static_balance_residual(radius, angle, surface_tension, permittivity, normal_field))) < 1.0e-12


def test_balanced_taylor_cone_field_and_voltage_proxy_are_consistent() -> None:
    angle = math.radians(49.292)
    radius = np.array([0.2, 0.4, 0.8])
    surface_tension = 1.5
    permittivity = 2.0
    gap = 0.25

    normal_field = balanced_taylor_cone_normal_field(radius, angle, surface_tension, permittivity)
    voltage = balanced_taylor_cone_voltage(radius, angle, surface_tension, permittivity, extractor_gap=gap)

    assert np.all(np.diff(normal_field) < 0.0)
    assert voltage == pytest.approx(gap * normal_field)
    assert np.max(np.abs(taylor_cone_static_balance_residual(radius, angle, surface_tension, permittivity, normal_field))) < 1.0e-12


def test_taylor_cone_level_set_balance_residual_samples_interface_band() -> None:
    angle = math.radians(49.292)
    tip_z = 1.0
    surface_tension = 0.072
    permittivity = 2.0
    r_centers = np.linspace(0.05, 0.95, 96)
    z_centers = np.linspace(0.05, 0.95, 96)
    rr, zz = np.meshgrid(r_centers, z_centers)
    normal_field = balanced_taylor_cone_normal_field(rr, angle, surface_tension, permittivity)

    residual, mask = taylor_cone_level_set_balance_residual(
        rr,
        zz,
        tip_z=tip_z,
        half_angle=angle,
        surface_tension=surface_tension,
        permittivity=permittivity,
        normal_electric_field=normal_field,
        band_width=0.015,
    )

    assert np.count_nonzero(mask) > 0
    assert np.nanmax(np.abs(residual)) < 1.0e-14
    assert normalized_taylor_cone_balance_residual(
        rr,
        zz,
        tip_z=tip_z,
        half_angle=angle,
        surface_tension=surface_tension,
        permittivity=permittivity,
        normal_electric_field=normal_field,
        band_width=0.015,
    ) < 1.0e-12


def test_taylor_cone_voltage_ramp_reaches_balanced_force_state() -> None:
    points = taylor_cone_voltage_ramp_balance(
        radius=0.2,
        half_angle=math.radians(49.292),
        surface_tension=0.072,
        permittivity=2.0,
        extractor_gap=0.25,
        voltage_fractions=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
    )

    assert all(isinstance(point, TaylorConeVoltageRampPoint) for point in points)
    assert np.all(np.diff([point.normal_electric_field for point in points]) > 0.0)
    assert np.allclose([point.capillary_pressure for point in points], points[-1].capillary_pressure)
    assert points[-1].maxwell_pressure == pytest.approx(points[-1].capillary_pressure)
    assert np.all(np.diff([point.normalized_balance_residual for point in points]) < 0.0)
    assert points[-1].normalized_balance_residual == pytest.approx(0.0, abs=1.0e-14)
