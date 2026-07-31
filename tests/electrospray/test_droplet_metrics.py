from __future__ import annotations

import numpy as np
import pytest

from droplet_metrics import (
    axis_extents_from_points,
    circulation_trend_from_charge_amplitude,
    deformation_parameter,
    droplet_covariance_from_alpha,
    droplet_deformation_from_alpha_moments,
    droplet_deformation_from_points,
    ellipse_axes_from_deformation,
    small_deformation_droplet_benchmark,
    small_deformation_parameter,
    surface_charge_trend_amplitude,
    taylor_melcher_deformation_slope,
    taylor_melcher_small_deformation,
    transient_deformation_time_constant,
    transient_taylor_melcher_deformation,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair


def test_deformation_parameter_matches_standard_definition() -> None:
    assert deformation_parameter(3.0, 1.0) == pytest.approx(0.5)
    assert deformation_parameter(2.0, 2.0) == pytest.approx(0.0)


def test_axis_extents_from_interface_points() -> None:
    points = np.array([[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]])
    assert axis_extents_from_points(points) == pytest.approx((4.0, 2.0))


def test_droplet_deformation_from_points_is_orientation_independent() -> None:
    points = np.array([[-1.0, -3.0], [1.0, -3.0], [1.0, 3.0], [-1.0, 3.0]])
    assert droplet_deformation_from_points(points) == pytest.approx(0.5)


def test_droplet_alpha_moments_give_zero_for_symmetric_circle() -> None:
    coords = np.linspace(-1.0, 1.0, 41)
    xx, yy = np.meshgrid(coords, coords)
    alpha = ((xx * xx + yy * yy) <= 0.5**2).astype(float)

    covariance = droplet_covariance_from_alpha(alpha, xx, yy)

    assert covariance[0, 1] == pytest.approx(0.0, abs=1.0e-5)
    assert droplet_deformation_from_alpha_moments(alpha, xx, yy) == pytest.approx(0.0, abs=1.0e-3)


def test_droplet_alpha_moments_track_elliptic_interface_deformation() -> None:
    coords = np.linspace(-1.0, 1.0, 101)
    xx, yy = np.meshgrid(coords, coords)
    alpha = (((xx / 0.6) ** 2 + (yy / 0.3) ** 2) <= 1.0).astype(float)

    deformation = droplet_deformation_from_alpha_moments(alpha, xx, yy)

    assert deformation == pytest.approx((0.6 - 0.3) / (0.6 + 0.3), rel=2.0e-2)


def test_small_deformation_parameter_is_linear_in_electric_capillary_number() -> None:
    assert small_deformation_parameter(0.02, deformation_slope=0.8) == pytest.approx(0.016)


def test_surface_charge_trend_uses_conductivity_minus_permittivity_ratio() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=8.0),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0),
    )

    assert surface_charge_trend_amplitude(pair, electric_field=0.2) == pytest.approx(1.2)
    assert circulation_trend_from_charge_amplitude(1.2) == "poleward"
    assert circulation_trend_from_charge_amplitude(-1.2) == "equatorward"
    assert circulation_trend_from_charge_amplitude(0.0) == "neutral"


def test_small_deformation_droplet_benchmark_combines_ca_e_and_charge_trend() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=8.0),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0),
    )
    benchmark = small_deformation_droplet_benchmark(
        phase_pair=pair,
        permittivity=1.0,
        electric_field=0.2,
        radius=1.0,
        surface_tension=1.0,
        deformation_slope=0.8,
    )

    assert benchmark.electric_capillary_number == pytest.approx(0.04)
    assert benchmark.deformation == pytest.approx(0.032)
    assert benchmark.surface_charge_amplitude == pytest.approx(1.2)
    assert benchmark.circulation_trend == "poleward"


def test_taylor_melcher_small_deformation_matches_reference_formula() -> None:
    slope = taylor_melcher_deformation_slope(conductivity_ratio=8.0, permittivity_ratio=2.0, viscosity_ratio=1.0)
    deformation = taylor_melcher_small_deformation(
        ca_e=0.04,
        conductivity_ratio=8.0,
        permittivity_ratio=2.0,
        viscosity_ratio=1.0,
    )

    assert slope == pytest.approx(0.39375)
    assert deformation == pytest.approx(0.01575)


def test_ellipse_axes_from_deformation_round_trips_standard_metric() -> None:
    major_axis, minor_axis = ellipse_axes_from_deformation(radius=1.0, deformation=0.05)

    assert major_axis * minor_axis == pytest.approx(4.0)
    assert deformation_parameter(major_axis, minor_axis) == pytest.approx(0.05)


def test_transient_taylor_melcher_deformation_approaches_steady_value() -> None:
    tau = transient_deformation_time_constant(radius=1.0e-3, ambient_viscosity=1.0, surface_tension=0.05, viscosity_ratio=1.0)
    steady = 0.01575
    times = tau * np.array([0.0, 1.0, 4.0])
    deformation = transient_taylor_melcher_deformation(times, steady, tau)

    assert tau == pytest.approx(0.04375)
    assert deformation[0] == pytest.approx(0.0)
    assert deformation[1] == pytest.approx(steady * (1.0 - np.exp(-1.0)))
    assert deformation[2] < steady
    assert np.all(np.diff(deformation) > 0.0)
