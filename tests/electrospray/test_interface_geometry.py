from __future__ import annotations

import numpy as np

from interface_geometry import (
    axisymmetric_level_set_curvature,
    circle_signed_distance,
    level_set_curvature,
    level_set_normals,
    regularized_heaviside,
)


def test_regularized_heaviside_is_bounded_and_monotone() -> None:
    phi = np.linspace(-2.0, 2.0, 101)
    h = regularized_heaviside(phi, width=0.5)

    assert np.all((0.0 <= h) & (h <= 1.0))
    assert np.all(np.diff(h) <= 1.0e-14)
    assert h[0] == 1.0
    assert h[-1] == 0.0


def test_flat_level_set_normals_and_zero_curvature() -> None:
    nx = 30
    ny = 20
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 0.5, ny)
    xx, _yy = np.meshgrid(x, y)
    phi = xx - 0.4
    normal_x, normal_y = level_set_normals(phi, dx=x[1] - x[0], dy=y[1] - y[0])
    curvature = level_set_curvature(phi, dx=x[1] - x[0], dy=y[1] - y[0])

    assert np.max(np.abs(normal_x - 1.0)) < 1.0e-12
    assert np.max(np.abs(normal_y)) < 1.0e-12
    assert np.max(np.abs(curvature)) < 1.0e-11


def test_circle_signed_distance_curvature_matches_inverse_radius_near_interface() -> None:
    n = 101
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    radius = 0.5
    phi = circle_signed_distance(xx, yy, center=(0.0, 0.0), radius=radius)
    curvature = level_set_curvature(phi, dx=x[1] - x[0], dy=y[1] - y[0])
    band = np.abs(phi) < 0.02

    assert abs(float(np.mean(curvature[band])) - 1.0 / radius) < 0.03


def test_axisymmetric_curvature_adds_cylindrical_radius_term() -> None:
    r = np.linspace(0.2, 1.0, 101)
    z = np.linspace(-0.5, 0.5, 81)
    rr, _zz = np.meshgrid(r, z)
    cylinder_radius = 0.6
    phi = rr - cylinder_radius

    curvature = axisymmetric_level_set_curvature(phi, dr=r[1] - r[0], dz=z[1] - z[0], radius=rr)
    band = np.abs(phi) < 0.01

    assert abs(float(np.mean(curvature[band])) - 1.0 / cylinder_radius) < 0.02


def test_axisymmetric_curvature_of_sphere_matches_two_over_radius_near_interface() -> None:
    r = np.linspace(0.1, 1.0, 121)
    z = np.linspace(-1.0, 1.0, 161)
    rr, zz = np.meshgrid(r, z)
    sphere_radius = 0.5
    phi = np.sqrt(rr * rr + zz * zz) - sphere_radius

    curvature = axisymmetric_level_set_curvature(phi, dr=r[1] - r[0], dz=z[1] - z[0], radius=rr)
    band = np.abs(phi) < 0.01

    assert abs(float(np.mean(curvature[band])) - 2.0 / sphere_radius) < 0.08
