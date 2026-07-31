from __future__ import annotations

import math

import numpy as np
import pytest

from axisymmetric_grid import annular_cell_volumes, axial_face_areas, axisymmetric_divergence, radial_face_areas


def test_annular_cell_volumes_sum_to_cylinder_volume() -> None:
    r_faces = np.linspace(0.0, 0.2, 5)
    z_faces = np.linspace(-0.5, 0.5, 7)
    volumes = annular_cell_volumes(r_faces, z_faces)
    assert np.sum(volumes) == pytest.approx(math.pi * 0.2**2 * 1.0)


def test_axisymmetric_face_area_shapes_and_axis_area() -> None:
    r_faces = np.array([0.0, 0.1, 0.3])
    z_faces = np.array([0.0, 0.2, 0.5])
    ar = radial_face_areas(r_faces, z_faces)
    az = axial_face_areas(r_faces, z_faces)
    assert ar.shape == (2, 3)
    assert az.shape == (3, 2)
    assert np.all(ar[:, 0] == 0.0)
    assert az[0, 1] == pytest.approx(math.pi * (0.3**2 - 0.1**2))


def test_axisymmetric_divergence_of_uniform_axial_flux_is_zero() -> None:
    r_faces = np.linspace(0.0, 1.0, 6)
    z_faces = np.linspace(0.0, 2.0, 5)
    fr = np.zeros((len(z_faces) - 1, len(r_faces)))
    fz = np.ones((len(z_faces), len(r_faces) - 1)) * 3.0
    div = axisymmetric_divergence(fr, fz, r_faces, z_faces)
    assert np.max(np.abs(div)) < 1.0e-14


def test_axisymmetric_divergence_of_radial_linear_field_is_two_a() -> None:
    r_faces = np.linspace(0.0, 1.0, 11)
    z_faces = np.linspace(0.0, 0.5, 4)
    a = 2.5
    fr = a * r_faces[np.newaxis, :] + np.zeros((len(z_faces) - 1, len(r_faces)))
    fz = np.zeros((len(z_faces), len(r_faces) - 1))
    div = axisymmetric_divergence(fr, fz, r_faces, z_faces)
    assert np.max(np.abs(div - 2.0 * a)) < 1.0e-12
