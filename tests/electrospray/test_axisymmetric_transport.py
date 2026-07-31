import numpy as np
import pytest

from axisymmetric_transport import (
    advect_scalar_upwind_axisymmetric,
    axisymmetric_open_axial_boundary_flux_rate,
    axisymmetric_scalar_content,
    cell_velocity_to_axisymmetric_faces,
    conservative_bound_axisymmetric_scalar,
)


def test_axisymmetric_no_through_transport_conserves_content_and_bounds() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    rr, zz = np.meshgrid(r_centers, z_centers)
    scalar = np.clip(np.exp(-40.0 * ((rr - 0.35) ** 2 + (zz - 1.0) ** 2)), 0.0, 1.0)
    velocity_r = 0.03 * rr * (1.0 - rr) * np.sin(np.pi * zz / z_faces[-1])
    velocity_z = 0.02 * np.sin(np.pi * rr) * np.sin(np.pi * zz / z_faces[-1])
    radial_faces, axial_faces = cell_velocity_to_axisymmetric_faces(velocity_r, velocity_z)

    before = axisymmetric_scalar_content(scalar, r_faces, z_faces)
    updated = advect_scalar_upwind_axisymmetric(
        scalar,
        radial_faces,
        axial_faces,
        r_faces,
        z_faces,
        dt=0.01,
        axial_boundary="no_through",
        clip_bounds=(0.0, 1.0),
    )
    after = axisymmetric_scalar_content(updated, r_faces, z_faces)

    assert after == pytest.approx(before)
    assert np.min(updated) >= 0.0
    assert np.max(updated) <= 1.0


def test_axisymmetric_open_transport_accounts_for_boundary_flux() -> None:
    r_faces = np.linspace(0.0, 1.0, 17)
    z_faces = np.linspace(0.0, 1.0, 33)
    scalar = np.ones((32, 16)) * 0.25
    radial_faces = np.zeros((32, 17))
    axial_faces = np.ones((33, 16)) * 0.04

    before = axisymmetric_scalar_content(scalar, r_faces, z_faces)
    updated = advect_scalar_upwind_axisymmetric(
        scalar,
        radial_faces,
        axial_faces,
        r_faces,
        z_faces,
        dt=0.02,
        axial_boundary="open",
    )
    after = axisymmetric_scalar_content(updated, r_faces, z_faces)

    assert after == pytest.approx(before)
    assert np.all(updated == pytest.approx(scalar))


def test_axisymmetric_open_transport_matches_boundary_flux_loss() -> None:
    r_faces = np.linspace(0.0, 1.0, 17)
    z_faces = np.linspace(0.0, 1.0, 33)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    rr, zz = np.meshgrid(r_centers, z_centers)
    scalar = 0.2 + 0.3 * rr + 0.4 * zz
    radial_faces = np.zeros((32, 17))
    axial_faces = np.ones((33, 16)) * 0.03
    dt = 0.01

    before = axisymmetric_scalar_content(scalar, r_faces, z_faces)
    outflow_rate = axisymmetric_open_axial_boundary_flux_rate(scalar, axial_faces, r_faces, z_faces)
    updated = advect_scalar_upwind_axisymmetric(
        scalar,
        radial_faces,
        axial_faces,
        r_faces,
        z_faces,
        dt=dt,
        axial_boundary="open",
    )
    after = axisymmetric_scalar_content(updated, r_faces, z_faces)

    assert after - before == pytest.approx(-dt * outflow_rate)


def test_axisymmetric_conservative_bound_limiter_preserves_content() -> None:
    r_faces = np.linspace(0.0, 1.0, 17)
    z_faces = np.linspace(0.0, 1.0, 17)
    scalar = np.ones((16, 16)) * 0.5
    scalar[3:6, 3:6] = 1.2
    scalar[9:12, 9:12] = -0.2
    target = axisymmetric_scalar_content(np.clip(scalar, 0.0, 1.0), r_faces, z_faces)

    limited = conservative_bound_axisymmetric_scalar(
        scalar,
        r_faces,
        z_faces,
        lower=0.0,
        upper=1.0,
        target_content=target,
    )

    assert np.min(limited) >= 0.0
    assert np.max(limited) <= 1.0
    assert axisymmetric_scalar_content(limited, r_faces, z_faces) == pytest.approx(target)


def test_axisymmetric_conservative_clip_transport_preserves_closed_content() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    scalar = np.zeros((64, 32))
    scalar[:, :18] = 1.0
    radial_faces = np.zeros((64, 33))
    radial_faces[:, 1:-1] = -0.02
    axial_faces = np.zeros((65, 32))

    before = axisymmetric_scalar_content(scalar, r_faces, z_faces)
    updated = advect_scalar_upwind_axisymmetric(
        scalar,
        radial_faces,
        axial_faces,
        r_faces,
        z_faces,
        dt=0.01,
        axial_boundary="no_through",
        clip_bounds=(0.0, 1.0),
        conserve_clip=True,
    )
    after = axisymmetric_scalar_content(updated, r_faces, z_faces)

    assert np.min(updated) >= 0.0
    assert np.max(updated) <= 1.0
    assert after == pytest.approx(before)
