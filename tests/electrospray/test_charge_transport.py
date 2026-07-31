from __future__ import annotations

import numpy as np
import pytest

from charge_transport import (
    advect_charge_upwind_1d,
    advect_confined_charge_1d,
    advect_charge_with_confined_flux_1d,
    confine_charge_to_liquid,
    confined_face_charge_flux_1d,
    gas_charge_leakage_fraction,
)


def test_charge_projection_removes_gas_leakage_and_conserves_total() -> None:
    alpha = np.array([1.0, 0.8, 0.2, 0.0, 0.0])
    charge = np.array([2.0, 1.0, 0.5, 0.25, 0.25])
    dx = 0.1
    projected = confine_charge_to_liquid(alpha, charge, cell_volume=dx)

    assert np.sum(projected * dx) == pytest.approx(np.sum(charge * dx), abs=1.0e-15)
    assert np.all(projected[alpha <= 1.0e-12] == 0.0)
    assert np.all(projected[alpha > 1.0e-12] >= 0.0)


def test_periodic_upwind_charge_advection_is_conservative() -> None:
    charge = np.array([0.0, 1.0, 2.0, 0.5])
    velocity_faces = np.ones(charge.size + 1) * 0.3
    dx = 0.25
    dt = 0.1
    updated = advect_charge_upwind_1d(charge, velocity_faces, dx, dt)

    assert np.sum(updated * dx) == pytest.approx(np.sum(charge * dx), abs=1.0e-15)
    assert np.all(updated >= -1.0e-15)


def test_confined_charge_advection_prevents_pure_gas_charge_leakage() -> None:
    alpha = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    charge = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    velocity_faces = np.ones(charge.size + 1) * 0.4
    dx = 0.2
    dt = 0.1

    updated = advect_confined_charge_1d(alpha, charge, velocity_faces, dx, dt)
    assert np.sum(updated * dx) == pytest.approx(np.sum(charge * dx), abs=1.0e-15)
    assert np.all(updated[alpha == 0.0] == 0.0)
    assert np.all(updated[alpha > 0.0] >= 0.0)
    assert gas_charge_leakage_fraction(alpha, updated, cell_volume=dx) == 0.0


def test_gas_charge_leakage_fraction_reports_charge_in_pure_gas_cells() -> None:
    alpha = np.array([1.0, 0.5, 0.0, 0.0])
    charge = np.array([2.0, -1.0, 0.25, -0.25])

    assert gas_charge_leakage_fraction(alpha, charge, cell_volume=1.0) == pytest.approx(0.5 / 3.5)


def test_confined_face_flux_blocks_pure_gas_interfaces() -> None:
    alpha = np.array([1.0, 1.0, 0.0, 0.0])
    charge = np.array([2.0, 3.0, 100.0, 100.0])
    velocity_faces = np.ones(charge.size + 1)

    flux = confined_face_charge_flux_1d(alpha, charge, velocity_faces)

    assert flux[1] == pytest.approx(2.0)
    assert flux[2] == 0.0
    assert flux[3] == 0.0


def test_confined_face_flux_uses_upwind_liquid_donor_for_negative_velocity() -> None:
    alpha = np.array([0.0, 1.0, 1.0])
    charge = np.array([100.0, 4.0, 7.0])
    velocity_faces = -np.ones(charge.size + 1)

    flux = confined_face_charge_flux_1d(alpha, charge, velocity_faces)

    assert flux[1] == 0.0
    assert flux[2] == pytest.approx(-7.0)


def test_confined_flux_update_is_conservative_on_periodic_domain() -> None:
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    charge = np.array([0.0, 1.0, 2.0, 0.5])
    velocity_faces = np.ones(charge.size + 1) * 0.2
    dx = 0.25
    dt = 0.1

    updated = advect_charge_with_confined_flux_1d(alpha, charge, velocity_faces, dx, dt)

    assert np.sum(updated * dx) == pytest.approx(np.sum(charge * dx), abs=1.0e-15)
    assert np.all(updated >= -1.0e-15)


def test_confined_flux_update_does_not_pull_charge_from_gas_donor() -> None:
    alpha = np.array([0.0, 1.0, 1.0])
    charge = np.array([100.0, 0.0, 0.0])
    velocity_faces = np.ones(charge.size + 1) * 0.2

    updated = advect_charge_with_confined_flux_1d(alpha, charge, velocity_faces, dx=1.0, dt=0.5)

    np.testing.assert_allclose(updated, charge)
