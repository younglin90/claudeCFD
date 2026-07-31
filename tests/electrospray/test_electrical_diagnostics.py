from __future__ import annotations

import numpy as np
import pytest

from electrical_diagnostics import (
    charge_conservation_residual,
    electrostatic_energy_density,
    electrostatic_energy_density_material,
    electrostatic_energy_density_phase_pair,
    ohmic_current_density,
    ohmic_current_density_material,
    ohmic_current_density_phase_pair,
    total_electrostatic_energy,
    total_electrostatic_energy_material,
    total_electrostatic_energy_phase_pair,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair


def test_ohmic_current_density_is_sigma_times_electric_field() -> None:
    sigma = np.array([2.0, 0.5])
    ex = np.array([3.0, -4.0])
    ey = np.array([1.0, 2.0])
    jx, jy = ohmic_current_density(sigma, ex, ey)
    np.testing.assert_allclose(jx, np.array([6.0, -2.0]))
    np.testing.assert_allclose(jy, np.array([2.0, 1.0]))


def test_ohmic_current_density_material_uses_validated_conductivity() -> None:
    material = LeakyDielectricMaterial(permittivity=2.0, conductivity=0.5)
    ex = np.array([3.0, -4.0])
    ey = np.array([1.0, 2.0])

    jx, jy = ohmic_current_density_material(material, ex, ey)

    np.testing.assert_allclose(jx, np.array([1.5, -2.0]))
    np.testing.assert_allclose(jy, np.array([0.5, 1.0]))


def test_ohmic_current_density_phase_pair_uses_mixed_conductivity() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0),
    )
    alpha = np.array([0.0, 0.5, 1.0])

    jx, jy = ohmic_current_density_phase_pair(pair, alpha, np.array([2.0, 2.0, 2.0]), np.ones(3))

    np.testing.assert_allclose(jx, np.array([2.0, 5.0, 8.0]))
    np.testing.assert_allclose(jy, np.array([1.0, 2.5, 4.0]))


def test_electrostatic_energy_density_matches_half_epsilon_e_squared() -> None:
    density = electrostatic_energy_density(4.0, np.array([3.0]), np.array([4.0]))
    assert density[0] == pytest.approx(50.0)


def test_electrostatic_energy_density_material_uses_validated_permittivity() -> None:
    material = LeakyDielectricMaterial(permittivity=4.0, conductivity=0.5)
    density = electrostatic_energy_density_material(material, np.array([3.0]), np.array([4.0]))

    assert density[0] == pytest.approx(50.0)


def test_electrostatic_energy_density_phase_pair_uses_mixed_permittivity() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=1.0),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=0.1),
    )
    alpha = np.array([0.0, 0.5, 1.0])

    density = electrostatic_energy_density_phase_pair(pair, alpha, np.array([2.0, 2.0, 2.0]), np.zeros(3))

    np.testing.assert_allclose(density, np.array([4.0, 10.0, 16.0]))


def test_total_electrostatic_energy_integrates_cell_volumes() -> None:
    eps = np.array([2.0, 4.0])
    ex = np.array([1.0, 2.0])
    ey = np.array([0.0, 0.0])
    volume = np.array([0.5, 0.25])
    assert total_electrostatic_energy(eps, ex, ey, volume) == pytest.approx(2.5)


def test_total_electrostatic_energy_material_integrates_cell_volumes() -> None:
    material = LeakyDielectricMaterial(permittivity=4.0, conductivity=0.5)
    ex = np.array([1.0, 2.0])
    ey = np.array([0.0, 0.0])
    volume = np.array([0.5, 0.25])

    assert total_electrostatic_energy_material(material, ex, ey, volume) == pytest.approx(3.0)


def test_total_electrostatic_energy_phase_pair_integrates_mixed_energy() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=1.0),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=0.1),
    )
    alpha = np.array([0.0, 0.5, 1.0])

    total = total_electrostatic_energy_phase_pair(pair, alpha, np.array([2.0, 2.0, 2.0]), np.zeros(3), 0.5)

    assert total == pytest.approx(15.0)


def test_charge_conservation_residual_is_zero_for_balanced_update() -> None:
    old = np.array([1.0, 2.0, 3.0])
    div_j = np.array([0.5, -1.0, 0.5])
    dt = 0.2
    new = old - dt * div_j
    residual = charge_conservation_residual(old, new, div_j, dt)
    assert np.max(np.abs(residual)) < 1.0e-14
