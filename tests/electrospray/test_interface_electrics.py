from __future__ import annotations

import pytest

from interface_electrics import (
    electric_shear_traction_jump,
    electric_shear_traction_jump_phase_pair,
    normal_ohmic_current_jump,
    normal_ohmic_current_jump_phase_pair,
    surface_charge_density,
    surface_charge_density_phase_pair,
    tangential_field_jump,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair


def test_surface_charge_density_is_normal_displacement_jump() -> None:
    q_s = surface_charge_density(2.0, (3.0, 1.0), 5.0, (1.0, 4.0), normal=(1.0, 0.0))
    assert q_s == pytest.approx(5.0 * 1.0 - 2.0 * 3.0)


def test_zero_surface_charge_when_displacement_is_continuous() -> None:
    assert surface_charge_density(2.0, (4.0, 0.0), 8.0, (1.0, 0.0), normal=(1.0, 0.0)) == pytest.approx(0.0)


def test_surface_charge_density_phase_pair_uses_phase_permittivities() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0),
        gas=LeakyDielectricMaterial(permittivity=8.0, conductivity=0.1),
    )

    assert surface_charge_density_phase_pair(pair, (4.0, 0.0), (1.0, 0.0), normal=(1.0, 0.0)) == pytest.approx(0.0)


def test_normal_ohmic_current_jump_uses_conductivity_weighting() -> None:
    jump = normal_ohmic_current_jump(3.0, (2.0, 0.0), 4.0, (5.0, 0.0), normal=(1.0, 0.0))
    assert jump == pytest.approx(14.0)


def test_normal_ohmic_current_jump_phase_pair_uses_phase_conductivities() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=3.0),
        gas=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
    )

    jump = normal_ohmic_current_jump_phase_pair(pair, (2.0, 0.0), (5.0, 0.0), normal=(1.0, 0.0))

    assert jump == pytest.approx(14.0)


def test_tangential_field_jump_is_zero_for_continuous_tangential_field() -> None:
    jump = tangential_field_jump((1.0, 2.0), (3.0, 2.0), normal=(1.0, 0.0))
    assert jump == pytest.approx(0.0)


def test_electric_shear_traction_jump_uses_normal_and_tangential_fields() -> None:
    jump = electric_shear_traction_jump(2.0, (3.0, 4.0), 5.0, (1.0, 2.0), normal=(1.0, 0.0))
    assert jump == pytest.approx(5.0 * 1.0 * 2.0 - 2.0 * 3.0 * 4.0)


def test_electric_shear_traction_jump_phase_pair_uses_phase_permittivities() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=3.0),
        gas=LeakyDielectricMaterial(permittivity=5.0, conductivity=4.0),
    )

    jump = electric_shear_traction_jump_phase_pair(pair, (3.0, 4.0), (1.0, 2.0), normal=(1.0, 0.0))

    assert jump == pytest.approx(-14.0)


def test_electric_shear_traction_jump_vanishes_without_tangential_field() -> None:
    jump = electric_shear_traction_jump(2.0, (3.0, 0.0), 5.0, (1.0, 0.0), normal=(1.0, 0.0))
    assert jump == pytest.approx(0.0)
