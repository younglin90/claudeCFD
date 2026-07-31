from __future__ import annotations

import numpy as np
import pytest

from material_properties import (
    LeakyDielectricMaterial,
    LeakyDielectricPhasePair,
    harmonic_face_property,
    leaky_dielectric_properties,
    mixture_property,
    phase_pair_harmonic_face_fields,
    phase_pair_leaky_dielectric_properties,
    validate_positive_material_properties,
)


def test_mixture_property_recovers_pure_phase_limits_and_bounds() -> None:
    alpha = np.array([0.0, 0.25, 1.0])
    mixed = mixture_property(alpha, liquid_value=10.0, gas_value=2.0)
    np.testing.assert_allclose(mixed, np.array([2.0, 4.0, 10.0]))
    assert np.all((2.0 <= mixed) & (mixed <= 10.0))


def test_harmonic_face_property_preserves_series_coefficient() -> None:
    assert harmonic_face_property(2.0, 8.0) == pytest.approx(3.2)
    arr = harmonic_face_property(np.array([1.0, 2.0]), np.array([1.0, 6.0]))
    np.testing.assert_allclose(arr, np.array([1.0, 3.0]))


def test_leaky_dielectric_properties_mix_epsilon_and_sigma() -> None:
    alpha = np.array([1.0, 0.5, 0.0])
    epsilon, sigma = leaky_dielectric_properties(alpha, 6.0, 2.0, 1.0e-5, 1.0e-9)
    np.testing.assert_allclose(epsilon, np.array([6.0, 4.0, 2.0]))
    np.testing.assert_allclose(sigma, np.array([1.0e-5, 5.0005e-6, 1.0e-9]))


def test_validate_positive_material_properties_names_bad_property() -> None:
    with pytest.raises(ValueError, match="conductivity"):
        validate_positive_material_properties(permittivity=1.0, conductivity=0.0)


def test_leaky_dielectric_material_validates_electrical_properties() -> None:
    material = LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-9)
    assert material.permittivity == pytest.approx(2.0)
    assert material.relaxation_time == pytest.approx(2.0e9)
    assert material.relaxation_factor(2.0e9) == pytest.approx(np.exp(-1.0))
    with pytest.raises(ValueError, match="permittivity"):
        LeakyDielectricMaterial(permittivity=0.0, conductivity=1.0)
    with pytest.raises(ValueError, match="dt"):
        material.relaxation_factor(-1.0)


def test_leaky_dielectric_material_accepts_optional_hydrodynamic_properties() -> None:
    material = LeakyDielectricMaterial(
        permittivity=2.0,
        conductivity=1.0e-9,
        density=1000.0,
        dynamic_viscosity=1.0e-3,
    )

    assert material.density == pytest.approx(1000.0)
    assert material.dynamic_viscosity == pytest.approx(1.0e-3)
    with pytest.raises(ValueError, match="density"):
        LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-9, density=0.0)


def test_phase_pair_leaky_dielectric_properties_uses_typed_materials() -> None:
    liquid = LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0e-5)
    gas = LeakyDielectricMaterial(permittivity=2.0, conductivity=2.0e-9)
    alpha = np.array([0.0, 0.5, 1.0])

    epsilon, sigma = phase_pair_leaky_dielectric_properties(alpha, liquid, gas)

    np.testing.assert_allclose(epsilon, np.array([2.0, 5.0, 8.0]))
    np.testing.assert_allclose(sigma, np.array([2.0e-9, 2.0001e-5, 4.0e-5]))


def test_leaky_dielectric_phase_pair_mixes_fields() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0e-5),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=2.0e-9),
    )

    epsilon, sigma = pair.mixture_fields(np.array([0.0, 0.5, 1.0]))

    assert pair.permittivity_ratio == pytest.approx(4.0)
    assert pair.conductivity_ratio == pytest.approx(2.0e4)
    np.testing.assert_allclose(epsilon, np.array([2.0, 5.0, 8.0]))
    np.testing.assert_allclose(sigma, np.array([2.0e-9, 2.0001e-5, 4.0e-5]))
    np.testing.assert_allclose(pair.relaxation_time_field(np.array([0.0, 0.5, 1.0])), epsilon / sigma)


def test_leaky_dielectric_phase_pair_mixes_hydrodynamic_fields() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(
            permittivity=8.0,
            conductivity=4.0e-5,
            density=1000.0,
            dynamic_viscosity=1.0e-3,
        ),
        gas=LeakyDielectricMaterial(
            permittivity=2.0,
            conductivity=2.0e-9,
            density=2.0,
            dynamic_viscosity=2.0e-5,
        ),
    )
    alpha = np.array([0.0, 0.5, 1.0])

    density = pair.density_field(alpha, fallback_density=10.0)
    dynamic_viscosity = pair.dynamic_viscosity_field(alpha, fallback_dynamic_viscosity=1.0)
    kinematic_viscosity = pair.kinematic_viscosity_field(alpha, fallback_density=10.0, fallback_kinematic_viscosity=0.0)

    np.testing.assert_allclose(density, np.array([2.0, 501.0, 1000.0]))
    np.testing.assert_allclose(dynamic_viscosity, np.array([2.0e-5, 5.1e-4, 1.0e-3]))
    np.testing.assert_allclose(kinematic_viscosity, dynamic_viscosity / density)


def test_leaky_dielectric_phase_pair_harmonic_face_fields() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0),
    )

    epsilon_face, sigma_face = pair.harmonic_face_fields(np.array([0.0, 1.0]), np.array([1.0, 0.0]))

    np.testing.assert_allclose(epsilon_face, np.array([3.2, 3.2]))
    np.testing.assert_allclose(sigma_face, np.array([1.6, 1.6]))


def test_phase_pair_harmonic_face_fields_function_matches_pair_method() -> None:
    liquid = LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0)
    gas = LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0)
    pair = LeakyDielectricPhasePair(liquid=liquid, gas=gas)

    expected = pair.harmonic_face_fields(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    actual = phase_pair_harmonic_face_fields(np.array([0.0, 1.0]), np.array([1.0, 0.0]), liquid, gas)

    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
