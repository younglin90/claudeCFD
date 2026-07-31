from __future__ import annotations

import math

import pytest

import numpy as np

from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair
from timestep import (
    advective_dt,
    capillary_dt,
    combined_explicit_dt,
    diffusive_dt,
    electric_relaxation_dt,
    electric_relaxation_dt_material,
    electric_relaxation_dt_phase_pair,
)


def test_advective_dt_uses_cfl_limit_and_handles_zero_speed() -> None:
    assert advective_dt(0.1, max_velocity=2.0, cfl=0.5) == pytest.approx(0.025)
    assert math.isinf(advective_dt(0.1, max_velocity=0.0, cfl=0.5))


def test_diffusive_dt_uses_dx_squared_scaling() -> None:
    base = diffusive_dt(0.1, diffusivity=0.02)
    refined = diffusive_dt(0.05, diffusivity=0.02)
    assert refined == pytest.approx(0.25 * base)
    assert math.isinf(diffusive_dt(0.1, diffusivity=0.0))


def test_combined_explicit_dt_takes_minimum_limit() -> None:
    assert combined_explicit_dt(math.inf, 0.2, 0.1) == pytest.approx(0.1)


def test_electric_relaxation_dt_uses_epsilon_over_sigma() -> None:
    assert electric_relaxation_dt(8.0e-10, conductivity=2.0e-8, safety=0.25) == pytest.approx(0.01)
    assert math.isinf(electric_relaxation_dt(8.0e-10, conductivity=0.0))


def test_electric_relaxation_dt_material_uses_validated_material() -> None:
    material = LeakyDielectricMaterial(permittivity=8.0e-10, conductivity=2.0e-8)

    assert electric_relaxation_dt_material(material, safety=0.25) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="safety"):
        electric_relaxation_dt_material(material, safety=0.0)


def test_electric_relaxation_dt_phase_pair_uses_minimum_mixed_timescale() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=3.0, conductivity=3.0),
    )

    assert electric_relaxation_dt_phase_pair(pair, np.array([0.0, 0.5, 1.0]), safety=0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="safety"):
        electric_relaxation_dt_phase_pair(pair, np.array([0.0]), safety=0.0)


def test_capillary_dt_uses_capillary_wave_scale() -> None:
    dt = capillary_dt(1.0e-4, density=1000.0, surface_tension=0.05, safety=0.5)
    assert dt == pytest.approx(0.5 * math.sqrt(1000.0 * (1.0e-4) ** 3 / 0.05))
    assert math.isinf(capillary_dt(1.0e-4, density=1000.0, surface_tension=0.0))
