from __future__ import annotations

import math

import numpy as np
import pytest

from cone_jet_metrics import (
    ConeJetErrorBudgetRow,
    ConeJetObservableSet,
    charge_to_mass_ratio,
    circular_jet_area,
    cone_jet_error_budget_passes,
    cone_jet_error_budget_rows,
    cone_jet_relative_errors,
    cone_jet_validation_passes,
    current_from_charge_to_mass,
    current_from_current_density,
    equivalent_diameter_from_area,
    mass_flow_rate_from_jet,
    sauter_mean_diameter,
)


def test_circular_jet_area_and_equivalent_diameter_are_inverse() -> None:
    diameter = 12.0e-6
    area = circular_jet_area(diameter)
    assert area == pytest.approx(math.pi * (6.0e-6) ** 2)
    assert equivalent_diameter_from_area(area) == pytest.approx(diameter)


def test_current_from_uniform_current_density() -> None:
    assert current_from_current_density(2.0e5, 10.0e-6) == pytest.approx(2.0e5 * math.pi * (5.0e-6) ** 2)


def test_charge_to_mass_ratio_round_trips_current_and_mass_flow() -> None:
    mdot = mass_flow_rate_from_jet(density=1200.0, axial_velocity=35.0, diameter=8.0e-6)
    current = current_from_charge_to_mass(2.5e5, mdot)

    assert mdot == pytest.approx(1200.0 * 35.0 * math.pi * (4.0e-6) ** 2)
    assert charge_to_mass_ratio(current, mdot) == pytest.approx(2.5e5)


def test_sauter_mean_diameter_matches_weighted_definition() -> None:
    diameters = np.array([1.0, 2.0, 3.0])
    counts = np.array([2.0, 1.0, 1.0])
    expected = np.sum(counts * diameters**3) / np.sum(counts * diameters**2)
    assert sauter_mean_diameter(diameters, counts) == pytest.approx(expected)


def test_cone_jet_quantitative_reference_errors_are_observable_specific() -> None:
    reference = ConeJetObservableSet(current=8.0e-8, jet_diameter=1.2e-6, droplet_diameter=3.0e-6, charge_to_mass=2.5e5)
    prediction = ConeJetObservableSet(current=8.8e-8, jet_diameter=1.1e-6, droplet_diameter=3.3e-6, charge_to_mass=2.25e5)
    tolerances = {"current": 0.20, "jet_diameter": 0.20, "droplet_diameter": 0.25, "charge_to_mass": 0.20}

    errors = cone_jet_relative_errors(prediction, reference)
    passed, returned_errors = cone_jet_validation_passes(prediction, reference, tolerances)

    assert errors["current"] == pytest.approx(0.10)
    assert errors["jet_diameter"] == pytest.approx(1.0 / 12.0)
    assert errors["droplet_diameter"] == pytest.approx(0.10)
    assert errors["charge_to_mass"] == pytest.approx(0.10)
    assert returned_errors == errors
    assert passed


def test_cone_jet_error_budget_rows_are_paper_table_ready() -> None:
    reference = ConeJetObservableSet(current=8.0e-8, jet_diameter=1.2e-6, droplet_diameter=3.0e-6, charge_to_mass=2.5e5)
    prediction = ConeJetObservableSet(current=8.8e-8, jet_diameter=1.1e-6, droplet_diameter=3.3e-6, charge_to_mass=2.25e5)
    tolerances = {"current": 0.20, "jet_diameter": 0.20, "droplet_diameter": 0.25, "charge_to_mass": 0.20}

    rows = cone_jet_error_budget_rows(prediction, reference, tolerances)

    assert len(rows) == 4
    assert all(isinstance(row, ConeJetErrorBudgetRow) for row in rows)
    assert [row.observable for row in rows] == ["current", "jet_diameter", "droplet_diameter", "charge_to_mass"]
    assert rows[0].relative_error == pytest.approx(0.10)
    assert all(row.passed for row in rows)
    assert cone_jet_error_budget_passes(rows)
