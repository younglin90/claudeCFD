from __future__ import annotations

import numpy as np
import pytest

from spacecraft_impingement import (
    accumulated_panel_mass_loading,
    deposited_current,
    deposited_mass_flow,
    effective_thrust_after_impingement,
    exposure_margin,
    exposure_margin_status,
    panel_current_density,
    panel_mass_flux,
    rectangular_panel_impingement_fraction,
    retained_current,
    retained_mass_flow,
    retained_thrust_fraction,
    thrust_loss_fraction,
    time_to_panel_mass_loading,
)


def test_rectangular_panel_impingement_counts_particles_inside_panel() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.4, 0.2, 2.0],
            [0.6, 0.0, 2.0],
            [0.0, 0.0, 1.5],
        ]
    )
    fraction = rectangular_panel_impingement_fraction(positions, plane_z=2.0, width=1.0, height=0.5)

    assert fraction == pytest.approx(0.5)


def test_rectangular_panel_impingement_supports_offset_center() -> None:
    positions = np.array([[1.0, 1.0, 3.0], [0.0, 0.0, 3.0]])

    assert rectangular_panel_impingement_fraction(positions, 3.0, 0.4, 0.4, center_xy=(1.0, 1.0)) == pytest.approx(0.5)


def test_deposited_current_scales_with_impingement_fraction() -> None:
    assert deposited_current(12.0e-6, 0.25) == pytest.approx(3.0e-6)
    assert retained_current(12.0e-6, 0.25) == pytest.approx(9.0e-6)


def test_panel_current_density_reports_surface_loading() -> None:
    assert panel_current_density(12.0e-6, 0.25, panel_area=0.02) == pytest.approx(1.5e-4)
    with pytest.raises(ValueError, match="panel_area"):
        panel_current_density(12.0e-6, 0.25, panel_area=0.0)


def test_panel_mass_flux_reports_contamination_loading() -> None:
    assert deposited_mass_flow(2.4e-9, 0.25) == pytest.approx(6.0e-10)
    assert retained_mass_flow(2.4e-9, 0.25) == pytest.approx(1.8e-9)
    assert panel_mass_flux(2.4e-9, 0.25, panel_area=0.02) == pytest.approx(3.0e-8)
    with pytest.raises(ValueError, match="total_mass_flow"):
        deposited_mass_flow(-1.0, 0.25)


def test_accumulated_panel_mass_loading_integrates_exposure_time() -> None:
    assert accumulated_panel_mass_loading(3.0e-8, exposure_time=3600.0) == pytest.approx(1.08e-4)
    with pytest.raises(ValueError, match="exposure_time"):
        accumulated_panel_mass_loading(3.0e-8, exposure_time=-1.0)


def test_time_to_panel_mass_loading_inverts_mass_flux() -> None:
    assert time_to_panel_mass_loading(1.0e-3, mass_flux=3.0e-8) == pytest.approx(33333.333333333336)
    with pytest.raises(ValueError, match="mass_flux"):
        time_to_panel_mass_loading(1.0e-3, mass_flux=0.0)


def test_exposure_margin_compares_limit_time_to_exposure_time() -> None:
    assert exposure_margin(33333.333333333336, exposure_time=3600.0) == pytest.approx(9.25925925925926)
    with pytest.raises(ValueError, match="exposure_time"):
        exposure_margin(1.0, exposure_time=0.0)


def test_exposure_margin_status_classifies_margin() -> None:
    assert exposure_margin_status(1.0) == "pass"
    assert exposure_margin_status(0.5) == "fail"
    with pytest.raises(ValueError, match="margin"):
        exposure_margin_status(-1.0)


def test_thrust_loss_fraction_uses_intercepted_axial_momentum() -> None:
    assert thrust_loss_fraction(0.2, axial_momentum_fraction=0.75) == pytest.approx(0.15)
    assert retained_thrust_fraction(0.2, axial_momentum_fraction=0.75) == pytest.approx(0.85)
    with pytest.raises(ValueError, match="impingement_fraction"):
        thrust_loss_fraction(1.2)


def test_effective_thrust_after_impingement_applies_retained_fraction() -> None:
    assert effective_thrust_after_impingement(2.4e-6, 0.25, axial_momentum_fraction=0.8) == pytest.approx(1.92e-6)
    with pytest.raises(ValueError, match="thrust"):
        effective_thrust_after_impingement(-1.0, 0.25)
