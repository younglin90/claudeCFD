from __future__ import annotations

from validation_cases_interface import (
    run_all_interface_cases,
    run_electric_shear_traction_case,
    run_phase_pair_interface_jump_case,
    run_ohmic_current_jump_case,
    run_surface_charge_jump_case,
    run_tangential_field_continuity_case,
)


def test_surface_charge_jump_validation_case_passes() -> None:
    assert run_surface_charge_jump_case().passed


def test_ohmic_current_jump_validation_case_passes() -> None:
    assert run_ohmic_current_jump_case().passed


def test_tangential_field_continuity_validation_case_passes() -> None:
    assert run_tangential_field_continuity_case().passed


def test_electric_shear_traction_validation_case_passes() -> None:
    assert run_electric_shear_traction_case().passed


def test_phase_pair_interface_jump_validation_case_passes() -> None:
    assert run_phase_pair_interface_jump_case().passed


def test_all_interface_validation_cases_pass() -> None:
    results = run_all_interface_cases()

    assert len(results) == 5
    assert all(result.passed for result in results)
