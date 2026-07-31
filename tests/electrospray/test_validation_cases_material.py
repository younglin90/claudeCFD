from __future__ import annotations

from validation_cases_material import (
    run_all_material_cases,
    run_harmonic_face_material_case,
    run_material_relaxation_factor_case,
    run_material_relaxation_time_case,
    run_mixture_property_bounds_case,
    run_phase_pair_electrical_diagnostics_case,
    run_phase_pair_harmonic_face_case,
    run_phase_pair_material_case,
    run_phase_pair_object_case,
    run_phase_pair_ratio_case,
    run_phase_pair_relaxation_step_case,
    run_phase_pair_relaxation_timestep_case,
    run_phase_pair_relaxation_time_field_case,
)


def test_mixture_property_bounds_validation_case_passes() -> None:
    assert run_mixture_property_bounds_case().passed


def test_phase_pair_material_validation_case_passes() -> None:
    assert run_phase_pair_material_case().passed


def test_phase_pair_object_validation_case_passes() -> None:
    assert run_phase_pair_object_case().passed


def test_phase_pair_ratio_validation_case_passes() -> None:
    assert run_phase_pair_ratio_case().passed


def test_phase_pair_electrical_diagnostics_validation_case_passes() -> None:
    assert run_phase_pair_electrical_diagnostics_case().passed


def test_material_relaxation_time_validation_case_passes() -> None:
    assert run_material_relaxation_time_case().passed


def test_phase_pair_relaxation_time_field_validation_case_passes() -> None:
    assert run_phase_pair_relaxation_time_field_case().passed


def test_phase_pair_relaxation_timestep_validation_case_passes() -> None:
    assert run_phase_pair_relaxation_timestep_case().passed


def test_phase_pair_relaxation_step_validation_case_passes() -> None:
    assert run_phase_pair_relaxation_step_case().passed


def test_material_relaxation_factor_validation_case_passes() -> None:
    assert run_material_relaxation_factor_case().passed


def test_harmonic_face_material_validation_case_passes() -> None:
    assert run_harmonic_face_material_case().passed


def test_phase_pair_harmonic_face_validation_case_passes() -> None:
    assert run_phase_pair_harmonic_face_case().passed


def test_all_material_validation_cases_pass() -> None:
    results = run_all_material_cases()

    assert len(results) == 12
    assert all(result.passed for result in results)
