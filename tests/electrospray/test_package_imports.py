from __future__ import annotations

import importlib.util
from pathlib import Path


def test_solver_package_exports_core_entry_points() -> None:
    package_path = Path(__file__).resolve().parents[2] / "solver_electrospray" / "__init__.py"
    spec = importlib.util.spec_from_file_location("solver_electrospray_package_test", package_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ElectrosprayState1D is not None
    assert module.application_report is not None
    assert module.application_report_json is not None
    assert module.application_report_artifact_is_current is not None
    assert module.accumulated_panel_mass_loading is not None
    assert module.LeakyDielectricMaterial is not None
    assert module.LeakyDielectricPhasePair is not None
    assert module.EHD3DBoundaryCondition is not None
    assert module.EHD3DRefinementIndicators is not None
    assert module.EHD3DWhippingObservables is not None
    assert module.GeometricVOFTransportDiagnostics is not None
    assert module.ReducedStepDiagnostics is not None
    assert module.advance_reduced_electrospray_1d is not None
    assert module.advance_reduced_electrospray_1d_phase_pair is not None
    assert module.advect_scalar_limited_linear_fvm is not None
    assert module.advect_vof_geometric_plic is not None
    assert module.alpha_values_from_boundaries is not None
    assert module.apply_boundary_velocity_fluxes is not None
    assert module.apply_contact_angle_curvature_3d is not None
    assert module.balanced_capillary_force_density_3d is not None
    assert module.charge_values_from_boundaries is not None
    assert module.compute_ehd3d_refinement_indicators is not None
    assert module.compute_whipping_observables_3d is not None
    assert module.backward_euler_relaxation_step_material is not None
    assert module.backward_euler_relaxation_step_phase_pair is not None
    assert module.electric_relaxation_dt_material is not None
    assert module.electric_relaxation_dt_phase_pair is not None
    assert module.electric_shear_traction_jump_phase_pair is not None
    assert module.electrical_power is not None
    assert module.electrostatic_energy_density_material is not None
    assert module.electrostatic_energy_density_phase_pair is not None
    assert module.effective_no_through_boundary_tags is not None
    assert module.exact_relaxation_step_material is not None
    assert module.exact_relaxation_step_phase_pair is not None
    assert module.failed_reduced_step_invariants is not None
    assert module.free_charge_loss_fraction is not None
    assert module.ideal_power_efficiency is not None
    assert module.kinetic_power is not None
    assert module.deposited_mass_flow is not None
    assert module.exposure_margin is not None
    assert module.exposure_margin_status is not None
    assert module.solve_electrostatic_1d is not None
    assert module.solve_laplace_2d is not None
    assert module.specific_impulse_from_thrust is not None
    assert module.layered_dielectric_exact is not None
    assert module.least_squares_gradient is not None
    assert module.ohmic_current_density_material is not None
    assert module.ohmic_current_density_phase_pair is not None
    assert module.normal_ohmic_current_jump_phase_pair is not None
    assert module.operating_point is not None
    assert module.microthruster_operating_point_report is not None
    assert module.microthruster_operating_point_report_json is not None
    assert module.microthruster_report_artifact_is_current is not None
    assert module.maxwell_stress_force_density_2d is not None
    assert module.maxwell_stress_force_density_3d is not None
    assert module.PressureVelocityProjection is not None
    assert module.panel_current_density is not None
    assert module.panel_mass_flux is not None
    assert module.phase_pair_harmonic_face_fields is not None
    assert module.phase_pair_leaky_dielectric_properties is not None
    assert module.potential_values_from_boundaries is not None
    assert module.plume_impingement_report is not None
    assert module.plume_impingement_report_json is not None
    assert module.plume_report_artifact_is_current is not None
    assert module.reduced_phase_pair_step_diagnostics is not None
    assert module.reduced_phase_pair_step_report is not None
    assert module.reduced_phase_pair_step_report_json is not None
    assert module.reduced_phase_pair_step_scenario is not None
    assert module.reduced_step_diagnostics is not None
    assert module.reduced_step_invariant_status is not None
    assert module.reduced_step_report_artifact_is_current is not None
    assert module.retained_current is not None
    assert module.retained_mass_flow is not None
    assert module.retained_thrust_fraction is not None
    assert module.run_application_contamination_case is not None
    assert module.run_application_effective_performance_case is not None
    assert module.run_application_loss_accounting_case is not None
    assert module.run_application_power_accounting_case is not None
    assert module.run_core_validation_suite is not None
    assert module.total_electrostatic_energy_material is not None
    assert module.total_electrostatic_energy_phase_pair is not None
    assert module.surface_charge_density_phase_pair is not None
    assert module.thrust_to_power is not None
    assert module.time_to_panel_mass_loading is not None
    assert module.core_validation_summary is not None
    assert module.executable_manifest_coverage is not None
    assert module.format_validation_markdown_with_summary is not None
    assert module.project_velocity_piso is not None
    assert module.project_velocity_pimple is not None
    assert module.rhie_chow_face_flux is not None
    assert module.validation_artifacts_are_current is not None
    assert module.validation_artifact_status is not None
    assert module.validation_summary_health_trace_is_current is not None
    assert module.write_validation_artifacts is not None
    assert module.whipping_frequency_from_centroid_history is not None


def test_solver_package_all_exports_validation_summary_health_trace_helper() -> None:
    package_path = Path(__file__).resolve().parents[2] / "solver_electrospray" / "__init__.py"
    spec = importlib.util.spec_from_file_location("solver_electrospray_package_all_test", package_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "validation_summary_health_trace_is_current" in module.__all__
    assert "EHD3DBoundaryCondition" in module.__all__
    assert "EHD3DWhippingObservables" in module.__all__
    assert "advect_scalar_limited_linear_fvm" in module.__all__
    assert "advect_vof_geometric_plic" in module.__all__
    assert "apply_boundary_velocity_fluxes" in module.__all__
    assert "apply_contact_angle_curvature_3d" in module.__all__
    assert "balanced_capillary_force_density_3d" in module.__all__
    assert "charge_values_from_boundaries" in module.__all__
    assert "compute_ehd3d_refinement_indicators" in module.__all__
    assert "compute_whipping_observables_3d" in module.__all__
    assert "effective_no_through_boundary_tags" in module.__all__
    assert "least_squares_gradient" in module.__all__
    assert "maxwell_stress_force_density_2d" in module.__all__
    assert "maxwell_stress_force_density_3d" in module.__all__
    assert "project_velocity_piso" in module.__all__
    assert "project_velocity_pimple" in module.__all__
    assert "rhie_chow_face_flux" in module.__all__
    assert "whipping_frequency_from_centroid_history" in module.__all__
    assert getattr(module, "validation_summary_health_trace_is_current") is not None
