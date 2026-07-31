from __future__ import annotations

import numpy as np

from ehd3d import (
    EHD3D_SCHEME_ID,
    EHD3DBoundaryCondition,
    EHD3DConfig,
    extract_droplet_components_3d,
    apply_boundary_velocity_fluxes,
    apply_contact_angle_curvature_3d,
    advect_vof_geometric_plic,
    advance_ehd3d_fvm,
    alpha_values_from_boundaries,
    balanced_capillary_force_density_3d,
    charge_values_from_boundaries,
    compute_ehd3d_refinement_indicators,
    compute_whipping_observables_3d,
    compress_vof_pairwise,
    ehd3d_stable_timestep,
    estimate_ehd3d_timestep_limits,
    initialize_perturbed_taylor_cone_column,
    maxwell_stress_force_density_3d,
    potential_values_from_boundaries,
    reconstruct_plic_3d,
    solve_electrostatic_3d,
    whipping_frequency_from_centroid_history,
)
from ehd3d import ElectrosprayState3D
from ehd3d import PLICReconstruction3D
from fvm_mesh import gauss_gradient
from fvm_mesh import structured_cartesian_mesh_3d, two_cell_skew_mesh_3d
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair
from validation_cases_3d import run_all_3d_face_mesh_cases


def _phase_pair() -> LeakyDielectricPhasePair:
    return LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=4.0, conductivity=0.5, density=1000.0),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-3, density=1.0),
    )


def test_unstructured_face_mesh_ehd_step_activates_3d_terms() -> None:
    mesh = two_cell_skew_mesh_3d()
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 0.0]),
        charge_density=np.array([0.1, 0.0]),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )

    next_state, diagnostics = advance_ehd3d_fvm(
        mesh,
        state,
        EHD3DConfig(
            phase_pair=_phase_pair(),
            potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
            dt=1.0e-4,
            reference_density=1000.0,
        ),
    )

    assert next_state.cell_count == state.cell_count
    assert diagnostics.scheme_id == EHD3D_SCHEME_ID
    assert diagnostics.unstructured_mesh_supported is True
    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_maxwell_acceleration > 0.0
    assert diagnostics.ohmic_conduction_active is True
    assert diagnostics.charge_relaxation_active is True
    assert diagnostics.pressure_projection_active is True
    assert diagnostics.pressure_corrector_count == 2
    assert diagnostics.rhie_chow_active is True
    assert diagnostics.pimple_outer_corrector_count == 1
    assert diagnostics.non_orthogonal_corrector_count == 0
    assert diagnostics.scalar_transport_scheme == "limited_linear"
    assert diagnostics.maxwell_force_scheme == "stress_divergence"
    assert diagnostics.capillary_force_scheme == "balanced_face"
    assert diagnostics.divergence_reduction_ratio <= 1.0
    assert diagnostics.gas_charge_leakage_fraction <= 1.0e-12


def test_plic_reconstruction_recovers_planar_interface_normal_and_zero_curvature() -> None:
    mesh = structured_cartesian_mesh_3d(4, 2, 2)
    alpha = mesh.cell_centers[:, 0]

    reconstruction = reconstruct_plic_3d(mesh, alpha)

    active = reconstruction.interface_cells
    assert np.count_nonzero(active) == mesh.cell_count
    assert np.max(np.abs(reconstruction.normals[active] - np.array([-1.0, 0.0, 0.0]))) <= 1.0e-12
    assert np.max(np.abs(reconstruction.curvature[active])) <= 1.0e-12


def test_maxwell_stress_force_vanishes_for_uniform_electric_field() -> None:
    mesh = structured_cartesian_mesh_3d(3, 2, 1)
    state = ElectrosprayState3D(
        alpha_liquid=np.ones(mesh.cell_count),
        charge_density=np.zeros(mesh.cell_count),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )
    electrostatic = solve_electrostatic_3d(mesh, state, _phase_pair(), {"x_min": 1.0, "x_max": 0.0})

    force = maxwell_stress_force_density_3d(mesh, electrostatic)

    assert np.max(np.abs(force)) <= 1.0e-12


def test_balanced_capillary_force_cancels_matching_pressure_gradient() -> None:
    mesh = structured_cartesian_mesh_3d(4, 1, 1)
    alpha = mesh.cell_centers[:, 0]
    reconstruction = PLICReconstruction3D(
        normals=np.tile(np.array([-1.0, 0.0, 0.0]), (mesh.cell_count, 1)),
        plane_constants=np.zeros(mesh.cell_count),
        curvature=np.ones(mesh.cell_count) * 2.0,
        interface_cells=np.ones(mesh.cell_count, dtype=bool),
    )

    capillary_force = balanced_capillary_force_density_3d(mesh, alpha, 0.05, reconstruction)
    balancing_pressure_gradient = gauss_gradient(mesh, -0.05 * reconstruction.curvature * alpha)

    assert np.max(np.abs(capillary_force + balancing_pressure_gradient)) <= 1.0e-12


def test_boundary_condition_schema_supplies_potential_and_alpha_values() -> None:
    mesh = structured_cartesian_mesh_3d(1, 1, 1)
    config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 0.0},
        dt=1.0e-4,
        reference_density=1000.0,
        boundary_conditions={
            "x_min": EHD3DBoundaryCondition(kind="inlet", alpha_liquid=1.0, potential=2.0),
            "x_max": EHD3DBoundaryCondition(kind="outlet", alpha_liquid=0.25),
        },
    )

    assert alpha_values_from_boundaries(config, mesh)["x_min"] == 1.0
    assert alpha_values_from_boundaries(config, mesh)["x_max"] == 0.25
    assert charge_values_from_boundaries(config) == {}
    assert potential_values_from_boundaries(config)["x_min"] == 2.0


def test_boundary_velocity_and_charge_inlet_are_applied_to_step() -> None:
    mesh = structured_cartesian_mesh_3d(1, 1, 1)
    state = ElectrosprayState3D(
        alpha_liquid=np.zeros(mesh.cell_count),
        charge_density=np.zeros(mesh.cell_count),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )
    config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 0.0, "x_max": 0.0},
        dt=1.0e-2,
        reference_density=1000.0,
        boundary_conditions={
            "x_min": EHD3DBoundaryCondition(kind="inlet", alpha_liquid=1.0, charge_density=0.25, velocity=(1.0, 0.0, 0.0)),
            "x_max": EHD3DBoundaryCondition(kind="outlet"),
            "y_min": EHD3DBoundaryCondition(kind="wall"),
            "y_max": EHD3DBoundaryCondition(kind="wall"),
            "z_min": EHD3DBoundaryCondition(kind="wall"),
            "z_max": EHD3DBoundaryCondition(kind="wall"),
        },
        pressure_projection=False,
        confine_charge_to_liquid=False,
    )
    raw_flux = np.zeros(mesh.face_count)
    raw_flux[np.array(mesh.boundary_tags) == "x_max"] = -0.5

    corrected_flux = apply_boundary_velocity_fluxes(mesh, raw_flux, config)
    next_state, diagnostics = advance_ehd3d_fvm(mesh, state, config)

    assert np.min(corrected_flux[np.array(mesh.boundary_tags) == "x_min"]) < 0.0
    assert np.all(corrected_flux[np.array(mesh.boundary_tags) == "x_max"] == 0.0)
    assert charge_values_from_boundaries(config)["x_min"] == 0.25
    assert next_state.alpha_liquid[0] > state.alpha_liquid[0]
    assert next_state.charge_density[0] > state.charge_density[0]
    assert diagnostics.pressure_projection_active is False


def test_ehd_step_can_use_pimple_non_orthogonal_projection() -> None:
    mesh = two_cell_skew_mesh_3d()
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 0.0]),
        charge_density=np.array([0.1, 0.0]),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )

    _next_state, diagnostics = advance_ehd3d_fvm(
        mesh,
        state,
        EHD3DConfig(
            phase_pair=_phase_pair(),
            potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
            dt=1.0e-4,
            reference_density=1000.0,
            pimple_outer_corrector_count=2,
            non_orthogonal_corrector_count=1,
            momentum_under_relaxation=0.8,
        ),
    )

    assert diagnostics.rhie_chow_active is True
    assert diagnostics.pimple_outer_corrector_count == 2
    assert diagnostics.non_orthogonal_corrector_count == 1
    assert diagnostics.pressure_corrector_count == 6
    assert diagnostics.divergence_reduction_ratio <= 1.0


def test_contact_angle_curvature_correction_modifies_wall_interface_cell() -> None:
    mesh = structured_cartesian_mesh_3d(1, 1, 1)
    reconstruction = PLICReconstruction3D(
        normals=np.array([[-1.0, 0.0, 0.0]]),
        plane_constants=np.zeros(mesh.cell_count),
        curvature=np.zeros(mesh.cell_count),
        interface_cells=np.ones(mesh.cell_count, dtype=bool),
    )

    corrected = apply_contact_angle_curvature_3d(
        mesh,
        reconstruction,
        {"x_min": EHD3DBoundaryCondition(kind="wall", contact_angle_degrees=90.0)},
    )

    assert corrected.curvature[0] < reconstruction.curvature[0]


def test_whipping_observables_report_amplitude_extent_and_frequency() -> None:
    mesh, state = initialize_perturbed_taylor_cone_column(nx=4, ny=4, nz=4, perturbation_amplitude=0.2)
    times = np.linspace(0.0, 0.9, 10)
    offsets = 0.01 * np.sin(2.0 * np.pi * 2.0 * times)

    observables = compute_whipping_observables_3d(mesh, state.alpha_liquid, times=times, centroid_offsets=offsets)
    frequency = whipping_frequency_from_centroid_history(times, offsets)

    assert observables.transverse_centroid_offset > 0.0
    assert observables.transverse_rms_radius > 0.0
    assert observables.axial_extent > 0.0
    assert observables.dominant_frequency == frequency
    assert abs(frequency - 2.0) <= 1.0e-12


def test_pairwise_vof_compression_is_bounded_conservative_and_sharpens() -> None:
    mesh = two_cell_skew_mesh_3d()
    alpha = np.array([0.8, 0.2])

    compressed = compress_vof_pairwise(mesh, alpha, strength=1.0)

    assert np.all((compressed >= 0.0) & (compressed <= 1.0))
    assert np.sum(compressed * mesh.cell_volumes) == np.sum(alpha * mesh.cell_volumes)
    assert compressed[0] - compressed[1] > alpha[0] - alpha[1]


def test_geometric_plic_vof_advection_is_bounded_and_conservative() -> None:
    mesh = structured_cartesian_mesh_3d(3, 1, 1)
    alpha = np.array([1.0, 0.5, 0.0])
    flux = np.zeros(mesh.face_count)
    flux[np.flatnonzero(mesh.internal_faces)] = np.array([0.02, 0.02])

    next_alpha, diagnostics = advect_vof_geometric_plic(mesh, alpha, flux, dt=0.1)

    assert np.all((next_alpha >= 0.0) & (next_alpha <= 1.0))
    assert diagnostics.alpha_bounds_violation == 0.0
    assert abs(diagnostics.relative_mass_error) <= 1.0e-15
    assert abs(float(np.sum(next_alpha * mesh.cell_volumes) - np.sum(alpha * mesh.cell_volumes))) <= 1.0e-15


def test_ehd_step_can_use_geometric_plic_vof_transport() -> None:
    mesh = two_cell_skew_mesh_3d()
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 0.0]),
        charge_density=np.array([0.1, 0.0]),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )

    _next_state, diagnostics = advance_ehd3d_fvm(
        mesh,
        state,
        EHD3DConfig(
            phase_pair=_phase_pair(),
            potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
            dt=1.0e-4,
            reference_density=1000.0,
            scalar_transport_scheme="geometric_plic",
        ),
    )

    assert diagnostics.geometric_vof_active is True
    assert diagnostics.scalar_transport_scheme == "geometric_plic"
    assert diagnostics.pass_metric <= 1.0e-8


def test_3d_timestep_limits_react_to_face_flux_relaxation_and_capillarity() -> None:
    mesh = structured_cartesian_mesh_3d(2, 1, 1, lengths=(1.0, 1.0, 1.0))
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 1.0]),
        charge_density=np.zeros(mesh.cell_count),
        velocity=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        pressure=np.zeros(mesh.cell_count),
    )
    config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
        dt=1.0e-4,
        reference_density=1000.0,
        surface_tension=0.0,
    )

    limits = estimate_ehd3d_timestep_limits(mesh, state, config, cfl=0.5, relaxation_safety=0.25)

    assert limits.advective_dt == 0.125
    assert limits.electric_relaxation_dt == 2.0
    assert np.isinf(limits.capillary_dt)
    assert limits.stable_dt == limits.advective_dt
    assert ehd3d_stable_timestep(mesh, state, config, cfl=0.5, relaxation_safety=0.25) == limits.stable_dt

    conductive_config = EHD3DConfig(
        phase_pair=LeakyDielectricPhasePair(
            liquid=LeakyDielectricMaterial(permittivity=4.0, conductivity=100.0, density=1000.0),
            gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0, density=1.0),
        ),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
        dt=1.0e-4,
        reference_density=1000.0,
    )
    conductive_limits = estimate_ehd3d_timestep_limits(mesh, state, conductive_config, cfl=0.5, relaxation_safety=0.25)
    assert conductive_limits.stable_dt == conductive_limits.electric_relaxation_dt
    assert conductive_limits.stable_dt < limits.stable_dt

    static_state = ElectrosprayState3D(
        alpha_liquid=np.ones(mesh.cell_count),
        charge_density=np.zeros(mesh.cell_count),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )
    capillary_config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
        dt=1.0e-4,
        reference_density=1000.0,
        surface_tension=10.0,
    )
    capillary_limits = estimate_ehd3d_timestep_limits(mesh, static_state, capillary_config, capillary_safety=0.25)
    stronger_capillary = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
        dt=1.0e-4,
        reference_density=1000.0,
        surface_tension=40.0,
    )
    stronger_limits = estimate_ehd3d_timestep_limits(mesh, static_state, stronger_capillary, capillary_safety=0.25)
    assert capillary_limits.stable_dt == capillary_limits.capillary_dt
    assert stronger_limits.capillary_dt == 0.5 * capillary_limits.capillary_dt


def test_adaptive_ehd_step_reports_accepted_dt_and_force_limiter() -> None:
    mesh = structured_cartesian_mesh_3d(2, 1, 1, lengths=(1.0, 1.0, 1.0))
    state = ElectrosprayState3D(
        alpha_liquid=np.ones(mesh.cell_count),
        charge_density=np.ones(mesh.cell_count) * 0.1,
        velocity=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        pressure=np.zeros(mesh.cell_count),
    )
    config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
        dt=1.0,
        reference_density=1000.0,
        adaptive_timestep=True,
        max_cfl=0.5,
        max_force_velocity_increment=1.0e-8,
        pressure_projection=False,
    )

    _next_state, diagnostics = advance_ehd3d_fvm(mesh, state, config)

    assert diagnostics.accepted_dt == 0.125
    assert diagnostics.rejected_step_count == 0
    assert diagnostics.max_velocity_increment > config.max_force_velocity_increment


def test_step_reject_retries_until_pass_metric_is_acceptable() -> None:
    mesh, state = initialize_perturbed_taylor_cone_column()
    config = EHD3DConfig(
        phase_pair=_phase_pair(),
        potential_boundary_values={"z_min": 1.0, "z_max": 0.0},
        dt=1.0e-2,
        reference_density=1000.0,
        pressure_projection=False,
        step_reject_retry_count=8,
        step_reject_pass_metric=1.0e-6,
    )

    _next_state, diagnostics = advance_ehd3d_fvm(mesh, state, config)

    assert diagnostics.rejected_step_count > 0
    assert diagnostics.accepted_dt < config.dt
    assert diagnostics.pass_metric <= config.step_reject_pass_metric


def test_droplet_component_extraction_reports_detached_component_properties() -> None:
    mesh = structured_cartesian_mesh_3d(1, 1, 4)
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 0.0, 1.0, 1.0]),
        charge_density=np.array([0.2, 0.0, 0.1, 0.1]),
        velocity=np.tile(np.array([0.1, 0.0, 1.0]), (mesh.cell_count, 1)),
        pressure=np.zeros(mesh.cell_count),
    )

    components = extract_droplet_components_3d(mesh, state, _phase_pair())

    assert len(components) == 2
    assert any(component.detached for component in components)
    assert all(component.volume > 0.0 for component in components)
    assert all(component.equivalent_diameter > 0.0 for component in components)


def test_perturbed_taylor_cone_column_keeps_nonaxisymmetric_signal() -> None:
    mesh, state = initialize_perturbed_taylor_cone_column()

    _next_state, diagnostics = advance_ehd3d_fvm(
        mesh,
        state,
        EHD3DConfig(
            phase_pair=_phase_pair(),
            potential_boundary_values={"z_min": 1.0, "z_max": 0.0},
            dt=1.0e-4,
            reference_density=1000.0,
            surface_tension=0.03,
            vof_compression=0.25,
        ),
    )

    assert diagnostics.unstructured_mesh_supported is False
    assert diagnostics.nonaxisymmetric_centroid_offset > 0.0
    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_capillary_acceleration > 0.0
    assert diagnostics.gas_charge_leakage_fraction <= 1.0e-12
    assert diagnostics.pass_metric <= 1.0e-4


def test_refinement_indicators_mark_interface_charge_and_electric_layers() -> None:
    mesh = structured_cartesian_mesh_3d(4, 1, 1)
    state = ElectrosprayState3D(
        alpha_liquid=np.array([1.0, 0.75, 0.25, 0.0]),
        charge_density=np.array([0.0, 0.2, 0.1, 0.0]),
        velocity=np.zeros((mesh.cell_count, 3)),
        pressure=np.zeros(mesh.cell_count),
    )

    indicators = compute_ehd3d_refinement_indicators(
        mesh,
        state,
        phase_pair=_phase_pair(),
        potential_boundary_values={"x_min": 1.0, "x_max": 0.0},
    )

    assert np.all((indicators.combined >= 0.0) & (indicators.combined <= 1.0))
    assert np.max(indicators.interface) == 1.0
    assert np.max(indicators.charge) == 1.0
    assert np.max(indicators.electric_field) == 1.0
    assert indicators.combined[1] > indicators.combined[0]
    assert indicators.combined[2] > indicators.combined[3]


def test_3d_face_mesh_validation_cases_pass() -> None:
    results = run_all_3d_face_mesh_cases()

    assert len(results) == 19
    assert all(result.passed for result in results)
