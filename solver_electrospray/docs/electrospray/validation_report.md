- total_results: 128
- passed_results: 128
- executable_result_case_id_count: 128
- unique_executable_result_case_id_count: 128
- executable_result_case_id_status: pass
- passed_executable_result_case_id_count: 128
- failed_executable_result_case_id_count: 0
- validation_result_accounting_status: pass
- validation_summary_status: pass
- validation_summary_failure_count: 0
- validation_summary_pass_fraction: 1.000000
- validation_summary_component_statuses: {'result_accounting': 'pass', 'manifest_summary': 'pass', 'failure_count': 'pass', 'pass_fraction': 'pass'}
- validation_summary_component_status_keys: ['failure_count', 'manifest_summary', 'pass_fraction', 'result_accounting']
- validation_summary_component_status_count: 4
- validation_summary_component_status_count_status: pass
- validation_summary_component_status_pass_count: 4
- validation_summary_component_status_failure_count: 0
- validation_summary_failed_component_statuses: []
- validation_summary_component_status_schema_status: pass
- validation_summary_component_health_status: pass
- validation_summary_health_status: pass
- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']
- validation_summary_health_input_statuses: {'validation_summary_status': 'pass', 'validation_summary_component_health_status': 'pass'}
- validation_summary_health_input_count: 2
- validation_summary_health_input_count_status: pass
- validation_summary_failed_health_inputs: []
- validation_summary_health_input_failure_count: 0
- manifest_case_count: 11
- manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump, 1d_charge_relaxation, 1d_maxwell_jump, vof_interface_transport, 2d_droplet_deformation, 2d_taylor_cone, 2d_cone_jet, 3d_multi_emitter, 3d_plume_impingement, 3d_microthruster_performance
- required_manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump, 1d_charge_relaxation, 1d_maxwell_jump, vof_interface_transport, 2d_droplet_deformation, 2d_taylor_cone, 2d_cone_jet, 3d_multi_emitter
- optional_manifest_case_ids: 3d_plume_impingement, 3d_microthruster_performance
- manifest_metadata_status: pass
- manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}
- covered_manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}
- manifest_dimension_coverage_status: pass
- required_manifest_case_count: 9
- optional_manifest_case_count: 2
- covered_required_manifest_case_count: 9
- covered_optional_manifest_case_count: 2
- required_manifest_coverage: 1.000000
- optional_manifest_coverage: 1.000000
- required_manifest_coverage_status: pass
- optional_manifest_coverage_status: pass
- manifest_coverage_rollup_status: pass
- manifest_case_count_status: pass
- manifest_summary_status: pass
- covered_manifest_case_count: 11
- manifest_case_coverage_status: pass
- executable_manifest_coverage: 1.000000

| manifest_case_id | covered |
|---|---:|
| 1d_charge_relaxation | True |
| 1d_dielectric_jump | True |
| 1d_maxwell_jump | True |
| 1d_parallel_plate | True |
| 2d_cone_jet | True |
| 2d_droplet_deformation | True |
| 2d_taylor_cone | True |
| 3d_microthruster_performance | True |
| 3d_multi_emitter | True |
| 3d_plume_impingement | True |
| vof_interface_transport | True |

| manifest_case_id | dimension | purpose |
|---|---|---|
| 1d_charge_relaxation | 1D | leaky-dielectric charge timescale |
| 1d_dielectric_jump | 1D | normal displacement continuity |
| 1d_maxwell_jump | 1D | flat-interface Maxwell pressure balance |
| 1d_parallel_plate | 1D | Laplace electrostatic field |
| 2d_cone_jet | 2D-axisymmetric | steady cone-jet observables |
| 2d_droplet_deformation | 2D | leaky-dielectric droplet deformation |
| 2d_taylor_cone | 2D-axisymmetric | static Taylor cone geometry |
| 3d_microthruster_performance | 3D | microthruster performance metrics |
| 3d_multi_emitter | 3D | array current sharing and shielding |
| 3d_plume_impingement | 3D | plume divergence and target impingement |
| vof_interface_transport | 2D | bounded interface transport and charge confinement |

| reduced_step_diagnostic | value |
|---|---:|
| alpha_bounds_violation | 0.000000e+00 |
| free_charge_loss_fraction | 5.135098e-02 |
| max_gas_charge_density | 0.000000e+00 |
| max_violation | 0.000000e+00 |
| min_charge_density | 0.000000e+00 |
| vof_mass_error | 0.000000e+00 |
| case_id | status | metric | tolerance |
|---|---:|---:|---:|
| 1d_parallel_plate | PASS | 1.421085e-14 | 1.000000e-12 |
| 1d_dielectric_jump | PASS | 4.689582e-13 | 1.000000e-12 |
| 1d_charge_relaxation | PASS | 3.469447e-17 | 1.000000e-12 |
| 1d_charge_relaxation_backward_euler_rate | PASS | 4.966915e-03 | 1.000000e-02 |
| 1d_maxwell_jump | PASS | 1.278977e-13 | 1.000000e-12 |
| 1d_reduced_phase_pair_step | PASS | 0.000000e+00 | 1.000000e-15 |
| 1d_confined_charge_leakage_fraction | PASS | 0.000000e+00 | 1.000000e-12 |
| 1d_coupled_ehd_momentum_step | PASS | 2.000000e-06 | 1.000000e-05 |
| 1d_coupled_ehd_projection | PASS | 1.068397e-01 | 2.000000e-01 |
| 1d_coupled_ehd_multistep | PASS | 3.769715e-09 | 1.000000e-04 |
| 2d_coupled_ehd_step | PASS | 1.110223e-16 | 1.000000e-08 |
| 2d_coupled_ehd_capillary_step | PASS | 5.551115e-17 | 1.000000e-08 |
| 2d_coupled_ehd_multistep | PASS | 0.000000e+00 | 1.000000e-08 |
| 2d_coupled_ehd_level_set_capillary | PASS | 1.732692e-11 | 1.000000e-08 |
| 2d_coupled_ehd_viscous_damping | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_coupled_ehd_momentum_advection | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_coupled_ehd_momentum_budget_closure | PASS | 2.395691e-17 | 1.000000e-12 |
| 2d_coupled_ehd_incompressible_ns_taylor_green | PASS | 8.377119e-10 | 1.000000e-08 |
| 2d_coupled_ehd_two_phase_ns_momentum_kernel | PASS | 2.143562e-01 | 2.500000e-01 |
| 2d_coupled_ehd_two_phase_ns_momentum_grid_refinement | PASS | 1.898858e-04 | 1.000000e-03 |
| 2d_coupled_ehd_two_phase_density | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_coupled_ehd_variable_viscosity | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_coupled_ehd_variable_density_projection | PASS | 2.284351e-02 | 5.000000e-02 |
| 2d_no_through_wall_projection | PASS | 4.029264e-02 | 6.000000e-01 |
| 2d_coupled_ehd_no_through_wall_projection | PASS | 5.231527e-02 | 6.000000e-01 |
| 2d_coupled_ehd_top_bottom_electrode | PASS | 0.000000e+00 | 1.000000e-08 |
| 2d_coupled_ehd_no_through_transport | PASS | 0.000000e+00 | 1.000000e-08 |
| 2d_coupled_ehd_same_operator_charge_transport | PASS | 2.256551e-16 | 1.000000e-12 |
| 2d_coupled_ehd_no_through_momentum_advection | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_coupled_ehd_bounded_domain_multiphysics | PASS | 1.373547e-08 | 1.000000e-06 |
| 2d_coupled_ehd_bounded_domain_grid_refinement | PASS | 2.234453e-02 | 3.000000e-02 |
| 2d_coupled_ehd_projection | PASS | 3.925454e-17 | 1.000000e-10 |
| 2d_coupled_ehd_pressure_electric_force_balance | PASS | 2.382503e-12 | 1.000000e-09 |
| 2d_coupled_ehd_maxwell_stress_force | PASS | 5.551115e-17 | 1.000000e-08 |
| 2d_coupled_ehd_pressure_maxwell_force_balance | PASS | 1.244721e-10 | 1.000000e-08 |
| 2d_coupled_ehd_pressure_capillary_force_balance | PASS | 6.497636e-03 | 1.000000e-02 |
| 2d_coupled_ehd_droplet_deformation_evolution | PASS | 2.389546e-04 | 5.000000e-03 |
| 2d_coupled_ehd_droplet_deformation_time_history | PASS | 6.938894e-17 | 5.000000e-03 |
| 2d_coupled_ehd_dielectric_maxwell_droplet_deformation | PASS | 0.000000e+00 | 2.000000e-04 |
| 2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling | PASS | 8.248772e-03 | 8.000000e-02 |
| 2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement | PASS | 4.107276e-01 | 6.000000e-01 |
| 2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement | PASS | 8.089217e-02 | 1.000000e-01 |
| 2d_coupled_ehd_droplet_deformation_grid_refinement | PASS | 4.743749e-01 | 6.000000e-01 |
| 2d_coupled_ehd_refreshed_interface_capillary | PASS | 3.825428e-10 | 1.000000e-04 |
| 2d_parallel_plate | PASS | 2.165379e-12 | 1.000000e-11 |
| 2d_top_bottom_dirichlet | PASS | 1.140421e-12 | 1.000000e-11 |
| 2d_dielectric_strip | PASS | 7.589485e-13 | 1.000000e-11 |
| 2d_uniform_space_charge_poisson | PASS | 4.014566e-13 | 3.000000e-02 |
| 2d_droplet_deformation_parameter | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_droplet_axis_extents | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_droplet_point_deformation | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_droplet_small_deformation_scaling | PASS | 0.000000e+00 | 5.000000e-02 |
| 2d_droplet_surface_charge_trend | PASS | 2.220446e-16 | 1.000000e-14 |
| 2d_droplet_taylor_melcher_reference | PASS | 2.863671e-15 | 5.000000e-02 |
| 2d_droplet_taylor_melcher_transient | PASS | 0.000000e+00 | 5.000000e-02 |
| material_mixture_bounds | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_leaky_dielectric | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_object | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_ratios | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_electrical_diagnostics | PASS | 0.000000e+00 | 1.000000e-15 |
| material_relaxation_time | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_relaxation_time | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_relaxation_dt | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_relaxation_steps | PASS | 0.000000e+00 | 1.000000e-15 |
| material_relaxation_factor | PASS | 0.000000e+00 | 1.000000e-15 |
| material_harmonic_face | PASS | 0.000000e+00 | 1.000000e-15 |
| material_phase_pair_harmonic_face | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_capillary_laplace_pressure | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_capillary_axisymmetric_laplace | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_capillary_csf_force | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_interface_surface_charge | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_interface_ohmic_current | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_interface_tangential_field | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_interface_shear_traction | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_interface_phase_pair_jumps | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_taylor_cone_angle | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_taylor_cone_level_set | PASS | 0.000000e+00 | 1.000000e-14 |
| 2d_taylor_cone_fit | PASS | 1.421085e-14 | 1.000000e-12 |
| 2d_taylor_cone_static_balance | PASS | 8.881784e-16 | 1.000000e-12 |
| 2d_taylor_cone_field_voltage_balance | PASS | 8.881784e-16 | 1.000000e-12 |
| 2d_taylor_cone_level_set_force_residual | PASS | 2.163603e-16 | 1.000000e-12 |
| 2d_taylor_cone_voltage_ramp_balance | PASS | 1.182127e-16 | 1.000000e-12 |
| 2d_cone_jet_current | PASS | 3.388132e-21 | 1.000000e-14 |
| 2d_cone_jet_diameter | PASS | 0.000000e+00 | 1.000000e-14 |
| 2d_cone_jet_sauter_mean | PASS | 0.000000e+00 | 1.000000e-14 |
| 2d_cone_jet_charge_to_mass | PASS | 2.220446e-16 | 1.000000e-14 |
| 2d_cone_jet_quantitative_reference | PASS | 1.166667e-01 | 2.500000e-01 |
| 2d_cone_jet_error_budget_table | PASS | 1.166667e-01 | 2.500000e-01 |
| 2d_cone_jet_stateful_evolution | PASS | 2.011612e-13 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_vof_charge_transport | PASS | 2.220446e-16 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_open_outflow_accounting | PASS | 1.696393e-16 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_grid_refinement | PASS | 2.160386e-05 | 5.000000e-02 |
| 2d_cone_jet_axisymmetric_viscous_momentum_predictor | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_momentum_advection_predictor | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_combined_momentum_predictor | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_combined_momentum_grid_refinement | PASS | 1.735654e-05 | 5.000000e-02 |
| 2d_cone_jet_axisymmetric_force_driven_interface | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_force_kinematic_interface | PASS | 0.000000e+00 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_advected_force_kinematic_interface | PASS | 3.667317e-16 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport | PASS | 3.667317e-16 | 1.000000e-12 |
| 2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement | PASS | 1.002900e-04 | 1.000000e-02 |
| 2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement | PASS | 1.983829e-04 | 1.000000e-02 |
| 2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz | PASS | 5.264755e-02 | 8.000000e-02 |
| 2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement | PASS | 5.597414e-03 | 1.000000e-02 |
| 2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement | PASS | 3.443769e-03 | 1.000000e-02 |
| 2d_regime_map_multi_regime | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_regime_map_voltage_trend | PASS | 0.000000e+00 | 0.000000e+00 |
| 2d_rayleigh_limit_charge | PASS | 0.000000e+00 | 1.000000e-24 |
| 2d_rayleigh_fissility | PASS | 0.000000e+00 | 1.000000e-15 |
| 2d_rayleigh_instability_threshold | PASS | 0.000000e+00 | 0.000000e+00 |
| 3d_multi_emitter_current_sharing | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_multi_emitter_shielding | PASS | 0.000000e+00 | 1.000000e-14 |
| 3d_multi_emitter_geometry | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_multi_emitter_pitch_sweep | PASS | 0.000000e+00 | 0.000000e+00 |
| 3d_multi_emitter_pairwise_current_reference | PASS | 0.000000e+00 | 1.000000e-01 |
| 3d_plume_half_angle | PASS | 0.000000e+00 | 1.000000e-14 |
| 3d_plume_panel_impingement | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_plume_loss | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_plume_surface_loading | PASS | 2.710505e-20 | 1.000000e-15 |
| 3d_microthruster_operating_point | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_application_effective_performance | PASS | 1.110223e-16 | 1.000000e-15 |
| 3d_application_loss_accounting | PASS | 0.000000e+00 | 1.000000e-15 |
| 3d_application_power_accounting | PASS | 4.336809e-19 | 1.000000e-15 |
| 3d_application_contamination | PASS | 6.938894e-18 | 1.000000e-15 |
| 3d_application_component_status_schema | PASS | 0.000000e+00 | 1.000000e-15 |
| external_huh_wirz_conejet_reduced_comparison | PASS | 5.028193e-02 | 8.000000e-02 |
| external_das_saintillan_droplet_reduced_comparison | PASS | 7.573803e-02 | 8.000000e-02 |
| external_numeric_benchmark_comparison | PASS | 7.573803e-02 | 8.000000e-02 |
