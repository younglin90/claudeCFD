# External Benchmark Numeric Comparison

Provenance: reference values are digitized or text-extracted from Das-Saintillan and Huh-Wirz. Predictions are current reduced validation-kernel outputs, not a claim of completed full two-phase Navier-Stokes CFD.

| Case | Observable | Prediction | Reference | Relative error | Tolerance | Status | Solver path | Scheme | Claim scope |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| huh_wirz_heptane_moderate_conductivity | droplet_diameter | 3.403561e-05 | 3.500000e-05 | 2.755387e-02 | 2.000000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| huh_wirz_tbp_high_conductivity | droplet_diameter | 1.050282e-05 | 1.000000e-05 | 5.028193e-02 | 2.000000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| huh_wirz_tbp_high_conductivity | jet_diameter | 6.201665e-06 | 6.000000e-06 | 3.361079e-02 | 2.500000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| huh_wirz_tbp_high_conductivity | total_current | 3.092885e-08 | 3.000000e-08 | 3.096164e-02 | 2.000000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| huh_wirz_tbp_high_conductivity | charge_to_mass_ratio | 6.200000e-01 | 6.500000e-01 | 4.615385e-02 | 2.000000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| huh_wirz_tbp_minimum_flow_cone_to_jet | cone_to_jet_length | 6.477766e-05 | 6.500000e-05 | 3.420653e-03 | 2.000000e-01 | PASS | axisymmetric_conejet_observable_stepper | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_transient_system_1b | deformation_parameter | -8.390757e-02 | -7.800000e-02 | 7.573803e-02 | 1.500000e-01 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_transient_system_1b | surface_charge_endpoint_difference | 1.000000e+00 | 1.000000e+00 | 2.220446e-16 | 2.000000e-01 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_transient_system_1c | deformation_parameter | -1.443106e-01 | -1.400000e-01 | 3.078978e-02 | 1.500000e-01 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_transient_system_1c | surface_charge_sign_change | 1.000000e+00 | 1.000000e+00 | 0.000000e+00 | 0.000000e+00 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_prolate_system_3 | deformation_parameter | 2.838335e-01 | 2.700000e-01 | 5.123537e-02 | 1.500000e-01 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |
| das_saintillan_steady_systems_2a_2b | deformation_parameter | -7.079345e-02 | -6.700000e-02 | 5.661866e-02 | 1.500000e-01 | PASS | advance_coupled_ehd_2d_phase_pair | charge_conservative_vof_ehd_reduced_stepper_v1 | reduced-kernel comparison only |

This table is valid only when every row is backed by digitized or tabulated external reference values.
