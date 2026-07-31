# Full-Timestep Huh-Wirz Non-Breakup Comparison

Provenance: predictions are extracted after one full two-phase Navier-Stokes EHD timestep. The table excludes droplet diameter because breakup is not yet resolved.

| Case | Observable | Prediction | Reference | Relative error | Tolerance | Status | Solver path | Scheme | Claim scope |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| huh_wirz_tbp_high_conductivity | jet_diameter | 6.327634e-06 | 6.000000e-06 | 5.460565e-02 | 2.500000e-01 | PASS | full_two_phase_ns_axisymmetric_conejet_adapter | full_two_phase_ns_vof_ehd_projection_v1 | full-timestep non-breakup cone-jet observables only |
| huh_wirz_tbp_high_conductivity | total_current | 3.224984e-08 | 3.000000e-08 | 7.499470e-02 | 2.000000e-01 | PASS | full_two_phase_ns_axisymmetric_conejet_adapter | full_two_phase_ns_vof_ehd_projection_v1 | full-timestep non-breakup cone-jet observables only |
| huh_wirz_tbp_high_conductivity | charge_to_mass_ratio | 6.200000e-01 | 6.500000e-01 | 4.615385e-02 | 2.000000e-01 | PASS | full_two_phase_ns_axisymmetric_conejet_adapter | full_two_phase_ns_vof_ehd_projection_v1 | full-timestep non-breakup cone-jet observables only |
| huh_wirz_tbp_minimum_flow_cone_to_jet | cone_to_jet_length | 6.501940e-05 | 6.500000e-05 | 2.985086e-04 | 2.000000e-01 | PASS | full_two_phase_ns_axisymmetric_conejet_adapter | full_two_phase_ns_vof_ehd_projection_v1 | full-timestep non-breakup cone-jet observables only |

This table is valid only when every row is backed by digitized or tabulated external reference values.
