# Iter 644 ACID term audit

Scope: diagnostic documentation for `solver/denner_1d`, plus mapping to the
passive `DENNER_ACID_TERM_EQUATION_AUDIT` JSONL rows. This document does not
define validation criteria and does not authorize production numerical changes.

Primary paper: Denner, Xiao, and van Wachem, "Pressure-based algorithm for
compressible interfacial flows with acoustically-conservative interface
discretisation", Journal of Computational Physics 367, 192-234, 2018,
DOI `10.1016/j.jcp.2018.04.028`.

Public landing pages used for this audit:

- ScienceDirect DOI page: https://www.sciencedirect.com/science/article/pii/S0021999118302535
- ResearchGate open text/PDF page: https://www.researchgate.net/publication/324606897_Pressure-based_algorithm_for_compressible_interfacial_flows_with_acoustically-conservative_interface_discretisation

## 1. Denner ACID equations and concepts

The paper goal relevant to this workspace is an interface discretisation that
keeps acoustic reflection/transmission at material interfaces consistent with
local Rankine-Hugoniot (RH) pressure and velocity jumps, while keeping the
pressure-based finite-volume system conservative.

Terms to track:

- Mixture density relation: density is evaluated from an EOS and composition.
  In volume-fraction form, local mixture density is a composition-weighted
  thermodynamic quantity, not a free primitive.
- Sound speed relation: the material interface needs a unique effective speed
  of sound for acoustic propagation and pressure-velocity coupling. The code's
  two-species Kapila/Denner form also uses phase bulk moduli `rho_k c_k^2`.
- Material compressibility factor `K`: for two species, the local source uses
  a coefficient equivalent to
  `K_1 = alpha_1 (1-alpha_1) (Z_2-Z_1)/(alpha_1 Z_2+(1-alpha_1)Z_1)`,
  where this code convention uses `Z_k = rho_k c_k^2` inside the alpha source.
- Interface RH pressure/velocity conditions: at a material face, the acoustic
  interface relation should couple a common pressure trace and normal velocity
  trace from the left/right impedances. In 1D diagnostic form, the two-impedance
  pressure and velocity target is:
  `u_f = (p_L-p_R+Z_L u_L+Z_R u_R)/(Z_L+Z_R)` and
  `p_f = (Z_R p_L+Z_L p_R+Z_L Z_R (u_L-u_R))/(Z_L+Z_R)`.
  Here `Z = rho c` is acoustic impedance.
- ACID density/enthalpy/sound-speed construction at an interface: ACID uses a
  face density/sound-speed construction that is consistent with acoustic
  interface propagation. In this code, the implemented ACID density is the
  Denner-style face density formula from local `rho` and `c`.
- Where terms should enter:
  - Continuity: `rho_face * theta_face` mass flux and pressure-density
    compressibility contributions.
  - Momentum: pressure face term `pface` and convective momentum flux.
  - Energy: `rho h` or `H` carrier in the advective energy flux, plus
    pressure-work/source terms.
  - MWI/Rhie-Chow: face velocity `theta_f = ubar_f + pressure correction +
    transient/history correction`, with the pressure coefficient using a
    consistent face density/momentum diagonal.
  - Pface: pressure face reconstruction should satisfy pressure/velocity
    RH compatibility at material interfaces.

## 2. Current implementation mapping

### Face density and MWI coefficient

- `solver/denner_1d/flux/mwi.py:11`: `acid_face_density(...)` computes
  `rho_face`.
- `solver/denner_1d/flux/mwi.py:45`: ACID face density numerator is
  `rho_L*rho_R*(rho_L*c_L + rho_R*c_R)`.
- `solver/denner_1d/flux/mwi.py:46`: ACID denominator is
  `rho_L^2*c_L + rho_R^2*c_R`.
- `solver/denner_1d/flux/mwi.py:49`: material interfaces use ACID density,
  smooth regions use harmonic density.
- `solver/denner_1d/flux/mwi.py:139`: `mwi_face_coeff_denner(...)` computes
  the Denner/Rhie-Chow pressure-gradient coefficient from momentum diagonal,
  face density, `dx`, and `dt`.

### MWI face velocity components

- `solver/denner_1d/flux/mwi.py:324`: `mwi_face_velocity_components(...)`
  documents the decomposed face velocity.
- `solver/denner_1d/flux/mwi.py:333`: code comment states
  `u_face = u_arith_f - d_hat_f * (p_{i+1}-p_i)/dx + transient_correction_f`.
- `solver/denner_1d/flux/mwi.py:378`: transient correction uses old face
  velocity and old face density when enabled.
- `solver/denner_1d/assembly.py:2959`: `_store_acid_face_property_capture(...)`
  passively captures production face quantities for accepted-step audits.
- `solver/denner_1d/assembly.py:3094`: captured `theta_flux_prod` source label
  is `theta_flux_k`.
- `solver/denner_1d/assembly.py:3100`: captured `mwi_d_hat`.
- `solver/denner_1d/assembly.py:3102`: captured `mwi_u_bar`.
- `solver/denner_1d/assembly.py:3104`: captured production pressure jump.
- `solver/denner_1d/assembly.py:3108`: captured total pressure correction.

### Pface and pressure flux

- `solver/denner_1d/assembly.py:3092`: captured `pface_prod`.
- `solver/denner_1d/assembly.py:3093`: source label is
  `momentum_pressure_face`.
- `solver/denner_1d/solver_a.py:2721`: new passive audit
  `_emit_acid_term_equation_rows(...)` compares this production `pface` to a
  local two-impedance RH diagnostic target.

### Energy carrier

- `solver/denner_1d/assembly.py:3086`: captured `H_acid_face_prod`.
- `solver/denner_1d/assembly.py:3089`: captured/rebuilt `c_face_prod`.
- `solver/denner_1d/assembly.py:3091`: captured/rebuilt `Z_face_prod`.
- `solver/denner_1d/assembly.py:3092`: iter645 adds exact/source-labeled aliases
  `c_face_prod_exact`, `c_face_prod_exact_source`, `Z_face_prod_exact`, and
  `Z_face_prod_exact_source`. Source labels distinguish
  `production_face_context` from `exact_rebuild_from_production_inputs`.
- `solver/denner_1d/assembly.py:3098`: captured mass flux source is
  `rho_face_prod_times_theta_flux_prod`.
- `solver/denner_1d/assembly.py:3099`: captured `rhoh_flux_prod`.
- `solver/denner_1d/solver_a.py:2918`: new passive audit computes
  `rhoh_flux_condition_defect = rhoh_flux_prod - rho_paper*H_paper*theta_prod`.

### K source in alpha/composition equation

- `solver/denner_1d/vof_cn.py:18`: `_compute_K(...)` implements the two-species
  Denner/Kapila compressibility factor.
- `solver/denner_1d/vof_cn.py:35`: denominator uses
  `Z2/(1-alpha) + Z1/alpha` in the two-species wrapper.
- `solver/denner_1d/vof_cn.py:36`: two-species `K = (Z2-Z1)/(den+eps)`.
- `solver/denner_1d/vof_cn.py:93`: `_compute_K_Ns(...)` documents the
  generalized coefficient.
- `solver/denner_1d/vof_cn.py:105`: `K_k = psi_k * (rho c^2/(rho_k c_k^2) - 1)`.
- `solver/denner_1d/vof_cn.py:110`: comment gives the two-species reduction.
- `solver/denner_1d/vof_cn.py:272`: alpha equation source form is documented.
- `solver/denner_1d/vof_cn.py:275`: source uses `(psi + K_val) * div(u)`.
- `solver/denner_1d/vof_cn.py:375`: RK-stage K diagnostics can return
  `K_stage1`, `K_stage2`, and `K_stage_average`.
- `solver/denner_1d/solver_a.py:10900`: iter645 makes
  `DENNER_ACID_TERM_EQUATION_AUDIT` request the production VOF RK-stage K
  diagnostics, so the acid-term audit reports `K_prod` from the accepted stage
  history rather than replaying paper K as production.
- `solver/denner_1d/solver_a.py:11062`: accepted alpha/K diagnostic context
  carries `K_stage1`, `K_stage2`, `K_stage_average`, `K_prod_current`, and
  source labels such as `vof_stage_average` or `mass_fraction_model_zero`.

### Accepted post-step hooks

- `solver/denner_1d/solver_a.py:13358`: accepted time is computed after the
  accepted step.
- `solver/denner_1d/solver_a.py:13370`: existing ACID/RH accepted audit hook.
- `solver/denner_1d/solver_a.py:13414`: coupled ACID/RH bounded relax audit hook.
- `solver/denner_1d/solver_a.py:13418`: new acid-term equation audit hook.

## 3. Mismatch table

| Paper term | Implementation term | Status | Evidence |
| --- | --- | --- | --- |
| ACID interface density | `acid_face_density(...)` | approximate implemented | Direct ACID formula in `flux/mwi.py:45-49`; audit emits `rho_paper` vs `rho_prod`. |
| Unique sound speed / impedance | `c_face_prod`, `Z_face_prod` capture | approximate / source-dependent | Capture can derive `c` from `Z/rho_face`; audit labels `c_prod_source` and `Z_prod_source`. |
| Material compressibility `K` | `_compute_K`, `_compute_K_Ns` | implemented for alpha source | `vof_cn.py:18`, `vof_cn.py:93`, and source use at `vof_cn.py:275`. |
| Interface pressure RH condition | `pface_prod` | approximate / potentially mismatched | Iter630-633 partition audits showed sink/coupling questions; new audit emits `pface_paper_condition_defect`. |
| Interface velocity RH condition | `theta_flux_prod` | approximate / potentially mismatched | Iter633/638 found bounded theta relaxation often lambda zero; new audit emits `theta_paper_condition_defect`. |
| ACID enthalpy face carrier | `H_acid_face_prod` | implemented but needs source-label audit | Capture at `assembly.py:3086`; new audit uses a clearly labeled proxy `H_paper`, not a claimed exact paper formula. |
| Energy-row enthalpy carrier | `H_prod_inline`, `F_energy_prod_inline` | exact passive capture | Iter647 adds `DENNER_ACID_ENTHALPY_CARRIER_EXACT_AUDIT`; values are captured from the inline energy-row face contribution path and labeled `energy_row_face_contribution`. |
| Energy `rhoh` consistency | `rhoh_flux_prod` | approximate | Capture at `assembly.py:3099`; new audit emits `rhoh_flux_condition_defect`. |
| MWI pressure correction | `mwi_d_hat`, `mwi_u_bar`, pressure correction | implemented, transient partly unresolved | Capture at `assembly.py:3100-3109`; transient source remains `unavailable_not_separated`. |
| MWI transient split | `mwi_transient_term_source` | missing separation | Capture labels transient as unavailable. Iter637 already found transient path/source ambiguity. |

## 4. Previous iteration evidence

- Iter628 captured production ACID face properties so accepted-post-step audits
  can compare actual production `rho_face`, `H`, `pface`, and `theta`.
- Iter630 found characteristic partition rows can be emitted from accepted
  material faces.
- Iter631 separated production pface and theta effects but still relied on
  approximate ideal two-impedance targets.
- Iter633 and iter638 showed bounded theta/transient relaxations can be blocked
  by local constraints and often collapse to lambda near zero.
- Iter641 bounded coupled ACID/RH relaxation audit emitted finite rows, but the
  N50 sample had `lambda_max=0` and no survivor under the conservative gates.
- Iter642 and iter643 added a standalone diagnostic micro-probe. Probe-only
  active relaxation also clamped to zero because the local velocity bound forced
  the target back to production.

## 5. Ranked redesign hypotheses

1. Coupled interface state must be redesigned as a single pface/theta/H/rho
   bundle. Component-only theta or H changes tend to either no-op under local
   bounds or trade pressure amplitude for velocity/correlation regressions.
2. The production MWI transient term is still not decomposed enough at material
   faces. A redesign should first expose exact transient components rather than
   use measured remainder or guessed splits.
3. The energy carrier and pface/theta conditions should be audited together.
   `rhoh_flux_condition_defect`, `pface_paper_condition_defect`, and
   `theta_paper_condition_defect` should identify whether the largest mismatch
   is thermodynamic, pressure, velocity, or coupled.
4. K/source consistency is likely necessary but not sufficient. The K source is
   implemented, but prior active pressure-compressibility replacement was
   unstable, so any redesign must preserve full residual form and conditioning.
5. Standalone probe harnesses are useful for rapid interface-transfer screening,
   but any active production candidate still needs normal case 07 N50/N200
   gates because probe-local bounds can hide packet-shape regressions.

## 6. Iter644 JSONL audit fields

Enable:

```bash
DENNER_ACID_TERM_EQUATION_AUDIT=1
DENNER_ACID_TERM_EQUATION_JSONL=/path/file.jsonl
```

Rows:

- `row_type="acid_term_equation"`
- `row_type="acid_term_equation_aggregate"`

Key fields:

- `rho_paper`, `rho_prod`
- `H_paper`, `H_prod`
- `c_paper`, `c_prod`
- `Z_paper`, `Z_prod`
- `K_paper`, `K_prod`
- `K_stage1`, `K_stage2`, `K_stage_average`
- `K_prod_source`, `K_stage_source`
- `theta_paper_condition_defect`
- `pface_paper_condition_defect`
- `rhoh_flux_condition_defect`
- `mwi_condition_defect`
- source labels for each paper/prod term
- `mapping_status`, `missing_terms`, `measured_remainder=false`

Important caveat: `H_paper` and `c_Z_paper` are diagnostic proxies in this
round. They are explicitly source-labeled and must not be used as active
replacement formulas without a follow-up derivation.

Iter645 source-label refinement:

- `c_face_prod_source="production_face_context"` means the value was captured
  as a face sound speed from assembly.
- `c_face_prod_source="exact_rebuild_from_production_inputs"` means the value
  was rebuilt from captured production face impedance and production face
  density, not invented from a paper formula.
- `Z_face_prod_source="production_face_context"` means assembly supplied a
  face impedance directly.
- `Z_face_prod_source="exact_rebuild_from_production_inputs"` means assembly
  supplied the same impedance ingredients used by the production pface/MWI
  context.
- `K_prod_source` identifies the production alpha-update source. Expected
  values include `vof_step_recomputes_K_at_RK_stages` for volume-fraction mode
  with K enabled and `mass_fraction_model_zero` for mass-fraction transport.

Iter647 exact enthalpy-carrier capture:

Enable:

```bash
DENNER_ACID_ENTHALPY_CARRIER_EXACT_AUDIT=1
DENNER_ACID_ENTHALPY_CARRIER_EXACT_JSONL=/path/file.jsonl
```

If `DENNER_ACID_ENTHALPY_CARRIER_EXACT_JSONL` is omitted, the audit appends to
`DENNER_ACID_TERM_EQUATION_JSONL` when that path is set.

Rows:

- `row_type="acid_enthalpy_carrier_exact"`
- `row_type="acid_enthalpy_carrier_exact_aggregate"`

Key fields:

- `H_prod_inline`
- `H_prod_inline_source="energy_row_face_contribution"`
- `theta_prod_inline`
- `F_energy_prod_inline`
- `rhoh_flux_prod_inline`
- `H_poststep_proxy`
- `H_candidate_acid`
- `H_candidate_source`
- `delta_H_inline_vs_candidate`
- `delta_F_energy_inline_vs_candidate`
- `rhoh_flux_condition_defect_exact`
- `mapping_status`
- `missing_terms`
- `measured_remainder=false`

Source-label caveat: this audit captures the exact production energy-row
carrier and flux contribution, but the current `H_candidate_acid` is still
labeled `proxy` unless a future derivation supplies a safe Denner-exact
enthalpy candidate. Proxy candidates must not be treated as active replacement
formulas.

Iter649 bounded carrier relaxation experiment (rejected/discarded):

```bash
DENNER_ACID_ENTHALPY_CARRIER_BOUNDED_RELAX=1
DENNER_ACID_ENTHALPY_CARRIER_BOUNDED_RELAX_JSONL=/path/file.jsonl
```

This active experiment was rejected and hard-disabled after catastrophic N50
regression. It is not an available active method. The rejected test changed only
the material-interface energy-row enthalpy carrier toward the existing
proxy-labeled ACID enthalpy direction, but N50 produced extreme pressure and
velocity amplification. The environment flag may remain recognized by diagnostic
plumbing, but assembly now forces the active gate off so it cannot alter solver
numerics.

Iter650 bound-scale diagnostic:

```bash
DENNER_ACID_ENTHALPY_CARRIER_BOUND_SCALE_AUDIT=1
DENNER_ACID_ENTHALPY_CARRIER_BOUND_SCALE_JSONL=/path/file.jsonl
```

Rows:

- `row_type="acid_enthalpy_carrier_bound_scale"`
- `row_type="acid_enthalpy_carrier_bound_scale_aggregate"`

Purpose: quantify why the iter649 carrier relaxation was unsafe and whether a
nonzero locally safe direction remains. This is diagnostic-only and does not
re-enable the rejected active path.

Key fields:

- `H_prod_inline`: exact captured production energy-row carrier.
- `H_target`: proxy-labeled ACID direction, not an exact replacement formula.
- `H_active_iter649_preview`: preview using `lambda_safe_min`, not production.
- `F_energy_prod_inline`, `F_energy_target`, `delta_F_energy_target`.
- `rhoh_defect_prod`, `rhoh_defect_target`.
- `H_min_lr`, `H_max_lr`: local left/right phase carrier interval.
- `incident_acoustic_energy_flux`, `pressure_work_scale`,
  `temporal_energy_source_scale`, `impedance_acoustic_scale`.
- `lambda_rhoh_defect_zero`, `lambda_H_lr_bound`,
  `lambda_acoustic_energy_bound`, `lambda_pressure_work_bound`,
  `lambda_impedance_bound`, `lambda_safe_min`.

Iter651 coupled pressure-energy residual-form audit:

```bash
DENNER_COUPLED_PRESSURE_ENERGY_RESIDUAL_FORM_AUDIT=1
DENNER_COUPLED_PRESSURE_ENERGY_RESIDUAL_FORM_JSONL=/path/file.jsonl
```

If `DENNER_COUPLED_PRESSURE_ENERGY_RESIDUAL_FORM_JSONL` is omitted, the audit
falls back to the enthalpy-carrier exact JSONL path, then the ACID term-equation
JSONL path. Enabling this audit also enables passive ACID face-property capture.

Rows:

- `row_type="coupled_pressure_energy_residual_form"`
- `row_type="coupled_pressure_energy_residual_form_aggregate"`

Purpose: combine the exact inline energy-row enthalpy carrier defect with a
passive pressure-work/RH pressure preview to determine whether a coupled
pressure-energy residual replacement is separable enough for a future active
method. This audit is diagnostic-only.

Key fields:

- `H_defect`: captured production inline carrier minus proxy-labeled ACID
  enthalpy direction.
- `rhoh_flux_defect_exact`: exact production inline energy flux contribution
  minus the proxy carrier flux using captured production `theta`.
- `pressure_work_defect`: captured production `pface` minus a local
  two-impedance RH pressure preview, multiplied by captured production `theta`.
- `elastic_balance_defect`: currently `null`; exact elastic storage separation
  is not available in this audit.
- `coupled_defect_prod`: `rhoh_flux_defect_exact + pressure_work_defect` when
  both preview terms are finite.
- `coupled_defect_candidate`, `delta_R_coupled`, `delta_A_coupled`,
  `delta_rhs_coupled`: currently `null` because the active residual/RHS/Jacobian
  form is not separable from the accepted post-step capture.
- `rhs_separable=false`, `jacobian_separable=false`,
  `active_form_available=false`.
- `dominant_blocker`: usually `rhs_jacobian_not_separable` for finite preview
  rows, or `missing_required_terms` when production capture/EOS inputs are
  unavailable.
- `missing_terms`: includes explicit blockers such as
  `elastic_storage_exact_separation_unavailable`,
  `coupled_residual_rhs_nonseparable`, and
  `coupled_jacobian_nonseparable`.
- `measured_remainder=false`.

Source-label caveat: `H_target` remains the proxy-labeled ACID direction from
the exact carrier audit, and `pface_RH_preview` is a diagnostic two-impedance
preview. Neither is labeled as a production replacement formula. No residual,
RHS, Jacobian, MWI, pface, pressure-work, alpha, CFL, or validation behavior is
changed by this flag.
- `dominant_bound`, `candidate_safe_nonzero`, `missing_fields`.
- `measured_remainder=false`.

All lambda values are local inequality projections. Missing bounds are reported
explicitly; no measured remainder or tuned coefficient is used.

Iter652 interface alpha-smearing audit:

```bash
DENNER_INTERFACE_ALPHA_SMEARING_AUDIT=1
DENNER_INTERFACE_ALPHA_SMEARING_JSONL=/path/file.jsonl
```

Rows:

- `row_type="interface_alpha_smearing"`

Purpose: accepted-post-step aggregate diagnostic for correlating material
interface alpha smearing with local acoustic energy loss. The audit emits one
compact row per sampled accepted step and does not instrument Newton
intermediate states.

Default sampled steps are `1,100,200,300,400,500,550,580,600,609,650,700`.
Override with `DENNER_INTERFACE_ALPHA_SMEARING_STEPS`.

Key fields:

- `interface_center`: gradient-weighted face center when alpha gradients are
  present; otherwise transition-cell mean if available.
- `alpha_min`, `alpha_max`, `alpha_tv`, `max_abs_grad_alpha`.
- `transition_cell_count_001_099`, `transition_cell_count_005_095`,
  `transition_cell_count_010_090`.
- `alpha_width_second_moment`, `alpha_width_change_from_prev`,
  `alpha_tv_change_from_prev`.
- `alpha_mass_left`, `alpha_mass_right`.
- `E_global`, `E_interface_band`, `E_right_material_side` using the same
  window-mean acoustic energy convention as accepted acoustic-energy history:
  `0.5*(p_prime^2/Z + Z*u_prime^2)` with `Z=rho*c`.
- `dE_global_from_prev`, `dE_interface_from_prev`, `interface_sink_ratio`.
- `sink_correlates_with_alpha_width`: true only when interface energy decreases
  while alpha width increases between sampled accepted rows.
- `K_prod`, `K_source`, and `alpha_source_Kdivu` when the accepted alpha/K
  production diagnostic context exposes them.
- `alpha_flux_left`, `alpha_flux_right`, `alpha_flux_divergence`, and
  `alpha_update_norm` remain `null` unless exact accepted alpha update flux
  values are available; missing terms are explicit in `missing_fields`.
- `measured_remainder=false`.

No alpha update, CICSAM/MSTACS/THINC, residual, flux, MWI, pface, CFL, mesh, or
validation behavior is changed by this audit.

Iter654 acoustic modified-equation/reconstruction/flux dissipation audit:

```bash
DENNER_ACOUSTIC_MODIFIED_EQUATION_AUDIT=1
DENNER_ACOUSTIC_MODIFIED_EQUATION_JSONL=/path/file.jsonl
```

Rows:

- `row_type="acoustic_modified_equation"`

Purpose: accepted-post-step diagnostic for local acoustic reconstruction/flux
dissipation proxies near the material interface. It compares accepted cell
`w+=p+Z*u` states with central and upwind two-point previews, and compares those
against captured production face `w+` formed from `pface_prod + Zface*theta`.

Default sampled steps are `1,50,100,200,500,550,580,600,609,650,700`.
Override with `DENNER_ACOUSTIC_MODIFIED_EQUATION_STEPS`.

Key fields:

- `wplus_L`, `wplus_R`, `wplus_face_prod`, `wplus_central_face`,
  `wplus_upwind_face`.
- `wplus_reconstruction_delta`: production face `w+` minus central preview when
  captured pface/theta are available.
- `limiter_name`: accepted solver limiter name when in scope.
- `limiter_phi`: currently `null`; exact limiter phi is not captured in the
  accepted state. `missing_fields` includes
  `limiter_phi_exact_unavailable`.
- `smoothness_r`: accepted-cell neighbor ratio preview, source-labeled
  `accepted_cell_wplus_neighbor_ratio`; this is not labeled as exact production
  limiter state.
- `local_wave_speed`, `local_cfl_acoustic`.
- `numerical_viscosity_recon`, `numerical_viscosity_flux`,
  `numerical_viscosity_total`: local accepted-state central/upwind proxy values,
  not production modified-equation coefficients.
- `dispersion_proxy`, `dissipation_proxy`, `energy_sink_proxy`.
- `operator_dominant`.
- `face_class`: `material`, `material_adjacent`, or
  `same_material_acoustic` for a compact stencil around material faces.
- `measured_remainder=false`.

No production residual, flux, reconstruction, limiter, MWI, pface, alpha, time
integration, CFL, mesh, or validation behavior is changed by this audit.

Iter656 reconstruction limiter provenance audit:

```bash
DENNER_RECON_LIMITER_PROVENANCE_AUDIT=1
DENNER_RECON_LIMITER_PROVENANCE_JSONL=/path/file.jsonl
```

Rows:

- `row_type="recon_limiter_provenance"`

Purpose: accepted-post-step diagnostic for provenance of the production
reconstruction branch near the material interface. Enabling this audit turns on
passive face-property capture and stores exact reconstructed face arrays already
computed in assembly's face-context path.

Default sampled steps are `1,50,100,200,500,550,580,600,609,650,700`.
Override with `DENNER_RECON_LIMITER_PROVENANCE_STEPS`.

Key fields:

- `branch`: source branch, currently `assembly_face_context` when exact arrays
  are captured.
- `limiter_name_p`, `limiter_name_u`, `limiter_name_T`,
  `limiter_name_face`.
- `wplus_cell_L/R`, `wminus_cell_L/R`.
- `wplus_face_L_exact`, `wplus_face_R_exact`,
  `wminus_face_L_exact`, `wminus_face_R_exact`: exact arrays captured from the
  production face-context reconstruction path.
- `wplus_face_exact_average`, `wminus_face_exact_average`.
- `wplus_face_prod_proxy`: final production face proxy formed from captured
  `pface_prod + Zface*theta_flux_prod`.
- `limiter_recon_delta_exact`: exact reconstructed `wplus` face average minus
  central accepted-cell `wplus`.
- `proxy_delta_from_iter655`: final production proxy minus central accepted-cell
  `wplus`.
- `proxy_matches_exact`: whether the final face proxy delta matches the exact
  reconstruction delta within roundoff-scale tolerance.
- `material_fallback_active`, `material_fallback_reason`.
- `characteristic_pface_active` when the exact face-context mask was captured.
- `limiter_phi_*`, `smoothness_r_*`, and primitive `slope_*` are currently
  `null` unless production exposes those internals at a future capture boundary.
  Missing reasons are explicit in `missing_fields`.
- `measured_remainder=false`.

No reconstruction formula, limiter behavior, flux, pface, MWI, alpha, time
integration, CFL, mesh, or validation behavior is changed by this audit.

Iter657 exact limiter source audit:

```bash
DENNER_RECON_LIMITER_SOURCE_AUDIT=1
DENNER_RECON_LIMITER_SOURCE_JSONL=/path/file.jsonl
```

Rows:

- `row_type="recon_limiter_source"`
- `row_type="recon_limiter_source_aggregate"`

Purpose: capture exact raw one-sided differences, implied smoothness ratios,
implied limiter phi values, and limited slopes at the assembly source function
boundary where production reconstruction uses `_dominant_slopes_array`.

Default sampled steps are `1,50,100,200,500,550,580,600,609,650,700`.
Override with `DENNER_RECON_LIMITER_SOURCE_STEPS`.

Variables:

- `p`
- `u`
- `T`
- `wplus`
- `wminus`

Key fields:

- `variable`
- `branch`
- `limiter_name`
- `r_left_exact`, `r_right_exact`: raw accepted production ratios computed from
  the exact one-sided differences at the production slope boundary.
- `phi_left_exact`, `phi_right_exact`: implied limiter phi from
  `limited_slope / forward_difference` where the denominator is finite.
- `slope_raw_left`, `slope_raw_right`
- `slope_limited_left`, `slope_limited_right`
- `cell_value_left`, `cell_value_right`
- `face_value_left`, `face_value_right`, `prod_exact_face_average`
- `material_fallback_active`, `fallback_reason`
- `source_function="assembly._dominant_slopes_array"`
- aggregate `branch_counts`, `fallback_reason_counts`, and
  `clipping_counts_by_variable`
- `measured_remainder=false`

The phi values are exact implied values for the limited slopes actually returned
by production's vectorized slope helper. They are not separately recomputed from
an independent limiter formula. No limiter, slope, reconstruction, flux, pface,
MWI, alpha, time integration, CFL, mesh, or validation behavior is changed.

## Iter660 flux-upwind dissipation provenance audit

Passive flag:

```bash
DENNER_FLUX_UPWIND_DISSIPATION_PROVENANCE_AUDIT=1
DENNER_FLUX_UPWIND_DISSIPATION_JSONL=/path/file.jsonl
```

Default sampled accepted steps are `1,50,100,200,500,550,580,600,609,650,700`.
Override with `DENNER_FLUX_UPWIND_DISSIPATION_STEPS`.

Rows:

- `row_type="flux_upwind_dissipation_provenance"`
- `row_type="flux_upwind_dissipation_provenance_aggregate"`

Purpose: compare exact accepted production upwind flux carriers against a local
two-point central preview using only accepted production state. This is a
diagnostic for same-material acoustic/material-adjacent/material interface
controls near the interface/packet; it does not alter upwind selectors,
production fluxes, pface, MWI, alpha, time integration, reconstruction, CFL,
mesh, or validation.

Production-source fields:

- `F_mass_prod` from captured `rho_face_prod * theta_flux_prod`
- `F_momentum_prod` from captured mass flux, exact upwind velocity, and pface
- `F_energy_prod` from inline energy-row face contribution when captured, with
  `theta * H_acid_face` fallback only when the inline field is absent
- `upwind_side` from the exact production `theta_flux` sign
- `u_up_prod` from the exact production theta-upwind velocity capture
- `H_up_prod`, `H_left_face_prod`, and `H_right_face_prod` from the inline
  energy carrier capture

Central-preview fields:

- `F_mass_central`
- `F_momentum_central`
- `F_energy_central`
- `H_central`

Central values are accepted-state local two-point previews, not exact/reference
data and not production replacements. Missing fields are explicit. All rows set
`measured_remainder=false`.

## Iter662 same-material acoustic momentum flux bound-scale audit

Passive flag:

```bash
DENNER_SMA_MOMENTUM_FLUX_BOUNDS_AUDIT=1
DENNER_SMA_MOMENTUM_FLUX_BOUNDS_JSONL=/path/file.jsonl
```

Default sampled accepted steps are `1,50,100,200,500,550,580,600,609,650,700`.
Override with `DENNER_SMA_MOMENTUM_FLUX_BOUNDS_STEPS`.

Rows:

- `row_type="sma_momentum_flux_bounds"`
- `row_type="sma_momentum_flux_bounds_aggregate"`

Purpose: preview a same-material acoustic momentum convective carrier using the
fixed production mass flux and central face velocity:

```text
Fmom_candidate = mass_flux_prod * 0.5*(u_L+u_R) + pface_prod
```

The audit does not replace momentum flux, pressure flux, pface, MWI, alpha,
energy carrier, reconstruction, time integration, CFL, mesh, or validation.

Key fields:

- `mass_flux_prod`, `u_up_prod`, `u_bar`
- `Fmom_prod`, `Fmom_candidate`, `delta_Fmom`
- `momentum_flux_bound_scale`, `delta_over_bound`
- `local_momentum_residual_scale`, `delta_over_residual`
- `pressure_velocity_partition_nonworse`
- `highk_nonworse`
- `opposite_energy_nonworse`
- `candidate_safe_nonzero`
- `reject_reason`

`candidate_safe_nonzero` can only be true for same-material acoustic faces.
Material, material-adjacent, boundary, and special controls are fallback/control
rows. High-k, partition, and opposite-characteristic checks are local
accepted-state previews, not exact/reference comparisons. Missing fields are
explicit and all rows set `measured_remainder=false`.

## Iter663 boundary characteristic reflection audit

Passive flag:

```bash
DENNER_BOUNDARY_CHARACTERISTIC_REFLECTION_AUDIT=1
DENNER_BOUNDARY_CHARACTERISTIC_REFLECTION_JSONL=/path/file.jsonl
```

Default sampled accepted steps are
`1,100,200,300,400,500,550,580,600,609,650,700,760,800,810`.
Override with `DENNER_BOUNDARY_CHARACTERISTIC_REFLECTION_STEPS`.

Rows:

- `row_type="boundary_characteristic_reflection"`
- `row_type="boundary_characteristic_reflection_aggregate"`

Characteristic orientation:

- Left boundary: outgoing characteristic is `w- = p' - Z u'`; incoming is
  `w+ = p' + Z u'`.
- Right boundary: outgoing characteristic is `w+`; incoming is `w-`.

Boundary `p/u/Z/c` are currently accepted adjacent-cell proxies, explicitly
source-labeled as `accepted_post_step_adjacent_cell_no_boundary_face_capture`.
No boundary condition, boundary face state, flux, pface, MWI, reconstruction,
time integration, CFL, mesh, or validation behavior is changed. No exact or
reference profile is used. All rows set `measured_remainder=false`.

## Iter664 true boundary face-state capture audit

Passive flag:

```bash
DENNER_BOUNDARY_FACE_STATE_CAPTURE_AUDIT=1
DENNER_BOUNDARY_FACE_STATE_CAPTURE_JSONL=/path/file.jsonl
```

Default sampled accepted steps match the boundary reflection audit:
`1,100,200,300,400,500,550,580,600,609,650,700,760,800,810`.
Override with `DENNER_BOUNDARY_FACE_STATE_CAPTURE_STEPS`.

Rows:

- `row_type="boundary_face_state_capture"`
- `row_type="boundary_face_state_capture_aggregate"`

Purpose: capture production boundary face values from the accepted assembly face
payload when available:

- `p_face_prod` / `pface_boundary_prod`
- `u_face_prod` / `theta_face_prod`
- `rho_face_prod`, `c_face_prod`, `Z_face_prod`
- `mass_flux_face_prod`
- `momentum_flux_face_prod`
- `energy_flux_face_prod`

The audit also reports a face-based characteristic reflection ratio and the
adjacent-cell proxy ratio from iter663 for comparison. Source labels distinguish
exact assembly face capture from adjacent-cell proxy comparison fields. Missing
captured face fields are explicit. No boundary condition, flux, pface, MWI,
reconstruction, time integration, CFL, mesh, or validation behavior is changed.
All rows set `measured_remainder=false`.

## Discarded active material-interface characteristic reconstruction

Iter659 tested:

```bash
DENNER_MATERIAL_INTERFACE_CHAR_RECON_BOUNDED=1
```

The active candidate replaced only the material-interface acoustic `p/u` trace
with a bounded two-impedance interface state while leaving same-material faces,
boundary/outlet/special/HLLC faces, `T`, thermodynamic carriers, MWI, pface,
alpha, time integration, CFL, mesh, and validation thresholds unchanged.

Validator outcome: discarded. N50 remained finite and the active face class was
limited to material faces, but the candidate produced the same unacceptable
p/u tradeoff seen in prior interface active tests: `u` amplitude/correlation
improved, while pressure amplitude, pressure correlation, pressure overshoot,
and `u` phase regressed. N200 was not allowed.

Current status:

- `DENNER_MATERIAL_INTERFACE_CHAR_RECON_BOUNDED` is hard-disabled in production.
- Setting the env var only enables request/capture metadata; it cannot alter
  reconstruction traces or solver numerics.
- The diagnostic reason is
  `rejected_iter659_pu_tradeoff_pressure_regression_overshoot_phase`.
- This is not an available active method unless a future planner explicitly
  reopens it with a new formulation.
