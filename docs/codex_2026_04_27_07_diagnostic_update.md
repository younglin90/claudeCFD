# five_eq_IMEX 07 Diagnostic Update

Date: 2026-04-27

## Completed Changes

- Added near-pure primitive recovery wrapper in `solver/five_eq_IMEX/primitive.py`.
- Added pure-face APEC fallback in `solver/five_eq_IMEX/energy_flux.py`.
- Exposed diagnostic pure branch through:
  - `solver/five_eq_IMEX/main.py::solve(..., pure_branch=False, alpha_pure_tol=1e-8)`
  - `solver/five_eq_IMEX/time_integrator.py::be1_step(...)`
  - `results/run_07_decompose.py --pure-branch`
- Added tests:
  - `tests/test_pure_phase_limit_recovery.py`
  - `tests/test_pe_correction_dpdU.py`
  - `tests/test_well_balanced_alpha_jump.py`
  - `tests/test_jacobian_stencil_consistency.py`

## Test Results

```text
python3 -m py_compile modified files
  PASS

python3 tests/test_pure_phase_limit_recovery.py
  PASS

python3 tests/test_pe_correction_dpdU.py
  PASS

python3 tests/test_jacobian_stencil_consistency.py
  PASS

python3 tests/test_well_balanced_alpha_jump.py
  PASS for one-step and 4-step stationary alpha-jump smoke.
  Important remaining issue: the same stationary alpha-jump can blow up by about 10 steps.

python3 tests/test_uniform_flow.py
  PASS

python3 tests/test_amplification_matrix.py
  PASS for BE1 gates:
    be1 raw rho(A) = 1.0008
    be1 pe_correct=True rho(A) = 1.0008
  Note: ARS222 raw remains unstable in this diagnostic, rho(A) ~= 9.021.

python3 tests/test_transport_eigenmode.py
  script exits PASS, top |lambda| ~= 1.0008.
  Note: top printed modes are still pressure-dominant, so this remains a diagnostic concern.

python3 tests/test_single_phase_acoustic_periodic.py
  PASS

python3 tests/test_single_phase_acoustic_reflective.py
  PASS

python3 tests/test_acoustic_impedance_coefficients.py
  PASS

python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
  PASS
  step=100, err_p=3.223e-07, err_u=1.005e-05, finite=True
  Plot saved: results/1D/02_A/diff_vs_exact.png
```

## Latest 07 Results

```text
python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --pe-projection-mode contact --max-steps07 1000

Air-Water:
  FAIL, finite=True
  L2p=8.238e+03, Lip=2.829e+04, L2u=1.404e+02, Liu=6.863e+02
  corr_p=-0.25, corr_u=-0.05

Helium-Air:
  FAIL but finite and near threshold
  L2p=2.983e-01, Lip=8.940e-01, L2u=2.538e-01, Liu=5.900e-01
  corr_p=0.30, corr_u=0.28

Argon-Air:
  FAIL but closest to PASS
  L2p=1.664e-01, Lip=4.844e-01, L2u=2.123e-01, Liu=5.910e-01
  frac_p=0.92, frac_u=0.82, corr_p=0.64, corr_u=0.64
  Only Liu remains clearly above the strict 0.50 gate.

Plot saved: results/1D/07_B/diff_vs_exact.png
```

Air-Water comparison:

```text
python3 results/run_02_07_five_eq_imex.py --case 07 --subcase07 Air-Water --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --pe-projection-mode always --max-steps07 1000

Air-Water:
  FAIL, finite=True
  L2p=5.363e-01, Lip=1.967e+00, L2u=8.946e-01, Liu=4.277e+00
  corr_p=-0.08, corr_u=0.01
```

Pure-branch diagnostic:

```text
python3 results/run_07_decompose.py --case-material helium-air --n 50 --cfl 0.1 --pe-correction tangent --pe-projection-mode contact --pure-branch --max-steps 1000
  finite complete, no primitive recovery failure
  results/1D/07/debug_helium-air_n50_cfl0p1_Eapec_Tacid_peTrue_contact_corrtangent_loTrue_pureTrue_D0p02.json

python3 results/run_07_decompose.py --case-material argon-air --n 50 --cfl 0.1 --pe-correction tangent --pe-projection-mode contact --pure-branch --max-steps 1000
  finite complete, no primitive recovery failure
  results/1D/07/debug_argon-air_n50_cfl0p1_Eapec_Tacid_peTrue_contact_corrtangent_loTrue_pureTrue_D0p02.json
```

## Why 07 Computations Looked Slow

- In failing 07 combinations, pressure/velocity blow-up increases the local acoustic speed.
- The adaptive acoustic dt then collapses toward zero.
- Without `dt_min` and `max_steps`, this looks like a hanging simulation rather than a clean failure.
- Current runners now expose this with `terminated_reason=dt_below_min` or bounded `max_steps`.

## Current Diagnosis

- 02-A is still protected; do not loosen the PE regression gates.
- 07 is not an `imp_dissipation` tuning problem.
- Single-phase acoustic periodic/reflective tests pass, so the base acoustic block and wall sign are not the first-order failure.
- `pe_projection_mode=contact` prevents the gas-gas NaN/overflow path, but Air-Water becomes much worse.
- `pe_projection_mode=always` is less bad for Air-Water but over-corrects gas-gas acoustic waves.
- Therefore the next fix should be a continuous/local PE projection sensor, not a global on/off switch.
- Pure-branch recovery and pure-face APEC fallback are now available, but with the current alpha floor of 1e-3 they do not by themselves fix 07.
- Remaining stationary alpha-jump long-time blow-up indicates there is still a low-level PE/contact amplification path separate from the 02-A finite PASS.

## Recommended Next Implementation

1. Replace binary PE projection mode with a local strength `theta_pe in [0,1]`.
2. Compute `theta_pe` from material-contact, acoustic, and impedance indicators.
3. Desired behavior:
   - gas-gas acoustic interface: `theta_pe -> 0`
   - stationary/moving PE material contact: `theta_pe -> 1`
   - Air-Water strong-impedance interface: partial projection, not always 0
4. Add diagnostics to `run_07_decompose.py`:
   - `max_theta_pe`
   - `mean_theta_pe`
   - `num_projected_cells`
   - `max_abs_pe_correction`
   - `max_ratio_pe_correction_to_raw`
5. After sensor implementation, rerun:
   - `python3 tests/test_well_balanced_alpha_jump.py`
   - `python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01`
   - `python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --max-steps07 1000`

## Experimental Sensor Attempt

Implemented optional `--pe-projection-mode sensor`.

Current behavior:

- Gas-gas cases are stable like `contact` mode.
- Air-Water is still much worse than `always` mode.
- Strong-impedance floor `1.0` caused Air-Water `dt_below_min`; reverted to `0.80`.

Latest retained sensor run:

```text
python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --pe-projection-mode sensor --max-steps07 1000

Air-Water:
  FAIL, finite=True
  L2p=6.000e+03, Lip=2.190e+04, L2u=4.274e+01, Liu=2.283e+02
  corr_p=0.20, corr_u=0.10

Helium-Air:
  FAIL but finite
  L2p=3.015e-01, Lip=9.041e-01, L2u=2.545e-01, Liu=6.046e-01
  corr_p=0.29, corr_u=0.28

Argon-Air:
  FAIL but closest to PASS
  L2p=1.662e-01, Lip=4.834e-01, L2u=2.120e-01, Liu=5.922e-01
  corr_p=0.64, corr_u=0.64
```

Interpretation:

- A simple local projection strength is not enough for Air-Water.
- Next likely root is acoustic/interface pressure-work consistency, not just PE projection gating.
- For Air-Water, compare exact/interface phase and pressure work using a reduced two-medium Riemann/acoustic diagnostic before further tuning.

## PE Operator Diagnostic Follow-Up

Added:

- `results/diagnose_pe_operator.py`
- output directory: `results/1D/PE_operator/`

Command:

```text
python3 results/diagnose_pe_operator.py
```

Outputs:

- `results/1D/PE_operator/pe_operator_diagnostics.csv`
- `results/1D/PE_operator/pe_operator_diagnostics.json`
- `results/1D/PE_operator/pe_operator_linearized.csv`
- `results/1D/PE_operator/pe_operator_linearized.json`
- `results/1D/PE_operator/diff_vs_exact.png`

Main result:

```text
Exact PE state:
  p_U dot L_E ~= 0
  p_U dot L_I ~= 0
  p_U dot (L_E + L_I) ~= 0
```

Therefore the leading failure is not a nonzero base PE residual. The issue is
linearized/roundoff amplification around the PE manifold.

Important linearized observations:

```text
02A_nasg, L_E, HO_secant, p_alt_1Pa:
  max |delta(p_U dot L)| / p0 ~= 3.884e-02

02A_nasg, L_E, HO_apec/default, alpha_alt_1e-6:
  max |delta(p_U dot L)| / p0 ~= 1.144e-10

07 Air-Water, L_I D=0.02, u_alt_1e-6:
  max |delta(p_U dot L)| / p0 ~= 4.106e-06
```

Interpretation:

- The current `secant` APEC implementation is not safe as a default for NASG pressure perturbations.
- The default `apec` path remains much safer for 02-A.
- 07 Air-Water is dominated by acoustic/interface behavior rather than exact PE residual at the base state.

## Zero-Update Guard

Added a no-op update guard in `solver/five_eq_IMEX/time_integrator.py::be1_step`.

If:

```text
max_k |dt * L_total[k]| / max(|U_n[k]|, 1) <= 1e-13
```

then the step returns `W_n` directly and skips conservative-to-primitive recovery.

Reason:

- Stationary PE contacts have zero physical update.
- Repeated `U -> W -> U` recovery on exact no-op steps introduced tiny pressure errors.
- Those errors were then amplified by the marginal BE1 pressure mode.

Validation:

```text
python3 tests/test_well_balanced_alpha_jump.py
  PASS
  stationary alpha-jump many-step smoke strengthened from 4 steps to 20 steps

python3 tests/test_uniform_flow.py
  PASS

python3 tests/test_amplification_matrix.py
  PASS
  be1 raw rho(A) = 1.0008

python3 tests/test_transport_eigenmode.py
  PASS script, top |lambda| ~= 1.0008

python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
  PASS
  err_p=3.223e-07, err_u=1.005e-05

python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --imp-dissipation 0.02 --pe-projection-mode contact --max-steps07 1000
  still FAIL 0/3
  Air-Water remains the main mismatch
  Helium-Air and Argon-Air remain finite near-threshold diagnostics
```

## Follow-up Loop: 07 Acoustic Decomposition

Implemented/added:

- `tests/test_single_phase_acoustic_water.py`
  - resolved stiffened-gas water acoustic propagation smoke test.
  - This guards the water EOS/acoustic path separately from under-resolved
    Air-Water interface transmission.
- `solver/five_eq_IMEX/residual.py`
  - material-interface acoustic Riemann face state for implicit `p,u` faces:
    `p* = (Z_R p_L + Z_L p_R + Z_L Z_R (u_L-u_R))/(Z_L+Z_R)`,
    `u* = (p_L-p_R + Z_L u_L + Z_R u_R)/(Z_L+Z_R)`.
  - This preserves constant `(p,u)` material contacts exactly.
- `solver/five_eq_IMEX/time_integrator.py`
  - diagnostic projection modes ending in `_explicit`, e.g. `interface_explicit`,
    project only `L_E` and leave the implicit acoustic block unprojected.
- `solver/five_eq_IMEX/limiters.py`
  - diagnostic `force_lo == "interface"` path. This was tested and is not a
    good production default.
  - alpha-floor-aware positivity limiter. The limiter now scales its internal
    alpha admissibility floor from the actual case alpha margin, instead of
    always using `1e-6`.
- `results/run_02_07_five_eq_imex.py`
  - CLI modes: `contact_explicit`, `interface_explicit`, `impedance_explicit`.
  - CLI options: `--alpha-floor07`, `--pure-branch07`, `--interface-force-lo`.

Regression after these changes:

```text
python3 tests/test_uniform_flow.py
  PASS

python3 tests/test_well_balanced_alpha_jump.py
  PASS

python3 tests/test_amplification_matrix.py
  PASS
  be1 raw rho(A) ~= 1.0009

python3 tests/test_transport_eigenmode.py
  PASS script
  top |lambda| ~= 1.0009

python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
  PASS
  err_p=3.576e-07, err_u=7.886e-06

python3 tests/test_single_phase_acoustic_water.py
  PASS
```

07 official-grid status:

```text
Argon-Air:
  python3 results/run_02_07_five_eq_imex.py --case 07 --subcase07 Argon-Air --n07 200 --cfl07 0.4 --imp-dissipation 0.0 --pe-projection-mode contact --max-steps07 2000
  PASS
  L2p=4.792e-02, Lip=2.281e-01, L2u=6.104e-02, Liu=3.113e-01, corr_p=0.97, corr_u=0.97

Helium-Air:
  same options, subcase Helium-Air
  FAIL only by pressure Linf
  L2p=1.377e-01, Lip=7.910e-01, L2u=6.338e-02, Liu=3.441e-01, corr_p=0.81, corr_u=0.89
  CFL 0.1 improves Lip to 6.398e-01 but still fails strict Lip < 0.50.

Air-Water:
  contact/sensor/impedance/interface_explicit projection modes are unstable or blow up at N=200.
  always projection remains finite but freezes/suppresses the acoustic structure:
  L2p~=5.37e-01, Lip~=2.0, L2u~=4.47e-01, Liu~=2.0.
```

Important negative results:

- `impedance` projection did not fix Air-Water.
- `interface_explicit` helps N=50 Air-Water but becomes unstable at N=200.
- `alpha_floor07=1e-8` with `pure_branch07` is still not a viable Air-Water fix.
  The limiter floor conflict was removed, but the high-order near-pure path
  remains unstable and the forced-LO path remains too inaccurate for Air-Water.
- `--interface-force-lo` behaves like globally disabling the stable LO path in
  gas-gas 07 and degrades Helium-Air/Argon-Air.
- Helium-Air `primitive_scheme='central'` gives identical N=200 metrics to
  `primitive_scheme='upwind'` under the current forced-LO path:
  `L2p=1.377e-01, Lip=7.910e-01, L2u=6.338e-02, Liu=3.441e-01`.
  The remaining Helium-Air failure is therefore not caused by primitive
  temperature upwinding.
- Helium-Air PE projection ablation at N=200:
  `--no-pe-project-explicit` gives the same metrics as `contact` mode:
  `L2p=1.377e-01, Lip=7.910e-01, L2u=6.338e-02, Liu=3.441e-01`.
  `pe_projection_mode=always` blows up (`dt_below_min`).  The Helium-Air
  pressure Linf failure is therefore not caused by contact PE projection, and
  global projection should not be used for 07 gas-gas acoustic cases.
- Helium-Air face-thermo ablation at N=200:
  `face_thermo=upwind` and `face_thermo=cell` give the same metrics as
  `face_thermo=acid`: `L2p=1.377e-01, Lip=7.910e-01, L2u=6.338e-02,
  Liu=3.441e-01`.  The local pressure Linf overshoot is therefore not caused
  by ACID EOS re-evaluation at faces.
- Air-Water interface impedance ablation:
  replacing the implicit material-face impedance by dominant-phase impedance
  did not improve the `always` profile and destabilized local projection modes.
  The change was reverted; the issue is not just alpha-floor pollution of the
  acoustic Riemann impedance.
- Air-Water Schur ablation:
  `schur=True` and `schur=False` produce identical profile metrics for the
  finite `always` case at N=100.  The current mismatch is not caused by the
  Schur pressure solve path.
- Air-Water projection-band ablation:
  an `interface_band` PE projection mode was added for diagnostics.  A narrow
  band is less stable than `always`, and increasing the band radius worsens the
  pressure profile.  This suggests the problem is not just projection support
  width; the pressure-work/interface operator needs a different correction.

Current interpretation:

- 02-A PE stability is protected.
- 07 gas-gas is mostly an accuracy/phase problem; Argon-Air now passes at the
  official grid, Helium-Air is close but pressure Linf remains too high.
- 07 Air-Water is still a separate strong-impedance problem. Global projection
  prevents blow-up but also removes physical acoustic propagation; weaker/local
  projection lets a stiff pressure mode grow.

Next concrete target:

1. Add a strict `tests/test_single_phase_acoustic_water.py` before changing
   Air-Water again.
2. Replace Air-Water stabilization by a face/source-level PE/acoustic correction
   instead of residual projection of the combined acoustic block.
3. For Helium-Air, reduce local pressure Linf overshoot without disabling the
   stable LO path; global HO and interface-only LO both degrade the result.
