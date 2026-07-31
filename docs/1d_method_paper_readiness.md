# 1D Euler Numerical Method Paper Readiness

Updated from the Euler-only evidence package generated on 2026-05-10.

## Scope Recommendation

The defensible near-term manuscript is a **1D Euler-equation numerical-method
paper** for an all-speed compressible two-phase five-equation finite-volume
solver.  Source-term cases 32--35 are intentionally excluded from this paper and
should be reserved for a follow-up source-term/phase-change manuscript.

The paper should not claim a complete multidimensional production solver yet.
The strongest current contribution is the 1D method combination:

- IMEX-SSP3 time integration.
- Adaptive-BVD volume-fraction transport with CICSAM on sharp pure interfaces.
- T-MLP-u primitive-variable reconstruction with a TVD limiter.
- SLAU2 material/advection flux.
- Regime-aware pressure closure and acoustic treatment.
- NASG/SG/ideal-gas thermodynamics across gas-liquid and gas-gas interfaces.

## Fixed Production Configuration

```text
FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ssp3
FIVE_EQ_IMEX_ALPHA_SCHEME=adaptive_bvd
FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu
FIVE_EQ_IMEX_TMLPU_TVD=superbee
FIVE_EQ_IMEX_MATERIAL_FLUX=slau2
FIVE_EQ_IMEX_PRESSURE_CLOSURE=regime_auto
FIVE_EQ_IMEX_CHARACTERISTIC_RECON=1
FIVE_EQ_IMEX_RUSANOV_FALLBACK=0
FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0
FIVE_EQ_CASE24_N=400
```

## Evidence Package

Run:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=.codex-loop python3 results/1D/paper_euler_evidence.py
```

Current generated artifacts:

- `results/1D/paper_euler_evidence/paper_euler_evidence.json`
- `results/1D/paper_euler_evidence/paper_euler_evidence.md`
- `results/1D/paper_euler_evidence/csv/core_metrics.csv`
- `results/1D/paper_euler_evidence/csv/grid_metrics.csv`
- `results/1D/paper_euler_evidence/csv/baseline_metrics.csv`
- `results/1D/paper_euler_evidence/csv/cfl_metrics.csv`
- `results/1D/paper_euler_evidence/csv/all_speed_metrics.csv`
- `results/1D/paper_euler_evidence/plots/*.png`

The package stores both manuscript-style PNGs and raw metric CSV tables.  The
CSV files are the raw data for the summary plots and should be used to build the
paper tables.

## Core Euler Validation Status

The production method passes the Euler-focused core sweep:

| Case | Status | Main claim |
|---|---:|---|
| 01_A | PASS | Static pressure-equilibrium interface. |
| 02_A | PASS | Periodic pressure-equilibrium material advection; p/u preserved. |
| 04_B | PASS | Air acoustic wave. |
| 05_B | PASS | Water acoustic wave. |
| 07_B | PASS | Acoustic reflection/transmission for Air-Water, Helium-Air, Argon-Air. |
| 13_E | PASS | HP-air / LP-water shock-interface case. |
| 14_E | PASS | HP-water / LP-air shock-interface case with close discontinuities. |
| 15_E | PASS | Cavitation case. |
| 16_T | PASS | Sharp temperature-contrast material advection. |
| 17_T | PASS | Smooth alpha Gaussian hot-gas transport. |
| 18_T | PASS | Smooth alpha thermal-wave transport with low wiggle. |
| 24_H | PASS | Hypersonic mixture case at N=400. |
| 25_H | PASS | Hypersonic Mach-10 air-water case. |

Key metrics from the latest package:

- 07_B Air-Water at N=400: `L2p=9.00e-2`, `Lip=3.55e-1`, pressure peak amplitude ratio `0.998`.
- 07_B Helium-Air: `L2p=2.22e-2`, `Lip=1.76e-1`, pressure peak amplitude ratio `0.968`.
- 07_B Argon-Air: `L2p=7.46e-3`, `Lip=2.92e-2`, pressure peak amplitude ratio `1.025`.
- 18_T: `rho_l1_ratio=6.98e-5`, `T1_active_hf_max=2.81e-4`, `T2_active_hf_max=5.75e-4`.
- 24_H: worst subcase `rho_profile_l2=2.17e-2`.

## Added Manuscript Evidence

### 1. Grid Refinement

Raw data:

```text
results/1D/paper_euler_evidence/csv/grid_metrics.csv
```

Main figure:

```text
results/1D/paper_euler_evidence/plots/grid_refinement_errors.png
```

Included representative refinement sweeps:

- 07_B Air-Water: N=100, 200, 400.
- 13_E: N=200, 400, 800.
- 14_E: N=200, 400, 800.
- 18_T: N=200, 400, 550.
- 24_H: N=100, 200, 400.
- 25_H: N=200, 400, 800.

Important interpretation: some coarse-grid runs intentionally fail the strict
acceptance criteria.  This is useful for the paper because it shows resolution
requirements.  For example, 07_B Air-Water fails strict peak-amplitude criteria
at N=100/200 but passes at N=400.

### 2. Baseline Comparisons

Raw data:

```text
results/1D/paper_euler_evidence/csv/baseline_metrics.csv
```

Figures:

```text
results/1D/paper_euler_evidence/plots/baseline_ablation_metrics.png
results/1D/paper_euler_evidence/plots/ablation_pass_heatmap.png
```

Compared variants:

- production method.
- primitive upwind.
- superbee-only primitive reconstruction without T-MLP-u.
- T-MLP-u + van Leer.
- T-MLP-u + minmod.
- alpha CICSAM.
- alpha MSTACS.
- HLLC material flux.

Representative target cases: 02_A, 07_B Air-Water, 13_E, 18_T.  The failures in
this table are not regressions; they are the ablation evidence that the full
method is needed.

### 3. Ablation Study

The ablation study uses the same raw table as the baseline comparison.  It is
suitable for a manuscript section such as:

- Removing T-MLP-u or using primitive upwind increases acoustic diffusion in 07_B.
- Minmod/van Leer reduce some shock/contact oscillation but lose acoustic peak amplitude.
- HLLC split is less robust for the current IMEX pressure split than SLAU2.
- Adaptive-BVD is needed because a single alpha compression scheme is not optimal for both sharp contacts and smooth composition waves.

### 4. Pressure-Equilibrium Preservation

Figure:

```text
results/1D/paper_euler_evidence/plots/pressure_equilibrium_preservation.png
```

Raw table:

```text
results/1D/paper_euler_evidence/csv/core_metrics.csv
```

The relevant pressure-equilibrium tests are 01_A, 02_A, 16_T, 17_T, and 18_T.
They check that p/u remain essentially unchanged when the exact solution is a
material or thermal transport problem at uniform pressure and velocity.

The paper should write the discrete argument around the split form:

```text
U_t + F_E(W)_x + F_I(W)_x = 0,
W = (alpha1, T1, T2, u, p).
```

For a pressure-equilibrium state with constant `p=p0` and `u=u0`, the implicit
acoustic pressure-gradient residual is zero at each face.  The explicit material
step transports alpha and thermodynamic scalars, while the pressure closure is
constructed so that the uniform pressure state is a fixed point.  The numerical
evidence is that `p_rel_linf` and `u_abs_linf` remain near roundoff in the PE
transport cases, with no periodic remap shortcut.

### 5. Low-Mach / All-Speed Evidence

Raw data:

```text
results/1D/paper_euler_evidence/csv/cfl_metrics.csv
results/1D/paper_euler_evidence/csv/all_speed_metrics.csv
```

Figure:

```text
results/1D/paper_euler_evidence/plots/acoustic_cfl_sweep.png
```

Representative all-speed runs:

- 03_B ultra-low-Mach acoustic pulse: PASS.
- 04_B low-Mach air acoustic wave: PASS.
- 07_B Air-Water acoustic interface at N=400: PASS.
- 25_H hypersonic Mach-10 air-water case: PASS.

The acoustic-CFL sweep is diagnostic at N=200, not the final production
acceptance table.  It shows stability and error trends with CFL, while the
production 07_B claim should use N=400.

## Manuscript Positioning

Recommended title direction:

```text
A one-dimensional all-speed IMEX finite-volume method for compressible two-phase five-equation flows with low-diffusion interface transport
```

Recommended contribution claims:

- A single 1D Euler solver configuration covers pressure-equilibrium advection,
  acoustic transmission/reflection, shock-interface interaction, cavitation,
  thermal contrast advection, and hypersonic mixture cases.
- The method avoids first-order upwind as the production path and uses second-
  order spatial/time ingredients.
- The strongest comparative evidence is the ablation table showing why the
  combined production method is needed.

Claims to avoid in this paper:

- Full 2D/3D validation.
- General source-term accuracy.
- Phase-change/source-term production readiness.
- A formal AP proof unless a separate discrete asymptotic analysis is added.

## Remaining Before Submission

The code/results are now strong enough to draft the manuscript.  Before actual
submission, the paper still needs:

1. A clean method section deriving the IMEX split, primitive reconstruction,
   adaptive-BVD alpha transport, and pressure-equilibrium preservation argument.
2. Tables generated directly from the CSV raw data.
3. A comparison paragraph against recent all-Mach multiphase and THINC-BVD
   literature, clearly stating that this paper is 1D Euler-focused.
4. A reproducibility appendix listing environment variables and exact commands.
