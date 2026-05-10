# 1D Numerical Method Paper Readiness

Generated from the evidence run on 2026-05-09.

## Scope Recommendation

The defensible manuscript scope is a 1D numerical-method paper for an all-speed
compressible two-phase five-equation finite-volume solver.  The paper should
not yet claim a complete multidimensional solver.  The strongest contribution is
the combined 1D method:

- IMEX-SSP3 time integration.
- Adaptive-BVD sharp volume-fraction transport.
- T-MLP-u primitive-variable reconstruction with a TVD limiter.
- SLAU2-type material/advection flux.
- Regime-aware pressure closure and acoustic treatment.
- NASG/SG/ideal-gas thermodynamics and source-term extensions.

The current evidence is good enough to justify a paper plan, but not yet enough
for submission as a complete "all benchmark cases pass" manuscript.

## Reproducible Evidence Driver

The paper evidence runner is:

```bash
MPLCONFIGDIR=/tmp/mpl python3 results/1D/paper_1d_evidence.py
```

It writes:

- `results/1D/paper_evidence/paper_1d_evidence.json`
- `results/1D/paper_evidence/paper_1d_evidence.md`

The recorded common method configuration is:

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
```

## Current Core Sweep

| Case | Result | Interpretation |
|---|---:|---|
| 01_A | PASS | Pressure-equilibrium static interface is preserved. |
| 02_A | FAIL | `p` and `u` are preserved to roundoff, but alpha/rho contact diffusion is still too large for the current strict criterion. |
| 04_B | PASS | Acoustic sinusoid in air is acceptable. |
| 05_B | PASS | Acoustic sinusoid in water is acceptable. |
| 07_B | PASS | Strong evidence: all Air-Water, Helium-Air, Argon-Air acoustic R/T cases pass at N=400. |
| 13_E | FAIL | Superbee limiter creates a small contact-density overshoot; van Leer/minmod limiter variants pass this case. |
| 14_E | FAIL | Close discontinuity location is good, but the rho envelope around 0.85 m still exceeds the strict overshoot criterion. |
| 15_E | PASS | Cavitation case is acceptable. |
| 16_T | FAIL | Thermal-mixture transport error remains too large. |
| 17_T | FAIL | Pressure/velocity are stable, but alpha/rho/Tmix peak preservation is below the strict criterion. |
| 18_T | FAIL | Pressure/velocity are stable, but alpha/rho wiggle guard still fails. |
| 24_H | FAIL | Timed out at the current common high-resolution setting. Needs a cheaper paper-grade run protocol or solver speedup. |
| 25_H | PASS | Hypersonic air-water case is acceptable. |
| 32-35 | PARTIAL | 32, 34, 35 pass; 33 gravity faucet fails pressure-spike criterion. |

## Limiter Evidence

The current common `superbee` limiter is best for the low-diffusion acoustic
07_B result, but it is too aggressive near material-contact shock-tube
interfaces:

- 13_E with `superbee`: FAIL due contact rho peak overshoot about 5.8%.
- 13_E with `vanleer`: PASS; contact rho peak overshoot is removed.
- 13_E with `minmod`: PASS; even more monotone but more diffusive.
- 14_E with `superbee`: FAIL; rho envelope overshoot about 5.4%.
- 14_E with `vanleer`: FAIL; overshoot reduces to about 4.9%.
- 14_E with `minmod`: FAIL but closer; overshoot reduces to about 1.8%, with correct shock/contact locations.

Conclusion: a single fixed TVD limiter is not yet an optimal paper method.  The
next numerical contribution should be a parameter-free local BVD/TVD selector
between a low-diffusion candidate and a monotone candidate, not a case-specific
manual switch.

## Required Work Before Submission

1. Implement a parameter-free primitive BVD-TVD selector.
   The selector should compare low-diffusion T-MLP-u+superbee/vanleer and
   monotone T-MLP-u+minmod candidates using local boundary variation and
   positivity/admissibility.  It must be one global method used for every case.

2. Improve alpha/rho contact transport in 02_A without periodic remap.
   Current p/u preservation is essentially exact, so the remaining weakness is
   purely interface diffusion.  A 1D geometric/THINC-BVD conservative contact
   transport variant is the most relevant fix.

3. Fix 14_E rho behavior around the close discontinuities.
   The shock/contact locations are already within about one cell, so the issue
   is not wave speed.  It is the rho reconstruction/contact-envelope behavior.

4. Add a paper-grade thermal transport method for 16_T, 17_T, 18_T.
   Pressure equilibrium is stable, but mixture temperature and alpha/rho smooth
   profile preservation are not strong enough for a temperature-difference
   section.

5. Make 24_H reproducible within a finite paper-run budget.
   The current setting timed out.  Either optimize the solver path or define a
   lower-resolution convergence table that still supports the numerical claim.

6. Add a well-balanced gravity source discretization for 33_S1.
   32_S1 proves hydrostatic equilibrium, but 33_S1 still shows a pressure-spike
   issue under through-flow gravity.

## Manuscript Positioning

The paper should emphasize:

- Pressure-equilibrium preservation: 01_A, 02_A p/u roundoff behavior.
- Low-diffusion acoustic interface transmission: 07_B at high impedance ratio.
- Shock/interface robustness: 13_E, 14_E, 15_E, 25_H after the limiter/contact
  fixes above.
- EOS generality: NASG/SG/ideal-gas tests.
- Source-term extensibility: 32_S1, 34_S2, 35_S2B, with 33_S1 fixed or reported
  as a remaining limitation.

The paper should not overclaim:

- Full 2D/3D validation.
- Universal source-term accuracy.
- Strictly exact sharp-contact advection without the remaining 02_A diffusion
  fix.
