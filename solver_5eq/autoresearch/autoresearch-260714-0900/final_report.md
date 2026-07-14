# Autoresearch final report — SUCCESS (2026-07-14)

Goal: pass the full strict validation suite with one solver, one scheme, and
zero user-tuned coefficients. **Achieved.**

## Winning configuration

Production configuration plus one scheme selection:

```
FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ssp3   FIVE_EQ_IMEX_ALPHA_SCHEME=adaptive_bvd
FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu      FIVE_EQ_IMEX_TMLPU_TVD=superbee
FIVE_EQ_IMEX_MATERIAL_FLUX=slau2         FIVE_EQ_IMEX_PRESSURE_CLOSURE=regime_auto
FIVE_EQ_IMEX_CHARACTERISTIC_RECON=1      FIVE_EQ_IMEX_RUSANOV_FALLBACK=0
FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0    FIVE_EQ_CASE24_N=400
FIVE_EQ_IMEX_ACOUSTIC_RECON=weno5        <- the new ingredient
```

`weno5` = componentwise Jiang–Shu WENO5 reconstruction of (p, u) on
material-clean acoustic faces, feeding the existing Z-weighted acoustic
Riemann solve, in both the explicit face helper and the implicit torch
residual. All constants are literature-fixed (JS linear weights 1/10, 6/10,
3/10; JS smoothness indicators; scale-invariant relative epsilon). The
implicit Jacobian uses a straight-through construction: the residual VALUE
keeps the full nonlinear WENO5; the DERIVATIVE is the linear-optimal
fifth-order stencil (a structural choice, not a coefficient). Zero user-tuned
constants anywhere.

## Final verification matrix (all with the single config above)

| Suite | Result |
|---|---|
| Core 13 cases (01,02,04,05,07x3,13,14,15,16,17,18,24,25) | **13/13 PASS** |
| 07-B Air-Water (the historically failing guard) | PASS: tv_excess 0.204 (limit 0.30), amp 1.00/0.98, L2p 0.060 (baseline 0.090) |
| 07-B Helium-Air / Argon-Air | PASS (tv 0.062 / 0.057) |
| Source terms 32/33/34/35 | 4/4 PASS |
| tests/test_uniform_flow.py | PASS |
| tests/test_amplification_matrix.py | be1 rho(A)=1.0009 < 1.005 PASS |
| Default path (env unset) | bit-identical golden 2.764863893389702e-15 |

## How it was found (iteration ledger, 16 candidates total)

Campaign phase (pre-loop, 9 candidates): time schemes (BE / interface-BE /
TR-BDF2), interface reconstruction switches, closure choice, characteristic
recon, BVD selectors x2, kappa=1/3 — all rejected; established that the
artifact is a phase-coherent accumulation inside the physical band.

Loop phase:
- iter1: term-level attribution — seed = beta*div_u fed by reconstructed
  faces; advection term ruled out (1e9x smaller); latent beta-tol bug found
  (negligible amplifier; fixed as default-off option).
- iter2: H11c damping restoration — tv passes but amp 0.45; proves
  band-blind damping cannot work (physical pulse sigma ~ 3.7 cells shares the
  band).
- iter3: H12b stencil-clean gating — first KEEP (tv -20%, amp intact).
- iter4: stacking matrix — no further gain from interface-local measures.
- iter5: ACID local-single-phase port — rejected (breaks physical R/T at
  mismatched impedance; designed for matched interfaces).
- iter6: WENO3/WENO5 — weno5 PASSES AW + full 07 (first ever); weno3
  over-damps. Two regressions surfaced in the 13-case sweep.
- iter7: root-caused and fixed the weno5 implicit-path defects (dropped i+-3
  Jacobian coupling; noise-driven jacrev of adaptive weights on flat fields
  -> straight-through Jacobian + widened stencil). 13/13 achieved.

## Adoption notes

- Current landing: env-gated, default off. The production default is still
  `component` (bit-identical to the paper evidence).
- To adopt weno5 as the production method: add the one env line to the fixed
  configuration (least invasive), or flip the in-code default (requires
  re-baselining the 02_A golden from 2.764863893389702e-15 to the weno5 value
  4.4e-16 — which is BETTER — and regenerating the evidence package + paper
  figures/numbers).
- The paper's Air-Water limitation paragraphs (Sections 4.3, 4.7, 5) can be
  replaced by the weno5 result after the evidence package is regenerated
  under the new configuration.
- Research options accumulated during the campaign (all default-off,
  bit-identical when unset): ACOUSTIC_SCHEME={interface_be,trbdf2},
  ACOUSTIC_RECON={characteristic,bvd,muscl3,weno3,weno5},
  ACOUSTIC_DISS_CONSISTENT, ACOUSTIC_STENCIL_CLEAN,
  ACOUSTIC_PURE_TOL_CONSISTENT, ACOUSTIC_ACID. Candidates for pruning before
  the C++ port: keep weno5 + stencil_clean + pure_tol_consistent; the rest
  are documented failures useful for the method paper's ablation narrative.
