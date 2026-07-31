# THINC thermodynamic-consistency research — case14 contact band (NEGATIVE RESULT)

Date: 2026-07-13. Status: **every zero-constant variant measured, NONE meets acceptance —
code reverted byte-identical, THINC (alpha-only, default ON) baseline retained.** Records the
evidence so the dead-ends are not retried (companion to MWI_DT_RESEARCH.md; background:
denner-pitfalls.md "case14 THINC band spread — DIAGNOSED").

## Goal

Remove the case14 rho contact band (42 cells under THINC vs 23 THINC-OFF) by making the
interface sharpening thermodynamically consistent: at THINC-active faces, rebuild the
convected face temperature from the TWO-SIDED neighbour states with the IDENTICAL normalized
tanh profile H (same beta=3.5, same interface position xi_c, same semi-Lagrangian evaluation
the alpha flux used): `T_f = (1-H)*T_{ic-1} + H*T_{ic+1}` (p and u stay untouched —
continuous at a contact). Zero new constants; suite must stay 19/19.

Acceptance: case14 band <= 23 AND l2_u <= 0.105 AND corr_u >= 0.97 AND l2_rho <= 0.031;
sharp wins preserved; case25 not degraded; 19/19.

## Measured variants (all DENNER_ACID=1; V0 = committed baseline a9a9193/6d241c0 lineage)

| variant | case14 band / l2_u / corr_u / l2_rho | case25 | suite |
|---|---|---|---|
| V0 baseline (alpha-only THINC) | 42 / 0.1131 / 0.9665 / 0.0312 | PASS | 19/19 |
| V1 full-T override, unguarded | **18** / 0.1079 / 0.9695 / **0.0253** | FAIL: interface iu 5.25 (gate 1.5); ip 0.012, positions, rho band all fine | 18/19 (only 25) |
| V2 V1 + !face_shock guard (existing sensor) | 42-ish: 0.1134 / 0.9663 / 0.0279 (gain lost) | still FAIL | — |
| V3 V1 + monotone-T guard (mirrors alpha indicator) | 0.1140 / 0.9659 / 0.0280 (gain lost) | still FAIL | — |
| V4 energy-only: rho(TU) x h(T_ene) decomposition | band 2 BUT case14 corr_u 0.101, case13 corr_u 0.994->0.138, case25 iu 537 | catastrophic | — |

Also measured under V1: case13 l2_rho 0.0171->0.0099 (improves), case02/30/31 unchanged
within noise, acoustics 07/35/36 pass (their interface T is continuous -> override ~no-op),
case01 linf exactly 0, 15/24/33/34 zero activations (byte-identical).

## Why the family cannot meet acceptance (measured, three independent walls)

1. **History wall (case14 ceiling).** The band is CELL-STATE history: T smears from the
   diaphragm breakup via the 1st-order energy transport at BULK faces, long before/behind
   the single THINC-active interface face. Overriding that one face's flux forward in time
   cannot un-smear the field. V1 — the maximal always-on variant — is the family's ceiling
   and reaches band 18 (target met) but l2_u 0.1079 vs <= 0.105 and corr_u 0.9695 vs >= 0.97.
   Any selector (BVD included) applies the override on a SUBSET of V1's instances, so it
   cannot beat this ceiling (V2/V3 confirm: subsets lose the gain).
2. **ACID-telescoping wall (case25).** The ACID flux design (Eqs. 40-44) ties the face
   partial densities to CELL upwind states so a moving contact is discretely source-free.
   Overriding the face T breaks that telescoping; at case25's ~100x T-jump contact this
   appears as an interface VELOCITY source — iu 5.25 vs gate 1.5 — even under the guards
   (V2/V3 keep failing: the contact face itself stays active in the late, "clean" phase).
   Distinguishing case25's contact from case14's is a magnitude question -> any threshold
   is a new constant (banned).
3. **Consistency wall (V4).** Splitting the face thermodynamics — mass flux at TU, enthalpy
   at T_ene (with either rho weighting) — is not a valid decomposition: the mass and energy
   fluxes then transport different material states through the same face, and the shock-tube
   solutions collapse outright (case13 corr_u 0.994 -> 0.138, case25 ip 1.40, iu 537).

## Residual mechanism, precisely

THINC sharpens the VOLUME-FRACTION transport only; the 4-equation model carries ONE mixture
energy/temperature per cell, so a cell whose alpha flips within 1-2 cells retains the
pre-front smeared T and relaxes over ~15 cells of the energy equation (the band). A face
reconstruction cannot fix a cell-state deficiency, and the two candidate face-level
injections both violate a discrete identity the scheme depends on (walls 2 and 3). The
correct fix is PHASE-CONSISTENT ENERGY TRANSPORT: a per-phase energy/temperature state
advected with the interface (5-equation-model-like), or a re-derivation of the ACID
old-level/flux identity (Eqs. 40-44) that includes the sub-cell T reconstruction on BOTH
the flux side and the transient side simultaneously. Either is a model/scheme extension,
not a reconstruction swap — out of scope for a zero-constant face-level change.

The case14 band remains the documented benign residual (gate passes; l2_rho better than
THINC-OFF: 0.031 vs 0.039).

Reproduce: `DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate` (V0 = committed
code); variant formulas above are complete for re-implementation if ever revisited.
