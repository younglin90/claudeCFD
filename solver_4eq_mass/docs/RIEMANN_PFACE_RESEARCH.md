# Linearized acoustic-Riemann face pressure at shock faces — DISCARD (autoresearch iter 2)

Date: 2026-07-14. Status: **measured (R1, R1b), G3 mechanism confirmed but suite-breaking;
code reverted byte-identical (19/19, case01 linf 0), doc + goal.md log only.**
Goal file: autoresearch/autoresearch-260714/goal.md (target G3: case25 amp_ratio_p <= 1.10).

## Formulation (parameter-free, at the EXISTING face_shock sensor faces only)

R1:  p*_f = (Z_R p_L + Z_L p_R)/(Z_L+Z_R) + Z_L Z_R (u_L - u_R)/(Z_L+Z_R),  Z = rho*a local.
R1b: p*_f = 0.5(p_L + p_R)              + Z_L Z_R (u_L - u_R)/(Z_L+Z_R)   (decomposition).
Analytic Jacobian extended exactly for both: impedance/central weights in d(pface)/dp and the
new momentum-row coupling d(p*)/du_L = -d(p*)/du_R = Z_LZ_R/(Z_L+Z_R) (frozen Z). Smooth and
acoustic faces untouched (the sensor never fires there — 04/05/07/35/36 unaffected by
construction; distinct from the documented bare-upwind pface dead-end, which deleted the
average: p* keeps it and only adds the physical jump term).

## Measured

| variant | case25 amp (hf) | suite | failures |
|---|---|---|---|
| baseline (iter 1) | 1.243 (0.842) | 19/19 | — |
| R1 | **1.032 (0.509)** | spot: 3+ fail | 13 (u_shock_jump_ratio 0.445->0.095 < 0.10), 14 (corr_u 0.945 < 0.95), 34 (plateau) |
| R1b | **1.024 (0.493)** | **14/19** | 13 (jump 0.094), 28, 31, 33, 34 (dip 0.0218 > 0.02) |

case13 detail under R1: ALL shock-band overshoot/TV metrics = 0 (perfectly clean) and the
position exact — the front is simply DIFFUSED 5x (per-face u-jump 0.445 -> 0.095 of the exact
jump). case24 plateau under R1b: dip 0.0170 (just inside 0.02) — even the surviving mixture
shock sits within 15% of its gate. R2 (Riemann u* in the mass flux) not tested: the family
fails by over-dissipation, which u* compounds. Courant probe skipped (dead at fixed Co).

## Why (the family's structural problem)

The acoustic-impedance jump term Z_LZ_R/(Z_L+Z_R)(u_L-u_R) is the LINEARIZED (weak-wave)
Riemann pressure: at a resolved strong shock the local |du| across every face of the ~4-cell
front is O(u_jump/4), so the added pressure dissipation is enormous — enough to cure case25's
genuinely under-damped reflected shock (amp 1.24 -> 1.03, the missing-dissipation mechanism
is CONFIRMED) and simultaneously 5-10x too much for the shocks that are already clean:
13/26/27/28/33/34 sit exactly at knife-edge sharpness (u_shock_jump_ratio >= 0.10) and
plateau-monotonicity (dip <= 0.02, hump <= 0.01) gates that ANY uniform front-diffusing
addition violates. The Z-weighted vs central average distinction is second-order (R1 vs R1b
both fail 13/34); the du term drives everything.

## Design constraint extracted for the next candidates (goal.md queue)

A G3 candidate must be SELF-LIMITING where the solution is already clean: its magnitude must
scale with a measured local IMBALANCE (e.g. the pressure-velocity decoupling residual the MWI
filter is supposed to kill, which is large at case25's reflected front and ~zero at a clean
Hugoniot front), not with the raw solution jump (|du|, |drho|), which is equally large at
clean and dirty shocks. Candidate 3 (Kurganov LLF on the ACID mass flux, -0.5 s_max (rho_R -
rho_L)) carries the same raw-jump risk at the contact/interface faces — recommend evaluating
its effect on 13's jump-ratio and 24/33/34's plateaus FIRST (cheap spot checks) before a full
sweep.

Reproduce: formulas complete above; the exact Jacobian rows are described in Sec. Formulation.

## Addendum — iter 3: CLAMP-GATED jump term (DISCARD; the gating study)

Candidate: fire the same jump term ONLY at faces with face_shock AND MWI-clamp saturation
(mwiOK_f == 0, |mwi_p_unclamped| >= a_f — the existing physical bound; term ADDED to the kept
average; exact gated dp*/du Jacobian). Measured firing map (dbg counters, per case:
fires / max |mwi_p|/a_f):

  13: 0 / 0.102   14: 108 / 9.47   24: 97 / 11.4   25: **0 / 0.394**   26: 0 / 0.394
  27: 0 / 0.217   28: 109 / 32.4   33: 1987 / 266.5   34: 113 / 9.11

The gate NEVER opens on the target (case25 amp 1.24284 = bit-identical baseline) and fires
precisely on the violent IC transients of the cases the term must not touch (case33 FAILS,
amp 1.433). The ACMWI study's "10 hits, case25-only" clamp statistic was a property of the
rejected ACMWI dhat, not of the current scheme. Decisive headroom datum: case25 sat_max
0.39430 vs case26 sat_max 0.39426 — a clean single-phase Mach-10 shock carries the SAME
face-level MWI-saturation signature as case25's under-damped reflected front (the incident
shock dominates both maxima). Conclusion: face-local INSTANTANEOUS signals (raw jump |du|,
MWI magnitude/saturation) cannot discriminate the case25 front from clean Hugoniots; the
G3 judgment and remaining options are logged in autoresearch/autoresearch-260714/goal.md
(iter 3).
