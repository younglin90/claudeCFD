# MWI dt-independence research — Bartholomew unified formulation (NEGATIVE RESULT)

Date: 2026-07-13. Status: **faithful implementation measured, REJECTED — code reverted,
baseline (transient-only a_P dhat) retained.** This documents the measured evidence so the
dead-end is not retried blindly (WIGGLE_RESEARCH.md-style note).

## Goal

Fix case25's reflected-shock overshoot at its root: the MWI pressure dissipation collapses
with dt (`dhat ~ dt` via the transient-dominated a_P), so small acoustic-CFL steps UNDER-damp
the pressure-velocity coupling at strong shocks (the collocated small-dt checkerboard;
Bartholomew et al., JCP 375 (2018) 177-208, Eq. 54-55 discussion). Mandate: implement the
paper's unified, principled dt-independent form verbatim — one GLOBAL formula, no invented
constants, no per-case knobs — from
`papers/library/pdf/2018_Bartholomew_Denner_MWI_collocated_main.pdf` (extracted via fitz).

## The paper's unified form (extracted, Secs. 3.3-3.4, 5.1, 6)

- Eq. 14: `d_P = V_P/a_P`, with `a` = the ADVECTIVE (+shear) momentum coefficients ONLY
  (inviscid Euler here → `a_adv,f = rho_f*|ubar_f|`, harmonic `rho_f` = the Eq. 48
  `d_f=(d_P+d_F)/2` face average).
- Eq. 52: `dhat_f = d_f/(1 + c_f d_f)`, `c_f = rho_f/dt` → stable closed form
  `dhat = dx/(rho_f*(|ubar| + dx/dt))`.
- Eq. 51/100 memory term: `+ c^O_f dhat_f (theta^O_f - ubar^O_f)` with `theta^O` = the
  previous step's ACTUAL advecting velocity. dt-independence mechanism: as dt→0 the per-step
  `dhat → dt/rho` vanishes but the feedback `c^O dhat → 1` accumulates the correction, giving
  the steady fixed-point dissipation `dhat/(1-c^O dhat) = dx/(rho|u|)` — the dt-independent
  advective limit.
- Eq. 91/100 density weighting: `gpbar = rho_f * avg(gradp_P/rho_P, gradp_F/rho_F)`
  (harmonic rho_f); required at density discontinuities (Sec. 5: without it the light phase
  is "accelerated without limit").

## Measured configurations (all on the 19/19 baseline, DENNER_ACID=1)

Baseline (transient-only a_P, arithmetic-rho, rebuild theta_o): 19/19;
case25 amp_ratio_p=1.2428, hf_p=0.8417, corr_p=0.9943 (the target of this work);
low-Mach 04/05/07/35/36 amp_ratio_p = 1.0011/1.0006/0.9858/0.9937/1.0008.

| config | change | suite | case25 amp/hf/corr | blockers (measured) |
|---|---|---|---|---|
| A | unified dhat only | 14/19 | 1.187 / 0.673 / 0.9946 | 07 amp 1730; 14, 15, 26, 28 fail |
| B | A + Eq.91 density-weighted gpbar (incl. dth_dp Jacobian weights) | 15/19 | 1.198 / 0.690 / 0.9945 | 07 amp 720.4 corr 0.307; 14 corr_p 0.519 hf 1.171; 15; 28 corr_p 0.167 |
| C | B + faithful Eq.100 memory (theta_o = previous CONVERGED theta instead of the steady-form rebuild; retry-safe theta_o restore) | — | 1.0003 / 0.617 / 0.9946 | ALL acoustic cases explode: 04 amp 20538, 05 amp 11.5, 07 amp 844491, 35 amp 18682, 36 amp 9051 |

Config B passes every case25 acceptance target (amp<=1.20, hf reduced, corr>=0.99) and keeps
04/05/35/36 at amp 1.000/1.000/0.993/0.999 — but is disqualified by 07/14/15/28.
Config C shows the paper's fixed-point analysis working perfectly at the shock (case25 amp
1.0003 — overshoot GONE) while its near-unity memory feedback (c^O dhat = dx/(dx+|u|dt) ≈
0.9997 at case07) integrates acoustic-scale MWI corrections into divergence.

## Why the faithful form fails here (analysis, consistent with the measurements)

1. The unified fixed-point dissipation is the ADVECTIVE limit `dx/(rho|u|)`. For a resolved
   acoustic wave in a slow mean flow (case07: u0=1 m/s) this exceeds the acoustically
   consistent scale `dx/(rho*a)` by a/|u| ~ O(10^3). Bartholomew's framework and all its
   validation cases are incompressible/low-Mach (elliptic pressure, Δ³p→0 when resolved);
   a compressible acoustic field has finite Δ³p everywhere, so the giant fixed-point filter
   actively injects velocity.
2. The Eq. 91 density weighting assumes a momentum-balanced flow (grad p ∝ rho). An acoustic
   wave at an impedance-jump interface (air-water Z ratio ~3.4e3) has grad p set by
   IMPEDANCE, not ∝ rho, so the weighting mis-cancels by O(1) exactly at the interface faces
   where the harmonic-rho unified dhat is ~200x the baseline (harmonic ≈ 2*rho_air vs
   arithmetic ≈ rho_water/2). Measured: weighting improved 07 from amp 1730 to 720 — the
   right sign, nowhere near enough. 35/36 (Z ratios <= 2.5) survive; 07 does not.
3. The baseline's ARITHMETIC-rho transient-only dhat is the accidental (but load-bearing)
   acoustic guard: at a liquid interface the heavy phase makes dhat tiny (dt/rho_arith),
   suppressing both effects. Any unified/advective form that raises the interface dhat to
   its "physical" magnitude re-exposes them.

## Conclusion / paths not taken

- The faithful unified MWI cannot be a single global formula for THIS suite: it provably
  fixes the case25 small-dt under-damping (config C: amp 1.0003) and provably destroys the
  compressible-acoustic cases. No amount of "one more term from the paper" bridges this —
  the missing ingredient would be an acoustically-consistent a_P (e.g. rho*(|u|+a)-type),
  which is NOT in the paper (an invented constant/scale → banned by mandate; note the
  a-scaled floor was explicitly disallowed).
- Denner's own ACID Eq. 21 e_P (advection-coefficient) lever was measured in earlier work
  (see .claude/rules/denner-pitfalls.md) with the same trade-off flavour and was removed as
  a per-case knob. Case25's single-cell overshoot remains the documented accepted residual
  (Denner's own Fig. 23 shows the same feature).
- If this is ever revisited: the decisive experiments are already tabulated above — start
  from config B (best all-round), and the open problem is a case-blind, paper-grounded
  acoustic limiter of the advective fixed point, not more faithfulness.

All numbers reproduce with: `DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate`
on the respective configuration (code reverted; configs preserved in this note only).
