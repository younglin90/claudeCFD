# case07 wake wiggle — TR-BDF2 investigation (measured)

Goal: eliminate the case07 air-wake pressure wiggle (Denner 7.3.2 acoustic
reflection/transmission) while keeping the wave amplitude. Approach implemented:
restructure the coupled energy residual to conservative total energy `rho*E = rho*H - p`
and replace the single-stage BDF2 with a 2-stage **TR-BDF2** (L-stable, 2nd order) on the
acoustic path only. All numbers below are measured in this worktree (N=750, DENNER_ACID=1).

Wiggle metric = slope-sign reversals of p in x∈[0.45,0.95], eps=1e-3 Pa (`wig07.py`).
Reference (exact) has 0 reversals. NOTE: this metric reads higher than the brief's stated
baseline (52 here vs 31 in the brief) — a counting-convention difference; what matters is the
before/after on ONE consistent metric plus the amplitude measures p2p / amp_ratio / corr.

## Result summary

| case | metric | BDF2 (baseline) | TR-BDF2 | verdict |
|------|--------|-----------------|---------|---------|
| 07 | reversals | 52 | **40** | reduced, not ≤5 |
| 07 | wiggle p2p (Pa) | 1.693 | **0.798** | halved |
| 07 | amp_ratio_p | 0.9904 | 0.9858 | ≥0.95 OK |
| 07 | amp_ratio_u | 0.8797 | **0.9959** | restored |
| 07 | corr_p | 0.99457 | **0.99829** | improved |
| 04 | amp_p / corr_p | 0.9418 / 0.98792 | **1.0011 / 0.99888** | improved |
| 05 | amp_p / corr_p | 0.9463 / 0.99432 | **1.0006 / 0.99995** | improved |
| 35 | amp_p / corr_p | 0.7103 / 0.97607 | **0.9937 / 0.99939** | amp 0.71→0.99 |
| 36 | amp_p / corr_p | 0.9177 / 0.99655 | **1.0008 / 0.99982** | improved |

Full suite: **19/19 PASS** both before and after. Shock/BE cases (01,02,13,14,15,24,25,26,
27,28,30,31,33,34) are byte-identical to the original (gated: `tr_bdf2 = bdf2 && coupled`,
false for every BE case) — verified by dump-comparing all 14 cases against the pre-change
HEAD binary (scratch comparison, not committed).

## Success-criteria assessment
1. case07: reversals ≤5 → **NOT MET (40)**. amp_p≥0.95 → MET (0.986). corr_p≥0.99 → MET
   (0.998). Still PASS → YES.
2. Acoustic siblings 04/05/35/36 → all still PASS, **all amplitude/correlation IMPROVED**
   (most notably case35 amp_p 0.71→0.99).
3. Full suite 19/19 → YES; shock/BE cases byte-identical.

## Why the reversal target (≤5) is not reachable — the key finding

The residual wake, decomposed in x∈[0.45,0.95] (TR-BDF2):
- total RMS(p−p0) = 0.0852 Pa
- 2Δx odd-even (checkerboard / MWI pressure-decoupling) RMS = **0.0024 Pa (≈3%)**
- smooth (dispersive trailing wave, ~30-cell wavelength) RMS = **0.0795 Pa (≈93%)**

So the wiggle is dominantly a **smooth numerical-dispersion trailing wave**, NOT
BDF2 time-dispersion (the brief's stated cause) and NOT the collocated MWI checkerboard.

Evidence that contradicts the "pure BDF2 time-dispersion" hypothesis:
- TR-BDF2 is **verified L-stable + 2nd-order** on a scalar decay test (scratch program, numbers below):
  convergence rate → 2.000; amplification R(z) → 0 as z=−λdt → −∞ (R ≈ −4.83/|z|), i.e. it
  drives the highest-frequency TIME modes to zero — exactly what would kill a time-dispersion
  wiggle. Yet it only *halved* the wake amplitude (1.69→0.80 Pa) and left 40 reversals. An
  L-stable 2nd-order integrator therefore cannot be the whole story — the dominant residual is
  spatial/scheme dispersion.
- `ACID_NO_TRBDF2` reproduces the baseline BDF2 result **exactly** (52, 1.693, amp_u 0.8797) →
  the gating is clean and the TR-BDF2 delta is real.
- Reducing the spatial order (diagnostic: disable the 4th-order low-dissipation reconstruction)
  does NOT help — it makes the wake p2p **7× larger** (0.80→5.93 Pa) and correlation **worse**
  (0.998→0.925). The high-order low-dissipation reconstruction is beneficial; the smooth ripple
  is the price of a low-dissipation high-order scheme at this grid.

This is the classic dispersion-vs-dissipation trade the brief already hit with BE: BE removes
the wiggle only by adding enough numerical diffusion to also halve the wave (amp 0.99→0.56,
FAIL). TR-BDF2 sits at the other end — it keeps (indeed restores) the amplitude but leaves a
sub-0.1 Pa smooth ripple that the exact reference does not have. Getting to ≤5 reversals at
eps=1e-3 Pa would require a near-exact (non-dispersive) solution — much finer grid or a
dispersion-optimized spatial scheme — neither of which is reachable by the time integrator and
both of which are outside the brief's scope (spatial dissipation changes were already tried and
rejected: JST diverges/worsens; Minmod clips extrema and loses amplitude).

## What TR-BDF2 DID achieve (worth keeping)
- Velocity amplitude restored (case07 amp_u 0.88→0.996; case35 amp_p 0.71→0.99).
- Correlation improved across all 5 acoustic cases (→ 0.999).
- Wake wiggle amplitude halved (p2p 1.69→0.80 Pa).
- Full suite still 19/19; every BE/shock case byte-identical.
- A clean, self-starting, L-stable 2nd-order acoustic time integrator (no BDF2 constant-step
  / o2-store dependency; robust from step 1).

## Implementation notes (acid.cpp)
- Gated entirely on `tr_bdf2 = bdf2 && coupled` (acoustic cases only). `ACID_NO_TRBDF2` reverts.
- Conservative energy residual `rho*E = rho*H - p` for the TR path (removes the (p-p_o)/dt
  pressure-work source); BE energy residual kept verbatim → shock cases byte-identical.
- Stage 1 = trapezoidal over γ·dt with frozen F(Uₙ) (flux_w=0.5 + flux_expl); stage 2 = BDF2 on
  {n, n+γ, n+1}. γ=2−√2, a=1/(γ(2−γ)), b=(1−γ)²/(γ(2−γ)), c=(1−γ)/(2−γ).
- MWI (dhat + memory) and the inlet BC use each stage's own timescale (dt_mwi) / time (t_stage).
- TR path uses the FD pentadiagonal Jacobian (defect-correction → Jacobian-independent
  converged solution); the analytic Jacobian's energy-diagonal already coincides with the rho*E
  form, and only needs a flux_w scaling on the flux-coupling rows to restore the fast path —
  left as follow-up (does not affect the converged result).

## Follow-up options (out of current scope)
1. Dispersion-optimized / upwind-biased high-order reconstruction (adds dissipation exactly at
   the dispersive modes) to shrink the smooth ripple without the BE amplitude penalty.
2. Restore the analytic Jacobian for the TR path (flux_w scaling on the 6 flux-coupling `add()`
   rows) to recover the fast path — measurement here used FD.
3. Re-examine whether the reversal-count gate (eps=1e-3 Pa) is the right acceptance metric vs an
   amplitude/energy measure of the wake, given the reference is the exact non-dispersive solution.
