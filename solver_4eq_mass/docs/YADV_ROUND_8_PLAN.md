# Round 8 Execution Brief — Phase 2 Stage 3a (T-pathway)

Produced by Agent(subagent_type="Plan", model=opus) during round 8 of the `yadv-round` loop.
Spot-checked by the Advisor session before implementing: case14's IC (`T_air=6.942K` reproduced
exactly from `temperature_for_density_pressure`), J1/J2 line anchors (all confirmed exact), and
the closed-form claim `hstat_mix = Y*h_a + (1-Y)*h_b` (independently re-derived: `N/D` is by
construction the mass-fraction-weighted average, and `Y := alpha*rho_a/rho` makes this identity
trivial, not something that depends on the a_p/a_T formulas at all).

## Measure-first methodology (the brief's own structure, followed exactly)

The brief's central finding, from static analysis before any code was written: round 5's
`hsT<0` lead for case14 was a **probe artifact** (it combined case14's cold-air temperature with
a mid-range `alpha=0.5` that no case14 cell actually occupies). Computed exactly: `hsT<0` requires
`T < 78.2K` uniformly over the air|water pair's full `(p,Y)` range, and every physically-reachable
mixture of case14's two IC states sits at least 4.5x above that bound. The brief recommended
measuring first via a temporary diagnostic rather than assuming the lead was dead or alive.

**Phase A (diagnostic, implemented and run first).** A default-off, `yadv`-gated stderr print of
`hsT`'s sign, added to the J1 loop, run against case14 and case15. Result: `hsT<0` **is** real,
but confined entirely to case14's very first timestep's Newton iterations (a single cell near the
interface, an interface-formation transient), never recurring at any later step. This matched the
brief's own decision table's middle row ("transient first-step artifact... implement 3a"), not its
"persistent, do not implement" row. The diagnostic was removed after use, per its own instructions.

**Phase B (Stage 3a implementation).** Starred the T-pathway (`D_T`->`D_Ts`, `N_T`->`N_Ts` via
`a_T = dalpha_dT_massfrac`) in the J1 loop, mirroring Stage 1's p-pathway exactly, and computed the
TOTAL derivatives `alp_p = a_p + a_T*dTp`, `alp_h = a_T*dTh`, `alp_u = a_T*dTu` (the ordering
dependency the brief flagged -- these must be computed AFTER `dTp/dTh/dTu`, which is what makes
them "total" rather than partial). Extended the J2 loop with two more diagonal columns (h, u)
mirroring the existing p-column exactly.

**A genuinely new mathematical result surfaced along the way**: because `hstat_mix = Y*h_a(p,T) +
(1-Y)*h_b(p,T)` exactly, and NASG `h_k` is linear in `T` and `p`, the starred partials have EXACT
closed forms:
```
hsT* = Y*cp_a + (1-Y)*cp_b     (strictly positive, in [min cp, max cp] -- can never cross zero)
hsp* = Y*b_a  + (1-Y)*b_b
D_p* = rho^2*(Y*zeta_a/rho_a^2 + (1-Y)*zeta_b/rho_b^2)
D_T* = rho^2*(Y*phi_a /rho_a^2 + (1-Y)*phi_b /rho_b^2)
```
Verified in the unit test to 6.8e-11 absolute (a test-tolerance bug was found and fixed along the
way -- see below). These retroactively validate Stage 1's already-shipped `hsp*`/`D_p*` code, which
had never been checked this way before, and prove starring the T-pathway REMOVES an existing
`1/hsT` near-singularity (the unstarred `hsT` provably crosses zero for the air|water pair below
~78K; the starred form never can).

## A live bug found in the round's own unit test (not the derivative formula)

The new closed-form identity test initially failed 510 checks with an absurd worst error
(`1.79e+287`). Root cause: `hsp_closed = Y*b_a+(1-Y)*b_b` is EXACTLY zero for the air|vapor pair
(both `b=0`), and the test's relative-error comparison divided by a `1e-300` floor -- the exact
same class of bug round 5's own unit test had (an actual roundoff-scale difference near a
legitimate zero, amplified to nonsense by too-tiny a denominator). Fixed with an absolute-or-
relative combined tolerance (`max(1e-9, 1e-12*|closed|)`), not a pure ratio. After the fix: worst
absolute error 6.8e-11, all checks pass, confirming the derivative formula itself is correct and
the bug was purely in the test's own tolerance logic.

## Result: Stage 3a is a measured REGRESSION on case14, gated behind a new flag

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` (Stage 1+2 only, unaffected by this round) stays at
14/19, bit-identical to rounds 6/7's case14 metrics. Adding Stage 3a's T-pathway UNCONDITIONALLY
(the brief's originally-specified design) does not flip case14's pass/fail (it was already
failing) but its solution quality collapses:

| metric | round 6/7 (Stage 1+2) | round 8 (+Stage 3a, unconditional) |
|---|---|---|
| `l2_p` | 0.0144718 | 0.511828 |
| `l2_u` | 0.132392 | 0.663105 |
| `corr_p` | 0.99956 | 0.594481 |
| `corr_u` | 0.954309 | 0.227335 |
| `corr_rho` | 0.979441 | 0.746994 |
| `amp_ratio_u` | 1.1221 | 4.59619 |

This confirms the risk the round's own brief flagged in advance (risk R8): giving the Jacobian
the FIXED-POINT T-derivative while the residual still computes the ONE-CALL-LAGGED map is a family
mismatch -- the mirror image of round 4's original mistake (there, the residual was nonlinear
while the Jacobian assumed zero derivative; here, the Jacobian assumes a derivative family the
residual does not itself evaluate).

**Because this sits inside the already-established, already-validated `ACID_YADV_ALPHA_IMPLICIT`
flag** (round 6/7's genuine win: case13/25 recovered, 14/19), merging Stage 3a unconditionally
would silently degrade that validated configuration. Gated behind a NEW flag,
`ACID_YADV_ALPHA_IMPLICIT_T` (default off), following the exact precedent round 4 set for its own
mixed result. Verified: `ACID_YADV_ALPHA_IMPLICIT=1` alone reproduces round 6/7's case14 metrics
bit-for-bit; adding `ACID_YADV_ALPHA_IMPLICIT_T=1` reproduces the regression exactly.

## Gates (all held)

OFF 19/19 + 9/9 byte-identical. Plain `ACID_YADV=1` 15/19. FD-invariance (`ACID_NO_AJAC=1`) 12/19,
exact same failure set as rounds 4-7. `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` (the new flag OFF)
14/19, bit-identical to round 6/7.

## Advisor decision on Stage 3b (escalated by the round, per protocol -- NOT designed or approved)

The round's brief surfaced a strong argument for 3b (substituting `alpha(Y,p,T)` inside
`T_from_hstat`'s own inner Newton): the closed-form result above means the h->T inversion becomes
EXACTLY linear in T at fixed Y, collapsing the current ~30-iteration inner Newton (documented as
"the hottest kernel... ~60 EOS evals per cell per compute_R") to a single division, AND eliminating
the non-monotonicity that causes `hsT<0` in the first place -- at the cost of flipping the
FD-invariance gate (a residual change, not just a Jacobian change) and touching `T_from_hstat`'s
signature.

**Advisor decision: decline 3b for now.** Reasoning: (1) 3b's own case, per the brief, is
performance/robustness/consistency, explicitly NOT a case14 fix -- case14's states don't reach the
non-monotone region anyway (confirmed by Phase A). (2) Phase 2's goal is already substantially met
(13/25 recovered, case15 explicitly out of scope since round 7, case14 now also shown not to be
reachable via T-pathway Jacobian tricks). (3) 3b is a larger, more invasive change (residual
signature change, new gate semantics, `ACID_DENSE` probe touched too) for a benefit that isn't
blocking any currently-open target. Revisit only if a future need for `T_from_hstat` performance or
the latent no-convergence-check robustness defect (noted, not fixed, this round) arises.

## Recommendation

Phase 2's alpha-implicit-Jacobian line of investigation (Stages 0-3) is complete: Stage 1 (p-pathway)
is a genuine, validated win (12/19 -> 14/19, case13/25 recovered); Stage 2 (flux-blend diagonal) is
a measured no-op with real diagnostic value (corrected case15's blocker); Stage 3a (T-pathway) is a
measured regression, gated off. Proceed to Stage 4 (consolidation) next round: full sweep tables,
wall-clock/iteration-count summary across all stages, and the Advisor decision on whether
`ACID_YADV_ALPHA_IMPLICIT` (Stage 1+2 only) should be considered for promotion consideration
(NOT `ACID_YADV` itself, which stays default OFF regardless per every prior round's rule).

## Non-goals honored

No edits to `cases.cpp`/`validation.cpp`. Cases 24/33/34 and case15 not chased. No tuning constants
(the two new env flags are structural gates matching round 4's own precedent, not tunables). 3b not
implemented, per the Advisor decision above.
