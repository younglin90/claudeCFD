# YADV Round 15 Plan — diagnose what actually drives case33's stall under `+ALPHA_IMPLICIT`

Planner output (Agent, subagent_type="Plan", model=opus), round 15. Diagnostic-only, no source
changes anticipated or made. Grounded in a fresh read of `cpp/denner_1d/src/acid.cpp` at
`ae2104f` (round 14 merged), `docs/YADV_RESEARCH.md` §19.2/21.1/22.1-22.7/23.1-23.5/24.3, and
`docs/YADV_ROUND_13_PLAN.md`.

## 0. Executive summary

Round 13 explained case24/34 (plain `ACID_YADV=1`) to textbook clarity: a dt-independent REMAP
mismatch in the pre-Newton alpha recovery makes `r_init` grow as `1/dt`. Its `+ALPHA_IMPLICIT`
control showed `dal_remap = 2.2204e-16` (DBL_EPSILON) -- the REMAP term is structurally absent
under that flag, because `+ALPHA_IMPLICIT` re-derives alpha at the CURRENT `(p,T)` on every Newton
call. Case33 stalls UNDER `+ALPHA_IMPLICIT`. Therefore case33's stall cannot be round 13's
mechanism -- this round measures what it actually is. Round 13 never ran `ACID_RINIT` on case33's
actual stall; this is the first time anyone looks at what case33 `+ALPHA_IMPLICIT` does when it
fails.

## 1. Code verification

`RINIT`/`RMISM` block confirmed unchanged by round 14 (round 14's `git show` diff touches only the
retry-exhaustion give-up block at a different location and the `if (diverged)` NaN fill -- neither
overlaps the instrument). Anchors re-grepped fresh: `rinit_dbg` decl `acid.cpp:606`, `RMISM` print
`~:1030`, `RINIT` print `~:1526`, `alpha_implicit` re-derivation inside `compute_R` `:1163-1172`
(runs at the top of every residual call), retry restart (`s0`, `Yv0`, `p_o`, `T_o`) `:745-750`.
Scheme facts ruling out confounders a priori: `coupled=true`, `bdf2=false` (no acoustic source,
transmissive BCs) for all of cases 24/33/34, so these run pure Backward Euler -- `bdf_c0 ≡ 1`,
eliminating round 13's own literature-clause BDF2/`Cold_*` exception without measurement.

A fresh build is mandatory: the worktree has no `build-cpp/`, and the main checkout's
pre-existing binaries predate round 13 (contain zero `RINIT`/`RMISM` strings).

## 2. Key structural insight -- what `dal_remap` measures differs by config

Under plain `ACID_YADV=1`: `dal_remap` is the genuine REMAP defect (alpha recovered at the
previous step's frozen Y meeting the current step's `(p_o,T_o)`) -- round 13's mechanism, real.

Under `+ALPHA_IMPLICIT`: since alpha is re-derived from `(Yv, s.p, s.T)` at the top of every
`compute_R`, `s0.alpha` at step entry already reflects the previous step's near-final `(p,T)`.
`dal_remap` under this config degenerates into a ONE-PICARD-ITERATE LAG -- `O(DBL_EPSILON)` if
the previous step converged, potentially non-negligible if the previous step was force-accepted
non-converged (`ACID_STALL_ACCEPT`). This distinction was not previously stated in any round's
documentation and is the key to correctly interpreting case33's numbers (which stall under this
config, unlike case24/34 which were only characterized under plain).

Also noted: `dal_remap`'s constancy ACROSS RETRIES (round 13's P2 finding) is structural, not
empirical -- every input to it is restored at the top of each retry, so it is bit-invariant within
a retry sweep by construction. The informative quantities are magnitudes and step-to-step
evolution, not retry-to-retry constancy per se. Round 13's P1/P3 findings and its DBL_EPSILON
control are unaffected by this correction.

## 3. Measurement plan (executed)

Whole-run captures (no `ACID_BLK_STEP` restriction) rather than per-step runs, to get the
step-over-step evolution needed to test compounding, and because trajectories under different
`ACID_STALL_ACCEPT` levels are not step-index-comparable (level 2 changes the `cfl_scale` policy).

- **R1**: `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RINIT=1 ACID_DBG=1`, case33. Canonical,
  unconfounded first-stall run (default `ACID_STALL_ACCEPT` off).
- **R2**: same + `ACID_STALL_ACCEPT=1`. Follows case33 through consecutive forced-accept steps to
  the eventual budget-exhausted give-up -- the compounding test.
- **C2**: `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RINIT=1`, case24 (a genuinely completing
  case under this config) -- the correct "healthy `+ALPHA_IMPLICIT` trajectory" baseline, sampled
  at every step, for a magnitude comparison.

## 4. Interpretation framework (applied)

A. Does `r_init` grow as `1/dt`? (compare retry>=1, since retry 0 has clamp/ramp peculiarities).
B. Which `RINIT` component carries it (`fene` -> energy dominant, or momentum/continuity)? C. Is
`dal_remap` zero, and does it STAY zero across forced-accept steps (the compounding test)? D.
(not needed this round -- A/B/C were decisive) `ACID_RHIST` per-iteration trace of how Newton
actually fails.

## 5. Decision rules

D-DIFF (genuinely different mechanism): `r_init` grows as `1/dt` (A1) AND/OR the carrier is NOT
round 13's alpha-REMAP channel (`dal_remap` clean) -- fires if case33 shares round 13's residual
SHAPE but not its SOURCE. D-SAME: variant of round 13's already-characterized/refuted mechanism.
D-INCONC: growth measured but no `RMISM` field explains it (needs a new instrument). D-NULL: stall
not reproducible as described (would outrank everything else).

## 6. Non-goals

No fix of any kind attempted or implemented this round, regardless of findings -- diagnostic only.
No re-attempt of `ACID_YADV_HREINIT` or any single-field initial-guess correction (already refuted,
round 13 §23.2 S4). No touching case24/34's status, `ACID_STALL_ACCEPT`'s mechanism, or the round-
14 `diverged` block. No `cases.cpp`/`validation.cpp`/script edits. No RH-residual re-measurement
(case33 `+IMPLICIT` never completes; no clean run exists to measure). No extension of `RINIT`/
`RMISM` itself (tempting, but a source edit -- note next-round instrument requests instead of
implementing them).

## 7. Gates (deliberately light -- no source changes)

Because this round writes no code, the four hard validate gates (which exist to prove a source
change is a no-op) are not required and were not run. Required and run: G0 build succeeds
(mandatory rebuild, confirmed necessary), G1 unit test passes, G2 `git status --short -- cpp/`
clean at round end (no source drift), G3 `ACID_RINIT` unset emits zero RINIT/RMISM lines
(re-verification of round 13's own gate on the current build), G4 (the important one) `RINIT`'s
`r` self-check against `RHIST`'s `n0` repeated specifically in case33's `+ALPHA_IMPLICIT`
configuration (round 13 only validated this on case24 plain) -- confirmed exact match, meaning
the instrument is valid in a configuration it had never been checked in before.

## 8. Candidate fixes for a future round (NOT implemented)

If a future round wants to pursue this: identify the physical/numerical origin of the large,
growing, dt-independent `dh`/`drho` mismatch at cell 79-81 (consistently, across all sampled
steps) -- likely needs a per-cell trace of local shock strength/state jump rather than the current
state-mismatch instrument, which characterizes the symptom (a mismatch exists) but not why it is
so much larger here than anywhere else measured. Round 13 §23.3's simultaneous `(T,rho,h)`
reconciliation remains a candidate but is now motivated by TWO independent findings (case24/34's
splitting lag AND case33's h/rho mismatch) with DIFFERENT root causes -- a fix for one is not
guaranteed to address the other.

## 9. Literature

No search performed. Round 13's clause (operator-splitting initial-guess lag is standard, no
citation needed) is inherited for the mechanism SHAPE; case33's actual root cause was not
identified this round (deferred, per §8), so there is nothing yet to search literature for --
premature to search before the mechanism itself is known.
