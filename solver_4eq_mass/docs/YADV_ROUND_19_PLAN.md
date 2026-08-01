# YADV Round 19 Plan — Locating the exact first divergence of case34 under `ACID_STALL_ACCEPT=1` + `ACID_TSAT_STALL=1`

Diagnostic / control-flow tracing round. No fix. No source change expected.

Advisor spot-check: confirmed `only_reason1` (acid.cpp:784 decl) is consumed ONLY at line 2377
(`const bool r1_only = (stall_accept_lvl >= 2 && retry > 0 && only_reason1);`) -- genuinely dead
at `stall_accept_lvl==1`, confirming the plan's key correction to round 18's "free exclusion"
framing (edit-free, not behaviour-free). `tsat_stall` block confirmed at line 2330.

## 0. Channel enumeration (the round's core analytical result)

`tsat_stall` has exactly one reachable use site (the `bad=true; stall_reason=5;` block). It can
only perturb the trajectory through `bad` and `stall_reason`. Two channels, exhaustive:

**C1**: a retry whose post-solve state was clean (`bad==false`) now has a saturated cell -> `bad`
flips false->true -> the step is rejected and retried at half `dt`. When it eventually succeeds at
a later retry index, the `cfl_ramp` update takes the `cfl_scale *= 0.5^retry` branch instead of
`*1.5` -- cascades into every subsequent step's `dt`.

**C2**: a retry that was ALREADY bad with `stall_reason==1` now gets `stall_reason==5` instead ->
excluded from `ACID_STALL_ACCEPT`'s candidate capture. Zero effect UNLESS that step exhausts all 14
retries AND the excluded retry was the ratio winner -- in which case the accepted candidate itself
changes (`acc_retry`/`acc_dt` would print differently in the `STALL-ACCEPT:` line).

Since round 18 reported all four `STALL-ACCEPT:` events identical (including `acc_retry`, an exact
integer), C2 is essentially pre-excluded by round 18's own data -- C1 is the primary hypothesis.

**Decisive test, no new code needed**: a C1 event at `(step, retry)` implies the BASELINE run's
`ACID_TSAT=2` block B (`TSAT-ACCEPT`, fires exactly when a step is accepted) prints a line at that
`(step, retry)`, and `accepted_steps_hi > 0` in its `TSAT-TOTAL` summary. One grep settles it.

## 1. Correction to round 18 (recorded here, not edited there)

Round 18's own prediction for this exact combination was keyed to the `+ALPHA_IMPLICIT` Stage-0
column (`calls_hi=0` for case34 there) -- but the run in question is PLAIN `ACID_YADV=1`, whose
Stage-0 column read `calls_hi=78, first_hi_step=14`. The deviation was predictable from round 18's
own data using the correct column.

## 2. Hypotheses (pre-registered)

H0 (null/artefact): the two runs aren't deterministic even without the new flag -- falsified by
running the SAME config twice and comparing. H1 (primary): C1 fires at the first step S where an
accepted state saturates; `dt_B/dt_A` deviates for a few steps after S then settles to `1+O(1e-6)`
-- same plateau, small persistent offset, not amplifying. H2 (fallback if `accepted_steps_hi==0`):
C2 -- a reason-5 exclusion changed the acceptance winner at one of the four retry-exhausting
steps (229/231/232/233), contradicting round 18's "identical events" report. H3 (residual): neither
-- an unexplained anomaly, report as such, do not speculate.

## 3. Protocol (executed)

Stage A0: build fresh (main-repo binaries stale, predate ACID_STALL_ACCEPT/ACID_TSAT entirely),
run the bare baseline (`ACID_YADV=1 ACID_STALL_ACCEPT=1`, no instrumentation) TWICE, compare --
determinism control. Stage A1: same config + `ACID_TSAT=2 ACID_RINIT=1 ACID_DBG=1`, compare against
Stage A0's bare baseline -- instrument-neutrality control (re-verifies round 17 G6 / round 13 G3 in
a never-before-checked configuration). Stage A2: grep the instrumented baseline's `TSAT-TOTAL` for
`accepted_steps_hi` -- the decisive one-grep test. Stage B: run with `ACID_TSAT_STALL=1` added,
diff the full RMISM/RINIT/TSAT/RETRY line sets against the baseline to find the exact first
divergence. Stage C: at the identified step S, confirm the extra `RETRY` line's `dt` matches the
baseline's accepted `dt` at that step (closing the attribution), and trace `dt_B/dt_A` for several
steps after S to characterize the propagation. Stage D (only if H1 refuted): check whether
`acc_retry` differs at any of the four `STALL-ACCEPT` events (H2). Stage E (cheap, bounded):
same-config sweep for case24/33 to see if one predicate (`accepted_steps_hi>0` under
`ACID_STALL_ACCEPT=1`, plain-ON) explains all three of round 18's G7 outcomes at once.

## 4. Non-goals

No fix (affects no published configuration -- round 12's numbers use `ACID_STALL_ACCEPT` alone).
No change to `ACID_STALL_ACCEPT`, `ACID_TSAT_STALL`, `T_from_hstat`, `compute_R`, the `cfl_ramp`
blocks, or the reason taxonomy. No `cases.cpp`/`validation.cpp` edits. No new env var. No
promotion decision for `ACID_TSAT_STALL`. No literature search.

Contingency (only if attribution is ambiguous): a single print-only hunk appending
`" step=%d reason=%d"` to the existing `ACID_DBG`-gated `RETRY` line -- default posture is to NOT
take this, prefer reporting ambiguous attribution honestly over adding code. If taken, full
round-16/17/18 gate battery applies.

## 5. Gates (round-15 light posture -- no source change expected)

G0 build+unit. G1 unit passes. G2 `git status --short -- cpp/` clean at round end. G3 determinism
control passes. G4 instrument-neutrality control passes (incidental new-configuration check). G5
round 18's G9 invariant re-confirmed on this exact configuration (zero `TSAT-ACCEPT` in the
flag-on run). G6 no scratch files left in the repo.
