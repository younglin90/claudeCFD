# Round 18 Plan — Implement F2'' (`ACID_TSAT_STALL`): report T-ceiling saturation as stall reason 5

Planner output (Agent, subagent_type="Plan", model=opus), round 18. Authorized by an explicit
Advisor decision: implement F2'' (round 17's corrected form of the F2 fix candidate).

Advisor spot-check: confirmed `bool T_from_hstat(...)` at acid.cpp:334; C1 call site
`if (T_from_hstat(hstat_i, s.p[i], s.alpha[i], A, B, s.T[i], Tnew))` at :1228;
`int stall_reason = 0;` at :753; `bool bad = (ajac && coupled && !conv_inner && rbest >= r_init);`
at :2296, `if (bad) { stall_reason = 1; stall_cell = -1; }` at :2297;
`if (stall_reason == 1 && r_init > 0.0 && std::isfinite(rbest)) {` (ACID_STALL_ACCEPT eligibility)
at :2355; `why[]` array at :2438 -- all exact matches.

## 1. Key design decision: NO signature change to T_from_hstat

The obvious approach (add an out-param reporting "did the clamp fire") is REJECTED. Three reasons:
(a) F2'' is specified as "saturated in the ACCEPTED iterate" -- a property of the state
(`s.T[i]>=1.0e6` after C1's loop), not of whether the internal 30-iteration clamp fired
transiently mid-solve (a strictly larger, unmeasured set -- the ceiling comment at :307-309
explicitly says transient overshoot-then-recovery is by design). An out-param would push an
UNMEASURED predicate into the accept/reject decision, forfeiting round 17's entire V-SAFE argument.
(b) Zero risk to C2 (the `ACID_AJAC_CHECK` probe, discards the bool already) or the four
`compute_R(); // restore` sites -- `T_from_hstat`/`compute_R` stay pure functions of state.
(c) The airtight OFF-path proof (round 17's `calls_hi==0` measurement) exists ONLY for the state
predicate `s.T[i]>=1.0e6`, which round 17's `ACID_TSAT` block A already scanned after every
`compute_R` call -- so `calls_hi==0` deductively implies the new predicate is false at every `bad`
evaluation. That deductive chain does not exist for a transient in-loop predicate.

Consequence: `T_from_hstat` is not edited. This is a deliberate deviation making F2'' more faithful
to its own pre-registration, not less.

## 2. Design decision: precedence 2/3/4 > 5 > 1, insert between existing lines 2297/2298

Insert the reason-5 check AFTER the reason-1 assignment (`if (bad) { stall_reason = 1; ... }`) and
BEFORE the finite/speed scan. This single placement gives three things at once: (1) reason 5
displaces reason 1 (necessary since `ACID_STALL_ACCEPT`'s eligibility keys off `stall_reason==1`);
(2) reasons 2/3/4 (hard failures) still override reason 5 if both occur, preserving existing
STALLED-DETAIL semantics; (3) `ACID_STALL_ACCEPT` exclusion is FREE -- zero edits at the
acceptance-eligibility site (:2355-2363) -- because a retry that sets reason 5 can never satisfy
`stall_reason==1` and therefore can never be captured into `acc_s`. The exclusion is exact by
construction, not approximate.

NOT `ajac`-gated (reason 1's term is `ajac &&`-gated; saturation is a property of state, not of
Jacobian mode, so this must be checked regardless -- exercises the FD-invariance config too).
`coupled`-gated (matches `ACID_TSAT` block A; provably equivalent to ungated since the segregated
path's convex T-update at :2263 can never reach 1e6 from below, but states intent explicitly).

## 3. Gating: new env var `ACID_TSAT_STALL`, default OFF, global (NOT yadv-gated)

Flag-gated, not unconditional: round 17's V-SAFE verdict covers OFF with the DEFAULT (analytic-
Jacobian) config only; plain ON, `+ALPHA_IMPLICIT`, and FD-invariance were never swept for
saturation before this round (Stage 0, below, does that sweep). Consistent with
`ACID_YADV_HREINIT`/`ACID_STALL_ACCEPT` precedent (both stayed flag-gated after their own rounds'
partial validation).

Global, NOT `yadv`-gated: a `yadv` condition would make the headline OFF-byte-identity gate
structurally vacuous (exactly the defect round 17 found in `ACID_RCELL`/`ACID_RINIT`'s `yadv`
gating -- they cannot observe OFF at all). Since default is 0/OFF, global scope cannot reach any
path unless explicitly set.

`ACID_TSAT_STALL` (integer, `atoi`+`max(0,...)`, 0/unset/malformed = fully OFF), pairs with round
17's `ACID_TSAT` naming family.

## 4. Exact diff (5 hunks, one file, only 1 hunk has executable code)

Hunk 1 (flag decl, after tsat_* counters ~line 549): `const int tsat_stall = []{ const char* e =
std::getenv("ACID_TSAT_STALL"); return e ? std::max(0, std::atoi(e)) : 0; }();` with full
justification comment per docs above.

Hunk 2 (comment-only, taxonomy at :753-754): extend the `stall_reason`/`stall_cell` comments to
document reason 5.

Hunk 3 (the only executable change, inserted between :2297 and :2298):
```cpp
        if (tsat_stall > 0 && coupled) {
            for (int i = 0; i < n; ++i)
                if (s.T[i] >= 1.0e6) { bad = true; stall_reason = 5; stall_cell = i; break; }
        }
```

Hunk 4 (comment-only, extend the Stage-3a eligibility comment near :2348-2352 explaining the free
exclusion).

Hunk 5 (`why[]` array at :2438-2439): add `"T-ceiling-saturated"` as the 6th entry (index 5).

Nothing else touched: no `T_from_hstat` edit, no C1/C2 edit, no OMP pragma edit, no `ACID_TSAT`
block edit, no `ACID_STALL_ACCEPT` code edit (only its adjacent comment), no
`cases.cpp`/`validation.cpp`/header/test edit.

## 5. Pre-registered predictions

**Stage 0 (mandatory, before any flag-on gate)**: sweep all 19 graded cases x 5 configs (OFF,
OFF+FD, plain-ON, plain-ON+FD, `+ALPHA_IMPLICIT`) with `ACID_TSAT=1` (proven no-op, round 17 G6 --
free measurement). This is the round's primary new empirical contribution -- round 17 only swept
OFF. Predictions to be falsified: OFF/OFF+FD `calls_hi=0` all 19 (already known); plain-ON
`calls_hi=0` all 19 (INFERRED from §26.2's "global drift not localized blister", never measured --
single most likely place for a surprise); `+ALPHA_IMPLICIT` `calls_hi>0` for case33 only
(~13719, matches round 17's G9), `calls_hi=0` for case24/34.

Per-config predictions (all conditional on Stage 0 confirming `calls_hi=0` where claimed): flag
unset -> byte-identical everywhere (only compiled change is one branch on an int read once per
case, no FP arithmetic added). OFF+flag set -> 19/19 byte-identical, zero STALLED lines (deductive
from Stage 0 + round 17). plain-ON+flag set -> 15/19 byte-identical, CONDITIONAL on Stage 0's
plain-ON column reading all-zero. `+ALPHA_IMPLICIT`+flag set -> case33 fails EARLIER (step 0, not
~100) and more cleanly (reason 5, not 1) but pass_count stays 14/19 and case33's JSON row is
predicted UNCHANGED (already all-NaN/finite=false/pass=false both ways, since case33 was already
`diverged` via round 14's Stage 3c) -- visible change is stderr-only. case24/34 must not move.
FD-invariance -> 13/19 byte-identical (reason-1 term is ajac-dead there; with calls_hi=0 no bad
change). `ACID_STALL_ACCEPT=1`+flag, case24/34 -> byte-identical (requires Stage 0's
`+ALPHA_IMPLICIT` column showing calls_hi=0 for them). `ACID_STALL_ACCEPT=1`+flag, case33 -> ZERO
STALL-ACCEPT lines (down from nonzero), STALLED at step 0 with reason=T-ceiling-saturated --
strictly better than round 12's "grind until budget exhausts": fails faster and cleaner, not a
different final outcome (case33 never completed either way).

Internal cross-check: with BOTH `ACID_TSAT=1` and `ACID_TSAT_STALL=1`, `accepted_steps_hi` must be
0 in every TSAT-TOTAL line, every config -- a nonzero reading is a self-contradiction (hunk 3's
placement or predicate would be wrong).

## 6. Gates

G0 build+unit. G-Stage0 (mandatory, before G2/G3 flag-on runs): the 5-config x 19-case sweep.
G1 (headline) OFF 19/19 byte-identical to fresh pre-edit baseline, BOTH flag-unset AND
`ACID_TSAT_STALL=1` (the flag-set variant is the round's central claim -- round 17 proved it
structurally, this confirms through the actual new code path). G2 plain-ON 15/19 byte-identical,
both flag states. G3 `+ALPHA_IMPLICIT` 14/19 byte-identical both flag states, case24/34 must not
move in either; case33's row predicted identical, deviation reported field-by-field if found. G4
FD-invariance 13/19 byte-identical both flag states, both configs (`ACID_NO_AJAC=1` alone and with
`ACID_YADV=1`). G5 flag-unset no-op across G1-G4, zero new stderr. G6 malformed `ACID_TSAT_STALL`
(empty/0/abc/-3) -> no crash, behaves as unset. G7 `ACID_STALL_ACCEPT` interaction: case24/34 byte-
identical + identical STALL-ACCEPT counts with flag on vs off; case33 STALL-ACCEPT
count/STALLED-step/reason recorded both ways. G8 (mandatory positive control) case33
`+ALPHA_IMPLICIT` with `ACID_TSAT_STALL=1 ACID_DBG=1` must print
`STALLED-DETAIL: reason=T-ceiling-saturated` with `T=1.0000e+06` at the reported cell -- if reason
5 never fires anywhere, the mechanism is untested and must be reported as such, not claimed working
from byte-identity alone. G9 the `accepted_steps_hi==0` invariant, both flags on, every config. G10
`git status --short -- cpp/` shows only `acid.cpp`.

Failure policy: any G1-G4 byte-identity failure with the flag UNSET is a hard stop (the change is
not inert). A change WITH the flag set is a measurement, not necessarily a failure -- but falsifies
sect.5's prediction for that config and must be reported as such.

## 7. Non-goals

No F1, no F3. No `ACID_YADV_HREINIT`/`ACID_RCELL`/`ACID_RINIT`/`ACID_TSAT` print edits (read
freely, edit nothing). No `cases.cpp`/`validation.cpp` edits. Case29 stays excluded (round 17
sect.27.5 out of scope). The 1e6/1e-6 clamp VALUES are not changed -- only how the upper one's
saturation is reported. No promotion to default under any finding -- `ACID_TSAT_STALL` stays
default OFF; `ACID_YADV` recommendation unchanged (default OFF, 15/19). No `T_from_hstat` signature
change (sect.1). No code change at the ACID_STALL_ACCEPT eligibility site itself (sect.2, comment
only).

## 8. Literature

Not needed -- implementing an already-designed, already-risk-assessed reporting/acceptance-policy
mechanism, no new numerical method or closure.
