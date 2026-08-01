# Round 20 Plan — Promote `ACID_TSAT_STALL` (F2'') to unconditional default

## 1. Context and decision

Round 18 implemented F2'' behind `ACID_TSAT_STALL` (default OFF, deliberately not `yadv`-gated so
its no-op claim on the published OFF path is testable rather than structural). Round 19 fully
closed the one open concern against it (case34's `ACID_STALL_ACCEPT` perturbation, `YADV_RESEARCH.md`
§29 — fully explained, settles to the same physical end state, not a mystery). Nothing has been
outstanding against the mechanism since round 19.

Round 18 §28.3 already measured that turning the flag on:
- is byte-identical on OFF, plain-ON, `+ALPHA_IMPLICIT` (all three swept, zero exceptions), and
- **recovers cases 27/28 from silent NaN to genuine PASS** on the FD-Jacobian path.

Per round 14's precedent for correctness fixes (`diverged=true` shipped with no opt-out and an
explicit comment forbidding one — reason 5 here is the same class of defect: a state the EOS cannot
represent, not a research toggle with two legitimately-supported settings), this round promotes F2''
to unconditional and **deletes** the env var entirely, rather than merely flipping its default.

## 2. The edit (4 hunks in `cpp/denner_1d/src/acid.cpp`, one executable line)

- Hunk 1 (~lines 550-563): delete the `ACID_TSAT_STALL` declaration and its 12-line rationale
  comment.
- Hunk 2 (~lines 767-769): update the `stall_reason` enum comment (reason 5) to no longer reference
  the deleted flag by name.
- Hunk 3 (~lines 2314-2333): the mechanism itself. **The only executable change in the round**:
  `if (tsat_stall > 0 && coupled) {` → `if (coupled) {`. Comment rewritten to record the promotion
  rationale and the last commit where the flag existed.
- Hunk 4 (~lines 2389-2392): update the `ACID_STALL_ACCEPT` eligibility comment (which references
  the flag by name) similarly.

No other file changes. `cases.cpp`/`validation.cpp` untouched (round's goal doesn't require it).

## 3. Verification design

The whole safety argument is one byte-for-byte diff: run the **pre-edit** binary with
`ACID_TSAT_STALL=1` forced (ground truth for "the flag was always on" — already proven a no-op or a
strict improvement by round 18), then diff the **post-edit** binary's default output (no env var —
it no longer exists) against it. Byte-identical proves the promotion changes nothing beyond what
round 18 already measured.

Extend round 18's 5-configuration sweep to 7 by adding config **G** (`ACID_YADV=1 ACID_NO_AJAC=1`,
FD without `+ALPHA_IMPLICIT`) — round 18's own §28.3 table already contains a row for this exact
combination but it was never added to `scripts/yadv_r9_sweep.py`'s tracked `CONFIGS`/`EXPECTED`.
Also re-run `ACID_STALL_ACCEPT` levels 1 and 2 against configs B and C (G7) and the `ACID_DBG`
positive control (G8).

## 4. Gates

- G1-G5: build clean, `denner1d_unit` PASS, `grep -rn "TSAT_STALL" cpp/` → zero lines, 7-config
  battery byte-identical to flag-forced-ON, `scripts/yadv_r9_sweep.py --verify --sweep` → ALL GATES
  OK (after `EXPECTED` is updated) + VERIFY OK (OFF path byte-identical to `solver_denner`
  published binary).
- G6 (deletion proof): the env var is genuinely inert because it no longer exists, not because it's
  merely unread — confirmed by the zero-grep-hits check plus a successful build.
- G7 (`ACID_STALL_ACCEPT` cross-check): case24 unchanged, case34 reproduces round 19's exact
  perturbation byte-for-byte, case33 faster/cleaner (fewer accepted retries, earlier stall).
- G8 (mandatory positive control): case33 under config C, `ACID_DBG=1`, no other env var →
  `STALLED-DETAIL: reason=T-ceiling-saturated`, step 43.
- G9-G13: permanent-invariant sweep, RH re-measurement (not expected to move — F2'' is a stopping-
  criterion change, not a flux change), broken-reproduce audit (the case33 `step=100` grep in §25.2
  will go stale — annotate, don't edit), perf sanity, diff hygiene (`git diff --stat -- cpp/` should
  show exactly one file, one executable line).

## 5. Expected baseline changes

D (`+ALPHA_IMPLICIT+FD`) and E (`FD` alone) are expected to move in the improving direction only,
per round 18's own §28.3 measurement of the same underlying mechanism. Config G (new) is expected to
reproduce round 18's §28.3 "FD (`ACID_YADV=1 ACID_NO_AJAC=1`)" row exactly. A/B/C/F are expected
byte-identical (round 18 already swept these clean).

## 6. Non-goals

No change to `ACID_YADV`'s own default (still OFF, 15/19). No attempt at round 13's harder
`(T,rho,h)` reconciliation, `max_steps` exhaustion, or case29 — all carried forward untouched.

## 7. Actual results

See `docs/YADV_RESEARCH.md` §30 for the full measured outcome, including one methodology detour
(the `DENNER_ACID=1` requirement, §30.2) and the config-G/round-18-table reconciliation (§30.4).
Summary: promotion confirmed safe and a net improvement (D 12→13/19, E 13→14/19, G 13→15/19, cases
27/28 recovered from silent NaN; case33 fails faster/cleaner). All hard gates held.
