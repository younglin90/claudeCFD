# YADV Round 14 Plan — Phase 3a Stage 3c: mark the retry-exhaustion give-up as `diverged`

Planner output (Agent, subagent_type="Plan", model=opus), round 14. Scope authorised by the Advisor
this session: implement the `diverged = true` correctness fix rounds 11/12/13 each deferred
(YADV_RESEARCH.md sect.21.1, 22.7 pt.1, 23.4 pt.5; docs/YADV_ROUND_12_PLAN.md sect.5). Narrow,
mechanical, one added statement. No numerics change, no literature search (agreed: pure bookkeeping/
correctness fix, no discretisation/closure decision involved).

Advisor spot-check: confirmed exactly `bool diverged = false;`=669, accept block condition=2303,
give-up `if (!stepped) {`=2325, stall break=2352, `if (diverged) {`=2399.

## 0. Key structural findings

**0.1 Accept-vs-give-up boundary**: one `!stepped` give-up check (2325), AFTER the accept attempt
(2303). A step accepted by `ACID_STALL_ACCEPT` sets `stepped=true` at 2323 and never reaches the
give-up block -- so accept-and-continue runs are correctly NOT marked diverged (their own
`STALL-ACCEPT-TOTAL` is the honest disclosure). Only "neither clean nor accepted was possible" lands
at 2325: `ACID_STALL_ACCEPT` unset, OR its budget exhausted, OR no retry was reason-1-eligible. No
extra condition is needed at the give-up site -- the control flow already encodes the distinction.
Do NOT add `if (n_stall_accept == 0)` or similar -- a run that accepted some steps and then still
hit the wall later (e.g. case33 `+IMPLICIT ACID_STALL_ACCEPT=1`, round 12: 104 steps, 4 accepts,
budget exhausted, still stalls) DID give up before `t_end` and must be marked diverged.

**0.2 Consumption reached, unchanged**: `break` at 2352 exits the time loop; only `fprintf`s follow
before the `if (diverged) { NaN fill }` at 2399 -- no early return. Verified end-to-end.

**0.3 CORRECTION**: this block is NOT `ACID_YADV`-gated (it's in the common time loop shared by OFF).
Safety for OFF rests on evidence (round 11 sect.21.1: STALLED fires on exactly the 3 known configs,
nowhere else, including OFF, measured against an isolated main-HEAD build), not structure -- so gate
G1 below re-measures this empirically on the post-edit build, not by inspection.

**0.4 Stale-binary hazard**: the main checkout's pre-built `denner1d_dump` predates round 11 (no
STALLED/STALL-ACCEPT/ACID-done strings at all) -- any baseline must come from a fresh worktree build,
never from that binary. `scripts/yadv_verify.py` also hardcodes the main-tree path and cannot verify
a worktree build in place -- use explicit commands, not that script, for this round's gates.

## 1. The change (`cpp/denner_1d/src/acid.cpp`, block at 2325)

Replace the existing "Round 11 Stage 1 ... NOT marked diverged" comment block with an updated one
explaining the Stage 3c decision and the accept/give-up boundary (full text in the Planner's
response, preserved in commit history), add:

```cpp
            diverged = true;   // -> the p/u NaN fill at the end of solve_case_acid
```

immediately after the comment, and change the STALLED message's tail from
`"-> stop (state returned as-is, NOT marked diverged)\n"` to
`"-> stop (marked DIVERGED: p,u,rho returned as NaN, validate finite=false)\n"` -- the prefix through
`... of %.3e` stays byte-identical (constraint: `scripts/yadv_r11_window.py`'s `_STALLED_RE` matches
only that prefix via `re.search`, unaffected by a tail change). Do NOT rename the `STALLED:` keyword.
Also add one clarifying line to the `if (diverged) {` consumption comment noting there are now two
producers (CFL-collapse, and Stage 3c retry-exhaustion).

Nothing else changes: no script edits, no `cases.cpp`/`validation.cpp`, no new env var, no new field.

## 2. Predicted effect (pre-registered)

| config | cases whose dump changes | `pass_count` |
|---|---|---|
| OFF | none | 19/19 -> 19/19 |
| plain ON | 24, 34 -> all-NaN | 15/19 -> 15/19, same failure set `{15,24,33,34}` |
| `+ALPHA_IMPLICIT` | 33 -> all-NaN | 14/19 -> 14/19, same set + 33 already in it |
| FD-invariance (both D=`ACID_YADV=1 ACID_NO_AJAC=1` and E=`ACID_NO_AJAC=1` alone) | measure | predicted unchanged |
| `ACID_STALL_ACCEPT=1`, case24/34 plain | none -- byte-identical | n/a (never reaches give-up) |
| `ACID_STALL_ACCEPT=1/2`, case33 `+IMPLICIT` | 33 -> all-NaN | n/a (budget exhausts, correctly diverged) |

Exactly three (case,config) pairs are expected to change under the four hard gates: 24/plain,
34/plain, 33/+IMPLICIT. Any other row moving is a stop-and-investigate signal.

## 3. Gates

G0: clean build + unit test pass. G1: OFF 19/19 byte-identical to published binary AND (empirically,
not by inspection) zero STALLED/DIVERGED lines across all 19 OFF dumps on the post-edit build. G2:
plain-ON pass_count=15/19, full per-case JSON diffed pre/post -- only case24/34 lines may differ,
and within them `finite: true->false`, `pass` stays false. G3: `+ALPHA_IMPLICIT` pass_count=14/19,
only case33's line differs; explicitly confirm case24/34's lines are byte-identical (they complete
under this flag and must not be touched). G4: FD-invariance D and E, resolve the ambiguous prior
figure (round 12/13 memory has D=12/19 vs round 11's "the FD gate" =13/19 without disambiguating
D-vs-E) by direct measurement this round, record both. G5: `ACID_STALL_ACCEPT=1` runs for case24/34
byte-identical pre/post (never reach give-up); case33 `+IMPLICIT +STALL_ACCEPT` changes to NaN as
intended, still prints both STALL-ACCEPT-TOTAL and STALLED. G6: `scripts/yadv_r11_window.py` and
`scripts/yadv_rh2.py` still run and classify the three known-stalled configs correctly (NULL RUN /
STALLED unaffected; IC-match heuristic cosmetically changes to 0.00/nan but classification survives
via the frac<0.9 clause). G7 (diagnostic, no code): sweep all 19 cases x {OFF, plain-ON, +IMPLICIT}
for `ACID done ... t=... of ...` lines with `t` materially short of `t_end` but NO `STALLED:` --
i.e. a max_steps-exhaustion silent-partial-exit, a SIBLING defect explicitly out of scope this round
(case15 legitimately terminates via max_steps and PASSES on OFF -- extending diverged there would
break the 19/19 gate; data collected for a future round's decision only).

## 4. Historical-artifact audit (corrective annotations added to new sect.24, history untouched)

Needs annotation: YADV_RESEARCH.md sect.14.3 table 1 (rows 24/34, v3 column, plain ACID_YADV=1 --
validate metrics of the silently-stalled run); sect.19.2's three cell-pairs for case24/34 (col B)
and case33 (col C); sect.20.3's NULL RUN rows' `IC-match=0.89` (becomes 0.00, classification
unaffected); sect.19.3's case24 timing description (becomes literally true, no number moves).
Already superseded, no action: sect.14.3 table 2 (retracted by sect.20.1), sect.19.4 (retracted by
sect.20.1/20.4), sect.5/6/10.2 (superseded code versions). Confirmed unaffected: sect.21.3/22.5 RH
tables (all from genuinely-completing runs), sect.21.1/21.2, sect.21.4 (stderr-parsed, prefix
unchanged), sect.22.4, sect.23.1/23.2. Latent (not live) breakage found, documented not fixed:
`scripts/yadv_table.py`/`yadv_table3.py` lack the lowercase-nan JSON fix `yadv_r9_sweep.py` carries
-- already non-runnable (hardcoded stale paths), would additionally fail on case24/34 plain-ON rows
now; any future script parsing validate JSON must copy the NaN fix.

## 5. Non-goals

No round-13 (T,rho)-consistency re-init. No case33 Newton-difficulty work, no `ACID_YADV_HREINIT`/
`ACID_RINIT` changes. No `ACID_STALL_ACCEPT` accept-logic changes. No threshold exempting
"nearly-complete" stalls from diverged (would be a forbidden tuning coefficient and re-opens the
hole). No extending `diverged` to `max_steps` exhaustion (case15 legitimately uses it and PASSES on
OFF -- would break 19/19; G7 collects data for a future round only). No script edits. No literature
search.

## 6. Risks

R1 (highest, decides the round): a `t_end`-clamped final step that stalls on a currently-PASSING
case -- `dt=min(dt,t_end-t)` means the last step's size is CFL-trajectory-independent, and round 11
sect.21.4 measured exactly this pathology under `ACID_TEND_SCALE` sweeps (clean/stall/clean
non-monotonicity). Mitigated: G1-G4 measure directly across all 19 cases x configs, not inferred.
If it fires on any currently-passing case: stop, do not merge, report honestly, escalate -- do not
threshold around it. R2: OFF-path safety is evidence-based not structural (sect.0.3) -- G1 mandatory,
empirical. R3: PNG writer on all-NaN field -- traced, no hang possible (every sample maps to the
same int, Bresenham dy=0 always). R4: "exactly three configs" is a round-11 measurement, re-verify
fresh this round (G1-G4) rather than trust carried forward. R5: FD-invariance figure ambiguity (D vs
E) -- resolve by direct measurement, record both, do not inherit an ambiguous baseline (how the
stale 12/19 figure persisted for several rounds previously). R6: a reviewer misreading changed
metrics as regression -- sect.24 must lead with "pass_count unchanged in all four hard gates; three
previously-finite garbage rows now read NaN, which is the point."

## 7. `n_stall_accept` in validate JSON: OUT OF SCOPE

Would require editing `validation.cpp` (forbidden unless the round's stated goal requires it; this
round's does not). The need is already met at the level that matters: `STALL-ACCEPT-TOTAL` prints
unconditionally to stderr, machine-greppable, already consumed by sect.22.4/23.2's tables. Not
implemented this round; a future round wanting it in JSON should carry its own explicit Advisor
exemption for `validation.cpp` as its stated goal.
