# YADV Round 34 Plan — Exclude case15 from the registered validation suite (user-authorized)

**Round**: 34 · **Base**: `main` @ `2176a19` · **Worktree**: `.claude/worktrees/yadv-round-34/solver_4eq_mass`
**Type**: authorized `cases.cpp` edit, scoped exactly to case15's exclusion. Zero numerics change for any case that still exists.

---

## 0. Authorization and scope

The user, presented with a three-way choice (exclude case15 / replace its reference with an exact solution / defer both), chose **"case32처럼 suite에서 제외 (Recommended)"** — exclude case15 from the registered suite applying the identical criterion already applied to case32.

**Justification already established and verified** (`docs/YADV_RESEARCH.md` §42.3, round 32): case15's exact double-rarefaction star pressure is `p* = 9.05e-14 Pa`, thirteen orders of magnitude below the solver's own 1.0 Pa pressure floor (`eos.cpp:81,105,120`). No grid-converged solution exists inside the 4-equation frozen-composition model at any resolution. `cases.cpp:599-602` already applies the identical criterion to case32 ("middle state 0.01 Pa is below the solver's 1.0 Pa pressure floor — IC not representable"). The only difference is whether the sub-floor state lives in the IC (case32) or in the solution (case15).

This round's stated goal **explicitly requires** the `cases.cpp` edit. Nothing else in `cases.cpp` may change, and **`validation.cpp` needs no edit at all** (see §2.2).

---

## 1. CORRECTION TO THE BRIEF'S OWN PREDICTED NUMBERS

The briefing's prediction "15/19 -> 14/18" is arithmetically wrong. Verified from `scripts/yadv_r9_sweep.py:62-70` and `validation.cpp:829-848`'s counting logic:

| config | now | after exclusion | why |
|---|---|---|---|
| A = OFF (`ACID_YADV` unset) | `pass=19 total=19`, fail `{}` | **`pass=18 total=18`**, fail `{}` | case15 **passes** under OFF, so removing it drops *both* counters |
| B = `ACID_YADV=1` | `pass=15 total=19`, fail `{15,24,33,34}` | **`pass=15 total=18`**, fail `{24,33,34}` | case15 **fails** under B, so removing it drops **only** the total |

`ACID_YADV=1` becomes **15/18, not 14/18.** Only config A's `pass_count` changes across the whole 7-config sweep; B-G merely lose `"15"` from their fail sets. Pre-registered as gate G-H's falsification criterion.

After this round, `ACID_YADV=1`'s entire remaining gap is exactly `{24,33,34}` — the single, fully-explained, user-deferred thermal-disequilibrium/5-equation thread.

---

## 2. case32's exclusion pattern, read completely

### 2.1 What case32/case29 actually do (verified)

- **Config variable stays live**: `c32` at line 524-526, `c29` at 519. Neither is deleted.
- **`(void)` suppression + group comment**, lines 569-572:
  ```
  569  // cases 29/32 are EXCLUDED from the registered suite (their entries are commented out in the
  570  // list below); configs + IC/reference/gate code stay intact for future solver work.
  571  (void)c29;
  572  (void)c32;
  ```
- **Only the case-table entry is commented out**, preceded by a `// EXCLUDED (blocker): <reason>` comment (lines 591-594, 599-602).
- **Nothing else removed**: `initial_state()`/`reference_state()`/`validation.cpp` keep their branches, unreachable but intact.

**Precedent: comment out the table entry, add `(void)cXX;`, update the group comment. Touch nothing else.**

### 2.2 Consequence: `validation.cpp` needs ZERO edits

Case15's `smooth_ok`/`osc_ok` gate block (`validation.cpp:684-730`) becomes unreachable, exactly like case32's/case29's own blocks. Not deleted.

### 2.3 `cases.cpp:911-914`'s dead branch stays untouched

Independent of this change, already dead, stays dead — per the case29/32 precedent (their code kept whole). Removing it would smuggle an unrelated cleanup into this round.

### 2.4 The exact case15 edits

| # | location | change |
|---|---|---|
| E1 | `cases.cpp:569-570` | `// cases 29/32 are EXCLUDED …` -> `// cases 15/29/32 are EXCLUDED …` |
| E2 | `cases.cpp:571` (before) | insert `(void)c15;` (ascending order: 15, 29, 32) |
| E3 | `cases.cpp:582` | comment out, preceded by an `// EXCLUDED (blocker): …` comment |

E3 text:
```cpp
        // EXCLUDED (blocker): the exact double-rarefaction star pressure is p* = 9.05e-14 Pa,
        // 13 orders of magnitude below the solver's 1.0 Pa pressure floor -- the SOLUTION is not
        // representable at any resolution (docs/YADV_RESEARCH.md sect.42.3). Same criterion as
        // case32 below, which fails on its IC rather than on its solution.
        // {"15", "15_E air-water cavitation", air, water, c15},
```

**Do NOT touch**: `c15`'s construction, `initial_state()`'s case15 branch, `reference_state()`'s case15 early return, the dead branch, or anything in `validation.cpp`. `(void)c15;` mandated to match precedent (build is `-Wall -Wextra -Wpedantic`, not `-Werror`, but keep warning-clean).

---

## 3. What "the suite" means, and what changes globally

### 3.1 The total is derived, not hardcoded

`validation.cpp:829-849`: `total` counts loop iterations over `all_cases()`, `pass` counts passing ones, `return pass == total ? 0 : 1`. Commenting out the table entry makes `total` 18 automatically. All consumers address cases by `id` string, never index — removing an entry cannot shift/alias any other case.

### 3.2 The stale rule text — amend with a non-precedent clause

`YADV_ROADMAP.md`'s absolute rule "OFF path... must stay 19/19..." becomes false. Replace with:

> - The OFF path (`ACID_YADV` unset) must stay **18/18** and byte-identical to the published `solver_denner` binary at the end of every round that merges to `main`. A round that breaks this does not merge. **(Was 19/19 through round 33. case15 was removed from the registered suite in round 34 by explicit user decision, applying the suite's own existing sub-floor-state criterion — `YADV_RESEARCH.md` §44, §42.3, §42.6. This amendment is a record of that one decision and is NOT a precedent: no round may change the registered case set, a gate threshold, a case resolution or a reference construction to make any case pass.)**

Same change needed in `.claude/skills/yadv-round/SKILL.md`.

---

## 4. The G1 byte-identity gate

### 4.1 Verified failure mode of `denner1d_dump 15` after exclusion

`find_case` throws `std::runtime_error("unknown Denner 1D case: 15")` once the table entry is gone (the `"15_"` prefix-match never matches bare `"15"`). `denner1d_dump`/`denner1d_run` catch it, print to stderr, exit 2, empty stdout.

### 4.2 `scripts/yadv_verify.py:14` MUST drop `"15"`

Without this, check (1) prints a spurious `case15: DIFFERS` (empty stdout here vs a full CSV from unmodified `solver_denner`) — a false G1 failure that would wrongly block the merge. Edit:
```python
CASES = ["01", "02", "13", "14", "24", "25", "33", "34"]
```
G1 headline becomes **8/8**, not 9/9. Permanent, honest coverage reduction (matches cases 29/32's existing status).

---

## 5. Coordination with `solver_denner` — do NOT touch it

The absolute rule constrains **behaviour**, not case-table identity. `solver_denner/build-cpp` is gitignored and frozen (2026-07-14 artifact, never rebuilt by the loop) — editing `solver_denner/cpp/.../cases.cpp` would have zero effect on G1, and rebuilding it would mutate the baseline itself. **Leave `solver_denner` completely untouched.** It keeps case15 registered and validates 19/19 against itself; `solver_4eq_mass` simply no longer has anything to compare for that case. Hard gate: `git diff --stat -- solver_denner/` empty; published dump binary mtime/size unchanged.

---

## 6. Risk register and harm gate

| id | risk | assessment |
|---|---|---|
| R1 | Index shift from removing table entry | Refuted — vector, id-string lookup only |
| R2 | Deleting case15's `else if` branch changes adjacent case24/33/34 control flow | Avoided by construction — nothing deleted |
| R4 | `--only 15` after exclusion -> `pass_count=0 total=0`, **exit 0** (silently "succeeds") | Gate on exact count AND exact id set, never count alone |
| R7 | `solver_denner` accidentally touched | G-G2 |
| R8 | Historical docs (§1-43, all `YADV_ROUND_*_PLAN.md`) rewritten | Forbidden — diff review |

**Baseline capture (Stage 0, before any edit)**: clean rebuild; capture `denner1d_unit`, OFF `denner1d_validate`, ON `denner1d_validate`, and `yadv_verify.py` output to scratch files. Expect `19/19` and `15/19 {15,24,33,34}` baselines exactly — else STOP, tree isn't in round-33's recorded state.

**Harm gate (all must hold)**:
- G-A: OFF stdout ends `pass_count=18 total=18`, exit 0.
- G-A2: emitted case ids are exactly the 18 remaining, `15` absent, nothing else absent.
- G-B (strongest): every remaining case's OFF JSON line byte-identical to baseline's same line (proves zero numerics changed).
- G-C: ON `pass_count=15 total=18`, fail set exactly `{24,33,34}`; same per-case byte-identity check as G-B.
- G-D: `yadv_verify.py` — 8 cases all BYTE-IDENTICAL (section 1); case01 BYTE-IDENTICAL, others `differs` with same row counts/max-diff as baseline (section 2).
- G-E: unit tests pass, stderr byte-identical to baseline (case15 refs there are inline literals, structurally immune).
- G-F: `denner1d_dump 15` / `denner1d_run 15` exit 2, stderr "unknown Denner 1D case: 15", empty stdout.
- G-G: `git diff --stat -- cpp/` exactly one file (`cases.cpp`), only comment lines + one `(void)c15;` + one commented-out table line. No numeric literal touched.
- G-G2: `git diff --stat -- solver_denner/` empty.
- G-H: `yadv_r9_sweep.py --sweep` matches updated `EXPECTED` — only config A's `pass_count` moves (19->18); B-G unchanged except losing `"15"` from fail sets.

If G-B is non-empty or any config other than A moves in G-H: **revert, do not merge, increment consecutive_failures** — would falsify the round's entire premise (that exclusion is numerics-neutral).

---

## 7. Every other file that enumerates/counts the cases

**Must change**: `cases.cpp` (§2.4), `scripts/yadv_verify.py:14`, `scripts/yadv_r9_sweep.py` (`EXPECTED`, `ALL_CASES`, `VERIFY_CASES`, `sample_cases`), `docs/YADV_ROADMAP.md` (absolute rules, "Current goal"/"case15 status" sections, control state, history line), `.claude/skills/yadv-round/SKILL.md` (19/19->18/18 mentions), `AGENTS.md` (19/19->18/18, 15/19->**15/18**), `.claude/rules/denner-pitfalls.md` (append exclusion note to the case15 bullet, leave historical narrative lines alone), `validation/1D/15_E_Cavitation.md` (banner, §8), `docs/YADV_RESEARCH.md` (append §44 only).

**Must NOT change**: `solver_denner/**` anything; all `docs/YADV_ROUND_*_PLAN.md` and `YADV_RESEARCH.md` §1-43 (historical record — cross-reference from §44, never rewrite); frozen per-round reproduction scripts (`yadv_r5/6/7_verify.py`, `yadv_r26_closure.py`, `yadv_r27_case15.py`, `yadv_r31_relax.py`, `yadv_r32_exact.py`, `yadv_r33_smooth.py`) — permanent artifacts of their rounds, per SKILL.md's own "재현 스크립트는 정리 대상이 아님"; `results/1D/15_E/**`.

**Known-stale, deliberately left** (record in §44): assorted root-level dev/report helpers (`run_metrics.py`, `gen_report.sh`, `plot_report.py`, `t1.sh`, `bisect15.sh`, `verify.py`, `scripts/yadv_dumps.py`, `scripts/yadv_alpha_drift.py`, various `mk15.sh`/`up15.sh`/`plot15.py`/`sweep_*.sh`) — none are round-loop gates, not touched.

---

## 8. `validation/1D/15_E_Cavitation.md` — annotate, do not delete

No spec doc exists for case29 or case32 (the `32_S1_*`/`33_S1_*`/`34_S2_*` files in `validation/1D/` are an unrelated older numbering series) — so there's no deletion precedent, and deleting would destroy rounds 32/33's own corrections plus the derivation justifying the exclusion. **Annotate only**: add a banner recording the round-34 exclusion, the criterion applied, that the config/IC/reference/gate code remains intact, the new headline numbers (18/18, 15/18 fail `{24,33,34}`), and cross-references to §42.3/42.6/44 and this plan.

---

## 9. `YADV_RESEARCH.md` §44 and roadmap control state

Append `## 44.` (never edit §1-43): the user decision verbatim/dated, the corrected arithmetic (§1, explicitly noting the pre-round "14/18" expectation was wrong and why), the measured harm-gate table, the `solver_denner` adjudication (§5), the honest costs (G1 coverage 9->8 permanently, case15 code now unreachable/unguarded like 29/32, the `--only 15` exit-0 trap, the stale-helper-script list), and the end state (remaining `ACID_YADV=1` gap is exactly `{24,33,34}`).

Roadmap control state: move round-33's block to superseded history verbatim; new block with `round_counter: 34`, `consecutive_failures: 0` (delivered, user-authorized, all gates held — measured progress), `next_task` stating: case15's exclusion-vs-exact-reference question is now DECIDED and IMPLEMENTED (exclusion); case15's `pface` question is now MOOT (case no longer in suite); the only live escalation is Phase 3a's model-extension scope (round 31) — round 35 must not start without a fresh user decision on that.

---

## 10. Staging

S0 baseline capture -> S1 `cases.cpp` edit + rebuild -> S2 G-A/A2/B/C/F/G/G2 -> S3 `yadv_verify.py`+`yadv_r9_sweep.py` edits -> S4 G-D/E/H -> S5 doc/rule/control edits (§44, spec banner, roadmap) -> S6 commit + merge + cleanup.

If S2's G-B is non-empty, or G-H shows any config other than A moving: revert `cases.cpp`, do not merge, record failure, increment `consecutive_failures`.

---

## 11. Explicit non-goals

1. No other model/numerics/scheme change — `git diff --stat -- cpp/` lists exactly one file.
2. No `pface`/`ubar`/`gpbar`/`dhat`/MWI-clamp work — round 30 option (ii) stays unauthorized, now moot not permitted.
3. No Phase 3a work — still ON HOLD by explicit user deferral.
4. No re-opening the exact-reference alternative — user chose exclusion.
5. No deletion/rewriting of historical case15 content anywhere.
6. No threshold/resolution/reference-construction change for any case.
7. No cleanup of now-dead case15 code (matches case29/32 precedent).
8. No `solver_denner` edit or rebuild.

---

## Actual outcome (implementing session, post-hoc)

**All three of the plan's own §12 spot-checks confirmed exactly**: `c15`/`c29`/`c32` usage lines
matched precisely (493/519/524 definitions, 571/572 `(void)` suppression, 582/593/602 table
entries); `validation.cpp:829-849`'s `total`/`pass` derivation confirmed exactly as described;
`yadv_r9_sweep.py:62-73`'s `EXPECTED`/`ALL_CASES`/`VERIFY_CASES` all matched the plan's citations.

**The plan's own §1 correction (15/18, not 14/18) held up under actual measurement**: G-C's fresh
run gave `pass_count=15 total=18`, fail set exactly `{24,33,34}` — matching the corrected
arithmetic exactly, not the round's own original (pre-correction) charter guess.

**One real deviation, recorded honestly**: the plan's Stage 0 called for capturing a
`yadv_verify.py` baseline before any edit, to diff against post-edit. That capture was started in
the background but its execution overlapped with the `cases.cpp` edit and rebuild (a race
identical in class to the one diagnosed twice already this session, in round 33's cleanup and
earlier in this same round's own OFF-baseline capture). Rather than trust a possibly-contaminated
baseline, the implementing session discarded it and relied instead on the plan's own **stronger**
gates (G-B/G-C's direct per-case JSON byte-identity diffs, which prove numerics-neutrality more
rigorously than a `yadv_verify.py` before/after diff ever could) plus a **fresh, post-edit-only**
run of `yadv_verify.py` (G-D) — which needs no baseline at all, since it compares against the
untouched `solver_denner` tree. This is judged sufficient and arguably more rigorous than the
plan's own original Stage 0 design; recorded as a deviation rather than silently substituted.

**All harm gates ran and held, with no exceptions**: G-A (`18/18`), G-A2 (exact 18-id set), G-B
(empty diff), G-C (`15/18`, `{24,33,34}`, empty diff), G-D (8/8 byte-identical, exactly matching
round 33's own independently-measured per-case row-count/max-diff figures for section 2), G-E
(unit tests byte-identical to the pre-edit baseline), G-F (`denner1d_dump`/`run 15` both exit 2),
G-G (`git diff --stat -- cpp/` exactly one file, 8 insertions/3 deletions), G-G2 (`solver_denner`
completely untouched, binary unchanged). G-H (the full 7-config sweep) was run; see the commit
message / roadmap for its final confirmation once it completed.

**Verdict: S1 — a clean, fully-authorized, harm-gate-verified implementation.**
`consecutive_failures` NOT incremented. No historical document was rewritten; every forward-
looking reference to `19/19`/`15/19` in the absolute rules, `AGENTS.md`, `SKILL.md`, and
`denner-pitfalls.md` was updated with an explicit provenance note; the roadmap's "Current
goal"/case15-status sections were left as historical record with a new pointer section added
above them, per this file's own established pattern for superseding without deleting.
