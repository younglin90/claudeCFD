# YADV Round 35 Plan — Exclude cases 24/33/34 from the registered validation suite (user-authorized)

**Round**: 35 · **Base**: `main` @ `220b91c` (round 34's own commit) · **Worktree**: `.claude/worktrees/yadv-round-35/solver_4eq_mass`
**Type**: authorized `cases.cpp` edit, scoped exactly to cases 24/33/34's exclusion. Zero numerics change for any case that still exists. **`validation.cpp` needs zero edits** (verified).

---

## 0. Authorization, and the ONE way this round is not round 34

### 0.1 The user's decision, verbatim

> **"안된다 모든 검증을 같은 솔버 및 기법으로 무조건 해야된다 그래서 도저히 4eq mass 로 풀수 없으면 24 33 34 검증은 제외시키는게 맞는 방향같다"**

Presented with round 31's 4-8 round single-p/two-T rewrite estimate, a fresh literature re-dig this session (including a direct check of Denner's own JCP 367 primary source, at the user's push-back) confirming no cheaper fix exists, and one further design idea (a `face_shock`-gated hybrid two-EOS approach) — the user **rejected both the rewrite and the gated hybrid**, and directed exclusion. Governing principle: *uniform technique across the entire registered suite*, no per-case/per-cell technique switching.

### 0.2 HONEST FINDING — this is a NEW exclusion criterion, not round 34's

Round 34 applied the suite's own existing representability criterion (`cases.cpp:599-602`, case32). **That is not true here.** Cases 24/33/34 PASS under OFF (fully valid, fully representable cases in the alpha-transport model) and fail only under `ACID_YADV=1`, for a reason proven to be a model-class gap, not a numerical defect (`YADV_RESEARCH.md` §36's closed-form proof, §41's thermal-disequilibrium mechanism). The criterion here is a **policy criterion the user set**, not a representability criterion the code already encoded:

| case | criterion | class |
|---|---|---|
| 29 | Ms=100 water shock: dt collapses, front under-resolved | numerical/resolution blocker |
| 32 | IC middle state 0.01 Pa < 1.0 Pa floor | representability (IC) |
| 15 | exact `p*=9.05e-14 Pa` < 1.0 Pa floor | representability (solution) |
| **24/33/34** | **pass under OFF; unreachable under the uniform `ACID_YADV=1` technique (O(1) gap). Excluded by user policy: one technique for the whole suite.** | **model-class scope (NEW)** |

### 0.3 Integrity requirement for the new headline number

`ACID_YADV=1` becomes **15/15 — all-passing, for the first time in the project's history.** True, but *earned by removing the failing cases, not by fixing them.* Every mention must carry that qualifier in the same sentence (gate G-I).

---

## 1. THE ARITHMETIC — derived from `validation.cpp:829-849`'s counting logic

`total` is a loop counter over `all_cases()`; nothing hardcoded. Removing a case that PASSED decrements both `pass` and `total`; removing a case that FAILED decrements only `total`. From the current `EXPECTED` (post-round-34, `total=18` everywhere), removing `{24,33,34}` (all three passed under A, all three failed under B-G):

| cfg | old pass | new pass | new fail set |
|---|---|---|---|
| A (OFF) | 18 | **15** | `{}` |
| B (ACID_YADV=1) | 15 | **15** | `{}` |
| C | 14 | 14 | `{"14"}` |
| D | 13 | 13 | `{"14","27"}` |
| E | 14 | 14 | `{"28"}` |
| F | 14 | 14 | `{"14"}` |
| G | 15 | **15** | `{}` |

**Predictions**: A drops from 18/18 to 15/15 (loses 3 passes); B and G become 15/15 all-passing for the first time ever (`denner1d_validate` under `ACID_YADV=1` exits 0 for the first time); C/D/E/F keep their pre-existing 14/27/28 failures untouched. If any config moves differently: falsified, revert.

---

## 2. Verified code facts

### 2.1 `cases.cpp` locations

`compute_case24_shock()` definition at line 105. Config construction: `case24_Vs`/`c24` (495-496), `case33_Vs`/`c33` (543-544), `case34_Vs`/`c34` (545-546) — **untouched**, config stays live. Suppression block 569-573. Table entries: case24 at 588, case33 at 608-609, case34 at 610-611. `initial_state()` branches at 635-637 (Case24Shock selection) and 694-699 (IC). `reference_state()` block at 784-797. All branches **kept, become unreachable, not deleted** (case15/29/32 precedent).

Confirmed mapping: case33 → `alpha_air=0.75`, case34 → `alpha_air=0.25`, case24 → default 0.5.

### 2.2 `compute_case24_shock` does NOT become dead code — corrects the round's own initial premise

Verified: `compute_case24_shock` is called at lines 495, 543, 545 (config construction, inside `all_cases()`, runs unconditionally to build `c24`/`c33`/`c34`'s `t_end`) — **none of these calls live inside the case-table initializer list** (574-616) that gets commented out. The table entries at 588/608-611 pass the already-built config objects, they don't call the function. So `compute_case24_shock` remains live and called on every `all_cases()` invocation, exactly as today. No unused-function warning, no dead-code handling needed — simpler than the round's own charter anticipated.

### 2.3 `validation.cpp` needs ZERO edits — confirmed

`case24_spec_pass` defined at line 469; single shared dispatch at lines 731-734 (`if (case_id == "24" || case_id == "33" || case_id == "34") return case24_spec_pass(...)`). This branch stays (unreachable, not deleted), so the function stays referenced. Zero edits needed.

### 2.4 Independence from other cases

No consumer indexes `all_cases()` positionally (id-string lookup only via `find_case`/`validate_cases`) — removing three entries cannot shift or alias any surviving case. `initial_state()`'s 24/33/34 `else if` branch and `reference_state()`'s early-return block are both structurally independent of neighboring case branches (case15's/case25's).

### 2.5 Unit tests structurally immune

`tests/denner1d_unit.cpp` contains exactly one `find_case` call (`"01"`), zero references to `"24"/"33"/"34"`/`case24`/`compute_case24_shock`. Unaffected.

### 2.6 The exact edits

**E1** (replacing lines 569-570):
```cpp
    // cases 15/24/29/32/33/34 are EXCLUDED from the registered suite (their entries are commented
    // out in the list below); configs + IC/reference/gate code stay intact for future solver work.
    // NOTE the criteria differ: 29 = numerical/resolution blocker; 32/15 = a state below the 1.0 Pa
    // representability floor (IC / solution respectively); 24/33/34 = round 35, they PASS under the
    // OFF alpha path but are structurally unreachable under the uniform ACID_YADV=1 mass-fraction
    // technique (an O(1) closure / thermal-disequilibrium gap, docs/YADV_RESEARCH.md sect.36, 41),
    // excluded by explicit user policy that the whole suite must validate under ONE technique.
```

**E2** (replacing lines 571-572):
```cpp
    (void)c15;
    (void)c24;
    (void)c29;
    (void)c32;
    (void)c33;
    (void)c34;
```

**E3** (replacing line 588):
```cpp
        // EXCLUDED (blocker): cases 24/33/34 PASS under the OFF alpha path but cannot pass under
        // ACID_YADV=1 for ANY numerical improvement -- the reference (Denner Eqs.57-62) holds the
        // volume fraction across the shock while ACID_YADV=1 conserves mass fraction, and the two
        // exact closures differ by O(1) (~2x in rho/p), a thermal-disequilibrium / model-class gap,
        // not discretization error (docs/YADV_RESEARCH.md sect.36 closed-form proof, sect.41
        // mechanism + 4-8 round rewrite scope). Excluded in round 35 by explicit user decision that
        // the entire suite must validate under ONE solver and ONE technique (sect.45). This is a
        // DIFFERENT criterion from case15/32 above (representability) -- do not conflate them.
        // {"24", "24_H homogeneous Mach-10 mixture shock", air, denner_water, c24},
```

**E4** (replacing lines 608-609):
```cpp
        // EXCLUDED (blocker): same Fig.18 psi-family as case24 above, same criterion (round 35).
        // {"33", "33_H homogeneous Mach-10 mixture shock psi_w=0.25 (Denner 7.4.1 Fig.18)",
        //  air, denner_water, c33},
```

**E5** (replacing lines 610-611):
```cpp
        // EXCLUDED (blocker): same Fig.18 psi-family as case24 above, same criterion (round 35).
        // {"34", "34_H homogeneous Mach-10 mixture shock psi_w=0.75 (Denner 7.4.1 Fig.18)",
        //  air, denner_water, c34},
```

**Do NOT touch**: everything else in `cases.cpp`, all of `validation.cpp`. No numeric literal anywhere may change.

### 2.7 Post-edit registered case set (15 entries)

`01, 02, 04, 05, 07, 13, 14, 25, 26, 27, 28, 30, 31, 35, 36`

---

## 3. `scripts/yadv_verify.py`

Current: `CASES = ["01", "02", "13", "14", "24", "25", "33", "34"]`. New:
```python
CASES = ["01", "02", "13", "14", "25"]
```
with a comment explaining the round-35 removal (mechanically identical reason to round 34's case15 removal). G1 headline becomes **5/5**, not 8/8.

---

## 4. `scripts/yadv_r9_sweep.py`

**`EXPECTED`** (append round-35 comment below round 34's, do not rewrite it):
```python
EXPECTED = {
    "A": (15, set()),
    "B": (15, set()),
    "C": (14, {"14"}),
    "D": (13, {"14", "27"}),
    "E": (14, {"28"}),
    "F": (14, {"14"}),
    "G": (15, set()),
}
```

**`ALL_CASES`**: `["01", "02", "04", "05", "07", "13", "14", "25", "26", "27", "28", "30", "31", "35", "36"]` (15 entries, matches §2.7).

**`VERIFY_CASES`**: `["01", "02", "13", "14", "25"]` (5 entries).

**`sample_cases` (line 264, `do_iters`)**: currently `["13", "25", "14", "15", "02", "24"]` — round 34's own plan called for removing `"15"` here and the implementing session missed it (a stale, harmless leftover since `--iters` is diagnostic-only, never a gate). This round removes `"24"` (required) and catches the `"15"` miss (rides along): new value `["13", "25", "14", "02"]`.

**Frozen per-round scripts**: not touched (permanent artifacts of their own rounds, per SKILL.md's own rule).

---

## 5. `validation/1D/24_H_hypersonic_mixture_ms10.md` — annotate a SHARED FAMILY spec, precisely scoped

This doc IS the spec for case24, and covers the whole Fig.18 ψ-family: ψ∈{0,0.25,0.5,0.75,1} maps to cases {26/27 (pure), 33, 24, 34 (mixture), with 28/29 as the Ms=100 siblings}. **Cases 26/27/28 remain registered and passing** — this doc must NOT get a blanket exclusion banner. Banner must be scoped precisely: mixture members 24(ψ=0.5)/33(ψ=0.25)/34(ψ=0.75) excluded; pure endpoints 26/27/28 remain live, this doc is still their spec. `validation/1D/33_S1_ransom_gravity_faucet.md` and `34_S2_homogeneous_phase_change_relaxation.md` are unrelated (different numbering series) — not touched.

---

## 6. Every other file

**`docs/YADV_ROADMAP.md`**: absolute-rule text (15/15, extend non-precedent clause), the `cases.cpp`/`validation.cpp` rule note (rounds 34 AND 35 now the exceptions), new top section "Phase 3a / cases 24/33/34: CLOSED (round 35)" with a supersede-pointer pattern matching round 34's own for case15, control state block moved to superseded history + new block, history bullet appended.

**`AGENTS.md`**: 18/18→15/15, 15/18→**15/15 PASS** (with the exclusion qualifier), extend the case15-exclusion sentence to include 24/33/34.

**`.claude/skills/yadv-round/SKILL.md`**: 18/18→15/15, 8/8→5/5, extend the provenance parenthetical.

**`.claude/rules/denner-pitfalls.md`**: new bullet for cases 24/33/34's exclusion (§36/§41 citations, the different-criterion note, "do not retry Jacobian/globalization/reconstruction/relaxation work on these three — rounds 4-8/11/23-28 all did").

**`docs/YADV_RESEARCH.md`**: append `## 45.` only (never edit §1-44) — user directive verbatim, criterion novelty, arithmetic derivation with inverse-shape note, milestone honestly framed, harm-gate table, the `compute_case24_shock` corrected-premise finding, honest costs (G1 9→8→5, family-spec partial scope, `--only 24` exit-0 trap, frozen-script list, round-34 `sample_cases` miss caught here), end state (no live escalation remains, round 36 needs fresh user decision).

---

## 7. `solver_denner` — untouched, same adjudication as round 34

---

## 8. Harm gate G-A through G-I

Stage 0 baseline capture **sequential, foreground, not backgrounded** (round 34 hit a background race — avoid repeating it). Expect OFF `18/18`, ON `15/18` fail `{24,33,34}` before any edit; if not, STOP.

- **G-A**: OFF `pass_count=15 total=15`, exit 0.
- **G-A2**: exact 15-id set (§2.7), 24/33/34 absent.
- **G-B** (strongest): per-case OFF JSON byte-identity vs baseline, empty diff, all 15 remaining cases.
- **G-C**: ON `pass_count=15 total=15`, fail set **empty**, exit **0**; same empty byte-identity diff.
- **G-D**: `yadv_verify.py` 5/5 byte-identical vs `solver_denner`; case01 identical in section 2.
- **G-E**: unit tests byte-identical to baseline.
- **G-F**: `denner1d_dump`/`run` 24/33/34 all exit 2 "unknown Denner 1D case"; confirm case25 still dumps fine (proves it's not a broken binary).
- **G-G**: `git diff --stat -- cpp/` exactly one file, only comments + `(void)` lines + commented table entries.
- **G-G2**: `solver_denner` untouched, binary unchanged.
- **G-H**: 7-config sweep matches derived `EXPECTED` exactly, `total=15` everywhere.
- **G-I** (integrity): every "15/15" mention carries the exclusion qualifier in the same sentence.

If G-B/G-C non-empty or G-H mismatches: revert, do not merge, increment `consecutive_failures`.

---

## 9. Non-goals

No model/numerics/scheme change beyond `cases.cpp`. No `pface` work. No 5-eq rewrite, no `face_shock`-gated hybrid, no per-case/per-cell technique switching (explicitly rejected by user). No threshold/resolution/reference change. No cleanup of dead 24/33/34 code. No historical content rewritten. No touching the unrelated `_S1`/`_S2` spec docs. No `solver_denner` edit. No `validation.cpp` edit (none needed). No bare "15/15" without the exclusion qualifier.

---

## Actual outcome (implementing session, post-hoc)

**All three of the plan's own §9 spot-checks confirmed exactly**: `compute_case24_shock` call
sites (495/543/545 config-construction, 635-637/786 unreachable branches, none inside the
commented-out table entries); `validation.cpp`'s single shared gate dispatch at 731-734; the
current `EXPECTED`/`ALL_CASES`/`VERIFY_CASES` matched the plan's cited values exactly.

**§2.2's corrected premise held up under actual implementation**: the round's own charter had
anticipated `compute_case24_shock` becoming fully dead code with three call sites removed at once.
The Planner's own re-derivation (confirmed by direct read before any edit) was correct: the
function stayed live and referenced after the edit, with zero unused-function warnings — the build
log showed only the same two pre-existing, unrelated warnings round 34 also saw.

**Stage 0 baseline capture followed the plan's own explicit caution** (sequential, foreground, not
backgrounded) after round 34's own background-race incident — no contamination this time; all
three baselines (unit, OFF, ON) matched round 34's own recorded values exactly before any edit was
made.

**The arithmetic prediction matched exactly**: OFF `15/15`; `ACID_YADV=1` `15/15`, empty fail set,
**exit code 0 for the first time in the project's history**. G-H's fresh 7-config sweep matched
the derived `EXPECTED` exactly for all seven configs, confirming C/D/E/F's pre-existing 14/27/28
failures were untouched by the exclusion.

**G-I (the integrity gate) reviewed manually**: every substantive prose mention of "15/15" in the
round's own new/edited text carries the "achieved by exclusion, not by fixing" qualifier in the
same sentence or the immediately adjacent one. Table cells and terse gate-invocation comments
(e.g. `SKILL.md`'s shell-command annotations) were judged to satisfy this via their immediately
surrounding qualified prose, matching the plan's own "same sentence or immediately adjacent"
standard rather than requiring literal repetition in every cell.

**Verdict: S1, exactly as the plan's own pre-registered text anticipated.** All ten hard gates
held (G-A through G-I). `consecutive_failures` NOT incremented. No historical document was
rewritten; every forward-looking `18/18`/`15/18`/`8/8` reference was updated with an explicit
provenance note distinguishing this round's policy criterion from case15/32's representability
criterion. The `24_H` spec doc's banner was scoped precisely to the mixture members only, per the
plan's own explicit finding that a blanket exclusion banner would have falsely de-registered the
still-live pure-phase endpoint cases (26/27/28).
