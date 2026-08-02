# ROUND 33 PLAN — literature/data housekeeping: four deferred loose ends from rounds 30–32 — **ZERO C++, ZERO `cases.cpp`/`validation.cpp`, ZERO BLOCKED-THREAD CONTACT**

**Target**: `solver_4eq_mass/`. **HEAD**: `0b968b1` (round 32's commit). **Worktree**: `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-33/`.
**Control state on entry**: `round_counter = 32`, `consecutive_failures = 0`, `done = false`.
**Charter**: the four items round 32 explicitly deferred (`docs/YADV_RESEARCH.md` §42.2, §42.3, §42.6, and `docs/YADV_ROUND_32_PLAN.md:125,248,345`). This round is *not* a continuation of any of round 32's substantive threads.

**Confidence notation**: `[VERIFIED]` = read/executed directly during this planning pass, path and content confirmed. `[EXTERNAL]` = obtained from an outside source (CrossRef/Unpaywall/web) during this planning pass, quoted with its source. `[ARITH]` = computed here in Python and reproducible in one line. `[JUDGEMENT]` = my recommendation, not a fact.

---

## 0. Executive summary — read this first, it changes two of the four items

**Answer to the framing question ("is this scope too small to need Planner ceremony?"): honestly, mostly yes.** Items 1, 2 and 4 were *fully resolvable inside this planning pass* and are resolved below — there is nothing left to design for them, only text to write down. Item 3 is a one-line `cp` whose only real content is a go/no-go judgement, which I give in §3. The round-loop protocol requires a plan, so here it is; but the implementing session should treat §1/§2/§4 as **findings to transcribe and re-verify**, not as instructions to go re-derive. Realistically this is a 30-minute documentation round with three cheap verification commands and one file copy.

Two of the four items did **not** come out where rounds 31/32 predicted:

> ### **FINDING R33-A (corrects rounds 31 C1 and 32 §42.3/§42.6): `papers/md/33_saurel_relaxation_multiphase.md` was never missing. It is present, git-tracked, and in this very worktree — one directory level above where rounds 31 and 32 looked.**
>
> `[VERIFIED]` `<worktree>/papers/md/33_saurel_relaxation_multiphase.md`, 97 409 bytes, 2 429 lines, `git ls-files` → tracked, `git check-ignore` → not ignored. Rounds 31/32 ran `ls papers/md/` from **inside `solver_4eq_mass/`**, where `papers` is a **symlink to `../solver_denner/papers`** (`[VERIFIED]` `readlink -f papers` → `<worktree>/solver_denner/papers`) — a *different, smaller* paper library containing four unrelated `.txt` files. Rounds 28/29/30 cited the path relative to the **repository root**, which is the `cfd-paper-search` skill's documented output root (`~/.claude/skills/cfd-paper-search/SKILL.md`: output tree is `papers/pdf`, `papers/md/{NN}_{author}_{year}_{keyword}.md`, `papers/{slug}_summary.md` — hence the `33_` sequence prefix). **Every line-number citation rounds 28/29/30 made against it is exact and verifiable** (see §1.2). Nothing needs to be fetched.

> ### **FINDING R33-B (resolves round 32's tentative attribution): the Yoo & Sung attribution is CONFIRMED at the bibliographic level, with an exact volume/page match.**
>
> `[EXTERNAL]` CrossRef, queried by DOI: **Young-Lin Yoo; Hong-Gye Sung, "Numerical investigation of an interaction between shock waves and bubble in a compressible multiphase flow using a diffuse interface method", *International Journal of Heat and Mass Transfer* **127** (December 2018) **210–221**, DOI `10.1016/j.ijheatmasstransfer.2018.08.012`.** Volume and page range match case14's own citation string (`14_E_shocktube_hp_water_lp_air.md:3`, "Yoo & Sung 2018 (IJHMT 127:210-221), §4.1 Validation") **exactly**. Three independent corroborations of the case15 link are in §2.2. Full text is **closed access** (`[EXTERNAL]` Unpaywall: `is_oa: False`, `oa_status: "closed"`), so the *section-number* claim (§4.1.3) remains inferred, not read.

The two remaining items resolve cleanly and negatively-in-a-good-way: the digitized CSV **is** the right data and copying it is safe (§3), and the `smooth_ok` mesh-invariant restatement is a **strict, bit-exact no-op** at current resolutions for a reason simpler than round 32 assumed (§4).

---

## 1. Item 1 — the Saurel/Petitpas/Berry 2009 paper

### 1.1 What was attempted and found

| Step | Result |
|---|---|
| `ls papers/md/` from `solver_4eq_mass/` | `[VERIFIED]` four unrelated `.txt` files — reproduces rounds 31/32's "absent" observation exactly |
| `readlink -f papers` from `solver_4eq_mass/` | `[VERIFIED]` `<worktree>/solver_denner/papers` — **`papers` is a symlink, not a directory**; this is the whole bug |
| `find` across the checkout for `*saurel*` | `[VERIFIED]` `<worktree>/papers/md/33_saurel_relaxation_multiphase.md` — **present** |
| `git ls-files papers/md/33_saurel_relaxation_multiphase.md` (from worktree root) | `[VERIFIED]` tracked. `git check-ignore` → exit 1 (not ignored) |
| Content spot-check | `[VERIFIED]` title, authors (R. Saurel, F. Petitpas, R.A. Berry), affiliations, `doi:10.1016/j.jcp.2008.11.002`, journal running head "*R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712*" — all match the `_needed.md` stub's metadata exactly |

**No fetch was attempted or is needed.** (For completeness: the paper-search MCP servers *are* available in this session, `pdf_to_md.py` exists at `~/.claude/skills/cfd-paper-search/pdf_to_md.py` — not at `papers/pdf_to_md.py` as the skill's description says — and `WebFetch`/`WebSearch` work. So had a fetch been required, the machinery existed. It wasn't.)

### 1.2 Every prior citation re-verified (line numbers are into the repo-root `papers/md/33_...md`)

| Cited by | Claim | Verdict |
|---|---|---|
| `YADV_ROUND_28_PLAN.md:407` | §3.3 "Relaxation step" at **lines 1088–1135** | `[VERIFIED]` line 1088 is exactly `3.3. Relaxation step`; lines 1090–1100 are the relaxation ODE system; line 1125–1135 is the `p̂_I` averaging derivation. **Exact.** |
| `YADV_ROUND_28_PLAN.md:407` | "volume-fraction positivity under strong expansion waves" at **lines 139, 292, 342** | `[VERIFIED]` :139 *"Volume fraction positivity, when dealing with shocks and strong expansion waves…"*; :292 *"…volume fraction positivity in the presence of shocks and even in the presence of strong rar[efactions]"*; :342 *"Volume fraction positivity. This is a particularly difficult issue…"*. **Exact, all three.** |
| `YADV_ROUND_29_PLAN.md:217`, `YADV_RESEARCH.md:4681` | §4.5 quote: *"As gas is present, the pressure cannot become negative. To maintain positive pressure, the gas volume fraction increases and creates a cavitation pocket."* | `[VERIFIED]` verbatim at lines 1341–1342. §4.5 "Cavitation test" begins at line 1336. **Exact.** |
| `YADV_ROUND_30_PLAN.md:376` + round 30 §8 | "**literally case15's own test, run at 1000 cells (2.5× ours)**" | `[VERIFIED]` **CONFIRMED VERBATIM.** §4.5, lines 1337–1340: 1 m tube, liquid water, `alpha_air = 1e-2`, velocity discontinuity at x=0.5, `u=-100/+100`, solution shown "at time t = 1.85 ms, using 1000 uniform mesh cells." |

**So round 30 §8's claim is now VERIFIED, not "unverified" as `YADV_ROUND_32_PLAN.md:125,248` recorded it.** Saurel §4.5 is the test's origin; Yoo & Sung §4.1.3 reproduces it, and `15_ref.png` is Yoo & Sung's figure, not Saurel's.

### 1.3 A genuinely new, first-class spec finding that falls out of this

`[VERIFIED]` Saurel §4.5 specifies **α_air = 10⁻² (1%)** and **t = 1.85 ms**. case15's own spec doc says "**공기 1%**" but `cases.cpp:682-688` codes `alpha = 0.055` (5.5%) and `t_end = 9.5e-4 s` (half of Saurel's). Record this. **Do not act on it**: changing `alpha`/`t_end` in `cases.cpp` is exactly the blocked case15 mesh/spec territory and is additionally G1-blocked.

### 1.4 What the implementing session must do

1. Re-verify R33-A. (done above)
2. **Do NOT create `solver_4eq_mass/papers/md/33_saurel_relaxation_multiphase.md`** — would duplicate a tracked file via the symlink into the wrong library.
3. **Do NOT delete/rewrite** the stub; append a status line correcting the "absent" record.
4. Write §1.2's verification table into `docs/YADV_RESEARCH.md` §43.

### 1.5 Does Saurel 2009 bear on any blocked thread? — yes, future-round context only

`[VERIFIED]` §4.5's own sentence — *"Excellent agreement with the exact solution of the 5-equation model [ref 37] is obtained"* — means the originating source compares against an exact 5-equation-model solution, relevant to both case15's exact-reference question and Phase 3a's model-extension scope. **Recorded as context. Licenses nothing now.**

---

## 2. Item 2 — the Yoo & Sung 2018 attribution

### 2.1-2.2 CONFIRMED (bibliographic), section number INFERRED (high confidence)

`[EXTERNAL]` **Young-Lin Yoo & Hong-Gye Sung**, *"Numerical investigation of an interaction between shock waves and bubble in a compressible multiphase flow using a diffuse interface method"*, **IJHMT 127 (2018) 210–221**, DOI **`10.1016/j.ijheatmasstransfer.2018.08.012`**. Three independent corroborations: (1) abstract mentions comparing against "Yeom et al.", matching `15_ref.png`'s legend; (2) paper's §4.1 uses a 1m/100-cell water-air shock tube matching case14's own cited IC; (3) `solver_5eq/results/1D/15_E/reference_digitized_15.csv` sits on exactly 100 uniform points, matching the source's own convention. Not confirmed: the literal "§4.1.3" string, the cavitation subcase's exact ICs, whether "Exact" is a homogeneous-mixture Riemann solution — full text is closed-access (Unpaywall: `oa_status: "closed"`), ScienceDirect gave HTTP 403 on WebFetch, no Sci-Hub fallback attempted (a licensing decision out of this loop's scope).

**Low-cost alternative path, recorded not acted on**: first author "Young-Lin Yoo" plausibly matches this repo owner's email (`younglin90@gmail.com`) — if so, asking directly would settle §4.1.3's details. Not raised via AskUserQuestion this round (would muddy the three already-pending blocked-thread questions).

### 2.3 Implementing session action

Rewrite `papers/2018_Yoo_Sung_cavitation_air_water_needed.md` with confirmed metadata (done, see commit). Update `validation/1D/15_E_Cavitation.md`'s "저자 미확정" hedge to the confirmed citation.

---

## 3. Item 3 — `results/1D/15_E/reference_digitized_15.csv`

### 3.1-3.2 Facts and validity checks

`[VERIFIED]` Present at `solver_5eq/results/1D/15_E/reference_digitized_15.csv` (12583 bytes, sha256 `5bb1e022...`), absent from `solver_4eq_mass/results/1D/15_E/` in both worktree and main checkout (gitignored in `solver_5eq`, not gitignored in `solver_4eq_mass`'s results dir). No code path reads it (grep across all trees → zero hits, only 3 `.md` references). Five independent checks confirm it IS the right data for case15 (far-field match, symmetry, timing bracket, figure-shape match, 100-cell grid convention match). Two honest limitations: it cannot resolve the core (digitization-floor artifact, `p_min=2000 Pa` vs the exact `9e-14 Pa`), and it cannot settle 1%-vs-5.5% air (both indistinguishable from zero on a 0-1.2 axis).

### 3.3 Recommendation: COPY — safe and appropriate, with mandatory provenance annotation

No code reads it; cannot appear in `git diff --stat -- cpp/`; cannot alter any gate. Required conditions: byte-identical copy (sha256-verified), copy only `reference_digitized_15.csv` (not the `_on_grid` variant), record provenance in the spec doc in the same commit including the explicit statement that this is auxiliary/non-gating and NOT a step toward replacing case15's reference (blocked-thread firewall).

---

## 4. Item 4 — offline no-op check of the mesh-invariant `smooth_ok` restatement

### 4.1-4.3 Definition and answer

Candidate: `cj <= max(3.04*dx/t_end, 1.10*cj_r)` where `3.04 = 8.0 * t_end/dx` exactly at case15's own N=400. **Leg A** (code-path): `smooth_ok`/`jump_stats` are declared entirely inside the `case_id=="15"` block (`validation.cpp:684-729`, verified: 4 hits for `smooth_ok`, all inside) — the other 18 cases provably never evaluate the restated expression, no run needed. **Leg B** (arithmetic): at case15's registered config, `3.04*dx/t_end == 8.0` bit-for-bit (using the solver's own `dx = (x1-x0)/n` definition). **Leg B′**: if `dx` were instead derived from dumped cell centres, off by 1 ulp — still a no-op given Leg C's margins, but the distinction must be reported honestly. **Leg C** (empirical margin): case15's own measured `cj` (6.044 OFF / ~30 ON) sit far from `8.0` in either direction — robust to any `dx`-definition wobble. **Leg D** (ground truth): confirm the 19×2 PASS/FAIL vectors are identical under current vs restated gate.

### 4.4-4.5 Scope limits

The check establishes the no-op ONLY at current registered resolutions — it says nothing about, and cannot test, other resolutions (that would need a `cases.cpp` edit, forbidden). This is not a defect; it's exactly why round 32 declined the restatement. Also: the candidate as written restates only the `cj` clause; `mj` and `cc` are left untouched by the candidate, an incompleteness worth recording.

### 4.6 Implementation

New script `scripts/yadv_r33_smooth.py`, importing `base_env()`/`dump()`/`validate_all()` from `yadv_r26_closure.py` (read-only reuse, not mutated). One mode, `--noop`, printing all four legs plus a final `NO-OP: CONFIRMED / REFUTED` line.

---

## 5. Blocked-thread firewall

Three threads (case15 exclusion-vs-exact-reference, case15's pface risk, Phase 3a scope) — **NONE contacted**. Two findings bear on them (Saurel's 5-eq comparison, §1.5; solver_5eq's own sibling-tree rejection of a same-solver reference, independently corroborating round 32 §42.3) — recorded as **FOR FUTURE USER DECISION — NOT ACTIONABLE IN THIS ROUND**, licensing nothing.

## 6. Zero-risk profile (pre-registered, all confirmed at execution)

`git diff --stat -- cpp/` empty; no `cases.cpp`/`validation.cpp` edit; no new env var; OFF 19/19 unchanged; `ACID_YADV=1` 15/19 `{15,24,33,34}` unchanged; unit tests unchanged; byte-identical vs published `solver_denner`; at most 2 new files (`scripts/yadv_r33_smooth.py`, the CSV copy); docs/data-only edits elsewhere.

## 7. Verdict framework

S1 expected (all four items resolved). `consecutive_failures` not incremented — correcting two standing errors in the research record (R33-A, R33-B) is measured progress. Round 33 cannot and does not try to produce a `pass_count` change. **Round 34's entry condition becomes strictly true** (no housekeeping backlog left) rather than "true modulo four loose ends" — round 34 needs a fresh user decision on one of the three blocked threads.

---

## Actual outcome (implementing session, post-hoc)

**R33-A and R33-B independently re-verified, not just transcribed from the plan.** The
implementing session ran its own `readlink -f papers`, `git ls-files`, and `sed -n` spot-checks
on the cited line numbers (1088, 1336-1342, 139/292/342) — all matched. For R33-B, the
implementing session made its own live `get_crossref_paper_by_doi` call (not reusing the
planning pass's cached result) and got an identical match (title, authors, volume 127, pages
210-221, published 2018-12-01) — per this project's own discipline of not trusting even a
Planner's external-tool-derived claims without independent re-confirmation.

**Item 3 (CSV copy)**: executed exactly as specified — `sha256sum` before, `cp -n`, `sha256sum`
after, both hashes identical (`5bb1e022...`). Single new tracked file confirmed via `git status
--porcelain`.

**Item 4 (no-op check)**: `scripts/yadv_r33_smooth.py` needed one fix during implementation — an
f-string syntax error (nested escaped quotes not valid in this Python's f-string grammar),
caught immediately by the script failing outright rather than producing a wrong number. Fixed by
extracting the needle string to a variable. After the fix, all four legs ran clean and confirmed
the no-op exactly as predicted. One clarification worth recording: Leg C's "ACID_YADV=1" config
measures `cj=2.306467` — this is *plain* config B (the actual configuration the loop's own
15/19 headline gate uses), not config C's `cj≈30` core-jet value from rounds 27-32 (which
requires the separate `ACID_YADV_ALPHA_IMPLICIT=1` flag). This is correct scoping, not an
inconsistency with prior rounds — the no-op check only needs to cover the two configs that
actually participate in the loop's own G3/G5 gates.

**Verdict: S1, exactly as predicted.** All hard gates held (`git diff --stat -- cpp/` empty, OFF
19/19, `ACID_YADV=1` 15/19 fail-set `{15,24,33,34}` unchanged and reconfirmed fresh, unit tests
unchanged). `consecutive_failures` NOT incremented. Round 34's entry condition is now strictly
"no authorized work without a fresh user decision" — confirmed, not just asserted.
