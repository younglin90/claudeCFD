# YADV Round-Loop Roadmap

State file the `yadv-round` skill (`.claude/skills/yadv-round/SKILL.md`) reads/writes every round.
Detailed research log stays in `docs/YADV_RESEARCH.md`; this file is a thin, compact control
document — round counter, stop conditions, next-task pointer, one-line-per-round history. Do not
put derivations or measurement tables here; put them in `YADV_RESEARCH.md` (or a per-round plan
doc) and link to them.

## Current goal — Phase 3a: the cases-24/33/34 conservation defect

**Re-armed after round 9.** Phase 2 (Stages 0-4) is COMPLETE (`YADV_RESEARCH.md` §19):
`pass_count >= 14` met and durable since round 6; `ACID_YADV_ALPHA_IMPLICIT` does NOT fold into
`ACID_YADV` (case14); both stay default OFF. That investigation is closed, not reopened by Phase 3.

**New goal, chosen by the Advisor from the two candidates round 9 left open** (P3a over P3b,
because P3a is the only one that could change `ACID_YADV`'s recommended status and it now has a
concrete, promising, previously-unmeasured lead): **investigate why case33's Rankine-Hugoniot jump
closes to machine precision under `ACID_YADV_ALPHA_IMPLICIT=1`** (momentum residual 88% ->
8.4e-13, `YADV_RESEARCH.md` §19.4) **while case24/34 instead show their shocks exiting the domain
before `t_end` under the same flag** -- not yet understood whether this is the same phenomenon
further along (a faster, still-admissible shock) or a different problem specific to those two.
This directly contradicts the prior belief (rounds 4-8) that Jacobian work moved 24/33/34 "by
nothing" -- that was only ever true of the validation-gate metrics, never checked against
conservation self-consistency with implicit alpha until round 9.

**Round 10's job (per the same pattern Phase 2 itself started with): produce
`docs/YADV_PHASE3_PLAN.md`.** A Planner call, grounded in `YADV_RESEARCH.md` §11 (the original
Y-consistent-Hugoniot derivation and closure-(A)-vs-(B) analysis), §14.3/§15.5 (round 3/4's
Rankine-Hugoniot measurements), and §19.4 (this round's finding) — NOT a re-derivation from
scratch. The plan should: (1) explain WHY case33 differs from case24/34 (their `alpha_pre` are
0.75, 0.50, 0.25 respectively -- not an obvious pattern, worth checking first); (2) determine
whether case24/34's early-exiting shock is genuinely faster (consistent with also being
admissible) or symptomatic of something else (extend `t_end`, or re-run with a larger domain, to
find where their shock actually lands, before assuming); (3) stage the actual investigation the
same way Phase 2 did -- smallest safe measurement first, then incremental, falsifiable steps; (4)
set an explicit stopping criterion up front (Phase 2 §12's own lesson: "(a) without (b) is
insufficient" for a two-part fix -- decide in advance what result would mean "this needs a
different closure entirely, stop chasing it incrementally").

**Backup candidate, not this round's goal**: (P3b) case15's central-jump defect (§17.4, `cj=30.02`
vs threshold `8.0` at the stagnation point). Narrower, would not change `ACID_YADV`'s status. Available
if P3a proves unproductive (three consecutive rounds of no measured progress triggers the stop
condition below regardless, at which point re-evaluate between P3a-continued and P3b).

## Control state

```
round_counter: 24
consecutive_failures: 0
done: false
next_task: Round 24 discovered round 23's "roundoff-null control" (ACID_PROJ_UNTIL=1) was a
           COMPLETE NO-OP, not a perturbation -- the exact-skip fires on all 800 cells at step 0
           (IC already PTE-consistent), so 0 writes occur for the whole run and P6' carried no
           information about H-B either way (a correction, not a retraction: the conclusion turned
           out right, the evidence for it was vacuous). Also corrected round 23 sect.33.4's
           mechanism reading of N=50: the "frozen shock" (from max|u|/maxp freezing in a coarsely-
           sampled trace) is actually a shock that COMPLETELY EXITED the domain, 84% overstrong,
           32% overfast, alpha collapsed 0.5->2e-4 -- an over-fast/over-strong/alpha-collapsing
           shock, not a stalled one. Built the REAL roundoff-null control this round
           (ACID_RECON_NULL, using the existing unit-tested 8*eps*kappa round-trip conditioning
           bound as alpha_roundtrip_floor -- no new constant): non-empty (2-4 cells/step genuinely
           written, never zero) and applied. Result: BYTE-IDENTICAL to plain B anyway --
           H-B (Newton-trajectory chaos) excluded, this time on solid evidence. Confirmed no
           GLOBAL withdrawal-point criterion exists (ntouch=0 only at step 0, never again -- the
           correction is needed continuously wherever the front is inside the domain) AND the
           always-on family member is itself wrong (stalls on its own round16 sect.26.1 blister at
           step399) -- so no withdrawal schedule within the ACID_PROJ_UNTIL family has a correct
           member to find. Verdict: S1 (question was mis-posed, as pre-registered/expected) -- no
           taper designed or built. ACID_RECON_NULL/alpha_roundtrip_floor committed as inert
           research infrastructure. consecutive_failures NOT incremented.
           Live threads for round 25, carried forward: (a) RECOMMENDED -- the recovery-site fix
           (round16 sect.26.3's F3, made concrete): recover alpha at the NEW Y's own PTE state
           (p*,T*)=pT_from_v_e_massfrac(1/rho,hstat-p/rho,Y,A,B) instead of the stale (p_o,T_o),
           writing ONLY s.alpha (never p,T, so it structurally cannot reproduce round22's
           Abgrall-type pressure perturbation on 13/14) -- pre-registered risk: the Eqs.43-44
           rebuild would then blend phase densities at (p_o,T_o) with an alpha from (p*,T*),
           breaking the documented "same triple" invariant -- must measure RMISM's drho before
           any gate; (b) case13's Jacobian-approximation-sensitivity finding (round22 sect.32.1) as
           its own narrower question; (c) round21's carried-forward thread -- rho_star continuity
           predictor and theta_o MWI memory, untouched, may matter for case34/33's residual floor;
           (d) case33's own difficulty (round15/16's T-ceiling-saturation-from-advection channel)
           remains fundamentally unsolved; (e) max_steps exhaustion (case15 legitimately uses it
           and PASSES on OFF -- needs careful design); (f) case29's (excluded) likely-explained
           blocker -- not pursued, recorded for the record.
           Grounded in YADV_RESEARCH.md sect.34, docs/YADV_ROUND_24_PLAN.md.
```

(Round counter starts at 4 because rounds 1-4 of the `ACID_YADV` experiment were already run
manually, before this loop existed — see `docs/YADV_RESEARCH.md`. Round 5 onward runs under
`yadv-round`.)

## Stop conditions (checked at the start of every round, before any work happens)

1. `done == true`
2. `consecutive_failures >= 3` — three rounds in a row with no measured progress or a hard-gate
   failure. This means the current approach needs human reconsideration, not another autonomous
   attempt at the same thing. This is the backstop that actually governs day-to-day; expect the
   loop to stop here long before either cap below, unless Phase 3 turns out to be unusually clean.
3. `round_counter >= 1000` — session cap set explicitly by the user for this Phase 3 run (2026-08-
   01). Not expected to be reached (condition 2 should fire first if Phase 3 stalls); exists as a
   hard ceiling regardless. Re-raise explicitly if the user wants more headroom after this caps out.
4. `round_counter >= 100000` (the original nominal cap from when this loop was first set up;
   condition 3 is stricter and fires first)

Any of these: the round skill calls `ScheduleWakeup({stop: true})`, records why here, and does
not start a new round.

## Absolute rules (every round, no exceptions — also mechanically enforced by
`.claude/hooks/agent_plan_only.py` + `.claude/hooks/block_destructive_bash.py` while
`.claude/round-loop-active` exists)

- No agent except `Agent(subagent_type="Plan")`. Implementation happens directly in the round's
  session.
- No `git push`, no `git reset --hard`, no `rm -rf`. Local commit + local merge to `main` only.
- No tuning constants, no per-case coefficients. Global physical constants only.
- No edits to `cases.cpp` / `validation.cpp` unless the round's stated goal explicitly requires it
  (none of the current roadmap does).
- The OFF path (`ACID_YADV` unset) must stay 19/19 and byte-identical to the published
  `solver_denner` binary at the end of every round that merges to `main`. A round that breaks this
  does not merge.
- Report negative/partial results honestly — this project's established culture (rounds 1-4) keeps
  failed experiments in the history, not just wins.

## History (compact — detail lives in YADV_RESEARCH.md / per-round plan docs)

- Round 1: Y-space non-conservative transport. case13 win (linf_p -36%), case02/14 THINC-averaging
  regression, case15/24/33/34 fail. → `YADV_RESEARCH.md` §1-8.
- Round 2: alpha-space THINC reconstruction — NEGATIVE (case02 3.6x worse). Y-consistent Hugoniot
  for 24/33/34 computed — overturns round-1's "unanswerable closure conflict" framing; Y-path
  violates momentum by 88%. → `YADV_RESEARCH.md` §9-12.
- Round 3: conservative `rho*Y` transport, sharing ACID's own per-cell mass-flux structure.
  §10.4's target (case02/14) still not recovered; case13 wins partly lost (`peak_delta_u` 0→397).
  New partial win: case24/34 now satisfy their own Rankine-Hugoniot jump to 1e-13 (were
  unmeasurable before); case33 unchanged (88% violation). → `YADV_RESEARCH.md` §14, commit
  `a26b7a4`.
- Round 4: alpha implicit in Newton via re-evaluation inside `compute_R` (no Jacobian edit).
  Confirms case15's grid-convergence defect was the frozen-alpha mechanism (cured under FD
  Jacobian, `amp_ratio_p` 0.33→1.00) but is a net regression under the DEFAULT analytic Jacobian
  (15/19→12/19, newly breaks 13/14/25) — the analytic Jacobian's missing `d(alpha)/dp` term is
  load-bearing. Gated OFF behind `ACID_YADV_ALPHA_IMPLICIT` (default off); plain `ACID_YADV=1`
  unaffected. → `YADV_RESEARCH.md` §15, commit `47d1bef`.
- Phase 2 plan produced (Planner, opus): analytic `d(alpha)/dp`/`d(alpha)/dT` derivation, staged
  Stage 0-4 implementation, grounded in 4 papers (Denner 2018 product-rule linearisation, Janodet
  2025 coupled-large-density precedent, Denner/Evrard/vanWachem 2020 barotropic-substitution
  consistency argument, the original ACID paper as historical baseline). → `docs/YADV_PHASE2_PLAN.md`.
- Round 5 (first run under `yadv-round`): Phase 2 Stage 0 -- `d(alpha)/dp`, `d(alpha)/dT` header
  helpers in `eos.hpp` (additive, no call sites yet) + unit-test verification. Confirmed Phase-2
  §1's numeric prediction exactly (case15 diagonal-error ratio 521.558 vs predicted ~500). Found
  and fixed a bug in the new unit test itself (FD-comparison tolerance floor was 1e6x too strict
  for the air|vapor pair's algebraic-zero case) -- the derivative formula was correct throughout,
  independently confirmed against a standalone probe. All four gates unchanged (additive stage):
  OFF 19/19+9/9, ON 15/19, +ALPHA_IMPLICIT 12/19 both Jacobians, same failure sets as round 4.
  Also the first real exercise of the round-loop mechanism itself -- found and fixed two
  infra bugs (worktree default branches from origin/main, invisible to local-only commits, needed
  `worktree.baseRef:"head"`; the Bash safety hook's substring match blocked its own guarded commit
  on a heredoc-quoted mention of what it blocks, fixed with heredoc-body stripping). → commit
  `7cd36ae`, `docs/YADV_ROUND_5_PLAN.md`.
- Round 6 (first round run under the actual autonomous `/loop`): Phase 2 Stage 1 -- augmented the
  analytic Jacobian's J1 cell-EOS-chain loop to star `D_p`/`N_p` with the `a_p` product-rule
  addend (Stage 0's already-derived, already-unit-tested helper). GENUINE SUCCESS:
  `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` under the default analytic Jacobian moved 12/19 -> 14/19.
  case13 and case25 both fully recovered; case15 moved from non-convergence to passing every
  quantitative gate criterion (blocked only by the TV/oscillation guard, `peak_delta_u` 321->0).
  Cases 24/33/34 unmoved as predicted (separate conservation defect); case14 unmoved (separate
  `hsT<0` lead). All hard gates held (OFF 19/19+9/9, plain ON 15/19, FD-invariance exact same
  failure set). → `YADV_RESEARCH.md` §16, `docs/YADV_ROUND_6_PLAN.md`, commit `33e006f`.
- Round 7: Phase 2 Stage 2 -- the J2 flux-blend diagonal (own-cell alpha sensitivity in the ACID
  mass/energy flux blend). MEASURED NO-OP on `pass_count` (stayed 14/19, identical failure set) --
  an honest negative result, consistent with the round's own prediction that 13/25 had nothing
  left for J2 to add. The real result: corrected round 6's case15 diagnosis. `peak_delta_u` is NOT
  part of case15's gate (verified against `validation.cpp` directly); computed the true blocker
  exactly -- a central-jump/concentration failure (`cj=30.02` vs threshold `8.0`) at the domain's
  stagnation point, structurally unrelated to alpha-Jacobian accuracy. **Roadmap goal re-scoped**
  (see "Current goal" above) -- case15 is no longer a target of this plan. All hard gates held.
  → `YADV_RESEARCH.md` §17, `docs/YADV_ROUND_7_PLAN.md`, commit `9b0a698`.
- Round 8: Phase 2 Stage 3a (T-pathway). Measure-first: a temporary diagnostic (removed after use)
  confirmed round 5's `hsT<0` case14 lead was real but confined to a single first-timestep
  transient cell, not persistent -- worth attempting per the round's own decision rule. Starred
  the T-pathway analogous to Stage 1's p-pathway; genuine side discovery: `hsT* = Y*cp_a+(1-Y)*cp_b`
  is an EXACT closed form (hstat_mix = Y*h_a+(1-Y)*h_b identically, NASG h linear in T), strictly
  positive, retroactively validating Stage 1 and proving the starred form removes an existing
  1/hsT near-singularity. MEASURED REGRESSION on its target: case14 doesn't flip pass/fail but its
  quality collapses (l2_p 0.0145->0.512, corr_p 0.9996->0.594) -- confirms a family-mismatch risk
  flagged in advance (mirror of round 4's original mistake). Gated behind a NEW flag
  `ACID_YADV_ALPHA_IMPLICIT_T` (default off, round-4 precedent) so `ACID_YADV_ALPHA_IMPLICIT=1`
  alone stays bit-identical to round 6/7's validated 14/19. Advisor declined the larger Stage 3b
  (residual-level fix) for now -- its case is performance/robustness, not a case14 fix, and it
  flips the FD-invariance gate for no currently-open target. All hard gates held.
  → `YADV_RESEARCH.md` §18, `docs/YADV_ROUND_8_PLAN.md`, commit `3446bc5`.
- Round 9: Phase 2 Stage 4 (consolidation). Zero solver code changed -- measurement/reporting only
  via a new `scripts/yadv_r9_sweep.py`. Full six-configuration sweep reproduced every prior round's
  numbers exactly, from one build, in one sitting (found and fixed a live parsing bug along the
  way: lowercase `nan`/`-nan` broke Python's `json.loads`, silently undercounting failure sets --
  `pass_count` itself, computed by the C++ binary, was never wrong). First direct wall-clock
  measurement in the whole investigation: `+ALPHA_IMPLICIT` costs 7.1% more than plain
  `ACID_YADV=1` on cases both configs solve (vs Phase-2's own <5% prediction); FD Jacobian costs
  54.7% more than analytic on the same config (consistent with round 4's qualitative ~1.7-1.9x).
  Promotion decision: does NOT fold `ACID_YADV_ALPHA_IMPLICIT` into `ACID_YADV` -- case14 is the
  ONLY case whose pass/fail flips between the two (plain ON passes it, +IMPLICIT fails it),
  confirmed fresh from this round's own sweep. **Post-merge follow-up (commit `6f1538d`) found a
  genuinely new, unpredicted result**: the deferred RH residual re-check
  (`scripts/yadv_rhcheck.py`) under `ACID_YADV_ALPHA_IMPLICIT=1` shows case33's Rankine-Hugoniot
  jump closing to machine precision (88% -> 8.4e-13) -- contradicting the standing "Stages 1/2/3a
  moved 24/33/34 by nothing" belief, which was only ever true of the validation-gate metrics.
  case24/34 instead show their shocks exiting the domain early under the same flag, unexplained.
  Phase 2 (Stages 0-4) declared complete; roadmap re-scoped to `done: true` with this new finding
  named as the sharpest lead for a future Phase 3 (P3a). All hard gates held throughout.
  → `YADV_RESEARCH.md` §19, `docs/YADV_ROUND_9_PLAN.md`, commits `51e2497`, `6f1538d`.
- Round 10: Phase 3a Stage 0. **RETRACTS round 3's "24/34 close to 1e-13" and round 9's "case33
  closes to machine precision under +ALPHA_IMPLICIT"** -- both were `yadv_rhcheck.py`'s
  undisturbed-cell search locking onto a SILENTLY STALLED run's pristine IC (`acid.cpp`'s
  `if (!stepped) break;` never sets `diverged`, so a stalled run scores as a normal completion).
  Independently reproduced via direct `ACID_DBG` traces (case24 plain, case33 +IMPLICIT both
  stall at <1% of `t_end`, mid-domain cells still bit-for-bit pristine IC). New
  `scripts/yadv_rh2.py` (null-run + IC-match guard, exited-shock plateau-window fallback):
  confirmed ALL six predicted null-run/completion classifications exactly. Load-bearing finding:
  **no case in {24,33,34} has ever had a plain and a +IMPLICIT run complete simultaneously** --
  every RH-residual number this whole investigation has produced for these three cases compares
  across different configurations, never a true A/B. Where 24/34 do complete (+IMPLICIT, shock
  exited domain), residual is large (~40-50%), not near-zero -- measured value disagrees with the
  round-10 Planner's own predicted ~2-7%/faster-shock figures (`YADV_PHASE3_PLAN.md`); likely a
  plateau-window artifact straddling internal wave structure, not yet resolved (needs Stage 2's
  `ACID_TEND_SCALE` in-domain measurement). `Y` conserved to 3-4 digits through the leading shock
  in every completing run, against closure-(A)'s 270-1620x growth requirement -- may make 24/33/34
  a structurally unreachable target for any Y-preserving scheme, pending Stage 1-4.
  `ACID_YADV` recommended status UNCHANGED (default OFF, 15/19). Zero solver code changed; all
  four hard gates held (unit PASS, OFF 19/19, 9/9 byte-identical vs published binary,
  ON-vs-OFF genuinely differs as expected). → `YADV_RESEARCH.md` §20, `docs/YADV_PHASE3_PLAN.md`,
  commit `666c6c8`.
- Round 11: Phase 3a Stages 1+2. Stage 1 makes the silent stall audible (`STALLED:` stderr
  message + `ACID_DBG`-gated detail, `diverged` deliberately still NOT set -- that's Stage 3c,
  needs an explicit Advisor decision). Verified stderr-only against an isolated main-HEAD build:
  all 19 OFF dumps and OFF/plain-ON/+IMPLICIT stdout byte-identical. Side finding: the
  FD-invariance gate is 13/19 (`{15,24,27,28,33,34}`), not the 12/19 this project's own memory
  had recorded -- a stale prior figure, corrected, confirmed identical on unmodified main HEAD
  (not a regression). Stage 2 adds `ACID_TEND_SCALE` (diagnostic observation-window knob,
  default 1.0 = no-op, verified byte-identical to Stage 1 when unset). **The sweep this knob
  enables RETRACTS round 10's own §20.2 finding**: `yadv_rh2.py`'s fixed `[0.3,0.6]` plateau box
  straddled internal wave structure exactly as its own caveat warned; a corrected front-derived
  window (found and fixed a bug along the way -- an earlier attempt sampled the wrong, still-
  undisturbed side of the front) gives a STABLE, converged plateau across scales 0.4-1.0 for
  cases 24/34 under `+ALPHA_IMPLICIT`: momentum residual `+7.351e-02`/`+2.063e-02`,
  `Vs(mass)/Vs_ref` 1.4946/1.3968 -- matching the round-10 Planner's original static prediction
  to 3-4 significant figures (the Planner was right; round 10's own measurement was wrong). An
  independent front-position-vs-time linear-fit cross-check disagrees (R²=0.82/0.96, smaller
  ratio) -- reported as an open, unreconciled discrepancy, not forced to match. No case in
  {24,33,34} has ever had plain and `+ALPHA_IMPLICIT` complete simultaneously (round 10's
  load-bearing finding stands). `ACID_YADV` status unchanged (default OFF, 15/19). Zero effect
  on numerics with either new env var unset; all hard gates held throughout both stages.
  → `YADV_RESEARCH.md` §21, `docs/YADV_ROUND_11_PLAN.md`, commits `272ce08`, `587c3f8`, `be83576`.
- Round 12: Phase 3a Stage 3, branch 3a (retry-exhaustion accept-best). **3b REFUTED** by direct
  evidence -- all three stalling configs show `cell=-1`, `max|u|` numerically frozen across the
  full retry sweep, no void cell ever. Decisive mechanism finding, independently reproduced: the
  pre-Newton residual `r_init` grows exactly as `1/dt` at the stall (falsifies the existing `bad`
  gate's own "dt too large" design comment). New `ACID_STALL_ACCEPT` (default 0, byte-identical):
  level 1 gets **case24 and case34 (plain) to complete for the first time ever**, zero
  `pass_count` regression -- far exceeding the round's own calibrated expectation (every accepted
  step is a tight cluster right at shock formation, <0.5% into each run; both then run clean to
  `t_end`). **Produces the first-ever controlled A/B for cases 24/34**: plain+accept shows
  momentum residual +98.0%/+45.6% vs `+ALPHA_IMPLICIT`'s clean +7.35%/+2.06% -- favors
  `+ALPHA_IMPLICIT` sharply, but reported with an explicit caveat (the plain runs are not clean
  solves; the large residual may partly reflect the accept mechanism's own defect at formation,
  not intrinsic physics -- not yet a clean result). Level 2 (CFL-neutrality) is a measured net
  regression (plain-ON 15/19→14/19, case28 newly fails) and is NOT adopted; level 1 is the
  recommendation. Case33 remains unsolved -- a qualitatively different, sustained difficulty (221
  accepted steps at a 5x larger budget, still only 14.8% of `t_end`), not a one-time glitch like
  24/34. `ACID_YADV` status unchanged (default OFF, 15/19 with the new var unset); all hard gates
  held. → `YADV_RESEARCH.md` §22, `docs/YADV_ROUND_12_PLAN.md`, commits `a21ed4a`, `5017362`.
- Round 13: Phase 3a Stage 3, follow-up 7a (localize + fix the `1/dt` mechanism). New `ACID_RINIT`
  instrument **confirms the mechanism to textbook clarity**: at case24's stalling step, `dal_remap`
  (alpha recovered at the PREVIOUS step's frozen Y meeting the CURRENT step's `p_o,T_o`) is
  constant to 4-5 sig figs across all 13 retries, while `dal_adv` (this step's own Y-transport)
  halves exactly every retry. Two controls land exactly on their predicted extremes: OFF prints
  zero diagnostic lines (structural immunity, no remap term exists); `+ALPHA_IMPLICIT`'s
  `dal_remap` measures at literal `DBL_EPSILON` -- giving round 12's A/B result a mechanism, not
  just a correlation. **The naive fix (`ACID_YADV_HREINIT`, reset `s.h` alone) is REFUTED (S4)**:
  case24's stall only moves from step 19 to step 28 (<2% of steps needed); case34 gets WORSE;
  combined with round 12's `ACID_STALL_ACCEPT=1`, case24 needs MORE accepts (9, not 2). Diagnosed
  why: `s.rho`/`s.T` stay stale until the first `compute_R()` re-derives them from the corrected
  `h`, so Newton still doesn't start from a genuinely self-consistent state -- a real fix needs
  simultaneous `(T,rho)` reconciliation, not attempted this round. Both new flags stay default
  OFF; round 12's `ACID_STALL_ACCEPT=1` (without `HREINIT`) remains the only working path to
  completion, its caveat unchanged. Per this project's negative-result culture, a
  correctly-instrumented refutation is measured progress -- `consecutive_failures` not
  incremented. All four hard gates held, byte-identical to round 12 with both new vars unset.
  → `YADV_RESEARCH.md` §23, `docs/YADV_ROUND_13_PLAN.md`, commits `aedb1b5`, `dea10ce`.
- Round 14: Phase 3a Stage 3c (`diverged=true` at the retry-exhaustion give-up), authorized this
  round by an explicit Advisor decision after rounds 11/12/13 each deferred it. One statement,
  correctness/reporting fix -- `pass_count` UNCHANGED in all four hard gates (OFF 19/19, plain-ON
  15/19, `+ALPHA_IMPLICIT` 14/19, FD-invariance 13/19 both configs, resolving the round-12/13
  ambiguous 12-vs-13/19 figure by direct measurement). Exactly the three predicted `(case,config)`
  pairs changed and nothing else: 24/plain, 34/plain, 33/`+IMPLICIT` now read `finite=false`/NaN
  instead of a finite garbage row -- closing the exact silent-stall gap that produced two retracted
  findings (§20). The accept/give-up boundary needed no new logic: `ACID_STALL_ACCEPT`'s
  accept-and-continue path is confirmed byte-identical for case24/34 (never reaches the give-up
  block), while its give-up path (case33's own budget exhaustion) now correctly diverges alongside
  its unchanged `STALL-ACCEPT-TOTAL` disclosure. OFF-path safety re-verified empirically (zero
  STALLED/DIVERGED lines across all 19 cases on the post-edit build), not just asserted -- the
  change is NOT `ACID_YADV`-gated by construction, only by evidence. Four historical-artifact
  annotations added (§14.3/§19.2/§20.3/§19.3), no history edited. `ACID_YADV` status unchanged
  (default OFF, 15/19). Two open threads remain (round 13's harder consistency fix, case33's
  sustained difficulty); a third (max_steps exhaustion, a sibling defect) named but deliberately
  not pursued -- case15 legitimately uses that cap and PASSES on OFF, so extending `diverged` there
  needs careful future design, not an autonomous default.
  → `YADV_RESEARCH.md` §24, `docs/YADV_ROUND_14_PLAN.md`, commit `b67850e`.
- Round 15: first-ever diagnosis of case33's stall (`ACID_RINIT` had only run on case24/34 before).
  **Same residual SHAPE as round 13's mechanism** (energy-dominant `fene->1.0`, `r_init` doubles
  almost exactly per dt-halving) **but provably NOT the same SOURCE**: `dal_remap` measures at
  literal `DBL_EPSILON` at every retry of the first stall AND across 4 consecutive forced-accept
  steps -- the alpha/Y-REMAP channel round 13 found for case24/34 is completely absent here, as its
  own `+ALPHA_IMPLICIT` control predicted structurally. The actual driver is `dh=|s.h-Htot_o|`,
  constant `~3.7e12` and dt-independent -- **six orders of magnitude beyond the worst value
  (`~4e6`) ever seen across case24's entire healthy `+ALPHA_IMPLICIT` trajectory**, measured as a
  control. Round 13's speculated "alpha inherits the previous step's lag" compounding is directly
  refuted for case33 (`dal_remap` stays clean through repeated forced accepts) -- but real
  compounding IS measured, through `dh`/`drho` directly (+51%/+5x across 4 accepted steps), a
  different and still-unidentified channel. Diagnostic-only round, no fix attempted or
  implemented, no source code changed (`git status` confirmed clean), no hard gates required (no
  source change to verify a no-op against). `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §25, `docs/YADV_ROUND_15_PLAN.md`, commit `a56627a`.
- Round 16: case33's `dh` **fully explained** via new `ACID_RCELL` per-cell diagnostic (read-only
  window print, right after the Eqs.43-44 rebuild). A single first-time-step Y-advection into a
  still-pre-shock cell maps through `alpha_from_mass_fraction` almost to pure phase (case33's
  `Y_post=0.9344` sits closest of {24,33,34} to the `alpha->1` singularity) -- recovered density
  collapses **79x in step 0 alone** (250.4->3.16), to a **literal vacuum** (`1.24e-6`, 1.3e8x below
  correct) by the stall. `T_from_hstat` silently saturates at its `1e6` ceiling there (confirmed by
  direct code read: still returns `true` when saturated) -- `dT/dh=0`, `drho/dh=0`, so Newton can
  never recover at ANY `dt`, exactly matching every round-15 symptom with no alpha/Y-REMAP
  involvement. **The shock never moves**: cells where the true front should be by the stall time
  are still pristine IC to 4+ sig figs -- the solver re-fights a frozen 2-cell blister at the
  initial discontinuity for all 100 steps. **No fix attempted** (per the plan's own decision rule --
  no isolated off-by-construction bug found; this is genuine Y-form ill-conditioning for
  homogeneous-alpha/large-Y-jump cases, closed into Newton by `+ALPHA_IMPLICIT`'s own per-call alpha
  re-derivation). Three candidate fixes named for a future round, priority order **F3 > F1 > F2**
  (break the p→alpha feedback in the residual; upper-bound `s.h`; make `T_from_hstat` report
  saturation -- each with a stated reason it isn't a safe drive-by). Full hard-gate battery held,
  byte-identical to round 14's baseline; `ACID_RCELL` stays default OFF. `ACID_YADV` status
  unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §26, `docs/YADV_ROUND_16_PLAN.md`, commit `004e4e3`.
- Round 17: F2's actual risk measured with new `ACID_TSAT` instrument (deliberately NOT
  `yadv`-gated -- unlike `ACID_RCELL`/`ACID_RINIT`, it can observe the OFF path). **V-SAFE, zero
  exceptions**: across all 19 graded cases, >400,000 residual evaluations, **not one cell EVER
  reaches the `T_from_hstat` ceiling** -- not even a rejected transient iterate of case28 (the
  analytically closest case, 0.587x). Round 16's stated risk basis corrected, not edited: case29
  isn't graded at all (excluded); 13/14/25 sit 3-4 orders of magnitude below the ceiling. **But F2
  as literally named is the WRONG SHAPE independent of the measurement**: its `false` branch
  freezes `s.T`, making `compute_R` a function of call history rather than state -- breaks the four
  `compute_R(); // restore` sites (load-bearing for the FD-Jacobian assembly) and gives `dT/dh=0`
  exactly, the very failure mode round 16 diagnosed as why Newton can't recover. **Corrected form
  `F2''` pre-registered** (state-pure `T_from_hstat`, report saturation to the caller as a new
  stall reason instead, reusing the existing dt-halving retry machinery) -- priority list now
  **F3 > F2'' > F1**. No fix implemented (round's own bar -- "free AND fully specified" -- not met
  until F2 was corrected to F2''); F2'' is now well-specified and risk-cleared enough for direct
  implementation in a future round. Side finding for the record: case29's analytic post-shock
  T=2.93e6K exceeds the solver's own clamp by 2.93x, plausibly explaining its long-unexplained
  blocker comment. Full hard-gate battery held, byte-identical to round 16's baseline; G6 (stronger
  than round 16's equivalent -- this flag actually executes on OFF) confirms zero perturbation.
  `ACID_TSAT` stays default OFF. `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §27, `docs/YADV_ROUND_17_PLAN.md`, commit `d4ac63a`.
- Round 18: F2'' implemented (`ACID_TSAT_STALL`, default OFF) -- state-pure `T_from_hstat` (no
  signature change), a saturated retry becomes a new stall reason 5 that displaces reason 1 and is
  therefore automatically ineligible for `ACID_STALL_ACCEPT`'s capture, no separate code change
  needed there. **OFF/plain-ON/`+ALPHA_IMPLICIT` all confirmed byte-identical**, exactly as round
  17 predicted -- the diagnostic instrument's deductive proof held under the real new code path.
  **Unpredicted, positive result**: Stage 0's first-ever 5-config saturation sweep found the
  FD-Jacobian path (`ACID_NO_AJAC=1`, independent of `ACID_YADV`) already silently accepting
  T-ceiling-saturated states as "converged" on cases 24/27/28/33/34 -- turning the flag on
  **RECOVERS cases 27 and 28 from silent NaN failure to genuine PASS** (FD gates move
  13/19→15/19 and 13/19→14/19). No regression found in any tested configuration.
  `ACID_STALL_ACCEPT` interaction: case33 fails faster and cleaner exactly as predicted (0 accepts
  vs 4, step 43 vs 104, same non-completion); case24 byte-identical; **case34 shows a small, real,
  honestly-reported non-byte-identical perturbation** (~6th significant figure, same physical end
  state) when BOTH mechanisms are combined -- does not affect round 12's published numbers (which
  use `ACID_STALL_ACCEPT` alone). G9's full 20-combination accepted-state invariant sweep: zero
  violations. Flag stays default OFF (this round establishes safety+benefit when explicitly
  enabled, not promotion). `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §28, `docs/YADV_ROUND_18_PLAN.md`, commit `207874b`.
- Round 19: round 18's case34+`ACID_STALL_ACCEPT`+`ACID_TSAT_STALL` perturbation **fully
  localized, no new source code needed** -- existing round 13/16/17/18 instrumentation sufficed.
  Channel enumeration (a code property, not case-specific): the new flag can only perturb a
  trajectory via two channels, C1 (a would-be-accepted retry rejected for carrying a saturated
  cell) or C2 (an already-rejected retry's exclusion changes the accept-candidate winner) -- C2
  re-confirmed excluded (all four `STALL-ACCEPT:` events identical, direct diff). **Decisive test**:
  the unmodified baseline, instrumented, shows `accepted_steps_hi=3` -- three normally-accepted
  steps (325/326/329) already silently carry a saturated cell today. With the flag added, the two
  runs are **bit-identical through step 325 exactly** (`dt` ratio 1.000000 at every step), diverge
  at 326 (ratio 0.28-6.05x through step 336 -- the predicted `cfl_scale` cascade), then settle to a
  few percent by step 337 and continue damping toward parity. Both runs reach the **identical**
  final `t` to 9 significant digits (2648 vs 2646 steps). H0 (nondeterminism) and instrument-
  neutrality controls both pass. Two corrective annotations to round 18 recorded (not edited
  there): `only_reason1` is dead at `ACID_STALL_ACCEPT` level 1 (edit-free exclusion, not
  behaviour-free); round 18's own prediction was keyed to the wrong Stage-0 column
  (`+ALPHA_IMPLICIT` instead of plain-ON). No fix attempted or needed -- affects no published
  configuration. `ACID_YADV`/`ACID_TSAT_STALL` status unchanged.
  → `YADV_RESEARCH.md` §29, `docs/YADV_ROUND_19_PLAN.md`, commit `0d0aa13`.
- Round 20: `ACID_TSAT_STALL` (F2'') **promoted to unconditional default, env var DELETED**
  (round 14's no-opt-out precedent for correctness fixes -- reason 5 is a real solver defect, not
  a research toggle). Safety proof: 7-config battery (A-G, G new = `ACID_YADV=1 ACID_NO_AJAC=1`)
  plus both `ACID_STALL_ACCEPT` levels, post-edit default byte-identical to pre-edit
  flag-forced-ON in every case. **Net improvement, not just neutral**: D/E/G (the FD-Jacobian
  configs) recover cases 27/28 from silent NaN to genuine PASS (D 12/19->13/19, E 13/19->14/19,
  G[new] 13/19->15/19 -- G's numbers byte-for-byte reproduce round 18's own §28.3 table row,
  previously misfiled as describing config D); case33 (config C) fails faster and more honestly
  (step 43 not 100/104, 0 accepts not 4-8). A/B/C/F unaffected. **BASELINE CHANGE NOTICE**:
  `scripts/yadv_r9_sweep.py`'s `EXPECTED` updated this round (old values preserved in a provenance
  comment); the case33 reproduce command at `YADV_RESEARCH.md` line ~2510 now emits empty output
  (step 100 -> 43); round 12's §22.4 case33 accept/step counts for both `ACID_STALL_ACCEPT` levels
  are superseded for the current default (case24/34 unaffected). Found and fixed a methodology
  pitfall this round, not a code bug: `denner1d_validate`/`_run`/`_dump` require `DENNER_ACID=1` in
  the environment or they silently run a non-ACID path reporting a plausible-but-wrong 11/19 --
  now recorded in `.claude/rules/denner-pitfalls.md`. `git diff --stat -- cpp/`: one file, one
  executable line changed, rest comment-only. All hard gates held (OFF 19/19, byte-identical to
  `solver_denner` published binary).
  → `YADV_RESEARCH.md` §30, `docs/YADV_ROUND_20_PLAN.md`, commit `78a1b12`.
- Round 21: designed and implemented a `(rho,e,Y)`-conserving closed-form `(p,T,alpha)`
  reconciliation (`ACID_YADV_RECON`, default OFF) targeting the alpha-remap lag (`dal_remap`)
  round 13 diagnosed. **Refutes round 13 §23.3's STATED mechanism** (direct code reading:
  `compute_R` already reconciles `T,rho` with `h` before `r_init` is ever measured, at
  `acid.cpp:1576` vs `:2022` -- Stage 0 Branch A, HREINIT flattens `r_init`'s `1/dt` growth yet the
  stall persists) while CONFIRMING its empirical finding. Delivers a reusable, unit-tested
  closed-form NASG p-T-equilibrium solver (`eos.hpp:pT_from_v_e_massfrac`, worst rel error
  `4.7e-11` vs an independent Newton) regardless of the fix's own fate -- prior art: Collis et al.
  2025 §2.3 (independent derivation, not transcribed). **Case24 gets real, mechanistically-clean
  20x progress** under plain `B+RECON` (stall step 19->399, failure re-types from vague
  retry-exhaustion to the correctly-diagnosed `T-ceiling-saturated`) -- but **cases 13/14 REGRESS
  from PASS to FAIL** (u-field quality collapse, `l2_u`/`corr_u` cross the gate). **Verdict: S5
  (harm)**, per the plan's own pre-registered stop rule (`B+RECON < 15/19`, specifically 13/19).
  Flag stays OFF, not promoted, committed as gated-off research infrastructure (round 4/8
  precedent -- a measured-regression mechanism is preserved, not deleted). A sub-prediction was
  falsified honestly (cases 26/27/28 are NOT bit-exact pure in practice -- `alpha~0.999886`, not
  `1.0` -- so the exact-skip rule doesn't exempt them; no pass/fail regression resulted there).
  `consecutive_failures` NOT incremented (mechanistically-explained negative result, round 4/8/13
  precedent). All hard gates held (OFF 19/19, flag-unset paths byte-identical, `ALL GATES OK`
  unchanged from round 20).
  → `YADV_RESEARCH.md` §31, `docs/YADV_ROUND_21_PLAN.md`, commit `19de476`.
- Round 22: implemented `ACID_YADV_RESYNC` (default OFF), the DUAL projection to round 21's
  `ACID_YADV_RECON` -- resyncs `Yv` from the current `(p,T,alpha)` each step (writes NO state
  field at all) instead of moving the state onto the Y-manifold. **Stage 0 found round 21's 13/14
  regression is not a uniform Abgrall-pressure-oscillation story**: `G+RECON` (FD Jacobian) gives a
  MIXED result -- case14 still fails (Abgrall-type, Jacobian-independent, as predicted) but case13
  now PASSES (traced to Jacobian-approximation sensitivity affecting which discrete admissible
  shock-location state a bounded Newton sweep lands near -- a narrow counterexample to
  "approximate Jacobian changes only iteration count" worth future attention, not pursued this
  round). **`ACID_YADV_RESYNC` recovers BOTH 13 and 14 on the pass/fail gate**
  (`B+RESYNC=15/19`, matching plain `B`'s fail set exactly, `dal_remap` collapses to `1.1e-16`) --
  but **case14's phase-mass drift is 16.1%** (measured via a new `ACID_RESYNC` meter), firing the
  plan's own pre-registered "1% drift -> non-promotable regardless of pass_count" rule. Case24
  gains only 2.6x (step 19->50) vs `RECON`'s 20x, despite `dal_remap` collapsing identically under
  both -- an open question whether RECON's larger case24 gain needs the state WRITE itself, not
  just `dal_remap` removal. **Verdict: gate-passing but non-promotable on an orthogonal,
  pre-registered conservation cost** -- not literally any of the round's own S1-S5 outcomes,
  recorded as its own category. Flag stays OFF, committed as gated-off research infrastructure
  (round 4/8/21 precedent). `consecutive_failures` NOT incremented. All hard gates held (OFF
  19/19, `ALL GATES OK` unchanged from round 21).
  → `YADV_RESEARCH.md` §32, `docs/YADV_ROUND_22_PLAN.md`, commit `ffcd83b`.
- Round 23: asked whether `ACID_YADV_RECON`'s 20x case24 gain (vs `ACID_YADV_RESYNC`'s 2.6x, round
  22's open question) needs the state WRITE itself or just `dal_remap` removal. New
  `ACID_PROJ_UNTIL` diagnostic caps the write to `step<N`. **The roundoff-null control (N=1)
  reproduces plain B's stall EXACTLY** (step 19, identical `rbest`/`r_init`) -- decisively RULES
  OUT trajectory-chaos as the mechanism. The state-accuracy mechanism IS real and measured: plain B
  loses ~500 of a cell's 499.58 true mass at steps 0-2 (`RMISM`'s `drho`, round 16 §26.1's
  collapse reproduced arithmetically for case24), `B+RECON` suppresses this to single digits by
  step 4. **But the dose-response is NOT monotone** (N=2 gives step 6, worse than N=1; N=10-100
  give no stall at all; N=200 gives step 501, further than N=400's own 399). **Unplanned discovery,
  checked directly**: N=50's "no stall" is `pass:false` on the actual gate -- a severely wrong
  solution (shock frozen mid-domain), not a success; "completes without STALLED" and "correct
  answer" had never been distinguished before because `B+RECON`'s own always-applied run never
  completes either. **Verdict: S4 (partial attribution)** -- trajectory-chaos excluded, the
  state-accuracy mechanism confirmed, but the practical dose-response is more complex than
  predicted (a withdrawal-point compounding effect, not characterized). The plan's own "third
  projection" secondary goal explicitly NOT attempted (its gate not met by S4).
  `consecutive_failures` NOT incremented. All hard gates held (OFF 19/19, `ALL GATES OK` unchanged
  from round 22).
  → `YADV_RESEARCH.md` §33, `docs/YADV_ROUND_23_PLAN.md`, commit `ed93f71`.
- Round 24: discovered round 23's "roundoff-null control" (`ACID_PROJ_UNTIL=1`) was a COMPLETE
  NO-OP -- the exact-skip fires on all 800 cells at step 0, so 0 writes occur and its P6' test
  carried no information about H-B (a correction to how the result was supported, not a
  retraction: the conclusion turned out right). Also corrected round 23 §33.4's "frozen shock"
  reading of `N=50`: the actual final state shows the shock COMPLETELY EXITED the domain (84%
  overstrong plateau, 32% overfast, alpha collapsed `0.5->2e-4`), not stalled mid-domain -- the
  coarse 200-step sampling in the trace obscured an over-fast, over-strong, alpha-collapsing
  shock. **Built the real roundoff-null control this round** (`ACID_RECON_NULL`, using the
  existing unit-tested `8*eps*kappa` round-trip conditioning bound, `alpha_roundtrip_floor` -- no
  new constant): non-empty (2-4 cells/step genuinely written, never zero) and applied. **Result:
  byte-identical to plain B anyway** -- H-B (Newton-trajectory chaos) excluded, this time on solid
  evidence. Confirmed no global withdrawal-point criterion exists (`ntouch=0` only at step 0, never
  again) AND the always-on family member is itself wrong (stalls on its own round 16 §26.1 blister
  at step 399) -- no withdrawal schedule within the `ACID_PROJ_UNTIL` family has a correct member
  to find. **Verdict: S1** (the question was mis-posed, exactly as pre-registered/expected) -- no
  taper designed or built. `ACID_RECON_NULL`/`alpha_roundtrip_floor` committed as inert research
  infrastructure. `consecutive_failures` NOT incremented. All hard gates held (OFF 19/19,
  `ALL GATES OK` unchanged from round 23, unit-test round-trip numbers unchanged after the
  `alpha_roundtrip_floor` refactor).
  → `YADV_RESEARCH.md` §34, `docs/YADV_ROUND_24_PLAN.md`, commit TBD.

## Setup reference

- Round-loop protocol: `.claude/skills/yadv-round/SKILL.md`
- Safety hooks: `.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`,
  wired in `.claude/settings.local.json` (gitignored, personal/experimental scope), gated by the
  presence of `.claude/round-loop-active` (also gitignored, transient).
- To start the loop: invoke `/loop` with the `yadv-round` skill as the recurring prompt (dynamic
  self-pacing, no fixed interval — the skill's own Step 0 decides whether to continue).
- To stop early by hand: `/oh-my-claudecode:cancel`-style interrupt, or just tell the loop to stop;
  it also self-stops on the conditions above.
