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
round_counter: 13
consecutive_failures: 0
done: false
next_task: Round 13 confirmed the 1/dt mechanism (REMAP term in alpha recovery) to textbook
           clarity but REFUTED the naive fix (ACID_YADV_HREINIT, resetting s.h alone -- S4,
           makes case34 worse and needs MORE accepts under STALL_ACCEPT, not fewer). Round
           13 sect.23.3's own diagnosis of why: s.rho/s.T stay stale (s0-consistent) until
           the first compute_R() re-derives them from the corrected h, so Newton still isn't
           handed a genuinely self-consistent starting point. Two open threads for round 14:
           (a) a MORE complete consistency fix that reconciles (T,rho) simultaneously with h
           at the new alpha, before Newton's it==0 -- not the same as round 13's single-field
           reinit; needs its own careful design, don't just retry sect.3's approach with more
           fields bolted on without re-deriving what "simultaneous" requires; or (b) abandon
           the fix-the-mismatch angle and accept ACID_STALL_ACCEPT=1 (round 12, still the
           only working path, its sect.22.5 caveat still standing) as the practical answer,
           pivoting to case33's still-unsolved sustained difficulty (round 12 sect.22.4,
           qualitatively different, 221 accepts at budget 20, still only 14.8% of t_end) or
           to Stage 3c (diverged=true at the stall site, still needs explicit Advisor
           decision, docs/YADV_ROUND_12_PLAN.md sect.5).
           Grounded in YADV_RESEARCH.md sect.23, docs/YADV_ROUND_13_PLAN.md.
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

## Setup reference

- Round-loop protocol: `.claude/skills/yadv-round/SKILL.md`
- Safety hooks: `.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`,
  wired in `.claude/settings.local.json` (gitignored, personal/experimental scope), gated by the
  presence of `.claude/round-loop-active` (also gitignored, transient).
- To start the loop: invoke `/loop` with the `yadv-round` skill as the recurring prompt (dynamic
  self-pacing, no fixed interval — the skill's own Step 0 decides whether to continue).
- To stop early by hand: `/oh-my-claudecode:cancel`-style interrupt, or just tell the loop to stop;
  it also self-stops on the conditions above.
