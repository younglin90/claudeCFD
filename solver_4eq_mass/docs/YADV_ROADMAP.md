# YADV Round-Loop Roadmap

State file the `yadv-round` skill (`.claude/skills/yadv-round/SKILL.md`) reads/writes every round.
Detailed research log stays in `docs/YADV_RESEARCH.md`; this file is a thin, compact control
document — round counter, stop conditions, next-task pointer, one-line-per-round history. Do not
put derivations or measurement tables here; put them in `YADV_RESEARCH.md` (or a per-round plan
doc) and link to them.

## Current goal

Complete Phase 2 (`docs/YADV_PHASE2_PLAN.md`, Stages 0-4): add analytic `d(alpha)/dp`,
`d(alpha)/dT` Jacobian contributions so `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` converges under
the default analytic Jacobian at least as well as round 4's FD-Jacobian result (recover
13/15/25, keep everything else, `pass_count >= 15`). When Phase 2's Stage 4 (consolidation) is
done, the round that completes it must explicitly re-evaluate this "Current goal" section: either
mark `done: true` (if the loop should idle / hand back to a human) or replace it with the next
concrete goal (e.g. the cases-24/33/34 conservation defect, §11.6/§15.5 of YADV_RESEARCH.md) and
reset `consecutive_failures`.

## Control state

```
round_counter: 4
consecutive_failures: 0
done: false
next_task: docs/YADV_PHASE2_PLAN.md Stage 0
```

(Round counter starts at 4 because rounds 1-4 of the `ACID_YADV` experiment were already run
manually, before this loop existed — see `docs/YADV_RESEARCH.md`. Round 5 onward runs under
`yadv-round`.)

## Stop conditions (checked at the start of every round, before any work happens)

1. `done == true`
2. `consecutive_failures >= 3` — three rounds in a row with no measured progress or a hard-gate
   failure. This means the current approach needs human reconsideration, not another autonomous
   attempt at the same thing.
3. `round_counter >= 100000` (nominal cap; `consecutive_failures` is the real backstop)

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
- Round 5+: not yet run. `next_task` above.

## Setup reference

- Round-loop protocol: `.claude/skills/yadv-round/SKILL.md`
- Safety hooks: `.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`,
  wired in `.claude/settings.local.json` (gitignored, personal/experimental scope), gated by the
  presence of `.claude/round-loop-active` (also gitignored, transient).
- To start the loop: invoke `/loop` with the `yadv-round` skill as the recurring prompt (dynamic
  self-pacing, no fixed interval — the skill's own Step 0 decides whether to continue).
- To stop early by hand: `/oh-my-claudecode:cancel`-style interrupt, or just tell the loop to stop;
  it also self-stops on the conditions above.
