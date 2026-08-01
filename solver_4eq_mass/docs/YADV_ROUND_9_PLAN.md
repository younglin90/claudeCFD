# Round 9 Execution Brief — Phase 2 Stage 4 (consolidation)

Produced by Agent(subagent_type="Plan", model=opus) during round 9 of the `yadv-round` loop.
Spot-checked by the Advisor session before running: `denner1d_validate --only/--out` flags
confirmed present (`apps/denner1d_validate.cpp:24-25`), `ACID_RHIST` probe confirmed present with
the claimed print format (`acid.cpp:1821-1823`).

This round wrote **zero lines of solver code**. Every configuration below is reachable with env
vars that already existed after round 8. The round's only new file is a measurement/reporting
script, `scripts/yadv_r9_sweep.py`.

## What actually ran (the brief's plan, executed, with real numbers)

### D1 — the consolidated six-configuration sweep

A live parsing bug was found and fixed during this round: the C++ binary prints lowercase
`nan`/`-nan` (printf `%g` style) for divergent cases' metrics, which Python's `json.loads` cannot
parse (it only recognizes capitalized `NaN`/`Infinity`). The script's first run silently dropped
every NaN-carrying case from its failure-set count, producing a spurious "GATE MISMATCH" on
configs D and E even though the underlying `pass_count` (computed by the C++ binary itself, never
touched by the Python parser) was already correct. Fixed with a regex substitution
(`-?nan` -> `NaN`) before `json.loads`. After the fix, all six configurations reproduced their
expected `pass_count` AND failure set exactly:

```
[A] OFF            : 19/19  fail=[]
[B] ON             : 15/19  fail=[15, 24, 33, 34]
[C] ON+IMPLICIT    : 14/19  fail=[14, 15, 24, 33, 34]
[D] ON+IMPLICIT+FD : 12/19  fail=[14, 15, 24, 27, 28, 33, 34]
[E] OFF+FD         : 13/19  fail=[15, 24, 27, 28, 33, 34]
[F] ON+IMPLICIT+T  : 14/19  fail=[14, 15, 24, 33, 34]
ALL GATES OK
```

`--verify`: OFF path 9/9 byte-identical against the published `solver_denner` binary; case01
byte-identical between `ACID_YADV=1` and unset. This is the first time all of rounds 3-8's
recorded configurations have been reproduced from a single build in one sitting.

### D2 — wall clock (the genuinely new measurement; never done directly before)

Per-case, min of 3 repeats, `denner1d_validate --only <case>`. Full table and the both-pass-subset
summary are in `YADV_RESEARCH.md` §19.3. Headline: on the 9 cases that pass under BOTH plain
`ACID_YADV=1` and `+ALPHA_IMPLICIT`, the Stage 1+2 analytic-Jacobian path costs **7.1%** more wall
clock (`ratio=1.071`) than plain `ACID_YADV=1` — close to, though somewhat above, Phase-2 §5 risk
12's "<5%" prediction. The FD Jacobian (config D) costs **54.7%** more than the analytic path on
the identical configuration (`D/C=1.547`), broadly consistent with round 4's qualitative "~1.7-1.9x"
claim (§15.5) though measured somewhat lower here.

One case-24-specific finding worth flagging on its own: under plain `ACID_YADV=1` it aborts in
0.64s (an early divergence exit), but under `+ALPHA_IMPLICIT` it runs 32.8s before still failing
— a 51.6x cost increase for a case that fails either way. This is exactly the artifact §15.5
already warned about (a diverged run can look artificially fast), now measured directly and
excluded from the both-pass summary as intended.

### Not run this round (deferred, not blocking)

- **Sampled inner-Newton iteration counts** (`--iters`, via `ACID_RHIST`): the brief's own
  instrument, left unrun this round for time budget — it is explicitly a secondary/optional
  deliverable in the brief, and the wall-clock measurement already answers the primary cost
  question. `scripts/yadv_r9_sweep.py --iters` is ready to run in a future round if the
  iteration-count question specifically becomes relevant.
- **RH residual re-check under implicit alpha** (`scripts/yadv_rhcheck.py`): this script hardcodes
  the MAIN tree path (`W = ".../solver_4eq_mass"`, not worktree-portable), so it needs to run
  post-merge, not from this round's worktree. Deferred to the Advisor to run directly from `main`
  after this round merges, using the existing unmodified script:
  `python3 scripts/yadv_rhcheck.py` (control) vs
  `ACID_YADV_ALPHA_IMPLICIT=1 python3 scripts/yadv_rhcheck.py` (Y+implicit rows) — both trivial,
  zero new code, exactly as the brief specified.

## Promotion decision (adopted from the brief's analysis, verified against the real numbers above)

Table 3 (B->C delta) confirms the brief's central claim precisely: **case14 is the only case that
flips from PASS (plain `ACID_YADV=1`) to FAIL (`+ALPHA_IMPLICIT`)** — `corr_u` 0.982->0.954 (not
directly in this round's table but consistent with round 6's recorded value), `l2_p` 0.0139->0.0145
(small), the FAIL verdict driven by `corr_u`/`l2_u`/other case14-specific criteria not `l2_p`
alone. Every other case that changes between B and C either stays PASS on both sides (05, 07, 13,
25, 26, 27, 28 — metrics move slightly, verdict doesn't) or stays FAIL on both sides (15, 24, 33,
34 — case15 improves dramatically in magnitude, still fails its own narrower gate per round 7).

Per the re-scoped bar (§19.5 of the research doc, restated below), criteria (i) `pass_count >=
15/19` and (ii) `no case that plain ACID_YADV=1 passes newly fails` both fail, on the same case
(14). **Decision, adopted: do not fold `ACID_YADV_ALPHA_IMPLICIT` into `ACID_YADV`.** It remains a
separate, default-OFF opt-in flag layered on `ACID_YADV=1`.

## Roadmap recommendation — adopted

The brief's `done: true` recommendation is adopted (see `YADV_ROADMAP.md`'s updated "Current
goal" section). Reasoning affirmed by the Advisor: the re-scoped goal (`pass_count >= 14` under
`+ALPHA_IMPLICIT`, cases 13/25 durably recovered) was met at round 6 and has now been re-verified
three more times (rounds 7, 8, 9) with zero drift; both remaining open items (24/33/34's
conservation defect, case15's central-jump defect) are different defect *classes* requiring their
own fresh investigations, not continuations of this plan; and the subject of the whole
investigation (`ACID_YADV`) stays a permanently-OFF research path regardless of outcome, so further
autonomous compute here has declining marginal value without a fresh human-set priority.

**Phase 2 (`docs/YADV_PHASE2_PLAN.md`, Stages 0-4) is complete as of this round.**
