# YADV Round 11 Plan — Phase 3a Stages 1 + 2 (diagnostics only)

Planner output (Agent, subagent_type="Plan", model=opus), round 11. Scope: `docs/YADV_PHASE3_PLAN.md`
Stage 1 ("make the silent stall audible") and Stage 2 (`ACID_TEND_SCALE` observation-window knob).
No fix, no `diverged` change, no `cases.cpp`/`validation.cpp` edit.

Advisor spot-check (round 11 session, before implementation): independently re-`grep`'d every
line-number claim below directly against `cpp/denner_1d/src/acid.cpp`/`cases.cpp`/`solver.cpp` in
this worktree. All confirmed exact: `thinc_dbg`=583, `diverged` declared=614, `while` loop=615,
`dt=min(...)`=653, `stepped`=661, retry loop=662, `bad=`=2072, `dt*=0.5`=2103, retry-loop
close=2104, stall break=2105. `solve_case_acid(const CaseDefinition& c)` (const ref, confirmed) in
`acid.hpp:19`. `cases.cpp:760` builds the reference at unscaled `c.config.final_time`, confirmed.
`solver.cpp:1025/1031` is the legacy non-ACID path (out of scope), confirmed.

---

## 0. Verification of the incoming claims (done first, against the actual files)

**Confirmed, line numbers still exact (working tree is clean, `worktree-yadv-round-11`, no local edits):**

| claim | actual |
|---|---|
| stall site | `acid.cpp:2105` -- `if (!stepped) break;  // could not advance even at the smallest dt` |
| `diverged` declared | `acid.cpp:614` |
| `diverged` set (the *other*, CFL-collapse path) | `acid.cpp:649`, message at 646-648 |
| `diverged` consumed (NaN fill) | `acid.cpp:2135-2140` |
| env-var pattern to copy | `acid.cpp:550` (`yadv`), `565` (`alpha_implicit`), `582` (`alpha_implicit_t`), `583` (`thinc_dbg`) |
| `dbg` flag | `acid.cpp:373`; per-step trace at `2123-2128` (`ACID step %d t=...`), retry trace at `2099-2102` |
| Stage 1 must NOT set `diverged` | `YADV_PHASE3_PLAN.md:132-133` -- "Do **not** set `diverged` yet ... deferred to Stage 3c". Confirmed; Stage 3c at line 168-171 requires an explicit Advisor decision. |
| Stage 1's extra detail is `ACID_DBG`-gated | `YADV_PHASE3_PLAN.md:134` -- "Under the existing `ACID_DBG` flag only, also report which retry-loop condition triggered the stall and the first offending cell." The main `STALLED:` line is unconditional (line 130). |
| Stage 2 knob | `YADV_PHASE3_PLAN.md:141-155`, exactly as described. |

**Newly established facts the plan text does not contain (found by reading the code, load-bearing for the diff):**

1. `solve_case_acid` takes `const CaseDefinition& c` (`acid.hpp:19`, `acid.cpp:366`). `c.config.final_time`
   cannot be assigned. Stage 2 must introduce a local `t_end` and change its two consumers -- exactly
   two exist in `acid.cpp`: line 615 (`while` condition) and line 653 (`dt = std::min(...)`).
   `solver.cpp:1025/1031` is the legacy non-ACID solver -- out of scope.
2. The reference solution is built at the unscaled `final_time` (`cases.cpp:760`), and
   `denner1d_dump` calls `reference_state(c)` independently. Consequence: with `ACID_TEND_SCALE != 1`
   the dump's `*_ref` columns and every `validate` metric are meaningless. Acceptable only because
   the RH scripts read solver columns 2/3/4 only. Must be stated in the code comment: never run a
   gate/validate with `ACID_TEND_SCALE` set.
3. Several existing shell scripts merge stderr into the captured dump (`yadv_off.sh`, `yadv_ab.sh`,
   `yadv_baseline.sh` use `> file 2>&1`). Under OFF nothing stalls today, but any *ON* dump captured
   this way will now gain a `STALLED:` line inside the CSV. This round's captures must use separate
   streams (`> out 2> err`).
4. `yadv_off.sh`/`yadv_ab.sh`/`yadv_final.sh`/`yadv_baseline.sh` hardcode the *main* worktree path --
   running them from the round-11 worktree silently tests the wrong tree. Use explicit commands.
5. `max_steps=20000` uniform; case33/plain's "last printed step 2000" (sect.20.1) is a print-cadence
   artifact (`dbg` prints every 200 steps), not a truncation -- the run continues to `t_end`. This
   exposes a fragility in `yadv_rh2.py`: its completion fraction is derived from the last *printed*
   step, so a fully completed run can read `frac=0.955`, only 0.055 above the 0.9 NULL-RUN threshold.
   Stage 2 needs a companion script that derives `t_end` from a new unconditional-under-`dbg` "ACID
   done" line instead of a hardcoded dict.

**Contradiction found, corrected this round:** `docs/YADV_ROADMAP.md`'s `next_task` said Stage 1
includes "setting `diverged=true`" -- this contradicts `YADV_PHASE3_PLAN.md:132-133` (Stage 1 must
NOT set it; that's Stage 3c, requiring an explicit Advisor decision). `YADV_PHASE3_PLAN.md` is
authoritative. Roadmap `next_task` wording corrected in Step 10 of this round.

No error found in the plan's success/failure/revert criteria; confirmed as written, with the
refinements below (tighten, not override).

---

## 1. Literature / prior-art check -- verdict: no new literature needed

Searched (web + Semantic Scholar) and dedup-checked the repo's existing `papers/` summaries.

**(a) Retry-exhaustion / stall diagnostics in implicit solvers.** Standard engineering practice, not
a research contribution -- the dt-cutback-on-nonlinear-failure loop with a minimum-dt floor and an
explicit failure report is the textbook pattern in production implicit codes (e.g. MOOSE's
"Troubleshooting Failed Solves" documents exactly this). The one gap `acid.cpp` currently has vs.
that convention is reporting the exhaustion as a failure rather than returning the last state
silently -- there is no paper to cite for "print a message when your retry loop gives up".

**(b) "Early-stop / windowed observation" before a wave exits the domain.** Also standard, with no
distinct name or canonical citation -- the universal convention in Riemann/shock-tube verification
(Sod 1978 / Toro) is to choose `t_end` so no wave reaches a boundary, avoiding reflection
contamination. No named methodology exists to cite.

**Verdict: no new papers, no "Papers needed" entries.** Both halves of this round are standard
diagnostic engineering. The honest, citation-free justification for Stage 2 (recorded in code and
in sect.21): the observation-window knob restores the standard shock-tube verification convention
that `cases.cpp`'s fixed `t_end = 0.7/Vs_ref` breaks whenever the computed `Vs` differs from
`Vs_ref`.

---

## 2. Round structuring decision: ONE round (round 11), TWO commits

Do both stages this round, landed as two separate commits, Stage 1 first, each with its own gate run.

- Risk is already isolated by construction: disjoint code regions, disjoint independently-unset env
  vars. Neither can mask the other's regression.
- Two commits give full bisection granularity at zero extra round cost.
- Stage 2 without Stage 1 is measurably weaker (can't distinguish "hit scaled t_end" from "stalled
  early" -- exactly the round-3/9 confusion); Stage 1 without Stage 2 produces no new physics.
- Combined wall-clock cost (~20-30 min) is well within one round.

Ordering is strict: Stage 1 committed and gated before Stage 2 is written. If Stage 1's gate fails
(any OFF-path `STALLED`), Stage 2 does not happen this round -- stop, re-plan.

---

## 3. Stage 1 -- make the silent stall audible

### 3.1 Anchor (current code, `cpp/denner_1d/src/acid.cpp`)

```cpp
661:        bool stepped = false;
662:        for (int retry = 0; retry < 14; ++retry) {
```
```cpp
2072:        bool bad = (ajac && coupled && !conv_inner && rbest >= r_init);
2073:        for (int i = 0; i < n; ++i)
2074:            if (!std::isfinite(s.p[i]) || !std::isfinite(s.u[i]) ||
2075:                std::abs(s.u[i]) > 10.0 * uref) { bad = true; break; }
2076:        if (!bad) {
...
2099:        if (dbg) {
2100:            double mxu = 0; for (int i = 0; i < n; ++i) mxu = std::max(mxu, std::abs(s.u[i]));
2101:            std::fprintf(stderr, "RETRY %d dt=%.3e -> max|u|=%.3e (uref=%.2e)\n", retry, dt, mxu, uref);
2102:        }
2103:        dt *= 0.5;
2104:        }  // retry loop
2105:        if (!stepped) break;  // could not advance even at the smallest dt
```

### 3.2 Edit A -- carry the stall reason out of the retry loop

Immediately after line 661 (`bool stepped = false;`), add declarations in the same scope:

```cpp
        // Stage 1 (round 11, DIAGNOSTIC ONLY): carry the last retry's failure reason out of the
        // retry loop so the stall report below can name it. Ints/doubles only -- no FP arithmetic
        // is added, so every accepted step is bit-identical to the pre-change build.
        int  stall_reason = 0;    // 1=Newton made no progress, 2=non-finite p, 3=non-finite u, 4=|u|>10*uref
        int  stall_cell   = -1;   // first offending cell for reasons 2-4
        double stall_dt   = 0.0;  // the last dt actually attempted (dt is halved after the check)
        int  stall_retry  = -1;
```

At 2072-2075, record without changing control flow:

```cpp
        bool bad = (ajac && coupled && !conv_inner && rbest >= r_init);
        if (bad) { stall_reason = 1; stall_cell = -1; }
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(s.p[i]) || !std::isfinite(s.u[i]) ||
                std::abs(s.u[i]) > 10.0 * uref) {
                bad = true;
                stall_reason = !std::isfinite(s.p[i]) ? 2 : (!std::isfinite(s.u[i]) ? 3 : 4);
                stall_cell = i;
                break;
            }
```

Immediately before line 2103 (`dt *= 0.5;`), record the attempted dt:

```cpp
        stall_dt = dt; stall_retry = retry;
        dt *= 0.5;
```

`stall_dt` prints the *last attempted* dt (at the break site `dt` has already been halved past the
last attempt). `stall_reason=1`'s assignment may be overwritten by 2/3/4 if a cell is also bad --
correct precedence (concrete non-finite cell more informative than "no progress"). Control flow
unchanged.

### 3.3 Edit B -- the unconditional report at line 2105

Replace line 2105 with:

```cpp
        if (!stepped) {
            // Round 11 Stage 1 (docs/YADV_PHASE3_PLAN.md): the retry loop exhausting all 14
            // dt halvings used to exit SILENTLY -- no message, and (deliberately, still) no
            // `diverged = true`, so solve_case_acid returns a FINITE partial state that
            // validate/dump score as a normal completed run. That silence is what made
            // YADV_RESEARCH.md sect.14.3/19.4's RH results measure a pristine initial condition
            // for two rounds (sect.20, retracted). This message is stderr-only and unconditional;
            // stdout and every validate metric are unchanged. Whether the state SHOULD be marked
            // diverged here is Phase 3a Stage 3c -- an explicit Advisor decision, not this change.
            static const char* const why[] = {"unknown", "newton-no-progress", "nonfinite-p",
                                              "nonfinite-u", "u>10*uref"};
            std::fprintf(stderr,
                "STALLED: case=%s no admissible step at dt=%.3e after %d retries, step %d, "
                "t=%.3e of %.3e -> stop (state returned as-is, NOT marked diverged)\n",
                c.id.c_str(), stall_dt, stall_retry + 1, step, t, t_end);
            if (dbg)
                std::fprintf(stderr,
                    "STALLED-DETAIL: reason=%s cell=%d x=%.5f p=%.4e u=%.4e rho=%.4e alpha=%.5f "
                    "T=%.4e (conv_inner=%d rbest=%.4e r_init=%.4e uref=%.3e)\n",
                    why[stall_reason], stall_cell,
                    stall_cell >= 0 ? st.x[stall_cell] : -1.0,
                    stall_cell >= 0 ? s.p[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.u[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.rho[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.alpha[stall_cell] : 0.0,
                    stall_cell >= 0 ? s.T[stall_cell] : 0.0,
                    (int)conv_inner, rbest, r_init, uref);
            break;  // could not advance even at the smallest dt
        }
```

Deviations from the plan's literal string (recorded, both accepted for this round):
1. `case=%s` added -- `denner1d_validate` runs 19 cases into one stderr stream; without the id a
   `STALLED` line is unattributable. Mirrors the existing `THINC case=%s ...` line at 2132.
2. `-> abort` changed to `-> stop (state returned as-is, NOT marked diverged)` -- `-> abort` would
   imply the run is treated as a failure, the exact false impression that caused the retraction.
3. `stall_retry + 1` so "after N retries" counts attempts, not the zero-based index.
4. Prints `t_end` (Stage 2's effective stop time) rather than `c.config.final_time`, so the message
   stays correct once Stage 2 lands in the same round; the `TEND_SCALE:` banner (sect.4.2) discloses
   any scaling.

`grep -n "STALLED" cpp/denner_1d/src/*.cpp` returns nothing before the edit (checked) -- no
colliding string.

### 3.4 Stage 1b (MAXSTEPS silent exit) -- deferred, not implemented this round

The `while` loop's `step < c.config.max_steps` bound is a third silent-incomplete exit. Verified NOT
currently firing for any case in {24,33,34} (sect.0 item 5). Not needed to close this round's bug;
skipped to keep the diff minimal. Revisit only if a future scale sweep ever approaches `max_steps`.

### 3.5 Additional `ACID_DBG` line -- implemented, needed by Stage 2's script

Added, right after the loop closes (unconditional-when-`dbg`, zero effect when `ACID_DBG` unset):

```cpp
    if (dbg)
        std::fprintf(stderr, "ACID done case=%s step=%d t=%.9e of %.9e\n",
                     c.id.c_str(), step, t, t_end);
```

Needed because `yadv_rh2.py`'s completion guard infers `t_last` from the last *printed* `ACID step`
line (every 200 steps) -- imprecise, and worse under Stage 2 scaling. This line gives the exact end
state and the effective `t_end`, letting the new script compute completion robustly and
scale-agnostically. Falls under the plan's "under `ACID_DBG` only, also report..." allowance.

---

## 4. Stage 2 -- `ACID_TEND_SCALE`, a diagnostic observation window

### 4.1 Edit C -- the env read

Inserted after line 583 (`thinc_dbg`), following the value-returning-lambda idiom already used
elsewhere in this file for similar env knobs:

```cpp
    // ACID_TEND_SCALE (round 11, Phase 3a Stage 2, DIAGNOSTIC ONLY, default 1.0 = byte-identical
    // when unset): multiplies THIS SOLVER's stop time only. It is an OBSERVATION WINDOW, not a
    // physical or tuning parameter -- the standard shock-tube verification convention is to sample
    // before a wave reaches a boundary, and cases.cpp's fixed t_end = 0.7/Vs_ref breaks that
    // convention whenever the COMPUTED shock speed differs from Vs_ref (cases 24/34 under
    // ACID_YADV_ALPHA_IMPLICIT: the shock has left the 800-cell domain by t_end, so there is no
    // clean post-shock plateau to sample -- YADV_RESEARCH.md sect.20.2).
    // WARNING, by design and not fixable here: cases.cpp builds the reference solution at the
    // UNSCALED c.config.final_time (cases.cpp:760) and denner1d_dump calls it independently, so
    // with scale != 1 the dump's *_ref columns and EVERY denner1d_validate metric are meaningless.
    // NEVER set this for a gate/validation run. Only the solver columns (p,u,rho) are valid.
    const double tend_scale = []{
        const char* e = std::getenv("ACID_TEND_SCALE");
        if (!e) return 1.0;
        const double v = std::atof(e);
        if (!(v > 0.0) || !std::isfinite(v)) {
            std::fprintf(stderr, "ACID_TEND_SCALE=%s invalid (need finite > 0) -> ignored, using 1.0\n", e);
            return 1.0;
        }
        return v;
    }();
```

### 4.2 Edit D -- the two consumers

Immediately before line 614 (`bool diverged = false;`):

```cpp
    // Stage 2: the effective stop time. The `== 1.0` early-out makes the unset path textually
    // identical to the pre-change code (multiplying by 1.0 is exact in IEEE-754 anyway, but this
    // makes byte-identity inspectable by reading rather than by FP reasoning).
    const double t_end = (tend_scale == 1.0) ? c.config.final_time
                                             : c.config.final_time * tend_scale;
    if (tend_scale != 1.0)
        std::fprintf(stderr, "TEND_SCALE: case=%s scale=%.6g -> t_end=%.9e (reference is still at "
                     "%.9e -- *_ref columns and all validate metrics are INVALID for this run)\n",
                     c.id.c_str(), tend_scale, t_end, c.config.final_time);
```

Then: line 615 `while (t < c.config.final_time && ...)` -> `while (t < t_end && ...)`; line 653
`dt = std::min(dt, c.config.final_time - t);` -> `dt = std::min(dt, t_end - t);`.

The `TEND_SCALE:` banner is unconditional when the var is set (never printed when unset). Not
`dbg`-gated -- exists for the same reason this round exists: a scaled run's dump would otherwise
look indistinguishable from a full-`t_end` dump.

### 4.3 Required companion script (new file, does not touch `yadv_rh2.py`)

`scripts/yadv_rh2.py`'s hardcoded `FINAL_TIME` dict would misclassify any scale < 0.9 as `NULL RUN`.
Do not edit it (round 10's published instrument; sect.20.3's numbers must stay reproducible). New
file `scripts/yadv_r11_window.py` (`__file__`-derived root):
- sets `ACID_TEND_SCALE` in the child env; derives effective `t_end` from the solver's own
  `ACID done ... of %e` line (sect.3.5) instead of a hardcoded dict;
- treats a `STALLED:` line in stderr as an automatic `NULL RUN` (stronger/earlier than the IC-match
  heuristic), records `reason`/`cell` from `STALLED-DETAIL`;
- reuses `yadv_rh2.py`'s `preshock_state`/`rh_residual` logic (copied, not imported);
- `capture_output=True` (separate streams) always.

### 4.4 Measurements this stage produces

**(a) Front-position-vs-window sweep (primary result).** For cases 24 and 34 under
`+ALPHA_IMPLICIT`, sweep `ACID_TEND_SCALE in {0.3,0.4,0.5,0.6,0.7,0.85,1.0}`; per scale locate the
front by `argmax|dp/dx|`, sample a plateau window derived from `x_front` (e.g.
`[x_front+0.05, x_front+0.20]`), not the fixed `[0.3,0.6]` box sect.20.2 flagged. Fit `Vs_front` from
`x_front` vs `scale*t_end` (assumption-free shock speed, no RH inference) and compare to `Vs(mass)`
from the jump -- this adjudicates the plan's static 1.40-1.49x prediction vs sect.20.2's measured
0.80-0.85x, which cannot both be right.

**(b) Stall-bracketing sweep.** For 24/plain, 34/plain, 33/+IMPLICIT (stalls at
`t/t_end ~ 0.0027/0.0045/0.0022`), sweep `ACID_TEND_SCALE in {0.0002,0.0005,0.001,0.0015,0.002,0.0025,0.003}`.
With Stage 1's messages, each run's stderr says unambiguously whether it hit its scaled `t_end`
cleanly or stalled (+reason+cell). The last clean scale before the first `STALLED` brackets the
failure step.

**(c) Control.** 33/plain at scale 1.0 through the new script must reproduce sect.20.2/20.3's
`+8.808e-01` / `Vs/Vs_ref=0.5355`. If the front-derived window changes that number, the *window* is
what changed and must be reported as such.

---

## 5. Gates, success / failure / revert criteria

### 5.1 Baseline capture -- before any edit (mandatory sequencing)

Build at clean `HEAD` first (no `build-cpp/` in this fresh worktree), capture all 19 case dumps with
separated stdout/stderr, before making any edit -- this is the byte-identity reference for Stage 1's
gate.

### 5.2 The five standing hard gates (run after each commit)

| # | command | required |
|---|---|---|
| 1 | `denner1d_unit` | 9/9 (script-internal count; overall "denner1d_unit ok") |
| 2 | `DENNER_ACID=1 denner1d_validate` | pass_count=19/19 |
| 3 | `+ACID_YADV=1` | pass_count=15/19 |
| 4 | `+ACID_YADV_ALPHA_IMPLICIT=1` | pass_count=14/19 |
| 5 | `+ACID_NO_AJAC=1` (FD-invariance) | pass_count=12/19, failure set exactly {14,15,24,27,28,33,34} |

None with `ACID_TEND_SCALE` set. OFF stdout byte-identical to pre-edit capture; all 19 dumps
byte-identical (`cmp -s`).

### 5.3 Stage 1 criteria

Success: all five gates hold; every OFF dump byte-identical; zero `STALLED` on OFF; exactly cases
24(plain)/34(plain)/33(+IMPLICIT) print one `STALLED:` each, correctly attributed; plain-ON and OFF
stdout both byte-identical to pre-edit captures (Stage 1 must not perturb any metric, not just OFF).
Failure: any OFF-path `STALLED` -- stop, do not start Stage 2, re-plan (bigger finding than Phase
3a). Revert trigger: any stdout/metric difference anywhere.

### 5.4 Stage 2 criteria

Success: (a) clean front-derived-window RH residual for 24/34 (+IMPLICIT) with shock verified inside
the domain; (b) void-cell formation step bracketed to within one step for >=1 stalling config; (c)
33/plain control reproduces `+8.808e-01`. Failure (still valid): clean residual not materially
different from sect.20.3's `+5.03e-01`/`+4.07e-01` -- contamination ruled out, record and continue.
If `Vs_front` disagrees with `Vs(mass)` by >~10%, report the disagreement itself rather than picking
the more convenient number. Revert trigger: `ACID_TEND_SCALE` unset changes any metric anywhere.

### 5.5 What gets written

`YADV_RESEARCH.md` sect.21 (new); `YADV_ROADMAP.md` round_counter=11 + `next_task` correction (Stage
1 does NOT set `diverged`; that's Stage 3c, needs Advisor approval); this plan saved as
`docs/YADV_ROUND_11_PLAN.md`.

---

## 6. Risk list (abbreviated -- full table in Planner's response, preserved in commit history)

1. stderr merged into a CSV by old shell scripts -- avoided by using separate streams this round.
2. Wrong tree tested (hardcoded main-worktree paths in old scripts) -- avoided by not running them.
3. Stage 2 env var left set during a gate -- detected by the unconditional `TEND_SCALE:` banner.
4. Stage 2 misclassified as NULL RUN by a hardcoded FINAL_TIME dict -- avoided via sect.4.3's
   `ACID done` line.
5. Compiler reordering changing OFF bit-exactness -- guarded by all-case `cmp -s`; no new FP math.
6. `stall_reason` reflecting the wrong retry -- cross-check against existing `RETRY` trace lines.
7. Front-window sampling re-catching the contact instead of the shock -- cross-check `argmax|dp/dx|`
   against the `Vs_front` linear fit's residual/R^2.
8. Scale sweep not actually sampling one consistent trajectory -- verify `ACID step` traces agree
   across two scales up to the shared clamp point.
9. `STALLED` appearing on an unexpected case -- report in sect.21 regardless of gate impact.
10. Stale roadmap `next_task` acted on (Stage 3c without approval) -- guarded by the plain-ON
    byte-identity gate; text corrected this round.

---

## 7. Non-goals (explicit, this round)

No Stage 3, no Stage 3c (`diverged=true` NOT implemented or scaffolded -- needs an explicit Advisor
decision, not this round's). No Stage 3a/3b. No `cases.cpp`/`validation.cpp` edit. No new tuning
constant/per-case coefficient (`ACID_TEND_SCALE` is diagnostic-only, same category as `ACID_DBG`).
No edit to `yadv_rhcheck.py`/`yadv_rh2.py`. No `solver.cpp` change. No promotion decision on
`ACID_YADV` (stays default OFF, 15/19). No `git push`.

## 8. Papers needed

None -- see sect.1.
