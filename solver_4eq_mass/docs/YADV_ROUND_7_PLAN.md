# Round 7 Execution Brief — Phase 2 Stage 2

Produced by Agent(subagent_type="Plan", model=opus) during round 7 of the `yadv-round` loop.
Spot-checked by the Advisor session before implementing: insertion point (line 1586, immediately
before the upwind-transport comment), `alp_p`/J1-loop line anchors, the `theta[]` BC-ordering
argument (line 1164 runs before the mdot loop at 1174), and the case15-gate correction (`grep` of
`validation.cpp` confirms `peak_delta_u` appears only in the generic `default_pass`/`metrics_json`,
never in case15's own gate at lines ~684-730) — all confirmed correct before implementing.

## What was implemented

Added a new, purely additive diagonal loop to the analytic Jacobian (`acid.cpp`, immediately after
the existing "flux coupling (frozen transport)" block, before "upwind-TRANSPORT derivatives"):
the OTHER product-rule addend of the ACID per-cell flux blend `mdot_f^(i) = (al_i*raup_f +
(1-al_i)*rbup_f) * theta_f` — the sensitivity of the blend weight `al_i` itself, reusing Stage 1's
already-computed `alp_p[]` (no new derivative math):

```cpp
if (yadv && alpha_implicit) {
    for (int i = 0; i < n; ++i) {
        const double ap = alp_p[i];
        const double dR = (raup[i+1] - rbup[i+1]) * theta[i+1];
        const double dL = (raup[i]   - rbup[i]  ) * theta[i];
        const double eR = (rHaup[i+1] - rHbup[i+1]) * theta[i+1];
        const double eL = (rHaup[i]  - rHbup[i]  ) * theta[i];
        add(i, i, 1, 1, (dR - dL) * ap);
        add(i, i, 0, 1, (dR*uconv[i+1] - dL*uconv[i]) * ap);
        add(i, i, 2, 1, (eR - eL) * ap);
    }
}
```

Boundary correctness verified rather than assumed: `theta[]` already carries every BC override
(inlet `uin`, reflective zeroing) before the mdot loop runs, and the `inlet_left` restatement of
`mdotL[0]` evaluates to exactly the same product `theta[0]` already gives, so no face-0/face-n
special case was needed.

## Gates (all held, identical to rounds 4-6)

OFF 19/19 + 9/9 byte-identical. Plain `ACID_YADV=1` 15/19. FD-invariance (`ACID_NO_AJAC=1`) 12/19,
exact same failure set {14,15,24,27,28,33,34}.

## The target measurement — measured no-op on pass_count, one precise new finding

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1`: **14/19, identical failure set {14,15,24,33,34} to
round 6.** case15's metrics moved by noise only (`amp_ratio_p` 1.00042→1.00041, `corr_p` unchanged
at 0.999285) — consistent with the brief's own prediction that J2's magnitude is much smaller than
Stage 1's ~500x fix, since 13/25 were already recovered and had nothing left for J2 to add. This is
reported as a measured negative/no-op, not spun as a win.

**The one real result this round produced**: round 6's roadmap/research-log entries claimed case15
was "close" to passing based on `peak_delta_u` moving 321→0. **That was the wrong metric** —
`peak_delta_u` is not part of case15's gate at all (`validation.cpp` lines ~684-730 use a
domain-restricted jump/concentration test `smooth_ok` and a total-variation-excess test `osc_ok`,
neither of which is `peak_delta_u`). Computed exactly from a fresh dump (transcribing
`validation.cpp`'s formulas verbatim):

```
cj=30.02   cj_r=3.55   threshold=max(8.0, 1.10*cj_r)=8.0     -> FAILS (central jump ~4x the limit)
mj=32.00   mj_r=18.08   threshold=max(8.0, 1.10*mj_r)=19.88   -> FAILS
cc=0.117   cc_r=0.084   threshold=max(0.04, 1.10*cc_r)=0.093  -> FAILS
smooth_ok = False
p_osc=0.0, r_osc=0.0  -> osc_ok = True  (the TV-excess side is completely clean)
```

**case15's actual blocker is a large central velocity jump** (`cj`/`mj`/`cc` all fail by a wide
margin at the domain's symmetry point x=0.5), not general oscillation. This is a genuinely
different failure mode from what any p-pathway alpha-Jacobian work (Stage 1, Stage 2, or the
contingent Stage 3 T-pathway) targets, and there is no structural reason to expect Stage 3 to close
it either -- it looks like a collocated-solver central-symmetry artifact (case15 is a symmetric
double rarefaction; `u=0` at the exact center by symmetry, and a stagnation-point spike in a
coupled pressure-velocity scheme is a known failure mode class), which is a scheme/discretization
question, not a Jacobian-accuracy one.

## Verdict

1. Stage 2 is implemented, correct (all gates held, no double-counting per the `ajblk`-style
   reasoning in the brief), and a measured no-op on `pass_count` -- an honest negative result.
2. The real deliverable is the corrected diagnosis of case15: its blocker is a central-jump defect,
   unrelated to the alpha-implicit Jacobian work this whole Phase-2 plan targets. Recovering
   case15 is very likely **out of scope** for Stages 3 (T-pathway) as currently planned.
3. Cases 24/33/34, 14 unmoved, exactly as predicted (separate defects, non-goals for this stage).
4. Recommendation for the next round's Planner: either (a) attempt Stage 3 (T-pathway) anyway,
   since it is the only remaining item in the current Phase-2 plan and might still help case14
   (whose `hsT<0` lead is genuinely T-related), while explicitly NOT expecting it to move case15;
   or (b) treat Phase-2's "recover 13/15/25" goal as partially met (13/25 done, 15 understood but
   out of scope) and have the roadmap's "Current goal" section explicitly re-scoped, per its own
   instructions, rather than keep chasing case15 with Jacobian work that cannot reach its actual
   defect.

## Housekeeping note (not acted on this round -- out of Stage 2's scope)

`git ls-files` shows two stray tracked files named `3,` and `=150` at the repo root, committed at
`325dc5b` (the original fork commit, long before this round loop existed) -- almost certainly a
shell-redirection accident from an earlier session, not from any round's work. Left alone this
round to keep the commit scoped to Stage 2 only; flagged for the user or a future round to clean up
if wanted.
