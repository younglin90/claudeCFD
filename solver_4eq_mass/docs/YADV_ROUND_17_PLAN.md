# YADV Round 17 Plan — is F2 (`T_from_hstat` reporting saturation) actually risky?

Planner output (Agent, subagent_type="Plan", model=opus), round 17. Diagnostic round. One new
stderr-only env var (`ACID_TSAT`), default OFF.

Advisor spot-check: confirmed C1 call site `if (T_from_hstat(...)) s.T[i] = Tnew;` at
`acid.cpp:1216`; `coupled = std::getenv("ACID_COUPLED")!=nullptr || c.config.coupled;` at line 379
is OVERRIDDEN at line 429 by `coupled = unic ? true : (p_ratio > 10.0);`, and `c.unic = true` is set
unconditionally in `base_config` (`cases.cpp:28`) with no per-case override found -- confirming
`coupled==true` for all 19 graded cases under the published default, exactly as the plan claims.
Confirmed case29's exclusion (`cases.cpp:569,591`) and all four `compute_R(); // restore` sites
(`:1592, :1685, :2062, :2129`).

## 0. Executive summary

Round 16 named F2 ("make `T_from_hstat` return false on saturation") as a candidate fix, flagged as
risky because "cases 13/14/25/28/29 legitimately hit the ceiling". This round found that claim is
**not accurate as stated**: case29 isn't graded at all (excluded, `cases.cpp:569-593`); of the
remaining four, three (13/14/25) sit 3-4 orders of magnitude below the 1e6 K ceiling in their
converged state, and only case28 (Ms=100 air) comes close (0.587x analytically, and its measured
converged max-T is still 41% below the ceiling). Measured via a zero-source-change technique:
recovering each case's final per-cell T from the existing `denner1d_dump` output by inverting
`alpha*rho_a(p,T)+(1-alpha)*rho_b(p,T)=rho` (both p/rho and phase constants are already printed/
known) -- no cell in any of the 10 shock cases sits at the ceiling in its converged state.

**More importantly, a structural objection to F2 as literally specified was found, independent of
any measurement**: `T_from_hstat`'s `false` branch means "keep `s.T[i]` at its previous value"
(the existing else-branch at `acid.cpp:1218-1219`), which makes `compute_R` a function of CALL
HISTORY, not of state alone -- breaking the four `compute_R(); // restore` sites that assume
re-evaluating from the same `(u,p,h)` reproduces the same residual (load-bearing for the FD
Jacobian assembly at `:1685`). **F2 as specified is the wrong shape regardless of measurement
results.** The corrected form, **F2''**, is pre-registered: keep `T_from_hstat` state-pure (still
returns the clamped T), but additionally report saturation to the CALLER, which treats "any cell
saturated in the accepted iterate" as a new stall reason (5, "T ceiling saturated") triggering the
existing dt-halving retry machinery -- composes with the existing reason 1-4 taxonomy and round
14's `diverged` marking, without touching residual purity.

## 1. T_from_hstat's return value -- exactly one consuming call site

Two call sites total. C1 (`acid.cpp:1216`, inside `compute_R`'s coupled h->T inversion, OMP-parallel
over cells) is the ONLY site that uses the boolean -- `if (T_from_hstat(...)) s.T[i] = Tnew;`, else
keeps old T. C2 (`acid.cpp:1625`, the `ACID_AJAC_CHECK` debug probe) discards the bool; `T_out` is
written unconditionally regardless. Since `coupled==true` unconditionally for all 19 graded cases
(sect.0), C1 is the SOLE `s.T` update on the published path -- F2/F2'' is not a niche-path change,
it touches every cell's temperature update in every graded case's every residual evaluation.

`T` is clamped at BOTH ends (`std::clamp(Tn, 1e-6, 1.0e6)`, `:356`). Lower saturation already
returns `false` today (existing, exercised or not -- measured in sect.5 P4); upper saturation
returns `true`. F2 is better framed as "make the existing failure signal symmetric" than as "add a
new failure mode".

## 2. Thermodynamic headroom -- measured for all 10 shock/interface cases

Recovered final-state max T per case (analytic Hugoniot cross-check + dump-inverted measurement,
agreeing to <1%): cases 13/14/15 at 3-9e2 K (0.0003-0.0009x ceiling); cases 24/25/26/27/33/34 at
0.6-3.0e4 K (0.006-0.030x ceiling); **case28 (Ms=100 air) at 5.872e5 K, 0.587x ceiling -- the only
case within a factor of 2**, still 41% below saturation in its converged state. Zero cells at the
ceiling in any converged/accepted state across all 10 cases measured. The remaining nine graded
cases (acoustic/advection, IC temperatures ~300-400K) are expected far below and confirmed cheaply
in sect.5 P1.

**Side finding, out of scope but recorded**: case29 (excluded, Ms=100 water) has an analytic
post-shock temperature of 2.932e6 K -- 2.93x ABOVE the solver's own 1e6 K clamp. Its initial
condition is not representable by the solver's own thermodynamic clamp, which very plausibly
explains the existing `cases.cpp:591` blocker comment ("dt collapses ~1e-9, front under-resolved")
that has never previously been explained. Not pursued this round (would require raising a global
physical clamp, a separate decision).

## 3. The instrument -- `ACID_TSAT` (new, ~35 lines, default OFF, stderr only, NOT yadv-gated)

Deliberately `yadv`-INDEPENDENT (unlike `ACID_RCELL`/`ACID_RINIT`, which are `yadv`-gated and
therefore structurally unable to observe the OFF path -- this round's question is specifically
about OFF). Probes `s.T[i] >= 1.0e6` (upper) / `<= 1.0e-6` (lower) immediately after the C1 loop
closes (`acid.cpp:1220`, before `eval_thermo`), serial scan, integer counters only, zero new FP
arithmetic when unset. Three insertion points: (A) per-residual-call detection after the C1 loop;
(B) accepted-state detection after both `stepped=true` sites (clean-accept and
`ACID_STALL_ACCEPT`-accept); (C) end-of-run summary (`TSAT-TOTAL`) alongside `STALL-ACCEPT-TOTAL`.
`ACID_TSAT=1` gives the summary; `=2` adds a per-event line. Stated limitation: the probe cannot
distinguish "saturated on THIS call" from "a previous call saturated and this one kept the frozen
value" -- it is therefore a strict UPPER BOUND on F2/F2''s blast radius (a zero reading is
conclusive; a nonzero reading needs one refinement pass, sect.5 P3).

## 4. Measurement protocol

P0: rebuild on unmodified HEAD (checked-in binaries predate round 11, contain none of the round 11+
diagnostic strings -- same trap round 16 hit). P1 (zero-source-change screen, before any edit):
recover final-state T from existing `denner1d_dump` output for all 19 graded cases via density
inversion, confirms sect.2's table. P2 (main measurement): `ACID_TSAT=1` on all 19 OFF-path dumps,
read `TSAT-TOTAL`. P3 (refinement, only if any `calls_hi>0`): `ACID_TSAT=2 ACID_BLK_STEP=<step>` to
localize; check whether `calls_lo>0` in the same window to determine whether the over-count caveat
is live. P4 (calibration, free from the same runs): how often does the EXISTING lower-saturation
`false` branch already fire on passing cases -- the single most decision-relevant number (nonzero
=> the reject-and-retry response is already an exercised path, lower incremental risk for F2'';
zero everywhere => the false branch is effectively dead code today, stronger case for caution). P5
(positive control): `ACID_TSAT=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` on case33 -- round 16
inferred ceiling saturation there from `dh=3.7e12` but never measured it directly; this is the
first direct confirmation, and doubles as gate G9 (instrument validity -- if this reads zero, either
the instrument or round 16's inference is wrong, and no other TSAT number may be trusted until
resolved).

## 5. Pre-registered verdict targets

V-SAFE (`calls_hi==0` for all 19 OFF cases): F2'' is provably byte-identical on OFF; round 16's
stated risk refuted; F2'' becomes a cheap, well-gated follow-up round. V-RISKY-TRANSIENT
(`calls_hi>0` but no accepted/final state ever saturated): perturbs the Newton trajectory on a
passing case (expected: case28 only, if any) -- needs its own A/B round, not free. V-RISKY-CONVERGED
(any accepted/final state saturated): the graded answer is itself not a solution of `hmix(T)=hstat`
-- sect.2 already rules this out for all 10 shock cases, so this can only arise from one of the nine
mild cases, which would itself be a surprise. V-INCONCLUSIVE: `calls_hi>0` and `calls_lo>0` in the
same window -- the over-count caveat is live, needs a refined probe (a future diagnostic round, not
a fix). Expected outcome, stated in advance so the round is falsifiable: V-SAFE or
V-RISKY-TRANSIENT-on-case28-only.

Recommending F2'' as a future round's implementation requires ALL of: (1) V-SAFE or narrowly-scoped
V-RISKY-TRANSIENT; (2) the fix implemented is F2'' (state-pure), not F2; (3) P4 shows `calls_lo`
nonzero somewhere (the reject-retry response is exercised, not theoretical); (4) that future round
runs its own full gate battery including an explicit A/B on case28. This round implements no fix --
the "essentially free and fully specified" bar is not met, because sect.1's structural objection
means F2-as-named is not merely risky but wrong-shaped; recommending it verbatim would be worse than
not touching it.

## 6. Gates (full battery -- a source change is made, even if diagnostic-only)

G0 build+unit. G1 OFF 19/19, stdout byte-identical to fresh pre-edit baseline. G2 plain-ON 15/19
byte-identical. G3 `+ALPHA_IMPLICIT` 14/19 byte-identical (case24/34 must not move). G4
FD-invariance byte-identical. G5 `ACID_TSAT` unset => zero TSAT lines in every G1-G4 run. G6
(critical, stronger than round 16's equivalent since this flag is NOT yadv-gated) `ACID_TSAT=1` on
an OFF validate run => stdout STILL byte-identical to G1; only stderr differs -- if this fails the
instrument is perturbing the solve and nothing else is trustworthy. G7 malformed
(empty/`0`/`abc`/negative) => no crash, no output, `=0` is fully OFF. G8 `git status --short --
cpp/` shows only `acid.cpp` modified, diff is exactly the flag + six counters + blocks A/B/C. G9
(mandatory) case33 `+ALPHA_IMPLICIT` positive control reports `calls_hi>0`, per sect.4/P5. G10 P1's
dump-recovered temperatures on the fresh build reproduce sect.2's table exactly.

## 7. Non-goals

No F1, F2, or F3 implemented (diagnostic only). No `ACID_YADV_HREINIT`, `ACID_STALL_ACCEPT`, case33
fix. No edits to `RINIT`/`RMISM`/`RCELL`/`STALLED-DETAIL`/`STALL-ACCEPT` prints. No
`cases.cpp`/`validation.cpp` edits -- in particular do NOT un-exclude case29, even though sect.2
explains its likely blocker (a separate decision requiring a global clamp change). No change to the
1e6/1e-6 clamp values themselves. No tuning constants, no per-case coefficients.

## 8. Literature

Not needed -- pure code-behavior verification (grep, code read, solver's own output), no numerical-
method question involved.
