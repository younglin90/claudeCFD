# Round 25 Plan — Recovery-site PTE alpha recovery (F3), made concrete

**Thread**: round 24 §8.1's explicitly RECOMMENDED thread — round 16 §26.3's F3 candidate,
concretized. Non-goals: `rho_star`/`theta_o`, case33's advection channel beyond what F3 touches,
`max_steps`, case29.

**Advisor verification note**: key structural claims checked directly against the code --
`acid.cpp:1229-1234` (recovery site, exact match to plan), `:1176-1178` ("same triple" comment,
exact match), `:1239-1250` (Eqs.43-44 rebuild, exact match), `eos.hpp:77` (`alpha_roundtrip_floor`)
and `:115` (`pT_from_v_e_massfrac`) both confirmed present. No structural error found.

**Advisor note on Stage 0's hand-computed tables**: the plan's §3.1-3.3 numbers were computed by
hand (not by running new code) and cannot be independently reproduced without implementing Stage 1
first. The plan itself designs around this: P0 ("the in-code ACID_F3 meter reproduces §3.1's
alpha_F3 at case24/33/34 step 0 cell 80 to >=5 significant digits") is the built-in self-check --
Stage 1's implementation is verified against Stage 0's hand arithmetic as the first gate, not
trusted blindly.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §35)**: P0 confirmed live
-- the `ACID_F3` meter's `dal@80` matches §3.1's hand-computed `alpha_F3` to 5 significant digits on
case24/33/34. T3 confirmed: `drho`/`dh` at the front cell improve 5.4-13.7x/349-1092x, matching
§3.2's table to every shown digit for `drho` and within ~0.1% for `dh` (hand-calc rounding) --
round 24's flagged same-triple-break risk did NOT materialize as harm (S6 does not fire). T1/T4/T6:
F3 repeatedly converts a STALLED-to-NaN failure into a finite-but-inaccurate completion (case34
under plain `B+F3`; case24 under `B+RECON+F3`, directly confirming round 24 §34.5's own prediction;
case33 under `C+F3`) -- but T5's full sweep is still `15/19`, the exact same fail set `{15,24,33,34}`
as plain B: no case flips its pass/fail gate. T2: case13's predicted risk lands exactly at the
predicted magnitude (`l2_p` +7.4%, matching the "6-8% worse" prediction) and well short of the
pre-registered S4 threshold (`u_shock_delta_cells` 1->2, not >3); case14 improves (`l2_rho` -12.5%).
T8: phase-mass conservation excluded from drifting by construction (the diff hunk contains zero
writes to `Yv`/`s.p`/`s.T`/`s.rho`/`s.hstat`, grep-verified). **Verdict: S2** (substantial partial)
-- exactly as the plan's own rule table defines it: T5 unchanged pass_count, 13/14 pass, T1 shows a
genuine mode change on 24/33/34, still fail. `consecutive_failures` NOT incremented.
`ACID_YADV_F3`/`ACID_F3` committed as gated-off research infrastructure, same precedent as
rounds 21/22/24. F3b (§8) NOT built -- its trigger condition only partially met. All hard gates held
(OFF 19/19, `ALL GATES OK` unchanged from round 24, unit-test numbers unchanged).

---

## 0. Executive summary

Round 24 §8.1 recommended: recover alpha at the NEW Y's own PTE equilibrium `(p*,T*) =
pT_from_v_e_massfrac(1/rho, hstat-p/rho, Y, A, B)` instead of the stale `(p_o,T_o)`, writing ONLY
`s.alpha` (never p,T -- so it cannot reproduce round22's Abgrall-type pressure perturbation on
13/14 by construction). Round24 flagged an unmeasured risk: this breaks the "same (alpha,p_o,T_o)
triple" invariant the code documents (`acid.cpp:1176-1178`), since the Eqs.43-44 rebuild still uses
`(p_o,T_o)` for phase densities.

**Stage 0's hand calculation (to be confirmed live in Stage 1) claims**: on case24/33/34's front
cell, F3 moves alpha 5-14x closer to the exact invariant (0.5/0.75/0.25) than the status-quo stale
recovery, AND the "same triple" break makes `drho` 5-14x BETTER (not worse) and `dh` 350-1090x
better than the status quo -- i.e. round24's feared risk, if the hand math is right, is actually a
net improvement. On case13/14, F3's alpha perturbation is measured 1-2 orders of magnitude smaller
than on 24/33/34 (confirming the "cannot reproduce Abgrall directly" argument concretely) -- but
case13 (not case14) shows `drho` getting WORSE by 6-8% at its worst cells, inverting the naive
expectation that case14 (RECON's actual failure) would be the risk.

**This round's job**: implement F3, verify the hand arithmetic (P0), then let the actual gates
decide -- not the hand calculation.

---

## 1. Code facts verified

`cpp/denner_1d/src/acid.cpp` unless noted (line numbers as of round24 HEAD `23c3ce9`).

| Line(s) | Content |
|---|---|
| `1176-1178` | "same triple" comment: "alpha stays a DERIVED quantity, recovered from the NEW Y at the OLD (p,T) right before the ACID old-level rho_o/h_o re-evaluation, so the two use the same (alpha, p_o, T_o) triple" -- verified verbatim |
| `1229-1234` | recovery site: `s.alpha[i] = clamp(alpha_from_mass_fraction(Yv[i], phase_props(pu,Tu,A).rho, phase_props(pu,Tu,B).rho),0,1)` at `(pu,Tu)=(max(p_o,1),max(T_o,1e-6))` -- verified |
| `1239-1250` | Eqs.43-44 rebuild: `rho_o,hstat_o,Htot_o` from `s.alpha[i]` (whatever F3 leaves there) at OLD `(p_o,T_o)` -- verified, unchanged by this round |
| `1252+` | `RMISM` (`ACID_RINIT`): `drho = max|s.rho[i]-rho_o[i]|`, `dh = max|s.h[i]-Htot_o[i]|` -- the round24-mandated observable, reused unmodified |
| `1293+` | `RCELL` (`ACID_RCELL`): per-cell window, the zero-code Stage-0 instrument |
| `eos.hpp:77` | `alpha_roundtrip_floor(rho_a,rho_b)` -- round24's `8*eps*kappa` bound, reused as the constant-free skip predicate |
| `eos.hpp:115` | `pT_from_v_e_massfrac(v,e,Y,a,b)` -- round21's closed-form PTE inversion, `ok=false` on `W<=0`/bad disc/no positive root/`p<1`/`T` outside `(1e-6,1e6)` |
| `848-916` | `ACID_YADV_RECON` block (read-only reference, not modified) -- `e_t = s.hstat[i]-s.p[i]*v_t` pattern F3 reuses |
| `918-968` | `ACID_YADV_RESYNC` block (read-only reference, not modified) |
| `709+` | `ACID_PROJ_UNTIL`/`ACID_RECON_NULL` declarations -- new flags go nearby |
| `1436-1444` | `+ALPHA_IMPLICIT`'s in-compute_R alpha re-derivation (read-only, T6 target only) |

State liveness verified: between `s=s0;`/`Yv=Yv0;` and the recovery site, nothing writes
`s.p/s.T/s.rho/s.hstat/s.h/s.u` -- `s.rho[i]`,`s.hstat[i]` at the recovery site are the accepted
previous-step values, `p_o[i]==s.p[i]` bitwise, consistent with `(p_o,T_o,s0.alpha)`.

---

## 2. Derivation

### 2.1 The candidate (F3)

At the recovery site, compute (but do not yet write):
```
v* = 1/s.rho[i]
e* = s.hstat[i] - p_o[i]*v*
(p*,T*) = pT_from_v_e_massfrac(v*, e*, Yv[i], A, B)   // NEW Y's own PTE state
alpha_F3 = clamp(alpha_from_mass_fraction(Yv[i], rho_a(p*,T*), rho_b(p*,T*)), 0, 1)
```
Write ONLY `s.alpha[i] = alpha_F3` (when `r.ok` and above the roundtrip floor; else keep the
status-quo `alpha_stale`). `s.p, s.T, s.rho, s.hstat, s.h, Yv` are never touched by this block.

### 2.2 The exact property F3 buys

`pT_from_v_e_massfrac` enforces `v(p*,T*,Y) = v*` where `v(p,T,Y) = bbar(Y)+T*S(p)` is exactly the
mass-fraction blend `1/rho = Y/rho_a+(1-Y)/rho_b`. So `alpha_F3*rho_a(p*,T*)+(1-alpha_F3)*rho_b(p*,T*)
== s.rho[i]` exactly (to the inversion's ~1e-11 accuracy) -- F3's alpha is, by construction, the
volume fraction that conserves the cell's ACTUAL mass at the NEW composition. The status-quo
recovery has no such property (it just re-reads the map at a lagged pressure).

### 2.3 What F3 gives up (round24's flagged risk)

Eqs.43-44 still evaluates phase densities at `(p_o,T_o)`, so `rho_o = alpha_F3*rho_a(p_o,T_o) +
(1-alpha_F3)*rho_b(p_o,T_o) != s.rho[i]` in general -- the "same triple" is broken. Whether this
helps or hurts `drho` was unmeasured by round24; Stage 0's hand arithmetic (§3 below) claims it
helps by 5-14x on 24/33/34's front cell, but this must be confirmed live (P0/P1).

There's also a derived, unavoidable regression channel: at a uniform-(p,T) interface, the status-quo
recovery is ALREADY the correct answer (the cell genuinely is at `(p_o,T_o)` with the new Y), while
F3 forces `p* != p_o` by `O(delta-Y * (cp_a-cp_b)/cpbar)` -- this is the case13/14 risk, measured
in §3.3 to be 1-2 orders of magnitude smaller in alpha-space than the 24/33/34 correction.

### 2.4 Why F3 cannot reproduce round22's Abgrall failure mode directly

Round22 root-caused case14's RECON failure as a STATE-level pressure perturbation (`s.p` written
with `p*!=p`) at a T-jump-at-constant-p contact. F3 never writes `s.p`/`s.T`. The fictitious `p*`
can still reach the solution indirectly through alpha -> rho_o/hstat_o/Htot_o and the mass flux,
but attenuated by `d(alpha)/d(p*)` -- measured in §3.3 to be ~2 orders of magnitude on 13/14.

---

## 3. Stage 0 hand-computed tables (to be confirmed live, P0/P1)

### 3.1 case24/33/34, step 0, front cell i=80, plain `ACID_YADV=1`

| case | alpha_stale | alpha_F3 | exact truth | error before | error after | improvement |
|---|---|---|---|---|---|---|
| 24 | 0.996661 | 0.564528 | 0.50 | 0.4967 | 0.0645 | 7.7x |
| 33 | 0.997989 | 0.795939 | 0.75 | 0.2480 | 0.0459 | 5.4x |
| 34 | 0.992395 | 0.304053 | 0.25 | 0.7424 | 0.0541 | 13.7x |

### 3.2 The round24-mandated drho/dh measurement (Eqs.43-44 rebuild at (p_o,T_o) with each alpha)

| case | drho now | drho w/ F3 | dh now | dh w/ F3 |
|---|---|---|---|---|
| 24 | 4.9509e2 | 6.4324e1 (7.7x better) | 7.1850e4 | 9.6302e1 (746x better) |
| 33 | 2.4721e2 | 4.5794e1 (5.4x better) | 1.0157e5 | 2.9064e2 (349x better) |
| 34 | 7.4005e2 | 5.3882e1 (13.7x better) | 3.6783e4 | 3.3718e1 (1091x better) |

Cross-checked against the live `ACID_RINIT` instrument for the "now" column -- exact match.

### 3.3 case13/14 -- the real risk (measured, not assumed)

F3's alpha perturbation on 13/14 is O(1e-3 - 2e-2), 1-2 orders of magnitude smaller than on
24/33/34's O(0.4-0.7). Case14's `drho`/`dh` are IMPROVED 3-6x by F3 at interface cells. **Case13 is
the only case where F3 makes `drho`/`dh` worse (6-8%)** -- and case13 is exactly the case RECON
broke, via `case13_u_shock_delta_cells` (a shock-location, not interface-oscillation, criterion).
**The round's highest risk is case13, not case14** -- inverting the naive expectation.

---

## 4. Stage 1 -- implementation

### 4.1 New flags (default OFF, no constants)

- `ACID_F3` -- diagnostic, stderr only, writes nothing (inert unless `ACID_YADV`).
- `ACID_YADV_F3` -- research, applies the alpha-only PTE recovery (inert unless `ACID_YADV`).
- No mutual exclusion with `ACID_YADV_RECON`/`ACID_YADV_RESYNC`/`ACID_YADV_HREINIT` -- `B+RECON+F3`
  is a deliberate, legal combination (target T4, direct test of round24 §34.5's claim).

### 4.2 The edit at the recovery site (`acid.cpp:1229-1234`)

Replace with: compute `al_stale` exactly as today; if `yadv_f3 || f3_dbg`, additionally compute
`(p*,T*)` via `pT_from_v_e_massfrac(1/max(s.rho[i],1e-300), s.hstat[i]-p_o[i]/max(s.rho[i],1e-300),
Yv[i], A, B)`, and if `r.ok` and `|al_eq-al_stale| > alpha_roundtrip_floor(pa.rho,pb.rho)`, set
`al_new = al_eq` (only under `yadv_f3`, not `f3_dbg` alone). Fail-safe: `!r.ok` keeps `al_stale`
exactly, no fallback. **Exactly one write to any `s.*` field in the whole block**:
`s.alpha[i] = al_new;` on its own line, clearly commented as the only write -- reviewer-verifiable
by grepping the hunk for `s\.`.

Zero-behaviour-change proof for flags unset: with both flags false, `al_new = al_stale` computed by
the same expression as before (hoisting `phase_props` calls into named `pa`/`pb` does not change
FP results) -- verified empirically by G1/G2/G4.

### 4.3 The `ACID_F3` meter (diagnostic, appended after the recovery loop)

One line per step/retry (honouring `ACID_BLK_STEP`): `ncell, nrej, nnull, nabove, worst_dal@i,
worst_dp_rel@i, worst_dT_rel@i`. No new drho/dh instrument needed -- `RMISM` already gives both
from whatever `s.alpha` is live.

### 4.4 Hygiene

Add `ACID_F3`, `ACID_YADV_F3` to `scripts/yadv_r9_sweep.py`'s `ACID_ENV_VARS`. `CONFIGS`/`EXPECTED`
unchanged. No new unit test required (F3 composes only already-tested pure functions).

---

## 5. Gates and targets

Hard gates (G1 `--verify` OFF byte-identical; G2 `--sweep` unset unchanged incl. `ALL GATES OK`;
G3 `denner1d_unit` clean, `pT_from_v_e_massfrac` numbers unchanged; G4 new-flag no-op via
`ACID_F3=1` byte-diff; G5 diff hygiene, only `acid.cpp`+`yadv_r9_sweep.py`, no new numeric
literals beyond reused floors).

Round-specific targets, execution order: **T3 (drho, first)** -> T7 (case01 `linf_p`) -> T2
(13/14 pass + exact metrics RECON moved) -> T1 (24/33/34 stall step AND validate pass, not stall
step alone) -> T5 (full sweep) -> T4 (B+RECON+F3 on case24, direct test of round24 §34.5) -> T6
(C+F3 on case33) -> T8 (global mass drift, 13/14/24).

**T1/T2 discipline (round23/24 lesson)**: "no STALLED line" is NOT success. Only
`denner1d_validate pass:true` counts. Watch case13's `u_shock_delta_cells` specifically (round22's
own correction target), not just `l2_u`/`corr_u`.

---

## 6. Stop / decision rules

| Outcome | Trigger | Consequence |
|---|---|---|
| S1 full success | T5 B+F3>=16/19 AND 13/14 pass AND >=1 of 24/33/34 pass | Headline. Promotion is a SEPARATE future round (round14/20 precedent) -- do not promote here. |
| S2 substantial partial | T5=15/19 (fail subset of {15,24,33,34}), 13/14 pass, T1 shows 24/33/34 materially further (>=RECON's 20x or mode change), still fail | Headline: first mechanism moving 24/33/34 without paying 13/14 or conservation. Flag stays OFF, gated-off infra. |
| S3 neutral | 13/14 pass, 24/33/34 essentially unmoved (<2x, no mode change) | Report honestly: drho/dh measurably repaired but not the binding constraint. Narrows §26.1's claim. |
| S4 case13 harm (predicted risk) | 13 regresses (esp. u_shock_delta_cells>3), 14 survives | Confirms §3.3's prediction. Different harm mechanism than RECON's. Flag OFF, clean negative. |
| S5 case14 harm / conservation harm | 14 regresses, or >1% new mass drift on any passing case | Falsifies "no s.p write => no Abgrall" as SUFFICIENT. Important correction to rounds22/24. Flag OFF. |
| S6 same-triple break judged harmful | T3 shows drho/dh WORSE than baseline (contradicts §3.2) | Round24's warning vindicated against hand math. Do NOT rescue with a selector/blend/limiter -- record and stop. |
| S7 hard gate failure | G1-G5 | Round does not merge until fixed. |

**Anti-rescue clause**: if S4/S5/S6 fires, the honest record IS the deliverable. No per-cell
selector, damping factor, or "apply only where it helps" predicate to convert measured harm into
reported success -- that would be a tuning coefficient in disguise (forbidden).

Corrections to §26/§31/§32/§33/§34 discovered this round: annotations in §35, never edits.

---

## 7. Literature

Already in repo (do not duplicate): all 12 stubs from rounds21-24, Collis 2025 (closed-form
source), papers/library/md's Denner/Fujiwara/AlahyariBeig&Johnsen/Terashima.

New stub: `papers/2026_Bai_Xie_Yang_Yi_Sun_RFQC_oscillation_free_real_fluid_needed.md`
(arXiv:2602.00658) -- freezes within-step EOS coefficients, re-projects only at step boundary =
exactly the F3a(this round)+RECON combination's literature precedent; argues for F3a over the
conditional F3b (§8) sketch below.

## 8. F3b (same-triple restoration), CONDITIONAL -- do not build unless S2 gate fires

If T5 shows F3a harmless AND T1 shows 24/33/34 still fail AND T3 shows the residual drho is the
plausible remaining obstacle: also evaluate Eqs.43-44 at `(p*,T*)` on F3-applied cells, making
`rho_o==s.rho[i]` exactly. Pre-registered risk: `Htot_o` feeds the energy residual's old level, a
larger structural change, re-opening an indirect Abgrall channel -- prior art (RFQC) argues against
it. NOT built this round unless the gate fires.
