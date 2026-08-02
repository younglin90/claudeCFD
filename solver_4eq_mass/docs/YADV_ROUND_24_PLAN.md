# Round 24 Plan — Is there a structural withdrawal point for `ACID_YADV_RECON` on case24?
## Advisor pre-verification says: no -- and two of round 23's inferences do not survive contact
## with the data

**Thread**: round 23 §33.6's live thread -- characterize the withdrawal-point compounding effect;
does `ACID_PROJ_UNTIL` ending AFTER the shock-formation transient (structural, not tuning-constant)
restore monotonicity and correctness together?

Non-goals: case13's Jacobian-sensitivity (reference only), `rho_star`/`theta_o`, case33 advection
channel, `max_steps`, case29. No `cases.cpp`/`validation.cpp` edits, no tuning constants anywhere.

**Expected outcome, declared in advance**: no promotion, no new default, no new mechanism. The
deliverable is a corrected attribution plus annotations to round 23 (§33.3, §33.4). If the
implementing session finds itself designing a taper, it has left the plan.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` sect.34)**: F1 confirmed
live -- `ACID_PROJ_UNTIL=1` and plain B are byte-identical (0 writes, exact-skip fires on all 800
cells at step 0), so round 23's P6' test carried no information about H-B either way. F5 confirmed
live -- N=50's final state shows the shock completely exited the domain (84% overstrong plateau,
32% overfast, alpha collapsed 0.5->2e-4), not frozen; §33.4's mechanism reading corrected as an
annotation. `ntouch` reconfirmed 0 only at step 0, never again -- no global withdrawal-point
criterion exists. Built the REAL roundoff-null control (`ACID_RECON_NULL`, using the existing
unit-tested `8*eps*kappa` bound, `alpha_roundtrip_floor`) -- non-empty (nnull 2-4 cells/step,
never zero) and genuinely applied. **Decisive result: byte-identical to plain B anyway** -- H-B
excluded, this time with real evidence, not a tautology. S1 fires exactly as expected: no
structural withdrawal point exists (the correction is needed continuously wherever the front is
inside the domain, and even always-on RECON stalls on its own §26.1 blister, so the family has no
correct member to withdraw to anyway). No taper designed or built. consecutive_failures NOT
incremented. `ACID_RECON_NULL`/`alpha_roundtrip_floor` committed as inert research infrastructure.
Round 25 recommended thread: §8.1's recovery-site fix (recover alpha at the NEW Y's own PTE state,
writing only s.alpha, never p/T).

**Advisor verification note**: the plan's most load-bearing claim -- that `ACID_PROJ_UNTIL=1` is
NOT a roundoff-null control but a complete no-op -- checked directly against `acid.cpp:843-866`.
Confirmed exactly: the exact-skip (`if (al_chk == s.alpha[i]) { ++nskip; continue; }`) runs BEFORE
the write gate (`if (yrecon && (proj_until<0 || step<proj_until))`), so a cell that is skipped never
reaches the write branch at all. At step 0 (IC is already a PTE state) this fires for all 800 cells,
so `N=1`'s write set is provably empty -- round 23's "N=1 reproduces plain B exactly" was never
evidence against trajectory-chaos; it was a tautology (nothing was ever applied). This single fact
undermines round 23 §33.3's P6' conclusion and is the round's starting point, not its conclusion --
Stage 0 must re-verify it live before anything is built on it.

---

## 0. Executive summary -- what changes about round 23's picture

**F1**: `ACID_PROJ_UNTIL=1` performs ZERO writes for the entire run (step-0 exact-skip fires on all
800 cells). Round 23's "roundoff-null control" was a no-op, not a roundoff-scale perturbation. H-B
(trajectory chaos) was never actually tested.

**F2**: RECON is already spatially local with NO tolerance -- under always-on B+RECON, `ntouch` is
28-92 of 800 cells per step, and `worst_dal@i` tracks a single moving cell (the front) exactly.

**F3**: `ntouch` never reaches 0 across the whole run (min observed 28). There is no step at which
RECON has nothing to do -- a GLOBAL withdrawal-point criterion cannot exist by this measurement.

**F4**: Always-on RECON does NOT prevent round 16 §26.1's collapse -- it relocates it. The step-399
stall is a literal vacuum blister at cell 97 (`alpha=0.99997, rho=6.1e-5, T=1.0e6`). The `N->inf`
member of the `ACID_PROJ_UNTIL` family is itself wrong, so no withdrawal schedule within this family
can produce a correct case24.

**F5 (correction to round 23 §33.4)**: N=50's "no stall" is not a frozen shock -- the actual final
state (never dumped in round 23) shows the shock has COMPLETELY EXITED the domain, the post-shock
plateau is 84% too strong (2.768e10 vs reference 1.5084e10), front speed is ~32% too fast vs exact
Vs, and alpha has collapsed from the exact invariant 0.5 to 2.3e-4 for x>~0.43. `max|u|`/`maxp`
freezing at step~200 is the CORRECT signature of a formed plateau, over-read as "frozen".

**F6**: In the completing window (N=10..300), the final-state alpha-collapse transition location and
far-field alpha agree to 3-4 significant digits regardless of N -- the withdrawal point does not
control the outcome at all in that window.

**F7**: complete/stall map re-measured: N=150,300 complete; N=200 stalls at 501; N>=400 stalls at
399 (identical trajectory to always-on, since branch never occurs before the stall).

**F8**: N=5 stalls at step 19 too, but with a DIFFERENT failure mode than plain B/N=1 --
`T-ceiling-saturated cell=80 alpha=1.00000 T=1.0e6` -- the §26.1 blister, at the same cell as
case33's, appearing in case24. Round 23's table row conflating "N=5: step 19, same as plain B" was
comparing two unrelated failure modes that happen to share a step number.

**Consequence**: the commissioned question is answered NEGATIVELY by derivation before Stage 1: F3
kills the global criterion, F2 shows the per-cell criterion already exists (the exact-skip) and
already works, F4 shows the best family member is wrong anyway, F6 shows withdrawal point doesn't
even control the outcome. Per the round's own discipline (do not manufacture a mechanism when the
structural search comes up empty), no taper is designed. What replaces it: the REAL roundoff-null
control round 23 believed it had already built, constructible without any new constant from the
already-unit-tested `8*eps*kappa` round-trip conditioning bound (round 21 §1.4,
`denner1d_unit.cpp:61-71`).

---

## 1. Code facts verified by direct reading

`cpp/denner_1d/src/acid.cpp` unless noted (line numbers as of round 23 HEAD `9071de2`).

| Line(s) | Content |
|---|---|
| `726-727` | `proj_until` decl |
| `835` | RECON gate `if (yadv && (yrecon\|\|recon_dbg))` |
| `843-847` | `al_chk` computed from CURRENT `(p,T)` |
| **`848`** | **exact skip, runs BEFORE write gate -- verified** |
| `849-852` | closed-form solve, `nrej` on failure |
| `853-856` | worst-tracking (dp,dT,dal) |
| **`857-866`** | **write gate** `if (yrecon && (proj_until<0 \|\| step<proj_until))` -- verified, comes AFTER the skip at 848 |
| `868-873` | `eval_thermo` + `h` refresh, touched cells only |
| `904, 920` | RESYNC gate/write, same `proj_until` pattern |
| `944-945` | `s0` snapshot -- both projections run strictly before this |
| `1201-1207` | alpha recovery site (round 16 §26.1's birth site) |
| `1212-1223` | Eqs.43-44 rebuild |
| `2496` | `bad` gate, `ajac`-only (FD path has no reason 1) |
| `2525-2527` | reason-5 T-ceiling scan |
| `tests/denner1d_unit.cpp:61-62` | `kappa = max(ra/rb,rb/ra); tol = 8.0*2.220446049250313e-16*max(kappa,1.0)` -- the existing, unit-tested bound this round reuses |
| `cases.cpp:148` | `s.alpha_post = s.alpha_pre;` -- case24's exact invariant, re-verified |
| `scripts/yadv_r9_sweep.py:34-38` | `ACID_ENV_VARS` -- new flag must be added |

---

## 2. Derivation

### 2.1 The `{run(N)}` family is a branching tree, not a dose-response curve

`proj_until` gates only the write. For N'>N, run(N) and run(N') are bitwise identical for step<N
(determinism, round 19 H0). Nothing is INJECTED at withdrawal -- the run simply stops correcting.
"Withdrawal-point compounding" can only mean: the uncorrected continuation is not monotone in N.
F6 shows something stronger: over N in [10,300] the continuation lands on the SAME attractor.

### 2.2 The correction is needed exactly where and for as long as the front exists

case24's exact solution is one shock crossing fresh cells until exit. The lag is created wherever Y
changes (the front). F2/F3 confirm this exactly: `worst_dal@i` tracks one moving cell, `ntouch`
never reaches 0. A GLOBAL step cutoff is structurally the wrong shape; the PER-CELL criterion the
thread asks for is already `acid.cpp:848`'s exact skip, already firing for ~95% of cells.

### 2.3 The N->inf member is itself wrong

F4: always-on RECON stalls at 399 on a genuine §26.1 blister. RECON repairs the STEP-BOUNDARY state
but the IN-STEP recovery (`1201-1207`) still evaluates at `(p_o,T_o)` with the NEW Y; at the front
the single-step delta-Y is O(0.36), so even from a consistent base the recovery can still drive
alpha->1. RECON reduces the amplification (raises p_o at the front, lowering dalpha/dY there) but
does not remove it -- preventive AT THE STEP BOUNDARY ONLY; the in-step recovery re-creates the
defect at the front every step.

### 2.4 The real roundoff-null control

H-B requires a control that (a) is ACTUALLY applied and (b) is provably roundoff-sized. N=1 satisfies
(b) vacuously by failing (a). The one constant-free bound available: the round-trip conditioning
floor already unit-tested, `floor_alpha(rho_a,rho_b) = 8*eps*max(rho_a/rho_b, rho_b/rho_a, 1)`.

**Control**: restrict RECON's write to cells where every write component is within this floor --
`|dal|<=floor_alpha`, `|dp|<=8*eps*|p|`, `|dT|<=8*eps*|T|`. This is the COMPLEMENT of the exact
skip (applies exactly where state is consistent to the map's own resolution but not bit-exact). If
this control still stalls at step 19/reason 1/matching rbest,r_init -- H-B is bounded for the first
time. If it moves, H-B is alive and every prior single-realization case24 stall-step number is a
noisy sample.

### 2.5 Three questions replacing the commissioned one

Q1 (attribution repair): is H-B bounded with a control that actually applies something? Q2
(criterion): does any structural signal mark a safe global withdrawal point? (derived: no). Q3
(metric): with stall-step replaced by the exact-invariant norm A(t)=max|alpha-0.5| and the actual
validate gate, is the N-response monotone, does any N pass?

---

## 3. Pre-registered predictions and falsifiers

- **P1**: `ACID_PROJ_UNTIL=1` byte-identical to plain B; meter shows `nskip=800 ntouch=0` at step 0.
  Falsified by any diff -> P6' partially rescued, §2.4 withdrawn.
- **P2**: `ntouch>=1` at every accepted step 1..stall under always-on. Falsified by any `ntouch==0`
  step -> candidate withdrawal point exists, test it immediately.
- **P3**: `worst_dal@i` within +-3 cells of the pressure-gradient peak. Falsified -> locality story
  wrong, rework before concluding further.
- **P4**: completing N in {10,20,50,100,150,250,300} agree on transition cell (+-5) and alpha[790]
  (3 sig figs). Falsified -> transition moves systematically with N -> S3, may DESIGN (not build)
  a per-cell criterion in §8.
- **P5**: no N passes `denner1d_validate --only 24`. Falsified -> S4, headline, but note a step-count
  cutoff is itself a tuning constant.
- **P6**: null control (§2.4) non-empty on some step AND reproduces plain B's stall (step19, reason
  newton-no-progress, rbest/r_init matching to >=3 sig figs). Falsifier A: control always empty ->
  H-B stays unbounded, record honestly. Falsifier B: stall moves >2 steps or reason changes -> S2,
  H-B alive, headline, annotate §31.6/§32.4-G5/§33.3.
- **P7**: `ACID_RCELL` around cell 97 at step 399 shows the §26.1 signature (rho_o >=4 orders below
  neighbours, T_o exactly 1.0e6, al->1). Falsified -> F4 wrong, rewrite §2.3.
- **P8**: completing-N front speed >=1.2x exact Vs, plateau >1.5x reference. Falsified -> F5 wrong,
  do not write the correction.
- **P9**: `A(t_end)` exactly 0 on OFF path. Falsified -> case24/33/34's invariant doesn't hold
  discretely on the published path; halt and report that instead.

---

## 4. Stage 0 -- zero solver code

`DENNER_ACID=1` on every invocation. Never hardcode an unread step number.

### 4.0 Reproduction gate (hard, S5 if it fails)
Plain B -> step19/newton-no-progress/rbest=2.7939e13,r_init=2.3095e13. Always-on B+RECON -> step399/
T-ceiling-saturated/cell=97,alpha=0.99997,T=1.0e6. RECON meter step0 -> nskip=800 ntouch=0 dp=0.

### 4.1 P1 -- byte-compare `ACID_YADV=1` vs `+ACID_YADV_RECON=1 ACID_PROJ_UNTIL=1` on case24.

### 4.2 The N-sweep, THREE observables per run (core table; no claim on one observable alone -- G6)
N in {1,2,5,10,20,50,100,150,200,250,300,350,399,400,unset}: (1) dump+DBG -> STALLED step+reason+
cell+rbest+r_init or "ACID done"; (2) validate --only 24 -> full JSON; (3) dump stdout -> A(t_end),
min alpha, first transition cell, p plateau, vs reference.

### 4.3 Locality/front (P2,P3): always-on, record nskip/ntouch/worst_dal@i per step, min/max ntouch.

### 4.4 Two collapse tables (P7 + correction to §33.4): RCELL around cell97@step399 (always-on) and
cell80@step19 (N=5) side by side with round16 §26.1's case33 table.

### 4.5 Withdrawal aftermath (sharpest new observable, zero code): N=50+ACID_RECON meter continues
measuring after withdrawal -- track nskip decay across the whole domain post-step-50.

### 4.6 Front kinematics (P8): ACID_TEND_SCALE in {0.25,0.5,0.75} with N=50 and OFF control.
CAVEAT: with tend_scale!=1 only solver columns (incl. alpha) are valid, not *_ref/validate metrics.

### 4.7 Determinism control: run one config twice, confirm byte-identical (round19 H0 precedent).

**Gate to Stage 1**: must resolve P1, produce the 4.2 table (all 3 observables), resolve P2 and P4.
If P4 falsified -> S3, skip Stage 1c, go to §8 design-only.

---

## 5. Stage 1 -- minimal code, default OFF, zero tuning constants

### 5.1 `eos.hpp` -- `alpha_roundtrip_floor(rho_a,rho_b)` (~8 lines): the existing 8*eps*kappa bound,
factored into one function. `denner1d_unit.cpp:61-62` refactored to call it (numbers must not move
-- gate G3), plus one assertion `epsilon()==2.220446049250313e-16`.

### 5.2 `ACID_RECON` meter extension (~6 lines, diagnostic, appended not inserted): `nnull`/`nabove`
counters among cells reaching the closed-form solve (not skipped, not rejected).

### 5.3 `ACID_RECON_NULL=1` (~4 lines): restricts RECON's write to cells satisfying all three
roundtrip-floor bounds (dal, dp, dT). Unset -> one boolean short-circuit, zero new FP arithmetic,
byte-identical (verified empirically, G4).

### 5.4 Optional pre-screen (budget permitting): `ACID_RECON=1` on cases 13/14, record nnull/nabove.
Kill criterion declared in advance: if nnull is 0 or <1% of nabove on both, the floor-relaxed-skip
idea is dead, record as dead. NOT the case13 Jacobian-sensitivity thread (out of scope) -- no
conclusion about that may be drawn here.

### 5.5 Hygiene: add `ACID_RECON_NULL` to `ACID_ENV_VARS`. `CONFIGS`/`EXPECTED` unchanged.

**Explicitly NOT built**: any taper/withdrawal mechanism; any change to the normal exact-skip
predicate; any change at the recovery site (`1201-1207`). §8.1 is a design sketch only.

---

## 6. Stage 2 measurements
M1 nnull/nabove time series (always-on). M2 `ACID_RECON_NULL=1` full STALLED-DETAIL (resolves
P6/S2). M3 config G supporting observation only (no A/B, ajac-gated reason1 doesn't exist there).
M4 §5.4 pre-screen.

---

## 7. Hard gates

G1 `--verify` OFF byte-identical, 9/9. G2 `--sweep` unset unchanged, `ALL GATES OK`, `EXPECTED` not
edited. G3 `denner1d_unit` clean, round-trip numbers UNCHANGED after refactor. G4 new-flag no-op
(unset: byte-identical on 24 and 13; `ACID_RECON=1` alone still 15/19). G5 diff hygiene (eos.hpp,
denner1d_unit.cpp, acid.cpp flag+meter+null_ok, yadv_r9_sweep.py ACID_ENV_VARS only). **G6
(round-specific)**: every case24 statement backed by BOTH a run-status observable AND a physical
one (validate JSON and/or A(t)/profile) -- a stall step alone is a gate failure (this is exactly
what prevents F5/F8-style errors). **G7 (round-specific)**: every correction to round 23 is an
ANNOTATION in §34, not an edit to §33.

---

## 8. Stop / decision rules

| Outcome | Trigger | Consequence |
|---|---|---|
| S1 "no structural withdrawal point; question mis-posed" (EXPECTED) | P2,P4,P5,P7 all confirmed | Write §34 with the derivation+tables. Annotate §33.3 (P6' vacuous), §33.4 (over-shock not frozen), §33.6 pt.1-2. No taper designed/built. Recommend round25=§8.1 recovery-site fix. consecutive_failures NOT incremented. |
| S2 "H-B is alive" | P6 falsifier B | Headline. Annotate §31.6/§32.4-G5/§33.3: case24 stall-step numbers are single noisy realizations. Declare stall step inadmissible as a metric without an ensemble; ACID_RECON_NULL is the ensemble generator for future rounds. Design nothing. |
| S3 "withdrawal point DOES control outcome" | P4 falsified | Only then may §8.2 be WRITTEN as a design (per-cell, structural). Still no implementation this round. |
| S4 "some N passes" | P5 falsified | Headline; note immediately a step-count cutoff is itself a tuning constant -> non-promotable as-is; follow-up = identify the structural quantity it proxies. |
| S5 baseline moved | §4.0 fails | Halt scientific content; report baseline discrepancy as the deliverable. |
| S6 hard gate failure | G1-G7 | Round does not merge until fixed. |

**"No structural criterion found" is an explicitly legitimate, EXPECTED round outcome. Do not
manufacture a mechanism to avoid it.**

### 8.1 Recommended round-25 thread (design sketch only, NOT implemented this round)

Round 16 §26.3's F3, made concrete by this round's F4: alpha recovered from NEW Y at OLD (p_o,T_o)
where dalpha/dY=859 and single-step delta-Y at the front is O(0.36). Candidate: recover alpha at
`(p*,T*) = pT_from_v_e_massfrac(1/rho, hstat-p/rho, Yv[i], A, B)` (the NEW Y's own PTE state)
instead of at `(p_o,T_o)`, writing ONLY `s.alpha` (never p,T) -- so it cannot reproduce round 22's
Abgrall-type pressure perturbation on 13/14. Distinct from round 23 §8 pt.2's refuted "alpha-only
projection" (that one evaluates at stale (p_o,T_o) -- i.e. today's code; this evaluates at the NEW
Y's own equilibrium). Pre-registered risk: Eqs.43-44 rebuild would then blend phase densities at
(p_o,T_o) with an alpha derived at (p*,T*) -- breaking the "same triple" invariant the code
documents. Must measure RMISM's drho before any gate.

### 8.2 Written only under S3
A per-cell structural withdrawal criterion using `alpha_roundtrip_floor` as the only bound.

---

## 9. Wall-clock budget
Stalling runs <1min. Completing runs (~2400 steps) 2-5min. 4.2's 15-run grid x3 observables ~60-75
min. 4.3-4.6 ~25min. Stage1 build+gates ~30min. Total ~2h. Exceeding budget -> "not evaluated",
never chased (round21 §31.6 case34 precedent).

---

## 10. Literature
Already in repo (do not duplicate): 12 stubs from rounds 21-23, papers/library/md's Denner 2018,
Collis 2025, Fujiwara 2023, AlahyariBeig & Johnsen 2015, Terashima 2025.

New stubs (DOIs to verify before commit) -- direct literature for F5 (new to this project):
- Hou & LeFloch 1994 (Math. Comp. 62, 497-530) -- canonical analysis of nonconservative schemes
  converging to wrong-speed/wrong-strength shocks, exactly F5's signature.
- Karni 1994 (JCP 112, 31-43) -- primitive/non-conservative remedy for pressure oscillations and
  its known shock-speed cost -- the RECON(state write)/RESYNC(16% drift) trade this project
  measures. Cite alongside the existing 2021 conservative-vs-non-conservative stub.

---

## 11. Deliverables
`docs/YADV_RESEARCH.md` §34 (34.1 no-op finding+P6' annotation; 34.2 corrected §33.4 mechanism;
34.3 N-sweep table w/3 observables; 34.4 locality/ntouch/post-withdrawal decay; 34.5 two collapse
tables; 34.6 ACID_RECON_NULL+H-B verdict; 34.7 gates; 34.8 verdict; 34.9 reproducing).
`docs/YADV_ROUND_24_PLAN.md` (this file + actual outcome). `docs/YADV_ROADMAP.md` update. Source
diff: eos.hpp, denner1d_unit.cpp, acid.cpp, yadv_r9_sweep.py only. 2 paper stubs. Commits per
convention, local merge only, round does not merge if G1-G7 fail.
