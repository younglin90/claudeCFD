# ROUND 28 PLAN — case15 under plain config B: the mass collapse is a *downstream symptom of an infeasible frozen-alpha Newton system*; a per-cell, constant-free implicit-alpha candidate

*(Planner output, round 28. Written to the standard of `docs/YADV_ROUND_27_PLAN.md`. All file:line citations below were read live at `HEAD = 0a5c0a8` in this worktree — none are copied from round 27's write-up.)*

**Advisor verification note**: key structural claims checked directly against the code -- `acid.cpp:2656/2663/2664` (`M_prev`/`M_new`/`adv`/`remap` definitions, exact match), `acid.cpp:1307-1314` (recovery site, exact match), `acid.cpp:1361-1377` (Eqs.43-44 rebuild loop, exact match), `acid.cpp:2390` (pressure floor clamp `s.p[i] = std::max(sbak.p[i] + dpi, 1.0);`, exact match), `acid.cpp:1563` (config C's `if (yadv && alpha_implicit)` branch in `compute_R`, exact match; plan cited `:1563-1571`), `acid.cpp:2137` (`const bool aimp = yadv && alpha_implicit;`, off-by-one from plan's cited `:2138`, immaterial), `acid.cpp:2117` ("factor 521.56" comment, off-by-8 from plan's cited `:2125`, immaterial). No structural error found.

---

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §38)**: T0 reproduced
exactly (B stalls at 86% of `r_init`, C converges to `2.4037e-07` in 7 iterations, `B+F3`
identical to B digit-for-digit). Implementation found and fixed one real bug beyond the plan's
own anticipation: the naive `cav[]`/`cav_dbg` split let `ACID_NFEAS` ALONE populate the real
mask, making the "diagnostic-only" flag solution-affecting -- caught by G4-early on cases 15/24,
fixed with a genuine shadow-mask/shadow-count split (`cav_shadow[]`/`cav_n_shadow`) so the
diagnostic never writes the array the solution-affecting consumers read. T1's blast-radius
census matched the plan's own prediction closely, with one welcome refinement: case14 (the
plan's single flagged at-risk case) fires ZERO times across its entire run, not merely rarely --
predicate P1 was adopted with no need for the pre-registered P2 fallback. Stage 2.0's harm gate
(checked first, per the plan's own anti-round-27 discipline) came back completely clean:
`pass_count=15/19`, fail set identical to plain B, with an unpredicted bonus (case24 becomes
finite). R1/R2/R3 all confirmed with comfortable margin. **R4 did not hold** -- `l2_rho`/
`corr_rho` fall short of the gate AND of the plan's own §0.7 prediction of matching config C.
Verdict: **S4 (neutral)**, exactly the outcome the plan's own decision table names for this
combination. `consecutive_failures` NOT incremented (stays at 1, the streak round 27's S5
started does not advance). The flag is committed gated-off; round 29's real open question is now
narrower and different from what round 27 left: not "why does B collapse" (answered, mechanism
fixed) but "why is `B+CAV`'s density field worse than config C's despite near-identical mass and
near-identical implicit-alpha coverage" -- proposed but not measured this round (the per-cell/
per-iteration-lagged nature of `cav[]` vs C's uniform treatment is the leading candidate
explanation, stated as a hypothesis, not a finding).

---

## 0. Executive summary

**Diagnosis approach.** Rather than look for a better repair *at* the Eqs.43-44 rebuild (round 27's `REBUILD_ADV` site), I asked the question the briefing's direction (1) posed: *why* does `REMAP` cancel against `ADV` under C/RECON and not under plain B? Answering that turned out to require only reading `ACID_MBAL`'s own arithmetic, and the answer immediately falsifies the framing that the collapse is a bookkeeping problem at all. I then ran four zero-code measurements (`ACID_RHIST`, `ACID_TEND_SCALE`, per-case floor census, per-case Newton census) that pin the actual cause.

**What I concluded, plainly:**

1. **`ADV + REMAP` is an algebraic identity, not an empirical cancellation.** From `acid.cpp:2656/2663/2664`: `adv = M_star − M_prev`, `remap = M_reb − M_star`, so `adv + remap ≡ M_reb − M_prev` *by construction, in every config*. Round 27's "exactly cancelled by `ADV` (net `0.0000`)" is therefore the statement **`M_reb == M_prev`** — "the Eqs.43-44 rebuild preserves the domain's total mass". Under C and RECON it does; under plain B it does not. This is a refinement (not a contradiction) of §37.4, and it makes the question sharp: *why does the rebuild destroy mass under B?*

2. **Because `p_o` is a lie, and it is a lie because the Newton system is infeasible.** The rebuild computes `rho_o = rho_mix(Y_new, p_o, T_o)` — mathematically the exact PTE mixture density at the old `(p,T)` (`acid.cpp:1369-1375`, since `alpha_from_mass_fraction` followed by the `al·ρ_a+(1−al)·ρ_b` blend is an identity for `1/ρ = Y/ρ_a + (1−Y)/ρ_b`). At `p_o = 1 Pa`, `ρ_air ≈ 1.2e−5` so *any* air mass fraction forces `alpha → 1` and `rho_mix → ~1e−2`. The rebuild is arithmetically correct; the state it is handed is not.

3. **The floor is not reached by accident: under plain B the discrete continuity equation has *no admissible solution* in a cavitating cell.** Measured live, case15 step 0, identical starting residual `n0 = 5.9172e+07` in all three configs:
   - **B**: `5.9172e7 → 5.1011e7 → 5.0925e7 → 5.0924e7 → 5.0923e7 → …` — **stalls at 86 % of the initial residual**, line search collapses to `al = 0.016`, `dp` clamped at exactly `5.00e4, 2.50e4, 1.25e4, 6.25e3` (i.e. the existing `|dp| ≤ 0.5·p` limiter binding on every iteration, Newton asking for `p → −∞`).
   - **C** (`+ALPHA_IMPLICIT`): `5.9172e7 → 4.847e7 → 3.18e7 → 5.99e6 → 6.28e5 → 5.5e3 → 0.44 → 2.4e−7` — **quadratic convergence in 7 iterations**. Same state, same dt, same everything else.
   - **B+F3**: `5.9172e7 → 5.1011e7 → 5.0925e7 → … al=0.016` — **bit-for-bit the same stall as B.** F3 lives outside the Newton, so it cannot and does not touch the infeasibility.

4. **Root cause, stated as physics.** In a 4-equation (full PTE) model the mixture's compressibility is dominated by `∂α/∂p` — that is exactly what makes the Wood sound speed soft (`c ≈ 54 m/s` for case15's IC, against `±100 m/s` initial velocities, i.e. a Mach-1.9 double rarefaction). Plain B freezes `alpha` for the whole Newton solve, so the residual sees only the *frozen-composition* compressibility, `s.drhodp = al/(R_a T) + (1−al)/(R_b T)` (`acid.cpp:326`), which for case15's state is smaller than the true PTE value by a factor the code's own comment already records: **"D_p 1.00196e-06 → D_p\* 5.22580e-04, a factor 521.56"** (`acid.cpp:2117`). A nearly-incompressible cell cannot expand, so no pressure satisfies continuity, `p` is driven to the 1 Pa clamp, and the next step's recovery + rebuild then delete the mass at the spurious floored state. Confirmed directly: at `ACID_TEND_SCALE=0.01`, config B has `p` down to exactly `1.2500e+04` (= `1e5·0.5³`) across ~40 cells with **`alpha` frozen at exactly `0.055000` and `rho` unchanged at `949.4`**, while config C at the identical time has `alpha = 0.3147`, `rho = 688.5`, `p = 1.2675e4`, confined to **2 cells**.

5. **Consequence — an entire family of candidates is retired.** Any repair located at the recovery site, at the Eqs.43-44 rebuild, or at a once-per-step state projection is *downstream* of an unsolvable system and cannot fix case15 under B. This is now supported three independent ways: `REBUILD_ADV` (round 27: NaN on 4 cases), `F3` (mass restored to 961.98 but **`nfloor = 400/400` — the entire domain at the floor, and `l2_p = 0.00000` is a *degenerate* number because the N=800 reference is equally floored**), and the residual-stall trace above.

6. **The Stage-2 candidate this round proposes therefore operates at the only place it can — inside the Newton residual — but per-cell, gated on an exact, constant-free infeasibility predicate**, so that its behaviour is bounded between two *already-measured* configurations (plain B, 15/19, and config C, 14/19) rather than being a genuinely new state as `REBUILD_ADV` was. Flag: `ACID_YADV_ALPHA_IMPLICIT_CAV`, default OFF, inert unless `ACID_YADV`, **byte-identical to today whenever `ACID_YADV_ALPHA_IMPLICIT` is already set** (so configs C/D/F are untouched by construction).

7. **Honest expectation, pre-registered.** This candidate is expected to *remove case15's mass collapse* and reproduce config C's accuracy metrics (`l2_rho ≈ 0.02`, `corr_rho ≈ 0.997`). It is **not** expected to flip case15 to `pass` by itself, because config C then hits its own separately-characterised `cj = 30` central-jump defect (round 27 §4.5) — a resolution defect at the stagnation point, explicitly *not* this round's target. That is an S2-class outcome by design, and I say so up front rather than overselling.

---

## 1. Verified code facts (all read live at `HEAD = 0a5c0a8`, this worktree)

| Thing | Exact location | Note |
|---|---|---|
| `ACID_YADV` flag | `cpp/denner_1d/src/acid.cpp:578` | |
| `ACID_YADV_ALPHA_IMPLICIT` flag | `acid.cpp:593` | config C |
| `ACID_YADV_RECON` flag | `acid.cpp:674` | |
| `ACID_YADV_F3` / `ACID_F3` flags | `acid.cpp:747` / `:746` | round 25 |
| **`ACID_MBAL` declaration + full budget identity comment** | `acid.cpp:748-782` (`const bool mbal` at `:782`) | identity `dM = ADV + REMAP − dt·BND − dt·LEAK + dt·RES`, `closure` self-test |
| `ACID_MBAL` `ADV` accumulator (`mbal_Mstar`) | declared `acid.cpp:1050`, accumulated `acid.cpp:1304` | |
| `ACID_MBAL` `REMAP` accumulator (`mbal_Mreb`) | declared `acid.cpp:1368`, accumulated `acid.cpp:1376` | |
| **`ACID_MBAL` print / term definitions** | `acid.cpp:2652-2680`; `adv` at `:2663`, `remap` at `:2664`, `M_prev/M_new` at `:2656` | source of the `adv+remap ≡ M_reb − M_prev` identity |
| Conservative `ρY` transport + `rho_star` | `acid.cpp:1296-1306` (`rho_star` at `:1301-1303`) | |
| **The alpha recovery site** (stale `(p_o,T_o)`) | `acid.cpp:1307-1314` | `Yv = anew;` at `:1307`; `s.alpha[i] = …` at `:1310-1313` |
| F3's alternative recovery | `acid.cpp:1316-1355` | writes only `s.alpha[i]` (`:1345`) |
| **The Eqs.43-44 old-level rebuild** | `acid.cpp:1361-1377`; `rho_o[i]` at `:1373`, `hstat_o` `:1374`, `Htot_o` `:1375` | phase props at `(max(p_o,1), max(T_o,1e-6))`, `:1371-1372` |
| RMISM `dal_remap`/`dal_adv` (round 13; the pieces `REBUILD_ADV` misused) | `acid.cpp:1379-1409` | diagnostic only, still present |
| **Config C's alpha-implicit path (in `compute_R`)** | `acid.cpp:1563-1571` | `if (yadv && alpha_implicit) { … s.alpha[i] = alpha_from_mass_fraction(Yv[i], ρ_a(s.p,s.T), ρ_b(s.p,s.T)) }` |
| **RECON's once-per-step reconciliation block** | `acid.cpp:926-985`; the state write `s.p[i] = r.p` at `:957`; `eval_thermo` refresh at `:968` | runs *before* the `s0` snapshot (`:1043`), so the whole retry sweep sees it |
| `compute_R` lambda definition | `acid.cpp:1544` (scratch decls `:1543`) | |
| Frozen-alpha pressure coefficient (`drhodp`) | `acid.cpp:326`; consumed only in the segregated Jacobian `acid.cpp:2471/2477/2483/2486` | |
| **Analytic-Jacobian per-cell `aimp` gate + starred `D_p*`** | `acid.cpp:2137` (`const bool aimp = yadv && alpha_implicit;`), starred terms `:2139-2148` | the "factor 521.56" comment is at `:2117` |
| Newton loop head / `compute_R()` per iteration | `acid.cpp:1926-1927` | iter cap `ajac ? 150 : 40` |
| **Linear solve producing `dxk`** | `acid.cpp:2351-2352` | |
| **Coupled 3×3 line search** (the site the new predicate reads) | `acid.cpp:2370-2397`; `sbak` at `:2375`, `dpi` clamp at `:2382`, **`s.p[i] = std::max(sbak.p[i] + dpi, 1.0)` at `:2390`** | the `1.0` here *is* the floor literal |
| keep-best / stall-break | `acid.cpp:2411`, `:2414` | |
| Segregated 2×2 line search (second `p` floor site) | `acid.cpp:2570-2583`; floor at `:2582` | not touched by this plan |
| `ACID_YADV_REBUILD_ADV` | **absent** — `grep -rn "REBUILD_ADV" --include=*.cpp --include=*.h` returns **0 hits** | revert confirmed |
| Sweep battery `CONFIGS` / `EXPECTED` / `ACID_ENV_VARS` | `scripts/yadv_r9_sweep.py:34-63` (`ACID_ENV_VARS` at `:33-41`) | `B: (15, {15,24,33,34})`, `C: (14, {14,15,24,33,34})` |
| case15 IC | `cpp/denner_1d/src/cases.cpp:682-689` | `alpha=0.055`, `u = ∓100`, `p = 1e5` |
| case15 reference | `cases.cpp:751-752` → `computed_reference(c, 800)` | same solver, same config, finer mesh |
| case14 IC (the at-risk case) | `cases.cpp:673-681` | `alpha = 1e−6 / 1−1e−6`, `p = 1e9 / 1e5` |
| case15 gate reimplementation | `scripts/yadv_r27_case15.py:107-124` | thresholds `corr_rho≥0.99`, `l2_rho≤0.05`, `cj≤max(8, 1.1·cj_r)` |

---

## 2. The diagnostic work: why cancellation happens under C/RECON but not under plain B

### 2.1 Step one — the "cancellation" is an identity; the real statement is `M_reb == M_prev`

From `acid.cpp:2656/2663/2664`:

```
adv   = mbal_Mstar - M_prev          // M_prev = Σ s0.rho·dx
remap = mbal_Mreb  - mbal_Mstar
⇒  adv + remap ≡ mbal_Mreb - M_prev      (M_star cancels identically)
```

`M_star` is a *bookkeeping intermediate* that appears in both terms with opposite sign. So round 27's observation "`REMAP` is exactly cancelled by `ADV` under C and B+RECON, net `0.0000`" is *equivalent to* **"the Eqs.43-44 rebuild does not change the domain's total mass under C and RECON."** Its raw magnitude (79.75 / 179.997) is the magnitude of `M_star`'s deviation from `M_prev`, which is a property of the explicit predictor, not of the rebuild.

Summing the accepted step: `M_new = M_reb − dt(BND + LEAK) + dt·RES`, hence

> **`dM = (M_reb − M_prev) − dt·BND − dt·LEAK + dt·RES`.**
> Domain mass conservation (up to real boundary flux, the Eqs.41-42 non-telescoping leak, and the accepted residual) requires exactly one thing of the rebuild: **`M_reb == M_prev`.**

### 2.2 Step two — when does `M_reb == M_prev` hold?

`rho_o[i] = al·ρ_a(p_o,T_o) + (1−al)·ρ_b(p_o,T_o)` with `al = alpha_from_mass_fraction(Y_new, ρ_a(p_o,T_o), ρ_b(p_o,T_o))`. Because `alpha_from_mass_fraction` inverts `1/ρ = Y/ρ_a + (1−Y)/ρ_b`, this composition is an identity:

> **`rho_o[i] ≡ ρ_mix(Y_new[i], p_o[i], T_o[i])` — the exact PTE mixture density at the old `(p,T)` with the new composition.**

So `M_reb == M_prev` iff, cell-wise to first order, `ρ_mix(Y_new, p_o, T_o) ≈ s0.rho[i]`. That holds **iff the previous accepted state was itself in PTE at `(p_o,T_o)` with `Y_old`**, plus an `O(ΔY)` term that is conservative because `ρY` transport is conservative.

- **Config C** — `compute_R` (`acid.cpp:1563-1571`) re-derives `alpha` from `Y` at the *current* `(p,T)` on **every** residual evaluation. The accepted state is therefore PTE-consistent by construction. Next step's recovery at `(p_o,T_o)` reproduces the same alpha, so `rho_o ≈ s0.rho`. **Cancellation is a structural property of C, not a coincidence.**
- **B + RECON** — `acid.cpp:926-985` re-solves `(p,T,alpha)` from the conserved `(ρ,e,Y)` once per step *before* the `s0` snapshot (`:1043`), so the step *starts* PTE-consistent. Same conclusion, obtained at the step boundary rather than at every iterate.
- **B + F3** — a different route: F3 recovers alpha at the cell's own `(p*,T*)` (from `pT_from_v_e_massfrac`, `acid.cpp:1330-1339`), for which `ρ_mix(Y,p*,T*) = s.rho[i]` exactly by construction of the inversion. But the rebuild at `:1371-1372` still evaluates phase densities at `(p_o,T_o)`, i.e. the alpha and the `(p,T)` no longer form one triple — hence `remap ≈ −0.01` (small) but `RES = 197.74` (huge): F3 moves the inconsistency from the rebuild into the accepted continuity residual.
- **Plain B** — the accepted state has `alpha` frozen at the *start-of-step* recovery while `(p,T)` moved freely inside Newton. Nothing enforces PTE at the step boundary. In the well-conditioned regime the mismatch is `O(dt)` and harmless; **at `p_o = 1 Pa` it is `O(1)`**, because `ρ_air(1 Pa, ~298 K) ≈ 1.2e−5` makes `alpha_from_mass_fraction` saturate to ≈1 for *any* non-trivial air mass fraction, so `ρ_mix(Y, 1 Pa, T) ≈ 1e−2` against a true cell mass of hundreds of kg/m³.

**This fully answers the briefing's direction (1).** It also answers its sub-question: yes, there is a "cheaper way to get C's cancellation property" — but every such way (RECON, F3, a hypothetical rebuild-at-`(p*,T*)` = round 25 §8's unbuilt "F3b") only repairs the *bookkeeping*, and §2.3 shows the bookkeeping is not the disease.

### 2.3 Step three — the disease: plain B's discrete continuity is **infeasible** in a cavitating cell

Measured live (`ACID_RHIST=1 ACID_BLK_STEP=<s> ACID_TEND_SCALE=0.05`, case15, `denner1d_dump`):

| config | step 0 residual trajectory (`n0 → …`) | line-search `al` | verdict |
|---|---|---|---|
| **B** | `5.9172e7 → 5.1011e7 → 5.0925e7 → 5.0924e7 → 5.0923e7 → 5.0923e7 → 5.0925e7 ↑` | `1.0, 0.25, 0.125, 0.016, 0.016, …` | **stalls at 86 % of `r_init`; residual starts *rising*** |
| **C** | `5.9172e7 → 4.8470e7 → 3.1795e7 → 5.9878e6 → 6.2798e5 → 5.5018e3 → 4.38e−1 → 2.40e−7` | `1.0` throughout | **converged (quadratic), 7 its** |
| **B+F3** | `5.9172e7 → 5.1011e7 → 5.0925e7 → 5.0924e7 → 5.0923e7 …` | `1.0, 0.25, 0.125, 0.016, …` | **identical to B, digit for digit** |

The `dp` column under B is `5.00e4, 2.50e4, 1.25e4, 6.25e3, 3.85e3, …` — the existing `|dp| ≤ 0.5·p` limiter (`acid.cpp:2382`) binding on iteration after iteration, i.e. Newton demanding an unboundedly negative pressure. Step 1 shows the same signature with `al = 1.0` and geometric `dp` halving to a limit while the residual freezes at `1.0431e7` (68 % of `r_init`).

**Closed-form confirmation.** case15's IC (`cases.cpp:682-689`): `α = 0.055` air in water, `p = 1e5`, `u = ∓100 m/s`, `ρ_mix ≈ 945.07` (matches round 27's measured `Σρdx = 945.07`). Wood mixture sound speed ≈ 54 m/s ⇒ a Mach-1.9 rarefaction. Over the first step the central cells must lose `Δρ = ρ·(∂u/∂x)·dt ≈ 6·10² kg/m³`. At frozen `α`, `∂ρ/∂p ≈ 5·10⁻⁶ kg m⁻³ Pa⁻¹`, so the required pressure drop is `≈ 1.2·10⁸ Pa` **below zero**. No admissible `p` exists. The scheme's only options are to clamp at 1 Pa and accept a large residual — which is exactly what it does.

The `ACID_TEND_SCALE` trace makes the same point in the state variables:

```
sigma=0.01 [B] M=949.756 min_p=1.2500e+04 alpha=[0.0550,0.0550] nfloor=0
sigma=0.01 [C] M=948.056 min_p=1.2675e+04 alpha=[0.0550,0.3147] nfloor=0
sigma=1.00 [B] M=  0.761 min_p=1.0000e+00 alpha=[0.9910,0.9998] nfloor=322
sigma=1.00 [C] M=870.610 min_p=3.2432e+00 alpha=[0.0550,0.9994] nfloor=0
```

and the per-cell dump at `sigma=0.01`: **B** has `p = 1.2500e+04` (= `1e5·0.5³`, the clamp signature) spread over ~40 cells with `alpha ≡ 0.055000` and `rho ≡ 949.4` *unchanged*; **C** has the expansion confined to 2 cells with `alpha = 0.314676`, `rho = 688.46`.

### 2.4 Step four — the causal chain, and what it retires

```
frozen alpha inside Newton (plain B)
  → mixture compressibility understated by ~5·10²  (acid.cpp:2117's own "factor 521.56")
  → continuity infeasible in the expanding core; Newton stalls at 86% of r_init
  → p pinned at the 1.0 Pa clamp (acid.cpp:2390)
  → next step's recovery (acid.cpp:1310) reads alpha_from_mass_fraction(Y, ρ_a(1 Pa,T), ρ_b(1 Pa,T)) → alpha ≈ 1
  → Eqs.43-44 rebuild (acid.cpp:1373) sets rho_o ≈ 1e−2, i.e. M_reb ≪ M_prev
  → a *converged-looking* implicit step faithfully propagates the deleted mass
  → neighbours flux into the vacuum, the floor spreads: 322/400 cells, 85 steps, ΣρΔx 945.07 → 0.761
```

Everything from line 4 down is **downstream**. That retires, on evidence rather than assertion:

- `ACID_YADV_REBUILD_ADV` (round 27) — already reverted;
- **F3b / "same-triple restoration"** (round 25 plan §8, never built, and the candidate I would otherwise have proposed): it would set `M_reb ≡ M_prev` *exactly* and drive `RES` down, but B+F3 already shows the endpoint of that road — `nfloor = 400/400`, the entire domain at the floor, `l2_p = 0.00000` only because the N=800 reference is *equally* collapsed. Fixing the bookkeeping of a garbage state yields a well-audited garbage state. **I explicitly decline to propose F3b this round and record why.**
- any Jacobian-only change (e.g. starring `D_p` under plain B without touching the residual): by defect-correction the residual is the single source of truth (`acid.cpp:2079-2081`), so for converging cases the answer is unchanged, and for case15 the residual remains infeasible — a softer Jacobian would only change *which* non-converged iterate is accepted. That is a fudge, not a fix, and I decline it too.

### 2.5 A correction to how the target has been framed

`M_ref` in `scripts/yadv_r27_case15.py`'s own output shows the N=800 reference is computed under the *same* config (`cases.cpp:751-752`). Measured:

```
plain B   l2_rho=0.16761 corr_rho=0.984514 l2_p=0.16653 M=  0.761 M_ref=  2.441 nfloor=322
B+F3      l2_rho=0.10300 corr_rho=0.988633 l2_p=0.00000 M=961.976 M_ref=956.259 nfloor=400
B+RECON   l2_rho=0.08293 corr_rho=0.986733 l2_p=0.18298 M=965.269 M_ref=962.016 nfloor=  0
B+RESYNC  l2_rho=0.17684 corr_rho=0.633657 l2_p=0.00000 M=999.150 M_ref=1002.014 nfloor=400
B+HREINIT l2_rho=0.16761 corr_rho=0.984514 l2_p=0.16653 M=  0.761 M_ref=  2.441 nfloor=322
C         l2_rho=0.01966 corr_rho=0.996734 l2_p=0.01439 M=870.610 M_ref=865.543 nfloor=  0
```

Two things follow that round 27 did not state: (a) under plain B the *reference is collapsed too* (`M_ref = 2.441`), so `l2_rho = 0.168` is a comparison of two collapsed runs and the failure is really "the collapse is not grid-convergent"; (b) **B+F3's and B+RESYNC's `l2_p = 0.00000, corr_p = 1.000000` are degenerate**, both solution and reference being uniformly at 1 Pa — they must not be read as accuracy. Round 27's §3.5 characterisation of B+F3 as "very close on `corr_rho`" is technically true but materially misleading; that is corrected here.

---

## 3. Stage-2 candidate: `ACID_YADV_ALPHA_IMPLICIT_CAV`

### 3.1 Derivation

The 4-equation model is a **full pressure–temperature-equilibrium** model: `alpha` is not an independent variable, it is *slaved* to `(Y, p, T)` by `1/ρ = Y/ρ_a(p,T) + (1−Y)/ρ_b(p,T)`. Config C implements that definition inside the residual. Plain B implements a *lagged* approximation to it (alpha evaluated once per step at `(p_o,T_o)`), which is uniformly accurate wherever `p` moves little within a step — and which becomes **not merely inaccurate but insoluble** wherever the required volume change exceeds what the frozen composition can supply at any admissible pressure.

So the correct scope of the correction is not "everywhere" (config C, which costs case14) and not "at the recovery site" (F3/F3b, which is downstream) but exactly: **the cells where the lagged closure has no solution.** The exact, constant-free statement of "no solution" available at the line-search site is:

> **the full Newton step requests a pressure at or below the solver's own hard floor:**
> `sbak.p[i] + om * dxk[i][1] <= 1.0`

The literal `1.0` is the floor already written two lines below at `acid.cpp:2390` (`s.p[i] = std::max(sbak.p[i] + dpi, 1.0)`). **No new constant is introduced.** The predicate does not tune anything: it fires precisely when the linearised frozen-alpha model demands a state the model cannot represent. Precedent for an exact clamp-membership test as a physics predicate already exists in this file — `ACID_TSAT` block A, `acid.cpp:1591-1593`: *"`s.T[i]>=1e6` is an EXACT bit-level test for 'this cell sits at the clamp, not a solution of hmix(T)=hstat'"*.

Physically the predicate says: *this cell has reached the vacuum limit, where pressure is bounded and only the volume fraction is free* — which is exactly when a PTE mixture must let `alpha` absorb the volume change.

### 3.2 Exact code-change spec

**Flag declaration** — insert immediately after the `ACID_YADV_ALPHA_IMPLICIT` declaration block (`acid.cpp:585-593`), with a full rationale comment in the file's established style (mechanism, why the predicate is constant-free, the C-endpoint bound, the case14 risk, a pointer to this plan):

```cpp
const bool alpha_implicit_cav = std::getenv("ACID_YADV_ALPHA_IMPLICIT_CAV") != nullptr;
// RESEARCH-ONLY, default OFF, inert unless ACID_YADV. Byte-identical no-op when
// ACID_YADV_ALPHA_IMPLICIT is also set (that flag already forces the same branch for
// every cell) -- configs C/D/F are untouched by construction.
```
Plus a paired diagnostic `const bool cav_dbg = std::getenv("ACID_NFEAS") != nullptr;` (stderr only, applies nothing) declared alongside it.

**(a) The mask.** Declare inside the retry body, *before* `compute_R`'s definition so the lambda's `[&]` capture sees it, i.e. immediately before `acid.cpp:1543`:

```cpp
// Round 28: per-cell "frozen-alpha continuity is infeasible here" mask. Reset at every
// retry restart (this declaration is inside the retry body), monotone within a retry.
std::vector<char> cav(n, 0);
int cav_n = 0;                       // count, for ACID_NFEAS
```

**(b) Consume it in the residual.** `acid.cpp:1563-1571` becomes:

```cpp
if (yadv && (alpha_implicit || cav_n > 0)) {
    for (int i = 0; i < n; ++i) {
        if (!(alpha_implicit || cav[i])) continue;
        const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
        s.alpha[i] = std::clamp(alpha_from_mass_fraction(Yv[i],
                                    phase_props(pu, Tu, A).rho,
                                    phase_props(pu, Tu, B).rho), 0.0, 1.0);
    }
}
```
With `alpha_implicit` set, this is the identical loop over all cells → **bitwise identical to today for configs C/D/F.** With `cav_n == 0` the outer test is false → **bitwise identical to today for plain B/G.**

**(c) Consume it in the Jacobian.** `acid.cpp:2137` becomes
```cpp
const bool aimp = yadv && (alpha_implicit || cav[i]);
```
(`aimpT` at `:2139` stays `aimp && alpha_implicit_t`, so `IMPLICIT_T` is *not* picked up by CAV — matching config C, which does not set it.) Nothing else in that block changes; the starred `D_ps/N_ps/D_Ts/N_Ts` at `:2139-2148` and `alp_p` at `:2155` are the existing, round-5-verified expressions. This is required for consistency: the file's own comment at `:2110-2117` states that when the residual re-derives alpha the frozen-alpha `D_p` is "the wrong derivative of the map compute_R actually evaluates", with the measured factor 521.56.

**(d) Set the mask — placement is load-bearing.** Insert **after** the line search closes (i.e. after `acid.cpp:2397`, before `backtracked_last` at `:2398`):

```cpp
if (yadv && (alpha_implicit_cav || cav_dbg)) {
    int grew = 0;
    for (int i = 0; i < n; ++i) {
        // exact, constant-free: the FULL Newton step asks for a pressure the model's own
        // floor (two lines above, acid.cpp:2390) cannot represent -> the frozen-alpha
        // closure has no admissible solution in this cell.
        if (!cav[i] && sbak.p[i] + om * dxk[i][1] <= 1.0) { cav[i] = 1; ++cav_n; ++grew; }
    }
    if (grew && alpha_implicit_cav && ajac) {   // the residual function just changed:
        rbest = std::numeric_limits<double>::max();  // do not compare rnorm3 across two
        best_it = it;                                // different residuals (acid.cpp:2411/2414)
    }
    if (cav_dbg) { /* stderr: case, step, retry, it, cav_n, grew, first cell, r_init, rnorm3() */ }
}
```

**Why *after* the line search, not before:** `n0` is taken at `acid.cpp:2373` from the residual evaluated with the mask as of the previous iteration. If the mask grew before the trial loop, the line search's `rnorm3() < n0` test at `:2396` would compare two *different* residual functions and could accept a non-descent step. Updating at the end means the next iteration's own `compute_R()` at `:1927` re-establishes a consistent baseline. It also avoids any extra `compute_R()` call — round 27's non-idempotency trap (`acid.cpp:2644-2651`) is structurally impossible here.

**Why the mask is monotone and retry-scoped:** monotone ⇒ the residual function is piecewise-constant in `it` and settles after 1-2 iterations (no oscillation between closures). Retry-scoped ⇒ `s = s0` (`acid.cpp:1074`) and a fresh `cav` go together, so a halved-dt retry gets a clean re-evaluation.

**Scope note:** the mask is set only on the coupled 3×3 path. The segregated 2×2 line search (`acid.cpp:2570-2583`) is left alone; on that path `cav` stays all-zero and everything is byte-identical. The round-0 census (§5, T0) confirms every one of the 19 cases exercises the 3×3 path, so this costs no coverage.

**Diff surface: 1 flag pair + 1 vector + 3 small edits, all in `acid.cpp`; plus 2 names appended to `ACID_ENV_VARS` in `scripts/yadv_r9_sweep.py:33-41` and `scripts/yadv_r27_case15.py:22-30`.** No new numeric literal anywhere.

### 3.3 Why this is not `REBUILD_ADV` in disguise — the concrete mechanism difference

The briefing demands this be stated concretely, not vaguely.

`REBUILD_ADV` built `rho_o/hstat_o/Htot_o` from `alpha_reb = s0.alpha + dal_adv`, an alpha **different from the `s.alpha` used by every other consumer in the same step** — the Eqs.41-42 mass-flux blend weight (`acid.cpp:1288-1289`), the new-level mixture density in `eval_thermo` (`acid.cpp:311-315`), and the enthalpy-flux blend (`acid.cpp:2588-2591`) all keep reading `s.alpha`. The implicit continuity residual then reads `(ρ(α_new) − ρ_o(α_reb))/dt + div(mdot(α_new))`: an `O(1)` mismatch between the old level and the new level, injected as a source that Newton has no way to reject. That is precisely the class of error round 13's `RMISM` note warned about and it explains 4 unrelated cases NaN-ing.

`ACID_YADV_ALPHA_IMPLICIT_CAV` **never touches the old level at all** (`acid.cpp:1361-1377` is untouched), never touches `Yv`, never touches `s.p`/`s.T`/`s.h`, and never introduces a second alpha. It changes *when* the single `s.alpha` is evaluated — from "once at `(p_o,T_o)`" to "at the current iterate" — for a subset of cells. Old level, new level, mass flux and enthalpy flux all continue to see one and the same `s.alpha` at every point in the step. There is no consistency to break.

Second, and decisively different from round 27: **the candidate's behaviour is bracketed by two already-measured configurations.** With the predicate never firing it *is* plain B (15/19); with it always firing it *is* config C (14/19, all finite, no NaN). `REBUILD_ADV` created a state no prior round had ever run. The mixed regime is not strictly an interpolation of the two, and I do not claim it is — but the worst credible outcome is bounded by a config the suite has run every round for 24 rounds, which is a categorically stronger prior than "unknown".

### 3.4 Blast-radius analysis — **done before proposing, as the briefing requires**

**Which cases touch the modified code path?** All 19 (`compute_R` and the 3×3 line search are the common path). So the correct question is: *in which cases does the predicate fire?* Evidence gathered this round, before proposing:

1. **Floor census at `t_end`, all 19 cases, plain B** (`denner1d_dump`, count of `p ≤ 1.0`):

| case | 01 | 02 | 04 | 05 | 07 | 13 | 14 | **15** | 24 | 25 | 26 | 27 | 28 | 30 | 31 | 33 | 34 | 35 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `nfloor` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **322/400** | (NaN) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | (NaN) | 0 | 0 |
| `min_p` | 1e5 | 1e5 | 1e5 | 8.7e4 | 1e5 | **1e4** | 1e5 | **1.0** | nan | 1e5 | 1e5 | 1e5 | 1e5 | 1.01e5 | 1.01e5 | 1e5 | nan | 1e5 | 1e5 |

**case15 is the only case in the suite that reaches the pressure floor.** The runner-up (case13) is four orders of magnitude away.

2. **Newton census at step 0, all 19 cases, plain B** (`ACID_RHIST`, `ACID_TEND_SCALE=0.02`). Cases 13/14/24/25/26/27/28/33/34 all show `al = 0.016` line-search collapse at their *initial-discontinuity* step, with raw `|dp|` up to `1.83e8`. **This is the honest residual risk**: a raw Newton step of that magnitude at a cell sitting at `p = 1e5` can transiently request `p < 1` even though the case is nowhere near cavitating. Cases 01/02/04/05/07/30/31/35/36 converge cleanly at step 0 (`al = 1.0`, residual to `1e−8`) and are essentially risk-free.

3. **Direction of the risk.** Because the candidate's endpoints are B and C, a case can only be harmed in a way C is harmed. Comparing `EXPECTED` (`scripts/yadv_r9_sweep.py:56-64`): `B` fails `{15,24,33,34}`, `C` fails `{14,15,24,33,34}`. **Exactly one case — case14 — is at risk of a pass→fail flip.** Measured this round, case14 under C is `pass:false, finite:true`, degrading on velocity only (`l2_u 0.0838 → 0.1324`, `corr_u 0.9816 → 0.9543`, `amp_ratio_u 1.0097 → 1.1221`) while density *improves* (`l2_rho 0.0769 → 0.0642`). It is an accuracy regression at a near-pure-phase interface (`alpha = 1e−6 / 1−1e−6`, `cases.cpp:674`), **not** a divergence.

4. **Cases 07/13/25 — the rest of the `REBUILD_ADV` cohort.** All pass under both B and C, so neither endpoint harms them; the concrete failure mechanism that broke them (a second, inconsistent alpha in the old level) does not exist here. They are nonetheless checked first in Stage 2.0 (§4), because "should not be affected" is exactly what round 27 assumed.

5. **OFF path.** `yadv == false` ⇒ the mask block never executes and `cav` stays zero-initialised. G1 byte-identity holds by construction, not by measurement.

**Pre-registered fallback if the predicate proves too broad.** If T1 (§4) shows the predicate firing in cases whose `min_p` never approaches the floor — i.e. one-off initial-discontinuity overshoots — switch to predicate **P2**: fire only when `sbak.p[i] + om*dxk[i][1] <= 1.0` holds on **two consecutive Newton iterations** of the same retry. Still exact, still constant-free (a persistence test, not a threshold), and case15's cells satisfy it trivially (they fail persistently for 85 consecutive steps). **The choice between P1 and P2 must be made from T1's firing census alone, and fixed in writing, before any case15 metric is measured under the applying flag** — it is a blast-radius decision, not a tuning decision, and must not be made by trying both against the gate.

---

## 4. Staging — with the cheap G1/G2-first insurance built in

### Stage 0 — reproduce the diagnosis (no code)
- **T0.** Re-run the §2.3 residual traces and confirm the three trajectories (B stalls at 86 %, C converges to 2.4e−7, B+F3 identical to B). ~2 minutes, zero risk. If T0 does not reproduce, **stop and re-plan** — the whole round rests on it.
- **T0b.** Re-run `python3 scripts/yadv_r27_case15.py overlays` and `tend`; confirm the tables in §2.5 and §2.3.

### Stage 1 — the diagnostic flag `ACID_NFEAS` only (applies nothing)
Implement **only** the mask computation + the stderr report (§3.2(a) and (d) with the `alpha_implicit_cav` consumer edits (b)/(c) **not yet written**, or written but unreachable because the flag is unset).
- **G4-early.** `ACID_NFEAS=1` must leave `denner1d_dump` stdout **byte-identical** on cases 01/13/15/24. (Round 27's lesson: an instrument that calls `compute_R()` is not idempotent under `ACID_YADV`. This one calls nothing — verify, do not assume.)
- **T1 (the blast-radius census).** Run all 19 cases under plain B with `ACID_NFEAS=1`. Record per case: first step at which the mask fires, peak `cav_n`, total steps with `cav_n > 0`, and whether firing is transient (dies within the step) or persistent.
- **T1 decision.** Choose P1 or P2 per §3.4(5) and **write the choice down in `YADV_RESEARCH.md` §38 before Stage 2 runs.**

### Stage 2 — the applying flag, **harm check first**
- **Stage 2.0 — the insurance check, ahead of every case15 metric.** In this exact order:
  1. **G1** — `python3 scripts/yadv_r9_sweep.py --verify`: OFF path byte-identical vs the published `solver_denner` binary, 9/9. *(flag unset)*
  2. **G2** — `python3 scripts/yadv_r9_sweep.py --sweep`: `ALL GATES OK`, configs A–G exactly at `EXPECTED`. *(flag unset)*
  3. **G4** — `ACID_YADV_ALPHA_IMPLICIT_CAV=1` set **together with** `ACID_YADV_ALPHA_IMPLICIT=1`: stdout must be byte-identical to `ACID_YADV_ALPHA_IMPLICIT=1` alone on cases 01/13/14/15/24 (the §3.2(b)/(c) no-op-under-C property, verified, not asserted).
  4. **The harm gate.** Full 19-case `denner1d_validate` under `B + CAV`. Read **only** `pass_count` and the fail set. **Do not look at any case15 metric yet.**
     - If `pass_count < 15`, or any case in `{01,02,04,05,07,13,14,25,26,27,28,30,31,35,36}` flips pass→fail, or any previously-finite case goes NaN → **S5-early fires immediately** (§5). Stop, revert, write up. Do not spend the round diagnosing why.
- **Stage 2.1 — only if 2.0 is clean.** case15 under `B + CAV`:
  - `python3 scripts/yadv_r27_case15.py overlays` extended with the `B+CAV` combo (append one entry to `combos`, `scripts/yadv_r27_case15.py:158-165`): `l2_rho, corr_rho, l2_p, corr_p, M, M_ref, nfloor`.
  - Per-predicate gate + `cj/mj/cc` via the same script's `case15_gate`.
  - `ACID_MBAL=1` budget: `Σadv`, `Σremap`, `Σbnd`, `Σleak`, `Σres`, `closure`.
  - `ACID_TEND_SCALE` trace (`tend` mode, add the `B+CAV` config) for the mass-vs-time and `nfloor`-vs-time curves.
- **Stage 2.2 — the cross-check that the mechanism, not luck, did the work.** `ACID_NFEAS` on the `B+CAV` run: confirm `cav_n` rises exactly when the old B run's floor cells would have appeared, and that `nfloor` in the final state is ≈0.

### Stage 3 — write-up
`YADV_RESEARCH.md` §38, `YADV_ROADMAP.md` control state, this plan committed as `docs/YADV_ROUND_28_PLAN.md`.

---

## 5. Gates, targets, and execution order

### Hard gates (all must pass before merge; non-negotiable)

| gate | command | pass criterion |
|---|---|---|
| **G1** | `python3 scripts/yadv_r9_sweep.py --verify` | OFF path byte-identical vs published `solver_denner`, 9/9; `ACID_YADV=1` vs unset on case01 byte-identical |
| **G2** | `python3 scripts/yadv_r9_sweep.py --sweep` | `ALL GATES OK`; A–G exactly `EXPECTED` (`B`=15, `C`=14, `G`=15, fail sets unchanged) |
| **G3** | `./build-cpp/cpp/denner_1d/denner1d_unit` | clean; `pT_from_v_e_massfrac` worst `rel_p` unchanged (`4.7e-11`) |
| **G4** | new-flag no-op | (a) `ACID_NFEAS=1` alone: `denner1d_dump` stdout byte-identical on 01/13/15/24. (b) `ACID_YADV_ALPHA_IMPLICIT_CAV=1` **with** `ACID_YADV_ALPHA_IMPLICIT=1`: byte-identical to C alone on 01/13/14/15/24. (c) Both new vars are in `ACID_ENV_VARS` in *both* scripts, so G1/G2 purge them |
| **G5** | `git diff --stat` | only `cpp/denner_1d/src/acid.cpp`, `scripts/yadv_r9_sweep.py`, `scripts/yadv_r27_case15.py`, `docs/*`. **Zero new numeric literals.** No edits to `cases.cpp`/`validation.cpp`. No edit to any `1.0` pressure-floor site (`acid.cpp:2390`, `acid.cpp:2582`, `solver.cpp:refresh_thermo`) |

### Round-specific targets (pre-registered, in priority order)

| id | target | threshold |
|---|---|---|
| **R1** | no suite regression under `B+CAV` | `pass_count ≥ 15`; fail set ⊆ `{15,24,33,34}` |
| **R2** | the mass collapse is removed on case15 | final `Σρdx ≥ 800` (vs `0.761` today; C reaches `870.6`) **and** `nfloor ≤ 5/400` (vs `322`) |
| **R3** | the rebuild becomes mass-preserving | `\|Σadv + Σremap\| / \|Σremap\| < 0.05` under `ACID_MBAL` (the `M_reb ≈ M_prev` signature C and RECON have), with `closure ≲ 1e-12` relative |
| **R4** | accuracy metrics clear their bars | `l2_rho ≤ 0.05` **and** `corr_rho ≥ 0.99` (C achieves `0.0197 / 0.9967`) |
| **R5** | (stretch, not expected) | case15 `pass:true`, i.e. R4 **and** `cj ≤ max(8.0, 1.10·cj_r)` |

### Execution order (strict)
`T0 → T0b → Stage 1 (G4-early, T1, predicate choice recorded) → Stage 2.0 (G1, G2, G4, harm gate) → Stage 2.1 → Stage 2.2 → G3, G5 → write-up.`
**The harm gate precedes every case15 measurement.** This is the round's single most important procedural rule.

---

## 6. Pre-registered stop / decision rules

- **S1 — full success.** R1 ∧ R2 ∧ R3 ∧ R4 ∧ R5: case15 passes under `B+CAV`, `pass_count ≥ 16/19`, all hard gates green. → Keep the flag; recommend a promotion evaluation (fold into `ACID_YADV`) as round 29's explicit, separately-gated decision — **do not promote it inside this round.** `consecutive_failures → 0`.
- **S2 — substantial partial (the expected outcome).** R1 ∧ R2 ∧ R3 ∧ R4 but not R5 (case15 still fails on `cj` alone). → Commit `ACID_NFEAS` + `ACID_YADV_ALPHA_IMPLICIT_CAV` as gated-off research infrastructure (same precedent as `RECON`/`RESYNC`/`F3`/`RECON_NULL`/`MBAL`). Record that case15's mass-collapse blocker is *solved* and the remaining blocker is the already-characterised `cj = 30` core-jet, handed to round 29 as the sole live thread. **`consecutive_failures` NOT incremented** (rounds 13/16/19/21/22/23/24/25 precedent).
- **S3 — diagnostic-only success.** Stage 1 completes cleanly but T1's census shows the predicate cannot be scoped (fires broadly under both P1 and P2), so Stage 2 is **not run**. → Commit `ACID_NFEAS` only, plus §2's diagnosis (which stands on zero-code measurements and is this round's real deliverable regardless). **`consecutive_failures` NOT incremented** — round 27's own S2 rule, invoked as the briefing permits.
- **S4 — neutral.** R1 holds, R2 holds, but R3/R4 fail (mass restored, accuracy still short). → Keep gated off, report honestly as "the collapse is removed and the accuracy blocker is a distinct, unidentified defect". `consecutive_failures` NOT incremented.
- **S5 — harm, caught early (the anti-round-27 rule).** At the **Stage 2.0 harm gate**: `pass_count < 15`, OR any case in `{01,02,04,05,07,13,14,25,26,27,28,30,31,35,36}` flips pass→fail, OR any previously-finite case becomes NaN.
  → **Stop the round's investigation immediately.** Do not proceed to Stage 2.1. Do not attempt a rescue. Revert the *applying* flag (`ACID_YADV_ALPHA_IMPLICIT_CAV` and its consumers, edits (b)/(c) and the `alpha_implicit_cav` half of (d)) **in full**; keep `ACID_NFEAS` (a pure instrument that passed G4). Record the exact failing case list and the T1 firing census for that case, which is the diagnosis. **`consecutive_failures` += 1.**
  → *One narrowly-scoped exception, and only one:* if the sole regression is **case14** and P2 was not the predicate used, P2 may be tried **once** — because P2 was pre-registered in §3.4(5) before any outcome was seen, and case14 was named in advance as the single at-risk case. If P2 also regresses case14, S5 fires as written with no further attempts. Any regression outside case14 admits **no** retry.
- **S6 — hard-gate failure.** Any of G1/G2/G3/G5 fails. → Revert everything, `consecutive_failures` += 1, unless the failure is a trivial hygiene omission (a missing `ACID_ENV_VARS` entry) fixable in one line, in which case fix and re-run the gate.
- **S7 — diagnosis does not reproduce.** T0 fails to reproduce §2.3's three residual trajectories. → Stop; do not implement anything; report the discrepancy as the round's finding. `consecutive_failures` NOT incremented (a falsified prior is information).

**Anti-rescue clause (explicit).** If S5 fires, the round is over for that candidate. No re-scoping of the predicate beyond the single pre-registered P2 exception, no per-case exclusion, no threshold, no "try it with `IMPLICIT_T` as well", no dt/CFL adjustment, no relaxation of any gate threshold. A second severe regression in a row is exactly the outcome the loop's stop condition exists to catch, and it must be reported as such rather than engineered around.

**Reporting discipline.** Every measured number goes into the write-up as measured, including ones that contradict this plan's own predictions (round 27 §37.4's precedent). In particular, if `B+CAV` reproduces config C's case15 *and* its `cj = 30`, that must be stated as "this candidate does not make case15 pass", not softened.

---

## 7. Non-goals — explicitly not attempted this round

1. **Config C's `cj = 30` central-jump defect.** A distinct, already-characterised failure (round 27 §4.5: under-resolved near-vacuum core at the stagnation point, 4-cell velocity sign reversal, `p` dropping 137× across one cell; resolved at the reference's N=800). Out of scope; handed to round 29.
2. **Any change to the 1.0 Pa pressure floor** (`acid.cpp:2390`, `acid.cpp:2582`, `solver.cpp:refresh_thermo`, and the `std::max(p,1.0)` EOS guards at `acid.cpp:311/336`). Forbidden by round 27's non-goals and by this plan. §2 shows the floor is the *trigger*; the candidate makes the solver stop demanding sub-floor pressures rather than letting it have them.
3. **Cases 24/33/34.** Closed by round 26 as a closure mismatch, not a numerical one. They appear in this round only as unchanged members of the fail set.
4. **Promoting anything to a default.** `ACID_YADV`'s recommended status stays OFF this round regardless of outcome. Even under S1, promotion is a separate, separately-gated round-29 decision.
5. **`ACID_YADV_REBUILD_ADV` or any variant of it.** Confirmed absent from the code; not re-proposed.
6. **F3b / "same-triple restoration"** (round 25 plan §8, evaluating Eqs.43-44 at `(p*,T*)`). Deliberately declined, with the reason recorded in §2.4: B+F3 already demonstrates the endpoint of that road (`nfloor = 400/400`, degenerate `l2_p = 0`). Its trigger condition from round 25 remains unmet and is now argued to be unmeetable.
7. **A Jacobian-only correction under plain B** (starring `D_p` without changing the residual). Declined, with reason, in §2.4.
8. **The segregated 2×2 solver path** (`acid.cpp:2520-2600`) and TR-BDF2. Untouched; `cav` stays zero there.
9. **`ACID_YADV_RESYNC`, `ACID_PROJ_UNTIL`, `ACID_RECON_NULL`, `ACID_YADV_HREINIT`** and every other existing research flag. Untouched.
10. **The `alpha_implicit_t` (T-pathway) starring.** Deliberately excluded from the CAV gate so the candidate matches config C exactly, not config F.
11. **Any per-case coefficient, threshold, or tuning constant.** The candidate introduces zero numeric literals; its only constant is the floor value already present two source lines from the predicate.

---

## 8. Literature

**Checked first, as instructed:** `papers/*.md` (51 summaries), `papers/md/**` (93 full texts), `papers/library/index.jsonl` (**empty, 0 bytes — the library index is not populated; `papers/library/md/` does not exist**). No duplication of existing coverage below.

**Already in the repo — read this round, directly load-bearing:**
- `papers/md/33_saurel_relaxation_multiphase.md` — **Saurel, Petitpas & Berry (2009), JCP 228:1678-1712.** This is one of the four titles round 27 flagged as a `_needed.md` stub; **it is already present as full text** (no summary file exists). §3.3 "Relaxation step" (lines 1088-1135) is the key contrast: the standard stiff pressure-relaxation step evolves `∂α₁/∂t = μ(p₁−p₂)` while holding **`∂(α_k ρ_k)/∂t = 0`, `∂(ρu)/∂t = 0`, `∂(ρE)/∂t = 0`** — i.e. *partial masses and total energy are exactly conserved through the relaxation, and only volume is redistributed.* ACID's Eqs.43-44 rebuild (`acid.cpp:1361-1377`) does the mirror image: it holds `(p_o,T_o)` fixed and re-derives `ρ_o` from the new `α`, so partial masses are **not** conserved through the remap. In the well-conditioned regime the difference is `O(dt)`; at the vacuum limit it is `O(1)`. The paper also explicitly motivates the whole approach by "dynamic appearance of interfaces … in cavitating flows" and by *volume-fraction positivity under strong expansion waves* (lines 139, 292, 342) — case15's exact regime.
- `papers/md/19_collis_2025_robust_4eq.md` — **Collis, Bezgin, Mirjalili & Mani (2025), "A robust four-equation model…"**, the closest published analogue of this solver's model. Line 1392 states plainly: *"Physical situations like cavitation can result in negative pressures, and the NASG equation of state used in this work permits this"*, and gives the admissibility conditions as positivity of **mixture density and internal energy (hence temperature)** — pressure is a derived quantity that may legitimately be negative. Corroborates §2's reading that a hard `p ≥ 1 Pa` clamp is not the model's natural invariant, while also confirming (per this round's non-goals) that the right response is not to change the clamp but to stop the discretisation from demanding sub-floor states.
- `papers/md/denner2020_jcp409_conservative_allspeed.md` — **Denner, Evrard & van Wachem (2020), JCP 409**, the pressure-based algorithm this solver descends from. Appendix A gives the discretised continuity equation's pressure coefficient as `A^{ρ,p}_P = C·V_P/[(γ−1)c_v T + b(p+Π)]/Δt + …` — exactly the single-phase NASG `∂ρ/∂p|_T` that `acid.cpp:326` blends with a **frozen** `α`. The mixture extension (a term in `∂α/∂p`) is not in the parent paper; it was never inherited, and its absence is precisely the 521× defect at `acid.cpp:2117`. Useful provenance for why the defect exists at all.
- `papers/md/12_murrone_2005_five_equation_reduced.md`, `papers/md/26_ten_eikelder_2017_acoustic_convective_kapila.md` — background on the reduced (Kapila) model, where the `α` equation carries an explicit `α(1−α)(ρ_b c_b² − ρ_a c_a²)/(…)·∇·u` compressibility source. `acid.cpp:1082-1084` documents that this solver uses the `K = 0` (Allaire/PE) form, i.e. no such source; under `ACID_YADV` the `α` equation is replaced entirely by `Y` transport + PTE recovery, which is the *right* structure for a 4-equation model — provided the recovery is done at the current state, which is the whole content of §2.

**Could not read — recorded as `_needed.md` candidates (with DOIs, per protocol):**
- `papers/1991_einfeldt_munz_roe_sjogreen_godunov_low_densities_needed.md` — Einfeldt, Munz, Roe & Sjögreen, *"On Godunov-type methods near low densities"*, JCP **92**(2):273-295 (1991), DOI `10.1016/0021-9991(91)90211-3`. The canonical treatment of the vacuum limit and of schemes that cease to admit an admissible state near it — the closest published framing of §2.3's infeasibility result. **Highest priority of the four.**
- `papers/2019_pelanti_shyue_multiphase_cavitation_needed.md` — Pelanti & Shyue, *"A numerical model for multiphase liquid–vapor–gas flows with interfaces and cavitation"*, Int. J. Multiphase Flow **113**:208-230 (2019), DOI `10.1016/j.ijmultiphaseflow.2019.01.010`; and the antecedent Pelanti & Shyue, *"A mixture-energy-consistent six-equation two-phase numerical model for fluids with interfaces, cavitation and evaporation waves"*, JCP **259**:331-357 (2014), DOI `10.1016/j.jcp.2013.12.003`.
- `papers/2018_saurel_pantano_diffuse_interface_review_needed.md` — Saurel & Pantano, *"Diffuse-Interface Capturing Methods for Compressible Two-Phase Flows"*, Annu. Rev. Fluid Mech. **50**:105-130 (2018), DOI `10.1146/annurev-fluid-122316-050109`.
- **Not** to be re-stubbed: `2009_Saurel_Petitpas_Berry` — already present in full at `papers/md/33_saurel_relaxation_multiphase.md`. Round 27's stub list should be corrected accordingly. (Writing a `papers/33_..._summary.md` would be useful but is out of this round's scope.)

Neither `search_semantic` nor `search_arxiv` returned anything on-topic for "implicit pressure-based volume fraction / cavitation Newton coupling" (searches run this round; arXiv results were off-domain). Recorded so a future round does not repeat the same queries.

---

## 9. Reproducing the diagnosis (exact commands, run this round)

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
D=./build-cpp/cpp/denner_1d/denner1d_dump

# 2.3 -- the infeasibility, three configs, same starting residual 5.9172e+07
DENNER_ACID=1 ACID_YADV=1 ACID_RHIST=1 ACID_BLK_STEP=0 ACID_TEND_SCALE=0.05 $D 15 2>&1 >/dev/null | grep RHIST | head -8
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RHIST=1 ACID_BLK_STEP=0 ACID_TEND_SCALE=0.05 $D 15 2>&1 >/dev/null | grep RHIST | head -8
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_F3=1 ACID_RHIST=1 ACID_BLK_STEP=0 ACID_TEND_SCALE=0.05 $D 15 2>&1 >/dev/null | grep RHIST | head -6
#   expect B: stalls 5.0923e7 with al->0.016 ; C: 2.4037e-07 in 7 its ; B+F3: identical to B

# 2.3 -- frozen alpha vs responding alpha at the same early time
DENNER_ACID=1 ACID_YADV=1 ACID_TEND_SCALE=0.01 $D 15 2>/dev/null | awk -F, 'NR==1||NR==182||NR==201'
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_TEND_SCALE=0.01 $D 15 2>/dev/null | awk -F, 'NR==1||NR==182||NR==201'
#   expect B: p=1.2500e4 (=1e5*0.5^3) with alpha=0.055000, rho=949.5, over ~40 cells
#          C: p=1.2675e4 with alpha=0.314676, rho=688.46, confined to 2 cells

# 2.5 -- the overlay table (incl. B+F3's degenerate l2_p=0 at nfloor=400/400)
python3 scripts/yadv_r27_case15.py overlays
python3 scripts/yadv_r27_case15.py tend

# 3.4 -- the floor census (case15 is the only case that ever floors)
for c in 01 02 04 05 07 13 14 15 24 25 26 27 28 30 31 33 34 35 36; do
  DENNER_ACID=1 ACID_YADV=1 $D $c 2>/dev/null | \
  python3 -c "import sys,csv;r=list(csv.DictReader(sys.stdin));p=[float(x['p']) for x in r if 'nan' not in x['p'].lower()];print('$c',sum(1 for v in p if v<=1.0+1e-12),'/',len(r),min(p) if p else 'nan')"
done
```

---

### Critical Files for Implementation
- `cpp/denner_1d/src/acid.cpp` — every code change: flag decls near `:585-593`, `cav` mask before `:1543`, residual consumer at `:1563-1571`, Jacobian `aimp` gate at `:2137`, mask setter after `:2397`
- `scripts/yadv_r9_sweep.py` — `ACID_ENV_VARS` (`:33-41`) hygiene; G1/G2 driver
- `scripts/yadv_r27_case15.py` — `ACID_ENV_VARS` (`:22-30`), add `B+CAV` to `combos` (`:158-165`) and to `tend`'s config list (`:142`)
- `docs/YADV_RESEARCH.md` — new §38
- `docs/YADV_ROADMAP.md` — control state
