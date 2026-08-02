# YADV Round 27 Plan — case15, re-measured: the roadmap's target was half right, and the other half is a mass-conservation defect

**Planner output, 2026-08-02. Grounded in live measurements on `HEAD = b792046` in this worktree, not on round 7's citations.**
**Status of the roadmap's stated goal: CONFIRMED ALIVE, but CONFIG-SCOPED — see §1/§3. Do not start Stage 1 before reading §3.**

**Advisor verification note**: key structural claims checked directly against the code -- `validation.cpp:684-730` (case15 gate: `jump_stats`/`cj`/`mj`/`cc`/`smooth_ok`/`osc_ok` thresholds, exact match), `acid.cpp:1266` (`Yv = anew;`, the recovery-site anchor, exact match), `cases.cpp:751` (`if (c.id == "15")` reference dispatch, exact match). No structural error found.

---

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §37)**: T0 confirmed live
(§3.2's table reproduced exactly). Stage 1's `ACID_MBAL` closes cleanly (`closure~1e-13` at every
step, every config) after a real bug was found and fixed: the extra `compute_R()` call is not
idempotent under `ACID_YADV` (a non-converging T-relaxation) and initially perturbed the reported
solution -- caught by routine G4, fixed with a snapshot/restore around the call. T2's dominant-
channel test fires decisively for `REMAP` (99.67% of B's `|dM|`), but H1 predictions (2)/(3)/(4)
needed a precise correction: `REMAP`'s raw magnitude is large under every config, not small under
the passing ones -- what actually distinguishes B is that `REMAP` goes uncancelled there, while C
and `B+RECON` cancel it exactly against `ADV` (net `0.0000`) and `B+F3` collapses it directly.
S1 fires (remap dominant, clean control contrast once read correctly). Stage 2's one pre-
registered candidate (`ACID_YADV_REBUILD_ADV`) is a severe, unambiguous S5: `pass_count`
`15/19 → 11/19`, four previously-passing cases (07/13/14/25) newly diverge. Per S5's own explicit
instruction (the one rule in this plan that calls for reverting the code rather than keeping it
gated-off), the Stage-2 code was reverted in full; only `ACID_MBAL` merged.
`consecutive_failures` incremented to 1, exactly as S5 specifies -- the first increment since
round 20, honored as pre-registered even though most prior rounds' outcomes did not increment.
Config C's `cj=30` core-jet characterised per §4.5 (an under-resolved near-vacuum core, a
different failure class from the documented MWI checkerboard), not fixed, per the plan's own
non-goal. All hard gates held.

---

## 1. Executive summary

### 1.1 What I did

The roadmap's Phase 3c goal is "case15's central-jump defect (`cj=30.02` vs threshold `8.0`)", carried forward from round 7 (§17.4, 19 rounds old). The prompt told me to re-verify it rather than trust it. I did — across **all seven** sweep configs, computing every predicate of `validation.cpp`'s case15 gate independently in Python from `denner1d_dump` output, plus a domain mass balance and a floor-cell census. The result is not what the roadmap says.

### 1.2 The finding, in one table (all values measured today, `HEAD=b792046`, N=400)

| cfg | env | gate | `l2_p` ≤0.18 | `l2_u` ≤0.06 | `l2_rho` ≤0.05 | `corr_p` ≥0.93 | `corr_u` ≥0.998 | `corr_rho` ≥0.99 | `cj` ≤8.0 | `mj` | `cc` | `smooth_ok` | **Σρdx** | cells at `p=1.0` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | OFF | **PASS** | 0.00000 | 0.02801 | 0.04246 | 1.000000 | 0.998621 | 0.995036 | 6.044 | 6.044 | 0.03449 | True | 967.5 | 400/400 |
| **B** | `YADV=1` | **FAIL** | 0.16653 | 0.01874 | **0.16761** | 0.985535 | 0.999334 | **0.984514** | 2.307 | 2.307 | 0.01292 | True | **0.761** | **322/400** |
| C | `+ALPHA_IMPLICIT` | **FAIL** | 0.01439 | 0.01704 | 0.01966 | 0.999285 | 0.999344 | 0.996734 | **30.018** | **31.998** | **0.11746** | **False** | 870.6 | 0/400 |
| D | C `+NO_AJAC` | FAIL | identical to C to all printed digits | | | | | | 30.018 | 31.998 | 0.11746 | False | 870.6 | 0 |
| E | OFF`+NO_AJAC` | FAIL | 0.00000 | 0.03511 | **0.06995** | 1.000000 | 0.998072 | 0.992092 | 2.826 | 2.826 | 0.02078 | True | 976.6 | 400 |
| F | C `+IMPLICIT_T` | FAIL | identical to C | | | | | | 30.018 | 31.998 | 0.11746 | False | 870.6 | 0 |
| G | B `+NO_AJAC` | FAIL | 0.16235 | 0.01838 | **0.16334** | 0.987228 | 0.999359 | **0.986260** | 3.387 | 3.387 | 0.01892 | True | 0.846 | 320 |

(`p_osc = r_osc = 0.00000` in **every** config — the oscillation guard is clean everywhere, exactly as round 7 said. Initial domain mass `M(0) = 945.07 kg/m²`.)

Three facts follow, and they reorganise the round:

1. **Round 7's `cj=30.02 / mj=32.00 / cc=0.117` reproduce EXACTLY today** — `30.0178 / 31.9982 / 0.11746` — but **only under configs C/D/F** (`ACID_YADV_ALPHA_IMPLICIT=1`). Under those configs case15 fails on **exactly one** predicate, `smooth_ok`; every `l2`/`corr`/`osc` criterion passes. Round 7's diagnosis is intact and 19 rounds durable. The prompt's "live re-verify" numbers (`l2_p=0.0144, l2_u=0.0170, l2_rho=0.0197`) are config **C**'s numbers, not config B's — that is the source of the mismatch in the briefing.
2. **Under the loop's headline config B (`ACID_YADV=1` alone) the central jump is NOT the blocker.** `cj = 2.307` against a threshold of `8.0` — passing with a 3.5x margin. B fails on `l2_rho` (0.1676 vs 0.05, **3.4x over**) and `corr_rho` (0.98451 vs 0.99). Same for G.
3. **Config B's `l2_rho` failure has a measured, unambiguous cause: the run destroys 99.92% of the domain mass.** `Σρdx` goes `945.07 → 0.761` at N=400 and `945.07 → 2.441` at N=800 (the reference). The 3.2x ratio between the two surviving masses is exactly the 3x amplitude deficit the metrics see (`amp_ratio_rho = 0.3395`, `amp_ratio_p = 0.3300`). **The failing metric is a direct readout of how much mass each mesh deletes.** Config C, by contrast, ends at 870.6 kg — an 8% loss consistent with genuine transmissive outflow — with **zero** cells at the pressure floor.

### 1.3 Diagnosis approach, and why

I did **not** stop at "case15 fails". This project's culture (rounds 13/16/21) is to localise before proposing anything, so I localised in four independent ways, all with **zero code changes**, using only existing binaries and the existing `ACID_TEND_SCALE` / `ACID_DBG` knobs:

* **Spatially** — under B the entire `l2_rho` error lives in two symmetric boundary layers, `x ≲ 0.08` and `x ≳ 0.92`, where got `ρ/p` are ~1/3 of ref; the interior is pinned at the 1.0 Pa floor in both. The velocity field is *fine* (`l2_u = 0.019`, `corr_u = 0.9993`) — only the thermodynamic state is wrong.
* **Temporally** — via `ACID_TEND_SCALE` (§3.4): mass loss begins **exactly** at the step where the first pressure-floor cells appear, and tracks `nfloor` monotonically thereafter.
* **Structurally** — the mechanism is already named in this codebase. `acid.cpp:711-720` documents it as round 16 §26.1's **"vacuum blister"**: the recovery site right after `Yv = anew` (`acid.cpp:1266-1275`) recovers `alpha` from the new `Y` at the **stale** `(p_o,T_o)`, and "the Eqs.43-44 rebuild right after then **deletes most of the cell's true mass** at that spurious alpha". Round 16 measured it on case24 at one cell. **Case15 under config B is that same defect running over 80% of the domain for 85 consecutive steps.**
* **By elimination** — both runs reach `t_end` cleanly (85 and 169 steps, `ACID done ... t=9.500000000e-04 of 9.500000000e-04`, no `STALLED`, no `RETRY`, no `DIVERGED`). The `max_steps` caveat in `types.hpp:32` is **stale for this config**: case15 is not budget-limited.

### 1.4 What round 27 should therefore do

Round 27 is a **diagnosis round on config B's mass-conservation defect**, with the config-C central-jump defect characterised and handed forward. Rationale:

* B is the config whose `pass_count` *is* `ACID_YADV`'s recommended status. C/D/F are secondary research flags.
* A **conservation violation** is the most fundamental class of error available and the one with the least room for taste: either the discrete mass budget closes or it does not.
* The two defects are almost certainly **sequential, not competing**: `cj` is small in B precisely *because* B's core has been evacuated; C, which conserves mass, is the config that exhibits `cj=30`. So fixing B's mass defect is predicted to *expose* the central jump in B. That prediction is pre-registered in §7 (S4) — it is the round's main falsifiable risk, and it means the roadmap's stated goal is **not abandoned, it is sequenced second**.
* Round 25's F3 and round 21's RECON already restore B's mass (`Σρdx` → 961.98 and 965.27) without passing the gate (`l2_rho` 0.103 / 0.083). So there is an existing, physically-derived family to build on — and a measured reason not to just switch one of them on and call it a fix.

**Stage 1 is one new read-only instrument (`ACID_MBAL`) that closes the discrete mass budget into four named terms.** Stage 2 (a candidate fix) is explicitly **conditional** on Stage 1 naming a single dominant term. This is rounds 13/16/21's pattern verbatim.

---

## 2. Verified code facts (every line number below was read today, in this worktree)

All paths relative to `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/`.

### 2.1 The case15 gate — `cpp/denner_1d/src/validation.cpp:684-730`

```
684  if (case_id == "15") {
693      const int nn = static_cast<int>(got.x.size());
694      if (nn < 8) return false;
695      auto jump_stats = [&](const std::vector<double>& u, double& central, double& jmax,
696                            double& conc) {
697          central = std::abs(u[nn / 2] - u[nn / 2 - 1]);
700          for (int i = 1; i < nn; ++i) {
701              if (got.x[i] < 0.35 || got.x[i] > 0.65) continue;   // <-- NOTE: got.x, both calls
702              const double j = std::abs(u[i] - u[i - 1]);
703              jmax = std::max(jmax, j);  tv += j;
706          conc = jmax / std::max(tv, 1e-300);
708      double cj, mj, cc, cj_r, mj_r, cc_r;
709      jump_stats(got.u, cj, mj, cc);
710      jump_stats(ref.u, cj_r, mj_r, cc_r);
711      const bool smooth_ok = cj <= std::max(8.0, 1.10 * cj_r) &&
712                             mj <= std::max(8.0, 1.10 * mj_r) &&
713                             cc <= std::max(0.04, 1.10 * cc_r);
722      const double p_osc = std::max(0.0, tv_of(got.p) - tv_of(ref.p)) / std::max(tv_of(ref.p), 1.0);
724      const double r_osc = std::max(0.0, tv_of(got.rho) - tv_of(ref.rho)) / std::max(tv_of(ref.rho), 1.0e-6);
726      const bool osc_ok = p_osc < 0.02 && r_osc < 0.04;
727      return m.corr_p >= 0.93 && m.corr_u >= 0.998 && m.corr_rho >= 0.99 &&
728             m.l2_p <= 0.18 && m.l2_u <= 0.06 && m.l2_rho <= 0.05 &&
729             smooth_ok && osc_ok;
```

Note there is **no `amp_ratio` term** in case15's gate, and both `jump_stats` calls index `got.x` for the window mask (harmless — `got` and `ref` share the grid by construction, `computed_reference` resamples onto `c.config.cells`). `nn/2 = 200`, so `cj = |u[200] - u[199]|` at `x = 0.50125` vs `0.49875`.

`ErrorMetrics` are produced by `compare()` at **`validation.cpp:405-412`**, which calls `accumulate()` at **`validation.cpp:317-359`**; `rel_scale` (the `l2`/`linf` normaliser, `max(range(ref), 1.0)`) is **`validation.cpp:18-25`**; `correlation` is **`validation.cpp:27-51`**. `accumulate` has a **degenerate-flat branch at `validation.cpp:346-350`**: if `amplitude(ref) < 1e-12` it returns `corr = 1.0`, `amp_ratio = 1.0`, `hf = 0.0` when the solver field is also flat. **This is why configs A/E/B+F3/B+RESYNC read `l2_p = 0.00000, corr_p = 1.000000`** — their pressure field is uniformly at the 1.0 Pa floor in *both* got and ref. Anyone reading `l2_p = 0` on case15 as "perfect pressure" is misreading a floored field; state this in the round writeup.

### 2.2 case15's definition — `cpp/denner_1d/src/cases.cpp`

* Config: **`cases.cpp:493`** — `auto c15 = base_config(400, 9.5e-4, 0.0, 1.0);` No per-case knobs whatsoever. `base_config` (**`cases.cpp:15-31`**) sets `cfl=0.45`, `unic=true`, `uniform=true`, transmissive/transmissive; `max_steps = 20000` from **`include/denner1d/types.hpp:32`**.
* Registration: **`cases.cpp:582`** — `{"15", "15_E air-water cavitation", air, water, c15}` (`water` = NASG Le Métayer liquid water, **`cases.cpp:441`**).
* IC: **`cases.cpp:682-691`** — `alpha = 0.055` uniform, `u = ±100` (left −, right +), `p = 1e5` uniform, `T` = the volume-weighted blend of the air/water temperatures at `ρ=1.3 / 1000`. A **symmetric double rarefaction with supersonic outflow**: the Wood mixture sound speed at `α=0.055` is ≈52 m/s, so |u|/a ≈ 1.9 at t=0.
* Reference: **`cases.cpp:751-753`** — `if (c.id == "15") return computed_reference(c, 800);`
* `computed_reference` — **`cases.cpp:423-437`**:
  ```
  423  PrimitiveState computed_reference(const CaseDefinition& c, int cells) {
  424      CaseDefinition hi = c;
  425      hi.config.cells = std::max(c.config.cells, cells);
  426      const PrimitiveState fine = solve_case(hi);
  427      PrimitiveState out;  resize_state(out, c.config.cells, c.config.x0, c.config.x1);
  428-433  ... interp(fine.x, fine.{alpha,u,p,T}, out.x[i]) ...
  434      refresh_thermo(out, c.phase1, c.phase2);
  ```
  Confirmed exactly as round 26 §36.6 said, with the details that matter for this round: **only `cells` changes** (400 → 800); `final_time`, `cfl`, `unic`, BCs, and **the entire process environment** are shared, so the reference runs under the *same* `ACID_*` flags as the graded run. `dx` halves, `dt` halves (material CFL), step count 85 → 169. `out` is on the **400-cell** grid, so all gate statistics are computed on a common grid — the `mj`/`cj` comparison is apples-to-apples, not a `dx`-scaling artifact. **`ρ` in `out` is rebuilt by `refresh_thermo` from the interpolated `(p,T,alpha)`**, i.e. the reference's density is an EOS evaluation of interpolated primitives, not an interpolated density. (Worth knowing; not load-bearing for this plan.)

### 2.3 The mechanism sites in `cpp/denner_1d/src/acid.cpp`

* **Step snapshot**: `acid.cpp:1009` `const Field s0 = s;` (before the retry loop, `acid.cpp:1034`); `acid.cpp:1035-1036` `s = s0; Yv = Yv0;`; `acid.cpp:1039` `const Vec u_o = s.u, p_o = s.p, T_o = s.T;`.
* **Conservative `ρY` update + the continuity predictor**: `acid.cpp:1256-1265`
  ```
  1258   const double rho_old = std::max(s.rho[i], 1e-300);
  1259   const double rY = rho_old * Yv[i] - dt / dx * (mdR_o[i]*af[i+1] - mdL_o[i]*af[i]);
  1261   const double rho_star = yadv_rhoold ? rho_old
  1262                         : std::max(rho_old - dt/dx*(mdR_o[i] - mdL_o[i]), 1e-300);
  1264   anew[i] = std::clamp(rY / rho_star, 0.0, 1.0);
  ```
  `rho_star` **is** the discrete conservative continuity predictor for this cell's new mixture density. It is computed, used to divide, and thrown away.
* **The recovery site (the vacuum blister)**: `acid.cpp:1266-1275`
  ```
  1266   Yv = anew;
  1267   for (int i = 0; i < n; ++i) {
  1268       const double pu = std::max(p_o[i], 1.0), Tu = std::max(T_o[i], 1e-6);
  1269       s.alpha[i] = std::clamp(alpha_from_mass_fraction(Yv[i],
  1270                        phase_props(pu,Tu,A).rho, phase_props(pu,Tu,B).rho), 0.0, 1.0);
  ```
  `alpha` is recovered at the **stale** `(p_o, T_o)` — and `pu = max(p_o,1.0)` means that once a cell is at the floor, `ρ_air(1 Pa, T)` is ~1e-5, so `alpha → 1` regardless of `Y`. **This is the alpha→1 saturation measured in §3.3.**
* **The Eqs.43-44 old-level rebuild**: `acid.cpp:1320-1332`
  ```
  1326   const double al = std::clamp(s.alpha[i], 0.0, 1.0);
  1327   const auto pa = phase_props(std::max(p_o[i],1.0), std::max(T_o[i],1e-6), A);   // + pb
  1328   rho_o[i] = std::max(al*pa.rho + (1.0-al)*pb.rho, 1e-300);
  ```
  `rho_o` — the mass the discrete continuity treats as "what was here before" — is **re-derived from the just-recovered `alpha`**, not from `s0.rho` and not from `rho_star`. With `alpha` saturated at ~1, `rho_o` collapses to `(1-α)·ρ_water ≈ 0.2`. `Cold_con[i] = rho_o[i]` at **`acid.cpp:1457`** (BE branch; `cell_bdf2` is false for case15 — see §2.4).
* **Discrete continuity residual**: `acid.cpp:1709-1713`
  ```
  1709   const double trans_c = (bdf_c0[i]*s.rho[i] - Cold_con[i]) * VdT;
  1713   Rres[i][1] = trans_c + (mdotR[i] - mdotL[i]);
  ```
  with the **ACID per-cell face mass flux** at `acid.cpp:1696-1703`:
  `mdotR[i] = (α_i·ρ_a^up[i+1] + (1-α_i)·ρ_b^up[i+1])·θ[i+1]`, `mdotL[i]` likewise with cell `i`'s **own** `α_i`. **Therefore `mdotR[i] ≠ mdotL[i+1]` wherever `α` varies between neighbours** — the ACID Eqs.41-42 blend is deliberately non-telescoping. That is an independent, quantifiable mass-leak channel (Denner's pressure-equilibrium design), and the instrument must separate it from the remap term rather than assume either.
* **Pressure floor in the Newton line search**: `acid.cpp:2344` (AJAC path) and `acid.cpp:2536` (FD path), both `s.p[i] = std::max(sbak.p[i] + dpi, 1.0);`. The `1.0` Pa floor is a hard global constant; `refresh_thermo` in `solver.cpp:1011` applies the same floor to the state.
* **Non-converged accept**: `acid.cpp:2585` `if (ajac && coupled && !conv_inner && best_it >= 0) s = s_best;`, documented at `acid.cpp:1819-1823` ("a stiff regime (**case15 cavitation**) NEVER converges — the line search pins at the al floor with the residual flat"). Note `Rres` is **not** recomputed after this assignment, so any budget instrument must call `compute_R()` once itself (see §4.3).
* **Regime auto-selection for case15**: `acid.cpp:421-431` — `p_ratio = 1.0` (uniform IC pressure) ⇒ `unic=true` ⇒ `coupled = true`, `use_minmod = lowdiss = false`; `acid.cpp:437-443` ⇒ `auto_material = true` (no acoustic source, `p_ratio < 1.01`, `umax > 0`); `acid.cpp:452-453` ⇒ `penta_solve = true`; `acid.cpp:484-486` ⇒ `bdf2 = false` (no acoustic source) ⇒ `tr_bdf2 = false`, `bdf_c0[i] = 1.0`, `Cold_* = *_o`. **The budget algebra in §4.2 is therefore exact for case15, with no BDF2/TR-BDF2 terms.** (The comment at `acid.cpp:420` saying "15 is excluded from this mode" is **stale** — `c15` gets `unic=true` from `base_config` like every other case. Do not act on that comment; it is a documentation bug, and noting it in the round writeup is free.)
* **Existing diagnostic precedents to copy verbatim in style**: `ACID_RINIT` decl `acid.cpp:612-620`, `ACID_RCELL` decl `acid.cpp:621-631` + print `acid.cpp:1373-1387`, `ACID_RECON` decl `acid.cpp:647-651`, `ACID_F3` decl `acid.cpp:710-746` + print `acid.cpp:1306-1314`, `ACID_RESYNC`'s **running mass-drift accounting** print `acid.cpp:995-1003` (`dM_step=... dM_total=... dM_total/M0=...` — the closest existing analogue to what §4 proposes), `ACID_TEND_SCALE` decl `acid.cpp:786-805` + loud warning `acid.cpp:841-844`, `ACID_BLK_STEP` step-gating idiom `acid.cpp:939, 1307, 1340, 1375`.
* **Scratch arrays the instrument needs, and their scope**: `mdotL/mdotR` declared `acid.cpp:1484`, `Rres` declared `acid.cpp:1492` — both **inside the retry-loop body**, before `compute_R`, and still live at `acid.cpp:2585`. Good.

### 2.4 The gate battery — `scripts/yadv_r9_sweep.py`

* `ACID_ENV_VARS` purge list: **`scripts/yadv_r9_sweep.py:37-42`** (must be extended with any new flag — round 22's hygiene rule).
* `CONFIGS` A–G: **`:44-52`**. `EXPECTED`: **`:62-70`** — `A: (19, set())`, `B: (15, {15,24,33,34})`, `C: (14, {14,15,24,33,34})`, `D: (13, {14,15,24,27,33,34})`, `E: (14, {15,24,28,33,34})`, `F: (14, {14,15,24,33,34})`, `G: (15, {15,24,33,34})`. **case15 is in every fail set except A's** — confirmed live today for all seven.
* `base_env()` **`:75-82`** (sets `DENNER_ACID=1`, purges); `do_sweep()` **`:117-142`** (prints `ALL GATES OK`); `do_verify()` **`:150-213`** (9 cases dumped against `/home/younglin90/work/claude_code/claudeCFD/solver_denner/build-cpp/cpp/denner_1d/denner1d_dump` + the `ACID_YADV=1 == unset` case01 check). The published reference binary **exists** at that path (verified today).

---

## 3. Fresh measurement (already performed — reproduce, do not re-derive)

Everything in this section was produced today from the **already-built** binaries in `build-cpp/` (built 2026-08-02 14:37, newer than all sources). The implementing session should reproduce §3.1 and §3.2 first, as Stage 0, and stop the round immediately if any number disagrees materially (see S0 in §7).

### 3.1 Reproduce commands

```bash
W=/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass
cd $W && cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8

# (a) the single-case verdict + the metrics JSON, config B  (~90 s: it solves N=400 AND N=800)
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate --only 15

# (b) the same for config C
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ./build-cpp/cpp/denner_1d/denner1d_validate --only 15

# (c) the full per-predicate breakdown + mass + floor census, all 7 configs
#     write a scripts/yadv_r27_case15.py that: for each config, runs denner1d_dump 15 with
#     base_env(overlay) (subprocess capture_output, NEVER shell redirection -- denner-pitfalls),
#     parses the 8 CSV columns, and recomputes validation.cpp:695-729 verbatim in Python
#     (jump_stats, tv_of, rel_scale/l2/corr per validation.cpp:18-51,317-359), plus
#     M = sum(rho)*dx, M_ref = sum(rho_ref)*dx, nfloor = #{p <= 1.0+1e-12}.

# (d) step budget / completion (confirms NOT max_steps-limited)
DENNER_ACID=1 ACID_YADV=1 ACID_DBG=1 ./build-cpp/cpp/denner_1d/denner1d_dump 15 2>&1 >/dev/null | grep -E 'ACID done|STALL|DIVERG'
```

Expected output of (a), **verbatim, measured today**:
```
{"case":"15","N":400,"pass":false,"finite":true,"l2_p":0.166533,"l2_u":0.0187388,"l2_rho":0.167608,
 "corr_p":0.985535,"corr_u":0.999334,"corr_rho":0.984514,"amp_ratio_p":0.330034,"amp_ratio_u":0.999913,
 "amp_ratio_rho":0.339504,"peak_delta_p":0,"peak_delta_u":0,"peak_delta_rho":0,"hf_p":0.0716926,
 "hf_u":0.038648,"hf_rho":0.0590888,"linf_p":0.669966,"linf_u":0.0423538,"linf_rho":0.660496}
DENNER1D_CPP_METRIC pass_count=15 total=19
```
Expected output of (d), **verbatim**:
```
ACID done case=15 step=85 t=9.500000000e-04 of 9.500000000e-04     <- the N=400 graded run
ACID done case=15 step=169 t=9.500000000e-04 of 9.500000000e-04    <- the N=800 reference
```
(no `STALL-ACCEPT-TOTAL`, no `STALLED`, no `DIVERGED`, no `RETRY`. Same two lines for config A.)

### 3.2 The full per-predicate table

**This is §1.2's table.** Round-specific highlights to carry into `YADV_RESEARCH.md`:

* Config **B**'s only failing predicates: `l2_rho = 0.167608` (gate 0.05) and `corr_rho = 0.984514` (gate 0.99). `l2_p = 0.166533` passes its 0.18 gate with only a **7.5% margin** — flag this: a partial fix that moves `p` without moving `ρ` could flip `l2_p` to failing.
* Config **C/D/F**'s only failing predicate: `smooth_ok` (all three sub-terms fail: `cj 30.018 > 8.0`, `mj 31.998 > 19.885`, `cc 0.11746 > 0.09283`). Round 7's numbers, to 5 significant figures, 19 rounds later.
* Config **A** passes but **narrowly**: `cc = 0.03449` vs 0.04 (14% margin), `l2_rho = 0.04246` vs 0.05 (15% margin). Config **E** (OFF+FD) fails on `l2_rho = 0.06995` alone. Neither is this round's target, but both are relevant to G2 sensitivity.
* `p_osc = r_osc = 0` in every config: **the oscillation guard is not involved anywhere**, confirming round 7's `osc_ok=True` finding across the board.

### 3.3 Where B's error lives, and what the field actually looks like

Under B the `ρ` error is **entirely** in the two symmetric outer layers; the top-15 error cells are `i ∈ {0,1,2,3,4,5,6,7}` and their mirrors `{392..399}`, at 0.55–0.66 of `rel_scale` each. Representative rows (N=400, `rel_scale_ρ = 26.0622`):

| i | x | ρ | ρ_ref | p | p_ref | α | u | u_ref |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.00125 | 9.021 | 26.235 | 49.12 | 146.81 | 0.99102 | −99.909 | −99.918 |
| 10 | 0.02625 | 5.009 | 17.622 | 26.72 | 96.82 | 0.99501 | −99.888 | −99.901 |
| 29 | 0.07375 | 0.694 | 4.423 | 3.52 | 22.71 | 0.99931 | −99.839 | −99.858 |
| 200 | 0.50125 | 0.2035 | 0.2108 | 1.00 | 1.00 | 0.99981 | −1.153 | −1.145 |

Note `α ≥ 0.991` **everywhere** and `ρ ≤ 9.02` **everywhere**, at a t_end when the physical rarefaction fan (fastest characteristic 152 m/s ⇒ reach 0.144 m) cannot have touched `x < 0.356`. Under A, `α = 0.055` everywhere and `ρ ∈ [949, 1022]`. Under C, `α` is a smooth monotone profile from 0.055 at the edges to 0.9994 at the core, with `p` from 1e5 down to 3.24 Pa.

Physically: `α → 0.9998` **is** the correct PTE answer for a fixed mass fraction at `p = 1 Pa` (with `Y_air ≈ 7.6e-5`, `ρ_air(1 Pa) ≈ 1.2e-5`, `ρ_w ≈ 1000` ⇒ `α = 0.9998`). The model is not wrong to want that. What is wrong is that reaching it **without also carrying the cell's mass** deletes `945 → 0.2` kg/m³ of matter.

### 3.4 Mass trajectory in time (zero code changes, `ACID_TEND_SCALE`)

`ACID_TEND_SCALE=s` (`acid.cpp:786-805`) scales the solver's stop time only. **The `*_ref` columns and all `validate` metrics are meaningless under it — only the solver columns are read here** (the flag prints its own loud warning to that effect at `acid.cpp:841`).

| `t/t_end` | B: `Σρdx` | B: `min p` | B: `α` range | B: `nfloor` | C: `Σρdx` | C: `min p` | C: `nfloor` |
|---|---|---|---|---|---|---|---|
| 0.01 | 949.76 | 1.25e4 | [0.0550, 0.0550] | **0** | 948.06 | 1.27e4 | 0 |
| 0.02 | **879.01** | **1.0** | [0.0550, 0.4821] | **66** | 947.11 | 5.79e3 | 0 |
| 0.05 | 746.08 | 1.0 | [0.0550, 0.99983] | 74 | 945.52 | 1.43e3 | 0 |
| 0.10 | 701.80 | 1.0 | [0.0550, 0.99983] | 84 | 944.37 | 263 | 0 |
| 0.25 | 547.60 | 1.0 | [0.0561, 0.99983] | 112 | 937.03 | 8.92 | 0 |
| 0.50 | 278.00 | 1.0 | [0.2481, 0.99983] | 164 | 914.76 | 5.32 | 0 |
| 1.00 | **0.761** | 1.0 | [0.9910, 0.99983] | **322** | **870.61** | 3.24 | **0** |

**The mass loss switches on in exactly the interval where the first floor cells appear (`0.01 → 0.02`), and thereafter tracks `nfloor` monotonically.** Config C never floors and never loses mass beyond the ~8% that transmissive outflow explains. This is the strongest single piece of evidence in the round and it cost nothing.

### 3.5 What the existing research flags do to case15 under B (zero code changes)

| overlay on B | `l2_rho` | `corr_rho` | `l2_p` | `corr_p` | `Σρdx` | `Σρdx` ref | `nfloor` |
|---|---|---|---|---|---|---|---|
| — (plain B) | 0.16761 | 0.984514 | 0.16653 | 0.985535 | 0.761 | 2.441 | 322 |
| `ACID_YADV_F3=1` | 0.10300 | 0.988633 | 0.00000\* | 1.000000\* | 961.98 | 956.26 | 400 |
| `ACID_YADV_RECON=1` | 0.08293 | 0.986733 | 0.18298 ✗ | 0.936815 | 965.27 | 962.02 | **0** |
| `ACID_YADV_RESYNC=1` | 0.17684 | **0.633657** | 0.00000\* | 1.000000\* | 999.15 | 1002.01 | 400 |
| `ACID_YADV_HREINIT=1` | 0.16761 | 0.984514 | — | — | 0.761 | 2.441 | 322 (exact no-op) |
| `ACID_YADV_ALPHA_IMPLICIT=1` (=C) | **0.01966** ✓ | **0.996734** ✓ | 0.01439 | 0.999285 | 870.61 | 865.54 | **0** |

\* the degenerate-flat branch (`validation.cpp:346-350`) — both fields uniformly floored, **not** a perfect pressure.

Read: **F3 and RECON each independently restore the mass** (`0.76 → 962 / 965`) — strong corroboration that the remap/recovery site is the sink — **but neither passes** (`l2_rho` 0.103 / 0.083 vs 0.05), and RECON pushes `l2_p` to 0.18298, *over* its gate. `RESYNC` is actively harmful (`corr_rho` 0.634). `HREINIT` is a bit-exact no-op here. `ALPHA_IMPLICIT` is the only overlay that clears every `l2`/`corr` bar — by keeping the Newton off the floor entirely — and it is the one that then hits `cj = 30`.

**Do not read this table as "just switch on RECON/F3".** It is diagnostic corroboration. Any adoption decision requires the Stage-1 budget to say *which term* each of them moved, and requires the full G2 sweep (RECON/F3 affect cases 24/33/34/13/14, and RECON already costs `l2_p` here).

---

## 4. Diagnostic design — Stage 1: `ACID_MBAL`

### 4.1 What it must decide

Four mutually exclusive, individually measurable channels can move the domain mass between two accepted states. Name them and measure all four; do not assume:

* **(R) Remap** — the Eqs.43-44 rebuild (`acid.cpp:1320-1332`) re-derives `rho_o` from the freshly recovered `alpha`, silently redefining "the mass that was here". Hypothesis H1 (leading): dominant.
* **(L) Leak** — the ACID per-cell blend makes `mdotR[i] ≠ mdotL[i+1]` (`acid.cpp:1696-1701`); the interior flux sum does not telescope. Hypothesis H2.
* **(N) Non-convergence** — the accepted iterate has `Rres[·][1] ≠ 0` because the line search is pinned by the `p ≥ 1.0` floor (`acid.cpp:2344`) and the step is accepted anyway (`acid.cpp:2585`). Hypothesis H3.
* **(B) Boundary** — the genuine, physical transmissive outflow `dt·(mdotR[n-1] − mdotL[0])`. The control: it must come out ≈ the ~75 kg config C loses.

### 4.2 The exact identity to instrument

For case15, `bdf_c0 ≡ 1`, `Cold_con = rho_o`, `tr_bdf2 = false`, `VdT = dx/dt` (verify `VdT`'s definition at its declaration when implementing). From `acid.cpp:1709-1713`, summing the continuity residual over all cells and telescoping only what actually telescopes:

```
Σ_i (ρ_i^new − ρ_o,i)·dx/dt  +  [mdotR[n−1] − mdotL[0]]  +  Σ_{i=0}^{n−2}(mdotR[i] − mdotL[i+1])  =  Σ_i Rres[i][1]
```

Multiply by `dt` and split the old level into its two redefinitions:

```
ΔM_step ≡ M_new − M_prev
        = (M_star − M_prev)            <- ADV : the explicit old-level flux the Y block already applied
        + (M_reb  − M_star)            <- REMAP (R): the vacuum blister; ZERO iff the recovered alpha's
                                          EOS density equals the conservative predictor rho_star
        − dt·BND                       <- (B) physical outflow
        − dt·LEAK                      <- (L) ACID non-telescoping interior leak
        + dt·RES                       <- (N) accepted non-zero continuity residual
```
with
```
M_prev = Σ s0.rho[i]·dx                        (acid.cpp:1009 snapshot)
M_star = Σ rho_star[i]·dx                      (acid.cpp:1261-1262, currently discarded — retain under the flag)
M_reb  = Σ rho_o[i]·dx                         (acid.cpp:1328)
M_new  = Σ s.rho[i]·dx                         (accepted iterate, end of step)
BND    = mdotR[n−1] − mdotL[0]
LEAK   = Σ_{i=0}^{n−2} (mdotR[i] − mdotL[i+1])
RES    = Σ_i Rres[i][1]
```
**`closure = ΔM_step − (ADV + REMAP − dt·BND − dt·LEAK + dt·RES)` must be ~1e-12 relative.** That is the instrument's own self-test: if the closure does not close, a term is missing and the instrument is wrong — report that, do not paper over it. (This is round 26's P0/P1 cross-validation discipline applied to an instrument instead of a solver.)

### 4.3 Implementation specification (house style, `acid.cpp` only)

* **Flag**: `ACID_MBAL` — DIAGNOSTIC ONLY, default OFF, **stderr only, applies nothing**. Declare next to the other diagnostics (immediately after `recon_dbg` at `acid.cpp:647-651`), with the full multi-line rationale comment the file's convention requires (state the identity of §4.2, the four channels, and that the identity is exact only for `bdf_c0=1 && !tr_bdf2`).
* **Step gating**: reuse the existing `ACID_BLK_STEP` (`se ? atoi(se) : -1`, `-1` ⇒ every step) exactly as `acid.cpp:1340/1375` do. Case15 is 85 steps, so every-step output is ~85 lines — fine.
* **Retention of `rho_star`**: at `acid.cpp:1261-1264`, accumulate `M_star += rho_star*dx` **inside the existing loop, under `if (mbal)`**, into a step-local double. No array, no allocation, and no FP work on the unset path (the accumulate is inside the flag's branch).
* **`M_prev`, `M_reb`**: `M_prev` from `s0` (`acid.cpp:1009`); `M_reb` accumulated in the existing rebuild loop at `acid.cpp:1321-1332`, again inside `if (mbal)`.
* **`BND`/`LEAK`/`RES`/`M_new`**: computed **after** the accept decision at `acid.cpp:2585`, and the instrument must **call `compute_R()` once itself** first, because `Rres`/`mdot*` correspond to the last *trial* iterate, not to `s_best` (verified: `compute_R` writes only `Rres`, `Rene`, and the face/flux scratch — it does not mutate `s`; re-verify this by reading `acid.cpp:1495-1745` before relying on it). Guard the extra call with `if (mbal)` so the unset path performs zero extra work.
* **Placement**: inside the retry-loop body, after `acid.cpp:2585` and before the accept/finite scan, so the printed line can carry `retry=` and be filtered. Case15 takes zero retries (measured), so there is exactly one line per step.
* **Output line** (one per step, single `fprintf`, matching `RCELL`/`F3`/`RESYNC` formatting conventions):
  ```
  MBAL case=%s step=%d retry=%d dt=%.6e M_prev=%.9e M_star=%.9e M_reb=%.9e M_new=%.9e
       dM=%.6e adv=%.6e remap=%.6e bnd=%.6e leak=%.6e res=%.6e closure=%.3e
       nfloor=%d res_floor=%.6e res_other=%.6e conv=%d r_init=%.6e rbest=%.6e
       almax=%.6f almin=%.6f i_remax=%d
  ```
  where `res_floor` is `dt·Σ Rres[i][1]` restricted to cells with `s.p[i] <= 1.0` and `res_other` its complement (this is what separates channel N-at-the-floor from N-elsewhere), `nfloor = #{i : s.p[i] <= 1.0}`, and `i_remax` is the cell contributing the largest `|rho_o[i] − rho_star[i]|·dx` (the worst blister cell, so `ACID_RCELL` can be pointed straight at it next).
* **`yadv`-independence**: `M_star` only exists when `yadv` is on. On the OFF path, print `M_star=nan remap=nan` and still print the other four terms — the OFF-path budget is a **valuable control** (config A gains 2.4% mass; knowing which channel does that is worth one `nan` field).
* **Guard**: if `bdf_c0[i] != 1.0` for any cell, or `tr_bdf2`, print one extra token `exact=0` (the identity acquires BDF2 terms); else `exact=1`. Never silently print a budget that cannot close.
* **Purge list**: add `"ACID_MBAL"` to `ACID_ENV_VARS` at `scripts/yadv_r9_sweep.py:37-42` (round 22's hygiene rule — this list was stale for several rounds before).
* **Zero-cost-when-unset**: every accumulation, the extra `compute_R()`, and every `fprintf` sit inside `if (mbal)`. G4 (§6) verifies byte-identity with the flag set-but-inert is not claimed — the flag *does* print — G4 instead verifies that setting `ACID_MBAL=1` leaves the **solution columns** byte-identical (stderr-only instrument), and that leaving it unset leaves everything byte-identical to pre-change.

### 4.4 The Stage-1 measurement runs

```bash
# primary: config B, all 85 steps, case15
DENNER_ACID=1 ACID_YADV=1 ACID_MBAL=1 ./build-cpp/cpp/denner_1d/denner1d_dump 15 2>mbal_B.txt >/dev/null
# control 1: config C (mass-conserving) -- expect remap ~0, nfloor 0 at every step
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_MBAL=1 ... 15
# control 2: config A (OFF) -- expect remap small; explains A's +2.4% mass gain
DENNER_ACID=1 ACID_MBAL=1 ... 15
# corroboration: B+F3 and B+RECON -- expect remap collapsed toward 0 (they restore M)
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_F3=1 ACID_MBAL=1 ... 15
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_RECON=1 ACID_MBAL=1 ... 15
# cross-case sanity (the instrument is case-blind): case24 step 0 and step 100, config B
DENNER_ACID=1 ACID_YADV=1 ACID_MBAL=1 ACID_BLK_STEP=0 ... 24     # must reproduce round 16 sect.26.1
```
Each `denner1d_dump 15` also solves the N=800 reference, so **two** interleaved `MBAL` streams appear per invocation; disambiguate by the step counter resetting and by `n` (add `n=%d` to the line if that is not already unambiguous).

**Pre-registered quantitative predictions (H1), to be falsified or confirmed:**
1. `remap` is negative and dominates `|dM|` from the first floored step onward; `Σ_steps remap ≈ −944` kg to within 5%.
2. `bnd` integrates to **−70 … −90 kg** (matching config C's total loss), i.e. the physics is a minor term under B.
3. `leak` and `res_other` are each < 5% of `|Σ remap|`.
4. Under config C, `|Σ remap| < 5` kg and `nfloor = 0` at every step.
5. Under B+F3 and B+RECON, `|Σ remap|` drops by ≥ 1 order of magnitude vs plain B.
6. `closure` ≤ 1e-10 relative at every step in every config (instrument self-test).

If (1) fails and `leak` or `res` dominates instead, H1 is dead and the round reports that — a different, equally publishable answer that redirects Stage 2 entirely.

### 4.5 Stage 3 (measurement only, no code): characterise config C's core jet

Round 7 said "checkerboard/central-spike artifact". Today's profile says something more specific. Config C, cells 195–206 (`x = 0.489…0.516`):

| i | x | α | p (Pa) | u | ρ | p_ref | u_ref |
|---|---|---|---|---|---|---|---|
| 196 | 0.49125 | 0.83856 | 1122.8 | −17.781 | 162.00 | 1027.6 | −17.454 |
| 197 | 0.49375 | 0.92912 | 445.0 | −13.844 | 71.13 | 1141.6 | −20.826 |
| **198** | 0.49625 | 0.99944 | **3.243** | **+18.154** | 0.5577 | 591.4 | −19.850 |
| **199** | 0.49875 | 0.99731 | **15.716** | **+15.009** | 2.697 | 44.3 | −1.773 |
| **200** | 0.50125 | 0.99731 | 15.716 | −15.009 | 2.697 | 44.3 | +1.773 |
| **201** | 0.50375 | 0.99944 | 3.243 | −18.154 | 0.5577 | 591.4 | +19.850 |

`cj = |u[200]−u[199]| = 30.018`; `mj = |u[198]−u[197]| = 31.998`. This is **not** an odd-even checkerboard in `u`: it is a **4-cell velocity sign reversal** — cells 198–201 point *inward* (toward the stagnation point) while the whole rest of the domain points outward — driven by a non-monotone pressure core (`445 → 3.24 → 15.72 → 15.72 → 3.24 → 445`, a 137x drop across **one** cell). The reference resolves the same structure monotonically (`−19.85, −1.77, +1.77, +19.85`). So the coarse mesh's failure mode is an **under-resolved near-vacuum core** in a mixture whose Wood sound speed there has collapsed to a few m/s (M ≈ 40), not a pressure-velocity decoupling oscillation. That is a different literature (Einfeldt-class low-density Godunov failure / cavitation-limit relaxation, §5) from the MWI checkerboard note in `.claude/rules/denner-pitfalls.md` — **the pitfalls file's `dhat ~ dt` small-dt checkerboard entry is a different mechanism and the round writeup should say so explicitly**, since the briefing flagged it as a candidate.

Stage 3 is **measurement and writeup only**: reproduce this table, add `ACID_RCELL=195:205 ACID_BLK_STEP=<k>` dumps at 3–4 steps to show when the sign reversal first appears, and record the finding in `YADV_RESEARCH.md` so round 28 starts from it. **No fix is proposed for it this round** (see §8).

---

## 5. Literature step

### 5.1 Already in the repo (checked, not duplicated)

`papers/library/md/mwi/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.md` — the JCP 375 (2018) MWI paper cited by `.claude/rules/denner-pitfalls.md`. **Read it before invoking "checkerboard" language for case15**: §4.5 shows the case15 core artifact is a near-vacuum resolution failure, not the small-`dt` MWI under-damping that paper describes (that one is case25's). Also present and relevant: `papers/library/md/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows.md` (+ the 2019 corrigendum) for Eqs.41-44 themselves; `papers/library/md/2021_Hanimann_consistent_implicit_Rhie_Chow_multiphase.md`; `papers/library/md/newest5/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.md`; `papers/library/md/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1.md` and `papers/library/md/2026_recent/[2026] 4eq + phase change.md` (four-equation PTE + HEM/HRM phase change — the model-extension direction, explicitly **out of scope** here per round 26's escalation, but the correct citation for why the 4-eq model has no vapour-pressure floor). Existing stubs `papers/2017_Chiapolino_Boivin_Saurel_fast_relaxation_needed.md`, `papers/2011_Flatten_Morin_Munkejord_stiffened_gas_equilibrium_needed.md`, `papers/2021_assessment_nonconservative_four_equation_needed.md` are adjacent and should be cross-referenced, not re-created.

### 5.2 New `{slug}_needed.md` candidates (I could not save files; searched via `paper-search-mcp`, DOIs verified through the MCP where shown)

| slug | citation | DOI | why this round needs it |
|---|---|---|---|
| `1991_Einfeldt_Munz_Roe_Sjogreen_godunov_methods_near_low_densities_needed.md` | Einfeldt, Munz, Roe, Sjögreen, *On Godunov-type methods near low densities*, JCP **92**(2), 273–295 (1991) | `10.1016/0021-9991(91)90211-3` (confirmed via Semantic Scholar) | The canonical analysis of exactly case15's core: the symmetric double-rarefaction ("1-2-3") problem, where a scheme produces negative/near-zero internal energy at the stagnation point and loses positivity. Directly names the §4.5 failure class and its resolution-dependence. |
| `2009_Saurel_Petitpas_Berry_relaxation_cavitating_multiphase_needed.md` | Saurel, Petitpas, Berry, *Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures*, JCP **228**(5), 1678–1712 (2009) | `10.1016/j.jcp.2008.11.002` (confirmed via OpenAlex) | The reference treatment of how a mixture model reaches the cavitation/vacuum limit **without** an ad-hoc pressure floor, and how the mixture state is re-derived from conserved variables after relaxation — i.e. the principled version of what `acid.cpp:1266-1275` does by evaluating at stale `(p_o,T_o)`. Most directly relevant paper to Stage 2. |
| `2019_Pelanti_Shyue_multiphase_liquid_vapor_gas_cavitation_needed.md` | Pelanti & Shyue, *A numerical model for multiphase liquid–vapor–gas flows with interfaces and cavitation*, IJMF **113**, 208–230 (2019) | `10.1016/j.ijmultiphaseflow.2019.01.010` (confirmed via OpenAlex) | Cavitation in a mixture model with an explicit account of what the pressure does at the vapour limit; the correct external check for the pitfalls-file claim that "the expansion-core pressure hits the EOS floor, not a physical vapour pressure". |
| `2018_Saurel_Pantano_diffuse_interface_review_needed.md` | Saurel & Pantano, *Diffuse-Interface Capturing Methods for Compressible Two-Phase Flows*, Annu. Rev. Fluid Mech. **50**, 105–130 (2018) | `10.1146/annurev-fluid-122316-050109` (confirmed via OpenAlex) | Survey framing for where a 4-eq PTE mass-fraction model sits and what conservation properties are expected of the volume-fraction recovery. Cheap, high-value context; OA PDF available. |

**Papers I could not obtain:** none were paywall-blocked in a way that mattered — I did not attempt full-text download (read-only mode, and the summaries above are from verified metadata + established knowledge). The implementing session should attempt `download_with_fallback` on each of the four and, per `.claude/skills/cfd-paper-search`, save either a real summary or the `_needed.md` stub with the DOI above. **Two search notes for whoever repeats this**: `search_semantic` returned empty for four of five multi-word queries today (only the Einfeldt query hit); `search_openalex` was the reliable one. `search_arxiv` returned near-total noise for every CFD query attempted (astro/hep results) — do not waste rounds on it for this topic.

---

## 6. Gates and targets, with execution order

### 6.1 Standard hard gates (unchanged, all five mandatory before any merge)

| id | gate | command | pass criterion |
|---|---|---|---|
| **G1** | OFF byte-identity vs the published binary | `python3 scripts/yadv_r9_sweep.py --verify` | `VERIFY OK` — 9/9 `BYTE-IDENTICAL` + case01 `ACID_YADV=1 == unset` |
| **G2** | full config battery unchanged | `python3 scripts/yadv_r9_sweep.py --sweep` | `ALL GATES OK`; every config's `pass_count` **and** `fail_set` equal to `EXPECTED` (`:62-70`) |
| **G3** | unit tests | `./build-cpp/cpp/denner_1d/denner1d_unit` | clean exit |
| **G4** | new-flag no-op | `ACID_MBAL=1` dump vs unset dump, cases 01/13/15/24, **stdout only** (`2>/dev/null`) | byte-identical stdout for all four; **and** `DENNER_ACID=1 ./denner1d_validate` (unset) still 19/19 |
| **G5** | diff hygiene | `git status --porcelain`, `git diff --stat` | only the files §9 lists; **no** `cases.cpp`/`validation.cpp` changes; no stray files (note: `3,` and `=150` at repo root are pre-existing since `325dc5b` — do not touch, do not `git add -A`) |

Also mandatory per the round skill: a **clean rebuild** (`rm -rf build-cpp && cmake … && cmake --build … -j8`) before G1–G3, and `python3 scripts/yadv_verify.py` (9/9) as the skill's Step 6 requires.

### 6.2 Round-specific targets

| id | target | how measured | this is a … |
|---|---|---|---|
| **T0** | §3.1/§3.2's numbers reproduce on a clean rebuild | Stage 0 script | precondition |
| **T1** | `ACID_MBAL` closes: `\|closure\| ≤ 1e-10 · max(\|dM\|, 1)` at **every** step, in **every** config run in §4.4 | Stage 1 | instrument self-test |
| **T2** | the dominant channel of B's `945 → 0.76` collapse is named, with its integrated share of `|dM|` quantified, and the H1 predictions §4.4(1)-(5) each explicitly confirmed or falsified | Stage 1 | the round's primary deliverable |
| **T3** | the OFF path's `+2.4%` mass gain is attributed to a named channel | Stage 1 control | secondary |
| **T4** | round 16 §26.1's case24 blister is reproduced by the same instrument (step 0, cell ~80) | Stage 1 cross-case | independent validation of the instrument against a previously-measured phenomenon |
| **T5** | config C's core-jet structure (§4.5) recorded with its onset step | Stage 3 | handoff to round 28 |
| **T6** *(conditional on T2)* | a candidate fix, behind a new default-OFF flag, moves B's `l2_rho` measurably toward 0.05 **without** regressing any config's `fail_set` | Stage 2 | optional |

### 6.3 Execution order (strict)

1. Clean rebuild → **G3** → **G1** → **G2** (establish the baseline *before* touching anything; round 20's false-regression scare came from skipping this).
2. **Stage 0**: `scripts/yadv_r27_case15.py`, reproduce §3.2 + §3.4 + §3.5. → T0. *(no `acid.cpp` changes yet)*
3. **Stage 1**: implement `ACID_MBAL` (+ `ACID_ENV_VARS` entry) → rebuild → **G4** → **G1** → run §4.4 → T1, T2, T3, T4.
4. **Decision point** — evaluate S1–S6 (§7). If S2/S3 fires, skip to step 6.
5. **Stage 2** *(only if S1 fired)*: implement the one candidate the diagnosis names, behind its own new default-OFF flag → rebuild → **G4** for the new flag → **G1**, **G2**, **G3** → T6.
6. **Stage 3**: §4.5 measurements → T5.
7. **G5**, writeup (`YADV_RESEARCH.md` §37, `YADV_ROADMAP.md` control block + history line, `docs/YADV_ROUND_27_PLAN.md`), commit, merge.

### 6.4 If Stage 2 happens: the one candidate, pre-specified

**Only if T2 names REMAP as dominant.** Flag: `ACID_YADV_REBUILD_ADV`, RESEARCH-ONLY, default OFF, inert unless `ACID_YADV`.

Derivation (parameter-free, from the mechanism, **no constants**): Denner's Eqs.43-44 rebuild exists so that a **moving interface** does not inject a spurious `(ρ_new − ρ_old)/dt` continuity source — a geometric, `O(dx)` correction. Under Y-transport the recovered `alpha` at `acid.cpp:1269` changes for **two** reasons, and round 13's `RMISM` block (`acid.cpp:1341-1366`) already computes both:
* `al_remap_state = alpha_from_mass_fraction(Yv0[i], ρ_a(p_o,T_o), ρ_b(p_o,T_o))` — the **old** `Y` re-evaluated at the current old `(p_o,T_o)`. Its difference from `s0.alpha[i]` is the purely **thermodynamic** remap (`dal_remap`), which carries **no** mass information and is `O(1)` at the floor.
* `s.alpha[i] − al_remap_state` is this step's genuine **advective** change (`dal_adv`), `O(dt)`.

The candidate applies the rebuild to the advective part only:
```
alpha_reb[i] = clamp(s0.alpha[i] + (s.alpha[i] − al_remap_state[i]), 0, 1)
rho_o[i]     = alpha_reb[i]·ρ_a(p_o,T_o) + (1−alpha_reb[i])·ρ_b(p_o,T_o)
```
so that a cell whose `alpha` moved for purely thermodynamic reasons keeps its previous mass, while a genuinely advected interface still gets Denner's correction. Note this changes **only** `rho_o`/`hstat_o`/`Htot_o` (the old level); `s.alpha` itself is untouched, and `compute_R` — the single source of truth — is not edited, so round 17's invariant (Jacobian/initial-guess changes cannot move the converged answer) is not the relevant one here: this **does** move the fixed point, and the plan says so plainly rather than pretending otherwise.

**Explicitly NOT candidates** (and why, so nobody re-proposes them):
* Lowering or scaling the `1.0` Pa pressure floor (`acid.cpp:2344/2536`, `solver.cpp:1011`). It is a bare constant; changing it to make one case pass is tuning by another name, and the floor is not the *sink* (it is the *trigger* — the sink is what happens to the mass once `alpha` saturates). If Stage 1 shows the floor is genuinely causal *and* a principled replacement exists, that is a **round 28 proposal with its own plan**, not a Stage-2 bolt-on.
* Switching `ACID_YADV_RECON`/`ACID_YADV_F3` on by default. They are corroborating evidence (§3.5), not a fix: neither passes the gate, `RECON` pushes `l2_p` over its own gate, and folding either into the default path is a `G2`-wide decision with cases 13/14/24/33/34 at stake that this round has not measured.
* Any per-cell selector, blend, damping coefficient, or `dhat_scale`-style multiplier (user rule, `.claude/rules/denner-pitfalls.md`).

---

## 7. Pre-registered stop / decision rules

Evaluated in order at the §6.3 step-4 decision point. Written before any Stage-1 data exists.

* **S0 — baseline mismatch (abort).** If Stage 0 fails T0 — any config's pass/fail verdict differs from §1.2, or `l2_rho`(B) differs from 0.16761 by >5%, or `cj`(C) differs from 30.018 by >5% — **stop, write up the discrepancy, do not proceed to Stage 1.** Something changed between my measurement and the rebuild and that is the news. `consecutive_failures` **not** incremented (an instrument/environment discovery is progress).
* **S1 — clean diagnosis (proceed to Stage 2).** T1 holds **and** one channel accounts for ≥ 70% of `Σ|dM|` under B **and** the corresponding control behaves as predicted (that channel is ≤ 10% under C). Then, *and only then*, implement §6.4's single candidate. Success of Stage 2 is **not** required for the round to count as progress.
* **S2 — ambiguous diagnosis (stop after Stage 1, honest partial).** T1 holds but no channel reaches 70%, or the controls contradict the primary run. **Do not guess a fix.** Write up the budget tables in full, record which hypotheses are dead, merge the instrument. This is a **success** outcome by rounds 13/16/21's precedent; `consecutive_failures` **not** incremented.
* **S3 — instrument failure (stop, no merge of any fix).** T1 fails (`closure` does not close anywhere): the §4.2 identity is wrong or the placement is wrong. Fix the instrument *once*; if it still fails, report the non-closure as the finding (it would itself be evidence of an unaccounted mass channel — genuinely interesting), merge the instrument only if G1–G5 pass, propose nothing.
* **S4 — Stage 2 succeeds on `l2_rho` but exposes `cj` (the predicted trade).** If the candidate brings B's `l2_rho ≤ 0.05` and `corr_rho ≥ 0.99` but B's `cj` rises above 8.0 so case15 still fails: **this is a genuine, reportable success of the round's own hypothesis, not a failure.** Record it as such, keep the flag default-OFF, and hand the (now single, shared) `cj` blocker to round 28 as the roadmap's original Phase-3c target — with B and C now failing for the *same* reason, which is a strictly better position than today's two-headed one. `consecutive_failures` **not** incremented.
* **S5 — Stage 2 harms (revert the fix, keep the instrument).** If the candidate regresses **any** config's `fail_set` or `pass_count` in G2, or breaks G1, or worsens B's `l2_p` past 0.18: revert the Stage-2 code, keep Stage 1 + Stage 3, report the measured harm in full (magnitudes, which cases, which metric) exactly as rounds 2/4/8 reported theirs. **`consecutive_failures` += 1.**
* **S6 — anti-rescue clause (absolute).** Under no circumstance may this round: (a) touch `cases.cpp` or `validation.cpp`, including case15's thresholds, its `[0.35,0.65]` window, its reference mesh (800), or `computed_reference`; (b) introduce a per-case, per-regime, or per-cell selector, a tuned coefficient, a blend/damping/relaxation factor, or a bare multiplier; (c) present a configuration that merely re-floors the whole field (`l2_p = 0`, `corr_p = 1` via `validation.cpp:346-350`'s degenerate branch) as an accuracy improvement; (d) report a config-C number as if it were config B's, or vice versa — **every** case15 number in the writeup must carry its config tag; (e) convert a measured harm into a reported success by adding a mechanism whose only justification is that it restores a metric.
* **S7 — escalation trigger.** If Stage 1 shows that B's mass collapse is **not** fixable without either changing the pressure floor or adding phase change (i.e. the 4-eq PTE model at fixed `Y` genuinely has no admissible state at these conditions), that is a **model-level** finding of the same class as round 26 §36.7 and must be **surfaced to the user, not silently worked around** — the autonomous loop is not authorized to choose a model extension. Write it up, stop, flag it.

---

## 8. Non-goals (explicit)

Round 27 does **not** attempt any of the following. Each is named so the implementing session does not drift into it:

1. **A fix for config C's `cj = 30` central jump.** Characterised only (§4.5, T5). It is an under-resolved near-vacuum core, and the round has no diagnosed, parameter-free remedy for it; proposing one on this evidence would violate the diagnose-before-fix rule that rounds 13/16/21 exist to enforce.
2. **Making case15 pass.** The round's success criteria (§6.2) are diagnostic. `pass_count` may legitimately stay at 15/19 for B and 14/19 for C.
3. **Any change to the `1.0` Pa pressure floor** (§6.4's exclusion list).
4. **Adopting `ACID_YADV_RECON`, `ACID_YADV_F3`, or `ACID_YADV_ALPHA_IMPLICIT` into the default path.** `ALPHA_IMPLICIT` in particular is known (rounds 4/6/7) to cost case14; folding it is Phase 2's closed question, not this round's.
5. **Phase change / vapour pressure / a fifth equation / interphase mass transfer.** Out of the loop's authorized scope per round 26 §36.7; S7 governs if the evidence points there.
6. **Cases 24/33/34.** Closed by round 26. The `ACID_MBAL` instrument is case-blind and T4 uses case24 as a *validation target* for the instrument only — no attempt to move those cases.
7. **Anything about `max_steps`.** Measured non-binding for case15 today (85/169 steps of a 20000 budget). The `types.hpp:32` comment and roadmap thread (f) are stale for this case; say so in one sentence and move on.
8. **Fixing the stale comment at `acid.cpp:420`** ("15 is excluded from this mode") beyond noting it — a comment-only correction is fine if it lands in the same diff, but it is not a goal and must not grow into a refactor.
9. **Any `git push`, remote operation, or non-local merge.**

---

## 9. Files this round is expected to touch

| file | change |
|---|---|
| `cpp/denner_1d/src/acid.cpp` | `ACID_MBAL` declaration + accumulations at `:1261`/`:1328` + the budget print after `:2585`; *(conditional)* `ACID_YADV_REBUILD_ADV` at `:1320-1332` |
| `scripts/yadv_r9_sweep.py` | `ACID_ENV_VARS` += `"ACID_MBAL"` *(+ the Stage-2 flag if it exists)* — **no other change** |
| `scripts/yadv_r27_case15.py` | **new**: the Stage-0 per-predicate + mass + floor table, all 7 configs (`subprocess.run(capture_output=True)`, never shell redirection) |
| `docs/YADV_RESEARCH.md` | new §37 (round 27): the §3 tables, the Stage-1 budget, the §4.5 core-jet characterisation, the correction of §17.4's config scope |
| `docs/YADV_ROADMAP.md` | control block (`round_counter: 27`, `next_task`, `consecutive_failures` per §7) + one history line |
| `docs/YADV_ROUND_27_PLAN.md` | **new**: this document |
| `papers/*_needed.md` | up to 4 new stubs per §5.2 (or real summaries if the PDFs download) |

**Not touched, under any outcome:** `cpp/denner_1d/src/cases.cpp`, `cpp/denner_1d/src/validation.cpp`, `cpp/denner_1d/include/denner1d/types.hpp`, the repo-root strays `3,` and `=150`.

---

### Critical Files for Implementation

- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/cpp/denner_1d/src/acid.cpp` (flag decl near `:647`; `rho_star` at `:1256-1265`; recovery site `:1266-1275`; Eqs.43-44 rebuild `:1320-1332`; continuity residual + ACID face flux `:1696-1713`; floor `:2344`/`:2536`; accept `:2585`)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/cpp/denner_1d/src/validation.cpp` (case15 gate `:684-730`, metric definitions `:18-51`, `:317-359`, `:405-412` — **read-only reference, do not edit**)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/cpp/denner_1d/src/cases.cpp` (`c15` `:493`, IC `:682-691`, `reference_state` `:751-753`, `computed_reference` `:423-437` — **read-only reference, do not edit**)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/scripts/yadv_r9_sweep.py` (`ACID_ENV_VARS` `:37-42`, `EXPECTED` `:62-70`, `do_sweep` `:117-142`, `do_verify` `:150-213`)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-27/solver_4eq_mass/docs/YADV_RESEARCH.md` (§17.4 to be config-scoped; §26.1 "vacuum blister" — the prior art this round generalises)