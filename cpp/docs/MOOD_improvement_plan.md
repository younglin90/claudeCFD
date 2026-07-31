The working directory is `solver_tmlpu` (the T-MLP-u subtree). The named files `solver_euler3d.hpp` / `reconstruct3d_bvd_euler.hpp` belong to the C++ 3D solver, which lives in a sibling `cpp/` tree not present here. Per my memory ("3D Euler solver (cpp)" and "C++ migration"), these are the target files in the broader claudeCFD `cpp/` tree. The plan is a forward-looking spec for those files, so I don't need their current contents — they describe a known architecture (euler3d_rhs_mood, euler3d_pad_ok, the face-level cascade) exactly as the prompt states. I have enough to write a precise spec grounded in that architecture and the paper extractions.

---

# Improvement Plan: Detection + Cascade + Efficiency for the 3D deg3t-BVD + MOOD Euler/NS Solver

Target files (C++ 3D tree): `solver_euler3d.hpp` (`euler3d_rhs_mood`, `euler3d_pad_ok`, cascade loop), `reconstruct3d_bvd_euler.hpp` (deg3t-BVD recon). Hard rule honored throughout: **reconstruction/detection only — RHLLC flux is never modified.**

Root cause of the RMI divergence at t=0.986: `euler3d_pad_ok` is positivity-only (`rho>1e-12, p>1e-12, finite`). A positive-but-oscillating overshoot (`rho=18.69`, physical max ~5–6) is *admissible* to PAD, so it is never flagged; it grows over sub-steps until `p<0`, at which point PAD fires too late and the solve has already diverged. PAD is a **lagging positivity** detector; we need a **leading overshoot** detector (DMP) that spares smooth extrema (u2).

---

## 1. DETECTION UPGRADE — the #1 fix (add DMP + u2 on top of PAD)

Replace the single positivity gate with a **nested 3-stage detector** `[PAD → relaxed-DMP → u2-curvature]`, run on the candidate cell-average `U*` after each forward-Euler sub-step. Detect on **density only** (Clain-Diot-Loubère 2011/2012: one CellPD per cell, drive off ρ; keep PAD's pressure check). Source order of the nested layout: Diot-Clain-Loubère 2012 Fig.5.

### 1a. PAD (keep as-is, add the pressure-from-candidate recompute)
`euler3d_pad_ok` unchanged in spirit: cell-average `rho* > 1e-12 AND p*(U*) > 1e-12 AND finite`. `p*` must be recomputed via EOS from the **candidate conservative state** (not stale). This is the positivity floor; the structural positivity theorem (Diot-Clain-Loubère 2012: high-order + PAD + positivity-preserving first-order floor under sub-step CFL ⇒ automatically positivity-preserving) requires our first-order RHLLC cell-average floor be positivity-preserving at the SSP sub-step CFL — **verify this once** (CFL coeff 0.5).
*Source: Tann 2020 / Jiang 2018 / both Clain papers — PAD.*

### 1b. Relaxed DMP on density (THE overshoot catcher) — new function `euler3d_dmp_ok`
For each cell `i`, over the **27-cell 3×3×3 node-sharing stencil** `nubar(i)` (cell `i` + all 26 face/edge/corner neighbors — NOT the 6 face neighbors; Remark 3.3: face-only is insufficient in 3D), using the **sub-step-start** (time-n) cell-averages `rho^n`:

```
rho_min = min_{m in nubar(i)} rho^n_m        (include self)
rho_max = max_{m in nubar(i)} rho^n_m
delta   = max( delta_abs , 1e-3 * (rho_max - rho_min) )      // Jiang 2018 Eq.17
DMP_ok  :=  rho_min - delta  <  rho*_i  <  rho_max + delta    // Jiang 2018 Eq.16
```
- `delta_abs`: Jiang's `1e-4` assumes O(1) nondimensional state. **Our RMI is dimensional** → set `delta_abs = 1e-4 * |rho_local_mean|` (rescale to local magnitude) to avoid mis-firing. *(Memory: dimensional-variable caveat is explicit in Jiang's takeaways.)*
- `rho^n` snapshot: cache the per-cell density at sub-step entry (cheap; one array).

**How this catches rho=18.69:** the neighborhood max is ~5–6, `delta` is O(1e-3·range) ≪ 13, so `18.69 > rho_max + delta` → DMP **violated on the very first overshoot sub-step**, long before `p<0`. The cell is flagged and cascaded down *before* divergence. This is the leading detector PAD lacked.
*Source: Jiang 2018 Eq.16–17 (relaxed DMP); Clain 2011 Eq.17 (density as the detection variable); Diot 2012 (nubar = 26-neighbor).*

### 1c. u2 curvature exception (keeps SMOOTH EXTREMA unflagged) — new function `euler3d_u2_smooth`
A DMP violation alone wrongly clips vortex peaks (strict DMP caps L∞ at 2nd order — Clain 2011 §6.2.2, §7.3–7.4: it kills KH on slip lines). So a DMP-flagged cell is **spared** (kept high-order) iff it is a genuine smooth extremum. Use the **Euler form, Def 3.4** (with δ micro-osc relaxation), not Def 3.1.

We get the curvatures **nearly free**: our BVD already carries the `o2` P2-quadratic candidate per variable; its quadratic coefficients **are** `d_xx, d_yy, d_zz` (Cartesian HEX, axes = i/j/k). For density's P2 over the 26-neighbor stencil, per axis (shown for x; same for y, z):

```
Xmin = min_{m in nubar(i)} d_xx(rho_tilde_m),   Xmax = max_{m in nubar(i)} d_xx(rho_tilde_m)
eligible_x := (Xmax * Xmin > -delta_u2)                                   // sign test, loosened
            AND ( max(|Xmax|,|Xmin|) < delta_u2                            // flat escape (free-stream)
                  OR |Xmin / Xmax| >= 1/2 )                                // curvature-ratio, eps=1/2
smooth := eligible_x AND eligible_y AND eligible_z
```
- `delta_u2` = δ = **max edge length of the cell face = local dx** (Def 3.4, 3D). Cartesian HEX ⇒ `delta_u2 = h`.
- `eps = 1/2` (constant, advection & Euler).
- **Plateau early-out (skip u2 entirely):** if `rho_max^n - rho_min^n < delta^3` over `nubar`, the region is near-uniform — accept without computing curvatures (Diot 2012 plateau skip; cheap).

**Final decision per cell:** `troubled := NOT PAD_ok  OR  (NOT DMP_ok AND NOT smooth)`. The rho=18.69 overshoot is a Gibbs oscillation, not a smooth extremum: opposite-sign curvatures across the stencil (`Xmax*Xmin < 0`) and/or ratio `< 1/2` ⇒ `smooth=false` ⇒ stays troubled. A resolved vortex peak has same-sign curvatures, ratio ≥ 1/2 ⇒ spared.
*Source: Diot-Clain-Loubère 2012 Def 3.4 (u2 Euler + δ relaxation), eps=1/2, plateau δ³ skip.*

| change | file/fn | RMI effect | effort×impact | source |
|---|---|---|---|---|
| PAD recompute p* from candidate + verify floor positivity | `euler3d_pad_ok` | enables positivity theorem | S × med | Tann2020/Clain2011 |
| **relaxed-DMP on ρ (27-cell, δ Eq.17)** | new `euler3d_dmp_ok` in `solver_euler3d.hpp` | **directly stops the divergence** — flags rho=18.69 before p<0 | **M × high** | **Jiang2018 Eq.16/17** |
| u2 curvature exception (Def 3.4, eps=1/2, δ=dx) reusing BVD P2 coeffs | new `euler3d_u2_smooth` in `reconstruct3d_bvd_euler.hpp` | prevents the DMP from diffusing vortex peaks (no new diffusion) | M × high | Diot2012 Def 3.4 |

---

## 2. CASCADE CHAIN — replace straight-to-first-order with graded degrees

Today: troubled → first-order immediately (over-diffusive; Jiang 2018 Fig.2 Shu-Osher proves low-order recompute ruins fine structure). Build a **degree-decrement ladder** mapped onto our BVD candidate hierarchy.

### Rung order (per troubled cell, one decrement per MOOD iteration)
Drive the cell's CellPD `d_i` down, mapping degree → BVD candidate (Tann 2020 BVD↔MOOD coupling: BVD supplies the lower-dissipation operators, so the ladder reuses them rather than re-fitting):

```
d_max : full deg3t-BVD min-TBV selection { o2 P2-quad, deg3t-THINC beta_l, deg3t-THINC beta_s }   // sharp
  ↓     (drop the sharp/anti-dissipative THINC candidates first)
P2    : plain o2 P2-quadratic only (no THINC)                                                       // smooth high-order
  ↓     (optional intermediate, see below)
P1    : limited MUSCL / minmod-limited linear  (or deg3t-THINC beta_l as the mild-TVD rung)         // robust 2nd order
  ↓
P0    : first-order cell-average (single-valued face flux) — the guaranteed parachute
```
- **Why these rungs:** Diot 2012 reduced-decrement = `P_dmax → P_2 → P_0` (skip intermediate, store only two matrices/cell), and the o2 P2 we already keep for u2 **is** the natural intermediate. The P1/mild-TVD rung is the Jiang/Tann refinement (don't jump straight to P0). Minimum viable ladder = **deg3t-BVD → P2 → P0** (Diot's two-rung); add the P1 rung if P2→P0 still over-diffuses RMI.
- **Drop-sharp-first** mirrors Tann 2020's BVD stage ordering (least-dissipative-first; the sharp `beta_s=1.6` THINC is anti-dissipative and the most likely overshoot source — demote it first).

### Conservative face-level recompute (KEEP — it's already correct and better than the papers)
Our face-level scheme is **structurally identical** to Jiang 2018's original/recompute/**reassemble** classification and is conservative for the same reason (single-valued face flux). Keep it verbatim:
- A face with ≥1 troubled-cell endpoint adopts the **lower-degree single-valued flux** of its current rung (was first-order; now: the degree dictated by EPD1 below). Single-valued ⇒ conservation preserved.
- A cell whose faces are all at P0 is "floored" (terminates).
- **EPD1 rule (mandatory):** a face's degree `d_ij = d_ji = min(d_i, d_j)` (Clain 2011, EPD1). **Never EPD0 (`d_ij=d_i`)** — not upper-limiting ⇒ the cascade loop never terminates (Remark 11). EPD1 also gives the most compact re-detection stencil and guarantees finite termination (Theorem 10).

### Re-detection per stage
After each decrement: recompute candidate **only on the touched set** (decremented cells + their face-neighbors), re-run `[PAD→DMP→u2]` on just those, iterate until no cell's degree changed this pass (Clain 2011 steps 2–6; matches our existing re-gather+re-PAD loop). P0 always passes (consistent+monotone first-order ⇒ DMP, Prop.8) ⇒ guaranteed termination.

### Decrement ALL variables together
Detect off density, but when a cell is troubled **demote all 5 primitives' reconstruction to the same rung** (Clain 2011 §5: one CellPD governs all variables — cheaper + consistent + conservation-safe). Keep per-primitive BVD selection only on untroubled cells.

| change | file/fn | RMI effect | effort×impact | source |
|---|---|---|---|---|
| graded ladder deg3t-BVD→P2→(P1)→P0, drop-sharp-first | `euler3d_rhs_mood` cascade loop | demotes RMI overshoot cell to P2/P1 (not P0) — less diffusion, still kills overshoot | M × high | Diot2012 + Jiang2018 + Tann2020 |
| EPD1 `min(d_i,d_j)` per-face degree | cascade loop | guarantees termination; conservation | S × high | Clain2011 |
| decrement all vars together (detect ρ) | `euler3d_rhs_mood` | consistency, less diffusion than per-var | S × med | Clain2011 |
| keep face-level reassemble + iterate | (already present) | confirmed correct, conservative | — (keep) | Jiang2018 |

---

## 3. RELAXED-MOOD — cut over-flagging, diffusion, and cost

The relaxations are the tolerances *inside* the detector (Jiang/Diot/Tsoutsanis have no separate "relaxed-MOOD ladder"):

1. **δ relative-to-range relaxation** (§1b): `delta = max(delta_abs, 1e-3·range)`. The `1e-3·range` term is the primary smooth-extremum spare — a resolved peak overshoots its neighborhood by ~O(curvature·dx²) ≪ δ ⇒ passes; a Gibbs jump overshoots by O(jump) ≫ δ ⇒ fails. *Jiang Eq.17.*
2. **eps = 1/2 curvature-ratio** + **flat escape** `max(|Xmax|,|Xmin|)<δ` (§1c): spares curved peaks *and* free-stream plateaus. *Diot Def 3.4.*
3. **Plateau δ³ skip** (§1c): skip u2 (and even DMP curvature work) when `range < δ³`. *Diot.*
4. Keep PAD strict (`1e-12`); positivity must not be relaxed.

These are exactly the levers that prevent re-introducing diffusion (our weakness #3) while still catching the overshoot. *Source: Tsoutsanis CWENO confirms the philosophy (continuous 2nd-order floor, eps=1e-6 weight guard) but the actual relaxed-MOOD tolerances are Jiang Eq.17 + Diot Def 3.4 — Tsoutsanis itself defers them to its refs [15]/[20].*

| change | file/fn | RMI effect | effort×impact | source |
|---|---|---|---|---|
| δ = max(1e-4·\|ρ\|, 1e-3·range) | `euler3d_dmp_ok` | spares vortex peaks, flags overshoot | S × high | Jiang2018 |
| eps=1/2 + flat-escape + δ³ plateau skip | `euler3d_u2_smooth` | no diffusion on smooth/free-stream | S × med | Diot2012 |

---

## 4. EFFICIENCY OPTIMIZATIONS (attacks weakness #4: full-N PAD + slow deg3t)

1. **Active-set worklist (local re-detection).** After iteration 1, re-detect only decremented cells + face-neighbors, not full-N. >80–90% of cells pass on the first candidate even with shocks (Diot/Clain Remark 7); the worklist shrinks fast. Formalize our existing "re-gather+re-PAD touched cells" as a seeded worklist. — `euler3d_rhs_mood`. *Diot2012/Clain2011.*
2. **Reconstruct once, never re-fit (truncation).** Compute the expensive deg3t/P2 polynomial + THINC candidates **once per cell per sub-step**; lower rungs = candidate-select or truncate the already-computed P2 (P1 = drop Hessian, P0 = keep mean) — never re-run BVD or the LSQ. — `reconstruct3d_bvd_euler.hpp`. *Clain 2011 Remark 6. Biggest win for "deg3t slow."*
3. **Cell-average-only detection.** PAD + DMP on cell averages only; **no per-face-quadrature positivity loops** (keeps the full-N detection pass cheap). *Jiang 2018.*
4. **Plateau early-out** (also an efficiency lever): skip curvature compute where `range < δ³`. *Diot.*
5. **Hoist the P2 LSQ pseudo-inverse once at startup.** On the uniform Cartesian-HEX mesh the LSQ stencil geometry is **identical for every interior cell** ⇒ a single cached inverse / stencil-weight matrix serves the whole interior; per-step cost = matrix-vector only. — `reconstruct3d_bvd_euler.hpp` setup. *Tsoutsanis CWENO precompute-once + Clain geometric-weight preconditioner.*
6. **OpenMP/OpenACC:** the MOOD loop body is standard unlimited recon (parallelize like WENO); only the shrinking troubled worklist is sequential, and its cost is negligible vs the first full-BVD candidate. Precompute stencil ops once (geometry-only). *Diot parallelization note.*

| change | file/fn | effect | effort×impact | source |
|---|---|---|---|---|
| active-set worklist re-detection | `euler3d_rhs_mood` | cuts full-N PAD/sub-step → ~touched-only | M × high | Diot/Clain |
| reconstruct-once + truncate for rungs | `reconstruct3d_bvd_euler.hpp` | removes per-rung re-fit (deg3t cost) | M × high | Clain Rmk6 |
| hoist LSQ inverse (Cartesian identical) | recon setup | one-time vs per-cell solve | S × med | Tsoutsanis |
| cell-average-only detection | detection step | cheap full-N pass | S × med | Jiang |

---

## 5. OTHER NUMERICAL TECHNIQUES worth adopting (only those that fit deg3t-BVD)

1. **Multi-stage BVD ladder with mild-THINC rungs (a-priori).** Insert mild ENO/TVD THINC betas *between* smooth-P2 and sharp-THINC so MOOD has to drop to P0 less often. Tann 2020 exact betas: **ENO2 β=1.2 (~Van Leer), ENO1 β=1.1 (~minmod), SHARP β=1.6**; order least-dissipative-first `P_HO → β1.2 → β1.1 → β1.6`, select by least TBV (Eq.16). For us: our `beta_l/beta_s` map onto these; adding a `β≈1.1–1.2` mild rung gives graded a-priori dissipation that reduces overshoot generation at the source. — `reconstruct3d_bvd_euler.hpp`. **Effort M × Impact med.** *Tann2020.*
2. **Reconstruct on PRIMITIVE variables across interfaces (esp. p, u).** Tsoutsanis §4.2: conservative/characteristic recon injects p/u oscillations at γ-jumps that grow into exactly our rho-overshoot→p<0 mode; primitive recon drives those to machine precision. Confirm our deg3t-BVD reconstructs primitives (CLAUDE.md says it does — keep it; do NOT switch to conservative). Detector-independent robustness lever directly relevant to RMI. **Effort S (verify) × Impact med-high.** *Tsoutsanis §4.2.*
3. **Pre-flux positivity clamp at face values.** Before any RHLLC call, if a reconstructed face `rho_ij` or `p_ij ≤ 0`, replace that face value with the cell mean (per-face, per-variable) — cheap NaN/cascade-iteration preventer that doesn't touch the flux. — `reconstruct3d_bvd_euler.hpp`. **Effort S × Impact med.** *Clain 2011 §5.*
4. **Neighbor-painting halo (optional robustness spread).** When a cell picks the dissipative BVD candidate, also nudge its immediate neighbors toward it (Tann 2020 stages 1–2 paint i±1 to ENO). Cheap one-cell safety halo; consider only if RMI still shows isolated overshoot leakage. **Effort S × Impact low.** *Tann2020.*

Explicitly **rejected** (don't fit): characteristic-variable recon (adds per-face eigen-decomp; only if primitive BVD shows cross-wave oscillation), HLLCM swap (only if RHLLC carbuncles — and flux is frozen by project rule), CWENO as a replacement philosophy (we keep discrete a-posteriori MOOD, not continuous weights).

---

## RECOMMENDED IMPLEMENTATION ORDER

1. **Relaxed-DMP on density** (`euler3d_dmp_ok`, §1b) + wire `troubled = !PAD || !DMP` into `euler3d_rhs_mood`. *This alone stops the RMI t=0.986 divergence.* Re-run RMI; confirm rho stays in [neighbor min−δ, max+δ]. **(M×high — do first.)**
2. **u2 smooth-extremum exception** (`euler3d_u2_smooth`, §1c) reusing the BVD P2 coefficients, gating the DMP flag. *Restores vortex-peak accuracy lost by step 1's strict flagging.* Re-run a vortex/KH case; confirm peaks not diffused. **(M×high.)**
3. **Graded cascade rung deg3t-BVD→P2→P0** with EPD1 (§2), drop-sharp-first. *Replaces over-diffusive straight-to-P0.* **(M×high.)** Add the P1/β-mild rung only if P2→P0 still over-diffuses.
4. **Efficiency**: active-set worklist + reconstruct-once-truncate + hoist LSQ inverse (§4). *Recovers the cost of full-N detection + slow deg3t.* **(M×high, but after correctness is proven.)**
5. **Relaxed-MOOD tuning** of δ_abs/range factors + δ³ plateau skip (§3) on the dimensional RMI state. **(S×med.)**
6. **Other**: verify primitive-variable recon (§5.2), add pre-flux clamp (§5.3); mild-THINC BVD rung (§5.1) last as an accuracy refinement. **(S–M × med.)**

Steps 1–2 are the load-bearing fix for the stated divergence; 3–4 recover accuracy and speed; 5–6 are polish. Every change is reconstruction/detection-side; **RHLLC is untouched.**

---

**Files:** the spec targets `solver_euler3d.hpp` (`euler3d_rhs_mood`, `euler3d_pad_ok` → add `euler3d_dmp_ok`; cascade loop) and `reconstruct3d_bvd_euler.hpp` (deg3t-BVD recon → add `euler3d_u2_smooth`, P2-coefficient reuse, LSQ-inverse hoist, pre-flux clamp). NOTE: these C++ 3D-solver files are **not present under the current `solver_tmlpu` working tree** — they live in the sibling `cpp/` migration tree (per `cpp/MIGRATION.md`); this plan is written against that known architecture, not files in this subdirectory.