# Faithful unstructured port of the structured ASVL single-valued-face limiter

## 0. Guiding principle: port the *inputs*, not the *algorithm*

The Adaptive Single-Valued Limiter (ASVL) of the structured paper is, at its core, a small closed-form **case table** that consumes a fixed, minimal set of scalars per face and returns a single-valued (zero-boundary-variation, BV=0) compressive face state. Its novelty — maximizing compression subject to *(1D TVD-admissible band)* AND *(single-valued face)* by active-set enumeration of patterns a..e — lives entirely in that table.

The design decision that makes an unstructured port both *faithful* and *reviewer-defensible* is:

> **Do not re-derive, re-tune, or relax the case table. Identify the exact set of quantities it consumes, and supply an unstructured-valid, structured-equivalent version of *each input*. The identical case table then runs unchanged, and on a Cartesian grid the inputs collapse to their structured definitions, so the whole pipeline reduces byte-for-byte to the validated structured scheme.**

The case table consumes exactly these per-face quantities (owner cell `C=i`, neighbour `D=j`, face jump `dp = q_D - q_C`):

| # | Input | Structured source | What unstructured breaks |
|---|-------|-------------------|--------------------------|
| I1 | `r_L, r_R` (two one-sided slope ratios) | aligned cells `q_{i-1}, q_{i+2}` | no aligned cells → need gradient-built virtual differences (Pillar 1) — but the naive port puts the vanishing face jump in a denominator (Pillar 2) |
| I2 | `grad_C, grad_D` (the sensor's raw material) | latent central difference, free | must be reconstructed (LSQ/Green-Gauss) — the ONE genuinely new approximation (Pillar 3) |
| I3 | band endpoints + full case table (A,B,C, patterns a..e, Eq.26 pick) | pure algebra of `(r_L,r_R,dp,a)` | nothing — geometry-agnostic, ports verbatim (Pillar 4) |
| I4 | multi-D boundedness of the reconstructed cell | free, via dimension splitting | genuinely multi-D stencil → Goodman–LeVeque forbids a free lunch; need a vertex max-principle cap (Pillar 5) |
| I5 | system/positivity/conservation closure | dimension-split HLLC on valid states | oblique faces + compressive recon → characteristic projection, sonic floor, primitive positivity floor, telescoping flux (Pillar 6) |

Each of Pillars 1–6 supplies one input, proves its exact reduction to the structured value on a uniform Cartesian grid with a central-difference gradient, and (for the three genuine additions NVD, IDW, vertex-MLP) names the Cartesian-implicit property it restores. The Master Reduction Theorem (§7) chains these into a single byte-exact "use-as-is" guarantee.

Notation: `dp := q_D - q_C`; `d := x_D - x_C`; skew offsets `d_Cf := x_f - x_C`, `d_Df := x_f - x_D` (`x_f` = face centroid); `φ_mm(r)=max(0,min(r,1))`; `φ_sb(r)=max(0,min(2r,1),min(r,2))`.

---

## 1. Pillar 1 — the smoothness sensor and its face-jump fragility

**Structured.** ASVL samples the sensor from four aligned averages on the line through face `i+1/2`:

```
r_L = (q_i - q_{i-1})/dp   [Eq.11],   r_R = (q_{i+2} - q_{i+1})/dp   [Eq.12],   dp = q_{i+1}-q_i
q_mm^L = q_i + ½φ_mm(r_L)dp,  q_sb^L = q_i + ½φ_sb(r_L)dp   (mirror on R)   [Eqs.9-10]
```

The crux: numerator `q_i-q_{i-1}` and denominator `q_{i+1}-q_i` of `r_L` are the *same* first-difference `Δq` sampled at *adjacent* half-points.

**Unstructured.** No aligned cells → virtual differences from the LSQ gradient (Darwish–Moukalled Eq.16):

```
Δ_minus = 2(grad_C·d) - dp   →  r_L = 2(grad_C·d)/dp - 1
Δ_plus  = 2(grad_D·d) - dp   →  r_R = 2(grad_D·d)/(-dp) - 1
q_U = q_D - 2(grad_C·d)   (virtual upwind);   ½dp → skew increment grad_C·d_Cf
```

**Exact reduction** (Cartesian, `grad_C=(q_{i+1}-q_{i-1})/2h`, `d=(h,0)`): `grad_C·d = (q_{i+1}-q_{i-1})/2`, so `Δ_minus = q_i-q_{i-1}` (★), `q_U = q_{i-1}` (virtual upwind IS the real aligned cell), and `r_L = (q_i-q_{i-1})/(q_{i+1}-q_i) =` Eq.11. Neighbour side → Eq.12. ∎

**Fragility (root cause of cone/hump wobble).** Off-Cartesian, `grad_C` carries an *independent* LSQ error `e_grad = O(h)` decoupled from the solution's extremum limit `dp→0`:

```
r_L = Δ_minus/dp = finite + 2 e_grad/dp = O(1) + O(h)/O(h) → O(1) noise, sign-indefinite
```

A non-vanishing gradient error over a vanishing face jump ⇒ spurious limiting of a smooth extremum. **Not a gradient bug**: even an optimal `O(h²)` LSQ gradient has a slope error whose vanishing (mesh refinement) is decoupled from `dp→0` (solution property). Dimension splitting tied those limits together for free; unstructured severs them.

---

## 2. Pillar 2 — co-vanishing repair via NVD normalization

Same sensor, re-expressed so the vanishing quantity leaves the denominator. Gradient span `G_C := q_D - q_U = 2(grad_C·d)`:

```
phiT_C = (q_C - q_U)/(q_D - q_U) = 1 - dp/(2 grad_C·d)   (code :1877, span=2 grad_C·d)
r_L = phiT_C/(1 - phiT_C),   phiT_C = r_L/(1 + r_L)      (exact bijection)
```

TVD region ↔ NVD CBC region `0<phiT<1`, so **minmod/superbee and the entire ASVL case table re-express as functions of `phiT` with zero information loss** — no new limiter.

**Principled, not a patch.** `dp` now sits in the numerator, the gradient span in the denominator:

- *Boundedness through the extremum:* `dp→0, grad_C≠0 ⇒ phiT_C→1` smoothly = the no-limiting/high-order state. No `0/0`.
- *Correct freeze location:* denominator vanishes only where `grad_C` itself vanishes = genuinely flat = the correct place to stop limiting (guard `|span|<ε → upwind`).
- *Vanishing sensitivity:* `d phiT_C = (dp/(2 grad_C·d)²) d(grad_C·d)` → sensitivity `→0` as `dp→0`, the exact opposite of `r`'s divergence.

**Reduction:** on Cartesian, `span=q_{i+1}-q_{i-1}`, `q_U=q_{i-1}`, `phiT_C=(q_i-q_{i-1})/(q_{i+1}-q_{i-1})` = the structured Gaskell–Lau normalized variable; the bijection returns Eq.11.

Van-Albada regularization (num, denom co-vanish) is a valid but less-principled fallback — it fixes the `0/0` symptom, not the coordinate. **NVD is chosen; parameter-free.**

---

## 3. Pillar 3 — the gradient: the single genuinely-new approximation

The one quantity the Cartesian grid supplied latently. Two admissible operators (both linearly exact, ≥`O(h)`):

**(a) IDW-LSQ (primary):** `grad_C = M⁻¹b`, `M = Σ_j w_j d_j d_jᵀ`, `b = Σ_j w_j Δq_j d_j`, `w_j=1/|d_j|^p`. **Column-norm precondition** `s_k=(Σ_j w_j d_{j,k}²)^{-1/2}` bounds `cond(M̃)` independent of aspect ratio, removing anisotropic bias.

**(b) Green-Gauss + skew:** `grad_C=(1/V_C)Σ_f q_f n_f A_f`, `q_f = ½(q_C+q_D)+½(grad_C+grad_D)·(x_f-½(x_C+x_D))`, 1–2 passes.

**Reduction:** symmetric LSQ normal equations decouple per axis ⇒ x-component `=(q_{i+1}-q_{i-1})/2h` (central difference) with `e_grad=0` identically. Off-limit `e_grad=O(h)` is not a defect but the irreducible cost of missing aligned neighbours (no free lunch); it enters the face value only as `O(h²)`, preserving second order.

Restated in gradient terms (`grad_C·d = D + e`, `e=O(h)`):

```
naive r:  r_L = (2D+2e-dp)/dp = finite + 2e/dp → O(1/h) noise  (blows up)
NVD:      d phiT_C = (dp/(2 grad_C·d)²) d(grad_C·d) → 0 as dp→0  (safe)
```

IDW+column-norm restores the *isotropy + linear-exactness* the aligned stencil had for free.

---

## 4. Pillar 4 — band, overlap, single-valued pick (the ASVL novelty)

Everything downstream of the two limiter values is **geometry-agnostic** and ports verbatim; the claim is an *invariance*, not a re-derivation.

```
BAND_L = [min,max](q_mm^L, q_sb^L),   BAND_R = [min,max](q_mm^R, q_sb^R)
A = φ_sb(r_L)+φ_sb(r_R)    (overlap ⇔ A ≥ 2)
B = (q_sb^L-q_mm^R)(q_sb^L-q_sb^R),   C = (q_sb^R-q_mm^L)(q_sb^R-q_sb^L)
```

| Condition | Pattern | Action |
|-----------|---------|--------|
| `A<2` | a | bands disjoint → each side independent superbee |
| `A≥2, B≥0, C≥0` | e | shared midpoint `q*=½(q_C+q_D)` |
| `A≥2, B≤0, C≤0` | b | `(e0,e1)=(q_sb^L,q_sb^R)` |
| `A≥2, B≤0, C>0` | c | `(e0,e1)=(q_mm^L,q_sb^L)` |
| `A≥2, B>0, C≤0` | d | `(e0,e1)=(q_mm^R,q_sb^R)` |

Single-valued upwind-compressive pick (Eq.26), `ξ_min/ξ_max = min/max(e0,e1)`:

```
q* = ξ_max  if  a·(q_C-q_D)<0  else  ξ_min,    then  q^L = q^R = q*
```

**Zero-BV is an algebraic identity, hence geometry-invariant:** `BV_f = |q*-q*| = 0` on every interior face (b,c,d,e) — no reference to normal, skewness, or topology. The *only* geometry-dependent choice, the skew increment `grad_C·d_Cf` replacing `½dp`, is a consistent generalization (genuine 2nd-order face value; reduces to `½dp` at `r=1`).

**Reduction:** with `r_L,r_R` reduced (Pillar 1) and the same `dp`, every endpoint, `A,B,C`, the ladder, and the Eq.26 pick are byte-for-byte structured. The verbatim-ness *is* the faithfulness proof.

**Why zero-BV isn't self-stabilizing:** a single-valued face makes `HLLC(q*,q*)=F(q*)` a *central, zero-dissipation* flux. Stabilized by two closures provably inert on Cartesian: (i) the **upwind** Eq.26 endpoint (midpoint verified divergent on LeVeque); (ii) the vertex-MLP cap of Pillar 5 (default on; without it `L1→10^63`). Both feed this pillar only through scalar inputs.

---

## 5. Pillar 5 — vertex-MLP / LMP cap: the multi-D max principle

**Cartesian supplies it free:** the dimension-split update is a sum of two 1D TVD operators (Harten per axis); axis-aligned nodes decouple, so the multi-D max principle is the direct product of two 1D ones (`α_C≡1`).

**Unstructured requires a separate cap — logically.** Faces are arbitrarily oriented; the planar polynomial couples all faces. **Goodman–LeVeque (1985):** a genuinely multi-D TVD FV scheme is at most 1st order. Contrapositive: no collection of per-edge 1D TVD conditions can bound a 2nd-order multi-D reconstruction. A separate, weaker-than-TVD (hence theorem-escaping) constraint is required = the vertex max-principle.

*Counterexample (triangle, `q̄_C=0`, neighbours `+1,+1,-1`):* each face's case table can return a legal in-band value, yet the planar polynomial reaches a vertex `> q_max=+1` along the reinforced gradient diagonal. Every 1D condition passes; the multi-D bound fails.

**Cap (Barth–Jespersen / Venkatakrishnan):**

```
Δ_v = grad_C·(x_v-x_C),   q_max/q_min = max/min_{n∈N(C)∪{C}} q̄_n
α_v = 1 (Δ_v=0);  min(1,(q_max-q̄_C)/Δ_v) (Δ_v>0);  min(1,(q_min-q̄_C)/Δ_v) (Δ_v<0)
α_C = min_v α_v      (wrapper:)   q*_f^{capped} = q̄_C + α_C(q*_f - q̄_C)
```

**Reduction (non-binding on smooth aligned data):** corner increment `Δ_v = ¼(q_{i+1,j}-q_{i-1,j})+¼(q_{i,j+1}-q_{i,j-1})`; since `q̄_C+Δ_v ∈ [q_min,q_max]`, `α_v≥1 ⇒ α_C=1 ⇒` the wrapper is the identity. Activates (`α_C<1`) only on non-aligned stencils where the product-of-1D breaks. Empirically **net anti-diffusive** (`L1~1.35e-2`, `~0.46×` MUSCL) — mandatory, not over-diffusive.

---

## 6. Pillar 6 — Euler / positivity / conservation assembly

**(a) Roe characteristic projection.** Scalar (`nvar<4`): projection = `1×1` identity → LeVeque untouched. Euler: decompose `(dρ,du,dv,dp)` at the Roe-averaged face into 4 waves:

```
a0=(dp-ρ_R c_R dun)/(2c2) [ac-],  a1=dρ-dp/c2 [entropy],  a2=dut [shear],  a3=(dp+ρ_R c_R dun)/(2c2) [ac+]
```

Apply the *full* ASVL (Pillars 1–5) to **each amplitude**, wave-family limiter: acoustic→van Leer (dissipative), entropy→compressive, shear→central. Recompose `dρ=a0+a1+a3, dp=c2(a0+a3), dun=(c_R/ρ_R)(a3-a0), dut=a2`. *Faithful:* restores the x/y diagonalization dimension-splitting gave for free (component-wise on an oblique face makes spurious entropy); vanishes on scalar.

**(b) Sonic-floored upwind pick.** `a0=UPW_A0·c_R`, `w=½(1+af/(|af|+a0))→0.5` (central) at a sonic point; unchanged where `|af|≫c`. Restores well-definedness the smooth field had.

**(c) Positivity floor on the FACE STATE (not flux), primitive vars** (`{ρ>0,p>0}` = complete admissible set): `ρ_face≥floorf·ρ_cell, p_face≥floorf·p_cell` (`floorf~0.2`). Positivity-preserving HLLC + CFL keeps cell average admissible; flooring the *state* leaves conservation intact.

**(d) Single-valued → consistent flux → conservation.** One `hllc_euler2d(WL,WR,n)` per face, telescoped:

```
U_i^{n+1} = U_i^n - (dt/|V_i|) Σ_{f∈i} orient(i,f) F_f A_f,   orient(owner)=+1, orient(nbr)=-1
```

Conserved to machine precision *for any reconstruction*. Zero-BV is a bonus: `HLLC(q*,q*)=F(q*)` = exact central physical flux (low diffusion), never violating conservation; supersonic HLLC is fully upwind by the `SL,SM,SR` ladder (carbuncle controlled by HLL blend / sonic guard). `dt = cfl·h_min/max_face(|u·n|+c)`, `cfl≤0.4` (SSP-RK2).

---

## 7. Master Reduction Theorem

> **On a uniform Cartesian grid with a central-difference cell gradient, for the scalar model, the full unstructured pipeline (§§1–6) produces face states, fluxes, and updates byte-for-byte identical to the structured ASVL** (`struct_leveque.py::recon_abvd`).

**Proof** (x-line face `i|i+1`): **P3** symmetric LSQ ⇒ `grad_C·d=(q_{i+1}-q_{i-1})/2`, `e_grad=0`. **P1** ⇒ `r_L=`Eq.11, `r_R=`Eq.12, `q_U=q_{i-1}`. **P2** ⇒ `phiT_C` = structured Gaskell–Lau variable; bijection returns the same `r`. **P4** ⇒ endpoints, `A,B,C`, ladder, `q*` all byte-identical (`a=U`); skew increment `→½dp` at `r=1`, transverse part `=0`. **P5** ⇒ `q̄_C+Δ_v∈[q_min,q_max] ⇒ α_C=1 ⇒` cap = identity (extremum: `ψ_LMP→0` pins to `q̄_C`, as `φ(r<0)=0` does). **P6** ⇒ Roe = `1×1` identity, positivity floor never binds, `HLLC` collapses to scalar upwind `af·(af≥0?qL:qR)`. Chaining P3→P1→P2→P4→P5→P6, every consumed quantity equals its structured counterpart and the machinery is identical. ∎

**No benchmark on a Cartesian grid can distinguish the port from the original.** The three additions switch *on* only off-Cartesian, precisely where the structured guarantee lapses.

---

## 8. Reviewer defense — each addition restores a Cartesian-implicit property

| Addition | Restores | Why forced, not tuned |
|----------|----------|-----------------------|
| **NVD** | Clean co-vanishing of the sensor at a smooth extremum (structured `r→1` because num/denom are the same first-difference at adjacent points) | LSQ-gradient error `O(h)` decoupled from `dp→0` ⇒ `O(1/h)` noise. NVD moves `dp` to numerator, non-vanishing gradient span to denominator. Parameter-free; reduces to structured normalized variable; vanishing sensitivity. Van-Albada rejected (fixes `0/0`, not the coordinate). |
| **IDW-LSQ + column-norm** | Isotropy + linear-exactness of the aligned central stencil (`e_grad=0` on Cartesian) | Missing aligned neighbours force the ONE new approximation (no free lunch). Column-norm bounds `cond(M)` vs aspect ratio. Linear-exact ⇒ `e_grad=0` on linear fields ⇒ exact reduction. |
| **Vertex-MLP / LMP** | The multi-D discrete max principle dimension-splitting gave as the product of two 1D TVD principles | Goodman–LeVeque **proves** 1D per-face bands can't bound a 2nd-order multi-D recon. Tightest linear cap consistent with a discrete max principle. Provably `α_C=1` (identity) on smooth aligned data; activates only where an uncapped recon injects a new extremum. Net anti-diffusive; mandatory (`L1→10^63` without). |

**"You never needed it on structured":** correct — stencil alignment and dimension splitting supplied co-vanishing and the multi-D max principle for free. Each addition restores *exactly* that property when alignment/splitting is gone, and each is provably inert on the Cartesian reduction (§7). **We claim no new limiter — a conditioning-correct, exactly-reducible port of the validated one, plus the single genuinely-new gradient it needs.**

**"Zero-BV → central/non-TVD":** we never claim global TVD (Goodman–LeVeque forbids it). We prove (i) bit-exact reduction to the validated structured TVD scheme on Cartesian, and (ii) a discrete *maximum principle* via the vertex-MLP (the correct multi-D TVD generalization). Positivity is a genuine sufficient condition (primitive face-state floor + CFL + positivity-preserving HLLC). Single-valued only makes `HLLC(q*,q*)=F(q*)` the *physical* flux (fully upwind supersonically).

---

## 9. Algorithm (per face, in assembly order)

**Per timestep, per cell `C`:**
1. **Gradient** `grad_C=M⁻¹b` (IDW weights, column-norm precondition). (P3)
2. **Vertex bounds + LMP factor** `ψ_LMP,C = min_v clamp((proj≥0?q_max-q̄_C:q̄_C-q_min)/|proj|,[0,1])`, `proj=grad_C·(x_v-x_C)`. (P5)

**Per interior face `f` (`C=i`, `D=j`):**
3. Geometry `d, d_Cf, d_Df, n, t`; `dp = q_D-q_C`.
4. **[Euler]** Roe averages → wave amplitudes `a0..a3`; run steps 5–9 per amplitude with its family limiter. **[Scalar]** skip (identity). (P6a)
5. **NVD sensor:** `span_C=2 grad_C·d`, `q_U=q_D-span_C`, `phiT_C=1-dp/span_C` (guard `|span|<ε→upwind`); `r_L=phiT_C/(1-phiT_C)`; mirror for `r_R`. (P1–2)
6. **Band endpoints** `q_mm^{L,R}, q_sb^{L,R}`. (P4)
7. **Overlap + case table** `A,B,C` → pattern a/b/c/d/e → `(e0,e1)`. (P4)
8. **Single-valued pick** `af=a·n` (Euler `½(u_C+u_D)·n`, sonic floor); `q*=(af·(q_C-q_D)<0?ξ_max:ξ_min)`; `q^L=q^R=q*`. (P4, 6b)
9. **Vertex-MLP cap** `q^{L,capped}=q̄_C+α_C(q^L-q̄_C)`; mirror `q^R`; if shared value outside intersection of LMP boxes → fall to pattern a. (P5)
10. **[Euler]** Recompose → `WL,WR`; **positivity floor** `ρ_face≥floorf·ρ_cell, p_face≥floorf·p_cell`. (P6a,c)
11. **Flux once per face** `F_f=hllc_euler2d(WL,WR,n)` (scalar: `af·(af≥0?q^L:q^R)`). (P6d)

**Assemble & advance:**
12. Telescoping FV update `U_i += -(dt/|V_i|)Σ_{f∈i} orient(i,f) F_f A_f`. (P6d)
13. SSP-RK2, `dt=cfl·h_min/max_face(|u·n|+c)`, `cfl≤0.4`.

Steps 5 (NVD), 1 (IDW), 9 (vertex-MLP) are the three genuine additions; each is provably inert on Cartesian (§7), so the algorithm reduces exactly to the structured ASVL there — the use-as-is guarantee.

---

The complete methodology is saved at `C:\Users\user\AppData\Local\Temp\claude\--wsl-localhost-ubuntu-home-younglin90-work-claude-code-claudeCFD-solver-tmlpu\2824fc82-7197-40d6-90ec-d8b0b4279cb6\scratchpad\asvl_methodology.md` (also reproduced verbatim above). It synthesizes all six pillars into: (0) the guiding principle (port the case-table's minimal input set, not the algorithm); (1–6) the six pillars in assembly order with formulas and per-pillar Cartesian reductions; (7) the chained Master Reduction Theorem proving byte-exact reduction with the vertex-MLP cap non-binding on smooth aligned data; (8) the reviewer-defense table naming the Cartesian-implicit property each of NVD/IDW/vertex-MLP restores; (9) the step-by-step per-face algorithm.---

## 10. Adversarial review — 3 hostile JCP reviewers, all verdict = BROKEN (code-grounded)

The synthesized methodology (§§0–9) is the intended TARGET design. Three adversarial reviewer agents (reduction-skeptic, novelty-integrity, double-counting) attacked it against the SHIPPING CODE (`reconstruct_bvd.hpp`, `reconstruct2d.hpp`, `metrics.csv`). All three returned **BROKEN**, each grounded in code, not theory. Their findings and the honest corrected claims:

### R1 — "byte-exact reduction on Cartesian" is FALSE → downgrade to O(h²) consistency
- **CODE FACT:** the gradient uses the **vertex 1-ring** stencil (`build_recon_ctx`, `reconstruct2d.hpp:41-49`) = 8 neighbours on a Cartesian grid **including the 4 diagonals**, NOT the axis-aligned central difference.
- **Verified numerically:** on a curved field `q = x + x·y² + 0.5·y³`, the LSQ face-projection `gx = 1.667` vs the structured central difference `= 1.0`. They are equal **only on globally-linear fields** — precisely where limiting is irrelevant.
- **Consequence:** `grad_C·d ≠ (q_{i+1}−q_{i-1})/2` on curved data ⇒ `q_U ≠ q_{i-1}`, `phiT_C ≠` structured normalized variable, `r_L ≠` Eq.11 — on exactly the curved/extremum data the sensor must discriminate. The Master Reduction Theorem's first link (`e_grad = 0`) is false off locally-linear regions.
- **Honest claim:** the port is **2nd-order CONSISTENT with (converges to)** the structured scheme on Cartesian; the diagonal contamination is O(h²) in the face value, vanishing under refinement. NOT "byte-exact."
- **Literal-reduction fix (if byte-exact is wanted):** build the ASVL sensor's gradient span from **axis-aligned face (edge) neighbours** so `span = q_{i+1}−q_{i-1}` exactly on Cartesian.

### R2 — the zero-BV single-valued novelty is BROKEN by the default per-side cap
- **CODE FACT:** default emit path (`:1959`) is `WL = capv(qi, q*, dpL); WR = capv(qj, q*, dpR)` — the vertex-MLP cap is applied **per side with DIFFERENT reach boxes** (`dpL` from cell i's gradient/LMP/skew, `dpR` from cell j's). On any skewed/non-orthogonal face where either clamp binds, `capv(qi,q*,dpL) ≠ capv(qj,q*,dpR)` ⇒ **q^L ≠ q^R**. The shared value is fictitious exactly at the compressive/forming-extremum faces where ASVL is supposed to act.
- The author already knew: comment `:1947` "Default: independent per-side capv (breaks q^L=q^R at clamped cells)". The fix (`BVD_ABVD_LMPFIRST` = clamp the shared value to the **intersection** of both LMP boxes before emit) exists but is **opt-in, OFF by default**.
- **Honest thesis:** "zero-BV holds on the SUBSET of faces where both cells' monotone reach boxes overlap the pick; on the empty-intersection subset the scheme falls to bounded one-sided TVD with BV>0." **Quantify that face fraction on every benchmark.**
- **Fix:** make LMPFIRST the only path (intersection clamp before emit; empty intersection → honest one-sided fall + report the fraction). Also use a skew-consistent upwind indicator `sign(a·d · dp)` for the Eq.26 pick (the current `sign(a·n)` goes downwind-biased when `n` and `d` diverge under skew — flagged at code `:1893`).

### R3 — the most structural: NVD REPLACES the case-table (does not recondition it) + triple-limiting
- **FATAL:** in code the NVD path is `if(NVD){ … continue; }` — when `BVD_ABVD_NVD` is set, the code **never computes** `r_L/r_R`, `sbL/sbR`, `A`, `B`, `C`, or patterns a–e. **The discrete case-table (the paper's actual novelty) is DEAD CODE under NVD.** NVD is a *separate* central/superbee-interval scheme that **replaces** the table. So Pillar 2 (NVD reconditions inputs) and Pillar 4 (case-table runs verbatim) are **mutually-exclusive branches**, not a stack; the Master Reduction Theorem chains P2→P4 as if both fire — they don't. The "byte-exact reduction to the structured ASVL" is vacuous for the NVD variant because the structured case-table path is never reached.
- **Empirical triple-limiting (`metrics.csv`):** plain NVD gives L1 = 1.94e-2 / 1.88e-2 / 2.87e-2 — **worse** than the advertised 1.35e-2. The headline is only reached by stacking **COMP** (`smoothstep(0.8,1.0,psi)`) **AND SGATE_C** (`smoothstep(0.8,1.0,psi)`) **AND capv-LMP** (box on `psi·grad·d`) — three independent throttles **all keyed on the same `psi_LMP` signal**. This is why unstructured stalls at 0.46–0.7× while structured hit 0.46× with ONE limiter: structured reads smoothness once (`r`); the port reads it 3–4×.
- **IDW is masking, not fixing:** NVD → NVD+IDW moves **only the 3rd** LeVeque shape (2.87→2.65e-2) and leaves shapes 1 & 2 **byte-identical** (1.9364e-2, 1.8765e-2). A genuine gradient-accuracy fix would move all three (all have smooth extrema). Moving exactly one stencil = the fingerprint of re-weighting away from one bad skewed-neighbour configuration, i.e. patching a downstream over-clip, not curing the O(h)/dp sensor noise attributed to it in Pillar 1.

### The decision this forces (pick ONE architecture)

- **(A) True discrete case-table port (the faithful port the paper claims).** Feed the case-table **NVD-normalized** `r` (rebuild `r_L = phiT/(1−phiT)` from the NVD donor, then run the SAME `sbL/sbR/A/B/C/patterns a–e/Eq.26`), and use **LMPFIRST intersection cap** so `q^L=q^R` survives. Then "verbatim case-table" + the reduction become *true*, and the single-valued novelty is genuinely preserved (on box-overlap faces). Collapse COMP/SGATE_C into the single vertex-MLP cap to kill the triple-limiting.
- **(B) Admit NVD is a NEW central-interval BVD scheme** (not Majima's case-table). Drop "ports verbatim" / "byte-exact reduction to structured ASVL"; prove reduction only to the normalized-variable scheme it actually implements; still collapse the redundant gates.

**Recommendation: (A).** It is the only version whose claims honestly match the paper's thesis (a faithful unstructured port of the structured single-valued case-table). The current code is secretly (B) wearing (A)'s narrative.

### Net honest status
- The **structured** ASVL is validated and correct (0.46× MUSCL, algorithm confirmed).
- The **methodology's target design (§§0–9) is sound as a design**, but three claims must be corrected before publication: (i) reduction is **O(h²) consistency**, not byte-exact, unless the sensor gradient is rebuilt from axis-aligned neighbours; (ii) the zero-BV novelty needs **LMPFIRST intersection capping** to be real (and must be reported as a face-fraction); (iii) the code must **route the discrete case-table through NVD-normalized r** (architecture A) instead of the current NVD-replaces-table branch, and **collapse the triple gates** to the single vertex-MLP cap.
