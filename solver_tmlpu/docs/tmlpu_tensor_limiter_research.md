# T-MLP-u Tensor (Directional) Limiter — Literature Research + Design Foundation

Research date: 2026-06-18. Source: deep-research harness (22 sources fetched, 97 claims,
25 verified). **CAVEAT: a provider rate-limit storm killed ~20 of 25 verification votes as
"0-0 abstain" = could NOT verify (API error), NOT genuinely refuted.** Only the 4 findings
below are verified (3-0 / 2-0 unanimous). The "unverified leads" are high-value and must be
re-verified before grounding the design on them.

## Goal recap
Advance the T-MLP-u reconstruction (per-vertex MLP-u bounded LSQ-linear + Ducros gate
s=ω²/(ω²+θ²)) so the **shear-tangential** gradient component survives (KH/slip-line roll-ups)
while the **shock-normal** component stays monotone + positive. **Flux (HLLC) is FROZEN** —
only the reconstruction/limiter may change.

## VERIFIED findings (high confidence)

1. **May & Berger (2013), SIAM J. Sci. Comput. — component-wise LP limiter.** Limits the x and
   y gradient COMPONENTS separately via a tiny per-cell linear program, "as opposed to a scalar
   limiter which limits all components simultaneously with one scalar." Inherently 2D,
   linearity-preserving, works on non-coordinate-aligned/unstructured meshes.
   **CRITICAL CAVEAT: it separates along MESH axes (x,y), NOT along flow-feature directions
   (shear-tangential vs shock-normal).** So it is direction-dependent but NOT feature-adaptive.
   DOI 10.1137/120875624; pdf cs.nyu.edu/~berger/lpLimiter_revised.pdf.

2. **MLP / MLP-u is confirmed SCALAR.** A single φ∈[0,1] per cell from a vertex maximum
   principle (vertex-interpolated value ∈ [min,max] neighbour cell-averages). The
   "multi-dimensional" label = the vertex STENCIL, not directional limiting of the gradient.
   T-MLP-u's baseline limiter has NO anisotropy mechanism — it must be EXTENDED.
   (Park & Kim, Comput. Fluids 2012; Park-Yoon-Kim JCP 2010.)

3. **Directional limiting of a LINEAR reconstruction is fundamentally a 2nd-order technique.**
   MLP-style monotonicity = a maximum-principle (L∞/LED) argument on the scalar-limited linear
   distribution. It cannot raise the convergence rate. → A tensor limiter REDUCES fixed-mesh
   DIFFUSION, it does NOT give "true higher order." (Consistent with [[high-order-recon-findings]]:
   true 3rd order needs WENO/CWENO.) (Answers research Q6.)

4. **DRSC = MUSCL-THINC-BVD (Xiao group) is the competing SOTA for slip-line sharpness.** It
   reaches low dissipation by BVD candidate SELECTION between a MUSCL-MLP polynomial candidate
   and a multidimensional THINC sigmoid candidate — NOT by directional limiting. → BVD-selection
   and directional-limiting are SEPARATE, possibly COMPLEMENTARY levers. (arXiv 2003.09223,
   2402.03002.)

## UNVERIFIED but HIGH-VALUE leads (rate-limited — re-verify before relying)

- **Kuzmin (2021), arXiv 2008.11981 — anisotropic per-derivative DG-P1 limiter.** Limits EACH
  directional derivative with its OWN correction factor α_ik (one per space direction k), via an
  explicit minmod formula against per-direction bounds U_ik^max/min built from a divergence-theorem
  reconstruction of neighbour derivatives. **This is the closest concrete directional-limiter
  formula in existence.** Key design insight it implies (see §Design): bound each direction by ITS
  OWN data, don't "leave one free."
- **ILR — Implicit/bound-constrained QP limiter, arXiv 1703.01055.** Optimizes over the FULL
  2-component gradient via a bound-constrained QP, letting the gradient ROTATE within the feasible
  set; Barth-Jespersen = the scalar special case. **Reported Double-Mach result: ILR gives a more
  stable Mach stem but Barth resolves the triple-point shear (KH) layer MORE sharply — i.e. a
  bounded multidimensional limiter TRADES shear-rollup sharpness for stability.** ← directly the
  tension we fight; must verify, it may already characterize our tradeoff.
- **Improved-Ducros (AIAA 2018-3710) + WADS (arXiv 2305.08369).** Original Ducros over-fires on
  divergence in low-vorticity regions (false dissipation); improved versions + a SEPARATE contact
  criterion fix shock-crossing-shear and contact false-positives. Relevant to hardening our gate.
- **CoDeS (arXiv 2605.21444) — IGNORE.** Tensor anisotropic regularization along compression
  eigenvectors would be perfectly on-point, but the arXiv id is anomalous (future-dated) and likely
  hallucinated. Do NOT cite without confirming the paper exists.

## Verdict on the 5 design questions
1. **Precedent / novelty?** Component-wise limiting EXISTS (May-Berger mesh-axes; Kuzmin
   per-derivative) but **feature-ALIGNED directional limiting for KH preservation under a fixed
   flux is unattested** → T-MLP-u's feature-aligned construction is *plausibly novel* (absence of
   evidence, not proof; Kuzmin/ILR are close — verify).
2. **Feature-direction extraction?** NOT answered by verified evidence (strain-eigenvector / CoDeS
   claims all rate-limited). Open. Candidates: strain-rate tensor eigenvectors, vorticity gradient,
   the existing Ducros s. Shock∩shear ambiguity is a known failure mode (improved-Ducros/WADS).
3. **Anisotropic extensions of BJ/Venk/MLP?** None standard; all canonically scalar. Kuzmin/ILR are
   the only directional generalizations found.
4. **Provable monotonicity+positivity with tangential UNLIMITED?** **NOT solved in literature.** The
   max-principle proof assumes ONE scalar φ bounding the whole vertex value; freeing a component
   breaks it. **#1 open risk.** (See §Design for the per-direction-own-bound fix.)
5. **Order ceiling?** 2nd-order (finding 3). Diffusion-reduction tool, not a higher-order scheme.

## DESIGN UPGRADE extracted from the research (key)
My original sketch ("decompose ∇W into normal/tangential, leave tangential at φ=1") is **NOT
provable** — it breaks the max principle (Q4). The Kuzmin-style fix makes it provable AND
naturally anisotropic:

> **Do NOT "leave the tangential component free." Instead bound EACH direction against ITS OWN
> data-driven limit.** Decompose the increment along feature axes (n̂ = shock-normal / compression,
> t̂ = shear-tangential). Apply a BJ/MLP-type max-principle bound to the n̂-projection using the
> n̂-direction neighbour range, and SEPARATELY to the t̂-projection using the t̂-direction range.
> In smooth shear the t̂-range is wide → α_t≈1 (no diffusion, KH survives) *by the data*, not by
> fiat; across the jump the n̂-range is tight → α_n small (monotone, positive). This keeps a
> provable per-direction maximum principle (each component ∈ its own [min,max]) → positivity of
> ρ,p follows from bounding the conservative/primitive increment per direction.

This converts the unprovable "free tangential" into a provable "anisotropic by construction."
n̂,t̂ come from the strain-rate eigenvectors (or the Ducros velocity gradients already computed).

## Honest risk register
- R1 (HARD): provability — the per-direction-own-bound design (above) is the candidate cure;
  must derive the actual bound + a positivity proof. No off-the-shelf proof to inherit.
- R2: feature-direction robustness at shock∩shear (improved-Ducros/WADS needed).
- R3: only 2nd-order — gains are fixed-mesh diffusion, must convergence-test (≥3 resolutions) to
  avoid the metric-noise trap [[high-order-recon-findings]].
- R4: competes with / may be complementary to BVD-selection (DRSC) — decide whether to combine.
