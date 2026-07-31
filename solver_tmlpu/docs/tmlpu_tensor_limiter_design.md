# T-MLP-u-D : Feature-Aligned Directional Box Limiter — Design

Date 2026-06-18. Builds on the VERIFIED foundations (see [[tmlpu_tensor_limiter_research.md]]):
- **ILR** (Chen-Hu-Li, CiCP 2018, arXiv 1703.01055): provable local maximum principle via an
  edge-midpoint box `m_j ≤ û(z_j) ≤ M_j` (Eq 3.3, Thm 4.1); Barth–Jespersen = the scalar special
  case `L=−αG⁻¹c` (Rem 3.2); and the VERBATIM Double-Mach tradeoff (§5.3, HLLC): tighter
  multidimensional box ⇒ more stable Mach stem but **Barth resolves the triple-point shear/KH layer
  more sharply**. ⇒ relaxing the box along the shear tangent is the lever to recover roll-up.
- **Kuzmin** (CMAME 2020, arXiv 2008.11981): per-direction limiting (limit each derivative vs its
  own bound, Eq 57/61), but the anisotropic slope part ALONE is NOT bound-preserving (Rem 7) — DMP
  must be carried elsewhere. ⇒ we must supply our own positivity guarantee.

Flux (HLLC) is FROZEN [[flux-fixed-reconstruction-only-rule]]. This is a RECONSTRUCTION-only design.

## Core idea (one sentence)
Replace the scalar MLP-u factor φ by a **two-direction box limiter aligned to the local flow
feature**: keep the full monotone box NORMAL to the feature (shock capturing, positivity), and
RELAX the box along the shear-TANGENTIAL direction — but only where the Ducros sensor says
"rotation/shear, not shock" — keeping a hard positivity floor. The relaxed tangential box lets the
small over/undershoot that *seeds* the Kelvin–Helmholtz roll-up survive (the exact thing ILR's tight
box suppresses), while never producing negative ρ or p.

## Notation
Per cell i: centroid x_i, LSQ gradient g = ∇W (per variable). Face f at x_f, d = x_f − x_i.
Standard MLP-u face value: W_f = W_i + φ (g·d), φ∈[0,1] scalar (one per cell). We split φ into a
2-direction operator.

## Step A — feature frame (n̂, t̂), per cell  [our novelty vs Kuzmin's mesh axes]
Use the velocity Jacobian J = ∇u (2×2), J_ab = ∂u_a/∂x_b, already available from the LSQ gradients
of (u,v). The layer-NORMAL is the direction of strongest velocity variation = the right singular
vector of J with the largest singular value (equivalently the top eigenvector of JᵀJ):
    JᵀJ = [[ux²+vx², ux·uy+vx·vy],[·, uy²+vy²]] ,  n̂ = top eigenvector,  t̂ = n̂⊥.
- Pure shear u=(U(y),0): JᵀJ top eigenvector = ŷ = across the layer ✓ (t̂ = along the slip line).
- Shock (compression): J dominated by dilatation along the shock normal ⇒ n̂ = shock normal ✓.
So the SAME definition gives the right normal for both shock and shear; the GATE distinguishes them.
Ducros sensor (already computed): s = ω²/(ω²+θ²+ε), ω=v_x−u_y, θ=u_x+v_y. s≈1 shear/vortex, s≈0 shock.
Degeneracy guard: if the top/bottom eigenvalues of JᵀJ are within tol (isotropic, no feature) →
fall back to scalar MLP-u (s·anisotropy → 0).

## Step B — directional box limiting
Decompose the gradient in the feature frame: g = g_n n̂ + g_t t̂,  g_n = g·n̂,  g_t = g·t̂.
Project the cell's MLP-u admissible range to each direction. Let [W_min, W_max] be the usual
vertex/neighbour bounds (as in MLP-u). Define per-direction limited factors:

  NORMAL  (always tight — monotone + positive):
    φ_n = BJ/MLP factor enforcing  W_min ≤ W_i + φ_n g_n (n̂·d_v) ≤ W_max  over face-vertices v
          (exactly the existing scalar MLP-u bound, but applied to the n̂-projected increment only).

  TANGENTIAL (relaxed by the sensor):
    φ_t^tight = same MLP bound applied to the t̂-projected increment.
    φ_t       = (1 − s)·φ_t^tight  +  s·φ_t^relax ,     φ_t^relax ∈ [φ_t^tight, 1]   (knob κ: φ_t^relax = φ_t^tight + κ(1−φ_t^tight))
  i.e. at a shock (s≈0) φ_t = φ_t^tight (full monotone box, identical to MLP-u); in shear (s≈1)
  φ_t → 1 (box opened along the slip line → roll-up seed survives).

Face value (pre-positivity):
    W_f* = W_i + φ_n g_n (n̂·d) + φ_t g_t (t̂·d).

## Step C — positivity floor (the guarantee Kuzmin says we must add ourselves)
The relaxed tangential box can overshoot the LMP range (intended), so the LMP positivity argument
no longer covers it. Restore positivity with a single Zhang–Shu-type scaling θ∈[0,1] of the
*tangential* increment only (normal increment is already in-bounds):
    let A = W_i + φ_n g_n (n̂·d)            (in [W_min,W_max], so ρ_A>0, p_A>0 for ρ,p vars)
        B = φ_t g_t (t̂·d)                  (the relaxed tangential increment)
    choose θ = max θ'∈[0,1] s.t. ρ(A+θ'B) ≥ ε_ρ ρ_i  AND  p(A+θ'B) ≥ ε_p p_i   (ε ~ 0.1–0.2)
    W_f = A + θ B.
For the primitive scalars ρ and p this is a scalar linear inequality in θ' → closed-form θ (no
iteration). u,v carry no positivity constraint (θ=1). Result: **ρ_f>0, p_f>0 provably**, for ANY
relaxation, because θ pulls back to the in-bounds state A when needed.

### What is provably preserved vs relaxed (honest)
- POSITIVITY ρ,p>0: PROVABLE (Step C, convex pullback to A). Holds for the FIXED HLLC too — a
  consistent Riemann solver fed positive L/R states with the contact between them stays positive.
- STRICT MONOTONICITY (no new extrema) along t̂: INTENTIONALLY RELAXED, and ONLY where s≈1 (no
  shock) and ONLY along the tangent. This is the ILR-confirmed lever: the suppressed over/undershoot
  is the physical KH seed, not a spurious shock oscillation. Normal direction keeps full monotonicity
  everywhere, so shocks remain monotone + sharp.
- Reduces to baseline EXACTLY when κ=0 or s=0 (φ_t=φ_t^tight, θ=1) → bit-identical to MLP-u; and
  when JᵀJ isotropic → scalar fallback. So default-off is trivially safe (regression-clean).

## Order of accuracy
Per BOTH verified papers, directional limiting of a LINEAR reconstruction is a 2nd-order technique;
it does NOT raise the convergence rate. The win is **lower fixed-mesh diffusion at the shear**, which
MUST be convergence-tested (≥3 resolutions on enstrophy/L1) to avoid the metric-noise trap
[[high-order-recon-findings]]. Do not claim higher order.

## Novelty (honest)
Feature-ALIGNED (eigenvector-frame) directional box-relaxation, sensor-gated to shear, for KH
preservation under a fixed flux, is UNATTESTED in the verified literature: Kuzmin limits mesh axes
and is not bound-preserving alone; ILR limits the whole gradient isotropically and explicitly leaves
shear sharpness on the table. Combining (i) ILR's provable box, (ii) Kuzmin's per-direction split,
(iii) an eigen-frame feature alignment, (iv) Ducros gating, and (v) a tangential positivity floor —
to deliberately recover the Barth-like roll-up ILR loses — appears to be new. (Absence of evidence,
not proof; the design stands on its own merits regardless.)

## Risks
- R1 feature-frame robustness at shock∩shear coincidence (s and n̂ both ambiguous). Mitigation:
  conservative s (bias toward shock), eigenvalue-ratio degeneracy fallback, κ small.
- R2 the relaxed tangential overshoot, even positivity-clamped, may roughen contacts on Mach3
  (global_artifact gate). Mitigation: κ sweep; relax only when |g_t|≫|g_n| (genuine shear) not at
  oblique contacts.
- R3 2nd-order only — gains are diffusion not order; convergence-test before claiming.
- R4 cost: one 2×2 eigfor JᵀJ per cell (cheap, closed-form for 2×2) + the split → ~cost-neutral,
  same single reconstruction. No second candidate (unlike BVD).

## Validation plan (cheap → expensive, gated)
1. Unit: κ=0 / s=0 / isotropic → bit-identical to MLP-u (regression).
2. vortex_bench (2 s): does φ_t relaxation lower L2 / raise omega vs MLP-u AND T-MLP-u-S? Gate: ≥ T-MLP-u-S, positivity OK.
3. KH/shear micro-bench: slip-line roll-up count vs MLP-u.
4. Convergence (≥3 N): enstrophy/L1 — is the gain convergent or metric noise?
5. Mach3 + DM (sparingly): 3-benchmark gate vs MLP-u and vs T-MLP-u-S; positivity at Mach10.

## Env knobs (default OFF → baseline unchanged)
TMLPU_DIR_LIMIT=1 (enable), TMLPU_DIR_KAPPA=κ (tangential relax, 0..1), TMLPU_DIR_FLOOR=ε
(positivity floor), TMLPU_DIR_ANISO_TOL (eigenvalue-ratio degeneracy fallback).
