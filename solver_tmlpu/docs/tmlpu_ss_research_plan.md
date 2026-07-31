# T-MLP-u-SS : Shear-Sensor-Gated Anti-Diffusive BVD (paper-track)

**Goal.** Resolve Double-Mach slip-line Kelvin–Helmholtz roll-ups at a *coarse* mesh by
reducing the reconstruction's numerical diffusion on the shear layer — NOT by mesh
refinement (last resort). Target: high-novelty top-tier journal.

## Novelty gap
Existing BVD / THINC anti-diffusion targets *material interfaces* = normal density jumps.
The Double-Mach slip line is a **shear (Kelvin–Helmholtz) layer**: small density jump,
large tangential-velocity jump. No established reconstruction selectively anti-diffuses the
*shear* layer while keeping shocks robust. That is the contribution.

## Method (incremental, evidence-gated — keep only what measurably helps)
1. **Multi-variable TBV** (DONE, `BVD_MULTIVAR=1`). BVD candidate selection scores the
   Total Boundary Variation summed over all primitive vars normalised by each var's range,
   so the velocity-shear of the slip line (not just density) triggers the sharp candidate.
2. **Ducros shear sensor** (DONE). Per cell s = ω²/(ω²+θ²+ε), θ=∇·u (dilatation, big at
   shocks), ω=∂v/∂x−∂u/∂y (vorticity, big on shear). s→1 shear/vortex, s→0 shock.
3. **Sensor-gated limiter relaxation** (DONE, `SHEAR_RELAX=k`). The sharp (order-2 quadratic)
   candidate's MLP vertex bound is relaxed by k·s·(local range) — less clipping on the shear
   layer (lower diffusion → KH grows) while shocks (s≈0) keep the tight positivity bound.
4. **NEXT candidates if 1-3 insufficient:**
   - THINC sharp candidate for the contact (BVD-THINC) — normal-direction tanh steepening.
   - Tangential anti-diffusion (Després–Lagoutière overcompression) along the shear.
   - Vorticity-confinement source term (Steinhoff) gated by s — explicit KH preservation.

## Validation gates (must hold)
- LeVeque / Mach3 / Sod / contact: unchanged or better (no regression; defaults off via env).
- DM positivity (rho>0, p>0) preserved at Mach 10.
- DM slip-line: more coherent KH roll-ups vs baseline BVD at the SAME mesh (the headline).
- Quantify: vorticity-extrema count along the slip line + primary-vortex coherence (aspect).

## Status
1-3 implemented + unit-tested (defaults bit-identical, 11/11). Diffusion sweep on DM
(baseline BVD vs +multivar vs +shear-relax k∈{0.5,1} vs order-2) pending core availability
(Mach3 graded run finishing). See ledger results/T-MLP-u/goal_iteration_log.md.
