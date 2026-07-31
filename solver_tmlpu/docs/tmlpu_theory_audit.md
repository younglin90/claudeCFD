# T-MLP-u theory/derivation audit (2026-06-15)

No tmlp_u_theory.md exists; the code (reconstruction.py TMLPU) IS the theory. Audited the
documented face-value formula and its psi/r/bound/t* implementation against TVD/MLP-u theory
and the on-machine evidence (40+ configs).

Documented formula:
  phi_fL = phi_L + psi_L * [ t*(phi_R - phi_L) + grad_phi_fcorr . (m_f - f0) ]
  psi = max(0, min(alpha*r, alpha, psi_TVD)),  r = far-upwind ratio (Delta+ / Delta-)
  psi_L = min over face-vertex pairs psi_Vi,  per-vertex TVD bound.

## Flaw 1 (fundamental) — r cannot distinguish resolved discontinuity from smooth gradient
r = Delta+/Delta- (consecutive-gradient ratio). A top-hat slot edge is steep on BOTH sides ->
r ~ 1. Every standard TVD limiter gives psi=1 at r=1 (the "no compression" point), so it cannot
sharpen the slot. Only pure_downwind (psi==2 for all r>0) sharpens it -- but that also gives
psi=2 in smooth monotone flow -> global roughness on Euler (Mach3 global_artifact).
On-machine: pd passes LeVeque, fails Mach3 roughness; all 6 r-adaptive limiters
(van_leer/superbee/mc/koren/mod_superbee/van_albada) fail LeVeque slot (iou<1). PROVEN: a single
STATIC TVD psi(r) cannot do slot-sharp AND smooth. Resolution requires compression gated by a
normalized JUMP/feature, not the raw r-ratio.

## Flaw 2 — zero_delta_psi default 2.0 over-compresses flat regions (reconstruction.py:10960)
At Delta+ ~ 0 (flat / smooth extremum), psi is forced to zero_delta_psi=2.0 (max compression)
where there is nothing to compress -> distorts cone/hump extrema and roughens flat Euler flow.
Theory-consistent value is 1.0 (central). (Goal text already flagged 1.0 as non-compressive.)

## Flaw 3 — phi_LL clipping loosens r at strong shocks (reconstruction.py:10908-10915)
Default clips phi_LL to [phi_min-tvb_eps, phi_max+tvb_eps]. The code's own comment says
phi_LL_unclipped=True is "theory-strict ... removes the artificial loosening of r at strong
shocks so the TVD limiter naturally drops psi->0 at shock faces." Default is the non-strict clip.

## Enhancement direction (SINGLE scheme only)
Flaw 1 is decisive. The single-limiter fix is a feature-gated compressive limiter:
- CICSAM (cicsam_full): NVD Hyper-C/UQ blend by cos^2(2theta) interface alignment -> compresses
  only discontinuities aligned with the face, gentle elsewhere. Single limiter; the project's
  original LeVeque winner (cicsam_co38). Untested on the unified Euler path.
- Plus zero_delta_psi=1.0 (Flaw 2 fix) and phi_LL_unclipped=True (Flaw 3 fix) as theory-true
  global constants.

## Test plan
Expose cicsam_full / cicsam_courant / zero_delta_psi / phi_LL_unclipped on v220, validate single
TMLPU on LeVeque 100x100 (relaxed gate) then Mach3 (does feature-gated compression keep slot
sharp AND smooth flow clean?). If CICSAM-on-Euler is unstable (Courant-based), fall back to a
smoothness-modulated Sweby-beta single limiter.
