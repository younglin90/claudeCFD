# Double Mach Reflection Strict Validation Spec

Updated: 2026-06-11

## Purpose

This project uses this spec for future Double Mach reflection validation runs.
The comparison target is simple:

`TMLP-u` must produce a better density-line-contour result than `MLP-u1`
under identical numerical conditions except the reconstruction scheme.

The desired result is a reference-style Double Mach reflection solution where
the lower-right shock / slip-line interaction region contains many coherent
vortices of different sizes.

## Physical Setup

- Case: Double Mach reflection.
- Domain: `x in [0, 4.0]`, `y in [0, 1.0]`.
- Gas model: Euler equations, `gamma = 1.4`.
- Final time: `t = 0.2`.
- Initial incident shock line: `x = 1/6 + y / sqrt(3)`.
- Pre-shock state: `(rho, u, v, p) = (1.4, 0.0, 0.0, 1.0)`.
- Post-shock state:
  - `rho = 8.0`
  - `u = 8.25*cos(pi/6)`
  - `v = -8.25*sin(pi/6)`
  - `p = 116.5`

## Boundary Conditions

- Left boundary: post-shock Dirichlet.
- Right boundary: transmissive outflow.
- Bottom boundary:
  - `x <= 1/6`: post-shock Dirichlet.
  - `x > 1/6`: reflective wall.
- Top boundary: exact moving-shock Dirichlet state.

## Mesh And Numerical Contract

- Baseline mesh: unstructured alternating triangular mesh.
- Current quick verification grid: `480 x 120`.
- Current final verification grid: `960 x 240`.
- Flux: `roe_rotated_hybrid`.
- Time integrator: `forward_euler`.
- CFL: `0.35`.
- TMLP-u and MLP-u1 must use the same mesh, flux, CFL, time integrator, boundary conditions, final time, contour levels, plot crop, and post-processing.
- No ROI-local scheme switching is allowed. The same reconstruction and flux choices must be applied over the full domain.

All non-reconstruction numerical choices are fixed to this current best Double
Mach reflection validation setup.  Future autoresearch iterations should keep
the mesh family, flux, CFL, time integration, boundary treatment, contour
levels, plot crop, and post-processing unchanged, varying only the high-order
reconstruction candidate unless the user explicitly opens a new
non-reconstruction study.

## Primary Visual ROI

The strict visual gate focuses on the lower-right complex shock interaction:

- Main ROI: `x in [2.1, 2.85]`, `y in [0.0, 0.6]`.
- Slip-line vortex-chain sub-ROI: `x in [2.25, 2.65]`, `y in [0.03, 0.32]`.
- Lower-right vortex-packet sub-ROI: `x in [2.55, 2.80]`, `y in [0.02, 0.20]`.

## PASS Criteria

TMLP-u passes only if all gates below pass.

### Physical Vortex-Shape Gate

Do not count raw `|vorticity|` blobs as vortices. A counted vortex must satisfy:

- coherent rotational core with `Q > 0`;
- coherent rotational core with `lambda_ci > 0`;
- connected support over several cells, not a one-cell speckle;
- density line contours visibly wind, hook, or wrap around the core;
- contour winding is locally coherent, not shock-line sawtooth noise.

Minimum target:

- at least three coherent vortices along the slip-line vortex-chain sub-ROI;
- at least one larger coherent lower-right vortex packet;
- at least two visibly different vortex size classes in the main ROI.

### Paper-Grade 1.5x Better-Than-MLP-u1 Gate

TMLP-u must be better than MLP-u1 in the main ROI under identical mesh,
flux, CFL, integrator, boundary conditions, final time, contour levels, plot
crop, and post-processing.  This is the decisive Double Mach reflection gate
for paper-grade validation; absolute visual gates alone are diagnostics and
are not sufficient for PASS.

- TMLP-u coherent vortex-shape count in the main ROI must be strictly greater
  than MLP-u1.
- TMLP-u ROI vortex clarity score must be at least `1.5 * MLP-u1`.  The score
  is based on coherent vortex count, multi-scale cluster support, compactness,
  and ROI vorticity strength.  This replaces a purely downstream-extent
  criterion.
- TMLP-u ROI vortex core strength score must be at least `1.5 * MLP-u1`.  This
  score is a support-weighted, compact rotational-core strength proxy based on
  coherent high-vorticity clusters in the ROI.
- TMLP-u ROI vortex separation score must be at least `1.2 * MLP-u1`.  This
  score rewards visibly separated vortex cores and penalizes one merged
  shear-sheet-like structure.
- TMLP-u must preserve more visibly coherent multi-scale roll-up structure near
  the lower-right shock / slip-line interaction.
- TMLP-u must not gain apparent vortices by creating checkerboard,
  carbuncle-like spots, isolated speckles, or shock-line sawtooth artifacts.

The automatic verifier records this as:

```text
double_mach_better_than_mlp_u1_pass = 1
```

The comparison is decomposed into three required sub-gates:

- `double_mach_vortex_better_than_mlp_u1_pass = 1`
- `double_mach_roi_vortex_clarity_better_than_mlp_u1_pass = 1`
- `double_mach_roi_vortex_core_strength_better_than_mlp_u1_pass = 1`
- `double_mach_roi_vortex_separation_better_than_mlp_u1_pass = 1`
- `double_mach_visual_better_than_mlp_u1_pass = 1`
- `double_mach_shock_integrity_better_than_mlp_u1_pass = 1`

The current mechanical proxy requires strictly more coherent ROI vortices than
MLP-u1, at least 1.5x ROI vortex clarity, at least 1.5x ROI vortex core
strength, at least 1.2x ROI vortex separation, no worse diffusion/shock
quality, no new major artifact, and positive density and pressure.

The old global pressure-jump checker is retained as a diagnostic only.  It is
not a primary pass/fail gate because it also increases when a method produces a
legitimately sharper shock or slip line.  The primary artifact check instead
uses:

```text
double_mach_smooth_region_checker
    <= double_mach_mlp_u1_smooth_region_checker
```

`double_mach_smooth_region_checker` is the 95th percentile local pressure
roughness in smooth/background cells after excluding strong shock, slip-line,
and strong-vorticity regions.  This separates physical sharpness from
nonphysical odd-even/checkerboard contamination.

### Shock Integrity Gate

The result must still be a valid Double Mach reflection solution:

- incident shock and reflected shock remain thin and continuous;
- Mach stem / triple-point structure remains physically plausible;
- wall jet and slip line remain connected to the shock interaction;
- no gross shock splitting, odd-even decoupling, or pressure/density negativity.

Shock integrity must be non-worse than MLP-u1.  A candidate with more apparent
vortices fails if it gets them by degrading shock quality, increasing
smooth-region checkerboard/carbuncle indicators, or introducing isolated
artifacts.

### Artifact Rejection

FAIL if the apparent vortices are mainly:

- disconnected contour fragments;
- one-cell or two-cell spots;
- grid-aligned sawtooth ripples;
- carbuncle-like blobs on a shock;
- extra contour complexity caused by nonphysical oscillation.

The global pressure-jump checker may be reported in validation tables, but a
candidate does not fail solely because this diagnostic is larger than MLP-u1 if
the smooth-region checker gate, shock-integrity gate, positivity gate, and ROI
clarity gate all pass.

## Future Use

Before future Double Mach reflection autoresearch runs, use this spec as the
default validation target. Automatic metrics should mirror the Mach 3 step
vortex-shape logic: `Q > 0`, `lambda_ci > 0`, and density-contour winding must
agree before a vortex is counted.
