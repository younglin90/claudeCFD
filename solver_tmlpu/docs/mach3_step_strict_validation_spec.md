# Mach 3 Forward-Facing Step Strict Validation Spec

Updated: 2026-06-05

## Purpose

This project uses this spec for Mach 3 forward-facing step validation.
The target is a physically coherent `t = 4.0` result with:

- no carbuncle;
- physically plausible flag-waving instability;
- upper-region roll-up vortex structure;
- a clean, unsplit reflected shock line above the step.

The comparison target remains:

`TMLP-u` must be evaluated consistently over the full domain and, when compared
against `MLP-u1`, all numerical settings except reconstruction must be held
identical.

## Physical Setup

- Case: 2D Mach 3 wind tunnel with a forward-facing step.
- Domain: `x in [0, 3.0]`, `y in [0, 1.0]`.
- Step location: `step_x = 0.6`.
- Step height: `step_h = 0.2`.
- Final time: `t = 4.0`.
- Gas model: Euler equations, `gamma = 1.4`.

Initial state:

- `rho = 1.4`
- `p = 1.0`
- `c = sqrt(gamma * p / rho)`
- `u = 3.0 * c`
- `v = 0.0`

## Boundary Conditions

- Left boundary: Dirichlet inflow with state `(rho, u, v, p)`.
- Right boundary: transmissive outflow.
- Solid walls, including the step wall: reflective.

## Mesh Contract

- Quick (non-strict) Mach 3 mesh: `200 x 80` on standard
  `triangulate_box_roi_graded`.
- Paper/final (strict/default) Mach 3 mesh: `480 x 160` on a full-domain
  non-ROI uniform triangular mesh (`tri_alternating` behavior).
- In strict/default autoresearch Mach 3 probes, paper/final mode must use
  `tri_alternating` on the `480 x 160` logical grid.
- Smaller quick grids, including `120 x 40`, are allowed only when an explicit
  quick/smoke option is requested. Such runs are non-strict and must not be
  reported as the default Mach 3 validation result.
- `tools/autoresearch/three_benchmark_probe.py` must not silently fall back to
  the historical `120 x 40`, `220 x 80`, or `280 x 100` quick grids for
  Mach 3 default quick runs.
- Upper validation ROI densification:
  - `x in [0.5, 3.0]`
  - `y in [0.6, 1.0]`
- Current piecewise mesh bands:
  - x bands: `[0.0, 0.5]`, `[0.5, 3.0]`
  - y bands: `[0.0, 0.18]`, `[0.18, 0.2]`, `[0.2, 0.6]`, `[0.6, 1.0]`

The mesh may be graded, but the numerical scheme must not switch between ROI
and non-ROI regions. ROI grading changes resolution only.

## Numerical Contract

- Flux: `roe_rotated_hybrid`.
- Time integrator: `forward_euler`.
- CFL: `0.45`.
- Face quadrature: `1`.
- Solver threads used in current production runs: `16`.
- Current TMLP-u run key: `tmlpu_mach3_step_on`.
- Baseline comparison key for MLP-u1: `mlp_u1`.

No region-local scheme switching is allowed. The same reconstruction and flux
choices must be applied over the full domain.

All non-reconstruction numerical choices are fixed to this current best Mach 3
step validation setup.  Future autoresearch iterations should keep the mesh,
flux, CFL, time integration, face quadrature, boundary treatment, and
post-processing unchanged, varying only the high-order reconstruction candidate
unless the user explicitly opens a new non-reconstruction study.

## PASS Criteria

The strict PASS target is comparative, not merely absolute.  TMLP-u and
MLP-u1 must be run under identical mesh, flux, CFL, integrator, boundary
conditions, final time, contour levels, plot crop, and post-processing.  The
TMLP-u candidate passes only if it is physically admissible and better than
MLP-u1 in the Mach 3 step validation gates below.

### Upper ROI Vortex-Shape Gate

ROI:

- `x in [0.5, 3.0]`
- `y in [0.6, 1.0]`

PASS requires a real vortex shape, not merely a high-vorticity cluster:

- coherent rotational core with `Q > 0`;
- coherent rotational core with `lambda_ci > 0`;
- connected support over multiple cells;
- density contours must wind, hook, or wrap around the rotational core;
- roll-up must be physically plausible and not grid-aligned sawtooth noise.

Comparative PASS additionally requires TMLP-u to be better than MLP-u1 in the
upper ROI.  The automatic verifier records this as:

```text
mach3_step_upper_rollup_better_than_mlp_u1_pass = 1
```

The current mechanical proxy is intentionally ROI-clarity based.  TMLP-u must
show a sharper upper-ROI vortex than MLP-u1, not merely a longer downstream
extent.  The decisive comparison uses:

- stronger or more numerous density-contour hooks around a resolved swirl core;
- larger continuous contour-hook angle around the core;
- more coherent `Q > 0` / `lambda_ci > 0` vortex-shape detections when hooks
  are tied;
- compact opposite-signed vortex-pair and density-roll-up counts only as weak
  tie-breakers;
- rejection of nearly horizontal shear-sheet wiggles.

The downstream signed-pair count, downstream density count, and pair
`x`-extent remain recorded as diagnostics, but they are not the upper ROI PASS
gate.  A candidate can pass the upper ROI gate when the vortex inside the ROI
is visibly clearer than MLP-u1 even if it does not extend farther downstream.

### Step-Top Reflected-Shock Gate

ROI:

- `x in [1.0, 1.6]`
- `y in [0.2, 0.35]`

PASS requires:

- one thin, continuous reflected shock line;
- no split into two shock branches;
- no upstream or leftward drift of one branch near the shock foot;
- pressure and density gradient ridges remain single and continuous.

Comparative PASS additionally requires:

```text
mach3_step_top_floor_shock_better_than_mlp_u1_pass = 1
```

The current mechanical proxy requires the TMLP-u step-top shock gate to pass
and the reflected-shock split score and split-band count to be no worse than
MLP-u1.

### Global Gate

The result must also satisfy:

- no carbuncle at `t = 4.0`;
- no pressure or density negativity;
- no global nonphysical isolated spots;
- flag-waving must appear as a physical instability, not a numerical artifact.

Comparative PASS additionally requires:

```text
mach3_step_global_artifact_better_than_mlp_u1_pass = 1
```

The current mechanical proxy requires positive density and pressure, the
global artifact gate to pass, and global shock-split score, nonphysical spot
count, and roughness score to be no worse than MLP-u1.

### Final Comparative Gate

The final Mach 3 step strict gate is:

```text
mach3_step_better_than_mlp_u1_pass = 1
```

This requires all of:

- `mach3_step_visual_better_than_mlp_u1_pass = 1`
- `mach3_step_upper_rollup_better_than_mlp_u1_pass = 1`
- `mach3_step_top_floor_shock_better_than_mlp_u1_pass = 1`
- `mach3_step_global_artifact_better_than_mlp_u1_pass = 1`

The older absolute fields remain recorded for diagnosis:

- `mach3_step_visual_pass`
- `mach3_step_upper_rollup_pass`
- `mach3_step_roi_vortex_clarity_score`
- `mach3_step_roi_vortex_clarity_better_than_mlp_u1_pass`
- `mach3_step_top_floor_shock_pass`
- `mach3_step_global_artifact_pass`
- `rho_min`
- `p_min`

However, these absolute fields are not sufficient by themselves.  TMLP-u must
also outperform MLP-u1 under the same validation contract.

## Artifact Expectations

Each completed Mach 3 step validation should record:

- run name;
- mesh family and logical grid;
- reconstruction key;
- flux, CFL, integrator, final time;
- metrics JSON;
- density line contour PNG;
- density contour PNG;
- pressure PNG;
- vorticity or vortex-diagnostic PNG when available;
- pass/fail state for the upper vortex gate and step-top shock gate.

Current project references:

- strict contract handoff: `.omx/specs/autoresearch-mach3-step-strict/CODEX_APP_HANDOFF.md`
- candidate runner: `tools/autoresearch/run_current_mach3_candidate.sh`
- grid runner: `tools/autoresearch/run_mach3_grid_probe.py`
- single-candidate probe: `tools/autoresearch/three_benchmark_probe.py`
- benchmark implementation: `solver/solve_T-MLP-u/tests/test_2d_tmlpu_paper_benchmarks.py`

## Future Use

Before future Mach 3 step autoresearch runs, use this file as the default
validation contract unless the user explicitly changes the benchmark.
