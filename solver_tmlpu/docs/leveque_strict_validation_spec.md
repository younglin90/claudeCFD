# LeVeque Rotation Strict Validation Spec

Updated: 2026-06-05

## Purpose

This project uses this spec for the LeVeque solid-body rotation validation.
The comparison target is:

`TMLP-u` must be better than `MLP-u1` under identical numerical conditions
except the reconstruction scheme.

This is a scalar advection benchmark. It checks whether the reconstruction
preserves three different shape classes after one full rigid-body rotation:

- slotted cylinder;
- cone;
- smooth cosine-bell hump.

## Physical And Numerical Setup

- Case: LeVeque solid-body rotation.
- Domain: `[0, 1] x [0, 1]`.
- Mesh: criss-cross / Union-Jack triangular mesh.
- Quick grid: `N = 100` (`100 x 100`), non-strict diagnostic/smoke mode.
- Paper/final strict grid: `N = 100` (`100 x 100`).
- Final time: `t = 1.0`, one full rotation period.
- Velocity field:
  - `u(x, y) = -2*pi*(y - 0.5)`
  - `v(x, y) =  2*pi*(x - 0.5)`
- Flux: `upwind`.
- Time integrator for the strict autoresearch verifier: `ssp_rk3`.
- Face quadrature: `n_face_quad = 2`.
- Face velocity mode: `central_avg`.
- Boundary condition: scalar Dirichlet state `0.0` on all boundary patches.
  The transported bodies do not touch the domain boundary during the exact
  one-period rotation.

## Initial Condition

The initial scalar field is the canonical LeVeque superposition:

- slotted cylinder centered at `(0.5, 0.75)`;
- cone centered at `(0.5, 0.25)`;
- smooth cosine-bell hump centered at `(0.25, 0.5)`;
- common body radius: `r0 = 0.15`.

After one period, the analytic solution returns to the initial field.

## Comparison Contract

The final strict comparison uses exactly two reconstructions:

- `MLP-u1`;
- `TMLP-u`.

Everything else must be identical:

- mesh;
- velocity field;
- flux;
- time integration;
- CFL;
- quadrature;
- boundary conditions;
- final time;
- scoring formula;
- plotting and post-processing.

Do not use scalar mass renormalization, case-tuned blending, fallback methods,
or different methods for different bodies.

All non-reconstruction numerical choices are fixed to the current best
LeVeque validation setup.  Future autoresearch iterations should vary only the
high-order reconstruction candidate unless the user explicitly opens a new
solver/flux/time-discretization study.

## Metrics

The strict verifier compares TMLP-u against MLP-u1 by final one-rotation
error under identical numerical conditions.  The primary PASS comparison is:

```text
TMLP-u global_E1 < MLP-u1 global_E1
```

For the current autoresearch quick gate, initial-shape preservation must also
be better than MLP-u1.  The smooth hump/cone/slotted-cylinder bodies should
retain their initial geometry after one rotation; for example, initially round
features should remain round rather than becoming visibly warped.  The
mechanical comparison uses the existing body-wise centroid, moment, peak, and
slot overlap diagnostics.

The verifier may still report body-wise `E1` ratios as diagnostics:

- `slot_ratio = TMLP-u slotted_cylinder_E1 / MLP-u1 slotted_cylinder_E1`
- `smooth_ratio = TMLP-u smooth_hump_E1 / MLP-u1 smooth_hump_E1`
- `cone_ratio = TMLP-u cone_E1 / MLP-u1 cone_E1`

The weighted ratio is user-defined:

```text
weighted_ratio = 0.10*slot_ratio + 0.45*smooth_ratio + 0.45*cone_ratio
```

The hump and cone are weighted more strongly because smooth-extremum and peak
preservation are central to the reconstruction claim.

## PASS Criteria

The primary PASS condition at the paper/final strict grid `N = 100` is:

```text
tmlpu_better_than_mlp_u1 = 1
```

Interpretation:

- TMLP-u must have lower final one-period global `E1` error than MLP-u1.
- TMLP-u must also beat MLP-u1 on the body-wise initial-shape preservation
  gate: centroid/moment/peak errors lower than MLP-u1 and slot overlap metrics
  higher than MLP-u1.
- The final scalar field boundedness and `max_wiggle` remain reported for
  review unless the user explicitly asks for a stricter boundedness gate.
- The slotted cylinder, cone, and smooth hump diagnostics should still be
  recorded so that a PASS result can be inspected for physical quality.

## Artifact Expectations

Each completed LeVeque validation should record:

- command used;
- reconstruction keys or candidate config;
- JSON metrics;
- plot comparing MLP-u1 and TMLP-u;
- pass/fail state;
- implementation/theory note for any method change.

Current project references:

- strict verifier: `solver/solve_T-MLP-u/tests/tmlpu_autoresearch_verify.py`
- rotation benchmark: `solver/solve_T-MLP-u/tests/test_2d_leveque_rotation.py`
- convergence runner: `solver/solve_T-MLP-u/tests/test_2d_leveque_convergence.py`
- prior handoff: `.omx/specs/autoresearch-tmlpu-leveque/CODEX_APP_HANDOFF.md`

## Future Use

Before future LeVeque autoresearch runs, use this file as the default
validation contract unless the user explicitly changes the benchmark.
