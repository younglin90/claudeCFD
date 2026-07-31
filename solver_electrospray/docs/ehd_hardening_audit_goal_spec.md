# 3D EHD Multiphase Solver Hardening Audit Goal

Produce an honest, evidence-graded hardening audit of the existing 3D EHD
multiphase solver. This is a verification goal, not a feature goal: add no new
physics.

Retest five specific prior claims at a stricter standard and assign each a
status:

- `UPHELD`: holds at the stricter bar.
- `APPROXIMATE`: holds only at the original looser bar.
- `DOWNGRADED`: fails the stricter bar, with the failing numbers.
- `BLOCKED`: cannot be tested, with what is missing.

A `DOWNGRADED` claim is a valid and expected outcome, not a failure to be
avoided.

## Hard Rules

- Do not relax, delete, disable, or skip any existing test. All 32 prior tests
  must still pass unchanged.
- Do not change core solver numerics to make a stricter check pass. If a
  stricter check exposes a genuine solver deficiency, record it as a finding
  with a recommendation and move on.
- External references must be genuinely external, never self-generated curves
  that beg the question.
- Fixing diagnostic/metric bugs and adding validation tests/fixtures is in
  scope.
- Evidence is live `cmake --build build && ctest --test-dir build
  --output-on-failure` output plus written CSV/log artifacts, never cached
  numbers.
- Boundaries: tests, fixtures, diagnostics, and benchmark/ledger modules of
  this repository. Core numerics modules are read-mostly; only
  diagnostic/metric bug fixes are allowed, and each must be flagged in the
  report.
- Eigen-only; no external CFD dependency.
- Never print, log, or commit secrets, credentials, or tokens.

## Audit Items

### A. EHD Terminal Claim

Current terminal evidence is a boundary pass: deformation error around 9.989%
against a 10% bar using leading-order Taylor theory.

Stricter audit:

- Validate steady deformation `D` against a published numerical or experimental
  leaky-dielectric dataset, such as Feng & Scott 1996 or Lac & Homsy 2007,
  reporting `D` versus electric capillary number.
- Require <=5% agreement in the small-deformation regime where the theory is
  valid.
- Sweep at least five `(permittivity ratio, conductivity ratio)` points spanning
  prolate and oblate.
- Run a 3-level Richardson mesh study showing `D` convergence.
- Report converged `D`, observed order, and gap to the external reference.

If no accessible external numeric dataset is available, mark `BLOCKED` and state
exactly what data would unblock the item.

### B. Static-Droplet Curvature Integrity

The adversarial static-droplet `Ca ~ 7e-11` is suspiciously good for 3D
unstructured meshes.

Stricter audit:

- Prove `kappa` is computed from the discrete alpha field, not prescribed nor
  implied by equilibrium initialization: add a test that perturbs alpha and
  confirms `kappa` responds.
- Report a curvature-convergence study for a sphere of known radius on at least
  three refined unstructured meshes using `kappa` error versus analytic `2/R`.
- Add a dynamic oscillating-droplet test matching Lamb mode-2 frequency within
  5% and the Prosperetti damping trend.
- If the original `Ca` came from prescribed or equilibrium curvature, state it
  and restate exactly what that number did and did not prove.

### C. TGV Metric De-Aliasing

`energy_error == enstrophy_error` to six digits suggests one quantity may have
been reported twice.

Stricter audit:

- Confirm whether this is true.
- Fix the diagnostic so energy and enstrophy are computed independently,
  computing enstrophy from resolved vorticity.
- Run a 3-level mesh study and report energy and enstrophy convergence orders
  separately.

### D. VoF Shape-Error Metric

The previous claim was "within target", but the threshold was not auditable.

Stricter audit:

- Define `shape_l1` explicitly as symmetric-difference volume normalized by
  initial volume.
- Pin a numeric pass threshold.
- Report mesh convergence of Rider-Kothe and Zalesak-3D interface errors.

### E. Operating-Envelope Characterization

The prior residual risks were listed but not probed. Push each to its failure
boundary. The boundary number itself is the deliverable.

Required envelope numbers:

1. Raise skew/non-orthogonality until diffusion MMS order drops below 1.9 or the
   solve diverges; report the ceiling angle.
2. Raise density ratio past 1000:1 (`1e4`, `1e5`, ...) until static-droplet `Ca`
   exceeds `1e-5` or the solver fails; report the ceiling.
3. Raise cell aspect ratio until Eigen ILUT yields NaN or breakdown; report the
   ceiling and whether a fallback preconditioner avoids it.
4. Lower `tau_e/dt` until quasi-implicit charge transport loses boundedness;
   report the stiffness limit.

## Ledger

Between iterations, append to an audit ledger:

- Item.
- Stricter test added.
- Live numbers.
- Assigned status.
- Next item.

## Completion

Before completion, run an adversarial review pass that rechecks whether any
`UPHELD` status secretly depends on a still-too-easy test, and downgrade it if
so.

The goal is complete when:

- Items A-E all carry evidence-backed statuses.
- The four envelope numbers in E are reported.
- Existing prior tests still pass; no prior test has been relaxed, deleted,
  disabled, or skipped.
- `cmake --build build && ctest --test-dir build --output-on-failure` passes.

The final report must contain:

- Claim-by-claim ledger: claim, stricter test and command, final numbers,
  status, remaining uncertainty.
- Characterized envelope: four boundary numbers.
- Every diagnostic or metric bug fixed.
- Recalibrated confidence score with justification for any change from `0.82`.
- Explicit list of original headline claims now weaker than they first appeared.
