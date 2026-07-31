# Terminal EHD Multiphase FVM Goal

Evolve the current 2D foundational FVM codebase into a validated full-3D unstructured-polyhedral, leaky-dielectric electrohydrodynamic (EHD) multiphase (VoF) solver.

This is a terminal Goal: it is complete ONLY when the final EHD coupling validation passes in 3D (Leg 5 below); every earlier leg is intermediate evidence, not completion. Progress through five ordered legs, advancing to the next leg only after the current leg's gate is green, and after every prior leg's ctest cases STILL pass (cumulative non-regression). All evidence is the live output of `cmake --build build && ctest --test-dir build --output-on-failure` plus the written CSV/log artifacts - never cached or remembered numbers.

## Leg 1: Lift Core To 3D

Generalize mesh, geometry (V, Cf, Sf, d, over-relaxed Sf=Delta_f+k_f), grad/div/snGrad/laplacian, Rhie-Chow, and pressure-velocity coupling to arbitrary 3D polyhedra.

Gate:
- 3D lid-driven cavity within 2% L2 of a trusted reference.
- 3D Taylor-Green vortex energy/enstrophy decay within 5% over the resolved window.
- 3D skewed-mesh diffusion MMS slope >=1.9.
- max|div u| <= 1e-10.
- Existing 2D cases retained as single-cell-thick degenerate guards that still pass.

## Leg 2: VoF/MULES

Bounded algebraic VoF with compression.

Gate:
- 3D deforming-sphere (Rider-Kothe) and slotted-sphere (Zalesak-3D) shape error within target.
- Relative mass-conservation drift <= 1e-3.
- 0<=alpha<=1 strictly enforced.

## Leg 3: Surface Tension, Balanced-Force CSF

HARD INVARIANT: `kappa_f*(grad alpha)_f` and `(grad p_rgh)_f` MUST use the identical snGrad face operator; any divergence is a defect, not a knob.

Gate:
- 3D static droplet (sigma only, g off) drives Ca = mu*Umax/sigma <= 1e-6 and holds it non-increasing for 1000 steps.
- Laplace jump matches 2*sigma/R within 2% on >=3 resolutions.
- Adversarial gate WITH numeric thresholds: re-run on a fully irregular 3D polyhedral mesh AND at density ratio 1000:1 and require Ca <= 1e-5; do not pass the leg otherwise.

## Leg 4: Electrostatics, Leaky Dielectric

Solve `-div(eps*grad phi)=rho_e` reusing the Leg-1 Laplacian machinery with harmonic-mean eps_f at interfaces, plus bounded charge transport.

Gate:
- Concentric-sphere and parallel-plate potential within 1% L2 of analytic.
- Charge field bounded and conservative.

## Leg 5: EHD Coupling, Terminal

Body-force form `f_e = rho_e*E - 0.5*|E|^2*grad(eps)`, with `grad(eps)` imposed at faces like surface tension; couple `phi->E->f_e` each PIMPLE outer iteration; use quasi-implicit charge transport when `tau_e=eps/sigma_e < dt`.

Gate (completion condition):
- A 3D leaky-dielectric droplet reaches steady deformation D matching Taylor's small-deformation D_T within 10% across a discriminating set of (permittivity, conductivity) ratios spanning BOTH prolate and oblate regimes, with the correct internal circulation sense.

## Constraints For All Legs

- Collocated Rhie-Chow mandatory.
- Symmetric systems (pressure, potential) solve with ConjugateGradient + Incomplete-Cholesky.
- Asymmetric systems (momentum, transport) solve with BiCGSTAB + ILUT.
- Eigen for linear algebra ONLY.
- No OpenFOAM or external FVM/CFD dependency.
- The snGrad and gradient operators stay as single reusable functions shared across legs.
- Timestep obeys both Courant and capillary `dt <= sqrt(rho*dx^3/(4*pi*sigma))`.
- `eps_f` by harmonic mean.
- The solver must be general 3D polyhedral, NOT 2D and NOT axisymmetric-only.
- Boundaries: this repository and its fvm modules + tests + fixtures.
- Existing 2D code may be refactored into 3D.
- Do not pull in external CFD frameworks.

## Iteration Ledger

Between iterations, append to a running ledger:
- Current leg.
- Exact change.
- The leg's key numbers: Ca, Laplace error %, MMS slope, mass drift, potential L2, deformation D vs D_T.
- Pass/fail of each gate.
- The single next highest-impact action.

At a budget limit, STOP substantive work, report which leg and gate were reached and the next step, and do NOT mark the Goal complete.

## Final Report Requirements

The final report MUST contain:
- Files created/changed, each with a one-line role.
- Exact verification commands and final numeric outputs per leg.
- Regression guards now in place.
- Remaining risks: 3D unstructured curvature noise, density-ratio ceiling, non-orthogonality angle limit, ILUT stability at high aspect ratio, charge-relaxation stiffness.
- Calibrated confidence score.

Never print, log, or commit secrets, credentials, or tokens.

If any leg's gate cannot be met or progress stalls with no defensible path under these boundaries, stop and report:
- Attempted approaches.
- Evidence gathered.
- Precise blocker.
- Exact input or constraint relaxation needed to proceed.
