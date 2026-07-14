# five_eq_IMEX Flux Scheme Research Plan

Date: 2026-04-28

## Current Finding

The 07-B Air-Water failure is no longer primarily a material-contact PE drift.
The current evidence points to an acoustic-interface flux inconsistency:

- `F_E` advects phase mass, momentum advection, total energy advection, and
  `alpha` with the face state from `face_state.py`.
- `F_I` uses a separate acoustic Riemann face state `(p*, u*)`.
- PE tangent projection can suppress interface pressure ringing, but it also
  modifies the momentum residual direction and shifts the `u` extremum location.
- Contact-only PE projection preserves the acoustic phase better, but removes
  the stabilizer and the Air-Water case becomes unstable.

Therefore, the next repair should not be another residual projection.  It should
make the face fluxes use one acoustic/interface model.

## New 07 Gate

The 07-B acceptance gate now also requires peak-location agreement:

```text
argmax_i |p_i - p0| == argmax_i |p_i^exact - p0|
argmax_i |u_i|      == argmax_i |u_i^exact|
```

Signed max/min locations are also recorded when the exact signed extremum is
not a near-zero plateau.  This prevents a diffused or phase-shifted pulse from
passing only by loose profile correlation.

## Paper-Derived Constraints

### He/Tan 2024 and Zhao/He 2025 MMACM/MMACM-Ex

The useful idea is not simply `G_alpha` sharpening.  The important part is the
conservative consistency of every correction flux:

```text
G_q1 =  rho1_up * G_alpha
G_q2 = -rho2_up * G_alpha
G_m  = (rho1_up - rho2_up) * u_up * G_alpha
G_E  = (rho1_up E1_up - rho2_up E2_up) * G_alpha
```

This is why the old He2024 m1 path did not show the same interface pressure
ringing: interface corrections were applied at the flux level, with mass,
momentum, energy, and volume fraction tied to the same face path.  In contrast,
the current `five_eq_IMEX` PE projection changes a residual after the fluxes
are assembled, and that can rotate the acoustic eigenvector.

### Deng/Shyue/Xiao MUSCL-THINC-BVD

High-order or sharp-interface reconstruction alone is not a cure.  The paper
explicitly warns that sharpening only the interface variable can induce pressure
and velocity oscillations unless the remaining variables are reconstructed or
corrected consistently.  This supports keeping THINC/G-alpha off while debugging
the acoustic flux, then adding any sharpening only with coupled flux corrections.

### Low-Mach/All-Speed Riemann Corrections

All-speed HLLC/AUSM/SLAU corrections reduce excessive pressure diffusion by
scaling velocity/pressure jumps using a local Mach sensor.  For this solver the
analogous requirement is narrower:

- do not use acoustic Rusanov diffusion in the explicit material flux,
- do not use PE projection as an acoustic stabilizer,
- apply any pressure damping through the acoustic face flux itself.

## Proposed Flux Direction

### Phase A: Unified Acoustic Face State

Add a new flux option, tentatively:

```text
interface_flux_mode = "unified_acoustic"
```

In a narrow band around `|delta alpha| > eps`, compute `(p*, u*)` once from the
linear acoustic Riemann formula:

```text
p* = (Z_R p_L + Z_L p_R + Z_L Z_R (u_L-u_R))/(Z_L+Z_R)
u* = (p_L-p_R + Z_L u_L + Z_R u_R)/(Z_L+Z_R)
```

Then use the same `u*` for the interface-band advective fluxes:

```text
F_q1    = alpha_f rho1_f u*
F_q2    = (1-alpha_f) rho2_f u*
F_alpha = alpha_f u*
F_ru,E  = rho_f u* u*
F_p     = p*
F_pu    = p* u*
```

Outside the interface band, keep the current low-dissipation face velocity.

Rationale: this removes the current split where material transport sees one
velocity and pressure work sees another.  It should preserve the acoustic peak
position better than residual PE projection.

### Phase B: Contact-Only PE Stabilizer

Keep PE projection off for acoustic interfaces:

```text
allow_PE_projection =
    |delta alpha| > eps_alpha
    and |delta p|/p_ref < eps_p
    and |delta u|/c_ref < eps_u
```

If a stabilizer is still needed for Air-Water, add it as a face-flux correction,
not as a residual projection.  The first candidate is an energy-only pressure
equilibrium correction; it should not touch the momentum row and therefore
should not shift `u` peak location.

### Phase C: Optional He2024-Style Consistency Flux

Only after Phase A passes 02 and improves 07, reintroduce MMACM-style
corrections as a bounded option:

```text
G_alpha enabled only for pressure-flat contacts
G_q1, G_q2, G_m, G_E applied together
```

For the user's current request, keep `G_alpha` off in 07 acoustic runs.

## Immediate Test Matrix

Run these in order:

1. Baseline current active path with new peak gate.
2. Contact-only PE projection, no residual acoustic projection.
3. New `unified_acoustic` interface-band velocity for `F_E`.
4. New `unified_acoustic` plus energy-only PE correction, if needed.

Required logs per 07 subcase:

```text
L2p Lip L2u Liu corr_p corr_u p_alt u_alt p_peak u_peak
```

PASS requires profile similarity, no interface oscillation, and exact cell
match of the pressure and velocity absolute peak locations.

## Expected Failure Modes

- If pressure ringing remains but `u_peak` is correct, add pressure-flux damping
  only in the acoustic face state.
- If `u_peak` shifts, the stabilization is still modifying the momentum
  characteristic and must be removed or moved to energy-only.
- If 02-A regresses, the interface-band detector is too broad or affects
  pressure-flat contact transport.

