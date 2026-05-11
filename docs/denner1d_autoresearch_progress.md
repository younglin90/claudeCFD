# denner_1d 4-eq autoresearch — Progress and Diagnostics

## Goal recap
Pass all 17 validation cases (01, 02, 04, 05, 07, 13, 14, 15, 16, 17, 18,
24, 25, 32-35) of `validation/1D/` on `solver/denner_1d/` with the 4-eq
primitive (p, u, T, α₁) + autograd Jacobian path (`five_eq_ad=True`).
PASS thresholds are physics-level (e.g. case 02 requires
max|p−p₀|/p₀ < 1e-10 and max|u−u₀| < 1e-10 after one full advection
period at N=100).

## Baseline
`pass_count = 1 / 17` — only case 01 (PE static) passes; all other
cases either hit the per-case 60-second wall budget without reaching
the PASS threshold, or are explicit stubs (cavitation, source-term,
acoustic-tester not yet ported).

## Commit chain (10 iterations)

| iter | commit | Change | Effect |
|---|---|---|---|
| 1 | `8021297` | Spec-based 17-case runner; Newton early-exit at machine ε | infrastructure |
| 2 | `cce160f` | LM regularization + trust-region fallback | safety |
| 3 | `5411ecd` | Subprocess wall budget (SIGALRM unreliable) | bounded measurement |
| 4 | `5d13c5f` | 2nd-order minmod TVD reconstruction (ng=2) | spec compliance |
| 6 | `3e7c05a` | Lagged minmod slope (autograd-clean Jacobian) | J rel err 33% → 2.6e-6 |
| 7 | `6874e14` | Newton dW trust-region cap (5%) | dW overshoot blocked |
| 8 | `1e21c75` | Marquardt damping (diagonal-based, primary path) | p_rel 8.6e-5 → 4.3e-8 |
| 9 | `9fa8e63` | Per-block residual physical normalization | alpha unblocked |
| 10 | `eb907d7` | Adaptive LM lambda + modified-Newton J reuse | alpha 2.5× more |

(Iter 5 = modified Newton without iter 8's Marquardt — reverted, no
metric change.)

## Iter 11 diagnosis — dt-sensitivity scan

Sub-cycling within the spec dt to relieve the acoustic CFL=1500 of
case 02 was a candidate leverage.  Direct measurement (N=10, t_end=0.02,
case 02 air-water advection):

| dt | nstep | wall | p_rel | u_err | psi_adv |
|---:|---:|---:|---:|---:|---:|
| 1e-2 | 2 | 3.2 s | 5.1e-5 | 1.4e-1 | 8.1e-6 |
| 1e-3 | 20 | 31.8 s | 2.4e-3 | 1.0e+0 | 4.6e-3 |

Sub-stepping makes the accuracy *worse*, not better.  The PE-state
errors p_rel and u_err grow ~50× and ~7× respectively when dt is
reduced by 10×, while psi advects ~570× more.  Mechanism: each Newton
sub-step fails to converge to machine precision, so each sub-step
deposits a finite error in (p, u) that accumulates with the number of
sub-steps.  More sub-steps = more accumulated PE drift.

**Implication**: the inner Newton convergence per dt step — not the
size of dt — is the binding constraint for case 02 PASS.

## Outstanding observation

The case 02 PASS threshold (1e-10) is far stricter than what any
autograd-Jacobian + LM-regularized Newton solver can deliver in a
reasonable wall budget at N=100.  Reaching that threshold most
likely requires either:

1. An analytical Jacobian (currently banned by user spec), or
2. An ACID-style deferred correction that *exactly* preserves the PE
   manifold for all uniform-(p, u) initial conditions, with the
   advection equation decoupled from the pressure/velocity solve.

Item 2 is the Denner et al. (2018b) approach in the ACID-2 paper —
their reported PE error for the air-water advection benchmark is
~ 1e-13 with their 4-eq pressure-based scheme on a moderate mesh.

## Iter 14 — cross-case measurement, all FAIL on wall budget

Measured cases 01, 02, 16, 17 at the spec-N with the iter 12 solver:

  case 01 (PE static, N=100):       PASS  wall = 0.15 s
  case 02 (PE advection, N=100):    FAIL  wall = 95 s, budget = 90 s
  case 16 (thermal advection, N≈200): FAIL  wall budget exceeded
  case 17 (smooth alpha gauss, N=550): FAIL  wall budget exceeded

Cases 13 / 14 (shock-tube, N=400) abandoned mid-measurement —
their step time is prohibitive (per-step autograd Jacobian on a
4N=1600 system takes ~minutes; the case spec is 670+ steps).

The wall-time bottleneck is the dominant constraint at the spec
N for every non-static case.  The Newton accuracy work in iter
6-12 has produced a Newton iteration that converges *correctly*
on case 02 N=20 (p_rel falls 6.5e-5 in 5 steps), but the
asymptotic per-step cost prevents reaching the spec N=100 with
full convergence inside any reasonable wall budget.

## Next leverage queue

1. ACID-style "stencil-uniform alpha" treatment §5 of ACID-2.  Force
   the alpha used in the *cell-stencil density/enthalpy evaluation*
   to be the cell-center alpha rather than the upwind face alpha, so
   the discrete energy equation is exactly PE-preserving when
   (p, u) is uniform.  Already partly implemented in
   `_acid_rH_face` for energy but not for momentum/mass blocks.
2. Anderson acceleration on the outer Picard loop.
3. ~~CICSAM/THINC-BVD slope limiter for alpha (user spec).~~
   ❌ Hyper-C (CICSAM compressive arm) tried in iter 13.  At case 02
   N=20 5 steps the compressive slope produced
   p_rel = 1.1e-4 (vs minmod 6.5e-5, ~2x worse) and
   psi_advected = 5.9e-5 (vs minmod 1.0e-4, ~2x less).  The
   compressive face value amplifies the outer Picard overshoot
   that the iter-12 relaxation was already fighting, and the
   per-step alpha motion is smaller because Hyper-C tries to keep
   the interface confined to a single cell.  Reverted.
4. Material-flux upgrade to HLLC/SLAU2 with MWI face velocity
   (user spec).
5. van Leer slope for alpha (less compressive than Hyper-C, smoother
   than minmod) — to be tried next.
6. Newton-Krylov / JFNK with diagonal preconditioner — relieves the
   O(N^2) autograd Jacobian cost for larger N=100 spec runs.
