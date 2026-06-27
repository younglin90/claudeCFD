# Cone-tip blow-up: diagnosis + fix (task #14)

## Symptom

The faithful solver runs stably on coarse meshes (nx=12/18) but **diverges on fine meshes**: nx=24
CaE 0.25 blows up at step ~1110 (~0.98 ms) — velocity 59 -> 756 -> O(1e6), mass -> 0 — localized at
the cone tip. The blow-up onset moves earlier as the mesh refines or the field strengthens.

## Not a precision issue

Confirmed the whole solver is **double precision** (`Vec3 = Eigen::Vector3d`, `ScalarField =
std::vector<double>`, all Eigen solvers `VectorXd`; zero `float`/`Vector3f`/`VectorXf` in the tree).
The conservation residuals stay ~1e-13 (double-level) right up to the blow-up, and the divergence is a
sudden ~30-step catastrophe, not gradual roundoff growth. It is an algorithmic (explicit-scheme
stability) problem, not a roundoff problem.

## Root cause — explicit Maxwell electric force exceeds the timestep stability limit

`dt` is selected once (CandidoTaylorConeJet3D.hpp:3036) as `min(dtAdv, dtCap, dtElectric)`, and on
these runs `dtElectric = 0.1*tau_e` (electric-relaxation) is binding and **mesh-independent**. The
explicit Maxwell body force `source += -scale*electric.faceCoupledForce` (line 3389) is **not bounded
by any dt term**, and its magnitude is unbounded at the cone tip (E ~ 1/tip-radius): `max_electric_
force` grows 110 (nx12) -> 740 (nx18) -> 9069 (nx24) with refinement. So as the mesh refines the
explicit force grows while dt stays fixed -> the per-step velocity kick `dt*F/rho` eventually
diverges. The capillary CSF force is tiny by comparison (~9), so it is not the cause.

## Ablation (fast reproducer: nx24, CaE 0.8 — baseline blows up at step ~60)

| configuration | stable? | radial asymmetry @ matched nondim 0.27 |
|---|---|---|
| baseline (explicit force, full dt) | **NO** — mass->0 by step ~60 | — |
| `use_poisson_bounded_vector_maxwell_force` | yes (mass 1.5e-14) | ~0.002 |
| `use_poisson_hybrid_maxwell_force` | yes (mass 1.7e-14) | ~0.002 |
| **explicit force + 3x smaller dt** (`electric_relaxation_timestep_safety` 0.1->0.03) | **yes (mass conserved)** | **~0.08** |

Two conclusions:
1. **The blow-up is a force-CFL (timestep) problem, not a force-formulation problem.** Keeping the
   *same* explicit force but shrinking dt 3x makes nx24 CaE 0.8 stable and conserving through the time
   where the baseline was destroyed.
2. **The cone-tip asymmetry growth is physical, not numerical.** With the faithful force + a stable
   (smaller) dt, the asymmetry still grows (~0.08 at nondim 0.27, ~40x larger than the Poisson-force
   runs). The Poisson bounded/hybrid forces stop the blow-up by **over-damping** the tip dynamics
   (asymmetry suppressed ~40x), which would corrupt the very whipping physics we are studying. This
   resolves the earlier "physical whipping vs numerical destabilization" caveat **toward physical**:
   the whipping survives a faithful-force, stable-dt integration.

## Fix — adaptive per-step electric-force CFL limit (physics-preserving)

Reject the Poisson force reformulations (over-damp). Instead add an explicit-force stability limit to
the per-step timestep, analogous to the capillary limit `dtCap = sqrt(dx^3/4pi)`:

```
dtForce = electricForceTimeStepSafety * min_cell sqrt(rho[ci] * dx / max(|appliedElectricForce[ci]|, eps))
dt_step = min(dtBase, dtForce)            # recomputed each step (lagged one step)
```

This keeps the faithful explicit Maxwell force (preserving the physical whipping) and only shrinks the
timestep where/when the force is large (the sharp tip), so coarse/low-field runs are unaffected
(dtForce > dtBase there -> no change, regression-safe) while fine/high-field runs stay stable. The
timestep is already consumed per-step via `rAU` (line 3137) and the Rhie-Chow projector is already
rebuilt per-step (line 3414), so making `dt` adaptive is localized. `electricForceTimeStepSafety` is a
new tunable option (default calibrated on the nx24 reproducer; must leave nx12/18 CaE 0.25 unchanged).

Cost: smaller dt at fine resolution = more steps to reach a given time (the faithfulness-vs-cost
tradeoff is intrinsic to resolving a near-singular tip explicitly), but the run stays physical and
stable. A future option is a semi-implicit Maxwell-force treatment to relax the dt penalty.
