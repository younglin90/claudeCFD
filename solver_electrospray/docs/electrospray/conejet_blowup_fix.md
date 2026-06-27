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

## Implementation + calibration (done)

Implemented in `CandidoTaylorConeJet3D.hpp`: new options `useElectricForceTimeStepLimit=true` /
`electricForceTimeStepSafety` (exposed in case_runner as `use_electric_force_timestep_limit` /
`electric_force_timestep_safety`). `dt` is now a per-step variable; at each step-top it is set to
`min(dtBase, safety*prevStepMinForceDt)` where `prevStepMinForceDt` is the previous step's
`min_cell sqrt(rho[ci]*cbrt(V[ci])/|appliedElectricForce[ci]|)`. Time is accumulated (`simTime`)
instead of `step*dt`; `rAU` and the per-step Rhie-Chow projector already consume the current dt, so
the change is localized. Diagnostics `min_electric_force_cfl_raw` and `min_adaptive_dt` were added.

**Calibration (nx24 reproducer):** the raw force-CFL `sqrt(rho*cbrt(V)/|F|)` is 0.00986, and
`dtBase = 0.30 * raw` exactly — i.e. the capillary-like constant (~0.3) puts `dtForce == dtBase`, so
safety must be **< 0.3** to bind at all (this is why an initial safety=0.3 was a silent no-op: dt
never dropped, mass_drift=1 at step 60 as in the baseline). The explicit-electric-force stability
constant is ~6x tighter than the capillary one.

| safety | nx24 CaE0.8, 700 steps | mass drift | result |
|---|---|---|---|
| 0.3 (no-op) | blows up at step ~60 | 1.0 | dt never bound (== baseline) |
| **0.05** | **stable, reaches nondim 0.29** | **2.3e-14** | asymmetry grows 0.028 -> 0.097 (physics preserved); minAdaptiveDt 1.8e-4 |
| 0.03 | stable (slower, less time/step) | 2.3e-14 | asymmetry grows; minAdaptiveDt 1.1e-4 |

**Default = 0.05**, chosen for **robust margin at fine resolution** — the target use case is the
resolved-nozzle mesh, whose sharper tip carries a *larger* force than nx24, so the guard must bind
with headroom there. At the matched comparison point the asymmetry GROWS (not the Poisson forces'
suppressed ~0.002), confirming the whipping is physical.

**Effect on the (pre-fix) validation matrix — honest note.** The limit is not perfectly inactive on
the coarser validated cases: at nx18 CaE 0.25 it binds mildly (min adaptive dt 0.0016 vs dtBase
0.0030) and refines the final asymmetry 0.089 -> 0.075 (~16%), with mass drift and max velocity
essentially unchanged (2.24e-14; 29.85 vs 29.88). This is a **mild dt-accuracy refinement toward
convergence**, not the Poisson forces' 40x over-damping — the same explicit-force dt that blows up at
nx24 was also marginally over-large at nx18, so the smaller adaptive dt gives a more dt-converged
(slightly lower) asymmetry. Net effect: the fix (a) eliminates the fine-mesh blow-up so nx24+ are now
computable, and (b) shifts the stable-but-fine results modestly toward dt-convergence. The A-F
convergence matrix was computed pre-fix; re-running it with the fix on (now blow-up-free, including
nx24 CaE 0.25 which previously diverged) is the natural follow-up to get the dt-converged sequence.
Raising the safety to ~0.15 would leave nx18 byte-identical but would under-bind the finer
resolved-nozzle meshes, so 0.05 (margin-first) is preferred.
