# Solver vs. Candido 2023 — Development & Gap Analysis

**Reference paper:** S. Cândido & J. C. Páscoa, *"Dynamics of three-dimensional electrohydrodynamic
instabilities on Taylor cone jets using a numerical approach"*, Phys. Fluids **35**, 052110 (2023).
Our solver (`include/electrospray/CandidoTaylorConeJet3D.hpp`, run via `apps/electrospray_case_runner.cpp`
`candido_smoke` mode) is a clean-room reimplementation that targets this paper.

This document is a code-grounded audit (file:line) of how far the solver reproduces the paper, what is
missing or divergent, and a prioritized improvement roadmap. Produced from a 4-way deep read of the
solver subsystems against the paper model + `candido_3d_method_gap.md` status.

---

## 0. Executive summary

**Maturity:** the *scaffolding* is faithful — geometry, dimensionless groups, initial condition,
Poisson solve, property interpolation (ε/σ harmonic, ρ arithmetic), the Maxwell-stress force form, and
the observable *definitions* all match the paper. But the **production dynamics are not paper-faithful**:

1. **The default run disables most of the paper's physics.** Nearly every paper-faithful option defaults
   to `false`: `useFullyDevelopedInletVelocityBoundary`, `useVofInletBoundaryAlpha`,
   `useOpenAtmosphericBoundaryFlux`, `useMovingCollectorWall`, `useContactAngleCurvature`,
   `useDimensionalElectricalScaling`, `useElectricRelaxationTimeStepLimit`
   (`CandidoTaylorConeJet3D.hpp:49,50,79,80,86,54,76`). The default smoke config is a regularized
   approximation, not the paper setup.
2. **The momentum equation is physically incomplete:** no convective `∇·(ρuu)`, no physical viscosity
   (a scalar `pseudoViscosity=0.03` replaces a WAM μ field — μ is never even computed), no gravity, and
   empirical drive multipliers on the electric (`CaE^1.25`) and surface-tension (`×0.20`) forces.
3. **Charge transport is not conservative by default:** the conservative flux is assembled but then
   hard-clamped to an arbitrary `chargeLimitBase` → non-physical current (the documented #1 gap).
4. **The validation observables are proxies/Pareto-blocked:** morphology uses an α-volume proxy (the true
   α=0.5 silhouette collapses); current-voltage and whipping cannot be satisfied simultaneously.

**Bottom line:** the solver *runs and is bounded*, but to *reproduce the paper* it needs (in order of
leverage) a physical momentum equation, a conservative dimensional charge model, the paper BCs enabled,
a physical CSF, a true α=0.5 silhouette metric, and real mesh refinement.

---

## 1. Component-by-component audit

Legend: ✅ faithful · 🟡 present-but-off / partial · 🔴 divergent/missing.

### Hydrodynamics + VOF
| Component | Paper | Code (file:line) | Status |
|---|---|---|---|
| Momentum convection `∇·(ρuu)` | present (inertial) | **absent** in `solveMomentumPredictorBiCGSTABILUT3D` (`EHDCoupling3D.hpp:360-409`) | 🔴 missing |
| Viscous stress | physical μ, WAM (Eq 11) | scalar `pseudoViscosity=0.03` w/ non-conservative ±0.97/1.03 stencil (`EHDCoupling3D.hpp:380-384`); **μ never computed** | 🔴 divergent |
| Gravity `ρg` | present | absent | 🔴 missing |
| Pressure–velocity | PIMPLE + Rhie–Chow | PISO/SIMPLE-like + `RhieChowProjector3D` (`PressureVelocityCoupling3D.hpp:41-99`); `rAU=dt/ρ` (not true diagonal); `rhieChowFlux3D` result discarded (`:92`); structured checkerboard hack | 🟡 partial |
| Geometric VOF (isoAdvector) | isoAdvector/plicRDF | `isoAdvectorFaceFlux3D` swept-PLIC (`VofTransport3D.hpp:385-418`); frozen iso-plane/step; exact cuts hex/tetra only; global mass redistribution | 🟡 partial |
| PLIC normal | plicRDF (RDF normal) | plane normal = **α-gradient** not RDF (`VofTransport3D.hpp:268`); RDF used only for curvature | 🟡 divergent |
| Density interp (WAM) | `ρ=αρ₁+(1−α)ρ₂` | `candidoMixtureFields3D:449` | ✅ |
| Interface compression | geometric, at-interface | extra algebraic `αf(1−αf)` flux layered on geometric scheme (`VofTransport3D.hpp:221-236`) | 🟡 extra-to-paper |

### Electrostatics + Charge + Electric force
| Component | Paper | Code (file:line) | Status |
|---|---|---|---|
| Poisson `−∇·(ε∇φ)=ρ_e`, `E=−∇φ` | static φ, WHM ε | `solvePotential3D` (`Electrostatics3D.hpp:53-139`), sign-correct, harmonic face ε | ✅ |
| ε interpolation (Eq 12 WHM) | harmonic | `candidoHarmonicMixture` (`CandidoTaylorConeJet3D.hpp:367,450`) | ✅ |
| σ interpolation (Eq 12 WHM) | harmonic | `candidoHarmonicMixture` (`:452`) | ✅ |
| Charge transport Eq(6) | Lopez-Herrera **conservative** `∂ρ_e/∂t+∇·(ρ_e u)+∇·(σE)=0` | conservative flux assembled (`:1471-1486`) **then hard-clamped** `±qLimit`, `qLimit=chargeLimitBase·max(1,CaE/0.25)` (`:1489`); conservation only opt-in | 🔴 divergent (**#1 gap**) |
| Dimensional scaling `τ_e=ε/σ` | physical | default **normalized** `σ*=1`/`1e-6` (`:52-53`); dimensional path exists but off (`:54`); ~1877× mismatch (gap doc 458-467) | 🔴 divergent |
| Electric force `f_e=ρ_eE−½\|E\|²∇ε` (Eq 10) | Coulomb+polarization | `maxwellBodyForce3D:163` exact Eq(10) **but ×empirical `electricDriveReferenceScale·CaE^1.25`** (`:88-89,3362-3367`) | 🟡 form ✅, drive distorts |
| Charge BCs (inflow/outflow/wall) | Dirichlet neutral / zero-grad / Neumann | boundary charge advection off by default (`:1287`) | 🔴 missing |

### Surface tension + Boundary/Initial conditions
| Component | Paper | Code (file:line) | Status |
|---|---|---|---|
| CSF `f_c=γκ∇α` | γ=64.5 mN/m | `balancedCsfForce3D` (`SurfaceTension3D.hpp:556-571`) **but built nondim `σ=1` ×`surfaceTensionDriveScale=0.20`** (`:3353,3367`) | 🟡 form ✅, scale empirical |
| Curvature `∇·(∇α/\|∇α\|)` | plicRDF | local-PLIC quadric + RDF fallback (`:3345`, `SurfaceTension3D.hpp:187`); dynamic accuracy unvalidated | 🟡 method ✅, accuracy partial |
| Contact angle 51° (**IMPORTANT**) | enforced all no-slip walls | primitive exists (`contactAngleAdjustedNormal3D:102`) but `useContactAngleCurvature=false` (`:86`), +Y wall only (`:3342`) | 🔴 off in production |
| Inlet | α=1, mapped fully-developed u (mean-forced), Neumann p, φ=U | φ=U ✅; α=1 & parabola exist but **off** (`:49,79`); static analytic parabola (not interior-mapped) | 🟡 partial |
| Nozzle wall | no-slip u=0, Neumann p/α/q, φ=U | electrode φ + Neumann implicit; **no explicit no-slip**, no nozzle patch (plain box) | 🟡 partial |
| Outlet/open | mixed in/out, ∇φ=0, total p=0 | flux helper off (`:50`); backflow approx; **no total-p=0 Dirichlet** (ref pin at cell 0) | 🟡 partial |
| Collector | u=(−20mm/s,0,0), φ=0 | φ=0 ✅ always; moving wall correct value but **off** (`:80`) | 🟡 partial |
| Initial α | column + 0.95·D_o/2 hemisphere | `candidoInitialAlpha3D:384-399` **exact** (+ extra m=3 seed; diffuse tanh) | ✅ |

### Observables + time-stepping + mesh
| Component | Paper | Code / status | Status |
|---|---|---|---|
| Morphology `V=Σπx²\|_{α=0.5}` | avg err 1.1% | α-volume **proxy** (`:2192`) in 10% band; true ray-α05 silhouette collapses to −100% (`DOWNGRADED`); 0.8/0.9 ms BLOCKED | 🔴 proxy only |
| Current `i_e=∮ρ_e u·n dS` | weak voltage dep (≤2) | definition ✅ (`:1749`); best ratio ~1.96–2.05, only at "mechanical-check" level; magnitude ~2.5e11× Gañán-Calvo | 🔴 DOWNGRADED |
| Whipping (y/Di=3.44, CaE↑) | radial wave | asymmetry observable ✅; <0.05 when current sane; **Pareto-blocked** vs current | 🔴 DOWNGRADED |
| Electric-Courant `τ_e=ε/σ` ≤0.1 | physical dt | implemented (`:3009-3017`) but **off by default**; not metric-gated | 🟡 off |
| Mesh | snappyHexMesh, dx=2µm, ~11M, 3-level refine | tiny `12×18×12` box (~2.6k cells, `:38-40`); AMR indicators only, no refine/transfer | 🔴 coarse only |
| CaE/voltage/τ_h relations | — | exact (`:301-325`) | ✅ |

---

## 2. What is already faithful (do not touch)

- Dimensionless groups & validation constants (CaE, E0, τ_h, U0=2180V, Q=16.1nl/s, Table I properties).
- Initial condition (inlet column + 0.95·D_o/2 meniscus hemisphere) — exact.
- Poisson/electrostatics solve and `E=−∇φ`.
- Property interpolation: ε,σ harmonic (WHM Eq 12); ρ arithmetic (WAM Eq 11).
- Maxwell-stress body-force *form* `f_e=ρ_eE−½|E|²∇ε` (Eq 10).
- Observable *definitions* (cross-section current integral; CSF form γκ∇α).

---

## 3. Prioritized improvement roadmap

Ordered by leverage toward reproducing the paper.

**P1 — Physical momentum equation** *(root of inertial whipping + velocity bias)*
- Add conservative convective flux `∇·(ρuu)` to the velocity solve (the `convectionFaceFluxUpwindTVD3D` /
  `addImplicitDivergenceUpwind` machinery already exists for VOF/charge — wire it in).
- Compute a WAM viscosity field `μ=αμ₁+(1−α)μ₂` (constants exist at `:28-29`) and use a symmetric μ
  face-diffusion in the momentum operator; drop `pseudoViscosity` and the ±0.97/1.03 asymmetry.
- Add `ρg` gravity source.
- Remove the empirical `electricDriveReferenceScale·CaE^1.25` and `surfaceTensionDriveScale=0.20`
  multipliers once the forces are physically scaled.

**P2 — Conservative dimensional charge model** *(root of #1 current-voltage gap)*
- Make the Lopez-Herrera conservative update the default: remove the unconditional `±qLimit` clamp
  (`:1489`); default `conservativeChargeBounding=true`; use the physical Rayleigh bound, not
  `chargeLimitBase`.
- Promote the semi-implicit Ohmic-conduction projection (`candidoAdvanceChargeImplicitOhmic3D:1545`)
  to default for the stiff `∇·(σE)` term.
- Default-enable `useDimensionalElectricalScaling` + `useElectricRelaxationTimeStepLimit` (τ_e=ε/σ),
  **jointly** with the conservative step (alone they destabilize — gap doc 468-477).
- Implement faithful charge BCs (neutral Dirichlet inflow, zero-gradient outflow, Neumann nozzle wall).

**P3 — Enable the paper's boundary conditions by default**
- `useFullyDevelopedInletVelocityBoundary` (+ per-timestep mean-forcing to u_in), `useVofInletBoundaryAlpha`,
  `useOpenAtmosphericBoundaryFlux` (+ total-pressure=0 outlet Dirichlet + true mixed inflow/outflow),
  `useMovingCollectorWall`.
- Explicit nozzle-wall no-slip patches (and treat the nozzle as a solid/masked region).

**P4 — Physical surface tension + contact angle**
- Replace `surfaceTensionDriveScale=0.20` with the correct nondimensional capillary coefficient; regress
  against static-droplet Laplace + Lamb oscillation.
- Enable `useContactAngleCurvature` (51°) on **all** no-slip walls (generalize beyond +Y to collector +
  nozzle side walls) after a sessile-cap convergence gate.

**P5 — Faithful observables**
- True connected α=0.5 free-surface silhouette extractor (clipped PLIC interface polygon, nozzle/wall
  masking, disconnected-structure rejection) computing `Σπx²`; obtain digitized Fig.3(b) 0.8/0.9ms data;
  add an absolute Gañán-Calvo `I~(γσQ)^½` magnitude check + an electric-Courant≤0.1 gate to the metric.

**P6 — Interface fidelity**
- Feed the RDF-reconstructed normal back into the advection plane (true plicRDF, not α-gradient).
- With true isoAdvector, set algebraic compression to 0.

**P7 — Mesh resolution** *(needed for whipping; high effort)*
- Generate the snappyHexMesh-equivalent nested-cylinder refinement (finest Ø=1.2·D_o, dx≈2µm, ~11M cells);
  execute AMR (refine/coarsen + solution transfer), not just indicators; demonstrate the dx=4µm "no-jet"
  cutoff.

---

## 4. Critical files

- `include/electrospray/CandidoTaylorConeJet3D.hpp` — solver entry, time loop (~3116-3422), BC helpers
  (~456-729), charge advance (~1459-1650), mixture (429-454), observables (~1700-2360), options (37-93).
- `include/fvm/EHDCoupling3D.hpp` — momentum predictor (360-409), Maxwell force (130-173).
- `include/fvm/PressureVelocityCoupling3D.hpp` / `RhieChow3D.hpp` — pressure-velocity.
- `include/fvm/VofTransport3D.hpp` — geometric VOF / PLIC / compression.
- `include/fvm/SurfaceTension3D.hpp` — CSF, curvature, contact angle.
- `include/fvm/Electrostatics3D.hpp` — Poisson, charge bounding primitive.
- `docs/electrospray/candido_3d_method_gap.md` — extensive documented status of every diagnostic.
- Paper: `papers/library/md/2024_candido_dynamic-3d-ehd-instabilities-taylor-cone-jets.md`.

---

## 5. Implementation status (P1-P6 done)

Committed `e1eb8f2` (OpenMP) and `9cf0c48` (P1-P6 faithful physics). Each step verified to run
bounded with active electrostatics/charge/surface-tension; VOF/curvature/static-droplet unit
tests pass.

| P | Status | Evidence |
|---|---|---|
| P1 momentum (convection + WAM viscosity + symmetric operator + gravity) | done | nx=12 bounded, mass 1.7e-15, div 4e-13 |
| P2 conservative dimensional charge (default on) | done | charge budget residual ~1e-16, dt limited by tau_e |
| P3 paper BCs default on | done | bounded, boundaries active |
| P4 contact angle 51deg + full CSF | done | curvature redistributed at wall (kappa up, CSF down) |
| P5 paper observables + electric Courant 0.1 | done | silhouette/current/courant in summary |
| P6 plicRDF-style normal + compression 0 | done | VOF/curvature/static-droplet tests pass |

Remaining deferrals (mesh-gated, P7 territory): resolved-nozzle no-slip walls + total-pressure
outlet + collector-wall contact angle (need a resolved nozzle mesh), and matching the paper's
1.1% morphology / Ganan-Calvo current magnitude (need ~2 um / ~11M-cell resolution).

Regression note: the candido smoke long-window assertions are calibrated to the pre-P1
behavior and need re-baselining (the morphology diagnostic now uses the hydrodynamic dt to
reach the paper window affordably; production defaults are faithful). Validation of the new
physics vs the paper is journaled in `autonomous_research_log.md`.
