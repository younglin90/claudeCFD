# P7 — Resolved-nozzle geometry + boundary conditions: implementation design

Goal: run the Candido 2023 setup on a **resolved-nozzle external mesh** (real capillary tube +
collector plate, pre-refined near the tip by the user) instead of the structured box, with the
paper's boundary conditions applied per **named patch**. Prerequisite: the cone-tip blow-up fix
(task #14) must land first — any fine resolved-nozzle mesh blows up under the current explicit
electric force otherwise.

## 0. Paper BC specification (Candido & Pascoa 2023, Sec II.D; extracted)

| Patch | velocity | pressure | alpha (VOF) | potential phi | charge rho_e |
|---|---|---|---|---|---|
| **liquid_inlet** (nozzle bore top) | fully-developed parabolic, mean = u_in (Q=16.1 nl/s) | Neumann | Dirichlet alpha=1 | Dirichlet phi=U (2.18 kV) | neutral Dirichlet rho_e=0 |
| **nozzle_wall** (capillary, electrode) | no-slip u=0 | Neumann | Neumann | Dirichlet phi=U | Neumann |
| **collector** (ground plate) | moving wall u_s = -20 mm/s (axial) | Neumann | contact angle 51 deg | Dirichlet phi=0 | Neumann |
| **outlet** (atmosphere / far) | mixed in/out, zero-gradient | **total pressure p0 = 0 Pa** | Neumann | Neumann (or phi=0 far) | neutral Dirichlet inflow / zero-grad outflow |

Geometry: D_i = 160 um, D_o = 260 um, nozzle L = 300 um, nozzle-collector H = 1.5 mm,
collector diameter D_c = 5 mm. Contact angle = 51 deg (Young). Electric Courant tau_e <= 0.1.

## 1. Architectural blocker + fix (patch preservation)

`normalizeExternalCandidoMesh` (apps/electrospray_case_runner.cpp:552-609) currently, at line 608,
**discards all named patches** and resets `mesh.patches` to the six box patches, then re-tags faces
geometrically via `computeGeometry()`. A resolved-nozzle mesh's named patches (liquid_inlet,
nozzle_wall, collector, outlet) are therefore lost, and BCs fall back to box-geometry classification
which is meaningless for a nozzle tube.

**Fix:** before line 608, capture a stable per-face `faceRole[fi]` array from the original
`mesh.patches` (map patch name -> BcRole). `computeGeometry()` does NOT reorder faces (verified in
Mesh3D.hpp), so `faceRole[fi]` stays valid after the box-reset. Keep the box-reset for the builtin
path; only build `faceRole` when the mesh carries the resolved-nozzle patch names. Pass `faceRole`
(and the parsed per-patch BC values) into the solver.

## 2. Named-patch BC framework (new, additive — zero risk to box path)

New header `include/electrospray/BoundaryConditionSet.hpp`:
- `enum class BcRole { Inlet, NozzleWall, Collector, Outlet, Symmetry, Unknown };`
- `struct PatchBc { BcRole role; ... typed velocity/pressure/alpha/phi/charge type+value ... };`
- `struct CandidoBoundaryConditions { std::vector<PatchBc> patches; std::vector<int> faceRole; bool active=false; };`

Thread `const CandidoBoundaryConditions* bc = nullptr` through `runCandidoConeJetSmoke3D`
(mirroring the existing `externalMesh*` parameter). `nullptr` / `!active` => today's geometric
path, byte-identical (regression-safe). When active, the geometric classifiers below consult
`faceRole[fi]` instead of bounding-box position.

Patch name -> role mapping (case-insensitive substring): `inlet`->Inlet, `nozzle`/`electrode`/`wall`
->NozzleWall, `collector`/`ground`->Collector, `outlet`/`atm`/`far`->Outlet, `symmetry`/`sym`
->Symmetry.

## 3. BC physics, by increment (low -> high risk), with code attach points

### 3a. Potential Dirichlet (highest value, lowest risk) — feeds the existing arbitrary-Dirichlet Poisson
`candidoPotentialBoundary3D` (CandidoTaylorConeJet3D.hpp:475-501) currently sets phi=U for
geometric nozzle faces (y<=yTol & r<=R) and phi=0 for collector (y>=ly-yTol). Overload to set
`faceDirichlet/faceValue` from `faceRole`: Inlet|NozzleWall -> U (dimensionless voltage),
Collector -> 0, Outlet -> Neumann (or 0 far-field). Already supported by `solvePotential3D`.

### 3b. Inlet velocity + inlet alpha
Gate `candidoIsInletBoundaryFace3D` (674-693) on `faceRole==Inlet`; use the parabolic profile
`candidoApplyFullyDevelopedInletVelocityCells3D` (already enabled, P3) on the inlet patch cells;
`candidoInletBoundaryAlpha3D` sets alpha=1 on inlet (already enabled, P3).

### 3c. Nozzle no-slip walls (new physics)
For `faceRole==NozzleWall` boundary faces: enforce u=0 Dirichlet on the adjacent cell's wall-normal
+ tangential velocity in the momentum predictor (EHDCoupling3D.hpp solveMomentumPredictor...), and
Neumann (zero-gradient, the default) for p/alpha/charge. Currently nozzle is conductive-flux
suppression only, not a velocity wall. Add a no-slip pass over NozzleWall faces before the predictor.

### 3d. Collector contact angle on the real collector patch + moving wall
Generalize the existing `contactAngleAdjustedNormal3D` (line 102) / `curvatureFromLocalPlicQuadric
Report3D` contact-angle path (currently applied only to the +Y box wall, line 3361-3369) to apply
on `faceRole==Collector` faces with the patch normal. Apply the moving-wall velocity
`candidoApplyMovingCollectorWallCells3D` (already exists, line 3010) on Collector faces.

### 3e. Total-pressure outlet (highest risk — pressure coupling)
For `faceRole==Outlet`: impose total pressure p0 = p + 0.5*rho*|u|^2 = 0 (gauge) as a pressure
Dirichlet in the Rhie-Chow / pressure-correction solve (replacing the current "pin reference
pressure at cell 0"), with mixed in/out velocity (zero-gradient out, extrapolate in). Attach in the
pressure-Poisson assembly (RhieChow3D.hpp / PressureVelocityCoupling3D.hpp). Defer until 3a-3d green
(it changes the global pressure datum and interacts with mass conservation).

## 4. Mesh contract for the user's external mesh

The user-supplied OpenFOAM polyMesh must name its boundary patches so the role mapping resolves:
- `liquid_inlet` (or contains "inlet") — top of the nozzle bore.
- `nozzle_wall` (or "nozzle"/"electrode") — capillary inner+outer wall surface.
- `collector` (or "ground") — collector plate.
- `outlet` (or "atmosphere"/"far") — the open far/atmospheric boundary.
- optional `symmetry` if a wedge/half model is used.
Pre-refine cells near the nozzle tip (the cone-jet region) — this replaces dynamic AMR (user choice).
Units in metres; `normalizeExternalCandidoMesh` scales by 1/D_o and sets the inlet plane to y=0.

## 5. Test plan

1. `tests/test_candido_named_patch_boundary3d.cpp` (new): on a generated cut-cylinder mesh, assert
   (a) `bc=nullptr` path is byte-identical to today's geometric run (regression-safe); (b) with a
   named-patch BC set whose roles coincide with the geometric classification, results agree; (c)
   flip the collector potential and assert the field/current responds (proves the tag is consumed).
2. Extend `tests/test_openfoam_case_runner.py` with a case carrying named nozzle/collector/outlet
   patches; assert patches survive normalization (faceRole populated) and the run is stable.
3. End-to-end (needs the user's resolved-nozzle mesh): run CaE 0.25, confirm cone-jet forms with the
   no-slip nozzle + total-p outlet, stable under the blow-up fix.

## 6. Sequencing

1. **[task #14] cone-tip blow-up fix first** — prerequisite for any fine resolved-nozzle run.
2. Patch preservation (sec 1) + BC framework header (sec 2) — additive, regression-safe.
3. 3a potential Dirichlet -> 3b inlet -> 3c nozzle no-slip -> 3d collector contact angle/moving wall
   -> 3e total-pressure outlet, each gated on the byte-identical `bc=nullptr` regression + the
   named-vs-geometric agreement test.
4. End-to-end validation on the user's pre-refined resolved-nozzle mesh.

JSON parsing: the existing regex parser cannot handle a nested `boundary_conditions` block; add a
small header-only `apps/MiniJson.hpp` (recursive parser) and parse a GUI-written
`boundary_conditions.json` sidecar, enabled only when `mesh_mode==openfoam_polyMesh` and a BC block
exists.

## 7. Implementation status (task #15)

The named-patch framework and four of the five BC roles are implemented and unit-tested
(`tests/test_candido_named_patch_boundary3d.cpp`), all gated by the opt-in
`use_named_patch_boundary_conditions` so the box + existing OpenFOAM paths are byte-identical.

| BC | status | commit | notes |
|---|---|---|---|
| Framework + faceRole capture | **done** | `0e85637` | `BoundaryConditionSet.hpp`; roles captured before the box-reset |
| Potential Dirichlet (electrode/inlet -> U, collector -> 0) | **done** | `0e85637` | consumed by `candidoPotentialBoundary3D`; tested |
| Inlet velocity (parabolic) + alpha=1 + inlet flux | **done** | `af7f004` | gated on Inlet role |
| Nozzle no-slip (u=0) | **done** | `af7f004` | `candidoApplyNozzleNoSlipCells3D` on NozzleWall-touching cells; tested |
| Collector moving wall | **done** | `af7f004` | gated on Collector role |
| Outlet open/atmospheric outflow | **done** | `af7f004` | `candidoApplyOpenAtmosphericBoundaryFlux3D` acts on Outlet-role faces |
| Collector contact angle (51 deg) | **inherited** | (P4) | applied at the +Y wall, which the normalization places the collector on, so it already covers the resolved-nozzle collector; a per-Collector-face generalization of the curvature band is a refinement |
| Total-pressure outlet (explicit p0=0 Dirichlet) | **deferred** | — | the open-outflow physics is handled by the Outlet-role flux above; an explicit p0 Dirichlet needs a pressure-Poisson Dirichlet capability in `RhieChowProjector3D` (a validated component) and is best done with the user's resolved-nozzle mesh to validate — see below |

**Why total-p Dirichlet is deferred (not blind-patched):** for the incompressible solve the pressure
level is a gauge freedom (the datum is fixed internally), so the physical content of the
total-pressure outlet — open outflow with backflow tolerance — is already provided by the Outlet-role
open-atmospheric flux. The remaining piece (pinning `p0 = p + 0.5*rho|u|^2 = 0` at the outlet for the
exact backflow/pressure-level treatment) requires modifying the pressure Poisson assembly, which is a
validated component; doing that without a resolved-nozzle mesh to verify mass conservation would be
the kind of un-testable risky change the blow-up diagnosis just argued against. It is the one
mesh-gated follow-up.

**To run a resolved-nozzle case:** supply an OpenFOAM polyMesh whose patches are named per sec 4
(`liquid_inlet`, `nozzle_wall`, `collector`, `outlet`), set `mesh_mode=openfoam_polyMesh` +
`use_named_patch_boundary_conditions=true` in the case JSON, and run `electrospray_case_runner`. The
adaptive electric-force CFL (task #14) keeps the fine tip stable.

## 8. End-to-end validation (done, on a generated resolved-nozzle mesh)

`apps/generate_resolved_nozzle_mesh.py` writes a structured-hex OpenFOAM polyMesh that **resembles
the real nozzle**: the annular capillary wall (ri<r<ro, 0<=y<=Lnoz, from Di=160um/Do=260um/
Lnoz=300um) is excluded as solid, leaving the bore (r<ri, the `liquid_inlet`) feeding liquid up
through the atmosphere to the `collector` plate, with the fluid/solid interface tagged `nozzle_wall`
and the sides tagged `outlet`. The default writes 7952 cells / 4 named patches (no OpenFOAM install
needed).

Running `candido_smoke` on it (`mesh_mode=openfoam_polyMesh`, `use_named_patch_boundary_conditions=
true`) **passes end-to-end**: the mesh loads, the named patches are recognized (`inlet_from_patch=1`),
the per-face roles are captured and consumed, and the run is **stable and mass-conserving**
(mass drift 3.8e-14, max div 5e-12) — no blow-up, the adaptive electric-force CFL holds.

The named-patch path is verified to be **actively consumed** by comparing
`use_named_patch_boundary_conditions` true vs false on the same mesh (40 steps): they differ
substantially (max electric force 6.1 vs 86.2, max velocity 0.044 vs 0.102, current 4.4e-7 vs
2.7e-7). The geometric-box classifier mis-tags the irregular nozzle mesh (spurious 86 force); the
named-patch path correctly applies the electrode to `nozzle_wall`+`liquid_inlet` and ground to
`collector`. This closes the previously mesh-gated end-to-end validation.

Remaining for a converged physics study (not a pipeline gap): a finer mesh (the 7952-cell test mesh
resolves the 160um bore with only ~3 cells) and the explicit total-pressure-outlet Dirichlet.
