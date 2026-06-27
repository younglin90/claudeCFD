# Mesh-setup guide — building resolved-nozzle cases for the tip-defect study

How to build a mesh + run a case for the Candido capillary electrospray with a tip defect. All
physical inputs are baked into the solver's `CandidoTaylorConeJetSetup`; you supply the **geometry
(mesh)** and choose the operating point. Keep every physical/operating input FIXED across
sharp/D1/D2/D3 and vary ONLY the mesh geometry.

## 1. Geometry (physical units = metres)

Candido capillary (the validated baseline):

| feature | symbol | value |
|---|---|---|
| bore (inner) diameter | Di | 160 um |
| outer (capillary) diameter | Do | 260 um |
| capillary wall thickness | (Do-Di)/2 | **50 um** |
| nozzle / capillary length | Lnoz | 300 um |
| nozzle exit -> collector | H (collectorDistance) | 1.5 mm |
| collector plate diameter | D_c | 5 mm |

Layout: the capillary tube (bore r<Di/2, wall Di/2<r<Do/2) sits at the **bottom**; liquid feeds up
the bore; the cone-jet forms at the nozzle **exit** (top of the capillary) and travels up to the
**collector** plate H above. Atmosphere surrounds the capillary and fills the gap to the collector.

**Defect geometries (vary the mesh only):**
- **D1 blunting** — round/erode the capillary tip rim with a fillet radius r_b (0 = sharp square rim;
  up to (Do-Di)/2 = 25 um = fully rounded rim). AO erosion rounds the field-concentrating edge.
- **D2 asymmetry** — tilt the capillary axis (e.g. 2-10 deg) or laterally offset the tip -> off-axis
  cone -> plume steering / thrust-vector error.
- **D3 roughness** — one or more micro-protrusions / pits on the tip rim -> local field spike.

A ready-made parametric generator (structured hex, no OpenFOAM needed) is
`apps/generate_resolved_nozzle_mesh.py`: `GRADED=1 TIP_DX_UM=12 BLUNT_UM=.. TIP_DEG=.. BUMP_UM=..
python3 ... <outdir>`. You can instead build the mesh in any tool (gmsh, blockMesh+snappyHexMesh,
cfMesh) and export an OpenFOAM polyMesh — just follow sections 2-4.

## 2. Mesh requirements (read this — defects are resolution-critical)

- **Named boundary patches** (the solver maps by case-insensitive name substring):
  - `liquid_inlet` (or contains "inlet"/"feed") -> the bore inlet face at the bottom of the bore.
  - `nozzle_wall` (or "nozzle"/"electrode"/"capillary"/"needle"/"wall") -> the whole capillary wall
    surface (no-slip + electrode).
  - `collector` (or "ground"/"plate"/"target") -> the collector plate.
  - `outlet` (or "atm"/"far"/"open"/"ambient") -> the open atmospheric boundary (domain sides + far).
  - optional `symmetry`/`wedge`/`axis` for a reduced (wedge/quarter) model.
- **Units: metres.** The solver auto-normalizes (scales by 1/Do, moves the `liquid_inlet` plane to
  y=0, centres the axis at x=z=0.5*radialWindowOuterDiameters). You do NOT pre-normalize.
- **Orientation:** y is the axis, inlet (y=0) -> collector (y=H); x,z radial.
- **RESOLUTION (the methodological core):** the defect lives in the tip geometry, and E ~ V/r_tip, so
  a defect is **unrepresentable** below ~2-3 cells across the 50 um wall. Resolve the **wall + tip
  with >= 3-4 cells** (tip cell size <= ~12-15 um); we verified blunting erodes 0 cells at ~1-cell
  wall (NX=24) but appears at >=2-3 cells. **Grade** the mesh: fine near the tip/cone (r < ~1.5 Do,
  y up to ~2-3 Lnoz), coarse in the far plume. Domain radius >= ~2-3 Do; height = H.
- **OpenFOAM polyMesh format:** points / faces / owner / neighbour / boundary. **Face ordering:
  internal faces first, then boundary faces grouped contiguously per patch**; internal faces have
  `owner < neighbour`; the `boundary` file gives each patch `nFaces` + `startFace`.

## 3. Operating conditions (fixed; set via the case, not the mesh)

Baseline = Candido validation. Fluid = ionic-liquid-like; all in `CandidoTaylorConeJetSetup`:

| property | value | | property | value |
|---|---|---|---|---|
| voltage U0 | 2180 V (= CaE 0.25) | | liquid density | 1208.4 kg/m^3 |
| flow rate Q | 16.1 nl/s | | gas density | 1.225 kg/m^3 |
| collector speed | 20 mm/s | | liquid viscosity | 60 mPa.s |
| contact angle | 51 deg | | gas viscosity | 0.012 mPa.s |
| surface tension gamma | 64.5 mN/m | | liquid rel. permittivity | 55.6 |
| liquid conductivity | 60 uS/m | | gas rel. permittivity | 1.0 |
| gas conductivity | 1e-15 S/m | | | |

Derived (the solver computes these): field scale e0 = U / (Do * ln(4H/Do)); electric capillary number
CaE = e0^2 * eps0 * Do / gamma (so `target_ca_e=0.25` <-> U ~ 2.18 kV); inlet velocity
u_in = Q / (pi*Di^2/4); hydrodynamic time tau_h = sqrt(rho_liq * Di^3 / gamma); electric relaxation
time tau_e = eps/sigma (sets the timestep, dt <= 0.1*tau_e, plus the adaptive electric-force CFL).

For the sweep: keep ALL of the above identical; change only the mesh defect. (To probe the stability
margin, optionally also run each at CaE 0.42.)

## 4. Boundary conditions (what the solver applies per named patch)

zeroGradient = homogeneous Neumann (the default for any field not listed).

| patch | velocity u | pressure p | alpha (VOF) | potential phi | charge rho_e |
|---|---|---|---|---|---|
| **liquid_inlet** | parabolic fully-developed, mean = u_in | zeroGradient | Dirichlet alpha = 1 (liquid) | Dirichlet phi = U | Dirichlet rho_e = 0 (neutral) |
| **nozzle_wall** | no-slip u = 0 | zeroGradient | zeroGradient | Dirichlet phi = U (electrode) | zeroGradient |
| **collector** | moving wall u_x = -20 mm/s | zeroGradient | contact angle 51 deg | Dirichlet phi = 0 (ground) | zeroGradient |
| **outlet** | zeroGradient (mixed in/out) | total pressure p0 = 0 (open-flux now; explicit Dirichlet is a follow-up) | zeroGradient | Neumann / far | Dirichlet rho_e=0 inflow, zeroGradient out |

Enabled by `use_named_patch_boundary_conditions=true`; the per-role values are
`paperDefaultPatchBc()` (Candido & Pascoa 2023 Sec II.D). Without the flag the solver falls back to
box-geometric classification (wrong for a nozzle mesh) - so always set it true for resolved meshes.

## 5. Initial conditions (set automatically at t=0 - you do not supply these)

- **alpha (VOF):** liquid fills the bore (r < Di/2) up to inletLength = min(Lnoz/Do, 0.8); a
  spherical-cap meniscus caps the nozzle exit; gas elsewhere. Interface smoothed over 0.22*Do (tanh).
  A small sinusoidal perturbation (amplitude ~0.002-0.008, sin(3*theta)) seeds the 3D instability.
- **velocity:** parabolic inlet profile u_y = 2*uScale*(1-(r/ri)^2) in the inlet; zero elsewhere.
- **phi = 0, rho_e = 0, p = 0** at t=0 (the field/charge build up over the first steps).
- mixture density rho = alpha*rho_liq + (1-alpha)*rho_gas; eps, sigma by harmonic (WHM) mean.

## 6. Run a case

```json
{ "case_name": "sharp", "run_mode": "candido_smoke",
  "mesh_mode": "openfoam_polyMesh",
  "openfoam_polyMesh": "/abs/path/to/case/constant/polyMesh",
  "use_named_patch_boundary_conditions": true,
  "target_ca_e": 0.25,
  "steps": 4000 }
```
`./build/electrospray_case_runner --case sharp.json --output-dir out_sharp`. Outputs `summary.json`
(final metrics) + `history.csv` (per-step time series: mass, tip_y, radial_asymmetry, max_velocity,
max_electric_force, currents, ...). The adaptive electric-force CFL keeps the (defect) tip stable;
isoAdvector keeps the cone-jet/droplet interface sharp.

## 7. Checklist for each defect run

1. Build the mesh in metres with the 4 named patches, wall/tip resolved >= 3-4 cells, graded.
2. Same operating point as the baseline (`target_ca_e=0.25`, default fluid/U/Q).
3. `use_named_patch_boundary_conditions=true`.
4. Run; confirm `status=pass`, mass/charge conservation (~1e-13/1e-14), then compare the observables
   (plume divergence, thrust-vector, droplet/q-m, asymmetry) vs the sharp baseline.
