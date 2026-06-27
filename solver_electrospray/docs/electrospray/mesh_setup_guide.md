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

## 6b. Detailed mesh construction (how to actually build it)

### A. What is fluid vs void (the topology)
The capillary is a TUBE standing in an atmosphere box. The FLUID domain you mesh is everything the
fluid occupies; the **capillary WALL is a void** (not meshed) whose surface is the `nozzle_wall`
boundary. Three fluid regions, one void:

```
 y=H  ______________________ collector  (Dirichlet phi=0, moving wall)
     |                      |
     |   atmosphere (gas)   |   <- coarse;  side faces = outlet
     |        |jet|         |
 y=Lnoz ......\cone/........  nozzle exit  (cone-jet forms here)
     |  ||wall||  ||wall||  |   wall void = annulus Di/2<r<Do/2, 0<y<Lnoz
     |  ||    ||bore||    || |   bore (r<Di/2) = liquid feed; atmosphere at r>Do/2
 y=0 |__||____||____||____||_|  liquid_inlet = bore bottom face (r<Di/2)
        Di/2  Do/2
```
- **bore** (r < Di/2): liquid feed channel; continues upward as the central jet column.
- **atmosphere** (r > Do/2 for y<Lnoz, and all r for Lnoz<y<H): gas; the cone-jet/plume lives here.
- **capillary wall VOID** (Di/2 < r < Do/2, 0<y<Lnoz): NOT meshed; its faces are `nozzle_wall`.

### B. The four patch surfaces (assign precisely)
- `liquid_inlet`: the bore **bottom** face only — the disc r < Di/2 at y = 0.
- `nozzle_wall`: ALL capillary surfaces — the inner bore wall (r=Di/2), the outer wall (r=Do/2), the
  **tip rim** at the top (the annulus Di/2..Do/2 at y=Lnoz, or its rounded fillet for D1), and the
  bottom annulus (Di/2..Do/2 at y=0). This is the no-slip electrode.
- `collector`: the top plane y = H (disc of radius >= Dc/2 = 2.5 mm, or the whole top face).
- `outlet`: the radial outer boundary (domain side) + any far-field opening.

### C. Resolution map (the part that decides whether the paper is correct)
| region | target cell size | why |
|---|---|---|
| **tip / cone / rim** (r<~1.5 Di, Lnoz-50um < y < Lnoz+~2 Lnoz) | **8-12 um** (wall >= 4 cells) | E~V/r_tip; the defect + the Taylor-cone apex live here. BELOW ~2-3 wall cells the defect is invisible. |
| bore + near-meniscus | 12-20 um (Di across >= 8 cells) | resolve the feed + meniscus |
| near plume (up to ~3-5 Lnoz) | 20-40 um, smooth grade | the thin jet is near-axis |
| far plume + collector approach | 50-100 um | spreading, low gradients |
Grade smoothly (expansion ratio < 1.3). Domain radius >= 2-3 Do; a full-3D mesh of ~150k-500k cells
is typical for a converged 3D run (our 89k smoke mesh is a minimal proof-of-pipeline, not converged).

### D. Recommended build workflows (pick one)
1. **STL + snappyHexMesh (recommended for the real study).** Make the capillary as a parametric CAD
   solid (bore + wall + the defect), export an STL of its surface; `blockMesh` a background box
   (y in [0,H], radius >= 2-3 Do); `snappyHexMesh` to carve the fluid around the capillary with a
   refinement region/surface at the tip (refinement level set so tip cells ~10 um). Name the snapped
   patches `nozzle_wall`/`collector`/`outlet` and the bore-bottom `liquid_inlet`. Hex-dominant,
   handles the defect geometry + local refinement cleanly.
2. **gmsh.** Build the geometry (bore, wall void, atmosphere) as an OpenCASCADE model; use a Box/Ball
   size field to refine near the tip; mesh hex-dominant or tets; `gmshToFoam`; rename patches.
3. **Extend the bundled generator.** `apps/generate_resolved_nozzle_mesh.py` already emits a graded
   structured-hex polyMesh with the four patches and the D1/D2/D3 env-var defects, no OpenFOAM needed
   - good for quick/structured cases; less flexible than (1) for arbitrary defect shapes.

### E. Per-defect mesh construction (each defect changes the mesh differently)
Keep the bore, Do, Lnoz, H, and ALL operating conditions identical across the sweep - vary ONLY the
tip-defect feature. The three defects have DIFFERENT topology/symmetry/resolution needs:

**D1 - blunting (rounded tip rim).** *Axisymmetric.* In the CAD, round the capillary tip edge with a
fillet radius r_b (0 = sharp square rim; up to (Do-Di)/2 = 25 um = fully rounded hemispherical rim;
beyond that = tip recession). Sweep e.g. r_b = {0, 8, 15, 20, 25} um (you may instead express it as a
tip half-angle to compare to ACS 2024). *Mesh:* the fillet is the field-relevant feature, so the mesh
must FOLLOW the rounded surface - need ~4-8 cells across the fillet arc (tip cells ~3-6 um for a clean
fillet; ~10 um is a coarse minimum). snappyHexMesh snaps to the STL fillet; the structured generator
stair-steps it (use fine tip cells). Because it is axisymmetric you *could* use a wedge for D1 alone,
but the solver's 3D sin(3*theta) seed + consistency with D2/D3 mean: run full 3D.

**D2 - tilt / asymmetry (off-axis tip).** *Breaks axisymmetry -> MUST be full 3D (no wedge).* In the
CAD, rotate the whole capillary solid by the tilt angle (sweep {0, 2, 5, 10} deg) about a point on
its axis - the bore and wall lean together - or translate the tip laterally. *Mesh:* the fine tip
zone must be centred on the TILTED tip location, not the domain axis. Make the domain WIDE enough for
the steered plume: a tilt theta steers the impact by ~H*tan(theta) at the collector (8 deg over
1.5 mm ~ 210 um off-axis), so domain radius >= that + margin. Resolution at the (tilted) tip = same as
the baseline.

**D3 - protrusion / roughness (local bump).** *Breaks axisymmetry -> full 3D, and the hardest to
mesh.* In the CAD, add a small boss (or notch/pit) on the tip rim at one azimuthal location, size
~5-20 um (sweep height {0, 5, 10, 20} um, or several bumps for distributed roughness). *Mesh:* the
bump is much smaller than the general tip cell, so it needs **LOCAL refinement at the bump (~2-5 um
cells)** to resolve the feature and its field spike - finer than the ~10 um general tip. snappyHexMesh
with a refinement surface/region around the bump is the clean way; a uniform tip mesh would have to be
~2-5 um everywhere (expensive).

See `docs/electrospray/figures/tip_defect_geometries.png` for the four tip cross-sections.

### F. Quality + sanity
- Hex-dominant, max non-orthogonality < ~65 deg, max skewness < ~4 near the tip (the pressure/Rhie-
  Chow + least-squares gradients want clean cells there).
- Confirm the four patch names resolve (the run prints `inlet_from_patch=1` and a nonzero
  `external_inlet_faces`).
- First always run a short smoke (steps ~30) and check `status=pass` + mass drift ~1e-14 before the
  long run.

## 7. Checklist for each defect run

1. Build the mesh in metres with the 4 named patches, wall/tip resolved >= 3-4 cells, graded.
2. Same operating point as the baseline (`target_ca_e=0.25`, default fluid/U/Q).
3. `use_named_patch_boundary_conditions=true`.
4. Run; confirm `status=pass`, mass/charge conservation (~1e-13/1e-14), then compare the observables
   (plume divergence, thrust-vector, droplet/q-m, asymmetry) vs the sharp baseline.
