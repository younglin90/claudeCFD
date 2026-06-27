# Paper design — tip defects in miniaturized electrospray thrusters (computational)

Working title: *"How fabrication and on-orbit (atomic-oxygen) tip defects reshape the electrospray:
a resolved 3D leaky-dielectric VOF study of droplet and plume degradation in miniaturized
electrospray thrusters."*

## 1. Novelty / gap (literature-grounded)

The literature splits into three camps that each miss the target (confirmed by a web review):
1. **Idealized-geometry sweeps** — tip half-angle (ACS Appl. Electron. Mater. 2024: 20 deg vs 30 deg),
   apex radius / onset voltage, clean shape families (Malik 2025). These treat geometry as a *design*
   variable on a pristine, smooth, symmetric emitter — never as a realistic manufacturing flaw.
2. **Degradation / lifetime studies** — document emitter corrosion, oxidation, byproduct growth and
   curvature increase experimentally/analytically (MDPI Aerospace 2020; Si-emitter lifetime work) but
   never feed a specific defect morphology into a resolved EHD/VOF solver.
3. **Atomic-oxygen (AO) erosion** — quantified for flat spacecraft surfaces and AO sensors, but never
   coupled to the electrostatic/emission performance of a sharp emitter.

**Our contribution:** the first resolved 3D leaky-dielectric VOF EHD simulation of *realistic
fabrication- and AO-oxidation-induced tip defects* -> quantitative droplet-size, plume-divergence and
thrust-vector changes -> fabrication-tolerance limits and a LEO lifetime/degradation map. We bridge
camps 1-3: take the real defect morphologies of camp 2/3 and run them through a faithful cone-jet
solver (validated against Candido & Pascoa 2023).

## 2. Tip-defect taxonomy (independent variables) — focus D1, D2, D3 (geometric)

| # | Defect | Origin | Mesh/solver representation | Primary physics |
|---|---|---|---|---|
| **D1** | tip blunting / apex radius increase | AO erosion (non-protective metals W/Ag), wear | rounded capillary tip, r_tip up | E ~ V/r_tip down; onset V_on ~ sqrt(r_tip); **non-monotonic divergence (see below)** |
| **D2** | asymmetry / off-axis tip | fabrication misalignment, asymmetric AO erosion | tip offset / tilt | off-axis Taylor cone -> **plume steering + thrust-vector error** |
| **D3** | roughness / micro-protrusion / pitting | etch burrs, AO pitting | micro-bump(s) on the tip | multiple emission sites, satellite droplets, local field spikes |

**SCOPE — geometry (mesh) defects only.** D1/D2/D3 are pure *mesh geometry* changes. The material/
property defects — D4 dielectric oxide layer (surface sigma/eps) and D6 wettability (contact angle) —
are **out of scope for this paper** and deferred. This sharpens the contribution to a clean
*tip-geometry -> electrospray* study, and makes **mesh-resolution accuracy the methodological
centerpiece**: because E~V/r_tip and the Maxwell force ~E^2, a faithful result demands the defect
geometry be meshed precisely and the (near-singular) tip integrated stably — exactly what the
resolved-nozzle generator + adaptive electric-force CFL + volume-matched isoAdvector provide. AO ->
geometry mapping: non-protective/eroding emitter metals blunt and roughen the tip (D1/D3); asymmetric
erosion/misalignment tilts it (D2).

**Key physics to capture + differentiate (ACS 2024):** tip sharpness affects plume divergence
**non-monotonically** — a *too-sharp* emitter (~20 deg half-angle) emits off-axis into a toroidal/
annular plume (angular efficiency collapses to ~55%), while moderate blunting (~30 deg) restores a
near-Gaussian on-axis beam. So D1 blunting is **not** simply "worse" — there is an optimum. ACS 2024
showed this experimentally for the tip *angle*; we show the mechanism computationally (the leaky-
dielectric cone-jet/meniscus response) and extend it to realistic blunting + asymmetry + roughness.

## 3. Conditions matrix

**Baseline (healthy tip)** = Candido & Pascoa 2023 validation point, on the resolved nozzle:
Di=160 um, Do=260 um, Lnoz=300 um, H=1.5 mm, U0=2.18 kV, Q=16.1 nl/s, **CaE=0.25**, contact angle 51 deg.

**Defect sweeps** (operating point fixed -> isolate the defect):
- D1 r_tip = {sharp(ref), 1.5x, 2x, 3x, 5x}; map to AO fluence (years in LEO) via erosion depth.
- D2 tilt = {0, 2, 5, 10 deg} or axis offset {0, 0.1, 0.2 Do}.
- D3 protrusion height/position on the tip rim.
Run each at **CaE 0.25 (stable)** and **0.42 (toward unstable/whipping)** to show how a defect erodes
the stability margin.

**AO -> defect anchors (literature):** AO flux at 400 km ~5.2e13 atoms/cm2/s; annual ram fluence
~5e20-3.5e21 atoms/cm2/yr (solar min->max); a multi-year LEO mission => 1e22-1e23 atoms/cm2 =>
erosion depth (via the material erosion yield) sets the blunting/recession magnitude per defect level.

## 4. Observables (dependent variables = thruster performance)

NOTE (literature): cone-jet **current is geometry-independent**, I ~ f(eps)*sqrt(gamma*K*Q/eps)
(Fernandez de la Mora 1994). So defects mainly change **onset voltage, divergence, emission mode, and
stability** — NOT the base current law. Frame accordingly:
- **plume divergence half-angle** (95% current) + **beam efficiency eta_theta ~ cos^2(theta)**.
- **thrust-vector deviation angle** (D2 headline).
- **droplet size / specific charge q/m** (q/m ~ sqrt(gamma*K/Q)); satellite-droplet fraction (D3).
- **emission mode / regime shift** (cone-jet -> intermittent/multi-jet as eta_min ~ 0.5-0.59 is
  approached).
- **cone-jet morphology** (silhouette, tip displacement) and **whipping/off-axis asymmetry** (stability).

Headline figures: defect-strength axis vs each metric (sensitivity curves); a **degradation map**
(which defect pushes the emission into which regime); thrust-vector error vs asymmetry; the
non-monotonic divergence-vs-blunting curve with the optimum.

## 5. Mesh strategy (why accuracy is decisive)

Because E ~ V/r_tip and the Maxwell force ~ E^2, the result is **acutely sensitive to the tip
geometry** — the same sensitivity that produced the cone-tip blow-up we fixed with the adaptive
electric-force CFL. So each defect must be **resolved precisely** in the mesh:
- extend `apps/generate_resolved_nozzle_mesh.py` to be *defect-parametric* (D1 rounded tip radius,
  D2 tip offset/tilt, D3 micro-bump), with **local tip refinement** (bore 6-8 cells; ideally graded
  fine-near-tip / coarse-in-plume to afford the plume region).
- the adaptive electric-force CFL keeps the fine/defect tip stable; the volume-matched isoAdvector
  keeps the droplet/plume interface sharp.

## 6. Solver readiness

Ready: validated 3D leaky-dielectric VOF EHD (Experiments A-F); resolved-nozzle + named-patch BCs
(end-to-end verified); adaptive electric-force CFL; genuine isoAdvector. To add: defect-parametric +
graded mesh generation; total-pressure outlet (for the open plume); finer mesh + longer runs for a
converged plume and droplet-break-up statistics (current solver resolves up to the jet; satellite/
break-up statistics need finer resolution).

## 7. Key references

- Saville 1997 (Annu. Rev. Fluid Mech.) — Taylor-Melcher leaky-dielectric model (governing equations).
- Herrada et al. 2012 (Phys. Rev. E 86, 026305) — canonical LDM cone-jet simulation.
- Gamero-Castano & Magnani (JFM 2018; 2025) — universal LDM cone-jet solutions; current/jet-radius.
- Fernandez de la Mora & Loscertales 1994 (JFM) — cone-jet current law I=f(eps)sqrt(gamma*K*Q/eps).
- Candido & Pascoa 2023 (Phys. Fluids 35, 052110) — our 3D VOF baseline / validation.
- ACS Appl. Electron. Mater. 2024 (tip angle 20 vs 30 deg) — closest prior work to differentiate from.
- Whittaker et al. (AIAA JPP, B39524) — emitter fabrication scatter: tip radius 25.8 +/- 20.9 um,
  heavy-tailed-blunt (motivates the D1 range + array-uniformity angle).
- Ziemer et al. IEPC-2009-242 — emitter geometries, eta_theta=cos^2(theta), divergence/efficiency.
- MIT OCW 16.522 (Lozano/Martinez-Sanchez) — scaling laws (r*, I, q/m, Rayleigh, eta_min).
- NASA Banks 2004; de Rooij (SPENVIS) — AO flux/fluence/erosion-yield in LEO.
- MDPI Aerospace 2020 — emitter degradation/lifetime (experimental).

## 8. Progress

- Healthy-tip baseline: resolved-nozzle NX=30 mesh (26832 cells, bore ~5 cells) runs stable +
  charge/mass-conserving with the named-patch BCs (mass drift 3.0e-14) under CaE 0.25; a long run to
  develop the cone-jet is in progress. Next: converge the baseline, then make the mesh generator
  defect-parametric (D1 first).

## 9. Case matrix (11 core cases; geometry differences)

Shared baseline (sharp = D1 r_b=0 = D2 tilt=0 = D3 bump=0). All cases: same Di/Do/Lnoz/H + operating
point (CaE 0.25); only the tip geometry differs. See figures/all_cases_geometry.png.

| # | case | defect | param | geometry vs sharp | symmetry / mesh |
|---|---|---|---|---|---|
| C0 | sharp | - | - | Candido capillary, square tip rim, on-axis (shared baseline) | axisymmetric |
| C1 | D1-08 | blunting | r_b=8um | tip rim fillet 8um | axisymmetric |
| C2 | D1-15 | blunting | r_b=15um | tip rim fillet 15um | axisymmetric |
| C3 | D1-20 | blunting | r_b=20um | tip rim fillet 20um | axisymmetric |
| C4 | D1-25 | blunting | r_b=25um | tip rim fully rounded (=(Do-Di)/2) | axisymmetric |
| C5 | D2-02 | tilt | 2 deg | capillary axis tilted 2 deg | 3D |
| C6 | D2-05 | tilt | 5 deg | capillary axis tilted 5 deg | 3D |
| C7 | D2-10 | tilt | 10 deg | capillary axis tilted 10 deg | 3D, wider domain |
| C8 | D3-05 | bump | h=5um | one-sided rim protrusion 5um | 3D + local refine |
| C9 | D3-10 | bump | h=10um | one-sided rim protrusion 10um | 3D + local refine |
| C10 | D3-20 | bump | h=20um | one-sided rim protrusion 20um | 3D + local refine |

= 1 baseline + 4 (D1) + 3 (D2) + 3 (D3) = **11 core cases** at CaE 0.25.

Optional stability-margin set at CaE 0.42 (baseline + strongest of each defect): C0', C4', C7', C10'
= **+4 cases** -> 15 total. Mesh reuse: D1 reuses the baseline mesh (only the fillet changes,
axisymmetric); D2/D3 need new 3D meshes (D3 also needs local refinement at the bump). Recommended
order: D1 sweep first (cleanest, mesh reuse, the ACS-2024 non-monotonic-divergence comparison), then
D2 (thrust vector), then D3 (satellite/local-spike).
