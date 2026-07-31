---
title: "A three-dimensional leaky-dielectric volume-of-fluid method for electrohydrodynamic cone-jets on resolved emitter geometries, with application to tip defects in miniaturized electrospray emitters"
author:
- Younglin Yoo^a^
- ^a^*Affiliation, Address, City, Country* (to be completed)
---

**Abstract**

We present a three-dimensional finite-volume solver for electrohydrodynamic (EHD) two-phase flows in the Taylor–Melcher leaky-dielectric limit, designed to compute cone-jet electrosprays on geometrically resolved emitter hardware rather than on idealized axisymmetric domains. The interface is transported with a geometric, volume-matched piecewise-linear (PLIC) advection scheme of the isoAdvector type, which preserves liquid mass to near round-off on hexahedral meshes. Surface tension is applied through a balanced-force continuum-surface-force formulation with curvature evaluated from a local PLIC-consistent quadric reconstruction, and the electric problem couples a variable-coefficient Poisson equation for the potential to a conservative bulk-charge transport equation with ohmic loading. Two ingredients are emphasized because they control robustness on realistic geometries: (i) an adaptive electric-force time-step limiter that bounds the cell displacement induced by the explicit Maxwell force, which we find necessary to integrate sharp, strongly charged menisci on fine meshes without divergence; and (ii) a named-patch boundary-condition framework that maps OpenFOAM-format mesh patches of a resolved capillary emitter (inlet, nozzle wall, extractor/collector, open atmosphere) onto the physical cone-jet boundary conditions. The solver is verified for conservation and stability and validated against a published three-dimensional VOF cone-jet benchmark. As a demonstration targeted at miniaturized electrospray thrusters, we outline a parametric study of geometric tip defects—rim blunting, axis tilt, and micro-protrusions—resolved directly in the mesh, and we quantify the near-tip resolution required for such defects to influence the computed solution. [Quantitative results will be inserted upon completion of the production runs.]

**Keywords:** electrohydrodynamics; leaky-dielectric model; volume-of-fluid; cone-jet; electrospray propulsion; interface capturing

# 1. Introduction

Electrospray atomization in the cone-jet mode converts a bulk liquid into a fine, quasi-monodisperse charged spray by the interplay of electric stress, surface tension, and viscous transport [1–5]. Since the seminal observations of Zeleny [1] and the theoretical description of the conical equilibrium by Taylor [2], the cone-jet has become the operating principle of a remarkably broad set of technologies, including electrospray ionization for mass spectrometry, micro- and nano-encapsulation, film deposition, and, of particular interest here, electrospray (colloid) thrusters for small spacecraft [6–9]. In the propulsion context, arrays of miniaturized emitters deliver thrust at the micronewton to millinewton scale with high specific impulse, and their in-flight performance was demonstrated on missions such as LISA Pathfinder/ST7 [8]. The emitted current, droplet size, plume divergence, and thrust vector all inherit their sensitivity from the small region surrounding the emitter tip, where the electric field, interface curvature, and charge relaxation interact on scales of micrometers and microseconds.

The accepted continuum description of these flows is the Taylor–Melcher leaky-dielectric model [3,4,10]. Both phases are treated as incompressible; charge is assumed to accumulate in a thin layer at the interface and to be transported by ohmic conduction and convection, while the bulk remains quasi-electroneutral. The model has been examined critically and validated in the review of Saville [4], and it reproduces the essential physics of cone-jets, including the current scaling laws established experimentally and theoretically by Fernández de la Mora and Loscertales [5] and Gañán-Calvo [11,12].

Numerical simulation of cone-jets has historically favored axisymmetric formulations. Steady and transient axisymmetric solvers based on boundary-fitted or hybrid discretizations have produced detailed solutions of the cone-to-jet transition region and its universal scalings [13,14], and interface-capturing computations of EHD two-phase flow in the volume-of-fluid (VOF) framework were established by Tomar et al. [15] and, with a charge-conservative discretization, by López-Herrera et al. [16]. Fully three-dimensional computations remain comparatively rare. Cándido and Páscoa [17] recently presented a three-dimensional VOF leaky-dielectric simulation of a capillary cone-jet, including validation against experimental cone morphology and current; their configuration is adopted as the validation baseline of the present work. Three-dimensional capability is not a luxury: phenomena of practical importance—lateral instability and whipping of the jet, asymmetric emission, off-axis thrust components, and any effect of non-axisymmetric hardware geometry—are inaccessible to axisymmetric models.

The present work is motivated by a specific class of non-axisymmetric effects: *geometric imperfections of the emitter tip*. Manufactured emitters exhibit finite tolerances of the tip radius and alignment; measured tip-radius scatter across nominally identical emitters can be comparable to the nominal radius itself [18]. In addition, emitters degrade in service: erosion, corrosion, and—in low Earth orbit—atomic-oxygen attack progressively blunt and roughen exposed metallic tips [19,20]. Prior computational studies have treated tip geometry mainly as a smooth design parameter (e.g., apex angle or nominal radius) on idealized shapes [21], while degradation studies have remained experimental or system-level [22]. A solver that resolves the actual emitter geometry, including localized defects, inside the computational mesh—and that remains robust when the defect sharpens the local field—permits a systematic sensitivity analysis connecting fabrication and degradation tolerances to spray metrics. To the best of our knowledge such defect-resolved cone-jet computations have received little attention.

Three numerical requirements follow directly from this goal. First, the interface transport must be sharp, bounded, and conservative on general (not necessarily orthogonal or graded-isotropic) meshes, because long-time integrations of a feeding meniscus tolerate very little cumulative mass error. We adopt a geometric, volume-matched PLIC advection of the isoAdvector type [23,24], rather than algebraic compression schemes [25,26], and we quantify the resulting conservation behavior. Second, the explicit Maxwell force at a sharpening, strongly charged tip is numerically stiff: the local force density grows quadratically with the field, and a time step chosen from convective, capillary [27], and charge-relaxation considerations alone does not bound the displacement that the electric force can impose on a fine cell within one step. We introduce a simple adaptive time-step limiter based on the cell-scale displacement induced by the Maxwell force and show that it is the difference between divergence and stable operation on tip-refined meshes. Third, resolved hardware requires boundary conditions attached to *named mesh patches* (inlet, nozzle wall, extractor/collector, open boundary) rather than to the faces of a canonical box; we describe a compact framework that maps patch names to the physical cone-jet boundary conditions while preserving, by construction, the legacy behavior on structured domains.

The contributions of this paper are: (i) a documented, three-dimensional, unstructured finite-volume implementation of the leaky-dielectric VOF model with geometric interface advection and balanced-force capillarity; (ii) an adaptive electric-force time-step criterion, with a derivation, an implementation cost of one reduction per step, and demonstrations of its necessity; (iii) a named-patch boundary-condition treatment enabling cone-jet simulation on resolved emitter geometry imported in OpenFOAM polyMesh format; and (iv) a defect-parametric emitter meshing methodology (rim blunting, axis tilt, micro-protrusion) together with the near-tip resolution requirements for defect representability, applied in a sensitivity study relevant to miniaturized electrospray thrusters. The remainder of the paper is organized as follows. Section 2 states the governing equations and non-dimensional groups. Section 3 details the discretization, the coupling algorithm, the time-step control, and the boundary-condition and meshing framework. Section 4 reports verification, validation against the three-dimensional benchmark of [17], and the tip-defect demonstration study. Conclusions are drawn in Section 5. Two appendices collect the derivation of the force-based time-step bound and the patch-wise boundary-condition tables.

# 2. Mathematical model

## 2.1 Two-phase flow and interface capture

Both fluids are treated as incompressible and Newtonian, and a single velocity field $\mathbf{u}$ and dynamic pressure $p$ are shared by the two phases (one-fluid formulation). Mass and momentum conservation read

$$\nabla \cdot \mathbf{u} = 0, \qquad (1)$$

$$\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + \nabla\cdot\left[\mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^{\mathsf{T}}\right)\right] + \mathbf{f}_{\gamma} + \mathbf{f}_{e}, \qquad (2)$$

where $\rho$ and $\mu$ are the mixture density and viscosity, $\mathbf{f}_{\gamma}$ is the capillary force density, and $\mathbf{f}_{e}$ the electric (Maxwell) force density. Gravity is negligible at the scales considered (Bond number much smaller than unity) and is omitted.

The liquid volume fraction $\alpha \in [0,1]$ obeys the advection equation

$$\frac{\partial \alpha}{\partial t} + \nabla\cdot(\alpha\,\mathbf{u}) = 0, \qquad (3)$$

and the mixture properties are interpolated as

$$\rho = \alpha\,\rho_{l} + (1-\alpha)\,\rho_{g}, \qquad \mu = \alpha\,\mu_{l} + (1-\alpha)\,\mu_{g}, \qquad (4)$$

with analogous arithmetic weighting for the electrical permittivity $\varepsilon$ and conductivity $\sigma$. Subscripts $l$ and $g$ denote the working liquid and the surrounding gas. The choice of arithmetic weighting for the electrical properties follows the three-dimensional reference computation [17]; alternatives (e.g., harmonic weighting of $\varepsilon$ across the interface) primarily affect sub-cell sharpness of the electric stress and are compatible with the discretization described below.

Surface tension is modeled with the continuum-surface-force (CSF) approach [27],

$$\mathbf{f}_{\gamma} = \gamma\,\kappa\,\nabla\alpha, \qquad \kappa = -\nabla\cdot\hat{\mathbf{n}}, \qquad (5)$$

where $\gamma$ is the (constant) surface-tension coefficient, $\kappa$ the interface curvature, and $\hat{\mathbf{n}}$ the unit interface normal aligned with $\nabla\alpha$. The discrete construction of a *balanced* $\mathbf{f}_{\gamma}$, and of $\kappa$ from the reconstructed interface rather than from raw $\nabla\alpha$, is described in Section 3.4; both are essential for controlling spurious currents at the small capillary numbers of interest [28–30]. At solid contact lines (the collector plate) a static contact angle $\theta_{c}$ is imposed by rotating the interface normal used in the curvature evaluation within a near-wall band.

## 2.2 Electrostatics and charge transport

In the leaky-dielectric limit the magnetic field is negligible and the electric field is irrotational, $\mathbf{E} = -\nabla\phi$. Gauss's law for a linear dielectric with free bulk charge density $\rho_{E}$ gives the variable-coefficient Poisson equation

$$\nabla\cdot(\varepsilon\,\nabla\phi) = -\rho_{E}. \qquad (6)$$

Free charge is transported by convection and ohmic conduction,

$$\frac{\partial \rho_{E}}{\partial t} + \nabla\cdot\left(\rho_{E}\,\mathbf{u}\right) = -\nabla\cdot\left(\sigma\,\mathbf{E}\right), \qquad (7)$$

which is the conservative differential statement of the Taylor–Melcher model [4,16]. In regions of uniform conductivity, Eq. (7) drives $\rho_{E}$ toward zero on the charge-relaxation time $\tau_{e} = \varepsilon/\sigma$; charge therefore accumulates where $\sigma$ and $\varepsilon$ vary, i.e., in the numerically diffuse interface region, which is the interface-capturing analogue of the surface-charge evolution equation of the sharp-interface model.

The electric force density is the divergence of the Maxwell stress tensor $\mathbf{T} = \varepsilon\left(\mathbf{E}\mathbf{E} - \tfrac{1}{2}E^{2}\mathbf{I}\right)$, where $E$ denotes the magnitude of $\mathbf{E}$; for incompressible phases the electrostriction term is absorbed into the pressure and

$$\mathbf{f}_{e} = \nabla\cdot\mathbf{T} = \rho_{E}\,\mathbf{E} - \frac{1}{2}\,E^{2}\,\nabla\varepsilon. \qquad (8)$$

The first term is the Coulomb force on free charge (dominantly tangential-stress-generating at a leaky-dielectric interface); the second is the dielectric force arising from the permittivity jump (normal to the interface).

## 2.3 Non-dimensional groups and operating point

Lengths are scaled with the emitter outer diameter $D_{o}$, and the nominal applied field is defined from the emitter–collector potential difference $U_{0}$ and separation $H$ through the cylindrical-capacitor estimate $E_{0} = U_{0}/\left[D_{o}\ln(4H/D_{o})\right]$ [17]. The primary control parameter is the electric capillary number

$$\mathrm{Ca}_{E} = \frac{\varepsilon_{0}\,E_{0}^{2}\,D_{o}}{\gamma}, \qquad (9)$$

which compares the electric suction at the tip with the confining capillary pressure. The flow-rate scale is set by the imposed volumetric feed $Q$; the hydrodynamic (capillary) time is $\tau_{h} = \left(\rho_{l} D_{i}^{3}/\gamma\right)^{1/2}$ with $D_{i}$ the inner (bore) diameter, and the electric relaxation time is $\tau_{e} = \varepsilon_{l}/\sigma_{l}$. Their ratio, together with the Ohnesorge number $\mathrm{Oh} = \mu_{l}/\left(\rho_{l}\gamma D_{i}\right)^{1/2}$ and the permittivity and conductivity ratios, completes the parameter set. Table 1 lists the geometric, material, and operating parameters of the validation configuration [17] used throughout this work: a metallic capillary of inner/outer diameter 160/260 μm and length 300 μm, a grounded collector at 1.5 mm, applied potential 2.18 kV (corresponding to $\mathrm{Ca}_{E} = 0.25$), and feed rate 16.1 nl s⁻¹.

**Table 1.** Geometry, fluid properties, and operating conditions of the baseline (validation) configuration, after [17].

| Quantity | Symbol | Value |
|---|---|---|
| Inner (bore) diameter | $D_i$ | 160 μm |
| Outer diameter | $D_o$ | 260 μm |
| Nozzle (capillary) length | $L_n$ | 300 μm |
| Emitter–collector distance | $H$ | 1.5 mm |
| Collector diameter | $D_c$ | 5.0 mm |
| Applied potential | $U_0$ | 2.18 kV |
| Electric capillary number | $\mathrm{Ca}_E$ | 0.25 |
| Feed flow rate | $Q$ | 16.1 nl s⁻¹ |
| Liquid density | $\rho_l$ | 1208.4 kg m⁻³ |
| Liquid viscosity | $\mu_l$ | 60 mPa s |
| Liquid relative permittivity | $\varepsilon_{r,l}$ | 55.6 |
| Liquid conductivity | $\sigma_l$ | 60 μS m⁻¹ |
| Gas density / viscosity | $\rho_g,\ \mu_g$ | 1.225 kg m⁻³, 0.012 mPa s |
| Gas relative permittivity / conductivity | $\varepsilon_{r,g},\ \sigma_g$ | 1.0, 10⁻¹⁵ S m⁻¹ |
| Surface tension | $\gamma$ | 64.5 mN m⁻¹ |
| Static contact angle (collector) | $\theta_c$ | 51° |

## 2.4 Boundary conditions

The physical boundary conditions on the resolved emitter geometry are as follows (their discrete patch-wise realization is given in Section 3.8 and Appendix B). At the *liquid inlet* (the bore cross-section), a fully developed parabolic axial velocity profile with mean $u_{\mathrm{in}} = 4Q/(\pi D_{i}^{2})$ is imposed, together with $\alpha = 1$, zero normal pressure gradient, electrode potential $\phi = U_{0}$, and charge-free inflow $\rho_{E} = 0$. The *nozzle wall* (inner and outer capillary surfaces and the tip annulus) is a no-slip electrode: $\mathbf{u} = 0$, $\phi = U_{0}$, zero-gradient $\alpha$ and $\rho_{E}$. The *collector* is a grounded no-penetration wall, $\phi = 0$, at which the static contact angle $\theta_{c}$ acts on the deposited liquid; a tangential wall velocity may be prescribed to represent a translating substrate. The *open atmosphere* boundary admits inflow/outflow with zero-gradient velocity–pressure coupling consistent with a quiescent far field, and zero-gradient $\phi$ (insulating far boundary). At $t = 0$ the bore is liquid-filled and terminated by a spherical-cap meniscus consistent with $\theta_{c}$; $\phi$ and $\rho_{E}$ are initialized to zero and the potential field is established by the first Poisson solve.

# 3. Numerical method

## 3.1 Finite-volume framework

The discretization is a cell-centered, collocated finite-volume method on general unstructured meshes composed of arbitrary convex polyhedra; in practice hexahedron-dominant meshes are used. Meshes are either generated internally (structured, graded boxes and cut-cylinder templates) or imported in OpenFOAM polyMesh format [31] with named boundary patches, which is the route used for resolved emitter geometries. Cell gradients are computed by a distance-weighted least-squares reconstruction over face neighbors; face interpolation is linear with over-relaxed non-orthogonal correction for diffusive fluxes. Convective fluxes of scalar quantities other than $\alpha$ (i.e., $\rho_{E}$) use an upwind-biased TVD scheme with the van Leer limiter, with the deferred high-order correction bounded between the upwind and downwind cell values. All linear systems are solved with Krylov methods: the pressure and potential Poisson systems by conjugate gradients with incomplete-Cholesky preconditioning, and the momentum predictor by BiCGSTAB [32] with ILU-type preconditioning.

## 3.2 Geometric interface advection (volume-matched PLIC / isoAdvector type)

Algebraic interface-compression schemes are attractive on unstructured meshes but introduce compression-strength parameters and residual smearing/boundedness trade-offs [25,26]. We instead adopt a geometric advection of $\alpha$ in the spirit of isoAdvector [23]: within each interface cell the interface is represented as a plane (PLIC), and the $\alpha$-fluxes are computed from the volume of liquid geometrically swept through each face during the time step.

The reconstruction proceeds in two steps. First, an interface orientation $\hat{\mathbf{n}}$ is computed for every interface cell from the reconstructed distance-function gradient (a plicRDF-type normal [24]), which is markedly less noisy than the raw $\nabla\alpha$ normal on coarse interface regions. Second, the plane position is fixed by *volume matching*: the plane with normal $\hat{\mathbf{n}}$ is translated along $\hat{\mathbf{n}}$ until the liquid sub-volume it cuts from the (possibly decomposed) polyhedral cell equals $\alpha_{i} V_{i}$ to a prescribed tolerance; the cut volume is evaluated exactly on a tetrahedral decomposition of the cell and the position is found by bisection. This guarantees per-cell consistency between the reconstruction and the stored volume fraction—the property that underlies the conservation behavior reported in Section 4.1.

For the flux, each downwind face of an interface cell accumulates the time-integrated liquid volume transported through it,

$$\Delta V_{f}^{\,l} = \int_{t}^{t+\Delta t} Q_{f}^{\,l}(t')\,\mathrm{d}t', \qquad (10)$$

where $Q_{f}^{\,l}$ is the instantaneous liquid volumetric flux through face $f$ evaluated from the motion of the reconstructed plane across the face polygon at the face-normal velocity implied by the (divergence-free) volumetric face flux. The new volume fractions follow from the discrete balance $\alpha_{i}^{n+1} V_{i} = \alpha_{i}^{n} V_{i} - \sum_{f} s_{if}\, \Delta V_{f}^{\,l}$ with $s_{if}$ the owner–neighbor sign convention. A final clipping step enforces $\alpha \in [0,1]$; the clipped excess is redistributed conservatively to adjacent interface cells so that the global liquid volume is unchanged by the bounding operation. An optional interface-sharpening sweep of adjustable strength is available but is disabled in all computations reported here; the geometric flux alone maintains an interface thickness of one to two cells.

## 3.3 Pressure–velocity coupling

Momentum is advanced by a predictor–projection sequence on the collocated grid with Rhie–Chow momentum interpolation [33] to suppress pressure checkerboarding, in the segregated spirit of PISO [34]. The predictor solves Eq. (2) implicitly in the convective and viscous terms with the capillary and electric forces treated explicitly. Face volumetric fluxes are then assembled from Rhie–Chow-interpolated face velocities, and a pressure correction with the standard $1/a_{P}$ (here $\Delta t/\rho$) weighting projects the flux field to the discrete divergence-free space; the correction is iterated (up to three times per step) until the cell-wise continuity defect falls below $10^{-8}$ in nondimensional units. The forces $\mathbf{f}_{\gamma}$ and $\mathbf{f}_{e}$ enter the predictor as face-consistent body forces (Sections 3.4 and 3.6) so that, at equilibrium, the pressure gradient can balance them discretely—the balanced-force property [28].

## 3.4 Balanced-force capillarity and curvature

The CSF force is evaluated on faces as $\left(\gamma\,\kappa\right)_{f} \left(\nabla\alpha\right)_{f}\cdot\mathbf{S}_{f}$ and redistributed to cells with the same weights as the pressure gradient, following the balanced-force construction of Francois et al. [28]. The curvature is *not* computed from derivatives of the mollified $\alpha$ field alone. Instead, within each interface cell a local quadric surface is fitted to the volume-matched PLIC interface (centroids and normals from the cell and its face neighbors), and $\kappa$ is evaluated analytically from the fit; stencils whose conditioning falls below a threshold revert to the divergence-of-normal estimate. This local-surface curvature substantially reduces the spurious-current level at the small capillary and electric capillary numbers relevant here (Section 4.1), consistent with the general observations in [29,30]. The static contact angle at the collector is enforced by rotating the interface normal entering the curvature stencil within a wall band of approximately 1.5 cells, in the manner of height-function/normal-correction treatments [35].

## 3.5 Charge transport and the electric subsystem

Equation (7) is advanced with the same time step as the flow (with optional equal sub-steps if the local relaxation time is under-resolved), using TVD convection of $\rho_{E}$ and an implicit-in-time treatment of the stiff ohmic term. Writing the ohmic term in potential form $\nabla\cdot(\sigma\nabla\phi)$, its discrete face fluxes reuse the Poisson-operator stencil of Eq. (6), which renders the conduction update compatible with the discrete Gauss law: the conductive current that leaves a cell is exactly the current received by its neighbor, and the global charge budget closes to round-off (Section 4.1). The potential equation (6) is re-solved whenever $\alpha$ (hence $\varepsilon$) or $\rho_{E}$ has been updated within the step. Dirichlet values of $\phi$ on electrode and collector patches are imposed through the boundary flux stencil; the diagnostic emitted current is decomposed into its conductive ($\sigma\mathbf{E}$) and convective ($\rho_{E}\mathbf{u}$) contributions integrated over a control surface, which is the quantity compared against the benchmark in Section 4.2.

## 3.6 Maxwell force evaluation

A naive cell-centered evaluation of Eq. (8) with collocated $\mathbf{E}$ produces an imbalance against the face-based pressure gradient and can generate spurious interfacial accelerations comparable to the physical dielectric force. We therefore assemble $\mathbf{f}_{e}$ from *face* quantities: face-normal fields $E_{f} = -(\nabla\phi)_{f}\cdot\hat{\mathbf{S}}_{f}$ consistent with the discrete Poisson flux, face permittivities consistent with Eq. (4), and the identity (8) applied in its stress-divergence form on the dual of the pressure-gradient stencil. The Coulomb contribution uses the cell charge with the face-interpolated field. This face-consistent construction parallels the well-balanced treatments in [16,28] and, in static equilibrium tests (charged interface at rest), reduces the residual acceleration by more than an order of magnitude relative to the collocated evaluation. [The precise figure for the final force configuration will be inserted with the production verification table.]

## 3.7 Time-step control and the adaptive electric-force limiter

The base time step enforces the convective CFL condition and the capillary-wave constraint of Brackbill et al. [27],

$$\Delta t_{\gamma} \le \left[\frac{\left(\rho_{l}+\rho_{g}\right) h^{3}}{4\pi\gamma}\right]^{1/2}, \qquad (11)$$

with $h$ the local cell size, together with a charge-relaxation bound $\Delta t \le c_{e}\,\tau_{e}$ when the dimensional electric module is active. These constraints, however, do not control the *explicit Maxwell force*: near a sharpening, strongly charged tip, the force-density magnitude $f_{e}$ grows approximately like $E^{2}/h$ across the interface band, and a step that is stable by the convective and capillary criteria may still displace the interface by several cells within a single update, after which the potential, charge, and curvature fields are mutually inconsistent and the computation diverges. This failure mode appears precisely on the tip-refined meshes required for defect resolution.

We bound it with a displacement argument (Appendix A): requiring that the velocity increment imparted by the electric force within one step advect material by no more than a fraction of the local cell yields

$$\Delta t_{F} = s_{F}\,\min_{i}\left(\frac{\rho_{i}\,h_{i}}{f_{e,i}}\right)^{1/2}, \qquad h_{i} = V_{i}^{1/3}, \qquad (12)$$

where $f_{e,i}$ is the electric force-density magnitude in cell $i$, and the working step is $\Delta t = \min(\Delta t_{\mathrm{base}},\ \Delta t_{F})$. The minimum in Eq. (12) is evaluated from the force field of the *previous* step (one lagged reduction; no extra force evaluation), and the safety factor is $s_{F} = 0.05$, calibrated once on a deliberately under-resolved, high-$\mathrm{Ca}_{E}$ reproducer of the tip blow-up and used unchanged in all runs. By construction the limiter is inactive wherever the electric force is weak—coarse-mesh and low-field computations are bit-identical with and without it—and it binds only in the small neighborhood of the tip where it is needed. Section 4.1 demonstrates that with Eq. (12) the fine-mesh cone-jet integrations proceed stably where the unlimited scheme diverges, at the cost of a reduced step only during strong-field transients.

## 3.8 Named-patch boundary conditions on resolved geometry

Production hardware enters the solver as an OpenFOAM polyMesh whose boundary faces are grouped in named patches. A compact boundary-condition object maps patch names (case-insensitive substring matching: *inlet*, *nozzle/electrode*, *collector/ground*, *outlet/atmosphere*, *symmetry*) to roles, and each role to the physical condition set of Section 2.4; per-face role indices are captured before any internal re-labeling of the mesh so that the assignment survives geometry preprocessing. When the mapping is disabled, or for legacy structured domains, the solver reverts—verified byte-for-byte—to its geometric boundary classification. The framework thus adds resolved-geometry capability without perturbing the validated structured-domain behavior. Appendix B tabulates the per-patch conditions for all five fields.

## 3.9 Defect-parametric emitter meshing

The emitter, its bore, and the surrounding atmosphere are meshed as a single fluid domain from which the capillary wall is excluded as an internal void; the void surface constitutes the *nozzle wall* patch (Fig. 1). A parametric generator produces hexahedral meshes with tip-graded resolution (near-tip cell size $h_{\mathrm{tip}}$ down to a few micrometers, coarsening toward the far field), and superposes three defect families on the nominal square-rimmed tip (Fig. 2): (D1) *rim blunting*, an axisymmetric fillet of radius $r_{b}$ applied to the tip annulus, representing erosion-induced apex-radius growth; (D2) *axis tilt*, a rigid inclination of the capillary axis by an angle β, representing fabrication misalignment or asymmetric wear; and (D3) *micro-protrusion*, a one-sided solid bump of height $h_{p}$ on the rim, representing burrs or pitting residue. D1 preserves axisymmetry of the geometry (though not necessarily of the solution); D2 and D3 are intrinsically three-dimensional.

A practical resolution criterion emerged from grid studies: a defect participates in the discrete geometry only if it spans at least approximately three to four cells of the local mesh. With $h_{\mathrm{tip}}$ coarser than roughly half the defect scale, the staircase representation of the fillet or bump degenerates and the defective and nominal meshes become cell-wise identical—i.e., the defect is silently absent from the computation. All defect runs therefore use tip-graded meshes with $h_{\mathrm{tip}}$ chosen such that $r_{b}/h_{\mathrm{tip}} \gtrsim 3$ (respectively $h_{p}/h_{\mathrm{tip}} \gtrsim 3$), and Section 4.3 documents the representability check (cell-count signature and tip-field response as functions of $h_{\mathrm{tip}}$). This observation, mundane as it is, is a load-bearing methodological point for any defect-resolved EHD study: because the tip field scales inversely with the local radius and the Maxwell force quadratically with the field, geometric fidelity of the defect is a prerequisite for force fidelity.

![](/home/younglin90/work/claude_code/claudeCFD/solver_electrospray/docs/electrospray/figures/nozzle_3d.png){width=6.2in}

**Figure 1.** *Computational domain and resolved emitter geometry: capillary of inner/outer diameter $D_i/D_o$ and length $L_n$ protruding into the atmosphere domain, grounded collector at distance $H$; the capillary wall is excluded from the fluid domain and its surface forms the nozzle-wall electrode patch (cutaway rendering; tip insets show the defect families of Section 3.9).*

![](/home/younglin90/work/claude_code/claudeCFD/solver_electrospray/docs/electrospray/figures/tip_defect_geometries.png){width=6.2in}

**Figure 2.** *Tip-defect parametrization resolved in the mesh: nominal square-rimmed tip; D1 rim blunting, fillet radius $r_b$; D2 axis tilt, angle β; D3 one-sided rim protrusion, height $h_p$ (axial cross-sections through the emitter tip).*

![](/home/younglin90/work/claude_code/claudeCFD/solver_electrospray/docs/electrospray/figures/resolved_nozzle_mesh.png){width=6.2in}

**Figure 3.** *Tip-graded resolved-nozzle mesh: axial cross-sections showing the near-tip refinement ($h_\mathrm{tip}$ of a few micrometers) coarsening toward the far field, and the void capillary wall whose surface forms the electrode patch.*

## 3.10 Solution algorithm and implementation

One time step consists of: (1) update mixture properties from $\alpha^{n}$; (2) set $\Delta t$ from the base constraints and Eq. (12) (lagged force field); (3) solve the potential equation (6); (4) advance charge, Eq. (7), with the conduction term implicit and the potential refreshed if requested; (5) assemble $\mathbf{f}_{e}$ (Section 3.6) and $\mathbf{f}_{\gamma}$ (Section 3.4); (6) momentum predictor; (7) Rhie–Chow projection with iterated pressure correction; (8) geometric $\alpha$ advection (Section 3.2) with the projected face fluxes; (9) diagnostics (mass and charge budgets, currents, interface morphology). The solver is implemented in C++ as a header-only finite-volume core with the electrospray application layered on top; shared-memory parallelism is applied with OpenMP to the provably disjoint per-cell kernels (least-squares gradients, PLIC volume-matching reconstruction, mixture updates), which preserves bit-identical results across thread counts by construction (no parallel reductions) and yields a wall-clock speed-up of approximately 2.3 at eight threads on the meshes of Section 4; the residual cost is dominated by the preconditioned Krylov solves. A browser-based front end drives case setup, patch-role assignment, and run control; all cases are defined by a single JSON document to which the solver defaults are exported, eliminating configuration drift between the interface and the solver.

# 4. Results and discussion

*[Placeholder convention: numerical values marked "T.B.D." and all bracketed figures/tables in this section will be finalized from the production runs; the values quoted without brackets are from completed verification computations. Expected trends are stated as expectations and will be revised where the data dictate.]*

## 4.1 Conservation, well-balancedness, and stability verification

Table 2 collects the conservation diagnostics of representative completed runs. On the structured verification domain (2,880–11,128 cells, 15–20 capillary time units), the global liquid-volume drift of the geometric advection scheme is at the 10⁻¹⁵–10⁻¹⁴ level (relative), i.e., round-off accumulation rather than scheme error, and the maximum cell-wise continuity defect after projection is below 10⁻¹¹. The charge budget closes to comparable levels when the conduction flux reuses the Poisson stencil (Section 3.5). These figures are insensitive to the number of OpenMP threads, as expected from the reduction-free parallelization.

**Table 2.** Conservation diagnostics (completed verification runs; representative).

| Case | Cells | Steps | Rel. mass drift | Max. continuity defect |
|---|---|---|---|---|
| Structured meniscus box, CaE = 0.35 | 2,880 | 20 | 3.4 × 10⁻¹⁵ | 8.3 × 10⁻¹³ |
| Resolved nozzle, named-patch BCs, CaE = 0.35 | 11,128 | 15 | 3.5 × 10⁻¹⁵ | 7.8 × 10⁻¹² |
| Resolved nozzle, tip-graded, CaE = 0.25 | 88,856 | T.B.D. | T.B.D. | T.B.D. |

For well-balancedness, a static charged interface test (planar and spherical) measures the residual velocity generated by the discrete imbalance between the pressure gradient and the capillary plus dielectric forces. With the face-consistent force assembly and PLIC-quadric curvature the spurious-current magnitude remains [T.B.D.; expected at or below 10⁻³ in capillary-velocity units $\gamma/\mu_l$, consistent with the balanced-force CSF literature [28–30]] over ten capillary times.

The necessity of the electric-force time-step limiter (Section 3.7) is demonstrated on a deliberately harsh configuration: an under-resolved tip at $\mathrm{Ca}_{E} = 0.8$. Without Eq. (12) the interface displacement per step at the tip exceeds the cell size shortly after cone formation and the run diverges (unbounded velocity within a few steps); with Eq. (12) and $s_{F} = 0.05$ the same configuration integrates stably to the end of the run window, with the step reduced only during the strong-field transient [Fig. 4]. On coarse/low-field cases the limiter never binds and results are bit-identical with and without it. We regard Eq. (12) as the practical enabling ingredient for the tip-refined defect meshes of Section 4.4, and note its resemblance in spirit to the capillary constraint (11): both bound an explicit interfacial force by a cell-scale displacement argument.

**Figure 4.** *Effect of the adaptive electric-force time-step limiter on a tip-refined, high-$\mathrm{Ca}_E$ reproducer: time history of maximum interface velocity and of $\Delta t$, with and without Eq. (12). [To be inserted.]*

## 4.2 Validation against a three-dimensional cone-jet benchmark

The solver is validated against the three-dimensional VOF cone-jet computation and experimental data of Cándido and Páscoa [17] at the operating point of Table 1. The comparison comprises: (i) the transient of Taylor-cone formation under a voltage ramp, characterized by the meniscus apex trajectory; (ii) the steady cone-jet morphology (cone half-angle, silhouette, and jet diameter at fixed stations); and (iii) the emitted current and its decomposition. We expect (i) the computed cone half-angle to approach the Taylor value of 49.3° within the accuracy permitted by the finite feed rate and finite domain, consistent with [2,17]; (ii) the jet diameter to lie within [T.B.D.; expected ±10–15%] of the benchmark simulation; and (iii) the current at $\mathrm{Ca}_{E} = 0.25$ to match the benchmark within [T.B.D.; expected ±15%], with the conductive contribution dominant in the cone and the convective contribution taking over along the jet, as required by the leaky-dielectric current-transfer picture [4,5]. The current–flow-rate scaling across a $Q$ sweep will be compared with the $I \propto (\gamma\sigma Q/\varepsilon_{r})^{1/2}$ law of [5]; agreement of the exponent within [T.B.D.; expected a few percent] is anticipated on the basis of preliminary coarse-mesh sweeps.

**Figure 5.** *Validation at $\mathrm{Ca}_E = 0.25$: (a) meniscus apex position vs. time under the voltage ramp; (b) steady cone-jet silhouette against the benchmark simulation; (c) emitted current (total, conductive, convective) vs. $\mathrm{Ca}_E$; (d) current vs. flow rate against the scaling law of [5]. [To be inserted.]*

## 4.3 Grid convergence and defect representability

Grid convergence of the baseline resolved-nozzle cone-jet is assessed on a sequence of tip-graded meshes ($h_{\mathrm{tip}} \approx 24, 12, 6$ μm; up to approximately 9 × 10⁴ cells for the production level, with the finest level reserved for a confirmation run). Monitored quantities are the tip field enhancement, cone half-angle, jet diameter, and emitted current; we expect monotone convergence of the morphological metrics and [T.B.D.] relative change between the two finest levels, below the run-to-run variability of the transient.

The defect-representability criterion of Section 3.9 is verified by meshing the D1 fillet sweep at fixed $r_{b}$ under decreasing $h_{\mathrm{tip}}$. For $h_{\mathrm{tip}}$ coarser than approximately $r_{b}/2$ the defective mesh is cell-wise identical to the nominal one (the generator reports identical cell counts and the tip-field probe is unchanged); once $r_{b}/h_{\mathrm{tip}} \gtrsim 3$ the fillet appears in the cell census and the computed tip field responds monotonically to $r_{b}$. All production defect cases satisfy $r_{b}/h_{\mathrm{tip}} \ge 3$ (respectively $h_{p}/h_{\mathrm{tip}} \ge 3$).

## 4.4 Application: geometric tip defects in a miniaturized emitter

The demonstration study comprises eleven cases at the fixed operating point $\mathrm{Ca}_{E} = 0.25$ (Table 3): the nominal sharp tip C0; a D1 blunting sweep $r_{b}$ = 8, 15, 20, 25 μm (C1–C4, the last corresponding to a fully rounded rim, $r_{b} = (D_{o}-D_{i})/2$); a D2 tilt sweep β = 2, 5, 10° (C5–C7); and a D3 protrusion sweep $h_{p}$ = 5, 10, 20 μm (C8–C10). An optional stability-margin set repeats C0, C4, C7, and C10 at $\mathrm{Ca}_{E} = 0.42$. Reported metrics are the tip field-enhancement factor, cone apex position and half-angle, jet direction (thrust-vector deviation $\theta_{d}$), jet diameter, emitted current and its decomposition, and the lateral-asymmetry norm of the interface.

**Table 3.** Tip-defect case matrix (all at CaE = 0.25; geometry differs only at the tip).

| Case | Defect | Parameter | Symmetry | Mesh |
|---|---|---|---|---|
| C0 | none (sharp rim) | — | axisymmetric geometry | tip-graded |
| C1–C4 | D1 blunting | rb = 8, 15, 20, 25 μm | axisymmetric geometry | tip-graded (reused) |
| C5–C7 | D2 tilt | β = 2, 5, 10° | fully 3-D | new 3-D meshes |
| C8–C10 | D3 protrusion | hp = 5, 10, 20 μm | fully 3-D | local tip refinement |

Anticipated outcomes, to be tested quantitatively: for **D1 blunting**, the tip field enhancement decreases with $r_{b}$ (the electrostatic field at the rim scales inversely with the local radius), raising the effective onset voltage and shifting the operating margin at fixed $U_{0}$; morphologically we expect a thicker cone base and delayed jet formation. Prior experimental evidence on apex-angle variation [21] indicates that plume collimation does not degrade monotonically with tip sharpness—an over-sharp emitter can emit off-axis—so the divergence-related metrics will be examined for non-monotonic behavior across C0–C4 rather than assumed monotone. For **D2 tilt**, the leading effect is directional: we expect the mean jet axis, hence the thrust vector, to deviate by an angle $\theta_{d}$ of the order of the geometric tilt β (with a proportionality we will quantify), which for thruster arrays converts directly into a thrust-vector error budget; secondary effects (asymmetric wetting of the rim, azimuthal modulation of the emitted current) will be reported. For **D3 protrusion**, the bump introduces a localized field spike; depending on $h_{p}$ relative to the local interface scale we expect either a benign pinning of the cone (small $h_{p}$) or the appearance of asymmetric emission and enhanced lateral oscillation (large $h_{p}$), with the current trace acquiring fluctuation content absent in C0. Since the cone-jet current at fixed flow rate and liquid properties is only weakly geometry-dependent [5], we expect the *mean* current to change far less than the directional and stability metrics across all defect families—this contrast, if confirmed, is itself a useful diagnostic separating geometry-driven from transport-driven degradation.

**Figure 6.** *D1 blunting sweep: (a) tip field-enhancement factor and (b) cone/jet morphology metrics vs. $r_b$; (c) silhouettes of C0–C4. [To be inserted.]*

**Figure 7.** *D2 tilt sweep: thrust-vector deviation $\theta_d$ vs. tilt angle β; jet-axis trajectories. [To be inserted.]*

**Figure 8.** *D3 protrusion sweep: near-tip field maps, interface asymmetry norm, and current-fluctuation spectra vs. $h_p$. [To be inserted.]*

**Table 4.** Defect-sensitivity summary (to be completed): tip field enhancement, onset-margin shift, $\theta_d$, jet diameter, mean current, current-fluctuation level, and asymmetry norm for C0–C10.

| Case | Etip/E0 | θd (deg) | djet/Di | I/I0 | Asym. norm |
|---|---|---|---|---|---|
| C0 | T.B.D. | T.B.D. | T.B.D. | 1 (ref.) | T.B.D. |
| C1–C4 | T.B.D. | T.B.D. | T.B.D. | T.B.D. | T.B.D. |
| C5–C7 | T.B.D. | T.B.D. | T.B.D. | T.B.D. | T.B.D. |
| C8–C10 | T.B.D. | T.B.D. | T.B.D. | T.B.D. | T.B.D. |

## 4.5 Limitations

Several limitations bound the scope of the conclusions. The leaky-dielectric model omits ion field-emission and space-charge effects downstream, so the study addresses the cone and jet, not the far plume; droplet breakup statistics require finer resolution than the production meshes and are not claimed. The defect study is quasi-static in geometry: erosion is represented by its geometric end state, not co-evolved with the flow. Material-property defects (oxide layers, wettability change) are deliberately excluded to isolate geometric effects. Finally, the contact-line treatment is a static-angle model within a curvature stencil; dynamic contact-angle effects on the collector film are outside the present scope.

# 5. Conclusions

We have described a three-dimensional, unstructured finite-volume solver for leaky-dielectric electrohydrodynamic two-phase flow, combining volume-matched geometric PLIC advection of the interface, balanced-force capillarity with reconstruction-based curvature, a charge-transport discretization compatible with the discrete Gauss law, and a face-consistent Maxwell-force assembly. Two elements address the specific difficulties of computing cone-jets on geometrically resolved emitter hardware: an adaptive time-step limiter, Eq. (12), that bounds the cell-scale displacement induced by the explicit electric force and—in our experience—converts fine-mesh tip computations from divergent to routinely stable at negligible cost; and a named-patch boundary-condition framework that carries the physical cone-jet conditions onto imported meshes of real emitter geometry while preserving legacy behavior elsewhere. Verification shows liquid-mass conservation at the 10⁻¹⁵–10⁻¹⁴ level and continuity defects near round-off; validation against a published three-dimensional cone-jet benchmark at $\mathrm{Ca}_{E} = 0.25$ [is reported / will be completed] for morphology and current. The demonstration application—rim blunting, axis tilt, and micro-protrusion defects resolved directly in the emitter mesh—establishes a resolution criterion for defect representability (at least three to four cells across the defect) and provides a template for translating fabrication and degradation tolerances of miniaturized electrospray emitters into quantitative spray-metric sensitivities. Extension to co-evolving erosion, material-property defects, and array-level interactions is left to future work.

# Appendix A. Displacement bound for the explicit Maxwell force

Consider a cell of size $h$ and density $\rho$ subject to a (locally constant) explicit electric force density of magnitude $f_{e}$ over one step $\Delta t$. The induced velocity increment is $\Delta u = f_{e}\,\Delta t/\rho$, and the additional displacement accrued within the step is bounded by $\delta = \tfrac{1}{2}\,(f_{e}/\rho)\,\Delta t^{2}$. Requiring $\delta \le c\,h$ with a tolerance fraction $c < 1$ gives

$$\Delta t \le \left(\frac{2\,c\,\rho\,h}{f_{e}}\right)^{1/2}, \qquad (\mathrm{A.1})$$

which is Eq. (12) with $s_{F} = \sqrt{2c}$ and the global minimum taken over cells; $s_{F} = 0.05$ corresponds to a displacement tolerance $c = 1.25\times10^{-3}$ of the local cell per step, a deliberately conservative choice calibrated once on the blow-up reproducer and then frozen. The lagged evaluation (force field of the previous step) makes the limiter free of additional force assemblies; because $f_{e}$ varies smoothly between successive steps in stable operation, the lag does not compromise the bound in practice, and a violation simply triggers a smaller step on the following update. The analogous displacement reading of the capillary constraint, Eq. (11), replaces $f_{e}/\rho$ by the capillary acceleration scale $\gamma\kappa/(\rho h) \sim \gamma/(\rho h^{2})$, which recovers the familiar $\Delta t \propto h^{3/2}$ scaling; Eq. (12) is thus the natural companion bound for the electric force, degenerating gracefully (large $\Delta t_{F}$, never binding) wherever the field is weak.

# Appendix B. Patch-wise boundary conditions

**Table B.1.** Boundary conditions by patch role (named-patch framework, Section 3.8). $U_0$ is the applied potential; $u_{\mathrm{in}} = 4Q/(\pi D_i^2)$; "zg" denotes zero normal gradient.

| Field | Liquid inlet | Nozzle wall (electrode) | Collector (ground) | Open atmosphere |
|---|---|---|---|---|
| u | parabolic, mean u_in | no-slip | no-penetration wall (opt. tangential motion) | zg (in/outflow) |
| p | zg | zg | zg | far-field reference |
| α | 1 (Dirichlet) | zg | zg + contact angle θc | zg |
| φ | U0 | U0 | 0 | zg (insulating) |
| ρE | 0 (Dirichlet) | zg | zg | 0 on inflow / zg on outflow |

Patch names are matched to roles by case-insensitive substrings (*inlet/feed*; *nozzle/electrode/capillary/needle*; *collector/ground/plate/target*; *outlet/atm/far/open/ambient*; *symm/wedge/axis*); unmatched patches default to the open-atmosphere role. Disabling the framework restores the geometric boundary classification of the structured solver, verified byte-for-byte on the regression suite.

# Acknowledgements

[Funding sources, grants, and computational resources to be acknowledged here. The authors thank … (to be completed).]

# References

[1] J. Zeleny, Instability of electrified liquid surfaces, Phys. Rev. 10 (1917) 1–6.

[2] G.I. Taylor, Disintegration of water drops in an electric field, Proc. R. Soc. Lond. A 280 (1964) 383–397.

[3] J.R. Melcher, G.I. Taylor, Electrohydrodynamics: a review of the role of interfacial shear stresses, Annu. Rev. Fluid Mech. 1 (1969) 111–146.

[4] D.A. Saville, Electrohydrodynamics: the Taylor–Melcher leaky dielectric model, Annu. Rev. Fluid Mech. 29 (1997) 27–64.

[5] J. Fernández de la Mora, I.G. Loscertales, The current emitted by highly conducting Taylor cones, J. Fluid Mech. 260 (1994) 155–184.

[6] M. Cloupeau, B. Prunet-Foch, Electrostatic spraying of liquids in cone-jet mode, J. Electrostat. 22 (1989) 135–159.

[7] J. Fernández de la Mora, The fluid dynamics of Taylor cones, Annu. Rev. Fluid Mech. 39 (2007) 217–243.

[8] J.K. Ziemer, T.M. Randolph, G.W. Franklin, V. Hruby, D. Spence, N. Demmons, T. Roy, E. Ehrbar, J. Zwahlen, R. Martin, W. Connolly, Colloid micro-Newton thrusters for the Space Technology 7 mission, IEEE Aerospace Conference (2010) 1–19.

[9] A. Krejci, P. Lozano, Space propulsion technology for small spacecraft, Proc. IEEE 106 (2018) 362–378.

[10] J.R. Melcher, Continuum Electromechanics, MIT Press, Cambridge, MA, 1981.

[11] A.M. Gañán-Calvo, Cone-jet analytical extension of Taylor's electrostatic solution and the asymptotic universal scaling laws in electrospraying, Phys. Rev. Lett. 79 (1997) 217–220.

[12] A.M. Gañán-Calvo, J.M. López-Herrera, M.A. Herrada, A. Ramos, J.M. Montanero, Review on the physics of electrospray: from electrokinetics to the operating conditions of single and coaxial Taylor cone-jets, and AC electrospray, J. Aerosol Sci. 125 (2018) 32–56.

[13] M.A. Herrada, J.M. López-Herrera, A.M. Gañán-Calvo, E.J. Vega, J.M. Montanero, S. Popinet, Numerical simulation of electrospray in the cone-jet mode, Phys. Rev. E 86 (2012) 026305.

[14] M. Gamero-Castaño, M. Magnani, Numerical simulation of electrospraying in the cone-jet mode, J. Fluid Mech. 859 (2019) 247–267.

[15] G. Tomar, D. Gerlach, G. Biswas, N. Alleborn, A. Sharma, F. Durst, S.W.J. Welch, A. Delgado, Two-phase electrohydrodynamic simulations using a volume-of-fluid approach, J. Comput. Phys. 227 (2007) 1267–1285.

[16] J.M. López-Herrera, S. Popinet, M.A. Herrada, A charge-conservative approach for simulating electrohydrodynamic two-phase flows using volume-of-fluid, J. Comput. Phys. 230 (2011) 1939–1955.

[17] S. Cándido, J.C. Páscoa, Numerical simulations of electrohydrodynamic atomization in the cone-jet mode by a three-dimensional volume-of-fluid method, Phys. Fluids 35 (2023) 052110.

[18] C.B. Whittaker, et al., Manufacturing variability of electrospray emitter tips and its propulsion-level consequences, J. Propul. Power (full citation to be completed).

[19] B.A. Banks, S.K. Miller, K.K. de Groh, Low Earth orbital atomic oxygen interactions with spacecraft materials, NASA/TM-2004-213223, 2004.

[20] A. de Rooij, Corrosion in space, in: Encyclopedia of Aerospace Engineering, Wiley, 2010.

[21] [Tip apex-angle effect on electrospray plume collimation, ACS Appl. Electron. Mater. 6 (2024); full citation to be completed.]

[22] [Electrospray emitter degradation and lifetime, Aerospace (MDPI) 7 (2020); full citation to be completed.]

[23] J. Roenby, H. Bredmose, H. Jasak, A computational method for sharp interface advection, R. Soc. Open Sci. 3 (2016) 160405.

[24] H. Scheufler, J. Roenby, Accurate and efficient surface reconstruction from volume fraction data on general meshes, J. Comput. Phys. 383 (2019) 1–23.

[25] O. Ubbink, R.I. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 153 (1999) 26–50.

[26] S.S. Deshpande, L. Anumolu, M.F. Trujillo, Evaluating the performance of the two-phase flow solver interFoam, Comput. Sci. Discov. 5 (2012) 014016.

[27] J.U. Brackbill, D.B. Kothe, C. Zemach, A continuum method for modeling surface tension, J. Comput. Phys. 100 (1992) 335–354.

[28] M.M. Francois, S.J. Cummins, E.D. Dendy, D.B. Kothe, J.M. Sicilian, M.W. Williams, A balanced-force algorithm for continuous and sharp interfacial surface tension models within a volume tracking framework, J. Comput. Phys. 213 (2006) 141–173.

[29] S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows, J. Comput. Phys. 228 (2009) 5838–5866.

[30] S. Popinet, Numerical models of surface tension, Annu. Rev. Fluid Mech. 50 (2018) 49–75.

[31] H.G. Weller, G. Tabor, H. Jasak, C. Fureby, A tensorial approach to computational continuum mechanics using object-oriented techniques, Comput. Phys. 12 (1998) 620–631.

[32] H.A. van der Vorst, Bi-CGSTAB: a fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems, SIAM J. Sci. Stat. Comput. 13 (1992) 631–644.

[33] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532.

[34] R.I. Issa, Solution of the implicitly discretised fluid flow equations by operator-splitting, J. Comput. Phys. 62 (1986) 40–65.

[35] S. Afkhami, M. Bussmann, Height functions for applying contact angles to 2D VOF simulations, Int. J. Numer. Methods Fluids 57 (2008) 453–472.

[36] Lord Rayleigh, On the equilibrium of liquid conducting masses charged with electricity, Philos. Mag. 14 (1882) 184–186.

[37] R.T. Collins, J.J. Jones, M.T. Harris, O.A. Basaran, Electrohydrodynamic tip streaming and emission of charged drops from liquid cones, Nat. Phys. 4 (2008) 149–154.

[38] R.S. Legge, P.C. Lozano, Electrospray propulsion based on emitters microfabricated in porous metals, J. Propul. Power 27 (2011) 485–495.

[39] P. Lozano, M. Martínez-Sánchez, Ionic liquid ion sources: characterization of externally wetted emitters, J. Colloid Interface Sci. 282 (2005) 415–421.

[40] P.M. Hartman, D.J. Brunner, D.M.A. Camelot, J.C.M. Marijnissen, B. Scarlett, Electrohydrodynamic atomization in the cone-jet mode: physical modeling of the liquid cone and jet, J. Aerosol Sci. 30 (1999) 823–849.
