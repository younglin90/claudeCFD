J. Fluid Mech. (2026), vol. 1029, A35, doi:10.1017/jfm.2026.11180


# Computational study of the separation of regular sphere clusters in high-Mach-number flow

Thomas Whalen

1, Ralf Deiterding

2 and Stuart Jon Laurence

1

1Department of Aerospace Engineering, University of Maryland, College Park, MD 20742, USA 2School of Engineering, University of Southampton, Boldrewood Campus, Southampton SO16 7QF, UK Corresponding author: Stuart Jon Laurence, stuartl@umd.edu

(Received 24 May 2025; revised 24 November 2025; accepted 11 January 2026)

A coupled computational-fluid-dynamics/finite-element methodology is implemented to investigate the free aerodynamic separation of clusters of equally sized spheres arranged in regular configurations in Mach-20 flow, representing an idealized meteoroidfragmentation scenario. The regular nature of the initial agglomeration geometries – touching sphere pairs, tetrahedral four-sphere arrangements and face-centred-cubic 13-sphere configurations – allows a systematic exploration of both individual sphere motions and bulk cluster dynamics as the initial orientation is varied. For sphere pairs, a stable lifting configuration arises when the spheres are in contact in a skewed configuration, a phenomenon that can also emerge in the more populous clusters. In the tetrahedral survey, comprising 38 initial orientations, shock surfing of downstream bodies is found to play a significant role in driving the separation dynamics. Despite substantial variations in detailed sphere motions with initial orientation, the trajectory type and final lateral velocity collapse reasonably well with the initial polar angle of the sphere within the cluster. Indices describing the bluntness and asymmetry of the initial configuration are introduced and correlate well with the collective cluster dynamics, though not always in an intuitive way. For the 13-sphere clusters, the dependency of individual sphere lateral velocities follows a similar trend with initial polar angle to the four-sphere case, suggesting that a simplified separation model may be possible for such configurations. The influence of the initial cluster bluntness on the bulk dynamics is somewhat reduced, however, indicating a tendency towards more homogeneous separation as the cluster population is increased.

Key words: flow-structure interactions, high-speed flow, hypersonic flow

© The Author(s), 2026. Published by Cambridge University Press. This is an Open Access article, distributed under the terms of the Creative Commons Attribution licence (https://creativecommons.org/ licenses/by/4.0/), which permits unrestricted re-use, distribution and reproduction, provided the original article is properly cited. 1029 A35-1

T. Whalen, R. Deiterding and S.J. Laurence

1. Introduction The recent discovery of the potential ‘city-killer’ asteroid, 2024 YR4, has again highlighted the risks posed to human life and property by the entry of celestial bodies into the Earth’s atmosphere. Such risks are multifaceted, including threats associated with the direct impact of the original object or its fragments at the terrestrial surface, generation of a tsunami for water impacts (Rumpf, Lewis & Atkinson 2017) and energy deposition from the body to the atmosphere, forming a shock wave that can propagate to the surface (Chyba, Thomas & Zahnle 1993). Larger bodies that could result in a mass extinction event (Alvarez et al. 1980) typically transit through to the surface unaffected by the atmosphere, but such bodies enter extremely infrequently and can be tracked using available technology. The residual risk to humans from entering meteoroids has thus been shifted to smaller objects (Bland & Artemieva 2003), which cannot be so easily tracked. The extremely high pressures that develop on the front-facing surfaces of these bodies (typically comparable to ρaV 2, where ρa is the atmospheric density and V the meteoroid speed relative to the atmosphere) will invariably result in their disruption at some point during atmospheric transit. Aerodynamic interactions between the generated fragments – potentially among other effects (Passey & Melosh 1980) – can greatly increase the ground footprint of those that survive transit, or augment the rate of energy deposition and thus the strength of the shock that propagates to the ground. Therefore, understanding the atmospheric disruption and subsequent separation processes in meteoroid fragmentation is crucial for determining the potential risk posed by such events. Two approaches have traditionally been taken in investigating the aerodynamic interactions of meteoroid fragments following atmospheric disruption: the ‘discretefragment’ approach, suitable for disruption into a limited number of distinct fragments; and the ‘debris-cloud’ approach, appropriate for catastrophic fragmentation or the disruption of strengthless, ‘rubble-pile’ type asteroids (Walsh 2018). These contrasting approaches are summarized in figure 1. The discrete-fragment approach is typified by the binary-fragment model introduced in Passey & Melosh (1980). These authors considered the purely lateral separation of two spherical bodies and derived a final lateral separation velocity of the smaller body, VT , as


$$
VT = V � C ρa
$$


$$
ρm
$$

r1 r2 . (1.1)

Here, ρm is the density of the meteoroid, r1 and r2 are the two fragment radii and C is a ‘constant’: by examining various terrestrial crater fields under the implicit assumption that separation was dominated by such binary interactions, Passey & Melosh (1980) determined C to lie between 0.03 and 2.28. Numerical simulations of the separation of two equal hemicylindrical bodies were subsequently performed by Artem’eva & Shuvalov (1996), and the derived value of C of 0.2 fell within this range. The theoretical and computational modelling of Laurence & Deiterding (2011) and subsequent experiments of Laurence, Parziale & Deiterding (2012), however, showed that the assumption of an exclusively lateral separation, while reasonable for equally sized bodies, is not appropriate for unequal bodies. In particular, the lower ballistic coefficient of the smaller body will typically cause it to be accelerated more quickly downstream, which can result in a phenomenon referred to as ‘shock-surfing’, whereby the smaller body rides the bow shock of the larger body downstream, significantly enhancing its separation velocity. The ‘debris-cloud’ approach, in which the fragmented body is treated as a strengthless, liquid-like agglomeration, is exemplified by the semitheoretical ‘pancake’ models of

1029 A35-2

Journal of Fluid Mechanics

Discrete-fragment model Debris-cloud model


> **Figure 1. The two approaches typically considered in modelling meteoroid-fragmentation events.**

Chyba et al. (1993) and Hills & Goda (1993). Essentially, the aerodynamic forces are assumed to compress the strengthless mass in the streamwise direction and expand it in the lateral direction, increasing the exposed area and thus the rate of energy deposited to the atmosphere, until a terminal radius (some multiple of the initial radius) is reached. Such pancake models have been employed, for example, by Register, Mathias & Wheeler (2017) and McMullan & Collins (2019) in attempts to match light curves of recorded entry events, but in some cases unrealistic expansion ratios (up to seven times the initial radius) have been found to be necessary, whereas Artemieva & Pierazzo (2009) caution against the use of a terminal radius above approximately twice the initial one. The binary-fragment model just discussed represents the low-population limit of the discrete-fragment approach, while the debris-cloud approach can be considered the other extreme of an effectively infinite fragment-cloud population. The intermediate population regime, however, has received comparatively limited attention. Artemieva & Shuvalov (2001) conducted limited numerical simulations of the separation of 13and 27-cube clusters using a hydrocode and derived a separation constant of C ≈1, though the fidelity and accuracy of these simulations is somewhat unclear. Another example is the study of Park & Park (2020), who performed experiments using a ring-like configuration of up to six spheres and proposed a basic model that would indicate a √

N dependence of the mean separation velocity on cluster population, N. The somewhat unphysical nature of the parent configuration employed in these experiments, however, does lead to questions regarding its applicability to actual fragmentation events. The need for more accurate models of discrete fragmentation has been highlighted by the observations of Borovicka & Kalenda (2003), who analysed video records of the Morávka fall and derived separation velocities up to an order of magnitude higher than those predicted by existing models. To help address the deficiencies in our current understanding of high-speed multibody separation, two of the present authors introduced a systematic experimental methodology for studying the aerodynamic separation of populous clusters (Whalen & Laurence 2021), whereby sphere agglomerations were released impulsively into a hypersonic wind-tunnel flow. Preliminary experiments revealed some of the key physics associated with smallto medium-population (up to 36-sphere) clusters. It was found, for example, that the separation process could be divided into two phases: a ‘primary’ phase, during which the individual separation velocities increased rapidly and almost linearly, and a secondary phase, which was typically characterized by strong subcluster interactions. The ultimate goal of such experimental investigations is to provide a validated statistical description of multibody separation, using random realizations over a specified parameter space. Due to the difficulties in precisely resolving sphere motions early in the separation process, however, these experiments are less well suited to provide a detailed understanding

1029 A35-3

T. Whalen, R. Deiterding and S.J. Laurence

of the governing physics as specific parameters governing the cluster arrangement are varied. Therefore, in parallel with these experimental efforts, we have developed a numerical methodology, combining computational fluid dynamics (CFD) and finite element analysis (FEA), for studying multibody separation. In the present work, we employ this methodology to study the separation of spheres from regular arrangements of two, four and 13 spheres in high-Mach-number flow. The regular nature of the configurations employed allows systematic variation of the relevant geometric parameters and an exploration of their effects on the separation dynamics. This article is structured as follows. In § 2, the simulation methodology is outlined, including the definition of important quantities, descriptions of the combined CFD–FEA framework employed here and the various configurations simulated, and verification and validation studies. Discussions of the important results for two-sphere, four-sphere and 13-sphere clusters are then provided in §§ 3, 4 and 5, respectively. In the latter two sections, we first discuss general separation characteristics and individual sphere motions, and then proceed to bulk cluster dynamics. Finally, conclusions are drawn in § 6.

2. Simulation methodology 2.1. Model problem and definitions The model problem of interest here, intended to represent a meteoroid-fragmentation event, is the separation of a regular cluster of equally sized spheres, released impulsively into a hypersonic flow, through mutual aerodynamic repulsion. Compared with a realistic fragmentation event, the configuration studied here is clearly highly idealized, in the specification both of the parent cluster (regular geometry) and the fragments (spherical and equal sized). This simplification is intentional, however, as it allows a systematic investigation of the problem, and the understanding gained is expected to form a foundation for future studies in which these assumptions can be relaxed. Although ablation effects will be present in a realistic entry situation, the simulations of Artemieva & Shuvalov (2001) have shown these to have a negligible influence on the separation dynamics, and they are thus ignored here. We further restrict our investigation to perfect-gas flows. Although high-temperature effects will be present at the hypervelocity conditions encountered during meteoroid entry (e.g. vibrational excitation, dissociation, ionization) and lead to deviations from perfect-gas behaviour, such effects generally have only a modest influence on pressures in shock-dominated flows (Vincenti & Kruger 1965), and the intersphere pressure will be the dominant effect in driving the cluster dynamics (Laurence & Deiterding 2011). Before proceeding further, it is useful to define several quantities that will be employed to characterize the separation behaviour of the cluster. The characteristic time scale and velocity are defined as


$$
τs =
�ρsph
ρ∞
$$


$$
rc u∞ and vs = �ρsph ρ∞ u∞, (2.1)
$$

where ρsph is the sphere density, ρ∞and u∞are the free stream density and velocity and rc is the initial circumscribed radius of the cluster. The lateral separation velocity of the ith sphere in the cluster, VT,i, is referenced to the non-stationary cluster centre of mass (subscript com), and is defined as the magnitude of the component in the y−z plane according to


$$
xrel,i = (xi −xcom) −(xi −xcom)ˆx, (2.2)
$$

1029 A35-4

Journal of Fluid Mechanics

VT,i = (vi −vcom) · xrel,i ||xrel,i||, (2.3)

where the ˆx subscript indicates the component in the streamwise direction. The collective lateral separation velocity of the cluster is then defined as

VT = 1

n

n �

i=1 VT,i, (2.4)

and the relevant non-dimensional quantities characterizing the cluster separation are

t′ = t/τs, V ′ T,i = VT,i/vs, V ′ T = VT /vs. (2.5)

2.2. Numerical methods To determine the dynamics of the spheres within the cluster, a foremost task is computing the flow field generated by the presence of the obstructing bodies, and in particular the pressure distribution over each sphere. The fluid pressure distributions over the spheres, in turn, dictate the aerodynamic forces experienced by each. We note, however, that the complete dynamical behaviour of the bodies will also be influenced by the surface contact that they will inevitably experience while in close proximity at early times, as well as the potential later periods of collision and sustained contact. The situation under investigation thus forms a multiphysics problem, which we approach by coupling a compressible CFD solver to a finite element analysis software for explicit solid mechanics. A primary challenge of simulating the flow around a group of separating spheres is appropriately modifying the topology of the fluid mesh while maintaining high accuracy in the flow field solution. Previous studies of computational free-flight sphere separation (Laurence et al. 2012; Butler et al. 2021) have implemented codes specifically designed to handle complex embedded boundaries robustly and adaptively refine the simulation mesh in regions of the flow field identified to contain features such as shock waves. Because many of the dynamically relevant features in the flow over a sphere at hypersonic conditions tend not to be viscous in nature (Laurence, Deiterding & Hornung 2007), we can model the physics approximately with the equations of inviscid flow to reduce computational cost. Although this choice comes at the expense of accuracy in the wake region of a sphere, as will be seen throughout this work (and as can be inferred from earlier work such as Laurence et al. 2012), the contribution of wake dynamics to the overall separation behaviour of equal-sphere clusters is minimal. The fluid solver chosen to compute the unsteady flow fields in this work is Adaptive Mesh Refinement in Object-Oriented C++ (AMROC) (Deiterding 2011), which solves the Euler equations for inviscid flow:


$$
∂tρ + ∇· (ρu) = 0, ∂t(ρu) + ∇· (ρu ⊗u) + ∇p = 0, ∂t(ρE) + ∇· [(ρE + p)u] = 0. (2.6)
$$

Here E represents the specific total energy, with the pressure determined from the polytropic equation of state, p = (γ −1)(ρE −(1/2)uT u). All solid features are treated as embedded boundaries within a Cartesian mesh (Deiterding 2009), while spatial discretization is formulated in a finite-volume flux-splitting scheme. The MUSCL-Hancock (monotonic upstream-centred scheme for conservation laws-Hancock) reconstruction method with a Min-Mod limiter is implemented via Van Leer flux vector

1029 A35-5

T. Whalen, R. Deiterding and S.J. Laurence

(a) (b)


> **Figure 2. (a) Representative computational mesh with three fluid refinement levels showing surface pressure, numerical schlieren and automatically refined mesh capturing the shock. (b) Collision of two spheres demonstrating the mesh structure employed and multibody contact capabilities of DYNA3D with cells coloured by principal stress in horizontal direction.**

splitting for estimation of numerical flux at cell interfaces. Away from shocks and discontinuities, this semidiscrete formulation provides second-order accuracy, reverting to first order near embedded boundaries across which a ghost-fluid-based interpolation scheme mirrors primitive variables. An explicit Euler time-marching scheme is used throughout. The embedded boundary method in AMROC is a variant of the first-orderaccurate ghost-fluid method proposed by Fedkiw et al. (1999). Further details of the Cartesian fluid schemes and also verification of the embedded boundary method can be found in Deiterding (2011); for validation comparisons for dynamically moving bodies we refer in particular to Laurence et al. (2012). The central advantage of AMROC is its use of a fully parallelized adaptive mesh refinement (AMR) scheme that permits the effective capture of transient flow features associated with moving boundaries. The patch-based approach divides the underlying Cartesian mesh into refinements subsets which are evaluated recursively, with the relative iteration count imposed by a prescribed target Courant–Friedrichs–Lewy (CFL) number. The refinement process is controlled by user-defined gradient thresholds of selected state variables (typically density) and by wall proximity as determined by the level-set function. The parallelized AMR method is equipped with both load-balancing and repartitioning to account for evolving mesh topologies. Figure 2(a) provides a demonstration of AMROC’s mesh-refinement capabilities; further details may be found in Deiterding (2011). A crucially important aspect of AMROC is its verified interface to Lagrangian finite element solvers in a software system called the Virtual Test Facility (VTF). While Deiterding et al. (2006) implemented and demonstrated fluid–structure coupling to a volumetric finite element solver, Cirak, Deiterding & Mauch (2007) provide rigorous validation for the fluid–structure coupling of a thin-shell finite element solver, including problems with fracture and fragmentation. The structural modelling component of the numerical framework here is accomplished using the DYNA3D solver. The DYNA3D solver is an explicit, nonlinear finite-element code commonly used to capture high-speed structural phenomena and allows for a range of material models and contact physics (Hallquist & Jin 2005). A structured grid of regular hexahedral elements is required for the current simulations, for which we employ a three-dimensional butterfly mesh with relaxation of the external cell blocks; this ensures reliable execution for a spherical structural domain, as illustrated by the touching spheres

1029 A35-6

Journal of Fluid Mechanics

of figure 2(b). A critical strength of DYNA3D is its robust and seminal contact detection algorithm (Hallquist, Goudreau & Benson 1985); a global search for proximity between principal surfaces and subsidiary nodes, supported by a detailed contact-checking routine, provides proper treatment of nodal penetration and directional pushback for multibody impacts in dynamic simulation settings. Sliding surface dynamics are computed using standard friction laws, with kinematic and static friction coefficients held constant at 0.5 and 0.7, respectively, in the present implementation. To examine the influence of the choice of friction coefficients, additional simulations were performed with values of 0.1/0.02 (i.e. a very smooth surface) and 1.5/0.9 (a highly roughened surface). For two-sphere separation, the influence on the sphere dynamics was entirely negligible; for four-sphere separation, modest changes in the sphere trajectories were observed, but these were substantially smaller than changes that resulted from variations in the initial cluster orientation. We further assume an elastic material model, although material failure along predefined fault lines would represent a natural extension of our methodology. Coupling the fluid and structural solvers is accomplished here similarly with VTF routines (cf. Deiterding et al. 2006; Cirak et al. 2007) and by transmitting to DYNA3D the pressure boundary conditions, to which the principal stresses of boundary cells are equated, and to AMROC the updated geometric boundaries, which determine the kinematics of embedded fluid ghost cells (Deiterding & Wood 2013). Boundary-condition transfer between solvers is updated serially, which increases computational cost in comparison with parallel execution but ensures a higher degree of numerical stability. Computation of the level-set function for imposing embedded wall boundary conditions in the Cartesian AMROC solver is performed by applying the ‘closest point transform’ of Mauch (2003) to the triangulated surface mesh of the solid. Solid and fluid meshes are constructed such that nodal spacing is roughly equivalent on the finest grid level, while the global time-stepping parameter is determined by selecting the minimum of the stresswave transmission in the solid and target CFL time scales from the fluid. In this study, DYNA3D is run on a single core, whereas AMROC is parallelized across a larger number of processors. We note that the situation under investigation here, i.e. a fully elastic separation problem lacking significant feedback from structural deformations into the flow field, does not ultimately require treatment of the material response with a finite element analysis, as less complex models (such as the discrete element method of Mishra & Rajamani (1992)) may produce similar results. The FEA-based approach was primarily chosen because AMROC had been earlier coupled to DYNA3D and successful verification and validation for various shock-driven fluid–structure interaction scenarios had already been achieved (Deiterding & Wood 2013), thereby circumventing the need for the development of new coupled computational tools.

2.3. Simulation parameters In all simulations, the fluid is a perfect gas (γ = 1.4) with an inflow Mach number of 20. This latter value is somewhat below that typical of meteoroid entry (M > 30) but is sufficiently high that the Mach-number independence principle (Anderson 2006) can be expected to hold over the extent of the simulated domain. The ratio of sphere-to-fluid density was set to 1×104 (in dimensional terms, 8000 versus 0.8 kg m−3); structurally, we employ a fully elastic material model with a Young’s modulus of 200 GPa and a Poisson ratio of 0.28 (i.e. values appropriate for iron). In each simulation, the cluster was initially held in place while the inflow velocity was ramped up; once steady state was achieved,

1029 A35-7

T. Whalen, R. Deiterding and S.J. Laurence


> **Figure 3. General geometric appearance of two-, fourand 13-sphere close-packed clusters.**

x

z

Pitch

Yaw

Flow

y


> **Figure 4. Rendering of the 13-sphere cluster geometry with principal attitude and pitch/yaw angles shown.**

the spheres were impulsively released and subsequently allowed to fly freely in response to their experienced forces, consistent with the model problem described above. Regular clusters of equal-radius spheres with three different populations – two, four and 13 bodies – are examined for various initial orientations in this work, and examples are shown in figure 3. For reference, the sphere radius, rsph, is set as 0.1 m in all cases. For the sphere-pair simulations, the bodies are initially in contact and rotated about their common centre-of-mass by a variety of pitch values. For more populous clusters, we maintain close-packed sphere configurations and vary both the pitch and yaw angles of the cluster from its principal attitude, as shown in figure 4 for a 13-sphere cluster (the exact details of the selected parameter values in each survey will be given in the appropriate subsections of later chapters). Four-sphere clusters are thus formed by positioning the sphere centres at the vertices of a regular tetrahedron, and 13 spheres by constructing two layers of a face-centred cubic lattice. Each structural mesh for the twoand four-sphere clusters contains 2400 surface elements, and the refinement levels of the solid and fluid domains are matched such that the maximum stable time-step size in DYNA3D roughly equates to a target CFL number of 0.8 for time-stepping control in AMROC. Due to the drastically increased computational demands of the 13-sphere cluster simulations, in this case we employ a coarsened mesh and a reduced CFL target of 0.7 to promote the stability of the computations. The exact spatial scales of the fluid domain vary between cluster populations, as does the simulation duration, and this information can be found in table 1 along with the utilized refinement factors. Typical computational times increased from ∼700 CPU hr for two bodies to ∼1300 and 4000 CPU hr for the fourand 13-sphere agglomerations, respectively, on a 56-core Dell Precision T7820 workstation. In total, 13 runs were conducted for two spheres, 36 for four spheres and 34 for 13 spheres.

1029 A35-8

Journal of Fluid Mechanics

Nsph Grid Base grid Domain size Refinement �xmin Duration (Structure) (Fluid) (· 1/rsph) Refinementfactors (· 1/rsph) (· 1/τs)

2 20 × 20 × 20 126 × 50 × 50 25 × 10 × 10 [2, 2, 0] 0.050 9.0 4 20 × 20 × 20 106 × 64 × 64 20 × 12 × 12 [2, 2, 0] 0.047 5.4 13 16 × 16 × 12 160 × 80 × 80 40 × 20 × 20 [2, 2, 0] 0.063 6.0


> **Table 1. Simulation parameters.**

Case Grid Base grid Refinement �xmin Steps Cores CPU no. (Structure) (Fluid) factors (· 1/rsph) [hr]

1 16 × 16 × 20 160 × 96 × 96 [2, 0, 0] 0.063 44 314 56 1 064 2 20 × 20 × 20 106 × 64 × 64 [2, 2, 0] 0.047 55 620 28 1 307 3 30 × 30 × 20 160 × 96 × 96 [2, 2, 0] 0.031 83 932 56 7 300 4 40 × 40 × 20 106 × 64 × 64 [2, 2, 2] 0.023 111 984 56 18 200


> **Table 2. Grid refinement study parameters.**

2.4. Model verification To assess the computational reliability of the present numerical model, we performed a grid-refinement study of a specific four-sphere configuration. The initial orientation for this configuration was 36◦-pitch/18◦-yaw (the convention for these angles is described at the beginning of § 4); this case was chosen as it exhibited many of the relevant interaction phenomena seen throughout the survey. The verification study consists of four simulations of successively refined fluid and structural domains; in our nomenclature, the numeric simulation labels (Cases 1–4) refer to the increasing degree of overall refinement in each case. Two base-grid resolutions, of dimensions 160 × 96 × 96 and 106 × 64 × 64, were used: in the coarsest simulation (i.e. Case 1), the finer base mesh was refined just once by a factor of two, yielding a minimum edge distance, �xmin, of 0.063rsph; while Case 2 achieved a �xmin value of 0.047rsph with two refinement passes of the coarse base mesh. Cases 3 and 4 added an extra level of refinement (of factor two) to each of these. With Case 2 representing the baseline resolution, the overall relative refinement levels of the other cases were 75 %, 150 % and 200 %. The solid meshes employed in each case were constructed to match the size of the finest fluid cells, and time-step sizes were automatically adjusted to satisfy a CFL condition of 0.8. All computations were run on a 56-core Dell Precision T7820 workstation: the coarsest simulation required ∼1000 CPU hr to complete and the finest simulation ∼18 000 CPU hr. Details of the grid dimensions, refinement strategies, and execution statistics can be found in table 2. Before quantifying the effects of varying resolution, we briefly describe the general behaviour exhibited by the spheres in the chosen configuration. For this arrangement, presented before release in figure 5(a), one body lies furthest upstream (sphere 1) and generates a bow shock that impinges on two bodies farther downstream (spheres 2 and 3), while a fourth sphere (sphere 4) is nearly aligned with sphere 1 in the streamwise direction and is therefore mostly shielded from the free stream flow. Following the visualizations of Case 4 (the right-hand column of figure 5), we see that the region of elevated surface pressure inboard on sphere 3 fosters immediate lateral repulsion (vertical, in this case). Sphere 2 experiences a broad band of high pressure near its centre and extending inboard; as it is repelled from the cluster, this irregular impingement pattern develops into the more

1029 A35-9

T. Whalen, R. Deiterding and S.J. Laurence

Case 1 Case 2 Case 3 Case 4

3

2

1

y z

4

(a)

(b)

(c)


> **Figure 5. Visualization sequences (looking from directly upstream) of all grid refinement cases in increments of 2τs with sphere surfaces coloured by pressure and the trajectory discrepancy of sphere 2 highlighted in (c).**

standard Edney Type-IV shock–shock interaction footprint (Edney 1968), which persists and broadens as the sphere moves downstream along a shock-surfing trajectory. Finally, spheres 1 and 4 engage in paired travel (to be described shortly): sphere 4 remains in contact with and in the aerodynamic shadow of sphere 1 for ∼2τs, before being pushed laterally outward from their mutual contact point. Now qualitatively comparing the highest-resolution Case 4 with the other cases visualized in figure 5, we first note that, while the motions of spheres 1 and 3 remain nearly identical across the grid survey, small discrepancies arise for sphere 4 and larger ones for sphere 2. In particular, although the trajectory of sphere 2 appears to be only weakly affected by reducing the resolution to Case 3, reducing it further leads to a decrease in the duration of shock surfing for this body, with an increased tendency towards entrainment within the shock layer of sphere 1. This is most clearly seen from the band of high pressure from shock impingement on this body at the last time step, which moves from the near the centre of the sphere for Case 4 to the outermost part of the sphere for Case 1. Indeed, the sensitivity of shock-surfing configurations to grid resolution has been already observed in Laurence et al. (2012), where it was attributed to a decreased effective bow shock radius generated by the upstream body at higher refinement levels. In figure 6, we quantify the errors accrued in the mean lateral velocities and force coefficients of the cluster. The lateral force coefficient, CT , is defined as CT = 2FT /ρ∞u2 ∞A, where FT is the lateral force and A = πr2 is the projected sphere area. Examining first the mean lateral velocity (figure 6a), the spheres of Case 3 appear to follow trajectories almost identical to those in the finest simulation, with the error remaining within 1 % for the majority of the flight. Cases 1 and 2, on the other hand, show elevated errors that are somewhat obfuscated by the residuals resulting from the series of collisions between spheres 1 and 4. Prior to a substantial divergence of the trajectories at ∼2.5τs (primarily from sphere 2, as discussed above), errors are limited

1029 A35-10

Journal of Fluid Mechanics

0.06

Case 1 Case 2 Case 3

Case 1 Case 2 Case 3 0.04

0.02

–0.02

–0.04

–0.06

0 1 2 3 4 0 1 2 3 4

0

0.06

0.04

0.02

–0.02

–0.04

–0.06

0

(a) (b)

t′ t′


$$
ε (� C′
$$

T)

ε (� VT ′)


> **Figure 6. Error in (a) mean lateral velocity and (b) mean lateral force coefficient relative to finest simulation**

(Case 4) for Cases 1–3; the end of primary separation phase is indicated with a dashed black line.

to ∼3 %, with no obvious differentiating trends between Cases 1 and 2 (though the lateral velocity errors for Case 1 deviate more sharply thereafter). The errors in the mean lateral force coefficient in figure 6(b) give a clearer picture of the purely aerodynamic contribution to these discrepancies. Here, the overall error experienced appears well-correlated to the refinement level of each simulation. Following the positive peak in force error (which is induced by a delayed abatement of repulsive forces on sphere 3), we observe approximately constant errors of 1.0 % for Case 1, 0.6 % for Case 2 and 0.1 % for Case 3 over a duration of approximately 1τs. As with the lateral velocity curves, however, diverging trajectories for the coarser simulations give rise to augmented errors thereafter. Nevertheless, despite the observed inconsistency of the various refinement cases after ∼2.5τs, the dynamical regime of most interest for these simulations is that during which the bodies are in close proximity; this so-called primary separation stage ends at t′ = 2.15 for the present configuration (the criterion for determining this will be described later). Clearly, the sphere trajectories are very nearly converged over this phase in Case 3, but the six-fold increase in computation time over Case 2 renders this choice (and even more so for Case 4) infeasible for parametric survey purposes. Instead, the modest force errors offered by refinement Case 2 (< 2 % over the primary separation), coupled with its significantly reduced computational expense, make it a suitable choice for the bulk of our numerical investigation. For the largest cluster population (N = 13), however, even this level of refinement resulted in a prohibitive computational cost, and the effective resolution was reduced to a level consistent with Case 1. While this may lead to limited accuracy for certain trajectories (particularly those involving shock surfing), we still expect the primary phase to be reasonably well captured at this refinement level.

2.5. Experimental validation To provide a further validation of the coupled numerical methodology, we compare results from an experiment involving separating free-flight spheres at Mach 6 with those from a simulation with fully matched conditions. The experiment (Shot 4A from Whalen & Laurence (2021)) features four spheres of diameter 6.35 mm arranged in a tetrahedron (as in the grid-refinement study of the previous subsection), with two leading spheres lying at similar streamwise locations and a pair of roughly aligned trailing spheres. Simulations of clusters with similar orientations produced extended shock surfing, a scenario highly sensitive to relative sphere positions and thus a challenging benchmark test. In figure 7, a sequence of images taken from both below and the side of the wind-tunnel test section (with backlit and shadowgraph set-ups, respectively) depict the sphere motions, along with a three-dimensional reconstruction of the extracted position in figure 7(c). Here, t = 0

1029 A35-11

T. Whalen, R. Deiterding and S.J. Laurence

(a) (b) (c)

t = 0.00 ms

t = 1.40 ms

t = 2.80 ms

t = 4.20 ms

t = 5.60 ms

Sphere 1 Sphere 2 Sphere 3 Sphere 4

y x z


> **Figure 7. Separation sequence of the four-sphere validation experiment with (a) vertical standard camera, (b) horizontal shadowgraph camera and (c) positional reconstruction. Markers in (a) and (b) indicate numerical sphere positions.**

represents the point from which we draw the initial kinematics for the counterpart numerical simulation, chosen such that the shells in which the spheres were initially suspended had separated sufficiently so as not to influence the sphere motions further. As shown in the t = 1.40 ms frames, the two leading spheres experience strong repulsive forces and separate laterally almost immediately, which exposes the trailing spheres to the free stream flow. Indeed, in the shadowgraph visualization of t = 2.80 ms, impingement of the shock from sphere 1 on sphere 4 is apparent and the resulting drag augmentation is reflected in the subsequent increased positional separation between the two. It appears that spheres 1 and 2 have ceased aerodynamic interaction by frame t = 2.80 ms, while spheres 3 and 4 remain under the influence of sphere 1 for an extended period of time. From the three-dimensional reconstruction, sphere 3 has begun to fall into the wake of sphere 1,

1029 A35-12

Journal of Fluid Mechanics

6

5

4

2

ε (x⃗/rsph)

1

0 t (ms) 2 4 6 0 t (ms) 2 4 6

3


$$
ε (VT ′)
$$

(a) 0.8

0.6

0.4

0.2

(b)

Sphere 1 Sphere 2 Sphere 3 Sphere 4


> **Figure 8. (a) Positional error between computation and experiment normalized by the sphere radius and (b) error in non-dimensional lateral velocity.**

while the shock–shock interaction on sphere 4 moves farther towards its inboard side (see figure 7b for t = 4.20 and 5.60 ms), indicating an eventual expulsion trajectory. The complementary simulation (performed with a resolution consistent with verification Case 2 above) exhibits qualitatively similar sphere-separation behaviour, as evidenced in the projected numerical sphere positions in figure 7(a,b). Note that the spheres are assigned non-zero initial velocities at t = 0 (in each case limited to less than 0.0015u∞) to match the experimental conditions at the same time; this was achieved by applying brief impulses to the spheres synchronous with the commencement of the coupled portion of the computation. As a quantitative measure of the agreement between the experiment and simulation, we present a comparison of total positional errors (normalized by the sphere radius) in figure 8(a) and non-dimensional lateral velocity errors in figure 8(b). Most simulated spheres follow the same general paths as in the experiment, remaining within one diameter of the corresponding experimental positions, but sphere 4 shows significant deviation beginning at roughly t = 4 ms. Relative to the experiment, this sphere exhibits reduced displacement in the streamwise direction, while its lateral motion more closely follows that of sphere 1; indeed, computational visualizations showed that this sphere is eventually entrained into the wake of sphere 1 rather than expelled as in the experiment. We note also that the leading spheres are accelerated more slowly downstream in the computation, probably because of the lack of a viscous drag component. Nevertheless, despite the accrual of these positional errors, the lateral velocities of spheres 1, 2 and 3 do not diverge significantly from the experimental results at later times, with stable errors of 0.038υs, 0.099υs and 0.040υs, respectively, at t = 5 ms. Considering that the separation velocity is the quantity of primary interest here, these errors can be considered sufficiently low for confidence in the equivalence between the simulated and experimental realizations of the separation process. The discrepancy with sphere 4, however, once again highlights the sensitivity of the sphere motions to the exact shock location in impinging configurations.

3. Two-sphere survey The simplest sphere-separation configuration is two equal-sized, initially touching spheres at various alignment angles. The aerodynamics of this configuration have been well characterized in previous studies (Laurence et al. 2007; Laurence & Deiterding 2011; Register et al. 2020), but the effects of surface contact between the pair have not yet

1029 A35-13

T. Whalen, R. Deiterding and S.J. Laurence

y x (a)

(b)

(c)


> **Figure 9. Sphere-pair separation sequences for initial alignment angle of (a) 90◦, (b) 120◦and (c) 172.5◦, in increments of 0.61τs, 1.20τs and 2.11τs, with colouring by surface pressure and centreline pseudoschlieren.**

been systematically investigated. We thus begin our investigation by elucidating the physics governing sphere-pair separation for various initial alignment angles. We adopt the convention whereby θ represents the alignment angle between the flow velocity vector and the line connecting the sphere centres, with an angle of 180◦indicating that the secondary (downstream) sphere is directly behind the primary (upstream) sphere. The simulated configurations span initial alignment angles, θ0, of 90◦to 180◦in increments of 7.5◦. To acquaint the reader with the basic qualitative aspects of sphere separation, we first present visualization sequences of several representative cases that demonstrate typical sphere-pair behaviours, some of which arise in more populous clusters. Figure 9(a) illustrates the dynamics resulting from a well-studied initial arrangement, two bodies positioned at the same streamwise coordinate (i.e. θ0 = 90◦). The spheres are initially subjected to pressures of similar magnitude to the stagnation-point value on their inboard regions as a result of their common bow shock. The resulting spanwise separation produces a bifurcation of the bow shock and an associated reduction in the extent of the region of elevated inboard pressures; by the fourth frame, they are effectively travelling independently of one another. Rotating the pair to an alignment angle of 120◦, the separating action of the common bow shock gives way to dynamics dictated by shock

1029 A35-14

Journal of Fluid Mechanics

impingement on the downstream sphere and the shock-surfing behaviour first noted by Laurence & Deiterding (2011). This significantly augments the streamwise and lateral velocity of the secondary body, but other than the resulting streamwise separation, the ultimate terminal behaviour of the pair relative to their common centre of mass is not all that different from the first case. The same can be said of figure 9(c), corresponding to θ0 = 172.5◦, though the means by which this final lateral separation is achieved is again quite different. In this configuration, the primary body experiences a higher drag than the secondary body, which yields a destabilizing moment. The resulting ‘rolling’ motion of the spheres is initially imperceptible but accelerates as the spheres’ inclination to the free stream grows, and by the fourth frame has led to a termination of the contact. The secondary body subsequently passes rapidly through the shock of the primary, before the spheres achieve aerodynamic independence in a manner similar to the lower θ0 arrangements. However, in contrast to the 120◦case, the primary sphere accrues more lateral momentum than the secondary, the lateral position of the latter remaining nearly stationary over the simulation; this signifies opposing trends for the motion of the system’s centre of mass for these two cases. For the θ0 = 135◦arrangement visualized in figure 10, however, a dramatic change in the spheres’ motions occurs. As in the 120◦configuration, the secondary sphere is subjected to the bow shock of the primary, but shock impingement now occurs on the upper-half of the body, which promotes prolonged contact between the bodies. Indeed, the bodies travel downstream in tandem, maintaining mechanical contact, and, due to an effective moment provided by the shock impingement, begin to rotate in the +θ direction about their common centre of mass. The alignment angle of the pair reaches a maximum in the fourth frame and, as evidenced by the reappearance of the high-pressure impingement region in the fifth frame, decreases thereafter, suggesting stable cyclical behaviour. At the same time, the effective angularity of the tandem bodies results in a common lift force (in this case in the negative y direction) which drives the pair in the lateral direction. In contrast to the previously explored scenarios, this potentially stable lifting behaviour relies on both the aerodynamics and contact mechanics of the constituent bodies and could represent an important mechanism for enhancing the lateral momentum of more populous clusters. Having established the general trends associated with sphere-pair aerodynamics, we now present in figure 11(a) the trajectories of all secondary (downstream) spheres in the polar coordinate system of the primary (where the vertical coordinate represents the normalized edge-to-edge separation of the spheres); the time histories of the lateral sphere velocities from the system’s centre of mass are meanwhile shown in figure 11(b). The behaviour near alignment angles of θ0 = 90◦closely conforms to expectations of mutual repulsion examined earlier, with the spheres achieving separation velocities of ∼0.2 over their ∼1.7τs interacting flight duration. As evidenced by the post-release increase in alignment angle for higher θ0, the transition from mutual repulsion to shock-surfinginduced separation occurs gradually between alignment angles of 90◦and 120◦and causes an appreciable enhancement to the lateral spread of the bodies, with terminal V ′ T values rising monotonically from 0.2 to 0.25 and the separation time scale extending to ∼4τs. At 127.5◦, the secondary sphere exhibits brief surfing before becoming entrained in the wake of and then colliding with the leading sphere (the brief excursion into a negative displacement is caused by deformation of the bodies upon collision). This simulation does not resolve the ultimate trajectories of the spheres, however, leaving open the possibility that a prolonged series of collisions is a sustainable configuration. Spheres initially positioned in 135◦and 142.5◦arrangements, in contrast, remain in contact for the duration of the simulations; in both cases, restorative moments drive the angular alignment of the pair about an apparent equilibrium θ. Indeed, for these two

1029 A35-15

T. Whalen, R. Deiterding and S.J. Laurence

y

x


> **Figure 10. Sphere-pair separation sequence from initial alignment angle of 135◦with colouring by surface**

pressure and centreline pseudoschlieren; panels are shown in increments of 1.83τs.

cases, both the surface separation of the spheres and lateral velocity of the system remain centred around zero, with lowand high-frequency residual oscillations representing the bulk rotational oscillations and the elastic mechanical vibrations from surface contact, respectively. Further examination of the forces and moments revealed that this restorative

1029 A35-16

Journal of Fluid Mechanics

100 120


$$
θ (deg.)
$$

140 160 180 0 2 4 6 8

0.3

0.2

0.1

–0.1

–0.2

0

2.0

1.5

1.0

0

0.5

VT

′

(a) (b)

t′


$$
∥x⃗sep∥/rsph
$$


> **Figure 11. (a) Polar trajectory map and (b) lateral-velocity histories in two-sphere survey. The inset in (a)**

shows only the 135◦and 142.5◦trajectories.

effect acts over a range of angles between approximately 132◦and 145.7◦, representing the extent of the stable region for the two-sphere dynamics. As the initial orientation is rotated further towards alignment with the free stream, the trailing sphere tends to exhibit the behaviour exemplified in the lower sequence of figure 9, ‘rolling’ along the surface of the primary until losing contact and undergoing aerodynamic separation. The 150◦ case features a secondary sphere that remains on a persistent shock-surfing trajectory, albeit without much enhancement to its lateral velocity. On the other hand, significant augmentation of the lateral velocity by shock-surfing following the loss of surface contact is observed for cases 157.5◦–172.5◦; in these instances, the angular momentum accrued by the pair endows the secondary sphere with enough lateral momentum to transit the impinging shock. The duration of contact here is somewhat dependent on the initial angle, ranging from ∼4 to 6τs (figure 11b), while the loss of contact in all cases appears to occur near 130◦(figure 11a). For an initial alignment angle of 180◦, no discernible change in the pair’s attitude was recorded, suggesting the presence of an equilibrium position (though we expect this to be unstable, given the higher drag of the upstream body in this configuration). Noticeably absent in figure 11(a) are secondary spheres that persist in the wake of the primary, which follows from the reduced drag and subsequent collisions that occur in such arrangements. This seems to signify three permitted trajectories – immediate separation, delayed separation following a period of contact, and indefinite contact – with sphere pairs in the former two categories achieving lateral velocities generally between 0.2 and 0.25. The final lateral velocities are shown in the left-hand part of figure 12(b), with this bimodal behaviour manifesting itself in the two distinct sets of values.

A particularly important consequence of stable sphere contact is the effective lift resulting from the pair’s asymmetry. Thus, treating the sphere pair as a binary rigid body, the lift-to-drag ratio experienced mutually by the sphere pair is shown in figure 12(b). This single curve was created by stitching together data over various angle ranges from multiple simulations (note that the region between 172.5◦and 180◦is linearly interpolated). We find a maximum L/D of 0.22 at θ = 141.5◦followed by an approximately linear decrease. While the mean L/D of 0.197 within the stable region is quite low, even for hypersonic aerodynamics (Anderson 2006), unlimited enhancement to the lateral velocity of the system would be theoretically possible. More likely, however, is the occurrence of limited contact, during which significant mutual momentum can still be accumulated. Indeed, for simulations with θ0 values of 150◦, 157.5◦and 172.5◦, the terminal centre-of-mass V ′

T,com values are 0.42, 0.32 and 0.28, respectively, higher than

1029 A35-17

T. Whalen, R. Deiterding and S.J. Laurence

100 120 θ0 (deg.) 140 160 180 130 140 150 θ (deg.) 160 170 180

0.25

0.20

0.15

0.05

0

–0.05

0.10

0.25

0.20

0.15

0.05

0

0.10 VT ′

(a) (b)

L/D


> **Figure 12. (a) Final lateral velocity with initial alignment angle; (b) lift-to-drag ratio of contacting sphere pairs, with the 132◦and 145.7◦stability limits indicated by dashed lines.**

the individual fragment velocities relative to this centre of mass. Thus, this lifting-pair contact mechanism may constitute an important aspect of equal-sphere separation in more populous settings, as has been confirmed by the experiments of Whalen & Laurence (2021).

4. Four-sphere tetrahedral arrangements 4.1. General characteristics and influence of initial sphere position The four-equal-sphere experiments of Whalen & Laurence (2021) illustrated through a small number of representative cases the division of sphere dynamics into distinct primary and secondary regimes: we now conduct a simulation survey of the same tetrahedral geometry to extract a more detailed description of sphere separation from this configuration. Compared with the two-sphere case, where a single angle (θ0) was sufficient to prescribe the initial arrangement, we now require two angles. We use the pitch and yaw angle for this purpose, the separation behaviour being degenerate under roll variations. We adopt a convention such that 0◦-pitch/0◦-yaw corresponds to three spheres forming a streamwise-normal plane behind a single leading sphere. The simulated parameter space spans angles of −36◦to 54◦in pitch and 0◦to 90◦in yaw at intervals of 18◦, with several additional tests performed to capture the full extent of separation behaviours. First, to highlight some of the governing dynamics of the four-body separation scenario, we review the features of a typical separation sequence. In figure 13, we present a set of snapshots visualizing instantaneous sphere positions coloured by surface pressure and the associated primary shock structures on the first refinement level from a cluster with an initial orientation of 36◦-pitch/18◦-yaw; corresponding lateral velocities and force coefficients are shown in figure 14. This is the same configuration employed in the grid refinement study of § 2.4, but now explored in additional detail. In the cluster’s initial state, an encompassing bow shock generated primarily by sphere 1 impinges on spheres 2 and 3, while sphere 4 is shielded in sphere 1’s wake. The shock–shock interaction on the inboard side of sphere 3 results in the previously noted region of high surface pressure on both spheres 1 and 3; the relatively large lateral force coefficient on sphere 3 (CT ≈0.45) promotes immediate repulsion, with this sphere becoming aerodynamically independent from the rest of the cluster by t′ = 2. Sphere 2 resides at a location farther downstream in the cluster and so is initially subjected to shock impingement from sphere 1 that transitions

1029 A35-18

Journal of Fluid Mechanics

3

2

1

4

y z


> **Figure 13. Visualizations of spheres separating from a 36◦-pitch/18◦-yaw tetrahedral cluster, with surfaces coloured by pressure and primary shock structure visualized in grey; blacked-out bodies have left the computational domain. Images are in steps of t′ = 0.94.**

0 2 1 4 3 5 0 2 1 4 3 5

0.6

0.4

0.2

0

–0.2

0.5

0.4

0.3

0.2

0

0.1

–0.1

CT, C�T

(a) (b)

Sphere 1 Sphere 2 Sphere 3 Sphere 4

t′ t′

V′ T, V�′ T , V′ T,com


> **Figure 14. (a) Lateral velocities and (b) lateral force coefficients for the above four-sphere cluster, coloured by sphere number: solid black V ′ T or CT ; dashed black V ′ T,com.**

to a swept shock–shock interaction between spheres 1 and 3. The resulting region of higher pressure occurs towards the front of the body, which induces augmented drag and increases the initial impulse in the streamwise direction, while maintaining a highly elevated lateral force coefficient beyond t′ = 1. Sphere 2 subsequently follows an extended shock surfing trajectory before becoming entrained in the bow shock of sphere 1. Sphere 4, on the other hand, is located initially in the wake of sphere 1 and experiences highly reduced drag, which initiates a prolonged series of collisions before it ‘rolls’ away from sphere 1, as in § 3. The primary separation phase lasts until t′ = 1.65, as inferred from the mean lateral force coefficient, at which point the collective lateral velocity has reached 0.27 and changes little thereafter. The centre-of-mass velocity, in contrast, continues to grow substantially during the secondary phase, reaching a value of 0.2 that is comparable to the collective velocity. While the spheres have not all reached their final aerodynamic state before exiting the computational domain (which is true of several simulations in this survey), the collective quantities – V ′ T and V ′ T,com – have essentially plateaued and their final simulated values can be considered generally representative of their terminal states.

1029 A35-19

T. Whalen, R. Deiterding and S.J. Laurence

0 10 5 0 10 5 0 10 5

1.0 θ0 = 0° – 15° θ0 = 15° – 30° θ0 = 30° – 45°

θ0 = 45° – 60° θ0 = 60° – 75° θ0 = 75° – 90°

θ0 = 90° – 105° θ0 = 105° – 120° θ0 = 120° – 135°

θ0 = 135° – 150° θ0 = 150° – 165° θ0 = 165° – 180°

0.5

0 1.0

0.5

0 1.0

0.5

0 1.0

0.5

0

VT

′

t′


> **Figure 15. Time-series of lateral velocities of spheres in tetrahedral clusters binned by initial polar angle.**

In the two-sphere case, the parameter governing the resultant dynamics is the initial polar alignment angle, and we might expect the initial position of a given sphere within its tetrahedral cluster to also play an important role in the present case. We thus bin all spheres over the survey into groups of 15◦width by initial polar angle within the cluster and compute the magnitude of the lateral velocity of each body relative to its cluster’s centre of mass. In figure 15, the stacked time series of grouped lateral velocities show a clear dependence of trajectory on initial polar angle, θ0. First, spheres positioned towards the front of the formation (0◦–30◦) tend to experience mild separation velocities, likely because the downstream spheres are largely incapable of influencing their dynamics. At intermediate forward positions (30◦–60◦), the separation characteristics seem dominated by collisions with other spheres, which manifest themselves as discontinuous jumps in V ′

T . This category is mainly occupied by the leader of a sphere pair repelled in tandem (e.g. spheres 1 and 4 in figure 13), and the enhancement in V ′

T through the two-sphere subcluster interaction can reach values of ∼0.2, which is consistent with the results of § 3. Spheres located slightly upstream of the flank (60◦–90◦), in contrast, are consistently subject to immediate expulsion or shock surfing, reaching relatively uniform and high lateral velocities close to 0.7. Polar angles above 90◦mark a transition from expulsion to entrainment trajectories, although the notable variation in separation velocities for θ0 = 90◦–120◦indicates some level of dependence on geometry-specific properties. Just as bins 30◦–60◦constitute in many cases the upstream (primary) bodies in a sphere pair, the spheres in bins 120◦–165◦represent their downstream (secondary) counterparts; these trajectories, too, are marked by collisions and a delayed increase in lateral velocity typical of the ‘rolling’ interactions. Finally, in the rear of the formation, rotational symmetry limits the lateral separation velocities to modest values.

Using these binned trajectories, we can now examine how well a sphere’s initial polar angle within the cluster predicts its final separation velocity. Defining the final velocity as that attained when either the sphere reaches the limits of the computational domain or the simulation ends, we present the mean lateral velocity of each binned group, along with

1029 A35-20

