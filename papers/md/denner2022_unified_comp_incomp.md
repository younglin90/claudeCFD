
## A unified algorithm for interfacial flows with incompressible and compressible fluids


![Equation](images/2201.07447_eq001.png)

Chair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany

Abstract

The majority of available numerical algorithms for interfacial two-phase flows either treat both fluid phases as incompressible (constant density) or treat both phases as compressible (variable density). This presents a limitation for the prediction of many two-phase flows, such as subsonic fuel injection, as treating both phases as compressible is computationally expensive due to the very stiff pressure-density-temperature coupling of liquids. A framework with the capability of treating one phase compressible and the other phase incompressible, therefore, has a significant potential to improve the computational performance and still capture all important physical mechanisms. We propose a numerical algorithm that can simulate interfacial flows in all Mach number regimes, ranging from M = 0 to M > 1, including interfacial flows in which compressible and incompressible fluids interact, within the same pressure-based framework and conservative finite-volume discretisation. For interfacial flows with only incompressible fluids or with only compressible fluids, the proposed pressure-based algorithm and finite-volume discretisation reduce to numerical frameworks that have already been presented in the literature. Representative test cases are used to validate the proposed algorithm, including mixed compressible-incompressible interfacial flows with acoustic waves, shock waves and rarefaction fans.

Keywords: Interfacial flows, All-speed flows, Finite-volume method, Volume-of-Fluid method, Shock capturing

1. Introduction

Numerical methods and algorithms for interfacial flows have, so far, typically been developed either for the simulation of compressible fluids or the simulation of incompressible fluids, as a result of the different dominant physical mechanisms, the different mathematical characteristics of the governing equations and the different numerical challenges at different flow speeds. While in reality all fluids are compressible, with a compressibility given as β = dρ/(ρ dp), where ρ is the density and p is the pressure, a frequent assumption in modelling fluid flows is that the fluid is incompressible (dρ = 0). From a numerical viewpoint, the distinction between incompressible and compressible fluids is important and the Mach number M = U/a, with U the flow speed and a the speed of sound, is of particular importance, as it determines the mathematical nature of the governing conservation laws. For incompressible fluids the density is constant along the fluid particle trajectories and the speed of sound is a →∞, with M →0, whereas for compressible fluids the density is variable (dρ ̸= 0), with a finite speed of sound (0 < a < ∞) and Mach number (0 < M < ∞). For large flow speeds (M > 0.1) pressure and density are strongly coupled, especially for supersonic flows, whereas this pressure-density coupling diminishes in the incompressible flow regime (M →0), where density changes vanish (dρ →0). This acoustic degeneration as well as the pressure-velocity coupling present the major challenges for the modelling of flows with M < 0.1, whereas the stability in different Mach number regimes and the robust resolution of discontinuities are pertinent issues for the modelling of flows with M > 0.1. These different mathematical characteristics and numerical requirements has

∗fabian.denner@ovgu.de

1


# arXiv:2201.07447v1  [physics.comp-ph]  19 Jan 2022

made the development of algorithms that can simulate mixed compressible-incompressible two-phase fluids a difficult task. We propose a fully-coupled pressure-based algorithm for interfacial flows at all speeds, including compressible and incompressible fluids, in which the discretised governing equations are solved for pressure, velocity and temperature. This algorithm features a conservative finite-volume discretisation of the governing equations that is identical for incompressible fluids (dρ = 0) and compressible fluids (dρ ̸= 0) [1]. Through an appropriate linearisation, the discretised continuity equation serves as a transport equation for the density in the case of a compressible fluid, with density formulated as a function of pressure by an equation of state, and as a constraint on the velocity field for incompressible fluids, with pressure acting as a Lagrange multiplier. Hence, the proposed algorithm is applicable to incompressible, compressible and, crucially, mixed compressible-incompressible interfacial flows in all Mach number regimes. The bulk phases are represented and advected using a Volume-of-Fluid method (VOF), and the bulk phases are coupled at the interface using the acoustically-conservative interface discretisation (ACID) method [2], which was originally proposed for fully compressible interfacial flows. Results of representative test-cases, including flows with surface tension, the pressuredriven collapse of a bubble and a shock-drop interaction, are presented to validate and scrutinise the proposed algorithm for compressible-incompressible interfacial flows.

2. State of the art

As a consequence of the very different mathematical characteristics and numerical requirements of the simulation of compressible and incompressible fluids, as explained above, two different classes of algorithms have emerged: density-based algorithms and pressure-based algorithms. In the following, the main developments of both classes of algorithms related to interfacial flows are briefly reviewed in Sections 2.1 and 2.2, respectively, and, in Section 2.3, currently available numerical algorithms able to simulate mixed compressible-incompressible interfacial flows are discussed.

2.1. Density-based algorithms

Density-based algorithms, where the sought primary variables are the density, momentum and energy of the flow, are widely applied for compressible flows. With respect to the modelling of compressible interfacial flows, notable examples are the two-fluid Baer-Nunziato (BN) model [3], where each fluid is represented by its own set of governing equations, and the five-equation models [4, 5], with a separate continuity equation for each phase, and shared momentum and energy equations. In these algorithms, an exact or approximate Riemann solver is employed to evaluate the fluxes at cell-faces of the computational mesh [6–8]. The ghost-fluid method (GFM) [9] has established itself as an alternative to solving a Riemann problem, partly due to its conceptual simplicity. Recent developments have also seen a combination of GFM with Riemann solvers [10, 11] to improve the stability of simulations with strong shock-interface interactions and compressible gas-liquid flows. Since pressure is not directly solved for with a density-based algorithm, the pressure field has to be reconstructed based on the applied thermodynamic model via an equation of state, which may be difficult in interfacial cells where two bulk phases coexist [4, 5, 12], and which requires hydrodynamically and thermodynamically plausible fluid properties. While density-based algorithms are naturally suited for compressible flows, they are poorly suited for low-Mach number flows [13, 14], where the coupling of pressure and density vanishes. Although density-based algorithms have been applied to low-Mach number flows with some success, this requires computationally expensive preconditioning techniques [15]. A different approach for single-phase flows at all speeds was pursued by van der Heul et al. [16], in which the energy equation is reformulated as an equation for pressure, while the continuity equation still serves as an equation for density. Similar methods were subsequently presented by Park and Munz [17] and Cordier et al. [18]. These algorithms formally converge to the incompressible flow equations in the limit of zero Mach number (M →0) [16, 17], with the reformulated energy equation enforcing a divergence-free velocity field. Boger et al. [19] extended the work of Park and Munz [17] to interfacial flows.

2

2.2. Pressure-based algorithms

Pressure-based algorithms, in which the continuity equation is cast into a discretised equation for pressure (with pressure acting as the Lagrange multiplier that enforces ∇· u →0 for M →0), while density is either constant (incompressible fluids) or evaluated using a suitable equation of state (compressible fluids), are preferably applied to simulate the flow of incompressible fluids and weakly compressible flows, and may yield significant advantages for low-Mach number compressible flows, since the acoustic degeneration at low Mach numbers does not pose a problem [20]. The success of pressure-based algorithms is facilitated by the unique role of pressure in all Mach number regimes [20], with the pressure-velocity coupling dominant at low Mach numbers and the pressure-density coupling dominant at high Mach numbers [21, 22], and the convenient fact that the fully-conservative formulation of the governing conservation laws can still be satisfied accurately [1, 21]. The majority of pressure-based algorithms for incompressible interfacial flows are founded on pressure-correction methods, such as projection methods [23, 24], the SIMPLE method [25] and its subsequent derivatives, or the PISO method [26], or fully-coupled algorithms [27]. Pressure-based algorithms for compressible flows are much less prominent in the literature than their density-based counterparts, partly because deriving stable and efficient numerical schemes for the transonic regime [28] as well as formulating consistent shock-capturing schemes have proven difficult for pressurebased algorithms [14, 29]. Nonetheless, because pressure plays an important role in all Mach number regimes, pressure-based algorithms have a distinct potential for applications in all Mach number regimes or in which the Mach number varies strongly. Starting with the work of Harlow and Amsden [30], a variety of pressure-based algorithms for compressible single-phase flows has been developed [13, 21, 28, 31, 32]. Denner et al. [1] presented a fully-coupled pressure-based algorithm able to predict single-phase flows of incompressible, ideal-gas and real-gas fluids at all speeds (0 ≤M ≤239) accurately and robustly, using the same fully-coupled pressure-based algorithm and conservative finite-volume discretisation. Despite these developments for single-phase flows, it was only recently that the first pressurebased algorithm in conservative form for compressible interfacial flows at all speeds was proposed [2]. This algorithm was proposed in conjunction with a new interface discretisation method that retains the acoustic features of the flow, without the need to employ a Riemann solver, and has been shown to be a promising alternative to traditional density-based algorithms, with the pressure-based formulation of the governing equations facilitating the definition of consistent mixture rules at the interface that apply naturally to flows in all Mach number regimes. This algorithm was subsequently extended to polytropic interfacial flows [33], i.e. treating the flow as a polytropic process.

2.3. Algorithms for compressible-incompressible flows

Caiden et al. [34] were the first to propose a numerical algorithm dedicated to the simulation of general compressible-incompressible flows, solving different governing equations for the compressible fluid and the incompressible fluid, coupling the bulk phases at the interface using the GFM. A similar method to describe the behaviour of compressible bubbles in an incompressible fluid was subsequently presented by Aanjaneya et al. [35]. Wadhwa et al. [36] proposed a method for computing incompressible liquid drops in a compressible gas, representing the interface as a matching moving mesh and solving the flow of the incompressible fluid using the artificial compressibility method of Chorin [23]. Others proposed algorithms for compressible-incompressible flows applying the same governing equations in non-conservative form in both fluids [37–39] and these algorithms are, in general, only applicable to incompressible and low Mach number flows. However, the application of the governing equations in conservative form together with a conservative discretisation is a prerequisite for an accurate prediction of flows in all Mach number regimes [40], in particular shock waves, rarefaction fans and contact discontinuities. The density-based algorithm of Boger et al. [19] is readily applicable to interfacial flows in all Mach number regimes, including the incompressible limit (M →0), yet incompressible fluids (M = 0) have not been considered in the context of such an algorithm. In addition to the difficulties encountered when developing numerical methods for interfacial flows, such as a precise discrete force balance between surface tension and the pressure gradient, Caiden et al. [34] and Billaud et al. [37] identified the transmission of waves at the interface, as well as physically realistic and compatible discrete formulations of the governing equations describing the

3

incompressible fluid and the compressible fluid, as the main additional difficulties associated with an accurate prediction of compressible-incompressible interfacial flows. The presented algorithm addresses and resolves these difficulties, as demonstrated by the chosen validation cases.

3. Governing equations

The conservation laws governing both incompressible and compressible fluid flow at all speeds are the conservation of mass, momentum and energy, given as


![Equation](images/2201.07447_eq002.png)


![Equation](images/2201.07447_eq003.png)


![Equation](images/2201.07447_eq004.png)


![Equation](images/2201.07447_eq005.png)


![Equation](images/2201.07447_eq006.png)


![Equation](images/2201.07447_eq007.png)


![Equation](images/2201.07447_eq008.png)

respectively, where t is time, u is the velocity vector, p is pressure, ρ is the density and h is the specific total enthalpy. The stress tensor τ for the considered Newtonian fluids is given as


![Equation](images/2201.07447_eq009.png)


![Equation](images/2201.07447_eq010.png)

where µ is the dynamic viscosity. The heat flux due to thermal conduction is typically described by Fourier’s law as q = −k∇T, with k the thermal conductivity and T the temperature. All external forces applied to the flow, e.g. the force due to surface tension, are gathered in the volumetric source term S. The Volume-of-Fluid (VOF) method [41] is adopted to capture the fluid interface between two immiscible bulk phases, applying an indicator function field ζ, defined as


![Equation](images/2201.07447_eq011.png)


![Equation](images/2201.07447_eq012.png)

where Ωa and Ωb are the subdomains occupied by fluid a and fluid b, respectively, and Ω= Ωa ∪Ωb is the computational domain. Because the interface is a material front propagating with the flow, the indicator function ζ is advected with the underlying fluid velocity by the advection equation


![Equation](images/2201.07447_eq013.png)


![Equation](images/2201.07447_eq014.png)

4. Thermodynamic closure

The governing conservation laws given above require closure by an appropriate thermodynamic model, defining the thermodynamic properties of the fluids. Following the approach recently proposed by Denner et al. [1], the density ρ and the specific isobaric heat capacity cp are defined by a set of input parameters (ρ0, cp,0, cv,0 and Π0) that enables the formulation of a unified thermodynamic closure for incompressible and compressible fluids. For a compressible fluid, the stiffened-gas model [42, 43] is applied, in which the density is defined as


![Equation](images/2201.07447_eq015.png)


![Equation](images/2201.07447_eq016.png)

where Π0 is a material-dependent pressure constant, R0 = cp,0 −cv,0 is the specific gas constant, with the constant reference specific isobaric heat capacity cp,0 and the constant reference specific isochoric heat capacity cv,0, and γ0 = cp,0/cv,0 is the heat capacity ratio. The specific isobaric heat capacity is given as [2]


![Equation](images/2201.07447_eq017.png)

4

and the specific total enthalpy is defined as h = cp T + u2/2. The speed of sound follows as

a =


![Equation](images/2201.07447_eq018.png)


![Equation](images/2201.07447_eq019.png)


![Equation](images/2201.07447_eq020.png)

For Π0 = 0, the stiffened-gas model describes a calorically perfect ideal gas, with ρ = p/(R0 T) and cp = cp,0. The density of an incompressible fluid is, by definition, constant and given as


![Equation](images/2201.07447_eq021.png)

with ρ0 a predefined density value. The specific isobaric heat capacity cp of an incompressible fluid is also assumed to be constant, with

cp = cp,0. (11)

In order to incorporate compressible and incompressible fluids in the same numerical framework, the definitions for the density ρ and the specific isobaric heat capacity cp are unified by the binary operator C, given as

C =


![Equation](images/2201.07447_eq022.png)

This binary operator is used as a coefficient for the compressible part and, analogously, 1 −C is used as a coefficient for the incompressible part of the unified closure model. The density for a given fluid is then defined, based on Eqs. (7) and (10), as


![Equation](images/2201.07447_eq023.png)


![Equation](images/2201.07447_eq024.png)

and the specific isobaric heat capacity is defined, based on Eqs. (8) and (11), as

cp = cp,0


![Equation](images/2201.07447_eq025.png)


![Equation](images/2201.07447_eq026.png)

The speed of sound is given, following Eq. (9), as

a = C


![Equation](images/2201.07447_eq027.png)


![Equation](images/2201.07447_eq028.png)


![Equation](images/2201.07447_eq029.png)

where a∞is a very large velocity (here: a∞= 1032 m s−1) that represents the infinite speed of sound of incompressible fluids and ensures a computationally meaningful definition of the speed of sound throughout the computational domain. The appropriate formulations of the density ρ, the specific isobaric heat capacity cp and the speed of sound a for incompressible fluids (C = 0), perfect-gas fluids (C = 1, Π0 = 0) and stiffened-gas fluids (C = 1, Π0 > 0) are readily recovered.

5. Numerical framework

The proposed numerical algorithm is based on a fully-coupled pressure-based algorithm for incompressible and compressible fluids with a finite-volume discretisation of the governing equations [1]. The numerical framework is predicated on a collocated variable arrangement, meaning that the primary solution variables p, u and T, as well as all fluid properties, are stored at the centre of the mesh cells. First, the finite-volume discretisation underpinning the numerical algorithm is briefly discussed (Section 5.1), followed by the definition of the advecting velocity (Section 5.2) and the discretisation of the governing equations (Section 5.3). The particular form the discretised governing equations assume in the specific case of the incompressible limit is discussed in Section 5.4 and the solution procedure is described in Section 5.5.

5


![Equation](images/2201.07447_eq030.png)


> **Figure 1: Schematic illustration of mesh cell P with its neighbour mesh cell Q and the shared mesh face f, where nf is the unit normal vector of face f, pointing out of cell P.**

5.1. Finite-volume discretisation Considering, for example, the advective-diffusive transport of a general fluid variable, φ, is given as


![Equation](images/2201.07447_eq031.png)


![Equation](images/2201.07447_eq032.png)

where t is the time, ρ is the fluid density, u is the fluid velocity and Γφ is the diffusion coefficient of φ. Reformulating Eq. (16) in its integral form with respect to the control volume V , given as ˚

V


![Equation](images/2201.07447_eq033.png)


![Equation](images/2201.07447_eq034.png)


![Equation](images/2201.07447_eq035.png)


![Equation](images/2201.07447_eq036.png)

allows a rather straightforward finite-volume discretisation of the advective-diffusive transport of φ. In the interest of simplicity and brevity, the mesh is henceforth assumed to be Cartesian with a local mesh spacing ∆x. Extending the presented methods to unstructured meshes is easily achieved by introducing corrections for mesh skewness and non-orthogonality, as for instance described in [1]. The transient term is discretised in the following using the Second-Order Backward Euler scheme, which is given for cell P as

˚

V


![Equation](images/2201.07447_eq037.png)


![Equation](images/2201.07447_eq038.png)

where Φ = ρφ, ∆t is the time-step, superscript (n + 1) denotes the implicitly sought solution, superscript (t −∆t) denotes the previous time-level and superscript (t −2∆t) denotes the previous previous time-level. The discrete form of the advection term is obtained by applying the divergence theorem, ˚

V ∇· (ρuφ) dV = ‹


![Equation](images/2201.07447_eq039.png)

where Σ is the outward-pointing surface vector on the surface ∂V of control volume V . Since the surface of a discrete control volume is constituted by a finite number of flat faces f of area Af, as illustrated in Fig. 1, and applying the midpoint rule [44, 45], the advection term follows as ‹


![Equation](images/2201.07447_eq040.png)


![Equation](images/2201.07447_eq041.png)


![Equation](images/2201.07447_eq042.png)

The face value φf is typically not readily available and has to be interpolated from the cell-centred values, which in a finite-volume sense represent the values of φ averaged over the respective cell volume, using an appropriate interpolation scheme. The advecting velocity ϑf = uf · nf, where nf is the normal vector of face f (pointing outwards with respect to cell P, see Fig. 1), represents the velocity normal to face f and forms, together with the face density ρf and the face area Af, the mass flux ˙mf = ρfϑfAf through face f. In the context of the presented algorithm, the advecting velocity ϑf is determined using a momentum-weighted interpolation from the cell-centred values of velocity and pressure, as further discussed in Section 5.2. Since the values of density and velocity reside at the cell centres, a suitable interpolation must be applied, such as upwind differencing, central differencing or a TVD scheme. Similar to the advection term, applying the divergence theorem in conjunction with the midpoint

6

rule, the diffusion term of the transport equation (17) is discretised as ˚

V ∇· (Γφ ∇φ) dV ≈ �


![Equation](images/2201.07447_eq043.png)

where the diffusion coefficient Γφ at face f is defined by a harmonic average of the cell-centred values as [46]


![Equation](images/2201.07447_eq044.png)

with cell Q the neighbour cell of cell P, as illustrated in Fig. 1. The face-centred gradient of φ along the normal vector nf is approximated with second-order accuracy as


![Equation](images/2201.07447_eq045.png)


![Equation](images/2201.07447_eq046.png)

where ∆x is the mesh spacing. Applying the divergence theorem, the spatial gradient of φ averaged over the cell volume VP is readily computed as


![Equation](images/2201.07447_eq047.png)


![Equation](images/2201.07447_eq048.png)


![Equation](images/2201.07447_eq049.png)


![Equation](images/2201.07447_eq050.png)

5.2. Advecting velocity

The momentum-weighted interpolation (MWI) is applied to define an advecting velocity ϑ = uf · nf at cell faces, which is used in the discretised advection terms of the governing equations. MWI provides a robust pressure-velocity coupling for flows with low Mach numbers and flows of incompressible fluids by introducing a cell-to-cell pressure coupling and by applying a low-pass filter acting on the third derivative of pressure [47, 48], thus avoiding pressure-velocity decoupling due to the collocated variable arrangement. Following the unified formulation of the MWI proposed by Bartholomew et al. [48], the definition of the advecting velocity includes modifications to the original formulation of the MWI, as introduced by Rhie and Chow [49], to account for large density ratios and source terms occurring in interfacial flows, and the transient nature of the considered problems. As demonstrated by Bartholomew et al. [48], only the driving pressure gradient ∇P = ∇p −S, which is the pressure gradient associated with the flow field, should be coupled to the velocity field, whereas source terms and other external contributions should not be coupled to the velocity field. Taking this into account, the advecting velocity ϑf at face f is given as


![Equation](images/2201.07447_eq051.png)


![Equation](images/2201.07447_eq052.png)


![Equation](images/2201.07447_eq053.png)


![Equation](images/2201.07447_eq054.png)


![Equation](images/2201.07447_eq055.png)


![Equation](images/2201.07447_eq056.png)


![Equation](images/2201.07447_eq057.png)


![Equation](images/2201.07447_eq058.png)


![Equation](images/2201.07447_eq059.png)


![Equation](images/2201.07447_eq060.png)

where the face velocities uf and u(t−∆t) f are obtained by linear interpolation, and


![Equation](images/2201.07447_eq061.png)


![Equation](images/2201.07447_eq062.png)

The face density ρ∗ f is interpolated by a harmonic average, which is necessary for a consistent definition of the coefficient of the pressure term as well as the efficacy of the density weighting [48]. The

7

coefficient ˆdf is defined as [27, 48]

ˆdf =


![Equation](images/2201.07447_eq063.png)


![Equation](images/2201.07447_eq064.png)


![Equation](images/2201.07447_eq065.png)


![Equation](images/2201.07447_eq066.png)


![Equation](images/2201.07447_eq067.png)


![Equation](images/2201.07447_eq068.png)


![Equation](images/2201.07447_eq069.png)

where eP and eQ are the sum of the coefficients of the primary variable u arising from the advection and shear stress terms of the momentum equations [1]. The MWI formulation given in Eq. (25) is independent of the applied time-step and the error in kinetic energy introduced by the MWI is proportional to ∆x3 [48]. Hence, the convergence of the second-order accurate finite-volume method remains unaffected. The density weighting applied to the cell-centred pressure gradient in Eq. (25) has been shown to yield robust results for flows with large and abrupt changes in density [48], demonstrated for incompressible interfacial flows with a density ratio of up to 1024 [27, 50]. Including the transient term has previously been shown to be important for a correct temporal evolution of pressure waves [32, 48], which is particularly pertinent for acoustic effects in compressible flows.

5.3. Discretised governing conservation laws

Since the discretisation of the governing conservation laws is identical for single-phase and interfacial flows, the following presentation of the discretisation of the governing conservation laws focuses on single-phase flows. The extension of this discretisation to interfacial flows by an appropriate definition of the fluid properties and using the ACID method is described in Section 6. Applying the numerical schemes described in the previous sections, the discretised continuity equation (1) for cell P is given as


![Equation](images/2201.07447_eq070.png)


![Equation](images/2201.07447_eq071.png)


![Equation](images/2201.07447_eq072.png)


![Equation](images/2201.07447_eq073.png)

where superscript (n + 1) denotes implicitly sought solutions. A Newton linearisation [28, 32] is applied to the advection term of the discretised continuity equation, given as

(ρf ϑf)(n+1) ≈ρ(n) f ϑ(n+1) f + ρ(n+1) f ϑ(n) f −ρ(n) f ϑ(n) f , (29)

where superscript (n) denotes the latest available solution. The Newton linearisation facilitates a smooth transition from low to high Mach number regions [21, 28] and provides an implicit pressurevelocity coupling for flows at low Mach numbers and flows of incompressible fluids [1, 32]. The discretised momentum equations (2) of cell P are given as


![Equation](images/2201.07447_eq074.png)


![Equation](images/2201.07447_eq075.png)


![Equation](images/2201.07447_eq076.png)


![Equation](images/2201.07447_eq077.png)


![Equation](images/2201.07447_eq078.png)


![Equation](images/2201.07447_eq079.png)

+ �


![Equation](images/2201.07447_eq080.png)


![Equation](images/2201.07447_eq081.png)


![Equation](images/2201.07447_eq082.png)


![Equation](images/2201.07447_eq083.png)


![Equation](images/2201.07447_eq084.png)


![Equation](images/2201.07447_eq085.png)

3


![Equation](images/2201.07447_eq086.png)


![Equation](images/2201.07447_eq087.png)


![Equation](images/2201.07447_eq088.png)


![Equation](images/2201.07447_eq089.png)


![Equation](images/2201.07447_eq090.png)


![Equation](images/2201.07447_eq091.png)


![Equation](images/2201.07447_eq092.png)

where □f denotes linear interpolation of the values at the adjacent cell centres and

(ρf ϑf uj,f)(n+1) ≈ρ(n) f ϑ(n) f u(n+1) j,f + ρ(n) f ϑ(n+1) f u(n) j,f


![Equation](images/2201.07447_eq093.png)

8

The discretised energy equation (3) of cell P is given as


![Equation](images/2201.07447_eq094.png)


![Equation](images/2201.07447_eq095.png)


![Equation](images/2201.07447_eq096.png)


![Equation](images/2201.07447_eq097.png)


![Equation](images/2201.07447_eq098.png)


![Equation](images/2201.07447_eq099.png)


![Equation](images/2201.07447_eq100.png)

+ �


![Equation](images/2201.07447_eq101.png)


![Equation](images/2201.07447_eq102.png)


![Equation](images/2201.07447_eq103.png)


![Equation](images/2201.07447_eq104.png)


![Equation](images/2201.07447_eq105.png)


![Equation](images/2201.07447_eq106.png)


![Equation](images/2201.07447_eq107.png)


![Equation](images/2201.07447_eq108.png)


![Equation](images/2201.07447_eq109.png)


![Equation](images/2201.07447_eq110.png)


![Equation](images/2201.07447_eq111.png)


![Equation](images/2201.07447_eq112.png)

3


![Equation](images/2201.07447_eq113.png)


![Equation](images/2201.07447_eq114.png)


![Equation](images/2201.07447_eq115.png)


![Equation](images/2201.07447_eq116.png)


![Equation](images/2201.07447_eq117.png)


![Equation](images/2201.07447_eq118.png)

+ u(n+1) i,P Si,P VP ,


![Equation](images/2201.07447_eq119.png)

with

(ρf ϑf hf)(n+1) ≈ρ(n) f ϑ(n) f h(n+1) f + ρ(n) f ϑ(n+1) f h(n) f


![Equation](images/2201.07447_eq120.png)

and

h(n+1) = cp T (n+1) + u(n),2


![Equation](images/2201.07447_eq121.png)

The advecting velocity ϑf is given by Eq. (25) and is the same in all equations, ensuring a consistent transport of the conserved quantities. Following the work of Ferziger [46], the viscosity µ∗ f and the thermal conductivity k∗ f at face f are given by a harmonic average of the values at the adjacent cell centres. Previous studies [2, 28] have demonstrated substantial improvements with respect to the performance and stability of the numerical solution algorithm associated with a Newton linearisation of all governing equations. In particular, such a linearisation enables a smooth transition from low to high Mach number regions [13, 28, 51]. To this end, the advection terms of the governing equations as well as the transient terms of the momentum and energy equations are discretised using a Newton linearisation, see Eqs. (31) and (33), as described in more detail in [28].

5.4. Incompressible limit

For flows in the incompressible limit, with M →0, the density is constant along the fluid particle trajectories [52], with


![Equation](images/2201.07447_eq122.png)


![Equation](images/2201.07447_eq123.png)


![Equation](images/2201.07447_eq124.png)

Consequently, the continuity equation is no longer effective as a transport equation for density, but becomes a constraint on the velocity field with ∇· u →0 [52], enforced by pressure which acts as a Lagrange multiplier [53]. Inserting Eq. (35) into Eqs. (1)–(3), the governing conservation laws for flows in the incompressible limit reduce to


![Equation](images/2201.07447_eq125.png)


![Equation](images/2201.07447_eq126.png)


![Equation](images/2201.07447_eq127.png)


![Equation](images/2201.07447_eq128.png)

Inserting ρ = ρ0, see Eq. (10), into the discretised governing conservation laws presented above, the discretised continuity equation (28) reduces to �


![Equation](images/2201.07447_eq129.png)

9

the discretised momentum equations (30) become


![Equation](images/2201.07447_eq130.png)


![Equation](images/2201.07447_eq131.png)


![Equation](images/2201.07447_eq132.png)


![Equation](images/2201.07447_eq133.png)


![Equation](images/2201.07447_eq134.png)


![Equation](images/2201.07447_eq135.png)


![Equation](images/2201.07447_eq136.png)


![Equation](images/2201.07447_eq137.png)


![Equation](images/2201.07447_eq138.png)


![Equation](images/2201.07447_eq139.png)

+ �


![Equation](images/2201.07447_eq140.png)


![Equation](images/2201.07447_eq141.png)


![Equation](images/2201.07447_eq142.png)


![Equation](images/2201.07447_eq143.png)


![Equation](images/2201.07447_eq144.png)


![Equation](images/2201.07447_eq145.png)


![Equation](images/2201.07447_eq146.png)


![Equation](images/2201.07447_eq147.png)


![Equation](images/2201.07447_eq148.png)


![Equation](images/2201.07447_eq149.png)


![Equation](images/2201.07447_eq150.png)


![Equation](images/2201.07447_eq151.png)

and the discretised energy equation (32) becomes


![Equation](images/2201.07447_eq152.png)


![Equation](images/2201.07447_eq153.png)


![Equation](images/2201.07447_eq154.png)


![Equation](images/2201.07447_eq155.png)


![Equation](images/2201.07447_eq156.png)


![Equation](images/2201.07447_eq157.png)


![Equation](images/2201.07447_eq158.png)


![Equation](images/2201.07447_eq159.png)


![Equation](images/2201.07447_eq160.png)


![Equation](images/2201.07447_eq161.png)


![Equation](images/2201.07447_eq162.png)


![Equation](images/2201.07447_eq163.png)

+ �


![Equation](images/2201.07447_eq164.png)


![Equation](images/2201.07447_eq165.png)


![Equation](images/2201.07447_eq166.png)


![Equation](images/2201.07447_eq167.png)


![Equation](images/2201.07447_eq168.png)


![Equation](images/2201.07447_eq169.png)


![Equation](images/2201.07447_eq170.png)


![Equation](images/2201.07447_eq171.png)


![Equation](images/2201.07447_eq172.png)


![Equation](images/2201.07447_eq173.png)


![Equation](images/2201.07447_eq174.png)


![Equation](images/2201.07447_eq175.png)


![Equation](images/2201.07447_eq176.png)

These are precisely the governing conservation laws of the incompressible limit, Eqs. (36)–(38), discretised with the schemes as discussed above. In addition, Eqs. (39) and (40) are identical to the discretised continuity and momentum equations of the fully-coupled pressure-based algorithm for incompressible interfacial flows of Denner et al. [27].

5.5. Solution procedure

The discretised governing conservation laws are solved simultaneously in a single linear system of equations, A · φ = b. Following Denner et al. [1], the system of equations is solved for the primary solution variables χ, which are the pressure p, the velocity vector u and the temperature T. For a three-dimensional computational mesh with N cells, the linear system of governing equations is given as 

    

Ap Au Av Aw 0 Bp Bu Bv Bw 0 Cp Cu Cv Cw 0 Dp Du Dv Dw 0 Ep Eu Ev Ew ET


![Equation](images/2201.07447_eq177.png)


![Equation](images/2201.07447_eq178.png)


![Equation](images/2201.07447_eq179.png)


![Equation](images/2201.07447_eq180.png)


![Equation](images/2201.07447_eq181.png)


![Equation](images/2201.07447_eq182.png)


![Equation](images/2201.07447_eq183.png)

where the coefficient submatrix Aχ of size N × N holds the coefficients of the primary variable χ associated with the continuity equation (28), submatrices Bχ, Cχ, Dχ, all of size N × N, hold the coefficients of the primary variable χ associated with the momentum equations (30), for the x-, yand z-axes of the Cartesian coordinate system, and submatrix Eχ, also of size N × N, contains the coefficients of the primary variable χ associated with the energy equation (32). The subvector φχ of length N holds the solution of the primary variable χ. All contributions from previous non-linear iterations or previous time-levels are gathered in the right-hand side vector b of length 5N. The solution procedure performs nonlinear iterations in which the linear system of governing equations (42) is solved using the Block-Jacobi preconditioner and the BiCGSTAB solver of the software library PETSc [54], as described in detail by Denner et al. [2]. No underrelaxation of the governing equations is required.

6. Interface treatment

The discretised governing equations are extended to interfacial flows using an interface advection method (Section 6.1), an appropriate definition of the fluid properties in the interface region (Section 6.2) and the acoustically-conservative interface discretisation (ACID)1 [2] (Section 6.3). In order to

1The term “acoustically-conservative” refers to the acoustic properties of this discretisation method in the context of fully compressible flows and is not indicative of its application to incompressible fluids.

10

represent the two interacting fluids discretely, the indicator function ζ translates into a colour function ψ, defined for cell P as


![Equation](images/2201.07447_eq184.png)


![Equation](images/2201.07447_eq185.png)

ˆ


![Equation](images/2201.07447_eq186.png)

The interface is, thus, located in every cell with 0 < ψ < 1.

6.1. Interface advection

To advect the fluid interface between two fluids, Eq. (6) is applied to the colour function, Eq. (43), and reformulated as


![Equation](images/2201.07447_eq187.png)


![Equation](images/2201.07447_eq188.png)

The advection of the fluid interface is then described by Eq. (44) using an appropriate discretisation method. Two different VOF methods, an algebraic VOF method [2] and a piecewise-linear interface calculation (PLIC) method with Lagrangian advection of the interface [55], are considered for the advection of the fluid interface. In the algebraic VOF method [2], the advection equation of the colour function, Eq. (44), is discretised using the Crank-Nicolson scheme for the discretisation of the transient term and the CICSAM scheme [56] for the spatial interpolation of the colour function ψ. In the applied VOFPLIC method, the interface is reconstructed based on the local colour function ψ and the normal vector of the interface [57]. The interface advection equation (44) is then rewritten in integral form and the reconstructed interface is advected using the Lagrangian split advection scheme of van Wachem and Schouten [55]. In both considered interface advection methods, the advection is based on the same advecting velocity ϑf as for all advection terms of the governing equations, thus ensuring an accurate prediction of volume changes [2]. Nevertheless, the finite-volume discretisation and pressure-based algorithm presented in Section 5 are not limited to the employed VOF methods and other methods to represent the bulk phases and advect the interface, including level-set and front-tracking methods, may equally be applied.

6.2. Fluid properties

The definitions of the fluid properties largely follow the principles outlined by Denner et al. [2]. The density of the fluid is defined based on the colour function ψ as


![Equation](images/2201.07447_eq189.png)

where the partial densities ρa and ρb of the bulk phases are given by Eq. (13). This linear interpolation of the density is required in order to satisfy the discrete conservation of mass, momentum and energy. In the context of compressible flows, it is equivalent to an isobaric closure assumption [4, 58], while for incompressible fluids this formulation reduces to the typically used definition of density [59]. The heat capacity ratio also follows from the isobaric closure assumption as


![Equation](images/2201.07447_eq190.png)

where Ca and Cb are the binary compressibility operators defined in Eq. (12) associated with fluid a and fluid b, respectively. The specific isobaric heat capacity is defined by a mass-weighted interpolation [2], which is essential for the conservation of the total energy, given as


![Equation](images/2201.07447_eq191.png)

where the partial densities ρa and ρb are given by Eq. (13), density ρ is given by Eq. (45), and the partial specific isobaric heat capacities cp,a and cp,b are given by Eq. (14). The viscosity µ and

11

thermal conductivity k are defined as


![Equation](images/2201.07447_eq192.png)


![Equation](images/2201.07447_eq193.png)

The Continuum Surface Force (CSF) model of Brackbill et al. [59] is applied to model the force due to surface tension, represented by the source term


![Equation](images/2201.07447_eq194.png)

where κ is the interface curvature, which is computed using a second-order height-function method [60], and σ is the surface tension coefficient. In the interest of conciseness, but without loss of generality, the surface tension coefficient is assumed to be constant.

6.3. Coupling of the bulk phases

The ACID method [2] assumes that, for the purpose of discretising the governing conservation laws for a given cell, all cells in its finite-volume stencil are assigned the same colour function value, i.e. the colour function is assumed to be constant in the entire finite-volume stencil. The relevant thermodynamic properties that are discontinuous at the interface, i.e. density and enthalpy, are then evaluated based on this locally constant colour function field, as described by Denner et al. [2] in detail. This recovers the contact discontinuity associated with the interface [61, 62] and enables the application of the conservative discretisation described in Section 5, identical to the one applied for single-phase flows. Denner et al. [2] reported robust and accurate results for acoustic and shock waves in interfacial flows, supporting the notion that the interface discretisation indeed conserves the acoustic features of the flow and retains the conservative discretisation of the governing equations.

6.3.1. Density treatment Under the assumption that the colour function ψ is constant throughout the finite-volume stencil of cell P, as proposed by Denner et al. [2], the density interpolated to face f of cell P is given as


![Equation](images/2201.07447_eq195.png)


![Equation](images/2201.07447_eq196.png)

where ξf is the flux limiter determined by the applied differencing scheme, e.g. a TVD differencing scheme [63]. The density ρ⋆ U at the upwind cell U and ρ⋆ D at the downwind cell D are given based on the colour function value of cell P by Eq. (45), so that


![Equation](images/2201.07447_eq197.png)

and


![Equation](images/2201.07447_eq198.png)

The densities at previous time-levels are evaluated in a similar fashion based on the colour function value of cell P, with

ρ(t−∆t) P = ρ(t−∆t) a,P + ψ(t) P � ρ(t−∆t) b,P −ρ(t−∆t) a,P � (54)

and analogously for ρ(t−2∆t) P .

6.3.2. Enthalpy treatment The specific total enthalpy at face f is given, again assuming the colour function ψ is constant throughout the finite-volume stencil of cell P, as [2]


![Equation](images/2201.07447_eq199.png)


![Equation](images/2201.07447_eq200.png)

12

with ρf given by Eq. (51). The specific total enthalpy of the upwind and downwind cells are

h⋆ U = c⋆ p,U TU + u2 U 2 , (56)

h⋆ D = c⋆ p,D TD + u2 D 2 , (57)

respectively, ρ⋆ U is given by Eq. (52) and ρ⋆ D is given by Eq. (53). The specific isobaric heat capacities c⋆ p,U and c⋆ p,D are defined by Eq. (47) with ψP as


![Equation](images/2201.07447_eq201.png)

and


![Equation](images/2201.07447_eq202.png)

Since the specific enthalpy is partially sought implicitly, formulated implicit with respect to the primary solution variable temperature, see Eq. (34), a deferred correction approach is applied to enforce Eq. (55) [2]. The specific total enthalpy at the previous time-levels is given as

h(t−∆t) P = c⋆,(t−∆t) p,P T (t−∆t) P + u(t−∆t),2 P


![Equation](images/2201.07447_eq203.png)

with

ρ(t−∆t) P c⋆,(t−∆t) p,P = ρ(t−∆t) a,P c(t−∆t) p,a,P + ψ(t) P � ρ(t−∆t) b,P c(t−∆t) p,b,P −ρ(t−∆t) a,P c(t−∆t) p,a,P � , (61)

and analogously for h(t−2∆t) P and c⋆,(t−2∆t) p,P .

6.3.3. Further observations The corrections applied by ACID to the discretised governing equations vanish in the bulk phases and are non-zero only at fluid interfaces [2]. The fluid properties are piecewise constant at the interface and the corresponding error is proportional to ∆x, as commonly found in VOF, level-set and front-tracking methods [64]. With respect to the different fluid combinations that can occur at the interface, the following can be observed:

• At compressible-compressible interfaces, problems associated with a discontinuous change of fluid properties are circumvented with ACID, while retaining the information carried by compression and expansion waves, as comprehensively demonstrated by Denner et al. [2]. Hence, acoustic waves, shock waves and rarefaction fans can interact with and at the interface. The proposed algorithm then becomes an enhanced version of the algorithm for compressible interfacial flows of Denner et al. [2], including viscous stresses and surface tension.

• At incompressible-incompressible interfaces, ACID retains the incompressible formulation of the discretised governing equations. In fact, the non-conservative formulation of the governing equations at the interface originally proposed by Brackbill et al. [59] is obtained. The proposed algorithm then reduces to a non-isothermal version of the fully-coupled algorithm for incompressible interfacial flows of Denner and van Wachem [27], as for instance used in [50].

• At compressible-incompressible interfaces, the compressibility of the compressible fluid stays, dependent on the local colour function value, partly intact and ACID provides a transition from the compressible fluid to the incompressible fluid. Compression and expansion waves are able to interact with such a compressible-incompressible interface, but compressible effects are not transmitted into the incompressible fluid.

13


![Equation](images/2201.07447_eq204.png)


![Equation](images/2201.07447_eq205.png)

0 0.5 1 1.5 2

Ekin [J]


![Equation](images/2201.07447_eq206.png)


> **Figure 2: Kinetic energy Ekin as a result of parasitic currents as a function of dimensionless time τ = t/tµ, with tµ = ρ0,fd2 B/µ the viscous timescale, of the bubble in equilibrium. The gas inside the bubble is compressible, whereas the fluid surrounding the bubble is incompressible.**

7. Validation

As already discussed, for incompressible-incompressible interfacial flows, the proposed algorithm is equivalent to the incompressible algorithm of Denner and van Wachem [27], which has been extensively tested and validated against analytical solutions [50, 65], experiments [66, 67] and other numerical methods [65]. For compressible-compressible interfacial flows, the proposed algorithm becomes the compressible algorithm of Denner et al. [2], which has been validated thoroughly with a broad range of test-cases. Thus, the validation presented below focuses on the application of the proposed algorithm to predict compressible-incompressible interfacial flows. Five representative test-cases are considered to demonstrate the application of the proposed algorithm to compressibleincompressible interfacial flows: a bubble with surface tension in equilibrium (Section 7.1), the viscous damping of capillary waves (Section 7.2), the reflection of an acoustic wave at a fluid interface (Section 7.3), the pressure-driven collapse of a bubble (Section 7.4) and the shock interaction with a water drop (Section 7.5).

7.1. Bubble in equilibrium A circular bubble of a compressible gas surrounded by an incompressible fluid is simulated to study the parasitic currents associated with the numerical treatment of the surface tension contribution. Similar to the test-cases used in previous studies [68, 69], the considered two-dimensional bubble has a diameter of dB = 0.8 m and a surface tension coefficient of σ = 1.0 N m−1. The ambient pressure is p0 = 1.0 Pa, the incompressible fluid has a density of ρ0,f = 1.0 kg m−3 and the compressible gas has a density of ρg = 1.0 kg m−3 at p0. The compressible gas has a heat capacity ratio of γ0 = 1.4 and both fluids have a specific isobaric heat capacity of cp,0 = 1008 J kg−1 K−1. The viscosity µ, which is the same for both fluids, follows from the considered Laplace number, La = ρ0,fdBσ/µ2 = 120. The bubble is placed at the centre of a domain with edge length 2 m × 2 m, represented by an equidistant Cartesian mesh with 64×64 mesh cells, and the applied time-step is ∆t = 10−3 s, which satisfies the capillary time-step constraint [50]. The fluid interface is advected using the VOF-PLIC method. Figure 2 shows the total kinetic energy Ekin = �N P =1 ρP u2 P /2, with N the total number of mesh cells, in the computational domain as a function of the dimensionless time τ = t/tµ, with tµ = ρ0,fd2 B/µ the viscous timescale. Since the bubble is in equilibrium, the observed kinetic energy is the result of parasitic currents only. The imbalance caused by errors in the numerical evaluation of the interface curvature leads to an initial production of parasitic currents and the associated kinetic energy. As the interface topology relaxes towards a numerical equilibrium, the production of parasitic currents diminishes and the existing parasitic currents dissipate as a result of viscous stresses. Consequently, the kinetic energy in the domain decays rapidly and the exact balance is recovered on the discrete level.

7.2. Capillary waves To demonstrate the accurate prediction of viscous flows and surface tension by the proposed algorithm, the oscillation of a standing capillary wave between two viscous fluids is simulated. This

14


![Equation](images/2201.07447_eq207.png)


![Equation](images/2201.07447_eq208.png)

0

0.5

1

0 5 10 15 20 25

A/A0


![Equation](images/2201.07447_eq209.png)

Simulation

Theory

(a) Compressible-compressible flow


![Equation](images/2201.07447_eq210.png)


![Equation](images/2201.07447_eq211.png)

0

0.5

1

0 5 10 15 20 25

A/A0


![Equation](images/2201.07447_eq212.png)

Simulation

Theory

(b) Compressible-incompressible flow


> **Figure 3: Temporal evolution of the dimensionless amplitude A/A0 of a capillary wave with wavelength λ = 0.1 m and initial amplitude A0 = 0.01λ as a function of dimensionless time tω0, obtained for a compressible-compressible flow and a compressible-incompressible flow. The results are compared against the analytical solution of Prosperetti [70].**

test case is particularly challenging because the wave amplitude is small and the temporal evolution of the wave amplitude, governed by the dispersion (due to surface tension) and attenuation (due to viscous stresses) of the capillary wave, is very sensitive to the implementation of the viscous stress terms, the surface tension model, numerical diffusion and spurious oscillations [65, 68]. The analytical solution for the initial value problem of a freely-oscillating capillary wave with infinitesimal amplitude, derived by Prosperetti [70], which is valid for two-phase systems in which both fluids have the same kinematic viscosity ν = µ/ρ or in which one fluid is neglected, serves as a reference. The considered capillary wave has a wavelength of λ = 0.1 m, both fluids have a kinematic viscosity of ν = 1.6394 × 10−4 m2 s−1 and uniform initial temperature T0 = 300 K, and the surface tension coefficient is σ = 0.25 π−3 N m−1. The density of the two fluids are ρa = 1 kg m−3 and ρb = 16 kg m−3. The computational domain has the dimensions λ × 3λ and is represented by an equidistant Cartesian mesh with ∆x = λ/32. A compressible-compressible flow, with γ0 = 1.4, as well as a compressibleincompressible flow (fluid ’b’ is considered to be incompressible) are considered. The fluid interface is advected using the algebraic VOF method. Figure 3 shows the dimensionless wave amplitude A/A0 as a function of the dimensionless time tω0, where ω0 = � σk3/(ρa + ρb) is the undamped angular frequency of the capillary wave and k = 2π/λ is the wavenumber, with initial wave amplitude A0 = 0.01λ, obtained for both flow types. In both cases the computed result is in excellent agreement with the analytical solution of Prosperetti [70], with respect to the amplitude as well as the frequency.

7.3. Acoustic waves

The propagation of a single acoustic wave in an air-water flow, where air is treated as a compressible fluid and water is taken to be either a compressible fluid described by the stiffened-gas model or an incompressible fluid with constant density, is simulated in a one-dimensional domain with mesh spacing ∆x = 2 × 10−3 m, with initial velocity u0 = 1 m s−1, initial pressure p0 = 105 Pa and initial temperature T0 = 300 K. Air has the fluid properties γ0,Air = 1.4 and Π0,Air = 0 Pa, and the density at initial conditions is ρAir = 1.16 kg m−3. Compressible water has the fluid properties γ0,Water = 4.1, Π0,Water = 4.4 × 108 Pa with a density at initial conditions of ρWater = 1000 kg m−3. Incompressible water has a constant density of ρ0,Water = 1000 kg m−3. The acoustic wave is initiated by the inlet-velocity

uin =


![Equation](images/2201.07447_eq213.png)


![Equation](images/2201.07447_eq214.png)


![Equation](images/2201.07447_eq215.png)


![Equation](images/2201.07447_eq216.png)

with ∆u0 the amplitude and f = 2500 s−1 the frequency of the velocity perturbation. Two perturbation amplitudes are considered, ∆u0 ∈{0.01 u0, 100 u0}, to study the linear and nonlinear acoustic regimes. In the linear acoustic regime (∆u ≪a0 and ∆ρ ≪ρ0), which applies to the considered velocity

15

0

3

6

9

0 0.2 0.4 0.6 0.8 1 1.2 1.4


![Equation](images/2201.07447_eq217.png)

x [m]


![Equation](images/2201.07447_eq218.png)


![Equation](images/2201.07447_eq219.png)

Interface


![Equation](images/2201.07447_eq220.png)


![Equation](images/2201.07447_eq221.png)

0

30

60

90

120

0 0.2 0.4 0.6 0.8 1 1.2 1.4


![Equation](images/2201.07447_eq222.png)

x [m]

Interface


![Equation](images/2201.07447_eq223.png)


![Equation](images/2201.07447_eq224.png)


> **Figure 4: Pressure profile of acoustic waves with different amplitudes in a one-dimensional domain after the interaction with an air-water interface at t = 2 ms, in which air (left of the interface) is treated as a compressible fluid and water (right of the interface) is treated as an incompressible fluid (aWater = ∞) or as a compressible fluid (aWater = 1343 m s−1). The amplitude of the applied initial velocity perturbation is (a) ∆u0 = 0.01 u0 and (b) ∆u0 = 100 u0. The pressure amplitudes of the reflected and transmitted waves according to linear acoustic theory are shown in (a) as a reference.**

perturbation amplitude of ∆u0 = 0.01 u0, the perturbation leads to a sound wave. Based on linear acoustic theory [61], the acoustic wave reflected at the air-water interface should have a pressure amplitude of

∆preflected Air,0 = ∆pincident Air,0 2ZWater ZWater −ZAir −1 , (63)

where Z = ρa is the characteristic acoustic impedance and ∆pincident Air,0 = ZAir ∆u0 is the pressure amplitude of the incident wave. While the theoretical pressure amplitude of the reflected wave is ∆preflected Air,0 = ∆pincident Air,0 if water is considered to be an incompressible fluid (ρWater = const., aWater = ∞), the reflected pressure amplitude is ∆preflected Air,0 ≃0.999 ∆pincident Air,0 if water is considered to be a compressible fluid (ρWater = f(p, T), aWater = 1343 m s−1). Hence, the differences in the results between the incompressible and compressible treatments of water should be negligible, at least in the linear acoustic regime, apart from the transmitted wave that should only be present for the compressible treatment of water. Figure 4 shows the pressure profile of the acoustic waves, after they have interacted with the interface, for both considered perturbation amplitudes using both the compressible treatment and the incompressible treatment of the liquid phase. For the small perturbation amplitude, ∆u0 = 0.01 u0, the amplitude of the reflected pressure wave is in excellent agreement with linear acoustic theory with both treatments of the liquid phase. When the liquid phase is treated as a compressible fluid, a pressure wave is transmitted through the interface, while no wave is transmitted when the liquid phase is treated as an incompressible fluid, as expected. For the large perturbation amplitude, ∆u0 = 100 u0, the pressure profile of the acoustic wave departs significantly from its originally sinusoidal shape as a result of nonlinear wave steepening. Nevertheless, the reflected waves predicted by the compressible treatment and the incompressible treatment of the liquid phase are in excellent agreement, suggesting that the compressible-incompressible treatment of gas-liquid flows can be applied even when large amplitude acoustic waves are present in the gas phase.

7.4. Bubble collapse

The pressure-driven growth and collapse of gas bubbles, in particular cavitation, is a widely observed phenomenon, and the ability to focus large amounts of energy by a collapsing bubble is being utilised in an increasing number of applications. The simplest analytical model for the pressuredriven collapse of a bubble in a liquid is the Rayleigh-Plesset equation [71] without considering viscous dissipation and surface tension, given as

R ¨R + 3


![Equation](images/2201.07447_eq225.png)


![Equation](images/2201.07447_eq226.png)

16

0 1 2 3 4 0


![Equation](images/2201.07447_eq227.png)

1


![Equation](images/2201.07447_eq228.png)


![Equation](images/2201.07447_eq229.png)


![Equation](images/2201.07447_eq230.png)


![Equation](images/2201.07447_eq231.png)

0 1 2 3 4 0


![Equation](images/2201.07447_eq232.png)

1


![Equation](images/2201.07447_eq233.png)


![Equation](images/2201.07447_eq234.png)


![Equation](images/2201.07447_eq235.png)


![Equation](images/2201.07447_eq236.png)


> **Figure 5: Temporal evolution of the dimensionless radius R/R0 of a collapsing gas bubble in an incompressible liquid, compared against the solution of the Rayleigh-Plesset equation (64). (a) Results for different time-steps ∆t obtained with a spatial resolution of ∆x = R0/400; (b) Results with different mesh spacings ∆x obtained with a time-step of ∆t = 10−4 tR. Time t is normalised by the Rayleigh collapse time, tR = 0.915 R0 � ρ0,ℓ/p∞.**

where R is the bubble radius, ρ0,ℓis the (constant) density of the liquid, pg is the (uniform) gas pressure inside the bubble and p∞is the ambient liquid pressure at infinite distance from the bubble. Interestingly, Eq. (64) is based on the assumption of an incompressible liquid, with ρ0,ℓ= const., and a compressible gas bubble. The collapse of a spherical gas bubble due to an overpressure in the liquid, the so-called Rayleigh collapse [72], is considered to validate the prediction of pressure-driven flows of a compressible gas in contact with an incompressible liquid by the proposed algorithm. Following the setup of Denner et al. [33], a bubble with initial radius R0 and an initial gas pressure of pg = 4000 kPa is situated in a liquid with an initial pressure of pℓ(r) = p∞+ (pg −p∞) R0/r, with p∞= 105 Pa and r the radial coordinate. The gas has a heat capacity ratio of γ0 = 1.4 and the liquid has a constant density of ρ0,ℓ= 1000 kg m−3. Viscosity, thermal conduction and surface tension are neglected. The fluid interface is advected using the algebraic VOF method. Because the liquid is incompressible and viscous dissipation and thermal conduction are neglected, the oscillations of the bubble radius should continue indefinitely with unchanged frequency and amplitude. Figure 5 shows the dimensionless radius R/R0 as a function of the dimensionless time t/tR, where tR = 0.915 R0 � ρ0,ℓ/p∞is the Rayleigh collapse time, obtained with different spatial and temporal resolutions using the proposed algorithm. The solution of the Rayleigh-Plesset equation (64) is also shown as a reference and regarded here as the exact solution of the dynamic bubble behaviour under consideration. The results predicted by the proposed algorithm and by the Rayleigh-Plesset equation are in excellent agreement, as shown in Fig. 5, for a sufficiently large spatial and temporal resolution.

7.5. Shock-drop interaction Following the work of Meng and Colonius [73], the interaction of a shock wave, initially travelling in air with Mach number Ms = 1.47, with a water drop is simulated. The two-dimensional domain is represented by an equidistant Cartesian mesh with 100 cells per drop diameter. The shock wave separates the post-shock region (I) and the pre-shock region (II), which are initialised with uI = 246.24 m s−1, pI = 2.35×105 Pa, TI = 450.56 K and uII = 0 m s−1, pII = 1.00×105 Pa, TII = 347.22 K, respectively. The fluid properties of air are γ0,Air = 1.4 and Π0,Air = 0 Pa, with ρII,Air = 1 kg m−3 and aII,Air = 374.17 m s−1 in the pre-shock region. The liquid drop is treated as an incompressible fluid, with constant density ρ0,Water = 1000 kg m−3, or as a compressible fluid, with the fluid properties γ0,Water = 4.4 and Π0,Water = 6 × 108 Pa, and ρII,Water = 1000 kg m−3 and aII,Water = 1624.94 m s−1

in the pre-shock region. The fluid interface is advected using the algebraic VOF method. Figure 6 shows the contours of the velocity magnitude |u| and the density gradient |∇ρ| obtained t = 16 µs after the first shock-drop interaction, treating water as an incompressible or a compressible

17

(a) Velocity (b) Density gradient


> **Figure 6: Contours of the velocity magnitude |u| and the density gradient |∇ρ| after a shock with Ms = 1.47 interacted with a two-dimensional water drop, simulating the water drop as an incompressible fluid (upper half) and as a compressible fluid (lower half). The density waves inside the compressible drop are clearly visible, while no density waves are transmitted to the incompressible drop.**

fluid. While the velocity profile and the shock structures are virtually identical in the gas phase, the compressible treatment of the water drop allows density waves to propagate in it, while no density waves are propagating in the incompressible drop. With respect to the execution time of the simulations, the incompressible treatment of the drop achieves a significant speed-up of the simulation, because the stiff pressure-density coupling imposed by the stiffened-gas model for the compressible liquid, and the associated decrease in convergence rate, can be circumvented with the incompressible treatment. In addition, a larger time-step can be applied with the incompressible treatment of the liquid, because no shock waves have to be resolved in the liquid. As a result, simply switching to an incompressible treatment of the water drop accelerates the simulation by factor 1.8, and also adjusting the applied time-step brings an overall acceleration of the simulation of factor 3.9.

8. Conclusions

We have presented a new and promising route to construct numerical algorithms for the simulation of both compressible and incompressible interfacial flows, as well as mixtures thereof, based on a unified thermodynamic closure model, a finite-volume discretisation and a fully-coupled pressurebased algorithm. The proposed thermodynamic closure model and treatment of the fluid properties at the interface, in conjunction with the ACID method to couple the interacting bulk phases [2] and the Newton linearisation of the governing equations [28], bridge the different numerical requirements for the simulation of incompressible and compressible fluids, and facilitate the simulation of compressibleincompressible interfacial flows. If, however, only incompressible fluids or only compressible fluids are considered, the presented algorithm is equivalent to previously proposed pressure-based algorithms for incompressible interfacial flows [27] or compressible interfacial flows [2], respectively. The proposed algorithm has been successfully validated using five representative test-cases, each featuring a combination of a compressible fluid and an incompressible fluid separated by an interface: a bubble with surface tension in equilibrium, the viscous damping of capillary waves, the reflection of an acoustic wave at a gas-liquid interface, the pressure-driven collapse of a bubble and the shock interaction with a water drop. In particular, the presented simulations of interactions of an acoustic wave with a compressible-incompressible interface and of a high-Mach compressible flow with an incompressible fluid (i.e. the shock-drop interaction) are unique capabilities that have not been demonstrated in the literature before. In addition, treating liquids as incompressible, while still resolving acoustic effects in the gas phase, has been shown to yield substantial performance benefits.

18

For gas-liquid flows in general, the strength of the proposed algorithm lies in the reliable prediction of compressible effects in the gas phase, without paying the additional cost of resolving marginal physical effects in the liquid phase, if the application allows this. For instance, acoustic waves in the gas phase are known to promote interfacial instabilities [74], which however have a negligible influence on the behaviour of the liquid phase in subsonic flows; especially, the compressibility of the liquid does not influence the acoustics in the gas phase, as shown by the presented results. With the presented algorithm the simplification of an incompressible fluid cannot only be invoked for low-Mach flows with respect to the gas phase, as considered in previous studies [37–39], but practically for any flow velocity. While a compressible liquid is not a valid simplification for large liquid Mach numbers, for subsonic gas flows, for instance in subsonic fuel injection and spray atomisation processes, the presented results suggest an incompressible liquid to be a reasonable assumption.

Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), grant numbers 420239128 and 447633787.


## References

[1] F. Denner, F. Evrard, B. van Wachem, Journal of Computational Physics 409, 109348 (2020). [2] F. Denner, C.N. Xiao, B. van Wachem, Journal of Computational Physics 367, 192 (2018). [3] M. Baer, J. Nunziato, International Journal of Multiphase Flow 12(6), 861 (1986). [4] G. Allaire, S. Clerc, S. Kokh, Journal of Computational Physics 181(2), 577 (2002). [5] A. Murrone, H. Guillard, Journal of Computational Physics 202(2), 664 (2005). [6] V. Coralic, T. Colonius, Journal of Computational Physics 274, 95 (2014). [7] C. Rohde, C. Zeiler, Applied Numerical Mathematics 95, 267 (2015). [8] D.P. Garrick, M. Owkes, J.D. Regele, Journal of Computational Physics 339, 46 (2017). [9] R. Fedkiw, T. Aslam, B. Merriman, S. Osher, Journal of Computational Physics 152(2), 457 (1999). [10] T. Liu, B. Khoo, K. Yeo, Journal of Computational Physics 190(2), 651 (2003). [11] C. Liu, C. Hu, Journal of Computational Physics 342, 43 (2017). [12] R. Abgrall, S. Karni, Journal of Computational Physics 169, 594 (2001) [13] S.M.H. Karimian, G.E. Schneider, Journal of Thermophysics and Heat Transfer 8(2), 267 (1994). [14] P. Wesseling, Principles of Computational Fluid Dynamics (Springer, 2001) [15] E. Turkel, A. Fiterman, B. van Leer, Preconditioning and the Limit to the Incompressible Flow Equations. Tech. rep., NASA CR-191500 (1993) [16] D. van der Heul, C. Vuik, P. Wesseling, Computers & Fluids 32(8), 1113 (2003). [17] J.H. Park, C.D. Munz, International Journal for Numerical Methods in Fluids 49(8), 905 (2005). [18] F. Cordier, P. Degond, A. Kumbaro, Journal of Computational Physics 231(17), 5685 (2012). [19] M. Boger, F. Jaegle, B. Weigand, C.D. Munz, Computers & Fluids 96, 338 (2014). [20] G. Hauke, T.J. Hughes, Computer Methods in Applied Mechanics and Engineering 153(1-2), 1 (1998). [21] J. Van Doormaal, G. Raithby, B. McDonald, ASME Journal of Turbomachinery 109(April 1987), 268 (1987) [22] K.C. Karki, S.V. Patankar, AIAA Journal 27(9), 1167 (1989). [23] A.J. Chorin, Journal of Computational Physics 2(1), 12 (1967). [24] J.B. Bell, P. Colella, H.M. Glaz, Journal of Computational Physics 85(2), 257 (1989). [25] S. Patankar, Numerical Heat Transfer and Fluid Flow (Hemisphere Publishing Company, 1980) [26] R. Issa, Journal of Computational Physics 62, 40 (1985) [27] F. Denner, B. van Wachem, Numerical Heat Transfer Part B: Fundamentals 65(3), 218 (2014). [28] F. Denner, Computers & Fluids 175, 53 (2018) [29] H. Bijl, P. Wesseling, Journal of Computational Physics 141, 153 (1998) [30] F.H. Harlow, A.A. Amsden, Journal of Computational Physics 8(2), 197 (1971). [31] M. Darwish, F. Moukalled, Numerical Heat Transfer, Part B: Fundamentals 65(5), 410 (2014). [32] C.N. Xiao, F. Denner, B. van Wachem, Journal of Computational Physics 346, 91 (2017). [33] F. Denner, F. Evrard, B. van Wachem, Fluids 5(2), 69 (2020). [34] R. Caiden, R. Fedkiw, C. Anderson, Journal of Computational Physics 166, 1 (2001) [35] M. Aanjaneya, S. Patkar, R. Fedkiw, Journal of Computational Physics 247, 17 (2013). [36] A.R. Wadhwa, J. Abraham, V. Magi, AIAA Journal 43(9), 1974 (2005). [37] M. Billaud, G. Gallice, B. Nkonga, Computer Methods in Applied Mechanics and Engineering 200(9-12), 1272 (2011). [38] J.P. Caltagirone, S. Vincent, C. Caruyer, Computers & Fluids 50(1), 24 (2011). [39] T. Yamamoto, K. Takatani, International Journal for Numerical Methods in Fluids (2018). [40] T.Y. Hou, P.G.L. Floch, Mathematics of Computation 62(206), 497 (1994). [41] C. Hirt, B. Nichols, Journal of Computational Physics 39(1), 201 (1981) [42] F. Harlow, A. Amsden, Fluid dynamics. Monograph LA-4700, Los Alamos National Laboratory (1971) [43] O. Le M´etayer, J. Massoni, R. Saurel, International Journal of Thermal Sciences 43(3), 265 (2004).

19

[44] J.H. Ferziger, M. Peric, R.L. Street, Computational Methods for Fluid Dynamics, 4th edn. (Springer International Publishing, 2020) [45] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab (Springer, 2016) [46] J. Ferziger, International Journal for Numerical Methods in Fluids 41, 551 (2003) [47] J. Ferziger, M. Peri´c, Computational Methods for Fluid Dynamics, 3rd edn. (Springer Verlag, Berlin Heidelberg New York, 2002) [48] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Journal of Computational Physics 375, 177 (2018). [49] C.M. Rhie, W.L. Chow, AIAA Journal 21(11), 1525 (1983). [50] F. Denner, B. van Wachem, Journal of Computational Physics 285, 24 (2015). [51] R. Kunz, W. Cope, S. Venkateswaran, Journal of Computational Physics 152(1), 78 (1999) [52] A.J. Chorin, J.E. Marsden, A Mathematical Introduction to Fluid Mechanics (Springer Verlag, 1993) [53] A. Toutant, Physics Letters A 381(44), 3739 (2017). [54] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M.G. Knepley, D.A. May, L.C. McInnes, W.D. Gropp, K. Rupp, P. Sanan, B.F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual. Tech. Rep. ANL-95/11 - Revision 3.8, Argonne National Laboratory (2017) [55] B. van Wachem, J. Schouten, AIChE Journal 48(12), 2744 (2002). [56] O. Ubbink, R. Issa, Journal of Computational Physics 153, 26 (1999) [57] R. Scardovelli, S. Zaleski, Journal of Computational Physics 164(1), 228 (2000). [58] K.M. Shyue, Shock Waves 15(6), 407 (2006). [59] J. Brackbill, D. Kothe, C. Zemach, Journal of Computational Physics 100, 335 (1992) [60] F. Evrard, F. Denner, B. van Wachem, Journal of Computational Physics: X 7, 100060 (2020). [61] J.D. Anderson, Modern Compressible Flow: With a Historical Perspective (McGraw-Hill New York, 2003) [62] E.F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical Introduction, 3rd edn. (Springer, 2009) [63] F. Denner, B. van Wachem, Journal of Computational Physics 298, 466 (2015). [64] S. Popinet, Annual Review of Fluid Mechanics 50, 49 (2018). [65] F. Denner, G. Par´e, S. Zaleski, Euro. Phys. J. Spec. Top. 226, 1229 (2017). [66] F. Denner, A. Charogiannis, M. Pradas, C.N. Markides, B. van Wachem, S. Kalliadasis, Journal of Fluid Mechanics 837, 491 (2018). [67] F. Denner, B. van Wachem, Journal of Computational Physics 279, 127 (2014). [68] S. Popinet, Journal of Computational Physics 228(16), 5838 (2009). [69] D. Fuster, S. Popinet, Journal of Computational Physics 374, 752 (2018). [70] A. Prosperetti, Physics of Fluids 24(7), 1217 (1981). [71] M.S. Plesset, Journal of Applied Mechanics 16, 277 (1949) [72] W. Lauterborn, C. Lechner, M. Koch, R. Mettin, IMA Journal of Applied Mathematics 83(4), 556 (2018). [73] J.C. Meng, T. Colonius, Shock Waves 25(4), 399 (2015). [74] Z.W. Zhou, S.P. Lin, Journal of Propulsion and Power 8(4), 736 (1992).

20

