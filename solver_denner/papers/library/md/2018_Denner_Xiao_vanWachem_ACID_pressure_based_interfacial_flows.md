
# Accepted Manuscript

Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation

Fabian Denner, Cheng-Nian Xiao, Berend G.M. van Wachem

PII: S0021-9991(18)30253-5 DOI: https://doi.org/10.1016/j.jcp.2018.04.028 Reference: YJCPH 7968

To appear in: Journal of Computational Physics

Received date: 4 September 2017 Revised date: 5 April 2018 Accepted date: 13 April 2018

Please cite this article in press as: F. Denner et al., Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, J. Comput. Phys. (2018), https://doi.org/10.1016/j.jcp.2018.04.028

This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

Highlights

• A novel pressure-based algorithm for the simulation of compressible interfacial flows is proposed. • Interface discretisation method that retains the acoustic properties of the compressible flow. • Accurate propagation, reflection and transmission of acoustic waves in interfacial flows. • Accurate capturing and prediction of shock waves and rarefaction fans, including shock-interface interaction.


## Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq001.png)

aDepartment of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom bChair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany

Abstract

A pressure-based algorithm for the simulation of compressible interfacial flows is presented. The algorithm is based on a fully-coupled finite-volume framework for unstructured meshes with collocated variable arrangement, in which the governing conservation laws are discretised in conservative form and solved in a single linear system of equations for velocity, pressure and specific total enthalpy, with the density evaluated by an equation of state. The bulk phases are distinguished using the Volume-of-Fluid (VOF) method and the motion of the fluid interface is captured by a state-of-the-art compressive VOF method. A new interface discretisation method is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy based on the assumption of thermodynamic equilibrium, and does not require a Riemann solver. This interface discretisation method yields a consistent definition of the fluid properties in the interface region, including a unique definition of the speed of sound and the Rankine-Hugoniot relations, and conserves the acoustic features of the flow, i.e. compression and expansion waves. A variety of representative test cases of gas-gas and gas-liquid flows, ranging from acoustic waves and shock tubes to shock-interface interactions in one-, twoand three-dimensional domains, is used to demonstrate the capabilities and versatility of the presented algorithm in all Mach number regimes. The propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows are predicted accurately, even for difficult cases that feature fluids with shock impedance matching, transonic shock tubes or strong shocks in gas-liquid flows, as well as on unstructured meshes.

Keywords: Interfacial flows, Compressible flows, Pressure-based methods, Finite-volume methods, Shock-capturing, Acoustics

1. Introduction

The numerical modelling of compressible flows is associated with a number of difficulties owing to the formulation of the conservation equations, the coupling of hydrodynamic and thermodynamic variables, and high pressure ratios. In addition to the numerical difficulties encountered when modelling compressible singlephase flows, the numerical solution of compressible interfacial flows, in which two (or more) immiscible fluids interact, is further complicated by different fluid properties and speeds of sound (and, thus, Mach numbers) of the bulk phases, complex acoustic behaviour as well as the numerical treatment of the fluid interface. It has proven particularly difficult to define the discrete interface between two compressible fluids in a consistent manner, which retains the main features of the solution, such as the propagation of acoustic waves and shock waves. According to Coralic and Colonius [1], numerical methods able to accurately predict the interaction of interfaces with shock waves, and in extension compressible interfacial flows in general, should satisfy the following criteria:

1. discrete conservation of mass, momentum and energy, 2. avoid the generation of spurious oscillations at the interface or at shock waves, and 3. provide high-order accuracy in smooth regions.

A variety of contemporary algorithms for compressible interfacial flows, e.g. [1–7], have been shown to discretely satisfy the global conservation of mass, momentum and energy in the computational domain, which is a prerequisite for the accurate prediction of the speed of shock waves [8–10], although the conservation in each individual phase is strongly dependent on the applied method. Flux-limited high-order schemes, most

∗Corresponding author: Email address: fabian.denner@gmail.com (Fabian Denner)

Preprint submitted to Journal of Computational Physics April 18, 2018

notably total variation diminishing (TVD) schemes [11, 12], or high-order schemes combined with artificial viscosity models [13], can provide second-order accuracy in smooth regions and avoid oscillations at shocks and discontinuities. Spurious oscillations at the interface as a result of the discontinuity in fluid properties, however, remain a common issue for interface capturing methods [1], in particular when the bulk phases are assumed to be in mechanical equilibrium. Karni [14] and Abgrall [2] were among the first to depart from a fully-conservative discretisation, i.e. a discretisation that is simultaneously conservative with regards to the entire computational domain and each individual phase [15], in favour of avoiding spurious oscillations. The application of a non-conservative discretisation at the interface has also been suggested in the context of incompressible interfacial flows by Brackbill et al. [16], to avoid spurious oscillations as a result of the changing fluid properties at the interface. A non-conservative discretisation of the governing equations is widely, and largely successfully, applied for incompressible interfacial flows, although a na¨ıve non-conservative discretisation does not allow compression and expansion waves to pass the interface. To this end, exact and approximate Riemann solvers are widely applied to allow the exchange of information between the interacting fluids through the interface. Recent studies [17, 18] also suggest that a strictly conservative discretisation is important for the robustness of the solution algorithm and the predictive accuracy for incompressible interfacial flows with large density ratios. State-of-the-art algorithms for compressible interfacial flows are almost exclusively founded on densitybased algorithms, which solve the governing conservation equations (momentum, continuity and energy) for the momentum, density and total energy of the flow. In the two-fluid Baer-Nunziato (BN) model [19], each phase is treated as a separate fluid with its own momentum, continuity and energy equations, with an additional transport equation for the volume fraction field of the interacting fluids (e.g. a colour or level-set function), or a topological equation that represents the fluid interface. BN models, also often referred to as seven-equation models, have well-defined hyperbolic properties [20–22] and conserve the mass, momentum and total energy of each phase. However, BN models involve a considerable computational complexity due to the seven, nine or eleven governing equations for one-, twoor three-dimensional flows, respectively, which have to be coupled through relaxation terms for pressure and velocity (usually under the assumption of local thermodynamic equilibrium), including non-conservative products [21–25] and a consistently defined interface velocity [6, 21], which is typically based on the solution of an exact or approximate Riemann problem in a Godunov-type, Rusanov-type or Roe-type discretisation. By assuming equilibrium of pressure and velocity in the limit of infinitely fast relaxation to mechanical equilibrium [6, 26, 27], BN models can be reduced to five-equation models [1, 4, 27–31], with a separate continuity equation for each phase, and shared momentum and energy equations. This model conserves the mass of each phase and globally conserves momentum and energy [28]. However, the total energy of each individual phase is not conserved and the relative volume change of the phases due to compression/expansion has to be incorporated in the interface transport equation [6, 27]. Abandoning the conservation of a separate density field for each phase, and considering instead only a single density field, leads to a four-equation model [2–4] with a single conservation equation for momentum (one for each space dimension), for mass and for energy, plus an interface transport equation. This approach is also frequently referred to as the one-fluid model, because the entire two-phase flow (both phases and mixtures thereof) is numerically treated as one fluid with locally varying properties. It is the simplest model able to simulate interfacial flows, but does not by design conserve discretely the momentum, mass or energy of each phase. Furthermore, the fourequation model is associated with difficulties when recovering the pressure field in density-based frameworks [4, 15, 27], for which a specific equation of state (EOS), e.g. a stiffened gas EOS, is required. To this end, the five-equation model of Allaire et al. [4], for instance, reduces to the four-equation model of Shyue [3] when both phases are perfect gases. In the vast majority of methods, such as in the seven-, fiveand four-equation models discussed above, a Riemann solver is applied to evaluate the fluxes at cell-faces of the computational mesh, effectively reducing the evaluation of the fluxes to a one-dimensional problem for which a Riemann problem is solved [32]. Assuming an interface coincides with a cell-face, a Riemann problem with different EOS can be solved at the fluid interface to couple the bulk phases in an effective manner [33]. Riemann solvers have a strong mathematical foundation, but exact Riemann solvers are for many practical cases prohibitively time consuming, which motivates the development of approximate Riemann solvers. The Harten-Lax-van Leer Contact (HLLC) formulation, introduced by Toro et al. [34], has become the most widely used approximate Riemann solver for interfacial flows, as it captures contact discontinuities. However, because approximate Riemann solvers rely on an accurate a priori approximation of the wave speeds, their derivation for complex interfacial flows can be cumbersome and the resulting methods are often limited to a specific application or parameter range. Various HLLC-type solvers have been proposed for interfacial flows [1, 35–37], including phase transition [38, 39] and surface tension [30, 38, 39]. The Ghost-Fluid Method (GFM), originally proposed by Fedkiw et al. [40], presents an alternative that does not, in general, require the solution of a Riemann problem. The GFM defines ghost cells on either side of the interface for the discretisation of the governing equations, where

2

the real fluid and the ghost fluid coexist, and ghost values are defined in these ghost cells for the fluid on the opposing side of the interface. Due to its conceptual simplicity, the GFM has been used in a variety of numerical frameworks for compressible interfacial flows, notably [41, 42]. The GFM has also been extended to interfacial flows with reactions [43, 44] and phase transition [44], and has been used in conjunction with discontinuous Galerkin methods [45]. However, the original GFM was found to lack stability for strong shockinterface interactions and compressible gas-liquid flows, a shortcoming which was addressed by coupling the GFM to a Riemann solver to compute the flow states at the interface [46–49]. Density-based algorithms are arguably the method of choice for flows with considerable compressibility, but are ill-suited for low-Mach number flows [50–52], requiring sophisticated preconditioning and solution methods, see e.g. [53–57]. The pressure field has to be reconstructed based on the applied EOS, since pressure is not a solution variable in density-based algorithms. This is generally less of a problem when GFM [40, 42, 47] or similar so-called sharp interface methods are used, where either phase is present in a mesh cell but never a mixture of them. However, interfacial mesh cells (where the normalised interface indicator function, e.g. a volume fraction, is 0 < Ψ < 1) contain a “mixture” of both phases when interface capturing methods or diffuse interface methods are applied, such as the Volume-of-Fluid method (VOF). This fluid mixture, which Abgrall and Saurel [23] aptly called numerical mixture, is of numerical origin and, from a continuum mechanics standpoint, has no physical basis. Reconstructing the pressure in such interfacial cells is problematic [4, 15, 27], since hydrodynamically and thermodynamically plausible fluid properties as well as a mixture EOS have to be defined, which may misrepresent the interface region by presuming a physical meaning of the finite interface thickness. As a consequence, the speed of sound in the interface region can, for instance, be lower than the speed of sound of either bulk phase [6, 58–60], which is inconsistent with the zero thickness of the interface assumed in continuum mechanics. Pressure-based algorithms, in which the continuity equation serves as an equation for pressure, while density is evaluated explicitly using a suitable EOS, are preferably applied for incompressible flows and yield significant advantages for low-Mach number flows, since density is not required to be a thermodynamic variable and the acoustic degeneration (i.e. vanishing pressure-density coupling) at low Mach numbers does not pose a problem. The success of pressure-based algorithms is facilitated by the unique role of pressure in all Mach number regimes, with the pressure-velocity coupling dominant at low Mach numbers and the pressure-density coupling dominant at high Mach numbers [61, 62], as well as the convenient fact that the fully-conservative formulation of the governing conservation laws can still be satisfied accurately, even if nonconserved quantities, such as pressure, are chosen as primary solution variables [61]. However, ensuring a stable numerical solution in the transonic regime, where pressure is strongly coupled with both velocity and density, and formulating consistent shock-capturing schemes for pressure-based algorithms has proven to be difficult [51, 63]. Although, since its original conception by Harlow and Amsden [64, 65], many pressurebased algorithms have been proposed for compressible single-phase flows [50, 61, 66–76], no pressure-based algorithm for compressible interfacial flows has been published yet. In this article, a fully-coupled pressure-based algorithm for compressible interfacial flows is proposed, based on the finite-volume framework of Xiao et al. [76] for compressible single-phase flows on unstructured meshes. The presented algorithm uses a compressive VOF method [77] for the advection of the interface, leading to a four-equation model. The discretised governing flow equations (momentum, continuity and energy) are solved in a single system of linearised equations for velocity, pressure and specific total enthalpy, with density being evaluated based on the applied EOS. The present study considers the perfect-gas and stiffened-gas models to describe interfacial flows of gases and liquids. A new discretisation method at the fluid interface is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy in the interface region, assuming local mechanical and thermal equilibrium, whereby the conservative discretisation of the governing equations remains unchanged. Since the proposed algorithm is pressure-based and applies a single pressure field for the entire computational domain, no partial pressures have to be considered and no mixture pressure has to be defined at the interface. Instead, only mixture rules for the fluid properties have to be defined. As demonstrated by representative test cases of compressible gas-gas and gas-liquid flows, the presented algorithm predicts the propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows accurately. In particular, the precise simulation of the reflection and transmission of acoustic waves at fluid interfaces as conducted in this study has not been reported in the literature before. In Section 2, the governing equations are introduced. The numerical framework and the pressure-based algorithm are presented in Section 3 and the applied compressive VOF method is described in Section 4. The new interface discretisation method is proposed in Section 5, followed by a discussion of the iterative solution procedure in Section 6. The results for a variety of representative compressible interfacial flows are presented and discussed in Section 7. The article is summarised and concluded in Section 8.

3

2. Governing equations

The considered compressible interfacial flows of inviscid fluids at all speeds are governed, assuming Cartesian coordinates, by the momentum equations


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq002.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq003.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq004.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq005.png)

the continuity equation ∂ρ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq006.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq007.png)

and the energy equation ∂ρh


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq008.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq009.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq010.png)

where t is time, ρ is the density, u is the velocity vector, p is the pressure and h = cp T + u2/2 is the specific total enthalpy, with cp the specific isobaric heat capacity and T the temperature. The enthalpy formulation is chosen for the energy equation, rather than the more common internal energy formulation, as it leads to a straightforward application in the proposed numerical algorithm, since the transient pressure term on the right-hand side of Eq. (3) does not require linearisation. The governing equations require closure by defining the thermodynamic properties of the fluids through an appropriate EOS. In this study, the stiffened-gas model [78] is applied, which provides a very good description of liquids and solids of practical interest [33, 79], and reduces to the perfect-gas model for gases. It is, therefore, widely used for interfacial flow modelling. The thermodynamic properties of the fluid are linked via the stiffened gas EOS


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq011.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq012.png)

where R is the specific gas constant, γ = cp/cv is the heat capacity ratio, cv is the specific isochoric heat capacity and Π is a material-dependent pressure constant, which is Π = 0 for an ideal gas. The speed of sound for a stiffened gas is


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq013.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq014.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq015.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq016.png)

from which the specific total enthalpy follows as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq017.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq018.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq019.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq020.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq021.png)

with the specific isobaric heat capacity


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq022.png)

and cp,0 = R/(1 −γ−1). For Π = 0, the stiffened-gas model reduces to the perfect-gas model. In order to distinguish the interacting bulk phases, the VOF method [80] is applied to represent the bulk phases by an indicator function ψ(x), typically called colour function, with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq023.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq024.png)

where Ω = Ωa ∪Ωb is the computational domain, with Ωa and Ωb the subdomains occupied by fluid a and b, respectively. Consequently, the interface is located in cells with 0 < ψ < 1. Since the density is discontinuous at the fluid interface and fluid does not flow through the interface (assuming no mass transfer), the interface between two immiscible fluids represents a contact discontinuity [23, 78]. Thus, the fluid interface is a material front propagating with the flow and, consequently, the material derivative of ψ is


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq025.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq026.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq027.png)

Accounting, in addition, for the different acoustic properties of the bulk phases [6, 59], Eq. (9) becomes


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq028.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq029.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq030.png)

4

where


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq031.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq032.png)

is a material-dependent compressibility factor [6, 59, 60], with ρ given by Eq. (4) and a given by Eq. (5). Assuming the bulk phases are in mechanical equilibrium, and since no mass transfer and surface tension are considered in this study, the interface conditions for velocity and pressure are ua · m = ub · m and pa = pb [81], respectively, where m is the normal vector of the interface.

3. Numerical framework

The presented algorithm is founded on a fully-coupled pressure-based numerical framework, based on the algorithm for compressible single-phase flows of Xiao et al. [76]. The framework is predicated on the finite-volume method with a collocated variable arrangement, and all discretisation methods presented below are applicable to unstructured meshes. Since the discretisation of the governing equations is largely identical for single-phase and interfacial flows, the discretisation of the governing equations presented in this section focuses on single-phase flows. The modifications to this discretisation for interfacial flows, including the appropriate definition of the fluid properties, is described in Section 5 and the iterative solution procedure applied to solve the discretised, linearised governing equations is described in Section 6.

3.1. Finite-volume method The finite-volume method, which forms the foundation for the numerical framework, is based on the integral formulation of the governing conservation laws. It is worth recalling that only the integral form of the governing conservation laws is valid at shocks and discontinuities [32]. Integrating, for example, the continuity equation (2) over the control volume V , the integral form of Eq. (2) is given as ˚


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq033.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq034.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq035.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq036.png)

where ˚


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq037.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq038.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq039.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq040.png)

follows from the divergence theorem, with S the outward pointing surface vector on the surface dV of control volume V . Assuming the surface of the control volume is constituted by a finite number of flat faces f and applying the midpoint rule [62, 82], the surface integral can be expressed with second-order accuracy [82] as ‹


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq041.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq042.png)

where Af is the area of face f and ϑf = ufnf is the advecting velocity, with nf the outward pointing unit normal vector of face f. Because a collocated variable arrangement is employed, all primary solution variables (u, p, h) as well as the density ρ are stored at cell centres. Therefore, the data required at face centres is obtained by interpolation from the data at adjacent cell centres, as further explained in Section 3.2, while the face velocity uf requires a special interpolation to prevent pressure-velocity decoupling, discussed in detail in Section 3.4. The discretisation of the temporal term is discussed in Section 3.3. The interested reader is referred to the work of Xiao et al. [76] for further details on the applied finite-volume method.

3.2. Spatial discretisation The central differencing scheme is applied for the interpolation from cell centres to face centres of variables that are not advected, given for a general flow variable φ as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq043.png)

The geometry coefficient lf is defined based on an inverse distance weighting, given as lf = |rP f|/Δsf, where Δsf is the distance between cell centres P and Q (the cell centres adjacent to face f), schematically illustrated in Fig. 1a, and rP f is the vector connecting cell centre P with face centre f. Advected variables are interpolated to face centres using the TVD interpolation method for general unstructured meshes proposed by Denner and van Wachem [83]. Considering the velocity vector u as indicated in Fig. 1b, the face value follows as (where ∼denotes a flux-limited interpolation)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq044.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq045.png)

5


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq046.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq047.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq048.png)

(a) General discretisation


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq049.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq050.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq051.png)

(b) TVD differencing


> **Figure 1: Schematic illustration of (a) cell P with its neighbour cell Q and the shared face f, where nf is the unit normal vector of face f and sf is the unit vector connecting cells P and Q (both outward pointing with respect to cell P), and (b) upwind cell U and downwind cell D of face f, where u represents the velocity vector.**

where subscripts U and D denote the upwind and downwind cells, ξf is the flux limiter and Lf = Δsf/|rUf| is a geometry coefficient. Dependent on the studied flow, the first-order upwind scheme (ξf = 0), the central differencing scheme (ξf = 1) or the Minmod scheme [84] are applied for the simulations presented as part of this study, but other TVD schemes could also be applied.

3.3. Temporal discretisation The First-Order Backward Euler scheme or the Second-Order Backward Euler scheme are applied for the discretisation of the transient terms of the governing flow equations. The First-Order Backward Euler scheme is readily given for cell P as ˆ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq052.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq053.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq054.png)

with Δt1 being the current time-step, superscript (t −Δt1) denotes values of the previous time-level and VP is the volume of mesh cell P. Assuming a varying time-step is applied to solve the governing equations, the Second-Order Backward Euler scheme is given as [85] ˆ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq055.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq056.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq057.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq058.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq059.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq060.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq061.png)

where Δτ = Δt1 +Δt2, Δt2 is the previous time-step and superscript (t−Δτ) denotes values of the previousprevious time-level. If Δt1 = Δt2, Eq. (18) reverts to the Second-Order Backward Euler scheme for a constant time-step [82] ˆ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq062.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq063.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq064.png)

In what follows, the discretised governing equations are presented using the First-Order Backward Euler scheme, although the Second-Order Backward Euler scheme is also considered in the results presented in Section 7. Irrespective of the chosen scheme, for consistency all transient terms of the governing equations, ∂ρuj/∂t in Eq. (1), ∂ρ/∂t in Eq. (2), as well as ∂ρh/∂t and ∂p/∂t in Eq. (3), are discretised with the same scheme.

3.4. Advecting velocity A momentum-weighted interpolation (MWI) method is applied to evaluate the advecting velocity ϑf = ufnf at cell faces f, with ϑf taking the role of flux-velocity in the discretised advection terms of the governing equations. MWI emulates a staggered variable arrangement by introducing a cell-to-cell pressure coupling through a low-pass filter acting on the third derivative of pressure [51, 82, 86], which avoids pressure-velocity decoupling as a result of the collocated variable arrangement and provides a robust pressure-velocity coupling for incompressible and low-Mach number flows [76], while preserving the second-order accuracy of the finitevolume method [51, 82]. Following previous studies, the definition of the advecting velocity includes various modifications to the MWI formulation originally introduced by Rhie and Chow [87], to account for non-orthogonal meshes [88–90], as well as the large density ratios occurring in interfacial flows and the transient nature of the considered

6

problems [90]. Based on the work of Denner and van Wachem [90], the advecting velocity ϑf = ufnf at face f, see Fig. 1a, is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq065.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq066.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq067.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq068.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq069.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq070.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq071.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq072.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq073.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq074.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq075.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq076.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq077.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq078.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq079.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq080.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq081.png)

The coefficient ˆdf is defined as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq082.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq083.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq084.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq085.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq086.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq087.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq088.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq089.png)

where eP and eQ are the sum of the coefficients of the primary variable u arising from the advection terms (and, if viscous flows are considered, the shear stress terms) of the momentum equations (see [90] for a detailed derivation). The face density ρ∗ f is interpolated by a harmonic average,


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq090.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq091.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq092.png)

which is necessary for a consistent definition of the coefficient of the pressure term as well as the efficacy of the density weighting [90]. This density weighting of the pressure gradients, with which robust incompressible interfacial flow simulations with density ratios of up to 1024 [90, 91] were previously presented, stabilises the numerical solution of interfacial flows with large density ratios.

3.5. Discretised governing equations Applying the spatial and temporal discretisation techniques described above to the integral formulation of the continuity equation, Eq. (12), the discretised continuity equation for cell P follows as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq093.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq094.png)

with the advecting velocity ϑf given by Eq. (20) and superscript (n + 1) denotes variables that are solved for implicitly. Although the discretised continuity equation is formulated conservative in mass, it is treated as an equation for pressure. Hence, based on Eq. (4), the implicit density ρ(n+1) P at cell centre P is given by the pressure-implicit formulation


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq095.png)

where the fluid properties γ, Π and R are defined based on the most recent available interface position, as explained in Section 5, and the temperature T is deferred, as further explained in Section 6. The advection term of Eq. (23) is linearised with a Newton linearisation, �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq096.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq097.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq098.png)

where superscript (n) denotes values from the previous nonlinear iteration, with the implicit formulation of the advecting velocity ϑ(n+1) f given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq099.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq100.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq101.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq102.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq103.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq104.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq105.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq106.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq107.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq108.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq109.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq110.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq111.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq112.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq113.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq114.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq115.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq116.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq117.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq118.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq119.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq120.png)

Applying the same discretisation principles to the momentum and energy equations, the discretised momentum equations (1) become


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq121.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq122.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq123.png)

7

and the discretised energy equation (3) is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq124.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq125.png)

Both the transient terms and the advection terms of the momentum and energy equations are linearised with a Newton linearisation. With primary solution variable χ, i.e. velocity uj in the momentum equations (27) and specific total enthalpy h in the energy equation (28), the transient terms are linearised as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq126.png)

where ρ(n+1) P is given by Eq. (24), and the advection terms are linearised with respect to the density and the advecting velocity, following as �

f ˜ρfϑf ˜χfAf = �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq127.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq128.png)

where ϑ(n+1) f is given by Eq. (26). Both the spatial pressure derivative on the right-hand side of Eq. (27) and the transient pressure derivative on the right-hand side of Eq. (28) are treated implicitly. The applied Newton linearisation promotes a smooth transition from elliptic/parabolic to hyperbolic behaviour of the governing equations in different Mach number regimes [50, 92, 93], which is widely understood to be of particular importance for the continuity equation [50, 61, 76], where it also provides the required pressure-velocity coupling at low Mach numbers [76, 90]. Furthermore, the Newton linearisation of all governing equations facilitates an implicit contribution of active flow-dependent variables, i.e. the density and the advecting velocity, in the context of the presented fully-coupled pressure-based algorithm, which improves the performance and stability of the solution algorithm.

3.6. A note on accuracy and conservation The accuracy of both the underpinning finite-volume formulation (Section 3.1) and the MWI used to evaluate the advecting velocity (Section 3.4) is second order [51, 82]. Thus, applying a second-order TVD scheme, such as the Minmod scheme considered in this study, for the discretisation of the advection terms and the Second-Order Backward Euler scheme for the discretisation of the transient terms, the discretised governing equations are overall second-order accurate. Moreover, because the velocity u is treated separately from the density ρ, rather than treating volumetric momentum ρu as a single quantity, a second-order discretisation can be applied to ˜uf at contact discontinuities (including fluid interfaces), where the density is discontinuous and ˜ρf requires a low-order discretisation to avoid spurious oscillations [94]. Although Ham and Iaccarino [89] identified a numerical dissipation of kinetic energy associated with the low-pass pressure filter of the MWI, this numerical dissipation has a marginal magnitude and is proportional to Δx2. As a consequence, the numerical dissipation of momentum and energy is negligible, since viscous stresses and heat conduction are neglected in this study, except at nonlinear waves where numerical dissipation is introduced by the TVD scheme to regulate spurious oscillations. Although the governing flow equations are solved for p, u and h, the governing equations are discretised in conservative form for mass, Eq. (23), momentum, Eq. (27), and energy, Eq. (28). The density is not a solution variable but is defined by the applied EOS as a function of pressure, temperature and the fluid properties. As demonstrated by Van Doormaal et al. [61] in the context of a pressure-based algorithm, choosing primitive variables (h is not a primitive variable but is treated in the same way) as primary solution variables, instead of the conserved variables, in a fully-conservative formulation of the governing conservation laws does not affect the conservation properties, if a consistent and appropriate discretisation is applied (see also [50, 68, 74, 76] for further examples). To this end, the continuity equation acts as a contraint on the pressure field and results in velocities, through the coupling with the momentum equations, and densities, via the applied EOS, that satisfy the conservation of mass in all Mach number regimes [61]. Furthermore, the same discretisation of ˜ρf and the same advecting velocity ϑf are applied in the discretised governing equations, ensuring a consistent formulation of the fluxes. Consequently, when the system of discretised nonlinear governing equations is converged, the conservative form of the governing conservation laws is satisfied on the discrete level within a predefined solution tolerance. To achieve convergence of the nonlinear system of governing equations, the iterative solution procedure described in Section 6 is applied.

8

4. Compressive VOF method

A consistent and precise transport of the colour function ψ, which represents the bulk phases and the fluid interface, is critical for the overall accuracy of the numerical algorithm. In order for the applied fluxes to be consistent, the advection term of the advection equation (10) is reformulated using the chain rule,


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq129.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq130.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq131.png)

an approach previously applied in [31, 95]. Equation (10) then becomes


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq132.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq133.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq134.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq135.png)

Now both spatial derivatives of velocity can be consistently discretised with the same advecting velocity ϑf as the discretised governing equations in Section 3.5.

4.1. Discretisation The compressive VOF method introduced by Denner and van Wachem [77], which does not require an explicit interface reconstruction and has been shown to conserve the volume of incompressible interfacial flows on structured and unstructured meshes accurately, is applied to advect the colour function field with Eq. (32). In the applied finite-volume framework, the colour function in cell P is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq136.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq137.png)

ˆ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq138.png)

thus representing the discrete volume fraction of the fluid occupying subdomain Ωb, as defined in Eq. (8). Applying the Crank-Nicolson scheme to discretise the transient term [77, 96], the discretised advection equation (32) follows as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq139.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq140.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq141.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq142.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq143.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq144.png)

where Δtψ is the time-step applied to advect the colour function field, the advecting velocity ϑf is given by Eq. (20) and the material-dependent compressibility factor K is given by Eq. (11). The face value ψf is interpolated using the CICSAM scheme [96], including an implicit correction of mesh skewness [77]. The CICSAM scheme is able to advect sharp interfaces with very high accuracy, as for instance demonstrated in [77, 96, 97], by taking into account the orientation of the interface and, contrary to other algebraic interface advection schemes, the available amount of colour function in the upwind cell that can be advected into the downwind cell over the applied time-step. Accounting for the available flux-volume, however, comes at the cost of a very small time-step Δtψ to maintain a sharp interface [97], and a dual time-stepping method is applied to optimise the computational performance [77], as further detailed in Section 6. Although a compressive VOF method is applied in this study, the proposed pressure-based algorithm is not limited to a specific method for the representation and transport of the interface. The applied compressive VOF method is chosen for its versatility, as it is also applicable on unstructured meshes. Any other suitable interface capturing method, such as the class of algebraic THINC schemes for compressible flows [49, 98], or interface tracking method, e.g. front-tracking methods [41, 48], may equally be used in conjunction with the proposed algorithm.

4.2. Validation The compressive VOF method [77] applied as part of this study has previously been succefully used for a variety of different incompressible, i.e. solenoidal (∇· u = 0), flows and has thereby been shown to be in excellent agreement with experimental data [77, 99] and analytical solutions [91, 100]. Contrary to incompressible flows, however, the applicability and accuracy of the modified method given above for compressible flows with ∇· u ̸= 0 is yet to be demonstrated. Following the work of Raessi et al. [101], a circular and a spherical interface with initial radius r0 = 0.25 m are simulated in a square and cubical domain, respectively, with edge length 1 m. In the two-dimensional (2D) case, the centre of the circular interface is situated at xc = yc = 0.5 m and the velocity field is prescribed as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq145.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq146.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq147.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq148.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq149.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq150.png)

9

0

0.2

0.4

0.6

0.8

1

0 0.05 0.1 0.15 0.2 0.25

A/A0

t [s]

Simulation

Theory

(a) Area enclosed by the circular interface.


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq151.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq152.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq153.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq154.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq155.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq156.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq157.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq158.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq159.png)

2D 3D


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq160.png)


> **Figure 2: (a) Area A of the of the shrinking circular interface, normalised by the initial area A0, on an equidistant Cartesian mesh with Δx = 256−1 m as a function of time t, and (b) the error of the computed area (2D) and volume (3D) enclosed by the interface as a function of mesh spacing Δx.**

In the three-dimensional (3D) case, the centre of the spherical interface is situated at xc = yc = zc = 0.5 m and the velocity field is prescribed as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq161.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq162.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq163.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq164.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq165.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq166.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq167.png)

The velocity field is, hence, not divergence free (∇· u ̸= 0) and is pointing to the centre of the circular or spherical interface with |u| = 1, causing the interface to shrink and the fluid mass enclosed by the interface to reduce. Note that the flow field is fixed and the result is, therefore, independent of the discretisation of the governing flow equations. Figure 2a shows the evolution of the area enclosed by the circular interface on an equidistant Cartesian mesh with Δx = 256−1 m alongside the exact solution, and Fig. 2b shows the errors of the computed area inside the circular interface and of the computed volume inside the spherical interface as a function of mesh spacing Δx. The change in area (2D) and volume (3D) is predicted accurately and the computational errors converge with second order in both 2D and 3D, demonstrating that the modified compressive VOF method presented above is suitable for compressible flows.

5. Acoustically-conservative interface discretisation

Recall the earlier observation, discussed in Section 2, that a fluid interface is a contact discontinuity. At a contact discontinuity, velocity and pressure are continuous, while density (and all variables that depend on density) is discontinuous [32, 102]. In a compressible single-phase flow, the contact discontinuity is a characteristic of the solution of the governing conservation laws and represents a weak solution to the discretised governing equations; a prominent example is a contact discontinuity in a shock tube. However, the discretised governing equations do not account for the contact discontinuity associated with an interface separating two immiscible fluids. This problem can be illustrated by considering the simple example sketched in Fig. 3, showing a two-phase flow at constant velocity (∂u/∂x = 0) and pressure (∂p/∂x = 0) in a one-dimensional domain. Because the left fluid is heavier than the right fluid, yet both fluids have the same velocity, their volumetric momentum ρu is different. Discretising the momentum equation (1) for this one-dimensional problem at cell P of the equidistant mesh shown in Fig. 3, the advection term follows, using standard finite differences with linear interpolation, as ∂ρu2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq168.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq169.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq170.png)

and, with a time-step Δt corresponding to a Courant number of Co = u Δt/Δx = 0.3, the transient term is


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq171.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq172.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq173.png)

10


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq174.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq175.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq176.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq177.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq178.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq179.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq180.png)


> **Figure 3: Fluid domain and computational mesh, with relevant cell-centred data at the current time-level, of a (quasi) onedimensional example of two-phase flow at constant velocity.**

Hence, in order to satisfy the discretised momentum equation (27), the jump in density at the fluid interface causes an unphysical pressure gradient and corresponding acceleration of the flow [2, 14, 16]. Thus, without modifications to the discretisation to account for the change in fluid properties at the fluid interface, the discretised governing equations as presented in Section 3.5 yield discontinuous and typically oscillatory velocity and pressure values. The underpinning principle of the proposed acoustically-conservative interface discretisation (ACID) is based on a conservative discretisation of the governing equations in each finite-volume stencil, with the aim of applying the discretisation derived for a single-phase flow in Section 3 and obtaining consistently defined fluid properties in the interface region where 0 < ψ < 1, while still respecting the contact discontinuity associated with the fluid interface. For the numerical framework applied in this study, the finite-volume stencil of a given cell P, schematically shown in Fig. 4a, contains all face-neighbour cells of cell P, as well as cell P itself. In order to satisfy the assumption of a single-phase flow in the finite-volume stencil of a given cell P for the purpose of discretising the governing equations, all cells in the finite-volume stencil of cell P are assigned the colour function value of cell P. This is schematically illustrated in Fig. 4; the colour function values in cell (i, j) and its neighbourhood, shown in Fig. 4a, are taken to be ψi−1,j = ψi+1,j = ψi,j−1 = ψi,j+1 = ψi,j, as shown in Fig. 4b. Thus, the colour function is kept piecewise constant in the entire finite-volume stencil, which enables the application of the fully-conservative discretisation scheme presented in Section 3, identical to the one applied for single-phase flows. The relevant thermodynamic properties that are discontinuous at the interface, i.e. density and enthalpy, are then evaluated and discretised based on this piecewise-constant colour function field, assuming mechanical and thermal equilibrium at the interface. Note that the twodimensional equidistant Cartesian mesh in Fig. 4 is chosen for illustration purposes; the application of this approach on unstructured and/or three-dimensional meshes is straightforward.

5.1. Density treatment In the proposed pressure-based algorithm, density ρ is not a solution variable. Instead, density is evaluated by a linear interpolation of the partial densities of the bulk phases, given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq181.png)

with ρa and ρb defined by the applied EOS, Eq. (4). This linear interpolation of the partial densities with respect to the colour function is necessary for the conservation of mass, momentum and energy, and is equivalent to an isobaric closure assumption [4, 35]. For the purpose of treating the density implicitly via Eq. (24), the linear interpolation of partial densities in Eq. (37) is interchangeable with defining the specific gas constant R and the stiffened-gas term γΠ as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq182.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq183.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq184.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq185.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq186.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq187.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq188.png)

respectively. Assuming the colour function ψ is piecewise-constant throughout the finite-volume stencil of cell P, as described above, the density at face f of cell P is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq189.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq190.png)

11


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq191.png)

1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq192.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq193.png)

1

1

1

1

1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq194.png)

1

1

1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq195.png)

1

0

0


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq196.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq197.png)

0

0

0

0

0


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq198.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq199.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq200.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq201.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq202.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq203.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq204.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq205.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq206.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq207.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq208.png)

1

1

1

1

1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq209.png)

1

1

1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq210.png)

1

0

0


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq211.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq212.png)

0

0

0

0

0


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq213.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq214.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq215.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq216.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq217.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq218.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq219.png)


> **Figure 4: Schematic illustration of the distribution of the colour function ψ in the interface region around cell (i, j) and its neighbours for the purpose of discretising the governing equations without and with ACID. The shaded cells represent the finite-volume stencil for cell (i, j) and the dotted line represents the fluid interface.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq220.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq221.png)

respectively, where the partial densities are evaluated using Eq. (4). In order for the discrete density field to be defined consistently, the density values at previous time-levels are evaluated using the same procedure, with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq222.png)

and, if required, ρ(t−Δτ) P = ρ(t−Δτ) a,P + ψP � ρ(t−Δτ) b,P −ρ(t−Δτ) a,P � . (44)

The density values in the bulk phases are unaffected by this discretisation.

5.2. Enthalpy treatment Similar to the treatment of density, the enthalpy in the interface region is reformulated using ACID. To ensure consistency, the total enthalpy H = ρh at face f is assumed to be


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq223.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq224.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq225.png)

because both density ρ and the specific isobaric heat capacity cp are properties of the respective bulk phase. The specific isobaric heat capacity is defined by a density-weighted average as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq226.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq227.png)

with the partial densities given by Eq. (4), the partial specific isobaric heat capacities given by Eq. (7) and ρ given by Eq. (37). Unlike density, the specific total enthalpy h is a primary solution variable of the applied algorithm and, therefore, cannot simply be replaced by a modified value. Thus, a deferred correction defined by a target face enthalpy ˆhf is applied to the implicitly computed value at every nonlinear iteration. Following the assumption defined in Eq. (45) and the density ˜ρf given by Eq. (40), the target enthalpy is defined as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq228.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq229.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq230.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq231.png)

12

where ρ⋆ U is given by Eq. (41), ρ⋆ D is given by Eq. (42), and the specific total enthalpy of the upwind and downwind cells are given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq232.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq233.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq234.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq235.png)

respectively. Based on Eq. (46), the specific isobaric heat capacities c⋆ p,U and c⋆ p,D are evaluated with colour function ψP as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq236.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq237.png)

and


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq238.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq239.png)

Based on the target face enthalpy ˆhf, the deferred enthalpy correction is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq240.png)

and the advection term of the discretised energy equation, Eq. (28), follows as ˆ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq241.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq242.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq243.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq244.png)

Away from the interface in the bulk phases, the face enthalpy ˜hf is equal to the target face enthalpy ˆhf, with δhf = 0, and the energy conservation in the bulk phases remains unaffected. In order for the specific total enthalpy to be defined consistently, the specific isobaric heat capacity of previous time-levels is also defined based on the current colour function ψP , given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq245.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq246.png)

with ρ(t−Δt1) P given by Eq. (43), and similarly for c⋆,(t−Δτ) p,P . The specific total enthalpy at the previous time-level then follows as h(t−Δt1) P = c⋆,(t−Δt1) p,P T (t−Δt1) P + 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq247.png)

and, if required,


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq248.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq249.png)

5.3. Thermodynamics properties The treatments for density and enthalpy defined above directly influence the thermodynamic properties of the interface region. Of particular interest in this context are the speed of sound and the Rankine-Hugoniot relations, as they represent fundamental solutions to the conservation laws governing compressible flows. With the density ρ given by Eq. (37) and the specific isobaric heat capacity cp given by Eq. (46), the speed of sound is defined based on Eq. (5) as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq250.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq251.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq252.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq253.png)

where 1 γ −1 = 1 −ψ


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq254.png)

follows from the isobaric closure assumption applied to the density, see Eq. (38), since R = (γ−1)cv. Thus, the speed of sound is given by a uniquely defined average of the speeds of sound in the bulk phases, a condition that Allaire et al. [4] associated with a well-possedness of the model governing equations. The accurate prediction of the speed of sound by Eq. (57) in the bulk phases and in the interface region is demonstrated in Section 7.3.1.

13

Considering a flow with a shock wave, the conditions in the post-shock state (I) can be related to the pre-shock state (II) by the Rankine-Hugoniot relations. Based on the Rankine-Hugoniot relations of the stiffened-gas model [33, 59] and assuming the pre-shock region is stationary (uII = 0), the pressure ratio for a given Mach number Ms = us/aII of the shock wave, with us being the shock speed, is


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq255.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq256.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq257.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq258.png)

where ˆΠ is the effective material-dependent pressure constant of the stiffened-gas model, which follows from Eq. (5) as ˆΠ = γ −1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq259.png)

with ρ given by Eq. (37), cp given by Eq. (46), and (γ −1) and γ given by Eq. (58). For an ideal gas, ˆΠ = 0. The density ratio across the shock wave is then given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq260.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq261.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq262.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq263.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq264.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq265.png)

with which the post-shock velocity uI is defined as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq266.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq267.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq268.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq269.png)

The post-shock temperature TI readily follows from density ρI and pressure pI via the applied EOS. Hence, the propagation of shock waves is uniquely defined in each bulk phase as well as in the interface region, and the accuracy of the jump relations given in Eqs. (59)-(62) is demonstrated in Section 7.4.1.

5.4. Some observations ACID is formulated using a general finite-volume discretisation and is independent of the numerical schemes applied to discretise the governing equations or the method used to advect the interface (e.g. VOF, level-set or front-tracking). No Riemann solver is required to compute the fluxes, no wave patterns have to be identified and no a priori assumptions about the wave structure have to be made, as for instance required when applying a classical Godunov scheme [32]. The proposed method assumes that the fluid properties are piecewise constant at the interface, i.e. the fluid properties are formally first-order accurate, and the corresponding error vanishes as Δx →0. This is consistent with the piecewise-constant definition of the fluid properties typically applied in VOF, level-set and front-tracking methods [103] and does not affect the local first-order accuracy of the discontinuity in fluid properties [104]. The discretisation of the governing equations remains formally fully conservative, as a set of governing equations in conservative form is solved for each mesh cell. A conservative discretisation of the governing equations is regarded as a prerequisite for the accurate prediction of both the speed and strength of shock waves and for satisfying the Rankine-Hugoniot relations [8, 9]. Issues associated with a discontinuous change of fluid properties at the interface are circumvented with ACID, while retaining the information associated with compression and expansion waves, thus allowing acoustic waves, shock waves and rarefaction fans to interact with the interface without artificial (numerical) obstruction. The assumed mechanical and thermal equilibrium at the interface can lead to an unphysical flux in the interface region when both bulk phases have different temperature but are stationary and at equal pressure. However, in practice, this spurious flux is typically small and appears to be no problem for dynamic interfacial flows, as observed in the results presented below. Applying a TVD scheme, the discretisation of velocity at the interface remains high-order, which is important when capturing instabilities at the interface (e.g. the Richtmyer-Meshkov instability). The proposed method shares conceptual similarities with the GFM [40] and with flux-splitting algorithms, such as the one of Abgrall and Karni [15]. GFM and ACID are both one-fluid models, since some form of a single fluid is assumed when discretising the governing equations. GFM copies variables required for the discretisation to ghost cells on “the other side” of the interface and assumes only one of the phases occupies a given mesh cell. ACID, on the other hand, defines the fluid properties in the entire computational stencil based on the value of the interface indicator function taken from the discretised cell, without making explicit assumptions about which phase, or combination of phases, is present in the cell. Similar to flux-splitting algorithms, ACID results in asymmetric fluxes at the cell-faces, i.e. a different flux is applied to a given face

14

dependent on which of its adjacent cells is discretised. To clarify, using ACID, the advecting velocity ϑf is symmetric at face f, but the density ˜ρf and the specific isobaric heat capacity cp,f may be asymmetric in the interface region. While ACID evaluates the fluxes based on the advecting velocity ϑf and the fluid properties only, previously presented flux-splitting algorithms, e.g. [15, 45], rely on a Riemann solver to evaluate the fluxes. Furthermore, ACID also adjusts the values at previous time-levels to consistently account for the changes in fluid properties at the interface. In preliminary studies, the proposed algorithm exhibited convergence issues in some cases where a perfectgas fluid (e.g. air) and a stiffened-gas fluid (e.g. water) interact, if the Minmod scheme (or other TVD schemes) was applied. Instead of a monotonically reducing residual of the discretised system of equations, oscillations about a constant finite value larger than the predefined solver tolerance η were obtained. After careful inspection, this behaviour is attributed to the inherent nonlinearity of TVD schemes [83], because the definition of the flux limiter depends on the advected scalar field itself. As a result of the typically large differences in density ρ and specific total enthalpy h of perfect and stiffened gases, such as air and water, even small changes of the TVD flux limiter ξf can have a significant impact on the solution and its convergence. Since both density ρ and specific total enthalpy h are discontinuous at the interface, TVD schemes revert to the first-order upwind scheme, with the flux limiter ξf →0, to avoid an oscillatory solution. Enforcing explicitly the application of the first-order upwind scheme (ξf = 0) at faces in the interface region, where |ψP −ψQ| > η, for the advection of density ρ and specific total enthalpy h is an effective remedy for the convergence issue associated with TVD schemes.

6. Solution procedure

The linear system, Aφ = b, of the governing equations for a three-dimensional flow discretised as described in Section 3, including the modifications at the interface proposed in Section 5, on a computational mesh with N cells is given as ⎛


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq270.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq271.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq272.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq273.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq274.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq275.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq276.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq277.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq278.png)

which is solved for the velocity vector u ≡(u, v, w)T , with u, v and w its Cartesian components, pressure p and specific total enthalpy h. Aχ, Bχ, Cχ are the N × N coefficient submatrices for primary variable χ of the momentum equations (27) associated with the x-, yand z-axes of the Cartesian coordinate system. The N × N submatrices Dχ and Eχ contain the coefficients of primary variable χ for the continuity equation (23) and the energy equation (28), respectively. The solution subvector φχ of length N holds the solution for primary variable χ and the right-hand side vector b of length 5N holds all known contributions, which are either deferred or from previous time-levels. This linear system of equations is solved using the Block-Jacobi preconditioner and the BiCGSTAB solver of the software library PETSc [105–107]. The linear system of equations is considered to be converged if [107]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq279.png)

where ∥· ∥denotes the L2-norm and η is the predefined solution tolerance. An inexact Newton method [108] is applied to account for the nonlinearity of the governing equations, with nonlinear iterations in which the linearised governing equations are solved and the deferred variables are subsequently updated. A flow chart illustrating this iterative solution procedure is shown in Fig. 5. Following the work of Xiao et al. [76], the linear system of equations (63) is solved in each nonlinear iteration by assuming the flow is barotropic, i.e. density is only a function of the pressure (but not the temperature). Note that the flow is not considered to be isothermal, but only the update of density neglects changes in temperature in the barotropic loop. When the L2-norm of the residual vector δ of the linear system of equations (63) satisfies


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq280.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq281.png)

the (barotropic) inner loop is considered converged. Subsequently, in the outer loop, the density is updated based on the new pressure and new temperature values. This iterative dual-loop procedure continues until Eq. (65) and


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq282.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq283.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq284.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq285.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq286.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq287.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq288.png)

15

Update previous time-levels


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq289.png)

Update all fluid properties


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq290.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq291.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq292.png)

Update all other fluid properties


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq293.png)

Eq. (65) satisfied?


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq294.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq295.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq296.png)

Eqs. (65) & (66) satisfied?

yes no

yes no

inner loop


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq297.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq298.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq299.png)


> **Figure 5: Flow chart of the iterative solution procedure applied in each time-step, with n the iteration counter of the (barotropic) inner loop and m the iteration counter of the outer loop.**

are both satisfied simultaneously. This solution procedure was shown to be robust for single-phase flows at all speeds and without underrelaxation of the governing equations [76]. The advection of the colour function field by means of Eq. (34) is conducted in a separate linear system of equations, before the system of linearised governing fluid equations is solved in each time-step [90], see Fig. 5. A dual time-stepping method is applied to satisfy the stringent time-step constraints of the CICSAM scheme [77, 97] at an only moderate increase of computational time, with a different time-step Δtψ applied to solve the VOF advection equation than the time-step Δt1 applied to solve the equations governing the fluid flow. In order for the advection of the colour function to be consistent, the fluid time-step Δt1 has to be equal to or an integer multiple of the VOF time-step Δtψ. For all presented simulations, the Courant number associated with Δtψ is Coψ = Δtψ|u|/Δx ≤0.05.

7. Results

In order to verify and validate the proposed algorithm, the results for different representative test cases are presented: interface advection in Section 7.1; the conservation of momentum, mass and energy in Section 7.2; the propagation, reflection and transmission of acoustic waves in Section 7.3; the propagation and interface interaction of shock waves in Section 7.4; shock tube problems in Section 7.5; the interaction of shock waves with bubbles and drops in Sections 7.6-7.8. The presented results focus on the propagation of acoustic waves, shock waves and rarefaction fans in interfacial flows. Results for single-phase flows are only briefly discussed to demonstrate the accurate prediction of acoustic phenomena and shock waves by the proposed algorithm, since a broad variety of results obtained with the applied numerical framework for single-phase flows at all speeds and on unstructured meshes have been published by Xiao et al. [76].

16

0.8

0.9

1

1.1

1.2

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]

with ACID without ACID


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq300.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq301.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq302.png)

100

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq303.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq304.png)

0

0.2

0.4

0.6

0.8

1

1.2

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq305.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq306.png)


> **Figure 6: Profiles of (a) velocity, (b) pressure difference Δp = |∂p/∂x| Δx and (c) density of the sharp interface advected with constant velocity u0 = 1 m s−1 on a mesh with 500 cells at t = 0.7 s, with and without ACID.**

7.1. Interface advection with constant velocity A one-dimensional interfacial flow with uniform velocity and different fluid properties of the bulk phases is considered. The one-dimensional domain has a length of 1.0 m, which is represented with 500 equidistant cells and a time-step equivalent to a Courant number of Co = u0Δt/Δx = 0.5. The velocity u0 = 1 m s−1, pressure p0 = 105 Pa and temperature T0 = 300 K are uniform throughout the domain, and the properties of the left and the right phases are ρL = 1.156 kg m−3, γL = 1.4 and ρR = 0.160 kg m−3, γR = 1.6, respectively. The inlet velocity and temperature are uin = u0 and Tin = T0, respectively. Since the advection equation of the colour function ψ, Eq. (10), is linear (assuming K = 0) and the applied velocity u0 is constant, this test case is equivalent to the Riemann problem of a contact discontinuity. The flow should be unaffected by the change in density and specific heat capacity at the interface, and the interface should move with the flow at the prescribed velocity u0. Although this might appear to be trivial, this has been notoriously difficult to achieve with conservative methods [79]. The interface is represented by a stepwise change in colour function ψ, initially located at x = 0.1 m. Figure 6 shows the profiles of velocity u, pressure difference Δp = |∂p/∂x| Δx and density ρ at time t = 0.7 s, with and without ACID. The interface and the associated discontinuity of the fluid properties do not affect the flow if ACID is applied, and the interface is advected with the correct speed of u = u0. The equilibrium of pressure p and velocity u as well as the stepwise change of density ρ at the interface are retained, contrary to previously published algorithms, e.g. [31]. Note that the discontinuity in colour function and density at the interface does not numerically diffuse, which is a common issue of diffuse interface methods methods for compressible flows [109–111]. However, if ACID is not applied, the flow field is clearly affected in an unphysical way, with large changes in velocity and pressure, and the interface is not advected with the correct speed, as evident in Fig. 6. As a second test-case, a smooth interface with a linear variation of ψ from ψ = 1 to ψ = 0 over an interface thickness of 0.1 m (50 cells) is simulated, with the interface initially located between x = 0.1 m and x = 0.2 m. Such cases, with cells partially occupied by both fluids, present a particular difficulty for densitybased algorithms with approximate Riemann solvers. The profile of the velocity difference Δu = |∂u/∂x|Δx, pressure difference Δp = |∂p/∂x|Δx and density ρ at t = 0.7 s are shown in Fig. 7. The differences of pressure and velocity are negligible in the entire domain, while the density profile and, consequently, the position of the interface are predicted accurately, demonstrating that a smooth interface does not present a problem for the presented algorithm. As in the case of the sharp interface discussed above, the interface (i.e. the discontinuity in colour function and fluid properties) does not artificially diffuse.

7.2. Conservation of momentum, mass and energy In order to avoid spurious oscillations at the interface, the conservation of momentum, mass and energy at the interface is often relaxed or sacrificed [2, 5, 14, 23]. As discussed in Section 5, the proposed discretisation method ACID applies a correction to the discretisation at the interface to account for the change in fluid properties. The fully-conservative formulation and discretisation of the governing flow equations is, nevertheless, retained in the finite-volume discretisation stencil of each mesh cell. Furthermore, the compressive VOF method applied in the proposed algorithm was previously shown to conserve the volume of each phase in incompressible flows [77]. The advection of a circular bubble with constant velocity is simulated. The fluid properties are identical to the one-dimensional interface advection in Section 7.1. The computational domain is represented by an equidistant Cartesian mesh with 25 cells per bubble diameter. Because the domain is periodic, flows over the domain boundaries do not have to be accounted for, and the relative error associated with the conserved

17


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq307.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq308.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq309.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq310.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq311.png)

0 0.2 0.4 0.6 0.8 1

|Δu| [m/s]

x [m]

(a) Velocity difference Δu


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq312.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq313.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq314.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq315.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq316.png)

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq317.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq318.png)

0

0.2

0.4

0.6

0.8

1

1.2

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq319.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq320.png)


> **Figure 7: Profiles of (a) velocity difference Δu = |∂u/∂x| Δx, (b) pressure difference Δp = |∂p/∂x| Δx and (c) density of the smooth interface advected with constant velocity u0 = 1 m s−1 on a mesh with 500 cells at t = 0.7 s.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq321.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq322.png)

0 2 4 6 8 10


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq323.png)

tu0/d0

Domain

Bubble

(a) Momentum conservation error


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq324.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq325.png)

0 2 4 6 8 10


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq326.png)

tu0/d0 (b) Mass conservation error


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq327.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq328.png)

0 2 4 6 8 10


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq329.png)

tu0/d0 (c) Energy conservation error


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq330.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq331.png)

0 2 4 6 8 10


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq332.png)

tu0/d0 (d) Volume conservation error


> **Figure 8: Conservation error of (a) momentum ερu, (b) mass ερ and (c) energy ερh with respect to the domain and the bubble, and (d) conservation error of the bubble volume Vb, for the advection of a circular bubble on an equidistant Cartesian mesh as a function of dimensionless time tu0/d0, where u0 is the velocity and d0 is the initial diameter of the bubble.**

variables Φ ∈{ρ|u|, ρ, ρh} is given as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq333.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq334.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq335.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq336.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq337.png)

where superscript (0) denotes values at reference time t = 0. Figure 8 shows the relative conservation errors ε of momentum, mass and energy for the entire domain and for the bubble, as a function of time. The global (domain) conservation errors are negligible and the governing conservation laws are satisfied accurately. With respect to the bubble, the conservation errors are still insignificant but up to one order of magnitude higher than the corresponding global errors. This can be attributed to the volume conservation error of the applied VOF method, shown in Fig. 8d, which is the leading error of the conservation of the bubble.

7.3. Acoustic waves The propagation of acoustic waves in a one-dimensional domain is simulated to study the capabilities of the proposed algorithm to predict acoustic effects. The accurate prediction of the formation and propagation of acoustic waves is an important feature for the simulation of compressible flows and was previously found to be a particularly challenging problem [76, 112], because even small inconsistencies in the discretisation or a lack of convergence can lead to a visible change in the amplitude and speed of the waves. In the presented simulations, the acoustic waves are generated at the domain-inlet by a sinusoidal velocity perturbation with (small) amplitude Δu0. For small perturbations to the flow, with density amplitude Δρ ≪ρ0 and velocity

18


> **Table 1: Fluid properties for the propagation of acoustic waves. The heat capacity ratio γ and pressure constant Π of water and copper are taken from [6].**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq338.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq339.png)

amplitude Δu ≪a0, the resulting wave is a sound wave. The propagation speed of such a small-amplitude acoustic wave is the speed of sound a0 and, according to linear acoustic theory, the resulting amplitudes of the density waves Δρ0 and the pressure waves Δp0 are [102]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq340.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq341.png)

Five different materials are considered, for which the fluid properties are given in Table 1. In all cases, the computational domain is initialised with uniform velocity u0 = 1.0 m s−1 and pressure p0 = 105 Pa. The central differencing scheme is applied for the presented single-phase flows and the Minmod scheme is applied for the interfacial flows, while the second-order backward Euler scheme is applied to discretise the transient terms in both single-phase and interfacial flows. At the domain-inlet, pressure and temperature are extrapolated from the closest cell centre, and at the domain-outlet, a zero-gradient condition is specified for all variables. The propagation of acoustic waves in single-fluid flows, either by considering only a single fluid or a constant mixture of two fluids (emulating the interface region between two fluids), is discussed in Section 7.3.1. Section 7.3.2 presents simulations of the reflection and transmission of acoustic waves in gas-gas and gas-liquid flows, and Section 7.3.3 investigates the prediction of acoustic waves in two-phase flows with acoustic impedance matching.

7.3.1. Propagation of acoustic waves Acoustic waves in air and water, as well as in the interface region of air-helium, air-water and watercopper flows are simulated to demonstrate the accurate prediction of the speed of sound and pressure waves in the bulk phases and the interface region. The computational domain has a length of 1 m, represented by a Cartesian mesh with mesh spacing Δx = 2 × 10−3 m, and the applied time-step corresponds to a Courant number of Co = a0Δt/Δx = 0.44 −0.52. The velocity at the domain-inlet is uin = u0 + Δu0 sin(2πft), with frequency f and amplitude Δu0 = 0.01 u0. Figure 9 shows the computed pressure profile Δp and density profile Δρ in air for f = 2000 s−1, alongside the pressure profile ρ0a0Δu and density profile ρ0Δu/a0 obtained from linear acoustic theory, Eq. (68), using the computed velocity wave Δu. The computationally predicted amplitudes of the pressure waves Δp = ±4.022 Pa and the density waves Δρ = ±3.324 × 10−5 kg m−3 are in excellent agreement with the theoretical amplitudes of the pressure waves Δp0 = ±4.025 Pa and the density waves Δρ0 = ±3.328 × 10−5 kg m−3

following Eq. (68). The computed wavelength of λ = 0.1740 m compares also well to the theoretical wavelength λ0 = a0/f = 0.1739 m. The pressure and density profile for acoustic waves in water (f = 6000 s−1) are shown in Fig. 10. The amplitude of the pressure and density waves (Δp = ±13416 Pa, Δρ = ±7.420 × 10−3 kg m−3) as well as the wavelength (λ = 0.2236 m) are in excellent agreement with the theoretical values based on linear acoustic theory, Eq. (68), which are Δp0 = ±13419 Pa, Δρ0 = ±7.422 × 10−3 kg m−3 and λ0 = 0.2241 m. The propagation of acoustic waves in the interface region of helium-air, air-water and water-copper twophase flows are simulated by explicitly setting a constant value of the colour function ψ in the entire domain. The fluid properties are then evaluated based on the definitions given in Section 5. Figure 11 shows the speed of sound based on the computed wavelength of the acoustic waves as a function of the colour function ψ for all three cases. The computed speeds of sound are in excellent agreement with the theoretical value given by Eq. (57), with an error of < 0.33%. In summary, the propagation of acoustic waves in gases, liquids and solids is predicted accurately, especially considering that the density wave typically has an amplitude that is five orders of magnitude smaller than the ambient density. The pressure and density waves exhibit a constant amplitude as they propagate downstream, which is expected for an inviscid flow simulated with second-order discretisation schemes, as discussed in Section 3.6. In fact, at the end of a domain with a length of 50 m (Δx = 2 × 10−3 m, Co = 0.5), the amplitude of velocity, pressure and density of an acoustic wave in air (f = 500 s−1) are all > 99.75% of their initial value at the domain inlet. The definition of the speed of sound given in Eq. (57) is found to be accurate in the bulk phases and throughout the interface region (i.e. where a mixture of two bulk phases is present), demonstrating that ACID yields a thermodynamically-consistent discretisation.

19


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq342.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq343.png)

0

2

4

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq344.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq345.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq346.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq347.png)

(a) Pressure profile


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq348.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq349.png)

0

20

40

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq350.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq351.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq352.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq353.png)

(b) Density profile


> **Figure 9: Pressure and density profiles of acoustic waves with velocity amplitude Δu0 = 0.01u0 and frequency f = 2000 s−1**

at t = 2.3 × 10−3 s in air, compared against the pressure and density profiles obtained with linear acoustic theory, Eq. (68), using the computed velocity profile. The amplitude of the pressure wave ±Δp0 and density wave ±Δρ0 based on linear acoustic theory, as well as the theoretical wavelength λ0, are shown as a reference.


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq354.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq355.png)

0 5 10 15

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq356.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq357.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq358.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq359.png)

(a) Pressure profile


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq360.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq361.png)

0

4

8

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq362.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq363.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq364.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq365.png)

(b) Density profile


> **Figure 10: Pressure and density profiles of acoustic waves with velocity amplitude Δu0 = 0.01u0 and frequency f = 6000 s−1**

at t = 6.5 × 10−4 s in water, compared against the pressure and density profiles obtained with linear acoustic theory, Eq. (68), using the computed velocity profile. The amplitude of the pressure wave ±Δp0 and density wave ±Δρ0 based on linear acoustic theory, as well as the theoretical wavelength λ0, are shown as a reference.

400

600

800

1000

0 0.2 0.4 0.6 0.8 1

a [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq366.png)

Simulation

Theory

(a) Air-helium flow

400

600

800

1000

1200

1400

0 0.2 0.4 0.6 0.8 1

a [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq367.png)

Simulation

Theory

(b) Air-water flow

1200 1600 2000 2400 2800 3200 3600 4000

0 0.2 0.4 0.6 0.8 1

a [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq368.png)

Simulation

Theory

(c) Water-copper flow


> **Figure 11: Computed speed of sound a as a function of colour function ψ in (a) an air-helium flow (air represented by ψ = 0, helium represented by ψ = 1), (b) an air-water flow (air represented by ψ = 0, water represented by ψ = 1) and (c) an water-copper flow (water represented by ψ = 0, copper represented by ψ = 1), compared against the theoretical value given by Eq. (57).**

20

0

1

2

3

4

5

0 0.2 0.4 0.6 0.8 1 1.2 1.4


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq369.png)

x [m]

ΔpHe,0 refl.


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq370.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq371.png)

Interface

t = 0.75 ms t = 1.75 ms

(a) Helium-air flow


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq372.png)

0 2 4 6 8 10

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq373.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq374.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq375.png)

Interface

t = 1.5 ms t = 2.5 ms

(b) Argon-air flow


> **Figure 12: Pressure profiles of a single acoustic wave in a helium-air flow (f = 5000 s−1) and an argon-air flow (f = 2000 s−1), with velocity amplitude Δu0 = 0.02 u0. The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference.**

7.3.2. Reflection and transmission at fluid interfaces The propagation of a single acoustic wave in a helium-air flow, an argon-air flow and an air-water flow is simulated in a one-dimensional domain with mesh spacing Δx = 2 × 10−3 m and with a time-step that corresponds to Co = a0Δt/Δx = 0.48. The single acoustic wave is initiated by the inlet-velocity


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq376.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq377.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq378.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq379.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq380.png)

with Δu0 = 0.02 u0. The frequency of the acoustic wave is f = 5000 s−1 in the helium-air flow and the air-water flow, and f = 2000 s−1 in the argon-air flow. Assuming the incident acoustic wave travels from left to right, a part of the pressure wave is transmitted to the right phase when the incident wave reaches the fluid interface, while the remaining part of the incident wave is reflected in the left phase. Based on linear acoustic theory, as the incident wave with its pressure amplitude Δpincid. L,0 in the left phase reaches the interface, the ratio of the pressure amplitudes of the transmitted acoustic wave in the right phase Δptrans. R,0 and of the reflected acoustic wave in the left phase Δprefl. L,0 is Δptrans. R,0 Δprefl. L,0 = 2ZR ZR −ZL , (70)

with Δptrans. R,0 = Δpincid. L,0 + Δprefl. L,0 , and ZL = ρLaL and ZR = ρRaR are the acoustic impedance of the left and right phase, respectively. The pressure amplitude of the reflected wave, thus, follows as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq381.png)

In Fig. 12, the pressure profiles of the acoustic wave in the helium-air flow and the argon-air flow are shown, before and after the interaction of the incident wave with the fluid interface. In the helium-air case, Fig. 12a, the pressure amplitude Δpincid. He = 3.306 Pa of the incident wave is predicted accurately compared to linear acoustic theory (Δpincid. He,0 = 3.307 Pa). Similarly, the computed pressure amplitudes of the reflected wave in the helium phase Δprefl. He = 1.377 Pa and of the transmitted wave in the air phase Δptrans. Air = 4.688 Pa are in excellent agreement with the theoretical values (Δprefl. He,0 = 1.379 Pa, Δptrans. Air,0 = 4.686 Pa). In the argon-air case, Fig. 12b, the pressure amplitudes of the incident, reflected and transmitted acoustic waves are also in very good agreement with linear acoustic theory. The ratio of pressure amplitudes is predicted as Δptrans. Air /Δprefl. Ar = −5.919, compared to the theoretical value of Δptrans. Air,0 /Δprefl. Ar,0 = −5.871. Since the presented flows of two interacting ideal gases with acoustic waves is isentropic, the change in specific entropy is Δs = 0, where


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq382.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq383.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq384.png)

The computed change in entropy as the acoustic wave propagates in the bulk phases of the helium-air flow is clearly negligible, as seen in the inset of Fig. 13. An increase of |Δs| can be observed in Fig. 13 when the acoustic wave interacts with the helium-air interface (at t/tΣ ≈1, with tΣ the time required for the peak of the acoustic wave to reach the interface), a change in specific entropy that can, nevertheless, be regarded as inconsequential given its insignificant magnitude.

21


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq385.png)

0


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq386.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq387.png)

0 0.25 0.5 0.75 1 1.25 1.5


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq388.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq389.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq390.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq391.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq392.png)

0.2 0.4 0.6 0.8


> **Figure 13: Change in specific entropy Δs while a single acoustic wave propagates in the helium-air flow, as a function of dimensionless time t/tΣ, where tΣ is the time required for the peak of the acoustic wave to reach the interface. The inset shows a magnified view of the change in specific entropy Δs on a logarithmic scale before the acoustic wave has reached the interface.**

0

4

8

12

16

0 0.2 0.4 0.6 0.8 1 1.2 1.4


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq393.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq394.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq395.png)

Interface

t = 1 ms t = 2 ms


> **Figure 14: Pressure profile of a single acoustic wave in an air-water flow (f = 5000 s−1), with velocity amplitude Δu0 = 0.02 u0. The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference. Note that Δpincid. Air,0 and Δprefl. Air,0 differ by only 0.06% and are, thus, not indicated separately.**


> **Figure 14 shows the pressure profile for the acoustic wave in the air-water flow, alongside the theoretical pressure amplitude and wavelength given by linear acoustic theory. Contrary to the interaction of two perfect gases discussed above, water is described by the stiffened-gas model and, thus, large differences in acoustic impedance and density between the two interacting bulk phases ensue; the acoustic impedance and the density of water are approximately three orders of magnitude larger than for air. The amplitudes of the reflected and transmitted pressure waves are in very good agreement with linear acoustic theory, with the ratio of pressure amplitudes obtained from the simulation being Δptrans. Water/Δprefl. Air = 1.995, compared to the theoretical value of Δptrans. Water,0/Δprefl. Air,0 = 2.001.**

7.3.3. Gas-gas flow with acoustic impedance matching An acoustically neutral two-phase flow, in which both bulk phases have the same acoustic impedance, is considered. Due to the acoustic impedance matching, no acoustic waves should be reflected at the interface. In the first case, the acoustic impedance of both bulk phases is Z = 423.588 Pa s m−1 with the fluid properties


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq396.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq397.png)

and in the second case, the acoustic impedance of both bulk phases is Z = 500 Pa s m−1 with the fluid properties


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq398.png)

The initial velocity in both cases is u0 = 0.30886 m s−1. The result of the first case (Z = 423.588 Pa s m−1) is shown in Fig. 15a for a sinusoidal velocity perturbation with f = 2000 s−1 and Δu0 = 0.01u0. The observed amplitude of the pressure wave is in excellent agreement with linear acoustic theory in both phases, while the change in wavelength as a result of the different speeds of sound is predicted accurately and is clearly visible. Note that a reflected wave in the left phase would lead to a strong interference with the oncoming incident waves, which would be visible in the pressure profile. The result of the second case (Z = 500 Pa s m−1) is shown in Fig. 15b for a single sinusoidal wave with f = 5000 s−1 and Δu0 = 0.02u0. A small reflected wave can be identified in the pressure profile of the left phase at t = 0.9 × 10−3 s. Nevertheless, this reflected wave has only a minor effect on the transmitted wave in the right phase, which has a pressure amplitude of Δptrans. R = 3.0797 Pa, compared to the theoretical

22


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq399.png)

0 0.5

1

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq400.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq401.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq402.png)

Interface


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq403.png)

0 0.5

1 1.5

2 2.5

3

0 0.2 0.4 0.6 0.8 1 1.2 1.4


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq404.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq405.png)

Interface

t = 0.4 ms t = 0.9 ms


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq406.png)


> **Figure 15: Pressure profiles of (a) sinusoidal acoustic waves with velocity amplitude Δu0 = 0.01 u0 and frequency f = 2000 s−1**

in an acoustically neutral two-phase flow with Z = 423.588 Pa s m−1 at t = 3.3 × 10−3 s, and (b) a single acoustic wave with velocity amplitude Δu0 = 0.02 u0 and frequency f = 5000 s−1 in an acoustically neutral two-phase flow with Z = 500 Pa s m−1, before and after the acoustic wave has reached the interface. The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference.

value of Δpincid. L,0 = Δptrans. R,0 = 3.0886 Pa. At the time the peak of the acoustic wave reaches the interface (t ≈1.1×10−3 s), the interface is located at xΣ = 5.0034×10−1 m and the corresponding colour function value in the interfacial cell (Δx = 2 × 10−3 m) is ψ = 0.83, with an acoustic impedance of Z = 489.46 Pa s m−1. This constitutes an error in acoustic impedance at the interface of 2.1%, which is due to the definition of the fluid properties in the interface region, discussed in Section 5. This error in acoustic impedance explains the small discrepancy (Δprefl. L /Δptrans. R = 8.77 × 10−3) in the pressure amplitude between the simulation (Δprefl. L = 0.0270 Pa) and linear acoustic theory (Δprefl. L,0 = 0).

7.4. Shock waves The simulation of the propagation of shock wave poses a particular difficulty for numerical algorithms, because shock waves are discontinuous and because valid solutions of the governing conservation laws (presented in Section 2) are not guaranteed to satisfy the second law of thermodynamics across shock waves [9]. This raises the question whether the proposed algorithm reliably converges to the physically-correct weak solution of the governing conservation laws, which is a prerequisite for the accurate prediction of both the speed and strength of strong shock waves [8, 9]. The simulation of strong shock waves in air and water is examined in Section 7.4.1 and the interaction of shock waves in a single-phase flow is studied in Section 7.4.2. The interaction of shock waves with an air-helium interface and an air-water interface are presented in Sections 7.4.3 and 7.4.4, respectively. The interaction of a shock wave with an interface separating to fluids with the same shock impedance is discussed in Section 7.4.5.

7.4.1. Shock wave propagation The propagation of a shock wave travelling to the right with Mach number Ms = us/aR in air and water, with γ and Π given in Table 1, in a computational domain with a length of 1 m is simulated. The pre-shock region (II) has pressure pII = 105 Pa and velocity uII = 0, with ρAir,II = 1.1574 kg m−3 and ρWater,II = 998 kg m−3, while the post-shock region (I) is initialised based on the Rankine-Hugoniot relations [33]. The first-order backward Euler scheme and the first-order upwind scheme are applied. The shock wave is initially located at xs,0 = 0.1 m and the simulations are concluded at tend = 0.7 m/us. Hence, at the end of the simulation the shock wave is theoretically located at xs,end = 0.8 m. The applied time-step corresponds to Co = usΔt/Δx = 0.5. Figures 16 and 17 show the pressure profile of a shock wave with Mach numbers 10 and 100, respectively, in air and water at the end of the simulation. The pressure profiles of the shocks, and in particular the shock position, are predicted accurately in both fluids. The shock with Ms = 100 in water has a pressure ratio of pI/pII = 7.0754 × 107 and the bulk modulus in the post-shock region is BWater,I = ρa2 = 2.9 × 1013 Pa (for comparison, BAir,II = 1.4 × 105 Pa), which demonstrates the robustness of the proposed numerical algorithm even for strong shock waves (large pressure ratio) and marginally compressible fluids. The smearing of the shock wave reduces with mesh refinement, shown in Fig. 16, and the computed shock wave reproduces the solution of the corresponding Riemann problem with increasing precision. The common intersection point of the pressure profiles for different mesh resolutions observed in Fig. 16, which coincides with the intersection point with the corresponding Riemann solution, demonstrates that the speed of the shock wave is predicted accurately irrespective of the mesh resolution. In Section 5.3, the Rankine-Hugoniot relations for the interface region are presented. These relations are tested using a constant air-water mixture, with the properties of the individual (pure) phases being the same as above. The domain is initialised based on the Rankine-Hugoniot relations given in Section 5.3. As for the

23

0

2

4

6

8

10

12

0.7  0.75  0.8  0.85  0.9  0.95 1

p [MPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq407.png)

(a) Air

0 10 20 30 40 50 60 70 80

0.7  0.75  0.8  0.85  0.9  0.95 1

p [GPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq408.png)

(b) Water


> **Figure 16: Pressure profiles on meshes with different Δx of a shock wave with Ms = 10 in (a) air and (b) water. The theoretical Riemann solution given by the Rankine-Hungoniot relations is shown as a reference.**

0

0.2

0.4

0.6

0.8

1

1.2

0 0.2 0.4 0.6 0.8 1

p [GPa]

x [m]

Theory Simulation

(a) Air

0 1 2 3 4 5 6 7 8

0 0.2 0.4 0.6 0.8 1

p [TPa]

x [m]

Theory Simulation

(b) Water


> **Figure 17: Pressure profiles of a shock wave with Ms = 100 in (a) air and (b) water on a mesh with Δx = 2 × 10−3 m. The theoretical Riemann solution given by the Rankine-Hungoniot relations is shown as a reference.**

single-phase cases discussed in the previous paragraph, the applied time-step corresponds to Co = 0.5, the shock wave is initially located at xs,0 = 0.1 m and the simulations are concluded at tend = 0.7 m/us. Figure 18 shows the pressure profile of a shock wave with Ms = 10 in an air-water flow with ψ ∈{0.25, 0.50, 0.75}, where ψ = 0 represents air and ψ = 1 represents water. The pressure profiles are in excellent agreement with the theoretical Riemann solution and converge with mesh refinement. Given the pressure-based formulation of the proposed algorithm, the explicit evaluation of the density may be considered the “Achilles’ heel” of the numerical framework. The L1-norm of the relative error associated with the density field, given for a computational mesh with N cells as


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq409.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq410.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq411.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq412.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq413.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq414.png)

where ρ(comp.) P is the density computed with the presented numerical algorithm at cell P and ρ(exact) P is the corresponding exact density given by the Riemann solution at the centre of cell P, on meshes with different resolutions for the shock wave with Ms = 10 is shown in Fig. 19 for air and water single-phase flows, as well

0

1

2

3

4

5

6

0.7  0.75  0.8  0.85  0.9  0.95 1

p [GPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq415.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq416.png)

0 2 4 6 8 10 12 14 16

0.7  0.75  0.8  0.85  0.9  0.95 1

p [GPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq417.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq418.png)

0 5 10 15 20 25 30 35

0.7  0.75  0.8  0.85  0.9  0.95 1

p [GPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq419.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq420.png)


> **Figure 18: Pressure profiles on meshes with different Δx of a shock wave with Ms = 10 in an air-water mixture with (a) ψ = 0.25, (b) ψ = 0.50 and (c) ψ = 0.75, where ψ = 0 represents air and ψ = 1 represents water. The theoretical Riemann solution given by the Rankine-Hungoniot relations is shown as a reference.**

24


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq421.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq422.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq423.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq424.png)

l1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq425.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq426.png)

(a) Air


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq427.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq428.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq429.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq430.png)

l1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq431.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq432.png)

(b) Water


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq433.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq434.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq435.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq436.png)

l1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq437.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq438.png)

(c) Air-water mixture


> **Figure 19: L1-norm of the density error ℓ1 as a function of the mesh spacing Δx of a shock wave with Ms = 10 in (a) air, (b) water and (c) an air-water mixture with ψ = 0.75.**

0 1 2 3 4 5 6 7

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq439.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq440.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq441.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq442.png)

0

5

10

15

20

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]

Reference

400 cells 3200 cells


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq443.png)


> **Figure 20: Profiles of density and velocity of the shock wave interaction on equistant meshes with 400 and 3200 cells at t = 0.016 s, compared against the reference results reported by Woodward and Colella [114].**

as the air-water mixture with ψ = 0.75. For all considered cases, the density error is of similar magnitude and converges with first order, as imposed by the applied monotone discretisation schemes and as expected for an oscillation-free numerical simulation of a shock wave [94]. The error convergence under mesh refinement, supported by the improved shock resolution observed qualitatively in Figs. 16 and 18, is expected from a discretely conservative algorithm and, by virtue of the Lax-Wendroff theorem [113], suggests convergence to the physically-correct weak solution of the governing conservation laws [8]. Note that a better shock resolution can, of course, also be obtained by applying a TVD scheme or by reducing the applied time-step, which is not shown here in the interest of conciseness, but was previously demonstrated by Xiao et al. [76] using the single-phase framework the proposed algorithm is based on.

7.4.2. Shock wave interaction The interaction of two shock waves in a closed, one-dimensional domain, as previously studied by Woodward and Colella [114], is simulated. The one-dimensional domain is 1 m in length and occupied by a gas with γ = 1.4 and cv = 720 J kg−1 K−1. Initially, the density and velocity are ρ0 = 1 kg m−3 and u0 = 0 m s−1, respectively, in the entire domain. The initial pressure is pL,0 = 1000 Pa in the left state (0 m ≤x ≤0.1 m), pM,0 = 0.01 Pa in the middle state (0.1 m < x ≤0.9 m), and pR,0 = 100 Pa in the right state (0.9 m < x ≤1.0 m). The resulting flow involves the formation and interaction of strong shock waves, rarefaction fans and contact discontinuities. The profiles of density and velocity on equistant meshes with 400 and 3200 cells at t ∈{0.016 s, 0.038 s} are shown in Figs. 20 and 21, respectively, alongside the results reported by Woodward and Colella [114]. Note that the results of Woodward and Colella [114] were obtained on a mesh with 3096 cells, which was adaptively refined around flow discontinuities by factor 8, yielding an effective resolution equivalent to an equidistant mesh with 24768 cells. The results obtained on the finer mesh are in very good agreement with the reference results and a significant improvement of the solution, in particular for the density profile, is observed when the mesh spacing is reduced.

7.4.3. Air-helium interface The interaction of a planar shock wave with a planar air-helium interface is simulated. The shock wave is initially located at xs,0 = 0.05 m and travels in the left phase, which is assumed to be air, with speed us towards the air-helium interface that is initially situated at xΓ,0 = 0.15 m. The shock wave separates the

25

0 1 2 3 4 5 6 7

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq444.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq445.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq446.png)

0

3

6

9

12

15

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]

Reference

400 cells 3200 cells


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq447.png)


> **Figure 21: Profiles of density and velocity of the shock wave interaction on equistant meshes with 400 and 3200 cells at t = 0.038 s, compared against the reference results reported by Woodward and Colella [114].**

0

0.3

0.6

0.9

1.2

1.5

0 0.1 0.2 0.3 0.4


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq448.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq449.png)

1

1.1

1.2

1.3

1.4

1.5

1.6

0 0.1 0.2 0.3 0.4

p [105 Pa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq450.png)

0

0.1

0.2

0.3

0.4

0.5

0 0.1 0.2 0.3 0.4

M

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq451.png)


> **Figure 22: Profiles of density, pressure and Mach number of the shock-interface interaction (Ms = 1.22) in an air-helium flow, t = 2 × 10−4 s after the shock has interacted with the interface, compared against the theoretical Riemann solution.**

post-shock region I and the pre-shock region II, which are initialised with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq452.png)

The heat capacity ratio and the specific heat capacity at constant volume of air are γAir = 1.4 and cv,Air = 720 J kg−1 K−1, respectively. Hence, the density of the air in the pre-shock region is ρII,Air = 1 kg m−3, resulting in a speed of sound of aII,Air = 376.65 m s−1. Following the experiments of Haas and Sturtevant [115] and the numerical study of Quirk and Karni [116] of the interaction between a shock and a helium bubble, the helium phase is assumed to be contaminated with 28% air by mass, with a heat capacity ratio of γHe = 1.648 and a specific heat capacity at constant volume of cv,He = 2440 J kg−1 K−1. The one-dimensional computational domain of 0.4 m length is represented by an equidistant mesh with 400 cells (Δx = 10−3 m) and a time-step of Δt = 1.25 × 10−7 s. The velocity and temperature at the domain-inlet are uin = uI and Tin = TI, and pressure is extrapolated from the closest cell centre. At the domain-outlet a zero-gradient condition is applied to all variables. According to the corresponding Riemann solution, the shock wave is observed to travel with a speed of us = 459.50 m s−1, which corresponds to Ms = us/aII,air = 1.22. Figure 22 shows the profiles of density, pressure and Mach number, at t = 2 × 10−4 s after the shock has interacted with the interface. A rarefaction fan is reflected in the air phase and a shock wave transmitted in the helium phase. All computed variables compare very well with the corresponding Riemann solution, including the strength and speed of the shock wave transmitted to the helium phase, which is propagating at us = 1079.90 m s−1 (Ms = 1.13). The position of the contact discontinuity (coinciding with the fluid interface) and the associated change in density and Mach number are also predicted accurately, as observed in Fig. 22.

7.4.4. Air-water interface The interaction of a shock wave with Ms = 10 with an air-water interface is simulated. Using the heat capacity ratio γ and pressure constant Π of air and water given in Table 1, the post-shock region I and pre-shock region II are initialised with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq453.png)

26

0

200

400

600

800

1000

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq454.png)

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq455.png)

0

20

40

60

80

100

0 0.2 0.4 0.6 0.8 1

p [MPa]

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq456.png)

0

500

1000

1500

2000

2500

3000

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq457.png)


> **Figure 23: Profiles of density, pressure and velocity of the shock-interface interaction (Ms = 10) in an air-water flow, t = 2.78 × 10−4 s after the shock has interacted with the interface, compared against the theoretical Riemann solution.**

0.8

1

1.2

1.4

0 0.1 0.2 0.3 0.4


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq458.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq459.png)

1

1.1

1.2

1.3

1.4

1.5

1.6

0 0.1 0.2 0.3 0.4

p [105 Pa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq460.png)

0

0.1

0.2

0.3

0 0.1 0.2 0.3 0.4

M

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq461.png)


> **Figure 24: Profiles of density, pressure and Mach number of the shock-interface interaction (Ms = 1.22) in a two-phase flow with shock impedance matching, t = 2 × 10−4 s after the shock wave has interacted with the interface, compared against the theoretical Riemann solution. The interface is located at xΓ = 0.175m.**

The applied one-dimensional domain is 1 m in length and represented with a mesh of 1000 equidistant cells (Δx = 10−3 m) and a time-step of Δt = 10−8 s. The shock wave and the air-water interface are initially situated at xs,0 = 0.25 m and xΓ,0 = 0.50 m, respectively. Figure 23 shows the profiles of density, pressure and velocity, at t = 2.78 × 10−4 s after the shock wave has interacted with the interface. The density profile computed with the proposed algorithm is in excellent agreement with the exact Riemann solution, despite the large density ratio of air and water. The pressure and velocity profiles are also in good agreement with the theoretical solution. However, small discrepancies can be identified in the pressure profile in comparison to the theoretical solution, with a wave forming upstream of the shock. This pressure wave actually originates during the initial contact of the shock wave with the interface, and propagates with the flow. Nevertheless, it does not have a lasting effect on the fidelity of the pressure distribution.

7.4.5. Gas-gas flow with shock impedance matching The interaction of a shock wave with a fluid interface when both bulk phases have the same shock impedance is considered. As a result of the shock impedance matching, no shock wave or rarefaction fan is reflected at the interface. Cases with shock impedance matching have previously been found to be problematic with GFM-based methods [46, 47], where spurious shock waves or rarefaction fans are reflected at the interface. For a non-reflective shock-interface interaction, the fluid properties and the pressure ratio across the shock wave have to satisfy [46]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq462.png)

Assuming the same setup as in Section 7.4.3 for the shock wave with Ms = 1.22 in an air-helium flow, γL = γair, ρII,L = ρII,air and the right phase has γR = 1.648, the specific isochoric heat capacity of the right phase follows from Eq. (74) as cv,R = 512.41 J kg−1 K−1. Figure 24 shows the profiles of density, pressure and Mach number, at t = 2 × 10−4 s after the shock-interface interaction. All variables are predicted accurately compared to the corresponding Riemann problem, without spurious reflections in the left phase or spurious oscillations at the interface. As observed for the previously discussed cases involving shock waves, the speed and position of the shock wave are computed with very high accuracy.

27

1

1.5

2

2.5

3

3.5

4

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq463.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq464.png)

1

1.2

1.4

1.6

1.8

2

0 0.2 0.4 0.6 0.8 1

p [105 Pa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq465.png)

0

20

40

60

80

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq466.png)

0

0.1

0.2

0.3

0 0.2 0.4 0.6 0.8 1

M

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq467.png)

400 500 600 700 800 900 1000 1100

0 0.2 0.4 0.6 0.8 1

Z [Pa s/m]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq468.png)


> **Figure 25: Profiles of density, pressure, velocity, Mach number and acoustic impedance of the subsonic gas-gas shock tube at t = 8 × 10−4 s, compared against the theoretical Riemann solution.**

7.5. Shock tubes Shock tubes are extensively used to test and scrutinise numerical frameworks for compressible flows, because an exact reference solution based on the associated Riemann problem is available and because they typically feature all three primary wave types (shock wave, rarefaction fan and contact discontinuity).

7.5.1. Gas-gas shock tube A subsonic and a transonic shock tube with two-phase flow are simulated to assess the handling of discontinuous data and the prediction of shock waves, rarefaction fans and contact discontinuities by the presented algorithm. In both cases the discontinuity of the fluid states is initially located in the middle of the one-dimensional domain with a length of 1 m, which is represented with 400 equidistant mesh cells (Δx = 2.5×10−3 s), with a time-step that corresponds to Co = aRΔt/Δx ≤0.27. The fluid interface initially coincides with the discontinuity of the fluid states. The initial conditions of the subsonic shock tube are


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq469.png)

and the initial conditions of the transonic shock tube are


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq470.png)

The profiles of density, pressure, velocity, Mach number and acoustic impedance are shown in Fig. 25 for the subsonic shock tube at t = 8 × 10−4 s and in Fig. 26 for the transonic shock tube at t = 6 × 10−4 s, alongside the corresponding theoretical Riemann solution. All shown quantities are in excellent agreement with the theoretical Riemann solution in both cases, irrespective of the Mach number regime. The accurate prediction of the acoustic impedance at the fluid interface further supports the notion that ACID conserves the acoustic properties of the two-phase flow, while the accurate prediction of the Mach number suggests that the hydrodynamic-thermodynamic coupling of the proposed algorithm is correct. The correct position of the shock wave, the rarefaction fan and the contact discontinuity in both cases demonstrates that the conservative discretisation of the governing flow equations leads to physically sound numerical results satisfying the Rankine-Hugoniot relations and entropy condition. Furthermore, the computational result converges to the theoretical Riemann solution under mesh refinement, as demonstrated by the density profile of the subsonic shock tube in Figs. 27a and 27b. The contact discontinuity occurring in the solution of the shock tubes, which coincides with the fluid interface, is a linearly degenerate wave and, thus, represents the primary difficulty and main source of error with respect to the convergence of the applied finite-volume method under mesh refinement [11, 104]. As

28

1 2 3 4 5 6 7 8 9

0 0.2 0.4 0.6 0.8 1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq471.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq472.png)

1

2

3

4

5

0 0.2 0.4 0.6 0.8 1

p [105 Pa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq473.png)

0

50

100

150

200

250

300

0 0.2 0.4 0.6 0.8 1

u [m/s]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq474.png)

0

0.2

0.4

0.6

0.8

1

1.2

0 0.2 0.4 0.6 0.8 1

M

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq475.png)

400

800

1200

1600

2000

2400

2800

0 0.2 0.4 0.6 0.8 1

Z [Pa s/m]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq476.png)


> **Figure 26: Profiles of density, pressure, velocity, Mach number and acoustic impedance of the transonic gas-gas shock tube at t = 6 × 10−4 s, compared against the theoretical Riemann solution.**

2.8

3

3.2

3.4

3.6

3.8

0.15  0.2  0.25  0.3  0.35  0.4  0.45


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq477.png)

x [m]

Theory 3200 cells 400 cells 50 cells

(a) At the rarefaction fan

1.5

2

2.5

3

0.45 0.5 0.55 0.6 0.65 0.7


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq478.png)

x [m]

Theory 3200 cells 400 cells 50 cells

(b) At the contact discontinuity


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq479.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq480.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq481.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq482.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq483.png)

l1


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq484.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq485.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq486.png)


> **Figure 27: Density profile at (a) the rarefaction fan and (b) at the contact discontinuity, and (c) the L1-norm of the density error ℓ1, of the subsonic gas-gas shock tube as a function of the mesh spacing. The theoretical Riemann solution is shown in (a) and (b) as a reference.**

shown rigorously by Banks et al. [104], the convergence of the L1-norm of the solution error associated with a linearly degenerate wave for a q-th order advection scheme without compressive limiting is of order q/(q +1). Hence, for the applied second-order Minmod scheme the order of convergence of the L1-norm of density ρ at the contact discontinuity is 2/3 [104]. The L1-norm of the density profile, given by Eq. (73), for the subsonic gas-gas shock tube is shown in Fig. 27c. The error of the density profile converges with order 0.657 (based on a regression fit of the data points, with R2 = 0.9987), which is in very good agreement with the theoretical value for the convergence rate of 2/3. This suggests that the ACID method does not adversely affect the convergence of the numerical framework, considering that the contact discontinuity coincides with the material interface.

7.5.2. Gas-liquid shock tube A gas-liquid shock tube is simulated, with air as the left phase and water as the right phase. The onedimensional domain of 2 m length is represented with an equidistant mesh of 800 cells (Δx = 2.5×10−3 s), and is initialised with a uniform velocity u0 = 0 and uniform temperature T0 = 300 K. The pressure in the left and right phases is initialised as pL = 109 Pa and pR = 104 Pa, respectively, with the gas-liquid interface initially situated at xΓ,0 = 0.5 m. The heat capacity ratio γ and pressure constant Π of air and water are taken from Table 1. The profiles of density, pressure, velocity, Mach number and acoustic impedance at t = 8 × 10−4 s are shown in Fig. 28, alongside the corresponding theoretical Riemann solution. Despite the large pressure ratio of pL/pR = 105, all quantities are in excellent agreement with the corresponding Riemann solution. A

29

0

2000

4000

6000

8000

10000

12000

0 0.4 0.8 1.2 1.6 2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq487.png)

x [m]

Simulation

Theory


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq488.png)

0

200

400

600

800

1000

0 0.4 0.8 1.2 1.6 2

p [MPa]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq489.png)

0

50

100

150

200

250

0 0.4 0.8 1.2 1.6 2

u [m/s]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq490.png)

0

0.2

0.4

0.6

0.8

0 0.4 0.8 1.2 1.6 2

M

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq491.png)

1 1.5

2 2.5

3 3.5

4 4.5

0 0.4 0.8 1.2 1.6 2

Z [MPa s/m]

x [m]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq492.png)


> **Figure 28: Profiles of density, pressure, velocity, Mach number and acoustic impedance of the air-water shock tube at t = 8 × 10−4 s, compared against the theoretical Riemann solution.**

small discrepancy of the acoustic impedance Z can be observed at the gas-liquid interface, associated with the nonlinear change of the speed of sound in the interface region (see Fig. 11b), which has, however, no lasting influence on the overall result of the shock tube.

7.6. Two-dimensional shock-bubble interaction The interaction of a shock wave with Ms = 1.22 in air with a circular bubble of helium and of R22 is simulated. These cases have previously been studied experimentally by Haas and Sturtevant [115] and considered in a number of numerical studies to investigate the governing physical mechanisms [116, 117] and to test numerical algorithms, e.g. [5, 31, 41]. The computational setup is schematically illustrated in Fig. 29. Since the problem is symmetric about the centreline of the domain, as indicated in Fig. 29, only one half of the domain is simulated. For the discrete representation of the computational domain, a Cartesian mesh with mesh spacing Δx = d0/500 is considered. The shock is initially situated at x = 0.17 m and travels from left to right at speed us. The shock wave separates the post-shock region I and the pre-shock region II, which are initialised with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq493.png)

The heat capacity ratio and the specific isochoric heat capacity of air are γAir = 1.4 and cv,Air = 720 J kg−1 K−1, respectively. The velocity and temperature at the domain-inlet are uin = uI and Tin = TI, and pressure is extrapolated from the closest cell centre. At the domain-outlet a zero-gradient condition is applied to all variables. The applied time-step for both cases satisfies a Courant number of Co = aAir,IIΔt/Δx = 0.38.

7.6.1. Helium bubble The properties of helium are identical to the experiments of Haas and Sturtevant [115], with heat capacity ratio γHe = 1.648 and specific isochoric heat capacity cv,He = 2440 J kg−1 K−1 (see also Section 7.4.3). The contours of the density gradients and the Mach number are shown for different time instances in Fig. 30. The results are in excellent agreement with the experimental results of Haas and Sturtevant [115], with regards to the interface shape, the shock position and the wave structure, cf. Fig. 7 in [115] where t = {62 μs, 102 μs, 245 μs, 427 μs} correspond to τ = t aHe,II/d0 = {0.92, 1.77, 4.50, 7.70} (note that the shock wave travelled from right to left in the experiments). At τ = 0.92 the shock wave reaches the far-side of the bubble interface, as in the experiments, and the computed position (and shape) of the reflected rarefaction fan, the Mach stem and the triple point are in very close agreement with the experimental observations. A good agreement between computations and experiments is also observed at τ = 1.77 with respect to the primary shock wave, the reflected waves immediately following the primary shock wave and the Mach stem.

30


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq494.png)

170 mm

220 mm

445 mm


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq495.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq496.png)

Inlet

Outlet

Wall


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq497.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq498.png)


> **Figure 29: Schematic illustration of the computational setup of the shock-bubble interaction with Mach number Ms = 1.22. The shock is initially located at x = 0.17 m and travels from left to right. The shaded area represents the bubble with a diameter of d0 = 0.05 m, with the bubble centre initially located at x = 0.22 m.**

Also note that the density gradient and, consequently, the interface representation stay sharp throughout the simulation, without artificial smearing of the interface. Furthermore, the presented results are also found to be in very good agreement with previous numerical studies, notably [31, 41, 116, 118]. The growth and magnitude of the instabilities forming at the interface, e.g. on the downstream side of the interface in Fig. 30f, exhibit significant differences between various numerical studies (cf. [5, 118]) and the “correct” development of the instabilities, although clearly present, cannot be precisely deduced from the experimental images of Haas and Sturtevant [115]. Johnsen and Colonius [118] showed that spurious pressure oscillations at the interface increase these instabilities artificially, leading to more pronounced interface instabilities predicted by methods that do not completely eliminate spurious pressure oscillations.

7.6.2. R22 bubble The properties of the R22 bubble are taken from the experimental study of Haas and Sturtevant [115], with heat capacity ratio γR22 = 1.249 and specific isochoric heat capacity cv,R22 = 365 J kg−1 K−1. The contours of the density gradients and the Mach number are shown for different time instances in Fig. 31. Similar to the results presented for the shock interaction with a helium bubble in the previous section, the simulation results obtained with the proposed pressure-based algorithm are in very good agreement with the experimental results reported by Haas and Sturtevant [115], cf. Fig. 11 in [115] where t = {55 μs, 135 μs, 187 μs, 247 μs} correspond to τ = t aR22,II/d0 = {0.20, 0.49, 0.68, 0.89} (note that the shock wave travelled from right to left in the experiments). The primary shock wave, the reflected shock wave in air and the transmitted shock wave in the R22 bubble are computed accurately at τ ∈{0.20, 0.49} compared to the experiments. After the primary shock wave has passed the bubble, see τ = 0.68 in Fig. 31c and τ = 0.89 in Fig. 31d, it collapses back on itself, leading to a number of reflected waves captured by the simulations, which are also observed in the experiments of Haas and Sturtevant [115]. The presented result also compare well to previous simulation results [110, 117, 119]. As mentioned in Section 3, the proposed algorithm is applicable on unstructured meshes. Figure 32 shows the R22 bubble during and after the impact of the shock wave on a triangular mesh with an equivalent mesh spacing (i.e. the mesh spacing of an equidistant Cartesian mesh with the same number of cells) of Δx = d0/293, alongside the results obtained on an equidistant Cartesian mesh with Δx = d0/300. The qualitative features pertaining to the shock wave and its reflections, as well as the interface shape, are in very good agreement. In general, more flow features are resolved in the bulk phases on the Cartesian mesh, while instabilities developing along the interface are more pronounced on the triangular mesh. Despite these small qualitative differences, the density, pressure and velocity profiles obtained on both meshes on a line along the x-axis at y = 0.002 m, shown in Fig. 33 for τ = 0.89, are in excellent agreement. Comparing the contours of the density gradient and the velocity magnitude obtained on Cartesian meshes with Δx ∈{d0/200, d0/300, d0/500} and triangular meshes with Δx ∈{d0/207, d0/293}, shown in Fig. 34, an increasing resolution of small flow features, particularly near the interface, is observed with increasing mesh resolution. This is to be expected, because viscous stresses, heat conduction or surface tension are neglected in these simulations and, thus, small flow features are not regulated. Nevertheless, the streamwise position xb and velocity ub of the centre of mass of the bubble, as well as the volume of the mass Vb, shown in Figs. 35 and 36, are in excellent agreement throughout the simulation on all considered meshes. For the same Cartesian and triangular meshes, the conservation of mass in the entire computational domain (accounting for the additional mass that enters through the domain inlet), shown in Fig. 37, is conserved accurately, with an error < 0.01% on the Cartesian meshes and < 0.03% on the triangular meshes. Similarly, the conservation of mass of the bubble is very good, see Fig. 38, being within 0.1% on the Cartesian meshes and within 1.0% on the triangular meshes.

31


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq499.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq500.png)


> **Figure 30: Contours of the density gradient (1 + 8ψ)|∇ρ| (upper half) and the Mach number M (lower half) of the shockinteraction (Ms = 1.22) with a two-dimensional helium bubble on an Cartesian mesh with Δx = 500/d0 at different time instances τ = t aHe,II/d0.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq501.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq502.png)


> **Figure 31: Contours of the density gradient (1 −0.75ψ)|∇ρ| (upper half) and the Mach number M (lower half) of the shockinteraction (Ms = 1.22) with a two-dimensional R22 bubble on an Cartesian mesh with Δx = 500/d0 at different time instances τ = t aR22,II/d0.**

32

(a) τ = 0.49 (b) τ = 0.68 (c) τ = 0.89


> **Figure 32: Contours of the density gradient (1 −0.75ψ)|∇ρ| on the triangular mesh with Δx = d0/293 (upper half) and the Cartesian mesh with Δx = d0/300 (lower half) of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble at different time instances τ = t aR22,II/d0.**

0

1

2

3

4

5

0.1 0.15 0.2 0.25 0.3


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq503.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq504.png)

0.1

0.15

0.2

0.25

0.1 0.15 0.2 0.25 0.3

p [MPa]

x [m]

Cartesian Triangular


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq505.png)

0

40

80

120

160

200

0.1 0.15 0.2 0.25 0.3

u [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq506.png)


> **Figure 33: Profiles of density, pressure and velocity along the x-axis at y = 0.002 m of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble on the Cartesian mesh (Δx = d0/300) and the triangular mesh (Δx = d0/293) at dimensionless time τ = t aR22,II/d0 = 0.89.**

33

(a) Cartesian, Δx = d0/200 (b) Cartesian, Δx = d0/300 (c) Cartesian, Δx = d0/500


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq507.png)


> **Figure 34: Contours of the density gradient (1 −0.75ψ)|∇ρ| (upper half) and the velocity magnitude |u| (lower half) of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble on the considered Cartesian and triangular meshes at τ = t aR22,II/d0 = 0.89.**

220

225

230

235

−0.2 0 0.2  0.4  0.6  0.8 1 1.2

xb [mm]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq508.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq509.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq510.png)

0

20

40

60

80

−0.2 0 0.2  0.4  0.6  0.8 1 1.2

ub [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq511.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq512.png)

0.6

0.7

0.8

0.9

1

1.1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq513.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq514.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq515.png)


> **Figure 35: Position xb and velocity ub of the centre of mass, and relative volume Vb/V (0) b , of the bubble of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble as a function of dimensionless time τ on the considered Cartesian meshes.**

220

225

230

235

−0.2 0 0.2  0.4  0.6  0.8 1 1.2

xb [mm]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq516.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq517.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq518.png)

0

20

40

60

80

−0.2 0 0.2  0.4  0.6  0.8 1 1.2

ub [m/s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq519.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq520.png)

0.6

0.7

0.8

0.9

1

1.1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq521.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq522.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq523.png)


> **Figure 36: Position xb and velocity ub of the centre of mass, and relative volume Vb/V (0) b , of the bubble of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble as a function of dimensionless time τ on the considered triangular meshes. The result obtained on the Cartesian mesh with Δx = d0/500 is shown as a reference.**

34


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq524.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq525.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq526.png)

0

0.01

0.02

0.03

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq527.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq528.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq529.png)

(a) Cartesian meshes


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq530.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq531.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq532.png)

0

0.01

0.02

0.03

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq533.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq534.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq535.png)

(b) Triangular meshes


> **Figure 37: Mass error ερ, given by Eq. (67), in the entire computational domain of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble as a function of dimensionless time τ for all considered Cartesian and triangular meshes.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq536.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq537.png)

0

0.05

0.1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq538.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq539.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq540.png)

(a) Cartesian meshes


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq541.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq542.png)

0

0.5

1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq543.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq544.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq545.png)

(b) Triangular meshes


> **Figure 38: Mass error ερ, given by Eq. (67), of the bubble of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble as a function of dimensionless time τ for all considered Cartesian and triangular meshes.**

7.6.3. Influence of the advection differencing scheme The applied numerical method has a direct influence on the development and evolution of the compression and expansion waves, as well as instabilities at the interface, as briefly mentioned in Section 7.6.1. For instance, Johnsen and Colonius [118] observed a substantial influence of spurious pressure oscillations at the interface on the onset and evolution of interfacial instabilities. Although spurious pressure oscillations are absent in the results presented above, a clear influence of the applied differencing schemes can be observed for the shock interaction with a helium bubble and a R22 bubble. Figures 39 and 40 show the contours of the density gradient and the velocity magnitude for the helium bubble at τ = 4.50 and τ = 7.70, respectively, obtained using the the Minmod scheme and the Superbee scheme [84], on a Cartesian mesh with Δx = d0/500. Because of the compressive limiting applied with the Superbee scheme, the compression and expansion waves are resolved sharper. Moreover, interface instabilities form more rapidly with the more compressive Superbee scheme.

(a) Minmod (b) Superbee


> **Figure 39: Contours of the density gradient (1 + 8ψ)|∇ρ| (upper half) and the velocity magnitude |u| (lower half) of the shockinteraction (Ms = 1.22) with a two-dimensional helium bubble on a Cartesian mesh with Δx = d0/500 at τ = t aHe,II/d0 = 4.50, using the Minmod scheme and the Superbee scheme.**

35

(a) Minmod (b) Superbee


> **Figure 40: Contours of the density gradient (1 + 8ψ)|∇ρ| (upper half) and the velocity magnitude |u| (lower half) of the shockinteraction (Ms = 1.22) with a two-dimensional helium bubble on a Cartesian mesh with Δx = d0/500 at τ = t aHe,II/d0 = 7.70, using the Minmod scheme and the Superbee scheme.**

(a) Upwind (b) Minmod (c) Superbee


> **Figure 41: Contours of the density gradient (1−0.75ψ)|∇ρ| (upper half) and the velocity magnitude |u| (lower half) of the shockinteraction (Ms = 1.22) with a two-dimensional R22 bubble on a Cartesian mesh with Δx = d0/500 at τ = t aR22,II/d0 = 0.89, using the first-order upwind scheme, the Minmod scheme and the Superbee scheme.**

A similar behaviour can be observed for the R22 bubble at τ = 0.89 using the first-order upwind scheme, the Minmod scheme and the Superbee scheme, for which the resulting density gradient and velocity magnitude are shown in Fig. 41. Apart from the expected sharper resolution of compression and expansion waves with increasing compression of the applied differencing scheme, the Mach number at the interface differs substantially; while the flow is entirely subsonic with the upwind scheme, the flow inside the bubble becomes transonic with the Minmod scheme and supersonic with the Superbee scheme. Although the temporal evolution of the volume of the R22 bubble is virtually the same irrespective of the applied differencing scheme, as seen in Fig. 42a, the conservation of the bubble mass is strongly affected by the applied differencing scheme, as observed in Fig. 42b. Based on these results, the Minmod scheme appears to provide the most feasible result, although this issue requires a further, more comprehensive investigation to draw firm conclusions about the most appropriate differencing scheme.

7.7. Three-dimensional shock-bubble interaction The interaction of a shock wave with Ms = 1.68 in air with a three-dimensional helium bubble is simulated, as previously studied in [117]. Since the problem is axisymmetric with respect to the x-axis, only one quarter of the domain in the y−z plane is simulated. The domain has the dimensions 6r0×4r0×4r0, where r0 = 0.0254 m is the initial radius of the bubble, and is represented by a Cartesian mesh with Δx = 5 × 10−4 m (50.8 cells per r0). Following Niederhaus et al. [117], the heat capacity ratio and specific isobaric heat capacity of air and helium are γAir = 1.399 and cp,Air = 1006.36 J kg−1 K−1, and γHe = 1.667 and cp,He = 5190.80 J kg−1 K−1, respectively. The centre of the bubble is initially located at xb,0 = 0.075 m and the shock wave (travelling in positive x-direction) is initialised at xs,0 = 0.02 m. The shock wave separates the post-shock region I and the

36

0.6

0.7

0.8

0.9

1

1.1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq546.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq547.png)

Upwind Minmod Superbee


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq548.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq549.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq550.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq551.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq552.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq553.png)

0

0.1

−0.2 0 0.2  0.4  0.6  0.8 1 1.2


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq554.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq555.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq556.png)


> **Figure 42: Relative volume Vb/V (0) b and mass error ερ, given by Eq. (67), of the bubble of the shock-interaction (Ms = 1.22) with a two-dimensional R22 bubble on a Cartesian mesh with Δx = d0/500 as a function of dimensionless time τ, using the first-order upwind scheme, the Minmod scheme and the Superbee scheme.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq557.png)


> **Figure 43: Contours of the density ρ (upper half) and the Mach number M (lower half) of the shock-interaction (Ms = 1.68) with a three-dimensional helium bubble on a Cartesian mesh at different time instances τs = t us,He/r0. Note that the same range and colour scale for density ρ are shown as previously used by Niederhaus et al. [117].**

pre-shock region II, which are initialised with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq558.png)

The contours of density and Mach number are shown in Fig. 43 for τs = t us,He/r0 ∈{1.4, 2.6, 5.2}, where us,He is the speed of the shock wave in helium, which are the same dimensionless time instances τs as considered by Niederhaus et al. [117]. The density field, position of the Mach stem, and the position and shape of the primary shock wave and the reflected rarefaction fan computed by the proposed algorithm agree very well with the results reported in [117] for all considered time instances. The density field at τ = 2.2 is shown in Fig. 44a together with the isocontour ρ = 2.5 kg m−3, which separates the pre-shock and the post-shock regions next to the bubble, and which follows the Mach stem and the rarefaction fan in the immediate vicinity and behind the bubble. The same result is shown in Fig. 44b but with the isocontour ρ = 1.5 kg m−3 that represents the shock front. Both isocontours in Fig. 44 are clearly axisymmetric, demonstrating that the proposed algorithm captures the three-dimensionality of the solution reliably.

7.8. Shock-drop interaction

The interaction of a shock wave with a liquid drop is simulated, with different Mach numbers of the shock wave. Contrary to the cases presented in Sections 7.6 and 7.7 where two perfect-gases interact, in the cases presented below a liquid drop described by a stiffened gas (Π ≫p) is surrounded by a perfect gas (Π = 0).

7.8.1. Shock-drop interaction with Ms = 1.47 The interaction of a shock wave with Mach number Ms = 1.47 in air with a circular water drop, as previously studied in [30, 120–122], is simulated. The flow is assumed to be symmetric and the computational setup is schematically illustrated in Fig. 45. The post-shock region I and the pre-shock region II are initialised with uI = 225.89 m s−1, pI = 2.386 × 105 Pa, TI = 381.20 K, uII = 0 m s−1, pII = 1.013 × 105 Pa, TII = 293.15 K.

37


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq559.png)


> **Figure 44: Contours of the density ρ in the symmetry plane and isocontours of the density for (a) ρ = 2.5 kg m−3 and (b) ρ = 1.5 kg m−3 of the shock-interaction (Ms = 1.68) with a three-dimensional helium bubble on a Cartesian mesh at dimensionless time τs = t us,He/r0 = 2.2. Note that not the entire computational domain is shown.**

Following the properties applied by Meng and Colonius [121], air has the properties γAir = 1.4, ΠAir = 0 and RAir = 287.08 J kg−1 K−1, while water is described by γWater = 6.12, ΠWater = 3.43 × 108 Pa and RWater = 7170.23 J kg−1 K−1. The computational domain is represented by an equidistant Cartesian mesh with Δx = d0/200 and the applied time-step corresponds to a Courant number of Co = aWater,IIΔt/Δx = 0.15. Figure 46 shows the density gradient and the velocity magnitude of the flow at different times t ∈ {7.74 μs, 16.18 μs, 23.00 μs}, with t = 7.74 μs corresponding to the dimensionless time t∗= 0.017 in [121]. The results are qualitatively in very good overall agreement with previously reported numerical results for this particular shock-drop interaction [121, 123]. The error in mass conservation ερ, shown in Fig. 47, is, however, about one order of magnitude larger than for the shock-bubble cases presented in Section 7.6.2. This increase in mass error extends to both the drop as well as the entire computational domain. Since the proposed framework is nominally mass-conservative, as demonstrated in Section 7.6.2, this error is likely the result of the finite interface thickness introduced by the applied VOF method and the associated misrepresentation of the fluid properties in this interface region, as further studied below.

7.8.2. Shock-drop interaction with Ms = 6 The interaction of a shock wave with Mach number Ms = 6 in gas with a circular liquid drop, similar to previous work of Shukla et al. [109], is simulated. The computational setup is schematically illustrated in Fig. 48. The post-shock region I and the pre-shock region II are initialised with


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq560.png)

Following the properties applied by Shukla et al. [109], the gas has the properties γgas = 1.4, Πgas = 0 and Rgas = 288 J kg−1 K−1, while the liquid is described by γliquid = 4.4, Πliquid = 6000 Pa and Rliquid = 7504 J kg−1 K−1. The applied time-step on all considered meshes corresponds to a Courant number of Co = aliquid,IIΔt/Δx = 0.23. The contours of the density gradient and the Mach number at time t ∈{0.08 s, 0.22 s, 0.34 s, 0.59 s} after the initial interaction of the shock wave and the drop are shown in Fig. 49 on an equidistant Cartesian mesh with Δx = d0/562. On this high resolution mesh, which has the same resolution as previously applied by Shukla et al. [109], the individual flow features are well resolved and clearly visible. The time instances t ∈{0.08 s, 0.22 s, 0.59 s} in Fig. 49 correspond to t ∈{0.25, 0.50, 0.75}, respectively, of Fig. 11 in [109]. The results are overall in very good agreement with the results reported by Shukla et al. [109]. Comparing the results on different meshes at t = 0.34 s, see Figs. 49c and 50, small but clearly visible differences in the shape and structure of the compression and expansion waves are observed. This indicates mesh-dependent differences in the interaction of the waves with the gas-liquid interface, since the mesh-dependence of the speed and position of these types of waves is in general negligible, as demonstrated in Sections 7.4 and 7.5. Because of the large differences in fluid properties, notably the density and heat capacity, in gas-liquid flows,

38


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq561.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq562.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq563.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq564.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq565.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq566.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq567.png)

Inlet

Outlet

Wall


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq568.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq569.png)


> **Figure 45: Schematic illustration of the computational setup of the shock-drop interaction with Mach number Ms = 1.47 in an air-water flow. The shock wave is initially located at xs,0 = 7.39 × 10−3 m and travels from left to right. The shaded area represents the water drop with a diameter of d0 = 4.8 × 10−3 m, with the drop centre initially located at xd,0 = 10.8 × 10−3 m.**

(a) t = 7.74 μs (b) t = 16.18 μs (c) t = 23.00 μs


> **Figure 46: Contours of the density gradient (1 −0.8ψ)|∇ρ| (upper half) and the velocity magnitude |u| (lower half) of the shock-interaction (Ms = 1.47) with a two-dimensional water drop on a Cartesian mesh with Δx = d0/200.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq570.png)

0

1

2

0 5 10 15 20 25


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq571.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq572.png)

Drop Domain


> **Figure 47: Mass error ερ, given by Eq. (67), of the drop and in the entire domain of the shock-interaction (Ms = 1.47) with a two-dimensional water drop as a function of time t on a Cartesian mesh with Δx = d0/200, where t = 0 corresponds to the initial interaction of the shock wave with the drop.**

39


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq573.png)

1 m

2 m

8 m

1 m


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq574.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq575.png)

Inlet

Outlet

Wall


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq576.png)

y


> **Figure 48: Schematic illustration of the computational setup of the shock-drop interaction with Mach number Ms = 6 in a gas-liquid flow. The shock wave is initially located at xs,0 = 1 m and travels from left to right. The shaded area represents the liquid drop with a diameter of d0 = 1.124 m, with the drop centre initially located at xd,0 = 2 m.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq577.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq578.png)


> **Figure 49: Contours of the density gradient (1 −0.8ψ)|∇ρ| (upper half) and the Mach number M (lower half) of the shockinteraction (Ms = 6) with a two-dimensional liquid drop on an equidistant Cartesian mesh with Δx = d0/562.**

even small differences of the interface position, likely caused by small errors in the interface advection (see Section 4.2), or mesh-dependent thickness of the interface, can have a significant effect on the wave-interface interactions. These differences also manifest in mass-conservation errors with respect to the computational domain and the drop, see Fig. 51. Yet, as expected from errors associated with the applied interface capturing scheme, the mass-conservation error reduces with mesh refinement. Even though qualitative differences are visible in the contour plots, and the mass-conservation errors are not negligible for the results obtained on different meshes, the position and velocity of the centre of mass of the drop and the volume of the drop, shown in Fig. 52, are in good agreement on the different meshes, and converge with mesh refinement.

8. Conclusions

A pressure-based algorithm for compressible interfacial flows has been presented, based on a fully-coupled finite-volume framework with a collocated variable arrangement that is applicable to unstructured meshes. The governing flow equations, which are discretely conservative in mass, momentum and energy, are solved for velocity, pressure and specific total enthalpy. Contrary to all previously published algorithms for compressible interfacial flows, density is not a solution variable but is computed based on pressure, temperature and the local fluid properties via an equation of state. To this end, the stiffened-gas model has been considered in this study to describe gases, liquids and solids. The proposed algorithm includes an acoustically-conservative

40


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq579.png)


> **Figure 50: Contours of the density gradient (1 −0.8ψ)|∇ρ| (upper half) and the Mach number M (lower half) of the shockinteraction (Ms = 6) with a two-dimensional liquid drop at t = 0.34 s, on equidistant Cartesian meshes with (a) Δx = d0/140.5 and (b) Δx = d0/281.**


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq580.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq581.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq582.png)

0

0 0.1 0.2 0.3 0.4 0.5 0.6


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq583.png)

t [s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq584.png)

(a) Domain


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq585.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq586.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq587.png)

0

0 0.1 0.2 0.3 0.4 0.5 0.6


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq588.png)

t [s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq589.png)

(b) Drop


> **Figure 51: Mass error ερ, given by Eq. (67), in (a) the entire computational domain and (b) of the drop of the shock-interaction (Ms = 6) with a two-dimensional liquid drop as a function of time t on the considered Cartesian meshes, where t = 0 corresponds to the initial interaction of the shock wave with the drop.**

1.99

2

2.01

2.02

2.03

2.04

0 0.1 0.2 0.3 0.4 0.5 0.6

xb [m]

t [s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq590.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq591.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq592.png)

0 0.02 0.04 0.06 0.08

0.1 0.12

0 0.1 0.2 0.3 0.4 0.5 0.6

ub [m/s]

t [s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq593.png)

0.994

0.996

0.998

1

1.002

0 0.1 0.2 0.3 0.4 0.5 0.6


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq594.png)

t [s]


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq595.png)


> **Figure 52: Position xb and velocity ub of the centre of mass, and relative volume Vb/V (0) b , of the drop of the shock-interaction (Ms = 6) with a two-dimensional liquid drop as a function of time t on the considered Cartesian meshes, where t = 0 corresponds to the initial interaction of the shock wave with the drop.**

41

interface discretisation (ACID), which performs a local manipulation to the discrete values of density and total enthalpy based on the interface position, and which does not require the solution of a Riemann problem or a priori knowledge of the wave structure to evaluate fluxes. Results for a variety of representative compressible gas-gas and gas-liquid interfacial flows have been presented, including the propagation of acoustic waves, shock tubes and the interaction of shock waves with one-, twoand three-dimensional interfaces. The transmission and reflection of acoustic waves at the interface has been shown to be captured accurately by the presented algorithm, including cases with acoustic impedance matching, a capability not previously reported in the literature. Similarly, the interaction of shock waves with the interface in one-, twoand three-dimensional flows, as well as gas-gas and gas-liquid shock tube problems, have been favourably compared against exact Riemann solutions and previously reported experimental and computational results, especially the temporal evolution of the shock position, of reflected shock waves and rarefaction fans, of the wave structure and the interface shape. In all considered cases, the presented algorithm reliably retains the acoustic properties of the fluids, while the accurate prediction of the Mach number suggests a correct hydrodynamic-thermodynamic coupling. Notably, the speed, position and strength of shock waves are predicted accurately and the Rankine-Hugoniot relations are satisfied even for strong shock waves, in the bulk phases and in the interface region, demonstrating that the proposed algorithm converges to the correct weak solution of the governing conservation laws [8–10]. This enforces the notion that, irrespective of small conservation errors associated with the applied interface capturing scheme, the governing equations remain discretely conservative and implies that the proposed algorithm implicitly satisfies the second law of thermodynamics. Despite the demonstrated success of the presented algorithm, several open questions remain. The EOS of the perfect-gas model or the stiffened-gas model is used in this study to evaluate the density based on pressure and temperature. However, most EOS are formulated in terms of internal energy rather than density. While this is not a problem for the considered gas models, and some real-gas models can readily be formulated in terms of density, such as the Peng-Robinson EOS [124], more complex EOS may not be easily inverted to a density formulation. Furthermore, the proposed interface discretisation method ACID implies a mechanical and thermal equilibrium. Mechanical equilibrium is a reasonable assumption for interfacial flows, since mechanical relaxation is associated with acoustic effects [21, 26] and, thus, occurs at a timescale comparable with the typically applied computational time-steps. The implicit assumption of thermal equilibrium, however, may present a limiting factor when more complex EOS are considered [4] and/or thermal relaxation is governed by diffusion [59], for instance if reactive flows are to be simulated. Neither of these questions have been addressed in the context of pressure-based methods and warrant further study. In summary, the proposed pressure-based algorithm has been shown to be a promising alternative to traditional density-based algorithms for the simulation of compressible interfacial flows. The pressure-based formulation of the governing equations facilitates the definition of consistent mixture rules at the interface, including the Rankine-Hugoniot relations, and applies naturally to flows in all Mach number regimes.

Acknowledgements

The authors gratefully acknowledge financial support from the Engineering and Physical Sciences Research Council (EPSRC) through grant EP/M021556/1 and from Shell Corporation. The contribution of all four anonymous reviewers through their detailed and critical comments is greatly appreciated.


## References

[1] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible multicomponent flows, Journal of Computational Physics 274 (2014) 95–121. [2] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: A quasi conservative approach, Journal of Computational Physics 125 (1996) 150–160. [3] K.-M. Shyue, An Efficient Shock-Capturing Algorithm for Compressible Multicomponent Problems, Journal of Computational Physics 142 (1998) 208–242. [4] G. Allaire, S. Clerc, S. Kokh, A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids, Journal of Computational Physics 181 (2002) 577–616. [5] A. Marquina, P. Mulet, A flux-split algorithm applied to conservative models for multicomponent compressible flows, Journal of Computational Physics 185 (2003) 120–138. [6] G. Perigaud, R. Saurel, A compressible flow model with capillary effects, Journal of Computational Physics 209 (2005) 139–178. [7] X. Hu, B. Khoo, N. Adams, F. Huang, A conservative interface method for compressible flows, Journal of Computational Physics 219 (2006) 553–578. [8] T. Y. Hou, P. G. L. Floch, Why Nonconservative Schemes Converge to Wrong Solutions: Error Analysis, Mathematics of Computation 62 (1994) 497–530. [9] C. B. Laney, Computational Gasdynamics, Cambridge University Press, Cambridge; New York, NY, 1998. [10] D. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Computers & Fluids 32 (2003) 1113–1132.

42

[11] A. Harten, High Resolution Schemes for Hyperbolic Conservation Laws, Journal of Computational Physics 49 (1983) 357–393. [12] P. Sweby, High Resolution Schemes Using Flux Limiters for Hyperbolic Conservation Laws, SIAM Journal on Numerical Analysis 21 (1984) 995–1011. [13] E. Caramana, M. Shashkov, P. Whalen, Formulations of artificial viscosity for multi-dimensional shock wave computations, Journal of Computational Physics 144 (1998) 70–97. [14] S. Karni, Multicomponent flow calculations by a consistent primitive algorithms, Journal of Computational Physics 112 (1994) 31–43. [15] R. Abgrall, S. Karni, Computations of compressible multifluids, Journal of Computational Physics 169 (2001) 594–623. [16] J. Brackbill, D. Kothe, C. Zemach, Continuum Method for Modeling Surface Tension, Journal of Computational Physics 100 (1992) 335–354. [17] M. Rudman, A volume-tracking method for incompressible multifluid flows with large density variations, International Journal for Numerical Methods in Fluids 28 (1998) 357–378. [18] M. Raessi, H. Pitsch, Consistent mass and momentum transport for simulating incompressible interfacial flows with large density ratios using the level set method, Computers & Fluids 63 (2012) 70–81. [19] M. Baer, J. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials, International Journal of Multiphase Flow 12 (1986) 861–889. [20] P. Embid, M. Baer, Mathematical analysis of a two-phase continuum mixture theory, Continuum Mechanics and Thermodynamics 4 (1992) 279–312. [21] R. Saurel, R. Abgrall, A Multiphase Godunov Method for Compressible Multifluid and Multiphase Flows, Journal of Computational Physics 150 (1999) 425–467. [22] A. Ambroso, C. Chalons, P.-A. Raviart, A Godunov-type method for the seven-equation model of compressible two-phase flow, Computers & Fluids 54 (2012) 67–91. [23] R. Abgrall, R. Saurel, Discrete equations for physical and numerical compressible multiphase mixtures, Journal of Computational Physics 186 (2003) 361–396. [24] N. Andrianov, G. Warnecke, The Riemann problem for the Baer–Nunziato two-phase flow model, Journal of Computational Physics 195 (2004) 434–464. [25] D. Schwendeman, C. Wahle, A. Kapila, The Riemann problem and a high-resolution Godunov method for a model of compressible two-phase flow, Journal of Computational Physics 212 (2006) 490–526. [26] A. K. Kapila, R. Menikoff, J. B. Bdzil, S. F. Son, D. S. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations, Physics of Fluids 13 (2001) 3002–3024. [27] A. Murrone, H. Guillard, A five equation reduced model for compressible two phase flow problems, Journal of Computational Physics 202 (2005) 664–698. [28] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the numerical simulation of interfaces in two-phase flows, Comptes Rendus de l’Acad´emie des Sciences - Series I - Mathematics 331 (2000) 1017–1022. [29] J. Massoni, R. Saurel, B. Nkonga, R. Abgrall, Proposition de m´ethodes et mod`eles eul´eriens pour les probl`emes `a interfaces entre fluides compressibles en pr´esence de transfert de chaleur, International Journal of Heat and Mass Transfer 45 (2002) 1287–1307. [30] D. P. Garrick, M. Owkes, J. D. Regele, A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension, Journal of Computational Physics 339 (2017) 46–67. [31] M. L. Wong, S. K. Lele, High-order localized dissipation weighted compact nonlinear scheme for shockand interfacecapturing in compressible flows, Journal of Computational Physics 339 (2017) 179–209. [32] E. F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical Introduction, Springer, third edition, 2009. [33] J. P. Cocchi, R. Saurel, J. C. Loraud, Treatment of interface problems with Godunov-type schemes, Shock Waves 5 (1996) 347–357. [34] E. F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the HLL-Riemann solver, Shock Waves 4 (1994) 25–34. [35] K.-M. Shyue, A volume-fraction based algorithm for hybrid barotropic and non-barotropic two-fluid flow problems, Shock Waves 15 (2006) 407–423. [36] S. Tokareva, E. Toro, HLLC-type Riemann solver for the Baer–Nunziato equations of compressible two-phase flow, Journal of Computational Physics 229 (2010) 3573–3604. [37] B. Tian, E. Toro, C. Castro, A path-conservative method for a five-equation model of two-phase flow with an HLLC-type Riemann solver, Computers & Fluids 46 (2011) 122–132. [38] C. Rohde, C. Zeiler, A relaxation Riemann solver for compressible two-phase flow with phase transition and surface tension, Applied Numerical Mathematics 95 (2015) 267–279. [39] S. Fechter, C.-D. Munz, C. Rohde, C. Zeiler, Approximate Riemann solver for compressible liquid vapor flow with phase transition and surface tension, Computers & Fluids (2017). [40] R. Fedkiw, T. Aslam, B. Merriman, S. Osher, A Non-oscillatory Eulerian Approach to Interfaces in Multimaterial Flows (the Ghost Fluid Method), Journal of Computational Physics 152 (1999) 457–492. [41] H. Terashima, G. Tryggvason, A front-tracking/ghost-fluid method for fluid interfaces in compressible flows, Journal of Computational Physics 228 (2009) 4012–4037. [42] W. Bo, J. W. Grove, A volume of fluid method based ghost fluid method for compressible multi-fluid flows, Computers & Fluids 90 (2014) 113–122. [43] R. P. Fedkiw, T. Aslam, S. Xu, The Ghost Fluid Method for Deflagration and Detonation Discontinuities, Journal of Computational Physics 154 (1999) 393–427. [44] R. W. Houim, K. K. Kuo, A ghost fluid method for compressible reacting flows with phase change, Journal of Computational Physics 235 (2013) 865–900. [45] S. Fechter, C.-D. Munz, A discontinuous Galerkin-based sharp-interface method to simulate three-dimensional compressible two-phase flow, International Journal for Numerical Methods in Fluids 78 (2015) 413–435. [46] T. Liu, B. Khoo, K. Yeo, Ghost fluid method for strong shock impacting on material interface, Journal of Computational Physics 190 (2003) 651–681. [47] C. W. Wang, T. G. Liu, B. C. Khoo, A Real Ghost Fluid Method for the Simulation of Multimedium Compressible Flow, SIAM Journal on Scientific Computing 28 (2006) 278–302. [48] H. Terashima, G. Tryggvason, A front-tracking method with projected interface conditions for compressible multi-fluid flows, Computers & Fluids 39 (2010) 1804–1814.

43

[49] C. Liu, C. Hu, Adaptive THINC-GFM for compressible multi-medium flows, Journal of Computational Physics 342 (2017) 43–65. [50] S. M. H. Karimian, G. E. Schneider, Pressure-based computational method for compressible and incompressible flows, Journal of Thermophysics and Heat Transfer 8 (1994) 267–274. [51] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. [52] F. Cordier, P. Degond, A. Kumbaro, An Asymptotic-Preserving all-speed scheme for the Euler and Navier–Stokes equations, Journal of Computational Physics 231 (2012) 5685–5704. [53] A. J. Chorin, A Numerical Method for Solving Incompressible Viscous Flow Problems, Journal of Computational Physics 135 (1997) 118–125. [54] F. Xiao, Unified formulation for compressible and incompressible flows by using multi-integrated moments I: Onedimensional inviscid compressible flow, Journal of Computational Physics 195 (2004) 629–654. [55] S. Y. Kadioglu, M. Sussman, S. Osher, J. P. Wright, M. Kang, A second order primitive preconditioner for solving all speed multi-phase flows, Journal of Computational Physics 209 (2005) 477–503. [56] F. Xiao, R. Akoh, S. Ii, Unified formulation for compressible and incompressible flows by using multi-integrated moments II: Multi-dimensional version for compressible and incompressible flows, Journal of Computational Physics 213 (2006) 31–56. [57] N. Kwatra, J. Su, J. T. Gr´etarsson, R. Fedkiw, A method for avoiding the acoustic time step restriction in compressible flow, Journal of Computational Physics 228 (2009) 4146–4161. [58] S. Venkateswaran, J. W. Lindau, R. F. Kunz, C. L. Merkle, Computation of Multiphase Mixture Flows with Compressibility Effects, Journal of Computational Physics 180 (2002) 54–77. [59] R. Saurel, O. Le M´etayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiff mechanical relaxation, Shock Waves 16 (2007) 209–232. [60] S. LeMartelot, B. Nkonga, R. Saurel, Liquid and liquid–gas flows at all speeds, Journal of Computational Physics 255 (2013) 53–82. [61] J. Van Doormaal, G. Raithby, B. McDonald, The Segregated Approach to Predicting Viscous Compressible Fluid Flows, ASME Journal of Turbomachinery 109 (1987) 268–277. [62] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. [63] H. Bijl, P. Wesseling, A Unified Method for Computing Incompressible and Compressible Flows in Boundary-Fitted Coordinates, Journal of Computational Physics 141 (1998) 153–173. [64] F. H. Harlow, A. A. Amsden, Numerical calculation of almost incompressible flow, Journal of Computational Physics 3 (1968) 80–93. [65] F. H. Harlow, A. A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, Journal of Computational Physics 8 (1971) 197–213. [66] R. Issa, A. Gosman, A. Watkins, The computation of compressible and incompressible recirculating flows by a non-iterative implicit scheme, Journal of Computational Physics 62 (1986) 66–82. [67] K. C. Karki, S. V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA Journal 27 (1989) 1167–1174. [68] K.-H. Chen, R. Pletcher, Primitive Variable, Strongly Implicit Calculation Procedure for Viscous Flows at All Speeds, AIAA Journal 29 (1991) 1241–1249. [69] I. Demirˇdzi´c, v. Lilek, M. Peri´c, A collocated finite volume method for predicting flows at all speeds, International Journal for Numerical Methods in Fluids 16 (1993) 1029–1050. [70] S. M. H. Karimian, G. E. Schneider, Pressure-based control-volume finite element method for flow at all speeds, AIAA Journal 33 (1995) 1611–1618. [71] R. I. Issa, M. H. Javareshkian, Pressure-Based Compressible Calculation Method Utilizing Total Variation Diminishing Schemes, AIAA Journal 36 (1998) 1652–1657. [72] F. Moukalled, M. Darwish, A unified formulation of the segregated class of algorithms for fluid flow at all speeds, Numerical heat transfer, Part B. 37 (2000) 103–139. [73] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165. [74] M. Darwish, F. Moukalled, A fully coupled navier-stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444. [75] A. Miettinen, T. Siikonen, Application of pressureand density-based methods for different flow speeds: Application of pressureand density-based methods for different flow speeds, International Journal for Numerical Methods in Fluids 79 (2015) 243–267. [76] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130. [77] F. Denner, B. van Wachem, Compressive VOF method with skewness correction to capture sharp interfaces on arbitrary meshes, Journal of Computational Physics 279 (2014) 127–144. [78] F. Harlow, A. Amsden, Fluid Dynamics, Monograph LA-4700, Los Alamos National Laboratory, 1971. [79] R. Saurel, R. Abgrall, A Simple Method for Compressible Multifluid Flows, SIAM Journal on Scientific Computing 21 (1999) 1115–1145. [80] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–225. [81] V. Levich, Physicochemical Hydrodynamics, Prentice Hall, 1962. [82] J. Ferziger, M. Peri´c, Computational Methods for Fluid Dynamics, Springer Verlag, Berlin Heidelberg New York, 3. edition, 2002. [83] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, Journal of Computational Physics 298 (2015) 466–479. [84] P. Roe, Characteristic-based schemes for the euler equations, Annual Review of Fluid Mechanics 18 (1986) 337–365. [85] F. Denner, Balanced-Force Two-Phase Flow Modelling on Unstructured and Adaptive Meshes, Ph.D. thesis, Imperial College London, 2013. [86] I. Demirˇdzi´c, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255.

44

[87] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. [88] P. Zwart, The Integrated Space-Time Finite Volume Method, Ph.D. thesis, University of Waterloo, 1999. [89] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annual Research Briefs, Center for Turbulence (2004) 3–14. [90] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. [91] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, Journal of Computational Physics 285 (2015) 24–40. [92] R. Kunz, W. Cope, S. Venkateswaran, Development of an implicit method for multi-fluid flow simulations, Journal of Computational Physics 152 (1999) 78–101. [93] M. Darbandi, V. Mokarizadeh, A modified pressure-based algorithm to solve flow fields with shock and expansion waves, Numerical Heat Transfer, Part B: Fundamentals 46 (2004) 497–504. [94] S. Osher, S. Chakravarthy, High resolution schemes and the entropy condition, SIAM Journal on Numerical Analysis 21 (1984) 955–984. [95] D. P. Garrick, W. A. Hagen, J. D. Regele, An interface capturing scheme for modeling atomization in compressible flows, Journal of Computational Physics 344 (2017) 260–280. [96] O. Ubbink, R. Issa, A Method for Capturing Sharp Fluid Interfaces on Arbitrary Meshes, Journal of Computational Physics 153 (1999) 26–50. [97] V. Gopala, B. van Wachem, Volume of fluid methods for immiscible-fluid and free-surface flows, Chemical Engineering Journal 141 (2008) 204–221. [98] K.-M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for compressible two-phase flow: The algebraic THINC approach, Journal of Computational Physics 268 (2014) 326–354. [99] F. Denner, A. Charogiannis, M. Pradas, C. N. Markides, B. van Wachem, S. Kalliadasis, Solitary waves on falling liquid films in the inertia-dominated regime, Journal of Fluid Mechanics 837 (2018) 491–519. [100] F. Denner, Frequency dispersion of small-amplitude capillary waves in viscous fluids, Physical Review E 94 (2016) 023110. [101] M. Raessi, J. Mostaghimi, M. Bussmann, Advecting normal vectors: A new method for calculating interface normals and curvatures when modeling two-phase flows, Journal of Computational Physics 226 (2007) 774–797. [102] J. D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw-Hill New York, 2003. [103] S. Popinet, Numerical models of surface tension, Annual Review of Fluid Mechanics 50 (2018) 49–75. [104] J. Banks, T. Aslam, W. Rider, On sub-linear convergence for linearly degenerate waves in capturing schemes, Journal of Computational Physics 227 (2008) 6985–7002. [105] S. Balay, W. Gropp, L. C. McInnes, B. F. Smith, Efficient Management of Parallelism in Object Oriented Numerical Software Libraries, in: E. Arge, A. Bruasat, H. Langtangen (Eds.), Modern Software Tools in Scientific Computing, Birkhaeuser Press, 1997, pp. 163–202. [106] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. [107] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. [108] R. Dembo, S. Eisenstat, T. Steihaug, Inexact newton methods, SIAM Journal on Numerical Analysis 19 (1982) 400–408. [109] R. K. Shukla, C. Pantano, J. B. Freund, An interface capturing method for the simulation of multi-phase compressible flows, Journal of Computational Physics 229 (2010) 7411–7439. [110] K. So, X. Hu, N. Adams, Anti-diffusion interface sharpening technique for two-phase compressible flow simulations, Journal of Computational Physics 231 (2012) 4304–4323. [111] A. Chiapolino, R. Saurel, B. Nkonga, Sharpening diffuse interfaces with compressible fluids on unstructured meshes, Journal of Computational Physics 340 (2017). [112] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure-velocity coupling allowing acoustic calculation in low Mach number flow, Journal of Computational Physics 231 (2012) 5522–5541. [113] P. Lax, B. Wendroff, Systems of conservation laws, Communications on Pure and Applied Mathematics 13 (1960) 217–237. [114] P. Woodward, P. Colella, The Numerical Simulation of Two-Dimensional Fluid Flow with Strong Shocks, Journal of Computational Physics 173 (1984) 115–173. [115] J.-F. Haas, B. Sturtevant, Interaction of weak shock waves with cylindrical and spherical gas inhomogeneities, Journal of Fluid Mechanics 181 (1987) 41. [116] J. J. Quirk, S. Karni, On the dynamics of a shock–bubble interaction, Journal of Fluid Mechanics 318 (1996) 129. [117] J. H. J. Niederhaus, J. A. Greenough, J. G. Oakley, D. Ranjan, M. H. Anderson, R. Bonazza, A computational parameter study for the three-dimensional shock–bubble interaction, Journal of Fluid Mechanics 594 (2008). [118] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, Journal of Computational Physics 219 (2006) 715–732. [119] R. Nourgaliev, T. Dinh, T. Theofanous, Adaptive characteristics-based matching for compressible multifluid dynamics, Journal of Computational Physics 213 (2006) 500–529. [120] D. Igra, Takayama, Investigation of aerodynamic breakup of a cylindrical water droplet, Atomization and Sprays 11 (2001) 20. [121] J. C. Meng, T. Colonius, Numerical simulations of the early stages of high-speed droplet breakup, Shock Waves 25 (2015) 399–414. [122] G. Xiang, B. Wang, Numerical study of a planar shock interacting with a cylindrical water column embedded with an air cavity, Journal of Fluid Mechanics 825 (2017) 825–852. [123] J. Meng, Numerical Simulations of Droplet Aerobreakup, Ph.D. thesis, California Institute of Technology, Pasadena, California, USA, 2016. [124] D.-Y. Peng, D. B. Robinson, A New Two-Constant Equation of State, Industrial & Engineering Chemistry Fundamentals 15 (1976) 59–64.

45

