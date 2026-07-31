
## Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq001.png)

Chair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany

Abstract

A conservative finite-volume framework, based on a collocated variable arrangement, for the simulation of flows at all speeds, applicable to incompressible, ideal-gas and real-gas fluids is proposed in conjunction with a fully-coupled pressure-based algorithm. The applied conservative discretisation and implementation of the governing conservation laws as well as the definition of the fluxes using a momentum-weighted interpolation are identical for incompressible and compressible fluids, and are suitable for complex geometries represented by unstructured meshes. Incompressible fluids are described by predefined constant fluid properties, while the properties of compressible fluids are described by the Noble-Abel-stiffened-gas model, with the definitions of density and specific static enthalpy of both incompressible and compressible fluids combined in a unified thermodynamic closure model. The discretised governing conservation laws are solved in a single linear system of equations for pressure, velocity and temperature. Together, the conservative finite-volume discretisation, the unified thermodynamic closure model and the pressure-based algorithm yield a conceptually simple, but versatile, numerical framework. The proposed numerical framework is validated thoroughly using a broad variety of test-cases, with Mach numbers ranging from 0 to 239, including viscous flows of incompressible fluids as well as the propagation of acoustic waves and transiently evolving supersonic flows with shock waves in ideal-gas and real-gas fluids. These results demonstrate the accuracy, robustness and the convergence, as well as the conservation of mass and energy, of the numerical framework for flows of incompressible and compressible fluids at all speeds, on structured and unstructured meshes. In particular, the precise recovery of a divergence-free velocity field in the incompressible limit, the accurate prediction of acoustic waves, and the convergence to the correct weak solution for strong shock waves with the same finite-volume discretisation and pressure-based algorithm are important features of the proposed numerical framework.

Keywords: Finite-volume methods, Pressure-based algorithms, Flows at all speeds, Compressible fluids, Incompressible fluids, Unstructured meshes

c⃝2020. This manuscript version is made available under the CC-BY-NC-ND 4.0 license. http://creativecommons.org/licenses/by-nc-nd/4.0/

1. Introduction

Since the seminal work of Harlow and Amsden [1, 2], the formulation of numerical algorithms that can be applied for fluid flows at any speed is a central quest in computational fluid dynamics (CFD). Yet, despite extensive research efforts over the past 50 years, the development of numerical methods and algorithms that are able to provide an accurate and robust prediction of the behaviour of fluids with different compressibility and of fluid flows at all speeds has proven difficult. Although the flow of any fluid and at any speed is described by the governing equations describing the conservation of mass, momentum and energy, different modelling assumptions with respect to the compressibility of the fluid and the different physical mechanisms dominating at different flow speeds yield dissimilar mathematical characteristics of the governing equations. This in turn leads to distinct and often contrasting numerical requirements. When developing numerical methods for flows at all speeds, it is important to recognise the numerical implications associated with the flow speed U, represented by the Mach number M = U/a, where a = � 1/(ρβs) is the speed of sound, and with the isentropic compressibility of the fluid, βs = {dρ/(ρ dp)}s, that relates changes in pressure p and density ρ of a fluid at constant entropy. While pressure and density are strongly coupled for large flow speeds (M > 0.1), in particular for supersonic flows (M > 1), the pressure-density coupling

∗Corresponding author: Email address: fabian.denner@ovgu.de (Fabian Denner)

1


# arXiv:2002.10482v2  [physics.comp-ph]  29 Feb 2020

diminishes at low Mach numbers and vanishes for M →0, where dρ →0. Founded on the observation that density changes are small at small speeds, a common assumption when modelling fluid flows is that the fluid is incompressible, with a constant density (dρ = 0) along the fluid particle trajectories and, consequently, βs = 0. Hence, pressure waves propagate with infinite speed (a →∞) in incompressible fluids, contrary to compressible fluids where βs > 0 and 0 < a < ∞. In fact, the convergence of solutions of the governing equations of the flow of compressible fluids to the governing equations of the flow of incompressible fluids for M →0 has been proven rigorously by Klainerman and Majda [3] and Hoff [4]. In addition to the governing conservation laws, compressible fluids require a thermodynamic closure model that describes the relationship between density, pressure and energy. The ideal-gas model represents the most simple and most widely used thermodynamic closure model, with p ∝(ρ, T), where T is the temperature. More complex formulations, so-called real-gas models, further include the effects of intermolecular repulsion [5], intermolecular attraction [6, 7] or both [8, 9], or other material properties, e.g. the acentric factor describing the shape of the molecules [10, 11]. For an incompressible fluid, however, no closure model is required, since the density is not coupled to pressure, and an isothermal flow of an incompressible fluid is fully described only by the momentum and continuity equations, i.e. the energy equation becomes redundant. The challenge in developing numerical frameworks that are applicable to incompressible fluids and compressible fluids at all flow speeds is, therefore, to construct a numerical method that combines a unified thermodynamic closure model, a uniform set of interpolation functions, a consistent handling of the incompressible limit, shock capturing capabilities, a method to advect the solution that is applicable in all speed regimes, as well as a set of solution variables that are physically meaningful for incompressible and compressible fluids [12]. The choice of solution variables is of particular importance in constructing a numerical method that is applicable to flows at all speeds, since a unified algorithm is predicated on a single set of solution variables [12]. Choosing the conserved variables, i.e. density, momentum and total energy, as solution variables for the continuity, momentum and energy equations, respectively, is desirable for compressible fluids at sufficiently large speeds (M > 0.1). However, the continuity equation is not effective as a transport equation for density in the incompressible limit, because dρ →0, and, instead, becomes a constraint on the velocity field with ∇· u →0 [13]. An attractive choice of the solution variables for numerical algorithms applicable to predict flows at all speeds is, therefore, the primitive variables including pressure [6, 12, 14, 15], i.e. pressure, velocity and temperature. Using pressure as a solution variable, the vanishing density differences in the incompressible limit do not pose a problem and the pressure acts as a Lagrange multiplier that enforces ∇· u →0 [16–18]. Conveniently, choosing primitive variables as solution variables still allows to discretise the governing equations in conservative form [19]. In practice, however, achieving accurate conservation of mass and energy, constructing robust shock capturing schemes and ensuring a stable numerical solution in the transonic regime has proven difficult in the context of primitive variables [20, 21]. It is, therefore, convenient to develop numerical algorithms either for incompressible fluids or for compressible fluids, which has led to two primary classes of algorithms: pressure-based algorithms and density-based algorithms. Pressure-based algorithms, in which the continuity equation serves as an equation for pressure, while density is constant (incompressible fluid) or evaluated explicitly using an equation of state (compressible fluid), may be used to predict flows at all speeds, see e.g. [14, 15, 22–33]. For both incompressible and compressible fluids, the majority of pressure-based algorithms are founded on pressure-correction methods, such as projection methods [34, 35], the SIMPLE method [36, 37] and its subsequent derivatives, or the PISO method [38, 39]. However, the weak coupling between density, pressure, velocity and energy of the discretised governing equations as a result of the iterative predictor-corrector solution procedure, which usually necessitates underrelaxation of the discretised equations to reach a converged solution, is a key shortcoming of segregated methods [31, 40]. This has motivated the development of coupled algorithms, where the discretised governing equations are solved in a single linear system of equations, for both incompressible fluids [30, 32, 41, 42] and compressible fluids [14, 15, 24, 31, 33, 43], showing great potential in terms of versatility, robustness and performance in all speed regimes. For instance, Darwish et al. [30] demonstrated substantial performance benefits for incompressible flows compared to pressure-correction methods and Denner et al. [44] reported robust results for flows with Mach numbers ranging from 0.001 to 100 with a fully-coupled pressure-based algorithm. Contemporary numerical methods for the simulation of compressible flows are typically predicated on densitybased algorithms, e.g. [45–47], where the conserved quantities are chosen as solution variables for the governing conservation equations and, in particular, the continuity equation serves as an equation for density. While density-based algorithms are naturally suited for compressible flows, they are poorly suited for low-Mach number flows [21, 48], where the coupling of pressure and density vanishes. Although density-based algorithms have been applied to low-Mach number flows with some success, this requires pre-conditioning techniques [47, 49–52] that are computationally very expensive, especially for transient problems, and the success of which is typically determined, at least in parts, by predefined constants [49, 53]. In order to improve the performance for low Mach number flows, recent work has been focusing on combining density-based methods with segregated pressurecorrection algorithms [54–58] and/or reformulating the energy equation as an equation for pressure [55, 59–62]. These density-based algorithms have been applied successfully to a wide range of flows, including flows ranging

2

from an incompressible flow to the propagation of strong shock waves, stationary and high-speed discontinuous waves as well as the propagation of linear acoustic waves [54, 56, 59, 63]. An important aspect for the design of numerical frameworks for fluid flows at all speeds is that pressure plays an important role in all Mach number regimes [22, 64]; pressure changes are, contrary to density changes, always finite. Exploiting this versatile role of pressure by including pressure as a primary solution variable in the numerical framework, thus, provides a seemingly distinct advantage for applications in all Mach number regimes: it provides a solution variable, i.e. pressure, which is meaningful in all Mach number regimes and does not require particular pre-conditioning techniques. This is further supported by the analysis of Hauke and Hughes [12], who identified the primitive variables (pressure, velocity and temperature) as particularly suitable solution variables to predict flows at all speeds. Remarkably, all of the numerical methods that stand out with respect to modelling fluid flows at all speeds, due to their versatility and robustness, incorporate the unique role of pressure, albeit in different ways. In pressure-based algorithms, the special role of pressure can be taken into account through an appropriate linearisation of the discretised continuity equation [19, 22, 33, 48]: for compressible flows, the continuity equations serves as a transport equation for density, with density formulated as a function of pressure by an equation of state, whereas for incompressible flows, the continuity equation serves as a constraint on the divergence of the velocity field [43], with pressure acting as a Lagrange multiplier. The extension of density-based algorithms to low Mach numbers, either by introducing a pressure-Poisson equation [54, 56] or by reformulating the energy equation as an equation for pressure [55, 59, 60], provides a pressure-velocity coupling at low speeds and enforces a divergence-free velocity field in the incompressible limit. However, despite the broad variety of numerical methods able to simulate flows at all speeds, a numerical framework based on a unified conservative discretisation that is able to incorporate incompressible fluids as well as ideal-gas and real-gas compressible fluids, and which can predict flows at all speeds accurately and robustly, including low-Mach acoustics, Riemann problems and multidimensional flows ranging from the incompressible limit to supersonic flow, has not been presented in the literature yet. In this article, a conservative, collocated, finite-volume framework in combination with a fully-coupled pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all Mach numbers is proposed. The governing equations describing the conservation of continuity, momentum and energy are discretised using standard finite-volume methods and are solved for pressure, velocity and temperature in a single linear system of equations. Incompressible fluids are described by predefined constant fluid properties, while compressible fluids are described by the Noble-Abel-stiffened-gas model [9], with the definitions of density and specific static enthalpy of both incompressible and compressible fluids combined in a unified thermodynamic closure model. This enables the design of a conceptually simple, but versatile, numerical algorithm that is able to predict flows of incompressible fluids as well as flows of compressible fluids at all speeds. The conservative discretisation and implementation of the governing equations are identical for incompressible and compressible fluids, employing a single definition of the fluxes based on a momentum-weighted interpolation [65]. A broad variety of representative test-cases featuring flows of incompressible and compressible fluids in all Mach number regimes are considered to scrutinise and validate the proposed numerical framework: the propagation of acoustic waves, contact discontinuities and shock waves, shock tubes in different Mach number regimes, Taylor vortices in an inviscid fluid, diffusion-dominated problems, a lid-driven cavity, supersonic flow over a forward-facing step, and Stokes flow around a rotating sphere. The presented results demonstrate the accuracy and robustness, as well as the conservation and convergence properties, of the numerical framework for all flow speeds on structured and unstructured meshes. In particular, the precise recovery of a divergence-free velocity field for M →0, the accurate prediction of acoustic waves and the convergence to the correct weak solution for M ≫1 are important features of the proposed numerical framework. As such, the proposed numerical framework stands out for the simplicity of its discretisation in conjunction with the broad range of flows that can be predicted accurately and robustly. The governing equations are introduced in Section 2. Subsequently, the three main building blocks of the proposed finite-volume method are presented: a unified thermodynamic closure model in Section 3, the finitevolume discretisation in Section 4, and the pressure-based algorithm used to solve the discretised governing equations in Section 5. The results of representative test-cases are presented and discussed in Section 6. The article is summarised and concluded in Section 7.

2. Governing equations

The conservation laws governing fluid flows at all speeds, applicable to both incompressible and compressible flows, formulated in a Cartesian coordinate system, are the conservation of mass


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq002.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq003.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq004.png)

the conservation of momentum ∂ρuj


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq005.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq006.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq007.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq008.png)

3

and the conservation of energy


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq009.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq010.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq011.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq012.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq013.png)

where t is time, u is the velocity vector, p is pressure, ρ is the density of the fluid and h = hs + u2/2 is the specific total enthalpy, with hs the specific static enthalpy. The stress tensor τ for the considered Newtonian fluids is given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq014.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq015.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq016.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq017.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq018.png)

where µ is the dynamic viscosity of the fluid. Heat conduction is modelled by Fourier’s law,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq019.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq020.png)

where k is the thermal conductivity of the fluid and T is the temperature. The enthalpy formulation is chosen for the energy equation, rather than the more common internal energy formulation, because it leads to a straightforward application in the numerical algorithm, since the transient pressure term on the right-hand side of Eq. (3) does not require linearisation [33, 58]. The governing conservation laws require closure through an appropriate model that defines the thermodynamic properties (see Section 3).

3. Thermodynamic closures

In order to close the governing conservation laws presented in Section 2, the thermodynamic properties of the fluid have to be defined. In the proposed formulation, this is achieved by defining the density ρ and the specific static enthalpy hs = cp T + e∗, (6)

where cp is the specific isobaric heat capacity and e∗is the specific residual energy, through a set of input quantities (ρ0, cv, cp, Π, b). This approach enables the formulation of a unified thermodynamic closure for incompressible, ideal-gas and real-gas fluids, which facilitates a straightforward finite-volume discretisation that is applicable for incompressible flows as well as compressible flows in all Mach number regimes. An incompressible fluid is characterised by a constant density, with dρ = 0, defined as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq021.png)

The specific isobaric heat capacity cp is assumed to be constant for incompressible fluids and the specific residual energy is e∗= 0. The speed of sound for an incompressible fluid is given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq022.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq023.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq024.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq025.png)

with subscript s denoting constant entropy. The Noble-Abel-stiffened-gas (NASG) model, originally proposed by Le M´etayer and Saurel [9], is chosen to represent ideal and real gases. The NASG model is a combination of the stiffened-gas model [6, 66] and the Noble-Abel-gas model (also called co-volume gas model) [5], with the motivation of defining a simple gas model that accounts for molecular attraction and repulsion. The thermal and caloric equations of state of the NASG model are given as [9]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq026.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq027.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq028.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq029.png)

respectively, where γ = cp/cv is the heat capacity ratio, cv is the specific isochoric heat capacity, v = 1/ρ is the specific volume, e is the specific internal energy and e0 is the specific reference energy. The pressure constant Π represents attraction between molecules and is typically relevant for condensed phases, e.g. to model liquids, while the co-volume b accounts for the volume occupied by the individual molecules of the fluid. The density is given by rearranging Eq. (9) as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq030.png)

and the specific total enthalpy, h = hs + u2/2, follows from Eqs. (6), (9) and (10) as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq031.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq032.png)

4


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq033.png)

In the following, the specific reference energy is assumed to be e0 = 0, because only single-phase flows without phase transition and reactions are considered. The specific heat capacities cv and cp are constant and the speed of sound is [9]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq034.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq035.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq036.png)

Contrary to the van der Waals gas model, which also accounts for molecular attraction and repulsion, the coefficients Π and b are constant; the NASG model thus represents these molecular interactions in the simplest possible form. Furthermore, the NASG model is, unlike, for instance, the van der Waals gas model, unconditionally convex. With respect to liquids, such as water, the NASG model resolves the inaccuracy of specific heat capacities resulting from applying the classical stiffened-gas model [9]. The NASG model reduces to the ideal-gas (IG) model for Π = 0 and b = 0, to the Noble-Abel (NA) gas model for Π = 0 and b > 0, and to the stiffened-gas (SG) model for Π > 0 and b = 0. In order to incorporate incompressible and compressible fluids in the same numerical framework, the definitions for the density ρ and the specific residual energy e∗are unified by the binary operators C and I = 1 −C. The binary operator C, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq037.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq038.png)

is used as a coefficient for the compressible part and, analogously, the binary operator I is used as a coefficient for the incompressible part of the unified closure model. The density of the fluid is then defined, based on Eqs. (7) and (11), as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq039.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq040.png)

and the specific residual energy is given based on Eq. (13), and assuming e0 = 0, as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq041.png)

The type of fluid considered in a simulation can be simply specified through the binary operator C, without changes to the thermodynamic closure model or the discretisation of the governing equations. An incompressible fluid (C = 0) is, thereby, fully defined by setting ρ0 and cp, while a compressible fluid (C = 1) is defined by setting cv, cp, Π and b.

4. Finite-volume discretisation

The proposed numerical framework is founded on a collocated finite-volume discretisation, which is based on the integral formulation of the governing conservation laws, for unstructured meshes. Taking the convectiondiffusion equation for the transport of a general flow variable, φ, as an example, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq042.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq043.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq044.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq045.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq046.png)

where Γφ is the diffusion coefficient of φ, its integral form with respect to control volume V is given as ˚


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq047.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq048.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq049.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq050.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq051.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq052.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq053.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq054.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq055.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq056.png)

The discretisation of each individual term is discussed in the following.

4.1. Gradient evaluation The spatial gradient at cell centre P is evaluated using the divergence theorem, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq057.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq058.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq059.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq060.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq061.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq062.png)

where f denotes the faces bounding cell P, VP is the volume of cell P, nf is the normal vector of face f pointing outwards with respect to cell P, and Af is the area of face f. The face value φf is interpolated from the adjacent cell centres P and Q, schematically illustrated in Fig. 1a, as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq063.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq064.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq065.png)

5


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq066.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq067.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq068.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq069.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq070.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq071.png)

(a) General discretisation


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq072.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq073.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq074.png)

(b) TVD differencing


> **Figure 1: Schematic illustration of (a) cell P with its neighbour cell Q and the shared face f, where nf is the unit normal vector of face f and sf is the unit vector connecting cell centres P and Q (both outward pointing with respect to cell P), with f′ the interpolation point associated with face f and rf the vector from interpolation point f′ to face centre f, and (b) upwind cell U and downwind cell D of face f, where u represents the velocity vector.**

where lP f is the inverse-distance weighting coefficient,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq075.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq076.png)

with ∆sf the distance between cell centres P and Q, and rP f is the vector connecting cell centre P with face interpolation point f ′. A formally second-order accurate gradient-based correction of mesh-skewness [25, 67] is included in Eq. (21), with rf the vector connecting the interpolation point f ′ of the face with face centre f on meshes with skewness, see Fig. 1a.

4.2. Transient terms The First-Order Backward Euler scheme, also widely known as BDF1 scheme, and the Second-Order Backward Euler scheme, also widely known as BDF2 scheme, are used to discretise the transient terms of the governing flow equations. The transient term of the transport equation (19), with Φ = ρφ, is given for cell P discretised with the First-Order Backward Euler scheme as ˚


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq077.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq078.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq079.png)

and discretised with the Second-Order Backward Euler scheme as [68] ˚

V


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq080.png)

∂t dV ≈ ��1 ∆t1 + 1


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq081.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq082.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq083.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq084.png)

with ∆τ = ∆t1 + ∆t2, where ∆t1 is the current time-step, ∆t2 is the previous time-step, superscript (t −∆t1) denotes values of the previous time-level and superscript (t −∆τ) denotes values of the previous-previous time-level. If the time-step is constant, with ∆t1 = ∆t2, the transient term of Eq. (19) discretised with the Second-Order Backward Euler scheme simplifies to the more familiar form

˚


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq085.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq086.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq087.png)

For consistency, all transient terms of the governing equations (1)-(3) are discretised with the same scheme [33].

4.3. Advection terms Applying the divergence theorem, the advection term of Eq. (19) is given as ˚


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq088.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq089.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq090.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq091.png)

where S is the outward-pointing surface vector on the surface ∂V of control volume V . Assuming the surface of the control volume has a finite number of flat faces f with area Af, and applying the midpoint rule [16, 64], the advection term follows in semi-discretised form as ‹


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq092.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq093.png)

6

where ϑf = uf · nf is the advecting velocity at face f, which will be discussed in detail in Section 5.1. The advected variable ˜φf and the density ˜ρf are interpolated using a TVD interpolation for three-dimensional unstructured meshes with an implicit correction of mesh skewness [69], given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq094.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq095.png)

where subscripts U and D denote the upwind and downwind cells, as illustrated in Fig. 1b, ξf is the flux limiter and rUf is the vector connecting the cell centre of the upwind cell U with face interpolation point f ′. A detailed description of the implementation of this TVD interpolation using common TVD schemes on skewed and non-equidistant meshes can be found in [69]. In this study, the first-order upwind scheme, ξf = 0, the central differencing scheme, ξf = 1, and the Minmod scheme [70], ξf(gf) = max(0, min(1, gf)), where gf is the ratio of the upwind and downwind gradients of φ [69], are considered.

4.4. Diffusion terms Applying the divergence theorem and the midpoint rule, the diffusion term of the transport equation (19) is given as ˚


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq096.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq097.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq098.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq099.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq100.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq101.png)

Following Ferziger [71], the diffusion coefficient Γφ at face f is defined as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq102.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq103.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq104.png)

Considering an orthogonal mesh, where the unit normal vector nf of face f and the unit vector sf connecting the adjacent cell centres P and Q are parallel, with nf = sf, the face-centred gradient is approximated with second-order accuracy as ∂φ ∂xi


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq105.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq106.png)

The decomposition and deferred correction approach of Demirdˇzi´c [72] is applied to correct for non-orthogonality of the mesh, as illustrated in Fig. 1a, with the face-centred gradient defined as [25]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq107.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq108.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq109.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq110.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq111.png)

The scaling factor αf = (nf · sf)−1 ensures a robust convergence even for large non-orthogonality of the mesh [73, 74]. Equation (32) reduces to Eq. (31) for an orthogonal mesh with nf = sf.

5. Pressure-based algorithm

A finite-volume framework with a pressure-based algorithm for the prediction of flows of incompressible fluids and compressible fluids at all speeds is proposed. To this end, the governing equations (1)-(3) are closed by the thermodynamic closure model and discretised using the finite-volume discretisation presented in Sections 3 and 4, respectively. Once discretised and linearised as detailed below, the governing equations are solved simultaneously in a single linear system of equations, Aψ = σ, for the pressure p, the velocity vector u ≡(u, v, w)T and the temperature T. For a three-dimensional computational mesh with N cells, the linear system of governing equations is given as 


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq112.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq113.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq114.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq115.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq116.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq117.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq118.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq119.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq120.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq121.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq122.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq123.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq124.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq125.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq126.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq127.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq128.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq129.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq130.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq131.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq132.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq133.png)

where Aζ,χ, with ζ the conserved quantity and χ the primary solution variable of a given governing equation, are the coefficient submatrices of size N ×N of the continuity equation (44) for ζ = ρ, the momentum equations (46) for ζ ∈{ρu, ρv, ρw}, and the energy equation (47) for ζ = ρh. The subvectors ψχ of length N hold the solution for primary solution variable χ and the subvectors σζ of length N hold all contributions from previous nonlinear iterations and previous time-levels. The solution procedure performs nonlinear iterations in which the linear system of governing equations (33) is solved using the Block-Jacobi preconditioner and the BiCGSTAB solver of the software library PETSc [75–77] until the residual of (33) satisfies ∥Aψ −σ∥< η ∥σ∥, where η is the predefined solution tolerance and ∥· ∥ denotes the L2-norm, as presented and tested in detail by Denner [33].

7

5.1. Advecting velocity

In the proposed numerical framework, the advecting velocity ϑf = uf ·nf is based on a momentum-weighted interpolation (MWI), originally introduced by Rhie and Chow [78], and serves to advect the conserved quantities ζ = {ρ, ρu, ρh}. Furthermore, for flows of incompressible fluids and low Mach number flows of compressible fluids, the advecting velocity allows to solve the continuity equation for pressure (see Section 5.4) and prevents pressure-velocity decoupling associated with the collocated variable arrangement [16, 65]. Following the work of Bartholomew et al. [65], the advecting velocity ϑf at face f is given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq134.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq135.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq136.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq137.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq138.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq139.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq140.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq141.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq142.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq143.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq144.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq145.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq146.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq147.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq148.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq149.png)

where the interpolated face velocities uf and u(t−∆t1) f are obtained by linear interpolation, and lP f is given by Eq. (22). As derived and discussed in detail by Bartholomew et al. [65], the coefficient ˆdf is defined as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq150.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq151.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq152.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq153.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq154.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq155.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq156.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq157.png)

where SP = �3 j=1 Dρuj,uj P and SQ = �3 j=1 Dρuj,uj Q are the sum of the diagonal matrix coefficients of the velocity arising from the advection and shear stress terms of the discretised momentum equations, see Eq. (A.10) in Appendix A. The face density is defined as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq158.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq159.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq160.png)

The MWI provides a robust pressure-velocity coupling for incompressible flows by introducing a cell-tocell pressure coupling and applying a low-pass filter acting on the third derivative of pressure [16, 21, 25, 65], thus avoiding pressure-velocity decoupling due to the collocated variable arrangement. The transient term of Eq. (34) ensures a time-step independent contribution of the MWI in conjunction with the coefficient ˆdf [65] and is important for a correct temporal evolution of pressure waves [43, 65]. However, the MWI is known to introduce numerical dissipation that manifests in an unphysical dissipation of kinetic energy [65, 79], a conservation error that converges with ∆x3 and that is, assuming the consistent formulation given by Eq. (34), independent of the applied time-step [65].

5.2. Discretised governing equations

Applying the finite-volume methods described in Section 4 and, in particular, using the BDF1 scheme for the transient term in the interest of clarity, the discretised continuity equation (1) for cell P is given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq161.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq162.png)

Similar to the discretisation of the continuity equation, applying the finite-volume scheme presented in Section 4, the discretised momentum equations (2) in cell P are given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq163.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq164.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq165.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq166.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq167.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq168.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq169.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq170.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq171.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq172.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq173.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq174.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq175.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq176.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq177.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq178.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq179.png)

where the viscosity µf at face f is defined by Eq. (30). In order to account for mesh non-orthogonality, the deferred correction approach given in Eq. (32) is applied to decompose the shear-stress term as � ∂uj ∂xi


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq180.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq181.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq182.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq183.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq184.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq185.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq186.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq187.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq188.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq189.png)

8

The discretised energy equation (3) in cell P, using the applied finite-volume discretisation, is given as

ρP hP −ρ(t−∆t1) P h(t−∆t1) P ∆t1 + �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq190.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq191.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq192.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq193.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq194.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq195.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq196.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq197.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq198.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq199.png)

3


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq200.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq201.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq202.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq203.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq204.png)

where the heat conduction term is decomposed as described by Eq. (32) and the thermal conductivity kf at face f is defined by Eq. (30).

5.3. Linearisation and implementation The details of the linearisation of the governing equations have been shown to be a critical aspect for all-Mach formulations and algorithms [19, 33, 40, 48] and provides additional potential with respect to the performance of fully-coupled algorithms [33]. To this end, a Newton linearisation is applied to facilitate an implicit treatment of all dominant pressure, velocity and temperature terms in the linear system resulting from the linearisation and discretisation of the governing equations (1)-(3), given for two generic fluid variables as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq205.png)

or for three generic fluid variables as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq206.png)

where n is the iteration counter associated with the nonlinear iterations performed to solve the system of discretised governing equations, Eq. (33), at each time-step. Superscript (n) denotes the most recent available solution, which is the solution of the previous time-step during the first nonlinear iteration of a given time-step or, otherwise, the solution of the previous nonlinear iteration, and superscript (n + 1) denotes the solution that is sought implicitly. Applying the Newton linearisation given in Eq. (41) to the advection term and formulating the cell-centered density ρP of the transient term as a semi-implicit function of pressure pP , given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq207.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq208.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq209.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq210.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq211.png)

the discretised continuity equation (37) follows as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq212.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq213.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq214.png)

Following previous studies [33, 44], the advecting velocity ϑ(n+1) f is defined by a semi-implicit formulation as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq215.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq216.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq217.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq218.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq219.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq220.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq221.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq222.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq223.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq224.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq225.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq226.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq227.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq228.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq229.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq230.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq231.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq232.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq233.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq234.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq235.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq236.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq237.png)

Linearising the transient terms and the advection terms with the Newton linearisation given in Eqs. (41) and (42), respectively, following the work of Denner [33], and treating cell-centered pressure and velocity contributions implicitly, the discretised momentum equations (38) follow as

ρ(n) P u(n+1) j,P + ρ(n+1) P u(n) j,P −ρ(n) P u(n) j,P −ρ(t−∆t1) P u(t−∆t1) j,P ∆t1 VP

+ �

f

� ˜ρ(n) f ϑ(n) f ˜u(n+1) j,f + ˜ρ(n) f ϑ(n+1) f ˜u(n) j,f + ˜ρ(n+1) f ϑ(n) f ˜u(n) j,f −2˜ρ(n) f ϑ(n) f ˜u(n) j,f � Af = − �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq238.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq239.png)

+ �

f µf

�


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq240.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq241.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq242.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq243.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq244.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq245.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq246.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq247.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq248.png)

3


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq249.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq250.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq251.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq252.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq253.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq254.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq255.png)

9

and the discretised energy equation (40) becomes

ρ(n) P h(n+1) P + ρ(n+1) P h(n) P −ρ(n) P h(n) P −ρ(t−∆t1) P h(t−∆t1) P ∆t1 VP

+ �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq256.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq257.png)

= p(n+1) P −p(t−∆t1) P ∆t1 VP + �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq258.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq259.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq260.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq261.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq262.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq263.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq264.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq265.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq266.png)

+ �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq267.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq268.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq269.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq270.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq271.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq272.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq273.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq274.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq275.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq276.png)

3


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq277.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq278.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq279.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq280.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq281.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq282.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq283.png)

with ρ(n+1) P given by Eq. (43) and ϑ(n+1) f given by Eq. (45). The implicitly computed specific total enthalpy

h(n+1) P at cell-centre P is formulated, following Eq. (12) and assuming e0 = 0, as an implicit function of temperature T and pressure p, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq284.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq285.png)

This treatment enables the implicit solution of the energy equation for temperature, pressure and velocity, which allows to solve the cell-centred values of temperature of the heat conduction term implicitly, see Eq. (47), and, thus, time-step restrictions associated with an explicit treatment of the heat conduction term [37] do not apply for the presented algorithm.

Following the work of Khosla and Rubin [80], the TVD interpolation of advected variables, see Eq. (28), is implemented using a deferred correction approach, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq286.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq287.png)

where the upwind contribution is treated implicitly and the high-order correction is based on the values of the previous nonlinear iteration. This interpolation is unconditionally stable [21, 64, 80], which is essential for the simulation of convection-dominated flows with Peclet numbers of Pe = ρ|u|∆x/µ ≫1 and, in particular, inviscid flows (Pe →∞). The coefficients of the linear equation system Aψ = σ, Eq. (33), for cell P follow after rearranging the discretised and linearised governing equations (44), (46) and (47) as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq288.png)

Aρuj,p P p(n+1) P + Aρuj,p Q p(n+1) Q + Aρuj,uj P u(n+1) j,P + Aρuj,uj Q u(n+1) j,Q + Aρuj,ui P u(n+1) i,P + Aρuj,ui Q u(n+1) i,Q = σρuj P (51)

Aρh,p P p(n+1) P + Aρh,p Q p(n+1) Q + Aρh,ui P u(n+1) i,P + Aρh,ui Q u(n+1) i,Q + Aρh,T


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq289.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq290.png)

respectively, with Q the neighbour cells of cell P. The individual coefficients A and right-hand side contributions σ are given in Appendix A. The strong implicit coupling of pressure, density and velocity through a Newton linearisation has been shown to be beneficial for the performance and stability of the solution algorithm in all Mach number regimes [33]. For instance, the Newton linearisation of the advection term of the continuity equation (44) facilitates a smooth transition from low to high Mach number regions [19, 33, 48], with the term �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq291.png)

dominant at low Mach numbers and the term �

f ˜ρ(n+1) f ϑ(n) f Af dominant in regions of high Mach numbers [43]. As a result, the Newton linearisation of the advection term also yields performance and stability benefits for flows with sharp changes in Mach number and strong compressibility [33], and provides the necessary implicit pressure-velocity coupling for incompressible flows [32, 43].

5.4. Incompressible limit The incompressible limit deserves special attention, as this is the Achilles’ heel of many previously proposed numerical frameworks for flows at all speeds. From a numerical viewpoint, the incompressible limit includes both the flow of compressible fluids with very small Mach numbers (M →0) and the flow of incompressible fluids (ρ = const.). As density changes of the fluid particles vanish in the incompressible limit, with dρ →0, the density is constant along the fluid particle trajectories [13], with


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq292.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq293.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq294.png)

10

Inserting Eq. (53) into the governing equations (1)-(3) yields


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq295.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq296.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq297.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq298.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq299.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq300.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq301.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq302.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq303.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq304.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq305.png)

for the continuity, momentum and energy equations, respectively, in the incompressible limit. Applying the discretisation and linearisation schemes presented in the previous sections to the governing equations in the incompressible limit, Eqs. (54)-(56), the discretised continuity equation follows as �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq306.png)

the discretised momentum equations are given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq307.png)



u(n+1) j,P −u(t−∆t1) j,P ∆t1 VP + �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq308.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq309.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq310.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq311.png)

= − �

f

p(n+1) f nj,f Af + �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq312.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq313.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq314.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq315.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq316.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq317.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq318.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq319.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq320.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq321.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq322.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq323.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq324.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq325.png)

and the discretised energy equation follows as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq326.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq327.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq328.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq329.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq330.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq331.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq332.png)

= p(n+1) P −p(t−∆t1) P ∆t1 VP + �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq333.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq334.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq335.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq336.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq337.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq338.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq339.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq340.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq341.png)

+ �


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq342.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq343.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq344.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq345.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq346.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq347.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq348.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq349.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq350.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq351.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq352.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq353.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq354.png)

The definition of the semi-implicit advecting velocity ϑ(n+1) f , with the implicit treatment of the cell-centred pressure values, as defined in Eq. (45), yields a consistent discretisation of Eq. (54) as a function of pressure. This allows pressure to enforce a divergence-free velocity field in the incompressible limit, as well as a robust implicit pressure-velocity coupling for the collocated variable arrangement. Furthermore, Eqs. (57)-(59) treat all the terms implicitly which Nerinckx et al. [55] identified to carry acoustic information, thereby eliminating the acoustic time-step restriction and enabling an efficient solution for M →0 and, specifically, for M = 0. In fact, Eqs. (57) and (58) are identical to the discretised continuity and momentum equations of the fullycoupled pressure-based algorithm for incompressible interfacial flows of Denner and van Wachem [32]. Thus, the discretised governing equations presented in Section 5.2 represent the incompressible limit accurately and facilitate the simulation of incompressible flows. If isothermal incompressible fluids are considered, the energy equation may be disregarded, removing Eq. (47) from Eq. (33), although this simplification is not taken into account in the results presented in Section 6.

6. Validation

The results for a broad variety of test-cases are presented here to scrutinise each aspect of the thermodynamic closure, the finite-volume discretisation and the fully-coupled pressure-based algorithm, including the convergence and conservation properties. In Section 6.1, the propagation of acoustic waves is considered to test the accurate prediction of acoustic effects for both ideal-gas and real-gas fluids, in particular the amplitude of pressure waves and the speed of sound. The propagation of a moving contact discontinuity is considered in Section 6.2 to test the convergence under mesh refinement for linearly degenerate waves, a distinct challenge for finite-volume methods [81]. In Section 6.3, the propagation of a strong shock wave with Mach number 100 is considered to check if the proposed finite-volume framework converges to the correct weak solution of the governing equations, for both ideal-gas and real-gas fluids. Shock tubes with flows in different Mach number

11


> **Table 1: Fluid properties considered for the propagation of acoustic waves.**


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq355.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq356.png)


> **Table 2: Density ρ and speed of sound a0 of the fluids defined in Table 1 for ambient pressure p0 = 105 Pa and ambient temperature T0 = 300 K, the applied time-step ∆t, the frequency f of the acoustic waves, the wavelength λ0 and pressure amplitude ∆p0 of the acoustic waves based on linear acoustic theory, as well as the wavelength λ and pressure amplitude ∆p of the acoustic waves computed with the proposed numerical framework.**

Fluid ρ [kg m−3] a0 [m s−1] ∆t [s] f [s−1] λ0 [m] λ [m] ∆p0 [Pa] ∆p [Pa] Air 1.1574 347.8 2.5 × 10−6 1750 0.199 0.199 4.025 4.025 JA2 propellant gas [84] 1.2214 316.9 2.7 × 10−6 1750 0.181 0.181 3.871 3.869 Water 1 [85] 1000.0 1449 6.0 × 10−7 7000 0.207 0.207 14490 14487 Water 2 [9] 1053.6 1615 5.4 × 10−6 7000 0.230 0.231 17016 17012

regimes, ranging from M = 8.5×10−3 to M = 239, are compared against the exact Riemann solution in Section 6.4. The evolution of Taylor vortices in an inviscid fluid is considered in Section 6.5 to test the conservation of kinetic energy of the proposed numerical framework. In Section 6.6, the Poiseuille flow of an incompressible fluid and the Couette flow of a compressible fluid are simulated to probe the prediction of diffusion-dominated flows, both momentum diffusion and heat conduction, by the proposed numerical framework. The flow of an incompressible fluid in a lid-driven cavity at different Reynolds numbers is considered in Section 6.7 to test the accurate prediction of flows in which both advection and diffusion play an important role, and to demonstrate the correct enforcement of ∇· u = 0 for incompressible fluids. In Section 6.8, a supersonic flow of an ideal gas and a real gas over a forward-facing step are simulated, predominantly to scrutinise the mass conservation for a complex flow in which different Mach number regimes coexist. Finally, in Section 6.9, the Stokes flow around a rotating sphere is simulated to demonstrate the reliable prediction of flows in complex geometries.

6.1. Acoustic waves As a first test, the propagation of acoustic waves in a one-dimensional domain is simulated. The formation and propagation of acoustic waves is an important feature of compressible flows and predicting acoustic waves reliably is known to be challenging [43, 44, 63, 82]. In these simulations, the acoustic waves are generated at the domain inlet by a sinusoidal velocity perturbation with amplitude ∆u0. For small perturbations to the flow, ∆u0 ≪a0, the resulting wave is a sound wave propagating with the speed of sound a0. According to linear acoustic theory, the pressure wave has an amplitude of ∆p0 = Z ∆u0 [83], where Z = ρa is the acoustic impedance. Four different fluids, with the fluid properties given in Table 1, are considered. In each case, the unperturbed flow velocity is u0 = 1 m s−1, the ambient pressure is p0 = 105 Pa and the ambient temperature is T0 = 300 K, leading to the density and speed of sound given in Table 2. The computational domain has a length of 1 m, which is represented by an equidistant mesh with mesh spacing ∆x = 2×10−3 m, and the applied time-steps, see Table 2, correspond to a Courant number of Co = a0∆t/∆x ≃0.43. The velocity at the domain inlet is uin = u0 + ∆u0 sin(2πft), with frequency f as given in Table 2 and amplitude ∆u0 = 0.01 u0. The computed pressure amplitude ∆p and the theoretical pressure amplitude ∆p0 based on linear acoustic theory, both given in Table 2, are in excellent agreement. Figure 2 shows the profiles of the pressure amplitude ∆p of the acoustic waves in the four considered fluids as a function of space, with good agreement of the minimum and maximum pressure amplitude with the theoretical pressure amplitude. In addition, the computed wavelength λ is predicted accurately compared to the theoretical wavelength λ0, given in Table 2, demonstrating a correct prediction of the speed of sound.

6.2. Moving contact discontinuity A contact discontinuity is a linearly degenerate wave and represents the main source of error in terms of convergence of the applied finite-volume method under mesh refinement [81, 86], with the contact discontinuity progressively smoothing over the course of the simulation [87, 88]. To test the accuracy of the proposed finitevolume framework in predicting contact discontinuities, a moving contact discontinuity in a one-dimensional domain with a length of 1 m is simulated, as considered in previous studies [59, 63]. The contact discontinuity is initially located at x0 = 0.5 m, with the initial conditions of the left and right states given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq357.png)

12

0 0.2 0.4 0.6 0.6 0.8 1.0 -6

-3

0

3

6


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq358.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq359.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq360.png)

∆p [Pa]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq361.png)

0 0.2 0.4 0.6 0.6 0.8 1.0 -6

-3

0

3

6


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq362.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq363.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq364.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq365.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq366.png)

0 0.2 0.4 0.6 0.6 0.8 1.0 -20

-10

0

10

20 ∆p0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq367.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq368.png)

∆p [kPa]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq369.png)

0 0.2 0.4 0.6 0.6 0.8 1.0 -20

-10

0

10

20


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq370.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq371.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq372.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq373.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq374.png)


> **Figure 2: Profiles of the pressure amplitude ∆p of acoustic waves in different fluids, with the fluid properties given in Table 1 and the frequency given in Table 2. The theoretical pressure amplitudes, ±∆p0, based on linear acoustic theory are given as a reference.**

The contact discontinuity is simulated in an IG fluid with γ = 1.4 and cp = 1008 J kg−1 K−1, as well as an NASG fluid with γ = 2.0, cp = 114.286 J kg−1 K−1, Π = 5.0 Pa and b = 10−3 m3 kg−1. The transient terms are discretised using the BDF2 scheme and the applied time-step corresponds to Co = uL∆t/∆x = 0.5. Figure 3 shows the density profiles at t = 0.3 s for both the IG fluid and the NASG fluid, using different mesh resolutions. The results of both fluids are in very good agreement and, irrespective of the mesh resolution, the contact discontinuity propagates with the correct velocity. The convergence of the L1-norm of the solution error associated with a linearly degenerate wave for a q-th order advection scheme without compressive limiting is of order q/(q + 1) [81]. The spatial convergence of the L1-norm of the density error,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq375.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq376.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq377.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq378.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq379.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq380.png)

where ρcomp. P is the computed density at cell P and ρexact P is the corresponding exact density value, obtained with the upwind scheme (q = 1) and the Minmod scheme (q = 2) matches the theoretical order of convergence closely in both fluids, as observed in Fig. 4, with convergence order 1/2 using the upwind scheme and order 2/3 using the Minmod scheme. Furthermore, the self-similarity of the transport of the contact discontinuity is not affected by the choice of fluid model, resulting in only minute differences in the L1-norm ℓ1(ρ) between the IG fluid and the NASG fluid for a given mesh resolution, with |ℓ1(ρ)IG −ℓ1(ρ)NASG|/ℓ1(ρ)IG < 10−2 using the Minmod scheme and |ℓ1(ρ)IG −ℓ1(ρ)NASG|/ℓ1(ρ)IG < 10−7 using the upwind scheme.

In order to test the progressive smearing of the contact discontinuity during the course of the simulation, the contact discontinuity in the IG fluid is simulated in a domain with a length of 3 m and with the contact discontinuity initially located at x0 = 0.1 m. The computational domain is resolved by 1200 equidistant mesh cells and the advection terms are discretised using the Minmod scheme. All other settings remain the same as above. The computed density profiles after n ∈{20, 200, 2000} time-steps are plotted in Fig. 5a, clearly showing a progressive smearing of the contact discontinuity. The width of the contact discontinuity should be proportional to n1/(q+1) for a q-th order finite-difference or finite-volume method [87, 88], where n is the number of time-steps. Figure 5b shows the width d of the contact discontinuity as a function of the time-step n, with the width d given by the distance between the points at which the density takes the values 0.55 kg m−3 and 0.95 kg m−3, as illustrated in the inset of Fig. 5b, since the density changes abruptly between 0.5 kg m−3 and 1.0 kg m−3 at the considered contact discontinuity. As shown in Fig. 5b, the width of the contact discontinuity increases with the number of time-steps with a slope closely matching n1/3, which is the increase expected for a consistently second-order finite-volume method [87].

13

0.4 0.5 0.6 0.7 0.8

0.5

0.6

0.7

0.8

0.9

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq381.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq382.png)

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq383.png)

(a) IG fluid

0.4 0.5 0.6 0.7 0.8

0.5

0.6

0.7

0.8

0.9

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq384.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq385.png)

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq386.png)

(b) NASG fluid


> **Figure 3: Profiles of the density ρ of a moving contact discontinuity in (a) an IG fluid and (b) an NASG fluid, on equidistant meshes with different resolutions. The advection terms are discretised using the Minmod scheme, the transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = uL∆t/∆x = 0.5.**


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq387.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq388.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq389.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq390.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq391.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq392.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq393.png)

Upwind Minmod

(a) IG fluid


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq394.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq395.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq396.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq397.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq398.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq399.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq400.png)

Upwind Minmod

(b) NASG fluid


> **Figure 4: Spatial convergence of the L1-norm of the density error, ℓ1(ρ), as defined in Eq. (60), of a moving contact discontinuity in (a) an IG fluid and (b) an NASG fluid, using the first-order upwind scheme and the Minmod scheme. The transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = uL∆t/∆x = 0.5.**

-0.1 0 0.1

0.5

0.6

0.7

0.8

0.9

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq401.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq402.png)

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq403.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq404.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq405.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq406.png)

100


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq407.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq408.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq409.png)

Minmod

-0.1 0 0.1

0.55

0.75

0.95


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq410.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq411.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq412.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq413.png)


> **Figure 5: (a) Profiles of the density ρ of a moving contact discontinuity in an IG fluid, after a different number of time-steps n, with x′ the position of the contact discontinuity, and (b) the width d of the contact discontinuity, with its definition illustrated in the inset, as a function of time-steps n. The advection terms are discretised using the Minmod scheme, the transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = uL∆t/∆x = 0.5.**

14

0.6 0.7 0.8 0.9 1 105

106

107

108

109


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq414.png)

p [Pa]

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq415.png)

(a) Air (IG fluid), t = 14.38 µs


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq416.png)

105

107

109

1011

1013


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq417.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq418.png)

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq419.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq420.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq421.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq422.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq423.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq424.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq425.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq426.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq427.png)

Air Water


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq428.png)


> **Figure 6: Profiles of the pressure p of a shock wave with Mach number Ms = 100 in (a) air, described as an IG fluid, and (b) water, described as an NASG fluid, and (c) spatial convergence of the L1-norm of the density error, ℓ1(ρ), as defined in Eq. (60). The exact solution given by the Rankine-Hugoniot relations, Eq. (61), is shown as a reference in (a) and (b). The applied time-step corresponds to Co = uL∆t/∆x = 0.5.**

6.3. Shock waves The propagation of a shock wave poses particular challenges for finite-volume methods, because a shock wave is discontinuous and valid solutions of the governing conservation laws are not guaranteed to satisfy the second law of thermodynamics across shock waves [89]. As such, simulating the propagation of a shock wave is well suited to test whether a numerical scheme reliably converges to the physically-correct weak solution of the governing conservation laws, which is a prerequisite for the accurate prediction of both the speed and strength of shock waves [89, 90]. To this end, the Lax-Wendroff theorem [91] stipulates that if a conservative numerical scheme for hyperbolic conservation laws converges, the computed solution converges towards a weak solution of the conservation laws. The propagation of a strong shock wave with Mach number Ms = 100 in air and water in a one-dimensional domain with a length of 1 m is simulated. Air is described by the IG model using the fluid properties given in Table 1 and water is described by the NASG model using the fluid properties proposed by Le M´etayer and Saurel [9], also given in Table 1 (see properties of Water 2). Viscous stresses and heat conduction are neglected, i.e. µ = k = 0, so the governing equations (1)-(3) reduce to the Euler equations [92], which are hyperbolic. From the Rankine-Hugoniot relations, the pressure and density ratios across a shock wave propagating with velocity us in a quiescent NASG fluid are given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq429.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq430.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq431.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq432.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq433.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq434.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq435.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq436.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq437.png)

where subscript I denotes the post-shock state, subscript II denotes the pre-shock state and Ms = us/aII is the Mach number of the shock wave. With the initial conditions of the pre-shock state (II) for both cases given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq438.png)

the shock relations yield the initial conditions of the post-shock state (I) for air,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq439.png)

and for water, pI = 7.62925 × 1012 Pa, uI = 44833.0 m s−1, TI = 278744 K.

The shock wave is initially located at xs,0 = 0.25 m and the applied time-step corresponds to Co = us∆t/∆x = 0.5. The Rankine-Hugoniot relations are reproduced accurately in both air and water, as seen in Fig. 6, despite the very large pressure discontinuities with pressure ratios of more than four and seven orders of magnitude, respectively. In both fluids the L1-norm of the density error, ℓ1(ρ), converges with first order under mesh refinement, as seen in Fig. 6c. The first order convergence is imposed by the applied monotone discretisation

15

0 0.2 0.4 0.6 0.8 1

10000.0

10000.5

10001.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq440.png)

p [Pa]

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq441.png)

(a) Pressure p


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq442.png)

24.999

25.000


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq443.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq444.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq445.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq446.png)

0.00845

0.00850

0.00855


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq447.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq448.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq449.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq450.png)


> **Figure 7: Profiles of pressure, density and Mach number of the low-Mach shock tube at t = 0.01 s, compared against the theoretical Riemann solution. In addition, a magnified view of the minute change in Mach number at the contact discontinuity is shown in (c).**

0 0.2 0.4 0.6 0.8 1 0

0.2

0.4

0.6

0.8

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq451.png)

p [Pa]

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq452.png)

(a) Pressure


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq453.png)

0.2

0.4

0.6

0.8

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq454.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq455.png)

(b) Density


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq456.png)

0.2

0.4

0.6

0.8

1.0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq457.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq458.png)

(c) Mach number


> **Figure 8: Profiles of pressure, density and Mach number of Sod’s shock tube at t = 0.15 s, compared against the theoretical Riemann solution.**

schemes, in this case the Minmod scheme, and is expected for an oscillation-free numerical simulation of a shock wave [93]. The robust convergence for strong shock waves further implies accurate conservation properties as well as convergence to the correct weak solution of the governing conservation laws using the proposed finite-volume framework and pressure-based algorithm.

6.4. Shock tubes

Shock tubes are routinely and extensively used to test numerical frameworks and schemes for compressible flows, because they feature shock waves, rarefaction fans as well as contact discontinuities and because an exact reference solution based on the associated Riemann problem exists. Three different shock tubes, covering Mach numbers over five orders of magnitude, are considered. In all cases, the fluid has a heat capacity ratio of γ = 1.4 and a specific gas constant of cp = 1008 J kg−1 K−1. A low-Mach shock tube, as proposed by Moguen et al. [94], is considered. The discontinuity is initially located at x0 = 0.5 m, with the initial conditions of the left and right states given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq459.png)

The applied time-step corresponds to a Courant number of Co = (uL + aL)∆t/∆x = 0.5. Overall, the results obtained on both meshes are in very good agreement with the theoretical Riemann solution, as seen in Fig. 7. Because the particle velocity is very small, umax = 0.202 m s−1, the contact discontinuity only moves by 0.002 m in the studied time frame and, thus, remains very sharp, as evident by the density profile in Fig. 7. A small wiggle is observed in the Mach number profile at the contact discontinuity, which however has no impact on the overall result.

16

0 0.2 0.4 0.6 0.8 1 0

3

6

9

12

15

18


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq460.png)

p [MPa]

Theory


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq461.png)

(a) Pressure


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq462.png)

25

50

75

100

125


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq463.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq464.png)

(b) Density


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq465.png)

50

100

150

200

250


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq466.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq467.png)

(c) Mach number


> **Figure 9: Profiles of pressure, density and Mach number of the high-Mach shock tube at t = 3.5 × 10−4 s, compared against the theoretical Riemann solution.**

The shock tube initially introduced by Sod [95] is considered as a shock tube with intermediate Mach number, with initial conditions ρL = 1.0 kg m−3, uL = 0 m s−1, pL = 1.0 Pa, ρR = 0.125 kg m−3, uR = 0 m s−1, pR = 0.1 Pa.

The discontinuity is initially located at x0 = 0.5 m and the applied time-step corresponds to a Courant number of Co = aL∆t/∆x = 0.6. The results obtained on both meshes, shown in Fig. 8, are in very good agreement with the theoretical Riemann solution. The high-Mach shock tube proposed by Xiao [54] is considered. The discontinuity is initially located at x0 = 0.5 m, with the initial conditions


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq468.png)

Notably, the flow of the left state has a Mach number of ML = 239. The applied time-step corresponds to a Courant number of Co = uL∆t/∆x = 0.5. As observed in Fig. 9, although the profile of the Mach number is not predicted very accurately on the coarse mesh, the density and pressure profiles are in good agreement with the theoretical Riemann solution. On the fine mesh, the computed results are in very good agreement with the theoretical Riemann solution, demonstrating the accurate prediction of high-Mach Riemann problems with the proposed numerical framework.

6.5. Taylor vortices

The conservation of kinetic energy is a fundamental property arising from the conservation of mass and momentum. Two-dimensional Taylor vortices in an inviscid (µ = 0), non-conducting (k = 0) fluid are simulated to analyse the conservation of kinetic energy by the proposed numerical framework. The domain has the dimensions 2 m × 2 m and is periodic in all directions, so that energy transfer across the domain boundaries does not have to be considered. The initial conditions, shown in Fig. 10, are u = −cos(πx) sin(πy), v = sin(πx) cos(πy) and p = −0.25 [cos(2πx) + cos(2πy)]. Since µ = k = 0, the Taylor vortices are steady and no energy dissipation occurs naturally, with a constant kinetic energy of


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq469.png)

2

ˆ


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq470.png)

2


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq471.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq472.png)

where Ωis the volume of the computational domain. Any dissipation of kinetic energy is, thus, the result of numerical dissipation induced by the applied discretisation. Figure 11a shows the evolution of the error in kinetic energy of the Taylor vortices,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq473.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq474.png)

with E(0) kin the kinetic energy of the initialised (t = 0) flow field, in an IG fluid with γ = 1.4 and cp = 1008 J kg−1 K−1, and Mach number M = 0.01. As expected, the error in kinetic energy is substantially larger

17


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq475.png)


> **Figure 10: Contours of the initial velocity u along the x-axis and the initial pressure p of the Taylor vortices.**

0 0.2 0.4 0.6 0.8 1 10−6


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq476.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq477.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq478.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq479.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq480.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq481.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq482.png)

Upwind Central

(a) Evolution of εkin with MWI.


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq483.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq484.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq485.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq486.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq487.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq488.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq489.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq490.png)

Central


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq491.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq492.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq493.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq494.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq495.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq496.png)

100


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq497.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq498.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq499.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq500.png)

Upwind Central


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq501.png)


> **Figure 11: Temporal evolution of εkin, Eq. (64), on an equidistant Cartesian mesh with ∆x = 0.04 m defining the advecting velocity ϑf (a) with the MWI as described in Section 5.1 and (b) without the MWI as ϑf = uf · nf, and (c) convergence of the error in kinetic energy εkin of the Taylor vortices with the MWI as described in Section 5.1. The first-order upwind scheme or the central differencing scheme are applied for the discretisation of the advection term. The applied time-step in all cases is ∆t = 2 × 10−3 s.**

using the first-order upwind scheme compared to the error in kinetic energy obtained using the central differencing scheme. Interestingly, the applied transient discretisation scheme, i.e. BDF1 or BDF2, does not affect the error in kinetic energy, which is consistent with the Taylor vortices being a steady flow in the absence of molecular viscosity and heat conduction. However, even with central differencing, kinetic energy is dissipated as a result of the MWI formulation of the advecting velocity [65], see Eq. (34). No appreciable distortion of the vortices is observed for the considered simulations when central differencing is applied, which is consistent with the only small error in kinetic energy (εkin < 1%) in theses cases.

The flow is sufficiently compressible (M = 0.01) and smooth, that pressure and velocity remain coupled even without MWI [65]. Exploiting this by omitting the correction introduced by the MWI, with the advecting velocity simply defined as ϑf = uf · nf, the error in kinetic energy remains constant for t ≳0.08 s, as seen in Fig. 11b, which indicates that the numerical dissipation of kinetic energy is negligible. This is to be expected when simulating a sufficiently smooth flow with a second-order accurate finite-volume framework without any explicitly introduced physical or numerical dissipation. Only a small error in kinetic energy is observed at the beginning of the simulation, caused by the initial conditions [65]. The error in kinetic energy converges with third order using central differencing under mesh refinement, as shown in Fig. 11c, which is consistent with the third-order convergence of the error in kinetic energy introduced by the MWI [65]. On the other hand, when the first-order upwind scheme is applied, the kinetic energy dissipated artificially by the MWI is insignificant compared to the numerical diffusion introduced by the upwind scheme, as evident by the first-order convergence of the error in kinetic energy shown in Fig. 11c. These results, therefore, suggest that the MWI is the only source of numerical dissipation in the proposed finite-volume discretisation, assuming a consistent second-order (or higher-order) interpolation of spatial and transient terms is applied, e.g. central differencing and BDF2.

18


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq502.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq503.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq504.png)

d

x

y

(a) Schematic


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq505.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq506.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq507.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq508.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq509.png)

1


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq510.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq511.png)

Simulation Theory

(b) Axial velocity profile


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq512.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq513.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq514.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq515.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq516.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq517.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq518.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq519.png)


> **Figure 12: Schematic of a planar Poiseuille flow, as well as the profile of the axial velocity compared against the analytical solution, Eq. (65), and spatial convergence of the L∞-norm of the error in axial velocity, Eq. (66), of the planar Poiseuille flow of an incompressible fluid. The axial velocity profile in (b) is obtained on a mesh with ∆y = d/20, with each dot representing a cell-centred value.**

6.6. Diffusion-dominated flows The test-cases discussed in the previous sections only test the discretisation of the transient and advection terms, not taking into account diffusion terms, i.e. viscous stresses, heat conduction and viscous heating. Two well-defined diffusion-dominated flows, a planar Poiseuille flow of an incompressible fluid and a planar Couette flow of a compressible fluid, are considered to test the discretisation and implementation of the diffusion of momentum and heat. The planar Poiseuille flow of an incompressible fluid between two parallel plates of infinite length separated by a constant distance d, illustrated schematically in Fig. 12a, is a flow that is entirely governed by viscous stresses. Assuming the viscosity µ is constant and the flow is laminar, the velocity profile is readily given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq520.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq521.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq522.png)

where −dp/dx is the driving pressure gradient. This type of flow, thus, allows a straightforward quantification of the solution error associated with the axial velocity. The computational domain is taken to be periodic in the streamwise direction, to circumvent any influence of inlet and outlet boundary conditions, and the flow is driven by a constant momentum source corresponding to the driving pressure gradient −dp/dx. The profile of the axial velocity U of the planar Poiseuille flow obtained on a mesh with a resolution of ∆y = d/20 is shown in Fig. 12b, alongside the spatial convergence of the L∞-norm of the error in axial velocity,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq523.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq524.png)

in Fig. 12c. The axial velocity profile is in excellent agreement with the analytical solution, Eq. (65), and the L∞-norm of the error in axial velocity converges with second order under mesh refinement, as expected given the second-order discretisation of the viscous stresses in Eq. (38). The planar Couette flow of a compressible fluid between two parallel plates of infinite length separated by a constant distance d, illustrated schematically in Fig. 13a, is a compressible flow that is dominated by viscous stresses and heat conduction. Assuming the viscosity µ is constant and the stationary wall is adiabatic, the velocity and temperature profiles only depend on the Prandtl number Pr = µ cp/k and the Mach number Mm = Um/am at the moving wall, with the velocity given as U(y)/Um = y/d and the temperature given as [96]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq525.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq526.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq527.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq528.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq529.png)

This type of flow, thus, allows a straightforward quantification of the solution error associated with the temperature. The considered fluid is an ideal gas with a Prandtl number of Pr = 1 and a heat capacity ratio of γ = 1.4. The computational domain is taken to be periodic in the streamwise direction, to circumvent any influence of inlet and outlet boundary conditions. The profile of the temperature T of the planar compressible Couette flow with Mm = 1.0 obtained on a mesh with a resolution of ∆y = d/20 is shown in Fig. 13b, alongside the spatial convergence of the L∞-norm of the error in temperature,


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq530.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq531.png)

19


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq532.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq533.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq534.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq535.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq536.png)

d

x

y

(a) Schematic


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq537.png)

1.05

1.10

1.15

1.20


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq538.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq539.png)

Simulation Theory

(b) Temperature profile


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq540.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq541.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq542.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq543.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq544.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq545.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq546.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq547.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq548.png)


> **Figure 13: Schematic of a planar Couette flow, as well as the profile of the temperature compared against the analytical solution, Eq. (67), and spatial convergence of the L∞-norm of the temperature, Eq. (68), of the planar Couette flow of a compressible fluid for both considered Mach numbers. The temperature profile in (b) is obtained on a mesh with ∆y = d/20, with each dot representing a cell-centred value.**

Wall (moving)

Wall (stationary)

Wall (stationary)

Wall (stationary)

L


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq549.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq550.png)

y

x

(a) Schematic (b) Polygonal mesh


> **Figure 14: Schematic illustration and polygonal mesh of the lid-driven cavity.**

at Mach numbers Mm ∈{0.1, 1.0} in Fig. 13c. The temperature profile is in excellent agreement with the analytical solution, Eq. (67), and the L∞-norm of the error in temperature converges with second order under mesh refinement for both Mach numbers, as expected given the second-order discretisation of the heat conduction term in Eq. (40). Notably, ℓ∞(T) is independent of the Mach number Mm for a sufficiently high spatial resolution, as seen in Fig. 13c.

6.7. Lid-driven cavity The lid-driven cavity, schematically shown in Fig. 14a, is a common test case to validate numerical methods for fluid flows, since it captures convective and diffusive momentum transport of the fluid. The considered two-dimensional domain is of size L × L, with no-slip boundary conditions imposed on all four walls. The top wall moves with velocity uw and the flow of the incompressible fluid has a Reynolds numbers of Re = ρ L uw/µ ∈ {100, 1000}. A polygonal mesh with 8708 cells, shown in Fig. 14b, represents the computational domain.

Figures 15 and 16 show the u-velocity profile in the y-direction and the v-velocity profile in the x-direction along lines that pass through the centre of the domain for the two considered Reynolds numbers, compared against the reference results of Ghia et al. [97]. The results are in very good agreement with the reference results of Ghia et al. [97], as well as other studies that have previously considered this test-case [43, 60, 67, 98], for both considered Reynolds numbers, demonstrating the accurate prediction of the convective-diffusive transport of momentum on unstructured meshes using the proposed algorithm. The contours of the divergence of velocity, ∇· u, at steady state are shown in Fig. 17 for the lid-driven cavity with Re = 1000, alongside the transient

20


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq551.png)

0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq552.png)

1


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq553.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq554.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq555.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq556.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq557.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq558.png)

0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq559.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq560.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq561.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq562.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq563.png)


> **Figure 15: Profiles of (a) the u-velocity along the y-centreline of the domain and (b) the v-velocity along the x-centreline of the domain of the lid-driven cavity with Re = 100. The results of Ghia et al. [97] are shown as a reference.**


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq564.png)

0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq565.png)

1


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq566.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq567.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq568.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq569.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq570.png)

-0.3

0

0.3

0.6

0.9


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq571.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq572.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq573.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq574.png)


> **Figure 16: Profiles of (a) the u-velocity along the y-centreline of the domain and (b) the v-velocity along the x-centreline of the domain of the lid-driven cavity with Re = 1000. The results of Ghia et al. [97] are shown as a reference.**

(a) Divergence of the velocity field at steady state


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq575.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq576.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq577.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq578.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq579.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq580.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq581.png)


> **Figure 17: (a) Contours of the divergence of the velocity field, ∇· u, at steady state and b) L1-norm of the error in the divergence of the velocity field, ℓ1(∇· u), for the lid-driven cavity with Re = 1000.**

21


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq582.png)


> **Figure 18: Contours of Mach number and pressure of the supersonic flow over a forward-facing step at t = 4 s with co-volume b = 0.**


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq583.png)


> **Figure 19: Contours of Mach number and pressure of the supersonic flow over a forward-facing step at t = 4 s with co-volume b = 0.1 m3 kg−1.**

evolution (considering an initially quiescent fluid) of the L1-norm of the error in the divergence of the velocity field, given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq584.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq585.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq586.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq587.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq588.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq589.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq590.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq591.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq592.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq593.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq594.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq595.png)

where f are the faces of cell P. The divergence-free condition of the velocity field imposed by the conservation of mass in conjunction with the considered incompressible fluid, see Eq. (54), is satisfied accurately, with only marginal errors subject to the applied tolerance of the iterative solver (see Section 5). This is to be expected from the proposed algorithm, as ∇· u = 0 is implicitly enforced by Eq. (57).

6.8. Forward-facing step The two-dimensional supersonic flow over a forward-facing step of an initially uniform flow features the spatiotemporal evolution of shock waves, developing transonic flow and large pressure gradients. This test-case is, thus, well suited to test the conservation properties of the finite-volume discretisation as well as the stability of the pressure-based algorithm during the transient development of large pressure gradients. Following Woodward and Colella [99], the height of the computational domain is 1 m, and the step has of height 0.2 m and is positioned at 0.6 m from the inlet of the domain. The flow entering the domain has a Mach number of M = u/a0 = 3 and a pressure of p0 = 1 Pa. The two-dimensional domain is represented by an equidistant Cartesian mesh with ∆x = 0.01 m and the applied time-step corresponds to Co = u ∆t/∆x = 0.75. The considered fluid has a heat capacity ratio of γ = 1.4 and a specific isobaric heat capacity of cp = 1008 J kg−1 K−1, with a co-volume of either b = 0 or b = 0.1 m3 kg−1. Figure 18 shows the contours of the Mach number and the pressure at t = 4 s for b = 0, which are in good agreement with previously reported results [33, 99, 100]. Changing the co-volume to b = 0.1 m3 kg−1, the position of the primary shock wave in front of the forward-facing step moves further upstream and fewer reflected shock waves can be observed, as seen in Fig. 19. Based on the initial mass m(0) at t = 0, the mass in the domain Ωand the mass entering and leaving the domain over its boundaries ∂Ω, the conservation error of mass at time t is given as


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq596.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq597.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq598.png)

0

‹


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq599.png)

where Σ is the outward-pointing surface vector of the surface ∂Ωof the computational domain Ω. The temporal evolution of the mass conservation error of the supersonic flow over the forward-facing step is shown

22

0 1 2 3 4 10−10


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq600.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq601.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq602.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq603.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq604.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq605.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq606.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq607.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq608.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq609.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq610.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq611.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq612.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq613.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq614.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq615.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq616.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq617.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq618.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq619.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq620.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq621.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq622.png)


> **Figure 20: Temporal evolution of the mass conservation error, εm, as defined in Eq. (70), of the supersonic flow over a forward-facing step, obtained with different solution tolerances η.**

100R


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq623.png)

y

x


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq624.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq625.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq626.png)

(a) Schematic (not to scale) (b) Mesh with velocity contours


> **Figure 21: Schematic of the flow around a rotating sphere (in the xy-plane through the centre of the domain) and applied mesh in the vicinity of the sphere together with the contours of the axial velocity.**

in Fig. 20, obtained with both considered co-volumes, b ∈{0, 0.1} m3 kg−1, with different solution tolerances, η ∈{10−5, 10−6, 10−7, 10−8}, applied for the solution of the system of governing equations (33). Overall, the proposed finite-volume framework conserves mass accurately and the mass conservation error is predominantly a function of the solution tolerance, with a decreasing mass conservation error for a decreasing solution tolerance.

6.9. Rotating sphere

The flow of an incompressible fluid around a sphere with radius R, rotating at angular velocity ω, in a Stokes flow with Reynolds number Re = ρR|u∞|/µ ≪1, where u∞is the free-stream velocity, is considered. As a result of the rotation, a lift force is acting on the sphere, also known as Magnus effect, with the analytical solution for the force on the sphere given as [101]


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq627.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq628.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq629.png)

where the first term on the right-hand side represents the drag force and the second term represents the lift force. The sphere is simulated in a cubical three-dimensional domain of size 100R × 100R × 100R, illustrated schematically in Fig. 21a, with the sphere placed at the centre of the domain. The considered flow has the free-stream velocity u∞= (ux,∞, 0, 0)T , corresponding to Re = 0.05, and the sphere rotates around its z-axis with ω = (0, 0, ωz)T . The computational domain is represented with a boundary-fitted hexahedral mesh with 384 000 cells, shown in Fig. 21b, which is strongly refined in the vicinity of the sphere and gradually coarsened (growth factor 1.2) with increasing distance from the sphere. The applied time-step is ∆t = 100 tµ, where tµ = ρR2/µ is the viscous timescale, which corresponds to a maximum Courant number of Co = 49 −1559,

23

0 2 000 4 000 6 000 8 000 10 000 0.96

0.98

1.00

1.02

1.04


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq630.png)

Cd/Cd,0


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq631.png)

(a) Drag coefficient

0 2 000 4 000 6 000 8 000 10 000 0.990

0.995

1.000

1.005

1.010


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq632.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq633.png)


![Equation](images/2020_Denner_conservative_pressure_based_all_speeds_v21_eq634.png)

(b) Lift coefficient


> **Figure 22: Drag coefficient Cd and lift coefficient Cl of the rotating sphere in Stokes flow for different dimensionless angular velocities ˆω as a function of the dimensionless time τ = t/tµ, normalised with the theoretical values, Cd,0 and Cl,0, based on Eq. (71).**

dependent on the angular velocity ωz, for the considered simulations. The transient term is discretised with the BDF2 scheme and the advection terms are discretised using the Minmod scheme. Fig. 22 shows the transient evolution of the drag coefficient, Cd = 2Fd/ρApu2 x,∞, and the lift coefficient, Cl = 2Fl/ρApu2 x,∞, with Ap = πR2 the projected area of the sphere, for three different dimensionless angular velocities, ˆω = R ωz/ux,∞, as a function of the dimensionless time τ = t/tµ. For all three angular velocities the drag and lift coefficients are predicted accurately compared to the analytical solution, Eq. (71), with errors < 1% for both drag and lift coefficients.

7. Conclusions

A conservative numerical framework for the prediction of flows of incompressible, ideal-gas and real-gas fluids at all speeds has been presented. This numerical framework is founded on a unified thermodynamic closure model for incompressible and compressible fluids, a standard finite-volume discretisation applicable to structured and unstructured meshes, a single flux definition based on a momentum-weighted interpolation, as well as a fully-coupled pressure-based algorithm with collocated variable arrangement. The proposed unified thermodynamic closure model combines the definitions of incompressible fluids with the Noble-Abel-stiffenedgas model [9] for ideal-gas and real-gas fluids, which facilitates a straightforward finite-volume discretisation that is applicable to incompressible flows as well as compressible flows in all Mach number regimes. Since the thermodynamic closure model requires only the definition of the density and specific static enthalpy, it can be extended to more complex gas models, such as the Peng-Robinson model [11], without changes to the finite-volume discretisation or the pressure-based algorithm. The employed finite-volume framework combines well-established conservative discretisation schemes to yield a consistently second-order accurate discretisation that is applicable to structured and unstructured meshes. The discretised governing equations are solved in a single linear system of equations for pressure, velocity and temperature, which enables a robust solution for flows at any speed. The main feature of the proposed finite-volume discretisation and pressure-based algorithm is the accurate and robust simulation of flows of incompressible and compressible fluids at all speeds without changes to the discretisation or the solution procedure. Using a Newton linearisation of the continuity equation in conjunction with the semi-implicit discretisation of the fluxes through the mesh faces by a momentum-weighted interpolation method, the discretised continuity equation acts as a transport equation for density in compressible flows and as a constraint on the velocity field in incompressible flows. This allows this numerical framework to represent the incompressible limit correctly and enables the simulation of flows of both incompressible and compressible fluids with the same algorithm. The proposed numerical framework has been validated using a broad variety of test-cases, demonstrating accurate and robust results, irrespective whether the considered flow was of an incompressible fluid, an ideal-gas fluid or a real-gas fluid, with an error convergence consistent of a second-order finite-volume discretisation. The propagation of acoustic waves demonstrated an accurate prediction of the speed of sound and acoustic effects in general, while the propagation of a moving contact discontinuity demonstrated convergence for linearly degenerate waves. The propagation of a strong shock wave as well as the shock tubes in different Mach number regimes scrutinised the resolution of strongly nonlinear and discontinuous flow features, which are predicted accurately in all Mach number regimes. In particular, the speed, position and strength of strong shock waves are predicted accurately, demonstrating that the finite-volume framework converges to the correct weak solution of the governing equations [90], further suggesting that the proposed algorithm implicitly satisfies the

24

second law of thermodynamics. The evolution of Taylor vortices in an inviscid fluid offered the possibility to test the conservation of energy of the proposed numerical framework, showing that the momentum-weighted interpolation is the only source of numerical energy dissipation, an error which however converges with third order under mesh refinement. The Poiseuille flow of an incompressible fluid and the Couette flow of a compressible fluid demonstrated the accurate simulation of flows in which viscous stresses and heat conduction play a dominant role. The flow of an incompressible fluid in a lid-driven cavity at different Reynolds numbers further demonstrated the accurate simulation of flows in which both advection and diffusion play an important role, and demonstrated the correct enforcement of ∇· u = 0 for incompressible fluids to any chosen solver tolerance (within the limit of machine precision), on unstructured meshes. The results presented for the supersonic flow of an ideal gas and a real gas over a forward-facing step demonstrated accurate mass conservation, even for complex flows in which different Mach number regimes coexist. Lastly, the Stokes flow around a rotating sphere demonstrated that flows in complex three-dimensional geometries can be predicted accurately with the proposed numerical framework. In this paper we have put forward a thermodynamic closure model, a finite-volume discretisation and a fully-coupled pressure-based algorithm for the prediction of the behaviour of the flow of incompressible fluids as well as compressible fluids described by idealor real-gas models on arbitrary meshes. We have combined these constituent parts into a fully-coupled pressure-based framework and have shown that this framework is able to predict realistic flows at any speed. However, these parts can also be used individually, for instance in existing frameworks.


## References

[1] F. H. Harlow, A. A. Amsden, Numerical calculation of almost incompressible flow, Journal of Computational Physics 3 (1968) 80–93. [2] F. H. Harlow, A. A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, Journal of Computational Physics 8 (1971) 197–213. [3] S. Klainerman, A. Majda, Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids, Communications on Pure and Applied Mathematics 34 (1981) 481–524. [4] D. Hoff, The Zero-Mach Limit of Compressible Flows, Communications in Mathematical Physics 192 (1998) 543–554. [5] E. F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical Introduction, Springer, third edition, 2009. [6] F. Harlow, A. Amsden, Fluid Dynamics, Monograph LA-4700, Los Alamos National Laboratory, 1971. [7] R. Saurel, O. Le M´etayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiff mechanical relaxation, Shock Waves 16 (2007) 209–232. [8] T. L. Hill, An Introduction to Statistical Thermodynamics, Dover Publications, New York, 1986. [9] O. Le M´etayer, R. Saurel, The Noble-Abel Stiffened-Gas equation of state, Physics of Fluids 28 (2016) 046102. [10] G. Soave, Equilibrium constants from a modified Redlich-Kwong equation of state, Chemical Engineering Science 27 (1972) 1197–1203. [11] D.-Y. Peng, D. B. Robinson, A New Two-Constant Equation of State, Industrial & Engineering Chemistry Fundamentals 15 (1976) 59–64. [12] G. Hauke, T. J. Hughes, A comparative study of different sets of variables for solving compressible and incompressible flows, Computer Methods in Applied Mechanics and Engineering 153 (1998) 1–44. [13] A. J. Chorin, J. E. Marsden, A Mathematical Introduction to Fluid Mechanics, Springer Verlag, 1993. [14] K.-H. Chen, R. Pletcher, Primitive Variable, Strongly Implicit Calculation Procedure for Viscous Flows at All Speeds, AIAA Journal 29 (1991) 1241–1249. [15] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165. [16] J. H. Ferziger, M. Peric, R. L. Street, Computational Methods for Fluid Dynamics, Springer International Publishing, 4th edition, 2020. [17] W. S. O˙za´nski, The Lagrange multiplier and the stationary Stokes equations, Journal of Applied Analysis 23 (2017). [18] A. Toutant, General and exact pressure evolution equation, Physics Letters A 381 (2017) 3739–3742. [19] J. Van Doormaal, G. Raithby, B. McDonald, The Segregated Approach to Predicting Viscous Compressible Fluid Flows, ASME Journal of Turbomachinery 109 (1987) 268–277. [20] H. Bijl, P. Wesseling, A Unified Method for Computing Incompressible and Compressible Flows in Boundary-Fitted Coordinates, Journal of Computational Physics 141 (1998) 153–173. [21] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. [22] K. C. Karki, S. V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA Journal 27 (1989) 1167–1174. [23] C. M. Rhie, Pressure-based Navier-Stokes solver using the multigrid method, AIAA Journal 27 (1989) 1017–1018. [24] S. M. H. Karimian, G. E. Schneider, Pressure-based control-volume finite element method for flow at all speeds, AIAA Journal 33 (1995) 1611–1618. [25] I. Demirdˇzi´c, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255. [26] F. Moukalled, M. Darwish, A High-Resolution Pressure-Based Algorithm for Fluid Flow at All Speeds, Journal of Computational Physics 168 (2001) 101–130. [27] S. Acharya, B. R. Baliga, K. Karki, J. Y. Murthy, C. Prakash, S. P. Vanka, Pressure-Based Finite-Volume Methods in Computational Fluid Dynamics, Journal of Heat Transfer 129 (2007) 407. [28] K. Javadi, M. Darbandi, M. Taeibi-Rahni, Three-dimensional compressible–incompressible turbulent flow simulation using a pressure-based algorithm, Computers & Fluids 37 (2008) 747–766. [29] Y.-Y. Tsui, T.-C. Wu, A Pressure-Based Unstructured-Grid Algorithm Using High-Resolution Schemes for All-Speed Flows, Numerical Heat Transfer, Part B: Fundamentals 53 (2008) 75–96.

25

[30] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, Journal of Computational Physics 228 (2009) 180–201. [31] M. Darwish, F. Moukalled, A fully coupled Navier-Stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444. [32] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. [33] F. Denner, Fully-coupled pressure-based algorithm for compressible flows: Linearisation and iterative solution strategies, Computers & Fluids 175 (2018) 53–65. [34] A. J. Chorin, Numerical solution of the Navier-Stokes equations, Mathematics of Computation 22 (1968) 745–745. [35] J. B. Bell, P. Colella, H. M. Glaz, A second-order projection method for the incompressible Navier-Stokes equations, Journal of Computational Physics 85 (1989) 257–283. [36] S. Patankar, D. Spalding, A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows, International Journal of Heat and Mass Transfer 15 (1972) 1787–1806. [37] S. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere Publishing Company, 1980. [38] R. Issa, Solution of the implicitly discretised fluid flow equations by operator-splitting, Journal of Computational Physics 62 (1985) 40–65. [39] R. Issa, A. Gosman, A. Watkins, The computation of compressible and incompressible recirculating flows by a non-iterative implicit scheme, Journal of Computational Physics 62 (1986) 66–82. [40] R. Kunz, W. Cope, S. Venkateswaran, Development of an implicit method for multi-fluid flow simulations, Journal of Computational Physics 152 (1999) 78–101. [41] B. van Wachem, V. Gopala, A coupled solver approach for multiphase flow calculations on collocated grids, in: European Conference on Computational Fluid Dynamics, ECCOMAS CFD, TU Delft, 2006, pp. 1–16. [42] B. van Wachem, A. Benavides, V. Gopala, A coupled solver approach for multiphase flow problems, in: 6th International Conference on Multiphase Flows 2007, Leipzig, Germany, p. Paper No 183. [43] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130. [44] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acousticallyconservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234. [45] R. M. Beam, R. F. Warming, An Implicit Factored Scheme for the Compressible Navier-Stokes Equations, AIAA Journal 16 (1978) 393–402. [46] R. W. MacCormack, A Numerical Method for Solving the Equations of Compressible Viscous Flow, AIAA Journal 20 (1982) 1275–1281. [47] E. Turkel, R. Radespiel, N. Kroll, Assessment of preconditioning methods for multidimensional aerodynamics, Computers & Fluids 26 (1997) 613–634. [48] S. M. H. Karimian, G. E. Schneider, Pressure-based computational method for compressible and incompressible flows, Journal of Thermophysics and Heat Transfer 8 (1994) 267–274. [49] E. Turkel, A. Fiterman, B. van Leer, Preconditioning and the Limit to the Incompressible Flow Equations, Technical Report, NASA CR-191500, 1993. [50] E. Turkel, Review of preconditioning methods for fluid dynamics, Applied Numerical Mathematics 12 (1993) 257–284. [51] S. Y. Kadioglu, M. Sussman, S. Osher, J. P. Wright, M. Kang, A second order primitive preconditioner for solving all speed multi-phase flows, Journal of Computational Physics 209 (2005) 477–503. [52] E. Turkel, Numerical Methods and Nature, Journal of Scientific Computing 28 (2006) 549–570. [53] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, Journal of Computational Physics 72 (1987) 277–298. [54] F. Xiao, Unified formulation for compressible and incompressible flows by using multi-integrated moments I: One-dimensional inviscid compressible flow, Journal of Computational Physics 195 (2004) 629–654. [55] K. Nerinckx, J. Vierendeels, E. Dick, Mach-uniformity through the coupled pressure and temperature correction algorithm, Journal of Computational Physics 206 (2005) 597–623. [56] F. Xiao, R. Akoh, S. Ii, Unified formulation for compressible and incompressible flows by using multi-integrated moments II: Multi-dimensional version for compressible and incompressible flows, Journal of Computational Physics 213 (2006) 31–56. [57] D. Fuster, S. Popinet, An all-Mach method for the simulation of bubble dynamics problems in the presence of surface tension, Journal of Computational Physics 374 (2018) 752–768. [58] M. V. Kraposhin, M. Banholzer, M. Pfitzner, I. K. Marchevsky, A hybrid pressure-based solver for nonideal single-phase fluid flows at all speeds, International Journal for Numerical Methods in Fluids 88 (2018) 79–99. [59] D. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Computers & Fluids 32 (2003) 1113–1132. [60] C.-D. Munz, S. Roller, R. Klein, K. Geratz, The extension of incompressible flow solvers to the weakly compressible regime, Computers & Fluids 32 (2003) 173–196. [61] J. H. Park, C.-D. Munz, Multiple pressure variables methods for fluid flow at all Mach numbers, International Journal for Numerical Methods in Fluids 49 (2005) 905–931. [62] F. Cordier, P. Degond, A. Kumbaro, An Asymptotic-Preserving all-speed scheme for the Euler and Navier–Stokes equations, Journal of Computational Physics 231 (2012) 5685–5704. [63] Y. Moguen, P. Bruel, E. Dick, A combined momentum-interpolation and advection upstream splitting pressure-correction algorithm for simulation of convective and acoustic transport at all levels of Mach number, Journal of Computational Physics 384 (2019) 16–41. [64] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. [65] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177–208. [66] O. Le M´etayer, J. Massoni, R. Saurel, ´Elaboration des lois d’´etat d’un liquide et de sa vapeur pour les mod`eles d’´ecoulements diphasiques, International Journal of Thermal Sciences 43 (2004) 265–276. [67] S. Karimian, A. Straatman, Discretization and parallel performance of an unstructured finite volume Navier–Stokes solver, International Journal for Numerical Methods in Fluids 52 (2006) 591–615. [68] F. Denner, B. van Wachem, Corrigendum to “Pressure-based algorithm for compressible interfacial flows with acousticallyconservative interface discretisation” [J. Comput. Phys. 367 (2018) 192–234], Journal of Computational Physics 381 (2019) 290–291.

26

[69] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, Journal of Computational Physics 298 (2015) 466–479. [70] P. Roe, Characteristic-based schemes for the euler equations, Annual Review of Fluid Mechanics 18 (1986) 337–365. [71] J. Ferziger, Interfacial transfer in Tryggvason’s method, International Journal for Numerical Methods in Fluids 41 (2003) 551–560. [72] I. Demirdˇzi´c, A Finite Volume Method for Computation of Fluid Flow in Complex Geometries, Ph.D. thesis, Imperial College London, 1982. [73] S. Mathur, J. Murthy, A pressure-based method for unstructured meshes, Numerical Heat Transfer Part B Fundamentals 31 (1997) 195–215. [74] Y.-Y. Tsui, Y.-F. Pan, A Pressure-Correction Method for Incompressible Flows Using Unstructured Meshes, Numerical Heat Transfer, Part B: Fundamentals 49 (2006) 43–65. [75] S. Balay, W. Gropp, L. C. McInnes, B. F. Smith, Efficient Management of Parallelism in Object Oriented Numerical Software Libraries, in: E. Arge, A. Bruasat, H. Langtangen (Eds.), Modern Software Tools in Scientific Computing, Birkh¨auser Press, 1997, pp. 163–202. [76] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. [77] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. [78] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. [79] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annual Research Briefs, Center for Turbulence (2004) 3–14. [80] P. K. Khosla, S. G. Rubin, A diagonally dominant second-order accurate implicit scheme, Computers & Fluids 2 (1974) 207–209. [81] J. Banks, T. Aslam, W. Rider, On sub-linear convergence for linearly degenerate waves in capturing schemes, Journal of Computational Physics 227 (2008) 6985–7002. [82] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure-velocity coupling allowing acoustic calculation in low Mach number flow, Journal of Computational Physics 231 (2012) 5522–5541. [83] J. D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw-Hill New York, 2003. [84] I. Johnston, The Noble-Abel Equation of State: Thermodynamic Derivations for Ballistics Modelling, Technical Report Technical Report DSTO–TN–0670, Defence Science and Technology Organisation, 2005. [85] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible multicomponent flows, Journal of Computational Physics 274 (2014) 95–121. [86] A. Harten, High Resolution Schemes for Hyperbolic Conservation Laws, Journal of Computational Physics 49 (1983) 357–393. [87] A. Harten, The artificial compression method for computation of shocks and contact discontinuities. I. Single conservation laws, Communications on Pure and Applied Mathematics 30 (1977) 611–638. [88] E. V. Vorozhtsov, N. N. Yanenko, Methods for the Localization of Singularities in Numerical Solutions of Gas Dynamics Problems, Springer Series in Computational Physics, Springer-Verlag, New York, 1990. [89] C. B. Laney, Computational Gasdynamics, Cambridge University Press, Cambridge; New York, NY, 1998. [90] T. Y. Hou, P. G. L. Floch, Why Nonconservative Schemes Converge to Wrong Solutions: Error Analysis, Mathematics of Computation 62 (1994) 497–530. [91] P. Lax, B. Wendroff, Systems of conservation laws, Communications on Pure and Applied Mathematics 13 (1960) 217–237. [92] H. S. G. Swann, The Convergence with Vanishing Viscosity of Nonstationary Navier-Stokes Flow to Ideal Flow in R3, Transactions of the American Mathematical Society 157 (1971) 373. [93] S. Osher, S. Chakravarthy, High resolution schemes and the entropy condition, SIAM Journal on Numerical Analysis 21 (1984) 955–984. [94] Y. Moguen, P. Bruel, E. Dick, Solving low Mach number Riemann problems by a momentum interpolation method, Journal of Computational Physics 298 (2015) 741–746. [95] G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, Journal of Computational Physics 27 (1978) 1–31. [96] M. Malik, J. Dey, M. Alam, Linear stability, transient energy growth, and the role of viscosity stratification in compressible plane Couette flow, Physical Review E 77 (2008). [97] U. Ghia, K. N. Ghia, C. T. Shin, High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method, Journal of Computational Physics 48 (1982) 387–411. [98] Z. Lilek, M. Peric, A fourth-order finite volume method with colocated variable arrangement, Computers and Fluids 24 (1995) 239–525. [99] P. Woodward, P. Colella, The Numerical Simulation of Two-Dimensional Fluid Flow with Strong Shocks, Journal of Computational Physics 173 (1984) 115–173. [100] H. Jasak, Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flow, Ph.D. thesis, Imperial College London, 1996. [101] S. I. Rubinow, J. B. Keller, The transverse force on a spinning sphere moving in a viscous fluid, Journal of Fluid Mechanics 11 (1961) 447–459.

Appendix A. Coefficients of the linear equation system

The coefficients of the discretised governing equations, Eqs. (50)-(52), are given below. In order to simplify the presentation, the coefficients are given based on the assumption that cell P is the upwind cell U of face f and using the BDF1 scheme for the discretisation of the transient terms. For the discretised continuity equation (50), the pressure coefficients associated with cell P and its neighbour

27

cells Q are

Aρ,p P = C VP � (γ −1) cv T (n) P + b (p(n) P + Π) � ∆t1 + �

f

� ˜ρ(n) f ˆdf ∆sf + C ϑ(n) f (γ −1) cv T (n) P + b (p(n) P + Π)

�

Af (A.1)

Aρ,p Q = �

f − ˜ρ(n) f ˆdf ∆sf Af, (A.2)

respectively. The velocity coefficients, which arise from the implicit treatment of the advecting velocity of the advection term, associated with cell P and its neighbour cells Q, are

Aρ,ui P = �

f ˜ρ(n) f (1 −lP f) ni,f Af (A.3)

Aρ,ui Q = �

f ˜ρ(n) f lP f ni,f Af, (A.4)

respectively. The coefficient of the right-hand side vector, σρ, associated with cell P is given as

σρ P =

�

ρ(t−∆t1) P −C Π

(γ −1) cv T (n) P + b (p(n) P + Π) −Iρ0

� VP ∆t1 + �

f

�

ϑ(n) f −rj,f

∂ui ∂xj

����

(n)

f ni,f

�

˜ρ(n) f Af

− �

f

 

 ˆdf



ρ∗(n) f



1 −lP f

ρ(n) P

∂p ∂xi

�����

(n)

p + lP f

ρ(n) Q

∂p ∂xi

�����

(n)

Q



si,f + ρ∗(t−∆t1) f

∆t1

� ϑ(t−∆t1) f −u(t−∆t1) i,f ni,f � 



 

˜ρ(n) f Af

− �

f

�

C

� Π −δf(p(n) P + Π)

(γ −1)cvT (n) P + b(p(n) P + Π) + δf(p(n) Q + Π)

(γ −1)cvT (n) Q + b(p(n) Q + Π)

�

+ I ρ0

�

ϑ(n) f Af,

(A.5)

where δf = ξf|rP f|/∆sf is the weighting coefficient that follows from the TVD discretisation of the advection term, see Section 4.3. For the discretised momentum equations (51), the pressure coefficients are given as

Aρuj,p P = C u(n) j,P VP � (γ −1) cv T (n) P + b (p(n) P + Π) � ∆t1

+ �

f

� ˜ρ(n) f ˜u(n) j,f ˆdf ∆sf + C ϑ(n) f ˜u(n) j,f (γ −1) cv T (n) P + b (p(n) P + Π) + (1 −lP f) nj,f

�

Af (A.6)

Aρuj,p Q = �

f

�

− ˜ρ(n) f ˜u(n) j,f ˆdf ∆sf + lP f nj,f

�

Af. (A.7)

The coefficients associated with velocity uj are given as

Aρuj,uj P = ρ(n) P VP ∆t1 + Dρuj,uj P (A.8)

Aρuj,uj Q = − �

f

αfµf

∆sf Af, (A.9)

where

Dρuj,uj P = �

f

� ˜ρ(n) f ϑ(n) f + αfµf

∆sf

� Af (A.10)

is the coefficient arising from the advection of velocity and the implicit velocity contribution of the decomposed shear stress term, which is used for the definition of the advection velocity ϑf, see Section 5.1. The coefficients of the velocity components that arise from the implicit treatment of the advecting velocity of the advection term are

Aρuj,ui P = �

f ˜ρ(n) f ˜u(n) j,f (1 −lP f) ni,f Af (A.11)

Aρuj,ui Q = �

f ˜ρ(n) f ˜u(n) j,f lP f ni,f Af. (A.12)

28

The coefficient of the right-hand side subvector σρuj follows as

σρuj P =

�

ρ(t−∆t1) P u(t−∆t1) j,P −C u(n) j,P Π

(γ −1) cv T (n) P + b (p(n) P + Π) −Iρ0u(n) j,P + ρ(n) P u(n) j,P

� VP ∆t1

− �

f

�

ϑ(n) f δf � u(n) j,Q −u(n) j,P � + ˜u(n) j,f rk,f

∂ui ∂xk

����

(n)

f ni,f

�

˜ρ(n) f Af

− �

f

 

 ˆdf



ρ∗(n) f



1 −lP f

ρ(n) P

∂p ∂xi

�����

(n)

p + lP f

ρ(n) Q

∂p ∂xi

�����

(n)

Q



si,f + ρ∗(t−∆t1) f

∆t1

� ϑ(t−∆t1) f −u(t−∆t1) i,f ni,f � 



 

˜ρ(n) f ˜u(n) j,f Af

− �

f

�

C

� Π −δf(p(n) P + Π)

(γ −1)cvT (n) P + b(p(n) P + Π) + δf(p(n) Q + Π)

(γ −1)cvT (n) Q + b(p(n) Q + Π)

�

+ I ρ0

�

ϑ(n) f ˜u(n) j,f Af

+ �

f

�

2˜ρ(n) f ϑ(n) f ˜u(n) j,f −ri,f

∂p ∂xi

����

(n)

f nj,f + µf

∂uj ∂xi

����

(n)

f (ni,f −αfsi,f) + µf

∂ui ∂xj

����

(n)

f ni,f −2

3 µf

∂uk ∂xk

����

(n)

f ni,f

�

Af.

(A.13)

The coefficients of the discretised energy equation (52) follow in a similar fashion, with the pressure coefficients given as

Aρh,p P =

�

C

�

ρ(n) P b + h(n) P (γ −1)cvT (n) P + b(p(n) P + Π)

�

−1

� VP ∆t1

+ �

f

� ˜ρ(n) f ˜h(n) f ˆdf ∆sf + C

�

˜ρ(n) f ϑ(n) f b + ϑ(n) f ˜h(n) f (γ −1)cvT (n) P + b(p(n) P + Π)

��

Af (A.14)

Aρh,p Q = �

f − ˜ρ(n) f ˜h(n) f ˆdf ∆sf Af, (A.15)

the velocity coefficients given as

Aρh,ui P = �

f (1 −lP f)

�

˜ρ(n) f ˜h(n) f ni,f −µf

� ∂uj ∂xi

����

(n)

f + ∂ui

∂xj

����

(n)

f −2

3

∂uk ∂xk

����

(n)

f nj,f

��

Af (A.16)

Aρh,ui Q = �

f lP f

�

˜ρ(n) f ˜h(n) f ni,f −µf

� ∂uj ∂xi

����

(n)

f + ∂ui

∂xj

����

(n)

f −2

3

∂uk ∂xk

����

(n)

f

�

nj,f

�

Af (A.17)

and the coefficients of the temperature given as

Aρh,T

P = cp



ρ(n) P VP ∆t1 + �

f ˜ρ(n) f ϑ(n) f Af



+ �

f

αf kf

∆sf Af (A.18)

Aρh,T

Q = − �

f

αf kf

∆sf Af. (A.19)

29

The coefficient of the right-hand side subvector σρh follows as

σρh P =

�

ρ(t−∆t1) P h(t−∆t1) P −ρ(n) P u(n),2 P

2 −C h(n) P Π

(γ −1) cv T (n) P + b (p(n) P + Π) −Iρ0h(n) P + ρ(n) P h(n) P −p(t−∆t1) P

� VP ∆t1

− �

f

�

ϑ(n) f u(n),2 P

2 + ϑ(n) f δf � h(n) Q −h(n) P � + ˜h(n) f rj,f

∂ui ∂xj

����

(n)

f ni,f

�

˜ρ(n) f Af

− �

f

 

 ˆdf



ρ∗(n) f



1 −lP f

ρ(n) P

∂p ∂xi

�����

(n)

p + lP f

ρ(n) Q

∂p ∂xi

�����

(n)

Q



si,f + ρ∗(t−∆t1) f

∆t1

� ϑ(t−∆t1) f −u(t−∆t1) i,f ni,f � 



 

˜ρ(n) f ˜h(n) f Af

− �

f

�

C

� Π −δf(p(n) P + Π)

(γ −1)cvT (n) P + b(p(n) P + Π) + δf(p(n) Q + Π)

(γ −1)cvT (n) Q + b(p(n) Q + Π)

�

+ I ρ0

�

ϑ(n) f ˜h(n) f Af

+ �

f

�

2˜ρ(n) f ϑ(n) f ˜h(n) f + kf

∂T ∂xi

����

(n)

f (ni,f −αfsi,f) + rl,f

∂ui ∂xl

����

(n)

f µf

� ∂uj ∂xi

����

(n)

f + ∂ui

∂xj

����

(n)

f −2

3

∂uk ∂xk

����

(n)

f

�

nj,f

�

Af.

(A.20)

30

