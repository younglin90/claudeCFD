# Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds 

Fabian Denner _[∗]_ , Fabien Evrard, Berend G.M. van Wachem 

_Chair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany_ 

## **Abstract** 

A conservative finite-volume framework, based on a collocated variable arrangement, for the simulation of flows at all speeds, applicable to incompressible, ideal-gas and real-gas fluids is proposed in conjunction with a fully-coupled pressure-based algorithm. The applied conservative discretisation and implementation of the governing conservation laws as well as the definition of the fluxes using a momentum-weighted interpolation are identical for incompressible and compressible fluids, and are suitable for complex geometries represented by unstructured meshes. Incompressible fluids are described by predefined constant fluid properties, while the properties of compressible fluids are described by the Noble-Abel-stiffened-gas model, with the definitions of density and specific static enthalpy of both incompressible and compressible fluids combined in a unified thermodynamic closure model. The discretised governing conservation laws are solved in a single linear system of equations for pressure, velocity and temperature. Together, the conservative finite-volume discretisation, the unified thermodynamic closure model and the pressure-based algorithm yield a conceptually simple, but versatile, numerical framework. The proposed numerical framework is validated thoroughly using a broad variety of test-cases, with Mach numbers ranging from 0 to 239, including viscous flows of incompressible fluids as well as the propagation of acoustic waves and transiently evolving supersonic flows with shock waves in ideal-gas and real-gas fluids. These results demonstrate the accuracy, robustness and the convergence, as well as the conservation of mass and energy, of the numerical framework for flows of incompressible and compressible fluids at all speeds, on structured and unstructured meshes. In particular, the precise recovery of a divergence-free velocity field in the incompressible limit, the accurate prediction of acoustic waves, and the convergence to the correct weak solution for strong shock waves with the same finite-volume discretisation and pressure-based algorithm are important features of the proposed numerical framework. 

_Keywords:_ Finite-volume methods, Pressure-based algorithms, Flows at all speeds, Compressible fluids, Incompressible fluids, Unstructured meshes 

_⃝_ c 2020. This manuscript version is made available under the CC-BY-NC-ND 4.0 license. http://creativecommons.org/licenses/by-nc-nd/4.0/ 

## **1. Introduction** 

Since the seminal work of Harlow and Amsden [1, 2], the formulation of numerical algorithms that can be applied for fluid flows at any speed is a central quest in _computational fluid dynamics_ (CFD). Yet, despite extensive research efforts over the past 50 years, the development of numerical methods and algorithms that are able to provide an accurate and robust prediction of the behaviour of fluids with different compressibility and of fluid flows at all speeds has proven difficult. Although the flow of any fluid and at any speed is described by the governing equations describing the conservation of mass, momentum and energy, different modelling assumptions with respect to the compressibility of the fluid and the different physical mechanisms dominating at different flow speeds yield dissimilar mathematical characteristics of the governing equations. This in turn leads to distinct and often contrasting numerical requirements. 

When developing numerical methods for flows at all speeds, it is important to recognise the numerical implications associated with the flow speed _U_ , represented by the Mach number _M_ = _U/a_ , where _a_ = �1 _/_ ( _ρβ_ s) is the speed of sound, and with the isentropic compressibility of the fluid, _β_ s = _{_ d _ρ/_ ( _ρ_ d _p_ ) _}_ s, that relates changes in pressure _p_ and density _ρ_ of a fluid at constant entropy. While pressure and density are strongly coupled for large flow speeds ( _M >_ 0 _._ 1), in particular for supersonic flows ( _M >_ 1), the pressure-density coupling 

> _∗_ Corresponding author: _Email address:_ `fabian.denner@ovgu.de` (Fabian Denner) 

1 

diminishes at low Mach numbers and vanishes for _M →_ 0, where d _ρ →_ 0. Founded on the observation that density changes are small at small speeds, a common assumption when modelling fluid flows is that the fluid is _incompressible_ , with a constant density (d _ρ_ = 0) along the fluid particle trajectories and, consequently, _β_ s = 0. Hence, pressure waves propagate with infinite speed ( _a →∞_ ) in incompressible fluids, contrary to compressible fluids where _β_ s _>_ 0 and 0 _< a < ∞_ . In fact, the convergence of solutions of the governing equations of the flow of compressible fluids to the governing equations of the flow of incompressible fluids for _M →_ 0 has been proven rigorously by Klainerman and Majda [3] and Hoff [4]. In addition to the governing conservation laws, compressible fluids require a thermodynamic closure model that describes the relationship between density, pressure and energy. The ideal-gas model represents the most simple and most widely used thermodynamic closure model, with _p ∝_ ( _ρ, T_ ), where _T_ is the temperature. More complex formulations, so-called _real-gas_ models, further include the effects of intermolecular repulsion [5], intermolecular attraction [6, 7] or both [8, 9], or other material properties, _e.g._ the acentric factor describing the shape of the molecules [10, 11]. For an incompressible fluid, however, no closure model is required, since the density is not coupled to pressure, and an isothermal flow of an incompressible fluid is fully described only by the momentum and continuity equations, _i.e._ the energy equation becomes redundant. 

The challenge in developing numerical frameworks that are applicable to incompressible fluids and compressible fluids at all flow speeds is, therefore, to construct a numerical method that combines a unified thermodynamic closure model, a uniform set of interpolation functions, a consistent handling of the incompressible limit, shock capturing capabilities, a method to advect the solution that is applicable in all speed regimes, as well as a set of solution variables that are physically meaningful for incompressible and compressible fluids [12]. 

The choice of solution variables is of particular importance in constructing a numerical method that is applicable to flows at all speeds, since a unified algorithm is predicated on a single set of solution variables [12]. Choosing the conserved variables, _i.e._ density, momentum and total energy, as solution variables for the continuity, momentum and energy equations, respectively, is desirable for compressible fluids at sufficiently large speeds ( _M >_ 0 _._ 1). However, the continuity equation is not effective as a transport equation for density in the incompressible limit, because d _ρ →_ 0, and, instead, becomes a constraint on the velocity field with _**∇** ·_ _**u** →_ 0 [13]. An attractive choice of the solution variables for numerical algorithms applicable to predict flows at all speeds is, therefore, the primitive variables including pressure [6, 12, 14, 15], _i.e._ pressure, velocity and temperature. Using pressure as a solution variable, the vanishing density differences in the incompressible limit do not pose a problem and the pressure acts as a Lagrange multiplier that enforces _**∇** ·_ _**u** →_ 0 [16–18]. Conveniently, choosing primitive variables as solution variables still allows to discretise the governing equations in conservative form [19]. In practice, however, achieving accurate conservation of mass and energy, constructing robust shock capturing schemes and ensuring a stable numerical solution in the transonic regime has proven difficult in the context of primitive variables [20, 21]. It is, therefore, convenient to develop numerical algorithms _either_ for incompressible fluids _or_ for compressible fluids, which has led to two primary classes of algorithms: pressure-based algorithms and density-based algorithms. 

_Pressure-based_ algorithms, in which the continuity equation serves as an equation for pressure, while density is constant (incompressible fluid) or evaluated explicitly using an equation of state (compressible fluid), may be used to predict flows at all speeds, see _e.g._ [14, 15, 22–33]. For both incompressible and compressible fluids, the majority of pressure-based algorithms are founded on pressure-correction methods, such as projection methods [34, 35], the SIMPLE method [36, 37] and its subsequent derivatives, or the PISO method [38, 39]. However, the weak coupling between density, pressure, velocity and energy of the discretised governing equations as a result of the iterative predictor-corrector solution procedure, which usually necessitates underrelaxation of the discretised equations to reach a converged solution, is a key shortcoming of segregated methods [31, 40]. This has motivated the development of coupled algorithms, where the discretised governing equations are solved in a single linear system of equations, for both incompressible fluids [30, 32, 41, 42] and compressible fluids [14, 15, 24, 31, 33, 43], showing great potential in terms of versatility, robustness and performance in all speed regimes. For instance, Darwish et al. [30] demonstrated substantial performance benefits for incompressible flows compared to pressure-correction methods and Denner et al. [44] reported robust results for flows with Mach numbers ranging from 0 _._ 001 to 100 with a fully-coupled pressure-based algorithm. 

Contemporary numerical methods for the simulation of compressible flows are typically predicated on _densitybased_ algorithms, _e.g._ [45–47], where the conserved quantities are chosen as solution variables for the governing conservation equations and, in particular, the continuity equation serves as an equation for density. While density-based algorithms are naturally suited for compressible flows, they are poorly suited for low-Mach number flows [21, 48], where the coupling of pressure and density vanishes. Although density-based algorithms have been applied to low-Mach number flows with some success, this requires pre-conditioning techniques [47, 49–52] that are computationally very expensive, especially for transient problems, and the success of which is typically determined, at least in parts, by predefined constants [49, 53]. In order to improve the performance for low Mach number flows, recent work has been focusing on combining density-based methods with segregated pressurecorrection algorithms [54–58] and/or reformulating the energy equation as an equation for pressure [55, 59–62]. These density-based algorithms have been applied successfully to a wide range of flows, including flows ranging 

2 

from an incompressible flow to the propagation of strong shock waves, stationary and high-speed discontinuous waves as well as the propagation of linear acoustic waves [54, 56, 59, 63]. 

An important aspect for the design of numerical frameworks for fluid flows at all speeds is that pressure plays an important role in all Mach number regimes [22, 64]; pressure changes are, contrary to density changes, always finite. Exploiting this versatile role of pressure by including pressure as a primary solution variable in the numerical framework, thus, provides a seemingly distinct advantage for applications in all Mach number regimes: it provides a solution variable, _i.e._ pressure, which is meaningful in all Mach number regimes and does not require particular pre-conditioning techniques. This is further supported by the analysis of Hauke and Hughes [12], who identified the primitive variables (pressure, velocity and temperature) as particularly suitable solution variables to predict flows at all speeds. Remarkably, all of the numerical methods that stand out with respect to modelling fluid flows at all speeds, due to their versatility and robustness, incorporate the unique role of pressure, albeit in different ways. In pressure-based algorithms, the special role of pressure can be taken into account through an appropriate linearisation of the discretised continuity equation [19, 22, 33, 48]: for compressible flows, the continuity equations serves as a transport equation for density, with density formulated as a function of pressure by an equation of state, whereas for incompressible flows, the continuity equation serves as a constraint on the divergence of the velocity field [43], with pressure acting as a Lagrange multiplier. The extension of density-based algorithms to low Mach numbers, either by introducing a pressure-Poisson equation [54, 56] or by reformulating the energy equation as an equation for pressure [55, 59, 60], provides a pressure-velocity coupling at low speeds and enforces a divergence-free velocity field in the incompressible limit. However, despite the broad variety of numerical methods able to simulate flows at all speeds, a numerical framework based on a unified conservative discretisation that is able to incorporate incompressible fluids as well as ideal-gas and real-gas compressible fluids, and which can predict flows at all speeds accurately and robustly, including low-Mach acoustics, Riemann problems and multidimensional flows ranging from the incompressible limit to supersonic flow, has not been presented in the literature yet. 

In this article, a conservative, collocated, finite-volume framework in combination with a fully-coupled pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all Mach numbers is proposed. The governing equations describing the conservation of continuity, momentum and energy are discretised using standard finite-volume methods and are solved for pressure, velocity and temperature in a single linear system of equations. Incompressible fluids are described by predefined constant fluid properties, while compressible fluids are described by the Noble-Abel-stiffened-gas model [9], with the definitions of density and specific static enthalpy of both incompressible and compressible fluids combined in a unified thermodynamic closure model. This enables the design of a conceptually simple, but versatile, numerical algorithm that is able to predict flows of incompressible fluids as well as flows of compressible fluids at all speeds. The conservative discretisation and implementation of the governing equations are identical for incompressible and compressible fluids, employing a single definition of the fluxes based on a momentum-weighted interpolation [65]. A broad variety of representative test-cases featuring flows of incompressible and compressible fluids in all Mach number regimes are considered to scrutinise and validate the proposed numerical framework: the propagation of acoustic waves, contact discontinuities and shock waves, shock tubes in different Mach number regimes, Taylor vortices in an inviscid fluid, diffusion-dominated problems, a lid-driven cavity, supersonic flow over a forward-facing step, and Stokes flow around a rotating sphere. The presented results demonstrate the accuracy and robustness, as well as the conservation and convergence properties, of the numerical framework for all flow speeds on structured and unstructured meshes. In particular, the precise recovery of a divergence-free velocity field for _M →_ 0, the accurate prediction of acoustic waves and the convergence to the correct weak solution for _M ≫_ 1 are important features of the proposed numerical framework. As such, the proposed numerical framework stands out for the simplicity of its discretisation in conjunction with the broad range of flows that can be predicted accurately and robustly. 

The governing equations are introduced in Section 2. Subsequently, the three main building blocks of the proposed finite-volume method are presented: a unified thermodynamic closure model in Section 3, the finitevolume discretisation in Section 4, and the pressure-based algorithm used to solve the discretised governing equations in Section 5. The results of representative test-cases are presented and discussed in Section 6. The article is summarised and concluded in Section 7. 

## **2. Governing equations** 

The conservation laws governing fluid flows at all speeds, applicable to both incompressible and compressible flows, formulated in a Cartesian coordinate system, are the conservation of mass 

**==> picture [275 x 23] intentionally omitted <==**

the conservation of momentum 

**==> picture [312 x 25] intentionally omitted <==**

3 

and the conservation of energy 

**==> picture [332 x 24] intentionally omitted <==**

where _t_ is time, _**u**_ is the velocity vector, _p_ is pressure, _ρ_ is the density of the fluid and _h_ = _h_ s + _**u**_[2] _/_ 2 is the specific total enthalpy, with _h_ s the specific static enthalpy. The stress tensor _**τ**_ for the considered Newtonian fluids is given as 

**==> picture [319 x 26] intentionally omitted <==**

where _µ_ is the dynamic viscosity of the fluid. Heat conduction is modelled by Fourier’s law, 

**==> picture [269 x 23] intentionally omitted <==**

where _k_ is the thermal conductivity of the fluid and _T_ is the temperature. 

The enthalpy formulation is chosen for the energy equation, rather than the more common internal energy formulation, because it leads to a straightforward application in the numerical algorithm, since the transient pressure term on the right-hand side of Eq. (3) does not require linearisation [33, 58]. The governing conservation laws require closure through an appropriate model that defines the thermodynamic properties (see Section 3). 

## **3. Thermodynamic closures** 

In order to close the governing conservation laws presented in Section 2, the thermodynamic properties of the fluid have to be defined. In the proposed formulation, this is achieved by defining the density _ρ_ and the specific static enthalpy 

**==> picture [274 x 11] intentionally omitted <==**

where _cp_ is the specific isobaric heat capacity and _e[∗]_ is the specific residual energy, through a set of input quantities ( _ρ_ 0, _cv_ , _cp_ , Π, _b_ ). This approach enables the formulation of a unified thermodynamic closure for incompressible, ideal-gas and real-gas fluids, which facilitates a straightforward finite-volume discretisation that is applicable for incompressible flows as well as compressible flows in all Mach number regimes. 

An incompressible fluid is characterised by a constant density, with d _ρ_ = 0, defined as 

**==> picture [257 x 11] intentionally omitted <==**

The specific isobaric heat capacity _cp_ is assumed to be constant for incompressible fluids and the specific residual energy is _e[∗]_ = 0. The speed of sound for an incompressible fluid is given as 

**==> picture [286 x 31] intentionally omitted <==**

with subscript _s_ denoting constant entropy. 

The Noble-Abel-stiffened-gas (NASG) model, originally proposed by Le M´etayer and Saurel [9], is chosen to represent ideal and real gases. The NASG model is a combination of the stiffened-gas model [6, 66] and the Noble-Abel-gas model (also called co-volume gas model) [5], with the motivation of defining a simple gas model that accounts for molecular attraction and repulsion. The thermal and caloric equations of state of the NASG model are given as [9] 

**==> picture [308 x 21] intentionally omitted <==**

**==> picture [305 x 19] intentionally omitted <==**

respectively, where _γ_ = _cp/cv_ is the heat capacity ratio, _cv_ is the specific isochoric heat capacity, _v_ = 1 _/ρ_ is the specific volume, _e_ is the specific internal energy and _e_ 0 is the specific reference energy. The pressure constant Π represents attraction between molecules and is typically relevant for condensed phases, _e.g._ to model liquids, while the co-volume _b_ accounts for the volume occupied by the individual molecules of the fluid. The density is given by rearranging Eq. (9) as 

**==> picture [304 x 24] intentionally omitted <==**

and the specific total enthalpy, _h_ = _h_ s + _**u**_[2] _/_ 2, follows from Eqs. (6), (9) and (10) as 

**==> picture [285 x 23] intentionally omitted <==**

4 

with specific residual energy 

_e[∗]_ = _b p_ + _e_ 0 _._ (13) 

In the following, the specific reference energy is assumed to be _e_ 0 = 0, because only single-phase flows without phase transition and reactions are considered. The specific heat capacities _cv_ and _cp_ are constant and the speed of sound is [9] 

**==> picture [284 x 31] intentionally omitted <==**

Contrary to the van der Waals gas model, which also accounts for molecular attraction and repulsion, the coefficients Π and _b_ are constant; the NASG model thus represents these molecular interactions in the simplest possible form. Furthermore, the NASG model is, unlike, for instance, the van der Waals gas model, unconditionally convex. With respect to liquids, such as water, the NASG model resolves the inaccuracy of specific heat capacities resulting from applying the classical stiffened-gas model [9]. The NASG model reduces to the ideal-gas (IG) model for Π = 0 and _b_ = 0, to the Noble-Abel (NA) gas model for Π = 0 and _b >_ 0, and to the stiffened-gas (SG) model for Π _>_ 0 and _b_ = 0. 

In order to incorporate incompressible and compressible fluids in the same numerical framework, the definitions for the density _ρ_ and the specific residual energy _e[∗]_ are unified by the binary operators _C_ and _I_ = 1 _−C_ . The binary operator _C_ , given as 

**==> picture [321 x 31] intentionally omitted <==**

is used as a coefficient for the compressible part and, analogously, the binary operator _I_ is used as a coefficient for the incompressible part of the unified closure model. The density of the fluid is then defined, based on Eqs. (7) and (11), as 

**==> picture [329 x 25] intentionally omitted <==**

and the specific residual energy is given based on Eq. (13), and assuming _e_ 0 = 0, as 

**==> picture [264 x 11] intentionally omitted <==**

The type of fluid considered in a simulation can be simply specified through the binary operator _C_ , without changes to the thermodynamic closure model or the discretisation of the governing equations. An incompressible fluid ( _C_ = 0) is, thereby, fully defined by setting _ρ_ 0 and _cp_ , while a compressible fluid ( _C_ = 1) is defined by setting _cv_ , _cp_ , Π and _b_ . 

## **4. Finite-volume discretisation** 

The proposed numerical framework is founded on a collocated finite-volume discretisation, which is based on the integral formulation of the governing conservation laws, for unstructured meshes. Taking the convectiondiffusion equation for the transport of a general flow variable, _φ_ , as an example, given as 

**==> picture [311 x 25] intentionally omitted <==**

where Γ _φ_ is the diffusion coefficient of _φ_ , its integral form with respect to control volume _V_ is given as 

**==> picture [365 x 25] intentionally omitted <==**

The discretisation of each individual term is discussed in the following. 

## _4.1. Gradient evaluation_ 

The spatial gradient at cell centre _P_ is evaluated using the divergence theorem, given as 

**==> picture [299 x 29] intentionally omitted <==**

where _f_ denotes the faces bounding cell _P_ , _VP_ is the volume of cell _P_ , _**n** f_ is the normal vector of face _f_ pointing outwards with respect to cell _P_ , and _Af_ is the area of face _f_ . The face value _φf_ is interpolated from the adjacent cell centres _P_ and _Q_ , schematically illustrated in Fig. 1a, as 

**==> picture [331 x 28] intentionally omitted <==**

5 

**==> picture [307 x 130] intentionally omitted <==**

**----- Start of picture text -----**<br>
u<br>f n f<br>P r f s f D<br>f [′]<br>f<br>Q<br>U<br>(a) General discretisation (b) TVD differencing<br>**----- End of picture text -----**<br>


Figure 1: Schematic illustration of (a) cell _P_ with its neighbour cell _Q_ and the shared face _f_ , where _**n** f_ is the unit normal vector of face _f_ and _**s** f_ is the unit vector connecting cell centres _P_ and _Q_ (both outward pointing with respect to cell _P_ ), with _f[′]_ the interpolation point associated with face _f_ and _**r** f_ the vector from interpolation point _f[′]_ to face centre _f_ , and (b) upwind cell _U_ and downwind cell _D_ of face _f_ , where _**u**_ represents the velocity vector. 

where _lP f_ is the inverse-distance weighting coefficient, 

**==> picture [269 x 25] intentionally omitted <==**

with ∆ _sf_ the distance between cell centres _P_ and _Q_ , and _**r** P f_ is the vector connecting cell centre _P_ with face interpolation point _f[′]_ . A formally second-order accurate gradient-based correction of mesh-skewness [25, 67] is included in Eq. (21), with _**r** f_ the vector connecting the interpolation point _f[′]_ of the face with face centre _f_ on meshes with skewness, see Fig. 1a. 

## _4.2. Transient terms_ 

The First-Order Backward Euler scheme, also widely known as BDF1 scheme, and the Second-Order Backward Euler scheme, also widely known as BDF2 scheme, are used to discretise the transient terms of the governing flow equations. The transient term of the transport equation (19), with Φ = _ρφ_ , is given for cell _P_ discretised with the First-Order Backward Euler scheme as 

**==> picture [337 x 28] intentionally omitted <==**

and discretised with the Second-Order Backward Euler scheme as [68] 

**==> picture [463 x 25] intentionally omitted <==**

with ∆ _τ_ = ∆ _t_ 1 + ∆ _t_ 2, where ∆ _t_ 1 is the current time-step, ∆ _t_ 2 is the previous time-step, superscript ( _t −_ ∆ _t_ 1) denotes values of the previous time-level and superscript ( _t −_ ∆ _τ_ ) denotes values of the previous-previous time-level. If the time-step is constant, with ∆ _t_ 1 = ∆ _t_ 2, the transient term of Eq. (19) discretised with the Second-Order Backward Euler scheme simplifies to the more familiar form 

**==> picture [368 x 27] intentionally omitted <==**

For consistency, all transient terms of the governing equations (1)-(3) are discretised with the same scheme [33]. 

## _4.3. Advection terms_ 

Applying the divergence theorem, the advection term of Eq. (19) is given as 

**==> picture [312 x 24] intentionally omitted <==**

where _**S**_ is the outward-pointing surface vector on the surface _∂V_ of control volume _V_ . Assuming the surface of the control volume has a finite number of flat faces _f_ with area _Af_ , and applying the midpoint rule [16, 64], the advection term follows in semi-discretised form as 

**==> picture [310 x 28] intentionally omitted <==**

6 

where _ϑf_ = _**u** f ·_ _**n** f_ is the advecting velocity at face _f_ , which will be discussed in detail in Section 5.1. The advected variable _φ_[˜] _f_ and the density _ρ_ ˜ _f_ are interpolated using a TVD interpolation for three-dimensional unstructured meshes with an implicit correction of mesh skewness [69], given as 

**==> picture [310 x 25] intentionally omitted <==**

where subscripts _U_ and _D_ denote the upwind and downwind cells, as illustrated in Fig. 1b, _ξf_ is the flux limiter and _**r** Uf_ is the vector connecting the cell centre of the upwind cell _U_ with face interpolation point _f[′]_ . A detailed description of the implementation of this TVD interpolation using common TVD schemes on skewed and non-equidistant meshes can be found in [69]. In this study, the first-order upwind scheme, _ξf_ = 0, the central differencing scheme, _ξf_ = 1, and the Minmod scheme [70], _ξf_ ( _gf_ ) = max(0 _,_ min(1 _, gf_ )), where _gf_ is the ratio of the upwind and downwind gradients of _φ_ [69], are considered. 

## _4.4. Diffusion terms_ 

Applying the divergence theorem and the midpoint rule, the diffusion term of the transport equation (19) is given as 

**==> picture [348 x 29] intentionally omitted <==**

Following Ferziger [71], the diffusion coefficient Γ _φ_ at face _f_ is defined as 

**==> picture [295 x 25] intentionally omitted <==**

Considering an orthogonal mesh, where the unit normal vector _**n** f_ of face _f_ and the unit vector _**s** f_ connecting the adjacent cell centres _P_ and _Q_ are parallel, with _**n** f_ = _**s** f_ , the face-centred gradient is approximated with second-order accuracy as 

**==> picture [289 x 27] intentionally omitted <==**

The decomposition and deferred correction approach of Demirdˇzi´c [72] is applied to correct for non-orthogonality of the mesh, as illustrated in Fig. 1a, with the face-centred gradient defined as [25] 

**==> picture [347 x 28] intentionally omitted <==**

The scaling factor _αf_ = ( _**n** f ·_ _**s** f_ ) _[−]_[1] ensures a robust convergence even for large non-orthogonality of the mesh [73, 74]. Equation (32) reduces to Eq. (31) for an orthogonal mesh with _**n** f_ = _**s** f_ . 

## **5. Pressure-based algorithm** 

A finite-volume framework with a pressure-based algorithm for the prediction of flows of incompressible fluids and compressible fluids at all speeds is proposed. To this end, the governing equations (1)-(3) are closed by the thermodynamic closure model and discretised using the finite-volume discretisation presented in Sections 3 and 4, respectively. Once discretised and linearised as detailed below, the governing equations are solved simultaneously in a single linear system of equations, _**Aψ**_ = _**σ**_ , for the pressure _p_ , the velocity vector _**u** ≡_ ( _u, v, w_ ) _[T]_ and the temperature _T_ . For a three-dimensional computational mesh with _N_ cells, the linear system of governing equations is given as 

**==> picture [383 x 60] intentionally omitted <==**

where _**A**[ζ,χ]_ , with _ζ_ the conserved quantity and _χ_ the primary solution variable of a given governing equation, are the coefficient submatrices of size _N × N_ of the continuity equation (44) for _ζ_ = _ρ_ , the momentum equations (46) for _ζ ∈{ρu, ρv, ρw}_ , and the energy equation (47) for _ζ_ = _ρh_ . The subvectors _**ψ**[χ]_ of length _N_ hold the solution for primary solution variable _χ_ and the subvectors _**σ**[ζ]_ of length _N_ hold all contributions from previous nonlinear iterations and previous time-levels. 

The solution procedure performs nonlinear iterations in which the linear system of governing equations (33) is solved using the _Block-Jacobi_ preconditioner and the _BiCGSTAB_ solver of the software library PETSc [75–77] until the residual of (33) satisfies _∥_ _**Aψ** −_ _**σ** ∥ < η ∥_ _**σ** ∥_ , where _η_ is the predefined solution tolerance and _∥· ∥_ denotes the _L_ 2-norm, as presented and tested in detail by Denner [33]. 

7 

## _5.1. Advecting velocity_ 

In the proposed numerical framework, the advecting velocity _ϑf_ = _**u** f ·_ _**n** f_ is based on a momentum-weighted interpolation (MWI), originally introduced by Rhie and Chow [78], and serves to advect the conserved quantities _ζ_ = _{ρ, ρ_ _**u** , ρh}_ . Furthermore, for flows of incompressible fluids and low Mach number flows of compressible fluids, the advecting velocity allows to solve the continuity equation for pressure (see Section 5.4) and prevents pressure-velocity decoupling associated with the collocated variable arrangement [16, 65]. 

Following the work of Bartholomew et al. [65], the advecting velocity _ϑf_ at face _f_ is given as 

**==> picture [481 x 44] intentionally omitted <==**

where the interpolated face velocities _**u** f_ and _**u**_ ~~[(]~~ _f[t][−]_[∆] _[t]_[1][)] are obtained by linear interpolation, and _lP f_ is given by Eq. (22). As derived and discussed in detail by Bartholomew et al. [65], the coefficient _d_[ˆ] _f_ is defined as 

**==> picture [304 x 52] intentionally omitted <==**

where _SP_ =[�][3] _j_ =1 _[D] P[ρu][j][,u][j]_ and _SQ_ =[�][3] _j_ =1 _[D] Q[ρu][j][,u][j]_ are the sum of the diagonal matrix coefficients of the velocity arising from the advection and shear stress terms of the discretised momentum equations, see Eq. (A.10) in Appendix A. The face density is defined as 

**==> picture [287 x 26] intentionally omitted <==**

The MWI provides a robust pressure-velocity coupling for incompressible flows by introducing a cell-tocell pressure coupling and applying a low-pass filter acting on the third derivative of pressure [16, 21, 25, 65], thus avoiding pressure-velocity decoupling due to the collocated variable arrangement. The transient term of Eq. (34) ensures a time-step independent contribution of the MWI in conjunction with the coefficient _d_[ˆ] _f_ [65] and is important for a correct temporal evolution of pressure waves [43, 65]. However, the MWI is known to introduce numerical dissipation that manifests in an unphysical dissipation of kinetic energy [65, 79], a conservation error that converges with ∆ _x_[3] and that is, assuming the consistent formulation given by Eq. (34), independent of the applied time-step [65]. 

## _5.2. Discretised governing equations_ 

Applying the finite-volume methods described in Section 4 and, in particular, using the BDF1 scheme for the transient term in the interest of clarity, the discretised continuity equation (1) for cell _P_ is given as 

**==> picture [321 x 32] intentionally omitted <==**

Similar to the discretisation of the continuity equation, applying the finite-volume scheme presented in Section 4, the discretised momentum equations (2) in cell _P_ are given as 

**==> picture [388 x 69] intentionally omitted <==**

where the viscosity _µf_ at face _f_ is defined by Eq. (30). In order to account for mesh non-orthogonality, the deferred correction approach given in Eq. (32) is applied to decompose the shear-stress term as 

**==> picture [412 x 31] intentionally omitted <==**

8 

The discretised energy equation (3) in cell _P_ , using the applied finite-volume discretisation, is given as 

**==> picture [427 x 67] intentionally omitted <==**

where the heat conduction term is decomposed as described by Eq. (32) and the thermal conductivity _kf_ at face _f_ is defined by Eq. (30). 

## _5.3. Linearisation and implementation_ 

The details of the linearisation of the governing equations have been shown to be a critical aspect for all-Mach formulations and algorithms [19, 33, 40, 48] and provides additional potential with respect to the performance of fully-coupled algorithms [33]. To this end, a Newton linearisation is applied to facilitate an implicit treatment of all dominant pressure, velocity and temperature terms in the linear system resulting from the linearisation and discretisation of the governing equations (1)-(3), given for two generic fluid variables as 

**==> picture [370 x 14] intentionally omitted <==**

or for three generic fluid variables as 

**==> picture [467 x 14] intentionally omitted <==**

where _n_ is the iteration counter associated with the nonlinear iterations performed to solve the system of discretised governing equations, Eq. (33), at each time-step. Superscript ( _n_ ) denotes the most recent available solution, which is the solution of the previous time-step during the first nonlinear iteration of a given time-step or, otherwise, the solution of the previous nonlinear iteration, and superscript ( _n_ + 1) denotes the solution that is sought implicitly. 

Applying the Newton linearisation given in Eq. (41) to the advection term and formulating the cell-centered density _ρP_ of the transient term as a semi-implicit function of pressure _pP_ , given as 

**==> picture [352 x 30] intentionally omitted <==**

the discretised continuity equation (37) follows as 

**==> picture [400 x 32] intentionally omitted <==**

Following previous studies [33, 44], the advecting velocity _ϑ_[(] _f[n]_[+1)] is defined by a semi-implicit formulation as 

**==> picture [438 x 67] intentionally omitted <==**

Linearising the transient terms and the advection terms with the Newton linearisation given in Eqs. (41) and (42), respectively, following the work of Denner [33], and treating cell-centered pressure and velocity contributions implicitly, the discretised momentum equations (38) follow as 

**==> picture [464 x 112] intentionally omitted <==**

and the discretised energy equation (40) becomes 

**==> picture [424 x 129] intentionally omitted <==**

with _ρ_[(] _P[n]_[+1)] given by Eq. (43) and _ϑ_[(] _f[n]_[+1)] given by Eq. (45). The implicitly computed specific total enthalpy _h_[(] _P[n]_[+1)] at cell-centre _P_ is formulated, following Eq. (12) and assuming _e_ 0 = 0, as an implicit function of temperature _T_ and pressure _p_ , given as 

**==> picture [329 x 25] intentionally omitted <==**

This treatment enables the implicit solution of the energy equation for temperature, pressure and velocity, which allows to solve the cell-centred values of temperature of the heat conduction term implicitly, see Eq. (47), and, thus, time-step restrictions associated with an explicit treatment of the heat conduction term [37] do not apply for the presented algorithm. 

Following the work of Khosla and Rubin [80], the TVD interpolation of advected variables, see Eq. (28), is implemented using a deferred correction approach, given as 

**==> picture [330 x 24] intentionally omitted <==**

where the upwind contribution is treated implicitly and the high-order correction is based on the values of the previous nonlinear iteration. This interpolation is unconditionally stable [21, 64, 80], which is essential for the simulation of convection-dominated flows with Peclet numbers of Pe = _ρ|_ _**u** |_ ∆ _x/µ ≫_ 1 and, in particular, inviscid flows (Pe _→∞_ ). 

The coefficients of the linear equation system _**Aψ**_ = _**σ**_ , Eq. (33), for cell _P_ follow after rearranging the discretised and linearised governing equations (44), (46) and (47) as 

**==> picture [470 x 53] intentionally omitted <==**

respectively, with _Q_ the neighbour cells of cell _P_ . The individual coefficients _A_ and right-hand side contributions _σ_ are given in Appendix A. 

The strong implicit coupling of pressure, density and velocity through a Newton linearisation has been shown to be beneficial for the performance and stability of the solution algorithm in all Mach number regimes [33]. For instance, the Newton linearisation of the advection term of the continuity equation (44) facilitates a smooth transition from low to high Mach number regions [19, 33, 48], with the term[�] _f[ρ]_[˜][(] _f[n]_[)] _[ϑ]_[(] _f[n]_[+1)] _Af_ of Eq. (44) dominant at low Mach numbers and the term[�] _f[ρ]_[˜][(] _f[n]_[+1)] _ϑ_[(] _f[n]_[)] _[A][f]_[dominant in regions of high Mach numbers [][43][].] As a result, the Newton linearisation of the advection term also yields performance and stability benefits for flows with sharp changes in Mach number and strong compressibility [33], and provides the necessary implicit pressure-velocity coupling for incompressible flows [32, 43]. 

## _5.4. Incompressible limit_ 

The incompressible limit deserves special attention, as this is the _Achilles’ heel_ of many previously proposed numerical frameworks for flows at all speeds. From a numerical viewpoint, the _incompressible limit_ includes both the flow of compressible fluids with very small Mach numbers ( _M →_ 0) and the flow of incompressible fluids ( _ρ_ = const.). As density changes of the fluid particles vanish in the incompressible limit, with d _ρ →_ 0, the density is constant along the fluid particle trajectories [13], with 

**==> picture [292 x 24] intentionally omitted <==**

10 

Inserting Eq. (53) into the governing equations (1)-(3) yields 

**==> picture [342 x 80] intentionally omitted <==**

for the continuity, momentum and energy equations, respectively, in the incompressible limit. 

Applying the discretisation and linearisation schemes presented in the previous sections to the governing equations in the incompressible limit, Eqs. (54)-(56), the discretised continuity equation follows as 

**==> picture [281 x 25] intentionally omitted <==**

the discretised momentum equations are given as 

**==> picture [464 x 72] intentionally omitted <==**

and the discretised energy equation follows as 

**==> picture [425 x 110] intentionally omitted <==**

The definition of the semi-implicit advecting velocity _ϑ_[(] _f[n]_[+1)] , with the implicit treatment of the cell-centred pressure values, as defined in Eq. (45), yields a consistent discretisation of Eq. (54) as a function of pressure. This allows pressure to enforce a divergence-free velocity field in the incompressible limit, as well as a robust implicit pressure-velocity coupling for the collocated variable arrangement. Furthermore, Eqs. (57)-(59) treat all the terms implicitly which Nerinckx et al. [55] identified to carry acoustic information, thereby eliminating the acoustic time-step restriction and enabling an efficient solution for _M →_ 0 and, specifically, for _M_ = 0. In fact, Eqs. (57) and (58) are identical to the discretised continuity and momentum equations of the fullycoupled pressure-based algorithm for incompressible interfacial flows of Denner and van Wachem [32]. Thus, the discretised governing equations presented in Section 5.2 represent the incompressible limit accurately and facilitate the simulation of incompressible flows. If _isothermal_ incompressible fluids are considered, the energy equation may be disregarded, removing Eq. (47) from Eq. (33), although this simplification is not taken into account in the results presented in Section 6. 

## **6. Validation** 

The results for a broad variety of test-cases are presented here to scrutinise each aspect of the thermodynamic closure, the finite-volume discretisation and the fully-coupled pressure-based algorithm, including the convergence and conservation properties. In Section 6.1, the propagation of acoustic waves is considered to test the accurate prediction of acoustic effects for both ideal-gas and real-gas fluids, in particular the amplitude of pressure waves and the speed of sound. The propagation of a moving contact discontinuity is considered in Section 6.2 to test the convergence under mesh refinement for linearly degenerate waves, a distinct challenge for finite-volume methods [81]. In Section 6.3, the propagation of a strong shock wave with Mach number 100 is considered to check if the proposed finite-volume framework converges to the correct weak solution of the governing equations, for both ideal-gas and real-gas fluids. Shock tubes with flows in different Mach number 

11 

Table 1: Fluid properties considered for the propagation of acoustic waves. 

|Fluid|_γ_|_cp_[J kg_−_1 K_−_1]|_b_ [m3 kg_−_1]|Π[Pa]|
|---|---|---|---|---|
|Air|1_._400|1008|0|0|
|JA2 propellant gas [84]|1_._225|1484|1_._00_×_10_−_3|0|
|Water 1 [85]|6_._120|1367|0|3_._430_×_108|
|Water 2 [9]|1_._187|4285|6_._61_×_10_−_4|7_._028_×_108|



Table 2: Density _ρ_ and speed of sound _a_ 0 of the fluids defined in Table 1 for ambient pressure _p_ 0 = 10[5] Pa and ambient temperature _T_ 0 = 300 K, the applied time-step ∆ _t_ , the frequency _f_ of the acoustic waves, the wavelength _λ_ 0 and pressure amplitude ∆ _p_ 0 of the acoustic waves based on linear acoustic theory, as well as the wavelength _λ_ and pressure amplitude ∆ _p_ of the acoustic waves computed with the proposed numerical framework. 

|Fluid|_ρ_ [kgm_−_3]|_a_0[m s_−_1]|∆_t_ [s]|_f_ [s_−_1]|_λ_0[m]|_λ_ [m]|∆_p_0[Pa]|∆_p_ [Pa]|
|---|---|---|---|---|---|---|---|---|
|Air|1_._1574|347_._8|2_._5_×_10_−_6|1750|0_._199|0_._199|4_._025|4_._025|
|JA2 propellant gas [84]|1_._2214|316_._9|2_._7_×_10_−_6|1750|0_._181|0_._181|3_._871|3_._869|
|Water 1 [85]|1000_._0|1449|6_._0_×_10_−_7|7000|0_._207|0_._207|14490|14487|
|Water 2 [9]|1053_._6|1615|5_._4_×_10_−_6|7000|0_._230|0_._231|17016|17012|



regimes, ranging from _M_ = 8 _._ 5 _×_ 10 _[−]_[3] to _M_ = 239, are compared against the exact Riemann solution in Section 6.4. The evolution of Taylor vortices in an inviscid fluid is considered in Section 6.5 to test the conservation of kinetic energy of the proposed numerical framework. In Section 6.6, the Poiseuille flow of an incompressible fluid and the Couette flow of a compressible fluid are simulated to probe the prediction of diffusion-dominated flows, both momentum diffusion and heat conduction, by the proposed numerical framework. The flow of an incompressible fluid in a lid-driven cavity at different Reynolds numbers is considered in Section 6.7 to test the accurate prediction of flows in which both advection and diffusion play an important role, and to demonstrate the correct enforcement of _**∇** ·_ _**u**_ = 0 for incompressible fluids. In Section 6.8, a supersonic flow of an ideal gas and a real gas over a forward-facing step are simulated, predominantly to scrutinise the mass conservation for a complex flow in which different Mach number regimes coexist. Finally, in Section 6.9, the Stokes flow around a rotating sphere is simulated to demonstrate the reliable prediction of flows in complex geometries. 

## _6.1. Acoustic waves_ 

As a first test, the propagation of acoustic waves in a one-dimensional domain is simulated. The formation and propagation of acoustic waves is an important feature of compressible flows and predicting acoustic waves reliably is known to be challenging [43, 44, 63, 82]. In these simulations, the acoustic waves are generated at the domain inlet by a sinusoidal velocity perturbation with amplitude ∆ _u_ 0. For small perturbations to the flow, ∆ _u_ 0 _≪ a_ 0, the resulting wave is a sound wave propagating with the speed of sound _a_ 0. According to linear acoustic theory, the pressure wave has an amplitude of ∆ _p_ 0 = _Z_ ∆ _u_ 0 [83], where _Z_ = _ρa_ is the acoustic impedance. Four different fluids, with the fluid properties given in Table 1, are considered. In each case, the unperturbed flow velocity is _u_ 0 = 1 m s _[−]_[1] , the ambient pressure is _p_ 0 = 10[5] Pa and the ambient temperature is _T_ 0 = 300 K, leading to the density and speed of sound given in Table 2. The computational domain has a length of 1 m, which is represented by an equidistant mesh with mesh spacing ∆ _x_ = 2 _×_ 10 _[−]_[3] m, and the applied time-steps, see Table 2, correspond to a Courant number of Co = _a_ 0∆ _t/_ ∆ _x ≃_ 0 _._ 43. The velocity at the domain inlet is _u_ in = _u_ 0 + ∆ _u_ 0 sin(2 _πft_ ), with frequency _f_ as given in Table 2 and amplitude ∆ _u_ 0 = 0 _._ 01 _u_ 0. 

The computed pressure amplitude ∆ _p_ and the theoretical pressure amplitude ∆ _p_ 0 based on linear acoustic theory, both given in Table 2, are in excellent agreement. Figure 2 shows the profiles of the pressure amplitude ∆ _p_ of the acoustic waves in the four considered fluids as a function of space, with good agreement of the minimum and maximum pressure amplitude with the theoretical pressure amplitude. In addition, the computed wavelength _λ_ is predicted accurately compared to the theoretical wavelength _λ_ 0, given in Table 2, demonstrating a correct prediction of the speed of sound. 

## _6.2. Moving contact discontinuity_ 

A contact discontinuity is a linearly degenerate wave and represents the main source of error in terms of convergence of the applied finite-volume method under mesh refinement [81, 86], with the contact discontinuity progressively smoothing over the course of the simulation [87, 88]. To test the accuracy of the proposed finitevolume framework in predicting contact discontinuities, a moving contact discontinuity in a one-dimensional domain with a length of 1 m is simulated, as considered in previous studies [59, 63]. The contact discontinuity is initially located at _x_ 0 = 0 _._ 5 m, with the initial conditions of the left and right states given as 

**==> picture [214 x 23] intentionally omitted <==**

12 

**==> picture [460 x 280] intentionally omitted <==**

**----- Start of picture text -----**<br>
6 6<br>∆ p 0 ∆ p 0<br>3 3<br>0 0<br>-3 − ∆ p 0 -3 − ∆ p 0<br>-6 -6<br>0 0.2 0.4 0.6 0.8 1.0 0 0.2 0.4 0.6 0.8 1.0<br>x [m] x [m]<br>(a) Air, t  = 2 . 5  ×  10 [−] [3] s (b) JA2 propellant gas, t  = 2 . 7  ×  10 [−] [3] s<br>20 20<br>∆ p 0<br>∆ p 0<br>10 10<br>0 0<br>-10 − ∆ p 0 -10 − ∆ p 0<br>-20 -20<br>0 0.2 0.4 0.6 0.8 1.0 0 0.2 0.4 0.6 0.8 1.0<br>x [m] x [m]<br>(c) Water 1, t  = 6 . 0  ×  10 [−] [4] s (d) Water 2, t  = 5 . 4  ×  10 [−] [4] s<br>[Pa] [Pa]<br>p p<br>∆ ∆<br>[kPa] [kPa]<br>p p<br>∆ ∆<br>**----- End of picture text -----**<br>


Figure 2: Profiles of the pressure amplitude ∆ _p_ of acoustic waves in different fluids, with the fluid properties given in Table 1 and the frequency given in Table 2. The theoretical pressure amplitudes, _±_ ∆ _p_ 0, based on linear acoustic theory are given as a reference. 

The contact discontinuity is simulated in an IG fluid with _γ_ = 1 _._ 4 and _cp_ = 1008 J kg _[−]_[1] K _[−]_[1] , as well as an NASG fluid with _γ_ = 2 _._ 0, _cp_ = 114 _._ 286 J kg _[−]_[1] K _[−]_[1] , Π = 5 _._ 0 Pa and _b_ = 10 _[−]_[3] m[3] kg _[−]_[1] . The transient terms are discretised using the BDF2 scheme and the applied time-step corresponds to Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. 

Figure 3 shows the density profiles at _t_ = 0 _._ 3 s for both the IG fluid and the NASG fluid, using different mesh resolutions. The results of both fluids are in very good agreement and, irrespective of the mesh resolution, the contact discontinuity propagates with the correct velocity. The convergence of the _L_ 1-norm of the solution error associated with a linearly degenerate wave for a _q_ -th order advection scheme without compressive limiting is of order _q/_ ( _q_ + 1) [81]. The spatial convergence of the _L_ 1-norm of the density error, 

**==> picture [313 x 30] intentionally omitted <==**

where _ρ_[comp.] _P_ is the computed density at cell _P_ and _ρ_[exact] _P_ is the corresponding exact density value, obtained with the upwind scheme ( _q_ = 1) and the Minmod scheme ( _q_ = 2) matches the theoretical order of convergence closely in both fluids, as observed in Fig. 4, with convergence order 1 _/_ 2 using the upwind scheme and order 2 _/_ 3 using the Minmod scheme. Furthermore, the self-similarity of the transport of the contact discontinuity is not affected by the choice of fluid model, resulting in only minute differences in the L1-norm _ℓ_ 1( _ρ_ ) between the IG fluid and the NASG fluid for a given mesh resolution, with _|ℓ_ 1( _ρ_ )IG _− ℓ_ 1( _ρ_ )NASG _|/ℓ_ 1( _ρ_ )IG _<_ 10 _[−]_[2] using the Minmod scheme and _|ℓ_ 1( _ρ_ )IG _− ℓ_ 1( _ρ_ )NASG _|/ℓ_ 1( _ρ_ )IG _<_ 10 _[−]_[7] using the upwind scheme. 

In order to test the progressive smearing of the contact discontinuity during the course of the simulation, the contact discontinuity in the IG fluid is simulated in a domain with a length of 3 m and with the contact discontinuity initially located at _x_ 0 = 0 _._ 1 m. The computational domain is resolved by 1200 equidistant mesh cells and the advection terms are discretised using the Minmod scheme. All other settings remain the same as above. The computed density profiles after _n ∈{_ 20 _,_ 200 _,_ 2000 _}_ time-steps are plotted in Fig. 5a, clearly showing a progressive smearing of the contact discontinuity. The width of the contact discontinuity should be proportional to _n_[1] _[/]_[(] _[q]_[+1)] for a _q_ -th order finite-difference or finite-volume method [87, 88], where _n_ is the number of time-steps. Figure 5b shows the width _d_ of the contact discontinuity as a function of the time-step _n_ , with the width _d_ given by the distance between the points at which the density takes the values 0 _._ 55 kg m _[−]_[3] and 0 _._ 95 kg m _[−]_[3] , as illustrated in the inset of Fig. 5b, since the density changes abruptly between 0 _._ 5 kg m _[−]_[3] and 1 _._ 0 kg m _[−]_[3] at the considered contact discontinuity. As shown in Fig. 5b, the width of the contact discontinuity increases with the number of time-steps with a slope closely matching _n_[1] _[/]_[3] , which is the increase expected for a consistently second-order finite-volume method [87]. 

13 

**==> picture [438 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 1.0<br>0.9 0.9<br>0.8 0.8<br>0.7 Theory 0.7 Theory<br>100∆ x 100∆ x<br>0.6 400∆ x 0.6 400∆ x<br>1600∆ x 1600∆ x<br>0.5 0.5<br>0.4 0.5 0.6 0.7 0.8 0.4 0.5 0.6 0.7 0.8<br>x [m] x [m]<br>(a) IG fluid (b) NASG fluid<br>] ]<br>3 3<br>− −<br>[kg m [kg m<br>ρ ρ<br>**----- End of picture text -----**<br>


Figure 3: Profiles of the density _ρ_ of a moving contact discontinuity in (a) an IG fluid and (b) an NASG fluid, on equidistant meshes with different resolutions. The advection terms are discretised using the Minmod scheme, the transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. 

**==> picture [440 x 159] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [−] [1] 10 [−] [1]<br>Upwind Upwind<br>Minmod Minmod<br>10 [−] [2] 10 [−] [2]<br>10 [−] [3] 10 [−] [3]<br>10 [−] [3] 10 [−] [2] 10 [−] [3] 10 [−] [2]<br>∆ x [m] ∆ x [m]<br>(a) IG fluid (b) NASG fluid<br>O (∆ x 1 / 2) O (∆ x 1 / 2)<br>2 / 3) 2 / 3)<br>O (∆ x O (∆ x<br>) )<br>ρ ρ<br>( (<br>1 1<br>ℓ ℓ<br>**----- End of picture text -----**<br>


Figure 4: Spatial convergence of the _L_ 1-norm of the density error, _ℓ_ 1( _ρ_ ), as defined in Eq. (60), of a moving contact discontinuity in (a) an IG fluid and (b) an NASG fluid, using the first-order upwind scheme and the Minmod scheme. The transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. 

**==> picture [433 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 d<br>Theory 10 [0] 0.95<br>0.9 n  = 20 0.75<br>n  = 200 0.55 Minmod<br>0.8<br>n  = 2000<br>-0.1 0 0.1<br>0.7 10 [−] [1] x − x [′] [m]<br>0.6<br>0.5<br>-0.1 0 0.1 10 [−] [2]<br>10 [2] 10 [3]<br>x − x [′] [m] n<br>(a) Profiles of the density ρ (b) Width d of the contact discontinuity<br>O ( n 1 / 3)<br>3]<br>−<br>3] [kg m<br>− ρ<br>[kg m [m] d<br>ρ<br>**----- End of picture text -----**<br>


Figure 5: (a) Profiles of the density _ρ_ of a moving contact discontinuity in an IG fluid, after a different number of time-steps _n_ , with _x[′]_ the position of the contact discontinuity, and (b) the width _d_ of the contact discontinuity, with its definition illustrated in the inset, as a function of time-steps _n_ . The advection terms are discretised using the Minmod scheme, the transient terms are discretised using the BDF2 scheme and the time-step corresponds to Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. 

14 

**==> picture [456 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [13]<br>10 [9] Theory Theory<br>200∆ x 200∆ x<br>10 [−] [2]<br>10 [11]<br>10 [8] 2000∆ x 2000∆ x<br>10 [9]<br>10 [7]<br>10 [−] [3]<br>10 [7]<br>10 [6]<br>Air<br>10 [5] Water<br>10 [5]<br>10 [−] [4]<br>0 . 6 0 . 7 0 . 8 0 . 9 1 0 . 6 0 . 7 0 . 8 0 . 9 1 10 [−] [3] 10 [−] [2]<br>x [m] x [m] ∆ x [m]<br>(a) Air (IG fluid), t  = 14 . 38  µ s (b) Water (NASG fluid), t  = 3 . 096  µ s (c) L 1-norm of the density<br>O (∆ x )<br>O (∆ x )<br>)<br>ρ<br>(<br>[Pa] [Pa] 1<br>p p ℓ<br>**----- End of picture text -----**<br>


Figure 6: Profiles of the pressure _p_ of a shock wave with Mach number _M_ s = 100 in (a) air, described as an IG fluid, and (b) water, described as an NASG fluid, and (c) spatial convergence of the _L_ 1-norm of the density error, _ℓ_ 1( _ρ_ ), as defined in Eq. (60). The exact solution given by the Rankine-Hugoniot relations, Eq. (61), is shown as a reference in (a) and (b). The applied time-step corresponds to Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. 

## _6.3. Shock waves_ 

The propagation of a shock wave poses particular challenges for finite-volume methods, because a shock wave is discontinuous and valid solutions of the governing conservation laws are not guaranteed to satisfy the second law of thermodynamics across shock waves [89]. As such, simulating the propagation of a shock wave is well suited to test whether a numerical scheme reliably converges to the physically-correct weak solution of the governing conservation laws, which is a prerequisite for the accurate prediction of both the speed and strength of shock waves [89, 90]. To this end, the _Lax-Wendroff theorem_ [91] stipulates that if a conservative numerical scheme for hyperbolic conservation laws converges, the computed solution converges towards a weak solution of the conservation laws. 

The propagation of a strong shock wave with Mach number _M_ s = 100 in air and water in a one-dimensional domain with a length of 1 m is simulated. Air is described by the IG model using the fluid properties given in Table 1 and water is described by the NASG model using the fluid properties proposed by Le M´etayer and Saurel [9], also given in Table 1 (see properties of _Water 2_ ). Viscous stresses and heat conduction are neglected, _i.e. µ_ = _k_ = 0, so the governing equations (1)-(3) reduce to the Euler equations [92], which are hyperbolic. From the Rankine-Hugoniot relations, the pressure and density ratios across a shock wave propagating with velocity _u_ s in a quiescent NASG fluid are given as 

**==> picture [342 x 77] intentionally omitted <==**

where subscript I denotes the post-shock state, subscript II denotes the pre-shock state and _M_ s = _u_ s _/a_ II is the Mach number of the shock wave. With the initial conditions of the pre-shock state (II) for both cases given as 

**==> picture [188 x 11] intentionally omitted <==**

the shock relations yield the initial conditions of the post-shock state (I) for air, 

**==> picture [269 x 11] intentionally omitted <==**

and for water, 

**==> picture [271 x 11] intentionally omitted <==**

The shock wave is initially located at _x_ s,0 = 0 _._ 25 m and the applied time-step corresponds to Co = _u_ s∆ _t/_ ∆ _x_ = 0 _._ 5. 

The Rankine-Hugoniot relations are reproduced accurately in both air and water, as seen in Fig. 6, despite the very large pressure discontinuities with pressure ratios of more than four and seven orders of magnitude, respectively. In both fluids the _L_ 1-norm of the density error, _ℓ_ 1( _ρ_ ), converges with first order under mesh refinement, as seen in Fig. 6c. The first order convergence is imposed by the applied monotone discretisation 

15 

**==> picture [473 x 379] intentionally omitted <==**

**----- Start of picture text -----**<br>
10001.0 0.00855<br>Theory 25.000 3 . 5 ×<br>250∆ x<br>1000∆ x<br>10000.5 0.00850<br>24.999<br>10000.0 0.00845<br>24.998<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1<br>x [m] x [m] x [m]<br>(a) Pressure p (b) Density ρ (c) Mach number M<br>7: Profiles of pressure, density and Mach number of the low-Mach shock tube at t  = 0 . 01 s, compared against the theoretical<br>Riemann solution. In addition, a magnified view of the minute change in Mach number at the contact discontinuity is shown<br>1.0 1.0 1.0<br>Theory<br>250∆ x<br>0.8 0.8 0.8<br>1000∆ x<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0 0 0<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1<br>x [m] x [m] x [m]<br>(a) Pressure (b) Density (c) Mach number<br>]<br>3<br>[Pa] − M<br>p [kg m<br>ρ<br>]<br>3<br>[Pa] − M<br>p [kg m<br>ρ<br>**----- End of picture text -----**<br>


Figure 7: Profiles of pressure, density and Mach number of the low-Mach shock tube at _t_ = 0 _._ 01 s, compared against the theoretical Riemann solution. In addition, a magnified view of the minute change in Mach number at the contact discontinuity is shown in (c). 

Figure 8: Profiles of pressure, density and Mach number of Sod’s shock tube at _t_ = 0 _._ 15 s, compared against the theoretical Riemann solution. 

schemes, in this case the Minmod scheme, and is expected for an oscillation-free numerical simulation of a shock wave [93]. The robust convergence for strong shock waves further implies accurate conservation properties as well as convergence to the correct weak solution of the governing conservation laws using the proposed finite-volume framework and pressure-based algorithm. 

## _6.4. Shock tubes_ 

Shock tubes are routinely and extensively used to test numerical frameworks and schemes for compressible flows, because they feature shock waves, rarefaction fans as well as contact discontinuities and because an exact reference solution based on the associated Riemann problem exists. Three different shock tubes, covering Mach numbers over five orders of magnitude, are considered. In all cases, the fluid has a heat capacity ratio of _γ_ = 1 _._ 4 and a specific gas constant of _cp_ = 1008 J kg _[−]_[1] K _[−]_[1] . 

A low-Mach shock tube, as proposed by Moguen et al. [94], is considered. The discontinuity is initially located at _x_ 0 = 0 _._ 5 m, with the initial conditions of the left and right states given as 

**==> picture [254 x 23] intentionally omitted <==**

The applied time-step corresponds to a Courant number of Co = ( _u_ L + _a_ L)∆ _t/_ ∆ _x_ = 0 _._ 5. Overall, the results obtained on both meshes are in very good agreement with the theoretical Riemann solution, as seen in Fig. 7. Because the particle velocity is very small, _u_ max = 0 _._ 202 m s _[−]_[1] , the contact discontinuity only moves by 0 _._ 002 m in the studied time frame and, thus, remains very sharp, as evident by the density profile in Fig. 7. A small wiggle is observed in the Mach number profile at the contact discontinuity, which however has no impact on the overall result. 

16 

**==> picture [424 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
18 125 250<br>Theory<br>15<br>250∆ x 100 200<br>1000∆ x<br>12<br>75 150<br>9<br>50 100<br>6<br>25 50<br>3<br>0 0 0<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1<br>x [m] x [m] x [m]<br>(a) Pressure (b) Density (c) Mach number<br>]<br>3<br>−<br>M<br>[MPa]<br>p [kg m<br>ρ<br>**----- End of picture text -----**<br>


Figure 9: Profiles of pressure, density and Mach number of the high-Mach shock tube at _t_ = 3 _._ 5 _×_ 10 _[−]_[4] s, compared against the theoretical Riemann solution. 

The shock tube initially introduced by Sod [95] is considered as a shock tube with intermediate Mach number, with initial conditions 

**==> picture [216 x 24] intentionally omitted <==**

The discontinuity is initially located at _x_ 0 = 0 _._ 5 m and the applied time-step corresponds to a Courant number of Co = _a_ L∆ _t/_ ∆ _x_ = 0 _._ 6. The results obtained on both meshes, shown in Fig. 8, are in very good agreement with the theoretical Riemann solution. 

The high-Mach shock tube proposed by Xiao [54] is considered. The discontinuity is initially located at _x_ 0 = 0 _._ 5 m, with the initial conditions 

**==> picture [219 x 23] intentionally omitted <==**

Notably, the flow of the left state has a Mach number of _M_ L = 239. The applied time-step corresponds to a Courant number of Co = _u_ L∆ _t/_ ∆ _x_ = 0 _._ 5. As observed in Fig. 9, although the profile of the Mach number is not predicted very accurately on the coarse mesh, the density and pressure profiles are in good agreement with the theoretical Riemann solution. On the fine mesh, the computed results are in very good agreement with the theoretical Riemann solution, demonstrating the accurate prediction of high-Mach Riemann problems with the proposed numerical framework. 

## _6.5. Taylor vortices_ 

The conservation of kinetic energy is a fundamental property arising from the conservation of mass and momentum. Two-dimensional Taylor vortices in an inviscid ( _µ_ = 0), non-conducting ( _k_ = 0) fluid are simulated to analyse the conservation of kinetic energy by the proposed numerical framework. The domain has the dimensions 2 m _×_ 2 m and is periodic in all directions, so that energy transfer across the domain boundaries does not have to be considered. The initial conditions, shown in Fig. 10, are _u_ = _−_ cos( _πx_ ) sin( _πy_ ), _v_ = sin( _πx_ ) cos( _πy_ ) and _p_ = _−_ 0 _._ 25 [cos(2 _πx_ ) + cos(2 _πy_ )]. Since _µ_ = _k_ = 0, the Taylor vortices are steady and no energy dissipation occurs naturally, with a constant kinetic energy of 

**==> picture [327 x 30] intentionally omitted <==**

where Ωis the volume of the computational domain. Any dissipation of kinetic energy is, thus, the result of numerical dissipation induced by the applied discretisation. 

Figure 11a shows the evolution of the error in kinetic energy of the Taylor vortices, 

**==> picture [286 x 30] intentionally omitted <==**

with _E_ kin[(0)][the][kinetic][energy][of][the][initialised][(] _[t]_[=][0)][flow][field,][in][an][IG][fluid][with] _[γ]_[=][1] _[.]_[4][and] _[c][p]_[=] 1008 J kg _[−]_[1] K _[−]_[1] , and Mach number _M_ = 0 _._ 01. As expected, the error in kinetic energy is substantially larger 

17 

**==> picture [169 x 137] intentionally omitted <==**

**==> picture [170 x 137] intentionally omitted <==**

**==> picture [242 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Velocity u (b) Pressure p<br>**----- End of picture text -----**<br>


Figure 10: Contours of the initial velocity _u_ along the _x_ -axis and the initial pressure _p_ of the Taylor vortices. 

**==> picture [463 x 157] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [0]<br>10 [−] [1] 10 [−] [1]<br>10 [−] [1]<br>10 [−] [2] 10 [−] [2]<br>10 [−] [2]<br>10 [−] [3] 10 [−] [3]<br>10 [−] [3]<br>10 [−] [4] 10 [−] [4]<br>10 [−] [5] Upwind 10 [−] [5] 10 [−] [4] Upwind<br>Central Central Central<br>10 [−] [6] 10 [−] [6] 10 [−] [5]<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1 0.01 0.1<br>t [s] t [s] ∆ x [m]<br>(a) Evolution of ε kin with MWI. (b) Evolution of ε kin without MWI. (c) Convergence of ε kin at t  = 1 s.<br>O (∆ x 1)<br>3)<br>O (∆ x<br>kin kin kin<br>ε ε ε<br>**----- End of picture text -----**<br>


Figure 11: Temporal evolution of _ε_ kin, Eq. (64), on an equidistant Cartesian mesh with ∆ _x_ = 0 _._ 04 m defining the advecting velocity _ϑf_ (a) with the MWI as described in Section 5.1 and (b) without the MWI as _ϑf_ = _**u** f ·_ _**n** f_ , and (c) convergence of the error in kinetic energy _ε_ kin of the Taylor vortices with the MWI as described in Section 5.1. The first-order upwind scheme or the central differencing scheme are applied for the discretisation of the advection term. The applied time-step in all cases is ∆ _t_ = 2 _×_ 10 _[−]_[3] s. 

using the first-order upwind scheme compared to the error in kinetic energy obtained using the central differencing scheme. Interestingly, the applied transient discretisation scheme, _i.e._ BDF1 or BDF2, does not affect the error in kinetic energy, which is consistent with the Taylor vortices being a steady flow in the absence of molecular viscosity and heat conduction. However, even with central differencing, kinetic energy is dissipated as a result of the MWI formulation of the advecting velocity [65], see Eq. (34). No appreciable distortion of the vortices is observed for the considered simulations when central differencing is applied, which is consistent with the only small error in kinetic energy ( _ε_ kin _<_ 1%) in theses cases. 

The flow is sufficiently compressible ( _M_ = 0 _._ 01) and smooth, that pressure and velocity remain coupled even without MWI [65]. Exploiting this by omitting the correction introduced by the MWI, with the advecting velocity simply defined as _ϑf_ = _**u** f ·_ _**n** f_ , the error in kinetic energy remains constant for _t_ ≳ 0 _._ 08 s, as seen in Fig. 11b, which indicates that the numerical dissipation of kinetic energy is negligible. This is to be expected when simulating a sufficiently smooth flow with a second-order accurate finite-volume framework without any explicitly introduced physical or numerical dissipation. Only a small error in kinetic energy is observed at the beginning of the simulation, caused by the initial conditions [65]. 

The error in kinetic energy converges with third order using central differencing under mesh refinement, as shown in Fig. 11c, which is consistent with the third-order convergence of the error in kinetic energy introduced by the MWI [65]. On the other hand, when the first-order upwind scheme is applied, the kinetic energy dissipated artificially by the MWI is insignificant compared to the numerical diffusion introduced by the upwind scheme, as evident by the first-order convergence of the error in kinetic energy shown in Fig. 11c. 

These results, therefore, suggest that the MWI is the only source of numerical dissipation in the proposed finite-volume discretisation, assuming a consistent second-order (or higher-order) interpolation of spatial and transient terms is applied, _e.g._ central differencing and BDF2. 

18 

**==> picture [450 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
1<br>10 [−] [2]<br>0 . 8<br>U t = 0<br>0 . 6 10 [−] [3]<br>d U ( y ) 0 . 4<br>10 [−] [4]<br>Simulation<br>0 . 2<br>Theory<br>y U b = 0 0 10 [−] [5]<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0.01 0.1<br>x y/d ∆ y/d<br>(a) Schematic (b) Axial velocity profile (c) L∞ -norm of the axial velocity<br>2)<br>(∆ y<br>) O<br>max ( U<br>∞<br>ℓ<br>U/U<br>**----- End of picture text -----**<br>


Figure 12: Schematic of a planar Poiseuille flow, as well as the profile of the axial velocity compared against the analytical solution, Eq. (65), and spatial convergence of the _L∞_ -norm of the error in axial velocity, Eq. (66), of the planar Poiseuille flow of an incompressible fluid. The axial velocity profile in (b) is obtained on a mesh with ∆ _y_ = _d/_ 20, with each dot representing a cell-centred value. 

## _6.6. Diffusion-dominated flows_ 

The test-cases discussed in the previous sections only test the discretisation of the transient and advection terms, not taking into account diffusion terms, _i.e._ viscous stresses, heat conduction and viscous heating. Two well-defined diffusion-dominated flows, a planar Poiseuille flow of an incompressible fluid and a planar Couette flow of a compressible fluid, are considered to test the discretisation and implementation of the diffusion of momentum and heat. 

The planar Poiseuille flow of an incompressible fluid between two parallel plates of infinite length separated by a constant distance _d_ , illustrated schematically in Fig. 12a, is a flow that is entirely governed by viscous stresses. Assuming the viscosity _µ_ is constant and the flow is laminar, the velocity profile is readily given as 

**==> picture [291 x 25] intentionally omitted <==**

where _−_ d _p/_ d _x_ is the driving pressure gradient. This type of flow, thus, allows a straightforward quantification of the solution error associated with the axial velocity. The computational domain is taken to be periodic in the streamwise direction, to circumvent any influence of inlet and outlet boundary conditions, and the flow is driven by a constant momentum source corresponding to the driving pressure gradient _−_ d _p/_ d _x_ . The profile of the axial velocity _U_ of the planar Poiseuille flow obtained on a mesh with a resolution of ∆ _y_ = _d/_ 20 is shown in Fig. 12b, alongside the spatial convergence of the _L∞_ -norm of the error in axial velocity, 

**==> picture [313 x 26] intentionally omitted <==**

in Fig. 12c. The axial velocity profile is in excellent agreement with the analytical solution, Eq. (65), and the _L∞_ -norm of the error in axial velocity converges with second order under mesh refinement, as expected given the second-order discretisation of the viscous stresses in Eq. (38). 

The planar Couette flow of a compressible fluid between two parallel plates of infinite length separated by a constant distance _d_ , illustrated schematically in Fig. 13a, is a compressible flow that is dominated by viscous stresses and heat conduction. Assuming the viscosity _µ_ is constant and the stationary wall is adiabatic, the velocity and temperature profiles only depend on the Prandtl number Pr = _µ cp/k_ and the Mach number _M_ m = _U_ m _/a_ m at the moving wall, with the velocity given as _U_ ( _y_ ) _/U_ m = _y/d_ and the temperature given as [96] 

**==> picture [324 x 25] intentionally omitted <==**

This type of flow, thus, allows a straightforward quantification of the solution error associated with the temperature. The considered fluid is an ideal gas with a Prandtl number of Pr = 1 and a heat capacity ratio of _γ_ = 1 _._ 4. The computational domain is taken to be periodic in the streamwise direction, to circumvent any influence of inlet and outlet boundary conditions. The profile of the temperature _T_ of the planar compressible Couette flow with _M_ m = 1 _._ 0 obtained on a mesh with a resolution of ∆ _y_ = _d/_ 20 is shown in Fig. 13b, alongside the spatial convergence of the _L∞_ -norm of the error in temperature, 

**==> picture [312 x 25] intentionally omitted <==**

19 

**==> picture [454 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.20<br>10 [−] [2]<br>U m, T m 1.15<br>10 [−] [3]<br>1.10<br>d U ( y )<br>10 [−] [4]<br>1.05 Simulation<br>Theory M∞ = 0 . 1<br>M∞ = 1 . 0<br>y U s = 0, [∂T] = 0 1.00 0 0 . 2 0 . 4 0 . 6 0 . 8 1 10 [−] [5] 0.01 0.1<br>x ∂y ����s y/d ∆ y/d<br>(a) Schematic (b) Temperature profile (c) L∞ -norm of the temperature<br>2)<br>(∆ y<br>O<br>m ) T<br>(<br>∞<br>T/T ℓ<br>**----- End of picture text -----**<br>


Figure 13: Schematic of a planar Couette flow, as well as the profile of the temperature compared against the analytical solution, Eq. (67), and spatial convergence of the _L∞_ -norm of the temperature, Eq. (68), of the planar Couette flow of a compressible fluid for both considered Mach numbers. The temperature profile in (b) is obtained on a mesh with ∆ _y_ = _d/_ 20, with each dot representing a cell-centred value. 

**==> picture [445 x 208] intentionally omitted <==**

**----- Start of picture text -----**<br>
u w<br>Wall (moving)<br>Wall (stationary)<br>y<br>L<br>x<br>(a) Schematic (b) Polygonal mesh<br>L<br>(stationary) (stationary)<br>Wall Wall<br>**----- End of picture text -----**<br>


Figure 14: Schematic illustration and polygonal mesh of the lid-driven cavity. 

at Mach numbers _M_ m _∈{_ 0 _._ 1 _,_ 1 _._ 0 _}_ in Fig. 13c. The temperature profile is in excellent agreement with the analytical solution, Eq. (67), and the _L∞_ -norm of the error in temperature converges with second order under mesh refinement for both Mach numbers, as expected given the second-order discretisation of the heat conduction term in Eq. (40). Notably, _ℓ∞_ ( _T_ ) is independent of the Mach number _M_ m for a sufficiently high spatial resolution, as seen in Fig. 13c. 

## _6.7. Lid-driven cavity_ 

The lid-driven cavity, schematically shown in Fig. 14a, is a common test case to validate numerical methods for fluid flows, since it captures convective and diffusive momentum transport of the fluid. The considered two-dimensional domain is of size _L × L_ , with no-slip boundary conditions imposed on all four walls. The top wall moves with velocity _u_ w and the flow of the incompressible fluid has a Reynolds numbers of Re = _ρ L u_ w _/µ ∈ {_ 100 _,_ 1000 _}_ . A polygonal mesh with 8708 cells, shown in Fig. 14b, represents the computational domain. 

Figures 15 and 16 show the _u_ -velocity profile in the _y_ -direction and the _v_ -velocity profile in the _x_ -direction along lines that pass through the centre of the domain for the two considered Reynolds numbers, compared against the reference results of Ghia et al. [97]. The results are in very good agreement with the reference results of Ghia et al. [97], as well as other studies that have previously considered this test-case [43, 60, 67, 98], for both considered Reynolds numbers, demonstrating the accurate prediction of the convective-diffusive transport of momentum on unstructured meshes using the proposed algorithm. The contours of the divergence of velocity, _**∇** ·_ _**u**_ , at steady state are shown in Fig. 17 for the lid-driven cavity with Re = 1000, alongside the transient 

20 

**==> picture [359 x 170] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 0 . 2<br>Simulation<br>Ghia et al. (1982)<br>0 . 5 0<br>0 − 0 . 2<br>Simulation<br>Ghia et al. (1982)<br>− 0 . 5 − 0 . 4<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1<br>y/L x/L<br>(a) u -velocity (b) v -velocity<br>w w<br>u/u v/u<br>**----- End of picture text -----**<br>


Figure 15: Profiles of (a) the _u_ -velocity along the _y_ -centreline of the domain and (b) the _v_ -velocity along the _x_ -centreline of the domain of the lid-driven cavity with Re = 100. The results of Ghia et al. [97] are shown as a reference. 

**==> picture [356 x 170] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 0.9<br>Simulation Simulation<br>Ghia et al. (1982) 0.6 Ghia et al. (1982)<br>0 . 5<br>0.3<br>0<br>0<br>-0.3<br>− 0 . 5 -0.6<br>0 0 . 2 0 . 4 0 . 6 0 . 8 1 0 0 . 2 0 . 4 0 . 6 0 . 8 1<br>y [m] x [m]<br>(a) u -velocity (b) v -velocity<br>1] 1]<br>− −<br>[m s [m s<br>u v<br>**----- End of picture text -----**<br>


Figure 16: Profiles of (a) the _u_ -velocity along the _y_ -centreline of the domain and (b) the _v_ -velocity along the _x_ -centreline of the domain of the lid-driven cavity with Re = 1000. The results of Ghia et al. [97] are shown as a reference. 

**==> picture [393 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [−] [11]<br>10 [−] [12]<br>10 [−] [13]<br>10 [−] [14]<br>0 5 10 15 20 25<br>t [s]<br>(a) Divergence of the velocity field at steady state (b)  L 1-norm of the error in the divergence of the<br>velocity field<br>)<br> u ·<br>∇<br>(<br>1<br>ℓ<br>**----- End of picture text -----**<br>


Figure 17: (a) Contours of the divergence of the velocity field, _**∇** ·_ _**u**_ , at steady state and b) _L_ 1-norm of the error in the divergence of the velocity field, _ℓ_ 1( _**∇** ·_ _**u**_ ), for the lid-driven cavity with Re = 1000. 

21 

**==> picture [227 x 107] intentionally omitted <==**

**==> picture [228 x 107] intentionally omitted <==**

**==> picture [302 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Mach number M (b) Pressure p<br>**----- End of picture text -----**<br>


Figure 18: Contours of Mach number and pressure of the supersonic flow over a forward-facing step at _t_ = 4 s with co-volume _b_ = 0. 

**==> picture [227 x 107] intentionally omitted <==**

**==> picture [228 x 107] intentionally omitted <==**

**==> picture [302 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Mach number M (b) Pressure p<br>**----- End of picture text -----**<br>


Figure 19: Contours of Mach number and pressure of the supersonic flow over a forward-facing step at _t_ = 4 s with co-volume _b_ = 0 _._ 1 m[3] kg _[−]_[1] . 

evolution (considering an initially quiescent fluid) of the _L_ 1-norm of the error in the divergence of the velocity field, given as 

**==> picture [358 x 37] intentionally omitted <==**

where _f_ are the faces of cell _P_ . The divergence-free condition of the velocity field imposed by the conservation of mass in conjunction with the considered incompressible fluid, see Eq. (54), is satisfied accurately, with only marginal errors subject to the applied tolerance of the iterative solver (see Section 5). This is to be expected from the proposed algorithm, as _**∇** ·_ _**u**_ = 0 is implicitly enforced by Eq. (57). 

## _6.8. Forward-facing step_ 

The two-dimensional supersonic flow over a forward-facing step of an initially uniform flow features the spatiotemporal evolution of shock waves, developing transonic flow and large pressure gradients. This test-case is, thus, well suited to test the conservation properties of the finite-volume discretisation as well as the stability of the pressure-based algorithm during the transient development of large pressure gradients. Following Woodward and Colella [99], the height of the computational domain is 1 m, and the step has of height 0 _._ 2 m and is positioned at 0 _._ 6 m from the inlet of the domain. The flow entering the domain has a Mach number of _M_ = _u/a_ 0 = 3 and a pressure of _p_ 0 = 1 Pa. The two-dimensional domain is represented by an equidistant Cartesian mesh with ∆ _x_ = 0 _._ 01 m and the applied time-step corresponds to Co = _u_ ∆ _t/_ ∆ _x_ = 0 _._ 75. The considered fluid has a heat capacity ratio of _γ_ = 1 _._ 4 and a specific isobaric heat capacity of _cp_ = 1008 J kg _[−]_[1] K _[−]_[1] , with a co-volume of either _b_ = 0 or _b_ = 0 _._ 1 m[3] kg _[−]_[1] . Figure 18 shows the contours of the Mach number and the pressure at _t_ = 4 s for _b_ = 0, which are in good agreement with previously reported results [33, 99, 100]. Changing the co-volume to _b_ = 0 _._ 1 m[3] kg _[−]_[1] , the position of the primary shock wave in front of the forward-facing step moves further upstream and fewer reflected shock waves can be observed, as seen in Fig. 19. 

Based on the initial mass _m_[(0)] at _t_ = 0, the mass in the domain Ωand the mass entering and leaving the domain over its boundaries _∂_ Ω, the conservation error of mass at time _t_ is given as 

**==> picture [366 x 26] intentionally omitted <==**

where **Σ** is the outward-pointing surface vector of the surface _∂_ Ωof the computational domain Ω. The temporal evolution of the mass conservation error of the supersonic flow over the forward-facing step is shown 

22 

**==> picture [458 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [−] [2] 10 [−] [2]<br>10 [−] [4] 10 [−] [4]<br>η = 10 [−] [5] η = 10 [−] [5]<br>η = 10 [−] [6] η = 10 [−] [6]<br>10 [−] [6] 10 [−] [6]<br>η = 10 [−] [7] η = 10 [−] [7]<br>η = 10 [−] [8] η = 10 [−] [8]<br>10 [−] [8] 10 [−] [8]<br>10 [−] [10] 10 [−] [10]<br>0 1 2 3 4 0 1 2 3 4<br>t [s] t [s]<br>(a) b  = 0 (b) b  = 0 . 1 m [3] kg [−] [1]<br>m m<br>ε ε<br>**----- End of picture text -----**<br>


Figure 20: Temporal evolution of the mass conservation error, _εm_ , as defined in Eq. (70), of the supersonic flow over a forward-facing step, obtained with different solution tolerances _η_ . 

**==> picture [419 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 R<br>ux,∞<br>ωz<br>y<br>100 R<br>x<br>(a) Schematic (not to scale) (b) Mesh with velocity contours<br>R<br>100<br>**----- End of picture text -----**<br>


Figure 21: Schematic of the flow around a rotating sphere (in the _xy_ -plane through the centre of the domain) and applied mesh in the vicinity of the sphere together with the contours of the axial velocity. 

in Fig. 20, obtained with both considered co-volumes, _b ∈{_ 0 _,_ 0 _._ 1 _}_ m[3] kg _[−]_[1] , with different solution tolerances, _η ∈{_ 10 _[−]_[5] _,_ 10 _[−]_[6] _,_ 10 _[−]_[7] _,_ 10 _[−]_[8] _}_ , applied for the solution of the system of governing equations (33). Overall, the proposed finite-volume framework conserves mass accurately and the mass conservation error is predominantly a function of the solution tolerance, with a decreasing mass conservation error for a decreasing solution tolerance. 

## _6.9. Rotating sphere_ 

The flow of an incompressible fluid around a sphere with radius _R_ , rotating at angular velocity _**ω**_ , in a Stokes flow with Reynolds number Re = _ρR|_ _**u** ∞|/µ ≪_ 1, where _**u** ∞_ is the free-stream velocity, is considered. As a result of the rotation, a lift force is acting on the sphere, also known as Magnus effect, with the analytical solution for the force on the sphere given as [101] 

**==> picture [344 x 25] intentionally omitted <==**

where the first term on the right-hand side represents the drag force and the second term represents the lift force. The sphere is simulated in a cubical three-dimensional domain of size 100 _R ×_ 100 _R ×_ 100 _R_ , illustrated schematically in Fig. 21a, with the sphere placed at the centre of the domain. The considered flow has the free-stream velocity _**u** ∞_ = ( _ux,∞,_ 0 _,_ 0) _[T]_ , corresponding to Re = 0 _._ 05, and the sphere rotates around its _z_ -axis with _**ω**_ = (0 _,_ 0 _, ωz_ ) _[T]_ . The computational domain is represented with a boundary-fitted hexahedral mesh with 384 000 cells, shown in Fig. 21b, which is strongly refined in the vicinity of the sphere and gradually coarsened (growth factor 1 _._ 2) with increasing distance from the sphere. The applied time-step is ∆ _t_ = 100 _tµ_ , where _tµ_ = _ρR_[2] _/µ_ is the viscous timescale, which corresponds to a maximum Courant number of Co = 49 _−_ 1559, 

23 

**==> picture [450 x 151] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.04 1.010<br>1.02 1.005<br>1.00 1.000<br>ˆ ˆ<br>ω = 0 . 1 ω = 0 . 1<br>0.98 ω ˆ = 1 0.995 ω ˆ = 1<br>ω ˆ = 10 ω ˆ = 10<br>0.96 0.990<br>0 2 000 4 000 6 000 8 000 10 000 0 2 000 4 000 6 000 8 000 10 000<br>τ τ<br>(a) Drag coefficient (b) Lift coefficient<br>/C d0 , /C l0 ,<br>C d C l<br>**----- End of picture text -----**<br>


Figure 22: Drag coefficient _C_ d and lift coefficient _C_ l of the rotating sphere in Stokes flow for different dimensionless angular velocities _ω_ ˆ as a function of the dimensionless time _τ_ = _t/tµ_ , normalised with the theoretical values, _C_ d _,_ 0 and _C_ l _,_ 0, based on Eq. (71). 

dependent on the angular velocity _ωz_ , for the considered simulations. The transient term is discretised with the BDF2 scheme and the advection terms are discretised using the Minmod scheme. 

Fig. 22 shows the transient evolution of the drag coefficient, _C_ d = 2 _F_ d _/ρA_ p _u_[2] _x,∞_[,][and][the][lift][coefficient,] _C_ l = 2 _F_ l _/ρA_ ˆ p _u_[2] _x,∞_[,][with] _[A]_[p][=] _[πR]_[2][the][projected][area][of][the][sphere,][for][three][different][dimensionless][angular] velocities, _ω_ = _R ωz/ux,∞_ , as a function of the dimensionless time _τ_ = _t/tµ_ . For all three angular velocities the drag and lift coefficients are predicted accurately compared to the analytical solution, Eq. (71), with errors _<_ 1% for both drag and lift coefficients. 

## **7. Conclusions** 

A conservative numerical framework for the prediction of flows of incompressible, ideal-gas and real-gas fluids at all speeds has been presented. This numerical framework is founded on a unified thermodynamic closure model for incompressible and compressible fluids, a standard finite-volume discretisation applicable to structured and unstructured meshes, a single flux definition based on a momentum-weighted interpolation, as well as a fully-coupled pressure-based algorithm with collocated variable arrangement. The proposed unified thermodynamic closure model combines the definitions of incompressible fluids with the Noble-Abel-stiffenedgas model [9] for ideal-gas and real-gas fluids, which facilitates a straightforward finite-volume discretisation that is applicable to incompressible flows as well as compressible flows in all Mach number regimes. Since the thermodynamic closure model requires only the definition of the density and specific static enthalpy, it can be extended to more complex gas models, such as the Peng-Robinson model [11], without changes to the finite-volume discretisation or the pressure-based algorithm. The employed finite-volume framework combines well-established conservative discretisation schemes to yield a consistently second-order accurate discretisation that is applicable to structured and unstructured meshes. The discretised governing equations are solved in a single linear system of equations for pressure, velocity and temperature, which enables a robust solution for flows at any speed. 

The main feature of the proposed finite-volume discretisation and pressure-based algorithm is the accurate and robust simulation of flows of incompressible and compressible fluids at all speeds without changes to the discretisation or the solution procedure. Using a Newton linearisation of the continuity equation in conjunction with the semi-implicit discretisation of the fluxes through the mesh faces by a momentum-weighted interpolation method, the discretised continuity equation acts as a transport equation for density in compressible flows and as a constraint on the velocity field in incompressible flows. This allows this numerical framework to represent the incompressible limit correctly and enables the simulation of flows of both incompressible and compressible fluids with the same algorithm. 

The proposed numerical framework has been validated using a broad variety of test-cases, demonstrating accurate and robust results, irrespective whether the considered flow was of an incompressible fluid, an ideal-gas fluid or a real-gas fluid, with an error convergence consistent of a second-order finite-volume discretisation. The propagation of acoustic waves demonstrated an accurate prediction of the speed of sound and acoustic effects in general, while the propagation of a moving contact discontinuity demonstrated convergence for linearly degenerate waves. The propagation of a strong shock wave as well as the shock tubes in different Mach number regimes scrutinised the resolution of strongly nonlinear and discontinuous flow features, which are predicted accurately in all Mach number regimes. In particular, the speed, position and strength of strong shock waves are predicted accurately, demonstrating that the finite-volume framework converges to the correct weak solution of the governing equations [90], further suggesting that the proposed algorithm implicitly satisfies the 

24 

second law of thermodynamics. The evolution of Taylor vortices in an inviscid fluid offered the possibility to test the conservation of energy of the proposed numerical framework, showing that the momentum-weighted interpolation is the only source of numerical energy dissipation, an error which however converges with third order under mesh refinement. The Poiseuille flow of an incompressible fluid and the Couette flow of a compressible fluid demonstrated the accurate simulation of flows in which viscous stresses and heat conduction play a dominant role. The flow of an incompressible fluid in a lid-driven cavity at different Reynolds numbers further demonstrated the accurate simulation of flows in which both advection and diffusion play an important role, and demonstrated the correct enforcement of _**∇** ·_ _**u**_ = 0 for incompressible fluids to any chosen solver tolerance (within the limit of machine precision), on unstructured meshes. The results presented for the supersonic flow of an ideal gas and a real gas over a forward-facing step demonstrated accurate mass conservation, even for complex flows in which different Mach number regimes coexist. Lastly, the Stokes flow around a rotating sphere demonstrated that flows in complex three-dimensional geometries can be predicted accurately with the proposed numerical framework. 

In this paper we have put forward a thermodynamic closure model, a finite-volume discretisation and a fully-coupled pressure-based algorithm for the prediction of the behaviour of the flow of incompressible fluids as well as compressible fluids described by ideal- or real-gas models on arbitrary meshes. We have combined these constituent parts into a fully-coupled pressure-based framework and have shown that this framework is able to predict realistic flows at any speed. However, these parts can also be used individually, for instance in existing frameworks. 

## **References** 

- [1] F. H. Harlow, A. A. Amsden, Numerical calculation of almost incompressible flow, Journal of Computational Physics 3 (1968) 80–93. 

- [2] F. H. Harlow, A. A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, Journal of Computational Physics 8 (1971) 197–213. 

- [3] S. Klainerman, A. Majda, Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids, Communications on Pure and Applied Mathematics 34 (1981) 481–524. 

- [4] D. Hoff, The Zero-Mach Limit of Compressible Flows, Communications in Mathematical Physics 192 (1998) 543–554. 

- [5] E. F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical Introduction, Springer, third edition, 2009. 

- [6] F. Harlow, A. Amsden, Fluid Dynamics, Monograph LA-4700, Los Alamos National Laboratory, 1971. 

- [7] R. Saurel, O. Le M´etayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiff mechanical relaxation, Shock Waves 16 (2007) 209–232. 

- [8] T. L. Hill, An Introduction to Statistical Thermodynamics, Dover Publications, New York, 1986. 

- [9] O. Le M´etayer, R. Saurel, The Noble-Abel Stiffened-Gas equation of state, Physics of Fluids 28 (2016) 046102. 

- [10] G. Soave, Equilibrium constants from a modified Redlich-Kwong equation of state, Chemical Engineering Science 27 (1972) 1197–1203. 

- [11] D.-Y. Peng, D. B. Robinson, A New Two-Constant Equation of State, Industrial & Engineering Chemistry Fundamentals 15 (1976) 59–64. 

- [12] G. Hauke, T. J. Hughes, A comparative study of different sets of variables for solving compressible and incompressible flows, Computer Methods in Applied Mechanics and Engineering 153 (1998) 1–44. 

- [13] A. J. Chorin, J. E. Marsden, A Mathematical Introduction to Fluid Mechanics, Springer Verlag, 1993. 

- [14] K.-H. Chen, R. Pletcher, Primitive Variable, Strongly Implicit Calculation Procedure for Viscous Flows at All Speeds, AIAA Journal 29 (1991) 1241–1249. 

- [15] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165. 

- [16] J. H. Ferziger, M. Peric, R. L. Street, Computational Methods for Fluid Dynamics, Springer International Publishing, 4th edition, 2020. 

- [17] W. S. O˙za´nski, The Lagrange multiplier and the stationary Stokes equations, Journal of Applied Analysis 23 (2017). 

- [18] A. Toutant, General and exact pressure evolution equation, Physics Letters A 381 (2017) 3739–3742. 

- [19] J. Van Doormaal, G. Raithby, B. McDonald, The Segregated Approach to Predicting Viscous Compressible Fluid Flows, ASME Journal of Turbomachinery 109 (1987) 268–277. 

- [20] H. Bijl, P. Wesseling, A Unified Method for Computing Incompressible and Compressible Flows in Boundary-Fitted Coordinates, Journal of Computational Physics 141 (1998) 153–173. 

- [21] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. 

- [22] K. C. Karki, S. V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA Journal 27 (1989) 1167–1174. 

- [23] C. M. Rhie, Pressure-based Navier-Stokes solver using the multigrid method, AIAA Journal 27 (1989) 1017–1018. 

- [24] S. M. H. Karimian, G. E. Schneider, Pressure-based control-volume finite element method for flow at all speeds, AIAA Journal 33 (1995) 1611–1618. 

- [25] I. Demirdˇzi´c, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255. 

- [26] F. Moukalled, M. Darwish, A High-Resolution Pressure-Based Algorithm for Fluid Flow at All Speeds, Journal of Computational Physics 168 (2001) 101–130. 

- [27] S. Acharya, B. R. Baliga, K. Karki, J. Y. Murthy, C. Prakash, S. P. Vanka, Pressure-Based Finite-Volume Methods in Computational Fluid Dynamics, Journal of Heat Transfer 129 (2007) 407. 

- [28] K. Javadi, M. Darbandi, M. Taeibi-Rahni, Three-dimensional compressible–incompressible turbulent flow simulation using a pressure-based algorithm, Computers & Fluids 37 (2008) 747–766. 

- [29] Y.-Y. Tsui, T.-C. Wu, A Pressure-Based Unstructured-Grid Algorithm Using High-Resolution Schemes for All-Speed Flows, Numerical Heat Transfer, Part B: Fundamentals 53 (2008) 75–96. 

25 

- [30] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, Journal of Computational Physics 228 (2009) 180–201. 

- [31] M. Darwish, F. Moukalled, A fully coupled Navier-Stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444. 

- [32] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. 

- [33] F. Denner, Fully-coupled pressure-based algorithm for compressible flows: Linearisation and iterative solution strategies, Computers & Fluids 175 (2018) 53–65. 

- [34] A. J. Chorin, Numerical solution of the Navier-Stokes equations, Mathematics of Computation 22 (1968) 745–745. 

- [35] J. B. Bell, P. Colella, H. M. Glaz, A second-order projection method for the incompressible Navier-Stokes equations, Journal of Computational Physics 85 (1989) 257–283. 

- [36] S. Patankar, D. Spalding, A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows, International Journal of Heat and Mass Transfer 15 (1972) 1787–1806. 

- [37] S. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere Publishing Company, 1980. 

- [38] R. Issa, Solution of the implicitly discretised fluid flow equations by operator-splitting, Journal of Computational Physics 62 (1985) 40–65. 

- [39] R. Issa, A. Gosman, A. Watkins, The computation of compressible and incompressible recirculating flows by a non-iterative implicit scheme, Journal of Computational Physics 62 (1986) 66–82. 

- [40] R. Kunz, W. Cope, S. Venkateswaran, Development of an implicit method for multi-fluid flow simulations, Journal of Computational Physics 152 (1999) 78–101. 

- [41] B. van Wachem, V. Gopala, A coupled solver approach for multiphase flow calculations on collocated grids, in: European Conference on Computational Fluid Dynamics, ECCOMAS CFD, TU Delft, 2006, pp. 1–16. 

- [42] B. van Wachem, A. Benavides, V. Gopala, A coupled solver approach for multiphase flow problems, in: 6th International Conference on Multiphase Flows 2007, Leipzig, Germany, p. Paper No 183. 

- [43] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130. 

- [44] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acousticallyconservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234. 

- [45] R. M. Beam, R. F. Warming, An Implicit Factored Scheme for the Compressible Navier-Stokes Equations, AIAA Journal 16 (1978) 393–402. 

- [46] R. W. MacCormack, A Numerical Method for Solving the Equations of Compressible Viscous Flow, AIAA Journal 20 (1982) 1275–1281. 

- [47] E. Turkel, R. Radespiel, N. Kroll, Assessment of preconditioning methods for multidimensional aerodynamics, Computers & Fluids 26 (1997) 613–634. 

- [48] S. M. H. Karimian, G. E. Schneider, Pressure-based computational method for compressible and incompressible flows, Journal of Thermophysics and Heat Transfer 8 (1994) 267–274. 

- [49] E. Turkel, A. Fiterman, B. van Leer, Preconditioning and the Limit to the Incompressible Flow Equations, Technical Report, NASA CR-191500, 1993. 

- [50] E. Turkel, Review of preconditioning methods for fluid dynamics, Applied Numerical Mathematics 12 (1993) 257–284. 

- [51] S. Y. Kadioglu, M. Sussman, S. Osher, J. P. Wright, M. Kang, A second order primitive preconditioner for solving all speed multi-phase flows, Journal of Computational Physics 209 (2005) 477–503. 

- [52] E. Turkel, Numerical Methods and Nature, Journal of Scientific Computing 28 (2006) 549–570. 

- [53] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, Journal of Computational Physics 72 (1987) 277–298. 

- [54] F. Xiao, Unified formulation for compressible and incompressible flows by using multi-integrated moments I: One-dimensional inviscid compressible flow, Journal of Computational Physics 195 (2004) 629–654. 

- [55] K. Nerinckx, J. Vierendeels, E. Dick, Mach-uniformity through the coupled pressure and temperature correction algorithm, Journal of Computational Physics 206 (2005) 597–623. 

- [56] F. Xiao, R. Akoh, S. Ii, Unified formulation for compressible and incompressible flows by using multi-integrated moments II: Multi-dimensional version for compressible and incompressible flows, Journal of Computational Physics 213 (2006) 31–56. 

- [57] D. Fuster, S. Popinet, An all-Mach method for the simulation of bubble dynamics problems in the presence of surface tension, Journal of Computational Physics 374 (2018) 752–768. 

- [58] M. V. Kraposhin, M. Banholzer, M. Pfitzner, I. K. Marchevsky, A hybrid pressure-based solver for nonideal single-phase fluid flows at all speeds, International Journal for Numerical Methods in Fluids 88 (2018) 79–99. 

- [59] D. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Computers & Fluids 32 (2003) 1113–1132. 

- [60] C.-D. Munz, S. Roller, R. Klein, K. Geratz, The extension of incompressible flow solvers to the weakly compressible regime, Computers & Fluids 32 (2003) 173–196. 

- [61] J. H. Park, C.-D. Munz, Multiple pressure variables methods for fluid flow at all Mach numbers, International Journal for Numerical Methods in Fluids 49 (2005) 905–931. 

- [62] F. Cordier, P. Degond, A. Kumbaro, An Asymptotic-Preserving all-speed scheme for the Euler and Navier–Stokes equations, Journal of Computational Physics 231 (2012) 5685–5704. 

- [63] Y. Moguen, P. Bruel, E. Dick, A combined momentum-interpolation and advection upstream splitting pressure-correction algorithm for simulation of convective and acoustic transport at all levels of Mach number, Journal of Computational Physics 384 (2019) 16–41. 

- [64] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. 

- [65] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177–208. 

- [66] O. Le M´etayer, J. Massoni, R. Saurel, Elaboration[´] des lois d’´etat d’un liquide et de sa vapeur pour les mod`eles d’´ecoulements diphasiques, International Journal of Thermal Sciences 43 (2004) 265–276. 

- [67] S. Karimian, A. Straatman, Discretization and parallel performance of an unstructured finite volume Navier–Stokes solver, International Journal for Numerical Methods in Fluids 52 (2006) 591–615. 

- [68] F. Denner, B. van Wachem, Corrigendum to “Pressure-based algorithm for compressible interfacial flows with acousticallyconservative interface discretisation” [J. Comput. Phys. 367 (2018) 192–234], Journal of Computational Physics 381 (2019) 290–291. 

26 

- [69] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, Journal of Computational Physics 298 (2015) 466–479. 

- [70] P. Roe, Characteristic-based schemes for the euler equations, Annual Review of Fluid Mechanics 18 (1986) 337–365. 

- [71] J. Ferziger, Interfacial transfer in Tryggvason’s method, International Journal for Numerical Methods in Fluids 41 (2003) 551–560. 

- [72] I. Demirdˇzi´c, A Finite Volume Method for Computation of Fluid Flow in Complex Geometries, Ph.D. thesis, Imperial College London, 1982. 

- [73] S. Mathur, J. Murthy, A pressure-based method for unstructured meshes, Numerical Heat Transfer Part B Fundamentals 31 (1997) 195–215. 

- [74] Y.-Y. Tsui, Y.-F. Pan, A Pressure-Correction Method for Incompressible Flows Using Unstructured Meshes, Numerical Heat Transfer, Part B: Fundamentals 49 (2006) 43–65. 

- [75] S. Balay, W. Gropp, L. C. McInnes, B. F. Smith, Efficient Management of Parallelism in Object Oriented Numerical Software Libraries, in: E. Arge, A. Bruasat, H. Langtangen (Eds.), Modern Software Tools in Scientific Computing, Birkh¨auser Press, 1997, pp. 163–202. 

- [76] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. 

- [77] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. 

- [78] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. 

- [79] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annual Research Briefs, Center for Turbulence (2004) 3–14. 

- [80] P. K. Khosla, S. G. Rubin, A diagonally dominant second-order accurate implicit scheme, Computers & Fluids 2 (1974) 207–209. 

- [81] J. Banks, T. Aslam, W. Rider, On sub-linear convergence for linearly degenerate waves in capturing schemes, Journal of Computational Physics 227 (2008) 6985–7002. 

- [82] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure-velocity coupling allowing acoustic calculation in low Mach number flow, Journal of Computational Physics 231 (2012) 5522–5541. 

- [83] J. D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw-Hill New York, 2003. 

- [84] I. Johnston, The Noble-Abel Equation of State: Thermodynamic Derivations for Ballistics Modelling, Technical Report Technical Report DSTO–TN–0670, Defence Science and Technology Organisation, 2005. 

- [85] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible multicomponent flows, Journal of Computational Physics 274 (2014) 95–121. 

- [86] A. Harten, High Resolution Schemes for Hyperbolic Conservation Laws, Journal of Computational Physics 49 (1983) 357–393. 

- [87] A. Harten, The artificial compression method for computation of shocks and contact discontinuities. I. Single conservation laws, Communications on Pure and Applied Mathematics 30 (1977) 611–638. 

- [88] E. V. Vorozhtsov, N. N. Yanenko, Methods for the Localization of Singularities in Numerical Solutions of Gas Dynamics Problems, Springer Series in Computational Physics, Springer-Verlag, New York, 1990. 

- [89] C. B. Laney, Computational Gasdynamics, Cambridge University Press, Cambridge; New York, NY, 1998. 

- [90] T. Y. Hou, P. G. L. Floch, Why Nonconservative Schemes Converge to Wrong Solutions: Error Analysis, Mathematics of Computation 62 (1994) 497–530. 

- [91] P. Lax, B. Wendroff, Systems of conservation laws, Communications on Pure and Applied Mathematics 13 (1960) 217–237. 

- [92] H. S. G. Swann, The Convergence with Vanishing Viscosity of Nonstationary Navier-Stokes Flow to Ideal Flow in R3, Transactions of the American Mathematical Society 157 (1971) 373. 

- [93] S. Osher, S. Chakravarthy, High resolution schemes and the entropy condition, SIAM Journal on Numerical Analysis 21 (1984) 955–984. 

- [94] Y. Moguen, P. Bruel, E. Dick, Solving low Mach number Riemann problems by a momentum interpolation method, Journal of Computational Physics 298 (2015) 741–746. 

- [95] G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, Journal of Computational Physics 27 (1978) 1–31. 

- [96] M. Malik, J. Dey, M. Alam, Linear stability, transient energy growth, and the role of viscosity stratification in compressible plane Couette flow, Physical Review E 77 (2008). 

- [97] U. Ghia, K. N. Ghia, C. T. Shin, High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method, Journal of Computational Physics 48 (1982) 387–411. 

- [98] Z. Lilek, M. Peric, A fourth-order finite volume method with colocated variable arrangement, Computers and Fluids 24 (1995) 239–525. 

- [99] P. Woodward, P. Colella, The Numerical Simulation of Two-Dimensional Fluid Flow with Strong Shocks, Journal of Computational Physics 173 (1984) 115–173. 

- [100] H. Jasak, Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flow, Ph.D. thesis, Imperial College London, 1996. 

- [101] S. I. Rubinow, J. B. Keller, The transverse force on a spinning sphere moving in a viscous fluid, Journal of Fluid Mechanics 11 (1961) 447–459. 

## **Appendix A. Coefficients of the linear equation system** 

The coefficients of the discretised governing equations, Eqs. (50)-(52), are given below. In order to simplify the presentation, the coefficients are given based on the assumption that cell _P_ is the upwind cell _U_ of face _f_ and using the BDF1 scheme for the discretisation of the transient terms. 

For the discretised continuity equation (50), the pressure coefficients associated with cell _P_ and its neighbour 

27 

cells _Q_ are 

**==> picture [466 x 72] intentionally omitted <==**

respectively. The velocity coefficients, which arise from the implicit treatment of the advecting velocity of the advection term, associated with cell _P_ and its neighbour cells _Q_ , are 

**==> picture [313 x 54] intentionally omitted <==**

respectively. The coefficient of the right-hand side vector, _**σ**[ρ]_ , associated with cell _P_ is given as 

**==> picture [470 x 123] intentionally omitted <==**

where _δf_ = _ξf |_ _**r** P f |/_ ∆ _sf_ is the weighting coefficient that follows from the TVD discretisation of the advection term, see Section 4.3. 

For the discretised momentum equations (51), the pressure coefficients are given as 

**==> picture [416 x 109] intentionally omitted <==**

The coefficients associated with velocity _uj_ are given as 

**==> picture [303 x 55] intentionally omitted <==**

where 

**==> picture [323 x 29] intentionally omitted <==**

is the coefficient arising from the advection of velocity and the implicit velocity contribution of the decomposed shear stress term, which is used for the definition of the advection velocity _ϑf_ , see Section 5.1. The coefficients of the velocity components that arise from the implicit treatment of the advecting velocity of the advection term are 

**==> picture [327 x 25] intentionally omitted <==**

**==> picture [327 x 25] intentionally omitted <==**

28 

The coefficient of the right-hand side subvector _**σ**[ρu][j]_ follows as 

**==> picture [488 x 194] intentionally omitted <==**

The coefficients of the discretised energy equation (52) follow in a similar fashion, with the pressure coefficients given as 

**==> picture [411 x 103] intentionally omitted <==**

the velocity coefficients given as 

**==> picture [444 x 70] intentionally omitted <==**

and the coefficients of the temperature given as 

**==> picture [363 x 68] intentionally omitted <==**

29 

The coefficient of the right-hand side subvector _**σ**[ρh]_ follows as 

**==> picture [489 x 193] intentionally omitted <==**

30 

