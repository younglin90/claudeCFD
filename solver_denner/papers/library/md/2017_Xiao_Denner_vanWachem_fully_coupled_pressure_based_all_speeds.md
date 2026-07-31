Journal of Computational Physics 346 (2017) 91–130


### Contents lists available at ScienceDirect


### www.elsevier.com/locate/jcp


# Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries


# Cheng-Nian Xiao, Fabian Denner, Berend G.M. van Wachem ∗

Department of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom


## a r t i c l e i n f o a b s t r a c t

Article history: Received 25 January 2017 Received in revised form 24 May 2017 Accepted 5 June 2017 Available online 9 June 2017

Keywords: All-speeds flows Fully-coupled framework Unstructured grids Shock-capturing schemes Transient accuracy

A generalized finite-volume framework for the solution of fluid flows at all speeds in complex geometries and on unstructured meshes is presented. Starting from an existing pressure-based and fully-coupled formulation for the solution of incompressible flow equations, the additional implementation of pressure–density–energy coupling as well as shock-capturing leads to a novel solver framework which is capable of handling flows at all speeds, including quasi-incompressible, subsonic, transonic and supersonic flows. The proposed numerical framework features an implicit coupling of pressure and velocity, which improves the numerical stability in the presence of complex sources and/or equations of state, as well as an energy equation discretized in conservative form that ensures an accurate prediction of temperature and Mach number across strong shocks. The framework is verified and validated by a large number of test cases, demonstrating the accurate and robust prediction of steady-state and transient flows in the quasiincompressible as well as subsonic, transonic and supersonic speed regimes on structured and unstructured meshes as well as in complex domains. © 2017 The Authors. Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).


### 1. Introduction

Since the widespread increase and availability of computing power and resources, computational fluid dynamics (CFD) has become a mainstay tool in academic research as well as commercial development for problems related to fluid dynamics. Thus, the demand for numerical frameworks able to simulate various complex flows has also become more pronounced. One well-known flow configuration that has proved to be problematic for most state-of-the-art numerical frameworks arises in the simultaneous presence of multiple flow speed regimes, ranging from the incompressible limit to supersonic flows. A variety of numerical frameworks for specific flow regimes, dealing with either high-speed or nearly incompressible flows, have been previously proposed in the literature [1–4]. However, each of them has its shortcomings in the flow speed regime for which they are not designed.

In the quasi-incompressible speed regime, i.e. for Mach numbers M = |u|/|c| ≪0.1 throughout, where u is the local flow velocity and c is the local speed of sound, the flow can be considered as (nearly) incompressible, i.e. the density, ρ of each fluid particle remains (almost) constant, Dρ/Dt = 0 [5]. Thus, the fluid density ρ is either decoupled or, at best, weakly coupled to velocity and pressure. The most common numerical frameworks specialized to these types of flows reflect this property by solving the incompressible flow equations in a decoupled manner [1,4]. These methods generally apply a segregated pressure-correction approach, where the velocity is first predicted by the momentum equations using a prelim-


# * Corresponding author. E-mail address: berend.van.wachem@gmail.com (B.G.M. van Wachem).

http://dx.doi.org/10.1016/j.jcp.2017.06.009 0021-9991/© 2017 The Authors. Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

92 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130

inary estimate of the pressure field, then both pressure and velocity are corrected to ensure the divergence-free condition as dictated by the continuity equation. The correction step is accomplished by the solution of a reformulated continuity equation as a Poisson equation for the pressure correction [1,6]. Different variants of this approach are the marker-and-cell (MAC) method, projection or fractional step methods, and the family of semi-implicit method for pressure linked equations (SIMPLE) [1,4,7]. Recently, a number of studies reported coupled numerical frameworks, notably [8–12], where the primary variables (velocity and pressure) are solved in a single linear system of equations. The implicit coupling of pressure and velocity is typically associated with an increased robustness and numerical stability for complex flow configurations, for instance flows with large density and viscosity discontinuities, flows in complex domains or in the presence of large source terms.

At high flow speeds with M > 0.2 the compressible nature of the flow becomes noticeable, since convective effects of the fluid flow can effectively compete with acoustic waves at the same time scale. Thus, the fluid motion itself has a significant effect on the distribution of density and other thermodynamic variables inside the flow field [13]. The most established algorithms for high speed flows are density-based, i.e. they are based on the algebraic equivalents of the governing equations in which the fluid density ρ is one of the unknowns to be solved for [1]. The governing equations for this approach are cast in form of a coupled nonlinear system of conservation laws, where the unknown variables are the density, momentum components as well as total energy or enthalpy. The main strengths of this approach derives from the large body of knowledge in applied mathematics relating to hyperbolic equation systems which encompass the governing equations for inviscid flows [14,15]. Some of the most well-known representatives of density-based numerical frameworks for compressible flows are the MacCormack scheme, the Beam–Warming scheme, and the Jameson–Schmidt–Turkel (JST) scheme [2]. The interested reader is referred to the textbooks of Anderson et al. [1] and Wesseling [2] for a detailed account of these and other density-based frameworks. However, the major drawback of the density-based formulation lies in the requirement of a strong coupling between density ρ and pressure p. This condition is violated for the simulation of flows at low Mach numbers (M < 0.1) if the chosen numerical time-step is adapted to the convective time scale of the flow and thus too large to resolve acoustic waves. Hence, such cases become numerically ill-conditioned for density-based frameworks [1,2].

More recently, a number of numerical frameworks that are suitable for all flow speeds have been proposed by extending solution methods originally designed for incompressible or compressible flows [16–18]. The most often used strategy to stabilize density-based solvers at the low-speed limit is preconditioning [19] or artificial compressibility [20]. These methods have, however, major drawbacks. Firstly, they require an additional numerical parameter which has to be readjusted for each individual flow case and, secondly, they are not able or too inefficient to reproduce correct transient results. Even with the addition of the most sophisticated techniques, density-based frameworks tend to be less efficient than pressurebased frameworks for (nearly) incompressible flows [2]. More recently, an adapted pressure-correction strategy based on the governing equations for compressible flows has been proposed by Bijl and Wesseling [18], which appears to be promising at addressing the aforementioned issues. However, it is predicated on the use of a non-conservative version of the energy equation as well as the segregated solution sequence which decouples momentum equations from the continuity equation.

Extensions of pressure-based frameworks designed for incompressible flows to the compressible regime is typically based on a variant of the SIMPLE or PISO (Pressure Implicit with Splitting of Operators) family of algorithms which solve a pressure-correction equation transformed from the continuity equation separately from the momentum equations [7,16–18, 21,22]. Demirdži´c et al. [16] proposed a form of pressure–density coupling in the continuity equation that is applied at the pressure-correction step, which was later improved and extended in several other studies [17,23]. Although the extension of pressure-based schemes are more efficient at low-speed flows than the density-based methods, the segregated solution algorithm as well as the use of pressure and velocity instead of density and momentum as the conserved quantities are less amenable to the well-established theory of hyperbolic equation systems that is typically the basis for fully-coupled densitybased frameworks. As a consequence, it is not straightforward to formally derive accurate shock-capturing schemes required at supersonic speeds within pressure-based frameworks. Wesseling and co-workers [2,18] presented simulation results for the shock tube problem using an all-speeds pressure-correction framework, which however uses a non-conservative form of the energy equation. This appears to cause inherent density and Mach number overshoots in the presence of strong shocks which do not vanish with mesh refinement [2]. Indeed, non-conservative formulations of the energy equation in terms of thermal energy, enthalpy or temperature appear to be commonly used for transient, pressure-based all-speeds frameworks [2,16–18,24]. Hence, it is to be expected that these numerical frameworks suffer from the same drawbacks as reported in [2] where significant overshoots have been observed when predicting density or related thermodynamic variables at strong shocks. The extension of the fully-coupled framework, which solves for the static pressure simultaneously with other flow variables, has been proposed by Chen and Przekwas [10] and Darwish and Moukalled [11].

However, none of the frameworks applicable to all speed regimes as presented above have demonstrated the capability to correctly predict transient flow behaviour at all Mach numbers. In fact, simulation results of pressure-based solvers for transient compressible flows are much more rare to find when compared to steady-state flow predictions used as benchmark cases in almost all published works. In some cases, transient flows have not been studied at all as the governing equations are formulated and solved in steady-state form, see for instance [7,23,25,26]. Transient simulation results for subsonic flows have been presented by Issa et al. [27] and Chen and Pletcher [28], whereas simulations of acoustic propagation via pressure-based frameworks have been reported by Moguen et al. [29]. Simulation results for shock tube problems using pressure-based formulations can be found in [2,18]. More recently, solver algorithms combining fully coupled density-based schemes with pressure-correction strategies derived from pressure-based frameworks have been developed and successfully

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 93

tested on transient compressible flow problems at both high and low Mach numbers [30–32]. In [30], an additional pressurecorrection step operating on a staggered grid, as used in the MAC family of methods, is applied after solving the coupled compressible flow equations via traditional density-based schemes, and the pressure field is computed by the pressureupdate equations rather than from the equation of state. This approach does indeed produce accurate results for a variety of transient problems at different flow speeds, and it has been applied to flow problems on more complicated geometries via the Chimera grid framework based on a multitude of overlapping Cartesian grids [31]. However, this particular strategy has been derived based on structured Cartesian grids and has not been tested on unstructured grids. Formulations applicable to staggered unstructured grid systems have been given in [32,33]. In [32], a discontinuous Galerkin (DG) method is introduced and demonstrated to be capable of solving transient flow problems at both high and low Mach numbers. However, no steady supersonic flow problems at supersonic Mach numbers have been presented in [32,33]. In addition, flow solver algorithms operating on staggered unstructured grids such as in [32] are known to be cumbersome and are associated with a substantial overhead in complexity, especially for non-symmetric geometries in multiple spatial dimensions. Other potential drawbacks, flow solvers based on staggered unstructured grids are less frequently employed than solvers applicable for collocated unstructured grids. This lack of in-depth discussions in the present literature on the ability of pressure-based formulations suitable on collocated unstructured grids to accurately predict transient flows at all speed regimes involving compressibility effects such as acoustic and shock waves suggests that it is a subject worthy of further investigations.

In this work, a fully-coupled pressure-based framework in finite-volume formulation suitable for the simulation of flows at all speeds, from quasi-incompressible low Mach number flows to supersonic flows, applicable in complex geometries and on unstructured meshes, is proposed. The presented numerical framework is based on the fully-coupled finite-volume framework for incompressible flows on unstructured meshes proposed by Denner and van Wachem [12]. This numerical framework is based on a collocated variable arrangement and features an implicit coupling of pressure and velocity, which does not require under-relaxation to achieve convergence. In this study, the numerical framework of Denner and van Wachem [12] is extended to accurately compute transient and steady compressible flows at all speeds, including low Mach number acoustics and shock tube problems. To achieve that goal, the unsteady energy equation in conservative form is related to the fully coupled momentum and continuity equations which ensure strong velocity–pressure and pressure–density coupling. The presented results demonstrate the accurate simulation of flows at all speeds, including acoustics at low Mach number and transonic/supersonic flows with complex shock structures. Even flows containing a mix of vastly different speed regimes are shown to be accurately and robustly predicted by the proposed framework. The results presented in this work, in particular the cases involving flows with transient behaviour, can also serve as a reference for future studies. To the best of the authors’ knowledge, the numerical framework dealing with compressible flows at all speeds introduced in this work is unique in its kind which combines all of the following characteristics: Firstly, it is derived from a fully-coupled, pressurebased formulation for incompressible flows, which does not involve operator splitting or any other additional intermediate steps between solving the linearized and discretized Navier–Stokes equations. Secondly, its formulation is based on a collocated variable arrangement on unstructured grids, hence facilitating the application in complex flow geometries. Thirdly, it is shown to be capable of accurately predicting both steady as well as transient flow problems on a wide range of Mach numbers, including compressible flows at the incompressible limit and hypersonic flows. The proposed framework is also noteworthy for its complete lack of any artificial numerical parameters such as under-relaxation factor that are required for convergence. It also ensures a seamless transition from solving flow problems at different Mach numbers by the imposition of the same CFL time step condition for flows at all speeds.

In Section 2, the governing equations are discussed. The numerical framework and the discretization of the governing equations are presented and discussed in Section 3. Section 4 presents the results for a range of representative transient flows and results for steady-state test cases commonly used as benchmark for compressible flows are discussed in Section 5. The results are summarized and the article is concluded in Section 6.


### 2. Governing equations

The system of governing flow equations, i.e. the momentum equations, the continuity equation as well as constitutive relations between the flow quantities, describe the flow motions of continuous fluid media. Let ρ, p, u, σ, S represent density, pressure, velocity, Cauchy stress tensor and source terms, respectively. The continuity and momentum equations in conservative form are


# ∂ρ


# ∂t + ∇· (ρu) = 0 (1)


# ∂(ρu)


## ∂t + ∇· (ρuu) = ∇· σ + S . (2)


### For Newtonian fluids, the Cauchy stress tensor is given as


# σ = σ(u, p) = − �


## p + 2


# 3μ∇· u �


# I + μ(∇u + (∇u)T ) = −pI + τ . (3)

94 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 1. Mesh cell center P with neighbour cell centers Q and Q ′, as well as the face center f shared between cell centers P and Q of a rectilinear two-dimensional mesh.**


### In order to correctly model the full physics of compressible flows, it is also necessary to include the energy equation describing the coupling between the flow and thermal quantities. Let ht = h + 1


## 2∥u∥2 denote the total enthalpy, i.e. the sum of a fluid particle’s static enthalpy h and kinetic energy, then the energy equation is given as


# ∂(ρht)


## ∂t + ∇· (ρhtu) = ∂p


# ∂t + ∇· (τ · u) −∇· ˙q , (4)


# where ˙q = −κ∇T is the heat flux that can be computed from the temperature field T and local thermal conductivity κ via Fourier’s law.

To close the system of governing equations, an equation of state (EOS) that couples pressure and density and a relationship defining the static enthalpy are required. For ideal gases, the static enthalpy is given by h = cpT , and the EOS linking pressure with density and temperature is given as


# p = ρRT , (5)


### where R is the specific gas constant.


### 3. Numerical framework

Let φ be a generic scalar, u denote the local fluid velocity vector and ρ be the fluid density, with u and ρ assumed to be known flow quantities. The convection-diffusion equation for φ in integral form is then given by


## �

V


# ∂(ρφ)


## ∂t dV + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq001.png)


## � �� � Fconv


## = �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq002.png)


## �∇φ · dA


## � �� � Fdiff


## +Sφ , (6)

which is the mathematical expression of the fact that the rate of change for φ within any control volume V is the sum of the convective flux Fconv and diffusive flux Fdiff across boundary surface ∂V as well as additional sources Sφ. It can be seen that by substituting φ with any flow velocity component ui or the total enthalpy ht and choosing the appropriate source terms for Sφ, the momentum and energy equations of the system of governing flow equations are obtained, respectively. In the following, a control volume (or cell) centered around point P with volume V P , is considered, shown schematically in Fig. 1, where n f and A f are the normal vector and area of each face f of the cell. We assume that the density ρ f at each face is obtained by a higher-order interpolation method from the known densities at cell centers.


### 3.1. Discretization of convective fluxes


### The discretized form of the convective contribution as specified in Eq. (6) is given as


## Fconv ≈ �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq003.png)

where f runs through all faces of the control volume centered around point P and A f is the surface area of face f . In order to obtain an algebraic expression for φ f which only contains the cell-centered values of φ, φ f at face f is obtained by interpolation from the adjacent cell centers P and Q as


# φ f = αconv P, f φP + αconv Q , f φQ , (8)

where the coefficients αconv P, f and αconv Q , f depend on the chosen interpolation method as well as the local geometry at face f . A simplified sketch of the geometry at a cell face with the relevant neighbouring points is shown in Fig. 1. After substituting Eq. (8) into Eq. (7), the discretized convective flux can be written in the canonical form

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 95


## Fconv ≈ �

f ∈faces(P) Q =Q (P, f )


# ρ f u f · n f A f � αconv P, f φP + αconv Q , f φQ � , (9)


## involving cell-centered values φP and φQ only.

There are several possible interpolation methods to express the face center value φ f in terms of the cell-centered values, each having its own advantages and disadvantages. Different schemes are considered in this study to determine the interpolation coefficients αconv P, f and αconv Q , f . In the following, let |P f |, |Q f | denote the distances between the cell centers P, Q and face center f , and |P Q | be the distance between the cell centers P and Q . The second-order central differencing scheme is defined as


## φ f = |Q f |


## |P Q | ���� αcentr P, f


## φP + |P f |


## |P Q | ���� αcentr Q , f


## φQ (10)


### and the first-order upwind scheme is given as


## φ f = max(sgn(n f · u), 0) � �� � αupw P, f


## ·φP + (1 −max(sgn(n f · u), 0)) � �� � αupw Q , f


## ·φQ . (11)

The central differencing scheme, Eq. (10) provides second-order accuracy but is susceptible to numerical instability, whereas the first-order upwind scheme, Eq. (11), is numerically stable but introduces a considerable amount of numerical diffusion that negatively impacts the spatial accuracy of the results [6].

More recently, hybrid or nonlinear schemes, such as flux-corrected (FCT) or total-variation-diminishing (TVD) schemes, are usually applied for the simulation of complex flows, for instance transonic flows with shocks, to optimize both accuracy and stability depending on the local flow field [6]. Interpolation by means of the general class of TVD schemes is, considering the flow direction (direction of u) as indicated in Fig. 1, given for a general unstructured mesh as [34,35]


## φ f = �


## 1 −ψ(r f )


## L f


## �


## � �� � αTVD P, f


## φP + ψ(r f )


## L f ���� αTVD Q , f


## φQ , (12)


## where ψ is the flux limiter determined by a suitable TVD scheme [6], the gradient ratio r f is defined as (with the help of an additional neighbour point Q’ upstream of P, cf. Fig. 1)


## r f = φP −φQ ′


## φQ −φP (13)


### and the geometry coefficient L f is readily computed as [34]


## L f = |P f | + |Q f |


## |P f | . (14)

In this study the Minmod scheme, ψ(r f ) = max{0, min(1, r f )}, and van Leer scheme ψ(r f ) = (0.5L f r f + 0.5L f |r f |)/(L f −1 + |r f |), are considered to compute the flux limiter ψ(r f ) [34]. Furthermore, the monotonicity-preserving skewness correction proposed by Denner and van Wachem [35] is applied for simulations on unstructured meshes to reduce numerical diffusion induced by the mesh topology.


### 3.2. Discretization of diffusive fluxes


### The discretized form of the diffusive flux as specified in Eq. (6) is approximated by the equation


## Fdiff ≈� �

f ∈faces(P) ∇φ f · n f A f = � �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq004.png)


## ∂φ f ∂n f A f . (15)

To obtain an algebraic expression in terms of the cell-centered quantities of φ, the directional derivative ∂φ f /∂n f at the face center f is evaluated using the standard second-order central-differencing approximation


## ∂φ f ∂n f ≈φQ −φP


## |P Q | = αdiff P, f φP + αdiff Q , f φP (16)


## for the simple case shown in Fig. 1, where the face normal vector n f is parallel with the vector −→ P Q connecting cell centers


## P and Q . In case of non-orthogonality of the mesh, i.e. when n f and −→ P Q are not parallel, the deferred correction proposed by Demirdži´c and Muzaferija [36] is applied.

96 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


### 3.3. Temporal discretization

The transient contribution in Eq. (6) is a volume integral that describes the total change of the scalar φ within a specified mesh cell volume V around point P with volume V P . It can be approximated as


## �

V


# ∂(ρφ)


## ∂t dV ≈∂(ρφ)


## ∂t


## ���� P · V P . (17)

In this study, the transient term is discretized using the first-order backward Euler scheme and the second-order backward Euler scheme [3]. Both of these schemes belong to the general class of linear multistep schemes, which can be written in the general form


# ∂(ρφ)


## ∂t


## ���� P ≈

m �

i=0


# γi(ρφ)n−i P �t (18)

where m is the order of the scheme, γi are time level weighting coefficients defined by the chosen discretization method and superscripts denote the time level at which φ is evaluated, n being the most current one.


### 3.4. Discretized convective–diffusion equation


### After employing the spatial and temporal discretizations as discussed in Sections 3.1-3.3, the fully discretized convection– diffusion equation for mesh cell P is given as

2 �

i=0


# V Pγi(ρφ)n−i P �t + �

f ∈faces(P) Q =Q (P, f )


# ρ f uC f · n f A f � αconv P, f φP + αconv Q , f φQ � = �

f ∈faces(P) Q =Q (P, f )


## A f � αdiff P, f φP + αdiff Q , f φQ � + Sn ,


## (19)


### where uC f denotes the convective face velocity (see Section 3.5.2 for details) and Sn denotes the source term at time level n.


### 3.5. Discretization of governing flow equations

In this section, the discretization procedure outlined in the previous section are applied to the system of flow equations. In contrast to the convection–diffusion equations for passive scalars, the system of governing flow equations are inherently nonlinear, thus additional treatments are required to linearize the nonlinear terms. For a clear illustration of the linearization process for the discretized equations, the notation adopted throughout this section designates unknown variables to be solved for with superscript n, i.e. at the most current time level n, whereas variables with superscript such as n −1 or without subscripts are assumed to be known, either from a previous time step as indicated by the superscript or from an available estimate (e.g. value from the most recent iteration).


### 3.5.1. Momentum equations


### The momentum equation for each Cartesian component j in integral form is given by


## �

V


# ∂(ρu j)


## ∂t dV + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq005.png)


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq006.png)


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq007.png)


## pe j · dA + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq008.png)


## � μ∇ju −2


# 3μ(∇· u)e j


## � · dA (20)

where e j denotes the j-th Cartesian unit vector. This is the same form as the convection–diffusion equation given by Eq. (6) where φ = u j := u · e j are the Cartesian velocity components and the source term Sφ = − �


## ∂V pe j · dA + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq009.png)


# 3μ(∇· u)e j � · dA is the sum of the pressure, as well as the parts of the shear stresses representing cross-


### diffusion and bulk viscosity.


### Each of the component momentum equations is then discretised and linearized as

2 �

i=0


# V Pγiρn−i P un−i j,P �t + �

f ∈faces(P) Q =Q (P, f )


# ρ f uC f · n f A f � αconv P, f un j,P + αconv Q , f un j,Q � = �

f ∈faces(P) Q =Q (P, f )


## A f � αdiff P, f un j,P + αdiff Q , f un j,Q �


## − �

f ∈faces(P) Q =Q (P, f )


## A f n f · e j � αcentr P, f pn P + αcentr Q , f pn Q � + μ �

f ∈faces(P) Q =Q (P, f )


## � (∇ju) f · n f −2


## 3(∇· u)e j


## �


## A f (21)

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 97


### where the only unknowns are the cell-centered values of velocity component un j and pressure pn. To achieve that, the most

recently available density value ρP is used in lieu of ρn P in the transient term of Eq. (21). As indicated by the superscript n, the pressure contribution is taken implicitly at time level n, at the same time level as the unknown velocity un j, in order to ensure a strong coupling between velocity and pressure. The cross-diffusion and bulk viscosity terms of the shear stresses are evaluated explicitly using the most recently available velocity estimates. The necessary interpolation of the pressure value at face centers f is achieved by second-order central differencing, as given by Eq. (10). It should be noted that in a collocated variable arrangement, both pressure p and velocity u are stored at cell centers of the mesh cells. The convective terms involving products of velocity components are linearized by explicitly evaluating the face velocity uC f from the most recently available estimate of the flow field, to be described in more detail in Section 3.5.2.


### 3.5.2. Continuity equation

In a pressure-based framework, the continuity equation requires special consideration since it has to be formulated as an equation for pressure rather than density. The continuity equation in integral formulation is given as


## �

V


# ∂ρ


## ∂t dV + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq010.png)


## ∂t


## ���� P · V P + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq011.png)

The first step towards a pressure-based formulation of the continuity equation is to express density ρ in Eq. (22) as a function of pressure. This is readily accomplished with the help of the EOS of the gas, for instance assuming


# ρ = k(p, T(ht, u)) · p (23)


### where ht is the total enthalpy and T is the static temperature. For (calorically) perfect gases, the above relations are given as


## T = c−1 p


## �


## ht −1


## 2|u|2 � , k = k(p, T) = 1


## R T (24)


### with specific heat capacity cp and gas constant R being fluid properties. Obviously, Eq. (23) also holds for other gas models if k is chosen appropriately.

The next step in the transformation is to further couple density with other flow variables, thus accounting for significant density changes due to compressibility. This is achieved via a Newton-linearization of the mass flux ρn f uC,n f · n f at face f :


# ρn f uC,n f · n f ≈ρ f uC,n f · n f + k(p, T) uC f · n f pn f −ρ f uC f · n f . (25)

The last remaining step is to derive an expression for the face velocity magnitude uC f · n f in terms of fluid velocity u and pressure p. This is particularly important for numerical frameworks with collocated variable arrangements in order to achieve a robust pressure–velocity coupling and to prevent spurious pressure oscillations [4]. The face velocity uC f · n f is discretized by applying the momentum-weighted interpolation approach (MWI), which was first proposed by Rhie and Chow [37] and subsequently extended and improved, see e.g. [10,12,38]. Two variants of the aforementioned MWI are considered for the proposed numerical framework (as before, the local geometry at cell volume V centered around P as shown in Fig. 1 is being considered for the notations used below):


### 1. Steady/non-transient MWI


## uC,n f · n f = ¯un f · n f −ˆd f � (∇pn) f −(∇p) f � · n f


## = �|Q f | |P Q | un P + |P f |


## |P Q | un Q


## � · n f −ˆd f pn Q −pn p |P Q | + ˆd f ρharm. f 2


## �(∇p)P ρP + (∇p)Q


# ρQ


## � · n f (26)


### 2. Transient MWI


## uC,n f · n f = ¯un f · n f −ˆd f � (∇pn) f −(∇p) f � · n f + ˆd f ρharm. f �t


## � uC,n−1 f −¯un−1 f � · n f


## = �|Q f | |P Q | un P + |P f |


## |P Q | un Q


## � · n f −ˆd f pn Q −pn p |P Q | + ˆd f ρharm. f 2


## �(∇p)P ρP + (∇p)Q


# ρQ


## � · n f


## + ˆd f ρharm. f �t


## �


## uC,n−1 f −|Q f |


## |P Q | un−1 P −|P f |


## |P Q | un−1 Q


## � · n f . (27)

98 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130

The overbar denotes interpolation of the underlying quantity to face f from the values at the adjacent cell centers. Whereas the face-centered velocity, ¯u f , is evaluated via linear interpolation from the cell-centered velocities, the pressure gradient at face center, (∇p) f , is obtained by one-half weighting of the adjacent cell-centered pressure gradients [12]. ρharm. f =

2/(ρ−1 P + ρ−1 Q ) is the density at face f interpolated by harmonic averaging. The density-weighting of the interpolated pressure terms using harmonic averaging, proposed by Denner and van Wachem [12], provides an improved convergence behaviour for large density jumps, as previously demonstrated for two-phase flows with density ratios of up to 1024 [12, 39]. The coefficient ˆd f is determined based on the coefficients of the momentum equations as derived by Denner and van

Wachem [12]. It can be seen that uC,n f is obtained as the sum of the interpolated face-center velocities un f and an additional term that can be summarized as a higher-order pressure term. It should be pointed out that although the transient MWI given in Eq. (27), containing the extra transient term, can be derived consistently by interpolating momentum equations at adjacent cell centers to the face center as shown by Denner and van Wachem [12], Eq. (26) without the transient term is regularly used in literature, see for instance [11,37]. On meshes with significant non-orthogonality, the deferred correction of the pressure terms in the MWI is applied as proposed by Zwart [40] and detailed in Denner and van Wachem [12], which is derived from the deferred correction of Demirdži´c and Muzaferija [36] for the discretization of diffusion terms.

Substituting the linearized mass flux given in Eq. (25) and either Eq. (26) or Eq. (27) into the continuity equation as given by Eq. (22), and replacing density ρ in the transient term with pressure p via Eq. (23), the pressure-based continuity equation follows as


## ∂ �p


## RT


## �


## ∂t


## ������� P


## · V P


## � �� �

transient term (compressible)


## + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq012.png)


## ⎧ ⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎩


# ρ f A f ¯un f · n f � �� �

velocity coupling (incompressible)


## −ˆd f A f � (∇pn) f −(∇p) f � · n f � �� �

Laplacian term (incompressible)


## + A f uC f · n f


## RT f pn f � �� �

convective pressure term (compressible)


## ⎫ ⎪⎪⎪⎪⎪⎪⎬


## ⎪⎪⎪⎪⎪⎪⎭


## = �

f ∈faces(P) ρ f A f � uC f −�un−1 f � · n f , (28)


## where �un−1 f = (ˆd f ρharm. f )/�t � uC,n−1 f −¯un−1 f � if the transient MWI, see Eq. (27), for uC f is chosen and �un−1 f = 0 other-

wise. The transient term of Eq. (28) is discretized via the same temporal scheme as applied in the momentum equations, given in Eq. (21), while using the most recent estimate T P instead of T n P for the temperature value at time level n such that the pressure value pn P at time level n is the only unknown variable. The face velocity ¯un f is approximated via linear interpolation, and the convective pressure term is evaluated in the same way as the velocity term in the convective contribution of the momentum equations, see Eq. (9).

The most distinctive feature of Eq. (28) is the fact that it is no longer a purely elliptic equation for pressure as in the case of incompressible flows, due to the addition of the convective pressure term that reflects the presence of pressure waves due to compressibility and the hyperbolic nature of compressible flow equations. The strength of the compressibility is, hence, given by the ratio between the coefficients of the (compressible) convective term, which is proportional to |uC f |/(RT f ), and

the coefficient of the (incompressible) Laplacian term for pressure pn, which is proportional to 1/|uC f | if the coefficient ˆd f is derived from the momentum equations [12]. Hence, the local Mach number, M f ∼uC f /�

RT f , determines the relative strength of the compressibility term. Thus, at very small Mach numbers, the continuity equation approximates the Poisson pressure-equation for incompressible flows since the Laplacian term is dominant, whereas at higher Mach numbers the convective term dominates the incompressible pressure terms as a reflection of the compressibility of the flow.


### 3.5.3. Energy equation


## The energy equation for total enthalpy ht = h + 1


## 2|u|2 in integral form is given as


## �

V


# ∂(ρht)


## ∂t dV + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq013.png)

V


## ∂p


## ∂t dV + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq014.png)


# (τ · u) · dA + �


![Equation](images/2017_Xiao_Denner_vanWachem_fully_coupled_pressure_based_all_speeds_eq015.png)

where τ is the viscous part of the stress tensor, see Eq. (3), and κ is the thermal conductivity. The discretization of Eq. (29) follows the same principles as applied to discretize the momentum equations, i.e. the enthalpy in the transient and convective terms are discretized in the same fashion as the corresponding velocity terms in Eq. (20). All source terms on the right hand side are evaluated explicitly; the viscous fluxes are approximated via central differencing and the transient pressure term is discretized with the same temporal scheme as all other transient terms. The resulting discretized energy equation is then given by

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 99

2 �

i=0


# V Pγiρn−i P hn−i t,P �t + �

f ∈face(P) Q =Q (P, f )


# ρ f uC f · n f A f � αconv P, f hn t,P + αconv Q , f hn t,Q �


## =

2 �

i=0


# V Pγi pn−i P �t + �

f ∈face(P) Q =Q (P, f )


# (τ · u) f · n f A f + κ �

f ∈face(P) Q =Q (P, f )


## (∇T) f · n f A f , (30)

which is an equation for the cell-centered values of the total enthalpy ht at time level n. As for the momentum equations (see Section 3.5.1), the density value ρP instead of ρn P is used at time level n in the discretized transient term containing the total enthalpy ht, whereas the most recent estimate pP for pressure is used at time level n in the discretization of the transient pressure term.


### 3.6. Solution procedure

After carrying out the discretization and linearization process as described in the previous sections, the momentum, continuity and energy equations are given by Eqs. (21), (28) and (30). This results in a linear algebraic equation system for the solution variable vector (un j,p, pn P) containing the velocity components and pressure at the most recent time level n located at the cell centers of the computational mesh as well as a separate linear equation system with the cell-centered total enthalpy hn t,P as unknowns. These equation systems are solved by using the Jacobi-preconditioner and the biconjugate gradient stabilized solver (BICGSTAB) of the PETSc linear algebra library [41,42]. By rearranging the terms in those equations, the discretized momentum and continuity equations for three spatial dimensions are given in matrix form as


## ⎡


## ⎢⎢⎢⎣


## Au1 u1 0 0 Ap u1 0 Au2 u2 0 Ap u2 0 0 Au3 u3 Ap u3 Au1 p Au2 p Au3 p Ap p


## ⎤


## ⎥⎥⎥⎦×


## ⎡


## ⎢⎢⎢⎣


## un 1 un 2 un 3 pn p


## ⎤


## ⎥⎥⎥⎦=


## ⎡


## ⎢⎢⎣


## Su1


## Su2


## Su3


## S p


## ⎤


## ⎥⎥⎦ (31)


### and the discretized energy equation is


## � Aht ht


## � × � hn t � = � Sht � . (32)

In the matrices above, the block entries of the form Aψ φ , with φ, ψ ∈{u1, u2, u3, p, ht}, are matrix coefficients which couple the solution variable ψ to φ. It can be seen from inspection of Eqs. (21), (28) and (30) that the velocities un j and pressure pn

are not directly coupled to the total enthalpy hn t provided all terms on the right hand side of the discretized total enthalpy equation, Eq. (30), are evaluated explicitly. In each momentum equation, Eq. (20), the corresponding velocity component is coupled to the pressure and in the continuity equation, Eq. (22), all velocity components are coupled to the pressure. This suggests a solution strategy that separates the solution for (un P, pn P) and for (hn t,P), as shown in Fig. 2. The solution sequence is organized into two iterative loops, as seen in Fig. 2: a) the inner loop and b) the outer loop. In the inner loop, also referred to as (constant) temperature cycle, new values of pressure and velocity are computed while keeping the temperature and total enthalpy unchanged. After each update of u and p, the density ρ is updated based on the new pressure value. Once convergence of u and p is achieved, the algorithm enters the outer loop, which only consists of solving the energy equation for ht, using converged pressure and velocity values from the inner loop. The newly obtained values for ht are then used to update the temperature T and the density ρ. This iterative procedure continues until all solution variables have converged to a predefined solution tolerance or a pre-specified maximal number Nmax of temperature cycles has been reached.

The advantage of this solution strategy is that it separates the coupling of both pressure and temperature to density via two different iteration loops; within each iteration loop only one of the thermodynamic variables is updated. This partially segregated approach does not require any numerical regularization technique, such as under-relaxation, and exhibits an enhanced convergence behaviour compared to a simultaneous solution of enthalpy, pressure and velocity, see e.g. [43], i.e. the density becomes a function of multiple updated flow variables at each iteration. In the latter case, simulations may become unstable in the presence of shocks and under-relaxation must be applied to ensure convergence, due to the strongly nonlinear coupling of total enthalpy to other flow variables as pointed out by Birkby [43].


### 4. Transient test cases

In this section, simulation results for transient compressible flows at different speeds are presented. The accurate prediction of transient flows at all speeds using the same numerical framework is a nontrivial problem and has yet to be

100 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 2. Solution sequence for the discretized system of governing flow equations.**

demonstrated in the literature. Numerical frameworks primarily designed for steady-state problems do not necessarily reproduce the correct transient flow history, even when solving the transient form of the governing equations. For example, the artificial compressibility method and preconditioned density-based frameworks without dual time-stepping do not have temporal accuracy for flows at the incompressible limit [2,6]. For pressure-based incompressible frameworks using collocated variable storage, it has been reported that any ill-suited method of face-velocity interpolation used in the continuity equation (see Section 3.5.2) may have a detrimental effect on the prediction accuracy of transient flows [38]. Transient inaccuracy has also been observed for pressure-based compressible flow solvers that employ a temporally inconsistent method for pressure-velocity coupling [29].

In the following, two classes of transient inviscid flows are considered: a) acoustic propagation at low Mach numbers [29], i.e. at the incompressible limit, and b) Riemann-problems with discontinuous initial data [2], i.e. shock waves involving transonic or supersonic flow speeds. The main focus is to study the influence of various options for face velocity interpolation in the discretized continuity equation, as described in Section 3.5.2, on the transient accuracy of the results.

The applied numerical time-step is evaluated based on the relevant CFL numbers within the simulation domain. Let �x be the local mesh cell length along the direction of the flow velocity vector. For a point within an unstructured grid cell, �x is defined as the minimum of the distances between the cell center and the cell faces. Two different types of the local CFL number will be considered in this work, namely the acoustic CFL, which is defined by CFLacoustic = (|u| + |c|) · �t/�x, where |c| is the (local) speed of sound, and the convective CFL, given by CFLconv = |u| · �t/�x.

For subsonic flows near the incompressible limit, the acoustic CFL number is orders of magnitudes smaller than the convective CFL number. Hence, density-based frameworks, which generally have to satisfy the acoustic CFL limit, are subject to much more restrictive time-steps than pressure-based frameworks, which have to satisfy only the convective CFL condition. At increasing flow speeds in the transonic and supersonic regimes, the two CFL numbers gradually approach each other and eventually for supersonic/hypersonic flows they become almost equal.

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 101


> **Fig. 3. Simulation of a monochromatic acoustic wave within low-speed flow: plots of pressure disturbance dP inside the channel at t = 0.002 s; comparison of results applying Eq. (26) and Eq. (27) for face velocity interpolation with analytic solution from linear acoustics.**


### 4.1. Low-Mach number acoustics

Following the work of [29], the transient flow field caused by a perturbation imposed on a low-speed constant flow is simulated. Since the magnitude of the perturbation is many magnitudes smaller than the steady flow values, linear acoustic theory can be applied to obtain exact reference solutions for this type of problem [13]. The computational domain is a rectangular channel of length 1 m, represented by a computational mesh of 500 cells in one direction only. Thus, the flow geometry is one-dimensional, where the direction of flow aligns with the x-axis and the longitudinal extent of the channel. The fluid has a density of ρ0 = 1.2046 kg m−3 and the steady flow has a velocity in positive x-direction of u0 = 0.30886 m s−1, a pressure of p0 = 1.013 × 105 Pa and a temperature of T0 = p0/(R ρ0), where R = 300 J kg−1 K−1 is the gas constant.


### 4.1.1. Excitation by a single frequency

The flow field is initialized to the steady-state values ρ0, p0, T0 as given above. At the inlet, an oscillating velocity with a frequency f = 2000 Hz and amplitude �u = 0.01u0 is imposed, i.e. the velocity boundary condition at the domain inlet is given by uin(t) = u0 + �u sin(2π f t). Pressure and temperature are extrapolated to the domain inlet from the closest cell center. At the domain outlet a zero normal gradient condition is applied for all flow quantities. The final simulation time, tend = 0.002 s, is chosen such that no waves have reached the outlet boundary at the end of the simulation. The numerical boundary conditions at the domain outlet have, hence, little influence on the analysed transient flow profile. The maximal flow speed throughout the domain is less than u0 + �u ≈0.31m/s, corresponding to a Mach number M < 1/1000 which is extremely close to the limit of incompressible flow. However, in order to achieve sufficient temporal resolution for the simulation of acoustic wave propagation, the chosen time-step needs to satisfy an acoustic CFL condition which is several orders of magnitude more restrictive than the convective CFL condition for this low Mach number flow. For the subsequent simulation, the time step is chosen such that CFLacoustic < 0.13.

Simulation results at t = tend using the second-order backwards Euler scheme for time integration, second-order centered differences for spatial discretization, and the steady MWI (see Eq. (26)) as well as the transient MWI (see Eq. (27)) are shown in Fig. 3. The analytic solution obtained from linear acoustic theory gives a monochromatic wave of wavelength λ ≈0.164 m and pressure amplitude �p ≈1.29 Pa. The wave speed is the speed of sound, c0 ≈346.6 m s−1, thus at tend the wave has travelled c0 tend ≈0.69 m, thus staying within the simulation domain.

As can be observed in Fig. 3, the amplitude and length of the wave is accurately predicted when applying implicit second-order time-discretization together with both formulations of the MWI for face velocity interpolation.

Typically, it is sufficient to limit the number of temperature cycles, each of which requiring about 2–4 nonlinear iterations, to Nmax = 5 per time step in order to obtain a converged result of high accuracy. If the number of constant temperature cycles in each time step is limited to a smaller number of Nmax, the transient accuracy can be affected as the wave speed is no longer accurately computed, as can be seen in Fig. 4, which compares results specifying different maximal numbers of temperature cycles (Nmax) executed per time step with the analytic solution as given by linear acoustic theory.

The convergence behaviour as displayed by the decay of velocity residuals (normed by the residuals at the first iteration) during the nonlinear iterations at the first time-step is shown in Fig. 5. It can be observed from Fig. 5 that the velocity residuals rapidly reduce within each temperature cycle where temperature is kept constant, and increases at the start of a new temperature cycle after an update of the temperature field. In order to ascertain the convergence of the temperature field, the behaviour of temperature residuals (normed by the residuals at the first temperature cycle) during the iteration

102 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 4. Effect of max. number of temperature cycles (subgroups of iterations based on fixed temperature) used per time step on transient accuracy of simulation for monochromatic wave propagation: comparison of simulated pressure disturbances dP at t = 0.002 s with exact result from linear acoustics.**


> **Fig. 5. Decrease of velocity residuals during nonlinear iterations at the first simulation time step for monochromatic wave propagation, normalized with the residual magnitude of the first iteration (N = 1).**

of temperature cycles at the first time step is shown in Fig. 6. It can be observed that the temperature residuals also consistently decay with an increasing number of temperature cycles. As a general observation not limited to this specific case, the number of iterations required for convergence reduces at later time-steps of the simulation.


### 4.1.2. Propagation of a pulse signal

A spatially confined pulse signal is imposed onto the constant flow field considered in the previous section. The initial conditions of the problem are given by ρ(x) = ρ0 + �ρ(x), u(x) = u0 + �u(x), p(x) = p0 + �p(x) and T(x) = p(x)/(R ρ(x)), with the perturbation pulse signal defined by


## �p(x) = dP exp{−(x −x0)2/(2 s2)}


## �u(x) = 1 ρ0c0 �p(x)


# �ρ(x) = 1


## c2 0 �p


## dP = 200 Pa .

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 103


> **Fig. 6. Decay of temperature residuals during iterations of temperature cycles at the first simulation time step for monochromatic wave propagation, normalized with the temperature residual magnitude of the first cycle (N = 1).**


> **Fig. 7. Simulation of an acoustic pulse signal propagation inside a low-speed flow: plots for pressure disturbance along the channel at t = 0.002 s; comparison of results applying Eqs. (26) and (27) for face-velocity interpolation with analytic solution from linear acoustics.**

The pulse signal is centered around the point x0 = 0.2 m and has a width of s = 0.1 m. The spatially narrow width of the pulse signal implies a broad spectrum of frequencies, in contrast to the single-frequency perturbation studied in the previous section. The velocity and density perturbations are chosen such that the original pressure pulse signal travels from left to right according to linear acoustic theory [29]. Due to the homogeneity of the initial field and lack of viscosity, the amplitude of the pulse signal should remain unchanged. To ensure that no external sources are acting on the flow field, zero normal gradient conditions for all flow variables u, p, T are imposed at the boundaries of the domain. The final simulation time, tend = 0.002 s, is chosen such that the pulse signal does not reach the domain outlet boundary during the simulation. Following the same practice as in the previous test case for single-frequency excitation, the time-step chosen here also has to satisfy an acoustic CFL condition in order to accurately resolve the acoustic wave propagation during the simulation. Due to the extremely small maximal flow speed u0 +�u < 2 m/s and the corresponding Mach number M < 0.01 throughout the domain, the acoustic CFL condition is again several orders of magnitude more restrictive than the convective CFL condition. For the subsequent simulation, the time step is chosen such that CFLacoustic < 0.07.

Simulation results at t = tend using the second-order backward Euler scheme for time integration, second-order centered differences for spatial discretization, and the steady MWI, see Eq. (26), as well as the transient MWI, see Eq. (27), are shown in Fig. 7. The analytic solution obtained from linear acoustic theory predicts the location of the pulse signal at t = tend to be xend = x0 + c0 tend ≈0.89 m. The shape and amplitude of the pulse signal are unchanged.

104 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130

Using the aforementioned simulation settings allows an accurate prediction of both the location and the amplitude of the pulse signal, as seen in Fig. 7. However, while applying the steady MWI for face velocity interpolation produces artificial wiggles in the vicinity of the propagated pulse signal, there is no such issue if the transient MWI is used instead, which suggests that neglecting the transient term in the MWI introduces a dispersive error, since this problem is only observed in the presence of a wide frequency band and is not perceivable for a single frequency. Thus, in order to achieve high temporal accuracy for a general acoustic problem, the transient MWI presented in Eq. (27) should be used in combination with the second-order implicit time integration.


### 4.2. One-dimensional shock waves

In this section, transient one-dimensional flows containing discontinuous initial data embedded within flow speeds up to M = 3, also referred to as shock tube problems, are being considered. In the presence of strong discontinuities or shocks, the convective terms should not be discretized using second-order central differencing in order to prevent numerical instabilities, and one of the alternative spatial schemes as described in Section 3.1 has to be employed. The computational domain is a rectangular tube of 20 m length, meshed along the direction of flow which aligns with the x-axis. Analytical solutions for this type of initial value problem are readily obtainable as described by Wesseling [2].


### 4.2.1. “Lax” test case

One class of shock tube problem arises when the initial flow field separates two regions at different thermodynamic states and where the high pressure part of the flow is moving towards the stagnant part at transonic speed.

Applying appropriate scaling to the non-dimensional initial conditions as specified by Wesseling [2] and preserving the initial pressure and density ratios given therein, the flow field is initialized with


# w(x,t = 0) = (u0(x), p0(x),ρ0(x))T = � wL if x < 10 wR if x ≥10


### where the discontinuous initial data is given by


## wL = (220.727 m s−1, 3.528 × 105 Pa, 0.445 kg m−3)T


## wR = (0 m s−1, 5.71 × 104 Pa, 0.5 kg m−3)T

Zero normal gradient boundary conditions are imposed for all flow variables at the inlet and outlet. The final simulation time is tend = 0.01 s and the waves do not reach the outlet boundary during the simulation.

The influence of various schemes applied to discretize convective terms, discussed in Section 3.1, are studied using a coarse mesh consisting of 50 cells and a time-step that satisfies CFLacoustic ≤0.3. The density profile obtained with different convection schemes, using the first-order backward Euler scheme, is shown in Fig. 8a and a comparison between the firstorder and second-order backward Euler schemes, while using the TVD-van-Leer scheme to discretize the convective terms, is shown in Fig. 8b. The choice of convection scheme evidently has a strong influence on the density profile, with more diffusive schemes leading to a smaller but wider density step. The choice of time integration scheme, on the other hand, does not have any visible effect. A comparison of simulation results applying different versions of MWI for face velocity interpolation in the continuity equation are shown in Figs. 8c and 8d. It can be seen from both plots that the transient version of the MWI does visibly reduce the overshoots of density and velocity at the shock compared to the steady MWI.

To ascertain that the results produced by the proposed framework do in fact converge to the analytical solution of the initial value problem, simulations run on a refined mesh with 100 cells and on a coarser grid consisting of 50 cells are compared to each other as well as to the result obtained from analytically solving the given one-dimensional gasdynamics problem. Guided by the findings from the simulations on the coarser grid, the TVD-van-Leer scheme is applied to discretize the convective terms, together with implicit first-order Euler time discretization and the transient MWI. The time-step is chosen such that CFLacoustic ≤0.35. The simulation results for pressure, velocity, density and Mach number for two different mesh resolutions are shown in Figs. 9a, 9b, 9c and 9d, respectively. The simulation results reproduce the discontinuity in the velocity, pressure, density and Mach number with very good accuracy. By comparison with the results on the coarser mesh, it is obvious that the observed overshoots significantly reduce upon mesh refinement. This is in stark contrast to the all-speeds numerical framework proposed by Bijl and Wesseling [18], which is unable to reduce a 4% density overshoot in the shock region even after mesh refinement. It is conjectured that this is due to the use of a non-conservative form of the energy equation by Bijl and Wesseling [18], which is not strictly valid across discontinuities.

In order to further verify that the exact locations of the expansion fan, shock and contact discontinuities as well as the correct jump relations across the shock are being captured by application of the proposed numerical framework, a more detailed mesh refinement study is carried out by employing three uniformly-spaced grids consisting of 200 cells, 400 cells, and 800 cells for the spatial discretization of the simulation domain. The regions of interest in the numerical and analytical solutions for this study are the expansion fan, contact discontinuity and shock, as shown in Fig. 10a. The numerical results obtained on meshes of different refinements in the vicinity of these regions are compared in Figs. 10b, 10c and 10d. A convergence study of a similar kind has also been carried out in [31]. It can be observed that the simulation results at these critical locations do converge to the analytical values with subsequent mesh refinement, thus lending additional confidence to the validity of the proposed framework.

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 105


> **Fig. 8. Simulation of the flow profile for Lax’ shock tube problem at t = 0.01 s, using 50 mesh cells along the longitudinal dimension for the discretization of the 1d tube; comparison of effects due to different spatial and temporal discretization schemes on the results are shown in (a)–(d).**


### 4.2.2. “Mach 3” test case

As seen in Fig. 9d, the flow field in the Lax shock tube problem remains subsonic throughout. In order to study a test case containing supersonic flow, the following initial data is considered, as previously studied by Wesseling [2] (after scaling the non-dimensional variables and preserving the pressure and density ratios):


## wL = (290.93 m s−1, 1.0333 × 106 Pa, 3.857 kg m−3)T


## wR = (1122.61 m s−1, 105 Pa, 1.0 kg m−3)T

The same boundary conditions as in Lax’ case are applied. The mesh consists of 100 cells and the time-step satisfies the same CFL restriction as before, i.e. CFLacoustic ≤0.35. Similar to the simulation of the test case as discussed in the previous section, the TVD-van-Leer scheme, implicit first-order Euler time discretization and the transient MWI are used.

The simulation results for pressure, velocity, density and Mach number at t = 0.006 s for two different mesh resolutions are shown in Figs. 11a, 11b, 11c and 11d, respectively. It can be seen that the presence of a sonic point does not affect the accuracy of the result as the location of the expansion fan, contact discontinuity and shock are accurately reproduced. Following the same approach as in the analysis of Lax’ shock tube problem, a more detailed mesh refinement study is carried out by employing three uniformly-spaced grids consisting of 200 cells, 400 cells, and 800 cells for the spatial

106 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 9. Simulation results on coarse and fine meshes for various flow quantities of Lax’ problem at t = 0.01 s are shown in (a)–(d), along with analytical results.**

discretization of the simulation domain. The regions of interest in the numerical and analytical solutions for this study are the expansion fan, contact discontinuity and shock, as shown in Fig. 12a. The numerical results obtained on meshes of different refinements in the vicinity of these regions are compared in Figs. 12b, 12c and 12d. It can be observed that the simulation results at these critical locations do converge to the analytical values with subsequent mesh refinement, though in comparison to Lax’ shock tube problem, the simulated weak shock towards the right border of the domain displays a higher degree of smearing. This phenomenon can be attributed to the weakness of the shock compared to the neighbouring contact discontinuity of the analytical solution, as explained in [2], and is not due to a fault in the proposed numerical algorithm.


### 4.3. Cylindrical explosion

A two-dimensional cylindrical explosion is considered, in which the initial flow field consists of an inner circular core filled with stagnant gas of high pressure and density, surrounded by low-pressure ambient gas at rest. Since the problem is rotationally symmetric, its governing equations can be reformulated and simplified in terms of cylindrical coordinates, which has the same form as the one-dimensional Euler equation with additional geometric source terms, given as [14]


# ∂ρ


# ∂t + ∂(ρur)


## ∂r = −ρur


## r (33)

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 107


> **Fig. 10. Comparison of simulation results to analytical solution of density distribution for Lax’ shock tube problem at t = 0.01 s, using 200, 400 and 800 uniformly spaced cells along the longitudinal dimension for the discretization of a 1d tube.**


# ∂(ρur)


## ∂t + ∂(ρu2 r ) ∂r = −∂p


# ∂r −ρu2 r r (34)


# ∂(ρht)


## ∂t + ∂(ρurht)


## ∂r = ∂p


# ∂t −ρurht


## r , (35)


### subject to the initial conditions


# (ur(0,r), p(0,r),ρ(0,r)) = �(0, pH,ρH) if r < Ri (0, pL,ρL) if r ≥Ri ,

where Ri is the radius of the highly pressurized core. This equation can be solved numerically in a one-dimensional domain to any desired accuracy, thus readily providing a reference solution against which the two-dimensional simulation results can be compared.

For the two-dimensional simulation, a disc-shaped domain with radius R = 10 m and Ri = 5 m is chosen. Two different types of unstructured meshes are applied to discretize this domain, shown in Fig. 13: a) a triangular mesh and b) a polyhedral mesh. Both meshes consist of approximately 13000 cells. The initial pressure and density values are pH = 105 Pa and ρH = 1.0 kg m−3 in the core region, and pL = 104 Pa and ρL = 0.125 kg m−3 in the exterior region. Zero normal gradients for all flow variables are imposed at the domain boundaries. The TVD-van-Leer scheme is used to discretize

108 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 11. Simulation results on coarse and fine meshes for various flow quantities of Mach 3 problem at t = 0.006 s are shown in (a)–(d), along with analytical results.**

convective terms, time discretization is accomplished by the first-order backward Euler scheme and the transient MWI is applied. The chosen time-step ensures that CFLconv. ≤0.25. A reference solution for this problem is obtained by solving Eqs. (33)–(35) in a one-dimensional domain of length R = 10 m using the same numerical framework with 1000 equidistant grid points.


> **Fig. 14 shows the density and pressure distribution obtained by the 2d simulation on the triangular mesh at t = 7.5 × 10−3 s. It can be observed that the solution consists of a circular shock-wave and contact discontinuity that both travel outwards as well as a circular rarefaction fan travelling in the opposite direction. The two-dimensional simulation result of pressure, velocity, density and Mach number obtained on the triangular and the polyhedral mesh along an arbitrary radial direction are compared with the reference solution, as shown in Figs. 15 and 16. The arbitrary choice of a radial direction along which simulation results are sampled has no observable effect on the quality of the presented results since the domain is rotationally symmetric and the used unstructured meshes do not align with any particular direction, as seen in Fig. 13. The results for both triangular and polyhedral meshes are in very good agreement with each other as well as with the reference solution. The velocity and Mach number appear to be slightly more accurately predicted on the polyhedral mesh than on the triangular mesh. Also note that the simulation on the triangular mesh requires approximately 30% more computational effort to complete than the simulation on the polyhedral mesh, due to a larger number of iterations required to achieve convergence.**

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 109


> **Fig. 12. Comparison of simulation results to analytical solution of density distribution for Mach 3 shock tube problem at t = 0.006 s, using 200, 400 and 800 uniformly spaced cells along the longitudinal dimension for the discretization of a 1d tube.**


> **Fig. 13. Different mesh types for discretization of the 2d disc domain: half of the domain using triangular (a) and polyhedral (b) elements is shown.**

Following the same approach as in the analysis of the one-dimensional problems, an additional mesh refinement study is carried out by employing three unstructured meshes consisting of approximately 13 000, 52 000, and 208 000 triangular wedge cells for the discretization of the two-dimensional disc domain. The regions of interest in the numerical and onedimensional reference solutions for this study are the expansion fan, contact discontinuity and shock, as shown in Fig. 17a. The numerical results obtained on meshes of different refinements in the vicinity of these regions are compared in Figs. 17b, 17c and 17d; the simulation results displayed therein are obtained from a slice in an arbitrary radial direction. It can be

110 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 14. Visualisation of simulated density (a) and pressure profile (b) for 2d cylindrical explosion on the unstructured mesh using polyhedral elements at t = 0.0075 s.**


> **Fig. 15. Cylindrical explosion at t = 0.0075 s: comparison of pressure (a) and velocity profile (b) across a chosen radial direction from 2d-simulation on triangular mesh and polyhedral meshes vs reference 1d result.**

observed that the simulation results at these critical locations do converge to the one-dimensional reference values with each refined mesh resolution, hence lending further confidence to the validity of the proposed framework on unstructured grids for two-dimensional domains.


### 4.4. Two-dimensional shock wave reflection on a wedge

An inviscid shock wave travelling with Mach number MS towards a wedge inclined at an angle ψ to the incoming wave velocity vector is considered as the next test case. When the shock is reflected at the oblique wall, the resulting self-similar flow pattern depends on both ψ and MS, and can be classified into one of four categories: (I) regular reflection, (II) single Mach reflection, (III) complex Mach reflection and (IV) double Mach reflection. A discussion on experimental and numerical results for this problem can be found in Refs. [14,44]. For the subsequently conducted simulations at different shock Mach numbers MS applying the proposed all-speeds compressible flow solver, the inviscid shock is initially placed at distance d = 0.05 m from the bottom corner of the oblique wall (see Fig. 18), and initial gas pressure and temperature ahead of the shock are set to pa = 105 Pa and Ta = 300 K. Assuming ideal gas relations and setting heat capacity ratio to γ = 1.4, these conditions correspond to a sound speed of ca ≈360 m/s ahead of the shock. Given a desired target value for MS and gas properties (pa, Ta) ahead of the shock, the properties pb and Tb for the quiescent gas behind the inviscid shock as well as the velocity ua ahead of the shock can be deduced from the Rankine–Hugoniot conditions, as described for example by Toro [14].

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 111


> **Fig. 16. Cylindrical explosion at t = 0.0075 s: comparison of density (a) and Mach number profile (b) across a chosen radial direction from 2d-simulation on triangular and polyhedral meshes vs reference 1d result.**


### 4.4.1. Single Mach reflection


## For the simulation of this type of shock wave reflection, the wedge angle and shock Mach number are set to ψ = 25◦

and MS = 1.7, respectively. The computational domain is a 0.8 m × 0.4 m rectangle uniformly extruded into the spanwise direction and from which a wedge region has been cut out. It is represented by an unstructured mesh with approximately 3.9 × 104 triangular wedge elements as shown in Fig. 18. Zero normal gradient for all variables is imposed at the domain inlet and outlet, and the top and bottom walls are considered to be impenetrable (the flow is inviscid and, hence, slip and no-slip conditions are equivalent). Similar to the cylindrical explosion case studied in the previous section, the TVD-van-Leer scheme, the first-order backward Euler scheme and the transient MWI are used.

The simulation results at time t = 0.75 × 10−3 s for pressure, velocity and density are shown in Fig. 19 and Fig. 20. The resulting self-similar waves belong to the class of single Mach reflection, as defined in [14]. It can be seen that the initial normal shock has travelled approximately 0.46 m and is now located 0.41 m downstream from the bottom wedge corner, consistent with the given initial shock speed of MS · ca ≈610 m/s. The distinctive feature of the shock pattern begins at the triple point where the incident shock meets the reflected shock and are joined orthogonally to the wedge by the Mach stem. Furthermore, the velocity profile reveals another weaker shock that emerges from the triple point and joins the wall obliquely, which is the so-called slip line. The simulation results displayed in Fig. 19 and Fig. 20 are in very good agreement with the experimental and numerical results reported in [45], [14] and [32]. It should be noted, however, that the numerical results of [14,45] were obtained on a structured Cartesian grid, contrary to the mesh with triangular wedge elements used in this study. The simulation for this test case as described in [32] also uses a unstructured mesh containing triangular cells and is based on a staggered placement of flow variables. However, it assumes a substantial fluid viscosity corresponding to a maximal Reynolds number of Re = 3400 and utilizes approx. 94 000 triangles to discretize the simulation domain, involving a refinement near the wall boundaries. In contrast, the numerical simulations in the present work assume inviscid gas properties and use less than half the number of triangular mesh elements to produce results of similar quality as those in [32].

A comparison between the results obtained with the TVD-van-Leer scheme and the first-order upwind scheme is shown in Fig. 21. It can be observed that as expected, application of the TVD scheme gives a better spatial resolution of the shock pattern than the first-order upwind scheme.


### 4.4.2. Double Mach reflection


## For the simulation of this type of shock wave reflection, the wedge angle and shock Mach number are set to ψ = 30◦

and MS = 10, respectively. The computational domain is a 0.8 m × 0.5 m rectangle uniformly extruded into the spanwise direction and from which a wedge region has been cut out. It is represented by an unstructured mesh with approximately 105 triangular wedge-shaped cells as shown in Fig. 18. Zero normal gradient for all variables is imposed at the domain inlet and outlet, and the top and bottom walls are considered to be impenetrable (the flow is inviscid and, hence, slip and no-slip conditions are equivalent). Mimicking the simulation for the single Mach reflection described in the previous subsection, the TVD-van-Leer scheme, the first-order backward Euler scheme and the transient MWI are used for the present simulation.

112 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 17. Comparison of simulation results to 1d reference solution of density distribution for 2d radial explosion problem at t = 0.0075 s, using meshes with 13 000, 52 000 and 208 000 triangular elements for the discretization of a 2d disc domain.**


> **Fig. 18. Simulation domain (a) and mesh (b) for the supersonic wedge reflection problem.**

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 113


> **Fig. 19. Simulated pressure (a) and velocity profile (b) for Mach 1.7 shock reflection at a wedge, t = 0.75 ms.**


> **Fig. 20. Simulated Mach number profile (a) and density contour (b) for Mach 1.7 shock reflection at a wedge, t = 0.75 ms.**


> **Fig. 21. Simulation of Mach 1.7 shock reflection at a wedge, t = 0.25 ms: (a)–(b) compare results obtained with different convective discretization schemes (zoomed view).**

114 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 22. Simulated pressure (a) and velocity profile (b) for Mach 10 shock reflection at a wedge, t = 0.13 ms.**


> **Fig. 23. Simulated Mach number profile (a) and density contour (b) for Mach 10 shock reflection at a wedge, t = 0.13 ms.**

The results at time t = 0.14 × 10−3 s for pressure, velocity and density are shown in Fig. 22 and Fig. 23. The resulting self-similar waves belong to the class of double Mach reflection, as defined in [14]. It can be seen that the initial normal shock has travelled approximately 0.49 m and is now located 0.44 m downstream from the bottom wedge corner, consistent with the given initial shock speed of MS · ca ≈3600 m/s.

A complex shock wave pattern involving several Mach stems can be observed at the point where the normal shock meets its reflected Mach shock and contact discontinuities. The simulation results displayed in Fig. 22 and Fig. 23 are in very good agreement with the experimental and numerical results reported in [44], [30] and [32]. It should be noted, however, that the numerical results in [30,44] were obtained on a structured Cartesian grid, contrary to the triangular mesh used in this present study. The simulation for the same test case as described in [32] also uses a unstructured mesh containing triangular cells and is based on a staggered placement of flow variables. However, it assumes a substantial fluid viscosity corresponding to a maximal Reynolds number of Re = 800 and utilizes approx. 4.4 × 105 triangles to discretize the simulation domain, involving a refinement near the wall boundaries. In contrast, the numerical simulations in the present work assumes inviscid gas properties and use less than a quarter of the number of triangular mesh elements to produce results of similar quality as those in [32].

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 115


> **Fig. 24. Geometry (a) and mesh (b) for the simulation of flow over a cylinder.**


### 4.5. Unsteady low-speed flow over a cylinder

As one of the most frequently employed standard test problem to assess Navier–Stokes solvers, the unsteady twodimensional flow field over a cylinder is studied, applying the proposed novel numerical framework for compressible flows at all speeds. It is well-known from extensive experimental as well as numerical data that up to a Reynolds number of about 47, the flow field is steady and symmetrical about the wake-centreline, whereas at higher Reynolds numbers, the flow field becomes unstable to small disturbances, leading to periodic von-Karman vortex shedding [46]. The flow field is essentially two-dimensional for Reynolds number Re < 180, and three-dimensional instabilities are exhibited at larger Reynolds number values [46].

Two-dimensional simulations at near zero Mach number and Reynolds numbers of 100, 150, 200 and 300 are performed using the proposed all-speeds solver framework as well as the well-established pressure-based incompressible flow solver described in [12]. In all simulations, the fluid is assumed to be an ideal gas with heat capacity ratio γ = 1.4; the far-field pressure and temperature values are set to p0 = 1 bar and T0 = 300 K, respectively, whereas the far-field velocity is u0 = 1 m/s, corresponding to a M < 0.003, i.e. quasi-incompressible flow. In comparison, numerical simulations for this test problem via compressible flow solvers as reported elsewhere, e.g. [28,32], typically impose a higher far-stream Mach number value of M = 0.2, thus further away from the incompressible limit as studied in the present work. The fluid viscosity is varied as to obtain different Reynolds number values ranging from 100 to 300. To determine the effect of mesh refinement, three unstructured meshes consisting of triangular elements with varying resolution are applied to discretize the simulation domain as shown in Fig. 24. It can be seen that the region around the cylinder wall has the finest resolution, and that the mesh elements grow in size with increasing distance to the cylinder. The coarsest mesh has approximately 18 000 cells, the length of its near-cylinder elements is D/15, where D = 0.2 m is the cylinder diameter. The medium mesh consists of 25 000 elements in total, and the near-cylinder resolution is approximately D/22, whereas the finest mesh has 32 000 elements with near-cylinder element size D/30. The time steps used for each simulation obeys the convective CFL number restriction CFLconv < 0.4, which are of the same order of magnitude as the time steps required for an incompressible flow solver. The acoustic CFL number, which is several orders of magnitude larger than the convective CFL number for this low Mach number test case, is of no consideration here for the choice of time step size since the accurate capturing of acoustic waves is not required in this set of simulations. The computed results are then compared with available numerical and experimental data in literature [46,47].


### 4.5.1. Comparison with literature data

The instantaneous spanwise vorticity contour plots and velocity magnitude field for Re = 200 after the onset of vortex instabilities are displayed in Fig. 25a and Fig. 25b, respectively, and both plots clearly indicate the expected phenomenon of von-Karman vortex shedding past the cylinder. The unsteady vortex shedding results in a oscillatory flow field in the wake of the cylinder. From the temporal flow profile at any given point downstream of the cylinder wall, the frequency f of the vortex shedding can be determined, and it can be related to the non-dimensional Strouhal number St = f D/u0, where D is the cylinder diameter and u0 is the far-field flow velocity. In order to validate the simulations performed via the novel all-speeds framework, the Strouhal number St as well as the mean base pressure coefficient C pB are computed from simulation data after the flow has established a stationary state, where C pB is the pressure coefficient evaluated at the base of the cylinder wall, i.e. the point on the cylinder which is furthest away from the inlet. Both of these flow quantities are computed from data accumulated after the flow has reached a stationary state. Figs. 26a and 26b display the Strouhal numbers and mean base pressure coefficients at different Reynolds numbers obtained from the simulations applying the novel all-speeds flow solver on the finest mesh, along with data from other numerical as well as experimental studies published in literature. It can be observed that though the simulations by the proposed all-speeds flow solver tend to overpredict the Strouhal numbers in comparison to values published in other works, the relative differences between the results of the current simulations and past numerical studies are indeed very small, i.e. less than 3%. These minor deviations can be attributed to the relatively small size of the domain used for the present simulations, which is smaller than the

116 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 25. Instantaneous vorticity contour (a) and velocity magnitude (b) inside simulation domain for quasi-incompressible flow past cylinder at Re = 200: Establishment of unsteady vortex shedding.**


> **Fig. 26. Comparisons of Strouhal numbers (a) and mean base pressure coefficient (b) from present simulations and other numerical studies for flow over cylinder at different Reynolds numbers.**

simulation domains applied in [48] that have lengths and heights of at least 30 times the cylinder diameter. The observation that smaller simulation domain sizes do correspond to a larger simulated Strouhal frequency for the unsteady cylinder flow problem has been pointed out in [49]. The agreement between the simulated mean base pressure coefficients with the pressure measurements carried out by Williamson and Roshko [50] is very good up to about Re = 200 beyond which the present simulation results as well as other numerical studies noticeably differ from the experiment. This discrepancy is due to the intrinsically three-dimensional nature of the flow at higher Reynolds numbers which are not being captured by two-dimensional simulations [48].


### 4.5.2. Comparison with incompressible flow simulation

After establishing the fidelity of the proposed all-speeds compressible flow solver algorithm for the simulation of unsteady flow past a cylinder at near zero Mach number, it is of interest to compare and contrast in greater detail the all-speeds flow solver framework’s features with those of an incompressible flow solver. An obvious distinction between those two families of flow solver strategies is that all-speeds flow solvers are based on the system of compressible flow equations which contains an additional energy equation that is coupled to the momentum and continuity equations via equations of state for the fluid density, pressure, and temperature, whereas incompressible flow solvers are only coupling the momentum and continuity equations and leaving fluid density and temperature decoupled from the pressure field. At very low Mach numbers and relatively small initial gradients of fluid properties, the resulting density and temperature field of the quasi-incompressible flow do not exhibit sufficiently strong variations to significantly affect the flow velocity and pressure, thus simulation results applying both types of flow solvers should be in close agreement under these conditions. To verify this, the simulations for the unsteady cylinder flow at different Reynolds numbers on the same geometry and

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 117


> **Table 1 Strouhal numbers for unsteady cylinder flow computed via all-speeds and incompressible flow solvers on different meshes (coarse: 18 000 cells, medium: 25 000 cells, fine: 32 000 cells)**

Re St (all-speeds, coarse) St (all-speeds, medium) St (all-speeds, fine) St (incompress., fine) St ([51])

100 0.167 0.167 0.167 0.167 0.163 150 0.185 0.185 0.185 0.185 0.183 200 0.192 0.194 0.203 0.196 0.195

meshes obtained via the proposed all-speeds compressible flow solver and the incompressible flow solver described in [12] are compared to each other. In Table 1, the Strouhal numbers extracted from these simulations are listed. Values for Strouhal numbers given in [51] obtained by solving the incompressible Navier–Stokes equations via a high order spectral element method are also listed as a reference. Again, it should be pointed out that the generally slightly higher Strouhal number values obtained in the present study compared to [51] can be attributed to the smaller domain size used for simulations in this work, an effect explained in [49]. It can be seen that there is perfect agreement up to the third decimal place for the values corresponding to Reynolds numbers Re = 100, 150; however, the all-speeds solver does produce an approximately 3% higher vortex shedding frequency at the largest displayed Reynolds number Re = 200. A possible explanation for this increase of frequency may be the fact that the all-speed compressible flow solver is capable of triggering additional instabilities caused by pressure–density–temperature coupling since it solves an additional energy equation which is linked to the other flow variables via equations of state. At lower Reynolds numbers, this effect appears to be negligible; however when it does become significant at higher Reynolds numbers, the additional compressibility instability generated by the compressible flow solver would make the flow field less stable than the flow field simulated via an incompressible solver, hence the computed vortex shedding frequency would be higher in the former case.


### 5. Steady-state results

In this section, several flow problems with well-defined steady-state solutions covering speed regimes ranging from quasi-incompressible and subsonic flows to transonic and supersonic/hypersonic flows which are frequently used as benchmark cases in the CFD literature are studied to further validate the proposed numerical framework. The transient terms are discretized using the first-order backwards Euler scheme and the applied time-step, irrespective of the Mach number, satisfies 0.3 ≤CFLconv ≤1.2 in all cases. The choice of discretization scheme for the convective terms depends on the nature of the simulated flow. For flows free of shocks (subsonic or supersonic) on structured grids, second-order central differencing is used, whereas for flows containing shocks or simulations on unstructured grids, the TVD-van-Leer scheme is applied due to its numerical stability and better convergence behaviour.


### 5.1. Lid-driven cavity

The two-dimensional lid-driven cavity case is an internal flow configuration frequently used to validate numerical frameworks for incompressible flows. High-resolution reference results for this test case have previously been reported by Ghia et al. [52]. The geometry is a closed cube of length l = 1 m, with the top wall (lid) moving at velocity U = 1 m s−1. The considered fluid has a density of ρ = 1 kg m−3 and a (constant) kinematic viscosity of ν = 0.01 m2 s−1, with a speed of sound at initial temperature T0 = 295 K of c0 = 355 m s−1. The resulting Reynolds number of the flow is Re = l U/ν = 100. The thermal conductivity of the fluid is assumed to be zero. All domain boundaries are treated as adiabatic, no-slip walls. As for pressure and temperature variables, their normal gradients are specified to be zero on all stationary walls whereas at the moving lid they are assigned their reference values of p0 = 1.013 × 105 Pa and T0 = 295 K, respectively. The cavity is two-dimensional and is represented by an unstructured mesh with 22800 triangular elements. The flow field is initialized by setting all velocities to zero and fixing pressure as well as temperature to their reference values throughout the computational domain. Since the maximal flow speed is 1 m s−1, the Mach number is M < 0.003. The time step used for the simulation is �t = 5 × 10−3 s which corresponds to a convective CFL number CFLconv less than 1.2 throughout. The acoustic CFL number, which is several orders of magnitude larger than the convective CFL number for this low Mach number flow problem, does not influence the choice of time step size here since the accurate capturing of acoustic waves is not required for the simulation to reach the steady solution.

The simulated velocity magnitude contours are shown in Fig. 27b, and the computed velocity profiles along the center lines of the cavity are shown in Fig. 28, together with the reference data of Ghia et al. [52]. The results are in very good agreement with the reference data, demonstrating the accurate prediction of flows in the quasi-incompressible regime by the proposed numerical framework.


### 5.2. Planar Couette flow with non-constant fluid properties

A planar Couette flow is considered in this section as the next validation test case due to its simplicity which renders itself to being tractable via analytical solutions. The flow configuration consists of fluid between two parallel infinite plates located at constant distance h from each other, one of which is moving at constant velocity U and the other is

118 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 27. Triangular mesh (a) and simulated velocity magnitude contours (b) for the lid-driven cavity problem at Re = 100.**


> **Fig. 28. Centerline velocity profiles for low-speed 2d lid-driven cavity flow at M ≪0.1: comparison of simulation results on equidistant Cartesian mesh along horizontal (a) and vertical (b) centerlines with corresponding reference data from [52].**

being held stationary. An analytical solution for this problem is available for both incompressible and compressible flow with constant as well as temperature-dependent fluid properties [53]. Compressibility effects enter this configuration via temperature-dependency of the viscosity and thermal conductivity of the fluid, which affects both heat transfer as well as velocity distribution. The symmetry of the configuration implies vanishing of all convective contributions, hence the flow is solely governed by the balance of viscous stresses and heat conduction. Constant heat capacity, cp, and heat capacity ratio, γ = cp/cv, are assumed and the temperature-dependency of the dynamic viscosity, μ, and thermal conductivity, κ, are modelled by a power-law, given as μ(T) = μ0(T/T0)0.67 and κ(T) = κ0(T/T0)0.67, which results in a constant Prandtl number and is sufficiently accurate over the considered temperature range. The computational domain is represented by a one-dimensional mesh with 200 equidistant cells between the two plates. The flow field is initialized with u0 = 0 m s−1, p0 = 105 Pa and T0 = 300 K. At the moving plate (y = 0 m), the only relevant velocity component is tangential and fixed to u = Umax, whereas a constant temperature T = T0 is imposed. Static pressure is extrapolated from the closest cell center. At the stationary plate (y = h = 1 m), u = 0 m s−1 and temperature and pressure are fixed to T = T0 and p = p0, respectively.

At this point, it should be mentioned that for the simulation of flow problems such as the one-dimensional Couette flow where thermal conduction is of at least equal importance as convective effects, the heat flux term in the energy equation should be reformulated in terms of total enthalpy and be treated as an implicit contribution in the linearized equation system. Otherwise the simulation becomes unstable for time steps which correspond to a maximal convective CFL number CFLmax conv that is not orders of magnitude smaller than unity due to the presence of explicitly treated heat flux term. For high-speed flows where convection is the predominant factor in shaping the flow field, this modification becomes unnecessary since the more restrictive convective CFL condition alone is sufficient to ensure stability of the energy equation.

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 119


> **Fig. 29. Velocity (a) and temperature distribution (b) along wall distance x for 1d Couette flow: effect of temperature-dependence of fluid viscosity and conductivity on flow profile (only half of the plot is shown due to symmetry).**

The simulation results for the steady profiles of velocity and temperature at different flow Mach numbers are shown in Fig. 29, along with the analytical solutions which can be found in [13,53]. The effect of temperature-dependent fluid viscosity and thermal conductivity on the temperature and velocity profiles can be clearly observed and is in good agreement with the analytical results.


### 5.3. Converging–diverging nozzle

In this section, the flow field inside a quasi one-dimensional, parabolic converging–diverging nozzle is simulated. The nozzle has a length of l = 1 m and its longitudinal dimension aligns with the x-axis. The cross-sectional area A of the nozzle varies along the x-axis as given by A(x) = 0.1 m2 [1 + 4(x/l)2] for −0.5 m < x < 0.5 m. Due to the quasi one-dimensional nature of the problem, it is sufficient to mesh the nozzle along its longitudinal direction only. By varying the boundary conditions at the nozzle inlet and outlet, it is possible to achieve different flow regimes inside the nozzle. The considered fluid is inviscid and has no thermal conductivity, which admits analytical solutions [13] against which the simulation results are validated.


### 5.3.1. Subsonic flow

In order to achieve subsonic flow inside the nozzle, the flow Mach number at the inlet must be sufficiently small such that sonic conditions are not achieved at the nozzle throat. For these simulations, the pressure at the domain outlet is fixed to pout = 105 Pa. At the nozzle inlet, a constant mass flow ˙min is specified by fixing the inlet velocity to uin = ˙min/ρ(pin, Tin), where the density ρ is calculated from the inlet pressure pin, obtained by extrapolation from the closest cell center, and the inlet temperature which is fixed to Tin = 300 K. Due to the symmetry of the subsonic flow field within the nozzle, the pressure at the inlet equals the fixed outlet pressure pout at steady-state, so the steady inlet density is given by ρ∞ in = ρ(pout, Tin). Hence, the inlet velocity at steady-state can be determined from the specified flow variables as u∞ in = ˙min/ρ(pout, Tin) = Min cin, where Min is the desired inlet Mach number and cin is the speed of sound at steady inlet conditions. It needs to be pointed out that this way of specifying inlet conditions permits the simulation to reach steady state even without implementing any form of sophisticated non-reflecting boundary conditions at the outlet. In contrast, if at the inlet boundary the values of the normal velocity component and temperature are fixed whereas pressure is extrapolated from the interior [10], then using the constant pressure outlet boundary condition as described here or in [10] without special treatments will cause excessive reflections at the outlet, thus resulting in an oscillating solution in time without ever converging to a steady state.

The considered fluid is assumed to be an ideal gas initially at rest, with an initial pressure of p0 = 105 Pa and initial temperature T0 = 300 K. The one-dimensional computational domain is discretized via an equidistant mesh of 100 cells. Fig. 30a shows the axial distribution of Mach number along the length of the nozzle for three different inlet Mach numbers, alongside analytical solutions obtained for quasi one-dimensional inviscid flow [13]. In all three cases, the simulation results are in excellent agreement with the analytical solution for compressible flow. As can be observed, numerical errors tend to be more visible at larger Mach numbers near the nozzle throat region located at x = 0 m.

120 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 30. Mach number profile for subsonic flow (a) and transonic flow (b) inside the nozzle: comparison of simulation and analytic solutions.**


### 5.3.2. Transonic flow

In order to obtain transonic flow with a normal shock inside the nozzle, a fixed pressure ratio between the domain inlet and the domain outlet of pin/pout = 1.2 is specified. The temperature at the inlet is set to Tin = 300 K. Due to the presence of a normal shock in the flow domain, the TVD-Minmod scheme is applied for the discretization of the convective terms. Simulations are carried out on a coarse mesh with 100 cells as well as on a fine mesh with 400 cells. The same initial conditions as for the subsonic case described in the previous section are applied. The resulting Mach number profiles along the axial direction of the nozzle as well as the corresponding analytical solution are shown in Fig. 30b. The shock location is accurately predicted on both meshes, although the magnitude of the shock is underpredicted on the coarse mesh. The results on the fine mesh are in very good agreement with the analytical solution. Small numerical oscillations downstream of the shock can be observed, which, however, reduce upon mesh refinement.


### 5.3.3. Supersonic flow

Supersonic, shock-free flow in the entire nozzle is achieved by specifying pressure, velocity and temperature at the domain inlet as to obtain an inlet Mach number of Min = 2.5 normal to the inlet, while extrapolating all flow variables at the domain outlet. The initial flow field is the same as in the subsonic and transonic cases. The computational domain is represented by an equidistant mesh with 400 cells. The computed Mach number profile in the nozzle is virtually indistinguishable from the corresponding analytical solution, as is shown in Fig. 31a.


### 5.3.4. Isentropic expansion from subsonic to supersonic flow

A second type of shock-free supersonic flow within the converging–diverging nozzle arises when gas isentropically expands in the nozzle from subsonic to supersonic conditions. In order to simulate such a flow, pressure and temperature at the domain inlet are fixed to pin = 105 K and Tin = 300 K, whereas at the supersonic outlet all flow variables are extrapolated. The inlet Mach number then adjusts itself during the course of the simulation to the specific subsonic value which results in an isentropic expansion inside the nozzle, reaching a Mach number of M = 1 at the nozzle throat. An equidistant mesh with 400 cells and the same initial conditions as in the previous supersonic cases are used. The simulation result, shown in Fig. 31b, is in excellent agreement with the corresponding analytic solution, further demonstrating the ability of the proposed numerical framework to accurately predict flows at all speeds.


### 5.4. Flow in a bumped channel

In this section, flows at different speed regimes over a bump inside a channel are studied. The channel has a height of h = 0.5 m and a length of l = 1.5 m. The circular bump is located in the center of the channel and has a width of 0.5 m. For subsonic and transonic test cases, to be presented in Sections 5.4.2 and 5.4.3, the thickness-to-chord ratio of the bump is 10%, whereas for the case with supersonic flow, to be presented in Section 5.4.4, the bump’s thickness-to-chord ratio is 5%. This is a validation test case heavily used in CFD literature and reliable reference results have been reported, for example, by Favini et al. [54].

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 121


> **Fig. 31. Shock-free supersonic flow inside a nozzle: comparison of simulated Mach number profiles for pure supersonic flow (a) and isentropic expansion (b) with analytic solutions.**


> **Fig. 32. Simulated pressure profile and contours inside a bumped channel for quasi-incompressible flow with Min = 0.01.**


### 5.4.1. Quasi-incompressible flow

At the domain inlet, a constant mass flow normal to the boundary as well as the static temperature are fixed, whereas a constant static pressure is specified at the domain outlet to achieve an inlet Mach number of Min = 0.01. Under these conditions, the steady flow can be considered to be nearly incompressible. The reference values for static pressure and temperature specified at the boundaries are p0 = 105 Pa and T0 = 300 K. The initial flow field is uniform with M = Min, p = p0 and T = T0. Simulations are carried out on two different triangular meshes with 1650 cells (coarse mesh) and 6500 cells (fine mesh). The pressure contours of the computed steady-state result are shown in Fig. 32, and the profiles of the velocity at the top and bottom walls are displayed in Fig. 33. A separate simulation for the same problem is carried out on the fine mesh under the assumption of incompressible flow, applying boundary conditions that result in the same steady inlet velocity corresponding to Min = 0.01 in the compressible flow case. In general, the pressure and velocity profiles are in very close agreement with the reference results for incompressible flow, as displayed in Fig. 33.


### 5.4.2. Subsonic flow

To achieve a subsonic flow at moderately high Mach numbers within the bumped channel, a steady inlet Mach number of Min = 0.5 is applied, and all other boundary and initial conditions are specified in the same manner as in the previous case. Simulations are carried out on two different triangular meshes with 1650 cells (coarse mesh) and 6500 cells (fine mesh). The pressure contours of the computed steady-state result are shown in Fig. 34, and the profiles of the Mach number at the top and bottom walls are displayed in Fig. 35. In theory, the subsonic laminar flow over the bump is symmetric due to its isentropy and the symmetry of the domain. On the coarse mesh, however, the Mach number profile on the bottom wall exhibits a slight asymmetry, as seen in Fig. 35. This asymmetry is significantly reduced upon mesh refinement. In general, the pressure and Mach number distribution are in very close agreement with the results reported by Favini et al. [54].


### 5.4.3. Transonic flow

In order to achieve transonic flow inside the channel, the boundary conditions are specified in the same way as for the subsonic flow case as described in the previous section, which results in a steady inlet Mach number of Min = 0.675. The

122 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 33. Quasi-incompressible flow inside a bumped channel with Min = 0.01: simulated velocity profiles at top (a) and bottom (b) walls on coarse grid (1650 triangular elements) and fine grid (6500 triangular elements) compared with simulation results for incompressible flow on fine mesh.**


> **Fig. 34. Simulated pressure profile and contours inside a bumped channel for subsonic flow with Min = 0.5.**


> **Fig. 35. Subsonic flow inside a bumped channel with Min = 0.5: simulated Mach number profiles at top (a) and bottom (b) walls on coarse grid (1650 triangular elements) and fine grid (6500 triangular elements) compared with results from Favini et al. [54].**

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 123


> **Fig. 36. Pressure contours inside a bumped channel for transonic flow with Min = 0.675.**


> **Fig. 37. Transonic flow inside a bumped channel with Min = 0.675: simulated Mach number profiles at top (a) and bottom (b) walls on coarse grid (1650 triangular elements) and fine grid (6500 triangular elements) compared with results from Favini et al. [54].**

reference values for pressure and temperature are p0 = 105 Pa and T0 = 300 K, and the initial flow field is uniform with M = Min, p = p0 and T = T0. Due to the expected occurrence of a normal shock, the TVD-van-Leer scheme is applied for the discretization of the convective terms. Simulations are carried out on the same unstructured meshes as in the subsonic flow case presented in the previous section. Fig. 36 shows the pressure contours of the computed steady-state results, and the profiles of the Mach number at the top and bottom walls are displayed in Fig. 37. As for the subsonic flow discussed in the previous section, the computed result is in very good agreement with the results provided by Favini et al. [54]. The predicted position of the shock is similar on both meshes, although the amplitude of the shock (i.e. the minimum and maximum Mach number) are less pronounced on the coarse mesh in comparison with the fine mesh.


### 5.4.4. Supersonic flow

For the supersonic flow case, velocity, pressure and temperature are specified at the inlet as to obtain an inlet Mach number of Min = 1.4. The initial flow field is uniform with M = Min, p = 105 Pa and T = 300 K. Simulations are conducted on two different triangular meshes with 3000 cells (coarse mesh) and 14500 cells (fine mesh). The pressure contours of the computed steady-state result are shown in Fig. 38 and the profiles of the Mach number at the bottom and top wall are given in Fig. 39. Similar to the transonic flow, the shock position is well predicted on both meshes. However, the coarse mesh predicts a smaller amplitude of the shock and, in fact, the fine mesh predicts a locally subsonic Mach number at the top wall which is not resolved on the coarse mesh. The shock structure as shown in Fig. 39a and Fig. 39b by the Mach number distribution on the fine mesh is in excellent agreement with the corresponding results of Favini et al. [54].


### 5.5. Supersonic flow around a cylindrical obstacle

In this section, the external flow around a cylinder at free-stream inlet Mach numbers Min = 3 and Min = 5 is simulated. The considered geometry is a parabolic domain from which one half of a circular cylinder with radius r = 1 m is cut out. The shape and dimensions of the simulation domain are shown in Fig. 40a. The domain is represented by an unstructured mesh consisting of 6 × 104 quadrilateral cells, shown in Fig. 40b. At the supersonic inlet of the domain velocity, pressure

124 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 38. Pressure contours inside a bumped channel for supersonic flow with Min = 1.4.**


> **Fig. 39. Supersonic flow inside a bumped channel with Min = 1.4: simulated Mach number profiles at top (a) and bottom (b) walls on coarse grid (3000 triangular elements) and fine grid (14500 triangular elements) compared with results from Favini et al. [54].**


> **Fig. 40. Entire domain (a) and partial view of the mesh (b) for the simulation of supersonic flow around the cylinder wall.**

and temperature are assigned fixed values such that a desired inlet Mach number Min is obtained, with pin = 105 Pa and Tin = 300 K. At the outlet boundaries of the domain, all flow variables are extrapolated from the closest cell center. The initial flow field is uniform with p = pin, T = Tin and M = Min. The TVD-Minmod scheme is applied for the discretization of the convective terms and the applied time-step satisfies CFLconv ≲0.5.

The steady-state Mach number contours for inlet Mach numbers Min = 3 and Min = 5 are shown in Fig. 41a and Fig. 41b. The minimal distance d between the bow-shaped shock and the cylinder reduces with increasing Mach number, as expressed by the empirical formula d ≈0.386 r exp{4.67/M2 in} [54]. Along the horizontal symmetry line of the domain passing through

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 125


> **Fig. 41. Mach number contours for supersonic flows with Min = 3 (a) and Min = 5 (b) around the cylinder.**

the stagnation point at the cylinder wall, a normal shock is encountered, which makes it possible to calculate the theoretical value of the total pressure downstream of the shock from the upstream Mach number Min and total inlet pressure pt,in [13]. The ratio of the total pressure pt to pt,in from the simulation on the symmetry line is shown in Fig. 42a, along with the theoretical values calculated from the normal shock relations. Fig. 42b shows the Mach number profile along the symmetry line, while indicating the shock location predicted by the empirical formula on the top axis. It can be seen that whereas the empirical formula appears to under-predict the simulated distance from the shock to the cylinder wall, the agreements between the presented simulations and the analytical values for total pressure ratios in both test cases are excellent. From the Mach number plots of the steady solution as shown in Figs. 41a, 41b and 42b, it is evident that the steady flow field contains regions of both very high and very low Mach numbers, i.e. the flow speed at regions near the domain inlet is close to the supersonic inlet Mach number Min > 1, whereas the flow Mach number at regions around the stagnation point, i.e. the only point on the cylinder wall that also lies on the horizontal line of symmetry, is nearly zero. This concurrent existence of both very high and low Mach numbers within the flow domain does not appear to have any adverse effects on the stability of the proposed solver algorithm during the simulations.


### 5.6. Supersonic flow over cones

The last set of test problems to be studied in this section is the inviscid supersonic flow field over a circular cone, which is a fully three-dimensional problem. It can exhibit a variety of interesting phenomena depending on physical parameters such as angle of attack, free-stream Mach number and conical angle of the configuration. The geometry of the flow problem to be studied in this section is shown in Fig. 43, where the angle of attack ψ is defined as the angle between the free-stream velocity vector and the cone axis. In the case of zero angle of attack ψ = 0, the resulting flow field is entirely axis-symmetric with respect to the cone axis. For a range of combinations of free-stream Mach number Min and cone angle θ, the generated shock wave pattern intersecting every plane parallel to the free-stream flow velocity vector is a straight line, and hence the steady flow field is also irrotational [13]. Thus with the help of adequate coordinate and variable transformations, the compressible flow equations for these particular cases can be reduced to a nonlinear ordinary differential equation, which is the so-called Taylor–Maccoll equation for irrotational conical flows [13]. Hence it is possible to obtain accurate reference solutions for these types of problems by numerically solving an ordinary differential equation, and the computed numerical results can be found in detailed tables as for exampled published in [55]. For nonzero angles of attack ψ > 0, the resulting flow-field over the cone is no longer axis-symmetric, and varies with the cone azimuthal angle φ. If ψ is considered to be small, then the flow-field can be fairly accurately predicted via a perturbation approach which assumes that the deviation from the axis-symmetric flow solution at ψ = 0 is linear with respect to ψ [56]. There also exists a large body of published experimental as well as numerical data for flows with various values of angle of attack at different free-stream Mach numbers. Hence, due to this ready availability of a large body of validation data, supersonic flows over circular cones appear to be an attractive proposition for a 3D problem to validate the reliability of novel compressible flow solver algorithms such as the one presented in this work.


### 5.6.1. Simulation results for flows at zero angle of attack

The domain used for the simulations is a cuboid of dimensions 0.135 m × 0.1 m × 0.1 m from which a portion of a cone with base diameter D = 0.07 m, length l = 0.1 m and cone angle θ = 20◦has been cut out. Due to the aforementioned axis-symmetry of the flow field at zero angle of attack, the domain does not need to capture the cone wall over its entire circumference, hence for this case a quarter of the cone wall has been included in the simulation domain, as shown in

126 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 42. Distribution of total pressure ratio (a) and Mach number (b) along horizontal symmetry line emanating from wall for inlet conditions Min = 3 and Min = 5 around cylinder: comparison of simulation with exact and empirical results.**


> **Fig. 43. Geometry for supersonic flow over circular cone with base diameter D, cone angle θ at small angle of attack ψ.**

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 127


> **Fig. 44. Simulation domain and boundary conditions for simulation of supersonic flows over cone at (a) zero angle of attack and (b) angle of attack ψ = 10◦. The inlet velocity vector is parallel to the x axis in all cases. The tetrahedral mesh in the vicinity of the cone wall is shown in (c).**


> **Fig. 44a. The mesh for this geometry consists of approximately 1.8 million tetrahedral elements which are densely clustered around the cone wall and become more sparsely distributed further away, as can be seen in Fig. 44c. The definition of flow boundary conditions is shown in Fig. 43 and Fig. 44a, i.e. the pressure and temperature are fixed at the inlet to pin = 1 bar, Tin = 300 K, respectively, and the inlet Mach number value is set to Min = 2, Min = 4 for two different test cases. All flow variables are extrapolated from the interior domain at the domain outlet, and no-slip condition is enforced at the cone wall. At slipwalls, the normal velocity component is set to zero, and all other variables are extrapolated from the interior. The initial flow condition is set to equal the free-stream values imposed at the inlet boundary. Accurate shock-capturing and resolution is achieved by applying the TVD van-Leer limiter scheme for the spatial convective discretization. The simulation time step satisfies CFLconv < 0.5, and a steady state result is reached after around 2500 time steps. The steady Mach number distribution for test cases with inlet Mach numbers Min = 2 and Min = 4 are shown in Figs. 45a and 45b. It can be seen that the computed flow field is supersonic throughout and the resulting oblique shock wave is attached to the cone, in contrast to the flow field past a cylinder wall studied previously.**

In order to validate the simulations results, two key quantities characterizing the steady conical flow field are computed, i.e. the shock angle θS between the shock wave and the cone axis, and the static pressure ratio pc/pin at the cone surface. It should be pointed out that due to the ‘roughness’ of the discretized domain at the cone wall caused by the edges of tetrahedral elements, the simulated pressure at the wall is not perfectly uniform. However, apart from a small region near the tip of the cone which is a corner point, the pressure distribution throughout the wall does not deviate from its mean value by more than 1%. The wall pressure value chosen here for representation and comparison is obtained at the domain outlet at which the flow field is almost perfectly axis-symmetric. The values for θS and pc/pin extracted from the simulations as well as the values calculated via Taylor–Maccoll theory and tabulated in [55] are displayed alongside each other in Table 2. It can be seen that the agreement between the listed values is very close, hence validating the accuracy of the proposed compressible flow solver framework for this 3D test case.

128 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130


> **Fig. 45. Mach number distribution for supersonic flows over a circular cone at zero angle of attack with Min = 2 (a) and Min = 4 (b).**


> **Table 2 Shock angles and wall pressure ratios for steady supersonic flow over a cone at zero angle of attack and different inlet Mach numbers: comparison of simulation results with tabulated values in [55].**

Min θ θS (simulation) pc/pin (simulation) θ [55] pc/pin [55]

2 20◦ 37◦ 1.93 38◦ 1.91 4 20◦ 27◦ 4.08 26.4◦ 4.01


### 5.6.2. Simulation results for flow at nonzero angle of attack

The final flow problem to be studied in this series of numerical experiments is the supersonic flow field over a cone at a small nonzero angle of attack. The domain for this simulation is a box region of dimensions 0.15 m × 0.14 m ×0.07 m from which one half of a cone with base diameter D = 0.1 m and cone angle θ = 10◦tilted at angle ψ = 10◦is cut out, as shown in Fig. 44b. It should be noted that due to the loss of axis-symmetry for this problem caused by the nonzero angle of attack, at least one half of the cone wall must be retained in the simulation domain to capture the full flow field. The mesh for this geometry consists of approximately 1.3 million tetrahedral elements which are densely clustered around the cone wall and become more sparsely distributed further away, as can be seen in Fig. 44c. The definition of flow boundary conditions is indicated in Fig. 43 and Fig. 44b. At the inlet, the pressure and temperature are fixed to pin = 1 bar, Tin = 300 K, respectively, and the inlet velocity is fixed to such a value as to achieve a Mach number of Min = 5. All flow variables are extrapolated from the interior domain at the outlet, and no-slip condition is enforced at the cone wall. At slipwalls, the normal velocity component is set to zero, and all other variables are extrapolated from the interior. The initial flow condition is set to equal the free-stream values imposed at the inlet boundary. Accurate shock-capturing and resolution is achieved by applying the TVD van-Leer limiter scheme for the spatial convective discretization. The simulation time step satisfies CFLconv < 0.5, and a steady state result is reached after around 2000 time steps. The steady Mach number distribution for this test problem with inlet Mach numbers is displayed in Fig. 46a. It can be observed that the computed supersonic flow field is no longer axis-symmetric, and the resulting oblique shock wave decreases in strength with growing azimuthal angle φ (see Fig. 43), in line with physical intuition and predictions from the linear theory [56].

In order to further validate the simulations results, the pressure distribution pc at the cone wall surface intersecting with the outlet boundary is extracted from the simulated steady flow field. Reference numerical results for the same test case have been published by Kutler and Lomax [57], and the values for pc normalized by the inlet total pressure pt,in computed from the present simulations as well as the values given in [57] are plotted in Fig. 46b. The agreement between the two sets of results is very close. It can also be observed that while the surface wall pressure initially decreases with growing azimuthal cone angle φ, it reaches a local minimum at about φ = 2.5 and no longer decreases from there on.


### 6. Conclusions

A pressure-based finite-volume framework with collocated variable arrangement for the simulation of flows at all speeds in complex domains has been presented. The numerical framework features an implicit coupling of velocity and pressure and solves the conservative form of the energy equation based on the total enthalpy of the flow. No under-relaxation is required and the framework is numerically stable even in the presence of strong shocks. For the simulation of low Mach number flow problems near the incompressible limit, the proposed all-speeds flow solver is numerically stable for time step sizes of a similar magnitude as those chosen for an incompressible flow solver, thus satisfying a convective CFL condition rather than the much more restrictive acoustic CFL condition.

Simulation results for a large number of benchmark test cases for different flow regimes, on structured and unstructured meshes, have been presented, demonstrating the versatile applicability and the accurate simulation capabilities of the proposed numerical framework. Steady-state results and transient flows have been shown to be accurately predicted in all flow regimes, from quasi-incompressible subsonic flows to transonic and supersonic flows. The accurate prediction of

C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130 129


> **Fig. 46. Steady simulation results for supersonic flows over a circular cone at angle of attack ψ = 10◦with Min = 5: Mach number distribution (a) and surface wall pressure distribution (b).**

transient compressible flows at all speeds, such as the robust capturing of shock and acoustic waves, is attributed to two distinct features of the proposed numerical framework: a) the transient version of the momentum-weighted interpolation (MWI) for face-velocity interpolation in the continuity equation and b) the conservative formulation of the energy equation with respect to total enthalpy. This is significant, since previously published studies have typically focused on steady-state predictions and did not demonstrate a capability to correctly predict transient flows at all speeds. Although the transient MWI can be derived consistently from the momentum equations, the MWI without transient terms is typically preferred in the literature for ease of implementation. The formulation of the time-dependent energy equation in conservative form is also rarely encountered in present literature on pressure-based solvers due to inherent numerical instabilities of its coupling to the remaining flow equations. However, the shortcomings of the non-conservative formulation of the energy equation, as used in most pressure-based frameworks, lies in its inaccuracy when predicting density or Mach number values across discontinuities and has rarely been highlighted. The solver framework proposed in this work is able to address these aforementioned issues and its validated accuracy as well as robustness for a wide range of flow configurations offer a promising prospect of studying more complicated flow phenomena in future research.


### Acknowledgements

The authors are grateful for the financial support provided by Shell Corporation and by the Engineering and Physical Sciences Research Council (EPSRC) through grant No. EP/M021556/1. Data supporting this publication can be obtained from https :/ /doi .org /10 .5281 /zenodo .803287 under a Creative Commons Attribution license.


## References

[1] D.A. Anderson, J.C. Tannehill, R.H. Pletcher, Computational Fluid Mechanics and Heat Transfer, second ed., Taylor & Francis, 1997. [2] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. [3] C. Hirsch, Numerical Computation of Internal and External Flows, Fundamentals of Numerical Discretization, vol. 1, Wiley, 1989. [4] J. Ferziger, M. Peri´c, Computational Methods for Fluid Dynamics, 3 ed., Springer-Verlag, Berlin, Heidelberg, New York, 2002. [5] A.J. Chorin, J.E. Marsden, A Mathematical Introduction to Fluid Mechanics, Springer-Verlag, 1993. [6] C. Hirsch, Numerical Computation of Internal and External Flows, Computational Methods for Inviscid and Viscous Flows, vol. 2, Wiley, 1990. [7] K.C. Karki, S.V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA J. 27 (1989) 1167–1174. [8] A. Cubero, N. Fueyo, Preconditioning based on a partially implicit implementation of momentum interpolation for coupled solvers, Numer. Heat Transf., Part B, Fundam. 53 (2008) 510–535. [9] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, J. Comput. Phys. 228 (2009) 180–201. [10] Z. Chen, A.J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, J. Comput. Phys. 229 (2010) 9150–9165. [11] M. Darwish, F. Moukalled, A fully coupled Navier–Stokes solver for fluid flow at all speeds, Numer. Heat Transf., Part B, Fundam. 65 (2014) 410–444. [12] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numer. Heat Transf., Part B, Fundam. 65 (2014) 218–255. [13] J.D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw–Hill, New York, 2003.

130 C.-N. Xiao et al. / Journal of Computational Physics 346 (2017) 91–130

[14] E.F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical Introduction, second ed., Springer, 1999. [15] R. LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, 2002. [16] I. Demirdži´c, Ž. Lilek, M. Peri´c, A collocated finite volume method for predicting flows at all speeds, Int. J. Numer. Methods Fluids 16 (1993) 1029–1050. [17] J. Van Doormaal, G. Raithby, B. McDonald, The segregated approach to predicting viscous compressible fluid flows, J. Turbomach. 109 (1987) 268–277. [18] H. Bijl, P. Wesseling, A unified method for computing incompressible and compressible flows in boundary-fitted coordinates, J. Comput. Phys. 141 (1998) 153–173. [19] E. Turkel, A. Fiterman, B. van Leer, Preconditioning and the Limit to the Incompressible Flow Equations, Technical Report, NASA-CR-191500, 1993. [20] A.J. Chorin, A numerical method for solving incompressible viscous flow problems, J. Comput. Phys. 135 (1997) 118–125. [21] R. Issa, Solution of the implicitly discretised fluid flow equations by operator-splitting, J. Comput. Phys. 62 (1985) 40–65. [22] J.J. McGuirk, G. Page, Shock capturing using a pressure-correction method, AIAA J. 28 (1990) 1751–1757. [23] R. Issa, C. Lockwood, On the prediction of two-dimensional supersonic viscous interactions near walls, AIAA J. 15 (1977) 182–188. [24] F. Moukalled, M. Darwish, A high-resolution pressure-based algorithm for fluid flow at all speeds, J. Comput. Phys. 168 (2001) 101–130. [25] A.W. Date, Solution of Navier–Stokes equations on nonstaggered grid at all speeds, Numer. Heat Transf., Part B, Fundam. 33 (1998) 451–467. [26] W. Shyy, M.E. Braaten, Adaptive grid computation for inviscid compressible flows using a pressure correction method, in: AIAA, ASME, SIAM, and APS, National Fluid Dynamics Congress, 1988, pp. 112–120. [27] R. Issa, A. Gosman, A. Watkins, The computation of compressible and incompressible recirculating flows by a non-iterative implicit scheme, J. Comput. Phys. 62 (1986) 66–82. [28] K.-H. Chen, R. Pletcher, Primitive variable, strongly implicit calculation procedure for viscous flows at all speeds, AIAA J. 29 (1991) 1241–1249. [29] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure–velocity coupling allowing acoustic calculation in low Mach number flow, J. Comput. Phys. 231 (2012) 5522–5541. [30] N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw, A method for avoiding the acoustic time step restriction in compressible flow, J. Comput. Phys. 228 (2009) 4146–4161. [31] L. Qiu, W. Lu, R. Fedkiw, An adaptive discretization of compressible flow using a multitude of moving Cartesian grids, J. Comput. Phys. 305 (2016) 75–110. [32] M. Tavelli, M. Dumbser, A pressure-based semi-implicit space–time discontinuous Galerkin method on staggered unstructured meshes for the solution of the compressible Navier–Stokes equations at all Mach numbers, J. Comput. Phys. 341 (2017) 341–376. [33] I. Wenneker, A. Segal, P. Wesseling, Computation of compressible flows on unstructured staggered grids, in: Proceedings ECCOMAS 2000, September, 2000, pp. 11–14, http://ta.twi.tudelft.nl/users/wesseling/pdf2000/Wen00SW.pdf. [34] J. Hou, F. Simons, R. Hinkelmann, Improved total variation diminishing schemes for advection simulation on arbitrary grids, Int. J. Numer. Methods Fluids 70 (2012) 359–382. [35] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, J. Comput. Phys. 298 (2015) 466–479. [36] I. Demirdži´c, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Comput. Methods Appl. Mech. Eng. 125 (1995) 235–255. [37] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532. [38] W. Shen, J. Michelsen, J. Sørensen, An improved Rhie–Chow interpolation for unsteady flow computations, AIAA J. 39 (2001). [39] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, J. Comput. Phys. 285 (2015) 24–40. [40] P. Zwart, The Integrated Space–Time Finite Volume Method, Ph.D. Thesis, University of Waterloo, 1999. [41] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W.D. Gropp, D. Kaushik, M.G. Knepley, L.C. McInnes, K. Rupp, B.F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 – Revision 3.7, Argonne National Laboratory, 2016, http://www.mcs.anl.gov/petsc. [42] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W.D. Gropp, D. Kaushik, M.G. Knepley, L.C. McInnes, K. Rupp, B.F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2016. [43] P. Birkby, Numerical Studies of Reacting and Non-Reacting Underexpanded Sonic Jets, Ph.D. Thesis, Loughborough University, 1998, http://ethos.bl.uk/ OrderDetails.do?uin=uk.bl.ethos.297581. [44] P. Woodward, P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, J. Comput. Phys. 173 (1984) 115–173. [45] E.P. Boden, An Adaptive Gridding Technique for Conservation Laws on Complex Domains, Ph.D. Thesis, Cranfield University, 1997, http://hdl. handle.net/1826/3398. [46] C. Williamson, The natural and forced formation of spot-like ‘vortex dislocations’ in the transition of a wake, J. Fluid Mech. 243 (1992) 393–441. [47] R. Mittal, H. Dong, M. Bozkurttas, F.M. Najjar, A. Vargas, A. von Loebbecke, A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries, J. Comput. Phys. 227 (2008) 4825–4852. [48] R. Mittal, S. Balachandar, Effect of three-dimensionality on the lift and drag of nominally two-dimensional cylinders, Phys. Fluids 7 (1995) 1841–1865. [49] M. Behr, D. Hastreiter, S. Mittal, T. Tezduyar, Incompressible flow past a circular cylinder: dependence of the computed flow field on the location of the lateral boundaries, Comput. Methods Appl. Mech. Eng. 123 (1995) 309–316. [50] C. Williamson, A. Roshko, Measurements of base pressure in the wake of a cylinder at low Reynolds numbers, Z. Flugwiss. Weltraumforsch. 14 (1990) 38–46. [51] O. Posdziech, R. Grundmann, Numerical simulation of the flow around an infinitely long circular cylinder in the transition regime, Theor. Comput. Fluid Dyn. 15 (2001) 121–141. [52] U. Ghia, K.N. Ghia, C.T. Shin, High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method, J. Comput. Phys. 48 (1982) 387–411. [53] F.M. White, Viscous Fluid Flow, McGraw–Hill, New York, 1974. [54] B. Favini, R. Broglia, A. Di Mascio, Multigrid acceleration of second-order ENO schemes from low subsonic to high supersonic flows, Int. J. Numer. Methods Fluids 23 (1996) 589–606. [55] J.L. Sims, Tables for supersonic flow around right circular cones at zero angle of attack, NASA SP-3004, NASA Spec. Publ. 3004 (1964). [56] A. Ferri, Supersonic Flow Around Circular Cones at Angles of Attack, NACA Report 1044, 1951. [57] P. Kutler, H. Lomax, A systematic development of the supersonic flow fields over and behind wings and wing-body configurations using a shockcapturing finite-difference approach, in: 9th Aerospace Sciences Meeting, 1971, p. 99.

