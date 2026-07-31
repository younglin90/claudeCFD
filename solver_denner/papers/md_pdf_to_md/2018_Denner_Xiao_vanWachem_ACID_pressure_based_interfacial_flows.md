Journal of Computational Physics 367 (2018) 192–234


### Contents lists available at ScienceDirect


### www.elsevier.com/locate/jcp


# Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation


# Fabian Denner a,∗, Cheng-Nian Xiao a, Berend G.M. van Wachem b

a Department of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom b Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany


## a r t i c l e i n f o a b s t r a c t

Article history: Received 4 September 2017 Received in revised form 5 April 2018 Accepted 13 April 2018 Available online 19 April 2018

Keywords: Interfacial flows Compressible flows Pressure-based methods Finite-volume methods Shock-capturing Acoustics

A pressure-based algorithm for the simulation of compressible interfacial flows is presented. The algorithm is based on a fully-coupled finite-volume framework for unstructured meshes with collocated variable arrangement, in which the governing conservation laws are discretised in conservative form and solved in a single linear system of equations for velocity, pressure and specific total enthalpy, with the density evaluated by an equation of state. The bulk phases are distinguished using the Volume-of-Fluid (VOF) method and the motion of the fluid interface is captured by a state-of-the-art compressive VOF method. A new interface discretisation method is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy based on the assumption of thermodynamic equilibrium, and does not require a Riemann solver. This interface discretisation method yields a consistent definition of the fluid properties in the interface region, including a unique definition of the speed of sound and the Rankine–Hugoniot relations, and conserves the acoustic features of the flow, i.e. compression and expansion waves. A variety of representative test cases of gas–gas and gas–liquid flows, ranging from acoustic waves and shock tubes to shock-interface interactions in one-, twoand three-dimensional domains, is used to demonstrate the capabilities and versatility of the presented algorithm in all Mach number regimes. The propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows are predicted accurately, even for difficult cases that feature fluids with shock impedance matching, transonic shock tubes or strong shocks in gas–liquid flows, as well as on unstructured meshes.


### © 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).


### 1. Introduction

The numerical modelling of compressible flows is associated with a number of difficulties owing to the formulation of the conservation equations, the coupling of hydrodynamic and thermodynamic variables, and high pressure ratios. In addition to the numerical difficulties encountered when modelling compressible single-phase flows, the numerical solution of compressible interfacial flows, in which two (or more) immiscible fluids interact, is further complicated by different fluid


# * Corresponding author. E-mail address: fabian .denner @gmail .com (F. Denner).

https://doi.org/10.1016/j.jcp.2018.04.028 0021-9991/© 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).


# i An update to this article is included at the end

F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234 193

properties and speeds of sound (and, thus, Mach numbers) of the bulk phases, complex acoustic behaviour as well as the numerical treatment of the fluid interface. It has proven particularly difficult to define the discrete interface between two compressible fluids in a consistent manner, which retains the main features of the solution, such as the propagation of acoustic waves and shock waves. According to Coralic and Colonius [1], numerical methods able to accurately predict the interaction of interfaces with shock waves, and in extension compressible interfacial flows in general, should satisfy the following criteria:


### 1. discrete conservation of mass, momentum and energy, 2. avoid the generation of spurious oscillations at the interface or at shock waves, and 3. provide high-order accuracy in smooth regions.

A variety of contemporary algorithms for compressible interfacial flows, e.g. [1–7], have been shown to discretely satisfy the global conservation of mass, momentum and energy in the computational domain, which is a prerequisite for the accurate prediction of the speed of shock waves [8–10], although the conservation in each individual phase is strongly dependent on the applied method. Flux-limited high-order schemes, most notably total variation diminishing (TVD) schemes [11,12], or high-order schemes combined with artificial viscosity models [13], can provide second-order accuracy in smooth regions and avoid oscillations at shocks and discontinuities. Spurious oscillations at the interface as a result of the discontinuity in fluid properties, however, remain a common issue for interface capturing methods [1], in particular when the bulk phases are assumed to be in mechanical equilibrium. Karni [14] and Abgrall [2] were among the first to depart from a fullyconservative discretisation, i.e. a discretisation that is simultaneously conservative with regards to the entire computational domain and each individual phase [15], in favour of avoiding spurious oscillations. The application of a non-conservative discretisation at the interface has also been suggested in the context of incompressible interfacial flows by Brackbill et al. [16], to avoid spurious oscillations as a result of the changing fluid properties at the interface. A non-conservative discretisation of the governing equations is widely, and largely successfully, applied for incompressible interfacial flows, although a naïve non-conservative discretisation does not allow compression and expansion waves to pass the interface. To this end, exact and approximate Riemann solvers are widely applied to allow the exchange of information between the interacting fluids through the interface. Recent studies [17,18] also suggest that a strictly conservative discretisation is important for the robustness of the solution algorithm and the predictive accuracy for incompressible interfacial flows with large density ratios.

State-of-the-art algorithms for compressible interfacial flows are almost exclusively founded on density-based algorithms, which solve the governing conservation equations (momentum, continuity and energy) for the momentum, density and total energy of the flow. In the two-fluid Baer–Nunziato (BN) model [19], each phase is treated as a separate fluid with its own momentum, continuity and energy equations, with an additional transport equation for the volume fraction field of the interacting fluids (e.g. a colour or level-set function), or a topological equation that represents the fluid interface. BN models, also often referred to as seven-equation models, have well-defined hyperbolic properties [20–22] and conserve the mass, momentum and total energy of each phase. However, BN models involve a considerable computational complexity due to the seven, nine or eleven governing equations for one-, twoor three-dimensional flows, respectively, which have to be coupled through relaxation terms for pressure and velocity (usually under the assumption of local thermodynamic equilibrium), including non-conservative products [21–25] and a consistently defined interface velocity [6,21], which is typically based on the solution of an exact or approximate Riemann problem in a Godunov-type, Rusanov-type or Roe-type discretisation.

By assuming equilibrium of pressure and velocity in the limit of infinitely fast relaxation to mechanical equilibrium [6,26,27], BN models can be reduced to five-equation models [1,4,27–31], with a separate continuity equation for each phase, and shared momentum and energy equations. This model conserves the mass of each phase and globally conserves momentum and energy [28]. However, the total energy of each individual phase is not conserved and the relative volume change of the phases due to compression/expansion has to be incorporated in the interface transport equation [6,27]. Abandoning the conservation of a separate density field for each phase, and considering instead only a single density field, leads to a fourequation model [2–4] with a single conservation equation for momentum (one for each space dimension), for mass and for energy, plus an interface transport equation. This approach is also frequently referred to as the one-fluid model, because the entire two-phase flow (both phases and mixtures thereof) is numerically treated as one fluid with locally varying properties. It is the simplest model able to simulate interfacial flows, but does not by design conserve discretely the momentum, mass or energy of each phase. Furthermore, the four-equation model is associated with difficulties when recovering the pressure field in density-based frameworks [4,15,27], for which a specific equation of state (EOS), e.g. a stiffened gas EOS, is required. To this end, the five-equation model of Allaire et al. [4], for instance, reduces to the four-equation model of Shyue [3] when both phases are perfect gases.

In the vast majority of methods, such as in the seven-, fiveand four-equation models discussed above, a Riemann solver is applied to evaluate the fluxes at cell-faces of the computational mesh, effectively reducing the evaluation of the fluxes to a one-dimensional problem for which a Riemann problem is solved [32]. Assuming an interface coincides with a cell-face, a Riemann problem with different EOS can be solved at the fluid interface to couple the bulk phases in an effective manner [33]. Riemann solvers have a strong mathematical foundation, but exact Riemann solvers are for many practical cases prohibitively time consuming, which motivates the development of approximate Riemann solvers. The Harten–Lax–van Leer

194 F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234

Contact (HLLC) formulation, introduced by Toro et al. [34], has become the most widely used approximate Riemann solver for interfacial flows, as it captures contact discontinuities. However, because approximate Riemann solvers rely on an accurate a priori approximation of the wave speeds, their derivation for complex interfacial flows can be cumbersome and the resulting methods are often limited to a specific application or parameter range. Various HLLC-type solvers have been proposed for interfacial flows [1,35–37], including phase transition [38,39] and surface tension [30,38,39]. The Ghost-Fluid Method (GFM), originally proposed by Fedkiw et al. [40], presents an alternative that does not, in general, require the solution of a Riemann problem. The GFM defines ghost cells on either side of the interface for the discretisation of the governing equations, where the real fluid and the ghost fluid coexist, and ghost values are defined in these ghost cells for the fluid on the opposing side of the interface. Due to its conceptual simplicity, the GFM has been used in a variety of numerical frameworks for compressible interfacial flows, notably [41,42]. The GFM has also been extended to interfacial flows with reactions [43,44] and phase transition [44], and has been used in conjunction with discontinuous Galerkin methods [45]. However, the original GFM was found to lack stability for strong shock-interface interactions and compressible gas–liquid flows, a shortcoming which was addressed by coupling the GFM to a Riemann solver to compute the flow states at the interface [46–49].

Density-based algorithms are arguably the method of choice for flows with considerable compressibility, but are illsuited for low-Mach number flows [50–52], requiring sophisticated preconditioning and solution methods, see e.g. [53–57]. The pressure field has to be reconstructed based on the applied EOS, since pressure is not a solution variable in densitybased algorithms. This is generally less of a problem when GFM [40,42,47] or similar so-called sharp interface methods are used, where either phase is present in a mesh cell but never a mixture of them. However, interfacial mesh cells (where the normalised interface indicator function, e.g. a volume fraction, is 0 < � < 1) contain a “mixture” of both phases when interface capturing methods or diffuse interface methods are applied, such as the Volume-of-Fluid method (VOF). This fluid mixture, which Abgrall and Saurel [23] aptly called numerical mixture, is of numerical origin and, from a continuum mechanics standpoint, has no physical basis. Reconstructing the pressure in such interfacial cells is problematic [4,15,27], since hydrodynamically and thermodynamically plausible fluid properties as well as a mixture EOS have to be defined, which may misrepresent the interface region by presuming a physical meaning of the finite interface thickness. As a consequence, the speed of sound in the interface region can, for instance, be lower than the speed of sound of either bulk phase [6,58–60], which is inconsistent with the zero thickness of the interface assumed in continuum mechanics.

Pressure-based algorithms, in which the continuity equation serves as an equation for pressure, while density is evaluated explicitly using a suitable EOS, are preferably applied for incompressible flows and yield significant advantages for low-Mach number flows, since density is not required to be a thermodynamic variable and the acoustic degeneration (i.e. vanishing pressure-density coupling) at low Mach numbers does not pose a problem. The success of pressure-based algorithms is facilitated by the unique role of pressure in all Mach number regimes, with the pressure-velocity coupling dominant at low Mach numbers and the pressure-density coupling dominant at high Mach numbers [61,62], as well as the convenient fact that the fully-conservative formulation of the governing conservation laws can still be satisfied accurately, even if non-conserved quantities, such as pressure, are chosen as primary solution variables [61]. However, ensuring a stable numerical solution in the transonic regime, where pressure is strongly coupled with both velocity and density, and formulating consistent shockcapturing schemes for pressure-based algorithms has proven to be difficult [51,63]. Although, since its original conception by Harlow and Amsden [64,65], many pressure-based algorithms have been proposed for compressible single-phase flows [50,61,66–76], no pressure-based algorithm for compressible interfacial flows has been published yet.

In this article, a fully-coupled pressure-based algorithm for compressible interfacial flows is proposed, based on the finite-volume framework of Xiao et al. [76] for compressible single-phase flows on unstructured meshes. The presented algorithm uses a compressive VOF method [77] for the advection of the interface, leading to a four-equation model. The discretised governing flow equations (momentum, continuity and energy) are solved in a single system of linearised equations for velocity, pressure and specific total enthalpy, with density being evaluated based on the applied EOS. The present study considers the perfect-gas and stiffened-gas models to describe interfacial flows of gases and liquids. A new discretisation method at the fluid interface is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy in the interface region, assuming local mechanical and thermal equilibrium, whereby the conservative discretisation of the governing equations remains unchanged. Since the proposed algorithm is pressure-based and applies a single pressure field for the entire computational domain, no partial pressures have to be considered and no mixture pressure has to be defined at the interface. Instead, only mixture rules for the fluid properties have to be defined. As demonstrated by representative test cases of compressible gas–gas and gas–liquid flows, the presented algorithm predicts the propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows accurately. In particular, the precise simulation of the reflection and transmission of acoustic waves at fluid interfaces as conducted in this study has not been reported in the literature before.

In Section 2, the governing equations are introduced. The numerical framework and the pressure-based algorithm are presented in Section 3 and the applied compressive VOF method is described in Section 4. The new interface discretisation method is proposed in Section 5, followed by a discussion of the iterative solution procedure in Section 6. The results for a variety of representative compressible interfacial flows are presented and discussed in Section 7. The article is summarised and concluded in Section 8.

F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234 195


### 2. Governing equations


### The considered compressible interfacial flows of inviscid fluids at all speeds are governed, assuming Cartesian coordinates, by the momentum equations


# ∂ρu j


## ∂t + ∂ρuiu j


## ∂xi = −∂p


## ∂x j , (1)


### the continuity equation


# ∂ρ


# ∂t + ∂ρui


## ∂xi = 0 , (2)


### and the energy equation


# ∂ρh


# ∂t + ∂ρuih


## ∂xi = ∂p


## ∂t , (3)

where t is time, ρ is the density, u is the velocity vector, p is the pressure and h = cp T + u2/2 is the specific total enthalpy, with cp the specific isobaric heat capacity and T the temperature. The enthalpy formulation is chosen for the energy equation, rather than the more common internal energy formulation, as it leads to a straightforward application in the proposed numerical algorithm, since the transient pressure term on the right-hand side of Eq. (3) does not require linearisation. The governing equations require closure by defining the thermodynamic properties of the fluids through an appropriate EOS.

In this study, the stiffened-gas model [78] is applied, which provides a very good description of liquids and solids of practical interest [33,79], and reduces to the perfect-gas model for gases. It is, therefore, widely used for interfacial flow modelling. The thermodynamic properties of the fluid are linked via the stiffened gas EOS


# ρ = p + γ �


## R T , (4)

where R is the specific gas constant, γ = cp/cv is the heat capacity ratio, cv is the specific isochoric heat capacity and �is a material-dependent pressure constant, which is � = 0 for an ideal gas. The speed of sound for a stiffened gas is


## a = � (γ −1)cp T =


## �


# γ p + �


# ρ (5)


### from which the specific total enthalpy follows as


## h = a2


# γ −1 + u2


## 2 = cp,0 p + �


# p + γ �T + u2


## 2 = cp T + u2


## 2 , (6)


### with the specific isobaric heat capacity


## cp = cp,0 p + �


# p + γ � (7)


# and cp,0 = R/(1 −γ −1). For � = 0, the stiffened-gas model reduces to the perfect-gas model.


### In order to distinguish the interacting bulk phases, the VOF method [80] is applied to represent the bulk phases by an indicator function ψ(x), typically called colour function, with


## ψ(x) =


## � 0 if x ∈�a 1 if x ∈�b (8)

where � = �a ∪�b is the computational domain, with �a and �b the subdomains occupied by fluid a and b, respectively. Consequently, the interface is located in cells with 0 < ψ < 1. Since the density is discontinuous at the fluid interface and fluid does not flow through the interface (assuming no mass transfer), the interface between two immiscible fluids represents a contact discontinuity [23,78]. Thus, the fluid interface is a material front propagating with the flow and, consequently, the material derivative of ψ is


## Dψ


## Dt = ∂ψ


## ∂t + ui ∂ψ ∂xi = 0 . (9)


### Accounting, in addition, for the different acoustic properties of the bulk phases [6,59], Eq. (9) becomes


## ∂ψ


## ∂t + ui ∂ψ ∂xi −K ∂ui


## ∂xi = 0 , (10)

196 F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234


### where


# K = ρb a2 b −ρa a2 a ρa a2 a 1 −ψ + ρb a2 b ψ


## , (11)

is a material-dependent compressibility factor [6,59,60], with ρ given by Eq. (4) and a given by Eq. (5). Assuming the bulk phases are in mechanical equilibrium, and since no mass transfer and surface tension are considered in this study, the interface conditions for velocity and pressure are ua · m = ub · m and pa = pb [81], respectively, where m is the normal vector of the interface.


### 3. Numerical framework

The presented algorithm is founded on a fully-coupled pressure-based numerical framework, based on the algorithm for compressible single-phase flows of Xiao et al. [76]. The framework is predicated on the finite-volume method with a collocated variable arrangement, and all discretisation methods presented below are applicable to unstructured meshes. Since the discretisation of the governing equations is largely identical for single-phase and interfacial flows, the discretisation of the governing equations presented in this section focuses on single-phase flows. The modifications to this discretisation for interfacial flows, including the appropriate definition of the fluid properties, is described in Section 5 and the iterative solution procedure applied to solve the discretised, linearised governing equations is described in Section 6.


### 3.1. Finite-volume method

The finite-volume method, which forms the foundation for the numerical framework, is based on the integral formulation of the governing conservation laws. It is worth recalling that only the integral form of the governing conservation laws is valid at shocks and discontinuities [32]. Integrating, for example, the continuity equation (2) over the control volume V , the integral form of Eq. (2) is given as


## ˚

V


# ∂ρ


## ∂t dV + ‹


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq001.png)


### where


## ˚

V


# ∂ρui


## ∂xi dV = ‹


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq002.png)

follows from the divergence theorem, with S the outward pointing surface vector on the surface dV of control volume V . Assuming the surface of the control volume is constituted by a finite number of flat faces f and applying the midpoint rule [62,82], the surface integral can be expressed with second-order accuracy [82] as


## ‹

dV ρui dSi ≈ �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq003.png)

where A f is the area of face f and ϑ f = u f n f is the advecting velocity, with n f the outward pointing unit normal vector of face f . Because a collocated variable arrangement is employed, all primary solution variables (u, p, h) as well as the density ρ are stored at cell centres. Therefore, the data required at face centres is obtained by interpolation from the data at adjacent cell centres, as further explained in Section 3.2, while the face velocity u f requires a special interpolation to prevent pressure-velocity decoupling, discussed in detail in Section 3.4. The discretisation of the temporal term is discussed in Section 3.3. The interested reader is referred to the work of Xiao et al. [76] for further details on the applied finite-volume method.


### 3.2. Spatial discretisation


### The central differencing scheme is applied for the interpolation from cell centres to face centres of variables that are not advected, given for a general flow variable φ as


## ¯φ f = φP + l f (φQ −φP) . (15)

The geometry coefficient l f is defined based on an inverse distance weighting, given as l f = |r P f |/�s f , where �s f is the distance between cell centres P and Q (the cell centres adjacent to face f ), schematically illustrated in Fig. 1a, and r P f is the vector connecting cell centre P with face centre f .

F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234 197


> **Fig. 1. Schematic illustration of (a) cell P with its neighbour cell Q and the shared face f , where n f is the unit normal vector of face f and s f is the unit vector connecting cells P and Q (both outward pointing with respect to cell P), and (b) upwind cell U and downwind cell D of face f , where u represents the velocity vector.**

Advected variables are interpolated to face centres using the TVD interpolation method for general unstructured meshes proposed by Denner and van Wachem [83]. Considering the velocity vector u as indicated in Fig. 1b, the face value follows as (where ∼denotes a flux-limited interpolation)


## ˜φ f = φU + ξ f


## L f (φD −φU) , (16)

where subscripts U and D denote the upwind and downwind cells, ξ f is the flux limiter and L f = �s f /|rU f | is a geometry coefficient. Dependent on the studied flow, the first-order upwind scheme (ξ f = 0), the central differencing scheme (ξ f = 1) or the Minmod scheme [84] are applied for the simulations presented as part of this study, but other TVD schemes could also be applied.


### 3.3. Temporal discretisation

The First-Order Backward Euler scheme or the Second-Order Backward Euler scheme are applied for the discretisation of the transient terms of the governing flow equations. The First-Order Backward Euler scheme is readily given for cell P as


## ˆ

V P


## ∂φ


## ∂t dV ≈φP −φ(t−�t1) P �t1 V P , (17)

with �t1 being the current time-step, superscript (t −�t1) denotes values of the previous time-level and V P is the volume of mesh cell P. Assuming a varying time-step is applied to solve the governing equations, the Second-Order Backward Euler scheme is given as [85]


## ˆ

V P


## ∂φ


## ∂t dV ≈


## � 1 �t1 −�τ


## �t2 1


## ���


## 1 −�t2 1 �τ 2


## �


## φP −φ(t−�t1) P + �t2 1 �τ 2 φ(t−�τ) P


## �


## V P , (18)

where �τ = �t1 + �t2, �t2 is the previous time-step and superscript (t −�τ) denotes values of the previous-previous time-level. If �t1 = �t2, Eq. (18) reverts to the Second-Order Backward Euler scheme for a constant time-step [82]


## ˆ

V P


## ∂φ


## ∂t dV ≈ 3φP −4φ(t−�t1) P + φ(t−2�t1) P 2�t1 V P . (19)

In what follows, the discretised governing equations are presented using the First-Order Backward Euler scheme, although the Second-Order Backward Euler scheme is also considered in the results presented in Section 7. Irrespective of the chosen scheme, for consistency all transient terms of the governing equations, ∂ρu j/∂t in Eq. (1), ∂ρ/∂t in Eq. (2), as well as ∂ρh/∂t and ∂p/∂t in Eq. (3), are discretised with the same scheme.


### 3.4. Advecting velocity

A momentum-weighted interpolation (MWI) method is applied to evaluate the advecting velocity ϑ f = u f n f at cell faces f , with ϑ f taking the role of flux-velocity in the discretised advection terms of the governing equations. MWI emulates a staggered variable arrangement by introducing a cell-to-cell pressure coupling through a low-pass filter acting on the third derivative of pressure [51,82,86], which avoids pressure-velocity decoupling as a result of the collocated variable arrangement and provides a robust pressure-velocity coupling for incompressible and low-Mach number flows [76], while preserving the second-order accuracy of the finite-volume method [51,82].

Following previous studies, the definition of the advecting velocity includes various modifications to the MWI formulation originally introduced by Rhie and Chow [87], to account for non-orthogonal meshes [88–90], as well as the large density

198 F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234

ratios occurring in interfacial flows and the transient nature of the considered problems [90]. Based on the work of Denner and van Wachem [90], the advecting velocity ϑ f = u f n f at face f , see Fig. 1a, is given as


## ϑ f = ¯u f ,i n f ,i −ˆd f


## � pQ −pP


## �s f −ρ∗ f


## � 1 −l f


# ρP


## ∂p ∂xi


## ���� P + l f ρQ


## ∂p ∂xi


## ���� Q


## �


## s f ,i


## �


## + ˆd f ρ∗(t−�t1) f


## �t1


## � ϑ(t−�t1) f −¯u(t−�t1) f ,i n f ,i � .


## (20)


## The coefficient ˆd f is defined as


## ˆd f =


## �V P eP + V Q


## eQ


## �


## 2 + ρ∗ f �t1


## �V P eP + V Q


## eQ


## �, (21)

where eP and eQ are the sum of the coefficients of the primary variable u arising from the advection terms (and, if viscous flows are considered, the shear stress terms) of the momentum equations (see [90] for a detailed derivation). The face density ρ∗ f is interpolated by a harmonic average,


## 1 ρ∗ f = 1 −l f


# ρP + l f ρQ , (22)

which is necessary for a consistent definition of the coefficient of the pressure term as well as the efficacy of the density weighting [90]. This density weighting of the pressure gradients, with which robust incompressible interfacial flow simulations with density ratios of up to 1024 [90,91] were previously presented, stabilises the numerical solution of interfacial flows with large density ratios.


### 3.5. Discretised governing equations


### Applying the spatial and temporal discretisation techniques described above to the integral formulation of the continuity equation, Eq. (12), the discretised continuity equation for cell P follows as


# ρ(n+1) P −ρ(t−�t1) P �t1 V P + �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq004.png)

with the advecting velocity ϑ f given by Eq. (20) and superscript (n + 1) denotes variables that are solved for implicitly. Although the discretised continuity equation is formulated conservative in mass, it is treated as an equation for pressure. Hence, based on Eq. (4), the implicit density ρ(n+1) P at cell centre P is given by the pressure-implicit formulation


# ρ(n+1) P = p(n+1) P + γP �P R P T P , (24)

where the fluid properties γ , �and R are defined based on the most recent available interface position, as explained in Section 5, and the temperature T is deferred, as further explained in Section 6. The advection term of Eq. (23) is linearised with a Newton linearisation,


## �

f ˜ρ f ϑ f A f = �

f


## � ˜ρ(n) f ϑ(n+1) f + ˜ρ(n+1) f ϑ(n) f −˜ρ(n) f ϑ(n) f � A f , (25)


## where superscript (n) denotes values from the previous nonlinear iteration, with the implicit formulation of the advecting velocity ϑ(n+1) f given as


## ϑ(n+1) f = ¯u(n+1) f ,i n f ,i −ˆd f


## ⎡


## ⎣p(n+1) Q −p(n+1) P �s f −ρ∗(n) f


## ⎛


## ⎝1 −l f


# ρ(n) P


## ∂p ∂xi


## �����

(n)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq005.png)


## ∂p ∂xi


## �����

(n)

Q


## ⎞


## ⎠s f ,i


## ⎤


## ⎦


## + ˆd f ρ∗(t−�t1) f


## �t1


## � ϑ(t−�t1) f −¯u(t−�t1) f ,i n f ,i � .


## (26)

F. Denner et al. / Journal of Computational Physics 367 (2018) 192–234 199


### Applying the same discretisation principles to the momentum and energy equations, the discretised momentum equations (1) become


# ρP uP, j −ρ(t−�t1) P u(t−�t1) P, j �t1 V P + �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq006.png)


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq007.png)


### and the discretised energy equation (3) is given as


# ρPhP −ρ(t−�t1) P h(t−�t1) P �t1 V P + �


![Equation](images/2018_Denner_Xiao_vanWachem_ACID_pressure_based_interfacial_flows_eq008.png)

Both the transient terms and the advection terms of the momentum and energy equations are linearised with a Newton linearisation. With primary solution variable χ, i.e. velocity u j in the momentum equations (27) and specific total enthalpy h in the energy equation (28), the transient terms are linearised as


# ρPχP −ρ(t−�t1) P χ(t−�t1) P �t1 V P = ρ(n) P χ(n+1) P + ρ(n+1) P χ(n) P −ρ(n) P χ(n) P −ρ(t−�t1) P χ(t−�t1) P �t1 V P , (29)


# where ρ(n+1) P is given by Eq. (24), and the advection terms are linearised with respect to the density and the advecting velocity, following as


## �

f ˜ρ f ϑ f ˜χ f A f = �

f


## � ˜ρ(n) f ϑ(n) f ˜χ(n+1) f + ˜ρ(n) f ϑ(n+1) f ˜χ(n) f + ˜ρ(n+1) f ϑ(n) f ˜χ(n) f −2 ˜ρ(n) f ϑ(n) f ˜χ(n) f � A f , (30)

where ϑ(n+1) f is given by Eq. (26). Both the spatial pressure derivative on the right-hand side of Eq. (27) and the transient pressure derivative on the right-hand side of Eq. (28) are treated implicitly.

The applied Newton linearisation promotes a smooth transition from elliptic/parabolic to hyperbolic behaviour of the governing equations in different Mach number regimes [50,92,93], which is widely understood to be of particular importance for the continuity equation [50,61,76], where it also provides the required pressure-velocity coupling at low Mach numbers [76,90]. Furthermore, the Newton linearisation of all governing equations facilitates an implicit contribution of active flow-dependent variables, i.e. the density and the advecting velocity, in the context of the presented fully-coupled pressure-based algorithm, which improves the performance and stability of the solution algorithm.


### 3.6. A note on accuracy and conservation

The accuracy of both the underpinning finite-volume formulation (Section 3.1) and the MWI used to evaluate the advecting velocity (Section 3.4) is second order [51,82]. Thus, applying a second-order TVD scheme, such as the Minmod scheme considered in this study, for the discretisation of the advection terms and the Second-Order Backward Euler scheme for the discretisation of the transient terms, the discretised governing equations are overall second-order accurate. Moreover, because the velocity u is treated separately from the density ρ, rather than treating volumetric momentum ρu as a single quantity, a second-order discretisation can be applied to ˜u f at contact discontinuities (including fluid interfaces), where the density is discontinuous and ˜ρ f requires a low-order discretisation to avoid spurious oscillations [94]. Although Ham and Iaccarino [89] identified a numerical dissipation of kinetic energy associated with the low-pass pressure filter of the MWI, this numerical dissipation has a marginal magnitude and is proportional to �x2. As a consequence, the numerical dissipation of momentum and energy is negligible, since viscous stresses and heat conduction are neglected in this study, except at nonlinear waves where numerical dissipation is introduced by the TVD scheme to regulate spurious oscillations.

Although the governing flow equations are solved for p, u and h, the governing equations are discretised in conservative form for mass, Eq. (23), momentum, Eq. (27), and energy, Eq. (28). The density is not a solution variable but is defined by the applied EOS as a function of pressure, temperature and the fluid properties. As demonstrated by Van Doormaal et al. [61] in the context of a pressure-based algorithm, choosing primitive variables (h is not a primitive variable but is treated in the same way) as primary solution variables, instead of the conserved variables, in a fully-conservative formulation of the governing conservation laws does not affect the conservation properties, if a consistent and appropriate discretisation is applied (see also [50,68,74,76] for further examples). To this end, the continuity equation acts as a constraint on the pressure field and results in velocities, through the coupling with the momentum equations, and densities, via the applied EOS, that satisfy the conservation of mass in all Mach number regimes [61]. Furthermore, the same discretisation of ˜ρ f and the same advecting velocity ϑ f are applied in the discretised governing equations, ensuring a consistent formulation of the fluxes. Consequently, when the system of discretised nonlinear governing equations is converged, the conservative form of the governing conservation laws is satisfied on the discrete level within a predefined solution tolerance. To achieve convergence of the nonlinear system of governing equations, the iterative solution procedure described in Section 6 is applied.

