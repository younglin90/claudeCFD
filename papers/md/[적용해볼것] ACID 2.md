[Journal of Computational Physics 367 \(2018\) 192–234](https://doi.org/10.1016/j.jcp.2018.04.028)

![](_page_0_Picture_3.jpeg)

Contents lists available at [ScienceDirect](http://www.ScienceDirect.com/)

[www.elsevier.com/locate/jcp](http://www.elsevier.com/locate/jcp)

![](_page_0_Picture_7.jpeg)

## Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation

![](_page_0_Picture_9.jpeg)

Fabian Denner <sup>a</sup>*,*∗, Cheng-Nian Xiao a, Berend G.M. van Wachem <sup>b</sup>

- <sup>a</sup> *Department of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom*
- <sup>b</sup> *Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany*

#### a r t i c l e i n f o a b s t r a c t

#### *Article history:* Received 4 September 2017 Received in revised form 5 April 2018 Accepted 13 April 2018 Available online 19 April 2018

*Keywords:* Interfacial flows Compressible flows Pressure-based methods Finite-volume methods Shock-capturing Acoustics

A pressure-based algorithm for the simulation of compressible interfacial flows is presented. The algorithm is based on a fully-coupled finite-volume framework for unstructured meshes with collocated variable arrangement, in which the governing conservation laws are discretised in conservative form and solved in a single linear system of equations for velocity, pressure and specific total enthalpy, with the density evaluated by an equation of state. The bulk phases are distinguished using the Volume-of-Fluid (VOF) method and the motion of the fluid interface is captured by a state-of-the-art compressive VOF method. A new interface discretisation method is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy based on the assumption of thermodynamic equilibrium, and does not require a Riemann solver. This interface discretisation method yields a consistent definition of the fluid properties in the interface region, including a unique definition of the speed of sound and the Rankine–Hugoniot relations, and conserves the acoustic features of the flow, *i.e.* compression and expansion waves. A variety of representative test cases of gas–gas and gas–liquid flows, ranging from acoustic waves and shock tubes to shock-interface interactions in one-, twoand three-dimensional domains, is used to demonstrate the capabilities and versatility of the presented algorithm in all Mach number regimes. The propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows are predicted accurately, even for difficult cases that feature fluids with shock impedance matching, transonic shock tubes or strong shocks in gas–liquid flows, as well as on unstructured meshes.

© 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license [\(http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/)).

#### **1. Introduction**

The numerical modelling of compressible flows is associated with a number of difficulties owing to the formulation of the conservation equations, the coupling of hydrodynamic and thermodynamic variables, and high pressure ratios. In addition to the numerical difficulties encountered when modelling compressible single-phase flows, the numerical solution of compressible *interfacial* flows, in which two (or more) immiscible fluids interact, is further complicated by different fluid

*E-mail address:* [fabian.denner@gmail.com](mailto:fabian.denner@gmail.com) (F. Denner).

<sup>\*</sup> Corresponding author.

properties and speeds of sound (and, thus, Mach numbers) of the bulk phases, complex acoustic behaviour as well as the numerical treatment of the fluid interface. It has proven particularly difficult to define the discrete interface between two compressible fluids in a consistent manner, which retains the main features of the solution, such as the propagation of acoustic waves and shock waves. According to Coralic and Colonius [\[1\]](#page-39-0), numerical methods able to accurately predict the interaction of interfaces with shock waves, and in extension compressible interfacial flows in general, should satisfy the following criteria:

- 1. discrete conservation of mass, momentum and energy,
- 2. avoid the generation of spurious oscillations at the interface or at shock waves, and
- 3. provide high-order accuracy in smooth regions.

A variety of contemporary algorithms for compressible interfacial flows, *e.g.* [\[1–7\]](#page-39-0), have been shown to discretely satisfy the global conservation of mass, momentum and energy in the computational domain, which is a prerequisite for the accurate prediction of the speed of shock waves [\[8–10\]](#page-39-0), although the conservation in each individual phase is strongly dependent on the applied method. Flux-limited high-order schemes, most notably *total variation diminishing* (TVD) schemes [\[11,12\]](#page-39-0), or high-order schemes combined with artificial viscosity models [\[13\]](#page-39-0), can provide second-order accuracy in smooth regions and avoid oscillations at shocks and discontinuities. Spurious oscillations at the interface as a result of the discontinuity in fluid properties, however, remain a common issue for interface capturing methods [\[1\]](#page-39-0), in particular when the bulk phases are assumed to be in mechanical equilibrium. Karni [\[14\]](#page-39-0) and Abgrall [\[2\]](#page-39-0) were among the first to depart from a fullyconservative discretisation, *i.e.* a discretisation that is simultaneously conservative with regards to the entire computational domain and each individual phase [\[15\]](#page-39-0), in favour of avoiding spurious oscillations. The application of a non-conservative discretisation at the interface has also been suggested in the context of incompressible interfacial flows by Brackbill et al. [\[16\]](#page-39-0), to avoid spurious oscillations as a result of the changing fluid properties at the interface. A non-conservative discretisation of the governing equations is widely, and largely successfully, applied for incompressible interfacial flows, although a naïve non-conservative discretisation does not allow compression and expansion waves to pass the interface. To this end, exact and approximate Riemann solvers are widely applied to allow the exchange of information between the interacting fluids through the interface. Recent studies [\[17,18\]](#page-39-0) also suggest that a strictly conservative discretisation is important for the robustness of the solution algorithm and the predictive accuracy for incompressible interfacial flows with large density ratios.

State-of-the-art algorithms for compressible interfacial flows are almost exclusively founded on *density-based* algorithms, which solve the governing conservation equations (momentum, continuity and energy) for the momentum, density and total energy of the flow. In the *two-fluid* Baer–Nunziato (BN) model [\[19\]](#page-39-0), each phase is treated as a separate fluid with its own momentum, continuity and energy equations, with an additional transport equation for the volume fraction field of the interacting fluids (*e.g.* a colour or level-set function), or a topological equation that represents the fluid interface. BN models, also often referred to as *seven-equation models*, have well-defined hyperbolic properties [\[20–22\]](#page-39-0) and conserve the mass, momentum and total energy of each phase. However, BN models involve a considerable computational complexity due to the seven, nine or eleven governing equations for one-, twoor three-dimensional flows, respectively, which have to be coupled through relaxation terms for pressure and velocity (usually under the assumption of local thermodynamic equilibrium), including non-conservative products [\[21](#page-39-0)[–25\]](#page-40-0) and a consistently defined interface velocity [\[6,21\]](#page-39-0), which is typically based on the solution of an exact or approximate Riemann problem in a Godunov-type, Rusanov-type or Roe-type discretisation.

By assuming equilibrium of pressure and velocity in the limit of infinitely fast relaxation to mechanical equilibrium [\[6,26](#page-39-0)[,27\]](#page-40-0), BN models can be reduced to *five-equation models* [\[1,4,27–](#page-39-0)[31\]](#page-40-0), with a separate continuity equation for each phase, and shared momentum and energy equations. This model conserves the mass of each phase and globally conserves momentum and energy [\[28\]](#page-40-0). However, the total energy of each individual phase is not conserved and the relative volume change of the phases due to compression/expansion has to be incorporated in the interface transport equation [\[6,27\]](#page-39-0). Abandoning the conservation of a separate density field for each phase, and considering instead only a single density field, leads to a *fourequation model* [\[2–4\]](#page-39-0) with a single conservation equation for momentum (one for each space dimension), for mass and for energy, plus an interface transport equation. This approach is also frequently referred to as the *one-fluid model*, because the entire two-phase flow (both phases and mixtures thereof) is numerically treated as one fluid with locally varying properties. It is the simplest model able to simulate interfacial flows, but does not by design conserve discretely the momentum, mass or energy of each phase. Furthermore, the four-equation model is associated with difficulties when recovering the pressure field in density-based frameworks [\[4,15,27\]](#page-39-0), for which a specific equation of state (EOS), *e.g.* a stiffened gas EOS, is required. To this end, the five-equation model of Allaire et al. [\[4\]](#page-39-0), for instance, reduces to the four-equation model of Shyue [\[3\]](#page-39-0) when both phases are perfect gases.

In the vast majority of methods, such as in the seven-, fiveand four-equation models discussed above, a Riemann solver is applied to evaluate the fluxes at cell-faces of the computational mesh, effectively reducing the evaluation of the fluxes to a one-dimensional problem for which a Riemann problem is solved [\[32\]](#page-40-0). Assuming an interface coincides with a cell-face, a Riemann problem with different EOS can be solved at the fluid interface to couple the bulk phases in an effective manner [\[33\]](#page-40-0). Riemann solvers have a strong mathematical foundation, but exact Riemann solvers are for many practical cases prohibitively time consuming, which motivates the development of approximate Riemann solvers. The Harten–Lax–van Leer Contact (HLLC) formulation, introduced by Toro et al. [\[34\]](#page-40-0), has become the most widely used approximate Riemann solver for interfacial flows, as it captures contact discontinuities. However, because approximate Riemann solvers rely on an accurate *a priori* approximation of the wave speeds, their derivation for complex interfacial flows can be cumbersome and the resulting methods are often limited to a specific application or parameter range. Various HLLC-type solvers have been proposed for interfacial flows [\[1,35–](#page-39-0)[37\]](#page-40-0), including phase transition [\[38,39\]](#page-40-0) and surface tension [\[30,38,39\]](#page-40-0). The *Ghost-Fluid Method* (GFM), originally proposed by Fedkiw et al. [\[40\]](#page-40-0), presents an alternative that does not, in general, require the solution of a Riemann problem. The GFM defines ghost cells on either side of the interface for the discretisation of the governing equations, where the *real fluid* and the *ghost fluid* coexist, and ghost values are defined in these ghost cells for the fluid on the opposing side of the interface. Due to its conceptual simplicity, the GFM has been used in a variety of numerical frameworks for compressible interfacial flows, notably [\[41,42\]](#page-40-0). The GFM has also been extended to interfacial flows with reactions [\[43,44\]](#page-40-0) and phase transition [\[44\]](#page-40-0), and has been used in conjunction with discontinuous Galerkin methods [\[45\]](#page-40-0). However, the original GFM was found to lack stability for strong shock-interface interactions and compressible gas–liquid flows, a shortcoming which was addressed by coupling the GFM to a Riemann solver to compute the flow states at the interface [\[46–49\]](#page-40-0).

Density-based algorithms are arguably the method of choice for flows with considerable compressibility, but are illsuited for low-Mach number flows [\[50–52\]](#page-40-0), requiring sophisticated preconditioning and solution methods, see *e.g.* [\[53–57\]](#page-40-0). The pressure field has to be reconstructed based on the applied EOS, since pressure is not a solution variable in densitybased algorithms. This is generally less of a problem when GFM [\[40,42,47\]](#page-40-0) or similar so-called *sharp* interface methods are used, where either phase is present in a mesh cell but never a mixture of them. However, interfacial mesh cells (where the normalised interface indicator function, *e.g.* a volume fraction, is 0 *< - <* 1) contain a "mixture" of both phases when interface capturing methods or *diffuse* interface methods are applied, such as the Volume-of-Fluid method (VOF). This fluid mixture, which Abgrall and Saurel [\[23\]](#page-39-0) aptly called *numerical mixture*, is of numerical origin and, from a continuum mechanics standpoint, has no physical basis. Reconstructing the pressure in such interfacial cells is problematic [\[4,15,27\]](#page-39-0), since hydrodynamically and thermodynamically plausible fluid properties as well as a mixture EOS have to be defined, which may misrepresent the interface region by presuming a physical meaning of the finite interface thickness. As a consequence, the speed of sound in the interface region can, for instance, be lower than the speed of sound of either bulk phase [\[6,58](#page-39-0)[–60\]](#page-40-0), which is inconsistent with the zero thickness of the interface assumed in continuum mechanics.

*Pressure-based* algorithms, in which the continuity equation serves as an equation for pressure, while density is evaluated explicitly using a suitable EOS, are preferably applied for incompressible flows and yield significant advantages for low-Mach number flows, since density is not required to be a thermodynamic variable and the acoustic degeneration (*i.e.* vanishing pressure-density coupling) at low Mach numbers does not pose a problem. The success of pressure-based algorithms is facilitated by the unique role of pressure in all Mach number regimes, with the pressure-velocity coupling dominant at low Mach numbers and the pressure-density coupling dominant at high Mach numbers [\[61,62\]](#page-40-0), as well as the convenient fact that the fully-conservative formulation of the governing conservation laws can still be satisfied accurately, even if non-conserved quantities, such as pressure, are chosen as primary solution variables [\[61\]](#page-40-0). However, ensuring a stable numerical solution in the transonic regime, where pressure is strongly coupled with both velocity and density, and formulating consistent shockcapturing schemes for pressure-based algorithms has proven to be difficult [\[51,63\]](#page-40-0). Although, since its original conception by Harlow and Amsden [\[64,65\]](#page-40-0), many pressure-based algorithms have been proposed for compressible single-phase flows [\[50,61,66](#page-40-0)[–76\]](#page-41-0), no pressure-based algorithm for compressible interfacial flows has been published yet.

In this article, a fully-coupled pressure-based algorithm for compressible interfacial flows is proposed, based on the finite-volume framework of Xiao et al. [\[76\]](#page-41-0) for compressible single-phase flows on unstructured meshes. The presented algorithm uses a compressive VOF method [\[77\]](#page-41-0) for the advection of the interface, leading to a four-equation model. The discretised governing flow equations (momentum, continuity and energy) are solved in a single system of linearised equations for velocity, pressure and specific total enthalpy, with density being evaluated based on the applied EOS. The present study considers the perfect-gas and stiffened-gas models to describe interfacial flows of gases and liquids. A new discretisation method at the fluid interface is proposed, derived from an analogy with a contact discontinuity, that performs local changes to the discrete values of density and total enthalpy in the interface region, assuming local mechanical and thermal equilibrium, whereby the conservative discretisation of the governing equations remains unchanged. Since the proposed algorithm is pressure-based and applies a single pressure field for the entire computational domain, no partial pressures have to be considered and no mixture pressure has to be defined at the interface. Instead, only mixture rules for the fluid properties have to be defined. As demonstrated by representative test cases of compressible gas–gas and gas–liquid flows, the presented algorithm predicts the propagation, reflection and transmission of acoustic waves, shock waves and rarefaction fans in interfacial flows accurately. In particular, the precise simulation of the reflection and transmission of acoustic waves at fluid interfaces as conducted in this study has not been reported in the literature before.

In Section [2,](#page-3-0) the governing equations are introduced. The numerical framework and the pressure-based algorithm are presented in Section [3](#page-4-0) and the applied compressive VOF method is described in Section [4.](#page-8-0) The new interface discretisation method is proposed in Section [5,](#page-9-0) followed by a discussion of the iterative solution procedure in Section [6.](#page-14-0) The results for a variety of representative compressible interfacial flows are presented and discussed in Section [7.](#page-15-0) The article is summarised and concluded in Section [8.](#page-38-0)

#### 2. Governing equations

The considered compressible interfacial flows of inviscid fluids at all speeds are governed, assuming Cartesian coordinates, by the momentum equations

$$\frac{\partial \rho u_j}{\partial t} + \frac{\partial \rho u_i u_j}{\partial x_i} = -\frac{\partial p}{\partial x_i} \,, \tag{1}$$

the continuity equation

$$\frac{\partial \rho}{\partial t} + \frac{\partial \rho u_i}{\partial x_i} = 0 \,, \tag{2}$$

and the energy equation

$$\frac{\partial \rho h}{\partial t} + \frac{\partial \rho u_i h}{\partial x_i} = \frac{\partial p}{\partial t} \,, \tag{3}$$

where t is time,  $\rho$  is the density,  ${\bf u}$  is the velocity vector, p is the pressure and  $h=c_p\,T+{\bf u}^2/2$  is the specific total enthalpy, with  $c_p$  the specific isobaric heat capacity and T the temperature. The enthalpy formulation is chosen for the energy equation, rather than the more common internal energy formulation, as it leads to a straightforward application in the proposed numerical algorithm, since the transient pressure term on the right-hand side of Eq. (3) does not require linearisation. The governing equations require closure by defining the thermodynamic properties of the fluids through an appropriate EOS.

In this study, the stiffened-gas model [78] is applied, which provides a very good description of liquids and solids of practical interest [33,79], and reduces to the perfect-gas model for gases. It is, therefore, widely used for interfacial flow modelling. The thermodynamic properties of the fluid are linked via the stiffened gas EOS

$$\rho = \frac{p + \gamma \,\Pi}{R \,T} \,, \tag{4}$$

where R is the specific gas constant,  $\gamma = c_p/c_v$  is the heat capacity ratio,  $c_v$  is the specific isochoric heat capacity and  $\Pi$  is a material-dependent pressure constant, which is  $\Pi = 0$  for an ideal gas. The speed of sound for a stiffened gas is

$$a = \sqrt{(\gamma - 1)c_p T} = \sqrt{\gamma \frac{p + \Pi}{\rho}}$$
 (5)

from which the specific total enthalpy follows as

$$h = \frac{a^2}{\gamma - 1} + \frac{\mathbf{u}^2}{2} = c_{p,0} \frac{p + \Pi}{p + \gamma \Pi} T + \frac{\mathbf{u}^2}{2} = c_p T + \frac{\mathbf{u}^2}{2},$$
 (6)

with the specific isobaric heat capacity

$$c_p = c_{p,0} \frac{p + \Pi}{p + \nu \Pi} \tag{7}$$

and  $c_{p,0} = R/(1-\gamma^{-1})$ . For  $\Pi = 0$ , the stiffened-gas model reduces to the perfect-gas model.

In order to distinguish the interacting bulk phases, the VOF method [80] is applied to represent the bulk phases by an indicator function  $\psi(\mathbf{x})$ , typically called *colour function*, with

$$\psi(\mathbf{x}) = \begin{cases} 0 & \text{if } \mathbf{x} \in \Omega_{\mathbf{a}} \\ 1 & \text{if } \mathbf{x} \in \Omega_{\mathbf{b}} \end{cases}$$
 (8)

where  $\Omega = \Omega_a \cup \Omega_b$  is the computational domain, with  $\Omega_a$  and  $\Omega_b$  the subdomains occupied by fluid a and b, respectively. Consequently, the interface is located in cells with  $0 < \psi < 1$ . Since the density is discontinuous at the fluid interface and fluid does not flow through the interface (assuming no mass transfer), the interface between two immiscible fluids represents a contact discontinuity [23,78]. Thus, the fluid interface is a material front propagating with the flow and, consequently, the material derivative of  $\psi$  is

$$\frac{D\psi}{Dt} = \frac{\partial\psi}{\partial t} + u_i \frac{\partial\psi}{\partial x_i} = 0. \tag{9}$$

Accounting, in addition, for the different acoustic properties of the bulk phases [6,59], Eq. (9) becomes

$$\frac{\partial \psi}{\partial t} + u_i \frac{\partial \psi}{\partial x_i} - K \frac{\partial u_i}{\partial x_i} = 0 , \qquad (10)$$

where

$$K = \frac{\rho_{\rm b} a_{\rm b}^2 - \rho_{\rm a} a_{\rm a}^2}{\frac{\rho_{\rm a} a_{\rm a}^2}{1 - \psi} + \frac{\rho_{\rm b} a_{\rm b}^2}{\psi}},\tag{11}$$

is a material-dependent compressibility factor [6,59,60], with  $\rho$  given by Eq. (4) and a given by Eq. (5). Assuming the bulk phases are in mechanical equilibrium, and since no mass transfer and surface tension are considered in this study, the interface conditions for velocity and pressure are  $\mathbf{u}_a \cdot \mathbf{m} = \mathbf{u}_b \cdot \mathbf{m}$  and  $p_a = p_b$  [81], respectively, where  $\mathbf{m}$  is the normal vector of the interface.

#### 3. Numerical framework

The presented algorithm is founded on a fully-coupled pressure-based numerical framework, based on the algorithm for compressible single-phase flows of Xiao et al. [76]. The framework is predicated on the finite-volume method with a collocated variable arrangement, and all discretisation methods presented below are applicable to unstructured meshes. Since the discretisation of the governing equations is largely identical for single-phase and interfacial flows, the discretisation of the governing equations presented in this section focuses on single-phase flows. The modifications to this discretisation for interfacial flows, including the appropriate definition of the fluid properties, is described in Section 5 and the iterative solution procedure applied to solve the discretised, linearised governing equations is described in Section 6.

#### 3.1. Finite-volume method

The finite-volume method, which forms the foundation for the numerical framework, is based on the integral formulation of the governing conservation laws. It is worth recalling that only the integral form of the governing conservation laws is valid at shocks and discontinuities [32]. Integrating, for example, the continuity equation (2) over the control volume V, the integral form of Eq. (2) is given as

$$\iiint\limits_{V} \frac{\partial \rho}{\partial t} dV + \iint\limits_{dV} \rho u_{i} dS_{i} = 0 , \qquad (12)$$

where

$$\iiint\limits_{V} \frac{\partial \rho u_{i}}{\partial x_{i}} dV = \iint\limits_{dV} \rho u_{i} dS_{i}$$
 (13)

follows from the divergence theorem, with S the outward pointing surface vector on the surface dV of control volume V. Assuming the surface of the control volume is constituted by a finite number of flat faces f and applying the midpoint rule [62,82], the surface integral can be expressed with second-order accuracy [82] as

$$\oint_{dV} \rho u_i dS_i \approx \sum_f \rho_f \vartheta_f A_f ,$$
(14)

where  $A_f$  is the area of face f and  $\vartheta_f = \boldsymbol{u}_f \boldsymbol{n}_f$  is the advecting velocity, with  $\boldsymbol{n}_f$  the outward pointing unit normal vector of face f. Because a collocated variable arrangement is employed, all primary solution variables  $(\boldsymbol{u}, p, h)$  as well as the density  $\rho$  are stored at cell centres. Therefore, the data required at face centres is obtained by interpolation from the data at adjacent cell centres, as further explained in Section 3.2, while the face velocity  $\boldsymbol{u}_f$  requires a special interpolation to prevent pressure-velocity decoupling, discussed in detail in Section 3.4. The discretisation of the temporal term is discussed in Section 3.3. The interested reader is referred to the work of Xiao et al. [76] for further details on the applied finite-volume method.

#### 3.2. Spatial discretisation

The *central differencing* scheme is applied for the interpolation from cell centres to face centres of variables that are not advected, given for a general flow variable  $\phi$  as

$$\bar{\phi}_f = \phi_P + l_f(\phi_Q - \phi_P) \ . \tag{15}$$

The geometry coefficient  $l_f$  is defined based on an inverse distance weighting, given as  $l_f = |\mathbf{r}_{Pf}|/\Delta s_f$ , where  $\Delta s_f$  is the distance between cell centres P and Q (the cell centres adjacent to face f), schematically illustrated in Fig. 1a, and  $\mathbf{r}_{Pf}$  is the vector connecting cell centre P with face centre f.

![](_page_5_Picture_2.jpeg)

**Fig. 1.** Schematic illustration of (a) cell P with its neighbour cell Q and the shared face f, where  $\mathbf{n}_f$  is the unit normal vector of face f and  $\mathbf{s}_f$  is the unit vector connecting cells P and Q (both outward pointing with respect to cell P), and (b) upwind cell Q and downwind cell Q of face Q, where Q represents the velocity vector.

Advected variables are interpolated to face centres using the TVD interpolation method for general unstructured meshes proposed by Denner and van Wachem [83]. Considering the velocity vector  $\boldsymbol{u}$  as indicated in Fig. 1b, the face value follows as (where  $\sim$  denotes a flux-limited interpolation)

$$\tilde{\phi}_f = \phi_U + \frac{\xi_f}{L_f} (\phi_D - \phi_U) , \qquad (16)$$

where subscripts U and D denote the upwind and downwind cells,  $\xi_f$  is the flux limiter and  $L_f = \Delta s_f/|\mathbf{r}_{Uf}|$  is a geometry coefficient. Dependent on the studied flow, the first-order upwind scheme ( $\xi_f = 0$ ), the central differencing scheme ( $\xi_f = 1$ ) or the Minmod scheme [84] are applied for the simulations presented as part of this study, but other TVD schemes could also be applied.

#### 3.3. Temporal discretisation

The First-Order Backward Euler scheme or the Second-Order Backward Euler scheme are applied for the discretisation of the transient terms of the governing flow equations. The First-Order Backward Euler scheme is readily given for cell *P* as

$$\int_{V_P} \frac{\partial \phi}{\partial t} \, dV \approx \frac{\phi_P - \phi_P^{(t - \Delta t_1)}}{\Delta t_1} \, V_P \,, \tag{17}$$

with  $\Delta t_1$  being the current time-step, superscript  $(t - \Delta t_1)$  denotes values of the previous time-level and  $V_P$  is the volume of mesh cell P. Assuming a varying time-step is applied to solve the governing equations, the Second-Order Backward Euler scheme is given as [85]

$$\int_{V_P} \frac{\partial \phi}{\partial t} dV \approx \left( \frac{1}{\Delta t_1} - \frac{\Delta \tau}{\Delta t_1^2} \right) \left[ \left( 1 - \frac{\Delta t_1^2}{\Delta \tau^2} \right) \phi_P - \phi_P^{(t - \Delta t_1)} + \frac{\Delta t_1^2}{\Delta \tau^2} \phi_P^{(t - \Delta \tau)} \right] V_P , \qquad (18)$$

where  $\Delta \tau = \Delta t_1 + \Delta t_2$ ,  $\Delta t_2$  is the previous time-step and superscript  $(t - \Delta \tau)$  denotes values of the previous-previous time-level. If  $\Delta t_1 = \Delta t_2$ , Eq. (18) reverts to the Second-Order Backward Euler scheme for a constant time-step [82]

$$\int_{V_P} \frac{\partial \phi}{\partial t} dV \approx \frac{3\phi_P - 4\phi_P^{(t-\Delta t_1)} + \phi_P^{(t-2\Delta t_1)}}{2\Delta t_1} V_P.$$
(19)

In what follows, the discretised governing equations are presented using the First-Order Backward Euler scheme, although the Second-Order Backward Euler scheme is also considered in the results presented in Section 7. Irrespective of the chosen scheme, for consistency all transient terms of the governing equations,  $\partial \rho u_j/\partial t$  in Eq. (1),  $\partial \rho/\partial t$  in Eq. (2), as well as  $\partial \rho h/\partial t$  and  $\partial \rho/\partial t$  in Eq. (3), are discretised with the same scheme.

#### 3.4. Advecting velocity

A momentum-weighted interpolation (MWI) method is applied to evaluate the advecting velocity  $\vartheta_f = \boldsymbol{u}_f \boldsymbol{n}_f$  at cell faces f, with  $\vartheta_f$  taking the role of flux-velocity in the discretised advection terms of the governing equations. MWI emulates a staggered variable arrangement by introducing a cell-to-cell pressure coupling through a low-pass filter acting on the third derivative of pressure [51,82,86], which avoids pressure-velocity decoupling as a result of the collocated variable arrangement and provides a robust pressure-velocity coupling for incompressible and low-Mach number flows [76], while preserving the second-order accuracy of the finite-volume method [51,82].

Following previous studies, the definition of the advecting velocity includes various modifications to the MWI formulation originally introduced by Rhie and Chow [87], to account for non-orthogonal meshes [88–90], as well as the large density

ratios occurring in interfacial flows and the transient nature of the considered problems [90]. Based on the work of Denner and van Wachem [90], the advecting velocity  $\vartheta_f = u_f n_f$  at face f, see Fig. 1a, is given as

$$\vartheta_{f} = \bar{u}_{f,i} n_{f,i} - \hat{d}_{f} \left[ \frac{p_{Q} - p_{P}}{\Delta s_{f}} - \rho_{f}^{*} \left( \frac{1 - l_{f}}{\rho_{P}} \frac{\partial p}{\partial x_{i}} \Big|_{P} + \frac{l_{f}}{\rho_{Q}} \frac{\partial p}{\partial x_{i}} \Big|_{Q} \right) s_{f,i} \right] 
+ \hat{d}_{f} \frac{\rho_{f}^{*(t - \Delta t_{1})}}{\Delta t_{1}} \left( \vartheta_{f}^{(t - \Delta t_{1})} - \bar{u}_{f,i}^{(t - \Delta t_{1})} n_{f,i} \right).$$
(20)

The coefficient  $\hat{d}_f$  is defined as

$$\hat{d}_f = \frac{\left(\frac{V_P}{e_P} + \frac{V_Q}{e_Q}\right)}{2 + \frac{\rho_f^*}{\Delta t_1} \left(\frac{V_P}{e_P} + \frac{V_Q}{e_Q}\right)},\tag{21}$$

where  $e_P$  and  $e_Q$  are the sum of the coefficients of the primary variable  $\boldsymbol{u}$  arising from the advection terms (and, if viscous flows are considered, the shear stress terms) of the momentum equations (see [90] for a detailed derivation). The face density  $\rho_f^*$  is interpolated by a harmonic average,

$$\frac{1}{\rho_f^*} = \frac{1 - l_f}{\rho_P} + \frac{l_f}{\rho_Q} \,, \tag{22}$$

which is necessary for a consistent definition of the coefficient of the pressure term as well as the efficacy of the density weighting [90]. This density weighting of the pressure gradients, with which robust incompressible interfacial flow simulations with density ratios of up to  $10^{24}$  [90,91] were previously presented, stabilises the numerical solution of interfacial flows with large density ratios.

#### 3.5. Discretised governing equations

Applying the spatial and temporal discretisation techniques described above to the integral formulation of the continuity equation, Eq. (12), the discretised continuity equation for cell P follows as

$$\frac{\rho_p^{(n+1)} - \rho_p^{(t-\Delta t_1)}}{\Delta t_1} V_P + \sum_f \tilde{\rho}_f \vartheta_f A_f = 0, \tag{23}$$

with the advecting velocity  $\vartheta_f$  given by Eq. (20) and superscript (n+1) denotes variables that are solved for implicitly. Although the discretised continuity equation is formulated conservative in mass, it is treated as an equation for pressure. Hence, based on Eq. (4), the implicit density  $\rho_p^{(n+1)}$  at cell centre P is given by the pressure-implicit formulation

$$\rho_P^{(n+1)} = \frac{p_P^{(n+1)} + \gamma_P \,\Pi_P}{R_P \,T_P} \,, \tag{24}$$

where the fluid properties  $\gamma$ ,  $\Pi$  and R are defined based on the most recent available interface position, as explained in Section 5, and the temperature T is deferred, as further explained in Section 6. The advection term of Eq. (23) is linearised with a Newton linearisation.

$$\sum_{f} \tilde{\rho}_{f} \vartheta_{f} A_{f} = \sum_{f} \left( \tilde{\rho}_{f}^{(n)} \vartheta_{f}^{(n+1)} + \tilde{\rho}_{f}^{(n+1)} \vartheta_{f}^{(n)} - \tilde{\rho}_{f}^{(n)} \vartheta_{f}^{(n)} \right) A_{f} , \qquad (25)$$

where superscript (n) denotes values from the previous nonlinear iteration, with the implicit formulation of the advecting velocity  $\vartheta_f^{(n+1)}$  given as

$$\vartheta_{f}^{(n+1)} = \bar{u}_{f,i}^{(n+1)} n_{f,i} - \hat{d}_{f} \left[ \frac{p_{Q}^{(n+1)} - p_{P}^{(n+1)}}{\Delta s_{f}} - \rho_{f}^{*(n)} \left( \frac{1 - l_{f}}{\rho_{P}^{(n)}} \frac{\partial p}{\partial x_{i}} \Big|_{P}^{(n)} + \frac{l_{f}}{\rho_{Q}^{(n)}} \frac{\partial p}{\partial x_{i}} \Big|_{Q}^{(n)} \right) s_{f,i} \right] + \hat{d}_{f} \frac{\rho_{f}^{*(t-\Delta t_{1})}}{\Delta t_{1}} \left( \vartheta_{f}^{*(t-\Delta t_{1})} - \bar{u}_{f,i}^{*(t-\Delta t_{1})} n_{f,i} \right) .$$
(26)

Applying the same discretisation principles to the momentum and energy equations, the discretised momentum equations (1) become

$$\frac{\rho_{P}u_{P,j} - \rho_{P}^{(t-\Delta t_{1})}u_{P,j}^{(t-\Delta t_{1})}}{\Delta t_{1}}V_{P} + \sum_{f}\tilde{\rho}_{f}\vartheta_{f}\tilde{u}_{f,j}A_{f} = -\sum_{f}\bar{p}_{f}^{(n+1)}n_{f,j}A_{f}, \qquad (27)$$

and the discretised energy equation (3) is given as

$$\frac{\rho_P h_P - \rho_P^{(t-\Delta t_1)} h_P^{(t-\Delta t_1)}}{\Delta t_1} V_P + \sum_f \tilde{\rho}_f \vartheta_f \tilde{h}_f A_f = \frac{p_P^{(n+1)} - p_P^{(t-\Delta t_1)}}{\Delta t_1} V_P . \tag{28}$$

Both the transient terms and the advection terms of the momentum and energy equations are linearised with a Newton linearisation. With primary solution variable  $\chi$ , *i.e.* velocity  $u_j$  in the momentum equations (27) and specific total enthalpy h in the energy equation (28), the transient terms are linearised as

$$\frac{\rho_P \chi_P - \rho_P^{(t-\Delta t_1)} \chi_P^{(t-\Delta t_1)}}{\Delta t_1} V_P = \frac{\rho_P^{(n)} \chi_P^{(n+1)} + \rho_P^{(n+1)} \chi_P^{(n)} - \rho_P^{(n)} \chi_P^{(n)} - \rho_P^{(t-\Delta t_1)} \chi_P^{(t-\Delta t_1)}}{\Delta t_1} V_P,$$
(29)

where  $\rho_P^{(n+1)}$  is given by Eq. (24), and the advection terms are linearised with respect to the density and the advecting velocity, following as

$$\sum_{f} \tilde{\rho}_{f} \vartheta_{f} \tilde{\chi}_{f} A_{f} = \sum_{f} \left( \tilde{\rho}_{f}^{(n)} \vartheta_{f}^{(n)} \tilde{\chi}_{f}^{(n+1)} + \tilde{\rho}_{f}^{(n)} \vartheta_{f}^{(n+1)} \tilde{\chi}_{f}^{(n)} + \tilde{\rho}_{f}^{(n+1)} \vartheta_{f}^{(n)} \tilde{\chi}_{f}^{(n)} - 2 \tilde{\rho}_{f}^{(n)} \vartheta_{f}^{(n)} \tilde{\chi}_{f}^{(n)} \right) A_{f} , \qquad (30)$$

where  $\vartheta_f^{(n+1)}$  is given by Eq. (26). Both the spatial pressure derivative on the right-hand side of Eq. (27) and the transient pressure derivative on the right-hand side of Eq. (28) are treated implicitly.

The applied Newton linearisation promotes a smooth transition from elliptic/parabolic to hyperbolic behaviour of the governing equations in different Mach number regimes [50,92,93], which is widely understood to be of particular importance for the continuity equation [50,61,76], where it also provides the required pressure-velocity coupling at low Mach numbers [76,90]. Furthermore, the Newton linearisation of all governing equations facilitates an implicit contribution of active flow-dependent variables, *i.e.* the density and the advecting velocity, in the context of the presented fully-coupled pressure-based algorithm, which improves the performance and stability of the solution algorithm.

#### 3.6. A note on accuracy and conservation

The accuracy of both the underpinning finite-volume formulation (Section 3.1) and the MWI used to evaluate the advecting velocity (Section 3.4) is second order [51,82]. Thus, applying a second-order TVD scheme, such as the Minmod scheme considered in this study, for the discretisation of the advection terms and the Second-Order Backward Euler scheme for the discretisation of the transient terms, the discretised governing equations are overall second-order accurate. Moreover, because the velocity  $\boldsymbol{u}$  is treated separately from the density  $\rho$ , rather than treating volumetric momentum  $\rho \boldsymbol{u}$  as a single quantity, a second-order discretisation can be applied to  $\tilde{\boldsymbol{u}}_f$  at contact discontinuities (including fluid interfaces), where the density is discontinuous and  $\tilde{\rho}_f$  requires a low-order discretisation to avoid spurious oscillations [94]. Although Ham and Iaccarino [89] identified a numerical dissipation of kinetic energy associated with the low-pass pressure filter of the MWI, this numerical dissipation has a marginal magnitude and is proportional to  $\Delta x^2$ . As a consequence, the numerical dissipation of momentum and energy is negligible, since viscous stresses and heat conduction are neglected in this study, except at nonlinear waves where numerical dissipation is introduced by the TVD scheme to regulate spurious oscillations.

Although the governing flow equations are solved for p, u and h, the governing equations are discretised in conservative form for mass, Eq. (23), momentum, Eq. (27), and energy, Eq. (28). The density is not a solution variable but is defined by the applied EOS as a function of pressure, temperature and the fluid properties. As demonstrated by Van Doormaal et al. [61] in the context of a pressure-based algorithm, choosing primitive variables (h is not a primitive variable but is treated in the same way) as primary solution variables, instead of the conserved variables, in a fully-conservative formulation of the governing conservation laws does not affect the conservation properties, if a consistent and appropriate discretisation is applied (see also [50,68,74,76] for further examples). To this end, the continuity equation acts as a constraint on the pressure field and results in velocities, through the coupling with the momentum equations, and densities, via the applied EOS, that satisfy the conservation of mass in all Mach number regimes [61]. Furthermore, the same discretisation of  $\tilde{\rho}_f$  and the same advecting velocity  $\vartheta_f$  are applied in the discretised governing equations, ensuring a consistent formulation of the fluxes. Consequently, when the system of discretised nonlinear governing equations is converged, the conservative form of the governing conservation laws is satisfied on the discrete level within a predefined solution tolerance. To achieve convergence of the nonlinear system of governing equations, the iterative solution procedure described in Section 6 is applied.

#### 4. Compressive VOF method

A consistent and precise transport of the colour function  $\psi$ , which represents the bulk phases and the fluid interface, is critical for the overall accuracy of the numerical algorithm. In order for the applied fluxes to be consistent, the advection term of the advection equation (10) is reformulated using the *chain rule*.

$$u_i \frac{\partial \psi}{\partial x_i} = \frac{\partial u_i \psi}{\partial x_i} - \psi \frac{\partial u_i}{\partial x_i} \,, \tag{31}$$

an approach previously applied in [31,95]. Equation (10) then becomes

$$\frac{\partial \psi}{\partial t} + \frac{\partial u_i \psi}{\partial x_i} - (\psi + K) \frac{\partial u_i}{\partial x_i} = 0. \tag{32}$$

Now both spatial derivatives of velocity can be consistently discretised with the same advecting velocity  $\vartheta_f$  as the discretised governing equations in Section 3.5.

#### 4.1. Discretisation

The compressive VOF method introduced by Denner and van Wachem [77], which does not require an explicit interface reconstruction and has been shown to conserve the volume of incompressible interfacial flows on structured and unstructured meshes accurately, is applied to advect the colour function field with Eq. (32). In the applied finite-volume framework, the colour function in cell P is given as

$$\psi_P = \frac{1}{V_P} \int_{V_P} \psi(\mathbf{x}) \, dV \,, \tag{33}$$

thus representing the discrete volume fraction of the fluid occupying subdomain  $\Omega_b$ , as defined in Eq. (8). Applying the Crank–Nicolson scheme to discretise the transient term [77,96], the discretised advection equation (32) follows as

$$\frac{\psi_P - \psi_P^{(t-\Delta t_\psi)}}{\Delta t_\psi} V_P + \sum_f \frac{\psi_f + \psi_f^{(t-\Delta t_\psi)}}{2} \vartheta_f A_f - \left(\frac{\psi_P + \psi_P^{(t-\Delta t_\psi)}}{2} + K_P\right) \sum_f \vartheta_f A_f = 0, \tag{34}$$

where  $\Delta t_{\psi}$  is the time-step applied to advect the colour function field, the advecting velocity  $\vartheta_f$  is given by Eq. (20) and the material-dependent compressibility factor K is given by Eq. (11). The face value  $\psi_f$  is interpolated using the CICSAM scheme [96], including an implicit correction of mesh skewness [77]. The CICSAM scheme is able to advect sharp interfaces with very high accuracy, as for instance demonstrated in [77,96,97], by taking into account the orientation of the interface and, contrary to other algebraic interface advection schemes, the available amount of colour function in the upwind cell that can be advected into the downwind cell over the applied time-step. Accounting for the available flux-volume, however, comes at the cost of a very small time-step  $\Delta t_{\psi}$  to maintain a sharp interface [97], and a dual time-stepping method is applied to optimise the computational performance [77], as further detailed in Section 6.

Although a compressive VOF method is applied in this study, the proposed pressure-based algorithm is not limited to a specific method for the representation and transport of the interface. The applied compressive VOF method is chosen for its versatility, as it is also applicable on unstructured meshes. Any other suitable interface capturing method, such as the class of algebraic THINC schemes for compressible flows [49,98], or interface tracking method, e.g. front-tracking methods [41,48], may equally be used in conjunction with the proposed algorithm.

#### 4.2. Validation

The compressive VOF method [77] applied as part of this study has previously been successfully used for a variety of different incompressible, *i.e.* solenoidal ( $\nabla \cdot \boldsymbol{u} = 0$ ), flows and has thereby been shown to be in excellent agreement with experimental data [77,99] and analytical solutions [91,100]. Contrary to incompressible flows, however, the applicability and accuracy of the modified method given above for compressible flows with  $\nabla \cdot \boldsymbol{u} \neq 0$  is yet to be demonstrated.

Following the work of Raessi et al. [101], a circular and a spherical interface with initial radius  $r_0 = 0.25$  m are simulated in a square and cubical domain, respectively, with edge length 1 m. In the two-dimensional (2D) case, the centre of the circular interface is situated at  $x_c = y_c = 0.5$  m and the velocity field is prescribed as

$$\mathbf{u} = \begin{pmatrix} \frac{x - x_{c}}{\sqrt{(x - x_{c})^{2} + (y - y_{c})^{2}}} \\ \frac{y - y_{c}}{\sqrt{(x - x_{c})^{2} + (y - y_{c})^{2}}} \end{pmatrix}.$$

![](_page_9_Figure_2.jpeg)

**Fig. 2.** (a) Area *A* of the of the shrinking circular interface, normalised by the initial area  $A_0$ , on an equidistant Cartesian mesh with  $\Delta x = 256^{-1}$  m as a function of time t, and (b) the error of the computed area (2D) and volume (3D) enclosed by the interface as a function of mesh spacing  $\Delta x$ .

![](_page_9_Figure_4.jpeg)

Fig. 3. Fluid domain and computational mesh, with relevant cell-centred data at the current time-level, of a (quasi) one-dimensional example of two-phase flow at constant velocity.

In the three-dimensional (3D) case, the centre of the spherical interface is situated at  $x_c = y_c = z_c = 0.5$  m and the velocity field is prescribed as

$$\mathbf{u} = \begin{pmatrix} \frac{x - x_c}{\sqrt{(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2}} \\ \frac{y - y_c}{\sqrt{(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2}} \\ \frac{z - z_c}{\sqrt{(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2}} \end{pmatrix}.$$

The velocity field is, hence, not divergence free  $(\nabla \cdot \boldsymbol{u} \neq 0)$  and is pointing to the centre of the circular or spherical interface with  $|\boldsymbol{u}| = 1$ , causing the interface to shrink and the fluid mass enclosed by the interface to reduce. Note that the flow field is fixed and the result is, therefore, independent of the discretisation of the governing flow equations.

Fig. 2a shows the evolution of the area enclosed by the circular interface on an equidistant Cartesian mesh with  $\Delta x = 256^{-1}$  m alongside the exact solution, and Fig. 2b shows the errors of the computed area inside the circular interface and of the computed volume inside the spherical interface as a function of mesh spacing  $\Delta x$ . The change in area (2D) and volume (3D) is predicted accurately and the computational errors converge with second order in both 2D and 3D, demonstrating that the modified compressive VOF method presented above is suitable for compressible flows.

#### 5. Acoustically-conservative interface discretisation

Recall the earlier observation, discussed in Section 2, that a fluid interface is a contact discontinuity. At a contact discontinuity, velocity and pressure are continuous, while density (and all variables that depend on density) is discontinuous [32,102]. In a compressible single-phase flow, the contact discontinuity is a characteristic of the solution of the governing conservation laws and represents a weak solution to the discretised governing equations; a prominent example is a contact discontinuity in a shock tube. However, the discretised governing equations do not account for the contact discontinuity associated with an interface separating two immiscible fluids.

This problem can be illustrated by considering the simple example sketched in Fig. 3, showing a two-phase flow at constant velocity  $(\partial u/\partial x = 0)$  and pressure  $(\partial p/\partial x = 0)$  in a one-dimensional domain. Because the left fluid is heavier than the right fluid, yet both fluids have the same velocity, their volumetric momentum  $\rho \mathbf{u}$  is different. Discretising the

![](_page_10_Figure_2.jpeg)

![](_page_10_Figure_3.jpeg)

(a) Original colour function field and finite-volume discretisation stencil for cell (i, j), without ACID.

(b) Approximation of the colour function with ACID for the finite-volume stencil of cell (i, j): the colour function value of cell (i, j) is taken for the whole stencil.

**Fig. 4.** Schematic illustration of the distribution of the colour function  $\psi$  in the interface region around cell (i, j) and its neighbours for the purpose of discretising the governing equations without and with ACID. The shaded cells represent the finite-volume stencil for cell (i, j) and the dotted line represents the fluid interface.

momentum equation (1) for this one-dimensional problem at cell P of the equidistant mesh shown in Fig. 3, the advection term follows, using standard finite differences with linear interpolation, as

$$\frac{\partial \rho u^2}{\partial x} \bigg|_{P} \approx \frac{\rho_E u_E^2 - \rho_W u_W^2}{2\Delta x} = -\frac{1}{2} \,, \tag{35}$$

and, with a time-step  $\Delta t$  corresponding to a Courant number of  $Co = u \Delta t / \Delta x = 0.3$ , the transient term is

$$\left. \frac{\partial \rho u}{\partial t} \right|_{P} \approx \frac{\rho_{P} u_{P} - \rho_{P}^{(t-\Delta t)} u_{P}^{(t-\Delta t)}}{\Delta t} = 1. \tag{36}$$

Hence, in order to satisfy the discretised momentum equation (27), the jump in density at the fluid interface causes an unphysical pressure gradient and corresponding acceleration of the flow [2,14,16]. Thus, without modifications to the discretisation to account for the change in fluid properties at the fluid interface, the discretised governing equations as presented in Section 3.5 yield discontinuous and typically oscillatory velocity and pressure values.

The underpinning principle of the proposed *acoustically-conservative interface discretisation* (ACID) is based on a conservative discretisation of the governing equations in each finite-volume stencil, with the aim of applying the discretisation derived for a single-phase flow in Section 3 and obtaining consistently defined fluid properties in the interface region where  $0 < \psi < 1$ , while still respecting the contact discontinuity associated with the fluid interface. For the numerical framework applied in this study, the finite-volume stencil of a given cell P, schematically shown in Fig. 4a, contains all face-neighbour cells of cell P, as well as cell P itself. In order to satisfy the assumption of a single-phase flow in the finite-volume stencil of a given cell P for the purpose of discretising the governing equations, all cells in the finite-volume stencil of cell P are assigned the colour function value of cell P. This is schematically illustrated in Fig. 4; the colour function values in cell (i,j) and its neighbourhood, shown in Fig. 4a, are taken to be  $\psi_{i-1,j} = \psi_{i,j-1} = \psi_{i,j-1} = \psi_{i,j+1} = \psi_{i,j}$ , as shown in Fig. 4b. Thus, the colour function is kept piecewise constant in the entire finite-volume stencil, which enables the application of the fully-conservative discretisation scheme presented in Section 3, identical to the one applied for single-phase flows. The relevant thermodynamic properties that are discontinuous at the interface, *i.e.* density and enthalpy, are then evaluated and discretised based on this piecewise-constant colour function field, assuming mechanical and thermal equilibrium at the interface. Note that the two-dimensional equidistant Cartesian mesh in Fig. 4 is chosen for illustration purposes; the application of this approach on unstructured and/or three-dimensional meshes is straightforward.

#### 5.1. Density treatment

In the proposed pressure-based algorithm, density  $\rho$  is not a solution variable. Instead, density is evaluated by a linear interpolation of the partial densities of the bulk phases, given as

$$\rho = \rho_{\mathsf{a}} + \psi \left( \rho_{\mathsf{b}} - \rho_{\mathsf{a}} \right) \,, \tag{37}$$

with  $\rho_a$  and  $\rho_b$  defined by the applied EOS, Eq. (4). This linear interpolation of the partial densities with respect to the colour function is necessary for the conservation of mass, momentum and energy, and is equivalent to an *isobaric closure* assumption [4,35]. For the purpose of treating the density implicitly via Eq. (24), the linear interpolation of partial densities in Eq. (37) is interchangeable with defining the specific gas constant R and the stiffened-gas term  $\gamma \Pi$  as

$$\frac{1}{R} = \frac{1 - \psi}{R_2} + \frac{\psi}{R_b} \,, \tag{38}$$

$$\frac{\gamma \Pi}{R} = (1 - \psi) \frac{\gamma_a \Pi_a}{R_a} + \psi \frac{\gamma_b \Pi_b}{R_b}, \tag{39}$$

respectively.

Assuming the colour function  $\psi$  is piecewise-constant throughout the finite-volume stencil of cell P, as described above, the density at face f of cell P is given as

$$\tilde{\rho}_f = \rho_U^{\star} + \frac{\xi_f}{L_f} \left( \rho_D^{\star} - \rho_U^{\star} \right) , \tag{40}$$

with the density at the upwind cell U and downwind cell D evaluated based on the colour function value of cell P as

$$\rho_U^* = \rho_{\mathsf{a},U} + \psi_P \left( \rho_{\mathsf{b},U} - \rho_{\mathsf{a},U} \right) \tag{41}$$

and

$$\rho_{\mathrm{D}}^{\star} = \rho_{\mathrm{a},\mathrm{D}} + \psi_{\mathrm{P}} \left( \rho_{\mathrm{b},\mathrm{D}} - \rho_{\mathrm{a},\mathrm{D}} \right) \,, \tag{42}$$

respectively, where the partial densities are evaluated using Eq. (4).

In order for the discrete density field to be defined consistently, the density values at previous time-levels are evaluated using the same procedure, with

$$\rho_p^{(t-\Delta t_1)} = \rho_{a,p}^{(t-\Delta t_1)} + \psi_P \left( \rho_{b,p}^{(t-\Delta t_1)} - \rho_{a,p}^{(t-\Delta t_1)} \right) \tag{43}$$

and, if required

$$\rho_P^{(t-\Delta\tau)} = \rho_{\mathbf{a},P}^{(t-\Delta\tau)} + \psi_P \left( \rho_{\mathbf{b},P}^{(t-\Delta\tau)} - \rho_{\mathbf{a},P}^{(t-\Delta\tau)} \right). \tag{44}$$

The density values in the bulk phases are unaffected by this discretisation.

#### 5.2. Enthalpy treatment

Similar to the treatment of density, the enthalpy in the interface region is reformulated using ACID. To ensure consistency, the total enthalpy  $H = \rho h$  at face f is assumed to be

$$\tilde{H}_{f} = H_{U}^{\star} + \frac{\xi_{f}}{L_{f}} \left( H_{D}^{\star} - H_{U}^{\star} \right) = \rho_{U}^{\star} h_{U}^{\star} + \frac{\xi_{f}}{L_{f}} \left( \rho_{D}^{\star} h_{D}^{\star} - \rho_{U}^{\star} h_{U}^{\star} \right) , \tag{45}$$

because both density  $\rho$  and the specific isobaric heat capacity  $c_p$  are properties of the respective bulk phase. The specific isobaric heat capacity is defined by a density-weighted average as

$$c_{p} = \frac{\rho_{a} c_{p,a} + \psi \left(\rho_{b} c_{p,b} - \rho_{a} c_{p,a}\right)}{\rho}, \tag{46}$$

with the partial densities given by Eq. (4), the partial specific isobaric heat capacities given by Eq. (7) and  $\rho$  given by Eq. (37).

Unlike density, the specific total enthalpy h is a primary solution variable of the applied algorithm and, therefore, cannot simply be replaced by a modified value. Thus, a deferred correction defined by a target face enthalpy  $\hat{h}_f$  is applied to the implicitly computed value at every nonlinear iteration. Following the assumption defined in Eq. (45) and the density  $\tilde{\rho}_f$  given by Eq. (40), the target enthalpy is defined as

$$\hat{h}_f = \frac{1}{\tilde{\rho}_f} \left[ \rho_U^* h_U^* + \frac{\xi_f}{L_f} \left( \rho_D^* h_D^* - \rho_U^* h_U^* \right) \right], \tag{47}$$

where  $\rho_U^{\star}$  is given by Eq. (41),  $\rho_D^{\star}$  is given by Eq. (42), and the specific total enthalpy of the upwind and downwind cells are given as

$$h_U^* = c_{p,U}^* T_U + \frac{1}{2} \mathbf{u}_U^2 \tag{48}$$

$$h_D^{\star} = c_{p,D}^{\star} T_D + \frac{1}{2} \mathbf{u}_D^2 , \qquad (49)$$

respectively. Based on Eq. (46), the specific isobaric heat capacities  $c_{p,U}^{\star}$  and  $c_{p,D}^{\star}$  are evaluated with colour function  $\psi_P$  as

$$c_{p,U}^{\star} = \frac{\rho_{a,U} c_{p,a,U} + \psi_{P} (\rho_{b,U} c_{p,b,U} - \rho_{a,U} c_{p,a,U})}{\rho_{U}^{\star}}$$
(50)

and

$$c_{p,D}^{\star} = \frac{\rho_{a,D} c_{p,a,D} + \psi_{P} (\rho_{b,D} c_{p,b,D} - \rho_{a,D} c_{p,a,D})}{\rho_{D}^{\star}}.$$
 (51)

Based on the target face enthalpy  $\hat{h}_f$ , the deferred enthalpy correction is given as

$$\delta h_f = \hat{h}_f - \tilde{h}_f \,, \tag{52}$$

and the advection term of the discretised energy equation, Eq. (28), follows as

$$\int_{V_0} \frac{\partial \rho u_i h}{\partial x_i} dV \approx \sum_f \tilde{\rho}_f \vartheta_f \left( \tilde{h}_f + \delta h_f \right) A_f . \tag{53}$$

Away from the interface in the bulk phases, the face enthalpy  $\tilde{h}_f$  is equal to the target face enthalpy  $\hat{h}_f$ , with  $\delta h_f = 0$ , and the energy conservation in the bulk phases remains unaffected.

In order for the specific total enthalpy to be defined consistently, the specific isobaric heat capacity of previous timelevels is also defined based on the current colour function  $\psi_P$ , given as

$$c_{p,P}^{\star,(t-\Delta t_1)} = \frac{\rho_{a,P}^{(t-\Delta t_1)} c_{p,a,P}^{(t-\Delta t_1)} + \psi_P \left(\rho_{b,P}^{(t-\Delta t_1)} c_{p,b,P}^{(t-\Delta t_1)} - \rho_{a,P}^{(t-\Delta t_1)} c_{p,a,P}^{(t-\Delta t_1)}\right)}{\rho_P^{(t-\Delta t_1)}},$$
(54)

with  $\rho_p^{(t-\Delta t_1)}$  given by Eq. (43), and similarly for  $c_{p,p}^{\star,(t-\Delta \tau)}$ . The specific total enthalpy at the previous time-level then follows

$$h_p^{(t-\Delta t_1)} = c_{p,p}^{\star,(t-\Delta t_1)} T_p^{(t-\Delta t_1)} + \frac{1}{2} \mathbf{u}_p^{(t-\Delta t_1),2}$$
(55)

and, if required,

$$h_p^{(t-\Delta\tau)} = c_{p,p}^{\star,(t-\Delta\tau)} T_p^{(t-\Delta\tau)} + \frac{1}{2} \boldsymbol{u}_p^{(t-\Delta\tau),2} . \tag{56}$$

#### 5.3. Thermodynamics properties

The treatments for density and enthalpy defined above directly influence the thermodynamic properties of the interface region. Of particular interest in this context are the speed of sound and the Rankine–Hugoniot relations, as they represent fundamental solutions to the conservation laws governing compressible flows.

With the density  $\rho$  given by Eq. (37) and the specific isobaric heat capacity  $c_p$  given by Eq. (46), the speed of sound is defined based on Eq. (5) as

$$a = \sqrt{(\gamma - 1) c_p T} = \sqrt{\frac{\rho_a c_{p,a} + \psi (\rho_b c_{p,b} - \rho_a c_{p,a})}{\rho \left(\frac{1 - \psi}{\gamma_a - 1} + \frac{\psi}{\gamma_b - 1}\right)} T},$$
(57)

where

$$\frac{1}{\gamma - 1} = \frac{1 - \psi}{\gamma_{\mathsf{a}} - 1} + \frac{\psi}{\gamma_{\mathsf{b}} - 1} \tag{58}$$

follows from the isobaric closure assumption applied to the density, see Eq. (38), since  $R = (\gamma - 1)c_{\nu}$ . Thus, the speed of sound is given by a uniquely defined average of the speeds of sound in the bulk phases, a condition that Allaire et al. [4] associated with a well-possedness of the model governing equations. The accurate prediction of the speed of sound by Eq. (57) in the bulk phases and in the interface region is demonstrated in Section 7.3.1.

Considering a flow with a shock wave, the conditions in the post-shock state (I) can be related to the pre-shock state (II) by the Rankine-Hugoniot relations. Based on the Rankine-Hugoniot relations of the stiffened-gas model [33,59] and assuming the pre-shock region is stationary ( $u_{\rm II}=0$ ), the pressure ratio for a given Mach number  $M_{\rm S}=u_{\rm S}/a_{\rm II}$  of the shock wave, with  $u_{\rm S}$  being the shock speed, is

$$\frac{p_{\rm I}}{p_{\rm II}} = 1 + \frac{2\gamma}{\gamma + 1} \left( M_{\rm s}^2 - 1 \right) \left( 1 + \frac{\hat{\Pi}}{p_{\rm II}} \right) \,, \tag{59}$$

where  $\hat{\Pi}$  is the effective material-dependent pressure constant of the stiffened-gas model, which follows from Eq. (5) as

$$\hat{\Pi} = \frac{\gamma - 1}{\gamma} \rho_{\text{II}} c_{p,\text{II}} T_{\text{II}} - p_{\text{II}}$$

$$\tag{60}$$

with  $\rho$  given by Eq. (37),  $c_p$  given by Eq. (46), and  $(\gamma - 1)$  and  $\gamma$  given by Eq. (58). For an ideal gas,  $\hat{\Pi} = 0$ . The density ratio across the shock wave is then given as

$$\frac{\rho_{\rm I}}{\rho_{\rm II}} = \frac{\frac{\gamma + 1}{\gamma - 1} \frac{p_{\rm I} + \hat{\Pi}}{p_{\rm II} + \hat{\Pi}} + 1}{\frac{\gamma + 1}{\gamma - 1} + \frac{p_{\rm I} + \hat{\Pi}}{p_{\rm II} + \hat{\Pi}}},$$
(61)

with which the post-shock velocity  $u_{\rm I}$  is defined as

$$u_{\rm I} = u_{\rm S} \left( 1 - \frac{\rho_{\rm II}}{\rho_{\rm I}} \right) \,. \tag{62}$$

The post-shock temperature  $T_1$  readily follows from density  $\rho_1$  and pressure  $p_1$  via the applied EOS. Hence, the propagation of shock waves is uniquely defined in each bulk phase as well as in the interface region, and the accuracy of the jump relations given in Eqs. (59)–(62) is demonstrated in Section 7.4.1.

#### 5.4. Some observations

ACID is formulated using a general finite-volume discretisation and is independent of the numerical schemes applied to discretise the governing equations or the method used to advect the interface (*e.g.* VOF, level-set or front-tracking). No Riemann solver is required to compute the fluxes, no wave patterns have to be identified and no *a priori* assumptions about the wave structure have to be made, as for instance required when applying a classical Godunov scheme [32]. The proposed method assumes that the fluid properties are piecewise constant at the interface, *i.e.* the fluid properties are formally first-order accurate, and the corresponding error vanishes as  $\Delta x \rightarrow 0$ . This is consistent with the piecewise-constant definition of the fluid properties typically applied in VOF, level-set and front-tracking methods [103] and does not affect the local first-order accuracy of the discontinuity in fluid properties [104].

The discretisation of the governing equations remains formally fully conservative, as a set of governing equations in conservative form is solved for each mesh cell. A conservative discretisation of the governing equations is regarded as a prerequisite for the accurate prediction of both the speed and strength of shock waves and for satisfying the Rankine–Hugoniot relations [8,9]. Issues associated with a discontinuous change of fluid properties at the interface are circumvented with ACID, while retaining the information associated with compression and expansion waves, thus allowing acoustic waves, shock waves and rarefaction fans to interact with the interface without artificial (numerical) obstruction. The assumed mechanical and thermal equilibrium at the interface can lead to an unphysical flux in the interface region when both bulk phases have different temperature and different pressure. However, in practice, this spurious flux is typically small and appears to be no problem for dynamic interfacial flows, as observed in the results presented below. Applying a TVD scheme, the discretisation of velocity at the interface remains high-order, which is important when capturing instabilities at the interface (e.g. the Richtmyer–Meshkov instability).

The proposed method shares conceptual similarities with the GFM [40] and with *flux-splitting* algorithms, such as the one of Abgrall and Karni [15]. GFM and ACID are both one-fluid models, since some form of a single fluid is assumed when discretising the governing equations. GFM copies variables required for the discretisation to ghost cells on "the other side" of the interface and assumes only one of the phases occupies a given mesh cell. ACID, on the other hand, defines the fluid properties in the entire computational stencil based on the value of the interface indicator function taken from the discretised cell, without making explicit assumptions about which phase, or combination of phases, is present in the cell. Similar to flux-splitting algorithms, ACID results in asymmetric fluxes at the cell-faces, *i.e.* a different flux is applied to a given face dependent on which of its adjacent cells is discretised. To clarify, using ACID, the advecting velocity  $\vartheta_f$  is symmetric at face f, but the density  $\tilde{\rho}_f$  and the specific isobaric heat capacity  $c_{p,f}$  may be asymmetric in the interface

region. While ACID evaluates the fluxes based on the advecting velocity  $\vartheta_f$  and the fluid properties only, previously presented flux-splitting algorithms, e.g. [15,45], rely on a Riemann solver to evaluate the fluxes. Furthermore, ACID also adjusts the values at previous time-levels to consistently account for the changes in fluid properties at the interface.

In preliminary studies, the proposed algorithm exhibited convergence issues in some cases where a perfect-gas fluid (e.g. air) and a stiffened-gas fluid (e.g. water) interact, if the Minmod scheme (or other TVD schemes) was applied. Instead of a monotonically reducing residual of the discretised system of equations, oscillations about a constant finite value larger than the predefined solver tolerance  $\eta$  were obtained. After careful inspection, this behaviour is attributed to the inherent nonlinearity of TVD schemes [83], because the definition of the flux limiter depends on the advected scalar field itself. As a result of the typically large differences in density  $\rho$  and specific total enthalpy h of perfect and stiffened gases, such as air and water, even small changes of the TVD flux limiter  $\xi_f$  can have a significant impact on the solution and its convergence. Since both density  $\rho$  and specific total enthalpy h are discontinuous at the interface, TVD schemes revert to the first-order upwind scheme, with the flux limiter  $\xi_f \to 0$ , to avoid an oscillatory solution. Enforcing explicitly the application of the first-order upwind scheme ( $\xi_f = 0$ ) at faces in the interface region, where  $|\psi_P - \psi_Q| > \eta$ , for the advection of density  $\rho$  and specific total enthalpy h is an effective remedy for the convergence issue associated with TVD schemes.

#### 6. Solution procedure

The linear system,  $\mathbf{A}\phi = \mathbf{b}$ , of the governing equations for a three-dimensional flow discretised as described in Section 3, including the modifications at the interface proposed in Section 5, on a computational mesh with N cells is given as

$$\begin{pmatrix} A_{u} & A_{v} & A_{w} & A_{p} & \mathbf{0} \\ B_{u} & B_{v} & B_{w} & B_{p} & \mathbf{0} \\ C_{u} & C_{v} & C_{w} & C_{p} & \mathbf{0} \\ D_{u} & D_{v} & D_{w} & D_{p} & \mathbf{0} \\ E_{u} & E_{v} & E_{w} & E_{p} & E_{h} \end{pmatrix} \cdot \begin{pmatrix} \phi_{u} \\ \phi_{v} \\ \phi_{w} \\ \phi_{p} \\ \phi_{h} \end{pmatrix} = \mathbf{b} ,$$

$$(63)$$

which is solved for the velocity vector  $\mathbf{u} = (u, v, w)^T$ , with u, v and w its Cartesian components, pressure p and specific total enthalpy h.  $\mathbf{A}_{\chi}$ ,  $\mathbf{B}_{\chi}$ ,  $\mathbf{C}_{\chi}$  are the  $N \times N$  coefficient submatrices for primary variable  $\chi$  of the momentum equations (27) associated with the x-, yand z-axes of the Cartesian coordinate system. The  $N \times N$  submatrices  $\mathbf{D}_{\chi}$  and  $\mathbf{E}_{\chi}$  contain the coefficients of primary variable  $\chi$  for the continuity equation (23) and the energy equation (28), respectively. The solution subvector  $\boldsymbol{\phi}_{\chi}$  of length N holds the solution for primary variable  $\chi$  and the right-hand side vector  $\boldsymbol{b}$  of length N holds all known contributions, which are either deferred or from previous time-levels. This linear system of equations is solved using the Block-Jacobi preconditioner and the BiCGSTAB solver of the software library PETSc [105–107]. The linear system of equations is considered to be converged if [107]

$$\|\mathbf{A}^{(n)}\boldsymbol{\phi}^{(n+1)} - \mathbf{b}^{(n)}\| < \eta \|\mathbf{b}^{(n)}\|,$$
 (64)

where  $\|\cdot\|$  denotes the  $L_2$ -norm and  $\eta$  is the predefined solution tolerance.

An *inexact Newton method* [108] is applied to account for the nonlinearity of the governing equations, with nonlinear iterations in which the linearised governing equations are solved and the deferred variables are subsequently updated. A flow chart illustrating this iterative solution procedure is shown in Fig. 5. Following the work of Xiao et al. [76], the linear system of equations (63) is solved in each nonlinear iteration by assuming the flow is *barotropic*, *i.e.* density is only a function of the pressure (but not the temperature). Note that the flow is not considered to be isothermal, but only the update of density neglects changes in temperature in the barotropic loop. When the  $L_2$ -norm of the residual vector  $\delta$  of the linear system of equations (63) satisfies

$$\|\boldsymbol{\delta}^{(n+1)}\| = \frac{\|\boldsymbol{A}^{(n+1)}\boldsymbol{\phi}^{(n+1)} - \boldsymbol{b}^{(n+1)}\|}{\|\boldsymbol{b}^{(n+1)}\|} < \eta , \tag{65}$$

the (barotropic) *inner* loop is considered converged. Subsequently, in the *outer* loop, the density is updated based on the new pressure and *new* temperature values. This iterative dual-loop procedure continues until Eq. (65) and

$$\varepsilon_{\rho}^{(m+1)} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} \left( \frac{\phi_{\rho,k}^{(m+1)} - \phi_{\rho,k}^{(m)}}{\phi_{\rho,k}^{(m)}} \right)^2} < \eta \tag{66}$$

are both satisfied simultaneously. This solution procedure was shown to be robust for single-phase flows at all speeds and without underrelaxation of the governing equations [76].

The advection of the colour function field by means of Eq. (34) is conducted in a separate linear system of equations, before the system of linearised governing fluid equations is solved in each time-step [90], see Fig. 5. A dual time-stepping

![](_page_15_Figure_2.jpeg)

**Fig. 5.** Flow chart of the iterative solution procedure applied in each time-step, with *n* the iteration counter of the (barotropic) *inner* loop and *m* the iteration counter of the *outer* loop.

method is applied to satisfy the stringent time-step constraints of the CICSAM scheme [77,97] at an only moderate increase of computational time, with a different time-step  $\Delta t_{\psi}$  applied to solve the VOF advection equation than the time-step  $\Delta t_1$  applied to solve the equations governing the fluid flow. In order for the advection of the colour function to be consistent, the fluid time-step  $\Delta t_1$  has to be equal to or an integer multiple of the VOF time-step  $\Delta t_{\psi}$ . For all presented simulations, the Courant number associated with  $\Delta t_{\psi}$  is  $\text{Co}_{\psi} = \Delta t_{\psi} |\mathbf{u}|/\Delta x \leq 0.05$ .

#### 7. Results

In order to verify and validate the proposed algorithm, the results for different representative test cases are presented: interface advection in Section 7.1; the conservation of momentum, mass and energy in Section 7.2; the propagation, reflection and transmission of acoustic waves in Section 7.3; the propagation and interface interaction of shock waves in Section 7.4; shock tube problems in Section 7.5; the interaction of shock waves with bubbles and drops in Sections 7.6–7.8. The presented results focus on the propagation of acoustic waves, shock waves and rarefaction fans in interfacial flows. Results for single-phase flows are only briefly discussed to demonstrate the accurate prediction of acoustic phenomena and shock waves by the proposed algorithm, since a broad variety of results obtained with the applied numerical framework for single-phase flows at all speeds and on unstructured meshes has been published by Xiao et al. [76].

#### 7.1. Interface advection with constant velocity

A one-dimensional interfacial flow with uniform velocity and different fluid properties of the bulk phases is considered. The one-dimensional domain has a length of 1.0 m, which is represented with 500 equidistant cells and a time-step equivalent to a Courant number of  $Co = u_0 \Delta t / \Delta x = 0.5$ . The velocity  $u_0 = 1$  m s<sup>-1</sup>, pressure  $p_0 = 10^5$  Pa and temperature  $T_0 = 300$  K are uniform throughout the domain, and the properties of the left and the right phases are  $\rho_L = 1.156$  kg m<sup>-3</sup>,  $\gamma_L = 1.4$  and  $\rho_R = 0.160$  kg m<sup>-3</sup>,  $\gamma_R = 1.6$ , respectively. The inlet velocity and temperature are  $u_{in} = u_0$  and  $T_{in} = T_0$ , respectively. Since the advection equation of the colour function  $\psi$ , Eq. (10), is linear (assuming K = 0) and the applied velocity  $u_0$  is constant, this test case is equivalent to the Riemann problem of a contact discontinuity. The flow should be unaffected by the change in density and specific heat capacity at the interface, and the interface should move with the flow

![](_page_16_Figure_2.jpeg)

**Fig. 6.** Profiles of (a) velocity, (b) pressure difference *<sup>p</sup>* = |*∂ <sup>p</sup>/∂x*| *<sup>x</sup>* and (c) density of the sharp interface advected with constant velocity *<sup>u</sup>*<sup>0</sup> = <sup>1</sup> m s−<sup>1</sup> on a mesh with 500 cells at *t* = 0*.*7 s, with and without ACID.

![](_page_16_Figure_4.jpeg)

**Fig. 7.** Profiles of (a) velocity difference *u* = |*∂u/∂x*| *x*, (b) pressure difference *p* = |*∂ p/∂x*| *x* and (c) density of the smooth interface advected with constant velocity *<sup>u</sup>*<sup>0</sup> = <sup>1</sup> m s−<sup>1</sup> on <sup>a</sup> mesh with 500 cells at *<sup>t</sup>* = <sup>0</sup>*.*<sup>7</sup> s.

at the prescribed velocity *u*0. Although this might appear to be trivial, this has been notoriously difficult to achieve with conservative methods [\[79\]](#page-41-0).

The interface is represented by a stepwise change in colour function *ψ*, initially located at *x* = 0*.*1 m. Fig. 6 shows the profiles of velocity *u*, pressure difference *p* = |*∂ p/∂x*| *x* and density *ρ* at time *t* = 0*.*7 s, with and without ACID. The interface and the associated discontinuity of the fluid properties do not affect the flow if ACID is applied, and the interface is advected with the correct speed of *u* = *u*0. The equilibrium of pressure *p* and velocity *u* as well as the stepwise change of density *ρ* at the interface are retained, contrary to previously published algorithms, *e.g.* [\[31\]](#page-40-0). Note that the discontinuity in colour function and density at the interface does not numerically diffuse, which is a common issue of *diffuse interface methods* methods for compressible flows [\[109–111\]](#page-41-0). However, if ACID is not applied, the flow field is clearly affected in an unphysical way, with large changes in velocity and pressure, and the interface is not advected with the correct speed, as evident in Fig. 6.

As a second test-case, a smooth interface with a linear variation of *ψ* from *ψ* = 1 to *ψ* = 0 over an interface thickness of 0*.*1 m (50 cells) is simulated, with the interface initially located between *x* = 0*.*1 m and *x* = 0*.*2 m. Such cases, with cells partially occupied by both fluids, present a particular difficulty for density-based algorithms with approximate Riemann solvers. The profile of the velocity difference *u* = |*∂u/∂x*|*x*, pressure difference *p* = |*∂ p/∂x*|*x* and density *ρ* at *t* = 0*.*7 s are shown in Fig. 7. The differences of pressure and velocity are negligible in the entire domain, while the density profile and, consequently, the position of the interface are predicted accurately, demonstrating that a smooth interface does not present a problem for the presented algorithm. As in the case of the sharp interface discussed above, the interface (*i.e.* the discontinuity in colour function and fluid properties) does not artificially diffuse.

#### *7.2. Conservation of momentum, mass and energy*

In order to avoid spurious oscillations at the interface, the conservation of momentum, mass and energy at the interface is often relaxed or sacrificed [\[2,5,14,23\]](#page-39-0). As discussed in Section [5,](#page-9-0) the proposed discretisation method ACID applies a correction to the discretisation at the interface to account for the change in fluid properties. The fully-conservative formulation and discretisation of the governing flow equations is, nevertheless, retained in the finite-volume discretisation stencil of each mesh cell. Furthermore, the compressive VOF method applied in the proposed algorithm was previously shown to conserve the volume of each phase in incompressible flows [\[77\]](#page-41-0).

The advection of a circular bubble with constant velocity is simulated. The fluid properties are identical to the onedimensional interface advection in Section [7.1.](#page-15-0) The computational domain is represented by an equidistant Cartesian mesh

![](_page_17_Figure_2.jpeg)

**Fig. 8.** Conservation error of (a) momentum  $\varepsilon_{\rho u}$ , (b) mass  $\varepsilon_{\rho}$  and (c) energy  $\varepsilon_{\rho h}$  with respect to the domain and the bubble, and (d) conservation error of the bubble volume  $V_b$ , for the advection of a circular bubble on an equidistant Cartesian mesh as a function of dimensionless time  $tu_0/d_0$ , where  $u_0$  is the velocity and  $d_0$  is the initial diameter of the bubble.

with 25 cells per bubble diameter. Because the domain is periodic, flows over the domain boundaries do not have to be accounted for, and the relative error associated with the conserved variables  $\Phi \in \{\rho | \mathbf{u} | \rho, \rho h\}$  is given as

$$\varepsilon_{\Phi} = \frac{\left| \int_{V} \Phi \, dV - \int_{V} \Phi^{(0)} \, dV \right|}{\int_{V} \Phi^{(0)} \, dV} \,, \tag{67}$$

where superscript (0) denotes values at reference time t = 0. Fig. 8 shows the relative conservation errors  $\varepsilon$  of momentum, mass and energy for the entire domain and for the bubble, as a function of time. The global (domain) conservation errors are negligible and the governing conservation laws are satisfied accurately. With respect to the bubble, the conservation errors are still insignificant but up to one order of magnitude higher than the corresponding global errors. This can be attributed to the volume conservation error of the applied VOF method, shown in Fig. 8d, which is the leading error of the conservation of the bubble.

#### 7.3. Acoustic waves

The propagation of acoustic waves in a one-dimensional domain is simulated to study the capabilities of the proposed algorithm to predict acoustic effects. The accurate prediction of the formation and propagation of acoustic waves is an important feature for the simulation of compressible flows and was previously found to be a particularly challenging problem [76,112], because even small inconsistencies in the discretisation or a lack of convergence can lead to a visible change in the amplitude and speed of the waves. In the presented simulations, the acoustic waves are generated at the domain-inlet by a sinusoidal velocity perturbation with (small) amplitude  $\Delta u_0$ . For small perturbations to the flow, with density amplitude  $\Delta \rho \ll \rho_0$  and velocity amplitude  $\Delta u \ll a_0$ , the resulting wave is a sound wave. The propagation speed of such a small-amplitude acoustic wave is the speed of sound  $a_0$  and, according to linear acoustic theory, the resulting amplitudes of the density waves  $\Delta \rho_0$  and the pressure waves  $\Delta p_0$  are [102]

$$\Delta \rho_0 = \rho_0 \frac{\Delta u_0}{a_0}, \quad \Delta p_0 = \rho_0 a_0 \Delta u_0. \tag{68}$$

Five different materials are considered, for which the fluid properties are given in Table 1. In all cases, the computational domain is initialised with uniform velocity  $u_0 = 1.0 \text{ m s}^{-1}$  and pressure  $p_0 = 10^5$  Pa. The central differencing scheme is applied for the presented single-phase flows and the Minmod scheme is applied for the interfacial flows, while the second-order backward Euler scheme is applied to discretise the transient terms in both single-phase and interfacial flows. At the domain-inlet, pressure and temperature are extrapolated from the closest cell centre, and at the domain-outlet, a zero-gradient condition is specified for all variables. The propagation of acoustic waves in single-fluid flows, either by considering only a single fluid or a constant mixture of two fluids (emulating the interface region between two fluids), is discussed in

**Table 1** Fluid properties for the propagation of acoustic waves. The heat capacity ratio  $\gamma$  and pressure constant  $\Pi$  of water and copper are taken from [6].

| Property | Unit                   | Air   | Helium | Argon | Water               | Copper                |
|----------|------------------------|-------|--------|-------|---------------------|-----------------------|
| γ        | -                      | 1.400 | 1.667  | 1.660 | 4.100               | 4.220                 |
| П        | Pa                     | 0     | 0      | 0     | $4.4 \times 10^{8}$ | $3.24 \times 10^{10}$ |
| $\rho_0$ | kg m <sup>-3</sup>     | 1.157 | 0.164  | 1.748 | 998                 | 8960                  |
| $a_0$    | ${\rm m}~{\rm s}^{-1}$ | 347.8 | 1008.2 | 308.2 | 1344.6              | 3906.4                |

![](_page_18_Figure_4.jpeg)

Fig. 9. Pressure and density profiles of acoustic waves with velocity amplitude  $\Delta u_0 = 0.01u_0$  and frequency  $f = 2000 \text{ s}^{-1}$  at  $t = 2.3 \times 10^{-3} \text{ s}$  in air, compared against the pressure and density profiles obtained with linear acoustic theory, Eq. (68), using the computed velocity profile. The amplitude of the pressure wave  $\pm \Delta p_0$  and density wave  $\pm \Delta \rho_0$  based on linear acoustic theory, as well as the theoretical wavelength  $\lambda_0$ , are shown as a reference.

![](_page_18_Figure_6.jpeg)

Fig. 10. Pressure and density profiles of acoustic waves with velocity amplitude  $\Delta u_0 = 0.01u_0$  and frequency  $f = 6000 \text{ s}^{-1}$  at  $t = 6.5 \times 10^{-4} \text{ s}$  in water, compared against the pressure and density profiles obtained with linear acoustic theory, Eq. (68), using the computed velocity profile. The amplitude of the pressure wave  $\pm \Delta p_0$  and density wave  $\pm \Delta \rho_0$  based on linear acoustic theory, as well as the theoretical wavelength  $\lambda_0$ , are shown as a reference.

Section 7.3.1. Section 7.3.2 presents simulations of the reflection and transmission of acoustic waves in gas–gas and gas–liquid flows, and Section 7.3.3 investigates the prediction of acoustic waves in two-phase flows with acoustic impedance matching.

#### 7.3.1. Propagation of acoustic waves

Acoustic waves in air and water, as well as in the interface region of air-helium, air-water and water-copper flows are simulated to demonstrate the accurate prediction of the speed of sound and pressure waves in the bulk phases and the interface region. The computational domain has a length of 1 m, represented by a Cartesian mesh with mesh spacing  $\Delta x = 2 \times 10^{-3}$  m, and the applied time-step corresponds to a Courant number of  $\text{Co} = a_0 \Delta t / \Delta x = 0.44 - 0.52$ . The velocity at the domain-inlet is  $u_{\text{in}} = u_0 + \Delta u_0 \sin(2\pi f t)$ , with frequency f and amplitude  $\Delta u_0 = 0.01 u_0$ .

Fig. 9 shows the computed pressure profile  $\Delta p$  and density profile  $\Delta \rho$  in air for  $f=2000~{\rm s}^{-1}$ , alongside the pressure profile  $\rho_0 a_0 \Delta u$  and density profile  $\rho_0 \Delta u/a_0$  obtained from linear acoustic theory, Eq. (68), using the computed velocity wave  $\Delta u$ . The computationally predicted amplitudes of the pressure waves  $\Delta p=\pm 4.022~{\rm Pa}$  and the density waves  $\Delta \rho=\pm 3.324\times 10^{-5}~{\rm kg~m}^{-3}$  are in excellent agreement with the theoretical amplitudes of the pressure waves  $\Delta p_0=\pm 4.025~{\rm Pa}$  and the density waves  $\Delta \rho_0=\pm 3.328\times 10^{-5}~{\rm kg~m}^{-3}$  following Eq. (68). The computed wavelength of  $\lambda=0.1740~{\rm m}$  compares also well to the theoretical wavelength  $\lambda_0=a_0/f=0.1739~{\rm m}$ . The pressure and density profile for acoustic waves in water ( $f=6000~{\rm s}^{-1}$ ) are shown in Fig. 10. The amplitude of the pressure and density waves ( $\Delta p=\pm 13416~{\rm Pa},~\Delta \rho=\pm 7.420\times 10^{-3}~{\rm kg~m}^{-3}$ ) as well as the wavelength ( $\lambda=0.2236~{\rm m}$ ) are in excellent agreement with the theoretical values based on linear acoustic theory, Eq. (68), which are  $\Delta p_0=\pm 13419~{\rm Pa},~\Delta \rho_0=\pm 7.422\times 10^{-3}~{\rm kg~m}^{-3}$  and  $\lambda_0=0.2241~{\rm m}.$ 

![](_page_19_Figure_2.jpeg)

**Fig. 11.** Computed speed of sound a as a function of colour function  $\psi$  in (a) an air–helium flow (air represented by  $\psi = 0$ , helium represented by  $\psi = 1$ ), (b) an air–water flow (air represented by  $\psi = 0$ , water represented by  $\psi = 1$ ) and (c) a water–copper flow (water represented by  $\psi = 0$ , copper represented by  $\psi = 1$ ), compared against the theoretical value given by Eq. (57).

The propagation of acoustic waves in the interface region of air-helium, air-water and water-copper two-phase flows are simulated by explicitly setting a constant value of the colour function  $\psi$  in the entire domain. The fluid properties are then evaluated based on the definitions given in Section 5. Fig. 11 shows the speed of sound based on the computed wavelength of the acoustic waves as a function of the colour function  $\psi$  for all three cases. The computed speeds of sound are in excellent agreement with the theoretical value given by Eq. (57), with an error of < 0.33%.

In summary, the propagation of acoustic waves in gases, liquids and solids is predicted accurately, especially considering that the density wave typically has an amplitude that is five orders of magnitude smaller than the ambient density. The pressure and density waves exhibit a constant amplitude as they propagate downstream, which is expected for an inviscid flow simulated with second-order discretisation schemes, as discussed in Section 3.6. In fact, at the end of a domain with a length of 50 m ( $\Delta x = 2 \times 10^{-3}$  m, Co = 0.5), the amplitude of velocity, pressure and density of an acoustic wave in air ( $f = 500 \text{ s}^{-1}$ ) are all > 99.75% of their initial value at the domain inlet. The definition of the speed of sound given in Eq. (57) is found to be accurate in the bulk phases and throughout the interface region (i.e. where a mixture of two bulk phases is present), demonstrating that ACID yields a thermodynamically-consistent discretisation.

#### 7.3.2. Reflection and transmission at fluid interfaces

The propagation of a single acoustic wave in a helium-air flow, an argon-air flow and an air-water flow is simulated in a one-dimensional domain with mesh spacing  $\Delta x = 2 \times 10^{-3}$  m and with a time-step that corresponds to  $\text{Co} = a_0 \Delta t / \Delta x = 0.48$ . The single acoustic wave is initiated by the inlet-velocity

$$u_{\rm in} = \begin{cases} u_0 + \Delta u \sin\left(2\pi f t + \frac{3}{2}\pi\right) & \text{if } t < f^{-1} \\ u_0 - \Delta u & \text{if } t \ge f^{-1}, \end{cases}$$
 (69)

with  $\Delta u_0 = 0.02 u_0$ . The frequency of the acoustic wave is  $f = 5000 \text{ s}^{-1}$  in the helium-air flow and the air-water flow, and  $f = 2000 \text{ s}^{-1}$  in the argon-air flow.

Assuming the incident acoustic wave travels from left to right, a part of the pressure wave is transmitted to the right phase when the incident wave reaches the fluid interface, while the remaining part of the incident wave is reflected in the left phase. Based on linear acoustic theory, as the incident wave with its pressure amplitude  $\Delta p_{\mathrm{L},0}^{\mathrm{incid.}}$  in the left phase reaches the interface, the ratio of the pressure amplitudes of the transmitted acoustic wave in the right phase  $\Delta p_{\mathrm{R},0}^{\mathrm{trans.}}$  and of the reflected acoustic wave in the left phase  $\Delta p_{\mathrm{L},0}^{\mathrm{refl.}}$  is

$$\frac{\Delta p_{R,0}^{\text{trans.}}}{\Delta p_{L,0}^{\text{refl.}}} = \frac{2Z_{R}}{Z_{R} - Z_{L}},\tag{70}$$

with  $\Delta p_{R,0}^{trans.} = \Delta p_{L,0}^{incid.} + \Delta p_{L,0}^{refl.}$ , and  $Z_L = \rho_L a_L$  and  $Z_R = \rho_R a_R$  are the acoustic impedance of the left and right phase, respectively. The pressure amplitude of the reflected wave, thus, follows as

$$\Delta p_{L,0}^{\text{refl.}} = \frac{\Delta p_{L,0}^{\text{incid.}}}{\frac{2Z_{R}}{Z_{R} - Z_{L}} - 1} \,. \tag{71}$$

In Fig. 12, the pressure profiles of the acoustic wave in the helium-air flow and the argon-air flow are shown, before and after the interaction of the incident wave with the fluid interface. In the helium-air case, Fig. 12a, the pressure amplitude  $\Delta p_{\text{He}}^{\text{incid.}} = 3.306 \text{ Pa}$  of the incident wave is predicted accurately compared to linear acoustic theory ( $\Delta p_{\text{He},0}^{\text{incid.}} = 3.307 \text{ Pa}$ ).

![](_page_20_Figure_2.jpeg)

**Fig. 12.** Pressure profiles of a single acoustic wave in a helium-air flow ( $f = 5000 \text{ s}^{-1}$ ) and an argon-air flow ( $f = 2000 \text{ s}^{-1}$ ), with velocity amplitude  $\Delta u_0 = 0.02 \, u_0$ . The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference.

![](_page_20_Figure_4.jpeg)

**Fig. 13.** Change in specific entropy  $\Delta s$  while a single acoustic wave propagates in the helium-air flow, as a function of dimensionless time  $t/t_{\Sigma}$ , where  $t_{\Sigma}$  is the time required for the peak of the acoustic wave to reach the interface. The inset shows a magnified view of the change in specific entropy  $\Delta s$  on a logarithmic scale before the acoustic wave has reached the interface.

![](_page_20_Figure_6.jpeg)

**Fig. 14.** Pressure profile of a single acoustic wave in an air–water flow ( $f = 5000 \text{ s}^{-1}$ ), with velocity amplitude  $\Delta u_0 = 0.02 u_0$ . The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference. Note that  $\Delta p_{\text{Air},0}^{\text{incid.}}$  and  $\Delta p_{\text{Air},0}^{\text{refl.}}$  differ by only 0.06% and are, thus, not indicated separately.

Similarly, the computed pressure amplitudes of the reflected wave in the helium phase  $\Delta p_{\rm He}^{\rm refl.}=1.377$  Pa and of the transmitted wave in the air phase  $\Delta p_{\rm Air}^{\rm trans.}=4.688$  Pa are in excellent agreement with the theoretical values ( $\Delta p_{\rm He,0}^{\rm refl.}=1.379$  Pa,  $\Delta p_{\rm Air,0}^{\rm trans.}=4.686$  Pa). In the argon-air case, Fig. 12b, the pressure amplitudes of the incident, reflected and transmitted acoustic waves also in very good agreement with linear acoustic theory. The ratio of pressure amplitudes is predicted as  $\Delta p_{\rm Air}^{\rm trans.}/\Delta p_{\rm Ar}^{\rm refl.}=-5.919$ , compared to the theoretical value of  $\Delta p_{\rm Air,0}^{\rm trans.}/\Delta p_{\rm Ar,0}^{\rm refl.}=-5.871$ . Since the presented flows of two interacting ideal gases with acoustic waves is isentropic, the change in specific entropy is  $\Delta s=0$ , where

$$\Delta s = s_2 - s_1 = c_p \ln \left(\frac{T_2}{T_1}\right) - R \ln \left(\frac{p_2}{p_1}\right). \tag{72}$$

The computed change in entropy as the acoustic wave propagates in the bulk phases of the helium–air flow is clearly negligible, as seen in the inset of Fig. 13. An increase of  $|\Delta s|$  can be observed in Fig. 13 when the acoustic wave interacts with the helium–air interface (at  $t/t_{\Sigma} \approx 1$ , with  $t_{\Sigma}$  the time required for the peak of the acoustic wave to reach the interface), a change in specific entropy that can, nevertheless, be regarded as inconsequential given its insignificant magnitude.

Fig. 14 shows the pressure profile for the acoustic wave in the air–water flow, alongside the theoretical pressure amplitude and wavelength given by linear acoustic theory. Contrary to the interaction of two perfect gases discussed above, water is described by the stiffened-gas model and, thus, large differences in acoustic impedance and density between the two interacting bulk phases ensue; the acoustic impedance and the density of water are approximately three orders of magnitude larger than for air. The amplitudes of the reflected and transmitted pressure waves are in very good agreement with linear acoustic theory, with the ratio of pressure amplitudes obtained from the simulation being  $\Delta p_{\text{Water}}^{\text{trans.}}/\Delta p_{\text{Air}}^{\text{refl.}} = 1.995$ , compared to the theoretical value of  $\Delta p_{\text{Water},0}^{\text{trans.}}/\Delta p_{\text{Air},0}^{\text{refl.}} = 2.001$ .

![](_page_21_Figure_2.jpeg)

**Fig. 15.** Pressure profiles of (a) sinusoidal acoustic waves with velocity amplitude  $\Delta u_0 = 0.01 \, u_0$  and frequency  $f = 2000 \, \text{s}^{-1}$  in an acoustically neutral two-phase flow with  $Z = 423.588 \, \text{Pa s m}^{-1}$  at  $t = 3.3 \times 10^{-3} \, \text{s}$ , and (b) a single acoustic wave with velocity amplitude  $\Delta u_0 = 0.02 \, u_0$  and frequency  $f = 5000 \, \text{s}^{-1}$  in an acoustically neutral two-phase flow with  $Z = 500 \, \text{Pa s m}^{-1}$ , before and after the acoustic wave has reached the interface. The pressure amplitude of the waves based on linear acoustic theory and the position of the fluid interface are shown as a reference.

#### 7.3.3. *Gas*–gas flow with acoustic impedance matching

An acoustically neutral two-phase flow, in which both bulk phases have the same acoustic impedance, is considered. Due to the acoustic impedance matching, no acoustic waves should be reflected at the interface. In the first case, the acoustic impedance of both bulk phases is Z = 423.588 Pa s m<sup>-1</sup> with the fluid properties

$$\rho_{\rm L} = 1.2650 \text{ kg m}^{-3}, \quad \gamma_{\rm L} = 1.40, \quad a_{\rm L} = 334.8522 \text{ m s}^{-1}$$

$$\rho_{\rm R} = 1.7537 \text{ kg m}^{-3}, \quad \gamma_{\rm R} = 1.01, \quad a_{\rm R} = 241.5396 \text{ m s}^{-1},$$

and in the second case, the acoustic impedance of both bulk phases is  $Z = 500 \text{ Pa s m}^{-1}$  with the fluid properties

$$\rho_L = 0.25 \text{ kg m}^{-3}$$
,  $\gamma_L = 9.872$ ,  $a_L = 2000 \text{ m s}^{-1}$ ,  $\rho_R = 1.00 \text{ kg m}^{-3}$ ,  $\gamma_R = 2.468$ ,  $a_R = 500 \text{ m s}^{-1}$ .

The initial velocity in both cases is  $u_0 = 0.30886 \text{ m s}^{-1}$ .

The result of the first case ( $Z = 423.588 \text{ Pa s m}^{-1}$ ) is shown in Fig. 15a for a sinusoidal velocity perturbation with  $f = 2000 \text{ s}^{-1}$  and  $\Delta u_0 = 0.01u_0$ . The observed amplitude of the pressure wave is in excellent agreement with linear acoustic theory in both phases, while the change in wavelength as a result of the different speeds of sound is predicted accurately and is clearly visible. Note that a reflected wave in the left phase would lead to a strong interference with the oncoming incident waves, which would be visible in the pressure profile.

The result of the second case ( $Z=500~{\rm Pa~s~m^{-1}}$ ) is shown in Fig. 15b for a single sinusoidal wave with  $f=5000~{\rm s^{-1}}$  and  $\Delta u_0=0.02u_0$ . A small reflected wave can be identified in the pressure profile of the left phase at  $t=0.9\times 10^{-3}~{\rm s}$ . Nevertheless, this reflected wave has only a minor effect on the transmitted wave in the right phase, which has a pressure amplitude of  $\Delta p_{\rm R}^{\rm trans.}=3.0797~{\rm Pa}$ , compared to the theoretical value of  $\Delta p_{\rm L,0}^{\rm incid.}=\Delta p_{\rm R,0}^{\rm trans.}=3.0886~{\rm Pa}$ . At the time the peak of the acoustic wave reaches the interface ( $t\approx1.1\times 10^{-3}~{\rm s}$ ), the interface is located at  $x_{\Sigma}=5.0034\times 10^{-1}~{\rm m}$  and the corresponding colour function value in the interfacial cell ( $\Delta x=2\times 10^{-3}~{\rm m}$ ) is  $\psi=0.83$ , with an acoustic impedance of  $Z=489.46~{\rm Pa~s~m^{-1}}$ . This constitutes an error in acoustic impedance at the interface of 2.1%, which is due to the definition of the fluid properties in the interface region, discussed in Section 5. This error in acoustic impedance explains the small discrepancy ( $\Delta p_{\rm L}^{\rm refl.}/\Delta p_{\rm R}^{\rm trans.}=8.77\times 10^{-3}$ ) in the pressure amplitude between the simulation ( $\Delta p_{\rm L}^{\rm refl.}=0.0270~{\rm Pa}$ ) and linear acoustic theory ( $\Delta p_{\rm L}^{\rm refl.}=0.0270~{\rm Pa}$ ).

#### 7.4. Shock waves

The simulation of the propagation of shock waves poses a particular difficulty for numerical algorithms, because shock waves are discontinuous and because valid solutions of the governing conservation laws (presented in Section 2) are not guaranteed to satisfy the second law of thermodynamics across shock waves [9]. This raises the question whether the proposed algorithm reliably converges to the physically-correct weak solution of the governing conservation laws, which is a prerequisite for the accurate prediction of both the speed and strength of strong shock waves [8,9]. The simulation of strong shock waves in air and water is examined in Section 7.4.1 and the interaction of shock waves in a single-phase flow is studied in Section 7.4.2. The interaction of shock waves with an air–helium interface and an air–water interface are presented in Sections 7.4.3 and 7.4.4, respectively. The interaction of a shock wave with an interface separating to fluids with the same shock impedance is discussed in Section 7.4.5.

#### 7.4.1. Shock wave propagation

The propagation of a shock wave travelling to the right with Mach number  $M_s = u_s/a_{II}$  in air and water, with  $\gamma$  and  $\Pi$  given in Table 1, in a computational domain with a length of 1 m is simulated. The pre-shock region (II) has pressure

![](_page_22_Figure_2.jpeg)

Fig. 16. Pressure profiles on meshes with different  $\Delta x$  of a shock wave with  $M_s = 10$  in (a) air and (b) water. The theoretical Riemann solution given by the Rankine–Hugoniot relations is shown as a reference.

![](_page_22_Figure_4.jpeg)

Fig. 17. Pressure profiles of a shock wave with  $M_s = 100$  in (a) air and (b) water on a mesh with  $\Delta x = 2 \times 10^{-3}$  m. The theoretical Riemann solution given by the Rankine–Hugoniot relations is shown as a reference.

 $p_{\rm II}=10^5$  Pa and velocity  $u_{\rm II}=0$ , with  $\rho_{\rm Air,II}=1.1574$  kg m<sup>-3</sup> and  $\rho_{\rm Water,II}=998$  kg m<sup>-3</sup>, while the post-shock region (I) is initialised based on the Rankine–Hugoniot relations [33]. The first-order backward Euler scheme and the first-order upwind scheme are applied. The shock wave is initially located at  $x_{\rm s,0}=0.1$  m and the simulations are concluded at  $t_{\rm end}=0.7$  m/ $u_{\rm s}$ . Hence, at the end of the simulation the shock wave is theoretically located at  $x_{\rm s,end}=0.8$  m. The applied time-step corresponds to Co =  $u_{\rm s}\Delta t/\Delta x=0.5$ .

Figs. 16 and 17 show the pressure profile of a shock wave with Mach numbers 10 and 100, respectively, in air and water at the end of the simulation. The pressure profiles of the shocks, and in particular the shock position, are predicted accurately in both fluids. The shock with  $M_s = 100$  in water has a pressure ratio of  $p_1/p_{II} = 7.0754 \times 10^7$  and the bulk modulus in the post-shock region is  $B_{\text{Water,I}} = \rho a^2 = 2.9 \times 10^{13}$  Pa (for comparison,  $B_{\text{Air,II}} = 1.4 \times 10^5$  Pa), which demonstrates the robustness of the proposed numerical algorithm even for strong shock waves (large pressure ratio) and marginally compressible fluids. The smearing of the shock wave reduces with mesh refinement, shown in Fig. 16, and the computed shock wave reproduces the solution of the corresponding Riemann problem with increasing precision. The common intersection point of the pressure profiles for different mesh resolutions observed in Fig. 16, which coincides with the intersection point with the corresponding Riemann solution, demonstrates that the speed of the shock wave is predicted accurately irrespective of the mesh resolution.

In Section 5.3, the Rankine–Hugoniot relations for the interface region are presented. These relations are tested using a constant air–water mixture, with the properties of the individual (pure) phases being the same as above. The domain is initialised based on the Rankine–Hugoniot relations given in Section 5.3. As for the single-phase cases discussed in the previous paragraph, the applied time-step corresponds to Co=0.5, the shock wave is initially located at  $x_{s,0}=0.1$  m and the simulations are concluded at  $t_{\rm end}=0.7$  m/ $u_{\rm s}$ . Fig. 18 shows the pressure profile of a shock wave with  $M_{\rm s}=10$  in an air–water flow with  $\psi\in\{0.25,0.50,0.75\}$ , where  $\psi=0$  represents air and  $\psi=1$  represents water. The pressure profiles are in excellent agreement with the theoretical Riemann solution and converge with mesh refinement.

Given the pressure-based formulation of the proposed algorithm, the explicit evaluation of the density may be considered the "Achilles' heel" of the numerical framework. The  $L_1$ -norm of the relative error associated with the density field, given for a computational mesh with N cells as

$$\ell_1 = \frac{1}{N} \sum_{p=1}^{N} \left| \frac{\rho_p^{\text{(comp.)}} - \rho_p^{\text{(exact)}}}{\rho_{\text{I}} - \rho_{\text{II}}} \right| , \tag{73}$$

where  $\rho_P^{(\text{comp.})}$  is the density computed with the presented numerical algorithm at cell P and  $\rho_P^{(\text{exact})}$  is the corresponding exact density given by the Riemann solution at the centre of cell P, on meshes with different resolutions for the shock wave with  $M_S = 10$  is shown in Fig. 19 for air and water single-phase flows, as well as the air-water mixture with  $\psi = 0.75$ . For

![](_page_23_Figure_2.jpeg)

**Fig. 18.** Pressure profiles on meshes with different *x* of a shock wave with *M*<sup>s</sup> = 10 in an air–water mixture with (a) *ψ* = 0*.*25, (b) *ψ* = 0*.*50 and (c) *ψ* = 0*.*75, where *ψ* = 0 represents air and *ψ* = 1 represents water. The theoretical Riemann solution given by the Rankine–Hugoniot relations is shown as a reference.

![](_page_23_Figure_4.jpeg)

**Fig. 19.** *L*1-norm of the density error <sup>1</sup> as a function of the mesh spacing *x* of a shock wave with *M*<sup>s</sup> = 10 in (a) air, (b) water and (c) an air–water mixture with *ψ* = 0*.*75.

all considered cases, the density error is of similar magnitude and converges with first order, as imposed by the applied monotone discretisation schemes and as expected for an oscillation-free numerical simulation of a shock wave [\[94\]](#page-41-0).

The error convergence under mesh refinement, supported by the improved shock resolution observed qualitatively in Figs. [16](#page-22-0) and 18, is expected from a discretely conservative algorithm and, by virtue of the *Lax–Wendroff theorem* [\[113\]](#page-41-0), suggests convergence to the physically-correct weak solution of the governing conservation laws [\[8\]](#page-39-0). Note that a better shock resolution can, of course, also be obtained by applying a TVD scheme or by reducing the applied time-step, which is not shown here in the interest of conciseness, but was previously demonstrated by Xiao et al. [\[76\]](#page-41-0) using the single-phase framework the proposed algorithm is based on.

#### *7.4.2. Shock wave interaction*

The interaction of two shock waves in a closed, one-dimensional domain, as previously studied by Woodward and Colella [\[114\]](#page-41-0), is simulated. The one-dimensional domain is 1 m in length and occupied by a gas with *γ* = 1*.*4 and *cv* = <sup>720</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1. Initially, the density and velocity are *ρ*<sup>0</sup> = <sup>1</sup> kg m−<sup>3</sup> and *<sup>u</sup>*<sup>0</sup> = <sup>0</sup> m s<sup>−</sup>1, respectively, in the entire domain. The initial pressure is *p*L*,*<sup>0</sup> = 1000 Pa in the left state (0 m ≤ *x* ≤ 0*.*1 m), *p*M*,*<sup>0</sup> = 0*.*01 Pa in the middle state (0*.*1 m *< x* ≤ 0*.*9 m), and *p*R*,*<sup>0</sup> = 100 Pa in the right state (0*.*9 m *< x* ≤ 1*.*0 m). The resulting flow involves the formation and interaction of strong shock waves, rarefaction fans and contact discontinuities. The profiles of density and velocity on equidistant meshes with 400 and 3200 cells at *t* ∈ {0*.*016 s*,* 0*.*038 s} are shown in Figs. [20](#page-24-0) and [21,](#page-24-0) respectively, alongside the results reported by Woodward and Colella [\[114\]](#page-41-0). Note that the results of Woodward and Colella [\[114\]](#page-41-0) were obtained on a mesh with 3096 cells, which was adaptively refined around flow discontinuities by factor 8, yielding an effective resolution equivalent to an equidistant mesh with 24768 cells. The results obtained on the finer mesh are in very good agreement with the reference results and a significant improvement of the solution, in particular for the density profile, is observed when the mesh spacing is reduced.

#### *7.4.3. Air–helium interface*

The interaction of a planar shock wave with a planar air–helium interface is simulated. The shock wave is initially located at *x*s*,*<sup>0</sup> = 0*.*05 m and travels in the left phase, which is assumed to be air, with speed *u*<sup>s</sup> towards the air–helium interface that is initially situated at *x,*<sup>0</sup> = 0*.*15 m. The shock wave separates the post-shock region I and the pre-shock region II, which are initialised with

$$u_{\rm I} = 125.65 \text{ m s}^{-1}, \quad p_{\rm I} = 1.59060 \times 10^5 \text{ Pa}, \quad T_{\rm I} = 402.67 \text{ K},$$
  
 $u_{\rm II} = 0 \text{ m s}^{-1}, \quad p_{\rm II} = 1.01325 \times 10^5 \text{ Pa}, \quad T_{\rm II} = 351.82 \text{ K}.$ 

![](_page_24_Figure_2.jpeg)

**Fig. 20.** Profiles of density and velocity of the shock wave interaction on equidistant meshes with 400 and 3200 cells at t = 0.016 s, compared against the reference results reported by Woodward and Colella [114].

![](_page_24_Figure_4.jpeg)

**Fig. 21.** Profiles of density and velocity of the shock wave interaction on equidistant meshes with 400 and 3200 cells at t = 0.038 s, compared against the reference results reported by Woodward and Colella [114].

![](_page_24_Figure_6.jpeg)

Fig. 22. Profiles of density, pressure and Mach number of the shock-interface interaction ( $M_s = 1.22$ ) in an air-helium flow,  $t = 2 \times 10^{-4}$  s after the shock has interacted with the interface, compared against the theoretical Riemann solution.

The heat capacity ratio and the specific heat capacity at constant volume of air are  $\gamma_{\rm Air}=1.4$  and  $c_{\nu,{\rm Air}}=720~{\rm J~kg^{-1}~K^{-1}}$ , respectively. Hence, the density of the air in the pre-shock region is  $\rho_{\rm II,Air}=1~{\rm kg~m^{-3}}$ , resulting in a speed of sound of  $a_{\rm II,Air}=376.65~{\rm m~s^{-1}}$ . Following the experiments of Haas and Sturtevant [115] and the numerical study of Quirk and Karni [116] of the interaction between a shock and a helium bubble, the helium phase is assumed to be contaminated with 28% air by mass, with a heat capacity ratio of  $\gamma_{\rm He}=1.648$  and a specific heat capacity at constant volume of  $c_{\nu,{\rm He}}=2440~{\rm J~kg^{-1}~K^{-1}}$ . The one-dimensional computational domain of 0.4 m length is represented by an equidistant mesh with 400 cells ( $\Delta x=10^{-3}~{\rm m}$ ) and a time-step of  $\Delta t=1.25\times 10^{-7}~{\rm s}$ . The velocity and temperature at the domain-inlet are  $u_{\rm in}=u_{\rm I}$  and  $T_{\rm in}=T_{\rm I}$ , and pressure is extrapolated from the closest cell centre. At the domain-outlet a zero-gradient condition is applied to all variables.

According to the corresponding Riemann solution, the shock wave is observed to travel with a speed of  $u_s = 459.50 \text{ m s}^{-1}$ , which corresponds to  $M_s = u_s/a_{\text{II},\text{air}} = 1.22$ . Fig. 22 shows the profiles of density, pressure and Mach number, at  $t = 2 \times 10^{-4}$  s after the shock has interacted with the interface. A rarefaction fan is reflected in the air phase and a shock wave transmitted in the helium phase. All computed variables compare very well with the corresponding Riemann solution, including the strength and speed of the shock wave transmitted to the helium phase, which is propagating at  $u_s = 1079.90 \text{ m s}^{-1}$  ( $M_s = 1.13$ ). The position of the contact discontinuity (coinciding with the fluid interface) and the associated change in density and Mach number are also predicted accurately, as observed in Fig. 22.

![](_page_25_Figure_2.jpeg)

Fig. 23. Profiles of density, pressure and velocity of the shock-interface interaction ( $M_s = 10$ ) in an air–water flow,  $t = 2.78 \times 10^{-4}$  s after the shock has interacted with the interface, compared against the theoretical Riemann solution.

![](_page_25_Figure_4.jpeg)

**Fig. 24.** Profiles of density, pressure and Mach number of the shock-interface interaction ( $M_s = 1.22$ ) in a two-phase flow with shock impedance matching,  $t = 2 \times 10^{-4}$  s after the shock wave has interacted with the interface, compared against the theoretical Riemann solution. The interface is located at  $x_{\Gamma} = 0.175$  m.

#### 7.4.4. Air-water interface

The interaction of a shock wave with  $M_s=10$  with an air–water interface is simulated. Using the heat capacity ratio  $\gamma$  and pressure constant  $\Pi$  of air and water given in Table 1, the post-shock region I and pre-shock region II are initialised with

$$u_{\rm I} = 2869.3~{\rm m~s^{-1}}, \quad p_{\rm I} = 1.165 \times 10^7~{\rm Pa}, \quad \rho_{{\rm air},{\rm I}} = 6.614~{\rm kg~m^{-3}}, \ u_{\rm II} = 0~{\rm m~s^{-1}}, \qquad p_{\rm II} = 10^5~{\rm Pa}, \qquad \rho_{{\rm air},{\rm II}} = 1.157~{\rm kg~m^{-3}}.$$

The applied one-dimensional domain is 1 m in length and represented with a mesh of 1000 equidistant cells ( $\Delta x = 10^{-3}$  m) and a time-step of  $\Delta t = 10^{-8}$  s. The shock wave and the air-water interface are initially situated at  $x_{\rm s,0} = 0.25$  m and  $x_{\rm \Gamma,0} = 0.50$  m, respectively. Fig. 23 shows the profiles of density, pressure and velocity, at  $t = 2.78 \times 10^{-4}$  s after the shock wave has interacted with the interface. The density profile computed with the proposed algorithm is in excellent agreement with the exact Riemann solution, despite the large density ratio of air and water. The pressure and velocity profiles are also in good agreement with the theoretical solution. However, small discrepancies can be identified in the pressure profile in comparison to the theoretical solution, with a wave forming upstream of the shock. This pressure wave actually originates during the initial contact of the shock wave with the interface, and propagates with the flow. Nevertheless, it does not have a lasting effect on the fidelity of the pressure distribution.

#### 7.4.5. Gas-gas flow with shock impedance matching

The interaction of a shock wave with a fluid interface when both bulk phases have the same shock impedance is considered. As a result of the shock impedance matching, no shock wave or rarefaction fan is reflected at the interface. Cases with shock impedance matching have previously been found to be problematic with GFM-based methods [46,47], where spurious shock waves or rarefaction fans are reflected at the interface. For a non-reflective shock-interface interaction, the fluid properties and the pressure ratio across the shock wave have to satisfy [46]

$$(\gamma_{L} - 1)\rho_{II,L} + (\gamma_{L} + 1)\rho_{II,L} \frac{p_{I}}{p_{II}} = (\gamma_{R} - 1)\rho_{II,R} + (\gamma_{R} + 1)\rho_{II,R} \frac{p_{I}}{p_{II}}.$$
(74)

Assuming the same setup as in Section 7.4.3 for the shock wave with  $M_{\rm s}=1.22$  in an air-helium flow,  $\gamma_{\rm L}=\gamma_{\rm air}$ ,  $\rho_{\rm II,L}=\rho_{\rm II,air}$  and the right phase has  $\gamma_{\rm R}=1.648$ , the specific isochoric heat capacity of the right phase follows from Eq. (74) as  $c_{\rm V,R}=512.41~\rm J~kg^{-1}~K^{-1}$ . Fig. 24 shows the profiles of density, pressure and Mach number, at  $t=2\times10^{-4}~\rm s$  after the shock-interface interaction. All variables are predicted accurately compared to the corresponding Riemann problem, without

![](_page_26_Figure_2.jpeg)

**Fig. 25.** Profiles of density, pressure, velocity, Mach number and acoustic impedance of the subsonic gas–gas shock tube at *t* = 8×10−<sup>4</sup> s, compared against the theoretical Riemann solution.

spurious reflections in the left phase or spurious oscillations at the interface. As observed for the previously discussed cases involving shock waves, the speed and position of the shock wave are computed with very high accuracy.

#### *7.5. Shock tubes*

Shock tubes are extensively used to test and scrutinise numerical frameworks for compressible flows, because an exact reference solution based on the associated Riemann problem is available and because they typically feature all three primary wave types (shock wave, rarefaction fan and contact discontinuity).

#### *7.5.1. Gas–gas shock tube*

A subsonic and a transonic shock tube with two-phase flow are simulated to assess the handling of discontinuous data and the prediction of shock waves, rarefaction fans and contact discontinuities by the presented algorithm. In both cases the discontinuity of the fluid states is initially located in the middle of the one-dimensional domain with a length of 1 m, which is represented with 400 equidistant mesh cells (*x* = 2*.*5 × 10−<sup>3</sup> s), with a time-step that corresponds to Co = *a*<sup>R</sup>*t/x* ≤ 0*.*27. The fluid interface initially coincides with the discontinuity of the fluid states. The initial conditions of the subsonic shock tube are

$$u_{L} = 0 \text{ m s}^{-1}, \quad p_{L} = 2.0 \times 10^{5} \text{ Pa}, \quad \rho_{L} = 3.57 \text{ kg m}^{-3}, \quad \gamma_{L} = 1.66,$$
  
 $u_{R} = 0 \text{ m s}^{-1}, \quad p_{R} = 1.0 \times 10^{5} \text{ Pa}, \quad \rho_{R} = 1.20 \text{ kg m}^{-3}, \quad \gamma_{R} = 1.40,$ 

and the initial conditions of the transonic shock tube are

$$u_{\rm L} = 200 \text{ m s}^{-1}, \quad p_{\rm L} = 5.0 \times 10^5 \text{ Pa}, \quad \rho_{\rm L} = 8.92 \text{ kg m}^{-3}, \quad \gamma_{\rm L} = 1.66,$$
  
 $u_{\rm R} = 0 \text{ m s}^{-1}, \quad p_{\rm R} = 1.0 \times 10^5 \text{ Pa}, \quad \rho_{\rm R} = 1.20 \text{ kg m}^{-3}, \quad \gamma_{\rm R} = 1.40.$ 

The profiles of density, pressure, velocity, Mach number and acoustic impedance are shown in Fig. 25 for the subsonic shock tube at *t* = 8 × 10−<sup>4</sup> s and in Fig. [26](#page-27-0) for the transonic shock tube at *t* = 6 × 10−<sup>4</sup> s, alongside the corresponding theoretical Riemann solution. All shown quantities are in excellent agreement with the theoretical Riemann solution in both cases, irrespective of the Mach number regime. The accurate prediction of the acoustic impedance at the fluid interface further supports the notion that ACID conserves the acoustic properties of the two-phase flow, while the accurate prediction of the Mach number suggests that the hydrodynamic–thermodynamic coupling of the proposed algorithm is correct. The correct position of the shock wave, the rarefaction fan and the contact discontinuity in both cases demonstrates that the conservative discretisation of the governing flow equations leads to physically sound numerical results satisfying the Rankine–Hugoniot relations and entropy condition. Furthermore, the computational result converges to the theoretical Riemann solution under mesh refinement, as demonstrated by the density profile of the subsonic shock tube in Figs. [27a](#page-27-0) and [27b](#page-27-0).

![](_page_27_Figure_2.jpeg)

**Fig. 26.** Profiles of density, pressure, velocity, Mach number and acoustic impedance of the transonic gas–gas shock tube at *t* = 6×10−<sup>4</sup> s, compared against the theoretical Riemann solution.

![](_page_27_Figure_4.jpeg)

**Fig. 27.** Density profile at (a) the rarefaction fan and (b) at the contact discontinuity, and (c) the *L*1-norm of the density error 1, of the subsonic gas–gas shock tube as a function of the mesh spacing. The theoretical Riemann solution is shown in (a) and (b) as a reference.

The contact discontinuity occurring in the solution of the shock tubes, which coincides with the fluid interface, is a linearly degenerate wave and, thus, represents the primary difficulty and main source of error with respect to the convergence of the applied finite-volume method under mesh refinement [\[11,104\]](#page-39-0). As shown rigorously by Banks et al. [\[104\]](#page-41-0), the convergence of the *L*1-norm of the solution error associated with a linearly degenerate wave for a *q*-th order advection scheme without compressive limiting is of order *q/(q* + 1*)*. Hence, for the applied second-order Minmod scheme the order of convergence of the *L*1-norm of density *ρ* at the contact discontinuity is 2*/*3 [\[104\]](#page-41-0). The *L*1-norm of the density profile, given by Eq. [\(73\)](#page-22-0), for the subsonic gas–gas shock tube is shown in Fig. 27c. The error of the density profile converges with order 0*.*657 (based on a regression fit of the data points, with *R*<sup>2</sup> = 0*.*9987), which is in very good agreement with the theoretical value for the convergence rate of 2*/*3. This suggests that the ACID method does not adversely affect the convergence of the numerical framework, considering that the contact discontinuity coincides with the material interface.

### *7.5.2. Gas–liquid shock tube*

A gas–liquid shock tube is simulated, with air as the left phase and water as the right phase. The one-dimensional domain of 2 m length is represented with an equidistant mesh of 800 cells (*x* = 2*.*5 × 10−<sup>3</sup> s), and is initialised with a uniform velocity *u*<sup>0</sup> = 0 and uniform temperature *T*<sup>0</sup> = 300 K. The pressure in the left and right phases is initialised as *p*<sup>L</sup> = 109 Pa and *p*<sup>R</sup> = 104 Pa, respectively, with the gas–liquid interface initially situated at *x,*<sup>0</sup> = 0*.*5 m. The heat capacity ratio *γ* and pressure constant of air and water are taken from Table [1.](#page-18-0) The profiles of density, pressure, velocity, Mach number and acoustic impedance at *t* = 8 × 10−<sup>4</sup> s are shown in Fig. [28,](#page-28-0) alongside the corresponding theoretical Riemann solution. Despite the large pressure ratio of *p*L*/p*<sup>R</sup> = 105, all quantities are in excellent agreement with the corresponding Riemann solution. A small discrepancy of the acoustic impedance *Z* can be observed at the gas–liquid interface, associated

![](_page_28_Figure_2.jpeg)

**Fig. 28.** Profiles of density, pressure, velocity, Mach number and acoustic impedance of the air–water shock tube at *t* = 8 × 10−<sup>4</sup> s, compared against the theoretical Riemann solution.

![](_page_28_Figure_4.jpeg)

**Fig. 29.** Schematic illustration of the computational setup of the shock-bubble interaction with Mach number *M*<sup>s</sup> = 1*.*22. The shock is initially located at *x* = 0*.*17 m and travels from left to right. The shaded area represents the bubble with a diameter of *d*<sup>0</sup> = 0*.*05 m, with the bubble centre initially located at *x* = 0*.*22 m.

with the nonlinear change of the speed of sound in the interface region (see Fig. [11b](#page-19-0)), which has, however, no lasting influence on the overall result of the shock tube.

#### *7.6. Two-dimensional shock-bubble interaction*

The interaction of a shock wave with *M*<sup>s</sup> = 1*.*22 in air with a circular bubble of helium and of R22 is simulated. These cases have previously been studied experimentally by Haas and Sturtevant [\[115\]](#page-41-0) and considered in a number of numerical studies to investigate the governing physical mechanisms [\[116,117\]](#page-41-0) and to test numerical algorithms, *e.g.* [\[5,31,](#page-39-0)[41\]](#page-40-0).

The computational setup is schematically illustrated in Fig. 29. Since the problem is symmetric about the centreline of the domain, as indicated in Fig. 29, only one half of the domain is simulated. For the discrete representation of the computational domain, a Cartesian mesh with mesh spacing *x* = *d*0*/*500 is considered. The shock is initially situated at *x* = 0*.*17 m and travels from left to right at speed *u*<sup>s</sup> . The shock wave separates the post-shock region I and the pre-shock region II, which are initialised with

$$u_{\rm I} = 125.65 \text{ m s}^{-1}, \quad p_{\rm I} = 1.59060 \times 10^5 \text{ Pa}, \quad T_{\rm I} = 402.67 \text{ K},$$
  
 $u_{\rm II} = 0 \text{ m s}^{-1}, \quad p_{\rm II} = 1.01325 \times 10^5 \text{ Pa}, \quad T_{\rm II} = 351.82 \text{ K}.$ 

The heat capacity ratio and the specific isochoric heat capacity of air are *γ*Air = <sup>1</sup>*.*4 and *cv,*Air = <sup>720</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1, respectively. The velocity and temperature at the domain-inlet are *u*in = *u*<sup>I</sup> and *T*in = *T*I, and pressure is extrapolated from the closest cell centre. At the domain-outlet a zero-gradient condition is applied to all variables. The applied time-step for both cases satisfies a Courant number of Co = *a*Air*,*II*t/x* = 0*.*38.

![](_page_29_Figure_2.jpeg)

**Fig. 30.** Contours of the density gradient *(*<sup>1</sup> + <sup>8</sup>*ψ)*|**∇***ρ*| (upper half) and the Mach number *<sup>M</sup>* (lower half) of the shock-interaction (*M*<sup>s</sup> = <sup>1</sup>*.*22) with <sup>a</sup> two-dimensional helium bubble on an Cartesian mesh with *x* = *d*0*/*500 at different time instances *τ* = *t a*He*,*II*/d*0. (For interpretation of the colours in the figure(s), the reader is referred to the web version of this article.)

#### *7.6.1. Helium bubble*

The properties of helium are identical to the experiments of Haas and Sturtevant [\[115\]](#page-41-0), with heat capacity ratio *γ*He = <sup>1</sup>*.*648 and specific isochoric heat capacity *cv,*He = <sup>2440</sup> J kg−<sup>1</sup> <sup>K</sup>−<sup>1</sup> (see also Section [7.4.3\)](#page-23-0). The contours of the density gradients and the Mach number are shown for different time instances in Fig. 30. The results are in excellent agreement with the experimental results of Haas and Sturtevant [\[115\]](#page-41-0), with regards to the interface shape, the shock position and the wave structure, *cf.* Fig. 7 in [\[115\]](#page-41-0) where *t* = {62 μs*,* 102 μs*,* 245 μs*,* 427 μs} correspond to *τ* = *t a*He*,*II*/d*<sup>0</sup> = {0*.*92*,* 1*.*77*,* 4*.*50*,* 7*.*70} (note that the shock wave travelled from right to left in the experiments). At *τ* = 0*.*92 the shock wave reaches the far-side of the bubble interface, as in the experiments, and the computed position (and shape) of the reflected rarefaction fan, the Mach stem and the triple point are in very close agreement with the experimental observations. A good agreement between computations and experiments is also observed at *τ* = 1*.*77 with respect to the primary shock wave, the reflected waves immediately following the primary shock wave and the Mach stem. Also note that the density gradient and, consequently, the interface representation stay sharp throughout the simulation, without artificial smearing of the interface. Furthermore, the presented results are also found to be in very good agreement with previous numerical studies, notably [\[31,41,116](#page-40-0)[,118\]](#page-41-0). The growth and magnitude of the instabilities forming at the interface, *e.g.* on the downstream side of the interface in Fig. 30f, exhibit significant differences between various numerical studies (*cf.* [\[5,118\]](#page-39-0)) and the "correct" development of the instabilities, although clearly present, cannot be precisely deduced from the experimental images of Haas and Sturtevant [\[115\]](#page-41-0). Johnsen and Colonius [\[118\]](#page-41-0) showed that spurious pressure oscillations at the interface increase these instabilities artificially, leading to more pronounced interface instabilities predicted by methods that do not completely eliminate spurious pressure oscillations.

#### *7.6.2. R22 bubble*

The properties of the R22 bubble are taken from the experimental study of Haas and Sturtevant [\[115\]](#page-41-0), with heat capacity ratio *γ*R22 = <sup>1</sup>*.*249 and specific isochoric heat capacity *cv,*R22 = <sup>365</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1. The contours of the density gradients and the Mach number are shown for different time instances in Fig. [31.](#page-30-0) Similar to the results presented for the shock interaction with a helium bubble in the previous section, the simulation results obtained with the proposed pressure-based algorithm are in very good agreement with the experimental results reported by Haas and Sturtevant [\[115\]](#page-41-0), *cf.* Fig. 11 in [\[115\]](#page-41-0) where *t* = {55 μs*,* 135 μs*,* 187 μs*,* 247 μs} correspond to *τ* = *t a*R22*,*II*/d*<sup>0</sup> = {0*.*20*,* 0*.*49*,* 0*.*68*,* 0*.*89} (note that the shock wave travelled from right to left in the experiments). The primary shock wave, the reflected shock wave in air and the transmitted shock wave in the R22 bubble are computed accurately at *τ* ∈ {0*.*20*,* 0*.*49} compared to the experiments. After the primary shock

![](_page_30_Figure_2.jpeg)

**Fig. 31.** Contours of the density gradient *(*<sup>1</sup> − <sup>0</sup>*.*75*ψ)*|**∇***ρ*| (upper half) and the Mach number *<sup>M</sup>* (lower half) of the shock-interaction (*M*<sup>s</sup> = <sup>1</sup>*.*22) with <sup>a</sup> two-dimensional R22 bubble on an Cartesian mesh with *x* = *d*0*/*500 at different time instances *τ* = *t a*R22*,*II*/d*0.

![](_page_30_Figure_4.jpeg)

**Fig. 32.** Contours of the density gradient *(*1−0*.*75*ψ)*|**∇***ρ*| on the triangular mesh with *<sup>x</sup>* = *<sup>d</sup>*0*/*293 (upper half) and the Cartesian mesh with *<sup>x</sup>* = *<sup>d</sup>*0*/*<sup>300</sup> (lower half) of the shock-interaction (*M*<sup>s</sup> = 1*.*22) with a two-dimensional R22 bubble at different time instances *τ* = *t a*R22*,*II*/d*0.

wave has passed the bubble, see *τ* = 0*.*68 in Fig. 31c and *τ* = 0*.*89 in Fig. 31d, it collapses back on itself, leading to a number of reflected waves captured by the simulations, which are also observed in the experiments of Haas and Sturtevant [\[115\]](#page-41-0). The presented result also compare well to previous simulation results [\[110,117,119\]](#page-41-0).

As mentioned in Section [3,](#page-4-0) the proposed algorithm is applicable on unstructured meshes. Fig. 32 shows the R22 bubble during and after the impact of the shock wave on a triangular mesh with an equivalent mesh spacing (*i.e.* the mesh spacing of an equidistant Cartesian mesh with the same number of cells) of *x* = *d*0*/*293, alongside the results obtained on an equidistant Cartesian mesh with *x* = *d*0*/*300. The qualitative features pertaining to the shock wave and its reflections, as well as the interface shape, are in very good agreement. In general, more flow features are resolved in the bulk phases on the Cartesian mesh, while instabilities developing along the interface are more pronounced on the triangular mesh. Despite these small qualitative differences, the density, pressure and velocity profiles obtained on both meshes on a line along the *x*-axis at *y* = 0*.*002 m, shown in Fig. [33](#page-31-0) for *τ* = 0*.*89, are in excellent agreement.

![](_page_31_Figure_2.jpeg)

**Fig. 33.** Profiles of density, pressure and velocity along the *x*-axis at *y* = 0*.*002 m of the shock-interaction (*M*<sup>s</sup> = 1*.*22) with a two-dimensional R22 bubble on the Cartesian mesh (*x* = *d*0*/*300) and the triangular mesh (*x* = *d*0*/*293) at dimensionless time *τ* = *t a*R22*,*II*/d*<sup>0</sup> = 0*.*89.

![](_page_31_Figure_4.jpeg)

**Fig. 34.** Contours of the density gradient *(*<sup>1</sup> − <sup>0</sup>*.*75*ψ)*|**∇***ρ*| (upper half) and the velocity magnitude |*u*| (lower half) of the shock-interaction (*M*<sup>s</sup> = <sup>1</sup>*.*22) with a two-dimensional R22 bubble on the considered Cartesian and triangular meshes at *τ* = *t a*R22*,*II*/d*<sup>0</sup> = 0*.*89.

Comparing the contours of the density gradient and the velocity magnitude obtained on Cartesian meshes with *x* ∈ {*d*0*/*200*,d*0*/*300*,d*0*/*500} and triangular meshes with *x* ∈ {*d*0*/*207*,d*0*/*293}, shown in Fig. 34, an increasing resolution of small flow features, particularly near the interface, is observed with increasing mesh resolution. This is to be expected, because viscous stresses, heat conduction and surface tension are neglected in these simulations and, thus, small flow features are not regulated. Nevertheless, the streamwise position *x*<sup>b</sup> and velocity *u*<sup>b</sup> of the centre of mass of the bubble, as well as the volume of the mass *V*b, shown in Figs. [35](#page-32-0) and [36,](#page-32-0) are in excellent agreement throughout the simulation on all considered meshes.

For the same Cartesian and triangular meshes, the conservation of mass in the entire computational domain (accounting for the additional mass that enters through the domain inlet), shown in Fig. [37,](#page-32-0) is conserved accurately, with an error *<* 0*.*01% on the Cartesian meshes and *<* 0*.*03% on the triangular meshes. Similarly, the conservation of mass of the bubble is very good, see Fig. [38,](#page-32-0) being within 0*.*1% on the Cartesian meshes and within 1*.*0% on the triangular meshes.

#### *7.6.3. Influence of the advection differencing scheme*

The applied numerical method has a direct influence on the development and evolution of the compression and expansion waves, as well as instabilities at the interface, as briefly mentioned in Section [7.6.1.](#page-29-0) For instance, Johnsen and

![](_page_32_Figure_2.jpeg)

**Fig. 35.** Position  $x_b$  and velocity  $u_b$  of the centre of mass, and relative volume  $V_b/V_b^{(0)}$ , of the bubble of the shock-interaction ( $M_s = 1.22$ ) with a two-dimensional R22 bubble as a function of dimensionless time  $\tau$  on the considered Cartesian meshes.

![](_page_32_Figure_4.jpeg)

**Fig. 36.** Position  $x_b$  and velocity  $u_b$  of the centre of mass, and relative volume  $V_b/V_b^{(0)}$ , of the bubble of the shock-interaction ( $M_s = 1.22$ ) with a two-dimensional R22 bubble as a function of dimensionless time  $\tau$  on the considered triangular meshes. The result obtained on the Cartesian mesh with  $\Delta x = d_0/500$  is shown as a reference.

![](_page_32_Figure_6.jpeg)

Fig. 37. Mass error  $\varepsilon_{\rho}$ , given by Eq. (67), in the entire computational domain of the shock-interaction ( $M_{\rm S}=1.22$ ) with a two-dimensional R22 bubble as a function of dimensionless time  $\tau$  for all considered Cartesian and triangular meshes.

![](_page_32_Figure_8.jpeg)

**Fig. 38.** Mass error  $\varepsilon_{\rho}$ , given by Eq. (67), of the bubble of the shock-interaction ( $M_{\rm S}=1.22$ ) with a two-dimensional R22 bubble as a function of dimensionless time  $\tau$  for all considered Cartesian and triangular meshes.

![](_page_33_Figure_2.jpeg)

**Fig. 39.** Contours of the density gradient  $(1 + 8\psi)|\nabla\rho|$  (upper half) and the velocity magnitude |u| (lower half) of the shock-interaction ( $M_s = 1.22$ ) with a two-dimensional helium bubble on a Cartesian mesh with  $\Delta x = d_0/500$  at  $\tau = t \, a_{\text{He,II}}/d_0 = 4.50$ , using the Minmod scheme and the Superbee scheme.

![](_page_33_Figure_4.jpeg)

**Fig. 40.** Contours of the density gradient  $(1 + 8\psi)|\nabla\rho|$  (upper half) and the velocity magnitude |u| (lower half) of the shock-interaction ( $M_s = 1.22$ ) with a two-dimensional helium bubble on a Cartesian mesh with  $\Delta x = d_0/500$  at  $\tau = t \, a_{\text{He,II}}/d_0 = 7.70$ , using the Minmod scheme and the Superbee scheme.

Colonius [118] observed a substantial influence of spurious pressure oscillations at the interface on the onset and evolution of interfacial instabilities. Although spurious pressure oscillations are absent in the results presented above, a clear influence of the applied differencing schemes can be observed for the shock interaction with a helium bubble and a R22 bubble.

Figs. 39 and 40 show the contours of the density gradient and the velocity magnitude for the helium bubble at  $\tau = 4.50$  and  $\tau = 7.70$ , respectively, obtained using the Minmod scheme and the Superbee scheme [84], on a Cartesian mesh with  $\Delta x = d_0/500$ . Because of the compressive limiting applied with the Superbee scheme, the compression and expansion waves are resolved sharper. Moreover, interface instabilities form more rapidly with the more compressive Superbee scheme.

A similar behaviour can be observed for the R22 bubble at  $\tau=0.89$  using the first-order upwind scheme, the Minmod scheme and the Superbee scheme, for which the resulting density gradient and velocity magnitude are shown in Fig. 41. Apart from the expected sharper resolution of compression and expansion waves with increasing compression of the applied differencing scheme, the Mach number at the interface differs substantially; while the flow is entirely subsonic with the upwind scheme, the flow inside the bubble becomes transonic with the Minmod scheme and supersonic with the Superbee scheme. Although the temporal evolution of the volume of the R22 bubble is virtually the same irrespective of the applied differencing scheme, as seen in Fig. 42a, the conservation of the bubble mass is strongly affected by the applied differencing scheme, as observed in Fig. 42b. Based on these results, the Minmod scheme appears to provide the most feasible result, although this issue requires a further, more comprehensive investigation to draw firm conclusions about the most appropriate differencing scheme.

#### 7.7. Three-dimensional shock-bubble interaction

The interaction of a shock wave with  $M_s = 1.68$  in air with a three-dimensional helium bubble is simulated, as previously studied in [117]. Since the problem is axisymmetric with respect to the x-axis, only one quarter of the domain in the y-z plane is simulated. The domain has the dimensions  $6r_0 \times 4r_0 \times 4r_0$ , where  $r_0 = 0.0254$  m is the initial radius of the bubble, and is represented by a Cartesian mesh with  $\Delta x = 5 \times 10^{-4}$  m (50.8 cells per  $r_0$ ). Following Niederhaus et al. [117], the heat capacity ratio and specific isobaric heat capacity of air and helium are  $\gamma_{Air} = 1.399$  and  $c_{p,Air} = 1006.36$  J kg<sup>-1</sup> K<sup>-1</sup>, and

![](_page_34_Figure_2.jpeg)

**Fig. 41.** Contours of the density gradient  $(1-0.75\psi)|\nabla\rho|$  (upper half) and the velocity magnitude  $|\pmb{u}|$  (lower half) of the shock-interaction  $(M_s=1.22)$  with a two-dimensional R22 bubble on a Cartesian mesh with  $\Delta x = d_0/500$  at  $\tau = t\,a_{R22,II}/d_0 = 0.89$ , using the first-order upwind scheme, the Minmod scheme and the Superbee scheme.

![](_page_34_Figure_4.jpeg)

**Fig. 42.** Relative volume  $V_b/V_b^{(0)}$  and mass error  $\varepsilon_\rho$ , given by Eq. (67), of the bubble of the shock-interaction ( $M_s=1.22$ ) with a two-dimensional R22 bubble on a Cartesian mesh with  $\Delta x = d_0/500$  as a function of dimensionless time  $\tau$ , using the first-order upwind scheme, the Minmod scheme and the Superbee scheme.

![](_page_34_Figure_6.jpeg)

**Fig. 43.** Contours of the density  $\rho$  (upper half) and the Mach number M (lower half) of the shock-interaction ( $M_s = 1.68$ ) with a three-dimensional helium bubble on a Cartesian mesh at different time instances  $\tau_s = t \, u_{s, \text{He}}/r_0$ . Note that the same range and colour scale for density  $\rho$  are shown as previously used by Niederhaus et al. [117].

 $\gamma_{\text{He}} = 1.667$  and  $c_{p,\text{He}} = 5190.80$  J kg<sup>-1</sup> K<sup>-1</sup>, respectively. The centre of the bubble is initially located at  $x_{b,0} = 0.075$  m and the shock wave (travelling in positive x-direction) is initialised at  $x_{s,0} = 0.02$  m. The shock wave separates the post-shock region I and the pre-shock region II, which are initialised with

$$u_{\rm I} = 310.19 \text{ m s}^{-1}, \quad p_{\rm I} = 3.166 \times 10^5 \text{ Pa}, \quad T_{\rm I} = 422.81 \text{ K},$$
  
 $u_{\rm II} = 0 \text{ m s}^{-1}, \qquad p_{\rm II} = 1.013 \times 10^5 \text{ Pa}, \quad T_{\rm II} = 293.00 \text{ K}.$ 

The contours of density and Mach number are shown in Fig. 43 for  $\tau_s = t \, u_{s,He}/r_0 \in \{1.4, 2.6, 5.2\}$ , where  $u_{s,He}$  is the speed of the shock wave in helium, which are the same dimensionless time instances  $\tau_s$  as considered by Niederhaus et al. [117]. The density field, position of the Mach stem, and the position and shape of the primary shock wave and the reflected rarefaction fan computed by the proposed algorithm agree very well with the results reported in [117] for all considered

![](_page_35_Figure_2.jpeg)

**Fig. 44.** Contours of the density  $\rho$  in the symmetry plane and isocontours of the density for (a)  $\rho = 2.5$  kg m<sup>-3</sup> and (b)  $\rho = 1.5$  kg m<sup>-3</sup> of the shock-interaction ( $M_{\rm S} = 1.68$ ) with a three-dimensional helium bubble on a Cartesian mesh at dimensionless time  $\tau_{\rm S} = t\,u_{\rm S,He}/r_0 = 2.2$ . Note that not the entire computational domain is shown.

![](_page_35_Figure_4.jpeg)

**Fig. 45.** Schematic illustration of the computational setup of the shock-drop interaction with Mach number  $M_s = 1.47$  in an air–water flow. The shock wave is initially located at  $x_{s,0} = 7.39 \times 10^{-3}$  m and travels from left to right. The shaded area represents the water drop with a diameter of  $d_0 = 4.8 \times 10^{-3}$  m, with the drop centre initially located at  $x_{d,0} = 10.8 \times 10^{-3}$  m.

time instances. The density field at  $\tau=2.2$  is shown in Fig. 44a together with the isocontour  $\rho=2.5$  kg m<sup>-3</sup>, which separates the pre-shock and the post-shock regions next to the bubble, and which follows the Mach stem and the rarefaction fan in the immediate vicinity and behind the bubble. The same result is shown in Fig. 44b but with the isocontour  $\rho=1.5$  kg m<sup>-3</sup> that represents the shock front. Both isocontours in Fig. 44 are clearly axisymmetric, demonstrating that the proposed algorithm captures the three-dimensionality of the solution reliably.

#### 7.8. Shock-drop interaction

The interaction of a shock wave with a liquid drop is simulated, with different Mach numbers of the shock wave. Contrary to the cases presented in Sections 7.6 and 7.7 where two perfect-gases interact, in the cases presented below a liquid drop described by a stiffened gas  $(\Pi \gg p)$  is surrounded by a perfect gas  $(\Pi = 0)$ .

#### 7.8.1. Shock-drop interaction with $M_s = 1.47$

The interaction of a shock wave with Mach number  $M_s = 1.47$  in air with a circular water drop, as previously studied in [30,120–122], is simulated. The flow is assumed to be symmetric and the computational setup is schematically illustrated in Fig. 45. The post-shock region I and the pre-shock region II are initialised with

$$u_{\rm I} = 225.89 \,\mathrm{m \, s^{-1}}, \quad p_{\rm I} = 2.386 \times 10^5 \,\mathrm{Pa}, \quad T_{\rm I} = 381.20 \,\mathrm{K},$$
  
 $u_{\rm II} = 0 \,\mathrm{m \, s^{-1}}, \qquad p_{\rm II} = 1.013 \times 10^5 \,\mathrm{Pa}, \quad T_{\rm II} = 293.15 \,\mathrm{K}.$ 

![](_page_36_Figure_2.jpeg)

**Fig. 46.** Contours of the density gradient *(*<sup>1</sup> − <sup>0</sup>*.*8*ψ)*|**∇***ρ*| (upper half) and the velocity magnitude |*u*| (lower half) of the shock-interaction (*M*<sup>s</sup> = <sup>1</sup>*.*47) with a two-dimensional water drop on a Cartesian mesh with *x* = *d*0*/*200.

![](_page_36_Figure_4.jpeg)

**Fig. 47.** Mass error *ερ* , given by Eq. [\(67\)](#page-17-0), of the drop and in the entire domain of the shock-interaction (*M*<sup>s</sup> = 1*.*47) with a two-dimensional water drop as a function of time *t* on a Cartesian mesh with *x* = *d*0*/*200, where *t* = 0 corresponds to the initial interaction of the shock wave with the drop.

Following the properties applied by Meng and Colonius [\[121\]](#page-41-0), air has the properties *γ*Air = 1*.*4, Air = 0 and *R*Air = <sup>287</sup>*.*<sup>08</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1, while water is described by *γ*Water = <sup>6</sup>*.*12, Water = <sup>3</sup>*.*<sup>43</sup> × 108 Pa and *<sup>R</sup>*Water = <sup>7170</sup>*.*<sup>23</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1. The computational domain is represented by an equidistant Cartesian mesh with *x* = *d*0*/*200 and the applied time-step corresponds to a Courant number of Co = *a*Water*,*II*t/x* = 0*.*15.

Fig. 46 shows the density gradient and the velocity magnitude of the flow at different times *t* ∈ {7*.*74 μs*,* 16*.*18 μs*,* 23*.*00 μs}, with *t* = 7*.*74 μs corresponding to the dimensionless time *t*<sup>∗</sup> = 0*.*017 in [\[121\]](#page-41-0). The results are qualitatively in very good overall agreement with previously reported numerical results for this particular shock-drop interaction [\[121,123\]](#page-41-0). The error in mass conservation *ερ* , shown in Fig. 47, is, however, about one order of magnitude larger than for the shock-bubble cases presented in Section [7.6.2.](#page-29-0) This increase in mass error extends to both the drop as well as the entire computational domain. Since the proposed framework is nominally mass-conservative, as demonstrated in Section [7.6.2,](#page-29-0) this error is likely the result of the finite interface thickness introduced by the applied VOF method and the associated misrepresentation of the fluid properties in this interface region, as further studied below.

#### *7.8.2. Shock-drop interaction with M*<sup>s</sup> = 6

The interaction of a shock wave with Mach number *M*<sup>s</sup> = 6 in gas with a circular liquid drop, similar to previous work of Shukla et al. [\[109\]](#page-41-0), is simulated. The computational setup is schematically illustrated in Fig. [48.](#page-37-0) The post-shock region I and the pre-shock region II are initialised with

$$u_{\rm I} = 5.789 \text{ m s}^{-1}, \quad p_{\rm I} = 42.388 \text{ Pa}, \quad T_{\rm I} = 2.794 \times 10^{-2} \text{ K},$$
  
 $u_{\rm II} = 0 \text{ m s}^{-1}, \quad p_{\rm II} = 1.013 \text{ Pa}, \quad T_{\rm II} = 3.518 \times 10^{-3} \text{ K}.$ 

![](_page_37_Figure_2.jpeg)

**Fig. 48.** Schematic illustration of the computational setup of the shock-drop interaction with Mach number *M*<sup>s</sup> = 6 in a gas–liquid flow. The shock wave is initially located at *x*s*,*<sup>0</sup> = 1 m and travels from left to right. The shaded area represents the liquid drop with a diameter of *d*<sup>0</sup> = 1*.*124 m, with the drop centre initially located at *x*d*,*<sup>0</sup> = 2 m.

![](_page_37_Figure_4.jpeg)

**Fig. 49.** Contours of the density gradient *(*<sup>1</sup> − <sup>0</sup>*.*8*ψ)*|**∇***ρ*| (upper half) and the Mach number *<sup>M</sup>* (lower half) of the shock-interaction (*M*<sup>s</sup> = 6) with <sup>a</sup> two-dimensional liquid drop on an equidistant Cartesian mesh with *x* = *d*0*/*562.

Following the properties applied by Shukla et al. [\[109\]](#page-41-0), the gas has the properties *γ*gas = 1*.*4, gas = 0 and *R*gas = <sup>288</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1, while the liquid is described by *γ*liquid = <sup>4</sup>*.*4, liquid = <sup>6000</sup> Pa and *<sup>R</sup>*liquid = <sup>7504</sup> J kg−<sup>1</sup> <sup>K</sup><sup>−</sup>1. The applied time-step on all considered meshes corresponds to a Courant number of Co = *a*liquid*,*II*t/x* = 0*.*23.

The contours of the density gradient and the Mach number at time *t* ∈ {0*.*08 s*,* 0*.*22 s*,* 0*.*34 s*,* 0*.*59 s} after the initial interaction of the shock wave and the drop are shown in Fig. 49 on an equidistant Cartesian mesh with *x* = *d*0*/*562. On this high resolution mesh, which has the same resolution as previously applied by Shukla et al. [\[109\]](#page-41-0), the individual flow features are well resolved and clearly visible. The time instances *t* ∈ {0*.*08 s*,* 0*.*22 s*,* 0*.*59 s} in Fig. 49 correspond to *t* ∈ {0*.*25*,* 0*.*50*,* 0*.*75}, respectively, of Fig. 11 in [\[109\]](#page-41-0). The results are overall in very good agreement with the results reported by Shukla et al. [\[109\]](#page-41-0). Comparing the results on different meshes at *t* = 0*.*34 s, see Figs. 49c and [50,](#page-38-0) small but clearly visible differences in the shape and structure of the compression and expansion waves are observed. This indicates mesh-dependent differences in the interaction of the waves with the gas–liquid interface, since the mesh-dependence of the speed and position of these types of waves is in general negligible, as demonstrated in Sections [7.4](#page-21-0) and [7.5.](#page-26-0) Because of the large differences in fluid properties, notably the density and heat capacity, in gas–liquid flows, even small differences of the interface position, likely caused by small errors in the interface advection (see Section [4.2\)](#page-8-0), or mesh-dependent thickness of the interface, can have a significant effect on the wave-interface interactions. These differences also manifest in mass-conservation errors with respect to the computational domain and the drop, see Fig. [51.](#page-38-0) Yet, as expected from errors associated with the applied interface capturing scheme, the mass-conservation error reduces with mesh refinement. Even though qualitative differences are visible in the contour plots, and the mass-conservation errors are not negligible for the results obtained on different meshes, the position and velocity of the centre of mass of the drop and the volume of the drop, shown in Fig. [52,](#page-38-0) are in good agreement on the different meshes, and converge with mesh refinement.

![](_page_38_Figure_2.jpeg)

**Fig. 50.** Contours of the density gradient  $(1 - 0.8\psi)|\nabla\rho|$  (upper half) and the Mach number M (lower half) of the shock-interaction ( $M_s = 6$ ) with a two-dimensional liquid drop at t = 0.34 s, on equidistant Cartesian meshes with (a)  $\Delta x = d_0/140.5$  and (b)  $\Delta x = d_0/281$ .

![](_page_38_Figure_4.jpeg)

**Fig. 51.** Mass error  $\varepsilon_{\rho}$ , given by Eq. (67), in (a) the entire computational domain and (b) of the drop of the shock-interaction ( $M_s = 6$ ) with a two-dimensional liquid drop as a function of time t on the considered Cartesian meshes, where t = 0 corresponds to the initial interaction of the shock wave with the drop.

![](_page_38_Figure_6.jpeg)

**Fig. 52.** Position  $x_b$  and velocity  $u_b$  of the centre of mass, and relative volume  $V_b/V_b^{(0)}$ , of the drop of the shock-interaction ( $M_s = 6$ ) with a two-dimensional liquid drop as a function of time t on the considered Cartesian meshes, where t = 0 corresponds to the initial interaction of the shock wave with the drop.

#### 8. Conclusions

A pressure-based algorithm for compressible interfacial flows has been presented, based on a fully-coupled finite-volume framework with a collocated variable arrangement that is applicable to unstructured meshes. The governing flow equations, which are discretely conservative in mass, momentum and energy, are solved for velocity, pressure and specific total enthalpy. Contrary to all previously published algorithms for compressible interfacial flows, density is not a solution variable but is computed based on pressure, temperature and the local fluid properties via an equation of state. To this end, the stiffened-gas model has been considered in this study to describe gases, liquids and solids. The proposed algorithm includes an acoustically-conservative interface discretisation (ACID), which performs a local manipulation to the discrete values of density and total enthalpy based on the interface position, and which does not require the solution of a Riemann problem or a priori knowledge of the wave structure to evaluate fluxes.

Results for a variety of representative compressible gas-gas and gas-liquid interfacial flows have been presented, including the propagation of acoustic waves, shock tubes and the interaction of shock waves with one-, twoand three-dimensional interfaces. The transmission and reflection of acoustic waves at the interface has been shown to be captured accurately by the presented algorithm, including cases with acoustic impedance matching, a capability not previously reported in the literature. Similarly, the interaction of shock waves with the interface in one-, twoand three-dimensional

flows, as well as gas–gas and gas–liquid shock tube problems, have been favourably compared against exact Riemann solutions and previously reported experimental and computational results, especially the temporal evolution of the shock position, of reflected shock waves and rarefaction fans, of the wave structure and the interface shape. In all considered cases, the presented algorithm reliably retains the acoustic properties of the fluids, while the accurate prediction of the Mach number suggests a correct hydrodynamic–thermodynamic coupling. Notably, the speed, position and strength of shock waves are predicted accurately and the Rankine–Hugoniot relations are satisfied even for strong shock waves, in the bulk phases and in the interface region, demonstrating that the proposed algorithm converges to the correct weak solution of the governing conservation laws [8–10]. This enforces the notion that, irrespective of small conservation errors associated with the applied interface capturing scheme, the governing equations remain discretely conservative and implies that the proposed algorithm implicitly satisfies the second law of thermodynamics.

Despite the demonstrated success of the presented algorithm, several open questions remain. The EOS of the perfect-gas model or the stiffened-gas model is used in this study to evaluate the density based on pressure and temperature. However, most EOS are formulated in terms of internal energy rather than density. While this is not a problem for the considered gas models, and some real-gas models can readily be formulated in terms of density, such as the Peng–Robinson EOS [\[124\]](#page-42-0), more complex EOS may not be easily inverted to a density formulation. Furthermore, the proposed interface discretisation method ACID implies a mechanical and thermal equilibrium. Mechanical equilibrium is a reasonable assumption for interfacial flows, since mechanical relaxation is associated with acoustic effects [21,26] and, thus, occurs at a time-scale comparable with the typically applied computational time-steps. The implicit assumption of thermal equilibrium, however, may present a limiting factor when more complex EOS are considered [4] and/or thermal relaxation is governed by diffusion [\[59\]](#page-40-0), for instance if reactive flows are to be simulated. Neither of these questions have been addressed in the context of pressure-based methods and warrant further study.

In summary, the proposed pressure-based algorithm has been shown to be a promising alternative to traditional densitybased algorithms for the simulation of compressible interfacial flows. The pressure-based formulation of the governing equations facilitates the definition of consistent mixture rules at the interface, including the Rankine–Hugoniot relations, and applies naturally to flows in all Mach number regimes.

#### **Acknowledgements**

The authors gratefully acknowledge financial support from the Engineering and Physical Sciences Research Council (EP-SRC) through grant EP/M021556/1 and from Shell Corporation. The contribution of all four anonymous reviewers through their detailed and critical comments is greatly appreciated. Data supporting this publication can be obtained from [https://](https://doi.org/10.5281/zenodo.1218933) [doi.org/10.5281/zenodo.1218933](https://doi.org/10.5281/zenodo.1218933) under a Creative Commons Attribution license.

#### **References**

- [1] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436F72616C696332303134s1) flows, J. Comput. Phys. 274 (2014) 95–121.
- [2] R. Abgrall, How to prevent pressure oscillations in [multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib41626772616C6C31393936s1) flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) [150–160.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib41626772616C6C31393936s1)
- [3] K.-M. Shyue, An efficient shock-capturing algorithm for compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536879756531393938s1) problems, J. Comput. Phys. 142 (1998) 208–242.
- [4] G. Allaire, S. Clerc, S. Kokh, A [five-equation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416C6C6169726532303032s1) model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577–616.
- [5] A. Marquina, P. Mulet, A flux-split algorithm applied to conservative models for [multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D61727175696E6132303033s1) compressible flows, J. Comput. Phys. 185 (2003) [120–138.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D61727175696E6132303033s1)
- [6] G. Perigaud, R. Saurel, A [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib506572696761756432303035s1) flow model with capillary effects, J. Comput. Phys. 209 (2005) 139–178.
- [7] X. Hu, B. Khoo, N. Adams, F. Huang, A conservative interface method for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib487532303036s1) flows, J. Comput. Phys. 219 (2006) 553–578.
- [8] T.Y. Hou, P.G.L. Floch, Why [nonconservative](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib486F7531393934s1) schemes converge to wrong solutions: error analysis, Math. Comput. 62 (1994) 497–530.
- [9] C.B. Laney, [Computational](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C616E657931393938s1) Gasdynamics, Cambridge University Press, Cambridge, New York, NY, 1998.
- [10] D. van der Heul, C. Vuik, P. Wesseling, A conservative [pressure-correction](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib76616E6465724865756C32303033s1) method for flow at all speeds, Comput. Fluids 32 (2003) 1113–1132.
- [11] A. Harten, High resolution schemes for hyperbolic [conservation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib48617274656E31393833s1) laws, J. Comput. Phys. 49 (1983) 357–393.
- [12] P. Sweby, High resolution schemes using flux limiters for hyperbolic [conservation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib537765627931393834s1) laws, SIAM J. Numer. Anal. 21 (1984) 995–1011.
- [13] E. Caramana, M. Shashkov, P. Whalen, Formulations of artificial viscosity for [multi-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436172616D616E6131393938s1) shock wave computations, J. Comput. Phys. 144 (1998) [70–97.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436172616D616E6131393938s1)
- [14] S. Karni, [Multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B61726E6931393934s1) flow calculations by a consistent primitive algorithms, J. Comput. Phys. 112 (1994) 31–43.
- [15] R. Abgrall, S. Karni, [Computations](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib41626772616C6C32303031s1) of compressible multifluids, J. Comput. Phys. 169 (2001) 594–623.
- [16] J. Brackbill, D. Kothe, C. Zemach, [Continuum](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib427261636B62696C6C31393932s1) method for modeling surface tension, J. Comput. Phys. 100 (1992) 335–354.
- [17] M. Rudman, A [volume-tracking](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5275646D616E31393938s1) method for incompressible multifluid flows with large density variations, Int. J. Numer. Methods Fluids 28 (1998) [357–378.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5275646D616E31393938s1)
- [18] M. Raessi, H. Pitsch, Consistent mass and momentum transport for simulating [incompressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5261657373693230313261s1) interfacial flows with large density ratios using the level set method, [Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5261657373693230313261s1) Fluids 63 (2012) 70–81.
- [19] M. Baer, J. Nunziato, A two-phase mixture theory for the [deflagration-to-detonation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4261657231393836s1) transition (ddt) in reactive granular materials, Int. J. Multiph. Flow 12 (1986) [861–889.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4261657231393836s1)
- [20] P. Embid, M. Baer, [Mathematical](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib456D62696431393932s1) analysis of a two-phase continuum mixture theory, Contin. Mech. Thermodyn. 4 (1992) 279–312.
- [21] R. Saurel, R. Abgrall, A multiphase Godunov method for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53617572656C31393939s1) multifluid and multiphase flows, J. Comput. Phys. 150 (1999) 425–467.
- [22] A. Ambroso, C. Chalons, P.-A. Raviart, A Godunov-type method for the [seven-equation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416D62726F736F32303132s1) model of compressible two-phase flow, Comput. Fluids 54 [\(2012\)](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416D62726F736F32303132s1) 67–91.
- [23] R. Abgrall, R. Saurel, Discrete equations for physical and numerical [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib41626772616C6C32303033s1) multiphase mixtures, J. Comput. Phys. 186 (2003) 361–396.
- [24] N. Andrianov, G. Warnecke, The Riemann problem for the [Baer–Nunziato](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416E647269616E6F7632303034s1) two-phase flow model, J. Comput. Phys. 195 (2004) 434–464.

- [25] D. Schwendeman, C. Wahle, A. Kapila, The Riemann problem and a [high-resolution](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53636877656E64656D616E32303036s1) Godunov method for a model of compressible two-phase flow, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53636877656E64656D616E32303036s1) Phys. 212 (2006) 490–526.
- [26] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of [deflagration-to-detonation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6170696C6132303031s1) transition in granular materials: reduced equations, Phys. Fluids 13 (2001) [3002–3024.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6170696C6132303031s1)
- [27] A. Murrone, H. Guillard, A five equation reduced model for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D7572726F6E6532303035s1) two phase flow problems, J. Comput. Phys. 202 (2005) 664–698.
- [28] G. Allaire, S. Clerc, S. Kokh, A [five-equation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416C6C6169726532303030s1) model for the numerical simulation of interfaces in two-phase flows, C. R. Acad. Sci., Sér. 1 Math. 331 (2000) [1017–1022.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416C6C6169726532303030s1)
- [29] J. Massoni, R. Saurel, B. Nkonga, R. Abgrall, Proposition de méthodes et modèles eulériens pour les problèmes à interfaces entre fluides [compressibles](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6173736F6E6932303032s1) en présence de transfert de chaleur, Int. J. Heat Mass Transf. 45 (2002) [1287–1307.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6173736F6E6932303032s1)
- [30] D.P. Garrick, M. Owkes, J.D. Regele, A [finite-volume](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4761727269636B32303137s1) HLLC-based scheme for compressible interfacial flows with surface tension, J. Comput. Phys. 339 [\(2017\)](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4761727269636B32303137s1) 46–67.
- [31] M.L. Wong, S.K. Lele, High-order localized dissipation weighted compact nonlinear scheme for shockand [interface-capturing](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib576F6E6732303137s1) in compressible flows, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib576F6E6732303137s1) Phys. 339 (2017) 179–209.
- [32] E.F. Toro, Riemann Solvers and Numerical Fluid Dynamics: A Practical [Introduction,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546F726F32303039s1) third edition, Springer, 2009.
- [33] J.P. Cocchi, R. Saurel, J.C. Loraud, Treatment of interface problems with [Godunov-type](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436F6363686931393936s1) schemes, Shock Waves 5 (1996) 347–357.
- [34] E.F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the [HLL–Riemann](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546F726F31393934s1) solver, Shock Waves 4 (1994) 25–34.
- [35] K.-M. Shyue, A [volume-fraction](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536879756532303036s1) based algorithm for hybrid barotropic and non-barotropic two-fluid flow problems, Shock Waves 15 (2006) 407–423.
- [36] S. Tokareva, E. Toro, HLLC-type Riemann solver for the [Baer–Nunziato](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546F6B617265766132303130s1) equations of compressible two-phase flow, J. Comput. Phys. 229 (2010) [3573–3604.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546F6B617265766132303130s1)
- [37] B. Tian, E. Toro, C. Castro, A [path-conservative](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5469616E32303131s1) method for a five-equation model of two-phase flow with an HLLC-type Riemann solver, Comput. Fluids 46 (2011) [122–132.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5469616E32303131s1)
- [38] C. Rohde, C. Zeiler, A relaxation Riemann solver for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib526F68646532303135s1) two-phase flow with phase transition and surface tension, Appl. Numer. Math. 95 (2015) [267–279.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib526F68646532303135s1)
- [39] S. Fechter, C.-D. Munz, C. Rohde, C. Zeiler, Approximate Riemann solver for compressible liquid vapor flow with phase transition and surface tension, Comput. Fluids (2017), [https://doi.org/10.1016/j.compfluid.2017.03.026,](https://doi.org/10.1016/j.compfluid.2017.03.026) in press.
- [40] R. Fedkiw, T. Aslam, B. Merriman, S. Osher, A [non-oscillatory](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665646B697731393939s1) Eulerian approach to interfaces in multimaterial flows (the ghost fluid method), J. Comput. Phys. 152 (1999) [457–492.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665646B697731393939s1)
- [41] H. Terashima, G. Tryggvason, A [front-tracking/ghost-fluid](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546572617368696D6132303039s1) method for fluid interfaces in compressible flows, J. Comput. Phys. 228 (2009) 4012–4037.
- [42] W. Bo, J.W. Grove, A volume of fluid method based ghost fluid method for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib426F32303134s1) multi-fluid flows, Comput. Fluids 90 (2014) 113–122.
- [43] R.P. Fedkiw, T. Aslam, S. Xu, The ghost fluid method for deflagration and detonation [discontinuities,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665646B69773139393961s1) J. Comput. Phys. 154 (1999) 393–427.
- [44] R.W. Houim, K.K. Kuo, A ghost fluid method for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib486F75696D32303133s1) reacting flows with phase change, J. Comput. Phys. 235 (2013) 865–900.
- [45] S. Fechter, C.-D. Munz, A discontinuous Galerkin-based sharp-interface method to simulate [three-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665636874657232303135s1) compressible two-phase flow, Int. J. Numer. Methods Fluids 78 (2015) [413–435.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665636874657232303135s1)
- [46] T. Liu, B. Khoo, K. Yeo, Ghost fluid method for strong shock impacting on material interface, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C697532303033s1) Phys. 190 (2003) 651–681.
- [47] C.W. Wang, T.G. Liu, B.C. Khoo, A real ghost fluid method for the simulation of [multimedium](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib57616E673230303661s1) compressible flow, SIAM J. Sci. Comput. 28 (2006) [278–302.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib57616E673230303661s1)
- [48] H. Terashima, G. Tryggvason, A [front-tracking](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546572617368696D6132303130s1) method with projected interface conditions for compressible multi-fluid flows, Comput. Fluids 39 (2010) [1804–1814.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib546572617368696D6132303130s1)
- [49] C. Liu, C. Hu, Adaptive THINC-GFM for compressible [multi-medium](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C697532303137s1) flows, J. Comput. Phys. 342 (2017) 43–65.
- [50] S.M.H. Karimian, G.E. Schneider, Pressure-based computational method for compressible and incompressible flows, [J. Thermophys.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6172696D69616E31393934s1) Heat Transf. 8 (1994) [267–274.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6172696D69616E31393934s1)
- [51] P. Wesseling, Principles of [Computational](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib57657373656C696E6732303031s1) Fluid Dynamics, Springer, 2001.
- [52] F. Cordier, P. Degond, A. Kumbaro, An [asymptotic-preserving](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436F726469657232303132s1) all-speed scheme for the Euler and Navier–Stokes equations, J. Comput. Phys. 231 (2012) [5685–5704.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib436F726469657232303132s1)
- [53] A.J. Chorin, A numerical method for solving [incompressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib43686F72696E31393937s1) viscous flow problems, J. Comput. Phys. 135 (1997) 118–125.
- [54] F. Xiao, Unified formulation for compressible and incompressible flows by using multi-integrated moments I: [one-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303034s1) inviscid compressible flow, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303034s1) Phys. 195 (2004) 629–654.
- [55] S.Y. Kadioglu, M. Sussman, S. Osher, J.P. Wright, M. Kang, A second order primitive [preconditioner](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6164696F676C7532303035s1) for solving all speed multi-phase flows, J. Comput. Phys. 209 (2005) [477–503.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6164696F676C7532303035s1)
- [56] F. Xiao, R. Akoh, S. Ii, Unified formulation for compressible and incompressible flows by using multi-integrated moments II: [multi-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303036s1) version for compressible and [incompressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303036s1) flows, J. Comput. Phys. 213 (2006) 31–56.
- [57] N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw, A method for avoiding the acoustic time step restriction in [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B776174726132303039s1) flow, J. Comput. Phys. 228 (2009) [4146–4161.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B776174726132303039s1)
- [58] S. [Venkateswaran,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib56656E6B61746573776172616E32303032s1) J.W. Lindau, R.F. Kunz, C.L. Merkle, Computation of multiphase mixture flows with compressibility effects, J. Comput. Phys. 180 [\(2002\)](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib56656E6B61746573776172616E32303032s1) 54–77.
- [59] R. Saurel, O. Le Métayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiff [mechanical](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53617572656C32303037s1) relaxation, Shock Waves 16 (2007) [209–232.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53617572656C32303037s1)
- [60] S. [LeMartelot,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C654D617274656C6F7432303133s1) B. Nkonga, R. Saurel, Liquid and liquid–gas flows at all speeds, J. Comput. Phys. 255 (2013) 53–82.
- [61] J. Van Doormaal, G. Raithby, B. McDonald, The segregated approach to predicting viscous [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib56616E446F6F726D61616C31393837s1) fluid flows, ASME J. Turbomach. 109 (1987) [268–277.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib56616E446F6F726D61616C31393837s1)
- [62] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in [Computational](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F756B616C6C656432303136s1) Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, [Springer,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F756B616C6C656432303136s1) 2016.
- [63] H. Bijl, P. Wesseling, A unified method for computing incompressible and compressible flows in [boundary-fitted](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib42696A6C31393938s1) coordinates, J. Comput. Phys. 141 (1998) [153–173.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib42696A6C31393938s1)
- [64] F.H. Harlow, A.A. Amsden, Numerical calculation of almost [incompressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4861726C6F7731393638s1) flow, J. Comput. Phys. 3 (1968) 80–93.
- [65] F.H. Harlow, A.A. Amsden, A numerical fluid dynamics [calculation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4861726C6F773139373161s1) method for all flow speeds, J. Comput. Phys. 8 (1971) 197–213.
- [66] R. Issa, A. Gosman, A. Watkins, The computation of compressible and [incompressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4973736131393836s1) recirculating flows by a non-iterative implicit scheme, J. Comput. Phys. 62 (1986) [66–82.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4973736131393836s1)
- [67] K.C. Karki, S.V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary [configurations,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B61726B6931393839s1) AIAA J. 27 (1989) 1167–1174.
- [68] K.-H. Chen, R. Pletcher, Primitive variable, strongly implicit calculation procedure for viscous flows at all speeds, AIAA J. 29 (1991) [1241–1249.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4368656E31393931s1)
- [69] I. [Demirdžic,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656D6972647A696331393933s1) ´ Ž. Lilek, M. Peric, ´ A collocated finite volume method for predicting flows at all speeds, Int. J. Numer. Methods Fluids 16 (1993) [1029–1050.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656D6972647A696331393933s1)
- [70] S.M.H. Karimian, G.E. Schneider, Pressure-based [control-volume](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B6172696D69616E31393935s1) finite element method for flow at all speeds, AIAA J. 33 (1995) 1611–1618.
- [71] R.I. Issa, M.H. Javareshkian, [Pressure-based](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4973736131393938s1) compressible calculation method utilizing total variation diminishing schemes, AIAA J. 36 (1998) [1652–1657.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4973736131393938s1)

- [72] F. Moukalled, M. Darwish, A unified [formulation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F756B616C6C656432303030s1) of the segregated class of algorithms for fluid flow at all speeds, Numer. Heat Transf., Part B 37 (2000) [103–139.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F756B616C6C656432303030s1)
- [73] Z. Chen, A.J. Przekwas, A coupled pressure-based computational method for [incompressible/compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4368656E32303130s1) flows, J. Comput. Phys. 229 (2010) [9150–9165.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4368656E32303130s1)
- [74] M. Darwish, F. Moukalled, A fully coupled [Navier–Stokes solver](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4461727769736832303134s1) for fluid flow at all speeds, Numer. Heat Transf., Part B, Fundam. 65 (2014) 410–444.
- [75] A. Miettinen, T. Siikonen, Application of pressureand density-based methods for different flow speeds: application of [pressureand](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D69657474696E656E32303135s1) density-based methods for different flow speeds, Int. J. Numer. Methods Fluids 79 (2015) [243–267.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D69657474696E656E32303135s1)
- [76] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled [pressure-based](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303137s1) finite-volume framework for the simulation of fluid flows at all speeds in complex [geometries,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616F32303137s1) J. Comput. Phys. 346 (2017) 91–130.
- [77] F. Denner, B. van Wachem, [Compressive](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313465s1) VOF method with skewness correction to capture sharp interfaces on arbitrary meshes, J. Comput. Phys. 279 (2014) [127–144.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313465s1)
- [78] F. Harlow, A. Amsden, Fluid Dynamics, [Monograph](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4861726C6F7731393731s1) LA-4700, Los Alamos National Laboratory, 1971.
- [79] R. Saurel, R. Abgrall, A simple method for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib53617572656C3139393961s1) multifluid flows, SIAM J. Sci. Comput. 21 (1999) 1115–1145.
- [80] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free [boundaries,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4869727431393831s1) J. Comput. Phys. 39 (1981) 201–225.
- [81] V. Levich, [Physicochemical](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C657669636831393632s1) Hydrodynamics, Prentice Hall, 1962.
- [82] J. Ferziger, M. Peric, ´ Computational Methods for Fluid Dynamics, 3rd edition, Springer-Verlag, Berlin, [Heidelberg, New](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4665727A6967657232303032s1) York, 2002.
- [83] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with [monotonicity-preserving](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313561s1) correction of mesh skewness, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313561s1) Phys. 298 (2015) 466–479.
- [84] P. Roe, [Characteristic-based](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib526F6531393836s1) schemes for the Euler equations, Annu. Rev. Fluid Mech. 18 (1986) 337–365.
- [85] F. Denner, [Balanced-Force](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E657232303133s1) Two-Phase Flow Modelling on Unstructured and Adaptive Meshes, Ph.D. thesis, Imperial College London, 2013.
- [86] I. Demirdžic, ´ S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using [unstructured](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656D6972647A696331393935s1) moving meshes with cells of arbitrary topology, Comput. Methods Appl. Mech. Eng. 125 (1995) [235–255.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656D6972647A696331393935s1)
- [87] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) [1525–1532.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5268696531393833s1)
- [88] P. Zwart, The Integrated [Space–Time](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5A7761727431393939s1) Finite Volume Method, Ph.D. thesis, University of Waterloo, 1999.
- [89] F. Ham, G. Iaccarino, Energy conservation in collocated [discretization](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib48616D32303034s1) schemes on unstructured meshes, Annu. Res. Briefs, Center for Turbulence (2004) [3–14.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib48616D32303034s1)
- [90] F. Denner, B. van Wachem, Fully-coupled [balanced-force](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313461s1) VOF framework for arbitrary meshes with least-squares curvature evaluation from volume [fractions,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E65723230313461s1) Numer. Heat Transf., Part B, Fundam. 65 (2014) 218–255.
- [91] F. Denner, B. van Wachem, Numerical time-step [restrictions](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E657232303135s1) as a result of capillary waves, J. Comput. Phys. 285 (2015) 24–40.
- [92] R. Kunz, W. Cope, S. [Venkateswaran,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4B756E7A31393939s1) Development of an implicit method for multi-fluid flow simulations, J. Comput. Phys. 152 (1999) 78–101.
- [93] M. Darbandi, V. Mokarizadeh, A modified [pressure-based](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44617262616E646932303034s1) algorithm to solve flow fields with shock and expansion waves, Numer. Heat Transf., Part B, Fundam. 46 (2004) [497–504.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44617262616E646932303034s1)
- [94] S. Osher, S. [Chakravarthy,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4F7368657231393834s1) High resolution schemes and the entropy condition, SIAM J. Numer. Anal. 21 (1984) 955–984.
- [95] D.P. Garrick, W.A. Hagen, J.D. Regele, An interface capturing scheme for modeling atomization in [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4761727269636B3230313761s1) flows, J. Comput. Phys. 344 (2017) [260–280.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4761727269636B3230313761s1)
- [96] O. Ubbink, R. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, [J. Comput.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib556262696E6B31393939s1) Phys. 153 (1999) 26–50.
- [97] V. Gopala, B. van Wachem, Volume of fluid methods for [immiscible-fluid](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib476F70616C6132303038s1) and free-surface flows, Chem. Eng. J. 141 (2008) 204–221.
- [98] K.-M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536879756532303134s1) two-phase flow: the algebraic THINC approach, J. Comput. Phys. 268 (2014) [326–354.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536879756532303134s1)
- [99] F. Denner, A. Charogiannis, M. Pradas, C.N. Markides, B. van Wachem, S. Kalliadasis, Solitary waves on falling liquid films in the [inertia-dominated](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E657232303138s1) regime, J. Fluid Mech. 837 (2018) [491–519.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E657232303138s1)
- [100] F. Denner, Frequency dispersion of [small-amplitude](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656E6E657232303136s1) capillary waves in viscous fluids, Phys. Rev. E 94 (2016) 023110.
- [101] M. Raessi, J. [Mostaghimi,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib52616573736932303037s1) M. Bussmann, Advecting normal vectors: a new method for calculating interface normals and curvatures when modeling [two-phase](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib52616573736932303037s1) flows, J. Comput. Phys. 226 (2007) 774–797.
- [102] J.D. Anderson, Modern [Compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib416E646572736F6E32303033s1) Flow: With a Historical Perspective, McGraw-Hill, New York, 2003.
- [103] S. Popinet, [Numerical](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib506F70696E657432303138s1) models of surface tension, Annu. Rev. Fluid Mech. 50 (2018) 49–75.
- [104] J. Banks, T. Aslam, W. Rider, On sub-linear [convergence](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib42616E6B7332303038s1) for linearly degenerate waves in capturing schemes, J. Comput. Phys. 227 (2008) 6985–7002.
- [105] S. Balay, W. Gropp, L.C. McInnes, B.F. Smith, Efficient [management](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib42616C617931393937s1) of parallelism in object oriented numerical software libraries, in: E. Arge, A. Bruasat, H. Langtangen (Eds.), Modern Software Tools in Scientific Computing, [Birkhäuser Press,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib42616C617931393937s1) 1997, pp. 163–202.
- [106] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W.D. Gropp, D. Kaushik, M.G. Knepley, L.C. McInnes, K. Rupp, B.F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, <http://www.mcs.anl.gov/petsc>, 2017.
- [107] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. [Buschelman,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib70657473632D757365722D726566s1) L. Dalcin, V. Eijkhout, D. Kaushik, M.G. Knepley, D.A. May, L.C. McInnes, W.D. Gropp, K. Rupp, P. Sanan, B.F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report [ANL-95/11](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib70657473632D757365722D726566s1) - Revision 3.8, Argonne National [Laboratory,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib70657473632D757365722D726566s1) 2017.
- [108] R. Dembo, S. [Eisenstat,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib44656D626F31393832s1) T. Steihaug, Inexact newton methods, SIAM J. Numer. Anal. 19 (1982) 400–408.
- [109] R.K. Shukla, C. Pantano, J.B. Freund, An interface capturing method for the simulation of multi-phase [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5368756B6C6132303130s1) flows, J. Comput. Phys. 229 (2010) [7411–7439.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5368756B6C6132303130s1)
- [110] K. So, X. Hu, N. Adams, [Anti-diffusion](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536F32303132s1) interface sharpening technique for two-phase compressible flow simulations, J. Comput. Phys. 231 (2012) [4304–4323.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib536F32303132s1)
- [111] A. Chiapolino, R. Saurel, B. Nkonga, Sharpening diffuse interfaces with [compressible](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib43686961706F6C696E6F32303137s1) fluids on unstructured meshes, J. Comput. Phys. 340 (2017) [389–417.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib43686961706F6C696E6F32303137s1)
- [112] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, [Pressure-velocity](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F6775656E32303132s1) coupling allowing acoustic calculation in low Mach number flow, J. Comput. Phys. 231 (2012) [5522–5541.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D6F6775656E32303132s1)
- [113] P. Lax, B. Wendroff, Systems of [conservation](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4C617831393630s1) laws, Commun. Pure Appl. Math. 13 (1960) 217–237.
- [114] P. Woodward, P. Colella, The numerical simulation of [two-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib576F6F647761726431393834s1) fluid flow with strong shocks, J. Comput. Phys. 173 (1984) 115–173.
- [115] J.-F. Haas, B. Sturtevant, Interaction of weak shock waves with cylindrical and spherical gas [inhomogeneities,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4861617331393837s1) J. Fluid Mech. 181 (1987) 41.
- [116] J.J. Quirk, S. Karni, On the dynamics of a [shock–bubble](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib517569726B31393936s1) interaction, J. Fluid Mech. 318 (1996) 129.
- [117] J.H.J. Niederhaus, J.A. Greenough, J.G. Oakley, D. Ranjan, M.H. Anderson, R. Bonazza, A computational parameter study for the [three-dimensional](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4E69656465726861757332303038s1) [shock–bubble](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4E69656465726861757332303038s1) interaction, J. Fluid Mech. 594 (2008).
- [118] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4A6F686E73656E32303036s1) flow problems, J. Comput. Phys. 219 (2006) 715–732.
- [119] R. Nourgaliev, T. Dinh, T. Theofanous, Adaptive [characteristics-based](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4E6F757267616C69657632303036s1) matching for compressible multifluid dynamics, J. Comput. Phys. 213 (2006) [500–529.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4E6F757267616C69657632303036s1)
- [120] D. Igra, K. Takayama, Investigation of [aerodynamic](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4967726132303031s1) breakup of a cylindrical water droplet, At. Sprays 11 (2001) 20.
- [121] J.C. Meng, T. Colonius, Numerical [simulations](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D656E6732303135s1) of the early stages of high-speed droplet breakup, Shock Waves 25 (2015) 399–414.

- [122] G. Xiang, B. Wang, Numerical study of a planar shock [interacting](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616E6732303137s1) with a cylindrical water column embedded with an air cavity, J. Fluid Mech. 825 (2017) [825–852.](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib5869616E6732303137s1)
- [123] J. Meng, Numerical Simulations of Droplet [Aerobreakup,](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib4D656E6732303136s1) Ph.D. thesis, California Institute of Technology, Pasadena, California, USA, 2016.
- [124] D.-Y. Peng, D.B. Robinson, A new [two-constant](http://refhub.elsevier.com/S0021-9991(18)30253-5/bib50656E6731393736s1) equation of state, Ind. Eng. Chem. Fundam. 15 (1976) 59–64.

## **Update**

## **Journal of Computational Physics**

**Volume 381, Issue , 15 March 2019, Page 290–291**

**DOI: https://doi.org/10.1016/j.jcp.2018.11.017**

ELSEVIER

Contents lists available at ScienceDirect

www.elsevier.com/locate/jcp

![](_page_44_Picture_5.jpeg)

#### Corrigendum

# Corrigendum to "Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation" [J. Comput. Phys. 367 (2018) 192–234]

![](_page_44_Picture_8.jpeg)

Fabian Denner\*, Berend G.M. van Wachem

Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany

#### ARTICLE INFO

Article history:
Received 9 November 2018
Accepted 10 November 2018
Available online 25 January 2019

The authors regret that the definition of the Second-Order Backward Euler scheme for a varying time-step, given in Eq. (18), is incorrect. Assuming a varying time-step is applied, the Second-Order Backward Euler scheme (also often called BDF2 scheme) for the transient derivative of a general flow variable  $\phi$  at cell P is defined, following the derivation given in Appendix A, as

$$\int_{V_{-}} \frac{\partial \phi}{\partial t} dV \approx \left[ \left( \frac{1}{\Delta t_{1}} + \frac{1}{\Delta \tau} \right) \phi_{p}^{(t)} - \left( \frac{1}{\Delta t_{1}} + \frac{1}{\Delta t_{2}} \right) \phi_{p}^{(t-\Delta t_{1})} + \frac{\Delta t_{1}}{\Delta t_{2} \Delta \tau} \phi_{p}^{(t-\Delta \tau)} \right] V_{p} + \mathcal{O}(\Delta t_{1} \Delta \tau), \tag{1}$$

where  $\Delta t_1$  is the current time-step,  $\Delta t_2$  is the previous time-step,  $\Delta \tau = \Delta t_1 + \Delta t_2$ , superscript (t) denotes the value at the new time-level, superscript  $(t - \Delta t_1)$  denotes the value at the previous time-level and superscript  $(t - \Delta \tau)$  denotes the value at the previous-previous time-level.

Since the correct version of the Second-Order Backward Euler scheme as given above was already implemented in the software framework used to develop the proposed pressure-based algorithm, this correction has no effect on the presented results or the findings of the article.

The authors would like to apologise for any inconvenience caused.

#### Appendix A. Derivation of the Second-Order Backward Euler scheme

The Second-Order Backward Euler scheme for varying time-steps is derived from a Taylor series expansion with respect to time, given for a general flow variable  $\phi$  as

$$\phi^{(t-\Delta t_1)} \approx \phi^{(t)} - \Delta t_1 \frac{\partial \phi}{\partial t} \Big|^{(t)} + \frac{\Delta t_1^2}{2} \frac{\partial^2 \phi}{\partial t^2} \Big|^{(t)} - \frac{\Delta t_1^3}{6} \frac{\partial^3 \phi}{\partial t^3} \Big|^{(t)} + \mathcal{O}(\Delta t^4), \tag{A.1}$$

E-mail address: fabian.denner@ovgu.de (F. Denner).

DOI of original article: https://doi.org/10.1016/j.jcp.2018.04.028.

<sup>\*</sup> Corresponding author.

$$\phi^{(t-\Delta\tau)} \approx \phi^{(t)} - \Delta\tau \left. \frac{\partial \phi}{\partial t} \right|^{(t)} + \left. \frac{\Delta\tau^2}{2} \left. \frac{\partial^2 \phi}{\partial t^2} \right|^{(t)} - \left. \frac{\Delta\tau^3}{6} \left. \frac{\partial^3 \phi}{\partial t^3} \right|^{(t)} + \mathcal{O}(\Delta\tau^4), \tag{A.2}$$

where  $\Delta t_1$  is the current time-step,  $\Delta t_2$  is the previous time-step,  $\Delta \tau = \Delta t_1 + \Delta t_2$ , superscript (t) denotes values at the new time-level, superscript  $(t - \Delta t_1)$  denotes values at the previous-previous time-level. After rearranging Eq. (A.1) and substituting Eq. (A.2) for  $\partial^2 \phi / \partial t^2$ , the transient derivative of  $\phi$  can be approximated as

$$\frac{\partial \phi}{\partial t}\Big|^{(t)} \approx \frac{1}{1 - \frac{\Delta t_1}{\Delta \tau}} \left[ \left( \frac{1}{\Delta t_1} - \frac{\Delta t_1}{\Delta \tau^2} \right) \phi^{(t)} - \frac{1}{\Delta t_1} \phi^{(t - \Delta t_1)} + \frac{\Delta t_1}{\Delta \tau^2} \phi^{(t - \Delta \tau)} + \frac{\Delta t_1}{6} \frac{\Delta t_2}{\partial t^3} \frac{\partial^3 \phi}{\partial t^3} \right]^{(t)} \right] + \text{HOT}, \tag{A.3}$$

where HOT denotes higher-order terms. After multiplying the numerator and the denominator on the right-hand side with  $\Delta \tau$ , Eq. (A.3) becomes

$$\frac{\partial \phi}{\partial t}\Big|^{(t)} \approx \left(\frac{1}{\Delta t_1} + \frac{1}{\Delta \tau}\right) \phi^{(t)} - \left(\frac{1}{\Delta t_1} + \frac{1}{\Delta t_2}\right) \phi^{(t-\Delta t_1)} + \frac{\Delta t_1}{\Delta t_2 \Delta \tau} \phi^{(t-\Delta \tau)} + \underbrace{\frac{\Delta t_1 \Delta \tau}{6} \frac{\partial^3 \phi}{\partial t^3}\Big|^{(t)}}_{\mathcal{O}(\Delta t_1 \Delta \tau)} + \text{HOT}.$$
(A.4)

If the time-step  $\Delta t$  is constant, with  $\Delta t = \Delta t_1 = \Delta t_2$  and  $\Delta \tau = 2\Delta t$ , Eq. (A.4) reduces to

$$\frac{\partial \phi}{\partial t}\Big|^{(t)} \approx \frac{3\phi^{(t)} - 4\phi^{(t-\Delta t)} + \phi_p^{(t-2\Delta t)}}{2\Delta t} + \underbrace{\frac{\Delta t^2}{3} \frac{\partial^3 \phi}{\partial t^3}\Big|^{(t)}}_{\mathcal{O}(\Delta t^2)} + \text{HOT}.$$
(A.5)