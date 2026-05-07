# A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible Navier-Stokes equations with general equation of state

Michael Dumbser<sup>a</sup> , Vincenzo Casulli<sup>a</sup>

*<sup>a</sup>Laboratory of Applied Mathematics, Department of Civil, Environmental and Mechanical Engineering University of Trento, Via Mesiano 77, 38123 Trento (TN), Italy*

# Abstract

In the present paper a new efficient semi-implicit finite volume method is proposed for the solution of the compressible Euler and Navier-Stokes equations of gas dynamics with general equation of state (EOS). The discrete flow equations lead to a mildly nonlinear system for the pressure, containing a diagonal nonlinearity due to the EOS. The remaining linear part of the system is symmetric and at least positive semi-definite. Mildly nonlinear systems with this particular structure can be very efficiently solved with a nested Newton-type technique.

The new numerical method has to obey only a mild CFL condition, which is based on the fluid velocity and not on the sound speed. This makes the scheme particularly interesting for low Mach number flows, because large time steps are permitted. Moreover, being locally and globally conservative, the new method behaves also very well in the presence of shock waves. The proposed algorithm is first validated against the exact solution of a large set of onedimensional Riemann problems for inviscid flows with three different EOS: the ideal gas law, the van der Waals EOS and the Redlich-Kwong EOS. In the final part of the paper, the method is extended to the two-dimensional viscous case.

*Keywords:* semi-implicit finite volume method, staggered grid, large time steps, mildly nonlinear system, compressible and incompressible flows, general equation of state (EOS), cubic equation of state (EOS), compressible Navier-Stokes equations

# 1. Introduction

The use of numerical methods for computational fluid dynamics (CFD) is nowadays widespread in scientific and engineering applications. For example, CFD is the backbone in the design process of modern cars, aircraft, engines or wind turbines, but also in the simulation of geophysical and environmental flows, e.g. for the simulation of atmospheric flows, oceanic currents and tides, storm surges, tsunami waves and for the modelling of water flow in rivers and lakes. The basic governing equations in all these different applications can be derived from first principles by considering the conservation of mass, momentum and energy, leading to the so-called Navier-Stokes equations or one of their simplifications, like the Euler equations or shallow watertype equations. A major difference between the various applications is the *Mach number M* <sup>=</sup> <sup>k</sup>v<sup>k</sup> /*c*, which is the ratio between the flow velocity <sup>v</sup> and the sound speed *c*. While typical industrial applications present moderate to high Mach numbers with the formation of shock waves, geophysical flows are usually characterized by low to very low Mach numbers. In general, each of these two flow regimes requires the design of different and specific numerical methods.

At the aid of asymptotic analysis it can be shown [36, 37, 42] that in the low Mach number limit *M* → 0 of the compressible Navier-Stokes equations, one retrieves the incompressible Navier–Stokes equations with their typical ∇ · v = 0 constraint on the divergence of the velocity field. In the incompressible limit, the pressure is composed of two parts: a spatially constant thermodynamic background pressure that satisfies the equation of state, and the hydrodynamic pressure fluctuations that are governed by an elliptic pressure Poisson equation. It is very challenging to construct numerical methods that apply to both flow regimes, namely to the compressible and to the incompressible one. While semi-implicit methods are the state-of-the-art for the solution of the incompressible Navier–Stokes equations with and without free surface [19, 20, 56, 7, 58, 35, 12], in the compressible case the family of explicit upwind finite difference and Godunov-type finite volume schemes [39, 33, 48, 44, 34, 29, 41, 54, 40, 53] is more popular. There have been several important contributions to the extension of staggered semi-implicit pressure-based methods to the compressible regime, see for example [15, 43, 45]. We also would like to refer to the semi-implicit family of high order discontinuous Galerkin finite element schemes proposed in [22, 23, 24].

For some applications the ideal gas law is a sufficiently accurate approximation. In other cases, a more complex equation of state is required to account for real gas effects, such as the van der Waals EOS [57], the Redlich-Kwong EOS [57], the Peng-Robinson EOS [46] or an even more complex EOS like the one of real water, vapor and steam [60, 30]. It is very important to note that the sound speed in liquids is of the order of one to two thousand meters per second, while it is of the order of only several hundreds of meters per second in gases. In the wet steam region, where liquid and gas phase coexist, the sound speed drops significantly to one to ten meters per second. Therefore, in the case of compressible multi-phase flows that contain at the same time liquid, vapor and wet steam, the local sound speed and thus the local Mach number changes dramatically, from very low values inside the liquid over moderate values inside the vapour to very high Mach numbers inside the wet steam. For such flows, it would therefore be useful to have one numerical method that is able to solve the governing PDE in all flow regimes accurately and efficiently, from very low Mach numbers up to the high Mach number regime. In [27] a high order accurate explicit Godunovtype finite volume scheme for general equations of state was presented. However, being explicit, that algorithm is not efficient in the case of liquid-dominated flows that contain only few local vapour bubbles, since in this case the time step of the scheme is limited by the large speed of sound in the liquid.

It is therefore the aim of this paper to provide a novel pressure-based semi-implicit method for the compressible Euler equations that is able to deal with general equations of state, which is locally and globally conservative and whose time step is only limited by the flow velocity and not by the sound speed. The new method is designed to apply simultaneously to very low Mach number flows as well as to highly compressible flows with shock waves. In the proposed scheme, the density equation as well as the nonlinear convective terms for momentum and kinetic energy are discretized explicitly, while pressure in the momentum equation and velocity in the energy equation are taken implicitly. This removes the stability condition on the sound speed, and requires only a mild restriction of the time step based on the flow velocity. Then, the discrete momentum equation is inserted into the discrete energy equation, leading to a reduced mildly nonlinear system for the pressure. This nonlinear system has the nonlinearity stemming from the equation of state only on the diagonal, while the remaining linear part of the system is symmetric and at least positive semi-definite. Hence, the pressure can be efficiently obtained with the new family of (nested) Newton-type techniques recently introduced and analyzed by Casulli et al. in [11, 17, 18, 8, 9]. The method proposed in this paper has several similarities with the one forwarded by Park and Munz [45], but while the latter was using a linear pressure correction equation and thus was restricted to the ideal gas case, the present approach solves a mildly nonlinear system for the pressure and is sufficiently general to handle complex real gas EOS. Concerning semi-implicit asymptotic-preserving schemes for all-Mach number flows and general equations of state, we also would like to refer to the nice work presented by Cordier et al. in [21], which, however, required the solution of a nonlinear system for pressure and enthalphy, while the present scheme proposed in this paper only requires the solution of a set of scalar systems for the unknown pressure. Furthermore, the method proposed in [21] did not employ the novel nested Newton technique of Casulli et al., which makes the scheme presented in this paper particularly simple, robust and efficient.

The rest of the paper is organized as follows: in Section 2 we present the governing differential equations and their numerical approximation in the one-dimensional inviscid case. A thorough validation of one-dimensional inviscid flows is presented in Section 3, while the extension to multi-dimensional viscous flows is presented in Section 4. Some computational results for the multi-dimensional case are shown in Section 5. Finally, in Section 6 we give some concluding remarks.

#### 2. Governing equations and numerical method

# 2.1. Governing PDE

The compressible Euler equations of gas dynamics, which represent the principles of conservation of mass, momentum and total energy, in one space dimension read as follows:

$$\frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix} + \frac{\partial}{\partial x} \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u (\rho E + p) \end{pmatrix} = 0, \tag{1}$$

where  $t \in \mathbb{R}_0^+$  denotes time and  $x \in \Omega = [x_L, x_R] \subset \mathbb{R}$  is the spatial coordinate. The one-dimensional computational domain is denoted by  $\Omega$ . Furthermore,  $\rho$  denotes the fluid density, p is the fluid pressure, u is the flow velocity,  $\rho E = \rho e + \rho k = \rho e + \frac{1}{2}\rho u^2$  denotes the total energy density,  $\rho k = \frac{1}{2}\rho u^2$  is the kinetic energy density of the fluid and  $e = e(T, \rho)$  is the specific internal energy per unit mass, which in general depends on temperature T and density  $\rho$ . This relation is the so-called *caloric* equation of state (EOS), while the relation  $p = p(T, \rho)$  is the so-called *thermal* equation of state (EOS). Usually, from these two relations, the temperature is eliminated, leading to one single equation of state of the form  $e = e(p, \rho)$ , which is typically used in Godunov-type finite volume schemes, and which will also be used in this paper. Another useful quantity is the so-called specific enthalpy, which is defined as  $h = e + p/\rho$  and which allows to rewrite the flux for the total energy density as follows:  $u(\rho E + p) = u(\rho k) + h(\rho u)$ . In order to simplify the notation, h will be used later throughout the paper. The PDE system (1) can be rewritten in terms of the primitive variables  $\rho$ , u and p as follows:

$$\frac{\partial}{\partial t} \begin{pmatrix} \rho \\ u \\ p \end{pmatrix} + \begin{pmatrix} u & \rho & 0 \\ 0 & u & \frac{1}{\rho} \\ 0 & K & u \end{pmatrix} \frac{\partial}{\partial x} \begin{pmatrix} \rho \\ u \\ p \end{pmatrix} = 0, \tag{2}$$

with the bulk modulus  $K = \rho c^2 = \frac{(p-\rho^2 e_\rho)}{\rho e_p}$ , the sound speed c and the abbreviations  $e_\rho = \partial e/\partial \rho$  and  $e_p = \partial e/\partial p$ . The eigenvalues of system (2), which are the same as for (1), are  $\lambda_1 = u - c$ ,  $\lambda_2 = u$  and  $\lambda_3 = u + c$ . The terms in the system matrix of eqn. (2) that give rise to the sound speed c are the pressure term  $1/\rho p_x$  in the momentum equation and the velocity term  $Ku_x$  in the pressure equation. Therefore, these terms will have to be discretized *implicitly* in the numerical method in order to avoid a CFL condition based on the sound speed; see also [13] for a similar discussion in the case of the shallow water equations.

# 2.1.1. Ideal gas EOS

For the ideal gas, for example, the thermal and caloric EOS read, respectively,

$$\frac{p}{\rho} = RT$$
, and  $e = c_v T$ , (3)

with the specific gas constant  $R = c_p - c_v$ , and the heat capacities  $c_v$  and  $c_p$  at constant volume and at constant pressure, respectively. From the two relations in (3) the temperature can be eliminated, yielding

$$e = e(p, \rho) = \frac{p}{(\gamma - 1)\rho},\tag{4}$$

where  $\gamma = c_p/c_v$  denotes the so-called ratio of specific heats.

#### 2.1.2. General cubic EOS

For a general cubic equation of state according to [59], the thermal EOS reads

$$p = \frac{RT}{v - b} - \frac{a(T)}{(v - br_1)(v - br_2)},\tag{5}$$

with the specific volume  $v = 1/\rho$ , the so-called co-volume b, the specific gas constant R, two constant parameters  $r_1$  and  $r_2$  and a function a(T) that governs the attraction term in the EOS. For the special case  $r_1 = r_2 = 0$  the famous

van der Waals EOS [57] is reproduced, while the Redlich-Kwong EOS [47] is obtained by setting  $r_1 = 0$  and  $r_2 = -1$ . Finally, the Peng-Robinson EOS [46] can be retrieved with the choice  $r_1 = -1 - \sqrt{2}$  and  $r_2 = -1 + \sqrt{2}$ . The caloric EOS of the general cubic equation of state (5) is given for constant  $c_v$  according to [59] by

$$e(T,\rho) = c_v T + \frac{a - Ta'(T)}{b} U(v, b, r_1, r_2), \tag{6}$$

with a'(T) = da/dT and U a known function that depends on the EOS. For the van der Waals gas  $U = -b\rho$ , while for all other EOS with  $r_1 \neq r_2$  one has  $U = \frac{1}{r_1 - r_2} \ln \left( \frac{v - b r_1}{v - b r_2} \right)$ , see [59]. From (5) and (6) it is clear that for a general cubic EOS the relation  $e = e(p, \rho)$  can become very complicated and is in general highly nonlinear. However, for a given density and for a physically meaningful choice of parameters, the specific internal energy e is a non-negative, non-decreasing function of the fluid pressure p.

## 2.2. Semi-implicit discretization on a staggered grid

For the discretization of PDE system (1) we introduce a *staggered* grid, where the *primary* control volumes  $\Omega_i = [x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}]$  are intervals of length  $\Delta x_i = x_{i+\frac{1}{2}} - x_{i-\frac{1}{2}}$  with barycenters  $x_i = \frac{1}{2}(x_{i-\frac{1}{2}} + x_{i+\frac{1}{2}})$ . The number of primary control volumes is denoted by  $N_x$ . For convenience, we also define the *dual* mesh spacing  $\Delta x_{i+\frac{1}{2}} = x_{i+1} - x_i = \frac{1}{2}(\Delta x_i + \Delta x_{i+1})$ . The discrete pressure  $p_i^n$  is defined in the centers of the primary cells  $\Omega_i$  and the velocity  $u_{i+\frac{1}{2}}$  is located at the boundaries of the control volumes. We also define the discrete density  $\rho_{i+\frac{1}{2}}^n$  at the zone boundaries, so that the discrete momentum density  $(\rho u)_{i+\frac{1}{2}}^n = \rho_{i+\frac{1}{2}}^n u_{i+\frac{1}{2}}^n$  is unambiguously defined at the cell boundary. The reason for this choice is motivated by the objective of obtaining a simple explicit discretization of the convective terms in the mass and momentum conservation equations that preserves uniform pressure and velocity flows.

The discrete total energy density  $(\rho E)_i^n$  is located at the cell centers. However, since  $(\rho E)_i^n$  is not only a function of pressure, but also of the fluid density and the kinetic energy, and hence of the fluid velocity, we need to link  $(\rho E)_i^n$  properly to the other flow quantities on the staggered grid. For this purpose, the cell-averaged density  $\rho_i^n$  and the cell-averaged kinetic energy density  $(\rho k)_i^n$  are defined as

$$\rho_i^n = \frac{1}{2} \left( \rho_{i-\frac{1}{2}}^n + \rho_{i+\frac{1}{2}}^n \right) \tag{7}$$

and

$$(\rho k)_{i}^{n} = \frac{1}{2} \left( (\rho k)_{i-\frac{1}{2}}^{n} + (\rho k)_{i+\frac{1}{2}}^{n} \right) = \frac{1}{2} \left( \frac{1}{2} \rho_{i-\frac{1}{2}}^{n} (u_{i-\frac{1}{2}}^{n})^{2} + \frac{1}{2} \rho_{i+\frac{1}{2}}^{n} (u_{i+\frac{1}{2}}^{n})^{2} \right), \tag{8}$$

respectively. The cell-averaged total energy density  $(\rho E)_i^n$  is then linked to the other flow quantities by the relation

$$(\rho E)_{i}^{n} = \rho_{i}^{n} e(p_{i}^{n}, \rho_{i}^{n}) + (\rho k)_{i}^{n} = \rho_{i}^{n} e(p_{i}^{n}, \rho_{i}^{n}) + \frac{1}{2} \left( \frac{1}{2} \rho_{i-\frac{1}{2}}^{n} (u_{i-\frac{1}{2}}^{n})^{2} + \frac{1}{2} \rho_{i+\frac{1}{2}}^{n} (u_{i+\frac{1}{2}}^{n})^{2} \right). \tag{9}$$

Finally, we also define the specific enthalpy h at the element interfaces, which will be needed later in our scheme:

$$h_{i+\frac{1}{2}}^{n} = e(p_{i+\frac{1}{2}}^{n}, \rho_{i+\frac{1}{2}}^{n}) + \frac{p_{i+\frac{1}{2}}^{n}}{\rho_{i+\frac{1}{2}}^{n}}, \quad \text{with} \quad p_{i+\frac{1}{2}}^{n} = \max(0, p_{i}^{n}, p_{i+1}^{n}).$$
 (10)

The density equation is now discretized explicity as follows:

$$\rho_{i+\frac{1}{2}}^{n+1} = \rho_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x_{i+\frac{1}{2}}} \left( f_{i+1}^{\rho} - f_{i}^{\rho} \right), \tag{11}$$

with the numerical flux

$$f_i^{\rho} = \frac{1}{2} \left( (\rho u)_{i+\frac{1}{2}}^n + (\rho u)_{i-\frac{1}{2}}^n \right) - \frac{1}{2} |u_i^{\text{max}}| \left( \rho_{i+\frac{1}{2}}^n - \rho_{i-\frac{1}{2}}^n \right)$$
 (12)

and the maximum convective velocity  $|u_i^{\max}| = \max\left(|u_{i-\frac{1}{2}}^n|, |u_{i+\frac{1}{2}}^n|\right)$ . Note that the flow velocity can be easily obtained from mass density and momentum density simply as  $u_{i+\frac{1}{2}}^n = (\rho u)_{i+\frac{1}{2}}^n / \rho_{i+\frac{1}{2}}^n$ .

The semi-implicit discretization of the momentum equation reads

$$(\rho u)_{i+\frac{1}{2}}^{n+1} = F(\rho u)_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x_{i+\frac{1}{2}}} \left( p_{i+1}^{n+1} - p_{i}^{n+1} \right), \tag{13}$$

where pressure is taken *implicitly*, while the *explicit* operator for the discretization of the nonlinear convective terms reads

$$F(\rho u)_{i+\frac{1}{2}}^{n} = (\rho u)_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x_{i+\frac{1}{2}}} \left( f_{i+1}^{\rho u} - f_{i}^{\rho u} \right), \tag{14}$$

with the numerical flux

$$f_i^{\rho u} = \frac{1}{2} \left( u_{i+\frac{1}{2}}^n (\rho u)_{i+\frac{1}{2}}^n + u_{i-\frac{1}{2}}^n (\rho u)_{i-\frac{1}{2}}^n \right) - \frac{1}{2} |u_i^{\text{max}}| \left( (\rho u)_{i+\frac{1}{2}}^n - (\rho u)_{i-\frac{1}{2}}^n \right). \tag{15}$$

Note that the discretization of the nonlinear convective terms in the momentum equation (14) has the same structure as the one of the density (11), in particular the numerical viscosity term is the same. This is important in order to preserve constant velocity and constant pressure flows.

A preliminary discretization of the total energy equation is chosen as follows:

$$\Delta x_{i} \rho_{i}^{n+1} e\left(p_{i}^{n+1}, \rho_{i}^{n+1}\right) + \frac{1}{2} \Delta x_{i} \left(\left(\tilde{\rho k}\right)_{i-\frac{1}{2}}^{n+1} + \left(\tilde{\rho k}\right)_{i+\frac{1}{2}}^{n+1}\right) = \Delta x_{i} F(\rho E)_{i}^{n} - \Delta t \left(\tilde{h}_{i+\frac{1}{2}}^{n+1}(\rho u)_{i+\frac{1}{2}}^{n+1} - \tilde{h}_{i-\frac{1}{2}}^{n+1}(\rho u)_{i-\frac{1}{2}}^{n+1}\right). \tag{16}$$

The meaning of the tilde symbols will be explained later. The explicit operator  $F(\rho E)_i^n$  contains only the discretization of the flux term due to the kinetic energy and is given by

$$F(\rho E)_{i}^{n} = (\rho E)_{i}^{n} - \frac{\Delta t}{\Delta x_{i}} \left( f_{i+\frac{1}{2}}^{\rho k} - f_{i-\frac{1}{2}}^{\rho k} \right), \tag{17}$$

with the numerical flux

$$f_{i+\frac{1}{2}}^{\rho k} = \frac{1}{2} u_{i+\frac{1}{2}}^{n} \left( (\rho k)_{i+1}^{n} + (\rho k)_{i}^{n} \right) - \frac{1}{2} |u_{i+\frac{1}{2}}^{n}| \left( (\rho k)_{i+1}^{n} - (\rho k)_{i}^{n} \right), \tag{18}$$

where we have used the cell-averaged kinetic energy density defined in (7). Note that in the explicit operators for  $\rho_{i+\frac{1}{2}}^{n+1}$ ,  $F(\rho u)_{i+\frac{1}{2}}^n$  and  $F(\rho E)_i^n$  one can recognize an explicit discretization of the advection system of the flux-vector splitting scheme of Toro and Vázquez-Cendón [55]. For a detailed discussion of the eigenstructures of the resulting convection system and the pressure system, see [55].

Inserting the discrete momentum from (13) into the discrete energy equation (16) yields the following preliminary system for the unknown pressure:

$$\Delta x_{i} \rho_{i}^{n+1} e\left(p_{i}^{n+1}, \rho_{i}^{n+1}\right) - \Delta t^{2} \left(\frac{\tilde{h}_{i+\frac{1}{2}}^{n+1}}{\Delta x_{i+\frac{1}{2}}} \left(p_{i+1}^{n+1} - p_{i}^{n+1}\right) - \frac{\tilde{h}_{i-\frac{1}{2}}^{n+1}}{\Delta x_{i-\frac{1}{2}}} \left(p_{i}^{n+1} - p_{i-1}^{n+1}\right)\right) = \Delta x_{i} \left(F(\rho E)_{i}^{n} - \frac{1}{2} \left((\tilde{\rho k})_{i-\frac{1}{2}}^{n+1} + (\tilde{\rho k})_{i+\frac{1}{2}}^{n+1}\right)\right) - \Delta t \left(\tilde{h}_{i+\frac{1}{2}}^{n+1} F(\rho u)_{i+\frac{1}{2}}^{n} - \tilde{h}_{i-\frac{1}{2}}^{n+1} F(\rho u)_{i-\frac{1}{2}}^{n}\right). \tag{19}$$

If the quantities marked with a tilde symbol are directly discretized at the new time level, the resulting system would become strongly nonlinear and difficult to control. For that purpose, we employ a *Picard iteration* technique, as the one suggested in [17]. From now on the iteration index of the Picard iteration is denoted by r. We obtain the following iterative scheme, which requires the solution of the following *mildly nonlinear* system for the pressure  $p_i^{n+1,r+1}$  at each Picard iteration,

$$\Delta x_{i} \rho_{i}^{n+1} e\left(p_{i}^{n+1,r+1}, \rho_{i}^{n+1}\right) - \Delta t^{2} \left(\frac{h_{i+\frac{1}{2}}^{n+1,r}}{\Delta x_{i+\frac{1}{2}}} \left(p_{i+1}^{n+1,r+1} - p_{i}^{n+1,r+1}\right) - \frac{h_{i-\frac{1}{2}}^{n+1,r}}{\Delta x_{i-\frac{1}{2}}} \left(p_{i}^{n+1,r+1} - p_{i-1}^{n+1,r+1}\right)\right) = b_{i}^{r}, \tag{20}$$

with the right hand side

$$b_{i}^{r} = \Delta x_{i} \left( F(\rho E)_{i}^{n} - \frac{1}{2} \left( (\rho k)_{i-\frac{1}{2}}^{n+1,r} + (\rho k)_{i+\frac{1}{2}}^{n+1,r} \right) \right) - \Delta t \left( h_{i+\frac{1}{2}}^{n+1,r} F(\rho u)_{i+\frac{1}{2}}^{n} - h_{i-\frac{1}{2}}^{n+1,r} F(\rho u)_{i-\frac{1}{2}}^{n} \right). \tag{21}$$

Note that the density  $\rho_i^{n+1}$  is already known from (11), hence in (20) the new pressure is the only unknown. Using a more compact notation, the above system (20) can be written as follows:

$$\mathbf{V}(\mathbf{p}^{n+1,r+1}) + \mathbf{T}^r \, \mathbf{p}^{n+1,r+1} = \mathbf{b}^r, \tag{22}$$

with the vector of the unknowns  $\mathbf{p}^{n+1,r+1} = (p_1^{n+1,r+1}, ..., p_i^{n+1,r+1}, ..., p_{N_x}^{n+1,r+1})$ . The vector  $\mathbf{b}^r$  contains the known right hand side of (20), with terms from the old time level and from the old Picard iteration. Matrix  $\mathbf{T}^r$  is symmetric and at least positive semi-definite and takes into account the linear part of the system, while the nonlinearity is contained in the vector function  $\mathbf{V}(\mathbf{p}^{n+1,r+1}) = \left(\Delta x_1 \rho_1^{n+1} e(p_1^{n+1,r+1}, \rho_1^{n+1}), ..., \Delta x_i \rho_i^{n+1} e(p_i^{n+1,r+1}, \rho_i^{n+1}), ..., \Delta x_{N_x} \rho_{N_x}^{n+1} e(p_{N_x}^{n+1,r+1}, \rho_{N_x}^{n+1})\right)$ , which means a componentwise evaluation of the internal energy density in terms of pressure and density. Note that the density  $\rho_i^{n+1}$  at the new time level is already known from (11).

Note that the time step, the mesh spacings and the enthalpy h are non-negative quantities. Furthermore, the specific internal energy  $e(p,\rho)$  is a non-negative, non-decreasing function of pressure. Thanks to the use of a semi-implicit discretization on a staggered mesh, the matrix  $\mathbf{T}^r$  in system (20) is symmetric and at least positive semi-definite. One therefore can apply the same (nested) Newton-type techniques for its solution as the ones recently introduced in the context of geophysical flows by Casulli et al. in [11, 16, 17, 18, 12] and analyzed by Brugnano and Casulli in [8, 9]. For all implementation details and the convergence proofs of these (nested) Newton techniques applied to mildly nonlinear systems such as the one given by (22), the reader is referred to the above references. These Newton-type techniques successfully apply within semi-implicit numerical methods in the context of incompressible fluids and weakly compressible barotropic fluids in compliant tubes, see [14, 31, 26].

From the new pressure  $p_i^{n+1,r+1}$  the momentum density at the next Picard iteration can be readily obtained as follows:

$$(\rho u)_{i+\frac{1}{2}}^{n+1,r+1} = F(\rho u)_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x_{i+\frac{1}{2}}} \left( p_{i+1}^{n+1,r+1} - p_{i}^{n+1,r+1} \right). \tag{23}$$

As already observed in [17] and as confirmed by our own numerical experiments, a few Picard iterations are sufficient to obtain a satisfactory solution. In the test problems, we stop the Picard process after r=3 iterations. At the end of the last Picard iteration, we set  $p_i^{n+1}=p_i^{n+1,r+1}$ ,  $(\rho u)_{i+\frac{1}{2}}^{n+1}=(\rho u)_{i+\frac{1}{2}}^{n+1,r+1}$ ,  $h_{i+\frac{1}{2}}^{n+1}=h_{i+\frac{1}{2}}^{n+1,r+1}$  and update the total energy density using the conservative formula

$$(\rho E)_{i}^{n+1} = F(\rho E)_{i}^{n} - \frac{\Delta t}{\Delta x_{i}} \left( h_{i+\frac{1}{2}}^{n+1} (\rho u)_{i+\frac{1}{2}}^{n+1} - h_{i-\frac{1}{2}}^{n+1} (\rho u)_{i-\frac{1}{2}}^{n+1} \right). \tag{24}$$

From (11), (13), (24) one can directly observe that the scheme is written in flux form for all conservation equations, hence the proposed method is locally and globally conservative for mass, momentum and energy. Its stability is only restricted by a *mild CFL condition* based on the fluid velocity u, and *not* based on the sound speed c. This makes the method particularly well suited for the discretization of *low Mach number* flows. However, being a locally and globally conservative scheme, it is also able to handle flows with very strong shocks properly, as shown via several numerical test problems in the next section.

For further stabilization at very strong shocks, the terms  $(\rho u)_{i+\frac{1}{2}}^{n+1}$  in (16) should be supplemented with a Rusanov-type dissipation, as proposed in [25], i.e.  $(\rho u)_{i+\frac{1}{2}}^{n+1}$  should be replaced with

$$(\tilde{\rho u})_{i+\frac{1}{2}}^{n+1} = (\rho u)_{i+\frac{1}{2}}^{n+1} - \frac{1}{2} s_{i+\frac{1}{2}} \left( p_{i+1}^{n+1} - p_i^{n+1} \right), \quad \text{with} \quad s_{i+\frac{1}{2}} = \left( |u_{i+\frac{1}{2}}^n| + c_{i+\frac{1}{2}}^n \right) \frac{\partial (\rho e)}{\partial p}, \quad (25)$$

and the sound speed at the interface  $c_{i+\frac{1}{2}}^n$ . The use of (25) instead of  $(\rho u)_{i+\frac{1}{2}}^{n+1}$  in (16) does not change the structure of system (22).

#### 3. Numerical results in 1D

In the following, the new semi-implicit method is applied to a large set of different Riemann problems and three different EOS, namely the ideal gas law, the van der Waals EOS and the Redlich-Kwong EOS. Other general EOS <sup>1</sup> could also be used, but here we limit ourselves to the three aforementioned equations of state.

## 3.1. Ideal gas EOS

Here, the new algorithm is applied to a set of well-known Riemann problems for the compressible Euler equations with ideal gas EOS. The exact Riemann solver has been taken from [53]. All test cases have been run on a uniform grid with mesh spacing  $\Delta x$ , using a Courant-Friedrichs-Lewy number of CFL =  $\frac{\max |u|\Delta t}{\Delta x}$  = 0.5 based on the maximum absolute value of the flow velocity |u|. For all test cases the computational grid consists of 200 cells and the ratio of specific heats is chosen as  $\gamma = 1.4$ . The data for the initial conditions are listed in Table 1. In all test cases we explicitly checked the global conservation error, which was always found to be of the order of machine accuracy.

Contact discontinuity in a low Mach number uniform velocity and pressure flow (RP0). This first test problem RP0 consists of a very strong contact discontinuity moving in a uniform pressure and velocity flow. The density jumps over five orders of magnitude and the local Mach number in this test problem ranges from  $M \approx 2.7 \cdot 10^{-4}$  to  $M \approx 0.085$ . The computational results are depicted in Fig. 1. In the density variable we can see the typical numerical dissipation at the contact wave that is characteristic for a first order method like the one used here. One can also observe that the flow velocity and the fluid pressure remain essentially undisturbed. This is obtained thanks to our particular staggering of the density, which is located at the same position as the fluid velocity. An alternative cell-centered staggering of the density led to significant velocity and pressure oscillations in this test problem and was therefore discarded.

**Sod problem (RP1) and Lax problem (RP2).** These two classical shock tube problems were proposed by Sod and Lax in [50] and [39], respectively, and have become common benchmark problems for Godunov-type finite volume methods. They consist in a right-moving shock wave, a left-moving rarefaction fan and an intermediate contact discontinuity. The numerical solution is compared against the exact solution for both problems in Figs. 2 and 3, respectively. We observe that the shocks are located at the right position and that the post-shock values are also correct.

**Riemann problems RP3 and RP4.** These two difficult Riemann problems are taken from the book of Toro [53] and have been solved with our new semi-implicit algorithm. They are well-suited to test the robustness of a numerical scheme, since they involve very strong shock waves, slowly moving shocks and strong contact waves. For a detailed discussion see [53]. In Figs. 4 and 5 we compare the numerical solution with the exact one for both problems and observe that the method behaves accurately and robustly in all cases. Again, the shocks are in the right position thanks to the use of a locally and globally conservative method.

**Double rarefaction (RP5) and colliding shocks (RP6)**. These two test cases for ideal gas consider a symmetric double rarefaction (RP5), as well as two strong colliding shocks (RP6). In the case of the double rarefaction problem, there are no shock waves involved and the flow can be considered as smooth. However, in Fig. 6 we can note a visible unphysical kink in the density, while the results obtained for pressure and velocity are in good agreement with the exact solution. For the colliding shock problem, which was also discussed in [53], we obtain a numerical solution that is overall in good agreement with the exact one, see Fig. 7, apart from some spurious oscillations at the shock waves, which are well known also for Godunov-type finite volume schemes (see [53]), but which are slightly more pronounced with our scheme. However, instead of the classical *wall-heating* error, which is typically known from Godunov-type finite volume schemes and that manifests itself in a spurious density drop at the origin, our new method produces a local density *increase* at the origin. In other words, our new method is affected by a local *wall-cooling* error. Further investigations about this unexpected behaviour will be necessary in the future.

**Modified Sod problem**. This last test problem (RP7), which is a modification of the original Sod problem, has been proposed by Toro in [53] in order to study the well-known entropy glitch or sonic glitch problem, which typically appears inside supersonic rarefaction waves with most first order Godunov-type schemes that use Riemann solvers with little dissipation. While this phenomenon is only little evident for the Godunov flux based on the exact Riemann

<sup>&</sup>lt;sup>1</sup>According to [18] the conditions on the EOS are that the function  $e(p,\rho)$  is a non-negative, non-decreasing function of pressure for a given density; furthermore, the partial derivative  $\partial e/\partial p$  must be of bounded variation, non-decreasing in the interval  $(-\infty, \ell]$  and non-increasing in  $[r, +\infty]$ , with two constants  $\ell, r \in \mathbb{R}$ .

solver [33] and the Osher flux [44], it is particularly evident for entropy-violating Riemann solvers like the one of Roe [48]. From the numerical results presented in Fig. 8 we can conclude that our new semi-implicit method does not suffer from the sonic glitch problem. To confirm this statement, we also ran the same test problem again on a very fine mesh, still without obtaining any spurious entropy glitch.

Table 1: Initial states left and right for density  $\rho$ , velocity u and pressure p for the Riemann problems solved for the compressible Euler equations with ideal gas EOS. The initial position of the discontinuity  $(x_d)$  and the initial computational domain  $\Omega = [x_L; x_R]$  are also specified. In all cases  $\gamma = 1.4$ .

| RP | $\rho_L$ | $u_L$   | $p_L$    | $\rho_R$ | $u_R$    | $p_R$    | $x_d$ | $x_L$ | $x_R$ |
|----|----------|---------|----------|----------|----------|----------|-------|-------|-------|
| 0  | 1000.0   | 1.0     | $10^{5}$ | 0.01     | 1.0      | $10^{5}$ | -0.25 | -0.5  | 0.5   |
| 1  | 1.0      | 0.0     | 1.0      | 0.125    | 0.0      | 0.1      | 0.0   | -0.5  | 0.5   |
| 2  | 0.445    | 0.698   | 3.528    | 0.5      | 0.0      | 0.571    | 0.0   | -0.5  | 0.5   |
| 3  | 1.0      | 0.0     | 1000     | 1.0      | 0.0      | 0.01     | 0.1   | -0.5  | 0.5   |
| 4  | 5.99924  | 19.5975 | 460.894  | 5.99242  | -6.19633 | 46.095   | 0.0   | -1.0  | 1.0   |
| 5  | 1.0      | -1.0    | 0.4      | 1.0      | +1.0     | 0.4      | 0.0   | -0.5  | 0.5   |
| 6  | 1.0      | +2.0    | 0.1      | 1.0      | -2.0     | 0.1      | 0.0   | -0.5  | 0.5   |
| 7  | 1.0      | 0.75    | 1.0      | 0.125    | 0.0      | 0.1      | -0.1  | -0.5  | 0.5   |

![](_page_7_Figure_3.jpeg)

Figure 1: Exact and numerical solution for the contact wave in very low Mach number ideal gas flow (RP0) at t = 0.5.

![](_page_7_Figure_5.jpeg)

Figure 2: Exact solution and numerical solution for the Sod shock tube problem (RP1) at time t = 0.2 for an ideal gas.

![](_page_8_Figure_0.jpeg)

Figure 3: Exact solution and numerical solution for the Lax shock tube problem (RP2) at time t = 0.14 for an ideal gas.

![](_page_8_Figure_2.jpeg)

Figure 4: Exact solution and numerical solution for the shock tube problem RP3 at time t = 0.012 for an ideal gas.

![](_page_8_Figure_4.jpeg)

Figure 5: Exact solution and numerical solution for the shock tube problem RP4 at time t = 0.035 for an ideal gas.

![](_page_9_Figure_0.jpeg)

Figure 6: Exact solution and numerical solution for the shock tube problem RP5 at time t = 0.15 for an ideal gas.

![](_page_9_Figure_2.jpeg)

Figure 7: Exact solution and numerical solution for the shock tube problem RP6 at time t = 0.8 for an ideal gas.

![](_page_9_Figure_4.jpeg)

Figure 8: Exact solution and numerical solution for the shock tube problem RP7 at time t = 0.2 for an ideal gas.

#### 3.2. Van der Waals EOS

In this section we use the van der Waals equation of state with the following parameters:  $c_v = 1$ , R = 0.4, a = 0.5, b = 0.5. For this purpose, one has to select  $r_1 = r_2 = 0$  in the thermal equation of state (5). The initial conditions for the Riemann problems solved in this section are listed in Table 2. The quasi-exact solution is obtained using the general EOS Riemann solver described in all detail in [27]. For comparison purposes, we also show for each test problem the exact solution that would have been obtained in the case of an ideal gas EOS for the same initial conditions for density, pressure and velocity. Obviously, the initial condition for temperature is in general different for a different EOS. The first Riemann problem VDW1 consists in two non-symmetric colliding shock waves. The numerical results are depicted in Fig. 9. We note overall a good agreement between exact and numerical solution. However, some spurious oscillations are visible at the shock waves, similar to the ones present in RP6 for the ideal gas EOS. The oscillations are even more visible in the temperature T. From Fig. 9 one can also clearly note the importance of the chosen equation of state, since in this case the results for the ideal gas EOS significantly differ from the ones obtained with the van der Waals EOS.

The second Riemann problem (VDW2) is again the Sod shock tube problem, but this time run with the van der Waals EOS. In Fig. 10 we again observe a good agreement between the numerical solution and the exact solution for all flow quantities. Fig. 10 also contains the exact solution for an ideal gas for comparison. In this case, the differences are visible, but not as significant as in the other two cases.

The third Riemann problem (VDW3) uses the same initial data as RP3 before and shows the most dramatic change in the solution due to the EOS. The numerical results are compared with the exact solution in Fig. 11, where an excellent agreement can be observed.

In all the test problems shown here, the Newton-type technique used for the solution of (22) converged in exactly one iteration, since for the van der Waals EOS the specific internal energy e is still a linear function of pressure.

Table 2: Initial states left and right for density  $\rho$ , velocity u and pressure p for the Riemann problems solved for the compressible Euler equations with the van der Waals EOS. The initial position of the discontinuity  $(x_d)$  and the initial computational domain  $\Omega = [x_L; x_R]$  are also specified. In all cases  $c_v = 1$ , R = 0.4 and a = b = 0.5.

| VDW | $\rho_L$ | $u_L$ | $p_L$ | $\rho_R$ | $u_R$ | $p_R$ | $x_d$ | $x_L$ | $x_R$ |
|-----|----------|-------|-------|----------|-------|-------|-------|-------|-------|
| 1   | 1.0      | +1.0  | 2.0   | 1.0      | -1.0  | 1.0   | 0.0   | -0.5  | 0.5   |
| 2   | 1.0      | 0.0   | 1.0   | 0.125    | 0.0   | 0.1   | 0.0   | -0.5  | 0.5   |
| 3   | 1.0      | 0.0   | 1000  | 1.0      | 0.0   | 0.01  | 0.1   | -0.5  | 0.5   |

#### 3.3. Redlich-Kwong EOS

The Redlich-Kwong EOS is a truly nonlinear equation of state, in the sense that the specific internal energy e is a nonlinear function of pressure. This is due to the nonlinear term  $a = a(T) = \alpha/\sqrt{T}$  in the thermal EOS (5). In all the following simulations, we choose  $\alpha = 0.5$ , b = 0.5,  $c_v = 1$  and R = 0.4. The Redlich-Kwong EOS is obtained from the general cubic EOS (5) by setting  $r_1 = 0$  and  $r_2 = -1$ . To compute the function  $e = e(p, \rho)$  one first needs to solve the thermal equation of state (5) for temperature. This can be either done exactly, using the formula for cubic equations of Cardano [10], or simply by a standard Newton method. Here, we choose the latter approach for simplicity. Once the temperature is known from the thermal equation of state, it is inserted into the caloric equation of state (6), thus yielding the desired relation  $e = e(p, \rho)$ .

The initial conditions for the Riemann problems solved in this section are listed in Table 3. In all cases the Newton-type iteration used for the solution of (22) converged to machine precision in at most three to four iterations. The numerical results are shown in Figs. 12 to 15, where in general a good agreement between exact and numerical solution can be appreciated.

![](_page_11_Figure_0.jpeg)

Figure 9: Exact solution and numerical solution for the shock tube problem VDW1 at t = 0.1 for the van der Waals gas. To highlight the importance of the EOS, the exact solution for the ideal gas case (a = b = 0) is also reported, for comparison.

Table 3: Initial states left and right for density  $\rho$ , velocity u and pressure p for the Riemann problems solved for the compressible Euler equations with the van der Waals EOS. The initial position of the discontinuity  $(x_d)$  and the initial computational domain  $\Omega = [x_L; x_R]$  are also specified. In all cases  $c_v = 1$ , R = 0.4 and a = b = 0.5.

| Rk | $\rho_L$ | $u_L$ | $p_L$ | $\rho_R$ | $u_R$ | $p_R$ | $x_d$ | $x_L$ | $x_R$ |
|----|----------|-------|-------|----------|-------|-------|-------|-------|-------|
| 1  | 1.0      | +1.0  | 2.0   | 1.0      | -1.0  | 1.0   | 0.0   | -0.5  | 0.5   |
| 2  | 1.0      | 0.0   | 1.0   | 0.125    | 0.0   | 0.1   | 0.0   | -0.5  | 0.5   |
| 3  | 1.0      | 0.0   | 1000  | 1.0      | 0.0   | 0.01  | 0.1   | -0.5  | 0.5   |
| 4  | 1.0      | 0.0   | 2.0   | 1.5      | 0.0   | 1.0   | 0.0   | -0.5  | 0.5   |

![](_page_12_Figure_0.jpeg)

Figure 10: Exact solution and numerical solution for the shock tube problem VDW2 at t = 0.2 for the van der Waals gas. To highlight the importance of the EOS, the exact solution for the ideal gas case (a = b = 0) is also reported, for comparison.

![](_page_13_Figure_0.jpeg)

Figure 11: Exact solution and numerical solution for the shock tube problem VDW3 at t = 0.008 for the van der Waals gas. To highlight the importance of the EOS, the exact solution for the ideal gas case (a = b = 0) is also reported, for comparison.

![](_page_14_Figure_0.jpeg)

Figure 12: Exact solution and numerical solution for the shock tube problem RK1 at t = 0.1 for the Redlich-Kwong EOS. The exact solution for the van der Waals EOS (a = b = 0.5,  $r_1 = r_2 = 0$ ) is also reported, for comparison.

![](_page_15_Figure_0.jpeg)

Figure 13: Exact solution and numerical solution for the shock tube problem RK2 at t = 0.2 for the Redlich-Kwong EOS. The exact solution for the van der Waals EOS (a = b = 0.5,  $r_1 = r_2 = 0$ ) is also reported, for comparison.

![](_page_16_Figure_0.jpeg)

Figure 14: Exact solution and numerical solution for the shock tube problem RK3 at t = 0.008 for the Redlich-Kwong EOS. The exact solution for the van der Waals EOS (a = b = 0.5,  $r_1 = r_2 = 0$ ) is also reported, for comparison.

![](_page_17_Figure_0.jpeg)

Figure 15: Exact solution and numerical solution for the shock tube problem RK4 at t = 0.1 for the Redlich-Kwong EOS. The exact solution for the van der Waals EOS (a = b = 0.5,  $r_1 = r_2 = 0$ ) is also reported, for comparison.

#### 4. Extension to viscous flows in multiple space dimensions

In multiple space dimensions, the compressible Navier-Stokes equations read

$$\frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho \mathbf{v} \\ \rho E \end{pmatrix} + \nabla \cdot \begin{pmatrix} \rho \mathbf{v} \\ \rho \mathbf{v} \otimes \mathbf{v} + \rho \mathbf{I} \\ \rho k \mathbf{v} + h \rho \mathbf{v} \end{pmatrix} = \nabla \cdot \begin{pmatrix} 0 \\ \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^T \right) - \frac{2}{3} (\mu \nabla \cdot \mathbf{v}) \mathbf{I} \\ \left( \mu \left( \nabla \mathbf{v} + \nabla \mathbf{v}^T \right) - \frac{2}{3} (\mu \nabla \cdot \mathbf{v}) \mathbf{I} \right) \cdot \mathbf{v} + \lambda \nabla T \end{pmatrix}, \tag{26}$$

with the identity matrix  $\mathbf{I}$ , the velocity vector  $\mathbf{v}=(u,v)$ , the temperature T, the kinetic energy density  $\rho k=\frac{1}{2}\rho \mathbf{v}^2$ , the specific enthalpy  $h=e+p/\rho$ , the kinematic viscosity  $\mu$  and the thermal conductivity  $\lambda$ . Again, the system needs to be closed by a thermal equation of state  $p=p(T,\rho)$  and by a caloric equation of state  $e=e(T,\rho)$ . The extension of our new semi-implicit scheme to viscous flows in multiple space dimensions is then straight forward. The discrete pressure  $p_{i,j}^n$  is defined in the control volumes associated with the primary grid, hence  $[x_{i-\frac{1}{2}},x_{i+\frac{1}{2}}]\times [y_{j-\frac{1}{2}},y_{j+\frac{1}{2}}]$ . In the following, we will suppose an equidistant mesh spacing of size  $\Delta x$  in x direction and  $\Delta y$  in y direction. The momentum density  $(\rho u)_{i+\frac{1}{2},j}^n$  and the velocity  $u_{i+\frac{1}{2},j}^n$  in x direction are defined on an edge-based staggered dual control volume given by  $[x_i,x_{i+1}]\times [y_{j-\frac{1}{2}},y_{j+\frac{1}{2}}]$ , while the momentum density  $(\rho v)_{i,j+\frac{1}{2}}^n$  and the velocity  $v_{i,j+\frac{1}{2}}^n$  in y direction are defined in the staggered control volumes  $[x_{i-\frac{1}{2}},x_{i+\frac{1}{2}}]\times [y_j,y_{j+1}]$ . As discussed already for the one-dimensional case before, an important novelty of this paper is that the density is defined at the same staggered locations as the discrete velocity, i.e. on the edges of the primary grid and not at the cell centers of the primary control volumes. As a matter of fact, we also define two density variables, one associated with the x momentum, denoted by  $\rho_{i,j+\frac{1}{2},j}^n$  respectively. The staggered velocities are therefore related to the associated momentum and mass densities simply by  $u_{i+\frac{1}{2},j}^n = (\rho u)_{i+\frac{1}{2},j}^n/\rho_{i+\frac{1}{2},j}^n$  and  $v_{i,j+\frac{1}{2}}^n = (\rho v)_{i,j+\frac{1}{2}}^n/\rho_{i,j+\frac{1}{2}}^n$ , respectively. The respective kinetic energies on the edges are  $(\rho k)_{i+\frac{1}{2},j}^n = (\rho u)_{i+\frac{1}{2},j}^n/\rho_{i+\frac{1}{2},j}^n$  and  $(\rho k)_{i,j+\frac{1}{2}}^n = (\rho v)_{i,j+\frac{1}{2}}^n/\rho_{i,j+\frac{1}{2}}^n$ . The

$$\rho_{i,j}^{n} = \frac{1}{4} \left( \rho_{i+\frac{1}{2},j}^{n} + \rho_{i-\frac{1}{2},j}^{n} + \rho_{i,j+\frac{1}{2}}^{n} + \rho_{i,j-\frac{1}{2}}^{n} \right)$$
(27)

For an efficient evaluation of the explicit transport operators for the mass density, the nonlinear convective and viscous terms in the momentum equation and for the transport of kinetic energy, we first average the staggered quantities onto the primary control volumes. There, we perform an explicit advection step and then the staggered quantities are averaged back onto their respective edge-based dual control volumes. This procedure has already been successfully applied in the context of higher order semi-implicit DG schemes, see [25]. We therefore define at the beginning of each time step the following cell-centered quantities:

$$(\rho_x)_{i,j}^n = \frac{1}{2} \left( \rho_{i+\frac{1}{2},j}^n + \rho_{i-\frac{1}{2},j}^n \right), \qquad (\rho_y)_{i,j}^n = \frac{1}{2} \left( \rho_{i,j+\frac{1}{2}}^n + \rho_{i,j-\frac{1}{2}}^n \right), \tag{28}$$

$$(\rho u)_{i,j}^{n} = \frac{1}{2} \left( (\rho u)_{i+\frac{1}{2},j}^{n} + (\rho u)_{i-\frac{1}{2},j}^{n} \right), \qquad (\rho v)_{i,j}^{n} = \frac{1}{2} \left( (\rho v)_{i,j+\frac{1}{2}}^{n} + (\rho v)_{i,j-\frac{1}{2}}^{n} \right), \tag{29}$$

$$(\rho k)_{i,j}^{n} = \frac{1}{2} \left( (\rho k)_{i+\frac{1}{2},j}^{n} + (\rho k)_{i-\frac{1}{2},j}^{n} \right) + \frac{1}{2} \left( (\rho k)_{i,j+\frac{1}{2}}^{n} + (\rho k)_{i,j-\frac{1}{2}}^{n} \right). \tag{30}$$

Denoting in the following a generic cell-centered quantity of (28)-(30) with  $q_{i,j}^n$ , the transport of that quantity is discretized in an explicit and conservative way as follows:

$$Fq_{i,j}^{n} = q_{i,j}^{n} - \frac{\Delta t}{\Delta x} \left( f_{i+\frac{1}{2},j}^{n} - f_{i-\frac{1}{2},j}^{n} \right) - \frac{\Delta t}{\Delta y} \left( g_{i,j+\frac{1}{2}}^{n} - g_{i,j-\frac{1}{2}}^{n} \right), \tag{31}$$

with the numerical fluxes

$$f_{i+\frac{1}{2},j}^{n} = \frac{1}{2}u_{i+\frac{1}{2},j}^{n}\left(q_{i+1,j}^{n} + q_{i,j}^{n}\right) - \frac{1}{2}\left|u_{i+\frac{1}{2},j}^{n}\right|\left(q_{i+1,j}^{n} - q_{i,j}^{n}\right), \qquad g_{i,j+\frac{1}{2}}^{n} = \frac{1}{2}v_{i,j+\frac{1}{2}}^{n}\left(q_{i,j+1}^{n} + q_{i,j}^{n}\right) - \frac{1}{2}\left|v_{i,j+\frac{1}{2}}^{n}\right|\left(q_{i,j+1}^{n} - q_{i,j}^{n}\right).$$

The operators on the original edge-based staggered grid can then be obtained again from averaging as

$$Fq_{i+\frac{1}{2},j}^{n} = \frac{1}{2} \left( Fq_{i+1,j}^{n} + Fq_{i,j}^{n} \right), \qquad Fq_{i,j+\frac{1}{2}}^{n} = \frac{1}{2} \left( Fq_{i,j+1}^{n} + Fq_{i,j}^{n} \right). \tag{32}$$

Note that the entire sequence of averaging to the primary grid (28)-(30), conservative update (31) and averaging back to the staggered grid (32) is conservative, since the two averaging operators are conservative interpolations. With (31)-(32), the discretization of the density equation on the staggered grid becomes

$$\rho_{i+\frac{1}{2},j}^{n+1} = F(\rho_x)_{i+\frac{1}{2},j}^n, \qquad \rho_{i,j+\frac{1}{2}}^{n+1} = F(\rho_y)_{i,j+\frac{1}{2}}^n.$$
(33)

With (33) and (27) the new density at the cell center  $\rho_{i,j}^{n+1}$  can be computed. The discrete momentum equations read

$$(\rho u)_{i+\frac{1}{2},j}^{n+1} = F(\rho u)_{i+\frac{1}{2},j}^{n} - \frac{\Delta t}{\Delta x} \left( p_{i+1,j}^{n+1} - p_{i,j}^{n+1} \right), \qquad (\rho v)_{i,j+\frac{1}{2}}^{n+1} = F(\rho v)_{i,j+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta v} \left( p_{i,j+1}^{n+1} - p_{i,j}^{n+1} \right), \tag{34}$$

where pressure is taken *implicitly*, while all nonlinear convective and viscous terms are discretized *explicitly* via the operators  $F(\rho u)_{i+\frac{1}{2},j}^n$  and  $F(\rho v)_{i,j+\frac{1}{2}}^n$ , respectively. The preliminary form of the discrete total energy equation is given by:

$$\rho_{i,j}^{n+1} e\left(p_{i,j}^{n+1}, \rho_{i,j}^{n+1}\right) + \frac{1}{2}\left(\left(\tilde{\rho k}\right)_{i-\frac{1}{2},j}^{n+1} + \left(\tilde{\rho k}\right)_{i+\frac{1}{2},j}^{n+1} + \left(\tilde{\rho k}\right)_{i,j-\frac{1}{2}}^{n+1} + \left(\tilde{\rho k}\right)_{i,j-\frac{1}{2}}^{n+1} + \left(\tilde{\rho k}\right)_{i,j+\frac{1}{2}}^{n+1}\right) = F(\rho E)_{i,j}^{n} - \frac{\Delta t}{\Delta x}\left(\tilde{h}_{i+\frac{1}{2},j}^{n+1}(\rho u)_{i+\frac{1}{2},j}^{n+1} - \tilde{h}_{i-\frac{1}{2},j}^{n+1}(\rho u)_{i-\frac{1}{2},j}^{n+1}\right) - \frac{\Delta t}{\Delta y}\left(\tilde{h}_{i,j+\frac{1}{2}}^{n+1}(\rho v)_{i,j+\frac{1}{2}}^{n+1} - \tilde{h}_{i,j-\frac{1}{2}}^{n+1}(\rho v)_{i,j-\frac{1}{2}}^{n+1}\right).$$

$$(35)$$

Here, we have used the abbreviation  $F(\rho E)_{i,j}^n = (\rho E)_{i,j}^n - (\rho k)_{i,j}^n + F(\rho k)_{i,j}^n$ . Inserting the discrete momentum equation (34) into the discrete energy equation (35) and making tilde symbols explicit via a simple Picard iteration yields the following discrete wave equation for the unknown pressure:

$$\rho_{i,j}^{n+1}e\left(p_{i,j}^{n+1,r+1},\rho_{i,j}^{n+1}\right) - \frac{\Delta t^{2}}{\Delta x^{2}}\left(h_{i+\frac{1}{2},j}^{n+1,r}\left(p_{i+1,j}^{n+1,r+1} - p_{i,j}^{n+1,r+1}\right) - h_{i-\frac{1}{2},j}^{n+1,r}\left(p_{i,j}^{n+1,r+1} - p_{i-1,j}^{n+1,r+1}\right)\right) - \frac{\Delta t^{2}}{\Delta y^{2}}\left(h_{i,j+\frac{1}{2}}^{n+1,r}\left(p_{i,j+1}^{n+1,r+1} - p_{i,j}^{n+1,r+1}\right) - h_{i,j-\frac{1}{2}}^{n+1,r}\left(p_{i,j}^{n+1,r+1} - p_{i,j-1}^{n+1,r+1}\right)\right) = b_{i,j}^{r},$$
(36)

with the known right hand side  $b_i^r$ ; given by terms discretized at the old time  $t^n$  or at the previous Picard iteration r as

$$b_{i,j}^{r} = F(\rho E)_{i,j}^{n} - \frac{1}{2} \left( (\rho k)_{i-\frac{1}{2},j}^{n+1,r} + (\rho k)_{i+\frac{1}{2},j}^{n+1,r} \right) - \frac{1}{2} \left( (\rho k)_{i,j-\frac{1}{2}}^{n+1,r} + (\rho k)_{i,j+\frac{1}{2}}^{n+1,r} \right) - \frac{\Delta t}{\Delta x} \left( h_{i+\frac{1}{2},j}^{n+1,r} F(\rho u)_{i+\frac{1}{2},j}^{n} - h_{i-\frac{1}{2},j}^{n+1,r} F(\rho u)_{i-\frac{1}{2},j}^{n} \right) - \frac{\Delta t}{\Delta y} \left( h_{i,j+\frac{1}{2}}^{n+1,r} F(\rho v)_{i,j+\frac{1}{2}}^{n} - h_{i,j-\frac{1}{2}}^{n+1,r} F(\rho v)_{i,j-\frac{1}{2}}^{n} \right).$$
(37)

The system for the pressure (36) is again a mildly nonlinear system with a linear part that is symmetric and as least positive semi-definite. Hence, with the usual assumptions on the nonlinearity detailed in [18], it can be again efficiently solved with the nested Newton method of Casulli and Zanolli [17, 18]. Once the new pressure is known, the momentum can be readily updated from (34). At the end of the Picard iterations, the total energy is updated as

$$(\rho E)_{i,j}^{n+1} = F(\rho E)_{i,j}^{n} - \frac{\Delta t}{\Delta x} \left( h_{i+\frac{1}{2},j}^{n+1} (\rho u)_{i+\frac{1}{2},j}^{n+1} - h_{i-\frac{1}{2},j}^{n+1} (\rho u)_{i-\frac{1}{2},j}^{n+1} \right) - \frac{\Delta t}{\Delta y} \left( h_{i,j+\frac{1}{2}}^{n+1} (\rho v)_{i,j+\frac{1}{2}}^{n+1} - h_{i,j-\frac{1}{2}}^{n+1} (\rho v)_{i,j-\frac{1}{2}}^{n+1} \right). \tag{38}$$

In the present paper, the viscous terms are simply discretized in an explicit manner by adding the corresponding discrete form of the right hand side of (26) to the operators  $F(\rho u)_{i,j}^n$ ,  $F(\rho v)_{i,j}^n$  and  $F(\rho k)_{i,j}^n$ . For that purpose the discrete velocity gradients on the control volume boundaries are computed according to

$$\nabla \mathbf{v}_{i+\frac{1}{2},j}^{n} = \frac{1}{2} \left( \nabla \mathbf{v}_{i+\frac{1}{2},j+\frac{1}{2}}^{n} + \nabla \mathbf{v}_{i+\frac{1}{2},j-\frac{1}{2}}^{n} \right), \qquad \nabla \mathbf{v}_{i,j+\frac{1}{2}}^{n} = \frac{1}{2} \left( \nabla \mathbf{v}_{i+\frac{1}{2},j+\frac{1}{2}}^{n} + \nabla \mathbf{v}_{i-\frac{1}{2},j+\frac{1}{2}}^{n} \right),$$
(39)

with the corner gradient defined as

$$\nabla \mathbf{v}_{i+\frac{1}{2},j+\frac{1}{2}}^{n} = \frac{1}{2} \begin{pmatrix} \frac{\mathbf{v}_{i+1,j+1}^{n} - \mathbf{v}_{i,j+1}^{n}}{\Delta x} + \frac{\mathbf{v}_{i+1,j}^{n} - \mathbf{v}_{i,j}^{n}}{\Delta x} \\ \frac{\mathbf{v}_{i+1,j+1}^{n} - \mathbf{v}_{i+1,j}^{n}}{\Delta y} + \frac{\mathbf{v}_{i,j+1}^{n} - \mathbf{v}_{i,j}^{n}}{\Delta y} \end{pmatrix}. \tag{40}$$

The temperature gradients are computed in the same way.

#### 5. Numerical results in 2D

In all the following numerical test problems, the ideal gas equation of state is used, in order to make the results comparable with existing data in the literature.

#### 5.1. Circular explosion problem

In this subsection we solve an inviscid cylindrical explosion problem in the domain  $\Omega = [-1, 1] \times [-1, 1]$ , whose initial condition is given by

$$(\rho, u, v, p) = \begin{cases} (1, 0, 0, 1) & \text{if } ||\mathbf{x}|| \le 0.5, \\ (0.125, 0, 0, 0.1) & \text{if } ||\mathbf{x}|| > 0.5. \end{cases}$$

$$(41)$$

We run our new semi-implicit scheme on a computational grid composed of  $1000 \times 1000$  elements until a final time of t = 0.25. Due to the angular symmetry of the problem, a reference solution can be computed according to [53] by solving an equivalent one-dimensional PDE in radial direction with geometrical source terms. For the computation of the 1D reference solution we use a second order TVD scheme with the Osher-type flux [28] on a very fine radial grid with 10000 cells. The numerical solution obtained with our two-dimensional semi-implicit scheme is compared against the reference solution in Figure 16. Overall, we observe that all waves are properly captured, i.e. the cylindrical shock wave, the cylindrical contact discontinuity and the inward-moving cylindrical rarefaction wave. In particular we find no spurious pressure oscillations at the contact wave. For this test problem, we have also computed the average CPU time necessary for one time step of one element by dividing the total CPU time of the simulation by the number of elements and time steps. On a single CPU core of a workstation equipped with an Intel Core i7-2600 CPU with 3.4 GHz clock speed and 12 GB of RAM we find an average CPU time per element and time step of only 1  $\mu$ s. The order of magnitude of this value is comparable to a standard explicit Godunov-type scheme applied to the same problem and thus clearly highlights the computational efficiency of our new semi-implicit formulation, in particular if we consider the fact that our method needs the solution of a sequence of linear systems for the pressure.

## 5.2. Two-dimensional Riemann problems

Here we consider a set of two-dimensional Riemann problems, whose initial conditions are given by

$$(\rho, u, v, p) = \begin{cases} (\rho_1, u_1, v_1, p_1) & \text{if} & x > 0 \land y > 0, \\ (\rho_2, u_2, v_2, p_2) & \text{if} & x \le 0 \land y > 0, \\ (\rho_3, u_3, v_3, p_3) & \text{if} & x \le 0 \land y \le 0, \\ (\rho_4, u_4, v_4, p_4) & \text{if} & x > 0 \land y \le 0. \end{cases}$$

$$(42)$$

A large set of such two–dimensional Riemann problems has been presented in great detail in the papers by Schulz-Rinne [49] and Kurganov and Tadmor [38]. These multi-dimensional Riemann problems can also be used as a building block of cell-centered Godunov-type finite volume schemes. For more details on so-called genuinely multi-dimensional Riemann solvers, see [2, 1, 4, 6, 3, 5]. Also for these test cases, the fluid is considered inviscid, hence  $\mu = \lambda = 0$ . The initial conditions for the four configurations solved in this article are listed in Table 4. The computational domain is given by  $\Omega = [-0.5; 0.5] \times [-0.5; 0.5]$  and is discretized with a uniform Cartesian grid composed of  $1000 \times 1000$  cells. The computational results obtained with our new semi-implicit scheme are depicted for all four configurations in Figure 17. We can note a reasonable qualitative agreement with the second and third order results published in [38]. However, we clearly observe that despite the very fine mesh used here, our numerical results show more dissipation, since our method is only low order accurate. Also in this set of test problems, the average CPU time was only about 1  $\mu$ s per element and time step.

![](_page_21_Figure_0.jpeg)

Figure 16: Reference solution and numerical solution obtained with the new semi-implicit method for the circular explosion problem at time *<sup>t</sup>* <sup>=</sup> <sup>0</sup>.25.

![](_page_22_Figure_0.jpeg)

Figure 17: Numerical solution obtained with the new semi-implicit method for configurations C4, C7, C8 and C16 of the two-dimensional Riemann problems discussed in [49, 38].

Table 4: Initial conditions for the two-dimensional Riemann problems.

| Problem C4 (Configuration 4 in [38])   |        |            |           |         |            |            |         |      |  |  |
|----------------------------------------|--------|------------|-----------|---------|------------|------------|---------|------|--|--|
|                                        |        | <i>x</i> ≤ | 0         |         | x > 0      |            |         |      |  |  |
|                                        | ρ      | и          | v         | p       | ρ          | и          | v       | p    |  |  |
| y > 0                                  | 0.5065 | 0.8939     | 0.0       | 0.35    | 1.1        | 0.0        | 0.0     | 1.1  |  |  |
| $y \le 0$                              | 1.1    | 0.8939     | 0.8939    | 1.1     | 0.5065     | 0.0        | 0.8939  | 0.35 |  |  |
| Problem C7 (Configuration 7 in [38])   |        |            |           |         |            |            |         |      |  |  |
|                                        |        | <i>x</i> ≤ | 0         |         |            | <i>x</i> > | 0       |      |  |  |
|                                        | ρ      | и          | v         | p       | ρ          | и          | v       | p    |  |  |
| y > 0                                  | 1.0    | 0.1        | 0.1       | 1.0     | 0.5197     | -0.6259    | 0.1     | 0.4  |  |  |
| $y \le 0$                              | 0.8    | 0.1        | 0.1       | 0.4     | 0.5197     | 0.1        | -0.6259 | 0.4  |  |  |
|                                        |        | Prob       | lem C8 (C | onfigur | ation 8 in | [38])      |         |      |  |  |
|                                        |        | <i>x</i> ≤ | 0         |         | x > 0      |            |         |      |  |  |
|                                        | ρ      | и          | v         | p       | ρ          | и          | v       | p    |  |  |
| y > 0                                  | 1.0    | -0.6259    | 0.1       | 1.0     | 0.5197     | 0.1        | 0.1     | 0.4  |  |  |
| $y \le 0$                              | 0.8    | 0.1        | 0.1       | 1.0     | 1.0        | 0.1        | -0.6259 | 1.0  |  |  |
| Problem C16 (Configuration 16 in [38]) |        |            |           |         |            |            |         |      |  |  |
|                                        |        | <i>x</i> ≤ | 0         |         | x > 0      |            |         |      |  |  |
|                                        | ρ      | и          | v         | p       | ρ          | и          | v       | p    |  |  |
| y > 0                                  | 1.0222 | -0.6179    | 0.1       | 1.0     | 0.5313     | 0.1        | 0.1     | 0.4  |  |  |
| $y \le 0$                              | 0.8    | 0.1        | 0.1       | 1.0     | 1.0        | 0.1        | 0.8276  | 1.0  |  |  |

#### 5.3. Lid-driven cavity

In this last example, we consider the full compressible Navier-Stokes equations (26) at low Mach number, where the proposed semi-implicit approach is particularly useful, since the time step is only restricted by a CFL condition based on the flow velocity v, and not by the sound speed  $c = \sqrt{\gamma p/\rho}$ . A further time step restriction comes from the explicit discretization of the viscous terms, which, however, is not so stringent for low viscosities. The computational domain is the box  $\Omega = [-0.5, 0.5] \times [-0.5, 0.5]$ , which is initialized with a density of  $\rho = 1$ , a velocity of  $\mathbf{v} = 0$  and the pressure is set to  $p = 10^5$ , which corresponds to standard atmospheric reference conditions in SI units. The dynamic viscosity is chosen as  $\mu = 10^{-2}$ , while heat conduction is neglected, i.e.  $\lambda = 0$ . The flow is driven by the upper boundary, whose velocity is set to  $\mathbf{v} = (1,0)$ . On the other three boundaries, a no-slip wall boundary condition  $\mathbf{v} = 0$ is imposed. This problem is also well-known in the literature as the lid-driven cavity problem, for which reference data have been provided in the paper of Ghia et al. [32]. We run our semi-implicit scheme for this problem on a mesh composed of  $150 \times 150$  elements until the final time of t = 50. The reference Mach number of this test case with respect to the speed of the lid  $u_0 = 1$  is  $M = u_0/c = 2.67 \cdot 10^{-3}$ . Even with an explicit discretization of the viscous terms in (26), the time step is still about 125 times larger compared to a fully explicit scheme, where the time step is limited by a CFL condition based on the sound speed. Our computational results are shown in Fig. 18, where also a comparison with the reference solution of Ghia et al. [32] is presented. We note a reasonable agreement between the two solutions. We essentially attribute the still visible discrepancy between our results and the ones of Ghia et al. to the fact that here we solve the compressible Navier-Stokes equations, while the results shown in [32] have been obtained for the fully incompressible case. Finally, we would like to point out that very recently, excellent numerical results have been obtained in [52] for this test problem using a new family of high order semi-implicit DG schemes on extremely coarse staggered unstructured triangular grids.

# 6. Conclusions

In this paper we have presented a new efficient semi-implicit method for the simulation of compressible flows with general equation of state. Our particular discretization on a staggered mesh allows us to reduce the problem

![](_page_24_Figure_0.jpeg)

![](_page_24_Figure_1.jpeg)

Figure 18: Computational results for the lid-driven cavity problem at time t = 50 and a Reynolds number of Re = 100. Left: horizontal velocity contours and streamlines obtained with the new semi-implicit method. Right: comparison of the semi-implicit scheme (solid lines) with the solution obtained by Ghia et al. [32] (squares).

to the solution of a mildly nonlinear system for the fluid pressure, which can be efficiently solved by a Newton-type technique. The linear part of the mildly nonlinear system is given by a symmetric and positive semi-definite M-matrix, while the nonlinearity resides only on the diagonal and is contained in the equation of state. The EOS must provide the specific energy as a function of fluid pressure and density and must satisfy the following conditions: for a given density e must be a non-negative non-decreasing function of p and its partial derivative w.r.t. p must be a function of bounded variation, non-decreasing in the interval  $(-\infty, \ell]$  and non-increasing in  $[r, +\infty]$ , with two constants  $\ell, r \in \mathbb{R}$ . The unknown kinetic energy at the new time level as well as the specific enthalpies that appear in the right hand side and in the linear part of the mildly nonlinear system are updated easily with a simple Picard process, following the suggestion of [17]. Once the pressure is known at the new time level, the momentum and total energy density can be readily obtained via a conservative update formula.

The time step of our scheme is only restricted by accuracy requirements, by the fluid velocity and eventually by the presence of the viscous terms, but not by the sound speed. Therefore, the proposed conservative semi-implicit algorithm allows to use large time steps and remains efficient also in the case of very low Mach number flows. The method has been carefully validated in 1D for the ideal gas EOS, for the van der Waals EOS and for the Redlich-Kwong EOS on a large set of test problems, for each of which an exact or quasi-exact reference solution has been provided.

In the second part of this paper, we have also shown how our new semi-implicit scheme can be easily extended to the multi-dimensional case and to the compressible Navier-Stokes equations.

Future work will consist in an extension of the present approach to unstructured meshes in multiple space dimensions and to higher order of accuracy, following the ideas outlined in [25, 51, 52] for staggered semi-implicit discontinuous Galerkin finite element schemes. We also plan an extension of this family of semi-implicit schemes to the viscous and resistive MHD equations.

#### **Dedication**

The new numerical method introduced in this paper is dedicated to **Claus-Dieter Munz** at the occasion of his 60th birthday and in honor of his scientific contributions to the field of numerical methods for computational fluid dynamics. His work covers the broad range from low Mach number and nearly incompressible flows to highly compressible flows with strong shock waves.

# Acknowledgements

The research presented in this paper was funded by the European Research Council (ERC) under the European Union's Seventh Framework Programme (FP7/2007-2013) within the research project *STiMulUs*, ERC Grant agreement no. 278267.

## References

- [1] D.S. Balsara. Multidimensional HLLE Riemann solver: Application to Euler and magnetohydrodynamic flows. *Journal of Computational Physics*, 229:1970–1993, 2010.
- [2] D.S. Balsara. A two-dimensional HLLC Riemann solver for conservation laws: Application to Euler and magnetohydrodynamic flows. *Journal of Computational Physics*, 231:7476–7503, 2012.
- [3] D.S. Balsara. Multidimensional Riemann Problem with Self-Similar Internal Structure Part I Application to Hyperbolic Conservation Laws on Structured Meshes. *Journal of Computational Physics*, 277:163–200, 2014.
- [4] D.S. Balsara. Three dimensional HLL Riemann solver for conservation laws on structured meshes; Application to Euler and magnetohydrodynamic flows. *Journal of Computational Physics*, 295:1–23, 2015.
- [5] D.S. Balsara and M. Dumbser. Multidimensional Riemann Problem with Self-Similar Internal Structure Part II Application to Hyperbolic Conservation Laws on Unstructured Meshes. *Journal of Computational Physics*, 287:269–292, 2015.
- [6] D.S. Balsara, M. Dumbser, and R. Abgrall. Multidimensional HLLC Riemann Solver for Unstructured Meshes With Application to Euler and MHD Flows. *Journal of Computational Physics*, 261:172–208, 2014.
- [7] J.B. Bell, P. Colella, and H.M. Glaz. A second–order projection method for the incompressible Navier–Stokes equations. *Journal of Computational Physics*, 85:257–283, 1989.
- [8] L. Brugnano and V. Casulli. Iterative solution of piecewise linear systems. *SIAM Journal on Scientific Computing*, 30:463–472, 2007.
- [9] L. Brugnano and V. Casulli. Iterative solution of piecewise linear systems and applications to flows in porous media. *SIAM Journal on Scientific Computing*, 31:1858–1873, 2009.
- [10] G. Cardano. *Artis magnae sive de regulis algebraicis liber unus*. Petreius, Nurnberg, Germany, 1545. ¨
- [11] V. Casulli. A high-resolution wetting and drying algorithm for free-surface hydrodynamics. *International Journal for Numerical Methods in Fluids*, 60:391–408, 2009.
- [12] V. Casulli. A semi–implicit numerical method for the free–surface Navier–Stokes equations. *International Journal for Numerical Methods in Fluids*, 74:605–622, 2014.
- [13] V. Casulli and E. Cattani. Stability, accuracy and efficiency of a semi-implicit method for three-dimensional shallow water flow. *Computers* & *Mathematics with Applications*, 27:99–112, 1994.
- [14] V. Casulli, M. Dumbser, and E. F. Toro. Semi-implicit numerical modeling of axially symmetric flows in compliant arterial systems. *International Journal for Numerical Methods in Biomedical Engineering*, 28:257–272, 2012.
- [15] V. Casulli and D. Greenspan. Pressure method for the numerical solution of transient, compressible fluid flows. *International Journal for Numerical Methods in Fluids*, 4(11):1001–1012, 1984.
- [16] V. Casulli and G. S. Stelling. Semi-implicit subgrid modelling of three-dimensional free-surface flows. *International Journal for Numerical Methods in Fluids*, 67:441–449, 2011.
- [17] V. Casulli and P. Zanolli. A nested Newton–type algorithm for finite volume methods solving Richards' equation in mixed form. *SIAM Journal on Scientific Computing*, 32:2255–2273, 2009.
- [18] V. Casulli and P. Zanolli. Iterative solutions of mildly nonlinear systems. *Journal of Computational and Applied Mathematics*, 236:3937– 3947, 2012.
- [19] A.J. Chorin. A numerical method for solving incompressible viscous flow problems. *Journal of Computational Physics*, 2:12–26, 1967.
- [20] A.J. Chorin. Numerical solution of the Navier–Stokes equations. *Mathematics of Computation*, 23:341–354, 1968.
- [21] F. Cordier, P. Degond, and A. Kumbaro. An Asymptotic-Preserving all-speed scheme for the Euler and Navier-Stokes equations. *Journal of Computational Physics*, 231:5685–5704, 2012.
- [22] V. Dolejsi. Semi-implicit interior penalty discontinuous galerkin methods for viscous compressible flows. *Communications in Computational Physics*, 4:231–274, 2008.
- [23] V. Dolejsi and M. Feistauer. A semi-implicit discontinuous galerkin finite element method for the numerical solution of inviscid compressible flow. *Journal of Computational Physics*, 198:727–746, 2004.
- [24] V. Dolejsi, M. Feistauer, and J. Hozman. Analysis of semi-implicit DGFEM for nonlinear convection-diffusion problems on nonconforming meshes. *Computer Methods in Applied Mechanics and Engineering*, 196:2813–2827, 2007.
- [25] M. Dumbser and V. Casulli. A staggered semi-implicit spectral discontinuous galerkin scheme for the shallow water equations. *Applied Mathematics and Computation*, 219(15):8057–8077, 2013.
- [26] M. Dumbser, U. Iben, and M. Ioriatti. An efficient semi-implicit finite volume method for axially symmetric compressible flows in compliant tubes. *Applied Numerical Mathematics*, 89:24–44, 2015.
- [27] M. Dumbser, U. Iben, and C.D. Munz. Efficient implementation of high order unstructured WENO schemes for cavitating flows. *Computers and Fluids*, 86:141–168, 2013.
- [28] M. Dumbser and E. F. Toro. On universal Osher–type schemes for general nonlinear hyperbolic conservation laws. *Communications in Computational Physics*, 10:635–671, 2011.
- [29] B. Einfeldt, C. D. Munz, P. L. Roe, and B. Sjogreen. On Godunov-type methods near low densities. ¨ *Journal of Computational Physics*, 92:273–295, 1991.
- [30] W. Wagner et al. The IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam. *Journal of Engineering for Gas Turbines and Power*, 122:150–182, 2000.

- [31] F. Fambri, M. Dumbser, and V. Casulli. An Efficient Semi-Implicit Method for Three-Dimensional Non-Hydrostatic Flows in Compliant Arterial Vessels. *International Journal for Numerical Methods in Biomedical Engineering*, 30:1170–1198, 2014.
- [32] U. Ghia, K. N. Ghia, and C. T. Shin. High-re solutions for incompressible flow using navier-stokes equations and multigrid method. *Journal of Computational Physics*, 48:387–411, 1982.
- [33] S. K. Godunov. Finite difference methods for the computation of discontinuous solutions of the equations of fluid dynamics. *Mat. Sb.*, 47:271–306, 1959.
- [34] A. Harten, P. D. Lax, and B. van Leer. On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Review*, 25(1):35–61, 1983.
- [35] C. W. Hirt and B. D. Nichols. Volume of fluid (VOF) method for dynamics of free boundaries. *Journal of Computational Physics*, 39:201–225, 1981.
- [36] S. Klainermann and A. Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluid. *Communications on Pure and Applied Mathematics*, 34:481–524, 1981.
- [37] S. Klainermann and A. Majda. Compressible and incompressible fluids. *Communications on Pure and Applied Mathematics*, 35:629–651, 1982.
- [38] A. Kurganov and E. Tadmor. Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers. *Numer. Methods Partial Di*ff*erential Equations*, 18:584–608, 2002.
- [39] P. D. Lax and B. Wendroff. Systems of conservation laws. *Communications in Pure and Applied Mathematics*, 13:217–237, 1960.
- [40] R. J. LeVeque. *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press, 2002.
- [41] C. D. Munz. On Godunov–type schemes for Lagrangian gas dynamics. *SIAM Journal on Numerical Analysis*, 31:17–42, 1994.
- [42] C.D. Munz, M. Dumbser, and S. Roller. Linearized acoustic perturbation equations for low Mach number flow with variable density and temperature. *Journal of Computational Physics*, 224:352–364, 2007.
- [43] C.D. Munz, R. Klein, S. Roller, and K.J. Geratz. The extension of incompressible flow solvers to the weakly compressible regime. *Computers and Fluids*, 2003.
- [44] S. Osher and F. Solomon. Upwind difference schemes for hyperbolic conservation laws. *Math. Comput.*, 38:339–374, 1982.
- [45] J.H. Park and C.D. Munz. Multiple pressure variables methods for fluid flow at all mach numbers. *International Journal for Numerical Methods in Fluids*, 49:905–931, 2005.
- [46] D.Y. Peng and D.P. Robinson. A New Two-Constant Equation of State. *Industrial and Engineering Chemistry Fundamentals*, 15:59–64, 1976.
- [47] O. Redlich and J.N.S. Kwong. On the Thermodynamics of Solutions. V. An Equation of State. Fugacities of Gaseous Solutions. *Chemical Reviews*, 44:233–244, 1949.
- [48] P. L. Roe. Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43:357–372, 1981.
- [49] C. W. Schulz-Rinne. Classification of the Riemann problem for two-dimensional gas dynamics. *SIAM J. Math. Anal.*, 24:76–88, 1993.
- [50] G.A. Sod. A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. *Journal of Computational Physics*, 27:1–31, 1978.
- [51] M. Tavelli and M. Dumbser. A high order semi–implicit discontinuous Galerkin method for the two dimensional shallow water equations on staggered unstructured meshes. *Applied Mathematics and Computation*, 234:623–644, 2014.
- [52] M. Tavelli and M. Dumbser. A staggered semi-implicit discontinuous Galerkin method for the two dimensional incompressible Navier-Stokes equations. *Applied Mathematics and Computation*, 248:70–92, 2014.
- [53] E. F. Toro. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer, third edition, 2009.
- [54] E. F. Toro, M. Spruce, and W. Speares. Restoration of the contact surface in the Harten-Lax-van Leer Riemann solver. *Journal of Shock Waves*, 4:25–34, 1994.
- [55] E.F. Toro and M.E. Vazquez-Cend ´ on. Flux splitting schemes for the Euler equations. ´ *Computers and Fluids*, 70:1–12, 2012.
- [56] Patankar V. S. *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing Corporation, 1980.
- [57] J.D. van der Waals. *Over de continuiteit van den gasen vloeistoftoestand*. Sijthoff, Leiden, 1873.
- [58] J. van Kan. A second-order accurate pressure correction method for viscous incompressible flow. *SIAM Journal on Scientific and Statistical Computing*, 7:870–891, 1986.
- [59] J. Vidal. *Thermodynamics: Applications in Chemical Engineering and the Petroleum Industry*. Editions Technip, 2001.
- [60] W. Wagner and A. Pruss. The IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use. *Journal of Physical and Chemical Reference Data*, 31:387–536, 2002.