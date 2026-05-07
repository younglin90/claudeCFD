# Second order Implicit-Explicit Total Variation Diminishing schemes for the Euler system in the low Mach regime

Giacomo Dimarco<sup>a</sup> , Raphael Loub ¨ ere ` b , Victor Michel-Dansac<sup>c</sup> , Marie-Hel´ ene Vignal ` c

*<sup>a</sup>Department of Mathematics and Computer Science, University of Ferrara, Ferrara, Italy <sup>b</sup>CNRS and Institut de Math´ematiques de Bordeaux (IMB) Universit´e de Bordeaux, France c Institut de Math´ematiques de Toulouse (IMT), Universit´e P. Sabatier, Toulouse*

### Abstract

In this work, we consider the development of implicit explicit total variation diminishing (TVD) methods (also termed SSP: strong stability preserving) for the compressible isentropic Euler system in the low Mach number regime. The scheme proposed is asymptotically stable with a CFL condition independent from the Mach number and it degenerates in the low Mach number regime to a consistent discretization of the incompressible system. Since, it has been proved that implicit schemes of order higher than one cannot be TVD (SSP) [\[29\]](#page-22-0), we construct a new paradigm of implicit time integrators by coupling first order in time schemes with second order ones in the same spirit as highly accurate shock capturing TVD methods in space. For this particular class of schemes, the TVD property is first proved on a linear model advection equation and then extended to the isentropic Euler case. The result is a method which interpolates from the first to the second order both in space and time, which preserves the monotonicity of the solution, highly accurate for all choices of the Mach number and with a time step only restricted by the non stiff part of the system. In the last part, we show thanks to one and two dimensional test cases that the method indeed possesses the claimed properties.

*Keywords:* Asymptotic Preserving, IMEX schemes, SSP-TVD property, Low Mach number limit, High-order schemes, Hyperbolic conservation laws.

### 1. Introduction

The analysis [\[40,](#page-22-1) [41,](#page-22-2) [56,](#page-22-3) [2,](#page-20-0) [44,](#page-22-4) [1\]](#page-20-1) and the development of numerical methods [\[37,](#page-22-5) [33,](#page-22-6) [54,](#page-22-7) [62,](#page-23-0) [42,](#page-22-8) [13,](#page-21-0) [31,](#page-22-9) [51,](#page-22-10) [47,](#page-22-11) [34,](#page-22-12) [30,](#page-22-13) [50,](#page-22-14) [48,](#page-22-15) [16,](#page-21-1) [19,](#page-21-2) [17,](#page-21-3) [15,](#page-21-4) [32,](#page-22-16) [10,](#page-20-2) [28,](#page-22-17) [39,](#page-22-18) [20,](#page-21-5) [11,](#page-20-3) [21\]](#page-21-6) for the passage from compressible to incompressible gas dynamics has been and is still a very active field of research. The compressible Euler equations which describe conservation of density, momentum and energy in a fluid flow become stiff when the Mach number tends to zero. This implies a fluid flow almost at rest. In this case, the pressure waves move at very large speed compared to the average speed of the gas. Thus, a standard model approximation consists in replacing the density conservation equation by a constraint on the velocity divergence, set consequently equal to zero. In addition, the momentum equation could be replaced by an elliptic equation for the pressure. We refer to that situation to as the incompressible Euler model which is used to describe many different flow conditions. However, there are situations in which the Mach number may be small in some part of the domain and large in others or may strongly change in time. In these cases, one should deal with the coupling of incompressible and compressible regions the topologies of which change in time. This causes from the numerical point of view many difficulties since standard domain decomposition techniques which couple the solution of the compressible equations with the solution of the incompressible system may be difficult to use [\[5\]](#page-20-4). Thus, one solution consists in solving the more complete compressible Euler system also in the stiff regime. However, this introduces strong drawbacks since the Mach number may become extremely small causing

*Email addresses:* giacomo.dimarco@unife.it (Giacomo Dimarco), raphael.loubere@math.u-bordeaux.fr (Raphael Loub ¨ ere), ` Victor.Michel-Dansac@math.univ-toulouse.fr (Victor Michel-Dansac), mhvignal@math.univ-toulouse.fr (Marie-Hel´ ene Vignal) `

*Preprint submitted to Elsevier October 23, 2017*

<sup>∗</sup>Corresponding author

severe time step limitations. To circumvent these problems, in the recent past, asymptotic stable techniques have been developed [18, 16, 17, 15, 32, 61, 10, 49, 11, 21]. These techniques permit to compute the solution of such stiff problems avoiding time step limitations directly related to the low Mach number regime. In addition, these methods lead to consistent approximations of the limit incompressible system when the Mach number goes to zero. In this context, in a recent work [21], a first order asymptotic preserving method has been developed. In particular, this work dealt with an analysis of the stability properties which led to a stability restriction on the numerical method independent from the Mach number. The  $L^2$  and  $L^\infty$  properties of the method have been analyzed in detail.

In the present work, we extend the previous study to the second order in time and space situation. We first present a second order extension of our previous method which is  $L^2$  stable. Successively, since it has been proved [29] that the  $L^{\infty}$  and Total Variation Diminishing (TVD, also named Strong Stability Preserving, SSP) properties cannot be assured for an unconstrained implicit in time scheme of order greater than one, we construct a new paradigm of highly accurate implicit in time schemes. We stress that as opposed to several recent studies on IMEX-SSP methods or Implicit-SSP methods [27, 36, 38, 14, 35, 58, 9] in which the authors look for the largest possible time step which allows the SSP property to be preserved, here we do not pursue in this direction since the stiffness of the equations typically requires numerical methods in which the time steps are disconnected from the stiff scales, and, possibly, several orders of magnitude larger. This as shown in [29] will not be possible with standard IMEX-SSP methods. For this reason, the direction chosen in this work consists in constructing a TVD Asymptotic Preserving (AP) scheme using a convex combination of first and second order implicit-explicit (IMEX) methods in the same spirit as high resolution shock capturing TVD methods in space [43, 59]. This permits to prove that our TVD AP scheme possesses both  $L^{\infty}$  and TVD properties and it opens the way to the construction of arbitrarily high order accurate methods with the same properties. Details of this approach and development of high resolution schemes by combining schemes of order higher than two with first order implicit methods in the case of the linear and non linear transport equations are currently under study [46].

In a second part of the work, we discuss limiters which allow to detect the troubled situations in which the TVD property is violated and subsequently to pass from the second order accurate scheme to the TVD-AP scheme without losing accuracy. The approach proposed is based on the so-called MOOD (Multidimensional Optimal Order Detection) method [12, 24, 25] originally developed to detect the loss of physical properties in space of high resolution methods and to reduce the order of the space discretization to restore the physical properties of the problem. Here, we extend the previous method to the case of the implicit time discretizations. Thus resuming, the proposed method ensures a non oscillatory approximation of our original problem which is more accurate than the one given by a first-order AP scheme, stable independently on the Mach number and which degenerates to an high order time-space discretization of the incompressible Euler equations in the limit when the Mach number goes to zero.

The article is organized as follows. In Section 2, we briefly recall the isentropic/ isothermal system of Euler equations and its low Mach number limit as well as the first order accurate Asymptotic preserving scheme presented in [21] which is the basis of the second order extension here considered. Then, in Section 3, we present a second order AP scheme in time for the isentropic Euler system and we show that even if stable, it presents some non physical oscillations when the explicit C.F.L. condition is violated. Thus, in Section 4, we introduce a model problem which will be used to construct a TVD AP scheme for the Euler system and we study its TVD,  $L^{\infty}$  and  $L^{2}$  stability properties. In Section 5 we extend the previous scheme to the low Mach number case and we introduce the MOOD procedure to detect the loss of  $L^{\infty}$  stability. Finally, in Section 6, we show the good behavior of our AP schemes with different numerical results for three different one dimensional test cases and a two-dimensional one. A concluding Section ends the paper.

### 2. The first-order asymptotic-preserving scheme for the Euler system in the low Mach number limit

We consider a bounded polygonal domain  $\Omega \in \mathbb{R}^d$ , where  $d \in \{1, 2, 3\}$ . The space and time variables are respectively denoted by  $x \in \Omega$  and  $t \in \mathbb{R}_+$ . We study the isentropic/isothermal rescaled Euler model with  $\varepsilon > 0$  the squared Mach number (see [40, 45] for instance). This reads

$$\partial_{t}\rho + \nabla \cdot (\rho U) = 0,$$
  

$$\partial_{t}(\rho U) + \nabla \cdot (\rho U \otimes U) + \frac{1}{\varepsilon} \nabla p(\rho) = 0,$$
(1)

where  $\rho(t, x) > 0$  is the density of the fluid,  $U(t, x) \in \mathbb{R}^d$  its velocity, and  $p(\rho) = \rho^{\gamma}$  its pressure. The parameter  $\gamma \ge 1$  is the ratio of specific heats,  $\gamma = 1$  corresponds to isothermal fluids while  $\gamma > 1$  to isentropic ones.

Equipped with suitable initial and boundary conditions (consistent with the limit model), system (1) tends to the incompressible isentropic/isothermal Euler system when  $\varepsilon \to 0$  (see [40, 41, 56, 2, 44, 45, 1] for rigorous results). A formal derivation of this limit is presented for instance in [21]. In the above cited works, the authors introduce well-preparedness and incompressibility assumptions on the initial and boundary conditions to show that the equations (1) tend to the following incompressible Euler equations in the low Mach number limit  $\varepsilon \to 0$ :

$$\rho = \rho_0, 
\nabla \cdot U = 0, 
\rho_0 \partial_t U + \rho_0 \nabla \cdot (U \otimes U) + \nabla \pi_1 = 0,$$
(2)

where the first-order correction of the pressure, denoted by  $\pi_1$ , is implicitly defined by the incompressibility constraint  $\nabla \cdot U = 0$ .

We now briefly recall the numerical method [21] which serves as a basis for the new method introduced in the next Section. The discretization of the space and time domains follows the usual finite volume framework. The solution  $W(t,x) = (\rho, \rho U)(t,x)$  of the Euler equations is approximated at time  $t^n = n\Delta t$ , where  $\Delta t$  is the time step, by  $W^n$ . The scheme relies on an IMEX (IMplicit-EXplicit) decomposition of system (1) (see [16, 17, 61, 21]): for all  $n \ge 0$ , the semi-discrete in time form of the scheme reads

$$\frac{W^{n+1} - W^n}{\Lambda t} + \nabla \cdot F_e(W^n) + \nabla \cdot F_i(W^{n+1}) = 0. \tag{3}$$

where  $F_e(W^n) = (0, \rho^n U^n \otimes U^n)$  and  $F_i(W^{n+1}) = (\rho^{n+1} U^{n+1}, p(\rho^{n+1})/\varepsilon \mathbb{I}_2)$  and  $F_e$  is taken explicitly while  $F_i$  implicitly. Note that the two systems associated to the two fluxes taken singularly are hyperbolic. An interesting property of such approach is that the resolution of the two equations composing the system can be decoupled. Indeed, in (3), taking the divergence of the momentum equation and inserting it into the mass equation yields:

$$\frac{\rho^{n+1} - \rho^n}{\Delta t} + (\nabla \cdot (\rho U))^n - \Delta t \left(\nabla^2 : (\rho U \otimes U)\right)^n - \frac{\Delta t}{\varepsilon} (\Delta p(\rho))^{n+1} = 0, \tag{4a}$$

$$\frac{(\rho U)^{n+1} - (\rho U)^n}{\Delta t} + (\nabla \cdot (\rho U \otimes U))^n + \frac{1}{\varepsilon} (\nabla p(\rho))^{n+1} = 0, \tag{4b}$$

where  $\nabla^2$  and : are respectively the tensor of second order derivatives and the contracted product of two tensors. Then, one can solve first the nonlinear equation (4a) which gives  $\rho^{n+1}$ , and, successively get the momentum from (4b). The implicit treatments of the pressure gradient and the mass flux respectively provide the asymptotic consistency and the uniform stability of the scheme [21].

We now present the space discretization, in one space dimension for the sake of clarity. The space domain is assumed to be partitioned in cells of center  $x_j$  and size  $\Delta x$ . Then, on  $[t^n, t^{n+1}]$  the fully discrete version of (4) reads as follows

$$\frac{W_{j}^{n+1} - W_{j}^{n}}{\Delta t} + \frac{(\mathcal{F}_{e})_{j+\frac{1}{2}}^{n} - (\mathcal{F}_{e})_{j-\frac{1}{2}}^{n}}{\Delta x} + \frac{(\mathcal{F}_{i})_{j+\frac{1}{2}}^{n,n+1} - (\mathcal{F}_{i})_{j-\frac{1}{2}}^{n,n+1}}{\Delta x} - \Delta t \left( \left( \Delta(\rho u^{2}) \right)_{j}^{n} + \frac{1}{\varepsilon} \left( \Delta p(\rho) \right)_{j}^{n+1} \right) = 0.$$
 (5)

The explicit numerical flux is given by

$$(\mathcal{F}_e)_{j+\frac{1}{2}}^n := \frac{F_e(W_j^n) + F_e(W_{j+1}^n)}{2} + (\mathcal{D}_e)_{j+\frac{1}{2}}^n (W_{j+1}^n - W_j^n), \tag{6}$$

with  $(\mathcal{D}_e)_{j+\frac{1}{2}}^n$  the explicit viscosity coefficient, taken as half of the maximum explicit eigenvalue and given by  $(\mathcal{D}_e)_{j+\frac{1}{2}}^n$  :=  $\max(|u_j^n|,|u_{j+1}^n|)$ . The implicit numerical flux is given by

$$(\mathcal{F}_i)_{j+\frac{1}{2}}^{n,n+1} := \frac{F_i(W_j^{n,n+1}) + F_i(W_{j+1}^{n,n+1})}{2} + (\mathcal{D}_i)_{j+\frac{1}{2}}^n (W_{j+1}^{n+1} - W_j^{n+1}), \tag{7}$$

where  $W_j^{n,n+1} = (\rho_j^{n+1}, q_j^n)$  and  $(\mathcal{D}_i)_{j+\frac{1}{2}}^n$  is the implicit viscosity coefficient, taken as half of the maximum implicit eigenvalue  $(\mathcal{D}_i)_{j+\frac{1}{2}}^n := \frac{1}{2} \max \left( \sqrt{p'(\rho_j^n)/\varepsilon}, \sqrt{p'(\rho_{j+1}^n)/\varepsilon} \right)$ . This choice for the implicit viscosity is enough to get an  $L^\infty$  stable scheme. However, by relaxing the request and fixing it to zero one can show that an  $L^2$  stable scheme is obtained. Finally, the second-order derivatives are approximated by classical second order centered differences while the time step is constrained by the following uniform C.F.L. condition:

$$\Delta t \le \frac{\Delta x}{\max_{j} \left( 2|u_{j}^{n}| \right)}. \tag{8}$$

Note that 2u corresponds to the first eigenvalue of the explicit flux and as expected, this C.F.L. condition does not depend on the Mach number  $\varepsilon$ . When  $\varepsilon$  tends to 0, this scheme yields a consistent discretization of the incompressible system (2). In the following Sections we discuss an extension of this method to the case of high order time and space discretizations.

# 3. A second order asymptotically accurate scheme for the isentropic Euler equations

The second order in time extension of the method described in the previous Section is based on an Implict-Explicit (IMEX) Runge-Kutta approach [3, 52, 53, 22, 6, 23, 7, 4]. In particular, we make use of the second order Ascher, Ruuth and Spiteri [3] scheme denoted in the sequel by ARS(2,2,2). Let us observe that this scheme has been originally constructed to deal with convection-diffusion equations and in particular to deal with cases in which the diffusion (the fast scale) is taken implicit while the convection explicit. In our case, the problem is different, since the fast and the slow scales are both of hyperbolic type. This causes a real challenge from the numerical point of view. In fact, as already mentioned, implicit methods of order higher than one for hyperbolic problems cannot be TVD [29]. Moreover, the situation does not change when implicit-explicit methods are employed as shown later. Thus, one can think to rely on optimal IMEX-SSP methods [38, 9, 35, 58] to solve the problem. However, in this case, time steps allowing the TVD property to be preserved are of the order of explicit time integrators. Unfortunately, since we are dealing with a limit problem, we look for a method which preserves the TVD property independently on the Mach number which eventually can also be set to zero. Thus, in order to bring remedy to this problematic situation, the idea explored in this work consists in blending together first and second order implicit time-space discretizations giving rise to a new class of high resolution in time methods which guarantees the preservation of the  $L^{\infty}$  stability and TVD property. Here, we discuss two given implicit time discretizations, while we refer to [46] for the construction of general TVD high resolution implicit-explicit time discretizations.

The Butcher tableau relative to the considered ARS(2,2,2) scheme is detailed in Table 1 with  $\beta = 1 - \sqrt{2}/2$  and  $\alpha = 1 - 1/(2\beta)$ . Note that on the left is reported the explicit tableau applied to the flux  $F_e$  while on the left the implicit tableau applied to the flux  $F_i$ . Remarking that  $\alpha = \beta - 1$  and so  $1 - \alpha = 2 - \beta$ , the corresponding semi-discretization

$$\begin{array}{c|ccccccccccccccccccccccccccccccccccc$$

Table 1: Butcher tableaux for the ARS(2,2,2) time discretization. Left panel: explicit tableau. Right panel: implicit tableau.

of the Euler system is given by

$$\frac{W^{\star} - W^n}{\Delta t} + \beta \nabla \cdot F_e(W^n) + \beta \nabla \cdot F_i(W^{\star}) = 0, \tag{9a}$$

$$\frac{W^{n+1} - W^n}{\Lambda t} + (\beta - 1) \nabla \cdot F_e(W^n) + (2 - \beta) \nabla \cdot F_e(W^*) + (1 - \beta) \nabla \cdot F_i(W^*) + \beta \nabla \cdot F_i(W^{n+1}) = 0.$$
 (9b)

Likewise for the first-order accurate scheme, the previous second-order accurate discretization has an uncoupled formulation. Let us first establish the first step (9a). Taking the divergence of the momentum equation of (9a) and

inserting the value of  $\nabla \cdot (\rho U)^*$  into the mass equation of (9a), yield the following uncoupled formulation:

$$\frac{W^{\star} - W^{n}}{\Delta t} + \beta \nabla \cdot F_{e}(W^{n}) + \beta \nabla \cdot F_{i}(W^{n,\star}) - \beta^{2} \Delta t \begin{pmatrix} \nabla^{2} : (\rho U \otimes U)^{n} + \frac{1}{\varepsilon} \Delta p(\rho^{\star}) \\ 0 \end{pmatrix} = 0,$$

where  $W^{n,\star} = (\rho^{\star}, (\rho U)^n)$ . Using now the same notation as in the previous section for the first-order accurate scheme, the full discretized uncoupled first step in one dimension, is given by

$$\frac{W_{j}^{\star} - W_{j}^{n}}{\Delta t} + \beta \frac{(\mathcal{F}_{e})_{j+\frac{1}{2}}^{n} - (\mathcal{F}_{e})_{j-\frac{1}{2}}^{n}}{\Delta x} + \beta \frac{(\mathcal{F}_{i})_{j+\frac{1}{2}}^{n,\star} - (\mathcal{F}_{i})_{j-\frac{1}{2}}^{n,\star}}{\Delta x} - \beta^{2} \Delta t \left( \left( \Delta(\rho u^{2}) \right)_{j}^{n} + \frac{1}{\varepsilon} \left( \Delta p(\rho) \right)_{j}^{\star} \right) = 0.$$
 (10a)

We turn to the uncoupled formulation of the second step (9b). We insert the divergence of  $\rho U^{n+1}$ , obtained with the momentum equation of (9b), into the mass equation of (9b). This yields

$$\frac{W^{n+1} - W^n}{\Delta t} + (\beta - 1) \nabla \cdot F_e(W^n) + (2 - \beta) \nabla \cdot F_e(W^*) + (1 - \beta) \nabla \cdot F_i(W^*) + \beta \nabla \cdot F_i(W^{n,n+1})$$
$$-\beta \Delta t \left( (\beta - 1) \nabla^2 : (\rho U \otimes U)^n + (2 - \beta) \nabla^2 : (\rho U \otimes U)^* + \frac{(1 - \beta)}{\varepsilon} \Delta p(\rho^*) + \frac{\beta}{\varepsilon} \Delta p(\rho^{n+1}) \right) = 0.$$

Using the same notation as before, the fully discretized second step in one dimension is given by

$$\frac{W_{j}^{n+1} - W_{j}^{n}}{\Delta t} + (\beta - 1) \frac{(\mathcal{F}_{e})_{j+\frac{1}{2}}^{n} - (\mathcal{F}_{e})_{j-\frac{1}{2}}^{n}}{\Delta x} + (2 - \beta) \frac{(\mathcal{F}_{e})_{j+\frac{1}{2}}^{\star} - (\mathcal{F}_{e})_{j-\frac{1}{2}}^{\star}}{\Delta x} + (1 - \beta) \frac{(\mathcal{F}_{i})_{j+\frac{1}{2}}^{\star,\star} - (\mathcal{F}_{i})_{j-\frac{1}{2}}^{\star,\star}}{\Delta x} + \beta \frac{(\mathcal{F}_{i})_{j+\frac{1}{2}}^{n,n+1} - (\mathcal{F}_{i})_{j-\frac{1}{2}}^{n,n+1}}{\Delta x} - \beta \Delta t \left( (\beta - 1) \left( \Delta(\rho u^{2}) \right)_{j}^{n} + (2 - \beta) \left( \Delta(\rho u^{2}) \right)_{j}^{\star} + \frac{(1 - \beta)}{\varepsilon} \left( \Delta p(\rho) \right)_{j}^{\star} + \frac{\beta}{\varepsilon} \left( \Delta p(\rho) \right)_{j}^{n+1} \right) = 0.$$
(10b)

**Lemma 1.** The scheme (9) is asymptotically consistent with system (2) in the limit  $\varepsilon \to 0$ .

*Proof.* We do not assume well-prepared initial conditions but general initial conditions  $\rho(0, x) = \rho^0(x)$  and  $U(0, x) = U^0(x)$ . Well prepared initial conditions will converge to a constant density and a divergence free velocity when  $\varepsilon$  tends to 0. We consider the boundary condition  $U(x, t) \cdot v(x) = 0$  for all  $t \ge 0$  and all  $x \in \partial \Omega$  the boundary of  $\Omega$ , where v is the outward unit normal.

We assume that all discrete quantities (densities and momentums) have a limit when  $\varepsilon \to 0$ , then at the first timestep n=0, multiplying the momentum equation of (9a) and letting  $\varepsilon$  tends to 0, gives  $\nabla p(\rho^*)=0$  and so  $\nabla \rho^*=0$ . Then, integrating the mass equation of (9a) on the domain and using the boundary condition  $U^* \cdot \nu = 0$  on  $\partial \Omega$ , one gets  $\rho^* = \langle \rho^0 \rangle = 1/|\Omega| \int_{\Omega} \rho^0(x) dx$ . Similarly, the second stage (9b) gives  $\nabla \rho^1 = 0$ , and integrating the mass equation and using the boundary conditions  $U^* \cdot \nu = U^1 \cdot \nu = 0$ , we obtain  $\rho^1 = \rho^* = \langle \rho^0 \rangle$ .

Note that inserting this result into the mass equation of the first stage (9a), we do not recover the incompressibility constraint for the first time-step since  $\nabla \cdot U^*(x) = (<\rho^0>-\rho^0(x))/(\Delta t<\rho^0>)$  which equals 0 if and only if the initial density  $\rho^0$  is well prepared and tends to a constant when  $\varepsilon$  tends to 0. But, for all  $n \ge 1$ , we recover the incompressibility constraint for the first stage  $\nabla \cdot U^* = 0$ . Finally, thanks to  $\nabla \cdot U^* = 0$ , the density equation gives  $\nabla \cdot U^{n+1} = 0$  for all  $n \ge 1$ . Consequently, the scheme projects the solution over the asymptotic incompressible limit even if the initial data are not well-prepared to this limit, we obtain  $\rho^{n+1} = \langle \rho^0 \rangle := \rho_0$  and  $\nabla \cdot U^{n+1} = 0$  for all  $n \ge 1$ . Concerning the pressure, for  $n \ge 1$ , the limit scheme becomes

$$\rho_0 \frac{U^* - U^n}{\Lambda t} + \beta \rho_0 \nabla \cdot (U \otimes U)^n + \beta \nabla \pi_1^* = 0, \tag{11a}$$

$$\rho_0 \frac{U^{n+1} - U^n}{\Delta t} + \alpha \rho_0 \nabla \cdot (U \otimes U)^n + (1 - \alpha)\rho_0 \nabla \cdot (U \otimes U)^* + (1 - \beta)\nabla \pi_1^* + \beta \nabla \pi_1^{n+1} = 0, \tag{11b}$$

where 
$$\pi_1^* = \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \left( p(\rho^*) - p(\rho_0) \right)$$
 and  $\pi_1^{n+1} = \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} \left( p(\rho^{n+1}) - p(\rho_0) \right)$ .

Now, we test this second order accurate uncoupled AP scheme (10) on a shock tube test case (see Section 6.1 for the details). The C.F.L is uniform and given by (8). The exact solution is constituted of a rarefaction wave and a shock wave and Figure 1 reports the numerical solution of the first and second order AP schemes described above for different values of the Mach number  $\varepsilon$  while the space discretization is always first order. As we can see, the second-order AP

![](_page_5_Figure_3.jpeg)

Figure 1: Approximations of the density  $\rho$  for a rarefaction-shock Riemann problem and different values of the Mach number.

scheme gives more accurate results but it presents some oscillations when the Mach number decreases for a fixed value of the time step given by the C.F.L condition (8). These oscillations disappear when the time step is reduced and in particular when the non uniform explicit C.F.L condition is satisfied ( $\Delta t \leq \Delta x/(\max_j \left(|u_j^n \pm \sqrt{p'(\rho_j^n)/\varepsilon}|\right)$ ). Thus, as anticipated, a second order implicit-explicit time discretization for this kind of problem suffers of the same limitations of standard implicit time discretizations for hyperbolic problems of order higher than one: TVD property is lost. In the sequel, we first study and find a solution to the above problematic situations in the simplified setting of linear transport equation and successively we extend the found result to the case of the Euler equations.

### 4. Study of the stability on a model problem

We consider the linear advection equation:

$$\partial_t w + c_e \partial_x w + \frac{c_i}{\sqrt{\varepsilon}} \partial_x w = 0, \tag{12}$$

with  $c_e > 0$  and  $c_i > 0$  fixed real numbers. Note that the dependency in  $\sqrt{\varepsilon}$  of the fast velocity is similar to that of the velocity of the pressure waves in the Euler system. The formal limit  $\varepsilon \to 0$  of the above equation is a constant and uniform solution. The first-order AP scheme detailed in the previous Section becomes

$$w_{j}^{n+1} = w_{j}^{n} - \frac{\Delta t}{\Delta x} c_{e} \left( w_{j}^{n} - w_{j-1}^{n} \right) - \frac{\Delta t}{\Delta x} \frac{c_{i}}{\sqrt{\varepsilon}} \left( w_{j}^{n+1} - w_{j-1}^{n+1} \right). \tag{13}$$

The following results hold

**Lemma 2.** For periodic boundary conditions  $w_0^n = w_L^n$  and  $w_{L+1}^n = w_1^n$  for all  $n \ge 0$  and if the following uniform C.F.L. holds true

$$\Delta t \le \frac{\Delta x}{c_e}.\tag{14}$$

Then, scheme (13) is Asymptotic Preserving and asymptotically  $L^2$ - and  $L^{\infty}$ -stable, that is

$$\left\| w^{n+1} \right\|_2 = \left( \sum_{j=1}^L |w_j^{n+1}|^2 \right)^{1/2} \le \|w^n\|_2 \qquad and \qquad \left\| w^{n+1} \right\|_\infty = \max_{j=1}^L |w_j^{n+1}| \le \|w^n\|_\infty.$$

and TVD, that is

$$TV(w^{n+1}) \le TV(w^n) = \sum_{i=1}^{L} \left| w_{j+1}^n - w_j^n \right|.$$

*Proof.* The proofs of asymptotically consistency and of the  $L^2$ - and  $L^\infty$ -stabilities can be easily established following the results proved in [21] and we omit them. It remains to prove the TVD property. Using (13) and the periodic boundary conditions, we have for all  $i = 1, \dots, L$ 

$$\left(w_{j+1}^{n+1}-w_{j}^{n+1}\right)\left(1+\frac{c_{i}\,\Delta t}{\sqrt{\varepsilon}\,\Delta x}\right)-\frac{c_{i}\,\Delta t}{\sqrt{\varepsilon}\,\Delta x}\left(w_{j}^{n+1}-w_{j-1}^{n+1}\right)=\left(w_{j+1}^{n}-w_{j}^{n}\right)\left(1-\frac{c_{e}\,\Delta t}{\Delta x}\right)+\frac{c_{e}\,\Delta t}{\Delta x}\left(w_{j}^{n}-w_{j-1}^{n}\right).$$

Taking the absolute value and remarking that for all a, b real numbers,  $|a| - |b| \le |a - b|$ , summing for all  $j = 1, \dots, L$  and using the periodic boundary conditions and the C.F.L. condition (14), we obtain

$$\begin{split} \sum_{j=1}^{L} \left| w_{j+1}^{n+1} - w_{j}^{n+1} \right| &= \left( 1 + \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} \right) \sum_{j=1}^{L} \left| w_{j+1}^{n+1} - w_{j}^{n+1} \right| - \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} \, \sum_{j=1}^{L} \left| w_{j}^{n+1} - w_{j-1}^{n+1} \right| \\ &\leq \sum_{j=1}^{L} \left| \left( w_{j+1}^{n+1} - w_{j}^{n+1} \right) \left( 1 + \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} \right) - \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} \left( w_{j}^{n+1} - w_{j-1}^{n+1} \right) \right| \\ &\leq \left( 1 - \frac{c_{e} \, \Delta t}{\Delta x} \right) \sum_{j=1}^{L} \left| w_{j+1}^{n} - w_{j}^{n} \right| + \frac{c_{e} \, \Delta t}{\Delta x} \, \sum_{j=1}^{L} \left| w_{j}^{n} - w_{j-1}^{n} \right| = \sum_{j=1}^{L} \left| w_{j+1}^{n} - w_{j}^{n} \right|. \end{split}$$

We turn now our attention to the two-step ARS(2,2,2) second-order time discretization, we refer to it to as second order AP-scheme. This reads

$$w_j^{\star} = w_j^n - \beta c_e \frac{\Delta t}{\Delta x} \left( w_j^n - w_{j-1}^n \right) - \beta \frac{c_i}{\sqrt{\varepsilon}} \frac{\Delta t}{\Delta x} \left( w_j^{\star} - w_{j-1}^{\star} \right), \tag{15a}$$

$$w_{j}^{n+1} = w_{j}^{n} - (\beta - 1)c_{e}\frac{\Delta t}{\Delta x}\left(w_{j}^{n} - w_{j-1}^{n}\right) - (1 - \beta)\frac{c_{i}}{\sqrt{\varepsilon}}\frac{\Delta t}{\Delta x}\left(w_{j}^{\star} - w_{j-1}^{\star}\right) - (2 - \beta)c_{e}\frac{\Delta t}{\Delta x}\left(w_{j}^{\star} - w_{j-1}^{\star}\right) - \beta\frac{c_{i}}{\sqrt{\varepsilon}}\frac{\Delta t}{\Delta x}\left(w_{j}^{n+1} - w_{j-1}^{n+1}\right).$$

$$(15b)$$

**Proposition 1.** For periodic boundary conditions, the two-step scheme (15) is asymptotically consistent: if  $w_j^0 = \overline{w}$  for all  $j = 1, \dots, L$ , then for all  $n \ge 0$ ,  $w_j^n = \overline{w}$ , for all  $j = 1, \dots, L$ .

*Proof.* Multiplying, both equations of (15), an passing to the limit  $\varepsilon \to 0$  yields  $w_j^* = w_1^*$  and  $w_j^{n+1} = w_1^{n+1}$  for for all  $j = 1, \dots, L$ . Now summing (15b) for  $j = 1, \dots, L$ , we obtain by induction  $w_j^{n+1} = w_1^{n+1} = w_1^n = \dots = \overline{w}$ .

Concerning the  $L^2$  stability of the scheme (15), we can prove the following result

**Proposition 2.** For periodic boundary conditions, the scheme (15) is  $L^2$ -stable under the C.F.L. condition (14).

*Proof.* Using Fourier analysis and setting  $w_j^{n+1,\star,n} = \sum_k \hat{w}_k^{n+1,\star,n} e^{ikj\Delta x}$ , we obtain that  $w_k^{\star} = f_k w_k^n$  and  $w_k^{n+1} = g_k w_k^n$  where  $f_k = (1 - \beta \sigma_e (1 - c + i s))/(1 + \beta \sigma_i^{\varepsilon} (1 - c + i s))$  with  $\sigma_e = \frac{c_e \Delta t}{\Delta x}$ ,  $\sigma_i^{\varepsilon} = \frac{c_i \Delta t}{\varepsilon \Delta x}$ ,  $c = \cos(k \Delta x)$ ,  $s = \sin(k \Delta x)$ , and where

$$g_{k} = \frac{1 - (\beta - 1) \,\sigma_{e} \,(1 - c + i \,s)}{1 + \beta \,\sigma_{i}^{\varepsilon} \,(1 - c + i \,s)} - \frac{\left((1 - \beta) \,\sigma_{i}^{\varepsilon} + (2 - \beta)\sigma_{e}\right) (1 - c + i \,s) \,(1 - \beta \,\sigma_{e} \,(1 - c + i \,s))}{(1 + \beta \,\sigma_{i}^{\varepsilon} \,(1 - c + i \,s))^{2}}.$$

Remarking that  $x(1-x) \in [0,1/4]$  for all  $x \in [0,1]$ , we easily obtain that under the condition  $\beta \sigma_e = \beta \frac{c_e \Delta t}{\Delta x} \le 1$ , we have  $|f_k|^2 \le 1 - 2(\beta \sigma_e)(1-c)(1-2\beta \sigma_e) \le 1$ , for all  $\sigma_i^\varepsilon \ge 0$  and  $c \in [-1,1]$ . Furthermore, an easy calculation shows

that  $|g_k|^2$  depends only on  $s^2$  and has a finite limit when  $\sigma_i^\varepsilon \to +\infty$ . Then, setting  $\sigma_e = 1$  and plotting the function  $(c, \sigma_i^\varepsilon) \mapsto |g_k|^2$  for  $c \in [-1, 1]$  and  $\sigma_i^\varepsilon \in [0, 1]$  and setting  $\sigma_e = 1$ ,  $\mu_i^\varepsilon = 1/\sigma_i^\varepsilon$  and plotting the function  $(c, \mu_i^\varepsilon) \mapsto |g_k|^2$  for  $c \in [-1, 1]$  and  $\mu_i^\varepsilon \in [0, 1]$ , we prove that, for all  $\sigma_i^\varepsilon \ge 0$  and  $c \in [-1, 1]$ 

$$\sigma_e = \frac{c_e \, \Delta t}{\Delta x} \le 1 \quad \Rightarrow \quad |g_k|^2 \le 1.$$

On the other hand, the above scheme is not uniformly  $L^{\infty}$ -stable and nor uniformly TVD. Let us see it with a counterexample. We consider the following initial data on the space domain [0, 1]

$$w(0, x) = \begin{cases} \varepsilon & \text{if } 0.25 < x \le 0.75, \\ -\varepsilon & \text{otherwise.} \end{cases}$$
 (16)

and on Figure 2, we display the results of the first-order AP scheme (13) and the second-order AP scheme (15) for two different values of the Mach number using the non-restrictive C.F.L. condition (14) and periodic boundary conditions. The results are the following: the first-order AP scheme is in-bounds but diffusive, the second-order AP scheme produces bounded spurious oscillations when the time step violates the explicit C.F.L. condition  $\Delta x/(c_e + c_i/\sqrt{\varepsilon})$ , thus preventing it from being  $L^{\infty}$ -stable or TVD independently on  $\varepsilon$ . This loss of stability for the case of an implicit-

![](_page_7_Figure_5.jpeg)

Figure 2: Advection with equation (12) of the rectangular pulse (16). (left panel:  $\varepsilon = 10^{-2}$ ; right panel:  $\varepsilon = 10^{-4}$ ). Comparison of first-order AP scheme (13) (dotted line) and the second-order AP scheme (15) (dashed line) against the exact solution (solid line).

explicit second order scheme shares many similarities with a negative result [29] proved in the case of sole implicit high order Runge-Kutta time discretizations that we recall here

**Theorem 1.** ([29]) There does not exist TVD implicit Runge-Kutta schemes with unconstrained time steps of order higher than one.

To tackle this problem and obtain a TVD numerical scheme that is more accurate than a first-order discretization, we propose to introduce a convex combination between a first-order implicit-explicit scheme and the IMEX ARS discretization, as follows:

$$w_j^{n+1} = \theta w_j^{n+1,O1} + (1-\theta) w_j^{n+1,O2},$$

where  $w_j^{n+1,O1}$  is given by the first-order AP scheme (13),  $w_j^{n+1,O2}$  by the second-order AP one (15) and  $\theta \in [0,1]$ . The spirit is the same of high resolution methods which employ the so-called flux limiter approach [43] for constructing high order TVD schemes. Since, as in the case of high order space discretizations, it is not possible to avoid spurious oscillations, we couple high order discretizations with first order ones and in cases in which the TVD property is violated we come back to the first order discretization which assures monotonicity. This approach gives the following

limited scheme

$$w_{j}^{\star} = w_{j}^{n} - \beta c_{e} \frac{\Delta t}{\Delta x} \left( w_{j}^{n} - w_{j-1}^{n} \right) - \beta \frac{c_{i}}{\sqrt{\varepsilon}} \frac{\Delta t}{\Delta x} \left( w_{j}^{\star} - w_{j-1}^{\star} \right), \tag{17a}$$

$$w_{j}^{n+1} = w_{j}^{n} - \theta(\beta - 1) c_{e} \frac{\Delta t}{\Delta x} \left( w_{j}^{n} - w_{j-1}^{n} \right) - \theta(1 - \beta) \frac{c_{i}}{\sqrt{\varepsilon}} \frac{\Delta t}{\Delta x} \left( w_{j}^{\star} - w_{j-1}^{\star} \right)$$

$$- \theta(2 - \beta) c_{e} \frac{\Delta t}{\Delta x} \left( w_{j}^{\star} - w_{j-1}^{\star} \right) - \theta \beta \frac{c_{i}}{\sqrt{\varepsilon}} \frac{\Delta t}{\Delta x} \left( w_{j}^{n+1} - w_{j-1}^{n+1} \right)$$

$$- (1 - \theta) c_{e} \frac{\Delta t}{\Delta x} \left( w_{j}^{n} - w_{j-1}^{n} \right) - (1 - \theta) \frac{c_{i}}{\sqrt{\varepsilon}} \frac{\Delta t}{\Delta x} \left( w_{j}^{n+1} - w_{j-1}^{n+1} \right). \tag{17b}$$

For which the following results hold true:

**Theorem 2.** With periodic boundary conditions, scheme (17) is asymptotically consistent.

*Proof.* The scheme (17) is a result of a convex combination of two asymptotically consistent schemes (13) and (15). Therefore, it is also asymptotically consistent.  $\Box$ 

**Theorem 3.** The scheme (17) is asymptotically stable, i.e. uniformly TVD and  $L^{\infty}$ -stable:

$$\|w^{\star}\|_{\infty} \le \|w^{n}\|_{\infty}, \qquad \|w^{n+1}\|_{\infty} \le \|w^{n}\|_{\infty}, \qquad TV(w^{\star}) \le TV(w^{n}), \qquad TV(w^{n+1}) \le TV(w^{n}),$$

if the following uniform C.F.L. conditions

$$(1 - \alpha) \frac{c_e \Delta t}{\Delta x} \le 1 - \alpha, \qquad \alpha \frac{c_e \Delta t}{\Delta x} \le \alpha \frac{1}{\frac{\beta}{1 - \beta} (2 - \beta)} = \alpha \sqrt{2}. \tag{18}$$

are verified for  $\alpha \in [0,1]$  and  $\theta = \alpha \frac{\beta}{1-\beta} = \alpha (\sqrt{2}-1) \in ]0,1[$ ,

*Proof.* Let us first note that, like for the first-order AP scheme, the first step (17a) of the scheme is TVD and  $L^{\infty}$ -stable under the non-restrictive explicit CFL condition  $c_e \Delta t/\Delta x \le 1/\beta$  which is true since  $1 \le \sqrt{2} \le 1/\beta$ . Therefore,  $\|w^*\|_{\infty} \le \|w^n\|_{\infty}$  and  $TV(w^*) \le TV(w^n)$ .

Let us prove the  $L^{\infty}$  stability of the second step. We denote by  $j_0$  the index in  $\{1, \dots, L\}$  such that  $w_{j_0}^{n+1} = \max_{j=1}^{L} w_j^{n+1}$ . Thanks to the periodic boundary conditions we have  $w_{j_0}^{n+1} = \max_{j=0}^{L+1} w_j^{n+1}$ . Then, we rewrite the second step (17b) as follows:

$$\begin{split} w_{j_0}^{n+1} & \leq w_{j_0}^{n+1} + (1 - \theta + \theta \beta) \, \frac{c_i \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} (w_{j_0}^{n+1} - w_{j_0-1}^{n+1}) = \, w_{j_0}^n \left( 1 - \theta (\beta - 1) \frac{c_e \, \Delta t}{\Delta x} - (1 - \theta) \frac{c_e \, \Delta t}{\Delta x} \right) \\ & + w_{j_0-1}^n \left( \theta (\beta - 1) \frac{c_e \, \Delta t}{\Delta x} + (1 - \theta) \frac{c_e \, \Delta t}{\Delta x} \right) \\ & - \theta (1 - \beta) \frac{c_i \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} (w_{j_0}^{\star} - w_{j_0-1}^{\star}) - \theta (2 - \beta) \frac{c_e \, \Delta t}{\Delta x} (w_{j_0}^{\star} - w_{j_0-1}^{\star}). \end{split}$$

From the first step (17a), we deduce  $-\frac{c_i \Delta t}{\sqrt{\epsilon}}(w_{j_0}^{\star} - w_{j_0-1}^{\star}) = \frac{1}{\beta}(w_{j_0}^{\star} - w_{j_0}^{n}) + \frac{c_e \Delta t}{\Delta x}(w_{j_0}^{n} - w_{j_0-1}^{n})$ . Plugging this expression into the previous equality leads to:

$$\max_{j=0}^{L+1} w_{j}^{n+1} = w_{j_0}^{n+1} \leq w_{j_0}^{n} \left( 1 - \theta \frac{1 - \beta}{\beta} - \frac{c_e \Delta t}{\Delta x} \left( 1 - \theta \frac{1 - \beta}{(3 - 2\beta)} \right) + w_{j_0 - 1}^{n} \frac{c_e \Delta t}{\Delta x} \left( 1 - \theta \frac{1 - \beta}{\beta} \right) + w_{j_0}^{\star} \left( \theta \frac{1 - \beta}{\beta} - \theta (2 - \beta) \frac{c_e \Delta t}{\Delta x} \right) + w_{j_0 - 1}^{\star} \theta (2 - \beta) \frac{c_e \Delta t}{\Delta x} .$$

Note that if all coefficients are positive, we will have two convex combinations, one at time index n and one at time index  $\star$ , and

$$\max_{j=0}^{L+1} w_j^{n+1} = w_{j_0}^{n+1} \le \max_{j=0}^{L+1} w_j^n \left( 1 - \theta \frac{1 - \beta}{\beta} \right) + \theta \frac{1 - \beta}{\beta} \max_{j=0}^{L+1} w_j^{\star} \le \max_{j=0}^{L+1} w_j^n.$$

From the above expression we deduce that necessary conditions for having positive coefficients are  $1 - \theta \frac{1-\beta}{\beta} \ge 0$  and  $\theta \frac{1-\beta}{\beta} \ge 0$ . This gives the condition  $\theta \in [0, \beta/(1-\beta)]$ . By setting  $\theta = \alpha \frac{\beta}{1-\beta}$ , with  $\alpha \in [0, 1]$ , then all coefficients are positive under the C.F.L. conditions (18) and consequently the  $L^{\infty}$  property is verified.

We can now prove the TVD property. Using (17b) and the periodic boundary conditions and using the first step, we have for all  $j = 1, \dots, L$ 

$$\begin{split} \left(w_{j+1}^{n+1} - w_{j}^{n+1}\right) \left(1 + (1 - \theta + \theta\beta) \, \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x}\right) - (1 - \theta + \theta\beta) \, \frac{c_{i} \, \Delta t}{\sqrt{\varepsilon} \, \Delta x} \left(w_{j}^{n} - w_{j-1}^{n}\right) \\ &= \left(w_{j+1}^{n} - w_{j}^{n}\right) \left(1 - \theta \, \frac{1 - \beta}{\beta} - \frac{c_{e} \, \Delta t}{\Delta x} \left(1 + \theta \, (1 - 2\beta)\right)\right) + \, \left(w_{j}^{n} - w_{j-1}^{n}\right) \, \frac{c_{e} \, \Delta t}{\Delta x} \left(1 + \theta \, (1 - 2\beta)\right) \\ &+ \, \left(w_{j+1}^{\star} - w_{j}^{\star}\right) \left(\theta \, \frac{1 - \beta}{\beta} - \theta (2 - \beta) \frac{c_{e} \, \Delta t}{\Delta x}\right) + \left(w_{j}^{\star} - w_{j-1}^{\star}\right) \theta (2 - \beta) \frac{c_{e} \, \Delta t}{\Delta x}. \end{split}$$

Taking the absolute value, remarking that for all a, b real numbers,  $|a| - |b| \le |a - b|$ , summing for all  $j = 1, \dots, L$  and using the C.F.L. condition (18) and the periodic boundary conditions, we conclude the proof:

$$\begin{split} \sum_{j=1}^{L} |w_{j+1}^{n+1} - w_{j}^{n+1}| &\leq \sum_{j=1}^{L} |w_{j+1}^{n} - w_{j}^{n}| \bigg(1 - \theta \, \frac{1 - \beta}{\beta} - \frac{c_{e} \, \Delta t}{\Delta x} \Big(1 + \theta \, (1 - 2 \, \beta)\Big) \bigg) \\ &\qquad \qquad + \sum_{j=1}^{L} |w_{j}^{n} - w_{j-1}^{n}| \, \frac{c_{e} \, \Delta t}{\Delta x} \Big(1 + \theta \, (1 - 2 \, \beta) \Big) \\ &\qquad \qquad + \sum_{j=1}^{L} |w_{j}^{\star} - w_{j-1}^{\star}| \, \left(\theta \, \frac{1 - \beta}{\beta} - \theta (2 - \beta) \frac{c_{e} \, \Delta t}{\Delta x}\right) + \sum_{j=1}^{L} |w_{j}^{\star} - w_{j-1}^{\star}| \theta (2 - \beta) \frac{c_{e} \, \Delta t}{\Delta x} \leq \sum_{j=1}^{L} |w_{j+1}^{n} - w_{j}^{n}|. \end{split}$$

**Remark 1.** 1. Theorem 3 shows that the largest possible value for  $\theta$  is  $\theta_M = \beta/(1-\beta) = \sqrt{2}-1 \approx 0.4142$  if TVD and  $L^{\infty}$  properties need to be assured. We refer to as TVD-AP scheme when discussing of scheme (17) with  $\theta = \theta_M$ . This value depends on the type of second order time discretization chosen. Other time discretizations, may allow larger values which may possibly improve the accuracy of the method. This situation will be discussed in detail in [46].

2. In order to additionally increase the accuracy of the method one introduces local values for  $\theta$  different for each spatial cell in equation (17). However, in this case, the proof of the TVD as well as of the  $L^{\infty}$  stability remain an open problem. In addition, numerical experiments performed suggest that such a local parameter must be chosen related to a stencil of neighbors which is proportional to the velocity  $c_i/\sqrt{\epsilon}$ . These aspects will be studied in detail in [46].

We discuss now limiters which allow to detect the situations in which the TVD or the  $L^{\infty}$  stability property is violated and consequently to switch from the second order accurate scheme to the TVD-AP scheme without loosing excessive accuracy by diminishing parameter  $\theta$ . The proposed approach is based on a detection technique borrowed from the MOOD (Multidimensional Optimal Order Detection) [12, 24, 25] framework. The idea behind this specific MOOD approach is to use the second-order oscillatory discretization (15) whenever possible, i.e. when no oscillations appear. Instead, if at time n the numerical solution presents oscillations, we discard it and we replace it by the limited TVD-AP scheme, i.e. scheme (17) with  $\theta = \theta_M = \beta/(1-\beta)$  which assures preservation of the demanded properties. Since, for this specific situation, the  $L^{\infty}$  norm of the solution is preserved in time, spurious oscillations are checked with respect to the initial condition instead of that of the previous time iteration of the scheme. Indeed, the relevant bounds are those of the initial condition rather than the ones of the diffusive numerical approximation. The procedure can be summarized by the following algorithm

**Algorithm 1.** 1. Compute a candidate solution  $w^{n+1,O2}$  using the second-order AP scheme (15).

2. Detect if this candidate solution satisfies the  $L^{\infty}$  stability and the TVD property.

3. If these two criteria are satisfied by  $w^{n+1,O2}$ , set  $w^{n+1} = w^{n+1,O2}$ ; otherwise, compute  $w^{n+1}$  using the TVD-AP scheme (17) with  $\theta = \theta_M = \beta/(1-\beta)$ .

We refer to as the AP-MOOD scheme when this Algorithm is used. In Figure 3, we report the results of the advection of the rectangular pulse given by (16), for different values of the parameter  $\varepsilon$ . The solution is computed by the first order AP scheme, the second order AP scheme, the TVD-AP scheme and the AP-MOOD scheme. The exact solution is also reported. One can clearly see the differences in terms of accuracy for the different methods proposed and the absence of spurious oscillations for the TVD-AP and the AP-MOOD methods. In the sequel, we will extend this approach to the case of the Euler equations.

![](_page_10_Figure_2.jpeg)

Figure 3: Advection with equation (12) of the rectangular pulse (16). (left panel:  $\varepsilon = 10^{-2}$ ; right panel:  $\varepsilon = 10^{-4}$ ). Comparison of the first-order AP scheme (13) (dotted line), the second-order AP scheme (15) (dashed line), the TVD-AP scheme (17) (red line) and of AP-MOOD scheme given by Algorithm 1 (blue line) against the exact solution (solid line).

### 5. Application to the isentropic Euler system

We now extend the idea developed in the previous Section to the isentropic Euler system. The TVD-AP scheme reads as

$$W_j^{n+1} = \theta W_j^{n+1,O1} + (1-\theta) W_j^{n+1,O2}, \tag{19}$$

where  $W_j^{n+1,O1}$  is given by the first-order AP scheme (5) while  $W_j^{n+1,O2}$  by the second-order AP scheme (10) and  $\theta$  is fixed equal to  $\theta_M = \beta/(1-\beta)$ . This is enough to assure the TVD and the  $L^\infty$  stability properties. However, since we observed that in many situations the full second order AP scheme (10) can be employed without formation of spurious oscillations, as for the case of the linear advection described before, we aim in constructing a MOOD like technique which permits to interpolate from the full second order to the TVD-AP scheme if needed, producing *de facto* an highly accurate method which is referred to as the AP-MOOD method. Unfortunately, in this case, one can not directly transpose to the Euler case the MOOD approach seen for the advection equation. In fact, in the Euler system, the variables  $\rho$  and q no longer satisfy either the TVD property or the  $L^\infty$  bound in the continuous case. As a consequence, we cannot apply the detection criteria seen before on the conservative variables  $\rho$  and q to get a non oscillating scheme. It turns out that characteristic or Riemann invariants variables constitute a better choice for detecting spurious oscillations since it can be shown that they verify some decoupled non linear advection equations [59]. We denote the Riemann invariants by  $\phi_+$  and  $\phi_-$ . For the isentropic Euler system case, they are given by

$$\phi_{+}(W) := \frac{q}{\rho} - h(\rho), \quad \text{and} \quad \phi_{-}(W) := \frac{q}{\rho} + h(\rho),$$
 (20)

where  $h(\rho)$  is the enthalpy given by  $h(\rho) = \frac{2}{\gamma - 1} \sqrt{\frac{\gamma \rho^{\gamma - 1}}{\varepsilon}}$  if  $\gamma > 1$  and  $h(\rho) = \ln(\rho)$  if  $\gamma = 1$  [59]. Now, since it is known that at continuous level and for a Riemann problem at least one of the two Riemann invariants  $\phi_+$  or  $\phi_-$  satisfy the maximum principle [57], one can think to introduce a MOOD-like detection criterion which relies on testing whether

both Riemann invariants break the maximum principle at the same time. In practice, the following stability detector is used

$$M_{\pm}^{0} = \|\Phi_{\pm}^{0}\|_{\infty},$$
  

$$M_{\pm}^{n} = \max\left(M_{\pm}^{n-1}, \|\Phi_{\pm}^{n}\|_{\infty}\right), \text{ for all } n > 0.$$

Equipped with this detector, the AP-MOOD algorithm for the Euler equations reads as follows

**Algorithm 2.** 1. Compute a candidate solution  $W^{n+1,O2}$  using the second order AP scheme (10).

- 2. Detect if this candidate solution satisfies the maximum principle of the Riemann invariants:  $\|\Phi_{-}^{n+1,02}\|_{\infty} \leq M_{-}^{n}$  and  $\|\Phi_{+}^{n+1,02}\|_{\infty} \leq M_{+}^{n}$ ;
- 3. If these two criteria are satisfied, set  $W^{n+1} = W^{n+1,O2}$ . otherwise, compute  $W^{n+1}$  using the TVD-AP scheme (19).

With this approach, at most one extra computation of the semi-implicit scheme is required to ensure the TVD property and the  $L^{\infty}$  stability.

We now turn to a second-order discretization in space. We present it in the case of one space dimension for the sake of simplicity. To this end, we use classical MUSCL techniques [60], other high order space reconstruction could be employed as well without changing the core of the method. The enunciated discretization works by introducing a linear reconstruction of the conserved variables  $W_i^n$ :

$$\widehat{W}_{i}^{n}(x) = W_{i}^{n} + \sigma_{i}^{n}(x - x_{i}), \tag{21}$$

where  $\sigma_j^n$  can be a limited (if TVD property should be assured) or unlimited slope. The case of unlimited slope is used in combination with the second order in time AP scheme (10) and it is given by

$$\sigma_{j}^{n} = \frac{1}{2} \left( \frac{W_{j}^{n} - W_{j-1}^{n}}{\Delta x} + \frac{W_{j+1}^{n} - W_{j}^{n}}{\Delta x} \right). \tag{22}$$

This gives rise to a genuine second order in time and space Asymptotic Preserving method which however does not enjoy the TVD and  $L^{\infty}$  property. On the other hand, the limited slope for which the minmod limiter is used, is employed together with AP-TVD scheme in time (19). In this case we have

$$\sigma_j^n = \operatorname{minmod}\left(\frac{W_j^n - W_{j-1}^n}{\Delta x}, \frac{W_{j+1}^n - W_j^n}{\Delta x}\right),\tag{23}$$

where the minmod function is given by:

$$\mathsf{minmod}(a,b) = \left\{ \begin{array}{ll} \min(a,b) & \text{if } a > 0 \text{ and } b > 0, \\ \max(a,b) & \text{if } a < 0 \text{ and } b < 0, \\ 0 & \text{otherwise.} \end{array} \right.$$

The above combination of space and time discretization gives rise to a TVD-AP highly accurate space and time discretization of the Euler equation. The reconstruction of variables W (21) is used for defining the numerical flux functions at the interfaces

$$W_{j,\pm}^n := \widehat{W}_j^n \left( x_j \pm \frac{\Delta x}{2} \right) = W_j^n \pm \frac{\Delta x}{2} \sigma_j^n, \tag{24}$$

and thus, the explicit numerical flux function  $\mathcal{F}_{e}$  becomes

$$(\mathcal{F}_e)_{j+\frac{1}{2}}^n := \frac{F_e(W_{j,+}^n) + F_e(W_{j+1,-}^n)}{2} + (\mathcal{D}_e)_{j+\frac{1}{2}}^n (W_{j+1,-}^n - W_{j,+}^n), \tag{25}$$

with  $(\mathcal{D}_e)_{j+\frac{1}{2}}^n := \max(|u_{j,+}^n|, |u_{j+1,-}^n|)$ , while the implicit numerical flux function  $\mathcal{F}_i$  becomes

$$(\mathcal{F}_{i})_{j+\frac{1}{2}}^{n,n+1} := \frac{F_{i}(W_{j,+}^{n,n+1}) + F_{i}(W_{j+1,-}^{n,n+1})}{2} + (\mathcal{D}_{i})_{j+\frac{1}{2}}^{n} (\tilde{W}_{j+1,-}^{n+1} - \tilde{W}_{j,+}^{n+1}), \tag{26}$$

where  $W_{j,+}^{n,n+1} = (\rho_{j,+}^{n+1}, q_{j,+}^n)$  and where  $(\mathcal{D}_i)_{j+\frac{1}{2}}^n$  is the implicit viscosity coefficient, taken as half of the maximum implicit eigenvalue and given by

$$(\mathcal{D}_i)_{j+\frac{1}{2}}^n := \frac{1}{2} \max \left( \sqrt{\frac{p'(\rho_{j,+}^n)}{\varepsilon}}, \sqrt{\frac{p'(\rho_{j+1,-}^n)}{\varepsilon}} \right), \tag{27}$$

and

$$\tilde{W}_{j,\pm}^{n+1} = W_j^{n+1} \pm \frac{\Delta x}{2} \sigma_j^n. \tag{28}$$

The second-order MUSCL extension in space is thus complete.

### 6. Numerical experiments

The schemes described in the previous parts are resumed and labeled below.

- The first-order AP scheme is given by (5), (6), (7).
- The <u>second-order AP scheme</u> is given by (10), (25), (26) corresponding to the ARS(2,2,2) time discretization, supplemented with the unlimited linear MUSCL space reconstruction.
- The <u>TVD-AP</u> scheme is given by (19) with  $\theta = \beta/(1-\beta)$  corresponding to convex combination of the first-order AP scheme and of the second-order AP scheme with (25), (26) and limiter (23) for the space discretization.
- The <u>AP-MOOD scheme</u> corresponds to the procedure detailed in Algorithm 2 with (25), (26) and limiter (23) for the space discretization.

In addition, for all the schemes the time step is constrained by the uniform C.F.L. condition

$$\Delta t \le C \frac{\Delta x}{\Lambda}, \quad \text{where } \Lambda = \max_{j} \left( 2 |u_{j}^{n}| \right),$$

with C = 0.9 for the first-order scheme and C = 0.45 for the other three schemes. Note that this restrictive C.F.L. (for the second-order AP, TVD-AP and AP-MOOD schemes) is uniform and is only due to the second order discretization in space. In the following, we first consider a Riemann problem and successively perform an assessment of the order of accuracy of the scheme using a smooth solution in one space dimension. Afterwards, we validate the scheme on a more complex test case and we verify its asymptotic stability again in one space dimension. Finally, we propose two two-dimensional numerical experiments in which we compare our scheme with a reference solution.

# 6.1. One dimensional shock tube

On the space domain [0, 1], we consider  $\gamma = 1.4$  and a Riemann problem with the following initial data:

$$\rho(0, x) = \begin{cases} 1 + \varepsilon & \text{if } x < 0.5, \\ 1 & \text{otherwise;} \end{cases} \qquad q(0, x) = 1.$$

Homogeneous Neumann boundary conditions are prescribed on each boundary. We compare the results from the three schemes in several regimes corresponding to different values of the Mach number:  $\varepsilon = 1$ ,  $\varepsilon = 10^{-2}$  and  $\varepsilon = 10^{-4}$ . The results are displayed in Figure 4 for  $\varepsilon = 1$  and N = 50, in Figure 5 for  $\varepsilon = 10^{-2}$  and N = 125 and in Figure 6 for  $\varepsilon = 10^{-4}$  and N = 500 on the left for the density and on the right for the momentum. As expected, the first-order AP scheme (dotted line) is very diffusive. On the other hand, the second-order scheme (dashed line) yields a better approximation of the intermediate states. However, overshoots and undershoots appear at the heads and tails of the rarefaction wave, and near the shock wave. The TVD-AP scheme (blue line) corrects both of these shortcomings. Finally, thanks to the MOOD procedure, the AP-MOOD scheme (blue dashed line) yields better results than the TVD-AP scheme. In conclusion, the oscillatory nature of the second-order AP scheme is removed by the TVD-AP and AP-MOOD schemes at the cost of an expected slightly increased diffusion.

![](_page_13_Figure_0.jpeg)

Figure 4: Shock tube with  $\varepsilon = 1$  and 50 discretization cells; results displayed at time  $t_{end} = 0.125$ .

![](_page_13_Figure_2.jpeg)

Figure 5: Shock tube with  $\varepsilon = 10^{-2}$  and 125 discretization cells, displayed at time  $t_{end} = 0.02$ .

![](_page_13_Figure_4.jpeg)

Figure 6: Shock tube with  $\varepsilon = 10^{-4}$  and 500 discretization cells, displayed at time  $t_{end} = 0.0025$ .

## 6.2. Order of accuracy assessment in one dimension

We consider a smooth solution from [63] with the following initial data

$$\rho(0,x) = 1 - \frac{\varepsilon}{2} \omega \left( \frac{2}{0.25} \left( x - \frac{1}{2} \right) \right) \quad \text{and} \quad u(0,x) = 1 + \frac{\varepsilon}{2} \omega \left( \frac{2}{0.25} \left( x - \frac{1}{2} \right) \right),$$

on the space domain [0, 1] and where the function  $\omega$  is given by  $\omega(z) = \left(\frac{2-|z|}{2}\right)^4 (1+2|z|)$  if  $|z| \le 2$ , and 0 otherwise. If  $\gamma = 3$ , see [63], both Riemann invariants are solution to the following Burgers equations:

$$\begin{cases} \partial_t \phi_+ + \phi_+ \partial_x \phi_+ = 0, \\ \partial_t \phi_- + \phi_- \partial_x \phi_- = 0. \end{cases}$$

Solving the above system requires a nonlinear equation solver, such as Newton's method.

For small enough time, the exact solution  $(\rho, q)$  is as smooth as the initial data. We use it to determine the Dirichlet boundary conditions for the four schemes and to compute the errors between the approximate solutions and the exact solution. We measure the  $L^{\infty}$  errors for the density and the momentum

$$e_{\infty}^{n}(\rho) = \max_{i} \left| \rho_{j}^{n} - (\rho_{ex})_{j}^{n} \right|, \ e_{\infty}^{n}(q) = \max_{i} \left| q_{j}^{n} - (q_{ex})_{j}^{n} \right|$$

where  ${}^t((\rho_{ex})_j^n, (q_{ex})_j^n)$  is the exact solution at time  $t^n$  in the cell of center  $x_j$ . The time at which the errors are computed are  $t_{end}=0.007$  for  $\varepsilon=1$ ,  $t_{end}=0.005$  for  $\varepsilon=10^{-2}$  and  $t_{end}=0.0005$  for  $\varepsilon=10^{-4}$ . For the four schemes, the density and momentum  $L^\infty$  errors are displayed in Figure 7 in logarithmic scale with respect to the number of discretization cells. For all values of  $\varepsilon$ , the first-order AP scheme and the second-order AP scheme are respectively of order 1 and 2, as expected. In addition, we note that the TVD-AP scheme is also numerically first-order accurate or barely larger but with an  $L^\infty$  error which is always lower than the one of the first order method. The AP-MOOD for  $\varepsilon \in \{10^{-2}, 10^{-4}\}$  is numerically of order two in spite of the slope limiter. For  $\varepsilon=1$ , the AP-MOOD scheme is of order more than one but less than two with an  $L^\infty$  error always smaller than the one of the first order method.

#### 6.3. Validation and asymptotic stability in one dimension

We now consider the problem introduced in Degond and Tang [17]. It consists in several interacting Riemann problems. The initial data are given on the space domain [0, 1] by

$$\rho(0,x) = \begin{cases} 2 & \text{if } x \in [0,0.2], \\ 2+\varepsilon & \text{if } x \in (0.2,0.3], \\ 2 & \text{if } x \in (0.3,0.7], \\ 2-\varepsilon & \text{if } x \in (0.7,0.8), \\ 2 & \text{if } x \in [0.8,1], \end{cases} \text{ and } q(0,x) = \begin{cases} 1-\varepsilon/2 & \text{if } x \in [0,0.2], \\ 1 & \text{if } x \in (0.2,0.3], \\ 1+\varepsilon/2 & \text{if } x \in (0.3,0.7], \\ 1 & \text{if } x \in (0.7,0.8), \\ 1-\varepsilon/2 & \text{if } x \in [0.8,1]. \end{cases}$$

supplemented by periodic boundary conditions. We choose  $\gamma=1.4$ . Here, the goal is to validate the proposed schemes in both the compressible and the incompressible regimes. The reference solution is computed with the first order-AP scheme on a refined mesh in space and time. Figure 8 reports the results for the density on the left and the momentum on the right for  $\varepsilon=1$  and N=100 discretization cells with final time  $t_{end}=0.075$ . In the top panels of Figure 9, we display the solution for  $\varepsilon=10^{-4}$  and  $t_{end}=0.0015$  obtained with 1500 cells. In the bottom panels, we have refined the space-time mesh to study the convergence of the numerical approximations.

As in the previous case, the first-order AP scheme is very diffusive and it smears out all shock waves. The second-order AP scheme yields a less diffusive approximation, but it is not TVD because of overshoots and oscillations, while the TVD-AP and the AP-MOOD scheme decrease the diffusion, and therefore greatly improve the numerical approximation compared to the first-order AP scheme.

In Figure 9, we observe that while the first-order AP scheme projects the approximate solution onto the incompressible limit and avoids computing the small structures and the fast waves present in the reference solution, the second order and the AP-MOOD scheme appropriately capture the micro-structure of the solution, still allowing for much larger time steps. Therefore, if one is interested into the small structures close to the incompressible limit, then high accurate numerical schemes seem to be highly relevant.

![](_page_15_Figure_0.jpeg)

Figure 7: Error lines. From top to bottom:  $\varepsilon = 1$ ,  $\varepsilon = 10^{-2}$  and  $\varepsilon = 10^{-4}$ . Left panels: density errors; right panels: momentum errors. Black line corresponds to the first order AP scheme, blue line to the TVD-AP scheme, red line to the AP-MOOD scheme, yellow line to the second order AP unlimited.

# 6.4. Order of accuracy in two dimensions

We measure the order of accuracy of the scheme in two space dimensions. We consider  $\gamma = 1$  and the following smooth exact solution of the isentropic Euler system (1):

$$\begin{split} & \rho_{ex}(x,y,t) = \rho_{\infty} - \frac{a^2 \varepsilon}{8d} e^{2d(b-\bar{r}(x,y,t)^2)}, \\ & u_{ex}(x,y,t) = u_{\infty} + a\bar{y}(t) \sqrt{\frac{\gamma}{2}} e^{d(b-\bar{r}(x,y,t)^2)} \left( \rho_{ex}(x,y,t) \right)^{\frac{\gamma}{2}-1}, \\ & v_{ex}(x,y,t) = v_{\infty} - a\bar{x}(t) \sqrt{\frac{\gamma}{2}} e^{d(b-\bar{r}(x,y,t)^2)} \left( \rho_{ex}(x,y,t) \right)^{\frac{\gamma}{2}-1}, \end{split}$$

where we have set  $\bar{r}(x, y, t)^2 = \bar{x}(t)^2 + \bar{y}(t)^2$ , with  $\bar{x}(t) = x - x_0 - u_\infty t$ ,  $\bar{y}(t) = y - y_0 - v_\infty t$ . This exact solution corresponds to a vortex initially centered at  $^t(x_0, y_0)$  and moving with the phase velocity  $^t(u_\infty, v_\infty)$ . For the numerical simulation,

![](_page_16_Figure_0.jpeg)

Figure 8: Degond-Tang experiment with  $\varepsilon=1$  and 100 discretization cells at time t=0.075 and N=100 mesh points. Density left image, momentum right image.

![](_page_16_Figure_2.jpeg)

Figure 9: Degond-Tang experiment with  $\varepsilon = 10^{-4}$  at time t = 0.0015 and N = 1500 points for the top panels and N = 3600 for the bottom panels. Density left images, momentum right images.

we take  $\rho_{\infty} = 1$ , a = 1, b = 0, d = 2,  $x_0 = 0$ ,  $y_0 = 0$  and  ${}^t(u_{\infty}, v_{\infty}) = {}^t(1,0)$ . The space domain is  $[-1.5, 2.5] \times [-2, 2]$ . The simulations are carried out for three different values of the squared Mach number  $\varepsilon$  ( $\varepsilon = 1$ ,  $\varepsilon = 10^{-2}$ ,  $\varepsilon = 10^{-4}$ ) until the final physical time  $t_{end} = 1$ . To assess the numerical order of accuracy, we compute the following  $L^{\infty}$  errors for several uniform meshes containing:

$$e_{\infty}^{n}(\rho) = \max_{j,k} \left| \rho_{j,k}^{n} - (\rho_{ex})_{j,k}^{n} \right|, \quad \text{and} \quad e_{\infty}^{n}(\rho U) = \max_{j,k} \left| \left( \rho \sqrt{u^{2} + v^{2}} \right)_{j,k}^{n} - \left( \rho_{ex} \sqrt{u_{ex}^{2} + v_{ex}^{2}} \right)_{j,k}^{n} \right|.$$

For the four schemes, the errors are collected in Table 2 for  $\varepsilon = 1$ , Table 3 for  $\varepsilon = 10^{-2}$  and Table 4 for  $\varepsilon = 10^{-4}$ . In addition, we display error lines in Figure 10.

|       | 1st-order AP             |       | TVD-AP                   |       | 2nd-order AP             |       | AP-MOOD                  |       |
|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|
| N     | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order |
| 625   | 4.30e-02                 | _     | 1.93e-02                 | _     | 8.84e-03                 | _     | 1.04e-02                 |       |
| 2500  | 3.36e-02                 | 0.35  | 6.05e-03                 | 1.67  | 1.66e-03                 | 2.41  | 2.14e-03                 | 2.28  |
| 10000 | 2.20e-02                 | 0.61  | 2.08e-03                 | 1.54  | 2.87e-04                 | 2.53  | 6.31e-04                 | 1.76  |
| 40000 | 1.30e-02                 | 0.76  | 7.63e-04                 | 1.45  | 5.63e-05                 | 2.35  | 1.80e-04                 | 1.81  |
| N     | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order |
| 625   | 1.07e-01                 | _     | 4.61e-02                 | _     | 1.62e-02                 | _     | 2.26e-02                 |       |
| 2500  | 7.59e-02                 | 0.50  | 1.25e-02                 | 1.88  | 3.02e-03                 | 2.42  | 4.40e-03                 | 2.36  |
| 10000 | 4.73e-02                 | 0.68  | 5.19e-03                 | 1.27  | 5.33e-04                 | 2.50  | 1.47e-03                 | 1.59  |
| 40000 | 2.69e-02                 | 0.81  | 2.54e-03                 | 1.03  | 1.09e-04                 | 2.29  | 4.84e-04                 | 1.60  |

Table 2: Density and momentum norm errors and order of accuracy with  $\varepsilon = 1$  for the four schemes.

|       | 1st-order AP             |       | TVD-AP                   |       | 2nd-order AP             |       | AP-MOOD                  |       |
|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|
| N     | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order |
| 625   | 5.58e-04                 | _     | 3.57e-04                 | _     | 1.57e-04                 | _     | 2.46e-04                 |       |
| 2500  | 5.16e-04                 | 0.11  | 1.41e-04                 | 1.34  | 3.31e-05                 | 2.25  | 4.49e-05                 | 2.46  |
| 10000 | 4.20e-04                 | 0.30  | 4.94e-05                 | 1.52  | 4.68e-06                 | 2.82  | 1.68e-05                 | 1.42  |
| 40000 | 3.02e-04                 | 0.48  | 1.55e-05                 | 1.67  | 6.33e-07                 | 2.89  | 4.37e-06                 | 1.94  |
| N     | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order |
| 625   | 1.51e-01                 | _     | 7.79e-02                 | _     | 3.19e-02                 | _     | 3.88e-02                 |       |
| 2500  | 1.28e-01                 | 0.24  | 2.84e-02                 | 1.46  | 6.04e-03                 | 2.40  | 6.81e-03                 | 2.51  |
| 10000 | 9.52e-02                 | 0.43  | 9.35e-03                 | 1.60  | 8.50e-04                 | 2.83  | 1.38e-03                 | 2.30  |
| 40000 | 6.31e-02                 | 0.59  | 2.81e-03                 | 1.74  | 1.15e-04                 | 2.89  | 4.57e-04                 | 1.59  |

Table 3: Density and momentum norm errors and order of accuracy with  $\varepsilon = 10^{-2}$  for the four schemes.

|       | 1st-order AP             |       | TVD-AP                   |       | 2nd-order AP             |       | AP-MOOD                  |       |
|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|--------------------------|-------|
| N     | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order | $e_{\infty}^{n}(\rho)$   | order |
| 625   | 2.42e-05                 | _     | 1.12e-05                 | _     | 5.32e-06                 | _     | 6.33e-06                 |       |
| 2500  | 2.21e-05                 | 0.13  | 1.27e-05                 | -0.18 | 1.75e-06                 | 1.60  | 1.79e-06                 | 1.82  |
| 10000 | 1.17e-05                 | 0.91  | 2.97e-06                 | 2.10  | 8.31e-07                 | 1.08  | 7.88e-07                 | 1.19  |
| 40000 | 9.33e-06                 | 0.33  | 2.06e-06                 | 0.53  | 1.19e-07                 | 2.80  | 1.18e-07                 | 2.74  |
| N     | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order | $e_{\infty}^{n}(\rho U)$ | order |
| 625   | 1.61e-01                 | _     | 8.81e-02                 | _     | 3.74e-02                 | _     | 4.43e-02                 |       |
| 2500  | 1.43e-01                 | 0.16  | 4.40e-02                 | 1.00  | 8.76e-03                 | 2.09  | 9.17e-03                 | 2.27  |
| 10000 | 1.17e-01                 | 0.29  | 1.72e-02                 | 1.36  | 1.65e-03                 | 2.41  | 1.75e-03                 | 2.39  |
| 40000 | 8.59e-02                 | 0.45  | 5.69e-03                 | 1.59  | 3.06e-04                 | 2.43  | 3.05e-04                 | 2.52  |

Table 4: Density and momentum norm errors and order of accuracy with  $\varepsilon=10^{-4}$  for the four schemes.

![](_page_18_Figure_0.jpeg)

Figure 10: Error lines for the 2D steady vortex. From top to bottom:  $\varepsilon = 1$ ,  $\varepsilon = 10^{-2}$  and  $\varepsilon = 10^{-4}$ . Left panels: density errors; right panels: momentum errors. Black line corresponds to the first order AP scheme, blue line to the TVD-AP scheme, red line to the AP-MOOD scheme, yellow line to the second order AP unlimited.

From these results, we draw similar conclusions as in the 1D case. The errors of TVD-AP and the AP-MOOD schemes lie in between the first and the second order slopes as well as the errors.

### 6.5. Asymptotically consistency of the schemes in two dimensions

We now perform a 2D validation experiment, initially described in [64] and used more recently in [8]. It is particularly relevant since, for small values of  $\varepsilon$  we can compare the compressible numerical approximations to an approximate solution of the incompressible Euler equations. In this way, we can measure the asymptotically consistency of our approximations in the low Mach number limit.

The initial data are well-prepared. Indeed, on the space domain  $[0, 2\pi]^2$ , we take a constant density  $\rho = \pi/15$ , and the initial incompressible velocity field U = (u, v) is given by:

$$u(x, y, 0) = \begin{cases} \tanh((y - \pi/2)/\rho), & \text{if } y \le \pi, \\ \tanh((3\pi/2 - y)/\rho), & \text{otherwise,} \end{cases}$$
  $v(x, y, 0) = 0.05 \sin(x).$ 

In addition, we take  $\gamma = 1$  and we prescribe periodic boundary conditions for the compressible Euler system.

To determine the incompressible approximate solution, we consider the vorticity formulation of the incompressible Euler system, given by:

$$\partial_t \omega + U \cdot \nabla \omega = 0, \tag{29}$$

and we recall that the vorticity  $\omega$  is given by  $\omega = \partial_x v - \partial_y u$ . Since  $\nabla \cdot U = 0$ , there exists a stream function  $\psi$  such that  $U = {}^t(\partial_y \psi, -\partial_x \psi)$  and  $-\Delta \psi = \omega$ . From these observations, we can obtain a reference solution. We compute the time evolution of the vorticity by repeating the following three steps: we first compute the stream function  $\psi$ , then the associated velocity field, and finally the time update of the vorticity with (29). To solve the Poisson equation  $-\Delta \psi = \omega$ , we use a classical discretization of the Laplace operator, and we prescribe periodic boundary conditions. Since this leads to a singular system, we also impose that the stream function has a null average. This does not alter the rest of the procedure since we are only interested in the derivatives of  $\psi$ . The velocity is then obtained by an application of a centered gradient discretization, and an upwind finite difference scheme provides an approximate solution for (29). Periodic boundary conditions are prescribed in both of these steps.

We stress that the reference solution is obtained from the incompressible Euler equations while the schemes under consideration approximately solve the compressible Euler system with a very small Mach number. The results are given in Figure 11. We take  $\varepsilon = 10^{-5}$  and  $t_{end} = 6$  to compare the numerical solutions provided by the four schemes for the compressible Euler system with the reference incompressible solution. The mesh is constituted of  $200 \times 200$  cells.

We can see that the first-order scheme loses the main structure of the solution (it is worth noting that, on a finer grid, the structure can be captured by the first-order scheme). The limited TVD-AP scheme provides a smeared numerical approximation, while the TVD-AP and second-order schemes yield similar numerical solutions. We note that, for these schemes, the main structure of the solution is captured. However, the small central structures are smeared because the grid is too coarse. Overall, the proposed compressible schemes offer a convincing approximation of the incompressible solution when  $\varepsilon$  is small enough.

Now, we use this test case in the  $\varepsilon = 1$ . The numerical solution given by the four schemes at the final time  $t_{end} = 10$  is compared in Figure 12 to a reference solution given by the first-order AP scheme with a very fine mesh. The reference solution is obtained by using the first-order scheme with  $400 \times 400$  cells. In this figure, we represent the vorticity of the solution, given by  $\omega = \partial_x v - \partial_y u$ , since this quantity is relevant for small  $\varepsilon$ .

In Figure 12 are plotted the results when a coarse mesh made of  $40 \times 40$  cells is employed. We note that the main structures of the reference solution are captured by the second-order and the AP-MOOD scheme, with the TVD-AP scheme being slightly more diffusive. However, the first-order scheme is so diffusive that it destroys most of the structures. In the bottom left corner of Figure 12, we have added a zoom on the domain [2.35, 3.85]  $\times$  [4.4, 5] of the reference and the second-order solutions. We note that the very fine structure present in this domain is smeared by the second-order scheme due to the use of too coarse a mesh.

#### 7. Conclusion

In this paper we have derived a new second order scheme for the compressible Euler equations in the low Mach number regime. Since, non physical oscillations cannot be avoided for C.F.L. conditions larger than the one imposed by explicit methods, we have constructed a new method based on the coupling of first order with second order in time and space schemes. This approach has permitted to get an highly accurate and asymptotic preserving scheme which additionally enjoys the TVD and  $L^{\infty}$  property independently from the time step. Successively, the introduction of limiters has permitted a passage from the second to the TVD method only when strictly necessary further improving the overall accuracy. For all the schemes presented, the stability constraints did not depend on the Mach number value and these schemes degenerate into a consistent highly accurate discretization of the incompressible system in the low Mach limit. Numerical experiments supported the proposed analysis. In the future, we aim in focusing on the generalization of such technique to the case of TVD schemes which couple AP schemes of order higher than two with first order in time AP methods. Moreover, we aim in exploring more in depth the use of limiters since in some cases some small oscillations remain present for the limited method. Local coupling techniques between the different order schemes can also largely improve the results obtained and they are now the subject of investigations. Extension to the full Euler equations are under study.

![](_page_20_Figure_0.jpeg)

Figure 11: Numerical solution for the double shear layer experiment with ε <sup>=</sup> <sup>10</sup>−<sup>5</sup> , using 200 × 200 = 40000 cells.

# References

- [1] T. Alazard, Incompressible limit of the nonisentropic Euler equations with the solid wall boundary conditions. Adv. Differential Equations, 10(1), 19–44, 2005.
- [2] K. Asano, On the incompressible limit of the compressible Euler equation. Japan J. Appl. Math. 4, 455–488, 1987.
- [3] U. M. Ascher, S. J. Ruuth, R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations. Appl. Numer. Math. 25(2-3) 151–167, 1997, Special issue on time integration (Amsterdam, 1996).
- [4] G. Bispen, M. Luka´cov ˇ a-Medvid'ov ´ a, L. Yelash, Asymptotic preserving IMEX finite volume schemes for low Mach number Euler equations ´ with gravitation, J. Comput. Phys. 335, 222–248, 2017.
- [5] M. Boger, F. Jaegle, R. Klein, C.-D. Munz, Coupling of compressible and incompressible flow regions using the multiple pressure variables approach. Math. Methods Appl. Sci. 38 (2015), no. 3, 458-477.
- [6] S. Boscarino, G. Russo, On a class of uniformly accurate IMEX Runge-Kutta schemes and applications to hyperbolic systems with relaxation, SIAM J. Sci. Comput. 31(3), 1926–1945, 2009.
- [7] S. Boscarino, G. Russo, Flux-explicit IMEX Runge-Kutta schemes for hyperbolic to parabolic relaxation problems, SIAM J. Numer. Anal. 51(1), 163–190, 2013.
- [8] S. Boscarino, G. Russo, L. Scandurra, All Mach Number Second Order Semi-Implicit Scheme for the Euler Equations of Gasdynamics, preprint.
- [9] C. Bresten, S. Gottlieb, Z. Grant, D. Higgs, D. I. Ketcheson, A. Nmeth, Explicit strong stability preserving multistep Runge–Kutta methods. Math. Comp. 86, 747–769, 2017.
- [10] C. Chalons, M. Girardin, S. Kokh, Large time step and asymptotic preserving numerical schemes for the gas dynamics equations with source terms, SIAM J. Sci. Comput. 35, 2874–2902, 2013.
- [11] C. Chalons, M. Girardin, S. Kokh, An all-regime Lagrange-Projection like scheme for the gas dynamics equations on unstructured meshes, Communications in Computational Physics (CICP), 20(1), 188–233, 2016.

![](_page_21_Figure_0.jpeg)

Figure 12: Numerical solution for the double shear layer experiment with ε <sup>=</sup> 1. The numerical solutions are made with 40 <sup>×</sup> <sup>40</sup> <sup>=</sup> 1600 cells. The bottom left of this figure contains a zoom on a very small structure smeared by the coarse mesh but present in the reference solution. The color legend in the bottom left corner is only applied to this zoom.

- [12] S. Clain, S. Diot, R. Loubere A high-order finite volume method for systems of conservation laws—Multi-dimensional Optimal Order ` Detection (MOOD), J. Comput. Phys. 230(10), 4028–4050, 2011.
- [13] Ph. Colella, K. Pao, A projection method for low speed flows, J. Comp. Phys. 149, 245–269, 1999.
- [14] E. M. Constantinescu, A. Sandu, Optimal explicit strong-stability-preserving general linear methods, SIAM J. Sci. Comput. 32, 3130–3150, 2010.
- [15] F. Cordier, P. Degond, A. Kumbaro, An Asymptotic-Preserving all-speed scheme for the Euler and Navier Stokes equations, J. Comp. Phys. 231, 5685–5704, 2012.
- [16] P. Degond, F. Deluzet, A. Sangam, M.-H. Vignal, An Asymptotic Preserving Scheme for the Euler equations in a strong magnetic field, Comp. Phys. 228, 3540–3558, 2009.
- [17] P. Degond, M. Tang, All speed scheme for the low Mach number limit of the isentropic Euler equations. Commun. Comput. Phys. 10(1), 1–31, 2011.
- [18] P. Degond, S. Jin, J.-G. Liu, Mach-number uniform asymptotic-preserving gauge schemes for compressible flows. Bull. Inst. Math. Acad. Sin. (N.S.) 2 (2007), no. 4, 851-892.
- [19] S. Dellacherie, Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number, J. Comp. Phys. 229, 978-1016, 2010.
- [20] S. Dellacherie, J. Jung, P. Omnes, Preliminary results for the study of the Godunov Scheme Applied to the Linear Wave Equation with Porosity at Low Mach Number, ESAIM ProcS. 52, 105-126, 2015.
- [21] G. Dimarco, R. Loubere, M.-H. Vignal, Study of a new asymptotic preserving scheme for the Euler system in the low Mach number limit, `

- SIAM J. Sci. Comput., 39(5), A2099–A2128, 2017.
- [22] G. Dimarco, L. Pareschi, Asymptotic-preserving IMEX Runge-Kutta methods for nonlinear kinetic equations, SIAM J. Num. Anal. 1064– 1087, 2013.
- [23] G. Dimarco, L. Pareschi, High order asymptotic preserving schemes for the Boltzmann equation, Comptes Rendus Mathematique 350, 9, 481–486, 2012.
- [24] S. Diot, S. Clain, R. Loubere, Improved detection criteria for the multi-dimensional optimal order detection (MOOD) on unstructured meshes ` with very high-order polynomials, Comput. & Fluids, 64, 43–63, 2012.
- [25] S. Diot, R. Loubere, and S. Clain, The multidimensional optimal order detection method in the three-dimensional case: very high-order finite ` volume method for hyperbolic systems, Internat. J. Numer. Methods Fluids, 73(4), 362–392, 2013.
- [26] R. Eymard, T. Gallouet, R. Herbin, Finite volume methods, Handbook of numerical analysis, Vol. VII, 713–1020, North-Holland, Amsterdam, ¨ 2000.
- [27] L. Ferracina, M. N. Spijker, Stepwize restrictions for the total-variation-diminishing property in general Runge-Kutta mathods, SIAM J. Numer. Anal. 42, 1073–1093, 2004.
- [28] N. Grenier, J.-P. Vila, P. Villedieu, An accurate low-Mach scheme for a compressible two-fluid model applied to free-surface flows, J. Comp. Phys. 252, 1–19, 2013.
- [29] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM Rev. 43(1), 89–112, 2001.
- [30] H. Guillard, A. Murrone, On the behavior of upwind schemes in the low Mach number limit : II. Godunov type schemes, Comp. & Fluids, 33, 655–675, 2004.
- [31] H. Guillard, C. Viozat, On the behavior of upwind schemes in the low Mach limit, Comp. & Fluid, 28, 63–86, 1999.
- [32] J. Haack, S. Jin, J.G. Liu, An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations, Commun. Comput. Phys., 12, 955–980, 2012.
- [33] F. H. Harlow, and A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, J. Comput. Phys, 8, 197–213, 1971.
- [34] D. R. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Comp. & Fluids, 32 (2003), pp. 1113-1132.
- [35] I. Higueras, N. Happenhofer, O. Koch, F. Kupka, Optimized strong stability preserving IMEX Runge-Kutta methods. J. Comput. Appl. Math. 272 (2014), 116-140.
- [36] I. Higueras, Representations of Runge-Kutta methods and strong stability preserving methods, SIAM J. Numer. Anal., 43, 924–948, 2005.
- [37] R.I. Issa, A.D.Gosman, A.P. Watkins, The computation of compressible and incompressible flow of fluid with a free surface, Phys. Fluids, 8, 2182–2189, 1965.
- [38] D. I. Ketcheson, C. B. Macdonald, S. Gottlieb Optimal implicit strong stability preserving Runge-Kutta methods. Appl. Numer. Math. 59 (2009), no. 2, 373-392.
- [39] W. Kheriji, R. Herbin, J.-C. Latche, Pressure correction staggered schemes for barotropic one-phase and two-phase flows, Comp. & Fluids, ´ 88, 524–542, 2013.
- [40] S. Klainerman, A. Majda, Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids, Comm. Pure Appl. Math. 34(4), 481–524, 1981.
- [41] S. Klainerman, A. Majda, Compressible and incompressible fluids, Comm. Pure Appl. Math. 35(5), 629–651, 1982.
- [42] R. Klein, Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: One-dimensional flow, J. Comp. Phys., 121, 213–237, 1995.
- [43] R. J. LeVeque, Finite volume methods for hyperbolic problems, Cambridge Texts in Applied Mathematics. Cambridge University Press, Cambridge, 2002.
- [44] P.-L. Lions, N. Masmoudi, Incompressible limit for a viscous compressible fluid, J. Math. Pures Appl. (9), 77(6), 585–627, 1998.
- [45] G. Metivier, S. Schochet. The incompressible limit of the non-isentropic Euler equations. Arch. Ration. Mech. Anal., 158(1), 61–90, 2001. ´
- [46] V. Michel-Dansac, G. Dimarco, R. Loubere, and M.-H. Vignal, Total Variation Diminishing implicit-explicit schemes, Work in progress, ` 2017.
- [47] C. D. Munz, S. Roller, R. Klein,K. J. Geratz, The extension of incompressible flow solvers to the weakly compressible regime, Comp. Fluid, 32, 173–196, 2002.
- [48] C. D. Munz, M. Dumbser, S. Roller, Linearized acoustic perturbation equations for low Mach number flow with variable density and temperature, J. Comput. Phys., 224, 352–364, 2007.
- [49] S. Noelle, G. Bispen, K.R. Arun, M. Luka´cov ˇ a-Medvid'ov ´ a, C. D. Munz, A weakly asymptotic preserving low Mach number scheme for the ´ Euler equations of gas dynamics. SIAM J. Sci. Comput. 36 (2014), no. 6, B989-B1024.
- [50] J. H. Park, C. D. Munz, Multiple pressure variables methods for fluid flow at all Mach numbers, Int. J. Numer. Meth. Fluid, 49, 905–931, 2005.
- [51] H. Paillere, C. Viozat, A. Kumbaro, I. Toumi, Comparison of low mach number models for natural convection problems. Heat & Mass Tran., ` 36, 567–573, 2000.
- [52] L. Pareschi, G. Russo, Implicit-explicit Runge-Kutta schemes for stiff systems of differential equations, In Recent trends in numerical analysis, volume 3 of Adv. Theory Comput. Math. pages 269–288. Nova Sci. Publ., Huntington, NY, 2001.
- [53] L. Pareschi, G. Russo, Implicit-Explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation, J. Sci. Comput. 25(1-2), 129–155, 2005.
- [54] S. V. Patankar, Numerical heat transfer and fluid flow, New York: McGraw-Hill, 1980.
- [55] V.V. Rusanov, Calculation of interaction of non-steady shock waves with obstacles. J. Comput. Math. Phys. USSR 1, 267–279, 1961.
- [56] S. Schochet. The compressible Euler equations in a bounded domain: existence of solutions and the incompressible limit. Comm. Math. Phys., 104(1), 49–75, 1986.
- [57] J. A. Smoller, J. L. Johnson. Global solutions for an extended class of hyperbolic systems of conservation laws, Arch. Rational Mech. Anal. 32, 169–189, 1969.
- [58] H. Song, Energy SSP-IMEX Runge-Kutta methods for the Cahn-Hilliard equation. J. Comput. Appl. Math. 292 (2016), 576-590.

- [59] E. F. Toro, Riemann solvers and numerical methods for fluid dynamics. A practical introduction, Springer-Verlag, Berlin, third edition, 2009.
- [60] B. van Leer, Towards the Ultimate Conservative Difference Scheme, V. A Second Order Sequel to Godunov's Method, J. Com. Phys. 32, 101–136, 1979.
- [61] M. Tang, Second order all speed method for the isentropic Euler equations. Kinet. Relat. Models 5 (2012), no. 1, 155-184.
- [62] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, J. Comp. Phys., 72, 277–298, 1987.
- [63] F. Vilar, P.-H. Maire, R. Abgrall, Cell-centered discontinuous Galerkin discretizations for two-dimensional scalar conservation laws on unstructured grids and for one-dimensional Lagrangian hydrodynamics, Comput. & Fluids, 46, 498–504, 2011.
- [64] E. Weinan, C.-W. Shu, A Numerical Resolution Study of High Order Essentially Non-oscillatory Schemes Applied to Incompressible Flow, J. Comput. Phys. 110, 39–46, 1994.