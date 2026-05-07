# A robust implicit-explicit acoustic-transport splitting scheme for two-phase flows

S. Peluchona,c, G. Gallice<sup>a</sup> , L. Mieussensb,c

*<sup>a</sup>CEA-CESTA, 15 avenue des sablières - CS 60001, 33116 Le Barp Cedex, France, <sup>b</sup>Univ. Bordeaux, CNRS, Bordeaux INP, IMB, UMR 5251, F-33400 Talence, c INRIA, F-33400 Talence, France.*

### **Abstract**

In this paper, a splitting strategy to simulate compressible two-phase flows using the five-equation model is presented. The main idea of the splitting is to separate the acoustic and transport phenomena. The acoustic step is solved in a non-conservative form using a scheme based on an approximate Riemann solver. Since the acoustic time step induced by the fast sound velocity is very restrictive, an implicit treatment of this step is performed. For the transport step driven by the slow material waves, an explicit scheme is used. Although non-conservative forms are used to derive numerical schemes for the two steps, the overall scheme resulting from this splitting operator strategy is conservative. It preserves contact discontinuities and reveals to be very robust compared to a standard unsplit scheme.

Numerical simulations of compressible two-phase flows are presented on two-dimensional structured grids. The implicit-explicit strategy allows large time steps, which do not depend on the fast acoustic waves.

*Keywords:* two-phase flows, positive Riemann solver, implicit scheme.

# **Introduction**

The present work takes place in the context of the atmospheric re-entry problem. This study can concern re-entry vehicles globally or partially made of metallic components, space debris for instance. During the re-entry phase, a solid undergoes a heating due to the friction of atmospheric gases. Conversion of kinetic energy to thermal energy leads to a sudden increase of the temperature of the object. This rise drives to a physical-chemical degradation of the thermal protective system, and to a boundary recession. Sublimation (injection of gas into the atmosphere) and the fusion of the metallic part (creation of a liquid phase into the gas) are the main causes of the solid ablation during the re-entry phase. A zonal approach is considered (see Fig 1) to cope with this very complex problem. In the gas flow region, far from the object or near a wall made of carbon where the ablation process is driven by the sublimation, classical schemes can be used. Numerical simulations of the sublimation process have already been studied in [MB14, MC13, BNM10, Mul10, Lat13]. In the region near a metallic wall, the velocity of the gas is small and with the appearance of the liquid phase, the dynamics of the flow is very different. From our own experience, usual numerical schemes are not very robust to compute such two-phase flows when large time steps are considered. In the present paper, we focus on the multiphase flow region and we propose a numerical method to simulate two-phase flows. The work presented here is really the first step of a global project since viscosity effects and heat transfers are not taken into account.

The modelling and computation of multiphase flows have been widely studied for the past decades. There are two main approaches to compute compressible flows with interfaces: sharp interface methods, and diffuse interface methods. In the first approach, the interface between the two media, considered as a sharp discontinuity, is followed explicitly and each phase can be computed with different models. In Lagrangian or Arbitrary Lagrangian-Eulerian methods, the mesh moves during the computation like the interface. Large distortions and interface topological changes can hardly be taken into account. Front capturing methods are Eulerian methods where the interface is reconstructed. In Level Set methods [OS88] the interface is located as the zero of an implicit function. In the Volume Of Fluid method [HN81], the interface is reconstructed from the volume fraction of each fluid. In the second approach, diffuse interface methods [BN86, KMB<sup>+</sup>01, ACK02, MSNA02, FBC<sup>+</sup>11, KL10, SA99], based on an Eulerian mesh, allow numerical diffusion of the interface. The same equations are solved in the entire domain. In addition, these models allow the creation of new interfaces and topological changes during the simulation. In the present context, the fusion of the metallic part leads to a significant topological change. Consequently, we choose to use a diffuse interface approach in order to simulate the two-phase

![](_page_1_Figure_0.jpeg)

Figure 1: Scheme of the re-entry of a metallic debris.

flow. Those methods have already been studied by many authors. The seminal work of Baer and Nunziato [BN86] (see also [SW84]) introduces a pressure and velocity non-equilibrium model for two-phase flows. The model of Kapila et al. [KMB<sup>+</sup>01] can be seen as the limit of instantaneous velocity and pressure relaxation, see [MG05]. In this case, a non-conservative term proportional to the divergence of the velocity appears in the equation of the volume fraction. This term describes the expansion/compression effects in case of a mixture. Several numerical schemes have been recently proposed for the Kapila et al. model or its variants [MG05, SPB09, LNS13, PS14, LMSN14]. However, as suggested by some experiments, we assume that the liquid phase in our application will not disperse nor mix with the gas flow. Consequently, immiscible fluids can be considered in this study and the non-conservative term of Kapila et al. model can be neglected. Indeed, at the continuous level, if the initial condition is made of two separated phases, then both Kapila et al. and five-equation models give the same solution. It is only at the discrete level that differences can appear inside the small mixture zone due to the numerical diffusion. Therefore the five-equation model [ACK02, MSNA02] is used in this work to track the pure interface between the two fluids.

A two-phase flow involving an almost incompressible liquid and a compressible gas is considered here. With the diffuse interface method, the same equations are solved in the entire domain, thus the liquid is modelled by a compressible fluid in which the Mach number is very low. Two numerical difficulties appear in this case. First, several time scales are involved in this configuration: the scale of the fast acoustic waves which leads to an important CFL restriction on the time step for explicit schemes, and the scale of slow material waves. Being concerned by the large time scales, the two-phase flow has to be computed with an implicit numerical scheme. In classical aero-thermodynamic computations, implicit scheme should also be considered in order to handle the dissipative effects and to compute efficiently stationary solutions. Up to our knowledge, implicit schemes for two-phase flows simulations are not very robust with large CFL conditions. Our main objective in this paper is to derive a scheme to compute two-phase flow using large time steps. The other difficulty is the lack of accuracy of the numerical schemes in the low Mach regime. This topic has been widely investigated in the literature, see [GV99, GM04, MG08, DJOR15] and the references therein. Preconditioned methods or specific corrections of the numerical flux have been developed in order to capture the low Mach limit [Tur87, TMD<sup>+</sup>08, Rie11, Del10]. For monophasic flows in the low Mach regime, several implicit-explicit schemes have been proposed, like [DJY07, HJL12, CDK12], to resolve the material wave scale only. Another approach [CGK16, CGK17] is to use a splitting of the Euler equations between acoustic and transport systems. The acoustic system is resolved by an implicit scheme while the transport system is resolved by an explicit scheme. A low Mach fix based on a correction of the pressure flux is developed for the acoustic step.

In this paper, a numerical scheme is proposed to handle three challenges: simulate a two phase flow which involves large ratios of density and pressure, derive an implicit scheme in order to use large time steps and compute an almost incompressible liquid as a compressible fluid by using a low Mach correction. To overcome these challenges, we propose an extension of the splitting strategy presented in [CGK16, CGK17] to two-phase flows using the five-equation system [ACK02, MSNA02]. The splitting strategy is done with a Lagrange Projection type [GR96] algorithm. The acoustic part is solved in a nonconservative form using a Godunov-type scheme based on an extension of the Riemann solver derived in

[Gal03]. The Riemann solver slopes are chosen in order to ensure the positivity of the solution and their moduli are a priori different unlike [CGK16, CGK17]. An implicit treatment of this step is done since the fast acoustic waves induce very small time steps. Several versions of this time-implicit scheme are derived and compared. The transport step is treated explicitly. Although non-conservative forms are used to derive numerical schemes for the acoustic and the transport parts, the scheme resulting from the splitting procedure is conservative and preserves contact discontinuity. Thus, the right shock speeds are obtained. The robustness and the efficiency of the implicit-explicit scheme for two-phase flows are demonstrated with the numerical computations and the comparison with a classical unsplit scheme. The correction of [CGK16] is used in our low Mach simulation, since it has been developed for the splitting strategy and its implementation is very simple. This low Mach fix is similar to the one proposed by Rieper [Rie11]. It was already shown in [CGK16] that this low Mach correction works well for the acoustic-transport splitting for monophasic flows. Finally, note that the scheme proposed in this work is first order only. Extension to second order will be discussed in another work.

The outline of this paper is as follows. In the first section we present the governing equations of an immiscible two-phase flow. In section 2, the operator splitting and the numerical scheme are described in the one dimensional case, several implicit treatments of the acoustic part are then presented and compared and we also describe the extension of the numerical scheme to two-dimensional structured grids. The overall scheme resulting from the splitting strategy is shown to be conservative. Finally some numerical results for one and two-dimensional two-phase flows are presented in section 3.

### 1. The five-equation system

We denote by  $\rho_i$ ,  $\epsilon_i$  and  $p_i$  the density, the internal energy and the pressure of the fluid phases i=1,2. Each fluid is equipped with an Equation Of State (EOS) of the form  $p_i = p_i(\rho_i, \epsilon_i)$ . The sound velocity  $c_i$  of each phase is defined by  $c_i^2 = \frac{\partial p_i}{\partial \rho_i}\Big|_{s_i}$ , where  $s_i$  is the entropy.

We introduce the volume fraction  $z_i$  of each fluid and we have the relation  $z_1 + z_2 = 1$ . In the sequel we denote by  $z = z_1$  the volume fraction of the first fluid. The mixture density and mixture internal energy can be defined through the volume fraction, the density and internal energy of each fluid:

$$\rho = z_1 \rho_1 + z_2 \rho_2, \tag{1}$$

$$\rho \epsilon = z_1 \rho_1 \epsilon_1 + z_2 \rho_2 \epsilon_2. \tag{2}$$

Both phases share the same velocity  $\mathbf{u}$ , and the same pressure p. The five-equation system with isobaric closure derived in [ACK02, MSNA02] reads:

$$\partial_{t} (\rho_{1}z_{1}) + \nabla \cdot (\rho_{1}z_{1}\mathbf{u}) = 0,$$

$$\partial_{t} (\rho_{2}z_{2}) + \nabla \cdot (\rho_{2}z_{2}\mathbf{u}) = 0,$$

$$\partial_{t} (\rho\mathbf{u}) + \nabla \cdot (\rho\mathbf{u} \otimes \mathbf{u}) + \nabla p = 0,$$

$$\partial_{t} (\rho e) + \nabla \cdot ((\rho e + p)\mathbf{u}) = 0,$$

$$\partial_{t}z + \mathbf{u} \cdot \nabla z = 0,$$

$$p = p_{1} = p_{2},$$
(3)

where  $e = \epsilon + \frac{\mathbf{u}^2}{2}$  is the total energy of the mixture. Note that the evolution of the volume fraction is governed by a non-conservative equation. This formulation preserves contact discontinuities, i.e. the evolution of constant pressure and velocity profiles, see [ACK02, KL10].

If we consider a stiffened gas EOS for each fluid, we have

$$p_i = \rho_i \epsilon_i \left( \gamma_i - 1 \right) - \gamma_i \pi_i, \tag{4}$$

where  $\gamma_i > 1$  is the adiabatic exponent and  $\pi_i \geq 0$  a reference pressure. Using the isobaric closure relation defined by:

$$\begin{cases} p_1(\rho_1, \epsilon_1) - p_2(\rho_2, \epsilon_2) = 0, \\ \rho \epsilon = z \rho_1 \epsilon_1 + (1 - z) \rho_2 \epsilon_2, \end{cases}$$

the pressure can be obtained as  $p = (\gamma - 1) \rho \epsilon - \gamma \pi$  where the mixture parameters  $\gamma$  and  $\pi$  are defined by:

$$\gamma = 1 + \frac{1}{\sum_{i=1}^{2} \frac{z_i}{\gamma_i - 1}} \quad \text{and} \quad \pi = \frac{\gamma - 1}{\gamma} \sum_{i=1}^{2} \frac{z_i \gamma_i \pi_i}{\gamma_i - 1}.$$

In the general case, the mixture sound speed is given by [ACK02]:

$$c^2 = \frac{1}{\xi} \sum_{i=1}^{2} y_i \xi_i c_i^2,$$

where  $y_i$  is the mass fraction,  $\xi_i$  the partial derivative  $\frac{\partial(\rho_i\epsilon_i)}{\partial p}$ ,  $\xi = \sum_{i=1}^2 z_i\xi_i$  and  $c_i$  the sound speed of the pure fluid i. Since the speed of sound for a stiffened gas is given by  $c_i^2 = \gamma_i \frac{p+\pi_i}{\rho_i}$ , the mixture sound speed for the mixture of two stiffened gases reads:

$$c^2 = \gamma \frac{p+\pi}{\rho}.\tag{5}$$

Even if, in case of a real mixture, the physical entropies of each phase are not convected, this model has mathematical entropies. Indeed, for two stiffened gases, [PS05] gives the phase entropy for each fluid  $s_i = \frac{p+\pi}{\rho_i^{\gamma}}$ , and the mixture entropy  $s = y_1s_1 + y_2s_2$ . It can be proved that their evolution for smooth solutions reads:

$$\partial_t s_k + \mathbf{u} \cdot \nabla s_k = 0,$$
  
 $\partial_t s + \mathbf{u} \cdot \nabla s = 0.$ 

Before going any further, let us rewrite the five-equation system (3). First, note that the evolution equation of  $\rho_2 z_2$  in (3) can be replaced by the evolution equation of the mixture density  $\rho$  which is  $\partial_t \rho + \nabla \cdot (\rho u) = 0$ . One can also introduce the mass fraction of the first fluid  $y = \frac{\rho_1 z}{\rho}$  in order to rewrite the evolution equation of  $\rho_1 z_1$ . Then, system (3) is expanded to:

$$\begin{cases}
\partial_{t}\rho + \nabla \cdot (\rho \mathbf{u}) = 0, \\
\partial_{t} (\rho y) + \nabla \cdot (\rho y \mathbf{u}) = 0, \\
\partial_{t} (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \nabla p = 0, \\
\partial_{t} (\rho e) + \nabla \cdot ((\rho e + p) \mathbf{u}) = 0, \\
\partial_{t} z + \mathbf{u} \cdot \nabla z = 0.
\end{cases} \tag{6}$$

# 2. Numerical scheme

#### 2.1. One-dimensional case

For the sake of simplicity, a one-dimensional problem is considered in this section. As in [GR96, CGK16, KL10], the idea is to separate the acoustic and the transport phenomena according to their own propagation speed. The five-equation system (6) is expanded to:

$$\begin{cases}
\partial_{t}\rho + \rho\partial_{x}u + u\partial_{x}\rho = 0, \\
\partial_{t}(\rho y) + \rho y\partial_{x}u + u\partial_{x}(\rho y) = 0, \\
\partial_{t}(\rho u) + \rho u\partial_{x}u + \partial_{x}p + u\partial_{x}(\rho u) = 0, \\
\partial_{t}(\rho e) + \rho e\partial_{x}u + \partial_{x}(pu) + u\partial_{x}(\rho e) = 0, \\
\partial_{t}z + u\partial_{x}z = 0.
\end{cases} (7)$$

This system is split into the following two systems. The first system will be called the acoustic system and only takes into account the propagation of the acoustic waves. It is given by

$$\begin{cases}
\partial_{t}\rho + \rho\partial_{x}u = 0, \\
\partial_{t}(\rho y) + \rho y\partial_{x}u = 0, \\
\partial_{t}(\rho u) + \rho u\partial_{x}u + \partial_{x}p = 0, \\
\partial_{t}(\rho e) + \rho e\partial_{x}u + \partial_{x}(\rho u) = 0, \\
\partial_{t}z = 0.
\end{cases} (8)$$

This system is hyperbolic and its propagation speeds are 0 and  $\pm c$ .

The second one will be designed as the transport system. It takes into account the propagation of the material waves through the fluid and it is given by

$$\begin{cases}
\partial_{t}\rho + u\partial_{x}\rho = 0, \\
\partial_{t}(\rho y) + u\partial_{x}(\rho y) = 0, \\
\partial_{t}(\rho u) + u\partial_{x}(\rho u) = 0, \\
\partial_{t}(\rho e) + u\partial_{x}(\rho e) = 0, \\
\partial_{t}z + u\partial_{x}z = 0.
\end{cases} \tag{9}$$

The propagation speed is here u only.

The overall algorithm for a time step between  $t^n$  and  $t^{n+1}$  reads:

**Step 1:** From a state  $(\rho, \rho y, \rho u, \rho e, z)^n$ , compute the approximation of the acoustic system (8)  $(\rho, \rho y, \rho u, \rho e, z)^{n+1-}$ .

**Step 2:** Find the fluid state  $(\rho, \rho y, \rho u, \rho e, z)^{n+1}$  by solving the transport system (9) with the initial state  $(\rho, \rho y, \rho u, \rho e, z)^{n+1-}$ .

This kind of algorithm has been already used in [FBC<sup>+</sup>11, KL10] in an explicit version for the five-equation system and in [CGK16, CGK17] with an implicit treatment of the first step for the gas dynamic and homogeneous models for two-phase flows.

The schemes for each step are described in the next sections, in which the following classical notations are used. We consider a discretization of the space variable x with cells  $([x_{i-1/2}, x_{i+1/2}])_{i \in \mathbb{Z}}$ . The mesh interfaces are  $x_{i+1/2} = i\Delta x$  for  $i \in \mathbb{Z}$ , where  $\Delta x > 0$  is the grid step. The centre of the cell i is  $x_i = \frac{x_{i+1/2} + x_{i-1/2}}{2}$ . The time variable is discretized by  $t^n = n\Delta t$  for  $n \in \mathbb{N}$ , where  $\Delta t > 0$  is the time step. We use a finite-volume method, in which  $\phi_i^n$  is the approximation of

$$\frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \phi(x, t^n) \mathrm{d}x,$$

for any  $\phi(x,t)$ .

# 2.2. Acoustic step

First, the non-conservative system (8) is written differently. The second, third and fourth equations are combined with the evolution equation of the mixture density. Then, the first equation is divided by the mixture density. Finally, the acoustic system reads:

$$\begin{cases}
\partial_t \vartheta - \vartheta \partial_x u = 0, \\
\partial_t y = 0, \\
\partial_t u + \vartheta \partial_x p = 0, \\
\partial_t e + \vartheta \partial_x (pu) = 0, \\
\partial_t z = 0,
\end{cases} \tag{10}$$

where  $\vartheta = \frac{1}{\varrho}$  is the specific volume.

Although we are not interested in this particular form by itself, it will allow us to derive a numerical scheme for the acoustic system (8) and a conservative global scheme for the five-equation system (7). System (10) can be written in the compact form:

$$\partial_t V + \vartheta \partial_x G(V) = 0. \tag{11}$$

with  $V = {}^t(\vartheta, y, u, e, z)$  and  $G = {}^t(-u, 0, p, pu, 0)$ . Note that this system is very similar to the one of the Lagrangian gas dynamics.

This system is written in a non-conservative form, which makes it delicate to derive a suitable numerical approximation. The definition of the non-conservative product  $\vartheta \partial_x G(V)$  will not be discussed here, and we refer to [DMLFM95] for details.

This system is strictly hyperbolic and its five real eigenvalues are  $(\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5) = (-a\vartheta, 0, 0, 0, a\vartheta)$ , where  $a = \rho c$  is the Lagrangian sound "speed". The first and the fifth characteristic fields are genuinely non-linear while the waves associated to  $\lambda_2, \lambda_3$  and  $\lambda_4$  are linearly degenerate.

A classical approach to solve conservative hyperbolic system is to use a Godunov-type scheme [HLL83]. More precisely, the notion of simple Riemann solver [Gal00, Gal03, Bou04] will be used here. A simple Riemann solver W consists of (m+1) constant states  $(V_k)_{k=1}^{m+1}$  separated by m discontinuities.

In [Gal02], this notion is used to construct Godunov-type scheme for non-conservative systems like (11). The solution obtained after one time step is given by:

$$V_i^{n+1} = \frac{1}{\Delta x} \left( \int_{x_{i-\frac{1}{2}}}^{x_i} W\left(\frac{x - x_{i-\frac{1}{2}}}{\Delta t}; V_{i-1}^n, V_i^n\right) dx + \int_{x_i}^{x_{i+\frac{1}{2}}} W\left(\frac{x - x_{i+\frac{1}{2}}}{\Delta t}; V_i^n, V_{i+1}^n\right) dx \right). \tag{12}$$

Numerical methods for hyperbolic non-conservative systems are also given by Parés [Par06]. Here, we present a numerical scheme for the non-conservative system (11) in the framework defined by [Gal02]. Let us denote by  $W(x/t; V_l, V_r)$  an approximation of the Riemann problem defined by (11) at each interface between the left state  $V_l$  and the right state  $V_r$ . This self similar function is made of four states (see Fig. 2) separated by discontinuities propagating at the velocities  $-\bar{a}_-\vartheta_l$  and  $\bar{a}_+\vartheta_r$  such as:

$$W(x/t; V_l, V_r) = \begin{cases} V_l & \text{if } x/t \le -\bar{a}_- \vartheta_l, \\ V_l^* = V_l + \phi_R_- & \text{if } -\bar{a}_- \vartheta_l < x/t \le 0, \\ V_r^* = V_r - \phi_+ R_+ & \text{if } 0 < x/t \le \bar{a}_+ \vartheta_r, \\ V_r & \text{if } \bar{a}_+ \vartheta_r < x/t, \end{cases}$$
(13)

where

$$R_{\pm} = {}^{t}(-1, 0, \pm \bar{a}_{\pm}, p_{1-\alpha} \pm u_{\alpha} \bar{a}_{\pm}, 0),$$
$$\phi_{\pm} = \frac{\Delta p \pm \bar{a}_{\mp} \Delta u}{\bar{a}_{-} \bar{a}_{+} + \bar{a}_{\pm}^{2}},$$

with  $p_{1-\alpha}=(1-\alpha)p_l+\alpha p_r$ ,  $u_{\alpha}=\alpha u_l+(1-\alpha)u_r$ , and the coefficient  $\alpha$  is defined below. As usual,  $\Delta p=p_r-p_l$  and  $\Delta u=u_r-u_l$ . One can see that the velocities of intermediate states  $u_l^*$  and  $u_r^*$  are equal. Note that the pressure of the intermediates states  $p_l^*=p(\rho_l^*,\epsilon_l^*)$  and  $p_r^*=p(\rho_r^*,\epsilon_r^*)$  are different. The choice of the positive quantities  $\bar{a}_-$  and  $\bar{a}_+$ , which are the Riemann solver slopes, will be examined in Section 2.5.

![](_page_5_Figure_8.jpeg)

Figure 2: The Riemann solver for system (11).

The simple Riemann solver (13) induces a Godunov-type scheme, and after some calculations, it is easy to show that the numerical scheme (12) for the acoustic system (11) reads:

$$V_i^{n+1-} = V_i^n - \frac{\Delta t}{\Delta x} \vartheta_i^n \left( H_{i+1/2}^n - H_{i-1/2}^n \right), \tag{14}$$

where the numerical "flux"  $H_{i+1/2}^n = H\left(V_i^n, V_{i+1}^n\right)$  is defined below. The canonical choice  $\alpha = \frac{\bar{a}_-}{\bar{a}_- + \bar{a}_+}$  gives a numerical flux equal to the continuous flux evaluated at an average state. Namely, it reads:

$$H(V_l, V_r) = {}^t(-\bar{u}, 0, \bar{p}, \bar{p}\bar{u}, 0),$$
 (15)

with

$$\bar{u} = \frac{\bar{a}_{-}u_{l} + \bar{a}_{+}u_{r} - \Delta p}{\bar{a}_{-} + \bar{a}_{+}} \quad \text{and} \quad \bar{p} = \frac{\bar{a}_{+}p_{l} + \bar{a}_{-}p_{r} - \bar{a}_{-}\bar{a}_{+}\Delta u}{\bar{a}_{-} + \bar{a}_{+}}.$$
 (16)

One can see that the velocity  $\bar{u}$  used in the flux is equal to the velocity of the intermediate state of the Riemann solver  $u_l^* = u_r^*$ . However, the pressure  $\bar{p}$  is different from  $p_l^*$  and  $p_r^*$ .

**Remark 1.** The Riemann solver (13) and the corresponding Godunov-type scheme (14) are natural extensions for the non-conservative system (11) of the one obtained in [Gal03] for the conservative gas dynamic system.

This scheme can be made implicit by taking the numerical flux at time  $t^{n+1-}$ . Both implicit and explicit schemes for the acoustic system (11) can be written as:

$$V_i^{n+1-} = V_i^n - \frac{\Delta t}{\Delta x} \vartheta_i^n \left( H_{i+1/2}^\# - H_{i-1/2}^\# \right). \tag{17}$$

In the explicit case, we have

$$H_{i+1/2}^{\#} = H_{i+1/2}^{n} = H(V_{i}^{n}, V_{i+1}^{n}),$$

while in the implicit case

$$H_{i+1/2}^{\#} = H_{i+1/2}^{n+1-} = H\left(V_i^{n+1-}, V_{i+1}^{n+1-}\right).$$

The explicit and implicit versions will be analysed in section 2.6 and 2.7.

**Remark 2.** We note that like in [CGK16], the non-conservative term  $\vartheta$  is treated explicitly.

Finally, we obtain a numerical scheme for the Eulerian variables  $\rho, \rho y, \rho u, \rho e$  and z of the acoustic system (8) as follows. Since  $\rho_i^{n+1-} = \frac{1}{\vartheta_i^{n+1-}}$ , the first equation of (17) gives the evolution of the density during the acoustic step:

$$L_i \rho_i^{n+1-} = \rho_i^n, \tag{18}$$

with  $L_i = 1 + \frac{\Delta t}{\Delta x} [\![\bar{u}^\#]\!]_i$ . Using (17), the update formulae for the Eulerian variables  $(\rho y)_i^{n+1-} = \rho_i^{n+1-} y_i^{n+1-}$ ,  $(\rho u)_i^{n+1-} = \rho_i^{n+1-} u_i^{n+1-}$  and  $(\rho e)_i^{n+1-} = \rho_i^{n+1-} e_i^{n+1-}$  are given by:

$$\begin{cases}
L_{i}\rho_{i}^{n+1-} = \rho_{i}^{n}, \\
L_{i}(\rho y)_{i}^{n+1-} = (\rho y)_{i}^{n}, \\
L_{i}(\rho u)_{i}^{n+1-} = (\rho u)_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{p}^{\#}]_{i}, \\
L_{i}(\rho e)_{i}^{n+1-} = (\rho e)_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{p}^{\#}\bar{u}^{\#}]_{i}, \\
z_{i}^{n+1-} = z_{i}^{n},
\end{cases} (19)$$

with the notation  $[\![\psi]\!]_i = \psi_{i+1/2} - \psi_{i-1/2}$  for any quantity  $(\psi)_i$  defined at the interfaces.

# 2.3. Transport step

The transport system (9) is an advection system of the unknown at the velocity u (the fluid velocity). Therefore, it is simply approximated by a standard usual upwind scheme:

$$\phi_i^{n+1} = \phi_i^{n+1-} - \frac{\Delta t}{\Delta x} \left( \left( \tilde{u}_{i+1/2} \right)^- \left( \phi_{i+1}^{n+1-} - \phi_i^{n+1-} \right) + \left( \tilde{u}_{i-1/2} \right)^+ \left( \phi_i^{n+1-} - \phi_{i-1}^{n+1-} \right) \right), \tag{20}$$

where  $\phi$  denotes  $\rho$ ,  $\rho y$ ,  $\rho u$ ,  $\rho e$  and z, with the classical notation  $u^{\pm} = \frac{u \pm |u|}{2}$ . The choice of the propagation velocity of the transport step  $\tilde{u}_{i+1/2}$  is defined below in order to ensure the conservation property of the global scheme (see section 2.4).

Note that the stability condition associated with this transport step reads

$$\frac{\Delta t}{\Delta x} \max_{i \in \mathbb{Z}} \left( \tilde{u}_{i-1/2}^+ - \tilde{u}_{i+1/2}^- \right) < 1. \tag{21}$$

### 2.4. Conservation property of the overall scheme

In this section, the properties of the overall scheme defined by (17) and (20) are detailed. First, the overall scheme is shown to be conservative if the same discrete interface velocity is used in both acoustic and transport steps. We have the following

**Proposition 1.** The scheme resulting from the splitting strategy for the five-equation system is conservative with respect to  $\rho$ ,  $\rho y$ ,  $\rho u$ ,  $\rho e$  if the velocity of the transport step is equal to the opposite of the first component of the numerical flux  $H^{\#}_{i+1/2}$  of the acoustic step, i.e.  $\tilde{u} = \bar{u}^{\#}$ . This is true regardless of the explicit or implicit treatment of the acoustic step.

*Proof.* Let us rewrite the transport step (20) as a conservative component plus a non-conservative term proportional to the discrete divergence of the velocity

$$\phi_i^{n+1} = \left(1 + \frac{\Delta t}{\Delta x} [\tilde{u}]_i\right) \phi_i^{n+1-} - \frac{\Delta t}{\Delta x} [\tilde{u}\phi^{n+1-}]_i,$$

where  $\phi_{i+1/2}^{n+1-} = \phi_i^{n+1-}$  if  $\tilde{u}_{i+1/2} > 0$ , and  $\phi_{i+1}^{n+1-}$  otherwise. We also denote by  $\tilde{L}_i$  the quantity  $1 + \frac{\Delta t}{\Delta x} [\![\tilde{u}]\!]_i$ . Combining this equation with the formula (19) for the acoustic step, we can see that the overall scheme to compute the state at the time  $t^{n+1}$  from the state at  $t^n$  reads:

$$\begin{cases}
\rho_{i}^{n+1} = \frac{\tilde{L}_{i}}{L_{i}} \rho_{i}^{n} - \frac{\Delta t}{\Delta x} [\tilde{u} \rho^{n+1-}]_{i}, \\
(\rho y)_{i}^{n+1} = \frac{\tilde{L}_{i}}{L_{i}} (\rho y)_{i}^{n} - \frac{\Delta t}{\Delta x} [\tilde{u} (\rho y)^{n+1-}]_{i}, \\
(\rho u)_{i}^{n+1} = \frac{\tilde{L}_{i}}{L_{i}} (\rho u)_{i}^{n} - \frac{\Delta t}{\Delta x} \left( [\tilde{u} (\rho u)^{n+1-}]_{i} + \frac{\tilde{L}_{i}}{L_{i}} [\bar{p}^{\#}]_{i} \right), \\
(\rho e)_{i}^{n+1} = \frac{\tilde{L}_{i}}{L_{i}} (\rho e)_{i}^{n} - \frac{\Delta t}{\Delta x} \left( [\tilde{u} (\rho e)^{n+1-}]_{i} + \frac{\tilde{L}_{i}}{L_{i}} [\bar{p}^{\#} \bar{u}^{\#}]_{i} \right), \\
z_{i}^{n+1} = z_{i}^{n} - \frac{\Delta t}{\Delta x} [\tilde{u} z^{n}]_{i} + z_{i}^{n} \frac{\Delta t}{\Delta x} [\tilde{u}]_{i}.
\end{cases} (22)$$

One can see that the overall scheme is conservative if we have  $\tilde{L}_i = L_i$ . This choice leads to define the propagation velocity at an interface as the opposite of the first component of the numerical flux  $H_{i+1/2}^{\#}$  of the acoustic scheme:

$$\tilde{u}_{i+1/2} = \bar{u}_{i+1/2}^{\#} = \frac{\bar{a}_{-}u_{i}^{\#} + \bar{a}_{+}u_{i+1}^{\#}}{\bar{a}_{-} + \bar{a}_{+}} - \frac{p_{i+1}^{\#} - p_{i}^{\#}}{\bar{a}_{-} + \bar{a}_{+}}.$$
(23)

Finally, with the choice (23) for the propagation velocity, the global scheme for the full 1D system (7) reads:

$$\begin{cases}
\rho_{i}^{n+1} = \rho_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{u}^{\#} \rho^{n+1-}]_{i}, \\
(\rho y)_{i}^{n+1} = (\rho y)_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{u}^{\#} (\rho y)^{n+1-}]_{i}, \\
(\rho u)_{i}^{n+1} = (\rho u)_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{u}^{\#} (\rho u)^{n+1-} + \bar{p}^{\#}]_{i}, \\
(\rho e)_{i}^{n+1} = (\rho e)_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{u}^{\#} (\rho e)^{n+1-} + \bar{p}^{\#} \bar{u}^{\#}]_{i}, \\
z_{i}^{n+1} = z_{i}^{n} - \frac{\Delta t}{\Delta x} [\bar{u}^{\#} z^{n}]_{i} + z_{i}^{n} \frac{\Delta t}{\Delta x} [\bar{u}^{\#}]_{i}.
\end{cases} (24)$$

where the quantities at time  $t^{n+1-}$  are defined by (19) and #=n or n+1-.

The overall scheme is clearly conservative with respect to  $\rho$ ,  $\rho y$ ,  $\rho u$ ,  $\rho e$  regardless of the explicit or implicit treatment of the acoustic step. Moreover, as in [KL10], it is easy to see that the scheme preserves contact discontinuities, i.e. the evolution of constant pressure and velocity profiles.

#### 2.5. Choice of the Riemann solver slopes and conditions of positivity

In this section, the choice of the Riemann solver slopes is discussed. This choice is based on a criterion of positivity of the solution. Indeed, some variables such as the density or the internal energy have to be positive while the sound speed has to be real. In the case of a mixture of two stiffened gases, the positivity of the internal energy can be directly deduced from the positivity of  $p + \pi$ . This leads to define the notion of admissible solutions, where  $\vartheta > 0$  and  $p + \pi > 0$ .

Proposition 2. For a mixture of two stiffened gases such that

$$(\gamma_2 - \gamma_1)(\pi_2 - \pi_1) \ge 0,$$
 (25)

the sets of admissible solutions

$$\mathcal{A}_{L} = \left\{ {}^{t}(\vartheta, y, u, e, \vartheta z) \mid \vartheta > 0 \text{ and } p + \pi > 0 \right\}$$
$$\mathcal{A}_{E} = \left\{ {}^{t}(\rho, \rho y, \rho u, \rho e, z) \mid \rho > 0 \text{ and } p + \pi > 0 \right\}$$

are convex.

The proof is given in Appendix A. Condition (25) is not restrictive in general, see for instance [LMMS04, Shy04].

Under the CFL condition (21), the solution remains in the convex set of admissible solutions during the transport step. Therefore, we only have to consider the positivity of the acoustic step. It is known that in order to have a positive Godunov-type scheme, it is sufficient to prove that the intermediate states are positive. Indeed, the solution after one time step is a convex combination of the intermediate states. This might be not very clear for  $(\vartheta z)$ , but if the equation of the volume fraction is combined with the specific volume equation, we also have a convex combination. For the conservative gas dynamic system, exact conditions of positivity are given in [Gal03].

Let us denote by  $r = \frac{\bar{a}_{\perp}}{\bar{a}_{-}}$  the ratio between the slopes of the Riemann solver. The following result gives exact conditions to guarantee that the intermediate states remain in the set of admissible solutions in the explicit case.

**Proposition 3.** For a given r, we have the following exact positivity conditions:

$$\begin{cases}
\vartheta_l^* \geq 0 & \text{if } d_l \leq 0, \quad \text{or if } d_l > 0 \text{ and } \bar{a}_- \geq \frac{-r\Delta u + \sqrt{d_l}}{2(1+r)\vartheta_l}, \\
\vartheta_r^* \geq 0 & \text{if } d_r \leq 0, \quad \text{or if } d_r > 0 \text{ and } \bar{a}_- \geq \frac{-\Delta u + \sqrt{d_r}}{2r(1+r)\vartheta_r}, \\
(p+\pi)_l^* \geq 0 & \text{if } \mathcal{D}_l \leq 0, \quad \text{or if } \mathcal{D}_l > 0 \text{ and } \bar{a}_- \geq \frac{r\Delta u(\Delta p + \Pi_l + 2r(1+r)\pi_l) + \sqrt{\mathcal{D}_l}}{4(1+r)^2\hat{\epsilon}_l + 2r^2(\Delta u)^2}, \\
(p+\pi)_r^* \geq 0 & \text{if } \mathcal{D}_r \leq 0, \quad \text{or if } \mathcal{D}_r > 0 \text{ and } \bar{a}_- \geq \frac{\Delta u(\Pi_r - r\Delta p + 2(1+r)\pi_r) + \sqrt{\mathcal{D}_r}}{4(1+r)^2\hat{\epsilon}_r + 2r(\Delta u)^2},
\end{cases}$$
(26)

where.

$$\begin{array}{lcl} d_l & = & r^2(\Delta u)^2 + 4\vartheta_l(1+r)\Delta p, & d_r & = & (\Delta u)^2 - 4r\vartheta_r(1+r)\Delta p, \\ \Pi_l & = & 2(rp_l+p_r) - \Delta p, & \Pi_r & = & 2(rp_l+p_r) + r\Delta p, \\ \hat{\epsilon}_l & = & \epsilon_l - \vartheta_l\pi_l, & \hat{\epsilon}_r & = & \epsilon_r - \vartheta_r\pi_r, \\ \mathcal{D}_l & = & r^2(\Delta u)^2(\Delta p + \Pi_l + 2r(1+r)\pi_l)^2 - 4\Delta p(\Pi_l + 2(1+r)\pi_l)(2(1+r)^2\hat{\epsilon}_l + r^2(\Delta u)^2), \\ \mathcal{D}_r & = & (\Delta u)^2(r\Delta p - \Pi_r - 2(1+r)\pi_r)^2 + 4\Delta p(\Pi_r + 2(1+r)\pi_r)(2r(1+r)^2\hat{\epsilon}_r + r(\Delta u)^2). \end{array}$$

*Proof.* By using the relations  $V_l^* = V_l + \phi_R_-$  and  $V_r^* = V_r - \phi_+ R_+$ , (see (13)), the positivity of  $\vartheta$  and  $p + \pi$  translates into quadratic inequalities in  $\bar{a}_-$  that lead to the conditions of the proposition. The details of the calculations are given in Appendix B.

The above proposition gives, for a fixed r, the maximal value of the slope  $\bar{a}_-$  denoted by  $\bar{C}_-(r)$  which guarantees the positivity of the intermediate states. In practice, we can choose  $r_0 = 1$  or  $r_0 = \frac{\rho_r c_r}{\rho_l c_l}$  and we take

$$\bar{a}_{-} = k \max \left( \bar{C}_{-}(r_0), \rho_l c_l \right), \tag{27}$$

where  $k \ge 1$  is a constant. In general, we take k = 1.01. Then, the other Riemann solver slope is equal to  $\bar{a}_+ = r_0 \bar{a}_-$ . With the choice  $r_0 = \frac{\rho_r c_r}{\rho_l c_l}$ , we directly get:

$$\bar{a}_{+} = k \max \left( r_0 \bar{C}_{-}(r_0), \rho_r c_r \right). \tag{28}$$

#### 2.6. Stability of the explicit acoustic step

The explicit update formulae for the Eulerian variables are directly given by (19) with the flux at time  $t^n$ . In this case, interface velocity and pressure (see (16)) are directly computed using the fluid state at time  $t^n$ :

$$\bar{u}_{i+1/2}^{n} = \frac{\bar{a}_{-}u_{i}^{n} + \bar{a}_{+}u_{i+1}^{n}}{\bar{a}_{-} + \bar{a}_{+}} - \frac{p_{i+1}^{n} - p_{i}^{n}}{\bar{a}_{-} + \bar{a}_{+}}, 
\bar{p}_{i+1/2}^{n} = \frac{\bar{a}_{+}p_{i}^{n} + \bar{a}_{-}p_{i+1}^{n}}{\bar{a}_{-} + \bar{a}_{+}} - \frac{\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-} + \bar{a}_{+}} \left(u_{i+1}^{n} - u_{i}^{n}\right).$$
(29)

The acoustic step is stable under the classical Courant-Friedrichs-Lewy stability condition, given by:

$$\frac{\Delta t}{\Delta x} \max_{i \in \mathbb{Z}} \left( \frac{\bar{a}_{i+1/2}^- + \bar{a}_{i-1/2}^+}{\rho_i^n} \right) \le 1.$$
 (30)

In the numerical results, we show that the choice of  $r_0 = \frac{\rho_r c_r}{\rho_l c_l}$  is a suitable choice in order to reduce this CFL constraint.

### 2.7. Different versions of the implicit acoustic step

When the fluid velocity is low and the acoustic inside the fluid is not the main phenomena, the CFL condition (30) of the acoustic part can be very restrictive. To overcome this difficulty, the idea is to use an implicit scheme to compute the acoustic step in order to have large time steps which are not constrained by the sound speed inside the fluid. Here, we propose several methods to derive an implicit scheme for the acoustic system.

The first version of this implicit scheme is given by (17) with # = n + 1 - (we find it more convenient than the equivalent formulation (19)). This leads to find the state  $V^{n+1-}$  which is the zero of the non-linear system

$$V - V^n + \frac{\Delta t}{\Delta x} \vartheta^n \llbracket H \rrbracket = 0. \tag{31}$$

Since the mass and volume fraction are not modified in the acoustic step, we will focus on the evolution of the specific volume, the velocity and the energy in this section. Let us denote by  $V^{\vartheta ue} = {}^t(\vartheta_i, u_i, e_i)_i$  the vector of unknowns and  $\llbracket H \rrbracket^{\vartheta ue} = {}^t(-(\bar{u}_{i+1/2} - \bar{u}_{i-1/2}), \bar{p}_{i+1/2} - \bar{p}_{i-1/2}, \bar{p}_{i+1/2} - \bar{p}_{i-1/2} \bar{u}_{i-1/2})_i$  the jumps of the numerical flux.

To solve this non-linear system (31), the first idea is to use a Newton-Raphson method with the classical unknowns  $V^{\vartheta ue}$ . Nevertheless, since the numerical flux depends on the velocity and the pressure, it seems natural to consider the velocity and the pressure as unknowns. The choice for the third unknown is not obvious. Indeed, the specific volume, the entropy or the temperature can be used. We denote by X this third unknown. This leads to consider another set of unknowns  $W = {}^t(u_i, p_i, X_i)_i$  to solve the non-linear system (31), which becomes F(W) = 0. We also note that the non-conservative product can be written as  $\vartheta^n \llbracket H \rrbracket^{\vartheta ue} = \mathbf{M}W$ , where the tridiagonal matrix  $\mathbf{M}$  is given by:

$$\mathbf{M} = \begin{bmatrix} \cdot & \cdot & & & 0 \\ \cdot & \cdot & \cdot & & \\ & \mathbf{A}_i & \mathbf{B}_i & \mathbf{C}_i & \\ & \cdot & \cdot & \cdot & \\ 0 & & \cdot & \cdot & \cdot \end{bmatrix}$$
(32)

with

$$\mathbf{A}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \left(\frac{\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} & \frac{1}{(\bar{a}_{-}+\bar{a}_{+})_{i-\frac{1}{2}}} & 0\\ \left(\frac{-\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} & \left(\frac{-\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} & 0\\ \mathbf{A}_{i}^{3,1} & \mathbf{A}_{i}^{3,2} & 0 \end{bmatrix},$$

$$\mathbf{B}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \left(\frac{\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{-\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & \left(\frac{-1}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{-}+\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & 0\\ \left(\frac{\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & \left(\frac{-\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & 0\\ \mathbf{B}_{i}^{3,1} & \mathbf{B}_{i}^{3,2} & 0 \end{bmatrix},$$

and

$$\mathbf{C}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \left(\frac{-\bar{a}_{+}}{\bar{a}_{-} + \bar{a}_{+}}\right)_{i + \frac{1}{2}} & \frac{1}{(\bar{a}_{-} + \bar{a}_{+})_{i + \frac{1}{2}}} & 0\\ \left(\frac{-\bar{a}_{-} \bar{a}_{+}}{\bar{a}_{-} + \bar{a}_{+}}\right)_{i + \frac{1}{2}} & \left(\frac{\bar{a}_{-}}{\bar{a}_{-} + \bar{a}_{+}}\right)_{i + \frac{1}{2}} & 0\\ \mathbf{C}_{i}^{3,1} & \mathbf{C}_{i}^{3,2} & 0 \end{bmatrix}.$$

The coefficients of the third line of each block are given in Appendix C.

To find the root of the non linear system (31) we use the Newton-Raphson method defined by:

$$\frac{\partial F}{\partial W} \left( W^{k+1} - W^k \right) = -F(W^k),\tag{33}$$

where

$$\frac{\partial F}{\partial W} = \frac{\partial V}{\partial W} + \frac{\Delta t}{\Delta x} \left( \mathbf{M} + \frac{\partial \mathbf{M}}{\partial W} W \right). \tag{34}$$

In the sequel, we denote by IMN this first implicit scheme for the acoustic step. However, the resolution of the Newton-Raphson method can be expensive in term of CPU time. It can be more efficient to use an approximation of the matrix  $\frac{\partial F}{\partial W}$  (34). In the following, we propose two different approximations. The first one is based on the following idea.

Since the numerical flux depends only on the velocity and the pressure, the idea is to solve a simpler subsystem in order to find a prediction of those quantities. Then, the numerical flux will be computed. This prediction is based on the following observation. For smooth solutions, i.e. without shock, we have  $\partial_t s = 0$  and the evolution equation of the specific volume can be replaced by an equation on the pressure. In this case, we have the following velocity-pressure system:

$$\begin{cases} \partial_t p + a^2 \vartheta \partial_x u = 0\\ \partial_t u + \vartheta \partial_x p = 0 \end{cases}$$
 (35)

Since  $\partial_t s = 0$  we have s = s(x), the Lagrangian sound speed a = a(p, x) and the specific volume  $\vartheta = \vartheta(p, x)$  depend only on (p, x). Thus, the pressure and the velocity can be computed with the (p, u) system (35). Therefore, the idea is to solve the first and second equations of (31) with the Newton-Raphson method to have a prediction of the numerical flux, while the third equation will be directly solved in a non-linear form.

In order to do so, Newton's method (33) is also used, and we consider the entropy as the third unknown X of W. Let us reorder the vector of unknown W by separating the entropy from the velocity and the pressure. We denote by  $W^{up} = {}^t(u_i, p_i)_i$  and by  $W^s = {}^t(s_i)_i$ . We also introduce  $V^{\vartheta u} = {}^t(\vartheta_i, u_i)_i$  and the matrix  $\mathbf{M}^{\vartheta u}_{up}$  is defined by  $\vartheta^n \llbracket H^{\vartheta u} \rrbracket = \mathbf{M}^{\vartheta u}_{up} W^{up}$ . Using these notations, one can express the Newton-Raphson method (33) for the first and second lines of the non-linear system  $(F_1, F_2) = 0$  as:

$$\left[\begin{array}{c} \frac{\partial V^{\vartheta u}}{\partial W^{up}} + \frac{\Delta t}{\Delta x} \left( \mathbf{M}_{up}^{\vartheta u} + \frac{\partial \mathbf{M}_{up}^{\vartheta u}}{\partial W^{up}} W^{up} \right) & \frac{\partial V^{\vartheta u}}{\partial W^{s}} \end{array}\right] \left(\begin{array}{c} W^{up^{k+1}} - W^{up^{k}} \\ W^{s^{k+1}} - W^{s^{k}} \end{array}\right) = -F_{1,2} \left( W^{k} \right)$$

First, let us make an approximation by fixing the value of the Riemann solver slopes at time  $t^n$ . With this assumption, the matrix  $\mathbf{M}_{up}^{\vartheta u}$  does not depend on W. If we consider that the entropy variations are small, the term  $W^{s^{k+1}} - W^{s^k}$  can be neglected. In this case, Newton's method for the velocity-pressure subsystem reads:

$$\left(\frac{\partial V^{\vartheta u}}{\partial W^{up}} + \frac{\Delta t}{\Delta x} \mathbf{M}_{up}^{\vartheta u}\right) \left(W^{up^{k+1}} - W^{up^k}\right) = -F_{1,2}\left(W^k\right).$$

Or equivalently,

$$\left(\mathbf{Id} + \frac{\Delta t}{\Delta x}\tilde{\mathbf{M}}\right)\left(W^{up^{k+1}} - W^{up^{k}}\right) = -\frac{\partial W^{up^{k}}}{\partial V^{\vartheta u}}\left(V^{\vartheta u^{k}} - V^{\vartheta u^{n}}\right) - \frac{\Delta t}{\Delta x}\tilde{\mathbf{M}}W^{up^{k}}$$

where  $\frac{\partial W^{up}}{\partial V^{\vartheta u}}$  is a matrix defined by:

$$\frac{\partial W^{up}}{\partial V^{\vartheta u}}_i = \left[ \begin{array}{cc} 0 & 1 \\ -a_i^2 & 0 \end{array} \right].$$

Note that the matrix  $\frac{\partial W^{up}}{\partial V^{\partial u}}$  depends only on the sound speed. The matrix  $\tilde{\mathbf{M}}$  has the same structure as the matrix  $\mathbf{M}$ , see (32), but with

$$\tilde{\mathbf{A}}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \left(\frac{-\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} & \left(\frac{-\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} \\ a_{i}^{2} \left(\frac{-\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} & \frac{-a_{i}^{2}}{(\bar{a}_{-}+\bar{a}_{+})_{i-\frac{1}{2}}} \end{bmatrix},$$

$$\tilde{\mathbf{B}}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \left(\frac{\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{-}\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & \left(\frac{-\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} \\ a_{i}^{2} \left(\left(\frac{-\bar{a}_{+}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}\right)_{i+\frac{1}{2}} & \frac{a_{i}^{2}}{(\bar{a}_{-}+\bar{a}_{+})_{i-\frac{1}{2}}} + \frac{a_{i}^{2}}{(\bar{a}_{-}+\bar{a}_{+})_{i+\frac{1}{2}}} \end{bmatrix},$$

and

$$\tilde{\mathbf{C}}_{i} = \vartheta_{i}^{n} \begin{bmatrix} \begin{pmatrix} -\bar{a}_{-}\bar{a}_{+} \\ \bar{a}_{-}+\bar{a}_{+} \end{pmatrix}_{i+\frac{1}{2}} & \begin{pmatrix} \bar{a}_{-} \\ \bar{a}_{-}+\bar{a}_{+} \end{pmatrix}_{i+\frac{1}{2}} \\ a_{i}^{2} \begin{pmatrix} \bar{a}_{+} \\ \bar{a}_{-}+\bar{a}_{+} \end{pmatrix}_{i+\frac{1}{2}} & \frac{-a_{i}^{2}}{(\bar{a}_{-}+\bar{a}_{+})_{i+\frac{1}{2}}} \end{bmatrix},$$

where we remind that the values of the Riemann solver slopes are frozen at time  $t^n$ . Since we are looking for a prediction of the numerical flux, the Lagrangian sound speed  $a_i$  is also frozen at time  $t^n$ , and we only consider the first iteration of the Newton-Raphson method. Namely, we have:

$$\left(\mathbf{Id} + \frac{\Delta t}{\Delta x}\tilde{\mathbf{M}}\right)W^{up^{n+1}} = W^{up^n} \tag{36}$$

Solving this linear system, we find a new velocity and pressure. With this pressure prediction, the numerical flux  $H^{n+1-}$  of the acoustic step can be computed by using (15). Then, the values of  $\vartheta^{n+1-}$  and  $e^{n+1-}$  can be updated by using the formulae (17). As well as in the explicit scheme, the volume and mass fractions are not modified by the implicit scheme for the acoustic step. Once all the conservative variables are computed at the end of the step, the real pressure is determined by using the equation of state. This second approach to derive an implicit scheme for the acoustic step, denoted by IM1, is summarized in the following algorithm:

# Algorithm 1 Implicit scheme for the acoustic step: IM1

- compute the Riemann solver slopes,
- solve the linear system (36),
- compute the numerical flux  $H^{n+1-}$  with the new velocity and pressure,
- update the specific volume and the energy with (17),
- compute the real pressure with the equation of state.

**Remark 3.** Solving the linear system (36) is consistent with the velocity-pressure system (35) for smooth solutions.

Finally, a third version of the implicit scheme can be obtained if one iterates over the algorithm 1. In this case, when the convergence is reached, the Lagrangian sound speed and the Riemann solver slopes are at time  $t^{n+1-}$ . Nonetheless, the prediction of the pressure obtained at convergence is not equal to the pressure computed with the equation of state. Hence, the solution of this third algorithm, denoted by IM2, is not equivalent to the solution of the non-linear system.

**Remark 4.** When the Riemann solver slopes are equal, we recover with the implicit scheme IM1 the scheme derived in [CGK16] with a Suliciu-type relaxation method.

**Remark 5.** When an implicit scheme is used in the acoustic step, the CFL condition (21) for the transport step becomes implicit. In this case, one has to check a posteriori if the condition (21) is satisfied with the new velocity  $\bar{u}_{i+1/2}$ . If not, a smaller time step is used and the acoustic step is computed once again.

![](_page_12_Picture_0.jpeg)

Figure 3: Parameters of a cell in two dimensions.

#### 2.8. Extension to two dimensions

Without loss of generality, a two dimensional problem discretized over a structured mesh is considered. Let us denote by  $\Omega_{l,m}$  a cell of our mesh,  $\Gamma_{l+1/2,m}$  (resp.  $\Gamma_{l,m+1/2}$ ) the face of the cell  $\Omega_{l,m}$  toward the first direction of the mesh (resp. the second one) and  $\mathbf{n}_{l+1/2,m}$  (resp.  $\mathbf{n}_{l,m+1/2}$ ) the unit vector normal to  $\Gamma_{l+1/2,m}$  (resp.  $\Gamma_{l,m+1/2}$ ), see Fig. 3.

System (3) is again split into two parts. The first part is the acoustic step:

$$\begin{cases}
\partial_{t}\rho + \rho \nabla \cdot \mathbf{u} = 0, \\
\partial_{t}(\rho y) + \rho y \nabla \cdot \mathbf{u} = 0, \\
\partial_{t}(\rho \mathbf{u}) + \rho \mathbf{u} \nabla \cdot \mathbf{u} + \nabla p = 0, \\
\partial_{t}(\rho e) + \rho e \nabla \cdot \mathbf{u} + \nabla \cdot (p \mathbf{u}) = 0, \\
\partial_{t} z = 0.
\end{cases} (37)$$

Like in the one-dimensional case, the following form is used to derive a numerical scheme for the acoustic step.

$$\begin{cases}
\partial_t \vartheta - \vartheta \nabla \cdot \mathbf{u} = 0, \\
\partial_t y = 0, \\
\partial_t \mathbf{u} + \vartheta \nabla p = 0, \\
\partial_t e + \vartheta \nabla \cdot (p \mathbf{u}) = 0, \\
\partial_t z = 0.
\end{cases} (38)$$

The second system is the transport step:

$$\begin{cases}
\partial_{t}\rho + \mathbf{u} \cdot \nabla \rho = 0, \\
\partial_{t}(\rho y) + \mathbf{u} \cdot \nabla (\rho y) = 0, \\
\partial_{t}(\rho \mathbf{u}) + \mathbf{u} \cdot \nabla (\rho \mathbf{u}) = 0, \\
\partial_{t}(\rho e) + \mathbf{u} \cdot \nabla (\rho e) = 0, \\
\partial_{t}z + \mathbf{u} \cdot \nabla z = 0.
\end{cases} \tag{39}$$

Let us introduce the following notations. The volume of a cell  $\Omega_{l,m}$  is denoted by  $|\Omega_{l,m}|$ . Likewise, the length of a face  $\Gamma_{l+1/2,m}$  is denoted by  $|\Gamma_{l+1/2,m}|$ . We denote by  $\mathcal{F}_{l,m}$  the set of indices f of the faces of a cell  $\Omega_{l,m}$ . Namely, in this two-dimensional structured framework,

$$\mathcal{F}_{l,m} = \{(l+1/2,m), (l-1/2,m), (l,m+1/2), (l,m-1/2)\}.$$

# Acoustic step:

Like in the one-dimensional case, the numerical scheme for the acoustic step reads:

$$V_{l,m}^{n+1-} = V_{l,m}^{n} - \frac{\Delta t}{\rho_{l,m}^{n} |\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| H_{f}^{\#}$$
(40)

where  $V = {}^{t}(\vartheta, y, \mathbf{u}, e, z)$  and  $H_f = {}^{t}(-\bar{u}_f, 0, \bar{p}_f \mathbf{n}_f, \bar{p}_f \bar{u}_f, 0)$ . The pressure and normal velocity on a face  $\Gamma_{l+1/2,m}$  are given by:

$$\bar{u}_{l+1/2,m} = \frac{\bar{a}_{-,l+1/2,m} \mathbf{u}_{l,m} + \bar{a}_{+,l+1/2,m} \mathbf{u}_{l+1,m}}{\bar{a}_{-,l+1/2,m} + \bar{a}_{+,l+1/2,m}} \cdot \mathbf{n}_{l+1/2,m} - \frac{p_{l+1,m} - p_{l,m}}{\bar{a}_{-,l+1/2,m} + \bar{a}_{+,l+1/2,m}},$$

$$\bar{p}_{l+1/2,m} = \frac{\bar{a}_{+,l+1/2,m} p_{l,m} + \bar{a}_{-,l+1/2,m} p_{l+1,m}}{\bar{a}_{-,l+1/2,m} + \bar{a}_{+,l+1/2,m}} - \frac{\bar{a}_{-,l+1/2,m} \bar{a}_{+,l+1/2,m}}{\bar{a}_{-,l+1/2,m} + \bar{a}_{+,l+1/2,m}} \left(\mathbf{u}_{l+1,m} - \mathbf{u}_{l,m}\right) \cdot \mathbf{n}_{l+1/2,m}.$$

where the slopes are computed using (27) and (28) at each cell interface. The non-linear system induced by the implicit scheme for the acoustic step can be solved like in the one-dimensional case, (see section 2.7).

#### Transport step:

We use an upwind finite volume scheme to solve the transport system like in the one-dimensional case. For  $\phi \in \{\rho, \rho_1 z, \rho \mathbf{u}, \rho e, z\}$ , the transport step reads:

$$\phi_{l,m}^{n+1} = \phi_{l,m}^{n+1-} - \frac{\Delta t}{|\Omega_{l,m}|} \left[ \left( \bar{u}_{l+1/2,m} \right)^{-} \left| \Gamma_{l+1/2,m} \right| \left( \phi_{l+1,m}^{n+1-} - \phi_{l,m}^{n+1-} \right) + \left( \bar{u}_{l-1/2,m} \right)^{+} \left| \Gamma_{l-1/2,m} \right| \left( \phi_{l,m}^{n+1-} - \phi_{l-1,m}^{n+1-} \right) + \left( \bar{u}_{l,m+1/2} \right)^{-} \left| \Gamma_{l,m+1/2} \right| \left( \phi_{l,m+1}^{n+1-} - \phi_{l,m}^{n+1-} \right) + \left( \bar{u}_{l,m-1/2} \right)^{+} \left| \Gamma_{l,m-1/2} \right| \left( \phi_{l,m}^{n+1-} - \phi_{l,m-1}^{n+1-} \right) \right].$$

$$(41)$$

### Overall scheme:

Like in the one-dimensional case, the acoustic velocity is used in the transport step. This choice also ensures that the overall scheme is conservative in 2d. Indeed, the global scheme resulting from the splitting strategy reads:

$$\begin{cases}
\rho_{l,m}^{n+1} = \rho_{l,m}^{n} - \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \bar{u}_{f}^{\#} \rho_{f}^{n+1-}, \\
(\rho y)_{l,m}^{n+1} = (\rho y)_{l,m}^{n} - \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \bar{u}_{f}^{\#} (\rho y)_{f}^{n+1-}, \\
(\rho \mathbf{u})_{l,m}^{n+1} = (\rho \mathbf{u})_{l,m}^{n} - \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \left( \bar{u}_{f}^{\#} (\rho \mathbf{u})_{f}^{n+1-} + \bar{p}_{f}^{\#} \mathbf{n}_{f} \right), \\
(\rho e)_{l,m}^{n+1} = (\rho e)_{l,m}^{n} - \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \left( \bar{u}_{f}^{\#} (\rho e)_{f}^{n+1-} + \bar{p}_{f}^{\#} \bar{u}_{f}^{\#} \right), \\
z_{l,m}^{n+1} = z_{l,m}^{n} - \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \bar{u}_{f}^{\#} z_{f}^{n} + z_{l,m}^{n} \frac{\Delta t}{|\Omega_{l,m}|} \sum_{f \in \mathcal{F}_{l,m}} |\Gamma_{f}| \bar{u}_{f}^{\#}.
\end{cases}$$
(42)

The scheme is clearly conservative for the densities, the momentum and the total energy equations.

#### 3. Numerical results

In this section we present numerical results for one and two-dimensional cases by using the splitting strategy presented before. In the sequel we denote by EXEX the explicit scheme for both acoustic and transport parts with the same Riemann solver slopes  $\bar{a}_{-} = \bar{a}_{+}$ , while IMEX is the implicit-explicit splitting scheme. We denote by EXEX(r\neq 1) the explicit-explicit scheme where the slopes of the Riemann solver are different. In this case, the ratio between the slopes of the Riemann solver is equal to  $r_0 = \frac{\rho_r c_r}{\rho_l c_l}$ . The results obtained with the schemes proposed in this paper are compared with the ones obtained by using a classical Roe-type scheme [ACK02]. The explicit (unsplit) Roe-type scheme is denoted by EX in the sequel. Its implicit formulation is denoted by IM.

#### 3.1. Comparison of implicit schemes for the acoustic step

We compare the different implicit schemes presented in section 2.7 by using classical shock tube tests. These test cases are characterized by a [0,1]m domain filled by two perfect gases. Initially, the interface

| Case  | $\rho_l \; (\mathrm{kg.m^{-3}})$ | $u_l \; ({\rm m.s^{-1}})$ | $p_l$ (Pa) | $\gamma_l$ | $\rho_r \; (\mathrm{kg.m^{-3}})$ | $u_r \; ({\rm m.s^{-1}})$ | $p_r$ (Pa) | $\gamma_r$ | $T_{end}$ (s) |
|-------|----------------------------------|---------------------------|------------|------------|----------------------------------|---------------------------|------------|------------|---------------|
| <br>1 | 1                                | 0                         | 1          | 1.4        | 0.125                            | 0                         | 0.1        | 1.6        | 0.1           |
| 2     | 1                                | 0                         | $10^{5}$   | 1.4        | 0.125                            | 0                         | 0.1        | 1.6        | $3.10^{-4}$   |
| 3     | 12.5                             | 0                         | $10^{5}$   | 1.4        | 0.125                            | 0                         | 0.1        | 1.6        | $1.10^{-3}$   |

Table 1: Comparison of implicit schemes for the acoustic step: Initial data

between the two fluids is located at 0.5m. The initial conditions are defined in table 1. In the first test case, the pressure and density ratios between the two gases are small. In the second test, the pressure ratio is equal to  $10^6$ . Finally, the third case yields a density ratio of 100 and a pressure ratio of  $10^6$ .

The time step is only driven by the stability condition of the transport step (21) with CFL number 0.5. In the results presented below, the third unknown X of the implicit scheme IMN is the entropy.

The first shock tube test case is computed on a 300 cells mesh. One can see in Fig. 4 that all the time-implicit schemes for the acoustic step give the same result relatively close to the exact solution for this first test case. Indeed, the shock velocity is well computed and the pressure remains constant through the contact discontinuity.

![](_page_14_Figure_5.jpeg)

Figure 4: Comparison of implicit schemes for the acoustic step: first case.

When the pressure ratio is large, namely for the second test case, the implicit scheme IMN based on the Newton Raphson method fails before the final time is reached. Nonetheless, the other approaches IM1 and IM2 show a good agreement with the exact solution, see Fig. 5. The computation has been done on 500 cells. The results of a mesh convergence for the IM1 scheme are given in Fig. 7(a). With a 1000 cells mesh, the right value of the density plateau is computed with the implicit scheme.

![](_page_14_Figure_8.jpeg)

Figure 5: Comparison of implicit schemes for the acoustic step: second case.

A 1000 cells mesh is used for the third test case. When large ratios of pressure and density are involved, the iterative approach for the implicit schemes IMN and IM2 does not work. In this case, only the IM1 implicit scheme gives a numerical solution, see Fig. 6. We note that the scheme, which is conservative, does not give a correct shock speed for this mesh. However, with a refined mesh, the results are more accurate, see Fig. 7(b). We would like to emphasize that this is a very challenging test case

which involves large ratios of density and pressure. In addition, the numerical scheme proposed here is a first order scheme. Note that the implicit unsplit scheme stops after some time for this test case.

![](_page_15_Figure_1.jpeg)

Figure 6: Comparison of implicit schemes for the acoustic step: third case.

Since we are interested in computations involving large density and pressure ratios, we use the most robust approach and the simplest one, namely the implicit scheme denoted by IM1.

![](_page_15_Figure_4.jpeg)

Figure 7: Comparison of implicit schemes for the acoustic step: mesh convergence for the density profiles.

### 3.2. Shock tube test

We consider here a classical one-dimensional shock tube problem between water and air. The stiffened gas equation of state is used for the liquid while the air is modelled by a perfect gas. At the initial time, the interface between the two fluids at rest is at x = 0.7m. Initial characteristics of water and air are

$$\begin{cases} (\rho, p, u, \gamma, \pi) = \left(1000 \text{ kg.m}^{-3}, 10^9 \text{ Pa}, 0 \text{ m.s}^{-1}, 4.4, 6 \times 10^8 \text{ Pa}\right) & \text{for } 0\text{m} \le x < 0.7\text{m}, \\ (\rho, p, u, \gamma, \pi) = \left(50 \text{ kg.m}^{-3}, 10^5 \text{ Pa}, 0 \text{ m.s}^{-1}, 1.4, 0 \text{ Pa}\right) & \text{for } 0.7\text{m} \le x \le 1\text{m}. \end{cases}$$

We note that the hypothesis (25) of the proposition 2 is satisfied. This ensures that the admissible domain is convex and that the scheme is positive.

The computational domain is discretised over a 1000 cells mesh and boundary conditions are constant states on both sides. For the implicit-explicit scheme, the time step is driven by the condition (21) with a Courant number of 0.9. For the EXEX scheme, the time step is determined by the conditions (30) and (21), also with a CFL number of 0.9. Fig. 8 shows results of our simulation and the exact solution of the Riemann problem after  $t = 240 \ \mu s$ .

![](_page_16_Figure_0.jpeg)

Figure 8: Shock tube test between two stiffened gases: exact solution and numerical results for the splitting strategy. The pressure profile is plotted in log-scale.

One can see that the gradient of pressure between the two fluids creates a shock wave that propagates inside the gas. There is also an expansion wave starting from the contact discontinuity and moving through the liquid. There is a good agreement between the numerical and the exact solutions despite the numerical diffusion of our first order solver. The density profile shows that the contact discontinuity and the shock wave are distinct.

We use a classical Roe-type numerical scheme as a comparison. A stability condition of 0.9 is also used for the explicit version, while the times steps of IM Roe scheme are in the same order of magnitude than the one of the implicit-explicit scheme. There is a good agreement between the explicit versions EX and EXEX and the implicit versions.

Mesh convergence. Convergence rate of the splitting strategy scheme is obtained by computing the relative error between the numerical and the exact solutions with  $L^1$  norm for several space steps. For  $\phi \in \{\rho, u, z, p\}$ , the relative error is given by  $E_{L^1}(\phi) = \|\phi - \phi_{exact}\|_{L^1} / \|\phi_{exact}\|_{L^1}$ , where  $\phi_{exact}$  is the exact solution after  $t = 240~\mu s$ . Fig. 9 gives the relative error in log-log scale as a function of the space step for the quantities  $\rho, u, z$  and p. Convergence rates, gathered in Table 2, are simply obtained by a linear regression.

One can see that the convergence rates for the explicit scheme and the implicit-explicit scheme are quite similar. Results for the explicit scheme show a good agreement with classical results for shock tube test, see [KL10, FBC<sup>+</sup>11].

![](_page_17_Figure_0.jpeg)

Figure 9: Relative error for the shock tube test. (Log-log scale)

| Variable            | Convergence rate, EXEX | Convergence rate, IMEX |
|---------------------|------------------------|------------------------|
| Density $\rho$      | 0.603                  | 0.657                  |
| Velocity $u$        | 0.883                  | 0.795                  |
| Volume fraction $z$ | 0.494                  | 0.507                  |
| Pressure $p$        | 0.810                  | 0.747                  |

Table 2: Shock tube test: convergence rate for density, velocity, volume fraction and pressure in  $L^1$  norm.

# 3.3. One-dimensional convection of a droplet

In our third test case, the convection of a liquid droplet inside a one dimensional domain filled with a gas is considered as in [CFA01]. The physical domain  $\Omega=[0,1]{\rm m}$  is plotted in Fig. 10. Initially, the liquid is set between  $x>0.4{\rm m}$  and  $x<0.6{\rm m}$  with a density  $\rho_l=1000~{\rm kg.m^{-3}}$  and a velocity  $u=100~{\rm m.s^{-1}}$ . The gas has no initial velocity and its density is  $\rho_g=1~{\rm kg.m^{-3}}$ . Both fluids are at the atmospheric pressure  $p=10^5$  Pa. The equation of state used for the liquid is the stiffened gas model with  $\gamma=4.4$  and  $\pi=6\times10^8$  Pa, while the gas is modelled by a perfect gas EOS where  $\gamma=1.4$ . Once again, the hypothesis of prop. 2 is fulfilled.

Constant boundary conditions are applied on both sides and 1000 cells are used for the computation. The CFL number for this test case is 0.5, and all the results are plotted after  $7.5 \times 10^{-4}$  seconds in figure 11.

![](_page_18_Figure_0.jpeg)

Figure 10: Convection of a droplet in one dimension: geometry.

![](_page_18_Figure_2.jpeg)

Figure 11: Convection of a droplet: Pressure, velocity, density and volume fraction profiles at  $t=7.5\times10^{-4}~\mathrm{s}$ .

During the simulation, the liquid travels to the right of the domain which leads to the compression of the gas in the right side and dilatation in the other side. Our results show good agreements with [CFA01] for density and velocity profiles and the interface position.

The profile of pressure inside the liquid is different for the explicit and semi-implicit scheme. With the implicit treatment of the acoustic waves, we recover the linear pressure profile of the incompressible solution inside the droplet. In the explicit case, we get the same results when the Riemann solver slopes are equal or not.

The number of time steps and the CPU time for explicit and implicit computations are given in table 3. With the implicit-explicit scheme, the number of iterations is divided by 145 as compared to the EXEX scheme and the CPU time is divided by 24. The explicit scheme with different slopes is only five times slower than the implicit one but does not give the incompressible pressure inside the liquid. As already mentioned above, the pressure profile is the same as the one obtained with equal slopes.

|                      | EXEX   | $EXEX(r\neq 1)$ | IMEX |
|----------------------|--------|-----------------|------|
| Number of time steps | 43 568 | 4875            | 300  |
| CPU time (s)         | 60.07  | 11.70           | 2.49 |

Table 3: Convection of a droplet: Number of time steps and CPU time for different schemes.

# *3.4. Two-dimensional liquid-gas interaction*

Now, we consider the two dimensional case of the interaction between a bubble of gas and a shock in a liquid studied in [SA99, KL10]. The computational domain Ω = [0*,* 2]m *×* [0*,* 1]m is described in Fig. 12. The gas is contained in a bubble of radius 0*.*4m located at the position (0*.*5m*,* 0*.*5m). The shock area is delimited by *x <* 0*.*04m. Initial conditions are provided in Table 4.

![](_page_19_Picture_2.jpeg)

Figure 12: Two-dimensional liquid-gas interaction: geometry of the test case.

|                | Density (kg.m−3<br>) | Pressure (Pa) | Velocity (m.s−1<br>) | γ   | π (Pa)    |
|----------------|----------------------|---------------|----------------------|-----|-----------|
| Liquid (shock) | 1030.9               | 3 × 109       | (300, 0)             | 4.4 | 6.8 × 108 |
| Liquid         | 1000.                | 105           | (0, 0)               | 4.4 | 6.8 × 108 |
| Gas            | 1.                   | 105           | (0, 0)               | 1.4 | 0         |

Table 4: Two-dimensional liquid-gas interaction: initial data.

We use a 600 *×* 300 space grid for our numerical computation. Wall boundary conditions are imposed on the upper and lower sides, and inflow/outflow boundary conditions on the other sides. For the explicit scheme, the time step is driven by the stability condition of the acoustic step (30) and the transport step (21) with a Courant number of 0*.*5. For the IMEX scheme, the time step is determined by the condition (21), also with a CFL number of 1*/*2.

The mixture density and the volume fraction of the liquid are plotted at several times in figures 13 and 14.

![](_page_19_Figure_8.jpeg)

Figure 13: Two-dimensional liquid-gas interaction: volume fraction field at several time, EXEX scheme.

![](_page_19_Figure_10.jpeg)

Figure 14: Two-dimensional liquid-gas interaction: volume fraction field at several time, IMEX scheme.

![](_page_19_Figure_12.jpeg)

Figure 15: Two-dimensional liquid-gas interaction: volume fraction field at several time, EX scheme.

The volume fraction field is similar to the results presented in [SA99] or [KL10]. We also compare this test case with the direct formulation. The explicit Roe-type scheme is not robust enough for this stiff problem: the code stops after some time. Instead, we use a Godunov-type scheme based on a simple Riemann solver with 2 intermediate states [Lat13]. The results of the unsplit explicit version are given in Fig. 15. There is an excellent agreement between the explicit and semi-implicit version of the scheme presented in the paper and the direct explicit formulation. The implicit version of the unsplit scheme fails because of the stiffness of the problem. This shows the robustness of the IMEX scheme despite the large density and pressure ratios of the initial conditions.

The numerical Schlieren images are presented in Figs. 16, 17 and 18. They are computed as in [QK96]. The propagation waves are very similar between the explicit scheme and the implicit-explicit scheme. When we compare the direct scheme with the splitting formulation, we can see that the splitting strategy is a bit more diffusive than the direct approach.

![](_page_20_Picture_2.jpeg)

Figure 16: Two-dimensional liquid-gas interaction: Schlieren images, EXEX scheme.

![](_page_20_Picture_4.jpeg)

Figure 17: Two-dimensional liquid-gas interaction: Schlieren images, IMEX scheme.

![](_page_20_Picture_6.jpeg)

Figure 18: Two-dimensional liquid-gas interaction: Schlieren images, EX scheme.

|                      | EXEX     | EXEX(r6=1) | IMEX   |
|----------------------|----------|------------|--------|
| Number of time steps | 705 251  | 16 490     | 1 589  |
| CPU time (s)         | 7 756.57 | 348.11     | 229.38 |

Table 5: Two-dimensional liquid-gas interaction: number of time steps and CPU time.

For this simulation, our scheme has been implemented in a parallel code. The measure of CPU time and number of iterations for a computation on 8 processors are given in Tab. 5. We can see that the implicit scheme is 34 times faster than the explicit one with the same slopes for the Riemann solver, since the implicit time step does not depend on the (fast) acoustic waves. When the slopes are different, the constraint on the acoustic time step is smaller. In this case, the explicit scheme is only 1.5 times as expensive as the implicit scheme in terms of computational time.

# *3.5. Low Mach test case*

Our last test case is a two-phase flow in low Mach conditions. In this regime, Godunov-type schemes are not accurate and a low Mach correction is needed. Here, we use the low Mach correction based on a modification of the pressure flux presented in [CGK16], see Appendix D for details. The idea with this test case is to compute a stationary solution and to use a low Mach correction with our scheme.

This last test case is an extension of the monophasic test studied in [Del10]. We consider a channel with a bump defined by

$$y(x) = \begin{cases} 0.1 \left(1 - \cos\left((x - 1)\pi\right)\right) & \text{if } x \in [1, 3]\text{m,} \\ 0 & \text{otherwise.} \end{cases}$$

The computational domain is included in  $\Omega=[0,4]\text{m}\times[0,1]\text{m}$ . A mixture of two perfect gases where  $\gamma_1=1.4$  and  $\gamma_2=1.6$  is considered. Note that the five-equation model is not justified in this case which involves a mixture of two fluids. Kapila et al. model will be better for this test case. Nevertheless, the challenges here are to recover the quasi incompressible solution with a low Mach correction and to compute faster a stationary solution with the implicit-explicit scheme. The fluid at the infinity is defined by the pressure  $p_{\infty}=10^5$  Pa, the densities  $\rho_{\infty}=\rho_{1,\infty}=\rho_{2,\infty}=14.8$  kg.m<sup>-3</sup> and the velocity  $(u_{\infty},0)$  such that the Mach number  $\text{Ma}=\frac{u_{\infty}}{\sqrt{\gamma\frac{p_{\infty}}{\rho_{\infty}}}}$  is equal to  $10^{-2}$ . Considering the values of the state at infinity we have the relation  $u_{\infty}=1$  m.s<sup>-1</sup>.

On the entrance of the channel, x=0m, a subsonic inflow boundary condition is enforced. Velocity and mixture density are taken at infinity values  $(u_{\infty}, \rho_{\infty})$ . Pressure is computed with the characteristic equation. The boundary condition on x=4m is defined by  $p=p_{\infty}$ , while velocity is given by the other characteristic equation. Other boundary conditions for the upper and lower side of the domain are wall boundary conditions. The initial state is given by  $(\rho, u, p) = (\rho_{\infty}, u_{\infty}, p_{\infty})$ .

Like in [Del10], computations are done on a 40 by 10 structured grid. A criteria based on the relative error of the conservative variables is used in order to get the stationary solution. All the results presented in this section are computed with the implicit-explicit scheme.

Without the low Mach correction, the incompressible solution is not captured by the numerical simulation as shown in Fig. 19. Note that the normalized pressure is defined as  $\frac{p-p_{\infty}}{p_{\infty}}$ . In Fig. 20, one can see that the numerical simulation converges to a steady state but Fig. 19 shows that this steady state is not the correct solution.

The low Mach correction presented in [CGK16] can be immediately applied to our diphasic scheme. With this correction, one can see in Fig. 21 that the isovalues of the Mach number and the normalized pressure are close to the incompressible solution. Indeed, fluctuations of the pressure are of order  $\mathrm{Ma}^{-2}$  and quasi symmetrical.

![](_page_21_Figure_7.jpeg)

Figure 19: Isovalues of Mach number and normalized pressure, without low Mach correction, Ma=10<sup>-2</sup>.

![](_page_21_Figure_9.jpeg)

Figure 20: Convergence of the residuals, without low Mach correction,  $Ma=10^{-2}$ .

![](_page_22_Figure_0.jpeg)

Figure 21: Isovalues of Mach number and normalized pressure, low Mach correction, Ma=10*−*<sup>2</sup> .

Figure 22 indicates that the more refined the mesh is, the more symmetrical the results are. Without the low Mach correction, the numerical scheme is not able to recover the incompressible solution even on refined meshes.

![](_page_22_Figure_3.jpeg)

Figure 22: Isovalues of Mach number, low Mach correction, Ma=10*−*<sup>2</sup> , meshes 80 *×* 20 and 160 *×* 40

In this low Mach situation, the time step condition based on the acoustic waves is very restrictive. Here the implicit strategy for the acoustic part is 35 times cheaper in term of CPU time than the explicit simulation, see Tab. 6. The EXEX(*r 6*=1) scheme gives the same results than the EXEX scheme since the ratio of the Lagrangian sound speed between two cells is small or in the same order of magnitude.

|                      | EXEX    | IMEX |
|----------------------|---------|------|
| Number of time steps | 222 476 | 574  |
| CPU time (s)         | 85.67   | 2.39 |

Table 6: Number of time steps and CPU time

A comparison with the direct Roe-type scheme is also performed. We use the implicit version of the unsplit scheme with different Low Mach preconditioning. Namely, the low Mach corrections of Rieper [Rie11], Dellacherie [Del10], Thornber et al. [TMD+08] are compared. The CFL number of the direct approach is 500. There is a good agreement between the isovalues of the normalized pressure, see Fig. 23. However, in terms of number of time steps, the direct formulations are more expensive to find the steady state (more than 1600 iterations) than the implicit-explicit scheme presented in the paper. We do not compare the computational time since the two solvers are not implemented in the same code. Note that an iterative process is used for the classical implicit scheme, whereas the IMEX scheme is computed with only one matrix inversion.

![](_page_22_Figure_9.jpeg)

Figure 23: Isovalues of normalized pressure with different low Mach corrections, unsplit formulation, Ma=10*−*<sup>2</sup> .

## **4. Conclusion**

In this work, an extension of the operator splitting strategy [CGK16, CGK17] to two-phase flows using the five-equation model [ACK02, MSNA02] has been presented. This strategy separates acoustic and transport phenomena. The acoustic step is solved in a non-conservative form using a Godunovtype scheme based on a simple Riemann solver. The Riemann solver slopes are computed using exact positivity conditions for the solution. The implicit treatment of the acoustic step allows large time steps that are based on the material waves velocity. The overall scheme resulting from the splitting strategy is shown to be conservative thanks to a proper tuning of the advection velocity. The robustness of the scheme is illustrated by the numerical simulations involving large ratios of density and pressure and the comparison with a classical unsplit scheme. Several versions of the implicit scheme for the splitting strategy are derived and compared. The simplest version is the most robust and only one matrix inversion is needed to compute a time step. The semi-implicit scheme has been shown to be more efficient in terms of CPU time compared to the explicit splitting scheme. When large density ratios are involved, the explicit scheme can be made more efficient with a suitable choice for the ratio between the Riemann solver slopes that greatly reduces the CFL constraint.

In the future, higher order methods will be considered. An extension to Navier-Stokes equations is already in progress to handle viscosity effects and heat transfers.

# Appendix A. Proof of Proposition 2

The aim of this appendix is to prove the convexity of the set of admissible solutions  $\mathcal{A}_L$  for a mixture of two stiffened gases such that  $(\gamma_2 - \gamma_1)(\pi_2 - \pi_1) \geq 0$ . The proof for  $\mathcal{A}_E$  is similar.

*Proof.* Let  $V_a = {}^t(\vartheta, y, u, e, \vartheta z)_a$  and  $V_b = {}^t(\vartheta, y, u, e, \vartheta z)_b$  be two admissible solutions. We show that for all positive  $\alpha$  and  $\beta$  such that  $\alpha + \beta = 1$ , the convex combination  $\alpha V_a + \beta V_b$  is also in  $\mathcal{A}_L$ .

In the sequel, we denote by  $V = \alpha V_a + \beta V_b$  the convex combination. The subscript a (resp. b), refers to the admissible solution  $V_a$  (resp.  $V_b$ ).

Positivity of  $\vartheta$ 

By definition of V, we have

$$\vartheta = \alpha \vartheta_a + \beta \vartheta_b.$$

Since  $V_a$  and  $V_b$  are admissible solutions,  $\vartheta_a$  and  $\vartheta_b$  are positive. Hence, the specific volume  $\vartheta$  is positive.

Positivity of  $p + \pi$ 

For a mixture of stiffened gases, the pressure is given by

$$p = (\gamma - 1)\rho\epsilon - \gamma\pi$$
.

Since the mixture coefficient  $\gamma$  is greater than one, the condition  $p+\pi \geq 0$  is equivalent to

$$\rho \epsilon \ge \pi. \tag{A.1}$$

Let  $\xi_i = \frac{1}{\gamma_i - 1}$  and  $\omega_i = \frac{\gamma_i \pi_i}{\gamma_i - 1}$  for each phase i = 1, 2. We denote by  $\xi = \frac{1}{\gamma - 1} = \sum_{i=1}^2 z_i \xi_i$  and  $\omega = \frac{\gamma \pi}{\gamma - 1} = \sum_{i=1}^2 z_i \omega_i$  the mixture parameters. With these notations, we have  $\pi = \frac{\omega}{1 + \xi}$ . Before going any further, let us note that

$$\vartheta(V)\omega(V) = \sum_{i=1}^{2} \vartheta(V)z_{i}(V)\omega_{i},$$

$$= \alpha \sum_{i=1}^{2} \vartheta(V_{a})z_{i}(V_{a})\omega_{i} + \beta \sum_{i=1}^{2} \vartheta(V_{b})z_{i}(V_{b})\omega_{i},$$

$$= \alpha \vartheta_{a}\omega(V_{a}) + \beta \vartheta_{b}\omega(V_{b}).$$

The same relation holds for  $\vartheta \xi$ :

$$\vartheta(V)\xi(V) = \alpha \vartheta_a \xi(V_a) + \beta \vartheta_b \xi(V_b).$$

Now, we want to prove the following inequality:

$$\alpha \epsilon_a + \beta \epsilon_b \ge \vartheta \pi$$
.

Since  $V_a$  and  $V_b$  are admissible solutions, we have  $\epsilon_{a,b} \geq \vartheta_{a,b} \pi_{a,b}$ . Thus, if

$$\alpha \frac{\vartheta_a \omega_a}{1 + \xi_a} + \beta \frac{\vartheta_b \omega_b}{1 + \xi_b} \ge \frac{\vartheta \omega}{1 + \xi},\tag{A.2}$$

then *p* + *π ≥* 0. After some manipulations, (A.2) reduces to

$$\frac{\alpha\beta\vartheta_a\vartheta_b}{(\alpha\vartheta_a(1+\xi_a)+\beta\vartheta_b(1+\xi_b))}(\xi_b-\xi_a)\left(\frac{\omega_a}{1+\xi_a}-\frac{\omega_b}{1+\xi_b}\right) \ge 0. \tag{A.3}$$

Since the first term of (A.3) is already positive, let us rewrite the last terms.

*•* First, we have

$$\xi_{a,b} = z_{a,b}\xi_1 + (1 - z_{a,b})\xi_2.$$

Hence the term *ξ<sup>b</sup> − ξ<sup>a</sup>* becomes

$$\xi_b - \xi_a = (z_b - z_a)(\xi_1 - \xi_2)$$

$$= \frac{z_b - z_a}{(\gamma_1 - 1)(\gamma_2 - 1)}(\gamma_2 - \gamma_1).$$

*•* The last term of (A.3) can be rewritten as

$$\frac{\omega_a}{1+\xi_a} - \frac{\omega_b}{1+\xi_b} = \frac{\omega_a - \omega_b + \xi_b \omega_a - \xi_a \omega_b}{(1+\xi_a)(1+\xi_b)},$$

where

$$\omega_a - \omega_b = (z_b - z_a)(\omega_2 - \omega_1),$$
  
$$\xi_b \omega_a - \xi_a \omega_b = (z_b - z_a)(\xi_1 \omega_2 - \xi_2 \omega_1).$$

Therefore,

$$\frac{\omega_a}{1+\xi_a} - \frac{\omega_b}{1+\xi_b} = \frac{z_b - z_a}{(1+\xi_a)(1+\xi_b)} (\omega_2(1+\xi_1) - \omega_1(1+\xi_2))$$
$$= \frac{(z_b - z_a)(1+\xi_1)(1+\xi_2)}{(1+\xi_a)(1+\xi_b)} (\pi_2 - \pi_1).$$

Finally, (A.3) is expressed as

$$\underbrace{\frac{\alpha\beta\vartheta_a\vartheta_b}{(\alpha\vartheta_a(1+\xi_a)+\beta\vartheta_b(1+\xi_b))} \frac{(z_b-z_a)^2(1+\xi_1)(1+\xi_2)}{(\gamma_1-1)(\gamma_2-1)(1+\xi_a)(1+\xi_b)}}_{\geq 0} (\gamma_2-\gamma_1)(\pi_2-\pi_1) \geq 0.$$

This proves that *p* + *π* is positive under the assumption (*γ*<sup>2</sup> *− γ*1)(*π*<sup>2</sup> *− π*1) *≥* 0.

### Appendix B. Proof of positivity conditions

This appendix gives the details of the calculations to find the positivity conditions of proposition 3.

*Proof.* By using the relation  $V_l^* = V_l + \phi_R_-$ , we get

$$\vartheta_{l}^{*} = \vartheta_{l} - \frac{\Delta p - r\bar{a}_{-}\Delta u}{\bar{a}_{-}^{2}(1+r)}, 
y_{l}^{*} = y_{l}, \nu_{l}^{*} = u_{l} - \frac{\Delta p - r\bar{a}_{-}\Delta u}{\bar{a}_{-}(1+r)}, \ne_{l}^{*} = e_{l} + \frac{\Delta p - r\bar{a}_{-}\Delta u}{\bar{a}_{-}^{2}(1+r)} (p_{1-\alpha} - u_{\alpha}\bar{a}_{-}), 
z_{l}^{*} = z_{l}.$$
(B.1)

where  $r = \frac{\bar{a}_+}{\bar{a}_-}$  and  $\alpha = \frac{\bar{a}_-}{\bar{a}_- + \bar{a}_+}$ .

Positivity of the specific volume

The first relation leads directly to the second order equation

$$(1+r)\bar{a}_{-}^{2}\vartheta_{l}^{*} = (1+r)\vartheta_{l}\bar{a}_{-}^{2} + r\Delta u\bar{a}_{-} - \Delta p.$$
(B.2)

Since the slopes and the specific volume  $\vartheta_l$  are positive, if the discriminant  $d_l$  of the second order equation is positive, the specific volume of the intermediate state  $\vartheta_l^*$  will be positive without any condition on the slope  $\bar{a}_-$ . If the discriminant is negative,  $\vartheta_l^*$  will be positive if  $\bar{a}_- \geq \frac{-r\Delta u + \sqrt{d_l}}{2(1+r)\vartheta_l}$ .

Positivity of the internal energy

By using the relations on the velocity and the total energy of the intermediate state, we get

$$\bar{a}_{-}^{2} \epsilon_{l}^{*} = \bar{a}_{-}^{2} \epsilon_{l} + \frac{(r\bar{a}_{-}\Delta u - \Delta p)(r\bar{a}_{-}\Delta u - \Pi_{l})}{2(1+r)^{2}},$$
(B.3)

or equivalently

$$2(1+r)^{2}\bar{a}_{-}^{2}\epsilon_{l}^{*} = \left(2(1+r)^{2}\epsilon_{l} + r^{2}(\Delta u)^{2}\right)\bar{a}_{-}^{2} - (r\Delta u(\Delta p + \Pi_{l}))\bar{a}_{-} + \Delta p\Pi_{l}.$$
 (B.4)

With the same considerations, exact positivity conditions for the internal energy can be derived.

Positivity of  $p + \pi$ 

For mixture of two stiffened gases, the mixture equation of state reads

$$p = (\gamma - 1)\rho\epsilon - \gamma\pi. \tag{B.5}$$

Therefore

$$(p+\pi)\vartheta = (\gamma - 1)(\epsilon - \pi\vartheta) = (\gamma - 1)\hat{\epsilon}.$$
 (B.6)

In order to prove the positivity of  $p + \pi$ , one can look for the positivity conditions of  $\hat{\epsilon} = \epsilon - \pi \vartheta$ . Since  $z_l^* = z_l$ , the mixture parameters of the intermediate state  $\gamma_l^*$  and  $\pi_l^*$  are equal to  $\gamma_l$  and  $\pi_l$ . Using the relations (B.2) and (B.3), we get

$$\bar{a}_{-}^{2}\hat{\epsilon}_{l}^{*} = \bar{a}_{-}^{2}\hat{\epsilon}_{l} + \frac{(\bar{a}_{-}\Delta u - \Delta p)(\bar{a}_{-}\Delta u - \Pi_{l})}{2(1+r)^{2}} - \frac{(\bar{a}_{-}\Delta u - \Delta p)\pi_{l}}{1+r}.$$
(B.7)

This leads to exact positivity condition of the proposition 3.

The positivity conditions for the right intermediate state are obtained by using the relation  $V_r^* = V_r - \phi_+ R_+$ .

# Appendix C. Coefficients of the matrix M

In this appendix, the coefficients of the matrix M (see section 2.7) are given.

Let us denote by  $\Delta = \frac{\bar{a}_+ - \bar{a}_-}{(\bar{a}_- + \bar{a}_+)^2}$ , and  $\Pi = \frac{\bar{a}_+ \bar{a}_-}{(\bar{a}_- + \bar{a}_+)^2}$ . The coefficients of the third line of  $\mathbf{A}_i$ ,  $\mathbf{B}_i$ , and  $\mathbf{C}_i$  read:

$$\begin{split} \mathbf{A}_{i}^{3,1} &= (-\bar{a}_{-}\Pi)_{i-\frac{1}{2}} \, u_{i-1} - \Pi_{i-\frac{1}{2}} p_{i-1} - (\bar{a}_{-}\bar{a}_{+}\Delta)_{i-\frac{1}{2}} \, \frac{u_{i}}{2} + (\bar{a}_{-}\Delta)_{i-\frac{1}{2}} \, \frac{p_{i}}{2}, \\ \mathbf{A}_{i}^{3,2} &= (-\Pi)_{i-\frac{1}{2}} \, u_{i-1} - \left(\frac{\Pi}{\bar{a}_{-}}\right)_{i-\frac{1}{2}} \, p_{i-1} - (\bar{a}_{+}\Delta)_{i-\frac{1}{2}} \, \frac{u_{i}}{2} + \Delta_{i-\frac{1}{2}} \frac{p_{i}}{2}, \\ \mathbf{B}_{i}^{3,1} &= -\left(\bar{a}_{-}\bar{a}_{+}\Delta\right)_{i-\frac{1}{2}} \, \frac{u_{i-1}}{2} - (\bar{a}_{+}\Delta)_{i-\frac{1}{2}} \, \frac{p_{i-1}}{2} + \left((\bar{a}_{+}\Pi)_{i-\frac{1}{2}} + (\bar{a}_{-}\Pi)_{i+\frac{1}{2}}\right) u_{i} + \left(-\Pi_{i-\frac{1}{2}} + \Pi_{i+\frac{1}{2}}\right) p_{i} \\ &+ (\bar{a}_{-}\bar{a}_{+}\Delta)_{i+\frac{1}{2}} \, \frac{u_{i+1}}{2} - (\bar{a}_{-}\Delta)_{i+\frac{1}{2}} \, \frac{p_{i+1}}{2}, \\ \mathbf{B}_{i}^{3,2} &= (\bar{a}_{-}\Delta)_{i-\frac{1}{2}} \, \frac{u_{i-1}}{2} + \Delta_{i-\frac{1}{2}} \, \frac{p_{i-1}}{2} + \left(-\Pi_{i-\frac{1}{2}} + \Pi_{i+\frac{1}{2}}\right) u_{i} + \left(\left(\frac{\Pi}{\bar{a}_{+}}\right)_{i-\frac{1}{2}} + \left(\frac{\Pi}{\bar{a}_{-}}\right)_{i+\frac{1}{2}}\right) p_{i} \\ &+ (\bar{a}_{+}\Delta)_{i+\frac{1}{2}} \, \frac{u_{i+1}}{2} - \Delta_{i+\frac{1}{2}} \, \frac{p_{i+1}}{2}, \\ \mathbf{C}_{i}^{3,1} &= (\bar{a}_{-}\bar{a}_{+}\Delta)_{i+\frac{1}{2}} \, \frac{u_{i}}{2} + (\bar{a}_{+}\Delta)_{i+\frac{1}{2}} \, \frac{p_{i}}{2} - (\bar{a}_{+}\Pi)_{i+\frac{1}{2}} u_{i+1} + \Pi_{i+\frac{1}{2}} p_{i+1}, \\ \mathbf{C}_{i}^{3,2} &= (-\bar{a}_{-}\Delta)_{i+\frac{1}{2}} \, \frac{u_{i}}{2} - \Delta_{i+\frac{1}{2}} \, \frac{p_{i}}{2} + \Pi_{i+\frac{1}{2}} u_{i+1} - \left(\frac{\Pi}{\bar{a}_{+}}\right)_{i+\frac{1}{2}} \, p_{i+1}. \end{split}$$

# Appendix D. Presentation of the low Mach correction of [CGK16]

The low Mach correction proposed by [CGK16] results in a modification of the Riemann solver of the acoustic step (13). Namely, the Riemann solver for the low Mach correction reads:

$$W^{\theta}(x/t; V_{l}, V_{r}) = \begin{cases} V_{l} & \text{if } x/t \leq -\bar{a}_{-}\vartheta_{l}, \\ V_{l}^{*,\theta} = V_{l} + \phi_{-}R_{-} + LM_{-} & \text{if } -\bar{a}_{-}\vartheta_{l} < x/t \leq 0, \\ V_{r}^{*,\theta} = V_{r} - \phi_{+}R_{+} - LM_{+} & \text{if } 0 < x/t \leq \bar{a}_{+}\vartheta_{r}, \\ V_{r} & \text{if } \bar{a}_{+}\vartheta_{r} < x/t, \end{cases}$$
(D.1)

where

$$R_{\pm} = {}^{t}(-1, 0, \pm \bar{a}_{\pm}, p_{1-\alpha} \pm u_{\alpha}\bar{a}_{\pm}, 0), \qquad LM_{\pm} = \frac{\bar{a}_{\mp}(\theta - 1)\Delta u}{\bar{a}_{-} + \bar{a}_{+}} {}^{t}(0, 0, 1, \bar{u}, 0),$$

$$\phi_{\pm} = \frac{\Delta p \pm \bar{a}_{\mp}\Delta u}{\bar{a}_{-}\bar{a}_{+} + \bar{a}_{+}^{2}}.$$

The notations are the same as in section 2.2. The coefficient  $\alpha$  is also equal to  $\frac{\bar{a}_{-}}{\bar{a}_{-}+\bar{a}_{+}}$ . The coefficient  $\theta$  is the parameter of the low Mach correction. In practice, this parameter is equal to  $\theta = \min\left(\frac{|\bar{u}|}{\max(c_{l},c_{r})},1\right)$ . The modified simple Riemann solver (D.1) also induces a Godunov-type scheme. Following the same lines than in section 2.2, the numerical scheme (12) for the acoustic system (11) reads:

$$V_i^{n+1-} = V_i^n - \frac{\Delta t}{\Delta x} \vartheta_i^n \left( H_{i+1/2}^{n,\theta} - H_{i-1/2}^{n,\theta} \right), \tag{D.2}$$

where the numerical "flux"  $H_{i+1/2}^{n,\theta}=H^{\theta}\left(V_{i}^{n},V_{i+1}^{n}\right)$  reads:

$$H^{\theta}(V_{l}, V_{r}) = {}^{t}(-\bar{u}, 0, \bar{p}^{\theta}, \bar{p}^{\theta}\bar{u}, 0)$$

with

$$\bar{u} = \frac{\bar{a}_{-}u_{l} + \bar{a}_{+}u_{r} - \Delta p}{\bar{a}_{-} + \bar{a}_{+}}$$
 and  $\bar{p}^{\theta} = \frac{\bar{a}_{+}p_{l} + \bar{a}_{-}p_{r} - \bar{a}_{-}\bar{a}_{+}\theta\Delta u}{\bar{a}_{-} + \bar{a}_{+}}$ 

The low Mach correction finally leads to a simple flux modification. Indeed, only the non-centred terms of the pressure flux is modified.

**Remark 6.** The low Mach number fix proposed by Rieper [Rie11] for Roe's approximate Riemann solver follows the same idea. The jump of the velocity is also multiplied by the local Mach number.

The good behaviour of the scheme in the low Mach regime is studied by Chalons et al. in [CGK16] with an analysis based on the truncation error of the scheme. The result is given in the following proposition:

**Proposition 4** ([CGK16])**.** In the low Mach regime, the rescaled discretization (D.2) of the acoustic step is consistent with

$$\begin{split} &\partial_{\tilde{t}}\tilde{\vartheta}-\tilde{\vartheta}\partial_{\tilde{x}}\tilde{u}=O(\Delta\tilde{t})+O(M\Delta\tilde{x}),\\ &\partial_{\tilde{t}}\tilde{u}+\frac{\tilde{\vartheta}}{M^2}\partial_{\tilde{x}}\tilde{p}=O(\Delta\tilde{t})+O\left(\frac{\theta\Delta\tilde{x}}{M}\right),\\ &\partial_{\tilde{t}}\tilde{e}+\tilde{\vartheta}\partial_{\tilde{x}}\left(\tilde{p}\tilde{u}\right)=O(\Delta\tilde{t})+O(M\Delta\tilde{x})+O(M\theta\Delta\tilde{x}). \end{split}$$

The rescaled discretization of the transport step is consistent with

$$\partial_{\tilde{t}}\tilde{\phi} + \tilde{u}\partial_{\tilde{x}}\tilde{\phi} = O(\Delta\tilde{t}) + O(\Delta\tilde{x}) + O(M\Delta\tilde{x})$$

and the equivalent equation verified by the rescaled scheme reads

$$\begin{cases} \partial_{\tilde{t}}\tilde{\rho} + \partial_{\tilde{x}}\left(\tilde{\rho}\tilde{u}\right) = O(\Delta\tilde{t}) + O(\Delta\tilde{x}) + O(M\Delta\tilde{x}), \\ \partial_{\tilde{t}}\left(\tilde{\rho}\tilde{u}\right) + \partial_{\tilde{x}}\left(\tilde{\rho}\tilde{u}^{2}\right) + \frac{1}{M^{2}}\partial_{\tilde{x}}\tilde{p} = O(\Delta\tilde{t}) + O(\Delta\tilde{x}) + O\left(\frac{\theta\Delta\tilde{x}}{M}\right), \\ \partial_{\tilde{t}}\left(\tilde{\rho}e\right) + \partial_{\tilde{x}}\left(\left(\tilde{\rho}\tilde{e} + \tilde{p}\right)\tilde{u}\right) = O(\Delta\tilde{t}) + O(\Delta\tilde{x}) + O(M\Delta\tilde{x}) + O(M\theta\Delta\tilde{x}). \end{cases}$$

As a consequence, provided that we impose the asymptotic behaviour *θi*+1*/*<sup>2</sup> = *O*(*M*), the truncation error is uniform with respect to the Mach number *M*.

**Remark 7.** The positivity conditions for the intermediates states (see section 2.5) can be naturally generalized to the low Mach case.

- [ACK02] G. Allaire, S. Clerc, and S. Kokh. A five-equation model for the simulation of interfaces between compressible fluids. *Journal of Computational Physics*, 181:577–616, 2002.
  - [BN86] M.R. Baer and J.W. Nunziato. A two-phase mixture theory for the deflagration-todetonation transition (DDT) in reactive granular materials. *International journal of multiphase flow*, 12(6):861–889, 1986.
- [BNM10] D. Bianchi, F. Nasuti, and E. Martelli. Navier-Stokes simulations of hypersonic flows with coupled graphite ablation. *Journal of Spacecraft and Rockets*, 47(4):554–562, 2010.
- [Bou04] F. Bouchut. *Nonlinear stability of finite Volume Methods for hyperbolic conservation laws: And Well-Balanced schemes for sources*. Springer Science & Business Media, 2004.
- [CDK12] F. Cordier, P. Degond, and A. Kumbaro. An Asymptotic-Preserving all-speed scheme for the Euler and Navier–Stokes equations. *Journal of Computational Physics*, 231(17):5685– 5704, 2012.
- [CFA01] R. Caiden, R. P. Fedkiw, and C. Anderson. A numerical method for two phase flow consisting of separate compressible and incompressible regions. *J. Comput. Phys.*, 166:1–27, 2001.
- [CGK16] C. Chalons, M. Girardin, and S. Kokh. An all-regime Lagrange-Projection like scheme for the gas dynamics equations on unstructured meshes. *Communications in Computational Physics*, 20(1):188–233, 006 2016.
- [CGK17] C. Chalons, M. Girardin, and S. Kokh. An all-regime Lagrange-Projection like scheme for 2D homogeneous models for two-phase flows on unstructured meshes. *Journal of Computational Physics*, 335:885 – 904, 2017.
  - [Del10] S. Dellacherie. Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number. *Journal of Computational Physics*, 229:978–1016, 2010.
- [DJOR15] S. Dellacherie, J. Jung, P. Omnes, and P.-A. Raviart. Construction of modified Godunov type schemes accurate at any Mach number for the compressible Euler system. working paper or preprint, November 2015.
  - [DJY07] P. Degond, S. Jin, and J. Yuming. Mach-number uniform asymptotic-preserving gauge schemes for compressible flows. *Bulletin of the Institute of Mathematics, Academia Sinica*, 2(4):851, 2007.
- [DMLFM95] G. Dal Maso, P. G. Le Floch, and F. Murat. Definition and weak stability of nonconservative products. *Journal de mathématiques pures et appliquées*, 74(6):483–548, 1995.
  - [FBC+11] M. Billaud Friess, B. Boutin, F. Caetano, G. Faccanoni, S. Kokh, F. Lagoutière, and L. Navoret. A second order anti-diffusive Lagrange-remap scheme for two-component flows. In *ESAIM: Proceedings*, volume 32, pages 149–162. EDP Sciences, 2011.
    - [Gal00] G. Gallice. Schémas de type Godunov entropiques et positifs préservant les discontinuités de contact. *Comptes Rendus de l'Académie des Sciences-Series I-Mathematics*, 331(2):149– 152, 2000.
    - [Gal02] G. Gallice. *Numerical Approximation of conservative or nonconservative non-linear hyperbolic systems*. Accreditation to supervise research, Université de Bordeaux I, June 2002.
    - [Gal03] G. Gallice. Positive and entropy stable Godunov-type schemes for gas dynamics and MHD equations in Lagrangian or Eulerian coordinates. *Numerische Mathematik*, 94:673–713, 2003.
    - [GM04] H. Guillard and A. Murrone. On the behavior of upwind schemes in the low Mach number limit: II. Godunov type schemes. *Computers & fluids*, 33(4):655–675, 2004.
    - [GR96] E. Godlevski and P.-A. Raviart. *Numerical approximation of hyperbolic systems of conservation laws*. Springer, 1996.

- [GV99] H. Guillard and C. Viozat. On the behaviour of upwind schemes in the low Mach number limit. *Computers & fluids*, 28(1):63–86, 1999.
- [HJL12] J. Haack, S. Jin, and J. Liu. An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations. *Communications in Computational Physics*, 12(04):955– 980, 2012.
- [HLL83] A. Harten, P. D Lax, and B. van Leer. On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Review*, 25(1):35–61, 1983.
- [HN81] C. W. Hirt and B. D. Nichols. Volume of fluid (VOF) method for the dynamics of free boundaries. *Journal of computational physics*, 39(1):201–225, 1981.
- [KL10] S. Kokh and F. Lagoutière. An anti-diffusive numerical scheme for the simulation of interfaces between compressible fluids by means of five-equation model. *Journal of Computational Physics*, 229:2773–2809, 2010.
- [KMB<sup>+</sup>01] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, and D. S. Stewart. Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations. *Physics of Fluids (1994-present)*, 13(10):3002–3024, 2001.
  - [Lat13] M. Latige. *Simulation numérique de l'ablation liquide*. PhD thesis, Université Sciences et Technologies-Bordeaux I, 2013.
- [LMMS04] O. Le Metayer, J. Massoni, and R. Saurel. Elaboration des lois d'état d'un liquide et de sa vapeur pour les modèles d'écoulements diphasiques. *Int. J. Thermal Sciences*, 43(3):265– 276, 2004.
- [LMSN14] S. Le Martelot, R. Saurel, and B. Nkonga. Towards the direct numerical simulation of nucleate boiling flows. *International Journal of Multiphase Flow*, 66:62–78, 2014.
  - [LNS13] S. LeMartelot, B. Nkonga, and R. Saurel. Liquid and liquid–gas flows at all speeds. *Journal of Computational Physics*, 255:53–82, 2013.
  - [MB14] A. Martin and I. D Boyd. Strongly coupled computation of material response and nonequilibrium flow for hypersonic ablation. *Journal of Spacecraft and Rockets*, 52(1):89–104, 2014.
  - [MC13] F. S Milos and Y-K Chen. Ablation, thermal response, and chemistry program for analysis of thermal protection systems. *Journal of Spacecraft and Rockets*, 50(1):137–149, 2013.
  - [MG05] A. Murrone and H. Guillard. A five equation reduced model for compressible two phase flow problems. *Journal of Computational Physics*, 202(2):664 – 698, 2005.
  - [MG08] A. Murrone and H. Guillard. Behavior of upwind scheme in the low Mach number limit: III. preconditioned dissipation for a five equation two phase model. *Computers & Fluids*, 37(10):1209–1224, 2008.
- [MSNA02] J. Massoni, R. Saurel, B. Nkonga, and R. Abgrall. Proposition de méthodes et modèles Eulériens pour les problèmes à interfaces entre fluides compressibles en présence de transfert de chaleur: Some models and Eulerian methods for interface problems between compressible fluids with heat transfer. *International journal of heat and mass transfer*, 45(6):1287–1307, 2002.
  - [Mul10] N. J. Mullenix. *Fully Coupled Model for High-Temperature Ablation and a Reative-Riemann Solver for its Solution*. PhD thesis, University of Akron, 2010.
  - [OS88] S. Osher and J. A. Sethian. Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations. *Journal of computational physics*, 79(1):12–49, 1988.
  - [Par06] C. Parés. Numerical methods for nonconservative hyperbolic systems: a theoretical framework. *SIAM Journal on Numerical Analysis*, 44(1):300–321, 2006.
  - [PS05] G. Perigaud and R. Saurel. A compressible flow model with capillary effects. *Journal of Computational Physics*, 209(1):139–178, 2005.

- [PS14] M. Pelanti and K.-M. Shyue. A mixture-energy-consistent six-equation two-phase numerical model for fluids with interfaces, cavitation and evaporation waves. *Journal of Computational Physics*, 259:331 – 357, 2014.
- [QK96] J. J Quirk and S. Karni. On the dynamics of a shock–bubble interaction. *Journal of Fluid Mechanics*, 318:129–163, 1996.
- [Rie11] F. Rieper. A low-Mach number fix for Roe's approximate Riemann solver. *Journal of Computational Physics*, 230(13):5263–5287, 2011.
- [SA99] R. Saurel and R. Abgrall. A simple method for compressible multifluid flows. *SIAM J. Sci. Comput.*, 21 (3):1115–1145, 1999.
- [Shy04] K.-M. Shyue. A fluid-mixture type algorithm for barotropic two-fluid flow problems. *Journal of Computational Physics*, 200:718–748, 2004.
- [SPB09] R. Saurel, F. Petitpas, and R. A. Berry. Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures. *Journal of Computational Physics*, 228(5):1678 – 1712, 2009.
- [SW84] H. B. Stewart and B. Wendroff. Two-phase flow: Models and methods. *Journal of Computational Physics*, 56(3):363 – 409, 1984.
- [TMD<sup>+</sup>08] B. Thornber, A. Mosedale, D. Drikakis, D. Youngs, and R.J.R. Williams. An improved reconstruction method for compressible flows with low Mach number features. *Journal of Computational Physics*, 227(10):4873 – 4894, 2008.
  - [Tur87] E. Turkel. Preconditioned methods for solving the incompressible and low speed compressible equations. *Journal of computational physics*, 72(2):277–298, 1987.