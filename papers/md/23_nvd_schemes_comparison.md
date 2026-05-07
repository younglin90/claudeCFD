# A general pressure equation based method for incompressible two-phase flows

Hormuzd Bodhanwalla<sup>a</sup> , Dheeraj Raghunathan<sup>a</sup> , Y. Sudhakara,<sup>∗</sup>

<sup>a</sup>School of Mechanical Sciences, Indian Institute of Technology Goa, Farmagudi, Goa-403401, India

# Abstract

We present a fully-explicit, iteration-free, weakly-compressible method to simulate immiscible incompressible two-phase flows. To update pressure, we circumvent the computationally expensive Poisson equation and use the general pressure equation which is solved explicitly. In addition, a less diffusive algebraic volume-of-fluid approach is used as the interface capturing technique and in order to facilitate improved parallel computing scalability, the technique is discretised temporally using the operator-split methodology. Our method is fully-explicit and stable with simple local spatial discretization, and hence, it is easy to implement. Several twoand three-dimensional canonical two-phase flows are simulated. The qualitative and quantitative results prove that our method is capable of accurately handling problems involving a range of density and viscosity ratios and surface tension effects.

Keywords: general pressure equation, two-phase flow, volume-of-fluid, Runge-Kutta, operator-split

#### 1. Introduction

Many practical applications of fluid mechanics involve two-phase flows in which two immiscible fluids of different densities and viscosities interact and generate complex flow patterns. To accurately simulate such flows, capturing the complex evolution of the interface topology is imperative. The common approaches followed are the volume-of-fluid (VOF) [\[1–](#page-31-0) [3\]](#page-31-1), level-set [\[4](#page-31-2)[–6\]](#page-31-3), and phase-field methods [\[7–](#page-31-4)[9\]](#page-32-0). The VOF and level-set methods are sharp interface approaches where there exists a jump in the material properties across the interface. On the other hand, in the case of the phase-field method, the interface is assumed to be of a finite thickness across which the properties vary rapidly but smoothly. All these approaches fall under the category of one-fluid formulation [\[10\]](#page-32-1), which is widely used in the simulation of incompressible two-phase flows. The numerical treatment of the incompressible Navier-Stokes (INS) equations requires solving a Poisson equation for pressure at each time step

<sup>∗</sup>Corresponding author, Email: sudhakar@iitgoa.ac.in

to enforce mass conservation [\[11\]](#page-32-2). Although the Poisson equation provides accurate flow fields, it requires a linear algebra solver due to its elliptic nature. This step makes the solver computationally expensive and offers limited parallel computing scalability.

#### 1.1. Overview of weakly compressible approaches

The scalability issue described above can be overcome by relaxing the incompressibility constraint. There are three such weakly-compressible alternatives to overcome this limitation: Lattice Boltzmann methods, artificial compressibility methods, and techniques based on low Mach number (Ma) pressure equation. While the first two categories of methods are wellestablished, the third is a recent development in the field.

# Lattice Boltzmann method (LBM).

Due to the attractive features of being explicit in time and instinct kinetic nature, LBM has been used to develop models for solving the two-phase flow problems [\[12,](#page-32-3) [13\]](#page-32-4). However, due to the numerical instability, their application is restricted to lowand moderate-density ratios. Several attempts have been made to develop LBM capable of handling large density ratios [\[14,](#page-32-5) [15\]](#page-32-6). Still, due to the well-known fact that such methods utilize many distribution functions and consume large memory, the technique becomes computationally expensive for complex multiphase problems.

## Artificial compressibility method (ACM).

The main idea of the ACM, pioneered by Chorin [\[16\]](#page-32-7), is to replace the velocity divergence condition with an artificial hyperbolic equation for pressure, which is computationally more convenient to solve than the elliptic equation. The original ACM is proposed to solve steady flows only and is subsequently extended to simulate unsteady flows by introducing a dual time-stepping procedure [\[17,](#page-32-8) [18\]](#page-32-9). Kelecy and Pletcher [\[19\]](#page-32-10) and Shah and Yuan [\[20\]](#page-32-11) employed such dual time-stepping ACM techniques to solve two-phase flows. Although such methods produce time-accurate results, the necessity of convergence of pseudo-time iterations within each physical time step makes them computationally inefficient.

#### Methods based on low Mach number (Ma) pressure equation.

In recent years, there has been a surge of interest in obtaining an evolution equation for pressure based on thermodynamic considerations of the compressible Navier-Stokes equations. In contrast to an artificial pressure evolution equation used in ACM, such methods rely on the equation for pressure derived based on the physical arguments at low Ma. Methods belonging to this category retain the primary advantage of ACMs that an explicit time integration can be employed; moreover, the use of a physics-based pressure equation eliminates the need for dual time-stepping. Thus, such methods are computationally more efficient than dual-time schemes for simulating unsteady fluid flows. Three such approaches are proposed in the literature:

- kinetically reduced local Navier Stokes equations (KRLNS)
- entropically damped artificial compressibility method (EDAC)

• a method based on general pressure equation (GPE)

In the following text, we briefly discuss these three methods, and emphasize their status in simulating two-phase flows.

KRLNS uses a grand potential as a thermodynamic variable to derive a low Ma description from the compressible Navier-Stokes equations [\[21\]](#page-32-12). Although this method solves an evolution equation for the grand potential instead of pressure, due to the algorithmic similarity with the relevant methods, we grouped it here. Simulations of standard test cases for single-phase flows prove the applicability of KRLNS to unsteady incompressible fluid flows [\[22–](#page-32-13)[25\]](#page-32-14). However, the simulation of two-phase flows using this equation is yet to be explored.

Using entropy to damp the acoustic pressure waves, Clausen [\[26\]](#page-32-15) proposed a parabolic equation for pressure, and the resulting framework is called EDAC. The method is shown to provide accurate results for laminar and turbulent single-phase flows [\[27,](#page-32-16) [28\]](#page-32-17). Kajzer and Pozorski [\[29\]](#page-33-0) developed an EDAC-based diffuse interface approach to simulate twophase flows. They turned off the pressure diffusion term in the vicinity of the interface to eliminate pressure oscillations. Qualitative comparisons of droplet problems showed that the method captures the complex topological features well. The same authors proposed an improved method [\[30\]](#page-33-1) to eliminate the drawbacks of [\[29\]](#page-33-0). The interface identification is made Ma independent, and the time-step restriction is relaxed. Both qualitative and quantitative investigations of standard test cases are used to demonstrate that the improved method can accurately capture complex two-phase flows.

Toutant [\[31\]](#page-33-2) derived the GPE using an asymptotic analysis of compressible Navier-Stokes equations. It has been shown that the GPE-based single-phase flow solver can accurately capture the transient incompressible flows [\[32\]](#page-33-3). Studies aimed at investigating the applicability of GPE [\[33,](#page-33-4) [34\]](#page-33-5) concluded that GPE can provide solutions very close to that of INS, even for wall-bounded turbulent flows. Huang [\[35\]](#page-33-6) proposed a method, based on GPE, to simulate two-phase flows. The interface evolution and the surface tension effects are modelled using a phase field approach. The method produces results of two-phase flows that are in good agreement with INS solvers and LBM.

Another recent weakly compressible approach that solves an evolving pressure projection equation to simulate two-phase flows is proposed by Yang and Aoki [\[36\]](#page-33-7). Although they solve a low Ma pressure equation, this approach involves, within each time step, a few iterations of the pressure evolution equation and the associated velocity correction. The inner iterations help in alleviating the acoustic effects. The method is successfully validated against complex two-phase flow problems.

#### 1.2. Novelty and a brief overview of the present methodology

Methods based on the low Ma pressure equation represent the state-of-the-art weakly compressible techniques to simulate incompressible flows. However, the development of numerical methods to solve two-phase flows under this framework is a fresh topic. To the best of our knowledge, only three research works, two using EDAC [\[29,](#page-33-0) [30\]](#page-33-1) and one (unpublished) using GPE [\[35\]](#page-33-6), reported two-phase flow simulations using low Ma pressure equations. All three of them used a phase-field approach to model the interface dynamics. While the phase-field approach has certain advantages, additional numerical parameters related to the interface width and compression need to be set appropriately, and these parameters noticeably influence the solution accuracy and solver stability. Moreover, existing works introduce additional complicated procedures to achieve accurate results: Kajzer and Pozorski [\[29,](#page-33-0) [30\]](#page-33-1) used a switch parameter and used a distinct discretization scheme in the interface region when compared to the bulk of fluids; Huang [\[35\]](#page-33-6) had to use second viscosity term to eliminate the checkerboard oscillations and employed a special averaging procedure to compute pressure gradient without which solution stopped abruptly. Despite these additional procedures, the method of [\[35\]](#page-33-6) can only deal with low density and viscosity ratios. This paper proposes a simple and robust fully-explicit GPE-based method to simulate complex two-phase flows with large viscosity and density ratios, using VOF to capture the interface.

Due to the ease of implementation and extension to three-dimension, the algebraic VOF scheme is used in the present work. Various algebraic VOF schemes can be found in the literature [\[37](#page-33-8)[–39\]](#page-33-9), each having its own advantages. In this work, we employ the modified switching technique for advection and capturing of surfaces (MSTACS) introduced by Anghan et al. [\[39\]](#page-33-9) since it offers low interface diffusivity. Traditionally, the algebraic VOF schemes employ implicit (unsplit) time discretization (except THINC scheme [\[40\]](#page-33-10)), which makes their application computationally expensive, especially in 3D. To overcome this drawback, Saincher and Sriram [\[41\]](#page-33-11) recently proposed the operator-split (OS) framework for algebraic VOF, which provides a fully-explicit treatment of the volume fraction advection equation resulting in faster computations than the traditional approach.

The contributions of this paper are as follows: (i) we incorporate the operator-split framework with MSTACS formulation as an interface capturing technique which gives us the benefit of explicit time integration and low diffusivity of the interface, and (ii) we propose a GPE-based iteration-free method to solve two-phase flows with large density and viscosity ratios without any special treatment in the interface region, or ad-hoc procedure for stabilisation. The combined benefit of GPE that eliminates the pressure Poisson equation, and the OS framework for algebraic VOF enables us to employ a fully-explicit solution algorithm.

#### 1.3. Structure of this paper

The rest of the article is structured as follows: The governing equations and details of the GPE-based two-phase solver with the VOF technique are described in § [2.](#page-3-0) Details of the discretisation, time integration, and algorithmic aspects are elaborated in § [3.](#page-7-0) The validation of the proposed method against existing literature results is reported in § [4.](#page-10-0) Finally, the conclusions drawn from the analysis are reported in § [5.](#page-28-0)

### 2. Mathematical Model

The present method is based on 'one fluid formulation', in which a single set of equations governs the behaviour of both fluid phases [\[10\]](#page-32-1). In this section, we discuss the governing equations and the details of algebraic VOF.

#### 2.1. Governing Equations

The weakly compressible isothermal Navier-Stokes equations considered in the present work are written as

$$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \nabla \cdot \left[ \mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right) \right] + \mathbf{F}, \tag{1}$$

$$\frac{\partial p}{\partial t} + \rho c_s^2 (\nabla \cdot \mathbf{u}) = \frac{1}{\rho} \nabla \cdot (\mu \nabla p), \qquad (2)$$

where equation (1) represents the well-known momentum equation, and equation (2) is the GPE derived by Toutant [31], with the assumption of  $\gamma = \Pr$  (see Appendix A). The GPE is derived for single phase flows, and we use the standard one-fluid formulation to simulate two-phase flows.  $\mathbf{u}$  and p refer to the velocity vector and pressure, respectively.  $c_s$  is the artificial speed of sound, whose value should be sufficiently large to replicate the behaviour of incompressible flows.  $\rho$  and  $\mu$  refer to the fluid's mixture density and dynamic viscosity. They are evaluated as

$$\rho = C\rho_1 + (1 - C)\rho_2, 
\mu = C\mu_1 + (1 - C)\mu_2,$$
(3)

where the subscripts 1 and 2 represent the primary and secondary phases, respectively. The volume fraction C is a Heaviside function that jumps across the interface between the two phases. The body force  $\mathbf{F}$  appearing in equation (1) is given as

$$\mathbf{F} = \mathbf{F_g} + \mathbf{F_s},\tag{4}$$

where  $\mathbf{F_g} = \rho \mathbf{g}$  is the gravitational force and  $\mathbf{F_s}$  is the surface tension force per unit volume given as  $\sigma \kappa \hat{\mathbf{n}} \delta$ .  $\mathbf{g}$  denotes gravitational acceleration,  $\sigma$  is the surface tension coefficient,  $\hat{\mathbf{n}}$  is the outward unit normal,  $\kappa$  is the local curvature of the interface, and  $\delta$  is the Dirac delta function.

For completeness, we present the non-dimensional form of the GPE (in the case of two-phase flow) below

$$\frac{\partial p}{\partial t} + \frac{\rho^*}{\mathrm{Ma}^2} (\nabla \cdot \mathbf{u}) = \frac{1}{\rho^* \mathrm{Re}} \nabla \cdot (\mu^* \nabla p), \qquad (5)$$

where Re and Ma denote the Reynolds number and the artificial Mach number, respectively.  $\rho^*$  and  $\mu^*$  refer to the non-dimensional density and viscosity, respectively. Additional parameters ( $\gamma$  and Pr) originally present in the GPE do not appear in the above expression, because we set  $\gamma = \text{Pr}$  based on [32].

#### 2.2. Two-phase model

Under the VOF framework, the governing equation for the transport of volume fraction C is provided as:

$$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u}C) = C \left(\nabla \cdot \mathbf{u}\right) \tag{6}$$

Due to the ease of implementation for 3D applications, the algebraic VOF technique is incorporated. In this method, the interface capturing is accomplished by blending a Compressive Differencing Scheme (CDS) and a High Resolution (HR) scheme. The switching between the two depends on the interface orientation with respect to the flow direction. Traditionally, in the algebraic VOF schemes, the Crank-Nicolson discretization is used for the volume fraction transport equation [\[37,](#page-33-8) [39\]](#page-33-9). However, we use the operator-split technique, recently introduced by Saincher and Sriram [\[41\]](#page-33-11), to make our algorithm free of any linear algebra solver. The OS is a multi-stage approach with the number of stages equal to the problem's spatial dimensions. For a 3D scenario, the volume fraction is updated as follows:

x − sweep :

$$C_P^* = C_P^{(n)} + \frac{\Delta t}{\Delta x} \left( C_w^{(n)} U_w^{(n)} - C_e^{(n)} U_e^{(n)} \right) + c f_P^{(n)} \frac{\Delta t}{\Delta x} \left( U_e - U_w \right)^{(n)}$$
(7)

y − sweep :

$$C_P^{**} = C_P^* + \frac{\Delta t}{\Delta y} \left( C_s^* V_s^{(n)} - C_n^* V_n^{(n)} \right) + c f_P^{(n)} \frac{\Delta t}{\Delta y} \left( V_n - V_s \right)^{(n)}$$
(8)

z − sweep :

$$C_P^{(n+1)} = C_P^{**} + \frac{\Delta t}{\Delta z} \left( C_b^{**} W_b^{(n)} - C_f^{**} W_f^{(n)} \right) + c f_P^{(n)} \frac{\Delta t}{\Delta z} \left( W_f - W_b \right)^{(n)}$$
(9)

where C <sup>∗</sup> and C ∗∗ are the updated volume fractions at the intermediate stages. The subscripts e, w, n, s, f, b denote the usual finite volume notation to represent the faces of the cell P. (U, V, W) represent components of u in the Cartesian coordinate system. The notation cf<sup>P</sup> refers to the colour function term that is active only on cells on which C > 0.5 [\[42\]](#page-33-12). The sequence of sweeping is changed after every time step [\[41\]](#page-33-11).

For a successful time advancement of the volume fraction field, it is essential to compute the flux of volume fraction over the cell face. In the algebraic VOF method, this is calculated based on the value of C in the cells straddling the face. Under the formulation of the donoracceptor scheme given by Ubbink and Issa [\[37\]](#page-33-8), the volume fraction at the face is written as:

$$C_{face} = (1 - \beta_{face}) C_D + \beta_{face} C_A \tag{10}$$

where C<sup>D</sup> and C<sup>A</sup> are the volume fractions of the donor and acceptor cells, respectively, determined based on the velocity direction at that face. The notation β refers to the weighting factor which determines the contribution of the donor and acceptor cells based on the gradient of volume fraction and is evaluated as:

$$\beta_{face} = \frac{\widetilde{C}_{face} - \widetilde{C}_{D}}{1 - \widetilde{C}_{D}} \tag{11}$$

where  $C_D = (C_D - C_U)/(C_A - C_U)$  is the normalized volume fraction value of the donor-cell with  $C_U$  being the volume fraction of the upwind cell.  $C_{face}$  is normalized volume fraction at the cell face which is given as:

$$\widetilde{C}_{face} = \gamma_{face} \left( \widetilde{C}_{face} \right)_{CDS} + \left( 1 - \gamma_{face} \right) \left( \widetilde{C}_{face} \right)_{HR} \tag{12}$$

which depicts that the volume fraction fluxed over the cell face is computed using a blend of CDS  $\left(\left(\widetilde{C}_{face}\right)_{CDS}\right)$  and HR  $\left(\left(\widetilde{C}_{face}\right)_{HR}\right)$  scheme (as mentioned earlier) through a blending function  $(\gamma_{face})$  that allows smooth switching between the two based on the orientation of the interface. The interface capturing scheme used in the present study that provides the formulation of CDS, HR and blending function is the modified switching technique for advection and capturing of surfaces (MSTACS) introduced by Anghan et al. [39]. The significant advantage that MSTACS offers is the low diffusivity over a wide range of Courant numbers,  $Cou_{adv} = U\Delta t/\Delta x$ . The scheme is formulated as follows:

$$\left(\widetilde{C}_{face}\right)_{CDS} = \begin{cases}
\min\left(\frac{\widetilde{C}_D}{Cou_{adv}}, 1\right), & 0 \leq \widetilde{C}_D \leq 1 ; 0 < Cou_{adv} \leq 1/3 \\
\min\left(3\widetilde{C}_D, 1\right), & 0 \leq \widetilde{C}_D \leq 1 ; 1/3 < Cou_{adv} \leq 1 \\
\widetilde{C}_D, & \widetilde{C}_D < 0 ; \widetilde{C}_D > 1
\end{cases}$$
(13)

$$\left(\widetilde{C}_{face}\right)_{CDS} = \begin{cases}
\min\left(\frac{\widetilde{C}_{D}}{Cou_{adv}}, 1\right), & 0 \le \widetilde{C}_{D} \le 1 ; 0 < Cou_{adv} \le 1/3 \\
\min\left(3\widetilde{C}_{D}, 1\right), & 0 \le \widetilde{C}_{D} \le 1 ; 1/3 < Cou_{adv} \le 1 \\
\widetilde{C}_{D}, & \widetilde{C}_{D} < 0 ; \widetilde{C}_{D} > 1
\end{cases}$$

$$\left(\widetilde{C}_{face}\right)_{HR} = \begin{cases}
3\widetilde{C}_{D}, & 0 \le \widetilde{C}_{D} < 1/5 \\
0.5 + 0.5\widetilde{C}_{D}, & 1/5 \le \widetilde{C}_{D} < 1/2 \\
3/8 + 3/4\widetilde{C}_{D}, & 1/2 \le \widetilde{C}_{D} < 5/6 \\
1, & 5/6 \le \widetilde{C}_{D} \le 1 \\
\widetilde{C}_{D}, & \widetilde{C}_{D} < 0 ; \widetilde{C}_{D} > 1
\end{cases}$$

$$(13)$$

$$\gamma_{face} = \min\left[\left(\cos\theta\right)^4, 1\right] \tag{15}$$

The  $\theta$  is the angle between the outward pointing unit vector normal to the interface  $(\hat{\mathbf{n}})$  and the unit vector connecting the donor and acceptor cell centers (d). Hence,  $\theta$  is defined as:

$$\theta = \cos^{-1}|\hat{\mathbf{n}} \cdot \hat{\mathbf{d}}|,\tag{16}$$

where the interface normal is computed using the Parker-Youngs method [43].

Compared with the unsplit methods, in the presence of shearing velocity fields, the operator-split approach leads to imbalanced velocities on each face of the computational cell. This imbalance leads to a violation of the boundedness criterion, implying undershoots

(C < 0) and overshoots (C > 0) in the volume fraction. Hirt and Nichols [\[1\]](#page-31-0) employed a truncation step in which a cell with C < 0 is reset to zero, and the cell with C > 1 is reset to one, which leads to a conservation error. In the present study, the conservative redistribution algorithm by Saincher and Banerjee [\[2\]](#page-31-6) is utilized to restore the boundedness of the volume fraction field, which conserves mass to zero machine accuracy. The algorithm is employed after each directional sweep of the operator-split method.

The interface capturing method, OS-MSTACS, used in the present work is fully explicit regarding time advancement. The operator-split framework offers better computational efficiency, Courant independence, and high scalability in terms of parallelization than the iterative-based solvers [\[41\]](#page-33-11). At the same time, the MSTACS scheme ensures the low diffusivity of the interface [\[39\]](#page-33-9).

## 3. Numerical Details

The governing equations described in section [2](#page-3-0) are discretized using the finite volume framework. For completeness, in this section, we present the full implementation details. Integrating the equations [\(1\)](#page-4-0) and [\(2\)](#page-4-1) over the control volume d∀ bounded by control surface dS and invoking the Gauss-divergence theorem would result in the integral form of equations:

$$\iiint_{C\forall} \frac{\partial \mathbf{u}}{\partial t} \, d\forall + \iint_{CS} \mathbf{u} \mathbf{u} \cdot \mathbf{dS} = -\frac{1}{\rho} \iiint_{C\forall} \nabla p \, d\forall + \frac{1}{\rho} \iint_{CS} \left[ \mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right) \right] \cdot \mathbf{dS}$$

$$+ \frac{1}{\rho} \iiint_{C\forall} \mathbf{F} \, d\forall$$
(17)

$$\iiint_{C\forall} \frac{\partial p}{\partial t} \, d\forall + \rho c_s^2 \iint_{CS} \mathbf{u} \cdot \mathbf{dS} = \frac{1}{\rho} \iint_{CS} \mu \nabla p \cdot \mathbf{dS}$$
(18)

#### 3.1. Spatial Discretization

The integral equations are discretized over the staggered grid arrangement to avoid checkerboard oscillations. Here, the velocities are defined at the cell faces and the scalar variables (p and C) are determined at the cell centres, as shown in figure [1.](#page-8-0) For brevity, this section will describe the discretization process for two-dimensional equations; the extension to three-dimensions is straightforward.

![](_page_8_Figure_0.jpeg)

Figure 1: Staggered grid arrangement. (a) Location of stored quantities, (b) and (c) denote zoomed view of control volume for scalar and u, respectively.

The x-component of the advection term in the momentum equations is approximated as:

$$\iint_{CS} u\mathbf{u} \cdot \mathbf{dS} \approx \left[ (Uu)_{i,j} - (Uu)_{i-1,j} \right] \Delta y + \left[ (Vu)_{i-\frac{1}{2},j+\frac{1}{2}} - (Vu)_{i-\frac{1}{2},j-\frac{1}{2}} \right] \Delta x \qquad (19)$$

where the notations in the uppercase form represent advecting component of velocity, which is obtained by linear interpolation of the neighboring cell-centered values, as shown below:

$$U_{i-1,j} = \left(u_{i-\frac{1}{2},j} + u_{i-\frac{3}{2},j}\right)/2$$

$$U_{i,j} = \left(u_{i+\frac{1}{2},j} + u_{i-\frac{1}{2},j}\right)/2$$

$$V_{i-\frac{1}{2},j+\frac{1}{2}} = \left(v_{i,j+\frac{1}{2}} + v_{i-1,j+\frac{1}{2}}\right)/2$$

$$V_{i-\frac{1}{2},j-\frac{1}{2}} = \left(v_{i,j-\frac{1}{2}} + v_{i-1,j-\frac{1}{2}}\right)/2$$
(20)

The notations in the lowercase form denote the advected velocity component. They can be approximated using various schemes (upwind, central difference, QUICK, TVD, etc.) depending on the advection dominance for the problem at hand. Hence, in the present study, the choice of advection scheme is mentioned for each test case in section 4.

The viscous term in the momentum equation for 2D space can be expanded as:

$$\mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right) = \begin{bmatrix} 2\mu \frac{\partial u}{\partial x} & \mu \left( \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \right) \\ \mu \left( \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \right) & 2\mu \frac{\partial v}{\partial y} \end{bmatrix}$$
(21)

and the x-component of its integral form can be approximated as:

$$\frac{1}{\rho} \iint_{CS} \left[ \mu \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right) \right] \cdot \mathbf{dS} \approx \frac{1}{\rho} \left\{ \left[ \left( 2\mu \frac{\partial u}{\partial x} \right)_{i,j} - \left( 2\mu \frac{\partial u}{\partial x} \right)_{i-1,j} \right] \Delta y + \left[ \left[ \mu \left( \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \right) \right]_{i-\frac{1}{2},j+\frac{1}{2}} - \left[ \mu \left( \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \right) \right]_{i-\frac{1}{2},j-\frac{1}{2}} \right\} \Delta x \right\}$$
(22)

where the mixture density ρ and viscosity µ are linearly interpolated during the discretization process.

The derivatives in the viscous term and the pressure gradient term are discretized using the second-order central difference scheme.

The discretization of the surface tension term should be consistent with the pressure gradient term, so that these two effects are in balance. The surface tension force F<sup>s</sup> = σκnˆδ is approximated using the Continuum Surface Force (CSF) model [\[44\]](#page-33-14), in which nˆδ = ∇C. The x-component of the surface tension force is discretized as:

$$\mathbf{F}_{\mathbf{s}(i-\frac{1}{2},j)} = \sigma \kappa_{i-\frac{1}{2},j} \left( \nabla C \right)_{i-\frac{1}{2},j}$$
where  $(\nabla C)_{i-\frac{1}{2},j} = \frac{C_{i,j} - C_{i-1,j}}{\Delta x}$  and  $\kappa_{i-\frac{1}{2},j} = \frac{\kappa_{i,j} + \kappa_{i-1,j}}{2}$ 

The interface curvature κ is calculated using the height function methodology [\[45\]](#page-33-15).

The velocity divergence term and the diffusion term in the GPE follow the same discretization scheme used for the advection and viscous terms in the momentum equations, respectively. The only difference is that the GPE is discretized over the p-control volume.

#### 3.2. Time integration

As mentioned earlier, the volume fraction transport equation is solved using the operatorsplit technique, as depicted by Saincher and Sriram [\[41\]](#page-33-11). For updating the solution of the momentum equation and GPE, we use a three-stage Strong Stability Preserving Runge-Kutta (SSP-RK) scheme [\[46\]](#page-33-16), which is an optimal third order scheme with SSP property. In this method, for a general partial differential equation of the following form,

$$\frac{\partial \Psi}{\partial t} = L\left(\Psi\right)$$

the variable Ψ(n) is updated to Ψ(n+1) in three stages as follows:

$$\Psi^{(1)} = \Psi^{(n)} + \Delta t L \left( \Psi^{(n)} \right) 
\Psi^{(2)} = \frac{3}{4} \Psi^{(n)} + \frac{1}{4} \Psi^{(1)} + \frac{1}{4} \Delta t L \left( \Psi^{(1)} \right) 
\Psi^{(n+1)} = \frac{1}{3} \Psi^{(n)} + \frac{2}{3} \Psi^{(2)} + \frac{2}{3} \Delta t L \left( \Psi^{(2)} \right)$$
(24)

where Ψ(1) and Ψ(2) are the variables computed at the intermediate stages. SSP schemes for time integration are proposed by Shu and Osher [\[47\]](#page-33-17). They possess large absolute stability region and small error constants [\[48\]](#page-33-18). Moreover, they preserve nonlinear stability properties even with a discontinuous solution. Although they are expensive when compared to the forward Euler method, such methods are widely used in the simulation of incompressible two-phase flows using the weakly compressible framework [\[29,](#page-33-0) [30,](#page-33-1) [36,](#page-33-7) [49–](#page-33-19)[51\]](#page-33-20).

#### 3.3. Solution algorithm

For clarity, we summarize the complete solution algorithm to march from time instant (n) to (n + 1) below,

- 1. Compute C (n+1) using C (n) and u (n) by operator-split techique (eqns.[\(7\)](#page-5-0)-[\(9\)](#page-5-1)) using MSTACS (eqns. [\(10\)](#page-5-2)-[\(16\)](#page-6-0)).
- 2. Calculate ρ (n+1) and µ (n+1) using C (n+1) (eqn. [\(3\)](#page-4-2)).
- 3. Compute curvature (κ) using C (n+1) by height function technique.
- 4. Calculate velocity and pressure at the current time level (u (n+1) and p (n+1)) using third order SSP-RK method (eqn. [\(24\)](#page-9-0)), the steps of which are as follows:
  - a. Compute u (1) using u (n) , p (n) , ρ (n+1) and µ (n+1) .
  - b. Evaluate p (1) using u (1) in the divergence term (acting as a source in GPE), p (n) , ρ (n+1) and µ (n+1) .
  - c. Calculate u (2) using u (n) , u (1) , p (1) , ρ (n+1) and µ (n+1) .
  - d. Compute p (2) using u (2) in the divergence term, p (n) , p (1) ρ (n+1) and µ (n+1) .
  - e. Calculate u (n+1) using u (n) , u (2) , p (2) , ρ (n+1) and µ (n+1) .
  - f. Finally compute p (n+1) using u (n+1) in the divergence term, p (n) , p (2) ρ (n+1) and µ (n+1) .

#### 4. Benchmarking with Literature

The present section elucidates the performance of the GPE-based two-phase flow solver proposed in this paper. We simulate twoand three-dimensional canonical two-phase problems involving large density and viscosity ratios, with and without surface tension. Results from our method are rigorously validated with the existing literature qualitatively and quantitatively.

A key computational challenge associated with weakly compressible approaches is the additional time-step restriction introduced by the artificial acoustic speed (cs). This is defined in terms of the acoustic Courant number (Couacs = cs∆t/∆x). In order to accurately represent incompressible flows, we need c<sup>s</sup> ≫ 1, but this requires a much smaller time step. Typically, this stability criterion decides the time step of the simulation, and this is more restrictive than the allowable time-step of conventional pressure Poisson equation based approaches (∆tINS); in general, ∆tGP E ≈ Ma∆tINS [\[32\]](#page-33-3). Hence, on a serial execution, weakly compressible approaches require a longer simulation time. This is a well-known limitation of weakly compressible techniques, including the Lattice Boltzmann method and approaches based on smoothed particle hydrodynamics. However, it is envisaged that these methods can achieve better computational efficiency due to the potential they offer for more effective HPC implementation. The focus of the present work is to propose an accurate GPE-based approach to model two-phase flows. Discussion on the computational benefits of the method with respect to conventional approaches requires careful investigation, which is beyond the scope of the current work.

Analogous to Lattice Boltzmann methods [15, 35] and GPE [35], we set the artificial speed of sound,  $c_s = \Delta x/(\sqrt{3}\Delta t)$ , which corresponds to  $Cou_{acs} = 1/\sqrt{3}$ . However, for problems without surface tension effects, we found that the solver is stable up to  $Cou_{acs} \leq 1.25$ , and the time-step is chosen appropriately.

Since the combination of OS framework and MSTACS formulation is used for the first time, it is necessary to validate this algorithm. Hence, to begin with, the ability of the OS-MSTACS scheme to accurately capture the interface is demonstrated. Then, to verify the correct implementation of the surface tension model, the static droplet test case is simulated. Later, the complete algorithm is tested by simulating several two-phase flow configurations and comparing the results with the published ones. We start with the problem of 2D Rayleigh-Taylor instability, which is a canonical test case with no surface tension effects and produces moderately complex interfacial changes. We simulate the dam break test to validate the solver for a high-density ratio problem. Next, we examine the bubble rise test case that involves a strong influence of surface tension effects. Finally, to extend our study to three dimensions, a 3D Rayleigh-Taylor instability is simulated and benchmarked against the existing results reported in the literature.

#### 4.1. Validation of the VOF method

To quantitatively demonstrate the accuracy of the interface tracking method (OS-MSTACS), a classical volume fraction advection test [41, 52] case involving time-reversing shearing velocity field is simulated in a 3D environment. The test case is initialized with a sphere of radius 0.15 centered at (0.5, 0.75, 0.25) in a cuboid of size  $1.0 \times 1.0 \times 2.25$ . The patch of sphere is subjected to a shearing velocity field given below as:

$$u = \sin(2\pi y)\sin^2(\pi x)\cos\left(\frac{\pi t}{T}\right),$$

$$v = -\sin(2\pi x)\sin^2(\pi y)\cos\left(\frac{\pi t}{T}\right),$$

$$w = \left(1 - 2\sqrt{(x - 0.5)^2 + (y - 0.5)^2}\right)^2\cos\left(\frac{\pi t}{T}\right),$$
(25)

where the cosine term implies time reversal: the sign of the velocity components change at t = T/2. The total simulation time T is taken as 9 and the time step corresponding to Courant number  $Cou_{adv} = 0.1$  is used. A uniform grid of  $128 \times 128 \times 288$  is used to simulate the problem.

The interface evolution predicted by OS-MSTACS is depicted in figure 2. The figure shows a smooth interface topology of the spiral (at t = T/2) which is better than the one obtained by [41] with OS-CICSAM. This is attributed to a lower numerical diffusion offered

by MSTACS scheme compared to CICSAM [39]. After t = T/2, the deformed sphere reverts to its initial position and the numerical error for the same is quantified by computing the  $L_1$ -norm and the global volume error (expressed in %) of the volume fraction which are defined as:

$$E_{L_1} = \frac{\sum_{n=1}^{N} |C_n^T \forall_n - C_n^0 \forall_n|}{\sum_{n=1}^{N} C_n^0 \forall_n},$$

$$E_G = \frac{|\sum_{n=1}^{N} C_n^T \forall_n - \sum_{n=1}^{N} C_n^0 \forall_n|}{\sum_{n=1}^{N} C_n^0 \forall_n} \times 100\%$$

where N denotes the total number of cells,  $\forall_n$  denotes the cell volume; superscripts 0 and T refer to the initial and final time volume fractions, respectively.

![](_page_12_Figure_3.jpeg)

Figure 2: Interface shape (C=0.5 iso-contour) of the volume fraction advection test case: t=0 in blue, t=T/2 (maximum interface deformation) in purple and t=T in red.

Table 1: Comparison of volume errors for MSTACS, OS-CICSAM and OS-MSTACS.

| Volume error | MSTACS [41]            | OS-CICSAM [41]         | OS-MSTACS              |
|--------------|------------------------|------------------------|------------------------|
| $E_{L_1}$    | $2.307 \times 10^{-1}$ | $2.075 \times 10^{-1}$ | $1.988 \times 10^{-1}$ |
| $E_G$        | $2.537 \times 10^{-4}$ | $8.175 \times 10^{-4}$ | $6.568 \times 10^{-8}$ |

The present simulation resulted in the  $L_1$  error of  $E_{L_1} = 1.988 \times 10^{-1}$  and global volume error of  $E_G = 6.568 \times 10^{-8}$  which is better than the OS-CICSAM method and the unsplit MSTACS, as can be seen from table 1. This is due to the superior formulation of MSTACS

scheme compared to CICSAM. We can conclude from these results that the MSTACS with operator split can accurately capture the evolution of the interface.

One of the contributions of the present work is the formulation of OS-MSTACS. Hence, it is of interest to compare the performance of MSTACS and OS-MSTACS. We repeated the volume fraction advection test for Couadv of 0.1, 0.5 and 1.0 and compared E<sup>L</sup><sup>1</sup> , E<sup>G</sup> and the computational time; these data are reported in table [2.](#page-13-0) Saincher and Sriram [\[41\]](#page-33-11) provided a comparison of different schemes, in which they solved the system of linear algebraic equations with tolerance set to 10<sup>−</sup><sup>13</sup>. In contrast, in this work, we set the tolerance for unsplit MSTACS to 10<sup>−</sup><sup>6</sup> in order to compare the accuracy for approximately the same computational cost. From table [2,](#page-13-0) it can be seen that for all Courant numbers, OS-MSTACS provides more accurate results despite requiring reduced computational time. The conclusion that OS-MSTACS is more accurate remains the same even if the tolerance for the unsplit MSTACS is set to 10<sup>−</sup><sup>13</sup>, as can be seen from table [1.](#page-12-1) We used an Intel Core i9-9920X CPU with a clock speed of 3.50GHz for these comparison studies.

Table 2: Comparison of simulation time and volume errors for OS-MSTACS and unsplit MSTACS with convergence criterion 10<sup>−</sup><sup>6</sup> .

|        |                   | MSTACS           |           |                   | OS-MSTACS         |          |
|--------|-------------------|------------------|-----------|-------------------|-------------------|----------|
| Couadv | EL1               | EG               | Time (s)  | EL1               | EG                | Time (s) |
| 0.1    | 10−1<br>2.82<br>× | 100<br>2.17<br>× | 101562.38 | 10−1<br>1.98<br>× | 10−8<br>6.51<br>× | 76713.58 |
| 0.5    | 10−1<br>5.32<br>× | 100<br>4.95<br>× | 23612.99  | 10−1<br>2.44<br>× | 10−6<br>1.43<br>× | 14909.49 |
| 1.0    | 10−1<br>×<br>9.42 | 101<br>×<br>1.14 | 14006.22  | 100<br>×<br>1.05  | 10−1<br>×<br>1.93 | 7447.58  |

## 4.2. 2D static droplet

To ensure correct implementation of the surface tension model, we examine the 2D static droplet test case. We place a droplet of diameter D = 1m at the centre of a domain of 2m×2m. We set the density of the two inviscid fluids as ρ<sup>1</sup> = 1000kg/m<sup>3</sup> and ρ<sup>2</sup> = 1kg/m<sup>3</sup> , the surface tension coefficient, σ = 1N/m and the speed of sound, c<sup>s</sup> = 5m/s (corresponding to Ma=0.2 based on the velocity scale, U<sup>σ</sup> = p σ/ρ2D [\[53\]](#page-34-1)). The problem setup is the same as that of Yang and Aoki [\[36\]](#page-33-7), including the initial condition for pressure. On all boundaries, we enforce the free-slip condition for velocity and zero normal gradient for pressure. A uniform grid of 128 × 128 is used and the time step ∆t = 1.25 × 10<sup>−</sup><sup>3</sup> s. We examine the results at t = 0.625s, as discussed by Yang and Aoki [\[36\]](#page-33-7).

![](_page_14_Figure_0.jpeg)

Figure 3: Results of the static droplet test case at t = 0.625s. (a) Surface plot of pressure in the domain (b) Line plot of pressure through the centreline along the x-axis.

In the static environment, the suspended droplet under equilibrium achieves a balance between the pressure and surface tension forces. The theoretical pressure jump across the interface is given as ∆p = 2σ/D, and the obtained pressure jump is in good agreement with the same. This is directly evident from figure [3,](#page-14-0) which shows the pressure in the computational domain.

Next, we examine the presence of parasitic (spurious) currents, which are indications of the discretisation error associated with the surface tension model used. Figure [4](#page-15-0) compares the magnitude of these spurious currents obtained from GPE and INS solvers. We observe that these are of the same order, and the difference is negligible. We emphasize that the obtained |u|max is lower than that reported by another weakly compressible solver [\[36\]](#page-33-7) for which |u|max= 8 × 10<sup>−</sup><sup>3</sup> .

![](_page_15_Figure_0.jpeg)

Figure 4: Magnitude of spurious currents for the static droplet test case from (a) GPE solver (b) INS solver, at t = 0.625s.

We further examine the effect of Ma on the spurious currents. In order to do this, we plot |u| for different Ma as shown in figure [5.](#page-15-1) We use c<sup>s</sup> = 10, 20 and 100 corresponding to Ma = 0.1, 0.05 and 0.01. Similar to the findings of [\[54\]](#page-34-2), we observe that Ma doesn't influence the magnitude of spurious currents. Although not visible clearly, the acoustic waves become less pronounced as we lower the Ma, an observation we discuss in detail in the subsequent test cases.

![](_page_15_Figure_3.jpeg)

Figure 5: Effect of Mach number on spurious currents associated with static droplet test case. The contours of velocity magnitude (|u|) is plotted for (a) Ma = 0.1 (b) Ma = 0.05 and (c) Ma = 0.01, at t = 0.625s.

Table 3: Effect of Laplace number on spurious currents for the static droplet test case quantified by the maximum value of velocity.

| La    | u max            |  |  |
|-------|------------------|--|--|
| ∞     | 10−4<br>×<br>4.6 |  |  |
| 12000 | 10−2<br>×<br>1.2 |  |  |
| 1200  | 10−2<br>2.5<br>× |  |  |
| 120   | 10−2<br>4.0<br>× |  |  |

Finally, we also examine the influence of Laplace number, defined as La = ρDσ/µ<sup>2</sup> , on the spurious currents. The values of |u|max are presented in table [3](#page-16-0) for different values of La ranging from 120 to infinity. We observe that, unlike other INS-based works [\[53\]](#page-34-1), where the |u|max decreases with decreasing La, we obtain an increase in |u|max when La is reduced. However, this result is consistent with those reported for other similar weakly compressible formulations [\[36,](#page-33-7) [54\]](#page-34-2). From these observations, we can conclude that the implemented model for the surface tension is accurate enough compared to other weakly compressible solvers.

#### 4.3. 2D Rayleigh-Taylor Instability

When a system comprising a heavier fluid gently placed atop a lighter fluid is slightly perturbed externally, an interfacial instability called Rayleigh-Taylor instability is manifested. Due to gravity, the heavier fluid penetrates the lighter one. This results in the evolution of the complex topology of the interface, and the prediction of the interface shape is a standard test case for the two-phase flows. The test case is adopted from Garoosi et al. [\[55\]](#page-34-3); the parameters are listed in table [4.](#page-17-0)

The initial configuration for simulating the RT instability is shown in figure [6](#page-17-1) wherein the heavier fluid (C = 1.0) is placed atop the lighter fluid (C = 0.0) separated by an interface (0 < C < 1). As shown in the figure, the interface is sinusoidally perturbed at the initial time instant to trigger the interfacial instability. All the boundaries are equipped with no-slip boundary condition, and the configuration is initialized with zero velocities and pressure. The advection scheme used for the test case is the second-order central difference scheme for discretizing the advected velocity component, and the simulation is carried out for 5 non-dimensional time units (t <sup>∗</sup> = t p g/L).

|  | Table 4: Parameters for Rayleigh-Taylor instability adopted from Garoosi et al. [55]. |  |
|--|---------------------------------------------------------------------------------------|--|
|  |                                                                                       |  |

| Name                    | Definition                | Value           |
|-------------------------|---------------------------|-----------------|
| Length                  | L<br>(m)                  | 1.0             |
| Height                  | W<br>(m)                  | 2.0             |
| Heavier fluid density   | (kg/m3<br>ρ1<br>)         | 1.8             |
| Heavier fluid viscosity | µ1<br>(kg/ms)             | 0.018           |
| Lighter fluid density   | (kg/m3<br>ρ2<br>)         | 1.0             |
| Lighter fluid viscosity | µ2<br>(kg/ms)<br>√        | 0.01            |
| Reynolds number         | Re =<br>ρ1L<br>gL/µ1<br>√ | 420             |
| Froude number           | 2/(gL)<br>Fr = (<br>gL)   | 1.0             |
| Grid resolution         | Nx<br>×<br>Ny             | 200<br>×<br>400 |

![](_page_17_Figure_2.jpeg)

Figure 6: Initial flow configuration for Rayleigh-Taylor instability including the initial perturbation and boundary conditions.

The temporal evolution of the interface is qualitatively assessed at t <sup>∗</sup> = 1, 3, 5 with a time step of 10<sup>−</sup><sup>4</sup> and Ma = 0.05 for the simulation. As shown in figure [7\(a\),](#page-18-0) due to the existence of gravity and density gradient between the two immiscible fluids, the heavier fluid accelerates into the lighter fluid volume. Simultaneously, owing to the buoyancy force, the lighter fluid flows upward into the heavier fluid. As a result, the amplitude of the sinuous perturbation starts evolving into mushroom-shaped plumes, as seen in figure [7\(b\).](#page-18-1) As time progresses, the instability increases and the resulting complex interface topology can be observed in figure [7\(c\).](#page-18-2) Qualitatively, our flow field matches well with those reported in the literature.

![](_page_18_Figure_0.jpeg)

Figure 7: Temporal evolution of the interface for RT instability visualized at (a)  $t^* = 1$  (b)  $t^* = 3$  and (c)  $t^* = 5$ , (d) position of the maximum and minimum fluid fronts with time.

The maximum (minimum) position of the lighter fluid (heavier fluid) is tracked with time and compared with Garoosi et al. [55]. It is evident from figure 7(d) that the evolution of the interface position with time is in excellent agreement with the existing literature.

![](_page_19_Figure_0.jpeg)

Figure 8: Effect of Mach number on RT instability. (a) and (b) denote the interface shapes for  $t^* = 2$  and  $t^* = 3$ , respectively, (c) position of the maximum and minimum fluid fronts with time.

Toutant [32] studied the effect of Mach number on the accuracy of single-phase flow simulations. The Mach number appearing in equation (5) is an artificial parameter in GPE. The choice of Ma is crucial to dictate the solver's accuracy and computational efficiency; hence, we study the effect of the Mach number for this problem. Figure 8 depicts the effect of Mach number for the proposed method, and in this study, the values considered are 0.01, 0.05, 0.1 and 0.2. It is apparent from figures 8(a) and 8(b) that for the case of Ma = 0.01 and Ma = 0.05, the discrepancies in the interface topology are negligible, which is also evidenced quantitatively from figure 8(c). However, for the case of Ma = 0.1, minute oscillations can be seen in the interface evolution plot for a finite simulation time. Furthermore, although Ma = 0.2 provides a spatially smooth interface shape, unphysical oscillations are predominantly observed in the time evolution plot, especially during the initial time instants. The oscillations are damped as time progresses. Clausen reports a

similar fluctuating nature of the plot for the case of single-phase flows using EDAC as the pressure transport equation [\[26\]](#page-32-15).

#### 4.4. Broken Dam flow

The previous test case involves immiscible fluids with a minor difference in density and viscosity. To investigate the accuracy of our method for problems involving a large density ratio, we simulate the broken dam flow, a classical benchmark test for multiphase flows. A rectangular liquid column collapses in this simulation and percolates over the bottom wall. The test case was first experimentally investigated by Martin and Moyce [\[56\]](#page-34-4) and numerically tested by many authors. In the present work, we adopt the numerical parameters of Kelecy and Pletcher [\[19\]](#page-32-10), also studied by Ling et al. [\[57\]](#page-34-5). The initial configuration for the simulation is shown in figure [9.](#page-20-0) Zero velocity and hydrostatic pressure are used as the initial conditions. The parameters considered for the simulation are listed in table [5.](#page-20-1) Due to the violent nature of the flow, a total variable diminishing scheme with a Van Leer limiter [\[58\]](#page-34-6) is used for discretizing the advected velocity component. The time step chosen for this problem is 10<sup>−</sup><sup>5</sup> . Analogous to lattice-Boltzmann simulations [\[15,](#page-32-6) [35\]](#page-33-6) we set the artificial speed of sound, c<sup>s</sup> = ∆x/( √ 3∆t) which corresponds to Ma = 0.018.

| Table 5: Parameters for the broken dam flow test case. |                           |                  |  |
|--------------------------------------------------------|---------------------------|------------------|--|
| Name                                                   | Definition                | Value            |  |
| Characteristic length                                  | a<br>(m)                  | 0.05715          |  |
| Length                                                 | L<br>(m)                  | 5a               |  |
| Height                                                 | W<br>(m)                  | 1.25a            |  |
| Heavier fluid density                                  | (kg/m3<br>ρ1<br>)         | 1000             |  |
| Heavier fluid viscosity                                | µ1<br>(kg/ms)             | 0.001            |  |
| Lighter fluid density                                  | (kg/m3<br>ρ2<br>)         | 1.25             |  |
| Lighter fluid viscosity                                | µ2<br>(kg/ms)             | 10−5<br>1.8<br>× |  |
| Surface tension coefficient                            | σ<br>(N/m)                | 0.0755           |  |
| Reynolds number                                        | √ga/µ<br>Re =<br>ρ1a<br>1 | 42792            |  |
| Grid resolution                                        | ×<br>Nx<br>Ny             | ×<br>400<br>100  |  |

![](_page_20_Figure_4.jpeg)

Figure 9: Initial flow configuration for the broken dam flow test case.

Figure [10](#page-22-0) depicts the temporal evolution of the interface for the considered test case. Initially, the liquid column collapses and flows over the bottom wall (figures [10\(a\)](#page-22-1) and [10\(b\)\)](#page-22-2). The liquid front reaches the right wall after time t = 0.2s, crawls up the wall (figures [10\(c\)](#page-22-3) and [10\(d\)\)](#page-22-4) and traps an air bubble near the corner, as shown in figure [10\(e\).](#page-22-5)

The propagation of the liquid front along the bottom wall and the reduction in the liquid height along the left wall are standard quantities to investigate the accuracy of this simulation. The time variation of these quantities is presented in figures [10\(f\)](#page-22-6) and [10\(g\),](#page-22-7) respectively. It can be clearly seen that both the parameters describing the interface tracking are in excellent agreement with the existing literature [\[19,](#page-32-10) [56,](#page-34-4) [57\]](#page-34-5).

![](_page_22_Figure_0.jpeg)

Figure 10: Temporal evolution of the interface for the broken dam flow visualized at (a) t = 0.1s (b) t = 0.2s and (c) t = 0.3s (d) t = 0.4s (e) t = 0.5s, (f) Propagation of the interface front along the bottom wall with time, and (g) Reduction of the liquid height along the left wall with time.

The qualitative and quantitative agreement of the present results with those reported in the literature strongly suggests that our GPE-based solver can accurately resolve the unsteady flow field of two-phase flows, even in the case of a large density ratio without any ad hoc modifications in the algorithm.

![](_page_23_Figure_0.jpeg)

![](_page_23_Figure_1.jpeg)

Figure 11: Effect of Mach number on the broken dam flow problem (a) Propagation of the interface front along the bottom wall with time, and (b) Reduction of the liquid height along the left wall with time.

Similar to the one performed for the 2D Rayleigh Taylor instability problem, we study the effect of Ma for this high-density ratio test case. The Ma values considered are 0.01, 0.018, 0.05 and 0.075; the simulations with Ma > 0.075 did not converge. However, we would like to emphasize that this is not a limitation of the present approach. In theory, weakly compressible approaches model incompressible flows when Ma ≪ 1. Figure [11](#page-23-0) depicts the effect of Mach number on the propagation of the interface front along the bottom wall and the reduction of the liquid height along the left wall with time. While figure [11\(a\),](#page-23-1) shows no noticeable difference, figure [11\(b\)](#page-23-2) indicates nonphysical oscillations at Ma = 0.075. We observe that the oscillations are clearly visible at Ma = 0.075, less pronounced when Ma = 0.05, and negligible for the lower Ma cases. This indicates that acoustic sound waves are quickly damped when Ma is small, imparting more stable and accurate results. This observation is similar to the results reported in the Rayleigh Taylor instability test case.

## 4.5. Bubble rise

In many practical applications of two-phase flows, the influence of the surface tension is high enough to manifest complex topological changes. It is of utmost importance that the numerical method is capable of producing accurate results for such cases. Hence, we test our solver against the problem of a rising gas bubble in a liquid-filled domain. This bubble rise test case was proposed as a rigorous benchmark case for two-phase flow solvers [\[59\]](#page-34-7). Two test cases from Hysing et al. [\[59\]](#page-34-7) are considered. The first test case involves moderate bubble deformation, while the second test case features significantly complex topological changes due to high density and viscosity ratios. The later test case offers a considerable challenge to the method.

The test simulation is initialized with a patch of a circle of radius r<sup>0</sup> = 0.25 centred at (0.5, 0.5) in a 1×2 rectangular domain. The initial velocities and pressure in the domain are zero, with gravity acting in the negative y-direction. The domain is equipped with no-slip boundaries in the y-direction and free-slip boundaries in the x-direction. The parameters considered for the simulation are listed in table [6.](#page-24-0) The third-order QUICK scheme is used to discretize the advected velocity component in the advection term. Note that unlike the test cases mentioned in the previous subsections, the gravitational constant for the current problem is taken as 0.98, as used in Hysing et al. [\[59\]](#page-34-7). The time step chosen for case 1 is 10<sup>−</sup><sup>4</sup> , whereas for case 2 (challenging test case) is 10<sup>−</sup><sup>5</sup> . Similar to the broken dam problem, c<sup>s</sup> = ∆x/( √ 3∆t).

Table 6: Parameters defining the bubble rise problem.

| Name                        | Definition              | Value          |                |
|-----------------------------|-------------------------|----------------|----------------|
|                             |                         | Case 1         | Case 2         |
| Surrounding fluid density   | ρ1                      | 1000           | 1000           |
| Surrounding fluid viscosity | µ1                      | 10             | 10             |
| Bubble density              | ρ2                      | 100            | 1              |
| Bubble viscosity            | µ2                      | 1              | 0.1            |
| Surface tension coefficient | σ<br>√                  | 24.5           | 1.96           |
| Reynolds number             | Re =<br>ρ1r0<br>2gr0/µ1 | 420            | 300            |
| E¨otv¨os number             | 2/σ<br>Eo = 4ρ1gr0      | 10             | 125            |
| Grid resolution             | Nx<br>×<br>Ny           | 80<br>×<br>160 | 80<br>×<br>160 |

The standard quantification parameters to describe the temporal evolution of the bubble are the circularity (C ) and the rise velocity (Vr). C is the ratio of the perimeter of the area-equivalent circle to the instantaneous perimeter of the bubble. V<sup>r</sup> refers to the mean velocity with which the bubble is rising. Mathematically,

$$\mathscr{C} = \frac{2\pi r_0}{P_b},\tag{26}$$

$$\mathscr{Y}_r = \frac{\sum_{n=1}^N |v| \forall_n}{\sum_{n=1}^N \forall_n},\tag{27}$$

where ∀<sup>n</sup> refers to the cell volume, v refers to the velocity opposing the gravity and P<sup>b</sup> denotes the instantaneous perimeter of the bubble, which is computed using the 'integrate variables' filter available in the open source tool, Paraview [\[60\]](#page-34-8). The first test case involves a high surface tension coefficient value and low density and viscosity ratios. As a result of prevailing surface tension effects, the bubble deforms moderately: the initially circular drop takes an ellipsoidal shape at the final time instant, as evidenced by figure [12\(a\).](#page-25-0) As can be seen from figures [12\(b\)](#page-25-1) and [12\(c\),](#page-25-2) the quantification parameters are in excellent agreement with Hysing et al. [\[59\]](#page-34-7) and better than the simulation results of Strubelj et al. [ ˇ [61\]](#page-34-9) and the EDAC based solver of Kajzer and Pozorski [\[30\]](#page-33-1). We want to point out that in the study of Yang and Aoki [\[36\]](#page-33-7), an evolving pressure equation is solved for a finite number of iterations to dampen the acoustic waves for case 1 of the bubble rise problem. The smooth curve obtained by Yang and Aoki, as observed in figure [12\(c\)](#page-25-2) (solid purple line), is a result of

![](_page_25_Figure_0.jpeg)

Figure 12: Results of the bubble rise problem for Case 1 (top row) and Case 2 (bottom row). Final shape of the bubble at t=3.0 (left column), and circularity ( $\mathscr{C}$ ) (centre column) and rise velocity  $\mathscr{V}_r$  (right column) of the bubble.

10 iterations of the evolving pressure equation. However, in the present study, no iterative technique is used in the algorithm, and the current results exhibit much less oscillatory behaviour than Yang and Aoki's counterpart (purple dashed-dot line).

In the second test case, the bubble undergoes a significant deformation due to the combined effect of low surface tension and a large jump in material properties across the interface, as depicted in figure 12(d). The quantification parameters are compared with those in the existing literature. As can be evidenced from figures 12(e) and 12(f), the present study results are in excellent agreement with Hysing et al. [59]. We stress that the oscillations in the plot are negligible when compared to the other relevant weakly compressible approaches [30, 36].

Figure 13(a) shows the velocity divergence  $(\nabla \cdot \mathbf{u})$  field for the challenging test case of bubble rise at time = 3.0. It is computed using the GPE-based solver and the traditional INS involving the pressure Poisson equation. The same is quantified in figure 13(b), which shows the velocity divergence along the vertical centreline of the domain at x = 0.5. The discretization details in the staggered framework for the pressure Poisson-based algorithm

![](_page_26_Figure_0.jpeg)

Figure 13: Comparison of the velocity divergence between the INS and the GPE-based solver for Case 2 of the bubble rise: (a) filled contour, with solid black lines representing interface, and (b) along the vertical centreline at x = 0.5.

are similar to the GPE-based solver, except that the algorithm features the conventional predictor-corrector approach [11]. Figure 13(b) shows that the GPE-based solver's velocity divergence is significantly higher than the INS. Despite such a large error in  $(\nabla \cdot \mathbf{u})$ , our solver can accurately capture the qualitative and quantitative features of the challenging test case. We reiterate that the oscillations present in our results are significantly lower than the other weakly compressible two-phase flow solvers [30, 36].

While the error in  $\nabla \cdot \mathbf{u}$  at a particular time presented in figure 13 reveals an important information, a more relevant quantity to consider is the root-mean-square (RMS) value of the velocity divergence, denoted as  $(\nabla \cdot \mathbf{u})_{RMS}$ , as the time evolves. For both variants of the bubble rise problem, the variation of  $(\nabla \cdot \mathbf{u})_{RMS}$  with time is illustrated in figure 14, for both GPE and INS solvers. As expected, the error produced by the INS solver is very small (of the order of  $10^{-5}$ ). However, for the GPE solver, we see that the error is of the order of  $10^{-2}$ . In comparison, Toutant [32] reported the RMS of velocity divergence error of the order of  $10^{-3}$  for simulations of single-phase flows using GPE. This larger error for the two-phase GPE solver is due to the peak occurring in the interface region, as shown in figure 13. However, key takeaways from figure 14 are as follows: (i) although the density ratio is increased by two orders of magnitude from case 1 to 2, the corresponding  $(\nabla \cdot \mathbf{u})_{RMS}$  increased only marginally, and (ii) the velocity divergence error does not get accumulated over time. The later observation indicates that the proposed method works for large density ratio problems also.

#### 4.6. 3D Rayleigh-Taylor Instability

All four previous test cases are two-dimensional. To show the proposed method's applicability to three-dimensional flows, we simulate the 3D Rayleigh-Taylor instability. The test case is adopted from Saito et al. [62] and the parameters are listed in table 7. The simulation

![](_page_27_Figure_0.jpeg)

Figure 14: Evolution of the RMS of velocity divergence with time for the INS and the GPE-based solver for (a) case 1 and (b) case 2 of the bubble rise problem.

is set up with the heavier fluid placed on top of the lighter fluid in a cuboidal domain of 1.0 × 4.0 × 1.0. The interface between the two fluids is initially perturbed sinusoidally, as shown in figure [15,](#page-28-1) to trigger the instability. The entire domain is employed with the freeslip condition on the lateral sides in conjunction with the no-slip condition at the top and bottom boundary. The advection scheme used for the test case is the second-order central difference scheme for discretizing the advected velocity component in the advection term.

Table 7: Parameters for 3D Rayleigh-Taylor instability adopted from Saito et al. [\[62\]](#page-34-10).

| Name                    | Definition                | Value                     |
|-------------------------|---------------------------|---------------------------|
| Length                  | L<br>(m)                  | 1.0                       |
| Height                  | H<br>(m)                  | 4.0                       |
| Width                   | W<br>(m)                  | 1.0                       |
| Heavier fluid density   | (kg/m3<br>ρ1<br>)         | 3.0                       |
| Heavier fluid viscosity | µ1<br>(kg/ms)             | 3.0                       |
| Lighter fluid density   | (kg/m3<br>ρ2<br>)         | 1.0                       |
| Lighter fluid viscosity | µ2<br>(kg/ms)<br>√        | 1.0                       |
| Reynolds number         | Re =<br>ρ1L<br>gL/µ1<br>√ | 512                       |
| Froude number           | 2/(gL)<br>Fr = (<br>gL)   | 1.0                       |
| Grid resolution         | Nx<br>×<br>Ny<br>×<br>Nz  | 64<br>×<br>256<br>×<br>64 |

The test case of RT instability was simulated for 4 non-dimensional time units (t <sup>∗</sup> = t p g/L), and the evolution of the interface is qualitatively assessed at t <sup>∗</sup> = 1, 2, 3, 4 with Ma = 0.05 and time step of 10<sup>−</sup><sup>4</sup> . Similar to the 2D test case, the acceleration of heavier fluid into the lighter fluid results in complex topological features. The process can be evidenced in

![](_page_28_Picture_0.jpeg)

Figure 15: Initial flow configuration for 3D Rayleigh-Taylor instability including the initial perturbation and boundary conditions.

figures [16\(a\)-](#page-29-0)[16\(d\).](#page-29-1) The position of the bubble, saddle and spike (as shown in figure [15\)](#page-28-1) are tracked with time and compared with the existing literature [\[13,](#page-32-4) [62,](#page-34-10) [63\]](#page-34-11). It is evident from figure [16\(e\)](#page-29-2) that the evolution of the interface shape and position with time is in excellent agreement with the existing literature.

As mentioned in the introduction, the only existing GPE-based method to solve twophase flows is by Huang [\[35\]](#page-33-6), who used two special procedures for a stable computation: the addition of a second viscosity term to eliminate checkerboard oscillations and a certain averaging of pressure for stabilizing 3D flows. However, we emphasize that neither of these procedures is needed in our method for accuracy and stability. Figure [17](#page-30-0) shows the pressure contour for the 3D Rayleigh-Taylor problem at the final time instant. In the figure, the contour lines depict smooth variation in pressure instead of a checkerboard pattern reported by Huang [\[35\]](#page-33-6). Moreover, the pressure gradient term is discretized in the usual manner and we observed no stability issues. Huang [\[35\]](#page-33-6) reported that the simulation stopped abruptly without the special pressure averaging. Thus, we stress that our solver offers simplicity and inherent stability for the accurate simulation of 3D flows.

#### 5. Conclusion

The pressure Poisson equation in the incompressible Navier-Stokes solver is replaced with the general pressure equation to simulate the two-phase flow problems. The method is tested against various two-phase configurations, and the results are validated with the existing literature. The validation study reveals that our algorithm accurately solves 2D and 3D problems. Our quantitative results show that the unphysical oscillations, due to pressure acoustics, are significantly smaller than the existing weakly compressible methods. In addition, the existing methods need at least one of the following for numerical stability: different treatments for the interface region, the addition of second viscosity or special aver-

![](_page_29_Figure_0.jpeg)

Figure 16: Temporal evolution of the interface for RT instability visualized at (a)  $t^* = 1$  (b)  $t^* = 2$  (c)  $t^* = 3$  and (d)  $t^* = 4$ , and (e) evolution of bubble, saddle and spike with time.

![](_page_30_Figure_0.jpeg)

Figure 17: Pressure contours at t <sup>∗</sup> = 4 for the RT instability.

aging techniques for pressure. However, our algorithm does not require such modifications, making it convenient to implement and stable even for the simulations involving high density and viscosity ratios and surface tension effects. Furthermore, thanks to the GPE and OS framework for VOF, the algorithm is fully-explicit. Hence, it can be extended to carry out high-performance simulations over multi-node CPUs or GPUs to solve more complex problems. In the world of perpetually developing high-performance resources, we need efficient solvers that can exploit computing power without compromising the accuracy of the problem. Thus, a performance study of our solver in terms of scalability can be pivotal in satisfying the aforementioned need, which serves as the future scope of our work.

#### Acknowledgements

We acknowledge the financial support provided by IIT Goa in the form of a startup grant. Simulations are run on computational resources setup from the DST-SERB Ramanujan fellowship (SB/S2/RJN-037/2018).

#### Data Availability

The data supporting this study's findings are available from the corresponding author upon reasonable request.

#### Appendix A.

In this section, we explain the form of the GPE used in the paper. The equation derived by Toutant (equation (31) of [\[31\]](#page-33-2)) is written as,

$$\frac{\partial p}{\partial t} + \rho c_s^2 \frac{\partial u_i}{\partial x_i} = \frac{\kappa}{\rho C_v} \frac{\partial^2 p}{\partial x_i \partial x_i}$$
(A.1)

where ρ is the density, µ is the dynamic viscosity, κ is the thermal conductivity, and C<sup>v</sup> is the isochoric heat capacity. We rewrite the above equation using the definitions of heat capacity ratio, γ = Cp/Cv, thermal diffusivity, α = κ/(ρCp) and Prandtl number, Pr = µ/(ρα), to obtain,

$$\frac{\partial p}{\partial t} + \rho c_s^2 \frac{\partial u_i}{\partial x_i} = \frac{\mu \gamma}{\rho \Pr} \frac{\partial^2 p}{\partial x_i \partial x_i}.$$
 (A.2)

We simplify the above equation by setting γ = Pr to obtain the following

$$\frac{\partial p}{\partial t} + \rho c_s^2 \frac{\partial u_i}{\partial x_i} = \frac{\mu}{\rho} \frac{\partial^2 p}{\partial x_i \partial x_i}.$$
 (A.3)

Equating γ and Pr is a common assumption employed in all GPE and EDAC based solvers for simulating single-phase [\[26,](#page-32-15) [28,](#page-32-17) [32–](#page-33-3)[34,](#page-33-5) [64\]](#page-34-12) and two-phase flows [\[29,](#page-33-0) [36,](#page-33-7) [54\]](#page-34-2). Clausen [\[26\]](#page-32-15) directly used this assumption while deriving the EDAC equation whereas, Toutant [\[31\]](#page-33-2) retained κ and C<sup>v</sup> in his formulation of GPE. However, while showcasing the capability of GPE to simulate benchmark test cases [\[32\]](#page-33-3), he set γ = Pr, and this is later followed by all the others working with GPE. Finally, expressing the diffusion term similar to that of the momentum equation, we wrote the above equation in vector notation to get equation (2) presented in the paper.

### References

- [1] C. W. Hirt and B. D. Nichols, "Volume of fluid (VOF) method for the dynamics of free boundaries," Journal of computational physics, vol. 39, no. 1, pp. 201–225, 1981.
- [2] S. Saincher and J. Banerjee, "A redistribution-based volume-preserving PLIC-VOF technique," Numerical Heat Transfer, Part B: Fundamentals, vol. 67, no. 4, pp. 338–362, 2015.
- [3] H. Bodhanwalla, C. Anghan, and J. Banerjee, "The effect of one-sided confinement on nappe oscillations in free falling liquid sheet," Physics of Fluids, vol. 34, no. 12, p. 124107, 2022.
- [4] M. Sussman, P. Smereka, and S. Osher, "A level set approach for computing solutions to incompressible two-phase flow," Journal of Computational Physics, vol. 114, no. 1, pp. 146–159, 1994.
- [5] V. H. Gada and A. Sharma, "On a novel dual-grid level-set method for two-phase flow simulation," Numerical Heat Transfer, Part B: Fundamentals, vol. 59, no. 1, pp. 26–57, 2011.
- [6] V. H. Gada and A. Sharma, "Analytical and level-set method based numerical study on oil–water smooth/wavy stratified-flow in an inclined plane-channel," International Journal of Multiphase Flow, vol. 38, no. 1, pp. 99–117, 2012.
- [7] S. M. Allen and J. W. Cahn, "A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening," Acta Metallurgica, vol. 27, no. 6, pp. 1085–1095, 1979.
- [8] S. Mirjalili, C. B. Ivey, and A. Mani, "A conservative diffuse interface method for two-phase flows with provable boundedness properties," Journal of Computational Physics, vol. 401, p. 109006, 2020.

- [9] A. Dadvand, M. Bagheri, N. Samkhaniani, H. Marschall, and M. W¨orner, "Advected phase-field method for bounded solution of the Cahn-Hilliard Navier-Stokes equations," Physics of Fluids, vol. 33, no. 5, p. 053311, 2021.
- [10] G. Tryggvason, R. Scardovelli, and S. Zaleski, Direct numerical simulations of gas–liquid multiphase flows. Cambridge university press, 2011.
- [11] J. H. Ferziger, M. Peri´c, and R. L. Street, Computational methods for fluid dynamics, vol. 3. Springer, 2002.
- [12] A. K. Gunstensen, D. H. Rothman, S. Zaleski, and G. Zanetti, "Lattice Boltzmann model of immiscible fluids," Physical Review A, vol. 43, no. 8, p. 4320, 1991.
- [13] X. He, S. Chen, and R. Zhang, "A lattice Boltzmann scheme for incompressible multiphase flow and its application in simulation of Rayleigh-Taylor instability," Journal of Computational Physics, vol. 152, no. 2, pp. 642–663, 1999.
- [14] T. Inamuro, T. Ogata, S. Tajima, and N. Konishi, "A lattice Boltzmann method for incompressible two-phase flows with large density differences," Journal of Computational Physics, vol. 198, no. 2, pp. 628–644, 2004.
- [15] Y. Wang, C. Shu, H. Huang, and C. J. Teo, "Multiphase lattice Boltzmann flux solver for incompressible multiphase flows with large density ratio," Journal of Computational Physics, vol. 280, pp. 404–423, 2015.
- [16] A. J. Chorin, "A numerical method for solving incompressible viscous flow problems," Journal of Computational Physics, vol. 135, no. 2, pp. 118–125, 1997.
- [17] W. Soh and J. W. Goodrich, "Unsteady solution of incompressible Navier-Stokes equations," Journal of Computational Physics, vol. 79, no. 1, pp. 113–134, 1988.
- [18] A. Malan, R. Lewis, and P. Nithiarasu, "An improved unsteady, unstructured, artificial compressibility, finite volume scheme for viscous incompressible flows: Part I. theory and implementation," International Journal for Numerical Methods in Engineering, vol. 54, no. 5, pp. 695–714, 2002.
- [19] F. Kelecy and R. Pletcher, "The development of a free surface capturing approach for multidimensional free surface flows in closed containers," Journal of Computational Physics, vol. 138, no. 2, pp. 939–980, 1997.
- [20] A. Shah and L. Yuan, "Numerical solution of a phase field model for incompressible two-phase flows based on artificial compressibility," Computers & Fluids, vol. 42, no. 1, pp. 54–61, 2011.
- [21] S. Ansumali, I. V. Karlin, and H. S. Ottinger, "Thermodynamic theory of incompressible hydrodynam- ¨ ics," Physical Review Letters, vol. 94, no. 8, p. 080602, 2005.
- [22] I. V. Karlin, A. G. Tomboulides, C. E. Frouzakis, and S. Ansumali, "Kinetically reduced local Navier-Stokes equations: An alternative approach to hydrodynamics," Physical Review E, vol. 74, no. 3, p. 035702, 2006.
- [23] T. Hashimoto, I. Tanno, Y. Tanaka, K. Morinishi, and N. Satofuka, "Simulation of doubly periodic shear layers using kinetically reduced local Navier-Stokes equations on a GPU," Computers & Fluids, vol. 88, pp. 715–718, 2013.
- [24] T. Hashimoto, I. Tanno, T. Yasuda, Y. Tanaka, K. Morinishi, and N. Satofuka, "Higher order numerical simulation of unsteady viscous incompressible flows using kinetically reduced local Navier-Stokes equations on a GPU," Computers & Fluids, vol. 110, pp. 108–113, 2015.
- [25] T. Hashimoto, T. Yasuda, I. Tanno, Y. Tanaka, K. Morinishi, and N. Satofuka, "Multi-GPU parallel computation of unsteady incompressible flows using kinetically reduced local Navier-Stokes equations," Computers & Fluids, vol. 167, pp. 215–220, 2018.
- [26] J. R. Clausen, "Entropically damped form of artificial compressibility for explicit simulation of incompressible flow," Physical Review E, vol. 87, no. 1, p. 013309, 2013.
- [27] Y. T. Delorme, K. Puri, J. Nordstrom, V. Linders, S. Dong, and S. H. Frankel, "A simple and efficient incompressible Navier-Stokes solver for unsteady complex geometry flows on truncated domains," Computers & Fluids, vol. 150, pp. 84–94, 2017.
- [28] A. Kajzer and J. Pozorski, "Application of the entropically damped artificial compressibility model to direct numerical simulation of turbulent channel flow," Computers & Mathematics with Applications,

- vol. 76, no. 5, pp. 997–1013, 2018.
- [29] A. Kajzer and J. Pozorski, "A weakly compressible, diffuse-interface model for two-phase flows," Flow, Turbulence and Combustion, vol. 105, pp. 299–333, 2020.
- [30] A. Kajzer and J. Pozorski, "A weakly compressible, diffuse interface model of two-phase flows: Numerical development and validation," Computers & Mathematics with Applications, vol. 106, pp. 74–91, 2022.
- [31] A. Toutant, "General and exact pressure evolution equation," Physics Letters A, vol. 381, no. 44, pp. 3739–3742, 2017.
- [32] A. Toutant, "Numerical simulations of unsteady viscous incompressible flows using general pressure equation," Journal of Computational Physics, vol. 374, pp. 822–842, 2018.
- [33] D. Dupuy, A. Toutant, and F. Bataille, "Analysis of artificial pressure equations in numerical simulations of a turbulent channel flow," Journal of Computational Physics, vol. 411, p. 109407, 2020.
- [34] X. Shi and C.-A. Lin, "Simulations of wall bounded turbulent flows using general pressure equation," Flow, Turbulence and Combustion, vol. 105, no. 1, pp. 67–82, 2020.
- [35] J.-J. Huang, "Numerical simulation of two-phase incompressible viscous flows using general pressure equation," arXiv preprint arXiv:2011.00814, 2020.
- [36] K. Yang and T. Aoki, "Weakly compressible Navier-Stokes solver based on evolving pressure projection method for two-phase flow simulations," Journal of Computational Physics, vol. 431, p. 110113, 2021.
- [37] O. Ubbink and R. Issa, "A method for capturing sharp fluid interfaces on arbitrary meshes," Journal of Computational Physics, vol. 153, no. 1, pp. 26–50, 1999.
- [38] A. Arote, M. Bade, and J. Banerjee, "An improved compressive volume of fluid scheme for capturing sharp interfaces using hybridization," Numerical Heat Transfer, Part B: Fundamentals, vol. 79, no. 1, pp. 29–53, 2020.
- [39] C. Anghan, M. H. Bade, and J. Banerjee, "A modified switching technique for advection and capturing of surfaces," Applied Mathematical Modelling, vol. 92, pp. 349–379, 2021.
- [40] F. Xiao, Y. Honma, and T. Kono, "A simple algebraic interface capturing scheme using hyperbolic tangent function," International Journal for Numerical Methods in Fluids, vol. 48, no. 9, pp. 1023– 1040, 2005.
- [41] S. Saincher and V. Sriram, "An efficient operator-split CICSAM scheme for three-dimensional multiphase-flow problems on Cartesian grids," Computers & Fluids, vol. 240, p. 105440, 2022.
- [42] G. D. Weymouth and D. K.-P. Yue, "Conservative volume-of-fluid method for free-surface simulations on Cartesian-grids," Journal of Computational Physics, vol. 229, no. 8, pp. 2853–2865, 2010.
- [43] B. Parker and D. Youngs, Two and three dimensional Eulerian simulation of fluid flow with material interfaces. Atomic Weapons Establishment, 1992.
- [44] M. M. Francois, S. J. Cummins, E. D. Dendy, D. B. Kothe, J. M. Sicilian, and M. W. Williams, "A balanced-force algorithm for continuous and sharp interfacial surface tension models within a volume tracking framework," Journal of Computational Physics, vol. 213, no. 1, pp. 141–173, 2006.
- [45] S. J. Cummins, M. M. Francois, and D. B. Kothe, "Estimating curvature from volume fractions," Computers & Structures, vol. 83, no. 6-7, pp. 425–434, 2005.
- [46] S. Gottlieb and C.-W. Shu, "Total variation diminishing Runge-Kutta schemes," Mathematics of computation, vol. 67, no. 221, pp. 73–85, 1998.
- [47] C.-W. Shu and S. Osher, "Efficient implementation of essentially non-oscillatory shock-capturing schemes," Journal of computational physics, vol. 77, no. 2, pp. 439–471, 1988.
- [48] S. Gottlieb, D. I. Ketcheson, and C.-W. Shu, "High order strong stability preserving time discretizations," Journal of Scientific Computing, vol. 38, no. 3, pp. 251–289, 2009.
- [49] R. Caiden, R. P. Fedkiw, and C. Anderson, "A numerical method for two-phase flow consisting of separate compressible and incompressible regions," Journal of Computational Physics, vol. 166, no. 1, pp. 1–27, 2001.
- [50] E. Bassano, "Numerical simulation of thermo-solutal-capillary migration of a dissolving drop in a cavity," International journal for numerical methods in fluids, vol. 41, no. 7, pp. 765–788, 2003.
- [51] S. Parameswaran and J. Mandal, "A stable interface-preserving reinitialization equation for conservative

- level set method," European Journal of Mechanics-B/Fluids, vol. 98, pp. 40–63, 2023.
- [52] P. Liovic, M. Rudman, J.-L. Liow, D. Lakehal, and D. Kothe, "A 3D unsplit-advection volume tracking algorithm with planarity-preserving interface reconstruction," Computers & Fluids, vol. 35, no. 10, pp. 1011–1032, 2006.
- [53] S. Popinet, "An accurate adaptive solver for surface-tension-driven interfacial flows," Journal of Computational Physics, vol. 228, pp. 5838–5866, Sept. 2009.
- [54] A. Kajzer and J. Pozorski, "Diffuse interface models for two-phase flows in artificial compressibility approach," Journal of Physics: Conference Series, vol. 1101, p. 012013, Oct. 2018.
- [55] F. Garoosi and T.-F. Mahdi, "Numerical simulation of three-fluid Rayleigh-Taylor instability using an enhanced volume-of-fluid (VOF) model: New benchmark solutions," Computers & Fluids, vol. 245, p. 105591, 2022.
- [56] J. C. Martin, W. J. Moyce, J. Martin, W. Moyce, W. G. Penney, A. Price, and C. Thornhill, "Part IV. An experimental study of the collapse of liquid columns on a rigid horizontal plane," Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences, vol. 244, no. 882, pp. 312–324, 1952.
- [57] K. Ling, S. Zhang, P.-Z. Wu, S.-Y. Yang, and W.-Q. Tao, "A coupled volume-of-fluid and level-set method (VOSET) for capturing interface of two-phase flows in arbitrary polygon grid," International Journal of Heat and Mass Transfer, vol. 143, p. 118565, 2019.
- [58] B. Van Leer, "Towards the ultimate conservative difference scheme. II. monotonicity and conservation combined in a second-order scheme," Journal of Computational Physics, vol. 14, no. 4, pp. 361–370, 1974.
- [59] S. Hysing, S. Turek, D. Kuzmin, N. Parolini, E. Burman, S. Ganesan, and L. Tobiska, "Quantitative benchmark computations of two-dimensional bubble dynamics," International Journal for Numerical Methods in Fluids, vol. 60, no. 11, pp. 1259–1288, 2009.
- [60] U. Ayachit, The paraview guide: a parallel visualization application. Kitware, Inc., 2015.
- [61] L. Strubelj, I. Tiselj, and B. Mavko, "Simulations of free surface flows with implementation of surface ˇ tension and interface sharpening in the two-fluid model," International Journal of Heat and Fluid Flow, vol. 30, no. 4, pp. 741–750, 2009.
- [62] S. Saito, Y. Abe, and K. Koyama, "Lattice Boltzmann modeling and simulation of liquid jet breakup," Physical Review E, vol. 96, no. 1, p. 013317, 2017.
- [63] H. G. Lee and J. Kim, "Numerical simulation of the three-dimensional Rayleigh-Taylor instability," Computers & Mathematics with Applications, vol. 66, no. 8, pp. 1466–1474, 2013.
- [64] D. Pan, "A high-order finite volume method solving viscous incompressible flows using general pressure equation," Numerical Heat Transfer, Part B: Fundamentals, vol. 82, pp. 146–163, Nov. 2022.