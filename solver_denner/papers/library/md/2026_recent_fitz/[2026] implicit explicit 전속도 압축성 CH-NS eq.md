manuscript No. (will be inserted by the editor)


## Implicit-explicit all-speed schemes for compressible Cahn-Hilliard-Navier-Stokes equations

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

the date of receipt and acceptance should be inserted later

Abstract We propose a second-order implicit-explicit (IMEX) time-stepping scheme for the isentropic, compressible Cahn-Hilliard-Navier-Stokes equations in the low Mach number regime. The method is based on finite differences on staggered grids and is specifically designed to handle the challenges posed by the low Mach number limit, where the system approaches to an incompressible behavior. In this regime, standard explicit schemes suffer from severe time-step restrictions due to fourth-order diffusion terms and the stiffness induced by fast acoustic waves. To overcome this, we employ an IMEX strategy which splits the governing equations into stiff and non-stiff components. The stiff terms, arising from pressure, viscous forces and fourth-order Cahn-Hilliard contributions, are treated implicitly, while the remaining are dealt explicitly.

Keywords Asymptotic preserving, low Mach number, implicit-explicit schemes, incompressible limit, isentropic compressible Cahn-Hilliard-Navier-Stokes

1 Introduction

In fluid dynamics, the Cahn-Hilliard (CH) equation describes the phenomenon of phase separation in two-phase systems [9]. It captures the temporal evolution of a mixture of immiscible fluids through the formation of diffuse interfaces between the phases. This diffusive-interface approach is well-suited for explaining topological changes, such as layered structures in sedimenting colloidal suspensions [35]. However, the classical CH model does not take into account the fluid dynamics such as motion of the medium, viscosities, or external forces like gravity. To incorporate these effects, the CH system is combined with the Navier-Stokes equations, which describe the conservation of mass and momentum balance of fluids. The resulting system of partial differential equations, known as the Cahn-HilliardNavier-Stokes (CHNS), provides a thermodynamically consistent framework for modeling multiphase flows [1,8,26]. The CHNS model has been applied to a wide variety of problems: separation of immiscible fluids, bubble dynamics or the evolution of interfaces in multiphase systems [22,35].

Department of Mathematics, Universitat de Val`encia (Spain); email: mulet@uv.es.


# arXiv:2602.20679v1  [math.NA]  24 Feb 2026

2 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

In many practical applications, the flow occurs in a low Mach number regime. The Mach number, defined as the ratio between the characteristic fluid velocity and the speed of sound, becomes small in these situations. Under these conditions, the flow behaves as a nearly incompressible [2,3,24], but compressible effects are still present. Designing efficient numerical schemes for the compressible CHNS equations for all Mach number regimes presents several challenges. First, as the squared low Mach number δ tends to zero, the system becomes increasingly stiff due to the presence of fast acoustic waves with characteristic speeds, namely, v ± 1

δ c, where v is the velocity filed and c represents the speed of sound. This stiffness is further incremented by the presence of up to fourth-order spatial derivatives in the CHNS equations leading to discrete operators with large eigenvalues. As a consequence, explicit solvers are severely constrained by stability conditions,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq001.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq002.png)

where ∆t and ∆x are the time and spatial step size, respectively. In addition, when solving ODE systems of the form z′ = f(z), the appearance of stiff, negativedefinite Jacobians allows implicit methods to take larger time steps, whereas explicit methods are limited to ∆t ≈O(|λ|−1), with |λ| the largest absolute magnitude of the Jacobian. Another challenge arises from the intrinsic structure of the CH equation, whose gradient-flow nature is associated with a non-convex energy functional [26]. To ensure stability, a common strategy is to decompose the energy into the difference of two differentiable convex functions. By treating the contractive part implicitly and the expansive part explicitly, one can construct unconditionally stable IMEX scheme, as shown in [16,36]. One way of dealing with these difficulties is to split the pressure into stiff and non-stiff components and treating the stiff part implicitly (for e.g. [11,12]). Therefore, the time step is no longer constrained by the Mach number. Similarly, the stiff fourth-order CH terms are treated implicitly, while the remaining are handled explicitly [30,32]. To maintain stability and preserve symmetry, one can discretize the variables on staggered grids [20,29]. As shown in [2], when the Mach number tends to zero, the solutions of the compressible CHNS equations converge to those of its incompressible counterpart. Building on this, numerical methods for the compressible CHNS equations are proposed in [19,30,32], while for the quasi-incompressible and incompressible CHNS models are presented in [10,18,21,25]. Numerical schemes for the low Mach number of the compressible Euler equations are discussed in [11,12,17,33]. The goal of this work is to develop a second-order IMEX Runge-Kutta scheme for the isentropic, compressible CHNS equations in low Machfor all Mach number regimes. The method, based on finite differences on staggered grids, is designed to be an Asymptotic Preserving (AP) scheme [12], meaning that its stability and accuracy are independent of the Mach number and that it correctly captures the incompressible limit as δ →0. The outline of the current work is organized as follows. In Section 2 we introduce the isentropic, compressible CHNS equations in a low Mach number regime. Section 3 presents a partitioned IMEX Runge-Kutta scheme for the two-

Title Suppressed Due to Excessive Length 3

dimensional case, and in Section 4 its AP property is proven. In Section 5, numerical experiments are performed in order to verify the stability, accuracy, and efficiency of our proposed scheme. Finally, Section 6 summarizes the main conclusions and discusses the future directions of our research.

2 Cahn-Hilliard-Navier-Stokes Equations

2.1 Model Description

We follow the model from [1,2] describing the dynamics of two immiscible, compressible, viscous Newtonian fluids in a bounded, open domain Ω⊂R3. Let ρi, ci, vi denote, respectively, the density, mass concentration, velocity vi of the fluid i = 1, 2. The mixture density is ρ and the barycentric velocity, v, is ρv = ρ1v1 + ρ2v2. The concentration difference c = c1 −c2, taking values in [−1, 1], serves as an order parameter distinguishing the two fluid components. The total Helmholtz free energy of the system is defined as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq003.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq004.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq005.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq006.png)

where the positive parameter ε is related to the diffuse interface thickness, which controls the width of the transition region between the two phases. The term ε 2|∇c|2 represents the classical Cahn-Hilliard regularizing term [26]. The specific Helmholtz free energy is assumed to have the form


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq007.png)

where fe is the potential energy and


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq008.png)

4


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq009.png)

is a double-well potential. The thermodynamic pressure is related to the potential energy through the relation p(ρ) = ρ2 ∂f(ρ,c)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq010.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq011.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq012.png)

for a positive constant Cp and γ > 1 is the adiabatic constant. The evolution of the mixture is governed by the isentropic, compressible CahnHilliard-Navier-Stokes with gravitational force,    


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq013.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq014.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq015.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq016.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq017.png)

where the operators div and ∆represents the divergence and laplacian operators, respectively. The first equation expresses the conservation of total mass of the mixture, the second equation represents the balance of momentum taking into account the gravitation acceleration g, and the third is a Cahn–Hilliard-type equation modeling the evolution of the concentration difference c. The viscous and capillary

4 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

effects [15] are incorporated into the model through the stress tensor T = T1 + T2 with

T1(v) = ν(∇v + ∇vT ) + λ div vI, T2(c) = ε


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq018.png)

where λ and ν are the viscosity coefficients, which are assumed to be positive. The chemical potential is defined as the variational derivative of the total free energy with respect to the order parameter c. In this setting, it is given by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq019.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq020.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq021.png)

The system (2) is supplemented with initial conditions for the density, velocity, and concentration, given by ρ(0, x) = ρ0(x), v(0, x) = v0(x), and c(0, x) = c0(x). In addition, we impose the boundary conditions


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq022.png)

where n denotes the outward unit normal vector to the boundary ∂Ω. In [1, Theorem 1.2], it was proven that, for γ > 3

2 and suitable initial data (ρ0, v0, c0), system (2) with boundary conditions (3) admits global-in-time weak solutions in the sense of Di Perna and Lions.

2.2 Isentropic compressible Cahn-Hilliard-Navier-Stokes in a low Mach number regime

In the present work, we focus on the low Mach number regime, which corresponds to taking Cp ≫0. In such regimes, the pressure becomes extremely large, and in many numerical methods (e.g., [30,32]), Cp appears explicitly in the time step stability restriction,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq023.png)

where ∆t and ∆x are the time and spacial step sizes, respectively, leading to severe restrictions on the time step. To this end, we split the pressure into a stiff and non-stiff component, treating the stiff part implicitly [11,12]. Specifically, we write


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq024.png)

We denote p1(ρ) = Cp,1ργ and p2(ρ) = Cp,2ργ with the squared Mach number defined as δ = C−1 p . In general, the choice of the pressure splitting depends on the characteristic fluid speed. For working in a hyperbolic framework and suppressing nonphysical oscillations, we consider Cp,1 > 0. However, for flows with stronger shocks, a larger value of Cp,1 is necessary, corresponding to an almost fully explicit treatment of the pressure [12]. In the current work, the effects of this choice are investigated numerically in the section of numerical experiments. We consider both the oneand two-dimensional setting. For the latter case, we denote the velocity field by v = (v1, v2) and the gravity g = (0, g) acting only on

Title Suppressed Due to Excessive Length 5

the vertical axes. Under this assumption, the governing equations reduce to the following two-dimensional form:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq025.png)

(ρv1)t + (ρv2 1 + p1(ρ))x + (ρv1v2)y = −(p2(ρ))x + ε


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq026.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq027.png)

(ρv2)t + (ρv2 2 + p1(ρ))y + (ρv1v2)x = ρg −(p2(ρ))y + ε


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq028.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq029.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq030.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq031.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq032.png)

For the one-dimensional case the system reads as follows:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq033.png)

(ρv)t + (ρv2 + p1(ρ))x = −(p2(ρ))x + ρg + � (2ν + λ) vx −ε


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq034.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq035.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq036.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq037.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq038.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq039.png)

3 Numerical Schemes

3.1 Spatial Semidiscretization

We consider the compressible, isentropic CHNS equations in two-spatial dimensions (5) on the square domain Ω= [0, 1]2. The computational grid is based on a MAC approach [20]. Let x = (x, y) denote the spatial variable. The cell-centered grid consists of M2 nodes


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq040.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq041.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq042.png)

for i, j = 1, · · · , M, and with uniform mesh size h = 1 M . The staggered (dual) grid consists in 2M(M −1) nodes:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq043.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq044.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq045.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq046.png)

where xi+ 1

2 = ih and yj+ 1

2 = jh. The continuity and the Cahn-Hilliard type equations are treated at cell centers, while the momentum equations are treated at the dual points: the horizontal component on vertical cell faces and the vertical component on horizontal cell faces. To compute momentum at staggered points,

6 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

we used local averages of the density. For instance, for the horizontal momentum we define (ρ∗,x)i,j = ρ∗,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq047.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq048.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq049.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq050.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq051.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq052.png)

for i = 1, · · · , M −1, j = 1, · · · , M. Notice that v1,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq053.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq054.png)

2 ,j. We denote by ρ∗v = (ρ∗,xv1, ρ∗,yv2). For the velocity components, we assume no-slip boundary conditions, i.e.


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq055.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq056.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq057.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq058.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq059.png)

For the points outside the wall, we assume symmetric reflection for the density and odd reflection for the velocity components. For instance,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq060.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq061.png)

for j = 1, . . . , M. Similarly, it is done in the other direction. After applying this spatial semi-discretization to (5), one needs to solve a system of N = 2M2 + 2M(M −1) ordinary differential equations given by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq062.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq063.png)

where U0 is the vector of initial conditions and the unknown variables U = (uk)4 k=1 are

ρi,j = u1,i,j, (ρv1)i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq064.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq065.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq066.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq067.png)

for i, j running over their respective grid indices. Dropping the time dependence of U, L(U) is the nonlinear operator storing the spatially discretized differential operators,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq068.png)

where nonzero terms of operators above are defined as follows:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq069.png)

C2(U)2,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq070.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq071.png)

C2(U)3,i,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq072.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq073.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq074.png)

L1(U)2,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq075.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq076.png)

L1(U)3,i,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq077.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq078.png)

L2(U)2,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq079.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq080.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq081.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq082.png)

L2(U)3,i,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq083.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq084.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq085.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq086.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq087.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq088.png)

L4(U)2,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq089.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq090.png)

L4(U)3,i,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq091.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq092.png)

Title Suppressed Due to Excessive Length 7

3.1.1 Basic Finite Difference Operators

In this section, we introduce the finite difference operators used to approximate the spatial derivatives of the system (5) on MAC grids. To approximate the first derivatives at the grid points xi,j, we employ central differences which are second-order accurate at interior points (1 < i, j < M), and first-order otherwise satisfying the boundary conditions (3). The resulting discrete derivative operator in one spatial direction can be written in matrix form as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq093.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq094.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq095.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq096.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq097.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq098.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq099.png)

For the dual grid, we define two finite difference matrices DM and D∗ M of size M ×(M −1) that approximate first derivatives at the cell interfaces xi+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq100.png)

2 which are second-order accurate at interior points and first-order otherwise. Both matrices incorporate the appropriate boundary conditions (3):

DM = 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq101.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq102.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq103.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq104.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq105.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq106.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq107.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq108.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq109.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq110.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq111.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq112.png)

We also define the averaging matrix AM ∈R(M−1)×M, which is used to interpolate quantities between the cell centers and the staggered grid:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq113.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq114.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq115.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq116.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq117.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq118.png)

We introduce f ∗g = (fi,jgi,j)i,j for matrices f, g in Rn×m.

3.1.2 The Operators C1 and C2

The convective part of the system is decomposed into two operators,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq119.png)

where C1 acts only on the continuity equation and C2 on the momentum and CahnHilliard type equations.

We consider the fluxes in the xand y-directions:

F(U) =


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq120.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq121.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq122.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq123.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq124.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq125.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq126.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq127.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq128.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq129.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq130.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq131.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq132.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq133.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq134.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq135.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq136.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq137.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq138.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq139.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq140.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq141.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq142.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq143.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq144.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq145.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq146.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq147.png)

8 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

Let ˆF ∗and ˆG∗denote the numerical fluxes associated to F ∗and G∗, respectively. The convective operators are approximated using numerical flux differences at cell centers and cell interfaces. Specifically,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq148.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq149.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq150.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq151.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq152.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq153.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq154.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq155.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq156.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq157.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq158.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq159.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq160.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq161.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq162.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq163.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq164.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq165.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq166.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq167.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq168.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq169.png)

The numerical fluxes are computed using the Rusanov flux. For the explicit terms of the operator C (see Section 3.3), we use WENO5 reconstructions, which are fifth-order accurate for finite difference schemes [5,6,27]. Let us describe it for the x-direction case. Denote by Wx : R5 −→R the WENO5 reconstruction operator and a function f such that fi,j = f(xi,j) for indexes i, j running both in the primal and dual grids. So the right and left state reconstructions are, respectively, for primal grids,

f+ i+1,j = Wx (fi+3,j, . . . , fi−1,j) , f− i,j = Wx (fi−2,j, . . . , fi+2,j) ,

and for dual grids,

f+ i+ 1

2 ,j = Wx � fi+ 5


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq170.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq171.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq172.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq173.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq174.png)

f+ i+1,j+ 1

2 = Wx � fi+3,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq175.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq176.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq177.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq178.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq179.png)

Then, for the terms in C1,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq180.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq181.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq182.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq183.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq184.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq185.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq186.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq187.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq188.png)

and for the terms in C2,

ˆF ρv1 i,j = 1

2

� (ρv2 1 + p1(ρ))+ i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq189.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq190.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq191.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq192.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq193.png)

ˆF ρv2 i+ 1

2 ,j+ 1

2 = 1

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq194.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq195.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq196.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq197.png)

2 2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq198.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq199.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq200.png)

ˆF c i+ 1

2 ,j = 1

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq201.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq202.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq203.png)

Title Suppressed Due to Excessive Length 9

The numerical viscosities λ∗are defined as the maximum of the upper bounds of the local characteristic speeds at the reconstructed states of each ˆF ∗, namely,

λρ i+ 1

2 ,j = λc i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq204.png)

λρv1 i,j = max ����v+ 1,i+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq205.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq206.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq207.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq208.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq209.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq210.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq211.png)

λρv2 i+ 1

2 ,j+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq212.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq213.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq214.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq215.png)

2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq216.png)

where s(f∗ i,j) = �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq217.png)

dual grids. Notice also that the values of ρ on the dual grid and of v1 on the primal grid are required. To this end, we employ a sixth-order grid transfer operator defined by the coefficients


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq218.png)

Thus,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq219.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq220.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq221.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq222.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq223.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq224.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq225.png)

and similarly in the y-direction. In particular, to evaluate ˆF ρv2, we must approximate the velocity component v1 at staggered locations where it is not directly defined. So, we first approximate


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq226.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq227.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq228.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq229.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq230.png)

and then apply the previous transfer grid operator in the x-direction to obtain v1,i,j+ 1

2 .

An analogous procedure is used for the flux ˆGρv1.

3.1.3 The Operator L1

The nonzero components of the operator L1 are approximated point-wise and taking central finite differences, specifically,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq231.png)

3.1.4 The Operator L2

The operator L2 involves the derivatives of the order parameter c in the momentum equation. For the approximation of L2(U)2 we use:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq232.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq233.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq234.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq235.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq236.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq237.png)

10 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

for i = 1, · · · , M −1 and j = 1, · · · , M. Similarly, for the L2(U)3 we use:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq238.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq239.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq240.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq241.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq242.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq243.png)

for i = 1, · · · , M and j = 1, · · · , M −1. These approximations satisfy the boundary conditions (3) and are second-order accurate at interior points and first-order accurate otherwise.

3.1.5 The Operator L3

The operator L3, which arises from the Cahn-Hilliard type equation, requires a special treatment. This is because, for stability, only negative definite terms should be treated implicitly. However, the term ∆ψ′(c) = div � ψ′′(c)∇c�changes sign in (−1, 1) since the potential ψ is of convex-concave type. To handle this, in [16] was shown that if ψ is split into the sum of a convex part ψ1 and a concave part ψ2, the resulting scheme for the Cahn-Hilliard equation treating ψ′ 1 implicitly and ψ′ 2 explicitly is unconditionally stable. In particular, we choose


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq244.png)

Let f ∈C4 such that ∇f(x, y) · n = 0 with fi,j = f(xi,j) for i, j = 1, · · · , M. We use a second-order accurate approximation for ∆fi,j ≈∆hfi,j = ∆x,hfi,j +∆y,hfi,j where


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq245.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq246.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq247.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq248.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq249.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq250.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq251.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq252.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq253.png)

and similarly for ∆y,h. For i, j = 1, · · · , M, yields


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq254.png)

and (∆ψ′ 2(c))(xi,j) ≈(ψ′′ 2(c)cx)x(xi,j) + (ψ′′ 2(c)cy)y(xi,j) where

(ψ′′ 2(c)cx)x(xi,j) ≈


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq255.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq256.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq257.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq258.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq259.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq260.png)

2h2 i = M, (8) and

(ψ′′ 2(c)cy)y(xi,j) ≈


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq261.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq262.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq263.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq264.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq265.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq266.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq267.png)

Title Suppressed Due to Excessive Length 11

Now, it only remains to approximate ∆ � 1 ρ∆c � . To this end, we employ the aforementioned second-order accurate approximation for the laplacian, namely,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq268.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq269.png)

where D is the diagonal operator on M × M matrices defined as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq270.png)

3.1.6 The Operator L4

The operator L4 stores the derivatives of the velocity field in the balance of momentum. For approximating the pure double derivatives, for instance, (v1)xx and (v1)yy at xi+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq271.png)

(v1)xx � xi+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq272.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq273.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq274.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq275.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq276.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq277.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq278.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq279.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq280.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq281.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq282.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq283.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq284.png)

for j = 1, · · · , M, and

(v1)yy � xi+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq285.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq286.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq287.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq288.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq289.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq290.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq291.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq292.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq293.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq294.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq295.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq296.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq297.png)

for i = 1, · · · , M −1. The approximation of the cross derivative, e.g., (v2)xy at xi+ 1

2 ,j is given by

(v2)xy � xi+ 1


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq298.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq299.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq300.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq301.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq302.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq303.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq304.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq305.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq306.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq307.png)

2 )


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq308.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq309.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq310.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq311.png)

for i = 1, · · · , M −1. The three expressions above verify the boundary conditions (3) and are secondorder accurate at its respective interior points and first-order accurate otherwise. In matrix form,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq312.png)

Similarly, the other nonzero component of L4 takes the form


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq313.png)

12 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

3.2 Vector Implementation

In this section, we reformulate system (7) in vector form for the two-dimensional case. The one-dimensional case follows analogously and is therefore omitted.

Let vec(A) denote the column-wise vectorization of a matrix A ∈Rn×m, defined by vec(A)i+m(j−1) = Ai,j, for 1 ≤i ≤n, 1 ≤j ≤m.

For simplicity, we will use the same symbols ϱ, V1, V2, C, Ck(U)i and Lj(U)i (for i, j = 1, · · · , 4 and k = 1, 2) to denote both the original matrices and their vectorizations, whenever there is no risk of confusion. Let ⊗denotes the Kronecker product and In the identity matrix of size n. With this notation, the nonzero blocks of C1 can be expressed in vector form as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq314.png)

with


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq315.png)

where Λx and Λy denote the diagonal matrices of the maximum characteristic speeds associated with the fluxes F ρ and Gρ in the xand y-directions, respectively, evaluated at the reconstructed states. Similarly, the nonzero blocks of the operators L1, L3, L4 can be written as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq316.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq317.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq318.png)

where A3 is the tensor constructed form the values of ψ′′ 2 in (8)-(9) and ∆h the Laplacian operator in tensor form. The nonzero blocks of L4 are the following:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq319.png)

where the matrices Ai,j are given by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq320.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq321.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq322.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq323.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq324.png)

Using this notation, system (7) can be expressed compactly in vector form by defining


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq325.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq326.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq327.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq328.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq329.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq330.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq331.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq332.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq333.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq334.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq335.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq336.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq337.png)

Title Suppressed Due to Excessive Length 13

3.3 Implicit-Explicit Schemes

To construct an implicit-explicit scheme, we employ the technique of doubling variables combined with a partitioned Runge-Kutta approach [7,28,30,32]. Consider a sufficiently smooth function


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq338.png)

defined as

˜L( ˜U, U) = C1( ˜U, U) + C2( ˜U) + ˜L1( ˜U, U) + L2( ˜U) + ˜L3( ˜U, U) + L4(U),

where the only nonzero component of the operators ˜L1 and ˜L3 are given by

˜C1( ˜U, U)1 = (IM ⊗DM)(ϱ∗,x ∗V1) + (DM ⊗IM)(ϱ∗,y ∗V2) + hA1( ˜U)˜ϱ,

˜L1( ˜U, U)2 = (IM ⊗DT M)p2(ϱ),

˜L1( ˜U, U)3 = (AM ⊗IM)g˜ϱ + (DT M ⊗IM)p2(ϱ), ˜L3( ˜U, U)4 = 2∆hC + A3( ˜C) ˜C −∆h(D(ϱ−1)∆hC).

Using this operator, the full discrete scheme given by (7) can be written as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq339.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq340.png)

which is equivalent to


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq341.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq342.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq343.png)

Here, all terms involving ˜U are treated explicitly, while those depending on U are handled implicitly. System (14) allows us to apply separate Runge-Kutta schemes to the explicit and the implicit parts. Therefore, we consider a pair of Butcher tableaus with s stages:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq344.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq345.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq346.png)

The first tableau defines the explicit part of the scheme with ˜αi,j = 0 for all j ≥i, while the second tableau represents the diagonally implicit part, where αi,j = 0 for j > i and αi,i ̸= 0. The γi and ˜γi coefficients are defined by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq347.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq348.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq349.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq350.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq351.png)

14 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

Using these tableaus, the stage values of the partitioned Runge-Kutta method applied to (14) are computed as follows:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq352.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq353.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq354.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq355.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq356.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq357.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq358.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq359.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq360.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq361.png)

If β = ˜β and Un = ˜Un, then both solutions remain identical at every time step, which eliminates the need of doubling the number of variables. In addition, as proven in [28], if both Butcher tableaus are second-order accurate, the resulting partitioned Runge-Kutta method is also second-order accurate. Consequently, the final scheme is given by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq362.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq363.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq364.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq365.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq366.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq367.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq368.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq369.png)

Henceforth, we restrict our analysis to Stiffly Accurate Runge-Kutta schemes, that is, those satisfying αs,j = βj for j = 1, . . . , s.

3.4 Solution to the Nonlinear Systems

At each intermediate stage i = 1, · · · , s, the scheme (15) reduces to solving the following nonlinear system for U(i):


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq370.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq371.png)

which is composed by two subsystems: a nonlinear system for the density and the velocities, and then a linear system for the c variable, following the approach described in [30,32].

Title Suppressed Due to Excessive Length 15

For the nonlinear subsystem one has to solve M2 + 2M(M −1) equations for equal number of unknowns, corresponding to ϱ, V1 and V2. The system reads as:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq372.png)

(ϱ∗,x ∗V1)(i) − � (ϱ∗,x ∗V1)(i) + ∆tαi,i � A1,1V (i) 1 + A1,2V (i) 2 −(IM ⊗DT M)p2(ϱ(i)) � = 0,

(ϱ∗,y ∗V2)(i) − � (ϱ∗,y ∗V2)(i) + ∆tαi,i � A2,1V (i) 1 + A2,2V (i) 2 −(DT M ⊗IM)p2(ϱ(i)) � = 0,

(17) where the terms marked with a hat �· are explicitly computed at the current stage i = 1, · · · , s. Once ϱ(i), V (i) 1 and V (i) 2 have been computed, the remaining step to solve in (16) is the linear system for C(i). The system takes the form,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq373.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq374.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq375.png)

which is equivalent to solving the following linear system for C(i)

� D(ϱ(i)) −2∆tαi,i∆h + ∆tαi,iε∆hD(ϱ(i))−1∆h � C(i) = � (ϱ ∗C)(i). (18)

Due to the convex splitting stated in Section 3.1.5, the coefficient matrix is symmetric and positive definite, provided that ϱ(i) k > 0 for all k = 1, · · · , M2.

3.5 Nonlinear Solvers

For the nonlinear subsystem (17) the damped Newton’s method is employed [4]. Dropping the superscript of the i-stage, the nonlinear system (17) expressed in compact form is H(z) ≡L(z) + ∆tαi,iD(z) −r = 0,

where

z =


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq376.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq377.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq378.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq379.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq380.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq381.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq382.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq383.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq384.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq385.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq386.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq387.png)

and the nonlinear operator is


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq388.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq389.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq390.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq391.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq392.png)

Then at each Newton iteration the solution is updated as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq393.png)

where the step δn is computed by solving the linear system


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq394.png)

16 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

Here, H′(zn) denotes the Jacobian matrix of H evaluated at the current iterate zn, and the damping parameter αn ∈(0, 1] is chosen to ensure that ||H(zn+1)||2 is decreasing. The Jacobian matrix has the form

H′(z) = L′(z) + ∆tαi,iD′(z) = � IM 2 0M 2 0M 2


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq395.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq396.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq397.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq398.png)

where 0M 2 denotes the M2 zero matrix,

DV =


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq399.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq400.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq401.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq402.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq403.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq404.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq405.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq406.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq407.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq408.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq409.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq410.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq411.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq412.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq413.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq414.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq415.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq416.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq417.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq418.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq419.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq420.png)

and


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq421.png)

For analyzing the invertibility of the Jacobian matrix (20) the following result, proven in [30], is needed.

Proposition 1 If ϱk > 0 for every k = 1, . . . , M2, and ν, λ > 0, then Dϱ + ∆tαi,iB is symmetric and strictly positive definite.

Consequently, assuming that ϱk > 0 for every k = 1, . . . , M2, the Jacobian matrix (20) is invertible provided that

det � IM 2 + ∆tαi,i � AV −Aϱ(Dϱ + ∆tαi,iB)−1(DV + ∆tαi,iAp2) �� ̸= 0.

Clearly, if ∆tαi,i were zero, the above condition holds. Hence, for sufficiently small values of ∆tαi,i the Jacobian matrix H′ is invertible.

In [30,32], multigrid V-cycle algorithm with a small number of preand postGauss-Seidel smoothings was proven to be effective for solving system (18). In particular, this approach was successfully applied in [30,32] to the system formed by the sub-block of the Jacobian matrix (20) given by


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq422.png)

The analysis of linear solvers for the complete Jacobian matrix (20) for approximating the solution of system (19) is beyond the scope of this work.

Title Suppressed Due to Excessive Length 17

3.6 Time-Step selection

The time step is chosen based only on the convective part of the system. It follows that the CFL stability condition for the proposed scheme takes the form


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq423.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq424.png)

where CFL∗is some constant less than one, and cs denotes the maximum of characteristic speeds, computed as

cs = max ����V (i) k,j ���+ � p′ 1(ϱ(i) j ) : i = 1, · · · , s, k = 1, 2, j = 1, · · · , M2 � . (21)

Since the stiff pressure component is treated implicitly, it does not influence the stability condition. As a result, the time step ∆t is independent of the parameter Cp,2, depending only on the non-stiff pressure part p1 and the velocity field v. This allows the method to avoid severe time-step restrictions typically faced on low Mach number regimes. On the other hand, the proposed scheme is not guaranteed to be boundpreserving: the density can be negative or the order parameter c can be outside the physical interval [−1, 1]. As discussed in the monograph [31], the polynomial double-well potential (1) does not satisfy the maximum principle. Consequently, one cannot expect the discrete approximation of c to remain strictly within [−1, 1]. Despite such limitation, this potential is widely used in the literature due to its simplicity [8,13,30,32], but it can be replaced by the classical logarithmic potentials described in [1,8,13,31] and references therein. Nevertheless, in our test the scheme preserves the positivity of the density and keeps c almost within bounds, up to a small deviation. Alternative techniques to mitigate these issues are presented in [30,32], where the time step is reduced whenever |c| exceeds a predefined threshold, and then it is gradually increased back.

4 Asymptotic Preserving Property

In [2] it was proven that in the low Mach number limit of the compressible CahnHilliard-Navier-Stokes system converges to its incompressible counterpart, under suitable initial conditions. The aim of this section, is to show that the proposed scheme is asymptotically stable. To formalize this notion, we recall the definition of an asymptotic preserving scheme provided in [11,12,17,33].

Definition 1 Let Mδ be a continuous physical model depending on a perturbation parameter δ. Define M0 as the limiting model obtained from Mδ when δ →0. A numerical scheme Mδ ∆for approximating Mδ, where ∆= (∆t, ∆x) denotes the temporal and spatial discretization parameters, is said to be asymptotic preserving (AP) if:

1. its stability condition is independent of δ, and 2. in the limit δ →0, the scheme Mδ ∆converges to a consistent discretization M0 ∆ of the continuous limiting model M0.

18 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq425.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq426.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq427.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq428.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq429.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq430.png)


> **Fig. 1 Diagram illustrates the asymptotic-preserving (AP) property. Mδ, M0 denotes the continuous compressible and incompressible system, while Mδ ∆, M0 ∆represents their discrete counterparts, respectively. The AP is verified if the diagram commutes.**

This concept is illustrated in Figure 1.

We denote by Mδ the compressible system (2) in the low Mach number regime, and by Mδ ∆its numerical discretization according to (15). The corresponding incompressible Cahn-Hilliard-Navier-Stokes equations with gravitational acceleration is denoted by M0 and reads as follows:      


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq431.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq432.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq433.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq434.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq435.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq436.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq437.png)

where ρ0 > 0 is the constant density of the incompressible mixture. Here, p(1) denotes the scalar pressure, which acts as a Lagrange multiplier associated with the incompressibility constraint div v = 0. We denote by M0 ∆the discretization of (22) according to (15). Assume that the density, velocity field, concentration difference, and pressure admit the following expansions [12,23,33]:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq438.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq439.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq440.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq441.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq442.png)

and the well-preparedness of the data, that is,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq443.png)

Here, the terms in the pressure expansion follow directly from a Taylor series around ρ(0), so that p(0) = p � ρ(0) �, p(1) = p′ � ρ(0) � ρ(1), and higher-order terms are obtained similarly.

Theorem 1 Consider an IMEX Stiffly Accurate partitioned Runge-Kutta scheme with ˜β = β given by (15). Assume that Un and each stage ˜U(l), U(l) admit the decomposition (23). If Un verifies (24), then so does U(l). Furthermore, if Un+1 admits the decomposition (23), then Un+1 is well-prepared and the scheme is AP.

Title Suppressed Due to Excessive Length 19

Proof For simplicity, in the proof, we shall assume that the pressure splitting stated in (4) is reformulated in terms of the Mach number, that is,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq444.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq445.png)

We define p(l) = p � ρ(l)� for each stage l. Let ∇h, divh and ∆h denote the discrete gradient, divergence and laplacian operators, and Λ∗be the diagonal matrix of the numerical viscosities associated to the flux of the continuity equation. We prove the result by induction on the number of stages s. For one stage s = 1. First, let us show that U(1) is well-prepared. We have that ˜U(1) = Un, so the momentum part in U(1) is

(ρv)(1) = (ρv)n + ∆tα1,1 � −(divh (ρv ⊗v + Cp,1p(ρ)I))n + ρng + divh � T1v(1)�


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq446.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq447.png)

Since α1,1 ̸= 0, taking limits when δ →0 it is obtained that


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq448.png)

By definition of the pressure yields that ρ(1) (0) is constant. The leading terms in the implicit stage for the mass conservation equation are given by

ρ(1) (0) = ρn (0) + ∆tα1,1 ��divh � ρ(0)v(0) ��(1) + divh �˜Λ∗∇hρn (0) ��


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq449.png)

Since both ρ(1) (0) and ρn (0) are constant, summing up the above expression over all spatial indices, a telescope sum in the velocity terms appear, and the boundary contributions vanish due to the boundary conditions (3). Therefore, ρ(1) (0) = ρn (0). Consequently, the divergence free condition is obtained from the mass conservation and that αi,i ̸= 0, i.e., divh v(1) (0) = 0. (26)

Since the scheme is Stiffly Accurate, then Un+1 = U(1), so Un+1 is well-prepared provided it admits the decomposition (23). Let us show that the scheme is AP for s = 1. For simplicity, we shall assume that ρ(1) (0) = 1, otherwise all the terms must be scaled by (ρ(1) (0))−1. The leading

terms in the momentum of U(1), are those involved in the O(1) and O(δ) terms, so applying that ˜ρ(1) (0) = ρn (0), (25) and (26), yields

v(1) (0) = vn (0) + ∆tα1,1 � g −�divh � v(0) ⊗v(0) ��n + divh T2cn (0) + ν∆hv(1) (0) −∇hp(1) (1) � .

Similarly, for the Cahn-Hilliard type equation the leading terms are the O(1), i.e.,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq450.png)

20 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

Since the scheme is Stiffly Accurate, then Un+1 = U(1), so the scheme is AP for s = 1.

We assume that the result is true for the first s −1 stages, and we prove it for stage s. The momentum equation in U(s) is given by

(ρv)(s) = (ρv)n + ∆t �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq451.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq452.png)

+ divh T2˜c(j) j −1 −Cp,1δ


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq453.png)

Taking the limit when δ →0 with αs,s ̸= 0 and applying that the first s−1 implicit stages are well-prepared, yields that


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq454.png)

so ρ(s) (0) is spatially constant. It follows inductively, that the leading terms in the conservation of mass in ˜U(l) for l = 1, · · · , s are

˜ρ(l) (0) = ρn (0) −∆t �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq455.png)

since U(l) is well-prepared for l = 0, · · · , s −1 and ρn (0) is constant. Therefore, ˜ρ(l) (0) is constant for l = 1, · · · , s. Applying again the induction hypothesis and (27), the leading terms in the conservation of mass of the implicit s-stage are:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq456.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq457.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq458.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq459.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq460.png)

Proceeding as before, adding all terms up in the previous expression yields that ρ(s) (0) = ρn (0), which implies that


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq461.png)

So Un+1 is well-prepared since αs,j = βj for every j = 1, · · · , s. It only remains to show that the scheme of s-stages is AP. Similarly, we assume that ρ(l) (0) = 1 for l = 1, . . . , s. Applying that U(l) is well-prepared for l = 1, . . . , s and (27), then the leading terms of the momentum equation at each l-stage are given by

˜v(l) (0) = vn (0) + ∆t �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq462.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq463.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq464.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq465.png)

v(l) (0) = vn (0) + ∆t �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq466.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq467.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq468.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq469.png)

Title Suppressed Due to Excessive Length 21

For the Cahn-Hilliard type equation it is obvious. The AP property follows from the fact that the scheme is Stiffly Accurate.

5 Numerical experiments

The numerical experiments are presented in this section. The main objectives are as follows:

1. To show that the order of the global convergence error agrees with the order of the numerical scheme. 2. To verify that the number of time steps required by the IMEX scheme is consistent with the stability restriction imposed by the convective subsystem (21). 3. To explain the properties preserved by the scheme, such as mass conservation, region preservation along with the CFL value (see Section 3.6), and others.

In all our experiments, the initial CFL number is set to 0.4, the adiabatic exponent γ fixed to 5

3 and the parameters are set to


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq470.png)

We define Cp,1 = � Cp and Cp,2 = Cp−Cp,1. This choice has proven to be effective, as the experiments conducted under this setting have been successful. All experiments were performed using a MATLAB R2024a implementation on a Linux machine running on 32 core of an AMD EPYC 7282. We consider Stiffly Accurate Runge-Kutta schemes. In particular, we use a first-order method defined by the following Butcher tableau:

EE-IE 0 0


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq471.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq472.png)

and a second-order method given by the ∗-DIRKSA scheme:

∗-DIRKSA 0 0 0 1 + s 1 + s 0 s 1 −s , 1 −s 1 −s 0 1 s 1 −s s 1 −s , s = 1 √


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq473.png)

5.1 Order Tests

In this section we show that the ∗-DIRKSA scheme attains second-order of convergence. To this end, we introduce a forcing term into the equations, ensuring that the solution follows a prescribed analytical form. Specifically, the exact solution for the one-dimensional (6) case is defined as


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq474.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq475.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq476.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq477.png)

22 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

and for the two-dimensional case (5),


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq478.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq479.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq480.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq481.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq482.png)

Notice that in both cases the initial velocity filed is divergence-free and the density is constant in space at O(δ). For the performance, the squared Mach numbers are taken as δ = 10−k for k = 1, . . . , 8. The time-step ∆t is determined by the convective subsystem according to (21). We consider meshes of size M = 2i for i = 3, · · · , 8. The global errors and the experimental orders of convergence (EOC) are evaluated at T = 0.01 and are computed as

eM = h2 4 �


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq483.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq484.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq485.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq486.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq487.png)

The results for the oneand two-dimensional cases are shown in Table 1. In both tables, the ∗-DIRKSA scheme achieves second-order convergence, while the EEIE scheme is first-order in the one-dimensional case and in the two-dimensional case for Cp ≤106 decreases away from two. When Cp = 107, 108 in the latter case, the order of convergence seems to tend to two. One possible explanation is that the value of Cp,1 increases significantly as Cp increases, making the timeintegrator more robust. Consequently, the spatial discretization order dominates the convergence. Nevertheless, this phenomenon also occurs for Cp ≤106 until M becomes sufficiently large, as illustrated in Table 1. We expect that for M > 1024 the mentioned orders tends to 1, but due to the computational cost, we did not perform such experiments. In the remainder of this work, we restrict our experiments to the ∗-DIRKSA scheme, since it consistently attains second-order of convergence.

5.2 Test 1, 2 and 3

In this section, we evaluate the performance of the following tests for several stiff pressure coefficients Cp = 102k for k = 1, . . . , 4. For this purpose, mass conservation, region preserving for the c-variable and the limit properties of the compressible scheme are discussed numerically.

Test 1 This test is designed to show that the method remains stable even when the initial condition for the c-variable lies within the unstable region (−1 √


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq488.png)

3) (see [14,30,32]). In particular, we consider the following initial conditions:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq489.png)

v0(x, y) = (1 + δ) ((1 −cos(2πx)) sin(2πy), (cos(2πy) −1) sin(2πx)) ,


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq490.png)

Title Suppressed Due to Excessive Length 23

1D 2D ∗-DIRKSA EE-IE ∗-DIRKSA EE-IE Cp M eM EOCM eM EOCM eM EOCM eM EOCM

10

8 1.317e-03 — 8.799e-04 — 2.4179e-02 — 2.0659e-02 — 16 2.608e-04 2.336 2.113e-04 2.058 6.8859e-03 1.81 6.3482e-03 1.70 32 6.411e-05 2.024 1.490e-04 0.504 1.8061e-03 1.93 1.8142e-03 1.81 64 1.609e-05 1.995 8.938e-05 0.738 4.5478e-04 1.99 5.0700e-04 1.84 128 4.028e-06 1.998 4.947e-05 0.853 1.1369e-04 2.00 1.4907e-04 1.77 256 1.010e-06 1.996 2.665e-05 0.893 2.8398e-05 2.00 4.8390e-05 1.62 512 2.525e-07 2.000 1.365e-05 0.965 7.0908e-06 1.86 1.7788e-05 1.19 1024 6.317e-08 1.999 6.933e-06 0.977 1.76861-06 2.00 7.3560e-06 1.27

102

8 1.216e-03 — 6.669e-04 — 1.9006e-02 — 1.6018e-02 — 16 2.904e-04 2.066 1.247e-04 2.419 6.0020e-03 1.66 5.6030e-03 1.52 32 7.263e-05 1.999 7.355e-05 0.762 1.6209e-03 1.89 1.6349e-03 1.78 64 1.814e-05 2.001 5.459e-05 0.430 4.1116e-04 1.98 4.5097e-04 1.86 128 4.540e-06 1.998 3.312e-05 0.721 1.0310e-04 2.00 1.2913e-04 1.80 256 1.135e-06 2.000 1.772e-05 0.902 2.5790e-05 2.00 4.0565e-05 1.67 512 2.837e-07 2.000 9.175e-06 0.949 6.4443e-06 2.00 1.4410e-05 1.49 1024 7.094e-08 2.000 4.686e-06 0.969 1.6087e-06 2.00 5.8013e-06 1.31

103

8 6.203e-04 — 3.995e-04 — 1.2832e-02 — 1.1383e-02 — 16 1.246e-04 2.316 9.067e-05 2.139 5.5906e-03 1.20 5.2857e-03 1.11 32 2.866e-05 2.120 2.532e-05 1.840 1.6172e-03 1.79 1.6096e-03 1.72 64 7.024e-06 2.029 1.422e-05 0.832 4.1831e-04 1.95 4.4163e-04 1.87 128 1.749e-06 2.006 8.406e-06 0.759 1.0544e-04 1.99 1.2178e-04 1.86 256 4.369e-07 2.002 4.508e-06 0.899 2.6408e-05 2.00 3.5516e-05 1.78 512 1.092e-07 2.000 2.367e-06 0.930 6.6054e-06 2.00 1.1369e-05 1.64 1024 2.730e-08 2.000 1.203e-06 0.977 1.6524e-06 2.00 4.0807e-06 1.48

104

8 1.312e-04 — 1.258e-04 — 6.5014e-03 — 6.1598e-03 — 16 6.469e-05 1.020 4.345e-05 1.534 4.7449e-03 0.45 4.6011e-03 0.42 32 1.728e-05 1.905 1.717e-05 1.340 1.5540e-03 1.61 1.5527e-03 1.57 64 4.273e-06 2.015 6.935e-06 1.308 4.1191e-04 1.92 4.2678e-04 1.86 128 1.063e-06 2.007 3.404e-06 1.027 1.0445e-04 1.98 1.1448e-04 1.90 256 2.653e-07 2.002 1.840e-06 0.888 2.6203e-05 1.99 3.1766e-05 1.85 512 6.630e-08 2.001 9.787e-07 0.911 6.5566e-06 2.00 9.4616e-06 1.75 1024 1.657e-08 2.000 5.067e-07 0.950 1.6400e-06 2.00 3.1202e-06 1.60

105

8 1.338e-04 — 1.152e-04 — 2.1418e-02 — 2.0674e-02 — 16 3.780e-05 1.824 3.461e-05 1.736 3.8763e-03 2.47 3.7918e-03 2.45 32 1.021e-05 1.888 1.175e-05 1.558 1.4648e-03 1.40 1.4683e-03 1.37 64 2.013e-06 2.343 4.435e-06 1.406 4.0482e-04 1.86 4.1362e-04 1.83 128 5.653e-07 1.832 1.907e-06 1.218 1.0380e-04 1.96 1.0967e-04 1.92 256 1.451e-07 1.962 8.772e-07 1.120 2.6128e-05 1.99 2.9357e-05 1.90 512 3.650e-08 1.991 4.304e-07 1.027 6.5437e-06 1.99 8.2499e-06 1.83 1024 9.139e-09 1.998 2.162e-07 0.994 1.6367e-06 2.00 2.5131e-06 1.71

106

8 1.295e-04 — 1.189e-04 — 5.4800e-02 — 5.3752e-02 — 16 3.100e-05 2.062 3.175e-05 1.905 3.7253e-03 3.88 3.6616e-03 3.88 32 8.049e-06 1.946 9.094e-06 1.804 1.3844e-03 1.43 1.3855e-03 1.40 64 2.135e-06 1.915 3.101e-06 1.552 3.9548e-04 1.81 4.0041e-04 1.79 128 5.455e-07 1.968 1.200e-06 1.370 1.0315e-04 1.94 1.0638e-04 1.91 256 1.345e-07 2.020 5.229e-07 1.198 2.6102e-05 1.98 2.7888e-05 1.93 512 3.307e-08 2.024 2.455e-07 1.091 6.5447e-06 1.99 7.5033e-06 1.83 1024 8.226e-09 2.007 1.194e-07 1.040 1.6374e-06 2.00 2.1390e-06 1.81

107

8 1.691e-04 — 1.596e-04 — 1.1107e-01 — 1.1000e-01 — 16 3.108e-05 2.444 3.056e-05 2.384 6.1995e-03 4.16 6.1436e-03 4.16 32 7.937e-06 1.969 8.362e-06 1.870 1.3430e-03 2.21 1.3431e-03 2.19 64 1.991e-06 1.995 2.387e-06 1.809 3.8404e-04 1.81 3.8672e-04 1.80 128 4.974e-07 2.001 8.252e-07 1.532 1.0202e-04 1.91 1.0379e-04 1.90 256 1.279e-07 1.960 3.265e-07 1.338 2.6019e-05 1.97 2.7000e-05 1.94 512 3.242e-08 1.980 1.447e-07 1.174 6.5401e-06 1.99 6.9520e-06 1.96 1024 8.413e-09 1.946 6.849e-08 1.079 — — — —

108

8 2.433e-04 — 2.374e-04 — 2.0294e-01 — 2.0219e-01 — 16 3.118e-05 2.964 3.076e-05 2.948 1.4330e-02 3.82 1.4267e-02 3.82 32 7.918e-06 1.977 8.071e-06 1.930 1.4663e-03 3.29 1.4650e-03 3.28 64 1.989e-06 1.993 2.134e-06 1.919 3.7428e-04 1.97 3.7570e-04 1.96 128 4.978e-07 1.998 6.250e-07 1.772 1.0034e-04 1.90 1.0132e-04 1.89 256 1.253e-07 1.990 2.207e-07 1.502 2.5881e-05 1.95 2.6416e-05 1.94 512 3.168e-08 1.984 8.928e-08 1.305 6.5312e-06 1.99 6.8161e-06 1.96 1024 8.156e-09 1.957 4.014e-08 1.153 — — — —


> **Table 1 L1 errors and experimental order of convergence for the DIRKSA and EE-IE IMEX schemes for both the oneand two-dimensional case, evaluated for different Cp values with Cp,1 = � Cp, in the test using a forced solution.**

24 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

which clearly verify the boundary conditions (3) and the divergence-free condition for the velocity field. The density is constant in space at leading orders of the low Mach number.

It is observed in Figures 2, 3, 4 that initially the density is dispersed and the order parameter c lies within the interval (−1 √


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq491.png)

3). However, as the simulation evolves, the density is gradually increased near the bottom boundary y = 0 due to the effects of gravitation. In addition, the evolution of the c-variable illustrates the process of phase separation where complex patterns are formed.


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq492.png)


> **Fig. 2 Results for Test 1, T = 0, 0.01, M = 128 and Cp = 108. Initially, c is lies within the unstable region. At the beginning of the simulation, phase separation occurs. Moreover, the density starts to become higher in the lower part of the domain due to gravity.**


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq493.png)


> **Fig. 3 Results for Test 1, T = 0.03, 0.05, M = 128 and Cp = 108. The process of spinodal decomposition continues, and the density is accumulating at the bottom of the domain due to gravity.**

Title Suppressed Due to Excessive Length 25


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq494.png)


> **Fig. 4 Results for Test 1, T = 0.07, 0.1, M = 128 and Cp = 108. It can be observed that density remains almost constant among distinct times and that phase separation has almost finished.**

Test 2 The objective of this test is to assess the performance of the scheme when the order parameter c initially lies outside the unstable region. We consider:


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq495.png)

v0(x, y) = (1 + δ) ((1 −cos(2πx)) sin(2πy), (cos(2πy) −1) sin(2πx)) ,

c0(x, y) = 3


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq496.png)

which verify (3) and div v = 0 and the density is constant in space at O(δ). Figures 5, 6 and 7 show that the order parameter c evolves from outside the spinodal region toward a constant value of 3/4. In this state, the system follows the compressible Navier-Stokes with gravitational forces behaving in a low Mach number regime when Cp becomes large.


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq497.png)


> **Fig. 5 Results for Test 2, T = 0, 0.01, M = 128 and Cp = 108. Initially, c lies outside the unstable region and the density is dispersed. At T = 0.01, the fluid starts to have denser regions near the bottom.**

Test 3 The aim of this test, taken from [30,32], is to show the spinodal decomposition. To this end, the initial conditions are set as ρ0 = 1, v0 = 0, and

26 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq498.png)


> **Fig. 6 Results for Test 2, T = 0.03, 0.05, M = 128 and Cp = 108. The bubbles formed order parameter start to merge around 3**

4 also growing in size. The density is higher at the bottom of the domain due to gravity.


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq499.png)


> **Fig. 7 Results for Test 2, T = 0.07, 0.1, M = 128 and Cp = 108. Density remains almost constant, while the order parameter starts tending to 3**

4 .

c0 is initialized as a uniform random sample of zero mean and 10−10 standard deviation. Figures 8 and 9 show the results up to T = 0.1, where the spinodal decomposition occurs at the beginning of the simulations. In addition, density becomes higher near the bottom boundary y = 0 due to the gravitational effects, but it seems to be stabilized as time evolves, see Figure 9.

5.2.1 Conservation of mass and bound-preserving properties


> **Figure 10 illustrates the conservation of mass for the three tests. Specifically, the mass conservation errors for ρ and q = ρc are computed using**


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq500.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq501.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq502.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq503.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq504.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq505.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq506.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq507.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq508.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq509.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq510.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq511.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq512.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq513.png)

Title Suppressed Due to Excessive Length 27


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq514.png)


> **Fig. 8 Results for Test 3, T = 0, 0.01, M = 128 and Cp = 108. Initially, density is constant, velocity is zero everywhere, and the order parameter is a random perturbation around c = 0. It can be seen at T = 0.01 that phase separation has started to occur, while density is higher at the bottom.**


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq515.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq516.png)


> **Fig. 9 Results for Test 3, T = 0.03, 0.05, 0.07, 0.1, M = 128 and Cp = 108. The process of phase separation continues, and the density seems to stabilize with higher values at the bottom of the domain.**

On the other hand, the order parameter c has not exceeded [−1, 1] considerably, and it has been kept below its bounds throughout all experiments. Table 2 shows the maxim and minimum value of c during the performance. Figure 11 shows the time evolution of the maximum and minimum values of the c-component, which rarely exceed the interval [−1, 1]. Therefore, the chosen CFL number of 0.4 can be considered safe for our simulations.

28 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez


> **Fig. 10 Time evolution of the mass conservation errors for both ρ and q with M = 128 for Test 1, 2, and 3 for Cp = 108.**


> **Fig. 11 Time evolution of the maximum and minimum values of the order parameter c with M = 128 for Test 1, 2, and 3 for Cp = 108.**


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq517.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq518.png)


![Equation](images/[2026] implicit explicit 전속도 압축성 CH-NS eq_eq519.png)


> **Table 2 Maximum and minimum values for the c evolution in Test 1 and 3.**

5.2.2 Low Mach number regime

Here, we analyze the limit of the compressible scheme Mδ ∆toward the incompressible scheme M0 ∆. To this end, we focus on the three previous tests specially when the squared Mach numbers are δ = 10−2k for k = 2, 3, 4. Note that in these tests, the initial conditions are well-prepared in the sense of (24). In addition, according to Theorem 1, each time-step must be well-prepared, ensuring that the scheme is AP. Figures 12 and 13 illustrate this behavior: the density approaches to 1, and the divergence free condition is satisfied.


> **Fig. 12 Divergence free condition for Test 1, Test 2 and Test 3 with M = 128 and Cp = 108.**

Title Suppressed Due to Excessive Length 29


> **Fig. 13 Well-preparedness of the solution for Test 1, 2, and 3 at T = 0.1 with M = 128 and Cp = 108.**

6 Conclusions and future work

In this work, we present an efficient second-order asymptotic-preserving IMEX schemes on staggered grids for the two-dimensional compressible isentropic CahnHilliard-Navier-Stokes equations for any Mach number regime. The proposed method avoids the severe restriction imposed by the high-order and stiff pressure terms. To validate the method, several numerical test have been performed, showing that second-order accuracy is achieved with the time-step constrained only by the convective subsystem of the equations. For future work, we aim to extend the present framework to the non-barotropic compressible Cahn-Hilliard-Navier-Stokes in a low Mach number regime, as well as to the three-dimensional case using Galerkin techniques. Regarding the pressure splitting defined in (4), we plan to further research on the possible range of Cp,i values. Our current strategy of setting Cp,1 = � Cp has proven to be successful in our experiments, although no formal proof is given. When solving systems (17) and (18), neither the positivity of the density ρ nor the boundedness of the order parameter c ∈[−1, 1] can be guaranteed. To address this issue, we plan to employ bound-preserving high-order reconstructions schemes which can effectively circumvent these physical constraints.

Conflict of interest

The authors declare that they have no conflict of interest.

Data Availability Statements

Data sharing is not applicable to this article as no datasets were generated or analyzed during the current study.

Acknowledgments

This paper has received financial support from the research projects PID2023146836NB-I00, granted by MCIN/ AEI /10.13039/ 501100011033, and CIAICO/2024/089, granted by GVA.


## References

1. H. Abels, and E. Feireisl. On a diffuse interface model for a two-phase flow of compressible viscous fluids. Indiana Univ. Math. J., 57(2):659–698, 2008.

30 Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez

2. H. Abels, Y. Liu, and ˇS. Neˇcasov´a. Low Mach number limit of a diffuse interface model for two-phase flows of compressible viscous fluids. GAMM-Mitteilungen, 47(4):e202470008, 2024. 3. T. Alazard. Low Mach number limit of the full Navier-Stokes equations. Arch. Ration. Mech. Anal., 180(1):1–73, 2006. 4. R. B¨urger, D. Inzunza, P. Mulet, and L. M. Villada. Implicit–explicit schemes for nonlinear nonlocal equations with a gradient flow structure in one space dimension. NMPDE, 35(3):1008–1034, 2019. 5. A. Baeza, R. Burger, P. Mulet, and D. Zorio. On the Efficient Computation of Smoothness Indicators for a Class of WENO Reconstructions. J. Sci. Comput., 80(2):1240–1263, 2019. 6. A. Baeza, R. B¨urger, P. Mulet, and D. Zorio. WENO Reconstructions of Unconditionally Optimal High Order. SIAM J. Numer. Anal., 57(6):2760–2784, 2019. 7. S. Boscarino, R. B¨urger, P. Mulet, G. Russo, and L.M. Villada. Linearly Implicit Imex Runge-Kutta Methods for a Class of Degenerate Convection-Diffusion Problems. SIAM J. Sci. Comp., 37(2):B305–B331, 2015. 8. F. Boyer. Mathematical study of multi-phase flow under shear through order parameter formulation. Asymptot. Anal., 20(2):175–212, 1999. 9. J.W. Cahn, and J.E. Hilliard. Free energy of a nonuniform system .3. Nucleation in a 2-component incompressible fluid. J. Chem. Phys., 31(3):688–699, 1959. 10. L. Chen, and J. Zhao. A novel second-order linear scheme for the Cahn-Hilliard-NavierStokes equations. J. Comput. Phys., 423, 2020. 11. F. Cordier, P. Degond, and A. Kumbaro. An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations. J. Comput. Phys., 231(17):5685–5704, 2012. 12. P. Degond, and M. Tang. All speed scheme for the low Mach number limit of the isentropic Euler equations. Commun. Comput. Phys., 10(1):1–31, 2011. 13. F. Dhaouadi, M. Dumbser, and S. Gavrilyuk. A first-order hyperbolic reformulation of the Cahn–Hilliard equation. Proc. Royal Soc. A, 481(2312):20240606, 2025. 14. C.M. Elliott. The Cahn-Hilliard model for the kinetics of phase separation. In Math. Models Phase Change Probl. ( ´Obidos, 1988), volume 88 of Internat. Ser. Numer. Math., pages 35–73. Birkh¨auser, Basel, 1989. 15. J.L. Ericksen. Liquid crystals with variable degree of orientation. Arch. Ration. Mech. Anal., 113(2), 97–120, 1991. 16. D.J. Eyre. Unconditionally gradient stable time marching the Cahn-Hilliard equation. In J.W. Bullard, L.Q. Chen, R.K. Kalia, and A.M. Stoneham, editors, Comput. Math. Models Microstruct. Evol., volume 529, pages 39–46. Mat. Res. Soc. (MRS), 1998. 17. J Haack, S Jin, and J Liu. An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations. Commun. Comput. Phys., 12(4), 955–980, 2012. 18. D. Han, and X. Wang. A second order in time, uniquely solvable, unconditionally stable numerical scheme for Cahn-Hilliard-Navier-Stokes equation. J. Comput. Phys., 290:139– 156, 2015. 19. Q. He, and X. Shi. Numerical Study of Compressible Navier-Stokes-Cahn-Hilliard System. Comm. Math. Sci., 18(2):571–591, 2020. 20. F. H. Harlow, and J. E. Welch. Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface. Phys. Fluids, 8(12):2182, 1965. 21. H. Jia, X. Wang, and K. Li. A novel linear, unconditional energy stable scheme for the incompressible Cahn-Hilliard-Navier-Stokes phase-field model. Comput. Math. Appl., 80(12):2948–2971, 2020. 22. G.J. Kynch. A Theory of Sedimentation. Trans. Faraday Soc., 48(2):166–176, 1952. 23. S Klainerman, and A Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids. Commun. Pure Appl. Math., 34(4), 481–524, 1981. 24. P.L. Lions, and N. Masmoudi. Incompressible limit for a viscous compressible fluid. J. Math. Pures Appl., 77(6), 585–627, 1998. 25. M. Li, and C. Xu. New efficient time-stepping schemes for the Navier-Stokes-Cahn-Hilliard equations. Comput. Fluids, 231, 2021. 26. J. Lowengrub, and L. Truskinovsky. Quasi-incompressible Cahn-Hilliard fluids and topological transitions. Proc. Royal Soc. A, 454(1978):2617–2654, 1998. 27. C. Par´es-Pulido, S. Mishra, and K.G. Pressel. Arbitrarily high-order (weighted) essentially non-oscillatory finite difference schemes for anelastic flows on staggered meshes. SAM Res. Rep., 2019, ETH Zurich.

Title Suppressed Due to Excessive Length 31

28. L. Pareschi, and G. Russo. Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation. J. Sci. Comput., 25(1/2):129–155, 2005. 29. S. Patankar. Numerical Heat Transfer and Fluid Flow. CRC Press, 2018. 30. A. Martorell, P. Mulet, and D. F. Y´a˜nez. Implicit-explicit schemes for compressible Cahn– Hilliard–Navier–Stokes equations on staggered grids. arXiv preprint arXiv:2512.20351, 2025. 31. A. Miranville. The Cahn—Hilliard equation: recent advances and applications. SIAM, 2019. 32. P. Mulet. Implicit-Explicit Schemes for Compressible Cahn–Hilliard–Navier–Stokes Equations. J. Sci. Comput., 101(2):36, 2024. 33. S. Noelle, G. Bispen, K.R. Arun, M. Luk´aˇcov´a-Medvidˇdov´a, and C.D. Munz. A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas dynamics. SIAM J. Sci. Comput., 36(6), B989–B1024, 2014. 34. C.W. Shu. High Order Weighted Essentially Nonoscillatory Schemes for Convection Dominated Problems. SIAM Rev., 51(1):82–126, 2009. 35. D.B. Siano. Layered sedimentation in suspensions of monodisperse spherical colloidal particles. J. Colloid Interface Sci., 68(1):111–127, 1979. 36. B.P. Vollmayr-Lee, and A.D. Rutenberg. Fast and accurate coarsening simulation with an unconditionally stable time step. Phys. Rev. E, 68(6, 2), 2003.

