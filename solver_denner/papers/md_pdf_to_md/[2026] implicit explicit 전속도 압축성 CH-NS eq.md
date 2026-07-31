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

