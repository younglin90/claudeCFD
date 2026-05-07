Journal of Computational Physics 397 (2019) 108847


### Contents lists available at ScienceDirect


# Journal of Computational Physics


### www.elsevier.com/locate/jcp


# Preconditioning a Newton-Krylov solver for all-speed melt pool flow physics


# Brian Weston a,∗, Robert Nourgaliev b, Jean-Pierre Delplanque c, Andrew T. Barker a

a Center for Applied Scientific Computing, Lawrence Livermore National Laboratory, Livermore, CA 94551, USA b Design Physics Division, Lawrence Livermore National Laboratory, Livermore, CA 94551, USA c Mechanical & Aerospace Engineering, University of California Davis, Davis, CA 95616, USA


## a r t i c l e i n f o a b s t r a c t

Article history: Received 25 December 2018 Received in revised form 22 June 2019 Accepted 19 July 2019 Available online 25 July 2019

Keywords: Block preconditioning Physics-based preconditioning Fully implicit Newton Krylov All speed fluid dynamics Reconstructed discontinuous Galerkin method

In this paper, we introduce a multigrid block-based preconditioner for solving linear systems arising from a Discontinuous Galerkin discretization of the all-speed Navier-Stokes equations with phase change. The equations are discretized in conservative form with a reconstructed Discontinuous Galerkin (rDG) method and integrated with fully-implicit time discretization schemes. To robustly converge the numerically stiff systems, we use the Newton-Krylov framework with a primitive-variable formulation (pressure, velocity, and temperature), which is better conditioned than the conservative-variable form at lowMach number. In the limit of large acoustic CFL number and viscous Fourier number, there is a strong coupling between the velocity-pressure system and the linear systems become non-diagonally dominant. To effectively solve these ill-conditioned systems, an approximate block factorization preconditioner is developed, which uses the Schur complement to reduce a 3 × 3 block system into a sequence of two 2 × 2 block systems: velocity-pressure, vP, and velocity-temperature, vT . We compare the performance of the vP-vT Schur complement preconditioner to classic preconditioning strategies: monolithic algebraic multigrid (AMG), element-block SOR, and primitive variable block Gauss-Seidel. The performance of the preconditioned solver is investigated in the limit of large CFL and Fourier numbers for low-Mach lid-driven cavity flow, Rayleigh-Bénard melt convection, compressible internally heated convection, and 3D laser-induced melt pool flow. Numerical results demonstrate that the vP-vT Schur complement preconditioned solver scales well both algorithmically and in parallel, and is robust for highly ill-conditioned systems, for all tested rDG discretization schemes (up to 4th-order).

Published by Elsevier Inc.


### 1. Introduction

This work is motivated by the development of large-scale multi-physics simulations of the powder bed fusion (PBF) process in 3D metal additive manufacturing and slow cook-off of energetic materials. Simulating these melt convection processes require interface tracking of multiple materials and phases, which have large variations in the thermo-physical properties across sharp solid-liquid-gas interfaces. The density variations associated with rapid phase change (melting/so-


# * Corresponding author. E-mail address: weston8@llnl.gov (B. Weston).

https://doi.org/10.1016/j.jcp.2019.07.045 0021-9991/Published by Elsevier Inc.

2 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847

lidification and evaporation/condensation) tightly couple the governing equations, rendering incompressible flow solvers, such as the SIMPLE/projection-family of algorithms, ineffective for these applications. Therefore, an adequate numerical approach for powder bed fusion and slow cook-off requires an all-speed fully-compressible formulation. The present study is focused on developing a parallel and algorithmically scalable preconditioner to robustly converge time-accurate solutions of ill-conditioned systems, arising from all-speed melt pool flow physics.

As outlined in [6,37,48], simulating low-Mach compressible flow is challenging because the governing equations change nature. In the limit of very low-Mach number, the compressible Euler equations transition from a hyperbolic type to mixed hyperbolic-elliptic type. As a result, two categories of all-speed solvers have been developed for simulating flows under these transitional conditions: density-based methods derived from high-speed compressible solvers and pressure-based methods derived from incompressible solvers.1 For small density variations, incompressible flow solvers utilizing the Boussinesq approximation are widely used to model low-speed compressible flows [17,24,68]. For large temperature gradients, which can subsequently result in large density variations, the Boussinesq approximation is no longer valid [6,38,46] and a compressible formulation is required. On the other hand, compressible flow solvers suffer from challenges of their own in the low-Mach regime. First, the numerical solutions are known to become inaccurate in the limit of low-Mach number due to the lack of numerical dissipation [27,41]. To address accuracy in the limit of low-Mach number, the Advection Upstream Splitting Method (AUSM) schemes were developed, which correctly mimic the pressure fluctuations of an incompressible flow solver in the asymptotic limit of small Mach number [40–42]. Second, due to the disparity between the acoustic and velocity time-scales, the system is numerically stiff with a large condition number, causing convergence to stall [11,41,48,65,66,69]. Due to the numerical stiffness and necessity of stepping over acoustic and advection timescales,2 numerical algorithms using explicit time integrators (time marching methods) or operator-splitting algorithms, such as projection algorithms, have severe time step restrictions due to stability requirements, making large-scale simulations prohibitively expensive and requiring weeks to months with hundreds of processors [30,36]. The work of Turkel [65,66], Choi and Merkle [11], Weiss and Smith [69], Van Leer et al. [67], and Briley et al. [8] have greatly improved the solvability of compressible flow solvers in the low-Mach limit. These time-derivative preconditioning techniques multiply the time-derivative term in the governing equations with a matrix that scales the acoustic time-scale relative to the velocity time-scale, effectively reducing the speed of sound to lower the condition number.

With a fully-implicit time discretization, large time steps can be taken, chosen based on the adequate resolution of the dynamic time-scales of the problem, rather than by numerical stability restrictions dictated by the physical time-scales of the problem. For all-speed melt pool flow, there are several physical time-scales associated with acoustic waves, viscous diffusion, thermal diffusion, material advection, Marangoni convection, phase transitions, and heat fluxes at the surface of the melt pool. Stepping over these time-scales, however, results in numerical systems that develop strong hyperbolic and parabolic stiffnesses [10]. To effectively step over these time-scales, we employ the BDF2 L-stable time integrator [52].

In order to solve the system of non-linear equations at each time step, a robust solution algorithm is needed. The Newton-Krylov framework, which uses an outer Newton method combined with an inner Krylov (iterative) method, provides such an approach [32,35]. We use a global line search strategy, where the descent direction and step size are computed with a Jacobian-Free Newton-Krylov (JFNK) framework. Since upwinding is embedded in the approximate Riemann solver, there is a non-symmetry in the underlying systems, and the conjugate gradient method cannot be used. As a result, GMRES is chosen as the Krylov (linear) solver because it is robust and guarantees a monotonically decreasing residual. Furthermore, there are three primary challenges for solving linear systems arising from the discretization of all-speed melt pool flow. First, there is a strong coupling between the velocity and pressure fields as the Mach number tends to zero. Second, for laser-induced melt pool flow, there is a tight-coupling between the velocity and temperature fields due to Marangoni convection and velocity suppression models, which are required to enforce the “no-velocity” condition in the solid phase. Third, velocity suppression models create non-diagonally dominant linear systems – presenting significant challenges for the solvability of the underlying linear algebra with classic iterative methods. Thus, the global linear system of discrete equations is highly ill-conditioned and preconditioning the Newton-Krylov solver is necessary for convergence.

Domain decomposition methods, such as additive Schwarz variants, are natural for unstructured meshes [64], taking a divide-and-conquer approach to parallelism [32,58]. Additive Schwarz techniques have been effective preconditioners for Newton-Krylov solvers in incompressible flow [39], compressible flow [62], reactive flow problems [60], and lowMach compressible combustion [33]. A drawback of one-level additive Schwarz methods is the locality assumption [61,63]. Neighbor-element degrees of freedom are coupled, while long-range interactions are ignored. As a result, the performance of additive Schwarz preconditioners generally degrades for elliptic problems as the number of processor domains increase, due to the lack of global coupling [39,61,63]. To address the lack of global coupling in domain decomposition methods, a global physics-based approach segregates all of the degrees of freedom of a particular field into separate blocks. The reduced scalar block systems are more amenable for iterative methods to approximate the action of their inverse. These physics-based preconditioning techniques often use legacy algorithms to solve reduced scalar block systems in an operator-split fashion,

1 The use of all-speed refers to the solver’s ability to solve the flow equations at all speeds without needing to modify the governing equations in the limit of low and high Mach number.

2 Dynamic time-scales of melt convection problems are rather large compared to CFL/Fourier number based time-scales. Cost-effective simulations necessitate stepping over advection timescales.

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 3

similar to operator-splitting methods for time-integration. Typically, these strategies implicitly couple the diagonal primitive variable blocks, while explicitly coupling the off-diagonal primitive variable blocks by moving them to the right-hand side. Variations of these physics-based preconditioning approaches have been successfully applied to the non-equilibrium radiation diffusion equations [47], the shallow water equations [49], solidifying flow applications [34], the incompressible Navier-Stokes equations [24,56], and more recently to the compressible Navier-Stokes equations [52,54,57]. However, many of these operator-split physics-based preconditioning strategies are ineffective for the compressible Navier-Stokes equations in the limit of low-Mach number, since the pressure and velocity fields become tightly coupled at large acoustic CFL numbers [52,54]. On the other hand, block LU decompositions of the Jacobian matrix is an effective preconditioning approach for these ill-conditioned systems, but full LU factorizations are prohibitively expensive for large-scale simulations, since their computational cost generally scales as O(n3).

As a result, this paper develops approximate block factorization strategies on the segregated primitive variable blocks, which are more robust than operator-split physics-based preconditioning strategies, but more computationally efficient than full block LU factorizations. Approximate primitive variable block factorizations have been explored as a preconditioner for the incompressible Navier-Stokes equations, MHD, and radiation diffusion using various approximations of the Schur complement matrix [9,10,15,16,22,23]. To effectively solve our tightly coupled linear systems, we develop a multigrid blockbased preconditioner, which uses the Schur complement to reduce a 3 × 3 block system into a sequence of two 2 × 2 block systems: velocity-pressure, vP, and velocity-temperature, vT . The Schur complement systems are multigrid block reductions and algebraic multigrid (AMG) methods can be effectively applied on these reduced systems, which are often linearly scalable algorithms, O(n) [7,13,19,28]. A similar strategy of reducing a 3 × 3 block into two 2 × 2 blocks systems was developed in [15] and applied to incompressible MHD applications. Our work differs in that we extend these techniques to the compressible all-speed Navier-Stokes equations with phase change using a high-order reconstructed Discontinuous Galerkin discretization in the limit of large CFL and Fourier numbers.

In this paper, we primarily focus on preconditioning linear systems that arise from our P0 P1 (2nd-order accurate) spatial discretization scheme. We have conducted an initial investigation for preconditioning our P1 P3 and P2 P3 (4nd-order accurate) discretization schemes, but a more in-depth study will be done in future work. Lastly, all of our test problems in this paper are subsonic because the focus of this study is on preconditioning for melt pool flow physics.3 Supersonic flow problems can be found in our recently published work [50], which uses the same discretization and Newton-Krylov algorithm.

The framework is implemented and tested within LLNL’s ALE3D code [1]. ALE3D is a multi-physics numerical simulation tool, focusing on modeling hydrodynamics and structural mechanics in all-speed multi-material applications. Additional ALE3D features include heat conduction, chemical kinetics and species diffusion, incompressible flow, a wide range of material models, chemistry models, multi-phase flow, and magneto-hydrodynamics for long-implicit and short-explicit time-scale applications.

The rest of the paper is organized as follows. The physical and mathematical models required for simulating all-speed melt pool flow are discussed in Section 2. In Section 3, a brief introduction of the reconstructed Discontinuous Galerkin (rDG) spatial discretization scheme is given, and its advantages for the underlying linear algebra are highlighted. Furthermore, the fully-implicit time discretization schemes and the Jacobian-Free Newton-Krylov (JFNK) solver and its preconditioning strategies are outlined. In Section 4, we compare the performance of the preconditioners for low-Mach lid-driven cavity flow, Rayleigh-Bénard melt convection, compressible thermal convection, and 3D laser-induced melt pool flow. Finally, concluding remarks and future directions are provided in Section 5.


### 2. Mathematical model


### 2.1. Conservation laws


### In this work, we consider the time-dependent compressible Navier-Stokes equations with solid-liquid phase change. The governing conservation equations in the flux vector form are given by


## ∂U


## ∂t + ∇· (G −D) = S, (1)


### where U is the solution vector of conservative variables, defined as


## U = �ρ,ρv,ρe �T , (2)


### while G, D, and S are the vectors of hyperbolic fluxes, diffusion fluxes, and sources, respectively, defined as

3 The use of “all-speed” refers to the Newton-Krylov’s ability to simulate compressible flow at a wide range of Mach numbers without modifying the governing equations, as stated in a previous footnote. From a preconditioning standpoint, the challenge occurs at low-Mach number, when the acoustic time-scale is several orders of magnitude faster than the velocity time-scale, as explained and addressed in this study.

4 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


## G =


## ⎡


## ⎣ ρv ρv ⊗v + PI ρve + Pv


## ⎤


## ⎦, D =


## ⎡


## ⎣ 0 ¯¯σ k∇T + v · ¯¯σ


## ⎤


## ⎦, S =


## ⎡


## ⎣ 0 ρg 0


## ⎤


## ⎦. (3)


# v = (vx, v y, vz) is the material velocity vector in Cartesian coordinates, P is the pressure, ρ is the density, g is gravity, ¯¯σ


## is the viscous stress tensor defined in 2.2, e = u + v2

2 is the specific total energy, u is the specific internal energy, k is the thermal conductivity, and T is the temperature. In this work, we neglect energy dissipation due to viscous stresses and pressure work, since these are negligible in the vanishing Mach number limit. Radiation, evaporation, and thermo-capillary convection are modeled as free-surface boundary conditions, which will be described in Section 2.5.


### We also introduce a vector of “primitive” variables, W, which is generally different from U, and chosen based on the “better system conditioning” considerations, as discussed in Section 3.4.


### 2.2. Viscous stress tensor


### In this work we consider Newtonian fluids. As defined in [2,5,70], the viscous stress tensor is


# ¯¯σ = 2μ¯¯ϵ + λ(∇· v)I, (4)


# where μ is the dynamic viscosity and λ is the second viscosity coefficient. Following Stokes hypothesis, the second viscosity is taken as λ = −2


# 3μ [2]. The strain rate tensor, ¯¯ϵ, is defined as


# ¯¯ϵ = 1


## 2


## � ∇v + (∇v)T � . (5)


### 2.3. Equations of state (EoS)


### 2.3.1. Isothermal polytropic equation of state


## To represent nearly-incompressible fluids, M = |v|


## c < 10−2, we use a simple compressible material formulation with a two-parameter (ρ0, c) isothermal polytropic EoS


# P(ρ) = ρ0c2 �ρ ρ0 −1 � , (6)

where ρ0 and c are the given reference density and sound speed, respectively. Since all of the test problems in this study are in the low-Mach regime, there is no loss in accuracy when the acoustic time-scale is stepped over. The isothermal polytropic EoS is useful as it allows for direct control of the sound speed, enabling us to study the performance of our preconditioned solver at various Mach numbers. With direct control of the sound speed in this EoS, we can artificially reduce the sound speed and limit the Mach number to tractable values (10−2 ≥M ≥10−5), which is favorable for the underlying linear algebra. Furthermore, pressure is decoupled from temperature in this EoS, which also leads to better conditioning of the linear systems.4 This EoS is used in all of the numerical examples except in Section 4.3.


# 2.3.2. γ -Gas equation of state


### We also implement the ideal gas law, defined as


# P(ρ, u) = ρu(γ −1) = ρRT, γ = cp


## cv , (7)


## where R is the universal gas constant, and a thermally perfect gas is assumed for u(T). This EoS is used in a compressible thermal convection problem in Section 4.3.


### 2.4. Modeling of melting and solidification


### 2.4.1. Phase transition

Phase change is implicitly tracked with an energy-based (homogeneous thermal equilibrium) approach. A transitional two-phase region is introduced, between the solid and liquid phases, to avoid a non-singular mapping between u and T . The thickness of the two-phase mushy region is defined by δ = T L −T S, where T L and T S are the boundary temperatures of the two-phase region, separating the pure liquid and pure solid.


### The three material state zones are shown in Fig. 1. The jump in internal energy between the solid and liquid phases is represented by the latent heat, defined as


## u f = uL −uS. (8)

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 5


> **Fig. 1. Thermal model, u(T), for equilibrium phase change.**

To suppress the velocity in the mushy and solid phases, we use a velocity suppression model as outlined in Section 2.4.2. More details are described in our previous work, explaining how transport and thermodynamic properties (i.e. viscosity, thermal conductivity, internal energy, and specific heat) transition between the phases [52].


### 2.4.2. Velocity suppression model

To suppress the velocity in the solid and mushy phases, we use the enhanced viscosity method, also known as the variable viscosity method [17,18,45]. In this model, material strength is a linear function of the strain tensor, where the dynamic viscosity is a function of temperature between the solid and liquid phases,


# μ(T) =


## ⎧ ⎨


## ⎩


# μL if T ≥T L μL fμ (T) if T S < T < T L μS if T ≤T S, (9)


## where fμ(T) is the viscosity factor, a function that smoothly varies from one to a large number, μs


### μl , [52]. This model progressively increases the viscosity of the material from the mushy region to the solid region, inhibiting material deformation.


### 2.5. Boundary conditions


### 2.5.1. Heat fluxes at the surface of the melt pool


### The heat flux at the surface of the melt pool consists of four contributions,


## qsurf ace = qlaser −qrad −qconv −qevap. (10)


### Energy from the laser is deposited at the surface and is modeled with a Gaussian beam profile,


## qlaser = qmax · exp


## � −2r2


## r2 laser


## �


## , (11)

where qmax is the maximum laser flux at the center, rlaser is the effective radius of the laser, and r is the radial distance from the center. The laser flux integrated over the surface is ensured to be equal to the total input power. The power values in the calculation already assume an effective absorption power. Radiation and convective heat losses are calculated with


# qrad = σφ(T 4 −T 4 a ) (12)


# qconv = αconv(T −Ta), (13)

4 In Section 4.2 and in our previous work [50], we numerically verified that the flow dynamics is independent of the Mach number below 10−1. This approach is similar to time-derivative preconditioning methods routinely used in low-speed compressible flow solvers [8,11,66,67,69].

6 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Table 1 Dimensionless numbers.**

Reynolds number Re vL


$$
ν Prandtl number Pr ν α Rayleigh number Ra gβ�T L3
$$


$$
να
$$

Stefan number Ste cp�T

L


$$
Marangoni number Ma −dγ
$$

dT L�T


$$
μα Mach number M |v|
$$

c


# where Ta is the ambient temperature, φ is the emissivity, and αconv is the convective heat transfer coefficient. Energy loss due to material evaporation is modeled with Anisimov’s theory [3],


## qevap = �H v · J v, (14)


## where �H v is the latent heat of vaporization, J v is the material evaporation flux,


## J v = C0 AP(T) √


# 2π MRT (15)


## P(T) = C1 Pa exp � −�


## kB


## �1


## T −1


## Tb


## �� , (16)

�is the evaporation energy per atom, kB is Boltzmann’s constant, T is the surface temperature, Tb is the boiling temperature of the material, Pa is the atmospheric pressure, M is the molar mass, R is the universal gas constant, C0 and C1 are constants, and A is a sticking coefficient, which is close to unity for metals. Evaporative mass loss is neglected and more details on the evaporation model can be found in [31].


### 2.5.2. Marangoni convection

In laser spot welding and selective laser melting, thermal Marangoni convection is the primary driver of convection in the melt pool [20,31]. Marangoni convection is due to surface tension gradients, driving the flow from areas of low to high surface tension (γ ). This is modeled by introducing a traction force (momentum flux) boundary condition, where the velocity gradient is now coupled to the temperature at the free surface,


# τMara = μl ∂vx


# ∂z = ∂γ


## ∂T · ∂T


## ∂r + �

i


# ∂γ ∂ai · ∂ai ∂r . (17)

The boundary shear stress is τMara, r and z are the coordinates parallel and perpendicular to the surface, respectively, vx is the radial velocity component at the surface, and ai is the thermodynamic activity of alloy element i. In this study, we neglect the solutal effects, setting ∂γ


## ∂ai = 0. Details of this Marangoni shear stress model can be found in [20].


### 2.6. Dimensionless numbers


### In Table 1, we define the dimensionless numbers referred to in this study.


### 3. Numerical method

To solve the governing equations in Section 2, we use a fully-implicit, Newton-Krylov solver with the reconstructed Discontinuous Galerkin spatial discretization scheme, developed in [52]. In this section, we describe our numerical approach and highlight its major benefits for the solvability of the underlying linear algebra.

Recently, the Discontinuous Galerkin (DG) method has become increasingly popular in computational fluid dynamics, owing to its flexibility of handling complex geometry, its compact stencil for arbitrarily high-order solutions, and its amenability to parallelization and hp-adaptation. In contrast to more traditional finite volume (FV) methods in CFD, high-order accuracy is achieved by simply adding degrees of freedom (DoFs) per element, per variable. As a result, the DG(Pp) of any order p has the same stencil (i.e., only face-neighbors are involved in discretization), which is a very attractive feature in terms of parallelization and code design. On the other hand, the size of the solution vector grows significantly, as more DoFs must be solved for. Such an increase in the size of the solution vector is unfavorable in the context of implicit solvers, imposing significant memory requirements (for storage of the matrices) and adversely affecting solution scalability, since the majority of linear solvers do not scale linearly [58]. In order to reduce high costs associated with DG, the least-squares based reconstructed DG (rDG) methods have been developed, which hierarchically reconstruct high-order DoFs in-cell to be consistent with solutions in face-neighboring elements. The combination of using in-cell and inter-cell reconstructions provides a flexible framework for generic PDE’s, including transient hyperbolic, parabolic, and elliptic operators [43,44,71].

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 7


> **Table 2 Solution vector size per equation per element for unreconstructed vs. reconstructed DG.**

Dim. DG P1 rDG P0 P1 DG P3 rDG P1 P3 rDG P2 P3

1D 2 DoFs 1 DoFs 4 DoFs 2 DoFs 3 DoFs 2D 3 DoFs 1 DoFs 10 DoFs 3 DoFs 6 DoFs 3D 4 DoFs 1 DoFs 20 DoFs 4 DoFs 10 DoFs


> **Table 3 Number of non-zeros (nnz) per row resulting from three 4th-order spatial discretization schemes for the Navier-Stokes equations.**

Dim. DG P3 rDG P1 P3 rDG P2 P3

1D 36 nnz/row 30 nnz/row 45 nnz/row 2D 200 nnz/row 156 nnz/row 312 nnz/row 3D 700 nnz/row 500 nnz/row 1250 nnz/row

We capitalize on recent work that extended the rDG discretization to a Newton-Krylov framework for solving highly ill-conditioned multi-physics problems [52]. In this work, we use the modal Taylor-based tensor-product basis functions. These basis functions are hierarchical, which naturally facilitates p-refinement, and can be easily implemented on hybrid meshes with AMR. The 0th-order degrees of freedom are cell-averaged quantities, while the high-order degrees of freedom correspond to integral derivatives of solved for quantities (slopes, curvatures, etc). In the rDG P N P M schemes, we solve for a P N scheme of polynomial order N and reconstruct to a P M scheme of polynomial order M, which corresponds to (M + 1)th order-accuracy in space.

In this work, we study the performance of the P0 P1 (2nd-order accurate), P1 P3, and P2 P3 (both 4th-order accurate) spatial discretization schemes. The benefits of reconstructed DG vs. unreconstructed DG with regards to solution vector size are highlighted in Table 2. The DG P3 scheme in 2D has a total of 10 DoFs per equation per element: one cell-averaged value + two slopes + three curvatures + four third-order derivatives, while in 3D it has a total of 20 DoFs per equation per element: one cell-averaged value + three slopes + six curvatures + ten third-order derivatives. The rDG P2 P3 scheme operates on the same polynomial space and has the same effective total number of DoFs and order of accuracy as the DG P3 scheme, but with a significantly smaller solution vector size. This is because only the cell-averaged value, slopes, and curvatures are solved for, while the third-order derivative DoFs are reconstructed.5 The benefits of reconstructed DG vs. unreconstructed DG increases as the dimension goes up, as evident from Table 2.

Furthermore, Table 3 shows the matrix bandwidth for three 4th-order spatial discretization schemes: unreconstructed DG P3, rDG P1P3, and rDG P2 P3. The unreconstructed DG P3 scheme requires only face-neighbors (stencil-1), while the rDG schemes requires information from face-neighbors of face-neighbors (stencil-2). It is evident that the global matrix resulting from the rDG P2 P3 discretization has the largest number of non-zeros per row, while the rDG P1 P3 discretization scheme has the least number of non-zeros per row.6 The benefit of the reduced sparsity relative to unreconstructed DG increases for higher dimensions, which is favorable for reducing linear solver time. Additionally, more aggressive reconstruction strategies, such as rDG P1 P5, were explored previously, leading to further reductions in matrix size and sparsity [53].

Since the base (0th-order) degrees of freedom are cell-averaged quantities, this DG method can be viewed as a generalized extension of the finite-volume (FV) algorithm to high-order (greater than the 2nd-order), without the need to extend the stencil.7


### 3.1. Spatial discretization

The computational domain �is subdivided into a collection of non-overlapping linear QUAD4 (4-node) or HEX8 (8-node) elements, �e. The solution is represented in the broken Sobolev space V (p) h , consisting of discontinuous vector-valued polynomial functions of degree p


## V (p) h = �υh ∈[L2 (�)]m : υh|�e ∈ � Vm p � ∀�e ∈� � , (18)

where m is the dimension of the unknown vector and Vp is the space of all polynomials of degree ≤p. To obtain the weak formulation of the governing equations, we multiply Eq. (1) by a test function, Wh, and then perform integration by parts over an element

5 It is worth emphasizing that the reconstructed degrees of freedom are not part of the solution vector in the non-linear or linear solver. Instead, they are dynamically computed locally to attain high-order accuracy, see Eq. (20).

6 In Table 3, the number of non-zeros per row assumes a constant number of equations per element (4 equations per element in 2D and 5 equations per element in 3D) and a constant number of degrees of freedom per element.

7 The P0 P1 scheme corresponds exactly to a cell-centered 2nd-order finite-volume method, common in most commercial CFD solvers.

8 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


## Rh � U(p) h � = ∂

∂t �

�e U(p) h Whd�+ �

�e


## � G j � U(p) h � −D j � U(p) h �� nWhd�−


## − �

�e


## �� G j � U(p) h � −D j � U(p) h �� ∂Wh


## ∂x j + S � U(p) h � Wh � d�, ∀Wh ∈V (p) h , (19)

where U(p) h are represented by piecewise-polynomial functions of degrees p, which are discontinuous between the cell interfaces, and n denotes the unit outward normal vector to the element face �e (i.e., the boundary of �e). The local


### residual function Rh � U(p) h � is defined by inner products between the solution residue representation (with a chosen set


## of basis functions) and the test functions, Wh. In our fully-implicit solution procedure, we are minimizing this residual function.


### The hyperbolic flux function G j � U(p) h � n appearing in the face integral term of Eq. (19) is replaced by a numerical


### Riemann flux function, H j � U(p)L h , U(p)R h � n, which is computed by an approximate Riemann solver, where U(p)L h and U(p)R h are


### the conservative state vectors at the left and right side of the element boundary. Here, we use the low-Mach version of the AUSM+-up scheme, as in our previous work [50].


### Numerical P N P M polynomial solutions W(M−1) h in each element are expressed using a chosen set of basis functions B(k) (x), as


## W(M−1) h (x,t) =

N−1 �

k=0 W(k)e (t)B(k) (x)


## � �� �

Solved-for


## +


$$
M−1 �
$$

k=N W(k)e (t)B(k) (x)


## � �� �

Reconstructed


## + O(h(M−1)), (20)

where W(k)e denotes degrees of freedom (DoFs) in an element e. Note that in this work, we use a DG discretization for primitive variables (pressure, velocity, temperature) instead of conservative variables (mass, momentum, energy), see Section 3.4. The residual Eq. (19) now becomes


## R(N−1) h � W(M−1) h � = ∂


$$
∂t �
$$


$$
�e U(N−1) h � W(M−1) h � Whd�+ �
$$

�e


## � G j � W(M−1) h � −D j � W(M−1) h �� nWhd�−


## − �

�e


## �� G j � W(M−1) h � −D j � W(M−1) h �� ∂Wh


## ∂x j + S � W(M−1) h � Wh � d�+ O(h(M−1)), ∀W(M−1) h ∈V (M−1) h . (21)


### As described in [50], neither the mass matrix nor the degrees of freedom for the conservative vector U(N−1) h are needed to


### be explicitly evaluated for the computation of element’s non-linear residual vector R(N−1) h .


### 3.2. Temporal discretization

To prevent severe time step restrictions due to either explicit [30] or semi-implicit [36] time discretizations, we use a fully-implicit method of lines (MOL) formulation. For all test problems in this study, we use the BDF2 L-stable time integrator and the pth-order explicit singly diagonal implicit Runge-Kutta scheme, ESDIRKp.


### 3.3. Jacobian-Free Newton-Krylov (JFNK) solver

In this section, we briefly review the Jacobian-free Newton-Krylov (JFNK) framework. Once the equations are discretized in space and time, we seek to minimize the residual vector using a globalized line search method.8 Newton’s method is used to compute the step direction by solving the non-linear system of equations


## F(x) = 0, (22)

where F is the nonlinear residual function and x is the solution vector, representing all of the degrees of freedom. Using Newton’s method, we iteratively search for better approximations to the roots of Eq. (22) by solving a sequence of linear problems


## Jkδxk = −F(xk), (23)


## where the Jacobian matrix is defined as J ≡∂F


## ∂x . Once the update vector, δxk, is found, it is used to increment the previous non-linear solution vector


## xk+1 = xk + δxk, (24)

8 In this work, we use either the cubic backtracking, critical point, or secant search line search strategies in PETSc [4].

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 9


### until the Newton convergence criterion is satisfied


## ||F(xk)||2 < tolN||F(x0)||2. (25)

In this paper, all runs converge to a relative Newton tolerance, tolN, of at least 10−5. For the linear solver, we use the Flexible Generalized Minimal Residual method (FGMRES) [59]. FGMRES allows the preconditioner to change between GMRES iterations, an attractive property that we found to be useful when using iterative preconditioners. Since FGMRES does not require individual elements of the Jacobian matrix, only the action of matrix-vector products is needed and an explicit Jacobian matrix does not need to be formed. The action of the Jacobian matrix-vector products is approximated by Fréchet derivatives


# J ⃗κ ≈F(x + h⃗κ) −F(x)


## h , (26)

where h is a small but finite number and ⃗κ is a Krylov vector. Eq. (26) is a first-order Taylor series expansion of the product of a Jacobian and a vector, ⃗κ. An inexact Newton method is used to ensure that the linear system is tightly solved only when the accuracy matters – i.e. near the end of the nonlinear iterations. With this approach, the convergence criteria of the linear residual is proportional to the non-linear residual,


# ||Jkδxk + F(xk)||2 < ηk||F(xk)||2, (27)


### with


# ηk = γN


## � ∥F(xk)∥2 ∥F(xk−1)∥2


## �α


## (28)

where α = 1.26 and γN = 0.9, as used in [21]. Our (inexact) JFNK solver is implemented within PETSc, a high-performance suite of non-linear and linear solvers developed at the Argonne National Laboratory [4].


### 3.4. Preconditioning

Although the governing equations are formulated to conserve mass, momentum, and total energy, we choose to solve for the primitive set of variables, [PvT ] (pressure, velocity, temperature), since it is a better conditioned set of variables for low-Mach flow [11,48]. Introducing the transformation


## δU = ∂U


## ∂WδW, (29)


# where U = (ρ, ρv, E) is the vector of conservative variables and W = (P, v, T) is the vector of primitive variables, the linear system in Eq. (23) can be transformed to


## ∂F(x)


## ∂U ∂U ∂WδW = −F(x). (30)

It is important to emphasize that the change of variables does not affect conservation, since the residual function is still written to satisfy the underlying conservation laws, ensuring that mass, momentum, and energy are conserved to the chosen non-linear tolerance level. The right-preconditioned form of the system is


## JM−1Mδx = −F(x), (31)


### where M is the preconditioning matrix.9 Choosing M as the approximate (finite-differenced) Jacobian matrix, the degrees of freedom can be ordered by primitive variable fields in a 3 × 3 block matrix,


## ⎡


## ⎣ Mvv MvP MvT MPv MP P MP T MT v MT P MT T


## ⎤


## ⎦


## ⎡


## ⎣ xv xP xT


## ⎤


## ⎦=


## ⎡


## ⎣ bv bP bT


## ⎤


## ⎦, (32)

where b is an incoming Krylov vector and x is the outgoing Krylov (solution) vector. Each element of M in Eq. (32) is a matrix of size nelems ×neqns ×nDoFs/elem, corresponding to degrees of freedom for a primitive variable field (velocity, pressure, and temperature).

9 Although no matrix needs to be explicitly formed for the Jacobian-free Newton-Krylov (JFNK) solver, we choose to explicitly form the matrix, M, for preconditioning purposes. It is worth mentioning that there are efficiency tradeoffs on how tightly to solve the preconditioning system, Mx = b. If the preconditioning system is only solved to a loose tolerance, more outer linear iterations is generally needed to converge the linear and non-linear systems to their desired tolerance, and vice versa.

10 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


### 3.4.1. High-Order rDG

High-order rDG schemes, P1 P3 or P2 P3, have additional degrees of freedom per equation per element (as listed in Table 2). Thus each element of M in Eq. (32) can be arranged as a block matrix, e.g. the P2 P3 scheme in 2D is a 6 × 6 block system, corresponding to one cell-averaged value + two slopes + three curvatures. In 3D, these block matrices become 10 × 10 block systems. For a generic P N P M scheme, we write


## Mαβ =


## ⎛


## ⎜⎝


## Mαoβ0 ... Mα0βN ... ... ... MαNβ0 ... MαNβN


## ⎞


## ⎟⎠, (33)


# where α, β = P, v, and T .


### 3.5. Preconditioning strategies

Our new preconditioning strategy is described in Section 3.5.4. In addition, we outline common preconditioners for JFNK solvers used in this study, which are all applied to an approximate Jacobian matrix, in Sections 3.5.1-3.5.3. A more extensive review of preconditioners for JFNK methods can be found in [32].


### 3.5.1. Element-block SOR preconditioner

An element-block SOR method is one of the simplest preconditioners for the outer Newton-FGMRES solver. The degrees of freedom in each element-block are tightly coupled for all three primitive variable fields: pressure, velocity, and temperature. In the P0 P1 scheme, element-blocks are 4 × 4 systems in 2D and 5 × 5 systems in 3D. In the P2 P3 scheme, element-blocks are 24 × 24 systems in 2D and 50 × 50 systems in 3D. In each sweep of the SOR method, the (diagonal) element-blocks are LU decomposed. In all runs, we use a maximum of 10 SOR sweeps per outer FGMRES iteration with under-relaxation parameter, ω = 0.4.


### 3.5.2. Monolithic algebraic multigrid preconditioner

The next preconditioner to the outer Newton-FGMRES solver used in this study is the Algebraic Multigrid (AMG) method on the fully-coupled monolithic system. We use HYPRE’s BoomerAMG, a parallel implementation of AMG developed at the Lawrence Livermore National Laboratory. In 2D, we use the default settings for BoomerAMG, which utilizes a symmetric SOR smoother on a single V-cycle with Falgout coarsening. We also find that using additional aggressive coarsening levels is effective for our systems. In 3D, we use a SOR/Jacobi smoother on a single V-cycle with a HMIS-coarsening strategy [25]. These settings were found to be most efficient for our problems.


### 3.5.3. Primitive variable block Gauss-Seidel


### As the third preconditioner, we consider a primitive variable block Gauss-Seidel method, which utilizes the lower triangular portion of Eq. (32),


## ⎡


## ⎣ MP P 0 0 MvP Mvv 0 MT P MT v MT T


## ⎤


## ⎦


## ⎡


## ⎣ xP xv xT


## ⎤


## ⎦=


## ⎡


## ⎣ bP bv bT


## ⎤


## ⎦. (34)


### The outgoing preconditioned Krylov vector is then solved for with one iteration of a block Gauss-Seidel method,


## xP = M−1 P P bP (35)


## xV = M−1 vv (bv −MvP xP) (36)


## xT = M−1 T T (bT −MT P xP −MT vxv). (37)

To approximate the action of the inverse for the three block solves in Eq.’s (35-37), we use BoomerAMG as a preconditioner to FGMRES. A similar primitive variable block Gauss-Seidel strategy was developed and implemented in [49,54].


### 3.5.4. Primitive variable block Schur complement (vP-vT )

Using the full primitive variable block matrix in Eq. (32), we propose the following Schur complement-based preconditioner. The MP T and MT P blocks can be neglected without a loss in robustness, since the pressure-temperature coupling is weak for our governing equations. Thus, the 3 × 3 block system reduces to


## ⎡


## ⎣ Mvv MvP MvT MPv MP P 0 MT V 0 MT T


## ⎤


## ⎦


## ⎡


## ⎣ xv xP xT


## ⎤


## ⎦=


## ⎡


## ⎣ bv bP bT


## ⎤


## ⎦. (38)


## The block LU decomposition of this 3 × 3 system is

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 11


## ⎡


## ⎣ Mvv MvP MvT MPv MP P 0 MT V 0 MT T


## ⎤


## ⎦=


## ⎡


## ⎣ I 0 0 MPvM−1 vv I 0 MT vM−1 vv −MT vM−1 vv MvP S−1 vP I


## ⎤


## ⎦


## ⎡


## ⎣ Mvv MvP MvT 0 SvP −MPvM−1 vv MvT 0 0 Z


## ⎤


## ⎦, (39)


### where SvP is the velocity-pressure Schur complement


## SvP = MP P −MP V M−1 vv MvP, (40)


### and Z is a nested Schur complement matrix, coupling the velocity, pressure, and temperature fields


## Z = MT T −MT vM−1 vv (I + MvP S−1 vP MPvM−1 vv )MvT . (41)

Instead of solving the fully-coupled 3 × 3 nested Schur complement system, we reduce the 3 × 3 system to a sequence of reduced 2 × 2 block systems, i.e. the velocity-pressure (vP) and velocity-temperature (vT ) sub-systems. The block LU decomposition of the 2 × 2 vP system is given by


## � Mvv MvP MPv MP P


## � = � Mvv 0 MPv SvP


## �� I M−1 vv MvP 0 I


## � . (42)


## Similarly, the block LU decomposition of the 2 × 2 vT system is given by,


## � Mvv MvT MT v MT T


## � = � Mvv 0 MT v SvT


## �� I M−1 vv MvT 0 I


## � , (43)


### where the vP and vT Schur complement matrices are defined as


## SvP = MP P −MPvM−1 vv MvP (44)


## SvT = MT T −MT vM−1 vv MvT . (45)


### Using these primitive variable block factorizations, the full vP-vT Schur complement preconditioning matrix is


## MvP−vT =


## ⎡


## ⎣ Mvv MvP MvT MPv MP P MPvM−1 vv MvT MT v MT vM−1 vv MvP MT T


## ⎤


## ⎦=


## ⎡


## ⎣ Mvv 0 0 MPv SvP 0 MT v 0 SvT


## ⎤


## ⎦


## ⎡


## ⎣ I M−1 vv MvP M−1 vv MvT 0 I 0 0 0 I


## ⎤


## ⎦. (46)


### With this, the preconditioning strategy proceeds in two steps. First, the intermediate velocity, pressure, and temperature solutions are solved for using forward substitution,


## x∗ v = M−1 vv bvv (47)


## xP = S−1 vP (bP −MPvx∗ v) (48)


## xT = S−1 vT (bT −MT vx∗ v). (49)


### The velocity solution is then corrected using backward substitution,


## xv = M−1 vv (bv −MvP xP −MvT xT ). (50)

Note that this predictor-corrector strategy is analogous to Chorin’s projection algorithm and operator-splitting based methods, such as SIMPLE/Uzawa, in incompressible flow solvers, [12,55]. It is instructive to note that a similar preconditioning matrix (Eq. 46) was developed in [15] for incompressible MHD physics. To our knowledge, our work is the first time this type of approximate block factorization preconditioner has been applied to compressible flow with phase change.

Explicitly forming the exact Schur complement matrices in Eq.’s (44-45) is prohibitively expensive, since the inverse of Mvv would be required. Instead, we implement three options for approximating the Schur complement systems.


## #1 The Schur complement matrix is approximated by using a diagonal approximation, Dvv = diag(Mvv). In this case, SvP and SvT in Eq.’s (44-45) are now replaced by ˜SvP and ˜SvT , respectively,


## ˜SvP = MP P −MPvD−1 vv MvP (51)


## ˜SvT = MT T −MT vD−1 vv MvT , (52)

where ˜SvP and ˜SvT are explicitly formed. #2 The exact Schur complement systems are solved for in a matrix-free fashion, without explicitly forming the Schur complement matrices. In this option, M−1 VV is iteratively approximated by AMG V-cycles.

12 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847

#3 The exact Schur complement systems are solved for in an iterative fashion, where M−1 VV is iteratively approximated by AMG V-cycles, as in option (#2). This system is now preconditioned with the explicitly formed Schur complement matrices from option (#1), ˜SvP and ˜SvT , as defined in Eq.’s (51-52). Thus, option (#3) can be viewed as using option (#1) as a preconditioner to option (#2).

Furthermore, the four block systems in Eq.’s (47-50) are solved with iterative methods. In this paper, we test five different block smoothers: (a) 1 V-cycle of AMG, (b) GMRES, (c) SOR-FGMRES, (d) LU-GMRES, and (e) AMG-FGMRES. With the exception of option (a), all other smoothers solve the block systems to a relative tolerance of at least 10−2.


### 3.6. Summary of preconditioning strategies

To summarize, the preconditioning strategy of the outer Newton-FGMRES solver is comprised of three levels. At the first level, we test different strategies on the approximate Jacobian matrix: monolithic AMG, element-block SOR, primitive variable block Gauss-Seidel, vP −vT Schur complement, and the LU factorization preconditioner. At the second level, we test the three strategies for approximating the Schur complement, which only applies to our vP −vT Schur complement preconditioner. Lastly, at the third level, we vary the choice of block smoothers in options (a)-(e), which also only apply to our vP −vT Schur complement preconditioner.

For example, AMG as the block smoother with the Schur complement approximation in strategy #1 is denoted as AMG (#1). Similarly, AMG as a preconditioner to FGMRES with the Schur complement approximation in strategy #3 is denoted as AMG-FGMRES (#3). We compare the performance of the Schur complement approximations and block smoothers in Sections 4.1.4, 4.2.2, and 4.4.3.


### 3.7. Additional details on the Newton-Krylov solver


### 3.7.1. Jacobian matrix assembly

As mentioned in Section 3.4, we assemble a global (approximate) Jacobian matrix via finite differences, which is needed for preconditioning purposes. The first-order and second-order finite difference formulas are given by,


## Jij ≈F j(xi + eihi) −F j(xi)


## hi or Jij ≈F j(xi + eihi) −F j(xi −eihi)


## 2hi , (53)


### respectively, where hi is computed by,


## hi = � erelxi, if |xi| > xmin erelxminsign(xi), otherwise. (54)

We set erel = 10−8 and xmin = 10−7, which is found to be accurate when the governing equations, Eq. (1), are properly non-dimensionalized. Here, we implemented two strategies for assembling the Jacobian via finite differences. The first strategy uses (parallel) graph coloring algorithms [14] to reduce the number of residual evaluations required to assemble the global Jacobian. From our numerical experiments, we find that the coloring algorithms (implemented in PETSc) result in an excessive number of residual evaluations, which is further exacerbated for high-order schemes or in 3D. Instead, we choose to form the global Jacobian by assembling element Jacobian matrices with a local solution perturbation, using either the first-order or second-order finite difference formula in Eq. (53). With this strategy, the approximate Jacobian is more accurate and the required number of residual evaluations is significantly less than with the graph coloring algorithms. Furthermore, these computations might be substantially accelerated on graphics processing units (GPU’s), since Jacobian matrix evaluations are an inherently local computation.


### 3.7.2. Jacobian re-evaluation

Since re-assembling the Jacobian matrix every Newton iteration is prohibitively expensive, we lag (freeze) the Jacobian over several Newton iterations. When the Jacobian is no longer a good preconditioning matrix, i.e. the number of outer FGMRES iterations exceeds a pre-defined threshold (typically 20-50 iterations) within a given Newton iteration, we activate re-evaluation of the Jacobian. We note that the approximate Jacobian matrix is only used for preconditioning purposes and the (implicit) JFNK Jacobian is always up-to-date.


### 4. Results

Numerical results are shown for four test problems of increasing complexity: low-Mach lid-driven cavity flow, RayleighBénard melt convection, compressible internally heated convection, and 3D laser-induced melt pool flow. Our rationale for choosing these test problems is the following. The low-Mach lid-driven cavity flow problem exhibits the numerical challenges associated with stepping over acoustic and advection time-scales. In the Rayleigh-Bénard melt convection problem, the challenge is to step over both the acoustic time-scale as well as viscous time-scale, associated with the introduction of a velocity suppression model in solid-liquid phase change. In the internally heated convection problem, a γ -gas equation of

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 13

state is used, introducing strong compressibility effects at very low-Mach number, since the speed of sound is not artificially reduced. Finally, the 3D laser-induced melt pool flow problem introduces challenging additional boundary conditions at the surface of the melt pool (Marangoni traction, radiation heat transfer, evaporation, and convection), which further increases the non-linearity and condition number of the resulting system.

All simulations solve the equations in non-dimensional form. Results for the internally heated convection and 3D laserinduced melt pool flow problem, however, are shown in dimensional units, as needed by the application of interest. Performance of the preconditioners are compared by counting outer linear (FGMRES) iterations and normalized CPU time per time step. The ESDIRK3 time integrator was used in Section 4.1.2, while the BDF2 time integrator was used for all other test cases. Furthermore, all runs were performed on the RZTopaz linux cluster at the Lawrence Livermore National Laboratory. The machine has 36 Intel Xeon E5-2695 processors (2.1 GHz) and 128 GB of memory per node.


### 4.1. Low-Mach lid-driven cavity flow

The Lid-Driven Cavity Flow (LDC) problem is a standard test problem of nearly incompressible fluid dynamics. Code verification for a low-Reynolds number case of Re = 400 was conducted in [51], and excellent agreement with results from [26] was shown. Snapshots of the flow evolution for the unsteady high-Reynolds number LDC flow problem are displayed in Fig. 2. The solution is unsteady and several secondary eddies are observed to be dynamically forming and decaying in the top and bottom corners. For this high-Reynolds number case, the large central vortex is unstable and gyrates around the center of the domain. It is worth noting that the advection CFL number does not define the dynamic time-scale of this problem. Instead, the dynamic time-scale corresponds to the life-time of the secondary eddies and therefore simulations can significantly step over advection CFL numbers without a loss in accuracy [52].

For performance analysis, we study the Re = 1, 000 and Pr = 0.7 case. The initial temperature of the entire domain is set to a constant temperature, T = 1, and all four walls have isothermal boundary conditions set to T = 1. The top wall is moving to the right with a prescribed velocity, vx = 1, and all other walls enforce a no-slip condition. In all of the runs, performance assessment started at t ≈1 and the number of outer FGMRES iterations and CPU time were counted per time step, averaged over 50 time steps. In each case, 60,000 DoFs per processor was used. The Mach number varied from 10−2


### to 10−4 and the P0 P1 discretization scheme was utilized.


### 4.1.1. Time step study

In this study, we analyze the effect of the time step on the performance of the tested preconditioners as shown in Fig. 3. In each run, we increase the time step by a factor of 10, which proportionally increases all of the CFL and Fourier numbers, as shown in Table 4. We used 18 processors for a 512 × 512 mesh resolution. The Mach number was set to 10−2.

We first observe that the AMG preconditioner on the fully-coupled monolithic system is not robust and was unable to converge problems with acoustic CFL numbers ≥10. This result is not surprising, since the high CFL numbers create a strong global stiffness, leading to a non-diagonally dominant system [10]. The primitive variable block Gauss-Seidel preconditioner was more effective in capturing the global coupling and as a result was very efficient when the acoustic CFL number was ≤100, but was unable to converge problems with larger acoustic CFL numbers. The element-block SOR preconditioner was more robust and was able to converge all of the runs, but the performance severely degraded as a function of time step, requiring hundreds of outer FGMRES iterations per time step in the most ill-conditioned case. The only preconditioners with a constant number of outer FGMRES iterations (independent of the time step) was our vP-vT Schur complement and the LU factorization, due to the robust coupling of the velocity-pressure system. LU factorization is of course non-scalable, in difference to the vP-vT Schur complement preconditioner developed in this work. We observe that the vP-vT Schur complement preconditioner has a slight increase in CPU time as a function of time step, due to the system becoming less diagonally dominant, requiring more inner block iterations.


### 4.1.2. Mach number study

Next, we conduct a Mach number study to analyze the performance of the tested preconditioners in the limit of low Mach number, as shown in Fig. 4. In each run, we increase the sound speed by a factor of 10, which proportionally increases the acoustic CFL number and decreases the Mach number for each run. We used 18 processors with a ESDIRK3 time integrator on a 512 × 512 mesh resolution.

We first observe that the primitive variable block Gauss-Seidel and the element-block SOR preconditioners are not robust and were unable to converge for Mach numbers lower than 10−3 and 10−4, respectively. The only preconditioners that were able to converge the Mach number as low as 10−6 was our vP-vT Schur complement and the LU factorization, due to the robust coupling of the velocity-pressure system. As discussed previously, LU factorization is of course non-scalable, in difference to the vP-vT Schur complement preconditioner developed in this work, and is only used as a benchmark. We observe that for the lowest Mach number of 10−6, our vP-vT Schur complement preconditioner has an increase in iterations and CPU time due to the very ill-conditioned linear system, requiring more inner block iterations. It is worth noting that although our vP-vT Schur complement preconditioner works in the intermediate to high Mach number range, it is only cost-effective in the low Mach regime.

14 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Fig. 2. Dynamics of the velocity magnitude and streamlines using the P2 P3 scheme with a 128 × 128 mesh and a Reynolds number of 104.**


> **Table 4 Time step and corresponding CFL and Fourier numbers for lowMach lid-driven cavity flow problem.**


$$
Time step CFLa CFLv Foα Foν
$$


$$
2 × 10−4 10.3 0.10 0.074 0.052 2 × 10−3 103 1.01 0.737 0.524 2 × 10−2 1030 10.1 7.37 5.24 2 × 10−1 10300 101 73.7 52.4
$$

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 15


> **Fig. 3. Time step study of the tested preconditioners for the low-Mach lid-driven cavity flow problem. CPU time is normalized relative to our vP-vT Schur complement preconditioner.**


> **Fig. 4. Mach number study for the lid-driven cavity flow problem. CPU time is normalized relative to our vP-vT Schur complement preconditioner.**


### 4.1.3. Weak scaling study

Next, we conduct a weak scaling (fixed CFL number) study to analyze the algorithmic scalability of the different preconditioners. In all of the runs, the number of degrees of freedom per processor is fixed, to ensure a constant workload per processor. The mesh resolution increased from a 512 × 512 mesh up to a 2048 × 2048 mesh, while the number of processors varied from 18 to 288. The acoustic CFL number was fixed at 103, while all other time-scales were resolved. We note that as the number of degrees of freedom increase in each run, the mesh size decreases, which quadratically increases the Fourier numbers.

As shown in Fig. 5, all of the preconditioners tested here scale well in the weak sense.10 We observe that the elementblock SOR converged with the most iterations and CPU time. The primitive variable block Gauss-Seidel preconditioner required a similar number of outer FGMRES iterations, but converged in less than half the time. The LU factorization and our vP-vT Schur complement preconditioners converged in the least number of iterations, but our vP-vT Schur complement preconditioner had the best performance and is demonstrated to be algorithmically scalable beyond 107 DoFs. A LU factor-

10 Linear iterations are measured over the full time step, which may include several Newton steps. Since the Jacobian matrix is only approximate, a LU factorization preconditioner will in general require more than one GMRES iteration.

16 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Fig. 5. Weak scaling (fixed CFL) study for the low-Mach lid-driven cavity flow problem. CPU time is normalized relative to our vP-vT Schur complement preconditioner.**


> **Fig. 6. Block smoother study for our vP-vT Schur complement preconditioner as a function of time step for the low-Mach lid-driven cavity flow problem. CPU time is normalized relative to the AMG-FGMRES (#3) smoother.**


### ization of the approximate Jacobian was found to competitive up to 106 DoFs, but a LU factorization of the full Jacobian ran out of memory for larger problems.


### 4.1.4. vP-vT Schur complement preconditioner: block smoother study

Finally, in this study, we compare the performance of different block smoothers as a function of time step for our vP-vT Schur complement preconditioner. The three Schur complement approximations and various block smoothers, described in Section 3.5.4, are investigated. The corresponding time-dependent CFL and Fourier numbers are listed in Table 4. The problem setup is identical to the time step study in Section 4.1.1.

In Fig. 6, we observe that as the time step increases, the average number of outer iterations and CPU time per time step increases for all cases, which is expected since the condition number of the underlying systems is a function of the CFL/Fourier numbers. It is found that by using the Schur complement approximation in strategy #3, all combinations of smoothers were robust, since all of the runs were able to converge the most ill-conditioned case. The AMG (#1) and the GMRES (#2), however, were unable to converge when the acoustic CFL number was above 10,000 and when the advection CFL number was higher than 100. The LU-GMRES (#3) smoother converged in the least number of outer iterations, but requires an unscalable LU factorization, and thus is not an option for larger problems, as evident from Section 4.1.3. The

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 17


> **Fig. 7. Snapshot of temperature and streamlines for Rayleigh-Bénard melt convection on a non-uniform, 400 × 800 mesh, with the P2 P3 scheme and a Rayleigh number of 106. Black contour lines represent the solid-liquid interface.**


> **Fig. 8. Mach number study showing the temperature and streamlines for three Mach numbers at steady-state with a Rayleigh number of 104. The flow dynamics is indistinguishable for all cases.**

AMG (#1) is observed to be the most efficient smoother for problems with acoustic CFL numbers ≤1,000, while the AMG-FGMRES (#3) smoother is found to be the most computationally efficient combination in general.


### 4.2. Rayleigh-Bénard melt convection

In this test problem, we simulate Rayleigh-Bénard natural convection with melting. To demonstrate the preconditioned solver can converge melt convection problems on a non-uniform mesh, a wedged-shape domain is chosen. The results are presented in dimensionless units. The material is initially solid and the temperature of the entire domain is set to a constant temperature of T = 1. The top and bottom walls have isothermal boundary conditions. The left and right walls have adiabatic (zero heat flux) temperature boundary conditions and all four walls enforce a no-slip condition. The solidus and liquidus transition temperatures were set to T S = 1.4 and T L = 1.6, respectively. In all runs, the Prandtl number was 0.1, the Rayleigh number was 106, the Stefan number was 2, and the Mach number was 10−3 at the fully-developed state. Thermal buoyancy effects are modeled with the Boussinesq approximation. The P0 P1 discretization was used in Sections 4.2.1 and 4.2.2, while the P1 P3 and P2 P3 discretizations were also tested in Section 4.2.3.

A snapshot of the temperature and streamlines is shown in Fig. 7. The bottom wall is quickly ramped to a higher temperature of T = 2, inducing melt convection from the bottom. Heating from the bottom produces the well known Rayleigh-Bénard natural convection, due to thermal buoyancy and unstable stratification. For the present Rayleigh number of 104, a Rayleigh-Bénard instability sets in and unsteady convection cells develop. The problem remains unsteady, since the eddies constantly break-apart and reform with time.

Since the dynamical time-scale of melt convection are much larger than the acoustic time-scale, we choose to step over the acoustic CFL number, as in the low-Mach lid-driven cavity flow problem. In Fig. 8 and 9, we conduct a Mach number

18 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Fig. 9. Temperature and velocity magnitude values measured at steady-state on the X-axis, along Y = −0.2, of Fig. 8. The flow dynamics is indistinguishable for all three Mach numbers.**


> **Table 5 Mach number study showing the number of outer FGMRES iterations per time step and CPU time per time step, averaged over 20 cycles.**

M FGMRES Its. CPU time (s)


$$
10−2 11 6 10−3 13 15 10−4 13 24 10−5 25 63 10−6 N/A N/A
$$

study, varying the Mach number from 10−1 to 10−3 by increasing the sound speed in the EoS, as described in Section 2.3.1. The qualitative and quantitative results demonstrate that the flow dynamics is independent of the Mach number, below 10−1. As shown in Table 5, as the Mach number decreases, the solver struggles and requires more outer FGMRES iterations and CPU time to converge, since larger sound speeds increase the condition number of the underlying matrix system. We note that the solver was unable to converge below a Mach number of 10−5 for this problem.


### 4.2.1. Time step study

In this study, similar to Section 4.1.1, we analyze the effect of the time step on the performance of the tested preconditioners. In each run, we increase the time step by a factor of 10, which proportionally increases all of the CFL and Fourier numbers. We used 36 processors for a 400 × 800 mesh resolution. The performance assessment started at t ≈1 and the number of outer FGMRES iterations and CPU time were counted per time step, averaged over 10 time steps. In each case, domain partitioning corresponded to 32,000 DoFs per processor. The viscosity ratio between the solid and liquid phase was 1,000:1.

The results are shown in Fig. 10. We observe that for small time steps, all of the preconditioned solvers require a comparable number of outer FGMRES iterations and CPU time to converge. As we increase the time step from 10−6 to 10−3 (CFLa from 0.6 to 600 and Foν from 0.27 to 270), the performance of the element-block SOR and the primitive variable Gauss-Seidel preconditioners substantially degrades, as expected. Both the non-scalable LU factorization and our vP-vT Schur complement preconditioner are competitive in both outer FGMRES iterations and CPU time and are relatively insensitive to the time step.


### 4.2.2. vP-vT Schur complement preconditioner: block smoother study

Next, in this study, we compare the performance of the different block smoothers in our vP-vT Schur complement preconditioner as a function of the viscosity ratio. Similar to the study in Section 4.1.4, we test three Schur complement approximations and various block smoothers, as described in Section 3.5.4. In each run, we increase the viscosity ratio by a factor of 10 (from 10:1 up to 1,000:1), causing the viscous Fourier number to increase from 10 to 103. The acoustic CFL number was 10 and all other time-scales were resolved. For an ideal preconditioner, the outer FGMRES iteration should be independent of the viscosity ratio (plots should have nearly flat lines). The performance assessment started at t ≈1 and the number of outer FGMRES iterations and CPU time were counted per time step, averaged over 10 time steps.

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 19


> **Fig. 10. Time step study of the tested preconditioners for the Rayleigh-Bénard melt convection problem. CPU time is normalized relative to our vP-vT Schur complement preconditioner.**


> **Fig. 11. Block smoother study for our vP-vT Schur complement preconditioner as a function of viscosity ratio for the Rayleigh-Bénard melt convection problem. CPU time is normalized relative to the AMG-FGMRES (#3) smoother.**

In Fig. 11, we observe that the AMG-FGMRES (#1) smoother results in a large number of outer FGMRES for all viscosity ratios. This is a consequence of diag(Mvv) in strategy (#1) no longer being a good approximation of the velocity system. More effective preconditioners are the GMRES (#2), the LU-GMRES (#3), and the AMG-FGMRES (#3) strategies. In all of these cases, the full Mvv is used in the pressure-velocity Schur complement system, resulting in a nearly constant number of outer FGMRES and CPU time per time step. The performance of the SOR-FGMRES (#3) smoother has a viscosity ratio dependence, due to the fact that SOR is not a robust smoother, requiring a large number of outer iterations and CPU time per time step to converge. The AMG-FGMRES (#3) strategy is found to be the most computationally efficient option, which demonstrates that smoothing by AMG-FGMRES is robust and that the Schur complement approximation strategy (#3) is an effective approximation.


## 4.2.3. Weak Scaling for vP −vT Schur complement preconditioner: high-order rDG

Finally, in Fig. 12, we compare the performance of three rDG schemes by conducting a weak scaling (fixed time step) study with the modal Taylor basis functions. The total number of degrees of freedom that are solved for is kept constant between the schemes. Table 6 shows the range of mesh sizes and acoustic CFL numbers for the three different rDG schemes. The number of degrees of freedom for each run ranges from 1.16 to 18.5 million DoFs. The number of degrees of freedom per processor is fixed at 8,022, 2,664, and 1,334 for the P0 P1, P1 P3, and P2 P3 schemes, respectively. For this study, we

20 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Table 6 Mesh size and corresponding CFL numbers for three rDG schemes.**

Scheme Smallest Mesh Largest Mesh CFLa

P0 P1 380 × 760 1520 × 3040 597–2390 P1 P3 219 × 438 876 × 1752 344–1380 P2 P3 155 × 310 620 × 1240 244–975 # of DoFs 1.2 million 18.5 million


> **Fig. 12. Weak scaling (fixed time step) study comparing P0 P1, P1 P3, and P2 P3 schemes with the same solution vector size for the Rayleigh-Bénard melt convection problem. CPU time is normalized relative to the rDG P0P1 scheme.**

start the performance assessment at t ≈0 and count the number of outer FGMRES iterations and CPU time per time step, averaged over 30 time steps. The number of processors varied from 36 to 576 and the AMG-FGMRES (#1) smoother was used for all three schemes.

In Fig. 12, we observe that good algorithmic scalability is achieved for both P0 P1 and P1 P3 schemes, since the number of outer iterations and CPU time only moderately increase with problem size. When our preconditioner is applied to the P2 P3 scheme, many more inner block iterations are required to converge the system, resulting in over 10 times more CPU time than the P0 P1 or the P1 P3 scheme, for the same number of degrees of freedom. Furthermore, the number of outer iterations required to converge the P2 P3 system grows dramatically for problems larger than 107 degrees of freedom. This is due to the increased condition number of the P2 P3 matrix system, requiring many more inner block iterations to converge to the same tolerance. We found it to be more computationally efficient to solve the P2 P3 systems to a loose relative tolerance of 10−2.


### 4.3. Compressible internally heated convection

This test problem is related to the slow cook-off process of energetic materials. When energetic material in a confined vessel begins to heat up, the material starts to melt and mix due to natural convection. Because of chemical reactions, the liquid material decomposes into product gases, forming a headspace (ullage) at the top of the vessel. Exothermic reactions continue to occur in the headspace, and are temperature-dependent. In this numerical example, we simulate convection of air in the headspace with an internally heated temperature-dependent source term. It is important to represent the true thermodynamic state in the system, and it is not appropriate to artificially reduce the sound speed, as done in previous examples. We use the γ -gas EoS to allow the fluid to compress due to heating, while the buoyancy effects are now fully captured without a Boussinesq model. The results will be presented in dimensional units.

The headspace domain is two inches tall by four inches wide. The initial temperature of the entire domain is set to a constant temperature, T = 293K , and the top and bottom walls have isothermal boundary conditions while the left and right walls have adiabatic (zero heat flux) temperature boundary conditions. Initially, the bottom wall is set to a higher temperature of T = 303K , while the top wall is set to T = 293K . All four walls enforce a no-slip condition. The Prandtl number was 0.7, the internal Rayleigh number was 107, and the Mach number varied from 10−7 at the start to 10−4 at the fully-developed state. Because we use the γ -gas EoS in this example, no artificial compressibility assumption (reduction of speed of sound) is made.

Starting at time, t = 0, the bottom and top wall temperature increases with a rate of 10 Kelvin per second. As in Section 4.2, there is a heating from the bottom, generating natural convection loops due to density-driven thermal buoyancy.

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 21


> **Fig. 13. The temperature (Kelvin) and density (kg/m3) fields are displayed for two snapshots in time (secs) on the left and right, respectively, for compressible internally heated convection. For visualization, the P0P1 scheme with a 400 × 200 mesh was used.**


> **Table 7 Mesh size and corresponding CFL numbers for three rDG schemes.**

Scheme Smallest mesh Largest mesh CFLa

P0 P1 380 × 760 1520 × 3040 3560 P1 P3 219 × 438 876 × 1752 2050 P2 P3 155 × 310 620 × 1240 1450 # of DoFs 1.2 million 18.5 million

Due to the large Rayleigh number, the problem remains unsteady, with large plumes constantly breaking-apart and reforming with time. Snapshots of the temperature and density fields are shown in Fig. 13. It is worth noting that due to the very small Mach numbers during the initial transient, the matrix systems are extremely ill-conditioned, presenting significant challenges for the preconditioned Newton-Krylov solver. Performance of the P0 P1, P1 P3, and P2 P3 schemes are compared in Section 4.3.1.


## 4.3.1. Weak Scaling for vP −vT Schur complement preconditioner: high-order rDG

In Fig. 14, we compare the performance of three rDG schemes by conducting a weak scaling (fixed CFL) study with the modal Taylor basis functions. The total number of degrees of freedom that are solved for is kept constant between the schemes. Table 7 shows the range of mesh sizes and acoustic CFL numbers for the three different rDG schemes. The number of degrees of freedom for each run ranges from 1.2 million to 18.5 million DoFs. The number of degrees of freedom per processor is fixed at 32,000, 10,656, and 5,336 for the P0 P1, P1 P3, and P2 P3 schemes, respectively. For this study, we start the performance assessment at t ≈1 and count the number of outer FGMRES iterations and CPU time per time step, averaged over 20 time steps. The number of processors varied from 36 to 576 and the AMG-FGMRES (#1) smoother was used for all three schemes.

In Fig. 14, we observe that good algorithmic scalability is achieved when applying our preconditioner to the P0 P1 and P1 P3 schemes, since the number of outer iterations and CPU time only moderately increase with problem size. Many more inner block iterations are required to converge the P2P3 systems, due to the increased condition number, resulting in over 5 times more CPU time than the P1P3 scheme, for the same number of degrees of freedom. Similar to the weak scaling study in Section 4.2.3, the number of outer iterations required to converge the P2 P3 scheme grows dramatically for problems larger than 107 degrees of freedom.

It is instructive to note that for the same number of degrees of freedom, high-order schemes result in significantly more accurate solutions, which has been demonstrated in previous studies (see in [53]). Since the number of degrees of freedom were kept constant between each of the schemes, it is not unexpected that high-order schemes converge with higher CPU time, since it is well known that the cost per DoF increases with high-order. At the same time, high-order schemes are more FLOP-intensive relative to memory accesses, which is more favorable for future heterogeneous architectures. Future work will investigate the efficiency of preconditioning using the solution accuracy as the figure of merit, which is more appropriate than just comparing discretization schemes with the same solution vector size, as used in Figs. 12 and 14.

22 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Fig. 14. Weak scaling (fixed CFL) study comparing P0 P1, P1 P3, and P2 P3 schemes with the same solution vector size for the compressible internally heated convection problem. CPU time is normalized relative to the rDG P0P1 scheme.**


> **Fig. 15. The computational domain for 3D laser-induced melt pool flow. 10 million elements (50 million DoFs). Processor domain partitioning with 612 cores.**


### 4.4. 3D laser-induced melt pool flow

In our final and most challenging numerical example, we consider a 3D laser-induced melt pool flow physics problem. As shown in Fig. 15, the domain is 3.4 mm × 3.4 mm × 1 mm. The material is initially solid metal at room temperature, T0 = 300 K. All walls have adiabatic (zero heat flux) temperature boundary conditions and enforce a no-slip condition. A 100 W laser of radius 100 μm is moved across the surface of the plate at a velocity of 2 m/s in a “Z”-shaped pattern. As the laser moves across the plate, a transient melt pool is formed, which subsequently re-solidifies before the flow is able to fully develop. At the surface of the melt pool, there is a Marangoni traction force, which strongly couples the velocity and temperature fields. Due to the presence of surfactants, the flow is driven by reverse Marangoni convection, resulting in a hydrodynamic instability with complex melt pool dynamics [50]. In addition to the rapid laser heating and Marangoni convection, radiation, evaporation, and convection heat transfer also occur at the surface of the melt pool, as described in Section 2.5. An in-depth analysis of the physics for the laser-induced melt pool flow problem is beyond the scope of this paper and more details can be found in [50].

All computations are done with the P0 P1 scheme. The solidus and liquidus transition temperatures were T S = 1, 700 K and T L = 1, 800 K, respectively. To inhibit the motion in the solid phase, a viscosity ratio of 1,000:1 was used. The Prandtl number was 0.1, the Rayleigh number was 10−2, the Stefan number was 2, the Marangoni number was 1,000, and the Mach number was 10−3.


### 4.4.1. vP-vT Schur complement preconditioner: Marangoni convection

To demonstrate the numerical challenges associated with Marangoni convection physics, we performed two runs, one with Marangoni convection and the other without. In both runs, all of the heat transfer mechanisms at the melt pool

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 23


> **Fig. 16. Snapshots of the free surface temperature and melting front for laser-induced melting. The laser delivers 100 W of power, has an effective radius of 100 μm, and is moving at 2 m/s over the test section in a “Z”-shaped pattern.**


> **Table 8 Marangoni convection study.**

No Marangoni convection Marangoni convection

FGMRES Iterations 22 120 CPU time (s) 23 216


### surface are included (laser absorption, radiation, evaporation, and convection). In the second run, in addition to the heat fluxes at the melt pool surface, Marangoni convection physics is activated.

In this study, we used our vP-vT Schur complement preconditioner and counted the FGMRES iterations and CPU time per time-step, averaged over 100 time-steps, starting from t = 0. A mesh resolution of 1 million elements (5 million DoFs) was used. The viscous Fourier number was 436, the acoustic CFL number was 35, and all other heat flux time-scales at the surface of the melt pool were resolved.

The performance results are summarized in Table 8. We observe that without Marangoni convection, the number of outer FGMRES iterations and CPU time required to converge is moderate, since we choose to resolve these heat flux time-scales for time-accurate solutions. When Marangoni convection is present, however, there is a strong non-linear coupling of the velocity field to the temperature gradient, requiring 5 times as many outer FGMRES iterations and nearly 10 times as much CPU time to converge (Fig. 16).


### 4.4.2. Weak scaling study

In this study, we analyze the performance of different preconditioners as the total problem size increases with Marangoni convection. The number of degrees of freedom for each run ranges from 1.5 million to 26 million, while the number of processors varied from 36 to 576. In all of the runs, the number of degrees of freedom per processor was fixed to 45,000, ensuring a constant workload per processor. For this study, we start the performance assessment at t ≈1 and count the number of outer FGMRES iterations and CPU time per time step, averaged over 20 time steps. The acoustic CFL number increased was 100, and all other time-scales were resolved.

As seen in Fig. 17, our vP-vT Schur complement preconditioner using the AMG-FGMRES (#3) smoother had the best algorithmic scalability and converged in the fastest run-time. The element-block SOR preconditioner required over 3 times as much CPU time to converge compared to our vP-vT Schur complement preconditioner. The primitive variable block GaussSeidel preconditioner was unable to converge in the largest run with 26 million DoFs. We note that the element-block SOR

24 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847


> **Fig. 17. Weak scaling (fixed CFL) comparing different preconditioners for the 3D laser-induced melt pool flow problem. CPU time is normalized relative to our vP-vT Schur complement preconditioner.**


> **Table 9 Strong scaling: normalized speedup factors for three block smoothers. The theoretical speedup factor is 1.0.**

DoFs/Proc AMG (#1) GMRES (#2) AMG-FGMRES (#3) Ghost zones

697,685 – – – 17% 348,840 0.99 N/A 1.02 22% 174,420 0.91 0.97 0.90 31% 87,210 0.82 0.88 0.76 41% 43,605 0.61 0.76 0.55 54%


### and primitive variable block Gauss-Seidel preconditioner were unable to converge for large CFL numbers on this problem, which required us to run this scaling study at a small enough CFL number.


### 4.4.3. vP-vT Schur complement preconditioner: strong scaling study

To demonstrate parallel scalability, we performed a strong scaling study, shown in Table 9 and Fig. 18, testing three different block smoothers for our vP-vT Schur complement preconditioner. In this study, we counted the total run-time over 20 time steps, starting from t = 0. A fine mesh resolution of 10 million elements (50 million DoFs) was used, and the number of processors varied from 72 to 1,152. The viscous Fourier number was 436, the acoustic CFL number was 35, and all other time-scales associated with nonlinear heat fluxes at the surface of the melt pool were resolved. It is worth noting that due to the presence of Marangoni convection, the problem is computationally challenging, requiring over 200 FGMRES iterations (as seen in Table 8) and 6-8 Newton iterations per time step.

In Table 9, we observe that all three block smoothers have excellent parallel scalability down to ∼85,000 DoFs per processor in 3D.11 At less than 85,000 DoFs per processor, the percentage of ghost zones is 54%, and thus the scaling flattens out, as expected, since the communication between processor domains begins to dominate. The GMRES (#2) block smoother ran out of memory for 697,685 DoFs per processor. In Fig. 18, we plot the total run-time for three different block smoothers. The AMG (#1) smoother converged the fastest and was an order of magnitude more efficient than the GMRES (#2) smoother. It is instructive to note that in 2D, we observe ideal scaling above 7,000 DoFs per processor, which is significantly lower than in 3D (85,000 DoFs per processor).


### 5. Conclusions

We have demonstrated that the developed vP-vT Schur complement technique is the most effective preconditioner for converging the ill-conditioned system of equations arising from a Discontinuous Galerkin discretization of the all-speed Navier-Stokes equations with phase change. This approximate block factorization preconditioner is a multigrid block reduction technique, which employs the Schur complement to reduce a fully-coupled 3 × 3 block system to a sequence of two 2 × 2 block systems: velocity-pressure and velocity-temperature.

11 For parallel domain decomposition, we use the ParMETIS library [29].

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 25


> **Fig. 18. Strong scaling of our vP-vT Schur complement preconditioner using different block smoothers for the 3D laser-induced melt pool flow problem.**

In the limit of large acoustic CFL number (corresponding to low-Mach flow), the velocity and pressure fields are strongly coupled, causing the element-based and the physics-based operator-split preconditioners to be ineffective. For large time steps and for problems with melting and solidification, the use of velocity suppression models lead to non-diagonally dominant systems, rendering the diagonal approximations within the approximate Schur complement systems ineffective. To remedy this, the exact Schur complement systems can be approximated with iterative methods, which is found to work very well for all tested cases. Furthermore, in all test problems, the AMG and AMG-FGMRES are found to be the most robust block smoothers in our vP-vT Schur complement preconditioner, enabling the outer Newton-Krylov solver to scale well both algorithmically and in parallel.

Finally, while the majority of test cases were for the 2nd-order accurate P0P1 discretization scheme, our preconditioner was also found to converge time-accurate solutions with the 4th-order rDG discretization schemes, P1P3 and P2P3, without any modifications due to high-order degrees of freedom. Future work will focus on improving performance of the preconditioning for larger block systems, arising from going to high-order and including additional physics, such as material strength and multi-species transport.


### Acknowledgements

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, and funded by the Laboratory Directed Research and Development Program at LLNL under project tracking code 17-ERD-076 and 19-ERD-015. Information management release number LLNL-JRNL745515.


## References

[1] ALE3D Web Page, https://wci .llnl .gov /simulation /computer-codes /ale3d, 2013. [2] J.D. Anderson, J. Wendt, Computational Fluid Dynamics, vol. 206, Springer, 1995. [3] S. Anisimov, K. V, Instabilities in laser-matter interaction, Laser Part. Beams 14 (4) (1996) 797. [4] S. Balay, K. Buschelman, V. Eijkhout, W.D. Gropp, D. Kaushik, M.G. Knepley, L.C. McInnes, B.F. Smith, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 – Revision 2.1.5, Argonne National Laboratory, 2004. [5] G.K. Batchelor, An Introduction to Fluid Dynamics, Cambridge university press, 2000. [6] A. Beccantini, E. Studer, S. Gounand, J.-P. Magnaud, T. Kloczko, C. Corre, S. Kudriakov, Numerical simulations of a transient injection flow at low Mach number regime, Int. J. Numer. Methods Eng. 76 (5) (2008) 662–696. [7] M. Benzi, Preconditioning techniques for large linear systems: a survey, J. Comput. Phys. 182 (2) (2002) 418–477. [8] W. Briley, L. Taylor, D. Whitfield, High-resolution viscous flow simulations at arbitrary Mach number, J. Comput. Phys. 184 (1) (2003) 79–105. [9] P.N. Brown, C.S. Woodward, Preconditioning strategies for fully implicit radiation diffusion with material-energy transfer, SIAM J. Sci. Comput. 23 (2) (2001) 499–516. [10] L. Chacón, A. Stanier, A scalable, fully implicit algorithm for the reduced two-field low-βextended MHD model, J. Comput. Phys. 326 (2016) 763–772. [11] Y.-H. Choi, C.L. Merkle, The application of preconditioning in viscous flows, J. Comput. Phys. 105 (2) (1993) 207–223. [12] A.J. Chorin, A numerical method for solving incompressible viscous flow problems, J. Comput. Phys. 135 (2) (1997) 118–125. [13] A. Cleary, R. Falgout, V. Henson, J. Jones, T. Manteuffel, S. McCormick, G. Miranda, J. Ruge, Robustness and scalability of algebraic multigrid, SIAM J. Sci. Comput. 21 (5) (2000) 1886–1908. [14] T.F. Coleman, J.J. Moré, Estimation of sparse Jacobian matrices and graph coloring problems, SIAM J. Numer. Anal. 20 (1) (1983) 187–209.

26 B. Weston et al. / Journal of Computational Physics 397 (2019) 108847

[15] E.C. Cyr, J.N. Shadid, R.S. Tuminaro, Stabilization and scalable block preconditioning for the Navier–Stokes equations, J. Comput. Phys. 231 (2) (2012) 345–363. [16] E.C. Cyr, J.N. Shadid, R.S. Tuminaro, R.P. Pawlowski, L. Chacón, A new approximate block factorization preconditioner for two-dimensional incompressible (reduced) resistive MHD, SIAM J. Sci. Comput. 35 (3) (2013) B701–B730. [17] I. Danaila, R. Moglan, F. Hecht, S.L. Masson, A Newton method with adaptive finite elements for solving phase-change problems with natural convection, J. Comput. Phys. 274 (2014) 826–840. [18] J.A. Dantzig, Modelling liquid–solid phase changes with melt convection, Int. J. Numer. Methods Eng. 28 (8) (1989) 1769–1785. [19] H. De Sterck, U. Yang, J. Heys, Reducing complexity in parallel algebraic multigrid preconditioners, SIAM J. Matrix Anal. Appl. 27 (4) (2006) 1019–1039. [20] G. Ehlen, A. Ludwig, P.R. Sahm, Simulation of time-dependent pool shape during laser spot welding: transient effects, Metall. Trans. A 34 (12) (2003) 2947–2961. [21] S.C. Eisenstat, H.F. Walker, Choosing the forcing terms in an inexact Newton method, SIAM J. Sci. Comput. 17 (1) (1995) 16–32. [22] H. Elman, V.E. Howle, J. Shadid, R. Shuttleworth, R. Tuminaro, A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations, J. Comput. Phys. 227 (3) (2008) 1790–1808. [23] H.C. Elman, V.E. Howle, J.N. Shadid, R.S. Tuminaro, A parallel block multi-level preconditioner for the 3d incompressible Navier–Stokes equations, J. Comput. Phys. 187 (2) (2003) 504–523. [24] K.J. Evans, D.A. Knoll, M. Pernice, Development of a 2-D algorithm to simulate convection and phase transition efficiently, J. Comput. Phys. 219 (1) (2006) 404–417. [25] R.D. Falgout, U.M. Yang, HYPRE: a library of high performance preconditioners, in: International Conference on Computational Science, Springer, 2002, pp. 632–641. [26] U. Ghia, K. Ghia, C. Shin, High-Re solutions for incompressible flow using the Navier-Stokes equations and a Multigrid method, J. Comput. Phys. 48 (1982) 347–411. [27] H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, Comput. Fluids 28 (1) (1999) 63–86. [28] V. Henson, U. Yang, BoomerAMG: a parallel algebraic multigrid solver and preconditioner, in: Developments and Trends in Iterative Methods for Large Systems of Equations – In Memorium Rudiger Weiss, Appl. Numer. Math. 41 (1) (2002) 155–177. [29] G. Karypis, Metis and parmetis, in: Encyclopedia of Parallel Computing, Springer, 2011, pp. 1117–1124. [30] S.A. Khairallah, A.T. Anderson, Mesoscopic simulation model of selective laser melting of stainless steel powder, J. Mater. Process. Technol. 214 (2014) 2627–2636. [31] S.A. Khairallah, A.T. Anderson, A. Rubenchik, W.E. King, Laser powder-bed fusion additive manufacturing: physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones, Acta Mater. 108 (2016) 36–45. [32] D. Knoll, D. Keyes, Jacobian-free Newton-Krylov methods: a survey of approaches and applications, J. Comput. Phys. 193 (2) (2004) 357–397. [33] D. Knoll, P. McHugh, D. Keyes, Newton-Krylov methods for low-Mach-number compressible combustion, AIAA J. 34 (5) (1996) 961–967. [34] D. Knoll, W. Vanderheyden, V. Mousseau, D. Kothe, On preconditioning Newton–Krylov methods in solidifying flow applications, SIAM J. Sci. Comput. 23 (2) (2001) 381–397. [35] D.A. Knoll, V. Mousseau, L. Chacón, J. Reisner, Jacobian–Free Newton–Krylov methods for the accurate time integration of Stiff wave systems, J. Sci. Comput. 25 (1) (2005) 213–230. [36] D. Korzekwa, Truchas – a multi-physics tool for casting simulation, Int. J. Cast Met. Res. 22 (1–4) (2009) 187–191. [37] M. Lappa, A mathematical and numerical framework for the analysis of compressible thermal convection in gases at very high temperatures, J. Comput. Phys. 313 (2016) 687–712. [38] P. Le Quéré, C. Weisman, H. Paillère, J. Vierendeels, E. Dick, R. Becker, M. Braack, J. Locke, Modelling of natural convection flows with large temperature differences: a benchmark problem for low Mach number solvers, part 1: reference solutions, Modél. Math. Anal. Numér. 39 (03) (2005) 609–616. [39] P. Lin, M. Sala, J. Shadid, R.S. Tuminaro, Performance of fully coupled algebraic multilevel domain decomposition preconditioners for incompressible flow and transport, Int. J. Numer. Methods Eng. 67 (2006) 208–225. [40] M.-S. Liou, A sequel to AUSM: AUSM+, J. Comput. Phys. 129 (2) (1996) 364–382. [41] M.-S. Liou, A sequel to AUSM, part II: AUSM+-up for all speeds, J. Comput. Phys. 214 (1) (2006) 137–170. [42] M.-S. Liou, C.J. Steffen, A new flux splitting scheme, J. Comput. Phys. 107 (1) (1993) 23–39. [43] H. Luo, Y. Xia, S. Li, R. Nourgaliev, C. Cai, A Hermite WENO reconstruction-based discontinuous Galerkin method for the Euler equations on tetrahedral grids, J. Comput. Phys. 231 (2012) 5489–5502. [44] H. Luo, Y. Xia, S. Spiegel, R. Nourgaliev, Z. Jiang, A reconstructed discontinuous Galerkin method based on a hierarchical WENO reconstruction for compressible flows on tetrahedral grids, J. Comput. Phys. 236 (2013) 477–492. [45] Z. Ma, Y. Zhang, Solid velocity correction schemes for a temperature transforming model for convection phase change, Int. J. Numer. Methods Heat Fluid Flow 16 (2) (2006) 204–225. [46] M.J. Martinez, D.K. Gartling, A finite element method for low-speed compressible flows, Comput. Methods Appl. Mech. Eng. 193 (21) (2004) 1959–1979. [47] V. Mousseau, D. Knoll, W. Rider, Physics-based preconditioning and the Newton–Krylov method for non-equilibrium radiation diffusion, J. Comput. Phys. 160 (2) (2000) 743–765. [48] C.-D. Munz, S. Roller, R. Klein, K.J. Geratz, The extension of incompressible flow solvers to the weakly compressible regime, Comput. Fluids 32 (2) (2003) 173–196. [49] C. Newman, D.A. Knoll, Physics-based preconditioners for ocean simulation, SIAM J. Sci. Comput. 35 (5) (2013) S445–S464. [50] R. Nourgaliev, P. Greene, B. Weston, R. Barney, A. Anderson, S. Khairallah, J.-P. Delplanque, High-order fully-implicit solver for all-speed fluid dynamics: AUSM ride from nearly-incompressible variable-density flows to shock dynamics, Int. J. Shock Waves Deton. Explos. (2019). [51] R. Nourgaliev, H. Luo, S. Schofield, T. Dunn, A. Anderson, B. Weston, J.-P. Delplanque, Fully-Implicit Orthogonal Reconstructed Discontinuous PetrovGalerkin Method for Multiphysics Problems, Technical Report LLNL-TR-664250, Lawrence Livermore National Laboratory, Livermore, USA, 2015. [52] R. Nourgaliev, H. Luo, B. Weston, A. Anderson, S. Schofield, T. Dunn, J.-P. Delplanque, Fully-implicit orthogonal reconstructed Discontinuous Galerkin method for fluid dynamics with phase change, J. Comput. Phys. 305 (2016) 964–996. [53] R. Nourgaliev, H. Park, V. Mousseau, Recovery discontinuous Galerkin Jacobian-Free Newton-Krylov method for multiphysics problems, in: Computational Fluid Dynamics Review, 2010. [54] H. Park, R. Nourgaliev, R.C. Martineau, D.A. Knoll, On physics-based preconditioning of the Navier–Stokes equations, J. Comput. Phys. 228 (24) (2009) 9131–9146. [55] S.V. Patankar, D.B. Spalding, A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows, Int. J. Heat Mass Transf. 15 (10) (1972) 1787–1806. [56] M. Pernice, M. Tocci, A multigrid-preconditioned Newton–Krylov method for the incompressible Navier–Stokes equations, SIAM J. Sci. Comput. 23 (2) (2001) 398–418. [57] P.-O. Persson, J. Peraire, Newton-GMRES preconditioning for discontinuous Galerkin discretizations of the Navier-Stokes equations, SIAM J. Sci. Comput. 30 (6) (2008) 2709–2733. [58] Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd edition, SIAM, Philadelphia, 2003.

B. Weston et al. / Journal of Computational Physics 397 (2019) 108847 27

[59] Y. Saad, M. Schultz, GMRES: A Generalized Minimal Residual algorithm for solving linear systems, SIAM J. Sci. Stat. Comput. 7 (1986) 856–869. [60] J. Shadid, R. Tuminaro, K. Devine, G. Hennigan, P. Lin, Performance of fully coupled domain decomposition preconditioners for finite element transport/reaction simulations, J. Comput. Phys. 205 (1) (2005) 24–47. [61] B. Smith, P. Bjorstad, W. Gropp, Domain Decomposition: Parallel Multilevel Methods for Elliptic Partial Differential Equations, Cambridge university press, 2004. [62] M.D. Tidriri, Hybrid Newton-Krylov/domain decomposition methods for compressible flows, in: Proceedings of the Ninth International Conference on Domain Decomposition Methods in Sciences and Engineering, 1998, pp. 532–539. [63] L.N. Trefethen, D. Bau III, Numerical Linear Algebra, vol. 50, SIAM, 1997. [64] R. Tuminaro, C. Tong, J. Shadid, K. Devine, D. Day, On a multilevel preconditioning module for unstructured mesh Krylov solvers: two-level Schwarz, Commun. Numer. Methods Eng. 18 (6) (2002) 383–389. [65] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, J. Comput. Phys. 72 (2) (1987) 277–298. [66] E. Turkel, Preconditioning techniques in computational fluid dynamics, Annu. Rev. Fluid Mech. 31 (1) (1999) 385–416. [67] B. van Leer, P. Roe, W.-T. Lee, Characteristic time-stepping or local preconditioning of the Euler equations, AIAA J. (1991). [68] V. Voller, C. Prakash, A fixed grid numerical modelling methodology for convection-diffusion mushy region phase-change problems, Int. J. Heat Mass Transf. 30 (8) (1987) 1709–1719. [69] J.M. Weiss, W.A. Smith, Preconditioning applied to variable and constant density flows, AIAA J. 33 (11) (1995) 2050–2057. [70] F. White, Viscous Fluid Flow, McGraw-Hill Series in Mechanical Engineering, McGraw-Hill, 1991. [71] Y. Xia, H. Luo, R. Nourgaliev, An implicit hermite WENO reconstruction-based discontinuous Galerkin on tetrahedral grids, Comput. Fluids 98 (2014) 134–151.

