**manuscript No.** (will be inserted by the editor) 

## **Implicit-explicit all-speed schemes for compressible Cahn-Hilliard-Navier-Stokes equations** 

## **Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez** 

## the date of receipt and acceptance should be inserted later 

**Abstract** We propose a second-order implicit-explicit (IMEX) time-stepping scheme for the isentropic, compressible Cahn-Hilliard-Navier-Stokes equations in the low Mach number regime. The method is based on finite differences on staggered grids and is specifically designed to handle the challenges posed by the low Mach number limit, where the system approaches to an incompressible behavior. In this regime, standard explicit schemes suffer from severe time-step restrictions due to fourth-order diffusion terms and the stiffness induced by fast acoustic waves. To overcome this, we employ an IMEX strategy which splits the governing equations into stiff and non-stiff components. The stiff terms, arising from pressure, viscous forces and fourth-order Cahn-Hilliard contributions, are treated implicitly, while the remaining are dealt explicitly. 

**Keywords** Asymptotic preserving, low Mach number, implicit-explicit schemes, incompressible limit, isentropic compressible Cahn-Hilliard-Navier-Stokes 

## **1 Introduction** 

In fluid dynamics, the Cahn-Hilliard (CH) equation describes the phenomenon of phase separation in two-phase systems [9]. It captures the temporal evolution of a mixture of immiscible fluids through the formation of diffuse interfaces between the phases. This diffusive-interface approach is well-suited for explaining topological changes, such as layered structures in sedimenting colloidal suspensions [35]. 

However, the classical CH model does not take into account the fluid dynamics such as motion of the medium, viscosities, or external forces like gravity. To incorporate these effects, the CH system is combined with the Navier-Stokes equations, which describe the conservation of mass and momentum balance of fluids. The resulting system of partial differential equations, known as the Cahn-HilliardNavier-Stokes (CHNS), provides a thermodynamically consistent framework for modeling multiphase flows [1,8,26]. The CHNS model has been applied to a wide variety of problems: separation of immiscible fluids, bubble dynamics or the evolution of interfaces in multiphase systems [22,35]. 

Department of Mathematics, Universitat de Val`encia (Spain); email: mulet@uv.es. 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

2 

In many practical applications, the flow occurs in a low Mach number regime. The Mach number, defined as the ratio between the characteristic fluid velocity and the speed of sound, becomes small in these situations. Under these conditions, the flow behaves as a nearly incompressible [2,3,24], but compressible effects are still present. 

Designing efficient numerical schemes for the compressible CHNS equations for all Mach number regimes presents several challenges. 

First, as the squared low Mach number _δ_ tends to zero, the system becomes increasingly stiff due to the presence of fast acoustic waves with characteristic speeds, namely, **v** _±_[1] _δ[c,]_[where] **[v]**[is][the][velocity][filed][and] _[c]_[represents][the][speed][of] sound. This stiffness is further incremented by the presence of up to fourth-order spatial derivatives in the CHNS equations leading to discrete operators with large eigenvalues. As a consequence, explicit solvers are severely constrained by stability conditions, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0002-05.png)


where _∆t_ and _∆x_ are the time and spatial step size, respectively. In addition, when solving ODE systems of the form _z[′]_ = _f_ ( _z_ ), the appearance of stiff, negativedefinite Jacobians allows implicit methods to take larger time steps, whereas explicit methods are limited to _∆t ≈O_ ( _|λ|[−]_[1] ), with _|λ|_ the largest absolute magnitude of the Jacobian. 

Another challenge arises from the intrinsic structure of the CH equation, whose gradient-flow nature is associated with a non-convex energy functional [26]. To ensure stability, a common strategy is to decompose the energy into the difference of two differentiable convex functions. By treating the contractive part implicitly and the expansive part explicitly, one can construct unconditionally stable IMEX scheme, as shown in [16,36]. 

One way of dealing with these difficulties is to split the pressure into stiff and non-stiff components and treating the stiff part implicitly (for e.g. [11,12]). Therefore, the time step is no longer constrained by the Mach number. Similarly, the stiff fourth-order CH terms are treated implicitly, while the remaining are handled explicitly [30,32]. To maintain stability and preserve symmetry, one can discretize the variables on staggered grids [20,29]. 

As shown in [2], when the Mach number tends to zero, the solutions of the compressible CHNS equations converge to those of its incompressible counterpart. Building on this, numerical methods for the compressible CHNS equations are proposed in [19,30,32], while for the quasi-incompressible and incompressible CHNS models are presented in [10,18,21,25]. Numerical schemes for the low Mach number of the compressible Euler equations are discussed in [11,12,17,33]. 

The goal of this work is to develop a second-order IMEX Runge-Kutta scheme for the isentropic, compressible CHNS equations in low Machfor all Mach number regimes. The method, based on finite differences on staggered grids, is designed to be an Asymptotic Preserving (AP) scheme [12], meaning that its stability and accuracy are independent of the Mach number and that it correctly captures the incompressible limit as _δ →_ 0. 

The outline of the current work is organized as follows. In Section 2 we introduce the isentropic, compressible CHNS equations in a low Mach number regime. Section 3 presents a partitioned IMEX Runge-Kutta scheme for the two- 

Title Suppressed Due to Excessive Length 

3 

dimensional case, and in Section 4 its AP property is proven. In Section 5, numerical experiments are performed in order to verify the stability, accuracy, and efficiency of our proposed scheme. Finally, Section 6 summarizes the main conclusions and discusses the future directions of our research. 

## **2 Cahn-Hilliard-Navier-Stokes Equations** 

## 2.1 Model Description 

We follow the model from [1,2] describing the dynamics of two immiscible, compressible, viscous Newtonian fluids in a bounded, open domain _Ω ⊂_ R[3] . Let _ρi_ , _ci_ , **v** _i_ denote, respectively, the density, mass concentration, velocity **v** _i_ of the fluid _i_ = 1 _,_ 2. The mixture density is _ρ_ and the barycentric velocity, **v** , is _ρ_ **v** = _ρ_ 1 **v** 1 + _ρ_ 2 **v** 2. The concentration difference _c_ = _c_ 1 _− c_ 2, taking values in [ _−_ 1 _,_ 1], serves as an order parameter distinguishing the two fluid components. The total Helmholtz free energy of the system is defined as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0003-06.png)


where the positive parameter _ε_ is related to the diffuse interface thickness, which controls the width of the transition region between the two phases. The term _ε_[represents][the][classical][Cahn-Hilliard][regularizing][term][[26].][The][specific] 2 _[|∇][c][|]_[2] Helmholtz free energy is assumed to have the form 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0003-08.png)


where _fe_ is the potential energy and 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0003-10.png)


is a double-well potential. The thermodynamic pressure is related to the potential _[∂][f]_[(] _[ρ,][c]_[)] energy through the relation _p_ ( _ρ_ ) = _ρ_[2] _∂ρ_ . For an isentropic process, we adopt 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0003-12.png)


for a positive constant _Cp_ and _γ >_ 1 is the adiabatic constant. 

The evolution of the mixture is governed by the isentropic, compressible CahnHilliard-Navier-Stokes with gravitational force, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0003-15.png)


where the operators div and _∆_ represents the divergence and laplacian operators, respectively. The first equation expresses the conservation of total mass of the mixture, the second equation represents the balance of momentum taking into account the gravitation acceleration **g** , and the third is a Cahn–Hilliard-type equation modeling the evolution of the concentration difference _c_ . The viscous and capillary 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

4 

effects [15] are incorporated into the model through the stress tensor T = T1 + T2 with 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0004-03.png)


where _λ_ and _ν_ are the viscosity coefficients, which are assumed to be positive. The chemical potential is defined as the variational derivative of the total free energy with respect to the order parameter _c_ . In this setting, it is given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0004-05.png)


The system (2) is supplemented with initial conditions for the density, velocity, and concentration, given by _ρ_ (0 _, x_ ) = _ρ_ 0( _x_ ), **v** (0 _, x_ ) = **v** 0( _x_ ), and _c_ (0 _, x_ ) = _c_ 0( _x_ ). In addition, we impose the boundary conditions 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0004-07.png)


where **n** denotes the outward unit normal vector to the boundary _∂Ω_ . 

In [1, Theorem 1.2], it was proven that, for _γ >_[3] 2[and][suitable][initial][data] ( _ρ_ 0 _,_ **v** 0 _, c_ 0), system (2) with boundary conditions (3) admits global-in-time weak solutions in the sense of Di Perna and Lions. 

2.2 Isentropic compressible Cahn-Hilliard-Navier-Stokes in a low Mach number regime 

In the present work, we focus on the low Mach number regime, which corresponds to taking _Cp ≫_ 0. In such regimes, the pressure becomes extremely large, and in many numerical methods (e.g., [30,32]), _Cp_ appears explicitly in the time step stability restriction, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0004-12.png)


where _∆t_ and _∆x_ are the time and spacial step sizes, respectively, leading to severe restrictions on the time step. 

To this end, we split the pressure into a stiff and non-stiff component, treating the stiff part implicitly [11,12]. Specifically, we write 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0004-15.png)


We denote _p_ 1( _ρ_ ) = _Cp,_ 1 _ρ[γ]_ and _p_ 2( _ρ_ ) = _Cp,_ 2 _ρ[γ]_ with the squared Mach number defined as _δ_ = _Cp[−]_[1][.] 

In general, the choice of the pressure splitting depends on the characteristic fluid speed. For working in a hyperbolic framework and suppressing nonphysical oscillations, we consider _Cp,_ 1 _>_ 0. However, for flows with stronger shocks, a larger value of _Cp,_ 1 is necessary, corresponding to an almost fully explicit treatment of the pressure [12]. In the current work, the effects of this choice are investigated numerically in the section of numerical experiments. 

We consider both the oneand two-dimensional setting. For the latter case, we denote the velocity field by **v** = ( _v_ 1 _, v_ 2) and the gravity **g** = (0 _, g_ ) acting only on 

Title Suppressed Due to Excessive Length 

5 

the vertical axes. Under this assumption, the governing equations reduce to the following two-dimensional form: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-04.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-05.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-06.png)


For the one-dimensional case the system reads as follows: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-08.png)


## **3 Numerical Schemes** 

3.1 Spatial Semidiscretization 

We consider the compressible, isentropic CHNS equations in two-spatial dimensions (5) on the square domain _Ω_ = [0 _,_ 1][2] . The computational grid is based on a MAC approach [20]. Let **x** = ( _x, y_ ) denote the spatial variable. The cell-centered grid consists of _M_[2] nodes 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-12.png)


for _i, j_ = 1 _, · · · , M_ , and with uniform mesh size _h_ = _M_ 1[.][The][staggered][(dual)][grid] consists in 2 _M_ ( _M −_ 1) nodes: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0005-14.png)


where _xi_ + 21[=] _[ih]_[and] _[y][j]_[+][1] 2[=] _[jh]_[.][The][continuity][and][the][Cahn-Hilliard][type] equations are treated at cell centers, while the momentum equations are treated at the dual points: the horizontal component on vertical cell faces and the vertical component on horizontal cell faces. To compute momentum at staggered points, 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

6 

we used local averages of the density. For instance, for the horizontal momentum we define ( _ρ∗,x_ ) _i,j_ = _ρ∗,i_ + 21 _[,j]_[=][1] 2[(] _[ρ][i]_[+] _[i,j]_[+] _[ ρ][i,j]_[)][so][that] 

( _ρv_ 1)( **x** _i_ + 12 _[,j]_[) =] _[ ρ][∗][,i]_[+][1] 2 _[,j][v]_[1] _[,i]_[+][1] 2 _[,j][,]_ 

for _i_ = 1 _, · · · , M −_ 1, _j_ = 1 _, · · · , M_ . Notice that _v_ 1 _,i_ + 12 _[,j]_[= (] _[ρv]_[1][)] _[i]_[+][1] 2 _[,j][/ρ][∗][,i]_[+][1] 2 _[,j]_[.][We] denote by _ρ∗_ **v** = ( _ρ∗,xv_ 1 _, ρ∗,yv_ 2). For the velocity components, we assume no-slip boundary conditions, i.e. 

_v_ 1 _,_ 12 _[,j]_[=] _[ v]_[1] _[,M]_[+][1] 2 _[,j]_[= 0] _[,]_[and] _[v]_[1] _[,i,]_[1] 2[=] _[ v]_[1] _[,i,M]_[+] 2[1][= 0] _[.]_ 

For the points outside the wall, we assume symmetric reflection for the density and odd reflection for the velocity components. For instance, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-07.png)


for _j_ = 1 _, . . . , M_ . Similarly, it is done in the other direction. 

After applying this spatial semi-discretization to (5), one needs to solve a system of _N_ = 2 _M_[2] + 2 _M_ ( _M −_ 1) ordinary differential equations given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-10.png)


where _U_ 0 is the vector of initial conditions and the unknown variables _U_ = ( _uk_ )[4] _k_ =1 are 

_ρi,j_ = _u_ 1 _,i,j ,_ ( _ρv_ 1) _i_ + 21 _[,j]_[=] _[ u]_[2] _[,i]_[+][1] 2 _[,j]_ ( _ρv_ 2) _i,j_ + 12[=] _[ u]_[3] _[,i,j]_[+] 2[1] _[,]_ ( _ρc_ ) _i,j_ = _u_ 4 _,i,j ,_ 

for _i, j_ running over their respective grid indices. Dropping the time dependence of _U_ , _L_ ( _U_ ) is the nonlinear operator storing the spatially discretized differential operators, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-14.png)


where nonzero terms of operators above are defined as follows: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-16.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-17.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0006-18.png)


Title Suppressed Due to Excessive Length 

7 

## _3.1.1 Basic Finite Difference Operators_ 

In this section, we introduce the finite difference operators used to approximate the spatial derivatives of the system (5) on MAC grids. 

To approximate the first derivatives at the grid points **x** _i,j_ , we employ central differences which are second-order accurate at interior points (1 _< i, j < M_ ), and first-order otherwise satisfying the boundary conditions (3). The resulting discrete derivative operator in one spatial direction can be written in matrix form as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0007-05.png)


For the dual grid, we define two finite difference matrices _DM_ and _DM[∗]_[of][size] _M ×_ ( _M −_ 1) that approximate first derivatives at the cell interfaces **x** _i_ + 12 _[,j]_[or] **[ x]** _[i,j]_[+][1] 2 which are second-order accurate at interior points and first-order otherwise. Both matrices incorporate the appropriate boundary conditions (3): 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0007-07.png)


We also define the averaging matrix _AM ∈_ R[(] _[M][−]_[1)] _[×][M]_ , which is used to interpolate quantities between the cell centers and the staggered grid: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0007-09.png)


We introduce _f ∗ g_ = ( _fi,j gi,j_ ) _i,j_ for matrices _f_ , _g_ in R _[n][×][m]_ . 

## _3.1.2 The Operators C_ 1 _and C_ 2 

The convective part of the system is decomposed into two operators, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0007-13.png)


where _C_ 1 acts only on the continuity equation and _C_ 2 on the momentum and CahnHilliard type equations. 

We consider the fluxes in the _x_ - and _y_ -directions: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0007-16.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

8 

Let _F_[ˆ] _[∗]_ and _G_[ˆ] _[∗]_ denote the numerical fluxes associated to _F[∗]_ and _G[∗]_ , respectively. The convective operators are approximated using numerical flux differences at cell centers and cell interfaces. Specifically, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0008-03.png)


The numerical fluxes are computed using the Rusanov flux. For the explicit terms of the operator _C_ (see Section 3.3), we use WENO5 reconstructions, which are fifth-order accurate for finite difference schemes [5,6,27]. Let us describe it for the _x_ -direction case. Denote by _W[x]_ : R[5] _−→_ R the WENO5 reconstruction operator and a function _f_ such that _fi,j_ = _f_ ( **x** _i,j_ ) for indexes _i_ , _j_ running both in the primal and dual grids. So the right and left state reconstructions are, respectively, for primal grids, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0008-05.png)


and for dual grids, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0008-07.png)


and for the terms in _C_ 2, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0008-09.png)


Title Suppressed Due to Excessive Length 

9 

The numerical viscosities _λ[∗]_ are defined as the maximum of the upper bounds of the local characteristic speeds at the reconstructed states of each _F_[ˆ] _[∗]_ , namely, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-03.png)


where _s_ ( _fi,j[∗]_[) =] � _p[′]_ 1 � _fi,j[∗]_ � is the speed of sound for indexes _i, j_ in the primal and dual grids. 

Notice also that the values of _ρ_ on the dual grid and of _v_ 1 on the primal grid are required. To this end, we employ a sixth-order grid transfer operator defined by the coefficients 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-06.png)


Thus, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-08.png)


and similarly in the _y_ -direction. In particular, to evaluate _F_[ˆ] _[ρv]_[2] , we must approximate the velocity component _v_ 1 at staggered locations where it is not directly defined. So, we first approximate 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-10.png)


and then apply the previous transfer grid operator in the _x_ -direction to obtain 

_v_ 1 _,i,j_ + 21[.] 

An analogous procedure is used for the flux _G_[ˆ] _[ρv]_[1] . 

## _3.1.3 The Operator L_ 1 

The nonzero components of the operator _L_ 1 are approximated point-wise and taking central finite differences, specifically, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-16.png)


## _3.1.4 The Operator L_ 2 

The operator _L_ 2 involves the derivatives of the order parameter _c_ in the momentum equation. For the approximation of _L_ 2( _U_ )2 we use: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0009-19.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

10 

for _i_ = 1 _, · · · , M −_ 1 and _j_ = 1 _, · · · , M_ . Similarly, for the _L_ 2( _U_ )3 we use: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-03.png)


for _i_ = 1 _, · · · , M_ and _j_ = 1 _, · · · , M −_ 1. 

These approximations satisfy the boundary conditions (3) and are second-order accurate at interior points and first-order accurate otherwise. 

## _3.1.5 The Operator L_ 3 

The operator _L_ 3, which arises from the Cahn-Hilliard type equation, requires a special treatment. This is because, for stability, only negative definite terms should be treated implicitly. However, the term _∆ψ[′]_ ( _c_ ) = div[�] _ψ[′′]_ ( _c_ ) _∇c_[�] changes sign in ( _−_ 1 _,_ 1) since the potential _ψ_ is of convex-concave type. To handle this, in [16] was shown that if _ψ_ is split into the sum of a convex part _ψ_ 1 and a concave part _ψ_ 2, the resulting scheme for the Cahn-Hilliard equation treating _ψ_ 1 _[′]_[implicitly][and] _[ψ]_ 2 _[′]_ explicitly is unconditionally stable. In particular, we choose 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-08.png)


Let _f ∈C_[4] such that _∇f_ ( _x, y_ ) _·_ **n** = 0 with _fi,j_ = _f_ ( **x** _i,j_ ) for _i, j_ = 1 _, · · · , M_ . We use a second-order accurate approximation for _∆fi,j ≈ ∆hfi,j_ = _∆x,hfi,j_ + _∆y,hfi,j_ where 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-10.png)


and similarly for _∆y,h_ . For _i, j_ = 1 _, · · · , M_ , yields 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-12.png)


and ( _∆ψ_ 2 _[′]_[(] _[c]_[))(] **[x]** _i,j_[)] _[ ≈]_[(] _[ψ]_ 2 _[′′]_[(] _[c]_[)] _[c][x]_[)] _[x]_[(] **[x]** _i,j_[) + (] _[ψ]_ 2 _[′′]_[(] _[c]_[)] _[c][y]_[)] _[y]_[(] **[x]** _i,j_[)][where] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-14.png)


and 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0010-16.png)


Title Suppressed Due to Excessive Length 

11 

Now, it only remains to approximate _∆_ � _ρ_ 1 _[∆c]_ �. To this end, we employ the aforementioned second-order accurate approximation for the laplacian, namely, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-03.png)


where _D_ is the diagonal operator on _M × M_ matrices defined as 

( _D_ ( _v_ ) _w_ ) _i,j_ = _vi,j wi,j , i, j_ = 1 _, . . . , M, v, w ∈_ R _[M][×][M] ._ 

_3.1.6 The Operator L_ 4 

The operator _L_ 4 stores the derivatives of the velocity field in the balance of momentum. For approximating the pure double derivatives, for instance, ( _v_ 1) _xx_ and ( _v_ 1) _yy_ at **x** _i_ + 12 _[,j]_[,][we][use] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-08.png)


for _j_ = 1 _, · · · , M_ , and 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-10.png)


for _i_ = 1 _, · · · , M −_ 1. The approximation of the cross derivative, e.g., ( _v_ 2) _xy_ at **x** _i_ + 12 _[,j]_[is][given][by] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-12.png)


for _i_ = 1 _, · · · , M −_ 1. 

The three expressions above verify the boundary conditions (3) and are secondorder accurate at its respective interior points and first-order accurate otherwise. In matrix form, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-15.png)


Similarly, the other nonzero component of _L_ 4 takes the form 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0011-17.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

12 

3.2 Vector Implementation 

In this section, we reformulate system (7) in vector form for the two-dimensional case. The one-dimensional case follows analogously and is therefore omitted. 

Let vec( _A_ ) denote the column-wise vectorization of a matrix _A ∈_ R _[n][×][m]_ , defined by 

vec( _A_ ) _i_ + _m_ ( _j−_ 1) = _Ai,j ,_ for 1 _≤ i ≤ n,_ 1 _≤ j ≤ m._ 

For simplicity, we will use the same symbols _ϱ_ , _V_ 1, _V_ 2, _C_ , _Ck_ ( _U_ ) _i_ and _Lj_ ( _U_ ) _i_ (for _i, j_ = 1 _, · · · ,_ 4 and _k_ = 1 _,_ 2) to denote both the original matrices and their vectorizations, whenever there is no risk of confusion. 

Let _⊗_ denotes the Kronecker product and _In_ the identity matrix of size _n_ . With this notation, the nonzero blocks of _C_ 1 can be expressed in vector form as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-08.png)


with 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-10.png)


where _Λ[x]_ and _Λ[y]_ denote the diagonal matrices of the maximum characteristic speeds associated with the fluxes _F[ρ]_ and _G[ρ]_ in the _x_ - and _y_ -directions, respectively, evaluated at the reconstructed states. Similarly, the nonzero blocks of the operators _L_ 1, _L_ 3, _L_ 4 can be written as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-12.png)


where _A_ 3 is the tensor constructed form the values of _ψ_ 2 _[′′]_[in][(8)-(9)][and] _[∆] h_[the] Laplacian operator in tensor form. The nonzero blocks of _L_ 4 are the following: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-14.png)


where the matrices _Ai,j_ are given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-16.png)


Using this notation, system (7) can be expressed compactly in vector form by defining 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0012-18.png)


Title Suppressed Due to Excessive Length 

13 

3.3 Implicit-Explicit Schemes 

To construct an implicit-explicit scheme, we employ the technique of doubling variables combined with a partitioned Runge-Kutta approach [7,28,30,32]. Consider a sufficiently smooth function 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-04.png)


defined as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-06.png)


where the only nonzero component of the operators _L_[˜] 1 and _L_[˜] 3 are given by 

˜ _C_ 1( ˜ _U, U_ )1 = ( _IM ⊗ DM_ )( _ϱ∗,x ∗ V_ 1) + ( _DM ⊗ IM_ )( _ϱ∗,y ∗ V_ 2) + _hA_ 1( ˜ _U_ )˜ _ϱ, L_ ˜1( ˜ _U, U_ )2 = ( _IM ⊗ DM[T]_[)] _[p]_[2][(] _[ϱ]_[)] _[,]_ 

_L_ ˜1( ˜ _U, U_ )3 = ( _AM ⊗ IM_ ) _gϱ_ ˜ + ( _DM[T][⊗][I] M_[)] _[p]_[2][(] _[ϱ]_[)] _[,]_ 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-10.png)


Using this operator, the full discrete scheme given by (7) can be written as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-12.png)


which is equivalent to 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-14.png)


Here, all terms involving _U_[˜] are treated explicitly, while those depending on _U_ are handled implicitly. System (14) allows us to apply separate Runge-Kutta schemes to the explicit and the implicit parts. Therefore, we consider a pair of Butcher tableaus with _s_ stages: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-16.png)


The first tableau defines the explicit part of the scheme with _α_ ˜ _i,j_ = 0 for all _j ≥ i_ , while the second tableau represents the diagonally implicit part, where _αi,j_ = 0 for _j > i_ and _αi,i_ = 0. The _γi_ and _γ_ ˜ _i_ coefficients are defined by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0013-18.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

14 

Using these tableaus, the stage values of the partitioned Runge-Kutta method applied to (14) are computed as follows: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0014-03.png)


If _β_ = _β_[˜] and _U[n]_ = _U_[˜] _[n]_ , then both solutions remain identical at every time step, which eliminates the need of doubling the number of variables. In addition, as proven in [28], if both Butcher tableaus are second-order accurate, the resulting partitioned Runge-Kutta method is also second-order accurate. Consequently, the final scheme is given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0014-05.png)


Henceforth, we restrict our analysis to Stiffly Accurate Runge-Kutta schemes, that is, those satisfying _αs,j_ = _βj_ for _j_ = 1 _, . . . , s_ . 

## 3.4 Solution to the Nonlinear Systems 

At each intermediate stage _i_ = 1 _, · · · , s_ , the scheme (15) reduces to solving the following nonlinear system for _U_[(] _[i]_[)] : 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0014-09.png)


which is composed by two subsystems: a nonlinear system for the density and the velocities, and then a linear system for the _c_ variable, following the approach described in [30,32]. 

Title Suppressed Due to Excessive Length 

15 

For the nonlinear subsystem one has to solve _M_[2] + 2 _M_ ( _M −_ 1) equations for equal number of unknowns, corresponding to _ϱ_ , _V_ 1 and _V_ 2. The system reads as: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-04.png)


where the terms marked with a hat � _·_ are explicitly computed at the current stage _i_ = 1 _, · · · , s_ . 

Once _ϱ_[(] _[i]_[)] , _V_ 1[(] _[i]_[)] and _V_ 2[(] _[i]_[)] have been computed, the remaining step to solve in (16) is the linear system for _C_[(] _[i]_[)] . The system takes the form, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-07.png)


which is equivalent to solving the following linear system for _C_[(] _[i]_[)] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-09.png)


Due to the convex splitting stated in Section 3.1.5, the coefficient matrix is symmetric and positive definite, provided that _ϱ_[(] _k[i]_[)] _>_ 0 for all _k_ = 1 _, · · · , M_[2] . 

## 3.5 Nonlinear Solvers 

For the nonlinear subsystem (17) the damped Newton’s method is employed [4]. Dropping the superscript of the _i_ -stage, the nonlinear system (17) expressed in compact form is 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-13.png)


where 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-15.png)


and the nonlinear operator is 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-17.png)


Then at each Newton iteration the solution is updated as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-19.png)


where the step _δ[n]_ is computed by solving the linear system 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0015-21.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

16 

Here, _H[′]_ ( _z[n]_ ) denotes the Jacobian matrix of _H_ evaluated at the current iterate _z[n]_ , and the damping parameter _αn ∈_ (0 _,_ 1] is chosen to ensure that _||H_ ( _z[n]_[+1] ) _||_ 2 is decreasing. The Jacobian matrix has the form 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0016-03.png)


where 0 _M_ 2 denotes the _M_[2] zero matrix, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0016-05.png)


and 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0016-07.png)


For analyzing the invertibility of the Jacobian matrix (20) the following result, proven in [30], is needed. 

**Proposition 1** _If ϱk >_ 0 _for every k_ = 1 _, . . . , M_[2] _, and ν, λ >_ 0 _, then Dϱ_ + _∆tαi,iB is symmetric and strictly positive definite._ 

Consequently, assuming that _ϱk >_ 0 for every _k_ = 1 _, . . . , M_[2] , the Jacobian matrix (20) is invertible provided that 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0016-11.png)


Clearly, if _∆tαi,i_ were zero, the above condition holds. Hence, for sufficiently small values of _∆tαi,i_ the Jacobian matrix _H[′]_ is invertible. 

In [30,32], multigrid V-cycle algorithm with a small number of preand postGauss-Seidel smoothings was proven to be effective for solving system (18). In particular, this approach was successfully applied in [30,32] to the system formed by the sub-block of the Jacobian matrix (20) given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0016-14.png)


The analysis of linear solvers for the complete Jacobian matrix (20) for approximating the solution of system (19) is beyond the scope of this work. 

Title Suppressed Due to Excessive Length 

17 

## 3.6 Time-Step selection 

The time step is chosen based only on the convective part of the system. It follows that the CFL stability condition for the proposed scheme takes the form 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0017-04.png)


where CFL _[∗]_ is some constant less than one, and _cs_ denotes the maximum of characteristic speeds, computed as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0017-06.png)


Since the stiff pressure component is treated implicitly, it does not influence the stability condition. As a result, the time step _∆t_ is independent of the parameter _Cp,_ 2, depending only on the non-stiff pressure part _p_ 1 and the velocity field **v** . This allows the method to avoid severe time-step restrictions typically faced on low Mach number regimes. 

On the other hand, the proposed scheme is not guaranteed to be boundpreserving: the density can be negative or the order parameter _c_ can be outside the physical interval [ _−_ 1 _,_ 1]. As discussed in the monograph [31], the polynomial double-well potential (1) does not satisfy the maximum principle. Consequently, one cannot expect the discrete approximation of _c_ to remain strictly within [ _−_ 1 _,_ 1]. Despite such limitation, this potential is widely used in the literature due to its simplicity [8,13,30,32], but it can be replaced by the classical logarithmic potentials described in [1,8,13,31] and references therein. 

Nevertheless, in our test the scheme preserves the positivity of the density and keeps _c_ almost within bounds, up to a small deviation. Alternative techniques to mitigate these issues are presented in [30,32], where the time step is reduced whenever _|c|_ exceeds a predefined threshold, and then it is gradually increased back. 

## **4 Asymptotic Preserving Property** 

In [2] it was proven that in the low Mach number limit of the compressible CahnHilliard-Navier-Stokes system converges to its incompressible counterpart, under suitable initial conditions. 

The aim of this section, is to show that the proposed scheme is asymptotically stable. To formalize this notion, we recall the definition of an asymptotic preserving scheme provided in [11,12,17,33]. 

**Definition 1** _Let M[δ] be a continuous physical model depending on a perturbation parameter δ. Define M_[0] _as the limiting model obtained from M[δ] when δ →_ 0 _. A numerical scheme M[δ] ∆[for][approximating][M][δ][,][where][∆]_[= (] _[∆t, ∆x]_[)] _[denotes][the][temporal] and spatial discretization parameters, is said to be asymptotic preserving (AP) if:_ 

_1. its stability condition is independent of δ, and_ 

_2. in the limit δ →_ 0 _, the scheme M[δ] ∆[converges][to][a][consistent][discretization][M]_[0] _∆ of the continuous limiting model M_[0] _._ 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

18 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0018-02.png)


**Fig. 1** Diagram illustrates the asymptotic-preserving (AP) property. _M[δ]_ , _M_[0] denotes the continuous compressible and incompressible system, while _M[δ] ∆_[,] _[M]_[0] _∆_[represents][their][discrete] counterparts, respectively. The AP is verified if the diagram commutes. 

## _This concept is illustrated in Figure 1._ 

We denote by _M[δ]_ the compressible system (2) in the low Mach number regime, and by _M[δ] ∆_[its][numerical][discretization][according][to][(15).][The][corresponding][in-] compressible Cahn-Hilliard-Navier-Stokes equations with gravitational acceleration is denoted by _M_[0] and reads as follows: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0018-06.png)


where _ρ_ 0 _>_ 0 is the constant density of the incompressible mixture. Here, _p_ (1) denotes the scalar pressure, which acts as a Lagrange multiplier associated with the incompressibility constraint div **v** = 0. We denote by _M_[0] _∆_[the][discretization][of] (22) according to (15). 

Assume that the density, velocity field, concentration difference, and pressure admit the following expansions [12,23,33]: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0018-09.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0018-10.png)


and the well-preparedness of the data, that is, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0018-12.png)


Here, the terms in the pressure expansion follow directly from a Taylor series around _ρ_ (0), so that _p_ (0) = _p_[�] _ρ_ (0)�, _p_ (1) = _p′_ � _ρ_ (0)� _ρ_ (1), and higher-order terms are obtained similarly. 

> **Theorem** ˜ **1** _Consider an IMEX Stiffly Accurate partitioned_ ˜ _Runge-Kutta scheme with β_ = _β given by_ (15) _. Assume that U[n] and each stage U_[(] _[l]_[)] _, U_[(] _[l]_[)] _admit the decomposition_ (23) _. If U[n] verifies_ (24) _, then so does U_[(] _[l]_[)] _. Furthermore, if U[n]_[+1] _admits the decomposition_ (23) _, then U[n]_[+1] _is well-prepared and the scheme is AP._ 

Title Suppressed Due to Excessive Length 

19 

_Proof_ For simplicity, in the proof, we shall assume that the pressure splitting stated in (4) is reformulated in terms of the Mach number, that is, 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0019-03.png)


We define _p_[(] _[l]_[)] = _p ρ_[(] _[l]_[)][�] for each stage _l_ . Let _∇h_ , div _h_ and _∆h_ denote the discrete � gradient, divergence and laplacian operators, and _Λ∗_ be the diagonal matrix of the numerical viscosities associated to the flux of the continuity equation. We prove the result by induction on the number of stages _s_ . ˜ For one stage _s_ = 1. First, let us show that _U_[(1)] is well-prepared. We have that _U_[(1)] = _U[n]_ , so the momentum part in _U_[(1)] is 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0019-05.png)


Since _α_ 1 _,_ 1 = 0, taking limits when _δ →_ 0 it is obtained that 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0019-07.png)


By definition of the pressure yields that _ρ_[(1)] (0)[is][constant.][The][leading][terms][in][the] implicit stage for the mass conservation equation are given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0019-09.png)


Since both _ρ_[(1)] (0)[and] _[ρ][n]_ (0)[are][constant,][summing][up][the][above][expression][over][all] spatial indices, a telescope sum in the velocity terms appear, and the boundary contributions vanish due to the boundary conditions (3). Therefore, _ρ_[(1)] (0)[=] _[ρ][n]_ (0)[.] Consequently, the divergence free condition is obtained from the mass conservation and that _αi,i_ = 0, i.e., 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0019-11.png)


Since the scheme is Stiffly Accurate, then _U[n]_[+1] = _U_[(1)] , so _U[n]_[+1] is well-prepared provided it admits the decomposition (23). 

Let us show that the scheme is AP for _s_ = 1. For simplicity, we shall assume that _ρ_[(1)] (0)[=][1,][otherwise][all][the][terms][must][be][scaled][by][(] _[ρ]_[(1)] (0)[)] _[−]_[1][.][The][leading] terms in the momentum of _U_[(1)] , are those involved in the _O_ (1) and _O_ ( _δ_ ) terms, so applying that _ρ_ ˜[(1)] (0)[=] _[ ρ][n]_ (0)[,][(25)][and][(26),][yields] 

_n n_ **v** (0)[(1)][=] **[ v]** (0) _[n]_[+] _[ ∆tα]_[1] _[,]_[1] � **g** _−_[�] div _h_ � **v** (0) _⊗_ **v** (0)�� + div _h_ T2 _c_ (0)[+] _[ ν∆] h_ **[v]** (0)[(1)] _[−∇][h][p]_[(1)] (1)� _._ Similarly, for the Cahn-Hilliard type equation the leading terms are the _O_ (1), i.e., _n c_[(1)] (0)[=] _[ c][n]_ (0)[+] _[ ∆tα]_[1] _[,]_[1] � _−_[�] div _h_ � **v** (0) _c_ (0)�� + _∆_ � _ψ_ + _[′]_ � _c_[(1)] (0)� + _ψ−[′]_ � _c[n]_ (0)� _− ε∆c_[(1)] (0)�� _._ 

Similarly, for the Cahn-Hilliard type equation the leading terms are the _O_ (1), i.e., 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

20 

Since the scheme is Stiffly Accurate, then _U[n]_[+1] = _U_[(1)] , so the scheme is AP for _s_ = 1. 

We assume that the result is true for the first _s −_ 1 stages, and we prove it for stage _s_ . The momentum equation in _U_[(] _[s]_[)] is given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-04.png)


Taking the limit when _δ →_ 0 with _αs,s_ = 0 and applying that the first _s−_ 1 implicit stages are well-prepared, yields that 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-06.png)


so _ρ_[(] (0) _[s]_[)][is][spatially][constant.] 

_U_ ˜[(] _[l]_[)] Itforfollows _l_ = 1 _, · · ·_ inductively, _, s_ are that the leading terms in the conservation of mass in 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-09.png)


since _U_[(] _[l]_[)] is well-prepared for _l_ = 0 _, · · · , s −_ 1 and _ρ[n]_ (0)[is][constant.][Therefore,] _[ρ]_[˜][(] (0) _[l]_[)] is constant for _l_ = 1 _, · · · , s_ . Applying again the induction hypothesis and (27), the leading terms in the conservation of mass of the implicit _s_ -stage are: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-11.png)


Proceeding as before, adding all terms up in the previous expression yields that _ρ_[(] (0) _[s]_[)][=] _[ ρ][n]_ (0)[,][which][implies][that] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-13.png)


So _U[n]_[+1] is well-prepared since _αs,j_ = _βj_ for every _j_ = 1 _, · · · , s_ . 

It only remains to show that the scheme of _s_ -stages is AP. Similarly, we assume that _ρ_[(] (0) _[l]_[)][=][1][for] _[l]_[=][1] _[, . . . , s]_[.][Applying][that] _[U]_[(] _[l]_[)][is][well-prepared][for] _[l]_[=][1] _[, . . . , s]_ and (27), then the leading terms of the momentum equation at each _l_ -stage are given by 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0020-16.png)


Title Suppressed Due to Excessive Length 

21 

For the Cahn-Hilliard type equation it is obvious. The AP property follows from the fact that the scheme is Stiffly Accurate. 

## **5 Numerical experiments** 

The numerical experiments are presented in this section. The main objectives are as follows: 

1. To show that the order of the global convergence error agrees with the order of the numerical scheme. 

2. To verify that the number of time steps required by the IMEX scheme is consistent with the stability restriction imposed by the convective subsystem (21). 

3. To explain the properties preserved by the scheme, such as mass conservation, region preservation along with the CFL value (see Section 3.6), and others. 

In all our experiments, the initial CFL number is set to 0 _._ 4, the adiabatic exponent _γ_ fixed to[5] 3[and][the][parameters][are][set][to] 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0021-09.png)


We define _Cp,_ 1 = ~~�~~ _Cp_ and _Cp,_ 2 = _Cp −Cp,_ 1. This choice has proven to be effective, as the experiments conducted under this setting have been successful. 

All experiments were performed using a MATLAB R2024a implementation on a Linux machine running on 32 core of an AMD EPYC 7282. 

We consider Stiffly Accurate Runge-Kutta schemes. In particular, we use a first-order method defined by the following Butcher tableau: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0021-13.png)


and a second-order method given by the _[∗]_ -DIRKSA scheme: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0021-15.png)


## 5.1 Order Tests 

In this section we show that the _[∗]_ -DIRKSA scheme attains second-order of convergence. To this end, we introduce a forcing term into the equations, ensuring that the solution follows a prescribed analytical form. Specifically, the exact solution for the one-dimensional (6) case is defined as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0021-18.png)


Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

22 

and for the two-dimensional case (5), 

_ρ[∗]_ ( _x, y, t_ ) = 1 + _δ_ cos(2 _πx_ ) cos( _πy_ )( _t_ + 1) _,_ 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0022-04.png)


Notice that in both cases the initial velocity filed is divergence-free and the density is constant in space at _O_ ( _δ_ ). 

For the performance, the squared Mach numbers are taken as _δ_ = 10 _[−][k]_ for _k_ = 1 _, . . . ,_ 8. The time-step _∆t_ is determined by the convective subsystem according to (21). We consider meshes of size _M_ = 2 _[i]_ for _i_ = 3 _, · · · ,_ 8. The global errors and the experimental orders of convergence (EOC) are evaluated at _T_ = 0 _._ 01 and are computed as 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0022-07.png)


The results for the oneand two-dimensional cases are shown in Table 1. In both tables, the _[∗]_ -DIRKSA scheme achieves second-order convergence, while the EEIE scheme is first-order in the one-dimensional case and in the two-dimensional case for _Cp ≤_ 10[6] decreases away from two. When _Cp_ = 10[7] _,_ 10[8] in the latter case, the order of convergence seems to tend to two. One possible explanation is that the value of _Cp,_ 1 increases significantly as _Cp_ increases, making the timeintegrator more robust. Consequently, the spatial discretization order dominates the convergence. Nevertheless, this phenomenon also occurs for _Cp ≤_ 10[6] until _M_ becomes sufficiently large, as illustrated in Table 1. We expect that for _M >_ 1024 the mentioned orders tends to 1, but due to the computational cost, we did not perform such experiments. In the remainder of this work, we restrict our experiments to the _[∗]_ -DIRKSA scheme, since it consistently attains second-order of convergence. 

## 5.2 Test 1, 2 and 3 

In this section, we evaluate the performance of the following tests for several stiff pressure coefficients _Cp_ = 10[2] _[k]_ for _k_ = 1 _, . . . ,_ 4. For this purpose, mass conservation, region preserving for the _c_ -variable and the limit properties of the compressible scheme are discussed numerically. 

**Test 1** This test is designed to show that the method remains stable even when the initial condition for the _c_ -variable lies within the unstable region ( _−_ ~~_√_~~[1] 3 _[,]_ ~~_√_~~ 13[)] (see [14,30,32]). In particular, we consider the following initial conditions: 

_ρ_ 0( _x, y_ ) = 1 + _δ_ cos(2 _πx_ ) cos( _πy_ ) _,_ 

**v** 0( _x, y_ ) = (1 + _δ_ ) ((1 _−_ cos(2 _πx_ )) sin(2 _πy_ ) _,_ (cos(2 _πy_ ) _−_ 1) sin(2 _πx_ )) _,_ 

_c_ 0( _x, y_ ) = 0 _._ 1(1 _− δ_ ) cos( _πx_ ) cos( _πy_ ) _,_ 

Title Suppressed Due to Excessive Length 

23 

|1D<br>2D|1D<br>2D|
|---|---|
|_∗_-DIRKSA<br>EE-IE<br>_∗_-DIRKSA<br>EE-IE||
|_Cp_<br>_M_<br>_eM_<br>EOC_M_<br>_eM_<br>EOC_M_|_eM_<br>EOC_M_<br>_eM_<br>EOC_M_|
|10<br>8<br>1.317e-03<br>—<br>8.799e-04<br>—<br>16<br>2.608e-04<br>2.336<br>2.113e-04<br>2.058<br>32<br>6.411e-05<br>2.024<br>1.490e-04<br>0.504<br>64<br>1.609e-05<br>1.995<br>8.938e-05<br>0.738<br>128<br>4.028e-06<br>1.998<br>4.947e-05<br>0.853<br>256<br>1.010e-06<br>1.996<br>2.665e-05<br>0.893<br>512<br>2.525e-07<br>2.000<br>1.365e-05<br>0.965<br>1024 6.317e-08<br>1.999<br>6.933e-06<br>0.977|2.4179e-02<br>—<br>2.0659e-02<br>—<br>6.8859e-03<br>1.81<br>6.3482e-03<br>1.70<br>1.8061e-03<br>1.93<br>1.8142e-03<br>1.81<br>4.5478e-04<br>1.99<br>5.0700e-04<br>1.84<br>1.1369e-04<br>2.00<br>1.4907e-04<br>1.77<br>2.8398e-05<br>2.00<br>4.8390e-05<br>1.62<br>7.0908e-06<br>1.86<br>1.7788e-05<br>1.19<br>1.76861-06<br>2.00<br>7.3560e-06<br>1.27|
|102<br>8<br>1.216e-03<br>—<br>6.669e-04<br>—<br>16<br>2.904e-04<br>2.066<br>1.247e-04<br>2.419<br>32<br>7.263e-05<br>1.999<br>7.355e-05<br>0.762<br>64<br>1.814e-05<br>2.001<br>5.459e-05<br>0.430<br>128<br>4.540e-06<br>1.998<br>3.312e-05<br>0.721<br>256<br>1.135e-06<br>2.000<br>1.772e-05<br>0.902<br>512<br>2.837e-07<br>2.000<br>9.175e-06<br>0.949<br>1024 7.094e-08<br>2.000<br>4.686e-06<br>0.969|1.9006e-02<br>—<br>1.6018e-02<br>—<br>6.0020e-03<br>1.66<br>5.6030e-03<br>1.52<br>1.6209e-03<br>1.89<br>1.6349e-03<br>1.78<br>4.1116e-04<br>1.98<br>4.5097e-04<br>1.86<br>1.0310e-04<br>2.00<br>1.2913e-04<br>1.80<br>2.5790e-05<br>2.00<br>4.0565e-05<br>1.67<br>6.4443e-06<br>2.00<br>1.4410e-05<br>1.49<br>1.6087e-06<br>2.00<br>5.8013e-06<br>1.31|
|103<br>8<br>6.203e-04<br>—<br>3.995e-04<br>—<br>16<br>1.246e-04<br>2.316<br>9.067e-05<br>2.139<br>32<br>2.866e-05<br>2.120<br>2.532e-05<br>1.840<br>64<br>7.024e-06<br>2.029<br>1.422e-05<br>0.832<br>128<br>1.749e-06<br>2.006<br>8.406e-06<br>0.759<br>256<br>4.369e-07<br>2.002<br>4.508e-06<br>0.899<br>512<br>1.092e-07<br>2.000<br>2.367e-06<br>0.930<br>1024 2.730e-08<br>2.000<br>1.203e-06<br>0.977|1.2832e-02<br>—<br>1.1383e-02<br>—<br>5.5906e-03<br>1.20<br>5.2857e-03<br>1.11<br>1.6172e-03<br>1.79<br>1.6096e-03<br>1.72<br>4.1831e-04<br>1.95<br>4.4163e-04<br>1.87<br>1.0544e-04<br>1.99<br>1.2178e-04<br>1.86<br>2.6408e-05<br>2.00<br>3.5516e-05<br>1.78<br>6.6054e-06<br>2.00<br>1.1369e-05<br>1.64<br>1.6524e-06<br>2.00<br>4.0807e-06<br>1.48|
|104<br>8<br>1.312e-04<br>—<br>1.258e-04<br>—<br>16<br>6.469e-05<br>1.020<br>4.345e-05<br>1.534<br>32<br>1.728e-05<br>1.905<br>1.717e-05<br>1.340<br>64<br>4.273e-06<br>2.015<br>6.935e-06<br>1.308<br>128<br>1.063e-06<br>2.007<br>3.404e-06<br>1.027<br>256<br>2.653e-07<br>2.002<br>1.840e-06<br>0.888<br>512<br>6.630e-08<br>2.001<br>9.787e-07<br>0.911<br>1024 1.657e-08<br>2.000<br>5.067e-07<br>0.950|6.5014e-03<br>—<br>6.1598e-03<br>—<br>4.7449e-03<br>0.45<br>4.6011e-03<br>0.42<br>1.5540e-03<br>1.61<br>1.5527e-03<br>1.57<br>4.1191e-04<br>1.92<br>4.2678e-04<br>1.86<br>1.0445e-04<br>1.98<br>1.1448e-04<br>1.90<br>2.6203e-05<br>1.99<br>3.1766e-05<br>1.85<br>6.5566e-06<br>2.00<br>9.4616e-06<br>1.75<br>1.6400e-06<br>2.00<br>3.1202e-06<br>1.60|
|105<br>8<br>1.338e-04<br>—<br>1.152e-04<br>—<br>16<br>3.780e-05<br>1.824<br>3.461e-05<br>1.736<br>32<br>1.021e-05<br>1.888<br>1.175e-05<br>1.558<br>64<br>2.013e-06<br>2.343<br>4.435e-06<br>1.406<br>128<br>5.653e-07<br>1.832<br>1.907e-06<br>1.218<br>256<br>1.451e-07<br>1.962<br>8.772e-07<br>1.120<br>512<br>3.650e-08<br>1.991<br>4.304e-07<br>1.027<br>1024 9.139e-09<br>1.998<br>2.162e-07<br>0.994|2.1418e-02<br>—<br>2.0674e-02<br>—<br>3.8763e-03<br>2.47<br>3.7918e-03<br>2.45<br>1.4648e-03<br>1.40<br>1.4683e-03<br>1.37<br>4.0482e-04<br>1.86<br>4.1362e-04<br>1.83<br>1.0380e-04<br>1.96<br>1.0967e-04<br>1.92<br>2.6128e-05<br>1.99<br>2.9357e-05<br>1.90<br>6.5437e-06<br>1.99<br>8.2499e-06<br>1.83<br>1.6367e-06<br>2.00<br>2.5131e-06<br>1.71|
|106<br>8<br>1.295e-04<br>—<br>1.189e-04<br>—<br>16<br>3.100e-05<br>2.062<br>3.175e-05<br>1.905<br>32<br>8.049e-06<br>1.946<br>9.094e-06<br>1.804<br>64<br>2.135e-06<br>1.915<br>3.101e-06<br>1.552<br>128<br>5.455e-07<br>1.968<br>1.200e-06<br>1.370<br>256<br>1.345e-07<br>2.020<br>5.229e-07<br>1.198<br>512<br>3.307e-08<br>2.024<br>2.455e-07<br>1.091<br>1024 8.226e-09<br>2.007<br>1.194e-07<br>1.040|5.4800e-02<br>—<br>5.3752e-02<br>—<br>3.7253e-03<br>3.88<br>3.6616e-03<br>3.88<br>1.3844e-03<br>1.43<br>1.3855e-03<br>1.40<br>3.9548e-04<br>1.81<br>4.0041e-04<br>1.79<br>1.0315e-04<br>1.94<br>1.0638e-04<br>1.91<br>2.6102e-05<br>1.98<br>2.7888e-05<br>1.93<br>6.5447e-06<br>1.99<br>7.5033e-06<br>1.83<br>1.6374e-06<br>2.00<br>2.1390e-06<br>1.81|
|107<br>8<br>1.691e-04<br>—<br>1.596e-04<br>—<br>16<br>3.108e-05<br>2.444<br>3.056e-05<br>2.384<br>32<br>7.937e-06<br>1.969<br>8.362e-06<br>1.870<br>64<br>1.991e-06<br>1.995<br>2.387e-06<br>1.809<br>128<br>4.974e-07<br>2.001<br>8.252e-07<br>1.532<br>256<br>1.279e-07<br>1.960<br>3.265e-07<br>1.338<br>512<br>3.242e-08<br>1.980<br>1.447e-07<br>1.174<br>1024 8.413e-09<br>1.946<br>6.849e-08<br>1.079|1.1107e-01<br>—<br>1.1000e-01<br>—<br>6.1995e-03<br>4.16<br>6.1436e-03<br>4.16<br>1.3430e-03<br>2.21<br>1.3431e-03<br>2.19<br>3.8404e-04<br>1.81<br>3.8672e-04<br>1.80<br>1.0202e-04<br>1.91<br>1.0379e-04<br>1.90<br>2.6019e-05<br>1.97<br>2.7000e-05<br>1.94<br>6.5401e-06<br>1.99<br>6.9520e-06<br>1.96<br>—<br>—<br>—<br>—|
|108<br>8<br>2.433e-04<br>—<br>2.374e-04<br>—<br>16<br>3.118e-05<br>2.964<br>3.076e-05<br>2.948<br>32<br>7.918e-06<br>1.977<br>8.071e-06<br>1.930<br>64<br>1.989e-06<br>1.993<br>2.134e-06<br>1.919<br>128<br>4.978e-07<br>1.998<br>6.250e-07<br>1.772<br>256<br>1.253e-07<br>1.990<br>2.207e-07<br>1.502<br>512<br>3.168e-08<br>1.984<br>8.928e-08<br>1.305<br>1024 8.156e-09<br>1.957<br>4.014e-08<br>1.153|2.0294e-01<br>—<br>2.0219e-01<br>—<br>1.4330e-02<br>3.82<br>1.4267e-02<br>3.82<br>1.4663e-03<br>3.29<br>1.4650e-03<br>3.28<br>3.7428e-04<br>1.97<br>3.7570e-04<br>1.96<br>1.0034e-04<br>1.90<br>1.0132e-04<br>1.89<br>2.5881e-05<br>1.95<br>2.6416e-05<br>1.94<br>6.5312e-06<br>1.99<br>6.8161e-06<br>1.96<br>—<br>—<br>—<br>—|


**Table 1** _L_ 1 errors and experimental order of convergence for the DIRKSA and EE-IE IMEX schemes for both the oneand two-dimensional case, evaluated for different _Cp_ values with _Cp,_ 1 = ~~�~~ _Cp_ , in the test using a forced solution. 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

24 

which clearly verify the boundary conditions (3) and the divergence-free condition for the velocity field. The density is constant in space at leading orders of the low Mach number. 

It is observed in Figures 2, 3, 4 that initially the density is dispersed and the order parameter _c_ lies within the interval ( _−_ ~~_√_~~[1] 3 _[,]_ ~~_√_~~ 13[).][However,][as][the][simulation] evolves, the density is gradually increased near the bottom boundary _y_ = 0 due to the effects of gravitation. In addition, the evolution of the _c_ -variable illustrates the process of phase separation where complex patterns are formed. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-04.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-05.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-06.png)


**Fig. 2** Results for Test 1, _T_ = 0 _,_ 0 _._ 01, _M_ = 128 and _Cp_ = 10[8] . Initially, _c_ is lies within the unstable region. At the beginning of the simulation, phase separation occurs. Moreover, the density starts to become higher in the lower part of the domain due to gravity. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-08.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-09.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0024-10.png)


**Fig. 3** Results for Test 1, _T_ = 0 _._ 03 _,_ 0 _._ 05, _M_ = 128 and _Cp_ = 10[8] . The process of spinodal decomposition continues, and the density is accumulating at the bottom of the domain due to gravity. 

Title Suppressed Due to Excessive Length 

25 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-02.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-04.png)


**Fig. 4** Results for Test 1, _T_ = 0 _._ 07 _,_ 0 _._ 1, _M_ = 128 and _Cp_ = 10[8] . It can be observed that density remains almost constant among distinct times and that phase separation has almost finished. 

**Test 2** The objective of this test is to assess the performance of the scheme when the order parameter _c_ initially lies outside the unstable region. We consider: 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-07.png)


which verify (3) and div **v** = 0 and the density is constant in space at _O_ ( _δ_ ). 

Figures 5, 6 and 7 show that the order parameter _c_ evolves from outside the spinodal region toward a constant value of 3 _/_ 4. In this state, the system follows the compressible Navier-Stokes with gravitational forces behaving in a low Mach number regime when _Cp_ becomes large. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-10.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-11.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0025-12.png)


**Fig. 5** Results for Test 2, _T_ = 0 _,_ 0 _._ 01, _M_ = 128 and _Cp_ = 10[8] . Initially, _c_ lies outside the unstable region and the density is dispersed. At _T_ = 0 _._ 01, the fluid starts to have denser regions near the bottom. 

**Test 3** The aim of this test, taken from [30,32], is to show the spinodal decomposition. To this end, the initial conditions are set as _ρ_ 0 = 1, **v** 0 = 0, and 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

26 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-02.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-04.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-05.png)


**Fig. 6** Results for Test 2, _T_ = 0 _._ 03 _,_ 0 _._ 05, _M_ = 128 and _Cp_ = 10[8] . The bubbles formed order parameter start to merge around 4[3][also][growing][in][size.][The][density][is][higher][at][the][bottom] of the domain due to gravity. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-07.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-08.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-09.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-10.png)


**Fig. 7** Results for Test 2, _T_ = 0 _._ 07 _,_ 0 _._ 1, _M_ = 128 and _Cp_ = 10[8] . Density remains almost constant, while the order parameter starts tending to[3] 4[.] 

_c_ 0 is initialized as a uniform random sample of zero mean and 10 _[−]_[10] standard deviation. 

Figures 8 and 9 show the results up to _T_ = 0 _._ 1, where the spinodal decomposition occurs at the beginning of the simulations. In addition, density becomes higher near the bottom boundary _y_ = 0 due to the gravitational effects, but it seems to be stabilized as time evolves, see Figure 9. 

## _5.2.1 Conservation of mass and bound-preserving properties_ 

Figure 10 illustrates the conservation of mass for the three tests. Specifically, the mass conservation errors for _ρ_ and _q_ = _ρc_ are computed using 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0026-16.png)


Title Suppressed Due to Excessive Length 

27 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-02.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-04.png)


**Fig. 8** Results for Test 3, _T_ = 0 _,_ 0 _._ 01, _M_ = 128 and _Cp_ = 10[8] . Initially, density is constant, velocity is zero everywhere, and the order parameter is a random perturbation around _c_ = 0. It can be seen at _T_ = 0 _._ 01 that phase separation has started to occur, while density is higher at the bottom. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-06.png)


<br>


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-07.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0027-08.png)


**Fig. 9** Results for Test 3, _T_ = 0 _._ 03 _,_ 0 _._ 05 _,_ 0 _._ 07 _,_ 0 _._ 1, _M_ = 128 and _Cp_ = 10[8] . The process of phase separation continues, and the density seems to stabilize with higher values at the bottom of the domain. 

On the other hand, the order parameter _c_ has not exceeded [ _−_ 1 _,_ 1] considerably, and it has been kept below its bounds throughout all experiments. Table 2 shows the maxim and minimum value of _c_ during the performance. Figure 11 shows the time evolution of the maximum and minimum values of the _c_ -component, which rarely exceed the interval [ _−_ 1 _,_ 1]. Therefore, the chosen CFL number of 0 _._ 4 can be considered safe for our simulations. 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

28 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-02.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-04.png)


**Fig. 10** Time evolution of the mass conservation errors for both _ρ_ and _q_ with _M_ = 128 for Test 1, 2, and 3 for _Cp_ = 10[8] . 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-06.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-07.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-08.png)


**Fig. 11** Time evolution of the maximum and minimum values of the order parameter _c_ with _M_ = 128 for Test 1, 2, and 3 for _Cp_ = 10[8] . 

||_Cp_ = 102<br>_Cp_ = 104<br>_Cp_ = 106<br>_Cp_ = 108|
|---|---|
|Test 1|0_._9976<br>_−_0_._9924<br>1_._0130<br>_−_1_._0128<br>1_._0113<br>_−_1_._0114<br>1_._0103<br>_−_1_._0103|
|Test 3|1_._0097<br>_−_1_._0100<br>1_._0121<br>_−_1_._0100<br>1_._0112<br>_−_1_._0128<br>1_._0122<br>_−_1_._0092|


**Table 2** Maximum and minimum values for the _c_ evolution in Test 1 and 3. 

## _5.2.2 Low Mach number regime_ 

Here, we analyze the limit of the compressible scheme _M[δ] ∆_[toward the incompress-] ible scheme _M_[0] _∆_[.][To][this][end,][we][focus][on][the][three][previous][tests][specially][when] the squared Mach numbers are _δ_ = 10 _[−]_[2] _[k]_ for _k_ = 2 _,_ 3 _,_ 4. Note that in these tests, the initial conditions are well-prepared in the sense of (24). In addition, according to Theorem 1, each time-step must be well-prepared, ensuring that the scheme is AP. Figures 12 and 13 illustrate this behavior: the density approaches to 1, and the divergence free condition is satisfied. 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-14.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-15.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0028-16.png)


**Fig. 12** Divergence free condition for Test 1, Test 2 and Test 3 with _M_ = 128 and _Cp_ = 10[8] . 

Title Suppressed Due to Excessive Length 

29 


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0029-02.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0029-03.png)


![](images/-2026-_implicit_explicit_전속도_압축성_CH-NS_eq.pdf-0029-04.png)


**Fig. 13** Well-preparedness of the solution for Test 1, 2, and 3 at _T_ = 0 _._ 1 with _M_ = 128 and _Cp_ = 10[8] . 

## **6 Conclusions and future work** 

In this work, we present an efficient second-order asymptotic-preserving IMEX schemes on staggered grids for the two-dimensional compressible isentropic CahnHilliard-Navier-Stokes equations for any Mach number regime. The proposed method avoids the severe restriction imposed by the high-order and stiff pressure terms. To validate the method, several numerical test have been performed, showing that second-order accuracy is achieved with the time-step constrained only by the convective subsystem of the equations. 

For future work, we aim to extend the present framework to the non-barotropic compressible Cahn-Hilliard-Navier-Stokes in a low Mach number regime, as well as to the three-dimensional case using Galerkin techniques. 

Regarding the pressure splitting defined in (4), we plan to further research on the possible range of _Cp,i_ values. Our current strategy of setting _Cp,_ 1 = ~~�~~ _Cp_ has proven to be successful in our experiments, although no formal proof is given. 

When solving systems (17) and (18), neither the positivity of the density _ρ_ nor the boundedness of the order parameter _c ∈_ [ _−_ 1 _,_ 1] can be guaranteed. To address this issue, we plan to employ bound-preserving high-order reconstructions schemes which can effectively circumvent these physical constraints. 

## Conflict of interest 

The authors declare that they have no conflict of interest. 

## Data Availability Statements 

Data sharing is not applicable to this article as no datasets were generated or analyzed during the current study. 

## **Acknowledgments** 

This paper has received financial support from the research projects PID2023146836NB-I00, granted by MCIN/ AEI /10.13039/ 501100011033, and CIAICO/2024/089, granted by GVA. 

## **References** 

> 1. H. Abels, and E. Feireisl. On a diffuse interface model for a two-phase flow of compressible viscous fluids. _Indiana Univ. Math. J._ , 57(2):659–698, 2008. 

Andreu Martorell, Pep Mulet, Dionisio F. Y´a˜nez 

30 

2. H. Abels, Y. Liu, and S. Neˇcasov´a.[ˇ] Low Mach number limit of a diffuse interface model for two-phase flows of compressible viscous fluids. _GAMM-Mitteilungen_ , 47(4):e202470008, 2024. 

3. T. Alazard. Low Mach number limit of the full Navier-Stokes equations. _Arch. Ration. Mech. Anal._ , 180(1):1–73, 2006. 

4. R. B¨urger, D. Inzunza, P. Mulet, and L. M. Villada. Implicit–explicit schemes for nonlinear nonlocal equations with a gradient flow structure in one space dimension. _NMPDE_ , 35(3):1008–1034, 2019. 

5. A. Baeza, R. Burger, P. Mulet, and D. Zorio. On the Efficient Computation of Smoothness Indicators for a Class of WENO Reconstructions. _J. Sci. Comput._ , 80(2):1240–1263, 2019. 

6. A. Baeza, R. B¨urger, P. Mulet, and D. Zorio. WENO Reconstructions of Unconditionally Optimal High Order. _SIAM J. Numer. Anal._ , 57(6):2760–2784, 2019. 

7. S. Boscarino, R. B¨urger, P. Mulet, G. Russo, and L.M. Villada. Linearly Implicit Imex Runge-Kutta Methods for a Class of Degenerate Convection-Diffusion Problems. _SIAM J. Sci. Comp._ , 37(2):B305–B331, 2015. 

8. F. Boyer. Mathematical study of multi-phase flow under shear through order parameter formulation. _Asymptot. Anal._ , 20(2):175–212, 1999. 

9. J.W. Cahn, and J.E. Hilliard. Free energy of a nonuniform system .3. Nucleation in a 2-component incompressible fluid. _J. Chem. Phys._ , 31(3):688–699, 1959. 

10. L. Chen, and J. Zhao. A novel second-order linear scheme for the Cahn-Hilliard-NavierStokes equations. _J. Comput. Phys._ , 423, 2020. 

11. F. Cordier, P. Degond, and A. Kumbaro. An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations. _J. Comput. Phys._ , 231(17):5685–5704, 2012. 

12. P. Degond, and M. Tang. All speed scheme for the low Mach number limit of the isentropic Euler equations. _Commun. Comput. Phys._ , 10(1):1–31, 2011. 

13. F. Dhaouadi, M. Dumbser, and S. Gavrilyuk. A first-order hyperbolic reformulation of the Cahn–Hilliard equation. _Proc. Royal Soc. A_ , 481(2312):20240606, 2025. 

14. C.M. Elliott. The Cahn-Hilliard model for the kinetics of phase separation. In _Math. Models Phase Change Probl. ( Obidos,[´] 1988)_ , volume 88 of _Internat. Ser. Numer. Math._ , pages 35–73. Birkh¨auser, Basel, 1989. 

15. J.L. Ericksen. Liquid crystals with variable degree of orientation. _Arch. Ration. Mech. Anal._ , 113(2), 97–120, 1991. 

16. D.J. Eyre. Unconditionally gradient stable time marching the Cahn-Hilliard equation. In J.W. Bullard, L.Q. Chen, R.K. Kalia, and A.M. Stoneham, editors, _Comput. Math. Models Microstruct. Evol._ , volume 529, pages 39–46. Mat. Res. Soc. (MRS), 1998. 

17. J Haack, S Jin, and J Liu. An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations. _Commun. Comput. Phys._ , 12(4), 955–980, 2012. 

18. D. Han, and X. Wang. A second order in time, uniquely solvable, unconditionally stable numerical scheme for Cahn-Hilliard-Navier-Stokes equation. _J. Comput. Phys._ , 290:139– 156, 2015. 

19. Q. He, and X. Shi. Numerical Study of Compressible Navier-Stokes-Cahn-Hilliard System. _Comm. Math. Sci._ , 18(2):571–591, 2020. 

20. F. H. Harlow, and J. E. Welch. Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface. _Phys. Fluids_ , 8(12):2182, 1965. 

21. H. Jia, X. Wang, and K. Li. A novel linear, unconditional energy stable scheme for the incompressible Cahn-Hilliard-Navier-Stokes phase-field model. _Comput. Math. Appl._ , 80(12):2948–2971, 2020. 

22. G.J. Kynch. A Theory of Sedimentation. _Trans. Faraday Soc._ , 48(2):166–176, 1952. 

23. S Klainerman, and A Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids. _Commun. Pure Appl. Math._ , 34(4), 481–524, 1981. 

24. P.L. Lions, and N. Masmoudi. Incompressible limit for a viscous compressible fluid. _J. Math. Pures Appl._ , 77(6), 585–627, 1998. 

25. M. Li, and C. Xu. New efficient time-stepping schemes for the Navier-Stokes-Cahn-Hilliard equations. _Comput. Fluids_ , 231, 2021. 

26. J. Lowengrub, and L. Truskinovsky. Quasi-incompressible Cahn-Hilliard fluids and topological transitions. _Proc. Royal Soc. A_ , 454(1978):2617–2654, 1998. 

27. C. Par´es-Pulido, S. Mishra, and K.G. Pressel. Arbitrarily high-order (weighted) essentially non-oscillatory finite difference schemes for anelastic flows on staggered meshes. _SAM Res. Rep._ , 2019, ETH Zurich. 

Title Suppressed Due to Excessive Length 

31 

28. L. Pareschi, and G. Russo. Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation. _J. Sci. Comput._ , 25(1/2):129–155, 2005. 

29. S. Patankar. Numerical Heat Transfer and Fluid Flow. CRC Press, 2018. 

30. A. Martorell, P. Mulet, and D. F. Y´a˜nez. Implicit-explicit schemes for compressible Cahn– Hilliard–Navier–Stokes equations on staggered grids. _arXiv preprint arXiv:2512.20351_ , 2025. 

31. A. Miranville. The Cahn—Hilliard equation: recent advances and applications. SIAM, 2019. 

32. P. Mulet. Implicit-Explicit Schemes for Compressible Cahn–Hilliard–Navier–Stokes Equations. _J. Sci. Comput._ , 101(2):36, 2024. 

33. S. Noelle, G. Bispen, K.R. Arun, M. Luk´aˇcov´a-Medviddov´a,[ˇ] and C.D. Munz. A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas dynamics. _SIAM J. Sci. Comput._ , 36(6), B989–B1024, 2014. 

34. C.W. Shu. High Order Weighted Essentially Nonoscillatory Schemes for Convection Dominated Problems. _SIAM Rev._ , 51(1):82–126, 2009. 

35. D.B. Siano. Layered sedimentation in suspensions of monodisperse spherical colloidal particles. _J. Colloid Interface Sci._ , 68(1):111–127, 1979. 

36. B.P. Vollmayr-Lee, and A.D. Rutenberg. Fast and accurate coarsening simulation with an unconditionally stable time step. _Phys. Rev. E_ , 68(6, 2), 2003. 

