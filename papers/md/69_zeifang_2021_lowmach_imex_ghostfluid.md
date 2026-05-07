Vol:.(1234567890)
Communications on Applied Mathematics and Computation (2023) 5:722–750
https://doi.org/10.1007/s42967-021-00137-2
1 3
ORIGINAL PAPER
A Low Mach Number IMEX Flux Splitting for the Level Set 
Ghost Fluid Method
Jonas Zeifang1,2   · Andrea Beck2,3
Received: 22 September 2020 / Revised: 6 April 2021 / Accepted: 14 April 2021 /  
Published online: 28 July 2021 
© The Author(s) 2021
Abstract
Considering droplet phenomena at low Mach numbers, large differences in the magnitude 
of the occurring characteristic waves are presented. As acoustic phenomena often play a 
minor role in such applications, classical explicit schemes which resolve these waves suffer 
from a very restrictive timestep restriction. In this work, a novel scheme based on a specific 
level set ghost fluid method and an implicit-explicit (IMEX) flux splitting is proposed to 
overcome this timestep restriction. A fully implicit narrow band around the sharp phase 
interface is combined with a splitting of the convective and acoustic phenomena away 
from the interface. In this part of the domain, the IMEX Runge-Kutta time discretization 
and the high order discontinuous Galerkin spectral element method are applied to achieve 
high accuracies in the bulk phases. It is shown that for low Mach numbers a significant 
gain in computational time can be achieved compared to a fully explicit method. Applica-
tions to typical droplet dynamic phenomena validate the proposed method and illustrate its 
capabilities.
Keywords  IMEX flux splitting · Level set method · Ghost fluid method · Low Mach 
number flows
Mathematics Subject Classification  65N30 · 65Z05 · 35Q35
 *	 Jonas Zeifang 
	
jonas.zeifang@uhasselt.be
	
Andrea Beck 
	
andrea.beck@ovgu.de
1	
Faculty of Sciences, Hasselt University, Agoralaan Gebouw D, BE‑3590 Diepenbeek, Belgium
2	
Institute of Aerodynamics and Gas Dynamics, University of Stuttgart, Pfaffenwaldring 21, 
70569 Stuttgart, Germany
3	
Laboratory of Fluid Dynamics and Technical Flows, Otto von Guericke University Magdeburg, 
Universitätsplatz 2, 39106 Magdeburg, Germany


723
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
1  Introduction
In many situations in nature or technical applications, more than one material phase 
is present. Often, two phases are separated by a distinct interface as it is the case for, 
e.g., rain drops or spray related processes. In such applications, the characteristic Mach 
number Ma = ‖u‖2
c  , relating the velocity of the convective phenomena u to the speed of 
sound c, often covers a wide range of values. Especially in the liquid phase, the Mach 
numbers are often low, whereas in the gaseous phase higher Mach numbers are pre-
sent. The simulation of such phenomena requires numerical methods which are capable 
of treating flows with a wide range of Mach numbers accurately (including the correct 
asymptotic scaling) and efficiency. Considering the characteristics of the Euler equa-
tions shows the difficulties numerical methods are facing. For large Mach numbers 
( Ma = O(1) ) the Euler equations are hyperbolic. In the limit ( Ma →0 ) they change 
their type to hyperbolic-elliptic. For small Mach numbers ( Ma ≪1 ), the speeds of the 
present characteristic waves differ tremendously: the acoustic waves are much faster 
than the convective phenomena. If acoustics plays a minor role, standard explicit meth-
ods are not efficient as they have to resolve the fast characteristic waves for stability 
reasons, leading to a prohibitively small timestep.
To overcome this timestep restriction, different approaches exist, mostly relying on 
the implicit time discretization: fully implicit time discretization for the modeling of 
two-phase flow are, e.g., proposed in [25, 31, 47]. Another class of schemes are pres-
sure-based methods, relying on the implicit solution of an elliptic equation for the pres-
sure. They can be seen as an extension of incompressible schemes and require a refor-
mulation of the equation of state (EOS), which is a non-trivial task for complex EOS. 
Pressure-based methods have been derived for multiphase flows, e.g., in [4, 15, 18, 26, 
35, 45]. Another option are implicit-explicit (IMEX) flux splitting schemes which are 
based on the separation of the fast acoustic and slow convective waves of the flux. The 
idea is to treat the fast waves implicitly, while the slow waves are treated explicitly. 
This splitting often serves as a starting point for the derivation of pressure-based meth-
ods, see, e.g., [13, 30, 51, 64]. The application of flux splittings with general EOS is 
often straightforward as the EOS does not have to be reformulated. Nevertheless, IMEX 
schemes have only rarely been applied to multiphase problems, see [7, 8, 55].
Low Mach number two-phase flows not only necessitate a specific design to tackle 
the different orders of magnitude of the characteristic waves. Additionally, the presence 
of two phases demands a modeling of the different states of matter and their coupling. 
For the modeling of multiphase flows mainly two concepts exist: diffuse and sharp 
interface approaches. Common approaches to capture the interface in a sharp interface 
scheme are the volume-of-fluid [33] and the level set method [54, 62]. The level set 
method has the advantage that the normals and the curvature of the phase interface are 
directly given by the derivatives of the level set function and hence the interface geom-
etry does not have to be reconstructed, as it would be the case for the volume-of-fluid 
method. For the modeling of the physical phenomena at the phase boundary, the ghost 
fluid method (GFM) has been introduced in [23]. In this work, we focus on a specific 
type of a sharp interface GFM based on the ideas of [49]. Here, a multiphase Riemann 
problem is solved at the phase boundaries. The idea has been adapted in [20, 21] and 
slightly modified in [37, 50]. A discontinuous Galerkin scheme [32] is used for the spa-
tial discretization of the bulk phases and at the phase boundary a finite volume sub-cell 
refinement [60] is applied. This framework will serve as a foundation of this work.


724
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
Since IMEX flux splitting schemes have been found to be well suited for low Mach 
number computations within a discontinuous Galerkin framework [72, 73], the idea in 
this work is to combine an IMEX flux splitting scheme with the sharp interface level set 
ghost fluid method (LSGFM). Although the fully implicit time discretization has been used 
within the LSGFM [31, 37, 47], IMEX flux splitting schemes have not been investigated 
in this context. As a flux splitting, the idea of Toro and Vázquez-Cendón (TV) [67] and its 
extension to general EOS [66] are used. Hence, the main novelty of this paper is the analy-
sis and presentation of how an IMEX flux splitting can be combined with a discontinuous 
Galerkin sharp interface LSGFM to overcome the restrictive timestep restriction of weakly 
compressible multiphase flows.
We will proceed as follows. In Sect. 2, the considered governing equations are summa-
rized. Following in Sect. 3, the IMEX LSGFM is described. Starting with a short summary 
on the level set ghost fluid framework in Sect. 3.1, it is described how IMEX flux splitting 
schemes can be applied in Sect. 3.2. Next, novel scheme is validated and its efficiency 
is investigated in Sect. 4. Following, some illustrative applications are shown in Sect. 5. 
Finally, in Sect. 6 conclusion and outlook are given.
2  Governing Equations
The idea of the sharp interface approach is to model two-phase flows as two pure bulk 
phases without a mixing zone. Inside the pure phases, the Euler equations are chosen to 
model the physical behavior of the fluids. In three dimensions they are given by
with density 휌 , velocity u = (u, v, w)T , total energy E, pressure p and the three dimensional 
identity matrix 퐈퐝 . The primitive state vector is given as wprim = (휌, u, p)T . The pressure is 
linked to the density and the specific internal energy 휖∶= E
휌−
‖u‖2
2
2  via an EOS. For gases, 
the ideal gas EOS
with the isentropic coefficient 훾 is chosen. Liquids are modeled with the stiffened gas EOS:
where the material-dependent parameter p∞ is used to model the increased stiffness of liq-
uids compared to gases. Note that the algorithm introduced in this work is not restricted to 
the use of such simple EOS. It is also possible to use more complex or tabulated EOS [24].
The phase interface separating the two phases is tracked by a level set function Φ , 
describing the signed distance to the phase interface. Hence its root describes the position 
of the phase interface. The level set function is transported with a velocity field 퐬 according 
to
(1)
𝜕tw + ∇x ⋅F(w) = 0
with
w ∶=
⎛
⎜
⎜⎝
𝜌
𝜌u
E
⎞
⎟
⎟⎠
and
F(w) ∶=
⎛
⎜
⎜⎝
𝜌u
𝜌u ⊗u + p ⋅퐈퐝
u(E + p)
⎞
⎟
⎟⎠
,
p(휌, 휖) = (훾−1)휌휖
p(휌, 휖) = (훾−1)휌휖−훾p∞,


725
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
The geometrical quantities of the phase interface, i.e., the normal vector 퐧Φ and the curva-
ture 휅Φ , are directly given by the derivatives of the level set field [17].
3  IMEX Flux Splitting for the LSGFM
In this section, first the compressible LSGFM according to [21, 37, 50] is briefly recalled. 
Following, it is outlined how IMEX flux splitting schemes can be used to enable efficient 
simulations at low Mach numbers.
3.1  The LSGFM
3.1.1  Domain Decomposition
The position of the phase interface is described by the zero position of the level set func-
tion. Hence, the sign of the level set field can be used as an identifier to divide the compu-
tational domain 훺 into a liquid 훺l and a gaseous domain 훺g . Discretely, the interface posi-
tion is shifted to the next boundary of the discretization elements, in the following called 
the surrogate surface. To obtain an accurate scheme, the high order discontinuous Galerkin 
spectral element method (DGSEM) [32, 41] is applied for the spatial discretization. The 
DGSEM is based on the polynomial approximation of the solution as a tensor-product of 
nodal one-dimensional Lagrange polynomials of degree N inside each element C of the 
computational mesh C ∈훺 . For this piecewise smooth function wh the weak formulation is 
then given by
for every polynomial test function 휙(x) of degree N. Across the element boundaries dis-
continuities are allowed and the elements are coupled via a numerical flux function f ∗ . 
Choosing the same nodal Lagrange polynomials of degree N as test functions and ansatz 
functions, leads to the DGSEM which is (N + 1)th-order accurate in space. A detailed 
description of the used DGSEM can be found in [32] and an overview on the actual 
implementation is provided in [42]. The core method is available as open source software 
FLEXI1.
Due to its solution representation, each grid element contains multiple (nodal) degrees 
of freedom (DOFs). In the LSGFM context this leads to a poor relation of discrete interface 
resolution per DOF. Therefore, a refinement of the elements adjacent to the phase interface 
is applied. For that purpose, the mixed finite volume (FV)/DG formulation from [60, 61] is 
used. Figure 1 illustrates this sub-cell refinement in the cells neighboring the phase inter-
face and the domain decomposition in a liquid and a gaseous domain 훺l and 훺g , respec-
tively which are separated by the surrogate surface 훤.
(2)
휕tΦ + 퐬⋅∇xΦ = 0.
(3)
휕
휕t ∫C
wh휙(x)dx + ∮휕C
f ∗
n휙(x)ds −∫C
F(wh) ⋅∇휙(x)dx = 0,
1  www.flexi-project.org, GNU GPL v3.0


726
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
3.1.2  Interface Treatment
Two-phase Riemann solver While in the pure bulk phases, no multiphase effects have to 
be considered, directly at the discrete phase interface a special treatment of the fluxes is 
necessary. The procedure at the phase boundary is based on the idea outlined in [49], to 
use the solution of a two-phase Riemann problem as a boundary condition for the pure 
phases. This approach has been modified in [20, 21] to use the approximate solution of 
the two-phase Riemann problem instead of the exact solution. In Fig. 1, positions on 
the surrogate surface, where two-phase Riemann problems are solved are marked with a 
small white square.
In this work, the HLLP Riemann solver from [20, 59] is used at the phase boundary. 
This Riemann solver is based on the well-known HLLC Riemann solver, where the con-
tact discontinuity “C” is considered as the phase boundary “P”. It enables the treatment 
of surface tension effects by considering the pressure jump of the Young-Laplace law 
in the jump conditions of the Riemann solver. From the solution of the HLLP Riemann 
solver three quantities can be extracted: fluxes for the liquid and the gaseous phase f ∗
l , 
f ∗
g , the velocity of the phase interface s# and so-called ghost states for the liquid and 
the gaseous phase. For a liquid cell adjacent to the interface, its ghost state is a gaseous 
state and vice versa. Here, the ghost states are chosen as the interchanged initial condi-
tions of the two-phase Riemann problem. They are used if the interface position moves 
and a sub-cell changes its assignment to the liquid or gaseous domain.
Note that the GFM is inherently non-conservative [23, 49]. This stems from the non-
equality of the fluxes at the interface f ∗
g ≠f ∗
l and from the definition of the new states 
if a sub-cell element changes its assignment from the liquid to the gaseous domain or 
vice versa. Attempts to overcome the non-conservativity of the GFM for incompress-
ible flows relying the divergence-free condition of the incompressible velocity field have 
been made in, e.g., [16, 52, 53]. For compressible flows, an attempt to improve the con-
servativity has been presented in [46]. The idea in [46] is to modify the novel states 
which are set when the interface moves. Nevertheless, it is not considered here as this 
modification can have an unfavorable influence on the stability.
Fig. 1   Schematic view of domain 
decomposition with the com-
pressible LSGFM into a liquid 
(dark gray) and a gaseous domain 
(light gray). Large squares denote 
the elements of the DG grid, 
while small squares denote the 
elements of the refined FV sub-
cell grid. The zero isocontour 
of the level set with its normal 
vector 퐧Φ is indicated with a 
dashed line and positions on 
the surrogate surface, where a 
two-phase Riemann problem is 
solved are marked with a small 
white square
nΦ
Ω
Ωg
l


727
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
Velocity extrapolation From the solution of the two-phase Riemann problem, the velocity 
at the phase boundary s# can be extracted. This pointwise available information is then used 
to construct a velocity field 퐬 for the transport of the level set field. This is done by solving a 
Hamilton-Jacobi-type equation [56]
with the numerical Hamiltonian
according to [19], with the componentwise absolute value of the level set normal |퐧Φ| and 
a numerical sign function sgn . The discretization of the up- and downwind gradients ∇+
x 
and ∇−
x is done with the HJ-WENO scheme from [36] and the pseudo time iteration is per-
formed with a 3rd order explicit Runge-Kutta method. More details on this can be found in 
[19].
Level set transport: The resulting velocity field is then used for the solution of Eq. (2). Dif-
ferently to the weak form of Eq. (1) given in Eq. (3), the variational form for the level set 
transport reads
As this is a non-conservative equation, the discretization has to be slightly adapted. Simi-
larly to the mixed DGSEM/FV formulation used for the bulk phases, the solution Φh is 
either represented with the tensor-product of nodal one-dimensional Lagrange polynomials 
of degree N (DGSEM) or with the piecewise constant refined FV representation. When 
choosing a path-conservative formulation for the discretization of the weak form introduces 
now modified numerical fluxes at the cell boundaries to account for the non-conservativity 
of the equation. Details on the derivation of this specific fluxes can be found in [37].
Level set reinitialization Unfortunately, the level set field does not retain its signed distance 
property during the advection and hence it has to be reinitialized. Again, a Hamilton-Jacobi 
equation
is solved with the WENO scheme from [36] and an explicit 3rd order Runge-Kutta scheme. 
The Hamiltonian Hreinit(∇xΦ) = sign(Φ)(|∇xΦ| −1) is approximated with the numerical 
Hamiltonian according to [17] by
where
휕휏퐬m + Hextvel(∇x퐬m) = 0,
for the directions m = 1, 2, 3
Hextvel(∇x퐬m) ≈HLF
extvel
(∇+
x sm, ∇−
x sm)
∶= 1
2
(sgn(Φ)(퐧Φ ⋅(∇+
x sm + ∇−
x sm)) −|퐧Φ| ⋅(∇+
x sm −∇−
x sm)),
휕
휕t ∫C
Φh휙(x)dx + ∮휕C
(퐬⋅∇Φh)휙(x)ds + ∫C휕C
(퐬⋅∇Φh)휙(x)dx = 0.
휕휏Φ + Hreinit(∇xΦ) = 0
Hreinit(∇xΦ) ≈HGod
reinit(∇+
x Φ, ∇−
x Φ)
∶= sgn(Φ)
 √
max(a1, b1) + max(a2, b2) + max(a3, b3) −1, if sgn(Φ) ⩽0,
√
max(c1, d1) + max(c2, d2) + max(c3, d3) −1,
else,
ai ∶=max(|(∇+
x Φ)i|2, 0),
bi ∶= −min(|(∇−
x Φ)i|2, 0),
ci ∶= −min(|(∇+
x Φ)i|2, 0),
di ∶= max(|(∇−
x Φ)i|2, 0).


728
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
The reinitialization and the velocity extrapolation procedure are proceeded in pseudo time 
until information from the level set zero position has propagated two to three mesh ele-
ments away from the phase interface.
Interface derivatives calculation As stated before, the geometrical quantities of the 
phase interface, i.e., normal vector and interface curvature, are directly given by the deriva-
tives of the level set field. It has been found to be beneficial to use second order finite dif-
ferences for the calculation of the derivatives, although the level set field is discretized with 
higher order polynomials [48]. This approach is also pursued here, i.e., the gradients of the 
level set, required for the normals of the phase interface 퐧Φ = ∇xΦ∕|∇xΦ| , are discretely 
calculated at a DOF ijk with
where Φx1 denotes the derivative of the level set in x1-direction. The level set normal vector 
is used to transform the states for the solution of the two-phase Riemann solver such that 
the Riemann problem is solved in the phase normal direction, i.e., in the normal direction 
of the physical phase interface. This ensures that the pressure jump introduced by the sur-
face tension acts in correct direction. The pressure jump is calculated via Young-Laplace 
law using the curvature of the phase interface
where Φx1x2 denotes the derivative 휕2Φ
휕x1휕x2 and is obtained by applying the finite difference 
operator in Eq. (4) twice.
These procedures now serve as building blocks for the construction of the IMEX 
LSGFM, which is introduced in the next subsection.
3.2  The IMEX LSGFM
The idea of IMEX flux splittings is to separate the fast (acoustic) and the slow (convective) 
waves. To obtain a timestep restriction independent of the fast waves, they are then treated 
implicitly and the slow waves are treated explicitly. Fast waves are presented in Eq. (1) but 
not in Eq. (2). Therefore, the level set transport is treated purely explicitly and only for the 
time discretization of the Euler equations and an IMEX flux splitting scheme is applied.
3.2.1  Overview of the Algorithm
In Fig. 2, it is illustrated how the different building blocks can now be assembled to a 
consistent algorithm, where the level set transport is handled explicitly. In a first step fully 
(4)
∇xΦijk = (Φx1, Φx2, Φx3
)T
≈
(Φ(i−1)jk −Φ(i+1)jk
2Δxijk
,
Φi(j−1)k −Φi(j+1)k
2Δxijk
,
Φij(k−1) −Φij(k+1)
2Δxijk
)T
,
휅Φ ∶=
Φ2
x1Φx2x2 −2Φx1Φx2Φx1x2 + Φ2
x2Φx1x1
|∇xΦ|3
+
Φ2
x1Φx3x3 −2Φx1Φx3Φx1x3 + Φ2
x3Φx1x1
|∇xΦ|3
+
Φ2
x2Φx3x3 −2Φx2Φx3Φx2x3 + Φ2
x3Φx2x2
|∇xΦ|3
,


729
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
implicit time discretization is applied to the flow field. The separation of the fast and slow 
waves using a flux splitting and leading to an IMEX discretization of the flow field is 
described thereafter.
	
(i)	 Update the DG/FV distribution of the level set field according to the information 
about the position of the interface.
	 (ii)	 Advance the level set field explicitly in time with the velocity field 퐬 to obtain Φn+1 
by solving Eq. (2).
	 (iii)	 Define 훺l and 훺g at the new time instance tn+1 according to the sign of the level set 
field. Use the ghost states from the old time instance tn if an assignment of a sub-cell 
has changed.
	 (iv)	 Repeat step (i) for the flow solver.
	 (v)	 Reinitialize the level set field to recover its signed distance property.
	 (vi)	 Calculate the geometrical properties 퐧n+1
Φ  and 휅n+1
Φ  of the phase interface at the new 
time instance.
	(vii)	 Iteratively advance the flow field using the implicit/IMEX time discretization.
–	 Use the geometrical information 퐧n+1
Φ  and 휅n+1
Φ  for the solution of the two-phase Rie-
mann problems to obtain f ∗
l , f ∗
g , s# and the ghost states at the new time instance tn+1.
Flow solver: w
implicit/IMEX
Interface resolution: Φ
explicit
tn
tn+1
Advance Φ
Update FV/DG
distribution
s
Evaluation of
level set root
Domain decompo-
sition into Ωv, Ωl
Update FV/DG
distribution
Reinitialize Φ
Compute interface
derivatives nΦ, κΦ
Solve two-phase
Riemann problem
nΦ
κΦ
Advance bulk phases
f∗, f∗
Iterate
Extend velocity ﬁeld
s#
s
g
l
Fig. 2   Schematic overview of IMEX LSGFM. Steps that are done in each Runge-Kutta stage are marked 
with dark gray, steps that are repeated only once per timestep are marked with light gray


730
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
–	 Use the fluxes from the HLLP solver f ∗
g and f ∗
l as boundary conditions for the bulk 
phases and advance the flow field (Eq. (1)).
	(viii)	 Construct the velocity field 퐬 for the level set transport from s# , when the iterative 
procedure from step (vii) has converged.
	 (ix)	 Repeat steps (i)–(viii) until the final time tend is reached.
This quasi-serial advancement of the level set function and the flow field ensures that the 
interface position of the initial guess for the implicit method is already at the correct loca-
tion for tn+1 . Note that this coupling is first order in time. Nevertheless, the errors of the 
temporal discretization can be reduced by applying Runge-Kutta methods for the level set 
transport and the advancement of the flow field.
3.2.2  TV Flux Splitting
If a proper splitting of the Euler equations  (1) is chosen and coupled with the above 
described algorithm, this allows timesteps being independent of the fast acoustic waves. 
There are a lot of options of how to split the flux of the Euler equations into an implicitly 
̃F(w) and explicitly treated flux ̂F(w) with F(w) = ̃F(w) + ̂F(w) , see, e.g., [5, 12, 29, 67, 
74]. In this work, the TV splitting approach according to [67] is used due to its simplicity 
and its ability to be applied to general EOS [66]. It splits the flux into a pressure system 
̃F(w) and an advection system ̂F(w) via
According to [66], the eigenvalues of the implicit subsystem are given by
with
For the explicit subsystem the eigenvalues can be calculated as
From this one can see, that only convective phenomena influence the timestep if only the 
advection system is treated explicitly as the timestep of the DGSEM is given by
for a characteristic length scale lref of the discretization elements and CFL ⪅1 . For the 
DGSEM, the CFL number is typically scaled with an empirically obtained constant which 
is specific for the chosen Runge-Kutta method and polynomial degree, see, e.g., [28]. In 
(5)
F(w) ∶=
⎛
⎜
⎜⎝
0
p ⋅퐈퐝
u(𝜌𝜖+ p)
⎞
⎟
⎟⎠
and
F(w) ∶=
⎛
⎜
⎜⎝
𝜌u
𝜌u ⊗u
1
2𝜌u‖u‖2
2
⎞
⎟
⎟⎠
.
̃휆1 = 1
2(u ⋅n −A),
̃휆2,3,4 = 0,
̃휆5 = 1
2(u ⋅n + A),
(6)
A ∶=

‖u‖2
2 + 4

휖
휌+ p
휌2

휕휖
휕p
−1
.
̂휆1 = 0,
̂휆2,3,4,5 = u ⋅n.
(7)
Δt =
(
CFL ⋅lref
|̂휆|max(2N + 1)
)
min훺
,


731
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
this work we do not use optimized scaling factors for the IMEX Runge-Kutta methods and 
simply choose the scaling factor 1. Therefore, the stability limit is not strictly CFL = 1 but 
can be slightly below or above.
Additionally, a numerical flux function for the subsystems is required. We choose the 
standard TV flux proposed in [66] or the less dissipative TVHLL-LM Riemann solver 
which are given in Appendix A.
3.2.3  Spatial Discretization
With the TV flux splitting only the flux in the bulk phases can be split into an explicit and 
an implicit part. For the flux obtained by the two-phase Riemann solver ( f ∗
g , f ∗
l ), this is not 
possible for general two-phase Riemann solvers as physical modeling of two-phase effects 
is considered. This can include surface tension such as, e.g., in [20, 59] and phase transi-
tion, see, e.g., [22, 34]. Therefore, one can either treat this flux fully implicitly or fully 
explicitly. The latter would cause a timestep restriction depending on the acoustic waves 
and hence this flux is treated fully implicitly. Moreover, sub-cell elements that change their 
assignment from the liquid to the gaseous domain or vice versa cannot be treated partially 
implicit and partially explicit. Therefore, the cells located around the phase interface also 
have to be treated fully implicitly. This is realized with a narrow band of two elements adja-
cent to the interface in each direction, where no flux splitting is applied. This is illustrated 
in Fig. 3. Away from the interface, the TV splitting is used, indicated with the hatched area 
in Fig. 3. Inside the fully implicit narrow band, a finite volume sub-cell discretization is 
applied, whereas the DGSEM is used in the remaining IMEX flux splitting domain.
For the coupling of the IMEX flux splitting and the fully implicit domain, a special 
treatment is necessary. In analogy to domain-based IMEX methods, see, e.g., [38], the flux 
at the interface is calculated separately for both domains. That means that, during the flux 
calculation the neighboring domain is assumed to belong to the same type of the domain as 
the current element. With this, the numerical fluxes can be calculated as
Ω
Ω
f ∗
f ∗
implicit = f ∗(wn+1
L
, wn+1
R
)
f ∗
IMEX = f
∗(wn+1
L
, wn+1
R
)
+f
∗(wn
L, wn
R)
l
g
Fig. 3   Schematic view of domain decomposition with the IMEX LSGFM. The zero isocontour of the level 
set is indicated with a dashed line. Hatched areas indicate the domain, where the TV splitting is applied. In 
the non-hatched domain, the Euler equations are treated fully implicitly. The fluxes at the interface of both 
domains is treated either fully implicitly (for the non-hatched domain) or partially implicitly and partially 
explicitly (for the hatched domain)


732
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
With this approach conservativity at the coupling interface cannot be guaranteed. However, 
since the ghost fluid method is not conservative by construction, this influence is supposed 
to be small. In Sec. 4.3.2 this is experimentally justified (see Fig. 9 right).
Note that choosing the TV flux instead of the TVHLL-LM flux in the sub-cell directly 
neighboring the phase interface, has a favorable influence on the stability. Therefore, in 
those sub-cells always the TV flux is chosen as a Riemann solver. In the remaining domain 
one can either use the TV or the TVHLL-LM flux.
3.2.4  Temporal Discretization
In Fig. 2, the first order in time procedure has been shown. To increase the temporal accu-
racy, IMEX Runge-Kutta schemes [39] can be used for the advancement of the bulk phases 
and the level set field. Note that the coupling of both fields is still done only once per 
timestep and hence the global order of accuracy is not increased. Nevertheless, one can 
expect more accurate results by using higher order IMEX Runge-Kutta schemes.
In this work, globally stiffly accurate IMEX Runge-Kutta methods from [39] are used. 
The time integration procedure in semi-discrete form for a scheme with s stages, integrat-
ing from tn to tn+1 , can be written as 
	
(i)	 for i = 1, ⋯, s , solve for wn,i
	 (ii)	 set wn+1 ∶= wn,s.
The coefficients ̃Aij , ̂Aij , ̃cj and ̂cj are given in the Butcher table of the chosen method. Note 
that the second step only holds due to the globally stiffly accurate property of the chosen 
IMEX Runge-Kutta methods. The arising non-linear equation system
with bn,i denoting the sum of all values known from previous stages, is solved with a Jac-
obian-free Newton-GMRES method, see, e.g., [40] for an overview. Newton’s method for 
the solution of Eq. (8) requires an initial guess w0 and a user defined tolerance 휖Newton and 
is given by 
	
(i)	 for k = 0, 1, ⋯ , solve 
	 (ii)	 set wn,i ∶= wk+1 when ‖g(wk+1)‖2 ⩽휖Newton‖g(w0)‖2.
f ∗
IMEX ∶= ̃f
∗(wn+1
L
, wn+1
R
, n) + ̂f
∗(wn
L, wn
R, n),
f ∗
implicit ∶= f ∗(wn+1
L
, wn+1
R
, n).
wn,i −wn + Δt
( i∑
j=1
̃Ai,j∇x ⋅̃F(wn,j, tn + ̃cjΔt) +
i−1
∑
j=1
̂Ai,j∇x ⋅̂F(wn,j, tn + ̂cjΔt)
)
= 0;
(8)
0 =
(
I −Δt̃Ai,i∇x ⋅̃F
)
wn,i −bn,i =∶g(wn,i),
(9)
dg(wk)
dwk Δwk = −g(wk), wk+1 = wk + Δwk;


733
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
As initial guess, the solution at the previous stage is chosen, i.e., w0 ∶= wn,i−1 . The linear sys-
tem in Eq. (9) is then solved with a Jacobian-free GMRES method to ensure a small mem-
ory footprint. An analysis on the memory requirements of a matrix-free and a matrix-based 
approach for the DGSEM can be found in [68]. The arising Jacobian-vector product in the 
GMRES method is approximated via a first or a second order finite difference
thus requiring one or two evaluations of the spatial operator per iteration of the linear 
solver. Note that g(wk) can be stored and only has to be evaluated once per Newton itera-
tion. The step sizes of the finite differences are chosen along the ideas from [1] and are 
given by
with the machine accuracy 휖machine and the free parameter 휖scale.
To accelerate the convergence of the GMRES method, a preconditioner is applied. As 
the purpose of the preconditioner is to approximate the Jacobian dg(w)∕dw but being much 
simpler to invert, an elementwise block-Jacobi preconditioner is applied. It accounts for 
inner-element dependencies but neglects inter-element dependencies. Hence, it consists of 
one independently invertible matrix for each element. More details on the derivation of the 
block-Jacobi preconditioner for the mixed DG/FV sharp interface formulation can be found 
in [71].
Timestep restriction If no surface tension effects are considered, the timestep of the 
method is given by Eq. (7). Otherwise, the capillary timestep restriction [14] has to be 
considered as well. Taking this constraint additionally into account and following [14], the 
timestep is given by
where the capillary eigenvalue 휆휎 can be calculated as
with the surface tension coefficient 휎 and the characteristic length of the FV sub-cells 
l∗
ref = lref∕(N + 1) . Note that the capillary timestep constraint only has to be evaluated at FV 
sub-cell DOFs directly adjacent to the surrogate surface 훤 . Differently to [14], we consider 
the total velocity of the flow at the interface |u|훤 and not only the velocity component tan-
gential to the phase interface.
(10)
⎧
⎪
⎨
⎪⎩
dg(wk)
dwk Δwk ≈
g(wk+ΔFDO1Δwk)−g(wk)
ΔFDO1
,
dg(wk)
dwk Δwk ≈
g(wk+ΔFDO2Δwk)−g(wk−ΔFDO2Δwk)
2ΔFDO2
,
(11)
ΔFDO1 ∶=

휖machine휖2
scale
‖Δwk‖2
and
ΔFDO2 ∶=
3
휖machine
2
휖2
scale
‖Δwk‖2
,
Δt = CFL ⋅min
(
lref
|̂휆|max(2N + 1)
,
l∗
ref
휆휎
)
min훺
,
휆휎=
√
2π휎
휌ll∗
ref
+ |u|훤


734
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
4  Validation and Efficiency of the IMEX LSGFM
In this section, the novel IMEX LSGFM is validated and the performance of the scheme 
is investigated. In particular, the separation of fast and slow waves and the semi-implicit 
interface coupling is validated and the efficiency gain compared to a fully explicit scheme 
is evaluated.
4.1  Validation of Domain Coupling
First, the correct coupling of the fully implicit and the IMEX domain is validated for one-
dimensional settings.
Acoustic wave interaction The first setup is taken from [15] and describes a single 
acoustic pulse which interacts with a phase boundary of water and air. With this setup, 
inconsistencies in the discretization, especially in the coupling of the different parts of the 
domain, would show up as they can have a large influence on amplitudes and propagation 
speeds of the occurring waves. The velocity of the phase boundary itself is small, such that 
no discrete interface movement takes place.
At the left boundary of the domain 훺= [0, 1.5 m] , a small velocity pulse with the fre-
quency f = 5 000 s−1 is prescribed by
The perturbation of the initial velocity V0 = 1 m ⋅s−1 is chosen as Δu = 0.02V0 . The 
remaining initial data are given by w0
prim = (1.157 kg ⋅m−3, V0 −Δu, 105 Pa)T for air and 
w0
prim = (998 kg ⋅m−3, V0 −Δu, 105 Pa)T for water. A Dirichlet boundary condition is pre-
scribed on the right of the domain. On the left side of the domain air is presented with 
훾air = 1.4 and on the right side the water is modeled as a stiffened gas with 훾water = 4.1 and 
p∞= 4.4 × 108 Pa . The interface is located at x1 = 0.5 m . With this initial data, the initial 
Mach number of the air is Ma ≈2.8 × 10−3.
We discretize the domain with 175 elements with N = 5 , couple them with the standard 
TV flux and choose the 3rd order IMEX-ARK3 scheme from [39] for the temporal discre-
tization. As a comparison, the explicit scheme according to [37] with the explicit 3rd order 
low storage Runge-Kutta scheme from [69] is chosen.
According to [15], the ratio between the amplitude of the transmitted and the reflected 
acoustic wave is Δptrans.
water∕Δpreﬂ.
air = 2.001 . We perform calculations with CFL = 1 with 
respect to the acoustic eigenvalue and a simulation with the timestep increased by a factor 
of 100. The results of the simulations are depicted in Fig. 4.
If CFL = 1 is chosen, a comparison with the explicit scheme is possible. One can hardly 
distinguish the two solutions (upper part of Fig. 4) and both have a similar ratio of the trans-
mitted and reflected waves Δptrans.
water∕Δpreﬂ.
air = 2.016 , which matches the analytical solution 
very well. Thus, our novel method reproduces the expected behavior and predicts the correct 
dynamics, if the associated waves are resolved sufficiently. For the increased timestep no cal-
culations with the explicit scheme are possible and deviations from the afore observed dynam-
ics are to be expected for the IMEX method (lower part of Fig. 4). One can see that due to 
the large timestep the dissipation and dispersion of the waves are increased. Nevertheless, the 
ratio of the waves still matches the analytical solution with Δptrans.
water∕Δpreﬂ.
air = 2.007 with high 
accuracy.
u =
{
V0 + Δu sin
(
2πft + 3π
2
)
,
if t < f −1,
V0 −Δu,
else.


735
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
Accelerated droplet The next setup, taken from [3], describes the acceleration of a liquid 
due to a pressure difference in the surrounding gas. Differently to the previous setup, the phase 
interface moves and thus allows further validation of the IMEX LSGFM.
On the domain 훺= [0, 1 m] with Dirichlet boundary conditions, the initialization is given 
by w0
prim = (1 kg ⋅m−3, 0 m ⋅s−1, p0)T with
The gaseous phase ( x1 < 0.4 m or x1 > 0. 6 m ) is modeled as an ideal gas with 훾gas = 1.4 
and the liquid phase as a stiffened gas, using 훾liquid = 7.15 and p∞= 3 308 Pa . The dis-
cretization is done with 100 elements with N = 3 , TV flux and the 3rd order Runge-
Kutta schemes from [39] and [69]. For the explicit scheme, the timestep is restricted with 
CFL = 1 , whereas for the IMEX scheme the timestep is increased by a factor of 150. At 
the final time tend = 0.065 s , the maximum Mach number in the domain is Ma ≈0.25 and 
Ma ≈2 × 10−3 for the gaseous and the liquid phase, respectively.
The results of the computations are shown in Fig. 5. It is shown that the solutions 
in density and velocity are indistinguishable for the explicit and the IMEX scheme. 
Only for the pressure, differences are visible. While the explicit scheme predicts fast 
p0 =
⎧
⎪
⎨
⎪⎩
2.5 Pa,
if x1 < 0.4 m,
1.0 Pa,
if x1 > 0.6 m,
1.75 Pa,
else.
0
0.5
1.0
1.5
0
5
10
15
Interface
∆p /Pa
CFL = 1
Explicit
IMEX
0
0.5
1.0
1.5
0
5
10
15
Interface
x1/m
CFL = 100
IMEX
∆p /Pa
Fig. 4   Pressure pulses before ( t = 1 ms , dashed) and after ( t = 2 ms , continuous) interaction with the phase 
interface at x1 = 0.5 m . Upper figure shows calculations with CFL = 1 for explicit and IMEX scheme and 
lower figure with a 100 times larger timestep for the IMEX scheme


736
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
traveling waves inside of the droplet, the IMEX scheme predicts an almost linear pres-
sure profile. This is to be expected as the IMEX scheme does not temporally resolve 
those waves, instead the linear pressure distribution of an incompressible material [3] is 
approximated.
4.2  Validation of Geometrical Coupling
Further validation of the novel scheme is provided by the simulation of the free drop-
let oscillation. This setup is particularly well suited for the validation of the consistent 
coupling of the different building blocks visualized in Fig. 2, as the whole dynamics is 
driven by the forces at the interface. Hence, a consistent coupling of interface proper-
ties, level set advection and the bulk phases is crucial.
The test setup describes an ellipsoidal shaped droplet at rest in a periodic domain 
훺= [−2 m, 2 m]3 . Due to surface tension, the droplet starts to oscillate. The initial data 
of the level set field is given by
and the remaining initial data are summarized in Fig. 6, with 휎 denoting the surface tension 
coefficient.
The choice of the material parameters ensures that the ratios of the speeds of sounds 
are similar to the ones given in [57] for water and air. For this setup the maximum 
occurring Mach numbers are Ma ≈5 × 10−3 and the analytical oscillation period for the 
lth mode can be calculated with the formula by Lamb [44]
Φ0(x) =
x2
1.252 + y2
12 + z2
12 −1,
T =
2π

r3
equi(l휌air + (l + 1)휌water)
√
휎(l −1)l(l + 1)(l + 2)
,
0
0.2
0.4
0.6
0.8
1.0
0.9
1.0
1.1
1.2
1.3
x/m
ρ /kg.m−3
Explicit
IMEX
0
0.2
0.4
0.6
0.8
1.0
0
0.10
0.15
0.20
0.25
0.30
x/m
u /m.s−1
Explicit
IMEX
0
0.2
0.4
0.6
0.8
1.0
1.0
1.2
1.4
1.6
1.8
2.0
2.2
2.4
2.6
x/m
p /Pa
Explicit
IMEX
0.05
Fig. 5   Density (left), velocity (middle) and pressure (right) distribution for accelerated one dimensional 
droplet at tend = 0.065 s for explicit and IMEX scheme. Dashed lines indicate the phase interface position 
and dotted lines indicate the interface between the fully implicit and the IMEX flux splitting domain


737
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
with the radius of the droplet in equilibrium requi . For the given initial data, the oscillation 
period of the second mode is T = 5 s.
A grid convergence study is performed to validate the novel scheme. For that purpose, 
grids with 153 , 203 and 303 elements with N = 3 and TV or TVHLL-LM flux are used. 
The temporal discretization is done with the 4th order scheme IMEX-ARK4 from [39]. 
Instead of calculating the capillary timestep [14], the maximum timestep is prescribed with 
Δtmax = 10−3 s for the coarse meshes and Δtmax = 7 × 10−4 s for the finest mesh.
Figure 7 shows the results in kinetic energy Ekin = ∫훺
1
2휌‖u‖2
2d훺 (left) and the mass 
loss (right). One can note that for a finer resolution the numerical viscosity is decreased 
and hence the damping of the oscillation is reduced, resulting in a smaller oscillation 
period. Considering the mass loss, also a convergent behavior can be observed, i.e., the 
mass loss decreases for a finer resolution. The figure shows that choosing the TVHLL-
LM Riemann solver instead of the TV Riemann solver significantly improves the solution. 
While the mass loss is almost the same, much less numerical dissipation is introduced, 
resulting in a more accurate solution.
Fig. 6   Initial data of three dimen-
sional test setup of oscillating 
droplet
ρ0
air = 1.157 kg . m−3
u0
air = 0 m . s−1
p0
air = 1 000 Pa
γair = 1.4
ρ0
water = 998 kg . m−3
u0
water = 0 m . s−1
p0
water = 1 456.7 Pa
γwater = 4.1
p∞= 5.7 106 Pa
σ = 246 J . m−2
×
/
/
/
/
M
Fig. 7   Temporal evolution of kinetic energy (left) and mass loss in percent (right) of surface tension-driven 
droplet oscillation, using different mesh resolutions. The simulation with the TVHLL-LM Riemann solver 
is indicated with “LM”


738
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
A quantitative analysis of the convergence study is provided in Table 1. One can see 
that the numerical solution approaches the analytical solution of T = 5 s and the mass loss 
decreases under grid refinement.
Summing up, the presented studies validate the novel scheme. Investigations on the effi-
ciency are provided in the next section.
4.3  Evaluation of the Efficiency
4.3.1  Influence of the Approximation of the Jacobi‑Vector Product
Choosing implicit time discretization opens the field of optimization of the (non-) linear 
solver parameters. One of the DOFs of the implicit method is the definition of the finite dif-
ference, approximating the Jacobian-vector product of the GMRES method, see Eq. (10). 
Here, one can choose a first or a second order finite difference and the step size of the finite 
difference (Eq. (11)). In literature, several options how to choose the step size can be found 
in Eq. (12)
with nDOF denoting the total number of DOFs. For the present scheme, the simple choice 
휖(1)
scale ∶= p0
air , denoting the background pressure in the gaseous domain, is proposed. The 
choice of the scaling factor is a trade-off between truncation error of the finite difference 
and round-off errors due to limited machine accuracy.
A comparison between the different options is provided by the analysis of the required 
iterations for a certain amount of timesteps. For that purpose, the setup from the previ-
ous Sect.  4.2 with 203 elements, N = 3 and TV flux is used. The amount of iterations 
are counted to advance the solution from t = 2 s for 50 timesteps with IMEX-ARK4 
and Δt = 10−3 s each. The relative convergence tolerance of Newton’s method is set to 
휖Newton = 10−4 and the relative convergence tolerance of the linear solver 휖GMRES is varied. 
Two sets of simulations are executed: one with the parameters summarized in Fig. 6 and 
one with an increased background pressure of p0
air = 105 Pa , p0
water = 1.004 567 × 105 Pa 
and p∞= 5.7 × 10−8 Pa , decreasing the occurring Mach numbers by a factor of 10. This 
allows an evaluation of the influence of the Mach number on the iterations. The results of 
the simulations are shown in Fig. 8.
(12)
⎧
⎪
⎨
⎪⎩
휖(1)
scale ∶= p0
air,
휖(2)
scale ∶=
√
1 + ‖wk‖2 [40],
휖(3)
scale ∶= wkT ⋅Δwk [12],
휖(4)
scale ∶= ‖wk‖2 + √휖machine‖Δwk‖2 [70],
휖(5)
scale ∶= 1 +
1
nDOF
∑nDOF
i
wk
i  [40], 휖(6)
scale ∶= √nDOF [11],
Table 1   Oscillation period and its relative error (left) and relative mass loss for the surface tension-driven 
droplet oscillation with different mesh resolutions after one period: results with TV Riemann solver (upper) 
and TVHLL-LM Riemann solver (lower)
Oscillation period T/s
Relative error/%
Mass loss/%
nelems = 153, TV
5.94
17.2
≈2.6
nelems = 203, TV
5.55
11.0
≈0.5
nelems = 303, TV
5.26
5.20
≈0.3
nelems = 203, TVHLL-LM
5.18
3.60
≈0.4


739
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
One can see that the choice of 휖scale strongly influences the required amount of itera-
tions. For the larger Mach number almost no differences between the different options are 
visible, except for the choices 휖(4)
scale and 휖(6)
scale , which require more iterations. In those two 
cases, the second order finite difference decreases the number of iterations. For the smaller 
Mach number, it is only possible to obtain a solution with 휖(1),(2),(5)
scale
 . With the remaining 
options, convergence of the linear solver was not obtained with a maximum of 500 itera-
tions per Newton step. From Fig. 8 one can conclude that with 휖(1)
scale or 휖(2)
scale a second order 
finite difference is not necessary for both considered setups. As the calculation of 휖(1)
scale does 
not require any further computational effort, it is chosen for the following computations.
4.3.2  Comparison with Fully Explicit Scheme
Finally, a comparison with a fully explicit scheme regarding the required computational 
time is presented. With this comparison, the influence of the block-Jacobi preconditioner 
and the chosen timestep size on the computational resources is evaluated.
Again, the setup from Sec. 4.2 is taken with 휖Newton = 10−3 , 휖GMRES = 10−1 and the CFL 
number is varied from CFL = 8 × 10−4 to CFL = 8 × 10−2 . Note that the CFL number is 
calculated with the maximum convective eigenvalue ̂휆max = |u ⋅n|max . The explicit refer-
ence scheme uses a 4th order low storage Runge-Kutta scheme from [6] for the time dis-
cretization with CFL = 0.9 with respect to the acoustic eigenvalue. A comparison of the 
computational results is provided in Fig. 9.
It is clearly visible that the simulation results with the different CFL numbers and the 
explicit schemes are almost not distinguishable. Due to the similar solution quality those 
setups can be used to evaluate the influence of the timestep size on the required GMRES 
iterations and computational time.
For that purpose, five simulations from t = 1.99 s to t = 2.04 s are executed for each con-
figuration on one node of the HPE “Hawk” system with 128 cores and the mean value is 
taken for the comparison. This choice leads to approximately 4 000 DOFs per core, what 
10−4
10−3
10−2
10−1
0.5
1.0
1.5
2.0
104
ϵGMRES
GMRES iterations
p0
air = 103 Pa
ϵ(1)
scale
ϵ(2)
scale
ϵ(3)
scale
ϵ(4)
scale
ϵ(5)
scale
ϵ(6)
scale
10−4
10−3
10−2
10−1
2
3
4
5
6
104
ϵGMRES
GMRES iterations
p0
air = 105 Pa
ϵ(1)
scale
ϵ(2)
scale
ϵ(5)
scale
×
×
Fig. 8   Required GMRES iterations to advance the surface tension-driven droplet from t = 2 s for 50 
timesteps with Δt = 10−3 s , using different convergence tolerances for the linear solver. Continuous lines 
indicate a first order finite difference and dashed lines indicate a second order finite difference for the 
approximation of the Jacobian-vector product


740
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
has turned out to be a good compromise between processor local work, communication and 
required memory.
First, the iterations of the linear solver are considered. Figure 10 (left) shows that the 
preconditioner has only a small influence on the convergence properties of the linear solver 
for small timesteps. For larger timesteps the block-Jacobi preconditioner allows to reduce 
the required amount of iterations significantly. While the required iterations are almost 
constant without preconditioner, the iterations are reduced for larger timesteps if the pre-
conditioner is chosen. The amount of evaluations of the spatial operator for the explicit 
scheme is marked with blue. One can see, that with the IMEX scheme with large timesteps 
up to a factor of ≈2 less operator evaluations than with the explicit scheme are necessary.
Considering the required wall-clock time as a measure of efficiency shows that large 
speed-ups can be obtained using the novel IMEX scheme. Figure 10 (right) shows that 
a speed-up up to a factor of approximately 25 can be obtained compared to the explicit 
scheme. This large gain in wall-clock time is mainly due to the reduced amount of expen-
sive operations that have to be executed in each timestep: velocity extrapolation, the evalu-
ation of the interface geometry (i.e., 퐧Φ and 휅Φ ) are done once per timestep and the level set 
reinitialization is done only each second timestep. As those operations have a significant 
/
/
/
/
M
Fig. 9   Kinetic energy (left) and relative mass loss (right) until tend = 2 s for surface tension-driven drop-
let oscillation using different timestep sizes and explicit reference scheme. Solution of explicit scheme and 
IMEX scheme with different CFL numbers are almost not distinguishable
10−3
10−2
10−1
103
103.5
104
Explicit
CFL
GMRES iterations
w/o preconditioner
With preconditioner
10−3
10−2
10−1
10−1
100
Explicit
CFL
Relative wall-clock time
w/o preconditioner
With preconditioner
Fig. 10   Required linear solver iterations (left) and relative computational time (right) to advance the surface 
tension-driven droplet oscillation from t = 1.99 s to t = 2.04 s . Wall-clock times are relative to the explicit 
reference scheme (marked with blue)


741
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
influence on the overall computational time, large timesteps are a key feature to reduce the 
computational costs. Additionally, one can see that also here the use of the block-Jacobi 
preconditioner is beneficial for large timesteps. Although for small timesteps the computa-
tional time is increased due to the costs of building and applying the preconditioner, this is 
counterbalanced for large timesteps by the fewer iterations.
Concluding, the investigations in this section have shown that the novel IMEX LSGFM 
allows for an efficient and accurate simulation of low Mach number droplet dynamic phe-
nomena. Compared to a fully explicit scheme, large gains in computational time can be 
achieved if large timesteps are chosen.
5  Application of the IMEX LSGFM
In this section, the novel scheme is applied to two more typical applications of low Mach 
number droplet dynamics to illustrate its capabilities.
5.1  Drop Impingement onto a Deep Pool
A typical field of low Mach number droplet dynamics is the impingement of droplets onto 
pools of various depth. In [58] a two dimensional numerical study on the influence of the 
Weber number, pool depth and impingement angle has been carried out. We utilize one 
specific setup of this study to illustrate and validate the capabilities of the IMEX LSGFM.
The impingement Weber number, defined as
with radius r, density and velocity of the droplet of the simulation setup is We = 217 . Grav-
ity with the gravitation constant g = 9.81 m ⋅s−2 is taken into account. The initial data and 
the boundary conditions of the computational domain are summarized in Fig. 11.
The domain 훺= [−20r, 20r] × [−5r, 15r] is discretized with 400 × 200 elements 
with N = 2 , which are coupled with the TVHLL-LM flux. The pool surface is located 
at x = 0 m and has a depth of 5r and the droplet center is initialized 2r above the pool 
surface. With the parameters summarized in Fig. 11, an initial Mach number of approxi-
mately Ma = ‖u0‖2drop∕cair = 1.7 × 10−2 can be calculated. The characteristic momentum 
(휌u)∗∶= ‖휌u‖2∕‖(휌0u0)water‖2 and the characteristic timescale 휏∗∶= 2r∕‖u0‖2 are used to 
relate the occurring quantities to the initial conditions. The temporal discretization is done 
with the IMEX-ARK3 [39]. For stability reasons, the CFL number had to be reduced to 
CFL = 0.1 for this example. Results of the simulation are shown in Fig. 12.
The figure shows that when the droplet hits the pool surface a jet is formed; in three dimen-
sions named crown. Following, the crown height and the crater depth increase and a satellite 
droplet is ejected. This has also been observed in [58]. At t = 13.5휏∗= 0.99 ms the crater 
depth, the crown height and the angle of the crown are measured. A comparison with the 
results from [58], provided in Table 2, shows a good agreement. Deviations, especially in the 
crater depth and crown height are most likely due to the neglected viscous effects. Moreover, 
deviations caused by different numerical resolutions are possible as the number of DOFs is not 
reported in [58].
We ∶= 2r휌‖u‖2
휎
,


742
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
5.2  Binary Droplet Collision
A comparison with experimental data is possible for a three dimensional setup, where viscos-
ity plays a minor role. This is the case for the binary head on collisions in [2] with a colli-
sion Weber number of We = 23 . For this Weber number one can expect the coalescence of 
the droplets followed by a separation without ejecting further satellite droplets [2], see [63] 
for incompressible simulations of this setup. The initial data are summarized in Fig. 13. The 
domain 훺= [0, 1 m]3 is discretized with 303 elements with N = 2 and describes one eighth 
of the total domain. At the boundaries, symmetry boundary conditions are prescribed and 
the TVHLL-LM flux is chosen as numerical flux function. With the parameters summarized 
in Fig. 13, an initial Mach number of Ma = ‖u0‖2water∕cair = 5.7 × 10−2 can be calculated. 
The temporal discretization is done with the IMEX-ARK3 scheme [39] and CFL = 0.9 . The 
results of the simulation are reported in Fig. 14, together with the experimental results from 
[2]. One can see that the experimental results are reproduced with high accuracy: first the 
droplets merge and form a torus; due to the high surface tension they then elongate and finally 
separate again. All different type of shapes are correctly captured with the IMEX LSGFM.
6  Conclusion and Outlook
In this work it has been shown how an explicit LSGFM can be extended to deal with low 
Mach number flows. To increase the prohibitive small timestep of the explicit method, an 
IMEX flux splitting technique has been applied. It has been shown that by separating the 
domain into a fully implicit narrow band around the phase interface and into an IMEX flux 
splitting region, the timestep can be increased significantly. The separating of the domain 
became necessary as the interface flux and the narrow band region cannot be treated par-
tially implicit and partially explicit.
ρ0
air = 1.157 kg . m−3
u0
air = 0 m . s−1
p0
air = 0.1 MPa
γair = 1.4
ρ0
water = 998 kg . m−3
u0
water = (0, −6) m . s−1
p0
water = 0.100 331 MPa
γwater = 4.1
p∞= 5.7 MPa
σ = 0.072 8 J . m−2
r = 0.22 mm
ρ0
water = 998 kg . m−3
u0
water = 0 m . s−1
p0
water = 0.1 MPa
γwater = 4.1
p∞= 5.7 MPa
σ = 0.072 8 J . m−2
Outﬂow
Slip wall
Slip wall
Slip wall
Fig. 11   Initial data of computational setup for drop impingement onto a deep pool


743
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
A mixed discontinuous Galerkin/finite volume method has been used for the spatial dis-
cretization and IMEX Runge-Kutta schemes with a Jacobian-free Newton-GMRES method 
have been used for the temporal discretization.
The novel IMEX level set ghost fluid method has been validated and it has been 
shown that a large gain in computational time can be achieved compared to a fully 
explicit method. Applications to typical droplet dynamic problems illustrate the broad 
applicability of the developed scheme.
t = 0
t = 4τ ∗
(ρu)∗
2
0.2
0.02
t = 8τ ∗
t = 12τ ∗
(ρu)∗
2
0.2
0.02
t = 16τ ∗
t = 20τ ∗
(ρu)∗
2
0.2
0.02
Fig. 12   Snapshots of characteristic momentum of drop impingement onto a deep pool with the phase 
boundary being indicated with a white line
Table 2   Comparison of drop 
impingement into a pool with 
depth 5r and We = 217 from [58] 
and with the IMEX LSGFM 
method
Present simulation
Results from [58]
Crater depth
≈3.5r
≈4.1r
Crown height
≈8.3r
≈9.0r
Crown angle
≈80◦
≈82◦


744
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
A further improvement of the scheme is the consideration of viscous effects, i.e., 
choosing the Navier-Stokes equations over the Euler equations. This requires a consist-
ent discretization of viscous effects at the phase boundary. Possible methods can, e.g., 
be based on the diffusive generalized Riemann problem [27, 28] or on the ideas in [43].
Appendix A Riemann Solvers for the TV Splitting
TV Riemann solver For the sake of completeness, the standard TV flux from [66] is 
summarized here. The implicit numerical flux is given by
and the explicit numerical flux is given by
̃f
∗(wL, wR, n) ∶=
⎛
⎜
⎜⎝
0
p∗퐈퐝
u∗(휌∗휖(휌∗, p∗) + p∗)
⎞
⎟
⎟⎠n
,
ρ0
air = 1.157 kg . m−3
u0
air = 0 m . s−1
p0
air = 1 000 Pa
γair = 1.4
ρ0
water = 998 kg . m−3
u0
water = (0, −2, 0) m . s−1
p0
water = 3 776 Pa
γwater = 4.1
p∞= 5.7 106 Pa
σ = 347 J . m−2
r = 0.25 m
ρ0
water = 998 kg . m−3
u0
water = (0, 2, 0) m . s−1
p0
water = 3 776 Pa
γwater = 4.1
p∞= 5.7 106 Pa
σ = 347 J . m−2
r = 0.25 m
×
×
Fig. 13   Initial data of computational setup for binary droplet collision with We = 23


745
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
t = 0
t = 0.4τ ∗
t = τ ∗
t = 1.8τ ∗
t = 3.4τ ∗
t = 4.4τ ∗
Fig. 14   Temporal evolution of droplet surface for binary droplet collision with We = 23 . Experimental 
results [2] (top) and current simulation (bottom)


746
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
with the interface density, velocity and pressure
with
and A is given by Eq. (6). In the fully implicit region, the numerical flux is simply the sum 
of both parts
TVHLL-LM Riemann solver For small Mach numbers, the low dissipation TVHLL-LM 
solver can be used. A low dissipation numerical flux for low Mach numbers is obtained by 
modifying the HLL flux from [65] with the low Mach number fix according to [9]. For the 
explicit system the original flux of the TV flux is used, see Eq. (13). The implicit numeri-
cal flux is given by
The approximate wave speeds SL and SR , which are required for the HLL scheme can be 
calculated with
with the wave speed estimates
(13)
f
∗(wL, wR, n) ∶= u∗
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
⎛
⎜
⎜⎝
𝜌
𝜌u
1
2𝜌‖u‖2
2
⎞
⎟
⎟⎠nL
,
if u∗⩾0,
⎛
⎜
⎜⎝
𝜌
𝜌u
1
2𝜌‖u‖2
2
⎞
⎟
⎟⎠nR
,
if u∗< 0,
𝜌∗∶=
{
𝜌L,
if u∗⩾0,
𝜌R,
if u∗< 0,
u∗∶= CRu1R −CLu1L
CR −CL
−
2
CR −CL
(pR −pL),
p∗∶= CRpL −CLpR
CR −CL
+ 1
2
CRCL
CR −CL
(u1R −u1L),
CL ∶= 휌L(u1L −AL)
and
CR ∶= 휌R(u1R + AR)
f ∗(wL, wR, n) = ̃f
∗(wL, wR, n) + ̂f
∗(wL, wR, n).
f
∗(wL, wR, n) ∶=
⎧
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪⎩
Fn(wL),
if SL > 0,
1
2

Fn(wL) + Fn(wR)
−SL + SR
SR −SL

Fn(wL) + Fn(wR)

−2SRSL
SR −SL
Δw𝜌
n + min(1, Mloc)Δwu
n

,
if SL ⩽0 ⩽SR,
Fn(wR),
if SR < 0.
SL ∶= min(SL1, SL2)
and
SR ∶= max(SR1, SR2),


747
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
The jump terms Δwu
n and Δw휌
n are given as
with the jump Δ(⋅) = (⋅)L −(⋅)R . The low dissipation properties at low Mach numbers is 
ensured by multiplying Δwu
n with a factor depending on the local Mach number
Acknowledgements  This paper contains some revised and extended results from the first author’s PhD the-
sis work [71]. The authors kindly acknowledge the financial support provided by the Deutsche Forschun-
gsgemeinschaft (DFG, German Research Foundation) through the project GRK 2160/1 “Droplet Interaction 
Technologies” and through the project no. 457811052. The simulations were performed on the national 
supercomputer HPE Hawk at the High Performance Computing Center Stuttgart (HLRS) under the hpcdg 
project.
Funding  Open Access funding enabled and organized by Projekt DEAL.
Compliance with Ethical Standards 
Conflicts of Interest  The authors declare that they have no conflict of interest.
Open Access  This article is licensed under a Creative Commons Attribution 4.0 International License, 
which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long 
as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Com-
mons licence, and indicate if changes were made. The images or other third party material in this article 
are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the 
material. If material is not included in the article’s Creative Commons licence and your intended use is not 
permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly 
from the copyright holder. To view a copy of this licence, visit http://​creat​iveco​mmons.​org/​licen​ses/​by/4.​0/.
References
	 1.	 An, H.B., Wen, J., Feng, T.: On finite difference approximation of a matrix-vector product in the Jaco-
bian-free Newton-Krylov method. J. Comput. Appl. Math. 236(6), 1399–1409 (2011)
	 2.	 Ashgriz, N., Poo, J.: Coalescence and separation in binary collisions of liquid drops. J. Fluid Mech. 
221, 183–204 (1990)
	 3.	 Boger, M., Jaegle, F., Klein, R., Munz, C.D.: Coupling of compressible and incompressible flow 
regions using the multiple pressure variables approach. Math. Methods Appl. Sci. 38(3), 458–477 
(2015)
	 4.	 Boger, M., Jaegle, F., Weigand, B., Munz, C.D.: A pressure-based treatment for the direct numerical 
simulation of compressible multi-phase flow using multiple pressure variables. Comput. Fluids 96, 
338–349 (2014)
SL1 ∶= min (u1L −cL, u1R −cR
)
n
and
SL2 ∶= 1
2 min (u1L −AL, u1R −AR
)
n,
SR1 ∶= max (u1L + cL, u1R + cR
)
n
and
SR2 ∶= 1
2 max (u1L + AL, u1R + AR
)
n.
Δwu
n ∶= 휌L + 휌R
2
⎛
⎜
⎜⎝
0
Δu
1
2Δ‖u‖2
2
⎞
⎟
⎟⎠n
and
Δw휌
n ∶= Δwn −Δwu
n,
Mloc ∶= max
‖u‖2
c

L
,
‖u‖2
c

R
, 10−4

.


748
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
	 5.	 Boscarino, S., Russo, G., Scandurra, L.: All Mach number second order semi-implicit scheme for the 
Euler equations of gasdynamics. J. Sci. Comput. 77(2), 850–884 (2018)
	 6.	 Carpenter, M., Kennedy, C.: Fourth-order 2N-storage Runge-Kutta schemes. Tech. rep, NASA Langley 
Research Center (1994)
	 7.	 Chalons, C., Coquel, F., Kokh, S., Spillane, N.: Large time-step numerical scheme for the seven-equa-
tion model of compressible two-phase flows. In: Fořt, J., Fürst, J., Halama, J., Herbin, R., Hubert, F. 
(eds) Finite Volumes for Complex Applications VI Problems & Perspectives, Springer Proceedings in 
Mathematics, vol. 4, pp. 225–233. Springer, Berlin, Heidelberg (2011)
	 8.	 Chalons, C., Girardin, M., Kokh, S.: An all-regime Lagrange-projection like scheme for 2D homoge-
neous models for two-phase flows on unstructured meshes. J. Comput. Phys. 335, 885–904 (2017)
	 9.	 Chen, S.S., Yan, C., Xiang, X.H.: Effective low-Mach number improvement for upwind schemes. 
Comput. Math. Appl. 75(10), 3737–3755 (2018)
	10.	 Chisholm, T.T., Zingg, D.W.: A Jacobian-free Newton-Krylov algorithm for compressible turbulent 
fluid flows. J. Comput. Phys. 228(9), 3490–3507 (2009)
	11.	 Choquet, R., Erhel, J.: Newton-GMRES algorithm applied to compressible flows. Int. J. Numer. Meth-
ods Fluids 23(2), 177–190 (1996)
	12.	 Cordier, F., Degond, P., Kumbaro, A.: An asymptotic-preserving all-speed scheme for the Euler and 
Navier-Stokes equations. J. Comput. Phys. 231, 5685–5704 (2012)
	13.	 Degond, P., Tang, M.: All speed scheme for the low Mach number limit of the isentropic Euler equa-
tion. Commun. Comput. Phys. 10, 1–31 (2011)
	14.	 Denner, F., van Wachem, B.G.: Numerical time-step restrictions as a result of capillary waves. J. Com-
put. Phys. 285, 24–40 (2015)
	15.	 Denner, F., Xiao, C.N., van Wachem, B.G.: Pressure-based algorithm for compressible interfacial 
flows with acoustically-conservative interface discretisation. J. Comput. Phys. 367, 192–234 (2018)
	16.	 Desjardins, O., Moureau, V., Pitsch, H.: An accurate conservative level set/ghost fluid method for sim-
ulating turbulent atomization. J. Comput. Phys. 227(18), 8395–8416 (2008)
	17.	 Du Chéné, A., Min, C., Gibou, F.: Second-order accurate computation of curvatures in a level 
set framework using novel high-order reinitialization schemes. J. Sci. Comput. 35(2/3), 114–131 
(2008)
	18.	 Duret, B., Canu, R., Reveillon, J., Demoulin, F.: A pressure based method for vaporizing compress-
ible two-phase flows with interface capturing approach. Int. J. Multiph. Flow 108, 42–50 (2018)
	19.	 Fechter, S.: Compressible multi-phase simulation at extreme conditions using a discontinuous 
Galerkin scheme. Ph.D. thesis, University of Stuttgart (2015)
	20.	 Fechter, S., Jaegle, F., Schleper, V.: Exact and approximate Riemann solvers at phase boundaries. 
Comput. Fluids 75, 112–126 (2013)
	21.	 Fechter, S., Munz, C.D.: A discontinuous Galerkin-based sharp-interface method to simulate three-
dimensional compressible two-phase flow. Int. J. Numer. Methods Fluids 78(7), 413–435 (2015)
	22.	 Fechter, S., Munz, C.D., Rohde, C., Zeiler, C.: Approximate Riemann solver for compressible liq-
uid vapor flow with phase transition and surface tension. Comput. Fluids 169, 169–185 (2018)
	23.	 Fedkiw, R.P., Aslam, T., Merriman, B., Osher, S.: A non-oscillatory Eulerian approach to inter-
faces in multimaterial flows (the ghost fluid method). J. Comput. Phys. 152(2), 457–492 (1999)
	24.	 Föll, F., Hitz, T., Müller, C., Munz, C.D., Dumbser, M.: On the use of tabulated equations of state 
for multi-phase simulations in the homogeneous equilibrium limit. Shock Waves 29(5), 769–793 
(2019)
	25.	 Frolov, R.: An efficient algorithm for the multicomponent compressible Navier-Stokes equations in 
low- and high-Mach number regimes. Comput. Fluids 178, 15–40 (2019)
	26.	 Fuster, D., Popinet, S.: An all-Mach method for the simulation of bubble dynamics problems in the 
presence of surface tension. J. Comput. Phys. 374, 752–768 (2018)
	27.	 Gassner, G., Lörcher, F., Munz, C.D.: A contribution to the construction of diffusion fluxes for 
finite volume and discontinuous Galerkin schemes. J. Comput. Phys. 224(2), 1049–1063 (2007)
	28.	 Gassner, G., Lörcher, F., Munz, C.D.: A discontinuous Galerkin scheme based on a space-time 
expansion II. Viscous flow equations in multi dimensions. J. Sci. Comput. 34(3), 260–286 (2008)
	29.	 Ghosh, D., Constantinescu, E.M.: Semi-implicit time integration of atmospheric flows with charac-
teristic-based flux partitioning. SIAM J. Sci. Comput. 38(3), A1848–A1875 (2016)
	30.	 Haack, J., Jin, S., Liu, J.G.: An all-speed asymptotic-preserving method for the isentropic Euler and 
Navier-Stokes equations. Commun. Comput. Phys. 12, 955–980 (2012)
	31.	 Henneaux, D., Schrooyen, P., Dias, B., Turchi, A., Chatelain, P., Magin, T.: Extended discontinuous 
Galerkin method for solving gas-liquid compressible flows with phase transition. In: AIAA Avia-
tion Forum (2020)


749
Communications on Applied Mathematics and Computation (2023) 5:722–750	
1 3
	32.	 Hindenlang, F., Gassner, G., Altmann, C., Beck, A., Staudenmaier, M., Munz, C.D.: Explicit dis-
continuous Galerkin methods for unsteady problems. Comput. Fluids 61, 86–93 (2012)
	33.	 Hirt, C.W., Nichols, B.D.: Volume of fluid (VOF) method for the dynamics of free boundaries. J. 
Comput. Phys. 39(1), 201–225 (1981)
	34.	 Hitz, T.: On the Riemann problem and the Navier-Stokes-Korteweg model for compressible mul-
tiphase flows. Ph.D. thesis, University of Stuttgart (2019)
	35.	 Huber, G., Tanguy, S., Béra, J.C., Gilles, B.: A time splitting projection scheme for compressible 
two-phase flows. Application to the interaction of bubbles with ultrasound waves. J. Comput. Phys. 
302, 439–468 (2015)
	36.	 Jiang, G.S., Peng, D.: Weighted ENO schemes for Hamilton-Jacobi equations. SIAM J. Sci. Com-
put. 21(6), 2126–2143 (2000)
	37.	 Jöns, S., Müller, C., Zeifang, J., Munz, C.D.: Recent advances and complex applications of the 
compressible ghost-fluid method. In: SEMA SIMAI Springer Series, Proceedings of Numhyp 2019. 
Springer (2020). Accepted
	38.	 Kanevsky, A., Carpenter, M.H., Gottlieb, D., Hesthaven, J.S.: Application of implicit-explicit high 
order Runge-Kutta methods to discontinuous-Galerkin schemes. J. Comput. Phys. 225(2), 1753–
1781 (2007)
	39.	 Kennedy, C.A., Carpenter, M.H.: Additive Runge-Kutta schemes for convection-diffusion-reaction 
equations. Appl. Numer. Math. 44, 139–181 (2003)
	40.	 Knoll, D.A., Keyes, D.E.: Jacobian-free Newton-Krylov methods: a survey of approaches and 
applications. J. Comput. Phys. 193, 357–397 (2004)
	41.	 Kopriva, D.A.: Implementing Spectral Methods for Partial Differential Equations: Algorithms for 
Scientists and Engineers, 1st edn. Springer Publishing Company Incorporated, Springer, Dordrecht 
(2009)
	42.	 Krais, N., Beck, A., Bolemann, T., Frank, H., Flad, D., Gassner, G., Hindenlang, F., Hoffmann, M., 
Kuhn, T., Sonntag, M., Munz, C.D.: FLEXI: a high order discontinuous Galerkin framework for 
hyperbolic-parabolic conservation laws. Comput. Math. Appl. 81, 186–219 (2021)
	43.	 Lalanne, B., Villegas, L.R., Tanguy, S., Risso, F.: On the computation of viscous terms for incom-
pressible two-phase flows with level set/ghost fluid method. J. Comput. Phys. 301, 289–307 (2015)
	44.	 Lamb, H.: Hydrodynamics. Cambridge University Press, Cambridge (1932)
	45.	 Lee, J., Son, G.: A level-set method for ultrasound-driven bubble motion with a phase change. 
Numer. Heat Transf. Part A Appl. 71(9), 928–943 (2017)
	46.	 Liu, W., Yuan, L., Shu, C.-W.: A conservative modification to the ghost fluid method for compress-
ible multiphase flows. Commun. Comput. Phys. 10(4), 785 (2011)
	47.	 Main, A., Farhat, C.: A second-order time-accurate implicit finite volume method with exact two-
phase Riemann problems for compressible multi-phase fluid and fluid-structure problems. J. Com-
put. Phys. 258, 613–633 (2014)
	48.	 Marchandise, E., Geuzaine, P., Chevaugeon, N., Remacle, J.F.: A stabilized finite element method 
using a discontinuous level set approach for the computation of bubble dynamics. J. Comput. Phys. 
225(1), 949–974 (2007)
	49.	 Merkle, C., Rohde, C.: The sharp-interface approach for fluids with phase change: Riemann prob-
lems and ghost fluid techniques. ESAIM Math. Model. Numer. Anal. 41(6), 1089–1123 (2007)
	50.	 Müller, C., Hitz, T., Jöns, S., Zeifang, J., Chiocchetti, S., Munz, C.D.: Improvement of the level-
set ghost-fluid method for the compressible Euler equations. In: Lamanna, G., Tonini, S., Cossali, 
G.E., Weigand, B. (eds) Droplet Interaction and Spray Processes. Springer, Heidelberg (2020)
	51.	 Noelle, S., Bispen, G., Arun, K.R., Lukácová-Medvid’ová, M., Munz, C.D.: A weakly asymptotic 
preserving low Mach number scheme for the Euler equations of gas dynamics. SIAM J. Sci. Com-
put. 36, B989–B1024 (2014)
	52.	 Olsson, E., Kreiss, G.: A conservative level set method for two phase flow. J. Comput. Phys. 210(1), 
225–246 (2005)
	53.	 Olsson, E., Kreiss, G., Zahedi, S.: A conservative level set method for two phase flow II. J. Comput. 
Phys. 225(1), 785–807 (2007)
	54.	 Osher, S., Sethian, J.A.: Fronts propagating with curvature-dependent speed: algorithms based on 
Hamilton-Jacobi formulations. J. Comput. Phys. 79(1), 12–49 (1988)
	55.	 Peluchon, S., Gallice, G., Mieussens, L.: A robust implicit-explicit acoustic-transport splitting 
scheme for two-phase flows. J. Comput. Phys. 339, 328–355 (2017)
	56.	 Peng, D., Merriman, B., Osher, S., Zhao, H., Kang, M.: A PDE-based fast local level set method. J. 
Comput. Phys. 155(2), 410–438 (1999)
	57.	 Perigaud, G., Saurel, R.: A compressible flow model with capillary effects. J. Comput. Phys. 
209(1), 139–178 (2005)


750
	
Communications on Applied Mathematics and Computation (2023) 5:722–750
1 3
	58.	 Ray, B., Biswas, G., Sharma, A.: Oblique drop impact on deep and shallow liquid. Commun. Com-
put. Phys. 11(4), 1386–1396 (2012)
	59.	 Schleper, V.: A HLL-type Riemann solver for two-phase flow with surface forces and phase transi-
tions. Appl. Numer. Math. 108, 256–270 (2016)
	60.	 Sonntag, M., Munz, C.D.: Shock capturing for discontinuous Galerkin methods using finite volume 
subcells. In: Fuhrmann, J., Mario, O., Christian, R. (eds) Finite Volumes for Complex Applications 
VII-Elliptic, Parabolic and Hyperbolic Problems, pp. 945–953. Springer (2014)
	61.	 Sonntag, M., Munz, C.D.: Efficient parallelization of a shock capturing for discontinuous Galerkin 
methods using finite volume sub-cells. J. Sci. Comput. 70(3), 1262–1289 (2017)
	62.	 Sussman, M., Smereka, P., Osher, S.: A level set approach for computing solutions to incompress-
ible two-phase flow. J. Comput. Phys. 114(1), 146–159 (1994)
	63.	 Tanguy, S., Berlemont, A.: Application of a level set method for simulation of droplet collisions. 
Int. J. Multiph. Flow 31(9), 1015–1035 (2005)
	64.	 Tavelli, M., Dumbser, M.: A pressure-based semi-implicit space-time discontinuous Galerkin 
method on staggered unstructured meshes for the solution of the compressible Navier-Stokes equa-
tions at all Mach numbers. J. Comput. Phys. 341, 341–376 (2017)
	65.	 Tiam Kapen, P., Ghislain, T.: A new flux splitting scheme based on Toro-Vazquez and HLL 
schemes for the Euler equations. J. Comput. Methods Phys. 2014, 827034 (2014)
	66.	 Toro, E.F., Castro, C.E., Lee, B.J.: A novel numerical flux for the 3D Euler equations with general 
equation of state. J. Comput. Phys. 303, 80–94 (2015)
	67.	 Toro, E.F., Vázquez-Cendón, M.E.: Flux splitting schemes for the Euler equations. Comput. Fluids 
70, 1–12 (2012)
	68.	 Vangelatos, S.: On the efficiency of implicit discontinuous Galerkin spectral element methods for 
the unsteady compressible Navier-Stokes equations. Ph.D. thesis, University of Stuttgart (2019)
	69.	 Williamson, J.H.: Low-storage Runge-Kutta schemes. J. Comput. Phys. 35, 48–56 (1980)
	70.	 Woodward, C.S., Gardner, D.J., Evans, K.J.: On the use of finite difference matrix-vector products 
in Newton-Krylov solvers for implicit climate dynamics with spectral elements. Proc. Comput. Sci. 
51, 2036–2045 (2015)
	71.	 Zeifang, J.: A discontinuous Galerkin method for droplet dynamics in weakly compressible flows. 
Ph.D. thesis, University of Stuttgart (2020)
	72.	 Zeifang, J., Kaiser, K., Beck, A., Schütz, J., Munz, C.D.: Efficient high-order discontinuous Galerkin 
computations of low Mach number flows. Commun. Appl. Math. Comput. Sci. 13(2), 243–270 (2018)
	73.	 Zeifang, J., Kaiser, K., Schütz, J., Massa, F.C., Beck, A.: An investigation of different splitting tech-
niques for the isentropic Euler equations. In: Lamanna, G., Tonini, S., Cossali, G.E., Weigand, B. (eds) 
Droplet Interaction and Spray Processes. Fluid Mechanics and Its Applications, vol. 121, pp. 45–55. 
Springer, Heidelberg (2020)
	74.	 Zeifang, J., Schütz, J., Kaiser, K., Beck, A., Lukácová-Medvid’ová, M., Noelle, S.: A novel full-Euler 
low Mach number IMEX splitting. Commun. Comput. Phys. 27, 292–320 (2020)
