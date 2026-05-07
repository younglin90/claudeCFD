A semi implicit compressible solver for two-phase
ﬂows of real ﬂuids
A. Urbanoa, M. Bibalb, S. Tanguyb
aISAE SUPAERO, Universit´e de Toulouse, Toulouse, France
bInstitut de M´ecanique des Fluides de Toulouse (IMFT), Universit´e de
Toulouse, CNRS, Toulouse, France
Abstract
The development of numerical solvers able to simulate compressible two-
phase ﬂows is still a great challenge in computational ﬂuid dynamics. The
interaction between acoustic waves and interfaces is of major concern for
several engineering and biomedical applications, among which atomization
in combustion chambers, cavitation problems, underwater explosions and
bubble shock interactions. For instance, there are experimental evidences
that acoustic waves can have an important eﬀect on the atomization pro-
cess, and this could have a great impact on combustion. However, usual
approaches for DNS of primary atomization are based on incompressible
solvers and therefore are not able to capture the propagation of acoustic
waves and therefore cannot be used to simulate such phenomena. The nu-
merical problem associated with the simulation of compressible two-phase
ﬂows is challenging, mostly because of the huge spatial variations of the
speed of sound and the corresponding low Mach number in the liquid phase.
In the present work, a numerical solver able to study subsonic compressible
two-phase ﬂows is presented. The solver is based on a complete formulation
of the Navier-Stokes equations with real ﬂuid equations of state, which are
solved with a semi-implicit projection method. It is shown that the solver
can handle a large range of compressible subsonic ﬂows, both for a single
phase or for two phases, as the ﬂow induced by free convection, a bubble
expansion in isothermal or isentropic conditions, and interaction between
acoustic waves and liquid-gas interfaces. Eventually, attention will be given
to the simulation of a water droplet in air, under the excitation of a sta-
tionary acoustic wave. It is also shown that the solver exhibits equivalent
performances as an incompressible solver in conﬁgurations where compress-
ible eﬀects have no eﬀects.
Keywords:
compressible two-phase ﬂows, real real ﬂuids equations of
Preprint submitted to Elsevier
February 1, 2022
© 2022 published by Elsevier. This manuscript is made available under the Elsevier user license
https://www.elsevier.com/open-access/userlicense/1.0/
Version of Record: https://www.sciencedirect.com/science/article/pii/S0021999122000961
Manuscript_baaf7034fe28f4ceed9a5cdc22cbaf36


state, acoustics interface interactions, ghost ﬂuid method, semi-implicit
projection method, free convection
1. Introduction
Numerical solvers for compressible ﬂows, with an interface capturing
or interface tracking approach, are mainly based on fully explicit shock-
capturing methods, and have been developed in the framework of high Mach
number ﬂows, in particular for the study of the interaction between drops or
bubbles and shock waves [17, 53, 9, 19], or atomisation with ultrasonics [44].
The major drawbacks of these explicit approaches are the stability condition
associated with the acoustic propagation and the loss of accuracy in the low
Mach number regime which can be particularly stringent in the liquid phase
(due to the high speed of sound). That makes these methodologies ineﬃcient
at low Mach numbers. To avoid the low Mach number constraint associated
with the liquid phase, mixed approaches, considering a compressible gas and
incompressible liquid, have been proposed (see for instance [7, 8, 1, 26]).
However, such approaches are limited to describe the interaction of a liquid
with acoustic waves since they do not allow the acoustic waves propagation
in the liquid phase.
In recent years another class of methods, based on a semi-implicit com-
pressible projection approach, have been developed for two-phase ﬂows [24,
20, 42, 18], which are derived from the single-phase formulations proposed
by [57] and by [27]. The main idea is to split the equations in an advec-
tion and an acoustic part. The advection part is solved with an explicit
scheme while the acoustic part is solved with an implicit scheme with a pro-
jection method, thus removing the stringent time step constraint associated
with the propagation of acoustic waves. The methodology is also suitable
to treat low Mach number compressible ﬂows since it is asymptotically pre-
serving [24], that is the incompressible formulation is retrieved when the
Mach number goes to zero. This class of algorithms diﬀers from another
class of pressure-based solvers, SIMPLE [39] and its variants (SIMPLEC,
SIMPLER,...)[16, 38], which require solving a nonlinear iterative system.
Such algorithms are beneﬁcial for the computation of steady-state ﬂows,
but their use may not be the most eﬃcient numerical strategy for perform-
ing Direct Numerical Simulation of unsteady ﬂows that require accurate
time-scale solving.
The compressible projection method has attractive numerical properties
for the simulation of two-phase ﬂows at low Mach number. However, for-
2


mulations of this algorithm are usually isentropic, and no viscous and heat
conduction eﬀects are taken into consideration, even though they are of
great importance in most applications. Only a few number of works present
a formulation for two-phase compressible ﬂows including non-isentropic ef-
fects [24, 30, 10]. In particular Jemison et al. [24] presented an approach
to include non-viscous terms, and later extended it to include heat conduc-
tion in [2]. Their solver is based on a conservative variables formulation,
and viscous eﬀects are considered in the total energy equation. However,
while conserving the total energy, this approach fails in retrieving a pressure
consistent with the equation of state at each iteration, as yet demonstrated
by [27]. A proper formulation of a compressible projection algorithm, using
a primitive variables formulation, considering non-isentropic eﬀects in the
conservation energy equation based on the pressure variable, has never been
proposed, and would be a signiﬁcant step forward for this class of solvers
in particular for the study of heat transfer problems. Indeed, there are two
major advantages to using a projection method based on a primitive vari-
ables formulation, as presented in this work, rather than solvers based on
a conservative variables formulation. Firstly, the present primitive formula-
tion guarantees that the pressure derived from the equation of state matches
the pressure computed by the semi-implicit pressure correction algorithm at
each time step. Secondly, the formulation of the energy equation as a pres-
sure equation allows the heat conduction term to be treated with an implicit
temporal scheme. On the other hand, the disadvantage of a primitive vari-
ables formulation is the poor conservation of total energy, which can be a
problem when trying to describe supersonic ﬂows. However, the focus of
the present solver is restricted to subsonic ﬂows and future developments
will allow the treatment of phase changes induced by pressure or tempera-
ture variations in compressible ﬂows at low Mach numbers. Note that the
present algorithm is not positive preserving for the pressure, as the one pro-
posed in [40], and it does not prevent for negative pressure values. However,
metastable thermodynamical states involving negative pressure values can
be observed in liquids, as in some cavitating ﬂows. Such states can be mod-
elled with suitable equations of state (EoS) for which positive temperatures
are associated to negative pressures, as for instance the van der Waals EoS
[21, 13].
Another important point addressed in this paper concerns the equations
of state (EoS). Previous studies were essentially based on ideal ﬂuid EoS,
such as Tait EoS for the liquid phase and perfect gas EoS for the gaseous
phase.
In the present work a cubic EoS will be considered.
The solver
beneﬁts from such a generic EoS which allows the description of both gas
3


and liquid phases.
The present work describes the development of a fully compressible nu-
merical solver for two-phase ﬂows based on a complete entropic (i.e. non-
isentropic) formulation of the Navier-Stokes and energy conservation equa-
tions, accounting for capillary eﬀects and making use of a generic equation of
state for both phases. The numerical tests proposed in this paper will focus
essentially on the van der Waals equation which is the simplest cubic EoS.
However, the numerical framework proposed is general enough to handle any
EoS, provided the sound speed can be computed in both phases. As a sharp
representation of the interface and related jump conditions is performed,
the deﬁnition of the density is still thermodynamically consistent with the
EoS on each side of the interface. Consequently, the overall solver is free of
nonphysical sound speed, that on another side can occur for solvers based
on a diﬀuse interface representation, due to the spurious density smoothing,
as remarked in [35].
The solver is derived from the one developed for Euler equations by [20]
and integrated in the home made code, DIVA. The innovative splitting of
the equations proposed in [20], to correctly handle capillary eﬀects, is edited
and extended to the more complete system of equations which is considered
here. As the proposed numerical solver is based on a primitive formulation
of the conservation equations, its applicability ﬁeld is restricted to subsonic
two-phase ﬂows, without any other assumptions on the Mach number while
Ma < 1.
In the following, the mathematical formalism and the numerical algo-
rithms are described. Then diﬀerent validation test cases are described and
analysed in order to verify several important characteristics of the solver:
acoustic waves propagation, thermal and density ﬁeld interaction, heat con-
duction in single and two-phase ﬂows, interface acoustic interaction and
thermodynamic features.
2. Mathematical formalism
2.1. Primitive conservation laws
A two-phase compressible ﬂow can be described with the following form
of the conservation equations written in terms of the primitive variables,
i.e. the density ρ and the velocity vector ⃗u, for the mass and momentum
4


conservation, respectively,
Dρ
Dt + ρ∇· ⃗u = 0,
(1)
D⃗u
Dt + ∇p
ρ = 1
ρ∇· τ + σκ⃗nδΓ
ρ
+ ⃗g,
(2)
where p is the pressure ﬁeld, ⃗g is the gravity acceleration, and τ the
tensor of viscous constraints deﬁned as,
τ = µ(∇⃗u + ∇⃗uT ) −2
3µ∇· ⃗uI,
(3)
which is valid for a compressible Newtonian ﬂuid, with µ the dynamic
viscosity and I the identity matrix. The variables σ, κ and ⃗n are related
to interface properties and are named respectively the surface tension, the
interface curvature and the normal vector pointing towards the liquid phase
and δΓ is a δ-Dirac localized at the interface Γ.
An additional equation
will be added in the following to compute these variables. Moreover, unlike
incompressible ﬂows, such a system of equations is not complete if com-
pressible ﬂows are considered (since ρ is no longer a constant but a variable)
and additional equations are required to close the system. In particular, as
compressible ﬂows involve a strong coupling between thermal and dynami-
cal eﬀects, the conservation energy equation has to be added to the overall
system. It is usual to express this conservation law as an evolution equation
for the internal energy e, such as,
ρDe
Dt = −p∇· ⃗u −∇· ⃗q + τ ⊗∇⃗u,
(4)
or an equation evolution for the enthalpy, h = e + p/ρ,
ρDh
Dt = Dp
Dt −∇· ⃗q + τ ⊗∇⃗u,
(5)
where ⃗q is the local heat ﬂux deﬁned, following the Fourier’s law, as
⃗q = −k∇T,
(6)
with k the thermal conductivity and T the temperature. The last term of
the conservation energy law is the thermal release due to the viscous friction
in the ﬂuid domain. Finally, as the pairs of unknown variables (e,T) or (h,T)
emerge from the latter equations, two equations of state must be speciﬁed,
respectively, for the internal energy or enthalpy, and for the temperature.
5


For example, if one considers a thermally perfect gas, the following relations
hold to deﬁne the internal energy or the enthalpy,
e(T) −e0 = cv(T −T0),
h(T) −h0 = cp(T −T0),
(7)
and a further equation of state from which the pressure ﬁeld can be
deduced,
p = ρRT,
(8)
where R is the gas constant (R = R/W with R = 8, 314 J/mol/K and W
molecular weight of the species). This results in a complete framework for
the description of gas dynamics. Such solvers based on the above equations
have enabled important advances in the ﬁeld of gas dynamics simulations to
calculate both subsonic ﬂows at intermediate Mach number (0.1 ≤Ma < 1)
and supersonic ﬂows at high Mach number (Ma ≥1). One should specify
here, that a conservative formulation of conservation laws have to be solved
in the latter case to compute a correct shock wave speed. However, one of
the drawbacks of these solvers is to perform poorly at low Mach number, in
regards to both stability and accuracy. Indeed, the time step is limited by
the acoustic wave propagation speed, if an explicit temporal integration has
been chosen. This results in a large diﬀerence between the stability time
step constraint and the physical characteristic time of the ﬂow at low Mach
number, and thus in an oversized number of temporal iterations. Moreover,
accuracy issues have also been reported for that kind of solver for which it
can be demonstrated that the error depends on O
  1
Ma

, as detailed in [14],
and thus increases if the Mach number decreases.
Some preconditioning
techniques, as the one proposed in [54] may be used to alleviate this, by
introducing a modiﬁed artiﬁcial sound speed. However these issues become
critical in the framework of compressible two-phase ﬂows for which a very
low Mach number is generally observed in the liquid phase due to its high
sound speed. To prevent these issues, another type of compressible solvers,
based on a pressure formulation of the energy conservation equation, have
been developed for two-phase ﬂows in the last decade [20, 24, 18]. As shown
in [24], such solvers have the nice property of preserving asymptotically the
incompressible limit, since the pressure equation tends to the well-known
pressure Poisson equation of the incompressible projection method, in the
limit of an inﬁnite sound speed. However, these solvers have never been
rigorously generalized to ﬂows for which entropic eﬀects, such as heat con-
duction and viscous friction, have to be considered. We propose hereafter
a compressible two-phase ﬂows solver with a energy conservation equation
6


based on an entropy/pressure formulation that will enable to account for
these entropic eﬀects.
To develop this formulation, we will adopt thermodynamic relations in
order to express the above system in terms of ρ, ⃗u and p as primitive vari-
ables. The pressure being expressed with the density and entropy p(ρ, s),
its diﬀerential will be developed as,
dp = c2dρ +
∂p
∂s

ρ
ds,
(9)
which allows to introduce explicitly the sound speed c deﬁned from the
following relation:
c2 =
∂p
∂ρ

s
.
(10)
The Maxwell relation can be used in order to express the pressure derivative
with respect to entropy,
∂p
∂s

ρ
∂s
∂ρ

p
∂ρ
∂p

s
= −1,
→
∂p
∂s

ρ
= −
c2

∂s
∂ρ

p
.
(11)
From the ﬁrst principle of thermodynamic (dh = Tds + dp/ρ) the thermo-
dynamic derivatives of the enthalpy h with respect to s and ρ are:
∂h
∂s

p
= T,
∂h
∂p

s
= 1
ρ.
(12)
Moreover the following equality holds,
∂h
∂ρ

p
=
 ∂h
∂T

p
∂T
∂ρ

p
,
→
∂h
∂ρ

p
= −cp
ρα,
(13)
with α isobaric expansion coeﬃcient (see Eq.(37)) and cp the speciﬁc heat
at constant pressure. Combining Eqs.(12) and (13) we obtain,
∂s
∂ρ

p
=
 ∂s
∂h

p
∂h
∂ρ

p
,
→
∂s
∂ρ

p
= −cp
Tρα.
(14)
Eventually, combining Eqs. (14), (11) and Eq. (9) we can express the total
derivative of the pressure with the following equation,
Dp
Dt = c2 Dρ
Dt + ρc2Tα
cp
Ds
Dt .
(15)
7


It is noteworthy that this equation is an expression of the conserva-
tion energy principle, based on the pressure/entropy variable, since the ﬁrst
law of thermodynamics has been used in Eq.(12) to derive this expression.
Pointing out that the entropy material derivative can be expressed as,
ρDs
Dt = −∇· ⃗q + τ ⊗∇⃗u
T
,
(16)
and the material derivative of ρ in Eq.(15) being expressed thanks to
Eq.(1), we ﬁnally obtain the following form of the conservation energy equa-
tion, expressed in terms of pressure,
Dp
Dt + ρc2∇· ⃗u = c2α
cp
(τ ⊗∇⃗u −∇· ⃗q) .
(17)
which in conjunction with Eq.(1) and (2) gives the following system of equa-
tions which is used as a basis in the present work,
∂ρ
∂t + ∇· (ρ⃗u) = 0,
(18)
∂⃗u
∂t + ⃗u · ∇⃗u + ∇p
ρ = 1
ρ∇· τ + σκ⃗nδΓ
ρ
+ ρ⃗g,
(19)
∂p
∂t + ⃗u · ∇p + ρc2∇· ⃗u = c2α
cp
[τ ⊗∇⃗u −∇· ⃗q] .
(20)
In [52], the following equivalent expression of a pressure based energy con-
servation equation, has been proposed,
∂p
∂t + ⃗u · ∇p + ρc2∇· ⃗u =
α
ρcvβ (τ ⊗∇⃗u −∇· ⃗q) ,
(21)
with β = 1
ρ

∂ρ
∂p

T .
However in the latter study, a fully explicit solver was used, which does
not enable to alleviate the acoustical time step constraint.
2.2. Jump conditions across the interface
We now present the jump conditions that must be taken into account to
maintain mass conservation, momentum conservation and energy conserva-
tion across the interface if two-phase ﬂows are considered.
Capillary eﬀects depend on σ, the surface tension and κ, the local in-
terface curvature. They can be taken into account with the surface tension
force term in the right hand side of the momentum equation Eq.(19) or as
8


a jump condition when solving a Poisson or a Helmholtz equation for the
pressure in the framework of incompressible solvers [25, 29] or compressible
solvers [20, 24], respectively.
The system of equations (18)-(20) describes the ﬂow in each phase and
at the interface the following set of jump conditions must be veriﬁed,
[⃗u]Γ = 0,
(22)
[p]Γ = σκ + 2

µ∂un
∂n

Γ
−2
3 [µ∇· ⃗u]Γ ,
(23)
[−k∇T · ⃗n]Γ = 0.
(24)
In the above formalism, no mass transfer is considered between the
phases: this means that no phase change is possible, nor diﬀusion between
the phases. Furthermore, thermodynamic equilibrium and zero entropy pro-
duction at the interface are assumed, which implies that the temperature is
continuous across the interface ([T]Γ = 0) [22].
2.3. Equation of state for the gas and liquid phases
An important feature of compressible ﬂows is the coupling between me-
chanical eﬀects and thermal eﬀects that can be described through the reso-
lution of Navier-Stokes equations with mass and energy conservation equa-
tions. However, an equation of state (EoS) is required in order to close the
system. In particular, one can remark that the local heat ﬂux computa-
tion requires the temperature ﬁeld evaluation, which can be expressed as a
function of the density and pressure T(ρ, p) in Eq.(20), since both variables
can be obtained by solving their corresponding evolution equations. Other
relations are mandatory in order to compute the sound speed c (deﬁned by
Eq.(10)), the isobaric coeﬃcient α and the speciﬁc heat at constant pressure
cp. These relations must be deduced from an EoS which should be able to
describe both the gas and liquid states. The well-known perfect gas EoS,
p = ρRT, is appropriate to describe gaseous states at high temperature and
low pressure in a wide range, but does not hold for liquid states. On the
other hand, liquid states can be described using for instance the Tate EoS,
that however is not valid for a gas. Among all existing EoS, a cubic EoS has
the interest of being able to describe both a gas state and a liquid state with
a generic expression. In this type of equations, the density obeys a cubic
polynomial equation,
ρ3 + a1ρ2 + a2ρ + a3 = 0,
(25)
9


where a1, a2 and a3 are parameters depending on the pressure, the tem-
perature, species characteristics and the speciﬁc cubic EoS. Among the well
known cubic EoS are van der Waals, Peng-Robinson and Redlich-Soave-
Kwong equations. In the present work, the van der Waals EoS has been
used and therefore in the following speciﬁc expressions obtained with the
van der Waals EoS will be given. However, the implementation presented in
this paper is valid for any cubic EoS and can be easily extended to be used
with Peng-Robinson or Redlich-Soave-Kwong equations that should enable
a more accurate description of the liquid and gas states than the van der
Waals equation, as pointed out in [41].
Eq.(25) can be rewritten in the following form, where the relation be-
tween p, ρ and T appears explicitly,
p =
ρRT
1 −ρB −
ρ2A
1 + ρUB + ρ2V B2
(26)
where U and V are integer constant values that depend on the selected
EoS. It can be demonstrated that the parameters A and B are functions
of the critical temperature Tc and critical pressure pc and that they are
selected in order to verify proper thermodynamic conditions. The values of
the coeﬃcients for the van der Waals EoS are,
U = V = 0
A = 27
64
R2T 2
c
pc
,
B = RTc
8pc
.
(27)
The corresponding ai coeﬃcients of Eq.(25) are,
a1 = −1
B ,
a2 = (PB + RT)
AB
,
a3 = −p
AB .
(28)
Combining Eq.(27) and Eq.(26), the explicit expression of p as a function
of ρ and T given by the van der Waals EoS is obtained,
p =
ρRT
1 −ρB −ρ2A.
(29)
Similarly, the expression of T as a function of p and ρ is,
T = (p + ρ2A)(1 −ρB)
ρR
.
(30)
In order to evaluate the density ρ for a given p and T the roots of the third
order polynomial Eq.(25) have to be computed.
Moreover, if saturation
conditions are reached, further thermodynamic equilibrium conditions have
10


to be applied in order to compute the vapour-liquid coexistence curve [31].
However, the present work only deals with phases of two distinct species,
diﬀusion at the interface is neglected and the operative conditions are in
the limit of zero Jacob numbers (Ja→0), that is without phase change.
For every test case that will be presented in this paper the thermodynamic
conditions are chosen in order to have a liquid phase and a vapour phase far
from the two phase region coexistence conditions. With these hypothesis,
in each phase the density can be computed by solving Eq.(25) using for
instance the Cardan method as demonstrated in Appendix A. The sound
speed, which is mandatory to compute several terms in Eq.(20), can be
computed remarking that,
c2 =
∂p
∂ρ

s
=
∂p
∂ρ

T
+
T
ρ2Cv
 ∂p
∂T
2
ρ
.
(31)
Other expressions can be found to compute the sound speed but the previous
one is well suited for the van der Waals EoS. Indeed, from Eq.(29), the
following thermodynamic derivatives can be easily computed,
∂p
∂ρ

T
=
RT
1 −ρB

1 +
Bρ
1 −ρB

−2ρA,
(32)
 ∂p
∂T

ρ
=
ρR
1 −ρB .
(33)
Combining Eqs.(32) and (31) the following complete expression of the sound
speed from the van der Wall EoS is obtained,
c =

RT
(1 −ρB)2

1 + R
Cv

−2ρA
1/2
.
(34)
The speciﬁc heat at constant volume cv is related to the c0
v at low density,
which only depends on the temperature, by the relation:
cv −c0
v =
Z ρ0
ρ
T
ρ2
 ∂2p
∂T 2

ρ
dρ,
(35)
which, for a cubic EoS, gives cv = c0
v. Constant values for c0
v are considered
in this work. To complete the thermodynamical system deﬁned in Eq.(20),
the Mayer relation can be used to compute the speciﬁc heat at constant
pressure cp,
cp = cv + Tα2
ρβ ,
(36)
11


where α and β are the isobaric and isothermal expansion coeﬃcients,
α = −1
ρ
 ∂ρ
∂T

p
,
β = 1
ρ
∂ρ
∂p

T
.
(37)
As the van der Waals EoS does not allow to deﬁne a simple expression
ρ(T, p), the following expressions are proposed to determine more easily the
coeﬃcients α and β,
α = 1
ρ

∂p
∂T

ρ

∂p
∂ρ

T
,
β =
1
ρ

∂p
∂ρ

T
.
(38)
3. Numerical solver
Most two-phase ﬂows imply low-Mach number conditions, especially in
the liquid phase. There is therefore signiﬁcant interest in developing a solver
for compressible two-phase ﬂows for which the terms responsible for acoustic
wave propagation are discretized with an implicit time scheme. An inter-
esting property of the entropy-pressure based formulation, previously pre-
sented, is the possibility to handle separately the acoustics terms and the
convection terms. This approach diﬀers from more classical formulations
of the conservation energy equation, based on internal energy or enthalpy,
for which the eigenvalues of the Jacobian matrix must be computed to de-
termine the characteristics variables of the hyperbolic system. As a result,
an implicit temporal discretization of the acoustical waves will be simpliﬁed
with the proposed approach since it will only result in a linear system to
solve the pressure equation, whereas more classical approaches would require
to solve a non-linear system coupling all the equations.
3.1. Single phase semi-implicit compressible solver
In this subsection an algorithm that can be used to describe any single-
phase subsonic ﬂow, is presented. The algorithm is suitable either for ﬂows
for which compressible eﬀects or density variations can be signiﬁcant or not,
while maintaining an unconditionally stable temporal discretization related
to acoustical waves propagation. The following elementary ﬁrst order tem-
poral discretization can be proposed for the whole system of equations,
ρn+1 −ρn
∆t
+ ∇· (ρ⃗u)n = 0,
(39)
12


un+1 −un
∆t
+ un · ∇⃗un = −∇pn+1
ρn+1 +
1
ρn+1 ∇· τ n + ⃗g,
(40)
pn+1 −pn
∆t
+un·∇pn+(ρc2)n∇·⃗un+1 =
c2α
cp
n
(τ n ⊗∇⃗un −∇· ⃗qn) , (41)
which is implicit for the acoustical terms.
By carrying out the velocity
splitting hereafter, which is classical in the framework of the projection
method for the incompressible ﬂow,
un+1 = u∗−∆t∇pn+1
ρn+1 ,
(42)
the following 4 steps algorithm can be obtained for the whole system.
First, solve the density ﬁeld,
ρn+1 = ρn −∆t∇· (ρ⃗u)n.
(43)
Next, the intermediate velocity ﬁeld u∗and the intermediate pressure p∗are
updated,
u∗= un −∆t

un · ∇⃗un −
1
ρn+1 ∇· τ n −⃗g

,
(44)
p∗= pn −∆t⃗un · ∇⃗pn.
(45)
Then, by injecting Eq.(42) in Eq.(41), one obtains the following Helmholtz
equation for the pressure ﬁeld,
pn+1−(ρc2)n∆t2∇·
∇pn+1
ρn+1

= p∗+∆t(ρc2)n∇·⃗u∗+∆t
c2α
cp
n
(τ n ⊗∇⃗un −∇· ⃗qn) ,
(46)
which can be solved as a linear system for the pressure ﬁeld and enables an
implicit temporal discretization of the acoustical waves. It is noteworthy,
that the resulting matrix is not symmetric deﬁnite positive. Indeed, one can
verify that if a spatially variable coeﬃcient appears in front of the Laplacian,
the matrix resulting from its spatial discretization is not symmetric. The
matrix symmetry can be easily retrieved by dividing all the terms of the
previous equation by (ρc2)n. The following equation is obtained,
pn+1
(ρc2)n −∆t2∇·
∇pn+1
ρn+1

=
p∗
(ρc2)n +∆t∇·⃗u∗+∆t
 α
ρcp
n
(τ n ⊗∇⃗un −∇· ⃗qn) ,
(47)
13


which results in a symmetric deﬁnite positive linear system, if a centered
scheme is applied to discretize the Laplace operator. Finally, the velocity
ﬁeld accounting for pressure eﬀects is updated with the following relation,
un+1 = u∗−∆t∇pn+1
ρn+1 .
(48)
To understand the choice that has been made to consider (ρc2)n instead
(ρc2)n+1 in Eq.(47) it is useful to consider the following equality:
1
ρc2
∂p
∂t =
∂

p
ρc2

∂t
−p
∂

1
ρc2

∂t
.
(49)
In order to verify the best choice to write the discretized form of the left
hand side of Eq.(49), a ﬁrst order discretization is operated over the right
hand side:
1
∆t
" p
ρc2
n+1
−
 p
ρc2
n
−pn+1
 1
ρc2
n+1
+ pn+1
 1
ρc2
n#
= 1
∆t
 pn+1
(ρc2)n −
pn
(ρc2)n

,
(50)
where pn+1 is considered in the development of the second term. Similarly,
if pn is considered instead of pn+1 the following equality is obtained:
1
∆t
" p
ρc2
n+1
−
 p
ρc2
n
−pn
 1
ρc2
n+1
+ pn
 1
ρc2
n#
= 1
∆t

pn+1
(ρc2)n+1 −
pn
(ρc2)n+1

.
(51)
Therefore, Eqs.(50) and (51) demonstrate that considering (ρc2)n in Eq.(47)
corresponds to consider pn+1 in the development of the second term in the
right hand side of Eq.(49) instead of pn. Moreover, this choice for the tem-
poral index of 1/(ρc2) in Eq.(47) is consistent with the explicit nature of the
terms multiplied by it in the right hand side, that is p∗and the non-isentropic
terms. One interest of the overall formulation is to remove the time step
constraint due to the acoustical wave propagation since an implicit temporal
discretisation of the acoustic term is performed. This can be justiﬁed by con-
sidering a one-dimensional linearized form of the pressure evolution equation
Eq.(46) which is similar to an unsteady convection-diﬀusion equation with
a source term f,
∂p
∂t + a∂p
∂x = D ∂2p
∂x2 + f,
(52)
where the pseudo-diﬀusion coeﬃcient D = c2∆t depends on the sound speed
and on the time step.
Applying a fully explicit temporal scheme and a
14


standard centered ﬁnite diﬀerence scheme, the following suﬃcient stability
conditions can be stated, from von Neumann stability analysis, as detailed
in [46],
2D∆t
∆x2 ≤1,
a2∆t2
∆x2
≤2D∆t
∆x2
(53)
from which the following stability conditions can be deduced,
∆t ≤∆x
√
2c,
a ≤
√
2c ,
(54)
which are valid if the Mach number Ma = a/c ≤
√
2, as stated by the second
condition. To the authors knowledge, a more complete analysis involving
higher Mach numbers is not available and is out of the scope of this pa-
per that focuses only on subsonic ﬂows. This simpliﬁed analysis shows that
an explicit temporal integration of Eq.(46) would require a stability condi-
tion based on the acoustical time step, which is usual for standard explicit
compressible solvers. On the other hand, if a ﬁrst order implicit temporal
scheme is applied to the diﬀusion term and a ﬁrst order explicit temporal
scheme to the convection term, as proposed in Eq.(46), it can be found that
the stability condition is alleviated and becomes,
∆t ≤∆x
a .
(55)
This condition is much less stringent, especially in the low Mach number
regime, than the time step Eq.(54) of the explicit counterpart. This justiﬁes
brieﬂy why the acoustic time step does not need to be imposed to ensure
stability in the computations carried out with the proposed solver. Another
interesting property of this solver can be highlighted by remarking in Eq.(46)
that in the limits of c2 →∞, µ →0, and k →0, the pressure equation tends
to the pressure equation for incompressible ﬂows in a classical projection
method,
∇·
∇pn+1
ρn+1

= ∇· ⃗u∗
∆t .
(56)
This means that, in the limits of low Mach number (Ma →0), high Reynolds
number (Re ≫1) and high Peclet Number (Pe ≫1), the present numerical
scheme will be asymptotically preserving relatively to the incompressible
regime. The interest of such a nice property will be demonstrated in the
results section by a direct comparison between a standard incompressible
two-phase ﬂows solver and the proposed compressible two-phase ﬂows solver.
15


3.2. Implicit treatment of the thermal diﬀusion source term
In Eq.(47) the heat conduction is treated with an explicit temporal
scheme and appears in the right hand side. The thermal ﬁeld T n is com-
puted from the pressure pn and the density ρn making use of the EoS Eq.(30).
Another possibility is to treat the heat conduction term in an implicit way
by considering ∇T n+1 instead of ∇T n. Indeed we can express T n+1 as a
function of ρn+1 which is known from the continuity equation propagation
Eq.(43), and pn+1 which is the unknown of Eq.(47) making use of the EoS
Eq.(30):
T n+1 = 1 −Bρn+1
Rρn+1
pn+1 + Aρn+1
R
(1 −Bρn+1) ,
(57)
which in the following will be written in a more compact way as:
T n+1 = Cn+1pn+1 + Dn+1 ,
(58)
where Cn+1 = (1 −Bρn+1)/(Rρn+1) and Dn+1 = Aρn+1(1 −Bρn+1)/R are
functions of ρn+1 only.
By replacing Eq.(58) in Eq.(47) (considering Eq.(6)) one gets the fol-
lowing equation where the heat conduction term is treated with an implicit
temporal scheme:
pn+1
(ρc2)n −∆t2∇·
∇pn+1
ρn+1

−∆t
 α
ρcp
n
∇· k∇
 Cn+1pn+1
= RHS ,
(59)
RHS =
p∗
(ρc2)n + ∆t∇· ⃗u∗+ ∆t
 α
ρcp
n  τ n ⊗∇⃗un + ∇· k∇(Dn+1)

.
(60)
Note that in Eq.(59) the thermal ﬁeld does not appear explicitly any-
more but through the two primitive variables ρ and p . The present approach
is made possible by the proposed formulation of the algorithm based on a
primitive variables formulation. Note that a linear relation between T and
p in the EoS, as for Van der Waals and perfect gas EoS, is required in order
to maintain a linear system for the pressure equation. Being able to treat
the temperature ﬁeld with an implicit scheme will have very beneﬁcial con-
sequences for future developments, including phase change problems which
will be the subject of future works.
16


3.3. Operator split formulation for reducing surface tension driven parasitic
currents
In this section, the algorithm for a two-phase ﬂow is presented.
As
pointed out in [20], spurious currents may occur when adding directly the
surface tension force as a jump on the pressure ﬁeld when solving a Helmhotz-
type equation, as Eq.(47).
In order to numerically solve the system of
Eqs.(18)-(20), while preventing from spurious velocities, a splitting of the
primitive variables and of the momentum equation is introduced, following
the idea yet developed in [20].The idea is to isolate the eﬀects of the cap-
illary terms in one equation associated to an incompressible ﬁeld, which is
characterized by a soleinodal velocity ﬁeld. Eventually, if the following split
is operated on the primitive variables,
ρ = ρ0 + ρst ,
⃗u = ⃗u0 + ⃗ust ,
p = p0 + pst ,
(61)
where ust is the solenoidal part of the velocity ﬁeld, we have ∇.⃗ust = 0
and ρst = const. For simplicity we will set ρst = 0. The above splitting is
inserted into the system of Eqs.(18)-(20) which is rewritten in Eqs.(62)-(65),
∂ρ
∂t + ∇· (ρ⃗u) = 0 ,
(62)
∂⃗u0
∂t + ⃗u · ∇⃗u + ∇p0
ρ
= 1
ρ∇· τ ,
(63)
∂⃗
ust
∂t + ∇pst
ρ
= σκ⃗nδΓ
ρ
,
(64)
∂p0
∂t + ∂pst
∂t + ⃗u · ∇p0 + ⃗u · ∇pst + ρc2∇· ⃗u0 = c2α
cp
[τ ⊗∇⃗u −∇· ⃗q] . (65)
The momentum equation Eq.(19) has been split into two equations: Eq.(64)
which contains the capillary eﬀects and the soleinodal part of the velocity
ﬁeld ⃗ust, and Eq.(63) which contains the non soleinodal part of the velocity
ﬁeld ⃗u0 and the viscous terms.
A level set function φ, deﬁned as the signed distance from the interface
(φ > 0 in the liquid, φ < 0 in the gas and φ = 0 at the interface), is
introduced to describe the movement of the interface [37] with the following
convection equation,
∂φ
∂t + ⃗uint · ∇φ = 0,
(66)
where ⃗uint is the interface velocity. A reinitialization algorithm, as proposed
in [47], is used to ensure that the φ-function maintains the signed distance
17


property at the interface, all along the computation. Both the normal vector
⃗n and the curvature κ can be evaluated from the level set function φ using
Eqs.(67),
⃗n =
∇φ
||∇φ||,
κ = −∇· ⃗n.
(67)
3.4. Semi-implicit projection algorithm for two-phase ﬂows
The system of Eqs.(62)-(65) with Eq.(66) is discretized on a Cartesian
staggered grid and in this section the temporal numerical algorithm em-
ployed to locally solve it is reported step by step. We will indicate with the
upper script n the time index, and ∆t is the time step.
1. The level set function is updated,
φn+1 = φn −∆t ⃗un · ∇φn.
(68)
2. The density ﬁeld is updated,
ρn+1 = ρn −∆t ∇· (ρ⃗u)n.
(69)
3. Since ∇.⃗ust = 0, a projection method is applied and pst can be evalu-
ated solving the linear system resulting from the discretization of the
following Poisson equation,
∇·
∇pst
ρ
n+1
= ∇·
σκ⃗nδΓ
ρ
n+1
.
(70)
4. Predictions p∗and ⃗u0∗are evaluated considering only the convection
eﬀects for the pressure p, and also the viscous and volumetric forces
for the velocity,
p∗= pn −∆t(⃗un · ∇pn),
(71)
⃗u0∗= ⃗u0n −∆t[(⃗un · ∇⃗un) −
1
ρn+1 ∇· (µ(∇⃗un + ∇⃗unT ) −2
3µ∇· ⃗un
0I) −⃗g].
(72)
5. A projection method is also applied in order to calculate the rest of
the pressure ﬁeld p0. Speciﬁcally using Eq.(63) and the u∗
0 evaluated
in the previous step we have,
un+1
0
= u∗
0−∆t
∇p0
ρ
n+1
→
∇·un+1
0
= ∇·u∗
0−∆t∇·
∇p0
ρ
n+1
.
(73)
18


Therefore, an Helmholtz equation for the pressure p0 is found and pn+1
0
is the solution of the following linear system,
(p0)n+1
(ρc2)n −∆t2∇·
∇p0
ρ
n+1
= p∗−pn+1
st
(ρc2)n
−∆t∇·⃗u∗
0+∆t
 α
ρcp
n
[τ n ⊗∇⃗un −∇· ⃗qn] .
(74)
6. Finally, the velocity ﬁeld is corrected with the following step,
⃗un+1 = ⃗un
st + ⃗u∗
0 −∆t
"∇p0
ρ
n+1
+
∇pst
ρ
n+1
−
σκ⃗nδΓ
ρ
n+1#
.
(75)
A Ghost Fluid method is employed in order to handle the sharp jump across
the interface [17, 25, 29, 33, 48]. Speciﬁcally, ghost ﬁelds are deﬁned for ρ and
pst, and constant extrapolations are computed following the methodology
described in [4]. For what concerns spatial discretization, staggered grids
are used for the velocity ﬁelds relative to the centered grid for the other
variables. Convective terms are evaluated with ﬁfth order WENO Z schemes
[5], other spatial derivatives are approximated with standard second order
centered schemes. One interesting feature of the proposed algorithm is to
be compatible with previous famous works in the community focusing on
numerical solvers for two-phase incompressible ﬂows. See for instance [33,
25, 48] for more details on spatial schemes to handle jump conditions across
the interface. The linear systems (70) and (74) are solved with a Black-
Box MultiGrid solver [15]. The temporal integration described previously
is based on ﬁrst order IMplicit-EXplicit (IMEX) Euler scheme. This can be
used as an elementary step to build a second order or third order Runge
Kutta scheme. However, if a standard explicit Runge-Kutta scheme is used,
this will just improve stability, but not accuracy. Indeed, a speciﬁc IMEX
Runge Kutta scheme would be required, as those proposed in [3, 6], for
achieving higher order of the temporal discretization. The development of
such higher order temporal discretizations for the proposed compressible
projection method will be the object of future works.
3.5. Temporal stability condition
The viscous and diﬀusion terms are computed explicitly.
Eventually,
the time step ∆t is limited by classical stability conditions on convection,
viscosity and surface tension, that is it has to respect the overall condition,
1
∆t =
1
∆tconv
+
1
∆tµ
+
1
∆tσ
,
(76)
19


where ∆tµ, ∆tconv and ∆tσ come from the classical stability conditions on
viscosity, convection and surface tension respectively and are given by,
∆tconv =
∆x
max||⃗u||,
∆tµ = 1
2
ρ∆x2
µ
,
∆tσ = 1
2
r
max(ρ)∆x3
σ
.
(77)
More details on the numerical methodologies implemented and validated in
the DIVA code can be found in the following references: [20, 49, 50, 51,
29, 32, 45, 36, 56, 55]. Several validations of the numerical methods with
experimental results can also be found in the following references [12, 11, 28].
4. Test cases: validation and demonstration
Two single-phase test cases are carried out to verify two aspects: the cor-
rect propagation of acoustic waves and the ability of the solver to simulate
compressible ﬂows at low Mach numbers where thermal eﬀects are of ma-
jor importance, such as free convection problems. Secondly, two-phase ﬂow
tests are carried out to verify the correct treatment of interfaces with large
deformations and the coupling between interface deformation and thermo-
dynamic eﬀects. Finally, a demonstration of the coupling between acoustic
waves and interface deformation will be presented.
4.1. Acoustic pulse propagation
The aim of this test case is to verify the ability of the solver in properly
describe acoustic waves propagation. We consider a non viscous and non
conductive perfect gas: under these hypothesis the ﬂow is isentropic. The
domain is a one-dimensional line, of length L =1 m with periodic boundary
conditions. The base ﬂow is at rest, u0 = 0, the pressure is p0 = 1 × 105 Pa,
the temperature is T0 = 300 K which for air corresponds to a density of ρ0
= 1.117 kg.m−3. The base ﬂow is initially perturbed by superimposing a p′
ﬁeld with a Gaussian shape,
p′ = ∆p0e−1
2( x
Σ)
2
,
(78)
with Σ = 0.1 m and ∆p0 = 0.1p0 for the present test case. Considering an
acoustic wave travelling towards positive x, the associated velocity pertur-
bation ﬁeld is,
u′ =
p′
ρ0c0
,
(79)
20


where c0 = 347.12m.s−1 is the speed of sound corresponding to the base
ﬂow. Finally, the initial density ﬁeld perturbation is imposed considering
the isentropic relation,
ρ′ = p′
c2
0
.
(80)
The initial solution is given by:
p = p0 + p′ ,
u = u′ ,
ρ = ρ0 + ρ′ .
(81)
The initial solution should theoretically be retrieved when the pulse, that
should travel with a c0 speed, has browsed a length L.
The theoretical
elapsed time is computed according to the following equation,
τ = L
c0
(82)
which gives τ =0.046 ms.
The test case is performed with a ﬁrst order
temporal integration. First, a space convergence analysis is carried and the
simulation is done using either 128, 256, 512 or 1024 cells.
An acoustic
Courant-Friedrichs-Lewy number, CFLa, is introduced,
CFLa = c0
∆t
∆x,
(83)
where ∆t and ∆x are the time step and the mesh cell size respectively. The
solver is implicit and no restriction associated with acoustic propagation is
required in order to ensure stability, therefore the simulation can be car-
ried out even with CFLa >1 . However, in order to properly observe the
acoustic propagation, we expect that the time step should be small enough.
Therefore, a CFLa = 0.5 is considered for the convergence analysis. Results
at t = τ are shown in Fig.1 for the 512 elements mesh, and compared with
the initial condition (solid thick line) which is also the theoretical solution
at t = τ. Results for the other meshes are not reported because almost su-
perimposed and diﬀerences between the curves are less than 0.0001%. This
demonstrates the independence of the results with respect to the grid resolu-
tion for the present meshes. Therefore, all the other simulations are carried
out with a 512 elements mesh. At t = τ the pulse retrieves the initial po-
sition, centered at x = 0, in agreement with the theory. This demonstrates
that the solver correctly reproduces the acoustic wave propagation at the
correct speed.
A parametric study is then carried out, varying the time step accordingly
to a variation of CFLa between 0.0625 and 8 and results at t = τ are
21


x [m]
p’ [Pa]
-0.4
-0.2
0
0.2
0.4
0
20
40
60
80
100
8
4
2
1
0.5
0.25
0.125
0.0625
IC
Figure 1: Pressure perturbation ﬁeld at the initial position (thick solid line, IC)
and after one period, at t = τ, for diﬀerent time steps (based on a
acoustic CFLa).
shown in Fig.1. The computation is stable even for the CFLa > 1 thus
demonstrating the expected stable behaviour (see Section 3.1). The pulse
retrieves its initial position at t = τ for CFLa < 1, and for CFLa > 1 a
slight shift of the position is observed with respect to the theoretical solution
thus indicating that the propagation speed is lower than the theoretical one.
For every CFLa, the amplitude of the pulse is lower than the theoretical
one but numerical dissipation strongly increases for CFLa > 1 and decreases
when CFLa decreases: the dissipation is 40 % for CFLa =1 and 5 % for
CFLa = 0.0625. As the order of convergence is around one, one could expect
higher accuracy by introducing a higher order IMEX temporal scheme, as
the one proposed in [3]. This will be the object of future works.
This test case validates the proposed solver for the simulation of acous-
tic waves propagation, and shows the necessity to use proper CFLa if the
objective is to follow the acoustic waves evolution, even if the computation
remains stable also for CFLa ≫1.
4.2. Natural convection in a square
The objective of the present test case is to test the ability of the solver in
reproducing a ﬂow induced by a temperature gradient under the inﬂuence
of the gravitational acceleration ﬁeld.
The test case of heat transfer by
22


natural convection in an enclosure presented by Le Qu´er´e et al.
[43] is
considered.
The domain is a square 2D cavity of side L, ﬁlled will air
initially at temperature T0 and pressure p0 (corresponding to a density ρ0),
and is shown in Fig.2(a). Bottom and up walls are adiabatic, left and right
walls are isotherm, at temperatures Twh and Twc respectively.
The left
wall is hotter than the right wall (Twh > T0 > Twc), resulting in a horizontal
temperature gradient. The gravity acceleration induces a upward movement
of hot air and downward movement of cold air.
T [K]
y
x
a)
b)
Twh
Twc
qw = 0
qw = 0
L
g
upward 
movement
hot air
downward 
movement
cold air
Figure 2: Natural convection test case. a) Computational domain and boundary
conditions. b) Temperature ﬁeld and streamlines in stationary condi-
tions (t= 15s for the grid 512*512).
It can be demonstrated that the present test case is deﬁned by a couple
of parameters: the Rayleigh number Ra and the temperature ratio ϵ deﬁned
by the following equations,
Ra = Prgρ2
0(Twh −Twc)L3
T0µ2
0
,
ϵ = Twh −Twc
Twh + Twc
,
(84)
where Pr is the Prandtl number and µ0 the dynamic viscosity at T0. We
consider the test case with Ra=106, ϵ = 0.6 and constant thermophysical
properties (constant viscosity µ and thermal conductivity k), which is called
T1 in the work of Le Qu´er´e et al [43]. The ideal gas equation of state is used.
The test case characteristics are summarised in Table 1. They correspond
to a cavity of side L = 0, 0460307 m.
23


T0
p0
Ra
ϵ
Twh
Twc
Pr
µ
[K]
[Pa]
[-]
[-]
[K]
[K]
[-]
[µPa.s]
600
101325
106
0.6
960
240
0.71
16.8
Table 1: Parameters for the natural convection test case T1 from [43].
Local and averaged Nusselt numbers, Nu and Nu, can be computed over
the left and right walls:
Nu =
L
k(Twh −Twc)k∂T
∂x ,
Nu = 1
L
Z L
0
Nu(y)dy.
(85)
When the steady state regime is eventually reached, Nu over the left and
right walls become equal: Nuh = Nuc. The static pressure at steady state
will be lower than the initial p0 because of the mechanical energy involved
in the movement in the cavity. The simulation is carried out with three dif-
ferent homogeneous grids, with 128×128, 256×256 and 512×512 elements.
A snapshot of the temperature ﬁeld superimposed with velocity iso-lines is
shown in Fig.2b) for the ﬁner grid at t=15 s. We observe the stratiﬁcation
of the temperature ﬁeld and the re-circulation of the ﬂuid induced by the as-
cension of hot air and descent of cold air. Temporal evolutions of the average
pressure and of the diﬀerence Nuh −Nuc are shown for the free mesh resolu-
tions in Fig.3 over 15 seconds. A convergence of the results is observed and
the ﬁner grid results tend towards a stationary condition. In particular, af-
ter t=5 s the temporal variation of the results is very slow and the diﬀerence
between hot and cold Nu number is around 2.3 × 10−3 at t=15s. Quantita-
tive results obtained with the ﬁner grid are in agreement with the reference
provided by [43] as reported in Table 2, with diﬀerences smaller than 0.02%
for the ratio p/p0 and 0.12 % for Nuc,h. The same simulations have been
carried out considering an explicit or implicit treatment of the heat conduc-
tion term in the energy equation. Results are almost superimposed with
small diﬀerences only in the transient phase (t < 5 s), smaller than 0.003
% for the Nu numbers and 0.0005 % for the pressure, and no diﬀerences in
the stationary phase. Indeed the curves would appear as superimposed in
Fig.2 (where only the solutions obtained with an explicit heat conduction
term are reported). This demonstrates the correct implementation of the
numerical implicit scheme for the heat conduction term.
This section demonstrates that the semi-implicit compressible solver is
well suited to handle free convection ﬂows without any assumptions on the
24


Mach number. Moreover, in such conﬁgurations, the acoustic time step has
not to be imposed, since acoustic waves do not impact the ﬂow.
a)
b)
Nuh - Nuc
t [s]
t [s]
p/p0
Figure 3: Temporal evolution of the average pressure a) and the diﬀerence be-
tween cold and hot Nu numbers b) for the three diﬀerent grids.
reference
present
Nuh
8.85978
8.86885
Nuc
8.85978
8.87113
p/p0
0.856338
0.856515
Table 2: Natural convection test case results at t = 15s : comparison between
present numerical results with a grid (512x512), considering either an
explicit or implicit treatment of the heat conduction term, and the ref-
erence values from [43].
4.3. Falling drop over a liquid layer
The aim of the present test case is to verify the ability of the solver to
simulate two phase ﬂows, featuring strong interface deformations, at low
Mach number. The falling of a liquid drop on a liquid layer is considered
for this purpose and the experimental conﬁguration of Manzello and Yang
[34] is taken as a reference.
The computational domain is a cylinder of
height H = 40 mm and radius R = 40 mm (see Fig.4). The bottom half
of the domain is ﬁlled with water and the upper half is ﬁlled with air (the
interface is located at 20 mm from the bottom). The water density 1 is
1The coeﬃcients of the Van Der Waals EoS have been modiﬁed in order to obtain this
value of ρl for p = 1 atm and T = 293.15 K. In fact, by default, the Van Der Waals EoS
25


5ms – compressible solver
5ms – incompressible solver
25ms – compressible solver
25ms – incompressible solver
45ms – compressible solver
45ms – incompressible solver
65ms – compressible solver
65ms – incompressible solver
5ms
25ms
45ms
65ms
2R
H
Figure 4: Instantaneous snapshots of the interface obtained with compressible
(left) and incompressible (middle) solvers and corresponding experi-
mental (right) images from [34].
ρ = 1000 kg.m−3 and the dynamic viscosity is µ = 89µ.Pa. The surface
tension is σ = 0.072 N.m−1, corresponding to water in air at 293.15 K. A
free boundary condition is imposed at the top of the domain to allow the
ﬂuid to leave and enter freely, while bottom and lateral walls are adiabatic.
Two dimensional axisymmetric simulations are carried out with a homoge-
neous mesh of 1024 × 1024 elements. A drop of water is initially placed
near the interface with a velocity of v = 2.1908 m/s, which corresponds
to a Weber number We=123 (We=ρvD/σ), in order to reproduce the ex-
for water in these conditions would give ρl = 494kg.m−3.
26


perimental conditions in terms of impact velocity [34]. The simulation is
carried out both with the compressible solver and with the incompressible
solver of the DIVA code. In terms of computational time, the simulation
with the compressible solver takes twice as long as with the incompressible
one. Indeed, for the compressible solver the number of steps involved in the
algorithm is increased because of three main reasons: the Helmholtz-type
equation for the pressure Eq.(47), the equation for the density ﬁeld Eq.(69),
and the equations to compute the extrapolations for the ghost ﬁelds deﬁned
for ρ and pst (see section 3.4).
Numerical results obtained with the compressible solver are compared
with the results obtained with the incompressible solver and the experimen-
tal data in Fig.4 where temporal snapshots at 5, 25, 45 and 65 ms after the
drop impact are shown.
At t = 5 ms, just after the drop impact, compressible and incompressible
solvers give identical results that are in agreement with the experiment. At
t = 25 ms , a crater is hollowing out and surface waves generated by the
impact of the drop are observed both in numerical and experimental images.
At t = 45 ms, a thick, high jet appears, from which droplets are detached. It
seems that there is a time phase shift in the droplet detachments between the
incompressible, compressible and experimental images and as a consequence
the instantaneous number of droplets detached diﬀers. However, these slight
diﬀerences could be partly explained by the chaotic nature of the atomization
process. At t = 65 ms , the jet has fallen and droplets are suspended both
in the numerical and experimental cases, even thought the instantaneous
observable droplets suspended is diﬀerent (three in the numerical results
and one in the experiments). Comprehensively, this comparison shows that
the compressible and incompressible solvers provide results very close to
the experiment and demonstrate the capability of the compressible solver
in handling two phase ﬂows with strong interface deformations in the low
Mach number regime.
4.4. Expansion of a bubble in a variable pressure environment
The objective of this test case is to verify the correct implementation of
the thermodynamic models available in the solver. In particular, the aim is
to verify the coupling between density and pressure variations arising from
the EoS. Two diﬀerent EoS are available: the perfect gas equation, suitable
for low pressure and high temperature gas, and the van der Waals, which is
valid for both vapour and liquid phases over the entire domain of existence.
27


open end, p(t)
wall
open end, p(t)
symetry
L
R
gas
liquid
z
r
Figure 5: Computational domain and boundary conditions for the test case of
expansion of a bubble in an variable pressure environment.
A bubble expansion in a liquid subjected to a variable pressure ﬁeld is
simulated.
The simulation domain is shown in Fig.5: it is a cylinder of
height L =4 mm and radius R =2 mm, with two open ends and a lateral
adiabatic wall. A bubble is initially placed at the center of the domain and
the gravity acceleration ﬁeld is artiﬁcially set to zero. The pressure in the
domain is changed in time by applying a time varying Dirichlet condition for
the pressure at the open ends borders, following the linear equation given
by Eq.(86),
p = pi −pi −pf
tf
× t,
(86)
where pi is the initial pressure, pf the ﬁnal pressure when t = tf, tf being
the ﬁnal time of the simulation. When the pressure decreases, the bubble
volume should increase as a consequence of the density bubble decrease. By
varying tf and keeping the same initial and ﬁnal pressure values, pi and pf,
the pressure variation speed is modiﬁed. Small tf will induce rapid bubble
expansions and large tf will induce slow bubble expansions. Analytically,
the relation between the pressure and the bubble volume can be expressed
28


using an isothermal or an isentropic transformation relation. A very slow
expansion is expected to tend toward an isothermal transformation because
a thermal equilibrium between the bubble and the surrounding liquid will
have time to be established. On the other hand, fast bubble expansions
are expected to follow an adiabatic transformation thus tending towards an
isentropic transformation. Indeed, one can neglect the irreversibilies in such
cases of fast expanding bubbles since the ﬂow is almost irrotational and the
expansion is faster than the characteristic time scale associated with thermal
diﬀusion. Therefore, we expect a real expansion to evolve between the ideal
isentropic and isothermal transformations. The objective of the present test
case will be to verify this behaviour for two diﬀerent sub-cases:
• a low pressure case, for which the perfect gas EoS can be used to
describe the vapour phase;
• a high pressure case, for which the perfect gas hypothesis does not
hold and the van der Waals EoS is used for the vapour phase.
In both cases, the van der Waals equation is required to describe the liquid
density, which is almost constant. In the ﬁrst case, the density in the liquid
will be around ρl = 493.93 kg.m−3 which is the solution of the van der
Waals EoS for water at ambient pressure and temperature. In the second
case the liquid density will be around ρl = 200 kg.m−3, which is the solution
of the van der Waals EoS for methane in a liquid state. It is noteworthy
that more accurate solutions for liquid densities can be obtained by using
more complex cubic EoS, as Peng-Robinson or Redlich-Soave-Kwong, for
instance. This will be the subject of future investigations.
4.4.1. Expansion at low pressure: perfect gas EoS description
Isothermal and isentropic transformations for a perfect gas can be ex-
pressed with the relations given by Eqs.
(87) and (88) between speciﬁc
volume v and pressure p:
isotherm :
v = pivi
p
,
(87)
isentropic :
v = vi
pi
p
1/γ
,
(88)
where subscript i refers to initial values, and γ is the ratio between the spe-
ciﬁc heats. We are neglecting mass diﬀusion through the interface, therefore
the bubble constitutes a closed system with a constant mass that is con-
served in time. As a consequence, Eqs. (87) and (88) hold for the bubble
29


p [bar]
a)
b)
v/v0
p [bar]
v/v0
Figure 6: Non-dimensional speciﬁc volume as a function of the average pressure
in the bubble, obtained with a tf = 50 ms, with 3 diﬀerent grids, and
comparison with the theoretical isentropic and isothermal transforma-
tions using a perfect gas EoS. Vapour: air. Liquid: water.a) Results
over the entire pressure range. b) Closer view.
volume.
A bubble of air in liquid water is considered. Initial and ﬁnal
pressures are pi = 2 bar and pf = 0.5 bar, and the initial temperature is
Ti = 300 K. In these conditions, a perfect gas EoS can be used to describe
the vapour phase. A ﬁrst simulation is carried out with tf = 50 ms, using
three diﬀerent grids containing 128×256, 256×512 and 512×1024 elements.
In this conﬁguration the stability time step is constrained by viscous eﬀects,
and the time steps in the simulations are chosen accordingly. The bubble
volume evolution versus the average pressure in the bubble is shown, for the
three grids, in Fig.6, where the isentropic and isothermal relations given by
Eqs.(87) and (88) are also reported. Results obtained with the three grids
are almost superimposed and behave as expected between the isothermal
and isentropic transformations, though much closer to the isothermal for
this tf. The closer view in Fig.6.b) shows that the ﬁner grid results tend to-
wards the isothermal relation. These results show a grid convergence for the
present setup, with diﬀerences for the bubble radius smaller than 9 × 10−3
%. For the following simulations, the coarse grid (128 × 256 elements) will
be used.
A parametric study is carried out, varying tf between 0.05 and 90 ms.
Results are shown in Fig. (7.a) over the entire pressure variation domain
and a closer view is proposed in Fig. (7.b). The bubble evolution tends
toward the isothermal expansion for large tf and the isentropic expansion
for low tf conﬁrming the theoretical expectations.
30


a)
b)
p [bar]
v/v0
v/v0
p [bar]
tf
Figure 7: Evolution of the bubble volume v as a function of the average pressure
p for diﬀerent speed of pressure temporal variation (higher tf imply
slower pressure variations.
a)
b)
tf =0.05 ms
T [K]
T [K]
tf =90 ms
p [bar]
T [K]
p [bar]
T [K]
Figure 8: Temperature ﬁelds reached at t = tf and corresponding average tem-
perature for a) a rapid bubble expansion (tf = 5ms) and b) a slow
bubble expansion (tf = 90ms).
31


The evolution of the average temperature in the vapour bubble and in
the liquid is shown as a function of the average pressure in the vapour
in Fig.8.a) for a rapid expansion (tf = 0.05 ms) and Fig.8.b) for a slow
expansion (tf = 90 ms). The temperature ﬁeld at the end of the simulation
is also reported.
For both cases the temperature in the liquid is almost
constant and equal to Ti =300 K. On the other hand, the behaviour of
the temperature in the vapour is very diﬀerent. For the faster case, the
temperature in the bubble undergoes a huge drop, a temperature around 200
K is reached in the center of the bubble center and a thin thermal boundary
layer is formed around the bubble.
The fast expansion of the bubble is
quasi adiabatic thus inducing a huge cooling of the gas. On the contrary,
the temperature in the bubble for the slow case is almost constant and only
slightly decreases during the expansion. The ﬁnal average temperature is
297 K and the minimum temperature is 293 K at the center of the bubble :
the slow expansion of the bubble is quasi isothermal.
4.4.2. Expansion at high pressure: Van de Walls EoS
For a real gas, the perfect gas equation is no longer valid. Isothermal and
isentropic transformations for a real gas can be expressed with the relations
Eqs.(89) and (90) between v and p, derived using the van der Waals EoS
Eq.(27):
isotherm :
pv3 −v2(pB + RT) + vA −AB = 0 ,
(89)
isentropic :
p =
K(R/cv)
(v −B)(1+cv/R) −A
v2 ,
(90)
where cv is the heat capacity at constant volume, R is the gas constant, A
and B are given by Eq.(27) and K is a constant that can computed from
vi and pi (see Appendix (27) for details on Eq.(90)). For the simulation in
real gas regime, a nitrogen bubble in liquid methane at Ti = 150 K is con-
sidered and the pressure is varied from pi = 40 bar to pf=10 bar. In these
conditions real gas eﬀects cannot be neglected and the van der Waals EoS is
used to describe both the vapour and the liquid phases. Two diﬀerent small
tf are considered in order to reproduce an adiabatic expansion: 0.2 and 1
ms. Results are shown in Fig. 9a), where both the real gas Eq.(90) and
perfect gas Eq.(88) isentropic transformations are also reported for compar-
ison. It appears that the two theoretical curves diﬀer, as expected in the
present high pressure conditions. The numerical results demonstrate that
decreasing tf, the bubble expansion tends towards the van der Waals EoS
isentropic transformation, as it was expected. The same approach is devel-
oped for the isothermal expansion. Two diﬀerent tf are considered in order
32


a)
b)
p [bar]
v/vi
v/vi
p [bar]
Figure 9: Evolution of the speciﬁc volume as a function of the average pressure
in the vapour for a bubble of nitrogen in liquid methane: numerical
results (color curves) and theoretical evolution using a perfect gas or a
Van der Waals EoS. a)Rapid bubble expansion that tends towards an
isentropic transformation. b) Slow bubble expansion that tends toward
the isothermal transformation.
to reproduce an isothermal expansion, i.e. 1 and 20 s. Results are shown in
Fig.9.b), where both the real gas (Eq.(90)) and perfect gas (Eq.(88)) isother-
mal transformations are also reported for comparison. When tf increases,
the bubble expansion tends toward the real gas isothermal transformation.
In conclusion, the above results demonstrate the correct behavior of the ther-
modynamic models in the compressible solver and the ability to account for
real gas eﬀects.
4.5. Drop shape evolution in an acoustic ﬁeld
In this test case the interaction between a droplet and an acoustic wave
is computed with an asymmetric simulation in order to observe the deforma-
tion and potential breakup of the droplet induced by acoustic waves. The
setup is shown in Fig.10. The domain is a cylindrical cavity of height L =
8 mm ﬁlled with air at T0 = 300 K and p0 = 101325 Pa. A water droplet
with an initial radius of r = 1 mm is positioned at the center of the domain
(r = z = 0). The gravity acceleration is artiﬁcially set to zero. The bottom
and lateral boundaries are adiabatic walls whereas the upper boundary is
an open end and an acoustic oscillation boundary condition is imposed,
p(t) = p0 + ∆p sin(ωt),
(91)
with ω = 2πf1L where f1L = c0/L is the ﬁrst acoustic mode frequency of
the cavity, with c0 = 347 m/s speed of sound in air at the present conditions.
33


p [bar]
z
r
L
t
wall
open end
p(t) = p0+p’(t)
wall
symetry
t + ¼ !
t + ½ !
t + ¾ !
z=L/2
Figure 10: Setup for the study of the drop shape evolution in an acoustic ﬁeld.
Evolution of the pressure and velocity ﬁelds over one temporal period
for the pressure oscillation.
With the above conditions, the pressure wave frequency is f1L= 21695 Hz.
The pressure wave amplitude is set to 2%p0 that is ∆p = 2026.5 Pa. By
imposing this boundary condition, a stationary acoustic wave is maintained
in the domain.
The setup is shown in Fig.
10, where the pressure and
velocity ﬁelds evolutions over one oscillation period τ (τ = 1/f1L) are shown.
The pressure wave propagates in time, both in the gas and in the liquid:
the pressure in the drop is slightly greater than in the gas, by a diﬀerence
corresponding to the surface tension (for the initial circular drop the pressure
jump across the interface is constant and equal to ∆pst = 2σ/r = 116 Pa).
The pressure nodal line, which corresponds to the axial velocity anti-nodal
line, is at the center of the domain (dashed line, z = L/2). As a consequence,
during one period, the drop is subjected to an oscillating velocity ﬁeld in
the axial direction (z). The surface tension for the case shown in this ﬁgure
is σ = 0.058N.s.
The maximum velocity at the nodal line, ∆v, can be
estimated considering an average impedance ρ0c0 of the volume of air at T0
and p0,
∆v = ∆p
ρ0c0
,
(92)
34


a) t + ¼ !
b) t + ¾ !
x [mm]
0
0.5
1
1.5
2
2.5
3
3.5
4
-8
-6
-4
-2
0
1024*2048
512*1024
256*512
v [m/s]
x [mm]
v [m/s]
0
0.5
1
1.5
2
2.5
3
3.5
4
0
2
4
6
8
p [bar]
1
2
3
4
1.0125
1.013
1.0135
1.014
1.0145
1.015
p [bar]
x [mm]
1
2
3
4
1.0125
1.013
1.0135
1.014
1.0145
1.015
x [mm]
Figure 11: Axial velocity evolution along the anti-nodal line (z = L/2) at two dif-
ferent times along one cycle corresponding to the maximum of velocity.
Results are shown for diﬀerent grid reﬁnements.
that results in a theoretical ∆v = 4.96 m/s, which corresponds to a Weber
number,
We = ρ0∆vD
σ
.
(93)
The axial velocity and pressure at z = L/2 are shown in Fig.11 at two diﬀer-
ent instants along one cycle corresponding to the maximum and minimum of
velocity oscillation (that is the second and fourth snapshots in Fig.10). The
ﬁrst observation that can be made from Figs.10 and 11 is that the pressure
waves propagate both in the gas and inside the liquid drop but induce a ve-
locity ﬁeld which is much more important in the gas, in agreement with the
higher impedance of the liquid ((ρlcl)/(ρvcv) = 2500 in the present case).
The axial velocity far from the drop reaches maximum absolute values of
4.7 m/s, as shown in Fig.(11), in agreement with the theoretical expected
value given by Eq.(92). The absolute velocity increases towards the drop
interface: this is a consequence of the acceleration of the gas ﬂow that goes
around the drop. In the drop the axial velocity falls to almost a zero value.
35


However, the tangential velocity is continuous across the interface and a
very thin boundary layer is established. The pressure jump across the in-
terface can be seen in Figs.11.a) and 11.b): it has a value of 110 Pa in
agreement with the theoretical value for the spherical drop at rest which is
(∆pst = 2σ/r = 116 Pa). The simulation has been carried out with several
grid reﬁnements in order to ensure a proper convergence analysis and results
are reported in Fig.11 for three uniform grids having a number of cells in
the r and z direction of: 256 × 512, 512 × 1024 and 1024 × 2048 correspond-
ing with cell sizes of 15 µm, 7.8 µm, and 3.9 µm, respectively. The axial
velocity ﬁeld is well captured with the three meshes and a convergence is
observed. An over-shoot and an under-shoot on the pressure are observed
with the coarser grids. These oscillations of the pressure at the interface
are associated with the pst ﬁeld that contains the jump across the interface
induced by the surface tension. If the mesh is not ﬁne enough, the interface
starts wrinkling since the beginning of the simulation, inducing oscillations
on the pst ﬁeld at the interface, in particular in this region where the veloc-
ity variation at the interface is greater. This phenomenon disappears when
increasing the mesh resolution and no oscillations are visible with the ﬁnest
grid. This behaviour is related to the well-known issue of parasitic currents
induced by surface tension, which can be reduced by reﬁning the grid.
The temporal evolution of the equivalent drop diameter is shown in
Fig.12.a) for the diﬀerent grids, over forty periods. The diameter decreases
for the coarser grid and a liquid mass loss around 10% is reached at t= 40
τ. The mass loss reduces while increasing the grid resolution showing that
the simulations converge towards a conservative solution. For the ﬁnest grid
the mass loss is lower than 1% at t=40 τ. Snapshots of the drop shape
at t=40 τ are shown in Fig.13 (We=1) and no signiﬁcant diﬀerence is ob-
served between the two ﬁnest grids. Considering the above observations, the
1024 × 2048 grid has be used to carry out the rest of the simulations for the
present test case.
Observing the drop shape at 40τ for We=1 in Fig.13,
it appears that the drop shape, which initially is spherical, only slightly
changes in time, in agreement with the low We number. In order to analyse
the impact of the acoustic ﬁeld on the interface deformation, a parametric
study has been carried out increasing the We number up to 40. This has
been achieved by decreasing the surface tension and keeping constant all the
other parameters. The Ohnesorge number, Oh= µl/(ρl ∗D ∗σ)1/2 is around
Oh=0.003. In these conditions, secondary atomization with a bag-breackup
mode is expected for We> 11 [23]. We do not aim to go up to atomization
in the present simulations and the axisymmetric hypothesis we make would
not allow it. However, we can simulate the strong drop shape modiﬁcation
36


Deq [mm]
0
5
10
15
20
25
30
35
40
1.97
1.98
1.99
2
1
4
10
20
40
0
5
10
15
20
25
30
35
40
1.94
1.96
1.98
2
1024*2948
512*1024
256*512
t / τ
t / τ
Deq [mm]
We
a)
b)
Figure 12: Temporal evolution of the drop equivalent diameter. a) Comparison
for diﬀerent grid reﬁnements for We=1 b)Comparison for diﬀerent We
number cases, ﬁne mesh 1024 × 2048.)
We = 1
We = 4
We = 10
We = 20
We = 40
initial 
condition
Figure 13: Shape at t=40τ of an initially spherical drop for diﬀerent We numbers.
preceding the atomization. In terms of characteristic time, the character-
istic time scale tc = D(ρl ρg)1/2/v associated with atomization is around 8
ms in the present conditions, thus t=40τ corresponds to t = 0.22tc. The
equivalent diameter Deq temporal evolutions over 40τ are shown in Fig.12.b)
and the corresponding snapshots of the drop interface at 40τ are shown in
Fig.13). For We=20 and We=40 the Deq shows a drop at 25 and 35 τ re-
spectively, while no particular drop in the diameter is observed for We≤10.
The diameter drop corresponds to a topological change in the drop shape as
observed in Fig.13. The acoustic velocity induces a deformation in the drop
shape that ﬂattens and eventually leads to the formation of two rims at an
instant that corresponds with the diameter drop early observed.
These results demonstrate the ability of the present solver in describ-
ing the interaction between an acoustic ﬁeld and a drop interface and open
the door for further analysis including three dimensional simulations of sec-
37


ondary atomization induced by an acoustic ﬁeld.
5. Conclusion
This paper presents an innovative numerical strategy for the study of
two-phase compressible ﬂows. The compressible solver proposed in this pa-
per has many interesting features. It is able to describe acoustic waves, but
it does not require to impose the stability constraint due to acoustic waves
propagation. Indeed, imposing the latter stability constraint can be pro-
hibitive in conﬁgurations with high density variations but for which sound
propagation does not play any role, as for instance free convection or the
expansion/compression of a bubble due to a pressure drop/increase. The
proposed formulation gives a clear framework to account for heat conduc-
tion and viscous eﬀects. It can be coupled to any equation of state, provided
the sound speed can be computed. It works as well for liquids as for gases. It
tends asymptotically to the incompressible projection method under usual
assumptions of incompressible solvers, but it is also well suited to simulate
low Mach number ﬂows with density variations, as free-convection ﬂows or a
bubble growth. Its generalisation to two-phase ﬂows is quite straightforward.
Even if an additional splitting on variables has to be carried out to impose
the surface tension, general state-of-art and powerful techniques previously
developed for incompressible two-phase ﬂows can be applied directly to this
two-phase compressible solver. Future works will tackle the generalization
of this solver to higher order temporal discretization schemes, that could be
based on IMEX Runge Kutta schemes. Coupling with extended physical
models, as liquid-vapor phase change, supersonic ﬂows, more complex equa-
tions of state and Immersed boundary methods, will be also investigated in
the future.
Appendix A. Cardan algorithm applied to compute the density
from the VdW EOS
In order to evaluate the density corresponding to a thermodynamic state
deﬁned by pressure p and temperature T, the roots of the Van der Waals
EoS which is a cubic equation, have to be computed. The method of Cardan
is employed. Let’s consider the form of the Van der Waals EoS given by
Eq.(25) with the coeﬃcients of Eq.(28). A new variable ˜ρ is introduced and
veriﬁes,
ρ = ˜ρ −a1
3 = ˜ρ + 1
3B .
(A.1)
38


Expressing Eq.(28) as a function of ˜ρ we obtain the following equation,
˜ρ3 + ˜a1˜ρ + ˜a0 = 0 ,
(A.2)
where the coeﬃcients ˜ai are given by,
˜a0
= −1
27

2
B2 −9p
A −9RT
AB

−
p
Ab ,
(A.3)
˜a1
= −
1
3B2 + p
A + RT
AB .
(A.4)
The cubic Eq.(A.2) has one, two or three real solutions depending on the
sign of the discriminant ∆= −(4˜a3
1 + 27˜a2
0). Moreover, ρ must be positive.
Crossing these requirements the following cases are possible,
• if ∆< 0, one solution given by ˜ρ1 = u + v where u and v are given by
the following relations,
u =

−˜a0 +
q
−∆
27
2


1/3
,
v =

−˜a0 −
q
−∆
27
2


1/3
;
(A.5)
• if ∆= 0, two solutions given by,
˜ρ1 = 3˜a0
˜a1
,
˜ρ2 = −3˜a0
2˜a1
.
(A.6)
If ρ1 > 0 two solutions are available and it will be ρv = ρ1 and ρl = ρ2.
If ρ1 < 0 there is only one solution: ρ = ρ2.
• if ∆> 0, three solutions given by,
˜ρk = 2
r
−˜a1
3
cos
"
1
3 cos−1
 
−˜a0
2
s
−27
˜a3
1
!
+ 2kπ
3
#
,
k = 1, 2, 3
(A.7)
If ρ1 > 0, there are two solutions, one for each phase: ρv = ρ1 and
ρl = ρ3. If ρ1 < 0 and ρ2 > 0 then there are two solutions, one for
each phase ρv = ρ2 and ρl = ρ3. If ρ1 < 0 and ρ2 < 0 there is a single
phase solution: ρ = ρ3.
Appendix B. Isentropic transformation for a real gas described by
the Van Der Waals EOS
This appendix demonstrates how to obtain the isentropic relation ex-
pression Eq.(90) for a ﬂuid described by the van der Waals EoS. Let’s recall
39


the ﬁrst Clapyeron relation,
δQ = cvdT + ldv,
(B.1)
where δQ is the heat absorbed during a reversible transformation, v = 1/ρ
is the speciﬁc volume and l is the isothermal dilation coeﬃcient deﬁned by,
l = T
 ∂p
∂T

v
.
(B.2)
The Clapyeron relation Eq.(B.1) can be combined with the ﬁrst thermody-
namic principle, δe = δQ −pdv, to express the internal energy variation,
de = cvdT + (l −p)dv.
(B.3)
The expressions for p and T given by the van der Waals EoS Eq.(27) can be
rewritten in terms of the speciﬁc volume v,
p + A
v2 =
RT
v −B ,
(B.4)
T = 1
R

p + A
v2

(v −B) .
(B.5)
Using the van der Waals EoS the dilation coeﬃcient l is given by,
l = p + A
v2 .
(B.6)
Replacing Eq.(B.6) in Eq.(B.3) gives,
de = cvdT + A
v2 dv.
(B.7)
For an isentropic transformation dS = 0 which gives de = −pdv, and thus,
cvdT+ = −(p + A
v2 )dv.
(B.8)
The right hand side of Eq.(B.8) is expressed using Eq.(B.4), thus obtaining
the following relation between dT and dv along an isentropic transformation,
cvdT = −RT
v −B dv,
(B.9)
which can be integrated between two states. Finally, the temperature T is
replaced by its expression as a function of p and v, Eq.(B.5), thus yielding to
the following relation between p and v along an isentropic transformation,

p + A
v2
α
(v −B)α+1 = K
with
α = cv
R ,
(B.10)
where K is a constant. Eq.(B.10) can be used to relate p and v between two
states along an isentropic transformation.
40


Acknowledgements
The authors gratefully acknowledge AID (Agence Innovation D´efense)
for funding the PhD thesis of Marie Bibal.
This work was granted access to the HPC resources of Curie/Occigen
under the allocation A0072B10285 by GENCI in France.
This work was supported by the Chair for Advanced Space Concepts
(SaCLab) resulting from the partnership between Airbus Defence and Space,
Ariane Group and ISAE-SUPAERO.
References
[1] M. Aanjaneya, S. Patkar, and R. Fedkiw. A monolithic mass tracking
formulation for bubbles in incompressible ﬂow.
J. Comput. Phys. ,
247:17–61, 2013.
[2] M. Arienti and M. Sussman. A numerical study of the thermal transient
in high-pressure diesel injection. Int. J. Multiphase Flow , 88:205–221,
2017.
[3] U. M. Ascher, S. J. Ruuth, and R. J. Spiteri. Implicit-explicit runge-
kutta methods for time-dependent partial diﬀerential equations. Ap-
plied Numerical Mathematics, 25:151–167, 1997.
[4] T. D. Aslam. A partial diﬀerential equation approach to multidimen-
sional extrapolation. J. Comput. Phys. , 193:349–355, 2003.
[5] R. Borges, M. Carmona, B. Costa, and W. Don. An improved weighted
essentially non-oscillatory scheme for hyperbolic conservation laws.
J. Comput. Phys. , 227:3191–3211, 2008.
[6] S. Boscarino, J.-M. Qiu, G. Russo, and T. Xiong. A high order semi-
implicit imex weno scheme for the all-mach isentropic euler system.
J. Comput. Phys. , 392:594–618, 2019.
[7] R. Caiden, R. P. Fedkiw, and C. Anderson. A numerical method for
two-phase ﬂow consisting of separate compressible and incompressible
regions.
J. Comput. Phys. , 166:1–27, 2001.
[8] J.-P. Caltagirone, S. Vincent, and C. Caruyer. A multiphase compress-
ible model for the simulation of multiphase ﬂows.
J. Comput. Phys. ,
50:24–34, 2011.
41


[9] C. Chang, X. Deng, and T. G. Theofanous. Direct numerical simulation
of interfacial instabilities: A consistent, conservative, all-speed, sharp-
interface method.
J. Comput. Phys. , 242:946–990, 2013.
[10] S. Cho and G. Son. Numerical simulation of acoustic droplet vaporiza-
tion near a wall. Int. Com. Heat and Mass Transfer, 99:7–17, 2018.
[11] A. Dalmon,
M. Lepilliez,
S. Tanguy,
R. Alis,
E.-R. Popescu,
R. Roumigui´e, T. Miquel, B. Busset, H. Bavestrello, and J. Mignot.
Comparison between the ﬂuidics experiment and direct numerical sim-
ulations of ﬂuid sloshing in spherical tanks under microgravity condi-
tions. Microgravity Sci and Technol, 31, issue 1, 2019.
[12] A. Dalmon,
M. Lepilliez,
S. Tanguy,
A. Pedrono,
B. Busset,
H. Bavestrello, and J. Mignot. Direct numerical simulation of a bub-
ble motion in a spherical tank under external forces and microgravity
conditions. J. Fluid Mech. , 849, 2018.
[13] K. Davitt, E. Rolley, F. Caupin, A. Arvengas, and S. Balibar. Equation
of state of water under negative pressure. J. Chem. Phys, 133:174507,
2010.
[14] P. Degond and M. Tang. All speed scheme for the low mach number
limit of the isentropic euler equations. Comm. Comput. Phys., 10:1–31,
2011.
[15] J. E. Dendy. Black box multigrid.
J. Comput. Phys. , 48:366–386,
1982.
[16] J. P. V. Doormaal and G. D. Raithby. Enhancements of the simple
method for predicting incompressible ﬂuid ﬂows.
Num. Heat Tran.,
7(2):147–163, 1984.
[17] R. P. Fedkiw, T. Aslam, B. Merriman, and S. Osher. A non-oscillatory
eulerian approach to interfaces in multimaterial ﬂows (the ghost ﬂuid
method).
J. Comput. Phys. , 152:475–492, 1999.
[18] D. Fuster and S. Popinet. An all-mach method for the simulation of
bubble dynamics problems in the presence of surface tension. J. Com-
put. Phys. , 374:752–768, 2018.
[19] M. Herrmann. A sharp interface in-cell-reconstruction method for vol-
ume tracking phase interfaces in compressible ﬂows. In Proceedings of
the Summer Program. CTR, 2016.
42


[20] G. Huber, S. Tanguy, J.-C. B´era, and B. Gilles. A time splitting projec-
tion scheme for compressible two-phase ﬂows. application to the interac-
tion of bubbles with ultrasound waves. J. Comput. Phys. , 302:439–468,
2015.
[21] A. R. Imre. On the existence of negative pressure states. Phys. Stat.
Sol., 244(3):893–899, 2007.
[22] M. Ishii and T. Hibiki.
Thermo-Fluid dynamics of two-phase ﬂow.
Springer-Verlag New York, 2 edition, 2011.
[23] M. Jain, R. S. Prakash, G. Tomar, and R. Ravikrishna.
Secondary
breakup of a drop at moderate weber numbers. Proc. R. Soc. Lond. A
, 471:2014.0930, 2014.
[24] M. Jemison, M. Sussman, and M. Arienti. Compressible, multiphase
semi-implicit method with moment of ﬂuid interface representation.
J. Comput. Phys. , 279:182–217, 2014.
[25] M. Kang, R. P. Fedkiw, and X.-D. Liu. A boundary condition capturing
method for multiphase incompressible ﬂow. J. Sci. Comput. , 15(323-
360), 2000.
[26] M. Kassemi, O. Kartuzova, and S. Hylton.
Validation of two-phase
cfd models for propellant tank self-pressurization: Crossing ﬂuid types,
scales, and gravity levels. Cryogenics, 89:1–15, 2018.
[27] N. Kwatra, J. Su, J. T. Gr´etarsson, and R. Fedkiw. A method for avoid-
ing the acoustic time step restriction in compressible ﬂow. J. Comput.
Phys. , 228:4146–4161, 2009.
[28] B. Lalanne, N. A. Chebel, J. Vejraˇzka, S. Tanguy, O. Masbernat, and
F. Risso.
Non-linear shape oscillations of rising drops and bubbles:
experiments and simulations. Phys of Fluids, 27, 2015.
[29] B. Lalanne, L. R. Villegas, S. Tanguy, and F. Risso. On the computation
of viscous terms for incompressible two-phase ﬂows with level set/ghost
ﬂuid method. J. Comput. Phys. , 301:289–307, 2015.
[30] J. Lee and G. Son. A sharp-interface level-set method for compressible
bubble growth with phase change. Int. Com. Heat and Mass Transfer,
86:1–11, 2017.
43


[31] J. Lekner. Parametric solution of the van der waals liquid-vapor coex-
istence curve.
Am. J. Phys., 50(2):161, 1982.
[32] M. Lepilliez, E. R. Popescu, F. Gibou, and S. Tanguy. On two-phase
ﬂow solvers in irregular domains with contact line.
J. Comput. Phys.
, 321:1217–1251, 2016.
[33] X.-D. Liu, R. P. Fedkiw, and M. Kang. A boundary condition capturing
method for poisson’s equation on irregular domains. J. Comput. Phys.
, 160:151–178, 2000.
[34] S. L. Manzello and J. C. Yang. An experimental study of a water droplet
impinging on a liquid surface. Experiments in Fluids, 32:580–589, 2002.
[35] O. L. Metayer, J. Massoni, and R. Saurel. Elaborating equations of
state of a liquid and its vapor for two-phase ﬂow models.
Int. J. of
Therm. Sci. , 43:265–276, 2004.
[36] A. Orazzo and S. Tanguy. Direct numerical simulations of droplet con-
densation.
Int. J. Heat and Mass Transfer , 129:432–448, 2019.
[37] S. Osher and J. Sethian. Fronts propagating with curvature-dependent
speed: algorithms based on hamilton–jacobi formulations. J. Comput.
Phys. , 79:12–49, 1988.
[38] S. V. Pantakar. Numerical Heat Transfer and Fluid Flow. Hemisphere
Publishing Co., 1980.
[39] S. V. Pantakar and D. B. Spalding. A calculation procedure for heat,
mass and momentum transfer in three-dimensional parabolic ﬂows. Int.
J. Heat and Mass Transfer , 15:1787–1806, 1972.
[40] S. Patkar, M. Aanjaneya, W. Lu, M. Lentine, and R. Fedkiw.
To-
wards positivity preservation for monolithic two-way solid-ﬂuid cou-
pling. J. Comput. Phys. , 312:82–114, May 2016.
[41] B. E. Poling, J. M. Prausnitz, and J. P. O’Connell. The properties of
gases and liquids. McGraw-Hill, 5th edition, 2001.
[42] L. Qiu, W. Lu, and R. Fedkiw. An adaptive discretization of compress-
ible ﬂow using a multitude of moving cartesian grids. J. Comput. Phys.
, 305:75–110, 2016.
44


[43] P. L. Qu´er´e, C. Weisman, H. Paill`ere, J. Vierendeels, E. Dick, R. Becker,
M. Braack, and J. B. B. Locke. Modelling of natural convection ﬂows
with large temperature diﬀerences: A benchmark problem for low mach
number solvers. part 1. reference solutions. ESAIM: Mathematical Mod-
elling and Numerical Analysis, 39(3):609–616, 2005.
[44] R. Rajan and A. Pandit. Correlations to predict droplet size in ultra-
sonic atomisation. Ultrasonics, 39:235–255, 2001.
[45] L. Rueda-Villegas, R. Alis, M. Lepilliez, and S. Tanguy.
A ghost
ﬂuid/level set method for boiling ﬂows and liquid evaporation: ap-
plication to the leidenfrost eﬀect.
J. Comput. Phys. , 316:789–813,
2016.
[46] E. Sousa. The controversial stability analysis. Applied Math Computa-
tion, 145:777–794, 2003.
[47] M. Sussman, P. Smereka, and S. Osher. A level set approach for com-
puting solutions to incompressible two-phase ﬂow. J. Comput. Phys. ,
114:146–159, 1994.
[48] M. Sussman, K. Smith, M. Hussaini, M. Ohta, and R. Zhi-Wei. A sharp
interface method for incompressible two-phase ﬂows. J. Comput. Phys.
, 221:469–505, 2007.
[49] S. Tanguy and A. Berlemont. Application of a level set method for
simulation of droplet collisions.
Int. J. Multiphase Flow , 31:1015–
1035, 2005.
[50] S. Tanguy, T. M´enard, and A. Berlemont.
A level set method for
vaporizing two-phase ﬂows.
J. Comput. Phys. , 221:837–853, 2007.
[51] S. Tanguy, M. Sagan, B. Lalanne, F. Couderc, and C. Colin. Bench-
marks and numerical methods for the simulation of boiling ﬂows.
J. Comput. Phys. , 264:1–22, 2014.
[52] H. Terashima and M. Koshi. Approach for simulating gas-liquid-like
ﬂows under supercritical pressures using a high-order central scheme.
J. Comput. Phys. , 231:6907–6923, 2012.
[53] H. Terashima and G. Tryggvason. A front-tracking/ghost-ﬂuid method
for ﬂuid interfaces in compressible ﬂows. J. Comput. Phys. , 228:4012–
4037, 2009.
45


[54] E. Turkel. Preconditioned methods for solving the incompressible and
low speed compressible equations.
J. Comput. Phys. , 72:277–298,
1987.
[55] A. Urbano, S. Tanguy, and C. Colin. Direct numerical simulation of
nucleate boiling in zero gravity conditions.
Int. J. Heat and Mass
Transfer , 143:118521, 2019.
[56] A. Urbano, S. Tanguy, G. Huber, and C. Colin. Direct numerical simu-
lation of nucleate boiling in micro-layer regime. Int. J. Heat and Mass
Transfer , 123:1128–1137, 2018.
[57] F. Xiao, R. Akoh, and S. Ii.
Uniﬁed formulation for compressible
and incompressible ﬂows by using multi-integrated moments ii: Multi-
dimensional version for compressible and incompressible ﬂows. J. Com-
put. Phys. , 213:31–56, 2006.
46
