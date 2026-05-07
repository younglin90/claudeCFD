ANALYSIS OF AN ASYMPTOTIC PRESERVING LOW MACH NUMBER
ACCURATE IMEX-RK SCHEME FOR THE WAVE EQUATION SYSTEM
K. R. ARUN, A. J. DAS GUPTA, AND S. SAMANTARAY
Abstract. In this paper the analysis of an asymptotic preserving (AP) IMEX-RK ﬁnite volume
scheme for the wave equation system in the zero Mach number limit is presented. The accuracy
of a numerical scheme at low Mach numbers is its ability to maintain the solution close to the
incompressible solution for all times, and this can be formulated in terms of the invariance of a
space of constant densities and divergence-free velocities. An IMEX-RK methodology is employed
to obtain a time semi-discrete scheme, and a space-time fully-discrete scheme is derived by using
standard ﬁnite volume techniques. The existence of a unique numerical solution, its uniform stability
with respect to the Mach number, the AP property, and the accuracy at low Mach numbers are
established for both time semi-discrete, and space-time fully-discrete schemes. Extensive numerical
case studies conﬁrm uniform second order convergence of the scheme with respect to the Mach
number, and all the above-mentioned properties.
1. Introduction
Singular perturbation problems containing small parameters arise in the mathematical modelling
of several problems in science and engineering. A typical example of a singularly perturbed problem
is the wellknown low Mach number ﬂow in ﬂuid dynamics, often encountered in magnetohydrody-
namics, atmospheric, and geophysical ﬂows, weather modelling, combustion theory, and so on. The
Euler equations of motion provide a simple yet optimal mathematical tool to model and simulate
many of the aforementioned physical processes. The scaled, isentropic Euler equations read
(1.1)
∂tρ + ∇· (ρu) = 0,
∂t(ρu) + ∇· (ρu ⊗u) + ∇p
ε2 = 0,
where the independent variables are time t > 0, and space x ∈Rd, d = 1, 2, 3, and the dependent
variables are ρ = ρ(t, x) > 0, the density, and u = u(t, x) ∈Rd, the velocity of the ﬂuid.The pressure
p is given by the equation of state p = P(ρ) := ργ, where γ a constant. Here, ε is the ratio of a
reference ﬂuid velocity to a reference sound velocity, and is known as the reference Mach number.
In low Mach number ﬂows, i.e. when ε ∼0, ε plays the role of a singular perturbation parameter,
and it is wellknown that in the limit ε →0, the solutions of (1.1) approximate their incompressible
counterparts; see [20] for more details. On the other hand, numerical schemes designed for the
system (1.1) suﬀer from a lot of predicaments in the limit ε →0; see, e.g. [21], and the references
therein. From a numerical analysis point of view, the main challenges faced by numerical schemes
in a low Mach number regime are the stiﬀness arising due to stringent CFL restrictions, creation of
spurious waves, dependence of the numerical viscosity of a scheme on the Mach number leading to
Date: November 11, 2021.
2010 Mathematics Subject Classiﬁcation. Primary 35L45, 35L60, 35L65, 35L67, 35L05; Secondary 65M06, 65M08.
Key words and phrases. Compressible Euler system, Incompressible Euler system, The wave equation system, Zero
Mach number limit, IMEX-RK schemes, Asymptotic preserving, Asymptotic accuracy, Finite volume method.
1
arXiv:1909.13103v1  [math.NA]  28 Sep 2019


2
ARUN, DAS GUPTA, AND SAMANTARAY
lack of stability, and the inability to respect the transitional behaviour of the system of equations
in the singular limit.
In our previous work [2], a second order accurate, semi-implicit ﬁnite volume approximation for
the Euler system (1.1) in a low Mach number regime was proposed and implemented. It is shown
that the above mentioned scheme overcomes the severe CFL restrictions, avoids the generation of
spurious waves, and is consistent in the singular limit. This paper is aimed to present a rigorous
analysis of the semi-implicit scheme of [2] by considering the linear wave equation system as a
simpliﬁed model. The scaled, purely hyperbolic, linear wave equation system with advection is
given by
(1.2)
∂tϱ + (u · ∇)ϱ + a
ε∇· u = 0,
∂tu + (u · ∇)u + a
ε∇ϱ = 0.
Here, ϱ denotes a scaled density with ρ(t, x) = ρ

1 + ε
aϱ(t, x)

, and the constants ρ, u, and a are,
respectively, the linearisation states of ρ, u and the sound velocity a.
Expanding all the dependent variables in (1.2) using the ansatz
(1.3)
f(t, x) = f(0)(t, x) + εf(1)(t, x) + ε2f(2)(t, x) + . . . ,
and performing a scale analysis with the use of appropriate boundary conditions yields the following
linearised, mixed hyperbolic-elliptic, incompressible system:
(1.4)
ϱ(0) = const.,
∂tu(0) + (u · ∇)u(0) + a∇ϱ(1) = 0,
∇· u(0) = 0
for the unknowns u(0) and ϱ(1).
It is essential for a discretisation of (1.2) to respect the limiting system (1.4) in the asymptotic
limit ε →0. In addition, it is desirable that its stability restrictions are independent of ε so that the
stiﬀness arising when ε →0 can be overcome. These two requirements fall under the purview of the
so-called asymptotic preserving (AP) methodology [9, 17, 18]. In a related work [10], the authors
study a strongly anisotropic singular elliptic problem arising in plasma physics. In the singular
limit, i.e. when the anisotropy parameter goes to zero, the problem converges to an ill-posed one.
As a cure, the authors propose an AP reformulation of the given anisotropic equation in order to
write the original problem in such a way that a continuous transition towards the limit can be
achieved. Hence, the AP methodology can also help to get rid of the possible ill-posedness which
might arise in the singular limit.
It is well known that being AP is predominantly dictated by the particular time discretisation
chosen. The scheme designed in [2] makes use of the implicit explicit Runge-Kutta (IMEX-RK)
time discretisation and an appropriate ﬂux decomposition to achieve the AP property for the Euler
system (1.1) in the zero Mach number limit; see [3, 5, 8, 11, 22, 23, 24, 27] for more detailed
discussions on IMEX-RK schemes, AP schemes, and their other applications. As mentioned in the
beginning, compressible ﬂow solvers suﬀer from severe loss of accuracy at low Mach numbers due
to the creation of spurious waves. Recently, a detailed analysis was carried out by Dellacherie in
[12] on the behaviour of explicit Godunov-type schemes in a low Mach number regime. The above
mentioned study using the linear wave equation system (1.2) reveals that the inaccuracies can be
avoided by enforcing the particular scheme to preserve a space of constant densities and divergence-
free velocities, known as the well-prepared space. Following this, in [2], a scheme which leaves the
well-prepared space invariant is designated as asymptotically accurate (AA).


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
3
The goal of the current paper is to present a detailed analysis of a time semi-discrete as well as
space-time fully-discrete IMEX-RK scheme for the wave equation system (1.2) under the low Mach
number scaling. The study focusses on addressing the following three issues:
(1) the existence of a unique numerical solution for a ﬁxed ε > 0;
(2) the uniform stability of the numerical solution with respect to ε, and the AP property;
(3) the invariance of the well-prepared space by the numerical solution implying asymptotic
accuracy.
The time semi-discrete scheme corresponds to the dual formulation of an elliptic equation for the
density, and the existence and uniqueness of its solution is obtained via the saddle point theory of
variational problems. The asymptotic consistency, as done, e.g. in [2], then reveals that we do not
encounter pathologies, such as ill-posedness of the limit problem, cf. [10]. The fully-discrete scheme
obtained by simple central diﬀerencing involves circulant matrices [13]. We exploit the theory of
these matrices to establish the above properties for the fully-discrete setup.
In order to carry out the analysis, the linear wave equation system (1.2) is rewritten in the
evolution form
(1.5)
∂tU + H(U) + 1
εL(U) = 0,
via the operators H and L deﬁned as
(1.6)
U :=

ϱ
u

, H(U) :=

u · ∇ϱ
(u · ∇)u

, and L(U) :=

a∇· u
a∇ϱ

.
Here, H is the convective operator with a timescale of order 1 and L/ε is the acoustic operator with
a timescale of the order of ε. As discussed above, when ε →0, the solutions U = (ϱ, u) of (1.6)
converge to U(0) = (ϱ(0), u(0)) in the well-prepared space, which happens to be the kernel of the
operator L. Hence, as done in [12], throughout this paper, we restrict our analysis to the following
initial value problem:
(1.7)
∂tU + 1
εL(U) = 0,
U(0, x) = U0(x), x ∈Td,
where Td denotes the d-dimensional torus to take into account of the periodic boundary conditions.
However, the numerical case studies are performed also on the model (1.5) with advection.
The rest of this paper is organised in the following way. In Section 2 we brieﬂy recall the results
from [12], which are relevant for the present study. Section 3 is devoted to a short presentation of
IMEX-RK time discretisation for stiﬀsystems of ODEs, and the notions of AP and AA properties.
The analysis of the time semi-discrete scheme obtained after employing the IMEX-RK method is
taken up in Section 4, where we prove the desired properties mentioned above in (1)-(3). In Section 5
we present a space-time fully-discrete scheme derived by using a ﬁnite volume technique. The theory
of circulant matrices is used to establish the same properties (1)-(3) for the fully-discrete scheme.
The results of numerical case studies are reported in Section 6, where we numerically corroborate
the theoretical claims. Finally, the paper is concluded with some remarks in Section 7.
2. Analysis of the Wave Equation System
In this section, we brieﬂy recall some of the results from [12], regarding the low Mach number
limit of the wave equation system (1.5). First, we consider the space of solutions of (1.5) which is


4
ARUN, DAS GUPTA, AND SAMANTARAY
the Hilbert space of square integrable functions L2(Td)1+d. The space L2(Td)1+d is equipped with
the innerproduct
(2.1)
(U1, U2) := (ϱ1, ϱ2) + (u1, u2),
where Um = (ϱm, um), m = 1, 2, in the above, and throughout the rest of this paper, (·, ·) denotes
the L2 innerproduct. The kernel of the wave operator L is given by
(2.2)
E :=
n
U ∈L2(Td)1+d : ∇ϱ = 0, and ∇· u = 0
o
,
which is the so-called well-prepared, incompressible, space of constant densities and divergence-free
velocities. The orthogonal complement ˜E of E is deﬁned as
(2.3)
˜E :=

U ∈L2(Td)1+d :
Z
Td ρdx = 0, and ∇× u = 0

.
The spaces E and ˜E, given by (2.2) and (2.3), yields the following Helmholtz-Hodge-Leray decom-
position of L2(Td)1+d:
(2.4)
E ⊕˜E = L2(Td)1+d, and E ⊥˜E.
As a consequence of (2.4), we can decompose any U ∈L2(Td)1+d; there exists a unique ˆU ∈E and
˜U ∈˜E, such that U = ˆU + ˜U. We deﬁne the Helmholtz-Hodge-Leray projection P: L2(Td)1+d →E,
via
(2.5)
PU := ˆU.
Deﬁnition 2.1. The energy E of the system (1.2) is deﬁned as
(2.6)
E := 1
2(U, U).
Proposition 2.2. Let U be a solution of the system (1.2) on Td. Then, the energy E introduced in
Deﬁnition 2.1 is preserved, i.e.
(2.7)
E(t) = E(0), for all t > 0.
Remark 2.3. Proposition 2.2 states that the linear wave equation system (1.2) conserves the energy
(2.1).
Proposition 2.4. Let U be the solution of the system (1.2) with initial data U0. Then,
• for all U0 ∈E, we have U(t, ·) ∈E for all t > 0;
• for all U0 ∈˜E, we have U(t, ·) ∈˜E for all t > 0.
Remark 2.5. Proposition 2.4 states that the wave equation system leaves both the spaces E and ˜E
invariant. In other words, if the solution lives in one of these spaces initially, then it lives there for
all times.
Theorem 2.6 ([12]). Let U be a solution of the IVP:
∂tU + H(U) + 1
εL(U) = 0, t > 0, x ∈Td,
(2.8)
U(0, x) = U0(x), x ∈Td,
(2.9)
and let ¯U be a solution of the IVP:
∂t ¯U + H( ¯U) = 0, t > 0, x ∈Td,
(2.10)
¯U(0, x) = ˆU0(x), x ∈Td,
(2.11)


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
5
where ˆU0 := PU0. Let U = ˆU + ˜U be the Helmholtz-Hodge-Leray decomposition of U. Then, the
following holds.
(i) ˆU = ¯U,
(ii) ˜U is the solution of (2.8) with initial condition ˜U0 := (I −P)U0.
Moreover, there holds the energy conservation:
(2.12)
Ein(t) = Ein(0) and Eac(t) = Eac(0), for all t > 0,
where Ein := ( ˆU, ˆU) and Eac := ( ˜U, ˜U). As a consequence, the following estimate holds:
(2.13)
∥U0 −PU0∥= O(ε) =⇒∥U(t) −PU(t)∥= O(ε) for all t > 0.
Proposition 2.6 lies at the core of the analysis of numerical schemes presented in [12]. Depending
on the order of accuracy, numerical schemes introduce numerical diﬀusion, dispersion or higher
order correction terms in the modiﬁed partial diﬀerential equations (MPDE). However, we desire
that the numerical solutions which satisfy the MPDE also exhibit properties close to those of the
solutions of the continuous system. One of the key properties is to satisfy the estimate (2.13) which
states that a solution remains close to E for all times t > 0 if it is so at time t = 0. The following
proposition guarantees a suﬃcient condition to ensure (2.13), which also accommodates any general
linear discretisation.
Proposition 2.7. [12] Let U be a solution of the IVP:
∂tU + FxU = 0, t > 0, x ∈Td,
(2.14)
U(0, x) = U0(x), x ∈Td,
(2.15)
which is assumed to be well-posed in L∞ [0, ∞); L2(Td)1+d
, with Fx a linear spatial diﬀerential
operator. Then the following conclusions hold.
(i) The solution U satisﬁes the estimate
(2.16)
∥U0 −PU0∥= O(ε) =⇒∥U(t) −¯U(t)∥= O(ε), for all t > 0,
where ¯U is a solution of (2.14) with the initial condition ¯U(0) = PU0. However, we don’t have
the apriori estimate ∥U(t) −PU(t)∥= O(ε) for all t > 0.
(ii) When the operator Fx leaves E invariant, i.e. whenever U0 ∈E implies U(t) ∈E for all t > 0,
then U satisﬁes the estimate (2.16), and in addition we have
(2.17)
∥U0 −PU0∥= O(ε) =⇒∥U(t) −PU(t)∥= O(ε) for all t > 0.
Remark 2.8. The second part of the above Proposition 2.7 give us the importance of E invariance for
any numerical scheme. It states that if a scheme is E-invariant, then if the initial data U0 is almost
in the well-prepared subspace E, the solution at all later times also lives close E. Loosely speaking,
the estimate (2.17) states that if the initial data is almost incompressible, then the solution for all
time t > 0 is also almost incompressible. It was observed in [12] that satisfying the condition (2.17)
avoids the creation of spurious acoustic waves in the numerical solution. Our numerical experiments
reported in Section 6 clearly validate this observation.
3. Asymptotic Preserving and Asymptotically Accurate IMEX-RK Schemes
We devote this section to recall the notions of AP and AA schemes as done in [2]. Further, in a
nutshell, we review the wellknown IMEX-RK schemes for stiﬀsystems of ODEs which are employed
to approximate the time derivatives in system (1.2).


6
ARUN, DAS GUPTA, AND SAMANTARAY
3.1. Asymptotic Preserving Property. One of the essential properties of a numerical approxi-
mation for a singular perturbation problem is its ability to capture the solution of the limit system
as well as the solution of the original problem. The AP methodology not only provides a framework
to address the convergence of the numerical solution to that of the limit system but also takes care
of the stability restrictions such that they don’t deteriorate in the singular limit.
Deﬁnition 3.1. Let Pε denote a singularly perturbed problem with the perturbation parameter
ε. Let P0 denote the limiting system of Pε for ε →0. A discretisation Pε
h of Pε, with h being the
discretisation parameter, is called AP, if
(i) P0
h is a consistent discretisation of the problem P0, called the asymptotic consistency, and
(ii) the stability constraints on h are independent of ε, called the asymptotic stability.
In other words, the following diagram commutes:
Pε
h
h→0
−−−−→Pε
yε→0
yε→0
P0
h
h→0
−−−−→P0
3.2. Asymptotic Accuracy. We note from Proposition 2.4 that a solution of the wave equation
system lives in E at all times if the initial data is taken from E. It was shown in [12] that a suﬃcient
condition for a numerical scheme for the wave equation system to be accurate at low Mach numbers,
i.e. it is free from the creation of spurious waves, is the E-invariance. Based on this idea, in [2], the
notion of asymptotic accuracy is deﬁned as the following.
Deﬁnition 3.2. A numerical approximation for the wave equation system (1.2) is said to be asymp-
totically accurate (AA), if it leaves the incompressible subspace E invariant.
3.3. IMEX-RK Time Discretisation. IMEX-RK schemes provide a robust and eﬃcient frame-
work to design AP schemes for singular perturbation problems. In this work, we only consider a sub-
class of the IMEX-RK schemes, namely diagonally implicit or (DIRK) schemes. An s-stage IMEX-
RK scheme is characterised by the two s × s lower triangular matrices ˜A = (˜ai,j), and A = (ai,j),
the coeﬃcients ˜c = (˜c1, ˜c2, . . . , ˜cs) and c = (c1, c2, . . . , cs), and the weights ˜ω = (˜ω1, ˜ω2, . . . , ˜ωs) and
ω = (ω1, ω2, . . . , ωs). Here, the entries of ˜A and A satisfy the conditions ˜ai,j = 0 for j ≥i, and
ai,j = 0 for j > i. Let us consider the following stiﬀsystem of ODEs in an additive form:
(3.1)
y′ = f(t, y) + 1
εg(t, y),
where 0 < ε ≪1 is called the stiﬀness parameter. The functions f and g are known as, respectively,
the non-stiﬀpart and the stiﬀpart of the system (3.1); see, e.g. [14], for a comprehensive treatment
of such systems.
Let yn be a numerical solution of (3.1) at time tn and let ∆t denote a ﬁxed timestep. An s-stage
IMEX-RK scheme, cf., e.g. [3, 23], updates yn to yn+1 through s intermediate stages:
Yi = yn + ∆t
i−1
X
j=1
˜ai,jf(tn + ˜cj∆t, Yj) + ∆t
s
X
j=1
ai,j
1
εg(tn + cj∆t, Yj), 1 ≤i ≤s,
(3.2)
yn+1 = yn + ∆t
s
X
i=1
˜ωif(tn + ˜ci∆t, Yi) + ∆t
s
X
i=1
ωi
1
εg(tn + ci∆t, Yi).
(3.3)
In order to further simplify the analysis of the schemes presented in this paper, we restrict
ourselves only to two types of DIRK schemes, namely the type-A and type-CK schemes which are
deﬁned below; see [19] for details.


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
7
Deﬁnition 3.3. An IMEX-RK scheme is said to be of
• type-A, if the matrix A is invertible;
• type-CK, if the matrix A ∈Rs×s, s ≥2, can be written as
A =

0
0
α
As−1

,
where α ∈Rs−1 and As−1 ∈Rs−1×s−1 is invertible.
The results presented in later sections, are obtained using both the ﬁrst order Euler(1,1,1), and
second order ARS(2,2,2) schemes for time discretisations; see [23, 24] for their deﬁnitions. Here, in
the triplet (s, σ, p), s is the number of stages of the implicit part, the number σ gives the number
of stages for the explicit part and p gives the overall order of the scheme. We refer the interested
reader to [14, 19, 23, 24] and the references therein for a detailed study of IMEX-RK schemes.
Hypothesis 3.4. We suppose that the IMEX-RK scheme under consideration is of Type-A or Type
CK.
4. Time Semi-discrete Scheme and Its Analysis
In this section we present our time semi-discrete scheme for the wave equation system (1.2)
obtained after approximation of the time derivatives using the IMEX-RK methodology described
in Section 3. We carry out a detailed analysis of the scheme, and show some if its key properties,
namely its solvability, i.e. the existence of a numerical solution for any ﬁxed ε > 0, the AP property
and the asymptotic accuracy.
4.1. Time Semi-discrete Scheme. As a ﬁrst step in deﬁning a time semi-discrete scheme, we
split the ﬂuxes in (1.2) into a stiﬀand a non-stiﬀpart, yielding
(4.1)
G(U) := a
ε

u
ϱ

, F(U) :=

ϱu
u ⊗u

.
Let 0 < t1 < t2 < · · · < tn < · · · be an increasing sequence of times. In the following, fn denotes
an approximation to the value of a function f(t, x) at time tn, i.e. fn(x) ∼f(tn, x). Treating F
explicitly, and G implicitly, the IMEX-RK time semi-discrete scheme can be obtained as follows.
Deﬁnition 4.1. Given an approximation (ϱn, un) of the numerical solution at time tn, and a
timestep ∆t, the kth stage of an s-stage IMEX-RK scheme for the wave equation system (1.2) is
deﬁned by
ϱk = ϱn −∆t˜ak,ℓ(u · ∇)ϱℓ−∆tak,l
a
ε∇· ul, for each k = 1, 2, . . . , s,
(4.2)
uk = un −∆t˜ak,ℓ(u · ∇)uℓ−∆tak,l
a
ε∇ϱl, for each k = 1, 2, . . . , s.
(4.3)
The approximate numerical solutions ϱn+1 and un+1 at time tn+1 are deﬁned as
ϱn+1 = ϱn −∆t˜ωk(u · ∇)ϱk −∆tωk
a
ε∇· uk,
(4.4)
un+1 = un −∆t˜ωk(u · ∇)uk −∆tωk
a
ε∇ϱk.
(4.5)
In the above, and throughout the rest of this paper, we follow the convention that a repeated
index always denotes the summation with respect to that index. Here, the index k assumes values in
{1, 2, . . . , s}, and the indices ℓand l are used to denote, respectively, the summation in the explicit
and implicit terms, i.e. they assume values in the sets {1, 2, . . . , k −1} and {1, 2, . . . , k}.


8
ARUN, DAS GUPTA, AND SAMANTARAY
4.2. Solvability of the Time Semi-discrete Scheme. The aim of this subsection is to establish
the existence of a numerical solution to (4.2)-(4.5) using variational formulations, and classical
saddle point theory; see, e.g. [6, 7] for more details.
To this end, let us consider the standard
function spaces
(4.6)
V := H(div; Td), and M := L2(Td).
In (4.2)-(4.5), we multiply the density updates by a test function λ ∈M and the velocity updates
by a test function v ∈V , and integrate by parts to get the following weak formulation.
For k = 1, 2, . . . , s, ﬁnd (uk, ϱk) ∈V × M satisfying
(uk, v) −∆tak,k
a
ε(∇· v, ϱk) = (un, v) + ∆tak,ℓ
a
ε(∇· v, ϱℓ), for all v ∈V,
(4.7)
−∆tak,k
a
ε(∇· uk, λ) −(ϱk, λ) = −(ϱn, λ) + ∆tak,ℓ
a
ε(∇· uℓ, λ), for all λ ∈M.
(4.8)
Finally, ﬁnd (un+1, ϱn+1) ∈V × M satisfying
(un+1, v) = (un, v) + ∆tωk
a
ε(∇· v, ϱk), for all v ∈V,
(4.9)
(ϱn+1, λ) = (ϱn, λ) −∆tωk
a
ε(∇· uk, λ), for all λ ∈M.
(4.10)
Note that the semi-discrete scheme (4.2)-(4.5) admits a solution if, and only if, the weak formulations
(4.7)-(4.10) admit a solution.
Theorem 4.2. Suppose an approximation (un, ϱn) ∈V × M of the numerical solution at time tn,
and a timestep ∆t are chosen. Then, under the Hypothesis 3.4, the weak formulations (4.7)-(4.8)
are uniquely solvable for a (uk, ϱk) ∈V × M for k = 1, 2, . . . , s. Consequently, (4.9)-(4.10) deﬁnes
uniquely an approximate numerical solution (un+1, ϱn+1) ∈V × M at time tn+1 = tn + ∆t.
The weak formulations (4.7)-(4.8) can be recast in the saddle point form: ﬁnd (uk, ϱk) ∈V × M
satisfying
(4.11)
ak(uk, v) + bk(v, ϱk) = lk(v), for all v ∈V,
bk(uk, λ) −ck(ϱk, λ) = χk(λ), for all λ ∈M.
Here, the bilinear forms ak : V × V →R, bk : V × M →R, and ck : M × M →R are deﬁned as
ak(v, w) := (v, w), for each v, w ∈V,
(4.12)
bk(v, λ) := −∆tak,k
a
ε(∇· v, λ), for each v ∈V, λ ∈M,
(4.13)
ck(λ, µ) := (λ, µ), for each λ, µ ∈M,
(4.14)
and the linear forms l: V →R, and χ: M →R are deﬁned as
(4.15)
lk(v) := (un, v) + ∆tak,ℓ
a
ε(∇· v, ϱℓ), for each v ∈V,
χk(λ) := −(ρn, λ) + ∆tak,ℓ
a
ε(∇· uℓ, λ), for each λ ∈M.
Once the existence of (uk, ϱk) ∈V × M is established for k = 1, 2, . . . , s, (4.9)-(4.10) then uniquely
deﬁnes the approximate numerical solution at tn+1. Note that the saddle point problem in (4.7)-
(4.8) is not in the standard form. We make use of the following result from [6] to establish the
existence and uniqueness of (4.11).


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
9
Theorem 4.3 (Babuˇska-Brezzi Inf-sup Theorem). Let V and M be two Hilbert spaces, and let
a: V ×V →R, b: V ×M →R, and c: M ×M be three continuous bilinear forms with the following
properties.
a is positive semi-deﬁnite, i.e. a(v, v) ≥0 for all v ∈V , and there exists a constant α > 0 such
that
(4.16)
inf
v0∈V0 sup
w0∈V0
a(v0, w0)
∥v0∥V ∥w0∥V
≥α, and
inf
w0∈V0 sup
v0∈V0
a(v0, w0)
∥v0∥V ∥w0∥V
≥α,
where V0 := {v0 ∈V : b(v0, λ) = 0 for all λ ∈M}.
There exists a constant β > 0 such that
(4.17)
sup
v∈V
|b(v, λ)|
∥v∥V
≥β
inf
λ0∈M0∥λ0 + λ∥M, for all λ ∈M,
where M0 := {λ0 ∈M : b(v, λ0) = 0 for all v ∈V }.
c is positive semi-deﬁnite and symmetric, i.e. c(λ, λ) ≥0 for all λ ∈M, and c(λ, µ) = c(µ, λ) for
all λ, µ ∈M. Further, there exists a constant γ > 0, such that for every λ ∈M⊥
0 , and for every
ϵ > 0, the solution λ0 ∈M0 of the equation
(4.18)
ϵ(λ0, µ)M + c(λ0, µ) = −c(λ, µ), for all µ ∈M,
is bounded by
(4.19)
γ∥λ0∥M ≤∥λ∥M,
where (·, ·)M and M⊥
0 are, respectively, the innerproduct in M, and the orthogonal complement of
M0.
Finally, let l: V →R and χ: M →R be two continuous linear forms. Then the variational
problem: ﬁnd (u, ϱ) ∈V × M, such that
(4.20)
a(u, v) + b(v, ϱ) = l(v), for all v ∈V,
b(u, λ) −c(ϱ, λ) = χ(λ), for all λ ∈M
has one and only one solution.
Proof of Theorem 4.2. Clearly, the bilinear forms ak, bk and ck are continuous on their respective
domains, and ak and ck are symmetric. The structural condition (4.19) is trivially satisﬁed for the
bilinear form c deﬁned in (4.14); see [6]. Hence, we are left with verifying only the conditions (4.16)
and (4.17).
From the deﬁnition of bk it follows easily that if v0 ∈V , then ∇· v0 = 0, and if λ0 ∈M0, then
∇λ0 = 0. Hence, for v0 ∈V0, we have ∥v0∥H(div;Td) = ∥v0∥L2(Td)d. To prove (4.16), let v0 ∈V0.
Now,
(4.21)
sup
w0∈V0
ak(v0, w0)
∥w0∥V
≥ak(v0, v0)
∥v0∥V
= (v0, v0)
∥v0∥V
= ∥v0∥V .
Hence,
(4.22)
inf
v0∈V0 sup
w0∈V0
ak(v0, w0)
∥w0∥V ∥v0∥V
≥1.
Next, we proceed to establish the condition (4.17). Let λ0 ∈M0, and λ ∈M. Corresponding to
˜λ := λ0 + λ ∈L2(Td), there exists a unique ˜w ∈H1
0(Td) satisfying
(4.23)
(∇˜w, ∇µ) = (˜λ, µ) for all µ ∈H1
0(Td).


10
ARUN, DAS GUPTA, AND SAMANTARAY
Note that ∇˜w ∈L2(Td)d and ˜w satisﬁes the elliptic problem −∆˜w = ˜λ in the sense of distributions.
Hence, ∇· ∇˜w = −˜λ ∈L2(Td). In other words, ∇˜w ∈H(div; Td). Therefore, setting µ = ˜w in
(4.23) yields
∥∇˜w∥2
L2(Td)d = (˜λ, ˜w)
≤∥˜λ∥L2(Td)∥˜w∥L2(Td)
≤∥˜λ∥L2(Td)∥˜w∥H1(Td)
≤C∥˜λ∥L2(Td)| ˜w|H1(Td)
= C∥˜λ∥L2(Td)∥∇˜w∥L2(Td)d.
(4.24)
Here, we have used the Poincar´e inequality in the last but one step. Therefore, we have
(4.25)
∥∇˜w∥L2(Td)d ≤C∥˜λ∥L2(Td).
Further, using −∆˜w = ˜λ, we get ∥∆˜w∥L2(Td) = ∥˜λ∥L2(Td). Combining the above two we obtain
∥∇˜w∥2
H(div;Td) = ∥∇˜w∥2
L2(Td)d + ∥∆˜w∥2
L2(Td)
≤(C2 + 1)∥˜λ∥2.
(4.26)
Since λ0 ∈M0, we must have b(v, λ0) = 0 for all v ∈V . Thus, for 0 ̸= ˜λ ∈M,
sup
v∈V
|bk(v, λ)|
∥v∥V
= sup
v∈V
|bk(v, ˜λ)|
∥v∥V
≥|bk(∇˜w, ˜λ)|
∥∇˜w∥V
= ∆t|ak,k|a
ε
|(∆˜w, ˜λ)|
∥∇˜w∥V
≥∆t|ak,k|a
ε
∥˜λ∥2
M
p
(C2 + 1)∥˜λ∥M
=
∆t|ak,k|
p
(C2 + 1)
a
ε∥˜λ∥M
=
∆t|ak,k|
p
(C2 + 1)
a
ε∥λ0 + λ∥M.
(4.27)
The inf-sup condition (4.17) now follows from (4.27) by taking the inﬁmum over λ0 ∈M0.
Hence, it follows from Theorem 4.3 that the kth stage (4.7)-(4.8) of the weak formulation admits
a unique solution (uk, ϱk) ∈V × M.
□
4.3. Asymptotic Preserving Property. The goal of this section is to prove the AP property
of the scheme (4.7)-(4.10). As mentioned before, proving the AP property consists of proving the
asymptotic stability and asymptotic consistency.
Theorem 4.4. Consider the semi-discrete scheme (4.7)-(4.10), and assume the conditions of The-
orem 4.2.
(1) Then there exists a constant Ck > 0, such that the numerical solution (uk, ϱk) ∈V × M of
the kth stage (4.7)-(4.8) satisﬁes the energy stability estimate:
(4.28)
Ek ≤CkEn,


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
11
where Ck depends only on the IMEX-RK coeﬃcients, but is independent of ε. Consequently,
there exists a constant C > 0, such that the numerical solution (un+1, ϱn+1) satisﬁes the
estimate:
(4.29)
En+1 ≤CEn,
where C depends only on the matrix A and the vector ω, but is independent of ε. In other
words, the time semi-discrete scheme (4.7)-(4.10) is stable in the L2-norm.
(2) If we assume that the solution (ϱn, un) at time tn is well-prepared i.e. it admits the decom-
position:
(4.30)
ϱn = ϱn
(0) + εϱn
(1), un = un
(0) + εun
(1),
where (ϱn
(0), un
(0)) ∈E, then, the numerical solution (ϱn+1, un+1) also admits a similar de-
composition
(4.31)
ϱn+1 = ϱn+1
(0) + εϱn+1
(1) , un+1 = un+1
(0) + εun+1
(1) ,
with (ϱn+1
(0) , un+1
(0) ) ∈E, which shows consistency with the asymptotic limit as ε →0.
Hence, the scheme (4.7)-(4.10) is asymptotic preserving.
Proof. We prove only the statement in (1), and the statement (2) follows as in [2]. In order to prove
(1), we proceed as follows. Considering the ﬁrst stage, i.e. for k = 1, we have the fully implicit
update:
(u1, v) −∆ta1,1
a
ε(ϱ1, ∇· v) = (un, v), for all v ∈V,
(4.32)
−∆ta1,1
a
ε(∇· u1, λ) −(ϱ1, λ) = −(ϱn, λ), for all λ ∈M.
(4.33)
In the above, taking v = u1 and λ = −ϱ1, adding the resulting equations gives
(4.34)
(u1, u1) + (ϱ1, ϱ1) = (un, u1) + (ϱn, ϱ1).
A successive application of the Cauchy-Schwarz inequality on the right hand side, and rearranging
the terms yields
(4.35)
E1 ≤En.
In order to get the estimate for the second stage, i.e. for k = 2, let us consider
(u2, v) −∆ta2,2
a
ε(ϱ2, ∇· v) = (un, v) + ∆ta2,1
a
ε(ϱ1, ∇· v), for all v ∈V,
(4.36)
−∆ta2,2
a
ε(∇· u2, λ) −(ϱ2, λ) = −(ϱn, λ) + ∆ta2,1
a
ε(∇· u1, λ), for all λ ∈M.
(4.37)
In (4.32)-(4.33) we set v = −a21
a11 u2, λ = a21
a11 ϱ2, in (4.36)-(4.37) we set v = u2, λ = −ϱ2, and add all
the resulting equations to get
(4.38)
(u2, u2) + (ϱ2, ϱ2) =

1 −a21
a11
 
(un, u2) + (ϱn, ϱ2)
	
+ a21
a11

(u1, u2) + (ϱ1, ϱ2)
	
.
Proceeding similarly as in the case of k = 1, we can obtain from (4.38)
(4.39)
E2 ≤C2En.
Note that the above procedure is similar to the usual forward elimination process: in the kth
stage, we let v = uk, λ = −ϱk, and eliminate the terms containing (ϱℓ, ∇· v) and (∇· uℓ, λ) for


12
ARUN, DAS GUPTA, AND SAMANTARAY
ℓ= 1, 2, . . . , k −1 by choosing the test functions v and λ appropriately in each of the k −1 previous
stages. The elimination process is valid under the Hypothesis 3.4. Hence, we have for k = 1, 2, . . . , s
(4.40)
Ek ≤CkEn
for an appropriate constant Ck > 0 which depends only on the coeﬃcients of the matrix A.
An analogous procedure using the update formulae (4.9)-(4.10) ﬁnally yields the stability esti-
mate:
(4.41)
En+1 ≤CEn,
where C > 0 depends only on A and ω, and is independent of ε.
□
4.4. Asymptotic Accuracy. The asymptotic accuracy follows under the suﬃcient condition of
E-invariance of the scheme. Since the proof follows similar lines as that [2], we omit the details here.
Theorem 4.5. The semi-discrete scheme (4.7)-(4.10) leaves the well-prepared space E invariant,
i.e. if the data (ϱn, un) at time tn is in E, then (ϱn+1, un+1) ∈E. As a consequence, the semi-discrete
scheme is asymptotically accurate.
5. Analysis of Space-Time Fully-discrete Scheme
In this section we present a space-time fully-discrete scheme obtained by a ﬁnite volume strategy,
and its analysis. Let the given cartesian spatial domain Ωbe discretised into rectangular cells of
length ∆x1 and ∆x2 in x1 and x2 directions, respectively. For notational conveniences, we deﬁne
the spatial diﬀerential operators µ and δ, e.g.
(5.1)
δx1ωi,j := ωi+ 1
2 ,j −ωi−1
2 ,j,
µx1ωi,j :=
ωi+ 1
2 ,j + ωi−1
2 ,j
2
,
in the x1-direction, with analogous deﬁnitions in the x2-direction.
In order to achieve second order accuracy in space, we follow a MUSCL strategy.
From the
piecewise constant cell averages Un
i,j of the unknown function U at time tn, we reconstruct a piece-
wise linear interpolant. In order to carry out the analysis of the fully-discrete scheme as done in
Section 4, we only consider smooth solutions, and hence, the discrete slopes in the linear recovery
are approximated using central diﬀerences without using any limiters.
5.1. Space-time Fully-discrete Scheme. Applying a ﬁnite volume discretisation for the ﬂuxes
F(U) and G(U), we obtain the following fully-discrete scheme corresponding to (4.2)-(4.5).
Deﬁnition 5.1. The kth stage of an s-stage space-time fully-discrete IMEX-RK scheme for the
wave equation system (1.2) is deﬁned as
(5.2)
Uk
i,j = Un
i,j −˜ak,ℓλmδxmFm(Uℓ)i,j −ak,lλmδxmGm(Ul)i,j, for each k = 1, 2, . . . , s,
and the ﬁnal update is given by
(5.3)
Un+1
i,j
= Un
i,j −˜ωkλmδxmFm(Uk)i,j −ωkλmδxmGm(Uk)i,j.
Here, the repeated index m takes values in 1, 2, and λm :=
∆t
∆xm denote the mesh ratios.
In our computations, we use a simple Rusanov-type ﬂux to approximate the explicit part F, and
a second-order central ﬂux for the implicit part G, e.g. in the x1-direction
(5.4)
F1,i+ 1
2 ,j(Uℓ) = 1
2

F1

Uℓ,+
i+ 1
2 ,j

+ F1

Uℓ,−
i+ 1
2 ,j

−u1
2

Uℓ,+
i+ 1
2 ,j −Uℓ,−
i+ 1
2 ,j

,
G1,i+ 1
2 ,j(Uℓ) = 1
2

G1

Uℓ
i+1,j

+ G1

Uℓ
i,j

.


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
13
Here U±
i+ 1
2 ,j denotes the right and left interpolated states at a right hand vertical edge.
Finally, to maintain the stability, the timestep ∆t is computed using the CFL condition
(5.5)
∆t max
 u1
∆x1
, u2
∆x2

= ν,
where ν < 1 is the given CFL number. Note that the above condition is the advective CFL condition,
and is independent of ε.
5.2. Solvability of the Space-time fully discrete scheme. The aim of this subsection is to
establish the existence of a unique solution to the fully-discrete scheme introduced in Deﬁnition 5.1;
cf. also Theorem 4.2. To this end, we use the theory of circulant matrices; see [13] for more details.
In order to make the exposition simple, we consider a one-dimensional scheme; extension to two
dimensions is straightforward.
Theorem 5.2. Suppose a discrete numerical approximation (ϱn
i , un
i ) at time tn, and a timestep
∆t are given. Then, under Hypothesis 3.4, each of the intermediate stages (5.2) admit a unique
solution (ϱk
i , uk
i ) for k = 1, 2, . . . , s. As a consequence, the update step (5.3) admits a unique solution
(ϱn+1
i
, un+1
i
) at time tn+1 = tn + ∆t.
Proof. The proof uses induction on k, the number of stages. Note that for any k = 1, 2, . . . , s, we
can rewrite the time semi-discrete scheme (4.2)-(4.3) as
(5.6)
ϱk = ˆϱk −∆ta
εak,k∂x1uk,
uk = ˆuk −∆ta
εak,k∂x1ϱk,
where we have denoted the explicit terms
(5.7)
ˆϱk = ϱn −∆ta
εak,ℓ∂x1uℓ,
ˆuk = un −∆ta
εak,ℓ∂x1ϱℓ.
In (5.6), when k = 1, we have the fully implicit ﬁrst stage
(5.8)
ϱ1
i = ϱn
i −∆ta
εa1,1
(u1
i+1 −u1
i−1)
2∆x1
,
u1
i = un
i −∆ta
εa1,1
(ϱ1
i+1 −ϱ1
i−1)
2∆x1
,
for all i = 1, 2, . . . , N, where N denotes the number of mesh points. Let us denote
(5.9)
Zk = (ϱk
1, ϱk
2, · · · , ϱk
N−1, ϱk
N),
for each k = 1, 2, . . . , s,
V k = (uk
1, uk
2, · · · , uk
N−1, uk
N),
for each k = 1, 2, . . . , s.
Therefore, (5.8) can be written as
(5.10)

Z1
V 1

=

Zn
V n

−β1

O
P
P
O
 
Z1
V 1

.
where β1 :=
∆t
2∆x
a
εa1,1, P := circ(0, 1, · · · , −1)N is an N × N circulant matrix [13], and O is the
N × N zero matrix. The equation (5.10) then gives the linear system
(5.11)
A1(ε)

ϱ1
u1

=

ˆϱ1
ˆu1

,


14
ARUN, DAS GUPTA, AND SAMANTARAY
where the block-matrix A1(ε) is given by
(5.12)
A1(ε) :=

1
β1P
β1P
1

,
with 1 being the N ×N identity matrix. Since 1 and P commute, the determinant of A1(ε) is given
by, see [4],
(5.13)
det(A1(ε)) = det
 1 −β2
1P 2
.
As a consequence of the Greshgorin’s circle theorem [15], it can be seen that the numerical range of
−β2
1P 2 is nonnegative and that of 1 is positive; see [16]. In fact, both these matrices are symmetric,
and they both have strictly positive eigenvalues. Due to the sub-additivity of the numerical range,
the eigenvalues of the matrix on the right hand side of (5.13) are then nonzero. Hence, A1(ε) is
invertible, which in turn conﬁrms the existence and uniqueness of (Z1, V 1).
Now, for each k = 2, · · · , s, we have
(5.14)

Zk
V k

=
 ˆZk
ˆV k

−βk

O
P
P
O
 
Zk
V k

,
where βk :=
∆t
2∆x
a
εak,k for k = 2, . . . , s. Note that ( ˆZk, ˆV k) can be written in terms of (Zℓ, V ℓ). As
in the case of k = 1, we can now construct a block matrix Ak(ε) with β1 replaced by βk in (5.12)
which can be shown to be invertible. Hence, by induction, we prove the existence and uniqueness of
the solution (Zk, V k). As a consequence, the existence and uniqueness of (Zn+1, V n+1) follows.
□
5.3. Asymptotic Preserving Property. We prove the AP property of the fully-discrete scheme
by showing l2-stability uniformly with respect to ε, and its consistency in the limit ε →0.
Theorem 5.3. Consider the fully-discrete scheme (5.2)-(5.4), and assume the conditions of Theo-
rem 5.2
(1) Then, there exists a constant Ck > 0 such that the numerical solution (uk
i , ϱk
i , ) of the kth
stage (5.2) satisﬁes the energy stability estimate:
(5.15)
Ek ≤CkEn,
where the constant Ck > 0 is independent of ε and depends only on the IMEX-RK co-
eﬃcients. Consequently, there exists a constant C > 0 such that the numerical solution
(un+1
i
, ϱn+1
i
, ) satisﬁes the estimate
(5.16)
En+1 ≤CEn,
where C > 0 is independent of ε, and depends only on the matrix A and the vector ω. In
other words, the time fully-discrete scheme (5.2)-(5.4) is stable in the l2-norm.
(2) If we assume that the solution (ϱn
i , un
i ) at time tn is well-prepared i.e. it admits the decom-
position:
(5.17)
ϱn
i = ϱn
(0),i + εϱn
(1),i, un
i = un
(0),i + εun
(1),i,
for all i = 1, . . . , N,
where δx1µx1
∆x1 ϱn
(0),i = 0 and δx1µx1
∆x1 un
(0),i = 0, or in other words (ϱn
(0),i, un
(0),i) lives in E. Here,
δx1µx1
∆x1
is the discrete derivative introduced by the implicit terms, i.e. by replacing the deriva-
tives by central diﬀerences. Then then, the numerical solution (ϱn+1
i
, un+1
i
) also admits the
same decomposition
(5.18)
ϱn+1
i
= ϱn+1
(0),i + εϱn+1
(1),i, un+1
i
= un+1
(0),i + εun+1
(1),i,
for all i = 1, . . . , N,


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
15
i.e. the numerical solution is well-prepared and (ϱn+1
(0),i, un+1
(0),i) lives in E, which shows consis-
tency with the asymptotic limit as ε →0.
Hence, the scheme (5.2)-(5.4) is asymptotic preserving.
Proof. We prove only the statement in (1), and the statement (2) follows as in [2]. The proof of (1)
uses induction on k. For k = 1 we have from (5.10)
(5.19)

Z1
V 1

= (A1(ε))−1

Zn
V n

,
where A1(ε) is given by (5.12). It has to be noted that any circulant matrix M can be diagonalised
as, see [13],
(5.20)
ΛM := F ∗
NMFN,
where ∗denotes the conjugate transpose, and the matrix FN is a unique unitary matrix consisting
of eigenvectors of a circulant matrix of size N. Hence, FN is independent of the entries of M, and
it is completely determined by the size N of the matrix. The diagonalisation of the matrix A1(ε) is
given by
(5.21)
Λ1(ε) = diag(FN, FN)A1(ε)diag(F ∗
N, F ∗
N)
=

1
β1ΛP
β1ΛP
1

,
where ΛP is the diagonal matrix consisting of the eigenvalues of the matrix P, cf. also proof of
Theorem 5.2. For any matrix norm, ∥(A1(ε))−1∥satisﬁes
(5.22)
∥(A1(ε))−1∥≤∥diag(FN, FN)∥∥diag(F ∗
N, F ∗
N)∥∥(Λ1(ε))−1∥.
The above inequality (5.22) implies that the dependence of the norm ∥(A1(ε))−1∥on ε is only
through ∥(Λ(ε))−1∥. By Proposition 2.8.7 in [4], the inverse of Λ1(ε) is given by
(5.23)
(Λ1(ε))−1 =

(1 −β2
1Λ2
P )−1
−β1ΛP (1 −β2Λ2
P )−1
β1ΛP (1 −β2
1Λ2
P )−1
(1 −β2
1Λ2
P )−1

.
It can be seen that each block in the above matrix is bounded uniformly with respect to ε, and
hence ∥(Λ1(ε))−1∥also. Therefore,
(5.24)
∥(A1(ε))−1∥≤C1,
where C1 > 0 is a constant independent of ε. As a result, from (5.19), we have the estimate
(5.25)
E1 ≤C1En,
where E := ∥(Z, V )∥l2 is the energy of the fully-discrete solution. For k = 2, the solution (Z2, V 2)
is given by
(5.26)

Z2
V 2

=

Zn
V n

−∆ta
ε
a2,1
2∆xD

Z1
V 1

−∆ta
ε
a2,2
2∆xD

Z2
V 2

,
where D is the 2N × 2N central diﬀerence discretisation matrix, cf. (5.10). From (5.26), we have
(5.27)
A2(ε)

Z2
V 2

=

Zn
V n

−M(ε)

Z1
V 1

.


16
ARUN, DAS GUPTA, AND SAMANTARAY
In (5.27), the matrix M(ε) = −∆t a
ε
a2,1
2∆xD. Now using (5.19) in (5.27) yields
(5.28)

Z2
V 2

= (A2(ε))−1

Zn
V n

−M(ε)(A1(ε))−1

Zn
V n

,
=
 (A2(ε))−1 −(A2(ε))−1M(ε)(A1(ε))−1 
Zn
V n

.
As done in the case of k = 1, it can be shown that the matrix ((A2(ε))−1−(A2(ε))−1M(ε)(A1(ε))−1)
is uniformly bounded with respect to ε. Hence, we have for any matrix norm, there exist a constant
C2, independent of ε, such that
(5.29)
∥
 (A2(ε))−1 −(A2(ε))−1M(ε)2(A1(ε))−1
∥≤C2,
which leads to the stability estimate
(5.30)
E2 ≤C2En.
In this fashion, we can show that for each k = 1, 2, . . . , s, there exist a constants Ck > 0, independent
of ε, such that
(5.31)
Ek ≤CkEn, for all k = 1, 2, . . . , s.
Substituting the expressions for (Zk, V k) in terms of (Zn, V n) in the update stage for (Zn+1, V n+1),
and estimating the the l2 norm ﬁnally yields the stability bound
(5.32)
En+1 ≤CEn,
where the constant C is independent of ε.
□
Remark 5.4. It has to be noted that the above stability analysis presented in Theorem 5.3 does
not require any condition on ∆t and ∆x. This is not surprising as we are dealing with a fully
implicit scheme. Carrying out a similar analysis including the advection terms will enforce a CFL-
like condition independent of ε. In [1, 2], we have presented the results of an analogous study for
a ﬁrst order accurate IMEX-RK scheme for the wave equation system with advection using the
modiﬁed equation analysis; see also [26, 27] for related studies on the shallow water model.
5.4. Asymptotic Accuracy. As in the semi-discrete case, the asymptotic accuracy is a conse-
quence of the E-invariance.
Theorem 5.5. Suppose that at time tn the numerical solution (ϱn
i , un
i ) is in E, i.e. ϱn
i = const. and
δx1µx1
∆x1 un
i = 0 for all i. Then, at time tn+1, the numerical approximation (ϱn+1
i
, un+1
i
) obtained from
the scheme (5.2)-(5.4) satisfy
(5.33)
ϱn+1
i
= const., δx1µx1
∆x1
un+1
i
= 0, for all i.
In other words, the fully-discrete scheme (5.2)-(5.4) keeps the well-prepared space E invariant.
6. Numerical Results and Their Analysis
This section is aimed at presenting the results of numerical computations performed using the
proposed scheme. A detailed analysis of the numerical results is carried out to support and validate
the theoretical ﬁndings. The analysis focuses to numerically corroborate the following four key
properties of the proposed scheme:
(i) uniform second-order convergence with respect to ε;
(ii) uniform stability with respect to ε;
(iii) asymptotic consistency;


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
17
(iv) invariance of the well-prepared space E, yielding asymptotic accuracy.
We consider four diﬀerent test cases to establish each of the above mentioned qualities of the
proposed IMEX-RK ﬁnite volume scheme. First, we consider a smooth data to demonstrate the
uniform second-order convergence by computing the experimental order of convergence (EOC) for
diﬀerent values of ε. Second, computations are carried out using a two-dimensional moving vortex in
order to testify that the energy dissipation of the scheme is independent of ε thereby establishing the
uniform stability, numerically. The third test-case is a two-dimensional well-prepared data, aimed at
demonstrating the asymptotic consistency. Lastly, we consider a two-dimensional smooth periodic
pulse in the well-prepared space E to demonstrate the asymptotic accuracy and the asymptotic
order of convergence (AOC); see also [2] for related numerical experiments and their results. We
have used the ARS(2,2,2) variant of the IMEX-RK scheme in all the test problems.
Remark 6.1. Our numerical computations are carried out using a reformulation of the semi-implicit
scheme. First, an elliptic equation for the density is obtained by eliminating the velocity between
the mass and the momentum updates. The linear system resulting from the elliptic equation is
solved using the linear algebra sparse matrix solver UMFPACK. Finally, an explicit ﬂux evaluation
using the computed density in the momentum update yields the updated velocity.
6.1. Experimental Order of Convergence. We consider the following one-dimensional cosine
wave data
(6.1)
ρ(0, x1) = 1 +
ε2
1.185(1 + cos(2πεx1)),
u(0, x1) = ε(1 + cos(2πεx1)).
The computational domain is [−1
ε, 1
ε], and the boundaries are assumed to be periodic. The cosine
wave train is let to complete three cycles in the domain with an advection velocity u = 1. The
ﬁnal time T is chosen to be the time taken by the wave to complete three cycles in domain, i.e.
T = 3 ×
2
¯u+1/ε.
The simulations are performed for diﬀerent values of ε ranging in {100, 10−1, 10−2, 10−3}. As the
computational domain and the ﬁnal time change with ε, the EOC is obtained with respect to the
mesh size rather than to the number of mesh points. The CFL number is ﬁxed at 0.45. The EOC is
computed using L1 and L2 errors in ρ and u using the exact solution of the problem as the reference
solution. The Tables 1-4 clearly show that the scheme achieves second order convergence uniformly
with respect to ε.
N
∆x1
L1 error in ρ
L1 error in u
EOC
L2 error in ρ
L2 error in u
EOC
25
0.080000
1.533e-02
1.818e-02
1.702e-02
2.018e-02
50
0.040000
3.412e-03
4.048e-03
2.1677
3.788e-03
4.494e-03
2.1676
100
0.020000
8.214e-04
9.741e-04
2.0549
9.122e-04
1.081e-03
2.0543
200
0.010000
2.035e-04
2.412e-04
2.0133
2.260e-04
2.679e-04
2.0131
Table 1.
L1, L2 errors in ρ, u, and EOC for Problem 6.1 corresponding to ε = 100.
6.2. Travelling Vortex. We formulate a travelling vortex problem as follows
(6.2)
ρ(0, x) = 1.0,
u1(0, x) = −K(r) sin θ,
u2(0, x) = K(r) cos θ.


18
ARUN, DAS GUPTA, AND SAMANTARAY
N
∆x1
L1 error in ρ
L1 error in u
EOC
L2 error in ρ
L2 error in u
EOC
50
0.400000
1.778e-03
2.108e-02
1.975e-03
2.341e-02
100
0.200000
4.604e-03
5.457e-03
1.9498
5.113e-04
6.061e-03
1.9500
200
0.100000
1.159e-04
1.374e-03
1.9891
1.288e-04
1.527e-03
1.9889
400
0.050000
2.907e-05
3.446e-04
1.9961
3.229e-05
3.827e-04
1.9961
Table 2.
L1, L2 errors in ρ, u, and EOC for Problem 6.1 corresponding to ε = 10−1.
N
∆x1
L1 error in ρ
L1 error in u
EOC
L2 error in ρ
L2 error in u
EOC
800
0.250000
6.954e-06
8.240e-04
7.724e-06
9.153e-04
1600
0.125000
1.771e-06
2.098e-04
1.9732
1.967e-06
2.331e-04
1.9732
3200
0.062500
4.443e-07
5.265e-05
1.9950
4.935e-07
5.848e-05
1.9950
6400
0.031250
1.113e-07
1.319e-05
1.9961
1.237e-07
1.465e-05
1.9961
Table 3.
L1, L2 errors in ρ, u, and EOC for Problem 6.1 corresponding to ε = 10−2.
N
∆x1
L1 error in ρ
L1 error in u
EOC
L2 error in ρ
L2 error in u
EOC
3200
0.625000
3.903e-07
4.626e-04
4.336e-07
5.138e-04
6400
0.312500
1.089e-07
1.291e-04
1.8410
1.210e-07
1.434e-04
1.8415
12800
0.156250
2.783e-08
3.298e-05
1.9692
3.091e-08
3.663e-05
1.9692
25600
0.078125
6.991e-09
8.284e-06
1.9932
7.765e-09
9.201e-06
1.9932
Table 4.
L1, L2 errors in ρ, u, and EOC for Problem 6.1 corresponding to ε = 10−3.
Here r =
p
(x1 −0.5)2 + (x2 −0.5)2, θ = tan−1 
x2−0.5
x1−0.5

, and the radial function K is deﬁned as
(6.3)
K(r) =





5r,
if r < 0.2,
2 −5r,
if 0.2 ≤r < 0.4,
0,
otherwise.
The vortex is set to move in the domain [0, 4]×[0, 1] by prescribing an advection velocity (u1, u2) =
(1, 0). The CFL number used is 0.45, and the boundaries are periodic in both the directions. The
computations are carried out for times T ∈{1, 2, 3} with ε ranging in {100, 10−1, 10−2, 10−3}. In
Figure 1 we provide the Mach number plots for the entire range of ε mentioned before and for
each time T from the time-range.
For reference, we also plot the initial Mach number proﬁle.
First, it can be observed from the Mach number plots that the shape of the vortex doesn’t deform,
almost. Second, we can note that the shape of the Mach number proﬁle can be visually seen to be
independent of ε. Hence, it can be concluded that the numerical dissipation stays independent of
ε. This is further conﬁrmed by the the kinetic energy decay plot in Figure 2 in which we plot the
kinetic energy versus time T ∈[0, 3] for diﬀerent values ε mentioned above. It can be noted that
the decay of kinetic energy is almost negligible and the energy decay stays independent of ε, as the
plots corresponding to diﬀerent values of ε overlap completely .


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
19
Figure 1. Pseudo-color plots of the Mach numbers for the vortex problem. Top:
ε = 1, second: ε = 10−1, third: ε = 10−2, bottom: ε = 10−3, for times t = 0, 1, 2
and 3.


20
ARUN, DAS GUPTA, AND SAMANTARAY
Figure 2. Relative kinetic energy from t = 0 to t = 3.0 for diﬀerent values of ε.
6.3. Asymptotic Consistency. This test problem is to demonstrate the AP property. Let us
consider the following well-prepared initial data similar to that in [11].
(6.4)
ρ(0, x1, x2) = 1 + ε2 sin2(2π(x1 + x2)),
u(0, x1, x2) = sin(2π(x1 −x2)) + ε sin(2π(x1 + x2)),
u(0, x1, x2) = sin(2π(x1 −x2)) + ε cos(2π(x1 + x2)).
We set a very small value of ε, namely ε = 10−4. The computational domain [0, 1] × [0, 1] is divided
into an under resolved mesh of 40 × 40 cells. The boundaries are all taken to be periodic, and the
CFL is 0.45. The linearisation parameters are (u1, u2) = (1, 1) , ρ = 1 and a = 1. The ﬁnal time is
T = 3.
In Figure 3 we plot the density and the divergence of the velocity at time T = 0. Note that the
density perturbation is O(ε2) and the divergence perturbation is O(ε) initially. The corresponding
plots obtained using the numerical solution at time T = 3, clearly show that the density is almost
constant, and the velocity divergence is zero. Hence, we culminate that the numerical solution
approximates the incompressible solution ρ(0) = const., and ∇· u(0) = 0, demonstrating the AP
property of the scheme.
Further, we show in Figure 4 the transient behaviour of ∥∇ρ∥L2 and
∥∇· u∥L2 verses time, from t = 0 to t = 3. The ﬁgure clearly shows that if the initial data is close
to an incompressible data, then the numerical solution remains close to E for all times.
6.4. Asymptotic Order of Convergence and E-invariance. The aim of this experiment is
to numerically validate the second order asymptotic convergence of the numerical solution to the
incompressible limit solution. To this end, we consider the following exact solution of the incom-
pressible system (1.4) considered in [25], in which ϱ(0)(t, x1, x2) = 1, and
(6.5)
u1,(0)(t, x1, x2) = 1 −2 cos(2π(x1 −t)) sin(2π(x2 −t)),
u2,(0)(t, x1, x2) = 1 + 2 sin(2π(x1 −t)) cos(2π(x2 −t)),
ϱ(1)(t, x1, x2) = −cos(4π(x1 −t)) −cos(4π(x2 −t)).
The linearisation parameters are ρ = 1, a = 1 and (u1, u2) = (1, 1). The computational domain
Ω= [0, 1] × [0, 1] is successively divided into 20 × 20, 40 × 40, 80 × 80, and 160 × 160 mesh cells
and the CFL number used is 0.45.
The initial data used is obtained by setting ϱ(0, x1, x2) =


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
21
Figure 3. Top Left: ρ at T = 0.0 . Top Right: ∇· u at T = 0.0. Bottom Left: ρ
at T = 3.0. Bottom right: ∇· u at T = 3.0. Here, ε = 10−4.
Figure 4. Left: ∥∇ρ∥L2 from time T = 0 to 3. Right: ∥∇· u∥L2 from time T = 0
to 3. Here, ε = 10−4.


22
ARUN, DAS GUPTA, AND SAMANTARAY
ϱ(0)(0, x1, x2), u1(0, x1, x2) = u1,(0)(0, x1, x2) and u2(0, x1, x2) = u2,(0)(0, x1, x2). The boundaries
are assumed to be periodic in nature and the ﬁnal time for computation is T = 3.0.
First, the plot of the l1-norm of the gradient of the density and the divergence of the velocity at
the initial time, and at ﬁnal time T = 3 is given in Figure 5, with an under-resolved 40 × 40 mesh
for ε = 10−4. The Figure clearly shows that the IMEX-RK scheme leaves the well-prepared space
invariant as the density is a constant, and the velocity is divergence-free.
As deﬁned in [2], the EOC computed using incompressible data (ϱ(0), u1,(0), u2,(0)), as the reference
solution is termed as the asymptotic order of convergence (AOC).The numerical results obtained
show that the density ϱ remains constant exactly at 1, and hence both the velocity components are
used to measure the AOC. We compute the AOC for very small values of ε, namely for ε = 10−3
and ε = 10−4. The AOC obtained in both the L1 and L2 norms are presented in Tables 5 and
6. From the tables it can easily be seen that as ε →0 the numerical solution converges to the
incompressible solution with second order accuracy. This observation reiterate also the fact that
the chosen variant ARS(2,2,2) is stiﬄy accurate; see also [3].
In Figure 6 we plot the L2 norms of the gradient of the density and divergence of the velocity
over the entire computational time range T ∈[0, 3]. It clearly shows that the numerical solution
stays in E for all times. Hence, it substantiate the E-invariance with respect to time
N
L1 error in u1
AOC
L2 error in u1
AOC
L1error in u2
AOC
L2 error in u2
AOC
20
2.670e-01
3.034e-01
2.670e-01
3.034e-01
40
6.931e-02
1.9461
7.749e-02
1.9692
6.931e-02
1.9461
7.749e-02
1.9692
80
1.734e-02
1.9984
1.930e-02
2.0054
1.734e-02
1.9984
1.930e-02
2.0054
160
4.332e-03
2.0015
4.814e-03
2.0034
4.332e-03
2.0015
4.814e-03
2.0034
Table 5.
L1, L2 errors in u1, u2, and AOC for Problem 6.4 corresponding to ε = 10−3.
N
L1 error in u1
AOC
L2 error in u1
AOC
L1error in u2
AOC
L2 error in u2
AOC
20
2.670e-01
3.034e-01
2.670e-01
3.034e-01
40
6.931e-02
1.9461
7.749e-02
1.9692
6.931e-02
1.9461
7.749e-02
1.9692
80
1.734e-02
1.9984
1.930e-02
2.0054
1.734e-02
1.9984
1.930e-02
2.0054
160
4.332e-03
2.0015
4.814e-03
2.0034
4.332e-03
2.0015
4.814e-03
2.0034
Table 6.
L1, L2 errors in u1, u2, and AOC for Problem 6.4 corresponding to ε = 10−4.
7. Concluding Remarks
In this paper, we have presented a detailed analysis of an IMEX-RK ﬁnite volume scheme for the
linear wave equation system in the zero Mach number regime. The existence of a unique numerical
solution, its uniform stability with respect to ε, the AP property, and the asymptotic accuracy are
shown for the time semi-discrete scheme using saddle point theory of variational problems. Results
from the theory of circulant matrices are used to establish the same features for the space-time fully-
discrete scheme, obtained via a ﬁnite volume discretisation. Extensive numerical studies are carried
out to test the various theoretical concepts discussed. Uniform second-order convergence is achieved
with respect to ε, experimentally. The dissipation of the scheme is shown to be independent of ε. The
experiments reveal that the scheme is AP, and also achieves second-order asymptotic convergence,
leaving the well-prepared space E invariant.
Hence, the numerical case studies validate all the
theoretical ﬁndings.


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
23
Figure 5. Top Left: ∥∇ρ∥l1 at T = 0. Top Right: ∇· u at T = 0. Bottom Left:
∥∇ρ∥l1 at T = 3. Bottom right: ∇· u at T = 3. Here, ε = 10−4.
Figure 6. Left: ∥∇ρ∥L2 from time T = 0 to 3. Right: ∥∇· u∥L2 from time T = 0
to 3. Here, ε = 10−4.


24
ARUN, DAS GUPTA, AND SAMANTARAY
References
[1] K. R. Arun, A. J. Das Gupta, and S. Samantaray. An implicit-explicit scheme accurate at low Mach numbers for
the wave equation system. In Theory, numerics and applications of hyperbolic problems. I, volume 236 of Springer
Proc. Math. Stat., pages 97–109. Springer, Cham, 2018.
[2] K. R. Arun and S. Samantaray. Asymptotic Preserving and Low Mach Number Accurate IMEX Finite Volume
Schemes for the Euler Equations. arXiv e-prints, page arXiv:1907.01711, Jul 2019.
[3] U. M. Ascher, S. J. Ruuth, and R. J. Spiteri. Implicit-explicit Runge-Kutta methods for time-dependent partial
diﬀerential equations. Appl. Numer. Math., 25(2-3):151–167, 1997. Special issue on time integration (Amsterdam,
1996).
[4] D. S. Bernstein. Matrix mathematics. Princeton University Press, Princeton, NJ, second edition, 2009. Theory,
facts, and formulas.
[5] G. Bispen, K. R. Arun, M. Luk´aˇcov´a-Medvid’ov´a, and S. Noelle. IMEX large time step ﬁnite volume methods
for low Froude number shallow water ﬂows. Commun. Comput. Phys., 16(2):307–347, 2014.
[6] F. Brezzi and M. Fortin. Mixed and hybrid ﬁnite element methods, volume 15 of Springer Series in Computational
Mathematics. Springer-Verlag, New York, 1991.
[7] P. G. Ciarlet. Linear and Nonlinear Functional Analysis with Applications. Society for Industrial and Applied
Mathematics, Philadelphia, PA, USA, 2013.
[8] F. Cordier, P. Degond, and A. Kumbaro. An asymptotic-preserving all-speed scheme for the Euler and Navier-
Stokes equations. J. Comput. Phys., 231(17):5685–5704, 2012.
[9] P. Degond. Asymptotic-preserving schemes for ﬂuid models of plasmas. In Numerical models for fusion, volume
39/40 of Panor. Synth`eses, pages 1–90. Soc. Math. France, Paris, 2013.
[10] P. Degond, F. Deluzet, A. Lozinski, J. Narski, and C. Negulescu. Duality-based asymptotic-preserving method
for highly anisotropic diﬀusion equations. Commun. Math. Sci., 10(1):1–31, 2012.
[11] P. Degond and M. Tang. All speed scheme for the low Mach number limit of the isentropic Euler equations.
Commun. Comput. Phys., 10(1):1–31, 2011.
[12] S. Dellacherie. Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number.
J. Comput. Phys., 229(4):978–1016, 2010.
[13] R. M. Gray. Toeplitz and circulant matrices:, 2006. A review.
[14] E. Hairer and G. Wanner. Solving ordinary diﬀerential equations. II, volume 14 of Springer Series in Computa-
tional Mathematics. Springer-Verlag, Berlin, second edition, 1996. Stiﬀand diﬀerential-algebraic problems.
[15] R. A. Horn and C. R. Johnson. Matrix analysis. Cambridge University Press, Cambridge, 1985.
[16] R. A. Horn and C. R. Johnson. Topics in matrix analysis. Cambridge University Press, Cambridge, 1991.
[17] S. Jin. Eﬃcient asymptotic-preserving (AP) schemes for some multiscale kinetic equations. SIAM J. Sci. Comput.,
21(2):441–454, 1999.
[18] S. Jin. Asymptotic preserving (AP) schemes for multiscale kinetic and hyperbolic equations: a review. Riv. Math.
Univ. Parma (N.S.), 3(2):177–216, 2012.
[19] C. A. Kennedy and M. H. Carpenter. Additive Runge-Kutta schemes for convection-diﬀusion-reaction equations.
Applied Numerical Mathematics, 44(1):139 – 181, 2003.
[20] S. Klainerman and A. Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the
incompressible limit of compressible ﬂuids. Comm. Pure Appl. Math., 34(4):481–524, 1981.
[21] R. Klein. Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics. I. One-
dimensional ﬂow. J. Comput. Phys., 121(2):213–237, 1995.
[22] S. Noelle, G. Bispen, K. R. Arun, M. Luk´aˇcov´a-Medviˇdov´a, and C.-D. Munz. A weakly asymptotic preserving
low Mach number scheme for the Euler equations of gas dynamics. SIAM J. Sci. Comput., 36(6):B989–B1024,
2014.
[23] L. Pareschi and G. Russo. Implicit-explicit Runge-Kutta schemes for stiﬀsystems of diﬀerential equations. In
Recent trends in numerical analysis, volume 3 of Adv. Theory Comput. Math., pages 269–288. Nova Sci. Publ.,
Huntington, NY, 2001.
[24] L. Pareschi and G. Russo. Implicit-Explicit Runge-Kutta schemes and applications to hyperbolic systems with
relaxation. J. Sci. Comput., 25(1-2):129–155, 2005.
[25] T. Schneider, N. Botta, K. J. Geratz, and R. Klein. Extension of ﬁnite volume compressible ﬂow solvers to
multi-dimensional, variable density zero Mach number ﬂows. J. Comput. Phys., 155(2):248–286, 1999.
[26] H. Zakerzadeh. Asymptotic Preserving Finite Volume Schemes For The Singularly-perturbed Shallow Water Equa-
tions with Source Terms. PhD thesis, RWTH Aachen, Germany, 2017.
[27] H. Zakerzadeh and S. Noelle. A note on the stability of implicit-explicit ﬂux-splittings for stiﬀsystems of hyper-
bolic conservation laws. Commun. Math. Sci., 16(1):1–15, 2018.


AN AP LOW MACH NUMBER ACCURATE SCHEME FOR THE WAVE EQUATION SYSTEM
25
School of Mathematics, Indian Institute of Science Education and Research Thiruvananthapuram,
Thiruvananthapuram - 695551, India
E-mail address: arun@iisertvm.ac.in
Department of Mathematics, Ramakrishna Mission Vidyamandira, Howrah - 711202, India
E-mail address: arnab.math@vidyamandira.ac.in
School of Mathematics, Indian Institute of Science Education and Research Thiruvananthapuram,
Thiruvananthapuram - 695551, India
E-mail address: sauravsam13@iisertvm.ac.in


