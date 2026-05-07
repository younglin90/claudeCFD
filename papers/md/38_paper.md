Accepted Manuscript
A ﬂux splitting method for the Baer–Nunziato equations of compressible two-phase ﬂow
S.A. Tokareva, E.F. Toro
PII:
S0021-9991(16)30314-X
DOI:
http://dx.doi.org/10.1016/j.jcp.2016.07.019
Reference:
YJCPH 6734
To appear in:
Journal of Computational Physics
Received date:
25 April 2016
Revised date:
15 July 2016
Accepted date:
19 July 2016
Please cite this article in press as: S.A. Tokareva, E.F. Toro, A ﬂux splitting method for the Baer–Nunziato equations of compressible
two-phase ﬂow, J. Comput. Phys. (2016), http://dx.doi.org/10.1016/j.jcp.2016.07.019
This is a PDF ﬁle of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing
this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is
published in its ﬁnal form. Please note that during the production process errors may be discovered which could affect the content, and all
legal disclaimers that apply to the journal pertain.


A ﬂux splitting method for the Baer-Nunziato equations of compressible
two-phase ﬂow
S. A. Tokarevaa, E. F. Torob
aInstitute of Mathematics, University of Zurich, Switzerland
bLaboratory of Applied Mathematics, DICAM, University of Trento, Italy
Abstract
Here we extend the Toro-V´azquez ﬂux vector splitting approach (TV), originally proposed for the ideal 1D
Euler equations in [1], to the Baer-Nunziato equations of compressible two-phase ﬂow. Following the TV
approach we identify corresponding advection and pressure operators. We perform a rigorous analysis of
the associated non-conservative pressure system and derive its complete characteristic structure. The choice
of the advection numerical ﬂux is obvious. For the pressure system, several schemes are presented. The
complete schemes are then implemented in the setting of ﬁnite volume and path-conservative methods and
are systematically assessed in terms of accuracy and eﬃciency, through a carefully selected suite of test
problems. The presented schemes constitute a building block for the construction of high-order numerical
methods for solving the Baer-Nunziato equations. Here, as an illustrative example of such possibility, we
present the construction of a second-order scheme.
Keywords:
Compressible multiphase ﬂow, non-conservative systems, ﬂux splitting.
1. Introduction
The Baer-Nunziato equations are a ﬁrst-order system of eleven non-linear partial diﬀerential equations
that model the dynamics of a three-dimensional ﬂowing mixture of two compressible materials or phases,
typically a solid particle phase and a gaseous phase. The model was ﬁrst proposed by Baer and Nunziato
[2] in the context of granular energetic combustible materials embedded in gaseous combustion products. A
distinctive feature of the Baer-Nunziato model is the admission of two velocity vectors and two pressures.
The equations are hyperbolic, except for some well identiﬁed situations, and the complete mathematical
structure of the 1D system, as well as split 3D system, is available [3, 4]. However, the equations cannot be
written in conservation-law form, which has for a long time remained a challenge from both the mathematical
and numerical points of view. Regarding the design of upwind schemes for these equations, perhaps the
earliest attempt to do so via the Riemann problem is that reported in [5], with encouraging results. More
recent works in this direction include [4, 6–16], to name but a few.
Essentially, most of the advanced computational methods for compressible ﬂows can be classiﬁed as
upwind methods and centred methods. In spite of their increased complexity, upwind methods tend to be
preferred when having to resolve ﬁne features, particularly those associated with intermediate characteristic
ﬁelds. Among upwind methods, two possible choices are the Godunov approach [17] and associated Riemann
solvers [18], and the Flux Vector Splitting (FVS) approach. FVS schemes provide upwinding for fast waves
at a lower computational eﬀort and algorithm complexity than the Godunov approach with a good Riemann
solver. It is highly desirable that numerical methods for solving hyperbolic equations are not just robust
but also able to resolve intermediate characteristic ﬁelds accurately.
The latter requirement is not met
neither by centred ﬂuxes, nor by incomplete Riemann solvers or classical ﬂux vector splitting methods. The
Email addresses: svetlana.tokareva@math.uzh.ch (S. A. Tokareva), eleuterio.toro@unitn.it (E. F. Toro)
Preprint submitted to Elsevier
July 21, 2016


inability to resolve intermediate characteristic ﬁelds badly aﬀects the correct resolution of contact waves,
material interfaces, shear waves, vortices and ignition fronts, for example. Early FVS schemes, such as those
reported in [19–21], suﬀered from excessive numerical dissipation for contacts, shear waves and shear layers.
However, this shortcoming has been resolved by a more recent FVS approach known as AUSM (Advection
Upstream Splitting Method) scheme of Liou and Steﬀen [22–24]. The AUSM scheme has also been applied
to equations of compressible multiphase ﬂow, see for instance [9, 25].
An alternative ﬂux splitting scheme, called the TV ﬂux splitting method, was ﬁrst presented in [1] for
the one-dimensional Euler equations for ideal gases. Recently, the TV scheme has been extended to the
three-dimensional Euler equations with general equation of state in [26] and to the equations of magnetohy-
drodynamics in [27]. A distinctive feature of the TV ﬂux is that it separates completely advection terms from
pressure terms, thus providing the possibility of taking advantage of their diverse speeds of propagation.
Toro and V´azquez [1] also proposed a numerical approach that emerges from two separate systems produced
by their ﬂux vector splitting, the advection system and the pressure system. For the Euler equations it
turned out that the pressure system is hyperbolic and a very simple solution of the associated Riemann
problem by applying a linearization across the characteristics provides all the items required for ﬂux evalu-
ation, and the resulting scheme proved to be simple, robust and very eﬃcient. In this paper, we extend the
TV ﬂux vector splitting approach to the case of Baer-Nunziato equations of compressible two-phase ﬂow.
We design several schemes to calculate the numerical ﬂuxes for the pressure system in ﬁnite-volume and
path-conservative setting. The resulting methods are systematically assessed for accuracy, robustness and
eﬃciency on a carefully selected suite of test problems.
The rest of the paper is organized as follows. In Section 2, we describe the Baer-Nunziato system for
the x-split three-dimensional equations. Next, in Section 3, we apply the idea of the TV ﬂux splitting to
the Baer-Nunziato equations and introduce the corresponding pressure and advection systems. In Section 4,
we present a rigorous analysis of the non-conservative pressure system, for which we derive its complete
characteristic structure. In Section 5 we propose several Riemann solvers for the pressure system, while
in Section 6 we assess their performance and illustrate, as an example, the construction of second-order
versions of the schemes.
2. The Baer-Nunziato equations
The Baer-Nunziato equations are a three-dimensional time-dependent system of eleven partial diﬀerential
equations with source terms. Our ultimate goal is to develop numerical schemes to solve these, using for
example ﬁnite volume or path-conservative methods. To this end it is helpful to consider the governing
equations in the direction normal to a cell boundary to ﬁnd a numerical ﬂux.
Hence, without loss of
generality we can consider the x-split equations:
∂tQ + ∂xF(Q) + T(Q)∂x ¯α = S(Q),
(1)
where
Q =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯α
¯α¯ρ
¯α¯ρ¯u
¯α¯ρ¯v
¯α¯ρ ¯w
¯α¯ρ ¯E
αρ
αρu
αρv
αρw
αρE
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
F(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯α¯ρ¯u
¯α

¯ρ¯u2 + ¯p
	
¯α¯ρ¯u¯v
¯α¯ρ¯u ¯w
¯α¯u

¯ρ ¯E + ¯p
	
αρu
α

ρu2 + p
	
αρuv
αρuw
αu (ρE + p)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
T(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯u
0
−p
0
0
−p¯u
0
p
0
0
p¯u
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
S(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
s1
s2
s3
s4
s5
s6
s7
s8
s9
s10
s11
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
.
The ﬁrst six equations for variables with bar refer to the solid phase and the remaining ﬁve equations
refer to the gas phase. Here ρ, u, v, w, p, E are gas density, velocity components, pressure and total energy,
and ¯ρ, ¯u, ¯v, ¯w, ¯p, ¯E are the corresponding variables for the solid phase; α and ¯α are volume fractions.
2


System (1) requires additional closure relations involving density, internal energy and pressure of each
phase. Such relations are provided by the equations of state (EOS). An ideal equation of state (EOS) for
the gas phase and a stiﬀened EOS for the solid phase are frequently used:
p = (γ −1)ρe,
¯p = (¯γ −1)¯ρ¯e −¯γ ¯P0,
where e and ¯e are the speciﬁc internal energies, γ and ¯γ are the speciﬁc heat ratios of the gas and solid
phases, respectively, and ¯P0 is a known constant. The sound speeds of the gas and solid phases are calculated
as follows
a =

γp
ρ , ¯a =

¯γ(¯p + ¯P0)
¯ρ
,
Solid and gas volume fractions are related through the saturation condition:
¯α + α = 1.
In this paper, we are primarily interested in the principal part of equations (1) and therefore we restrict
ourselves to the homogeneous case S(Q) = 0. Equations (1) include also the purely one-dimensional Baer-
Nunziato equations in the case of no tangential velocity components.
Therefore the study of the split
three-dimensional equations is useful both for one-dimensional and multidimensional cases.
3. Toro-V´azquez ﬂux splitting method for the Baer-Nunziato equations
Consider now the homogeneous one-dimensional Baer-Nunziato equations
∂tQ + ∂xF(Q) + T(Q)∂x ¯α = 0
(2)
with
Q =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯α
¯α¯ρ
¯α¯ρ¯u
¯α¯ρ ¯E
αρ
αρu
αρE
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
F(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯α¯ρ¯u
¯α

¯ρ¯u2 + ¯p
	
¯α¯u

¯ρ ¯E + ¯p
	
αρu
α

ρu2 + p
	
αu (ρE + p)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
T(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯u
0
−p
−p¯u
0
p
p¯u
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
.
We follow the Toro-V´azquez ﬂux splitting approach [1] taking into account that the equations of in-
terest here do not have a conservation-law form. First, we identify the conservative part and express the
conservative ﬂux as the sum of advection and pressure ﬂuxes as follows
F(Q) = A(Q) + P(Q),
(3)
that is,
F(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯α¯ρ¯u
¯α

¯ρ¯u2 + ¯p
	
¯α¯u
 1
2 ¯ρ¯u2 + ¯ρ¯e + ¯p
	
αρu
α

ρu2 + p
	
αu
 1
2ρu2 + ρe + p
	
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
=
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯α¯ρ¯u
¯α¯ρ¯u2
1
2 ¯α¯ρ¯u3
αρu
αρu2
1
2αρu3
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
+
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
¯α¯p
¯α¯u (¯ρ¯e + ¯p)
0
αp
αu (ρe + p)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
(4)
3


with the respective advection and pressure ﬂuxes deﬁned as
A(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯α¯ρ¯u
¯α¯ρ¯u2
1
2 ¯α¯ρ¯u3
αρu
αρu2
1
2αρu3
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
P(Q) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
¯α¯p
¯α¯u (¯ρ¯e + ¯p)
0
αp
αu (ρe + p)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
.
(5)
Following [1, 26], we consider two systems, the advection system (A-system) and the pressure system
(P-system), noting however that here the pressure system is augmented by the nonconservative term present
in the Baer-Nunziato equations. Thus, the two systems are
∂tQ + ∂xA(Q) = 0,
(advection system, conservative)
∂tQ + ∂xP(Q) + T(Q)∂x ¯α = 0.
(pressure system, non-conservative)
(6)
The target of this paper is the numerical solution of the full Baer-Nunziato system of equations. The TV
ﬂux splitting approach consists of approximating the numerical ﬂuxes for the pressure system and advection
system separately and constructing the numerical ﬂuxes for the full system based on these. To this end, the
analysis of the eigenstructure and the study of the Riemann problem for the pressure system are necessary.
4. Non-conservative pressure system
The construction of the numerical ﬂux corresponding to the advection system is straightforward and
follows directly from [1]. We now turn our attention to the non-conservative pressure system by considering
the associated Riemann problem
∂tQ + ∂xP(Q) + T(Q)∂x ¯α = 0,
Q(x, 0) =

QL, if x < 0,
QR, if x > 0
(7)
with piece-wise constant initial data QL, QR. In terms of primitive variables the Riemann problem (7) can
be reformulated as
∂tV + B(V)∂xV = 0,
Q(x, 0) =

VL, if x < 0,
VR, if x > 0,
(8)
where
V =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯α
¯ρ
¯u
¯p
ρ
u
p
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
B(V) =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯u
0
0
0
0
0
0
−¯ρ¯u
¯α
0
0
0
0
0
0
−Δp
¯α¯ρ
0
0
1
¯ρ
0
0
0
¯u
¯α¯e¯
p (¯e¯ρ¯ρ + ¯e)
¯u(¯ρ¯e¯
ρ+¯e)
¯ρ¯e¯
p
¯h
¯e¯
p
¯u
0
0
0
ρ¯u
α
0
0
0
0
0
0
0
0
0
0
0
0
1
ρ
1
αep

−¯ueρρ −pΔu
ρ
−ue

0
0
0
u(ρeρ+e)
ρep
h
ep
u
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
and
Δp = p −¯p,
Δu = u −¯u.
The eigenvalues of the matrix B(V) are
λ1 = 1
2(u −A),
λ2 = 0,
λ3 = 1
2(u + A),
λ4 = 1
2(¯u −¯A),
λ5 = 0,
λ6 = 1
2(¯u + ¯A),
λ7 = ¯u,
(9)
4


where
A =

u2 + 4h
ρep
,
¯A =

¯u2 + 4¯h
¯ρ¯e¯p
.
The corresponding linearly independent right eigenvectors are found to be
K1 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
0
0
0
2
ρ(u −A)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K2 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
0
0
1
−u(ρeρ+e)
ρh
0
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K3 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
0
0
0
2
ρ(u + A)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K4 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
2
¯ρ(¯u −¯A)
0
0
0
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K5 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
1
−¯u(¯ρ¯e¯
ρ+¯e)
¯ρ¯h
0
0
0
0
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K6 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
2
¯ρ(¯u + ¯A)
0
0
0
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
,
K7 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
1
−¯ρ
¯α
0
Δp
¯αρ
α
Δu(p−eρρ2)
αρ(h+epρ¯uΔu)
¯uΔu(p−eρρ2)
α(h+epρ¯uΔu)
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
.
(10)
As it can be observed from (10), the phase concentration ¯α only changes across the λ7 characteristic ﬁeld.
Moreover, the gas density ρ is constant across the λ1 and λ3 characteristic ﬁelds, while the solid density
¯ρ is constant across the λ4 and λ6 characteristic ﬁelds. The gas pressure remains constant across the λ2
characteristic and the solid pressure does not change across the λ5 characteristic. Finally, the solid velocity
is constant across the λ7 characteristic.
A typical characteristic structure of the solution of the Riemann problem for the pressure system is
shown in Figs. 1, 2, where SL, SM, SR and ¯SL, S0, ¯SM, ¯SR denote the speeds of the characteristics of
the gas and solid phase, respectively. In this paper, we restrict ourselves to the so-called ”subsonic” wave
conﬁgurations, for which SL < ¯SM = ¯u < SR, since such situation is considered by many authors to be more
physically relevant [4, 16, 28, 29]. The case illustrated in Figs. 1, 2 corresponds to the right subsonic wave
conﬁguration, i. e. SL < ¯SM = ¯u < SR, when ¯u > 0. By inspecting the components of the eigenvectors
Ki, i = 1, . . . , 7, we immediately conclude that ρ∗
L = ρL, ρ∗
R = ρR, p0 = p∗
L for the gas phase and ¯ρ∗
L = ¯ρL,
¯ρ∗
R = ¯ρR, ¯p0 = ¯p∗
L, ¯u0 = ¯u∗
R for the solid phase.
Since λ1 = 1
2(u −A) < λ2 = 0 < λ3 = 1
2(u + A) and λ4 = 1
2(¯u −¯A) < λ5 = 0 < λ6 = 1
2(¯u + ¯A), the
Godunov state in the subsonic wave conﬁguration will be completely deﬁned by the sign of λ7 = ¯u resulting
in signiﬁcant CPU time savings in the sampling procedure. We also note that due to the above mentioned
conditions no entropy ﬁx will be needed for linearized ﬂuxes.
5. Riemann solvers for the pressure system
In this section, we construct several Riemann solvers and corresponding numerical ﬂuxes for the pressure
system.
5


x
0
t
SR
SL
¯SM
SM
ρL
uL
pL
ρ∗
L
u∗
L
p∗
L
ρ0
u0
p0
ρ∗
R
u∗
R
p∗
R
ρR
uR
pR
αL
αR
Figure 1: Intermediate states for gas phase
x
0
t
¯SR
¯SL
¯SM
¯S0
¯ρL
¯uL
¯pL
¯ρ∗
L
¯u∗
L
¯p∗
L
¯ρ0
¯u0
¯p0
¯ρ∗
R
¯u∗
R
¯p∗
R
¯ρR
¯uR
¯pR
¯αL
¯αR
Figure 2: Intermediate states for solid phase
5.1. Linearized Riemann solver for the P-system
Consider the conﬁguration of characteristic waves as illustrated in Figs. 1, 2. This wave conﬁguration
corresponds to the ”subsonic” case, when the solid contact travelling with speed ¯SM is situated between the
left and right nonlinear waves of the gas phase, i.e. when the corresponding wavespeeds satisfy the condition
SL < ¯SM < SR. Moreover, we assume that ¯SM > 0 and therefore the solid contact lies to the right of the
gas contact.
5.1.1. Fully linearized Riemann solver
Here we attempt to approximate the intermediate states of the Riemann problem using the linearization
of the generalized Riemann invariants across all the characteristic ﬁelds.
Gas phase. The generalized Riemann invariants for the left non-linear characteristic wave of the gas
phase have the form
dρ
0 = du
2 =
dp
ρ(u −A),
(11)
6


which can be obviously rewritten as a system of ordinary diﬀerential equations in phase space
dρ
0 = du
2 ,
du
2 =
dp
ρ(u −A).
(12)
The ﬁrst equation of the system (12) gives ρ = const, while the second equation is equivalent to
du =
2
ρ(u −A)dp.
(13)
We linearize equation (13) by evaluating the term C = ρ(u −A) at the foot of the characteristic and
therefore assuming C ≈CL = ρL(uL −AL). Then the solution of (13) will be written as
u −uL =
2
CL
(p −pL),
(14)
and taking u = u∗
L and p = p∗
L we obtain the following approximation for the left intermediate gas pressure:
p∗
L = pL + 1
2CL(u∗
L −uL).
(15)
For the left intermediate density we have
ρ∗
L = ρL.
(16)
For the right characteristic we have the generalized Riemann invariants in the form
dρ
0 = du
2 =
dp
ρ(u + A),
(17)
leading to ρ = const and therefore
ρ∗
R = ρR,
(18)
and an approximation for p∗
R as
p∗
R = pR + 1
2CR(u∗
R −uR),
(19)
with CR = ρR(uR + AR).
The invariants corresponding to the gas contact are
dρ
1 =
du
−u(ρeρ+e)
ρh
= dp
0 ,
(20)
meaning that p = const and therefore
p0 = p∗
L.
(21)
Note that for the ideal or stiﬀened EOS the term −u(ρeρ+e)
ρh
vanishes and therefore u = const across the
gas contact wave, however, this may not be the case in general.
Solid phase. The generalized Riemann invariants for the left characteristic wave of the solid phase are
d¯ρ
0 = d¯u
2 =
d¯p
¯ρ(¯u −¯A),
(22)
resulting in ¯ρ = const. Proceeding with the linearization as for the gas phase we get
¯p∗
L = ¯pL + 1
2
¯CL(¯u∗
L −¯uL),
(23)
7


where ¯CL = ¯ρL(¯uL −¯AL), and
¯ρ∗
L = ¯ρL .
(24)
Similarly, for the right characteristic one has
d¯ρ
0 = d¯u
2 =
d¯p
¯ρ(¯u + ¯A)
(25)
and therefore ¯ρ = const. The linearization of the right Riemann invariant gives an approximation for the
pressure as
¯p∗
R = ¯pR + 1
2
¯CR(¯u∗
R −¯uR),
(26)
where ¯CR = ¯ρR(¯uR + ¯AR), and for the right intermediate density
¯ρ∗
R = ¯ρR .
(27)
The generalized Riemann invariants corresponding to the solid contact associated to λ5 = 0 are
d¯ρ
1 =
d¯u
−¯u(¯ρ¯e¯
ρ+¯e)
¯ρ¯h
= d¯p
0 ,
(28)
and hence
¯p0 = ¯p∗
L.
(29)
The generalized Riemann invariants at the solid contact corresponding to λ7 = ¯u take the form
d¯α
1 =
d¯ρ
−¯ρ/¯α = d¯u
0 =
d¯p
Δp/¯α = dρ
ρ/α =
du
B/(αρ) =
dp
¯uB/α,
(30)
with deﬁnitions
B =
Δu(p −eρρ2)
αρ(h + epρ¯uΔu),
Δp = p −¯p,
Δu = u −¯u.
From (30) we immediately get the following expressions for the solid and gas densities ¯ρ0 and ρ0:
¯ρ0 = ¯αR¯ρR
¯αL
,
(31)
ρ0 = αRρR
αL
.
(32)
We also note that ¯u = const across the solid contact, therefore
¯u0 = ¯u∗
R.
(33)
Consider now the equation
d¯ρ
−¯ρ/¯α =
d¯p
Δp/¯α ,
which becomes
d¯ρ +
¯ρ
Δpd¯p = 0 .
(34)
We choose to evaluate the nonlinear term ¯ρ/Δp at the right initial state and use it to linearize equation
(34):
d¯ρ +
¯ρR
ΔpR
d¯p = 0.
(35)
8


The solution of (35) gives the following relation between the values ¯ρ0 and ¯p0:
¯p0 −¯p∗
R = −ΔpR
¯ρR
(¯ρ0 −¯ρR) = −ΔpR
¯ρR
 ¯αR¯ρR
¯αL
−¯ρR

= −ΔpR
¯αL
(¯αR −¯αL).
(36)
Combining (23), (26), (29) and (33) with (36) and assuming
¯u0 ≈¯u∗
L ≡¯u∗
0,
(37)
which becomes an equality in the case of ideal EOS for the solid phase, we get
¯pL + 1
2
¯CL(¯u∗
0 −¯uL) −¯pR −1
2
¯CR(¯u∗
0 −¯uR) ≈−ΔpR
¯αL
(¯αR −¯αL),
(38)
so that the approximation of ¯u∗
0 becomes
¯u∗
0 ≈
¯CR¯uR −¯CL¯uL
¯CR −¯CL
−
2
¯CR −¯CL
(¯pR −¯pL) +
2
¯CR −¯CL
ΔpR
¯αL
(¯αR −¯αL).
(39)
Remark 1. In the case of ideal or stiﬀened EOS for the solid phase the approximate formula (39) becomes
equality.
Remark 2. If ¯αL = ¯αR, then the intermediate velocity approximation (39) coincides with the approxima-
tion of [1, 26].
Consider next the equations
dρ −ρ2
B du = 0
(40)
and
du −1
ρ¯udp = 0.
(41)
Linearization of these equations at the right state gives
dρ −ρ2
R
BR
du = 0
(42)
and
du −
1
ρR¯uR
dp = 0,
(43)
and therefore we can write
ρ0 −ρ∗
R −ρ2
R
BR
(u0 −u∗
R) = 0
(44)
and
(u0 −u∗
R) −
1
ρR¯uR
(p0 −p∗
R) = 0.
(45)
Assuming
u0 ≈u∗
L ≡u∗
0
(46)
and using (15), (19) and (32) in equations (44), (45), we obtain the following linear system with respect to
u∗
0 and u∗
R:
αR −αL
αL
−ρR
BR
(u∗
0 −u∗
R) ≈0,
(47)
u∗
0 −u∗
R −
1
ρR¯uR

pL + 1
2CL(u∗
L −uL) −pR −1
2CR(u∗
R −uR)

≈0.
(48)
9


The solution of the system (47),(48) is given by
u∗
0 ≈CRuR −CLuL
CR −CL
−
2
CR −CL
(pR −pL) −
2
CR −CL
αR −αL
αL
BR

¯uR −1
2
CR
ρR

,
(49)
u∗
R ≈u∗
0 −BR
ρR
αR −αL
αL
.
(50)
Remark 3. For ideal gas phase EOS the assumption (46) is exact, therefore the formulae (49),(50) become
equalities.
Remark 4. If αL = αR, then the velocity approximation (49) reduces to the approximation of [1, 26].
Remark 5. Note that for the linearization of the Riemann invariants across the solid contact at λ7 = ¯u we
have chosen to evaluate the reference parameters at the right state. Unfortunately, this choice does not lead
to a very robust Riemann solver for a wide range of test problems. However, such linearized solver provides
a suﬃciently accurate guess value for the solution of the nonlinear thin-layer equations connecting the ﬂow
parameters across the solid contact. These equations are derived in the next subsection.
5.1.2. Thin layer equations for the pressure system
Our preliminary numerical experiments have shown that the fully linearized Riemann solver for the
pressure system provides an accurate approximation to the solution for rather mild Riemann problems, but
fails to resolve correctly some more demanding test problems. Therefore, at the solid contact, we propose
to include the non-linear thin-layer equations which relate the intermediate states across this wave. The
derivation of these equations is analogous to the one presented in [28].
Recall the equations of the pressure system:
∂t¯α + ¯u∂x ¯α = 0,
(51)
∂t(¯α¯ρ) = 0,
(52)
∂t(¯α¯ρ¯u) + ∂x(¯α¯p) −p∂x ¯α = 0,
(53)
∂t(¯α¯ρ ¯E) + ∂x

¯α¯u(¯ρ¯e + ¯p)
	
−p¯u∂x ¯α = 0,
(54)
∂t(αρ) = 0,
(55)
∂t(αρu) + ∂x(αp) + p∂x ¯α = 0,
(56)
∂t(αρE) + ∂x

αu(ρe + p)
	
+ p¯u∂x ¯α = 0.
(57)
Introducing a self-similarity variable ξ = x −Ut with U = const, we can rewrite the equations (51)–(57)
with respect to ξ as
−U∂ξ ¯α + ¯u∂ξ ¯α = 0,
(58)
−U∂ξ(¯α¯ρ) = 0,
(59)
−U∂ξ(¯α¯ρ¯u) + ∂ξ(¯α¯p) −p∂ξ ¯α = 0,
(60)
−U∂ξ(¯α¯ρ ¯E) + ∂ξ

¯α¯u(¯ρ¯e + ¯p)
	
−p¯u∂ξ ¯α = 0,
(61)
−U∂ξ(αρ) = 0,
(62)
−U∂ξ(αρu) + ∂ξ(αp) + p∂ξ ¯α = 0,
(63)
−U∂ξ(αρE) + ∂ξ

αu(ρe + p)
	
+ p¯u∂ξ ¯α = 0.
(64)
Equation (58) leads to U = ¯u = const, and from equation (59) as well as equation (61) we obtain
¯α¯ρ = const.
(65)
Equation (60) gives
p∂ξ ¯α = ∂ξ(¯α¯p).
(66)
10


From equation (62) we get
αρ = const.
(67)
From equation (63) together with (66) we can derive
−αρu¯u + αp + ¯α¯p = const.
(68)
Equation (64) with (66) leads to
αρe(u −¯u) −1
2αρ¯uu2 + αup + ¯α¯u¯p = const.
(69)
Finally, the complete system of thin-layer equations with respect to the unknowns
(u∗
L, u∗
R, ¯u∗
L, ¯u∗
R, p∗
L, p∗
R, ¯p∗
L, ¯p∗
R)
combined with the linearized Riemann invariants across left and right nonlinear wave of both phases reads
¯u∗
R −¯u∗
L = 0,
(70)
−αRρR(u∗
R¯u∗
R −u∗
L¯u∗
L) + αRp∗
R + ¯αR¯p∗
R −αLp∗
L −¯αL¯p∗
L = 0,
(71)
αR
γ −1p∗
R(u∗
R −¯u∗
R) −1
2αRρR¯u∗
R(u∗
R)2 + αRu∗
Rp∗
R + ¯αR¯u∗
R¯p∗
R
−
αL
γ −1p∗
L(u∗
L −¯u∗
L) + 1
2αRρR¯u∗
L(u∗
L)2 −αLu∗
Lp∗
L −¯αL¯u∗
L¯p∗
L = 0,
(72)
p∗
R −p∗
L = 0,
(73)
pL + 1
2CL(u∗
L −uL) −p∗
L = 0,
(74)
pR + 1
2CR(u∗
R −uR) −p∗
R = 0,
(75)
¯pL + 1
2
¯CL(¯u∗
L −¯uL) −¯p∗
L = 0,
(76)
¯pR + 1
2
¯CR(¯u∗
R −¯uR) −¯p∗
R = 0,
(77)
where we have assumed for simplicity that the gas phase satisﬁes the ideal EOS and the extension to general
EOS is straightforward.
We also note that equation (73) has been added to the system to impose the
Abgrall’s free-streaming condition [30].
As mentioned before, we use the intermediate states provided by the fully linearized Riemann solver
as guess values to start an iterative process for the solution of the thin-layer equations (70)–(77).
We
use Newton’s method to solve this nonlinear system with respect to the vector of unknown variables
(u∗
L, u∗
R, ¯u∗
L, ¯u∗
R, p∗
L, p∗
R, ¯p∗
L, ¯p∗
R).
The iterations are stopped when the relative error of the unknown vec-
tor in the 2-norm computed with respect to the previous iteration becomes less than ε = 10−6. It appears
that only 1-2 iterations are suﬃcient for convergence of Newton’s method for all the test cases considered
here.
5.1.3. Solution sampling and Godunov state
In order to construct a ﬁnite volume scheme for the governing hyperbolic system we need an approx-
imation of the intercell ﬂux. This can be done by deﬁning the Godunov state at the interface, locally at
x = 0.
The sampling of the solution of the Riemann problem for the pressure system is much simpler than
for the full Baer-Nunziato system, due to a particular conﬁguration of the characteristic waves; hence the
Godunov state at x = 0 and the corresponding numerical ﬂux can be computed very eﬃciently. For the
wave conﬁguration illustrated in Figs. 1,2 the sampling is done as follows:
11


• for the solid phase
¯αi+1/2 = ¯αL,
¯ui+1/2 = ¯u∗
L ≈¯u∗
0,
¯pi+1/2 = ¯p∗
L = ¯p0,
¯ρi+1/2 = ¯ρ∗
L = ¯ρL;
(78)
• for the gas phase
αi+1/2 = αL,
ui+1/2 = u∗
L ≈u∗
0,
pi+1/2 = p∗
L = p0,
ρi+1/2 =

ρ∗
L = ρL,
if u∗
0 > 0,
ρ0,
if u∗
0 < 0.
(79)
5.1.4. Numerical ﬂux for the conservative part
Having computed the intermediate states
(u∗
L, u∗
R, ¯u∗
L, ¯u∗
R, p∗
L, p∗
R, ¯p∗
L, ¯p∗
R),
we sample the solution to deﬁne the Godunov state
Wi+1/2 = (¯αi+1/2, ¯ρi+1/2, ¯ui+1/2, ¯pi+1/2, ρi+1/2, ui+1/2, pi+1/2)
and construct the conservative numerical ﬂuxes for the pressure and advection systems as follows:
Pi+1/2 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
¯αi+1/2¯pi+1/2
¯αi+1/2¯ui+1/2

¯ρi+1/2¯ei+1/2 + ¯pi+1/2
	
0
αi+1/2pi+1/2
αi+1/2ui+1/2

ρi+1/2ei+1/2 + pi+1/2
	
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
(80)
Ai+1/2 = ¯αi+1/2¯ui+1/2
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
¯ρ
¯ρ¯u
1
2 ¯ρ¯u2
0
0
0
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
n
k
+ αi+1/2ui+1/2
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎣
0
0
0
0
ρ
ρu
1
2ρu2
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎦
n
l
,
(81)
where we take
k =

i,
if ¯ui+1/2 >= 0,
i + 1,
if ¯ui+1/2 < 0,
and
l =

i,
if ui+1/2 >= 0,
i + 1,
if ui+1/2 < 0
The complete ﬂux for the conservative term is deﬁned as
Fi+1/2 = Ai+1/2 + Pi+1/2.
(82)
12


Finally, we use the following approximation of the non-conservative terms at the cell interface xi+1/2
proposed in [28]:
Ti+1/2 =
⎡
⎢⎢⎢⎢⎢⎢⎢⎢⎢⎣
¯ui+1/2(¯αi+1 −¯αi)
0
−(¯p∗
R,i+1/2 ¯αi+1 −¯p∗
L,i+1/2 ¯αi)
−¯ui+1/2(¯p∗
R,i+1/2¯αi+1 −¯p∗
L,i+1/2¯αi)
0
¯p∗
R,i+1/2 ¯αi+1 −¯p∗
L,i+1/2 ¯αi
¯ui+1/2(¯p∗
R,i+1/2 ¯αi+1 −¯p∗
L,i+1/2¯αi).
⎤
⎥⎥⎥⎥⎥⎥⎥⎥⎥⎦
(83)
5.1.5. Resulting ﬁnite volume scheme
The numerical ﬂux constructed in the previous section can be used directly in the ﬁnite volume scheme
which in 1D takes the form [28]
Qn+1
i
= Qn
i −Δtn
Δxi

H−
i+1/2 −H+
i−1/2

,
(84)
where H−
i+1/2 and H+
i+1/2 are deﬁned by
H−
i+1/2 =

Fi+1/2 + Ti+1/2,
if ¯ui+1/2 ⩽0,
Fi+1/2,
if ¯ui+1/2 > 0,
(85)
H+
i+1/2 =

Fi+1/2,
if ¯ui+1/2 ⩽0,
Fi+1/2 −Ti+1/2,
if ¯ui+1/2 > 0.
(86)
5.2. Roe-type method for the pressure system
We start the presentation of a Roe-type solver [31] for the non-conservative pressure system by writing
the non-conservative P-system
∂tQ + ∂xP(Q) + T(Q)∂x ¯α = 0
(87)
in the equivalent form
∂tQ + M(Q)∂xQ = 0,
(88)
Q(x, 0) =

QL, x < 0,
QR, x > 0
(89)
with
M = ∂P
∂Q + ˆT(Q)
and ˆT = [T, 0, . . . , 0].
5.2.1. Numerical Roe approach
We can locally linearize the system (88) and write the corresponding Riemann problem for a linear
hyperbolic system with constant coeﬃcient matrix ˆM(QL, QR):
∂tQ + ˆM(QL, QR)∂xQ = 0,
(90)
Q(x, 0) =

QL, x < 0,
QR, x > 0.
(91)
13


We note that deriving a Roe matrix strictly and ﬁnding the Roe averaged values for the equations considered
here is a formidable task, not yet attempted, to our knowledge, Even if this was possible, it is unclear to us
whether it would result in a useful scheme.
Instead, we follow [32, 33] and calculate the Roe matrix ˆM(QL, QR) numerically as
ˆM(QL, QR) =
 1
0
M

ϕ(s, QL, QR)
	
ds ≈

i
wiM

ϕ(si, QL, QR)
	
,
where ϕ is the canonical path
ϕ(s, QL, QR) = QL + s(QR −QL).
Then the averaged state ˆ
W can be deﬁned from equation
M( ˆ
W) = ˆM(QL, QR)
and used to evaluate
ˆrj = rj( ˆ
W),
where rj are right eigenvectors of M corresponding to the eigenvalues λj.
However, since the the derivation of the analytical expression for the averaged state ˆ
W proves to be
too complicated, we use an alternative approach and compute the matrix of averaged right eigenvectors ˆrj
directly:
ˆR(QL, QR) =
 1
0
R

ϕ(s, QL, QR)
	
ds ≈

i
wiR

ϕ(si, QL, QR)
	
.
The averaged left eigenvectors ˆlj and eigenvalues ˆλj are deﬁned in a similar way.
To evaluate the numerical ﬂux we need the Godunov state Q(0), which is a solution of the linearized
Riemann problem (90) at x/t = 0. The Godunov state is given by
Q(0) = QL +

ˆλj<0
αjˆrj,
with wave strengths
αj = ˆlj · (QR −QL).
A remarkable feature of the numerical Roe-type solver for the pressure system in the TV ﬂux splitting
framework is the fact that no entropy ﬁx is needed to ensure the correct recognition of the sonic points, in
contrast to the Roe approach applied to the full system. This is due to the conﬁguration of the characteristic
waves with SL < 0 < SR and ¯SL < 0 < ¯SR, so that the left and right non-linear waves cannot possibly
contain the sonic point x/t = 0. Therefore the numerical Roe-type solver applied to the P-system becomes
very much simpliﬁed compared to a potential Roe scheme, not currently available, following the original
Roe approach [31]. This allows to speed up the computation of the numerical ﬂux.
5.2.2. A path-conservative scheme
A ﬁrst-order path-conservative scheme is given by
Qn+1
i
= Qn
i −Δtn
Δxi

D+
i−1/2 + D−
i+1/2

,
(92)
where
D+
i−1/2 =
 1
0
M

ϕ+
i−1/2(s, Qn
i−1/2, Qn
i )
	∂ϕ+
i−1/2
∂s
ds −Ai−1/2,
D−
i+1/2 =
 1
0
M

ϕ−
i+1/2(s, Qn
i , Qn
i+1/2)
	∂ϕ−
i+1/2
∂s
ds + Ai+1/2,
14


using the canonical paths
ϕ+
i−1/2(s, Qn
i−1/2, Qn
i ) = Qn
i−1/2 + s(Qn
i −Qn
i−1/2),
ϕ−
i+1/2(s, Qn
i , Qn
i+1/2) = Qn
i + s(Qn
i+1/2 −Qn
i ) ,
with Qn
i±1/2 being the Godunov state at the corresponding cell interface.
5.3. Nonconservative HLL-PVM method for the pressure system
We shall consider the pressure system written in the nonconservative form (88), (89) and use the ﬁrst-
order path-conservative scheme (92) with ﬂuctuations D±
i±1/2 computed according to the HLL method for
nonconservative systems as described in [34].
The intermediate HLL state, denoted as Q∗, is given by
Q∗=
1
(SR −SL)

SRQR −SLQL −
 1
0
M

ϕ(s, QL, QR)
	∂ϕ
∂s ds

,
(93)
where SL and SR are speed estimations for the slowest and fastest characteristic waves of the P-system,
respectively. To calculate these estimates we use a simple formula
SL = min
1
2(uL −AL), 1
2(¯uL −¯
AL)

(94)
SR = max
1
2(uR + AR), 1
2(¯uR + ¯
AR)

,
(95)
that is, the characteristic speeds are simply evaluated at the left or right data.
Next, the HLL ﬂuctuations are computed according to the fast PVM-1U method proposed in [34] and
used in the ﬂux splitting context as follows:
D±
i+1/2 = 1
2

(1 ± α1) Mi+1/2

Qi+1 −Qi
	
± α0

Qi+1 −Qi
	
∓Ai+1/2,
where
α0 = SR|SL| −SL|SR|
SR −SL
,
α1 = |SR| −|SL|
SR −SL
.
The matrix Mi+1/2 satisﬁes
Mi+1/2

Qi+1 −Qi
	
=
 1
0
M

ϕ(s, Qi, Qi+1)
	∂ϕ
∂s ds
and if the canonical path ϕ = Qi + s(Qi+1 −Qi) is used, it is simply given by
Mi+1/2 =
 1
0
M

ϕ(s, Qi, Qi+1)
	
ds.
5.4. Nonconservative HLLEM method for the pressure system
Next, we describe a very recently proposed general scheme for conservative and nonconservative systems
[15], the HLLEM method. Here we apply the nonconservative HLLEM method to our P-system in our ﬂux
splitting framework. The extension from HLL to HLLEM method is achieved as follows [15]. First, the
intermediate state is modiﬁed according to
QHLLEM(ξ) = QHLL + ϕR∗( ¯Q)2δ∗( ¯Q)L∗( ¯Q)QR −QL
SR −SL

ξ −1
2(SL + SR)

,
SL < ξ < SR,
15


where QHLL is given by (93), and then the HLLEM ﬂuctuation is computed as
D±
HLLEM(QL, QR) = D±
HLL(QL, QR) ± ϕ SLSR
SR −SL
R∗( ¯Q)δ∗( ¯Q)L∗( ¯Q)(QR −QL).
The estimates for wavespeeds SL and SR are again computed according to (94), (95). The averaged state
¯Q and the matrices R∗( ¯Q), L∗( ¯Q) are deﬁned in [15]. Following the authors’ advise [15], the parameter
δ∗( ¯Q), called ﬂattener, is set to some value δ∗< 1 in order to improve the stability of the method. The
Godunov state needed for the approximation of the advection ﬂux is simply given by QHLLEM(0).
So far, we have described several schemes to deal with the P-system associated with the Baer-Nunziato
equations. In the next section we apply the methods to a carefully selected set of test problems.
6. Numerical results
In this section, we test the performance of the TV numerical ﬂux implemented in the ﬁrst-order ﬁnite
volume or path-conservative frameworks.
6.1. Test problems
We consider six Riemann problems as test problems, deﬁned in the domain [0, 1], and for which the
exact solutions are available. The initial data consists of two constant states separated by a discontinuity
at x = x0; all the gas and solid phase parameters as well as the initial data are listed in Tables 1–3.
Transmissive boundary conditions are imposed at x = 0 and x = 1. Results are displayed graphically,
showing comparisons between computed proﬁles and exact proﬁles, at speciﬁed ﬁxed output times. All
results presented in Figs. 3–17 were computed using the mesh of N = 200 cells in the one-dimensional
domain [0, 1] at Courant number coeﬃcient CCFL speciﬁed below for each solver. For the HLLEM solver we
set ϕ = 0.6.
We use the following estimation for the time step:
Δtn = CCFLΔx/Sn
max,
where CCFL is prescribed and the expression for Sn
max is given by
Sn
max = max
i {|un
i | + an
i , |¯un
i | + ¯an
i }, i = 1 . . . N,
and ai
i and ¯ai
i are the sound speeds of the gas and solid phase, respectively.
Table 1: EOS parameters and initial discontinuity position x0
Test 1
Test 2
Test 3
Test 4
Test 5
Test 6
¯γ
1.4
3.0
1.4
1.4
3.0
3.0
γ
1.4
1.35
1.4
1.4
1.4
1.4
¯P0
0.0
3400.0
0.0
0.0
10.0
100.0
x0
0.5
0.5
0.5
0.5
0.5
0.8
Numerical results for all six test cases are shown in Figs. 3–17. In these ﬁgures, the curve denoted as
”ExactSol” stands for the exact solution of the Riemann problem, the curve ”HLLC full” corresponds to
the ﬁnite-volume scheme with the HLLC-type Riemann solver for the Baer-Nunziato equations from [4], the
curve ”HLL full” is the numerical solution obtained by the nonconservative HLL-PVM method applied to the
complete unsplit Baer-Nunziato system according to [34], while other curves correspond to versions of the
TV ﬂux splitting scheme according to the Riemann solver used for the pressure system: ”HLL-TV” denotes
the HLL-PVM Riemann solver, ”HLLEM-TV” denotes the HLLEM solver, ”NumRoe-TV” corresponds to
the numerical Roe approach and, ﬁnally, ”LinRS-TV” illustrates the results from the linearized Riemann
solver applied to the pressure system. In practice, we have observed that the maximum CFL coeﬃcient
which guarantees stable results for all the test problems presented here depends on the Riemann solver used
for the P-system. Therefore for the linearized Riemann solver and full HLL solver we set CCFL = 0.9, for
HLL-PVM and HLLEM Riemann solvers CCFL = 0.8 and for the numerical Roe approach CCFL = 0.6.
16


Table 2: Initial data (solid phase)
Test
¯αL
¯ρL
¯uL
¯pL
¯αR
¯ρR
¯uR
¯pR
1
0.8
1.0
0.0
1.0
0.3
1.0
0.0
1.0
2
0.2
1900.0
0.0
10.0
0.9
1950.0
0.0
1000.0
3
0.8
1.0
0.75
1.0
0.3
0.125
0.0
0.1
4
0.8
1.0
−2.0
0.4
0.5
1.0
2.0
0.4
5
0.6
1.4
0.0
2.0
0.3
1.0
0.0
3.0
6
0.7
1.0
−19.5975
1000.0
0.2
1.0
−19.5975
0.01
Table 3: Initial data (gas phase)
Test
αL
ρL
uL
pL
αR
ρR
uR
pR
1
0.2
0.2
0.0
0.3
0.7
1.0
0.0
1.0
2
0.8
2.0
0.0
3.0
0.1
1.0
0.0
1.0
3
0.2
1.0
0.75
1.0
0.7
0.125
0.0
0.1
4
0.2
1.0
−2.0
0.4
0.5
1.0
2.0
0.4
5
0.4
1.4
0.0
1.0
0.7
1.0
0.0
1.0
6
0.3
1.0
−19.5975
1000.0
0.8
1.0
−19.5975
0.01
6.2. First order numerical results
Test 1: ideal EOS
Test 1 was presented in [28] and results are shown in Figs. 3 and 4 for the gas and solid phases, respectively.
The solid-phase wave pattern consists of a left rarefaction, a right shock wave and a right travelling solid
contact, while the gas phase consists of a left shock wave, two contacts and a right rarefaction wave. The
equations of state for both phases are assumed ideal, with ¯γ = γ = 1.4. This is a very mild test case
without any strong jumps in the parameters of the solution. Nonetheless it is still useful for assessing the
performance of the solvers. For example, we have observed that the TV ﬂux splitting method combined
with numerical Roe solver for the pressure system fails to resolve the intermediate solid velocity accurately
while all the other solvers provide a reasonably good approximation, see Fig. 4. As expected, a crucial issue
is the resolution of intermediate waves.
Test 2: stiﬀened EOS
Test 2 was also presented in [28] and results are shown in Figs. 5, 6. This test problem is more demanding
than Test 1 as it includes large variations of initial data and non-ideal EOS. For this problem, the linearized
Riemann solver and the full HLLC solver produce the most accurate solution, while the HLL-based solvers
are quite diﬀusive and the numerical Roe solver generates a visible undershoot in the gas density.
Test 3: sonic point
Test 3 was presented in [4] to test the resolution of the sonic point and results are shown in Figs. 7, 8.
The solution, for both phases, consists of a right shock wave, a right travelling contact discontinuity and
a left sonic rarefaction wave. The correct resolution of the sonic point is very important in assessing the
entropy satisfaction property of the numerical scheme. For this test problem all the solvers considered here
give similar results. It should be mentioned that there is no non-physical entropy glitch at the sonic point
generated by any of the solvers, not even by the Roe solver which has been used in the TV ﬂux splitting
framework without additional entropy ﬁx.
Test 4: 123-problem
Test 4 is an extension of the so-called 123-problem [4, 18] for two-phase ﬂows and results are shown in
Figs. 9, 10. Both solid and gas phases consist of a two symmetric rarefaction waves and a trivial stationary
contact wave. The region between the rarefaction waves is close to vacuum, therefore this test case is useful
17


0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.8
−0.7
−0.6
−0.5
−0.4
−0.3
−0.2
−0.1
0.0
0.1
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 3: Test 1. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15.
to assess the pressure positivity in diﬀerent numerical methods. The results of computations show that the
volume fraction is most accurately resolved by the linearized Riemann solver and full HLLC-type solver
while all the other solid and gas variables are similarly approximated by all the schemes, except for the
numerical Roe solver which fails to resolve the volume fraction accurately.
Test 5: stationary contact
Test 5 was designed in [4] to assess the ability of numerical methods to resolve the stationary isolated
contact waves; the results are shown in Fig. 11. The exact solution allows the existence of the stationary
contact waves in the solid and gaseous phases when the volume fraction and solid pressure gradients are
present across the solid contact. The solution of this test problem contains isolated contacts in both solid
and gas phases. As it can be seen from ﬁgures below this type of discontinuities is resolved exactly by
linearized, HLLC and Roe solvers, which is expected as these Riemann solvers are complete, that is, they
take into account all the intermediate characteristic waves, unlike the HLL and HLLEM Riemann solvers
which smear out the contact discontinuity.
18


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.92
0.94
0.96
0.98
1.00
1.02
1.04
1.06
1.08
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.01
0.00
0.01
0.02
0.03
0.04
0.05
0.06
0.07
0.08
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.90
0.95
1.00
1.05
1.10
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 4: Test 1. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15.
Test 6: strong shock
Test 6 is a strong-shock test problem from [4], which was designed to assess the robustness and accuracy
of numerical methods. The results are presented in Figs. 12 and 13. The solution of this problem contains,
for each phase, a right travelling shock wave, a contact discontinuity and a left rarefaction wave. As the
jump of initial pressures is very large, strong shock waves are generated in each phase; the distance between
the right shock and contact waves is small in the gas phase. These ﬂow features can lead to inaccuracies
in numerical solution. The computations show that, for this test problem, the HLLC-type solver generates
the most accurate solution among the schemes considered here, followed closely by linearized and numerical
Roe Riemann solvers.
6.3. Veriﬁcation of free-streaming solution
Here we test the TV ﬂux splitting approach for the Baer-Nunziato equations by verifying Abgrall’s
criterion proposed in [30]: if ui = ¯ui = V = const as well as pi = ¯pi = P = const for all i at time t = 0, then
the velocities and the pressures will keep the same values for t > 0. Such solution is called free-streaming
solution and its correct resolution is crucial to prevent pressure oscillations. It can be shown analytically
that for all the pressure system solvers of this paper used in TV ﬂux splitting approach the free-streaming
19


0.0
0.2
0.4
0.6
0.8
1.0
x
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
1.0
1.2
1.4
1.6
1.8
2.0
2.2
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.2
0.0
0.2
0.4
0.6
0.8
1.0
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
1.0
1.5
2.0
2.5
3.0
3.5
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 5: Test 2. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15.
solution is satisﬁed. If un
i = ¯un
i = V = const as well as pn
i = ¯pn
i = P = const, then un+1
i
= ¯un+1
i
= V as well
as pn+1
i
= ¯pn+1
i
= P, and the volume fractions and densities evolve according to linear advection equations
with characteristic velocity V :
¯αn+1
i
= ¯αn
i −V Δt
Δx (¯αn
i −¯αn
i−1),
¯αn+1
i
¯ρn+1
i
= ¯αn
i ¯ρn
i −V Δt
Δx (¯αn
i ¯ρn
i −¯αn
i−1¯ρn
i−1),
αn+1
i
ρn+1
i
= αn
i ρn
i −V Δt
Δx (αn
i ρn
i −αn
i−1ρn
i−1).
We also verify the free-streaming criterion numerically by solving the Riemann problem with the following
initial data: u = ¯u = 1.0, p = ¯p = 1.0, (¯αL, ¯ρL, ρL) = (0.8, 0.5, 0.2) and (¯αR, ¯ρR, ρR) = (0.3, 1.0, 1.0).
Figs. 14, 15 show the corresponding numerical solution for all the solvers considered in this paper.
20


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
1800
1850
1900
1950
2000
2050
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.18
−0.16
−0.14
−0.12
−0.10
−0.08
−0.06
−0.04
−0.02
0.00
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0
200
400
600
800
1000
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 6: Test 2. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15.
6.4. Second-order path-conservative scheme
An explicit second-order TVD path-conservative scheme [15] reads
Qn+1
i
= Qn
i −Δt
Δx

D+
i−1/2 + D−
i+1/2

−Δt
ΔxM(Qn+1/2
i
)ΔQn
i ,
(96)
where the numerical ﬂuctuations are evaluated at the boundary-extrapolated solution:
D+
i−1/2 = D+
i−1/2

Qn+1/2,−
i−1/2
, Qn+1/2,+
i−1/2
	
,
D−
i+1/2 = D−
i+1/2

Qn+1/2,−
i+1/2
, Qn+1/2,+
i+1/2
	
.
The slope ΔQn
i is deﬁned as
ΔQn
i = minmod

Qn
i+1 −Qn
i , Qn
i −Qn
i−1
	
21


0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 7: Test 3. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15.
and the boundary-extrapolated solution values at time tn are
Qn,−
i−1/2 = Qn
i−1 + 1
2ΔQn
i−1,
Qn,+
i−1/2 = Qn
i −1
2ΔQn
i ,
Qn,−
i+1/2 = Qn
i + 1
2ΔQn
i ,
Qn,+
i+1/2 = Qn
i+1 −1
2ΔQn
i+1.
The solution at half time-step is obtained as follows:
Qn+1/2
i
= Qn
i + 1
2Δt∂tQn
i
with the derivative calculated as
∂tQn
i = −M(Qn
i )ΔQn
i
Δx −1
Δx

A

Qn,−
i+1/2
	
−A

Qn,+
i−1/2
	
.
22


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 8: Test 3. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15.
The boundary-extrapolated solution values at tn+1/2 are given by
Qn+1/2,±
i−1/2
= Qn,±
i−1/2 + 1
2Δt∂tQn
i
Qn+1/2,±
i+1/2
= Qn,±
i+1/2 + 1
2Δt∂tQn
i .
Note that if the conservative and non-conservative ﬂuxes are approximated separately, e.g. as (82), (83)
in the case of the linearized Riemann solver, the second-order TVD path-conservative scheme (96) reduces
to a TVD ﬁnite-volume second order scheme [18].
Here we repeat one of the numerical tests, namely Test 2, to demonstrate the performance of diﬀerent
Riemann solvers used in the second-order scheme described above, the results are shown in Fig. 16, 17. As
in the case of the ﬁrst-order scheme, the computations show that the linearized Riemann solver provides,
overall, the most accurate numerical solution among all the solvers considered in this paper.
In addition, in Fig. 18 we empirically illustrate the convergence trend of the numerical solution in the
zoomed region near the solid contact wave. The numerical solution has been computed on a sequence of
meshes using the the second-order TV ﬂux splitting scheme with linearized Riemann solver for the pressure
23


0.0
0.2
0.4
0.6
0.8
1.0
x
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.0
0.2
0.4
0.6
0.8
1.0
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−2.0
−1.5
−1.0
−0.5
0.0
0.5
1.0
1.5
2.0
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 9: Test 4. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15.
system.
6.5. Eﬃciency
For the eﬃciency study of the proposed ﬂux splitting method with various solvers for the P-system we
performed computations for Test 2, on a sequence of meshes using ﬁrst order schemes formulated in Section 5
as well as the ﬁnite-volume scheme with HLL and HLLC Riemann solvers for the full Baer-Nunziato system.
Fig. 19 is an eﬃciency plot, namely an Error versus CPU time plot. Five curves (straight lines) are plotted,
corresponding to ﬁve numerical methods for the P-system. Each curve displays six points of the form (Error,
CPU time), corresponding to six meshes. Errors were computed in the L1-norm for the variable ¯ρ.
To interpret the results we proceed as follows. First we choose a small error; this value determines a
horizontal line in the direction of the CPU time axis. This line will intersect the various curves corresponding
to the considered numerical schemes. The associated intersection points give the CPU time required for
each scheme to attain the chosen error. The point of the exercise is this: given an error, indicate the scheme
that attains that error at the smallest computational time.
In Fig. 19 we choose Error = 0.002. We see that the TV ﬂux splitting method with the linearized
Riemann solver for the P-system is the most eﬃcient; it takes only 1.88 sec of CPU time to attain the
24


0.0
0.2
0.4
0.6
0.8
1.0
x
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.0
0.2
0.4
0.6
0.8
1.0
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
−2.0
−1.5
−1.0
−0.5
0.0
0.5
1.0
1.5
2.0
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 10: Test 4. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15.
chosen error. This solver is slightly more eﬃcient than the HLLC scheme of [4] which takes 2.37 sec to
reach the indicated error; moreover, the implementation of the HLLC Riemann solver for the Baer-Nunziato
equations and the solution sampling for the numerical ﬂux computation are more complicated than in the
linearized Riemann solver for the P-system. The next most eﬃcient method is the HLL scheme applied
directly to the full Baer-Nunziato system, no ﬂux vector splitting; its is however 15.7 times more expensive
than the present TV splitting with the linearized Riemann solver for the P-system. Very close to the HLL-
full is the TV splitting with the HLL ﬂux for the P-system, followed by the TV splitting with numerical Roe
ﬂux for the P-system. The most ineﬃcient scheme turns out to be the TV splitting with HLLEM scheme.
However, it should be noted that these last two schemes are rather general and can easily be implemented
to solve more general hyperbolic systems than the one considered in this paper.
25


0.0
0.2
0.4
0.6
0.8
1.0
x
0.40
0.45
0.50
0.55
0.60
0.65
0.70
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.9
1.0
1.1
1.2
1.3
1.4
1.5
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
2.0
2.2
2.4
2.6
2.8
3.0
3.2
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
1.00
1.05
1.10
1.15
1.20
1.25
1.30
1.35
1.40
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid pressure
solid density
Figure 11: Test 5. Results: computed (symbol) and exact solution (line) at t = 0.15.
26


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0
1
2
3
4
5
6
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−20
−15
−10
−5
0
5
10
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
−200
0
200
400
600
800
1000
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 12: Test 6. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.007.
27


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.8
1.0
1.2
1.4
1.6
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
−20
−18
−16
−14
−12
−10
−8
−6
−4
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0
200
400
600
800
1000
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 13: Test 6. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.007.
28


0.0
0.2
0.4
0.6
0.8
1.0
x
0.1
0.2
0.3
0.4
0.5
0.6
0.7
α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
0.94
0.96
0.98
1.00
1.02
1.04
1.06
v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.94
0.96
0.98
1.00
1.02
1.04
1.06
p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
gas velocity
gas pressure
Figure 14: Free-streaming solution. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15.
29


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
¯α
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.4
0.5
0.6
0.7
0.8
0.9
1.0
¯ρ
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
0.94
0.96
0.98
1.00
1.02
1.04
1.06
¯v
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
0.0
0.2
0.4
0.6
0.8
1.0
x
0.94
0.96
0.98
1.00
1.02
1.04
1.06
¯p
ExactSol
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
solid velocity
solid pressure
Figure 15: Free-streaming solution. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15.
30


0.0
0.2
0.4
0.6
0.8
1.0
x
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
α
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
0.0
0.2
0.4
0.6
0.8
1.0
x
1.0
1.2
1.4
1.6
1.8
2.0
2.2
ρ
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
gas volume fraction
gas density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
v
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
0.0
0.2
0.4
0.6
0.8
1.0
x
1.0
1.5
2.0
2.5
3.0
3.5
p
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
gas velocity
gas pressure
Figure 16: Test 2. Results for the gas phase: computed (symbol) and exact solution (line) at t = 0.15. Second order scheme.
31


0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
¯α
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
0.0
0.2
0.4
0.6
0.8
1.0
x
1800
1850
1900
1950
2000
2050
¯ρ
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
solid volume fraction
solid density
0.0
0.2
0.4
0.6
0.8
1.0
x
−0.18
−0.16
−0.14
−0.12
−0.10
−0.08
−0.06
−0.04
−0.02
0.00
¯v
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
0.0
0.2
0.4
0.6
0.8
1.0
x
0
200
400
600
800
1000
¯p
ExactSol
HLL-TV-o2
HLLEM-TV-o2
NumRoe-TV-o2
LinRS-TV-o2
solid velocity
solid pressure
Figure 17: Test 2. Results for the solid phase: computed (symbol) and exact solution (line) at t = 0.15. Second order scheme.
32


0.40
0.42
0.44
0.46
0.48
0.50
0.52
0.54
x
1800
1850
1900
1950
2000
2050
¯ρ
ExactSol
100 cells
200 cells
400 cells
800 cells
1600 cells
3200 cells
Figure 18: Convergence of the linearized Riemann solver at the solid contact: numerical solution for a sequence of 5 meshes
for Test 2 using second-order scheme.
10−1
100
101
102
103
CPU time
10−3
10−2
L1-error
Error = 0.002
2.37 [s]
Error = 0.002
29.51 [s]
Error = 0.002
45.22 [s]
Error = 0.002
235.37 [s]
Error = 0.002
125.85 [s]
Error = 0.002
1.88 [s]
HLLC full
HLL full
HLL-TV
HLLEM-TV
NumRoe-TV
LinRS-TV
Figure 19: Eﬃciency plot for Test 2: error vs CPU time for six numerical schemes denoted as HLLC full, HLL-TV, HLLEM-TV,
NumRoe-TV and LinRS-TV. For each method, a sequence of meshes has been used, giving rise to the points on the Error/CPU
plane; a curve is then plotted through these points, resulting in six curves. The eﬃciency analysis consists of choosing a ﬁxed
error and searching for the intersection between the corresponding horizontal line with the six curves of the schemes studied.
The actual CPU times are obtained by interpolation. The most eﬃcient method is the TV splitting with the linearized Riemann
solver for the pressure system.
33


7. Conclusions
We have extended the TV ﬂux splitting approach to solve the Baer-Nunziato equations for compressible
two-phase ﬂow.
For the present we have constructed several Riemann solvers for the pressure system,
in order to obtain an approximation of the pressure ﬂux. These solvers and ﬂuxes have been tested as
building blocks for the ﬁnite volume and path-conservative schemes of ﬁrst and second order of accuracy for
a range of one-dimensional test problems. Numerical solutions have been compared with exact solutions.
Moreover, a computational eﬃciency analysis has been performed; this indicates that the present TV scheme
in conjunction with the linearized Riemann solver for the pressure system is the most eﬃcient scheme. The
closest competitor is eight times more ineﬃcient.
The linearized Riemann solver for the P-system uses a simple linearization of the Riemann invariants for
the nonlinear characteristic ﬁelds, except for the solid contact, where one still has to consider the thin-layer
equations for the pressure system. However, this nonlinear system is, in a certain sense, simpler than the
original thin-layer equations for the full Baer-Nunziato system; we removed part of the nonlinearity due to
the convective terms. Therefore the solver results in the solution of the nonlinear system of equations across
the solid contact combined with simpliﬁed linearized relations for the left and right waves of each phase.
The quality of the numerical solution provided by this solver is very satisfactory, taking into account the
savings of the CPU time due to the linearization and the absence of sophisticated wave speed estimates e.g.
as in [4]. We did not come across any problems with the convergence of our iterative solver for the nonlinear
thin-layer equations. Moreover, upgrading to second-order accuracy greatly improves the solution quality, as
one would have expected. Moreover, the schemes described may constitute the building block for schemes of
even higher order of accuracy through by now standard high-order ﬁnite volume and discontinuous Galerkin
approaches, in one and multiple space dimensions.
The HLL-TV method turns out to be very diﬀusive in a ﬁrst-order method, though the situation improves
as the order of accuracy of the scheme is increased. However, in terms of eﬃciency this solver turned out to
be almost identical to the nonconservative HLL solver for the full Baer-Nunziato system. The HLLEM-TV
Riemann solver for the pressure system can be viewed as an improvement upon HLL-TV solver as it provides
much better solution quality on a given grid. However, in terms of computational eﬃciency, it turned out
to be very expensive, around 125 times more expensive than the linearized Riemann solver; this is partly
explained by the need to evaluate some of the eigenvectors of the pressure system. The performance of
the numerical Roe solver for the pressure system was comparable to that of the linearized Riemann solver
in terms of solution quality on a given grid in most of the test cases. However, as already pointed out,
the numerical Roe solver in the TV framework is signiﬁcantly more expensive than the linearized Riemann
solver, by a factor of 67.
The attraction of the TV splitting is the simplicity with which one can construct the numerical ﬂux for
the full scheme. Such simplicity is two fold, ﬁrst the work is centred on a reduced system, namely pressure
system; then for such system one can devise very simple numerical schemes for the associated pressure
numerical ﬂux. The end result is a very simple and eﬃcient scheme for the full system, without sacriﬁcing
robustness and accuracy. As soon as one devises complicated and expensive methods for the pressure system,
the attraction of the Toro-V´azquez ﬂux vector splitting approach (TV) is lost, as demonstrated in this paper.
Further work would include the generalization of the linearized pressure system solver to the multidi-
mensional case, increase the order of accuracy above two, on unstructured meshes, and application of the
resulting schemes to realistic multiphase ﬂow problems involving more advanced mathematical models.
References
[1] E. F. Toro, M. E. V´azquez-Cend´on, Flux splitting schemes for the Euler equations, Computers & Fluids 70 (2012) 1–12.
[2] M. R. Baer, J. W. Nunziato, A two-phase mixture theory for the deﬂagration-to-detonation transition (DDT) in reactive
granular materials, J. Multiphase Flow 12 (1986) 861–889.
[3] P. Embid, M. Baer, Mathematical analysis of a two-phase continuum mixture theory, Continuum Mech. Thermodyn. 4
(1992) 279–312.
[4] S. A. Tokareva, E. F. Toro, HLLC-type Riemann solver for the Baer-Nunziato equations of compressible two-phase ﬂow,
J. Comput. Phys. 229 (2010) 3573–3604.
34


[5] E. F. Toro, Riemann-problem based techniques for computing reactive two-phase ﬂows, Lecture Notes in Physics 351
(1989) 472–481.
[6] I. Toumi, An upwind numerical method for two-ﬂuid two-phase ﬂow models, Nuclear Sci. Eng. 123 (1996) 147–168.
[7] R. Saurel, R. Abgrall, A simple method for compressible multiﬂuid ﬂows, SIAM J. Sci. Comput. 21 (1999) 1115–1145.
[8] H. Pailliere, A. Kumbaro, D. Bestion, S. Mimouni, A. Laporta, H. Staedtke, G. Franchello, U. Graf, P. Romstedt, E. F.
Toro, E. Romenski, H. Deconinck, E. Valero, F. de Cachard, B. Smith, Advanced three-dimensional two-phase ﬂow
simulation tools for application to reactor safety (ASTAR), Nuclear Engineering and Design 235 (2005) 379–400.
[9] C. H. Chang, M. S. Liou, A robust and accurate approach to computing compressible multiphase ﬂow: stratiﬁed ﬂow
model and AUSM+ up scheme, J. Comput. Phys. 225 (2007) 840–873.
[10] R. Saurel, O. L. M´etayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiﬀmechanical
relaxation, Shock Waves 16 (2007) 209–232.
[11] M. Dumbser, A. Hidalgo, M. Castro, C. Par´es, E. Toro, FORCE schemes on unstructured meshes ii: Nonconservative
hyperbolic systems, Comp. Methods Appl. Mech. Eng. 199 (2010) 625–647.
[12] F. Crouzet, F. Daude, P. Galon, P. Helluy, J.-M. H´erard, O. Hurisse, Y. Liu, Approximate solutions of the Baer-Nunziato
model, ESAIM Proceedings 40 (2013) 63–82.
[13] F. Coquel, J.-M. H´erard, K. Saleh, A splitting method for the isentropic Baer-Nunziato two-phase ﬂow model, ESAIM
Proceedings 38 (2013) 241–256.
[14] F. Coquel, J.-M. H´erard, K. Saleh, N. Seguin, A robust entropy-satisfying ﬁnite volume scheme the isentropic Baer-
Nunziato model, Math. Mod. Numer. Anal. 48 (2014) 165–206.
[15] M. Dumbser, D. S. Balsara, A new eﬃcient formulation of the HLLEM Riemann solver for general conservative and
non-conservative hyperbolic systems, J. Comput. Phys. 304 (2016) 275–319.
[16] F. Daude, P. Galon, On the computation of the Baer-Nunziato model using ALE formulation with HLL- and HLLC-type
solvers towards ﬂuid–structure interactions, J. Comput. Phys. 304 (2016) 189–230.
[17] S. K. Godunov, A ﬁnite diﬀerence method for the computation of discontinuous solutions of the equations of ﬂuid dynamics,
Mat. Sb. 47 (1959) 357–393.
[18] E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, Springer-Verlag, 2009.
[19] J. L. Steger, R. F. Warming, Flux vector splitting of the inviscid gasdynamic equations with applications to ﬁnite diﬀerence
methods, J. Comput. Phys. 40 (1981) 263–293.
[20] B. van Leer, Flux-vector splitting for the Euler equations, Technical report ICASE 82-30, NASA Langley Research Center,
USA.
[21] G.-C. Zha, E. Bilgen, Numerical solution of Euler equations by a new ﬂux vector splitting scheme, Int. J. Numer. Methods
Fluids 17 (1993) 115–144.
[22] M. S. Liou, C. J. Steﬀen, A new ﬂux splitting scheme, J. Comput. Phys. 107 (1993) 23–39.
[23] M. S. Liou, A sequel to AUSM: AUSM+, J. Comput. Phys. 129 (1996) 364–382.
[24] M. S. Liou, A sequel to AUSM, part ii: AUSM+-up for all speeds, J. Comput. Phys. 214 (2006) 137–170.
[25] H. Paillere, C. Corre, J. R. Garcia-Cascales, On the extension of the AUSM+ scheme to compressible two-ﬂuid models,
Computers & Fluids 32 (2003) 891–916.
[26] E. F. Toro, C. E. Castro, B. J. Lee, A novel numerical ﬂux for the 3D Euler equations with general equation of state,
J. Comput. Phys. 303 (2015) 80–94.
[27] D. S. Balsara, G. I. Montecinos, E. F. Toro, Exploring various ﬂux vector splittings for the magnetohydrodynamic system,
J. Comput. Phys. 311 (2016) 1–21.
[28] D. W. Schwendeman, C. W. Wahle, A. K. Kapila, The Riemann problem and a high-resolution Godunov method for a
model of compressible two-phase ﬂow, J. Comput. Phys. 212 (2006) 490–526.
[29] F. Coquel, J.-M. H´erard, K. Saleh, N. Seguin, A positive and entropy-satisfying ﬁnite volume scheme for the Baer-Nunziato
model, https://hal.archives-ouvertes.fr/hal-01261458 (2016).
[30] R. Abgrall, How to prevent pressure oscillations in mulicomponent ﬂow calculations: a quasi conservative approach,
J. Comput. Phys. 125 (1996) 150–160.
[31] P. L. Roe, Approximate Riemann solvers, parameter vectors, and diﬀerence schemes, J. Comput. Phys. 43 (1981) 357–372.
[32] C. E. Castro, E. F. Toro, Roe-type Riemann solvers for general hyperbolic systems, Int. J. Numer. Meth. Fluids 75 (2014)
467–486.
[33] M. Dumbser, E. F. Toro, A simple extension of the Osher Riemann solver to non-conservative hyperbolic systems,
J. Sci. Comput. 48 (2011) 70–88.
[34] M. J. Castro D´ıaz, E. D. Fern´andez-Nieto, A class of computationally fast ﬁrst order ﬁnite volume solvers: PVM methods,
SIAM J. Sci. Comput. 34 (2012) A2173–A2196.
35
