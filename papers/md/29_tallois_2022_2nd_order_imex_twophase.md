A second-order extension of a robust implicit–explicit acoustic-transport splitting
scheme for two-phase ﬂows
Lucas Talloisa,c,∗, Simon Peluchona, Philippe Villedieuc,b
aCEA-CESTA, 15 avenue des sablières - CS 60001, 33116 Le Barp Cedex, France.
bONERA - The French Aerospace Lab, Toulouse F-31055, France.
cINSA, 135 Avenue de Rangueil, 31400 Toulouse, France.
Abstract
Diﬀuse interface methods have proven their ability to simulate complex two-phase ﬂows. A number of robust numerical
schemes have been developed to simulate such ﬂows involving large density and pressure ratios. Diﬀusion induced by
these methods, however, makes it diﬃcult to localize the interface between the two ﬂuids. To overcome this issue, while
retaining the advantages of diﬀuse interface methods, a second-order extension using a MUSCL-type method of the
implicit–explicit acoustic-transport splitting scheme introduced in [35] is presented. A speciﬁc compressive limiter is
used for the volume fraction in order to limit the diﬀusion of the interface between the two ﬂuids.
Numerical simulations are presented to illustrate the capability of the proposed new method to simulate highly
complex compressible two-phase ﬂows.
Keywords:
Two-phase ﬂows, MUSCL, Unstructured meshes, Implicit, Compressive limiter
1. Introduction
This work takes place in the context of liquid ablation.
When an object enters the atmosphere at very high speed
(see Fig. 1), it is subjected to signiﬁcant heat ﬂux which
may degrade its structure. Depending on its composition,
the object may sublimate or liquefy. Thus, when a liquid
phase appears, it is necessary to deal with the ﬂow of both
the liquid and gas phases by an appropriate model.
liquid
solid
hypersonic
gas ﬂow
viscous boundary layer
bow shock
Figure 1: Schematic representation of an object entering the atmo-
sphere.
The modelling of two-phase ﬂows has been the subject
of numerous studies.
Several methods have been devel-
oped, each with advantages and disadvantages. They can
∗Corresponding author.
Email addresses: lucas.tallois@cea.fr (Lucas Tallois),
simon.peluchon@cea.fr (Simon Peluchon),
philippe.villedieu@onera.fr (Philippe Villedieu)
be grouped into two distinct families: the sharp interface
methods and the diﬀuse interface methods.
Among these methods, the ﬁrst one is the so-called
Lagrangian method [47]. Each phase can be modeled with
diﬀerent equations, and the interface is followed explicitly
by moving the mesh at the material speed. In the so-called
ALE methods (Arbitrary Lagrangian-Eulerian methods),
the mesh follows the displacement of the interface. The
mesh can, however, be subject to strong distortions that
can impact the robustness of the calculations.
Moreover, among the Eulerian methods with no or al-
most no diﬀusion, there are the Front capturing meth-
ods as Volume Of Fluid (VOF) [24] and Moment Of Fluid
(MOF) [1] methods, the Front tracking methods [7] and
the Level Set method [34]. Although widely used for their
satisfactory results, these methods are complex to imple-
ment and some may be not conservative, which can be a
serious drawback depending on the targeted applications.
Diﬀuse interface methods, on the other hand, are based
on two-phase models, i.e. models containing conservation
equations for each of the two phases that are artiﬁcially as-
sumed to be present at any point in space. The equations
are solved on a ﬁxed mesh in a Eulerian manner. The same
equations are solved over the whole domain. These meth-
ods allow the diﬀusion of the interface between each ﬂuid.
Thus, mixing zones appear, corresponding to a numerical
spreading of the interface over a few cells.
These areas
require special processing to preserve the thermodynamic
consistency of the model. Many studies have been carried
out with the aim of developing two-phase models, following
the work of Baer & Nunziato [3] and their seven-equation
Preprint submitted to Elsevier
February 7, 2022
© 2022 published by Elsevier. This manuscript is made available under the Elsevier user license
https://www.elsevier.com/open-access/userlicense/1.0/
Version of Record: https://www.sciencedirect.com/science/article/pii/S0045793022001633
Manuscript_ca84c3f0b0e0e1e63a23cb5ce2b56137


model.
Indeed, there is a whole hierarchy of two-phase
models deduced from the seven-equation model [17, 29].
A local velocity equilibrium between the two ﬂuids [40]
allows us to reduce to a six-equation model. If the equilib-
rium of the pressures is assumed, the ﬁve-equation model
of Kapila et al. [26] or the reduced ﬁve-equation model of
[2, 31, 32] can be obtained.
Another approach to simulate two-phase ﬂow consists
in writing the same set of equations for each ﬂuid, and to
close the model by deﬁning the mixing quantities as the
quantities of each ﬂuid weighted by its volume fraction [11,
21, 5]. An isobaric closure assumption and the resolution
of a 2 × 2 system allows the volume fraction in the mixing
region to be determined.
In this work, a diﬀuse interface method is considered.
The advantages of such method are numerous: the same
equations are solved over the whole domain, interfaces do
not require any speciﬁc processing and appearance of new
interfaces as well as the changes of topology are achieved
naturally. In the case where one wishes to model the two-
phase ﬂow between a gaseous phase and a liquid phase,
where the second phase is not present initially and appears
from solid fusion, these methods seem the most relevant
and the easiest to implement. The counterpart of these
methods is the diﬀusion of the interface between the two
ﬂuids, which can deteriorate the solution when the ﬂu-
ids are immiscible. Since we do not consider that ﬂuids
slide relative to each other at the mesh scale, a two-phase
model with one velocity and one pressure is used. As the
two phases are not initially mixed, the non-conservative
term in the Kapila et al. model [26] is null. The reduced
ﬁve-equation model of [2, 31] will therefore be used in this
work. Several methods have been developed in order to
limit this numerical diﬀusion of the interface. Kokh and
Lagoutière used the anti-diﬀusive scheme [27] developed
in [16], which consists in using a ﬂux as downwind as pos-
sible, while guaranteeing the stability and consistency of
the scheme. This method limits the diﬀusion of the in-
terface to 2 mixture cells. Jung et al. [23] combined the
Glimm’s random projection method near the material in-
terface with the upwind scheme to simulate two-phase ﬂow
without any diﬀusion cells. They extended this method to
the second order in space and time. Extension to higher
dimension is achieved with directional splitting.
Shyue
and Xiao [42] adapted the THINC method of [48] as a
sharpening technique to simulate two-phase ﬂows.
The
main idea is to reconstruct the solution with the hyperbolic
tangent function, controlling the compressive character of
the reconstruction with a β-parameter.
This is widely
combined with the Boundary Variation Diminishing algo-
rithm [43, 15, 12] to simulate two-phase ﬂows on struc-
tured and unstructured grids. In the context of MUSCL-
type schemes [28], where ﬂux are computed with piecewise
linear reconstruction with slope limitation, slope limiters
can be designed to have compressive properties and give a
sharp interface when applied to the volume fraction trans-
port equation. This approach has been widely used in the
literature as for instance in Qian et al. [37], Blanchard [5],
Chiapolino et al. [13] and De Vuyst et al. [14].
This paper is an extension of the work presented in [35]
where the splitting strategy of [8] is used to solve the ﬁve-
equation system of [31, 2]. The strategy consists in solving
the acoustic part with an implicit scheme while the trans-
port step is solved explicitly. The eﬀects related to heat
dissipation and viscosity can be easily added to the ﬁrst
step [36], which justiﬁes its implicit treatment. We will fo-
cus here only on the hyperbolic part of the equations. The
aim is to increase the accuracy of the ﬁrst-order method
described in [35] and to improve the resolution of the mate-
rial interface. A MUSCL-type scheme on an unstructured
grid is used, in order to improve the accuracy of the numer-
ical scheme. A strategy to improve the resolution of the
two-phase interface on an unstructured grid is employed.
Indeed, the method must be able to handle a body-ﬁtted
mesh composed of quadrilaterals and deformed by the ab-
lation. In the context of the splitting strategy, a new and
more accurate implicit-explicit time-scheme is presented.
The paper is organized as follows. First, the governing
equations used to simulate two-phase ﬂow are introduced
and the ﬁrst order numerical scheme is brieﬂy recalled in
Section 2. Then, a second-order in space extension and
improvement of the time scheme are derived in Section 3.
Some numerical results are presented in Section 4.
2. Model and numerical scheme
2.1. The ﬁve-equation system
We denote by ρk, εk and pk the density, internal energy
and pressure of ﬂuid k = 1, 2. Each ﬂuid is equipped with
an Equation Of State (EOS) of the form pk = pk(ρk, εk).
The volume fraction zk, such that z1 + z2 = 1, allows the
position of the interface to be located. In the sequel, we
denote by z = z1 the volume fraction of the ﬁrst ﬂuid. The
mixture density and internal energy are given by
ρ = ρ1z + ρ2(1 −z),
ρε = ρ1ε1z + ρ2ε2(1 −z).
(2.1)
Both ﬂuids share the same velocity u and the same pres-
sure p. The ﬁve-equation system of [2, 31] for non-miscible
ﬂuids with isobaric closure reads



















∂t(ρ1z)
+
∇· (ρ1zu)
=
0,
∂t(ρ2(1 −z)) +
∇· (ρ2(1 −z)u)
=
0,
∂t(ρu)
+
∇· (ρu ⊗u) + ∇p =
0,
∂t(ρE)
+
∇· (ρEu + pu)
=
0,
∂tz
+
u · ∇z
=
0,
(2.2)
where E = ε + ||u||2
2
is the total energy of the two-ﬂuid
media.
2


Each ﬂuid is considered to be governed by a stiﬀened
gas EOS, since we consider ﬂows with a liquid phase and
a gas phase. The EOS of phase k thus reads
pk = ρkεk(γk −1) −γkπk,
(2.3)
where γk > 1 is the adiabatic exponent and πk ≥0 is a
reference pressure. Note that perfect gas EOS is obtained
with πk = 0. The mixture pressure p is the solution of the
system
(
p1(ρ1, ε1) = p2(ρ2, ε2),
ρε = ρ1ε1z + ρ2ε2(1 −z).
(2.4)
In the case of two stiﬀened gases, (2.4) can be solved ex-
plicitly and we get the following expression for the mixture
pressure
p = ρε(γ −1) −γπ,
(2.5)
where the adiabatic exponent mixture γ and the reference
pressure π are given by
γ = 1 +
1
2P
k=1
zk
γk−1
and π = γ −1
γ
2
X
k=1
zkγkπk
γk −1 .
(2.6)
The sound velocity ck of each phase is deﬁned by c2
k =
∂pk
∂ρk |sk, where sk is the entropy. The mixture sound speed
for two stiﬀened gas reads
c2 = γ p + π
ρ
.
(2.7)
Note before going further that the system (2.2) can
be written, introducing the mass fraction of the ﬁrst ﬂuid
y = ρ1z
ρ , in the following form



















∂tρ
+
∇· (ρu)
=
0,
∂t(ρy) +
∇· (ρyu)
=
0,
∂t(ρu) +
∇· (ρu ⊗u) + ∇p =
0,
∂t(ρE) +
∇· (ρEu + pu)
=
0,
∂tz
+
u · ∇z
=
0.
(2.8)
The evolution equation of ρ is obtained by summing the
evolution equation of ρ1z1 and ρ2z2 of system (2.2). The
evolution equation of ρy is nothing but the evolution equa-
tion of ρ1z1 by deﬁnition. This form allows us to write in
the Lagrangian form the acoustic step, described below.
Remark 1
If the volume fraction only takes the values 0 and 1, then
(2.8) is equivalent to the 4-equation model i.e (2.8) without
the equation on ρy.
2.2. Splitting strategy
We will brieﬂy recall the method of solving the ﬁve-
equation system, where a detailed approach is described
in [35]. The expansion of system (2.8) reads



















∂tρ
+
ρ∇· u
+
u · ∇ρ
=
0,
∂t(ρy) +
ρy∇· u
+
u · ∇(ρy) =
0,
∂t(ρu) +
ρu∇· u + ∇p
+
u · ∇(ρu) =
0,
∂t(ρE) +
ρE∇· u + ∇· (pu) +
u · ∇(ρE) =
0,
∂tz
+
u · ∇z
=
0.
(2.9)
The ﬁve-equation system is then split into two subsys-
tems [8, 9, 35]. The ﬁrst system contains only the acous-
tic waves, while the second system takes into account the
propagation of material waves through the ﬂuid.
The system corresponding to the acoustic step is hy-
perbolic with eigenvalues 0 and ±c. It is rewritten and
then solved in its Lagrangian form, which reads
∂tV + ϑ∇· G(V ) = 0,
(2.10)
where V = (ϑ, y, u, E, z) are the Lagrangian variables,
G(V ) = (u, 0, p, pu, 0) is the ﬂux and ϑ = 1/ρ is the spe-
ciﬁc volume. Note that during this step, volume fraction
and mass fraction are not modiﬁed.
The transport step system, where the propagation speed
is only u, reads
∂tU + u∇· U = 0,
(2.11)
with U = (ρ, ρy, ρu, ρE, z) the conservative variables.
2.3. Numerical scheme
Let us ﬁrst consider a domain Ω∈R2 discretized in N
cells Ωi such that v(i) is the set of cells neighboring the
cell Ωi by the edges, |Ωi| is the area of the cell Ωi, |Γij| is
the length of the edge common to the cells Ωi and Ωj and
nij is the unit vector normal to Ωi (see Fig. 2). The mass
centers xi of cell Ωi is deﬁned as
xi =
1
|Ωi|
Z
Ωi
xdx.
(2.12)
Ωi
Ωj
•xi
•xj
•
xf
Γij
nij
Figure 2: Notations associated with the unstructured mesh.
The time is discretized by tn = n∆t for n ∈N, where
∆t > 0 is the time step. In the ﬁnite volume approach
that is used, the notation φn
i is an approximation of
1
|Ωi|
Z
Ωi
φ(x, tn)dx,
for any quantity φ(x, t).
In the sequel, the ﬁrst order in space and time scheme
is described.
3


2.3.1. Acoustic step
An approximate four-state Riemann solver [19], or the
Lagrangian scheme EUCCLHYD [30] can be used to solve
this system. The numerical scheme for the acoustic step
reads
V † = V n −∆t
|Ωi|ϑn
i
X
j∈v(i)
|Γij| G#
ij,
(2.13)
where G#
ij = (−¯uij, 0, ¯pijnij, ¯pij ¯uij, 0)# is the numerical
ﬂux and † corresponds to the intermediate state between
the acoustic step and the transport step. The scheme can
be explicit by taking # = n or implicit with # = †. For
the approximate four-state Riemann solver of Gallice [19],
the ﬂux reads
¯uij =
¯C−
ijui + ¯C+
ijuj
¯C−
ij + ¯C+
ij
· nij −
pj −pi
¯C−
ij + ¯C+
ij
,
¯pij =
¯C+
ijpi + ¯C−
ijpj
¯C−
ij + ¯C+
ij
−¯C−
ij ¯C+
ij
uj −ui
¯C−
ij + ¯C+
ij
· nij
(2.14)
where ¯C−
ij and ¯C+
ij are the Riemann solver slopes. The
choice of slopes is crucial to guarantee the positivity of
the intermediate states of the Riemann solver and thus of
the associated Godunov type scheme [19] (see also [10]).
Their computation is detailed in [35]. In the explicit case,
this step is stable under the classical Courant-Friedrichs-
Lewy (CFL) condition, given by
∆t max
1≤i≤N
 ϑi
|Ωi| max
j∈v(i)
 |Γij| ¯Cij

< 1
2.
(2.15)
When the two phases have large pressure or density
ratios, the CFL (2.15) related to the acoustic system can
be very constraining. This is the case for liquid/gas inter-
actions. Thus, the acoustic step is solved with an implicit
time-scheme [8, 9, 35]. This approach is extensively de-
tailed in [35].
We will recall here the main ideas.
For
the sake of simplicity, a one-dimensional problem is con-
sidered. The extension in dimension 2 follows the same
lines. The sound of speed and the Riemann solver slopes
are frozen at time n.
Since the numerical ﬂux Gij de-
pends only on the velocity and the pressure, we solve the
following subsystem



∂tp +
a2ϑ∂xu =
0,
∂tu +
ϑ∂xp
=
0,
(2.16)
where a = ρc is the Lagrangian sound speed. We now look
for the solution of the system
X† −Xn + ∆t
∆xϑn[[H]] = 0,
(2.17)
where X = (pi, ui) and
[[H]] = (a2
i (¯u†
i+1/2 −¯u†
i−1/2), ¯p†
i+1/2 −¯p†
i−1/2),
(2.18)
is the jumps of the numerical ﬂux, computed with (2.14).
Since the system (2.16) is linear in (p, u), it can be rewrit-
ten as
X† −Xn + ∆t
∆xMX† = 0,
(2.19)
where MX = ϑn[[H]]. The matrix M depends on the
choice made to evaluate the ﬂux. It is given in [35] for the
four-state Riemann solver of [19]. Basically, it depends on
the Riemann solver slope and the sound speed. In practice,
the system obtained is solved under the delta-form

Id + ∆t
∆xM
  X† −Xn
= −∆t
∆xMXn.
(2.20)
The resolution of this linear system then gives a new
velocity and a prediction of the pressure. These pressure
predictions are used to evaluate the numerical ﬂux G†
ij and
update the acoustic variables ϑ† and E† explicitly with
formulae (2.13). The real pressure is then determined by
the EOS (2.5).
2.3.2. Transport step
The transport system is resolved under the equivalent
form
∂tψ + ∇· (uψ) −ψ∇· u = 0,
(2.21)
where ψ corresponds to the variables (ρ, ρy, ρu, ρE, z) of
the system (2.11). The transport equation (2.21) consists
of a ﬁrst conservative term ∇· (uψ) followed by a second
non-conservative term ψ∇· u.
The ﬁrst-order numerical scheme of the transport step
is as follows
ψn+1
i
= ψ†
i −∆t
|Ωi|
X
j∈v(i)
|Γij| ψ†
ijuij +ψ†
i
∆t
|Ωi|
X
j∈v(i)
|Γij| uij.
(2.22)
It is solved with an explicit time scheme. The most natural
choice is to use the upwind scheme to solve the conservative
term. A judicious choice on the material velocity uij gives
a globally conservative scheme.
This choice consists in
taking the opposite of the ﬁrst component of the acoustic
ﬂux [35]. In the case of the Gallice’s Riemann solver [19],
we have uij = ¯u#
ij. The transport step is stable under the
following CFL
∆t max
1≤i≤N

1
|Ωi|
X
j∈v(i)
|Γij| |uij|

< 1.
(2.23)
2.3.3. Global scheme
The overall algorithm for a time step between tn and
tn+1 reads
Step 1: From a state (ρ, ρy, ρu, ρε, z)n, compute
(ρ, ρy, ρu, ρε, z)†, the approximation of the acoustic
system (2.13).
Step 2: Find the ﬂuid state (ρ, ρy, ρu, ρε, z)n+1 by
solving the transport system (2.11) with the initial
state (ρ, ρy, ρu, ρε, z)†.
The global scheme reads
4













































ρn+1
i
=
ρn
i
−
∆t
|Ωi|
X
j∈v(i)
|Γij| ρ†
ij ¯u#
ij,
(ρy)n+1
i
=
(ρy)n
i −
∆t
|Ωi|
X
j∈v(i)
|Γij| (ρy)†
ij ¯u#
ij,
(ρu)n+1
i
=
(ρu)n
i −
∆t
|Ωi|
X
j∈v(i)
|Γij|

(ρu)†
ij ¯u#
ij + ¯p#
ij

,
(ρE)n+1
i
=
(ρE)n
i −
∆t
|Ωi|
X
j∈v(i)
|Γij|

(ρE)†
ij ¯u#
ij + ¯p#
ij ¯u#
ij

,
zn+1
i
=
zn
i
−
∆t
|Ωi|
X
j∈v(i)
|Γij| zn
ij ¯u#
ij + zn
i
∆t
|Ωi|
X
j∈v(i)
|Γij| ¯u#
ij.
(2.24)
This scheme is clearly conservative for the densities,
momentum, and total energy equations. It preserves the
contact discontinuities, namely constant u and p states.
Remark 2
Numerical ﬂux depends on ∆t. Indeed, the variables of the
ﬂux with the exponent † are updated by the acoustic step
(2.13) which depends on the time step. Thus, stationary
solutions depend on the time step. We can see the two
steps as an implicit method to calculate the numerical ﬂux
associated with (2.24).
3. Second-order extension
3.1. Second-order in space extension
When using diﬀuse interface methods, the volume frac-
tion will naturally be diﬀused at the interface between
the two ﬂuids considered. The simple ﬁrst-order in space
scheme used in both the acoustic and the transport step
will produce a numerical diﬀusion of the solution that is
far too large to accurately follow the interface. To over-
come this problem, we use a second-order MUSCL-type
method [28] called U-MUSCL [6]. It consists in using in
the numerical ﬂux the polynomial reconstructions ˆαij and
ˆαji of the solution on Γij such that
ˆαij = αi + κ
2 (αj −αi) + (1 −κ)∇αi.(xf −xi),
ˆαji = αj + κ
2 (αi −αj) + (1 −κ)∇αj.(xf −xj),
(3.1)
with κ ∈[−1, 1] a parameter. By writing it in the form
ˆαij = κ(αj + αi)
2
+ (1 −κ) (αi + ∇αi.(xf −xi)) , (3.2)
it can be seen as a combination of the linear interpolation
(ﬁrst term) and the linear extrapolation (second term) of
ˆαij. It is an extension of van Leer’s κ-scheme on unstruc-
tured grids.
Note that κ = 0 returns the conventional
MUSCL reconstruction. As shown in [33], the U-MUSCL
scheme with κ = 1/3 gives more accurate results than
κ = 0. However, this method is second-order only if the
center of the face xf lies exactly between the mass centers
of cells xi and xj (see Fig. 2). Therefore, a modiﬁcation
from [33] must be employed to preserve the second-order
accuracy.
This consists in modifying the reconstruction
(3.1) such that
ˆαij = αi + κ
2 (αp −αi) + (1 −κ)∇αi.(xf −xi),
(3.3)
where
αp = αj + ∇αj.(2xf −xi −xj).
(3.4)
The U-MUSCL scheme (3.3) with the second-order pre-
serving modiﬁcation will be used as the reconstruction
method due to its simplicity and accuracy when using
κ = 1/3.
3.1.1. Gradient computation on unstructured meshes
The MUSCL-type scheme presented previously requires
the computation of the gradient of the unknowns on un-
structured meshes. There are two methods main for gradi-
ent computation: the Green-Gauss method (GG) and the
Least SQuares method (LSQ). The ﬁrst one is based on the
application of the Green-Ostrogradski theorem. Although
easy to implement and inexpensive, it is not very accurate
on deformed meshes [44]. Thus, the LSQ method is pre-
ferred here for its robustness and accuracy. It is based on
the Taylor development of the solution within a cell i on a
stencil v(i) composed of its neighbors. It can be composed
of neighbors per edge, neighbors per vertex, or from an
even more extended stencil. These ﬁrst two conﬁgurations
are represented in Fig. 3. An enlarged stencil for gradient
computation enables greater accuracy, especially when the
mesh is highly distorted.
The LSQ method consists in solving an over-determined
system AX = b where X = ∇αi (see Appendix A). By
construction, the method is exact for any linear function,
regardless of the mesh. This is essential to obtain second-
order accuracy with MUSCL-type schemes. The accuracy
of the gradient computation can be improved by using a
weighted matrix W, usually diagonal, with coeﬃcients ωkk
such as
ωkk = ||xjk −xi||−q,
(3.5)
5


•xi
•
xj1
•
xj2
•
xj3
•
xj4
•
xj5
•
xj6
•
xj7
•
xj8
Figure 3:
Schematic representation of the neighborhood by edge
(gray) and the neighborhood by vertex (white and gray) for the cal-
culation of the gradient on cell i by the LSQ method.
where q is a positive parameter. When q = 0, we ﬁnd the
unweighted method. With q = 1, the weight corresponds
to the inverse of the distance between the mass center of
cell i and cell j. In this work, q is set to 3/2 [44]. We
thus need to resolve WAX = Wb. This over-determined
system is solved by going through the normal equations,
i.e., by multiplying the system by (WA)T , and the solution
reads
X = (AT W T WA)−1AT W T Wb.
(3.6)
In the case where the matrix is ill-conditioned however, the
gradient calculation could be imprecise. Another approach
to avoid such a problem is to use the QR decomposition.
The matrix WA can be decomposed as WA = QR where
Q is an orthogonal matrix and R is an upper triangular
matrix. The solution is then written
X = R−1QT Wb.
(3.7)
The second method is thus preferred. The QR decompo-
sition is performed with the modiﬁed Gram-Schmidt algo-
rithm [20]. Note that the matrix R−1QT W consists only
of geometrical parameters. When the mesh is not moving,
it is then suﬃcient to calculate it once when initializing the
calculations and to store it. This method can give second-
order accurate gradient reconstruction for smooth curvi-
linear mesh or on meshes of identical parallelograms [44].
3.1.2. Gradient limiter
In order to eliminate oscillations inherent to high-order
schemes and capture discontinuities, these reconstructions
are limited by a function of the local gradient, called “lim-
iter”.
Many limiters exist in the literature, designed to
have diﬀerent properties. In the case of the ﬁve-equation
system, the volume fraction is a discontinuous variable,
which follows a transport equation. Following the lines of
[37, 5, 13], the idea here is to apply a compressive limiter
to this variable, which allows a better representation of the
interface between the two ﬂuids. For the other variables,
namely densities, velocity and pressure, the limiter should
be as robust as possible, given the implicit treatment of
the acoustic step.
The reconstruction ˆαn
ij is rewritetten in the form
ˆαij = αi + φi∆αi,j,
(3.8)
where φi is the gradient limiter, and
∆αi,j = κ
2 (αp −αi) + (1 −κ)∇αi.(xf −xi).
(3.9)
Proposition 1
Let φi the limiter of the reconstruction of αi in cell i, such
that
φi =





min

β, αi −αmin
i
|∆αi|max , αmax
i
−αi
|∆αi|max

if |∆αi|max̸= 0,
0
otherwise.
(3.10)
with |∆αi|max= maxj∈v(i)|∆αi,j|, αmin
i
= minj∈v(i)(αi, ¯αij)
and αmax
i
= maxj∈v(i)(αi, ¯αj). Depending on the parame-
ters β and ¯αj, this limiter displays diﬀerent properties :
• If β = 1 and ¯αj = αiωi + αjωj
ωi + ωj
with ωi = ||xjk −
xi||−1 then the limiter is monotonicity preserving,
i.e.
∀j ∈v(i), αi ≤αj =⇒ˆαij ≤ˆαji.
(3.11)
• If β > 1 and ¯αj = αj, the limiter is compressive.
Proof. The design of the previous limiter is provided in
Appendix B.
This approach is used in [37] in the framework of one-
dimensional limiter for two-phase ﬂows, where the com-
pressive limiter is used on the density whereas the min-
mod limiter is used on the velocity and the pressure. In
[5], a similar β-limiter is designed without the monotonic-
ity preserving condition. The compressive limiter (3.10)
with β = 2 is used on the volume fraction of a two–phase
model while the less compressive limiter with β = 1 is used
on the others variables. In the work presented in [13], a
family of β-limiter is designed from the approach of [4]
where as previously, the compressive limiter is used on the
volume fraction only.
Monotonicity preserving. When dealing with large pres-
sure or density ratio, as in the study of liquid-gas interac-
tion, the monotonicity preserving property can be essen-
tial, in order to improve robustness. The famous minmod
limiter veriﬁes this condition, which explains why it is very
popular in industrial codes.
Compressive property. The compressive nature of a method
or limiter reﬂects the ability to accurately reproduce a dis-
continuity. For example, the superbee limiter [45] is well-
known to sharpen sinusoidal proﬁles and to behave well
on discontinuity transport. This property is particularly
6


valuable in the context of immiscible ﬂuid ﬂows, as it im-
proves the representation of the interface between these
two ﬂuids [37, 5, 13]. When the limiter (3.10) is allowed
to take values greater than 1, it is then compressive.
3.1.3. Second-order in space scheme in the splitting strat-
egy
The limiter (3.10) is used in the context of the ﬁve-
equation system, while reconstructions are performed with
the U-MUSCL scheme with κ = 1/3. The objective is to
sharpen the ﬂuid’s discontinuities by reducing the num-
ber of mixture cells. The reconstructed variables are the
primitive variables, in order to limit velocity and pressure
oscillations. They are reconstructed as follows
Acoustic step: (u, p)n are reconstructed with β =
1.
Transport step: (ρ1, ρ2, u, p)# are reconstructed
with β = 1 and z# with β = 2. The conservative
variables are deduced from the primitive ones with





















ρ
=
zρ1 + (1 −z)ρ2,
ρy =
zρ1,
ρu =
(zρ1 + (1 −z)ρ2) u,
ρε =
P
k zkρkεk = P
k zk
p + γkπk
γk −1 ,
ρE =
ρε + 1
2ρu · u.
(3.12)
Remark 3
The limiter needs the computation of αmin
i
and αmax
i
, the
local extrema.
The 5-point stencil, i.e., with the edges
neighbors, is used for the acoustic step, while the 9-point
stencil i.e., with the vertex neighbors is utilized for the
transport step (see Fig. 3). Better convergence properties
have been observed numerically on the acoustic step with
only 5 points.
3.2. Time-scheme
3.2.1. The Heun method
The time scheme is important to increase the accuracy
of the overall scheme, especially when high-order in space
schemes are used.
This prevents the appearance of nu-
merical artifacts that will degrade the solution, as shown
in [14] for instance. To this end, we could use the Heun
method also called Runge-Kutta TVD, which we apply on
the global scheme (2.24). It reads
U∗
=
Un + ∆tF(Un, ∆t),
U∗∗
=
U∗+ ∆tF(U∗, ∆t),
Un+1 =
1
2 (Un + U∗∗) ,
(3.13)
with U being the conservative variables (ρ, ρy, ρu, ρε, z)
and F is a numerical ﬂux discretized in (2.24). It is not
clear that this scheme is second-order in the case of the
splitting strategy because of the time step dependence of
the numerical ﬂux (see Rem. 2). Indeed, the numerical
ﬂux is a ﬁrst-order in time approximation of the spatial
operator. However, it does improve the numerical solution,
which will be shown in the results section.
3.2.2. An alternative implicit time-scheme
The scheme described above is simple to implement
and robust.
When an implicit approach is used in the
acoustic step however, it must be repeated twice per time
step, which induces a signiﬁcant cost in computation time.
For this, a new implicit scheme based on the Crank-Nicolson
scheme has been developed.
The modiﬁed scheme pro-
posed here reads
V † −V n
∆t
= −1
2ϑn  ∇· G(V †) + ∇· G(V n)

.
(3.14)
This scheme is not the Crank-Nicolson scheme because it
is not second-order since we kept ϑn for the explicit and
implicit terms. It can be seen as the average of the ﬂux
given by the explicit (2.13) and the implicit (2.13) versions
of the acoustic step. The truncation error of this modiﬁed
Crank-Nicolson scheme is smaller than that of the ﬁrst
implicit scheme (see Appendix C). The numerical scheme
for the acoustic step reads
V † = V n −∆t
|Ωi|ϑn
i
X
j∈v(i)
|Γij| 1
2

Gn
ij + G†
ij

.
(3.15)
The resolution of (3.15) follows the same line as the ﬁrst-
order implicit time scheme. In the one-dimensional case,
it is easy to show that the new scheme leads to solve under
the delta-form the system

Id + ∆t
2∆xM
  X† −Xn
= −∆t
∆xMXn.
(3.16)
From an implementation point of view, this represents a
small modiﬁcation of the velocity-pressure system (2.16)
resolution algorithm. Once again, the other equations are
solved explicitly. In order to have a globally conservative
scheme, the material velocity used in the transport step
must read
uij =
¯un
ij + ¯u†
ij
2
.
(3.17)
This comes from the same arguments used in [8, 35] in
order to have a globally conservative scheme.
3.2.3. Extension with the second-order in space scheme
When using the second-order scheme in space, nonlin-
earities introduced by the limiter make it impossible to
write the ﬂux jump as a matrix-vector product of X. A
conventional way to solve non-linear equations is to use a
quasi-Newton method. Using this method, the solution is
7


obtained by iterating over the following equation

Id + ∆t
∆xM
  Xk+1 −Xk
= −Xk + Xn
−∆t
∆xϑn[[Hk]],
(3.18)
with k = 0, ..., ∞. The matrix M is built with the ﬁrst-
order ﬂux while the right-hand side of (3.18) is computed
with the second order ﬂux. Note that a single iteration
over (3.18) gives exactly (2.20), the system solved when
ﬁrst-order in space scheme is used. Through this method,
the implicit time scheme converges to the second-order in
space solution. When using the second implicit scheme,
equation (3.18) becomes

Id + ∆t
2∆xM
  Xk+1 −Xk
= −Xk + Xn
−∆t
2∆xϑn  [[Hk]] + [[Hn]]

.
(3.19)
Remark 4
In practice, we use a convergence threshold ϵ set to 10−5
with a maximum of 10 iterations. These values have proven
to be suﬃcient in the simulations performed.
4. Numerical results
In this section, a selection of numerical results for the
one and two-dimensional cases are presented.
The ﬁrst
cases will illustrate the accuracy of the compressive limiter
(β = 2) on the classical pure transport equation.
The
RK2-TVD time scheme presented in the previous section
is always used for pure transport cases.
The next cases will show the ability of the second-order
method to simulate complex two-phase ﬂows. We denote
by EXEX the explicit scheme for both the acoustic and
transport steps, while IMEX is the implicit-explicit strat-
egy of the splitting scheme. When adding RK2, the RK2-
TVD scheme is thus used on the overall algorithm. The
CNEX scheme denotes the new implicit-explicit scheme
built from a modiﬁcation of the Crank-Nicolson scheme.
For two-dimensional liquid-gas or gas-gas cases, the den-
sity gradient will be computed in log scale during postpro-
cessing, in order to reproduce numerical Schlieren.
4.1. Zalesak’s disk
We consider here the classical Zalesak’s disk test case
[49], namely a disk with a slot rotating in the velocity
ﬁeld u(x, y) = (0.5 −y, x −0.5) m.s−1 on the domain
Ω= [0, 1] m × [0, 1] m. The disk is deﬁned at the initial
time by
z(x, y) =





1 if (||x −c||< r) and
(|x −cx|≥e or y −cy > l) ,
0 otherwise,
with x = (x, y), c = (0.5, 0.75) m, r = 0.15 m, e =
0.025 m and l = 0.1 m. The CFL is set to 0.4 and 256×256
cells are used with constant boundary conditions. The re-
sults in Fig. 4 are obtained after one revolution at the ﬁnal
time tf = 2πs.
The present limiter gives very good results. The in-
terface is very sharp and the shape of the disk is well-
preserved, almost symmetrical. The use of κ = 1/3 (see
Fig. 4.a) gives a good resolution of the slot and the cor-
ners, compared to the conventional MUSCL method with
κ = 0 (see Fig. 4.b).
4.2. Kothe-Rider advection
We now consider the Kothe-Rider forward-backward
advection test case [38] on the domain Ω= [0, 1] m ×
[0, 1] m.
This test case consists in transport a circle in
a backward-forward velocity ﬁeld such that
u(x, y, t) =
 
−sin(πy) cos(πy) sin(πx)2 cos(πt/tf)
sin(πx) cos(πx) sin(πy)2 cos(πt/tf)
!
.
The circle of radius R = 0.15 m is set at (0.5, 0.75) m.
The CFL is set to 0.2 and 256 × 256 cells are used with
constants boundary conditions. The results in Fig. 5 are
obtained at the intermediate time t = 3s and at the ﬁnal
time tf = 6s.
At ﬁrst, until t = 3s, the velocity ﬁeld deforms the disk
into a ﬁlament. The ﬁlament obtained (see Fig. 5.a) is well-
captured in regions where it is large enough regarding the
cell’s size. In a second time, the velocity ﬁeld rewinds the
ﬁlament to reform a disk. At the end, the shape of the
disk is quite well-preserved (see Fig. 5.b). Some diﬀusion
from the forward part deteriorates the results.
A mesh
with more cells gives a better disk resolution.
4.3. One-dimensional liquid-gas shock tube
We consider here the classical one-dimensional liquid-
gas shock tube test on Ω= [0, 1] m, studied in [46] for
instance.
The stiﬀened gas EOS is used for the liquid,
while the gas is modelled by the perfect gas EOS. The
initial conditions read
(ρ, u, p, γ, π) =
(
(103, 0, 109, 4.4, 6 × 108)
if x ≤0.7 m,
(5, 0, 105, 1.4, 0)
else.
The CFL number for the transport step is set to 0.25, and
we use 1000 cells with constant boundary conditions. The
ﬁnal time is set to tf = 240µs. This case serves to test
the accuracy of the diﬀerent time schemes to represent
the three waves of the problem, that are rarefaction wave,
contact discontinuity and shock. This is a challenging test
case, since large pressure and density ratios are involved.
The results are presented in Fig. 6.
The results are in very good agreement with the exact
solution. The three waves are well-represented, especially
8


(a)
(b)
(c)
Figure 4: Zalesak’s disk test case. Initial condition (a), solution after one rotation with κ = 1/3 (b) and with κ = 0 (c).
(a)
(b)
(c)
Figure 5: Kothe-Rider advection test case. Initial condition (a), solution obtained at the intermediate time t = 3s (b) and solution at the
ﬁnal time t = 6s (c).
the contact discontinuity, thanks to the compressive lim-
iter used on the volume fraction. One can see an under-
shot on contact discontinuity of the density (see Fig. 6.a).
This observation is quite common when sharp interface
techniques are used, as the anti-diﬀusive scheme [27, 18].
Regarding velocity and pressure, the CNEX time scheme
gives more accurate results than the IMEX scheme or the
IMEX+RK2 strategy, especially near the rarefaction wave.
It also reduces the overshoot on the density and better lo-
calizes the shock position.
4.4. Two-dimensional Air-R22 interaction
We shall now consider the experiment of Hass & Sturte-
vant [22]. In this test case, a shock wave at Mach 1.22
propagating through the air hits an R22 gas cylinder of
radius R = 25 mm located at (130 mm, 44.5 mm). Many
simulations of this case can be found in the literature,
such as [41, 27, 23].
The computational domain Ω=
[0, 200] mm × [0, 89] mm is described in Fig. 7.
The initial states of both ﬂuids in the pre- and post-
shock regions are given in Tab. 1. Both ﬂuids are modeled
by perfect gas EOS with γ = 1.4 for the air and γ = 1.249
for the R22. The pre-shock density, velocity, and pressure
are computed with the shock wave relation (see [45] for
instance) such that the Mach shock is 1.22. Each phase is
modeled by perfect gas EOS.
Location
ρ (kg.m−3 )
u (m.s−1 )
p (105Pa)
Air (shock)
1.926919
(-105.495,0)
1.5698
Air
1.4
(0,0)
1
R22
4.4154
(0,0)
1
Table 1: Two-dimensional Air-R22 interaction : initial data.
Because of the symmetry of the problem, we use a do-
main of Ω= [0, 200] mm × [44.5, 89] mm with boundary
symmetric condition at the lower side, a wall boundary
condition at the upper side, an outﬂow boundary con-
ditions on the left side and a supersonic inlet boundary
9


 0
 200
 400
 600
 800
 1000
 0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0.9
 1
ρ (kg.m−3)
x (m)
solution
IMEX
IMEX RK2
CNEX
 0
 10
 20
 30
 40
 50
 0.8
 0.826
 0.852
 0.878
(a)
 0
 100
 200
 300
 400
 500
 0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0.9
 1
u (m.s−1)
x (m)
solution
IMEX
IMEX RK2
CNEX
 486
 488
 490
 492
 494
 0.35
 0.526
 0.702
 0.878
(b)
 0
 2e+08
 4e+08
 6e+08
 8e+08
 1e+09
 0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0.9
 1
p (Pa)
x (m)
solution
IMEX
IMEX RK2
CNEX
−1e+06
 0
 1e+06
 2e+06
 3e+06
 0.28
 0.433
 0.586
 0.739
(c)
 0
 0.2
 0.4
 0.6
 0.8
 1
 0
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
 0.7
 0.8
 0.9
 1
z
x (m)
solution
IMEX
IMEX RK2
CNEX
 0
 0.2
 0.4
 0.6
 0.8
 1
 0.814
 0.817
 0.82
 0.823
(d)
Figure 6: One-dimensional liquid-gas shock tube test case. The density (upper left), velocity (upper right), pressure (lower left) and volume
fraction (lower right) are compared at the ﬁnal time tf = 240µs with the exact solution. Except for the volume fraction, only one cell in 10
is represented by points.
R22
Air
Air (shock)
45 mm
50 mm
200 mm
89 mm
44.5 mm
Figure 7: Two-dimensional Air-R22 interaction : geometric descrip-
tion of the test case.
condition on the right side.
The mesh is composed of
1600 × 356 cells. In this case, the implicit time scheme
is not useful since the ratio of densities and pressures is
approximately 1. The explicit version of the time scheme
is therefore used. The CFL number for both steps is set
to 0.25. The ﬁnal time is tf = 1020µs.
The numerical Schlieren of this case is presented in
Fig. 8. For the lower half, the results are obtained with
the EXEX time-scheme, while the EXEX+RK2 strategy is
used for the upper half. For both simulations, the second-
order in space scheme with compressive limiter is used.
When the shock impacts the R22 gas, the cylinder is de-
formed.
As expected, the sound speed is greater in air
than in the R22 gas, such that the transmitted shock
waves propagate slower than the reﬂected ones. Kelvin-
Helmholtz type instabilities develop at the interface be-
tween the two gases. The scheme is able to reproduce a
large scale and ﬁner ﬂow structure, while maintaining a
very sharp interface throughout the simulation. The re-
sults are in good agreement with the experiments of [22]
and with numerical results from the literature [27, 25, 12].
The use of the EXEX-RK2 scheme appears to attenuate
the spurious waves that appear with the EXEX scheme.
10


(a) t = 0µs.
(b) t = 204µs.
(c) t = 408µs.
4.5. Two-dimensional liquid-gas interaction
We shall now consider a two-dimensional liquid-gas in-
teraction, widely studied in the literature [39, 27, 25, 35].
The rectangular computational domain Ω= [0, 2] m ×
[0, 1] m is described in Fig. 9. A gas cylinder of radius
R = 0.4 m is initially at rest in a liquid at (0.5, 0.5) m
while a shock at x = 0.04 m propagates in the liquid from
left to right.
The initial conditions of each phase are provided in
Tab. 2. The air is modeled by perfect gas EOS with γ =
1.4 while the liquid is modeled by stiﬀened gas EOS with
γ = 4.4 and π = 6.8 × 108.
Because of the symmetry of the problem, a domain of
Ω= [0, 2] m×[0.5, 1] m is used with a boundary symmetric
condition at the lower side, a wall boundary condition at
the upper side and inﬂow/outﬂow boundary conditions on
(d) t = 612µs.
(e) t = 816µs.
(f) t = 1020µs.
Figure 8: Numerical Schlieren of the air-R22 interaction test case.
Results of the EXEX scheme (lower half) and results with the
EXEX+RK2 scheme (upper half) at several times.
Location
ρ (kg.m−3 )
u (m.s−1 )
p (Pa)
Liquid (shock)
1030.9
(300,0)
3 × 109
Liquid
1000
(0,0)
105
Gas
1
(0,0)
105
Table 2: Two-dimensional liquid-gas interaction : initial data
the other sides. The mesh is composed of 1600 × 400 cells.
The CNEX time scheme is used with the second-order in
space strategy. Since the acoustic step is implicit, the time
step is computed with the transport condition (2.23) with
a CFL number set to 0.25.
Volume fraction and numerical Schlieren are presented
11


gas
liquid
liquid (shock)
0.04 m
0.5 m
2 m
1 m
0.5 m
0.4 m
Figure 9: Two-dimensional liquid-gas interaction : geometric descrip-
tion of the test case.
(a) t = 100µs.
(b) t = 200µs.
(c) t = 300µs.
(d) t = 400µs.
(e) t = 500µs.
(f) t = 600µs.
Figure 10: Two-dimensional liquid-gas interaction test case. Numer-
ical Schlieren (upper half) and volume fraction (lower half) at several
times.
on Fig. 10. The results are in good agreement with the
results presented in [39, 27, 25, 35]. The compressive lim-
iter gives a sharp interface of the volume fraction. The
propagation waves are shown on numerical Schlieren. The
shock wave hits the gas cylinder and compresses it while
propagating inside. When the shock reaches at the left of
the cylinder, the gas is split into two diﬀerent cylinders.
Remark 5
Many schemes are unable to cope with this test case be-
cause of the high ratio of pressure and density, even with
explicit time schemes. The robustness of the method come
from the exact conditions on the Riemann solver slopes to
12


ensure positiveness of the intermediate states [35]. How-
ever, these conditions are derived from the explicit time-
scheme of the acoustic step.
In our case, when an im-
plicit time-scheme is used, no conditions can be derived to
guarantee that the solution remains in the convex set of
admissible solutions, which ensures that the sound speed
given by (2.7) is real, and the mixture density is positive.
When using the compressive limiter however, it happens
in this test case that p + π becomes negative in mixture
cells where the volume fraction of liquid is approximately
10−4 and the mixture density is about 10−3. To overcome
this issue, when the square of the velocity of the mixing
sound speed in cell i is negative, it is calculated with
c2
i = max(c2
i,liq, c2
i,gas)
(4.1)
where ci,liq and ci,gas are the sound speed of the liquid
phase and gas phase in cell i.
This ﬁx can be seen as
adding diﬀusion to the implicit resolution of the acoustic
step, where the slopes of the Riemann solver are locally
increased. It is the only part where the sound speed is
needed.
4.6. Two-dimensional liquid-gas interaction on deformed
mesh
All the cases presented above were performed on Carte-
sian grids.
Thus, the last liquid-gas test case is repro-
duced on a deformed mesh, in order to show that the
method works on unstructured grids.
The deformation
is performed from a Cartesian mesh, where the nodes are
randomly moved. A small portion of the mesh is shown in
Fig. 11.
Figure 11: Mesh used for the Two-dimensional liquid-gas interaction
on deformed mesh.
The same parameters, scheme and number of cells are
used. This time, the calculation is performed on the com-
plete domain, without using any symmetry conditions.
The numerical Schlieren at several times are presented
in Fig. 12. The results are in good agreement with the
previous simulation on a Cartesian grid. One can observe
that the shape of the gas cylinders are not perfectly sym-
metrical with the horizontal axis.
This comes from the
non-symmetry of the randomly deformed grid. The diﬀer-
ent waves are, however, well-captured, and the two-phase
interface remains sharp.
(a) t = 100µs.
(b) t = 200µs.
(c) t = 300µs.
(d) t = 400µs.
13


(e) t = 500µs.
(f) t = 600µs.
Figure 12: Two-dimensional liquid-gas interaction test case on ran-
dom mesh. Numerical Schlieren at several times.
5. Conclusion
In this work, several methods for reducing numerical
diﬀusion when studying compressible two-phase ﬂows with
immiscible ﬂuids were presented. Since non-miscible ﬂu-
ids are considered, the ﬁve-equation system of [2, 31] was
used. A robust implicit–explicit acoustic-transport split-
ting scheme for two-phase ﬂows is used to solve the gov-
erning equations. The main contribution of this work is
to improve the accuracy of numerical methods used on
meshes potentially distorted by ablation. A new and more
accurate implicit-explicit time scheme was developed. The
second-order extension of the method was performed with
a MUSCL-type scheme, that serves to improve the reso-
lution of the numerical scheme. The U-MUSCL scheme
[6] with κ = 1/3 is used, with the correction of [33] for
deformed grids. A multidimensional limiter is designed to
capture shock and contact discontinuity without oscilla-
tions. A β parameter serves to control the compressive
property of the limiter, used only on the volume fraction.
It sharpens the interface between phases, reduce the nu-
merical diﬀusion and reproduces ﬁner ﬂow structures. The
continuation of this work consists in enriching the model
by taking into account capillary eﬀects. These physics will
be included in the acoustic step, still with an implicit time-
scheme.
14


Appendix A. Least-squares method
The least-squares method used to compute the local
gradient is described in this appendix. We consider the
solution αi in xi, the center of the cell i. We then deﬁne
the following linear model
ˆαi(x) = αi + ∇αi.(x −xi).
(A.1)
The method consists in ﬁnding the minimum of the sum
Si =
X
j∈v(i)
(ˆαi(xj) −αj)2,
(A.2)
where v(i) is a set of neighbors of cell i. The minimization
problem (A.2) can be written in the form
Si = ||AX −b||2
(A.3)
with
X =



∂αi
∂x
∂αi
∂y


,
A =





xj1 −xi
yj1 −yi
xj2 −xi
yj2 −yi
...
...
xjN −xi
yjN −yi





and
b =





αj1 −αi
αj2 −αi
...
αjN −αi




.
where N is the number of neighboring cells. Thus, ﬁnding
the minimum of (A.2) consists in ﬁnding the minimum of
the function f(X) = ||AX −b||2. Its gradient is zero when
AX = b.
Appendix B. Design of the limiter
In this appendix, the main lines used to obtain the
limiter from the inequalities on the local monotonicity pre-
serving principle are detailed.
First, the monotonicity principle is derived. To satisfy
the monotonicity principle, the reconstruction of αi need
to hold between minj∈v(i)(αi, ¯αj) and maxj∈v(i)(αi, ¯αj)
where ¯αj is the average value between αi and αj on their
common face. It is such that
¯αj −αi
||xjk −xi|| =
αj −¯αj
||xjk −xj||.
(B.1)
Thus,
¯αj = αiωi + αjωj
ωi + ωj
(B.2)
with ωi = ||xjk −xi||−1.
Lemma 1
Let ˆαij be the extrapolation of αi on Γij such that
ˆαij = αi + φi∆αi,j,
(B.3)
where φi is the gradient limiter, and
∆αi,j = κ
2 (αp −αi) + (1 −κ)∇αi.(xf −xi).
(B.4)
The reconstruction (B.3) and its symmetrical preserve the
monotonicity of the solution i.e.
∀j ∈v(i), αmin
i
≤αi ± φi∆αi,j ≤αmax
i
.
(B.5)
where αmin
i
= minj∈v(i)(αi, ¯αj), αmax
i
= maxj∈v(i)(αi, ¯αj),
the limiter φi reads
φi =





min
αi −αmin
i
|∆αi|max , αmax
i
−αi
|∆αi|max

if ∆αi,j ̸= 0,
0
otherwise,
(B.6)
with |∆αi|max= maxj∈v(i)|∆αi,j|.
Proof. From B.5, we have ∀j ∈v(i)



αmin
i
≤
αi + φi∆αi,j
≤
αmax
i
,
αmin
i
≤
αi −φi∆αi,j
≤
αmax
i
,
which gives



αmin
i
−αi
≤
φi∆αi,j
≤
αmax
i
−αi,
αi −αmax
i
≤
φi∆αi,j
≤
αi −αmin
i
.
Then, two cases should be distinguished
• if ∆αi,j > 0 then







αmin
i
−αi
∆αi,j
≤
φi
≤
αmax
i
−αi
∆αi,j
,
αi −αmax
i
∆αi,j
≤
φi
≤
αi −αmin
i
∆αi,j
,
• if ∆αi,j < 0 then







αi −αmax
i
|∆αi,j|
≤
φi
≤
αi −αmin
i
|∆αi,j| ,
αmin
i
−αi
|∆αi,j|
≤
φi
≤
αmax
i
−αi
|∆αi,j|
.
Because φi has to be positive, the left inequalities are al-
ways true. The right inequalities have to be true ∀j ∈v(i)
so the minimum value is chosen. The limiter thus reads
φi = min
j∈v(i)













min
αmax
i
−αi
∆αi,j
, αi −αmin
i
∆αi,j

if ∆αi,j < 0,
min
αi −αmin
i
|∆αi,j| , αmax
i
−αi
|∆αi,j|

if ∆αi,j > 0,
0
otherwise,
(B.7)
which is equivalent to (B.6).
15


The constraint φi ≤1 must be added in order to en-
sure the reconstruction to be exact for linear solution on
admissible mesh. The ﬁnal expression of limiter is thus
φi =





min

1, αi −αmin
i
|∆αi|max , αmax
i
−αi
|∆αi|max

if ∆αi,j ̸= 0,
0
otherwise,
(B.8)
Secondly, if the limiter must be compressive, the only
condition that need to be satisfy is the maximum principle
∀j, min
j∈v(i)(αi, αj) ≤ˆαij ≤max
j∈v(i)(αi, αj).
(B.9)
The limiter is thus
φi =





min

β, αi −αmin
i
|∆αi|max , αmax
i
−αi
|∆αi|max

if ∆αi,j ̸= 0,
0
otherwise,
(B.10)
with αmin
i
= minj∈v(i)(αi, αj) and αmax
i
= maxj∈v(i)(αi, αj).
The β parameter is added to control the compressive prop-
erty of the limiter. For the volume fraction which is theo-
retically piecewise constant, preservation of the maximum
principle is the only desired property. It is not necessary
to preserve the linearity of the solution so β > 1 can be
employed.
Appendix C. Truncation error of implicit acoustic
time-schemes
In this appendix, the calculation of the truncation error
of both implicit time-scheme used for the acoustic step
(2.10) is detailed.
Lemma 2
The truncation error of the implicit time-scheme (2.13)
that reads
V n+1 −V n
∆t
= −ϑn∇· G(V n+1),
(C.1)
is given by
τ n = ∆t
∂2
t V (t)
2
+ ϑ(t)∇· ∂tG(V (t))

+ O(∆t2), (C.2)
while the truncation error of the modiﬁed Crank-Nicolson
time-scheme (3.14) that reads
V n+1 −V n
∆t
= −1
2ϑn  ∇· G(V n) + ∇· G(V n+1)

,
(C.3)
is given by
τ n = ∆t
∂2
t V (t)
2
+ ϑ(t)∇· ∂tG(V (t))
2

+O(∆t2). (C.4)
Proof. The truncation error of the ﬁrst implicit time-scheme
scheme is
τ n =
V (t + ∆t) −V (t)
∆t
+ ϑ(t)∇· G(V (t + ∆t)),
=
∂tV (t) + ∆t
2 ∂2
t V (t) + ϑ(t)∇· G(V (t)
+∆t∇· (∂tG(V (t))) + O(∆t2),
=
∆t
∂2
t V (t)
2
+ ϑ(t)∇· ∂tG(V (t))

+ O(∆t2).
(C.5)
The truncation error of the modiﬁed Crank-Nicolson
scheme is
τ n =
V (t + ∆t) −V (t)
∆t
+1
2ϑ(t) (∇· G(V (t + ∆t)) + ∇· G(V (t))) ,
=
∂tV (t) + ∆t
2 ∂2
t V (t) + ϑ(t)∇· G(V (t)
+∆t
2 ∇· (∂tG(V (t))) + O(∆t2),
=
∆t
2

∂2
t V (t) + ϑ(t)∇· ∂tG(V (t))

+ O(∆t2).
(C.6)
From the truncation errors calculated earlier, one can
see that the new implicit scheme, called the modiﬁed Crank-
Nicolson scheme, produces a smaller error than the ﬁrst
implicit scheme. We then expect to achieve more accurate
numerical results with the second scheme.
References
[1] Ahn, H., Shashkov, M., 2007. Multi-material interface recon-
struction on generalized polyhedral meshes. J. Comput. Phys.
226, 2096–2132.
[2] Allaire, G., Clerc, S., Kokh, S., 2002. A ﬁve-equation model
for the simulation of interfaces between compressible ﬂuids. J.
Comput. Phys. 181, 577–616.
[3] Baer, M., Nunziato, J., 1986.
A two-phase mixture theory
for the deﬂagration-to-detonation transition (DDT) in reactive
granular materials. Int. J. Multiphase Flow 12, 861–889.
[4] Barth, T., Jespersen, D.C., 1989. The design and application of
upwind schemes on unstructured meshes. AIAA Paper 89-0366
.
[5] Blanchard, G., 2015. Modélisation et simulation multi-échelles
de l’atomisation d’une nappe liquide cisaillée. Ph.D. thesis. Uni-
versité de Toulouse.
[6] Burg, C., 2005. Higher Order Variable Extrapolation for Un-
structured Finite Volume RANS Flow Solvers.
AIAA Paper
2005-4999 .
[7] Caramana, E., Shashkov, M., 1998.
Elimination of Artiﬁ-
cial Grid Distortion and Hourglass-Type Motions by Means
of Lagrangian Subzonal Masses and Pressures.
J. Comput.
Phys. 142, 521–561.
URL: https://www.sciencedirect.com/
science/article/pii/S0021999198959526,
doi:https://doi.
org/10.1006/jcph.1998.5952.
[8] Chalons, C., Girardin, M., Kokh, S., 2016.
An all-regime
Lagrange-Projection like scheme for the gas dynamics equations
on unstructured meshes. Comm. Comput. Phys. 20, 188–233.
16


[9] Chalons, C., Girardin, M., Kokh, S., 2017.
An all-regime
Lagrange-Projection like scheme for 2D homogeneous mod-
els for two-phase ﬂows on unstructured meshes.
J. Comput.
Phys. 335, 885–904.
URL: https://www.sciencedirect.com/
science/article/pii/S002199911730027X,
doi:https://doi.
org/10.1016/j.jcp.2017.01.017.
[10] Chan, A., Gallice, G., Loubère, R., Maire, P.H., 2021. Posi-
tivity preserving and entropy consistent approximate Riemann
solvers dedicated to the high-order MOOD-based Finite Volume
discretization of Lagrangian and Eulerian gas dynamics. Com-
put. Fluids 229, 105056.
URL: https://www.sciencedirect.
com/science/article/pii/S0045793021002206,
doi:https://
doi.org/10.1016/j.compfluid.2021.105056.
[11] Chanteperdrix, G., Villedieu, P., Vila, J., 2002. A Compressible
Model for Separated Two-Phase Flows Computations Volume 1:
Fora, Parts A and B, 809–816.
[12] Cheng, L., Deng, X., Xie, B., Jiang, Y., Xiao, F., 2021. Low-
dissipation BVD schemes for single and multi-phase compress-
ible ﬂows on unstructured grids. J. Comput. Phys. 428, 110088.
[13] Chiapolino, A., Saurel, A., Nkonga, B., 2017.
Sharpening
diﬀuse
interfaces
with
compressible
ﬂuids
on
unstruc-
tured meshes.
J. Comput. Phys. 340,
389–417.
URL:
https://www.sciencedirect.com/science/article/pii/
S0021999117302371,
doi:https://doi.org/10.1016/j.jcp.
2017.03.042.
[14] De Vuyst, F., Fochesato, C., Mahy, V., Motte, R., Peybernes,
M., 2021. A geometrically accurate low-diﬀusive conservative in-
terface capturing method suitable for multimaterial ﬂows. Com-
put. Fluids , 104897URL: https://www.sciencedirect.com/
science/article/pii/S0045793021000633,
doi:https://doi.
org/10.1016/j.compfluid.2021.104897.
[15] Deng, X., Inaba, S., Xie, B., Shyue, K., Xiao, F., 2018. High
ﬁdelity discontinuity-resolving reconstruction for compressible
multiphase ﬂows with moving interfaces. J. Comput. Phys. 371,
945–966. doi:https://doi.org/10.1016/j.jcp.2018.03.036.
[16] Després, B., Lagoutière, F., 2002. Contact discontinuity captur-
ing schemes for linear advection and compressible gas dynamics.
J. Sci. Comput 16, 16–479.
[17] Flåtten, T., Lund, H., 2011. Relaxation two-phase ﬂow mod-
els and the subcharacteristic condition. Mathematical Models
and Methods in Applied Sciences 21, 2379–2407. doi:10.1142/
S0218202511005775.
[18] Friess, M.B., Boutin, B., Caetano, F., Faccanoni, G., Kokh, S.,
Lagoutière, F., Navoret, L., 2011. A second order anti-diﬀusive
Lagrange-remap scheme for two-component ﬂows.
ESAIM:
Proc. 32, 149–162.
[19] Gallice, G., 2003.
Positive and entropy stable Godunov-type
schemes for gas dynamics and MHD equations in Lagrangian or
Eulerian coordinates. Numerische Mathematik 94, 673–713.
[20] Gander, W., 2003. Algorithms for the QR-Decomposition .
[21] Grenier,
N.,
Vila,
J.P.,
Villedieu,
P.,
2013.
An
ac-
curate
low-mach
scheme
for
a
compressible
two-ﬂuid
model
applied
to
free-surface
ﬂows.
J.
Comput.
Phys.
252,
1–19.
URL:
https://www.sciencedirect.com/
science/article/pii/S0021999113004312,
doi:https:
//doi.org/10.1016/j.jcp.2013.06.008.
[22] Haas, J.F., Sturtevant, B., 1987.
Interaction of weak shock
waves with cylindrical and spherical gas inhomogeneities.
J.
Fluid Mech. 181, 41–76.
[23] Helluy, P., Jung, J., 2013. OpenCL numerical simulations of
two-ﬂuid compressible ﬂows with a 2D random choice method.
IJFV 10, 1–38.
[24] Hirt,
C.,
Nichols,
B.,
1981.
Volume
of
ﬂuid
(VOF)
method
for
the
dynamics
of
free
boundaries.
J.
Comput.
Phys.
39,
201–225.
URL:
https://www.
sciencedirect.com/science/article/pii/0021999181901455,
doi:https://doi.org/10.1016/0021-9991(81)90145-5.
[25] Jung, J., 2013. Schémas numériques adaptés aux accélérateurs
multicœurs pour les écoulements biﬂuide. Ph.D. thesis. Univer-
sité de Strasbourg.
[26] Kapila, A., Menikoﬀ, R., Bdzil, J., Son, S., Stewart, D.S., 2001.
Two-phase modeling of deﬂagration-to-detonation transition in
granular materials: Reduced equations. Phys. Fluids 13, 3002–
3024.
[27] Kokh, S., Lagoutière, F., 2010.
An anti-diﬀusive numerical
scheme for the simulation of interfaces between compressible
ﬂuids by means of ﬁve-equation model. J. Comput. Phys. 229,
2773–2809.
[28] van Leer, B., 1979. Towards the ultimate conservative diﬀerence
scheme. V. A second-order sequel to Godunov’s method.
J.
Comput. Phys. 32, 101–136.
[29] Lund, H., 2012.
A hierarchy of relaxation models for two-
phase ﬂow. SIAM J. App. Math. 72, 1713–1741. doi:10.1137/
12086368X.
[30] Maire, P.H., Abgrall, R., Breil, J., Ovadia, J., 2007.
A cell-
centered lagrangian scheme for two-dimensional compressible
ﬂow problems. SIAM J. Sci. Comput. 29, 1781–1824.
[31] Massoni, J., Saurel, R., Nkonga, B., Abgrall, R., 2002. Some
models and Eulerian methods for interface problems between
compressible ﬂuids with heat transfer. Int. J. Heat Mass Trans-
fer 45, 1287–1307.
[32] Murrone, A., Guillard, H., 2005.
A ﬁve equation reduced
model for compressible two phase ﬂow problems. J. Comput.
Phys. 202, 664 – 698. URL: http://www.sciencedirect.com/
science/article/pii/S0021999104003018.
[33] Nishikawa, H., 2020. On the loss and recovery of second-order
accuracy with U-MUSCL. J. Comput. Phys. 417, 109600.
[34] Osher, S., Sethian, J., 1988. Fronts propagating with curvature-
dependent
speed:
Algorithms
based
on
Hamilton-Jacobi
formulations. J. Comput. Phys. 79, 12–49. URL: https://www.
sciencedirect.com/science/article/pii/0021999188900022,
doi:https://doi.org/10.1016/0021-9991(88)90002-2.
[35] Peluchon, S., Gallice, G., Mieussens, L., 2017.
A robust
implicit–explicit acoustic-transport splitting scheme for two-
phase ﬂows. J. Comput. Phys. 339, 328–355.
[36] Peluchon,
S.,
Gallice,
G.,
Mieussens,
L.,
2021.
Devel-
opment
of
numerical
methods
to
simulate
the
melt-
ing
of
a
thermal
protection
system.
J.
Comput.
Phys.
,
110753URL:
https://www.sciencedirect.com/
science/article/pii/S0021999121006483,
doi:https:
//doi.org/10.1016/j.jcp.2021.110753.
[37] Qian, L., Causon, D., Mingham, C., Ingram, D., 2006. A free-
surface capturing method for two ﬂuid ﬂows with moving bod-
ies. Proc. R. Soc. A. 462, 21–42. doi:10.1098/rspa.2005.1528.
[38] Rider, W., Kothe, D., 1998. Reconstructing Volume Tracking.
J. Comput. Phys. 141, 112–152. doi:10.1006/jcph.1998.5906.
[39] Saurel, R., Abgrall, R., 1999. A simple method for compressible
multiﬂuid ﬂows. SIAM J. Sci. Comput. 21 (3), 1115–1145.
[40] Saurel, R., Petitpas, F., Berry, R., 2009. Simple and eﬃcient
relaxation methods for interfaces separating compressible ﬂuids,
cavitating ﬂows and shocks in multiphase mixtures. J. Comput.
Phys. 228, 1678–1712. URL: https://www.sciencedirect.com/
science/article/pii/S0021999108005895,
doi:https://doi.
org/10.1016/j.jcp.2008.11.002.
[41] Shyue, K., 2006.
A wave-propagation based volume tracking
method for compressible multicomponent ﬂow in two space di-
mensions. J. Comput. Phys. 215, 219–244.
[42] Shyue, K., Xiao, F., 2014. An Eulerian interface sharpening al-
gorithm for compressible two-phase ﬂow: The algebraic THINC
approach. J. Comput. Phys. 268, 326–354.
[43] Sun,
Z.,
Inaba,
S.,
Xiao,
F.,
2016.
Boundary
vari-
ation
diminishing
(BVD)
reconstruction:
A
new
ap-
proach to improve Godunov schemes.
J. Comput. Phys.
322,
309–325.
URL:
https://www.sciencedirect.com/
science/article/pii/S0021999116302765,
doi:https://doi.
org/10.1016/j.jcp.2016.06.051.
[44] Syrakos, A., Varchanis, S., Dimakopoulos, Y., Goulas, A.,
Tsamopoulos, J., 2017. A critical analysis of some popular meth-
ods for the discretisation of the gradient operator in ﬁnite vol-
ume methods. Phys. Fluids 29, 127103. doi:10.1063/1.4997682.
[45] Toro, E., 1997. Riemann solvers and Numerical Methods for
Fluid Dynamics. Springer.
17


[46] Vilar, F., Shu, C., Maire, P.H., 2016.
Positivity-preserving
cell-centered lagrangian schemes for multi-material compress-
ible ﬂows:
From ﬁrst-order to high-order. Part I: The one-
dimensional case. J. Comput. Phys. .
[47] VonNeumann, J., Richtmyer, R.D., 1950. A method for the nu-
merical calculation of hydrodynamic shocks. J. App. Phys. 21,
232–237. URL: https://doi.org/10.1063/1.1699639, doi:10.
1063/1.1699639, arXiv:https://doi.org/10.1063/1.1699639.
[48] Xiao, F., Honma, Y., Kono, T., 2005. A simple algebraic inter-
face capturing scheme using hyperbolic tangent function. Int. J.
Numer. Methods Fluids 48, 1023 – 1040. doi:10.1002/fld.975.
[49] Zalesak,
S.,
1979.
Fully
multidimensional
ﬂux-corrected
transport
algorithms
for
ﬂuids.
J.
Comput.
Phys.
31,
335–362.
URL:
https://www.sciencedirect.com/
science/article/pii/0021999179900512,
doi:https:
//doi.org/10.1016/0021-9991(79)90051-2.
18
