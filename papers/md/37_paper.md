Contents lists available at ScienceDirect
Journal of Computational Physics
journal homepage: www.elsevier.com/locate/jcp
Semi-implicit method for two-phase ﬂows in pressure 
disequilibrium 
Solène Schropﬀ
∗, Fabien Petitpas
, Eric Daniel
Aix Marseille Univ, CNRS, IUSTI, Marseille, France
a r t i c l e  i n f o
Keywords:
Two-phase ﬂows
Compressible ﬂows
Low-Mach ﬂows
Implicit-explicit
IMEX Scheme
Pressure-based
 
a b s t r a c t
Explicit ﬁnite-volume schemes face diﬃculties when simulating low-Mach and weakly compress-
ible ﬂows: the solution is not asymptotically preserved and the time-step becomes too small due 
to the acoustic stability criterion. Together, both of these aspects cause numerical simulations 
to be invalid and impractical. To circumvent these issues and obtain eﬃcient and accurate re-
sults, we propose a new semi-implicit (IMEX) 1D collocated scheme for solving two-phase ﬂows 
in pressure disequilibrium. This IMEX scheme relies upon a hyperbolic ﬂux-splitting that fully 
decouples the acoustic and convective properties of the ﬂow. In doing so, the acoustic subsystem 
is implicitly solved and the usual time step stability restriction is lifted. A Picard iteration method 
is used to linearize the pressure system, thus considerably simplifying the resolution. Due to the 
explicit solving of the advection subsystem, a convective stability criterion remains, which is less 
restrictive than in conventional fully explicit methods. A general formulation of the equation of 
state is used to obtain a ﬂuid-independent numerical scheme. Numerical results for single-phase 
and two-phase ﬂows are presented: the accuracy and overall adaptability and eﬃciency of the 
new computational method are shown for a wide range of Mach numbers, going from weakly-
compressible ﬂows to large pressure discontinuity problems.
1.  Introduction
Compressible multiphase ﬂows are encountered in many ﬁelds, such as nuclear energy, aeronautics, aerospace, medicine [1–11]. 
It is therefore crucial to properly understand and model them. A multiphase ﬂow study implies many challenges: the ﬂow could 
include both high-Mach regions, where shock waves are of interest, and low-Mach regions, where material velocities are high. This 
is the case for example for axial pumps, where the ﬂow in the main direction is a low speed one while the ﬂow velocity may be very 
large at the edge of the blade. This could also be observed in heat exchangers, where the appearance of vapor bubbles may completely 
change the behavior of the ﬂow, initially very weakly compressible.
Given their complexity, many methods have been put forward to study these ﬂows, with each their own advantages and drawbacks. 
A key approach to solve interface problems eﬃciently and accurately (which may involve pure media, mixtures and wave dynamics) is 
numerical simulation. This work focuses on a class of numerical methods, which are Diﬀuse Interface Methods (DIM), where interfaces 
are allowed to diﬀuse numerically [12,13]. These methods are appealing as they are able to manage the dynamic appearance and 
disappearance of interfaces. Additionally, they uniquely provide a well-deﬁned thermodynamic framework for mixtures, supported 
by speciﬁc equations of state for each phase.
∗Corresponding author.
 
E-mail addresses: solene.schropﬀ@univ-amu.fr (S. Schropﬀ), fabien.petitpas@univ-amu.fr (F. Petitpas), eric.daniel@univ-amu.fr (E. Daniel).
https://doi.org/10.1016/j.jcp.2025.114545
Received 17 March 2025; Received in revised form 15 November 2025; Accepted 22 November 2025
Journal of Computational Physics 547 (2026) 114545 
Available online 28 November 2025 
0021-9991/© 2025 The Author(s). 
Published by Elsevier Inc. 
This is an open access article under the CC BY license 
( http://creativecommons.org/licenses/by/4.0/ ). 


S. Schropﬀ, F. Petitpas and E. Daniel
DIM methods rely on hyperbolic multiphase ﬂow models, which can model mixtures in full disequilibrium (Baer-Nunziato model 
[14]) or in mechanical equilibrium (Kapila model [15,16]). However, the latter includes additional non-conservative terms in the 
volume fraction equations to account for diﬀerences in phase compressibilities within the mixing regions, which leads to numerical 
instabilities. To address this challenge, the velocity-equilibrium/pressure-disequilibrium model was developed (see for example [17,
18]), associated with a relaxation of pressure [19]. This model is an hyperbolic overdetermined system where the monotonic behavior 
of the associated mixture sound speed (called frozen sound speed) facilitates the development of numerical methods and ensures 
improved accuracy in capturing wave propagation across diﬀuse interfaces. By using the velocity-equilibrium model containing the 
internal energies and relaxing it, one avoids the resolution of the non-conservative volume fraction equation, the diﬃculty to preserve 
positivity of volume fractions, and numerical instabilities stemming from strong compression and rarefaction waves near the interface 
[19]; the asymptotic limit of the Kapila model is also obtained. Later, [20] proposed a more general way to use this two-pressure 
model and showed that pressure disequilibrium is mandatory to ensure good wave propagation through interfaces.
One crucial point is that the methods presented in the above references are explicit and suﬀer from the same drawback: they are 
restricted by the CFL criterion associated to the propagation of the acoustic waves. For a mixture in a velocity equilibrium, only a 
slight amount of liquid — which is usually very weakly compressible — will increase the frozen (mixture) sound speed towards the 
liquid phase sound speed. As the velocity of the acoustic waves depends on the speed of sound of the mixture, the time step is strongly 
aﬀected and decreases greatly: this leads to a time-consuming simulation. Moreover, another issue of compressible solvers is that they 
are not asymptotically preserving in the low Mach number limit [21–24]: the discretized solution of the compressible equations may 
fail to provide an accurate approximation of the incompressible equations.
Therefore, a solution must be put forward to dampen these issues, while keeping the consistency of the numerical scheme. The 
goal of this paper is to propose a new numerical scheme able to circumvent the acoustic time step restriction associated to explicit 
schemes for compressible ﬂows and also yield accurate results in the low-Mach range.
First hand, one may think of fully implicit schemes to remove the stability restriction, as the scheme would be stable whatever 
the time step used. But this can lead to two kinds of problems: classical upwind discretization is highly non-linear and very diﬃcult 
to solve implicitly. In addition, while implicit versions of classical schemes alleviate the time-step restriction associated with acoustic 
waves, they can introduce excessive numerical dissipation on the slower, physically relevant waves (particularly at low Mach numbers) 
ultimately leading to a loss of accuracy despite improved stability. Investigation on fully implicit schemes can be found in [8,23,25,26]. 
Therefore, semi-implicit methods have been proposed instead to reduce the time step restriction [27–39]: this kind of scheme replaces 
the acoustic stability condition with a convective stability condition. Besides, by only being semi-implicit, the scheme is much simpler 
and easier to solve.
To continue, the new numerical scheme should indeed be able to handle low-Mach ﬂows, but also preferably keep a good resolution 
of compressible ﬂows. But depending on the Mach number, the numerical methods may diﬀer as they do not aim for the same goal. 
When the Mach number is low, there are no shock discontinuities and the conservative approach may not be mandatory. However, 
when the Mach number increases, shock discontinuities can arise and conservative schemes are required. This is why methods 
suited for incompressible ﬂows [40,41] could work for low-Mach ﬂows, but would fail whenever the compressibility eﬀects increase. 
However, the idea behind incompressible solvers is retained, which is the pressure-based method.
The ﬁrst implementation of pressure-based solvers for compressible ﬂows was done by Harlow and Amsden [27,28]. Many ﬁnite 
volume methods initially developed for incompressible ﬂows have been extended to the weakly compressible and low Mach regime 
[29,30,36,42–46]. Techniques were explicitly designed to treat low Mach number regimes, based on low Mach number asymptotics 
[47,48]. In the cases where the Mach number can change by several orders of magnitude, an adaptive low Mach number scheme 
based on a non-conservative formulation was developed [49]. But generally, when equations are not written in conservative form, the 
scheme is unable to deal accurately with shock waves. Moreover, with pressure-based methods, major issues arise when considering 
a variable density ﬁeld: the system must be adjusted by including the time derivative term on the mass conservation equation and 
by adapting also the momentum equation to account for spatial density variation. In addition, an equation of state is needed to close 
the system of equations.
For those reasons, the gap between incompressible/low-Mach solvers and compressible ﬂow solvers should be bridged, if the goal 
is to be accurate both in low-Mach and compressible regimes. This kind of solver is called "all-Mach" or "all speed": it should be 
conservative to ensure correct shock computation, but also asymptotically preserving to ensure the correct behavior of the method in 
the incompressible limit [44,50,51]. The ﬁrst true all-Mach number ﬂow solver for single-phase ﬂow is from [52], which comes from 
a former extension of the SIMPLE method [36]: it is pressure-based and uses a conservative formulation, which can capture shocks 
when the Mach number increases. Since then, many other algorithms were developed for all Mach number ﬂows: the pressure-based 
technique combined to a conservative approach seems to be the most ﬁtting method. When used in a semi-implicit scheme, where 
the pressure part is implicitly solved, the acoustic restriction can ﬁnally be lifted. The conservative semi-implicit scheme consists of 
splitting hyperbolically the ﬂux vector into two parts, one of which will be treated implicitly and the other explicitly. Base ﬂux-splitting 
work has been developed under diﬀerent forms and for diﬀerent models (Euler equations [53–56], Baer-Nunziato model [57], Kapila 
model [58]). From these studies, semi-implicit schemes were developed in conservative form [39,59–67].
Our work derives the reformulation of the Zha-Bilgen (ZB) ﬂux-splitting [54] proposed by Toro and Vazquez [55] for the Euler 
equations, now applied to the velocity equilibrium model. No ﬂux-splitting has been proposed for this model so far: by being based on 
the ZB splitting, this novel ﬂux-splitting fully decouples the advection and acoustic properties of the initial two-phase ﬂow model. This 
allows for independent resolution of each subsystem. In addition, the model is developed with a general equation of state expression, 
to remain applicable to as many conﬁgurations as possible. Based on this novel ﬂux-splitting, we propose a semi-implicit method. 
On one hand, the pressure subsystem is solved implicitly: this removes the acoustic time step limitation, as an implicit method is 
Journal of Computational Physics 547 (2026) 114545 
2 


S. Schropﬀ, F. Petitpas and E. Daniel
unconditionally stable. Picard iterations are used in order to remove the non-linearity of the subsystem and therefore simplifying the 
numerical resolution. Additionally, a semi-discrete collocated 1D formulation is used in order to keep a compact stencil, in the idea 
of [63,64]. On the other hand, the advection subsystem is solved explicitly: this maintains a stability condition on the scheme based 
on the material velocity of the ﬂow, but much less restrictive than the former. Finally, the internal energies are corrected as in [20] 
to ensure total energy conservation. The solution we propose proves to be eﬀective and eﬃcient on various 1D test cases, recovering 
a wide range of compressible dynamic ﬂows.
2.  Two-phase ﬂow model and mathematical properties
In this part, the governing equations of a single velocity two-phase ﬂow are reminded as well as their main mathematical properties. 
This model can be derived from the model of Baer & Nunziato [14] as it was ﬁrst presented in [15]. The formulation presented in 
[19] is here retained.
A splitting of the model is then proposed in order to decouple the convective and acoustic properties of the model. The basis of 
this splitting is developed in [54,55] for the Euler equations.
2.1.  Flow equations and properties
The compressible ﬂow model retained in the study is well suited for ﬂows evolving with a single velocity. The governing equations 
are written thereafter:
⎧
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪⎩
𝜕𝑡(𝛼1) + u ⋅∇𝛼1 = 𝜇(𝑝1 −𝑝2)
𝜕𝑡(𝛼2) + u ⋅∇𝛼2 = −𝜇(𝑝1 −𝑝2)
𝜕𝑡(𝛼1𝜌1) + ∇⋅(𝛼1𝜌1u) = 0
𝜕𝑡(𝛼2𝜌2) + ∇⋅(𝛼2𝜌2u) = 0
𝜕𝑡(𝛼1𝜌1𝑒1) + ∇⋅(𝛼1𝜌1𝑒1u) + 𝛼1𝑝1∇⋅u = −𝜇𝑝𝐼(𝑝1 −𝑝2)
𝜕𝑡(𝛼2𝜌2𝑒2) + ∇⋅(𝛼2𝜌2𝑒2u) + 𝛼2𝑝2∇⋅u = 𝜇𝑝𝐼(𝑝1 −𝑝2)
𝜕𝑡(𝜌u) + ∇⋅(𝜌u ⊗u) + ∇𝑝= 0
(1)
where 𝛼𝑘 represents the volume fraction, 𝜌𝑘 the speciﬁc mass, 𝑝𝑘 the pressure and 𝑒𝑘 the internal speciﬁc energy related to phase 𝑘. 
The saturation condition is denoted by ∑
𝑘𝛼𝑘= 1.
For the mixture, we deﬁne 𝜌= ∑
𝑘𝛼𝑘𝜌𝑘 the mixture speciﬁc mass, 𝑝= ∑
𝑘𝛼𝑘𝑝𝑘 the mixture pressure and 𝑒= ∑
𝑘𝑌𝑘𝑒𝑘 the mixture 
speciﬁc internal energy, with 𝑌𝑘= 𝛼𝑘𝜌𝑘∕𝜌 the mass fraction. The velocity u is common to both phases and the total energy is 𝐸=
𝑒+ 1
2u2.
Combining the internal energy equations of each phase with the momentum equation recovers the evolution equation of total 
energy 𝐸 in a conservative form. The total energy equation is therefore:
𝜕𝑡(𝜌𝐸) + ∇⋅((𝜌𝐸+ 𝑝)u) = 0.
(2)
Each ﬂuid is governed by its own convex equation of state (EOS), 𝑒𝑘= 𝑒𝑘(𝑝𝑘, 𝜌𝑘). The general expression for the speed of sound is 
the following:
𝑐2
𝑘=
𝑝𝑘
𝜌2
𝑘𝑒𝑝,𝑘
−
𝑒𝜌,𝑘
𝑒𝑝,𝑘
,
(3)
with partial derivatives deﬁned as:
𝑒𝑝,𝑘= 𝜕𝑒𝑘
𝜕𝑝𝑘
)
𝜌𝑘
and
𝑒𝜌,𝑘= 𝜕𝑒𝑘
𝜕𝜌𝑘
)
𝑝𝑘
.
(4)
In the present paper, the Stiﬀened-Gas (SG) EOS [27] is chosen to describe the ﬂow behavior, as it is well suited to model both 
liquids and gases; but the method can easily be extended to other forms of EOS (i.e. Noble-Abel Stiﬀened-Gas [68] or Mie-Grüneisen 
[69,70]). For the SG EOS, the internal energy is deﬁned as:
𝑒𝑘(𝑝𝑘, 𝜌𝑘) =
𝑝𝑘+ 𝛾𝑘𝑝𝑘,∞
(𝛾𝑘−1)𝜌𝑘
+ 𝑒𝑘,𝑟𝑒𝑓.
(5)
The speed of sound is given by:
𝑐2
𝑘=
𝛾𝑘(𝑝𝑘+ 𝑝𝑘,∞)
𝜌𝑘
,
(6)
with 𝛾𝑘, 𝑝𝑘,∞ and 𝑒𝑘,𝑟𝑒𝑓 parameters determined using reference thermodynamic curves [71].
The mixture goes back to mechanical equilibrium thanks to relaxation terms in the right hand-side of System (1), depending on 
the parameter 𝜇, which is referred as pressure relaxation coeﬃcient. The choice of this parameter depends on the physical problem 
under study and it can be chosen between 0 and +∞ [20].
Journal of Computational Physics 547 (2026) 114545 
3 


S. Schropﬀ, F. Petitpas and E. Daniel
The entropy equations corresponding to System (1) are the following:
⎧
⎪
⎨
⎪⎩
𝛼1𝜌1𝑇1
𝑑(𝑠1)
𝑑𝑡
= (𝑝1 −𝑝𝐼)𝜇(𝑝1 −𝑝2)
𝛼2𝜌2𝑇2
𝑑(𝑠2)
𝑑𝑡
= −(𝑝2 −𝑝𝐼)𝜇(𝑝1 −𝑝2)
(7)
with 𝑠𝑘 the speciﬁc entropy of phase 𝑘 and 𝑇𝑘 its temperature. The interfacial pressure 𝑝𝐼 is obtained as the asymptotic limit of 
the interfacial pressure of the symmetric non-equilibrium model with 7-equations of [72]. In order to ensure entropy production in 
each phase, a possible choice for this interfacial pressure can be 𝑝𝐼= 𝑍2𝑝1+𝑍1𝑝2
𝑍1+𝑍2
, where 𝑍𝑘= 𝜌𝑘𝑐𝑘 is the acoustic impedance of phase 
k [73,74].
Model (1) is hyperbolic, with the following three wave speeds: 𝑢, 𝑢−𝑐𝑓 and 𝑢+ 𝑐𝑓 (a detailed mathematical anaysis of this model 
can be found in [75]). The mixture sound speed is called the frozen sound speed, which is monotonic with respect to the volume 
fraction. It is deﬁned as:
𝑐2
𝑓=
∑
𝑘
𝑌𝑘𝑐2
𝑘.
(8)
2.2.  Splitting convective and acoustic parts
We now propose a new ﬂux-splitting of System (1) in order to have fully decoupled pressure and advection properties. The core 
idea was proposed by Toro and Vazquez for the Euler equations [55], who developed a Godunov-like reformulation of the Zha-Bilgen
(ZB) splitting [54]. Based on this reformulation and from System (1), the novel splitting is done by regrouping the ﬂuxes as well as 
the non-conservative terms related to the pressure eﬀects in the ﬁrst subsystem, named pressure subsystem. The remaining advection 
terms are regrouped in the second subsystem, the advection subsystem.
The equations are presented for a two-phase ﬂow, the extension to Π phases is straightforward. The relaxation terms are also 
omitted because they are not relevant to the following mathematical analysis.
The vector of primitive variables is deﬁned as:
W = (𝛼1, 𝛼2, 𝜌1, 𝜌2, 𝑝1, 𝑝2, u)𝑇
.
(9)
Pressure subsystem. The pressure subsystem is ﬁrst considered:
⎧
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪⎩
𝜕𝑡(𝛼1) = 0
𝜕𝑡(𝛼2) = 0
𝜕𝑡(𝛼1𝜌1) = 0
𝜕𝑡(𝛼2𝜌2) = 0
𝜕𝑡(𝛼1𝜌1𝑒1) + 𝛼1𝑝1∇⋅u = 0
𝜕𝑡(𝛼2𝜌2𝑒2) + 𝛼2𝑝2∇⋅u = 0
𝜕𝑡(𝜌u) + ∇𝑝= 0
.
(10)
The corresponding system of equations for the primitive variables evolution reads: 𝜕𝑡W + Q𝑝(W)∇⋅W = 0, with W the primitive 
vector deﬁned in equation (9) and the propagation matrix Q𝑝 deﬁned as:
Q𝑝(W) =
⎛
⎜
⎜
⎜
⎜
⎜
⎜
⎜
⎜
⎜⎝
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
𝑝1Γ1
0
0
0
0
0
0
𝑝2Γ2
𝑝1
𝜌
𝑝2
𝜌
0
0
𝛼1
𝜌
𝛼2
𝜌
0
⎞
⎟
⎟
⎟
⎟
⎟
⎟
⎟
⎟
⎟⎠
.
(11)
The eigenvalues of the propagation matrix Q𝑝 are:
𝜆= (0, 0, 0, 0, 0, −𝐴, 𝐴)𝑇,
(12)
with
𝐴2 = 1
𝜌
∑
𝑘
𝛼𝑘𝑝𝑘Γ𝑘.
(13)
As one can notice, the wave speeds ±𝐴 do not depend on material speed u and therefore the wave pattern is always subsonic. The 
Grüneisen parameter [76] for phase 𝑘 is Γ𝑘= (𝜌𝑘𝑒𝑝,𝑘)−1. If 𝑝𝑘Γ𝑘> 0, System (10) is hyperbolic, as in the speciﬁc case of the SG EOS, 
where the wave speed is:
𝐴2 =
∑
𝑘
𝛼𝑘𝑝𝑘(𝛾𝑘−1) = 𝑐2
𝑓−
𝑝+ ∑
𝑘𝛼𝑘𝛾𝑘𝑝𝑘,∞
𝜌
,
(14)
Journal of Computational Physics 547 (2026) 114545 
4 


S. Schropﬀ, F. Petitpas and E. Daniel
with 𝑐𝑓 the frozen sound speed (8), 𝜌= ∑
𝑘𝛼𝑘𝜌𝑘 the mixture density and 𝑝= ∑
𝑘𝛼𝑘𝑝𝑘 the mixture pressure. In the case of the Ideal 
Gas EOS, one simply has 𝐴2 = 𝑐2
𝑓−𝑝∕𝜌.
The sum of speciﬁc internal energies coupled with the momentum and mass equations leads to the mixture total energy equation 
for the pressure subsystem:
𝜕𝑡(𝜌𝐸) + ∇⋅(𝑝u) = 0.
(15)
Advection subsystem. Secondly, we study the advection subsystem written as:
⎧
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪⎩
𝜕𝑡(𝛼1) + ∇⋅(𝛼1u) −𝛼1∇⋅u = 0
𝜕𝑡(𝛼2) + ∇⋅(𝛼2u) −𝛼2∇⋅u = 0
𝜕𝑡(𝛼1𝜌1) + ∇⋅(𝛼1𝜌1u) = 0
𝜕𝑡(𝛼2𝜌2) + ∇⋅(𝛼2𝜌2u) = 0
𝜕𝑡(𝛼1𝜌1𝑒1) + ∇⋅(𝛼1𝜌1𝑒1u) = 0
𝜕𝑡(𝛼2𝜌2𝑒2) + ∇⋅(𝛼2𝜌2𝑒2u) = 0
𝜕𝑡(𝜌u) + ∇⋅(𝜌u ⊗u) = 0
.
(16)
The recombination of systems (10) and (16) recovers the whole system (1), without relaxations terms. Compared to this system, 
note that the evolution equations for 𝛼𝑘 are written in a diﬀerent but equivalent form.
With the same vector of primitive variables W deﬁned in Eq. (9), we obtain the following system: 𝜕𝑡W + Q𝑎(W)∇⋅W = 0, with
Q𝑎(W) =
⎛
⎜
⎜
⎜
⎜
⎜
⎜
⎜
⎜⎝
u
0
0
0
0
0
0
0
u
0
0
0
0
0
0
0
u
0
0
0
𝜌1
0
0
0
u
0
0
𝜌2
0
0
0
0
u
0
−𝑒𝜌,1𝜌2
1Γ1
0
0
0
0
0
u
−𝑒𝜌,2𝜌2
2Γ2
0
0
0
0
0
0
u
⎞
⎟
⎟
⎟
⎟
⎟
⎟
⎟
⎟⎠
.
(17)
The eigenvalues of the propagation matrix Q𝑎 are:
𝜆= (u, u, u, u, u, u, u)𝑇.
(18)
The sum of speciﬁc internal energies coupled with the momentum and mass equations leads to the mixture total energy equation 
for the advection subsystem:
𝜕𝑡(𝜌𝐸) + ∇⋅(𝜌𝐸u) = 0.
(19)
The mixture total energy Eq. (2) of the complete model (1) is recovered by summing Eqs. (15) and (19): this equation is redundant 
and thus useless in this mathematical analysis, but is recalled for both subsystems as it will be useful in the numerical method.
3.  Classical explicit ﬁnite volume scheme
Numerous numerical solutions of System (1) can be found in literature, based on explicit schemes (most often ﬁnite volume 
schemes). The main purpose of reminding the reader of the following classic numerical scheme is that it will serve as a reference 
for the various comparisons with the new method, which will focus on the numerical solutions of the equations, as well as on 
computational performance. Moreover, as the method described hereunder is rewritten in the formalism of our paper, this will help 
the understanding of the new method proposed later.
We recall brieﬂy in this part the method presented in [20] for solving the hyperbolic part of System (1) (without relaxation terms). 
This hyperbolic system of equations is written in general form for a multiphase ﬂow of 𝑘 phases, and is augmented by the conservative 
equation of mixture total energy: indeed, the computation of this quantity will be useful to provide a correction to the numerical 
solution of the non-conservative equations for the phases internal energy:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝜕𝑡(𝛼𝑘) + ∇⋅(𝛼𝑘u) −𝛼𝑘∇⋅u = 0
𝜕𝑡(𝛼𝑘𝜌𝑘) + ∇⋅(𝛼𝑘𝜌𝑘u) = 0
𝜕𝑡(𝛼𝑘𝜌𝑘𝑒𝑘) + ∇⋅(𝛼𝑘𝜌𝑘𝑒𝑘u) + 𝛼𝑘𝑝𝑘∇⋅u = 0
𝜕𝑡(𝜌u) + ∇⋅(𝜌u ⊗u) + ∇𝑝= 0
𝜕𝑡(𝜌𝐸) + ∇⋅((𝜌𝐸+ 𝑝)u) = 0
.
(20)
When compactly rewritten, System (20) yields:
𝜕𝑡U + ∇⋅F(U) + H(U)∇⋅u = 0.
(21)
Journal of Computational Physics 547 (2026) 114545 
5 


S. Schropﬀ, F. Petitpas and E. Daniel
We denote U as the conservative vector, F as the ﬂux vector and H as the non-conservative vector, all completed by the total 
energy equation:
U =
⎛
⎜
⎜
⎜
⎜
⎜⎝
𝛼𝑘
𝛼𝑘𝜌𝑘
𝛼𝑘𝜌𝑘𝑒𝑘
𝜌u
𝜌𝐸
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
F =
⎛
⎜
⎜
⎜
⎜
⎜⎝
𝛼𝑘u
𝛼𝑘𝜌𝑘u
𝛼𝑘𝜌𝑘𝑒𝑘u
𝜌u ⊗u + 𝑝I
(𝜌𝐸+ 𝑝)u
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
H =
⎛
⎜
⎜
⎜
⎜
⎜⎝
−𝛼𝑘
0
𝛼𝑘𝑝𝑘
0
0
⎞
⎟
⎟
⎟
⎟
⎟⎠
.
(22)
As System (20) remains non-conservative, we follow the numerical method proposed in [20]: it consists of a prediction step, 
followed by a correction step to ensure total energy conservation.
Prediction step
The time evolution of the solution is ﬁrst predicted between time steps (𝑛) and (𝑛+ 1). This is done by using an explicit ﬁnite 
volume method, which in 1D is:
U𝑝𝑟𝑒𝑑
𝑖
= U𝑛
𝑖−Δ𝑡
Δ𝑥
(
F∗
𝑖+ 1
2
−F∗
𝑖−1
2
)
−Δ𝑡
Δ𝑥H𝑛
𝑖
(
𝑢∗
𝑖+ 1
2
−𝑢∗
𝑖−1
2
)
,
(23)
where (⋅)𝑖 denotes the mesh cell spatial index and Δ𝑥 is the mesh cell size. The intercell ﬂuxes F∗
𝑖±1∕2 and velocities 𝑢∗
𝑖±1∕2 are determined 
using the solution of a Riemann solver. Practically, an approximate HLLC Riemann solver [19,77] is used.
The numerical time-step is limited by the Courant-Friedrichs Lewy (CFL) criterion of stability. In explicit numerical methods, the 
stability of the scheme is given by the critical time-step:
Δ𝑡∗= 𝑡𝑛+1 −𝑡𝑛= Δ𝑥
|𝜎| ,
(24)
where in this case, 𝜎= max |𝑢± 𝑐𝑓|.
Due to the non-conservative form of internal speciﬁc energies equations in (22), the numerical scheme (23) cannot guaranty 
energy conservation for the mixture as is and it has to be supplemented by a correction as detailed hereafter.
Energy correction
In this model, the pressure of each phase is obtained from their respective internal energy equation. Unfortunately, these equations 
are not conservative and therefore the total energy conservation is not guaranteed. To circumvent this issue, a correction is performed 
with the help of the total energy equation solution, which is solved in a conservative way [20]. Assuming that the conservative 
variables as well as the volume fraction are correctly calculated by the scheme during the prediction step, we can make the following 
statements:
(𝛼𝑘)𝑛+1 = (𝛼𝑘)𝑝𝑟𝑒𝑑,
(𝛼𝑘𝜌𝑘)𝑛+1 = (𝛼𝑘𝜌𝑘)𝑝𝑟𝑒𝑑,
(𝜌𝑢)𝑛+1 = (𝜌𝑢)𝑝𝑟𝑒𝑑,
(𝜌𝐸)𝑛+1 = (𝜌𝐸)𝑝𝑟𝑒𝑑.
Then, a correction is performed to ensure that the sum of the internal energies of the phases is equal to the total mixture energy, 
through the following method:
(𝛼𝑘𝜌𝑘𝑒𝑘)𝑛+1 = (𝛼𝑘𝜌𝑘𝑒𝑘)𝑝𝑟𝑒𝑑+ 𝛼𝑛
𝑘𝛿𝜀
,
(25)
with 𝛿𝜀 the correction factor computed as
𝛿𝜀= 𝛿𝜌𝐸−𝛿1
2 𝜌𝑢2 −
∑
𝑘
𝛿𝑝𝑟𝑒𝑑
𝛼𝑘𝜌𝑘𝑒𝑘,
(26a)
with
𝛿𝜌𝐸= (𝜌𝐸)𝑛+1 −(𝜌𝐸)𝑛,
(26b)
𝛿1
2 𝜌𝑢2 = 1
2
(
((𝜌𝑢)𝑛+1)2
𝜌𝑛+1
−((𝜌𝑢)𝑛)2
𝜌𝑛
)
,
(26c)
𝛿𝑝𝑟𝑒𝑑
𝛼𝑘𝜌𝑘𝑒𝑘= (𝛼𝑘𝜌𝑘𝑒𝑘)𝑝𝑟𝑒𝑑−(𝛼𝑘𝜌𝑘𝑒𝑘)𝑛.
(26d)
4.  A semi-implicit (IMEX) scheme
The original numerical method presented in Section 3 has proven its eﬃciency and robustness for the numerical simulation 
of compressible two-phase ﬂows in velocity equilibrium (see [20]). However, it becomes very restrictive when the ﬂow is weakly 
compressible or low-Mach. Indeed, in such case, the material velocity and frozen speed of sound can diﬀer greatly in magnitude. As 
Journal of Computational Physics 547 (2026) 114545 
6 


S. Schropﬀ, F. Petitpas and E. Daniel
the stability criterion (24) shows, the time-step inversely depends on the speed of sound. If 𝑢≪𝑐𝑓, the time step drops and the explicit 
scheme becomes extremely ineﬃcient and inaccurate: ineﬃcient due to the extremely small value of the time-step, and inaccurate 
due to the associated numerical dissipation. Moreover, the discretized solution of the compressible equations does not always provide 
an accurate approximation of the incompressible equations; an asymptotic analysis at low Mach number in the incompressible limit 
of the equilibrium velocity model (20) is available in Appendix A.Therefore, we propose a new method to overcome these issues for 
the velocity-equilibrium model.
In the same way as for the complete system of Eq. (21), the hyperbolic system composed of the split Eqs. (10) and (16) is also 
augmented by the split total energy Eqs. (15) and (19). It is written in a vector form:
𝜕𝑡U + ∇⋅(P(U) + A(U)) + (H𝑝(U) + H𝑎(U))∇⋅u = 0.
(27)
Again, the conservative vector is U. The original ﬂux F is split into a pressure ﬂux vector P and an advection ﬂux vector A. The 
non-conservative vector H is also split into two pressure and advection non-conservative vectors, H𝑝 and H𝑎. As all are completed by 
the total energy equation, they are deﬁned as:
U =
⎛
⎜
⎜
⎜
⎜
⎜⎝
𝛼𝑘
𝛼𝑘𝜌𝑘
𝛼𝑘𝜌𝑘𝑒𝑘
𝜌u
𝜌𝐸
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
P =
⎛
⎜
⎜
⎜
⎜
⎜⎝
0
0
0
𝑝I
𝑝u
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
A =
⎛
⎜
⎜
⎜
⎜
⎜⎝
𝛼𝑘u
𝛼𝑘𝜌𝑘u
𝛼𝑘𝜌𝑘𝑒𝑘u
𝜌u ⊗u
𝜌𝐸u
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
H𝑝=
⎛
⎜
⎜
⎜
⎜
⎜⎝
0
0
𝛼𝑘𝑝𝑘
0
0
⎞
⎟
⎟
⎟
⎟
⎟⎠
,
H𝑎=
⎛
⎜
⎜
⎜
⎜
⎜⎝
−𝛼𝑘
0
0
0
0
⎞
⎟
⎟
⎟
⎟
⎟⎠
.
(28)
A novel ﬂux-splitting scheme inspired by ﬂux-splitting numerical schemes for the Euler equations [54,55] is proposed in Ap-
pendix B. It slightly alleviates the original time-step restriction depending on |𝑢± 𝑐𝑓|, as the new acoustic wave-speed |𝐴| (14) does 
not depend on the material speed 𝑢. But as this wave-speed still depends on the acoustic properties of the ﬂow, the time step remains 
restricted in the case where the mixture sound speed is quite high.
Both complete and split explicit numerical schemes thus suﬀer from the same drawbacks due to the acoustic features of the 
models. Therefore, based on the numerical ﬂux-splitting decomposition (27), we now develop a semi-implicit scheme. Using a fully 
implicit scheme would lead to a strongly non-linear resolution, involving large matrixes and therefore a counter-productive numerical 
resolution in terms of eﬃciency. The purpose of the semi-implicit scheme is to remove the acoustic restriction by implicitly solving 
the corresponding subsystem, while keeping the advection restriction based on the ﬂow velocity by using an explicit method on the 
latter.
In this regard, the time evolution of the solution between times (𝑛) and (𝑛+ 1) is carried out by applying the following operators:
U𝑛+1 = [][][](U𝑛).
(29)
This procedure, known as “operator splitting” applies independently each operator from right to left, during a time-step Δ𝑡=
𝑡𝑛+1 −𝑡𝑛. Each operator solves a speciﬁc part of the model:
• the implicit operator is deﬁned as U𝑛+⋆= [](U𝑛) (see (37)),
• the explicit operator is deﬁned as U𝑝𝑟𝑒𝑑= [](U𝑛+⋆) = [][](U𝑛) (see (46)),
• the energy correction operator is deﬁned as U𝑛+1 = [](U𝑝𝑟𝑒𝑑) = [][][](U𝑛) (see (49)).
The notations (⋅)𝑛+⋆ and (⋅)𝑝𝑟𝑒𝑑 denote numerical intermediate time-steps, which are not physical but deﬁne multiple stages of the 
operators.
The implicit operator is based on the determination of pressure equations for each phase, which is done by combining the momen-
tum equation and the internal energy equations. The internal energy is then reformulated as a function of pressure, which enables 
an extension of the scheme to a general equation of state, to allow for a wider range of applications. A semi-discrete collocated 1D 
formulation is used in order to keep a compact stencil, in the idea of [63,64]. The explicit operator uses a usual explicit ﬁnite volume 
approach, which will yield the critical time step, with a Rusanov ﬂux and a second-order scheme [39,78]. As the decomposition of 
equations presented in (27) remains non-conservative, the correction operator will also follow the numerical method proposed in 
[20] to ensure total energy conservation, as in the classical ﬁnite volume scheme presented in Section 3.
4.1.  Pressure subsystem: Implicit operator []
We seek to solve the pressure subsystem implicitly in order to remove the time-step restriction on the acoustic waves. The pressure 
subsystem written in compact form is:
𝜕𝑡U + ∇⋅P(U) + H𝑝(U)∇⋅u = 0
(30)
with U, P and H𝑝 deﬁned in Eq. (28).
In the following method, we take advantage of the fact that the solutions for volume fraction and mass are trivial, and combine 
the remaining equations (energy and momentum) as well as the equation of state. This yields a pressure equation evolution for each 
phase, which is solved implicitly. Its solution is included in an iterative system to obtain the velocity and the energy for each phase.
Therefore, the pressure equations are ﬁrst determined and implicitly solved. Then, we will present the corresponding implicit 
numerical scheme.
Journal of Computational Physics 547 (2026) 114545 
7 


S. Schropﬀ, F. Petitpas and E. Daniel
4.1.1.  Determination of the pressure equations
We remind the pressure subsystem written in primitive form (see propagation matrix (11)). This system groups the equations for 
each phase 𝑘 (volume fractions 𝛼𝑘, speciﬁc masses 𝜌𝑘 and pressures 𝑝𝑘) along with the mixture velocity u:
⎧
⎪
⎪
⎨
⎪
⎪⎩
𝜕𝑡(𝛼𝑘) = 0
𝜕𝑡(𝜌𝑘) = 0
𝜕𝑡(𝑝𝑘) + 𝑝𝑘Γ𝑘∇⋅u = 0
𝜕𝑡(u) + 1
𝜌
∑
𝑘
(𝑝𝑘∇𝛼𝑘+ 𝛼𝑘∇𝑝𝑘
) = 0
.
(31)
When implicitly discretized in time between (𝑛) and (𝑛+ ⋆), System (31) reads:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝛼𝑛+⋆
𝑘,𝑖
= 𝛼𝑛
𝑘,𝑖
𝜌𝑛+⋆
𝑘,𝑖
= 𝜌𝑛
𝑘,𝑖
𝑝𝑛+⋆
𝑘,𝑖
= 𝑝𝑛
𝑘,𝑖−Δ𝑡(𝑝𝑘,𝑖Γ𝑘,𝑖
)𝑛+⋆∇⋅u𝑛+⋆
𝑖
u𝑛+⋆
𝑖
= u𝑛
𝑖−Δ𝑡
1
𝜌𝑛+⋆
𝑖
∑
𝑘
(𝑝𝑘,𝑖∇𝛼𝑘,𝑖+ 𝛼𝑘,𝑖∇𝑝𝑘,𝑖
)𝑛+⋆
.
(32)
On one hand, the ﬁrst two equations of System (32) on the volume fractions and the speciﬁc masses show that these variables do 
not change during the implicit time-step.
On the other hand, the pressure equation for phase 𝑘 is non-linear: it is dependent on the EOS (5) (Γ𝑛+⋆
𝑘
= Γ(𝑒𝑛+⋆
𝑝,𝑘) = Γ(𝑝𝑛+⋆
𝑘
, 𝜌𝑛
𝑘)), 
and the velocity depends on the pressure (u𝑛+⋆= u(𝑝𝑛+⋆
𝑘
)). Therefore, we use a Picard iteration technique, as in [61,63,79]: it is an 
iterative method that reduces the original strongly non-linear system to a mildly non-linear one. For a given Picard iteration, the 
values of some of the variables are set constant in order to simplify the resolution.
Therefore, in this idea, we denote some variables with the exponent (𝑟), corresponding to values known at a given Picard iteration. 
The unknown variables that are determined implicitly at that iteration are denoted with the exponent (𝑟+ 1).
Given the two previous points, System (32) becomes:
⎧
⎪
⎨
⎪⎩
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
= 𝑝𝑛
𝑘,𝑖−Δ𝑡(𝑝𝑘,𝑖Γ𝑘,𝑖
)𝑛+⋆,𝑟∇⋅u𝑛+⋆,𝑟+1
𝑖
u𝑛+⋆,𝑟+1
𝑖
= u𝑛
𝑖−Δ𝑡1
𝜌𝑛
𝑖
∑
𝑘
(
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
∇𝛼𝑛
𝑘,𝑖+ 𝛼𝑛
𝑘,𝑖∇𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
)
.
(33)
By inserting the velocity equation into the pressure equation, and not yet spatially discretizing ∇⋅u in order to keep a compact 
stencil [64], we obtain the following pressure equation for each phase 𝑘:
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
−(Δ𝑡)2𝜈𝑛+⋆,𝑟
𝑘,𝑖
Π
∑
𝑙=1
(
𝜂𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖−1
+ 𝜁𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖
+ 𝛽𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖+1
)
= 𝑝𝑛
𝑘,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
𝑘,𝑖
∇⋅u𝑛
𝑖.
(34)
The full mathematical process as well as the deﬁnition of coeﬃcients 𝜈, 𝜂, 𝜁 and 𝛽 is provided in Appendix C, alongside their 1D 
discretization.
In the case of a two-phase ﬂow (Π = 2), we have the following pressure equations: 
(𝑘= 1) ∶
𝑎𝑛+⋆,𝑟
1,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖−1
+ 𝑏𝑛+⋆,𝑟
1,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖−1
+ 𝑎𝑛+⋆,𝑟
2,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖
+𝑏𝑛+⋆,𝑟
2,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖
+ 𝑎𝑛+⋆,𝑟
3,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖+1
+ 𝑏𝑛+⋆,𝑟
3,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖+1
= 𝑐𝑛+⋆,𝑟
𝑖
,
(35a)
(𝑘= 2) ∶
𝑚𝑛+⋆,𝑟
1,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖−1
+ 𝑞𝑛+⋆,𝑟
1,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖−1
+ 𝑚𝑛+⋆,𝑟
2,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖
+𝑞𝑛+⋆,𝑟
2,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖
+ 𝑚𝑛+⋆,𝑟
3,𝑖
𝑝𝑛+⋆,𝑟+1
1,𝑖+1
+ 𝑞𝑛+⋆,𝑟
3,𝑖
𝑝𝑛+⋆,𝑟+1
2,𝑖+1
= 𝑔𝑛+⋆,𝑟
𝑖
.
(35b)
The diﬀerent parameters involved, 𝑎, 𝑏, 𝑐, 𝑚, 𝑞 and 𝑔 are also detailed in Appendix C.
We solve the following heptadiagonal linear system of size 2𝑁 (𝑁 being the number of mesh cells):
A𝑛+⋆,𝑟p𝑛+⋆,𝑟+1 = B𝑛+⋆,𝑟,
(36)
with A𝑛+⋆,𝑟 the matrix of coeﬃcients 𝑎, 𝑏, 𝑚 and 𝑞, p𝑛+⋆,𝑟+1 the vector of unknown phase pressures 𝑝𝑛+⋆,𝑟+1
1
 and 𝑝𝑛+⋆,𝑟+1
2
, and B𝑛+⋆,𝑟 the 
right hand-side vector containing 𝑐 and 𝑔. To solve System (36), a Gauss-Seidel method is used and improved for banded matrixes.
4.1.2.  Implicit numerical scheme
The implicit operator U𝑛+⋆= [](U𝑛) is detailed: it is an implicitly-solved 1D solution based on a ﬁnite volume scheme applied on 
the pressure subsystem (30):
U𝑛+⋆
𝑖
= U𝑛
𝑖−Δ𝑡
Δ𝑥
(
P∗,𝑛+⋆
𝑖+ 1
2
−P∗,𝑛+⋆
𝑖−1
2
)
−Δ𝑡
Δ𝑥H𝑛+⋆
𝑝,𝑖
(
𝑢∗,𝑛+⋆
𝑖+ 1
2
−𝑢∗,𝑛+⋆
𝑖−1
2
)
.
(37)
Journal of Computational Physics 547 (2026) 114545 
8 


S. Schropﬀ, F. Petitpas and E. Daniel
The pressure ﬂux vector P∗,𝑛+⋆
𝑖+1∕2 and the non-conservative vector H𝑛+⋆
𝑝,𝑖 are deﬁned in (28), whereas 𝑢∗,𝑛+⋆
𝑖+1∕2 is the intercell velocity. 
They are implicitly evaluated with the method described thereafter.
The ﬁrst step is to solve System (36) to obtain the updated pressures, thanks to an iterative Picard algorithm. For the initialization 
of the Picard iterations (𝑟= 1), we use 𝑝𝑛+⋆,1
𝑘
= 𝑝𝑛
𝑘 and 𝑒𝑛+⋆,1
𝑝,𝑘
= 𝑒𝑛
𝑝,𝑘. If p𝑛 is solution of System (36), it is not necessary to update the 
solution.
If not, by solving System (36), we determine the pressure of each phase 𝑝𝑛+⋆,𝑟+1
𝑘
. Then, the momentum of the pressure subsys-
tem (30) can be updated:
(𝜌𝑢)𝑛+⋆,𝑟+1
𝑖
= (𝜌𝑢)𝑛
𝑖−Δ𝑡
Δ𝑥
∑
𝑘𝛼𝑛
𝑘,𝑖+1𝑝𝑛+⋆,𝑟+1
𝑘,𝑖+1
−∑
𝑘𝛼𝑛
𝑘,𝑖−1𝑝𝑛+⋆,𝑟+1
𝑘,𝑖−1
2
.
(38)
We then update the following variables, 𝑒𝑛+⋆,𝑟+1
𝑝,𝑘
= 𝑒𝑝,𝑘(𝑝𝑛+⋆,𝑟+1
𝑘
, 𝜌𝑛
𝑘), by using the EOS (5), and then move on to the next Picard 
iteration.
The iterative process is carried out until the standard deviation between the pressure of two consecutive Picard iterations is below a 
given tolerance, or a maximum number of iterations has been reached. Then, the mixture pressure can be updated as 𝑝𝑛+⋆= ∑
𝑘𝛼𝑛
𝑘𝑝𝑛+⋆
𝑘
.
After the convergence of the iterative process, the energies are updated. The internal speciﬁc energy for each phase k is estimated 
as follows:
(𝛼𝑘𝜌𝑘𝑒𝑘)𝑛+⋆
𝑖
= (𝛼𝑘𝜌𝑘𝑒𝑘)𝑛
𝑖−Δ𝑡
Δ𝑥𝛼𝑛
𝑘,𝑖𝑝𝑛+⋆
𝑘,𝑖
(
𝑢∗,𝑛+⋆
𝑖+ 1
2
−𝑢∗,𝑛+⋆
𝑖−1
2
)
.
(39)
The intercell velocity is simply computed from the intercell mixture mass and momentum, which can be explicitly written as:
𝑢∗,𝑛+⋆
𝑖+ 1
2
=
(𝜌𝑢)𝑛+⋆
𝑖
+ (𝜌𝑢)𝑛+⋆
𝑖+1
𝜌𝑛
𝑖+ 𝜌𝑛
𝑖+1
.
(40)
The total mixture energy can also be computed using the following scheme:
(𝜌𝐸)𝑛+⋆
𝑖
= (𝜌𝐸)𝑛
𝑖−Δ𝑡
Δ𝑥
(
(𝑝𝑢)𝑛+⋆
𝑖+ 1
2
−(𝑝𝑢)𝑛+⋆
𝑖−1
2
)
.
(41)
We recall the mixture pressure deﬁnition below:
𝑝𝑛+⋆
𝑖
=
∑
𝑘
𝛼𝑛
𝑘,𝑖𝑝𝑛+⋆
𝑘,𝑖.
(42)
The intercell ﬂux is evaluated as follows, allowing pressure and velocity to be preserved across a contact discontinuity:
(𝑝𝑢)𝑛+⋆
𝑖+ 1
2
= 𝑝𝑛+⋆
𝑖+ 1
2
𝑢𝑛+⋆
𝑖+ 1
2
(43)
with 𝑢𝑛+⋆
𝑖+ 1
2
 as in Eq. (40) and 𝑝𝑛+⋆
𝑖+ 1
2
= (𝑝𝑛+⋆
𝑖
+ 𝑝𝑛+⋆
𝑖+1 )∕2.
With this formulation, the “checkerboard problem” (non-physical numerical oscillations of speciﬁc quantities) [80–82] was not 
encountered in the various test cases under study. These oscillations can occur when pressure and velocity are deﬁned at the same 
location (mesh center). Various authors suggest that the checkerboard problem may not appear in 1D because it is dissipated by the 
1D discretization, but could manifest itself in a multidimensional study [22,78,83]. Others suggest that pressure relaxation mitigates 
the phenomenon [15,84].
In summary, after having determined the pressure, the implicit scheme is the following: 
U𝑛+⋆
𝑖
=
⎧
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪⎩
𝛼𝑛+⋆
𝑘,𝑖
= 𝛼𝑛
𝑘,𝑖
(𝛼𝑘𝜌𝑘)𝑛+⋆
𝑖
= (𝛼𝑘𝜌𝑘)𝑛
𝑖
(𝛼𝑘𝜌𝑘𝑒𝑘)𝑛+⋆
𝑖
= (𝛼𝑘𝜌𝑘𝑒𝑘)𝑛
𝑖−Δ𝑡
Δ𝑥𝛼𝑛
𝑘,𝑖𝑝𝑛+⋆
𝑘,𝑖
(
𝑢∗,𝑛+⋆
𝑖+ 1
2
−𝑢∗,𝑛+⋆
𝑖−1
2
)
(𝜌𝑢)𝑛+⋆
𝑖
= (𝜌𝑢)𝑛
𝑖−Δ𝑡
Δ𝑥
∑
𝑘𝛼𝑛
𝑘,𝑖+1𝑝𝑛+⋆
𝑘,𝑖+1 −∑
𝑘𝛼𝑛
𝑘,𝑖−1𝑝𝑛+⋆
𝑘,𝑖−1
2
(𝜌𝐸)𝑛+⋆
𝑖
= (𝜌𝐸)𝑛
𝑖−Δ𝑡
Δ𝑥
(
(𝑝𝑢)𝑛+⋆
𝑖+ 1
2
−(𝑝𝑢)𝑛+⋆
𝑖−1
2
)
(44)
4.2.  Advection subsystem: Explicit operator []
We now seek to solve the advection subsystem explicitly. Due to the explicit nature of this part of the scheme, a stability criterion 
remains, as deﬁned in Eq. (24). However, it is now only restricted by an advection stability condition, with 𝜎= max |𝑢|. This will be 
much less restrictive than the former acoustic stability condition, especially with low-Mach or weakly compressible ﬂows.
We recall the advection subsystem written under compact form:
𝜕𝑡U + ∇⋅A(U) + H𝑎(U)∇⋅u = 0
(45)
with U, A and H𝑎 deﬁned in Eq. (28).
Journal of Computational Physics 547 (2026) 114545 
9 


S. Schropﬀ, F. Petitpas and E. Daniel
We apply the explicit operator U𝑝𝑟𝑒𝑑= [](U𝑛+⋆) on the advection subsystem (45). This operator completes the prediction step ; in 
the same way as for the usual fully explicit scheme (see Section 3), we will then correct the wrongly estimated internal energies.
The advection subsystem is explicitly solved in 1D with a ﬁnite volume scheme:
U𝑝𝑟𝑒𝑑
𝑖
= U𝑛+⋆
𝑖
−Δ𝑡
Δ𝑥
(
A∗,𝑛+⋆
𝑖+ 1
2
−A∗,𝑛+⋆
𝑖−1
2
)
−Δ𝑡
Δ𝑥H𝑛+⋆
𝑎,𝑖
(
𝑢∗,𝑛+⋆
𝑖+ 1
2
−𝑢∗,𝑛+⋆
𝑖−1
2
)
.
(46)
The intercell advection ﬂux A∗
𝑖+1∕2 is given by a Rusanov ﬂux:
A∗,𝑛+⋆
𝑖+ 1
2
= 1
2
(
A(U𝑛+⋆,+
𝑖+ 1
2
) + A(U𝑛+⋆,−
𝑖+ 1
2
)
)
−1
2|𝜎|
(
U𝑛+⋆,+
𝑖+ 1
2
−U𝑛+⋆,−
𝑖+ 1
2
)
(47)
where U𝑛+⋆,−
𝑖+1∕2  and U𝑛+⋆,+
𝑖+1∕2  denote the left and right boundary extrapolated states at the cell interface 𝑥𝑖+1∕2. For a ﬁrst-order scheme, 
one simply has U𝑛+⋆,−
𝑖+1∕2 = U𝑛+⋆
𝑖
 and U𝑛+⋆,+
𝑖+1∕2 = U𝑛+⋆
𝑖+1 . As the Rusanov ﬂux is quite dissipative, we perform this explicit step with a 
second-order MUSCL-Hancock-type TVD scheme [39,78]. The details of the second-order scheme can be found in Appendix D.
The intercell velocity comes directly from the implicit scheme:
𝑢∗,𝑛+⋆
𝑖+1∕2 = (𝑢𝑛+⋆,−
𝑖+1∕2 + 𝑢𝑛+⋆,+
𝑖+1∕2 )∕2
with 𝑢𝑛+⋆,±
𝑖+1∕2 = (𝜌𝑢)𝑛+⋆,±
𝑖+1∕2 ∕𝜌𝑛+⋆,±
𝑖+1∕2 .
The magnitude of the material wave speed 𝜎 is: 
𝜎= 𝐷× max(|𝑢𝑛+⋆,−
𝑖+1∕2 |, |𝑢𝑛+⋆,+
𝑖+1∕2 |).
(48)
We use 𝐷≥1, a coeﬃcient introducing some additional dissipation into the explicit scheme to prevent the appearance of undesired 
oscillations and maintain a stable resolution through the diﬀerent operators.
The time-step Δ𝑡 that is used in the explicit numerical scheme (46) is the same as the one used in the implicit numerical scheme. 
For the next time step of scheme, Δ𝑡 is deduced from the critical time-step Δ𝑡∗ (24), but now based on the value of the material wave 
speed 𝜎 (48).
4.3.  Energy correction: Operator []
Finally, we apply the correction operator U𝑛+1 = [](U𝑝𝑟𝑒𝑑). Again, the discretization of mass, momentum and mixture speciﬁc total 
energy equations are reliable in terms of conservation, as well as the volume fraction (46). However, the internal energy equations are 
wrongly estimated as they include non-conservative terms. Based on [20] and in the same way as for the fully explicit scheme (25), 
to ensure total energy conservation, we correct the internal energy between (𝑛) and (𝑛+ 1).
Given these considerations, we obtain the following numerical scheme:
U𝑛+1
𝑖
=
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
(𝛼𝑘)𝑛+1
𝑖
= (𝛼𝑘)𝑝𝑟𝑒𝑑
𝑖
(𝛼𝑘𝜌𝑘)𝑛+1
𝑖
= (𝛼𝑘𝜌𝑘)𝑝𝑟𝑒𝑑
𝑖
(𝜌𝑢)𝑛+1
𝑖
= (𝜌𝑢)𝑝𝑟𝑒𝑑
𝑖
(𝜌𝐸)𝑛+1
𝑖
= (𝜌𝐸)𝑝𝑟𝑒𝑑
𝑖
(𝛼𝑘𝜌𝑘𝑒𝑘)𝑛+1
𝑖
= (𝛼𝑘𝜌𝑘𝑒𝑘)𝑝𝑟𝑒𝑑
𝑖
+ 𝛼𝑛
𝑘,𝑖𝛿𝜀
.
(49)
The correction term 𝛿𝜀 computation is detailed in equation (26) and all predicted values come from the semi-implicit prediction 
scheme (46).
Finally, the vector U𝑛+1 is fully updated, from the semi-implicit prediction scheme, the correct estimation of conservative quantities 
and correction of the internal energies.
4.4.  Summary of the semi-implicit scheme
Fig. 1 is a functional diagram which recaps the multiple steps of the new IMEX scheme used to solve the hyperbolic system (27) 
through the multiple operators deﬁned in Eq. (29).
The pressure relaxation that is performed after the IMEX scheme in order to ensure pressure equilibrium is fully independent and 
details of the numerical method are given in [19,20].
5.  Numerical results
In this section, the ability of the new numerical scheme to solve classical ﬂow problems is proven on various 1D test cases. 
Single-phase ﬂows are ﬁrst considered and then two-phase ﬂows with diﬀerent ﬂuids. Shock tubes, interfaces, as well as mixture ﬂow 
problems will be addressed.
The ﬂuids are governed by the Stiﬀened Gas equation of state (5) and are diﬀerentiated by their respective EOS parameters (see 
Table 1 for the parameters used in the test cases). The SG EOS also includes the Ideal Gas EOS, when 𝑝∞ and 𝑒𝑟𝑒𝑓 are null.
Journal of Computational Physics 547 (2026) 114545 
10 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 1. Functional diagram of the IMEX scheme (𝜀= 10−6).
Table 1 
Stiﬀened gas parameters used in the test cases.
𝛾 (-)
𝑝∞ (Pa)
𝑒𝑟𝑒𝑓 (J/kg)
 Air
 1.4
 0
 0
 Water
 4.4
6 ⋅108
 0
 Epoxy
 2.43
5.3 ⋅109
 0
 Spinel
 1.62
141 ⋅109
 0
For each test case, the physical domain is a pipe of length 𝐿= 1 m with constant section. The mesh is composed of 𝑁𝑥= 𝐿∕Δ𝑥
cells of size Δ𝑥. Computations are performed on two diﬀerent meshes: 200 cells or 1000 cells.
The time step used for the IMEX scheme is Δ𝑡𝐼𝑀𝐸𝑋= 0.5 × Δ𝑥∕|𝜎|, with 𝜎 as in Eq. (48). The dissipation coeﬃcient 𝐷 that might 
be used in the explicit step is kept constant for a given test case, whatever the mesh reﬁnement, and is speciﬁed in the test case tables 
presented thereafter.
The initial condition is a Riemann problem: we specify the state of the ﬂow on the left (𝐿) and on the right (𝑅) of a given abscissa 
𝑥𝑖𝑛𝑡𝑒𝑟. The solutions obtained with the IMEX scheme are systematically compared to the analytical solutions to prove the quality of 
the results.
One reminds that this new IMEX method is primarily useful to circumvent the low-Mach transport problem and the stability 
condition that comes from it. Its computational performance is compared to that of a classical ﬁnite volume solver based on an 
explicit Godunov scheme (see Section 3) ; the latter uses an HLLC approximate Riemann solver [77] and is also performed with 
a second-order MUSCL-Hancock-type scheme. The comparisons that will be presented hereunder will therefore concern the most 
critical test cases, which are indeed related to interface transport problem at low-Mach number. Nevertheless, as the next subsection 
will present, our new method is also able to compute accurately and eﬃciently shock tube solutions. A full comparison between the 
semi-implicit and the explicit scheme performances is proposed in Appendix E.
Journal of Computational Physics 547 (2026) 114545 
11 


S. Schropﬀ, F. Petitpas and E. Daniel
Table 2 
Initial conﬁgurations of single-phase ﬂow test cases.
𝑡𝑓
𝜌𝐿
𝑢𝐿
𝑝𝐿
𝑥𝑖𝑛𝑡𝑒𝑟
𝜌𝑅
𝑢𝑅
𝑝𝑅
𝐷
 RP
 (s)
 (kg/m3)
 (m/s)
 (Pa)
 (m)
 (kg/m3)
 (m/s)
 (Pa)
 (-)
 Low-Mach air advection
 0.5
 1000
 1
105
 0.25
 0.01
 1
105
 1.1
 Low-Mach water shock tube
10−4
 1000
 0
108
 0.5
 1000
 15
0.98 ⋅108
 40
 Lax air shock tube
 0.14
 0.445
 0.698
 3.528
 0.5
 0.5
 0
 0.571
 1.5
 Strong air shock tube
2.5 ⋅10−6
 1
 0
1010
 0.5
 0.125
 0
 0.1
 1.2
 Two rarefaction fans
 0.15
 1
-1
 0.4
 0.5
 1
 1
 0.4
 1
5.1.  Single-phase ﬂow test cases
In this section, we propose a series of single-phase ﬂow test cases drawn from literature. Initial conditions for each test case are 
summarized in Table 2.
Low-Mach air advection. The ﬁrst Riemann problem (RP) under study is a low-Mach problem considering a density discontinuity 
advected at constant pressure and constant velocity. The initial density is a step-like function of ﬁve orders of magnitude.
Results are displayed in Fig. 2 where the evolutions of density, velocity, pressure and Mach number along the domain are plotted 
at the ﬁnal time 𝑡𝑓. These proﬁles are compared with the exact solution as well as the solution obtained with the explicit scheme (see 
Section 3).
The ﬂow is clearly a low-Mach one, characterized by a Mach number ranging from 2.7 ⋅10−4 to 8.5 ⋅10−2. The interface moves 
at 1 m/s. The density discontinuity is preserved, as well as the pressure and velocity which remain constant. The main diﬀerence 
between all the results is related to the diﬀusion of the interface. With the IMEX method, one can compare the eﬀect of the mesh size 
(200 and 1000 cells).
We observe that the IMEX method and the HLLC based solver results do not show the same numerical diﬀusion: the IMEX solution 
with 1000 mesh cells is more accurate than the corresponding explicit solution. That is due to the number of time steps that each 
method performs to reach the ﬁnal time 𝑡𝑓.
The time step ratio deﬁned as 𝑅= Δ𝑡𝐼𝑀𝐸𝑋∕Δ𝑡𝐻𝐿𝐿𝐶 is an important criterion to assess the computational performance, as it 
compares at the last iteration the semi-implicit time step Δ𝑡𝐼𝑀𝐸𝑋 and the explicit scheme time step Δ𝑡𝐻𝐿𝐿𝐶. When 𝑅> 1, the IMEX 
time step is larger than the explicit time step, and therefore there is a gain on the total amount of time-iterations performed during 
the numerical scheme to reach the physical ﬁnal time 𝑡𝑓.
The explicit time step is deﬁned as Δ𝑡𝐻𝐿𝐿𝐶= 0.9 × Δ𝑥∕(max |𝑢± 𝑐|) (24). For 1000 mesh cells, 𝑅= 1890, meaning that the explicit 
scheme will have to compute 𝑅 times the amount of time steps of the IMEX scheme, with a much smaller time step, leading in the 
end to a larger numerical diﬀusion and longer computation. As a result, with 1000 mesh cells, given the large acoustic waves and the 
low-Mach behavior, the explicit scheme computational time is approximately 𝑡𝐻𝐿𝐿𝐶= 513 s. For the IMEX scheme, the computational 
time is only 𝑡𝐼𝑀𝐸𝑋= 0.45 s.
Low-Mach water shock tube. The next Riemann problem is a low-Mach shock tube presented in [85]. The initial problem is a domain 
containing water (the EOS parameters are listed in Table 1), with a very slight discontinuity of both pressure and velocity.
Results are presented in Fig. 3. The discontinuities are correctly located and plateau values are also correct. Due to the high speed 
of sound of the medium, the Mach number is extremely low and reaches a maximum value equal to 8.6 ⋅10−3, while the interface 
moves at 8.07 m/s. Because dissipation is added to the material wave speed used during the explicit time integration, the proﬁle of 
the IMEX solution shows diﬀused discontinuities. Yet, it remains very accurate and when reﬁning the mesh, it converges towards the 
exact solution. This test case also demonstrates the ability of the IMEX solver to handle more complex EOS than just Ideal Gases, by 
taking into account the cohesive eﬀects represented by the non-zero parameter 𝑝∞.
Lax shock tube. The Lax shock tube is a benchmark in gas dynamics (here tested with air), used to study and check the robustness 
of numerical solvers handling discontinuities. Results are displayed in Fig. 4: the solution is discontinued by a right-moving shock 
wave, a left-moving rarefaction fan and an intermediate contact discontinuity moving at 1.53 m/s. The Mach number ranges from 0 
to 0.95, making it a suitable test case to check the behavior of the IMEX scheme for subsonic ﬂows ranging across a larger variation 
of Mach number.
The shock, interface and rarefaction fan propagate with the accurate velocity, the plateau values are correct, and the mesh 
reﬁnement leads to a convergence towards the exact solution. The behavior of the ﬂow is correct for the entire range of Mach 
number, demonstrating its robustness for low-Mach and subsonic ﬂows.
Fig. 5 is used an example to show the impact of the dissipation coeﬃcient 𝐷, varying from 𝐷= 1 (no dissipation) to 𝐷= 1.5 and 
𝐷= 2, with NX = 200. One can notice that without any numerical dissipation, there are spurious oscillations on the shock wave. 
This problem is immediatly mitigated when the coeﬃcient is increased to 𝐷= 1.5, and fully disappears with 𝐷= 2. Indeed, there is 
a slight loss of precision on the waves location (see for example the contact wave at 𝑥= 0.73 m), and moreover the time step will 
decrease (see Eq. (48)), but the results remain very accurate: the compromise between quality of results and stability of resolution is 
therefore satisfactory.
Strong shock tube. The next single-phase ﬂow test case comes from [86] and is referred as "strong shock tube": the initial problem 
involves a pressure discontinuity of air ranging over 11 orders of magnitude. The Mach number increases from 0 up to 1.9, with the 
Journal of Computational Physics 547 (2026) 114545 
12 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 2. Low-Mach air advection, results at 𝑡𝑓= 0.5 s. Left to right, top to bottom: density, velocity, pressure and Mach number. Each sub-ﬁgure 
displays a comparison between the analytical solution (black solid line) and the numerical solutions, both the IMEX solution with two diﬀerent 
meshing (solid colored lines) and the usual explicit scheme using a HLLC solver (green dashed line). (For interpretation of the references to colour 
in this ﬁgure legend, the reader is referred to the web version of this article.)
Fig. 3. Low-Mach water shock tube, results at 𝑡𝑓= 10−4 s.
Journal of Computational Physics 547 (2026) 114545 
13 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 4. Lax shock tube, results at 𝑡𝑓= 0.14 s.
Fig. 5. Lax shock tube: results of density for variable coeﬃcient 𝐷 (left), with a zoom on the contact and shock wave location (right) (NX = 200).
material wave moving at 118 ⋅103 m/s. This test case allows to check the behavior of the IMEX scheme for an even larger range of 
Mach number, including supersonic behavior.
The numerical solution in Fig. 6 shows a slight decrease of the density across the material interface: this is improved when the 
mesh is reﬁned. Again, the IMEX numerical solution displays a correct behavior: the shock, interface and rarefaction fan are correctly 
located and the plateau values are correct.
Two rarefaction fans. The last single-phase ﬂow test case addresses the capability of the solver to simulate two rarefaction 
fans, here with air. The initial problem is a simple discontinuity of velocity, going in two opposite directions at −1 m/s on 
the left of the interface, and 1 m/s on its right, thus generating a double rarefaction fan. The initial pressure and density are 
constant. The Mach number increases from 0 up to 1.34, a large interval useful to evaluate the behavior of the semi-implicit
scheme.
Fig. 7 shows the usual comparison between the diﬀerent numerical results and the exact solution: overall, they are almost indis-
tinguishable. The numerical solutions behave adequately, with a correct location of the rarefaction fans. As usual, the less reﬁned 
semi-implicit solution displays more diﬀusion. Diﬀerences in results are mostly noticeable at the central zone: non-physical numerical 
oscillations are created, which are mitigated by mesh reﬁnement.
Journal of Computational Physics 547 (2026) 114545 
14 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 6. Strong shock tube, results at 𝑡𝑓= 2.5 ⋅10−6 s.
5.1.1.  Comparison with another low Mach scheme
Up until now, low Mach schemes have usually been developed for the Euler equations or Baer-Nunziato equations; more rarely for 
the Kapila equations or the pressure-disequilibrium system. Therefore, Model (1) has never been solved with a semi-implicit scheme, 
and especially one based on the Zha-Bilgen ﬂux-splitting. Thus, it is diﬃcult to compare this new scheme with another of similar or 
equivalent class. However, some authors [8] propose either a preconditioned solver or a fully implicit scheme, which imply diﬀerent 
numerical constraints.
Next, we propose to compare the results of a test case selected from [8], who also develops a low-Mach scheme for the pressure-
disequilibrium model, but with a fully implicit and preconditioned approach. The test case deals with a liquid-liquid shock tube, 
where the domain is a pipe of 𝐿= 1 m containing two chambers. They are separated by an interface at the location 𝑥𝑖𝑛𝑡𝑒𝑟= 0.5 m and 
each chamber contains pure liquid water (see Table 1 for SG EOS water parameters) at an initial density of 𝜌𝐿= 𝜌𝑅= 1000kg/m3. 
The initial pressure in the left chamber is 𝑝𝐿= 106 Pa and the initial pressure in the right chamber is 𝑝𝑅= 105 Pa.
Fig. 8 compares the semi-implicit and implicit with low Mach preconditioning numerical solutions with the exact solution at a 
physical time equal to 𝑡= 0.15 ms. These computations are performed on a mesh composed of NX = 200 cells. Both numerical results 
show good wave propagation, predicting correct jumps through the shock and expansion waves, as well as the correct mean wave 
positions. However, the implicit scheme diﬀuses the shock and expansion waves a lot more than the semi-implicit scheme, which, 
in comparison, diﬀuses the contact wave more, due to the introduction of the dissipative coeﬃcient 𝐷 during the explicit advection 
resolution.
5.2.  Numerical results for two-phase ﬂows
We now perform test cases for two-phase ﬂows. All presented tests are performed using an instantaneous pressure relaxation 
procedure as described in [19]. In this speciﬁc case, one recovers the 5-equation Kapila model [15] for which exact solutions are 
available and used as reference.
In this regard, in order to study the Mach number 𝑀= |𝑢|∕𝑐, we must deﬁne the mixture sound speed corresponding to the Kapila 
model, which is the Wood sound speed [87]:
1
𝜌𝑐2 =
∑
𝑘
𝛼𝑘
𝜌𝑘𝑐2
𝑘
(50)
Table 3 lists the two-phase ﬂow test cases performed with the new IMEX scheme, which are to be compared to the exact solution 
and the explicit scheme with a HLLC solver for the critical cases. The SG EOS parameters can be found in Table 1.
Journal of Computational Physics 547 (2026) 114545 
15 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 7. Two rarefaction fans, results at 𝑡𝑓= 0.15 s.
Table 3 
Initial conﬁgurations of two-phase ﬂow test cases (𝜀𝛼= 10−7).
𝑡𝑓
𝛼1,𝐿
𝜌1,𝐿
𝜌2,𝐿
𝑢𝐿
𝑝𝐿
𝑥𝑖𝑛𝑡𝑒𝑟
𝛼1,𝑅
𝜌1,𝑅
𝜌2,𝑅
𝑢𝑅
𝑝𝑅
𝐷
 RP
 (s)
 (-)
 (kg/m3)  (kg/m3)
 (m/s)
 (Pa)  (m)
 (-)
 (kg/m3)
 (kg/m3)  (m/s)
 (Pa)  (-)
 Water(1)-air(2) advection
 0.25
1 −𝜀𝛼
 1000
 0.01
 1
105
 0.25
𝜀
 1000
 0.01
 1
105
 3.2
 Water(1)-air(2) shock tube
241 ⋅10−6
1 −𝜀𝛼
 1000
 50
 0
109
 0.7
𝜀
 1000
 50
 0
105
 2
 Epoxy(1)-spinel(2) shock tube
30 ⋅10−6
 0.5954  1185
 3622
 0
1010
 0.6
 0.5954  1185
 3622
 0
105
 6
Low-Mach water-air advection. The ﬁrst two-phase ﬂow under study is a discontinuity of volume fraction that separates nearly pure 
water and air. This discontinuity is moving in a uniform ﬂow, with a constant pressure of 1 Bar and a constant velocity of 1 m/s. The 
Mach number is around 2.7 ⋅10−3.
The numerical solution is shown in Fig. 9: one can observe the eﬀect of numerical diﬀusion when using approximate numerical 
schemes, compared to the exact solution. The uniformity of velocity and pressure are preserved, whereas the discontinuity is correctly 
advected. At the interface, the deﬁnition of the mechanical equilibrium mixture sound speed (50) creates an oscillation and an increase 
of the Mach number, up to 0.042. It is not a numerical oscillation and depends solely on the mechanical equilibrium model properties: 
because the Wood sound speed is non-monotonous and drops in the mixture, the Mach number increases at the interface between 
the pure phases, where the mesh contains the mixture cells. This issue also is detailed in [20] and underlines the necessity of using 
the pressure-disequilibrium model instead of the Kapila model.
When reﬁning the mesh, the numerical solution converges towards the exact solution. For 1000 mesh cells, the IMEX and explicit 
results are close, but the IMEX solution does appear slightly closer to the exact solution. Again, the time steps are not at all of the same 
magnitude: the time step ratio is 𝑅= 517.72, which explains the more diﬀused behavior of the explicit solution. The computational 
time for the semi-implicit scheme is 0.66 s for 1000 mesh cells, whereas for the explicit scheme, it is approximately 257.71 𝑠.
Water-air shock tube. The two-phase ﬂow studied in this part is taken from [20], adapted from [19]: a shock tube initially contains 
almost pure ﬂuids on each side of a water-air interface. The goal of this test case is to demonstrate the ability of the numerical scheme 
to handle high density jumps.
Numerical results are displayed in Fig. 10 and show an overall accurate solution. The Mach number goes from 0 to 0.26 to 1.84 
at most, while the interface moves at 482 m/s. On the density proﬁle, the shock and the interface seem to merge into one single 
wave, due to the numerical diﬀusion. The underestimation of the pressure at the tail of the rarefaction fan is typically observed in 
literature [20,75]; furthermore, the logarithmic scale used exaggerates the visualization of this ﬂuctuation. An additional meshing 
level (5000 cells) has been added to demonstrate that when the mesh is reﬁned, the spatial amplitude of the pressure drop is reduced.
Journal of Computational Physics 547 (2026) 114545 
16 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 8. Semi-implicit and implicit schemes comparison with a low Mach water shock tube, results at 𝑡𝑓= 0.15 ms.
Epoxy-spinel shock tube. This last test case from [19] considers the following initial conditions: the pipe is divided in two chambers, 
each containing the same mixture of epoxy and spinel, but at diﬀerent pressures. Once the ﬂow is triggered, the material interface 
moves at a speed of 550 m/s and the Mach number reaches at maximum 0.136.
Fig. 11 shows the numerical results for this test case: the discontinuities are correctly located and the plateau values are correct. 
One can notice that the mesh reﬁnement again leads to better results. One may ﬁnd more details on the convergence of multiphase 
ﬂow shock problems for non-conservative systems of equations in [19,70].
In general, the semi-implicit scheme shows good results and performance in the low-Mach case. To conclude this section and 
validate the convergence of numerical solutions, we carry out a convergence study depending on the mesh size NX = 𝐿∕Δ𝑥, presented 
in Fig. 12. We compute the 𝐿1 norm of the error between the numerical mixture mass 𝜌𝑛𝑢𝑚 and the analytical mixture mass 𝜌𝑎𝑛𝑎 at 
the ﬁnal time 𝑡𝑓, normalized by the 𝐿1 norm of the initial mixture mass, as proposed by [66]:
𝐿1,𝜌(NX) =
∑NX
𝑖=1 |𝜌𝑛𝑢𝑚
𝑖
(𝑡𝑓) −𝜌𝑎𝑛𝑎(𝑥𝑖, 𝑡𝑓)|Δ𝑥
∑NX
𝑖=1 |𝜌𝑎𝑛𝑎(𝑥𝑖, 𝑡0)|Δ𝑥
(51)
Fig. 12 indicates the various regression equations of the error as a function of mesh reﬁnement, of the type 𝐿1,𝜌(NX) = 𝑎NX𝑏, with 
𝑏∼−0.7.
Journal of Computational Physics 547 (2026) 114545 
17 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 9. Low-Mach water-air transport, results at 𝑡𝑓= 0.25 s. Left to right, top to bottom: mixture density, velocity, pressure, Mach number, volume 
fraction and density of water.
Journal of Computational Physics 547 (2026) 114545 
18 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 10. Water-air shock tube, results at 𝑡𝑓= 241 𝜇s.
Journal of Computational Physics 547 (2026) 114545 
19 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 11. Epoxy-spinel shock tube, results at 𝑡𝑓= 30 𝜇s.
Journal of Computational Physics 547 (2026) 114545 
20 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. 12. 𝐿1 norm of the mixture mass error 𝜌 as a function of mesh reﬁnement, for diﬀerent two-phase test cases.
6.  Conclusion
In conclusion, the main goal of this paper is reached: a semi-implicit numerical scheme has been developed from hyperbolic ﬂux-
splitting, in order to circumvent the low-Mach problem related to the usual acoustic stability criterion present in classical explicit 
numerical schemes. This method works for both single-phase ﬂows and two-phase ﬂows evolving with one velocity. Besides, the 
extension to Π phases (which is included in the theoretical development of the pressure equations) is straightforward. In order to 
remain applicable to as many ﬂuid conﬁgurations as possible, the scheme is developed with a general equation of state expression.
By using a semi-discrete spatial discretization, the stencil is kept compact, which leads to a banded matrix with a small width. 
The explicit part of the method is performed with a second-order scheme: the Rusanov ﬂux, which is quite dissipative, is improved 
with sharper interfaces. The numerical scheme is developed in 1D as it is largely suﬃcient to display the main characteristics of the 
solver. Extension to multiple dimensions will present no theoretical diﬃculties, apart from the eﬀective management of wider-band 
matrix solutions.
For both single phase and multiphase ﬂows, the new semi-implicit scheme has demonstrated its ability to simulate low-Mach ﬂows, 
which was the intended purpose, as well as ﬂows triggered by pressure discontinuities, with an accurate computation of interfaces, 
shocks and rarefaction fans.
In terms of computational eﬃciency, for critical low-Mach test cases, the time step ratio between the semi-implicit and explicit 
methods is very high due to the removal of the acoustic stability criterion: as predicted, the semi-implicit scheme generally demon-
strates a better performance. For other cases such as shock-tube problems, the gain is less noticeable but may still exist.
Future developments will aim to verify the asymptotically-preserving property of this novel semi-implicit scheme, by developing 
semi-implicit boundary conditions and extending the numerical scheme to multiple spatial dimensions in order to study steady-state 
cases.
CRediT authorship contribution statement
Solène Schropﬀ: Writing - review & editing, Writing - original draft, Visualization, Validation, Methodology, Investigation, Formal 
analysis, Data curation, Conceptualization; Fabien Petitpas: Writing - review & editing, Validation, Supervision, Resources, Project 
administration, Methodology, Investigation, Funding acquisition, Formal analysis, Conceptualization; Eric Daniel: Writing - review 
& editing, Validation, Supervision, Resources, Project administration, Funding acquisition, Formal analysis, Conceptualization.
Data availability
No data was used for the research described in the article.
Declaration of competing interest
The authors declare that they have no known competing ﬁnancial interests or personal relationships that could have appeared to 
inﬂuence the work reported in this paper. 
Journal of Computational Physics 547 (2026) 114545 
21 


S. Schropﬀ, F. Petitpas and E. Daniel
Appendix A.  Incompressible low-Mach number asymptotic analysis
The low-Mach number limit of the hyperbolic part of the velocity-equilibrium model (20) can be obtained by rescaling it in the 
following way:
̃𝛼𝑘= 𝛼𝑘
̃𝜌𝑘= 𝜌𝑘∕𝜌𝑟
̃𝑝𝑘= 𝑝𝑘∕(𝜌𝑟𝑐2
𝑓,𝑟)
̃𝑒𝑘= 𝑒𝑘∕𝑐2
𝑓,𝑟
̃u = u∕u𝑟
̃x = x∕x𝑟
̃𝑡= 𝑡∕𝑡𝑟
(A.1)
where variables denoted by (⋅)𝑟 are reference values typically encountered in the problems under study, with 𝑡𝑟= x𝑟∕u𝑟 the reference 
time linked to the ﬂuid velocity and 𝑐𝑓,𝑟 the frozen mixture sound speed (8).
We obtain the following mixture variables:
̃𝑝=
∑
𝑘
̃𝛼𝑘̃𝑝𝑘= 𝑝∕(𝜌𝑟𝑐2
𝑓,𝑟)
̃𝜌=
∑
𝑘
̃𝛼𝑘̃𝜌𝑘= 𝜌∕𝜌𝑟
̃𝑒=
∑
𝑘
(̃𝛼𝑘̃𝜌𝑘̃𝑒𝑘)∕̃𝜌= 𝑒∕𝑐2
𝑓,𝑟.
(A.2)
By deﬁning the reference Mach number 𝑀𝑟= 𝑢𝑟∕𝑐𝑓,𝑟, when the rescaling given by (A.1) and (A.2) is inserted in the hyperbolic 
model (20), it becomes the following dimensionless system:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝜕̃𝑡(̃𝛼𝑘) + ̃u ⋅∇̃x(̃𝛼𝑘) = 0
𝜕̃𝑡(̃𝛼𝑘̃𝜌𝑘) + ∇̃x ⋅(̃𝛼𝑘̃𝜌𝑘̃u) = 0
𝜕̃𝑡(̃𝛼𝑘̃𝜌𝑘̃𝑒𝑘) + ∇̃x ⋅(̃𝛼𝑘̃𝜌𝑘̃𝑒𝑘̃u) + ̃𝛼𝑘̃𝑝𝑘∇̃x ⋅̃u = 0
𝜕̃𝑡( ̃𝜌̃u) + ∇̃x ⋅( ̃𝜌̃u ⊗̃u) +
1
𝑀2
𝑟
∇̃x ̃𝑝= 0
.
(A.3)
To examine the zero-Mach number limit of this system of equations, the dimensionless variables (A.1) and (A.2) are expanded in 
terms of the reference Mach number:
⎧
⎪
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪
⎪⎩
̃𝛼𝑘(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝛼𝑘,0(̃x, ̃𝑡) + 𝑀𝑟̃𝛼𝑘,1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝛼𝑘,2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝜌𝑘(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝜌𝑘,0(̃x, ̃𝑡) + 𝑀𝑟̃𝜌𝑘,1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝜌𝑘,2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝑒𝑘(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝑒𝑘,0(̃x, ̃𝑡) + 𝑀𝑟̃𝑒𝑘,1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝑒𝑘,2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝑝𝑘(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝑝𝑘,0(̃x, ̃𝑡) + 𝑀𝑟̃𝑝𝑘,1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝑝𝑘,2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝜌(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝜌0(̃x, ̃𝑡) + 𝑀𝑟̃𝜌1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝜌2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝑝(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝑝0(̃x, ̃𝑡) + 𝑀𝑟̃𝑝1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝑝2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃𝑒(̃x, ̃𝑡, 𝑀𝑟)
= ̃𝑒0(̃x, ̃𝑡) + 𝑀𝑟̃𝑒1(̃x, ̃𝑡) + 𝑀2
𝑟̃𝑒2(̃x, ̃𝑡) + (𝑀3
𝑟)
̃u(̃x, ̃𝑡, 𝑀𝑟)
= ̃u0(̃x, ̃𝑡) + 𝑀𝑟̃u1(̃x, ̃𝑡) + 𝑀2
𝑟̃u2(̃x, ̃𝑡) + (𝑀3
𝑟)
.
(A.4)
Introducing the expansion of the dimensionless variables (A.4) in the dimensionless system of Eq. (A.3) leads to, when 𝑀𝑟→0:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝜕̃𝑡(̃𝛼𝑘,0) + ̃u0 ⋅∇̃x(̃𝛼𝑘,0) + (𝑀𝑟) = 0
𝜕̃𝑡(̃𝛼𝑘,0 ̃𝜌𝑘,0) + ∇̃x ⋅(̃𝛼𝑘,0 ̃𝜌𝑘,0 ̃u0) + (𝑀𝑟) = 0
𝜕̃𝑡(̃𝛼𝑘̃𝜌𝑘,0 ̃𝑒𝑘,0) + ∇̃x ⋅(̃𝛼𝑘,0 ̃𝜌𝑘,0 ̃𝑒𝑘,0 ̃u0) + ̃𝛼𝑘,0 ̃𝑝𝑘,0∇̃x ⋅̃u0 + (𝑀𝑟) = 0
𝜕̃𝑡( ̃𝜌0 ̃u0) + ∇̃x ⋅( ̃𝜌0 ̃u0 ⊗̃u0) + ∇̃x ̃𝑝2 +
1
𝑀2
𝑟
∇̃x ̃𝑝0 +
1
𝑀𝑟
∇̃x ̃𝑝1 + (𝑀𝑟) = 0
.
(A.5)
Nullifying the terms in 𝑀−1
𝑟 and 𝑀−2
𝑟 in the momentum equation of system (A.5) leads to:
∇̃x ̃𝑝0 = 0
and
∇̃x ̃𝑝1 = 0
.
(A.6)
As a consequence, the two variables ̃𝑝0 and ̃𝑝1 only depend on the dimensionless time variable ̃𝑡, which means that the pressure 
is constant in space up to the terms in 𝑀2
𝑟. As discussed in [88] in the context of barotropic Euler equations, in the case where the 
initial and boundary conditions are well-prepared [4,6,47], then ̃𝑝1 is uniformly equal to 0. Therefore, the asymptotic expansion of 
the pressure (A.5) becomes:
̃𝑝= ̃𝑝0(̃𝑡) + 𝑀2
𝑟̃𝑝2(̃x, ̃𝑡) + (𝑀3
𝑟)
.
(A.7)
From the asymptotic expansion of dimensionless mixture energy ̃𝑒 and velocity ̃u (A.4), associated to the deﬁnition of total mixture 
energy, the dimensionless total mixture energy can be written as:
̃𝐸= ̃𝑒+ 1
2𝑀2
𝑟̃u2
and thus
̃𝐸= ̃𝑒0 + 𝑀𝑟̃𝑒1 + 𝑀2
𝑟(̃𝑒2 + 1
2 ̃u2
0) + (𝑀3
𝑟)
.
(A.8)
It can directly be deduced that ̃𝐸0 = ̃𝑒0 = ∑
𝑘̃𝛼𝑘,0 ̃𝜌𝑘,0 ̃𝑒𝑘,0∕̃𝜌0. Finally, the total mixture energy equation is the following:
𝜕̃𝑡(𝜌̃𝐸) + ∇̃x ⋅(( ̃𝜌̃𝐸+ ̃𝑝)̃u) = 0
with
̃𝐸= 𝐸∕𝑐2
𝑓,𝑟
.
(A.9)
Using the previous relation ∇̃x ̃𝑝0 = 0 (A.6), the asymptotic expansion of the dimensionless total mixture energy Eq. (A.9) gives:
Journal of Computational Physics 547 (2026) 114545 
22 


S. Schropﬀ, F. Petitpas and E. Daniel
𝜕̃𝑡( ̃𝜌0 ̃𝑒0) + ∇̃x ⋅( ̃𝜌0 ̃𝑒0 ̃u0) + ̃𝑝0∇̃x ⋅̃u0 + (𝑀𝑟) = 0
.
(A.10)
By associating the expansion of the dimensionless internal energy of phase 𝑘 (A.4) and the general expression of the EOS, one 
notices that:
̃𝑒𝑘= ̃𝑒𝑘( ̃𝑝𝑘, ̃𝜌𝑘)
= ̃𝑒𝑘
( ̃𝑝𝑘,0 + 𝑀𝑟̃𝑝𝑘,1 + (𝑀2
𝑟), ̃𝜌𝑘,0 + 𝑀𝑟̃𝜌𝑘,1 + (𝑀2
𝑟))
= ̃𝑒𝑘,0 + 𝑀𝑟̃𝑒𝑘,1 + (𝑀2
𝑟),
(A.11)
with ̃𝑒𝑘,0 = ̃𝑒𝑘,0( ̃𝑝𝑘,0, ̃𝜌𝑘,0) and ̃𝑒𝑘,1 = ̃𝑒̃𝜌,𝑘,0 ̃𝜌𝑘,1 + ̃𝑒̃𝑝,𝑘,0 ̃𝑝𝑘,1. Thus, the internal energy equation of phase 𝑘 (A.5) can be re-written as an 
equation for the dimensionless pressure:
̃𝛼𝑘,0𝜕̃𝑡( ̃𝑝𝑘,0) + ̃𝛼𝑘,0 ̃u0 ⋅∇̃x ̃𝑝𝑘,0 + ̃𝛼𝑘,0 ̃𝜌𝑘,0 ̃𝑐2
𝑘,0∇̃x ⋅̃u0 = 0,
(A.12)
with ̃𝑐𝑘,0 = ( ̃𝑝𝑘,0∕̃𝜌2
𝑘,0 −̃𝑒̃𝜌,𝑘,0)∕̃𝑒̃𝑝,𝑘,0 the dimensionless speed of sound of phase 𝑘 at order 0 (3). When summing Eq. (A.12) on each 
phase 𝑘, introducing the volume fraction ̃𝛼𝑘,0 transport Eq. (A.5) and using the condition ∇̃x ̃𝑝0 = 0 (A.6), one obtains the following 
dimensionless mixture pressure equation:
𝜕̃𝑡( ̃𝑝0) + ̃𝜌0 ̃𝑐2
𝑓,0∇̃x ⋅̃u0 = 0,
(A.13)
with ̃𝑐2
𝑓,0 = ∑
𝑘̃𝛼𝑘,0 ̃𝜌𝑘,0 ̃𝑐2
𝑘,0∕̃𝜌0 the dimensionless frozen mixture sound speed at order 0 (8). As discussed in [22], the time variation 
depends on the source terms (which are neglected here) and on the boundary conditions: by considering the speciﬁc case of open 
boundaries, the pressure ̃𝑝0 is equal to the exterior pressure, which is in addition assumed constant in space and time. Therefore, 
Eq. (A.13) becomes:
∇̃x ⋅̃u0 = 0,
(A.14)
showing that the velocity ﬁeld ̃u0 is divergence-free in the zero-Mach number limit. In addition, by considering order 0 in the 
momentum equation, one recovers the incompressible equations:
{
∇̃x ⋅̃u0
= 0
̃𝜌0𝜕̃𝑡(̃u0) + ̃𝜌0(̃u0 ⋅∇̃x)̃u0 + ∇̃x ̃𝑝2
= 0
.
(A.15)
Because of the condition ∇̃x ⋅̃u0 = 0 (A.14), system (A.5) yields for each phase 𝑘:
⎧
⎪
⎨
⎪⎩
𝜕̃𝑡(̃𝛼𝑘,0) + ̃u0 ⋅∇̃x(̃𝛼𝑘,0)
= 0
𝜕̃𝑡( ̃𝜌𝑘,0) + ̃u0 ⋅∇̃x( ̃𝜌𝑘,0)
= 0
𝜕̃𝑡( ̃𝑝𝑘,0) + ̃u0 ⋅∇̃x( ̃𝑝𝑘,0)
= 0
(A.16)
which represents a transport scheme for the volume fraction, mass and pressure of each phase 𝑘.
Appendix B.  Explicit ﬂux-splitting numerical scheme
We brieﬂy present the numerical scheme for the explicit ﬂux-splitting introduced in Section 4. Explicit ﬂuxes are described, as 
well as the structure and solution of the Riemann problem.
The split sub-models (10) and (16) present the same non conservative terms as the previous complete model (20). Therefore, the 
numerical solution uses the same prediction and correction steps. The solution is ﬁrst estimated using the following scheme, which 
determines the time evolution between time steps (𝑛) and predicted (𝑛+ 1):
U𝑝𝑟𝑒𝑑
𝑖
= U𝑛
𝑖−Δ𝑡
Δ𝑥
((
P∗
𝑖+ 1
2
+ A∗
𝑖+ 1
2
)
−
(
P∗
𝑖−1
2
+ A∗
𝑖−1
2
))
−Δ𝑡
Δ𝑥H𝑛
𝑖
(
𝑢∗
𝑖+ 1
2
−𝑢∗
𝑖−1
2
)
.
(B.1)
The diﬀerent vectors are deﬁned in (28). The intercell ﬂuxes P∗, A∗ as well as velocity 𝑢∗ are determined thanks to solutions of 
subsystems (10) and (16) as presented below.
Pressure ﬂux. We seek for the solution of a Riemann problem at intercell (⋅)𝑖+1∕2 for System (B.1), between left cell (⋅)𝑖 (also denoted 
as (⋅)𝐿) and right cell (⋅)𝑖+1 (also denoted as (⋅)𝑅). The structure of the Riemann problem corresponding to the pressure subsystem is 
depicted in Fig. B.1.
The pressure ﬂux deﬁned in (28) is the following:
P∗
𝑖+ 1
2
= (0, 0, 0, 𝑝∗, 𝑝∗𝑢∗)𝑇.
(B.2)
Volume fraction, speciﬁc masses and internal energies do not vary across acoustic waves 𝜎= {−𝐴; 𝐴} (see (12)). The Rankine-
Hugoniot relations for the conservative part are used: P∗= P + 𝜎(U∗−U). This leads to expressions for 𝑝∗ and 𝑢∗: 
𝑢∗= (𝑝𝐿−𝑚𝐿𝑢𝐿) −(𝑝𝑅−𝑚𝑅𝑢𝑅)
𝑚𝑅−𝑚𝐿
(B.3a)
Journal of Computational Physics 547 (2026) 114545 
23 


S. Schropﬀ, F. Petitpas and E. Daniel
Fig. B.1. Structure of the Riemann problem for the pressure subsystem: it includes two acoustic wave groups ±𝐴 (depicted) and a stationary contact 
discontinuity aligned with the t-axis (not depicted). The wave pattern is always subsonic, and the solution focuses on determining the values 𝑢∗ and 
𝑝∗ in the star (*) region.
𝑝∗= 𝑚𝑅(𝑝𝐿−𝑚𝐿𝑢𝐿) −𝑚𝐿(𝑝𝑅−𝑚𝑅𝑢𝑅)
𝑚𝑅−𝑚𝐿
(B.3b)
with 𝑚𝐿= 𝜌𝐿𝜎𝐿 and 𝑚𝑅= 𝜌𝑅𝜎𝑅. The minimum and maximum wave speeds 𝜎𝐿 and 𝜎𝑅 are estimated as in [89], such as 𝜎𝐿=
min(−𝐴𝐿, −𝐴𝑅) and 𝜎𝑅= max(𝐴𝐿, 𝐴𝑅).
Advection ﬂux. The proposed solution for the advection ﬂux deﬁned in (28) is:
A∗
𝑖+ 1
2
= 𝑢∗
𝑖+ 1
2
⎧
⎪
⎨
⎪⎩
U𝑛
𝑖
if 𝑢∗
𝑖+ 1
2
> 0
U𝑛
𝑖+1
else
(B.4)
with 𝑢∗
𝑖+1∕2 the intercell advection velocity. We use the intercell velocity 𝑢∗
𝑖+1∕2 emerging from the solution (B.3a), like proposed in 
the TV splitting [55].
Summary of the ﬂux-splitting scheme. The full algorithm to determine the numerical solution of System (20) with the present ﬂux-
splitting scheme (B.1) is the following:
• Pressure ﬂux. Evaluate the intercell pressure 𝑝∗
𝑖+1∕2 and velocity 𝑢∗
𝑖+1∕2 from the solution of the Riemann problem given in (B.3), 
to compute the pressure ﬂux P∗
𝑖+1∕2 described in Eq. (B.2).
• Advection ﬂux. Evaluate the advection ﬂux A∗
𝑖+1∕2 as proposed in (B.4).
• Predicted solution. Estimate the vector U𝑝𝑟𝑒𝑑 as in (B.1). The intercell velocity 𝑢∗
𝑖+1∕2 comes from Eq. (B.3a).
• Corrected solution. Update the ﬁnal solution U𝑛+1 by applying the usual energy correction detailed in Section 3.
Appendix C.  Coeﬃcients of implicit subsystem
We develop thereafter the mathematical process to deﬁne the pressure Eq. (34) from System (33). First, we insert the velocity 
equation in the pressure equation:
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
= 𝑝𝑛
𝑘,𝑖−Δ𝑡(𝑝𝑘,𝑖Γ𝑘,𝑖
)𝑛+⋆,𝑟
[
∇⋅u𝑛
𝑖−Δ𝑡∇⋅
∑
𝑘
(∇𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
+
𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
∇𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
)]
.
We deﬁne the following coeﬃcient:
𝜈𝑛+⋆,𝑟
𝑘,𝑖
=
1
𝜌𝑛
𝑘,𝑖
( 𝑝𝑘,𝑖
𝑒𝑝,𝑘,𝑖
)𝑛+⋆,𝑟
= (𝑝𝑘,𝑖Γ𝑘,𝑖
)𝑛+⋆,𝑟.
(C.1)
Therefore, with a slight rearrangement, we obtain:
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
−(Δ𝑡)2𝜈𝑛+⋆,𝑟
𝑘,𝑖
∑
𝑘
∇⋅
( ∇𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
+
𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
∇𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
)
= 𝑝𝑛
𝑘,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
𝑘,𝑖
∇⋅u𝑛
𝑖.
By developing the sum and product derivatives, the previous equation becomes:
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
−(Δ𝑡)2𝜈𝑛+⋆,𝑟
𝑘,𝑖
∑
𝑘
[𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
∇⋅∇𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
+
(
∇
( 𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
)
+
∇𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
)
⋅∇𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
+ ∇⋅
(∇𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
)
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
]
= 𝑝𝑛
𝑘,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
𝑘,𝑖
∇⋅u𝑛
𝑖.
The gradient and Laplace operators are discretized as follows in 1D:
⎧
⎪
⎨
⎪⎩
∇𝑝𝑘,𝑖=
𝑝𝑘,𝑖+1 −𝑝𝑘,𝑖−1
2Δ𝑥
∇⋅∇𝑝𝑘,𝑖=
𝑝𝑘,𝑖+1 −2𝑝𝑘,𝑖+ 𝑝𝑘,𝑖−1
(Δ𝑥)2
.
Journal of Computational Physics 547 (2026) 114545 
24 


S. Schropﬀ, F. Petitpas and E. Daniel
By inserting this discretization in the pressure equation and rearranging the pressure terms by spatial index, one obtains Eq. (34):
𝑝𝑛+⋆,𝑟+1
𝑘,𝑖
−(Δ𝑡)2𝜈𝑛+⋆,𝑟
𝑘,𝑖
Π
∑
𝑙=1
(
𝜂𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖−1
+ 𝜁𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖
+ 𝛽𝑛
𝑙,𝑖𝑝𝑛+⋆,𝑟+1
𝑙,𝑖+1
)
= 𝑝𝑛
𝑘,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
𝑘,𝑖
∇⋅u𝑛
𝑖
with the following coeﬃcients: 
𝜂𝑛
𝑘,𝑖= 1
𝜌𝑛
𝑖
[ 𝛼𝑛
𝑘,𝑖
(Δ𝑥)2 −
1
2Δ𝑥
(
2∇𝛼𝑛
𝑘,𝑖−
𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
∇𝜌𝑛
𝑖
)]
,
(C.2a)
𝜁𝑛
𝑘,𝑖= 1
𝜌𝑛
𝑖
[
∇⋅∇𝛼𝑛
𝑘,𝑖−
∇𝛼𝑛
𝑘,𝑖⋅∇𝜌𝑛
𝑖
𝜌𝑛
𝑖
−2
𝛼𝑛
𝑘,𝑖
(Δ𝑥)2
]
,
(C.2b)
𝛽𝑛
𝑘,𝑖= 1
𝜌𝑛
𝑖
[ 𝛼𝑛
𝑘,𝑖
(Δ𝑥)2 +
1
2Δ𝑥
(
2∇𝛼𝑛
𝑘,𝑖−
𝛼𝑛
𝑘,𝑖
𝜌𝑛
𝑖
∇𝜌𝑛
𝑖
)]
.
(C.2c)
For a two-phase ﬂow, the coeﬃcients of Eq. (35) are the following:
⎧
⎪
⎨
⎪⎩
𝑎𝑛+⋆,𝑟
1,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝜂𝑛
1,𝑖
𝑎𝑛+⋆,𝑟
2,𝑖
= 1 −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝜁𝑛
1,𝑖
𝑎𝑛+⋆,𝑟
3,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝛽𝑛
1,𝑖
,
⎧
⎪
⎨
⎪⎩
𝑏𝑛+⋆,𝑟
1,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝜂𝑛
2,𝑖
𝑏𝑛+⋆,𝑟
2,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝜁𝑛
2,𝑖
𝑏𝑛+⋆,𝑟
3,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
1,𝑖
𝛽𝑛
2,𝑖
,
𝑐𝑛+⋆,𝑟
𝑖
= 𝑝𝑛
1,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
1,𝑖
∇⋅u𝑛
𝑖
,
⎧
⎪
⎨
⎪⎩
𝑚𝑛+⋆,𝑟
1,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝜂𝑛
1,𝑖
𝑚𝑛+⋆,𝑟
2,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝜁𝑛
1,𝑖
𝑚𝑛+⋆,𝑟
3,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝛽𝑛
1,𝑖
,
⎧
⎪
⎨
⎪⎩
𝑞𝑛+⋆,𝑟
1,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝜂𝑛
2,𝑖
𝑞𝑛+⋆,𝑟
2,𝑖
= 1 −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝜁𝑛
2,𝑖
𝑞𝑛+⋆,𝑟
3,𝑖
= −(Δ𝑡)2𝜈𝑛+⋆,𝑟
2,𝑖
𝛽𝑛
2,𝑖
,
𝑔𝑛+⋆,𝑟
𝑖
= 𝑝𝑛
2,𝑖−Δ𝑡𝜈𝑛+⋆,𝑟
2,𝑖
∇⋅u𝑛
𝑖.
In 1D, the discretization of the spatial operators is the following:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
∇𝛼𝑘,𝑖=
𝛼𝑘,𝑖+1 −𝛼𝑘,𝑖−1
2Δ𝑥
∇⋅∇𝛼𝑘,𝑖=
𝛼𝑘,𝑖+1 −2𝛼𝑘,𝑖+ 𝛼𝑘,𝑖−1
(Δ𝑥)2
∇𝜌𝑖= 𝜌𝑖+1 −𝜌𝑖−1
2Δ𝑥
∇⋅u𝑖= 𝑢𝑖+1 −𝑢𝑖−1
2Δ𝑥
.
(C.3)
Appendix D.  Second-order MUSCL-Hancock scheme applied to the advection subsystem
The second-order MUSCL-Hancock-type TVD scheme from [78] is recalled with the notations of our method, for a 1D scheme.
Step 1: Data reconstruction
The vector of primitive variables W𝑛+⋆= (𝛼𝑛+⋆
1
, 𝛼𝑛+⋆
2
, 𝜌𝑛+⋆
1
, 𝜌𝑛+⋆
2
, 𝑝𝑛+⋆
1
, 𝑝𝑛+⋆
2
, 𝑢𝑛+⋆)T is deduced from the implicit scheme.
In the data reconstruction step, data cell average values W𝑛+⋆
𝑖
 are locally replaced by a piece-wise linear function of slope Δ𝑖 in 
each cell (𝑖) bounded by the 1D coordinates [𝐿≡𝑥𝑖−1∕2, 𝑅≡𝑥𝑖+1∕2]. The vector W𝑛+⋆
𝑖
 is extrapolated and computed at the boundaries 
of cell (𝑖) in the following way:
W𝑛+⋆
𝑖,𝐿
= W𝑛+⋆
𝑖
−1
2 Δ𝑖
,
W𝑛+⋆
𝑖,𝑅= W𝑛+⋆
𝑖
+ 1
2Δ𝑖
.
(D.1)
The slope Δ𝑖 is also a vector: it contains the slope of each primitive variable, computed with a minmod limiter in our case [90].
Step 2: Half-time prediction
From the extrapolated primitive vectors W𝑛+⋆
𝑖,𝐿 and W𝑛+⋆
𝑖,𝑅, the extrapolated conservative vectors U(W𝑛+⋆
𝑖,𝐿), U(W𝑛+⋆
𝑖,𝑅) and advection 
ﬂux vectors A(U𝑛+⋆
𝑖,𝐿), A(U𝑛+⋆
𝑖,𝑅) can be computed (see (28) for expressions of U(W) and A(U)).
For each cell (𝑖), the boundary extrapolated values are evolved to a half-time Δ𝑡∕2, according to:
U
𝑛+⋆
𝑖
= U𝑛+⋆
𝑖
−Δ𝑡∕2
Δ𝑥
(
A(U𝑛+⋆
𝑖,𝑅) −A(U𝑛+⋆
𝑖,𝐿)
)
−Δ𝑡∕2
Δ𝑥H𝑛+⋆
𝑎,𝑖
(
𝑢𝑛+⋆
𝑖,𝑅−𝑢𝑛+⋆
𝑖,𝐿
)
.
(D.2)
During this prediction step, the determination of the ﬂux is local to cell (𝑖) and there is no Riemann problem involved. From the 
predicted conservative vector U
𝑛+⋆
𝑖
, the predicted primitive vector W
𝑛+⋆
𝑖
(U
𝑛+⋆
𝑖
) is deduced and is again extrapolated at the boundaries 
of cell (𝑖) as W
𝑛+⋆
𝑖,𝐿 and W
𝑛+⋆
𝑖,𝑅, as in Eq. (D.1).
Journal of Computational Physics 547 (2026) 114545 
25 


S. Schropﬀ, F. Petitpas and E. Daniel
Table E.1 
Time steps comparison between the semi-implicit and explicit schemes (NX = 200 and NX 
= 1000).
Δ𝑡200
𝐼𝑀𝐸𝑋 (s)
Δ𝑡200
𝐻𝐿𝐿𝐶 (s)
Δ𝑡1000
𝐼𝑀𝐸𝑋 (s)
Δ𝑡1000
𝐻𝐿𝐿𝐶 (s)
𝑅
 Low-Mach air advection
2.27 × 10−3
1.20 × 10−6
4.55 × 10−4
2.40 × 10−4
 1890
 Low-Mach water shock tube
4.17 × 10−6
2.55 × 10−6
8.33 × 10−7
5.09 × 10−7
 1.64
 Lax air shock tube
1.09 × 10−3
9.56 × 10−4
2.17 × 10−4
1.91 × 10−4
 1.14
 Strong air shock tube
1.75 × 10−8
2.10 × 10−8
3.52 × 10−9
4.21 × 10−9
 0.83
 Two rarefaction fans
2.50 × 10−3
2.57 × 10−3
5.00 × 10−4
5.15 × 10−4
 0.97
 Water-air advection
6.25 × 10−4
1.21 × 10−6
1.25 × 10−4
2.41 × 10−7
 517.7
 Water-air shock tube
2.57 × 10−6
1.70 × 10−6
5.13 × 10−7
3.39 × 10−7
 1.51
 Epoxy-spinel shock tube
7.55 × 10−7
5.75 × 10−7
1.51 × 10−7
3.39 × 10−7
 1.32
Table E.2 
Computational time comparison between the semi-implicit and explicit 
schemes (NX = 200 and NX = 1000).
 RP
𝑡200
𝐼𝑀𝐸𝑋 (s)
𝑡200
𝐻𝐿𝐿𝐶 (s)
𝑡1000
𝐼𝑀𝐸𝑋 (s)
𝑡1000
𝐻𝐿𝐿𝐶 (s)
 Low-Mach air advection
 0.03
 20.22
 0.45
 513.50
 Low-Mach water shock tube
 0.01
 0.01
 0.11
 0.06
 Lax air shock tube
 0.04
 0.02
 0.50
 0.20
 Strong air shock tube
 0.06
 0.01
 0.72
 0.18
 Two rarefaction fans
 0.02
 0.01
 0.20
 0.10
 Water-air advection
 0.04
 10.21
 0.66
 257.71
 Water-air shock tube
 0.06
 0.02
 0.92
 0.20
 Epoxy-spinel shock tube
 0.02
 0.01
 0.22
 0.09
Step 3: Riemann problem
We remind that superscripts (⋅)− and (⋅)+ respectively denote the left and right boundary extrapolated states at the cell interface 
𝑥𝑖+1∕2. The extrapolated conservative vectors are computed and deﬁned as U(W
𝑛+⋆
𝑖,𝐿) ≡U𝑛+⋆,+
𝑖−1∕2  and U(W
𝑛+⋆
𝑖,𝑅) ≡U𝑛+⋆,−
𝑖+1∕2 . We now solve 
the conventional Riemann problem presented in Section 4.2: the intercell ﬂux A∗,𝑛+⋆
𝑖+1∕2 in Eq. (47) is now computed with the newly 
extrapolated conservative vectors.
Appendix E.  Test cases performances comparison
The numerical performances of each test cases are displayed in Tables E.2 and E.1, comparing the semi-implicit scheme and 
the explicit HLLC-based scheme. Generally, the semi-implicit scheme performs slightly better (𝑅≳1), and its performance greatly 
outdoes the explicit scheme when the Mach number is very low (𝑅≫1). For the less improved test cases (generally with a higher 
Mach number), the computational time ratio remains satisfying (𝑅≲1). The computational times are of the same order of magnitude, 
taking generally less than a second for both schemes even with reﬁned meshes; the diﬀerence is mostly noticed, again, for low-Mach 
advection problems where the semi-implicit scheme is much faster.
References
[1] M. Bernard, S. Dellacherie, G. Faccanoni, B. Grec, O. Laﬁtte, T.T. Nguyen, Y. Penel, Study of a low Mach nuclear core model for single-phase ﬂows, in: ESAIM: 
Proceedings,  38, EDP Sciences, 2012, pp. 118–134.
[2] M. Bernard, S. Dellacherie, G. Faccanoni, B. Grec, Y. Penel, Study of a low Mach nuclear core model for two-phase ﬂows with phase transition I: stiﬀened gas 
law, ESAIM 48 (6) (2014) 1639–1679.
[3] S. Dellacherie, On a diphasic low Mach number system, ESAIM 39 (3) (2005) 487–514.
[4] S. Dellacherie, Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number, J. Comput. Phys. 229 (4) (2010) 978–1016.
[5] S. Dellacherie, On a low Mach nuclear core model, in: ESAIM: Proceedings,  35, EDP Sciences, 2012, pp. 79–106.
[6] S. Dellacherie, J. Jung, P. Omnes, P.A. Raviart, Construction of modiﬁed Godunov-type schemes accurate at any Mach number for the compressible Euler system, 
Math. Models Methods Appl. Sci. 26 (13) (2016) 2525–2615.
[7] Y. Penel, S. Dellacherie, B. Després, Coupling strategies for compressible-low Mach number ﬂows, Math. Models Methods Appl. Sci. 25 (06) (2015) 1045–1089.
[8] S. Le Martelot, B. Nkonga, R. Saurel, Liquid and liquid–gas ﬂows at all speeds, J. Comput. Phys. 255 (2013) 53–82.
[9] M. Pelanti, Low Mach number preconditioning techniques for roe-type and HLLC-type methods for a two-phase compressible ﬂow model, Appl. Math. Comput. 
310 (2017) 112–133.
[10] M.J. Del Razo, R.J. LeVeque, Numerical methods for interface coupling of compressible and almost incompressible media, SIAM J. Sci. Comput. 39 (3) (2017) 
B486–B507.
[11] J. Cazé, F. Petitpas, E. Daniel, M. Queguineur, S. Le Martelot, Modeling and simulation of the cavitation phenomenon in turbopumps, J. Comput. Phys. 502 
(2024) 112817.
[12] R. Abgrall, S. Karni, Computations of compressible multiﬂuids, J. Comput. Phys. 169 (2) (2001) 594–623.
[13] R. Saurel, R. Abgrall, A simple method for compressible multiﬂuid ﬂows, SIAM J. Sci. Comput. 21 (3) (1999) 1115–1145.
[14] M.R. Baer, J.W. Nunziato, A two-phase mixture theory for the deﬂagration-to-detonation transition (DDT) in reactive granular materials, Int. J. Multiphase Flow 
12 (6) (1986) 861–889.
[15] A.K. Kapila, R. Menikoﬀ, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deﬂagration-to-detonation transition in granular materials: reduced equations, 
Phys. Fluids 13 (10) (2001) 3002–3024.
[16] A. Murrone, H. Guillard, A ﬁve equation reduced model for compressible two phase ﬂow problems, J. Comput. Phys. 202 (2) (2005) 664–698.
Journal of Computational Physics 547 (2026) 114545 
26 


S. Schropﬀ, F. Petitpas and E. Daniel
[17] G. Allaire, S. Clerc, S. Kokh, A ﬁve-equation model for the simulation of interfaces between compressible ﬂuids, J. Comput. Phys. 181 (2) (2002) 577–616.
[18] J. Massoni, R. Saurel, B. Nkonga, R. Abgrall, Proposition de méthodes et modèles eulériens pour les problèmes à interfaces entre ﬂuides compressibles en présence 
de transfert de chaleur: some models and Eulerian methods for interface problems between compressible ﬂuids with heat transfer, Int. J. Heat Mass Transf. 45 
(6) (2002) 1287–1307.
[19] R. Saurel, F. Petitpas, R.A. Berry, Simple and eﬃcient relaxation methods for interfaces separating compressible ﬂuids, cavitating ﬂows and shocks in multiphase 
mixtures, J. Comput. Phys. 228 (5) (2009) 1678–1712.
[20] K. Schmidmayer, J. Cazé, F. Petitpas, É. Daniel, N. Favrie, Modelling interactions between waves and diﬀused interfaces, Int. J. Numer. Methods Fluids 95 (2) 
(2023) 215–241.
[21] S. Jin, Eﬃcient asymptotic-preserving (AP) schemes for some multiscale kinetic equations, SIAM J. Sci. Comput. 21 (2) (1999) 441–454.
[22] H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, Comput. Fluids 28 (1) (1999) 63–86.
[23] E. Turkel, R. Radespiel, N. Kroll, Assessment of preconditioning methods for multidimensional aerodynamics, Comput. Fluids 26 (6) (1997) 613–634.
[24] S. Roller, C.D. Munz, A low Mach number scheme based on multi-scale asymptotics, Comput. Vis. Sci. 3 (2000) 85–91.
[25] C. Viozat, Implicit Upwind Schemes for Low Mach Number Compressible Flows, Ph.D. thesis, Inria, 1997.
[26] F. Miczek, F.K. Röpke, P.V.F. Edelmann, New numerical solver for ﬂows at various Mach numbers, Astron. Astrophys. 576 (2015) A50.
[27] F.H. Harlow, A.A. Amsden, Numerical calculation of almost incompressible ﬂow, J. Comput. Phys. 3 (1) (1968) 80–93.
[28] F.H. Harlow, A.A. Amsden, A numerical ﬂuid dynamics calculation method for all ﬂow speeds, J. Comput. Phys. 8 (2) (1971) 197–213.
[29] V. Casulli, D. Greenspan, Pressure method for the numerical solution of transient, compressible ﬂuid ﬂows, Int. J. Numer. Methods Fluids 4 (11) (1984) 
1001–1012.
[30] R. Klein, Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics i: one-dimensional ﬂow, J. Comput. Phys. 121 (2) (1995) 
213–237.
[31] F. Coquel, M. Postel, N. Poussineau, Q.H. Tran, Multiresolution technique and explicit-implicit scheme for multicomponent ﬂows, J. Numer. Math. 14 (3) (2006) 
187.
[32] F. Coquel, Q.L. Nguyen, M. Postel, Q.H. Tran, Large time step positivity-preserving method for multiphase ﬂows, in: Hyperbolic Problems: Theory, Numerics, 
Applications, Springer, 2008, pp. 849–856.
[33] F. Coquel, Q.L. Nguyen, M. Postel, Q.H. Tran, Local time stepping with adaptive time step control for a two-phase ﬂuid system, in: ESAIM: Proceedings,  29, 
EDP Sciences, 2009, pp. 73–88.
[34] F. Coquel, Q.L. Nguyen, M. Postel, Q.H. Tran, Local time stepping applied to implicit-explicit methods for hyperbolic systems, Multiscale Model. Simul. 8 (2) 
(2010) 540–570.
[35] F. Coquel, Q. Nguyen, M. Postel, Q. Tran, Entropy-satisfying relaxation method with large time-steps for Euler IBVPs, Math. Comput. 79 (271) (2010) 1493–1533.
[36] C.D. Munz, S. Roller, R. Klein, K.J. Geratz, The extension of incompressible ﬂow solvers to the weakly compressible regime, Comput. Fluids 32 (2) (2003) 
173–196.
[37] M. Jemison, M. Sussman, M. Arienti, Compressible, multiphase semi-implicit method with moment of ﬂuid interface representation, J. Comput. Phys. 279 (2014) 
182–217.
[38] A. Bermúdez, S. Busto, M. Dumbser, J.L. Ferrín, L. Saavedra, M.E. Vázquez-Cendón, A staggered semi-implicit hybrid FV/FE projection method for weakly 
compressible ﬂows, J. Comput. Phys. 421 (2020) 109743.
[39] M. Dumbser, D.S. Balsara, M. Tavelli, F. Fambri, A divergence-free semi-implicit ﬁnite volume scheme for ideal, viscous, and resistive magnetohydrodynamics, 
Int. J. Numer. Methods Fluids 89 (1–2) (2019) 16–42.
[40] S. Patankar, Numerical Heat Transfer and Fluid Flow (1st ed.), CRC press, 1980.
[41] S.V. Patankar, D.B. Spalding, A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic ﬂows, in: Numerical Prediction of 
Flow, Heat Transfer, Turbulence and Combustion, Elsevier, 1983, pp. 54–73.
[42] V. Dolejší, Semi-implicit interior penalty discontinuous galerkin methods for viscous compressible ﬂows, Commun. Comput. Phys 4 (2) (2008) 231–274.
[43] R. Knikker, A comparative study of high-order variable-property segregated algorithms for unsteady low Mach number ﬂows, Int. J. Numer. Methods Fluids 66 
(4) (2011) 403–427.
[44] F. Cordier, P. Degond, A. Kumbaro, An asymptotic-preserving all-speed scheme for the euler and Navier–Stokes equations, J. Comput. Phys. 231 (17) (2012) 
5685–5704.
[45] E. Motheau, J. Abraham, A high-order numerical algorithm for DNS of low-Mach-number reactive ﬂows with detailed chemistry and quasi-spectral accuracy, J. 
Comput. Phys. 313 (2016) 430–454.
[46] J. Ventosa-Molina, J. Chiva, O. Lehmkuhl, J. Muela, C.D. Pérez-Segarra, A. Oliva, Numerical analysis of conservative unstructured discretisations for low Mach 
ﬂows, Int. J. Numer. Methods Fluids 84 (6) (2017) 309–334.
[47] S. Klainerman, A. Majda, Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible ﬂuids, Commun. 
Pure Appl. Math. 34 (4) (1981) 481–524.
[48] S. Klainerman, A. Majda, Compressible and incompressible ﬂuids, Commun. Pure Appl. Math. 35 (1982) 629–651.
[49] A. Nonaka, A.S. Almgren, J.B. Bell, M.J. Lijewski, C.M. Malone, M. Zingale, MAESTRO: an adaptive low Mach number hydrodynamics algorithm for stellar ﬂows, 
Astrophys. J. Suppl. Ser. 188 (2) (2010) 358.
[50] P. Degond, M. Tang, All speed scheme for the low Mach number limit of the isentropic Euler equations, Commun. Comput. Phys. 10 (1) (2011) 1–31.
[51] E. Abbate, A. Iollo, G. Puppo, An asymptotic-preserving all-speed scheme for ﬂuid dynamics and nonlinear elasticity, SIAM J. Sci. Comput. 41 (5) (2019) 
A2850–A2879.
[52] J.H. Park, C.D. Munz, Multiple pressure variables methods for ﬂuid ﬂow at all mach numbers, Int. J. Numer. Methods Fluids 49 (8) (2005) 905–931.
[53] M.S. Liou, C.J. Steﬀen, Jr, A new ﬂux splitting scheme, J. Comput. Phys. 107 (1) (1993) 23–39.
[54] G.C. Zha, E. Bilgen, Numerical solutions of Euler equations by using a new ﬂux vector splitting scheme, Int. J. Numer. Methods Fluids 17 (2) (1993) 115–144.
[55] E.F. Toro, M.E. Vázquez-Cendón, Flux splitting schemes for the Euler equations, Comput. Fluids 70 (2012) 1–12.
[56] E.F. Toro, C.E. Castro, B.J. Lee, A novel numerical ﬂux for the 3D euler equations with general equation of state, J. Comput. Phys. 303 (2015) 80–94.
[57] S.A. Tokareva, E.F. Toro, A ﬂux splitting method for the Baer–Nunziato equations of compressible two-phase ﬂow, J. Comput. Phys. 323 (2016) 45–74.
[58] M.F.P. ten Eikelder, F. Daude, B. Koren, A.S. Tijsseling, An acoustic-convective splitting-based approach for the Kapila two-phase ﬂow model, J. Comput. Phys. 
331 (2017) 188–208.
[59] M. Dumbser, V. Casulli, A staggered semi-implicit spectral discontinuous Galerkin scheme for the shallow water equations, Appl. Math. Comput. 219 (15) (2013) 
8057–8077.
[60] S. Noelle, G. Bispen, K. Arun, M. Lukacova-Medvidova, C. Munz, A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas 
dynamics, SIAM J. Sci. Comput. 36 (6) (2014) B989–B1024.
[61] M. Dumbser, V. Casulli, A conservative, weakly nonlinear semi-implicit ﬁnite volume scheme for the compressible Navier- Stokes equations with general equation 
of state, Appl. Math. Comput. 272 (2016) 479–497.
[62] M. Tavelli, M. Dumbser, A pressure-based semi-implicit space-time discontinuous Galerkin method on staggered unstructured meshes for the solution of the 
compressible Navier–Stokes equations at all Mach numbers, J. Comput. Phys. 341 (2017) 341–376.
[63] W. Boscheri, G. Dimarco, R. Loubère, M. Tavelli, M.H. Vignal, A second order all Mach number IMEX ﬁnite volume solver for the three dimensional Euler 
equations, J. Comput. Phys. 415 (2020) 109486.
[64] W. Boscheri, L. Pareschi, High order pressure-based semi-implicit IMEX schemes for the 3D Navier-Stokes equations at all Mach numbers, J. Comput. Phys. 434 
(2021) 110206.
[65] W. Boscheri, M. Tavelli, High order semi-implicit schemes for viscous compressible ﬂows in 3D, Appl. Math. Comput. 434 (2022) 127457.
Journal of Computational Physics 547 (2026) 114545 
27 


S. Schropﬀ, F. Petitpas and E. Daniel
[66] B. Re, R. Abgrall, A pressure-based method for weakly compressible two-phase ﬂows under a Baer–Nunziato type model with generic equations of state and 
pressure and velocity disequilibrium, Int. J. Numer. Methods Fluids 94 (8) (2022) 1183–1232.
[67] B. Battisti, W. Boscheri, A linearly implicit shock capturing scheme for compressible two-phase ﬂows at all Mach numbers, 2025. Working paper or preprint, 
https://hal.science/hal-04963046.
[68] O. Le Métayer, R. Saurel, The Noble-Abel stiﬀened-gas equation of state, Phys. Fluids 28 (4) (2016).
[69] O. Heuzé, General form of the Mie–Grüneisen equation of state, Comptes Rendus. Mécanique 340 (10) (2012) 679–687.
[70] F. Petitpas, R. Saurel, E. Franquet, A. Chinnayya, Modelling detonation waves in condensed energetic materials: multiphase CJ conditions and multidimensional 
computations, Shock Waves 19 (2009) 377–401.
[71] O. Le Métayer, J. Massoni, R. Saurel, Élaboration des lois d’état d’un liquide et de sa vapeur pour les modèles d’écoulements diphasiques, Int. J. Therm. Sci. 43 
(3) (2004) 265–276.
[72] R. Saurel, S. Gavrilyuk, F. Renaud, A multiphase model with internal degrees of freedom: application to shock–bubble interaction, J. Fluid Mech. 495 (2003) 
283–321.
[73] T. Gallouët, J.M. Hérard, N. Seguin, Numerical modeling of two-phase ﬂows using the two-ﬂuid two-pressure approach, Math. Models Methods Appl. Sci. 14 
(05) (2004) 663–700.
[74] F. Coquel, T. Gallouët, J.M. Hérard, N. Seguin, Closure laws for a two-ﬂuid two-pressure model, C.R. Math. 334 (10) (2002) 927–932.
[75] A. Zein, M. Hantke, G. Warnecke, Modeling phase transition for compressible two-phase ﬂows applied to metastable liquids, J. Comput. Phys. 229 (8) (2010) 
2964–2998.
[76] F.D. Stacey, J.H. Hodgkinson, Thermodynamics with the Grüneisen parameter: fundamentals and applications to high pressure physics and geophysics, Phys. 
Earth Planet. Inter. 286 (2019) 42–68.
[77] E.F. Toro, The HLL and HLLC Riemann solvers, in: Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer, 2009, pp. 
315–344.
[78] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer Science & Business Media, 2013.
[79] V. Casulli, P. Zanolli, A nested Newton-type algorithm for ﬁnite volume methods solving Richards’ equation in mixed form, SIAM J. Sci. Comput. 32 (4) (2010) 
2255–2273.
[80] O. Sigmund, J. Petersson, Numerical instabilities in topology optimization: a survey on procedures dealing with checkerboards, mesh-dependencies and local 
minima, Struct. Optim. 16 (1998) 68–75.
[81] S. Dellacherie, Checkerboard modes and wave equation, in: Proceedings of ALGORITMY,  2009, 2009, pp. 71–80.
[82] A. Shukla, A. Misra, S. Kumar, Checkerboard problem in ﬁnite element based topology optimization, Int. J. Adv. Eng. Technol. 6 (4) (2013) 1769.
[83] C.M. Rhie, W.L. Chow, Numerical study of the turbulent ﬂow past an airfoil with trailing edge separation, AIAA J. 21 (11) (1983) 1525–1532.
[84] R. Saurel, R. Abgrall, A multiphase Godunov method for compressible multiﬂuid and multiphase ﬂows, J. Comput. Phys. 150 (2) (1999) 425–467.
[85] E. Abbate, A. Iollo, G. Puppo, An all-speed relaxation scheme for gases and compressible materials, J. Comput. Phys. 351 (2017) 1–24.
[86] N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw, A method for avoiding the acoustic time step restriction in compressible ﬂow, J. Comput. Phys. 228 (11) (2009) 
4146–4161.
[87] A.B. Wood, R.B. Lindsay, A textbook of sound, 1956.
[88] P. Bruel, S. Delmas, J. Jung, V. Perrier, A low Mach correction able to deal with low mach acoustics, J. Comput. Phys. 378 (2019) 723–759.
[89] S.F. Davis, Simpliﬁed second-order Godunov-type methods, SIAM J. Sci. Stat. Comput. 9 (3) (1988) 445–473.
[90] P.K. Sweby, High resolution schemes using ﬂux limiters for hyperbolic conservation laws, SIAM J. Numer. Anal. 21 (5) (1984) 995–1011.
Journal of Computational Physics 547 (2026) 114545 
28 
