Contents lists available at ScienceDirect
Journal of Computational Physics
journal homepage: www.elsevier.com/locate/jcp
A linearly implicit shock capturing scheme for compressible 
two-phase ﬂows at all Mach numbers
Beatrice Battisti a, Walter Boscheri
a,b,∗
a Laboratoire de Mathématiques UMR 5127 CNRS, Université Savoie Mont Blanc, 73376, Le Bourget du Lac, France
b Department of Mathematics and Computer Science, University of Ferrara, 44121, Ferrara, Italy
a r t i c l e  i n f o
Keywords:
All Mach solver
Compressible two-phase ﬂows
Semi-implicit IMEX scheme
Asymptotic preserving
Well-balanced
Baer–Nunziato model
 
a b s t r a c t
We present a semi-implicit solver for the solution of compressible two-phase ﬂows governed by 
the Baer–Nunziato model. A novel linearly implicit discretization is proposed for the pressure 
ﬂuxes as well as for the relaxation source terms, whereas an explicit scheme is retained for the 
nonlinear convective contributions. Consequently, the CFL-type stability condition on the maxi-
mum admissible time step is based only on the mean ﬂow velocity and not on the sound speed 
of each phase, so that the novel scheme works uniformly for all Mach numbers. Central ﬁnite 
diﬀerence operators on Cartesian grids are adopted for the implicit terms, thus avoiding any need 
of numerical diﬀusion that might destroy accuracy in the low Mach number regime. To comply 
with high Mach number ﬂows, shock capturing ﬁnite volume schemes are employed for the ap-
proximation of the convective ﬂuxes. The discretization of the non-conservative terms ensures 
the preservation of moving equilibrium solutions, making the new method well-balanced. The 
new scheme is also proven to be asymptotic preserving in the low Mach limit of the mixture 
model. Second order of accuracy is achieved by means of an implicit-explicit (IMEX) time step-
ping algorithm combined with a total variation diminishing (TVD) reconstruction technique. The 
novel method is benchmarked against a set of test cases involving diﬀerent Mach number regimes, 
permitting to validate both accuracy and robustness.
1.  Introduction
Several natural processes are concerned with multi-phase ﬂows, which are mathematically described by nonlinear systems of 
time-dependent hyperbolic partial diﬀerential equations. Relevant examples are avalanches, meteorological ﬂows with cloud forma-
tion, volcanic eruptions, sediment transport in rivers, and debris ﬂows in mountain regions. On the other hand, also many industrial 
applications involve multi-phase ﬂows, mainly focused on combustion processes encountered in aerospace engineering, automotive 
industry, nuclear reactor safety, paper and food manufacturing, as well as turbo-machinery for renewable energy production. De-
pending on the Mach number, which measures the ratio between the ﬂuid velocity and the sound speed, the behavior of the ﬂow 
may be quite disparate, ranging from the compressible regime for high Mach numbers to the incompressible regime in the low Mach 
number limit. Indeed, for industrial processes, compressible phases can be typically considered as they describe the combustion of 
liquid and solid fuels or solid and bio-mass. However, for natural phenomena, one phase may exhibit a compressible behavior, e.g. 
air, while the other phase lies in the incompressible regime, e.g. water.
∗Corresponding author.
 
E-mail addresses: beatrice.battisti@univ-smb.fr (B. Battisti), walter.boscheri@univ-smb.fr (W. Boscheri).
https://doi.org/10.1016/j.jcp.2025.114227
Received 23 February 2025; Received in revised form 23 June 2025; Accepted 6 July 2025
Journal of Computational Physics 539 (2025) 114227 
Available online 8 July 2025 
0021-9991/© 2025 The Author(s). 
Published by Elsevier Inc. 
This is an open access article under the CC BY license 
( http://creativecommons.org/licenses/by/4.0/ ). 


B. Battisti and W. Boscheri
Due to the complexity of such ﬂows, no universally agreed model exists in the literature. Nevertheless, most of the models are 
based on the Baer–Nunziato (BN) equations for compressible two-phase ﬂows, originally introduced for detonation waves in solid–
gas combustion processes [1]. In this case, the exchange terms of mass transfer and convective heat are discarded, and each phase is 
governed by its own equation of state. The resulting hyperbolic system is composed by balance laws for the mass, momentum, and 
energy of each phase, and the evolution of the volume fraction. In addition, some source terms account for the interaction between 
the phases, due to friction forces and pressure. These equations allow each phase to have its own velocity and pressure. Simpliﬁed 
versions may be obtained making hypotheses on the problem at hand. For example, an asymptotic reduction of the model in the 
limit of the velocity, the pressure, or the temperature equilibrium yields a six equation model [2–6]. If both velocity and pressure 
are common to the two phases, the system further simpliﬁes to ﬁve equations [7–11], and if velocity, pressure, and also temperature 
undergo instantaneous equilibrium, then the model becomes a four-equations one [12]. However, these models represent a subset of 
the complete system, and their simpliﬁed nature often comes with additional modeling diﬃculties [13,14]. We then concentrate on 
the full seven-equation Baer–Nunziato model.
The numerical solution of multi-phase ﬂows is very challenging due to the need of treating diﬀerent Mach number regimes 
that may coexist in the same computation. Simulating ﬂows with diﬀerent orders of magnitude of the Mach number is not trivial. 
Furthermore, the BN model involves non-conservative terms accounting for the interaction between the inter-phase pressure and the 
gradient of the volume fraction, whose discretization must ensure physical consistency. For compressible multi-phase ﬂows, shock 
capturing ﬁnite volume schemes are often employed [4,15–20], that resolves complex structures developing when the Mach number 
is of order one [18,20,21]. Such methods are normally explicit. However, for very low Mach numbers, the eﬀect of numerical viscosity 
on the slow convective waves introduced by upwind-type schemes is proven to degrade the accuracy [22,23]. Moreover, the CFL-type 
stability condition dictates a severe restriction on the maximum admissible time step while approaching the low Mach limit, hence 
making an explicit scheme no longer useful in practice. Indeed, in the incompressible or weakly compressible limit, the acoustic waves 
are negligible, so there is no need to supply the scheme with a large amount of numerical viscosity, that must be counterbalanced 
by a smaller time step and a higher mesh resolution. In this case, implicit methods are instead preferred, although they imply the 
solution of a globally nonlinear system deﬁned on the entire computational domain. A successful idea consists in treating implicitly 
only one part of the system to be solved, while keeping the remaining explicit, thus both incompressible and compressible regimes 
can be handled [24–29]. This approach permits to design space and time discretizations in which the implicit part of the system 
is relatively easy to be inverted, typically avoiding nonlinear systems, while keeping robustness and shock-capturing properties in 
the explicit part. This family of scheme is typically referred to as IMplicit-EXplicit (IMEX) methods [30–37], which are proven to be 
asymptotic preserving, meaning that they recover a consistent discretization of the incompressible model in the low Mach number 
limit. A slightly diﬀerent approach to design all Mach solvers, i.e. numerical schemes that can uniformly deal with a large spectrum 
of Mach number regimes, is given by semi-implicit methods [38–45] that aim at obtaining a linearly implicit scheme for the stiﬀ 
terms in the governing equations, thus avoiding any need of iterative methods.
Although developing all Mach solvers is an active research topic, most of the numerical methods are addressed to single-phase 
ﬂows, as the aforementioned works. Only few contributions may be found, in the literature, for two-phase ﬂows. In [46], a pressure-
based formulation of the Baer–Nunziato model is proposed, to simulate one-dimensional weakly-compressible ﬂows with generic 
equation of state. Instead of solving for the total energy, a non-conservative evolution equation for the pressure is formulated and 
discretized, so that a more physically relevant set of variables is adopted. The method is thus adapted for low Mach regimes, but 
it lacks accuracy in the high Mach regime, especially for the resolution of shock waves, which are no longer captured [47]. The 
asymptotic behavior of the scheme is demonstrated by considering a uniﬁed reference Mach number for both phases, and the method 
is ﬁrst order accurate in space and time. In [48], a two-dimensional pressure-based solver is developed for low Mach ﬂows. In this 
approach, a binary factor, which might be set either to unity or to zero, switches the ﬂow behavior from compressible to incom-
pressible regime. Other two-phase solvers have been proposed in Varsakelis and Papalexandris[49,50], where the incompressible or 
compressible modeling is set a priori for each phase. These latter can be regarded as numerical methods for mixed models rather 
than as asymptotic preserving schemes. Very recently, an asymptotic preserving IMEX method for the full Baer–Nunziato model in 
multiple space dimensions has been presented in Malusà and Alaia[51]. The Mach number is deﬁned separately for the two phases, 
to account for potentially very diﬀerent regimes.
In this work, we aim at designing a semi-implicit all Mach solver for the BN model on three-dimensional Cartesian grids. Central 
ﬁnite diﬀerence operators are used for treating the implicit terms, while we resort to an explicit ﬁnite volume method to deal with 
shock waves induced by the nonlinear convective terms. Particular attention is devoted to ensure the asymptotic preserving property 
of the scheme in the low Mach number limit for the mixture. The resulting scheme is second order accurate in space and time for 
a wide range of Mach numbers. Furthermore, the scheme is required to be well-balanced, thus moving stationary equilibria of the 
ﬂow are exactly maintained. As pointed out in Re and Abgrall[46], this is the physical guidance to construct physically consistent 
discretizations of the non-conservative terms related to the gradient of the volume fraction. Diﬀerently from Malusà and Alaia[51], the 
volume fraction equation will be treated implicitly, and a linearly implicit solver is devised by suitable time linearization techniques 
borrowed from previous existing works for single-phase ﬂows [43]. The relaxation source terms are solved implicitly while being 
explicitly computable via a source splitting technique, which can achieve second order of accuracy by means of semi-implicit time 
integrators [41]. Consequently, the relaxation limits of the model induced by the stiﬀ source terms does not pose any problem from the 
numerical viewpoint. We do not resort to any predictor-corrector strategy as in Malusà and Alaia[51], and the novel scheme exactly 
recovers the asymptotic preserving method forwarded in Boscheri and Pareschi[43] in the case of single-phase ﬂow. The proposed 
methodology is intended as a simple, but not trivial, discretization for multi-phase ﬂows, which can become highly complex, as in the 
case of volcanic applications [52]. Although such ﬂows may involve challenging relaxation terms, strongly interacting phases, and 
Journal of Computational Physics 539 (2025) 114227 
2 


B. Battisti and W. Boscheri
very diﬀerent regimes, they share the same homogeneous and hyperbolic structure of the Baer–Nunziato model. Therefore, this work 
stands as the groundwork for potentially highly complex real-world multi-phase applications tailored to the simulation of magma 
ﬂows.
The rest of the paper is organized as follows. Section 2 introduces the governing equations and their non-dimensional formulation, 
for a clear understanding of the potential sources of stiﬀness due to diﬀerent Mach number regimes. This analysis deﬁnes the basis for 
the development of the numerical scheme, which is described in Section 3. Section 4 provides a variety of test cases that demonstrate 
the accuracy and the robustness of the new method, while facing diﬀerent ﬂow regimes. Finally, in Section 5, some conclusions are 
drawn, and an outline to future developments is forwarded.
2.  Mathematical model
Let Ω ∈ℝ𝑑 be a bounded domain of dimension 𝑑∈{1, 2, 3}, deﬁned by a spatial position vector 𝐱= (𝑥, 𝑦, 𝑧) ∈ℝ𝑑, and let 𝑡∈ℝ+
denote the temporal variable. The system of governing equations is given by the compressible two-phase ﬂow Baer–Nunziato model 
written in the form derived by Saurel and Abgrall[15]:
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓) + ∇⋅(𝜌𝑓𝜙𝑓𝐮𝑓) = 0,
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓𝐮𝑓) + ∇⋅(𝜌𝑓𝜙𝑓𝐮𝑓𝐮𝑓) + ∇(𝑝𝑓𝜙𝑓) + 𝑝𝐼∇𝜙𝑓= 𝛿(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓𝐸𝑓) + ∇⋅(𝜙𝑓𝐮𝑓(𝜌𝑓𝐸𝑓+ 𝑝𝑓)) + 𝑝𝐼𝐮𝐼∇𝜙𝑓= 𝜇𝑝𝐼(𝑝𝑠−𝑝𝑓) + 𝛿𝐮𝐼(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠) + ∇⋅(𝜌𝑠𝜙𝑠𝐮𝑠) = 0,
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠𝐮𝑠) + ∇⋅(𝜌𝑠𝜙𝑠𝐮𝑠𝐮𝑠) + ∇(𝑝𝑠𝜙𝑠) −𝑝𝐼∇𝜙𝑠= −𝛿(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠𝐸𝑠) + ∇⋅(𝜙𝑠𝐮𝑠(𝜌𝑠𝐸𝑠+ 𝑝𝑠)) −𝑝𝐼𝐮𝐼∇𝜙𝑠= −𝜇𝑝𝐼(𝑝𝑠−𝑝𝑓) −𝛿𝐮𝐼(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡𝜙𝑠+ 𝐮𝐼⋅∇𝜙𝑠= 𝜇(𝑝𝑠−𝑝𝑓).
(1)
System (1) represents a non-equilibrium model, where each phase 𝛼= {𝑓, 𝑠}, with 𝑓 the ﬂuid phase and 𝑠 the solid phase, has its 
own density 𝜌𝛼, pressure 𝑝𝛼, velocity 𝐮𝛼, volume fraction 𝜙𝛼, and speciﬁc total energy 𝐸𝛼. The volume fraction is a non-dimensional 
variable, bounded in the interval [0, 1], and the sum of the volume fractions must be equal to unity for consistency, here 𝜙𝑠+ 𝜙𝑓= 1. 
The speciﬁc total energy is deﬁned as the sum of the speciﬁc internal energy 𝑒𝛼, and the speciﬁc kinetic energy 𝑘𝛼: 𝐸𝛼= 𝑒𝛼+ 𝑘𝛼, with 
𝑘𝛼= 1
2 |𝐮𝛼|2. In this paper, we consider the stiﬀened gas equation of state (EOS) for each phase:
𝑒𝛼= 𝑝𝛼+ 𝛾𝛼𝜋𝛼
𝜌𝛼(𝛾𝛼−1) ,
(2)
with 𝛾𝛼 and 𝜋𝛼 being, respectively, the polytropic index, and the pressure constant of each phase. The ideal gas EOS can be retrieved 
from (2) by setting 𝜋𝛼= 0. The source terms in (1) account for inter-phase drag and pressure relaxation through the parameters 𝛿
and 𝜇, respectively, which might become stiﬀ. Indeed, the dynamics of wave propagation is typically signiﬁcantly slower than the 
relaxation of pressure and velocity, to the point that they are frequently treated as instantaneous processes, with inﬁnite 𝜇 and 𝛿. The 
interface pressure and velocity, respectively 𝑝𝐼 and 𝐮𝐼, may be deﬁned in diﬀerent ways [53], and here we consider two deﬁnitions: 
𝑝𝐼= 𝑝𝑓,
𝐮𝐼= 𝐮𝑠,
(3a)
𝑝𝐼=
∑
𝛼
𝜙𝛼𝑝𝛼,
𝐮𝐼=
∑
𝛼𝜌𝛼𝜙𝛼𝐮𝛼
∑
𝛼𝜌𝛼𝜙𝛼
.
(3b)
The ﬁrst deﬁnition (3a), based on a physical viewpoint, is adopted in the original Baer–Nunziato model [1], as well as in other works 
[54,55]. However, many authors [13,15,46,51] prefer the weighted formulation (3b). Here, we allow for both deﬁnitions, and the 
selection is problem-dependent. Such a choice does not aﬀect the numerical method, but it may inﬂuence the results. The system of 
equations tends to the single-phase Euler equations far from the interfaces, and when considering either 𝜙𝑠= 1 or 𝜙𝑓= 1. As analyzed 
in Andrianov and Warnecke[56], the eigenvalues of the Baer–Nunziato model, projected in the normal direction, write
𝝀= [ 𝐮𝑓⋅𝐧, 𝐮𝑓⋅𝐧± 𝑐𝑓, 𝐮𝑠⋅𝐧, 𝐮𝑠⋅𝐧± 𝑐𝑠, 𝐮𝐼⋅𝐧],
(4)
with 𝐧 denoting a unit vector pointing normal to the ﬂow direction, and 𝑐𝛼=
√
𝛾𝛼(𝑝𝛼+ 𝜋𝛼)∕𝜌𝛼 representing the speed of sound of the 
diﬀerent phases.
The Baer–Nunziato system (1) may be cast in the following general non-conservative form:
𝜕𝐐
𝜕𝑡+ ∇⋅𝐅(𝐐) + 𝐁⋅∇𝐐= 𝐒(𝐐),
𝐱∈Ω ⊂ℝ𝑑,
𝑡∈ℝ+,
(5)
where 𝐐= (𝜌𝑓𝜙𝑓, 𝜌𝑓𝜙𝑓𝐮𝑓, 𝜌𝑓𝜙𝑓𝐸𝑓, 𝜌𝑠𝜙𝑠, 𝜌𝑠𝜙𝑠𝐮𝑠, 𝜌𝑠𝜙𝑠𝐸𝑠, 𝜙𝑠)⊤ is the state vector, 𝐅(𝐐) is the conservative nonlinear ﬂux tensor, the 
matrix 𝐁(𝐐) accounts for the non-conservative contributions, and the vector 𝐒(𝐐) contains the source terms.
Journal of Computational Physics 539 (2025) 114227 
3 


B. Battisti and W. Boscheri
If the equations are discretized by means of a fully explicit scheme, the time step Δ𝑡 is constrained by a CFL-type stability condition, 
that is
Δ𝑡≤CFL min
Ω
(
𝜚
max(|𝐮𝛼⋅𝐧| ± 𝑐𝛼)
)
,
(6)
where 𝜚 is the characteristic mesh size of the control volumes used to pave the computational domain. For very slow dynamics, i.e. in 
the low Mach regime, the sound speed increases to inﬁnity, and Δ𝑡 becomes prohibitively small. Furthermore, an excessive numerical 
dissipation in explicit schemes causes a loss of accuracy, for low Mach number ﬂows. To overcome these issues simultaneously, the 
terms that are aﬀected by the Mach number have to be discretized implicitly.
2.1.  Non-dimensional form of the Baer–Nunziato model
To identify those terms which are responsible for the stiﬀness of the problem in the low Mach regime, the non-dimensional form of 
the equations is needed. To this purpose, we deﬁne non-dimensional quantities, characterized by a tilde symbol, scaled by reference 
values, with the subscript “0”. Assuming that the convective scales of the two phases are comparable, we can use reference values 
common to both phases, hence:
̃𝐱𝛼= 𝐱𝛼
𝐿0
,
̃𝐮𝛼= 𝐮𝛼
𝐮0
,
̃𝑡= 𝑡
𝜏0
= |𝐮0|
𝐿0
𝑡,
̃𝜌𝛼= 𝜌𝛼
𝜌0
,
̃𝑝𝛼= 𝑝𝛼
𝑝0
,
̃𝐸𝛼= 𝜌0
𝑝0
𝐸𝛼,
̃𝛿= 𝛿
𝛿0
= 𝜏0
𝜌0
𝛿,
̃𝜇= 𝜇
𝜇0
= 𝜌0𝐮2
0𝜏0𝜇.
The volume fraction 𝜙𝛼 is inherently dimensionless. The scaling of the time variable holds when considering the Strouhal number, 
𝑆𝑡=
𝐿0
𝜏0|𝐮0| , assumed to be 𝑆𝑡≈1, as usually done in the literature [57], meaning that the characteristic time scale of the ﬂow ﬁeld 
evolution corresponds to that one related to the convection.
The non-dimensional form of the Baer–Nunziato system is then given by
𝜌0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑓̃𝜙𝑓) + 𝜌0𝐮0
𝐿0
∇⋅( ̃𝜌𝑓̃𝜙𝑓̃𝐮𝑓) = 0,
𝜌0𝐮0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑓̃𝜙𝑓̃𝐮𝑓) +
𝜌0𝐮2
0
𝐿0
∇⋅( ̃𝜌𝑓̃𝜙𝑓̃𝐮𝑓̃𝐮𝑓) + 𝑝0
𝐿0
∇( ̃𝑝𝑓̃𝜙𝑓) + 𝑝0
𝐿0
̃𝑝𝐼∇̃𝜙𝑓= 𝜌0𝐮0
𝜏0
̃𝛿(̃𝐮𝑠−̃𝐮𝑓),
𝑝0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑓̃𝜙𝑓̃𝐸𝑓) + 𝑝0𝐮0
𝐿0
∇⋅( ̃𝜙𝑓̃𝐮𝑓( ̃𝜌𝑓̃𝐸𝑓+ ̃𝑝𝑓)) + 𝑝0𝐮0
𝐿0
̃𝑝𝐼̃𝐮𝐼∇̃𝜙𝑓=
𝑝2
0
𝜌0𝐮2
0𝜏0
̃𝜇̃𝑝𝐼( ̃𝑝𝑠−̃𝑝𝑓) +
𝜌0𝐮2
0
𝜏0
̃𝛿̃𝐮𝐼(̃𝐮𝑠−̃𝐮𝑓),
𝜌0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑠̃𝜙𝑠) + 𝜌0𝐮0
𝐿0
∇⋅( ̃𝜌𝑠̃𝜙𝑠̃𝐮𝑠) = 0,
𝜌0𝐮0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑠̃𝜙𝑠̃𝐮𝑠) +
𝜌0𝐮2
0
𝐿0
∇⋅( ̃𝜌𝑠̃𝜙𝑠̃𝐮𝑠̃𝐮𝑠) + 𝑝0
𝐿0
∇( ̃𝑝𝑠̃𝜙𝑠) −𝑝0
𝐿0
̃𝑝𝐼∇̃𝜙𝑠= −𝜌0𝐮0
𝐿0
̃𝛿(̃𝐮𝑠−̃𝐮𝑓),
𝑝0
𝜏0
𝜕
𝜕̃𝑡( ̃𝜌𝑠̃𝜙𝑠̃𝐸𝑠) + 𝑝0𝐮0
𝐿0
∇⋅( ̃𝜙𝑠̃𝐮𝑠( ̃𝜌𝑠̃𝐸𝑠+ ̃𝑝𝑠)) −𝑝0𝐮0
𝐿0
̃𝑝𝐼̃𝐮𝐼∇̃𝜙𝑠= −
𝑝2
0
𝜌0𝐮2
0𝜏0
̃𝜇̃𝑝𝐼( ̃𝑝𝑠−̃𝑝𝑓) −
𝜌0𝐮2
0
𝜏0
̃𝛿̃𝐮𝐼(̃𝐮𝑠−̃𝐮𝑓),
1
𝜏0
𝜕
𝜕̃𝑡
̃𝜙𝑠+ 𝐮0
𝐿0
̃𝐮𝐼⋅∇̃𝜙𝑠=
𝑝0
𝜌0𝐮2
0𝜏0
̃𝜇( ̃𝑝𝑠−̃𝑝𝑓).
(7)
Introducing the stiﬀness parameter 𝜖= 𝛾0𝑀2, related to a global Mach number 𝑀= |𝐮0|∕𝑐0, with the speed of sound 𝑐0 =
√
𝛾0𝑝0∕𝜌0, 
and dropping the tildes for readability, the non-dimensional system ﬁnally reads
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓) + ∇⋅(𝜌𝑓𝜙𝑓𝐮𝑓) = 0,
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓𝐮𝑓) + ∇⋅(𝜌𝑓𝜙𝑓𝐮𝑓𝐮𝑓) + 1
𝜖∇(𝑝𝑓𝜙𝑓) + 1
𝜖𝑝𝐼∇𝜙𝑓= 𝛿(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑓𝜙𝑓𝐸𝑓) + ∇⋅(𝜙𝑓𝐮𝑓(𝜌𝑓𝐸𝑓+ 𝑝𝑓)) + 𝑝𝐼𝐮𝐼∇𝜙𝑓= 1
𝜖𝜇𝑝𝐼(𝑝𝑠−𝑝𝑓) + 𝜖𝛿𝐮𝐼(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠) + ∇⋅(𝜌𝑠𝜙𝑠𝐮𝑠) = 0,
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠𝐮𝑠) + ∇⋅(𝜌𝑠𝜙𝑠𝐮𝑠𝐮𝑠) + 1
𝜖∇(𝑝𝑠𝜙𝑠) −1
𝜖𝑝𝐼∇𝜙𝑠= −𝛿(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡(𝜌𝑠𝜙𝑠𝐸𝑠) + ∇⋅(𝜙𝑠𝐮𝑠(𝜌𝑠𝐸𝑠+ 𝑝𝑠)) −𝑝𝐼𝐮𝐼∇𝜙𝑠= −1
𝜖𝜇𝑝𝐼(𝑝𝑠−𝑝𝑓) −𝜖𝛿𝐮𝐼(𝐮𝑠−𝐮𝑓),
𝜕
𝜕𝑡𝜙𝑠+ 𝐮𝐼⋅∇𝜙𝑠= 1
𝜖𝜇(𝑝𝑠−𝑝𝑓).
(8)
The above formulation (8) shows that the stiﬀness is present in the momentum equations, due to the acoustic waves, but also in 
the energy equations. More precisely, following [43], the energy ﬂux can be split into two contributions, that is
𝜙𝛼𝐮𝛼(𝜌𝛼𝐸𝛼+ 𝑝𝛼) = 𝜖𝜌𝛼𝜙𝛼𝑘𝛼𝐮𝛼+ 𝜌𝛼𝜙𝛼ℎ𝛼𝐮𝛼,
(9)
Journal of Computational Physics 539 (2025) 114227 
4 


B. Battisti and W. Boscheri
with the enthalpy deﬁned as
ℎ𝛼= 𝑒𝛼+ 𝑝𝛼
𝜌𝛼
.
(10)
Flows at low Mach and high Mach number behave quite diﬀerently. For low Mach, the regime is quasi-incompressible, with small 
pressure variations, and with convection and diﬀusion domination. On the contrary, for high Mach numbers, the pressure varies 
signiﬁcantly and the ﬂow is highly compressible, with the appearance of shock waves. Those behaviors are visible also from the 
scaled system (8), where the stiﬀness parameter may assume values 𝜖→0 or 𝜖≫1. Along the lines of Re and Abgrall[46], a Mach 
number common to both phases is identiﬁed to expresses the overall compressibility of the mixture ﬂow ﬁeld. The divergence-free 
constraint of the velocity ﬁeld in the low Mach regime will then be recovered for the mixture velocity, see Section 3.6.
3.  Numerical scheme
We design a semi-implicit numerical scheme for the solution of the Baer–Nunziato model (1), which is second order accurate in 
space and time. The method is proven to be well-balanced and asymptotic preserving in the low Mach number limit of the mixture. 
Firstly, we solve the homogeneous model over a time step Δ𝑡, by neglecting the source terms. Next, we apply a relaxation operator, 
to solve the system of ordinary diﬀerential equations (ODE) that is left. Regardless the Mach regime, the time step is unconstrained 
from the acoustic waves, even in the limit 𝜖→0, and the scheme obeys the following stability condition:
Δ𝑡≤CFL min
Ω
𝜚
max(max(|𝐮𝑓⋅𝐧|, |𝐮𝑠⋅𝐧|)),
(11)
which is only based on the maximum material speed of the phases.
3.1.  First order semi-discrete scheme in time
Let us assume 𝐒(𝐐) = 𝟎 in (5), and let us adopt the general subscript 𝛼 for the two phases. The time coordinate is deﬁned in the 
interval 𝑡∈[0, 𝑡𝑓], with 𝑡𝑓 the ﬁnal time. The solution advances in time as 𝑡𝑛+1 = 𝑡𝑛+ Δ𝑡, in which the time step Δ𝑡 is deﬁned at 
each iteration according to the CFL condition (11). The discretization in time is based on a semi-implicit formulation, that yields the 
following scheme: 
(𝜌𝛼𝜙𝛼)𝑛+1 = (𝜌𝛼𝜙𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
),
(12a)
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1 = (𝜌𝛼𝜙𝛼𝐮𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝐮𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
) −Δ𝑡∇(𝜙𝛼𝑝𝛼)𝑛+1 + Δ𝑡𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
,
(12b)
(𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1 = (𝜌𝛼𝜙𝛼𝐸𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝑘𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
) −Δ𝑡∇⋅(ℎ𝑛
𝛼(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1) + Δ𝑡𝑝𝑛
𝐼𝐮𝑛
𝐼∇𝜙𝑛+1
𝛼
,
(12c)
𝜙𝑛+1
𝑠
= 𝜙𝑛
𝑠−Δ𝑡𝐮𝑛
𝐼∇𝜙𝑛+1
𝑠
,
(12d)
where the splitting of the energy ﬂux (9) in (12c) allows us to treat the part related to the kinetic energy explicitly, while the part 
related to the enthalpy implicitly. It is evident that the nonlinear convective terms are treated explicitly, and the implicit terms are 
the stiﬀ ones, involving pressure and volume fraction gradients. However, the volume fraction in the convective ﬂuxes is discretized 
implicitly in order to ensure the well-balanced property (see Section 3.7). Since the volume fraction Eq. (12d) can be readily solved, 
the quantity 𝜙𝑛+1
𝑠 is computed ﬁrst, and then it can be used in the other equations, making the nonlinear convective ﬂuxes explicitly 
computable. We underline that the scheme is linearly implicit, meaning that any non-linearity in time is avoided. Indeed, in our 
semi-implicit scheme the interface quantities are considered explicitly, as well as the enthalpy in the energy ﬂux.
Let us introduce the following abbreviations to account for the explicit contributions: 
𝜌⋆
𝛼= (𝜌𝛼𝜙𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
),
(13a)
𝐪⋆
𝛼= (𝜌𝛼𝜙𝛼𝐮𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝐮𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
),
(13b)
𝐸⋆
𝛼= (𝜌𝛼𝜙𝛼𝐸𝛼)𝑛−Δ𝑡∇⋅((𝜌𝛼𝑘𝛼𝐮𝛼)𝑛𝜙𝑛+1
𝛼
).
(13c)
Letting aside the volume fraction Eq. (12d), which is the ﬁrst one to be solved, and using the above deﬁnitions, the remaining system 
can now be written as 
(𝜌𝛼𝜙𝛼)𝑛+1 = 𝜌⋆
𝛼,
(14a)
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1 = 𝐪⋆
𝛼−Δ𝑡∇(𝜙𝛼𝑝𝛼)𝑛+1 + Δ𝑡𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
,
(14b)
(𝜌𝛼𝜙𝛼𝑒𝛼)𝑛+1 + (𝜌𝛼𝜙𝛼𝑘𝛼)𝑛+1 = 𝐸⋆
𝛼−Δ𝑡∇⋅(ℎ𝑛
𝛼(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1) + Δ𝑡𝑝𝑛
𝐼𝐮𝑛
𝐼∇𝜙𝑛+1
𝛼
.
(14c)
In the energy Eq. (14c), the total energy at the new time level is split into the internal and the kinetic energy contributions. In 
particular, to guarantee a linearly implicit scheme, we resort to the semi-implicit linearization proposed in Boscheri and Pareschi[43], 
hence taking
(𝜌𝛼𝜙𝛼𝑘𝛼)𝑛+1 ∶= 1
2
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1.
(15)
Journal of Computational Physics 539 (2025) 114227 
5 


B. Battisti and W. Boscheri
Inserting the momentum Eq. (14b) into the energy Eq. (14c) and expressing the internal energy as a function of the pressure by means 
of the EOS (2), we obtain a wave equation of the form
(𝜙𝛼𝑝𝛼)𝑛+1
𝛾𝛼−1
+
𝜙𝑛+1
𝛼
𝛾𝛼𝜋𝛼
𝛾𝛼−1
+ 1
2
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛
(𝐪⋆
𝛼−Δ𝑡∇(𝜙𝛼𝑝𝛼)𝑛+1 + Δ𝑡𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
) =
= 𝐸⋆
𝛼−Δ𝑡∇⋅(ℎ𝑛
𝛼𝐪⋆
𝛼) + Δ𝑡2 ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) −Δ𝑡2 ∇⋅(ℎ𝑛
𝛼𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
) + Δ𝑡𝑝𝑛
𝐼𝐮𝑛
𝐼∇𝜙𝑛+1
𝛼
.
(16)
Shifting all the unknown pressure elements on the left-hand-side, we get
(𝜙𝛼𝑝𝛼)𝑛+1
𝛾𝛼−1
+
𝜙𝑛+1
𝛼
𝛾𝛼𝜋𝛼
𝛾𝛼−1
−Δ𝑡
2
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛∇(𝜙𝛼𝑝𝛼)𝑛+1 −Δ𝑡2 ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) =
= 𝐸⋆
𝛼−1
2 𝐪⋆
𝛼
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛
−Δ𝑡
2 𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛
−Δ𝑡∇⋅(ℎ𝑛
𝛼𝐪⋆
𝛼) −Δ𝑡2 ∇⋅(ℎ𝑛
𝛼𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
) + Δ𝑡𝑝𝑛
𝐼𝐮𝑛
𝐼∇𝜙𝑛+1
𝛼
.
(17)
The stiﬀened gas EOS ensures a linear relation between the internal energy and the pressure, hence yielding a system of linear 
equations that can be directly solved for the unknown quantities (𝜙𝛼𝑝𝛼)𝑛+1. It must be noted that, using a diﬀerent EOS, this property 
may not hold any more, but the pressure formulation keeps the non-linearity only along the diagonal of the system matrix [42,43]. 
In this case, the nested Newton technique proposed in Brugnano and Casulli[58] can be eﬃciently applied. The solution of the above 
system is only meant to provide a stable numerical ﬂux to be used in the conservative update of momentum and total energy according 
to (14b) and (14c), diﬀerently from the non-conservative pressure update proposed in Re and Abgrall[46]. In principle, the wave 
equation could be solved for the total energy as done in Boscarino et al.[45] for single-phase ﬂows, which however would yield a 
fully nonlinear system for general EOS. Finally, we remark that the enthalpy must be discretized as
ℎ𝑛
𝛼∶=
𝜌𝑛
𝛼ℎ𝑛
𝛼
𝜌𝑛+1
𝛼
,
(18)
in order to achieve the preservation of constant velocity and pressure ﬂows at the discrete level, as previously studied in [43] for 
single-phase ﬂows. A more detailed analysis concerning the well-balanced property of the scheme is conducted in Section 3.7.
Remark 1 (On the thermodynamic consistency of the total energy). Once the pressure wave Eq. (17) is solved, the total energy is 
updated with (14c). By doing so, thermodynamic consistency is violated up to the time order of accuracy of the scheme. Indeed, the 
new total energy of our semi-discrete scheme is equivalent to
(𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1 = (𝜙𝛼𝑝𝛼)𝑛+1
𝛾𝛼−1
+
𝜙𝑛+1
𝛼
𝛾𝛼𝜋𝛼
𝛾𝛼−1
+ 1
2
(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛
(𝜌𝛼𝜙𝛼)𝑛(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1,
therefore the error is of order (Δ𝑡𝑞), with 𝑞 being the order of the method. To obtain exact thermodynamic consistency, one can 
avoid the semi-implicit linearization of the kinetic energy (15), which implies the solution of a nonlinear system, and apply a ﬁxed 
point iteration method, see for instance [42,59]. Another option, recently forwarded in Boscheri et al.[60], is the following one: the 
pressure system is solved, for the ﬁrst time, with the proposed linearization of the kinetic energy, for obtaining a pressure ﬂux. This 
pressure will readily be used to update the momentum according to (14b). Next, the pressure system is solved for the second time for 
the pressure state, hence employing the kinetic energy at the new time level which is known at this stage. At the price of two linear 
systems to be solved, thermodynamic consistency is strictly ensured.
3.2.  Second order extension in time
Second order accuracy in time is achieved relying on the class of semi-implicit IMEX schemes forwarded in Boscarino et al.[41]. 
Here, we limit us to brieﬂy recall the procedure, which starts by writing the governing Eq. (5) under the form of an autonomous 
system, that is
𝜕𝐐
𝜕𝑡= (𝐐𝐸(𝑡), 𝐐𝐼(𝑡)),
∀𝑡> 0,
𝐐0 = 𝐐(𝑡= 𝑡0).
(19)
The function (𝐐𝐸(𝑡), 𝐐𝐼(𝑡)) represents the numerical discretization of the spatial operators, and is partitioned in an explicit and an 
implicit argument, respectively. The ﬁrst order semi-discrete scheme presented in the previous section ﬁts the general formulation
(19), thus an IMEX Runge–Kutta time marching algorithm can be adopted to achieve second order of accuracy in time. Such schemes 
are, in general, multi-step methods based on 𝑠 stages, for which the following Butcher tableaux can be deﬁned:
̂c
̂A
̂b𝑇
c
A
b𝑇,
(20)
with matrices ( ̂A, A) ∈ℝ𝑠×𝑠 and vectors (̂b, b, ̂c, c) ∈ℝ𝑠. The tableau on the left is applied for the explicit scheme, and the one on the 
right for the implicit scheme. We use the Stiﬄy Accurate semi-implicit Runge–Kutta method LSDIRK2(2,2,2) [41]:
0
0
0
̂c
̂c
0
1 −𝛽
𝛽
𝛽
𝛽
0
1
1 −𝛽
𝛽
1 −𝛽
𝛽
,
(21)
Journal of Computational Physics 539 (2025) 114227 
6 


B. Battisti and W. Boscheri
with 𝛽= 1 −1∕
√
2 and ̂c = 1∕(2𝛽). For further information and details on the semi-implicit IMEX schemes, the reader is referred to 
[41], while applications to asymptotic preserving low Mach number schemes can be found, for instance, in Avgerinos et al.[37],
Boscheri and Pareschi[43], Boscheri and Tavelli[44].
3.3.  Fully discrete scheme in space and time
The discrete computational domain is deﬁned, for 𝑑= 3, by Ω(𝐱) = [𝑥min, 𝑥max] × [𝑦min, 𝑦max] × [𝑧min, 𝑧max], and it is paved with a 
Cartesian grid of characteristic dimensions
Δ𝑥= 𝑥max −𝑥min
𝑁𝑥
Δ𝑦= 𝑦max −𝑦min
𝑁𝑦
Δ𝑧= 𝑧max −𝑧min
𝑁𝑧
,
(22)
with {𝑁𝐱, 𝑁𝐲, 𝑁𝐳} representing the number of cells in each spatial direction, hence accounting for a total of 𝑁𝑒= 𝑁𝑥× 𝑁𝑦× 𝑁𝑧
elements. The index triplet (𝑖, 𝑗, 𝑘) permits to uniquely deﬁne a cell 𝐶𝑖,𝑗,𝑘 of volume |𝐶𝑖,𝑗,𝑘| = Δ𝑥Δ𝑦Δ𝑧 and center 𝐱𝑖,𝑗,𝑘= (𝑥𝑖, 𝑦𝑗, 𝑧𝑘). 
The cell faces in the diﬀerent directions are labeled with indexes (𝑖± 1∕2, 𝑗, 𝑘), (𝑖, 𝑗± 1∕2, 𝑘), (𝑖, 𝑗, 𝑘± 1∕2), have center points at 
𝐱𝑖±1∕2,𝑗,𝑘= ( 𝑥𝑖+𝑥𝑖±1
2
, 𝑦𝑗, 𝑧𝑘), 𝐱𝑖,𝑗±1∕2,𝑘= (𝑥𝑖,
𝑦𝑗+𝑦𝑗±1
2
, 𝑧𝑘), and 𝐱𝑖,𝑗,𝑘±1∕2 = (𝑥𝑖, 𝑦𝑗, 𝑧𝑘+𝑧𝑘±1
2
), and normal vectors 𝐧𝑥= (1, 0, 0), 𝐧𝑦= (0, 1, 0), and 
𝐧𝑧= (0, 0, 1).
For sake of clarity, we present the numerical scheme for the one-dimensional case considering the 𝑥-direction only, bearing in 
mind that the extension to the remaining spatial directions is straightforward.
The volume fraction Eq. (12d) is the ﬁrst to be solved, and contains only non-conservative terms. The chosen discretization, in 
ﬂuctuation form, writes
𝜙𝑛+1
𝑠,𝑖+ Δ𝑡
2Δ𝑥
(
𝑢𝑛
𝐼,𝑖+1∕2 (𝜙𝑛+1
𝑠,𝑖+1 −𝜙𝑛+1
𝑠,𝑖) −𝑢𝑛
𝐼,𝑖−1∕2 (𝜙𝑛+1
𝑠,𝑖−1 −𝜙𝑛+1
𝑠,𝑖)
)
= 𝜙𝑛
𝑠,𝑖,
(23)
where the interface velocities are evaluated as
𝑢𝑛
𝐼,𝑖+1∕2 = 1
2
(
𝑢𝑛
𝐼,𝑖+1 + 𝑢𝑛
𝐼,𝑖
)
,
𝑢𝑛
𝐼,𝑖−1∕2 = 1
2
(
𝑢𝑛
𝐼,𝑖+ 𝑢𝑛
𝐼,𝑖−1
)
,
(24)
using one of the deﬁnitions (3a) and (3b). The above discretization (23) is conceived such that the scheme can preserve a constant 
pressure and velocity ﬁeld for a traveling contact wave [46], thus it follows from physical observations. However, it also worth 
noticing that the same discretization can be retrieved by means of the path-conservative approach (see Parés[61] and references 
therein) using a simple straight line segment path and a midpoint integration rule for the evaluation of the integral of the jump terms 
across the element boundaries. The linear system (23) is solved at the aid of the GMRES method [62], where we prescribe a tolerance 
of 10−12 to stop the iterative procedure.
Once the new volume fraction 𝜙𝑛+1
𝑠,𝑖 is known, the explicit terms are solved relying on a ﬁnite volume method:
𝑚⋆
𝑖= 𝑚𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝑚
𝑖+1∕2 −𝑓𝑚
𝑖−1∕2
)
,
(25)
where 𝑚 represents any variable which we solve for in the Baer–Nunziato equations, and 𝑓𝑚
𝑖±1∕2 are the numerical ﬂuxes given by
𝑓𝑚
𝑖+1∕2 = 1
2 (𝑓(𝑚𝑛
𝑖+1) + 𝑓(𝑚𝑛
𝑖)) −1
2 𝑎𝑛
Ω(𝑚𝑛
𝑖+1 −𝑚𝑛
𝑖),
𝑓𝑚
𝑖−1∕2 = 1
2 (𝑓(𝑚𝑛
𝑖) + 𝑓(𝑚𝑛
𝑖−1)) −1
2 𝑎𝑛
Ω(𝑚𝑛
𝑖−𝑚𝑛
𝑖−1),
(26)
with 𝑓(⋅) being the physical ﬂux related to the 𝑚 variable. We use a global Lax-Friedrichs approximate Riemann solver, thus the 
numerical dissipation is computed as
𝑎𝑛
Ω = max
Ω
(
max(|𝑢𝑛
𝑓|, |𝑢𝑛
𝑠|)
)
,
(27)
considering only the eigenvalues associated to the convective terms. We remark that the numerical dissipation in the energy equations 
takes into account the jump of the kinetic energy only, and not the jump of the total energy, which would also contain the internal 
energy contribution. The ﬁnite volume scheme (26) is, thus, used for the computation of the 𝑚⋆ quantities in (13a)–(13c): 
𝜌⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼
𝑖+1∕2 −𝑓𝜌𝛼𝜙𝛼
𝑖−1∕2
)
= (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
,
(28a)
𝑞⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼𝑢𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼𝑢𝛼
𝑖+1∕2
−𝑓𝜌𝛼𝜙𝛼𝑢𝛼
𝑖−1∕2
)
,
(28b)
𝐸⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼𝐸𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼𝑘𝛼
𝑖+1∕2
−𝑓𝜌𝛼𝜙𝛼𝑘𝛼
𝑖−1∕2
)
.
(28c)
For the implicit terms in the momentum and energy Eqs. (14b) and (14c), central ﬁnite diﬀerence operators are used, which 
provide second order of accuracy on Cartesian grids. The wave equation for the pressure (17) needs to be solved ﬁrst, and it is 
Journal of Computational Physics 539 (2025) 114227 
7 


B. Battisti and W. Boscheri
discretized as follows:
(𝜙𝛼𝑝𝛼)𝑛+1
𝛾𝛼−1
+
𝜙𝑛+1
𝛼
𝛾𝛼𝜋𝛼
𝛾𝛼−1
−Δ𝑡
4Δ𝑥
(𝜌𝛼𝜙𝛼𝑢𝛼)𝑛
𝑖
(𝜌𝛼𝜙𝛼)𝑛
𝑖
(
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖+1 −(𝜙𝛼𝑝𝛼)𝑛+1
𝑖−1
)
−Δ𝑡2 ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) =
= 𝐸⋆
𝛼,𝑖−1
2𝑞⋆
𝛼,𝑖
(𝜌𝛼𝜙𝛼𝑢𝛼)𝑛
𝑖
(𝜌𝛼𝜙𝛼)𝑛
𝑖
−Δ𝑡
4Δ𝑥
(
𝑝𝑛
𝐼,𝑖+1∕2(𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖) −𝑝𝑛
𝐼,𝑖−1∕2(𝜙𝑛+1
𝛼,𝑖−1 −𝜙𝑛+1
𝛼,𝑖)
)(𝜌𝛼𝜙𝛼𝑢𝛼)𝑛
𝑖
(𝜌𝛼𝜙𝛼)𝑛
𝑖
−Δ𝑡
2Δ𝑥
((ℎ𝑛
𝛼𝑞⋆
𝛼)𝑖+1 −(ℎ𝑛
𝛼𝑞⋆
𝛼)𝑖−1
) −Δ𝑡2 ∇⋅(ℎ𝑛
𝛼𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
)
+ Δ𝑡
2Δ𝑥
(
𝑝𝑛
𝐼,𝑖+1∕2𝑢𝑛
𝐼,𝑖+1∕2(𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖) −𝑝𝑛
𝐼,𝑖−1∕2𝑢𝑛
𝐼,𝑖−1∕2(𝜙𝑛+1
𝛼,𝑖−1 −𝜙𝑛+1
𝛼,𝑖)
)
,
(29)
where the same ﬂuctuation form of (23) has been adopted for the non-conservative terms involving the gradient of the volume 
fraction ∇𝜙𝑛+1
𝑠
. Let us notice that two terms in the above Eq. (29) are still kept continuous in space, as they need careful attention 
in their discretization to ensure a compact stencil and the well-balanced property for moving equilibrium solutions. In particular, a 
consistent discretization of the ﬁrst term could be
Δ𝑡2 ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) = Δ𝑡2
2Δ𝑥
(
ℎ𝑛
𝛼,𝑖+1
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖+2 −(𝜙𝛼𝑝𝛼)𝑛+1
𝑖
2Δ𝑥
−ℎ𝑛
𝛼,𝑖−1
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖
−(𝜙𝛼𝑝𝛼)𝑛+1
𝑖−2
2Δ𝑥
)
.
(30)
However, this approximation would widen the stencil, as it would span {𝑖−2, 𝑖−1, 𝑖, 𝑖+ 1, 𝑖+ 2}. Using Lagrange interpolation poly-
nomials of second degree on the classical stencil {𝑖−1, 𝑖, 𝑖+ 1}, following the approach forwarded in Boscheri and Pareschi[43], 
would instead lead to the following compact expression:
Δ𝑡2 ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) = Δ𝑡2
Δ𝑥2
[
ℎ𝑛
𝛼,𝑖−1
ℎ𝑛
𝛼,𝑖
ℎ𝑛
𝛼,𝑖+1
]⎡
⎢
⎢⎣
3∕4
−1
1∕4
0
0
0
1∕4
−1
3∕4
⎤
⎥
⎥⎦
⎡
⎢
⎢⎣
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖−1
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖
(𝜙𝛼𝑝𝛼)𝑛+1
𝑖+1
⎤
⎥
⎥⎦
+ (Δ𝑥2),
(31)
which is still second order accurate. The second term Δ𝑡2 ∇⋅(ℎ𝑛
𝛼𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
) is approximated by means of the same second order discrete 
operator (31), which is proven to guarantee constant pressure and velocity preservation across a contact discontinuity, as detailed in 
Section 3.7.
We solve the pressure system (29) directly for the quantity (𝜙𝛼𝑝𝛼)𝑛+1 using again the GMRES solver. Next, the momentum (14b), 
and then the energy (14c) equations are ﬁnally updated: 
(𝜌𝛼𝜙𝛼𝑢𝛼)𝑛+1
𝑖
= 𝑞⋆
𝛼,𝑖−Δ𝑡
2Δ𝑥((𝜙𝛼𝑝𝛼)𝑛+1
𝑖+1 −(𝜙𝛼𝑝𝛼)𝑛+1
𝑖−1 ) + Δ𝑡
2Δ𝑥
(
𝑝𝑛
𝐼,𝑖+1∕2(𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖) −𝑝𝑛
𝐼,𝑖−1∕2(𝜙𝑛+1
𝛼,𝑖−1 −𝜙𝑛+1
𝛼,𝑖)
)
(32a)
(𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1
𝑖
= 𝐸⋆
𝛼,𝑖−Δ𝑡
2Δ𝑥
((ℎ𝑛
𝛼(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1)𝑖+1 −(ℎ𝑛
𝛼(𝜌𝛼𝜙𝛼𝐮𝛼)𝑛+1)𝑖−1
)+
+ Δ𝑡
2Δ𝑥
(
𝑝𝑛
𝐼,𝑖+1∕2𝑢𝑛
𝐼,𝑖+1∕2(𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖) −𝑝𝑛
𝐼,𝑖−1∕2𝑢𝑛
𝐼,𝑖−1∕2(𝜙𝑛+1
𝛼,𝑖−1 −𝜙𝑛+1
𝛼,𝑖)
)
.
(32b)
3.4.  Second order numerical dissipation
As the central ﬁnite diﬀerence method is inherently second order accurate on a Cartesian grid of uniform spacing Δ𝑥, no other 
action is needed for the spatial operators deﬁned in Section 3.3. However, particular care must be paid to the deﬁnition of the 
numerical viscosity in the ﬁnite volume ﬂuxes (26), as that must also be of order (Δ𝑥2) to preserve the scheme’s overall accuracy. 
Therefore, any generic cell centered quantity 𝑚𝑛
𝑖 undergoes a linear reconstruction procedure to compute the polynomial 𝑤𝑛
𝑖(𝑥) deﬁned 
as
𝑤𝑛
𝑖(𝑥) ∶= 𝑚𝑛
𝑖+ (𝑥−𝑥𝑖) 𝛼𝑥,𝑖.
(33)
The unknown reconstruction coeﬃcient 𝛼𝑥,𝑖 is evaluated using the minmod limiter, that is
𝛼𝑥,𝑖=
⎧
⎪
⎨
⎪⎩
0
if
𝛼𝑥,𝓁𝛼𝑥,𝑒𝑟𝑟≤0
𝛼𝑥,𝓁
if
|𝛼𝑥,𝓁| ≤|𝛼𝑥,𝑒𝑟𝑟|
𝛼𝑥,𝑒𝑟𝑟
if
|𝛼𝑥,𝓁| > |𝛼𝑥,𝑒𝑟𝑟|
,
𝛼𝑥,𝓁=
𝑚𝑛
𝑖−𝑚𝑛
𝑖−1
Δ𝑥
,
𝛼𝑥,𝑒𝑟𝑟=
𝑚𝑛
𝑖+1 −𝑚𝑛
𝑖
Δ𝑥
.
(34)
The numerical dissipation jumps in (26) are then given in terms of the second order boundary extrapolated data within each cell, 
hence
𝐷𝑚
𝑖+1∕2 = 1
2 𝑎𝑛
Ω(𝑤𝑛
𝑖+1(𝑥𝑖+1∕2) −𝑤𝑛
𝑖(𝑥𝑖+1∕2)),
𝐷𝑚
𝑖−1∕2 = 1
2 𝑎𝑛
Ω(𝑤𝑛
𝑖(𝑥𝑖−1∕2) −𝑤𝑛
𝑖−1(𝑥𝑖−1∕2)),
(35)
while letting the central ﬂux contribution untouched, as written in (26). The same procedure applies whenever an artiﬁcial viscosity 
term is added to the scheme.
Journal of Computational Physics 539 (2025) 114227 
8 


B. Battisti and W. Boscheri
Numerical dissipation in the volume fraction equation. For strong gradients in the volume fraction distribution, an additional stabiliza-
tion term might be needed in (23), that is taken explicitly on the right hand side:
𝜙𝑛+1
𝑠,𝑖+ Δ𝑡
2Δ𝑥
(
𝑢𝑛
𝐼,𝑖+1∕2 (𝜙𝑛+1
𝑠,𝑖+1 −𝜙𝑛+1
𝑠,𝑖) −𝑢𝑛
𝐼,𝑖−1∕2 (𝜙𝑛+1
𝑠,𝑖−1 −𝜙𝑛+1
𝑠,𝑖)
)
= 𝜙𝑛
𝑠,𝑖+ Δ𝑡
Δ𝑥2
(
𝜈+ (𝜙𝑛
𝑠,𝑖+1 −𝜙𝑛
𝑠,𝑖) −𝜈−(𝜙𝑛
𝑠,𝑖−𝜙𝑛
𝑠,𝑖−1)
)
,
(36)
with a viscosity coeﬃcient chosen as
𝜈±(𝜎) =
𝜎𝑛
𝑖+ 𝜎𝑛
𝑖±1
2
,
𝜎𝑛
𝑖= 1
2 Δ𝑥𝑎𝑛
Ω.
(37)
We remark that the artiﬁcial viscosity contribution introduced in (36) must be accounted for also in the pressure system (29), and in 
the conservative update of the total energy (32b), with the following rescaled dissipation coeﬃcients 𝜈±(𝜎⋆):
𝜎𝑛
𝑖,⋆= 𝜎𝑛
𝑖
𝜕(𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖
𝜕𝜙𝑛
𝛼,𝑖
.
(38)
More details can be found in Section 3.7, where the well-balanced property of the scheme is shown.
Numerical dissipation in the energy equation. For further stabilization at high Mach number ﬂows, the convective ﬂuxes (26) related 
to the energy equation may be supplemented with an artiﬁcial viscosity term on the pressure jump:
𝑓𝜌𝛼𝜙𝛼𝑘𝛼
𝑖+1∕2
=1
2
( ((𝜌𝛼𝑘𝛼𝑢𝛼)𝑛𝜙𝑛+1
𝛼
)𝑖+1 + ((𝜌𝛼𝑘𝛼𝑢𝛼)𝑛𝜙𝑛+1
𝛼
)𝑖
) −1
2 𝑎𝑛
Ω
( (𝜌𝛼𝜙𝛼𝑘𝛼)𝑛
𝑖+1 −(𝜌𝛼𝜙𝛼𝑘𝛼)𝑛
𝑖
)
−1
Δ𝑥𝜈+(𝜎⋆⋆)
(
𝑝𝑛
𝛼,𝑖+1 −𝑝𝑛
𝛼,𝑖
)
,
(39)
with the rescaled coeﬃcients 𝜈±(𝜎⋆⋆) given by
𝜎𝑛
𝑖,⋆⋆= 𝜎𝑛
𝑖
𝜕(𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖
𝜕𝑝𝑛
𝛼,𝑖
.
(40)
3.5.  Relaxation solver
Until this point, the numerical scheme has been presented for the homogeneous version of the Baer–Nunziato system, thus the 
new solution obtained with the semi-implicit scheme detailed in Section 3.3 is labeled with 𝐐𝐻, according to the compact notation 
introduced in (5). A Strang splitting scheme [63] is here adopted, thus we aim at solving the remaining ODE system:
𝐐𝑛+1
𝑖
−𝐐𝐻
𝑖
Δ𝑡
= 𝐒(𝐐𝑛+1
𝑖
),
𝑖= 1, … , 𝑁𝑥,
(41)
where the vector of potentially stiﬀ source terms is discretized using again a semi-implicit scheme, since we want to avoid any 
non-linearity. Speciﬁcally, only in the 𝑥-direction, the sources are discretized as
𝐒(𝐐𝑛+1
𝑖
) =
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
0
𝛿(𝑢𝑛+1
𝑠,𝑖−𝑢𝑛+1
𝑓,𝑖)
𝜇𝑝𝐻
𝐼,𝑖(𝑝𝐻
𝑠,𝑖−𝑝𝐻
𝑓,𝑖) + 𝛿𝑢𝐻
𝐼,𝑖(𝑢𝑛+1
𝑠,𝑖−𝑢𝑛+1
𝑓,𝑖)
0
−𝛿(𝑢𝑛+1
𝑠,𝑖−𝑢𝑛+1
𝑓,𝑖)
−𝜇𝑝𝐻
𝐼,𝑖(𝑝𝐻
𝑠,𝑖−𝑝𝐻
𝑓,𝑖) −𝛿𝑢𝐻
𝐼,𝑖(𝑢𝑛+1
𝑠,𝑖−𝑢𝑛+1
𝑓,𝑖)
𝜇(𝑝𝐻
𝑠,𝑖−𝑝𝐻
𝑓,𝑖)
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
.
(42)
The pressure terms are directly taken from the solution of the wave Eq. (29), thus they are readily computable. Inspired from 
Re and Abgrall[46], the new velocities (𝑢𝑛+1
𝑠,𝑖, 𝑢𝑛+1
𝑓,𝑖) are evaluated by solving a linear algebraic system, locally deﬁned within each 
computational cell. Once the velocities have been obtained, the sources in the energy equations are ﬁnally computed. Second order 
of accuracy in space is automatically achieved, since the computations are performed at the cell center, which corresponds to the 
midpoint quadrature rule. For time accuracy, we resort to the semi-implicit IMEX scheme (21).
Summary of the linearly semi-implicit scheme. The numerical scheme is now complete, and it can be summarized by the following 
steps.
1. Solve implicitly the equation of the volume fraction (23) with the GMRES iterative method, and obtain 𝜙𝑛+1
𝑠
.
2. Solve explicitly the convective ﬂuxes with the ﬁnite volume scheme (25) and (26), and obtain 𝜌⋆
𝛼, 𝑞⋆
𝛼, and 𝐸⋆
𝛼. We recall that 
𝜌⋆
𝛼= (𝜌𝛼𝜙𝛼)𝑛+1.
3. Solve the linearly implicit wave Eq. (29) directly for (𝜙𝛼𝑝𝛼)𝑛+1 with the GMRES iterative method.
4. Update the momentum equations according to (32a), and obtain (𝜌𝛼𝜙𝛼𝑢𝛼)𝑛+1.
5. Update the energy equations with (32b), and obtain (𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1.
6. Starting from the previously obtained solution of the homogeneous system, relax the system through the source terms, evaluated 
with the semi-implicit discretization (42).
Journal of Computational Physics 539 (2025) 114227 
9 


B. Battisti and W. Boscheri
3.6.  Asymptotic preserving property
Given a mathematical model characterized by a stiﬀness parameter 𝜖, a numerical method which solves this problem over time 
with step size Δ𝑡 is considered Asymptotic Preserving (AP) if it retrieves, at the discrete level, a consistent and stable discretization 
of the limit model obtained for 𝜖→0, for any Δ𝑡 independent of 𝜖. For single-phase ideal compressible gases, this means that an AP 
scheme can preserve the correct asymptotic transition from the compressible to the incompressible equations, as the Mach number 
tends to zero.
The numerical method presented in this work is AP, as it remains accurate and stable in the stiﬀ regime. To prove it, we ﬁrst 
derive the asymptotic limit of the continuous system of equations, and then we analyze the temporal semi-discrete scheme to show 
that it leads to a consistent discretization of the incompressible model for 𝜖→0.
Limit model of the mixture equations. Apart from the pioneering analysis of Varsakelis and Papalexandris[64], low Mach asymptotics 
have not yet been generalized to multi-phase ﬂow models. It is yet not clear how to derive asymptotic limits for multi-phase ﬂows, 
since the Mach number can be deﬁned either separately for each phase, or uniquely for the mixture. Here, we analyze the low Mach 
number limit of the mixture model. Let us assume a generic bounded domain Ω, with impermeable boundary conditions on 𝜕Ω
𝐮𝛼⋅𝐧= 0,
(43)
where 𝐧 denotes the unit outward normal vector to the boundary 𝜕Ω, and let the 𝑘th order power expansion for the generic variable 
𝑚 be given in terms of the parameter 𝜖 as
𝑚= 𝑚(0) + 𝜖𝑚(1) + 𝜖2𝑚(2) + … + (𝜖𝑘).
(44)
By assuming well-prepared initial data, all the variables of the non-dimensional governing Eq. (8) can be expanded with respect to 
the common stiﬀness parameter 𝜖 according to (44).
Substituting the expansion (44) in the non-dimensional volume fraction equation of (8) and assuming a ﬁnite relaxation parameter 
𝛿, one immediately obtains
(𝜖−1) ∶
𝜇(𝑝𝑠(0) −𝑝𝑓(0)) = 0
⇒
𝑝𝑠(0) = 𝑝𝑓(0) = 𝑝𝐼(0) = 𝑝(0),
(45)
and considering that 𝜙𝑓 is governed by the same equation as 𝜙𝑠, in general we have
(𝜖0) ∶
𝜕𝑡𝜙𝛼(0) + 𝑢𝐼(0)∇𝜙𝛼(0) = ±𝜇(𝑝𝑠(1) −𝑝𝑓(1)).
(46)
The continuity equation at the 0−th leading order reads
𝜕𝑡(𝜌𝛼(0)𝜙𝛼(0)) + ∇⋅(𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0)) = 0.
(47)
By collecting like powers of 𝜖 in the momentum equation, we have 
(𝜖−1) ∶
∇(𝑝𝛼(0)𝜙𝛼(0)) −𝑝𝐼(0)∇𝜙𝛼(0) = 0,
(48a)
(𝜖0) ∶
𝜕𝑡(𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0)) + ∇⋅(𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0)𝐮𝛼(0)) + ∇(𝑝𝛼(1)𝜙𝛼(1)) −𝑝𝐼(1)∇𝜙𝛼(1) = ∓𝛿(𝐮𝑠(0) −𝐮𝑓(0)).
(48b)
Summing Eq. (48a) for both phases, using the result of (45), and recalling that the volume fractions sum up to unity, one gets
∇(𝑝𝑠(0)𝜙𝑠(0) + 𝑝𝑓(0)𝜙𝑓(0)) = 𝑝𝐼(0)∇(𝜙𝑠(0) + 𝜙𝑓(0)),
∇(𝑝(0)(𝜙𝑠(0) + 𝜙𝑓(0))) = 𝑝(0) ⋅0,
∇𝑝(0) = 0
⇒
𝑝(0) = 𝑝(0)(𝑡),
(49)
which shows that, at the 0−th leading order, the pressure is constant in space, and changes in time only due to boundary conditions. 
Finally, from the energy equation, we get 
(𝜖0) ∶
𝜕𝑡(𝜌𝛼(0)𝜙𝛼(0)𝑒𝛼(0)) + ∇⋅(𝜌𝛼(0)𝜙𝛼(0)ℎ𝛼(0)𝐮𝛼(0)) −𝑝𝐼(0)𝐮𝐼(0)∇𝜙𝛼(0) = ∓𝜇𝑝𝐼(1)(𝑝𝑠(1) −𝑝𝑓(1)),
(50a)
(𝜖1) ∶
𝜕𝑡(𝜌𝛼(0)𝜙𝛼(0)𝑘𝛼(0)) + ∇⋅(𝜌𝛼(0)𝜙𝛼(0)𝑘𝛼(0)𝐮𝛼(0)) = ∓𝛿𝐮𝐼(0)(𝐮𝑠(0) −𝐮𝑓(0)).
(50b)
For the sake of simplicity, we assume now to deal with the ideal gas EOS, hence setting 𝜋𝛼= 0 in Eq. (2). Substituting the deﬁnitions 
of internal energy and enthalpy as functions of pressure and density, and integrating Eq. (50a) on the domain Ω, we obtain
1
𝛾𝛼−1 ∫Ω
d(𝜙𝛼(0)𝑝(0))
d𝑡
𝑑𝐱+
𝛾𝛼
𝛾𝛼−1 𝑝(0) ∫𝜕Ω
𝜙𝛼(0)𝐮𝛼(0) ⋅𝐧𝑑𝑆−𝑝(0) ∫Ω
𝐮𝐼(0)∇𝜙𝛼(0) 𝑑𝐱= ∓𝜇∫Ω
𝑝𝐼(1)(𝑝𝑠(1) −𝑝𝑓(1)) 𝑑𝐱.
(51)
Now, summing the above equation for both phases, using the impermeable boundary conditions (43), and inserting (46) to compute 
the relaxation source terms, lead to
𝛾𝑓+ 𝛾𝑠−2
(𝛾𝑠−1)(𝛾𝑓−1) |Ω|
d𝑝(0)
d𝑡
−𝑝(0) ∫Ω
𝐮𝐼(0)∇(𝜙𝑠(0) + 𝜙𝑓(0)) 𝑑𝐱= −∫Ω
𝑝𝐼(1)𝑢𝐼(0)∇(𝜙𝑠(0) + 𝜙𝑓(0)) 𝑑𝐱,
(52)
thus all terms vanish except for
d𝑝(0)
d𝑡
= 0,
(53)
Journal of Computational Physics 539 (2025) 114227 
10 


B. Battisti and W. Boscheri
which suggests that, for the mixture, the pressure 𝑝(0) is a constant both in space and time. Considering again Eq. (50a) for the mixture, 
and using the last result on the pressure, we remain with
𝑝(0)∇⋅(𝜙𝑠0𝐮𝑠0 + 𝜙𝑓0𝐮𝑓0) = 0.
(54)
The above equation shows that the mixture velocity is divergence-free, which is typical of an incompressible ﬂow. Moreover, taking 
into account that the relaxation processes are instantaneous, or at least faster than the scale associated to the homogeneous part of 
the system, we can say that 𝐮𝑠0 = 𝐮𝑓0 = 𝐮𝐼0 = 𝐮(0). With this hypothesis, Eq. (54) takes the classical form
∇⋅𝐮(0) = 0.
(55)
We underline that the resulting system composed by this last equation, which is indeed the energy equation, and the leading order 
continuity (47) and momentum (48b) equations computed for the mixture, represents the incompressible Euler model, in the case of 
a single-phase ﬂow. This is the asymptotic limit of the continuous model for the mixture.
Asymptotic preserving scheme of the limit mixture model. We consider the semi-discrete ﬁrst order scheme (12a)–(12d), bearing in mind 
that the second order IMEX method preserves the asymptotic preserving property and is asymptotically accurate [41]. Mimicking the 
analysis carried out at the continuous level, we start by inserting the expansion (44) in the semi-discrete scheme, with the addition 
of the source terms, discretized according to (42). Well-prepared initial data are thus assumed at the discrete level as well. From the 
volume fraction Eq. (12d), we obtain
(𝜖−1) ∶
𝜇(𝑝𝐻
𝑠(0) −𝑝𝐻
𝑓(0)) = 0
⇒
𝑝𝐻
𝑠(0) = 𝑝𝐻
𝑓(0) = 𝑝𝐻
𝐼(0) = 𝑝𝐻
(0).
(56)
In the same way, from the momentum Eq. (12b), it follows that the pressure is constant in space, meaning that Eq. (49) holds true 
also at the semi-discrete level:
∇(𝑝𝐻
𝑠(0)𝜙𝑛+1
𝑠(0) + 𝑝𝐻
𝑓(0)𝜙𝑛+1
𝑓(0) ) = 𝑝𝑛
𝐼(0)∇(𝜙𝑠(0) + 𝜙𝑓(0))𝑛+1,
∇(𝑝𝐻
(0)(𝜙𝑛+1
𝑠(0) + 𝜙𝑛+1
𝑓(0) )) = 𝑝𝑛
𝐼(0) ⋅0,
⇒
∇𝑝𝐻
(0) = 0.
(57)
Therefore, the entire system, at the 0−th leading order, is discretized as follows: 
(𝜌𝛼(0)𝜙𝛼(0))𝑛+1 = (𝜌𝛼(0)𝜙𝛼(0))𝑛−Δ𝑡∇⋅((𝜌𝛼(0)𝐮𝛼(0))𝑛𝜙𝑛+1
𝛼(0) ),
(58a)
(𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0))𝑛+1 = (𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0))𝑛−Δ𝑡∇⋅((𝜌𝛼(0)𝐮𝛼(0)𝐮𝛼(0))𝑛𝜙𝑛+1
𝛼(0) ) −Δ𝑡∇(𝜙𝑛+1
𝛼(1) 𝑝𝐻
𝛼(1)) + Δ𝑡𝑝𝑛
𝐼(1)∇𝜙𝑛+1
𝛼(1) ∓Δ𝑡𝛿(𝐮𝑛+1
𝑠(0) −𝐮𝑛+1
𝑓(0) ),
(58b)
(𝜌𝛼(0)𝜙𝛼(0)𝑒𝛼(0))𝑛+1 = (𝜌𝛼(0)𝜙𝛼(0)𝑒𝛼(0))𝑛−Δ𝑡∇⋅(ℎ𝑛
𝛼(0) (𝜌𝛼(0)𝜙𝛼(0)𝐮𝛼(0))𝑛+1) + Δ𝑡𝑝𝑛
𝐼(0)𝐮𝑛
𝐼(0)∇𝜙𝑛+1
𝛼(0) ∓Δ𝑡𝜇𝑝𝑛+1
𝐼(1) (𝑝𝐻
𝑠(1) −𝑝𝐻
𝑓(1)),
(58c)
𝜙𝑛+1
𝛼(0) = 𝜙𝑛
𝛼(0) −Δ𝑡𝐮𝑛
𝐼(0)∇𝜙𝑛+1
𝛼(0) ± Δ𝑡𝜇(𝑝𝐻
𝑠(1) −𝑝𝐻
𝑓(1)),
(58d)
while the kinetic terms drop from the energy equation, since they are of ﬁrst order in the expansion:
(𝜖1) ∶
(𝜌𝛼(0)𝜙𝛼(0)𝑘𝛼(0))𝑛+1 = (𝜌𝛼(0)𝜙𝛼(0)𝑘𝛼(0))𝑛−Δ𝑡∇⋅((𝜌𝛼(0)𝑘𝛼(0)𝐮𝛼(0))𝑛𝜙𝑛+1
𝛼(0) ) ∓𝛿𝐮𝑛+1
𝐼(0) (𝐮𝑛+1
𝑠(0) −𝐮𝑛+1
𝑓(0) ).
(59)
Using the deﬁnitions of internal energy and enthalpy as functions of pressure and density with the ideal gas EOS (𝜋𝛼= 0 in Eq. (2)), 
in addition to the discretization of the enthalpy according to (18), the energy Eq. (58c) is integrated on the computational domain, 
hence yielding 
𝑝𝑛+1
(0) ∫Ω
𝜙𝑛+1
𝛼0 𝑑𝐱= 𝑝𝑛
(0) ∫Ω
𝜙𝑛
𝛼0 𝑑𝐱−Δ𝑡𝑝𝑛
(0) + ∫𝜕Ω
𝜙𝑛+1
𝛼0 𝐮𝑛+1
𝛼0
⋅𝐧𝑑𝑆+ Δ𝑡𝑝𝑛
(0) ∫Ω
𝐮𝑛
𝐼0∇𝜙𝑛+1
𝛼0 𝑑𝐱∓Δ𝑡𝜇∫Ω
𝑝𝐻
𝐼1(𝑝𝐻
𝑠1 −𝑝𝐻
𝑓1) 𝑑𝐱.
(60)
For the mixture, we sum the equation for both phases, apply the impermeable boundary conditions, and use the relation (56), so that 
we are left with the discrete analogous of Eq. (53):
𝑝𝑛+1
(0) = 𝑝𝑛
(0),
(61)
which means that the pressure at the leading order remains constant for all the time steps. Using the last result to rewrite the energy 
equation, we have
𝑝(0)𝜙𝑛+1
𝛼(0) = 𝑝(0)𝜙𝑛
𝛼(0) −Δ𝑡𝑝(0) ∇⋅(𝜙𝛼(0)𝐮𝛼(0))𝑛+1 + Δ𝑡𝑝(0)𝐮𝑛
𝐼(0)∇𝜙𝑛+1
𝛼(0) ∓Δ𝑡𝜇𝑝𝐻
𝐼1(𝑝𝐻
𝑠(1) −𝑝𝐻
𝑓(1)),
(62)
which, for the mixture, yields ∇⋅(𝜙𝑠(0)𝐮𝑠(0) + 𝜙𝑓(0)𝐮𝑓(0))𝑛+1 = 0. Therefore, we obtain the semi-discrete version of (54), and the scheme 
is asymptotic preserving.
Remark 2. When the phases are treated independently rather than as a mixture, the relaxation terms vanish. However, the results 
presented here cannot be obtained under such conditions. In that case, a diﬀerent approach would be required, involving the deﬁnition 
of distinct Mach numbers for each phase, and treating them separately [51]. For the single-phase case, the above discrete asymptotic 
analysis exactly retrieves the semi-discrete AP scheme proposed in Boscheri and Pareschi[43] for the Euler equations of compressible 
gas dynamics. 
The results of this section prove that the numerical scheme preserves the asymptotic behavior of the mixture problem at the 
discrete level. However, it is important to assure that the numerical dissipation does not dominate for small values of 𝜖, as studied 
in Dellacherie[22]. As discussed in Section 3.4, the numerical viscosity is only proportional to the material velocity of the phases, so 
it is independent of 𝜖 by construction. The additional dissipation (39) given in terms of the pressure jumps is only activated for very 
high Mach number ﬂows.
Journal of Computational Physics 539 (2025) 114227 
11 


B. Battisti and W. Boscheri
3.7.  Well-balanced discretization property
A numerical scheme is well-balanced if it preserves some steady-state solutions of the governing equations also at the discrete 
level. Hereafter, we show that the proposed semi-implicit scheme for the Baer–Nunziato model is well-balanced, as it maintains a 
constant pressure and velocity ﬁeld across a moving contact discontinuity [65]. This means that a multi-phase ﬂow, traveling at 
constant velocity and pressure for all phases, must only transport the volume fraction density, with no other perturbations.
The following proof is treated for the one-dimensional case, but it easily extends to the three-dimensional case. Let Ω = [𝑥𝐿, 𝑥𝑅]
be the computational domain, with 𝑥𝐷∈Ω being the position of the initial discontinuity. The initial condition is deﬁned at 𝑡= 𝑡0, for 
all cells 𝑖= 1, … , 𝑁𝑥, and it writes
𝜌𝛼,𝑖(𝑡0) =
{𝜌𝛼𝐿
𝑥≤𝑥𝐷
𝜌𝛼𝑅
𝑥> 𝑥𝐷
,
𝜙𝛼,𝑖(𝑡0) =
{𝜙𝛼𝐿
𝑥≤𝑥𝐷
𝜙𝛼𝑅
𝑥> 𝑥𝐷
,
𝑢𝛼,𝑖(𝑡0) = 𝑢0,
𝑝𝛼,𝑖(𝑡0) = 𝑝0,
(63)
with 𝑝0 and 𝑢0 being a constant pressure and velocity ﬁeld, respectively. Moreover, 𝜌𝛼𝐿≠𝜌𝛼𝑅 and 𝜙𝛼𝐿≠𝜙𝛼𝑅 are non-negative real 
numbers. This initial condition corresponds to a stationary moving contact wave, since the source terms cancel out, leaving the 
homogeneous system of equations only. The solution of the convective sub-system (28) leads to 
𝜌⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼
𝑖+1∕2 −𝑓𝜌𝛼𝜙𝛼
𝑖−1∕2
)𝑛
∶= (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
,
(64a)
𝑞⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼𝑢𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼𝑢𝛼
𝑖+1∕2
−𝑓𝜌𝛼𝜙𝛼𝑢𝛼
𝑖−1∕2
)𝑛
= 𝑢0
(
(𝜌𝛼𝜙𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼
𝑖+1∕2 −𝑓𝜌𝛼𝜙𝛼
𝑖−1∕2
)𝑛)
= 𝑢0(𝜌𝛼𝜙𝛼)𝑛+1
𝑖
,
(64b)
𝐸⋆
𝛼,𝑖= (𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛
𝑖−Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼𝑘𝛼
𝑖+1∕2
−𝑓𝜌𝛼𝜙𝛼𝑘𝛼
𝑖−1∕2
)𝑛
= (𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛
𝑖−
𝑢2
0
2
Δ𝑡
Δ𝑥
(
𝑓𝜌𝛼𝜙𝛼
𝑖+1∕2 −𝑓𝜌𝛼𝜙𝛼
𝑖−1∕2
)𝑛
= (𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
,
(64c)
𝜙⋆
𝛼,𝑖= 𝜙𝑛
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
∶= 𝜙𝑛+1
𝛼,𝑖.
(64d)
So far, no discretization of the non-conservative terms has been carried out, as it would not be possible to exactly balance the second 
order terms in the pressure sub-system by simple substitution. Recalling the wave equation for the pressure (17) and restricting 
ourselves to the ideal gas EOS with 𝜋𝛼= 0 in (2), one gets
𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 −Δ𝑡
4Δ𝑥𝑢0𝑝0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
−Δ𝑡2
Δ𝑥2 𝑝0
(( 3
4ℎ𝑛
𝛼,𝑖−1 + 1
4ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖−1 −
(
ℎ𝑛
𝛼,𝑖−1 + ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖+
( 1
4ℎ𝑛
𝛼,𝑖−1 + 3
4ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖+1
)
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
= 𝑝0
𝜙𝑛
𝛼,𝑖
𝛾−1 +
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
..............
−
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
..............
−Δ𝑡
4Δ𝑥𝑝0𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
−Δ𝑡
2Δ𝑥𝑢0
(
(𝜌𝛼𝜙𝛼)𝑛+1
𝑖+1 ℎ𝑛
𝛼,𝑖+1 −(𝜌𝛼𝜙𝛼)𝑛+1
𝑖−1 ℎ𝑛
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑝0𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
−Δ𝑡2
Δ𝑥2 𝑝0
((3
4 ℎ𝑛
𝛼,𝑖−1 + 1
4ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖−1 −
(
ℎ𝑛
𝛼,𝑖−1 + ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖+
( 1
4 ℎ𝑛
𝛼,𝑖−1 + 3
4 ℎ𝑛
𝛼,𝑖+1
)
𝜙𝑛+1
𝛼,𝑖+1
)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
.
(65)
As previously mentioned, the term ∇⋅(ℎ𝑛
𝛼𝑝𝑛
𝐼∇𝜙𝑛+1
𝛼
) on the right-hand-side of the above equation has been discretized in the same 
manner as the term ∇⋅(ℎ𝑛
𝛼∇(𝜙𝛼𝑝𝛼)𝑛+1) on the left-hand-side, thus resorting to the compact ﬁnite diﬀerence operator (31). This choice 
allows the two underlined terms in the above equation to cancel out. Also the other underlined terms, dotted and dashed accordingly, 
vanish. From the right-hand-side of Eq. (65), the term
−Δ𝑡
2Δ𝑥𝑢0
(
(𝜌𝛼𝜙𝛼)𝑛+1
𝑖+1 ℎ𝑛
𝛼,𝑖+1 −(𝜌𝛼𝜙𝛼)𝑛+1
𝑖−1 ℎ𝑛
𝛼,𝑖−1
)
,
can be written by inserting the discretization of the enthalpy (18), which is
ℎ𝑛
𝛼∶=
𝜌𝑛
𝛼
𝜌𝑛+1
𝛼
(
𝑒𝑛
𝛼+
𝑝𝑛
𝛼
𝜌𝑛
𝛼
)
,
hence obtaining
−Δ𝑡
2Δ𝑥𝑢0
(
(𝜌𝛼𝑒𝛼)𝑛
𝑖+1𝜙𝑛+1
𝛼,𝑖+1 −(𝜌𝛼𝑒𝛼)𝑛
𝑖−1𝜙𝑛+1
𝛼,𝑖−1
)
.
In Eq. (65), substituting the above operator, we are, thus, left with
𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 = 𝑝0
𝜙𝑛
𝛼,𝑖
𝛾−1 −Δ𝑡
2Δ𝑥𝑢0
(
(𝜌𝛼𝑒𝛼)𝑛
𝑖+1𝜙𝑛+1
𝛼,𝑖+1 −(𝜌𝛼𝑒𝛼)𝑛
𝑖−1𝜙𝑛+1
𝛼,𝑖−1
)
−Δ𝑡
2Δ𝑥𝑢0𝑝0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑝0𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
.
(66)
Journal of Computational Physics 539 (2025) 114227 
12 


B. Battisti and W. Boscheri
Once again, the last two underlined dashed terms cancel out. Now, expressing the internal energy as a function of pressure and density 
by virtue of the ideal gas EOS, one obtains
𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 = 𝑝0
𝜙𝑛
𝛼,𝑖
𝛾−1 −Δ𝑡
2Δ𝑥𝑢0
𝑝0
𝛾−1
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
= 𝑝0
1
𝛾−1
(
𝜙𝑛
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
))
= 𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 ,
(67)
since the right-hand-side exactly corresponds to the fully discrete evolution of the volume fraction given by (64d). Consequently, 𝑝0
is the solution of the pressure sub-system, which remains constant.
Finally, the update of momentum and total energy is performed as given in (32a) and (32b), respectively, with the enthalpy 
deﬁnition (18): 
(𝜌𝛼𝜙𝛼𝑢𝛼)𝑛+1
𝑖
= 𝑞⋆
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑝0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑝0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
= 𝑞⋆
𝛼,𝑖= 𝑢0(𝜌𝛼𝜙𝛼)𝑛+1
𝑖
,
(68a)
(𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1
𝑖
= 𝐸⋆
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
( 𝑝0
𝛾−1 + 𝑝0
)(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑝0𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
=
𝑝0
𝛾−1
(
𝜙𝑛
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
))
+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
=
𝑝0
𝛾−1 𝜙𝑛+1
𝛼,𝑖+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
.
(68b)
The fully discrete scheme presented is, therefore, well-balanced in the sense that it can exactly preserve the moving equilibrium 
originating from the initial condition (63). Numerical evidence supporting this property is provided by the test case RP0 discussed in 
Section 4.2.
Well-balanced discretization property with numerical dissipation. The well-balanced property is respected even in the presence of the 
additional numerical stabilization terms in the volume fraction equation. In this case, the maximum convective eigenvalue according 
to (27) reduces to the constant velocity, i.e. 𝑎𝑛
Ω = 𝑢0, and thus the viscosity coeﬃcient (37) simpliﬁes to
𝜈± = 1
2 Δ𝑥𝑢0.
Consequently, the update of the volume fraction Eq. (36) writes
𝜙𝑛+1
𝑠,𝑖
= 𝜙𝑛
𝑠,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝑠,𝑖+1 −𝜙𝑛+1
𝑠,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛
𝑠,𝑖+1 −2𝜙𝑛
𝑠,𝑖+ 𝜙𝑛
𝑠,𝑖−1
)
.
(69)
The stabilization term in the above equation must be added to the right hand side of the wave equation for the pressure (65), so that 
Eq. (67) now becomes
𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 = 𝑝0
𝜙𝑛
𝛼,𝑖
𝛾−1 −Δ𝑡
2Δ𝑥𝑢0
𝑝0
𝛾−1
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥
𝑝0
𝛾−1 𝑢0
(
𝜙𝑛
𝑠,𝑖+1 −2𝜙𝑛
𝑠,𝑖+ 𝜙𝑛
𝑠,𝑖−1
)
= 𝑝0
1
𝛾−1
(
𝜙𝑛
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛
𝑠,𝑖+1 −2𝜙𝑛
𝑠,𝑖+ 𝜙𝑛
𝑠,𝑖−1
))
= 𝑝0
𝜙𝑛+1
𝛼,𝑖
𝛾−1 ,
(70)
where the numerical dissipation has been rescaled in order to match the correct units by the factor
𝜕(𝜌𝛼𝜙𝛼𝑒𝛼)𝑛
𝑖
𝜕𝜙𝑛
𝛼,𝑖
=
𝜕
𝜕𝜙𝑛
𝛼,𝑖
((𝜙𝛼𝑝𝛼)𝑛
𝑖
𝛾𝛼−1
)
=
𝑝0
𝛾𝛼−1 .
(71)
Finally, the update of the total energy (68b) is also modiﬁed by adding the stabilization term used in (69) properly rescaled by the 
factor (71), hence leading to
(𝜌𝛼𝜙𝛼𝐸𝛼)𝑛+1
𝑖
= 𝐸⋆
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
( 𝑝0
𝛾−1 + 𝑝0
)(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑝0𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥
𝑝0
𝛾−1 𝑢0
(
𝜙𝑛
𝑠,𝑖+1 −2𝜙𝑛
𝑠,𝑖+ 𝜙𝑛
𝑠,𝑖−1
)
=
𝑝0
𝛾−1
(
𝜙𝑛
𝛼,𝑖−Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛+1
𝛼,𝑖+1 −𝜙𝑛+1
𝛼,𝑖−1
)
+ Δ𝑡
2Δ𝑥𝑢0
(
𝜙𝑛
𝑠,𝑖+1 −2𝜙𝑛
𝑠,𝑖+ 𝜙𝑛
𝑠,𝑖−1
))
+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
=
𝑝0
𝛾−1 𝜙𝑛+1
𝛼,𝑖+
𝑢2
0
2 (𝜌𝛼𝜙𝛼)𝑛+1
𝑖
.
(72)
Journal of Computational Physics 539 (2025) 114227 
13 


B. Battisti and W. Boscheri
Fig. 1. Smooth vortex problem at time 𝑡𝑓= 5, on a mesh with characteristic size 𝜚= 1∕16. Numerical distribution of the pressure of the solid phase, 
with a background pressure of 𝑝0 = 1 (left), 𝑝0 = 102 (middle), and 𝑝0 = 104 (right).
4.  Numerical results
This section presents a suite of numerical test cases for the Baer–Nunziato model that aim at validating the novel semi-implicit 
ﬁnite volume scheme in terms of accuracy and robustness. The new scheme is labeled with SIFV-BN. Diﬀerent ﬂow regimes are 
examined, to show the versatility of the method in the presence of diﬀerent Mach numbers ranging in the interval [10−3, 101]. The 
time step is always computed according to the CFL-type stability condition (11) with CFL = 0.9. If the initial ﬂow speed is zero, only 
the ﬁrst time step is given by (6) so that it is limited by the sound speed. If not stated otherwise, the dimensionless form of the 
governing equations is used.
4.1.  Numerical convergence study
For the convergence study, a modiﬁed version of the smooth vortex problem proposed in Hu and Shu[66] is chosen, following 
the setup of Dumbser et al.[67]. All the details related to the deﬁnition of the initial condition for this test problem can be found in 
Dumbser et al.[67], therefore only a qualitative description is provided hereafter. An isentropic vortex is superimposed to a constant 
background state on the computational domain Ω = [−5, 5] × [−5, 5], with periodic boundary conditions. The diﬀerent phases are 
characterized by the ideal gas EOS, with 𝛾𝑓= 1.35, and 𝛾𝑠= 1.4. The exact solution of this problem is the passive convection of the 
vortex with the mean advection velocity 𝐯𝑐= (2, 2), which is not aligned with the mesh directions. At the ﬁnal time, 𝑡𝑓= 5, the vortex 
has done a complete revolution and is back to its initial location. The test is conducted on a series of successively reﬁned computational 
meshes. Moreover, to study the convergence of the scheme at diﬀerent Mach numbers, diﬀerent background pressures, 𝑝0 ∈[1, 104], 
are imposed to deﬁne the initial condition. Fig. 1 shows the vortex at the ﬁnal time of the simulation, at diﬀerent Mach numbers 
on the same mesh, demonstrating that the solution keeps the same accuracy, and no spurious oscillations arise. The new all Mach 
solver ensures that the solution remains independent of the Mach number, preserving its vortical structure with no degradation. The 
errors are measured in 𝐿2 norm with respect to the analytical solution given in Dumbser et al.[67], for density, 𝑥−momentum, and 
volume fraction of the solid phase. The results are reported in Table 1, demonstrating both the accuracy and the asymptotic preserving 
property of the novel SIFV-BN scheme. The second order of accuracy is still conserved for low Mach numbers.
4.2.  Riemann problems
In one space dimension, several Riemann problems, for which the exact solution is known [68–70], are tested. The one-dimensional 
domain is Ω = [0, 1], which is discretized by 𝑁𝑥= 1000 cells, with Dirichlet boundary conditions. The initial discontinuity is located 
at 𝑥𝐷= 0.5. The initial conditions for the left (L) and right (R) states are listed in Table 2, together with the EOS parameters of the 
two phases, the source term coeﬃcients, and the ﬁnal time of each Riemann Problem (RP). Figs. 2–9 depict a comparison between 
the numerical and the reference solution, that is computed using an explicit second order TVD ﬁnite volume scheme on a ﬁne 
mesh composed of 10’000 cells. The ﬁrst test, RP0, is actually the implementation of a moving contact discontinuity, to numerically 
demonstrate the well-balanced property proven in Section 3.7. Fig. 2 shows the exact preservation of the constant pressure and 
velocity for both phases, while the jumps in density and volume fraction are captured. RP1, RP2, RP3, and RP4 are shock tube 
problems with diﬀerent EOS parameters, involving higher diﬀerence in the variable jumps or the initial velocity. In Figs. 3–6, the 
rarefaction waves, and in particular shock waves, are nicely captured, with a slightly increased diﬃculty for RP3 and RP4. The 
numerical scheme exhibits small oscillations for RP5 and RP6 (Figs. 7 and 8), but the agreement of the numerical solution with the 
reference is still very good. Finally, in RP7, the classical Sod shock tube, the source terms are present, both for pressure (which is 
stiﬀ) and velocity. The exact solution, in this case, is obtained using an exact Riemann solver for the compressible Euler equations, 
with diﬀerent polytropic indexes on the left and right side of the contact discontinuity. Also in this case, the numerical scheme is able 
to provide very satisfying results, see Fig. 9. This collection of test cases shows the ability of our method to comply with the high 
Mach regime, although it is not speciﬁcally designed for simulating sharp discontinuities.
Journal of Computational Physics 539 (2025) 114227 
14 


B. Battisti and W. Boscheri
Table 1 
Numerical convergence results for the smooth vortex problem, for diﬀerent background pressure 𝑝0. The errors for density, 𝑥−momentum, 
and volume fraction of the solid phase are given in 𝐿2 norm at the ﬁnal time 𝑡𝑓= 5.
𝑁𝐱
𝜚
𝐿2(𝜌𝑠𝜙𝑠)
(𝜌𝑠𝜙𝑠)
𝐿2(𝜌𝑠𝜙𝑠𝑢𝑠)
(𝜌𝑠𝜙𝑠𝑢𝑠)
𝐿2(𝜙𝑠)
(𝜙𝑠)
𝑝0 = 1
 20
 0.5
 4.2250E−01
 –
 1.0690E+00
 –
 1.6308E−01
 –
 40
 0.25
 1.6787E−01
 1.33
 3.4438E−01
 1.63
 6.3713E−02
 1.36
 80
 0.125
 4.6463E−02
 1.85
 8.7884E−02
 1.97
 1.6970E−02
 1.91
 160
 0.0625
 1.1957E−02
 1.96
 2.5092E−02
 1.81
 4.2119E−03
 2.01
𝑝0 = 101
 20
 0.5
 3.5478E−02
 –
 1.0726E+00
 –
 1.6402E−01
 –
 40
 0.25
 1.3797E−02
 1.36
 3.5107E−01
 1.61
 6.3683E−02
 1.36
 80
 0.125
 3.2493E−03
 2.09
 8.8656E−02
 1.99
 1.6530E−02
 1.95
 160
 0.0625
 7.6859E−04
 2.08
 2.5126E−02
 1.82
 4.0399E−03
 2.03
𝑝0 = 102
 20
 0.5
 4.9215E−03
 –
 1.0725E+00
 –
 1.6412E−01
 –
 40
 0.25
 1.6694E−03
 1.56
 3.5247E−01
 1.61
 6.3743E−02
 1.36
 80
 0.125
 4.2384E−04
 1.98
 8.9128E−02
 1.98
 1.6527E−02
 1.95
 160
 0.0625
 1.0384E−04
 2.03
 2.5228E−02
 1.82
 4.0357E−03
 2.03
𝑝0 = 103
 20
 0.5
 2.8493E−03
 –
 1.0675E+00
 –
 1.6386E−01
 –
 40
 0.25
 1.2504E−03
 1.19
 3.4870E−01
 1.61
 6.3264E−02
 1.37
 80
 0.125
 3.3387E−04
 1.90
 8.7773E−02
 1.99
 1.6292E−02
 1.96
 160
 0.0625
 8.2761E−05
 2.01
 2.4899E−02
 1.82
 3.9670E−03
 2.04
𝑝0 = 104
 20
 0.5
 9.4160E−03
 –
 1.0305E+00
 –
 1.6343E−01
 –
 40
 0.25
 4.2722E−03
 1.14
 3.2406E−01
 1.67
 6.2452E−02
 1.39
 80
 0.125
 9.1133E−04
 2.23
 7.9132E−02
 2.03
 1.5855E−02
 1.98
 160
 0.0625
 1.4772E−04
 2.63
 2.2924E−02
 1.79
 3.8419E−03
 2.05
Table 2 
Initial condition for the left (L) and right (R) states of the Riemann problems. The values for the EOS of the two phases are reported as well 
as the source terms parameters (𝛿,𝜇) and the ﬁnal time.
𝜌𝑓
𝐮𝑓
𝑝𝑓
𝜌𝑠
𝐮𝑠
𝑝𝑠
𝜙𝑠
𝑡𝑓
 RP0
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 0, 𝜇= 0
 L
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 0.8
0.2
 R
 0.5
 1.0
 1.0
 0.5
 1.0
 1.0
 0.4
 RP1 [69]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 0, 𝜇= 0
 L
 0.5
 0.0
 1.0
 1.0
 0.0
 1.0
 0.4
0.1
 R
 1.5
 0.0
 2.0
 2.0
 0.0
 2.0
 0.8
 RP2 [69]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 3.0, 𝜋𝑠= 100
𝛿= 0, 𝜇= 0
 L
 1.5
 0.0
 2.0
 800.0
 0.0
 500.0
 0.4
0.1
 R
 1.0
 0.0
 1.0
 1000.0
 0.0
 600.0
 0.3
 RP3 [69]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 0, 𝜇= 0
 L
 1.0
 0.0
 1.0
 1.0
 0.9
 2.5
 0.9
0.1
 R
 1.2
 1.0
 2.0
 1.0
 0.0
 1.0
 0.2
 RP4 [70]
𝛾𝑓= 1.35, 𝜋𝑓= 0
𝛾𝑠= 3.0, 𝜋𝑠= 3400
𝛿= 0, 𝜇= 0
 L
 2.0
 0.0
 3.0
 1900.0
 0.0
 10.0
 0.2
0.15
 R
 1.0
 0.0
 1.0
 1950.0
 0.0
 1000.0
 0.9
 RP5 [70]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 0, 𝜇= 0
 L
 0.2
 0.0
 0.3
 1.0
 0.0
 1.0
 0.8
0.2
 R
 1.0
 0.0
 1.0
 1.0
 0.0
 1.0
 0.3
 RP6 [68]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 0, 𝜇= 0
 L
 0.5806
 1.5833
 1.375
 0.2068
 1.4166
 0.0416
 0.1
0.1
 R
 0.4890
-0.70138
 0.986
 2.2263
 0.9366
 6.0
 0.2
 RP7 [54]
𝛾𝑓= 1.67, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 103, 𝜇= 102
 L
 1.0
 0.0
 1.0
 1.0
 0.0
 1.0
 0.99
0.2
 R
 0.125
 0.0
 0.1
 0.125
 0.0
 0.1
 0.01
Journal of Computational Physics 539 (2025) 114227 
15 


B. Battisti and W. Boscheri
Fig. 2. Riemann problem RP0 at time 𝑡𝑓= 0.2. Numerical results for density, volume fraction, horizontal velocity and pressure (from top left to 
bottom right) compared against the reference solution.
Explosion problem with and without inter-phase drag relaxation. To better investigate the inﬂuence of the relaxation terms, we solve a 
cylindrical explosion problem on the domain Ω = [−1, 1]2 discretized with 𝑁𝑥× 𝑁𝑦= 1000 × 1000 cells. The initial condition is given 
by the left and the right state of RP2 in Table 2 separated by a circular discontinuity located at radius 𝑅= 0.4. We run the simulation 
until time 𝑡𝑓= 0.15 with and without inter-phase drag relaxation terms, namely setting 𝛿= 105 and 𝛿= 0. The reference solution is 
computed relying on an explicit second order TVD ﬁnite volume scheme on a ﬁne mesh composed of 10’000 cells. The results are 
depicted in Fig. 10, showing an excellent agreement with the reference solution. The eﬀect of the stiﬀ source terms is particularly 
evident in the phase velocities, which relax to the mixture velocity when 𝛿= 105. The cylindrical symmetry of the solution can be 
also qualitatively appreciated looking at the three-dimensional distribution of the ﬂuid phase density. A comparison in terms of 
computational eﬃciency against a fully explicit scheme is reported in Table 5 for the case 𝛿= 105, showing a gaining factor of 43. 
This is obviously due to the presence of the stiﬀ drag relaxation coeﬃcient, which is implicitly solved in our numerical method.
4.3.  Two-dimensional Riemann problems
Increasing the spatial dimension does not change the methodology, since we use Cartesian grids, but it increases the burden of the 
computational cost. The novel numerical scheme is parallelized using MPI, allowing simulations with a very high number of degrees 
of freedom. Here, we run a couple of genuinely two-dimensional Riemann problems proposed in Dumbser and Boscheri[54], which 
are inspired from Zhang and Zheng[71], Kurganov and Tadmor[72]. The initial condition is deﬁned by four piecewise constant states, 
prescribed on each of the four quadrants of the squared domain Ω = [−0.5, 0.5] × [−0.5, 0.5]. Reﬂective wall boundary conditions are 
applied everywhere. The initial conditions and the parameters for the two test cases are summarized in Table 3. The ﬁrst Riemann 
Journal of Computational Physics 539 (2025) 114227 
16 


B. Battisti and W. Boscheri
Fig. 3. Riemann problem RP1 at time 𝑡𝑓= 0.1. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
17 


B. Battisti and W. Boscheri
Fig. 4. Riemann problem RP2 at time 𝑡𝑓= 0.1. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
18 


B. Battisti and W. Boscheri
Fig. 5. Riemann problem RP3 at time 𝑡𝑓= 0.1. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
19 


B. Battisti and W. Boscheri
Fig. 6. Riemann problem RP4 at time 𝑡𝑓= 0.15. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
20 


B. Battisti and W. Boscheri
Fig. 7. Riemann problem RP5 at time 𝑡𝑓= 0.2. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
21 


B. Battisti and W. Boscheri
Fig. 8. Riemann problem RP6 at time 𝑡𝑓= 0.1. Numerical results for density (top), horizontal velocity (middle) and pressure (bottom) compared 
against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
22 


B. Battisti and W. Boscheri
Fig. 9. Riemann problem RP7 at time 𝑡𝑓= 0.2. Numerical results for mixture density, horizontal velocity, pressure and volume fraction (from top 
left to bottom right) compared against the reference solution.
problem involves stiﬀ relaxation sources, while the second one only considers the homogeneous system. The reference solution is 
computed with a high order ADER scheme on a very ﬁne mesh composed of 2’277’668 triangles, with characteristic mesh spacing 
ℎ= 1∕1000, see Dumbser et al.[67], Toro et al.[73].
The numerical results obtained with the SIFV-BN scheme are displayed in the left column of Figs. 11 and 12, and compared to the 
reference solution on the right column, for the density of both phases and the volume fraction. The results are strikingly in accordance 
with the reference from the qualitative viewpoint.
4.4. 2D Taylor–Green vortex at low Mach number
The two-dimensional version of the original Taylor–Green vortex [74] is used here to perform a numerical convergence study in 
the low Mach regime. The analytical solution for a single-phase ﬂuid in the incompressible limit reads
𝜌= 𝜌0,
𝑢= sin(𝑥) cos(𝑦),
𝑣= −cos(𝑥) sin(𝑦),
𝑝= 𝑝0 + 1
4 (cos(2𝑥) + cos(2𝑦)).
(73)
Journal of Computational Physics 539 (2025) 114227 
23 


B. Battisti and W. Boscheri
Fig. 10. Explosion problem at time 𝑡𝑓= 0.15 with (left column) and without (right column) inter-phase drag relaxation 𝛿= 105. Numerical results 
for the ﬂuid density (top row) and for the horizontal velocity of both phases with comparison against the reference solution (bottom row)..
The computational domain is Ω = [0, 2𝜋]2 with periodic boundaries, and it is discretized with a mesh composed of 𝑁𝑥× 𝑁𝑦= 50 × 50
cells. This ﬂow is analytically divergence-free, and the objective is to verify that the numerical scheme achieves a discrete divergence-
free condition as the Mach number goes to zero with second order of convergence. For this purpose, we set 𝜌0 = 1 and we assign 
the initial condition (73) to each phase. We adopt the ideal gas EOS with 𝛾𝑓= 𝛾𝑠= 1.4. This test case is run on a sequence of 
successively diminishing Mach number regimes, by increasing the background pressure 𝑝0, until a ﬁnal time of 𝑡𝑓= 0.2, using a 
constant time step of Δ𝑡= 2.5 ⋅10−3. For retrieving a single-phase behavior, the volume fraction is set to 𝜙𝑠= 0.5, while we impose 
𝜙𝑠= 0.5 + 0.1 (cos(2𝑥) + cos(2𝑦)) to allow interactions between the two phases. The error in the velocity divergence for the mixture 
velocity ∇⋅𝐮= 0 is computed in the 𝐿1, 𝐿2, and 𝐿∞ norms using the discrete cell-centered divergence operator
∇ℎ⋅𝐮= 𝑢𝑖+1 −𝑢𝑖−1
2Δ𝑥
+ 𝑣𝑖+1 −𝑣𝑖−1
2Δ𝑦
.
The convergence rates and the errors are reported in Table 4, showing that the scheme achieves second order Mach convergence for 
Mach numbers up to 𝑀= 10−3, hence imposing a background pressure up to 𝑝0 = 106. Fig. 13 shows the distribution of the divergence 
of the mixture velocity ﬁeld for 𝑀= 10−3, conﬁrming that the scheme approaches the divergence-free condition with second order 
of convergence, up to errors of order 10−7. One-dimensional cuts of the numerical solution are also presented and compared to the 
reference solution. It is worth noting that for even lower Mach numbers the scheme does not converge, as recently observed in Orlando 
et al.[75]. This is due to the cell-centered spatial discretization adopted in our numerical scheme, that is not a divergence preserving 
operator. Future research will concern the development of div-curl preserving schemes, such as those presented in Boscheri et al.[59], 
in the context of multi-phase ﬂows.
Journal of Computational Physics 539 (2025) 114227 
24 


B. Battisti and W. Boscheri
Fig. 11. 2D Riemann problem C1 at time 𝑡𝑓= 0.15. Numerical results for solid density, ﬂuid density and volume fraction (from top to bottom). Left: 
second order SIFV-BN scheme. Right: reference solution computed with a third order ADER-WENO ﬁnite volume scheme [67,73].
Journal of Computational Physics 539 (2025) 114227 
25 


B. Battisti and W. Boscheri
Fig. 12. 2D Riemann problem C2 at time 𝑡𝑓= 0.15. Numerical results for solid density, ﬂuid density and volume fraction (from top to bottom). Left: 
second order SIFV-BN scheme. Right: reference solution computed with a third order ADER-WENO ﬁnite volume scheme [67,73].
Journal of Computational Physics 539 (2025) 114227 
26 


B. Battisti and W. Boscheri
Table 3 
Initial condition for the top right (TR), top left (TL), bottom right (BR), and bottom left (BL) quadrant 
states of the two-dimensional Riemann problems. The values for the EOS of the two phases are given 
as well as the source terms parameters and the ﬁnal time.
𝜌𝑓
𝐮𝑓
𝑝𝑓
𝜌𝑠
𝐮𝑠
𝑝𝑠
𝜙𝑠
𝑡𝑓
 RP2D1 [54]
𝛾𝑓= 1.67, 𝜋𝑓= 0
𝛾𝑠= 1.4, 𝜋𝑠= 0
𝛿= 103, 𝜇= 102
 TR
 1.5
 (0.0,0.0)
 2.0
 2.0
 (0.0,0.0)
 2.0
 0.8
0.15
 TL
 0.5
 (0.0,0.0)
 1.0
 1.0
 (0.0,0.0)
 1.0
 0.4
 BL
 1.5
 (0.0,0.0)
 2.0
 2.0
 (0.0,0.0)
 2.0
 0.8
 BR
 0.5
 (0.0,0.0)
 1.0
 1.0
 (0.0,0.0)
 1.0
 0.4
 RP2D2 [54]
𝛾𝑓= 1.4, 𝜋𝑓= 0
𝛾𝑠= 3.0, 𝜋𝑠= 100
𝛿= 0, 𝜇= 0
 TR
 1.0
 (0.0,0.0)
 1.0
 1000.0
 (0.0,0.0)
 600.0
 0.3
0.15
 TL
 1.5
 (0.0,0.0)
 2.0
 800.0
 (0.0,0.0)
 500.0
 0.4
 BL
 1.0
 (0.0,0.0)
 1.0
 1000.0
 (0.0,0.0)
 600.0
 0.3
 BR
 1.5
 (0.0,0.0)
 2.0
 800.0
 (0.0,0.0)
 500.0
 0.4
Table 4 
Numerical convergence results for the Taylor–Green vortex problem, for diﬀerent Mach numbers 𝑀. The errors are given in 
𝐿1, 𝐿2, and 𝐿∞ norms, at the ﬁnal time 𝑡𝑓= 0.2, for the single-phase and two-phase case.
𝑀
𝐿1
(∇⋅𝐮= 0)
𝐿2
(∇⋅𝐮= 0)
𝐿∞
(∇⋅𝐮= 0)
 Single-phase
10−1
 2.071400E−2
 –
 4.188600E−3
 –
 1.533700E−3
 –
10−2
 4.964400E−4
 1.62
 1.026100E−4
 1.61
 3.603900E−5
 1.63
10−3
 4.143400E−6
 2.08
 8.689600E−7
 2.07
 3.011100E−7
 2.08
 Two-phase
10−1
 2.9392E−02
 –
 5.5657E−03
 –
 1.9747E−03
 –
10−2
 7.9792E−04
 1.57
 1.4673E−04
 1.58
 5.2690E−05
 1.57
10−3
 4.1945E−06
 2.28
 8.7848E−07
 2.22
 3.0879E−07
 2.23
4.5.  Shock-bubble interaction
The shock-bubble interaction (SBI) problem [76] considers a planar shock wave in a liquid, interacting with a gas bubble, which 
leads to shock refraction and reﬂection. Such a problem ﬁnds many practical applications [77]. This test case allows us to show 
the ability of the SIFV-BN scheme to work also with multiple and interacting shocks. The computational domain is Ω = [−0.5, 3] ×
[−0.75, 0.75], with a gas bubble of radius 0.25 located at [0.5, 0.0], and governed by the ideal gas EOS, with 𝛾𝑓= 1.4. The liquid phase 
is moving at high speed from left to right in the longitudinal direction, and is characterized by the stiﬀened gas EOS, with 𝛾𝑠= 3.0
and 𝜋𝑠= 100. The domain is characterized by inﬂow conditions on the left boundary, slip-wall conditions on the other boundaries, 
and is discretized with 𝑁𝑥× 𝑁𝑦= 2048 × 1024 cells. In Fig. 14, several snapshots of the numerical solution are shown until the ﬁnal 
time 𝑡𝑓= 0.025, both for the volume fraction and the liquid density. The shock waves are accurately captured, accounting for the 
several interactions with the wall boundaries on the top and bottom, and with one another. The results are qualitatively in very good 
agreement with those available in the literature [78]. This test case shows the ability of the numerical scheme to deal with highly 
complex multidimensional compressible ﬂows.
4.6.  Water-air mixture
To show the eﬀectiveness of the numerical scheme also in the low Mach regime, a test involving water and air is conducted. The 
test is a Riemann problem inspired from Saurel et al.[19], and subsequently modiﬁed in Re and Abgrall[46]. The computational 
domain Ω = [−0.65, 0.65] is discretized with 𝑁𝑥= 1000 cells, with Dirichlet boundary conditions. The air is characterized by a density 
of 𝜌𝑓= 1.2, and follows the ideal gas EOS, with 𝛾𝑓= 1.4. The water is characterized by density 𝜌𝑠= 1050, and follows the stiﬀened gas 
EOS, with 𝛾𝑠= 4.4 and 𝜋𝑠= 6.8 × 108. The two ﬂuids are initially at rest, with uniform density, while pressure and volume fraction are 
discontinuous, with same values for both phases: 𝑝= 106 and 𝜙= 0.3 in the center of the domain, and 𝑝= 105 and 𝜙= 0.7 elsewhere. 
The air ﬂow lies in the compressible regime and may show shock waves, while the water regime is incompressible. In this case, the 
deﬁnition of the interface velocity and pressure slightly modiﬁes the water behavior. The version in Re and Abgrall[46] considers 
the deﬁnition (3b). The results are shown in Fig. 15 at the ﬁnal time 𝑡𝑓= 350 μs, showing an excellent agreement with the reference 
solution that has been computed resorting on the same TVD scheme used for the Riemann problems presented in Section 4.2. The 
Mach number of the liquid phase is of the order 10−3, while the air ﬂow reaches a Mach number of 1, thus involving a jump of 
four orders of magnitude within the same simulation. The SIFV-BN all Mach solver has proven to handle also this type of situations. 
The computational eﬃciency is compared against a fully explicit ﬁnite volume scheme, and the results are shown in Table 5. The 
SIFV-BN scheme is 4 times faster than the explicit method, which is particularly relevant for low Mach number ﬂows like for this test 
case.
Journal of Computational Physics 539 (2025) 114227 
27 


B. Battisti and W. Boscheri
Fig. 13. Numerical solution of the two-dimensional Taylor–Green vortex for 𝑀= 10−3 at the ﬁnal time 𝑡𝑓= 0.2, using the SIFV-BN scheme. Top 
left: Mach number contours. Top right: mixture velocity divergence contours. Bottom left: 1D cuts along the 𝑥− and the 𝑦−axis, and comparison 
with the reference solution for the ﬂuid mixture velocity components 𝑢 and 𝑣. Bottom right: 1D cut along the 𝑥−axis for the ﬂuid mixture pressure 
𝑝, and comparison to the reference solution.
Table 5 
Computational time of the SIFV-BN scheme compared against the explicit Fi-
nite Volume scheme for the test cases presented in Sections 4.6 and 4.2. Times 
are measured in seconds, and the simulations have been run on a 12th Gen In-
tel(R) Core(TM) i7-1255U 1.70 GHz processor on 4 CPU. The ratio is computed 
as 𝛽= Explicit FV∕SIFV-BN.
 Test case
 SIFV-BN
 Explicit FV
𝛽
 Explosion problem
 131.46
 529.22
 4.03
 Water-air mixture
 14194.19
 614122.20
 43.27
Furthermore, this problem can be extended to three space dimensions, on a cubic computational domain Ω = [−0.65, 0.65]3. Two 
diﬀerent simulations are run until the ﬁnal time 𝑡𝑓= 300 μs, to show mesh convergence in the incompressible limit of the liquid phase: 
the ﬁrst one involves 7’077’888 cells, while the second one is concerned with 16’777’216 cubic control volumes. The results are plot 
in Fig. 16, where we can see that no spurious oscillations arise, and that the symmetry of the numerical solution is well preserved.
Journal of Computational Physics 539 (2025) 114227 
28 


B. Battisti and W. Boscheri
Fig. 14. Shock-bubble interaction problem at times 𝑡= 0.005, 𝑡= 0.010, 𝑡= 0.015, 𝑡= 0.020 and 𝑡= 0.025 (from top to bottom). Numerical results 
for volume fraction (left) and solid phase density (right) depicted with 21 equidistant contour lines.
Journal of Computational Physics 539 (2025) 114227 
29 


B. Battisti and W. Boscheri
Fig. 15. Water-air mixture problem at time 𝑡𝑓= 350 μs. Numerical results for density (top), horizontal velocity (middle) and Mach number (bottom) 
compared against the reference solution. Left: solid phase. Right: ﬂuid phase.
Journal of Computational Physics 539 (2025) 114227 
30 


B. Battisti and W. Boscheri
Fig. 16. Water-air mixture problem in 3D at time 𝑡= 300 μs. Top: numerical distribution of the liquid velocity magnitude |𝐮𝑠| (left), and pressure 𝑝𝑠
(right), along the planes 𝑥= 0, 𝑦= 0, 𝑧= 0. Bottom: air (left) and water (right) pressure solution obtained with two diﬀerent computational meshes.
5.  Conclusions
In this paper, a semi-implicit numerical scheme is presented for the solution of the Baer–Nunziato model for compressible two-
phase ﬂows, to deal with all Mach number regimes. The pressure ﬂuxes and the potentially stiﬀ source terms are discretized im-
plicitly with central ﬁnite diﬀerences, while shock capturing properties are guaranteed by an explicit ﬁnite volume method for the 
remaining convective terms. In this way, the new scheme avoids an excessive restriction on the time step, that is only based on 
the mean ﬂow velocity. The numerical approximation of the non-conservative products relies on physical considerations, which 
ultimately allows the scheme to exactly preserve moving equilibrium solutions of the governing equations. The scheme is proven 
to be second order in time and in space, with a judicious discretization of the numerical viscosity. To identify the diﬀerent Mach 
number regimes, the non-dimensional equations are retrieved using a Mach number common to both phases, hence dealing with 
the mixture ﬂow. The asymptotic preserving property of the scheme in the low Mach number limit model for the mixture is also 
demonstrated. The novel solver shows accuracy and robustness both in the compressible regime and approaching the incompressible
limit.
Future works will consider the deﬁnition of diﬀerent Mach numbers for each phase, to design asymptotic preserving schemes. 
More physics will also be tackled by including the viscosity terms and the gravity terms, yielding to the identiﬁcation of diﬀerent 
Reynolds and Froude numbers. This will allow us to deal with more complex models, such as the simulation of magma ﬂows in 
a volcanic conduit. Finally, since primitive variables are physically measurable in real world scenarios, we plan to discretize the 
Baer–Nunziato model written in fully non-conservative form, while ensuring conservation properties.
Journal of Computational Physics 539 (2025) 114227 
31 


B. Battisti and W. Boscheri
CRediT authorship contribution statement
Beatrice Battisti: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investiga-
tion, Formal analysis, Data curation; Walter Boscheri: Writing – review & editing, Writing – original draft, Visualization, Validation, 
Supervision, Software, Project administration, Methodology, Investigation, Funding acquisition, Formal analysis, Conceptualization.
Data availability
Data will be made available on request.
Declaration of competing interest
The authors declare the following ﬁnancial interests/personal relationships which may be considered as potential competing 
interests:
Walter Boscheri reports ﬁnancial support was provided by French National Research Agency and by Italian Ministry of University 
and Research. If there are other authors, they declare that they have no known competing ﬁnancial interests or personal relationships 
that could have appeared to inﬂuence the work reported in this paper. 
Acknowledgments
WB received ﬁnancial support by the Italian Ministry of University and Research (MUR) with the PRIN Project 2022 No. 
2022N9BM3N and by the “Agence Nationale de la Recherche” (ANR) with project No. ANR-23-EXMA-0004. 
References
[1] M.R. Baer, J.W. Nunziato, A two-phase mixture theory for the deﬂagration-to-detonation transition (ddt) in reactive granular materials, Int. J. Multiph. Flow 12 
(6) (1986) 861–889. 
[2] M. Hillairet, On baer-nunziato multiphase ﬂow models, ESAIM: ProcS 66 (2019) 61–83. 
[3] M. Hantke, S. Müller, L. Grabowsky, News on Baer–Nunziato-type model at pressure equilibrium, Contin. Mech. Thermodyn. 33 (3) (2021) 767–788. 
[4] R. Saurel, F. Petitpas, R.A. Berry, Simple and eﬃcient relaxation methods for interfaces separating compressible ﬂuids, cavitating ﬂows and shocks in multiphase 
mixtures, J. Comput. Phys. 228 (5) (2009) 1678–1712. 
[5] M. Pelanti, K.M. Shyue, A mixture-energy-consistent six-equation two-phase numerical model for ﬂuids with interfaces, cavitation and evaporation waves, J. 
Comput. Phys. 259 (2014) 331–357. 
[6] M. Lukáˇcová-Medvid’ová, I. Peshkov, A. Thomann, An implicit-explicit solver for a two-ﬂuid single-temperature model, J. Comput. Phys. 498 (2024) 112696. 
[7] M.G. Rodio, R. Abgrall, An innovative phase transition modeling for reproducing cavitation through a ﬁve-equation model and theoretical generalization to six 
and seven-equation models, Int. J. Heat Mass Transf. 89 (2015) 1386–1401. 
[8] R. Saurel, F. Petitpas, R. Abgrall, Modelling phase transition in metastable liquids: application to cavitating and ﬂashing ﬂows, J. Fluid Mech. 607 (2008) 
313–350. 
[9] A.K. Kapila, R. Menikoﬀ, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deﬂagration-to-detonation transition in granular materials: reduced equations, 
Phys. Fluids 13 (10) (2001) 3002–3024. 
[10] A. Murrone, H. Guillard, A ﬁve equation reduced model for compressible two phase ﬂow problems, J. Comput. Phys. 202 (2) (2005) 664–698. 
[11] B. Braconnier, B. Nkonga, An all-speed relaxation scheme for interface ﬂows with surface tension, J. Comput. Phys. 228 (16) (2009) 5722–5739. 
[12] A. Morin, T. Fl ˙atten, A two-ﬂuid four-equation model with instantaneous thermodynamical equilibrium, ESAIM: M2AN 50 (4) (2016) 1167–1192. 
[13] A. Zein, M. Hantke, G. Warnecke, Modeling phase transition for compressible two-phase ﬂows applied to metastable liquids, J. Comput. Phys. 229 (8) (2010) 
2964–2998. 
[14] H. Bruce Stewart, B.B. Wendroﬀ, Two-phase ﬂow: models and methods, J. Comput. Phys. 56 (3) (1984) 363–409. 
[15] R. Saurel, R. Abgrall, A multiphase godunov method for compressible multiﬂuid and multiphase ﬂows, J. Comput. Phys. 150 (2) (1999) 425–467. 
[16] R. Abgrall, S. Karni, Computations of compressible multiﬂuids, J. Comput. Phys. 169 (2) (2001) 594–623. 
[17] S.A. Tokareva, E.F. Toro, HLLC-type Riemann solver for the Baer–Nunziato equations of compressible two-phase ﬂow, J. Comput. Phys. 229 (10) (2010) 
3573–3604. 
[18] F. Fraysse, C. Redondo, G. Rubio, E. Valero, Upwind methods for the Baer–Nunziato equations and higher-order reconstruction using artiﬁcial viscosity, J. 
Comput. Phys. 326 (2016) 805–827. 
[19] R. Saurel, A. Chinnayya, Q. Carmouze, Modelling compressible dense and dilute two-phase ﬂows, Phys. Fluids 29 (6) (2017) 063301. 
[20] M. Alekseev, E. Savenkov, Runge–Kutta discontinuous Galerkin method for Baer–Nunziato model with ‘simple WENO’ limiting of conservative variables, Russian 
J. Numer. Anal. Math. Model. 36 (2) (2021) 57–74. 
[21] L. Pan, G. Zhao, B. Tian, S. Wang, A gas kinetic scheme for the Baer–Nunziato two-phase ﬂow model, J. Comput. Phys. 231 (22) (2012) 7518–7536. 
[22] S. Dellacherie, Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number, J. Comput. Phys. 229 (2010) 978–1016. 
[23] C.-D. Munz, S. Roller, R. Klein, K.J. Geratz, The extension of incompressible ﬂow solvers to the weakly compressible regime, Comput. Fluids 32 (2) (2003) 
173–196. 
[24] S.V. Patankar, Numerical Heat Transfer and Fluid Flow, New York: McGraw-Hill, 1980. 
[25] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, J. Comput. Phys. 72 (1987) 277–298. 
[26] C. Chalons, M. Girardin, S. Kokh, Large time step and asymptotic preserving numerical schemes for the gas dynamics equations with source terms, SIAM J. Sci. 
Comput. 35 (2013) 2874–2902. 
[27] M. Boger, F. Jaegle, R. Klein, C.D. Munz, Coupling of compressible and incompressible ﬂow regions using the multiple pressure variables approach, Math. 
Methods Appl. Sci. 38 (2015) 458–477. 
[28] G. Dimarco, R. Loubère, V.M. Dansac, M.H. Vignal, Second-order implicit-explicit total variation diminishing schemes for the Euler system in the low Mach 
regime, J. Comput. Phys. 372 (2018) 178–201. 
[29] F. Cordier, P. Degond, A. Kumbaro, An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations, J. Comput. Phys. 231 (2012) 
5685–5704. 
[30] U.M. Ascher, S.J. Ruuth, R.J. Spiteri, Implicit-explicit Runge–Kutta methods for time-dependent partial diﬀerential equations, Appl. Numer. Math. 25 (1982) 
151–167. 
[31] S. Boscarino, L. Pareschi, On the asymptotic properties of IMEX Runge–Kutta schemes for hyperbolic balance laws, J. Comput. Appl. Math. 316 (2017) 60–73. 
Journal of Computational Physics 539 (2025) 114227 
32 


B. Battisti and W. Boscheri
[32] S. Boscarino, G. Russo, On a class of uniformly accurate IMEX Runge–Kutta schemes and applications to hyperbolic systems with relaxation, SIAM J. Sci. Comput. 
31 (2009) 1926–1945. 
[33] L. Pareschi, G. Russo, Implicit-explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation, J. Sci. Comput. 25 (2005) 129–155. 
[34] S. Boscarino, L. Pareschi, G. Russo, A uniﬁed IMEX Runge–Kutta approach for hyperbolic systems with multiscale relaxation, SIAM J. Numer. Anal. 55 (4) (2017) 
2085–2109. 
[35] G. Orlando, P.F. Barbante, L. Bonaventura, An eﬃcient IMEX-DG solver for the compressible Navier-Stokes equations for non-ideal gases, J. Comput. Phys. 471 
(2022) 111653. 
[36] J. Haack, S. Jin, J.G. Liu, An all-speed asymptotic-preserving method for the isentropic Euler and Navier–Stokes equations, Commun. Comput. Phys. 12 (4) 
(2012) 955–980. 
[37] S. Avgerinos, F. Bernard, A. Iollo, G. Russo, Linearly implicit all Mach number shock capturing schemes for the Euler equations, J. Comput. Phys. 393 (2019) 
278–312. 
[38] R. Klein, Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: one-dimensional ﬂow, J. Comput. Phys. 121 (1995) 
213–237. 
[39] V. Casulli, Semi-implicit ﬁnite diﬀerence methods for the two–dimensional shallow water equations, J. Comput. Phys. 86 (1990) 56–74. 
[40] J.H. Park, C.D. Munz, Multiple pressure variables methods for ﬂuid ﬂow at all Mach numbers, Int. J. Numer. Meth. Fluids 49 (2005) 905–931. 
[41] S. Boscarino, F. Filbet, G. Russo, High order semi-implicit schemes for time dependent partial diﬀerential equations, J. Sci. Comput. 68 (3) (2016) 975–1001. 
[42] M. Dumbser, V. Casulli, A conservative, weakly nonlinear semi-implicit ﬁnite volume scheme for the compressible Navier–Stokes equations with general equation 
of state, Appl. Math. Comput. 272 (2016) 479–497. 
[43] W. Boscheri, L. Pareschi, High order pressure-based semi-implicit IMEX schemes for the 3D Navier–Stokes equations at all Mach numbers, J. Comput. Phys. 434 
(2021) 110206. 
[44] W. Boscheri, M. Tavelli, High order semi-implicit schemes for viscous compressible ﬂows in 3D, Appl. Math. Comput. 434 (2022) 127457. 
[45] S. Boscarino, J. Qiu, G. Russo, T. Xiong, High order semi-implicit WENO schemes for all-Mach full Euler system of gas dynamics, SIAM J. Sci. Comput. 44 (2022) 
B368–B394. 
[46] B. Re, R. Abgrall, A pressure-based method for weakly compressible two-phase ﬂows under a Baer–Nunziato type model with generic equations of state and 
pressure and velocity disequilibrium, Int. J. Numer. Methods Fluids 94 (8) (2022) 1183–1232. 
[47] T.Y. Hou, P.G. LeFloch, Why nonconservative schemes converge to wrong solutions: error analysis, Math. Comput. 62 (1994) 497–530. 
[48] R. Rana, N.K. Singh, A pressure-based uniﬁed solver for low Mach compressible two-phase ﬂows, Int. J. Heat Fluid Flow 110 (2024) 109657. 
[49] C. Varsakelis, M.V. Papalexandris, A numerical method for two-phase ﬂows of dense granular mixtures, J. Comput. Phys. 257 (2014) 737–756. 
[50] C. Varsakelis, M.V. Papalexandris, Time-accurate calculation of two-phase granular ﬂows exhibiting compaction, dilatancy and nonlinear rheology, J. Comput. 
Phys. 372 (2018) 799–822. 
[51] S. Malusà, A. Alaia, A well-balanced all-Mach scheme for compressible two-phase ﬂow, Comput. Phys. Commun. 299 (2024) 109131. 
[52] G. Narbona-Reina, D. Bresch, A. Burgisser, M. Collombet, Two-phase magma ﬂow with phase exchange: part I. Physical modeling of a volcanic conduit, Stud. 
Appl. Math. 153 (3) (2024) e12741. 
[53] V. Perrier, E. Gutiérrez, Derivation and closure of Baer and Nunziato type multiphase models by averaging a simple stochastic model, Multiscale Model. Simul. 
19 (1) (2021) 401–439. 
[54] M. Dumbser, W. Boscheri, High-order unstructured Lagrangian one-step WENO ﬁnite volume schemes for non-conservative hyperbolic systems: applications to 
compressible multi-phase ﬂows, Comput. Fluids 86 (2013) 405–432. 
[55] F. Daude, P. Galon, On the computation of the Baer–Nunziato model using ALE formulation with HLL- and HLLC-type solvers towards ﬂuid–structure interactions, 
J. Comput. Phys. 304 (2016) 189–230. 
[56] N. Andrianov, G. Warnecke, The Riemann problem for the Baer–Nunziato two-phase ﬂow model, J. Comput. Phys. 195 (2) (2004) 434–464. 
[57] R. Klein, N. Botta, T. Schneider, C.-D. Munz, S. Roller, A. Meister, L. Hoﬀmann, T. Sonar, Asymptotic adaptive methods for multi-scale problems in ﬂuid 
mechanics, J. Eng. Math. 39 (2001) 261–343. 
[58] L. Brugnano, V. Casulli, Iterative solution of piecewise linear systems, SIAM J. Sci. Comput. 30 (1) (2008) 463–472. 
[59] W. Boscheri, M. Dumbser, M. Ioriatti, I. Peshkov, E. Romenski, A structure-preserving staggered semi-implicit ﬁnite volume scheme for continuum mechanics, 
J. Comput. Phys. 424 (2021) 109866. 
[60] W. Boscheri, S. Busto, M. Dumbser, An all Mach number semi-implicit hybrid ﬁnite volume/virtual element method for compressible viscous ﬂows on Voronoi 
meshes, Comput. Methods Appl. Mech. Eng. 433 (2025) 117502. 
[61] C. Parés, Numerical methods for nonconservative hyperbolic systems: a theoretical framework, SIAM J. Numer. Anal. 44 (1) (2006) 300–321. 
[62] Y. Saad, M. Schultz, GMRES: a generalized minimal residual algorithm for solving nonsymmetric linear systems, SIAM J. Sci. Stat. Comput. 7 (1986) 856–869. 
[63] G. Strang, On the construction and comparison of diﬀerence schemes, SIAM J. Numer. Anal. 5 (3) (1968) 506–517. 
[64] C. Varsakelis, M.V. Papalexandris, Low-Mach-number asymptotics for two-phase ﬂows of granular materials, J. Fluid Mech. 669 (2011) 472–497. 
[65] G. Billet, R. Abgrall, An adaptive shock-capturing algorithm for solving unsteady reactive ﬂows, Comput. Fluids 32 (10) (2003) 1473–1495. 
[66] C. Hu, C. Shu, Weighted essentially non-oscillatory schemes on triangular meshes, J. Comput. Phys. 150 (1999) 97–127. 
[67] M. Dumbser, A. Hidalgo, M. Castro, C. Parés, E.F. Toro, FORCE schemes on unstructured meshes II: non-conservative hyperbolic systems, Comput. Methods 
Appl. Mech. Eng. 199 (9) (2010) 625–647. 
[68] N. Andrianov, G. Warnecke, The Riemann problem for the Baer–Nunziato two-phase ﬂow model, J. Comput. Phys. 195 (2) (2004) 434–464. 
[69] V. Deledicque, M.V. Papalexandris, An exact Riemann solver for compressible two-phase ﬂow models containing non-conservative products, J. Comput. Phys. 
222 (1) (2007) 217–245. 
[70] D.W. Schwendeman, C.W. Wahle, A.K. Kapila, The Riemann problem and a high-resolution Godunov method for a model of compressible two-phase ﬂow, J. 
Comput. Phys. 212 (2) (2006) 490–526. 
[71] T. Zhang, Y.X. Zheng, Conjecture on the structure of solutions of the Riemann problem for two-dimensional gas dynamics systems, SIAM J. Math. Anal. 21 (3) 
(1990) 593–630. 
[72] A. Kurganov, E. Tadmor, Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers, Numer. Methods Part. Diﬀer. Equ. 
18 (5) (2002) 584–608. 
[73] E.F. Toro, A. Hidalgo, M. Dumbser, FORCE schemes on unstructured meshes I: conservative hyperbolic systems, J. Comput. Phys. 228 (9) (2009) 3368–3389. 
[74] G.I. Taylor, A.E. Green, Mechanism of the production of small eddies from large ones, Proc. R. Soc. Lond. Ser. A - Math. Phys. Sci. 158 (895) (1937) 499–521. 
[75] G. Orlando, S. Boscarino, G. Russo, A quantitative comparison of high-order asymptotic-preserving and asymptotically-accurate IMEX methods for the Euler 
equations with non-ideal gases, Comput. Methods Appl. Mech. Eng. 442 (2025) 118037. 
[76] K.M. Shyue, A ﬂuid-mixture type algorithm for compressible multicomponent ﬂow with van der Waals equation of state, J. Comput. Phys. 156 (1) (1999)
43–88. 
[77] C.E. Brennen, Cavitation and Bubble Dynamics, Cambridge University Press, 2014. 
[78] M. Dumbser, A. Hidalgo, O. Zanotti, High order space–time adaptive ADER-WENO ﬁnite volume schemes for non-conservative hyperbolic systems, Comput. 
Methods Appl. Mech. Eng. 268 (2014) 359–387. 
Journal of Computational Physics 539 (2025) 114227 
33 
