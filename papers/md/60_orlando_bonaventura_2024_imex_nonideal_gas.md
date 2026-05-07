Highlights
Asymptotic-preserving IMEX schemes for the Euler equations of non-ideal gases
Giuseppe Orlando,Luca Bonaventura
• Analysis of the asymptotic-preserving (AP) property of a general class of IMEX schemes for a general EOS
• Non-trivial extension to a general EOS of the asymptotic analysis of two length scale models
• Development of a high-order numerical method in combination with a DG space discretization effective for a
wide range of Mach numbers
• Development of an AP scheme without operator splitting, flux splitting or relaxation techniques
• Non-trivial extension of classical benchmarks for low Mach flows to the SG-EOS and to the general cubic EOS
arXiv:2402.09252v4  [math.NA]  22 Oct 2025


Asymptotic-preserving IMEX schemes for the Euler equations of
non-ideal gases
Giuseppe Orlandoa,∗, Luca Bonaventurab
aCMAP, CNRS, École polytechnique, Institut Polytechnique de Paris, Route de Saclay, Palaiseau, 91120, France
bDipartimento di Matematica, Politecnico di Milano, Piazza Leonardo da Vinci 32, Milano, 20133, Italy
A R T I C L E I N F O
Keywords:
Asymptotic-preserving
Euler equations
IMEX
Discontinuous Galerkin
Non-ideal gas
A B S T R A C T
We analyze schemes based on a general Implicit-Explicit (IMEX) time discretization for the
compressible Euler equations of gas dynamics, showing that they are asymptotic-preserving (AP)
in the low Mach number limit. The analysis is carried out for a general equation of state (EOS).
We consider both a single asymptotic length scale and two length scales. We then show that, when
coupling these time discretizations with a Discontinuous Galerkin (DG) space discretization with
appropriate fluxes, a numerical method effective for a wide range of Mach numbers is obtained.
A number of benchmarks for ideal gases and their non-trivial extension to non-ideal EOS validate
the performed analysis.
1. Introduction
The compressible Euler equations of gas dynamics are the standard mathematical model in several applications
such as atmosphere dynamics [81], combustion or astrophysics. For these equations, one can consider two opposite
regimes. In the first one, the acoustic waves are much faster than the local fluid velocity, while in the second one the
fluid moves at high speed and compressibility plays a key role. The relevant non-dimensional number which identifies
the regime is the local Mach number 𝑀𝑙𝑜𝑐, defined as 𝑀𝑙𝑜𝑐= |𝐮|
𝑐, where |𝐮| is the magnitude of the local fluid velocity
and 𝑐is the speed of sound. When the Mach number tends to zero, under suitable conditions, the compressible Euler
equations converge to the incompressible Euler equations, see [37, 59], and the references therein for the analysis of
singular limits of compressible flows. Weakly compressible flows are an example of problem with multiple length and
time scales. The design of efficient and stable numerical schemes for such models is a challenging task and typically
requires a specific numerical treatment of the terms related to compressibility effects.
The concept of asymptotic-preserving (AP) schemes has been introduced for this purpose, see, e.g., [46]. Consider
a continuous physical model 𝜀which involves a small perturbation parameter 𝜀≪1. Denote by 0 the limit of
𝜀when 𝜀→0, e.g. the incompressible Euler equations in our framework. Let now 𝜀
Δ𝑡be a time discretization
method which provides a consistent discretization of 𝜖. The scheme 𝜀
Δ𝑡is said to be asymptotic-preserving (AP)
if its stability condition is independent of 𝜀and if its limit 𝜀
Δ𝑡for 𝜀→0 provides a consistent discretization of the
continuous limit model 0. We analyze here the Euler equations of gas dynamics and the parameter 𝜀is represented
by the Mach number 𝑀, as defined in Section 2. Since the seminal contribution [60], several AP schemes for Euler
equations have been proposed in the literature, see among many others [1, 12, 24, 29, 31, 62, 63, 68, 86] and the
references therein. Methods that work at all values of the Mach number (including 𝑀≥1) are also available, see
for example the seminal paper [76]. While a complete review of all the different approaches for low Mach flows is
out of the scope of the present work, we briefly outline some of the strategies proposed in the literature to deal with
low Mach flows, in order to highlight the main differences with the numerical method considered here. Following the
discussion in [60], a class of AP methods [24, 25] proposes to decouple acoustic and transport phenomenon, leading
to the so-called Lagrange-Projection schemes. In these approaches, an operator splitting is applied, solving first the
transport subsystem and dealing with acoustic effects afterwards. Following again [60], another class of AP schemes
[29, 68] considers a splitting of the fluxes into non-stiff and stiff parts. More specifically, effects of global compression
or long-wave acoustics are considered explicitly and then an implicit pressure correction is applied. Another class of
popular methods are the so-called pressure correction schemes. They extend the projection techniques widely used
∗Corresponding author
giuseppe.orlando@polytechnique.edu (G. Orlando); luca.bonaventura@polimi.it (L. Bonaventura)
ORCID(s): 0000-0002-7119-4231 (G. Orlando); 0000-0002-1994-0217 (L. Bonaventura)
G. Orlando et al.: Preprint submitted to Elsevier
Page 1 of 38


AP IMEX schemes for Euler equations: non-ideal gases
for incompressible flows [27, 74, 84] and, starting from [48, 49], several approaches have been proposed [50, 51, 85].
Finally, a Suliciu type relaxation scheme [82], splitting the pressure in a slow and a fast acoustic part, was proposed in
[86], whereas a Jin-Xin type relaxation method, building a linear hyperbolic relaxation system with a small dissipative
correction to approximate the Euler equations, was presented in [1].
We analyze here the AP properties of a general class of Implicit-Explicit (IMEX) time discretization schemes.
The key observation is that, as first proposed in [23], it suffices to adopt an implicit treatment of the pressure gradient
term within the momentum equation and of the pressure work term in the energy equation to remove the acoustic
CFL restriction and to decouple acoustic and transport effects, see also Appendix A. Similar approaches have been
proposed, e.g., in [12, 19]. Here, we consider a general equation of state (EOS), to which only a small number of
studies have been devoted [1, 29, 31]. In particular, the single spatial scale analysis performed in [60] was first extended
to the general EOS case in [31]. Here, the corresponding extension to a general EOS is introduced also for the case
of two length scales. Notice that several low Mach schemes have been proposed for a barotropic equation of state
[17, 43, 52]. As discussed in [60], the assumption of a barotropic fluid, for which a direct relation between the pressure
and the density exists, restricts the analysis to constant-entropy data and the limit case is an incompressible flow
with constant density. However, large amplitude density fluctuations are crucial for an accurate description of reacting
flows [60], for atmospheric applications, and for the analysis of relevant fluid dynamics instabilities, as we will see
in Section 5. Finally, we show that a high-order numerical method effective for a wide range of Mach number values
can be obtained coupling these time discretizations with a Discontinuous Galerkin (DG) space discretization [41] with
appropriate fluxes. In a recent work, Jung and Perrier [56] analyzed the behaviour of the DG method for low Mach
regimes, showing under which conditions a low Mach number accurate method is obtained. We discuss the practical
implications of these results for our method, which however is shown to provide accurate results for Mach number
values corresponding to fluids typically modelled as incompressible. The numerical verification is based on the higher
order extension of the IMEX-DG method proposed in [70, 71, 72].
The paper is structured as follows. In Section 2, we present the formal limits of the continuous model considering
both a single length scale and two length scales. In Section 3, we show the AP property of a general class of IMEX-RK
methods, whereas in Section 4 we discuss some details of the DG formulation that allows us to obtain a numerical
scheme effective for a wide range of Mach numbers. In Section 5, some numerical results are presented to verify the
robustness of the proposed approach with 𝑀< 1 and 𝑀≪1, using the higher order extension of the numerical
method developed in [70, 71, 72]. Finally, some conclusions and perspectives for future work are discussed in Section
6.
2. Asymptotic analysis for the continuous model
Our goal is to discuss here the limit of the fully compressible Euler equations of gas dynamics as the Mach number
goes to zero. For this purpose, we introduce the Euler equations and recall their non-dimensional formulation. Let
Ω ⊂ℝ𝑑, 1 ≤𝑑≤3 be a connected open bounded set with a sufficiently smooth boundary 𝜕Ω and denote by 𝐱the
spatial coordinates and by 𝑡the temporal coordinate. The mathematical model reads as follows:
𝜕𝜌
𝜕𝑡+ ∇⋅(𝜌𝐮)
=
0
𝜕𝜌𝐮
𝜕𝑡+ ∇⋅(𝜌𝐮⊗𝐮) + ∇𝑝
=
𝟎
(1)
𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(𝜌𝐸+ 𝑝) 𝐮]
=
0.
Here 𝜌is the density, 𝐮is the fluid velocity, 𝑝is the pressure, and 𝐸is the total energy per unit of mass. The previous set
of equations has to be completed by an equation od state (EOS). Notice that no external source terms, such as gravity
terms, are considered in (1). The total energy 𝜌𝐸can be rewritten as 𝜌𝐸= 𝜌𝑒+ 𝜌𝑘, where 𝑒is the internal energy and
𝑘= 1
2 |𝐮|2 is the kinetic energy per unit of mass. We also introduce the specific enthalpy ℎ= 𝑒+ 𝑝
𝜌and we notice that
one can rewrite the energy flux as
(𝜌𝐸+ 𝑝) 𝐮=
(
𝑒+ 𝑘+ 𝑝
𝜌
)
𝜌𝐮= (ℎ+ 𝑘) 𝜌𝐮.
(2)
G. Orlando et al.: Preprint submitted to Elsevier
Page 2 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Hence, (1) can be rewritten as
𝜕𝜌
𝜕𝑡+ ∇⋅(𝜌𝐮)
=
0
𝜕𝜌𝐮
𝜕𝑡+ ∇⋅(𝜌𝐮⊗𝐮) + ∇𝑝
=
𝟎
(3)
𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(ℎ+ 𝑘) 𝜌𝐮]
=
0.
We now proceed to recall the non-dimensional version of system (3), along the lines of the analysis presented, e.g., in
[62], to which we refer for a more extensive discussion of the underlying hypotheses. We introduce reference scaling
values , , and for time, length, and velocity, respectively. We also introduce reference values for the pressure
and for the density. The Buckingham 𝜋theorem [18] states that there are 𝑛−𝑚relevant non-dimensional parameters
that characterize the model, where 𝑛is the number of independent physical variables and 𝑚is the rank of the matrix
which associates to each physical variable its unit of measure. Here, 𝑛= 5 and 𝑚= 3, as it can be easily verified
and discussed in detail in [62]. Hence, there are 2 non-dimensional parameters associated to (3). We assume that the
internal energy scales as ≈
and that the total energy scales as ≈+ 2. Finally, we assume that the specific
enthalpy scales as ≈+ 
. We then introduce the following non-dimensional parameters
𝑆𝑡=


𝑀2 = 2

(4)
and notice that
+ 


≈
2 


+ 2 =
2
𝑀2
1
𝑀2 + 1
= (1)
𝑈2
≈
2

+ 2 =
1
1 +
1
𝑀2
= (𝑀2) .
(5)
As a consequence, the non-dimensional version of (3) reads as follows:
𝑆𝑡𝜕𝜌
𝜕𝑡+ ∇⋅(𝜌𝐮)
=
0
𝑆𝑡𝜕𝜌𝐮
𝜕𝑡+ ∇⋅(𝜌𝐮⊗𝐮) +
1
𝑀2 ∇𝑝
=
𝟎
(6)
𝑆𝑡𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(ℎ+ 𝑘𝑀2) 𝜌𝐮]
=
0,
where, with a slight abuse of notation, the non-dimensional variables are denoted with the same symbols of the
dimensional ones. Finally, as customary in the literature, see, e.g., [62, 67], we assume that 𝑆𝑡≈1, so as to obtain
𝜕𝜌
𝜕𝑡+ ∇⋅(𝜌𝐮)
=
0
𝜕𝜌𝐮
𝜕𝑡+ ∇⋅(𝜌𝐮⊗𝐮) +
1
𝑀2 ∇𝑝
=
𝟎
(7)
𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(ℎ+ 𝑘𝑀2) 𝜌𝐮]
=
0.
Our goal is to present the formal limit of the continuous model both in the case of single length scale and two length
scales. Notice that the asymptotic limit for single length scale for a general EOS was already present in [31]. Before
achieving the proposed goal, we present the EOS that will be employed for the numerical simulations in Section 5.
2.1. The equation of state
System (7) has to be completed with an equation of state (EOS). In this work, we will consider the ideal gas law,
the stiffened gas EOS (SG-EOS) [64] and the general cubic EOS [80, p. 221], [89, p. 119], even though we point out
that the analyses which will be carried out in Sections 2.2, 2.3, and 3 are valid for a general EOS.
For an ideal gas, the equation that links together pressure, density, and internal energy is given by
𝑝= (𝛾−1) 𝜌𝑒= (𝛾−1)
(
𝜌𝐸−1
2𝑀2𝜌𝐮⋅𝐮
)
.
(8)
G. Orlando et al.: Preprint submitted to Elsevier
Page 3 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Notice that (8) is valid only for a constant value of the ratio 𝛾between the specific heat at constant pressure and the
specific heat at constant volume [89]. The analogous relation for the SG-EOS reads as follows:
𝑝= (𝛾−1)
(𝜌𝑒−𝜌𝑞∞
) −𝛾𝜋∞= (𝛾−1)
(
𝜌𝐸−1
2𝑀2𝜌𝐮⋅𝐮−𝜌𝑞∞
)
−𝛾𝜋∞,
(9)
with 𝑞∞and 𝜋∞representing constant parameters which determine the characteristics of the fluid. Notice that for
𝑞∞= 𝜋∞= 0 in (9), we recover (8). Finally, for the general cubic EOS the equation linking together internal energy,
density and temperature is given by [71], [89, p. 118]
𝑒= 𝑒♯(𝑇) +
𝑎(𝑇) + 𝑇𝑑𝑎
𝑑𝑇
𝑏
𝑈(𝜌, 𝑏, 𝑟1, 𝑟2
) .
(10)
Here, 𝑒♯(𝑇) denotes the internal energy of an ideal gas at temperature 𝑇, 𝑟1 and 𝑟2 are suitable constants, whereas
the parameters 𝑎(𝑇), 𝑏determine fluid characteristics [89]. More specifically, 𝑎(𝑇) is related to intermolecular forces,
while 𝑏, the so called co-volume, takes into account the volume occupied by the molecules. The expression of 𝑈is:
𝑈(𝜌, 𝑏, 𝑟1, 𝑟2
) =
1
𝑟1 −𝑟2
log
(1 −𝜌𝑏𝑟1
1 −𝜌𝑏𝑟2
)
.
(11)
Notice that, for 𝑟1 →0 and 𝑟2 →0, then 𝑈
→−𝑏𝜌, which corresponds to the van der Waals EOS. For
𝑟1 = −1 −
√
2, 𝑟2 = −1 +
√
2, we get the Peng-Robinson EOS [71], [80, p. 231,p. 482], [89, p. 118]. For the sake of
simplicity, we will assume in our numerical experiments that the coefficient 𝑎(𝑇) and the quantity 𝑑𝑒♯
𝑑𝑇are constants.
We refer to [71, 73] for the specific numerical treatment of the general cubic EOS in the more general case
𝑑𝑎
𝑑𝑇≠0
𝑑2𝑒♯
𝑑𝑇2 ≠0.
Nevertheless, we recall once more that the analyses which will be carried out in Sections 2.2, 2.3, and 3 are valid for
a general EOS, without requiring these simplifying assumptions. Finally, the equation linking pressure, density and
temperature for the general cubic EOS can be expressed as follows:
𝑝= 𝜌𝑅𝑇
1 −𝜌𝑏−
𝑎(𝑇)𝜌2
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
),
(12)
with 𝑅denoting the specific gas constant. We refer to [30] for a detailed discussion of the relationship between (12)
and (10). Notice that for 𝑎= 𝑏= 0, the equation for an ideal gas equation 𝑝= 𝜌𝑅𝑇is obtained. If 𝑎(𝑇) is constant,
(12) can be easily inverted so as to provide 𝑇(𝜌, 𝑝), i.e.
𝑇= 1 −𝜌𝑏
𝑅
(
𝑝
𝜌+
𝑎𝜌
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
)
)
.
(13)
Hence, substituting (13) into (10), the equation that links internal energy, pressure, and density that we consider for
our numerical simulations is the following
𝑒= 1 −𝜌𝑏
𝛾−1
(
𝑝
𝜌+
𝑎𝜌
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
)
)
+ 𝑎
𝑏𝑈(𝜌, 𝑏, 𝑟1, 𝑟2
) ,
(14)
with 𝛾denoting the specific heats ratio associated to 𝑒♯. We also recall here the expression of the speed of sound, which
will be employed to compute the acoustic Courant number (see Section 5). The speed of sound is defined for a generic
equation of state as [71, 89]:
𝑐2 = 𝜕𝑝
𝜕𝜌
||||𝑠
=
𝑝
𝜌2 −𝜕𝑒
𝜕𝜌
𝜕𝑒
𝜕𝑝
= −
𝜕ℎ
𝜕𝜌
𝜕𝑒
𝜕𝑝
,
(15)
G. Orlando et al.: Preprint submitted to Elsevier
Page 4 of 38


AP IMEX schemes for Euler equations: non-ideal gases
with 𝑠denoting the specific entropy. Hence, for the ideal gas law (8) we obtain
𝑐2 = 𝛾𝑝
𝜌.
(16)
For the SG-EOS, one has instead
𝑐2 = 𝛾𝑝+ 𝜋∞
𝜌
.
(17)
Finally, the speed of sound for the general cubic EOS reads as follows:
𝑐2
=
𝛾𝑝
𝜌
1
1 −𝜌𝑏−
𝑎𝜌
1 −𝜌𝑏
⎛
⎜
⎜⎝
𝜕𝑈
𝜕𝜌
𝑏(𝛾−1) +
1 −2𝜌𝑏
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
)
⎞
⎟
⎟⎠
−
𝑎𝑏𝜌2 𝑟1
(1 −𝜌𝑏𝑟2
) + 𝑟2
(1 −𝜌𝑏𝑟1
)
(1 −𝜌𝑏𝑟1
)2 (1 −𝜌𝑏𝑟2
)2
,
(18)
with
𝜕𝑈
𝜕𝜌= −
𝑏
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
).
(19)
Notice once more that, (18) is valid only if 𝑑𝑎
𝑑𝑇= 0 and 𝑑𝑒♯
𝑑𝑇is constant.
2.2. Asymptotic expansion for single length scale
In this Section, we analyze the formal limit of (7) as 𝑀→0 assuming that the solution depends on a single length
scale. We consider the following expansion for density, velocity, and pressure, respectively:
𝜌(𝐱, 𝑡)
=
̄𝜌(𝐱, 𝑡) + 𝑀𝜌
′(𝐱, 𝑡) + 𝑀2𝜌
′′(𝐱, 𝑡) + (𝑀3)
(20)
𝐮(𝐱, 𝑡)
=
̄𝐮(𝐱, 𝑡) + 𝑀𝐮
′(𝐱, 𝑡) + 𝑀2𝐮
′′(𝐱, 𝑡) + (𝑀3)
(21)
𝑝(𝐱, 𝑡)
=
̄𝑝(𝐱, 𝑡) + 𝑀𝑝
′(𝐱, 𝑡) + 𝑀2𝑝
′′(𝐱, 𝑡) + (𝑀3).
(22)
From now on, for the sake of simplicity in the notation, we omit the explicit dependence on space and time for all the
variables. Substituting (20) and (21) into the continuity equation in (7), the leading order term relation is
𝜕̄𝜌
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮) = 0.
(23)
For what concerns the momentum balance, the first two terms in the expansion reduce to
∇̄𝑝= 𝟎,
∇𝑝
′ = 𝟎,
(24)
which implies that ̄𝑝, 𝑝
′ do not depend on space. Moreover, the second order term reads as follows:
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝑝
′′ = 𝟎,
(25)
where 𝑝
′′ represents a dynamical pressure [29, 86], namely the standard pressure variable for incompressible flows
[60]. Finally, the leading order term for the energy equation is
𝜕̄𝜌𝑒( ̄𝑝, ̄𝜌)
𝜕𝑡
+ ∇⋅( ̄𝜌ℎ( ̄𝑝, ̄𝜌) ̄𝐮) = 0.
(26)
Notice that here we do not assume a Hilbert expansion for the internal energy 𝑒, and that 𝑒( ̄𝑝, ̄𝜌) and ℎ( ̄𝑝, ̄𝜌) denote
the expressions obtained from the equation of state evaluated at ̄𝑝, ̄𝜌. Other contributions in the literature, such as [63],
assume a Hilbert expansion also for the energy. The limit model obtained is the same in the case of single length
G. Orlando et al.: Preprint submitted to Elsevier
Page 5 of 38


AP IMEX schemes for Euler equations: non-ideal gases
scale, provided that 𝜌𝑒
̄𝜌= 𝑒( ̄𝑝, ̄𝜌), whereas some differences can arise in the case of the two length scale model. Since
̄𝜌𝑒( ̄𝑝, ̄𝜌) = ̄𝜌ℎ( ̄𝑝, ̄𝜌) −̄𝑝, we obtain
𝜕̄𝜌ℎ( ̄𝑝, ̄𝜌)
𝜕𝑡
−𝜕̄𝑝
𝜕𝑡+ ∇⋅( ̄𝜌ℎ( ̄𝑝, ̄𝜌) ̄𝐮) = 0,
(27)
or, equivalently, thanks to (23)
̄𝜌
(𝜕ℎ( ̄𝑝, ̄𝜌)
𝜕𝑡
+ ̄𝐮⋅∇ℎ( ̄𝑝, ̄𝜌)
)
−𝜕̄𝑝
𝜕𝑡= 0.
(28)
From (28), we get
̄𝜌𝜕ℎ( ̄𝑝, ̄𝜌)
𝜕̄𝜌
(𝜕̄𝜌
𝜕𝑡+ ̄𝐮⋅∇̄𝜌
)
+ ̄𝜌𝜕ℎ( ̄𝑝, ̄𝜌)
𝜕̄𝑝
(𝜕̄𝑝
𝜕𝑡+ ̄𝐮⋅∇̄𝑝
)
−𝜕̄𝑝
𝜕𝑡= 0.
(29)
Thanks to (23) and (24), we obtain
−̄𝜌2 𝜕ℎ( ̄𝑝, ̄𝜌)
𝜕̄𝜌
(∇⋅̄𝐮) + ̄𝜌𝜕𝑒( ̄𝑝, ̄𝜌)
𝜕̄𝑝
𝑑̄𝑝
𝑑𝑡= 0.
(30)
If ̄𝜌≠0 and 𝜕ℎ( ̄𝑝, ̄𝜌)
𝜕̄𝜌
≠0, as it holds away from vacuum, thanks to (15), relation (30) can be rewritten as
∇⋅̄𝐮= −
1
̄𝜌𝑐2 ( ̄𝑝, ̄𝜌)
𝑑̄𝑝
𝑑𝑡
(31)
Summing up, the asymptotic limit of (7) is
𝜕̄𝜌
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮)
=
0
∇̄𝑝
=
𝟎
∇𝑝
′
=
𝟎
(32)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝑝
′′
=
𝟎
𝜕̄𝜌𝑒( ̄𝑝, ̄𝜌)
𝜕𝑡
+ ∇⋅( ̄𝜌ℎ( ̄𝑝, ̄𝜌) ̄𝐮)
=
0,
or, equivalently,
𝜕̄𝜌
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮)
=
0
∇̄𝑝
=
𝟎
∇𝑝
′
=
𝟎
(33)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝑝
′′
=
𝟎
∇⋅̄𝐮
=
−
1
̄𝜌𝑐2 ( ̄𝑝, ̄𝜌)
𝑑̄𝑝
𝑑𝑡0.
The asymptotic limit (33) was already present in [31] and represents the extension to non-ideal gases of the system
of equations derived in [60]. Analogous relations have been derived in [29] for the case 𝜕̄𝑝
𝜕𝑡= 0. Under periodic or
free-slip boundary conditions, thanks to the divergence theorem, we have
∫Ω
∇⋅̄𝐮𝑑Ω = 0,
so that, by integrating (31) on Ω, we find 𝑑̄𝑝
𝑑𝑡= 0. However, as is evident from the last relation in (31), a time dependent
pressure with large amplitude variations imposed by Dirichlet boundary conditions leads to a non-incompressible flow,
G. Orlando et al.: Preprint submitted to Elsevier
Page 6 of 38


AP IMEX schemes for Euler equations: non-ideal gases
i.e. ∇⋅̄𝐮≠0, as we will verify numerically in Section 5.4. Hence, under periodic or free-slip boundary conditions or
if 𝑑̄𝑝
𝑑𝑡= 0, all the equations of state lead to the same limit, namely the incompressible Euler equations. On the other
hand, if 𝑑̄𝑝
𝑑𝑡≠0, then ∇⋅̄𝐮depends on the specific EOS and on its parameters. For the ideal gas law (8), we obtain
∇⋅̄𝐮= −1
𝛾
𝑑log ̄𝑝
𝑑𝑡
.
(34)
Hence, the compressibility of a fluid described by the ideal gas law is uniform in space and changes only in time. This
is no longer valid for a general EOS, as we will also show in Section 5.
2.3. Asymptotic expansion for two length scales
In this Section, following [6, 60], we try to account for the fact that, for sufficiently small values of the Mach
number, two decoupled spatial scales can be identified. More specifically, since the speed of sound 𝑐is much larger
than the typical flow velocity |𝐮| and if a unique time scale is considered, the typical length scale associated to acoustic
phenomena is much larger than that associated with the material flow. In order to properly highlight this fact, we
assume that the solution depends on the material scale variable 𝐱and also on the acoustic scale variable 𝝃= 𝑀𝐱.
Separate equations will then be derived for the material information, which moves at speed |𝐮|, and for the acoustic
information, which moves approximately at the speed of sound 𝑐[17, 38]. Relevant applications which show the
interaction between the two scales arise in reacting flows [61], in the interaction of shocks with large density gradients
[16] and in atmospheric models, as we will show in Section 5.6. An analogous analysis can be performed considering
a single length scale and two time scales, as done, e.g., in [17, 38]. In an asymptotic analysis with two spatial scales,
we consider the following expansion for any dependent variable:
𝑓(𝐱, 𝝃, 𝑡) = ̄𝑓(𝐱, 𝝃, 𝑡) + 𝑀𝑓
′ (𝐱, 𝝃, 𝑡) + 𝑀2𝑓
′′ (𝐱, 𝝃, 𝑡) + (𝑀3),
(35)
so that a large scale spatial derivative operator appears in the asymptotic expansion. More specifically, we get
∇𝑓= ∇𝐱𝑓+ 𝑀∇𝝃𝑓.
(36)
One can easily notice from (36) that the leading order relations are not modified introducing 𝝃, provided that we
reinterpret ∇□and ∇⋅□as ∇𝐱□and ∇𝐱⋅□, respectively. Equations (24) and (25) change because of 𝝃. Indeed,
since
∇𝑝= ∇𝐱̄𝑝+ 𝑀
(
∇𝐱𝑝
′ + ∇𝝃̄𝑝
)
+ 𝑀2 (
∇𝐱𝑝
′′ + ∇𝝃𝑝
′)
+ (𝑀3),
(37)
we obtain
∇𝐱𝑝
′ + ∇𝝃̄𝑝
=
𝟎
(38)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝐱𝑝
′′ + ∇𝝃𝑝
′
=
𝟎.
(39)
We also consider the first order term of the continuity equation, which reduces to
𝜕𝜌
′
𝜕𝑡+ ∇𝐱⋅
(
𝜌
′ ̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌𝐮
′)
+ ∇𝝃⋅( ̄𝜌̄𝐮) = 0.
(40)
Finally, we consider the first order term of the energy equation, which reads as follows:
𝜕𝜌
′𝑒
(
𝑝
′, 𝜌
′)
𝜕𝑡
+ ∇𝐱⋅
(
𝜌
′ℎ
(
𝑝
′, 𝜌
′)
̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌ℎ( ̄𝑝, ̄𝜌) 𝐮
′)
+ ∇𝝃⋅( ̄𝜌̄𝐮ℎ( ̄𝑝, ̄𝜌)) = 0.
(41)
Notice that, relation (41) implicitly assumes that a Hilbert expansion holds for 𝜌𝑒, so that the first order contribution
for 𝜌𝑒reduces to 𝜌
′𝑒
(
𝑝
′, 𝜌
′)
. However, other options are possible; as an example, assuming a Hilbert expansion for 𝑒
would lead to
𝜕
(
̄𝜌𝑒
′ + 𝜌
′ ̄𝑒
)
𝜕𝑡
+ ∇𝐱⋅
(
𝜌
′ ̄𝑒̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌𝑒
′ ̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌̄𝑒𝐮
′)
+ ∇𝐱⋅
(
̄𝑝𝐮
′)
+ ∇𝐱⋅
(
𝑝
′ ̄𝐮
)
+ ∇𝝃⋅( ̄𝜌̄𝐮̄ℎ) = 0.
(42)
G. Orlando et al.: Preprint submitted to Elsevier
Page 7 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Summing up, the asymptotic limit of (7) for a two-scale analysis is
𝜕̄𝜌
𝜕𝑡+ ∇𝐱⋅( ̄𝜌̄𝐮)
=
0
∇𝐱̄𝑝
=
𝟎
∇𝐱𝑝
′ + ∇𝝃̄𝑝
=
𝟎
(43)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇𝐱⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝐱𝑝
′′ + ∇𝝃𝑝
′
=
𝟎
𝜕̄𝜌𝑒( ̄𝑝, ̄𝜌)
𝜕𝑡
+ ∇𝐱⋅( ̄𝜌ℎ( ̄𝑝, ̄𝜌) ̄𝐮)
=
0
𝜕𝜌
′
𝜕𝑡+ ∇𝐱⋅
(
𝜌
′ ̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌𝐮
′)
+ ∇𝝃⋅( ̄𝜌̄𝐮)
=
0
𝜕𝜌
′𝑒
(
𝑝
′, 𝜌
′)
𝜕𝑡
+ ∇𝐱⋅
(
𝜌
′ℎ
(
𝑝
′, 𝜌
′)
̄𝐮
)
+ ∇𝐱⋅
(
̄𝜌ℎ( ̄𝑝, ̄𝜌) 𝐮
′)
+ ∇𝝃⋅( ̄𝜌̄𝐮ℎ( ̄𝑝, ̄𝜌))
=
0.
Following the discussion in [60], we then focus on the regime in which variations on the material scale are negligible
and only variations on the large acoustic scale are relevant. Starting from (43), these assumptions imply that
𝜕̄𝜌
𝜕𝑡
=
0
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇𝝃𝑝
′
=
𝟎
∇𝝃̄𝑝
=
𝟎
(44)
𝜕̄𝑝
𝜕𝑡
=
0
𝜕𝜌
′
𝜕𝑡+ ∇𝝃⋅( ̄𝜌̄𝐮)
=
0
𝜕𝜌
′𝑒
(
𝑝
′, 𝜌
′)
𝜕𝑡
+ ∇𝝃⋅( ̄𝜌̄𝐮ℎ( ̄𝑝, ̄𝜌))
=
0.
The relation 𝜕̄𝑝
𝜕𝑡= 0 is a direct consequence of the fact that ∇𝐱⋅̄𝐮= 0, since we neglect variations on the material scale.
Moreover, we notice that ̄𝑝reduces to a constant. In the particular case of the ideal gas law (8), system (44) reduces to
𝜕̄𝜌
𝜕𝑡
=
0
𝜕̄𝐮
𝜕𝑡+
1
̄𝜌(𝝃) ∇𝝃𝑝
′
=
𝟎
∇𝝃̄𝑝
=
𝟎
(45)
𝜕𝜌
′
𝜕𝑡+ ∇𝝃⋅( ̄𝜌̄𝐮)
=
0
𝜕̄𝑝
𝜕𝑡
=
0
𝜕𝑝
′
𝜕𝑡+ 𝛾̄𝑝∇𝝃⋅̄𝐮
=
0.
Taking the time derivative of the last equation, we obtain
𝜕2𝑝
′
𝜕𝑡2 = ∇𝝃⋅
(
𝑐( ̄𝑝, ̄𝜌)2 ∇𝝃𝑝
′)
,
(46)
G. Orlando et al.: Preprint submitted to Elsevier
Page 8 of 38


AP IMEX schemes for Euler equations: non-ideal gases
with 𝑐( ̄𝑝, ̄𝜌)2 = 𝛾̄𝑝
̄𝜌. Equation (46) is the wave equation for 𝑝
′ already derived in [60]. The time derivative of the first
order term of the energy equation in (44) reduces to
𝜕2𝜌
′𝑒
(
𝜌
′, 𝑝
′)
𝜕𝑡2
= ∇𝝃⋅
(
ℎ( ̄𝑝, ̄𝜌) ∇𝝃𝑝
′)
.
(47)
Starting from (47), one can verify that (46) is valid also for the SG-EOS (9). Indeed, since
𝜕2𝜌
′𝑒
(
𝜌
′, 𝑝
′)
𝜕𝑡2
=
1
𝛾−1
𝜕2𝑝
′
𝜕𝑡2 + 𝑞∞
𝜕2𝜌
′
𝜕𝑡2
and
𝜕2𝜌
′
𝜕𝑡2 = ∇𝝃⋅∇𝝃𝑝
′,
we obtain
𝜕2𝑝
′
𝜕𝑡2 = ∇𝝃⋅
[
(𝛾−1)
(ℎ( ̄𝑝, ̄𝜌) −𝑞∞
) ∇𝝃𝑝
′]
.
(48)
Since
ℎ( ̄𝑝, ̄𝜌) =
𝛾( ̄𝑝+ 𝜋∞
)
̄𝜌(𝛾−1)
+ 𝑞∞,
we recover relation (46) thanks to (17). Relation (46) is instead in general no longer valid for a general EOS and
supplementary terms arise for the general cubic EOS (14).
3. Asymptotic analysis for a class of IMEX-RK schemes
We analyze now the AP property of a general class of Implicit-Explicit Runge-Kutta (IMEX-RK) methods for
the time discretization of system (7). Following [23, 35], we couple implicitly the energy equation to the momentum
one, while the continuity equation is discretized in a fully explicit fashion. As a result, at each stage of the IMEX-RK
method, we will obtain a nonlinear Helmholtz equation for the pressure, which is solved through a fixed point procedure
[35, 71]. The time discretization is based on an IMEX-RK scheme [57], as done in [70, 71, 72]. IMEX-RK schemes
are represented compactly by the companion Butcher tableaux [20]:
𝐜
𝐀
𝐛𝑇
̃𝐜
̃𝐀
̃𝐛𝑇
with 𝐀= {𝑎𝑙𝑚
} , 𝐛= {𝑏𝑙
} , 𝐜= {𝑐𝑙
} , ̃𝐀= { ̃𝑎𝑙𝑚
} ,̃𝐛= {̃𝑏𝑙
}, and ̃𝐜= {̃𝑐𝑙
}, 𝑙, 𝑚= 1 … 𝑠, where 𝑠denotes the number
of stages of the method. Notice that matrix 𝐀corresponds to the explicit part of the scheme, i.e. 𝑎𝑖,𝑗= 0 for 𝑗≥𝑖,
while ̃𝐀corresponds to the implicit part of the scheme. Coefficients 𝑎𝑙𝑚, ̃𝑎𝑙𝑚, 𝑐𝑙, ̃𝑐𝑙, 𝑏𝑙, and ̃𝑏𝑙are determined so that the
method is consistent of a given order. In particular, the following relation has to be satisfied [57]:
𝑠
∑
𝑙=1
𝑏𝑙=
𝑠
∑
𝑙=1
̃𝑏𝑙= 1.
(49)
We then introduce the following Definition, which characterizes different IMEX-RK schemes according to the structure
of the implicit method:
Definition 3.1. An IMEX-RK method is said to be of type I [10, 75] if the matrix ̃𝐀is invertible. It is said to be of type
II [10, 57] if the matrix ̃𝐀can be written in the form
̃𝐀=
(0
0
̃𝐚
̃
)
,
with ̃𝐚= ( ̃𝑎21, … , ̃𝑎𝑠1
)⊤∈ℝ𝑠−1 and the matrix ̃∈ℝ(𝑠−1)×(𝑠−1) is invertible. In the special case ̃𝐚= 0, 𝑏1 = 0,
the method is said of type ARS (Ascher, Ruuth and Spiteri) [4] and the implicit method is reducible to a method using
𝑠−1 stages.
G. Orlando et al.: Preprint submitted to Elsevier
Page 9 of 38


AP IMEX schemes for Euler equations: non-ideal gases
We assume that the implicit scheme is a Diagonally Implicit Runge-Kutta (DIRK) method, namely ̃𝑎𝑙𝑚= 0 for
𝑙> 𝑚. Following the Butcher tableaux introduced above, for a time dependent problem
𝑑𝐲
𝑑𝑡= 𝐟𝐸(𝐲, 𝑡) + 𝐟𝐼(𝐲, 𝑡) ,
(50)
the generic 𝑙-stage of an IMEX-RK method can be defined as
𝐯(𝑛,𝑙)
=
𝐯𝑛+ Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚𝐟𝐸
(𝐯(𝑛,𝑚), 𝑡𝑛+ 𝑐𝑚Δ𝑡) + Δ𝑡
𝑙∑
𝑚=1
̃𝑎𝑙𝑚𝐟𝐼
(𝐯(𝑛,𝑚), 𝑡𝑛+ ̃𝑐𝑚Δ𝑡) ,
(51)
where 𝑙= 1, … , 𝑠, Δ𝑡is the time discretization step, 𝐯𝑛≈𝐲(𝑡𝑛), 𝐟𝐸is the term treated explicitly, and 𝐟𝐼is the term
treated implicitly. After computation of the intermediate stages, the updated solution is computed as follows:
𝐯𝑛+1 = 𝐯𝑛+ Δ𝑡
𝑠
∑
𝑙=1
𝑏𝑙𝐟𝐸
(𝐯(𝑛,𝑙), 𝑡𝑛+ 𝑐𝑙Δ𝑡) + Δ𝑡
𝑠
∑
𝑙=1
̃𝑏𝑙𝐟𝐼
(𝐯(𝑛,𝑙), 𝑡𝑛+ ̃𝑐𝑙Δ𝑡) .
(52)
The formulation (51)-(52) is valid for an IMEX scheme of arbitrary order. We recall that implicit methods of order
higher than one cannot be unconditionally total variation diminishing (TVD) for hyperbolic problems [42]. This also
holds for IMEX methods [12, 34]. In this work, as done, e.g., in [12], we do not focus on this limit imposed by high order
schemes and we consider therefore numerical methods which, in principle, may not guarantee 𝐿∞-stability. Notice also
that the existence of the Hilbert expansion (55) can be justified only for smooth functions [63]. The development of
a numerical treatment to avoid this issue goes beyond the scope of the present work and will be carried out as future
development. For our analysis, we assume
𝑠
∑
𝑚=1
𝑎𝑙𝑚= 𝑐𝑙
𝑠
∑
𝑚=1
̃𝑎𝑙𝑚= ̃𝑐𝑙.
(53)
Relation (53) is an usual assumption for Runge-Kutta schemes [9, 57], which simplifies the order conditions and,
moreover, guarantees that a method of at least first order is employed at each stage. Notice that, for IMEX-RK methods
of type I, 𝑐1 ≠̃𝑐1 because of (53). For the following analyses, we consider methods of type I for which 𝑐𝑙= ̃𝑐𝑙for
𝑙> 1 and methods of type II with 𝐜= ̃𝐜. The assumption 𝐜= ̃𝐜also allows to simplify the order conditions and has
been employed, e.g., in [4, 12]. A generic stage of the Euler equations reads as follows:
𝜌(𝑛,𝑙)
=
𝜌𝑛−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇⋅(𝜌(𝑛,𝑚)𝐮(𝑛,𝑚))
𝜌(𝑛,𝑙)𝐮(𝑛,𝑙) +
1
𝑀2 ̃𝑎𝑙𝑙Δ𝑡∇𝑝(𝑛,𝑙)
=
𝜌𝑛𝐮𝑛−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇⋅(𝜌(𝑛,𝑚)𝐮(𝑛,𝑚) ⊗𝐮(𝑛,𝑚))
−
1
𝑀2
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇𝑝(𝑛,𝑚)
(54)
𝜌(𝑛,𝑙)𝐸(𝑛,𝑙) + ̃𝑎𝑙𝑙Δ𝑡∇⋅(ℎ(𝑛,𝑙)𝜌(𝑛,𝑙)𝐮(𝑛,𝑙))
=
𝜌𝑛𝐸𝑛−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇⋅(ℎ(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚))
−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝑀2 ∇⋅(𝑘(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)) .
We analyze now the behaviour of the time semi-discretization as 𝑀→0, so as to verify that it provides a consistent
semi-discretization for the two limit models identified in Section 2.2 and Section 2.3, respectively.
3.1. Asymptotic analysis in the single length scale case
In this Section, we consider the limit model (32)-(33). Following, e.g., [63], we make the assumption that, at each
stage, the discrete quantities admit a formal expansion analogous to the continuous case.
G. Orlando et al.: Preprint submitted to Elsevier
Page 10 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Assumption 3.2. The physical variables 𝜌, 𝐮, and 𝑝admit at each stage a formal Hilbert expansion of the form (written,
e.g., for 𝜌𝑛)
𝜌𝑛(𝐱) = ̄𝜌𝑛(𝐱) + 𝑀𝜌
′,𝑛(𝐱) + 𝑀2𝜌
′′,𝑛(𝐱) + (𝑀3).
(55)
We also make the following assumption:
Assumption 3.3. In the case of schemes of type II that are not of type ARS, the initial datum 𝑝0 is well-prepared,
namely ∇̄𝑝0 = ∇𝑝
′,0 = 𝟎.
Then, the following result holds:
Theorem 3.4. Under Assumption 3.2 and Assumption 3.3, (54) provides a consistent discretization of (32)-(33) in the
limit 𝑀→0.
Proof. We plug asymptotic expansions of the form (55) into (54). The discrete limit system reads therefore as follows:
̄𝜌(𝑛,𝑙)
=
̄𝜌𝑛−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇⋅( ̄𝜌(𝑛,𝑚)̄𝐮(𝑛,𝑚))
̃𝑎𝑙𝑙∇̄𝑝(𝑛,𝑙)
=
−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇̄𝑝(𝑛,𝑚)
̃𝑎𝑙𝑙∇𝑝
′,(𝑛,𝑙)
=
−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇𝑝
′,(𝑛,𝑚)
(56)
̄𝜌(𝑛,𝑙)̄𝐮(𝑛,𝑙) + ̃𝑎𝑙𝑙Δ𝑡∇𝑝
′′,(𝑛,𝑙)
=
̄𝜌𝑛̄𝐮𝑛−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇⋅( ̄𝜌(𝑛,𝑚)̄𝐮(𝑛,𝑚) ⊗̄𝐮(𝑛,𝑚))
−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇𝑝
′′,(𝑛,𝑚)
̄𝜌(𝑛,𝑙)𝑒( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙)) + ̃𝑎𝑙𝑙Δ𝑡∇⋅(ℎ( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙)) ̄𝜌(𝑛,𝑙)̄𝐮(𝑛,𝑙))
=
̄𝜌𝑛𝑒( ̄𝜌𝑛, ̄𝑝𝑛) −
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇⋅(ℎ( ̄𝜌𝑛, ̄𝑝𝑛) ̄𝜌(𝑛,𝑚)̄𝐮(𝑛,𝑚)) .
First, we focus on the leading order terms of the momentum equation. For schemes of type I, since ̃𝑎11 ≠0, we obtain
∇̄𝑝(𝑛,1) = 𝟎and therefore ∇̄𝑝(𝑛,𝑙) = 𝟎for 𝑙≥1. For schemes of type ARS, since ̃𝑎𝑙1 = 0, we obtain ∇̄𝑝(𝑛,𝑙) = 𝟎for
𝑙> 1. For the other schemes, we need Assumption (3.3) to obtain a consistent discretization. Analogous considerations
hold for the discretization of ∇𝑝
′. The consistency of the remaining relations is a direct consequence of the consistency
of the IMEX method. Nevertheless, we want to show that the last relation yields a consistent discretization for (30), so as
to prove that (56) is a consistent discretization of (33). After a few manipulations, taking into account that ∇̄𝑝(𝑛,𝑙) = 𝟎,
we get
̄𝜌(𝑛,𝑙)ℎ( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙)) −̄𝑝(𝑛,𝑙) + ̃𝑎𝑙𝑙Δ𝑡
[
̄𝜌(𝑛,𝑙)ℎ( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙)) (∇⋅̄𝐮(𝑛,𝑙)) + ̄𝐮(𝑛,𝑙) ⋅∇̄𝜌(𝑛,𝑙) 𝜕̄𝜌(𝑛,𝑙)ℎ( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙))
𝜕̄𝜌(𝑛,𝑙)
]
=
̄𝜌𝑛ℎ( ̄𝜌𝑛, ̄𝑝𝑛) −̄𝑝𝑛−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡
[
̄𝜌(𝑛,𝑚)ℎ( ̄𝜌𝑛, ̄𝑝𝑛)
(∇⋅̄𝐮(𝑛,𝑚)) + ̄𝐮(𝐧,𝐦) ⋅∇̄𝜌(𝑛,𝑚) 𝜕̄𝜌(𝑛,𝑚)ℎ( ̄𝜌𝑛, ̄𝑝𝑛)
𝜕̄𝜌(𝑛,𝑚)
]
.
(57)
From now, for the sake of simplicity in the notation, we denote ℎ( ̄𝜌(𝑛,𝑙), ̄𝑝(𝑛,𝑙)) by ̄ℎ(𝑛,𝑙) and ℎ( ̄𝜌𝑛, ̄𝑝𝑛) by ̄ℎ𝑛. The error
obtained applying (57) to the exact solution reads therefore as follows:
G. Orlando et al.: Preprint submitted to Elsevier
Page 11 of 38


AP IMEX schemes for Euler equations: non-ideal gases
̂𝜏(𝑛,𝑙)
=
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) −[ ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
]
+
̃𝑎𝑙𝑙Δ𝑡[ ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) (∇⋅̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡))]
+
̃𝑎𝑙𝑙Δ𝑡
[
̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ⋅∇̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) 𝜕̄𝜌̄ℎ
𝜕̄𝜌
(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡)]
(58)
+
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡[ ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) (∇⋅̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡))]
+
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡
[
̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ⋅∇̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) 𝜕̄𝜌̄ℎ
𝜕̄𝜌
(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡)]
.
Since ̄𝜌̄ℎ= ̄𝜌̄ℎ( ̄𝜌, ̄𝑝), using a Taylor expansion, we get
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡)
=
̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) + 𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛)
[ ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛)
]
+
𝜕̄𝜌̄ℎ
𝜕̄𝑝(𝐱, 𝑡𝑛)
[ ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
]
(59)
+
𝑜( ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛)
) + 𝑜( ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
) .
Employing now the discretization of the continuity equation in (56), we obtain for 𝑙> 1
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡)
=
̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) −
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ∇⋅( ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡))
+
𝜕̄𝜌̄ℎ
𝜕̄𝑝(𝐱, 𝑡𝑛)
[ ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
]
(60)
+
𝑜( ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛)
) + 𝑜( ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
) ,
or, equivalently,
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄ℎ(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡)
=
̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) −
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ∇⋅̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡)
−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ⋅∇̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡)
+
𝜕̄𝜌̄ℎ
𝜕̄𝑝(𝐱, 𝑡𝑛)
[ ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
]
(61)
+
𝑜( ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛)
) + 𝑜( ̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝑝(𝐱, 𝑡𝑛)
) .
Using again a Taylor expansion, we get
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) = ̄𝜌(𝐱, 𝑡𝑛) + 𝑐𝑙Δ𝑡𝜕̄𝜌
𝜕𝑡(𝐱, 𝑡𝑛) + (Δ𝑡2)
̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) = ̄𝐮(𝐱, 𝑡𝑛) + 𝑐𝑙Δ𝑡𝜕̄𝐮
𝜕𝑡(𝐱, 𝑡𝑛) + (Δ𝑡2)
(62)
̄𝑝(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) = ̄𝑝(𝐱, 𝑡𝑛) + 𝑐𝑙Δ𝑡𝜕̄𝑝
𝜕𝑡(𝐱, 𝑡𝑛) + (Δ𝑡2).
Substituting (61) and (62) into (58), we obtain
G. Orlando et al.: Preprint submitted to Elsevier
Page 12 of 38


AP IMEX schemes for Euler equations: non-ideal gases
̂𝜏(𝑛,𝑙)
=
−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ̄𝜌(𝐱, 𝑡𝑛) ∇⋅̄𝐮(𝐱, 𝑡𝑛) −
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ̄𝐮(𝐱, 𝑡𝑛) ⋅∇̄𝜌(𝐱, 𝑡𝑛)
+
(𝜕̄𝜌̄ℎ
𝜕̄𝑝(𝐱, 𝑡𝑛) −1
)
𝑐𝑙Δ𝑡𝜕̄𝑝
𝜕𝑡(𝐱, 𝑡𝑛) +
𝑙∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡[ ̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) ∇⋅̄𝐮(𝐱, 𝑡𝑛)
]
(63)
+
𝑙∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡
[
̄𝐮(𝐱, 𝑡𝑛) ⋅∇̄𝜌(𝐱, 𝑡𝑛) 𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛)
]
+ (Δ𝑡2) .
Since
𝑙∑
𝑚=1
̃𝑎𝑙𝑚=
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚= 𝑐𝑙(53) and ̄𝜌(𝐱, 𝑡𝑛) ̄ℎ(𝐱, 𝑡𝑛) −𝜕̄𝜌̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ̄𝜌(𝐱, 𝑡𝑛) = −̄𝜌2 (𝐱, 𝑡𝑛) 𝜕̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛), we obtain
̂𝜏(𝑛,𝑙) = −𝑐𝑙Δ𝑡̄𝜌2 (𝐱, 𝑡𝑛) 𝜕̄ℎ
𝜕̄𝜌(𝐱, 𝑡𝑛) ∇⋅̄𝐮(𝐱, 𝑡𝑛) +
(𝜕̄𝜌̄ℎ
𝜕̄𝑝(𝐱, 𝑡𝑛) −1
)
𝑐𝑙Δ𝑡𝜕̄𝑝
𝜕𝑡(𝐱, 𝑡𝑛) + (Δ𝑡2) = (Δ𝑡2) ,
(64)
thanks to (30). Finally, the update stage for the energy equation reads as follows:
̄𝜌𝑛+1𝑒( ̄𝜌𝑛+1, ̄𝑝𝑛+1) = ̄𝜌𝑛𝑒( ̄𝜌𝑛, ̄𝑝𝑛) −
𝑠
∑
𝑙=1
̃𝑏𝑙∇⋅( ̄𝜌(𝑛,𝑚)ℎ( ̄𝜌(𝑛,𝑚), ̄𝑝(𝑛,𝑚)) ̄𝐮(𝑛,𝑚))
(65)
The error obtained applying (65) to the exact solution is
̂𝜏𝑛+1
=
̄𝜌(𝐱, 𝑡𝑛+ Δ𝑡) 𝑒( ̄𝜌(𝐱, 𝑡𝑛+ Δ𝑡) , ̄𝑝(𝐱, 𝑡𝑛+ Δ𝑡)) −̄𝜌(𝐱, 𝑡𝑛) 𝑒( ̄𝜌(𝐱, 𝑡𝑛) , ̄𝑝(𝐱, 𝑡𝑛))
−
𝑠
∑
𝑙=1
Δ𝑡̃𝑏𝑙∇⋅( ̄𝜌(𝐱, 𝑡𝑛+ ̃𝑐𝑙Δ𝑡) ℎ( ̄𝜌(𝐱, 𝑡𝑛+ ̃𝑐𝑙Δ𝑡) , ̄𝑝(𝐱, 𝑡𝑛+ ̃𝑐𝑙Δ𝑡)) ̄𝐮(𝐱, 𝑡𝑛+ ̃𝑐𝑙Δ𝑡)) .
(66)
Thanks to a Taylor expansion, we get
̂𝜏𝑛+1 = Δ𝑡
[
𝜕̄𝜌𝑒( ̄𝜌, ̄𝑝)
𝜕𝑡
(𝐱, 𝑡𝑛) +
𝑠
∑
𝑙=1
̃𝑏𝑙∇⋅( ̄𝜌(𝐱, 𝑡𝑛) ℎ( ̄𝜌(𝐱, 𝑡𝑛) , ̄𝑝(𝐱, 𝑡𝑛)) ̄𝐮(𝐱, 𝑡𝑛))
]
+ (Δ𝑡2) .
(67)
Since
𝑠∑
𝑙=1
̃𝑏𝑙= 1 and thanks to (30), we obtain
̂𝜏𝑛+1 = Δ𝑡
[𝜕̄𝜌𝑒( ̄𝜌, ̄𝑝)
𝜕𝑡
(𝐱, 𝑡𝑛) + ∇⋅( ̄𝜌(𝐱, 𝑡𝑛) ℎ( ̄𝜌(𝐱, 𝑡𝑛) , ̄𝑝(𝐱, 𝑡𝑛)) ̄𝐮(𝐱, 𝑡𝑛))
]
+ (Δ𝑡2) = (Δ𝑡2) .
(68)
The consistency for the remaining relations can be shown in an analogous manner.
Since we are considering an implicit coupling between the momentum and the energy balance, the stability
condition of the numerical method does not depend on 𝑀or on the acoustic speed of sound (see, e.g., [23, 35, 83]),
meaning that (54) provides an AP scheme for (32)-(33). Only a mild CFL-type restriction based on the flow velocity is
necessary for these schemes [35]. Schemes of type I and of type ARS are also strongly asymptotic-preserving, namely
they are asymptotic-preserving for general initial data. On the contrary, schemes of type II which are not of type ARS
turn out to be weakly asymptotic-preserving. Indeed, they require a well-prepared initial datum for the pressure (see
Assumption 3.3). However, when the limit model reduces to the incompressible Euler equations thanks to suitable
boundary conditions, we do not need a divergence-free initial velocity field to recover the incompressible limit, as we
will also verify numerically in Section 5.3. This is not valid for all the AP methods presented in the literature, see, e.g.,
[68], which has been subsequently corrected in [8], or [86].
The AP property guarantees the consistency of the discretization as 𝑀→0, but it does not imply that a scheme
preserves its order of accuracy as 𝑀→0. In this latter case, the scheme is said to be asymptotically-accurate (AA).
Since the seminal paper [75], it is quite established that L-stability is necessary to guarantee asymptotic accuracy. A
G. Orlando et al.: Preprint submitted to Elsevier
Page 13 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Runge-Kutta scheme is said to be L-stable [90] if it is A-stable and 𝑅(𝑧) →0 as 𝑧→∞, where 𝑅(𝑧) is the stability
function. Following the result in [90], a L-stable scheme results from the combination of a A-stable scheme with a
stiffly-accurate (SA) scheme, i.e. a scheme for which the update stage is identical to the last internal stage. However,
for methods of type II, this combination does not necessarily lead to a L-stable scheme, because the matrix ̃𝐀is not
invertible [11]. In the case of methods of type II which are stiffly-accurate, a supplementary condition is required to
obtain the L-stability, i.e. [11]
𝑅∞=
𝑠
∑
𝑚=2
̂𝑤𝑠𝑚̃𝑎𝑚1 = 0,
(69)
where ̂𝑤𝑙𝑚denotes the elements of the inverse of ̃. Hence, SA schemes of type ARS are also L-stable.
3.2. Asymptotic analysis for two length scales
In this Section, we consider the limit model (43). We replace Assumption 3.2 with the following one:
Assumption 3.5. The physical variables 𝜌, 𝐮, and 𝑝admit at each stage a formal Hilbert expansion of the form (written,
e.g., for 𝜌𝑛)
𝜌𝑛(𝐱, 𝝃) = ̄𝜌𝑛(𝐱, 𝝃) + 𝑀𝜌
′,𝑛(𝐱, 𝝃) + 𝑀2𝜌
′′,𝑛(𝐱, 𝝃) + (𝑀3),
(70)
with 𝝃= 𝑀𝐱.
Moreover, we replace Assumption 3.3 with the
Assumption 3.6. In the case of IMEX-RK schemes of type II that are not of type ARS, the initial datum 𝑝0 is well-
prepared, namely ∇𝐱̄𝑝0 = ∇𝐱̄𝑝
′ = ∇𝝃𝑝
′ = 𝟎.
Then, the following result holds:
Theorem 3.7. Under Assumption 3.5 and Assumption 3.6, (54) provides an AP scheme for (43).
Proof. As pointed out for the continuous model, the leading order term relations do not change when also introducing
the acoustic length scale variable 𝝃= 𝑀𝐱. We plug asymptotic expansion of the form (70) into the semi-discretized
momentum equation, so as to obtain for the first order term
̃𝑎𝑙𝑙∇𝐱𝑝
′,(𝑛,𝑙) + ̃𝑎𝑙𝑙∇𝝃̄𝑝(𝑛,𝑙) = −
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇𝐱𝑝
′,(𝑛,𝑚) −
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇𝝃̄𝑝(𝑛,𝑚).
(71)
Analogous considerations to those reported in Section 3.1 hold for the above relation. More specifically, for schemes
of type I, since ̃𝑎11 ≠0, we obtain
∇𝐱𝑝
′,(𝑛,𝑙) + ∇𝝃̄𝑝(𝑛,𝑙) = 𝟎
for 𝑙≥1.
For what concerns schemes of type ARS, since ̃𝑎𝑙1 = 0, 𝑙= 1 … 𝑠, we get
∇𝐱𝑝
′,(𝑛,𝑙) + ∇𝝃̄𝑝(𝑛,𝑙) = 𝟎
for 𝑙> 1.
Assumption 3.6 is instead necessary to obtain a consistent discretization of (38) for the other schemes. For what
concerns the second order term, we get
̄𝜌(𝑛,𝑙)̄𝐮(𝑛,𝑙) + ̃𝑎𝑙𝑙Δ𝑡
(
∇𝐱𝑝
′′,(𝑛,𝑙) + ∇𝝃𝑝
′,(𝑛,𝑙))
=
̄𝜌𝑛̄𝐮𝑛−
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇𝐱⋅( ̄𝜌(𝑛,𝑚)̄𝐮(𝑛,𝑚) ⊗̄𝐮(𝑛,𝑚))
(72)
−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡
(
∇𝐱𝑝
′′,(𝑛,𝑚) + ∇𝝃𝑝
′,(𝑛,𝑚))
.
One can easily verify that (72) is a consistent discretization of (39). Indeed, the error obtained applying (72) to the
exact solution reads as follows:
G. Orlando et al.: Preprint submitted to Elsevier
Page 14 of 38


AP IMEX schemes for Euler equations: non-ideal gases
̂𝜏(𝑛,𝑙)
𝜉
=
̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) ̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑙Δ𝑡) −̄𝜌(𝐱, 𝑡𝑛) ̄𝐮(𝐱, Δ𝑡)
+
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚Δ𝑡∇𝐱⋅( ̄𝜌(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) ⊗̄𝐮(𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡))
+
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡
(
∇𝐱𝑝
′′ (𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡) + ∇𝝃𝑝
′ (𝐱, 𝑡𝑛+ 𝑐𝑚Δ𝑡))
(73)
Thanks to a Taylor expansion, we get
̂𝜏(𝑛,𝑙)
𝜉
=
Δ𝑡
(
𝑐𝑙
𝜕̄𝜌̄𝐮
𝜕𝑡(𝐱, 𝑡𝑛) +
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇𝐱⋅( ̄𝜌(𝐱, 𝑡𝑛) ̄𝐮(𝐱, 𝑡𝑛) ⊗̄𝐮(𝐱, 𝑡𝑛)) +
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚
(
∇𝐱𝑝
′′ (𝐱, 𝑡𝑛) + ∇𝝃𝑝
′ (𝐱, 𝑡𝑛)
))
+
(Δ𝑡2).
(74)
Since
𝑙∑
𝑚=1
̃𝑎𝑙𝑚=
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚= 𝑐𝑙(53), we obtain
̂𝜏(𝑛,𝑙)
𝜉
=
𝑐𝑙Δ𝑡
(𝜕̄𝜌̄𝐮
𝜕𝑡(𝐱, 𝑡𝑛) + ∇𝐱⋅( ̄𝜌(𝐱, 𝑡𝑛) ̄𝐮(𝐱, 𝑡𝑛) ⊗̄𝐮(𝐱, 𝑡𝑛)) +
(
∇𝐱𝑝
′′ (𝐱, 𝑡𝑛) + ∇𝝃𝑝
′ (𝐱, 𝑡𝑛)
))
+
(Δ𝑡2) = (Δ𝑡2).
(75)
Finally, for the first order term in the energy equation, we obtain
𝜌
′,(𝑛,𝑙)𝑒
′,(𝑛,𝑙) + ̃𝑎𝑙𝑙Δ𝑡∇𝐱⋅
(
ℎ
′,(𝑛,𝑙)𝜌
′,(𝑛,𝑙)𝐮
′,(𝑛,𝑙))
+ ̃𝑎𝑙𝑙Δ𝑡∇𝝃⋅(̄ℎ(𝑛,𝑙) ̄𝜌(𝑛,𝑙)̄𝐮(𝑛,𝑙)) =
̄𝜌𝑛̄𝑒𝑛−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇𝐱⋅
(
ℎ
′,(𝑛,𝑚)𝜌
′,(𝑛,𝑚)𝐮
′,(𝑛,𝑚))
−
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚Δ𝑡∇𝝃⋅(̄ℎ(𝑛,𝑚) ̄𝜌(𝑛,𝑚)̄𝐮(𝑛,𝑚)) .
(76)
Analogously, one can show that (76) is a consistent discretization of (41). Similar computations show the consistency
for the final update stage. Hence, (54) provides an AP scheme for (43).
4. Spatial discretization
In this Section, we briefly outline the spatial discretization for (54), which is based on the Discontinuous Galerkin
(DG) method [41] as implemented in the deal.II library [2, 5]. We use quadrilateral elements and the corresponding
polynomial spaces 𝑄𝑟of degree 𝑟[77]. More specifically, the shape functions correspond to the products of Lagrange
polynomials for the support points of (𝑟+ 1)-order Gauss-Lobatto quadrature rule in each coordinate direction, where
𝑟is the polynomial degree. However, the proposed approach can also be applied to tetrahedral meshes and 𝑃-spaces.
We consider a decomposition of the domain Ω into a family of quadrilaterals and denote each element by 𝐾. We
denote by the set of all the element faces, so that = 𝐼∪𝐵, with 𝐼and 𝐵denoting the subset of interior
and boundary faces, respectively. A face Γ ∈𝐼shares two elements, 𝐾+ with outward unit normal 𝐧+ and 𝐾−with
outward unit normal 𝐧−. Finally, we denote by 𝐧the outward unit normal for a face Γ ∈𝐵. Hence, following, e.g.,
[3], for a scalar function 𝜑, we define the jump as
[[𝜑]] = 𝜑+𝐧+ + 𝜑−𝐧−if Γ ∈𝐼
[[𝜑]] = 𝜑𝐧if Γ ∈𝐵,
(77)
where we define the average as
{{𝜑}} = 1
2
(𝜑+ + 𝜑−) if Γ ∈𝐼
{{𝜑}} = 𝜑if Γ ∈𝐵.
(78)
Analogous definitions apply for a vector function 𝝋. More specifically, we define
[[𝝋]] = 𝝋+ ⋅𝐧+ + 𝝋−⋅𝐧−if Γ ∈𝐼
[[𝝋]] = 𝝋⋅𝐧if Γ ∈𝐵
(79)
G. Orlando et al.: Preprint submitted to Elsevier
Page 15 of 38


AP IMEX schemes for Euler equations: non-ideal gases
{{𝝋}} = 1
2
(𝝋+ + 𝝋−) if Γ ∈𝐼
{{𝝋}} = 𝝋if Γ ∈𝐵.
(80)
For vector functions, it is also useful to define a tensor jump as follows:
⟨⟨𝝋⟩⟩= 𝝋+ ⊗𝐧+ + 𝝋−⊗𝐧−if Γ ∈𝐼
⟨⟨𝝋⟩⟩= 𝝋⊗𝐧if Γ ∈𝐵.
(81)
Given these definitions, the weak formulation for the momentum equation at each stage (54) reads as follows [71, 72]:
𝐀(𝑛,𝑙)𝐔(𝑛,𝑙) + 𝐁(𝑛,𝑙)𝐏(𝑛,𝑙) = 𝐅(𝑛,𝑙),
(82)
with 𝐔(𝑛,𝑙) denoting the vector of the degrees of freedom associated to the velocity field and 𝐏(𝑛,𝑙) denoting the vector
of the degrees of freedom associated to the pressure. Here we have set
𝐴(𝑛,𝑙)
𝑖𝑗
=
∑
𝐾∈∫𝐾
𝜌(𝑛,𝑙)𝝋𝑗⋅𝝋𝑖𝑑Ω
(83)
𝐵(𝑛,𝑙)
𝑖𝑗
=
∑
𝐾∈∫𝐾
−̃𝑎𝑙𝑙
Δ𝑡
𝑀2 ∇⋅𝝋𝑖Ψ𝑗𝑑Ω +
∑
Γ∈∫Γ
̃𝑎𝑙𝑙
Δ𝑡
𝑀2
{{Ψ𝑗
}} [[𝝋𝑖
]] 𝑑Σ
(84)
𝐹(𝑛,𝑙)
𝑖
=
∑
𝐾∈∫𝐾
𝜌𝑛𝐮𝑛⋅𝝋𝑖𝑑Ω +
𝑙−1
∑
𝑚=1
∑
𝐾∈∫𝐾
𝑎𝑙𝑚Δ𝑡(𝜌(𝑛,𝑚)𝐮(𝑛,𝑚) ⊗𝐮(𝑛,𝑚)) ∶∇𝝋𝑖𝑑Ω
+
𝑙−1
∑
𝑚=1
∑
𝐾∈∫𝐾
̃𝑎𝑙𝑚
Δ𝑡
𝑀2 𝑝(𝑛,𝑚) ∇⋅𝝋𝑖𝑑Ω
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
𝑎𝑙𝑚Δ𝑡{{𝜌(𝑛,𝑚)𝐮(𝑛,𝑚) ⊗𝐮(𝑛,𝑚)}} ∶⟨⟨𝝋𝑖⟩⟩𝑑Σ
(85)
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
𝑎𝑙𝑚Δ𝑡𝜆(𝑛,𝑚)
2
⟨⟨𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)⟩⟩∶⟨⟨𝝋𝑖⟩⟩𝑑Σ −
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
̃𝑎𝑙𝑚
Δ𝑡
𝑀2
{{𝑝(𝑛,𝑚)}} [[𝝋𝑖
]] 𝑑Σ,
with 𝝋𝑖and Ψ𝑖denoting the basis function of the space of polynomial functions employed to discretize the velocity
and the pressure, respectively. Following the discussion in [70, 71], one can notice that we employ a centered flux
for the quantities defined implicitly and upwind-biased flux for the quantities computed explicitly. The choice of the
upwind-biased flux influences the numerical dissipation. Ideally, a flux appropriate for all Mach numbers should be
used, as done e.g. in [76]. In order to obtain a numerical method effective for a wide range of Mach numbers, we take
𝜆(𝑛,𝑚) = max
[
𝑓
(
𝑀+,(𝑛,𝑚)
𝑙𝑜𝑐
) (|||𝐮+,(𝑛,𝑚)||| + 1
𝑀𝑐+,(𝑛,𝑚))
, 𝑓
(
𝑀−,(𝑛,𝑚)
𝑙𝑜𝑐
) (|||𝐮−,(𝑛,𝑚)||| + 1
𝑀𝑐−,(𝑛,𝑚))]
,
(86)
with 𝑀±,(𝑛,𝑚)
𝑙𝑜𝑐
= 𝑀|𝐮|±,(𝑛,𝑚)
𝑐±,(𝑛,𝑚)
and 𝑓(𝑀𝑙𝑜𝑐
) = min (1, 𝑀𝑙𝑜𝑐
). This choice corresponds to the convex combination between
a centered flux and a Rusanov flux [79] discussed in [1]. More specifically, for a generic flux ̂𝐅, we employ
̂𝐅= (1 −𝑓(𝑀𝑙𝑜𝑐
)) 𝐅𝑐+ 𝑓(𝑀𝑙𝑜𝑐
) 𝐅𝑅,
(87)
with 𝐅𝑐and 𝐅𝑅denoting the centered flux and the Rusanov flux, respectively. Hence, for 𝑀𝑙𝑜𝑐≈1, we obtain the
Rusanov flux, whereas for 𝑀𝑙𝑜𝑐≪1, we obtain a local Lax-Friedrichs flux. Analogously, the energy equation in (54)
can be expressed as
𝐂(𝑛,𝑙)𝐔(𝑛,𝑙) + 𝐃(𝑛,𝑙)𝐏(𝑛,𝑙) = 𝐆(𝑛,𝑙),
(88)
with
𝐶(𝑛,𝑙)
𝑖𝑗
=
∑
𝐾∈∫𝐾
−̃𝑎𝑙𝑙Δ𝑡ℎ(𝑛,𝑙)𝜌(𝑛,𝑙)𝝋𝒋⋅∇Ψ𝑖𝑑Ω +
∑
Γ∈∫Γ
̃𝑎𝑙𝑙Δ𝑡{{ℎ(𝑛,𝑙)𝜌(𝑛,𝑙)𝝋𝒋
}} ⋅[[Ψ𝑖
]] 𝑑Σ
(89)
G. Orlando et al.: Preprint submitted to Elsevier
Page 16 of 38


AP IMEX schemes for Euler equations: non-ideal gases
𝐷(𝑛,𝑙)
𝑖𝑗
=
∑
𝐾∈∫𝐾
𝜌(𝑛,𝑙)𝑒(𝑛,𝑙)(𝜌(𝑛,𝑙), Ψ𝑗)Ψ𝑖𝑑Ω
(90)
𝐺(𝑛,𝑙)
𝑖
=
𝑙−1
∑
𝑚=1
∑
𝐾∈∫𝐾
𝜌(𝑛,𝑙)𝐸(𝑛,𝑙)𝜓𝑖𝑑Ω
+
𝑙−1
∑
𝑚=1
∑
𝐾∈∫𝐾
𝑎𝑙𝑚Δ𝑡𝑀2 (𝑘(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)) ⋅∇Ψ𝑖𝑑Ω
+
𝑙−1
∑
𝑚=1
∑
𝐾∈∫𝐾
̃𝑎𝑙𝑚Δ𝑡(ℎ(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)) ⋅∇Ψ𝑖𝑑Ω
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
𝑎𝑙𝑚Δ𝑡𝑀2 {{𝑘(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)}} ⋅[[Ψ𝑖
]] 𝑑Σ
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
̃𝑎𝑙𝑚Δ𝑡{{ℎ(𝑛,𝑚)𝜌(𝑛,𝑚)𝐮(𝑛,𝑚)}} ⋅[[Ψ𝑖
]] 𝑑Σ
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
𝑎𝑙𝑚Δ𝑡𝑀2 𝜆(𝑛,𝑚)
2
[[𝜌(𝑛,𝑚)𝑘(𝑛,𝑚)]] ⋅[[Ψ𝑖
]] 𝑑Σ
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
̃𝑎𝑙𝑚Δ𝑡𝜆(𝑛,𝑚)
2
[[𝜌(𝑛,𝑚)𝑒(𝑛,𝑚)]] ⋅[[Ψ𝑖
]] 𝑑Σ
−
∑
𝐾∈ℎ∫𝐾
𝑀2𝜌(𝑛,𝑙)𝑘(𝑛,𝑙)Ψ𝑖𝑑Ω −
∑
Γ∈∫Γ
̃𝑎𝑙𝑙Δ𝑡𝜆(𝑛,𝑙)
2
[[𝜌(𝑛,𝑙)𝑒(𝑛,𝑙)]] ⋅[[Ψ𝑖
]] 𝑑Σ.
(91)
Notice that, the upwind flux has been slightly modified with respect to the one employed in [71], so as to guarantee
the preservation of uniform velocity and pressure fields (see the discussion in [70]). Formally, one can derive
𝐔(𝑛,𝑙) = (𝐀(𝑛,𝑙))−1 (𝐅(𝑛,𝑙) −𝐁(𝑛,𝑙)𝐏(𝑛,𝑙)) ,
(92)
so as to obtain
𝐃(𝑛,𝑙)𝐏(𝑛,𝑙) + 𝐂(𝑛,𝑙) (𝐀(𝑛,𝑙))−1 (𝐅(𝑛,𝑙) −𝐁(𝑛,𝑙)𝐏(𝑛,𝑙)) = 𝐆(𝑛,𝑙).
(93)
The above system can be solved following the fixed point procedure described in [35, 71]. More specifically, setting
𝐏(𝑛,𝑙,0) = 𝐏(𝑛,𝑙−1), one solves for 𝐿= 0, … , ̃𝐿
(
𝐃(𝑛,𝑙,𝐿) −𝐂(𝑛,𝑙,𝐿) (𝐀(𝑛,𝑙))−1 𝐁(𝑛,𝑙))
𝐏(𝑛,𝑙,𝐿+1) = 𝐆(𝑛,𝑙,𝐿) −𝐂(𝑛,𝑙,𝐿) (𝐀(𝑛,𝑙))−1 𝐅(𝑛,𝑙,𝐿)
(94)
and then updates the velocity solving
𝐀(𝑛,𝑙)𝐔(𝑛,𝑙,𝐿) = 𝐅(𝑛,𝑙,𝐿) −𝐁(𝑛,𝑙)𝐏(𝑛,𝑙,𝐿+1).
(95)
Notice that, as discussed for the time discretization in Section 3, the employed spatial discretization is not TVD for
𝑟> 0. A discussion of possible approaches to overcome this issue is out of the scope of the work. However, a number
of approaches have been proposed to obtain essentially monotone schemes using high order DG methods, see e.g.
[36, 69].
The DG method naturally allows for high-order accuracy. However, as discussed in [56], its accuracy in the very
low Mach regime depends on the numerical flux and on the shape of the elements. More specifically, a mesh of
triangular/tetrahedral elements is needed to guarantee accuracy at all Mach numbers. A low Mach number fix for
the Euler equations resolved employing the finite volume method on Cartesian grids was presented in [6]. We will
further discuss this point in Section 5.1.
G. Orlando et al.: Preprint submitted to Elsevier
Page 17 of 38


AP IMEX schemes for Euler equations: non-ideal gases
5. Numerical results
The analysis outlined in Sections 3 and 4 is now validated in a number of benchmarks covering the 𝑀< 1 and
𝑀≪1 regimes. The implementation is carried out in the framework of the deal.II library [2, 5]. We use a time
discretization based on the third order IMEX scheme presented in [57], for which the coefficients of both the explicit
and implicit methods are reported in the Butcher tableaux Table 1. Moreover, in order to assess the convergence
properties of the method and to exploit the high-order accuracy provided by the DG method, we also consider in
Section 5.1 the fourth order time discretization scheme proposed in [21], for which the coefficients of the explicit and
of the implicit companion method are reported in Table 2. Notice that the implicit method of the fourth order time
discretization scheme is of type ARS. One can easily check that the implicit companion methods of both schemes are
stiffly-accurate. Hence, the implicit scheme in Table 2 is L-stable. For what concerns the implicit scheme in Table 1,
relation (69) leads to 𝑅∞= 0 and therefore it is also L-stable.
0
0
0
0
0
1767732205903
2027836641118
1767732205903
2027836641118
0
0
0
3
5
5535828885825
10492691773637
788022342437
10882634858940
0
0
1
6485989280629
16251701735622
−4246266847089
9704473918619
10755448449292
10357097424841
0
1471266399579
7840856788654
−4482444167858
7529755066697
11266239266428
11593286722821
1767732205903
4055673282236
0
0
0
0
0
1767732205903
2027836641118
1767732205903
4055673282236
1767732205903
4055673282236
0
0
3
5
2746238789719
10658868560708
−640167445237
6845629431997
1767732205903
4055673282236
0
1
1471266399579
7840856788654
−4482444167858
7529755066697
11266239266428
11593286722821
1767732205903
4055673282236
1471266399579
7840856788654
−4482444167858
7529755066697
11266239266428
11593286722821
1767732205903
4055673282236
Table 1
Butcher tableaux of the third order time discretization scheme. Top: explicit method. Bottom: implicit method.
0
0
0
0
0
0
0
1
4
1
4
0
0
0
0
0
3
4
−1
4
1
0
0
0
0
11
20
−13
100
43
75
8
75
0
0
0
1
2
−6
85
42
85
179
1360
−15
272
0
0
1
0
79
24
−5
8
25
2
−85
6
0
0
25
24
−49
48
125
16
−85
12
1
4
0
0
0
0
0
0
0
1
4
0
1
4
0
0
0
0
3
4
0
1
2
1
4
0
0
0
11
20
0
17
50
−1
25
1
4
0
0
1
2
0
371
1360
−137
2720
15
544
1
4
0
1
0
25
24
−49
48
125
16
−85
12
1
4
0
25
24
−49
48
125
16
−85
12
1
4
Table 2
Butcher tableaux of the fourth order time discretization scheme. Top: explicit method. Bottom: implicit method.
G. Orlando et al.: Preprint submitted to Elsevier
Page 18 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Notice that both the schemes are of type II. Results employing numerical schemes of type I can be found in [73].
We set = min {diam (𝐾) |𝐾∈
} and we define two Courant numbers, one based on the speed of sound (acoustic
Courant number), denoted by 𝐶, and one based on the local velocity of the flow (advective Courant number), denoted
by 𝐶𝑢:
𝐶= 1
𝑀𝑟𝑐Δ𝑡

√
𝑑
𝐶𝑢= 𝑟𝑢Δ𝑡

√
𝑑,
(96)
where 𝑐is the speed of sound and 𝑢is the magnitude of the flow velocity. Recall that 𝑟denotes the polynomial degree
of the space discretization. For the tests using the ideal gas law (8), the value 𝛾= 1.4 is employed.
5.1. Isentropic vortex
First, we consider the isentropic vortex benchmark studied in [14, 92, 93] using the ideal gas law (8), for which an
analytical solution is available. Following, e.g., [14, 94], the steady solution of system (7) as a function of the Mach
number reads as follows:
𝐮(𝐱, 𝑡)
=
𝐮(𝐱, 0) = 𝑀𝛽
2𝜋exp
(
1 −̃𝑟2
2
) (
−(𝑦−𝑦0)
𝑥−𝑥0
)
(97a)
𝜌(𝐱, 𝑡)
=
𝜌(𝐱, 0) = (1 + 𝛿𝑇)
1
𝛾−1
(97b)
𝑝(𝐱, 𝑡)
=
𝑝(𝐱, 0) = 𝑀2 (1 + 𝛿𝑇)
𝛾
𝛾−1 = 𝑀2𝜌𝛾,
(97c)
where ̃𝑟2 = (𝑥−𝑥0
)2 + (𝑦−𝑦0
)2, with 𝑥0 and 𝑦0 denoting the coordinates of the center of the vortex. Moreover, 𝛽is
the vortex strength and the temperature perturbation 𝛿𝑇is given by
𝛿𝑇= −𝑀2 𝛾−1
𝛾
𝛽2
8𝜋2 exp (1 −̃𝑟2) .
(98)
A travelling vortex configuration can be found instead in [73], to which we refer also for the impact of different time
discretization strategies. To avoid problems with the definition of the boundary conditions, we choose a sufficiently
large domain Ω = (−10, 10)2, with 𝑥0 = 𝑦0 = 0, and periodic boundary conditions. Finally, we set 𝛽= 5 and
𝑇𝑓= 10. The purpose of this test is twofold. First, we assess the convergence properties of the method, employing
the IMEX schemes in Table 1 and Table 2. Next, we verify the asymptotic expansion in the small Mach number limit
outlined in Section 2.2. We set 𝑀= 10−3 for the convergence analysis, which is performed at fixed acoustic Courant
number 𝐶≈3.5. We report results for the case of polynomial degree 𝑟= 2 in combination with the third order time
discretization scheme (Table 3) and for the case of polynomial degree 𝑟= 3 in combination with the fourth order
time discretization scheme (Table 4). The expected convergence rate is achieved for the third order method, whereas
an order reduction is experienced for the fourth order method as the resolution increases and the time step decreases.
Since the solution is steady, this is likely related to an early manifestation of a low Mach inaccuracy (see the discussion
below).
Next, we analyze the behaviour in the 𝑀→0 limit. We employ the third order time discretization scheme in Table
1 with 𝑟= 2 and 𝑁𝑒𝑙= 120 elements along each coordinate direction. Until 𝑀= 10−5, the density fluctuations scale
as (𝑀2) and the divergence of the velocity field scales as (𝑀) (Table 5), as expected [56]. The convergence with
respect to 𝑀of the divergence of the velocity field deserves some comments. Since ∇⋅̄𝐮= 0, the first order term of
the energy equation for a steady state solution reduces to ̄𝑝∇⋅𝐮
′ = 0 (see (41)). The initial velocity field is indeed
solenoidal and a quadratic convergence with respect to 𝑀could be therefore expected. However, the divergence-free
property is not imposed pointwise and, since a term proportional to 𝑀is present in the velocity field, we obtain a linear
scaling with respect to 𝑀for ∇⋅𝐮(Table 5). A quadratic convergence was obtained recently in [93] for the Taylor-Green
vortex. This is likely related to the use of compatible finite elements which allow the imposition of the divergence-free
property for the initial datum, so as to observe the quadratic convergence rate predicted by the asymptotic expansion
of the continuous model. Indeed, in our framework, the error associated to ∇⋅𝐮is basically constant in time and it is
therefore related to the interpolation of the initial datum into the employed finite element space.
At 𝑀= 10−6, we observe a small degradation for the density fluctuations. This is likely related to the well-known
inf-sup stability condition for DG discretizations of incompressible flows [88]. Indeed, we have verified that a slightly
improved scaling of the density fluctuations is obtained employing a polynomial degree 𝑟+ 1 for the velocity field,
G. Orlando et al.: Preprint submitted to Elsevier
Page 19 of 38


AP IMEX schemes for Euler equations: non-ideal gases
i.e. third order polynomials (Table 5). Moreover, it is worth to remark that at 𝑀= 10−6, the density and the pressure
are basically constant and the 𝐿2 error for the pressure is below the machine precision, so that round-off errors play a
relevant role. Indeed, as remarked in [13], the use of quadruple precision is crucial to maintain the theoretical scaling
at very small Mach numbers.
A similar behaviour is experienced employing the fourth order time discretization scheme in Table 2 with 𝑟= 3
and 𝑁𝑒𝑙= 80. Here we notice that the density fluctuations start scaling as (𝑀) from 𝑀= 10−4 (Table 6). The loss
of low Mach accuracy from 𝑀= 10−4 is also related to well-known order reduction phenomenon experienced for
very stiff problems using high-order time discretization methods [58, 90] (see also [73]). Indeed, we have verified that,
using the third order time discretization scheme with polynomial degree 𝑟= 3, the correct scaling is established up to
𝑀= 10−5. The use of polynomial degree 𝑟+ 1 = 4 for the velocity field allows us to recover the correct scaling up to
𝑀= 10−5 (Table 6). At 𝑀= 10−6, for which round-off errors play a major role, a degradation is still experienced.
It has to be recalled that well known issues arise using quadrilateral elements for strongly subsonic flows. The
seminal work of Guillard and Viozat [45] showed through an asymptotic analysis of the first order Roe scheme that
a pressure term of order (𝑀) appears on Cartesian meshes as 𝑀→0. A number of fixes for numerical fluxes
that preserve contact discontinuities (HLLC, Roe, etc...) have been proposed in the literature [32, 33, 38, 78]. They
have been developed in the framework of the Finite Volume method, but they can in principle be extended to the
DG method. The high-order accuracy of the DG method typically counterbalances the lack of low Mach accuracy for
strongly subsonic flows. A loss of accuracy in this limit was already reported in [7]. In the recent work of Jung and
Perrier [56], the authors show that the DG method employing numerical fluxes that preserve contact discontinuities
is low Mach number accurate using triangular elements, while the same does not hold for quadrilateral elements.
Moreover, as reported in [55], low Mach fixes are similar to schemes based on specific IMEX time discretizations.
As an example, the fix proposed in [33] imposes a zero velocity jump in the artificial viscosity term, so as to obtain a
centering of the pressure gradient in the momentum equation. The method presented in this work uses a centered flux
for the quantities defined implicitly, including the pressure gradient (see Section 4). Since the focus of this work is to
show the AP property of a general class of IMEX time discretizations, we do not investigate further these issues, that
are related to the spatial discretization. It can be observed, however, that the high-order accuracy of the DG method
allows to simulate correctly flows down to 𝑀= 10−4 −10−5. This is lower than the typical values of the Mach number
for fluids like water, that are modelled as incompressible in most realistic applications. Moreover, in [87] (see also
[53]), the authors show that inaccuracies of standard Godunov schemes at low Mach number are linked to spurious
entropy generation. Entropy-stable DG schemes, as those developed, e.g., in [39, 91], could therefore improve the low
Mach accuracy. The use of entropy-stable DG methods and of exterior calculus and compatible finite element to further
improve the low Mach accuracy of the spatial discretization will be the the focus of future work.
𝑁𝑒𝑙
𝐿2 rel. error 𝐮
𝐿2 rate 𝐮
𝐿2 rel. error 𝜌
𝐿2 rate 𝜌
𝐿2 rel. error 𝑝
𝐿2 rate 𝑝
15
4.70 × 10−2
9.91 × 10−8
1.31 × 10−7
30
5.06 × 10−3
3.2
1.77 × 10−9
5.8
2.27 × 10−9
5.9
60
6.42 × 10−4
3.0
8.18 × 10−11
4.4
3.51 × 10−11
6.0
120
8.07 × 10−5
3.0
1.07 × 10−11
2.9
2.03 × 10−12
4.1
240
1.02 × 10−5
3.0
1.92 × 10−12
2.5
2.46 × 10−13
3.0
Table 3
Convergence analysis for the isentropic vortex test case using the time discretization scheme in Table 1 together with
polynomial degree 𝑟= 2. Here, 𝑁𝑒𝑙denotes the number of elements along each direction.
G. Orlando et al.: Preprint submitted to Elsevier
Page 20 of 38


AP IMEX schemes for Euler equations: non-ideal gases
𝑁𝑒𝑙
𝐿2 rel. error 𝐮
𝐿2 rate 𝐮
𝐿2 rel. error 𝜌
𝐿2 rate 𝜌
𝐿2 rel. error 𝑝
𝐿2 rate 𝑝
10
3.27 × 10−2
8.67 × 10−7
1.21 × 10−6
20
2.18 × 10−3
3.9
2.97 × 10−8
4.9
4.15 × 10−8
4.9
40
1.47 × 10−4
3.9
2.58 × 10−9
3.5
3.62 × 10−9
3.5
80
1.32 × 10−5
3.5
4.76 × 10−10
2.4
6.66 × 10−10
2.4
160
1.82 × 10−6
2.9
6.20 × 10−11
2.9
8.68 × 10−11
2.9
Table 4
Convergence analysis for the isentropic vortex test case using the time discretization scheme in Table 2 together with
polynomial degree 𝑟= 3. Here, 𝑁𝑒𝑙denotes the number of elements along each direction.
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm ∇𝜌
Rate ∇𝜌
𝐿2
norm
∇𝜌(𝑄3 −𝑄2)
Rate
∇𝜌(𝑄3 −𝑄2)
10−1
3.52 × 10−4
1.09 × 10−2
10−2
3.46 × 10−5
1.0
1.09 × 10−4
2.0
10−3
3.44 × 10−6
1.0
1.09 × 10−6
2.0
10−4
3.44 × 10−7
1.0
1.10 × 10−8
2.0
10−5
3.44 × 10−8
1.0
1.29 × 10−10
1.9
10−6
3.47 × 10−9
1.0
9.66 × 10−12
1.1
8.34 × 10−12
1.2
Table 5
Mach number scaling of the density fluctuations and of the divergence of the velocity field for the isentropic vortex test
case. The results are obtained using the third order time discretization scheme in Table 1 together with polynomial degree
𝑟= 2 and 𝑁𝑒𝑙= 120. The last two columns report the results obtained using polynomial degree 𝑟+1 = 3 for the velocity and
polynomial degree 𝑟= 2 for the other variables. We recall that 𝑄𝑟denotes polynomial spaces of degree 𝑟for quadrilateral
elements.
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm ∇𝜌
Rate ∇𝜌
𝐿2
norm
∇𝜌(𝑄4 −𝑄3)
Rate
∇𝜌(𝑄4 −𝑄3)
10−1
1.12 × 10−4
1.09 × 10−2
10−2
1.21 × 10−5
1.0
1.09 × 10−4
2.0
10−3
1.25 × 10−6
1.0
1.19 × 10−6
2.0
10−4
1.26 × 10−7
1.0
5.05 × 10−8
1.4
1.09 × 10−8
2.0
10−5
1.26 × 10−8
1.0
4.95 × 10−9
1.0
1.10 × 10−10
2.0
10−6
1.27 × 10−9
1.0
4.97 × 10−10
1.0
1.53 × 10−11
0.9
Table 6
Mach number scaling of the density fluctuations and of the divergence of the velocity field for the isentropic vortex test
case. The results are obtained using the fourth order time discretization scheme in Table 2 together with polynomial degree
𝑟= 3 and 𝑁𝑒𝑙= 80. The last two columns report the results obtained using polynomial degree 𝑟+1 = 4 for the velocity and
polynomial degree 𝑟= 3 for the other variables. We recall that 𝑄𝑟denotes polynomial spaces of degree 𝑟for quadrilateral
elements.
5.2. Colliding acoustic pulses
This benchmark, proposed in [60], consists of two colliding acoustic pulses in the domain Ω = (−𝐿, 𝐿), namely, a
right-running pulse initially located in (−𝐿, 0) and a left-running pulse initially located in (0, 𝐿). Following [60], we
G. Orlando et al.: Preprint submitted to Elsevier
Page 21 of 38


AP IMEX schemes for Euler equations: non-ideal gases
set 𝑀=
1
11 and we define the half-length of the domain 𝐿=
2
𝑀= 22. Periodic boundary conditions are prescribed.
The initial conditions read as follows:
𝜌(𝑥, 0)
=
̄𝜌0 + 1
2𝑀𝜌
′
0
(
1 −cos
(2𝜋𝑥
𝐿
))
̄𝜌0 = 0.955
𝜌
′
0 = 2
(99a)
𝑢(𝑥, 0)
=
−1
2 sgn (𝑥) ̄𝑢0
(
1 −cos
(2𝜋𝑥
𝐿
))
̄𝑢0 = 2
√
𝛾
(99b)
𝑝(𝑥, 0)
=
̄𝑝0 + 1
2𝑀𝑝
′
0
(
1 −cos
(2𝜋𝑥
𝐿
))
̄𝑝0 = 1
𝑝
′
0 = 2𝛾
(99c)
The final time is 𝑇𝑓= 1.63. We consider a number of elements 𝑁𝑒𝑙= 55 with 𝑟= 2, i.e. polynomial degree of order 2,
whereas the time step is Δ𝑡= 1.63×10−2, leading to a maximum advective Courant number 𝐶𝑢≈0.1 and a maximum
acoustic Courant number 𝐶≈0.56. A reference solution is computed using the explicit third order strong stability
preserving (SSP) scheme described in [42], to which we refer for all the details. We employ 𝑁𝑒𝑙= 880 elements
with Δ𝑡= 2.54687 × 10−4, which corresponds to an acoustic Courant number 𝐶≈0.14. The pressure profiles at
𝑡=
𝑇𝑓
2 = 0.815 and 𝑡= 𝑇𝑓are in agreement with the reference results and with the results present in the literature
[29, 60, 68] (Figure 1). One can easily notice that at 𝑡=
𝑇𝑓
2 , the two pulses are superposed and the pressure reaches
its maximum value. At 𝑡= 𝑇𝑓, the pulses are separated from each other and assume almost the initial configuration.
However, as explained in [29, 60], weakly nonlinear acoustic effects start steepening the pulses and distort the final
profile, since shocks are beginning to form around 𝑥= ±18.5. We also compare the accuracy of the IMEX scheme
for increasing Courant numbers. More specifically, we consider Δ𝑡= 3.26 × 10−2 and Δ𝑡= 8.15 × 10−2, which lead
to 𝐶𝑢≈0.2, 𝐶≈1.16 and 𝐶𝑢≈0.5, 𝐶≈2.9, respectively. For larger time step, stability restrictions imposed by
the explicit component of the IMEX scheme arise [28]. Moreover, as we will discuss later on, for very large values of
the acoustic Courant number, the profile of the pulses is damped. One can easily notice that an excellent agreement
is established and we can correctly capture the acoustic pulses also at acoustic Courant number which are moderately
higher than 1 (Figure 2). Small differences arise at 𝑡= 𝑇𝑓, where the pulses start steepening and shocks are beginning
to form.
Finally, we employ the SG-EOS (9). We take 𝛾= 4.4, 𝑞∞= 0, and we consider two different values of 𝜋∞, namely
𝜋∞= 6.8×10−3 and 𝜋∞= 6.8×103. Notice that we do not modify the initial conditions, namely we keep ̃𝑢0 = 2
√
1.4
and 𝑝
′
0 = 2 ⋅1.4 = 2.8. First, we analyze the results with 𝜋∞= 6.8 × 10−3. A reference solution is computed using the
third order SSP scheme with Δ𝑡= 1.27344 × 10−4 and 𝑁𝑒𝑙= 880 elements, leading to a maximum acoustic Courant
number 𝐶≈0.13 and a maximum advective Courant number 𝐶≈0.01. The time step employed for the IMEX scheme
is not modified, yielding a maximum acoustic Courant number 𝐶≈1.09 and a maximum advective Courant number
𝐶≈0.09. The pulses collide at 𝑡=
𝑇𝑓
5 = 0.326 and an excellent agreement between the results obtained using the
IMEX method and the reference ones is established (Figure 3). At 𝑡= 𝑇𝑓, shocks form around 𝑥= ±7 and spurious
oscillations due to the high-order discretization methods arise.
Next, we consider the configuration with 𝜋∞= 6.8 × 103. A reference solution is computed using the third order
SSP scheme with Δ𝑡= 1.59179 × 10−6 and 𝑁𝑒𝑙= 880 elements, yielding a maximum acoustic Courant number
𝐶≈0.13 and a maximum advective Courant number 𝐶≈0.01. For what concerns the IMEX method, a stable
solution can be obtained without modifying the time step, but the pressure profiles are completely damped (Figure 4).
This is related to the fact the maximum acoustic Courant number is 𝐶≈80 and therefore we can no longer capture the
acoustic wave. In order to correctly resolve the acoustic pulse, we have to decrease the value of the acoustic Courant
number. We consider therefore Δ𝑡= 2.0375×10−4, namely a time step 80 times smaller than the previous one, so as to
obtain 𝐶≈1. One can easily notice that a good agreement is established with the reference solution and the pressure
profile is no longer damped, with the pulses colliding around 𝑡= 3
10𝑇𝑓(Figure 4). Moreover, no spurious oscillations
arise. While the primary goal in the use of IMEX schemes is to obtain a method capable to resolve the material waves
filtering out the acoustic waves, on the other hand, if a sufficiently small time step is employed, the method seems to be
naturally able to deal with low Mach acoustics. Notice that this is not valid in general for other low Mach fixes. In [17],
for example, the authors show that the low Mach fixes proposed in [31] and [78] can suffer of spurious oscillations
and of order reduction when applied to low Mach acoustics. A correction able to deal with low Mach acoustic was
developed in [17] and, more recently, in [38]. A more detailed analysis of the low Mach acoustic behaviour will be the
focus of future work.
G. Orlando et al.: Preprint submitted to Elsevier
Page 22 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 1: Colliding acoustic pulses test case, pressure profile. Left: 𝑡=
𝑇𝑓
2 . Right: 𝑡= 𝑇𝑓. The initial profile is in dashed
black line, the solid blue lines provide the results at the corresponding time obtained with the IMEX method at 𝐶𝑢≈0.1,
whereas the red dots show the reference results obtained with the explicit method.
Figure 2: Colliding acoustic pulses test case, pressure profile. Comparison of the IMEX method employing different time
step. Left: 𝑡=
𝑇𝑓
2 . Right: 𝑡= 𝑇𝑓. The solid blue lines provide the results at the obtained at 𝐶𝑢≈0.1, the black dots show
the results obtained at 𝐶𝑢≈0.2, whereas the red crosses report the results obtained at 𝐶𝑢≈0.5.
Figure 3: Colliding acoustic pulses test case employing the SG-EOS (9) with 𝑝∞= 6.8 × 10−3, pressure profile. Left: 𝑡=
𝑇𝑓
5 .
Right: 𝑡= 𝑇𝑓. The initial profile is in dashed black line, the solid blue lines provide the results at the corresponding time
obtained with the IMEX method, whereas the red dots show the reference results obtained with the explicit method.
5.3. Density layering
We consider now the test case II proposed in [60] and described also, e.g., in [68]. The domain is Ω = (−𝐿, 𝐿),
with 𝐿=
1
0.02 = 50. The initial conditions read as follows:
G. Orlando et al.: Preprint submitted to Elsevier
Page 23 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 4: Colliding acoustic pulses test case employing the SG-EOS (9) with 𝑝∞= 6.8×103, pressure profile. Left: 𝑡=
3
10𝑇𝑓.
Right: 𝑡= 𝑇𝑓. The solid blue lines provide the results at the corresponding time obtained with the IMEX method at acoustic
Courant number 𝐶≈80, the solid black lines report the results obtained with the IMEX method at 𝐶≈1, whereas the
red dots show the reference results obtained with the explicit method.
𝜌(𝑥, 0)
=
̄𝜌0 + Φ (𝑥) ̃𝜌0 sin
(40𝜋𝑥
𝐿
)
+ 1
2𝑀𝜌1
(
1 + cos
(𝜋𝑥
𝐿
))
(100a)
𝑢(𝑥, 0)
=
1
2 ̃𝑢0
(
1 + cos
(𝜋𝑥
𝐿
))
(100b)
𝑝(𝑥, 0)
=
̄𝑝0 + 1
2𝑀𝑝1
(
1 + cos
(𝜋𝑥
𝐿
))
,
(100c)
with ̄𝜌0 = 1, ̃𝜌0 = 1
2, 𝜌1 = 2, ̃𝑢0 = 2√𝛾= 2
√
1.4, ̄𝑝0 = 1, and 𝑝1 = 2𝛾= 2.8. Finally, the function Φ (𝑥) is given by
Φ (𝑥) =
{ 1
2
(
1 −cos
(
5𝜋𝑥
𝐿
))
if 0 ≤𝑥≤2
5𝐿
0
otherwise.
(101)
The initial data describe a density layering of large amplitude and small wavelengths, which is driven by the motion of
a right-moving periodic acoustic wave with long wavelength. Periodic boundary conditions are prescribed. The final
time is 𝑇𝑓= 5.071. We consider a computational grid composed by 𝑁𝑒𝑙= 250 elements with 𝑟= 2, whereas the
time step is Δ𝑡= 1.6903 × 10−2. Following [68], we start considering 𝑀=
1
50. Hence, the advective Courant 𝐶𝑢
is around 0.2, while the acoustic Courant number 𝐶is around 7. A comparison between the initial and the final time
for both the density and the pressure displays a good agreement with the results presented in [60, 68] (Figure 5). One
can easily notice that the acoustic wave transports the density layer and the shape of the layer is undistorted. As in the
previous test case, due to weakly non-linear effects, the pulse starts steepening, leading to shock formation. A reference
solution has been computed using the explicit third order SSP scheme. The time step employed for the explicit scheme
is Δ𝑡= 5.071 × 10−4, namely around 33 times smaller than that used with the IMEX scheme. An excellent agreement
is established between the two solutions.
For the sake of completeness, we also consider a case even closer to the incompressible regime, taking 𝑀= 10−4,
which results in an acoustic Courant number 𝐶≈1400. The analytical solution of the leading order term of the limit
model (43) with initial conditions (100a)-(100c) is
̄𝜌= ̄𝜌0 + Φ (𝑥−̄𝑢(𝑡) 𝑡) ̃𝜌0 sin
(40𝜋(𝑥−̄𝑢(𝑡) 𝑡)
𝐿
)
̄𝑢= ̄𝑢(𝑡)
̄𝑝= ̄𝑝0,
(102)
̄𝑢(𝑡) being a function only of time. Since we are considering periodic boundary conditions, the integral over the domain
of 𝜌𝑢is constant and therefore the steady value of ̄𝑢is
̄𝑢=
∫Ω
(
̄𝜌0 + Φ (𝑥) ̃𝜌0 sin
(
40𝜋(𝑥)
𝐿
)) (
1
2 ̃𝑢0
(
1 + cos
(
𝜋𝑥
𝐿
)))
𝑑Ω
∫Ω
(
1
2 ̃𝑢0
(
1 + cos
(
𝜋𝑥
𝐿
)))
𝑑Ω
= ̃𝑢0
2 −164375 −32875
√
5
1320441408𝜋
̃𝜌0 ̃𝑢0
̄𝜌0
≈̃𝑢0
2 =
√
𝛾.
G. Orlando et al.: Preprint submitted to Elsevier
Page 24 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 5: Density layering test case at 𝑀= 0.02 with the ideal gas law (8). Left: density. Right: pressure. The dashed
black lines represent the initial condition, the continuous blue lines show the solution at the final time, whereas the red
dots report the solution obtained with the third order optimal explicit SSP scheme.
(103)
A comparison at 𝑡= 𝑇𝑓between the analytical solution as 𝑀→0 and the numerical results shows an excellent
agreement for both the density and the pressure profile (Figure 6). Notice that the initial velocity field is not divergence-
free, namely it is not well-prepared. However, the numerical method leads to the incompressible limit, as already
discussed in Section 2.2 (Figure 6). For further reference, we include the solution obtained employing the explicit
scheme with Δ𝑡= 2.5355 × 10−6, i.e. a time step around 6666 times smaller. While on the one hand, the use of
high-order discretization schemes reduces the numerical diffusion and allows for preserving the shape of the layer
also employing the explicit method, on the other hand, the incompressible limit is not achieved (Figure 6). This result
confirms the necessity to employ asymptotic-preserving methods as 𝑀→0 to obtain reliable results as well as to be
much more efficient.
We now consider a configuration of this test case for the SG-EOS (9). We take 𝛾= 4.4, 𝜋∞= 6.8 × 10−3, and
𝑞∞= 0 in (9). Notice that, we do not modify the initial conditions (100a)-(100c), namely we keep ̃𝑢0 = 2
√
1.4 and
𝑝1 = 2.8. We start considering 𝑀=
1
50 = 0.02, which yields an acoustic Courant number 𝐶≈12.4 and a maximum
advective Courant number 𝐶𝑢≈0.2. Figure 7 shows a comparison between the initial and the final time for both the
density and the pressure. A reference solution has been computed using the third order explicit SSP scheme, with
a time step Δ𝑡= 2.5355 × 10−4, namely a time step around 66 times smaller than that employed with the IMEX
scheme. One can easily notice that the density layer is transported without too much damping. Moreover, an excellent
agreement with the explicit solution is established. Finally, for what concerns the incompressible limit at 𝑀= 10−4,
since 𝜕̄𝑝0
𝜕𝑡= 0, all the equations of state lead to the same limit (see (31)). This is further confirmed by the density and
pressure profiles reported in Figure 8.
5.4. Flow in an open tube
We consider now the test case III proposed in [60] for an ideal gas, which we recall here for the convenience of
the reader. A flow in an open tube represented by the domain Ω = (0, 10) is analyzed; at the left-end a time dependent
density and velocity are prescribed, whereas at the right-end a time dependent outflow pressure with large amplitude
variation is imposed. This kind of boundary conditions is employed e.g. in the case of subsonic inflow and subsonic
outflow [22]. More specifically, the initial conditions read as follows:
(𝜌, 𝑢, 𝑝) (𝑥, 0) = (1, 1, 1) ,
(104)
while the boundary conditions are
𝜌(0, 𝑡) = 1 + 3
10 sin (4𝑡)
𝑢(0, 𝑡) = 1 + 1
2 sin (2𝑡)
𝑝(𝐿, 𝑡) = 1 + 1
4 sin (3𝑡) ,
(105)
with 𝐿= 10. The final time is 𝑇𝑓= 7.47. The Mach number is set to 𝑀= 10−4. We consider a number of elements
𝑁𝑒𝑙= 50 with 𝑟= 2, whereas the time step is Δ𝑡= 9.3375 × 10−4, leading to a maximum advective Courant
G. Orlando et al.: Preprint submitted to Elsevier
Page 25 of 38


AP IMEX schemes for Euler equations: non-ideal gases
a)
b)
c)
Figure 6: Density layering test case at 𝑀= 10−4 with the ideal gas law (8). a) density, b) pressure, c) velocity. The
continuous black lines represent the analytical solution of the limit model (43), the blue dots report the numerical results
obtained with the IMEX method, whereas the red crosses show the results achieved with the fully explicit scheme.
Figure 7: Density layering test case at 𝑀= 0.02 with the SG-EOS (9). Left: density. Right: pressure. The dashed black
lines represent the initial condition, the continuous blue lines show the solution at the final time, whereas the red dots
report the solution obtained with the third order optimal explicit SSP scheme.
number 𝐶𝑢≈0.07 and a maximum acoustic Courant number 𝐶≈155. The results at 𝑡= 𝑇𝑓are those expected by
the asymptotic analysis for both the density and velocity profile (Figure 9). The limit solution as 𝑀→0 has been
included for comparison in Figure 9. For an ideal gas, (30) reduces to
∇⋅̄𝐮= −1
𝛾̄𝑝
𝜕̄𝑝
𝜕𝑡= −1
𝛾̄𝑝
𝑑̄𝑝
𝑑𝑡,
(106)
since ∇̄𝑝= 𝟎. Hence, for 𝑀→0, in one space dimension, 𝜕̄𝐮
𝜕𝑥is a function only of time and, therefore, the velocity
is a linear function of space with a given time dependent slope and boundary value at 𝑥= 0. For what concerns the
G. Orlando et al.: Preprint submitted to Elsevier
Page 26 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 8: Density layering test case at 𝑀= 10−4 with the SG-EOS (9). Left: density. Right: pressure. The continuous black
lines represent the analytical solution of the limit model (43), while the blue dots report the numerical results.
density, we rewrite (23) as follows:
𝜕̄𝜌
𝜕𝑡+ ̄𝐮⋅∇̄𝜌+ ̄𝜌∇⋅̄𝐮= 𝜕̄𝜌
𝜕𝑡+ ̄𝐮⋅∇̄𝜌−1
𝛾
̄𝜌
̄𝑝
𝑑̄𝑝
𝑑𝑡= 0,
(107)
or, equivalently,
𝐷log ̄𝜌
𝐷𝑡
= 1
𝛾
𝑑log ̄𝑝
𝑑𝑡
,
(108)
with 𝐷
𝐷𝑡= 𝜕
𝜕𝑡+ 𝐮⋅∇denoting the Lagrangian derivative. Hence, as discussed in [60], the material elements undergo
a quasi-static adiabatic compression and expansion following the particle paths described by ̄𝑢. One can easily notice
from the density profile in Figure 9 that mass elements, after entering the domain at the left-end, are correctly
compressed and expanded.
Figure 9: Open tube test case with the ideal gas law (8), results at 𝑡= 𝑇𝑓= 7.47. Left: density. Right: velocity. The
continuous black line shows the leading order solution as 𝑀→0, whereas the blue dots report the numerical results.
We consider now an extension of this test case for the SG-EOS (9). Equation (30) reduces to
∇⋅̄𝐮= −
1
𝛾( ̄𝑝+ 𝜋∞
) 𝑑̄𝑝
𝑑𝑡.
(109)
Hence, the velocity is still a linear function of space with a different time dependent slope with respect to that of the
ideal gas. Analogous considerations hold for the continuity equation, which reduces to
𝐷log ̄𝜌
𝐷𝑡
=
1
𝛾( ̄𝑝+ 𝜋∞
) 𝑑̄𝑝
𝑑𝑡= 1
𝛾
𝑑log ( ̄𝑝+ 𝜋∞
)
𝑑𝑡
.
(110)
G. Orlando et al.: Preprint submitted to Elsevier
Page 27 of 38


AP IMEX schemes for Euler equations: non-ideal gases
We take 𝛾= 4.4, 𝜋∞= 6.8 × 103, and 𝑞∞= 0 in (9). The time step is not modified. Hence, the maximum
advective Courant number is 𝐶𝑢≈0.015, while the maximum acoustic Courant number is 𝐶≈19300. Notice that,
with this configuration, an explicit scheme would require a time step around 65000 times smaller to achieve a stable
solution, yielding therefore a computational cost orders of magnitude larger. A comparison at the final time between
the numerical results and the leading order solution for both the density and the velocity displays a good agreement
for both profiles (Figure 10). The leading order term solution as 𝑀→0 for the ideal gas with 𝛾= 1.4, i.e. the
previous configuration, has been included in Figure 10. One can easily notice a visible difference in the behaviour of
both density and velocity. In particular, considering the large value of 𝜋∞, the velocity field is almost constant (Figure
10). Hence, if large amplitude pressure variations are considered, the limit regime depends on the equation of state and
on its parameters and does not necessarily correspond to the incompressible Euler equations.
Figure 10: Open tube test case with the SG-EOS (9), results at 𝑡= 𝑇𝑓= 7.47. Left: density. Right: velocity. The continuous
black line shows the leading order solution as 𝑀→0, the blue dots report the numerical results, while the red line shows
the leading order solution as 𝑀→0 with the ideal gas employing 𝛾= 1.4.
Finally, we consider the Peng-Robinson EOS (14). The asymptotic analysis becomes much more involved. First of
all, notice that
𝜕̄𝜌̄𝑒
𝜕̄𝑝= 1 −̄𝜌𝑏
𝛾−1
𝜕̄ℎ
𝜕̄𝜌= −
𝛾
𝛾−1
̄𝑝
̄𝜌2 + 𝑔( ̄𝜌) ,
(111)
where we have set
𝑔( ̄𝜌) =
𝑎(1 −2 ̄𝜌𝑏)
(𝛾−1)
(1 −̄𝜌𝑏𝑟1
) (1 −̄𝜌𝑏𝑟2
) +
𝑎𝑏̄𝜌(1 −̄𝜌𝑏)
(𝑟1
(1 −̄𝜌𝑏𝑟2
) + 𝑟2
(1 −̄𝜌𝑏𝑟1
))
(𝛾−1)
(1 −̄𝜌𝑏𝑟1
)2 (1 −̄𝜌𝑏𝑟2
)2
+ 𝑎
𝑏
𝜕𝑈
𝜕̄𝜌.
(112)
Hence, (30) reduces to
(
−
𝛾
𝛾−1 ̄𝑝+ ̄𝜌2𝑔( ̄𝜌)
)
∇⋅̄𝐮= 1 −̄𝜌𝑏
𝛾−1
𝑑̄𝑝
𝑑𝑡,
(113)
or, equivalently, to
∇⋅̄𝐮= −
1 −̄𝜌𝑏
𝛾̄𝑝−(𝛾−1) ̄𝜌2𝑔( ̄𝜌)
𝑑̄𝑝
𝑑𝑡.
(114)
Notice that, ∇⋅̄𝐮is now a function of both space and time. Hence, in one space dimension, the velocity is no longer a
linear profile. The continuity equation (23) reads as follows:
𝐷log ̄𝜌
𝐷𝑡
= −∇⋅̄𝐮=
1 −̄𝜌𝑏
𝛾̄𝑝−(𝛾−1) ̄𝜌2𝑔( ̄𝜌)
𝑑̄𝑝
𝑑𝑡.
(115)
We take 𝛾= 1.4, 𝑎= 1, and 𝑏= 0.15. The time step is Δ𝑡= 9.3375 × 10−4, yielding a maximum advective Courant
number 𝐶𝑢≈0.07 and a maximum acoustic Courant number 𝐶≈150. A comparison at the final time between the
numerical results and the leading order solution for both the density and the velocity shows a good agreement for
both profiles (Figure 11). The results are similar to those obtained with the ideal gas. Weakly non-ideal gas effects are
present in particular between 𝑥= 4 and 𝑥= 6, namely in correspondence of the peak density.
G. Orlando et al.: Preprint submitted to Elsevier
Page 28 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 11: Open tube test case with the Peng-Robinson EOS (14), results at 𝑡= 𝑇𝑓= 7.47. Left: density. Right: velocity.
The continuous black line shows the leading order solution as 𝑀→0, whereas the blue dots report the numerical results.
5.5. Gresho vortex
In this Section, we perform simulations of the so-called Gresho vortex [44, 66], which is a stationary solution
of the incompressible Euler equations. The centrifugal force, indeed, is balanced by the gradient of the pressure. A
rotating vortex is positioned at the center (0.5, 0.5) of the computational domain Ω = (0, 1)2. The initial conditions for
dimensional variables read as follows:
𝜌(𝑥, 0)
=
1
𝑢(𝑥, 0) = −𝑢𝜑sin (𝜑)
𝑣(𝑥, 0) = 𝑢𝜑cos (𝜑)
(116a)
𝑝(𝑥, 0)
=
⎧
⎪
⎨
⎪⎩
𝑝0 + 25
2 𝜌𝑟2
if 0 ≤̃𝑟< 0.2
𝑝0 + 25
2 𝜌̃𝑟2 + 4𝜌(1 −5̃𝑟−log(0.2) + log(̃𝑟))
if 0.2 ≤̃𝑟< 0.4
𝑝0 −𝜌(2 −4 log(2))
if ̃𝑟≥0.4.
(116b)
Here, ̃𝑟=
√
(𝑥−0.5)2 + (𝑦−0.5)2, 𝜑= arctan
(
𝑦−0.5
𝑥−0.5
)
, 𝑝0 =
𝜌0𝑢2
𝜑,𝑚𝑎𝑥
𝛾𝑀2 , with 𝜌0 = 1 kg m−3 and 𝑢𝜑,𝑚𝑎𝑥= 1 m s−1 for
̃𝑟= 0.2. Finally, 𝑢𝜑is
𝑢𝜑=
⎧
⎪
⎨
⎪⎩
5̃𝑟
if 0 ≤̃𝑟< 0.2
2 −5̃𝑟
if 0.2 ≤̃𝑟< 0.4
0
if ̃𝑟≥0.4.
(117)
Notice that, as discussed in [47], the pressure 𝑝0 is chosen in such a way that the maximum value of |𝐮|
𝑐matches
𝑀, so as to consider low Mach effects. We transform the initial conditions in non-dimensional quantities by using
= 1 kg m−3, = 1 m, and = 1 m s−1. Periodic boundary conditions are imposed for all the boundaries. We
simulate the flow until 𝑇𝑓= 3, when three full rotations are completed. The computational grid is composed by
80 × 80 elements with polynomial degree 𝑟= 2, whereas the time step is Δ𝑡= 2 × 10−3, leading an advective Courant
number 𝐶𝑢≈0.32. We consider 𝑀= 10−3 and 𝑀= 10−4. Hence, the acoustic Courant number is 𝐶≈320 for
𝑀= 10−3 and 𝐶≈3200 for 𝑀= 10−4. A comparison of the local Mach number 𝑀𝑙𝑜𝑐= 𝑀|𝐮|
𝑐
at initial time and the
final time for the two tests shows that the numerical method accurately preserves the shape of the vortex (Figure 12).
We also monitor the behaviour over time of the kinetic energy, which should be conserved. Table 7 reports the total
kinetic energy relative to the initial one after each rotation. The kinetic energy is conserved and these results compare
very well with those presented in [1], [86] where a loss of about 1.5 percent of the initial kinetic energy occurs after
one rotation of the vortex. Analogous results are achieved for 𝑀= 10−4. Hence, the preservation of the kinetic energy
holds independently of the Mach number.
G. Orlando et al.: Preprint submitted to Elsevier
Page 29 of 38


AP IMEX schemes for Euler equations: non-ideal gases
𝑀
𝑡= 1
𝑡= 2
𝑡= 3
10−3
0.999981
0.999977
0.999968
10−4
0.999981
0.999977
0.999968
Table 7
Total kinetic energy relative to its initial value for different Mach numbers after each full rotation of the Gresho vortex
with the ideal gas law (8).
Figure 12: Gresho vortex test case with the ideal gas law (8), comparison of local Mach number 𝑀𝑙𝑜𝑐= 𝑀|𝐮|
𝑐. From bottom
to top: results at 𝑀= 10−4 and 𝑀= 10−3. From left to right: initial condition and results at 𝑡= 𝑇𝑓= 3, after three full
rotations.
We now adapt the standard Gresho vortex test case to a water flow. As discussed in [1], it suffices to modify 𝑝0 for
the SG-EOS (9) as follows:
𝑝0 =
𝜌0𝑢2
𝜑,𝑚𝑎𝑥
𝛾𝑀2
−𝜋∞,
(118)
with 𝑀= 10−4, 𝜌0 = 1000 kg m−3, 𝛾= 4.4, and 𝜋∞= 6.8 × 108 Pa. We also take 𝑞∞= 0 in (9). The initial
density is now 𝜌(𝑥, 0) = 𝜌0 and we employ = 1000 kg m−3 to compute the non-dimensional counter part of initial
conditions (116a)-(116b). A comparison of 𝑀𝑙𝑜𝑐between the initial and the final time shows that the shape of the
vortex is accurately preserved also for a fluid with parameters corresponding to those of water (Figure 13). Table 8
reports the total kinetic energy relative to the initial one after each rotation, from which we notice that the kinetic
energy is conserved.
G. Orlando et al.: Preprint submitted to Elsevier
Page 30 of 38


AP IMEX schemes for Euler equations: non-ideal gases
𝑀
𝑡= 1
𝑡= 2
𝑡= 3
10−4
0.999984
0.999981
0.999977
Table 8
Total kinetic energy relative to its initial value after each full rotation of the Gresho vortex with the SG-EOS (9).
Figure 13: Gresho vortex test case with SG-EOS (9), comparison of local Mach number 𝑀𝑙𝑜𝑐= 𝑀|𝐮|
𝑐. Left: initial condition.
Right: results at 𝑡= 𝑇𝑓= 3, after three full rotations.
𝑀
𝑡= 1
𝑡= 2
𝑡= 3
10−4
0.999981
0.999977
0.999968
Table 9
Total kinetic energy relative to its initial value after each full rotation of the Gresho vortex with the Peng-Robinson EOS
(14).
Finally, we consider a configuration of the Gresho vortex for the Peng-Robinson EOS (14). The new expression of
the background pressure 𝑝0 reads as follows:
𝑝0 =
[𝑢2
𝜑,𝑚𝑎𝑥
𝑀2
+ 𝑓(𝜌0
)
]
𝜌0
(1 −𝜌0𝑏)
𝛾
,
(119)
with
𝑓(𝜌0
) =
𝑎𝜌0
1 −𝜌0𝑏
⎛
⎜
⎜⎝
𝜕𝑈
𝜕𝜌0
𝑏(𝛾−1) +
1 −2𝜌0𝑏
(1 −𝜌0𝑏𝑟1
) (1 −𝜌0𝑏𝑟2
)
⎞
⎟
⎟⎠
+ 𝑎𝑏𝜌2
0
𝑟1
(1 −𝜌0𝑏𝑟2
) + 𝑟2
(1 −𝜌0𝑏𝑟1
)
(1 −𝜌0𝑏𝑟1
)2 (1 −𝜌0𝑏𝑟2
)2
.
(120)
We take 𝛾= 1.4, 𝜌0 = 1 kg m−3, 𝑎= 500 m5 s−2 kg−1, and 𝑏= 10−3 m3 kg−1. Finally, we consider 𝜌(𝑥, 0) = 1 kg m−3
and this value is also employed to compute non-dimensional quantities. Figure 14 shows a comparison of 𝑀𝑙𝑜𝑐between
the initial and the final time, while Table 9 reports the total kinetic energy relative to the initial one after each rotation,
from which we notice that the kinetic energy is conserved. The same considerations done for the ideal gas law (8) and
for the SG-EOS (9) are therefore valid also for this particularly challenging and complex equation of state.
5.6. Baroclinic vorticity generation problem
We now consider a test case proposed in [40] and discussed also in [61, 68], which consists of a right-going acoustic
wave crossing a density fluctuation in the vertical direction. This test case illustrates the nontrivial interaction between
large-scale acoustic waves and small-scale density fluctuations. Following the discussion in [60, 61] and in Section
G. Orlando et al.: Preprint submitted to Elsevier
Page 31 of 38


AP IMEX schemes for Euler equations: non-ideal gases
Figure 14: Gresho vortex test case with the Peng-Robinson EOS (14), comparison of local Mach number 𝑀𝑙𝑜𝑐= 𝑀|𝐮|
𝑐. Left:
initial condition. Top: results at 𝑡= 𝑇𝑓= 3, after three full rotations.
3.2, we notice that the mass and momentum balance for (43) read as follows:
𝜕̄𝜌
𝜕𝑡
=
0
(121)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇𝝃𝑝
′
=
0.
(122)
Suppose now that two neighbouring mass elements characterized by densities 𝜌1 and 𝜌2, with 𝜌1 ≠𝜌2 as in the
present test case, are accelerated by a common large-scale acoustic pressure gradient. Since the time derivative of the
momentum is the same for both mass elements, their velocities must differ by a factor of 𝜌2
𝜌1 . As a consequence of
different accelerations, vorticity is generated. This phenomenon is also known as baroclinic effect and it is the result of
mutual interaction between the quasi-incompressible small-scale and the large-scale acoustic flow. Indeed, baroclinic
instabilities are well known to play a major role in large scale atmospheric dynamics [26, 54], as well as in other areas
of compressible fluid dynamics, see, e.g., [16].
We first consider this test assuming that the ideal gas law (8) holds. The computational domain is Ω = (−𝐿, 𝐿) ×
(
0, 2
5𝐿
)
. Following [61, 68], we set 𝑀= 5 × 10−2 and we take 𝐿=
1
𝑀. The initial conditions read as follows:
̄𝜌(𝐱, 0)
=
̄𝜌0 + 𝑀𝜌
′
0
(
1 + cos
(𝜋𝑥
𝐿
))
+ Φ (𝑦)
(123a)
𝑢(𝐱, 0)
=
̄𝑢0
(
1 + cos
(𝜋𝑥
𝐿
))
(123b)
𝑣(𝐱, 0)
=
0
(123c)
𝑝(𝐱, 0)
=
̄𝑝0 + 𝑀𝑝
′
0
(
1 + cos
(𝜋𝑥
𝐿
))
,
(123d)
with ̄𝜌0 = 1, 𝜌
′
0 = 0.2, ̄𝑝0 = 1.𝑝
′
0 = 𝛾, and ̄𝑢0 = √𝛾. The function Φ (𝑦) is defined by
Φ (𝑦) =
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝜌2
𝑦
2
5 𝐿
if 𝑦≤1
5𝐿−𝜀
𝜌2
(
𝑦
2
5 𝐿−1
2
)
−0.4
if 𝑦≥1
5𝐿+ 𝜀
𝜌2
1
5 𝐿−𝜀
2
5 𝐿+ 1
2𝜀
(
𝜌2
( 1
5 𝐿+𝜀
2
5 𝐿−1
2
)
−0.4 −𝜌2
1
5 𝐿−𝜀
2
5 𝐿
) (
𝑦−1
5𝐿+ 𝜀
)
otherwise,
(124)
where 𝜌2 = 0.8 and 𝜀= 10−2. Notice that, unlike in [61, 68], the function Φ is regularized to obtain a continuous
profile. Periodic boundary conditions are prescribed, whereas the final time is 𝑇𝑓= 16. The computational grid is
composed by 200 × 40 elements with 𝑟= 2. The time step is Δ𝑡= 4 × 10−3, yielding a maximum advective Courant
G. Orlando et al.: Preprint submitted to Elsevier
Page 32 of 38


AP IMEX schemes for Euler equations: non-ideal gases
number 𝐶𝑢≈0.27 and a maximum acoustic Courant number 𝐶≈2.5. Figure 15 shows a comparison of the density
between the initial and the final time. The initial density profile consists of two layers with different acceleration. Hence,
a rotational motion is induced along the separating layer and a Kelvin-Helmholtz instability develops.
Figure 15: Baroclinic vorticity generation with the ideal gas law (8), contour plot of the density. Top: 𝑡= 0. Bottom:
𝑡= 𝑇𝑓= 16.
We then replicate the test considering the SG-EOS (9). We take 𝛾= 4.4, 𝜋∞= 6.8 × 10−3, and 𝑞∞= 0. The same
initial conditions of the configuration with the ideal gas law are employed. The maximum acoustic Courant number is
𝐶≈3.5, whereas the maximum advection Courant number 𝐶𝑢≈0.23. One can easily notice that the development of
the Kelvin-Helmholtz instability depends on the EOS and on the fluid parameters (Figure 16).
Figure 16: Baroclinic vorticity generation with the SG-EOS (9), contour plot of the density at 𝑡= 𝑇𝑓= 16.
6. Conclusions
We have presented the asymptotic-preserving (AP) analysis of a general class of IMEX-RK schemes for the time
discretization of the compressible Euler equations. Based on the results of [23, 35], these approaches consider an
implicit coupling between the momentum and the energy balance, while treating the density explicitly. Third order
and fourth order time discretization schemes, in combination with a Discontinuous Galerkin (DG) for the space
discretization, have been employed for numerical simulations. The AP property of the proposed method is valid for a
general equation of state as well as for two length scales models. A number of classical benchmarks for ideal gases
and their non trivial extension for equations of state of real gases, in particular for the general cubic equation of state,
validate the proposed method in the low Mach number regime and in the limit of incompressible flows. In particular,
in spite of the use of quadrilateral meshes, the proposed method yields correct results for Mach number values that
G. Orlando et al.: Preprint submitted to Elsevier
Page 33 of 38


AP IMEX schemes for Euler equations: non-ideal gases
are typical of fluids, such as water, usually modelled as incompressible. Notice that no operator splitting, flux splitting
or relaxation techniques have been employed, differently from the approaches proposed, e.g., in [1, 24, 25, 60, 68].
In future work, we aim to consider gravity effects, so as to perform an asymptotic analysis in the limit of low Froude
numbers, and to consider an extension to two-phase flows. Moreover, as already mentioned at the end of Section 5.1,
we aim to analyze more in detail the spatial discretization and, in particular, the use of compatible finite elements, as
recently done in [93], and entropy-stable schemes.
Acknowledgements
We thank the three anonymous reviewers and the Associate Editor who handled the paper for their very useful and
constructive comments and remarks, which have greatly helped in improving the quality of the presentation of our
results. G.O. is part of the INdAM-GNCS National Research Group. The simulations have been partly run at CINECA
thanks to the computational resources made available through the ISCRA-C project FEM-GPU - HP10CQYKJ1. We
acknowledge the CINECA award for the availability of high-performance computing resources and support.
A. Eigenvalues of the implicit and explicit part
In this Appendix, we analyze the eigenvalues for the Euler equations (7). More specifically, we compute the
eigenvalues for the two subsystems obtained considering the IMEX approach described in Section 3. For the sake
of simplicity, we focus on 1D case, so that the equations can be written as follows:
𝜕𝜌
𝜕𝑡+ 𝜕𝑞
𝜕𝑥
=
0
𝜕𝑞
𝜕𝑡+ 𝜕
𝜕𝑥
(𝑞2
𝜌
)
+
1
𝑀2
𝜕𝑝
𝜕𝑥
=
0
(125)
𝜕̂𝐸
𝜕𝑡+ 𝜕ℎ𝑞
𝜕𝑥+ 1
2𝑀2 𝜕
𝜕𝑥
(𝑞3
𝜌2
)
=
0,
with 𝑞= 𝜌𝑢and ̂𝐸= 𝜌𝐸. Hence, considering the time discretization reported in Section 3, the system can be written
in the following quasi-linear form:
𝜕̃𝐖
𝜕𝑡+ ̃𝐀𝐼
𝜕̃𝐖
𝜕𝑥+ ̃𝐀𝐸
𝜕̃𝐖
𝜕𝑥= 𝟎,
(126)
with
̃𝐖=
⎡
⎢
⎢⎣
𝜌
𝑞
̂𝐸
⎤
⎥
⎥⎦
̃𝐀𝐼=
⎡
⎢
⎢
⎢⎣
0
0
0
1
𝑀2
𝜕𝑝
𝜕𝜌
1
𝑀2
𝜕𝑝
𝜕𝑞
1
𝑀2
𝜕𝑝
𝜕̂𝐸
𝑞𝜕ℎ
𝜕𝜌
𝑞𝜕ℎ
𝜕𝑞+ ℎ
𝑞𝜕ℎ
𝜕̂𝐸
⎤
⎥
⎥
⎥⎦
̃𝐀𝐸=
⎡
⎢
⎢⎣
0
1
0
−𝑢2
2𝑢
0
−𝑀2𝑢3
3
2𝑀2𝑢2
0
⎤
⎥
⎥⎦
.
Here, ̃𝐀𝐼and ̃𝐀𝐸denote matrices related to the fluxes discretized implicitly and explicitly, respectively. After some
manipulations (see [71]), we can rewrite (126) as follows:
𝜕𝐖
𝜕𝑡+ 𝐀𝐼
𝜕𝐖
𝜕𝑥+ 𝐀𝐸
𝜕𝐖
𝜕𝑥= 𝟎,
(127)
with
𝐖=
⎡
⎢
⎢⎣
𝜌
𝑢
𝑝
⎤
⎥
⎥⎦
𝐀𝐼=
⎡
⎢
⎢
⎢
⎢⎣
0
0
0
0
0
1
𝜌𝑀2
0
𝑝
𝜌−𝜌𝜕𝑒
𝜕𝜌
𝜕𝑒
𝜕𝑝
𝑢
⎤
⎥
⎥
⎥
⎥⎦
𝐀𝐸=
⎡
⎢
⎢⎣
𝑢
𝜌
0
0
𝑢
0
0
0
0
⎤
⎥
⎥⎦
.
The eigenvalues of 𝐀𝐼are
𝑢
2 −
√
𝑐2
𝑀2 + 𝑢2
4
0
𝑢
2 +
√
𝑐2
𝑀2 + 𝑢2
4 ,
G. Orlando et al.: Preprint submitted to Elsevier
Page 34 of 38


AP IMEX schemes for Euler equations: non-ideal gases
where the expression of the speed of sound 𝑐is reported in (15), while the eigenvalues of 𝐀𝐸are
0
𝑢
𝑢.
The eigenvalues of 𝐀𝐸are always real and the subsystem discretized explicitly does not take into account any acoustic
effect. However, the subsystem is only weakly hyperbolic, since 𝐀𝐸is not diagonalizable. This is related to the fact that
the terms treated explicitly in the continuity equation and in the momentum balance form the well-known pressureless
gas dynamics system [15, 65]. Since this system is weakly hyperbolic, delta-shocks can develop and the vacuum
state can occur, yielding an expansion which propagates at infinite velocity. Nevertheless, the vacuum state cannot
form spontaneously and we need to start from the vacuum to obtain the infinite velocity expansion [15]. Moreover,
in the case of regular solutions, as we are mainly interested in this work, the momentum equation decouples from the
continuity equation and reduces to the Burgers’ equation [65]. Hence, the velocity field can be computed solving the
Burgers’ equation and the continuity equation reduces to
𝜕𝜌
𝜕𝑡+ 𝑢𝜕𝜌
𝜕𝑥= −𝜌𝜕𝑢
𝜕𝑥,
(128)
which is an evolution equation for 𝜌along the characteristics for which the advecting field 𝑢and the source term
contribution 𝜕𝑢
𝜕𝑥are known.
References
[1] Abbate, E., Iollo, A., Puppo, G., 2019. An asymptotic-preserving all-speed scheme for fluid dynamics and nonlinear elasticity. SIAM Journal
on Scientific Computing , A2850–A2879.
[2] Arndt, D., Bangerth, W., Bergbauer, M., Feder, M., Fehling, M., Heinz, J., Heister, T., Heltai, L., Kronbichler, M., Maier, M., Munch, P., J-P.,
P., Turcksin, B., Wells, D., Zampini, S., 2023. The deal.II library, version 9.5. Journal of Numerical Mathematics , 231–246.
[3] Arnold, D., Brezzi, F., Cockburn, B., Marini, L., 2002. Unified analysis of discontinuous galerkin methods for elliptic problems. SIAM journal
on numerical analysis 39, 1749–1779.
[4] Ascher, U., Ruuth, S., Spiteri, R., 1997. Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations. Applied
Numerical Mathematics 25, 151–167.
[5] Bangerth, W., Hartmann, R., Kanschat, G., 2007. deal.II: a general-purpose object-oriented finite element library. ACM Transactions on
Mathematical Software (TOMS) , 24–51.
[6] Barsukow, W., 2021. Truly multi-dimensional all-speed schemes for the Euler equations on Cartesian grids. Journal of Computational Physics
435, 110216.
[7] Bassi, F., De Bartolo, C., Hartmann, R., Nigro, A., 2009. A discontinuous Galerkin method for inviscid low Mach number flows. Journal of
Computational Physics 228, 3996–4011.
[8] Bispen, G., Lukáčová-Medvid’ová, M., Yelash, L., 2017. Asymptotic preserving IMEX finite volume schemes for low Mach number Euler
equations with gravitation. Journal of Computational Physics 335, 222–248.
[9] Boscarino, S., Filbet, F., Russo, G., 2016. High order semi-implicit schemes for time dependent partial differential equations. Journal of
Scientific Computing 68, 975–1001.
[10] Boscarino, S., Pareschi, L., Russo, G., 2024. Implicit-explicit methods for evolutionary partial differential equations. 1 ed., In press to SIAM
books.
[11] Boscarino, S., Russo, G., 2009. On a class of uniformly accurate IMEX Runge-Kutta schemes and applications to hyperbolic systems with
relaxation. SIAM Journal on Scientific Computing 31, 1926–1945.
[12] Boscheri, W., Dimarco, G., Loubère, R., Tavelli, M., Vignal, M.H., 2020. A second order all Mach number IMEX finite volume solver for the
three dimensional Euler equations. Journal of Computational Physics , 109486.
[13] Boscheri, W., Dumbser, M., Ioriatti, M., Peshkov, I., Romenski, E., 2021. A structure-preserving staggered semi-implicit finite volume scheme
for continuum mechanics. Journal of Computational Physics 424, 109866.
[14] Boscheri, W., Pareschi, L., 2021. High order pressure-based semi-implicit IMEX schemes for the 3D Navier-Stokes equations at all Mach
numbers. Journal of Computational Physics 434, 110206.
[15] Bouchut, F., Jin, S., Li, X., 2003. Numerical approximations of pressureless and isothermal gas dynamics. SIAM Journal on Numerical
Analysis 41, 135–158.
[16] Brouillette, M., 2002. The Richtmyer-Meshkov instability. Annual Review of Fluid Mechanics 34, 445–468.
[17] Bruel, P., Delmas, S., Jung, J., Perrier, V., 2019. A low Mach correction able to deal with low Mach acoustics. Journal of Computational
Physics , 723–759.
[18] Buckingham, E., 1914. On physically similar systems; illustrations of the use of dimensional equations. Physical review , 345.
[19] Busto, S., Río-Martín, L., Vázquez-Cendón, M., Dumbser, M., 2021. A semi-implicit hybrid finite volume/finite element scheme for all Mach
number flows on staggered unstructured meshes. Applied Mathematics and Computation , 126117.
[20] Butcher, J., 2008. Numerical Methods for Ordinary Differential Equations. 2 ed., Wiley.
[21] Calvo, M., De Frutos, J., Novo, J., 2001. Linearly implicit Runge-Kutta methods for advection-reaction-diffusion equations. Applied Numerical
Mathematics 37, 535–549.
G. Orlando et al.: Preprint submitted to Elsevier
Page 35 of 38


AP IMEX schemes for Euler equations: non-ideal gases
[22] Carlson, J., 2011. Inflow/Outflow Boundary Conditions with Application to FUN3D: Tech. Rep. Technical Report. NASA/TM–2011-217181:
NASA.
[23] Casulli, V., Greenspan, D., 1984. Pressure method for the numerical solution of transient, compressible fluid flows. International Journal for
Numerical Methods in Fluids , 1001–1012.
[24] Chalons, C., Girardin, M., Kokh, S., 2013. Large time step and asymptotic preserving numerical schemes for the gas dynamics equations with
source terms. SIAM Journal on Scientific Computing , A2874–A2902.
[25] Chalons, C., Girardin, M., Kokh, S., 2016. An all-regime Lagrange-Projection like scheme for the gas dynamics equations on unstructured
meshes. Communications in Computational Physics , 188–233.
[26] Charney, J.G., 1947. The dynamics of long waves in a baroclinic westerly current. Journal of the Atmospheric Sciences 4, 136–162.
[27] Chorin, A., 1967. A numerical method for solving incompressible viscous flow problems. Journal of Computational Physics 2, 12–26.
[28] Cockburn, B., Shu, C.W., 2001. Runge-Kutta discontinuous Galerkin methods for convection-dominated problems. Journal of scientific
computing 16, 173–261.
[29] Cordier, F., Degond, P., Kumbaro, A., 2012. An asymptotic-preserving all-speed scheme for the Euler and Navier-Stokes equations. Journal
of Computational Physics , 5685–5704.
[30] Cowperthwaite, M., 1969. Relationships between incomplete equations of state. Journal of the Franklin Institute 287, 379–387.
[31] Dellacherie, S., 2010.
Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number.
Journal of
Computational Physics 229, 978–1016.
[32] Dellacherie, S., Jung, J., Omnes, P., Raviart, P.A., 2016. Construction of modified Godunov-type schemes accurate at any Mach number for
the compressible Euler system. Mathematical Models and Methods in Applied Sciences 26, 2525–2615.
[33] Dellacherie, S., Omnes, P., Rieper, F., 2010. The influence of cell geometry on the Godunov scheme applied to the linear wave equation.
Journal of Computational Physics 229, 5315–5338.
[34] Dimarco, G., Loubere, R., Michel-Dansac, V., Vignal, M.H., 2018. Second-order implicit-explicit total variation diminishing schemes for the
Euler system in the low Mach regime. Journal of Computational Physics , 178–201.
[35] Dumbser, M., Casulli, V., 2016. A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible Navier-Stokes
equations with general equation of state. Applied Mathematics and Computation , 479–497.
[36] Dumbser, M., Loubère, R., 2016. A simple robust and accurate a posteriori sub-cell finite volume limiter for the discontinuous Galerkin
method on unstructured meshes. Journal of Computational Physics , 163–199.
[37] Feireisl, E., Klein, R., Novotn`y, A., Zatorska, E., 2016. On singular limits arising in the scale analysis of stratified fluid flows. Mathematical
Models and Methods in Applied Sciences , 419–443.
[38] Galié, T., Jung, J., Lannabi, I., Perrier, V., 2024. Extension of an all-Mach Roe scheme able to deal with low Mach acoustics to full Euler
system. ESAIM: Proceedings and Surveys 76, 35–51.
[39] Gassner, G., Winters, A., 2021. A novel robust strategy for discontinuous Galerkin methods in computational fluid mechanics: Why? When?
What? Where? Frontiers in Physics 8, 500690.
[40] Geratz, K., 1998. Erweiterung eines Godunov-Typ-Verfahrens für mehrdimensionale kompressible Strömungen auf die Fälle kleiner und
verschwindender Machzahl. Ph.D. thesis. RWTH Aachen.
[41] Giraldo, F., 2020. An Introduction to Element-Based Galerkin Methods on Tensor-Product Bases. Springer Nature.
[42] Gottlieb, S., Shu, C.W., Tadmor, E., 2001. Strong stability-preserving high-order time discretization methods. SIAM Review , 89–112.
[43] Grenier, N., Vila, J.P., Villedieu, P., 2013. An accurate low-Mach scheme for a compressible two-fluid model applied to free-surface flows.
Journal of Computational Physics , 1–19.
[44] Gresho, P., 1990. On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite
element method that also introduces a nearly consistent mass matrix. Part 1: Theory. International Journal for Numerical Methods in Fluids ,
587–620.
[45] Guillard, H., Viozat, C., 1999. On the behaviour of upwind schemes in the low Mach number limit. Computers & Fluids 28, 63–86.
[46] Haack, J., Jin, S., Liu, J.G., 2012.
An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations.
Communications in Computational Physics , 955–980.
[47] Happenhofer, N., Grimm-Strele, H., Kupka, F., Löw-Baselli, B., Muthsam, H., 2011. A low Mach number solver: enhancing stability and
applicability. ArXiv e-prints .
[48] Harlow, F., Amsden, A., 1968. Numerical calculation of almost incompressible flow. Journal of Computational Physics , 80–93.
[49] Harlow, F., Amsden, A., 1971. A numerical fluid dynamics calculation method for all flow speeds. Journal of Computational Physics 8,
197–213.
[50] Hennink, A., Tiberga, M., Lathouwers, D., 2021. A pressure-based solver for low-Mach number flow using a discontinuous Galerkin method.
Journal of Computational Physics 425, 109877.
[51] Herbin, R., Kheriji, W., Latché, J.C., 2014. On some implicit and semi-implicit staggered schemes for the shallow water and euler equations.
ESAIM: Mathematical Modelling and Numerical Analysis , 1807–1857.
[52] Herbin, R., Latché, J.C., Saleh, K., 2021. Low Mach number limit of some staggered schemes for compressible barotropic flows. Mathematics
of Computation , 1039–1087.
[53] Hope-Collins, J., di Mare, L., 2023. Artificial diffusion for convective and acoustic low Mach number flows I: Analysis of the modified
equations, and application to Roe-type schemes. Journal of Computational Physics 475, 111858.
[54] Jablonowski, C., Williamson, D., 2006. A baroclinic instability test case for atmospheric model dynamical cores. Quarterly Journal of the
Royal Meteorological Society 132, 2943–2975.
[55] Jung, J., Perrier, V., 2022. Steady low Mach number flows: identification of the spurious mode and filtering method. Journal of Computational
Physics 468, 111462.
G. Orlando et al.: Preprint submitted to Elsevier
Page 36 of 38


AP IMEX schemes for Euler equations: non-ideal gases
[56] Jung, J., Perrier, V., 2024. Behavior of the Discontinuous Galerkin Method for Compressible Flows at Low Mach Number on Triangles and
Tetrahedrons. SIAM Journal on Scientific Computing 46, A452–A482.
[57] Kennedy, C., Carpenter, M., 2003.
Additive Runge-Kutta schemes for convection-diffusion-reaction equations.
Applied Numerical
Mathematics , 139–181.
[58] Kennedy, C., Carpenter, M., 2019.
Higher-order additive Runge-Kutta schemes for ordinary differential equations.
Applied numerical
mathematics 136, 183–205.
[59] Klainerman, S., Majda, A., 1981. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of
compressible fluids. Communications on pure and applied Mathematics , 481–524.
[60] Klein, R., 1995. Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: One-dimensional flow. Journal
of Computational Physics , 213–237.
[61] Klein, R., 2002. Numerical modelling of high speed and low speed combustion, in: Nonlinear PDE’s in Condensed Matter and Reactive Flows.
Springer, pp. 189–226.
[62] Klein, R., Botta, N., Schneider, T., Munz, C.D., Roller, S., Meister, A., Hoffmann, L., Sonar, T., 2001. Asymptotic adaptive methods for
multi-scale problems in fluid mechanics. Journal of Engineering Mathematics , 261–343.
[63] Kučera, V., Lukáčová-Medvid’ová, M., Noelle, S., Schütz, J., 2022. Asymptotic properties of a class of linearly implicit schemes for weakly
compressible Euler equations. Numerische Mathematik , 1–25.
[64] Le Métayer, O., Saurel, R., 2016. The Noble-Abel Stiffened-Gas equation of state. Physics of Fluids , 046102.
[65] LeVeque, R., 2002. Finite Volume Methods for Hyperbolic Problems. Cambridge University Press.
[66] Liska, R., Wendroff, B., 2003. Comparison of several difference schemes on 1D and 2D test problems for the Euler equations. SIAM Journal
on Scientific Computing , 995–1017.
[67] Munz, C., Roller, S., Klein, R., Geratz, K.J., 2003. The extension of incompressible flow solvers to the weakly compressible regime. Computers
& Fluids 32, 173–196.
[68] Noelle, S., Bispen, G., Arun, K., Lukáčová-Medvid’ová, M., Munz, C.D., 2014. A weakly asymptotic preserving low Mach number scheme
for the Euler equations of gas dynamics. SIAM Journal on Scientific Computing , B989–B1024.
[69] Orlando, G., 2023a. A filtering monotonization approach for DG discretizations of hyperbolic problems. Computers & Mathematics with
Applications , 113–125.
[70] Orlando, G., 2023b. Modelling and simulations of two-phase flows including geometric variables. Ph.D. thesis. Politecnico di Milano.
http://hdl.handle.net/10589/198599.
[71] Orlando, G., Barbante, P., Bonaventura, L., 2022a. An efficient IMEX-DG solver for the compressible Navier-Stokes equations for non-ideal
gases. Journal of Computational Physics , 111653.
[72] Orlando, G., Benacchio, T., Bonaventura, L., 2023. An IMEX-DG solver for atmospheric dynamics simulations with adaptive mesh refinement.
Journal of Computational and Applied Mathematics , 115124.
[73] Orlando, G., Boscarino, S., Russo, G., 2025. A quantitative comparison of high-order asymptotic-preserving and asymptotically-accurate
IMEX methods for the Euler equations with non-ideal gases. URL: https://arxiv.org/abs/2501.12733, arXiv:2501.12733.
[74] Orlando, G., Della Rocca, A., Barbante, P., Bonaventura, L., Parolini, N., 2022b.
An efficient and accurate implicit DG solver for the
incompressible Navier-Stokes equations. International Journal for Numerical Methods in Fluids , 1484–1516.
[75] Pareschi, L., Russo, G., 2005. Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation. Journal of
Scientific computing 25, 129–155.
[76] Park, J., Munz, C., 2005. Multiple pressure variables methods for fluid flow at all Mach numbers. International Journal for Numerical Methods
in Fluids 49, 905–931.
[77] Quarteroni, A., Valli, A., 2008. Numerical approximation of partial differential equations. volume 23. Springer Science & Business Media.
[78] Rieper, F., 2011. A low-Mach number fix for Roe’s approximate Riemann solver. Journal of Computational Physics 230, 5263–5287.
[79] Rusanov, V., 1962. The calculation of the interaction of non-stationary shock waves and obstacles. USSR Computational Mathematics and
Mathematical Physics , 304–320.
[80] Sandler, S., 2017. Chemical, biochemical, and engineering thermodynamics. John Wiley & Sons.
[81] Steppeler, J., Hess, R., Doms, G., Schättler, U., Bonaventura, L., 2003. Review of numerical methods for nonhydrostatic weather prediction
models. Meteorology and Atmospheric Physics , 287–301.
[82] Suliciu, I., 1990. On modelling phase transitions by means of rate-type constitutive equations. Shock wave structure. International Journal of
Engineering Science , 829–841.
[83] Tavelli, M., Dumbser, M., 2017. A pressure-based semi-implicit space-time discontinuous Galerkin method on staggered unstructured meshes
for the solution of the compressible Navier-Stokes equations at all Mach numbers. Journal of Computational Physics , 341–376.
[84] Temam, R., 1969. Sur l’approximation de la solution des équations de Navier-Stokes par la méthode des pas fractionnaires (II). Archive for
Rational Mechanics and Analysis , 377–385.
[85] Therme, N., Zaza, C., 2014. Comparison of cell-centered an staggered pressure-correction schemes for all-Mach flows, in: Finite Volumes for
Complex Applications VII -Elliptic, Parabolic and Hyperbolic. J. Fuhrmann, M. Ohlberger, and C. Rohde, editors, pp. 975–983.
[86] Thomann, A., Zenk, M., Puppo, G., Klingenberg, C., 2019. An all speed second order IMEX relaxation scheme for the Euler equations.
Communications in Computational Physics , 591–620.
[87] Thornber, B., Drikakis, D., Williams, R., Youngs, D., 2008. On entropy generation and dissipation of kinetic energy in high-resolution
shock-capturing schemes. Journal of Computational Physics 227, 4853–4872.
[88] Toselli, A., 2002. ℎ𝑝Discontinuous Galerkin approximations for the Stokes problem. Mathematical Models and Methods in Applied Sciences
12, 1565–1597.
[89] Vidal, J., 2001. Thermodynamics: Applications to chemical engineering and petroleum industry. Editions Technip.
[90] Wanner, G., Hairer, E., 1996. Solving Ordinary Differential Equations II. volume 375. Springer Berlin Heidelberg New York.
G. Orlando et al.: Preprint submitted to Elsevier
Page 37 of 38


AP IMEX schemes for Euler equations: non-ideal gases
[91] Waruszewski, M., Kozdon, J., Wilcox, L., Gibson, T., Giraldo, F., 2022. Entropy stable discontinuous Galerkin methods for balance laws in
non-conservative form: Applications to the Euler equations with gravity. Journal of Computational Physics 468, 111507.
[92] Yee, H., Sandham, N., Djomehri, M., 1999. Low-dissipative high-order shock-capturing methods using characteristic-based filters. Journal
of computational physics 150, 199–238.
[93] Zampa, E., Dumbser, E., 2025. An asymptotic-preserving and exactly mass-conservative semi-implicit scheme for weakly compressible flows
based on compatible finite elements. Journal of Computational Physics 521, 113551.
[94] Zeifang, J., Schütz, J., Kaiser, K., Beck, A., Lukáčová-Medvid’ová, M., Noelle, S., 2019. A Novel Full-Euler Low Mach Number IMEX
Splitting. Communications in Computational Physics 27, 292–320.
G. Orlando et al.: Preprint submitted to Elsevier
Page 38 of 38


