Highlights
A quantitative comparison of high-order asymptotic-preserving and asymptotically-accurate
IMEX methods for the Euler equations with non-ideal gases
Giuseppe Orlando,Sebastiano Boscarino,Giovanni Russo
• Quantitative comparison between IMEX and SI-IMEX asymptotic-preserving methods
• Asymptotically-accurate schemes up to fourth order
• Extension of SI-IMEX method for non-ideal gases
• Analysis of the stiff dependence for non-ideal gases so as to avoid the solution of a nonlinear equation
arXiv:2501.12733v5  [math.NA]  22 Oct 2025


A quantitative comparison of high-order asymptotic-preserving and
asymptotically-accurate IMEX methods for the Euler equations
with non-ideal gases
Giuseppe Orlandoa,∗, Sebastiano Boscarinob and Giovanni Russob
aCMAP, CNRS, École polytechnique, Institut Polytechnique de Paris, Route de Saclay, Palaiseau, 91120, France
bDepartment of Mathematics and Computer Science, University of Catania, , Catania, 95125, Italy
A R T I C L E I N F O
Keywords:
Asymptotic-preserving
Asymptotically-accurate
Euler equations
IMEX
Semi-implicit
Discontinuous Galerkin
Non-ideal gas
A B S T R A C T
We present a quantitative comparison between two different Implicit-Explicit Runge-Kutta
(IMEX-RK) approaches for the Euler equations of gas dynamics, specifically tailored for the low
Mach limit. In this regime, a classical IMEX-RK approach involves an implicit coupling between
the momentum and energy balance so as to avoid the acoustic CFL restriction, while the density
can be treated in a fully explicit fashion. This approach leads to a mildly nonlinear equation
for the pressure, which can be solved according to a fixed point procedure. An alternative
strategy consists of employing a semi-implicit temporal integrator based on IMEX-RK methods
(SI-IMEX-RK). The stiff dependence is carefully analyzed, so as to avoid the solution of a
nonlinear equation for the pressure also for equations of state (EOS) of non-ideal gases. The
spatial discretization is based on a Discontinuous Galerkin (DG) method, which naturally allows
high-order accuracy. The asymptotic-preserving (AP) and the asymptotically-accurate (AA)
properties of the two approaches are assessed on a number of classical benchmarks for ideal
gases and on their extension to non-ideal gases.
1. Introduction
The Euler equations of gas dynamics represent the standard mathematical model for a wide range of applications in
fluid mechanics, mechanical engineering, and environmental engineering [76]. Several numerical methods have been
developed over the years, which can generally be divided into two categories, depending on a dimensionless parameter
called Mach number. The Mach number 𝑀represents the ratio between the local fluid velocity and the speed of sound in
the medium [44]. For moderate to high Mach numbers, compressible effects have to be taken into account and numerical
discretization strategies typically rely on Godunov-type shock-capturing schemes [42, 49, 63, 65, 82, 86]. On the other
hand, in the low Mach number regime, the flow can be considered weakly compressible or even incompressible.
Explicit time discretization methods are very popular for high Mach number flows [65, 68, 82]. The time step has to
satisfy a Courant-Friedrichs-Lewy (CFL) condition, given by the mesh size divided by the fastest wave speed [15, 33].
For moderate to high Mach numbers, this restriction is not a problem, since one is interested in resolving all the waves.
However, for flows characterized by low values of the Mach number, severe time step restrictions may be required by
these schemes. In this regime, acoustic waves usually carry a negligible amount of energy and, therefore, one may not
be interested in resolving them. Hence, the system becomes stiff and stability limitations on the time step are much
stricter than the restrictions imposed by accuracy.
The use of implicit and semi-implicit time discretization methods, so as to avoid the acoustic CFL restriction,
has a long tradition in low Mach number flows [7, 15, 25, 58, 71, 72, 74]. Several numerical methods for weakly
compressible flows have been proposed in the literature, see, e.g., [27, 32, 35, 61, 67, 85] and the references therein.
Since the seminal paper [25], an effective approach to deal with low Mach flows is given by pressure-based algorithms.
Indeed, an implicit treatment of the pressure gradient term within the momentum equation and of the pressure work
term in the energy equation is sufficient to remove the acoustic CFL restriction and to decouple acoustic and transport
effects [25].
∗Corresponding author
giuseppe.orlando@polytechnique.edu (G. Orlando); sebastiano.boscarino@unict.it (S. Boscarino);
giovanni.russo1@unict.it (G. Russo)
ORCID(s): 0000-0002-7119-4231 (G. Orlando)
G. Orlando et al.: Preprint submitted to Elsevier
Page 1 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Another aspect to consider is that weakly compressible flows are characterized by multiple length and time scales.
Under specific assumptions, the compressible Euler equations converge to the incompressible ones as the Mach number
goes to zero [38, 57]. Hence, the density remains constant along the fluid particle trajectories and the pressure acts as a
Lagrange multiplier to enforce incompressibility of the flow [15, 58]. Robust numerical methods for the Euler equations
should recover the incompressible limit for vanishing Mach number. For this purpose, the concept of asymptotic-
preserving (AP) schemes has been introduced [47]. We also refer to [17] for a review of asymptotic-preserving methods
for quasilinear hyperbolic systems with stiff relaxation. A numerical method for the compressible Euler equations is
said to be asymptotic-preserving if its stability condition does not depend on the Mach number 𝑀and if it provides
a consistent discretization of the incompressible Euler equations as 𝑀→0. The AP property of the aforementioned
approach was proven in [74].
Preserving incompressibility and resolving vortex dynamics are among the main purposes of numerical discretiza-
tions for weakly-compressible flows and high-order methods can help to reach this goal. The aim of the present work
is to provide a quantitative comparison between two different Implicit-Explicit Runge-Kutta (IMEX-RK) approaches
for the Euler equations, specifically tailored for the low Mach limit. The first approach uses an IMEX-RK solver, as
proposed in [71] and validated in [72] for atmospheric applications. A key feature of this approach is the implicit
treatment of acoustic waves, while material waves are handled explicitly. This method combines IMEX-RK schemes,
carefully designed for stability and accuracy, with a time-stepping size that is independent of the Mach number 𝑀. The
spatial discretization is based on the Discontinuous Galerkin (DG) method [40], which naturally allows for high-order
accuracy and has proven highly effective for a wide range of computational fluid dynamics problems, across various
flow regimes (see, e.g., [11, 30, 31, 54]). Additionally, the IMEX-DG method can handle a general equation of state
(EOS) [71, 74], for which only a few studies have been devoted [1, 32], and that, as we will discuss later, poses additional
numerical challenges. However, this approach leads to a mildly nonlinear equation for the pressure also for ideal gases.
In order to avoid this nonlinearity, an alternative strategy consists of employing a semi-implicit temporal integrator
based on IMEX-RK schemes (SI-IMEX-RK), similar to the one adopted in [13]. This method leads to a linearized
equation for both the pressure and the EOS. We refer to this scheme as the SI-IMEX-DG.
Furthermore, in this work, we show that the time discretization methods satisfy the AP property and the
asymptotically accuracy (AA) property, i.e. they maintain their high-order accuracy also in the case of 𝑀→0.
Moreover, if specific boundary conditions are considered, the limit model can differ from the incompressible Euler
equations and depends on the employed EOS [74]. Specific numerical treatments for non-ideal EOS in the framework
of the IMEX-DG scheme were presented in [71]. In [21], for non-ideal EOS, a nonlinear equation is solved through a
Newton method. We propose here a novel different strategy, so as to avoid any nonlinear equation for the semi-implicit
approach.
The paper is structured as follows. In Section 2, we briefly recall the mathematical model and its limit as 𝑀→0.
In Section 3, we present the numerical method. More specifically, we outline the IMEX and the SI-IMEX time
discretization methods. Moreover, we provide suitable strategies to deal with a general class of EOS. Some details
of the DG formulation will also be discussed, specifying some advantages and disadvantages of this method for low
Mach number flows. In Section 4, some numerical results to assess the properties of the two methods and to compare
them are presented. Finally, some conclusions and perspectives for future work are discussed in Section 5.
2. The mathematical model
Let Ω ⊂ℝ𝑑, 1 ≤𝑑≤3 be a connected open bounded set with a sufficiently smooth boundary 𝜕Ω and denote by
𝐱the spatial coordinates and by 𝑡the temporal coordinate. The mathematical model consists of the fully compressible
Euler equations of gas dynamics, written in non-dimensional form as follows [15, 21, 58, 74]:
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
(1)
𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(𝜌𝐸+ 𝑝) 𝐮]
=
0.
Here, 𝜌is the density, 𝐮is the fluid velocity, 𝑝is the pressure, and 𝐸is the total energy per unit of mass. Moreover,
𝑀≡𝑢0∕
√
𝑝0∕𝜌0, where 𝑢0, 𝑝0 and 𝜌0 are reference fluid speed, pressure and density, respectively. It is related to
G. Orlando et al.: Preprint submitted to Elsevier
Page 2 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
the Mach number 𝑀0, i.e. the ratio between a typical fluid velocity and a typical speed of sound. For a 𝛾-law gas,
for example, it is 𝑀= 𝑀0
√𝛾. We are mainly interested in the so-called low Mach regime, i.e. 𝑀≪1, for which
material waves are much slower than acoustic waves. The previous set of equations has to be completed by an equation
of state (EOS). Further details on the EOS will be discussed in the upcoming Section 2.1. The total energy 𝜌𝐸can be
rewritten as 𝜌𝐸= 𝜌𝑒+ 𝑀2𝜌𝑘, where 𝑒denotes the internal energy and 𝑘= |𝐮|2 ∕2 the kinetic energy. For the sake of
convenience, we also introduce the specific enthalpy ℎ= 𝑒+ 𝑝∕𝜌, and we notice that the energy flux can be rewritten
as
(𝜌𝐸+ 𝑝) 𝐮=
(
𝑒+ 𝑀2𝑘+ 𝑝
𝜌
)
𝜌𝐮= (ℎ+ 𝑀2𝑘) 𝜌𝐮.
(2)
Hence, system (1) reads equivalently as follows:
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
(3)
𝜕𝜌𝐸
𝜕𝑡
+ ∇⋅[(ℎ+ 𝑀2𝑘) 𝜌𝐮]
=
0.
2.1. The equation of state
System (3) has to be completed with an equation of state (EOS). In this work, we focus on the classical ideal gas
law, the stiffened gas EOS (SG-EOS) [62], and the general cubic EOS [87]. The equation that links together pressure,
density, and internal energy for an ideal gas is given by [87]
𝑝= (𝛾−1) 𝜌𝑒= (𝛾−1)
(
𝜌𝐸−1
2𝑀2𝜌𝐮⋅𝐮
)
.
(4)
Notice that (4) is valid only for a constant value of the specific heats ratio 𝛾[87]. The analogous relation for the SG-EOS
reads as follows:
𝑝= (𝛾−1)
(𝜌𝑒−𝜌𝑞∞
) −𝛾𝜋∞= (𝛾−1)
(
𝜌𝐸−1
2𝑀2𝜌𝐮⋅𝐮−𝜌𝑞∞
)
−𝛾𝜋∞,
(5)
with 𝑞∞and 𝜋∞representing constant parameters that determine the characteristics of the fluid. Notice that, for
𝑞∞= 𝜋∞= 0 in (5), we recover (4). The last relation that we consider is the so-called general cubic EOS, for
which the link between pressure, density, and temperature can be expressed as follows [81, p. 221], [87, p. 119]:
𝑝=
𝜌𝑅𝑔𝑇
1 −𝜌𝑏−
𝑎(𝑇)𝜌2
(1 −𝜌𝑏𝑟1
) (1 −𝜌𝑏𝑟2
).
(6)
After some algebraic manipulations, (6) can be expressed as [81, p. 222]
𝑧3 + 𝑐0𝑧2 + 𝑐1𝑧+ 𝑐2 = 0,
(7)
which is a cubic polynomial for the compressibility factor 𝑧= 𝑝∕(𝜌𝑅𝑔𝑇). The parameters 𝑐0, 𝑐1, and 𝑐2 depend on the
thermodynamic state. Moreover, 𝑅𝑔= 𝑅∕𝑚∗denotes the specific gas constant, with 𝑅being the gas constant and 𝑚∗
the molar mass of the gas. Notice that for 𝑎= 𝑏= 0, the expression of the pressure of an ideal gas is recovered. As
discussed in [81, 87], it is convenient to express thermodynamic functions such as the internal energy or the enthalpy
as the sum of a contribution due to the ideal gas and a residual contribution due to non-ideality. Hence, after some
manipulations, the equation linking together internal energy, density, and temperature, is given by [81, p. 231], [87,
p. 116]
𝑒= 𝑒#(𝑇(𝑝, 𝜌)) +
𝑎(𝑇(𝑝, 𝜌)) −𝑇(𝑝, 𝜌) 𝑑𝑎(𝑇(𝑝,𝜌))
𝑑𝑇
𝑏
1
𝑟1 −𝑟2
log
(1 −𝜌𝑏𝑟1
1 −𝜌𝑏𝑟2
)
.
(8)
Here, 𝑒#(𝑇(𝑝, 𝜌)) denotes the internal energy of an ideal gas, which is function solely of the temperature 𝑇, 𝑟1 and 𝑟2
are suitable constants, whereas the parameters 𝑎(𝑇) and 𝑏determine fluid characteristics [87]. More specifically, 𝑎(𝑇)
G. Orlando et al.: Preprint submitted to Elsevier
Page 3 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
is related to intermolecular forces, whereas 𝑏, the so called co-volume, takes into account the volume occupied by the
molecules. Notice that, for 𝑟1 →0 and 𝑟2 →0, then
1
𝑟1−𝑟2 log
( 1−𝜌𝑏𝑟1
1−𝜌𝑏𝑟2
)
→−𝜌𝑏, which corresponds to the van der
Waals EOS. For 𝑟1 = −1 −
√
2, 𝑟2 = −1 +
√
2, we get the Peng-Robinson EOS [77], [87, p. 118]. If 𝑐𝑣= 𝑑𝑒#
𝑑𝑇is
constant, as in the case of calorically perfect gas, relation (8) can be simplified
𝑒= 𝑐𝑣𝑇(𝑝, 𝜌) +
𝑎(𝑇(𝑝, 𝜌)) −𝑇(𝑝, 𝜌) 𝑑𝑎(𝑇(𝑝,𝜌))
𝑑𝑇
𝑏
1
𝑟1 −𝑟2
log
(1 −𝜌𝑏𝑟1
1 −𝜌𝑏𝑟2
)
.
(9)
2.2. Asymptotic expansion
In this Section, we analyze the asymptotic limit of (3) as 𝑀→0. Consider the following expansion for density,
velocity, and pressure, respectively:
𝜌(𝐱, 𝑡)
=
̄𝜌(𝐱, 𝑡) + 𝑀𝜌′(𝐱, 𝑡) + 𝑀2𝜌′′(𝐱, 𝑡) + 𝑜(𝑀2)
(10)
𝐮(𝐱, 𝑡)
=
̄𝐮(𝐱, 𝑡) + 𝑀𝐮′(𝐱, 𝑡) + 𝑀2𝐮′′(𝐱, 𝑡) + 𝑜(𝑀2)
(11)
𝑝(𝐱, 𝑡)
=
̄𝑝(𝐱, 𝑡) + 𝑀𝑝′(𝐱, 𝑡) + 𝑀2𝑝′′(𝐱, 𝑡) + 𝑜(𝑀2)
(12)
From now on, to simplify the notation, we omit the explicit dependence on space and time for all the variables.
Substituting (10) and (11) into the continuity equation in (3), the leading order term relation is
𝜕̄𝜌
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮) = 0.
(13)
The leading order term relation for the momentum balance reduces to
∇̄𝑝= 𝟎.
(14)
Hence, ̄𝑝is a function solely of time. Analogously, for the first order term, we obtain
∇𝑝′ = 𝟎.
(15)
In addition, the second order term relation reads as follows:
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝑝′′ = 𝟎,
(16)
where 𝑝′′ represents a dynamical pressure [32, 58, 85]. Finally, the leading order term relation for the energy equation
reads as follows:
𝜕̄𝜌̄𝑒
𝜕𝑡+ ∇⋅( ̄𝜌̄ℎ̄𝐮) = 0,
(17)
where ̄𝑒= 𝑒( ̄𝜌, ̄𝑝) and ̄ℎ= ℎ( ̄𝜌, ̄𝑝). Since ̄𝜌̄𝑒= ̄𝜌̄ℎ−̄𝑝, we obtain
𝜕̄𝜌̄ℎ
𝜕𝑡−𝜕̄𝑝
𝜕𝑡+ ̄𝐮⋅∇( ̄𝜌̄ℎ) + ̄𝜌̄ℎ(∇⋅𝐮) = 0,
(18)
or equivalently, considering ̄𝜌̄ℎ= ( ̄𝜌̄ℎ)
( ̄𝜌, ̄𝑝),
𝜕̄𝜌̄ℎ
𝜕̄𝜌
(𝜕̄𝜌
𝜕𝑡+ ̄𝐮⋅∇̄𝜌
)
+ 𝜕̄𝜌̄ℎ
𝜕̄𝑝
(𝜕̄𝑝
𝜕𝑡+ ̄𝐮⋅∇̄𝑝
)
−𝜕̄𝑝
𝜕𝑡+ ̄𝜌̄ℎ(∇⋅̄𝐮) = 0.
(19)
Thanks to (13) and (14), we obtain
(
̄𝜌̄ℎ−𝜕̄𝜌̄ℎ
𝜕̄𝜌̄𝜌
)
(∇⋅̄𝐮) +
(𝜕̄𝜌̄ℎ
𝜕̄𝑝−1
) 𝑑̄𝑝
𝑑𝑡= 0,
(20)
or, since
̄𝜌̄ℎ−𝜕̄𝜌̄ℎ
𝜕̄𝜌̄𝜌= −̄𝜌2 𝜕̄ℎ
𝜕̄𝜌
G. Orlando et al.: Preprint submitted to Elsevier
Page 4 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
and
𝜕̄𝜌̄ℎ
𝜕̄𝑝−1 = 𝜕̄𝜌̄𝑒
𝜕̄𝑝,
equivalently,
−̄𝜌2 𝜕̄ℎ
𝜕̄𝜌(∇⋅̄𝐮) + 𝜕̄𝜌̄𝑒
𝜕̄𝑝
𝑑̄𝑝
𝑑𝑡= 0.
(21)
We assume 𝜕̄ℎ
𝜕̄𝜌≠0, as it holds away from vacuum. If 𝑑̄𝑝
𝑑𝑡= 0, we recover the incompressibility constraint
∇⋅̄𝐮= 0.
(22)
Summing up, the asymptotic limit of (3) is
𝜕̄𝜌
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮)
=
0
∇̄𝑝
=
𝟎
∇𝑝′
=
𝟎
(23)
𝜕̄𝜌̄𝐮
𝜕𝑡+ ∇⋅( ̄𝜌̄𝐮⊗̄𝐮) + ∇𝑝′′
=
𝟎
−̄𝜌2 𝜕̄ℎ
𝜕̄𝜌(∇⋅̄𝐮) + 𝜕̄𝜌̄𝑒
𝜕̄𝑝
𝜕̄𝑝
𝜕𝑡
=
0.
System (23) represents a more general asymptotic limit with respect to the EOS and the boundary conditions of the
compressible Euler equations for vanishing Mach number [34, 74]. We notice that we can rewrite the last equation of
system (23) as follows:
∇⋅̄𝐮=
(𝜕̄𝜌̄𝑒
𝜕̄𝑝
) (
̄𝜌2 𝜕̄ℎ
𝜕̄𝜌
)−1 𝑑̄𝑝
𝑑𝑡= −
1
̄𝜌𝑐2 ( ̄𝜌, ̄𝑝)
𝑑̄𝑝
𝑑𝑡,
(24)
where 𝑐denotes the speed of sound. Indeed, from the definition of ̄ℎ, one has
(
̄𝜌2 𝜕̄ℎ
𝜕̄𝜌
) (𝜕̄𝜌̄𝑒
𝜕̄𝑝
)−1
= 𝜌
(
𝜕̄𝑒
𝜕̄𝜌−̄𝑝
̄𝜌2
) (
𝜕̄𝑒
𝜕̄𝑝
)−1
.
From the first principle of thermodynamics, denoting by 𝑠the specific entropy, one has
𝑇𝑑𝑠= 𝑑𝑒+ 𝑝𝑑1
𝜌= 𝜕𝑒
𝜕𝜌𝑑𝜌+ 𝜕𝑒
𝜕𝑝𝑑𝑝−𝑝
𝜌2 𝑑𝜌=
(
𝜕𝑒
𝜕𝜌−𝑝
𝜌2
)
𝑑𝜌+ 𝜕𝑒
𝜕𝑝𝑑𝑝.
As 𝑐2 = 𝜕𝑝
𝜕𝜌
||||𝑠
, one has
𝑐2 = 𝜕𝑝
𝜕𝜌
||||𝑠
=
(
𝜕𝑒
𝜕𝑝
)−1 ( 𝑝
𝜌2 −𝜕𝑒
𝜕𝜌
)
= −
(
𝜕𝑒
𝜕𝑝
)−1 𝜕ℎ
𝜕𝜌,
(25)
and therefore
1
𝜌𝑐2 = −1
𝜌
(
𝜕𝑒
𝜕𝑝
)−1 𝜕ℎ
𝜕𝜌,
which proves (24) when applied to the lowest order terms in the asymptotic expansions (10), (12). For more details
consult [74, 87, 89]. Under periodic or free-slip boundary conditions, thanks to the divergence theorem, we have
∫Ω
∇⋅̄𝐮𝑑Ω = 0,
G. Orlando et al.: Preprint submitted to Elsevier
Page 5 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
so that, by integrating (24) on Ω, we find 𝑑̄𝑝
𝑑𝑡= 0. On the other hand, as one can easily notice from (24), a time-
dependent pressure with large amplitude variations imposed by a Dirichlet outflow boundary condition leads to a
non-incompressible flow, i.e. ∇⋅̄𝐮≠0 and depending on the specific EOS. Consider, e.g., the ideal gas law (4). We
get
𝜕̄𝜌̄𝑒
𝜕̄𝑝=
1
𝛾−1
̄𝜌2 𝜕̄ℎ
𝜕̄𝜌= −
𝛾
𝛾−1 ̄𝑝,
(26)
so that (24) reduces to
∇⋅̄𝐮= −1
𝛾
𝑑log ̄𝑝
𝑑𝑡
.
(27)
Hence, the compressibility of a fluid described by the ideal gas law (4) is uniform in space and changes only in time.
This is no longer valid for a general EOS [74].
3. The numerical framework
In the low Mach number limit, pressure gradients terms, which are proportional to 1∕𝑀2, yield stiff components for
the resulting semi-discretized ODE system [25, 66, 71]. Implicit-Explicit Runge-Kutta (IMEX-RK) methods [13, 55]
are widely employed for ODE systems that include both stiff and non-stiff components, to which the implicit and explicit
schemes are applied, respectively. Therefore, an implicit coupling between the energy equation and the momentum one
is appropriate, while the continuity equation can be treated in a fully explicit fashion. The spatial discretization is based
on the Discontinuous Galerkin (DG) method, which easily allows for high-order accuracy. We refer to [40] for a general
introduction to the method. In this Section, we review some well-known concepts of IMEX-RK schemes. Then, we
present the IMEX and the SI-IMEX time discretization for system (3), respectively. Finally, some details concerning
the spatial discretization will be also discussed.
3.1. IMEX Runge-Kutta schemes
Implicit-Explicit Runge–Kutta (IMEX-RK) methods find extensive application in the numerical solution of
PDEs, such as hyperbolic systems with relaxations [14, 75], convection–diffusion equations [8] and convec-
tion–diffusion–reaction equations [55, 56]. Let us start considering the following initial value problem for a system of
ODE’s
𝑑𝐲
𝑑𝑡= 𝐟𝐸(𝐲, 𝑡) + 𝐟𝐼(𝐲, 𝑡) ,
𝐲(0) = 𝐲0,
(28)
where 𝐲(𝑡) ∈ℝ𝑀, 𝑀≥1 and we assume that 𝐟𝐸and 𝐟𝐼∶ℝ𝑀× ℝ→ℝ𝑀are Lipschitz functions of 𝐲(𝑡). We assume
that the term 𝐟𝐼is stiff and the term 𝐟𝐸is non-stiff.
An 𝑠-stage IMEX-RK scheme applied to system (28) takes the form:
𝐯(𝑙)
=
𝐯𝑛+ Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚𝐟𝐸
(𝑡𝑛+ ̃𝑐𝑚Δ𝑡, 𝐯(𝑚)) + Δ𝑡
𝑠
∑
𝑚=1
𝑎𝑙𝑚𝐟𝐼
(𝑡𝑛+ 𝑐𝑚Δ𝑡, 𝐯(𝑚)) ,
(29a)
𝐯𝑛+1
=
𝐯𝑛+ Δ𝑡
𝑠
∑
𝑙=1
̃𝑏𝑙𝐟𝐸
(𝑡𝑛+ ̃𝑐𝑙Δ𝑡, 𝐯(𝑙)) + Δ𝑡
𝑠
∑
𝑙=1
𝑏𝑙𝐟𝐼
(𝑡𝑛+ 𝑐𝑙Δ𝑡, 𝐯(𝑙)) .
(29b)
where the quantities 𝐯(𝑙) for 𝑙= 1, … , 𝑠, are called internal stages and approximate the exact solution 𝐲(𝑡), at time
𝑡= 𝑡𝑛+ 𝑐𝑙Δ𝑡, whereas 𝐯𝑛+1 is the numerical solution that approximates the exact solution 𝐲(𝑡) at time 𝑡= 𝑡𝑛+ Δ𝑡.
An 𝑠-stage IMEX-RK method is defined by two 𝑠× 𝑠real matrices ̃𝐀= { ̃𝑎𝑙𝑚
} and 𝐀= {𝑎𝑙𝑚
}, where the matrix ̃𝐀
corresponds to the explicit method and is a lower triangular matrix with zero diagonal, i.e., ̃𝑎𝑙𝑚= 0 for 𝑙≤𝑚, while
𝐀is the one corresponding to the implicit scheme. We consider Diagonally Implicit Runge-Kutta (DIRK) methods
for the implicit scheme so that 𝑎𝑙𝑚= 0 for 𝑙< 𝑚. The use of a DIRK method for the treatment of 𝐟𝐼provides a
sufficient condition to guarantee that the function 𝐟𝐸is always evaluated explicitly. The method is also characterized
by the quadrature nodes ̃𝐜= (0, ̃𝑐2, … ̃𝑐𝑠
)⊤, 𝐜= (𝑐1, 𝑐2, ..., 𝑐𝑠
)⊤, given by the usual relation
𝑠
∑
𝑚=1
𝑎𝑙𝑚= 𝑐𝑙
𝑠
∑
𝑚=1
̃𝑎𝑙𝑚= ̃𝑐𝑙,
𝑙= 1, … , 𝑠,
(30)
G. Orlando et al.: Preprint submitted to Elsevier
Page 6 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
and by the weights ̃𝐛⊤= (̃𝑏1, ̃𝑏2, … , ̃𝑏𝑠
) and 𝐛⊤= (𝑏1, 𝑏2, … , 𝑏𝑠
) in ℝ𝑠. IMEX-RK methods can be represented in
the usual Butcher notation [23]
̃𝐜
̃𝐀
̃𝐛⊤
𝐜
𝐀
𝐛⊤.
Notice that the relation (30) is a usual assumption for Runge-Kutta methods [48].
It is useful to characterize the different IMEX-RK methods presented in the literature in two main types according
to the structure of the matrix of the DIRK method. Following [14], we have
Definition 3.1. An IMEX-RK method is said to be of type I [13, 75] if the matrix 𝐀is invertible. It is said to be of type
II [13, 55] if the matrix 𝐀can be written in the form
𝐀=
(
0
0
𝐚

)
,
with 𝐚= (𝑎21, … , 𝑎𝑠1)⊤∈ℝ𝑠−1 and the matrix ∈ℝ(𝑠−1)×(𝑠−1) is invertible. In the special case 𝐚= 0, 𝑏1 = 0, the
method is said of type ARS (see [8]) and the DIRK method is reducible to a method using 𝑠−1 stages.
Schemes of type II allow some simplifying assumptions, that make order conditions easier to treat for the
construction of higher order schemes [56]. On the other hand, schemes of type I are more suited to a theoretical
analysis [12, 16] because of the invertibility of 𝐀.
Definition 3.2. We call an IMEX-RK method stiffly accurate (SA) if the corresponding DIRK method is stiffly accurate,
namely [88]
𝑎𝑠𝑙= 𝑏𝑙,
𝑖= 1, … , 𝑠.
(31)
All the IMEX-RK schemes employed in this work for the numerical simulations are stiffly accurate (see Appendix A).
The asymptotic properties of IMEX-RK methods are strongly related to the L-stability of the implicit part of the
scheme. An implicit Runge-Kutta scheme is said to be L-stable [88] if it is A-stable and 𝑅(𝑧) →0 as 𝑧→∞, where 𝑅(𝑧)
is the stability function of the DIRK scheme. Following the result in [88], L-stability is typically obtained combining
the A-stability property with the SA property. However, for methods of type II, this combination does not necessarily
lead to a L-stable scheme for the implicit part, because the matrix 𝐀is not invertible [16]. For SA schemes of type II,
a supplementary condition is required to obtain L-stability, i.e. [16]
𝐞⊤
𝑠−1𝐚=
𝑠
∑
𝑚=2
̂𝑤𝑠𝑚𝑎𝑚1 = 0,
(32)
where 𝐞⊤
𝑠= (0, … , 0, 1)⊤and ̂𝑤𝑙𝑚denotes the elements of the inverse of . One can easily verify that all the implicit
companion methods reported in Appendix A are L-stable.
In the sequel, to identify the different IMEX-RK schemes, we shall use the notation (𝑠, 𝜎, 𝑝), where 𝑠is the number
of function evaluations of the implicit companion method, 𝜎is the number of function evaluations of the explicit
companion method, and 𝑝is the order of the IMEX scheme. In this work, we employ second, third, and fourth order
time discretization schemes (see Appendix A).
3.2. IMEX time discretization for the Euler equations
In this Section, we outline the IMEX time discretization for the Euler equations (3). Following [25, 36], we consider
an implicit treatment of the pressure gradient term within the momentum equation and of the pressure work term in the
energy equation, while the continuity equation is discretized in a fully explicit fashion. A generic stage reads therefore
as follows [71, 74]:
𝜌(𝑙)
=
𝜌𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅(𝐪)(𝑚)
G. Orlando et al.: Preprint submitted to Elsevier
Page 7 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝐪(𝑙) +
1
𝑀2 𝑎𝑙𝑙Δ𝑡∇𝑝(𝑙)
=
𝐪𝑛−Δ𝑡
𝑀2
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇𝑝(𝑚) −Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅(𝐪⊗𝐮)(𝑚)
(33)
(𝑙) + 𝑎𝑙𝑙Δ𝑡∇⋅(ℎ𝐪)(𝑙)
=
𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇⋅(ℎ𝐪)(𝑚) −Δ𝑡𝑀2
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅(𝑘𝐪)(𝑚) ,
where
𝐪(𝑙) = (𝜌𝐮)(𝑙)
𝐮(𝑙) = 𝐪(𝑙)
𝜌(𝑙)
(𝑙) = (𝜌𝐸)(𝑙).
(34)
Notice that, substituting formally 𝐪(𝑙) into the energy equation and taking into account the definitions 𝜌𝐸= 𝜌𝑒+𝑀2𝜌𝑘
and ℎ= 𝑒+ 𝑝∕𝜌, the following nonlinear Helmholtz-type equation for the pressure is obtained:
𝜌(𝑙) [𝑒(𝑝(𝑙), 𝜌(𝑙)) + 𝑀2𝑘(𝑙)] −𝑎2
𝑙𝑙
Δ𝑡2
𝑀2 ∇⋅
[(
𝑒(𝑝(𝑙), 𝜌(𝑙)) + 𝑝(𝑙)
𝜌(𝑙)
)
∇𝑝(𝑙)
]
+ 𝑎𝑙𝑙Δ𝑡∇⋅
[(
𝑒(𝑝(𝑙), 𝜌(𝑙)) + 𝑝(𝑙)
𝜌(𝑙)
)
𝐦(𝑙)
]
= ̂𝑒(𝑙),
(35)
where
𝐦(𝑙)
=
𝐪𝑛−Δ𝑡
𝑀2
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇𝑝(𝑚) −Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅(𝐪⊗𝐮)(𝑚) ,
(36a)
̂𝑒(𝑙)
=
𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇⋅(ℎ𝐪)(𝑚) −𝑀2Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅(𝑘𝐪)(𝑚) .
(36b)
Equation (35) is solved through a fixed point procedure [36, 71]. More specifically, setting 𝜉(0) = 𝑝(𝑙−1), 𝑘(𝑙,0) = 𝑘(𝑙−1),
one solves for ̃𝑘= 0, … , 𝐿the equation
𝜌(𝑙)𝑒(𝜉(̃𝑘+1), 𝜌(𝑙))
−
𝑎2
𝑙𝑙
Δ𝑡2
𝑀2 ∇⋅
[(
𝑒(𝜉(̃𝑘), 𝜌(𝑙)) + 𝜉(̃𝑘)
𝜌(𝑙)
)
∇𝜉(̃𝑘+1)
]
=
̂𝑒(𝑙) −𝑀2𝜌(𝑙)𝑘(𝑙,̃𝑘) −𝑎𝑙𝑙Δ𝑡∇⋅
[(
𝑒(𝜉(̃𝑘), 𝜌(𝑙)) + 𝜉(̃𝑘)
𝜌(𝑙)
)
𝐦(𝑙)
]
and then updates the velocity as
𝐮(𝑙,̃𝑘+1) + 𝑎𝑙𝑙Δ𝑡
𝜌(𝑙)𝑀2 ∇𝜉(̃𝑘+1) = 𝐦(𝑙)
𝜌(𝑙) .
As already discussed in [36], solving directly (35) keeping a full implicit treatment of the enthalpy as in a classical
Newton method yields a system strongly nonlinear and difficult to control. For this purpose, one adopts a Picard iteration
technique in which the contribution of the enthalpy is computed at the previous fixed point iteration so as to reduce
the nonlinearity of (35). Moreover, this choice is justified by the fact that two/three iterations are typically sufficient to
obtain a satisfactory solution, as already observed in [26, 36] and as further confirmed by our numerical experiments
(see in particular Section 4.2.1).
3.3. Semi-Implicit IMEX (SI-IMEX) time discretization for the Euler equations
The procedure outlined in the previous Section always requires the solution of a nonlinear system at each stage. In
this Section, we outline the semi-implicit IMEX (SI-IMEX) time discretization for the Euler equations (3) [15], which
provides similar results with less computational time. The governing partial differential equations (3) can be cast into
a compact and general form as
𝜕𝐔
𝜕𝑡= 𝐇(𝐔, 𝐔) .
(37)
G. Orlando et al.: Preprint submitted to Elsevier
Page 8 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Here 𝐔= (𝜌, 𝜌𝐮, 𝜌𝐸)⊤is the vector of conserved variables, and
𝐇(𝐔, 𝐔) = −∇⋅
⎛
⎜
⎜⎝
𝜌𝐮
𝜌𝐮⊗𝐮
𝑀2𝜌𝑘𝐮
⎞
⎟
⎟⎠
−∇⋅
⎛
⎜
⎜⎝
0
1
𝑀2 𝑝
ℎ𝜌𝐮
⎞
⎟
⎟⎠
,
(38)
where 𝐇∶ℝ𝑑+2 × ℝ𝑑+2 →ℝ𝑑+2 is a sufficiently regular mapping. Following [13], the governing partial differential
equations (3) are written under the form of an autonomous system (37), for all 𝑡> 𝑡0 with the initial condition
𝐔(𝑡0) = 𝐔0.
We refer to semi-implicit schemes as numerical methods that address problems of the form (37), wherein the
variable 𝐔, appearing as first argument of 𝐇, is treated explicitly and will be denoted by 𝐔𝐸, while the variable 𝐔
appearing as the second argument is treated implicitly and denoted by 𝐔𝐼. Thus, we obtain a partitioned system of the
form
𝜕𝐔𝐸
𝜕𝑡
= 𝐇(𝐔𝐸, 𝐔𝐼),
𝜕𝐔𝐼
𝜕𝑡
= 𝐇(𝐔𝐸, 𝐔𝐼).
(39)
with
𝐇(𝐔𝐸, 𝐔𝐼) = −∇⋅𝐅𝐸−∇⋅𝐅𝑆𝐼,
(40)
and
𝐅𝐸=
⎛
⎜
⎜⎝
(𝜌𝐮)𝐸
(𝜌𝐮⊗𝐮)𝐸
𝑀2(𝜌𝑘𝐮)𝐸
⎞
⎟
⎟⎠
,
𝐅𝑆𝐼=
⎛
⎜
⎜⎝
0
1
𝑀2 𝑝𝐼
ℎ𝐸(𝜌𝐮)𝐼
⎞
⎟
⎟⎠
.
(41)
Moreover, 𝐔𝐸= (𝜌𝐸, (𝜌𝐮)𝐸, (𝜌𝐸)𝐸
)⊤and 𝐔𝐼= (𝜌𝐼, (𝜌𝐮)𝐼, (𝜌𝐸)𝐼
)⊤. Subscripts 𝐸and 𝑆𝐼in (40) indicate the explicit
and semi-implicit treatment of the first and the second term, respectively. Notice that the number of unknowns in (39)
has been doubled. However, when specific time discretizations are chosen for autonomous systems, this doubling
is only apparent [13]. Finally, the kinetic energy in the total energy definition splits into an explicit and an implicit
contribution, namely:
(𝜌𝐸)𝐼= (𝜌𝑒)𝐼+ 𝑀2
2 𝐮𝐸⋅(𝜌𝐮)𝐼.
(42)
High-order time discretization is achieved making use of IMEX-RK schemes. More specifically, we adopt methods
for which ̃𝑏𝑙= 𝑏𝑙, 𝑙= 1, … , 𝑠. We observe that, since ̃𝑏𝑙= 𝑏𝑙, the numerical solutions are the same, i.e., if 𝐔0
𝐸= 𝐔0
𝐼,
then 𝐔𝑛
𝐸= 𝐔𝑛
𝐼for all 𝑛> 0. Hence, the duplication of the system is only apparent.
Under the assumption that system (39) is autonomous, a SI-IMEX-RK method is obtained as follows. First, set
𝐔𝑛
𝐸= 𝐔𝑛
𝐼= 𝐔𝑛. Then, the internal stage values read
𝐔(𝑙)
𝐸= 𝐔𝑛
𝐸+ Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚𝐇
(
𝐔(𝑚)
𝐸, 𝐔(𝑚)
𝐼
)
𝐔(𝑙)
𝐼= 𝐔𝑛
𝐼+ Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚𝐇
(
𝐔(𝑚)
𝐸, 𝐔(𝑚)
𝐼
)
+ Δ𝑡𝑎𝑙𝑙𝐇
(
𝐔(𝑙)
𝐸, 𝐔(𝑙)
𝐼
)
,
(43)
for 𝑙= 1, … , 𝑠. Finally, the numerical solution is updated with
𝐔𝑛+1 = 𝐔𝑛+ Δ𝑡
𝑠
∑
𝑙=1
𝑏𝑙𝐇
(
𝐔(𝑙)
𝐸, 𝐔(𝑙)
𝐼
)
.
(44)
G. Orlando et al.: Preprint submitted to Elsevier
Page 9 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
For the sake of clarity in the notation, we denote
𝐪= 𝜌𝐮
= 𝜌𝐸
as done in the previous Section. As an example, we present the first order semi-implicit scheme solving system (3) to
compute the numerical solution 𝐔𝑛+1 = (𝜌𝑛+1, 𝐪𝑛+1, 𝑛+1)⊤. We focus on the time discretization, while keeping the
space continuous. We consider the first order IMEX-RK scheme
0
0
1
1
1
1 .
Formally applying the above tableau to the partitioned system (39), it reads
𝐔(1)
𝐸= 𝐔𝑛
𝐔(1)
𝐼
= 𝐔𝑛+ Δ𝑡𝐇
(
𝐔𝑛, 𝐔(1)
𝐼
)
,
𝐔𝑛+1 = 𝐔(1)
𝐼,
(45)
and explicitly we get
𝐔𝑛+1 = 𝐔𝑛−Δ𝑡∇⋅𝐅𝐸(𝐔𝑛) −Δ𝑡∇⋅𝐅𝑆𝐼(𝐔𝑛, 𝐔𝑛+1)
(46)
with
𝑛+1 = (𝜌𝑒)𝑛+1 + 𝑀2
2 𝐮𝑛⋅𝐪𝑛+1.
A generic stage 𝑙for the Euler equations using a SI-IMEX-RK scheme reads therefore as follows.
Explicit step. Set
𝜌(𝑙)
𝐸
=
𝜌𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚∇⋅𝐪(𝑚)
𝐸
𝐪(𝑙)
𝐸
=
𝐪𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚
[
∇⋅(𝐪⊗𝐮)(𝑚)
𝐸+
1
𝑀2 ∇𝑝(𝑚)
𝐼
]
(47)
(𝑛,𝑙)
𝐸
=
𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
̃𝑎𝑙𝑚
[
𝑀2 ∇⋅(𝑘𝐪)(𝑚)
𝐸+ ∇⋅(ℎ𝐸𝐪𝐼
)(𝑚)]
.
Implicit step. Solve 𝐔(𝑙)
𝐼:
𝜌(𝑙)
𝐼
=
̄𝜌(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅𝐪(𝑙)
𝐸,
𝐪(𝑙)
𝐼+
1
𝑀2 𝑎𝑙𝑙Δ𝑡∇𝑝(𝑙)
𝐼
=
𝐦(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅(𝐪⊗𝐮)(𝑙)
𝐸
(48)
(𝑙)
𝐼+ 𝑎𝑙𝑙Δ𝑡∇⋅(ℎ𝐸𝐪𝐼
)(𝑙)
=
̂𝑒(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡𝑀2 ∇⋅(𝑘𝐪)(𝑙)
𝐸,
where
̄𝜌(𝑙)
𝐼
=
𝜌𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚∇⋅𝐪(𝑚)
𝐸
𝐦(𝑙)
𝐼
=
𝐪𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚
[
∇⋅(𝐪⊗𝐮)(𝑚)
𝐸+
1
𝑀2 ∇𝑝(𝑚)
𝐼
]
(49)
̂𝑒(𝑙)
𝐼
=
𝑛−Δ𝑡
𝑙−1
∑
𝑚=1
𝑎𝑙𝑚
[
𝑀2 ∇⋅(𝑘𝐪)(𝑚)
𝐸−∇⋅(ℎ𝐸𝐪𝐼
)(𝑚)]
.
G. Orlando et al.: Preprint submitted to Elsevier
Page 10 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
To solve system (48), we substitute 𝐪(𝑙)
𝐼in the energy equation, so as to obtain an elliptic equation for 𝑝(𝑙)
𝐼[15], which
reads as follows:
𝜌(𝑙)
𝐼𝑒𝐼(𝑝(𝑙)
𝐼, 𝜌(𝑙)
𝐼)
−
𝑎2
𝑙𝑙
Δ𝑡2
𝑀2 ∇⋅
(
ℎ(𝑙)
𝐸∇𝑝(𝑙)
𝐼
)
−1
2𝑎𝑙𝑙Δ𝑡𝐮(𝑙)
𝐸⋅∇𝑝(𝑙)
𝐼
=
̂𝑒(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡𝑀2 ∇⋅(𝑘𝐪)(𝑙)
𝐸−𝑀2
2 𝐮(𝑙)
𝐸⋅
[
𝐦(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅(𝐪⊗𝐮)(𝑙)
𝐸
]
−
𝑎𝑙𝑙Δ𝑡∇⋅
[
ℎ(𝑙)
𝐸
[
𝐦(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅(𝐪⊗𝐮)(𝑙)
𝐸
]]
.
(50)
Next, one computes 𝐪(𝑙)
𝐼and (𝑙)
𝐼from (48). Finally, one updates the numerical solution 𝐔𝑛+1 from (44).
3.4. Impact of the EOS
Equations (37) and (50) need the relation between the internal energy and the pressure before being solved. In the
case of the ideal gas law (4), since
𝜌𝑒=
1
𝛾−1𝑝,
equations (37) and (50) constitute a linear system for 𝜉(̃𝑘+1) and 𝑝(𝑙)
𝐼, respectively. Analogous considerations hold for
the SG-EOS (5), since
𝜌𝑒=
𝑝
𝛾−1 + 𝛾𝜋∞
𝛾−1 + 𝜌𝑞∞.
Hence, only a supplementary term depending on the already updated density is present.
On the other hand, the use of a more general equation of state, such as the general cubic EOS (6), leads to a
nonlinear relation between internal energy and pressure and therefore a nonlinear equation for the pressure should be
solved [21]. We rewrite the term 𝜌𝑒as 𝜌𝑒
𝑝𝑝, so that, following the discussion in [22], the nonlinear equation is solved
by the following Picard iteration
𝜌(𝑙)𝑒(𝜉(̃𝑘), 𝜌(𝑙))
𝜉(̃𝑘)
𝜉(̃𝑘+1)
−
𝑎2
𝑙𝑙
Δ𝑡2
𝑀2 ∇⋅
[(
𝑒(𝜉(̃𝑘), 𝜌(𝑙)) + 𝜉(̃𝑘)
𝜌(𝑙)
)
∇𝜉(̃𝑘+1)
]
=
̂𝑒(𝑙) −𝑀2𝜌(𝑙)𝑘(𝑙,̃𝑘) −𝑎𝑙𝑙Δ𝑡∇⋅
[(
𝑒(𝜉(̃𝑘), 𝜌(𝑙)) + 𝜉(̃𝑘)
𝜌(𝑙)
)
𝐦(𝑙)
]
.
(51)
It is worth noting that in the case of the ideal gas law (4),
𝜌(𝑙)𝑒(𝜉(̃𝑘), 𝜌(𝑙))
𝜉(̃𝑘)
𝜉(̃𝑘+1) =
1
𝛾−1𝜉(̃𝑘+1) = 𝜌(𝑙)𝑒(𝜉(̃𝑘+1)),
so that (51) reduces to (37). Notice also that (51) corresponds to a slightly different linearization with respect to the
one proposed in [71], which was tailored for the general cubic EOS, while (51) is applicable to a general EOS.
Analogous considerations hold for (50): one can rewrite
(𝜌𝑒)𝐼= (𝜌𝑒)𝐼
𝑝𝐼
𝑝𝐼
(52)
and solve the resulting mild nonlinear equation according to the Picard iteration described in [22]. Note that the fixed
point procedure proposed in [22] corresponds to a Newton-type method which can also be applied to non-differentiable
relations, like those that could be obtained for tabulated EOS, unlike the standard Newton method. We refer to [22] for
a detailed description and analysis of the algorithm. Nevertheless, in the case of the semi-implicit time discretization,
in order to avoid the solution of a nonlinear equation and the use of a fixed point loop, we approximate
(𝜌𝑒)𝐼
𝑝𝐼
𝑝𝐼≈(𝜌𝑒)𝐸
𝑝𝐸
𝑝𝐼,
(53)
G. Orlando et al.: Preprint submitted to Elsevier
Page 11 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
so that (50) modifies as
(𝜌𝑒)(𝑙)
𝐸
𝑝(𝑙)
𝐸
𝑝(𝑙)
𝐼
−
𝑎2
𝑙𝑙
Δ𝑡2
𝑀2 ∇⋅
(
ℎ(𝑙)
𝐸∇𝑝(𝑙)
𝐼
)
−1
2𝑎𝑙𝑙Δ𝑡𝐮(𝑙)
𝐸⋅∇𝑝(𝑙)
𝐼
=
̂𝑒(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡𝑀2 ∇⋅(𝑘𝐪)(𝑙)
𝐸−𝑀2
2 𝐮(𝑙)
𝐸⋅
[
𝐦(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅(𝐪⊗𝐮)(𝑙)
𝐸
]
−
𝑎𝑙𝑙Δ𝑡∇⋅
[
ℎ(𝑙)
𝐸
[
𝐦(𝑙)
𝐼−𝑎𝑙𝑙Δ𝑡∇⋅(𝐪⊗𝐮)(𝑙)
𝐸
]]
.
(54)
It is to be noted that, in the case of the ideal gas law (4),
(𝜌𝑒)(𝑙)
𝐸
𝑝(𝑙)
𝐸
𝑝(𝑙)
𝐼=
1
𝛾−1𝑝(𝑙)
𝐼= (𝜌𝑒)(𝑙)
𝐼,
so that (54) reduces to (50).
3.5. The spatial discretization strategy
In this Section, we briefly outline the spatial discretization adopted for (33) and (47)-(48), which is based on the
Discontinuous Galerkin (DG) method [40] as implemented in the deal.II library [3, 9]. We consider a decomposition
of the domain Ω into a family of quadrilaterals ℎand denote each element by 𝐾. The skeleton = 𝐼∪𝐵denotes
the set of all the element faces, with 𝐼and 𝐵being the subset of interior and boundary faces, respectively. A face
Γ ∈𝐼shares two elements, 𝐾+ with outward unit normal 𝐧+, and 𝐾−with outward unit normal 𝐧−, while we simply
denote by 𝐧the outward unit normal for a face Γ ∈𝐵(see Figure 1). For a scalar function 𝜑, the jump is defined as
[[𝜑]] = 𝜑+𝐧+ + 𝜑−𝐧−if Γ ∈𝐼
[[𝜑]] = 𝜑𝐧if Γ ∈𝐵,
(55)
while the average reads
{{𝜑}} = 1
2
(𝜑+ + 𝜑−) if Γ ∈𝐼
{{𝜑}} = 𝜑if Γ ∈𝐵.
(56)
Analogous definitions apply for a vector function 𝝋. More specifically, we define
[[𝝋]] = 𝝋+ ⋅𝐧+ + 𝝋−⋅𝐧−if Γ ∈𝐼
[[𝝋]] = 𝝋⋅𝐧if Γ ∈𝐵
(57a)
{{𝝋}} = 1
2
(𝝋+ + 𝝋−) if Γ ∈𝐼
{{𝝋}} = 𝝋if Γ ∈𝐵.
(57b)
Finally, for vector functions, it is also useful to define a tensor jump as follows:
⟨⟨𝝋⟩⟩= 𝝋+ ⊗𝐧+ + 𝝋−⊗𝐧−if Γ ∈𝐼
⟨⟨𝝋⟩⟩= 𝝋⊗𝐧if Γ ∈𝐵.
(58)
We also introduce the following finite element spaces
𝑄𝑟= {𝑣∈𝐿2(Ω) ∶𝑣|𝐾∈ℚ𝑟
∀𝐾∈
}
𝐕𝑟= [𝑄𝑟
]𝑑,
where ℚ𝑟is the space of polynomials of degree 𝑟in each coordinate direction. We then denote by 𝝋𝑖(𝐱) the basis
functions for the space 𝐕𝑟and by 𝜓𝑖(𝐱) the basis functions for the space 𝑄𝑟, the finite element spaces chosen for the
discretization of the velocity and of the pressure (as well as the density), respectively, so that
𝐮≈
|ℎ|(𝑟+1)𝑑
∑
𝑗=1
𝑢𝑗(𝑡)𝝋𝑗(𝐱)
𝑝≈
|ℎ|(𝑟+1)𝑑
∑
𝑗=1
𝑝𝑗(𝑡)𝜓𝑗(𝐱).
Here ||ℎ|| denotes the number of elements of the computational mesh. Recall that 𝑑denotes the space dimension. The
number of degrees of freedom for scalar variable is indeed equal to ||ℎ|| (𝑟+ 1)𝑑[40]. The shape functions correspond
to the products of Lagrange interpolation polynomials for the support points of (𝑟+ 1)-order Gauss-Lobatto quadrature
G. Orlando et al.: Preprint submitted to Elsevier
Page 12 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 1: Example of two neighboring elements for a nodal DG formulation based on Lagrange polynomials. The nodes
correspond to the support points of (𝑟+ 1)-order Gauss-Lobatto quadrature rule (in the image 𝑟= 1).
rule in each coordinate direction (Figure 1). In particular, we have grid points at the boundaries of the elements, where
the solution can be discontinuous and this simplifies the evaluation of the integrals at the boundary itself [60]. Hence,
for any given edge, the only shape functions with non-zero values are exactly those whose node points are located on
that edge [60].
Given these definitions, the weak formulation for the momentum equation at each stage (33) reads as follows
[71, 74]:
𝐀(𝑙)𝐔(𝑙) + 𝐁(𝑙)𝐏(𝑙) = 𝐅(𝑛,𝑙),
(59)
with 𝐔(𝑙) denoting the vector of the degrees of freedom associated to the velocity field and 𝐏(𝑙) denoting the vector of
the degrees of freedom associated to the pressure. Here we have set
𝐴(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
𝜌(𝑙)𝝋𝑗⋅𝝋𝑖𝑑Ω
(60)
𝐵(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
−𝑎𝑙𝑙
Δ𝑡
𝑀2 ∇⋅𝝋𝑖Ψ𝑗𝑑Ω +
∑
Γ∈∫Γ
𝑎𝑙𝑙
Δ𝑡
𝑀2
{{Ψ𝑗
}} [[𝝋𝑖
]] 𝑑Σ
(61)
𝐹(𝑙)
𝑖
=
∑
𝐾∈ℎ∫𝐾
𝜌𝑛𝐮𝑛⋅𝝋𝑖𝑑Ω
+
𝑙−1
∑
𝑚=1
∑
𝐾∈ℎ∫𝐾
̃𝑎𝑙𝑚Δ𝑡(𝜌(𝑚)𝐮(𝑚) ⊗𝐮(𝑚)) ∶∇𝝋𝑖𝑑Ω +
𝑙−1
∑
𝑚=1
∑
𝐾∈ℎ∫𝐾
𝑎𝑙𝑚
Δ𝑡
𝑀2 𝑝(𝑚) ∇⋅𝝋𝑖𝑑Ω
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
̃𝑎𝑙𝑚Δ𝑡{{𝜌(𝑚)𝐮(𝑚) ⊗𝐮(𝑚)}} ∶⟨⟨𝝋𝑖⟩⟩𝑑Σ
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
̃𝑎𝑙𝑚Δ𝑡𝜆(𝑚)
2
⟨⟨𝜌(𝑚)𝐮(𝑚)⟩⟩∶⟨⟨𝝋𝑖⟩⟩𝑑Σ
−
𝑙−1
∑
𝑚=1
∑
Γ∈∫Γ
𝑎𝑙𝑚
Δ𝑡
𝑀2
{{𝑝(𝑚)}} [[𝝋𝑖
]] 𝑑Σ.
(62)
Following the discussion in [71, 74], one can notice that a centered flux is employed for the quantities defined implicitly,
while an upwind-biased flux is adopted for the quantities computed explicitly. Moreover, following [1, 74], in order to
obtain a numerical method effective for a wider range of Mach numbers, we take
𝜆(𝑚) = max
[
𝑓
(
𝑀+,(𝑚)
𝑙𝑜𝑐
) (|||𝐮+,(𝑛,𝑚)||| + 1
𝑀𝑐+,(𝑚))
, 𝑓
(
𝑀−,(𝑚)
𝑙𝑜𝑐
) (|||𝐮−,(𝑚)||| + 1
𝑀𝑐−,(𝑚))]
,
(63)
G. Orlando et al.: Preprint submitted to Elsevier
Page 13 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
with 𝑀±,(𝑚)
𝑙𝑜𝑐
= 𝑀|𝐮|±,(𝑚)
𝑐±,(𝑚) and 𝑓(𝑀𝑙𝑜𝑐
) = min (1, 𝑀𝑙𝑜𝑐
). This choice corresponds to the convex combination between a
centered flux and a Rusanov flux [80] as proposed in [1], so that, for 𝑀𝑙𝑜𝑐≥1, we resort to a Rusanov flux, whereas for
𝑀𝑙𝑜𝑐≪1, only the local fluid velocity is relevant for the numerical dissipation. Further considerations on the numerical
flux will be discussed at the end of the Section. The numerical integration is based on the so-called over-integration or
consistent integration, so as to guarantee exact integration. In particular, we employ 2𝑟+1 Gauss-Legendre quadrature
points along each coordinate direction [73]. Analogously, the energy equation in (33) can be expressed as
𝐂(𝑙)𝐔(𝑙) + 𝐃(𝑙)𝐏(𝑙) = 𝐆(𝑙).
(64)
For the sake of completeness, as well as to point out the contribution due to the novel strategy presented in Section 3.4
to handle a generic EOS, we report the expression of the components 𝐂(𝑙) and 𝐃(𝑙). The expression of 𝐆(𝑙) can be easily
inferred from (33) and its definition entails that centered fluxes are employed for the quantities defined implicitly, while
an upwind-biased flux is used for the quantities computed explicitly (see also [74]). Hence, we obtain
𝐶(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
−𝑎𝑙𝑙Δ𝑡ℎ(𝑙)𝜌(𝑙)𝝋𝒋⋅∇Ψ𝑖𝑑Ω +
∑
Γ∈∫Γ
𝑎𝑙𝑙Δ𝑡{{ℎ(𝑙)𝜌(𝑙)𝝋𝒋
}} ⋅[[Ψ𝑖
]] 𝑑Σ
(65)
𝐷(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
(𝜌𝑒)(𝑙)
𝑝(𝑙) Ψ𝑗Ψ𝑖𝑑Ω
(66)
Formally, we can derive
𝐔(𝑙) = (𝐀(𝑙))−1 (𝐅(𝑙) −𝐁(𝑙)𝐏(𝑙)) ,
(67)
so as to obtain
𝐃(𝑙)𝐏(𝑙) + 𝐂(𝑙) (𝐀(𝑙))−1 (𝐅(𝑙) −𝐁(𝑙)𝐏(𝑙)) = 𝐆(𝑙).
(68)
The above system is then solved following the fixed point procedure described in [22, 36, 71]. More specifically, setting
𝐔(𝑙,0) = 𝐔(𝑙−1), 𝐏(𝑙,0) = 𝐏(𝑙−1), one solves for ̃𝑘= 0, … , 𝐿
(
𝐃(𝑙,̃𝑘) −𝐂(𝑙,̃𝑘) (𝐀(𝑙))−1 𝐁(𝑙))
𝐏(𝑙,̃𝑘+1) = 𝐆(𝑙,̃𝑘) −𝐂(𝑙,̃𝑘) (𝐀(𝑙))−1 𝐅(𝑙)
(69)
and then updates the velocity solving
𝐀(𝑙)𝐔(𝑙,̃𝑘) = 𝐅(𝑙) −𝐁(𝑙)𝐏(𝑙,̃𝑘+1).
(70)
The algebraic formulation associated to (54) is obtained substituting the degrees of freedom of the velocity into the
algebraic formulation of the energy equation. Relations (67) and (68) can be therefore employed to achieve this goal.
For the sake of completeness, we report the new definitions of 𝐂(𝑙) and 𝐃(𝑙), while analogous modifications apply to
the other variables. Hence, 𝐂(𝑙) and 𝐃(𝑙) now read as follows:
𝐶(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
−𝑎𝑙𝑙Δ𝑡ℎ(𝑙)
𝐸𝜌(𝑙)
𝐼𝝋𝒋⋅∇Ψ𝑖𝑑Ω
+
∑
Γ∈∫Γ
𝑎𝑙𝑙Δ𝑡
{{
ℎ(𝑙)
𝐸𝜌(𝑙)
𝐼𝝋𝒋
}}
⋅∇Ψ𝑖𝑑Σ +
∑
𝐾∈ℎ∫𝐾
𝑀2
2 𝐮(𝑙)
𝐸⋅𝜌(𝑙)
𝐼𝝋𝒋Ψ𝑖𝑑Ω
(71)
𝐷(𝑙)
𝑖𝑗
=
∑
𝐾∈ℎ∫𝐾
(𝜌𝑒)(𝑙)
𝐸
𝑝(𝑙)
𝐸
Ψ𝑗Ψ𝑖𝑑Ω.
(72)
A matrix-free approach is employed [3], meaning that no global sparse matrix is built and only the action of the
linear operators on a vector is actually implemented. Matrices 𝐀(𝑙) and 𝐃(𝑙) are symmetric and positive definite, while
the matrix 𝐂(𝑙) (𝐀(𝑙))−1 𝐁(𝑙) is not symmetric. We point out that if one directly discretizes (35) and (50), as done, e.g.,
in [21, 78], a symmetric positive definite linear system can be obtained. However, this approach implies the direct
G. Orlando et al.: Preprint submitted to Elsevier
Page 14 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
numerical solution of an elliptic equation and the discretization of a second order operator that, in the framework of a
DG method, would require, e.g., the use of the Symmetric Interior Penalty method [4]. The use of a Schur complement
type technique, as the one described in this work, allows one to employ only the standard numerical fluxes of hyperbolic
problems (Rusanov and upwind-biased in this work), without defining and setting penalization constants typical of the
aforementioned numerical strategy for elliptic equations. A comparison between the approach employed in this work
and the direct solution of the Helmholtz-type equations (35) and (50) will be matter of future work. In view of these
considerations, a preconditioned conjugate gradient method with a geometric multigrid preconditioner is employed to
solve the symmetric positive definite linear systems. The GMRES solver with a Jacobi preconditioner is employed for
the solution of the non-symmetric linear systems. In future developments, we aim to implement and employ multigrid
preconditioners also for the non-symmetric linear systems in the matrix-free framework so as to further improve the
performance of the solver.
The DG method naturally allows for high-order accuracy without the use of reconstructions which involve large
stencils. However, as discussed in [52], its accuracy in the very low Mach regime depends on the numerical flux and
on the shape of the elements. More specifically, a simplicial mesh is needed to establish low Mach accuracy. The lack
of accuracy in the very low Mach limit can lead to a numerical scheme which is not convergent with a finite volume
scheme, while an order reduction is observed in the case of the Discontinuous Galerkin method [52]. A low Mach fix
for the Euler equations resolved with the finite volume method on Cartesian grids was proposed in [10]. Moreover, it
is known that an upwind scheme fails to solve very subsonic flows [46]. However, for moderate low Mach numbers,
i.e. 𝑀> 10−4, the convex combination (63) leads to a correct scaling of the pressure fluctuations [1, 74] and the low
Mach number inaccuracy is counterbalanced by the high-order nature of the numerical scheme [52, 74].
A detailed analysis of the spatial discretization and the development of possible remedies for very subsonic flows
goes beyond the scope of the present work and will subject of future developments (see also Sections 4.1 and 4.2).
Since the main focus of this work is the comparison of the time discretization methods presented in Section 3.2 and
3.3, we believe that considering a minimum Mach number around 10−3 −10−4 allows us to perform this analysis in the
low Mach limit and compares well with the minimum Mach number chosen to analyze asymptotic-preserving schemes
in the literature, see, e.g., [1, 19, 18, 32, 67, 85].
4. Numerical results
The numerical methods outlined in Section 3 are now validated in a number of relevant benchmarks. The
implementation is carried out in the framework of the deal.II library [3, 9], that is a C++ open-source software
supporting the creation of finite element codes. Several libraries based on deal.II have been developed in the last
years [2, 45, 59]. All the simulations are performed in double precision. The employed time discretization schemes are
reported in Appendix A. Discrete parameter choices are associated to two Courant numbers, one based on the speed
of sound denoted by 𝐶, the so-called acoustic Courant number, and one based on the local velocity of the flow, the
so-called advective Courant number, denoted by 𝐶𝑢:
𝐶= 1
𝑀𝑟𝑐Δ𝑡

√
𝑑
𝐶𝑢= 𝑟𝑢Δ𝑡

√
𝑑.
(73)
Here, = min {diam(𝐾)|𝐾∈ℎ
}, 𝑟is the polynomial degree employed for the spatial discretization, 𝑐is the speed
of sound, and 𝑢is the magnitude of the flow velocity. Recall that 𝑑denotes the space dimension. For what concerns
the tests with the ideal gas law, 𝛾= 1.4 is employed in (4). Finally, following [71], the fixed point loop (37) is stopped
at the iteration ̃𝑘for which the maximum relative difference for the pressure is below a tolerance 𝜂, namely
‖‖‖𝜉(̃𝑘) −𝜉(̃𝑘−1)‖‖‖∞
‖‖‖𝜉(̃𝑘)‖‖‖∞
< 𝜂.
(74)
4.1. Taylor-Green vortex
As a first benchmark to verify the scaling properties of the numerical methods with respect to the Mach number
𝑀, we consider the Taylor-Green vortex [29, 90], that represents an exact steady solution of the incompressible Euler
equations. The initial condition in non-dimensional variables reads as follows:
𝜌(𝐱, 0) = 1
𝐮(𝐱, 0) =
(
sin(𝑥) cos(𝑦)
−cos(𝑥) sin(𝑦)
)
𝑝= 1 + 1
4𝑀2 (cos(2𝑥) + cos(2𝑦)) .
(75)
G. Orlando et al.: Preprint submitted to Elsevier
Page 15 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
The computational domain is Ω = (0, 2𝜋)2 endowed with periodic boundary conditions. The time step is such that the
maximum advective Courant number is 𝐶𝑢≈0.095, while the maximum acoustic Courant number is 𝐶≈0.11∕𝑀.
We employ 𝜂= 10−10.
First, we consider the IMEX-RK(3,3,3) scheme of type II (Table 21) in combination with polynomial degree 𝑟= 2
and 𝑁𝑒𝑙= 60 elements along each coordinate direction. We employ the IMEX-DG method. One can easily notice that
pressure fluctuations scale as (𝑀2) up to 𝑀= 10−4, as expected, whereas the density fluctuations scale as (𝑀2)
up to 𝑀= 10−3 and a degradation is experienced at 𝑀= 10−4. This degradation is likely related to the low Mach
inaccuracy of the DG method on quadrilateral cells discussed in Section 3.5. For what concerns the divergence of the
velocity field, it scales as (2) (see the convergence analysis in Section 4.2), but it does not vanish as 𝑀→0
(Table 1). As discussed in [74], since the initial velocity field is solenoidal and the vortex is stationary, a quadratic
convergence with respect to 𝑀is expected for the divergence of the velocity field. This result is dependent on the
spatial discretization. Indeed, since our method employs a standard nodal DG method, the divergence-free property
is not imposed pointwise and the error associated to ∇⋅𝐮is basically constant in time and it is therefore related to
the interpolation of the initial datum into the employed finite element space (Figure 2). A quadratic convergence with
respect to 𝑀was recently obtained in [90] and preliminary results in our framework suggest that the use of Raviart-
Thomas finite elements [5] for the velocity field improves the scaling properties as 𝑀→0. As already discussed at
the end of Section 3.5, a more detailed analysis of the spatial discretization is currently under investigation, while the
primary goal of the present work is to perform a quantitative comparison between two different IMEX-RK approaches
for the Euler equations. However, further considerations about the spatial discretization will be added at the end of the
upcoming Section 4.2.
Next, we consider the SI-IMEX-DG method. A stable solution is obtained up to 𝑀= 10−3 and the density
fluctuations scale as (𝑀2) only up to 𝑀= 10−2 (Table 2). This degradation is again likely mainly related to an
early manifestation of the low Mach inaccuracy of the DG method on quadrilateral cells. However, we point out that
some issues in the low Mach regime employing schemes of type II for the semi-implicit time discretization were already
experienced in [15]
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm 𝛿𝜌
Rate 𝛿𝜌
𝐿2 norm 𝛿𝑝
Rate 𝛿𝑝
10−1
5.76 × 10−3
1.38 × 10−3
1.57 × 10−2
10−2
1.82 × 10−3
0.5
1.57 × 10−5
1.9
1.57 × 10−4
2.0
10−3
1.82 × 10−3
−
1.58 × 10−7
2.0
1.57 × 10−6
2.0
10−4
1.82 × 10−3
−
1.24 × 10−8
1.1
1.62 × 10−8
2.0
Table 1
Mach number scaling of the density and pressure fluctuations and of the divergence of the velocity field for the Taylor-Green
vortex test case. Here, and in the following Tables, 𝛿𝜌= 𝜌−1 and 𝛿𝑝= 𝑝−1. The results are obtained using the IMEX
method with the IMEX-RK(3,3,3) scheme of type II in Table 21 together with polynomial degree 𝑟= 2 and 𝑁𝑒𝑙= 60.
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm 𝛿𝜌
Rate 𝛿𝜌
𝐿2 norm 𝛿𝑝
Rate 𝛿𝑝
10−1
5.76 × 10−3
1.38 × 10−3
1.57 × 10−2
10−2
1.82 × 10−3
0.5
1.61 × 10−5
1.9
1.57 × 10−4
2.0
10−3
1.82 × 10−3
−
4.28 × 10−6
0.6
1.57 × 10−6
2.0
Table 2
Mach number scaling of the density and pressure fluctuations and of the divergence of the velocity field for the Taylor-Green
vortex test case. The results are obtained using the SI-IMEX method with the IMEX-RK(3,3,3) scheme of type II in Table
21 together with polynomial degree 𝑟= 2 and 𝑁𝑒𝑙= 60.
Next, we consider the IMEX-RK(4,3,3) scheme of type I (Table 20). For what concerns the IMEX method,
analogous results with respect to the scheme of type II are obtained up to 𝑀= 10−3 with an improvement of the
scaling of the density fluctuations at 𝑀= 10−4 (Table 3). For what concerns the SI-IMEX method, a stable solution
is established at 𝑀= 10−4 with a stagnation of the density fluctuations (Table 4). These results confirm the superior
G. Orlando et al.: Preprint submitted to Elsevier
Page 16 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 2: Taylor-Green vortex test case, time evolution of the divergence of the velocity at 𝑀= 10−3. The results are
obtained using the IMEX method with the IMEX-RK(3,3,3) scheme of type II in Table 21 together with polynomial degree
𝑟= 2 and 𝑁𝑒𝑙= 60.
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm 𝛿𝜌
Rate 𝛿𝜌
𝐿2 norm 𝛿𝑝
Rate 𝛿𝑝
10−1
5.76 × 10−3
1.38 × 10−3
1.57 × 10−2
10−2
1.82 × 10−3
0.5
1.58 × 10−5
1.9
1.57 × 10−4
2.0
10−3
1.82 × 10−3
−
1.69 × 10−6
1.0
1.57 × 10−6
2.0
10−4
1.82 × 10−3
−
1.68 × 10−6
−
1.63 × 10−8
2.0
Table 4
Mach number scaling of the density and pressure fluctuations and of the divergence of the velocity field for the Taylor-Green
vortex test case. The results are obtained using the SI-IMEX method with the IMEX-RK(4,3,3) scheme of type I in Table
20 together with polynomial degree 𝑟= 2 and 𝑁𝑒𝑙= 60.
stability of schemes of type I with respect to schemes of type II for low Mach numbers flows already experienced in
[15]. Moreover, we can infer that the low Mach accuracy is influenced also by the time discretization strategy and by
the time discretization scheme.
𝑀
𝐿2 norm ∇⋅𝐮
Rate ∇⋅𝐮
𝐿2 norm 𝛿𝜌
Rate 𝛿𝜌
𝐿2 norm 𝛿𝑝
Rate 𝛿𝑝
10−1
5.76 × 10−3
1.38 × 10−3
1.57 × 10−2
10−2
1.82 × 10−3
0.5
1.58 × 10−5
1.9
1.57 × 10−4
2.0
10−3
1.82 × 10−3
−
1.58 × 10−7
2.0
1.57 × 10−6
2.0
10−4
1.82 × 10−3
−
4.66 × 10−9
1.5
1.63 × 10−8
2.0
Table 3
Mach number scaling of the density and pressure fluctuations and of the divergence of the velocity field for the Taylor-Green
vortex test case. The results are obtained using the IMEX method with the IMEX-RK(4,3,3) scheme of type I in Table 20
together with polynomial degree 𝑟= 2 and 𝑁𝑒𝑙= 60.
G. Orlando et al.: Preprint submitted to Elsevier
Page 17 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
4.2. Traveling vortex at low Mach
Next, we consider for an ideal gas a two-dimensional traveling vortex inspired by the inviscid isentropic vortex
studied, e.g., in [71, 84, 91]. For this test, a time-dependent analytic solution is available and the convergence properties
of a numerical scheme can be therefore assessed. The exact solution is indeed a propagation of the initial condition at
the background velocity
𝜌(𝐱, 𝑡) = 𝜌(𝐱−𝐮∞𝑡, 0)
𝐮(𝐱, 𝑡) = 𝐮(𝐱−𝐮∞𝑡, 0)
𝑝(𝐱, 𝑡) = 𝑝(𝐱−𝐮∞𝑡, 0).
Notice that a different version of this test case, which allows for a steady solution, was employed in [18, 21, 74].
Following [91], in order to emphasize the role of the Mach number 𝑀, we define the perturbation as
𝛿𝑇= 1 −𝛾
8𝛾𝜋2 𝑀2𝛽2𝑒1−̃𝑟2,
(76)
with
̃𝑟2 = (𝑥−𝑥0)2 + (𝑦−𝑦0)2
𝑟2
0
denoting the scaled radial coordinate and 𝛽being the vortex strength. We set 𝛽= 10, 𝑟0 = 1, and
𝜌(𝐱, 0) = (1 + 𝛿𝑇)
1
𝛾−1
𝑝(𝐱, 0) = 1 + 𝑀2 (1 + 𝛿𝑇)
𝛾
𝛾−1 = 1 + 𝑀2𝜌𝛾.
(77)
For what concerns the velocity, we define its perturbation as
𝛿𝐮= 𝛽𝑀
(−(𝑦−𝑦0
)
(𝑥−𝑥0
)
)
𝑒
1
2
(1−̃𝑟2)
2𝜋
(78)
Finally, we set 𝐮∞= [1, 1]⊤, whereas the final time is 𝑇𝑓= 3. To avoid problems related to the definition of boundary
conditions, we choose a sufficiently large domain Ω = (−10, 10)2 and periodic boundary conditions. The behavior of
the numerical methods is investigated at different Mach numbers. More specifically, we consider 𝑀∈[10−4, 10−1].
The tolerance in (74) is set to 𝜂= 10−10.
First, we analyze the results obtained employing the IMEX-RK(2,2,2) scheme (Table 19) with polynomial degree
𝑟= 1 for the space discretization. The time step is chosen in such a way that the maximum advective Courant number is
𝐶𝑢≈0.16 and the maximum acoustic Courant number is 𝐶≈0.12∕𝑀. One can easily notice that IMEX and SI-IMEX
time discretization methods yield the same level of accuracy (Tables 5-6). A degradation is experienced at 𝑀= 10−2
and analogous results are obtained for lower values of 𝑀. Contour plots of the pressure perturbation show that the
shape of the vortex is not preserved (Figure 3). This is likely related to well known issues of collocated finite element
type discretization on quadrilateral meshes in the low Mach regime [52, 79] (see also the discussion at the end of
Section 3.5). A simple workaround consists of increasing the polynomial degree of the finite element space employed
for the discretization of the velocity field. We also refer to [53], where a space enrichment for the velocity has been
used in the framework of a finite volume scheme. One can easily notice that, if we take 𝑟= 2 for the finite element
space of the velocity, the shape of the vortex is preserved (Figure 4) and the expected second order convergence is
established (Table 7).
Next, we focus on third order time discretization schemes (Tables 20-21) in combination with polynomial degree
𝑟= 2. The time step is such that the maximum advective Courant number is 𝐶𝑢≈0.094, while the maximum acoustic
Courant number is 𝐶≈0.07∕𝑀. All the schemes provide a similar level of accuracy and the expected third order
convergence rate is established (Tables 8-11) up to 𝑀= 10−3, except for the SI-IMEX method in combination with
the scheme of type II at 𝑀= 10−3. As already remarked in Section 4.1, some issues in the low Mach regime employing
schemes of type II for the semi-implicit time discretization were already experienced in [15]. Moreover, for 𝑀= 10−4,
a stable solution for the SI-IMEX method is obtained only employing the scheme of type I. This further confirms the
superior stability of schemes of type I with respect to schemes of type II for low Mach numbers when using the
SI-IMEX method.
A saturation of the error is experienced at 𝑀= 10−4. Analogous results with a saturation of the error at 𝑀= 10−4
were obtained in [91]. Since in most low Mach number applications the main interest lies on the velocity field rather
G. Orlando et al.: Preprint submitted to Elsevier
Page 18 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
40
1.28 × 10−1
1.33 × 10−1
80
3.99 × 10−2
1.7
5.18 × 10−2
1.3
160
1.02 × 10−2
2.1
2.27 × 10−2
𝟏.𝟐
320
2.03 × 10−3
𝟐.𝟏
Table 5
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(2,2,2) scheme (Table
19) and polynomial degree 𝑟= 1. Here, and in the following Tables, 𝑁𝑒𝑙denotes the number of elements along each
direction.
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
40
1.28 × 10−1
1.32 × 10−1
80
3.99 × 10−2
1.7
5.18 × 10−2
1.3
160
1.02 × 10−2
2.0
2.27 × 10−2
𝟏.𝟐
320
2.03 × 10−3
𝟐.𝟑
Table 6
Convergence analysis for the traveling vortex test case using the SI-IMEX method with the IMEX-RK(2,2,2) scheme (Table
19) and polynomial degree 𝑟= 1.
𝑁𝑒𝑙
𝑀= 10−2
𝐿2 relative error 𝛿𝐮
𝐿2 rate 𝛿𝐮
20
2.48 × 10−2
40
3.08 × 10−3
3.0
80
4.86 × 10−4
2.7
160
9.30 × 10−5
𝟐.𝟒
Table 7
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(2,2,2) scheme (Table
19), polynomial degree 𝑟= 2 for the velocity and 𝑟= 1 for the remaining variables.
than on the acoustics, we notice, similarly to [91], that no visible spurious effect arises in the velocity field in spite
of the order reduction (Figure 5). As already discussed in Section 3.5, the DG method on quadrilateral cells becomes
low Mach inaccurate for 𝑀< 10−4 and therefore this order reduction is likely related to a manifestation of this
phenomenon. Moreover, round-off errors are dominant in this configuration [91] and the use of quadruple precision is
crucial to reach really very low Mach numbers [20]. However, it is worth to notice that in the case of the steady vortex,
a correct scaling was established at 𝑀= 10−4 using the third order method of type II in [74]. Hence, the saturation of
the error could be also related to the well-known order reduction phenomenon experienced for stiff ODE’s problems
[88]. In future work, as already mentioned, we aim to analyze the behavior employing a spatial discretization based on
compatible finite elements, as recently done in [90], so as to guarantee that the initial condition of the velocity field
is setup as the discrete derivative of a vector potential, and to use simplicial meshes [52, 90] or Voronoi meshes [18],
which have been shown to be low Mach accurate for a steady vortex.
The IMEX method described in Section 3.2 allows the use of a larger time step ensuring the same level of accuracy,
because of the superior stability provided by the fixed point loop [36]. In particular, the time step can be doubled,
yielding a maximum advective Courant number 𝐶𝑢≈0.19. In spite of the use of a smaller time step, the SI-IMEX
G. Orlando et al.: Preprint submitted to Elsevier
Page 19 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 3: Traveling vortex test case with the IMEX-RK(2,2,2) scheme (Table 19) and polynomial degree 𝑟= 1, contour
plots of the pressure perturbation 𝑝−(1 −𝑀2) at 𝑀= 10−2 with 𝑁𝑒𝑙= 160. Left: initial field. Right: field at final time
𝑡= 𝑇𝑓= 3.
Figure 4: Traveling vortex test case with the IMEX-RK(2,2,2) scheme (Table 19), 𝑟= 2 for the velocity field and 𝑟= 1 for
the remaining variables, contour plots of the pressure perturbation 𝑝−(1 + 𝑀2) at 𝑀= 10−2 with 𝑁𝑒𝑙= 160. Left: initial
field. Right: field at final time 𝑡= 𝑇𝑓= 3.
method and, more specifically, the SI-IMEX time discretization using the scheme of type II, provides in general better
computational performance (Figure 6). As already discussed, the scheme of type I is more robust for 𝑀= 10−3 and
it becomes also more efficient as the spatial resolution increases. In the low Mach regime, differences in terms of
computational cost between the IMEX method and the SI-IMEX method are significantly smaller: the efficiency gain
reduces to around 5% for 𝑀= 10−3, meaning that the two methods are essentially equivalent in this respect. We will
further discuss this point in Section 4.2.1.
Next, we employ the fourth order time discretization schemes of type ARS (Table 22) and of type II (Table 23)
in combination with polynomial degree 𝑟= 3. The time step is such that the maximum advective Courant number is
𝐶𝑢≈0.05 and the maximum acoustic Courant number is 𝐶≈0.035∕𝑀. The SI-IMEX method requires a smaller
time step for 𝑀≤10−2. More specifically, an advective Courant number 𝐶𝑢≈0.025 is required to achieve a stable
solution. One can easily notice that the expected convergence rates are established up to 𝑀= 10−3 (Tables 12-13),
except for the IMEX method at 𝑀= 10−2, for which an order reduction is experienced. Analogous considerations to
those reported for the third order discretization schemes for 𝑀≤10−4 are valid. In particular, the correct scaling at
𝑀= 10−4 for the steady vortex using the IMEX method with the scheme of type ARS was obtained in [74], so that the
G. Orlando et al.: Preprint submitted to Elsevier
Page 20 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
20
3.12 × 10−2
3.55 × 10−2
3.55 × 10−2
3.59 × 10−2
40
2.40 × 10−3
3.7
2.32 × 10−3
3.9
2.32 × 10−3
3.9
5.51 × 10−3
2.7
80
3.17 × 10−4
2.9
2.92 × 10−4
3.0
2.90 × 10−4
3.0
5.71 × 10−3
−
160
4.13 × 10−5
𝟐.𝟗
3.73 × 10−5
𝟑.𝟎
4.61 × 10−5
𝟐.𝟕
Table 8
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(3,3,3) scheme of type
II (Table 21) and polynomial degree 𝑟= 2.
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝐿2
relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2
relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2
relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
20
3.12 × 10−2
3.56 × 10−2
3.57 × 10−2
40
2.40 × 10−3
3.7
2.33 × 10−3
3.9
2.33 × 10−3
3.9
80
3.17 × 10−4
2.9
2.92 × 10−4
3.0
3.85 × 10−4
2.6
160
4.13 × 10−5
𝟐.𝟗
3.73 × 10−5
𝟑.𝟎
2.42 × 10−4
0.7
Table 9
Convergence analysis for the traveling vortex test case using the SI-IMEX method with the IMEX-RK(3,3,3) scheme of
type II (Table 21) and polynomial degree 𝑟= 2.
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
20
3.12 × 10−2
3.55 × 10−2
3.55 × 10−2
3.63 × 10−2
40
2.40 × 10−3
3.7
2.32 × 10−3
3.9
2.31 × 10−3
3.9
3.57 × 10−3
3.3
80
3.17 × 10−4
2.9
2.92 × 10−4
3.0
2.90 × 10−4
3.0
4.50 × 10−3
−
160
4.13 × 10−5
𝟐.𝟗
3.73 × 10−5
𝟑.𝟎
4.61 × 10−5
𝟐.𝟕
Table 10
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(4,3,3) scheme of type
I (Table 20) and polynomial degree 𝑟= 2.
experienced order reduction is likely dependent on both the low Mach inaccuracy of the DG method on quadrilateral
cells and the order reduction phenomenon typical of stiff problems [88].
For what concerns the time discretization method of type II (Table 23), the expected convergence rates are
established for the IMEX method and the accuracy is preserved up to 𝑀= 10−3 (Table 14). On the contrary, the
SI-IMEX method shows the expected convergence rates for 𝑀≥10−2, whereas severe issues start from 𝑀= 10−3
(Table 15), unlike the scheme of type ARS. As already mentioned, schemes of type II show some issues in the low Mach
regime for the semi-implicit time discretization [15]. Moreover, following also the results in Section 4.1, we can further
infer that the activation of spurious modes (hence the low Mach accuracy) is influenced by the time discretization
strategy and by the time discretization scheme.
4.2.1. Investigation of fixed point iterations
In this Section, we analyze the impact of the value of the tolerance 𝜂in the fixed point loop on the overall
performance of the IMEX method depicted in Section 3.2. For the sake of brevity, we focus on 𝑀= 10−1 and on
𝑀= 10−3. We consider the third order scheme of type II (Table 21), but analogous considerations hold for the other
G. Orlando et al.: Preprint submitted to Elsevier
Page 21 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
20
3.12 × 10−2
3.55 × 10−2
3.56 × 10−2
4.70 × 10−2
40
2.40 × 10−3
3.7
2.32 × 10−3
3.9
2.32 × 10−3
3.9
2.12 × 10−2
1.1
80
3.17 × 10−4
2.9
2.92 × 10−4
3.0
2.90 × 10−4
3.0
1.01 × 10−2
1.1
160
4.13 × 10−5
𝟐.𝟗
3.73 × 10−5
𝟐.𝟗
4.69 × 10−5
𝟐.𝟕
Table 11
Convergence analysis for the traveling vortex test case using the SI-IMEX method with IMEX-RK(4,3,3) scheme of type I
(Table 20) and polynomial degree 𝑟= 2
Figure 5: Traveling vortex test case, contour plot of the velocity perturbation at 𝑀= 10−4. Left: initial field. Right: field
at final time 𝑡= 𝑇𝑓= 3. The results are obtained using the IMEX method with the IMEX-RK(3,3,3) scheme (Table 21)
and polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 40.
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
10
4.85 × 10−2
4.88 × 10−2
4.88 × 10−2
9.15 × 10−2
20
2.88 × 10−3
4.1
2.73 × 10−3
4.2
2.71 × 10−3
4.2
6.37 × 10−2
0.5
40
1.23 × 10−4
4.5
1.30 × 10−4
𝟒.𝟒
1.31 × 10−4
4.4
2.23 × 10−2
1.5
80
8.47 × 10−6
𝟑.𝟗
6.41 × 10−5
1.0
1.22 × 10−5
𝟑.𝟒
Table 12
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(5,5,4) scheme of type
ARS (Table 22) and polynomial degree 𝑟= 3.
schemes. For moderate values of the Mach number, a sufficiently low value of the tolerance 𝜂has to be chosen in
order to achieve full convergence and the number of fixed point iterations depends on the value of 𝜂(Table 16). For
what concerns low values of the Mach number, one fixed point iteration is sufficient to achieve full convergence,
independently of 𝜂(Table 16). This explains why the IMEX method and the SI-IMEX method behave similarly in this
regime in terms of computational cost. These considerations are further confirmed by the experimental contraction
G. Orlando et al.: Preprint submitted to Elsevier
Page 22 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 6: Traveling vortex test case, comparison of wall-clock times employing third order time discretization schemes and
polynomial degree 𝑟= 2. Top-left: 𝑀= 10−1. Top-right: 𝑀= 10−2. Bottom: 𝑀= 10−3. An advective Courant number
𝐶𝑢≈0.19 is employed for the IMEX method. The solid black lines denote the results obtained with the IMEX method with
the scheme of type II (Table 21), the solid blue lines show the results with the SI-IMEX method with the scheme of type
II (Table 21), the dashed red lines report the results achieved with the IMEX method with the scheme of type I (Table
20), whereas the dashed green lines represent the results established with the SI-IMEX method with the scheme of type I
(Table 20).
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
10
4.85 × 10−2
4.89 × 10−2
4.88 × 10−2
7.38 × 10−2
20
2.88 × 10−3
4.1
2.74 × 10−3
4.2
2.71 × 10−3
4.2
6.52 × 10−2
0.2
40
1.23 × 10−4
4.5
1.30 × 10−4
4.4
1.31 × 10−4
4.4
6.27 × 10−2
−
80
8.47 × 10−6
𝟑.𝟗
1.10 × 10−5
𝟑.𝟔
1.24 × 10−5
𝟑.𝟒
Table 13
Convergence analysis for the traveling vortex test case using the SI-IMEX method with the IMEX-RK(5,5,4) scheme of
type ARS (Table 22) and polynomial degree 𝑟= 3. Note that for 𝑀= 10−1 the advective Courant number is 𝐶𝑢≈0.05,
while for 𝑀≤10−2 the advective Courant number is 𝐶𝑢≈0.025 to achieve a stable solution.
rate (ECR), as defined in [67], i.e.
ECR(̃𝑘) =
‖‖𝑝̃𝑘+1 −𝑝̃𝑘‖‖∞
‖‖𝑝̃𝑘−𝑝̃𝑘−1‖‖∞
, ̃𝑘> 0,
(79)
which is reported in Table 17. One can easily notice that the error is significantly reduced after three fixed point
iterations as evident by the small value of the contraction rate. This results confirms that very few iterations of the
G. Orlando et al.: Preprint submitted to Elsevier
Page 23 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝑀= 10−3
𝑀= 10−4
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2 relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
10
4.85 × 10−2
4.88 × 10−2
4.88 × 10−2
5.90 × 10−2
20
2.88 × 10−3
4.1
2.73 × 10−3
4.2
2.71 × 10−3
4.2
1.45 × 10−2
2.0
40
1.23 × 10−4
4.5
1.30 × 10−4
4.4
1.30 × 10−4
4.4
5.56 × 10−3
1.4
80
8.47 × 10−6
𝟑.𝟗
1.10 × 10−5
𝟑.𝟔
1.26 × 10−5
𝟑.𝟒
Table 14
Convergence analysis for the traveling vortex test case using the IMEX method with the IMEX-RK(6,6,4) scheme of type
II (Table 23) and polynomial degree 𝑟= 3.
𝑁𝑒𝑙
𝑀= 10−1
𝑀= 10−2
𝐿2
relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
𝐿2
relative
error 𝛿𝐮
𝐿2 rate 𝛿𝐮
10
4.85 × 10−2
4.88 × 10−2
20
2.88 × 10−3
4.1
2.74 × 10−3
4.2
40
1.23 × 10−4
4.5
1.30 × 10−4
4.4
80
8.47 × 10−6
𝟑.𝟗
1.10 × 10−5
𝟑.𝟔
Table 15
Convergence analysis for the traveling vortex test case using the SI-IMEX method with the IMEX-RK(6,6,4) scheme of
type II (Table 23) and polynomial degree 𝑟= 3. Note that for 𝑀= 10−1 the advective Courant number is 𝐶𝑢≈0.05, while
for 𝑀= 10−2 the advective Courant number is 𝐶𝑢≈0.025 to achieve a stable solution.
𝜂
𝑀= 10−1
𝑀= 10−4
𝐿2
relative
error 𝛿𝐮
N. iters
𝐿2
relative
error 𝛿𝐮
N. iters
10−4
3.35 × 10−4
1
2.90 × 10−4
1
10−6
3.17 × 10−4
3
2.90 × 10−4
1
10−10
3.17 × 10−4
5
2.90 × 10−4
1
Table 16
Traveling vortex test case, investigation of the fixed point loop using the third order scheme of type II (Table 21) and
polynomial degree 𝑟= 2. Here, 𝜂denotes the tolerance for the stopping criterion of the fixed point loop (74), while N.
iters represents the average number of iterations in the fixed point loop. The computational mesh of the reported results
is composed by 80 × 80 = 6400 elements.
fixed point loop are sufficient to obtain a satisfactory solution. We also monitor the experimental order of convergence
of the fixed point method (EOC), defined as
EOC(̃𝑘) =
log
(‖𝑝̃𝑘+1−𝑝̃𝑘‖∞
‖𝑝̃𝑘−𝑝̃𝑘−1‖∞
)
log
( ‖𝑝̃𝑘−𝑝̃𝑘−1‖∞
‖𝑝̃𝑘−1−𝑝̃𝑘−2‖∞
), ̃𝑘> 1
(80)
and we notice that, as expected, the fixed point method is of order 1 (Table 17). From now on, we set 𝜂= 10−6 in the
following test cases.
4.3. Sod shock tube for the Peng-Robinson EOS
The main aim of the proposed time discretization methods is to use them for low Mach number flows so as to avoid
the acoustic CFL restriction. In the case of high Mach number flows, acoustic waves are not negligible. Hence, one
G. Orlando et al.: Preprint submitted to Elsevier
Page 24 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
̃𝑘
1st time step
200th time step
‖‖𝑝̃𝑘+1 −𝑝̃𝑘‖‖∞
ECR
EOC
‖‖𝑝̃𝑘+1 −𝑝̃𝑘‖‖∞
ECR
EOC
0
1.02 × 10−5
−
−
5.84 × 10−6
−
−
1
5.42 × 10−6
0.53
−
5.34 × 10−7
0.91
−
2
6.31 × 10−8
0.012
7.1
4.25 × 10−8
0.0080
54
3
1.10 × 10−9
0.017
0.91
3.03 × 10−10
0.0071
1.02
4
3.42 × 10−11
0.031
0.86
4.68 × 10−12
0.015
0.84
5
2.75 × 10−12
0.08
0.73
1.42 × 10−13
0.03
0.84
6
1.86 × 10−13
0.07
𝟏.𝟎𝟕
7.55 × 10−15
0.05
𝟎.𝟖𝟒
Table 17
Traveling vortex test case at 𝑀= 10−1, investigation of the fixed point loop using the third order scheme of type II
(Table 21) and polynomial degree 𝑟= 2. Here ̃𝑘is the index of the fixed point iteration, ECR represents the experimental
convergence rate (79), whereas EOC denotes the experimental order of convergence (80). The computational mesh of the
reported results is composed by 80 × 80 = 6400 elements.
is interested in resolving both acoustic and material waves and explicit time discretization schemes are well suited to
achieve this goal. However, we show that, when coupled with a monotonicity preserving spatial discretization, both
IMEX and SI-IMEX methods can be effective also for high Mach number flows. We consider the Sod shock tube
problem [83] for the Peng-Robinson EOS (9). The computational domain is Ω = (−0.5, 0.5), whereas the final time is
𝑇𝑓= 0.2. The initial condition reads as follows:
(𝜌, 𝑢, 𝑝) (𝑥, 0) =
{
(1, 0, 1)
if 𝑥< 0
(0.125, 0, 0.1)
if 𝑥> 0.
(81)
Dirichlet boundary conditions are imposed. Following [36], we take 𝑐𝑣= 1 and 𝑅𝑔= 0.4. Moreover, we take 𝑏= 0.5
and 𝑎(𝑇) = 0.5∕
√
𝑇. We employ the IMEX-RK(2,2,2) scheme (Table 19) with polynomial degree 𝑟= 0 so as to avoid
the oscillations that arise in the case of discontinuous solution when using high-order space discretization methods.
The computational mesh is composed by 𝑁𝑒𝑙= 500 elements and the time step is Δ𝑡≈1.33 × 10−3, yielding a
maximum acoustic Courant number 𝐶≈1.23 and 𝐶𝑢≈0.47. In particular, we point out that the acoustic Courant
number is greater than one. A reference solution is computed using the optimal third order explicit strong stability-
preserving scheme presented in [43] with 𝑁𝑒𝑙= 32000. An excellent agreement is established between the IMEX and
the SI-IMEX method and a good agreement is established with the reference solution (Figure 7). Moreover, one can
start appreciating the effectiveness of the linearization presented in Section 3.4. We will further discuss this point in
Section 4.5.
Finally, we point out that the employed spatial discretization is not TVD for 𝑟> 0. Hence, spurious oscillations
arise in the case of discontinuous solutions. A detailed discussion of possible approaches to overcome this issue is
not in the scope of the present work. However, a number of approaches have been proposed in the literature to obtain
essentially monotone schemes using high order DG methods, see, e.g., [37, 69].
4.4. Flow in an open tube
Next, we consider the test case III originally proposed in [58] for an ideal gas and also employed in [74]. The
domain is Ω = (0, 10) with a time-dependent density and velocity prescribed at the left-end, while a time-dependent
outflow pressure with a large amplitude variation is imposed at the right-end. More specifically, the initial conditions
read as follows:
(𝜌, 𝑢, 𝑝) (𝑥, 0) = (1, 1, 1) ,
(82)
while the boundary conditions are
𝜌(0, 𝑡)
=
1 + 3
10 sin (4𝑡)
(83a)
G. Orlando et al.: Preprint submitted to Elsevier
Page 25 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
a)
b)
c)
Figure 7: Sod shock tube for the Peng-Robinson EOS (9) with polynomial degree 𝑟= 0, a) density, b) velocity, c) pressure.
The continuous black lines represent the reference solution computed using the third order optimal explicit strong stability-
preserving scheme in [43], the continuous blue show the results obtained with the IMEX method using the IMEX-RK(2,2,2)
scheme in Table 19, whereas the red dots report the numerical results obtained using the SI-IMEX method.
𝑢(0, 𝑡)
=
1 + 1
2 sin (2𝑡)
(83b)
𝑝(𝐿, 𝑡)
=
1 + 1
4 sin (3𝑡) ,
(83c)
with 𝐿= 10. The final time is 𝑇𝑓= 7.47, whereas the Mach number is set to 𝑀= 10−4. Since (24) reduces to
∇⋅̄𝐮= −1
𝛾̄𝑝
𝑑̄𝑝
𝑑𝑡,
(84)
the velocity field is not solenoidal as 𝑀→0, and, in one space dimension, it is a linear function of the space with
a given time-dependent slope and boundary value at 𝑥= 0 [58, 74]. In particular, the leading order solution for the
velocity reads as follows:
̄𝑢(𝑥, 𝑡) = 𝑢(0, 𝑡) −
1
𝛾𝑝(𝐿, 𝑡)
𝑑𝑝(𝐿, 𝑡)
𝑑𝑡
𝑥= 1 + 1
2 sin (2𝑡) −
3 cos (3𝑡)
4𝛾
(
1 + 1
4 sin (3𝑡)
)𝑥.
(85)
Moreover, the leading order relation (13) reduces to
𝐷log ̄𝜌
𝐷𝑡
= −𝜕̄𝑢
𝜕𝑥=
1
𝛾𝑝(𝐿, 𝑡)
𝑑𝑝(𝐿, 𝑡)
𝑑𝑡
=
3 cos (3𝑡)
4𝛾
(
1 + 1
4 sin (3𝑡)
),
(86)
where 𝐷log ̄𝜌
𝐷𝑡
= 𝜕log ̄𝜌
𝜕𝑡
+ ̄𝑢𝜕log ̄𝜌
𝜕𝑥.
First, we employ the second order IMER-RK(2,2,2) (Table 19) with polynomial degree 𝑟= 1. We consider a
number of elements 𝑁𝑒𝑙= 50, whereas the time step is Δ𝑡= 3.735 × 10−3, leading to a maximum advective Courant
G. Orlando et al.: Preprint submitted to Elsevier
Page 26 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
number 𝐶𝑢≈0.13 and a maximum acoustic Courant number 𝐶≈310. The results at 𝑡=
𝑇𝑓
2 and at 𝑡= 𝑇𝑓obtained
with the IMEX method are those expected by the asymptotic analysis [58, 74] for both the density and velocity profiles
(Figure 8), while the SI-IMEX method does not achieve a stable solution employing this time step. We also notice that
no significant issue seems to arise in the low Mach regime.
The SI-IMEX method requires a smaller time step to achieve a stable solution. More specifically, we take
Δ𝑡= 1.8675 × 10−3, which leads to a maximum acoustic Courant number 𝐶≈155 and a maximum advective
Courant number 𝐶𝑢≈0.065. The use of the SI-IMEX method yields a computational time saving of around 37% at
fixed time step. One can easily notice that a good agreement with the leading order solution is established for both the
density and the velocity (Figure 8).
Figure 8: Open tube test case with the ideal gas law (4), comparison between the IMEX method and the SI-IMEX method
using the second order scheme (Table 19) and polynomial degree 𝑟= 1 with 𝑁𝑒𝑙= 50. Top: results at 𝑡=
𝑇𝑓
2 = 3.735.
Bottom: results at 𝑡= 𝑇𝑓= 7.47. Left: density. Right: velocity. The continuous black lines show the leading order solution
as 𝑀→0, the blue dots report the numerical results with the IMEX method and Δ𝑡= 3.735 × 10−3, whereas the red
crosses represent the results with the SI-IMEX method and Δ𝑡= 1.8675 × 10−3.
Next, we consider the third order scheme IMEX-RK(4,3,3) of type I (Table 20) and the third order scheme IMEX-
RK(3,3,3) of type II (Table 21) using polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 50. We take Δ𝑡= 1.8675×10−3, yielding a
maximum advective Courant number 𝐶𝑢≈0.14 and a maximum acoustic Courant number 𝐶≈292. Similarly to the
second order scheme, a good agreement with the leading order solution is established for both the time discretization
schemes, in particular for the scheme of type I, while the SI-IMEX method does not achieve a stable solution. We
take Δ𝑡= 4.66875 × 10−4, for which a stable solution for the scheme of type II is obtained. On the contrary, a stable
solution is not achieved for the scheme of type I. This is likely related to the employed boundary conditions. The
AP property and the AA property of the SI-IMEX method indeed has been proven considering periodic or no-flux
boundary conditions [6, 15, 50], while we are considering time-dependent Dirichlet boundary conditions. Hence, the
system is not autonomous and therefore it is outside of the theoretical framework of the SI-IMEX method depicted in
Section 3.3. Since 𝑐≠̃𝑐for schemes of type I, 𝐔𝐸and 𝐔𝐼are computed at different time instants, which seems to
cause issues for time-dependent Dirichlet boundary conditions. An excellent agreement with the leading order solution
is established for both the density and velocity field for Δ𝑡= 4.66875 × 10−4. Hence, for configurations which involve
G. Orlando et al.: Preprint submitted to Elsevier
Page 27 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
large variations of density and pressure, and time-dependent boundary conditions, the IMEX method is globally more
robust and allows for sizeable larger time steps.
Figure 9: Open tube test case with the ideal gas law (4), comparison between the IMEX method and the SI-IMEX method
using the third order schemes (Tables 20-21) in combination with polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 50. Top: results
at 𝑡=
𝑇𝑓
2 = 3.735. Bottom: results at 𝑡= 𝑇𝑓= 7.47. Left: density. Right: velocity. The continuous black lines show the
leading order solution as 𝑀→0, the green squares report the numerical results with the IMEX method of type II and
Δ𝑡= 2.334375×10−4, the orange circles show the numerical results with the IMEX method of type I and Δ𝑡= 2.334375×10−4,
the red crosses represent the results with the SI-IMEX method of type II and Δ𝑡= 2.334375 × 10−4, the blue triangles show
the results with the IMEX method of type II and Δ𝑡= 1.245 × 10−3, while the magenta diamonds report the results with
the IMEX method of type I and Δ𝑡= 1.245 × 10−3.
Next, we consider an extension of this test case using the stiffened gas equation of state (SG-EOS) (5). We take
𝛾= 4.4, 𝜋∞= 6.8 × 102, and 𝑞∞= 0 in (5). We employ the third-order scheme of type II (Table 21) in combination
with 𝑟= 2. We take Δ𝑡= 2.334375 × 10−4, yielding a maximum acoustic Courant number 𝐶≈1530. An excellent
agreement is established between the IMEX method and the SI-IMEX method (Figure 10). Moreover, one can easily
notice that the leading order solution changes with the equation of state and modifying its parameters [74] (Figure 10).
In particular, relation (24) for the SG-EOS reduces to
∇⋅̄𝐮= −
1
𝛾( ̄𝑝+ 𝜋∞
) 𝑑̄𝑝
𝑑𝑡.
(87)
and therefore the leading order solution for the velocity reads as follows:
̄𝑢(𝑥, 𝑡) = 1 + 1
2 sin (2𝑡) −
3 cos (3𝑡)
4 (𝛾+ 𝜋∞
) (
1 + 1
4 sin (3𝑡)
)𝑥.
(88)
Hence, since from (83c) ̄𝑝≥3
4 and |||
𝑑̄𝑝
𝑑𝑡
||| ≤3
4, we obtain
|∇⋅̄𝐮| ≤
1
𝛾
(
3
4 + 𝜋∞
) 3
4 ≈2.5 × 10−4,
(89)
G. Orlando et al.: Preprint submitted to Elsevier
Page 28 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
meaning that the velocity field is almost constant.
Figure 10: Open tube test case with the SG-EOS (5), comparison between the IMEX method and the SI-IMEX method
using the third order schemes (Table 21) and polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 50. Results at 𝑡= 𝑇𝑓= 7.47. Left:
density. Right: velocity. The continuous black lines show the leading order solution as 𝑀→0, the continuous blue lines
report the leading order solution as 𝑀→0 for the ideal gas with 𝛾= 1.4, the green squares report the numerical results
with the IMEX method, while the red crosses represent the results with the SI-IMEX method.
4.5. Kelvin-Helmholtz instability at low Mach
In a final test, we consider the Kelvin-Helmholtz instability at low Mach studied, e.g., in [90], which we briefly
recall here for the convenience of the reader. The computational domain is the square Ω = (−1, 1)2 endowed with
periodic boundary conditions, while the final time is 𝑇𝑓= 5. The initial conditions are
𝜌(𝐱, 0)
=
1 −1
4 tanh
[
25
(
|𝑦| −1
2
)]
(90a)
𝑝(𝐱, 0)
=
104
𝛾
(90b)
𝑢(𝐱, 0)
=
−1
2 tanh
[
25
(
|𝑦| −1
2
)]
(90c)
𝑣(𝐱, 0)
=
1
100 sin (2𝜋𝑥) cos (2𝜋𝑦) ,
(90d)
with 𝛾= 1.4. Notice that, as discussed in [90], this configuration is in the incompressible regime, but the density is
not constant. This shows that assuming a constant background density in (10), as done in some contributions, is not
always a valid assumption, even in the incompressible regime. First, we employ the ideal gas law (4). We consider the
IMEX-RK(2,2,2) scheme (Table 19) with polynomial degree 𝑟= 1 for the space discretization. The computational
mesh is composed by 𝑁𝑒𝑙= 120×120 = 14400 elements along each direction, whereas the time step is Δ𝑡= 2×10−3,
yielding a maximum acoustic Courant number 𝐶≈15 and a maximum advective Courant number 𝐶𝑢≈0.10. The
contours of density at 𝑡= 2 and at 𝑡= 5 are in good agreement with the reference results reported in [90] (Figure
11). Moreover, since we are analyzing a fluid mechanic instability, every small variation in the flow can lead to large
variations [72, 70], and the excellent agreement obtained between the IMEX method and the SI-IMEX further confirms
the correctness of our implementation and the properties of both methods.
Next, we employ the third order scheme of type II reported (Table 21) using polynomial degree 𝑟= 2 and
𝑁𝑒𝑙= 80 × 80 = 6400 elements, so that the number of degrees of freedom does not change. Recall indeed that
the total number of degrees of freedom per scalar variable is 𝑁𝑒𝑙(𝑟+ 1)2. One can easily notice that the use of higher
order methods is beneficial to resolve the fine details of the solution (Figure 12). An excellent agreement is once more
established between the IMEX method and the SI-IMEX method (Figure 12).
Next, we consider an extension of this test case employing the general cubic EOS (8). Following [21], we set
𝑎(𝑇) =
1
2
√
𝑇
(91)
G. Orlando et al.: Preprint submitted to Elsevier
Page 29 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 11: Kelvin-Helmholtz instability with the ideal gas law (4), results using the second order scheme (Table 19) and
polynomial degree 𝑟= 1 with 𝑁𝑒𝑙= 120. Top: results at 𝑡= 2. Bottom: results at 𝑡= 𝑇𝑓= 5. Left: contour plots of the
density field obtained with the IMEX method. Right: comparison between the IMEX method and the SI-IMEX method for
the isoline equal to 1. The continuous black lines show the results with IMEX method, while the dashed red lines represent
the results with the SI-IMEX.
for the attraction term in (8). Moreover, we set 𝑏= 5 × 10−3. Finally, we assume that (9) is valid, with 𝑐𝑣≈742.0,
and 𝑅𝑔≈296.8 in (6), which correspond to the thermodynamic properties of the nitrogen. An excellent agreement is
obtained between the IMEX method and the SI-IMEX method, in spite of the linearization described in (54) (Figure
13). For the sake of completeness, we have also computed the solution obtained with the SI-IMEX method without
linearizing the relation between internal energy and pressure, solving therefore a nonlinear equation for the pressure.
No difference arises in the development of the Kelvin-Helmholtz instability (Figure 13) and a computational time
saving of around 70% is obtained thanks to the linearization proposed in (54).
Finally, we consider two different configurations. First, we modify the initial condition, taking
𝑝(𝐱, 0) = 105
𝛾,
(92)
so that a more realistic maximum temperature 𝑇≈320 is obtained. In this case, we need to decrease the time step of
the SI-IMEX method to Δ𝑡= 5 × 10−4 in order to achieve a stable solution. Next, we consider the proper expression
G. Orlando et al.: Preprint submitted to Elsevier
Page 30 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 12: Kelvin-Helmholtz instability with the ideal gas law (4), results using the third order scheme of type II (Table
21) and polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 80 at 𝑡= 𝑇𝑓= 5. Left: contour plots of the density field obtained with the
IMEX method. Right: comparison between the IMEX method, the SI-IMEX method, and the second order IMEX method
for the isoline equal to 1. The continuous black line shows the results with third order IMEX method, the dashed red line
represents the results with the third order SI-IMEX method, whereas the dashed-dotted blue line reports the results with
the second order IMEX method.
A
28.98641
𝐵
1.853978
𝐶
−9.647459
𝐷
16.63537
𝐸
0.000117
Table 18
Values of the coefficients in (94)
of the attraction term 𝑎(𝑇) and of the co-volume 𝑏for the Peng-Robinson EOS [39], [81, p. 263]:
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝑎(𝑇)
= 0.45724
𝑅2
𝑔𝑇2
𝑐
𝑝𝑐𝛼(𝑇)2
𝛼(𝑇)
= 1 + Γ
(
1 −
√
𝑇
𝑇𝑐
)
Γ
= 0.37464 + 1.54226𝜔−0.26992𝜔2
𝑏
= 0.07780
𝑅𝑔𝑇𝑐
𝑝𝑐,
(93)
where 𝑇𝑐denotes the critical temperature, 𝑝𝑐the critical pressure, and 𝜔the acentric factor. For what concerns the
nitrogen, we find 𝑇𝑐= 126.19 K, 𝑝𝑐= 3.3978 ⋅106 Pa, and 𝜔= 0.0372 [64, 51]. Moreover, we consider the following
relation for 𝑒#(𝑇) in (8) [28, 64]:
𝑒# (𝑇)
=
[
𝐴𝑇
1000 + 1
2𝐵
( 𝑇
1000
)2
+ 1
3𝐶
( 𝑇
1000
)3
+ 1
4𝐷
( 𝑇
1000
)4
−𝐸1000
𝑇
]
106
𝑀𝑤
−𝑅𝑔𝑇,
(94)
with 𝑀𝑤= 28.0134 g mol−1 and 𝐴, 𝐵, 𝐶, 𝐷, 𝐸denoting suitable coefficients whose values are reported in Table
18. Notice that the polynomial expansion employed in [64] provides results expressed in kJ mol−1. Hence, a proper
conversion to obtain results in kJ kg−1 has to applied. The same consideration holds for the factor 1000, since the
argument of the polynomial is expressed in thousandths of Kelvin. The IMEX method can achieve a stable solution
for this configuration, whereas severe issues arise for the SI-IMEX method in the development of the instability.
G. Orlando et al.: Preprint submitted to Elsevier
Page 31 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
Figure 13: Kelvin-Helmholtz instability with the Peng-Robinson EOS (4), results using the third order scheme of type II
(Table 21) and polynomial degree 𝑟= 2 with 𝑁𝑒𝑙= 80. Top-left: contour plots of the density field obtained with the IMEX
method at 𝑡= 𝑇𝑓= 5. Top-right: comparison between the IMEX method and the SI-IMEX method for the isoline equal to
1. The continuous black lines show the results with IMEX method, while the dashed red lines represent the results with
the SI-IMEX method. Bottom: comparison between the SI-IMEX method solving the nonlinear pressure equation and the
SI-IMEX method using the linearization proposed in Section 3.4 for the isoline equal to 1. The continuous black lines show
the results with SI-IMEX method, while the dashed red lines represent the results with the linearized SI-IMEX.
5. Conclusions
Based on the experience of [15] and [74], we have performed a quantitative comparison between two different
Implicit-Explicit Runge-Kutta (IMEX-RK) approaches for the Euler equations of gas dynamics. The two methods are
particularly well suited for low Mach number flows, but keep their full accuracy for moderate values of the Mach
number. The spatial discretization is based on the Discontinuous Galerkin (DG) method, which naturally allows for
high-order accuracy, even though it is characterized by some limitations in the very low Mach limit on quadrilateral
cells. The two schemes, namely the IMEX-DG method and the SI-IMEX-DG method, have been compared in a number
of relevant benchmarks for ideal gases and on their non-trivial extension for non-ideal gases. The stiff dependence has
been carefully analyzed in order to avoid the solution of a nonlinear pressure equation for a general class of equations
of state (EOS).
First, we have assessed the convergence properties of the two methods. We have shown that they are asymptotic-
preserving (AP) and asymptotically-accurate (AA) in the range of low Mach accuracy established by the spatial
discretization. The SI-IMEX method provides a sizeable computational time saving ensuring the same level of
accuracy, in particular for moderate values of the Mach number. Moreover, we have noticed an impact of the time
G. Orlando et al.: Preprint submitted to Elsevier
Page 32 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
discretization strategy and also of the specific time discretization scheme in the activation of spurious modes for low
Mach numbers. More specifically, schemes of type I provide a superior stability for low values of the Mach number.
A detailed analysis of the spatial discretization will be matter of future work.
Next, we have considered non-trivial and less standard configurations. First, we have analyzed the case in which
a time-dependent pressure is imposed at the boundary, for which the asymptotic limit does not coincide with the
incompressible Euler equations. Notice that this configuration is outside of the theoretical framework of the SI-IMEX
method and, indeed, only some IMEX-RK schemes of type II are suitable. The SI-IMEX method requires a significant
smaller time step with respect to that needed by the IMEX method to achieve a stable solution.
Finally, we have focused on a Kelvin-Helmholtz instability at low Mach, that is in the incompressible regime, but
for which the density is not constant. Here we have tested for non-ideal gases a novel linearization in the relation
between internal energy and pressure, so as to avoid the solution of a nonlinear equation for the pressure. The proposed
linearization is applicable to a general EOS and automatically recovers the linear systems obtained using the ideal gas
law. No evident loss of accuracy occurs and a significant computational time saving is established.
In future work, as already mentioned, we aim to employ a spatial discretization based on simplices or Voronoi
meshes that has been shown to be low Mach accurate for steady flows. Moreover, we aim to employ a spatial
discretization based on compatible finite elements so as to improve the scaling properties with respect to the Mach
number 𝑀. Finally, we aim to further analyze the stability properties of the two methods and to consider an extension
of these approaches for the compressible Navier-Stokes equations and for two-phase flows.
Acknowledgements
We thank the two anonymous reviewers for their very useful and constructive comments and remarks, which have
greatly helped in improving the quality of the presentation of our results. G. Orlando would like to acknowledge Vincent
Perrier for useful discussions on related topics. G. Orlando, S. Boscarino, and G. Russo are part of the INdAM-GNCS
National Research Group.
The simulations have been partly run at CINECA thanks to the computational resources made available through the
ISCRA-C projects FEM-GPU - HP10CQYKJ1 and DGNWP - HP10C121HQ. We acknowledge the CINECA award,
for the availability of high-performance computing resources and support.
G. Russo and S. Boscarino thank the Italian Ministry of Instruction, University and Research (MIUR) to support
this research with funds coming from PRIN Project 2022, 2022KA3JBA, entitled “Advanced numerical methods for
time dependent parametric partial differential equations and applications” and from PRIN 2022 PNRR “FIN4GEO:
Forward and Inverse Numerical Modeling of hydrothermal systems in volcanic regions with application to geothermal
energy exploitation.”, (No. P2022BNB97). Both authors also have been supported for this work by the Spoke
1“FutureHPC&BigData” of the Italian Research Center on High-Performance Computing, Big Data and Quantum
Computing (ICSC) funded by MUR Missione 4 Componente 2 Investimento 1.4: Potenziamento strutture di ricerca e
creazione di “campioni nazionali di R&S (M4C2-19 )” - Next Generation EU (NGEU).
A. Coefficients of employed IMEX-RK schemes
We report here for the convenience of the reader some information concerning the IMEX-RK schemes employed
in the numerical simulations. We consider the second order IMEX-RK scheme proposed in [41] and also employed in
[71, 74], whose coefficients are reported in the following Butcher tableaux
0
0
0
0
2 −
√
2
2 −
√
2
0
0
1
1
2
1
2
0
√
2
4
√
2
4
1 −
√
2
2
0
0
0
0
2 −
√
2
1 −
√
2
2
1 −
√
2
2
0
1
√
2
4
√
2
4
1 −
√
2
2
√
2
4
√
2
4
1 −
√
2
2
Table 19
Butcher tableaux of the IMEX-RK(2,2,2) scheme in [41]. Left: explicit method. Right: implicit method.
For what concerns third order time discretization methods, we consider the following method of type I [13]:
G. Orlando et al.: Preprint submitted to Elsevier
Page 33 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
0
0
0
0
0
0.435866521508
0.435866521508
0
0
0
0.717933260754
0.435866521508
0.282066739245
0
0
1
−0.733534082748750
2.150527381100
−0.416993298352
0
0
1.208496649176
−0.644363170684
0.435866521508
0.435866521508
0.435866521508
0
0
0
0.435866521508
0
0.435866521508
0
0
0.717933260754
0
0.282066739245
0.435866521508
0
1
0
1.208496649176
−0.644363170684
0.435866521508
0
1.208496649176
−0.644363170684
0.435866521508
Table 20
Butcher tableaux of the IMEX-RK(4,3,3) scheme in [15]. Top: explicit method. Bottom: implicit method.
and the following scheme of type II [55]:
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
Table 21
Butcher tableaux of the IMEX-RK(3,3,3) scheme. Top: explicit method. Bottom: implicit method.
Finally, we employ the fourth order time discretization method of type ARS proposed in [24]
G. Orlando et al.: Preprint submitted to Elsevier
Page 34 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
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
Table 22
Butcher tableaux of the IMEX-RK(5,5,4) scheme. Top: explicit method. Bottom: implicit method.
and the fourth order time discretization method of type II presented in [56]
0
0
0
0
0
0
0
0
2𝛾
2𝛾
0
0
0
0
0
0
𝑐3
̃𝑎31
̃𝑎32
0
0
0
0
0
𝑐4
̃𝑎41
̃𝑎42
̃𝑎43
0
0
0
0
𝑐5
̃𝑎51
̃𝑎52
̃𝑎53
̃𝑎54
0
0
0
𝑐6
̃𝑎61
̃𝑎62
̃𝑎63
̃𝑎64
̃𝑎65
0
0
1
̃𝑎71
̃𝑎72
̃𝑎73
̃𝑎74
̃𝑎75
̃𝑎76
0
0
0
𝑏3
𝑏4
𝑏5
𝑏6
𝛾
0
0
0
0
0
0
0
0
2𝛾
𝛾
𝛾
0
0
0
0
0
𝑐3
𝑎32
𝑎32
𝛾
0
0
0
0
𝑐4
𝑎42
𝑎42
𝑎43
𝛾
0
0
0
𝑐5
𝑎52
𝑎52
𝑎53
𝑎54
𝛾
0
0
𝑐6
𝑎62
𝑎62
𝑎63
𝑎64
𝑎65
𝛾
0
1
0
0
𝑏3
𝑏4
𝑏5
𝑏6
𝛾
0
0
𝑏3
𝑏4
𝑏5
𝑏6
𝛾
Table 23
Butcher tableaux of the IMEX-RK(6, 6, 4) scheme. Top: explicit method. Bottom: implicit method.
with
𝛾= 247∕2000,
𝑐2 = 2𝛾,
𝑐3 = (2 +
√
2)𝛾,
𝑐4 = 67∕200,
𝑐5 = 3∕40,
𝑐6 = 7∕10,
G. Orlando et al.: Preprint submitted to Elsevier
Page 35 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
𝑏3 =
9164257142617∕17756377923965,
𝑏4 = −10812980402763∕74029279521829,
𝑏5 =
1335994250573∕5691609445217,
𝑏6 =
2273837961795∕8368240463276,
𝑏7 =
247∕2000,
𝑎32 =
624185399699∕4186980696204,
𝑎42 =
1258591069120∕10082082980243,
𝑎43 = −322722984531∕8455138723562,
𝑎52 = −436103496990∕5971407786587,
𝑎53 = −2689175662187∕11046760208243,
𝑎54 =
4431412449334∕12995360898505,
𝑎62 = −2207373168298∕14430576638973,
𝑎63 =
242511121179∕3358618340039,
𝑎64 =
3145666661981∕7780404714551,
𝑎65 =
5882073923981∕14490790706663.
and
̃𝑎31 = 247∕4000,
̃𝑎32 = 2694949928731∕7487940209513
̃𝑎41 = 464650059369∕8764239774964,
̃𝑎42 = 878889893998∕2444806327765,
̃𝑎43 = −952945855348∕12294611323341,
̃𝑎51 = 476636172619∕8159180917465,
̃𝑎52 = −1271469283451∕7793814740893,
̃𝑎53 = −859560642026∕4356155882851,
̃𝑎54 = 1723805262919∕4571918432560,
̃𝑎61 = 6338158500785∕11769362343261,
̃𝑎62 = −4970555480458∕10924838743837,
̃𝑎63 = 3326578051521∕2647936831840,
̃𝑎64 = −880713585975∕1841400956686,
̃𝑎65 = −1428733748635∕8843423958496,
̃𝑎71 = 760814592956∕3276306540349,
̃𝑎72 = 760814592956∕3276306540349,
̃𝑎73 = −47223648122716∕6934462133451,
̃𝑎74 = 71187472546993∕9669769126921,
̃𝑎75 = −13330509492149∕9695768672337,
̃𝑎76 = 11565764226357∕8513123442827.
References
[1] Abbate, E., Iollo, A., Puppo, G., 2019. An asymptotic-preserving all-speed scheme for fluid dynamics and nonlinear elasticity. SIAM Journal
on Scientific Computing 41, A2850–A2879.
[2] Africa, P., 2022. lifex: A flexible, high performance library for the numerical solution of complex finite element problems. SoftwareX 20,
101252.
[3] Arndt, D., Bangerth, W., Bergbauer, M., Feder, M., Fehling, M., Heinz, J., Heister, T., Heltai, L., Kronbichler, M., Maier, M., Munch, P., J-P.,
P., Turcksin, B., Wells, D., Zampini, S., 2023. The deal.II library, version 9.5. Journal of Numerical Mathematics 31, 231–246.
[4] Arnold, D., 1982. An interior penalty finite element method with discontinuous elements. SIAM Journal of Numerical Analysis 19, 742–760.
[5] Arnold, D., Boffi, D., Falk, R., 2005. Quadrilateral H(div) finite elements. SIAM Journal on Numerical Analysis 42, 2429–2451.
[6] Arun, K., Gupta, A., Samantaray, S., 2021. Analysis of an asymptotic preserving low Mach number accurate IMEX-RK scheme for the wave
equation system. Applied Mathematics and Computation 411, 126469.
[7] Arun, K., Samantaray, S., 2020. Asymptotic preserving low Mach number accurate IMEX finite volume schemes for the isentropic Euler
equations. Journal of Scientific Computing 82, 1–32.
[8] Ascher, U., Ruuth, S., Spiteri, R., 1997. Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations. Applied
Numerical Mathematics 25, 151–167.
[9] Bangerth, W., Hartmann, R., Kanschat, G., 2007. deal.II: a general-purpose object-oriented finite element library. ACM Transactions on
Mathematical Software (TOMS) 33, 24–51.
[10] Barsukow, W., 2021. Truly multi-dimensional all-speed schemes for the Euler equations on Cartesian grids. Journal of Computational Physics
435, 110216.
[11] Bassi, F., Rebay, S., 1997. High-order accurate discontinuous finite element solution of the 2d Euler equations. Journal of Computational
Physics 138, 251–285.
[12] Boscarino, S., 2007. Error analysis of IMEX Runge-Kutta methods derived from differential-algebraic systems. SIAM Journal on Numerical
Analysis 45, 1600–1621.
[13] Boscarino, S., Filbet, F., Russo, G., 2016. High order semi-implicit schemes for time dependent partial differential equations. Journal of
Scientific Computing 68, 975–1001.
[14] Boscarino, S., Pareschi, L., Russo, G., 2024. Implicit-explicit methods for evolutionary partial differential equations. 1 ed., Society for
Industrial and Applied Mathematics. doi:10.1137/1.9781611978209.
[15] Boscarino, S., Qiu, J., Russo, G., Xiong, T., 2022. High order semi-implicit WENO schemes for all-Mach full Euler system of gas dynamics.
SIAM Journal on Scientific Computing 44, B368–B394.
[16] Boscarino, S., Russo, G., 2009. On a class of uniformly accurate IMEX Runge-Kutta schemes and applications to hyperbolic systems with
relaxation. SIAM Journal on Scientific Computing 31, 1926–1945.
[17] Boscarino, S., Russo, G., 2024. Asymptotic preserving methods for quasilinear hyperbolic systems with stiff relaxation: a review. SeMA
Journal 81, 3–49.
[18] Boscheri, W., Busto, S., Dumbser, M., 2025. An all Mach number semi-implicit hybrid Finite Volume/Virtual Element method for compressible
viscous flows on Voronoi meshes. Computer Methods in Applied Mechanics and Engineering 433, 117502.
G. Orlando et al.: Preprint submitted to Elsevier
Page 36 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
[19] Boscheri, W., Dimarco, G., Loubère, R., Tavelli, M., Vignal, M.H., 2020. A second order all mach number IMEX finite volume solver for the
three dimensional Euler equations. Journal of Computational Physics 415, 109486.
[20] Boscheri, W., Dumbser, M., Ioriatti, M., Peshkov, I., Romenski, E., 2021. A structure-preserving staggered semi-implicit finite volume scheme
for continuum mechanics. Journal of Computational Physics 424, 109866.
[21] Boscheri, W., Pareschi, L., 2021. High order pressure-based semi-implicit IMEX schemes for the 3D Navier-Stokes equations at all Mach
numbers. Journal of Computational Physics 434, 110206.
[22] Brugnano, L., Casulli, V., 2008. Iterative solution of piecewise linear systems. SIAM Journal on Scientific Computing 30, 463–472.
[23] Butcher, J., 2008. Numerical Methods for Ordinary Differential Equations. 2 ed., Wiley.
[24] Calvo, M., De Frutos, J., Novo, J., 2001. Linearly implicit Runge-Kutta methods for advection-reaction-diffusion equations. Applied Numerical
Mathematics 37, 535–549.
[25] Casulli, V., Greenspan, D., 1984. Pressure method for the numerical solution of transient, compressible fluid flows. International Journal for
Numerical Methods in Fluids 4, 1001–1012.
[26] Casulli, V., Zanolli, P., 2010. A nested Newton-type algorithm for finite volume methods solving Richards’ equation in mixed form. SIAM
Journal on Scientific Computing 32, 2255–2273.
[27] Chalons, C., Girardin, M., Kokh, S., 2013. Large time step and asymptotic preserving numerical schemes for the gas dynamics equations with
source terms. SIAM Journal on Scientific Computing 35, A2874–A2902.
[28] Chase, M., 1996. NIST-JANAF thermochemical tables for oxygen fluorides. Journal of Physical and Chemical Reference Data 25, 551–603.
[29] Chorin, A., 1968. Numerical solution of the Navier-Stokes equations. Mathematics of Computation 22, 745–762.
[30] Cockburn, B., Lin, S., Shu, C., 1989. TVB Runge-Kutta local projection discontinuous Galerkin finite element method for conservation laws.
III. One-dimensional systems. Journal of Computational Physics 84, 90–113.
[31] Cockburn, B., Shu, C., 1989. TVB Runge-Kutta local projection discontinuous Galerkin finite element method for conservation laws. II.
General framework. Mathematics of Computation 52, 411–435.
[32] Cordier, F., Degond, P., Kumbaro, A., 2012. An asymptotic-preserving all-speed scheme for the Euler and Navier-Stokes equations. Journal
of Computational Physics 231, 5685–5704.
[33] Courant, R., Friedrichs, K., Lewy, H., 1928. Über die partiellen Differenzengleichungen der mathematischen physik. Mathematische annalen
100, 32–74.
[34] Dellacherie, S., 2010.
Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number.
Journal of
Computational Physics 229, 978–1016.
[35] Dimarco, G., Loubère, R., Vignal, M.H., 2017. Study of a new asymptotic preserving scheme for the Euler system in the low Mach number
limit. SIAM journal on Scientific Computing 39, A2099–A2128.
[36] Dumbser, M., Casulli, V., 2016. A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible Navier-Stokes
equations with general equation of state. Applied Mathematics and Computation 272, 479–497.
[37] Dumbser, M., Loubère, R., 2016. A simple robust and accurate a posteriori sub-cell finite volume limiter for the discontinuous Galerkin
method on unstructured meshes. Journal of Computational Physics 319, 163–199.
[38] Feireisl, E., Klein, R., Novotn`y, A., Zatorska, E., 2016. On singular limits arising in the scale analysis of stratified fluid flows. Mathematical
Models and Methods in Applied Sciences 26, 419–443.
[39] Fernandez, M., 2009. Propellant tank pressurization modeling for a hybrid rocket. Master’s thesis. Rochester Institute of Technology.
[40] Giraldo, F., 2020. An Introduction to Element-Based Galerkin Methods on Tensor-Product Bases. Springer Nature.
[41] Giraldo, F., Kelly, J., Constantinescu, E., 2013. Implicit-explicit formulations of a three-dimensional nonhydrostatic unified model of the
atmosphere (NUMA). SIAM Journal of Scientific Computing 35, 1162–1194.
[42] Godlewski, E., Raviart, P.A., 2013. Numerical approximation of hyperbolic systems of conservation laws. volume 118. Springer Science &
Business Media.
[43] Gottlieb, S., Shu, C.W., Tadmor, E., 2001. Strong stability-preserving high-order time discretization methods. SIAM review 43, 89–112.
[44] Graebel, W., 2018. Engineering fluid mechanics. CRC Press.
[45] Guermond, J.L., Kronbichler, M., Maier, M., Popov, B., Tomas, I., 2022. On the implementation of a robust and efficient finite element-based
parallel solver for the compressible Navier-Stokes equations. Computer Methods in Applied Mechanics and Engineering 389, 114250.
[46] Guillard, H., Viozat, C., 1999. On the behaviour of upwind schemes in the low Mach number limit. Computers & fluids 28, 63–86.
[47] Haack, J., Jin, S., Liu, J.G., 2012.
An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations.
Communications in Computational Physics 12, 955–980.
[48] Hairer, E., Norsett, S., Wanner, G., 1993. Solving Ordinary Differential Equations I: Nonstiff Problems.
[49] Harten, A., Lax, P., van Leer, B., 1983. On upstream differencing and godunov-type schemes for hyperbolic conservation laws. SIAM review
25, 35–61.
[50] Huang, G., Xing, Y., Xiong, T., 2022. High order well-balanced asymptotic preserving finite difference WENO schemes for the shallow water
equations in all Froude numbers. Journal of Computational Physics 463, 111255.
[51] Jacobsen, R., Stewart, R., Jahangiri, M., 1986. Thermodynamic properties of nitrogen from the freezing line to 2000 K at pressures to 1000
MPa. Journal of Physical and Chemical Reference Data 15, 735–909.
[52] Jung, J., Perrier, V., 2024a. Behavior of the discontinuous Galerkin method for compressible flows at low Mach number on triangles and
tetrahedrons. SIAM Journal on Scientific Computing 46, A452–A482.
[53] Jung, J., Perrier, V., 2024b. A curl preserving finite volume scheme by space velocity enrichment. application to the low Mach number accuracy
problem. Journal of Computational Physics 515, 113252.
[54] Karniadakis, G., Sherwin, S., 2005. Spectral ℎ𝑝−Element Methods for Computational Fluid Dynamics. Oxford University Press.
[55] Kennedy, C., Carpenter, M., 2003.
Additive Runge-Kutta schemes for convection-diffusion-reaction equations.
Applied Numerical
Mathematics 44, 139–181.
G. Orlando et al.: Preprint submitted to Elsevier
Page 37 of 38


AP and AA IMEX methods for Euler equations: non-ideal gases
[56] Kennedy, C., Carpenter, M., 2019.
Higher-order additive runge–kutta schemes for ordinary differential equations.
Applied numerical
mathematics 136, 183–205.
[57] Klainerman, S., Majda, A., 1981. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of
compressible fluids. Communications on pure and applied Mathematics 34, 481–524.
[58] Klein, R., 1995. Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: One-dimensional flow. Journal
of Computational Physics 121, 213–237.
[59] Kronbichler, M., Heister, T., Bangerth, W., 2012.
High accuracy mantle convection simulation through modern numerical methods.
Geophysical Journal International 191, 12–29.
[60] Kronbichler, M., Kormann, K., 2019. Fast matrix-free evaluation of discontinuous Galerkin finite element operators. ACM Transactions on
Mathematical Software (TOMS) 45, 1–40.
[61] Kučera, V., Lukáčová-Medvid’ová, M., Noelle, S., Schütz, J., 2022. Asymptotic properties of a class of linearly implicit schemes for weakly
compressible euler equations. Numerische Mathematik 150, 79–103.
[62] Le Métayer, O., Saurel, R., 2016. The Noble-Abel Stiffened-Gas equation of state. Physics of Fluids 28, 046102.
[63] LeVeque, R., 2002. Finite volume methods for hyperbolic problems. volume 31. Cambridge university press.
[64] Lias, S., Bartmess, J., Liebman, J., Holmes, J., Levin, R., Mallard, G., 2010. NIST chemistry webbook standard reference database number
69. National Institute of Standards Technology .
[65] Munz, C., 1994. On Godunov-type schemes for Lagrangian gas dynamics. SIAM Journal on Numerical Analysis 31, 17–42.
[66] Munz, C., Roller, S., Klein, R., Geratz, K., 2003. The extension of incompressible flow solvers to the weakly compressible regime. Computers
and Fluids 32, 173–196.
[67] Noelle, S., Bispen, G., Arun, K., Lukáčová-Medvid’ová, M., Munz, C.D., 2014. A weakly asymptotic preserving low Mach number scheme
for the Euler equations of gas dynamics. SIAM Journal on Scientific Computing 36, B989–B1024.
[68] Ocher, S., Solomon, F., 1982. Upwind difference schemes for hyperbolic conservation laws. Math. Comput 38, 339–374.
[69] Orlando, G., 2023. A filtering monotonization approach for DG discretizations of hyperbolic problems. Computers & Mathematics with
Applications 129, 113–125.
[70] Orlando, G., 2024. An implicit DG solver for incompressible two-phase flows with an artificial compressibility formulation. International
Journal for Numerical Methods in Fluids 96, 1932–1959.
[71] Orlando, G., Barbante, P., Bonaventura, L., 2022. An efficient IMEX-DG solver for the compressible Navier-Stokes equations for non-ideal
gases. Journal of Computational Physics 471, 111653.
[72] Orlando, G., Benacchio, T., Bonaventura, L., 2023. An IMEX-DG solver for atmospheric dynamics simulations with adaptive mesh refinement.
Journal of Computational and Applied Mathematics 427, 115124.
[73] Orlando, G., Benacchio, T., Bonaventura, L., 2024. Impact of curved elements for flows over orography with a Discontinuous Galerkin scheme.
Journal of Computational Physics 519, 113445.
[74] Orlando, G., Bonaventura, L., 2025.
Asymptotic-preserving IMEX schemes for the Euler equations of non-ideal gases.
Journal of
Computational Physics 529, 113889.
[75] Pareschi, L., Russo, G., 2005. Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation. Journal of
Scientific computing 25, 129–155.
[76] Patankar, S., 2018. Numerical heat transfer and fluid flow. CRC press.
[77] Peng, D.Y., Robinson, D.B., 1976. A new two-constant equation of state. Industrial & Engineering Chemistry Fundamentals 15, 59–64.
[78] Reddy, S., Waruszewski, M., de Braganca A., F.A., Giraldo, F., 2023. Schur complement IMplicit-EXplicit formulations for discontinuous
Galerkin non-hydrostatic atmospheric models. Journal of Computational Physics 491, 112361.
[79] Rieper, F., Bader, G., 2009. The influence of cell geometry on the accuracy of upwind schemes in the low Mach number regime. Journal of
Computational Physics 228, 2918–2933.
[80] Rusanov, V., 1962. The calculation of the interaction of non-stationary shock waves and obstacles. USSR Computational Mathematics and
Mathematical Physics , 304–320.
[81] Sandler, S., 2017. Chemical, biochemical, and engineering thermodynamics. John Wiley & Sons.
[82] Shu, C.W., 2009. High order weighted essentially nonoscillatory schemes for convection dominated problems. SIAM review 51, 82–126.
[83] Sod, G., 1978. A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. Journal of computational
physics 27, 1–31.
[84] Tavelli, M., Dumbser, M., 2017. A pressure-based semi-implicit space-time discontinuous Galerkin method on staggered unstructured meshes
for the solution of the compressible Navier-Stokes equations at all Mach numbers. Journal of Computational Physics 341, 341–376.
[85] Thomann, A., Zenk, M., Puppo, G., Klingenberg, C., 2019. An all speed second order IMEX relaxation scheme for the Euler equations.
Communications in Computational Physics 28, 591–620.
[86] Toro, E., 2009. Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction. Springer Science & Business Media.
[87] Vidal, J., 2001. Thermodynamics: Applications to chemical engineering and petroleum industry. Editions Technip.
[88] Wanner, G., Hairer, E., 1996. Solving ordinary differential equations II. volume 375. Springer Berlin Heidelberg New York.
[89] Whitham, G., 2011. Linear and nonlinear waves. volume 42. John Wiley & Sons.
[90] Zampa, E., Dumbser, E., 2025. An asymptotic-preserving and exactly mass-conservative semi-implicit scheme for weakly compressible flows
based on compatible finite elements. Journal of Computational Physics 521, 113551.
[91] Zeifang, J., Schütz, J., Kaiser, K., Beck, A., Lukáčová-Medvid’ová, M., Noelle, S., 2019. A Novel Full-Euler Low Mach Number IMEX
Splitting. Communications in Computational Physics 27, 292–320.
G. Orlando et al.: Preprint submitted to Elsevier
Page 38 of 38


