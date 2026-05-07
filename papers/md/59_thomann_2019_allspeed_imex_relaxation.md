arXiv:1907.08398v1  [math.NA]  19 Jul 2019
An all speed second order IMEX relaxation scheme
for the Euler equations
Andrea Thomann∗†‡, Markus Zenk§, Gabriella Puppo¶,
Christian Klingenberg§
July 22, 2019
Abstract
We present an implicit-explicit ﬁnite volume scheme for the Euler equations. We start from the
non-dimensionalised Euler equations where we split the pressure in a slow and a fast acoustic
part. We use a Suliciu type relaxation model which we split in an explicit part, solved using a
Godunov-type scheme based on an approximate Riemann solver, and an implicit part where we
solve an elliptic equation for the fast pressure. The relaxation source terms are treated projecting
the solution on the equilibrium manifold. The proposed scheme is positivity preserving with
respect to the density and internal energy and asymptotic preserving towards the incompressible
Euler equations. For this ﬁrst order scheme we give a second order extension which maintains the
positivity property. We perform numerical experiments in 1D and 2D to show the applicability
of the proposed splitting and give convergence results for the second order extension.
Keywords
ﬁnite volume methods, Euler equations, positivity preserving, asymptotic preserv-
ing, relaxation, low Mach scheme, IMEX schemes
1
Introduction
We consider the non-dimensional Euler equations in d-space dimensions which are given by the
following set of equations [1]
ρt + ∇· (ρu) = 0
(ρu)t + ∇· (ρu ⊗u +
1
M2 pI) = 0
Et + ∇· (u(E + p)) = 0,
(1.1)
∗Dipartimento di Scienze e Alta Tecnologia, Universit`a degli Studi dell’Insubria, Via Valleggio 11, 22100 Como,
Italy
†Correspondence to: Andrea Thomann, University of Insubria, Via Valleggio 11, 22100 Como (CO), Italy,
Email: acthomann@uninsubria.it
‡Marie Sk lodowska-Curie fellow of the Istituto Nazionale di Alta Matematica Francesco Severi, Rome, Italy
§Fakult¨at f¨ur Mathematik und Informatik, Universit¨at W¨urzburg, Emil-Fischer-Str.
40, 97074 W¨urzburg,
Germany
¶Dipartimento di Matematica, La Sapienza Universit`a di Roma, Piazzale Aldo Moro 5, 00185 Roma, Italy
1


where the total energy is given by
E = ρe + 1
2M2ρ|u|2
(1.2)
and e > 0 denotes the internal energy. The density is denoted by ρ > 0, u ∈Rd is the velocity
vector and M is a given Mach number which controls the ratio between the velocity of the
gas and the sound speed. Depending on the magnitude of the Mach number the characteristic
nature of the ﬂow changes. This makes the numerical simulation of these ﬂows very challenging
but also a very interesting research subject with a wide range of applications, for example in
astrophysical stellar evolution or multiphase ﬂows [2, 3]. For large Mach numbers, the ﬂow
is governed by compressible eﬀects whereas in the low Mach limit the compressible equations
converge towards the incompressible regime. This behaviour was studied for example in [4, 1, 5].
We refer to [6] for a study of the full Euler equations.
Standard schemes designed for compressible ﬂows like the Roe scheme [7] or Godunov type
schemes fail due to exessive diﬀusion when applied in the low Mach regime.
A lot of work
is dedicated to cure this defect, see for instance [8, 9, 1, 2]. Another way to ensure accurate
solutions in the low Mach regime is the development of asymptotic preserving schemes which
are consistent with its limit behaviour as M tends to zero, see for example [10, 11, 12] and
references therein.
Due to the hyperbolic nature of (1.1) the time step for an explicit scheme is restricted by
a stability CFL condition that depends on the inverse of the fastest wave speed. In the case of
(1.1) the acoustic wave speeds tend to inﬁnity as M tends to zero which leads to very small time
steps to guarantee the stability of the explicit scheme. As a side eﬀect all waves will be resolved
by the explicit scheme although the fast waves are not necessary to capture the motion of the
ﬂuid as they carry a negligible amount of energy. Implicit methods on the other hand allow
larger time steps but introduce diﬀusion on the slow wave which leads to a loss of accuracy. In
addition at each iteration a non-linear often ill conditioned algebraic system has to be solved.
Implicit-explicit (IMEX) methods try to overcome those disadvantages by treating the stiﬀparts
implicitly and thus allow for a Mach number independent time step. Many of those schemes are
based on a splitting of the pressure in the spirit of Klein [13] since the stiﬀness of the system is
closely related with the pressure, see for example [10, 14, 13].
Another way to avoid solving non-linear implicit systems is using a relaxation approach.
See [15, 16, 17, 18, 19] for references on relaxation. The idea of using relaxation is to linearise
the equations which makes it easier to solve them implicitly as done in [20] through Jin Xin
relaxation [15]. The linear degenerate structure of the resulting relaxation models allows to use
accurate Godunov-type ﬁnite volume methods due to the knowledge about the Riemann solution
also with implicit schemes as done in [21]. Here, we want to combine a Suliciu type relaxation
with an IMEX scheme and the splitting of the pressure. The relaxation model we use is based
on relaxing the slow and the fast pressure separately. The ﬂux is then split such that only some
relaxation variables are treated implicitly which leads to solving only one equation implicitly.
This results in a computationally cheap scheme. We do not split inside the ﬂux of the Euler
equations but treat all terms explicitly. This leads to a conservative explicit part for which we
use a Godunov-type approximate Riemann solver. The use of a Riemann solver allows us to
prove with little eﬀort the positivity preserving property of the scheme which is important in
physical applications. In addition the way the ﬂux is splitted leads to an asymptotic preserving
scheme.
To have a relevant scheme for applications an extension to second or higher order is necessary.
2


Higher order schemes in time can be achieved by using IMEX Runge Kutta (RK) methods as in
[12, 22]. As standard in ﬁnite volume schemes higher order in space is achieved by reconstructing
the cell interface values using WENO schemes for example [23, 24]. We use a MUSCL approach
[25] to achieve a second order extension to the ﬁrst order scheme which preserves the positivity
of density and internal energy.
The paper is organized as follows.
In the next section we describe the relaxation model
that is used to derive the IMEX scheme. Section 3 is dedicated to the splitting of the ﬂux into
implicit and explicit terms and the structure of the ﬁrst order time semi-discrete scheme. The
asymptotic preserving property is proved in Section 4. Next, we give the derivation of the fully
discrete scheme in Section 5. Therein the Godunov type scheme for the explicit part is given as
well as the proof of the positivity of the resulting numerical scheme. In addition, we show that
the diﬀusion introduced by the Riemann solver is independent of the Mach number due to the
splitting described in Section 3. In Section 6 we give a second order extension for the ﬁrst order
scheme for which we show that it preserves the positivity property. It is followed by a section
of numerical results to validate the theoretical results. A section of conclusion completes this
paper.
2
Suliciu Relaxation model
To simplify the non-linear structure of the Euler equations (1.1) we make use of the Suliciu
relaxation approach [16, 26, 27] and references therein. Compared to the Jin Xin relaxation
[15] the original system of equations remains unchanged. Whereas in the Jin Xin relaxation
approach the linearisation of the equations is achieved by relaxing every component of the ﬂux
function, in the Suliciu type relaxation single variables are relaxed in a fashion that is tailored
to the problem. This leads to a reduced diﬀusion introduced by the relaxation process compared
to the Jin Xin relaxation.
Following the usual Suliciu relaxation procedure, the key element is the relaxation of the
pressure by introducing a new variable π,
(ρπ)t + ∇· (ρπu) + a2∇· u = ρ
ǫ (p −π).
(2.1)
Here ε denotes the relaxation time. Equation (2.1) is derived by multiplying the density equation
by ∂ρp. The non-linearity ρ2∂ρp(ρ, e) that thereby arises is replaced by a constant parameter a2,
in the following called the relaxation parameter. To guarantee a stable diﬀusive approximation of
the original Euler equations the relaxation parameter must meet the sub-characteristic condition
a > ρ
p
∂ρp(ρ, e) > 0 [17]. We want to proﬁt from the properties of the Suliciu relaxation also
for the non-dimensional Euler equations. In the following, we describe the relaxation model that
was ﬁrst introduced in [21]. Following [28], in a ﬁrst step the pressure is decomposed into a slow
dynamics and a fast acoustic component
p
M2 = p + 1 −M2
M2
p.
Approximating the slow and the fast pressure in the momentum equation in (1.1) by the variables
π and ψ respectively, momentum equation becomes
(ρu)t + ∇·

ρu ⊗u + π + 1 −M2
M2
ψ

= 0
3


The evolution of the new variables π and ψ are then developed in the spirit of a Suliciu relaxation
approach.
The evolution for π is given by the Suliciu relaxation equation (2.1).
However,
applying the standard Suliciu relaxation method also on the pressure ψ, would only lead to non
relevant diﬀusion terms and not to a low Mach scheme. To overcome this, the authors of [21]
introduced a new velocity variable ˆu ∈Rd which is relaxed to the system velocity u and couples
to the pressure ψ. The form of the evolution equations for ˆu and ψ is chosen, such that
• the resulting model is in conservation form
• the resulting model has ordered eigenvalues, which results in a clear wave structure
• the resulting model is a stable diﬀusive approximation of the non-dimensional Euler equa-
tions (1.1)
• the resulting numerical scheme has a Mach number independent diﬀusion.
Considering the above points, this leads to the following relaxation model
ρt + ∇· (ρu) = 0,
(ρu)t + ∇·

ρu ⊗u + π + 1 −M2
M2
ψ

= 0,
Et + ∇· (u(E + M2π + (1 −M2)ψ)) = 0,
(ρπ)t + ∇· (ρuπ + a2u) = ρ
ε(p −π),
(ρˆu)t + ∇· (ρu ⊗ˆu +
1
M2 ψ) = ρ
ε(u −ˆu),
(ρψ)t + ∇· (ρuψ + a2ˆu) = ρ
ε(p −ψ).
(2.2)
The relaxation model given in (2.2) diﬀers from the one given in [21] in the following points:
1. In the equation for ˆu, we use
1
M2 instead of
1
M4 as proposed by the authors in [21]. This is
due to the upwind discretization used in [21] which requires
1
M4 in order to have a Mach
number independent diﬀusion of the numerical scheme.
Here instead we use centered
diﬀerences in the implicit part to ensure the Mach number independent diﬀusion of the
numerical scheme.
2. We have simpliﬁed the model in the sense that we do not distinguish between the given
Mach number M and the local Mach number Mloc. This is not a restriction in application,
because the choice of M is given by the application as illustrated in the numerical results.
Especially for the Mach number dependent shock test case in Section 7.1.2 we directly
compare our results to a scheme that uses the local Mach number Mloc.
The following lemma sums up some properties of system (2.2). The proof can be found in [21].
Lemma 1 The relaxation system (2.2) is hyperbolic and is a stable diﬀusive approximation of
(1.1) under the Mach number independent sub-characteristic condition a > ρ
p
∂ρp(ρ, e). It has
the following linearly degenerate eigenvalues
λu = u, λ± = u ± a
ρ, λ±
M = u ±
a
Mρ,
where λu has multiplicity 4. For M < 1, the eigenvalues have the ordering
λ−
M < λ−< λu < λ+ < λ+
M.
4


In the case of M = 1 the waves given by λ± and λ±
M collapse to the waves λ± which have then
multiplicity 2 respectively.
To shorten notations we will refer to the original system (1.1) by
wt + ∇· f(w) = 0,
(2.3)
where w = (ρ, ρu, E)T denotes the physical variables and the ﬂux function is given by f(w) =


ρu
ρu ⊗u +
p
M2I
u(E + p)

. The relaxation model (1.1) is given by
Wt + ∇· F(W) = 1
εR(W),
(2.4)
where W = (ρ, ρu, E, ρπ, ρˆu, ρψ) denotes the state vector, F the ﬂux function as deﬁned in (2.2)
and R the relaxation source term given by
R(W) =








0
0
0
ρ(p −π)
ρ(u −ˆu)
ρ(p −ψ)








.
The relaxation equilibrium state is deﬁned as
W eq = (ρ, ρu, E, ρp(ρ, e), ρu, ρp(ρ, e)).
(2.5)
The connection between (2.3) and (2.4) can be established, following [17], through the matrix
Q ∈R2+d×2(2+d) deﬁned by
Q =
 I2+d
02+d

(2.6)
where d is the dimension. For all states W it is satisﬁed QR(W) = 0 and the physical variables
are then recovered by w = QW and the ﬂuxes are connected by f(w) = QF(W eq).
3
Time semi-discrete scheme
As we have seen in Lemma 1, the largest eigenvalue |λ±
M| of the relaxation model (2.2) tends to
inﬁnity as M goes to 0. Using a time explicit scheme results in a very restrictive CFL condition.
By using an IMEX approach as done in [29, 12], we can avoid the Mach number dependence of
the time step.
We rewrite the relaxation system (2.2)/(2.4) in the following form:
Wt + ∇· F(W) +
1
M2 ∇· G(W) = 1
εR(W).
(3.1)
In (3.1), we have split the ﬂux F in (2.4) into a ﬂux function F which will contain the explicit
terms and a ﬂux function G which will contain the terms treated implicitly. For eﬃciency, we
want to have as many explicit terms as possible as long as the eigenvalues of F are independent
5


of the Mach number.
To avoid especially inverting a large non-linear system, we treat the
non-linear advection terms explicitly. This results in the following following ﬂux functions
F(W) =








ρu
ρu ⊗u + π
1 + 1−M2
M2 ψ
1
(E + M2π + (1 −M2)ψ)u
ρπu + a2u
ρu ⊗ˆu
ρψu








and G(W) =








0
0
0
0
ψ
a2M2ˆu








.
(3.2)
We see, that F contains the ﬂux f of the Euler equations. (1.1) whereas G and R only act
on the relaxation variables. To obtain a time semi-discrete scheme we order the implicit and
explicit steps as
Implicit: Wt +
1
M2 ∇· G(W) = 0,
(3.3)
Explicit:
Wt + ∇· F(W) = 0,
(3.4)
Projection:
Wt = 1
εR(W).
(3.5)
The relaxation source term in (3.5) is solved by projecting the variables onto the equilibrium
manifold and thereby reaching the relaxation equilibrium state (2.5). The formal time semi-
discrete scheme is then given by
W (1) −W n,eq + ∆t
M2 ∇· G(W (1)) = 0,
(3.6)
W (2) −W (1) + ∆t∇· F(W (1)) = 0,
(3.7)
W n+1 = W (2),eq,
(3.8)
where we consider the data at time tn to be at relaxation equilibrium W n,eq. First we solve the
implicit equation (5.1) to gain W (1), followed by the explicit step where we calculate W (2). To
get the variables W n+1 at the new time level, we project W (2) onto its equilibrium state. This
procedure results into a ﬁrst order scheme in time.
4
Asymptotic properties
We consider as continuous limit equations the incompressible Euler equations given by
ρ = const.
ut + u · ∇u + ∇Π = 0
∇· u = 0
(4.1)
with a dynamical pressure described by Π.
Since the solution must converge to the solution of the incompressible system (4.1) as M →0,
see for example [8, 1], its asymptotic expansion under slipping or periodic boundary conditions
must satisfy
ρ = ρ0 + O(M),
ρ0 = const.
(4.2)
u = u0 + O(M),
∇· u0 = 0
(4.3)
p = p0 + O(M2),
p0 = const.
(4.4)
6


We refer to (4.2), (4.3), (4.4) as well-prepared data and summarize it in the following set
Ωwp =
n
w ∈Rd+2, ∇ρ0 = 0, ∇p0 = 0, ∇· u0 = 0
o
.
(4.5)
For simplicity we show the AP property for the time semi-implicit scheme. The same steps can
be followed with the fully discretized scheme given in Section 5.
To show the AP property we will exploit some properties of the fast pressure ψ(1) obtained
in the implicit step (4.6).
4.1
Asymptotic behaviour of ψ(1)
Due to the sparse structure of G deﬁned in (3.2), the implicit part reduces to solving only two
coupled equations in the relaxation variables ˆu, ψ given by
(ρˆu)t +
1
M2 ∇ψ = 0,
(ρψ)t + a2∇· ˆu = 0,
(4.6)
with the eigenvalues λM = ± a
ρM . As done in [10], we rewrite the coupled system (4.6) into one
equation for ψ starting from the time-semi-discrete scheme
ρ(1) −ρn
∆t
= 0,
(4.7)
(ρˆu)(1) −(ρˆu)n
∆t
+
1
M2 ∇ψ(1) = 0,
(4.8)
(ρψ)(1) −(ρψ)n
∆t
+ a2∇· ˆu(1) = 0.
(4.9)
To emphasize that (4.6) also depends on the density, we have included the density update (4.7)
into the time-semi-discrete system. From equation (4.7) we see that ρ(1) = ρn. To simplify
notation we deﬁne τ n
i =
1
ρn
i . Inserting (4.8) into (4.9) we can reduce the implicit system to only
one equation with an elliptic operator for ψ given as
ψ(1) −∆t2a2
M2 τ n∇· (τ n∇ψ(1)) = ψn −∆ta2τ n∇· un.
(4.10)
On the right hand side of (4.10) we have already made use of the fact that ˆun = un since we
start from equilibrium data. We will see that the correct scaling of ψ(1) with respect to the
Mach number is important not only for showing the AP property of the scheme, but also for the
positivity of density and internal energy as well as for the Mach number independent diﬀusion
of the fully discretized scheme.
To prevent O(M) pressure perturbations at the boundaries which would destroy the well-
prepared nature of the pressure, we require boundary conditions on ψ which preserve the scaling
of the pressure in time. For a computational domain D, we set
ψ(1)
0
= pn
0
ψ(1)
1
= 0
)
on ∂D.
(4.11)
7


Lemma 2 (Scaling of ψ(1)) Let wn ∈Ωwp be equipped with the boundary conditions (4.11).
Then the Mach number expansion of ψ(1) after the ﬁrst stage satisﬁes
ψ(1) = p0 + M2ψ2 + O(M3),
(4.12)
where p0 is constant.
Proof. Let us assume that the expansion of ψ(1) is given by
ψ(1) = ψ0 + Mψ1 + M2ψ2 + O(M3).
(4.13)
Since we start in Ωwp, we have well-prepared data as given in (4.2)(4.3)(4.4). We insert the well-
prepared data wn and (4.13) into the implicit update equation (4.10). Separating the O(M−2)
terms, we ﬁnd using the boundary conditions (4.11)
(
∆ψ0
= 0 in D
ψ0
= p0 on ∂D .
This leads to ψ0 = p0 on D. Separating the O(M−1) terms and using that ψ0 = p0 = const, we
ﬁnd
(
∆ψ1
= 0 in D
ψ1
= 0 on ∂D
(4.14)
which leads to ψ1 = 0 on D. As a last step, we collect the O(M−1) terms and using that ψ0 = p0
as well as ψ1 = 0 on D. It is not necessary to impose special boundary conditions for ψ2. Thus
we ﬁnd
∆ψ2 = 0 in D.
(4.15)
This shows that the fast pressure ψ(1) after the implicit step is still well-prepared.
4.2
Asymptotic preserving property
We show that the time discretization of (2.2) in the M →0 limit coincides with a time discretiza-
tion of the incompressible Euler equations (4.1). We consider well-prepared data wn ∈Ωwb and
the Mach number expansion of ψ(1) from Lemma 2. For the total energy deﬁned in (1.2), we
ﬁnd the following Mach number expansion
E = ρ0e0 + M(ρ1e0 + ρ0e1) + M2(|u0|2 + ρ2e0 + ρ1e1 + ρ0e2) + O(M3)
(4.16)
Inserting the Mach number expansions of wn, ψ(1) and En into the density, momentum and
energy equation of (3.7) and considering the O(M0) order terms we have
ρn+1
0
−ρn
0
∆t
+ ∇· ρn
0un
0 = 0,
ρn+1
0
un+1
0
−ρn
0un
0
∆t
+ ∇·

ρn
0un
0 ⊗un
0 + ψ(1)
2

= 0,
ρn+1en+1
0
−ρn
0en
0
∆t
+ ∇· un
0(ρn
0en
0 + ψ(1)
0 ) = 0,
ρn+1
0
en+1
0
−ρn
0en
0
∆t
+ ∇· un
0(ρn
0en
0 + ψ(1)
0 ) = 0
(4.17)
8


Let us assume, the pressure at time tn+1 has the Mach number expansion pn+1 = pn+1
0
+Mpn+1
1
+
O(M2). Note that we have from Lemma 2 that ψ(1)
0
= p0 and thus ∇ψ(1)
0
= ∇p0 = 0. Since
wn ∈Ωwp we have ∇ρn
0 = 0 and ∇· un
0 = 0. Further we use p0 = (γ −1)ρ0e0. Equipped with
that we can simplify (4.17) to
ρn+1
0
−ρn
0
∆t
= 0,
(4.18)
un+1
0
−un
0
∆t
+ un
0 · ∇un
0 + ∇ψ(1)
2
ρn
0
= 0,
(4.19)
pn+1
0
−pn
0
∆t
= 0.
(4.20)
Especially from equations (4.18) and (4.20) we see that ρn+1
0
and pn+1
0
are constants. Looking
at the O(M1) terms we have from the energy equation
pn+1
1
+ ∆t∇· un
1 = 0.
(4.21)
This means the density and pressure at time tn+1 are well-prepared up to O(∆t) perturbation as
in (4.2), (4.4). To be consistent with a time discretization of the incompressible Euler equations
(4.1) the divergence of u0 at time tn+1 deﬁned by ∇· un+1
0
has to be at least of order ∆t. To
show this, we apply the divergence on the velocity update (4.19) which gives
∇· un+1
0
= ∇· un
0 + ∆t∇· (un
0 · ∇un
0 + ∆ψ(1)
2
ρn
0
).
(4.22)
In the proof of Lemma 2, we have shown that ∆ψ(1)
2
= 0 on ∂D, see (4.15). Using (4.15) together
with ∇· un
0 = 0, we can simplify (4.22) to
∇· un+1
0
= ∆t∇· (un
0 · ∇un
0) = O(∆t).
In summary, we have shown the following theorem.
Theorem 3 (AP property) Let wn ∈Ωwp. Then under the boundary conditions (4.11) the
scheme (3.6), (3.7), (3.8) is asymptotic preserving when M tends to 0, in the sense that if
wn ∈Ωwp then it is wn+1 ∈Ωwp and in the limit M →0 the time-semi-discrete scheme is a
consistent discretization of the incompressible Euler equations (4.1).
5
Derivation of the fully discrete scheme
For simplicity, we develop the fully discretized scheme in one space dimension, but it can be
straightforwardly extended to d dimensions. In the implicit update (4.10), the space derivatives
read
∇· (τ∇ψ) = ∂x1(τ∂x1ψ) + · · · + ∂xd(τ∂xdψ) and ∇· u = ∂x1u1 + · · · + ∂xdud
and in the explicit part we apply dimensional splitting.
In the following we use a cartesian grid on a computational domain D devided in N cells
Ci = (xi−1/2, xi+1/2) of step size ∆x. We use a standard ﬁnite volume setting, where we deﬁne
at time tn the piecewise constant functions
w(x, tn) = wn
i , for x ∈Ci.
9


Using this notation, we apply centered diﬀerences for the implicit update (4.10) and have
ψ(1)
i
−∆t2
∆x2
a2
M2 τ n
i

τ n
i−1/2ψ(1)
i−1 −(τ n
i−1/2 + τ n
i+1/2)ψ(1)
i
+ τ n
i+1/2ψ(1)
i+1

=
ψn
i −∆t
2∆xa2τ n
i
 un
i+1 −un
i−1

,
(5.1)
where τi+1/2 = 1
2 (τi+1 + τi).
For the explicit part, we will use a Godunov type ﬁnite volume scheme following [30] which
we will describe in the following section.
5.1
Godunov type ﬁnite volume scheme
In the explicit step we consider the following equations as deﬁned in (3.2), (3.7)
∂tρ + ∂xρu = 0,
∂tρu + ∂x(ρu2 + π + 1 −M2
M2
ψ) = 0,
∂tE + ∂x((E + M2π + (1 −M2)ψ)u) = 0,
∂tρπ + ∂x((ρπ + a2)u) = 0,
∂tρˆu + ∂x(ρuˆu) = 0,
∂tρψ + ∂x(ρψu) = 0.
(5.2)
In the next result the structure of system (5.2) is summarized.
Lemma 4 System (5.2) admits the linear degenerate eigenvalues λ± = u± a
ρ and λu = u, where
the eigenvalue λu has multiplicity 4. The relaxation parameter a is independent of the Mach
number as well as all eigenvalues. The Riemann Invariants with respect to λu are
Iu
1 = u, Iu
2 = M2π + (1 −M2)ψ
(5.3)
and with respect to λ±
I±
1 = u ± a
ρ, I±
2 = π ∓au,
I±
3 = e −M2
2a2 π2 −1 −M2
a2
πψ,
I±
4 = ˆu,
I±
5 = ψ.
(5.4)
Proof. We rewrite the equations (5.2) using primitive variables V = (ρ, u, e, π, ˆu, ψ) in non-
conservative form
Vt + A(V )Vx = 0,
(5.5)
where the matrix A(V ) is given by
A(V ) =










u
ρ
0
0
0
0
0
u
0
1
ρ
0
1−M2
M2
0
M2π+(1−M2)ψ
ρ
u
0
0
0
0
a2
ρ
0
u
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
u










.
(5.6)
10


It is easy to check that λu, λ± are eigenvalues of A(V ). Associated to the eigenvalues, we ﬁnd
the respective eigenvectors
ru
1 =








0
0
0
1 −
1
M2
0
1








ru
2 =








0
0
0
0
1
0








ru
3 =








0
0
1
0
0
0








ru
4 =








1
0
0
0
0
0








r± =









ρ2
a2
± 1
a
M2π+(1−M2)ψ
a2
1
0
0









We see that ∂V λu · ri = 0, ∀i = 1, . . . , 4 and ∂V λ± · r± = 0. Thus all eigenvalues are linearly
degenerate. A scalar function I(V ) is a Riemann invariant if for all eigenvectors r associated
to an eigenvalue λ, ∂V I(V ) · r = 0 holds. It is straightforward to check, that (5.3) and (5.4)
are Riemann invariants. Since Riemann invariants are invariant under change of variables, the
Riemann invariants of (5.5) are the same as for the equations in conservation form (5.2). For
more details, see [26, 21].
We will follow the theory of Harten, Lax and van Leer [30] for deriving an approximate
Riemann solver WRS

x
t ; W (1)
L , W (1)
R

based on the states W (1) after the implicit step. Due
to the linear-degeneracy from Lemma 4, the structure of the approximate Riemann solver, as
displayed in Figure 5.1, is given as follows
WRS
x
t ; W (1)
L , W (1)
R

=











W (1)
L
x
t < λ−,
W ∗
L
λ−< x
t < λu,
W ∗
R
λu < x
t < λ+,
W (1)
R
λ+ < x
t .
(5.7)
To compute the intermediate states W ∗
L,R, we use the Riemann invariants as given in Lemma 4.
u −a/ρ
x = 0
u + a/ρ
u
WL
WR
W ∗
L
W ∗
R
Figure 1: Structure of the Riemann solution.
Note that since the eigenvalues λ± have multiplicity 1, we get the expected 5 Riemann invariants.
This does not hold in general for eigenvalues with multiplicity larger than 1. Nevertheless, the
invariants (5.3) and (5.4) give enough relations to determine the solution to a Riemann problem
for (5.2) as shown in the following lemma.
Lemma 5 Consider an initial value problem with initial data W = W (1) given by
W0(x) =
(
WL
x < 0,
WR
x > 0.
(5.8)
11


Then the solution consists of four constant states separated by contact discontinuities with the
structure given in (5.7). The solution for the states W ∗
L/R is given by
1
ρ∗
L
= 1
ρL
+ 1
a(u∗−uL),
1
ρ∗
R
= 1
ρR
+ 1
a(uR −u∗),
u∗= u∗
L = u∗
R = 1
2(uL + uR) + 1
2a

(πL −πR) + 1 −M2
M2
(ψL −ψR)

,
π∗
L = 1
2(πL + πR) + a
2(uL −uR) −1 −M2
M2
1
2(ψL −ψR),
π∗
R = 1
2(πL + πR) + a
2(uL −uR) + 1 −M2
M2
1
2(ψL −ψR),
e∗
L = eL −
1
2a2

(π2
L −(π∗
L)2) + (1 −M2)(πL −π∗
L)ψL

,
e∗
R = eR −
1
2a2

(π2
R −(π∗
R)2) + (1 −M2)(πR −π∗
R)ψR

,
ψ∗
L,R = ψL,R,
ˆu∗
L,R = ˆuL,R.
(5.9)
Proof. Since a
ρ > 0, we have the following order of the eigenvalues u −a
ρ < u < u + a
ρ. The
solution structure follows directly from the linear degeneracy of the eigenvalues given in Lemma
4 and the ordering of the eigenvalues. To derive the solution for the intermediate states W ∗
L,R
one uses the Riemann invariants given in (5.3) and (5.4) and solves the resulting system of
equations.
Given the solution of the Riemann problem (5.7), we deﬁne the numerical ﬂuxes as follows
Fi+1/2 =











F(W (1)
i
)
λ−> 0
F(W ∗,(1)
i
)
λu > 0 > λ−
F(W ∗,(1)
i+1 )
λ+ > 0 > λu
F(W (1)
i+1)
λ+ < 0
(5.10)
where the superscript (1) emphasizes that the states after the implicit step are used. To avoid
interactions between the approximate Riemann solvers at the interfaces xi+1/2, we have a CFL
restriction on the time step of
∆t ≤1
2
∆x
max
i |ui ± a/ρi|
(5.11)
which is independent of the Mach number. This leads to the following update of the explicit
part
W (2)
i
= W (1)
i
−∆t
∆x(Fi+1/2 −Fi−1/2).
(5.12)
At this point, we can make some remarks on the numerical scheme as it is given by (5.1),(5.12)
and (3.8).
12


Remark 6 The ﬂux function G in (3.2) in the implicit part and the relaxation source term R
only act on the relaxation variables (π, ˆu, ψ). Thus, we can immediately give the update of the
physical variables w as
wn+1
i
= wn
i −∆t
∆x

f

QWRS

0; W (1)
i
, W (1)
i+1

−f

QWRS

0; W (1)
i−1, W (1)
i

,
(5.13)
where the matrix Q is the projection on the ﬁrst 2 + d components as deﬁned in (2.6).
Remark 7 We see from (5.1) and (5.9), that the updates in the implicit and explicit step are
independent of ˆu.
Therefore it is not necessary to explicitly compute ˆu in the scheme thus
reducing the computational costs.
Remark 8 For the limit case M = 1 the relaxation model (2.2) reduces to a Suliciu relaxation
model for the compressible Euler equations since all ψ-terms are being cancelled and the eigen-
values λ± and λ±
M collapse. Analogously, in the scheme the implicit step becomes redundant,
since in the Riemann solution (5.9) the ψ-terms are cancelled, too. Thus the scheme reduces to
a Godunov-type approximate Riemann solver based explicit scheme for the compressible Euler
equations as can be found in [26].
For M > 1 the order of the eigenvalues changes to λ−< λ−
M < λu < λ+
M < λ+. The scheme
is still stable because the CFL condition (5.11) comes from the explicit eigenvalues which are
now the largest ones.
5.2
Positivity of density and internal energy
For physical applications it is necessary that the density and internal energy be positive. We
deﬁne the physical admissible states associated with the Euler equations (1.1) as
Ωphy =
n
w ∈Rd+2, ρ > 0, e > 0
o
.
(5.14)
The property of our scheme to preserve the domain Ωphy is linked to how the ﬂuxes are calculated.
In our case it is essential that the density and internal energy of the Riemann solution (5.7),(5.9)
are positive. This is shown in the following lemma.
Lemma 9 Let the initial data of the Riemann problem (5.8) be given as W (1)
L,R ∈Ωphy ∩Ωwp
satisfying the boundary conditions (4.11). Then there is a relaxation parameter large enough but
independent of M such that WRS(x
t , W (1)
L , W (1)
R ) ∈Ωphy.
Proof. Since the proof only concerns data after the implicit step, we will drop the superscript
(1). We have to prove, that ρ∗
L,R > 0 and e∗
L,R > 0. From the ordering of eigenvalues uL−a
ρL < u∗
and the formula for ρ∗
L from (5.9), we get
1
ρ∗
L
= 1
ρL
+ u∗−uL
a
≥1
ρL
−1
ρL
= 0.
13


Analogously, we ﬁnd ρ∗
R ≥0. For the internal energy e∗
L, we insert the deﬁnition of π∗
L from
(5.9) into e∗
L to obtain a formula depending only on the left and right states WL,R
e∗
L = eL + 1
8(uL −uR)2+
1
2a2
 
−π2
L + 1
4

πL + πR + 1 −M2
M2
(ψR −ψL)
2
+1
2ψL(1 −M2)

πR −πL + 1 −M2
M2
(ψR −ψL)

−1
4a (uL −uR)

πL + πR + 1 −M2
M2
(ψR −ψL) + (1 −M2)ψL

.
From Lemma 2, we know ψL,R = p0 + O(M2). The diﬀerence ψR −ψL = O(M2) cancels with
the 1/M2 which leads to e∗
L = O(1). All possibly negative terms in e∗
L can then be controlled
by the relaxation parameter a > 0 independent of the Mach number. The same argument holds
for e∗
R.
The positivity property for the ﬁrst order scheme is given in the next result.
Theorem 10 (Positivity property 1) Let the initial state be given as
wn
i ∈Ω= Ωphy ∩Ωwp
in d dimensions satisfying the boundary condition described in (4.11). Then under the Mach
number independent CFL condition
∆t
∆xmax
i
|λ±(wn
i )| <
1
2d, the numerical scheme deﬁned by
(5.1),(5.13) preserves the positivity of density and internal energy, that is
wn+1
i
∈Ωphy.
Proof. Due to the construction of the numerical scheme, the update of the physical variables
is only done through the explicit step (5.13). Therefore we can adopt the proof of Theorem 3
in [31] using the positivity of the Riemann solver from Lemma 9. The key element is, that we
can write the update wn+1
i
in one dimension as a convex combination of Riemann solvers which
satisfy WRS ∈Ωphy according to Lemma 9:
wn+1
i
=
1
∆x


Z xi
xi−1
2
QWRS
 x
tn+1 , Wi−1, Wi

dx +
Z xi+ 1
2
xi
QWRS
 x
tn+1 , Wi, Wi+1

dx


(5.15)
Since we are using dimensional splitting, the update in d dimensions can be written as a sum of
updates as given in (5.15) in each dimension and due to convexity we have wn+1
i
∈Ωphy. More
details can be found for example in [31].
5.3
Mach number independent diﬀusion
Although we are using a Godunov type upwind scheme in the explicit part, our scheme does not
suﬀer from an excessive numerical diﬀusion as M tends to 0. As we will show in the following
14


this is due to the well-prepared implicitly treated fast pressure ψ(1).
In order to do so, we
investigate the numerical diﬀusion vector D ∈R2+d deﬁned by
D = f(wn
i ) + f(wn
i+1)
2
−f

QWRS

W (1)
i
, W (1)
i+1

,
(5.16)
where the Matrix Q is deﬁned in (2.6). Given well-prepared initial data wn
i ∈Ωwp, we have the
following Mach number expansion for the physical variables as given in (4.2),(4.3) and (4.4)
ρi
=
ρ0 + O(M),
ρi+1
=
ρ0 + O(M)
ui
=
u0,i + O(M),
ui+1
=
u0,i+1 + O(M)
ei
=
e0 + O(M),
ei+1
=
e0 + O(M)
πi
=
p0 + O(M2),
πi+1
=
p0 + O(M2)
(5.17)
From Lemma 2, we have for ψ after the implicit step
ψi = p0 + O(M2),
ψi+1 = p0 + O(M2).
(5.18)
The Mach number expansion of the states W (1)
i
, W (1)
i+1 used in the Riemann solver WRS is com-
posed of the expansions (5.17) and (5.18). Inserting them into the formulas of the intermediate
states (5.9) of the Riemann solution (5.7), we have the following scaling of W ∗
i , W ∗
i+1 with respect
to the Mach number
τ ∗
i
=
τ0 + O(1),
τ ∗
i+1
=
τ0 + O(1)
u∗
i+1/2
=
u0,i+u0,i+1
2
+ O(1) = u0,i+1/2 + O(1)
e∗
i
=
e0 + O(1),
e∗
i+1
=
e0 + O(1)
π∗
i
=
p0 + O(1),
π∗
i+1
=
p0 + O(1),
ψ∗
i
=
p0 + O(M2),
ψ∗
i+1
=
p0 + O(M2).
(5.19)
From (5.19) it is evident that the lowest order of M in the intermediate states is O(M0).
Inserting (5.19) in the interface ﬂux gives
f

QWRS

W (1)
i
, W (1)
i+1

=


ρ0u0,i+1/2 + O(1)
ρ0u2
0,i+1/2 + p0
M2 + O(1)
u0,i+1/2(E0 + p0) + O(1)

.
(5.20)
Therefore, using (5.20) in (5.16), the diﬀusion vector is given by
D =


O(1)
O(1)
O(1)

.
This shows that the diﬀusion introduced by the Riemann solver does not suﬀer from a O(M−1)
dependent diﬀusion in the momentum equation.
6
Second order extension
In this section we extend the ﬁrst order scheme to second order accuracy. We seek a natural
extension of the ﬁrst order scheme that preserves the positivity property as shown in Theorem
10.
15


6.1
Second order time integration scheme
For the second order extension, we use a modiﬁed two stage time integration as can be found in
[25]. It is a convex combination of ﬁrst order steps and is given by
W
(1) =W eq,n + ∆t
M2 ∇· G(W
(1)),
W
(2) =W
(1) + ∆t∇· F(W
(1)),
W
(1) =W
eq,(2) + ∆t
M2 ∇· G(W
(1)),
W
(2) =W
(1) + ∆t∇· F(W
(1)),
W n+1 =1
2W
eq,(2) + 1
2W eq,n.
(6.1)
The relaxation equilibrium states W eq,n and W eq,(2) are deﬁned as in (2.5). As in the ﬁrst order
case, we can rewrite the integration scheme (6.1) in terms of the update for the physical variables
given in Remark 6. The ﬁrst order time update (5.13) can be written as
wn+1 = wn −∆t∇· f(QW (1)),
where Q is deﬁned in (2.6). Based on this we can reformulate (6.1) in terms of updates of the
physical variables as
w =wn −∆t∇· f(QW
(1))
w =w −∆t∇· f(QW
(1))
wn+1 = 1
2wn + 1
2w.
(6.2)
where we obtain W (1) and W
(1) from the implicit step (3.6) using the update formula (5.1) to
compute ψ(1) and ψ
(1). That the time integration scheme is second order accurate is numerically
validated in Section 7. The time integration scheme (6.2) can be extended to variable step sizes
∆t1, ∆t2 for each stage respectively as given in [25].
This has the advantage that the CFL
criterion can be met at each stage independently. It is given by
w =wn −∆t1∇· f(QW
(1))
w =w −∆t2∇· f(QW
(1))
wn+1 =

1 −2∆t1∆t2
∆t1 + ∆t2

wn + 2∆t1∆t2
∆t1 + ∆t2
w.
(6.3)
6.2
Second order space reconstruction
Following Remark 6, we focus on the update of the physical variables in the explicit step. As it
is standard in the ﬁnite volume setting, we apply a reconstruction to get a higher accuracy for
the interface values wi+1/2. To get second order, we consider piecewise linear functions in the
conserved variables w = (ρ, ρu, E) at time level tn and in ψ(1). As we are working on a Cartesian
16


grid, we reconstruct along each dimension separately. In one dimension the linear function in
(wi, xi) on (xi−1/2, xi1/2) is given by
wn(x) = wn
i + σi(x −xi),
where σi = (σρ
i , σρu
i , σE
i ). For the multi-dimensional notation, we refer to [31]. The slopes σi
are computed from the neighbouring cells using a limiter function. To have a seconder order
extension that preserves the positivity properties of the ﬁrst order scheme, see Theorem 10, we
choose the minmod limiter deﬁned as
minmod(x, y) =





min(x, y)
if x, y ≥0
max(x, y)
if x, y ≤0
0
else
and deﬁne the slopes as
σi = minmod(wn
i −wn
i−1
∆x
, wn
i+1 −wn
i
∆x
).
(6.4)
The same procedure is applied on ψ(1)
i
. Since we are using the minmod limiter on the conservative
variables to determine the slope, we immediately get for wn
i ∈Ωphy ∩Ωwp that for the interface
values holds
w+
i−1/2, w−
i+1/2 ∈Ωphy ∩Ωwp
and the expansion of the interface values ψ(1),+
i−1/2, ψ(1),−
i+1/2 with respect to the Mach number is
preserved. This means by Lemma 9 that the Riemann problem applied on the interface values
w∓
i+1/2, ψ∓
i+1/2 still ensures the positivity of density and internal energy. By Theorem 10 the
ﬁrst order scheme has the positivity property and the second order scheme (6.2) is a convex
combination of states in Ωphy.
From this we have wn+1
i
∈Ωphy.
We have thus proven the
following result for the second order scheme:
Theorem 11 (Positivity property 2) Let the initial state be given as
wn
i ∈Ωphy ∩Ωwp
with the boundary condition described in (4.11).
Then under the Mach number independent
CFL condition
∆t
∆xmax
i |λ±(wn
i )| <
1
2·2d, the numerical scheme deﬁned by (6.1)/ (6.2) preserves
the positivity of density and internal energy, that is
wn+1
i
∈Ωwp.
7
Numerical results
In the following section, we numerically validate the theoretical properties of the proposed
scheme. Throughout the test cases, we assume an ideal gas law with the equation of state given
by
p = (γ −1)ρe.
For solving the implicit non-symmetric linear system given by (5.1), we use the GMRES algo-
rithm combined with a Jacobi diagonal scaling preconditioner for periodic boundary conditions
used in the test cases in Section 7.2 and an preconditioner based on an incomplete LU decom-
position for von Neumann boundary conditions used in the test cases from Section 7.1.
17


7.1
Shock test cases
To verify that our scheme captures discontinuities accurately, we perform a SOD shock tube
test [32] in the regime M = 1 and a Mach number dependent shock test case taken from [20].
7.1.1
SOD shock tube test
The computational domain for the SOD shock tube test [32] is [0, 1] and the initial data is given
using γ = 1.4 by
ρL = 1 kg
m3 ,
ρR = 0.125 kg
m3 ,
uL = 0 m
s ,
uR = 0 m
s ,
pL = 1
kg
ms2 ,
pR = 0.1
kg
ms2 ,
where we place the initial discontinuity at x = 0.5. Since the regime is compressible, we set
M = 1 and the initial data is given in dimensional form. This test also demonstrates that the
collapse of the eigenvalues λ± and λ±
M in the case of M = 1 as mentioned in Remark 8 is not
problematic. We see in Figure 2 that the ﬁrst order as well as the second order scheme captures
the shock positions correctly. As expected, the second order scheme is more accurate than the
ﬁrst order scheme.
0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.4
0.6
0.8
1.0
Density
exact
N=200 (1st)
N=200 (2nd)
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
Velocity
exact
N=200 (1st)
N=200 (2nd)
0.0
0.2
0.4
0.6
0.8
1.0
x
0.2
0.4
0.6
0.8
1.0
Pressure
exact
N=200 (1st)
N=200 (2nd)
Figure 2: SOD: First order scheme, T = 0.1644, γ = 1.4
7.1.2
Mach number dependent shock problem
The following test case is a low Mach ﬂow, where the contact wave travels with the Mach number
M = 6.2 · 10−3. The initial data is taken from [20] and is given as
ρL = 1 kg
m3 ,
ρR = 1 kg
m3 ,
uL = 0 m
s ,
uR = 0.008 m
s ,
pL = 0.4
kg
ms2 ,
pR = 0.399
kg
ms2 .
(7.1)
The discontinuity is placed at x0 = 0.5 on the domain [0, 1] with γ = 1.4 and the ﬁnal time
is given by T = 0.25s. To transform the initial data (7.1) into non-dimensional quantities, we
deﬁne for a variable φ the relation φ = φr ˆφ, where φ denotes the dimensional variable, φr the
18


reference value which contains the units and ˆφ the non-dimensional quantity. For the reference
values we have the following relations
tr = xr
tr
, pr = ρrc2
r, cr = ur
M .
(7.2)
We choose the scaling in space to be xr = 1m, the scaling in density to be ρr = 1 kg
m3 and the
scaling in velocity to be ur = M m
s . This yields the following scaling of the sound speed cr = 1m
s ,
the time tr = Ms and the pressure pr = 1 kg
ms2 . Then the non-dimensional initial data is given
as
ρL = 1,
ρR = 1,
uL = 0,
uR = 0.008/M,
pL = 0.4,
pR = 0.399.
(7.3)
In the simulation, we choose M = 6.2 · 10−3 which is the Mach number on the contact wave.
In Figure 3 we show the inﬂuence of the space and time step on the density proﬁle computed
with the ﬁrst order scheme (IMEX1) and the second order scheme (IMEX2). The contact wave
is always reproduced sharply whereas the acoustic waves are smoothed since the time step is
chosen according to the CFL restriction (5.11) associated with the contact wave.
In Figure 4 our results computed with IMEX1 and IMEX2 are plotted against the implicit
Jin Xin relaxation scheme presented in [20] and an explicit upwind scheme. Our results obtained
with the IMEX schemes are in good agreement with the results of the implicit scheme. Since
the time step for the explicit upwind scheme depends on the Mach number dependent acoustic
waves, all waves are resolved. This is rather costly since it results in a very small time step.
The implicit scheme is unconditionally stable and the time step can be chosen with respect to
the desired accuracy of the numerical solution.
7.2
Gresho Vortex test
In order to demonstrate the low Mach properties, we calculate the solution to the Gresho
Vortex test as given in [2]. The Gresho vortex is a steady state solution of the compressible
Euler equations and the velocity ﬁeld is divergence free.
The initial angular velocity in m
s is given by
uφ =





5r
for 0 ≤r < 0.2
2 −5r
for 0.2 ≤r < 0.4
0
for 0.4 ≤r
,
where r =
p
(x −x0)2 + (y −y0)2 with x0 = 0.5, y0 = 0.5 on a computational domain of [0, 1]
and γ = 5/3. The pressure distribution in
kg
ms2 is given by
p =





p0 + 12.5r2
for 0 ≤r < 0.2
p0 + 12.5r2 + 4(1 −5r −log(0.2) + log(r))
for 0.2 ≤r < 0.4
p0 −2 + 4 log(2)
for 0.4 ≤r
19


0.0
0.2
0.4
0.6
0.8
1.0
x
0.994
0.995
0.996
0.997
0.998
0.999
1.000
density [kg/m3]
exact
IMEX1 N = 1500
IMEX1 N = 2000
IMEX2 N = 1500
IMEX2 N = 2000
Figure 3: Density proﬁle for the Mach number dependent test case with diﬀerent number of grid
point and time steps. Time steps for IMEX1 ∆t = 7.8 · 10−3s (blue), ∆t = 5.8 · 10−3s (orange),
time steps for IMEX2 ∆t = 3.9 · 10−3s (green), ∆t = 2.9 · 10−3s (red)
0.0
0.2
0.4
0.6
0.8
1.0
x
0.994
0.995
0.996
0.997
0.998
0.999
1.000
density [kg/m3]
expl
impl
IMEX1
IMEX2
0.0
0.2
0.4
0.6
0.8
1.0
x
0.000
0.001
0.002
0.003
0.004
0.005
0.006
0.007
0.008
velocity [m/s]
expl
impl
IMEX1
IMEX2
0.0
0.2
0.4
0.6
0.8
1.0
x
0.3965
0.3970
0.3975
0.3980
0.3985
0.3990
0.3995
0.4000
pressure [kg/ms2]
expl
impl
IMEX1
IMEX2
Figure 4: Mach dependent shock test case: Time steps are given by ∆t = 5.3 · 10−4s (expl.),
∆t = 2.2 · 10−2s (impl.), ∆t = 5.0 · 10−3s (IMEX1), ∆t = 2.5 · 10−3s (IMEX2).
with p0 =
ρ0u2
Φ,max
γM2
kg
ms2, where ρ0 = 1. The initial density is given by ρ = 1 kg
m3 and we transform
the initial condition in non-dimensional quantities by using xr = 1m, ρr = 1 kg
m3 , ur = 2·0.2 π m
s ,
pr = ρ0u2
0
γM2
kg
ms2 and tr = 1 m
ur . The computational domain is given by [0, 1] × [0, 1] and we use a
40 × 40 grid with periodic boundary conditions. The results for a full turn of the vortex using
the IMEX2 scheme together with the initial distribution of the Mach number are given in Figure
5. We see that even for a low resolution, the solution at T = 1 shows only little dissipation
throughout all tested Mach numbers. To further check the quality of the numerical simulation,
we monitor the loss of kinetic energy. Since the Gresho vortex is a stationary solution, the kinetic
energy should be preserved. In Figure 6 we show the ratio between the initial kinetic energy
Ekin,0 and the kinetic energy after each time step Ekin,t for the Mach numbers M = 10−2, 10−3.
The graphs for the diﬀerent Mach numbers are indistinguishable which shows that the loss of
kinetic energy does not depend on the chosen Mach number but depends on the chosen space
discretization and time-step.
20


0.2
0.4
0.6
0.8
x
0.2
0.4
0.6
0.8
y
M
0.00
0.02
0.04
0.06
0.08
0.10
0.2
0.4
0.6
0.8
x
0.2
0.4
0.6
0.8
y
M
0.00
0.02
0.04
0.06
0.08
0.10
0.2
0.4
0.6
0.8
x
0.2
0.4
0.6
0.8
y
M
0.000
0.002
0.004
0.006
0.008
0.010
0.2
0.4
0.6
0.8
x
0.2
0.4
0.6
0.8
y
M
0.0000
0.0002
0.0004
0.0006
0.0008
0.0010
Figure 5: Mach number distribution for diﬀerent maximal Mach numbers. Top left: Initial state
for M = 10−1. Top right: M = 10−1, bottom left: M = 10−2, bottom right: M = 10−3 at
t = 1.
7.3
A smooth Gresho Vortex test
In this section we want to derive a test case to numerically validate the second order accuracy
of the second order extension of Section 6. Unfortunately the Gresho vortex is not a smooth
enough solution to test second order accuracy, since its velocity proﬁle is only continuous but
not continuously diﬀerentiable. Therefore we propose a smoothed velocity proﬁle with which
we calculate a pressure proﬁle to gain a stationary vortex. A twice continuously diﬀerentiable
angular velocity in m
s with uΦ,max = 1 and uΦ(0) = 0 and uΦ(0.4) = 0 is given by
uΦ =





75r2 −250r3,
for 0 ≤r < 0.2
−4 + 60r −225r2 + 250r3,
for 0.2 ≤r < 0.4
0
for 0.4 ≥r
(7.4)
with the radius r =
p
(x −0.5)2 + (y −0.5)2. The proﬁle can be easily modiﬁed to be a Ck
function, where k ∈N denotes the degree of continuous diﬀerentiability. Under the condition
that the centrifugal forces are balanced that is ∂rp = uΦ(r)2
r
, we can calculate the pressure in
kg
ms2 as
p =





p0 + 1406.25r4 −7500r5 + (10416 + 2
3)r6
for 0 ≤r < 0.2
p0 + p2(r)
for 0.2 ≤r < 0.4
p0 + p2(0.4)
for 0.4 ≥r
(7.5)
21


0.0
0.2
0.4
0.6
0.8
1.0
t
0.90
0.92
0.94
0.96
0.98
1.00
Ekin, t/Ekin, 0
M = 10−2, N = 40
M = 10−3, N = 40
M = 10−2, N = 80
M = 10−3, N = 80
M = 10−2, N = 160
M = 10−3, N = 160
Figure 6: Loss of kinetic energy for diﬀerent grids and Mach numbers after one full turn of the
vortex (non-dimensional).
where
p2(r) =65.8843399322788 −480r + 2700r2 −(9666 + 2
3)r3
(7.6)
+ 20156.25r4 −22500r5 + (10416 + 2
3)r6 + 16 ln(r).
(7.7)
As in the Gresho test case, the background pressure is scaled with the Mach number as
p0 =
ρ0u2
Φ,max
γM2
kg
ms2 .
(7.8)
To transform the dimensional data into non-dimensional quantities, we use the same reference
values as in the Gresho vortex test case.
To show the accuracy of the IMEX2 scheme, we compute the solution of the smooth Gresho
vortex test on the domain [0, 1]2 with periodic boundary conditions. In Table 1 the L1-error
between the numerical solution at T = 0.05 and the initial conﬁguration in non-dimensional
quantities as well as the convergence rates are displayed for M = 10−1, 10−2, 10−3. It can be
seen that we reach the expected accuracy independently of the chosen Mach number.
8
Conclusion and future developments
We have proposed an all-speed IMEX scheme for the Euler equations of gas dynamics which is
based on Suliciu relaxation model. The proposed IMEX scheme is an improvement of explicit
schemes since the time step is restricted by the material and not the acoustic wave speeds. It
is also an improvement of implicit schemes since the implicit part consists only of one linear
equation and can be solved very eﬃciently. The scheme has the correct numerical viscosity for all
Mach numbers as shown by the Gresho vortex test case, can capture the correct shock positions
as shown by the SOD shock test case as well as for Mach number dependent shock problems.
In addition it is positivity preserving and shows the expected second order convergence rates.
In a future work, the scheme will be extended to the Euler equations with a gravitational
source term. The aim is to design a multi-dimensional well-balanced scheme, that inherits all
the properties of the IMEX scheme presented here.
22


M
N
ρ
u1
u2
p
10−1
20
1.810·10−3
—
1.729·10−2
—
1.729·10−2
—
1.921·10−3
—
40
3.705·10−4
2.288
5.070·10−3
1.770
5.070·10−3
1.770
3.956·10−4
2.279
60
1.246·10−4
2.688
2.403·10−3
1.842
2.403·10−3
1.842
1.343·10−3
2.665
80
5.510·10−5
2.835
1.396·10−3
1.887
1.396·10−3
1.887
5.922·10−5
2.845
10−2
20
1.812·10−3
—
1.731·10−2
—
1.731·10−2
—
1.912·10−3
—
40
3.582·10−4
2.339
5.057·10−3
1.775
5.057·10−3
1.775
3.781·10−4
2.337
60
1.162·10−4
2.777
2.402·10−3
1.837
2.402·10−3
1.837
1.226·10−4
2.777
80
4.881·10−5
3.014
1.386·10−3
1.910
1.386·10−3
1.910
5.151·10−5
3.014
10−3
20
1.811·10−3
—
1.731·10−2
—
1.731·10−2
—
1.912·10−3
—
40
3.580·10−4
2.339
5.057·10−3
1.775
5.057·10−3
1.775
3.778·10−4
2.339
60
1.162·10−4
2.775
2.402·10−3
1.836
2.402·10−3
1.836
1.227·10−4
2.775
80
4.875·10−5
3.019
1.382·10−3
1.920
1.382·10−3
1.920
5.146·10−5
3.019
Table 1: L1-error and convergence rates for the solution of the smooth Gresho vortex test at
T = 0.05 (non-dimensional).
Aknowledgements
G. Puppo ackwowledges the support by the GNCS-INDAM 2019 research project and A. Thomann
the support of the INDAM-DP-COFUND-2015, grant number 713485. Some material of this
work has been improved during the SHARK-FV conference (Sharing Higher-order Advanced
Know-how on Finite Volume http://www.SHARK-FV.eu/) held in 2019.
References
[1] St´ephane Dellacherie. Analysis of Godunov type schemes applied to the compressible Euler
system at low Mach number. Journal of Computational Physics, 229(4):978–1016, 2010.
[2] Fabian Miczek, Friedrich K R¨opke, and Philip VF Edelmann. New numerical solver for
ﬂows at various Mach numbers. Astronomy & Astrophysics, 576:A50, 2015.
[3] Emanuela Abbate, Angelo Iollo, and Gabriella Puppo. An implicit scheme for moving walls
and multi-material interfaces in weakly compressible materials. to appear on CiCP, 2019.
[4] Sergiu Klainerman and Andrew Majda. Compressible and incompressible ﬂuids. Commu-
nications on Pure and Applied Mathematics, 35(5):629–651, 1982.
[5] Steven Schochet. The mathematical theory of low Mach number ﬂows. ESAIM: Mathemat-
ical Modelling and Numerical Analysis, 39(3):441458, 2005.
[6] E. Feireisl, C. Klingenberg, and S. Markfelder. On the low Mach number limit for the
compressible Euler system.
SIAM Journal on Mathematical Analysis, 51(2):1496–1513,
2019.
23


[7] Philip L Roe. Approximate Riemann solvers, parameter vectors, and diﬀerence schemes.
Journal of computational physics, 43(2):357–372, 1981.
[8] Herv´e Guillard and C´ecile Viozat. On the behaviour of upwind schemes in the low Mach
number limit. Computers & ﬂuids, 28(1):63–86, 1999.
[9] Eli Turkel. Preconditioned methods for solving the incompressible and low speed compress-
ible equations. Journal of Computational Physics, 72(2):277 – 298, 1987.
[10] Floraine Cordier, Pierre Degond, and Anela Kumbaro. An asymptotic-preserving all-speed
scheme for the Euler and Navier–Stokes equations.
Journal of Computational Physics,
231(17):5685–5704, 2012.
[11] G. Dimarco, R. Loubre, and M. Vignal. Study of a new asymptotic preserving scheme for
the Euler system in the low Mach number limit. SIAM Journal on Scientiﬁc Computing,
39(5):A2099–A2128, 2017.
[12] S Boscarino, G Russo, and L Scandurra.
All Mach number second order semi-implicit
scheme for the Euler equations of gas dynamics. Journal of Scientiﬁc Computing, 77(2):850–
884, 2018.
[13] R. Klein. Semi-implicit extension of a Godunov-type scheme based on low Mach number
asymptotics I: One-dimensional ﬂow. Journal of Computational Physics, 121(2):213 – 237,
1995.
[14] Sebastian Noelle, Georgij Bispen, Koottungal Revi Arun, Maria Lukacova-Medvidova, and
C-D Munz. A weakly asymptotic preserving low mach number scheme for the Euler equa-
tions of gas dynamics. SIAM Journal on Scientiﬁc Computing, 36(6):B989–B1024, 2014.
[15] Shi Jin and Zhouping Xin. The relaxation schemes for systems of conservation laws in
arbitrary space dimensions. Communications on pure and applied mathematics, 48(3):235–
276, 1995.
[16] I Suliciu. On modelling phase transitions by means of rate-type constitutive equations.
Shock wave structure. International Journal of Engineering Science, 28(8):829–841, 1990.
[17] Gui-Qiang Chen, C David Levermore, and Tai-Ping Liu. Hyperbolic conservation laws with
stiﬀrelaxation terms and entropy. Communications on Pure and Applied Mathematics,
47(6):787–830, 1994.
[18] Tai-Ping Liu. Hyperbolic conservation laws with relaxation. Communications in Mathe-
matical Physics, 108(1):153–175, Mar 1987.
[19] Roberto Natalini. Convergence to equilibrium for the relaxation approximations of conser-
vation laws. Communications on Pure and Applied Mathematics, 49(8):795–823, 1996.
[20] Emanuela Abbate, Angelo Iollo, and Gabriella Puppo. An all-speed relaxation scheme for
gases and compressible materials. Journal of Computational Physics, 351:1–24, 2017.
[21] Christophe
Berthon,
Christian
Klingenberg,
and
Markus
Zenk.
An
all
Mach
number relaxation upwind scheme.
submitted, Preprint,https://ifm.mathematik.uni-
wuerzburg.de/˜klingen/Publications ﬁles/BerthonKlingenZenk2018.pdf, 2018.
24


[22] Giacomo Dimarco, Rapha¨el Loub`ere, Victor Michel-Dansac, and Marie-H´el`ene Vignal.
Second-order implicit-explicit total variation diminishing schemes for the Euler system in
the low Mach regime. Journal of Computational Physics, 372:178–201, 2018.
[23] Chi-Wang Shu. Essentially non-oscillatory and weighted essentially non-oscillatory schemes
for hyperbolic conservation laws. In Advanced numerical approximation of nonlinear hyper-
bolic equations, pages 325–432. Springer, 1998.
[24] Isabella Cravero, Gabriella Puppo, Matteo Semplice, and Giuseppe Visconti. CWENO:
uniformly accurate reconstructions for balance laws.
Mathematics of Computation,
87(312):1689–1719, 2018.
[25] Christophe Berthon. Stability of the MUSCL schemes for the Euler equations. Communi-
cations in Mathematical Sciences, 3(2):133–157, 2005.
[26] Fran¸cois Bouchut. Nonlinear stability of ﬁnite volume methods for hyperbolic conservation
laws and well-balanced schemes for sources. Frontiers in Mathematics. Birkh¨auser Verlag,
Basel, 2004.
[27] F. Coquel and B. Perthame. Relaxation of energy and approximate Riemann solvers for
general pressure laws in ﬂuid dynamics. SIAM Journal on Numerical Analysis, 35(6):2223–
2249, 1998.
[28] Rupert Klein, Nicola Botta, Thomas Schneider, Claus-Dieter Munz, Sabine Roller, Andreas
Meister, L Hoﬀmann, and Thomas Sonar. Asymptotic adaptive methods for multi-scale
problems in ﬂuid mechanics. Journal of Engineering Mathematics, 39(1):261–343, 2001.
[29] Lorenzo Pareschi and Giovanni Russo. Implicit–explicit Runge–Kutta schemes and applica-
tions to hyperbolic systems with relaxation. Journal of Scientiﬁc computing, 25(1):129–155,
2005.
[30] Amiram Harten, Peter D. Lax, and Bram Van Leer. On upstream diﬀerencing and Godunov-
type schemes for hyperbolic conservation laws. SIAM Review, 25:35–61, 1983.
[31] Andrea Thomann, Markus Zenk, and Christian Klingenberg. A second-order positivity-
preserving well-balanced ﬁnite volume scheme for Euler equations with gravity for arbitrary
hydrostatic equilibria. International Journal for Numerical Methods in Fluids, 89(11):465–
482, 2019.
[32] Gary A Sod. A survey of several ﬁnite diﬀerence methods for systems of nonlinear hyperbolic
conservation laws. Journal of computational physics, 27(1):1–31, 1978.
25


