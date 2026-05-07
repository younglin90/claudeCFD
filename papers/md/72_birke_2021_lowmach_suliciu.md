# A low Mach two-speed relaxation scheme for the compressible Euler equations with gravity

Claudius Birke∗†, Christophe Chalons‡ , Christian Klingenberg<sup>∗</sup>

March 6, 2023

# Abstract

We present a numerical approximation of the solutions of the Euler equations with a gravitational source term. On the basis of a Suliciu type relaxation model with two relaxation speeds, we construct an approximate Riemann solver, which is used in a first order Godunov-type finite volume scheme. This scheme can preserve both stationary solutions and the low Mach limit to the corresponding incompressible equations. In addition, we prove that our scheme preserves the positivity of density and internal energy, that it is entropy satisfying and also guarantees not to give rise to numerical checkerboard modes in the incompressible limit. Later we give an extension to second order that preserves positivity, asymptotic-preserving and well-balancing properties. Finally, the theoretical properties are investigated in numerical experiments.

Keywords Euler equations, finite volume methods, relaxation, well-balancing, low Mach, asymptotic-preserving, entropy satisfying, checkerboard modes, positivity preserving

AMS subject classification 65M08, 76M12

# 1 Introduction

The goal of this paper is to find a numerical approximation of the solutions of the Euler equations including a gravitational source term. In a dimensionless form these equations are defined by

$$\partial_{t}\rho + \nabla \cdot (\rho \boldsymbol{u}) = 0,$$

$$\partial_{t}(\rho \boldsymbol{u}) + \nabla \cdot (\rho \boldsymbol{u} \otimes \boldsymbol{u}) + \frac{1}{M^{2}} \nabla p = -\frac{1}{M^{2}} \rho \nabla \Phi,$$

$$\partial_{t}E + \nabla \cdot ((E + p)\boldsymbol{u}) = -\rho \boldsymbol{u} \cdot \nabla \Phi,$$
(1.1)

<sup>∗</sup>Fakult¨at f¨ur Mathematik und Informatik, Universit¨at W¨urzburg, Emil-Fischer-Str. 40, 97074 W¨urzburg, Germany

<sup>†</sup>Correspondence to: Claudius Birke, Universit¨at W¨urzburg, Emil-Fischer-Str. 40, 97074 W¨urzburg, Germany, Email: claudius.birke@mathematik.uni-wuerzburg.de

<sup>‡</sup>Laboratoire de Math´ematiques de Versailles, UVSQ, CNRS, Universit´e Paris-Saclay, 78035 Versailles, France

where  $\rho(\boldsymbol{x},t): \mathbb{R}^d \times \mathbb{R}_{\geq 0} \to \mathbb{R}^+$  denotes the density,  $\boldsymbol{u}(\boldsymbol{x},t): \mathbb{R}^d \times \mathbb{R}_{\geq 0} \to \mathbb{R}^d$  the velocity vector,  $E(\boldsymbol{x},t): \mathbb{R}^d \times \mathbb{R}_{\geq 0} \to \mathbb{R}^+$  the total energy and  $\Phi(\boldsymbol{x}): \mathbb{R}^d \to \mathbb{R}$  a given smooth gravitational potential. In this dimensionless formulation, the parameter M represents the Mach number, which controls the ratio between the velocity of the gas and the speed of sound. In this work, we consider the combined low Mach/low Froude number limit, which is the reason why we set Fr = M. As an effect, only the Mach number M appears in the dimensionless equations in (1.1).

The pressure is given by a pressure law  $p(\tau, e) : \mathbb{R}^+ \times \mathbb{R}^+ \to \mathbb{R}$ , where  $\tau = 1/\rho$  denotes the specific volume and e > 0 the internal energy. The total energy can then be expressed by

$$E = \rho e + \frac{1}{2} M^2 \rho |\mathbf{u}|^2.$$
 (1.2)

The pressure law closing this model obeys the second law of thermodynamics so that a specific entropy  $s(\tau, e) : \mathbb{R}^+ \times \mathbb{R}^+ \to \mathbb{R}^+$ , which satisfies the relation

$$-Tds = de + pd\tau \tag{1.3}$$

for some temperature  $T(\tau, e) > 0$ , exists. In this work, we assume  $(\tau, e) \mapsto s(\tau, e)$  to be strictly convex.

The phase space to which the system (1.1) is associated is denoted by

$$\Omega = \{ \omega = (\rho, \rho \mathbf{u}, E)^T \in \mathbb{R}^{d+2}; \rho > 0, e > 0 \}.$$

$$(1.4)$$

This model can be used in various fields of application, such as the simulation of gas flows in the interior of stars in astrophysics. Depending on the application, the flows can have large scale differences, e.g. the sound speed can be much higher than the speed of the fluid flow. In these low Mach number regimes standard finite volume schemes suffer from excessive diffusion, which can erase the structure of the solution beyond recognition. A number of different strategies have been developed to overcome this problem. One simple but efficient strategy is to modify the diffusion term in the numerical flux by rescaling it with the local Mach number and thereby reduce the viscosity on the velocity. First introduced for the homogeneous Euler equations [26, 1, 25, 12, 17, 27, 28, 35], this approach was also extended to the Euler equations including a gravitational source term [2]. A second approach introduced by Klein [24] relies on a pressure splitting, which decomposes the system of equations into one slow, non-linear part and into a linear part for the fast acoustic dynamics. In [7] this splitting is combined with a Suliciu type relaxation model and an implicit time integration. Thomann et al et al. modify the approach to an implicit-explicit (IMEX) scheme, in which only the acoustic part is solved implicitly, while the non-linear part is solved explicitly by a Godunov-type method based on an approximate Riemann solver. Later this IMEX approach was extended to the Euler equations with gravity [32].

Basis of the herein presented scheme is a third alternative introduced by Chalons *et al.* in [9], where a two-speed relaxation scheme for the barotropic, homogeneous Euler equations is proposed. The use of two different relaxation speeds enables an independent control of the numerical viscosity on the density and on the velocity. By special definitions of the speeds in the low Mach regime, viscosity is transferred from the velocity to the density. Therefore, this approach is related to the previously described rescaling of the viscosity term. The key advantage of this method is that under a subcharacteristic condition it is stable and provably

entropy satisfying. Later the two-speed relaxation system was used to develop an IMEX scheme for the homogeneous Euler equations [\[10\]](#page-33-4).

In contrast, the method presented in this paper is fully explicit. The basic structure of the two-speed relaxation system is adopted and extended by gravitational source terms. From the exact resolution of the Riemann problem associated with this relaxation system, a Godunov-type finite volume method is constructed. The modification of the relaxation speeds is adopted from the original approach. The resulting approximate Riemann solver satisfies a discrete entropy inequality. Based on this inequality, it is shown that no checkerboard modes can arise in the variables fluid velocity and pressure. Checkerboard modes pose an instability characterized by a decoupling of the spatial approximation, which can occur in numerical solutions of incompressible fluid equations computed on collocated grids [\[19\]](#page-34-7). It is well-known that most of the asymptoticpreserving schemes exhibit such nonphysical checkerboard modes in low Mach regimes [\[16,](#page-33-5) [28\]](#page-34-3). When studying the Euler equations with gravity source terms one has to consider their influence on the behaviour of steady states. In several applications such as astrophysics one deals with problems close to the hydrostatic equilibrium

$$\begin{cases} \mathbf{u} = 0, \\ \nabla p = -\rho \nabla \Phi. \end{cases}$$
 (1.5)

Standard finite volume schemes do not automatically satisfy a discrete equivalent of [\(1.5\)](#page-2-0). Therefore these steady states are not preserved exactly by such schemes and small perturbations around this equilibrium cannot be resolved unless the resolution of the scheme is increased, so that the truncation error is sufficiently small. In order to avoid this potentially high computational effort, well-balanced schemes [\[14,](#page-33-6) [13,](#page-33-7) [32,](#page-34-6) [4,](#page-33-8) [3,](#page-32-2) [36,](#page-35-0) [23,](#page-34-8) [31,](#page-34-9) [22\]](#page-34-10) were introduced, which satisfy exactly a discrete equivalent of the steady state.

The well-balancing mechanism in the herein presented relaxation scheme is taken over from [\[18\]](#page-33-9). The key idea is to add a transport relaxation equation for the gravitational potential to the relaxation system, which leads to a Riemann-problem that is under-determined. This gives an additional degree of freedom and allows to introduce a closure equation that is a discrete equivalent of [\(1.5\)](#page-2-0) and ensures the well-balanced property. This approach is exact for certain families of hydrostatic equilibria, i.e. isothermal, incompressible and polytropic ones. In all other cases it maintains the equilibrium to second order. We extend this approach so that it can be applied to any hydrostatic solution for the Euler equations with any equation of state if the hydrostatic solution is known a priori. The extension is based on a second order approximation of the difference in the gravitational potential using the given hydrostatic states for the density and pressure. This is useful for applications in stellar astrophysics, in which the equation of state (EoS) is given in form of a table. Since hydrostatic solutions depend on the EoS, they can then only be found through numerical simulations carried out beforehand and are therefore available in the form of discrete data.

The paper is organized as follows. In Sect. [2,](#page-3-0) the two-speed relaxation model is derived. In addition, the approximate Riemann solver associated with this system and its intermediate states are determined. The following Sect. [3](#page-6-0) contains the first order Godunov-type finite volume scheme, which is based on the previously introduced approximate Riemann solver. Its properties are described and proven in Sect. [4.](#page-7-0) A suitable extension to second order in space is given in Sect. [6.](#page-25-0) In Sect. [7,](#page-26-0) the properties of the second order scheme are checked in numerical tests. Finally, Sect. [8](#page-31-0) provides the conclusion and an outlook.

# 2 The relaxation model

The one dimensional relaxation system described below is based at its core on the Suliciu relaxation model [\[30,](#page-34-11) [8,](#page-33-10) [15\]](#page-33-11). The pressure p is approximated by the relaxation variable π and we add an additional equation describing its behaviour to the system

$$\partial_t \rho \pi + \partial_x (\rho \pi v) + ab \partial_x v = \rho \frac{p - \pi}{\varepsilon}.$$
 (2.1)

While only one relaxation speed is used in the classical Suliciu relaxation model, here two speeds a > 0 and b > 0 appear, as proposed in [\[9\]](#page-33-3). This will be useful to control viscosity for pressure and velocity separately. These speeds will be defined later in Sect. [4.4](#page-16-0) so that they meet stability criteria and keep the viscosity bounded in the low Mach regime.

In addition, also the velocity u is approximated by a relaxation variable v and the following equation is introduced

$$\partial_t (\rho v) + \partial_x (\rho v^2) + \frac{a}{b} \partial_x \frac{\pi}{M^2} = \rho \frac{u - v}{\varepsilon} - \frac{a}{b} \frac{1}{M^2} \rho \partial_x \Phi.$$
 (2.2)

In the next step, we also want to include the gravitational potential in the approximate Riemann solver. According to [\[18\]](#page-33-9) this can be done by approximating the gravitational potential Φ by the relaxation variable Z and adding a transport relaxation equation to the relaxation system

$$\partial_t \rho Z + \partial_x \rho v Z = \rho \frac{\Phi - Z}{\varepsilon}.$$
 (2.3)

Finally, we derive the following relaxation model

$$\partial_{t}\rho + \partial_{x}\left(\rho v\right) = 0,$$

$$\partial_{t}\left(\rho u\right) + \partial_{x}\left(\rho u v + \frac{\pi}{M^{2}}\right) = -\frac{1}{M^{2}}\rho\partial_{x}Z,$$

$$\partial_{t}E + \partial_{x}\left((E + \pi)v\right) = -\rho v\partial_{x}Z,$$

$$\partial_{t}\left(\rho \pi\right) + \partial_{x}\left(\rho \pi v\right) + ab\partial_{x}v = \rho\frac{p - \pi}{\varepsilon},$$

$$\partial_{t}\left(\rho v\right) + \partial_{x}\left(\rho v^{2}\right) + \frac{a}{b}\partial_{x}\frac{\pi}{M^{2}} = \rho\frac{u - v}{\varepsilon} - \frac{a}{b}\frac{1}{M^{2}}\rho\partial_{x}Z,$$

$$\partial_{t}\rho Z + \partial_{x}\rho v Z = \rho\frac{\Phi - Z}{\varepsilon},$$

$$\partial_{t}a + v\partial_{x}a = 0,$$

$$\partial_{t}b + v\partial_{x}b = 0.$$

$$(2.4)$$

The solutions to this relaxation model can be seen as a viscous approximation of the solutions of the original system [\(1.1\)](#page-0-0) as long as the subcharacteristic conditions

$$a \ge b$$
 and  $ab \ge \rho^2 c^2$  (2.5)

are satisfied.

Remark 1 By choosing u = v and a = b one recovers the standard Suliciu relaxation model.

The homogeneous system, denoted by  $(2.4)_{\varepsilon=\infty}$ , has the following properties.

**Lemma 2** The relaxation system  $(2.4)_{\varepsilon=\infty}$  is hyperbolic and all characteristic fields are linearly degenerate. The eigenvalues of the system are given by

$$\sigma^v = v, \quad \sigma^{\pm} = v \pm \frac{a}{M\rho}$$
 (2.6)

where  $\sigma^v$  has multiplicity six. The eigenvalues have the fixed ordering

$$\sigma^{-} < \sigma^{v} < \sigma^{+}. \tag{2.7}$$

The Riemann invariant corresponding to the eigenvalue  $\sigma^v$  is

$$I_1^v = v \tag{2.8}$$

П

and those corresponding to  $\sigma^{\pm}$  are

$$I_{1}^{\pm} = v \pm \frac{a}{M\rho}, \quad I_{2}^{\pm} = u \pm \frac{b}{M\rho}, \quad I_{3}^{\pm} = \frac{1}{\rho} + \frac{\pi}{ab},$$

$$I_{4}^{\pm} = e + \frac{(a-b)b + 2\rho(\pi \pm bM(v-u))}{2\rho^{2}},$$

$$I_{5}^{\pm} = a, \quad I_{6}^{\pm} = b, \quad I_{7}^{\pm} = Z.$$

$$(2.9)$$

**Proof.** The computations are straightforward and left to the reader.

**Remark 3** The relaxation system  $(2.4)_{\varepsilon=\infty}$  provides only one Riemann invariant  $I_1^v$  for the contact wave. As a result, the associated Riemann problem is under-determined.

Let us now consider a single Riemann problem associated with the system  $(2.4)_{\varepsilon=\infty}$ . In order to simplify the notations we introduce the state vector

$$W = (\rho, \rho u, E, \rho \pi, \rho v, \rho Z, a, b)^{T}$$
(2.10)

in the phase space

$$\mathcal{O} = \{ W \in \mathbb{R}^8 : \ \rho > 0, \ e > 0 \}$$
 (2.11)

and additionally for  $\omega \in \Omega$  and given gravitational potential  $\Phi$  the state vector at relaxation equilibrium denoted by

$$W^{eq}(\omega) = (\rho, \rho u, E, \rho p(\tau, e), \rho u, \rho \Phi, a, b)^{T}.$$
(2.12)

Then the initial data of the Riemann problem is given by two constant states  $W^L$  and  $W^R$  separated by one discontinuity located at x=0

$$W_0(x) = \begin{cases} W^L, & x < 0, \\ W^R, & x > 0. \end{cases}$$
 (2.13)

The solution to this problem consists of four constant states, each separated by a contact discontinuity. Therefore the approximate Riemann solver  $W_{\mathcal{R}}(x/t; W^L, W^R)$  has the structure

$$W_{\mathcal{R}}(\frac{x}{t}; W^{L}, W^{R}) = \begin{cases} W^{L}, & \frac{x}{t} < \sigma^{-}, \\ W^{L*}, & \sigma^{-} < \frac{x}{t} < \sigma^{v}, \\ W^{R*}, & \sigma^{v} < \frac{x}{t} < \sigma^{+}, \\ W^{R}, & \sigma^{+} < \frac{x}{t}. \end{cases}$$
(2.14)

![](_page_5_Picture_0.jpeg)

**Fig. 1** Schematic diagram of the Riemann fan for the relaxation system (2.4). The Riemann fan consists of the two intermediate states  $W^{L*}$  and  $W^{R*}$  for given states  $W^L$  and  $W^R$ . The states are separated by the three wave speeds  $v - a/(M\rho)$ , v and  $v + a/(M\rho)$ 

This structure of the solution is also shown in Fig. 1. For the computation of the intermediate states  $W^{L*}$  and  $W^{R*}$  we can use the Riemann invariants given in lemma 2. Since Riemann invariants are constant across their corresponding wave, each Riemann invariant provides one equation. However, counting the Riemann invariants reveals that only 15 Riemann invariants face 16 unknown intermediate states. Therefore, the Riemann problem (2.13) is, as already stated in remark 3, under-determined. In order to overcome this problem, it is suggested in [18] to introduce an additional relation

$$\pi^{R*} - \pi^{L*} = -\bar{\rho} \left( W^L, W^R \right) \left( Z^R - Z^L \right), \tag{2.15}$$

where the function  $\bar{\rho}$  denotes a  $\rho$ -average function. This equation is chosen because it is a discrete representation of the steady states at rest in (1.5) in one spatial dimension and therefore will be useful for the well-balancing of hydrostatic equilibria. The explicit definition of the function  $\bar{\rho}$  depends on the underlying hydrostatic equilibrium and will be given later in Sect. 4.5.

With the newly added closure equation, it is now possible to compute the intermediate states in the Riemann solution.

**Lemma 4** The solution of the Riemann problem (2.13) associated with the relaxation system  $(2.4)_{\varepsilon=\infty}$  has the structure given in (2.14) with the intermediate states

$$v^* = \frac{Mb^L v^L + Mb^R v^R + \pi^L - \pi^R - \bar{\rho} (W^L, W^R) (Z^R - Z^L)}{M(b^L + b^R)},$$
(2.16)

$$\frac{1}{\rho^{L*}} = \frac{1}{\rho^{L}} + \frac{Mb^{R} \left(v^{R} - v^{L}\right) + \pi^{L} - \pi^{R} - \bar{\rho} \left(W^{L}, W^{R}\right) \left(Z^{R} - Z^{L}\right)}{a^{L} \left(b^{L} + b^{R}\right)},\tag{2.17}$$

$$\frac{1}{\rho^{R*}} = \frac{1}{\rho^R} + \frac{Mb^L \left(v^R - v^L\right) + \pi^R - \pi^L + \bar{\rho} \left(W^L, W^R\right) \left(Z^R - Z^L\right)}{a^R \left(b^L + b^R\right)},\tag{2.18}$$

$$u^{L*} = u^{L} + \frac{b^{L} \left( b^{R} M \left( v^{R} - v^{L} \right) + \pi^{L} - \pi^{R} - \bar{\rho} \left( W^{L}, W^{R} \right) \left( Z^{R} - Z^{L} \right) \right)}{Ma^{L} \left( b^{L} + b^{R} \right)}, \tag{2.19}$$

$$u^{R*} = u^{R} + \frac{b^{R} \left(b^{L} M \left(v^{L} - v^{R}\right) + \pi^{L} - \pi^{R} - \bar{\rho} \left(W^{L}, W^{R}\right) \left(Z^{R} - Z^{L}\right)\right)}{M a^{R} \left(b^{L} + b^{R}\right)}, \tag{2.20}$$

$$\pi^{L*} = \frac{b^R \pi^L + b^L \pi^R + M b^L b^R \left( v^L - v^R \right) + b^L \bar{\rho} \left( W^L, W^R \right) \left( Z^R - Z^L \right)}{b^L + b^R}, \tag{2.21}$$

$$\pi^{R*} = \frac{b^R \pi^L + b^L \pi^R + M b^L b^R \left( v^L - v^R \right) - b^R \bar{\rho} \left( W^L, W^R \right) \left( Z^R - Z^L \right)}{b^L + b^R}, \tag{2.22}$$

$$e^{L*} = e^{L} + \frac{(\pi^{L*})^2 - (\pi^{L})^2}{2a^{L}b^{L}} + \frac{(v^* - u^{L*})^2 - (v^{L} - u^{L})^2}{2(\frac{a^{L}}{b^{L}} - 1)},$$
(2.23)

$$e^{R*} = e^R + \frac{(\pi^{R*})^2 - (\pi^R)^2}{2a^R b^R} + \frac{(v^* - u^{R*})^2 - (v^R - u^R)^2}{2(\frac{a^R}{L^R} - 1)},$$
(2.24)

$$a^{L*} = a^L, \ a^{R*} = a^R, \ b^{L*} = b^L, \ b^{R*} = b^R, \ Z^{L*} = Z^L, \ Z^{R*} = Z^R.$$
 (2.25)

**Proof.** The intermediate states can be computed by solving the system of equations given by the Riemann invariants and the closure equation (2.15). The precise steps are straightforward and therefore left to the reader.

**Remark 5** At this point, we do not explicitly define the relaxation speeds  $a^L$ ,  $a^R$ ,  $b^L$  and  $b^R$ , since later, in the proofs of the properties of the relaxation method, various conditions are placed on these speeds. The explicit definitions are then provided in Sect. 4.4.

Equipped with the approximate Riemann solver, we can now define the overall discretization of the scheme in the next section.

### 3 The relaxation scheme

Before we derive a complete finite volume scheme for the Euler equations with a gravitational source (1.1), we introduce some useful notations. The spatial domain is divided into cells  $C_i = (x_{i-1/2}, x_{i+1/2})$  with  $i \in \mathbb{Z}$  that have the size  $\Delta x = x_{i+1/2} - x_{i-1/2}$ . The cell centers are denoted by  $x_i$ . The time discretization is given by  $t^n = n\Delta t$  with  $n \in \mathbb{N}$  and a timestep  $\Delta t$  that is restricted by the CFL condition

$$\frac{\Delta t}{\Delta x} \max_{i} \left\{ \left| v_i - \frac{a_i}{M\rho_i} \right|, \left| v_i + \frac{a_i}{M\rho_i} \right| \right\} \le \frac{1}{2}. \tag{3.1}$$

The cell average  $\omega_i^n$  then approximates the value over the cell  $\mathcal{C}_i$  at time  $t^n$ 

$$\omega_i^n \approx \frac{1}{\Delta x} \int_{\mathcal{C}_i} \omega(x, t^n) dx.$$
 (3.2)

At the start of each time step, we assume to be at the relaxation equilibrium. Therefore the initial data for the relaxation variables at time level n is defined by

$$\pi_i^n = p_i^n, \quad v_i^n = u_i^n, \quad Z_i^n = \Phi_i^n.$$
 (3.3)

Starting from the equilibrium we solve the homogeneous relaxation system  $(2.4)_{\varepsilon=\infty}$  using the Riemann solver  $W_{\mathcal{R}}$  defined in (2.14) and update the cell averages to the next time level  $t^{n+1}$  by a Godunov method of the form

$$\omega_{i}^{n+1} = \omega_{i}^{n} - \frac{\Delta t}{\Delta x} \left( F_{i+1/2}^{n} - F_{i-1/2}^{n} \right) 
+ \frac{\Delta t}{2} \left( S_{i-1/2}^{+,n} \frac{\Phi_{i}^{n} - \Phi_{i-1}^{n}}{\Delta x} + S_{i+1/2}^{-,n} \frac{\Phi_{i+1}^{n} - \Phi_{i}^{n}}{\Delta x} \right), 
F_{i-1/2}^{n} = F(\omega_{i-1}^{n}, \Phi_{i-1}^{n}, \omega_{i}^{n}, \Phi_{i}^{n}), \quad F_{i+1/2}^{n} = F(\omega_{i}^{n}, \Phi_{i}^{n}, \omega_{i+1}^{n}, \Phi_{i+1}^{n}), 
S_{i-1/2}^{+,n} = S^{+}(\omega_{i-1}^{n}, \Phi_{i-1}^{n}, \omega_{i}^{n}, \Phi_{i}^{n}), \quad S_{i+1/2}^{-,n} = S^{-}(\omega_{i}^{n}, \Phi_{i}^{n}, \omega_{i+1}^{n}, \Phi_{i+1}^{n}).$$
(3.4)

The numerical flux is defined by

$$F(\omega^L, \Phi^L, \omega^R, \Phi^R) = \begin{cases} F(\omega^L), & \text{if } \sigma^- > 0, \\ F^{L*}, & \text{if } \sigma^- < 0 \le \sigma^v, \\ F^{R*}, & \text{if } \sigma^v < 0 < \sigma^+, \\ F(\omega^R), & \text{if } \sigma^+ < 0, \end{cases}$$
(3.5)

where according to the left-hand sides of the first three equations of (2.4) the intermediate fluxes can be written as

$$F^{L*} = \left(\rho^{L*}v^*, \rho^{L*}u^{L*}v^* + \frac{\pi^{L*}}{M^2}, (E^{L*} + \pi^{L*})v^*\right),$$

$$F^{R*} = \left(\rho^{R*}v^*, \rho^{R*}u^{R*}v^* + \frac{\pi^{R*}}{M^2}, (E^{R*} + \pi^{R*})v^*\right).$$
(3.6)

The numerical source terms are set as follows

$$S^{+}(\omega^{L}, \Phi^{L}, \omega^{R}, \Phi^{R}) = -(\operatorname{sgn}(v^{*}) + 1) \left(0, \frac{1}{M^{2}} \bar{\rho}(W^{L}, W^{R}), \bar{\rho}(W^{L}, W^{R})v^{*}\right)^{T},$$

$$S^{-}(\omega^{L}, \Phi^{L}, \omega^{R}, \Phi^{R}) = (\operatorname{sgn}(v^{*}) - 1) \left(0, \frac{1}{M^{2}} \bar{\rho}(W^{L}, W^{R}), \bar{\rho}(W^{L}, W^{R})v^{*}\right)^{T}.$$
(3.7)

We note that in this procedure only the variables of the original Euler equations (1.1) in the vector  $\omega$  are updated to the next time level. For the upcoming time step we again assume to be at the equilibrium. As a consequence of this projection approach, the relaxation parameter  $\varepsilon$  does not appear in the relaxation scheme (3.4) and thus does not have to be set explicitly.

# 4 Properties of the relaxation scheme

In this section we focus on the properties of the relaxation scheme just described. We start with the property of entropy stability.

#### 4.1 Entropy inequality

We seek those correct solutions that satisfy the entropy inequality. In practice, it can be observed that searching for entropy solutions makes a finite volume method more stable. This is partly because an entropy inequality can help to ensure the positivity of density and/or internal energy. Going back to the Euler equations (1.1) and assuming smooth solutions, it is possible to derive the additional conservation law

$$\partial_t \rho \mathcal{F}(s) + \partial_x \rho \mathcal{F}(s) u = 0 \tag{4.1}$$

for all smooth functions  $\mathcal{F}$ . Assuming that  $\mathcal{F}$  is increasing and  $\omega \mapsto \rho \mathcal{F}(s)$  is convex, the pair  $(\rho \mathcal{F}(s), \rho \mathcal{F}(s)u)$  defines a Lax entropy pair for the system (1.1). Thus, equation (4.1) states that the entropy is conserved for smooth solutions. However, since the Euler equations are non-linear, discontinuities can arise in the solution in finite time despite of smooth initial conditions. At discontinuities the equation (4.1) is not valid, since it does not consider the entropy dissipation at shocks. Therefore, we replace the equality in (4.1) by an inequality, which leads to the following entropy inequality

$$\partial_t \rho \mathcal{F}(s) + \partial_x \rho \mathcal{F}(s) u \le 0. \tag{4.2}$$

Our scheme should now mimic this behaviour in the sense that its solutions satisfy a discrete version of (4.2).

**Theorem 1** Let us assume that  $w_i^n$  belongs to  $\Omega$  for all  $i \in \mathbb{Z}$ . Furthermore, we assume that at each interface with initial left state  $\omega^L$  and initial right state  $\omega^R$  the intermediate states for density and internal energy in the Riemann solution are positive, i.e.  $\rho^{L*}$ ,  $\rho^{R*}$ ,  $e^{L*}$ ,  $e^{R*} > 0$ , and that the relaxation speeds  $a^{L,R}$  and  $b^{L,R}$  are such that they satisfy the subcharacteristic Whitham conditions

$$a^{L}b^{L} > p(\tau^{L}, e^{L})\partial_{e}p(\tau^{L}, e^{L}) - \partial_{\tau}p(\tau^{L}, e^{L}), \tag{4.3}$$

$$a^{L}b^{L} > p(\tau^{L*}, e^{L*})\partial_{e}p(\tau^{L*}, e^{L*}) - \partial_{\tau}p(\tau^{L*}, e^{L*}),$$
 (4.4)

$$a^R b^R > p(\tau^{R*}, e^{R*}) \partial_e p(\tau^{R*}, e^{R*}) - \partial_\tau p(\tau^{R*}, e^{R*}),$$
 (4.5)

$$a^R b^R > p(\tau^R, e^R) \partial_e p(\tau^R, e^R) - \partial_\tau p(\tau^R, e^R). \tag{4.6}$$

Moreover, we assume that the pressure law satisfies assumption (3).

Then for all  $i \in \mathbb{Z}$ , the updated state  $\omega_i^{n+1}$ , computed with the relaxation scheme (3.4) under the CFL condition (3.1), satisfies the discrete entropy inequality

$$\rho_i^{n+1} \mathcal{F}(s_i^{n+1}) - \rho_i^n \mathcal{F}(s_i^n) - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s) u \}_{i+1/2}^n - \{ \rho \mathcal{F}(s) u \}_{i-1/2}^n \right) \le 0, \tag{4.7}$$

where we define the numerical entropy flux by

$$\{\rho \mathcal{F}u\}_{i-1/2}^{n} = \{\rho \mathcal{F}(s)u\} \left(W^{eq}(\omega_{i-1}^{n}), W^{eq}(\omega_{i}^{n})\right), \tag{4.8}$$

$$\{\rho \mathcal{F}u\}^{L,R} = \{\rho \mathcal{F}(s)u\} \left(W^{eq}(\omega^{L}), W^{eq}(\omega^{R})\right) = \begin{cases} \rho^{L} \mathcal{F}(s(\tau^{L}, e^{L}))u^{L}, & \text{if } \sigma^{-} > 0, \\ \rho^{L*} \mathcal{F}(\hat{s}(W^{L*}))v^{*}, & \text{if } \sigma^{-} < 0 \leq \sigma^{v}, \\ \rho^{R*} \mathcal{F}(\hat{s}(W^{R*}))v^{*}, & \text{if } \sigma^{v} < 0 < \sigma^{+}, \\ \rho^{R} \mathcal{F}(s(\tau^{R}, e^{R}))u^{R}, & \text{if } \sigma^{+} < 0. \end{cases}$$
(4.9)

**Remark 2** At the beginning of this theorem, we assume the intermediate states of density and internal energy to be positive. In Section 4.3 we show that the approximate Riemann solver (2.14) satisfies this property for suitably chosen relaxation speeds.

**Proof.** (**Proof of Theorem 1.**) The proof of this theorem closely follows the steps of a similar proof in [18, p. 113]. Therefore, we only give a sketch of the proof here and do not prove every intermediate step. For more details see [18].

First of all, it is easy to check that

$$I(W) = \pi + ab\tau$$
 and  $J(W) = e - \frac{M^2(v - u)^2}{2(\frac{a}{b} - 1)} - \frac{\pi^2}{2ab}$  (4.10)

are strong Riemann invariants of  $(2.4)_{\varepsilon=\infty}$ . Therefore, weak solutions of  $(2.4)_{\varepsilon=\infty}$  satisfy

$$\partial_t \rho \Psi(I, J) + \partial_x \rho \Psi(I, J) v = 0 \tag{4.11}$$

for all smooth functions  $\Psi : \mathbb{R}^2 \to \mathbb{R}$ . As a consequence, for a function  $W \mapsto \hat{s}(W)$ , which only depends on I and J, weak solutions of  $(2.4)_{\varepsilon=\infty}$  satisfy the additional conservation law

$$\partial_t \rho \mathcal{F}(\hat{s}) + \partial_x \rho \mathcal{F}(\hat{s}) v = 0. \tag{4.12}$$

We define the function  $\hat{s}$  by

$$\hat{s}(W) = s(\hat{\tau}(I(W), J(W)), \hat{e}(I(W), J(W))), \tag{4.13}$$

where  $\hat{\tau}(I,J)$  is the the largest root within  $\mathbb{R}^+$  of the function  $f_{I,J}:\mathbb{R}^+\to\mathbb{R}$  defined by

$$f_{I,J}(\tau) = \tau p \left(\tau, e(\tau, I - ab\tau)\right) + ab\tau^2 - I\tau \tag{4.14}$$

and  $\hat{e}$  is defined by

$$\hat{e}(I,J) = e(\hat{\tau}(I,J), I - ab\hat{\tau}(I,J)). \tag{4.15}$$

For the further steps, the following assumption is made about the pressure law.

**Assumption 3** We assume that the pressure law is such that the function  $\tau \mapsto f_{I,J}$  is strictly convex for all fixed pair (I,J).

This condition is fulfilled by most common pressure laws, including the ideal gas law [18]. Under this assumption, it can be proven (see [18]) that for all W, for which the pair (I(W), J(W)) is in

$$\mathcal{A} = \{ (I, J) \in \mathbb{R}^2, \exists \tau > 0, \exists e > 0, \exists v, \exists u \text{ such that:}$$

$$I = p(\tau, e) + ab\tau,$$

$$(4.16)$$

$$J = e - \frac{p(\tau, e)^2}{2ab},\tag{4.17}$$

$$ab > p(\tau, e)\partial_e p(\tau, e) - \partial_\tau p(\tau, e)\},$$

$$(4.18)$$

the function  $\hat{s}$  is larger than the specific entropy of the original system, i.e.

$$\hat{s}(W) \ge s(\tau, e) \tag{4.19}$$

and that equality is reached in the relaxation equilibrium, i.e.

$$\hat{s}(W_{|\pi=p(\tau,e),v=u}) = s(\tau,e). \tag{4.20}$$

Let us now go back to the additional conservation law (4.12) and integrate it over  $[0, \Delta x/2) \times [0, \Delta t)$ 

$$\int_{0}^{\Delta x/2} (\rho \mathcal{F}(\hat{s}) \left( W_{\mathcal{R}} \left( \frac{x}{\Delta t}; W^{eq}(\omega_{L}), W^{eq}(\omega_{R}) \right) \right) = \int_{0}^{\Delta x/2} (\rho \mathcal{F}(\hat{s})) (W(x, 0)) dx$$

$$-\Delta t (\rho \mathcal{F}(\hat{s}) v) \left( W_{\mathcal{R}} \left( \frac{\Delta x}{2\Delta t}; W^{eq}(\omega_{L}), W^{eq}(\omega_{R}) \right) \right)$$

$$+\Delta t (\rho \mathcal{F}(\hat{s}) v) (W_{\mathcal{R}}(0; W^{eq}(\omega_{L}), W^{eq}(\omega_{R}))). \tag{4.21}$$

Under consideration of the CFL condition (3.1) and equality (4.20), this can be rewritten as

$$\frac{1}{\Delta x} \int_{0}^{\Delta x/2} (\rho \mathcal{F}(\hat{s}) \left( W_{\mathcal{R}} \left( \frac{x}{\Delta t}; W^{eq}(\omega_{L}), W^{eq}(\omega_{R}) \right) \right) dx$$

$$= \frac{\rho^{R} \mathcal{F}(s^{R})}{2} - \frac{\Delta t}{\Delta x} \left( \rho^{R} \mathcal{F}(s^{R}) u^{R} - \{ \rho \mathcal{F} u \}^{L,R} \right). \tag{4.22}$$

The replacement of v by u in the entropy fluxes is due to the fact that the input values of the approximate Riemann solver are at equilibrium and therefore left and right states of u and v are equal in each case. Just in the intermediate states both velocities differ, which is the reason why we write  $v^*$  in (4.9). Due to the inequality (4.19), it follows

$$\hat{s}\left(W_{\mathcal{R}}\left(\frac{x}{\Delta t}; W^{eq}(\omega^L), W^{eq}(\omega^R)\right)\right) \ge s\left(\left(\tau^{eq}, e^{eq}\right)\left(\frac{x}{\Delta t}; \omega^L, \omega^R\right)\right). \tag{4.23}$$

The quantities  $\tau^{eq}$ ,  $e^{eq}$  on the right hand side originate from the approximate Riemann solver  $W_{\mathcal{R}}(x/\Delta t; W^{eq}(\omega^L), W^{eq}(\omega^R))$ . Since we assume  $\mathcal{F}$  to be increasing, it in turn follows that

$$\mathcal{F}(\hat{s})\left(W_{\mathcal{R}}\left(\frac{x}{\Delta t}; W^{eq}(\omega^L), W^{eq}(\omega^R)\right)\right) \geq \mathcal{F}(s)\left(W_{\mathcal{R}}^{(\rho,\rho u,E)}\left(\frac{x}{\Delta t}; W^{eq}(\omega^L), W^{eq}(\omega^R)\right)\right). \quad (4.24)$$

By replacing the content of the integral in (4.22), we obtain the inequality

$$\frac{1}{\Delta x} \int_{0}^{\Delta x/2} (\rho \mathcal{F}(s)) \left( W_{\mathcal{R}}^{(\rho,\rho u,E)} \left( \frac{x}{\Delta t}; W^{eq}(\omega^{L}), W^{eq}(\omega^{R}) \right) \right) dx$$

$$\leq \frac{\rho^{R} \mathcal{F}(s^{R})}{2} - \frac{\Delta t}{\Delta x} \left( \rho^{R} \mathcal{F}(s^{R}) u^{R} - \{ \rho \mathcal{F}(s) u \}^{L,R} \right). \tag{4.25}$$

Inserting  $\omega^L = \omega_{i-1}^n$  and  $\omega^R = \omega_i^n$  leads to

$$\frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_i} (\rho \mathcal{F}(s)) \left( \frac{x - x_{i-1/2}}{\Delta t}; \omega_{i-1}^n, \omega_i^n \right) dx$$

$$\leq \frac{\rho_i^n \mathcal{F}(s_i^n)}{2} - \frac{\Delta t}{\Delta x} \left( \rho_i^n \mathcal{F}(s_i^n) u_i^n - \{ \rho \mathcal{F}(s) u \}_{i-1/2}^n \right). \tag{4.26}$$

For the other half of the cell, on the other hand, integrating over  $(-\Delta x/2, 0] \times [0, \Delta t)$  and applying similar steps as before results in

$$\frac{1}{\Delta x} \int_{-\Delta x/2}^{0} (\rho \mathcal{F}(s)) \left( W_{\mathcal{R}}^{(\rho,\rho u,E)} \left( \frac{x}{\Delta t}; W^{eq}(\omega^{L}), W^{eq}(\omega^{R}) \right) \right) dx$$

$$\leq \frac{\rho^{L} \mathcal{F}(s^{L})}{2} - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s) u \}^{L,R} - \rho^{L} \mathcal{F}(s^{L}) u^{L} \right), \tag{4.27}$$

and inserting  $\omega^L = \omega^n_i$  and  $\omega^R = \omega^n_{i+1}$  leads to

$$\frac{1}{\Delta x} \int_{x_i}^{x_{i+1/2}} (\rho \mathcal{F}(s)) \left( \frac{x - x_{i+1/2}}{\Delta t}; \omega_i^n, \omega_{i+1}^n \right) dx$$

$$\leq \frac{\rho_i^n \mathcal{F}(s_i^n)}{2} - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s) u \}_{i+1/2}^n - \rho_i^n \mathcal{F}(s_i^n) u_i^n \right). \tag{4.28}$$

Summing up the inequalities (4.26) and (4.28) results in the inequality

$$\frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} (\rho \mathcal{F}(s)) (\omega^n(x, t^{n+1})) dx \leq \rho_i^n \mathcal{F}(s_i^n) - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s)u \}_{i+1/2}^n - \{ \rho \mathcal{F}(s)u \}_{i-1/2}^n \right). \tag{4.29}$$

Since we assume  $\rho \mathcal{F}(s)$  to be strictly convex, by applying Jensen's inequality we get

$$\rho \mathcal{F}(s) \left( \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \omega^n(x, t^{n+1}) dx \right) \le \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} (\rho \mathcal{F}(s)) (\omega^n(x, t^{n+1})) dx. \tag{4.30}$$

Finally, we obtain the desired discrete entropy inequality

$$\rho_i^{n+1} \mathcal{F}(s_i^{n+1}) \le \rho_i^n \mathcal{F}(s_i^n) - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s) u \}_{i+1/2}^n - \{ \rho \mathcal{F}(s) u \}_{i-1/2}^n \right). \tag{4.31}$$

П

#### 4.2 Prevention of checkerboard modes

For asmptotic preserving methods, stationary and non-constant solutions may occur in the low-Mach regime, jumping between two different values. This behaviour can arise from the fact that the divergence or gradient of a variable is supposed to be zero in the limit equations, while the discretisation of this term allows a jumping solution. Such solutions are sometimes called checkerboards modes. Of course, it is desirable to prevent the occurrence of this unphysical phenomenon.

**Theorem 4** For the relaxation scheme, velocity and pressure are constant in space for steady periodic solutions.

**Proof.** The proof builds on the entropy inequality of the previous section and follows the strategy of a similar proof in [9]. First of all, using the notations used in the entropy proof, we can write

$$\rho_i^{n+1} \mathcal{F}(s_i^{n+1}) \leq \rho_i^n \mathcal{F}(s_i^n) - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s) u \}_{i+1/2}^n - \{ \rho \mathcal{F}(s) u \}_{i-1/2}^n \right) \\
= \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} (\rho \mathcal{F}(\hat{s}) \left( W_{\mathcal{R}}(t^{n+1}, x) \right) dx. \tag{4.32}$$

Additionally, by applying Jensen's inequality to the left hand side we get the following inequalities

$$\rho_i^{n+1} \mathcal{F}(s_i^{n+1}) \le \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \rho_i^{n+1} \mathcal{F}(s_i^{n+1}) dx \le \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} (\rho \mathcal{F}(\hat{s}) \left( W_{\mathcal{R}}(t^{n+1}, x) \right) dx. \tag{4.33}$$

We now define the left-hand side of the entropy inequality (4.7) by

$$D_i^n := \rho_i^{n+1} \mathcal{F}(s_i^{n+1}) - \rho_i^n \mathcal{F}(s_i^n) - \frac{\Delta t}{\Delta x} \left( \{ \rho \mathcal{F}(s)u \}_{i+1/2}^n - \{ \rho \mathcal{F}(s)u \}_{i-1/2}^n \right). \tag{4.34}$$

For steady and space periodic solutions we then have

$$\sum_{i} D_i^n = 0. (4.35)$$

In combination with the entropy inequality (4.7) we get

$$D_i^n = 0 \quad \forall i. (4.36)$$

From this follows directly that all the inequalities in (4.33) are replaced by equalities and therefore the entropy is equal to the relaxation entropy

$$\rho_i^{n+1} \mathcal{F}(s_i^{n+1}) = (\rho \mathcal{F}(\hat{s}) \left( W_{\mathcal{R}}(t^{n+1}, x) \right). \tag{4.37}$$

In the proof of the entropy inequality it is shown that this is just the case in the relaxation equilibrium, so only if

$$\pi = p(\rho, e), \ u = v, \ \tau = \frac{1}{\rho}, \ \hat{s} = s.$$
 (4.38)

As a consequence, the following relations apply to a single Riemann problem

$$\tau^{L*} = \frac{1}{\rho^{L*}}, \quad \tau^{R*} = \frac{1}{\rho^{R*}}, \quad v^* = u^{L*} = u^{R*},$$

$$\pi^{L*} = p(\rho^{L*}, e^{L*}) = p(\rho^{L*}, s^{L*}), \quad \pi^{R*} = p(\rho^{R*}, e^{R*}) = p(\rho^{R*}, s^{R*}).$$

$$(4.39)$$

Since  $\tau$  is a Riemann invariant for  $\sigma^-$  and  $\sigma^+$ , it holds

$$\tau^{L*} = \tau^L, \ \tau^{R*} = \tau^R. \tag{4.40}$$

We can use this fact to gain more information about the intermediate densities

$$\frac{1}{\rho^{L*}} = \tau^{L*} = \frac{1}{\rho^{L}} \quad \Rightarrow \quad \rho^{L*} = \rho^{L},$$

$$\frac{1}{\rho^{R*}} = \tau^{R*} = \frac{1}{\rho^{R}} \quad \Rightarrow \quad \rho^{R*} = \rho^{R}.$$
(4.41)

From the explicit definition of the intermediate states in (2.17) and (2.18) we can deduce that

$$\frac{1}{\rho^{L*}} - \frac{1}{\rho^L} = \frac{Mb^R \left( v^R - v^L \right) + \pi^L - \pi^R - \bar{\rho} \left( W^L, W^R \right) \left( Z^R - Z^L \right)}{a^L \left( b^L + b^R \right)} = 0, \tag{4.42}$$

$$\frac{1}{\rho^{R*}} - \frac{1}{\rho^R} = \frac{Mb^L \left( v^R - v^L \right) + \pi^R - \pi^L + \bar{\rho} \left( W^L, W^R \right) \left( Z^R - Z^L \right)}{a^R \left( b^L + b^R \right)} = 0. \tag{4.43}$$

With a look at the intermediate states  $u^{L*}$  and  $u^{R*}$ , we see that we can use (4.42) and (4.43) to get

$$u^{L*} = u^{L} + \frac{b^{L}}{M} \frac{Mb^{R} \left(v^{R} - v^{L}\right) + \pi^{L} - \pi^{R} - \bar{\rho} \left(W^{L}, W^{R}\right) \left(Z^{R} - Z^{L}\right)}{a^{L} \left(b^{L} + b^{R}\right)} = u^{L}, \tag{4.44}$$

$$u^{R*} = u^{R} + \frac{b^{R}}{M} \frac{Mb^{L} \left(v^{L} - v^{R}\right) + \pi^{L} - \pi^{R} + \bar{\rho} \left(W^{L}, W^{R}\right) \left(Z^{R} - Z^{L}\right)}{a^{R} \left(b^{L} + b^{R}\right)} = u^{R}. \tag{4.45}$$

Since we are at equilibrium we can conclude that

$$v^* = u^{L*} = u^{R*} = u^L = u^R = v^L = v^R. (4.46)$$

In the next part we will show that the left and the right state at the interface are equal for  $\pi$ . From the Riemann invariants in (2.9) we take

$$I_4^{\pm} = \frac{1}{\rho} + \frac{\pi}{2ab}.\tag{4.47}$$

This quantity is constant across the left and right waves in the Riemann fan which means

$$\frac{1}{\rho^{L*}} + \frac{\pi^{L*}}{2a^Lb^L} = \frac{1}{\rho^L} + \frac{\pi^L}{2a^Lb^L},$$

$$\frac{1}{\rho^{R*}} + \frac{\pi^{R*}}{2a^Rb^R} = \frac{1}{\rho^R} + \frac{\pi^R}{2a^Rb^R}.$$
(4.48)

It has already been established in (4.41) that the density has only two states and therefore we can simplify the equations to

$$\pi^{L*} = \pi^L$$

$$\pi^{R*} = \pi^R.$$
(4.49)

From the explicit definition of the intermediate states and the closure equation (2.15) follows

$$\pi^{L*} = \pi^{L} = \frac{b^{R}\pi^{L} + b^{L}\pi^{R} + Mb^{L}b^{R} \left(v^{L} - v^{R}\right) - b^{L}\bar{\rho}\left(W^{L}, W^{R}\right)\left(Z^{R} - Z^{L}\right)}{b^{L} + b^{R}}$$

$$\stackrel{(4.46)}{=} \frac{b^{R}\pi^{L} + b^{L}\pi^{R} - b^{L}\bar{\rho}\left(W^{L}, W^{R}\right)\left(Z^{R} - Z^{L}\right)}{b^{L} + b^{R}}$$

$$\stackrel{(2.15)}{=} \frac{b^{R}\pi^{L} + b^{L}\pi^{R} + b^{L}\left(\pi^{R} - \pi^{L}\right)}{b^{L} + b^{R}}.$$

$$(4.50)$$

Solving for  $\pi^L$  gives

$$\pi^L = \pi^R. \tag{4.51}$$

Thus we have shown that for both velocities and the pressure, the left and right states at the interface are equal. The solution in these quantities is therefore constant in space.

**Remark 5** For pressure laws that depend only on density, it can also be proven that the density and the internal energy are constant for steady and space periodic solutions.

For steady periodic solutions of the relaxation method the velocity and pressure are constant, which contradicts the non-constant nature of checkerboard modes. The result of the above lemma can thus be interpreted that in velocity and pressure no checkerboard modes can occur.

#### 4.3 Positivity preserving property

For the robustness of a scheme it is essential to keep especially the density but also the internal energy positive. The following lemma will guarantee this property.

**Lemma 6** Given  $\omega^L, \omega^R \in \Omega$ . If the relaxation speeds  $a^L$  and  $a^R$  are large enough to ensure

$$v^{L} - \frac{a^{L}}{M\rho^{L}} < v^{*} < v^{R} + \frac{a^{R}}{M\rho^{R}}, \tag{4.52}$$

$$e^{L} + \frac{(\pi^{L*})^{2} - (\pi^{L})^{2}}{2a^{L}b^{L}} + \frac{(v^{*} - u^{L*})^{2} - (v^{L} - u^{L})^{2}}{2(\frac{a^{L}}{bL} - 1)} > 0, \tag{4.53}$$

$$e^{R} + \frac{(\pi^{R*})^{2} - (\pi^{R})^{2}}{2a^{R}b^{R}} + \frac{(v^{*} - u^{R*})^{2} - (v^{R} - u^{R})^{2}}{2(\frac{a^{R}}{b^{R}} - 1)} > 0, \tag{4.54}$$

then the approximate Riemann solver  $W_{\mathcal{R}}$  preserves the positivity of the density and internal energy.

**Proof.** First, it is trivial that the conditions (4.52), (4.53) and (4.54) are satisfied for a sufficiently large a. To prove the positivity of the density in a next step, we start with the Riemann invariants  $I_1^{\pm}$  from lemma 2, which give us

$$v^{L} - \frac{a^{L}}{M\rho^{L}} = v^{*} - \frac{a^{L}}{M\rho^{L*}} \quad \text{and} \quad v^{R} + \frac{a^{R}}{M\rho^{R}} = v^{*} + \frac{a^{R}}{M\rho^{R*}}.$$
 (4.55)

Using these relations, we can rewrite (4.52) by

$$-\rho^{L*} < 0 < \rho^{R*}. \tag{4.56}$$

So, the intermediate states for the density are positive. The positivity of the internal energy directly follows from (4.53) and (4.54), since the left-hand sides of these conditions represent the left and right intermediate states of the internal energy.

Clearly, this lemma is of limited use in practice. It states that in principle it is possible to preserve the positivity, but it does not help to find a suitable definition of the relaxation speeds that works generally. The following lemma gives stricter conditions for the relaxation speeds, which can also be used for their explicit definition. Under these conditions, it can be proven that the density is kept positive.

**Lemma 7** Consider the relaxation solver with intermediate values and speeds defined by (2.16)-(2.25) with the initial data at equilibrium. Assume that the relaxation speeds  $a^L$ ,  $a^R$ ,  $b^L$ ,  $b^R$  satisfy

$$a^L \ge b^L, \quad a^R \ge b^R, \tag{4.57}$$

$$\frac{b^L}{\rho^L} \ge a_q^L, \quad \frac{b^R}{\rho^R} \ge a_q^R, \tag{4.58}$$

$$\frac{\sqrt{a^L b^L}}{\rho^L} \ge c^L \left( 1 + \beta X^L \right), \quad \frac{\sqrt{a^R b^R}}{\rho^R} \ge c^R \left( 1 + \beta X^R \right). \tag{4.59}$$

for some  $a_q^L$  and  $a_q^R$  depending on  $\omega^L$ ,  $\omega^R$  and  $X^L$ ,  $X^R$  defined by (4.62) and (4.63) with a parameter  $\beta \geq 1$ . The quantities  $c^L$ ,  $c^R$  represent the sound speed. Then the approximate Riemann solver  $W_R$  preserves the positivity of the density.

**Proof.** We start with the definition of the intermediate density (2.17)

$$\frac{1}{\rho^{L*}} = \frac{1}{\rho^{L}} + \frac{Mb^{R} \left(v^{R} - v^{L}\right) + \pi^{L} - \pi^{R} - \bar{\rho} \left(W^{L}, W^{R}\right) \left(Z^{R} - Z^{L}\right)}{a^{L} \left(b^{L} + b^{R}\right)} \\
\geq \frac{1}{\rho^{L}} - \frac{Mb^{R} (v^{L} - v^{R})_{+}}{a^{L} (b^{L} + b^{R})} - \frac{\left(\pi^{R} - \pi^{L} + \bar{\rho} (W^{L}, W^{R}) (Z^{R} - Z^{L})\right)_{+}}{a^{L} (b^{L} + b^{R})} \\
\geq \frac{1}{\rho^{L}} - \frac{M(v^{L} - v^{R})_{+}}{a^{L}} - \frac{\left(\pi^{R} - \pi^{L} + \bar{\rho} (W^{L}, W^{R}) (Z^{R} - Z^{L})\right)_{+}}{a^{L} (\rho^{L} a^{L}_{a} + \rho^{R} a^{R}_{a})}. \tag{4.60}$$

Analogously, we get

$$\frac{1}{\rho^{R*}} \ge \frac{1}{\rho^R} - \frac{M(v^L - v^R)_+}{a^R} - \frac{\left(\pi^L - \pi^R + \bar{\rho}(W^L, W^R)(Z^L - Z^R)\right)_+}{a^R(\rho^L a_q^L + \rho^R a_q^R)}.$$
 (4.61)

Let us now define the variables

$$X^{L} = \frac{1}{c^{L}} \left[ M \left( v^{L} - v^{R} \right)_{+} + \frac{\left( \pi^{R} - \pi^{L} + \bar{\rho}(W^{L}, W^{R})(Z^{R} - Z^{L}) \right)_{+}}{\rho^{L} a_{q}^{L} + \rho^{R} a_{q}^{R}} \right], \tag{4.62}$$

$$X^{R} = \frac{1}{c^{R}} \left[ M \left( v^{L} - v^{R} \right)_{+} + \frac{\left( \pi^{L} - \pi^{R} + \bar{\rho}(W^{L}, W^{R})(Z^{L} - Z^{R}) \right)_{+}}{\rho^{L} a_{q}^{L} + \rho^{R} a_{q}^{R}} \right], \tag{4.63}$$

in order to rewrite the former inequalities in the form

$$\frac{1}{\rho^{L*}} \ge \frac{1}{\rho^L} \left( 1 - \frac{\rho^L c^L}{a^L} X^L \right),$$

$$\frac{1}{\rho^{R*}} \ge \frac{1}{\rho^R} \left( 1 - \frac{\rho^R c^R}{a^R} X^R \right).$$
(4.64)

From combining the conditions (4.57) and (4.59), it follows that

$$\frac{a^L}{\rho^L} \ge c^L (1 + \beta X^L) \Rightarrow \frac{\rho^L c^L}{a^L} \le \frac{1}{1 + \beta X^L}, 
\frac{a^R}{\rho^R} \ge c^R (1 + \beta X^R) \Rightarrow \frac{\rho^R c^R}{a^R} \le \frac{1}{1 + \beta X^R}.$$
(4.65)

With these inequalities we rewrite (4.64)

$$\frac{1}{\rho^{L*}} \ge \frac{1}{\rho^L} \left( 1 - \frac{X^L}{1 + \beta X^L} \right),$$

$$\frac{1}{\rho^{R*}} \ge \frac{1}{\rho^R} \left( 1 - \frac{X^R}{1 + \beta X^R} \right).$$
(4.66)

Because of the definitions in (4.62) and (4.63) we know that  $X^L, X^R \ge 0$  and therefore we can conclude that

$$\rho^{L*} > 0, \quad \rho^{R*} > 0.$$
(4.67)

П

A similar proof for the positivity of the internal energy would be complicated due to the more complex structure of its intermediate states. Therefore, we choose a different way of proof, based on the proof of the entropy inequality in Sect. 4.1.

**Lemma 8** Under the conditions of the entropy theorem 1 and for given  $\omega^L$ ,  $\omega^R \in \Omega$ , the relaxation scheme using the approximate Riemann solver  $W_R$  defined in (2.14) preserves the positivity of the internal energy.

**Proof.** In the proof of the entropy inequality, it is stated in (4.19) that the specific relaxation entropy is larger than the specific entropy of the original system. Therefore, we can conclude

$$\hat{s}_i^{n+1} \ge s(\rho_i^{n+1}, e_i^{n+1}) = s_i^{n+1}. \tag{4.68}$$

In a next step we show that  $\hat{s}_i^{n+1}$  is positive. For this we consider one Riemann problem and investigate the input of the function  $\hat{s}(W) = s(\hat{\tau}(I(W),J(W)),\hat{e}(I(W),J(W)))$ . Here  $\hat{\tau}$  is already positive by definition. For  $\hat{e}(I(W),J(W))$  depending on  $W=W^L$  or  $W=W^R$  its positivity is trivial. For  $W=W^{L*}$  we can rewrite  $\hat{e}$  using J and additionally make use of the fact that I and J are strong Riemann invariants for  $\sigma^L$ . It follows

$$\begin{split} \hat{e}^{L*} &= J^L + \frac{M^2(v^* - u^{L*})^2}{2(\frac{a}{b} - 1)} + \frac{(I^L - a^L b^L \hat{\tau}^{L*})^2}{2a^L b^L} \\ &= e^L - \frac{M^2(v^L - u^L)^2}{2(\frac{a^L}{b^L} - 1)} - \frac{(\pi^L)^2}{2a^L b^L} + \frac{M^2(v^* - u^{L*})^2}{2(\frac{a^L}{b^L} - 1)} + \frac{(\pi^L + a^L b^L \tau^L - a^L b^L \hat{\tau}^{L*})^2}{2a^L b^L} \\ &\geq e^L + \frac{M^2(v^* - u^{L*})^2}{2(\frac{a^L}{b^L} - 1)} + \frac{(a^L b^L \tau^L - a^L b^L \hat{\tau}^{L*})^2}{2a^L b^L}. \end{split}$$

Then for given  $e^L > 0$  it follows that  $\hat{e}^{L*} > 0$ . The same arguments lead to  $\hat{e}^{R*} > 0$ . The positivity of  $\hat{s}$  then follows from the definition of the function  $s(\tau, e) : \mathbb{R}^+ \times \mathbb{R}^+ \to \mathbb{R}^+$ . Furthermore, from relation (1.3) it follows that

$$\partial_s e(\tau, s) = -T(\tau, e) < 0. \tag{4.69}$$

П

For the positive specific relaxation entropy  $\hat{s}_i^{n+1}$  and the definition of  $e(\rho, s) : \mathbb{R}^+ \times \mathbb{R}^+ \to \mathbb{R}^+$  we then obtain a positive updated internal energy

$$e_i^{n+1} = e(\rho_i^{n+1}, s_i^{n+1}) \ge e(\rho_i^{n+1}, \hat{s}_i^{n+1}) > 0.$$
 (4.70)

#### 4.4 Asymptotic preserving property

In the low Mach limit, the solutions of the Euler equations (1.1) tend to the solutions of the incompressible Euler equations. This behaviour can be illustrated by inserting expansions in terms of M given by

$$\rho = \rho_0 + M\rho_1 + M^2\rho_2 + \mathcal{O}(M^3), \quad \mathbf{u} = \mathbf{u}_0 + M\mathbf{u}_1 + M^2\mathbf{u}_2 + \mathcal{O}(M^3),$$

$$e = e_0 + Me_1 + M^2e_2 + \mathcal{O}(M^3), \quad p = p_0 + Mp_1 + M^2p_2 + \mathcal{O}(M^3),$$
(4.71)

into the Euler equations (1.1). Now one can collect terms of order  $\mathcal{O}(M^{-2})$ 

$$\nabla p_0 = -\rho_0 \nabla \Phi, \tag{4.72}$$

of order  $\mathcal{O}(M^{-1})$ 

$$\nabla p_1 = -\rho_1 \nabla \Phi, \tag{4.73}$$

and finally of order  $\mathcal{O}(1)$ 

$$\nabla \cdot (\rho_0 \mathbf{u}_0) = 0,$$

$$\partial_t \mathbf{u}_0 + \mathbf{u}_0 \cdot \nabla \mathbf{u}_0 + \frac{\nabla p_2}{\rho_0} = -\frac{\rho_2 \nabla \Phi}{\rho_0},$$

$$\partial_t e_0 + \mathbf{u}_0 \cdot \nabla e_0 + \frac{1}{\rho_0} \nabla \cdot (p_0 \mathbf{u}_0) = -\mathbf{u}_0 \cdot \nabla \Phi.$$
(4.74)

These equations describe incompressible flows. The conditions (4.72) and (4.73) show that the couples  $\rho_0, p_0$  and  $\rho_1, p_1$  fulfil the hydrostatic equilibrium and are therefore time independent. This property is used in the derivation of the limit equations in (4.74).

**Remark 9** The third equation in (4.74) vanishes because of (4.72). In consequence, the limit equations contain an unknown  $\rho_2$ , to which no conditions seem to be attached that determine its behavior. Under the assumption that the term  $\frac{\nabla p_2}{\rho_0} + \frac{\rho_2 \nabla \Phi}{\rho_0}$  can be written as a gradient, this term can be replaced by  $\nabla \Pi$ , so that the second equation in (4.74) changes to

$$\partial_t \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} + \nabla \Pi = 0. \tag{4.75}$$

With this equation all variables can be determined. For similar arguments, see [11].

In the next step, we want to analyse to what extent the solutions of the compressible Euler equations correspond to those of the incompressible equations. Under the assumptions that in the density no constant fluctuations occur, i.e.

$$\rho = \rho_0 + \mathcal{O}(M^2),\tag{4.76}$$

and that the hydrostatic equilibrium is fulfilled up to errors of order  $\mathcal{O}(M^2)$ 

$$\nabla p + \rho \nabla \Phi = \mathcal{O}(M^2), \tag{4.77}$$

the Euler equations (1.1) become

$$\nabla \cdot (\rho \boldsymbol{u}) = \mathcal{O}(M^2),$$

$$\partial_t \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\nabla p_2}{\rho_0} = -\frac{\rho_2 \nabla \Phi}{\rho_0} + \mathcal{O}(M^2),$$

$$\partial_t e + \boldsymbol{u} \cdot \nabla e + \frac{1}{\rho_0} \nabla \cdot (p\boldsymbol{u}) = -\boldsymbol{u} \cdot \nabla \Phi + \mathcal{O}(M^2).$$
(4.78)

The solutions of (1.1) thus agree with those of the incompressible model up to an error of order  $M^2$ .

Following these theoretical results, the numerical scheme should be consistent with the limit behaviour as M tends to zero, in the sense that the discretization for the compressible Euler equations should tend to the incompressible Euler equations when the Mach number tends to zero. The key to achieve this behaviour for the presented relaxation scheme is the definition of the relaxation speeds a and b. In the former sections several conditions are imposed on these speeds that have to be satisfied so that the scheme is stable and has the properties presented in the former sections. A suitable choice that indeed fulfils the so far stated requirements is the classical one, in which a and b are set to be equal

$$a_q^{\alpha} = c^{\alpha},$$

$$a^{\alpha} = b^{\alpha} = \rho^{\alpha} c^{\alpha} (1 + \beta X^{\alpha}).$$
(4.79)

This definition closely follows the condition (4.59) in Lemma 7. Unfortunately, this definition does not lead to an appropriate discretization, but to excessive diffusion in the low Mach limit. In order to change this behaviour the speeds have to be redefined. In this context it is important to ensure that not only the diffusion is reduced, but also that the sub-characteristic condition remains fulfilled. A suitable choice proposed by the authors of [9] is given by

$$a_q^{\alpha} = \min(1, M)c^{\alpha},$$

$$a^{\alpha} = \frac{\rho^{\alpha}}{\min(1, M)}c^{\alpha}(1 + \beta X^{\alpha}),$$

$$b^{\alpha} = \min(1, M)\rho^{\alpha}c^{\alpha}(1 + \beta X^{\alpha}).$$

$$(4.80)$$

By this definition the speeds are rescaled in the case of small Mach numbers, i.e. for M < 1.

**Remark 10** In the case of Mach numbers  $M \ge 1$ , the relaxation speeds are equal (a = b) and we obtain a classical relaxation system with only one relaxation speed.

**Remark 11** The new scaling of the relaxation speed a has the effect that the maximum wave speed increases by an order of magnitude M. As a consequence, the CFL condition (3.1) becomes stricter and the time step must be chosen smaller accordingly, i.e.

$$\Delta t \sim \frac{M^2 \Delta x}{c}.\tag{4.81}$$

As shown in [9], by replacing M by  $\hat{M} = \max\{M^2, k\Delta x\}$  in the relaxation scheme the CFL condition can be reduced to the parabolic-type condition

$$\Delta t \sim \frac{\max\{M^2, k\Delta x\}\Delta x}{c}.\tag{4.82}$$

**Theorem 12** The two-speed relaxation scheme with the relaxation speeds (4.80) is asymptotic preserving in the sense that:

- a) it is first order uniformly with respect to the Mach number M and
- b) for  $M < \sqrt{k\Delta x}$  and k constant it is consistent at first order with the incompressible limit model (4.74).

**Proof.** In order to prove the first statement of the theorem we evaluate the consistency error by expanding the numerical flux (3.5) in terms of M and then subtract the central flux  $(F(\omega^L) + F(\omega^R))/2$ .

In the low Mach limit  $M \to 0$ , the wave speeds  $\sigma^-$  and  $\sigma^+$  in (2.6) tend towards infinity. Therefore it is sufficient just to consider the intermediate fluxes  $F^{L*}$  and  $F^{R*}$  for the numerical flux. In a first step of the analysis we rewrite the relaxation speeds as expansions in terms of M, so we get

$$X^{\alpha} = \mathcal{O}(M), \quad b^{\alpha} = M\bar{b}^{\alpha} + \mathcal{O}(M^2), \quad a^{\alpha} = \frac{\bar{b}^{\alpha}}{M}(1 + \mathcal{O}(M))$$
 (4.83)

with

$$\bar{b}^{\alpha} = \rho^{\alpha} c^{\alpha}. \tag{4.84}$$

Since

$$\bar{b}^R - \bar{b}^L = \mathcal{O}(M^2), \tag{4.85}$$

we can write  $\bar{b}$  instead of  $\bar{b}^L$  and  $\bar{b}^R$  up to errors of  $\mathcal{O}(M^2)$ . Expanding the intermediate states (2.16)-(2.22) in terms of M yields

$$v^* = \frac{u^L + u^R}{2} + \frac{\pi^L - \pi^R}{2M^2\bar{b}} - \frac{\rho(Z^R - Z^L)}{2M^2\bar{b}}$$
(4.86)

$$+\mathcal{O}(M(u^L - u^R)) + \mathcal{O}(\frac{\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)}{M}),$$
 (4.87)

$$\pi^{L*} = \frac{\pi^L + \pi^R}{2} + M^2 \bar{b} \frac{u^L - u^R}{2} + \frac{\bar{\rho}(Z^R - Z^L)}{2\bar{b}}$$
(4.88)

+ 
$$\mathcal{O}(M^3(u^L - u^R)) + \mathcal{O}(M(\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L))),$$
 (4.89)

$$\pi^{R*} = \frac{\pi^L + \pi^R}{2} + M^2 \bar{b} \frac{u^L - u^R}{2} - \frac{\bar{\rho}(Z^R - Z^L)}{2\bar{b}}$$
(4.90)

+ 
$$\mathcal{O}(M^3(u^L - u^R)) + \mathcal{O}(M(\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L))),$$
 (4.91)

$$\frac{1}{\rho^{L*}} = \frac{1}{\rho^L} + \mathcal{O}(M^2(u^L - u^R)) + \mathcal{O}(\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)), \tag{4.92}$$

$$\frac{1}{\rho^{R*}} = \frac{1}{\rho^R} + \mathcal{O}(M^2(u^L - u^R)) + \mathcal{O}(\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)), \tag{4.93}$$

$$u^{L*} = u^{L} + \mathcal{O}(M^{2}(u^{L} - u^{R})) + \mathcal{O}(\pi^{L} - \pi^{R} + \bar{\rho}(Z^{R} - Z^{L})), \tag{4.94}$$

$$u^{R*} = u^R + \mathcal{O}(M^2(u^L - u^R)) + \mathcal{O}(\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)). \tag{4.95}$$

We can derive these expansion and put the terms  $\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)$  into the error estimates, since, as stated in (4.72) and (4.73), the hydrostatic equilibrium is satisfied up to terms of order  $\mathcal{O}(M^2)$  in the low Mach limit, i.e.

$$p^{L} - p^{R} + \bar{\rho}(Z^{L} - Z^{R}) = \mathcal{O}(M^{2}).$$
 (4.96)

With the help of these expansions, we calculate the flux differences component by component.

i) The difference for the left intermediate flux  $F^{L*}$  in the first component writes

$$\begin{split} & \rho^{L*}v^* - \frac{\rho^L u^L + \rho^R u^R}{2} \\ & = -\frac{\rho^L u^L + \rho^R u^R}{2} + \frac{\rho^L}{2\bar{b}} \left( \frac{p^L - p^R}{M^2} + \frac{\bar{\rho}(Z^L - Z^R)}{M^2} \right) \\ & + \rho^L \frac{u^L + u^R}{2} + \mathcal{O}(M(u^L - u^R)) + \mathcal{O}(\frac{\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)}{M}). \end{split}$$

This difference can be further simplified. In the low Mach limit, the density is constant up to errors of  $\mathcal{O}(M^2)$ . Therefore we can write

$$\rho^R - \rho^L = \mathcal{O}(M^2) \tag{4.97}$$

and replace  $\rho^R$  in the difference by  $\rho^L$ . Additionally, we replace the differences between the left and right states by numerical derivatives, i.e.

$$u^{L} - u^{R} = -\Delta x \partial_{x} u + \mathcal{O}(\Delta x^{2}),$$

$$p^{L} - p^{R} = -\Delta x \partial_{x} p + \mathcal{O}(\Delta x^{2}),$$

$$Z^{L} - Z^{R} = -\Delta x \partial_{x} Z + \mathcal{O}(\Delta x^{2}).$$

$$(4.98)$$

Applying these simplifications results in

$$\rho^{L*}v^* - \frac{\rho^L u^L + \rho^R u^R}{2} = -\frac{\Delta x}{2} \frac{\rho^L}{\bar{b}} \left( \partial_x \frac{p}{M^2} + \bar{\rho} \partial_x \frac{Z}{M^2} \right) + \mathcal{O}(\Delta x^2) + \mathcal{O}(M \Delta x). \tag{4.99}$$

The denominator  $M^2$  does not lead to excessive diffusion at this point, as again the hydrostatic equilibrium is fulfilled up to  $\mathcal{O}(M^2)$ . Analogous calculations for the right intermediate flux  $F^{R*}$  lead to

$$\rho^{R*}v^* - \frac{\rho^L u^L + \rho^R u^R}{2} = -\frac{\Delta x}{2} \frac{\rho^R}{\bar{b}} \left( \partial_x \frac{p}{M^2} + \bar{\rho} \partial_x \frac{Z}{M^2} \right) + \mathcal{O}(\Delta x^2) + \mathcal{O}(M\Delta x). \tag{4.100}$$

ii) The second component for the left flux can be expressed by

$$\begin{split} & \rho^{L*}u^{L*}v^* + \frac{\pi^{L*}}{M^2} - \frac{\rho^L(u^L)^2 + \frac{\pi^L}{M^2} + \rho^R(u^R)^2 + \frac{\pi^R}{M^2}}{2} \\ & = & \bar{b}\frac{u^L - u^R}{2} + \rho^L u^L \frac{u^L + u^R}{2} - \rho^L u^R \frac{u^L - u^R}{2} + \rho^L u^R \frac{u^L - u^R}{2} \\ & - \frac{\rho^L(u^L)^2 + \rho^R(u^R)^2}{2} + \rho^L u^L \frac{p^L - p^R + \bar{\rho}(Z^L - Z^R)}{2\bar{b}M^2} \\ & - \frac{\bar{\rho}(Z^L - Z^R)}{2M^2} + \mathcal{O}(M(u^L - u^R)) + \mathcal{O}(\frac{\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)}{M}) \\ & = & \bar{b}\frac{u^L - u^R}{2} + \rho^L u^R \frac{u^L - u^R}{2} + \rho^L u^L \frac{p^L - p^R + \bar{\rho}(Z^L - Z^R)}{2\bar{b}M^2} \\ & - \frac{\bar{\rho}(Z^L - Z^R)}{2M^2} + \mathcal{O}(M(u^L - u^R)) + \mathcal{O}(\frac{\pi^L - \pi^R + \bar{\rho}(Z^R - Z^L)}{M}) \\ & = & - \frac{\Delta x}{2} \left(\bar{b} + \rho^L u^R\right) \partial_x u - \frac{\Delta x}{2} \frac{\rho^L u^L}{\bar{b}} \left(\partial_x \frac{p}{M^2} + \bar{\rho}\partial_x \frac{Z}{M^2}\right) + \frac{\Delta x}{2} \bar{\rho}\partial_x \frac{Z}{M^2} \\ & + \mathcal{O}(\Delta x^2) + \mathcal{O}(M\Delta x) \end{split}$$

and for the right flux by

$$\begin{split} & \rho^{R*}u^{R*}v^* + \frac{\pi^{R*}}{M^2} - \frac{\rho^L(u^L)^2 + \frac{\pi^L}{M^2} + \rho^R(u^R)^2 + \frac{\pi^R}{M^2}}{2} \\ & = -\frac{\Delta x}{2} \left( \bar{b} + \rho^R u^L \right) \partial_x u - \frac{\Delta x}{2} \frac{\rho^R u^R}{\bar{b}} \left( \partial_x \frac{p}{M^2} + \bar{\rho} \partial_x \frac{Z}{M^2} \right) - \frac{\Delta x}{2} \bar{\rho} \partial_x \frac{Z}{M^2} \\ & + \mathcal{O}(\Delta x^2) + \mathcal{O}(M \Delta x). \end{split}$$

In this flux difference, the new scaling of the relaxation speeds defined in (4.80) unfolds its importance. Clearly, the viscosity on the velocity, represented by the first term, is independent of the Mach number and therefore does not increase in the low Mach limit. With the classical scaling (4.79), on the other hand, this term would have the size  $\mathcal{O}(1/M)$  leading to excessive diffusion for low Mach numbers. While a Mach number dependence in the first term would be problematic, it is not in the second term due to (4.96). The remaining third term containing the derivative of the gravitational potential, which also depends on  $1/M^2$ , cancels out with the gravitational source term (3.7) in the relaxation scheme.

iii) For the difference in the third component, similar steps for the left flux result in

$$\left( \left( \frac{1}{2} M^2 \rho^{L*} (u^{L*})^2 + \rho^{L*} e^{L*} \right) + \pi^{L*} \right) v^* - \frac{(E^L + p^L) u^L + (E^R + p^R) u^R}{2}$$

$$= \rho^L u^R \frac{e^L - e^R}{2} + u^R \frac{p^L - p^R}{2} + \frac{\rho^L e^L + p^L}{2\bar{b}} \frac{p^L - p^R + \bar{\rho} (Z^L - Z^R)}{M^2}$$

$$+ \mathcal{O}(M(u^L - u^R)) + \mathcal{O}(\frac{\pi^L - \pi^R + \bar{\rho} (Z^R - Z^L)}{M})$$

$$= -\frac{\Delta x}{2} \rho^L u^R \partial_x e - \frac{\Delta x}{2} u^R \partial_x p - \frac{\Delta x}{2} \frac{\rho^L e^L + p^L}{\bar{b}} \left( \partial_x \frac{p}{M^2} + \partial \frac{Z}{M^2} \right)$$

$$+ \mathcal{O}(\Delta x^2) + \mathcal{O}(M \Delta x).$$

and for the right flux in

$$\left(\left(\frac{1}{2}M^{2}\rho^{R*}(u^{R*})^{2} + \rho^{R*}e^{R*}\right) + \pi^{R*}\right)v^{*} - \frac{(E^{L} + p^{L})u^{L} + (E^{R} + p^{R})u^{R}}{2}$$

$$= \frac{\Delta x}{2}\rho^{R}u^{L}\partial_{x}e + \frac{\Delta x}{2}u^{L}\partial_{x}p - \frac{\Delta x}{2}\frac{\rho^{R}e^{R} + p^{R}}{\bar{b}}\left(\partial_{x}\frac{p}{M^{2}} + \partial\frac{Z}{M^{2}}\right)$$

$$+ \mathcal{O}(\Delta x^{2}) + \mathcal{O}(M\Delta x).$$

The expansions for all three components are first-order uniformly in M. It is particularly important that the viscosity on the velocity u is independent of M.

The result of the first statement can now be used to prove the second statement of the theorem. We have proven that the solution wM,∆<sup>x</sup> of the relaxation scheme is consistent with the exact solution w<sup>M</sup> of the dimensionless Euler equations [\(1.1\)](#page-0-0) up to order O(∆x) independent of the Mach number, i.e.

$$w_{M,\Delta x} - w_M = \mathcal{O}(\Delta x). \tag{4.101}$$

Additionally, we can deduce from the system [\(4.78\)](#page-17-2) that w<sup>M</sup> is consistent with the solution w of the incompressible Euler equations up to order <sup>O</sup>(M<sup>2</sup> ), i.e.

$$w_M - w = \mathcal{O}(M^2). \tag{4.102}$$

Combining [\(4.101\)](#page-21-1) and [\(4.102\)](#page-21-2) with the condition that <sup>M</sup><sup>2</sup> <sup>=</sup> <sup>O</sup>(∆x) finally results in

$$w_{M,\Delta x} - w = \mathcal{O}(\Delta x) \tag{4.103}$$

and therefore meets the second statement of the theorem.

#### 4.5 Well-balanced property

As explained in the introduction, the well-balanced property is important for solving problems close to hydrostatic equilibrium. In a first step, we will show that the approximate Riemann solver satisfies this property. Building on this, we will then prove in the second step that the entire scheme has this property.

Lemma 13 Assume two given states at equilibrium W<sup>L</sup> and W<sup>R</sup> satisfy

$$u^L = u^R = 0, (4.104)$$

$$p^{R} - p^{L} + \bar{\rho}(W^{L}, W^{R})(\Phi^{R} - \Phi^{L}) = 0.$$
(4.105)

Then the approximate Riemann solver W<sup>R</sup> preserves the steady state, i.e.

$$W_{\mathcal{R}}(x/t, W^L, W^R) = \begin{cases} W^L & \text{if } x/t < 0, \\ W^R & \text{if } x/t > 0. \end{cases}$$
(4.106)

Proof. The result directly follows from the definition of the intermediate states given in [\(2.16\)](#page-5-4)- [\(2.25\)](#page-6-3). Consider the intermediate state v ∗ . Since we start at equilibrium, we can replace the relaxation variables by their corresponding original variables. Using the conditions [\(4.104\)](#page-21-3)- [\(4.105\)](#page-21-4) results in

$$v^* = \frac{1}{b^L + b^R} \left( M b^L u^L + M b^R u^R + p^L - p^R - \bar{\rho} \left( W^L, W^R \right) \left( \Phi^R - \Phi^L \right) \right) = 0.$$

Similar calculations for the other intermediate states complete the proof.

Lemma 13 is rather general, as it assumes that the conditions in (4.104) and (4.105) are satisfied. Clearly, these conditions depend on the definition of the  $\bar{\rho}$ -function. For a simple definition like the arithmetic mean, which is not adjusted to the underlying hydrostatic equilibrium, the scheme maintains the equilibrium to second order [18]. Since we are free to define  $\bar{\rho}$  we can adjust it to the hydrostatic equilibrium and maintain it even up to machine precision. The only limiting requirement for  $\bar{\rho}$  that has to be considered is the consistency property

$$\rho^L = \rho^R = \rho \quad \Rightarrow \quad \bar{\rho}(W^L, W^R) = \rho. \tag{4.107}$$

The following lemma describes the adjusted definitions for isothermal, incompressible and polytropic equilibria. These definitions were already described in [18].

**Lemma 14** i) Let  $W^L$  and  $W^R$  be two states satisfying the isothermal equilibrium

$$\begin{cases} u^{L} = u^{R} = 0, \\ \rho^{L,R} = \exp \frac{C - \Phi^{L,R}}{K}, \\ p^{L,R} = K \exp \frac{C - \Phi^{L,R}}{K}, \end{cases}$$
(4.108)

П

with K > 0 and  $C \in \mathbb{R}$ . If the function  $\bar{\rho}$  is defined by

$$\bar{\rho}(W^L, W^R) = \begin{cases} \frac{\rho^R - \rho^L}{\ln(\rho^R) - \ln(\rho^L)} & \text{if } \rho^L \neq \rho^R, \\ \rho^L & \text{if } \rho^L = \rho^R, \end{cases}$$
(4.109)

then the approximate Riemann solver  $W_{\mathcal{R}}$  preserves the steady state.

ii) Let  $W^L$  and  $W^R$  be two states satisfying the incompressible equilibrium

$$\begin{cases} u^{L} = u^{R} = 0, \\ \rho^{L} = \rho^{R}, \\ p^{L} + \rho^{L} \Phi^{L} = p^{R} + \rho^{R} \Phi^{R}. \end{cases}$$
(4.110)

If the function  $\bar{\rho}$  satisfies the consistency condition (4.107), then the approximate Riemann solver  $W_{\mathcal{R}}$  preserves the steady state.

iii) Let  $W^L$  and  $W^R$  be two states satisfying the polytropic equilibrium

$$\begin{cases} u^{L} = u^{R} = 0, \\ \rho^{L,R} = \left(\frac{\Gamma - 1}{\Gamma K} (C - \Phi^{L,R})\right)^{\frac{\Gamma}{\Gamma - 1}}, \\ p^{L,R} = K^{\frac{1}{1 - \Gamma}} \left(\frac{\Gamma - 1}{\Gamma} (C - \Phi^{L,R})\right)^{\frac{\Gamma}{\Gamma - 1}}, \end{cases}$$

$$(4.111)$$

with  $\Gamma \in (0,1) \cup (1,+\infty)$ , K>0 and  $C \in \mathbb{R}$ . If the function  $\bar{\rho}$  is defined by

$$\bar{\rho}(W^L, W^R) = \begin{cases} \frac{\Gamma - 1}{\Gamma} \frac{(\rho^R)^{\Gamma} - (\rho^L)^{\Gamma}}{(\rho^R)^{\Gamma - 1} - (\rho^L)^{\Gamma - 1}} & \text{if } \rho^L \neq \rho^R, \\ \rho^L & \text{if } \rho^L = \rho^R, \end{cases}$$
(4.112)

then the approximate Riemann solver  $W_{\mathcal{R}}$  preserves the steady state.

**Proof.** In order to prove this lemma it is sufficient to show that with the explicit definition of  $\bar{\rho}$  the conditions (4.104) and (4.105) are satisfied. If so, we can use lemma 13 and the proof is complete. Using the definitions of the isothermal equilibrium states, we can determine the following differences

$$\Phi^R - \Phi^L = K(\ln(\rho^R) - \ln(\rho^L)),$$
  
$$p^R - p^L = K(\rho^R - \rho^L).$$

By inserting these differences together with  $\bar{\rho}$  defined by (4.109) into equation (4.105), it becomes clear that this condition is satisfied. Together with the velocities, which are zero, lemma 13 can be applied and the proof of i) is complete. The proofs for incompressible and polytropic equilibria work in the same way. For more details we refer the reader to [18].

**Remark 15** To ensure the exact preservation of steady states at rest, it is important to consider the following two points in the implementation:

- 1. The comparative operators in the approximate Riemann solver (3.5) must be adjusted to the definition of the sign function in the programming language used. The choice provided here is adapted to sign(0) = 1.
- 2. The implementation of the classical definition of the logarithmic mean can lead to problems if left and right input are very close. Ismail and Roe provide an alternative way of implementation in [21], which avoids this problem.

In practical applications, e.g. in astrophysics, the hydrostatic states are often just available as discrete data generated by previously performed simulations. The following lemma provides an approach to maintain these hydrostatic equilibria as well.

**Lemma 16** Let  $W^L$  and  $W^R$  be two states satisfying some hydrostatic equilibrium

$$\begin{cases} u^{L} = u^{R} = 0, \\ \rho^{L,R} = \rho_{hs}^{L,R}, \\ p^{L,R} = p_{hs}^{L,R}, \end{cases}$$
(4.113)

with  $\rho_{hs}$  and  $p_{hs}$  given hydrostatic states. If the function  $\bar{\rho}$  is defined by

$$\bar{\rho}(W^L, W^R) = \frac{1}{2}(\rho^L + \rho^R)$$
 (4.114)

and the difference of the gravitational potential in the intermediate states is approximated by

$$Z^{R} - Z^{L} \approx -\frac{p_{hs}^{R} - p_{hs}^{L}}{\frac{1}{2}(\rho_{hs}^{L} + \rho_{hs}^{R})},$$
(4.115)

then the approximate Riemann solver  $W_{\mathcal{R}}$  preserves the steady state.

**Proof.** As can be seen in the proof of lemma 14 it is sufficient to show that the conditions (4.104) and (4.105) are fulfilled so that lemma 13 can be applied. In order to do so we plug the

states from (4.113) and the approximation (4.115) into (4.105) and use definition (4.114) for  $\bar{\rho}$ . This results in

$$p_{hs}^R - p_{hs}^L - \frac{1}{2} (\rho_{hs}^L + \rho_{hs}^R) \frac{p_{hs}^R - p_{hs}^L}{\frac{1}{2} (\rho_{hs}^L + \rho_{hs}^R)} = 0.$$
 (4.116)

П

Now that it has been shown that the approximate Riemann solver satisfies the well-balanced property, it remains to show that the entire scheme does so as well.

**Theorem 17** Let us consider an initial data  $\omega_i^0, \omega_{i+1}^0$  that satisfies

$$u_i^0 = u_{i+1}^0 = 0,$$

$$\frac{1}{\Delta x}(p_{i+1}^0 - p_i^0) + \bar{\rho}(W_i^0, W_{i+1}^0) \frac{\Phi_{i+1} - \Phi_i}{\Delta x} = 0.$$
(4.117)

Then the updated state  $\omega^{n+1}$  stays at rest, and thus satisfies  $\omega_i^{n+1} = \omega_i^n$  for all  $i \in \mathbb{Z}$ .

**Proof.** Since both conditions (4.104) and (4.105) of lemma 13 are fulfilled, the approximate Riemann solver stays at rest. The updated state  $\omega_i^1$  at time  $t = \Delta t$  is in essence the sequence of approximate Riemann solvers. Since the approximate Riemann solver is at rest, it directly follows  $\omega_i^1 = \omega_i^0$  for all  $i \in \mathbb{Z}$ .

### 5 Extension to 2D

For two spatial dimensions the Euler equations (1.1) can be written in the form

$$\omega_t + F(\omega)_x + G(\omega)_y = \mathcal{S}(\omega, \Phi). \tag{5.1}$$

On a regular cartesian grid, we extend the numerical scheme described in Sect. 3 to two dimensions by applying an unsplit finite volume method [34], in which the contributions of both directions are used in only one step to update the numerical solution by the formula

$$\omega_{i,j}^{n+1} = \omega_{i,j}^{n} - \frac{\Delta t}{\Delta x} \left( F_{i+1/2,j}^{n} - F_{i-1/2,j}^{n} \right) - \frac{\Delta t}{\Delta y} \left( G_{i,j+1/2}^{n} - G_{i,j-1/2}^{n} \right)$$

$$+ \frac{\Delta t}{2} \left( S_{i-1/2,j}^{+,n} \frac{\Phi_{i,j}^{n} - \Phi_{i-1,j}^{n}}{\Delta x} + S_{i+1/2,j}^{-,n} \frac{\Phi_{i+1,j}^{n} - \Phi_{i,j}^{n}}{\Delta x} \right)$$

$$+ \frac{\Delta t}{2} \left( S_{i,j-1/2}^{+,n} \frac{\Phi_{i,j}^{n} - \Phi_{i,j-1}^{n}}{\Delta x} + S_{i,j+1/2}^{-,n} \frac{\Phi_{i,j+1}^{n} - \Phi_{i,j}^{n}}{\Delta y} \right).$$

$$(5.2)$$

The definitions of the numerical fluxes and source terms are straightforward extension of the ones in Sect. 3. The numerical fluxes continue to use the one-dimensional approximate Riemann solver, so that it is applied separately in xand y-direction. This Riemann solver corresponds to that defined in (2.14), in which additionally the intermediate states for the transversal velocity are set by the left and right values at the interface, respectively, since this velocity is a Riemann invariant for the outer waves  $\sigma^-$  and  $\sigma^+$ .

Since the two-dimensional method is still based on the one-dimensional Riemann solver, the properties proven in Sect. 4 also apply to this method. From this follows the entropy inequality, the absence of checkerboard modes, positivity and the asymptotic conservation property. In addition, the well-balanced property is also preserved, since the approximate Riemann solvers is at rest for initial data in hydrostatic equilibrium in both spatial directions and thus in both momentum equations the pressure gradient cancels out with the source term.

#### 6 Second order extension

In this section we give a possible extension of the proposed scheme to second order in space. We use a linear reconstruction in the primitive variables  $\omega^p = (\rho, \boldsymbol{u}, p)$ . In order to obtain the values  $\omega^R_{i-1/2}$  and  $\omega^L_{i+1/2}$ , which serve as initial data for the Riemann problems at the interface, we evaluate the function

$$\omega^p(x) = \omega_i^p + \sigma(x - x_i) \tag{6.1}$$

in each cell  $C_i$  at its boundaries  $x_{i-1/2}$  and  $x_{i+1/2}$ . The slope  $\sigma$  depends on the neighbouring cells and is computed for each primitive variable separately. In order to ensure that the reconstructed values for the density and internal energy remain positive, which is essential for the positivity property given by the lemmata 6 and 7, we use a limiting procedure introduced in [31] that builds on the work by Berthon in [6]. Then the slopes are defined by

$$\sigma^{\rho} = \rho_{i} \max \left( -1, \min \left( 1, \frac{\bar{\sigma}^{\rho}}{\rho_{i}} \right) \right),$$

$$\sigma^{u} = \kappa \bar{\sigma}^{u},$$

$$\sigma^{p} = p_{i} \max \left( -1, \min \left( 1, \frac{\bar{\sigma}^{p}}{p_{i}} \right) \right),$$
(6.2)

with

$$\bar{\sigma} = \operatorname{minmod}\left(\frac{\omega_i^p - \omega_{i-1}^p}{\Delta x}, \frac{\omega_{i+1}^p - \omega_i^p}{\Delta x}\right)$$
(6.3)

and

$$\kappa = \min(1, \bar{\kappa}),$$

$$\bar{\kappa} = \begin{cases}
\frac{-\sigma^{\rho}(\boldsymbol{u}_{i} \cdot \bar{\boldsymbol{\sigma}}^{\boldsymbol{u}}) + \sqrt{(\sigma^{\rho})^{2}(\boldsymbol{u}_{i} \cdot \bar{\boldsymbol{\sigma}}^{\boldsymbol{u}})^{2} + \|\bar{\boldsymbol{\sigma}}^{\boldsymbol{u}}\|^{2} \frac{\rho_{i} p_{i}}{\gamma - 1}}}{\rho_{i} \|\bar{\boldsymbol{\sigma}}^{\boldsymbol{u}}\|^{2}}, & \text{if } \bar{\boldsymbol{\sigma}}^{\boldsymbol{u}} \neq 0, \\
1, & \text{if } \bar{\boldsymbol{\sigma}}^{\boldsymbol{u}} = 0.
\end{cases}$$
(6.4)

Additionally, we also want to preserve the well-balanced property for the second-order scheme. To achieve this, we adjust the pressure slope by using a hydrostatic reconstruction [23, 31, 32]. Instead of directly using the pressure values of the neighbouring cells, one first applies the transformations

$$q_{i-1} = p_{i-1} - \bar{\rho}(W_{i-1}, W_i)(\Phi_i - \Phi_{i-1}),$$
  

$$q_{i+1} = p_{i+1} + \bar{\rho}(W_i, W_{i+1})(\Phi_{i+1} - \Phi_i),$$
(6.5)

and then computes the slope for the pressure by

$$\bar{\sigma}^p = \text{minmod}\left(\frac{p_i - q_{i-1}}{\Delta x}, \frac{q_{i+1} - p_i}{\Delta x}\right).$$
 (6.6)

In the case of hydrostatic equilibrium, the slope becomes zero and the interface values for the pressure thus reduce to the cell averages. The approximate Riemann solver then stays at rest due to lemma 13 and all results of the former section about well-balancing remain valid for the second order scheme.

The second order scheme remains asymptotic preserving since the differences in (4.98) are due to the linear reconstruction of order  $\mathcal{O}(\Delta x^2)$ . In the following, we illustrate this for the velocity

u using backward slopes for σ u :

$$u^{L} - u^{R} = u_{i} + \sigma_{i} \frac{\Delta x}{2} - \left(u_{i+1} - \sigma_{i+1} \frac{\Delta x}{2}\right)$$

$$= u_{i} + \frac{u_{i} - u_{i-1}}{\Delta x} \frac{\Delta x}{2} - u_{i+1} + \frac{u_{i+1} - u_{i}}{\Delta x} \frac{\Delta x}{2}$$

$$= -\frac{1}{2} u_{i+1} + u_{i} - \frac{1}{2} u_{i-1}$$

$$= -\frac{1}{2} (\Delta x)^{2} \partial_{x} u.$$

Thanks to these second order approximations of the derivatives, all first order terms in the consistency error can be replaced by second order terms so that the scheme becomes second order uniformly in M. The steps of proof of the second part of theorem [12](#page-18-0) work analogously as for the first order case, if we assume for the last step the new condition M < <sup>√</sup> k∆x 2 .

# 7 Numerical results

In this section we numerically investigate the theoretical properties of the relaxation scheme presented in the previous sections. The approximate Riemann solver in the scheme is equipped with the intermediate states defined in [\(2.16\)](#page-5-4)-[\(2.25\)](#page-6-3) and the relaxation speeds [\(4.80\)](#page-17-0) with β = 1.1. Various definitions are used for the ¯ρ-function. Definition [\(4.109\)](#page-22-1) is used by default. If a different choice is made, this is indicated in the respective test. The second order spatial scheme is combined with a third order Runge Kutta method [\[29\]](#page-34-14) for time integration. For all test set-ups we assume an ideal gas law p = (γ − 1)ρe. The computations are performed on a regular cartesian grid.

#### 7.1 Accuracy

In a first numerical test, which is suggested by [\[36\]](#page-35-0), we investigate the experimental order of convergence of the relaxation scheme presented. For the Euler equations [\(1.1\)](#page-0-0) on the domain [0, 1]<sup>2</sup> with a linear gravitational potential Φ(x, y) = x+y, one possible exact solution is defined by

$$\rho(x, y, t) = 1 + 0.2 \sin \left(\pi \left(x + y - t(u_{1_0} + u_{2_0})\right)\right),$$

$$\mathbf{u}(x, y, t) = (u_{1_0}, u_{2_0}),$$

$$p(x, y, t) = 4.5 + (u_{1_0} + u_{2_0})t - (x + y) + 0.2 \cos \left(\left(\pi \left(x + y - (u_{1_0} + u_{2_0})t\right)\right)/\pi,$$
(7.1)

with u1<sup>0</sup> = u2<sup>0</sup> = 20 and p<sup>0</sup> = 4.5. The exact solution is also used for the boundary conditions. The adiabatic coefficient is set to γ = 5/3. We compare the numerical and exact solutions computed on a N × N grid at final time T = 0.01. The resulting L 1 errors and experimental orders of convergence can be found in Table [1.](#page-27-0) As expected, we obtain orders of convergence of nearly two. Without the use of limiters full second order is reached.

### 7.2 Strong rarefaction test

In this section we want to numerically verify the theoretical results of Sect. [4.3,](#page-13-0) i.e. the positivity of density and internal energy. One suitable test for which density and pressure become very

| N   | $L^1(\rho)$ | $EOC(\rho)$ | $L^1(\rho u_1)$ | $EOC(\rho u_1)$ | $L^1(\rho u_2)$ | $EOC(\rho u_2)$ | $L^1(E)$ | EOC(E) |
|-----|-------------|-------------|-----------------|-----------------|-----------------|-----------------|----------|--------|
| 32  | 7.26E-04    | -           | 1.45E-02        | -               | 1.45E-02        | -               | 2.90E-01 | -      |
| 64  | 1.97E-04    | 1.88        | 3.93E-03        | 1.88            | 3.93E-03        | 1.88            | 7.87E-02 | 1.88   |
| 128 | 5.22E-05    | 1.92        | 1.04E-03        | 1.92            | 1.04E-03        | 1.92            | 2.08E-02 | 1.92   |
| 256 | 1.37E-05    | 1.92        | 2.73E-04        | 1.93            | 2.73E-04        | 1.93            | 5.47E-03 | 1.93   |
| 512 | 3.60E-06    | 1.94        | 7.10E-05        | 1.94            | 7.10E-05        | 1.94            | 1.42E-03 | 1.95   |

**Table 1**  $L^1$  errors and experimental orders of convergence

![](_page_27_Figure_2.jpeg)

Fig. 2 Numerical solution for density, velocity and total energy at final time T=0.1

small is the 1-2-0-3 strong rarefaction test [32]. In this test set-up, two rarefaction waves are launched in x-direction on top of an isothermal atmosphere. Therefore, on the domain  $[0,1]^2$  the density  $\rho$  and pressure p are initially defined by (4.108) with the constants C=-0.01 and  $K=\gamma-1$ , an adiabatic coefficient  $\gamma=1.4$  and a quadratic gravitational potential  $\Phi(x,y)=\frac{1}{2}[(x-0.5)^2+(y-0.5)^2]$ . The initial velocities are set to

$$u_1 = \begin{cases} -2, & x < 0.5, \\ 2, & x \ge 0.5, \end{cases} \quad \text{and} \quad u_2 = 0.$$
 (7.2)

One slice along the x-axis of the numerical solution at final time T = 0.1 computed on a  $128 \times 128$  grid by our relaxation scheme is presented in Fig. 2. Although the values for density and total pressure become very small during the simulation, they always remain positive. This outcome underlines the theoretical results stated in lemmata 6 and 7.

#### 7.3 Isothermal atmosphere

The following set-up is taken from [13]. The aim of this experiment is to illustrate the exact preservation of an isothermal equilibrium. We consider the gravitational potential

$$\Phi(x,y) = x + y. \tag{7.3}$$

The initial conditions on the domain [0, 1]<sup>2</sup> are given by

$$\rho(x, y, 0) = \rho_0 \exp(-\rho_0 g(x+y)/p_0), 
\mathbf{u}(x, y, 0) = 0, 
p(x, y, 0) = p_0 \exp(-\rho_0 g(x+y)/p_0),$$
(7.4)

with the parameters ρ<sup>0</sup> = 1.21, p<sup>0</sup> = 1 and g = 1. In this test we set γ = 1.4. The solution should be preserved up to any final time. Here we choose T = 1.0. Since the solution is in hydrostatic equilibrium, the choice of the ¯ρ-average plays an important role. As this is an isothermal equilibrium, we use for ¯ρ the definition [\(4.109\)](#page-22-1). The L 1 error between the approximated solution and the exact solution is given in Table [2](#page-28-0) and is in the order of magnitude of the machine accuracy.

| N   | 1<br>L<br>(ρ) | 1<br>L<br>(ρu1) | 1<br>L<br>(ρu2) | 1<br>L<br>(E) |
|-----|---------------|-----------------|-----------------|---------------|
| 32  | 8.95E-17      | 5.21E-16        | 5.21E-16        | 4.18E-16      |
| 64  | 1.73E-16      | 1.62E-16        | 1.62E-16        | 7.24E-16      |
| 128 | 3.40E-16      | 3.47E-16        | 3.47E-16        | 1.63E-15      |
| 256 | 6.30E-16      | 6.89E-16        | 6.89E-16        | 3.46E-15      |
| 512 | 1.22E-15      | 1.54E-15        | 1.54E-15        | 7.43E-15      |

Table 2 L 1 errors for an isothermal atmosphere

#### 7.4 General steady state

In practice, steady states that do not belong to the class of polytropic equilibria can also occur. In order to investigate the behaviour of the well-balancing mechanism for these cases, we now apply the scheme to a general steady state. We take the initial conditions from the set-up in Sect. [7.1](#page-26-1) and set the initial velocities u1<sup>0</sup> and u2<sup>0</sup> to zero. Then it is easy to check that the initial data is in hydrostatic equilibrium.

In a first step, we use the ¯ρ-average tuned to isothermal equilibria [\(4.109\)](#page-22-1) and compute the solution at final time T = 1. As expected, the L 1 error shown in Table [3](#page-28-1) is now no longer in the order of magnitude of the machine accuracy, but the hydrostatic equilibrium is still preserved up to second order. This result remains true even if we use a constant reconstruction and consequently a first order scheme. As the convergence rates in Table [4](#page-29-0) show, the hydrostatic equilibrium is maintained up to second order despite the constant reconstruction. Mathematically, this can be explained by the fact that equation [\(4.117\)](#page-24-0) is satisfied up to second order.

| N   | L1<br>(ρ) | EOC(ρ) | L1<br>(ρu1) | EOC(ρu1) | L1<br>(ρu2) | EOC(ρu2) | L1<br>(E) | EOC(E) |
|-----|-----------|--------|-------------|----------|-------------|----------|-----------|--------|
| 32  | 9.43E-06  | -      | 1.36E-05    | -        | 1.36E-05    | -        | 5.08E-05  | -      |
| 64  | 2.35E-06  | 2.01   | 3.43E-06    | 1.99     | 3.43E-06    | 1.99     | 1.26E-05  | 2.01   |
| 128 | 5.88E-07  | 2.00   | 8.60E-07    | 2.00     | 8.60E-07    | 2.00     | 3.14E-06  | 2.01   |
| 256 | 1.47E-07  | 2.00   | 2.16E-07    | 1.99     | 2.16E-07    | 1.99     | 7.85E-07  | 2.00   |
| 512 | 3.69E-08  | 2.00   | 5.42E-08    | 2.00     | 5.42E-08    | 2.00     | 1.97E-07  | 2.00   |

Table 3 L 1 errors and experimental orders of convergence of the second order scheme for a general steady state using the ¯ρ-average [\(4.109\)](#page-22-1)

Let us now assume that we know the hydrostatic equilibrium a priori and it is given as discrete data for the density and pressure. In this case, the approach described in lemma [16](#page-23-3) should be able to maintain this particular hydrostatic equilibrium up to machine precision. In order

| N   | 1<br>L<br>(ρ) | EOC(ρ) | 1<br>L<br>(ρu1) | EOC(ρu1) | 1<br>L<br>(ρu2) | EOC(ρu2) | 1<br>L<br>(E) | EOC(E) |
|-----|---------------|--------|-----------------|----------|-----------------|----------|---------------|--------|
| 32  | 9.74E-06      | -      | 1.40E-05        | -        | 1.40E-05        | -        | 5.15E-05      | -      |
| 64  | 2.39E-06      | 2.03   | 3.48E-06        | 2.01     | 3.48E-06        | 2.01     | 1.27E-05      | 2.02   |
| 128 | 5.93E-07      | 2.01   | 8.67E-07        | 2.01     | 8.67E-07        | 2.01     | 3.15E-06      | 2.01   |
| 256 | 1.48E-07      | 2.00   | 2.17E-07        | 2.00     | 2.17E-07        | 2.00     | 7.86E-07      | 2.00   |
| 512 | 3.70E-08      | 2.00   | 5.43E-08        | 2.00     | 5.43E-08        | 2.00     | 1.97E-07      | 2.00   |

Table 4 L 1 errors and experimental orders of convergence of the first order scheme for a general steady state using the ¯ρ-average [\(4.109\)](#page-22-1)

to check this, we set the values ρhs and phs equal to the initial values for density respective pressure. The L 1 errors in Table [5](#page-29-1) show that the hydrostatic equilibrium is maintained up to machine precision.

| N   | 1<br>L<br>(ρ) | 1<br>L<br>(ρu1) | 1<br>L<br>(ρu2) | 1<br>L<br>(E) |
|-----|---------------|-----------------|-----------------|---------------|
| 32  | 6.54E-17      | 9.10E-16        | 9.10E-16        | 1.33E-15      |
| 64  | 1.85E-16      | 1.95E-15        | 1.95E-15        | 4.78E-15      |
| 128 | 2.98E-16      | 4.78E-15        | 4.78E-15        | 8.97E-15      |
| 256 | 6.25E-16      | 8.32E-16        | 8.32E-16        | 2.04E-14      |
| 512 | 1.25E-15      | 1.83E-14        | 1.83E-14        | 4.24E-14      |

Table 5 L 1 errors for a general steady state using the approach for a-priori known hydrostatic equilibria from lemma [16](#page-23-3)

## 7.5 Perturbation of an isothermal atmosphere

One main advantage of well-balanced schemes is their ability to resolve small perturbations on the hydrostatic equilibrium even on coarse grids. It is precisely this effect that we are investigating with the following test. For this purpose, we take the initial values from Sect. [7.3,](#page-27-2) which are in hydrostatic equilibrium, and add a perturbation on the pressure

$$p(x,y,0) = p_0 \exp\left(-\rho_0 g(x+y)/p_0\right) + \eta \exp\left(-100\rho_0 g((x-0.3)^2 + (y-0.3)^2)/p_0\right). \tag{7.5}$$

The strength of the perturbation is controlled by the parameter η. The numerical solutions are computed on a 64×64 mesh up to a final time t = 0.15. In order to investigate the well-balancing effect, we compare the results of our well-balanced scheme with a non-well-balanced scheme. The non-well-balanced scheme uses a Rusanov flux in combination with a linear reconstruction limited by the minmod limiter.

The numerical solutions of the two schemes for a large perturbation η = 0.1 are illustrated in the two upper plots of Fig. [3.](#page-30-0) Looking at the two solutions, it can be said that they are visually very similar. Both methods are capable of resolving the perturbation well. For a significantly smaller perturbation (η = 1E−10), the situation is completely different. While our well-balanced relaxation scheme is still capable to resolve the perturbation (in fact one cannot see any difference in the resolution in comparison to the larger perturbation), the non-wellbalanced scheme completely destroys the structure of the initial pressure pulse. This underlines the functionality of the well-balanced mechanism in the relaxation scheme and also demonstrates the importance of this property for problems close to hydrostatic equilibrium.

![](_page_30_Figure_0.jpeg)

**Fig. 3** Pressure perturbation of an isothermal atmosphere at t = 0.15

#### 7.6 Stationary vortex in a gravitational field

In this section, we investigate the effect of the new scaling of the relaxation speeds a and b for problems with low Mach numbers. Therefore we compare the two-speed relaxation scheme using the speeds defined in (4.80) with the one-speed relaxation scheme using the speeds (4.79). As a test, we use a version of the Gresho vortex modified for the Euler equations with a gravitational source term that was already given in [32]. The density in this set-up is defined by

$$\rho = \exp\left(-\frac{\Phi}{RT}\right). \tag{7.6}$$

The rest of the initial data is given in radial coordinates  $(r, \theta)$ . The velocity field has the form

$$u_{\theta}(r) = \frac{1}{u_r} \begin{cases} 5r, & r \le 0.2\\ 2 - 5r, & 0.2 < r \le 0.4\\ 0, & r > 0.4 \end{cases}$$
 (7.7)

and the gravitational potential is defined by

$$\Phi(r) = \begin{cases}
12r^2, & r \le 0.2 \\
0.5 - \ln(0.2) + \ln(r), & 0.2 < r \le 0.4 \\
\ln(2) - 0.5 \frac{r_c}{r_c - 0.4} + 2.5 \frac{r_c}{r_c - 0.4} r - 1.25 \frac{1}{r_c - 0.4} r^2, & 0.4 < r \le r_c \\
\ln(2) - 0.5 \frac{r_c}{r_c - 0.4} + 1.25 \frac{r_c^2}{r_c - 0.4}, & r > r_c.
\end{cases} \tag{7.8}$$

The pressure p is departed into a hydrostatic pressure  $p_0$  and a pressure  $p_2$  associated with the centrifugal forces and given by  $p = p_0 + M^2 p_2$ , where  $p_0 = RT\rho$  and

$$p_2(r) = \frac{RT}{u_r^2} \begin{cases} p_{21}(r), & r \le 0.2\\ p_{21}(0.2) + p_{22}(r), & 0.2 < r \le 0.4\\ p_{21}(0.2) + p_{22}(0.4), & 0.4 < r \le r_c \end{cases}$$
(7.9)

with

$$p_{21}(r) = \left(1 - \exp\left(-12.5 \frac{r^2}{RT}\right)\right),$$

$$p_{22}(r) = \frac{1}{(1 - M^2)(1 - 0.5M^2)} \exp\left(\frac{-0.5 + \ln(0.2)}{RT}\right)$$

$$\left(r^{-\frac{1}{RT}} \left(M^4 (r(10 - 12.5r) - 2) - 4 + M^4 (r(12.5r - 20) + 6)RT\right) + \exp\left(\frac{-\ln(0.2)}{RT}\right) \left(4 - 2.5M^4RT + 0.5M^4\right)\right).$$

The reference values are given by  $u_r = 2 \cdot 0.2 \cdot \pi$  and  $RT = 1/M^2$ . We choose  $\gamma = 5/3$  for the adiabatic coefficient. The spatial domain is  $D = [0,1]^2$  and has periodic boundary conditions. The computations are carried out on a  $40 \times 40$  grid until a final time T = 1, which corresponds to one turn of the vortex. We solve this initial value problem for various maximum Mach numbers M using the two different schemes. The solutions generated by the one-speed relaxation scheme are depicted in the top row of Fig. 4, while the solutions computed by the two-speed relaxation scheme are shown in the bottom row. It becomes clear that for decreasing Mach numbers, the vortex in the upper row smears out very quickly and even loses its shape completely. The vortex produced by the two-speed scheme in the lower row, on the other hand, retains its shape regardless of the Mach number, so that no difference is visually discernible.

This outcome can be explained by the theoretical results from Sect. 4.4. While the diffusion for the one-speed scheme increases for decreasing Mach numbers, the use of two relaxation speeds, as shown in the proof of theorem 12, results in a Mach number independent diffusion. Further evidence for this behaviour can be found in the analysis of the kinetic energy. Table 6 contains the percentages of kinetic energy compared to the initial value after one full turnover. The final amount of kinetic energy in the solutions of the one-speed scheme strongly decreases for decreasing Mach numbers. In contrast, the asymptotic preserving two-speed scheme is able to keep the loss almost constant at 14%, regardless of the Mach number.

| scheme  | M = 0.1 | M = 0.01 | M = 0.001 |
|---------|---------|----------|-----------|
| 1-speed | 62.74   | 47.49    | 50.08     |
| 2-speed | 86.03   | 86.00    | 85.99     |

**Table 6** Percentage of kinetic energy compared to the initial value after one full turnover (T = 1) for the stationary vortex in a gravitational field

### 8 Conclusion

The proposed scheme extends the two-speed relaxation approach to the full Euler equations with a gravitational source term. In order to preserve steady states at rest, a well-balancing mechanism is installed in the approximate Riemann solver. The resulting scheme is provably asymptotic preserving and maintains all hydrostatic equilibria up to second order, certain families and a-priori known equilibria even up to machine precision. The approximate Riemann solver is positivity preserving, entropy satisfying and prevents the occurrence of checkerboard modes in the velocity and pressure variables. The properties of the method proven in theory are substantiated in numerical tests. Further steps may be the development of an IMEX scheme

![](_page_32_Figure_0.jpeg)

Fig. 4 Numerical solutions for different maximum Mach numbers M after one full turnover. The local Mach number relative to the respective M is color coded

based on the herein presented full time explicit scheme in order to overcome the severe time step restriction for problems with very low Mach numbers, and the extension to other PDE systems like the MHD equations.

### Acknowledgements

The authors thank Jonas Berberich for his comments on the well-balancing of a-priori known hydrostatic equilibria. We acknowledge the use of the Seven-League Hydro Code (https://slh-code.org) for our numerical experiments. Claudius Birke acknowledges the support by the German Research Foundation (DFG) under the project no. KL 566/22-1. All three authors acknowledge the project "Bayerisch-Französische Hochschulzentrum FK40\_2019" which supported this work.

#### References

- [1] Barsukow, W., Edelmann, P.V.F., Klingenberg, C., Miczek, F., Röpke, F. K.: A Numerical Scheme for the Compressible Low-Mach Number Regime of Ideal Fluid Dynamics. J. of Sci. Comput. **72**, 623–646 (2017)
- [2] Barsukow, W., Edelmann, P.V.F., Klingenberg, C., Röpke, F.K.: A low Mach Roe-type solver for the Euler equations allowing for gravity source terms. ESAIM: Proc. Surv. 58, 1–10 (2017)
- [3] Berberich, J.P., Chandrashekar, P., Klingenberg, C.: High order well-balanced finite volume methods for multi-dimensional systems of hyperbolic balance laws. Comput. Fluids **219** (2021)

- [4] Berberich, J.P., K¨appeli, R., Chandrashekar, P., Klingenberg, C.: High order discretely well-balanced method for arbitrary hydrostatic atmospheres. Commun. Comput. Phys. 30(3), 666–708 (2021)
- [5] Berberich, J.P., Klingenberg, C.: Entropy Stable Numerical Fluxes for Compressible Euler Equations which are Suitable for All Mach Numbers. Proc. Numhyp 2019, SEMA SIMAI Springer Series (2020)
- [6] Berthon, C.: Stability of the MUSCL schemes for the Euler equations. Commun Math Sci. 3(2), 133–157 (2005)
- [7] Berthon, C., Klingenberg, C., Zenk, M.: An all Mach number relaxation upwind scheme. SMAI J. of Comput. Math. 6, 1–31 (2020)
- [8] Bouchut, F.: Nonlinear stability of finite volume methods for hyperbolic conservation laws and well-balanced schemes for sources. Frontiers in Mathematics, Birkh¨auser Verlag, Basel (2004)
- [9] Bouchut, F., Chalons, C., Guisset, S.: An entropy satisfying two-speed relaxation system for the barotropic Euler equations. Application to the numerical approximation of low Mach number flows. Numer. Math. 145, 35–76 (2020)
- [10] Bouchut, F., Franck, E., Navoret, L.: A low cost semi-implicit low-Mach relaxation scheme for the full Euler equations. J. of Sci. Comput. 83, 24 (2020)
- [11] Bruell, G., Feireisl, E.: On a singular limit for stratified compressible fluids. Nonlinear Analys.: Real World Appl. 44, 334–346 (2018)
- [12] Chalons, C., Girardin, M., Kokh, S.: An all-regime Lagrange-Projection like scheme for the gas dynamics equations on unstructured meshes. Commun. Comput. Phys. 20, 188–233 (2016)
- [13] Chandrashekar, P., Klingenberg, C.: A second order well-balanced finite volume scheme for Euler equations with gravity. SIAM J. of Sci. Comput. 37(3), B382–B402 (2015)
- [14] Chertock, A., Cui, S., Kurganov, A., Ozcan, S. N., Tadmor, E.: Well-Balanced Schemes for the Euler Equations with Gravitation: Conservative Formulation Using Global Fluxes. J. Comput. Phys. 358, 36–52 (2018)
- [15] Coquel, F., Perthame, B.: Relaxation of energy and approximate Riemann solvers for general pressure laws in fluid dynamics. SIAM J. Num. Anal. 35(6), 2223–2249 (1998)
- [16] Dechallerie, S.: Checkerboard modes and wave equation. Proc. of Algoritm., vol. 2009: 71–80 (2009)
- [17] Dellacherie, S.: Analysis of Godunov type schemes applied to the compressible Euler system at low mach number. J. Comput. Phys. 229(4), 978–1016 (2010)
- [18] Desveaux, V., Zenk, M., Berthon, C., Klingenberg, C.: Well-balanced schemes to capture non-explicit steady states in the Euler equations with gravity. Int. J. Num. Methods Fluids 81(2), 104–127 (2016)

- [19] Ferziger, J., Peric, M.: Computational Methods for Fluid Dynamics. 3rd ed., Springer, Berlin (2002)
- [20] Harten, A., Lax, P.D., Van Leer, B.: On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. SIAM rev. 25, 35–61 (1983)
- [21] Ismail, F., Roe, P.L.: Affordable, entropy-consistent Euler flux functions II: Entropy production at shocks. J. Comput. Phys. 228, 5410–5436 (2009)
- [22] K¨appeli, R., Mishra, S.: Well-balanced schemes for the Euler equations with gravitation. J. Comput. Phys. 259, 199–219 (2014)
- [23] K¨appeli, R., Mishra, S.: A well-balanced finite volume scheme for the Euler equations with gravitation - the exact preservation of hydrostatic equilibrium with arbitrary entropy stratification. Astron. Astrophys. 587(A94) (2016)
- [24] Klein, R.: Semi-implicit extension of a Godunov-type scheme based on low mach number asymptotics I: one-dimensional flow. J. Comput. Phys. 121, 213–237 (1995)
- [25] Li, X.s., Gu, C.w.: Mechanism of Roe-type schemes for all-speed flows and its application. Comput. Fluids 86, 56–70 (2013)
- [26] Miczek, F., R¨opke, F.K., Edelmann, P.V.F.: New numerical solver for flows at various Mach numbers. Astron. Astrophys. 576(A50) (2015)
- [27] Oßwald, K., Siegmund, A., Birken, P., Hannemann, V., Meister, A.: L2Roe: a low dissipation version of Roe's approximate Riemann solver for low Mach numbers. Int. J. Numer. Methods Fluids 81(2), 71–86 (2015)
- [28] Rieper, F.: A low-Mach number fix for Roe's approximate Riemann solver. J. Comput. Phys. 230(13), 5263–5287 (2011)
- [29] Shu, C.-W., Osher, S.: Efficient implementation of essentially non-oscillatory shockcapturing schemes. J. Comput. Phys. 77, 439–471 (1988)
- [30] Suliciu, I.: On modelling phase transitions by means of rate-type constitutive equations. Shock wave structure. Int. J. Eng. Sci. 28(8), 829–841 (1990)
- [31] Thomann, A., Zenk, M., Klingenberg, C.: A second-order positivity-preserving wellbalanced finite volume scheme for Euler equations with gravity for arbitrary hydrostatic equilibria. Int. J. Numer. Methods Fluids 89(11), 465–482 (2018)
- [32] Thomann, A., Puppo, G., Klingenberg, C.: An all speed second order well-balanced IMEX relaxation scheme for the Euler equations with gravity. J. Comput. Phys. 420 (2020)
- [33] Thomann, A., Zenk, M., Puppo, G., Klingenberg, C.: An all speed second order IMEX relaxation scheme for the Euler equations. Commun. Comput. Phys. 28(2), 591–620 (2020)
- [34] Toro, E.F.: Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction. Springer-Verlag, Berlin, third ed., 2009.
- [35] Turkel, E.: Preconditioning techniques in computational fluid dynamics. Ann. Rev. Fluid Mech. 31, 385–416 (1999)

[36] Xing, Y., Shu, C.-W.: High order well-balanced WENO scheme for the gas dynamics equations under gravitational fields. J. of Sci. Comput. 54, 645–662 (2013)