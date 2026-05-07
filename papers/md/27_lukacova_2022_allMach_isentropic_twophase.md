# An all Mach number finite volume method for isentropic two-phase flow

Mária Lukáčová-Medviďová, Gabriella Puppo, Andrea Thomann\*‡ February 4, 2022

## Abstract

We present an implicit-explicit finite volume scheme for isentropic two phase flow in all Mach number regimes. The underlying model belongs to the class of symmetric hyperbolic thermodynamically compatible models. The key element of the scheme consists of a linearisation of pressure and enthalpy terms at a reference state. The resulting stiff linear parts are integrated implicitly, whereas the non-linear higher order and transport terms are treated explicitly. Due to the flux splitting, the scheme is stable under a CFL condition which determined by the resolution of the slow material waves and allows large time steps even in the presence of fast acoustic waves. Further the singular Mach number limits of the model are studied and the asymptotic preserving property of the scheme is proven. In numerical simulations the consistency with single phase flow, accuracy and the approximation of material waves in different Mach number regimes are assessed.

**Keywords.** All-speed scheme, RS-IMEX, two-phase flow, asymptotic preserving, Symmetric Hyperbolic Thermodynamically Compatible models

# 1 Introduction

Multi-phase flows are omnipresent in environmental and industrial processes. The broad range of applications poses an intrinsic problem of modeling two-phase flows. A widely used model was introduced by Baer & Nunziato [4] and still forms the basis of many models used to describe compressible two-phase flows. Since then a wide range of modifications and extensions towards different applications have been proposed, see [1, 3, 23, 26, 28, 38] and contributions mentioned therein. These models are based on conservation laws of mass, momentum and energy for each phase. However the system cannot be written in a flux conservative form which causes problems in predicting correct shock speeds and the formulation of Rankine-Hugoniot conditions [19]. Therefore special techniques in the numerical treatment of non-conservative products are required as proposed, for example, in [2].

Here, we consider an alternative to the Baer & Nunziato formulation of two-phase mixtures, namely a symmetric hyperbolic model in conservation form proposed in [34, 36]. It is based on the

<sup>\*</sup>Institut für Mathematik, Johannes-Gutenberg-Universität Mainz, Staudingerweg 9, 55099 Mainz, Germany (lukacova@mathematik.uni-mainz.de,athomann@uni-mainz.de)

<sup>&</sup>lt;sup>†</sup>Dipartimento di Matematica, La Sapienza Università di Roma, Piazzale Aldo Moro 5, 00185 Roma, Italy (gabriella.puppo@uniroma1.it)

<sup>&</sup>lt;sup>‡</sup>Corresponding author

theory of Symmetric Hyperbolic Thermodynamically Compatible (SHTC) systems [\[16,](#page-32-4) [17,](#page-32-5) [18\]](#page-32-6). The latter class of equations is derived from thermodynamics [\[17,](#page-32-5) [18,](#page-32-6) [37\]](#page-33-3) and variational principles [\[31\]](#page-32-7). The approach is versatile and not restricted to the modeling of two-phase flows. It constitutes a monolithic mathematical framework that encompasses the evolution of all considered materials and provides a unified mathematical description of multi-physics systems, see e.g. [\[33\]](#page-33-4) for a generalization of the two-phase flow model given in [\[34,](#page-33-1) [36\]](#page-33-2) to an arbitrary number of phases, [\[13,](#page-31-4) [30\]](#page-32-8) for applications of the SHTC theory to fluid and solid mechanics modeling, or [\[35\]](#page-33-5) for recent advances in the description of poroelastic fluid-saturated media.

In this work, we focus on the isentropic setting of the two-phase flow model given in [\[34,](#page-33-1) [36\]](#page-33-2). Due to the conservative model formulation, the characteristic fields and wave relations were recently analysed in [\[39\]](#page-33-6). Here, we are interested in the numerical simulation of gas-liquid interactions as they occur in air-water mixtures in form of droplets in air or dispersed bubbles in water. Other possible applications include pipe flows where the transported medium can exist in its liquid and gas state due to depressurization events. Thereby the considered phases exhibit different behaviour with respect to flow properties ranging from compressible for gases to almost incompressible for some liquids. Depending on the application this can imply a significant difference in the propagation speed of acoustic waves. Consequently, the Mach numbers that characterize the flow regime of each phase can differ in several orders of magnitude.

The construction of schemes that are designed for applications in low Mach number regimes is an active field of research, for models based on the Baer-Nunziato model see [\[11,](#page-31-5) [32,](#page-33-7) [29,](#page-32-9) [8\]](#page-31-6). In [\[41\]](#page-33-8) the low Mach limit is considered for the compressible gas phase only, where the liquid phase is described by incompressible flow. The novelty in our work lies in the fact that in the non-dimensionalisation of the here considered model two Mach numbers are considered. They are given by the ratio between the local flow velocity of the mixture and the respective sound speeds of the phases.

A severe difficulty in the construction of a numerical scheme applied to weakly compressible flow regimes is posed by the scale differences between acoustic waves and the material wave. The focus of the numerical simulation usually lies on the evolution of the slower material waves for which a time step oriented towards the local flow speed suffices. The time step of an explicit scheme, as proposed in [\[34,](#page-33-1) [36\]](#page-33-2) for compressible two-phase flow, is bounded by the smallest appearing Mach number. This leads to vanishingly small time steps in low Mach number regimes and consequently to long computational times, especially when long time periods are considered. This defect can be overcome by considering implicit-explicit (IMEX) time integrators, where fast waves are treated implicitly leading to the Courant-Friedrichs-Levy (CFL) condition that is restricted only be the local flow velocity. This allows larger time steps while keeping the material wave well resolved. Additionally, an implicit treatment of the associated stiff pressure terms, which trigger fast acoustic waves, has the advantage that centered differences can be applied without loss of stability. This is important for obtaining the correct numerical viscosity in low Mach number flows as established in [\[10,](#page-31-7) [20\]](#page-32-10), see also [\[14,](#page-31-8) [15\]](#page-31-9) for a particular successive linearisation approach. In particular, the upwind schemes suffer from an excessive numerical diffusion [\[20,](#page-32-10) [25\]](#page-32-11) and are therefore not applicable. Indeed, the correct amount of numerical diffusion is an integral part to obtain so-called asymptotic preserving (AP) schemes [\[21\]](#page-32-12). Since the flow regime of the two-phase flow considered here is characterized by two potentially distinct phase Mach numbers, different singular Mach number limits can be obtained which depend on the constitution of the mixture. For their formal derivation we apply asymptotic expansions, as done for the (isentropic) Euler equations [\[7,](#page-31-10) [9,](#page-31-11) [10,](#page-31-7) [20,](#page-32-10) [24,](#page-32-13) [25\]](#page-32-11), see also references therein. To obtain physically admissible solutions, especially in the weakly compressible flow regime, the numerical scheme has to preserve these asymptotics. This means a consistent discretization of the limit equations as the Mach numbers tend to zero. To the best of our knowledge, this is the first systematic study of the effect of the Mach number scalings in two phase flows indicating the important terms in weakly compressible regimes.

The profound knowledge of the structure of well-prepared initial data can be used to construct an AP scheme by applying a reference solution (RS)-IMEX approach. This approach was successfully applied to construct AP schemes for the (isentropic) Euler equations [\[5,](#page-31-12) [22,](#page-32-14) [27,](#page-32-15) [40\]](#page-33-9). Here, we only linearise the nonlinear pressure based terms around a reference state given by well-prepared data. The stiff linear part is then treated implicitly whereas the non-linear higher order terms are integrated explicitly respecting the asymptotics in the low Mach number limit. By doing this, nonlinear implicit solvers can be avoided which are computationally costly.

The paper is organized as follows. In Section [2](#page-2-0) we revisit the model formulation from [\[36\]](#page-33-2) and give the non-dimensional formulation for which afterwards well-prepared initial conditions and singular Mach number limits are derived. Motivated by the structure of the well-prepared data, the numerical scheme is constructed in Section [3](#page-9-0) based on the reference solution approach. The derivation of the time semi-discrete scheme is discussed in detail and the fully discrete RS-IMEX scheme based on a finite volume framework is formulated. The subsequent section is dedicated to the AP proof. The scheme is validated by numerical tests in different Mach number regimes in Section [5.](#page-23-0) In particular, the consistency with single phase flow, first order accuracy and the behaviour of the scheme for different Riemann problems are assessed. Final conclusions are drawn in Section [6.](#page-27-0)

# 2 Isentropic two-phase flow

# 2.1 The compressible model

The one-dimensional isentropic two phase model as introduced in [\[34\]](#page-33-1) is given by

$$\partial_t \rho + \partial_x (\rho u) = 0,$$
 (2.1a)

$$\partial_t(\alpha_1 \rho) + \partial_x(\alpha_1 \rho u) = -\frac{1}{\tau} (p_2 - p_1), \qquad (2.1b)$$

$$\partial_t(\alpha_1 \rho_1) + \partial_x(\alpha_1 \rho_1 u_1) = 0, \tag{2.1c}$$

$$\partial_t(\rho u) + \partial_x \left(\alpha(\rho_1 u_1^2 + p_1) + \alpha_2(\rho_2 u_2^2 + p_2)\right) = 0,$$
 (2.1d)

$$\partial_t(u_1 - u_2) + \partial_x \left(\frac{u_1^2}{2} + h_1 - \frac{u_2^2}{2} - h_2\right) = -\zeta \chi_1 \chi_2(u_1 - u_2). \tag{2.1e}$$

System [\(2.1\)](#page-2-1) consists of the conservation of mixture mass ρ [\(2.1a\)](#page-2-2) and partial mass αρ<sup>1</sup> [\(2.1c\)](#page-2-3). Equation [\(2.1b\)](#page-2-4) gives the balance of the evolution of the volume fraction α with respect to a pressure relaxation source term, where τ denotes the relaxation rate. Further, equation [\(2.1d\)](#page-2-5) gives the conservation of mixture momentum ρu and equation [\(2.1e\)](#page-2-6) the balance of the relative velocity u<sup>1</sup> − u<sup>2</sup> against a friction source term with a friction coefficient ζ. The densities of the respective phases are denoted by ρ<sup>1</sup> and ρ<sup>2</sup> and the mixture density is given by

$$\rho = \alpha_1 \rho_1 + \alpha_2 \rho_2,$$

where the volume fraction α<sup>1</sup> ∈ (0, 1) is associated to phase one and obeys α<sup>1</sup> + α<sup>2</sup> = 1. The phase velocities are given by u<sup>1</sup> and u2, respectively, and the mixture velocity is defined as

$$u = \chi_1 u_1 + \chi_2 u_2.$$

where  $\chi_1 = \frac{\alpha_1 \rho_1}{\rho}$  denotes the mass fraction of phase one and obeys the relation  $\chi_1 + \chi_2 = 1$ . To close the system, we consider two different equations of states (EOS), the ideal gas law

$$p(\rho) = \kappa \left(\frac{\rho}{\rho_0}\right)^{\gamma}, \quad e(\rho) = \frac{p}{\rho(\gamma - 1)},$$
 (2.2)

and the stiffend gas equation to model liquids

$$p(\rho) = \kappa \left(\frac{\rho}{\rho_0}\right)^{\gamma} - p_{\infty}, \quad e(\rho) = \frac{p + \gamma p_{\infty}}{\rho(\gamma - 1)}.$$
 (2.3)

The parameter  $\gamma$  denotes the adiabatic constant and  $p_{\infty}$  and  $\kappa$  denote positive constants describing the considered medium. The internal energy of the mixture is given by a linear combination of the internal energies  $e_1, e_2$  of the respective phases

$$e(\rho_1, \rho_2) = \chi_1 e_1 + \chi_2 e_2. \tag{2.4}$$

The mixture pressure P is obtained from (2.4) by

$$P = \rho^2 \frac{\partial e}{\partial \rho} = \alpha_1 p_1 + \alpha_2 p_2, \tag{2.5}$$

which is a linear combination of the phase pressures  $p_1, p_2$ . The phase enthalpies  $h_1, h_2$  in equation (2.1e) are defined as  $h_1 = e_1 + \frac{p_1}{\rho_1}$ ,  $h_2 = e_2 + \frac{p_2}{\rho_2}$  and are determined by the respective EOS (2.2),(2.3). Further, we can define the mixture sound speed from (2.5) by

$$c^{2} = \frac{\partial P}{\partial \rho} = \chi_{1}c_{1}^{2} + \chi_{2}c_{2}^{2}, \tag{2.6}$$

where the phase sound speeds  $c_1, c_2$  are given by

$$c_1^2 = \frac{\partial p_1(\rho_1)}{\partial \rho_1}, \quad c_2^2 = \frac{\partial p_2(\rho_2)}{\partial \rho_2}.$$

We distinguish the following types of variables, state variables

$$W = (\rho, \alpha_1 \rho, \alpha_1 \rho_1, \rho u, u_1 - u_2)^T, \tag{2.7}$$

mixture variables

$$Q = (\alpha_1, \rho, \chi_1, u, u_1 - u_2)^T$$
(2.8)

and the phase variables

$$V = (\alpha_1, \rho_1, u_1, \rho_2, u_2)^T.$$

Using the state variables W, we can write (2.1) in a compact form as

$$\partial_t W + \partial_x f(W) = r(W),$$

where f denotes the nonlinear flux function and r the physical relaxation source terms. Following [36], model (2.1) is strictly hyperbolic with the following eigenvalues

$$\lambda^u = u, \quad \lambda_1^{\pm} = u_1 \pm c_1, \quad \lambda_2^{\pm} = u_2 \pm c_2.$$

They can be obtained by diagonalising the Jacobian  $\partial_W f(W)$ . Thus, model (2.1) exhibits two acoustic waves for each phase  $\lambda_1^{\pm}$ ,  $\lambda_2^{\pm}$  and the mixture velocity  $\lambda^u$  which is also referred to as material velocity. Especially, when the sound speeds  $c_1$  and/or  $c_2$  are large, the acoustic waves  $\lambda_1^{\pm}$  and/or  $\lambda_2^{\pm}$  travel consistently faster than the material wave which introduces different scales into the model.

#### 2.2 Non-dimensional formulation

To obtain a better understanding of the scales that are present in the model, we rewrite (2.1) in non-dimensional form. Let us denote the non-dimensional quantities by ( $\tilde{\phantom{x}}$ ) and the corresponding reference value by ( $\cdot$ )<sup>r</sup>. We assume that the convective scales are of the same order, ie.  $u_1^r = u_2^r = u^r = x^r/t^r$  which can be expressed through the reference length  $x^r$  and time  $t^r$ . The ratio of the phase densities however can be large especially when considering mixtures of light gases and liquids. To take this potentially large difference into account, we define two different reference densities which are connected to the reference mixture density by  $\rho_1^r = \varrho_1 \rho^r$ ,  $\rho_2^r = \varrho_2 \rho^r$ , where  $\varrho_1, \varrho_2 \in \mathbb{R}$  are scaling constants. Further we define two different reference pressures  $p_1^r$ ,  $p_2^r$  from which we can compute the reference sound speeds given by

$$(c_1^r)^2 = \frac{p_1^r}{\rho_1^r} = \frac{1}{\varrho_1} \frac{p_1^r}{\rho^r}, \quad (c_2^r)^2 = \frac{p_2^r}{\rho_2^r} = \frac{1}{\varrho_2} \frac{p_2^r}{\rho^r}.$$

For each phase, we can define a reference Mach number which is given by the ratio of the reference velocities and sound speeds

$$M_1 = \frac{u^r}{c_1^r} = \sqrt{\varrho_1} \; \frac{u^r}{\sqrt{p_1^r/\rho^r}} = \sqrt{\varrho_1} \; M_1^*, \quad M_2 = \frac{u^r}{c_2^r} = \sqrt{\varrho_2} \; \frac{u^r}{\sqrt{p_2^r/\rho^r}} = \sqrt{\varrho_2} \; M_2^*.$$

We note that the Mach numbers  $M_1, M_2$  are defined using reference phase densities, whereas  $M_1^*, M_2^*$  are defined using the reference mixture density only. Summarizing, we can express the dimensional variables as the product of non-dimensional quantity and reference value as follows

$$\rho = (\alpha_1 \varrho_1 \widetilde{\rho}_1 + \alpha_2 \varrho_2 \widetilde{\rho}_2) \rho^r, \qquad \rho_1 = \widetilde{\rho}_1 \varrho_1 \rho^r, \qquad \rho_2 = \widetilde{\rho}_2 \varrho_2 \rho^r, \nu = \left(\frac{\alpha_1 \varrho_1 \widetilde{\rho}_1}{\widetilde{\rho}} \widetilde{u}_1 + \frac{\alpha_2 \varrho_2 \widetilde{\rho}_2}{\widetilde{\rho}} \widetilde{u}_2\right) u^r, \qquad u_1 = \widetilde{u}_1 u^r, \qquad u_2 = \widetilde{u}_2 u^r, \qquad (2.9)$$

$$p_1 = \widetilde{p}_1 p_1^r, \qquad p_2 = \widetilde{p}_2 p_2^r.$$

In addition, we have also the following reference values for the pressure relaxation rate and friction coefficient given respectively by

$$\tau^r = (u^r)^2 t^r$$
 and  $\zeta^r = \frac{1}{t^r}$  defining  $\tau = \tilde{\tau} \tau^r$ ,  $\zeta = \tilde{\zeta} \zeta^r$ . (2.10)

Inserting expressions (2.9), (2.10) into (2.1) and dropping the  $(\widetilde{\cdot})$ , we obtain the following non-dimensional formulation

$$\partial_t \rho + \partial_x (\rho u) = 0,$$
 (2.11a)

$$\partial_t(\alpha_1 \rho) + \partial_x(\alpha_1 \rho u) = -\frac{1}{\tau} \left( \frac{\varrho_2 p_2}{M_2^2} - \frac{\varrho_1 p_1}{M_1^2} \right), \qquad (2.11b)$$

$$\partial_t(\alpha_1 \rho_1) + \partial_x(\alpha_1 \rho_1 u_1) = 0, \tag{2.11c}$$

$$\partial_t(\rho u) + \partial_x \left( \alpha_1 \varrho_1 \rho_1 u_1^2 + \frac{\alpha_1 \varrho_1 p_1}{M_1^2} + \alpha_2 \varrho_2 \rho_2 u_2^2 + \frac{\alpha_2 \varrho_2 p_2}{M_2^2} \right) = 0, \tag{2.11d}$$

$$\partial_t(u_1 - u_2) + \partial_x \left( \frac{u_1^2}{2} - \frac{u_2^2}{2} + \frac{h_1}{M_1^2} - \frac{h_2}{M_2^2} \right) = -\zeta \chi_1 \chi_2(u_1 - u_2)$$
 (2.11e)

with

$$\rho = \alpha_1 \varrho_1 \rho_1 + \alpha_2 \varrho_2 \rho_2, \quad u = \chi_1 u_1 + \chi_2 u_2, \quad \chi_1 = \varrho_1 \frac{\alpha_1 \rho_1}{\rho}.$$

Note that the mass fraction still obeys  $\chi_1 + \chi_2 = 1$ . Analogously to the dimensional formulation (2.1), the non-dimensional model (2.11) is strictly hyperbolic and exhibits 5 waves given by

$$\lambda^{u} = u, \quad \lambda_{1}^{\pm} = u_{1} \pm \frac{c_{1}}{M_{1}}, \quad \lambda_{2}^{\pm} = u_{2} \pm \frac{c_{2}}{M_{2}}.$$

We see that the acoustic waves scale with the phase Mach numbers  $M_1, M_2$  respectively and propagate significantly faster than the material wave for low Mach numbers. To obtain more insight on the behaviour of the solution in the low Mach number limit, we perform an asymptotic analysis of the non-dimensional model (2.11). This is subject of the next section.

## 2.3 Well-prepared data and the low Mach limit

According to [10] an initial condition of (2.11) is called well-prepared, if it is close to a solution of a limit model for small Mach numbers. For the derivation of well-prepared initial data and the low Mach limit model of (2.11), we focus on the scales induced by the Mach numbers  $M_1, M_2$ . The scaling factors of the densities  $\varrho_1, \varrho_2$  are considered to be fixed values independent of the Mach regimes. For simplicity of notation, we will neglect the scaling parameters  $\varrho_1, \varrho_2$  in the following analysis. We consider the subsequent cases in detail:

- Case 1: Phase one is compressible, i.e.  $M_1 = 1$ , and phase two is characterized by a low Mach number  $M_2 \ll 1$ .
- Case 2: Both phases are in the same Mach regime, i.e.  $M_1 = M_2 = M$ .

Case 1 Let phase one be compressible with  $\rho_1 = \mathcal{O}(1)$  and  $u_1 = \mathcal{O}(1)$ . The variables of the weakly incompressible phase two are expanded with respect to a Mach number M in the following way

$$\rho_2 = \rho_2^{(0)} + M\rho_2^{(1)} + M^2\rho_2^{(2)} + \mathcal{O}(M^3), \quad u_2 = u_2^{(0)} + Mu_2^{(1)} + \mathcal{O}(M^2). \tag{2.12}$$

The Mach number expansion of the pressure can be obtained from the density expansion (2.12) via the respective EOS. With  $p_2(\rho_2^{(0)}) = p_2^{(0)}$  and  $(c_2^{(0)})^2 = \gamma_2 \frac{p_2(0)}{\rho_2^{(0)}}$  we obtain the following expansion

$$p_2(x,t) = p_2^{(0)} + (c_2^{(0)})^2 \rho_2^{(1)} M + \left(\frac{1}{2} (1 - \gamma_2) \frac{(c_2^{(0)})^2}{\rho_2^{(0)}} (\rho_2^{(1)})^2 + (c_2^{(0)})^2 \rho_2^{(2)}\right) M^2 + \mathcal{O}(M^3). \tag{2.13}$$

Further we assume a Mach number expansion of the volume fraction given by

$$\alpha = \alpha^{(0)} + \mathcal{O}(M). \tag{2.14}$$

Inserting the expansions into the non-dimensional equations (2.11) and sorting by orders of the Mach number we find from (2.11b), (2.11d) and (2.11e) for the  $\mathcal{O}(M^{-2})$  order terms

$$p_2^{(0)} = 0$$
,  $\partial_x \left( \alpha^{(0)} p_2^{(0)} \right) = 0$ ,  $\frac{\partial_x p_2^{(0)}}{\rho_2^{(0)}} = 0$ .

We immediately find from (2.13) and the EOS that  $\rho_2^{(0)}$  is a non-negative constant. Especially for an ideal gas follows  $\rho_2^{(0)} = 0$  which means phase two is vanishing or in vacuum at leading order. Since we are interested in obtaining a mixture of two phases also in the limit, we assume phase two

to be associated with the stiffened gas equation, thus  $\rho_2^{(0)}$  is positive and constant. Analogously, we find for the order  $\mathcal{O}(M^{-1})$  terms

$$p_2^{(1)} = (c_2^{(0)})^2 \rho_2^{(1)} = 0$$

which implies  $\rho_2^{(1)} = 0$  according to (2.13). Summarizing, the well-prepared data for phase density two is given by

$$\rho_2 = \rho_2^{(0)} + \mathcal{O}(M^2), \quad \rho_2^{(0)} \text{ const.}$$

For  $\mathcal{O}(1)$  terms we find for the mixture density and velocity the following relations

$$\rho^{(0)} = \alpha^{(0)}\rho_1 + (1 - \alpha^{(0)})\rho_2^{(0)}, \quad u^{(0)} = \frac{\alpha^{(0)}\rho_1}{\rho^{(0)}}u_1 + \frac{(1 - \alpha^{(0)}\rho_2^{(0)})}{\rho^{(0)}}u_2^{(0)} = \chi_1^{(0)}u_1 + \chi_2^{(0)}u_2^{(0)}.$$

Using this notation, we find

$$\partial_t \alpha_2^{(0)} + \partial_x (\alpha_2^{(0)} u_2^{(0)}) = 0 (2.15)$$

$$\partial_t \alpha^{(0)} + u^{(0)} \partial_x \alpha^{(0)} = -\frac{1}{\tau \rho^{(0)}} (p_2^{(2)} - p_1)$$
 (2.16)

$$\partial_t(\alpha^{(0)}\rho_1) + \partial_x(\alpha^{(0)}\rho_1 u_1) = 0 (2.17)$$

$$\partial_t(\rho^{(0)}u^{(0)}) + \partial_x(\alpha^{(0)}\rho_1u_1^2 + \alpha^{(0)}p_1 + \alpha_2^{(0)}\rho_2^{(0)}(u_2^{(0)})^2 + \alpha_2^{(0)}p_2^{(2)} = 0$$
(2.18)

$$\partial_t(u_1 - u_2^{(0)}) + \partial_x \left(\frac{1}{2}u_1^2 - \frac{1}{2}(u_2^{(0)})^2\right) + \frac{\partial_x p_1}{\rho_1} - \frac{\partial_x p_2^{(2)}}{\rho_2^{(0)}} = \zeta \chi_1^{(0)} \chi_2^{(0)}(u_1 - u_2^{(0)}). \tag{2.19}$$

Integrating (2.15) over the domain  $\Omega$  with periodic or no-flux boundary conditions, as done in [10, 9], we obtain that the average of  $\alpha^{(0)}$  is constant in time. From this it follows immediately that

$$u^{(0)}\partial_x \alpha^{(0)} = -\frac{1}{\rho^{(0)}} (p_2^{(2)} - p_1)$$
(2.20)

and we obtain

$$\partial_x u_2^{(0)} = \frac{u_2^{(0)}}{\alpha_2^{(0)}} \partial_x \alpha^{(0)}.$$

Further we get from (2.18) and (2.19)

$$\partial_t u_2^{(0)} + u_2^{(0)} \partial_x u_2^{(0)} + \frac{\partial_x p_2^{(2)}}{\rho_2^{(0)}} + \frac{p_2^{(2)} - p_1}{\rho^{(0)}} \partial_x \alpha^{(0)} = \zeta(\chi_1^{(0)})^2 \chi_2^{(0)} (u_1 - u_2^{(0)}). \tag{2.21}$$

For the compressible phase we obtain from (2.17), (2.18) and (2.21)

$$\partial_t(\alpha^{(0)}\rho_1) + \partial_x(\alpha^{(0)}\rho_1 u_1) = 0$$

$$\partial_t u_1 + u_1 \partial_x u_1 + \frac{\partial_x p_1}{\rho_1} + \frac{p_2^{(2)} - p_1}{\rho} \partial_x \alpha^{(0)} = -\zeta \chi_1^{(0)} (\chi_2^{(0)})^2 (u_1 - u_2^{(0)}).$$
(2.22)

Note that  $\partial_x \alpha^{(0)} = 0$ , i.e. constant  $\alpha^{(0)}$ , yields  $\partial_x u_2^{(0)} = 0$  and in the limit  $M \to 0$  we formally obtain the incompressible Euler equations with friction for phase two and the compressible isentropic Euler equations with friction for phase one. In particular, in the limit we have only one pressure given by  $p_1$  and both phases are decoupled.

Summarizing, we have derived the following result regarding well-prepared initial data for two phase flow consisting of a compressible and a weakly incompressible phase.

Lemma 2.1 (Well-prepared data for compressible/weakly compressible flow). Let phase one be compressible, i.e. characterized by M<sup>1</sup> = 1 and phase two be weakly compressible, i.e. characterized by M<sup>2</sup> = M 1. Let α, ρ2, u<sup>2</sup> be given by the Mach number expansions [\(2.14\)](#page-5-2), [\(2.12\)](#page-5-0) and the set of well-prepared initial data be defined as

$$\Omega_1^{wp} = \left\{ W \in \mathbb{R}^{2d+3} : \rho_2^{(0)} \ const. \ , \ p_2(\rho_2^{(0)}) = 0, \ \rho_1^{(1)} = 0, \ \partial_t \alpha^{(0)} = 0, \ \partial_x u_2^{(0)} = \frac{u_2^{(0)}}{\alpha_2^{(0)}} \partial_x \alpha^{(0)} \right\}. \tag{2.23}$$

Then formally for W ∈ Ω wp 1 for M → 0 the limit equations are given by [\(2.20\)](#page-6-5),[\(2.21\)](#page-6-4),[\(2.22\)](#page-6-6). If in addition holds ∂xα (0) = 0, i.e. α (0) is constant, then the limit equations are given by

$$\partial_t u_2^{(0)} + \frac{\partial_x p_1}{\rho_2^{(0)}} = \zeta(\chi_1^{(0)})^2 \chi_2^{(0)} (u_1 - u_2^{(0)}), \quad \partial_x u_2^{(0)} = 0$$

$$\partial_t \rho_1 + \partial_x (\rho_1 u_1) = 0,$$

$$\partial_t (\rho_1 u_1) + \partial_x (\rho_1 (u_1)^2 + \partial_x p_1) = -\zeta \chi_1^{(0)} (\chi_2^{(0)})^2 (u_1 - u_2^{(0)}).$$
(2.24)

Case 2 When the flow of both phases can be characterized by the same Mach number M, i.e. M<sup>1</sup> = O(M) and M<sup>2</sup> = O(M), we consider a Mach number expansion of the variables of both phases. Denoting the different phases by k = 1, 2, we obtain for the densities and velocities

$$\rho_k(x,t) = \rho_k^{(0)}(x,t) + \rho_k^{(1)}(x,t)M + \rho_k^{(2)}(x,t)M^2 + \mathcal{O}(M^3), \tag{2.25}$$

$$u_k(x,t) = u_k^{(0)}(x,t) + u_k^{(1)}(x,t)M + \mathcal{O}(M^2).$$
(2.26)

As in the previous case, we can write a Mach number expansion of the associated phase pressures given by [\(2.13\)](#page-5-1) for each phase respectively and we consider a Mach number expansion of the volume fraction given by [\(2.14\)](#page-5-2).

Inserting the expansions [\(2.13\)](#page-5-1) into the balance law for the volume fraction [\(2.11b\)](#page-4-3) and sorting the terms by orders of the Mach number, we find for O(M−<sup>2</sup> ) the following relation

$$p_2^{(0)} = p_1^{(0)}. (2.27)$$

Using [\(2.27\)](#page-7-0) in the momentum equation [\(2.11d\)](#page-4-4), we find

$$0 = \partial_x (\alpha_1^{(0)} p_1^{(0)} + \alpha_2^{(0)} p_2^{(0)}) = \partial_x p_1^{(0)}.$$

Analogously, we obtain p (1) <sup>2</sup> = p (1) 1 and ∂xp (1) <sup>1</sup> = 0 for the O(M−<sup>1</sup> ) terms. Summarizing, the pressure expansions are given by

$$p_1(x,t) = p^{(0)}(t) + M^2 p_1^{(2)}(x,t) + \mathcal{O}(M^3), \quad p_2(x,t) = p^{(0)}(t) + M^2 p_2^{(2)}(x,t) + \mathcal{O}(M^3)$$
 (2.28)

with spatially constant component p (0). Comparing [\(2.28\)](#page-7-1) with [\(2.13\)](#page-5-1), we obtain the following expansions for the phase densities

$$\rho_1(x,t) = \rho_1^{(0)}(t) + M^2 \rho_1^{(2)}(x,t) + \mathcal{O}(M^3), \quad \rho_2(x,t) = \rho_2^{(0)}(t) + M^2 \rho_2^{(2)}(x,t) + \mathcal{O}(M^3),$$

where the zero order components only depend on time. Following [\[9,](#page-31-11) [10\]](#page-31-7), we obtain by integrating [\(2.11a\)](#page-4-6) and [\(2.11c\)](#page-4-7) on a domain Ω with periodic or no-flux boundary conditions the following conditions on the densities

$$\partial_t \rho^{(0)} = 0, \quad \partial_t \left( \alpha_1^{(0)} \rho_1^{(0)} \right) = 0.$$
 (2.29)

From (2.29) we derive, using  $\partial_x \rho_1^{(0)} = 0$  and  $\partial_x \rho_2^{(0)} = 0$ , the following conditions on the velocity derivatives

$$\partial_x u_1^{(0)} = -\frac{u_1^{(0)}}{\alpha_1^{(0)}} \partial_x \alpha^{(0)}, \quad \partial_x u_2^{(0)} = \frac{u_2^{(0)}}{\alpha_2^{(0)}} \partial_x \alpha^{(0)}. \tag{2.30}$$

Looking at the  $\mathcal{O}(1)$  terms we obtain the following limit system when M goes to zero

$$\partial_t \alpha^{(0)} + u^{(0)} \partial_x \alpha^{(0)} = -\frac{1}{\tau \rho^{(0)}} (p_2^{(2)} - p_1^{(2)})$$
 (2.31a)

$$\partial_t u_1^{(0)} + u_1^{(0)} \partial_x u_1^{(0)} + \frac{\partial_x p_1^{(2)}}{\rho_1^{(0)}} + \frac{p_2^{(2)} - p_1^{(2)}}{\rho^{(0)}} \partial_x \alpha^{(0)} = -\zeta \chi_1^{(0)} (\chi_2^{(0)})^2 (u_1^{(0)} - u_2^{(0)})$$
(2.31b)

$$\partial_t u_2^{(0)} + u_2^{(0)} \partial_x u_2^{(0)} + \frac{\partial_x p_2^{(2)}}{\rho_2^{(0)}} + \frac{p_2^{(2)} - p_1^{(2)}}{\rho^{(0)}} \partial_x \alpha^{(0)} = \zeta(\chi_1^{(0)})^2 \chi_2^{(0)} (u_1^{(0)} - u_2^{(0)})$$
(2.31c)

where  $\partial_t \rho^{(0)} = 0$ . Analogously to Case 1, we see from (2.30) that for  $\partial_x \alpha^{(0)} = 0$  the derivatives of the velocities vanish and the phases are only coupled via the pressure relaxation and friction source terms. Moreover, equations (2.31b) and (2.31c) are two coupled incompressible Euler equations with variable densities. For constant  $\alpha^{(0)}$  however, we immediately find from (2.29) that  $\rho_1^{(0)}, \rho_2^{(0)}$  are constant and for the pressures holds  $p_2^{(2)} = p_1^{(2)}$  in (2.31a). In particular, this leads to a system of incompressible Euler equations coupled via the friction source term with a single pressure  $p^{(2)}$ .

Summarizing, we have derived the following result regarding well-prepared initial data for two weakly compressible phases in the same Mach regime.

**Lemma 2.2** (Well-prepared data for weakly compressible flow). Let both phases be weakly compressible in the same Mach number regime, i.e. characterized by the same Mach number M. Let the phase variables  $V \in \mathbb{R}^{2d+3}$  be given in the Mach number expansions (2.25), (2.26) and (2.14) and the set of well-prepared data be defined as

$$\Omega_M^{wp} = \left\{ w \in \mathbb{R}^{2d+3} : \partial_t \rho^{(0)} = 0, \ \partial_t (\alpha^{(0)} \rho_1^{(0)}) = 0, \ \partial_x \rho_1^{(0)} = 0, \ \partial_x \rho_2^{(0)} = 0, \ \rho_1^{(1)} = 0, \ \rho_2^{(1)} = 0, \\
p_1^{(0)} = p_2^{(0)}, \ \partial_x u_1^{(0)} = -\frac{u_1^{(0)}}{\alpha^{(0)}} \partial_x \alpha_1^{(0)}, \ \partial_x u_2^{(0)} = \frac{u_2^{(0)}}{\alpha_2^{(0)}} \partial_x \alpha^{(0)} \right\}.$$
(2.32)

Then formally for  $W \in \Omega_M^{wp}$  for  $M \to 0$  the limit equations are given by (2.31). If in addition  $\alpha^{(0)}$  is constant, we obtain constant  $\rho_1^{(0)}, \rho_2^{(0)}$  and the following limit equations

$$\partial_{t}u_{1}^{(0)} + \frac{\partial_{x}p^{(2)}}{\rho_{1}^{(0)}} = -\zeta\chi_{1}^{(0)}(\chi_{2}^{(0)})^{2}(u_{1}^{(0)} - u_{2}^{(0)}), \quad \partial_{x}u_{1}^{(0)} = 0$$

$$\partial_{t}u_{2}^{(0)} + \frac{\partial_{x}p^{(2)}}{\rho_{2}^{(0)}} = \zeta(\chi_{1}^{(0)})^{2}\chi_{2}^{(0)}(u_{1}^{(0)} - u_{2}^{(0)}), \quad \partial_{x}u_{2}^{(0)} = 0$$

$$(2.33)$$

with a single pressure  $p^{(2)} = p_1^{(2)} = p_2^{(2)}$ .

For completeness, we shortly mention the case where the two phases are weakly compressible and in different Mach number regimes. Without loss of generality, we assume  $M_2 \ll M_1 \ll 1$ . To obtain the simultaneous limit for  $M_2 \to 0$  and  $M_1 \to 0$  we consider the following Mach number expansions in  $M_1, M_2$  of the phase variables

$$V = \sum_{j,l=0}^{\infty} M_1^j M_2^l V^{(j,l)}.$$

Via the EOS we obtain an analogous Mach number expansion of the pressures. Following the above procedure for Cases 1 and 2, we obtain the following set of well-prepared data given by

$$\Omega_2^{wp} = \left\{ W \in \mathbb{R}^{2d+3} : \alpha^{(0,0)} \text{ const. }, \ \rho_1^{(0,0)} \text{ const. }, \right.$$

$$\rho_2^{(0,0)} = \text{ const. }, \rho_1^{(j,l)} = 0, \ \rho_2^{(j,l)} = 0 \text{ for } j+l=1, j=l=1,$$

$$p_1(\rho_1^{(0,0)}) = 0, \ p_2(\rho_2^{(0,0)}) = 0, \ p_1^{(2,0)} = p_2^{(0,2)}, \ \partial_x u_1^{(0,0)} = 0, \ \partial_x u_2^{(0,0)} = 0. \right\} (2.34)$$

In particular we obtain

$$\rho_1 = \rho_1^{(0,0)} + \mathcal{O}(M_1^2), \quad \rho_2 = \rho_2^{(0,0)} + \mathcal{O}(M_2^2).$$

The limit equations with constant  $\alpha^{(0,0)}$  are given by

$$\partial_t u_1^{(0,0)} + \frac{\partial_x p^{(2)}}{\rho_1^{(0,0)}} = -\zeta \chi_1^{(0,0)} (\chi_2^{(0,0)})^2 (u_1^{(0,0)} - u_2^{(0,0)}), \quad \partial_x u_1^{(0,0)} = 0$$

$$\partial_t u_2^{(0,0)} + \frac{\partial_x p^{(2)}}{\rho_2^{(0,0)}} = \zeta (\chi_1^{(0,0)})^2 \chi_2^{(0,0)} (u_1^{(0,0)} - u_2^{(0,0)}), \quad \partial_x u_2^{(0,0)} = 0$$

with a single pressure  $p^{(2)} = p_1^{(2,0)} = p_2^{(0,2)}$ .

# 3 The numerical scheme

The main objective for constructing a numerical scheme for the two phase model (2.11) is to achieve stability under a time step restriction independently of the Mach numbers  $M_1$  and  $M_2$ . This allows to follow the material wave  $\lambda^u$  while neglecting the resolution of the acoustic waves. Therefore we desire a CFL condition of the type

$$\Delta t \le \nu_u \frac{\Delta x}{\max |\lambda_u|}.\tag{3.1}$$

An explicit scheme requires for stability a quite severe time step restriction of

$$\Delta t \le \nu_{ac} \frac{\Delta x}{\max(|\lambda_1^{\pm}|, |\lambda_2^{\pm}|)} \le \nu_{ac} \min(M_1, M_2) \frac{\Delta x}{\max(|M_2 u_2 \pm c_2|, |M_1 u_1 \pm c_1|)}$$

which vanishes as one of the Mach numbers tends to 0. To avoid the costs that arise from being forced to use small time steps in low Mach number regimes, we construct a numerical scheme based on an implicit-explicit (IMEX) approach where the fast waves are integrated implicitly thus do not contribute to the CFL condition.

## 3.1 Reference solution approach

To determine which terms should be treated implicitly, we analyse the eigenstructure of the model in non-dimensional formulation (2.11). Thereby we find that the fast acoustic components of the eigenvalues  $\lambda_1^{\pm}$ ,  $\lambda_2^{\pm}$  stem from the respective phase pressure and enthalpy terms. Therefore it is necessary to treat those terms implicitly in order to obtain a CFL condition that is independent of the Mach number regimes. Considering the EOS given in (2.2) and (2.3), both pressures  $p_1, p_2$  and enthalpies  $h_1, h_2$  are non-linear functions of the densities  $\rho_1, \rho_2$  which would require

a nonlinear solver. To avoid this, we linearise the phase pressures and enthalpies with respect to a known reference solution  $\rho_1^{RS}$  and  $\rho_2^{RS}$ , respectively, as proposed in [22]. Those reference states are motivated by the leading order terms of the Mach number expansions obtained in the derivation of well-prepared initial data given in (2.23), (2.32) or (2.34). Here we focus on the case where the reference states are constant throughout the simulation thus considering the transition from weakly compressible to incompressible flow in the limit.

In the following, we detail the computations for the first phase only. The formulations for the second phase are similar. To linearise pressure and enthalpy, we consider the Taylor expansion with respect to the reference state  $\rho_1^{RS}$  which reads

$$p_{1}(\rho_{1}) = p_{1}^{RS} + \left(c_{1}^{RS}\right)^{2} (\rho_{1} - \rho_{1}^{RS}) + \mathcal{O}\left(\left(\rho_{1} - \rho_{1}^{RS}\right)^{2}\right),$$
  
$$h_{1}(\rho_{1}) = h_{1}^{RS} + \frac{\left(c_{1}^{RS}\right)^{2}}{\rho_{1}^{RS}} \left(\rho_{1} - \rho_{1}^{RS}\right) + \mathcal{O}\left(\left(\rho_{1} - \rho_{1}^{RS}\right)^{2}\right),$$

We split  $p_1$  and  $h_1$  into a part linear in  $\rho_1$ 

$$\hat{p}_1(\rho_1) = p_1^{RS} + \left(c_1^{RS}\right)^2 (\rho_1 - \rho_1^{RS}), \tag{3.2}$$

$$\hat{h}_1(\rho_1) = h_1^{RS} + \frac{\left(c_1^{RS}\right)^2}{\rho_1^{RS}} \left(\rho_1 - \rho_1^{RS}\right) \tag{3.3}$$

and non-linear higher order terms

$$\bar{p}_1(\rho_1) = p_1(\rho_1) - \hat{p}_1(\rho_1) = \mathcal{O}\left(\left(\rho_1 - \rho_1^{RS}\right)^2\right),$$
 (3.4)

$$\bar{h}_1(\rho_1) = h_1(\rho_1) - \hat{h}_1(\rho_1) = \mathcal{O}\left(\left(\rho_1 - \rho_1^{RS}\right)^2\right).$$
 (3.5)

Especially, if  $\rho_1$  is well-prepared, ie.  $\rho_1 = \rho_1^{RS} + \mathcal{O}(M_1^2)$ , we obtain  $\bar{p}_1 = \mathcal{O}(M_1^4)$  and  $\bar{h}_1 = \mathcal{O}(M_1^4)$ , whereas  $\hat{p}_1 = \mathcal{O}(1)$  and  $\hat{h}_1 = \mathcal{O}(1)$ . We will refer to  $\hat{p}_1, \hat{h}_1$  as the fast acoustic pressure and enthalpy, respectively, as they are the main contributors to the acoustic waves, whereas  $\bar{p}_1, \bar{h}_1$  vanish as  $M_1$  tends to 0. Therefore, in the first steps of the construction of the numerical scheme, we will neglect  $\bar{p}_1, \bar{p}_2$  and  $\bar{h}_1, \bar{h}_2$  obtaining a pure low Mach number scheme. The higher order pressure and enthalpy terms will be included afterwards leading to an all-speed scheme.

#### 3.2 Time semi-discrete scheme

We first consider the homogeneous model

$$\partial_t W + \partial_x \hat{f}(W) = 0,$$

where  $\hat{f}(W)$  denotes the flux function with truncated pressure and enthalpy terms and is given by

$$\hat{f}(W) = \begin{pmatrix} \rho u \\ \alpha_1 \rho u \\ \alpha_1 \rho_1 u_1 \\ \alpha_1 \varrho_1 \rho_1 u_1^2 + \frac{\alpha_1 \varrho_1 \hat{p}_1}{M_1^2} + \alpha_2 \varrho_2 \rho_2 u_2^2 + \frac{\alpha_2 \hat{p}_2}{M_2^2} \\ \frac{u_1^2}{2} - \frac{u_2^2}{2} + \frac{\hat{h}_1}{M_1^2} - \frac{\hat{h}_2}{M_2^2} \end{pmatrix}$$
(3.6)

The pressure relaxation and friction source terms will be added in the last step of the construction of the all-speed scheme. In total the equations [\(2.11\)](#page-4-2) will be split in four parts since they contain two fast scales connected to the pressures and the enthalpies, stiff relaxation terms and order one terms connected to the material velocity. The challenge in identifying suitable subsystems of the homogeneous system lies firstly in their well-posedness, i.e. constituting hyperbolic systems on their own, and secondly they should be easily solvable. The order in which they are described in the next subsections follows also their order in the final numerical scheme given in Section [3.3.](#page-17-0)

#### 3.2.1 Fast acoustics

As the pressure terms in the momentum equation are stiff for small Mach numbers, they are treated implicitly. From the flux function [\(3.6\)](#page-10-0), we see that they couple with the mixture mass flux ρu which is therefore also treated implicitly. This strategy is also used in the construction of IMEX schemes for single phase isentropic Euler equations [\[9\]](#page-31-11). Therefore we propose to solve the following system implicitly

$$\partial_t \rho + \partial_x (\rho u) = 0 \tag{3.7a}$$

$$\partial_t(\rho u) + \partial_x \left( \alpha \varrho_1 \frac{\hat{p}_1}{M_1^2} + \alpha_2 \varrho_2 \frac{\hat{p}_2}{M_2^2} \right) = 0.$$
 (3.7b)

The system is strictly hyperbolic with the following eigenvalues

$$\lambda^{\pm} = \pm \sqrt{\frac{\partial \hat{P}}{\partial \rho}} = \pm \sqrt{\chi_1 \frac{(c_1^{RS})^2}{M_1^2} + \chi_2 \frac{(c_2^{RS})^2}{M_2^2}} = \pm c^{RS}(\chi). \tag{3.8}$$

Thereby c RS(χ) denotes the mixture sound speed with the reference phase sound speeds. The associated eigenvectors are given by

$$\mathbf{v}^{\pm} = \begin{pmatrix} \pm \frac{1}{c^{RS}(\chi)} \\ 1 \end{pmatrix}.$$

Note that ˆp<sup>1</sup> and ˆp<sup>2</sup> are linear functions of ρ1, ρ2, respectively, but are nonlinear in ρ when rewritten in state variables W as follows

$$\rho_1 = \frac{\alpha \rho_1}{\alpha \rho} \rho = \frac{W_3}{W_2} W_1, \quad \rho_2 = \frac{\rho - \alpha \varrho_1 \rho_1}{\varrho_2 (\rho - \alpha \rho)} \rho = \frac{W_1 - W_3}{\varrho_2 (W_1 - W_2)} W_1. \tag{3.9}$$

Considering the phase densities in terms of mixture variables Q defined in [\(2.8\)](#page-3-4) however leads to a linear dependency on ρ given by

$$\rho_1 = \frac{\chi}{\varrho_1 \alpha} \rho = \frac{Q_3}{\varrho_1 Q_2} Q_1, \quad \rho_2 = \frac{1 - \chi}{\varrho_2 (1 - \alpha)} \rho = \frac{1 - Q_3}{\varrho_2 (1 - Q_2)} Q_1. \tag{3.10}$$

Note that the formulation of ρ<sup>1</sup> in state or mixture variables coincides whereas the structure of ρ<sup>2</sup> differs with respect to ρ. We choose to the linear formulation [\(3.10\)](#page-11-0) in mixture variables for the phase densities. Then we can write [\(3.7b\)](#page-11-1) as follows

$$\partial_t(\rho u) + \partial_x \left( \frac{\alpha \varrho_1}{M_1^2} \left( p_1^{RS} - \rho_1^{RS} (c_1^{RS})^2 \right) + \frac{(1 - \alpha)\varrho_2}{M_2^2} \left( p_2^{RS} - \rho_2^{RS} (c_2^{RS})^2 \right) + \rho \left( c^{RS}(\chi) \right)^2 \right) = 0. \quad (3.11)$$

We discretize [\(3.7a\)](#page-11-2) and [\(3.11\)](#page-11-3) implicitly by applying a backward Euler scheme in time. The volume fraction α and mass fraction χ are not evolved in this step [\(3.7\)](#page-11-4) and thus treated explicitly. Discrete time increments are defined as  $t^{n+1} = t^n + \Delta t$ , where  $\Delta t$  is is the time dependent time step and obeys a time step restriction given by a CFL condition. Since the backward Euler scheme is unconditionally stable, we do not obtain a CFL restriction from this step of the numerical scheme. Further, we substitute  $(\rho u)^{n+1}$  in the density flux by the relation obtained from the momentum update. Let  $\rho^*$  be the update of  $\rho$  at the end of the integration of (3.11) with time step  $\Delta t$  starting from data at the previous time  $t^n$ . Then we obtain the following linear implicit equation for  $\rho^*$  given by

$$\rho^{\star} - \Delta t^{2} \partial_{x}^{2} \left( (c^{RS}(\chi^{n}))^{2} \rho^{\star} \right) = \rho^{n} - \Delta t \partial_{x} (\rho u)^{n} + \Delta t^{2} \partial_{x}^{2} \left( \frac{\alpha^{n} \varrho_{1}}{M_{1}^{2}} \eta_{1}^{RS} + \frac{(1 - \alpha^{n}) \varrho_{2}}{M_{2}^{2}} \eta_{2}^{RS} \right), \quad (3.12)$$

where in  $\eta_1^{RS} = p_1^{RS} - \rho_1^{RS}(c_1^{RS})^2$  the reference terms of phase one are collected. Analogously we define  $\eta_2^{RS} = p_2^{RS} - \rho_2^{RS}(c_2^{RS})^2$ . Note that  $c^{RS}(\chi^n)$  is always positive and the coefficient matrix after applying a centred differences on the space derivates is a symmetric positive definite matrix. For details on the space discretisation we refer to Section 3.3 Thus, the linear system is well-defined and can be solved efficiently with standard linear solvers. The momentum is then updated in state variables by

$$(\rho u)^* = (\rho u)^n - \Delta t \partial_x \left( \frac{(\rho \alpha)^n}{\rho^*} \frac{\varrho_1 \hat{p}_1(\rho_1^*)}{M_1^2} + \frac{\rho^* - (\rho \alpha)^n}{\rho^*} \frac{\varrho_2 \hat{p}_2(\rho_2^*)}{M_2^2} \right),$$

 $\rho_1^{\star}, \rho_2^{\star}$  are calculated according to (3.9). The updated state variables after the first stage of the numerical scheme are given by  $W^{\star} = (\rho^{\star}, (\alpha \rho)^n, (\alpha \rho_1)^n, (\rho u)^{\star}, (u_1 - u_2)^n)^T$ .

#### 3.2.2 Nonlinear transport

In the next step we identify nonlinear transport terms of the flux function  $\hat{f}$  defined in (3.6). In order to introduce the mixture velocity u into the flux function, we rewrite the flux (3.6) in terms of state variables which reads

$$\hat{f}(W) = \begin{pmatrix} \rho u \\ \alpha_1 \rho u \\ \alpha_1 \rho_1 u + \alpha_1 \varrho_1 \rho_1 \left( 1 - \frac{\alpha_1 \varrho_1 \rho_1}{\rho} \right) (u_1 - u_2) \\ \rho u^2 + \frac{\alpha_1 \varrho_1 \hat{p}_1}{M_1^2} + \frac{\alpha_2 \hat{p}_2}{M_2^2} + \alpha_1 \varrho_1 \rho_1 \left( 1 - \frac{\alpha_1 \varrho_1 \rho_1}{\rho} \right) (u_1 - u_2)^2 \\ u(u_1 - u_2) + \left( \frac{\hat{h}_1}{M_1^2} - \frac{\hat{h}_2}{M_2^2} \right) + \left( 1 - 2 \frac{\alpha_1 \varrho_1 \rho_1}{\rho} \right) \frac{(u_1 - u_2)^2}{2} \end{pmatrix}.$$
(3.13)

Considering that the mass flux was already treated in the previous step, the remaining transport terms are given by

$$\partial_{t}\rho = 0,$$

$$\partial_{t}(\alpha\rho) + \partial_{x}(\alpha\rho u) = 0,$$

$$\partial_{t}(\alpha\rho_{1}) + \partial_{x}(\alpha\rho_{1}u) = 0,$$

$$\partial_{t}(\rho u) + \partial_{x}(\rho u^{2}) = 0,$$

$$\partial_{t}(u_{1} - u_{2}) + \partial_{x}\left(u(u_{1} - u_{2}) + \left(1 - 2\frac{\alpha_{1}\rho_{1}\rho_{1}}{\rho}\right)\frac{(u_{1} - u_{2})^{2}}{2}\right) = 0.$$

$$(3.14)$$

This system is hyperbolic with the following eigenvalues

$$\lambda_1 = 0, \quad \lambda_{2,3} = u, \quad \lambda_4 = 2u, \quad \lambda_5 = u_1 + u_2 - u.$$
 (3.15)

The associated eigenvectors read

$$\mathbf{v}_{1} = \begin{pmatrix} 1\\ \frac{\alpha}{2}\\ \frac{u}{2}\\ \frac{\chi_{1}}{2}\\ \frac{\varepsilon_{1}}{2} \end{pmatrix}, \quad \mathbf{v}_{2} = \begin{pmatrix} 0\\1\\0\\0\\0 \end{pmatrix}, \quad \mathbf{v}_{3} = \begin{pmatrix} 0\\0\\0\\1\\\varepsilon_{2} \end{pmatrix}, \quad \mathbf{v}_{4} = \begin{pmatrix} 0\\1\\\frac{\chi_{1}}{\alpha_{1}}\\\frac{u}{\alpha_{1}}\\\frac{\varepsilon_{3}}{\alpha_{1}} \end{pmatrix}, \quad \mathbf{v}_{5} = \begin{pmatrix} 0\\0\\0\\0\\1 \end{pmatrix}$$

with

$$\varepsilon_{1} = -\frac{(u_{1} - u_{2})(\chi_{1}(u_{1} - u_{2}) - u)}{-(1 - 2\chi_{1})\rho(u_{1} - u_{2}) + \rho u}, \quad \varepsilon_{2} = \frac{(u_{1} - u_{2})}{(1 - 2\chi_{1})\rho}, \quad \varepsilon_{3} = \frac{(u_{1} - u_{2})(\chi_{1}(u_{1} - u_{2}) - u)}{-(1 - 2\chi_{1})\rho(u_{1} - u_{2}) - \rho u}.$$
(3.16)

Note that the double eigenvalue u has two linearly independent eigenvectors and all eigenvectors are linearly independent. Since all waves exhibited by system (3.14) are of the order of the material velocity, we discretize the equations (3.14) explicitly. Let  $W^{\star\star}$  denote the new state after the advection step. Then the time discretization of (3.14) is given by

$$\rho^{\star\star} = \rho^{\star},$$

$$(\alpha\rho)^{\star\star} = (\alpha\rho)^{n} - \Delta t \partial_{x} \left( \frac{(\alpha\rho)^{n}(\rho u)^{\star}}{\rho^{\star}} \right),$$

$$(\alpha\rho_{1})^{\star\star} = (\alpha\rho_{1})^{n} - \Delta t \partial_{x} \left( \frac{(\alpha\rho_{1})^{n}(\rho u)^{\star}}{\rho^{\star}} \right),$$

$$(\rho u)^{\star\star} = (\rho u)^{\star} - \Delta t \partial_{x} \left( \frac{(\rho u)^{\star}(\rho u)^{\star}}{\rho^{\star}} \right),$$

$$(u_{1} - u_{2})^{\star\star} = (u_{1} - u_{2})^{n} - \Delta t \partial_{x} \left( \frac{(\rho u)^{\star}(u_{1} - u_{2})^{n}}{\rho^{\star}} + \left( 1 - 2\rho_{1} \frac{(\alpha\rho_{1})^{n}}{\rho^{\star}} \right) \frac{((u_{1} - u_{2})^{n})^{2}}{2} \right).$$

$$(3.17)$$

Therein we have used the values at time  $t^n$  for the state variables  $(\alpha \rho)$ ,  $(\alpha_1 \rho_1)$  and  $u_1 - u_2$  since they were not evolved in the acoustic stage (3.7). Due to the explicit time integration, the following CFL condition

$$\Delta t \le \nu_u \frac{\Delta x}{\max(2|u|, |u_1 + u_2 - u|)}$$
 (3.18)

has to be met which is independent of the Mach numbers and follows the material wave.

## 3.2.3 Stiff mixture terms

Collecting the remaining terms in  $\hat{f}$ , that were not yet treated, results in the following subsystem

$$\partial_t(\alpha \rho_1) + \partial_x \left( \alpha \varrho_1 \rho_1 \left( 1 - \frac{\alpha_1 \varrho_1 \rho_1}{\rho} \right) (u_1 - u_2) \right) = 0, \tag{3.19a}$$

$$\partial_t(\rho u) + \partial_x \left(\alpha \varrho_1 \rho_1 \left(1 - \frac{\alpha_1 \varrho_1 \rho_1}{\rho}\right) (u_1 - u_2)^2\right) = 0, \tag{3.19b}$$

$$\partial_t(u_1 - u_2) + \partial_x \left(\frac{\hat{h}_1}{M_1^2} - \frac{\hat{h}_2}{M_2^2}\right) = 0.$$
 (3.19c)

This system is strictly hyperbolic with the eigenvalues

$$\lambda^{0} = 0, \quad \lambda^{\pm} = \frac{\varrho_{1}}{2} (1 - 2\chi_{1}) (u_{1} - u_{2}) \pm \frac{1}{2} \sqrt{\mathcal{A}}$$
 (3.20)

with

$$\mathcal{A} = \varrho_1^2 (1 - 2\chi_1)^2 (u_1 - u_2)^2 + 4\rho \chi (1 - \chi) \left( \frac{1}{\alpha_1 M_1^2} \frac{(c_1^{RS})^2}{\rho_1^{RS}} + \frac{1}{\alpha_2 M_2^2} \frac{(c_2^{RS})^2}{\rho_2^{RS}} \right).$$

Since  $\mathcal{A}$  is positive, the eigenvalues are real. The associated eigenvectors are given by

$$\mathbf{v}^{0} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \mathbf{v}^{\pm} = \begin{pmatrix} \frac{1}{2\rho\mathcal{C}^{RS}} \left( \varrho_{1}(u_{1} - u_{2})\alpha_{1}\alpha_{2}\rho_{1}^{RS}\rho_{2}^{RS}(1 - 2\chi) \pm \sqrt{\mathcal{A}} \right) \\ -\frac{w}{\rho\mathcal{C}^{RS}} \left( (\varrho_{1} - \varrho_{2})(u_{1} - u_{2})\alpha_{1}\alpha_{2}\rho_{1}^{RS}\rho_{2}^{RS}(1 - 2\chi) \pm \sqrt{\mathcal{A}} \right) \\ 1 \end{pmatrix},$$

where

$$\mathcal{C}^{RS} = \frac{\alpha \rho_1^{RS}}{\rho} (c_1^{RS})^2 + \frac{\alpha_2 \rho_2^{RS}}{\rho} (c_2^{RS})^2.$$

From the eigenstructure of (3.19) we can deduce that the relative velocity equation (3.19c) is coupled with the partial density equation (3.19a) via the enthalpy terms  $\hat{h}_1, \hat{h}_2$  resulting in the Mach number dependent eigenvalues  $\lambda^{\pm}$ . Therefore, the enthalpy terms must be treated implicitly to obtain an overall Mach number independent CFL restriction. The momentum equation (3.19b) is decoupled from (3.19a) and (3.19c) and can be updated directly once the partial density  $\alpha_1 \rho_1$  and relative velocity  $u_1 - u_2$  are obtained. This does not yield a restriction on the time step since  $\rho u$  is associated with a zero eigenvalue. Therefore we consider the subsystem consisting of (3.19a) and (3.19c) implicitly and obtain with the backward Euler scheme, starting from stage  $W^{\star\star}$ , the following time discretization

$$(\alpha \rho_1)^{\star\star\star} = (\alpha \rho_1)^{\star\star} - \Delta t \partial_x \left( \varrho_1 (\alpha \rho_1)^{\star\star\star} \left( 1 - \varrho_1 \frac{(\alpha \rho_1)^{\star\star\star}}{\rho^{\star\star}} \right) (u_1 - u_2)^{\star\star\star} \right), \tag{3.21}$$

$$(u_1 - u_2)^{***} = (u_1 - u_2)^{**} - \Delta t \partial_x \left( \frac{(c_1^{RS})^2}{\rho_1^{RS} M_1^2} \frac{(\alpha \rho_1)^{***}}{\alpha^{**}} - \frac{(c_2^{RS})^2}{\rho_2^{RS} M_2^2} \frac{\rho^{**} - \varrho_1(\alpha \rho_1)^{***}}{1 - \alpha^{**}} \right).$$
(3.22)

In (3.22) we have already inserted the definitions of  $\hat{h}_1$ ,  $\hat{h}_2$  given in (3.3). Substituting  $(u_1 - u_2)^{\star\star\star}$  in (3.21) by the relation given in (3.22), we obtain

$$(\alpha\rho_1)^{\star\star\star} - (\alpha\rho_1)^{\star\star\star} + \Delta t \partial_x \left( \varrho_1(\alpha\rho_1)^{\star\star\star\star} \left( 1 - \frac{\varrho_1(\alpha\rho_1)^{\star\star\star\star}}{\rho^{\star\star\star}} \right) (u_1 - u_2)^{\star\star\star} \right)$$

$$- \Delta t^2 \partial_x \left\{ \varrho_1(\alpha\rho_1)^{\star\star\star\star} \left( 1 - \frac{\varrho_1(\alpha\rho_1)^{\star\star\star\star}}{\rho^{\star\star}} \right) \times$$

$$\partial_x \left( \left( \frac{(c_1^{RS})^2}{\rho_1^{RS} M_1^2} \frac{1}{\alpha_1^{\star\star\star}} + \frac{(c_2^{RS})^2}{\rho_2^{RS} M_2^2} \frac{1}{\alpha_2^{\star\star}} \right) (\alpha\rho_1)^{\star\star\star} - \frac{(c_2^{RS})^2}{\rho_2^{RS} M_2^2} \frac{\rho^{\star\star\star}}{\alpha_2^{\star\star}} \right) \right\} = 0.$$

$$(3.23)$$

It is a nonlinear elliptic problem in  $(\alpha \rho_1)^{\star\star\star}$  that can be solved with a linearised iterative scheme

as follows

$$(\alpha \rho_{1})^{l+1} - (\alpha \rho_{1})^{**} + \Delta t \partial_{x} \left( \varrho_{1}(\alpha \rho_{1})^{l} \left( 1 - \frac{\varrho_{1}(\alpha \rho_{1})^{l}}{\rho^{**}} \right) (u_{1} - u_{2})^{**} \right)$$

$$- \Delta t^{2} \partial_{x} \left\{ \varrho_{1}(\alpha \rho_{1})^{l} \left( 1 - \frac{\varrho_{1}(\alpha \rho_{1})^{l}}{\rho^{*}} \right) \times$$

$$\partial_{x} \left( \left( \frac{(c_{1}^{RS})^{2}}{\rho_{1}^{RS} M_{1}^{2}} \frac{1}{\alpha_{1}^{**}} + \frac{(c_{2}^{RS})^{2}}{\rho_{2}^{RS} M_{2}^{2}} \frac{1}{\alpha_{2}^{**}} \right) (\alpha \rho_{1})^{l+1} - \frac{(c_{2}^{RS})^{2}}{\rho_{2}^{RS} M_{2}^{2}} \frac{\rho^{**}}{\alpha_{2}^{**}} \right) \right\} = 0,$$

$$(3.24)$$

where  $l \in \mathbb{N}_0$  denotes the number of iteration and we set the start value as  $(\alpha \rho_1)^0 = (\alpha \rho_1)^{\star\star}$ . When the stopping criterion given by the relative  $L^1$  error

$$\frac{\|\alpha\rho_1^{l+1} - (\alpha\rho_1)^l\|_{L^1}}{\|(\alpha\rho_1)^l\|_{L^1}} < \delta,$$

for a given  $\delta > 0$  is fulfilled, then we set  $(\alpha \rho_1)^{\star\star\star} = (\alpha \rho_1)^{l+1}$ . We note that after applying a suitable space discretization, the coefficient matrix for solving  $(\alpha \rho_1)^{l+1}$  is symmetric and positive definite, since the terms

$$s(W) = \varrho_1(\alpha \rho_1)^l \left( 1 - \frac{\varrho_1(\alpha \rho_1)^l}{\rho^*} \right) \text{ and } \tilde{s}(W) = \frac{(c_1^{RS})^2}{\rho_1^{RS} M_1^2} \frac{1}{\alpha_1^{\star \star}} + \frac{(c_2^{RS})^2}{\rho_2^{RS} M_2^2} \frac{1}{\alpha_2^{\star \star}}$$
(3.25)

are positive. The linear system is therefore well-posed and can be solved efficiently with standard linear solvers. The relative velocity is then updated explicitly according to (3.22) and the momentum by

$$(\rho u)^{\star\star\star} = (\rho u)^{\star\star} - \Delta t \partial_x (\rho^{\star\star} \chi_1^{\star\star\star} \chi_2^{\star\star\star} ((u_1 - u_2)^{\star\star\star})^2).$$

The updated state vector after the third step of the time semi-discrete scheme is then given by  $W^{\star\star\star} = (\rho^{\star\star}, (\alpha\rho)^{\star\star}, (\alpha_1\rho_1)^{\star\star\star}, (\rho u)^{\star\star\star}, (u_1 - u_2)^{\star\star\star})^T$ .

## 3.2.4 Treatment of higher order pressure and enthalpy terms

Till now we have considered the truncated pressure and enthalpy terms (3.2), (3.3). This is sufficient to construct a scheme that is applicable on low Mach number flows only. To obtain accurate results also for compressible flow regimes, we have to take into account the higher order terms in the Taylor expansions (3.4), (3.5). In those regimes,  $\bar{p}_k$ ,  $\bar{h}_k$  are of order  $\mathcal{O}(1)$  and neglecting them results in large errors and spurious results. As these higher order terms vanish for well-prepared initial data in the low Mach number limit, they can be treated explicitly without generating a dependence of the Mach numbers on the CFL condition. Moreover  $\bar{p}$  and  $\bar{h}$  are nonlinear and treating them explicitly does not add to the complexity of the scheme. Nevertheless, they have to be added carefully to the existing low Mach number scheme to ensure the hyperbolic structure of the subsystems (3.7), (3.14) and (3.19). We add them in the third step of the time semi-discrete scheme, which is then given by

- 1. Calculate  $(\alpha \rho_1)^{\star\star\star}$  by solving (3.24) iteratively.
- 2. Update  $u_1 u_2$  using the untruncated enthalpies  $h_k = \hat{h}_k + \bar{h}_k$  (k = 1, 2) by

$$(u_1 - u_2)^{***} = (u_1 - u_2)^{**} - \Delta t \partial_x \left( \frac{h_1(\rho_1^{***})}{M_1^2} - \frac{h_2(\rho_2^{***})}{M_2^2} \right), \tag{3.26}$$

where  $\rho_1$  and  $\rho_2$  are calculated by relation (3.9).

3. Update  $\rho u$  including  $\bar{p}_k$  terms (k=1,2) by

$$(\rho u)^{\star\star\star} = (\rho u)^{\star\star} - \Delta t \partial_x \left( (\alpha \rho_1)^{\star\star\star} (1 - \frac{(\alpha \rho_1)^{\star\star\star}}{\rho^{\star}}) ((u_1 - u_2)^{\star\star\star})^2 \right) - \Delta t \partial_x \left( \alpha^{\star\star} \frac{\bar{p}_1 \left( \rho_1^{\star\star\star} \right)}{M_1^2} + (1 - \alpha^{\star\star}) \frac{\bar{p}_2 \left( \rho_2^{\star\star\star} \right)}{M_2^2} \right). \quad (3.27)$$

This results in a correction of the momentum and relative velocity for compressible flow regimes leading to an all-speed scheme for the simulation of two phase flows.

#### 3.2.5 Treatment of the relaxation source terms

After having established the main scheme for the homogeneous case, we focus now on the numerical approximation of the relaxation source terms acting on the volume fraction and relative velocity. They are given by

$$\partial_t \rho \alpha = -\frac{1}{\tau} \left( \frac{\varrho_2}{M_2^2} \ p_2 \left( \rho_2 \right) - \frac{\varrho_1}{M_1^2} \ p_1(\rho_1) \right), \tag{3.28}$$

$$\partial_t(u_1 - u_2) = -\zeta \frac{\varrho_1 \alpha \rho_1}{\rho} \left( 1 - \frac{\varrho_1 \alpha \rho_1}{\rho} \right) (u_1 - u_2). \tag{3.29}$$

We first notice that both equations are decoupled and can be solved simultaneously. Since the remaining variables are not affected by the relaxation processes, the state vector at time  $t^{n+1}$  is given by  $W^{n+1} = (\rho^{\star\star\star}, \alpha^{n+1}\rho^{\star\star\star}, (\alpha\rho_1)^{\star\star\star}, (\rho u)^{\star\star\star}, (u_1 - u_2)^{n+1})^T$ . Since the friction parameter  $\zeta > 0$  can be large, equation (3.29) is integrated implicitly. Due to the linearity of the source term, we can find an analytic update of the relative velocity given by

$$\left(1 + \Delta t \zeta \frac{\varrho_1(\alpha \rho_1)^{n+1}}{\rho^{n+1}} \left(1 - \frac{\varrho_1(\alpha \rho_1)^{n+1}}{\rho^{n+1}}\right)\right) (u_1 - u_2)^{n+1} = (u_1 - u_2)^{\star \star \star}.$$
(3.30)

The pressure relaxation rate is a non-negative parameter  $\tau \in [0, \infty)$ , where  $\tau = 0$  gives an instantaneous pressure relaxation. The homogeneous equation corresponds to " $\tau = \infty$ ". For fast relaxation rates with  $\tau \ll 1$ , equation (3.29) becomes stiff and is discretized implicitly. Since we are free to choose the set of variables, we rewrite (3.29) in terms of mixture variables Q which is consistent with model (2.1). We obtain the following implicit equation

$$\alpha^{n+1} = \alpha^{\star\star\star} - \frac{\Delta t}{\tau \rho^{n+1}} \left( \frac{\varrho_2}{M_2^2} \ p_2 \left( \frac{1 - \chi^{n+1}}{1 - \alpha^{n+1}} \rho^{n+1} \right) - \frac{\varrho_1}{M_1^2} \ p_1 \left( \frac{\chi^{n+1}}{\alpha^{n+1}} \rho^{n+1} \right) \right). \tag{3.31}$$

Since the phase pressures  $p_1, p_2$  are nonlinear in  $\alpha$  we use the Newton method applied on  $g(\alpha) = 0$  to obtain  $\alpha^{n+1}$  with

$$\begin{split} g(\alpha) &= -\alpha - \frac{\Delta t}{\tau \rho^{n+1}} \left( \frac{\varrho_2}{M_2^2} p_2(\alpha) - \frac{\varrho_1}{M_1^2} p_1(\alpha) \right) + \alpha^{\star\star\star}, \\ \frac{\partial g}{\partial \alpha} &= -1 - \frac{\Delta t}{\tau} \left( \frac{\varrho_2}{M_2^2} c_2(\alpha)^2 \frac{1 - \chi^{n+1}}{1 - \alpha} + \frac{\varrho_1}{M_1^2} c_1(\alpha)^2 \frac{\chi^{n+1}}{\alpha} \right), \end{split}$$

where the sound speeds  $c_1$ ,  $c_2$  also depend on  $\alpha$ . The focus of this work is on the construction of an asymptotic preserving scheme that is consistent with the singular Mach number limits. For a different approach on how to treat the relaxation source terms which was used in the context of the Baer-Nunziato model we refer the interested reader to [6].

To summarize, we have split the non-dimensional model (2.11) in the following way

$$\partial_t W + \partial_x f^a(W) + \partial_x f^b(W) + \partial_x f^c(W) = -r(W),$$

where the fluxes of the subsystems (3.7), (3.14), (3.19) are given by

$$f^{a}(W) = \begin{pmatrix} \rho u \\ 0 \\ 0 \\ \frac{\alpha \hat{\rho}_{1}}{M_{1}^{2}} + \frac{(1-\alpha)\hat{\rho}_{2}}{M_{2}^{2}} \end{pmatrix}, \quad f^{b}(W) = \begin{pmatrix} 0 \\ \alpha \rho u \\ \alpha \rho_{1} u \\ \rho u^{2} \\ \frac{1}{2}(u_{1}^{2} - u_{2}^{2}) \end{pmatrix},$$

$$f^{c}(W) = \begin{pmatrix} 0 \\ 0 \\ \alpha \rho_{1} \left(1 - \frac{\alpha \rho_{1}}{\rho}\right) (u_{1} - u_{2}) \\ \alpha \rho_{1} \left(1 - \frac{\alpha \rho_{1}}{\rho}\right) (u_{1} - u_{2})^{2} + \frac{\alpha \bar{\rho}_{1}}{M_{1}^{2}} + \frac{(1-\alpha)\bar{p}_{2}}{M_{2}^{2}} \\ \frac{h_{1}}{M^{2}} - \frac{h_{2}}{M^{2}} \end{pmatrix}$$

and the relaxation source term reads

$$r(W) = \begin{pmatrix} 0 \\ \frac{1}{\tau} \left( \frac{p_2}{M_2^2} - \frac{p_1}{M_1^2} \right) \\ 0 \\ \zeta \chi \left( 1 - \chi \right) \left( u_1 - u_2 \right) \end{pmatrix}.$$

The time semi-discrete scheme is then given by the following operator splitting

$$W^* = W^n - \Delta t \, \partial_x f^a \left( W^* \right), \tag{3.32a}$$

$$W^{\star\star} = W^{\star} - \Delta t \, \partial_x f^b \left( W^{\star} \right), \tag{3.32b}$$

$$W^{\star\star\star} = W^{\star\star} - \Delta t \, \partial_x f^c \left( W^{\star\star\star} \right), \tag{3.32c}$$

$$W^{n+1} = W^{\star\star\star} - \Delta t \ r \left( W^{n+1} \right). \tag{3.32d}$$

#### 3.3 Fully discrete scheme

In this section, we describe the fully discrete numerical scheme associated with the time semidiscrete stages derived in the previous section. In time, we set as before  $t^{n+1} = t^n + \Delta t$ , where  $\Delta t$  obeys a time step restriction given by the CFL condition (3.18). In space, we consider a computational domain  $\Omega$  divided into cells  $C_i = (x_{i-1/2}, x_{i+1/2})$  of uniform step size  $\Delta x$  with the cell center  $x_i = i\Delta x$  for i = 1, ..., N. We use a finite volume framework, where the solution on cell  $C_i$  at time  $t^n$  is approximated by the average given by

$$W_i^n \approx \frac{1}{\Delta x} \int_{\Omega_i} w(x, t^n) dx.$$

For the explicit advective step (3.32b), we apply a standard finite volume scheme using the Rusanov flux. The update on cell  $C_i$  is given by

$$W_i^{\star\star} = W_i^{\star} - \frac{\Delta t}{\Delta x} \left( F^a(W_i^{\star}, W_{i+1}^{\star}) - F^a(W_{i-1}^{\star}, W_i^{\star}) \right),$$

where the numerical flux function F a (W<sup>i</sup> , Wi+1) is given by

$$F^{a}(W_{i}, W_{i+1}) = \frac{1}{2} (f^{a}(W_{i}) + f^{a}(W_{i+1}) - \frac{1}{2} a_{i+1/2} (W_{i+1} - W_{i})$$

with ai+1/<sup>2</sup> = maxk=1,...,5(|λk(W<sup>i</sup> , Wi+1)|). The eigenvalues λ<sup>k</sup> are given in [\(3.15\)](#page-13-5) by the advection step and are of the order of the material wave. Since this step is the only explicit part of the numerical scheme, the CFL condition is given by [\(3.18\)](#page-13-4). For the implicit steps [\(3.32a\)](#page-17-2) and [\(3.32c\)](#page-17-3) we apply centered differences for the space derivatives. Thereby, the mixed derivatives [\(3.25\)](#page-15-1) in [\(3.24\)](#page-15-0) are discretized by

$$\partial_x \left( s(W) \ \partial_x \left( \tilde{s}(W)W \right) \right) \cong \frac{1}{\Delta x^2} \left( s_{i+1/2} \tilde{s}(W_{i+1}) W_{i+1} - \left( s_{i+1/2} + s_{i-1/2} \right) \tilde{s}(W_i) W_i + s_{i-1/2} \tilde{s}(W_{i-1}) W_{i-1} \right),$$

$$s_{i+1/2} \cong \frac{1}{2} \left( s(W_i) + s(W_{i+1}) \right).$$

Note that by construction, the diffusion of the all-speed scheme is independent of the Mach number due to the use of centred differences on the stiff pressure and enthalpy terms and Mach number independent eigenvalues in the transport step. The scheme is therefore well suited to simulate flows in the low Mach number regime.

# 4 AP property

As we have seen in Section [2.3,](#page-5-3) for well-prepared initial data, the continuous compressible model converges formally towards incompressible equations when the Mach number tends to zero. For the all-speed RS-IMEX scheme we show the discrete analogue, i.e. the numerical scheme is a consistent discretization of the limit model in the low Mach number limit. This property is called asymptotic preserving (AP) and it is essential for ensuring the correct behaviour of the numerical solution in low Mach number regimes.

We will show this property for the time semi-discrete scheme. Indeed, to obtain a consistent discretization with the limit equations an appropriate time discretization is essential. Thereby we use techniques that are used in the context of proving the AP property of IMEX schemes for (isentropic) Euler equations, see e.g. [\[7,](#page-31-10) [9\]](#page-31-11). First, we consider the case of a compressible and a weakly compressible phase (Case 1 in Section [2.3\)](#page-5-3), then the case of two weakly compressible phases in the same low Mach number regime, i.e. M<sup>1</sup> = M<sup>2</sup> = M 1 (Case 2 in Section [2.3\)](#page-5-3). As done in the derivation of the well-prepared data, we neglect the scaling parameters %1, %2.

Case 1 Let the data at time t <sup>n</sup> be well-prepared, i.e. W<sup>n</sup> ∈ Ω wp 1 given in [\(2.23\)](#page-7-5) with constant reference state ρ (0) <sup>2</sup> = ρ RS 2 and α (0) constant. The Mach number expansions for the variables of the second phase is then given by

$$\alpha^{n} = \alpha^{(0)} + \mathcal{O}(M), \quad \alpha^{(0)} \text{ const.}$$

$$\rho_{2}^{n} = \rho_{2}^{RS} + M^{2} \rho_{2}^{(2),n} + \mathcal{O}(M^{3}),$$

$$u_{2}^{n} = u_{2}^{(0),n} + \mathcal{O}(M) \text{ with } \partial_{x} u_{2}^{(0),n} = 0$$

$$p_{2}^{RS} = 0, \quad p_{2}^{(2)} = p_{1}.$$

$$(4.1)$$

In particular, we find

$$\chi^{n} = \frac{\alpha^{(0)}\rho_{1}^{RS}}{\rho^{(0)}} + M \frac{\alpha^{(1)}\rho_{1}^{RS}\rho^{(0)} - \alpha^{(0)}\rho_{1}^{RS}\rho^{(1)}}{\rho^{(0)^{2}}} + \mathcal{O}(M^{2}) = \chi^{(0),n} + M\chi^{(1),n} + \mathcal{O}(M^{2}).$$

We assume for all sub-steps of the numerical scheme that we have the following Mach number expansions of phase two

$$\alpha^{\star} = \alpha^{(0),\star} + \mathcal{O}(M),$$

$$\rho_2^{\star} = \rho_2^{(0),\star} + M\rho_2^{(1),\star} + M^2\rho_2^{(2),\star} + \mathcal{O}(M^3),$$

$$u_2^{\star} = u_2^{(0),\star} + \mathcal{O}(M).$$

An analogue notation holds for the sub-steps  $(\cdot)^{**}$ ,  $(\cdot)^{***}$  and the final update  $(\cdot)^{n+1}$ . Considering the update of the density  $\rho^*$  given in (3.7a), we can rewrite it with  $p_2^{RS} = 0$  as

$$\rho^{\star} - \Delta t^{2} \partial_{x}^{2} \left( \frac{(c_{2}^{RS})^{2}}{M^{2}} \chi_{2}^{n} \rho^{\star} \right) = \rho^{n} - \Delta t \partial_{x} (\rho u)^{n} + \Delta t^{2} \partial_{x}^{2} \left( \alpha^{n} \hat{p}_{1}^{\star} - \frac{(c_{2}^{RS})^{2}}{M^{2}} \alpha_{2}^{n} \rho_{2}^{RS} \right). \tag{4.2}$$

Inserting the Mach number expansions (4.1) and sorting by order of the Mach numbers, we obtain for the  $\mathcal{O}(M^{-2})$  terms

$$\partial_x^2(\chi_2^{(0),n}\rho^{(0),\star}) = 0 \Leftrightarrow \partial_x^2\left(\frac{\chi_2^{(0),n}}{\alpha^{(0),n}}\rho^{(0),\star}\right) = 0 \Leftrightarrow \partial_x^2\rho_2^{(0),\star} = 0.$$

The last equivalence holds since we solve (4.2) in mixture variables  $\rho, \alpha, \chi$  and  $\alpha^* = \alpha^n$ ,  $\chi^* = \chi^n$ . For periodic or no-flux boundary conditions we formally obtain that  $\rho_2^{(0),*}$  is constant. For the  $\mathcal{O}(M^{-1})$  terms, we find

$$\partial_x^2 \left( \frac{\rho_2^{RS}}{\rho^{(0),n}} \rho^{(1),\star} - \frac{\rho_2^{RS} \rho^{(1),n}}{\rho^{(0),n^2}} \rho^{(0),\star} \right) = 0 \Leftrightarrow \partial_x^2 \rho_2^{(1),\star} = 0.$$

This implies formally  $\rho_2^{(1),\star}$  is constant for the above given boundary conditions. Integrating the zero order terms of the density update

$$\rho^{(0),\star} - \rho^{(0),n} + \Delta t \partial_x (\rho^{(0),\star} u^{(0),\star}) = 0$$
(4.3)

on the computational domain, we obtain  $\rho^{(0),\star} = \rho^{(0),n}$  and analogously for the first order terms  $\rho^{(1),\star} = \rho^{(1),n}$ . Since  $\alpha$  and  $\chi$  do not change in the acoustic step, we obtain  $\rho_2^{(0),\star} = \rho_2^{RS}$ . Further we obtain with  $(\alpha \rho_1)^{(1),\star} = \alpha^{(1),n} \rho_1^n$  that  $\rho_2^{(1),\star} = 0$ . Regarding the velocity  $u_2^{(0),\star}$ , we obtain

$$\partial_x u_2^{(0),\star} = \partial_x \left( u^{(0),\star} - \chi^{(0),n} (u_1^n - u_2^{(0),n}) \right) = \mathcal{O}(\Delta t).$$

In the transport step, we find after some simple reformulations

$$\alpha^{(0),\star\star} = \alpha^{(0),n} + \Delta t \alpha^{(0),n} \partial_x ((\rho^{(0),\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),n}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),\star}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),\star}) u^{(0),\star}) = \alpha^{(0),n} \partial_x ((\rho^{(0),\star\star} - \rho^{(0),\star}) u^{(0),\star}) = \alpha^$$

from which follows  $\rho_2^{(0),\star\star} = \rho_2^{RS}$ . Analogously to  $u_2^{(0),\star}$ , we also get  $\partial_x u_2^{(0),\star\star} = \mathcal{O}(\Delta t)$ .

In the mixture step, the  $\mathcal{O}(M^{-2})$  terms in (3.23) yield

$$\partial_x \left( (\alpha \rho_1)^{\star\star\star} \left( 1 - \frac{(\alpha \rho_1)^{\star\star\star}}{\rho^{(0),\star\star}} \right) \partial_x \rho_2^{(0),\star\star\star} \right) = 0.$$

Together with  $\partial_x \rho_2^{(0),\star\star\star} = 0$  from the zero order terms of the relative velocity equation (3.22) it follows that  $\rho_2^{(0),\star\star\star}$  is constant. In addition holds

$$\alpha_2^{(0),n} \rho_2^{(0),\star\star\star} = \rho^{(0),\star} - \alpha^{(0),n} \rho_1^{\star\star\star} = \alpha^{(0),n} (\rho_1^{\star} - \rho_1^{\star\star\star}) + \alpha_2^{(0),n} \rho_2^{RS}.$$

Therefore we have  $\rho_2^{(0),\star\star\star} = \rho_2^{RS} + \mathcal{O}(\Delta t)$ . Analogously, we find from the  $\mathcal{O}(M^{-1})$  terms in (3.23) and (3.22) that  $\rho_2^{(1),\star\star\star}$  is constant. Integration of (3.21) on the computational domain leads to

$$(\alpha_2 \rho_2)^{(1),\star\star\star} \alpha_2^{(0),n} \rho_2^{(1),\star\star\star} - \alpha_2^{(1),\star\star\star} \rho_2^{(0),\star\star\star} = \rho^{(1),\star} - \alpha^{(1),\star\star} \rho_1^{\star\star\star}$$

and thus  $\rho_2^{(1),\star\star\star}=\mathcal{O}(\Delta t)$ . Regarding the pressure relaxation source term, we obtain immediately  $p_2^{(0)}=0=p_2^{RS}$  for the  $\mathcal{O}(M^{-2})$  terms and hence  $\alpha^{(0),n+1}=\alpha^{(0),n}$  constant. For the  $\mathcal{O}(M^{-1})$  terms we obtain using the pressure expansion (2.13) that  $\rho_2^{(1),n+1}=0$ . Consequently,  $\alpha^{(1),n+1}=\alpha^{(1),\star\star}$ . Since  $\alpha^{(0),n+1}$  is constant, we find from the zero order terms  $p_2^{(2)}=p_1$  which gives the pressure constraint in the well-prepared data.

From the density update and the final update of  $\alpha \rho_1$ , we obtain the final update of  $\rho_2 \alpha_2$  and thus  $\partial_x u^{(0),n+1} = \mathcal{O}(\Delta t)$ . Summarizing, the data  $W^{n+1}$  preserves the asymptotics for  $\Delta t \to 0$ . We have proved the following theorem.

**Theorem 4.1** (AP property compressible/weakly compressible). For well-prepared initial data  $W^n \in \Omega_1$  and periodic or no-flux boundary conditions, the time semi discrete scheme described in Section (3.2) is asymptotic preserving and a consistent time discretization of the limit equations (2.24) when  $M \to 0$ .

Case 2 We consider the case of two weakly compressible phases in the same Mach number regime with constant reference states  $\rho_1^{(0),n} = \rho_1^{RS}$  and  $\rho_2^{(0)} = \rho_2^{RS}$  and  $\alpha^{(0),n}$  constant. According to (2.32), the well-prepared data at time  $t^n$  is then given by

$$\alpha^{n}(x) = \alpha^{(0)} + \mathcal{O}(M), \quad \alpha^{(0)} \text{ const.},$$

$$\rho_{1}^{n}(x) = \rho_{1}^{RS} + M^{2}\rho_{1}^{(2)} + \mathcal{O}(M^{3}), \quad \rho_{2}^{n}(x) = \rho_{2}^{RS} + M^{2}\rho_{2}^{(2)} + \mathcal{O}(M^{3}),$$

$$u_{1}^{n}(x) = u_{1}^{(0)} + \mathcal{O}(M), \quad u_{2}^{n}(x) = u_{2}^{(0)} + \mathcal{O}(M), \quad \partial_{x}u_{1}^{(0)} = 0, \quad \partial_{x}u_{2}^{(0)} = 0,$$

$$p_{1}^{RS} = p_{2}^{RS}, \quad p^{(2)} = p_{1}^{(2)} = p_{2}^{(2)}.$$

$$(4.4)$$

Analogously to Case 1, we consider a general Mach number expansion for the state variables of the respective steps of the numerical scheme. For the first step, we define

$$W^*(x) = W^{(0),*} + MW^{(1),*} + \mathcal{O}(M^2).$$

Using well-prepared data (4.4), we can rewrite the update for  $\rho^*$  as

$$\rho^{\star} - \Delta t^2 \partial_x^2 \left( \left( c^{RS} \left( \frac{\alpha^n \rho_1^n}{\rho^n} \right) \right)^2 \rho^{\star} \right) = \rho^n - \Delta t \partial_x (\rho u) - \Delta t^2 \partial_x^2 \left( \left( c^{RS} \left( \frac{\alpha^n \rho_1^{RS}}{\rho^n} \right) \right)^2 \rho^n \right),$$

where  $c^{RS}$  is defined by (3.8) and depends on 1/M. Inserting the Mach number expansions and sorting by terms of the Mach number, we find for the  $\mathcal{O}(M^{-2})$  terms

$$\partial_x^2 \rho^{(0),\star} = \partial_x^2 \rho^{(0),n} = 0.$$

This implies formally that  $\rho^{(0),\star}$  is constant for periodic or no-flux boundary conditions. Integrating the zero order density terms (4.3) over the computational domain, we get  $\rho^{(0),\star} = \rho^{(0),n}$ . Further, since we solve the acoustic step in the variables  $(\rho, \alpha, \chi, \rho u, u_1 - u_2)$ , we have  $\alpha^{\star} = \alpha^n$  and therefore  $\alpha^{(0),\star} = \alpha^{(0),n}$  constant,  $\alpha^{(1),\star} = \alpha^{(1),n}$ . Since  $\alpha^{(0)}$  constant and positive, it follows  $\rho_1^{(0),\star} = \rho_1^{RS}$  and  $\rho_2^{(0),\star} = \rho_2^{RS}$  which implies  $\partial_x u_1^{(0),\star} = 0$ . Since  $\chi^{(0),\star}$  is constant it follows  $\partial_x u_1^{(0),\star}, \partial_x u_2^{(0),\star} = 0$ . For the  $\mathcal{O}(M^{-1})$  terms we find

$$\partial_x^2 \rho^{(1),\star} = \partial_x^2 \rho^{(1),n} \Leftrightarrow \partial_x^2 (\alpha^{(0),n} \rho_1^{(1),\star} + (1 - \alpha^{(0),n}) \rho_2^{(1),\star}) = 0.$$

Thus, formally  $\rho_1^{(1),\star}$ ,  $\rho_2^{(1),\star}$  are constant. Integrating the analogue equation to (4.3) for the first order density perturbation  $\rho^{(1),\star}$ , we find  $\alpha^{(0),n}\rho_1^{(1),\star} + \alpha_2^{(0),n}\rho_2^{(1),\star} = 0$  hence  $\rho_1^{(1),\star}$ ,  $\rho_2^{(1),\star} = 0$ . From the momentum update, we obtain then

$$u^{(0),\star} = u^{(0),n} - \Delta t \frac{\partial_x (\alpha^{(0),n} \hat{p}_1^{(2),\star} + \alpha_2^{(0),n} \hat{p}_2^{(2),\star})}{\rho^{(0),n}}.$$

Hence  $W^*$  preserves the asymptotics for  $\Delta t \to 0$ . We assume that the update after the transport step obeys  $W^{**} = W^{(0),**} + MW^{(1),**} + \mathcal{O}(M^2)$ . Then step (3.17) with  $\partial_x u^{(0),*} = 0$  for the  $W^{(0),**}$  terms leads to the following relations

$$\rho^{(0),\star\star} = \rho^{(0),\star} = \rho^{(0),n},$$

$$\alpha^{(0),\star\star} = \alpha^{(0),n},$$

$$\rho_1^{(0),\star\star} = \rho_1^{RS},$$

$$u^{(0),\star\star} = u^{(0),\star},$$

$$(u_1 - u_2)^{(0),\star\star} = (u_1 - u_2)^{(0),n}.$$

In particular, this yields  $\partial_x u_1^{(0),\star\star} = 0 = \partial_x u_2^{(0),\star\star}$ . Considering the  $\mathcal{O}(M)$  terms we obtain

$$\alpha^{(1),\star\star} = \alpha^{(1),n} - \Delta t u^{(0),\star} \partial_x \alpha^{(1),n}$$

which is consistent with the continuous model. Using this relation we obtain

$$\rho_1^{(1),\star\star} = -\Delta t \rho_1^{RS} \partial_x \alpha^{(1),n} = \mathcal{O}(\Delta t).$$

Therefore,  $W^{\star\star}$  after the transport step preserves the asymptotics for  $\Delta t \to 0$ . Assume that the update after the mixture step obeys  $W^{\star\star\star} = W^{(0),\star\star\star} + MW^{(0),\star\star\star} + \mathcal{O}(M^2)$ . Inserting the Mach number expansions of  $W^{\star\star\star}$  and  $W^{\star\star}$  into the update (3.23), we derive for the  $\mathcal{O}(M^{-2})$  terms the following relation

$$\partial_x \left( \rho_1^{(0),\star\star\star} (\rho^{(0),n} - \alpha^{(0),n} \rho_1^{(0),\star\star\star}) \partial_x \rho_1^{(0),\star\star\star} \right) = 0.$$

Further, from the relative velocity update (3.22) it follows  $\partial_x \rho_1^{(0),\star\star\star} = 0$ . Consequently,  $\rho_1^{(0),\star\star\star}$  is constant for periodic or non-flux boundary conditions. Integrating (3.21) on the computational domain yields  $(\alpha \rho_1)^{(0),\star\star\star} = \alpha^{(0),n} \rho_1^{RS}$  and thus  $\rho_1^{(0),\star\star\star} = \rho_1^{RS}$ . Note that  $\alpha$  does not change

in this step and we have  $\alpha^{(0),\star\star\star} = \alpha^{(0),n}$ . An immediate consequence is  $\rho_2^{(0),\star\star\star} = \rho_2^{RS}$  and  $\partial_x (u_1 - u_2)^{(0),\star\star\star} = 0$ . For the  $\mathcal{O}(M^{-1})$  terms we obtain then

$$\partial_x^2 \left( \frac{(c_1^{RS})^2}{\rho_1^{RS}} \rho_1^{(1),\star\star\star} + \frac{(c_2^{RS})^2}{\rho_2^{RS}} \rho_2^{(1),\star\star\star} \right) = 0,$$

for which we formally obtain constant  $\rho_1^{(1),\star\star\star}, \rho_2^{(1),\star\star\star}$ . Considering the first order terms of (3.21) we obtain that they are at least of order  $\mathcal{O}(\Delta t)$ . For the higher order terms  $\bar{p}$  we find that they are of the order  $\bar{p} = \mathcal{O}(\Delta t M^2)$ . Therefore (3.27) yields for the velocity  $u^{(0),\star\star\star} = u^{(0),\star\star} + \mathcal{O}(\Delta t^2)$  and hence its space derivative is zero. Since  $\rho_1^{(1),\star\star\star}, \rho_2^{(1),\star\star\star}$  are constant, the following relation at order  $\mathcal{O}(1)$  is obtained from (3.26)

$$(u_1 - u_2)^{(0),\star\star\star} = (u_1 - u_2)^{(0),n} - \Delta t \partial_x (h_1^{(2),\star\star\star} - h_2^{(2),\star\star\star}).$$

Summarizing,  $W^{\star\star\star}$  preserves the asymptotics for  $M\to 0$ . Considering also the friction relaxation terms in the relative velocity (3.30), we find the following time semi-discrete scheme for the limit equations

$$u^{(0),n+1} = u^{(0),n} - \Delta t \frac{\partial_x (\alpha^{(0),n} \hat{p}_1^{(2),\star} + \alpha_2^{(0),n} \hat{p}_2^{(2),\star})}{\rho^{(0),n}}$$

$$(u_1 - u_2)^{(0),n+1} = (u_1 - u_2)^{(0),n} - \Delta t \partial_x (h_1^{(2),\star\star\star} - h_2^{(2),\star\star\star}) - \zeta \chi^{(0),n+1} \chi_2^{(0),n+1} (u_1 - u_2)^{(0),n+1}$$

$$(4.5)$$

which is consistent with the formulation of the limit equations (2.31) in terms of  $u_1, u_2$ . Note that  $\partial_x p_k^{(2)} = \partial_x \hat{p}_k^{(2)}$  for k = 1, 2 and that adding the friction source term does not influence the zero order velocity derivatives  $\partial_x u_1^{(0),n+1} = 0 = \partial_x u_2^{(0),n+1}$ .

In the pressure relaxation (3.31), we find for the  $\mathcal{O}(M^{-2})$  terms

$$p_2(\rho_2^{(0),n+1}) = p_1(\rho_2^{(0),n+1}).$$

Hence  $\alpha^{(0),n+1}=\alpha^{(0),n}$  since  $\rho^{(0),n+1}$  and  $\chi^{(0),n+1}$  are fulfilling the well-prepared constraint on the pressure  $p_1^{(0)}=p_2^{(0)}$ . For the  $\mathcal{O}(M^{-1})$  terms we find

$$p_2(\rho_2^{(1),n+1}) = p_1(\rho_2^{(1),n+1}) \Leftrightarrow c_1^{(0)}\rho_1^{(1),n+1} = c_2^{(0)}\rho_2^{(1),n+1}$$

where  $c_1^{(0),n+1}$  and  $c_2^{(0),n+1}$  are known from the zero order terms. Using the Mach number expansions for  $\alpha^{(1),n+1}=\alpha^{(1),\star\star\star}$  leads to  $\rho_1^{(1),n+1}=\rho_1^{(0),\star\star\star}=\mathcal{O}(\Delta t)$ . Analogously, one obtains  $\rho_2^{(1),n+1}=\mathcal{O}(\Delta t)$ . Since  $\alpha^{(0),n}$  and  $\alpha^{(0),n+1}$  coincide, we obtain for the  $\mathcal{O}(1)$  terms that

$$p_2\left(\rho_2^{(2),n+1}\right) = p_1\left(\rho_2^{(2),n+1}\right) \Leftrightarrow \hat{p}_2^{(2),n+1} = \hat{p}_1^{(2),n+1}$$

which means at time  $t^{n+1}$  the pressure constraint on the well-prepared data is fulfilled. Summarizing the above discussion,  $W^{n+1}$  preserves the asymptotics for  $\Delta t \to 0$  with the consistent time semi-discrete limit equations (4.5) and we have proved the following theorem.

**Theorem 4.2** (AP property for two weakly compressible phases). For well-prepared initial data  $W^n \in \Omega_M^{wp}$  and periodic or no-flux boundary conditions, the time semi-discrete scheme described in Section (3.2) is asymptotic preserving and a consistent time discretization of the limit equations (2.33) for  $M \to 0$ .

The AP property for the case of two different low Mach number regimes with well-prepared data  $W^n \in \Omega_2^{wp}$  can be obtained following the lines of the proof of the AP property discussed for Case 1 and 2.

## 5 Numerical results

In this section, we illustrate by numerical experiments theoretical properties of the proposed RS-IMEX scheme (3.32). The first set of test cases concerns the homogeneous equations without pressure relaxation and friction terms. The initial conditions, if not mentioned otherwise, are given in non-dimensional form. They can be straightforwardly transformed into dimensional variables following the scaling procedure in Section 2.2. We compare the numerical results obtained with our scheme (4.5) with a reference solution computed by a first order explicit Rusanov scheme. The latter requires an acoustic time stepping given by the following CFL condition

$$\Delta t \le \nu_{ac} \frac{\Delta x}{\max(|u_1^n \pm c_1^n/M_1|, |u_2^n \pm c_2^n/M_2|)}.$$

## 5.1 Isentropic Euler equations

With the following test we validate the consistency of the RS-IMEX all-speed scheme (3.32) with the isentropic Euler equations. Note that the model (2.1) reduces to the isentropic Euler equations for single phase flow. We consider the low Mach number test case by Degond & Tang described in [9, 12]. We assign to each phase the ideal gas law with  $\gamma = 1.4, \kappa = 1$  and set  $\alpha = 0.5$ . The initial data are well-prepared and given by

$$\rho(x,0) = \begin{cases}
2 & \text{for } x < 0.2, x \in (0.3, 0.7], x > 0.8, \\
2 + M^2 & \text{for } x \in (0.2, 0.3], \\
2 - M^2 & \text{for } x \in (0.7, 0.8],
\end{cases}$$

$$\nu(x,0) = \begin{cases}
1 - M^2/2 & \text{for } x < 0.2, x > 0.8, \\
1, & \text{for } x \in (0.2, 0.3] \cup (0.7, 0.8], \\
1 + M^2/2 & \text{for } x \in (0.3, 0.7].
\end{cases}$$
(5.1)

The solution consists of several Riemann problems leading to shocks and contact discontinuities that are stronger the larger the Mach number is chosen. The computational domain is given by [0,1] and is discretized using  $\Delta x = 10^{-3}$ . We first consider the compressible regime with M = 0.99 with  $\nu_{ac} = 0.9$  leading to  $\Delta t = 1.6 \cdot 10^{-4}$ . The results are given in Figure 1a where we see that our RS-IMEX all-speed scheme captures all shock positions correctly.

Next, we consider a weakly compressible regime with  $M=10^{-2}$ . For the RS-IMEX scheme, we consider different time step sizes given by  $\nu_{ac}=0.9$  and  $\nu_{u}=0.05, 0.2$  which corresponds to  $\Delta t=3.6\cdot 10^{-6}$  (417 time steps),  $\Delta t=2.5\cdot 10^{-5}$  (60 time steps) and  $\Delta t=10^{-4}$  (15 time steps), respectively. The reference solution computed with the explicit Rusanov scheme thus  $\nu_{ac}=0.9$  and  $\Delta t=3.6\cdot 10^{-6}$  (417 time steps). The results are given in Figure 1b. We can clearly observe that the scheme is able to capture accurately all shocks and rarefactions for an acoustic time step, whereas, as expected, the acoustic waves are more diffused the larger the time step is chosen. Note, that the wave fan of the isentropic Euler equations consists only of acoustic waves which are smeared for large time steps in our RS-IMEX scheme.

#### 5.2 Accuracy

The following numerical test concerns the accuracy of the RS-IMEX scheme. We recall that by construction, the scheme is formally first order. However due to the operator splitting in several sub-systems, we want to validate the experimental order of convergence (EOC). We consider a

![](_page_24_Figure_0.jpeg)

Figure 1: Riemann Problem from Section [5.1:](#page-23-1) Numerical results for density ρ and velocity u for M = 0.99 (top panel) and M = 10−<sup>2</sup> (bottom panel) at final time T<sup>f</sup> = 0.075, 0.0015, respectively, and grid size ∆x = 10−<sup>3</sup> .

double rarefaction test case with smooth initial data given by

$$\rho_1(x,0) = 1, \quad \rho_2(x,0) = 1, \quad \alpha = 0.99$$

$$u_1(x,0) = \begin{cases}
-2 & \text{for } x < 0.4 \\
30 - 16x + 200x^2 & \text{for } x \in [0.4, 0.5) \\
-70 + 240x - 200x^2 & \text{for } x \in [0.5, 0.6) \\
2 & \text{for } x \ge 0.6
\end{cases}$$

$$u_2(x,0) = u_1(x,0).$$

We assign to phase one the ideal gas law with γ<sup>1</sup> = 1.4 and κ<sup>1</sup> = 1 and to phase two the stiffened gas equation with γ<sup>2</sup> = 2.8, p<sup>∞</sup> = 1, κ<sup>2</sup> = 2. The reference solution was computed with the Rusanov scheme on a fine grid with N = 2<sup>15</sup> grid cells on the domain [0, 1] up to a final time T<sup>f</sup> = 0.1M<sup>2</sup> using νac = 0.9. In Table [1](#page-25-0) the L 1 error and EOC with respect to that reference solution is displayed. The EOC is of first order for all considered Mach number regimes. In addition, the magnitude of the error in the density ρ as well as in αρ<sup>1</sup> is of order

of  $\max(M_1, M_2)^2$ . This confirms the asymptotic preserving property of the RS-IMEX scheme (3.32). The initial jump in the velocities  $u_1, u_2$  triggers a perturbation of the densities  $\rho_1, \rho_2$  of order  $\mathcal{O}(M_1^2), \mathcal{O}(M_2^2)$  respectively. Analogous results have been obtained for different initial values for  $\alpha$  which we do not present here.

| $M_1, M_2$         | N        | $\rho$       |      | $\alpha_1 \rho_1$ |      | $\rho u$    |      | $u_1 - u_2$  |      |
|--------------------|----------|--------------|------|-------------------|------|-------------|------|--------------|------|
| $10^{-1}, 10^{-1}$ | $2^{6}$  | 4.235e-03    |      | 3.650e-03         |      | 5.500e-02   |      | 1.008e-01    |      |
|                    | $2^7$    | 2.089e-03    | 1.02 | 1.844e-03         | 0.98 | 2.708e-02   | 1.02 | 5.832e-02    | 0.79 |
|                    | $2^8$    | 1.057e-03    | 0.98 | 9.541e-04         | 0.95 | 1.401 e-02  | 0.95 | 3.319e-02    | 0.81 |
|                    | $2^{9}$  | 5.476e-04    | 0.94 | 4.999e-04         | 0.93 | 7.389e-03   | 0.92 | 1.820 e-02   | 0.86 |
|                    | $2^{10}$ | 2.514e-04    | 1.12 | 2.308e-04         | 1.11 | 3.379 e-03  | 1.12 | 9.320 e-03   | 0.96 |
|                    | $2^{11}$ | 1.209 e-04   | 1.05 | 1.111e-04         | 1.06 | 1.647e-03   | 1.03 | 4.629 e-03   | 1.01 |
|                    | $2^{12}$ | 5.921 e-05   | 1.03 | 5.456 e - 05      | 1.02 | 8.156e-04   | 1.02 | 2.176e-03    | 1.09 |
| $10^{-3}, 10^{-3}$ | $2^6$    | 3.480 e - 05 |      | 3.502 e-05        |      | 4.182e-02   | _    | 1.324 e - 01 | _    |
|                    | $2^7$    | 1.672 e-05   | 1.06 | 1.673 e-05        | 1.06 | 1.983 e-02  | 1.07 | 7.081e-02    | 0.90 |
|                    | $2^8$    | 8.512e-06    | 0.97 | 8.523 e-06        | 0.97 | 1.007e-02   | 0.97 | 3.843 e-02   | 0.88 |
|                    | $2^9$    | 4.465 e - 06 | 0.93 | 4.468e-06         | 0.93 | 5.298 e-03  | 0.92 | 2.056e-02    | 0.90 |
|                    | $2^{10}$ | 2.000e-06    | 1.16 | 2.005e-06         | 1.15 | 2.367e-03   | 1.16 | 9.961 e-03   | 1.05 |
|                    | $2^{11}$ | 9.512 e-07   | 1.07 | 9.537e-07         | 1.07 | 1.128e-03   | 1.07 | 4.936e-03    | 1.01 |
|                    | $2^{12}$ | 4.607e-07    | 1.05 | 4.610 e - 07      | 1.05 | 5.513e-04   | 1.03 | 2.401e-03    | 1.03 |
| $10^{-1}, 10^{-2}$ | $2^6$    | 1.054 e - 03 | —    | 1.049 e-03        | —    | 3.167e-02   | —    | 7.785e-02    | —    |
|                    | $2^7$    | 5.261 e-04   | 1.00 | 5.243 e-04        | 1.00 | 1.581 e02 | 1.00 | 3.977e-02    | 0.97 |
|                    | $2^8$    | 2.611e-04    | 1.01 | 2.604 e-04        | 1.01 | 7.865e-03   | 1.01 | 2.053e-02    | 0.95 |
|                    | $2^{9}$  | 1.290 e-04   | 1.02 | 1.286 e - 04      | 1.02 | 3.891e-03   | 1.02 | 9.990 e-03   | 1.03 |
|                    | $2^{10}$ | 6.277 e - 05 | 1.04 | 6.266 e - 05      | 1.04 | 1.907e-03   | 1.05 | 4.961e-03    | 1.01 |
|                    | $2^{11}$ | 3.072 e-05   | 1.03 | 3.076 e-05        | 1.03 | 9.189 e-03  | 1.09 | 2.469 e-03   | 1.01 |
|                    | $2^{12}$ | 1.583e-05    | 0.96 | 1.589 e-05        | 0.95 | 4.314e-03   | 1.02 | 1.082e-03    | 1.19 |
| $10^{-2}, 10^{-3}$ | $2^6$    | 1.083e-04    |      | 1.084e-04         |      | 3.130e-02   | _    | 7.721e-02    | _    |
|                    | $2^7$    | 5.466 e - 05 | 0.99 | 5.463 e-05        | 0.99 | 1.564 e-02  | 1.00 | 3.905 e-02   | 0.98 |
|                    | $2^8$    | 2.713e-05    | 1.01 | 2.710e-05         | 1.01 | 7.785e-03   | 1.01 | 2.007e-02    | 0.96 |
|                    | $2^{9}$  | 1.341e-05    | 1.02 | 1.338e-05         | 1.02 | 3.864 e-03  | 1.01 | 9.749 e-03   | 1.04 |
|                    | $2^{10}$ | 6.662 e-06   | 1.01 | 6.650 e - 06      | 1.01 | 1.900 e-03  | 1.02 | 4.833e-03    | 1.01 |
|                    | $2^{11}$ | 3.435 e-06   | 0.96 | 3.431e-06         | 0.95 | 9.228e-04   | 1.04 | 2.404e-03    | 1.01 |
|                    | $2^{12}$ | 1.901e-06    | 0.85 | 1.899e-06         | 0.85 | 4.407e-04   | 1.07 | 1.051e-03    | 1.19 |

Table 1: Riemann Problem from Section 5.2:  $L^1$  error and convergence rates for the smooth rarefaction test case with  $\nu_{ac} = 9$  and final time  $T_f = 0.2 M_2$  on the computational domain [0, 1].

#### 5.3 Riemann Problem

The basis of the next series of numerical results is a Riemann problem consisting of a jump in the phase densities  $\rho_1, \rho_2$  analogously to the Riemann problem for the isentropic Euler equations (5.1). For all tests we choose the ideal gas law with  $\gamma_1 = 1.4, \kappa_1 = 1$  for phase one and the stiffened gas equation with  $\gamma = 2.8, \kappa_2 = 2, p_{\infty} = 1$  for phase two. The initial data for phase densities and velocities are given by

$$\rho_1(x,0) = \begin{cases}
1 + M_1^2 \rho_1^{(2)} & \text{for } x < 0.5 \\
1 & \text{for } x \ge 0.5
\end{cases}, \quad \rho_2(x,0) = \begin{cases}
1 + M_2^2 \rho_2^{(2)} & \text{for } x < 0.5 \\
1 & \text{for } x \ge 0.5
\end{cases}$$

$$\nu_1(x,0) = 0.25 = u_2(x,0). \tag{5.2}$$

The initial values for  $\rho_1^{(2)}$  and  $\rho_2^{(2)}$  will be specified below.

Same Mach number regime. In the first series of numerical tests we consider two phases in the same flow regime with  $M_1 = M = M_2$  and  $\rho_1^{(2)} = 1 = \rho_2^{(2)}$ . This leads to well-prepared initial data with  $p_1^{RS} = 1 = p_2^{RS}$ . We start with a homogenous mixture governed by the homogeneous model with constant volume fraction  $\alpha$ . Afterwards we consider a jump in  $\alpha$  and a smooth transition modelling a sharp and a diffusive interface. The computational domain is [0,1] with a uniform mesh size of  $\Delta x = 10^{-3}$ . The final time  $T_f = 0.2M$  is chosen such that all waves are still contained in the computational domain to avoid boundary effects. In Figure 2 the numerical results are displayed for  $M = 10^{-1}$  in the top panel and  $M = 10^{-3}$  in the bottom panel. The computations are done with an acoustic time step with  $\nu_{ac} = 0.9$  and a larger time step with  $\nu_{ac} = 18$  which corresponds to  $\Delta t = 2.9 \cdot 10^{-5}$ ,  $2.9 \cdot 10^{-4}$  for  $M = 10^{-1}$  and  $\Delta t = 2.9 \cdot 10^{-7}$ ,  $2.9 \cdot 10^{-6}$  for  $M = 10^{-3}$ , respectively. We can see that the acoustic waves are captured accurately by the RS-IMEX scheme with a time step oriented to the acoustic waves, whereas they are faded out for larger time steps. Note that since  $\alpha$  is constant the material wave at x = 0.5 is not visible.

To capture the material wave, we consider a jump in the volume fraction at x=0.5 given by  $\alpha_L=0.8$  and  $\alpha_R=0.2$  which is transported with u. In Figure 3 the numerical results for  $M=10^{-2}$  are presented. The simulation is done with an acoustic time step with  $\nu_{ac}=0.9$  and a material time step resulting in  $\nu_{ac}=180$  leading to  $\Delta t=2.9\cdot 10^{-6}$  and  $\Delta t=5.8\cdot 10^{-4}$  respectively. We want to stress that the material wave at x=0.5 is captured also for large time steps as sharp as the reference solution calculated with N=30000 cells although using a time step that is 200 times larger.

We repeat this test using an initial smooth tangent transition between  $\alpha_L$  and  $\alpha_R$ . The results are given in Figure 4. Analogously to the previous case, the RS-IMEX scheme (3.32) captures accurately the diffusive interface even for large time steps with  $\nu_{ac} = 180$ .

Different Mach number regimes. The next numerical test concerns the Riemann problem with initial data (5.2) for different flow regimes of the respective phase given by  $M_1 = 10^{-2}$ ,  $M_2 = 10^{-3}$ . We consider a well-prepared homogeneous mixture governed by the homogeneous model with constant  $\alpha = 0.9$  and  $\rho_1^{(2)} = 1$  and  $p_2^{(2)} = p_1^{(1)}$ . Since the acoustic waves of phase two are significantly faster than the ones of phase one, we consider a larger computational domain given by [0,50] and  $\Delta x = 10^{-2}$  and the final time  $T_f = 7.5 \cdot 10^{-2}$ .

The numerical results are displayed in Figure 5. Figure 5a presents a zoom on the acoustic waves of phase one in the domain [23, 27] whereas the acoustic waves of phase two have already reached the domains [3, 12] and [37, 49] as depicted in Figure 5b. We run the simulation with three different time steps. The first time step is focused on the resolution of all acoustic waves with  $\nu_{ac} = 0.9$  given by  $\Delta t = 2.1 \cdot 10^{-5}$ . With the second time step given by  $\nu_{ac} = 9$  leading to  $\Delta t = 2.1 \cdot 10^{-4}$ , the fast acoustic waves of phase two are smoothened but the acoustic waves of

![](_page_27_Figure_0.jpeg)

Figure 2: Riemann Problem from Section [5.3:](#page-26-1) Numerical results for mixture density ρ, mixture velocity u and relative velocity u<sup>1</sup> − u<sup>2</sup> for M = 10−<sup>1</sup> (top panel) and M = 10−<sup>3</sup> (bottom panel) at final time T<sup>f</sup> = 0.2M, respectively, and grid size ∆x = 10−<sup>3</sup> .

phase one are resolved well. The last time step given by νac = 90 leading to ∆t = 2.1 · 10−<sup>3</sup> is oriented to the material wave neglecting the resolution of all acoustic waves.

Influence of pressure relaxation. We consider an initial homogeneous mixture with M<sup>1</sup> = M = M<sup>2</sup> and α = 0.5 with pressure relaxation source term. The initial condition is given by the Riemann problem [\(5.2\)](#page-26-0). We consider different values for the relaxation time τ = ∞, M, M−<sup>1</sup> for two different Mach number regimes M = 10−<sup>1</sup> , 10−<sup>3</sup> . As the pressure relaxation acts on the volume fraction α, we expect a change in α depending on the relaxation time. The numerical results are presented in Figure [6.](#page-34-0) We can see that for τ = M, the difference in the pressures goes to zero, leading to a change in α of order M<sup>2</sup> .

Influence of friction. In the final numerical test we rerun the test case with a jump in α for M = 10−<sup>1</sup> , M = 10−<sup>3</sup> with the friction source term acting on the relative velocity u<sup>1</sup> − u2. We consider different values for the friction coefficent ζ. The numerical results are shown in Figure [7.](#page-35-0) We can see that for large friction coefficients ζ the relative velocity goes to zero, leading to similar phase velocities. For lower Mach number flows a larger friction coefficient is needed to obtain the same effect, as can be seen from the results for M = 10−<sup>3</sup> .

# 6 Conclusions

We have proposed a first order implicit-explicit numerical method for the simulation of one dimensional isentropic two-phase flow based on the Symmetric Hyperbolic Thermodynamically Compatible model [\[36\]](#page-33-2). The scheme is proved to be consistent with single phase flow, first order accurate and captures accurately material waves in different Mach number regimes. Moreover, the numerical scheme is asymptotic preserving as demonstrated in Theorem [4.1](#page-20-1) and [4.2.](#page-22-1) We

![](_page_28_Figure_0.jpeg)

Figure 3: Riemann Problem from Section [5.3](#page-26-1) with an initial jump in α: Numerical results for mixture density ρ, mixture velocity u and relative velocity u<sup>1</sup> − u<sup>2</sup> for M = 10−<sup>2</sup> at final time T<sup>f</sup> = 0.2M with grid size ∆x = 10−<sup>3</sup> .

have applied a reference solution approach, where stiff non-linear quantities, as pressure and enthalpy, are linearised around a reference state given by the leading order of well-prepared initial data. These data were obtained by an asymptotic analysis of the singular Mach number limits for which the model was reformulated in non-dimensional form. The resulting linear stiff parts were integrated implicitly, whereas the transport terms were treated explicitly leading to the CFL condition that is only restricted by material wave speeds. The final solution contains also explicit nonlinear corrections of pressure and enthalpy yielding an all-speed scheme that performs well also in compressible regimes. Due to the complexity of the model, the flux terms were split in three hyperbolic sub-systems. This motivates the need to study more complex operator splittings beyond the two subsystems implicit-explicit splittings. The question of high order accuracy in the time integration remains open.

![](_page_29_Figure_0.jpeg)

Figure 4: Riemann Problem from Section [5.3](#page-26-1) with smooth initial α: Numerical results for mixture density ρ, mixture velocity u and relative velocity u<sup>1</sup> − u<sup>2</sup> for M = 10−<sup>2</sup> at final time T<sup>f</sup> = 0.2M with grid size ∆x = 10−<sup>3</sup> .

A key element of the scheme lies in the reformulation of stiff subsystems in the elliptic form. Even though the system is extremely coupled, we succeeded to solve the homogenous part efficiently with direct or iterative linear solvers. Moreover the model contains stiff relaxation source terms describing the interaction of the phases via friction and pressure relaxation processes. In this work, standard methods were used to solve the stiff non-linear pressure relaxation source term implicitly. To improve the implicit treatment of the relaxation source terms, we plan combine the homogeneous all-speed scheme with techniques presented in [\[6\]](#page-31-13). Therein robust and efficient solvers for relaxation source terms arising in two-phase flows are discussed. Further, we aim to extend the scheme to two dimensional problems, as well as to address the full two-phase model given in [\[34\]](#page-33-1).

![](_page_30_Figure_0.jpeg)

(a) Zoom on acoustic waves associated to phase one.

![](_page_30_Figure_2.jpeg)

(b) Zoom on acoustic waves associated to phase two.

Figure 5: Riemann Problem from Section [5.3](#page-26-1) for two phases in different Mach number regimes: Numerical results for M<sup>1</sup> = 10−<sup>2</sup> and M<sup>2</sup> = 10−<sup>3</sup> at final time T<sup>f</sup> = 7.5 · 10−<sup>2</sup> and grid size ∆x = 10−<sup>2</sup> .

# Acknowledgements

A.T. and M.L. have been partially supported by the Gutenberg Research College, JGU Mainz. Further, M.L. is grateful for the support of the Mainz Institute of Multi-Scale Modelling. G.P. is a member of GNCS and acknowledges the support of PRIN2017 and Sapienza, Progetto di Atteneo [RM120172B41DBF3A].

# References

- [1] A. Ambroso, C. Chalons, and P.-A. Raviart. A Godunov-type method for the seven-equation model of compressible two-phase flow. Comput. & Fluids, 54:67–91, 2012.
- [2] N. Andrianov, R. Saurel, and G. Warnecke. A simple method for compressible multiphase mixtures and interfaces. Int. J. Numer. Methods Fluids, 41(2):109–131, 2003.
- [3] N. Andrianov and G. Warnecke. The Riemann problem for the Baer–Nunziato two-phase flow model. J. Comput. Phys., 195(2):434–464, 2004.
- [4] M. R. Baer and J. W. Nunziato. A two-phase mixture theory for the deflagration-todetonation transition (DDT) in reactive granular materials. Int. J. Multiph. Flow, 12(6):861– 889, 1986.
- [5] G. Bispen, M. Luk´aˇcov´a-Medvid'ov´a, and L. Yelash. Asymptotic preserving IMEX finite volume schemes for low Mach number Euler equations with gravitation. J. Comput. Phys., 335:222–248, 2017.
- [6] S. Chiocchetti and C. M¨uller. A solver for stiff finite-rate relaxation in Baer-Nunziato twophase flow models. Fluid Mech. Appl., 121:31–44, 2020.
- [7] F. Cordier, P. Degond, and A. Kumbaro. An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations. J. Comput. Phys., 231(17):5685–5704, 2012.
- [8] V. Daru, P. Le Qu´er´e, M.-C. Duluc, and O. Le Maitre. A numerical method for the simulation of low Mach number liquid–gas flows. J. Comput. Phys., 229(23):8844–8867, 2010.
- [9] P. Degond and M. Tang. All speed scheme for the low Mach number limit of the isentropic Euler equations. Commun. Comput. Phys., 10(1):1–31, 2011.
- [10] S. Dellacherie. Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number. J. Comput. Phys., 229(4):978–1016, 2010.
- [11] A. D. Demou, N. Scapin, M. Pelanti, and L. Brandt. A pressure-based diffuse interface method for low-Mach multiphase flows with mass transfer. J. Comput. Phys., 448:110730, 2022.
- [12] G. Dimarco, R. Loub`ere, V. Michel-Dansac, and M.-H. Vignal. Second-order implicit-explicit total variation diminishing schemes for the Euler system in the low Mach regime. J. Comput. Phys., 372:178–201, 2018.
- [13] M. Dumbser, I. Peshkov, E. Romenski, and O. Zanotti. High order ADER schemes for a unified first order hyperbolic formulation of continuum mechanics: Viscous heat-conducting fluids and elastic solids. J. Comput. Phys., 314:824–862, 2016.
- [14] M. Feistauer, V. Dolejˇs´ı, and V. Kuˇcera. On the discontinuous Galerkin method for the simulation of compressible flow with wide range of Mach numbers. Computing and Visualization in Science, 10(1):17–27, 2007.
- [15] M. Feistauer and V. Kuˇccera. A new technique for the numerical solution of the compressible Euler equations with arbitrary Mach numbers. In Hyperbolic Problems: Theory, Numerics, Applications, pages 523–531. Springer, 2008.

- [16] S. K. Godunov. An interesting class of quasilinear systems. Dokl. Akad. Nauk SSSR, 139(3):521–523, 1961.
- [17] S. K. Godunov and E. I. Romenskii. Elements of continuum mechanics and conservation laws. Kluwer Academic/Plenum Publishers, 2003.
- [18] S. K. Godunov and E. I. Romensky. Thermodynamics, conservation laws and symmetric forms of differential equations in mechanics of continuous media. 95:19–31, 1995.
- [19] H. Gouin and S. Gavrilyuk. Hamilton's principle and Rankine–Hugoniot conditions for general motions of mixtures. Meccanica, 34(1):39–47, 1999.
- [20] H. Guillard and C. Viozat. On the behaviour of upwind schemes in the low Mach number limit. Comput. & Fluids, 28(1):63–86, 1999.
- [21] S. Jin. Efficient asymptotic-preserving (AP) schemes for some multiscale kinetic equations. SIAM J. Sci. Comput., 21(2):441–454, 1999.
- [22] K. Kaiser, J. Sch¨utz, R. Sch¨obel, and S. Noelle. A new stable splitting for the isentropic Euler equations. J. Sci. Comput., 70(3):1390–1407, 2017.
- [23] A. K. Kapila, R. Menikoff, J. B. Bdzil, S. F. Son, and D. S. Stewart. Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations. Phys. Fluids, 13(10):3002–3024, 2001.
- [24] S. Klainerman and A. Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids. Commun. Pur. Appl. Math., 34(4):481–524, 1981.
- [25] R. Klein. Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: One-dimensional flow. J. Comput. Phys., 121(2):213 – 237, 1995.
- [26] J. J. Kreeft and B. Koren. A new formulation of Kapila's five-equation model for compressible two-fluid flow, and its numerical treatment. J. Comput. Phys., 229(18):6220–6242, 2010.
- [27] V. Kuˇcera, M. Luk´aˇcov´a-Medvid'ov´a, S. Noelle, and J. Sch¨utz. Asymptotic properties of a class of linearly implicit schemes for weakly compressible Euler equations. Numer. Math., 150(1):79–103, 2022.
- [28] S. M¨uller, M. Hantke, and P. Richter. Closure conditions for non-equilibrium multicomponent models. Contin. Mech. Thermodyn., 28(4):1157–1189, 2016.
- [29] M. Pelanti. Low Mach number preconditioning techniques for Roe-type and HLLC-type methods for a two-phase compressible flow model. Appl. Math. Comput., 310:112–133, 2017.
- [30] I. Peshkov, M. Dumbser, W. Boscheri, E. Romenski, S. Chiocchetti, and M. Ioriatti. Simulation of non-Newtonian viscoplastic flows with a unified first order hyperbolic model and a structure-preserving semi-implicit scheme. Comput. & Fluids, 224:104963, 2021.
- [31] I. Peshkov, M. Pavelka, E. Romenski, and M. Grmela. Continuum mechanics and thermodynamics in the Hamilton and the Godunov-type formulations. Contin. Mech. Thermodyn., 30(6):1343–1378, 2018.

- [32] B. Re and R. Abgrall. A pressure-based method for weakly compressible two-phase flows under a Baer-Nunziato type model with generic equations of state and pressure and velocity disequilibrium. arXiv preprint arXiv:2107.12408, 2021.
- [33] E. Romenski, A. A. Belozerov, and I. M. Peshkov. Conservative formulation for compressible multiphase flows. Quart. Appl. Math., 74(1):113–136, 2016.
- [34] E. Romenski, D. Drikakis, and E. Toro. Conservative models and numerical methods for compressible two-phase flow. J. Sci. Comput., 42(1):68, 2010.
- [35] E. Romenski, G. Reshetova, I. Peshkov, and M. Dumbser. Two-Phase Computational Model for Small-Amplitude Wave Propagation in a Saturated Porous Medium. In Continuum Mechanics, Applied Mathematics and Scientific Computing: Godunov's Legacy, pages 313–320. Springer International Publishing, Cham, 2020.
- [36] E. Romenski and E. F. Toro. Compressible two-phase flows: Two-pressure models and numerical methods. Comput. Fluid Dyn. J, 13:403–416, 2004.
- [37] E. I. Romensky. Hyperbolic systems of thermodynamically compatible conservation laws in continuum mechanics. Mathematical and Computer Modelling, 28(10):115–130, 1998.
- [38] R. Saurel and R. Abgrall. A multiphase Godunov method for compressible multifluid and multiphase flows. J. Comput. Phys., 150(2):425–467, 1999.
- [39] F. Thein, E. Romenski, and M. Dumbser. Exact and numerical solutions of the Riemann problem for a barotropic shtc model of compressible two-phase flows. private communication (to appear), 2022.
- [40] J. Zeifang, J. Sch¨utz, K. Kaiser, A. Beck, M. Luk´aˇcov´a-Medvid'ov´a, and S. Noelle. A novel full-Euler low Mach number IMEX splitting. Commun. Comput. Phys., 27:292–320, 2020.
- [41] Z. Zou, E. Audit, N. Grenier, and C. Tenaud. An accurate sharp interface method for two-phase compressible flows at low-Mach regime. Flow, Turbulence and Combustion, 105(4):1413–1444, 2020.

![](_page_34_Figure_0.jpeg)

Figure 6: Riemann Problem from Section [5.3](#page-26-1) with pressure relaxation: Numerical results for mixture density ρ, mixture velocity u, pressure difference p<sup>2</sup> − p<sup>1</sup> and volume fraction α for different values of τ computed with an acoustic time stepping (solid line) and a material time stepping (dashed line). <sup>35</sup>

![](_page_35_Figure_0.jpeg)

Figure 7: Riemann Problem from Section [5.3](#page-26-1) with friction and an initial jump in α: Numerical results for mixture density ρ, mixture velocity u and relative velocity u<sup>1</sup> − u<sup>2</sup> for different values of the friction coefficient ζ at final time T<sup>f</sup> = 0.2M, respectively, and grid size ∆x = 10−<sup>3</sup>